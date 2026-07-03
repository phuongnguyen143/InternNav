#!/usr/bin/env python3
"""
InternVLA-N1 inference script (from inference_only_demo.ipynb and inference_only_demo_v2.ipynb).

Run inference on a local scene folder containing RGB images and an instruction
file. Each run writes to a timestamped subfolder under SAVE_DIR and stitches
annotated frames into an mp4 video.

Prerequisites (not handled by this script):
  - Install the environment of InternNav
  - InternVLA-N1 checkpoint downloaded locally
  - DepthAnything checkpoint in checkpoints/
  - Scene folder prepared with instruction.txt and debug_raw_*.jpg images
    + example: debug_raw_0001.jpg | debug_raw_0002_look_down.jpg

Notes:
  - Without --depth-dir, a constant dummy depth map is used (see DUMMY_DEPTH_METERS).
  - With --depth-dir, depth files use the same basename as each RGB frame (e.g. debug_raw_0001.jpg).
  - If not use `attn_implementation="flash_attention_2"`, set `attn_implementation="sdpa"`.
  - If you meet the error about the `No module named LongCLIP (or diffusion policy)`, you should run the `git submodule update --init` in the root directory of InternNav.
  - If you meet the error about the `No module named LongCLIP (or diffusion policy)`, you should run the `git submodule update --init` in the root directory of InternNav. 
Usage:
  python inference_only.py
  python inference_only.py --project-root /path/to/InternNav --scene-dir /path/to/scene --depth-dir /path/to/depth --model-path /path/to/checkpoint


  python inference_only.py  --scene-dir /home/khang/Documents/VR/InternNav/assets/bkhn_data1/rgb --depth-dir /home/khang/Documents/VR/InternNav/assets/bkhn_data1/depth
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont
import yaml

# TODO:
# later fix to depth anything checkpoint path input
# add depth image option into inference


# =============================================================================
# CONFIGURATION 
# =============================================================================

# InternNav repo root (directory that contains the `internnav` package).
# The notebook uses `../../` relative to scripts/notebooks/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Path to the InternVLA-N1 checkpoint directory.
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "InternVLA-N1-DualVLN"

# Scene folder: must contain instruction.txt and RGB frames (debug_raw_*.jpg).
SCENE_DIR = PROJECT_ROOT / "assets" / "realworld_sample_data2"

# Base output directory. Each run creates a required timestamped subfolder:
# output_test/{scene_name}_{YYYYMMDD_HHMMSS}/
SAVE_DIR = Path(__file__).resolve().parent / f"output_test"

# Compute device.
DEVICE = "cuda:0"

# Model / preprocessing settings.
RESIZE_W = 384
RESIZE_H = 384
NUM_HISTORY = 8
PLAN_STEP_GAP = 4

# Default Realsense D455 intrinsics (4x4), used when --camera-intrinsic is omitted.
DEFAULT_CAMERA_INTRINSIC = np.array(
    [
        [386.5, 0.0, 328.9, 0.0],
        [0.0, 386.5, 244.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DEFAULT_CAMERA_IMAGE_WIDTH = 640
DEFAULT_CAMERA_IMAGE_HEIGHT = 480

# Instruction file name in the scene dir
INSTRUCTION_FILENAME = "instruction.txt"
RGB_GLOB_PATTERN = "debug_raw_*.jpg"

# When real depth is unavailable, set with constant
# Unit is meter
# The sample dataset has no depth; trajectories will be shorter than ~2 m.
DUMMY_DEPTH_METERS = 10.0

# Default identity camera pose (4x4) when odometry is not recorded.
DEFAULT_CAMERA_POSE = np.eye(4)

# Set False to skip saving annotated PNGs (stdout logging only).
SAVE_VISUALIZATIONS = True

# Stitch saved annotated frames into an mp4 after inference.
SAVE_VIDEO = True
VIDEO_FILENAME = "trajectory_video.mp4"
VIDEO_FPS = 2.0

# Warm up the model with a dummy forward pass before processing the scene.
WARMUP_MODEL = True


# ==========================================================
# IMPLEMENTATION
# =====================================================


def load_camera_intrinsic(yaml_path: Path) -> np.ndarray:
    """Load a 4x4 camera intrinsic matrix from a ROS-style camera_matrix YAML."""
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "camera_matrix" not in data or "data" not in data["camera_matrix"]:
        raise ValueError(
            f"Expected 'camera_matrix.data' in {yaml_path}; "
            "see camera_intrinsic_realworld_sample_data.yaml for format."
        )

    k = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = k
    return intrinsic


def resolve_camera_intrinsic(yaml_path: Path | None) -> np.ndarray:
    if yaml_path is None:
        print("Using default camera intrinsic")
        return DEFAULT_CAMERA_INTRINSIC.copy()
    path = yaml_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Camera intrinsic file not found: {path}")
    return load_camera_intrinsic(path)


def save_camera_intrinsic(
    yaml_path: Path,
    intrinsic: np.ndarray,
    image_width: int = DEFAULT_CAMERA_IMAGE_WIDTH,
    image_height: int = DEFAULT_CAMERA_IMAGE_HEIGHT,
) -> Path:
    """Write camera intrinsics to a ROS-style YAML (same format as the demo notebook)."""
    k = np.asarray(intrinsic, dtype=np.float64)
    if k.shape == (4, 4):
        k = k[:3, :3]
    elif k.shape != (3, 3):
        raise ValueError(f"Expected 3x3 or 4x4 intrinsic matrix, got shape {k.shape}")

    camera_data = {
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": k.reshape(-1).tolist(),
        },
        "image_width": int(image_width),
        "image_height": int(image_height),
    }

    out_path = Path(yaml_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(camera_data, f)
    return out_path


class Args:
    def __init__(
        self,
        device: str,
        model_path: str | Path,
        resize_w: int,
        resize_h: int,
        num_history: int,
        camera_intrinsic: np.ndarray,
        plan_step_gap: int,
    ):
        self.device = device
        self.model_path = str(model_path)
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.num_history = num_history
        self.camera_intrinsic = camera_intrinsic
        self.plan_step_gap = plan_step_gap


def setup_python_path(project_root: Path) -> None:
    project_root = project_root.resolve()
    if not project_root.exists():
        raise FileNotFoundError(
            f"PROJECT_ROOT does not exist: {project_root}\n"
            "Set PROJECT_ROOT to your InternNav repository root."
        )
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src" / "diffusion-policy"))


def load_instruction(scene_dir: Path, instruction_filename: str) -> str:
    instruction_path = scene_dir / instruction_filename
    if not instruction_path.exists():
        raise FileNotFoundError(
            f"Instruction file not found: {instruction_path}\n"
            f"Each scene folder needs an `{instruction_filename}` file."
        )
    return instruction_path.read_text(encoding="utf-8").strip()


def load_rgb_paths(scene_dir: Path, glob_pattern: str) -> list[str]:
    rgb_paths = sorted(glob.glob(str(scene_dir / glob_pattern)))
    if not rgb_paths:
        raise FileNotFoundError(
            f"No RGB images matched `{glob_pattern}` in {scene_dir}"
        )
    return rgb_paths


def make_run_save_dir(base_dir: Path, scene_dir: Path, timestamp: datetime | None = None) -> Path:
    """Create a unique run directory: {base}/{scene_name}_{YYYYMMDD_HHMMSS}."""
    timestamp = timestamp or datetime.now()
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{scene_dir.name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_image_id(rgb_path: str) -> str:
    basename = os.path.basename(rgb_path)
    look_down = "look_down" in basename
    if look_down:
        return basename.replace("debug_raw_", "").replace("_look_down.jpg", "")
    return basename.replace("debug_raw_", "").replace(".jpg", "")


def depth_path_for_rgb(rgb_path: str | Path, depth_dir: Path) -> Path | None:
    """Resolve depth file in depth_dir with the same basename as the RGB file."""
    rgb_name = Path(rgb_path).name
    depth_dir = depth_dir.resolve()
    candidates = [
        depth_dir / rgb_name,
        depth_dir / f"{Path(rgb_name).stem}.npy",
        depth_dir / f"{Path(rgb_name).stem}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _depth_array_to_meters(
    depth: np.ndarray,
    depth_scale: float | None,
) -> np.ndarray:
    depth = np.squeeze(depth).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D after squeeze, got shape {depth.shape}")

    if depth_scale is not None:
        depth = depth * float(depth_scale)
        return depth

    if depth.dtype == np.uint16 or np.max(depth) > 20.0:
        # Typical uint16 depth in millimeters.
        return depth / 1000.0

    if np.max(depth) <= 1.0:
        # Normalized depth maps in [0, 1] -> assume 0-10 m.
        return depth * DUMMY_DEPTH_METERS

    if np.max(depth) <= 255.0:
        # 8-bit visualization or grayscale -> map to 0-10 m.
        return depth * (DUMMY_DEPTH_METERS / 255.0)

    return depth


def load_depth_image_file(
    depth_path: Path,
    target_shape: tuple[int, int],
    depth_scale: float | None,
) -> np.ndarray:
    depth_path = depth_path.resolve()
    suffix = depth_path.suffix.lower()

    if suffix == ".npy":
        depth = np.load(depth_path)
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to read depth image: {depth_path}")
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = _depth_array_to_meters(depth, depth_scale)
    h, w = target_shape
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(depth, dtype=np.float32)


def load_depth_for_frame(
    rgb: np.ndarray,
    rgb_path: str | Path,
    dummy_depth_meters: float,
    depth_dir: Path | None = None,
    depth_scale: float | None = None,
) -> np.ndarray:
    """Load depth (meters, float32 HxW). Uses depth_dir file matching RGB basename if set."""
    h, w = rgb.shape[:2]
    if depth_dir is None:
        return dummy_depth_meters * np.ones((h, w), dtype=np.float32)

    depth_path = depth_path_for_rgb(rgb_path, depth_dir)
    if depth_path is None:
        raise FileNotFoundError(
            f"No depth file for {Path(rgb_path).name} in {depth_dir.resolve()}\n"
            "Depth files must use the same basename as RGB (e.g. debug_raw_0001.jpg)."
        )
    return load_depth_image_file(depth_path, (h, w), depth_scale)


def validate_depth_dir(rgb_paths: list[str], depth_dir: Path) -> None:
    depth_dir = depth_dir.resolve()
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    missing = []
    for rgb_path in rgb_paths:
        if depth_path_for_rgb(rgb_path, depth_dir) is None:
            missing.append(os.path.basename(rgb_path))

    if missing:
        sample = "\n  ".join(missing[:5])
        more = f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} RGB frame(s) have no matching depth file in {depth_dir}\n"
            f"  {sample}{more}"
        )


def create_video_from_frames(
    frames: list[np.ndarray],
    video_path: Path,
    fps: float,
) -> None:
    if not frames:
        print("No annotated frames; skipping video.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved video to: {video_path} ({len(frames)} frames @ {fps} fps)")


def annotate_image(
    idx,
    image,
    llm_output=None,
    trajectory=None,
    pixel_goal=None,
    output_dir="./",
    frame_index: int | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)

    font_size = 20
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    text_content = [
        f"Frame Id : {idx}",
        f"Output   : {llm_output}",
    ]

    max_width = 0
    total_height = 0
    line_height = 26

    for line in text_content:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        max_width = max(max_width, text_width)
        total_height += line_height

    padding = 10
    box_x, box_y = 10, 10
    box_width = max_width + 2 * padding
    box_height = total_height + 2 * padding

    draw.rectangle(
        [box_x, box_y, box_x + box_width, box_y + box_height],
        fill="black",
    )

    y_position = box_y + padding
    for line in text_content:
        draw.text((box_x + padding, y_position), line, fill="white", font=font)
        y_position += line_height

    image = np.array(image)

    if trajectory is not None and len(trajectory) > 0:
        img_height, img_width = image.shape[:2]
        window_size = 200
        window_x = max(0, img_width - window_size)
        window_y = 0

        traj_points = []
        for point in trajectory:
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                traj_points.append([float(point[0]), float(point[1])])

        if len(traj_points) > 0:
            traj_array = np.array(traj_points)
            x_coords = traj_array[:, 0]
            y_coords = traj_array[:, 1]

            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            fig.patch.set_alpha(0.6)
            fig.patch.set_facecolor("gray")
            ax.set_facecolor("lightgray")

            ax.plot(y_coords, x_coords, "b-", linewidth=2)
            ax.plot(y_coords[0], x_coords[0], "go", markersize=6)
            ax.plot(y_coords[-1], x_coords[-1], "ro", markersize=6)
            ax.plot(0, 0, "w+", markersize=10, markeredgewidth=2)

            ax.set_xlabel("Y", fontsize=8)
            ax.set_ylabel("X", fontsize=8)
            ax.invert_xaxis()
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_aspect("equal", adjustable="box")

            plt.tight_layout(pad=0.3)

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            plot_img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)

            plot_img = cv2.resize(plot_img, (window_size, window_size))
            image[
                window_y : window_y + window_size,
                window_x : window_x + window_size,
            ] = plot_img

    if pixel_goal is not None:
        pixel_goal = np.asarray(pixel_goal).astype(int)
        row, col = int(pixel_goal[0]), int(pixel_goal[1])
        h, w = image.shape[:2]
        if 0 <= row < h and 0 <= col < w:
            cv2.circle(image, (col, row), 6, (255, 0, 0), -1)

    if frame_index is not None:
        save_path = os.path.join(output_dir, f"frame_{frame_index:05d}_annotated.png")
    else:
        save_path = os.path.join(output_dir, f"rgb_{idx}_annotated.png")
    Image.fromarray(image).convert("RGB").save(save_path)
    return image


def build_agent(args: Args):
    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

    print("Loading model...")
    print(f"  model_path: {args.model_path}")
    print(f"  device:     {args.device}")
    print(f"  image size: {args.resize_w}x{args.resize_h}")
    print(f"  history:    {args.num_history}")
    print(f"  plan gap:   {args.plan_step_gap}")

    agent = InternVLAN1AsyncAgent(args)

    if WARMUP_MODEL:
        print("Warming up model...")
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_depth = np.zeros((480, 640), dtype=np.float32)
        dummy_pose = np.eye(4)
        agent.reset()
        agent.step(
            dummy_rgb,
            dummy_depth,
            dummy_pose,
            "hello",
            intrinsic=args.camera_intrinsic,
        )

    print("Model loaded successfully!")
    return agent


def run_inference(
    agent,
    args: Args,
    scene_dir: Path,
    save_dir: Path,
    instruction: str,
    rgb_paths: list[str],
    dummy_depth_meters: float,
    depth_dir: Path | None,
    depth_scale: float | None,
    save_visualizations: bool,
    save_video: bool,
    video_fps: float,
    video_filename: str,
) -> None:
    agent.reset()
    print("=" * 80)
    print(f"Processing scene: {scene_dir.name}")
    print(f"Instruction: '{instruction}'")
    print(f"Total images: {len(rgb_paths)}")
    print("=" * 80)
    print()

    if save_visualizations:
        save_dir.mkdir(parents=True, exist_ok=True)

    annotated_frames: list[np.ndarray] = []

    for frame_index, rgb_path in enumerate(rgb_paths):
        look_down = "look_down" in rgb_path
        image_id = parse_image_id(rgb_path)

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = load_depth_for_frame(
            rgb,
            rgb_path,
            dummy_depth_meters,
            depth_dir=depth_dir,
            depth_scale=depth_scale,
        )
        camera_pose = DEFAULT_CAMERA_POSE.copy()

        with torch.no_grad():
            dual_sys_output = agent.step(
                rgb,
                depth,
                camera_pose,
                instruction,
                intrinsic=args.camera_intrinsic,
                look_down=look_down,
            )

        # trajectory: from System 1 (NavDP) when S2 already produced a pixel goal + latent; not from the LLM text directly.
        # pixel_goal: from S2 when the LLM output contains digits (waypoint in the image).

        
        output_action = dual_sys_output.output_action
        output_traj = dual_sys_output.output_trajectory
        output_pixel = dual_sys_output.output_pixel

        if output_action is not None and output_action != []:
            print(f"Output action: {output_action}")
            llm_output = output_action
            trajectory = None
            pixel_goal = None
        else:
            if output_traj is not None:
                try:
                    trajectory = output_traj.tolist()
                except AttributeError:
                    trajectory = output_traj
                print(f"output_trajectory: {trajectory}")
            else:
                trajectory = None
                print("output_trajectory: None")

            if output_pixel is not None:
                print(f"output_pixel: {output_pixel}")

            llm_output = "traj"
            pixel_goal = output_pixel

        if save_visualizations:
            annotated = annotate_image(
                image_id,
                rgb,
                llm_output=llm_output,
                trajectory=trajectory,
                pixel_goal=pixel_goal,
                output_dir=str(save_dir),
                frame_index=frame_index,
            )
            annotated_frames.append(annotated)

    print(f"\nScene {scene_dir.name} completed!")
    if save_visualizations:
        print(f"Saved {len(annotated_frames)} annotated frames to: {save_dir}")
        if save_video:
            create_video_from_frames(
                annotated_frames,
                save_dir / video_filename,
                video_fps,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InternVLA-N1 inference on a local scene folder.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="InternNav repository root",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to InternVLA-N1 checkpoint directory",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=SCENE_DIR,
        help="Scene folder with instruction.txt and RGB images",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=SAVE_DIR,
        help="Base output directory; a timestamped subfolder is always created per run",
    )
    parser.add_argument("--device", default=DEVICE, help="Torch device, e.g. cuda:0")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving annotated visualization PNGs",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip dummy warmup forward pass",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip stitching annotated frames into an mp4",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=VIDEO_FPS,
        help="Output video frame rate",
    )
    parser.add_argument(
        "--video-name",
        default=VIDEO_FILENAME,
        help="Output video filename inside save-dir",
    )
    parser.add_argument(
        "--camera-intrinsic",
        type=Path,
        default="./camera_intrinsic_realworld_sample_data.yaml",
        help=(
            "Path to camera intrinsic YAML (camera_matrix.data, 3x3). "
            "If omitted, uses the built-in Realsense D455 default matrix."
        ),
    )
    parser.add_argument(
        "--save-camera-intrinsic",
        type=Path,
        default=None,
        metavar="YAML_PATH",
        help=(
            "Write the resolved camera intrinsic matrix to YAML and exit "
            "(does not run inference). Uses --camera-intrinsic if set, else default."
        ),
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help=(
            "Directory of depth images with the same filenames as RGB "
            "(e.g. scene/rgb/debug_raw_0001.jpg -> depth-dir/debug_raw_0001.jpg). "
            "If omitted, uses a constant dummy depth map."
        ),
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help=(
            "Multiply raw depth file values by this factor to get meters. "
            "If omitted, auto-detect (uint16 mm -> /1000, uint8 -> 0-10m, etc.)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    global WARMUP_MODEL

    cli = parse_args()
    WARMUP_MODEL = WARMUP_MODEL and not cli.no_warmup

    setup_python_path(cli.project_root)

    camera_intrinsic = resolve_camera_intrinsic(cli.camera_intrinsic)

    scene_dir = cli.scene_dir.resolve()
    model_path = cli.model_path.resolve()
    save_base_dir = cli.save_dir.resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Download InternVLA-N1 and set MODEL_PATH or --model-path."
        )
    if not scene_dir.is_dir():
        raise FileNotFoundError(
            f"Scene directory not found: {scene_dir}\n"
            "Set SCENE_DIR or --scene-dir to your prepared scene folder."
        )

    instruction = load_instruction(scene_dir, INSTRUCTION_FILENAME)
    rgb_paths = load_rgb_paths(scene_dir, RGB_GLOB_PATTERN)

    depth_dir = cli.depth_dir.resolve() if cli.depth_dir is not None else None
    if depth_dir is not None:
        validate_depth_dir(rgb_paths, depth_dir)

    save_dir = make_run_save_dir(save_base_dir, scene_dir)

    print(f"Scene directory: {scene_dir}")
    print(f"Output directory: {save_dir}")
    print(f"Instruction: {instruction}")
    print(f"Found {len(rgb_paths)} images")
    print("First 5 images:")
    for i, path in enumerate(rgb_paths[:5]):
        print(f"  {i + 1}. {os.path.basename(path)}")
    print()

    if cli.camera_intrinsic is not None:
        print(f"Camera intrinsic: {cli.camera_intrinsic.resolve()}")
    else:
        print("Camera intrinsic: default Realsense D455 matrix")
    if depth_dir is not None:
        print(f"Depth directory: {depth_dir}")
        if cli.depth_scale is not None:
            print(f"Depth scale: {cli.depth_scale} (raw values -> meters)")
    else:
        print(f"Depth: dummy constant ({DUMMY_DEPTH_METERS} m)")
    print()

    args = Args(
        device=cli.device,
        model_path=model_path,
        resize_w=RESIZE_W,
        resize_h=RESIZE_H,
        num_history=NUM_HISTORY,
        camera_intrinsic=camera_intrinsic,
        plan_step_gap=PLAN_STEP_GAP,
    )

    agent = build_agent(args)
    save_visualizations = SAVE_VISUALIZATIONS and not cli.no_save
    run_inference(
        agent=agent,
        args=args,
        scene_dir=scene_dir,
        save_dir=save_dir,
        instruction=instruction,
        rgb_paths=rgb_paths,
        dummy_depth_meters=DUMMY_DEPTH_METERS,
        depth_dir=depth_dir,
        depth_scale=cli.depth_scale,
        save_visualizations=save_visualizations,
        save_video=SAVE_VIDEO and not cli.no_video and save_visualizations,
        video_fps=cli.video_fps,
        video_filename=cli.video_name,
    )

if __name__ == "__main__":
    main()