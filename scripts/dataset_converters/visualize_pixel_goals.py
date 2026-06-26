#!/usr/bin/env python3
"""
Debug visualization for pixel-goal projection.

Overlays projected goals on RGB frames and saves:
  - per-frame annotated images
  - a contact-sheet grid
  - a floor-trajectory plot with current / lookahead markers

Example:
  conda activate internnav
  python scripts/dataset_converters/visualize_pixel_goals.py \\
    --keyframe_root scripts/instruction_generator/keyframe_output_round2_bkhn \\
    --lerobot_root scripts/dataset_converters/lerobot_data_1/round2_bkhn \\
    --episode_index 0 \\
    --frame_stride 30 \\
    --out_dir /tmp/goal_debug_ep0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPTS_ROOT))

from utils.floor_pose import (
    load_floor_calibration,
    resolve_floor_plane_for_keyframe_root,
)  # noqa: E402
from internvla_labels import (  # noqa: E402
    DEFAULT_LOOKAHEAD_FRAMES,
    INVALID_GOAL,
    build_optical_camera_extrinsic,
    compute_discrete_actions,
    compute_goals_for_setting,
    extract_floor_xyyaw,
    get_T_world_base_from_pose,
    project_world_point_with_camera,
)

GOAL_SETTING = (125, 30)

DEFAULT_INTRINSIC = np.array(
    [
        [323.52050781, 0.0, 318.65130615],
        [0.0, 430.93546549, 247.24151611],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
IMG_W, IMG_H = 640, 480
ACTION_NAMES = {-1: "START", 1: "FWD", 2: "LEFT", 3: "RIGHT"}


def load_poses_json(path: Path) -> Dict[int, Dict]:
    data = json.loads(path.read_text())
    return {int(p["frame_idx"]): p for p in data}


def get_episode_frame_range(episode_dir: Path) -> Tuple[int, int]:
    kf_paths = sorted(episode_dir.glob("kf_*.jpg")) or sorted(
        episode_dir.glob("kf_*.png")
    )
    if not kf_paths:
        return 0, -1
    idxs = [int(p.stem.split("_")[-1]) for p in kf_paths]
    return min(idxs), max(idxs)


def collect_poses_for_episode(
    n_frames: int,
    start_frame: int,
    poses_by_frame_idx: Dict[int, Dict],
) -> List[Dict]:
    poses: List[Dict] = []
    last_pose: Optional[Dict] = None
    for i in range(n_frames):
        global_idx = start_frame + i
        pose = poses_by_frame_idx.get(global_idx)
        if pose is None:
            pose = last_pose or {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "z": 0.0,
                "pose_frame": "floor",
            }
        else:
            last_pose = pose
        poses.append(pose)
    return poses


def resolve_image_path(
    lerobot_root: Optional[Path],
    episode_index: int,
    local_frame: int,
    pitch_deg: int = 30,
    height_cm: int = 125,
) -> Optional[Path]:
    if lerobot_root is None:
        return None
    chunk = episode_index // 1000
    rel = (
        f"videos/chunk-{chunk:03d}/"
        f"observation.images.rgb.{height_cm}cm_{pitch_deg}deg/"
        f"episode_{episode_index:06d}_{local_frame}.jpg"
    )
    path = lerobot_root / rel
    return path if path.is_file() else None


def read_video_frame(video_path: Path, local_frame: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_frame_image(
    episode_dir: Path,
    lerobot_root: Optional[Path],
    episode_index: int,
    local_frame: int,
) -> Optional[np.ndarray]:
    lerobot_img = resolve_image_path(lerobot_root, episode_index, local_frame)
    if lerobot_img is not None:
        bgr = cv2.imread(str(lerobot_img))
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    video_path = episode_dir / "rgb.mp4"
    if video_path.is_file():
        return read_video_frame(video_path, local_frame)
    return None


def draw_floor_path_on_image(
    rgb: np.ndarray,
    poses: List[Dict],
    local_i: int,
    floor_plane: tuple,
    camera_intrinsic: np.ndarray,
    path_horizon: int = 120,
    apply_body2optical: bool = True,
) -> np.ndarray:
    """Overlay projected on-floor trajectory segment (green) from current frame."""
    out = rgb.copy()
    h, w = out.shape[:2]
    if local_i >= len(poses) or "camera_matrix" not in poses[local_i]:
        return out

    cam_i = np.array(poses[local_i]["camera_matrix"], dtype=np.float64)
    pts: List[Tuple[int, int, int]] = []
    end = min(local_i + path_horizon, len(poses))
    for f in range(local_i, end):
        tgt = pose_floor_world_xyz(poses[f], floor_plane)
        goal, ok = project_world_point_with_camera(
            cam_i,
            tgt,
            camera_intrinsic,
            w,
            h,
            apply_body2optical=apply_body2optical,
            camera_frame=poses[local_i].get("camera_frame"),
        )
        if ok:
            pts.append((int(goal[0]), int(goal[1]), f))

    if len(pts) >= 2:
        for j in range(len(pts) - 1):
            cv2.line(out, pts[j][:2], pts[j + 1][:2], (60, 220, 60), 2, cv2.LINE_AA)
    for u, v, f in pts[::2]:
        cv2.circle(out, (u, v), 3, (40, 200, 40), -1)
        if f == local_i or f == local_i + 40 or f == local_i + 80:
            cv2.putText(
                out,
                str(f),
                (u + 4, v - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (40, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def draw_goal_on_image(
    rgb: np.ndarray,
    goal: np.ndarray,
    rel_id: int,
    action: int,
    frame_idx: int,
    global_frame_idx: int,
    target_global_frame: int,
    parquet_goal: Optional[Tuple[int, int]] = None,
    draw_path: bool = False,
    poses: Optional[List[Dict]] = None,
    floor_plane: Optional[tuple] = None,
    camera_intrinsic: Optional[np.ndarray] = None,
    apply_body2optical: bool = True,
) -> np.ndarray:
    out = rgb.copy()
    if (
        draw_path
        and poses is not None
        and floor_plane is not None
        and camera_intrinsic is not None
    ):
        out = draw_floor_path_on_image(
            out,
            poses,
            frame_idx,
            floor_plane,
            camera_intrinsic,
            apply_body2optical=apply_body2optical,
        )

    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    foot = (w // 2, h - 8)

    valid = not np.array_equal(goal, INVALID_GOAL)
    if valid:
        u, v = int(goal[0]), int(goal[1])
        cv2.arrowedLine(out, foot, (u, v), (255, 200, 0), 2, tipLength=0.12)
        cv2.circle(out, (u, v), 10, (255, 40, 40), 2)
        cv2.drawMarker(out, (u, v), (255, 40, 40), cv2.MARKER_CROSS, 18, 2)
        goal_text = f"proj goal ({u}, {v})"
    else:
        goal_text = "proj goal INVALID"

    if parquet_goal is not None:
        pu, pv = parquet_goal
        if pu >= 0 and pv >= 0:
            cv2.circle(out, (pu, pv), 8, (40, 255, 80), 2)
            cv2.drawMarker(out, (pu, pv), (40, 255, 80), cv2.MARKER_TILTED_CROSS, 14, 2)

    lines = [
        f"local frame {frame_idx} | global {global_frame_idx}",
        f"lookahead +{rel_id} -> frame {target_global_frame}",
        f"action {ACTION_NAMES.get(action, action)} ({action})",
        goal_text,
    ]
    if parquet_goal is not None:
        lines.append(f"parquet goal {parquet_goal}")

    y0 = 24
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1)
    cv2.circle(out, (cx, cy), 5, (0, 0, 0), 1)
    return out


def load_parquet_goals(
    lerobot_root: Optional[Path],
    episode_index: int,
    setting: str = "125cm_30deg",
) -> Optional[Dict[int, Tuple[int, int, int]]]:
    if lerobot_root is None:
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    chunk = episode_index // 1000
    pq_path = (
        lerobot_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    if not pq_path.is_file():
        return None

    df = pq.read_table(pq_path).to_pandas()
    goal_col = f"goal.{setting}"
    rel_col = f"relative_goal_frame_id.{setting}"
    if goal_col not in df.columns:
        return None

    out: Dict[int, Tuple[int, int, int]] = {}
    for i in range(len(df)):
        g = df[goal_col].iloc[i]
        rel = (
            int(df[rel_col].iloc[i][0])
            if hasattr(df[rel_col].iloc[i], "__len__")
            else int(df[rel_col].iloc[i])
        )
        out[i] = (int(g[0]), int(g[1]), rel)
    return out


def save_trajectory_plot(
    poses: List[Dict],
    sampled_local_frames: List[int],
    start_frame: int,
    lookahead: int,
    out_path: Path,
) -> None:
    xs = [p["x"] for p in poses]
    ys = [p["y"] for p in poses]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.plot(xs, ys, "-", color="#4C78A8", linewidth=1.5, label="floor trajectory")
    ax.scatter(xs[0], ys[0], c="green", s=60, zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], c="black", s=60, zorder=5, label="end")

    for local_i in sampled_local_frames:
        if local_i >= len(poses):
            continue
        w = min(local_i + lookahead, len(poses) - 1)
        ax.scatter(xs[local_i], ys[local_i], c="orange", s=35, zorder=6)
        ax.scatter(xs[w], ys[w], c="red", s=35, zorder=6)
        ax.annotate(
            "",
            xy=(xs[w], ys[w]),
            xytext=(xs[local_i], ys[local_i]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
        )
        ax.text(xs[local_i], ys[local_i], f" {local_i}", fontsize=7, color="orange")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("floor x (m)")
    ax.set_ylabel("floor y (m)")
    ax.set_title(f"Floor trajectory (global start frame {start_frame})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_contact_sheet(images: List[np.ndarray], out_path: Path, cols: int = 4) -> None:
    if not images:
        return
    cols = max(1, min(cols, len(images)))
    rows = int(math.ceil(len(images) / cols))
    h, w = images[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        resized = cv2.resize(img, (w, h))
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = resized
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pixel goals on navigation frames."
    )
    parser.add_argument(
        "--keyframe_root",
        type=str,
        required=True,
        help="Path to keyframe_output_* containing poses.json and episodes/",
    )
    parser.add_argument(
        "--lerobot_root",
        type=str,
        default=None,
        help="Optional converted LeRobot scene root for JPG frames / parquet comparison",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD_FRAMES)
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=30,
        help="Sample every N local frames for debug images",
    )
    parser.add_argument(
        "--max_frames", type=int, default=12, help="Max annotated frames to save"
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--draw_path",
        action="store_true",
        help="Overlay projected floor trajectory segment (green) on each frame",
    )
    parser.add_argument(
        "--camera_intrinsic",
        type=str,
        default=None,
        help="9 floats or JSON path; default matches rosbag2lerobot",
    )
    body2optical_group = parser.add_mutually_exclusive_group()
    body2optical_group.add_argument(
        "--body2optical",
        action="store_true",
        help="Apply ROS body→OpenCV optical on camera_matrix (LiDAR odom)",
    )
    body2optical_group.add_argument(
        "--no-body2optical",
        action="store_true",
        help="Skip body→optical transform (SLAM/DROID optical odom)",
    )
    args = parser.parse_args()

    apply_body2optical = True
    if args.no_body2optical:
        apply_body2optical = False
    elif args.body2optical:
        apply_body2optical = True

    keyframe_root = Path(args.keyframe_root)
    lerobot_root = Path(args.lerobot_root) if args.lerobot_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poses_by_idx = load_poses_json(keyframe_root / "poses.json")
    floor_plane = resolve_floor_plane_for_keyframe_root(
        keyframe_root, derive_if_missing=True
    )
    if floor_plane is None:
        raise FileNotFoundError(
            f"No floor plane in {keyframe_root}; add floor_calibration.json or floor_trajectory.txt"
        )

    if args.camera_intrinsic:
        p = Path(args.camera_intrinsic)
        if p.suffix == ".json":
            K = np.array(json.loads(p.read_text()), dtype=np.float64).reshape(3, 3)
        else:
            K = np.array(
                [float(x) for x in args.camera_intrinsic.replace(",", " ").split()],
                dtype=np.float64,
            ).reshape(3, 3)
    else:
        K = DEFAULT_INTRINSIC.copy()

    episode_dir = keyframe_root / "episodes" / f"episode_{args.episode_index:04d}"
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode dir not found: {episode_dir}")

    start_frame, end_frame = get_episode_frame_range(episode_dir)
    if end_frame < start_frame:
        video_path = episode_dir / "rgb.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"No keyframes or rgb.mp4 in {episode_dir}")
        cap = cv2.VideoCapture(str(video_path))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        start_frame, end_frame = 0, n_video - 1

    n_frames = end_frame - start_frame + 1
    poses = collect_poses_for_episode(n_frames, start_frame, poses_by_idx)
    xyyaw = extract_floor_xyyaw(poses, floor_plane, None)
    actions = compute_discrete_actions(xyyaw)
    goal_h, goal_p = GOAL_SETTING
    world_cam_optical = [
        build_optical_camera_extrinsic(
            pose,
            get_T_world_base_from_pose(pose, floor_plane, None),
            goal_h,
            goal_p,
            apply_body2optical=apply_body2optical,
            measured_setting=GOAL_SETTING,
        )
        for pose in poses
    ]
    goals, rel_ids = compute_goals_for_setting(
        poses,
        world_cam_optical,
        floor_plane,
        K,
        IMG_W,
        IMG_H,
        primary=True,
        lookahead_frames=args.lookahead,
    )
    parquet = load_parquet_goals(lerobot_root, args.episode_index)

    sampled = list(range(0, n_frames, max(1, args.frame_stride)))[: args.max_frames]
    annotated: List[np.ndarray] = []
    manifest = []

    for local_i in sampled:
        rgb = load_frame_image(episode_dir, lerobot_root, args.episode_index, local_i)
        if rgb is None:
            print(f"[WARN] missing image for local frame {local_i}")
            continue

        global_i = start_frame + local_i
        rel = int(rel_ids[local_i])
        target_global = start_frame + local_i + rel if rel > 0 else -1
        parquet_goal = None
        if parquet and local_i in parquet:
            pu, pv, _ = parquet[local_i]
            parquet_goal = (pu, pv)

        vis = draw_goal_on_image(
            rgb,
            goals[local_i],
            rel,
            int(actions[local_i]),
            local_i,
            global_i,
            target_global,
            parquet_goal=parquet_goal,
            draw_path=args.draw_path,
            poses=poses,
            floor_plane=floor_plane,
            camera_intrinsic=K,
            apply_body2optical=apply_body2optical,
        )
        out_path = out_dir / f"frame_{local_i:04d}_g{global_i}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        annotated.append(vis)

        entry = {
            "local_frame": local_i,
            "global_frame": global_i,
            "lookahead_global_frame": target_global,
            "relative_goal_frame_id": int(rel_ids[local_i]),
            "projected_goal": goals[local_i].tolist(),
            "action": int(actions[local_i]),
            "floor_xy_yaw_current": [
                poses[local_i]["x"],
                poses[local_i]["y"],
                poses[local_i]["yaw"],
            ],
            "floor_xy_yaw_target": (
                [
                    poses[local_i + rel]["x"],
                    poses[local_i + rel]["y"],
                    poses[local_i + rel]["yaw"],
                ]
                if rel > 0 and local_i + rel < len(poses)
                else None
            ),
            "world_xyz_target": (
                [
                    poses[local_i + rel]["world_x"],
                    poses[local_i + rel]["world_y"],
                    poses[local_i + rel]["world_z"],
                ]
                if rel > 0 and local_i + rel < len(poses)
                else None
            ),
        }
        if parquet_goal is not None:
            entry["parquet_goal"] = list(parquet_goal)
        manifest.append(entry)
        print(
            f"frame {local_i:4d} (global {global_i}) "
            f"rel={rel_ids[local_i]:3d} goal={goals[local_i].tolist()} "
            f"action={ACTION_NAMES.get(int(actions[local_i]), actions[local_i])}"
        )

    if annotated:
        make_contact_sheet(annotated, out_dir / "contact_sheet.jpg", cols=4)

    save_trajectory_plot(
        poses,
        sampled,
        start_frame,
        args.lookahead,
        out_dir / "floor_trajectory.png",
    )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    legend = out_dir / "README.txt"
    legend.write_text(
        "Pixel goal debug outputs\n"
        "========================\n"
        "frame_XXXX_gYYYY.jpg  : annotated RGB frame\n"
        "  - red cross       : projected goal (internvla_labels)\n"
        "  - green tilted X  : parquet goal from lerobot (if --lerobot_root set)\n"
        "  - yellow arrow    : bottom-center (robot foot) -> projected goal\n"
        "  - green line      : projected floor path (--draw_path)\n"
        "  - white dot       : image center reference (not the goal)\n"
        "contact_sheet.jpg   : grid of annotated frames\n"
        "floor_trajectory.png: top-down floor x,y with lookahead arrows\n"
        "manifest.json       : numeric debug values per saved frame\n"
    )
    print(f"\nSaved debug visuals to {out_dir}")


if __name__ == "__main__":
    main()
