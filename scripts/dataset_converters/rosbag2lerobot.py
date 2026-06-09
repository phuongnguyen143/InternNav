"""
Convert keyframe_output episodes to LeRobot v2.1 InternVLA-N1 System2 format (GdvgFV5R1Z5-style).

Output layout:
  {lerobot_out}/{scene_id}/
    meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl
    data/chunk-*/episode_*.parquet
    videos/chunk-*/observation.images.{rgb,depth}.{h}cm_{p}deg/episode_{ep}_{frame}.jpg|png
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import datasets
import numpy as np
import torch
from datasets import concatenate_datasets
from loguru import logger

_CONVERTERS = Path(__file__).resolve().parent
sys.path.insert(0, str(_CONVERTERS))
sys.path.insert(0, "/home/lenguyen1/hoangpqn/vln/InternNav/scripts/dataset_converters/lerobot/src")
_INSTR_GEN = Path(__file__).resolve().parents[1] / "instruction_generator"
sys.path.insert(0, str(_INSTR_GEN))

from floor_pose import (  # noqa: E402
    FLOOR_CALIBRATION_FILENAME,
    floor_2d_pose_to_action_matrix,
    load_floor_calibration,
)

from internvla_labels import (  # noqa: E402
    CAMERA_SETTINGS,
    DEFAULT_LOOKAHEAD_FRAMES,
    build_frame_labels,
    setting_key,
)

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    hf_transform_to_torch,
    validate_episode_buffer,
    write_episode,
    write_episode_stats,
    write_info,
)
from lerobot.datasets.video_utils import get_safe_default_codec

# ── CONFIG ────────────────────────────────────────────────────────────────────
KEYFRAME_ROOT = Path("./keyframe_output")
LEROBOT_OUT = Path("./lerobot_data")
SCENE_ID = "round2_bkhn"

DATASET_FPS = 30
TGT_W, TGT_H = 640, 480
# RealSense color intrinsics from bag; scaled from native 1280x720 to TGT_W x TGT_H.
NATIVE_CAMERA_INTRINSIC = np.array(
    [[647.04101562, 0.0, 637.3026123], [0.0, 646.40319824, 370.86227417], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
NATIVE_RGB_SIZE = (1280, 720)


def scale_camera_intrinsic(
    intrinsic: np.ndarray,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> np.ndarray:
    sw, sh = src_size
    dw, dh = dst_size
    sx, sy = dw / sw, dh / sh
    K = np.array(intrinsic, dtype=np.float64).reshape(3, 3).copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K.astype(np.float32)


DEFAULT_CAMERA_INTRINSIC = scale_camera_intrinsic(
    NATIVE_CAMERA_INTRINSIC, NATIVE_RGB_SIZE, (TGT_W, TGT_H)
)

# Image folders required by NavPixelGoalDataset (bkhn_125cm_0_30)
IMAGE_RGB_SETTINGS = [(125, 0), (125, 30)]
IMAGE_DEPTH_SETTINGS = [(125, 30)]


def get_internvla_features() -> Dict:
    """Parquet feature schema matching GdvgFV5R1Z5 / InternData-N1."""
    features: Dict = {
        "action": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["action_index"],
        },
    }
    for height_cm, pitch_deg in CAMERA_SETTINGS:
        sk = setting_key(height_cm, pitch_deg)
        features[f"pose.{sk}"] = {
            "dtype": "float32",
            "shape": (4, 4),
            "names": [f"pose.{sk}"],
        }
        features[f"goal.{sk}"] = {
            "dtype": "int32",
            "shape": (2,),
            "names": [f"goal.{sk}"],
        }
        features[f"relative_goal_frame_id.{sk}"] = {
            "dtype": "int32",
            "shape": (1,),
            "names": [f"relative_goal_frame_id.{sk}"],
        }
    return features


def image_rgb_key(height_cm: int, pitch_deg: int) -> str:
    return f"observation.images.rgb.{setting_key(height_cm, pitch_deg)}"


def image_depth_key(height_cm: int, pitch_deg: int) -> str:
    return f"observation.images.depth.{setting_key(height_cm, pitch_deg)}"


def load_poses_json(keyframe_root: Path) -> Dict[int, Dict]:
    poses_path = keyframe_root / "poses.json"
    if not poses_path.exists():
        raise FileNotFoundError(
            f"poses.json not found at {poses_path}. Run extract_keyframe first."
        )
    with open(poses_path) as f:
        data = json.load(f)
    return {int(p["frame_idx"]): p for p in data}


def parse_frame_idx_from_name(stem: str) -> int:
    return int(stem.split("_")[-1])


def get_episode_frame_range(episode_dir: Path) -> Tuple[int, int]:
    kf_paths = sorted(episode_dir.glob("kf_*.jpg")) or sorted(episode_dir.glob("kf_*.png"))
    if not kf_paths:
        return 0, -1
    idxs = [parse_frame_idx_from_name(p.stem) for p in kf_paths]
    return min(idxs), max(idxs)


def get_instruction(episode_dir: Path, episode_name: str) -> str:
    summary = episode_dir / "summary.txt"
    if summary.exists():
        text = summary.read_text().strip()
        if text:
            return text

    instr_json = episode_dir / "instructions.json"
    if instr_json.exists():
        try:
            data = json.loads(instr_json.read_text())
            if isinstance(data, dict):
                text = data.get("instruction", "").strip()
            elif isinstance(data, list) and data:
                text = " ".join(
                    d.get("instruction", "") for d in data if isinstance(d, dict)
                ).strip()
            else:
                text = ""
            if text:
                return text
        except Exception as e:
            print(f"  [WARN] Failed to parse instructions.json: {e}")

    instr_txt = episode_dir / "instructions.txt"
    if instr_txt.exists():
        lines = [line.strip() for line in instr_txt.read_text().splitlines() if line.strip()]
        if lines:
            return " ".join(lines)

    print(f"  [WARN] No instruction for {episode_name}, using fallback.")
    return f"Navigate through {episode_name}"


def load_camera_intrinsic(path: Optional[str]) -> np.ndarray:
    if path is None:
        return DEFAULT_CAMERA_INTRINSIC.copy()
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        return np.array(data, dtype=np.float32).reshape(3, 3)
    vals = [float(x) for x in path.replace(",", " ").split()]
    if len(vals) != 9:
        raise ValueError("camera_intrinsic must be 9 floats or a JSON file")
    return np.array(vals, dtype=np.float32).reshape(3, 3)


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    ep_index: int,
    ext: str,
    resize_wh: Optional[Tuple[int, int]] = None,
    is_depth: bool = False,
) -> int:
    """Extract video frames to episode_{ep:06d}_{frame}.{ext}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_wh:
            frame = cv2.resize(frame, resize_wh)
        fname = f"episode_{ep_index:06d}_{count}.{ext}"
        if is_depth:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(out_dir / fname), frame)
        else:
            cv2.imwrite(str(out_dir / fname), frame)
        count += 1
    cap.release()
    return count


def duplicate_frame_dir(src_dir: Path, dst_dir: Path) -> None:
    """Copy all frame files from src_dir to dst_dir."""
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_file in sorted(src_dir.glob("*")):
        if img_file.is_file():
            shutil.copy2(img_file, dst_dir / img_file.name)


def export_episode_images(
    rgb_video: Path,
    depth_video: Optional[Path],
    scene_root: Path,
    chunk: int,
    ep_index: int,
    resize_wh: Tuple[int, int],
) -> Dict[str, Path]:
    """
    Extract per-frame images and duplicate into all required setting folders.
    Returns files dict mapping observation key -> temp directory with frames.
    """
    tmp_base = scene_root / "_tmp" / f"episode_{ep_index:06d}"
    files: Dict[str, Path] = {}

    rgb_0_tmp = tmp_base / "rgb_125cm_0deg"
    n_rgb = extract_frames_from_video(
        rgb_video, rgb_0_tmp, ep_index, "jpg", resize_wh=resize_wh
    )
    if n_rgb == 0:
        raise RuntimeError(f"No RGB frames extracted from {rgb_video}")

    files[image_rgb_key(125, 0)] = rgb_0_tmp

    rgb_30_tmp = tmp_base / "rgb_125cm_30deg"
    duplicate_frame_dir(rgb_0_tmp, rgb_30_tmp)
    files[image_rgb_key(125, 30)] = rgb_30_tmp

    for height_cm, pitch_deg in [(125, 45), (60, 15), (60, 30)]:
        dup_tmp = tmp_base / f"rgb_{height_cm}cm_{pitch_deg}deg"
        duplicate_frame_dir(rgb_0_tmp, dup_tmp)
        files[image_rgb_key(height_cm, pitch_deg)] = dup_tmp

    if depth_video is not None and depth_video.exists():
        depth_30_tmp = tmp_base / "depth_125cm_30deg"
        n_depth = extract_frames_from_video(
            depth_video,
            depth_30_tmp,
            ep_index,
            "png",
            resize_wh=resize_wh,
            is_depth=True,
        )
        if n_depth != n_rgb:
            logger.warning(
                f"Depth frames ({n_depth}) != RGB frames ({n_rgb}) for episode {ep_index}"
            )
        files[image_depth_key(125, 30)] = depth_30_tmp

        for height_cm, pitch_deg in [(125, 0), (125, 45), (60, 15), (60, 30)]:
            dup_tmp = tmp_base / f"depth_{height_cm}cm_{pitch_deg}deg"
            duplicate_frame_dir(depth_30_tmp, dup_tmp)
            files[image_depth_key(height_cm, pitch_deg)] = dup_tmp

    return files


def compute_episode_stats(episode_data: dict, features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if key not in features:
            continue
        ft = features[key]
        if ft["dtype"] in ("string", "video", "image"):
            continue
        ep_ft_array = np.array(data)
        if ep_ft_array.ndim == 1:
            if key == "episode_index":
                ep_ft_array = ep_ft_array.reshape(-1, 1)
            else:
                shape = ft["shape"]
                ep_ft_array = ep_ft_array.reshape(
                    -1, int(np.prod(shape)) if len(shape) > 1 else 1
                )
        try:
            ep_stats[key] = get_feature_stats(ep_ft_array, axis=(0,), keepdims=True)
        except Exception as e:
            logger.warning(f"Stats failed for {key}: {e}")
    return ep_stats


class NavDatasetMetadata(LeRobotDatasetMetadata):
    def get_data_file_path(self, ep_index: int) -> Path:
        chunk = self.get_episode_chunk(ep_index)
        return Path("data") / f"chunk-{chunk:03d}" / f"episode_{ep_index:06d}.parquet"

    def get_video_file_path(self, ep_index: int, key: str) -> Path:
        chunk = self.get_episode_chunk(ep_index)
        return Path("videos") / f"chunk-{chunk:03d}" / key

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict,
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


class NavDataset(LeRobotDataset):
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
    ) -> "NavDataset":
        obj = cls.__new__(cls)
        obj.meta = NavDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=False,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.episode_buffer = obj.create_episode_buffer()
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend or get_safe_default_codec()
        return obj

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps

        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        for key, value in frame.items():
            if key not in self.features:
                raise ValueError(f"Key '{key}' not in features.")
            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, files: dict) -> None:
        if not self.episode_buffer:
            return

        episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(dict.fromkeys(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames,
            self.meta.total_frames + episode_length,
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(task) for task in tasks]
        )

        for key, ft in self.features.items():
            if key in ("index", "episode_index", "task_index"):
                continue
            stacked = np.stack(episode_buffer[key])
            if key == "action":
                episode_buffer[key] = stacked.reshape(-1, 1)
            elif key.startswith("relative_goal_frame_id"):
                episode_buffer[key] = stacked.reshape(-1, 1)
            else:
                episode_buffer[key] = stacked

        for key, source_path in files.items():
            if not key.startswith("observation.images."):
                continue
            dest_dir = self.root / self.meta.get_video_file_path(episode_index, key)
            dest_dir.mkdir(parents=True, exist_ok=True)
            source_dir = Path(source_path)
            if source_dir.exists():
                for img_file in source_dir.glob("*"):
                    if img_file.is_file():
                        shutil.copy2(img_file, dest_dir / img_file.name)

        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )
        self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(
            episode_dict, features=self.hf_features, split="train"
        )
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)


def write_internvla_info_json(
    root: Path,
    fps: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> None:
    """Finalize meta/info.json to match GdvgFV5R1Z5 schema."""
    features: Dict = {
        "action": {
            "dtype": "int32",
            "shape": [1],
            "names": ["action_index"],
        },
    }
    for height_cm, pitch_deg in CAMERA_SETTINGS:
        sk = setting_key(height_cm, pitch_deg)
        features[f"pose.{sk}"] = {
            "dtype": "float32",
            "shape": [4, 4],
            "names": [f"pose.{sk}"],
        }
        features[f"goal.{sk}"] = {
            "dtype": "int32",
            "shape": [2],
            "names": [f"goal.{sk}"],
        }
        features[f"relative_goal_frame_id.{sk}"] = {
            "dtype": "int32",
            "shape": [1],
            "names": [f"relative_goal_frame_id.{sk}"],
        }

    for col in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[col] = {"dtype": "float32" if col == "timestamp" else "int64", "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": None,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": 0,
        "total_chunks": max(1, (total_episodes - 1) // 1000 + 1) if total_episodes else 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    write_info(info, root)


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
            if last_pose is None:
                pose = {"x": 0.0, "y": 0.0, "yaw": 0.0, "z": 0.0, "pose_frame": "floor"}
            else:
                pose = last_pose
        else:
            last_pose = pose
        poses.append(pose)
    return poses


def convert_episode(
    episode_dir: Path,
    poses_by_frame_idx: Dict[int, Dict],
    lerobot_dataset: NavDataset,
    camera_intrinsic: np.ndarray,
    fps: int,
    floor_plane: Optional[tuple],
    pitch_horizon: int,
    pitch_lookdown: int,
    height_cm: int,
    goal_lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
) -> bool:
    lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer()

    src_rgb = episode_dir / "rgb.mp4"
    src_depth = episode_dir / "depth.mp4"
    if not src_rgb.exists():
        print(f"  [SKIP] No rgb.mp4 in {episode_dir.name}")
        return False

    instruction = get_instruction(episode_dir, episode_dir.name)
    ep_index = lerobot_dataset.meta.total_episodes
    start_frame, end_frame = get_episode_frame_range(episode_dir)
    resize = (TGT_W, TGT_H)

    tmp_dir = lerobot_dataset.root / "_tmp" / episode_dir.name

    try:
        n_video = get_video_frame_count(src_rgb)
        if end_frame >= start_frame:
            n_pose_span = end_frame - start_frame + 1
        else:
            n_pose_span = n_video
            start_frame, end_frame = 0, n_video - 1
            print(f"  [WARN] No keyframes in {episode_dir.name}, using frame range 0..{end_frame}")

        n_frames = min(n_video, n_pose_span)
        if n_video != n_pose_span:
            print(
                f"  [WARN] Video frames ({n_video}) != pose span ({n_pose_span}), "
                f"using {n_frames} frames"
            )

        if n_frames == 0:
            print("  [SKIP] No frames.")
            return False

        print("  Extracting frames …", end=" ", flush=True)
        chunk = lerobot_dataset.meta.get_episode_chunk(ep_index)
        image_files = export_episode_images(
            src_rgb,
            src_depth if src_depth.exists() else None,
            lerobot_dataset.root,
            chunk,
            ep_index,
            resize,
        )
        print("done")

        poses = collect_poses_for_episode(n_frames, start_frame, poses_by_frame_idx)
        frame_records = build_frame_labels(
            poses,
            floor_plane,
            floor_2d_pose_to_action_matrix,
            camera_intrinsic,
            TGT_W,
            TGT_H,
            goal_setting=(height_cm, pitch_lookdown),
            goal_lookahead_frames=goal_lookahead_frames,
        )

        print(f"  Adding {n_frames} frames …", end=" ", flush=True)
        for i, rec in enumerate(frame_records):
            frame = {k: v for k, v in rec.items()}
            lerobot_dataset.add_frame(
                frame=frame,
                task=instruction,
                timestamp=float(i) / fps,
            )
        print("done")

        lerobot_dataset.save_episode(files=image_files)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(lerobot_dataset.root / "_tmp" / f"episode_{ep_index:06d}", ignore_errors=True)

        ep_idx = lerobot_dataset.meta.total_episodes - 1
        print(
            f"  ✓ {episode_dir.name} → episode {ep_idx:06d} "
            f"({n_frames} frames @ {fps} fps, global frames {start_frame}-{start_frame + n_frames - 1})"
        )
        return True

    except Exception:
        lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(lerobot_dataset.root / "_tmp", ignore_errors=True)
        raise


def resolve_floor_plane(
    keyframe_root: Path,
    floor_calibration: Optional[Path],
) -> Optional[tuple]:
    cal_path = floor_calibration
    if cal_path is None:
        cal_path = keyframe_root / FLOOR_CALIBRATION_FILENAME
    if not cal_path.exists():
        return None
    cal = load_floor_calibration(cal_path)
    print(f"Loaded floor calibration from {cal_path}")
    return cal["floor_plane"]


def main(
    keyframe_root: Path,
    lerobot_out: Path,
    scene_id: str,
    fps: int,
    camera_intrinsic: np.ndarray,
    height_cm: int,
    pitch_horizon: int,
    pitch_lookdown: int,
    overwrite: bool,
    floor_calibration: Optional[Path] = None,
    goal_lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
) -> None:
    episodes_dir = keyframe_root / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes dir not found: {episodes_dir}")

    episode_dirs = sorted(
        x for x in episodes_dir.iterdir()
        if x.is_dir() and x.name.startswith("episode_")
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_ folders in {episodes_dir}")

    scene_root = lerobot_out / scene_id
    if scene_root.exists():
        if overwrite:
            shutil.rmtree(scene_root)
        else:
            raise FileExistsError(
                f"Output exists: {scene_root}. Pass --overwrite to replace."
            )

    print(f"Found {len(episode_dirs)} episodes")
    print(f"Scene ID          : {scene_id}")
    print(f"FPS               : {fps}")
    print(f"Resolution        : {TGT_W}×{TGT_H}")
    print(f"Camera setting    : {height_cm}cm horizon {pitch_horizon}° / lookdown {pitch_lookdown}°")
    print(f"Goal lookahead    : {goal_lookahead_frames} frames")
    print(f"Output            : {scene_root}")

    poses_by_frame_idx = load_poses_json(keyframe_root)
    print(f"Loaded {len(poses_by_frame_idx)} poses from poses.json")

    floor_plane = resolve_floor_plane(keyframe_root, floor_calibration)
    if floor_plane is not None:
        print("Using floor calibration for base poses and camera extrinsics")
    else:
        print("No floor_calibration.json — using floor x,y,yaw from poses.json")

    features = get_internvla_features()
    lerobot_dataset = NavDataset.create(
        repo_id=scene_id,
        root=scene_root,
        robot_type="unknown",
        fps=fps,
        features=features,
    )

    success = 0
    for ep_dir in episode_dirs:
        print(f"\n[{ep_dir.name}]")
        try:
            if convert_episode(
                ep_dir,
                poses_by_frame_idx,
                lerobot_dataset,
                camera_intrinsic,
                fps,
                floor_plane=floor_plane,
                pitch_horizon=pitch_horizon,
                pitch_lookdown=pitch_lookdown,
                height_cm=height_cm,
                goal_lookahead_frames=goal_lookahead_frames,
            ):
                success += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()

    meta = lerobot_dataset.meta
    write_internvla_info_json(
        scene_root,
        fps=fps,
        total_episodes=meta.total_episodes,
        total_frames=meta.total_frames,
        total_tasks=meta.info.get("total_tasks", meta.total_episodes),
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Converted {success}/{len(episode_dirs)} episodes")
    print(f"Output → {scene_root}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert keyframe episodes to InternVLA-N1 System2 LeRobot format (GdvgFV5R1Z5-style)"
    )
    parser.add_argument("--keyframe_root", type=str, default=str(KEYFRAME_ROOT))
    parser.add_argument("--lerobot_out", type=str, default=str(LEROBOT_OUT))
    parser.add_argument("--scene_id", type=str, default=SCENE_ID)
    parser.add_argument("--fps", type=int, default=DATASET_FPS)
    parser.add_argument(
        "--camera_intrinsic",
        type=str,
        default=None,
        help="9 comma-separated floats or path to JSON 3x3 matrix",
    )
    parser.add_argument("--height", type=int, default=125, help="Camera height in cm")
    parser.add_argument("--pitch_horizon", type=int, default=0, help="Horizon pitch in degrees")
    parser.add_argument("--pitch_lookdown", type=int, default=30, help="Look-down pitch in degrees")
    parser.add_argument(
        "--goal_lookahead",
        type=int,
        default=DEFAULT_LOOKAHEAD_FRAMES,
        help="Fixed frame lookahead for relative_goal_frame_id and pixel goals",
    )
    parser.add_argument(
        "--floor_calibration",
        type=str,
        default=None,
        help="Path to floor_calibration.json (default: <keyframe_root>/floor_calibration.json)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    floor_cal = Path(args.floor_calibration) if args.floor_calibration else None

    main(
        keyframe_root=Path(args.keyframe_root),
        lerobot_out=Path(args.lerobot_out),
        scene_id=args.scene_id,
        fps=args.fps,
        camera_intrinsic=load_camera_intrinsic(args.camera_intrinsic),
        height_cm=args.height,
        pitch_horizon=args.pitch_horizon,
        pitch_lookdown=args.pitch_lookdown,
        overwrite=args.overwrite,
        floor_calibration=floor_cal,
        goal_lookahead_frames=args.goal_lookahead,
    )
