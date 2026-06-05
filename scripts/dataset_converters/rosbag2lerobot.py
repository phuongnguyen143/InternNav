"""
Convert keyframe_output episodes to LeRobot v2.1 NavDP format (1LXtFkjw3qL-style).

Output layout:
  {lerobot_out}/{scene_id}/
    meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl
    data/chunk-*/episode_*.parquet
    videos/chunk-*/observation.images.rgb/episode_*.mp4
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import datasets
import numpy as np
import torch
from datasets import concatenate_datasets
from loguru import logger
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/lenguyen1/hoangpqn/vln/InternNav/scripts/dataset_converters/lerobot/src")
_INSTR_GEN = Path(__file__).resolve().parents[1] / "instruction_generator"
sys.path.insert(0, str(_INSTR_GEN))

from floor_pose import (  # noqa: E402
    FLOOR_CALIBRATION_FILENAME,
    floor_2d_pose_to_action_matrix,
    load_floor_calibration,
)

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    TASKS_PATH,
    append_jsonlines,
    check_timestamps_sync,
    get_episode_data_index,
    hf_transform_to_torch,
    validate_episode_buffer,
    write_info,
)
from lerobot.datasets.video_utils import get_safe_default_codec

# ── CONFIG ────────────────────────────────────────────────────────────────────
KEYFRAME_ROOT = Path("./keyframe_output")
LEROBOT_OUT = Path("./lerobot_data")
SCENE_ID = "1LXtFkjw3qL"

DATASET_FPS = 30
TGT_W, TGT_H = 640, 480
DEFAULT_CAMERA_INTRINSIC = np.array(
    [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
DEFAULT_CAMERA_EXTRINSIC = np.eye(4, dtype=np.float32)


def get_features(fps: int, include_depth_video: bool = True) -> Dict:
    """Parquet + optional video keys aligned with InternData-N1 / NavDP."""
    vid_info = {
        "video.fps": fps,
        "video.codec": "libx264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    depth_vid_info = {**vid_info, "video.is_depth_map": True}

    features = {
        "observation.camera_intrinsic": {
            "dtype": "float32",
            "shape": (3, 3),
        },
        "observation.camera_extrinsic": {
            "dtype": "float32",
            "shape": (4, 4),
        },
        "action": {
            "dtype": "float32",
            "shape": (4, 4),
        },
        "observation.images.rgb": {
            "dtype": "video",
            "shape": (TGT_H, TGT_W, 3),
            "names": ["height", "width", "channel"],
            "info": vid_info,
        },
    }
    if include_depth_video:
        features["observation.images.depth"] = {
            "dtype": "video",
            "shape": (TGT_H, TGT_W, 3),
            "names": ["height", "width", "channel"],
            "info": depth_vid_info,
        }
    return features


def pose_xyyaw_to_matrix(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    rot = Rotation.from_euler("z", yaw).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    T[2, 3] = float(z)
    return T


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


def reencode_video(
    src: Path,
    dst: Path,
    fps: int,
    resize_wh: Optional[Tuple[int, int]] = None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf_parts = []
    if resize_wh:
        vf_parts.append(f"scale={resize_wh[0]}:{resize_wh[1]}")
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += [
        "-r", str(fps),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning(
            f"Re-encode failed for {src}, falling back to copy.\n"
            f"{result.stderr.decode()[:300]}"
        )
        shutil.copyfile(src, dst)


def _pose_to_camera_extrinsic(
    pose: Dict,
    default_extrinsic: np.ndarray,
    camera_height: float,
    warned: list,
) -> np.ndarray:
    """Observation camera pose (T_world_cam) — never from floor x,y,yaw."""
    if "camera_matrix" in pose:
        return np.array(pose["camera_matrix"], dtype=np.float32).reshape(4, 4)

    if "camera_x" in pose:
        return pose_xyyaw_to_matrix(
            float(pose["camera_x"]),
            float(pose["camera_y"]),
            float(pose["camera_yaw"]),
            z=float(pose.get("camera_z", camera_height)),
        )

    if not warned[0]:
        logger.warning(
            "poses.json missing camera_matrix; using default camera_extrinsic."
        )
        warned[0] = True
    return default_extrinsic.copy()


def _pose_to_action_matrix(
    pose: Dict,
    floor_plane: Optional[tuple],
    camera_height: float,
    warned: list,
) -> np.ndarray:
    """NavDP action: floor embodiment only — never from camera_matrix."""
    if "action_matrix" in pose:
        return np.array(pose["action_matrix"], dtype=np.float32).reshape(4, 4)

    x = float(pose["x"])
    y = float(pose["y"])
    yaw = float(pose["yaw"])
    z = float(pose.get("z", 0.0))
    pose_frame = pose.get("pose_frame", "floor")

    if pose_frame == "floor" and floor_plane is not None:
        return floor_2d_pose_to_action_matrix(x, y, yaw, z, floor_plane)

    if not warned[0]:
        logger.warning(
            "poses.json missing action_matrix; building action from floor x,y,yaw "
            "(legacy fallback)."
        )
        warned[0] = True
    if pose_frame == "floor":
        z = 0.0
    else:
        z = camera_height if z == 0.0 and camera_height != 0.0 else z
    return pose_xyyaw_to_matrix(x, y, yaw, z=z)


def build_frame_records(
    n_frames: int,
    start_frame: int,
    poses_by_frame_idx: Dict[int, Dict],
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    camera_height: float,
    fps: int,
    floor_plane: Optional[tuple] = None,
) -> List[Dict]:
    intrinsic = camera_intrinsic.astype(np.float32)
    default_extrinsic = camera_extrinsic.astype(np.float32)
    records = []
    last_pose = None
    cam_warn = [False]
    act_warn = [False]

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

        cam_ext = _pose_to_camera_extrinsic(
            pose, default_extrinsic, camera_height, cam_warn
        )
        action = _pose_to_action_matrix(
            pose, floor_plane, camera_height, act_warn
        )

        records.append({
            "observation.camera_intrinsic": intrinsic.copy(),
            "observation.camera_extrinsic": cam_ext,
            "action": action,
            "timestamp": float(i) / fps,
        })

    return records


def compute_episode_stats(episode_data: dict, features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if key not in features:
            continue
        ft = features[key]
        if ft["dtype"] in ("string", "video"):
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
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "NavDatasetMetadata":
        obj = super().create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            robot_type=robot_type,
            root=root,
            use_videos=use_videos,
        )
        obj._next_task_index = 0
        obj._global_image_index = 0
        return obj

    def get_data_file_path(self, ep_index: int) -> Path:
        chunk = self.get_episode_chunk(ep_index)
        return Path("data") / f"chunk-{chunk:03d}" / f"episode_{ep_index:06d}.parquet"

    def get_video_file_path(self, ep_index: int, key: str) -> Path:
        chunk = self.get_episode_chunk(ep_index)
        return Path("videos") / f"chunk-{chunk:03d}" / key / f"episode_{ep_index:06d}.mp4"

    def register_task_dict(self, task: dict) -> int:
        task_index = self._next_task_index
        self._next_task_index += 1
        self.info["total_tasks"] = self._next_task_index
        append_jsonlines({"task_index": task_index, "task": task}, self.root / TASKS_PATH)
        return task_index

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        instruction: str,
        episode_stats: dict,
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        write_info(self.info, self.root)

        last_frame = max(0, episode_length - 1)
        sub_task = {
            "sub_instruction": instruction,
            "sub_indexes": [0, last_frame],
            "revised_sub_instruction": instruction,
        }
        sum_task = {
            "sum_instruction": instruction,
            "sum_indexes": [0, last_frame],
        }
        task_idx_sub = self.register_task_dict(sub_task)
        task_idx_sum = self.register_task_dict(sum_task)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": [
                {**sub_task},
                {**sum_task},
            ],
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        append_jsonlines(episode_dict, self.root / EPISODES_PATH)

        image_min = self._global_image_index
        image_max = self._global_image_index + max(0, episode_length - 1)
        stats_entry = {
            "episode_index": episode_index,
            "task_index": {
                "min": min(task_idx_sub, task_idx_sum),
                "max": max(task_idx_sub, task_idx_sum),
                "count": 2,
            },
            "image_index": {
                "min": image_min,
                "max": image_max,
                "count": episode_length,
            },
        }
        append_jsonlines(stats_entry, self.root / EPISODES_STATS_PATH)
        self._global_image_index += episode_length

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats


class NavDataset(LeRobotDataset):
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
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
            use_videos=use_videos,
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
        episode_index = episode_buffer["episode_index"]
        instruction = tasks[0] if tasks else ""

        episode_buffer["index"] = np.arange(
            self.meta.total_frames,
            self.meta.total_frames + episode_length,
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        if self.meta.get_task_index(instruction) is None:
            self.meta.add_task(instruction)
        episode_buffer["task_index"] = np.full(
            (episode_length,), self.meta.get_task_index(instruction)
        )

        for key, ft in self.features.items():
            if key in ("index", "episode_index", "task_index") or ft["dtype"] == "video":
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        for key, src in files.items():
            src = Path(src)
            if self.features.get(key, {}).get("dtype") != "video":
                continue
            if not src.exists():
                logger.warning(f"Video not found, skipping: {src}")
                continue
            dst_rel = self.meta.get_video_file_path(episode_index, key)
            dst = self.root / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            episode_buffer[key] = str(dst_rel)

        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, instruction, ep_stats)

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


def write_navdp_info_json(root: Path, fps: int, total_episodes: int, total_frames: int, total_tasks: int):
    """Finalize meta/info.json to match 1LXtFkjw3qL sample schema."""
    info = {
        "codebase_version": "v2.1",
        "robot_type": "unknown",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes,
        "total_chunks": max(1, (total_episodes - 1) // 1000 + 1) if total_episodes else 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.camera_intrinsic": {"dtype": "float32", "shape": [3, 3]},
            "observation.camera_extrinsic": {"dtype": "float32", "shape": [4, 4]},
            "action": {"dtype": "float32", "shape": [4, 4]},
        },
    }
    write_info(info, root)


def convert_episode(
    episode_dir: Path,
    poses_by_frame_idx: Dict[int, Dict],
    lerobot_dataset: NavDataset,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    camera_height: float,
    fps: int,
    floor_plane: Optional[tuple] = None,
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
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        tmp_rgb = tmp_dir / "rgb.mp4"
        print("  Re-encoding rgb.mp4 …", end=" ", flush=True)
        reencode_video(src_rgb, tmp_rgb, fps=fps, resize_wh=resize)
        print("done")

        has_depth = src_depth.exists()
        tmp_depth = None
        if has_depth:
            tmp_depth = tmp_dir / "depth.mp4"
            print("  Re-encoding depth.mp4 …", end=" ", flush=True)
            reencode_video(src_depth, tmp_depth, fps=fps, resize_wh=resize)
            print("done")

        n_video = get_video_frame_count(tmp_rgb)
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
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False

        frame_records = build_frame_records(
            n_frames,
            start_frame,
            poses_by_frame_idx,
            camera_intrinsic,
            camera_extrinsic,
            camera_height,
            fps,
            floor_plane=floor_plane,
        )

        print(f"  Adding {n_frames} frames …", end=" ", flush=True)
        for rec in frame_records:
            lerobot_dataset.add_frame(
                frame={
                    "observation.camera_intrinsic": rec["observation.camera_intrinsic"],
                    "observation.camera_extrinsic": rec["observation.camera_extrinsic"],
                    "action": rec["action"],
                },
                task=instruction,
                timestamp=rec["timestamp"],
            )
        print("done")

        files = {"observation.images.rgb": tmp_rgb}
        if has_depth and tmp_depth is not None:
            files["observation.images.depth"] = tmp_depth

        lerobot_dataset.save_episode(files=files)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        ep_idx = lerobot_dataset.meta.total_episodes - 1
        print(
            f"  ✓ {episode_dir.name} → episode {ep_idx:06d} "
            f"({n_frames} frames @ {fps} fps, global frames {start_frame}-{start_frame + n_frames - 1})"
        )
        return True

    except Exception:
        lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer()
        shutil.rmtree(tmp_dir, ignore_errors=True)
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
    camera_extrinsic: np.ndarray,
    camera_height: float,
    overwrite: bool,
    floor_calibration: Optional[Path] = None,
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
    print(f"Output            : {scene_root}")

    poses_by_frame_idx = load_poses_json(keyframe_root)
    print(f"Loaded {len(poses_by_frame_idx)} poses from poses.json")

    floor_plane = resolve_floor_plane(keyframe_root, floor_calibration)
    if floor_plane is not None:
        print(
            "Pose/action split: camera_extrinsic <- camera_matrix, "
            "action <- action_matrix (floor embodiment)"
        )
    else:
        print(
            "No floor_calibration.json — action fallback from floor x,y,yaw; "
            "set camera_odom_file in extract_keyframe for camera_extrinsic"
        )

    features = get_features(fps, include_depth_video=True)
    lerobot_dataset = NavDataset.create(
        repo_id=scene_id,
        root=scene_root,
        robot_type="unknown",
        fps=fps,
        use_videos=True,
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
                camera_extrinsic,
                camera_height,
                fps,
                floor_plane=floor_plane,
            ):
                success += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()

    meta = lerobot_dataset.meta
    write_navdp_info_json(
        scene_root,
        fps=fps,
        total_episodes=meta.total_episodes,
        total_frames=meta.total_frames,
        total_tasks=getattr(meta, "_next_task_index", meta.info.get("total_tasks", 0)),
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Converted {success}/{len(episode_dirs)} episodes")
    print(f"Output → {scene_root}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert keyframe episodes to LeRobot NavDP format (1LXtFkjw3qL-style)"
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
    parser.add_argument("--camera_height", type=float, default=0.0)
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
        camera_extrinsic=DEFAULT_CAMERA_EXTRINSIC.copy(),
        camera_height=args.camera_height,
        overwrite=args.overwrite,
        floor_calibration=floor_cal,
    )
