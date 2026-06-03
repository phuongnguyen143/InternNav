import json
import subprocess
import numpy as np
import torch
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import datasets

from PIL import Image
from datasets import concatenate_datasets
import sys
from loguru import logger

sys.path.insert(0, "/home/lenguyen1/hoangpqn/vln/InternNav/scripts/dataset_converters/lerobot/src")

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    check_timestamps_sync,
    embed_images,
    get_episode_data_index,
    hf_transform_to_torch,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
)
from lerobot.datasets.video_utils import get_safe_default_codec

# ── CONFIG ────────────────────────────────────────────────────────────────────
KEYFRAME_ROOT = Path("./keyframe_output")
LEROBOT_OUT   = Path("./lerobot_data")
REPO_NAME     = "nav_keyframes"

VIDEO_FPS         = 30    # native FPS of your episode videos
IMAGE_EXTRACT_FPS = 5     # FPS at which frames are saved into observation.images.*
                          # e.g. 5 means 1 frame every 0.2 s  (change freely)
DATASET_FPS       = IMAGE_EXTRACT_FPS   # LeRobot dataset FPS = extracted image FPS

# Target resolution for all stored frames / videos
TGT_W, TGT_H = 640, 480


def get_features() -> Dict:
    """
    Four observation keys, all saved under videos/chunk-XXX/:
      observation.images.rgb    → folder of PNGs  (extracted at IMAGE_EXTRACT_FPS)
      observation.images.depth  → folder of PNGs  (extracted at IMAGE_EXTRACT_FPS)
      observation.video.rgb     → episode_NNNNNN.mp4  (full video)
      observation.video.depth   → episode_NNNNNN.mp4  (full video)
    """
    def _img_info(is_depth: bool) -> dict:
        return {
            "video.fps": IMAGE_EXTRACT_FPS,
            "video.codec": "png",           # images stored as PNG files
            "video.pix_fmt": "rgb24",
            "video.is_depth_map": is_depth,
            "has_audio": False,
        }

    def _vid_info(is_depth: bool) -> dict:
        return {
            "video.fps": VIDEO_FPS,
            "video.codec": "libx264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": is_depth,
            "has_audio": False,
        }

    return {
        "observation.images.rgb": {
            "dtype": "video",          # LeRobot treats image folders as "video"
            "shape": (TGT_H, TGT_W, 3),
            "names": ["height", "width", "channel"],
            "info": _img_info(False),
        },
        "observation.images.depth": {
            "dtype": "video",
            "shape": (TGT_H, TGT_W, 3),
            "names": ["height", "width", "channel"],
            "info": _img_info(True),
        },
        "observation.video.rgb": {
            "dtype": "video",
            "shape": (TGT_H, TGT_W, 3),
            "names": ["height", "width", "channel"],
            "info": _vid_info(False),
        },
        "observation.video.depth": {
            "dtype": "video",
            "shape": (TGT_H, TGT_W, 3),
            "names": ["height", "width", "channel"],
            "info": _vid_info(True),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "yaw"],
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["dx", "dy", "dyaw"],
        },
    }


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_keyframes_json(keyframe_root: Path) -> Dict[int, Dict]:
    json_path = keyframe_root / "keyframes.json"
    if not json_path.exists():
        print(f"[WARN] keyframes.json not found at {json_path}, poses will be zero.")
        return {}
    with open(json_path) as f:
        data = json.load(f)
    return {entry["frame_idx"]: entry for entry in data}


def parse_frame_idx_from_name(stem: str) -> int:
    return int(stem.split("_")[-1])


def normalize_angle(rad: float) -> float:
    return (rad + np.pi) % (2 * np.pi) - np.pi


def delta_yaw(a: float, b: float) -> float:
    return normalize_angle(b - a)


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
        lines = [l.strip() for l in instr_txt.read_text().splitlines() if l.strip()]
        if lines:
            return " ".join(lines)

    print(f"  [WARN] No instruction for {episode_name}, using fallback.")
    return f"Navigate through {episode_name}"


# ── VIDEO UTILITIES ───────────────────────────────────────────────────────────
def reencode_video(
    src: Path,
    dst: Path,
    resize_wh: Optional[Tuple[int, int]] = None,
) -> None:
    """Re-encode src → dst as libx264/yuv420p, optionally resizing."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf_parts = []
    if resize_wh:
        vf_parts.append(f"scale={resize_wh[0]}:{resize_wh[1]}")

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast", str(dst)]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning(f"Re-encode failed for {src}, falling back to copy.\n"
                       f"{result.stderr.decode()[:300]}")
        shutil.copyfile(src, dst)


def extract_frames_at_fps(
    video_path: Path,
    out_dir: Path,
    extract_fps: int,
    ep_index: int,
    resize_wh: Optional[Tuple[int, int]] = None,
) -> List[Path]:
    """
    Extract frames from video_path at extract_fps into out_dir.
    Files are named: episode_NNNNNN_FFFFFF.png  (episode index + frame index).
    Returns sorted list of written PNG paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    vf_parts = [f"fps={extract_fps}"]
    if resize_wh:
        vf_parts.append(f"scale={resize_wh[0]}:{resize_wh[1]}")

    # Use a temp pattern then rename to the correct naming convention
    tmp_pattern = str(out_dir / "tmp_%06d.png")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", ",".join(vf_parts),
        "-q:v", "1",           # highest PNG quality
        tmp_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed for {video_path}:\n"
            f"{result.stderr.decode()}"
        )

    # Rename tmp_NNNNNN.png → episode_EEEEEE_FFFFFF.png
    tmp_files = sorted(out_dir.glob("tmp_*.png"))
    final_paths = []
    for frame_idx, tmp_file in enumerate(tmp_files):
        final_name = f"episode_{ep_index:06d}_{frame_idx:06d}.png"
        final_path = out_dir / final_name
        tmp_file.rename(final_path)
        final_paths.append(final_path)

    return final_paths


# ── STATS ─────────────────────────────────────────────────────────────────────
def compute_episode_stats(episode_data: dict, features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if key not in features:
            continue
        ft = features[key]
        if ft["dtype"] in ("string", "video"):
            # video keys are stored as path strings — skip array stats
            continue
        # scalar features
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


# ── METADATA SUBCLASS ─────────────────────────────────────────────────────────
class NavDatasetMetadata(LeRobotDatasetMetadata):
    def get_data_file_path(self, ep_index: int) -> Path:
        chunk = self.get_episode_chunk(ep_index)
        return Path("data") / f"chunk-{chunk:03d}" / f"episode_{ep_index:06d}.parquet"

    def get_video_file_path(self, ep_index: int, key: str) -> Path:
        """
        Full-video keys   (observation.video.*)   → videos/chunk-000/<key>/episode_NNNNNN.mp4
        Image-folder keys (observation.images.*)  → videos/chunk-000/<key>/   (a directory)
        """
        chunk = self.get_episode_chunk(ep_index)
        if key.startswith("observation.video."):
            return Path("videos") / f"chunk-{chunk:03d}" / key / f"episode_{ep_index:06d}.mp4"
        else:
            # image folder — return the directory path (no filename)
            return Path("videos") / f"chunk-{chunk:03d}" / key

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list,
        episode_stats: dict,
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"]   += episode_length
        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if self.video_keys:
            self.update_video_info()
        write_info(self.info, self.root)
        episode_dict = {
            "episode_index": episode_index,
            "tasks":  episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)
        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


# ── DATASET SUBCLASS ──────────────────────────────────────────────────────────
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
        obj.repo_id            = obj.meta.repo_id
        obj.root               = obj.meta.root
        obj.revision           = None
        obj.tolerance_s        = tolerance_s
        obj.image_writer       = None
        obj.episode_buffer     = obj.create_episode_buffer()
        obj.episodes           = None
        obj.hf_dataset         = obj.create_hf_dataset()
        obj.image_transforms   = None
        obj.delta_timestamps   = None
        obj.delta_indices      = None
        obj.episode_data_index = None
        obj.video_backend      = video_backend or get_safe_default_codec()
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
        """
        files = {
            # Full MP4 videos — copied as-is to videos/chunk/observation.video.*/episode_N.mp4
            "observation.video.rgb":   Path("…/rgb.mp4"),
            "observation.video.depth": Path("…/depth.mp4"),

            # Image folders — contents copied to videos/chunk/observation.images.*/
            "observation.images.rgb":   Path("…/rgb_frames/"),
            "observation.images.depth": Path("…/depth_frames/"),
        }
        """
        if not self.episode_buffer:
            return

        episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks          = episode_buffer.pop("task")
        episode_tasks  = list(set(tasks))
        episode_index  = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames,
            self.meta.total_frames + episode_length,
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(t) for t in tasks])

        # Stack scalar features
        for key, ft in self.features.items():
            if key in ("index", "episode_index", "task_index") or ft["dtype"] == "video":
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        # ── Handle all four video/image-folder keys ───────────────────────
        for key, src in files.items():
            src = Path(src)
            ft  = self.features.get(key, {})
            if ft.get("dtype") != "video":
                continue

            dst_rel = self.meta.get_video_file_path(episode_index, key)
            dst     = self.root / dst_rel

            if key.startswith("observation.video."):
                # ── Full video: copy MP4 file ─────────────────────────────
                if not src.exists():
                    logger.warning(f"Video not found, skipping: {src}")
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                # store relative path string for parquet
                episode_buffer[key] = str(dst_rel)

            elif key.startswith("observation.images."):
                # ── Image folder: copy all PNGs into dst directory ────────
                if not src.is_dir():
                    logger.warning(f"Image dir not found, skipping: {src}")
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                for png in sorted(src.glob("*.png")):
                    shutil.copy2(png, dst / png.name)
                # store the relative directory path for parquet
                episode_buffer[key] = str(dst_rel)

        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index    = get_episode_data_index(self.meta.episodes, [episode_index])
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
        ep_dataset   = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)


# ── POSE LOADING ──────────────────────────────────────────────────────────────
def load_keyframe_poses(
    episode_dir: Path,
    pose_by_frame_idx: Dict[int, Dict],
    instruction: str,
    n_frames: int,
    extract_fps: int = IMAGE_EXTRACT_FPS,
    video_fps: int   = VIDEO_FPS,
) -> List[Dict]:
    """
    Build one metadata dict per extracted frame.
    Maps each extracted frame → nearest keyframe in keyframes.json for pose.
    """
    kf_paths = sorted(episode_dir.glob("kf_*.jpg")) or sorted(episode_dir.glob("kf_*.png"))
    kf_idxs  = [parse_frame_idx_from_name(p.stem) for p in kf_paths]

    frames = []
    for i in range(n_frames):
        # extracted frame i corresponds to video time i / extract_fps
        video_frame_idx = int(i * video_fps / extract_fps)
        nearest_kf      = min(kf_idxs, key=lambda k: abs(k - video_frame_idx)) if kf_idxs else 0
        meta            = pose_by_frame_idx.get(nearest_kf, {})
        state = np.array(
            [meta.get("x", 0.0), meta.get("y", 0.0), meta.get("yaw", 0.0)],
            dtype=np.float32,
        )
        frames.append({
            "state":       state,
            "timestamp":   float(i) / extract_fps,
            "instruction": instruction,
        })

    # Action = delta pose to next frame
    for i in range(len(frames)):
        if i < len(frames) - 1:
            c = frames[i]["state"]
            n = frames[i + 1]["state"]
            action = np.array(
                [n[0] - c[0], n[1] - c[1], delta_yaw(c[2], n[2])],
                dtype=np.float32,
            )
        else:
            action = np.zeros(3, dtype=np.float32)
        frames[i]["action"] = action

    return frames


# ── EPISODE CONVERSION ────────────────────────────────────────────────────────
def convert_episode(
    episode_dir: Path,
    pose_by_frame_idx: Dict[int, Dict],
    lerobot_dataset: NavDataset,
) -> bool:
    """
    episode_dir/rgb.mp4   ──► videos/chunk-000/observation.images.rgb/   (PNGs @ IMAGE_EXTRACT_FPS)
                          ──► videos/chunk-000/observation.video.rgb/episode_N.mp4
    episode_dir/depth.mp4 ──► videos/chunk-000/observation.images.depth/ (PNGs @ IMAGE_EXTRACT_FPS)
                          ──► videos/chunk-000/observation.video.depth/episode_N.mp4
    """
    lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer()

    src_rgb   = episode_dir / "rgb.mp4"
    src_depth = episode_dir / "depth.mp4"

    if not src_rgb.exists():
        print(f"  [SKIP] No rgb.mp4 in {episode_dir.name}")
        return False

    instruction = get_instruction(episode_dir, episode_dir.name)
    ep_index    = lerobot_dataset.meta.total_episodes
    resize      = (TGT_W, TGT_H)

    tmp_dir = lerobot_dataset.root / "_tmp" / episode_dir.name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. Re-encode videos → consistent codec & resolution ───────────
        tmp_rgb_vid   = tmp_dir / "rgb.mp4"
        tmp_depth_vid = tmp_dir / "depth.mp4"

        print(f"  Re-encoding rgb.mp4 …",   end=" ", flush=True)
        reencode_video(src_rgb, tmp_rgb_vid, resize_wh=resize)
        print("done")

        has_depth = src_depth.exists()
        if has_depth:
            print(f"  Re-encoding depth.mp4 …", end=" ", flush=True)
            reencode_video(src_depth, tmp_depth_vid, resize_wh=resize)
            print("done")
        else:
            print(f"  [WARN] No depth.mp4 found.")

        # ── 2. Extract frames at IMAGE_EXTRACT_FPS into tmp dirs ──────────
        tmp_rgb_frames   = tmp_dir / "rgb_frames"
        tmp_depth_frames = tmp_dir / "depth_frames"

        print(f"  Extracting RGB frames at {IMAGE_EXTRACT_FPS} fps …", end=" ", flush=True)
        rgb_frame_paths = extract_frames_at_fps(
            tmp_rgb_vid, tmp_rgb_frames, IMAGE_EXTRACT_FPS, ep_index, resize_wh=resize
        )
        n_frames = len(rgb_frame_paths)
        print(f"{n_frames} frames")

        depth_frame_paths: List[Path] = []
        if has_depth:
            print(f"  Extracting depth frames at {IMAGE_EXTRACT_FPS} fps …", end=" ", flush=True)
            depth_frame_paths = extract_frames_at_fps(
                tmp_depth_vid, tmp_depth_frames, IMAGE_EXTRACT_FPS, ep_index, resize_wh=resize
            )
            print(f"{len(depth_frame_paths)} frames")

        if n_frames == 0:
            print(f"  [SKIP] No frames extracted.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False

        # ── 3. Build pose metadata (one entry per extracted frame) ────────
        frame_metas = load_keyframe_poses(
            episode_dir, pose_by_frame_idx, instruction, n_frames
        )

        # ── 4. Add scalar frames to the episode buffer ────────────────────
        #    NOTE: video/image keys are NOT added via add_frame — they go via files={}
        print(f"  Adding {n_frames} frames to buffer …", end=" ", flush=True)
        for meta in frame_metas:
            lerobot_dataset.add_frame(
                frame={
                    "observation.state": meta["state"],
                    "action":            meta["action"],
                },
                task=meta["instruction"],
                timestamp=meta["timestamp"],
            )
        print("done")

        # ── 5. Save episode — pass all four paths ─────────────────────────
        files = {
            # Full videos → observation.video.*
            "observation.video.rgb":    tmp_rgb_vid,
            # Extracted image folders → observation.images.*
            "observation.images.rgb":   tmp_rgb_frames,
        }
        if has_depth:
            files["observation.video.depth"]   = tmp_depth_vid
            files["observation.images.depth"]  = tmp_depth_frames

        lerobot_dataset.save_episode(files=files)

        # ── 6. Clean up tmp ───────────────────────────────────────────────
        shutil.rmtree(tmp_dir, ignore_errors=True)

        ep_idx = lerobot_dataset.meta.total_episodes - 1
        print(
            f"  ✓ {episode_dir.name} → lerobot episode {ep_idx:06d}"
            f"  ({n_frames} extracted frames @ {IMAGE_EXTRACT_FPS} fps, {TGT_W}×{TGT_H})"
        )
        return True

    except Exception as e:
        lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise e


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(
    keyframe_root: Path = KEYFRAME_ROOT,
    lerobot_out:   Path = LEROBOT_OUT,
    repo_name:     str  = REPO_NAME,
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

    print(f"Found {len(episode_dirs)} episodes")
    print(f"Image extract FPS : {IMAGE_EXTRACT_FPS}  (change IMAGE_EXTRACT_FPS in CONFIG)")
    print(f"Video native FPS  : {VIDEO_FPS}")
    print(f"Target resolution : {TGT_W}×{TGT_H}")

    pose_by_frame_idx = load_keyframes_json(keyframe_root)
    print(f"Loaded pose metadata for {len(pose_by_frame_idx)} keyframes")

    lerobot_dataset = NavDataset.create(
        repo_id=repo_name,
        root=lerobot_out / repo_name,
        robot_type="mobile_robot",
        fps=DATASET_FPS,
        use_videos=True,
        features=get_features(),
    )

    success = 0
    for ep_dir in episode_dirs:
        print(f"\n[{ep_dir.name}]")
        try:
            ok = convert_episode(ep_dir, pose_by_frame_idx, lerobot_dataset)
            if ok:
                success += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Converted {success}/{len(episode_dirs)} episodes")
    print(f"Output → {lerobot_out / repo_name}")
    print(sep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert episode videos to LeRobot format")
    parser.add_argument("--keyframe_root",     type=str, default=str(KEYFRAME_ROOT))
    parser.add_argument("--lerobot_out",       type=str, default=str(LEROBOT_OUT))
    parser.add_argument("--repo_name",         type=str, default=REPO_NAME)
    parser.add_argument("--image_extract_fps", type=int, default=IMAGE_EXTRACT_FPS,
                        help="FPS at which frames are saved into observation.images.* folders")
    args = parser.parse_args()

    IMAGE_EXTRACT_FPS = args.image_extract_fps
    DATASET_FPS       = IMAGE_EXTRACT_FPS

    main(
        keyframe_root=Path(args.keyframe_root),
        lerobot_out=Path(args.lerobot_out),
        repo_name=args.repo_name,
    )