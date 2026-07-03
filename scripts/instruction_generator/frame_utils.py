"""Shared utilities for rosbag / stream frame extraction."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.config import get_config
from utils.depth_codec import (
    decode_compressed_depth,
    decode_raw_depth_image,
    save_depth_png_mm,
)


def ros_stamp_to_sec(stamp) -> float:
    """Convert builtin_interfaces/Time or any msg with sec/nanosec to float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def decode_rgb_compressed(data: bytes) -> Optional[np.ndarray]:
    """Decode sensor_msgs/CompressedImage JPEG/PNG payload to BGR uint8."""
    rgb = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    return rgb


def decode_depth_compressed(data: bytes, format_hint: str = "") -> Optional[np.ndarray]:
    """Decode compressedDepth payload to float32 meters."""
    return decode_compressed_depth(data, format_hint=format_hint)


@dataclass
class StampedDepth:
    timestamp: float
    data: bytes
    format_hint: str = ""
    encoding: str = ""
    height: int = 0
    width: int = 0
    step: int = 0

    @property
    def is_raw_image(self) -> bool:
        return bool(self.encoding)


def decode_stamped_depth(msg: StampedDepth) -> Optional[np.ndarray]:
    """Decode a stamped depth message (compressedDepth or raw sensor_msgs/Image)."""
    if msg.is_raw_image:
        return decode_raw_depth_image(
            msg.data,
            msg.encoding,
            msg.height,
            msg.width,
            msg.step,
        )
    return decode_depth_compressed(msg.data, msg.format_hint)


def align_depth_to_rgb(depth_m: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Resize depth to match RGB height/width with nearest-neighbor."""
    h, w = rgb.shape[:2]
    if depth_m.shape[0] == h and depth_m.shape[1] == w:
        return depth_m
    return cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)


def depth_m_to_preview_bgr(
    depth_m: np.ndarray,
    vis_scale: float | None = None,
) -> np.ndarray:
    """Convert depth in meters to an 8-bit BGR preview frame for mp4 debug video."""
    if vis_scale is None:
        vis_scale = float(get_config().depth.get("vis_scale", 10000.0))
    depth_vis = cv2.convertScaleAbs(depth_m, alpha=255.0 / vis_scale)
    return cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)


def save_rgb_frame(path: Path, rgb_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rgb_bgr):
        raise RuntimeError(f"Failed to write RGB frame: {path}")


def save_depth_frame_mm(path: Path, depth_m: np.ndarray) -> None:
    save_depth_png_mm(path, depth_m)


def open_mp4_writer(path: Path, fps: float, size_wh: Tuple[int, int]):
    """Open an mp4v VideoWriter. size_wh = (width, height)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size_wh)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    return writer


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def get_frame_from_video(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    return read_video_frame(cap, frame_idx)


def write_rgb_mp4_segment(
    src_video: Path,
    dst_video: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    size_wh: Tuple[int, int],
) -> int:
    """Copy inclusive [start_frame, end_frame] from src mp4 to dst mp4. Returns frame count."""
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_video}")

    writer = open_mp4_writer(dst_video, fps, size_wh)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    count = 0
    for _ in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        count += 1

    writer.release()
    cap.release()
    return count


def copy_depth_frame_range(
    src_dir: Path,
    dst_dir: Path,
    start_frame: int,
    end_frame: int,
) -> int:
    """Copy depth_frames/frame_XXXXXX.png for an inclusive frame range."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame_idx in range(start_frame, end_frame + 1):
        src = src_dir / f"frame_{frame_idx:06d}.png"
        if not src.exists():
            raise RuntimeError(f"Missing depth frame: {src}")
        shutil.copy2(src, dst_dir / f"frame_{frame_idx:06d}.png")
        count += 1
    return count


@dataclass
class StreamFrame:
    frame_idx: int
    timestamp: float
    rgb_bgr: np.ndarray
    depth_m: np.ndarray


def _find_closest(
    items: Sequence[StampedDepth],
    target_ts: float,
    max_dt: float,
) -> Optional[StampedDepth]:
    if not items:
        return None
    idx = int(np.searchsorted([m.timestamp for m in items], target_ts))
    candidates = []
    if 0 <= idx < len(items):
        candidates.append(items[idx])
    if idx - 1 >= 0:
        candidates.append(items[idx - 1])
    best = min(candidates, key=lambda m: abs(m.timestamp - target_ts))
    if abs(best.timestamp - target_ts) > max_dt:
        return None
    return best


def sync_rgb_depth_messages(
    rgb_messages: Iterable[Tuple[float, bytes]],
    depth_messages: Iterable[StampedDepth],
    sync_slop_sec: float | None = None,
) -> Iterator[StreamFrame]:
    """Yield synchronized RGB/depth frames using the RGB stream as reference."""
    if sync_slop_sec is None:
        sync_slop_sec = float(get_config().ros.get("sync_slop_sec", 0.05))
    depth_list = sorted(depth_messages, key=lambda m: m.timestamp)

    frame_idx = 0
    for rgb_ts, rgb_data in rgb_messages:
        rgb = decode_rgb_compressed(rgb_data)
        if rgb is None:
            continue

        depth_msg = _find_closest(depth_list, rgb_ts, sync_slop_sec)
        if depth_msg is None:
            continue

        depth = decode_stamped_depth(depth_msg)
        if depth is None:
            continue

        depth = align_depth_to_rgb(depth, rgb)
        yield StreamFrame(
            frame_idx=frame_idx,
            timestamp=rgb_ts,
            rgb_bgr=rgb,
            depth_m=depth,
        )
        frame_idx += 1
