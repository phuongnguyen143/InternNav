"""Parse/write camera and floor trajectory text files; timestamp matching."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, List, Optional, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from utils.config import get_config
from utils.extrinsics import apply_body2optical_transform

T = TypeVar("T")


@dataclass
class FloorEntry:
    timestamp: float
    x: float
    y: float
    yaw: float
    z: float = 0.0
    # World-frame 3D point on the floor plane (map/SLAM frame).
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0


@dataclass
class OdomEntry:
    timestamp: float
    matrix: np.ndarray
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    yaw: float = 0.0


class TimestampMatcher(Generic[T]):
    """Binary-search closest entry by timestamp within max_dt."""

    def __init__(self, entries: List[T], max_dt: float = 0.5):
        self.entries = entries
        self.timestamps = [e.timestamp for e in entries]
        self.max_dt = max_dt

    def find_closest(self, query_ts: float) -> Optional[T]:
        ts = self.timestamps
        idx = bisect.bisect_left(ts, query_ts)
        candidates = []
        if idx < len(ts):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best_idx = min(candidates, key=lambda i: abs(ts[i] - query_ts))
        if abs(ts[best_idx] - query_ts) > self.max_dt:
            return None
        return self.entries[best_idx]


FloorMatcher = TimestampMatcher[FloorEntry]
OdomMatcher = TimestampMatcher[OdomEntry]


def parse_floor_trajectory_txt(filepath: str | Path) -> List[FloorEntry]:
    """
    Format (blocks of 2 lines):
        <timestamp>
        <x> <y> <yaw> [<legacy_z> [<world_x> <world_y> <world_z>]]

    legacy_z is the old world-Z scalar (kept for backward compatibility).
    When world_x/y/z are omitted they default to 0 and should be recomputed
    from (x, y) + floor calibration at load time.
    """
    entries: List[FloorEntry] = []
    lines = Path(filepath).read_text().strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            timestamp = float(line)
        except ValueError:
            i += 1
            continue
        if i + 1 >= len(lines):
            break
        parts = lines[i + 1].strip().split()
        if len(parts) < 3:
            i += 1
            continue
        x, y, yaw = float(parts[0]), float(parts[1]), float(parts[2])
        legacy_z = float(parts[3]) if len(parts) >= 4 else 0.0
        if len(parts) >= 7:
            world_x, world_y, world_z = float(parts[4]), float(parts[5]), float(parts[6])
        else:
            world_x = world_y = world_z = 0.0
        entries.append(
            FloorEntry(
                timestamp=timestamp,
                x=x,
                y=y,
                yaw=yaw,
                z=legacy_z,
                world_x=world_x,
                world_y=world_y,
                world_z=world_z,
            )
        )
        i += 2

    if not entries:
        raise ValueError(f"No floor trajectory entries parsed from {filepath}")
    print(
        f"[FloorParser] Loaded {len(entries)} entries from {filepath} "
        f"({entries[0].timestamp:.3f} -> {entries[-1].timestamp:.3f})"
    )
    return entries


def write_floor_trajectory_txt(filepath: str | Path, entries: List[FloorEntry]) -> Path:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for e in entries:
        lines.append(f"{e.timestamp:.9f}")
        lines.append(f"{e.x:.8f} {e.y:.8f} {e.yaw:.8f} {e.z:.8f} " f"{e.world_x:.8f} {e.world_y:.8f} {e.world_z:.8f}")
    path.write_text("\n".join(lines) + "\n")
    print(f"[FloorParser] Wrote {len(entries)} entries to {path}")
    return path


def parse_tum_poses_txt(filepath: str | Path) -> List[tuple[int, np.ndarray]]:
    """
    Parse TUM pose lines: ``frame_idx tx ty tz qx qy qz qw`` -> (index, 4x4 c2w).
    Used by WildGS-SLAM ``est_poses_full.txt``.
    """
    poses: List[tuple[int, np.ndarray]] = []
    for line in Path(filepath).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        frame_idx = int(parts[0])
        t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(q).as_matrix()
        matrix[:3, 3] = t
        poses.append((frame_idx, matrix))
    if not poses:
        raise ValueError(f"No TUM poses parsed from {filepath}")
    return poses


def load_frame_timestamps(frames_json: str | Path) -> List[float]:
    """Load per-frame ROS timestamps from ``frames.json`` (extract_bag_frames output)."""
    records = json.loads(Path(frames_json).read_text())
    if not records:
        raise ValueError(f"No frames in {frames_json}")
    return [float(r["timestamp"]) for r in records]


def merge_tum_poses_with_frame_timestamps(
    tum_poses_file: str | Path,
    frames_json: str | Path,
    stride: int = 1,
) -> List[OdomEntry]:
    """
    Attach bag timestamps to SLAM poses.

    ``tum_poses_file`` indices are over the strided image list used by WildGS-SLAM.
    ``frames_json`` lists every extracted frame; pose index *i* maps to
    ``frames_json[i * stride]``.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    tum_poses = parse_tum_poses_txt(tum_poses_file)
    timestamps = load_frame_timestamps(frames_json)

    entries: List[OdomEntry] = []
    for pose_idx, matrix in tum_poses:
        if pose_idx != len(entries):
            raise ValueError(
                f"Expected consecutive pose indices starting at 0; "
                f"got index {pose_idx} at line {len(entries) + 1} in {tum_poses_file}"
            )
        frame_record_idx = pose_idx * stride
        if frame_record_idx >= len(timestamps):
            raise ValueError(
                f"Pose index {pose_idx} (frame {frame_record_idx} with stride={stride}) "
                f"exceeds frames.json length ({len(timestamps)})"
            )
        ts = timestamps[frame_record_idx]
        rot_matrix = matrix[:3, :3]
        quat = Rotation.from_matrix(rot_matrix).as_quat()
        entries.append(
            OdomEntry(
                timestamp=ts,
                matrix=matrix,
                x=float(matrix[0, 3]),
                y=float(matrix[1, 3]),
                z=float(matrix[2, 3]),
                qx=float(quat[0]),
                qy=float(quat[1]),
                qz=float(quat[2]),
                qw=float(quat[3]),
                yaw=float(Rotation.from_matrix(rot_matrix).as_euler("xyz")[2]),
            )
        )
    return entries


def write_odom_txt(filepath: str | Path, entries: List[OdomEntry]) -> Path:
    """
    Write camera odometry in GaussTrace format:
    timestamp line + 4x4 T_world_cam matrix rows + blank line between poses.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: List[str] = []
    for entry in entries:
        blocks.append(f"{entry.timestamp:.9f}")
        for row in range(4):
            blocks.append(" ".join(f"{entry.matrix[row, col]:.9f}" for col in range(4)))
        blocks.append("")
    path.write_text("\n".join(blocks))
    print(f"[OdomWriter] Wrote {len(entries)} entries to {path}")
    if entries:
        print(
            f"  Time range: {entries[0].timestamp:.3f} -> "
            f"{entries[-1].timestamp:.3f} "
            f"({entries[-1].timestamp - entries[0].timestamp:.1f}s)"
        )
    return path


def export_odom_from_tum_and_frames(
    tum_poses_file: str | Path,
    frames_json: str | Path,
    output_file: str | Path,
    stride: int = 1,
) -> Path:
    """Merge WildGS-SLAM TUM poses with bag timestamps and write odom txt."""
    entries = merge_tum_poses_with_frame_timestamps(
        tum_poses_file=tum_poses_file,
        frames_json=frames_json,
        stride=stride,
    )
    return write_odom_txt(output_file, entries)


def parse_odom_txt(filepath: str, apply_body2optical: bool = False) -> list[OdomEntry]:
    """Parse camera odometry txt: timestamp line + 4x4 T_world_cam matrix."""
    entries = []
    lines = Path(filepath).read_text().strip().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            timestamp = float(line)
        except ValueError:
            i += 1
            continue

        matrix_lines = []
        for j in range(1, 5):
            if i + j < len(lines):
                matrix_lines.append(lines[i + j].strip())

        if len(matrix_lines) < 4:
            break

        try:
            matrix = np.array([[float(v) for v in row.split()] for row in matrix_lines])
        except ValueError:
            i += 1
            continue

        if matrix.shape != (4, 4):
            i += 5
            continue

        matrix = apply_body2optical_transform(matrix, apply=apply_body2optical)

        tx, ty, tz = matrix[0, 3], matrix[1, 3], matrix[2, 3]
        rot_matrix = matrix[:3, :3]
        quat = Rotation.from_matrix(rot_matrix).as_quat()
        yaw = Rotation.from_matrix(rot_matrix).as_euler("xyz")[2]

        entry = OdomEntry(
            timestamp=timestamp,
            matrix=matrix,
            x=tx,
            y=ty,
            z=tz,
            qx=quat[0],
            qy=quat[1],
            qz=quat[2],
            qw=quat[3],
            yaw=yaw,
        )
        entries.append(entry)
        i += 6

    print(f"[OdomParser] Loaded {len(entries)} entries from {filepath}")
    print(
        f"  Time range: {entries[0].timestamp:.3f} → "
        f"{entries[-1].timestamp:.3f} "
        f"({entries[-1].timestamp - entries[0].timestamp:.1f}s)"
    )
    return entries
