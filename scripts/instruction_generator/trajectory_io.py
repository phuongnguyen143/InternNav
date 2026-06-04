"""Parse/write camera and floor trajectory text files; timestamp matching."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, List, Optional, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from constants import FLOOR_TRAJECTORY_FILENAME

T = TypeVar("T")


@dataclass
class FloorEntry:
    timestamp: float
    x: float
    y: float
    yaw: float
    z: float = 0.0


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


# Readable aliases for call sites
FloorMatcher = TimestampMatcher[FloorEntry]
OdomMatcher = TimestampMatcher[OdomEntry]


def parse_floor_trajectory_txt(filepath: str | Path) -> List[FloorEntry]:
    """
    Format (blocks of 2 lines):
        <timestamp>
        <x> <y> <yaw> [<z>]
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
        z = float(parts[3]) if len(parts) >= 4 else 0.0
        entries.append(FloorEntry(timestamp=timestamp, x=x, y=y, yaw=yaw, z=z))
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
        lines.append(f"{e.x:.8f} {e.y:.8f} {e.yaw:.8f} {e.z:.8f}")
    path.write_text("\n".join(lines) + "\n")
    print(f"[FloorParser] Wrote {len(entries)} entries to {path}")
    return path


def parse_odom_txt(filepath: str) -> list[OdomEntry]:
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
            matrix = np.array([
                [float(v) for v in row.split()]
                for row in matrix_lines
            ])
        except ValueError:
            i += 1
            continue

        if matrix.shape != (4, 4):
            i += 5
            continue

        tx, ty, tz = matrix[0, 3], matrix[1, 3], matrix[2, 3]
        rot_matrix = matrix[:3, :3]
        quat = Rotation.from_matrix(rot_matrix).as_quat()
        yaw = Rotation.from_matrix(rot_matrix).as_euler('xyz')[2]

        entry = OdomEntry(
            timestamp=timestamp,
            matrix=matrix,
            x=tx, y=ty, z=tz,
            qx=quat[0], qy=quat[1], qz=quat[2], qw=quat[3],
            yaw=yaw,
        )
        entries.append(entry)
        i += 6

    print(f"[OdomParser] Loaded {len(entries)} entries "
          f"from {filepath}")
    print(f"  Time range: {entries[0].timestamp:.3f} → "
          f"{entries[-1].timestamp:.3f} "
          f"({entries[-1].timestamp - entries[0].timestamp:.1f}s)")
    return entries
