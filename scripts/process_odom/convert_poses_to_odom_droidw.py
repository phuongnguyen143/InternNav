#!/usr/bin/env python3
"""Convert WildGS-SLAM est_poses_full.txt to GaussTrace / BKHN odometry format.

Input (TUM-style, one line per strided frame):
    frame_idx tx ty tz qx qy qz qw

Output (matches odometry_*_point2plane.txt):
    <timestamp>
    r00 r01 r02 tx
    r10 r11 r12 ty
    r20 r21 r22 tz
    0   0   0   1

    <next timestamp>
    ...

Timestamps come from frames.json produced by extract_bag_frames. SLAM pose index i
maps to frames_json[i * stride]. By default, poses for intermediate frames are
interpolated (linear translation + quaternion SLERP) using camera timestamps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def parse_tum_poses(path: Path) -> list[tuple[int, np.ndarray]]:
    poses: list[tuple[int, np.ndarray]] = []
    for line in path.read_text().splitlines():
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
        raise ValueError(f"No poses parsed from {path}")
    return poses


def load_frame_timestamps(frames_json: Path) -> list[float]:
    records = json.loads(frames_json.read_text())
    if not records:
        raise ValueError(f"No frames in {frames_json}")
    return [float(r["timestamp"]) for r in records]


def merge_poses_with_timestamps(
    poses: list[tuple[int, np.ndarray]],
    timestamps: list[float],
    stride: int,
) -> list[tuple[float, np.ndarray]]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    entries: list[tuple[float, np.ndarray]] = []
    for pose_idx, matrix in poses:
        if pose_idx != len(entries):
            raise ValueError(
                f"Expected consecutive pose indices starting at 0; " f"got index {pose_idx} at line {len(entries) + 1}"
            )
        frame_record_idx = pose_idx * stride
        if frame_record_idx >= len(timestamps):
            raise ValueError(
                f"Pose index {pose_idx} (frame {frame_record_idx} with stride={stride}) "
                f"exceeds frames.json length ({len(timestamps)})"
            )
        entries.append((timestamps[frame_record_idx], matrix))
    return entries


def interpolate_pose(
    t0: float,
    m0: np.ndarray,
    t1: float,
    m1: np.ndarray,
    ts: float,
) -> np.ndarray:
    if ts <= t0:
        return m0.copy()
    if ts >= t1:
        return m1.copy()
    if t1 <= t0:
        return m0.copy()

    alpha = (ts - t0) / (t1 - t0)
    trans = (1.0 - alpha) * m0[:3, 3] + alpha * m1[:3, 3]
    rots = Rotation.from_matrix(np.stack([m0[:3, :3], m1[:3, :3]]))
    rot = Slerp([0.0, 1.0], rots)([alpha]).as_matrix()[0]

    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = trans
    return matrix


def interpolate_poses_to_all_frames(
    keyframe_entries: list[tuple[float, np.ndarray]],
    timestamps: list[float],
    stride: int,
) -> list[tuple[float, np.ndarray]]:
    if not keyframe_entries:
        return []

    last_frame_idx = (len(keyframe_entries) - 1) * stride
    if last_frame_idx >= len(timestamps):
        raise ValueError(f"Last SLAM frame index {last_frame_idx} exceeds frames.json length " f"({len(timestamps)})")

    key_ts = [entry[0] for entry in keyframe_entries]
    key_mats = [entry[1] for entry in keyframe_entries]
    target_timestamps = timestamps[: last_frame_idx + 1]

    entries: list[tuple[float, np.ndarray]] = []
    key_idx = 0
    for ts in target_timestamps:
        while key_idx + 1 < len(key_ts) and key_ts[key_idx + 1] < ts:
            key_idx += 1

        if ts <= key_ts[0]:
            matrix = key_mats[0].copy()
        elif ts >= key_ts[-1]:
            matrix = key_mats[-1].copy()
        elif ts == key_ts[key_idx]:
            matrix = key_mats[key_idx].copy()
        else:
            matrix = interpolate_pose(
                key_ts[key_idx],
                key_mats[key_idx],
                key_ts[key_idx + 1],
                key_mats[key_idx + 1],
                ts,
            )
        entries.append((ts, matrix))
    return entries


def write_odom_txt(path: Path, entries: list[tuple[float, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for timestamp, matrix in entries:
        blocks.append(f"{timestamp:.9f}")
        for row in range(4):
            blocks.append(" ".join(f"{matrix[row, col]:.9f}" for col in range(4)))
        blocks.append("")
    path.write_text("\n".join(blocks))


def convert_poses_to_odom(
    poses_file: Path,
    frames_json: Path,
    output_file: Path,
    stride: int,
    interpolate: bool = True,
) -> Path:
    poses = parse_tum_poses(poses_file)
    timestamps = load_frame_timestamps(frames_json)
    keyframe_entries = merge_poses_with_timestamps(poses, timestamps, stride)
    if interpolate:
        entries = interpolate_poses_to_all_frames(keyframe_entries, timestamps, stride)
    else:
        entries = keyframe_entries

    write_odom_txt(output_file, entries)
    print(f"Wrote {len(entries)} poses to {output_file}")
    if interpolate and len(keyframe_entries) != len(entries):
        print(
            f"  Interpolated {len(entries) - len(keyframe_entries)} frames "
            f"from {len(keyframe_entries)} SLAM keyframes (stride={stride})"
        )
    if entries:
        t0, t1 = entries[0][0], entries[-1][0]
        print(f"  Time range: {t0:.3f} -> {t1:.3f} ({t1 - t0:.1f}s)")
    return output_file


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_poses = repo_root / "output/wildgs_slam_custom/office/traj/est_poses_full.txt"
    default_frames = Path(
        "/home/lenguyen1/hoangpqn/vln/InternNav/scripts/process_odom/" "keyframe_output_offline_office/frames.json"
    )
    default_output = repo_root / "output/wildgs_slam_custom/office/traj/odometry_camera.txt"

    parser = argparse.ArgumentParser(
        description=(
            "Convert WildGS-SLAM est_poses_full.txt to BKHN/GaussTrace odometry format "
            "with camera timestamps from frames.json."
        )
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=default_poses,
        help="Input est_poses_full.txt (frame_idx tx ty tz qx qy qz qw)",
    )
    parser.add_argument(
        "--frames-json",
        type=Path,
        default=default_frames,
        help="frames.json with per-frame ROS timestamps from bag extraction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output odometry txt (timestamp + 4x4 T_world_cam matrix)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Frame stride used by WildGS-SLAM (must match config stride)",
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Only output SLAM poses at strided frames (skip in-between frames)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.poses.is_file():
        print(f"Error: poses file not found: {args.poses}", file=sys.stderr)
        return 1
    if not args.frames_json.is_file():
        print(f"Error: frames.json not found: {args.frames_json}", file=sys.stderr)
        return 1

    try:
        convert_poses_to_odom(
            poses_file=args.poses,
            frames_json=args.frames_json,
            output_file=args.output,
            stride=args.stride,
            interpolate=not args.no_interpolate,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
