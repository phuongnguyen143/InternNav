#!/usr/bin/env python3
"""
Offline floor estimation + embodiment trajectory from camera odometry.

Run once per scene before the live bag/keyframe pipeline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from constants import FLOOR_CALIBRATION_FILENAME, FLOOR_TRAJECTORY_FILENAME
from floor_pose import (
    camera_matrix_to_floor_pose,
    estimate_floor_local,
    load_pcd_points,
    save_floor_calibration,
)
from trajectory_io import FloorEntry, parse_odom_txt, write_floor_trajectory_txt


def _auto_patch_params(num_points: int) -> dict:
    """Coarser grid for large clouds to avoid 700k+ RANSAC cells."""
    if num_points > 500_000:
        return {"patch_radius": 2.0, "stride": 2.0, "min_patch_points": 150}
    if num_points > 100_000:
        return {"patch_radius": 1.5, "stride": 1.0, "min_patch_points": 200}
    return {"patch_radius": 1.0, "stride": 0.5, "min_patch_points": 200}


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute floor plane and floor embodiment trajectory txt.")
    parser.add_argument("--pcd", required=True, help="Scene point cloud (.ply/.pcd)")
    parser.add_argument(
        "--camera_odom",
        required=True,
        help="Camera odometry txt (timestamp + 4x4 T_world_cam)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for floor_calibration.json and floor_trajectory.txt",
    )
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--patch_radius", type=float, default=None)
    parser.add_argument("--stride", type=float, default=None)
    args = parser.parse_args()

    pcd_path = Path(args.pcd)
    camera_odom_path = Path(args.camera_odom)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pcd_path.is_file():
        raise FileNotFoundError(pcd_path)
    if not camera_odom_path.is_file():
        raise FileNotFoundError(camera_odom_path)

    t0 = time.time()
    raw = load_pcd_points(pcd_path)
    print(f"[precompute] Point cloud: {len(raw):,} points")

    patch_kw = _auto_patch_params(len(raw))
    if args.patch_radius is not None:
        patch_kw["patch_radius"] = args.patch_radius
    if args.stride is not None:
        patch_kw["stride"] = args.stride
    print(f"[precompute] estimate_local params: {patch_kw}")

    print("[precompute] Estimating floor plane (this may take several minutes) ...")
    floor_plane, floor_normal, floor_point, inliers, up_axis = estimate_floor_local(
        pcd_path,
        voxel_size=args.voxel_size,
        **patch_kw,
    )
    print(f"[precompute] Floor plane {floor_plane} | inliers={len(inliers):,} | " f"elapsed={time.time() - t0:.1f}s")

    save_floor_calibration(
        output_dir,
        floor_plane,
        floor_normal,
        floor_point,
        pcd_path=pcd_path,
        extra={
            "num_inliers": int(len(inliers)),
            "up_axis": up_axis.tolist(),
            "patch_params": patch_kw,
        },
    )

    camera_entries = parse_odom_txt(str(camera_odom_path))
    floor_entries: list[FloorEntry] = []

    for entry in tqdm(camera_entries, desc="Project camera odom to floor"):
        x, y, yaw, z, _ = camera_matrix_to_floor_pose(entry.matrix, floor_plane)
        floor_entries.append(FloorEntry(timestamp=entry.timestamp, x=x, y=y, yaw=yaw, z=z))

    traj_path = output_dir / FLOOR_TRAJECTORY_FILENAME
    write_floor_trajectory_txt(traj_path, floor_entries)

    print(f"[precompute] Done in {time.time() - t0:.1f}s")
    print(f"  calibration: {output_dir / FLOOR_CALIBRATION_FILENAME}")
    print(f"  trajectory:  {traj_path}")
    print(f"  entries:     {len(floor_entries)}")


if __name__ == "__main__":
    main()
