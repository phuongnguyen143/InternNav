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

from utils.config import get_config, load_config
from utils.floor_pose import (
    camera_matrix_to_floor_pose,
    estimate_floor_local,
    load_pcd_points,
    save_floor_calibration,
    sync_floor_entry_world_xyz,
)
from utils.trajectory_io import FloorEntry, parse_odom_txt, write_floor_trajectory_txt
from utils.trajectory_smooth import (
    SMOOTH_METHODS,
    draw_smooth_comparison,
    smooth_config_dict,
    smooth_floor_trajectory,
)


def _auto_patch_params(num_points: int) -> dict:
    if num_points > 500_000:
        return {"patch_radius": 2.0, "stride": 2.0, "min_patch_points": 150}
    if num_points > 100_000:
        return {"patch_radius": 1.5, "stride": 1.0, "min_patch_points": 200}
    return {"patch_radius": 1.0, "stride": 0.5, "min_patch_points": 200}


def main() -> None:
    import sys

    scene = None
    if "--scene" in sys.argv:
        idx = sys.argv.index("--scene")
        if idx + 1 < len(sys.argv):
            scene = sys.argv[idx + 1]
    vln_cfg = load_config(scene) if scene else get_config()
    precompute = vln_cfg.precompute
    paths = vln_cfg.get("paths", default={})

    parser = argparse.ArgumentParser(description="Precompute floor plane and floor embodiment trajectory txt.")
    parser.add_argument("--scene", type=str, default=scene)
    parser.add_argument("--pcd", type=Path, default=Path(paths["pcd"]) if paths.get("pcd") else None)
    parser.add_argument(
        "--camera_odom",
        type=Path,
        default=Path(paths["camera_odom"]) if paths.get("camera_odom") else None,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(paths.get("floor_trajectory", ".")).parent if paths.get("floor_trajectory") else None,
    )
    parser.add_argument("--voxel_size", type=float, default=float(precompute.get("voxel_size", 0.1)))
    parser.add_argument("--patch_radius", type=float, default=None)
    parser.add_argument("--stride", type=float, default=None)
    parser.add_argument("--smooth", choices=SMOOTH_METHODS, default=str(precompute.get("smooth", "moving_average")))
    parser.add_argument("--smooth_window", type=int, default=int(precompute.get("smooth_window", 5)))
    parser.add_argument("--smooth_s", type=float, default=float(precompute.get("smooth_s", 1.0)))
    parser.add_argument("--draw_smooth", action="store_true")
    parser.add_argument("--draw_smooth_save", default=None)
    parser.add_argument("--draw_smooth_no_interactive", action="store_true")
    args = parser.parse_args()

    if args.smooth_window % 2 == 0 or args.smooth_window < 3:
        parser.error("--smooth_window must be an odd integer >= 3")
    for label, value in (("pcd", args.pcd), ("camera_odom", args.camera_odom), ("output_dir", args.output_dir)):
        if value is None:
            parser.error(f"--{label} is required (or set paths.{label} in scene config)")

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
    print(f"[precompute] Floor plane {floor_plane} | inliers={len(inliers):,} | elapsed={time.time() - t0:.1f}s")

    smooth_meta = smooth_config_dict(args.smooth, window=args.smooth_window, bspline_s=args.smooth_s)
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
            "trajectory_smooth": smooth_meta,
        },
    )

    b2o_val = vln_cfg.get("odom_apply_body2optical")
    apply_body2optical = True if b2o_val is None else bool(b2o_val)
    camera_entries = parse_odom_txt(str(camera_odom_path))

    floor_entries: list[FloorEntry] = []

    for entry in tqdm(camera_entries, desc="Project camera odom to floor"):
        x, y, yaw, legacy_z, _, world_xyz = camera_matrix_to_floor_pose(
            entry.matrix,
            floor_plane,
            apply_body2optical=apply_body2optical,
        )
        floor_entries.append(
            FloorEntry(
                timestamp=entry.timestamp,
                x=x,
                y=y,
                yaw=yaw,
                z=legacy_z,
                world_x=float(world_xyz[0]),
                world_y=float(world_xyz[1]),
                world_z=float(world_xyz[2]),
            )
        )

    floor_entries_raw = list(floor_entries)
    if args.smooth != "none":
        floor_entries = smooth_floor_trajectory(
            floor_entries,
            method=args.smooth,
            window=args.smooth_window,
            bspline_s=args.smooth_s,
        )

    if args.draw_smooth:
        if args.draw_smooth_no_interactive and not args.draw_smooth_save:
            raise SystemExit("--draw_smooth_no_interactive requires --draw_smooth_save")
        save_path = args.draw_smooth_save
        if save_path is None and args.draw_smooth_no_interactive:
            save_path = str(output_dir / "trajectory_smooth_compare.png")
        draw_smooth_comparison(
            floor_entries_raw,
            floor_entries,
            method=args.smooth,
            save_path=save_path,
            interactive=not args.draw_smooth_no_interactive,
            window=args.smooth_window,
            bspline_s=args.smooth_s,
        )

    traj_path = output_dir / vln_cfg.floor_trajectory_filename()
    floor_entries = sync_floor_entry_world_xyz(floor_entries, floor_plane)
    write_floor_trajectory_txt(traj_path, floor_entries)

    print(f"[precompute] Done in {time.time() - t0:.1f}s")
    print(f"  calibration: {output_dir / vln_cfg.floor_calibration_filename()}")
    print(f"  trajectory:  {traj_path}")
    print(f"  entries:     {len(floor_entries)}")


if __name__ == "__main__":
    main()
