#!/usr/bin/env python3
"""Derive floor_calibration.json from floor_trajectory.txt (no point cloud).

Use when floor_trajectory was exported from SLAM camera odometry
(project_slam_path.py --export-floor-trajectory) and includes world_x/y/z.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.config import get_config, load_config
from utils.floor_pose import derive_floor_calibration_from_trajectory


def main() -> None:
    scene = None
    if "--scene" in sys.argv:
        idx = sys.argv.index("--scene")
        if idx + 1 < len(sys.argv):
            scene = sys.argv[idx + 1]

    vln_cfg = load_config(scene) if scene else get_config()
    paths = vln_cfg.get("paths", default={})

    parser = argparse.ArgumentParser(
        description="Derive floor_calibration.json from floor_trajectory world_x/y/z (PCD-free)."
    )
    parser.add_argument("--scene", type=str, default=scene)
    parser.add_argument(
        "--floor_trajectory",
        type=Path,
        default=Path(paths["floor_trajectory"]) if paths.get("floor_trajectory") else None,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(paths.get("floor_trajectory", ".")).parent if paths.get("floor_trajectory") else None,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.floor_trajectory is None:
        parser.error("--floor_trajectory is required (or set paths.floor_trajectory in scene config)")
    if args.output_dir is None:
        parser.error("--output_dir is required")

    traj_path = Path(args.floor_trajectory)
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)

    out_path = derive_floor_calibration_from_trajectory(
        traj_path,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
