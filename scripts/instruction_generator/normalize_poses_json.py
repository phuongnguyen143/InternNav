#!/usr/bin/env python3
"""Normalize poses.json camera_matrix to OpenCV optical T_world_cam (c2w)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from utils.config import get_config, load_config
from utils.extrinsics import apply_body2optical_transform
from utils.floor_pose import (
    derive_floor_calibration_from_trajectory,
    floor_2d_pose_to_action_matrix,
    load_floor_calibration,
)
from utils.floor_plane import floor_xy_to_world_on_plane
from utils.trajectory_io import FloorMatcher, OdomMatcher, parse_floor_trajectory_txt, parse_odom_txt


def _load_floor_plane(poses_dir: Path, cfg) -> dict | None:
    cal_path = poses_dir / cfg.floor_calibration_filename()
    if cal_path.is_file():
        return load_floor_calibration(cal_path)["floor_plane"]
    traj_path = poses_dir / cfg.floor_trajectory_filename()
    if traj_path.is_file():
        derived = derive_floor_calibration_from_trajectory(traj_path, poses_dir)
        return load_floor_calibration(derived)["floor_plane"]
    return None


def merge_floor_fields(
    poses: list[dict],
    *,
    floor_traj_path: Path,
    floor_plane: dict,
    max_dt: float,
) -> int:
    entries = parse_floor_trajectory_txt(floor_traj_path)
    matcher = FloorMatcher(entries, max_dt=max_dt)
    merged = 0
    for pose in poses:
        ts = float(pose["timestamp"])
        floor_entry = matcher.find_closest(ts)
        if floor_entry is None:
            continue
        if abs(floor_entry.world_x) + abs(floor_entry.world_y) + abs(floor_entry.world_z) > 1e-9:
            world_xyz = [
                float(floor_entry.world_x),
                float(floor_entry.world_y),
                float(floor_entry.world_z),
            ]
        else:
            world_xyz = floor_xy_to_world_on_plane(
                pose["x"], pose["y"], floor_plane
            ).tolist()
        pose["x"] = float(floor_entry.x)
        pose["y"] = float(floor_entry.y)
        pose["yaw"] = float(floor_entry.yaw)
        pose["z"] = float(floor_entry.z)
        pose["world_x"], pose["world_y"], pose["world_z"] = world_xyz
        pose["action_matrix"] = floor_2d_pose_to_action_matrix(
            pose["x"],
            pose["y"],
            pose["yaw"],
            pose.get("z", 0.0),
            floor_plane,
        ).tolist()
        merged += 1
    return merged


def _apply_body2optical_from_cfg(cfg, args) -> bool:
    b2o_val = cfg.get("odom_apply_body2optical")
    apply_b2o = True if b2o_val is None else bool(b2o_val)
    if args.no_body2optical:
        return False
    if args.body2optical:
        return True
    return apply_b2o


def normalize_poses(
    poses: list[dict],
    *,
    apply_body2optical: bool,
) -> tuple[list[dict], int]:
    updated = 0
    for pose in poses:
        if "camera_matrix" not in pose:
            continue
        if pose.get("camera_frame") == "optical":
            continue
        T = apply_body2optical_transform(
            pose["camera_matrix"], apply=apply_body2optical
        )
        pose["camera_matrix"] = T.astype(float).tolist()
        pose["camera_frame"] = "optical"
        for key in ("camera_x", "camera_y", "camera_z", "camera_yaw"):
            pose.pop(key, None)
        updated += 1
    return poses, updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert poses.json camera_matrix to OpenCV optical frame."
    )
    parser.add_argument(
        "--poses-json",
        type=Path,
        required=True,
        help="Path to poses.json (or keyframe output dir containing it)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene config for odom_apply_body2optical (default: VLN_SCENE / base)",
    )
    parser.add_argument(
        "--camera-odom",
        type=Path,
        default=None,
        help="Re-merge camera_matrix from odom txt (recommended after fixing legacy poses)",
    )
    parser.add_argument("--body2optical", action="store_true")
    parser.add_argument("--no-body2optical", action="store_true")
    parser.add_argument(
        "--merge-floor",
        action="store_true",
        help="Merge action_matrix / world_x/y/z from floor_trajectory.txt in poses dir",
    )
    parser.add_argument(
        "--floor-trajectory",
        type=Path,
        default=None,
        help="Floor trajectory txt (default: <poses dir>/floor_trajectory.txt)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file (default: write poses_optical.json alongside)",
    )
    args = parser.parse_args()

    poses_path = args.poses_json
    if poses_path.is_dir():
        poses_path = poses_path / "poses.json"
    if not poses_path.is_file():
        print(f"Error: not found: {poses_path}", file=sys.stderr)
        return 1

    cfg = load_config(args.scene) if args.scene else get_config()
    apply_b2o = _apply_body2optical_from_cfg(cfg, args)

    poses = json.loads(poses_path.read_text())

    if args.camera_odom is not None:
        if not args.camera_odom.is_file():
            print(f"Error: camera odom not found: {args.camera_odom}", file=sys.stderr)
            return 1
        entries = parse_odom_txt(str(args.camera_odom))
        matcher = OdomMatcher(entries, max_dt=float(cfg.ros.get("offline_match_max_dt", 0.5)))
        remerged = 0
        for pose in poses:
            cam = matcher.find_closest(float(pose["timestamp"]))
            if cam is None:
                continue
            T = apply_body2optical_transform(cam.matrix, apply=apply_b2o)
            pose["camera_matrix"] = T.astype(float).tolist()
            pose["camera_frame"] = "optical"
            for key in ("camera_x", "camera_y", "camera_z", "camera_yaw"):
                pose.pop(key, None)
            remerged += 1
        print(f"Re-merged {remerged}/{len(poses)} camera poses from {args.camera_odom}")
    else:
        poses, n = normalize_poses(poses, apply_body2optical=apply_b2o)
        print(f"Normalized {n}/{len(poses)} camera poses in place")

    poses_dir = poses_path.parent
    floor_traj = args.floor_trajectory
    if floor_traj is None and (args.merge_floor or not any("action_matrix" in p for p in poses)):
        default_floor = poses_dir / cfg.floor_trajectory_filename()
        if default_floor.is_file():
            floor_traj = default_floor
    if floor_traj is not None:
        if not floor_traj.is_file():
            print(f"Error: floor trajectory not found: {floor_traj}", file=sys.stderr)
            return 1
        floor_plane = _load_floor_plane(poses_dir, cfg)
        if floor_plane is None:
            print(
                f"Error: need floor_calibration.json or derivable floor plane in {poses_dir}",
                file=sys.stderr,
            )
            return 1
        n_floor = merge_floor_fields(
            poses,
            floor_traj_path=floor_traj,
            floor_plane=floor_plane,
            max_dt=float(cfg.ros.get("offline_match_max_dt", 0.5)),
        )
        print(f"Merged floor fields for {n_floor}/{len(poses)} frames from {floor_traj}")

    out_path = poses_path if args.in_place else poses_path.with_name("poses_optical.json")
    out_path.write_text(json.dumps(poses, indent=2))
    print(
        f"Wrote {out_path} (camera_frame=optical, "
        f"body2optical={'yes' if apply_b2o else 'no'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
