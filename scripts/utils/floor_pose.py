"""
Floor estimation and camera-to-base pose projection for VLN data synthesis.

Uses local FloorEstimator.estimate_local on a scene PLY, then projects
embodiment poses onto the estimated floor plane.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from utils.config import get_config
from utils.extrinsics import (
    apply_body2optical_transform,
    build_T_cam2base,
    camera_matrix_to_base_T,
    get_T_cam2base,
)
from utils.floor_estimator import FloorEstimator
from utils.floor_plane import (
    build_floor_frame,
    floor_plane_from_world_points,
    floor_xy_to_world_on_plane,
    project_points_to_plane,
)
from utils.trajectory_io import FloorEntry, parse_floor_trajectory_txt


def load_pcd_points(pcd_path: str | Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {pcd_path}")
    return np.asarray(pcd.points, dtype=np.float64)


def estimate_floor_local(
    pcd_path: str | Path,
    voxel_size: float = 0.1,
    patch_radius: float = 1.0,
    stride: float = 0.5,
    min_patch_points: int = 200,
) -> Tuple[tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = load_pcd_points(pcd_path)
    estimator = FloorEstimator(voxel_size=voxel_size)
    (
        floor_plane,
        floor_normal,
        floor_point,
        merged_inliers,
        _scene_points,
        up_axis,
    ) = estimator.estimate_local(
        raw,
        patch_radius=patch_radius,
        stride=stride,
        min_patch_points=min_patch_points,
    )
    return floor_plane, floor_normal, floor_point, merged_inliers, up_axis


def save_floor_calibration(
    out_dir: str | Path,
    floor_plane: tuple,
    floor_normal: np.ndarray,
    floor_point: np.ndarray,
    pcd_path: Optional[str | Path] = None,
    extra: Optional[dict] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "floor_plane": list(floor_plane),
        "floor_normal": floor_normal.tolist(),
        "floor_point": floor_point.tolist(),
        "pcd_path": str(pcd_path) if pcd_path else None,
    }
    if extra:
        payload.update(extra)
    out_path = out_dir / get_config().floor_calibration_filename()
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def load_floor_calibration(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    data["floor_plane"] = tuple(data["floor_plane"])
    data["floor_normal"] = np.array(data["floor_normal"], dtype=np.float64)
    data["floor_point"] = np.array(data["floor_point"], dtype=np.float64)
    return data


def floor_world_points_from_entries(entries: list[FloorEntry]) -> np.ndarray:
    """Collect non-zero world_x/y/z from floor trajectory entries."""
    pts = []
    for entry in entries:
        if abs(entry.world_x) + abs(entry.world_y) + abs(entry.world_z) > 1e-9:
            pts.append([entry.world_x, entry.world_y, entry.world_z])
    if not pts:
        raise ValueError("floor_trajectory has no world_x/y/z columns; re-export with project_slam_path")
    return np.asarray(pts, dtype=np.float64)


def derive_floor_plane_from_trajectory(
    floor_trajectory_path: str | Path,
) -> tuple[tuple, np.ndarray, np.ndarray, list[FloorEntry]]:
    """Estimate floor plane from floor_trajectory.txt world coordinates (no PCD)."""
    entries = parse_floor_trajectory_txt(floor_trajectory_path)
    world_pts = floor_world_points_from_entries(entries)
    floor_plane = floor_plane_from_world_points(world_pts)
    a, b, c, d = floor_plane
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    floor_point = project_points_to_plane(world_pts.mean(axis=0).reshape(1, 3), floor_plane)[0]
    return floor_plane, normal, floor_point, entries


def derive_floor_calibration_from_trajectory(
    floor_trajectory_path: str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write floor_calibration.json by fitting a plane to trajectory world_x/y/z."""
    out_dir = Path(out_dir)
    out_path = out_dir / get_config().floor_calibration_filename()
    if out_path.exists() and not overwrite:
        return out_path

    floor_plane, normal, floor_point, entries = derive_floor_plane_from_trajectory(
        floor_trajectory_path
    )
    world_pts = floor_world_points_from_entries(entries)
    return save_floor_calibration(
        out_dir,
        floor_plane,
        normal,
        floor_point,
        pcd_path=None,
        extra={
            "source": "floor_trajectory",
            "floor_trajectory_path": str(Path(floor_trajectory_path).resolve()),
            "num_trajectory_points": len(entries),
            "num_world_points": int(len(world_pts)),
        },
    )


def resolve_floor_plane_for_keyframe_root(
    keyframe_root: str | Path,
    *,
    floor_calibration: Optional[str | Path] = None,
    floor_trajectory: Optional[str | Path] = None,
    derive_if_missing: bool = True,
) -> Optional[tuple]:
    """Load floor plane from calibration, or derive from floor_trajectory.txt."""
    keyframe_root = Path(keyframe_root)
    cal_path = Path(floor_calibration) if floor_calibration else keyframe_root / get_config().floor_calibration_filename()
    if cal_path.is_file():
        return load_floor_calibration(cal_path)["floor_plane"]

    if not derive_if_missing:
        return None

    traj_path = Path(floor_trajectory) if floor_trajectory else keyframe_root / get_config().floor_trajectory_filename()
    if not traj_path.is_file():
        return None

    derived_path = derive_floor_calibration_from_trajectory(traj_path, keyframe_root)
    print(f"Derived floor calibration from {traj_path} -> {derived_path}")
    return load_floor_calibration(derived_path)["floor_plane"]


def enrich_poses_from_floor_trajectory(
    poses_by_frame_idx: dict[int, dict],
    floor_trajectory_path: str | Path,
    *,
    max_dt: float = 0.5,
) -> int:
    """Attach world_x/y/z from floor_trajectory.txt to poses (timestamp match)."""
    from utils.trajectory_io import FloorMatcher

    entries = parse_floor_trajectory_txt(floor_trajectory_path)
    matcher = FloorMatcher(entries, max_dt=max_dt)
    updated = 0
    for pose in poses_by_frame_idx.values():
        entry = matcher.find_closest(float(pose.get("timestamp", 0.0)))
        if entry is None:
            continue
        if abs(entry.world_x) + abs(entry.world_y) + abs(entry.world_z) < 1e-9:
            continue
        pose["world_x"] = float(entry.world_x)
        pose["world_y"] = float(entry.world_y)
        pose["world_z"] = float(entry.world_z)
        if "x" not in pose or pose.get("pose_frame") == "floor":
            pose["x"] = float(entry.x)
            pose["y"] = float(entry.y)
            pose["yaw"] = float(entry.yaw)
            pose["z"] = float(entry.z)
        updated += 1
    return updated


def yaw_on_floor_plane(R_world: np.ndarray, floor_plane: tuple) -> float:
    """Heading angle in floor 2D frame from base +X axis projected onto the plane."""
    _, x_ax, y_ax, n = build_floor_frame(floor_plane)
    forward = R_world[:3, 0].astype(np.float64)
    forward_proj = forward - np.dot(forward, n) * n
    norm = np.linalg.norm(forward_proj)
    if norm < 1e-9:
        forward_proj = x_ax
    else:
        forward_proj /= norm
    return float(math.atan2(np.dot(forward_proj, y_ax), np.dot(forward_proj, x_ax)))


def pose_floor_world_xyz(pose: dict, floor_plane: Optional[tuple] = None) -> np.ndarray:
    if all(k in pose for k in ("world_x", "world_y", "world_z")):
        return np.array(
            [float(pose["world_x"]), float(pose["world_y"]), float(pose["world_z"])],
            dtype=np.float64,
        )
    if floor_plane is None:
        raise ValueError("floor_plane required when pose lacks world_x/y/z")
    return floor_xy_to_world_on_plane(float(pose["x"]), float(pose["y"]), floor_plane)


def project_base_pose_to_floor(
    T_world_base: np.ndarray,
    floor_plane: tuple,
) -> Tuple[float, float, float, float, np.ndarray]:
    T_world_base = np.asarray(T_world_base, dtype=np.float64).reshape(4, 4)
    pos = T_world_base[:3, 3]
    pos_proj = project_points_to_plane(pos.reshape(1, 3), floor_plane)[0]

    origin, x_ax, y_ax, _ = build_floor_frame(floor_plane)
    delta = pos_proj - origin
    x_floor = float(np.dot(delta, x_ax))
    y_floor = float(np.dot(delta, y_ax))
    yaw = yaw_on_floor_plane(T_world_base[:3, :3], floor_plane)

    return x_floor, y_floor, yaw, float(pos_proj[2]), pos_proj.copy()


def base_pose_to_action_matrix(
    T_world_base: np.ndarray,
    floor_plane: tuple,
) -> np.ndarray:
    T_world_base = np.asarray(T_world_base, dtype=np.float64).reshape(4, 4)
    pos_proj = project_points_to_plane(T_world_base[:3, 3].reshape(1, 3), floor_plane)[0]
    yaw = yaw_on_floor_plane(T_world_base[:3, :3], floor_plane)

    _, x_ax, y_ax, n = build_floor_frame(floor_plane)
    if n[2] < 0:
        n = -n
        x_ax = -x_ax
        y_ax = -y_ax

    forward = math.cos(yaw) * x_ax + math.sin(yaw) * y_ax
    y_dir = np.cross(n, forward)
    y_dir /= np.linalg.norm(y_dir) + 1e-12

    R = np.column_stack([forward, y_dir, n])

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = pos_proj.astype(np.float32)
    return T


def camera_matrix_to_floor_pose(
    T_world_cam: np.ndarray,
    floor_plane: tuple,
    apply_body2optical: bool = True,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    T_world_cam = apply_body2optical_transform(T_world_cam, apply=apply_body2optical)
    T_base = camera_matrix_to_base_T(T_world_cam)
    x, y, yaw, legacy_z, world_xyz = project_base_pose_to_floor(T_base, floor_plane)
    action = base_pose_to_action_matrix(T_base, floor_plane)
    return x, y, yaw, legacy_z, action, world_xyz


def yaw_from_forward_on_floor_plane(forward: np.ndarray, floor_plane: tuple) -> float:
    """Heading in floor 2D from a world forward vector projected onto the plane."""
    _, x_ax, y_ax, n = build_floor_frame(floor_plane)
    forward_proj = np.asarray(forward, dtype=np.float64).reshape(3) - np.dot(forward, n) * n
    norm = np.linalg.norm(forward_proj)
    if norm < 1e-9:
        forward_proj = x_ax
    else:
        forward_proj /= norm
    return float(math.atan2(np.dot(forward_proj, y_ax), np.dot(forward_proj, x_ax)))


def camera_c2w_to_floor_pose(
    T_world_cam: np.ndarray,
    floor_plane: tuple,
    *,
    camera_pitch_deg: float = 30.0,
    ground_offset_y: float = 1.5,
) -> Tuple[float, float, float, float, np.ndarray]:
    """SLAM export: leveled-camera floor contact → floor (x, y, yaw) + world xyz."""
    from utils.slam_ground import floor_world_from_camera_c2w, leveled_camera_rotation

    T = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    world_proj = project_points_to_plane(
        floor_world_from_camera_c2w(T, camera_pitch_deg, ground_offset_y).reshape(1, 3),
        floor_plane,
    )[0]
    R_level = leveled_camera_rotation(T[:3, :3], camera_pitch_deg)
    yaw = yaw_from_forward_on_floor_plane(R_level[:, 2], floor_plane)

    origin, x_ax, y_ax, _ = build_floor_frame(floor_plane)
    delta = world_proj - origin
    return (
        float(np.dot(delta, x_ax)),
        float(np.dot(delta, y_ax)),
        yaw,
        float(world_proj[2]),
        world_proj.copy(),
    )


def floor_2d_pose_to_action_matrix(
    x: float,
    y: float,
    yaw: float,
    z: float,
    floor_plane: tuple,
) -> np.ndarray:
    _, x_ax, y_ax, n = build_floor_frame(floor_plane)
    if n[2] < 0:
        n = -n
        x_ax = -x_ax
        y_ax = -y_ax

    pos_world = floor_xy_to_world_on_plane(x, y, floor_plane)
    forward = math.cos(yaw) * x_ax + math.sin(yaw) * y_ax
    y_dir = np.cross(n, forward)
    y_dir /= np.linalg.norm(y_dir) + 1e-12
    R = np.column_stack([forward, y_dir, n])

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = pos_world.astype(np.float32)
    return T


def floor_xyyaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    q = Rotation.from_euler("z", yaw).as_quat()
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def sync_floor_entry_world_xyz(
    entries: list[FloorEntry],
    floor_plane: tuple,
) -> list[FloorEntry]:
    """Recompute world_x/y/z from floor (x, y) after smoothing."""
    synced: list[FloorEntry] = []
    for entry in entries:
        world = floor_xy_to_world_on_plane(entry.x, entry.y, floor_plane)
        synced.append(
            FloorEntry(
                timestamp=entry.timestamp,
                x=entry.x,
                y=entry.y,
                yaw=entry.yaw,
                z=float(world[2]),
                world_x=float(world[0]),
                world_y=float(world[1]),
                world_z=float(world[2]),
            )
        )
    return synced


def init_floor_from_pcd(
    pcd_path: str | Path,
    calibration_dir: Optional[str | Path] = None,
) -> dict:
    pcd_path = Path(pcd_path)
    print(f"[floor] Estimating floor from {pcd_path} ...")
    floor_plane, floor_normal, floor_point, inliers, up_axis = estimate_floor_local(pcd_path)
    print(f"[floor] plane={floor_plane} | inliers={len(inliers):,} | normal={floor_normal}")

    cal_path = None
    if calibration_dir is not None:
        cal_path = save_floor_calibration(
            calibration_dir,
            floor_plane,
            floor_normal,
            floor_point,
            pcd_path=pcd_path,
            extra={"num_inliers": int(len(inliers)), "up_axis": up_axis.tolist()},
        )
        print(f"[floor] Saved calibration: {cal_path}")

    return {
        "floor_plane": floor_plane,
        "floor_normal": floor_normal,
        "floor_point": floor_point,
        "calibration_path": cal_path,
    }


def __getattr__(name: str):
    if name == "FLOOR_CALIBRATION_FILENAME":
        return get_config().floor_calibration_filename()
    if name == "build_T_cam2base":
        return build_T_cam2base
    if name == "get_T_cam2base":
        return get_T_cam2base
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
