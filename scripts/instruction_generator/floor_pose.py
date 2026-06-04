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

from constants import (
    CAMERA_PITCH_RAD,
    CAMERA_TO_BASE_TRANSLATION,
    FLOOR_CALIBRATION_FILENAME,
)
from floor_estimator import FloorEstimator


def project_points_to_plane(points: np.ndarray, plane: tuple) -> np.ndarray:
    """Orthographic projection onto plane ax+by+cz+d=0."""
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n_norm_sq = np.dot(n, n)
    pts = np.atleast_2d(points).astype(np.float64)
    dist = (pts @ n + d) / n_norm_sq
    return pts - dist[:, None] * n


def build_floor_frame(plane: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (origin, x_ax, y_ax, normal) for the floor plane."""
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    origin = -d / (np.dot(n, n) + 1e-12) * n

    arb = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_ax = np.cross(arb, n)
    x_ax /= np.linalg.norm(x_ax) + 1e-12
    y_ax = np.cross(n, x_ax)
    return origin, x_ax, y_ax, n


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
    """
    Run FloorEstimator.estimate_local on a scene PLY.

    Returns:
        floor_plane (a,b,c,d), floor_normal, floor_point, merged_inliers, up_axis
    """
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
    out_path = out_dir / FLOOR_CALIBRATION_FILENAME
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def load_floor_calibration(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    data["floor_plane"] = tuple(data["floor_plane"])
    data["floor_normal"] = np.array(data["floor_normal"], dtype=np.float64)
    data["floor_point"] = np.array(data["floor_point"], dtype=np.float64)
    return data


def build_T_cam2base() -> np.ndarray:
    """Camera-to-base extrinsics (fixed robot mount)."""
    t_base2cam = np.array(CAMERA_TO_BASE_TRANSLATION, dtype=np.float64)
    c, s = math.cos(CAMERA_PITCH_RAD), math.sin(CAMERA_PITCH_RAD)
    R_base2cam = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    R_cam2base = R_base2cam.T
    t_cam2base = -R_cam2base @ t_base2cam

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base
    return T


_T_CAM2BASE: Optional[np.ndarray] = None


def get_T_cam2base() -> np.ndarray:
    global _T_CAM2BASE
    if _T_CAM2BASE is None:
        _T_CAM2BASE = build_T_cam2base()
    return _T_CAM2BASE


def camera_matrix_to_base_T(T_world_cam: np.ndarray) -> np.ndarray:
    """T_world_base = T_world_cam @ T_cam2base."""
    T_world_cam = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    return T_world_cam @ get_T_cam2base()


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


def project_base_pose_to_floor(
    T_world_base: np.ndarray,
    floor_plane: tuple,
) -> Tuple[float, float, float, float]:
    """
    Project base pose onto floor plane.

    Returns:
        x, y, yaw in floor 2D frame, z (projected height along world Z of base origin)
    """
    T_world_base = np.asarray(T_world_base, dtype=np.float64).reshape(4, 4)
    pos = T_world_base[:3, 3]
    pos_proj = project_points_to_plane(pos.reshape(1, 3), floor_plane)[0]

    origin, x_ax, y_ax, _ = build_floor_frame(floor_plane)
    delta = pos_proj - origin
    x_floor = float(np.dot(delta, x_ax))
    y_floor = float(np.dot(delta, y_ax))
    yaw = yaw_on_floor_plane(T_world_base[:3, :3], floor_plane)

    return x_floor, y_floor, yaw, float(pos_proj[2])


def base_pose_to_action_matrix(
    T_world_base: np.ndarray,
    floor_plane: tuple,
) -> np.ndarray:
    """
    Build 4x4 world pose for NavDP action: translation on floor, yaw aligned to plane.
    """
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
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Full pipeline: camera odom -> base -> floor projection.

    Returns:
        x, y, yaw, z, action_matrix (4x4 float32)
    """
    T_base = camera_matrix_to_base_T(T_world_cam)
    x, y, yaw, z = project_base_pose_to_floor(T_base, floor_plane)
    action = base_pose_to_action_matrix(T_base, floor_plane)
    return x, y, yaw, z, action


def floor_2d_pose_to_action_matrix(
    x: float,
    y: float,
    yaw: float,
    z: float,
    floor_plane: tuple,
) -> np.ndarray:
    """Build world-frame 4x4 action from floor-frame (x, y, yaw, z)."""
    origin, x_ax, y_ax, n = build_floor_frame(floor_plane)
    if n[2] < 0:
        n = -n
        x_ax = -x_ax
        y_ax = -y_ax

    pos_world = origin + x * x_ax + y * y_ax + z * n
    forward = math.cos(yaw) * x_ax + math.sin(yaw) * y_ax
    y_dir = np.cross(n, forward)
    y_dir /= np.linalg.norm(y_dir) + 1e-12
    R = np.column_stack([forward, y_dir, n])

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = pos_world.astype(np.float32)
    return T


def floor_xyyaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion (x,y,z,w) for rotation about world Z from floor-frame yaw."""
    q = Rotation.from_euler("z", yaw).as_quat()
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def init_floor_from_pcd(
    pcd_path: str | Path,
    calibration_dir: Optional[str | Path] = None,
) -> dict:
    """
    Estimate floor and optionally persist calibration JSON.

    Returns dict with keys: floor_plane, floor_normal, floor_point, calibration_path
    """
    pcd_path = Path(pcd_path)
    print(f"[floor] Estimating floor from {pcd_path} ...")
    floor_plane, floor_normal, floor_point, inliers, up_axis = estimate_floor_local(
        pcd_path
    )
    print(
        f"[floor] plane={floor_plane} | inliers={len(inliers):,} | "
        f"normal={floor_normal}"
    )

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
