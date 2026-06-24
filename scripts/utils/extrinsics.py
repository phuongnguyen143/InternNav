"""Robot camera extrinsics and navigation-frame transforms."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from utils.config import get_config

_T_CAM2BASE: Optional[np.ndarray] = None
_T_OPTICAL_TO_NAV: Optional[np.ndarray] = None
_R_ROS_TO_HABITAT_BODY: Optional[tuple] = None


def camera_to_base_translation() -> np.ndarray:
    return np.array(get_config().robot.get("camera_to_base_translation", [0.1067, 0.0, 0.77566]), dtype=np.float64)


def camera_pitch_rad() -> float:
    return float(get_config().robot.get("camera_pitch_rad", 0.0))


def r_body2optical_matrix() -> np.ndarray:
    rows = get_config().robot.get(
        "r_body2optical",
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    return np.array(rows, dtype=np.float64)


def r_body2optical_4x4() -> np.ndarray:
    return r_body2optical_matrix().reshape(4, 4)


def apply_body2optical_transform(
    T_world_cam: np.ndarray,
    apply: bool = True,
) -> np.ndarray:
    """Map T_world_cam from ROS body frame to OpenCV optical (Z fwd, Y down).

    SLAM odometry (e.g. DROID-W) is already optical; set apply=False.
    LiDAR / legacy odom txt needs apply=True.
    """
    T = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    if not apply:
        return T
    return T @ r_body2optical_matrix().T


def _mount_rotation(pitch_deg: float) -> np.ndarray:
    pitch_rad = math.radians(pitch_deg)
    c, s = math.cos(pitch_rad), math.sin(pitch_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def make_ros_to_habitat_body() -> tuple:
    ref_R = np.array(
        [[0.0, -0.5, 0.866], [-1.0, 0.0, 0.0], [0.0, -0.866, -0.5]],
        dtype=np.float64,
    )
    R_mount = _mount_rotation(30.0)
    R_fix = ref_R @ R_mount.T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_fix
    return tuple(tuple(float(v) for v in row) for row in T.tolist())


def r_ros_to_habitat_body() -> tuple:
    global _R_ROS_TO_HABITAT_BODY
    if _R_ROS_TO_HABITAT_BODY is None:
        _R_ROS_TO_HABITAT_BODY = make_ros_to_habitat_body()
    return _R_ROS_TO_HABITAT_BODY


def build_T_cam2base() -> np.ndarray:
    """Camera-to-base extrinsics (fixed robot mount)."""
    t_base2cam = camera_to_base_translation()
    pitch = camera_pitch_rad()
    c, s = math.cos(pitch), math.sin(pitch)
    R_base2cam = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    R_cam2base = R_base2cam.T
    t_cam2base = -R_cam2base @ t_base2cam

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base
    return T


def get_T_cam2base() -> np.ndarray:
    global _T_CAM2BASE
    if _T_CAM2BASE is None:
        _T_CAM2BASE = build_T_cam2base()
    return _T_CAM2BASE


def camera_matrix_to_base_T(T_world_cam: np.ndarray) -> np.ndarray:
    T_world_cam = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    return T_world_cam @ get_T_cam2base()


def build_T_optical_to_nav() -> np.ndarray:
    """Optical camera frame (Z fwd, Y down) -> navigation base (X fwd, Y left, Z up)."""
    R_body2optical = r_body2optical_matrix()[:3, :3]
    t_base2cam = camera_to_base_translation()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_body2optical
    T[:3, 3] = -R_body2optical.T @ t_base2cam
    return T


def get_T_optical_to_nav() -> np.ndarray:
    global _T_OPTICAL_TO_NAV
    if _T_OPTICAL_TO_NAV is None:
        _T_OPTICAL_TO_NAV = build_T_optical_to_nav()
    return _T_OPTICAL_TO_NAV


def nav_pose_from_camera(T_world_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_world_cam = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    T_world_nav = T_world_cam @ get_T_optical_to_nav()
    return T_world_nav[:3, :3], T_world_nav[:3, 3]
