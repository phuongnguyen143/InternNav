"""Robot camera extrinsics and navigation-frame transforms."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from utils.config import get_config

_T_CAM2BASE: Optional[np.ndarray] = None


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


def poses_json_camera_to_optical(
    camera_matrix: np.ndarray,
    *,
    camera_frame: str | None = None,
    apply_body2optical: bool = True,
) -> np.ndarray:
    """Return OpenCV optical T_world_cam from a poses.json camera_matrix entry."""
    T = np.asarray(camera_matrix, dtype=np.float64).reshape(4, 4)
    if camera_frame == "optical":
        return T
    if camera_frame in (None, "body"):
        return apply_body2optical_transform(T, apply=apply_body2optical)
    return T


def build_T_cam2base() -> np.ndarray:
    """Camera-to-base extrinsics (fixed robot mount)."""
    return build_T_cam2base_for_mount()


def build_T_cam2base_for_mount(
    height_m: float | None = None,
    pitch_deg: float | None = None,
) -> np.ndarray:
    """Camera-to-base extrinsics for a mount height (m) and pitch (deg, nose down)."""
    t_base2cam = camera_to_base_translation().copy()
    if height_m is not None:
        t_base2cam[2] = float(height_m)

    pitch = (
        math.radians(float(pitch_deg))
        if pitch_deg is not None
        else camera_pitch_rad()
    )
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
