"""Ground contact point under the camera (SLAM / DROID optical frame)."""

from __future__ import annotations

import math

import numpy as np


def rotation_x_pitch_deg(deg: float) -> np.ndarray:
    """OpenCV optical frame: positive pitch rotates Y toward Z (look down)."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def ground_point_from_pose(
    R: np.ndarray,
    t: np.ndarray,
    pitch_deg: float,
    offset_y: float,
) -> np.ndarray:
    """World point on the floor below the camera (pitch down + offset along +Y)."""
    R_pitch = R @ rotation_x_pitch_deg(pitch_deg)
    return np.asarray(t, dtype=np.float64) + R_pitch @ np.array([0.0, offset_y, 0.0], dtype=np.float64)


def ground_world_from_camera_c2w(
    T_world_cam: np.ndarray,
    camera_pitch_deg: float = 30.0,
    ground_offset_y: float = 1.5,
) -> np.ndarray:
    """Ground contact world XYZ from T_world_cam (DROID OpenCV optical c2w)."""
    T = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    return ground_point_from_pose(T[:3, :3], T[:3, 3], camera_pitch_deg, ground_offset_y)
