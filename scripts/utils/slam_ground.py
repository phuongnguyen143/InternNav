"""Floor contact under a pitched camera (DROID OpenCV optical frame).

Mount heuristic: undo fixed mount pitch so optical +Z is horizontal, then offset
along optical +Y (down) to reach the floor contact point for SLAM floor_trajectory.
"""

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


def floor_world_from_camera_c2w(
    T_world_cam: np.ndarray,
    camera_pitch_deg: float = 30.0,
    ground_offset_y: float = 1.5,
) -> np.ndarray:
    """Floor contact world XYZ from T_world_cam (DROID OpenCV optical c2w)."""
    T = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    R_level = T[:3, :3] @ rotation_x_pitch_deg(-camera_pitch_deg)
    return T[:3, 3] + R_level @ np.array([0.0, ground_offset_y, 0.0], dtype=np.float64)


def yaw_from_leveled_camera(
    R: np.ndarray,
    mount_pitch_deg: float,
) -> float:
    """Heading from leveled optical +Z projected onto world XY."""
    R_level = np.asarray(R, dtype=np.float64) @ rotation_x_pitch_deg(-mount_pitch_deg)
    forward = R_level[:, 2]
    return float(math.atan2(forward[1], forward[0]))
