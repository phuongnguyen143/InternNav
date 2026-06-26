"""Floor contact under a pitched camera (DROID OpenCV optical frame).

Mount heuristic: camera is pitched down by ``mount_pitch_deg``. Undo that pitch
so optical +Z is horizontal, then offset ``drop_y`` along +Y (down) to reach the
floor contact point used for SLAM floor_trajectory export.
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


def leveled_camera_rotation(R: np.ndarray, mount_pitch_deg: float) -> np.ndarray:
    """Undo fixed mount pitch so optical +Z is horizontal (parallel to floor)."""
    return np.asarray(R, dtype=np.float64) @ rotation_x_pitch_deg(-mount_pitch_deg)


def floor_contact_from_pose(
    R: np.ndarray,
    t: np.ndarray,
    mount_pitch_deg: float = 30.0,
    drop_y: float = 1.5,
) -> np.ndarray:
    """World XYZ on the floor: level camera, then offset along optical +Y (down)."""
    R_level = leveled_camera_rotation(R, mount_pitch_deg)
    return np.asarray(t, dtype=np.float64) + R_level @ np.array(
        [0.0, drop_y, 0.0], dtype=np.float64
    )


def floor_world_from_camera_c2w(
    T_world_cam: np.ndarray,
    camera_pitch_deg: float = 30.0,
    ground_offset_y: float = 1.5,
) -> np.ndarray:
    """Floor contact world XYZ from T_world_cam (DROID OpenCV optical c2w)."""
    T = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    return floor_contact_from_pose(
        T[:3, :3], T[:3, 3], camera_pitch_deg, ground_offset_y
    )


def yaw_from_leveled_camera(
    R: np.ndarray,
    mount_pitch_deg: float,
) -> float:
    """Heading from leveled optical +Z projected onto world XY."""
    R_level = leveled_camera_rotation(R, mount_pitch_deg)
    forward = R_level[:, 2]
    return float(math.atan2(forward[1], forward[0]))
