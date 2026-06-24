"""Label computation for InternVLA-N1 System2 LeRobot format (GdvgFV5R1Z5-style)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from utils.extrinsics import apply_body2optical_transform, r_ros_to_habitat_body
from utils.floor_pose import pose_floor_world_xyz
from utils.slam_ground import ground_world_from_camera_c2w

_R_ROS_TO_HABITAT_BODY = np.array(r_ros_to_habitat_body(), dtype=np.float64)

CAMERA_SETTINGS: List[Tuple[int, int]] = [
    (125, 0),
    (125, 30),
    (125, 45),
    (60, 15),
    (60, 30),
]

STEP_SIZE_M = 0.25
TURN_ANGLE_DEG = 15.0
# Habitat VLN discrete steps: 0.25 m forward, 15 deg turn (InternVLA-N1 paper).
# Use slightly lower thresholds so quantized steps are detected reliably.
TURN_THRESH_RAD = math.radians(12.0)
FORWARD_THRESH_M = 0.20
# Fixed frame lookahead: project floor path point at frame (i + lookahead).
DEFAULT_LOOKAHEAD_FRAMES = 200
INVALID_GOAL = np.array([-1, -1], dtype=np.int32)


def setting_key(height_cm: int, pitch_deg: int) -> str:
    return f"{height_cm}cm_{pitch_deg}deg"


def pose_xyyaw_to_matrix(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    rot = Rotation.from_euler("z", yaw).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[0, 3] = float(x)
    T[1, 3] = float(y)
    T[2, 3] = float(z)
    return T


def _mount_transform(height_m: float, pitch_deg: float) -> np.ndarray:
    """Camera mount relative to robot base (Habitat-style pitch down around X)."""
    pitch_rad = math.radians(pitch_deg)
    c, s = math.cos(pitch_rad), math.sin(pitch_rad)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[2, 3] = height_m
    return T


def build_camera_extrinsic(
    T_world_base: np.ndarray,
    height_cm: int,
    pitch_deg: int,
) -> np.ndarray:
    """World camera extrinsic (T_camera_world) for a (height, pitch) setting."""
    T_world_base = np.asarray(T_world_base, dtype=np.float64).reshape(4, 4)
    T_mount = _mount_transform(height_cm / 100.0, pitch_deg)
    # ROS SLAM action_matrix uses X-fwd/Y-left/Z-up; InternData-N1 expects Habitat base axes.
    return (T_world_base @ _R_ROS_TO_HABITAT_BODY @ T_mount).astype(np.float32)


def get_T_world_base_from_pose(
    pose: Dict,
    floor_plane: Optional[tuple],
    floor_2d_pose_to_action_matrix_fn,
) -> np.ndarray:
    if "action_matrix" in pose:
        return np.array(pose["action_matrix"], dtype=np.float32).reshape(4, 4)

    x = float(pose.get("x", 0.0))
    y = float(pose.get("y", 0.0))
    yaw = float(pose.get("yaw", 0.0))
    z = float(pose.get("z", 0.0))
    pose_frame = pose.get("pose_frame", "floor")

    if pose_frame == "floor" and floor_plane is not None:
        return floor_2d_pose_to_action_matrix_fn(x, y, yaw, z, floor_plane)

    if pose_frame == "floor":
        z = 0.0
    return pose_xyyaw_to_matrix(x, y, yaw, z=z)


def extract_floor_xyyaw(
    poses: List[Dict],
    floor_plane: Optional[tuple],
    floor_2d_pose_to_action_matrix_fn,
) -> np.ndarray:
    """Return (N, 3) floor-frame x, y, yaw per frame for actions and waypoints."""
    xyyaw = np.zeros((len(poses), 3), dtype=np.float64)
    for i, pose in enumerate(poses):
        if pose.get("pose_frame") == "floor" and "x" in pose and "yaw" in pose:
            xyyaw[i, 0] = float(pose["x"])
            xyyaw[i, 1] = float(pose["y"])
            xyyaw[i, 2] = float(pose["yaw"])
            continue
        T = get_T_world_base_from_pose(
            pose, floor_plane, floor_2d_pose_to_action_matrix_fn
        )
        xyyaw[i, 0] = T[0, 3]
        xyyaw[i, 1] = T[1, 3]
        xyyaw[i, 2] = math.atan2(T[1, 0], T[0, 0])
    return xyyaw


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def select_waypoint_indices(
    xyyaw: np.ndarray, step_size: float = STEP_SIZE_M
) -> List[int]:
    """Subsample trajectory indices at ~step_size spacing."""
    if len(xyyaw) == 0:
        return []
    waypoints = [0]
    last_xy = xyyaw[0, :2]
    for i in range(1, len(xyyaw)):
        dist = np.linalg.norm(xyyaw[i, :2] - last_xy)
        if dist >= step_size:
            waypoints.append(i)
            last_xy = xyyaw[i, :2]
    if waypoints[-1] != len(xyyaw) - 1:
        waypoints.append(len(xyyaw) - 1)
    return waypoints


def _robot_frame_delta(
    prev_xyyaw: np.ndarray,
    curr_xyyaw: np.ndarray,
) -> Tuple[float, float, float]:
    """Return (forward_m, lateral_m, dyaw_rad) in the robot frame at prev."""
    dyaw = _normalize_angle(curr_xyyaw[2] - prev_xyyaw[2])
    delta = curr_xyyaw[:2] - prev_xyyaw[:2]
    forward = delta[0] * math.cos(prev_xyyaw[2]) + delta[1] * math.sin(prev_xyyaw[2])
    lateral = -delta[0] * math.sin(prev_xyyaw[2]) + delta[1] * math.cos(prev_xyyaw[2])
    return forward, lateral, dyaw


def _discrete_action_between(
    prev_xyyaw: np.ndarray,
    curr_xyyaw: np.ndarray,
    turn_thresh_rad: float = TURN_THRESH_RAD,
    forward_thresh: float = FORWARD_THRESH_M,
) -> Optional[int]:
    """
    Classify one Habitat VLN discrete step between consecutive poses.

    Encoding: 1=MOVE_FORWARD (0.25 m), 2=TURN_LEFT (15 deg), 3=TURN_RIGHT (15 deg).
    In InternData-N1 / Habitat poses, forward motion appears along the robot
    lateral (+Y) axis; turns are +/-15 deg yaw changes.
    """
    forward, lateral, dyaw = _robot_frame_delta(prev_xyyaw, curr_xyyaw)

    if abs(dyaw) >= turn_thresh_rad:
        return 2 if dyaw > 0 else 3
    if abs(lateral) >= forward_thresh:
        return 1
    if abs(forward) >= forward_thresh:
        return 1
    return None


def compute_discrete_actions(xyyaw: np.ndarray) -> np.ndarray:
    """Per-frame discrete actions: -1 at frame 0, then 1/2/3."""
    n = len(xyyaw)
    actions = np.ones(n, dtype=np.int32)
    if n == 0:
        return actions
    actions[0] = -1
    for i in range(1, n):
        step_action = _discrete_action_between(xyyaw[i - 1], xyyaw[i])
        if step_action is None:
            actions[i] = actions[i - 1] if actions[i - 1] != -1 else 1
        else:
            actions[i] = step_action
    return actions


def camera_matrix_to_optical(
    camera_matrix: np.ndarray,
    apply_body2optical: bool = True,
) -> np.ndarray:
    """Convert raw odom T_world_cam to optical frame (image_projector.py)."""
    return apply_body2optical_transform(camera_matrix, apply=apply_body2optical)


def project_world_point_with_camera(
    camera_matrix: np.ndarray,
    target_world: np.ndarray,
    camera_intrinsic: np.ndarray,
    img_w: int,
    img_h: int,
    apply_body2optical: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    Project a world-frame 3D point into the current camera image.

    Matches GaussTrace/image_projector.py world_to_pixels:
    T_cam_world = inv(T_world_cam_optical), optical frame Z-forward, u = fx*X/Z, v = fy*Y/Z.
    """
    T_world_cam = camera_matrix_to_optical(camera_matrix, apply_body2optical=apply_body2optical)
    T_cam_world = np.linalg.inv(T_world_cam)

    target_homo = np.asarray([*target_world, 1.0], dtype=np.float64)
    p_cam = (T_cam_world @ target_homo)[:3]

    if p_cam[2] <= 0.01:
        return INVALID_GOAL.copy(), False

    K = np.asarray(camera_intrinsic, dtype=np.float64).reshape(3, 3)
    u = K[0, 0] * (p_cam[0] / p_cam[2]) + K[0, 2]
    v = K[1, 1] * (p_cam[1] / p_cam[2]) + K[1, 2]
    if 0 <= u < img_w and 0 <= v < img_h:
        return np.array([int(round(u)), int(round(v))], dtype=np.int32), True
    return INVALID_GOAL.copy(), False


def goal_target_world_xyz(
    pose_w: Dict,
    floor_plane: Optional[tuple],
    *,
    camera_pitch_deg: float = 30.0,
    ground_offset_y: float = 1.5,
    use_ground_contact: bool = True,
    apply_body2optical: bool = True,
) -> np.ndarray:
    """3D goal point at lookahead frame: ground contact (SLAM) or floor-plane xy."""
    if use_ground_contact and "camera_matrix" in pose_w:
        T_world_cam = np.array(pose_w["camera_matrix"], dtype=np.float64)
        if apply_body2optical:
            T_world_cam = camera_matrix_to_optical(T_world_cam, apply_body2optical=True)
        return ground_world_from_camera_c2w(
            T_world_cam,
            camera_pitch_deg=camera_pitch_deg,
            ground_offset_y=ground_offset_y,
        )
    return pose_floor_world_xyz(pose_w, floor_plane)


def resolve_goal_use_ground_contact(
    apply_body2optical: bool,
    scene_goal_use_ground_contact: Optional[bool] = None,
) -> bool:
    """
    SLAM/DROID optical odom: ground contact under pitched camera (matches path viz).
    LiDAR body odom: floor-trajectory world_x/y/z on the estimated floor plane.
    """
    if scene_goal_use_ground_contact is not None:
        return scene_goal_use_ground_contact
    return not apply_body2optical


def _project_goal_at_offset(
    poses: List[Dict],
    i: int,
    offset: int,
    floor_plane: Optional[tuple],
    camera_intrinsic: np.ndarray,
    img_w: int,
    img_h: int,
    *,
    apply_body2optical: bool,
    camera_pitch_deg: float,
    ground_offset_y: float,
    use_ground_contact: bool,
) -> Tuple[np.ndarray, bool]:
    """Project a lookahead 3D target from frame i+offset into image i."""
    w = i + offset
    if (
        w >= len(poses)
        or "camera_matrix" not in poses[i]
        or poses[i].get("pose_frame") != "floor"
        or ("camera_matrix" not in poses[w] and "x" not in poses[w])
    ):
        return INVALID_GOAL.copy(), False

    cam_i = np.array(poses[i]["camera_matrix"], dtype=np.float64)
    pose_w = poses[w]

    target_modes = [use_ground_contact]
    if not use_ground_contact and apply_body2optical and "camera_matrix" in pose_w:
        target_modes.append(True)

    for ground_mode in target_modes:
        target_world = goal_target_world_xyz(
            pose_w,
            floor_plane,
            camera_pitch_deg=camera_pitch_deg,
            ground_offset_y=ground_offset_y,
            use_ground_contact=ground_mode,
            apply_body2optical=apply_body2optical,
        )
        goal, visible = project_world_point_with_camera(
            cam_i,
            target_world,
            camera_intrinsic,
            img_w,
            img_h,
            apply_body2optical=apply_body2optical,
        )
        if visible:
            return goal, True

    return INVALID_GOAL.copy(), False


def compute_goals_for_setting(
    poses: List[Dict],
    floor_plane: Optional[tuple],
    camera_intrinsic: np.ndarray,
    img_w: int,
    img_h: int,
    primary: bool,
    lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
    goal_lookahead_adaptive: bool = True,
    apply_body2optical: bool = True,
    camera_pitch_deg: float = 30.0,
    ground_offset_y: float = 1.5,
    use_ground_contact: bool = True,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Compute goal and relative_goal_frame_id per frame for one camera setting.

    Projects a ground contact point from a future frame into image i via
    poses[i].camera_matrix (image_projector.py pinhole convention).

    When goal_lookahead_adaptive is True (default), tries offsets from
    min(lookahead_frames, n - 1 - i) down to 1 and uses the farthest
    visible projection. This keeps goals valid near episode ends and when
    the exact lookahead point is off-screen.
    """
    n = len(poses)
    goals: List[np.ndarray] = []
    rel_ids: List[int] = []
    lookahead_frames = max(int(lookahead_frames), 1)

    if not primary:
        for _ in range(n):
            goals.append(INVALID_GOAL.copy())
            rel_ids.append(-1)
        return goals, rel_ids

    if floor_plane is None:
        for _ in range(n):
            goals.append(INVALID_GOAL.copy())
            rel_ids.append(-1)
        return goals, rel_ids

    for i in range(n):
        max_offset = min(lookahead_frames, n - 1 - i)
        if max_offset < 1:
            goals.append(INVALID_GOAL.copy())
            rel_ids.append(-1)
            continue

        if goal_lookahead_adaptive:
            offsets = range(max_offset, 0, -1)
        else:
            offsets = (lookahead_frames,) if i + lookahead_frames < n else ()

        found = False
        for offset in offsets:
            goal, visible = _project_goal_at_offset(
                poses,
                i,
                offset,
                floor_plane,
                camera_intrinsic,
                img_w,
                img_h,
                apply_body2optical=apply_body2optical,
                camera_pitch_deg=camera_pitch_deg,
                ground_offset_y=ground_offset_y,
                use_ground_contact=use_ground_contact,
            )
            if visible:
                goals.append(goal)
                rel_ids.append(offset)
                found = True
                break

        if not found:
            goals.append(INVALID_GOAL.copy())
            rel_ids.append(-1)

    return goals, rel_ids


def build_frame_labels(
    poses: List[Dict],
    floor_plane: Optional[tuple],
    floor_2d_pose_to_action_matrix_fn,
    camera_intrinsic: np.ndarray,
    img_w: int,
    img_h: int,
    goal_setting: Tuple[int, int] = (125, 30),
    goal_lookahead_frames: int = DEFAULT_LOOKAHEAD_FRAMES,
    goal_lookahead_adaptive: bool = True,
    apply_body2optical: bool = True,
    camera_pitch_deg: float = 30.0,
    ground_offset_y: float = 1.5,
    use_ground_contact: bool = True,
) -> List[Dict]:
    """Build parquet frame dicts for all camera settings."""
    xyyaw = extract_floor_xyyaw(poses, floor_plane, floor_2d_pose_to_action_matrix_fn)
    actions = compute_discrete_actions(xyyaw)

    extrinsics_by_setting: Dict[str, List[np.ndarray]] = {}
    for height_cm, pitch_deg in CAMERA_SETTINGS:
        sk = setting_key(height_cm, pitch_deg)
        extrinsics_by_setting[sk] = []
        for pose in poses:
            T_base = get_T_world_base_from_pose(
                pose, floor_plane, floor_2d_pose_to_action_matrix_fn
            )
            extrinsics_by_setting[sk].append(
                build_camera_extrinsic(T_base, height_cm, pitch_deg)
            )

    goals_by_setting: Dict[str, List[np.ndarray]] = {}
    rel_by_setting: Dict[str, List[int]] = {}
    goal_sk = setting_key(*goal_setting)
    for height_cm, pitch_deg in CAMERA_SETTINGS:
        sk = setting_key(height_cm, pitch_deg)
        goals, rel_ids = compute_goals_for_setting(
            poses,
            floor_plane,
            camera_intrinsic,
            img_w,
            img_h,
            primary=(sk == goal_sk),
            lookahead_frames=goal_lookahead_frames,
            goal_lookahead_adaptive=goal_lookahead_adaptive,
            apply_body2optical=apply_body2optical,
            camera_pitch_deg=camera_pitch_deg,
            ground_offset_y=ground_offset_y,
            use_ground_contact=use_ground_contact,
        )
        goals_by_setting[sk] = goals
        rel_by_setting[sk] = rel_ids

    records = []
    for i, _pose in enumerate(poses):
        frame: Dict = {"action": np.int32(actions[i])}
        for height_cm, pitch_deg in CAMERA_SETTINGS:
            sk = setting_key(height_cm, pitch_deg)
            frame[f"pose.{sk}"] = extrinsics_by_setting[sk][i]
            frame[f"goal.{sk}"] = goals_by_setting[sk][i]
            frame[f"relative_goal_frame_id.{sk}"] = np.int32(rel_by_setting[sk][i])
        records.append(frame)
    return records
