#!/usr/bin/env python3
"""Visualize floor plane, scene point cloud, trajectories, and LeRobot datasets with Rerun."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import rerun as rr

from utils.config import get_config, load_config
from utils.extrinsics import apply_body2optical_transform
from utils.floor_plane import build_floor_frame, floor_xy_to_world_on_plane, project_points_to_plane
from utils.floor_pose import load_floor_calibration, load_pcd_points
from utils.slam_ground import floor_world_from_camera_c2w
from utils.trajectory_io import FloorEntry, parse_floor_trajectory_txt, parse_odom_txt


@dataclass(frozen=True)
class CameraPose:
    timestamp: float
    T_world_cam: np.ndarray


def _optical_frustum_local(
    *,
    depth: float,
    half_width: float,
    half_height: float,
) -> np.ndarray:
    """OpenCV RDF frustum wireframe in camera frame (Z forward)."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
            [-half_width, -half_height, depth],
        ],
        dtype=np.float64,
    )


def _frustum_half_extents(
    depth: float,
    *,
    intrinsic: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Half-width/height of frustum base from camera intrinsics (or compact fallback)."""
    if intrinsic is not None and image_size is not None:
        w, h = image_size
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        if fx > 0 and fy > 0:
            return depth * (w * 0.5) / fx, depth * (h * 0.5) / fy
    return depth * 0.45, depth * 0.28


def _frustum_edges() -> list[tuple[int, int]]:
    return [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ pts.T).T + t


def _camera_poses_from_poses(poses: list[dict]) -> list[CameraPose]:
    out: list[CameraPose] = []
    for pose in poses:
        if "camera_matrix" not in pose:
            continue
        T = np.asarray(pose["camera_matrix"], dtype=np.float64).reshape(4, 4)
        out.append(CameraPose(timestamp=float(pose["timestamp"]), T_world_cam=T))
    return out


def _camera_poses_from_odom(
    odom_path: Path,
    *,
    apply_body2optical: bool,
) -> list[CameraPose]:
    out: list[CameraPose] = []
    for entry in parse_odom_txt(str(odom_path)):
        T = apply_body2optical_transform(entry.matrix, apply=apply_body2optical)
        out.append(CameraPose(timestamp=entry.timestamp, T_world_cam=T))
    return out


def _resolve_path(candidates: list[Path | None]) -> Path | None:
    for path in candidates:
        if path is not None and Path(path).is_file():
            return Path(path)
    return None


def _resolve_dir_file(candidates: list[Path | None], filename: str) -> Path | None:
    for base in candidates:
        if base is None:
            continue
        base = Path(base)
        if base.is_file():
            return base
        candidate = base / filename
        if candidate.is_file():
            return candidate
    return None


def _load_poses(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _floor_world_positions_from_poses(poses: list[dict]) -> np.ndarray:
    pts = []
    for pose in poses:
        if all(k in pose for k in ("world_x", "world_y", "world_z")):
            pts.append([pose["world_x"], pose["world_y"], pose["world_z"]])
    return np.asarray(pts, dtype=np.float64)


def _floor_world_positions_from_entries(entries: list[FloorEntry]) -> np.ndarray:
    return np.array(
        [[e.world_x, e.world_y, e.world_z] for e in entries],
        dtype=np.float64,
    )


def _floor_world_on_plane_from_entries(
    entries: list[FloorEntry],
    floor_plane: tuple,
) -> np.ndarray:
    return np.array(
        [
            floor_xy_to_world_on_plane(e.x, e.y, floor_plane)
            for e in entries
        ],
        dtype=np.float64,
    )


def _slam_ground_path_from_camera_poses(
    camera_poses: list[CameraPose],
    *,
    camera_pitch_deg: float,
    ground_offset_y: float,
) -> np.ndarray:
    """SLAM floor contact: undo mount pitch, offset along optical +Y (project_slam_path trick)."""
    pts = []
    for pose in camera_poses:
        w = floor_world_from_camera_c2w(
            pose.T_world_cam,
            camera_pitch_deg=camera_pitch_deg,
            ground_offset_y=ground_offset_y,
        )
        pts.append(w)
    return np.asarray(pts, dtype=np.float64)


def _resolve_floor_path_mode(
    args,
    *,
    has_pcd: bool,
    cal: dict,
) -> str:
    if args.floor_path_mode != "auto":
        return args.floor_path_mode
    if has_pcd:
        return "pcd"
    if cal.get("pcd_path") or cal.get("source") != "floor_trajectory":
        return "pcd"
    return "trick"


def _load_point_cloud(
    pcd_path: Path,
    *,
    voxel_size: float,
    max_points: int,
) -> np.ndarray:
    if voxel_size > 0:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        if len(pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {pcd_path}")
        pcd = pcd.voxel_down_sample(voxel_size)
        points = np.asarray(pcd.points, dtype=np.float64)
    else:
        points = load_pcd_points(pcd_path)

    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    return points


def _color_points_by_plane_distance(
    points: np.ndarray,
    floor_plane: tuple,
    *,
    inlier_thresh: float,
) -> np.ndarray:
    a, b, c, d = floor_plane
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n) + 1e-12
    dist = np.abs(points @ n + d) / n_norm
    colors = np.tile(np.array([160, 160, 170], dtype=np.uint8), (len(points), 1))
    inliers = dist < inlier_thresh
    colors[inliers] = np.array([120, 220, 140], dtype=np.uint8)
    return colors


def _floor_plane_mesh(
    floor_plane: tuple,
    anchor_points: np.ndarray,
    *,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    origin, x_ax, y_ax, _ = build_floor_frame(floor_plane)
    if len(anchor_points) == 0:
        half = 5.0
        corners = np.array(
            [
                origin - half * x_ax - half * y_ax,
                origin + half * x_ax - half * y_ax,
                origin + half * x_ax + half * y_ax,
                origin - half * x_ax + half * y_ax,
            ],
            dtype=np.float64,
        )
    else:
        rel = anchor_points - origin
        u = rel @ x_ax
        v = rel @ y_ax
        u0, u1 = float(u.min() - margin), float(u.max() + margin)
        v0, v1 = float(v.min() - margin), float(v.max() + margin)
        corners = np.array(
            [
                origin + u0 * x_ax + v0 * y_ax,
                origin + u1 * x_ax + v0 * y_ax,
                origin + u1 * x_ax + v1 * y_ax,
                origin + u0 * x_ax + v1 * y_ax,
            ],
            dtype=np.float64,
        )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return corners, triangles


def _log_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    rr.log(
        "world/map",
        rr.Points3D(points, colors=colors, radii=0.04),
        static=True,
    )


def _uniform_pcd_colors(n: int) -> np.ndarray:
    return np.tile(np.array([170, 170, 175], dtype=np.uint8), (n, 1))


def _log_floor_plane(
    floor_plane: tuple,
    anchor_points: np.ndarray,
    floor_point: np.ndarray | None,
    *,
    margin: float,
) -> None:
    corners, triangles = _floor_plane_mesh(floor_plane, anchor_points, margin=margin)
    rr.log(
        "world/floor_plane",
        rr.Mesh3D(
            vertex_positions=corners,
            triangle_indices=triangles,
            vertex_colors=[[80, 200, 120, 90]] * 4,
        ),
        static=True,
    )
    if floor_point is not None:
        rr.log(
            "world/floor_plane/anchor",
            rr.Points3D([floor_point], colors=[255, 220, 80], radii=0.15),
            static=True,
        )


def _log_static_paths(
    camera_pts: np.ndarray,
    floor_pts: np.ndarray,
    *,
    floor_entity: str = "world/floor_path",
) -> None:
    if len(camera_pts) > 1:
        rr.log(
            "world/camera_path",
            rr.LineStrips3D([camera_pts], colors=[80, 160, 255], radii=0.05),
            static=True,
        )
    if len(floor_pts) > 1:
        rr.log(
            floor_entity,
            rr.LineStrips3D([floor_pts], colors=[50, 220, 120], radii=0.06),
            static=True,
        )
    if len(camera_pts):
        rr.log(
            "world/camera_start",
            rr.Points3D(camera_pts[:1], colors=[255, 80, 80], radii=0.15),
            static=True,
        )
        rr.log(
            "world/camera_end",
            rr.Points3D(camera_pts[-1:], colors=[255, 200, 60], radii=0.15),
            static=True,
        )
    if len(floor_pts):
        rr.log(
            "world/floor_start",
            rr.Points3D(floor_pts[:1], colors=[80, 255, 180], radii=0.15),
            static=True,
        )
        rr.log(
            "world/floor_end",
            rr.Points3D(floor_pts[-1:], colors=[40, 180, 255], radii=0.15),
            static=True,
        )


def _log_static_camera_poses(
    camera_poses: list[CameraPose],
    *,
    stride: int,
    axis_len: float,
    frustum_depth: float,
    frustum_half_width: float,
    frustum_half_height: float,
) -> None:
    if not camera_poses:
        return

    sampled = camera_poses[:: max(1, stride)]
    local_frustum = _optical_frustum_local(
        depth=frustum_depth,
        half_width=frustum_half_width,
        half_height=frustum_half_height,
    )
    edges = _frustum_edges()

    axis_origins: list[list[float]] = []
    axis_vectors: list[list[float]] = []
    axis_colors: list[list[int]] = []
    frustum_strips: list[np.ndarray] = []

    axis_colors_rgb = ([255, 80, 80], [80, 255, 80], [255, 220, 80])
    for pose in sampled:
        T = pose.T_world_cam
        t = T[:3, 3]
        R = T[:3, :3]
        for col, color in enumerate(axis_colors_rgb):
            axis_origins.append(t.tolist())
            axis_vectors.append((R[:, col] * axis_len).tolist())
            axis_colors.append(color)

        world_frustum = _transform_points(T, local_frustum)
        for i0, i1 in edges:
            frustum_strips.append(np.vstack([world_frustum[i0], world_frustum[i1]]))

    rr.log(
        "world/camera_axes",
        rr.Arrows3D(origins=axis_origins, vectors=axis_vectors, colors=axis_colors),
        static=True,
    )
    if frustum_strips:
        rr.log(
            "world/camera_frustums",
            rr.LineStrips3D(frustum_strips, colors=[120, 180, 255], radii=0.008),
            static=True,
        )


def _log_camera_rig(
    entity: str,
    T: np.ndarray,
    *,
    axis_len: float,
    frustum_depth: float,
    intrinsic: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
) -> None:
    rr.log(
        entity,
        rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
    )
    rr.log(entity, rr.ViewCoordinates.RDF, static=True)
    rr.log(
        f"{entity}/axes",
        rr.Arrows3D(
            origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            vectors=[
                [axis_len, 0.0, 0.0],
                [0.0, axis_len, 0.0],
                [0.0, 0.0, axis_len],
            ],
            colors=[[255, 80, 80], [80, 255, 80], [255, 220, 80]],
        ),
    )
    half_w, half_h = _frustum_half_extents(
        frustum_depth, intrinsic=intrinsic, image_size=image_size
    )
    local = _optical_frustum_local(
        depth=frustum_depth,
        half_width=half_w,
        half_height=half_h,
    )
    strips = [local[[i0, i1]] for i0, i1 in _frustum_edges()]
    rr.log(
        f"{entity}/frustum",
        rr.LineStrips3D(strips, colors=[120, 180, 255], radii=0.008),
    )


def _log_timeline_from_camera_poses(
    camera_poses: list[CameraPose],
    stride: int,
    *,
    axis_len: float,
    frustum_depth: float,
) -> None:
    for pose in camera_poses[:: max(1, stride)]:
        rr.set_time("timestamp", timestamp=pose.timestamp)
        _log_camera_rig(
            "world/camera",
            pose.T_world_cam,
            axis_len=axis_len,
            frustum_depth=frustum_depth,
        )


def _log_timeline_from_poses(
    poses: list[dict],
    camera_poses: list[CameraPose],
    stride: int,
    *,
    axis_len: float,
    frustum_depth: float,
) -> None:
    if camera_poses:
        _log_timeline_from_camera_poses(
            camera_poses,
            stride,
            axis_len=axis_len,
            frustum_depth=frustum_depth,
        )

    for pose in poses[:: max(1, stride)]:
        if not camera_poses:
            ts = float(pose["timestamp"])
            rr.set_time("timestamp", timestamp=ts)
        if all(k in pose for k in ("world_x", "world_y", "world_z")):
            p = np.array(
                [pose["world_x"], pose["world_y"], pose["world_z"]],
                dtype=np.float64,
            )
            rr.log(
                "world/floor_pose", rr.Points3D([p], colors=[80, 255, 120], radii=0.08)
            )


def _log_timeline_from_floor(entries: list[FloorEntry], stride: int) -> None:
    for entry in entries[:: max(1, stride)]:
        rr.set_time("timestamp", timestamp=entry.timestamp)
        p = np.array([entry.world_x, entry.world_y, entry.world_z], dtype=np.float64)
        rr.log("world/floor_pose", rr.Points3D([p], colors=[80, 255, 120], radii=0.08))


_SETTING_RE = re.compile(r"^(\d+)cm_(\d+)deg$")
_DEFAULT_LEROBOT_INTRINSIC = np.array(
    [[323.5205, 0.0, 318.6513], [0.0, 323.2016, 185.4311], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
_DEFAULT_LEROBOT_NATIVE_SIZE = (1280, 720)


def _parse_setting(setting: str) -> tuple[int, int]:
    match = _SETTING_RE.match(setting)
    if not match:
        raise ValueError(
            f"Invalid setting {setting!r}; expected format like 125cm_0deg"
        )
    return int(match.group(1)), int(match.group(2))


def _scale_intrinsic_to_image(
    intrinsic: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    sw, sh = src_size
    dw, dh = dst_size
    sx, sy = dw / sw, dh / sh
    k = np.array(intrinsic, dtype=np.float64).reshape(3, 3).copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def _pose_cell_to_matrix(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=object)
    if arr.shape == (4, 4):
        return np.asarray(value, dtype=np.float64).reshape(4, 4)
    if arr.ndim == 1 and len(arr) == 4:
        rows = [np.asarray(row, dtype=np.float64) for row in value]
        return np.stack(rows, axis=0)
    return np.asarray(value, dtype=np.float64).reshape(4, 4)


def _goal_cell_to_uv(value: object) -> tuple[int, int]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return int(arr), -1
    return int(arr[0]), int(arr[1])


def _lerobot_parquet_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return (
        root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def _lerobot_rgb_path(
    root: Path,
    episode_index: int,
    local_frame: int,
    height_cm: int,
    pitch_deg: int,
) -> Path:
    chunk = episode_index // 1000
    return (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.rgb.{height_cm}cm_{pitch_deg}deg"
        / f"episode_{episode_index:06d}_{local_frame}.jpg"
    )


def _list_lerobot_episode_indices(root: Path) -> list[int]:
    data_dir = root / "data"
    if not data_dir.is_dir():
        return []
    indices: list[int] = []
    for pq in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        indices.append(int(pq.stem.split("_")[-1]))
    return sorted(indices)


def _load_lerobot_episode_table(root: Path, episode_index: int):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for --lerobot-root: pip install pandas pyarrow") from exc

    pq_path = _lerobot_parquet_path(root, episode_index)
    if not pq_path.is_file():
        raise FileNotFoundError(pq_path)
    return pd.read_parquet(pq_path)


def _annotate_goal_on_rgb(rgb: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    out = rgb.copy()
    u, v = goal
    h, w = out.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(out, (u, v), 8, (255, 60, 60), -1)
        cv2.circle(out, (u, v), 10, (255, 255, 255), 1)
    return out


def _ground_points_from_pose_matrices(
    pose_matrices: list[np.ndarray],
    *,
    camera_pitch_deg: float,
    ground_offset_y: float,
) -> np.ndarray:
    pts = []
    for T in pose_matrices:
        pts.append(
            floor_world_from_camera_c2w(
                T,
                camera_pitch_deg=camera_pitch_deg,
                ground_offset_y=ground_offset_y,
            )
        )
    return np.asarray(pts, dtype=np.float64)


def _log_pinhole(entity: str, intrinsic: np.ndarray, width: int, height: int) -> None:
    k = intrinsic.reshape(3, 3)
    rr.log(
        entity,
        rr.Pinhole(
            focal_length=[float(k[0, 0]), float(k[1, 1])],
            principal_point=[float(k[0, 2]), float(k[1, 2])],
            width=width,
            height=height,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )


def _log_lerobot_frame(
    *,
    entity_prefix: str,
    T_world_cam: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    goal: tuple[int, int],
    axis_len: float,
    frustum_depth: float,
    draw_goal_on_image: bool,
) -> None:
    h, w = rgb.shape[:2]
    _log_camera_rig(
        entity_prefix,
        T_world_cam,
        axis_len=axis_len,
        frustum_depth=frustum_depth,
        intrinsic=intrinsic,
        image_size=(w, h),
    )
    _log_pinhole(entity_prefix, intrinsic, w, h)

    vis_rgb = _annotate_goal_on_rgb(rgb, goal) if draw_goal_on_image else rgb
    rr.log(f"{entity_prefix}/rgb", rr.Image(vis_rgb))

    u, v = goal
    if 0 <= u < w and 0 <= v < h:
        rr.log(
            f"{entity_prefix}/rgb/goal",
            rr.Points2D(
                positions=[[float(u), float(v)]],
                colors=[[255, 60, 60]],
                radii=10.0,
            ),
        )


def _infer_scene_from_lerobot_root(root: Path) -> str | None:
    """Guess scene config name from dataset folder (e.g. bkhn_round1, office_round1)."""
    name = root.name
    if name.startswith(("bkhn_", "office_")):
        return name
    return None


def _resolve_lerobot_pcd_and_calibration(
    args: argparse.Namespace,
    root: Path,
) -> tuple[Path | None, dict | None]:
    """Resolve scene PCD and optional floor_calibration for LeRobot visualization."""
    cal: dict | None = None
    cal_path = args.floor_calibration

    scene = args.scene or _infer_scene_from_lerobot_root(root)
    cfg_paths: dict = {}
    if scene:
        try:
            cfg_paths = load_config(scene).get("paths", default={})
        except FileNotFoundError:
            pass

    cal_candidates: list[Path | None] = [cal_path]
    if cfg_paths.get("output_dir"):
        cal_candidates.append(Path(cfg_paths["output_dir"]) / "floor_calibration.json")
    if cfg_paths.get("bag"):
        cal_candidates.append(Path(cfg_paths["bag"]) / "floor_calibration.json")
    if cfg_paths.get("floor_trajectory"):
        cal_candidates.append(
            Path(cfg_paths["floor_trajectory"]).parent / "floor_calibration.json"
        )

    resolved_cal = _resolve_path(cal_candidates)
    if resolved_cal is not None:
        cal = load_floor_calibration(resolved_cal)

    pcd_candidates: list[Path | None] = [args.pcd]
    if cal and cal.get("pcd_path"):
        pcd_candidates.append(Path(cal["pcd_path"]))
    if cfg_paths.get("pcd"):
        pcd_candidates.append(Path(cfg_paths["pcd"]))
    if cfg_paths.get("bag"):
        bag_dir = Path(cfg_paths["bag"])
        if scene:
            pcd_candidates.append(bag_dir / f"{scene}_point2plane.pcd")
        for pattern in ("*_point2plane.pcd", "*.pcd"):
            matches = sorted(bag_dir.glob(pattern))
            if matches:
                pcd_candidates.append(matches[0])
                break

    pcd_path = _resolve_path(pcd_candidates)
    return pcd_path, cal


def _filter_map_points_near_floor(
    points: np.ndarray,
    floor_plane: tuple,
    *,
    max_dist: float = 2.5,
) -> np.ndarray:
    """Drop ceiling/far clutter so the map reads as floor-level geometry in Rerun."""
    if len(points) == 0:
        return points
    a, b, c, d = floor_plane
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n) + 1e-12
    dist = np.abs(points @ n + d) / n_norm
    return points[dist <= max_dist]


def _signed_dist_to_plane(points: np.ndarray, floor_plane: tuple) -> np.ndarray:
    a, b, c, d = floor_plane
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n) + 1e-12
    pts = np.atleast_2d(points).astype(np.float64)
    return (pts @ n + d) / n_norm


def _lerobot_floor_path_points(
    pose_matrices: list[np.ndarray],
    *,
    floor_plane: tuple | None,
    cal: dict | None,
    use_slam_trick: bool,
    camera_pitch_deg: float,
    ground_offset_y: float,
    keyframe_poses_path: Path | None,
    n_frames: int,
    global_indices: list[int] | None = None,
) -> np.ndarray:
    """Floor path for LeRobot view: embodiment on plane (BKHN), not SLAM ground trick."""
    if keyframe_poses_path is not None and keyframe_poses_path.is_file():
        poses = json.loads(keyframe_poses_path.read_text())
        if poses and all(k in poses[0] for k in ("world_x", "world_y", "world_z")):
            if global_indices is not None:
                pts = []
                for gi in global_indices:
                    if 0 <= gi < len(poses):
                        p = poses[gi]
                        pts.append([p["world_x"], p["world_y"], p["world_z"]])
                if pts:
                    return np.array(pts, dtype=np.float64)
            n = min(len(poses), n_frames)
            return np.array(
                [[p["world_x"], p["world_y"], p["world_z"]] for p in poses[:n]],
                dtype=np.float64,
            )

    if use_slam_trick:
        return _ground_points_from_pose_matrices(
            pose_matrices,
            camera_pitch_deg=camera_pitch_deg,
            ground_offset_y=ground_offset_y,
        )

    if floor_plane is not None:
        camera_pts = np.array([T[:3, 3] for T in pose_matrices], dtype=np.float64)
        return project_points_to_plane(camera_pts, floor_plane)

    return _ground_points_from_pose_matrices(
        pose_matrices,
        camera_pitch_deg=camera_pitch_deg,
        ground_offset_y=ground_offset_y,
    )


def _log_map_pcd(
    pcd_path: Path,
    cal: dict | None,
    *,
    raw: bool,
    pcd_voxel: float,
    pcd_max_points: int,
    plane_inlier_thresh: float,
    show_floor_plane: bool = False,
    plane_margin: float = 3.0,
    anchor_points: np.ndarray | None = None,
    floor_plane: tuple | None = None,
) -> np.ndarray:
    """Load and log scene point cloud. raw=True: file coords, no filter/coloring."""
    map_points = _load_point_cloud(
        pcd_path,
        voxel_size=pcd_voxel,
        max_points=pcd_max_points,
    )
    resolved_plane = floor_plane
    if resolved_plane is None and cal and cal.get("floor_plane") is not None:
        resolved_plane = tuple(cal["floor_plane"])

    if raw:
        map_colors = _uniform_pcd_colors(len(map_points))
        print(f"[rerun] raw pcd: {len(map_points):,} pts (no filter or plane coloring)")
    else:
        if resolved_plane is not None:
            before = len(map_points)
            map_points = _filter_map_points_near_floor(map_points, resolved_plane)
            if len(map_points) < before:
                print(
                    f"[rerun] pcd filter: kept {len(map_points):,}/{before:,} pts "
                    "near floor plane"
                )
        if resolved_plane is None:
            resolved_plane = (0.0, 0.0, 1.0, 0.0)
        map_colors = _color_points_by_plane_distance(
            map_points,
            resolved_plane,
            inlier_thresh=plane_inlier_thresh,
        )

    _log_point_cloud(map_points, map_colors)

    if show_floor_plane and cal is not None and resolved_plane is not None:
        floor_point = np.asarray(cal.get("floor_point", [0, 0, 0]), dtype=np.float64)
        anchor = (
            anchor_points
            if anchor_points is not None and len(anchor_points)
            else map_points
        )
        _log_floor_plane(
            resolved_plane,
            anchor,
            floor_point,
            margin=plane_margin,
        )
    return map_points


def _log_lerobot_episode(
    root: Path,
    episode_index: int,
    *,
    setting: str,
    stride: int,
    max_frames: int | None,
    axis_len: float,
    frustum_depth: float,
    draw_goal_on_image: bool,
    intrinsic: np.ndarray,
    native_image_size: tuple[int, int],
    timeline: bool,
    camera_pitch_deg: float | None,
    ground_offset_y: float,
    draw_trajectory: bool,
    floor_cal: dict | None = None,
    keyframe_poses_path: Path | None = None,
    use_slam_trick_floor_path: bool = False,
) -> tuple[int, int]:
    height_cm, pitch_deg = _parse_setting(setting)
    pose_col = f"pose.{setting}"
    goal_col = f"goal.{setting}"

    df = _load_lerobot_episode_table(root, episode_index)
    if pose_col not in df.columns or goal_col not in df.columns:
        available = [c for c in df.columns if c.startswith("pose.")]
        raise KeyError(f"{pose_col} missing in parquet; available: {available}")

    n_frames = len(df)
    goals_all = np.array([_goal_cell_to_uv(v) for v in df[goal_col].tolist()])
    n_valid = int(np.sum((goals_all[:, 0] >= 0) & (goals_all[:, 1] >= 0)))
    if n_valid == 0:
        other: list[str] = []
        for col in df.columns:
            if not col.startswith("goal."):
                continue
            g = np.array([_goal_cell_to_uv(v) for v in df[col].tolist()])
            if int(np.sum((g[:, 0] >= 0) & (g[:, 1] >= 0))) > 0:
                other.append(col.removeprefix("goal."))
        hint = f" Try --setting {other[0]}." if other else ""
        print(f"  Warning: no valid pixel goals for setting={setting}.{hint}")

    frame_ids = list(range(0, n_frames, max(1, stride)))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

    pose_matrices = [
        _pose_cell_to_matrix(df.iloc[i][pose_col]) for i in range(n_frames)
    ]
    global_indices: list[int] | None = None
    if "index" in df.columns:
        global_indices = [int(v) for v in df["index"].tolist()]

    mount_pitch = float(pitch_deg if camera_pitch_deg is None else camera_pitch_deg)
    floor_plane = (
        tuple(floor_cal["floor_plane"])
        if floor_cal and floor_cal.get("floor_plane") is not None
        else None
    )
    ground_pts = _lerobot_floor_path_points(
        pose_matrices,
        floor_plane=floor_plane,
        cal=floor_cal,
        use_slam_trick=use_slam_trick_floor_path,
        camera_pitch_deg=mount_pitch,
        ground_offset_y=ground_offset_y,
        keyframe_poses_path=keyframe_poses_path,
        n_frames=n_frames,
        global_indices=global_indices,
    )

    camera_pts = [T[:3, 3].tolist() for T in pose_matrices]
    if floor_plane is not None and camera_pts:
        sd = _signed_dist_to_plane(np.array(camera_pts), floor_plane)
        below = int(np.sum(sd < 0))
        print(
            f"  camera height vs floor plane: median={float(np.median(sd)):+.2f} m "
            f"(+ = above plane, {below}/{len(sd)} below)"
        )

    episode_entity = f"world/episode_{episode_index:06d}"
    if len(camera_pts) > 1:
        rr.log(
            f"{episode_entity}/camera_path",
            rr.LineStrips3D([camera_pts], colors=[80, 160, 255], radii=0.04),
            static=True,
        )
    if draw_trajectory and len(ground_pts) > 1:
        rr.log(
            f"{episode_entity}/floor_path",
            rr.LineStrips3D([ground_pts.tolist()], colors=[50, 220, 120], radii=0.06),
            static=True,
        )
        rr.log(
            f"{episode_entity}/floor_start",
            rr.Points3D([ground_pts[0]], colors=[80, 255, 180], radii=0.12),
            static=True,
        )
        rr.log(
            f"{episode_entity}/floor_end",
            rr.Points3D([ground_pts[-1]], colors=[40, 180, 255], radii=0.12),
            static=True,
        )

    logged = 0
    valid_goals = 0
    scaled_k: np.ndarray | None = None

    for local_i in frame_ids:
        row = df.iloc[local_i]
        img_path = _lerobot_rgb_path(root, episode_index, local_i, height_cm, pitch_deg)
        if not img_path.is_file():
            print(f"[warn] missing image: {img_path}")
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[warn] failed to read: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if scaled_k is None:
            scaled_k = _scale_intrinsic_to_image(
                intrinsic, native_image_size, (rgb.shape[1], rgb.shape[0])
            )

        T = _pose_cell_to_matrix(row[pose_col])
        goal = _goal_cell_to_uv(row[goal_col])
        if goal[0] >= 0 and goal[1] >= 0:
            valid_goals += 1

        if timeline:
            rr.set_time("frame", sequence=local_i)
            if "timestamp" in df.columns:
                rr.set_time("timestamp", timestamp=float(row["timestamp"]))

        frame_entity = (
            f"{episode_entity}/camera"
            if timeline
            else f"{episode_entity}/camera/frame_{local_i:06d}"
        )
        _log_lerobot_frame(
            entity_prefix=frame_entity,
            T_world_cam=T,
            rgb=rgb,
            intrinsic=scaled_k,
            goal=goal,
            axis_len=axis_len,
            frustum_depth=frustum_depth,
            draw_goal_on_image=draw_goal_on_image,
        )
        logged += 1

    return logged, valid_goals


def main_lerobot(args: argparse.Namespace) -> int:
    root = Path(args.lerobot_root)
    if not root.is_dir():
        print(f"Error: --lerobot-root is not a directory: {root}", file=sys.stderr)
        return 1

    setting = args.setting
    try:
        _parse_setting(setting)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    episodes = (
        [args.episode]
        if args.episode is not None
        else _list_lerobot_episode_indices(root)
    )
    if not episodes:
        print(f"Error: no episode parquet files under {root / 'data'}", file=sys.stderr)
        return 1

    intrinsic = _DEFAULT_LEROBOT_INTRINSIC
    native_size = _DEFAULT_LEROBOT_NATIVE_SIZE
    if args.camera_intrinsic is not None:
        vals = [float(x) for x in args.camera_intrinsic.replace(",", " ").split()]
        if len(vals) != 9:
            print("Error: --camera-intrinsic expects 9 values (3x3 row-major)", file=sys.stderr)
            return 1
        intrinsic = np.asarray(vals, dtype=np.float64).reshape(3, 3)

    app_id = f"lerobot_{root.name}_{setting}"
    spawn = args.spawn or args.save is None
    rr.init(app_id, spawn=spawn)
    if args.save is not None:
        rr.save(args.save)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            colors=[[255, 80, 80], [80, 255, 80], [80, 120, 255]],
        ),
        static=True,
    )

    show_pcd = args.pcd is not None or args.with_pcd or args.raw_pcd
    floor_cal: dict | None = None
    if show_pcd:
        pcd_path, floor_cal = _resolve_lerobot_pcd_and_calibration(args, root)
        if pcd_path is not None and pcd_path.is_file():
            n_pts = len(
                _log_map_pcd(
                    pcd_path,
                    floor_cal,
                    raw=args.raw_pcd,
                    pcd_voxel=args.pcd_voxel,
                    pcd_max_points=args.pcd_max_points,
                    plane_inlier_thresh=args.plane_inlier_thresh,
                    show_floor_plane=args.show_floor_plane,
                    plane_margin=args.plane_margin,
                )
            )
            print(f"[lerobot] pcd: {pcd_path} ({n_pts:,} pts)")
            if args.show_floor_plane and floor_cal is not None:
                print("[lerobot] floor plane mesh: yes")
        else:
            tried = args.pcd or "(auto-resolve failed)"
            print(
                f"[lerobot] warning: point cloud not found ({tried}). "
                "Pass --pcd PATH, --floor-calibration, or --scene bkhn_round1.",
                file=sys.stderr,
            )

    scene = args.scene or _infer_scene_from_lerobot_root(root)
    keyframe_poses_path: Path | None = None
    use_slam_trick = False
    if scene:
        try:
            cfg_paths = load_config(scene).get("paths", default={})
            if cfg_paths.get("output_dir"):
                candidate = Path(cfg_paths["output_dir"]) / "poses.json"
                if candidate.is_file():
                    keyframe_poses_path = candidate
            if floor_cal is None and cfg_paths.get("bag"):
                cal_cand = Path(cfg_paths["bag"]) / "floor_calibration.json"
                if cal_cand.is_file():
                    floor_cal = load_floor_calibration(cal_cand)
        except FileNotFoundError:
            pass
    if floor_cal and floor_cal.get("source") == "slam_odom_export":
        use_slam_trick = True

    total_logged = 0
    total_valid_goals = 0
    for episode_index in episodes:
        print(f"[lerobot] episode {episode_index:06d} setting={setting}")
        logged, valid_goals = _log_lerobot_episode(
            root,
            episode_index,
            setting=setting,
            stride=max(1, args.stride),
            max_frames=args.max_frames,
            axis_len=args.camera_axis_len,
            frustum_depth=args.camera_frustum_depth,
            draw_goal_on_image=not args.no_draw_goal_on_image,
            intrinsic=intrinsic,
            native_image_size=native_size,
            timeline=args.mode == "timeline",
            camera_pitch_deg=args.camera_pitch_deg,
            ground_offset_y=(
                args.ground_offset_y if args.ground_offset_y is not None else 1.5
            ),
            draw_trajectory=not args.no_trajectory,
            floor_cal=floor_cal,
            keyframe_poses_path=keyframe_poses_path,
            use_slam_trick_floor_path=use_slam_trick,
        )
        print(
            f"  logged {logged} frame(s), {valid_goals} with valid pixel goals"
        )
        total_logged += logged
        total_valid_goals += valid_goals

    if total_logged == 0:
        print("Error: no frames logged (check RGB image paths and parquet).", file=sys.stderr)
        return 1

    print(
        f"[lerobot] done: episodes={len(episodes)} frames={total_logged} "
        f"valid_goals={total_valid_goals} mode={args.mode}"
    )
    if args.save:
        print(f"Saved recording to {args.save}")
    else:
        print("Rerun viewer opened. Scrub timeline by 'frame' or 'timestamp'.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize estimated floor plane, point cloud, and trajectories."
    )
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument(
        "--floor-calibration",
        type=Path,
        default=None,
        help="floor_calibration.json (default: search near poses dir / floor trajectory)",
    )
    parser.add_argument(
        "--pcd",
        type=Path,
        default=None,
        help="Scene point cloud (default: pcd_path in floor_calibration.json; "
        "with --lerobot-root use --with-pcd to auto-resolve)",
    )
    parser.add_argument(
        "--floor-trajectory",
        type=Path,
        default=None,
        help="floor_trajectory.txt (embodiment path on the plane)",
    )
    parser.add_argument(
        "--camera-odom",
        type=Path,
        default=None,
        help="Camera odom txt for camera trajectory (default: paths.camera_odom)",
    )
    parser.add_argument(
        "--poses-json",
        type=Path,
        default=None,
        help="Optional poses.json (overrides camera/floor paths when present)",
    )
    parser.add_argument(
        "--pcd-voxel",
        type=float,
        default=0.2,
        help="Voxel size (m) for downsampling the map cloud; 0=full resolution",
    )
    parser.add_argument(
        "--pcd-max-points",
        type=int,
        default=250_000,
        help="Random cap on logged map points (0=no cap)",
    )
    parser.add_argument(
        "--plane-inlier-thresh",
        type=float,
        default=0.08,
        help="Distance (m) to plane for green inlier coloring",
    )
    parser.add_argument(
        "--plane-margin",
        type=float,
        default=3.0,
        help="Extra margin (m) around trajectory for floor plane quad",
    )
    parser.add_argument(
        "--floor-path-mode",
        choices=("auto", "trick", "plane", "pcd"),
        default="auto",
        help=(
            "Floor path source: trick=SLAM ground contact (project_slam_path); "
            "plane=project floor x,y onto fitted plane; pcd=world_x/y/z from "
            "precompute floor_trajectory; auto=trick for office/SLAM, pcd for LiDAR"
        ),
    )
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=None,
        help="SLAM mount pitch to undo for trick path (default: slam_path.camera_pitch_deg)",
    )
    parser.add_argument(
        "--ground-offset-y",
        type=float,
        default=None,
        help="SLAM offset along optical +Y for trick path (default: slam_path.ground_offset_y)",
    )
    parser.add_argument(
        "--show-floor-plane",
        action="store_true",
        help="Draw fitted floor plane mesh (default: only for PCD/LiDAR scenes)",
    )
    parser.add_argument(
        "--mode",
        choices=("static", "timeline"),
        default="static",
        help="static=overview; timeline=scrub poses over time",
    )
    parser.add_argument(
        "--stride", type=int, default=5, help="Log every Nth frame in timeline mode"
    )
    parser.add_argument(
        "--camera-stride",
        type=int,
        default=None,
        help="Log every Nth camera pose (static mode default: 30; timeline uses --stride)",
    )
    parser.add_argument(
        "--camera-axis-len",
        type=float,
        default=0.15,
        help="Length (m) of RGB camera axis arrows",
    )
    parser.add_argument(
        "--camera-frustum-depth",
        type=float,
        default=0.12,
        help="Depth (m) of camera frustum wireframe along optical +Z",
    )
    parser.add_argument(
        "--no-camera-poses",
        action="store_true",
        help="Skip camera axis/frustum pose visualization",
    )
    parser.add_argument(
        "--save", type=Path, default=None, help="Save .rrd instead of spawning viewer"
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Open Rerun viewer (default when not --save)",
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=None,
        help=(
            "LeRobot dataset root (e.g. DATA/final/office_round1_ver2.0). "
            "Visualizes parquet poses, RGB frames, and pixel goals."
        ),
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="125cm_0deg",
        help="Camera setting suffix for pose./goal. columns (default: 125cm_0deg)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to visualize (default: all episodes in dataset)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap number of frames logged per episode (after stride)",
    )
    parser.add_argument(
        "--camera-intrinsic",
        type=str,
        default=None,
        help="Optional 3x3 camera intrinsic row-major (9 floats) for Pinhole logging",
    )
    parser.add_argument(
        "--no-draw-goal-on-image",
        action="store_true",
        help="Skip drawing red dot on RGB; only log Points2D goal overlay",
    )
    parser.add_argument(
        "--with-pcd",
        action="store_true",
        help="With --lerobot-root: load scene point cloud into world/map "
        "(resolve via --pcd, --floor-calibration, or --scene / dataset folder name)",
    )
    parser.add_argument(
        "--raw-pcd",
        action="store_true",
        help="PCD as stored in file: no floor filter, no plane coloring "
        "(use with --with-pcd or --scene)",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Skip 3D floor trajectory (floor_path line in world view)",
    )
    args = parser.parse_args()

    if args.lerobot_root is not None:
        return main_lerobot(args)

    cfg = load_config(args.scene) if args.scene else get_config()
    paths = cfg.get("paths", default={})
    b2o_val = cfg.get("odom_apply_body2optical")
    apply_body2optical = True if b2o_val is None else bool(b2o_val)

    search_dirs = [
        args.poses_json.parent
        if args.poses_json and args.poses_json.is_file()
        else None,
        args.poses_json if args.poses_json and args.poses_json.is_dir() else None,
        Path(paths["output_dir"]) if paths.get("output_dir") else None,
        Path(paths["floor_trajectory"]).parent
        if paths.get("floor_trajectory")
        else None,
        Path(paths["bag"]) if paths.get("bag") else None,
    ]

    cal_candidates: list[Path | None] = [args.floor_calibration]
    for d in search_dirs:
        if d is None:
            continue
        cal_candidates.append(Path(d) / cfg.floor_calibration_filename())
    floor_cal_path = _resolve_path(cal_candidates)

    traj_candidates: list[Path | None] = [args.floor_trajectory]
    if paths.get("floor_trajectory"):
        traj_candidates.append(Path(paths["floor_trajectory"]))
    for d in search_dirs:
        if d is None:
            continue
        traj_candidates.append(Path(d) / cfg.floor_trajectory_filename())
    floor_traj_path = _resolve_path(traj_candidates)

    poses_path = args.poses_json
    if poses_path is not None and poses_path.is_dir():
        poses_path = poses_path / "poses.json"
    elif poses_path is None:
        poses_path = _resolve_dir_file(search_dirs, "poses.json")

    if floor_cal_path is None:
        print(
            "Error: floor_calibration.json not found. Pass --floor-calibration or run "
            "precompute_floor_trajectory.py first.",
            file=sys.stderr,
        )
        return 1

    cal = load_floor_calibration(floor_cal_path)
    floor_plane = cal["floor_plane"]
    floor_point = np.asarray(cal.get("floor_point", [0, 0, 0]), dtype=np.float64)
    slam_cfg = cfg.slam_path
    camera_pitch_deg = (
        args.camera_pitch_deg
        if args.camera_pitch_deg is not None
        else float(slam_cfg.get("camera_pitch_deg", 30.0))
    )
    ground_offset_y = (
        args.ground_offset_y
        if args.ground_offset_y is not None
        else float(slam_cfg.get("ground_offset_y", 1.5))
    )

    pcd_path = args.pcd
    if pcd_path is None and cal.get("pcd_path"):
        pcd_path = Path(cal["pcd_path"])
    has_pcd = pcd_path is not None and Path(pcd_path).is_file()
    floor_path_mode = _resolve_floor_path_mode(args, has_pcd=has_pcd, cal=cal)
    show_floor_plane = args.show_floor_plane or (floor_path_mode == "pcd")
    floor_path_entity = (
        "world/slam_ground_path"
        if floor_path_mode == "trick"
        else "world/floor_path"
    )

    floor_entries: list[FloorEntry] = []
    if floor_traj_path is not None:
        floor_entries = parse_floor_trajectory_txt(floor_traj_path)

    poses: list[dict] = []
    if poses_path is not None and poses_path.is_file():
        poses = _load_poses(poses_path)

    camera_poses = _camera_poses_from_poses(poses)
    if not camera_poses:
        camera_odom = args.camera_odom
        if camera_odom is None and paths.get("camera_odom"):
            camera_odom = Path(paths["camera_odom"])
        if camera_odom is not None and camera_odom.is_file():
            camera_poses = _camera_poses_from_odom(
                camera_odom, apply_body2optical=apply_body2optical
            )

    floor_pts = np.empty((0, 3), dtype=np.float64)
    if floor_path_mode == "trick":
        if camera_poses:
            floor_pts = _slam_ground_path_from_camera_poses(
                camera_poses,
                camera_pitch_deg=camera_pitch_deg,
                ground_offset_y=ground_offset_y,
            )
        elif floor_entries:
            floor_pts = _floor_world_positions_from_entries(floor_entries)
    elif floor_path_mode == "plane":
        if floor_entries:
            floor_pts = _floor_world_on_plane_from_entries(floor_entries, floor_plane)
        else:
            pose_xy = [
                (float(p["x"]), float(p["y"]))
                for p in poses
                if all(k in p for k in ("x", "y"))
            ]
            floor_pts = np.array(
                [floor_xy_to_world_on_plane(x, y, floor_plane) for x, y in pose_xy],
                dtype=np.float64,
            )
    else:
        floor_pts = _floor_world_positions_from_poses(poses)
        if len(floor_pts) == 0 and floor_entries:
            floor_pts = _floor_world_positions_from_entries(floor_entries)

    camera_pts = (
        np.array([p.T_world_cam[:3, 3] for p in camera_poses], dtype=np.float64)
        if camera_poses
        else np.empty((0, 3), dtype=np.float64)
    )

    if len(floor_pts) == 0 and len(camera_pts) == 0:
        print(
            "Error: no trajectory found. Provide --floor-trajectory and/or --camera-odom or poses.json.",
            file=sys.stderr,
        )
        return 1

    print(f"[rerun] calibration: {floor_cal_path}")
    if has_pcd:
        print(f"[rerun] pcd:           {pcd_path}")
    else:
        print("[rerun] pcd:           skipped (no PCD for this scene; normal for office/SLAM)")
    if floor_traj_path:
        print(f"[rerun] floor traj:    {floor_traj_path} ({len(floor_entries)} pts)")
    print(
        f"[rerun] floor path:    mode={floor_path_mode} entity={floor_path_entity}"
    )
    if floor_path_mode == "trick":
        print(
            f"[rerun]               pitch={camera_pitch_deg:.1f}° "
            f"ground_offset_y={ground_offset_y:.2f}m (project_slam_path trick)"
        )
    if len(camera_pts):
        print(
            f"[rerun] camera traj:   {len(camera_pts)} pts ({len(camera_poses)} poses)"
        )
    if len(floor_pts):
        print(f"[rerun] floor 3d:      {len(floor_pts)} pts")

    anchor_pts = np.empty((0, 3), dtype=np.float64)
    if len(floor_pts):
        anchor_pts = floor_pts
    if len(camera_pts):
        anchor_pts = (
            np.vstack([anchor_pts, camera_pts]) if len(anchor_pts) else camera_pts
        )

    app_id = f"vln_floor_{cfg.scene_name or 'viz'}"
    spawn = args.spawn or args.save is None
    rr.init(app_id, spawn=spawn)
    if args.save is not None:
        rr.save(args.save)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            colors=[[255, 80, 80], [80, 255, 80], [80, 120, 255]],
        ),
        static=True,
    )

    map_points = np.empty((0, 3), dtype=np.float64)
    if has_pcd:
        map_points = _log_map_pcd(
            Path(pcd_path),
            cal,
            raw=args.raw_pcd,
            pcd_voxel=args.pcd_voxel,
            pcd_max_points=args.pcd_max_points,
            plane_inlier_thresh=args.plane_inlier_thresh,
            show_floor_plane=show_floor_plane and not args.raw_pcd,
            plane_margin=args.plane_margin,
            anchor_points=anchor_pts,
            floor_plane=floor_plane,
        )
        print(f"[rerun] pcd logged: {len(map_points):,} pts")
    elif show_floor_plane:
        _log_floor_plane(
            floor_plane,
            anchor_pts,
            floor_point,
            margin=args.plane_margin,
        )
    _log_static_paths(
        camera_pts,
        floor_pts,
        floor_entity=floor_path_entity,
    )

    camera_stride = args.camera_stride
    if camera_stride is None:
        camera_stride = args.stride if args.mode == "timeline" else 30

    if camera_poses and not args.no_camera_poses:
        frustum_hw = _frustum_half_extents(
            args.camera_frustum_depth,
            intrinsic=_DEFAULT_LEROBOT_INTRINSIC,
            image_size=_DEFAULT_LEROBOT_NATIVE_SIZE,
        )
        if args.mode == "static":
            _log_static_camera_poses(
                camera_poses,
                stride=camera_stride,
                axis_len=args.camera_axis_len,
                frustum_depth=args.camera_frustum_depth,
                frustum_half_width=frustum_hw[0],
                frustum_half_height=frustum_hw[1],
            )
            print(
                f"[rerun] camera poses:  {len(camera_poses[:: max(1, camera_stride)])} shown (stride={camera_stride})"
            )

    if args.mode == "timeline":
        if camera_poses and not args.no_camera_poses:
            _log_timeline_from_camera_poses(
                camera_poses,
                max(1, args.stride),
                axis_len=args.camera_axis_len,
                frustum_depth=args.camera_frustum_depth,
            )
        elif poses:
            _log_timeline_from_poses(
                poses,
                camera_poses,
                max(1, args.stride),
                axis_len=args.camera_axis_len,
                frustum_depth=args.camera_frustum_depth,
            )
        elif floor_entries:
            _log_timeline_from_floor(floor_entries, max(1, args.stride))

    print(
        f"Logged map={len(map_points):,} pts, plane={'yes' if show_floor_plane else 'no'}, "
        f"camera={len(camera_pts)}, floor={len(floor_pts)}"
    )
    if args.save:
        print(f"Saved recording to {args.save}")
    else:
        print("Rerun viewer opened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
