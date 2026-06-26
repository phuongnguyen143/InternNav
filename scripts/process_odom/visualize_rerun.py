#!/usr/bin/env python3
"""Visualize floor plane, scene point cloud, and trajectories with the Rerun SDK."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import rerun as rr

from utils.config import get_config, load_config
from utils.extrinsics import apply_body2optical_transform
from utils.floor_plane import build_floor_frame, floor_xy_to_world_on_plane
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
            rr.LineStrips3D(frustum_strips, colors=[120, 180, 255], radii=0.02),
            static=True,
        )


def _log_camera_rig(
    entity: str, T: np.ndarray, *, axis_len: float, frustum_depth: float
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
    local = _optical_frustum_local(
        depth=frustum_depth,
        half_width=frustum_depth * 0.65,
        half_height=frustum_depth * 0.45,
    )
    strips = [local[[i0, i1]] for i0, i1 in _frustum_edges()]
    rr.log(
        f"{entity}/frustum",
        rr.LineStrips3D(strips, colors=[120, 180, 255], radii=0.02),
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
        help="Scene point cloud (default: pcd_path in floor_calibration.json)",
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
        default=0.35,
        help="Length (m) of RGB camera axis arrows",
    )
    parser.add_argument(
        "--camera-frustum-depth",
        type=float,
        default=0.5,
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
    args = parser.parse_args()

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

    map_points = np.empty((0, 3), dtype=np.float64)
    map_colors = np.empty((0, 3), dtype=np.uint8)
    if has_pcd:
        map_points = _load_point_cloud(
            Path(pcd_path),
            voxel_size=args.pcd_voxel,
            max_points=args.pcd_max_points,
        )
        map_colors = _color_points_by_plane_distance(
            map_points,
            floor_plane,
            inlier_thresh=args.plane_inlier_thresh,
        )

    anchor_pts = map_points
    if len(floor_pts):
        anchor_pts = (
            np.vstack([map_points, floor_pts]) if len(map_points) else floor_pts
        )
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

    if len(map_points):
        _log_point_cloud(map_points, map_colors)
    if show_floor_plane:
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
        if args.mode == "static":
            _log_static_camera_poses(
                camera_poses,
                stride=camera_stride,
                axis_len=args.camera_axis_len,
                frustum_depth=args.camera_frustum_depth,
                frustum_half_width=args.camera_frustum_depth * 0.65,
                frustum_half_height=args.camera_frustum_depth * 0.45,
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
