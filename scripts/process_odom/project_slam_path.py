#!/usr/bin/env python3
"""Project future SLAM trajectory onto extracted RGB frames.

DROID-W odometry stores c2w poses in the OpenCV optical frame (X right,
Y down, Z forward). Pinhole projection:

    p_cam = inv(c2w) @ p_world
    u = fx * X/Z + cx,  v = fy * Y/Z + cy

The walking path drops from each camera pose by applying a fixed pitch
rotation (default 30° down), then offsetting +ground_offset_y along the
pitched camera Y axis so the line sits on the floor in the image.

Export mode (--export-floor-trajectory) writes floor_trajectory.txt for the
keyframe pipeline. DROID odometry is OpenCV optical (X right, Y down, Z forward);
export converts each pose to a navigation base frame (X forward, Y left, Z up)
via R_BODY2OPTICAL, projects the base origin onto an estimated floor plane, and
writes floor-frame x, y, yaw (positive yaw = turn left). No PCD required.

Use --visualize to also save floor_trajectory.png and project the path onto
RGB frames (same as GaussTrace/project_slam_path.py).
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.config import VlnConfig, get_config, load_config
from utils.extrinsics import nav_pose_from_camera
from utils.floor_plane import build_floor_frame, project_points_to_plane
from utils.floor_pose import save_floor_calibration, sync_floor_entry_world_xyz
from utils.image_projector import (
    OdomExtrinsicEntry,
    draw_future_path,
    load_odometry_txt,
)
from utils.slam_ground import ground_point_from_pose, rotation_x_pitch_deg
from utils.trajectory_io import FloorEntry, write_floor_trajectory_txt
from utils.trajectory_smooth import SMOOTH_METHODS, smooth_floor_trajectory


def normalize_export_dir(path: Path, cfg: VlnConfig | None = None) -> Path:
    """Accept export directory; if user passes floor_trajectory.txt, use its parent."""
    cfg = cfg or get_config()
    if path.name == cfg.floor_trajectory_filename() or (path.suffix == ".txt" and path.is_file()):
        print(f"Note: --export-floor-trajectory should be a directory; " f"using parent of {path}")
        return path.parent
    return path


def load_intrinsics_from_cfg(
    cfg_path: Path,
    use_slam_resolution: bool = False,
) -> tuple[int, int, np.ndarray]:
    """Load pinhole K from DROID-W cfg.yaml (matches visualize_traj scaling)."""
    cfg = yaml.safe_load(cfg_path.read_text())
    cam = cfg["cam"]

    width = float(cam["W"])
    height = float(cam["H"])
    fx = float(cam["fx"])
    fy = float(cam["fy"])
    cx = float(cam["cx"])
    cy = float(cam["cy"])

    if use_slam_resolution:
        w_edge = float(cam.get("W_edge", 0.0))
        h_edge = float(cam.get("H_edge", 0.0))
        width_out = float(cam["W_out"])
        height_out = float(cam["H_out"])
        width_with_edge = width_out + 2.0 * w_edge
        height_with_edge = height_out + 2.0 * h_edge
        fx *= width_with_edge / width
        fy *= height_with_edge / height
        cx = cx * width_with_edge / width - w_edge
        cy = cy * height_with_edge / height - h_edge
        img_w, img_h = int(round(width_out)), int(round(height_out))
    else:
        img_w, img_h = int(round(width)), int(round(height))

    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return img_w, img_h, K


def floor_plane_from_odom_entries(
    entries: list[OdomExtrinsicEntry],
) -> tuple[float, float, float, float]:
    """Estimate floor plane from navigation-base origins (PCA normal ~= world up)."""
    base_positions = np.stack(
        [nav_pose_from_camera(entry.T)[1] for entry in entries],
        axis=0,
    )
    centroid = base_positions.mean(axis=0)
    centered = base_positions - centroid
    if len(entries) >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        up = vh[-1].astype(np.float64)
    else:
        up = -np.asarray(entries[0].R, dtype=np.float64)[:, 1]
    if np.dot(up, -np.asarray(entries[0].R, dtype=np.float64)[:, 1]) < 0.0:
        up = -up
    up /= np.linalg.norm(up) + 1e-12
    height = float(np.median(base_positions @ up))
    return (float(up[0]), float(up[1]), float(up[2]), -height)


def yaw_on_floor_plane(R_world_nav: np.ndarray, floor_plane: tuple) -> float:
    """Heading from navigation +X (forward) projected onto the floor plane."""
    _, x_ax, y_ax, n = build_floor_frame(floor_plane)
    forward = R_world_nav[:3, 0].astype(np.float64)
    forward_proj = forward - np.dot(forward, n) * n
    norm = np.linalg.norm(forward_proj)
    if norm < 1e-9:
        forward_proj = x_ax
    else:
        forward_proj /= norm
    return float(math.atan2(np.dot(forward_proj, y_ax), np.dot(forward_proj, x_ax)))


def odom_entry_to_floor_entry(
    entry: OdomExtrinsicEntry,
    floor_plane: tuple,
) -> FloorEntry:
    """Map DROID optical c2w to floor_trajectory.txt (nav X fwd, Y left, +yaw = left)."""
    R_nav, t_nav = nav_pose_from_camera(entry.T)
    pos_proj = project_points_to_plane(t_nav.reshape(1, 3), floor_plane)[0]
    origin, x_ax, y_ax, _ = build_floor_frame(floor_plane)
    delta = pos_proj - origin
    return FloorEntry(
        timestamp=entry.timestamp,
        x=float(np.dot(delta, x_ax)),
        y=float(np.dot(delta, y_ax)),
        yaw=yaw_on_floor_plane(R_nav, floor_plane),
        z=float(pos_proj[2]),
        world_x=float(pos_proj[0]),
        world_y=float(pos_proj[1]),
        world_z=float(pos_proj[2]),
    )


def build_trajectory(
    entries: list[OdomExtrinsicEntry],
    smooth_window: int = 1,
    ground_offset_y: float = 1.5,
    camera_pitch_deg: float = 30.0,
) -> np.ndarray:
    """World positions for the ground path (pitched camera frame, +Y offset)."""
    points = []
    for entry in entries:
        points.append(
            {
                "traj": ground_point_from_pose(entry.R, entry.t, camera_pitch_deg, ground_offset_y),
                "timestamp": entry.timestamp,
            }
        )
    trajectory = np.array(points)

    if smooth_window > 1:
        coords = np.stack([trajectory[i]["traj"] for i in range(len(trajectory))])
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        for axis in range(3):
            coords[:, axis] = np.convolve(coords[:, axis], kernel, mode="same")
        for i in range(len(trajectory)):
            trajectory[i]["traj"] = coords[i]

    return trajectory


def trajectory_from_floor_entries(floor_entries: list[FloorEntry]) -> np.ndarray:
    """Build trajectory array from floor entries (not used for RGB projection)."""
    return np.array(
        [
            {
                "traj": np.array(
                    [entry.world_x, entry.world_y, entry.world_z],
                    dtype=np.float64,
                ),
                "timestamp": entry.timestamp,
            }
            for entry in floor_entries
        ]
    )


def save_floor_trajectory_plot(floor_entries: list[FloorEntry], save_path: Path) -> None:
    """Save top-down floor-frame path overview (x, y)."""
    xs = np.array([e.x for e in floor_entries], dtype=np.float64)
    ys = np.array([e.y for e in floor_entries], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#1e1e1e")
    ax.plot(xs, ys, color="cyan", linewidth=1.5, label="floor path")
    ax.scatter(xs[0], ys[0], color="lime", s=36, zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], color="red", s=36, zorder=5, label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (floor frame, m)", color="white")
    ax.set_ylabel("y (floor frame, m)", color="white")
    ax.set_title("Exported floor trajectory", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.legend(loc="upper right", facecolor="#2a2a2a", labelcolor="white")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved floor trajectory plot: {save_path}")


def export_floor_trajectory_from_odom(
    odom_path: Path,
    output_dir: Path,
    *,
    cfg: VlnConfig,
    apply_body2optical: bool,
    smooth: str,
    smooth_window: int,
    smooth_s: float,
) -> tuple[int, list[OdomExtrinsicEntry], list[FloorEntry]]:
    """Write floor_trajectory.txt from shifted camera odometry."""
    entries = load_odometry_txt(str(odom_path), apply_body2optical=apply_body2optical)
    if not entries:
        print("Error: no odometry entries loaded", file=sys.stderr)
        return 1, [], []

    floor_plane = floor_plane_from_odom_entries(entries)
    floor_entries = [odom_entry_to_floor_entry(entry, floor_plane) for entry in entries]

    if smooth != "none":
        floor_entries = smooth_floor_trajectory(
            floor_entries,
            method=smooth,
            window=smooth_window,
            bspline_s=smooth_s,
        )
        floor_entries = sync_floor_entry_world_xyz(floor_entries, floor_plane)

    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / cfg.floor_trajectory_filename()
    write_floor_trajectory_txt(traj_path, floor_entries)

    a, b, c, d = floor_plane
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    world_pts = np.stack(
        [[e.world_x, e.world_y, e.world_z] for e in floor_entries],
        axis=0,
    )
    floor_point = project_points_to_plane(world_pts.mean(axis=0).reshape(1, 3), floor_plane)[0]
    cal_path = save_floor_calibration(
        output_dir,
        floor_plane,
        n,
        floor_point,
        pcd_path=None,
        extra={
            "source": "slam_odom_export",
            "odom_path": str(Path(odom_path).resolve()),
            "apply_body2optical": apply_body2optical,
            "trajectory_smooth": smooth,
        },
    )
    print(
        f"Exported floor trajectory: nav base (X fwd, Y left), "
        f"plane n=[{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}] h={-d:.3f}, "
        f"smooth={smooth}"
    )
    print(f"  entries: {len(floor_entries)}")
    print(f"  output:  {traj_path}")
    print(f"  calibration: {cal_path}")
    return 0, entries, floor_entries


def load_frames_json(path: Path) -> list[dict]:
    records = json.loads(path.read_text())
    if not records:
        raise ValueError(f"No frames in {path}")
    return records


def find_nearest_entry(
    odom_ts: list[float],
    entries: list[OdomExtrinsicEntry],
    query_ts: float,
    max_dt: float,
) -> tuple[OdomExtrinsicEntry | None, int | None]:
    idx = bisect.bisect_left(odom_ts, query_ts)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(odom_ts):
        candidates.append(idx)
    if not candidates:
        return None, None
    best = min(candidates, key=lambda i: abs(odom_ts[i] - query_ts))
    if abs(odom_ts[best] - query_ts) > max_dt:
        return None, None
    return entries[best], best


def infer_cfg_path(odom_path: Path) -> Path | None:
    for parent in odom_path.parents:
        candidate = parent / "cfg.yaml"
        if candidate.is_file():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    import sys

    scene = None
    if "--scene" in sys.argv:
        idx = sys.argv.index("--scene")
        if idx + 1 < len(sys.argv):
            scene = sys.argv[idx + 1]
    vln_cfg = load_config(scene) if scene else get_config()
    slam = vln_cfg.slam_path
    paths = vln_cfg.get("paths", default={})

    parser = argparse.ArgumentParser(
        description=("Shift DROID-W odometry onto a ground path and/or project it onto RGB frames.")
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=scene,
        help="Scene name under utils/configs/scenes/ (loads paths and defaults)",
    )
    parser.add_argument(
        "--odom",
        type=Path,
        default=Path(paths["camera_odom"]) if paths.get("camera_odom") else None,
    )
    parser.add_argument(
        "--frames-json",
        type=Path,
        default=Path(paths["frames_json"]) if paths.get("frames_json") else None,
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=Path(paths["rgb_dir"]) if paths.get("rgb_dir") else None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(paths["projected_frames"]) if paths.get("projected_frames") else Path("projected_frames"),
    )
    parser.add_argument(
        "--export-floor-trajectory",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write floor_trajectory.txt to DIR (default: parent of floor_trajectory path when --scene set)",
    )
    parser.add_argument("--project-images", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(paths["droid_cfg"]) if paths.get("droid_cfg") else None,
        help="DROID-W cfg.yaml for intrinsics",
    )
    parser.add_argument("--slam-intrinsics", action="store_true")
    parser.add_argument("--lookahead-m", type=float, default=float(slam.get("lookahead_m", 10.0)))
    parser.add_argument("--lookahead-s", type=float, default=float(slam.get("lookahead_s", 10.0)))
    parser.add_argument("--max-time-diff", type=float, default=float(vln_cfg.ros.get("max_time_diff", 0.05)))
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=int(slam.get("smooth_window", 5)),
    )
    parser.add_argument(
        "--ground-offset-y",
        type=float,
        default=float(slam.get("ground_offset_y", 1.5)),
    )
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=float(slam.get("camera_pitch_deg", 30.0)),
    )
    parser.add_argument("--body2optical", action="store_true")
    parser.add_argument(
        "--smooth",
        choices=SMOOTH_METHODS,
        default=str(slam.get("smooth", "moving_average")),
    )
    parser.add_argument(
        "--floor-smooth-window",
        type=int,
        default=int(slam.get("floor_smooth_window", 5)),
    )
    parser.add_argument(
        "--floor-smooth-s",
        type=float,
        default=float(slam.get("floor_smooth_s", 1.0)),
    )
    args = parser.parse_args()
    args.vln_cfg = vln_cfg
    if args.export_floor_trajectory is None and paths.get("floor_trajectory"):
        args.export_floor_trajectory = Path(paths["floor_trajectory"]).parent
    return args


def project_images(
    args: argparse.Namespace,
    *,
    odom_entries: list[OdomExtrinsicEntry] | None = None,
) -> int:
    """Project ground path onto RGB frames (same trajectory model as GaussTrace)."""
    for label, path in (
        ("frames-json", args.frames_json),
        ("rgb-dir", args.rgb_dir),
    ):
        if path is None or not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            return 1

    if args.stride < 1:
        print("Error: --stride must be >= 1", file=sys.stderr)
        return 1

    cfg_path = args.config or infer_cfg_path(args.odom)
    if cfg_path is None or not cfg_path.is_file():
        print(
            "Error: could not find cfg.yaml; pass --config explicitly",
            file=sys.stderr,
        )
        return 1

    img_w, img_h, K = load_intrinsics_from_cfg(cfg_path, use_slam_resolution=args.slam_intrinsics)
    print(f"Intrinsics from {cfg_path} ({img_w}x{img_h})")
    print(f"  fx={K[0, 0]:.3f} fy={K[1, 1]:.3f} cx={K[0, 2]:.3f} cy={K[1, 2]:.3f}")

    entries = odom_entries or load_odometry_txt(str(args.odom), apply_body2optical=args.body2optical)
    if args.body2optical:
        print("Extrinsics: applying R_body2optical at load")
    else:
        print("Extrinsics: DROID optical c2w (no R_body2optical)")

    # Always use pitched-camera ground points for projection (GaussTrace parity).
    # Exported floor_trajectory world_x/y/z live on a flat horizontal plane and
    # do not reproject correctly onto images.
    trajectory = build_trajectory(
        entries,
        smooth_window=args.smooth_window,
        ground_offset_y=args.ground_offset_y,
        camera_pitch_deg=args.camera_pitch_deg,
    )
    print(f"Ground path: pitch {args.camera_pitch_deg:.1f}° then " f"+{args.ground_offset_y:.2f} m on camera Y")
    odom_ts = [e.timestamp for e in entries]
    frames = load_frames_json(args.frames_json)

    last_odom_ts = odom_ts[-1]
    frames = [f for f in frames if float(f["timestamp"]) <= last_odom_ts + 1e-6]

    args.output.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    limit = len(frames) if args.max_frames < 0 else min(args.max_frames, len(frames))

    for i, record in enumerate(frames[:limit]):
        if i % args.stride != 0:
            continue

        frame_idx = int(record["frame_idx"])
        ts = float(record["timestamp"])
        image_path = args.rgb_dir / f"frame_{frame_idx:06d}.jpg"
        if not image_path.is_file():
            skipped += 1
            continue

        odom_entry, odom_idx = find_nearest_entry(odom_ts, entries, ts, args.max_time_diff)
        if odom_entry is None or odom_idx is None:
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        if image.shape[1] != img_w or image.shape[0] != img_h:
            image = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_AREA)

        ref = np.asarray(trajectory[odom_idx]["traj"], dtype=np.float64)
        result = draw_future_path(
            image=image,
            trajectory=trajectory,
            cam_odom_entry=odom_entry,
            K=K,
            reference_world=ref,
            lookahead_m=args.lookahead_m,
            lookahead_s=args.lookahead_s,
            apply_body2optical=False,
            axis_perm=(0, 1, 2),
        )

        out_path = args.output / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), result)
        saved += 1

        if saved % 500 == 0:
            print(f"  saved {saved} frames...")

    print(f"Done. Saved {saved} frames to {args.output} " f"(skipped {skipped}, trajectory points {len(trajectory)})")
    return 0


def main() -> int:
    args = parse_args()

    if args.odom is None or not args.odom.exists():
        print("Error: --odom not set or not found (use --scene or pass --odom)", file=sys.stderr)
        return 1

    if args.floor_smooth_window % 2 == 0 or args.floor_smooth_window < 3:
        print("Error: --floor-smooth-window must be an odd integer >= 3", file=sys.stderr)
        return 1

    do_export = args.export_floor_trajectory is not None
    can_project_rgb = args.frames_json.exists() and args.rgb_dir.exists()
    do_project = args.project_images or (not do_export) or (args.visualize and can_project_rgb)

    odom_entries: list[OdomExtrinsicEntry] | None = None
    floor_entries: list[FloorEntry] | None = None

    if do_export:
        export_dir = normalize_export_dir(args.export_floor_trajectory, args.vln_cfg)
        rc, odom_entries, floor_entries = export_floor_trajectory_from_odom(
            args.odom,
            export_dir,
            cfg=args.vln_cfg,
            apply_body2optical=args.body2optical,
            smooth=args.smooth,
            smooth_window=args.floor_smooth_window,
            smooth_s=args.floor_smooth_s,
        )
        if rc != 0:
            return rc

        if args.visualize and floor_entries:
            save_floor_trajectory_plot(
                floor_entries,
                export_dir / "floor_trajectory.png",
            )
            if not can_project_rgb:
                print("Note: frames-json or rgb-dir missing; " "saved floor_trajectory.png only (no RGB projection)")

    if do_project:
        return project_images(args, odom_entries=odom_entries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
