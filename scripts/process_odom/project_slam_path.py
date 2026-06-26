#!/usr/bin/env python3
"""Project SLAM trajectory onto RGB frames and export floor_trajectory.txt.

DROID-W odometry stores c2w poses in the OpenCV optical frame (X right,
Y down, Z forward). Pinhole projection:

    p_cam = inv(c2w) @ p_world
    u = fx * X/Z + cx,  v = fy * Y/Z + cy

Floor trajectory export (--export-floor-trajectory) uses a fixed mount pitch
heuristic: undo that pitch so optical +Z is horizontal, offset ground_offset_y
along optical +Y to reach the floor contact point, and write floor_trajectory.txt
(x, y, yaw, world_x/y/z) for the keyframe pipeline.

RGB projection (--visualize / --project-images) draws the future path from those
world points.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.config import VlnConfig, get_config, load_config
from utils.floor_plane import floor_plane_from_world_points
from utils.floor_pose import save_floor_calibration
from utils.image_projector import (
    OdomExtrinsicEntry,
    draw_future_path,
    load_odometry_txt,
)
from utils.slam_ground import (
    floor_world_from_camera_c2w,
    yaw_from_leveled_camera,
)
from utils.trajectory_io import FloorEntry, write_floor_trajectory_txt
from utils.trajectory_smooth import SMOOTH_METHODS, smooth_slam_floor_trajectory


def normalize_export_dir(path: Path, cfg: VlnConfig | None = None) -> Path:
    """Accept export directory; if user passes floor_trajectory.txt, use its parent."""
    cfg = cfg or get_config()
    if path.name == cfg.floor_trajectory_filename() or (
        path.suffix == ".txt" and path.is_file()
    ):
        print(
            f"Note: --export-floor-trajectory should be a directory; "
            f"using parent of {path}"
        )
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


def build_floor_entries_from_odom(
    entries: list[OdomExtrinsicEntry],
    *,
    camera_pitch_deg: float,
    ground_offset_y: float,
) -> list[FloorEntry]:
    """Floor trajectory from leveled-camera offset (SLAM, no PCD)."""
    world_pts = np.stack(
        [
            floor_world_from_camera_c2w(entry.T, camera_pitch_deg, ground_offset_y)
            for entry in entries
        ],
        axis=0,
    )
    origin_xy = world_pts[0, :2].copy()
    floor_entries: list[FloorEntry] = []
    for i, entry in enumerate(entries):
        wx, wy, wz = world_pts[i]
        yaw = yaw_from_leveled_camera(entry.T[:3, :3], camera_pitch_deg)
        floor_entries.append(
            FloorEntry(
                timestamp=entry.timestamp,
                x=float(wx - origin_xy[0]),
                y=float(wy - origin_xy[1]),
                yaw=yaw,
                z=float(wz),
                world_x=float(wx),
                world_y=float(wy),
                world_z=float(wz),
            )
        )
    return floor_entries


def trajectory_from_floor_entries(floor_entries: list[FloorEntry]) -> np.ndarray:
    """Build trajectory array for RGB projection from floor_trajectory world xyz."""
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


def save_floor_trajectory_plot(
    floor_entries: list[FloorEntry], save_path: Path
) -> None:
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
    camera_pitch_deg: float,
    ground_offset_y: float,
) -> tuple[int, list[OdomExtrinsicEntry], list[FloorEntry]]:
    """Write floor_trajectory.txt from SLAM camera odometry."""
    entries = load_odometry_txt(str(odom_path), apply_body2optical=apply_body2optical)
    if not entries:
        print("Error: no odometry entries loaded", file=sys.stderr)
        return 1, [], []

    floor_entries = build_floor_entries_from_odom(
        entries,
        camera_pitch_deg=camera_pitch_deg,
        ground_offset_y=ground_offset_y,
    )

    if smooth != "none":
        floor_entries = smooth_slam_floor_trajectory(
            floor_entries,
            method=smooth,
            window=smooth_window,
            bspline_s=smooth_s,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / cfg.floor_trajectory_filename()
    write_floor_trajectory_txt(traj_path, floor_entries)

    world_pts = np.stack(
        [[e.world_x, e.world_y, e.world_z] for e in floor_entries],
        axis=0,
    )
    floor_plane = floor_plane_from_world_points(world_pts)
    a, b, c, d = floor_plane
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    floor_point = world_pts.mean(axis=0)
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
            "camera_pitch_deg": camera_pitch_deg,
            "ground_offset_y": ground_offset_y,
        },
    )
    print(
        f"Exported floor trajectory: leveled-camera offset "
        f"(undo {camera_pitch_deg:.1f}° pitch, +{ground_offset_y:.2f} m on Y), "
        f"calibration plane n=[{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}] h={-d:.3f}, "
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
        description=("Export SLAM floor trajectory and/or project it onto RGB frames.")
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
        default=Path(paths["projected_frames"])
        if paths.get("projected_frames")
        else Path("projected_frames"),
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
    parser.add_argument(
        "--lookahead-m", type=float, default=float(slam.get("lookahead_m", 10.0))
    )
    parser.add_argument(
        "--lookahead-s", type=float, default=float(slam.get("lookahead_s", 10.0))
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=float(vln_cfg.ros.get("max_time_diff", 0.05)),
    )
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--ground-offset-y",
        type=float,
        default=float(slam.get("ground_offset_y", 1.5)),
        help="Floor contact offset along optical +Y after undoing mount pitch (m)",
    )
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=float(slam.get("camera_pitch_deg", 30.0)),
        help="Fixed mount pitch to undo before floor contact (deg)",
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
    floor_entries: list[FloorEntry] | None = None,
) -> int:
    """Project floor_trajectory world path onto RGB frames."""
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

    img_w, img_h, K = load_intrinsics_from_cfg(
        cfg_path, use_slam_resolution=args.slam_intrinsics
    )
    print(f"Intrinsics from {cfg_path} ({img_w}x{img_h})")
    print(f"  fx={K[0, 0]:.3f} fy={K[1, 1]:.3f} cx={K[0, 2]:.3f} cy={K[1, 2]:.3f}")

    entries = odom_entries or load_odometry_txt(
        str(args.odom), apply_body2optical=args.body2optical
    )
    if args.body2optical:
        print("Extrinsics: applying R_body2optical at load")
    else:
        print("Extrinsics: DROID optical c2w (no R_body2optical)")

    if floor_entries is None:
        floor_entries = build_floor_entries_from_odom(
            entries,
            camera_pitch_deg=args.camera_pitch_deg,
            ground_offset_y=args.ground_offset_y,
        )

    trajectory = trajectory_from_floor_entries(floor_entries)
    print(
        f"Floor path: undo {args.camera_pitch_deg:.1f}° mount pitch, "
        f"+{args.ground_offset_y:.2f} m on optical Y"
    )
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

        odom_entry, odom_idx = find_nearest_entry(
            odom_ts, entries, ts, args.max_time_diff
        )
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

    print(
        f"Done. Saved {saved} frames to {args.output} "
        f"(skipped {skipped}, trajectory points {len(trajectory)})"
    )
    return 0


def main() -> int:
    args = parse_args()

    if args.odom is None or not args.odom.exists():
        print(
            "Error: --odom not set or not found (use --scene or pass --odom)",
            file=sys.stderr,
        )
        return 1

    if args.floor_smooth_window % 2 == 0 or args.floor_smooth_window < 3:
        print(
            "Error: --floor-smooth-window must be an odd integer >= 3", file=sys.stderr
        )
        return 1

    do_export = args.export_floor_trajectory is not None
    can_project_rgb = args.frames_json.exists() and args.rgb_dir.exists()
    do_project = (
        args.project_images or (not do_export) or (args.visualize and can_project_rgb)
    )

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
            camera_pitch_deg=args.camera_pitch_deg,
            ground_offset_y=args.ground_offset_y,
        )
        if rc != 0:
            return rc

        if args.visualize and floor_entries:
            save_floor_trajectory_plot(
                floor_entries,
                export_dir / "floor_trajectory.png",
            )
            if not can_project_rgb:
                print(
                    "Note: frames-json or rgb-dir missing; "
                    "saved floor_trajectory.png only (no RGB projection)"
                )

    if do_project:
        return project_images(
            args, odom_entries=odom_entries, floor_entries=floor_entries
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
