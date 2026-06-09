import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import open3d as o3d

# Default paths for BKHN round2 + precomputed floor (keyframe_output)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FLOOR_CALIB = (
    _REPO_ROOT
    / "vln/InternNav/scripts/instruction_generator/keyframe_output/floor_calibration.json"
)
_DEFAULT_ODOM = (
    _REPO_ROOT
    / "GaussTrace/dataset/raw/scenes/BKHN_data/bkhn_round2/odometry_bkhn_round2_point2plane.txt"
)


POINT_SIZE = 0.8
BACKGROUND = [0.02, 0.02, 0.02]
MAX_POINTS = 5_000_000

R_body2optical = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

@dataclass
class OdomExtrinsicEntry:
    timestamp: float
    t: np.ndarray  # (3,)
    R: np.ndarray  # (3,3)
    T: np.ndarray  # (4,4) T_world_cam

def load_odometry_txt(filepath: str) -> List[OdomExtrinsicEntry]:
    entries = []
    with open(filepath, "r") as f:
        lines = [l.rstrip() for l in f.readlines()]

    i = 0
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        try:
            timestamp = float(lines[i].strip())
        except ValueError:
            i += 1
            continue
        if i + 4 >= len(lines):
            break
        try:
            row0 = [np.longdouble(v) for v in lines[i + 1].split()]
            row1 = [np.longdouble(v) for v in lines[i + 2].split()]
            row2 = [np.longdouble(v) for v in lines[i + 3].split()]
            row3 = [np.longdouble(v) for v in lines[i + 4].split()]
        except (ValueError, IndexError):
            i += 1
            continue

        T_raw = np.array([row0, row1, row2, row3], dtype=np.longdouble)
        T = T_raw @ R_body2optical.T

        entries.append(
            OdomExtrinsicEntry(
                timestamp=timestamp,
                t=T[:3, 3].copy(),
                R=T[:3, :3].copy(),
                T=T,
            )
        )
        i += 6

    print(
        f"Loaded {len(entries)} odom entries  "
        f"[{entries[0].timestamp:.3f} -> {entries[-1].timestamp:.3f}]"
    )
    return entries


# ── Floor projection helpers ──────────────────────────────────────────

def project_points_to_plane(points: np.ndarray, plane: tuple) -> np.ndarray:
    """
    Orthographic projection of 3D points onto plane ax+by+cz+d=0.
        p_proj = p - ( (n·p + d) / |n|² ) * n
    """
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n_norm_sq = np.dot(n, n)
    dist = (points @ n + d) / n_norm_sq   # signed distance  (N,)
    return points - dist[:, None] * n     # projected 3-D points (N, 3)


def build_floor_frame(plane: tuple):
    """
    Build an orthonormal frame on the floor plane.
    Returns (origin, x_ax, y_ax, normal).
    origin = closest point on the plane to the world origin.
    """
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n)
    origin = -d / np.dot(n, n) * n

    arb = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    x_ax = np.cross(arb, n);  x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(n, x_ax)
    return origin, x_ax, y_ax, n


def project_to_floor_2d(points_3d: np.ndarray, plane: tuple) -> np.ndarray:
    """Return (u, v) 2-D coordinates of points projected onto the floor frame."""
    origin, x_ax, y_ax, _ = build_floor_frame(plane)
    proj_3d = project_points_to_plane(points_3d, plane)
    delta = proj_3d - origin
    return np.stack([delta @ x_ax, delta @ y_ax], axis=1)


def make_floor_plane_mesh(plane: tuple, pts: np.ndarray,
                          margin: float = 5.0) -> o3d.geometry.TriangleMesh:
    """Semi-transparent quad mesh sized to cover the trajectory + margin."""
    origin, x_ax, y_ax, _ = build_floor_frame(plane)
    proj = project_points_to_plane(pts, plane)
    delta = proj - origin
    u = delta @ x_ax;  v = delta @ y_ax
    corners = np.array([
        origin + (u.min()-margin)*x_ax + (v.min()-margin)*y_ax,
        origin + (u.max()+margin)*x_ax + (v.min()-margin)*y_ax,
        origin + (u.max()+margin)*x_ax + (v.max()+margin)*y_ax,
        origin + (u.min()-margin)*x_ax + (v.max()+margin)*y_ax,
    ])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
    mesh.paint_uniform_color([0.15, 0.45, 0.75])
    mesh.compute_vertex_normals()
    return mesh


def make_normal_arrow(plane: tuple) -> o3d.geometry.TriangleMesh:
    """Yellow arrow showing the floor-plane normal direction."""
    origin, _, _, n = build_floor_frame(plane)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.05, cone_radius=0.10,
        cylinder_height=1.5,  cone_height=0.4,
    )
    z = np.array([0., 0., 1.])
    axis_rot = np.cross(z, n)
    sin_a = np.linalg.norm(axis_rot)
    cos_a = float(np.dot(z, n))
    if sin_a > 1e-6:
        axis_rot /= sin_a
        K_skew = np.array([[ 0,           -axis_rot[2],  axis_rot[1]],
                            [ axis_rot[2],  0,           -axis_rot[0]],
                            [-axis_rot[1],  axis_rot[0],  0          ]])
        R_arr = np.eye(3) + sin_a * K_skew + (1 - cos_a) * K_skew @ K_skew
    else:
        R_arr = np.eye(3) if cos_a > 0 else -np.eye(3)
    T_arr = np.eye(4)
    T_arr[:3, :3] = R_arr
    T_arr[:3, 3]  = origin
    arrow.transform(T_arr)
    arrow.paint_uniform_color([1.0, 0.8, 0.0])
    return arrow


# ── Existing helpers ──────────────────────────────────────────────────

def pose_color(idx: int, total: int) -> list:
    t = idx / max(total - 1, 1)
    return [t, max(1.0 - abs(2 * t - 1), 0.0), 1.0 - t]


def make_trajectory_line(entries: List[OdomExtrinsicEntry],
                         projected_pts: np.ndarray = None) -> o3d.geometry.LineSet:
    positions = (projected_pts.astype(np.float64) if projected_pts is not None
                 else np.array([e.t.astype(np.float64) for e in entries]))
    n = len(positions)
    ls = o3d.geometry.LineSet()
    ls.points  = o3d.utility.Vector3dVector(positions)
    ls.lines   = o3d.utility.Vector2iVector([[i, i+1] for i in range(n-1)])
    ls.colors  = o3d.utility.Vector3dVector([pose_color(i, n-1) for i in range(n-1)])
    return ls


def make_frustum(T_world_cam, K, img_w, img_h, scale, color):
    cam = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=img_w, view_height_px=img_h,
        intrinsic=K,
        extrinsic=np.linalg.inv(T_world_cam.astype(np.float64)),
        scale=scale,
    )
    cam.paint_uniform_color(color)
    return cam


# ── Top-down jitter map ───────────────────────────────────────────────

def entries_to_floor_uv(
    entries: List[OdomExtrinsicEntry],
    floor_plane: Optional[tuple] = None,
) -> np.ndarray:
    """Project all poses to floor (u, v); fall back to world XY if no plane."""
    pts_3d = np.array([e.t.astype(np.float64) for e in entries])
    if floor_plane is not None:
        return project_to_floor_2d(pts_3d, floor_plane)
    return pts_3d[:, :2].copy()


def load_pcd_floor_uv(
    pcd_path: str,
    floor_plane: tuple,
    max_pts: int = 200_000,
) -> np.ndarray:
    """Load PCD, project to floor frame, and downsample for scatter background."""
    print(f"[topdown] Loading PCD for background: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    print(f"  {len(pts):,} points")
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        print(f"  downsampled to {max_pts:,}")
    return project_to_floor_2d(pts, floor_plane)


def _smooth_path_3pt(uv: np.ndarray) -> np.ndarray:
    """Simple 3-point moving average; endpoints unchanged."""
    if len(uv) < 3:
        return uv.copy()
    smooth = uv.copy()
    smooth[1:-1] = (uv[:-2] + uv[1:-1] + uv[2:]) / 3.0
    return smooth


def compute_jitter_stats(uv: np.ndarray) -> dict:
    """Quantify 2D trajectory smoothness from floor-projected poses."""
    if len(uv) < 2:
        return {
            "step_disp_mean": 0.0,
            "step_disp_std": 0.0,
            "lateral_std": 0.0,
            "max_step_jump": 0.0,
        }

    steps = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    smooth = _smooth_path_3pt(uv)

    lateral = np.empty(len(uv), dtype=np.float64)
    lateral[0] = np.linalg.norm(uv[0] - smooth[0])
    lateral[-1] = np.linalg.norm(uv[-1] - smooth[-1])
    for i in range(1, len(uv) - 1):
        seg = smooth[i + 1] - smooth[i - 1]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-9:
            lateral[i] = np.linalg.norm(uv[i] - smooth[i])
        else:
            t = np.dot(uv[i] - smooth[i - 1], seg) / (seg_len * seg_len)
            t = np.clip(t, 0.0, 1.0)
            proj = smooth[i - 1] + t * seg
            lateral[i] = np.linalg.norm(uv[i] - proj)

    return {
        "step_disp_mean": float(steps.mean()),
        "step_disp_std": float(steps.std()),
        "lateral_std": float(lateral.std()),
        "max_step_jump": float(steps.max()),
    }


def pick_jitter_zoom_center(uv: np.ndarray, zoom_span: float) -> Tuple[float, float]:
    """
    Pick the trajectory region with highest step-displacement variance.
    Jitter is easiest to see on nominally straight segments at high zoom.
    """
    if len(uv) < 2:
        return float(uv[0, 0]), float(uv[0, 1])

    steps = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    half = zoom_span / 2.0
    best_score = -1.0
    best_center = (float(uv[len(uv) // 2, 0]), float(uv[len(uv) // 2, 1]))

    for i in range(len(uv)):
        u0, v0 = uv[i]
        in_window = (
            (uv[:, 0] >= u0 - half) & (uv[:, 0] <= u0 + half)
            & (uv[:, 1] >= v0 - half) & (uv[:, 1] <= v0 + half)
        )
        if not in_window.any():
            continue
        idx = np.where(in_window)[0]
        if len(idx) < 2:
            continue
        local_steps = steps[max(0, idx[0] - 1) : min(len(steps), idx[-1] + 1)]
        if len(local_steps) == 0:
            continue
        score = float(local_steps.var())
        if score > best_score:
            best_score = score
            best_center = (float(u0), float(v0))

    return best_center


def _draw_topdown_panel(
    ax,
    uv: np.ndarray,
    pcd_uv: Optional[np.ndarray],
    ulim: Tuple[float, float],
    vlim: Tuple[float, float],
    title: str,
    zoom_rect: Optional[Tuple[float, float, float, float]] = None,
):
    """Draw PCD background, trajectory, pose dots, and optional zoom rectangle."""
    ax.set_facecolor("#1a1a1a")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#555555")

    if pcd_uv is not None and len(pcd_uv) > 0:
        mask = (
            (pcd_uv[:, 0] >= ulim[0]) & (pcd_uv[:, 0] <= ulim[1])
            & (pcd_uv[:, 1] >= vlim[0]) & (pcd_uv[:, 1] <= vlim[1])
        )
        pts = pcd_uv[mask]
        if len(pts) > 0:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=0.3, c="#888888", alpha=0.15, linewidths=0, rasterized=True,
            )

    ax.plot(uv[:, 0], uv[:, 1], color="#4fc3f7", linewidth=0.8, alpha=0.9, zorder=3)
    ax.scatter(uv[:, 0], uv[:, 1], s=2, c="cyan", linewidths=0, zorder=4)
    ax.scatter(uv[0, 0], uv[0, 1], s=60, c="lime", edgecolors="white",
               linewidths=0.5, zorder=5, label="start")
    ax.scatter(uv[-1, 0], uv[-1, 1], s=60, c="red", edgecolors="white",
               linewidths=0.5, zorder=5, label="end")

    if zoom_rect is not None:
        ru0, rv0, ru1, rv1 = zoom_rect
        rect = Rectangle(
            (ru0, rv0), ru1 - ru0, rv1 - rv0,
            linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--", zorder=6,
        )
        ax.add_patch(rect)

    ax.set_xlim(ulim)
    ax.set_ylim(vlim)
    ax.set_aspect("equal")
    ax.grid(True, color="#333333", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8, facecolor="#2a2a2a", labelcolor="white")


def _default_topdown_save_path(floor_calibration: Optional[str]) -> Path:
    if floor_calibration:
        return Path(floor_calibration).parent / "traj_topdown.png"
    return Path.cwd() / "traj_topdown.png"


def visualize_topdown(
    entries: List[OdomExtrinsicEntry],
    floor_plane: Optional[tuple] = None,
    pcd_path: Optional[str] = None,
    zoom_span: float = 5.0,
    save_path: Optional[str] = None,
    pcd_max_pts: int = 200_000,
    block: bool = True,
):
    """Two-panel top-down map: overview + high-magnification jitter view."""
    if floor_plane is None:
        print("[topdown] WARNING: no floor plane; using XY projection")

    uv = entries_to_floor_uv(entries, floor_plane)
    stats = compute_jitter_stats(uv)
    print(
        f"\n[topdown] jitter stats — "
        f"step mean={stats['step_disp_mean']:.4f} m  "
        f"step std={stats['step_disp_std']:.4f} m  "
        f"lateral std={stats['lateral_std']:.4f} m  "
        f"max jump={stats['max_step_jump']:.4f} m"
    )

    pcd_uv = None
    if pcd_path is not None and floor_plane is not None:
        pcd_uv = load_pcd_floor_uv(pcd_path, floor_plane, max_pts=pcd_max_pts)
    elif pcd_path is not None:
        print("[topdown] Skipping PCD background (floor plane required for projection)")

    margin = 2.0
    u_over = (uv[:, 0].min() - margin, uv[:, 0].max() + margin)
    v_over = (uv[:, 1].min() - margin, uv[:, 1].max() + margin)

    zoom_cu, zoom_cv = pick_jitter_zoom_center(uv, zoom_span)
    half = zoom_span / 2.0
    u_zoom = (zoom_cu - half, zoom_cu + half)
    v_zoom = (zoom_cv - half, zoom_cv + half)
    zoom_rect = (u_zoom[0], v_zoom[0], u_zoom[1], v_zoom[1])

    fig, (ax_over, ax_zoom) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    fig.patch.set_facecolor("#121212")

    _draw_topdown_panel(
        ax_over, uv, pcd_uv, u_over, v_over,
        title="Overview (full trajectory)",
        zoom_rect=zoom_rect,
    )
    _draw_topdown_panel(
        ax_zoom, uv, pcd_uv, u_zoom, v_zoom,
        title=f"Jitter zoom ({zoom_span:.1f} m viewport)",
    )

    fig.tight_layout()
    out_path = Path(save_path) if save_path else None
    if out_path is None:
        out_path = _default_topdown_save_path(None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[topdown] Saved figure: {out_path.resolve()}")

    if block:
        print("[topdown] Close the matplotlib window to continue.")
        plt.show(block=True)
    else:
        print("[topdown] Opening 2D map window (non-blocking)")
        plt.ion()
        plt.show(block=False)
        fig.canvas.flush_events()
        plt.pause(0.5)


# ── Main visualizer ───────────────────────────────────────────────────

def visualize_open3d(
    entries: List[OdomExtrinsicEntry],
    K: np.ndarray,
    img_w: int, img_h: int,
    step: int, scale: float,
    pcd_path: str = None,
    floor_plane: tuple = None,   # (a, b, c, d)
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Camera Poses + Map", width=1600, height=900)

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # ── Point cloud ──
    if pcd_path is not None:
        print(f"Loading PCD: {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)
        print(f"  {len(pts):,} points")
        if len(pts) > MAX_POINTS:
            idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
            pts = pts[idx]
            pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
        vis.add_geometry(pcd)

    subset = entries[::step]

    # ── Original trajectory (dim gray) ──
    traj_orig = make_trajectory_line(subset)
    traj_orig.paint_uniform_color([0.35, 0.35, 0.35])
    vis.add_geometry(traj_orig)

    # ── Floor projection ──
    if floor_plane is not None:
        a, b, c, d = floor_plane
        print(f"\n[floor] plane  : {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

        orig_pts = np.array([e.t.astype(np.float64) for e in subset])
        proj_pts = project_points_to_plane(orig_pts, floor_plane)

        # 1. Projected trajectory (full color gradient)
        vis.add_geometry(make_trajectory_line(subset, projected_pts=proj_pts))

        # 2. Vertical drop lines: original → projected
        all_pts   = np.concatenate([orig_pts, proj_pts], axis=0)
        N         = len(orig_pts)
        drop_ls   = o3d.geometry.LineSet()
        drop_ls.points  = o3d.utility.Vector3dVector(all_pts)
        drop_ls.lines   = o3d.utility.Vector2iVector([[i, i+N] for i in range(N)])
        drop_ls.colors  = o3d.utility.Vector3dVector([[0.25, 0.25, 0.25]] * N)
        vis.add_geometry(drop_ls)

        # 3. Floor plane mesh
        vis.add_geometry(make_floor_plane_mesh(floor_plane, orig_pts, margin=3.0))

        # 4. Normal arrow
        vis.add_geometry(make_normal_arrow(floor_plane))

        # 5. Stats
        uv = project_to_floor_2d(orig_pts, floor_plane)
        path_len = np.linalg.norm(np.diff(proj_pts, axis=0), axis=1).sum()
        print(f"[floor] 2D extent — U: [{uv[:,0].min():.2f}, {uv[:,0].max():.2f}]  "
              f"V: [{uv[:,1].min():.2f}, {uv[:,1].max():.2f}]")
        print(f"[floor] Projected path length : {path_len:.2f} m")
        heights = orig_pts @ np.array([a,b,c]) / math.sqrt(a*a+b*b+c*c)
        print(f"[floor] Height above plane    : "
              f"min={heights.min():.3f} m  max={heights.max():.3f} m  "
              f"mean={heights.mean():.3f} m")

    # ── Render options ──
    opt = vis.get_render_option()
    opt.background_color = np.array(BACKGROUND, dtype=np.float64)
    opt.point_size = POINT_SIZE
    opt.light_on   = False

    all_t  = np.array([e.t.astype(np.float64) for e in entries])
    center = all_t.mean(axis=0)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.35)
    ctr.set_front([-0.5, -0.8, 0.5])
    ctr.set_lookat(center)
    ctr.set_up([0, 0, 1])

    print("\nControls:  Mouse drag — rotate  |  Scroll — zoom  |  Ctrl+drag — pan")
    print("           S — screenshot  |  Q / Esc — quit\n")

    vis.run()
    vis.destroy_window()


def load_floor_calibration_json(path: str | Path) -> dict:
    """Load floor_plane (a,b,c,d) and optional pcd_path from precompute output."""
    data = json.loads(Path(path).read_text())
    out = {
        "floor_plane": tuple(float(x) for x in data["floor_plane"]),
        "pcd_path": data.get("pcd_path"),
        "floor_normal": data.get("floor_normal"),
        "floor_point": data.get("floor_point"),
    }
    a, b, c, d = out["floor_plane"]
    print(
        f"[calib] Loaded floor plane from {path}\n"
        f"        (a,b,c,d) = ({a:.6f}, {b:.6f}, {c:.6f}, {d:.6f})"
    )
    if out.get("pcd_path"):
        print(f"        pcd_path  = {out['pcd_path']}")
    return out


def resolve_floor_plane_and_pcd(
    floor_calibration: Optional[str],
    floor_plane_cli: Optional[List[float]],
    pcd_cli: Optional[str],
) -> Tuple[Optional[tuple], Optional[str]]:
    floor_plane = None
    pcd_path = pcd_cli

    if floor_calibration:
        cal_path = Path(floor_calibration)
        if not cal_path.is_file():
            raise FileNotFoundError(f"floor_calibration not found: {cal_path}")
        cal = load_floor_calibration_json(cal_path)
        floor_plane = cal["floor_plane"]
        if pcd_path is None and cal.get("pcd_path"):
            pcd_path = cal["pcd_path"]
    elif floor_plane_cli is not None:
        floor_plane = tuple(floor_plane_cli)

    return floor_plane, pcd_path


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize camera odometry with optional floor plane overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (BKHN round2 + keyframe_output calibration):\n"
            "  python scene_viz.py \\\n"
            f"    --floor_calibration {_DEFAULT_FLOOR_CALIB}\n"
            f"    --odom {_DEFAULT_ODOM}\n"
            "  # pcd is read from floor_calibration.json if --pcd is omitted\n"
            "\n"
            "  python scene_viz.py --topdown --topdown_zoom_m 3.0\n"
        ),
    )
    parser.add_argument(
        "--odom",
        default=str(_DEFAULT_ODOM) if _DEFAULT_ODOM.is_file() else None,
        help="Camera odometry .txt (4x4 per timestamp)",
    )
    parser.add_argument("--pcd", default=None, help="Scene .pcd (default: from calibration JSON)")
    parser.add_argument(
        "--floor_calibration",
        default=str(_DEFAULT_FLOOR_CALIB) if _DEFAULT_FLOOR_CALIB.is_file() else None,
        help="floor_calibration.json from precompute_floor_trajectory.py",
    )
    parser.add_argument("--step", type=int, default=5, help="Draw every Nth pose")
    parser.add_argument("--scale", type=float, default=0.15, help="Frustum scale (m)")
    parser.add_argument("--img_w", type=int, default=1280)
    parser.add_argument("--img_h", type=int, default=720)
    parser.add_argument("--fx", type=float, default=647.04)
    parser.add_argument("--fy", type=float, default=646.40)
    parser.add_argument("--cx", type=float, default=637.30)
    parser.add_argument("--cy", type=float, default=370.86)
    parser.add_argument(
        "--floor_plane",
        type=float,
        nargs=4,
        metavar=("A", "B", "C", "D"),
        default=None,
        help=(
            "Manual floor plane ax+by+cz+d=0 (overrides --floor_calibration). "
            "BKHN precomputed example:\n"
            "  --floor_plane -0.033709 0.012467 0.999354 3.160882"
        ),
    )
    parser.add_argument(
        "--no_floor",
        action="store_true",
        help="Do not load floor plane (ignore --floor_calibration)",
    )
    parser.add_argument(
        "--topdown",
        action="store_true",
        help="Show 2D top-down jitter map (always saves PNG; blocks until window closed)",
    )
    parser.add_argument(
        "--topdown_only",
        action="store_true",
        help="Only show/save the 2D top-down map; skip Open3D viewer",
    )
    parser.add_argument(
        "--topdown_zoom_m",
        type=float,
        default=5.0,
        help="Jitter zoom panel viewport size in meters (smaller = more jitter visible)",
    )
    parser.add_argument(
        "--topdown_save",
        default=None,
        help="Optional PNG output path for the top-down figure",
    )
    parser.add_argument(
        "--topdown_pcd_max",
        type=int,
        default=200_000,
        help="Max PCD points for top-down background scatter",
    )
    args = parser.parse_args()

    if not args.odom:
        parser.error("--odom is required (or set a valid default BKHN path)")

    K = np.array(
        [[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
        dtype=np.float64,
    )

    floor_cal = None if args.no_floor else args.floor_calibration
    floor_plane, pcd_path = resolve_floor_plane_and_pcd(
        floor_cal, args.floor_plane, args.pcd
    )

    entries = load_odometry_txt(args.odom)

    if args.topdown or args.topdown_only:
        save_path = args.topdown_save
        if save_path is None and floor_cal:
            save_path = str(_default_topdown_save_path(floor_cal))
        visualize_topdown(
            entries,
            floor_plane=floor_plane,
            pcd_path=pcd_path,
            zoom_span=args.topdown_zoom_m,
            save_path=save_path,
            pcd_max_pts=args.topdown_pcd_max,
            block=True,
        )

    if not args.topdown_only:
        visualize_open3d(
            entries,
            K,
            img_w=args.img_w,
            img_h=args.img_h,
            step=args.step,
            scale=args.scale,
            pcd_path=pcd_path,
            floor_plane=floor_plane,
        )


if __name__ == "__main__":
    main()