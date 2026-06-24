"""Smooth floor embodiment trajectories before writing floor_trajectory.txt."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.interpolate import splprep, splev, splrep

from utils.trajectory_io import FloorEntry

SMOOTH_METHODS = ("none", "moving_average", "bspline")


def _extract_arrays(entries: List[FloorEntry]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([e.x for e in entries], dtype=np.float64)
    y = np.array([e.y for e in entries], dtype=np.float64)
    yaw = np.array([e.yaw for e in entries], dtype=np.float64)
    z = np.array([e.z for e in entries], dtype=np.float64)
    return x, y, yaw, z


def _yaw_unwrap(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(yaw)


def _yaw_wrap(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw))


def _rebuild_entries(
    entries: List[FloorEntry],
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    z: np.ndarray,
) -> List[FloorEntry]:
    return [
        FloorEntry(
            timestamp=e.timestamp,
            x=float(x[i]),
            y=float(y[i]),
            yaw=float(yaw[i]),
            z=float(z[i]),
            world_x=e.world_x,
            world_y=e.world_y,
            world_z=e.world_z,
        )
        for i, e in enumerate(entries)
    ]


def _normalize_window(window: int, n: int) -> int:
    if window < 3:
        raise ValueError(f"smooth_window must be >= 3, got {window}")
    if window % 2 == 0:
        raise ValueError(f"smooth_window must be odd, got {window}")
    if n < 3:
        return 1
    return min(window, n if n % 2 == 1 else n - 1)


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge replication (avoids zero-pad endpoint artifacts)."""
    if window <= 1 or len(values) < 2:
        return values.copy()
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_moving_average(
    entries: List[FloorEntry],
    window: int = 5,
) -> List[FloorEntry]:
    n = len(entries)
    if n < 2:
        return list(entries)

    w = _normalize_window(window, n)
    if w <= 1:
        return list(entries)

    x, y, yaw, z = _extract_arrays(entries)
    yaw_uw = _yaw_unwrap(yaw)

    x_s = _moving_average_1d(x, w)
    y_s = _moving_average_1d(y, w)
    z_s = _moving_average_1d(z, w)
    yaw_s = _yaw_wrap(_moving_average_1d(yaw_uw, w))

    return _rebuild_entries(entries, x_s, y_s, yaw_s, z_s)


def smooth_bspline(
    entries: List[FloorEntry],
    bspline_s: float = 1.0,
    fallback_window: int = 5,
) -> List[FloorEntry]:
    n = len(entries)
    if n < 2:
        return list(entries)
    if n < 4:
        print(
            f"[smooth] bspline needs >= 4 points (got {n}); " f"falling back to moving_average window={fallback_window}"
        )
        return smooth_moving_average(entries, window=fallback_window)

    x, y, yaw, z = _extract_arrays(entries)
    yaw_uw = _yaw_unwrap(yaw)
    u = np.linspace(0.0, 1.0, n)
    k = min(3, n - 1)

    tck, _ = splprep([x, y, z], u=u, s=bspline_s, k=k)
    x_s, y_s, z_s = splev(u, tck)

    yaw_tck = splrep(u, yaw_uw, s=bspline_s, k=k)
    yaw_s = _yaw_wrap(splev(u, yaw_tck))

    return _rebuild_entries(entries, np.asarray(x_s), np.asarray(y_s), yaw_s, np.asarray(z_s))


def smooth_floor_trajectory(
    entries: List[FloorEntry],
    method: str,
    *,
    window: int = 5,
    bspline_s: float = 1.0,
) -> List[FloorEntry]:
    """
    Smooth floor trajectory poses while preserving timestamps.

    Args:
        entries: Raw projected floor entries.
        method: ``none``, ``moving_average``, or ``bspline``.
        window: Odd window size for moving average.
        bspline_s: Smoothing factor for B-spline (0 = interpolate, larger = smoother).
    """
    if method == "none":
        return list(entries)
    if method == "moving_average":
        return smooth_moving_average(entries, window=window)
    if method == "bspline":
        return smooth_bspline(entries, bspline_s=bspline_s, fallback_window=window)
    raise ValueError(f"Unknown smooth method {method!r}; expected one of {SMOOTH_METHODS}")


def _smooth_label(method: str, *, window: int = 5, bspline_s: float = 1.0) -> str:
    if method == "moving_average":
        return f"moving_average (window={window})"
    if method == "bspline":
        return f"bspline (s={bspline_s})"
    return method


def _pick_compare_zoom_center(
    bx: np.ndarray,
    by: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    zoom_span: float,
) -> Tuple[float, float]:
    """Pick region where before/after paths differ most."""
    uv = np.stack([bx, by], axis=1)
    delta = np.linalg.norm(np.stack([ax - bx, ay - by], axis=1), axis=1)
    half = zoom_span / 2.0
    best_score = -1.0
    best_center = (float(bx[len(bx) // 2]), float(by[len(by) // 2]))

    for i in range(len(uv)):
        u0, v0 = uv[i]
        in_window = (
            (uv[:, 0] >= u0 - half) & (uv[:, 0] <= u0 + half) & (uv[:, 1] >= v0 - half) & (uv[:, 1] <= v0 + half)
        )
        if not in_window.any():
            continue
        score = float(delta[in_window].mean())
        if score > best_score:
            best_score = score
            best_center = (float(u0), float(v0))

    return best_center


def _style_compare_axes(plot_ax) -> None:
    plot_ax.set_facecolor("#1a1a1a")
    plot_ax.set_aspect("equal")
    plot_ax.grid(True, color="#333333", linewidth=0.5)
    plot_ax.tick_params(colors="white")
    for spine in plot_ax.spines.values():
        spine.set_color("#555555")


def _draw_compare_paths(
    plot_ax,
    bx: np.ndarray,
    by: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    zoom_rect: Tuple[float, float, float, float] | None = None,
) -> None:
    """Draw before (dots + dashed) and after (solid) so both remain visible."""
    _style_compare_axes(plot_ax)
    plot_ax.set_xlim(xlim)
    plot_ax.set_ylim(ylim)

    plot_ax.plot(ax, ay, color="#4fc3f7", linewidth=1.0, alpha=0.9, zorder=2, label="after smooth")
    plot_ax.plot(bx, by, color="#ff8a65", linewidth=1.0, linestyle="--", alpha=0.95, zorder=3, label="before smooth")
    plot_ax.scatter(bx, by, s=4, c="#ffab91", edgecolors="none", alpha=0.9, zorder=4, label="before poses")

    plot_ax.scatter(bx[0], by[0], s=50, c="lime", edgecolors="white", linewidths=0.5, zorder=5)
    plot_ax.scatter(bx[-1], by[-1], s=50, c="red", edgecolors="white", linewidths=0.5, zorder=5)

    if zoom_rect is not None:
        ru0, rv0, ru1, rv1 = zoom_rect
        plot_ax.add_patch(
            Rectangle(
                (ru0, rv0),
                ru1 - ru0,
                rv1 - rv0,
                linewidth=1.5,
                edgecolor="yellow",
                facecolor="none",
                linestyle="--",
                zorder=6,
            )
        )

    plot_ax.set_title(title, color="white")
    plot_ax.set_xlabel("x (floor frame, m)", color="white")
    plot_ax.set_ylabel("y (floor frame, m)", color="white")
    plot_ax.legend(loc="upper right", facecolor="#2a2a2a", labelcolor="white", fontsize=8)


def _comparison_arrays(before: List[FloorEntry], after: List[FloorEntry]):
    bx, by, _, _ = _extract_arrays(before)
    ax, ay, _, _ = _extract_arrays(after)
    delta = np.linalg.norm(np.stack([ax - bx, ay - by], axis=1), axis=1)
    margin = 2.0
    all_x = np.concatenate([bx, ax])
    all_y = np.concatenate([by, ay])
    u_over = (all_x.min() - margin, all_x.max() + margin)
    v_over = (all_y.min() - margin, all_y.max() + margin)
    return bx, by, ax, ay, delta, u_over, v_over


def _save_static_comparison(
    before: List[FloorEntry],
    after: List[FloorEntry],
    save_path: Path,
    method: str,
    *,
    window: int,
    bspline_s: float,
    zoom_span: float,
) -> Path:
    """Dual-panel PNG: overview + auto-zoom on highest-deviation region."""
    bx, by, ax, ay, delta, u_over, v_over = _comparison_arrays(before, after)

    zoom_cu, zoom_cv = _pick_compare_zoom_center(bx, by, ax, ay, zoom_span)
    half = zoom_span / 2.0
    u_zoom = (zoom_cu - half, zoom_cu + half)
    v_zoom = (zoom_cv - half, zoom_cv + half)
    zoom_rect = (u_zoom[0], v_zoom[0], u_zoom[1], v_zoom[1])

    label = _smooth_label(method, window=window, bspline_s=bspline_s)
    fig, (ax_over, ax_zoom) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    fig.patch.set_facecolor("#121212")

    _draw_compare_paths(
        ax_over,
        bx,
        by,
        ax,
        ay,
        xlim=u_over,
        ylim=v_over,
        title=f"Overview ({label})",
        zoom_rect=zoom_rect,
    )
    _draw_compare_paths(
        ax_zoom,
        bx,
        by,
        ax,
        ay,
        xlim=u_zoom,
        ylim=v_zoom,
        title=f"Zoom where paths differ most ({zoom_span:.0f} m)",
    )
    fig.suptitle(
        f"max deviation={delta.max():.4f} m  mean={delta.mean():.4f} m",
        color="white",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(
        f"[smooth] Saved comparison plot: {save_path.resolve()}  "
        f"(max dev={delta.max():.4f} m, mean={delta.mean():.4f} m)"
    )
    return save_path


def show_smooth_comparison_interactive(
    before: List[FloorEntry],
    after: List[FloorEntry],
    method: str,
    *,
    window: int = 5,
    bspline_s: float = 1.0,
) -> None:
    """
    Open an interactive matplotlib window for before/after smooth investigation.

    Use the toolbar to pan/zoom anywhere on the path. Close the window to continue.
    """
    bx, by, ax, ay, delta, u_over, v_over = _comparison_arrays(before, after)
    label = _smooth_label(method, window=window, bspline_s=bspline_s)

    fig, plot_ax = plt.subplots(figsize=(14, 10), dpi=100)
    fig.patch.set_facecolor("#121212")
    _draw_compare_paths(
        plot_ax,
        bx,
        by,
        ax,
        ay,
        xlim=u_over,
        ylim=v_over,
        title=f"Before vs after smooth ({label})",
    )
    fig.suptitle(
        f"max deviation={delta.max():.4f} m  mean={delta.mean():.4f} m  |  " "toolbar: pan, zoom, home (reset)",
        color="white",
        y=0.98,
        fontsize=11,
    )
    fig.tight_layout()

    print(
        "\n[smooth] Interactive comparison window opened.\n"
        "  Toolbar : pan (cross arrows) | zoom (magnifier, drag a box) | home (reset view)\n"
        "  Layers  : orange dashed + dots = before | cyan solid = after\n"
        "  Close the window to finish precompute.\n"
    )
    plt.show(block=True)
    plt.close(fig)


def draw_smooth_comparison(
    before: List[FloorEntry],
    after: List[FloorEntry],
    method: str,
    *,
    save_path: Union[str, Path, None] = None,
    interactive: bool = True,
    window: int = 5,
    bspline_s: float = 1.0,
    zoom_span: float = 8.0,
) -> Path | None:
    """
    Compare floor trajectory before and after smoothing.

    By default opens an interactive window (zoom/pan). Optionally saves a static PNG.
    """
    if interactive:
        show_smooth_comparison_interactive(before, after, method, window=window, bspline_s=bspline_s)

    if save_path is not None:
        return _save_static_comparison(
            before,
            after,
            Path(save_path),
            method,
            window=window,
            bspline_s=bspline_s,
            zoom_span=zoom_span,
        )
    return None


def smooth_config_dict(method: str, *, window: int = 5, bspline_s: float = 1.0) -> dict:
    """Build metadata dict for floor_calibration.json."""
    if method == "none":
        return {"method": "none"}
    if method == "moving_average":
        return {"method": "moving_average", "window": int(window)}
    if method == "bspline":
        return {"method": "bspline", "s": float(bspline_s), "window_fallback": int(window)}
    raise ValueError(f"Unknown smooth method {method!r}")
