"""Floor plane geometry helpers shared across floor_pose and SLAM export."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def project_points_to_plane(points: np.ndarray, plane: tuple) -> np.ndarray:
    """Orthographic projection onto plane ax+by+cz+d=0."""
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n_norm_sq = np.dot(n, n)
    pts = np.atleast_2d(points).astype(np.float64)
    dist = (pts @ n + d) / n_norm_sq
    return pts - dist[:, None] * n


def build_floor_frame(plane: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (origin, x_ax, y_ax, normal) for the floor plane."""
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    origin = -d / (np.dot(n, n) + 1e-12) * n

    arb = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_ax = np.cross(arb, n)
    x_ax /= np.linalg.norm(x_ax) + 1e-12
    y_ax = np.cross(n, x_ax)
    return origin, x_ax, y_ax, n


def floor_xy_to_world_on_plane(x: float, y: float, floor_plane: tuple) -> np.ndarray:
    """Map floor-frame (x, y) to the corresponding 3D point on the floor plane."""
    origin, x_ax, y_ax, _ = build_floor_frame(floor_plane)
    return origin + float(x) * x_ax + float(y) * y_ax


def floor_plane_from_world_points(points: np.ndarray) -> tuple[float, float, float, float]:
    """Fit ax+by+cz+d=0 to 3D points (e.g. floor_trajectory world_x/y/z).

    Uses PCA: the smallest-variance axis is the plane normal. Points should lie
    on or near the floor (from SLAM ground projection or PCD inliers).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.shape[0] < 3:
        raise ValueError(f"Need at least 3 points to fit a floor plane, got {pts.shape[0]}")

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)

    # Prefer +Z as up when ambiguous (common in gravity-aligned SLAM).
    if normal[2] < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal) + 1e-12

    height = float(np.median(pts @ normal))
    return (float(normal[0]), float(normal[1]), float(normal[2]), -height)
