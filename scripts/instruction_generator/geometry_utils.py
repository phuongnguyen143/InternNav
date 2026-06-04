# Vendored from GaussTrace/src/utils/geometry.py and utils/common.py (remove_outliers).

from __future__ import annotations

from typing import Tuple

import numpy as np
import open3d as o3d


def remove_outliers(raw_points: np.ndarray, voxel_size: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points)
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    return np.asarray(pcd.points)


def estimate_up_axis_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """PCA-based up-axis candidates (smallest- and largest-variance eigenvectors)."""
    pts = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    u_small = eigvecs[:, np.argmin(eigvals)]
    u_large = eigvecs[:, np.argmax(eigvals)]
    u_small /= np.linalg.norm(u_small) + 1e-12
    u_large /= np.linalg.norm(u_large) + 1e-12
    return u_small, u_large


def estimate_up_axis_voting(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normal-voting up-axis estimate; better for large outdoor scans."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.voxel_down_sample(voxel_size=0.3)

    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    normals = np.asarray(pcd_down.normals)

    pts_centered = points - points.mean(axis=0)
    cov = np.cov(pts_centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    candidates_init = [eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]]

    best_u, best_votes = None, -1
    for u_init in candidates_init:
        u_init = u_init / (np.linalg.norm(u_init) + 1e-12)
        dots = np.abs(normals @ u_init)
        votes = np.sum(dots > np.cos(np.radians(20)))

        if votes > best_votes:
            best_votes = votes
            aligned = normals[dots > np.cos(np.radians(20))]
            signs = np.sign(aligned @ u_init)
            aligned = aligned * signs[:, None]
            best_u = aligned.mean(axis=0)
            best_u /= np.linalg.norm(best_u) + 1e-12

    return best_u, -best_u
