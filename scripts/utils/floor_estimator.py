# Vendored from GaussTrace/src/floor_estimator.py (CLI __main__ block omitted).

import numpy as np
import open3d as o3d

from utils.geometry_utils import (
    estimate_up_axis_pca,
    estimate_up_axis_voting,
    remove_outliers,
)


class FloorEstimator:
    def __init__(
        self,
        voxel_size: float = 0.1,
        bottom_ratio: float = 0.35,
        ransac_dist: float = 0.01,
        ransac_n: int = 3,
        num_iter: int = 3000,
        normal_max_angle_deg: float = 20.0,
        try_multiple_planes: int = 3,
        floor_threshold: float = 0.01,
        local_only: bool = False,
    ):
        self.voxel_size = voxel_size
        self.bottom_ratio = bottom_ratio
        self.ransac_dist = ransac_dist
        self.ransac_n = ransac_n
        self.num_iter = num_iter
        self.normal_max_angle_deg = normal_max_angle_deg
        self.try_multiple_planes = try_multiple_planes
        self.floor_threshold = floor_threshold
        self.local_only = local_only

        self.floor_plane = None
        self.floor_normal = None
        self.floor_point = None
        self.inlier_points = None
        self.scene_points = None
        self.up_axis = None

    def estimate(self, raw_points: np.ndarray) -> tuple:
        """
        Run floor estimation on a (N, 3) point cloud.
        Populates all result attributes and returns self for chaining.
        """
        points = remove_outliers(raw_points, self.voxel_size)
        if self.local_only:
            u_small, u_large = estimate_up_axis_voting(points)
        else:
            u_small, u_large = estimate_up_axis_pca(points)
        candidates = [u_small, -u_small, u_large, -u_large]

        best, best_u = None, None
        for u in candidates:
            result = self.score_up_axis(points, u)
            if result is None or result["angle_deg"] > self.normal_max_angle_deg:
                continue
            if best is None or result["score"] > best["score"]:
                best, best_u = result, u

        if best is None:
            raise RuntimeError("Floor estimation failed: no valid plane found.")

        floor_plane = best["plane"]
        floor_normal = best["normal"]
        inlier_points = best["inlier_pts"]
        floor_point = inlier_points.mean(axis=0)
        scene_points = points
        up_axis = best_u

        self._print_results(raw_points, floor_plane, floor_normal, inlier_points, scene_points)
        return (
            floor_plane,
            floor_normal,
            floor_point,
            inlier_points,
            scene_points,
            up_axis,
        )

    def estimate_local(
        self,
        raw_points: np.ndarray,
        patch_radius: float = 1.0,
        stride: float = 0.5,
        min_patch_points: int = 200,
    ) -> tuple:

        points = remove_outliers(raw_points, voxel_size=self.voxel_size)
        u_small, _u_large = estimate_up_axis_voting(points)
        up = u_small

        arb = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_ax = np.cross(arb, up)
        x_ax /= np.linalg.norm(x_ax)
        y_ax = np.cross(up, x_ax)

        coords_2d = np.stack([points @ x_ax, points @ y_ax], axis=1)

        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

        cx = np.arange(x_min + patch_radius, x_max, stride)
        cy = np.arange(y_min + patch_radius, y_max, stride)

        all_inliers = []
        all_planes = []

        for pcx in cx:
            for pcy in cy:
                dist2 = (coords_2d[:, 0] - pcx) ** 2 + (coords_2d[:, 1] - pcy) ** 2
                mask = dist2 <= patch_radius**2
                patch_pts = points[mask]

                if len(patch_pts) < min_patch_points:
                    continue

                h = patch_pts @ up
                bot_mask = h <= np.quantile(h, self.bottom_ratio)
                candidates = patch_pts[bot_mask]
                if len(candidates) < 50:
                    continue

                pcd_patch = o3d.geometry.PointCloud()
                pcd_patch.points = o3d.utility.Vector3dVector(candidates)
                try:
                    plane, inliers = pcd_patch.segment_plane(
                        distance_threshold=self.ransac_dist,
                        ransac_n=self.ransac_n,
                        num_iterations=self.num_iter,
                    )
                except Exception:
                    continue

                a, b, c, d = plane
                n = np.array([a, b, c])
                n /= np.linalg.norm(n) + 1e-12
                angle = np.degrees(np.arccos(np.clip(abs(np.dot(n, up)), -1, 1)))

                if angle > self.normal_max_angle_deg:
                    continue

                inlier_pts = candidates[inliers]
                all_inliers.append(inlier_pts)
                all_planes.append(plane)

        if not all_inliers:
            raise RuntimeError("Local RANSAC: không tìm được floor patch nào.")

        merged_inliers = np.vstack(all_inliers)

        pcd_merged = o3d.geometry.PointCloud()
        pcd_merged.points = o3d.utility.Vector3dVector(merged_inliers)
        global_plane, _ = pcd_merged.segment_plane(
            distance_threshold=self.floor_threshold,
            ransac_n=self.ransac_n,
            num_iterations=1000,
        )
        a, b, c, d = global_plane
        global_normal = np.array([a, b, c])
        global_normal /= np.linalg.norm(global_normal)
        floor_point = merged_inliers.mean(axis=0)

        return global_plane, global_normal, floor_point, merged_inliers, points, up

    def compute_basis_transform_matrix(self, floor_normal, floor_point) -> np.ndarray:
        """Return the 4×4 transform matrix mapping world → floor frame."""

        z_new = np.asarray(floor_normal, dtype=np.float64)
        z_new = z_new / (np.linalg.norm(z_new) + 1e-12)

        if abs(z_new[0]) < 0.9:
            arb = np.array([1.0, 0.0, 0.0])
        else:
            arb = np.array([0.0, 1.0, 0.0])

        x_new = np.cross(arb, z_new)
        x_new /= np.linalg.norm(x_new) + 1e-15
        y_new = np.cross(z_new, x_new)

        R = np.column_stack([x_new, y_new, z_new])

        if abs(np.linalg.det(R) - 1.0) > 1e-6:
            raise ValueError(f"Rotation matrix determinant = {np.linalg.det(R):.6f}, expected ~1.0")

        R_new_from_old = R.T

        if floor_point is None:
            t_old = np.zeros(3)
        else:
            t_old = np.asarray(floor_point, dtype=np.float64)

        T = np.eye(4)
        T[:3, :3] = R_new_from_old
        T[:3, 3] = -R_new_from_old @ t_old

        return T

    def to_floor_frame(self, scene_points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return scene_points transformed into the floor coordinate frame."""
        points_hom = np.hstack([scene_points, np.ones((len(scene_points), 1))])
        points_floor = (T @ points_hom.T).T[:, :3]
        return points_floor

    def score_up_axis(self, points: np.ndarray, up: np.ndarray):
        """
        Fit RANSAC planes to the bottom/top slice of points along direction u
        and score them by inlier count, normal alignment, and plane height.
        Returns the best result dict or None if too few candidates.
        """
        h = points @ up

        low_mask = h <= np.quantile(h, self.bottom_ratio)
        high_mask = h >= np.quantile(h, 1.0 - self.bottom_ratio)
        candidates_low = points[low_mask]
        candidates_high = points[high_mask]

        if len(candidates_low) >= 5000:
            candidates, up_score = candidates_low, up
        elif len(candidates_high) > len(candidates_low):
            candidates, up_score = candidates_high, -up
        else:
            return None

        cand_pcd = o3d.geometry.PointCloud()
        cand_pcd.points = o3d.utility.Vector3dVector(candidates)
        cur_pcd = cand_pcd
        best = None

        for _ in range(max(1, self.try_multiple_planes)):
            plane, inliers = cur_pcd.segment_plane(
                distance_threshold=self.ransac_dist,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iter,
            )
            a, b, c, d = plane
            n = np.array([a, b, c], dtype=np.float64)
            n = n / (np.linalg.norm(n) + 1e-12)

            if np.dot(n, up_score) < 0:
                n, a, b, c, d = -n, -a, -b, -c, -d

            inlier_pts = np.asarray(cur_pcd.points)[inliers]
            if len(inlier_pts) < 100:
                break

            angle = np.degrees(np.arccos(np.clip(np.dot(n, up_score), -1.0, 1.0)))
            plane_height = float(np.mean(inlier_pts @ up_score))
            score = len(inlier_pts) - 200.0 * angle - 50.0 * plane_height

            result = {
                "score": score,
                "plane": (a, b, c, d),
                "normal": n,
                "inlier_pts": inlier_pts,
                "angle_deg": angle,
                "plane_height": plane_height,
            }
            if best is None or score > best["score"]:
                best = result

            cur_pcd = cur_pcd.select_by_index(inliers, invert=True)
            if len(cur_pcd.points) < 300:
                break

        return best

    def _print_results(
        self,
        raw_points: np.ndarray,
        floor_plane: tuple,
        floor_normal: np.ndarray,
        inlier_points: np.ndarray,
        scene_points: np.ndarray,
    ) -> None:
        print(f"[load]  raw points:          {len(raw_points):,}")
        print(f"[prep]  after voxel + SOR:   {len(scene_points):,}")
        print(f"[floor] plane (a,b,c,d):     {floor_plane}")
        print(f"[floor] normal:              {floor_normal}")
        print(f"[floor] inliers:             {len(inlier_points)}")
