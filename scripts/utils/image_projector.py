import numpy as np
import cv2
import math
import os

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from utils.config import get_config
from utils.extrinsics import apply_body2optical_transform, r_body2optical_4x4
from utils.slam_ground import floor_world_from_camera_c2w

R_body2optical = r_body2optical_4x4()


@dataclass
class OdomExtrinsicEntry:
    timestamp: float  # float64, sufficient for selection
    t: np.ndarray  # (3,)  longdouble
    R: np.ndarray  # (3,3) longdouble
    T: np.ndarray  # (4,4) longdouble  T_world_cam


def load_odometry_txt(
    filepath: str,
    apply_body2optical: bool = True,
) -> List[OdomExtrinsicEntry]:
    entries = []
    with open(filepath, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]

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
        T = apply_body2optical_transform(
            T_raw.astype(np.float64), apply=apply_body2optical
        ).astype(np.longdouble)
        entries.append(
            OdomExtrinsicEntry(
                timestamp=timestamp,
                t=T[:3, 3].copy(),
                R=T[:3, :3].copy(),
                T=T,
            )
        )
        i += 6

    print(f"Loaded {len(entries)} odom entries")
    return entries


def world_points_to_pixels(
    T_world_cam: np.ndarray,
    K: np.ndarray,
    points_world: np.ndarray,
    image_shape: Tuple[int, int],
    apply_body2optical: bool = False,
    axis_perm: Tuple[int, int, int] = (0, 1, 2),
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = image_shape
    T = np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4)
    T = apply_body2optical_transform(T, apply=apply_body2optical)

    T_cam_world = np.linalg.inv(T)
    ones = np.ones((len(points_world), 1), dtype=np.float64)
    traj_homo = np.hstack([points_world.astype(np.float64), ones])
    traj_cam = (T_cam_world @ traj_homo.T).T[:, :3]

    X = traj_cam[:, axis_perm[0]]
    Y = traj_cam[:, axis_perm[1]]
    Z = traj_cam[:, axis_perm[2]]
    valid = Z > 0.01

    result = np.full((len(points_world), 2), -1.0)
    if not np.any(valid):
        return result, valid

    Xv = X[valid]
    Yv = Y[valid]
    Zv = Z[valid]
    u = K[0, 0] * (Xv / Zv) + K[0, 2]
    v = K[1, 1] * (Yv / Zv) + K[1, 2]
    pts_2d = np.stack([u, v], axis=1)

    in_frame = (
        (pts_2d[:, 0] >= 0)
        & (pts_2d[:, 0] < W)
        & (pts_2d[:, 1] >= 0)
        & (pts_2d[:, 1] < H)
    )

    valid_indices = np.where(valid)[0]
    for local_i, global_i in enumerate(valid_indices):
        if in_frame[local_i]:
            result[global_i] = pts_2d[local_i]

    return result, valid


def draw_future_path(
    image: np.ndarray,
    trajectory: Union[np.ndarray, List[dict]],
    cam_odom_entry: OdomExtrinsicEntry,
    K: np.ndarray,
    reference_world: np.ndarray,
    lookahead_m: float = 7.0,
    lookahead_s: float = 7.0,
    path_color: Tuple = (0, 255, 0),
    path_thickness: int = 3,
    apply_body2optical: bool = False,
    axis_perm: Tuple[int, int, int] = (0, 1, 2),
) -> np.ndarray:
    output = image.copy()
    current_ts = cam_odom_entry.timestamp
    ref = np.asarray(reference_world, dtype=np.float64).reshape(3)

    future_points = []
    for traj in trajectory:
        ts = float(traj["timestamp"])
        if ts < current_ts:
            continue
        if ts > current_ts + lookahead_s:
            break
        p = np.asarray(traj["traj"], dtype=np.float64)
        if np.linalg.norm(p - ref) <= lookahead_m:
            future_points.append(p)

    if len(future_points) < 2:
        return output

    pts_2d, _ = world_points_to_pixels(
        cam_odom_entry.T.astype(np.float64),
        K,
        np.array(future_points),
        image.shape[:2],
        apply_body2optical=apply_body2optical,
        axis_perm=axis_perm,
    )

    valid_pts = [(int(p[0]), int(p[1])) for p in pts_2d if p[0] >= 0]
    for i in range(len(valid_pts) - 1):
        cv2.line(
            output,
            valid_pts[i],
            valid_pts[i + 1],
            path_color,
            path_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(output, valid_pts[i], 4, path_color, -1)

    return output


class PathImageProjector:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        odom_txt_path: str,
        path_color: Tuple = (0, 255, 0),
        path_thickness: int = 3,
        lookahead: float = 1000.0,
        camera_pitch_deg: Optional[float] = None,
        ground_offset_y: Optional[float] = None,
    ):
        slam = get_config().slam_path
        self.camera_pitch_deg = float(
            slam.get("camera_pitch_deg", 30.0)
            if camera_pitch_deg is None
            else camera_pitch_deg
        )
        self.ground_offset_y = float(
            slam.get("ground_offset_y", 1.5)
            if ground_offset_y is None
            else ground_offset_y
        )

        self.K = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.path_color = path_color
        self.path_thickness = path_thickness
        self.lookahead = lookahead

        self._build_transformation_matrices()
        self.entries = load_odometry_txt(odom_txt_path, apply_body2optical=True)
        self.odom_ts = np.array([e.timestamp for e in self.entries], dtype=np.float64)
        self._build_trajectory()

    def _build_transformation_matrices(self):
        self.t_base2cam = np.array([0.1067, 0.0, 0.77566])
        pitch_camera = 0  # 10 degrees down

        def Ry(angle):
            c, s = math.cos(angle), math.sin(angle)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        R_base2cam_body = Ry(pitch_camera)
        self.R_base2cam = R_base2cam_body
        self.R_cam2base = self.R_base2cam.T
        self.t_cam2base = -self.R_cam2base @ self.t_base2cam

        self.T_base2cam = np.eye(4)
        self.T_base2cam[:3, :3] = self.R_base2cam
        self.T_base2cam[:3, 3] = self.t_base2cam

        self.T_cam2base = np.eye(4)
        self.T_cam2base[:3, :3] = self.R_cam2base
        self.T_cam2base[:3, 3] = self.t_cam2base

    def _build_trajectory(self):
        timestamps = []
        traj_pts = []
        for entry in self.entries:
            traj_pts.append(
                floor_world_from_camera_c2w(
                    entry.T.astype(np.float64),
                    self.camera_pitch_deg,
                    self.ground_offset_y,
                )
            )
            timestamps.append(entry.timestamp)

        traj_pts = np.array(traj_pts, dtype=np.float64)
        self.trajectory = np.array(
            [
                {"traj": traj_pts[i], "timestamp": timestamps[i]}
                for i in range(len(traj_pts))
            ]
        )

    @staticmethod
    def load_odometry_txt(filepath: str) -> List[OdomExtrinsicEntry]:
        return load_odometry_txt(filepath, apply_body2optical=True)

    @staticmethod
    def load_camera_info_from_bag(
        bag_path: str, topic: str = "/camera/camera/color/camera_info"
    ):
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with Reader(bag_path) as reader:
            for conn in reader.connections:
                if conn.topic != topic:
                    continue
                for _, _, rawdata in reader.messages(connections=[conn]):
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    K = np.array(msg.k).reshape(3, 3)
                    D = np.array(msg.d)
                    print("K:")
                    print(K)
                    print("D:")
                    print(D)
                    return K, D
        raise RuntimeError(f"Topic {topic} not found")

    def find_nearest_odom(self, rgb_ts: float) -> OdomExtrinsicEntry:
        idx = np.searchsorted(self.odom_ts, rgb_ts)

        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(self.entries):
            candidates.append(idx)

        best_idx = min(candidates, key=lambda j: abs(self.odom_ts[j] - rgb_ts))
        return self.entries[best_idx], abs(self.odom_ts[best_idx] - rgb_ts)

    def match_images_to_odom(
        self,
        bag_path: str,
        topic: str = "/camera/camera/color/image_raw/compressed",
        max_time_diff: float = 0.05,
    ) -> List[dict]:

        typestore = get_typestore(Stores.ROS2_HUMBLE)
        matched = []
        skipped = 0

        with Reader(bag_path) as reader:
            rgb_conns = [c for c in reader.connections if c.topic == topic]
            if not rgb_conns:
                raise RuntimeError(f"Topic {topic} not found")

            total = rgb_conns[0].msgcount
            print(f"Total images: {total}")

            for i, (_, ts_ns, raw) in enumerate(reader.messages(connections=rgb_conns)):
                rgb_ts = float(ts_ns) / 1e9

                odom_entry, time_diff = self.find_nearest_odom(rgb_ts)

                if time_diff > max_time_diff:
                    skipped += 1
                    continue

                msg = typestore.deserialize_cdr(raw, rgb_conns[0].msgtype)
                np_arr = np.frombuffer(bytes(msg.data), np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                matched.append(
                    {
                        "rgb_timestamp": rgb_ts,
                        "image": image,
                        "odom_entry": odom_entry,
                        "time_diff_ms": time_diff * 1000.0,
                    }
                )

                if i % 1000 == 0:
                    print(
                        f"  [{i:5d}/{total}]  "
                        f"rgb={rgb_ts:.6f}  "
                        f"odom={odom_entry.timestamp:.6f}  "
                        f"diff={time_diff * 1000:.3f}ms"
                    )

        print(f"Matched : {len(matched)}  |  Skipped : {skipped}")
        return matched

    def get_base_link_pose(self, cam_odom_entry: OdomExtrinsicEntry) -> np.ndarray:
        return cam_odom_entry.T.astype(np.float64) @ self.T_cam2base

    def world_to_pixels(
        self,
        traj_world: np.ndarray,
        cam_odom_entry: OdomExtrinsicEntry,
        image_shape: Optional[Tuple[int, int]] = None,
    ):
        H, W = image_shape if image_shape is not None else (720, 1280)
        return world_points_to_pixels(
            cam_odom_entry.T.astype(np.float64),
            self.K,
            traj_world,
            (H, W),
            apply_body2optical=False,
        )

    def draw_path_on_image(
        self,
        image: np.ndarray,
        cam_odom_entry: OdomExtrinsicEntry,
        lookahead_time: float = 1.0,
    ) -> np.ndarray:
        T_base_now = self.get_base_link_pose(cam_odom_entry)
        return draw_future_path(
            image=image,
            trajectory=self.trajectory,
            cam_odom_entry=cam_odom_entry,
            K=self.K,
            reference_world=T_base_now[:3, 3],
            lookahead_m=self.lookahead,
            lookahead_s=lookahead_time,
            path_color=self.path_color,
            path_thickness=self.path_thickness,
            apply_body2optical=False,
        )


if __name__ == "__main__":
    BAG_PATH = (
        "/home/lenguyen1/hoangpqn/GaussTrace/dataset/raw/scenes/BKHN_data/bkhn_round1"
    )
    ODOM_TXT = "/home/lenguyen1/hoangpqn/GaussTrace/dataset/raw/scenes/BKHN_data/bkhn_round1/odometry_bkhn_round1_point2plane.txt"
    RGB_TOPIC = "/camera/camera/color/image_raw/compressed"
    OUTPUT_DIR = "projected_frames1"

    K, D = PathImageProjector.load_camera_info_from_bag(BAG_PATH)

    projector = PathImageProjector(
        camera_matrix=K,
        dist_coeffs=D,
        odom_txt_path=ODOM_TXT,
        lookahead=7.0,
    )

    matched_frames = projector.match_images_to_odom(
        bag_path=BAG_PATH,
        topic=RGB_TOPIC,
        max_time_diff=0.05,
    )
    print(f"Matched frames: {len(matched_frames)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, frame in enumerate(matched_frames):
        image = frame["image"]
        odom_entry = frame["odom_entry"]

        result = projector.draw_path_on_image(
            image=image,
            cam_odom_entry=odom_entry,
            lookahead_time=7.0,
        )

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{i:06d}.jpg"), result)
        cv2.imshow("Path Projection", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print(f"Done. Saved {i + 1} frames to '{OUTPUT_DIR}/'")
