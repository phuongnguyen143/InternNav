import json
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage

from utils.config import get_config
from utils.floor_pose import (
    derive_floor_calibration_from_trajectory,
    floor_2d_pose_to_action_matrix,
    floor_xy_to_world_on_plane,
    load_floor_calibration,
)
from utils.depth_codec import decode_compressed_depth
from keyframe_selection import KeyframeConfig, extract_keyframes
from utils.trajectory_io import FloorMatcher, OdomMatcher, parse_floor_trajectory_txt, parse_odom_txt
from frame_utils import (
    align_depth_to_rgb,
    copy_depth_frame_range,
    decode_rgb_compressed,
    depth_m_to_preview_bgr,
    get_frame_from_video,
    open_mp4_writer,
    ros_stamp_to_sec,
    save_depth_frame_mm,
)


def _clear_episode_dir(episode_dir: Path) -> None:
    """Remove stale keyframes and videos from a previous extraction run."""
    for pattern in ("kf_*.jpg", "kf_*.png", "rgb.mp4", "depth.mp4"):
        for path in episode_dir.glob(pattern):
            path.unlink(missing_ok=True)
    depth_frames = episode_dir / "depth_frames"
    if depth_frames.exists():
        shutil.rmtree(depth_frames)


class KeyframeExtractor(Node):
    def __init__(self):
        super().__init__("keyframe_extractor")

        self.frame_idx = 0
        self.frame_metadata = []
        self.rgb_writer = None
        self.depth_writer = None
        cfg = get_config()
        ros = cfg.ros
        keyframe_cfg = cfg.keyframe

        self.fps = float(keyframe_cfg.get("record_fps", 10.0))

        self.declare_parameter("rgb_topic", ros.get("rgb_topic"))
        self.declare_parameter("odom_topic", ros.get("odom_matched_topic"))
        self.declare_parameter("depth_topic", ros.get("depth_topic"))
        self.declare_parameter("output_dir", "./keyframe_output")
        self.declare_parameter("camera_odom_file", "")

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.output_dir = Path(self.get_parameter("output_dir").value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_odom_file = self.get_parameter("camera_odom_file").value

        self.floor_plane = None
        self.pose_frame = "floor"
        self._load_floor_plane()

        self.tmp_dir = self.output_dir / "tmp"
        self.tmp_dir.mkdir(exist_ok=True)
        self.tmp_rgb_video = self.tmp_dir / "rgb_full.mp4"
        self.tmp_depth_video = self.tmp_dir / "depth_full.mp4"
        self.tmp_depth_frames_dir = self.tmp_dir / "depth_frames"
        self.tmp_depth_frames_dir.mkdir(parents=True, exist_ok=True)

        self.kf_dir = self.output_dir / "keyframes"
        self.kf_dir.mkdir(exist_ok=True)
        self.keyframes_per_episode = int(keyframe_cfg.get("keyframes_per_episode", 10))

        self.poses = []
        kf = keyframe_cfg
        self.config = KeyframeConfig(
            sharp_turn_thresh_deg=float(kf.get("sharp_turn_thresh_deg", 25.0)),
            curvature_thresh_deg=float(kf.get("curvature_thresh_deg", 25.0)),
            curvature_window=int(kf.get("curvature_window", 10)),
            max_dist_between_keyframes=float(kf.get("max_dist_between_keyframes", 6.0)),
            min_dist_between_keyframes=float(kf.get("min_dist_between_keyframes", 3.0)),
            merge_window_frames=int(kf.get("merge_window_frames", 5)),
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.rgb_sub = Subscriber(
            self,
            CompressedImage,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.depth_sub = Subscriber(
            self,
            CompressedImage,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.odom_sub = Subscriber(
            self,
            Odometry,
            self.odom_topic,
            qos_profile=qos,
        )
        self.get_logger().info(f"Keyframe extractor ready (odom sync via {self.odom_topic})")

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=20,
            slop=float(ros.get("sync_slop_sec", 0.05)),
        )

        self.sync.registerCallback(self.synced_callback)

    def _load_floor_plane(self) -> None:
        cfg = get_config()
        cal_path = self.output_dir / cfg.floor_calibration_filename()
        if cal_path.exists():
            self.floor_plane = load_floor_calibration(cal_path)["floor_plane"]
            self.get_logger().info(f"Loaded floor calibration from {cal_path}")
            return

        traj_path = self.output_dir / cfg.floor_trajectory_filename()
        if traj_path.is_file():
            try:
                derived = derive_floor_calibration_from_trajectory(traj_path, self.output_dir)
                self.floor_plane = load_floor_calibration(derived)["floor_plane"]
                self.get_logger().info(
                    f"Derived floor calibration from {traj_path} -> {derived}"
                )
                return
            except (ValueError, OSError) as exc:
                self.get_logger().warn(f"Could not derive floor plane from {traj_path}: {exc}")

        self.get_logger().info(
            "No floor calibration; place floor_calibration.json or floor_trajectory.txt "
            "with world_x/y/z in output_dir before finalize."
        )

    @staticmethod
    def _decode_compressed_depth(depth_msg: CompressedImage):
        return decode_compressed_depth(bytes(depth_msg.data), format_hint=depth_msg.format or "")

    def _odom_to_pose(self, odom_msg: Odometry, frame_idx: int, timestamp: float) -> dict:
        q = odom_msg.pose.pose.orientation
        yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        x = float(odom_msg.pose.pose.position.x)
        y = float(odom_msg.pose.pose.position.y)
        z = float(odom_msg.pose.pose.position.z)

        pose = {
            "frame_idx": frame_idx,
            "x": x,
            "y": y,
            "z": z,
            "yaw": float(yaw),
            "timestamp": timestamp,
            "pose_frame": self.pose_frame,
        }

        return pose

    def _merge_camera_and_action_poses(self) -> None:
        """Attach camera_matrix (pose) and action_matrix (embodiment on floor)."""
        cfg = get_config()
        ros = cfg.ros
        if self.floor_plane is None:
            self._load_floor_plane()

        camera_matcher = None
        if self.camera_odom_file and Path(self.camera_odom_file).is_file():
            camera_entries = parse_odom_txt(self.camera_odom_file)
            camera_matcher = OdomMatcher(camera_entries, max_dt=float(ros.get("offline_match_max_dt", 0.5)))
            self.get_logger().info(
                f"Merging camera odom from {self.camera_odom_file} " f"({len(camera_entries)} entries)"
            )
        else:
            self.get_logger().warn("camera_odom_file not set; poses.json will lack camera_matrix.")

        floor_matcher = None
        floor_traj_path = self.output_dir / get_config().floor_trajectory_filename()
        if floor_traj_path.is_file():
            floor_entries = parse_floor_trajectory_txt(floor_traj_path)
            floor_matcher = FloorMatcher(floor_entries, max_dt=float(get_config().ros.get("offline_match_max_dt", 0.5)))
            self.get_logger().info(f"Merging floor world xyz from {floor_traj_path} ({len(floor_entries)} entries)")

        merged = 0
        for pose in self.poses:
            ts = pose["timestamp"]
            if camera_matcher is not None:
                cam = camera_matcher.find_closest(ts)
                if cam is not None:
                    pose["camera_x"] = float(cam.x)
                    pose["camera_y"] = float(cam.y)
                    pose["camera_z"] = float(cam.z)
                    pose["camera_yaw"] = float(cam.yaw)
                    pose["camera_matrix"] = cam.matrix.astype(float).tolist()
                    merged += 1

            if self.floor_plane is not None:
                world_xyz = None
                if floor_matcher is not None:
                    floor_entry = floor_matcher.find_closest(ts)
                    if floor_entry is not None:
                        if abs(floor_entry.world_x) + abs(floor_entry.world_y) + abs(floor_entry.world_z) > 1e-9:
                            world_xyz = np.array(
                                [floor_entry.world_x, floor_entry.world_y, floor_entry.world_z],
                                dtype=np.float64,
                            )
                        pose["x"] = float(floor_entry.x)
                        pose["y"] = float(floor_entry.y)
                        pose["yaw"] = float(floor_entry.yaw)
                        pose["z"] = float(floor_entry.z)

                if world_xyz is None:
                    world_xyz = floor_xy_to_world_on_plane(pose["x"], pose["y"], self.floor_plane)

                pose["world_x"] = float(world_xyz[0])
                pose["world_y"] = float(world_xyz[1])
                pose["world_z"] = float(world_xyz[2])

                action = floor_2d_pose_to_action_matrix(
                    pose["x"],
                    pose["y"],
                    pose["yaw"],
                    pose.get("z", 0.0),
                    self.floor_plane,
                )
                pose["action_matrix"] = action.tolist()

        self.get_logger().info(
            f"Merged camera poses for {merged}/{len(self.poses)} frames; "
            f"action_matrix={'yes' if self.floor_plane else 'no'}"
        )

    def synced_callback(
        self,
        rgb_msg: CompressedImage,
        depth_msg: CompressedImage,
        odom_msg: Odometry,
    ):
        rgb = decode_rgb_compressed(bytes(rgb_msg.data))
        if rgb is None:
            self.get_logger().warn("Failed to decode RGB image.")
            return

        depth = self._decode_compressed_depth(depth_msg)
        if depth is None:
            self.get_logger().warn("Failed to decode compressed depth image.")
            return

        if self.rgb_writer is None:
            h, w = rgb.shape[:2]
            self.rgb_writer = open_mp4_writer(self.tmp_rgb_video, self.fps, (w, h))
            self.depth_writer = open_mp4_writer(self.tmp_depth_video, self.fps, (w, h))
            self.get_logger().info(f"Video writers initialized ({w}x{h})")

        self.rgb_writer.write(rgb)

        depth = align_depth_to_rgb(depth, rgb)
        save_depth_frame_mm(self.tmp_depth_frames_dir / f"frame_{self.frame_idx:06d}.png", depth)
        self.depth_writer.write(depth_m_to_preview_bgr(depth))

        timestamp = ros_stamp_to_sec(rgb_msg.header.stamp)
        pose = self._odom_to_pose(odom_msg, self.frame_idx, timestamp)
        self.poses.append(pose)
        self.frame_metadata.append(
            {
                "frame_idx": self.frame_idx,
                "timestamp": timestamp,
            }
        )

        if self.frame_idx % 100 == 0:
            self.get_logger().info(f"Recorded frame {self.frame_idx}")
        self.frame_idx += 1

    def finalize(self):
        if self.rgb_writer is not None:
            self.rgb_writer.release()
        if self.depth_writer is not None:
            self.depth_writer.release()

        self.get_logger().info(f"Processing {len(self.poses)} poses...")
        self._merge_camera_and_action_poses()
        poses_json = self.output_dir / "poses.json"

        with open(poses_json, "w") as f:
            json.dump(self.poses, f, indent=2)

        self.get_logger().info(f"Saved poses: {poses_json}")
        if len(self.poses) < 2:
            self.get_logger().warn("Not enough poses to extract keyframes.")
            return

        keyframes = extract_keyframes(self.poses, self.config)

        if not keyframes:
            self.get_logger().warn("No keyframes extracted.")
            return

        episodes_dir = self.output_dir / "episodes"
        episodes_dir.mkdir(exist_ok=True)

        rgb_cap = cv2.VideoCapture(str(self.tmp_rgb_video))

        if not rgb_cap.isOpened():
            raise RuntimeError(f"Cannot open {self.tmp_rgb_video}")

        width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for episode_start_idx in range(0, len(keyframes) - 1, self.keyframes_per_episode):
            episode_id = episode_start_idx // self.keyframes_per_episode
            episode_dir = episodes_dir / f"episode_{episode_id:04d}"
            episode_dir.mkdir(exist_ok=True)
            _clear_episode_dir(episode_dir)

            episode_keyframes = keyframes[
                episode_start_idx : min(
                    episode_start_idx + self.keyframes_per_episode + 1,
                    len(keyframes),
                )
            ]

            start_frame = episode_keyframes[0].frame_idx
            end_frame = episode_keyframes[-1].frame_idx

            self.get_logger().info(f"Creating episode {episode_id} frames [{start_frame}, {end_frame}]")

            rgb_writer = open_mp4_writer(episode_dir / "rgb.mp4", self.fps, (width, height))
            episode_depth_dir = episode_dir / "depth_frames"
            episode_depth_dir.mkdir(exist_ok=True)

            rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current = start_frame
            while current <= end_frame:
                ok_rgb, rgb_frame = rgb_cap.read()
                if not ok_rgb:
                    break
                rgb_writer.write(rgb_frame)
                current += 1

            rgb_writer.release()
            copy_depth_frame_range(self.tmp_depth_frames_dir, episode_depth_dir, start_frame, end_frame)

            for local_idx, kf in enumerate(episode_keyframes):
                frame = get_frame_from_video(rgb_cap, kf.frame_idx)
                if frame is None:
                    continue
                dst = episode_dir / f"kf_{local_idx:04d}_{kf.reason}_{kf.frame_idx:06d}.jpg"
                cv2.imwrite(str(dst), frame)

        metadata = []
        for idx, kf in enumerate(keyframes):
            frame = get_frame_from_video(rgb_cap, kf.frame_idx)
            if frame is not None:
                dst = self.kf_dir / f"kf_{idx:04d}_{kf.reason}_{kf.frame_idx:06d}.jpg"
                cv2.imwrite(str(dst), frame)
            kf_meta = {
                "keyframe_idx": idx,
                "frame_idx": kf.frame_idx,
                "reason": kf.reason,
                "x": kf.pose["x"],
                "y": kf.pose["y"],
                "yaw": kf.pose["yaw"],
                "timestamp": kf.pose["timestamp"],
                "pose_frame": kf.pose.get("pose_frame", "camera"),
            }
            if "z" in kf.pose:
                kf_meta["z"] = kf.pose["z"]
            for wk in ("world_x", "world_y", "world_z"):
                if wk in kf.pose:
                    kf_meta[wk] = kf.pose[wk]
            if "action_matrix" in kf.pose:
                kf_meta["action_matrix"] = kf.pose["action_matrix"]
            metadata.append(kf_meta)

        rgb_cap.release()

        try:
            self.tmp_rgb_video.unlink(missing_ok=True)
            self.tmp_depth_video.unlink(missing_ok=True)
            if self.tmp_depth_frames_dir.exists():
                shutil.rmtree(self.tmp_depth_frames_dir)
        except Exception as e:
            self.get_logger().warn(f"Failed to remove temp files: {e}")

        json_path = self.output_dir / "keyframes.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.get_logger().info(f"Saved keyframes metadata: {json_path}")
        self.visualize(keyframes)
        self.get_logger().info(f"Done. {len(keyframes)} keyframes saved.")

    def visualize(self, keyframes):
        xs = [p["x"] for p in self.poses]
        ys = [p["y"] for p in self.poses]

        plt.figure(figsize=(10, 8))
        plt.plot(xs, ys, linewidth=1.0, alpha=0.5, label="trajectory")

        color_map = {
            "start": "green",
            "end": "red",
            "sharp_turn": "orange",
            "curvature": "yellow",
            "distance": "cyan",
        }

        for kf in keyframes:
            color = color_map.get(kf.reason, "white")
            plt.scatter(kf.pose["x"], kf.pose["y"], c=color, s=80, zorder=5, label=kf.reason)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.axis("equal")
        plt.grid(True)
        plt.title("Keyframe Trajectory")

        save_path = self.output_dir / "trajectory.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close()
        self.get_logger().info(f"Saved trajectory plot: {save_path}")


if __name__ == "__main__":
    rclpy.init()
    node = KeyframeExtractor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping node...")
    finally:
        try:
            node.finalize()
        except Exception as e:
            print(f"Finalize error: {e}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
