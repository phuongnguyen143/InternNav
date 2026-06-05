import json
import struct
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import qos_profile_sensor_data

from scipy.spatial.transform import Rotation

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from message_filters import Subscriber
from message_filters import ApproximateTimeSynchronizer
from cv_bridge import CvBridge

from constants import (
    COMPRESSED_DEPTH_HEADER_SIZE,
    DEFAULT_DEPTH_TOPIC,
    DEFAULT_DEPTH_VIS_SCALE,
    DEFAULT_KEYFRAMES_PER_EPISODE,
    DEFAULT_ODOM_MATCHED_TOPIC,
    DEFAULT_OFFLINE_MATCH_MAX_DT,
    DEFAULT_RECORD_FPS,
    DEFAULT_RGB_TOPIC,
    DEFAULT_SYNC_SLOP_SEC,
    FLOOR_CALIBRATION_FILENAME,
)
from floor_pose import load_floor_calibration, floor_2d_pose_to_action_matrix
from keyframe_selection import KeyframeConfig, extract_keyframes, get_frame_from_video
from trajectory_io import OdomMatcher, parse_odom_txt


def _clear_episode_dir(episode_dir: Path) -> None:
    """Remove stale keyframes and videos from a previous extraction run."""
    for pattern in ("kf_*.jpg", "kf_*.png", "rgb.mp4", "depth.mp4"):
        for path in episode_dir.glob(pattern):
            path.unlink(missing_ok=True)


class KeyframeExtractor(Node):
    def __init__(self):
        super().__init__("keyframe_extractor")

        self.bridge = CvBridge()
        self.frame_idx = 0
        self.frame_metadata = []
        self.rgb_writer = None
        self.depth_writer = None
        self.fps = DEFAULT_RECORD_FPS

        self.declare_parameter("rgb_topic", DEFAULT_RGB_TOPIC)
        self.declare_parameter("odom_topic", DEFAULT_ODOM_MATCHED_TOPIC)
        # Depth arrives as CompressedImage (compressedDepth transport).
        self.declare_parameter("depth_topic", DEFAULT_DEPTH_TOPIC)
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
        cal_path = self.output_dir / FLOOR_CALIBRATION_FILENAME
        if cal_path.exists():
            cal = load_floor_calibration(cal_path)
            self.floor_plane = cal["floor_plane"]
            self.get_logger().info(f"Loaded floor calibration from {cal_path}")
        else:
            self.get_logger().info(
                "No floor_calibration.json in output_dir; action_matrix "
                "will be built at finalize if calibration is added."
            )

        self.tmp_dir = self.output_dir / "tmp"
        self.tmp_dir.mkdir(exist_ok=True)
        self.tmp_rgb_video = self.tmp_dir / "rgb_full.mp4"
        self.tmp_depth_video = self.tmp_dir / "depth_full.mp4"

        self.kf_dir = self.output_dir / "keyframes"
        self.kf_dir.mkdir(exist_ok=True)
        self.keyframes_per_episode = DEFAULT_KEYFRAMES_PER_EPISODE

        self.poses = []
        self.config = KeyframeConfig()

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
            slop=DEFAULT_SYNC_SLOP_SEC,
        )

        self.sync.registerCallback(self.synced_callback)

    @staticmethod
    def _decode_compressed_depth(depth_msg: CompressedImage):
        """Decode sensor_msgs/CompressedImage from compressedDepth transport.

        The payload is not a raw PNG: the first 12 bytes are three float32
        quantization parameters, followed by the compressed image bytes.
        """
        data = bytes(depth_msg.data)
        if len(data) <= COMPRESSED_DEPTH_HEADER_SIZE:
            return None

        depth_quant_a = 0.0
        depth_quant_b = 0.0
        image_data = data

        fmt = depth_msg.format or ""
        has_depth_header = "compressedDepth" in fmt
        if not has_depth_header and len(data) > 16:
            # Live bags may omit the format hint; PNG magic starts at byte 12.
            has_depth_header = data[12:16] == b"\x89PNG"

        if has_depth_header:
            depth_quant_a, depth_quant_b, _ = struct.unpack("<fff", data[:COMPRESSED_DEPTH_HEADER_SIZE])
            image_data = data[COMPRESSED_DEPTH_HEADER_SIZE:]

        depth = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            return None

        # Inverse RLE quantization used when depth_quant_a != 0.
        if depth_quant_a != 0.0:
            depth = depth.astype(np.float32)
            valid = depth != 0
            depth_out = np.zeros_like(depth, dtype=np.float32)
            depth_out[valid] = depth_quant_a / (depth[valid].astype(np.float32) - depth_quant_b)
            depth = depth_out

        return depth

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
        if self.floor_plane is None:
            cal_path = self.output_dir / FLOOR_CALIBRATION_FILENAME
            if cal_path.exists():
                self.floor_plane = load_floor_calibration(cal_path)["floor_plane"]

        camera_matcher = None
        if self.camera_odom_file and Path(self.camera_odom_file).is_file():
            camera_entries = parse_odom_txt(self.camera_odom_file)
            camera_matcher = OdomMatcher(camera_entries, max_dt=DEFAULT_OFFLINE_MATCH_MAX_DT)
            self.get_logger().info(
                f"Merging camera odom from {self.camera_odom_file} " f"({len(camera_entries)} entries)"
            )
        else:
            self.get_logger().warn("camera_odom_file not set; poses.json will lack camera_matrix.")

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
        np_arr = np.frombuffer(rgb_msg.data, np.uint8)
        rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if rgb is None:
            self.get_logger().warn("Failed to decode RGB image.")
            return

        depth = self._decode_compressed_depth(depth_msg)
        if depth is None:
            self.get_logger().warn("Failed to decode compressed depth image.")
            return

        if self.rgb_writer is None:
            h, w = rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.rgb_writer = cv2.VideoWriter(str(self.tmp_rgb_video), fourcc, self.fps, (w, h))
            self.depth_writer = cv2.VideoWriter(str(self.tmp_depth_video), fourcc, self.fps, (w, h))

            if not self.rgb_writer.isOpened():
                raise RuntimeError(f"Cannot open RGB writer: {self.tmp_rgb_video}")
            if not self.depth_writer.isOpened():
                raise RuntimeError(f"Cannot open Depth writer: {self.tmp_depth_video}")

            self.get_logger().info(f"Video writers initialized ({w}x{h})")

        self.rgb_writer.write(rgb)

        depth_vis = cv2.convertScaleAbs(depth, alpha=255.0 / DEFAULT_DEPTH_VIS_SCALE)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        self.depth_writer.write(depth_vis)

        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
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
        depth_cap = cv2.VideoCapture(str(self.tmp_depth_video))

        if not rgb_cap.isOpened():
            raise RuntimeError(f"Cannot open {self.tmp_rgb_video}")
        if not depth_cap.isOpened():
            raise RuntimeError(f"Cannot open {self.tmp_depth_video}")

        width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

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

            rgb_writer = cv2.VideoWriter(str(episode_dir / "rgb.mp4"), fourcc, self.fps, (width, height))
            depth_writer = cv2.VideoWriter(str(episode_dir / "depth.mp4"), fourcc, self.fps, (width, height))

            rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            depth_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current = start_frame
            while current <= end_frame:
                ok_rgb, rgb_frame = rgb_cap.read()
                ok_depth, depth_frame = depth_cap.read()
                if not ok_rgb or not ok_depth:
                    break
                rgb_writer.write(rgb_frame)
                depth_writer.write(depth_frame)
                current += 1

            rgb_writer.release()
            depth_writer.release()

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
            if "action_matrix" in kf.pose:
                kf_meta["action_matrix"] = kf.pose["action_matrix"]
            metadata.append(kf_meta)

        rgb_cap.release()
        depth_cap.release()

        try:
            self.tmp_rgb_video.unlink(missing_ok=True)
            self.tmp_depth_video.unlink(missing_ok=True)
        except Exception as e:
            self.get_logger().warn(f"Failed to remove temp videos: {e}")

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
        rclpy.shutdown()
