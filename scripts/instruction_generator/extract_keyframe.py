import json
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Pose2D
from message_filters import Subscriber
from message_filters import ApproximateTimeSynchronizer
from cv_bridge import CvBridge
from utils import get_frame_from_video, extract_keyframes, KeyframeConfig


class KeyframeExtractor(Node):

    def __init__(self):
        super().__init__('keyframe_extractor')

        self.bridge = CvBridge()
        self.frame_idx = 0
        self.frame_metadata = []
        self.rgb_writer = None
        self.depth_writer = None
        self.fps = 10.0

        # params
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('odom_topic', '/odom_txt/xy_yaw')
        self.declare_parameter('depth_topic','/camera/camera/aligned_depth_to_color/image_raw/raw_depth')
        self.declare_parameter('output_dir', './keyframe_output')

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # directory declare
        self.tmp_dir = self.output_dir / "tmp"
        self.tmp_dir.mkdir(exist_ok=True)
        self.tmp_rgb_video = self.tmp_dir / "rgb_full.mp4"
        self.tmp_depth_video = self.tmp_dir / "depth_full.mp4"

        self.kf_dir = self.output_dir / "keyframes"
        self.kf_dir.mkdir(exist_ok=True)
        self.keyframes_per_episode = 30

        self.frame_idx = 0
        self.latest_pose = None
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
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.odom_sub = self.create_subscription(Pose2D, self.odom_topic, self.odom_callback, qos)
        self.get_logger().info("Keyframe extractor ready")

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=20,
            slop=0.05,
        )

        self.sync.registerCallback(self.synced_callback)

    def synced_callback(
        self,
        rgb_msg: CompressedImage,
        depth_msg: Image,
    ):
        if self.latest_pose is None:
            return
        np_arr = np.frombuffer(rgb_msg.data, np.uint8)
        rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if rgb is None:
            self.get_logger().warn("Failed to decode RGB image.")
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"Failed to decode depth image: {e}")
            return
        if depth is None:
            self.get_logger().warn("Decoded depth image is None.")
            return

        # writer
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
        depth_vis = cv2.convertScaleAbs(
            depth,
            alpha=255.0 / 10000.0
        )

        depth_vis = cv2.cvtColor(
            depth_vis,
            cv2.COLOR_GRAY2BGR
        )
        self.depth_writer.write(depth_vis)

        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

        pose = {
            "frame_idx": self.frame_idx,
            "x": float(self.latest_pose.x),
            "y": float(self.latest_pose.y),
            "yaw": float(self.latest_pose.theta),
            "timestamp": timestamp,
        }
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

    def odom_callback(self, msg):
        self.latest_pose = msg

    def finalize(self):
        if self.rgb_writer is not None:
            self.rgb_writer.release()
        if self.depth_writer is not None:
            self.depth_writer.release()

        self.get_logger().info(f"Processing {len(self.poses)} poses...")
        poses_json = self.output_dir / "poses.json"

        with open(poses_json, "w") as f:
            json.dump(
                self.poses,
                f,
                indent=2
            )

        self.get_logger().info(
            f"Saved poses: {poses_json}"
        )
        if len(self.poses) < 2:
            self.get_logger().warn("Not enough poses to extract keyframes.")
            return

        keyframes = extract_keyframes(
            self.poses,
            self.config,
        )

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

        # --------------------------------------------------
        # Episodes
        # --------------------------------------------------

        for episode_start_idx in range(
            0,
            len(keyframes) - 1,
            self.keyframes_per_episode,
        ):

            episode_id = episode_start_idx // self.keyframes_per_episode

            episode_dir = episodes_dir / f"episode_{episode_id:04d}"

            episode_dir.mkdir(exist_ok=True)

            episode_keyframes = keyframes[
                episode_start_idx : min(
                    episode_start_idx + self.keyframes_per_episode + 1,
                    len(keyframes),
                )
            ]

            start_frame = episode_keyframes[0].frame_idx

            end_frame = episode_keyframes[-1].frame_idx

            self.get_logger().info(f"Creating episode {episode_id} " f"frames [{start_frame}, {end_frame}]")

            rgb_writer = cv2.VideoWriter(
                str(episode_dir / "rgb.mp4"),
                fourcc,
                self.fps,
                (width, height),
            )

            depth_writer = cv2.VideoWriter(
                str(episode_dir / "depth.mp4"),
                fourcc,
                self.fps,
                (width, height),
            )

            rgb_cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                start_frame,
            )

            depth_cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                start_frame,
            )

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
            try:
                self.tmp_rgb_video.unlink(missing_ok=True)
                self.tmp_depth_video.unlink(missing_ok=True)
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to remove temp videos: {e}"
                )
            
            #save keyframe episodes
            for local_idx, kf in enumerate(episode_keyframes):

                frame = get_frame_from_video(
                    rgb_cap,
                    kf.frame_idx,
                )

                if frame is None:
                    continue

                dst = episode_dir / f"kf_{local_idx:04d}_" f"{kf.reason}_" f"{kf.frame_idx:06d}.jpg"

                cv2.imwrite(
                    str(dst),
                    frame,
                )

        #save global keyframe
        metadata = []

        for idx, kf in enumerate(keyframes):

            frame = get_frame_from_video(
                rgb_cap,
                kf.frame_idx,
            )

            if frame is not None:

                dst = self.kf_dir / f"kf_{idx:04d}_" f"{kf.reason}_" f"{kf.frame_idx:06d}.jpg"

                cv2.imwrite(
                    str(dst),
                    frame,
                )

            metadata.append(
                {
                    "keyframe_idx": idx,
                    "frame_idx": kf.frame_idx,
                    "reason": kf.reason,
                    "x": kf.pose["x"],
                    "y": kf.pose["y"],
                    "yaw": kf.pose["yaw"],
                    "timestamp": kf.pose["timestamp"],
                }
            )

        rgb_cap.release()
        depth_cap.release()

        json_path = self.output_dir / "keyframes.json"

        with open(json_path, "w") as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        self.get_logger().info(f"Saved keyframes metadata: {json_path}")

        self.visualize(keyframes)

        self.get_logger().info(f"Done. {len(keyframes)} keyframes saved.")

    def visualize(self, keyframes):
        xs = [p['x'] for p in self.poses]
        ys = [p['y'] for p in self.poses]

        plt.figure(figsize=(10, 8))
        plt.plot(xs, ys, linewidth=1.0, alpha=0.5, label='trajectory')

        color_map = {
            'start': 'green',
            'end': 'red',
            'sharp_turn': 'orange',
            'curvature': 'yellow',
            'distance': 'cyan',
        }

        for kf in keyframes:
            color = color_map.get(kf.reason, 'white')
            plt.scatter(kf.pose['x'], kf.pose['y'], c=color, s=80, zorder=5, label=kf.reason)

        # deduplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.axis('equal')
        plt.grid(True)
        plt.title("Keyframe Trajectory")

        save_path = self.output_dir / "trajectory.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f"Saved trajectory plot: {save_path}")


if __name__ == '__main__':
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
