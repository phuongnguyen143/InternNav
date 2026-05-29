import json
import shutil
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

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose2D


@dataclass
class KeyframeConfig:
    sharp_turn_thresh_deg = (40.0,)
    curvature_thresh_deg = (40.0,)
    max_dist_between_keyframes = (8.0,)
    min_dist_between_keyframes = (4.0,)
    merge_window_frames = (10,)


@dataclass
class KeyframeResult:

    frame_idx: int
    reason: str
    pose: dict

    delta_yaw_deg: float = 0.0
    accumulated_yaw_deg: float = 0.0
    dist_from_last: float = 0.0


def normalize_angle(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def delta_yaw_deg(a, b):
    return np.degrees(normalize_angle(b - a))


def euclidean_dist(a, b):
    return np.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2)


def merge_close_keyframes(keyframes, window):

    if len(keyframes) <= 2:
        return keyframes

    priority = {
        'start': 0,
        'end': 0,
        'sharp_turn': 1,
        'curvature': 2,
        'distance': 3,
    }

    merged = [keyframes[0]]
    for kf in keyframes[1:]:
        last = merged[-1]
        if (kf.frame_idx - last.frame_idx) <= window:
            if priority.get(kf.reason, 99) < priority.get(last.reason, 99):
                merged[-1] = kf
        else:
            merged.append(kf)

    return merged


def extract_keyframes(poses, config, verbose=True):
    if len(poses) < 2:
        return []

    keyframes = []
    keyframes.append(KeyframeResult(frame_idx=poses[0]['frame_idx'], reason='start', pose=poses[0]))

    last_kf_pose = poses[0]
    delta_yaws = [0.0]

    for i in range(1, len(poses)):
        dyaw = delta_yaw_deg(poses[i - 1]['yaw'], poses[i]['yaw'])
        delta_yaws.append(dyaw)

    for i in range(1, len(poses) - 1):
        pose = poses[i]
        dist = euclidean_dist(pose, last_kf_pose)

        if dist < config.min_dist_between_keyframes:
            continue

        reasons = []

        abs_delta = abs(delta_yaws[i])

        # sharp turn
        if abs_delta >= config.sharp_turn_thresh_deg:
            reasons.append(('sharp_turn', abs_delta))

        # curvature
        window_start = max(0, i - config.curvature_window)

        accum = sum(abs(delta_yaws[j]) for j in range(window_start, i + 1))

        if accum >= config.curvature_thresh_deg:
            reasons.append(('curvature', accum))

        # distance
        if dist >= config.max_dist_between_keyframes:
            reasons.append(('distance', dist))

        if len(reasons) > 0:

            priority = ['sharp_turn', 'curvature', 'distance']

            best_reason = next((r for p in priority for r, _ in reasons if r == p), reasons[0][0])

            kf = KeyframeResult(
                frame_idx=pose['frame_idx'],
                reason=best_reason,
                pose=pose,
                delta_yaw_deg=delta_yaws[i],
                accumulated_yaw_deg=accum,
                dist_from_last=dist,
            )

            keyframes.append(kf)

            last_kf_pose = pose

            if verbose:
                print(
                    f"[KF] "
                    f"frame={pose['frame_idx']:6d} | "
                    f"{best_reason:12s} | "
                    f"Δyaw={delta_yaws[i]:+6.1f}° | "
                    f"accum={accum:6.1f}° | "
                    f"dist={dist:.2f}m"
                )

    keyframes.append(KeyframeResult(frame_idx=poses[-1]['frame_idx'], reason='end', pose=poses[-1]))
    keyframes = merge_close_keyframes(keyframes, config.merge_window_frames)

    return keyframes


class KeyframeExtractor(Node):

    def __init__(self):

        super().__init__('keyframe_extractor')

        # params
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('odom_topic', '/odom_txt/xy_yaw')
        self.declare_parameter('output_dir', './keyframe_output')

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.output_dir / "all_frames"
        self.kf_dir = self.output_dir / "keyframes"

        self.raw_dir.mkdir(exist_ok=True)
        self.kf_dir.mkdir(exist_ok=True)

        self.frame_idx = 0
        self.latest_pose = Pose2D()
        self.poses = []
        self.config = KeyframeConfig()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.rgb_sub = self.create_subscription(CompressedImage, self.rgb_topic, self.rgb_callback, qos)
        self.odom_sub = self.create_subscription(Pose2D, self.odom_topic, self.odom_callback, qos)
        self.get_logger().info("Keyframe extractor ready")

    def odom_callback(self, msg):
        self.latest_pose = msg

    def rgb_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return
        frame_path = self.raw_dir / f"{self.frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), image)

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        pose = {
            "frame_idx": self.frame_idx,
            "x": float(self.latest_pose.x),
            "y": float(self.latest_pose.y),
            "yaw": float(self.latest_pose.theta),
            "timestamp": timestamp,
        }

        self.poses.append(pose)
        if self.frame_idx % 50 == 0:
            self.get_logger().info(f"Saved frame {self.frame_idx}")

        self.frame_idx += 1

    def finalize(self):
        self.get_logger().info(f"Processing {len(self.poses)} poses...")
        if len(self.poses) < 2:
            self.get_logger().warn("Not enough poses")
            return

        keyframes = extract_keyframes(self.poses, self.config, verbose=True)
        episodes_dir = self.output_dir / "episodes"
        episodes_dir.mkdir(exist_ok=True)

        keyframes_per_episode = 30

        for episode_idx in range(0, len(keyframes) - 1, keyframes_per_episode):
            episode_id = episode_idx // keyframes_per_episode
            episode_dir = episodes_dir / f"episode_{episode_id:04d}"
            episode_dir.mkdir(exist_ok=True)
            episode_keyframes = keyframes[episode_idx : min(episode_idx + keyframes_per_episode + 1, len(keyframes))]
            self.get_logger().info(f"Creating episode {episode_id}")

            # create episode video
            video_path = episode_dir / "episode.mp4"

            episode_start_frame = episode_keyframes[0].frame_idx
            episode_end_frame = episode_keyframes[-1].frame_idx

            first_frame_path = self.raw_dir / (f"{episode_start_frame:06d}.jpg")
            if first_frame_path.exists():

                first_image = cv2.imread(str(first_frame_path))
                height, width = first_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
                written = 0

                for frame_idx in range(episode_start_frame, episode_end_frame + 1):
                    frame_path = self.raw_dir / (f"{frame_idx:06d}.jpg")
                    if not frame_path.exists():
                        continue
                    frame = cv2.imread(str(frame_path))
                    if frame is None:
                        continue
                    cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    video_writer.write(frame)
                    written += 1

                video_writer.release()
                self.get_logger().info(f"Saved episode video: {video_path}")

        metadata = []
        for idx, kf in enumerate(keyframes):

            src = self.raw_dir / (f"{kf.frame_idx:06d}.jpg")
            dst = self.kf_dir / (f"kf_{idx:04d}_" f"{kf.reason}_" f"{kf.frame_idx:06d}.jpg")

            if src.exists():

                image = cv2.imread(str(src))
                cv2.putText(image, kf.reason, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imwrite(str(dst), image)

            metadata.append(
                {
                    "keyframe_idx": idx,
                    "frame_idx": kf.frame_idx,
                    "reason": kf.reason,
                    "x": kf.pose['x'],
                    "y": kf.pose['y'],
                    "yaw": kf.pose['yaw'],
                    "timestamp": kf.pose['timestamp'],
                }
            )

        json_path = self.output_dir / "keyframes.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.visualize(keyframes)
        self.get_logger().info(f"Saved {len(keyframes)} keyframes")
        self.get_logger().info(f"Output: {self.output_dir}")

    def visualize(self, keyframes):

        xs = [p['x'] for p in self.poses]
        ys = [p['y'] for p in self.poses]

        plt.figure(figsize=(10, 8))
        plt.plot(xs, ys, linewidth=1.0, alpha=0.5)

        color_map = {
            'start': 'green',
            'end': 'red',
            'sharp_turn': 'orange',
            'curvature': 'yellow',
            'distance': 'cyan',
        }

        for kf in keyframes:

            plt.scatter(kf.pose['x'], kf.pose['y'], c=color_map.get(kf.reason, 'white'), s=80)

        plt.axis('equal')
        plt.grid(True)
        save_path = self.output_dir / "trajectory.png"

        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f"Saved trajectory plot")


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
