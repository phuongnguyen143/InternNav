#!/usr/bin/env python3
"""ROS2 publishers for camera odom txt and precomputed floor trajectory txt."""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import rclpy
import tf2_ros
from geometry_msgs.msg import Pose2D, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from constants import (
    DEFAULT_CHILD_FRAME_ID,
    DEFAULT_FRAME_ID,
    DEFAULT_MAX_TIME_DIFF,
    DEFAULT_RGB_TOPIC,
)
from floor_pose import floor_xyyaw_to_quaternion
from trajectory_io import (
    FloorEntry,
    FloorMatcher,
    OdomEntry,
    OdomMatcher,
    parse_floor_trajectory_txt,
    parse_odom_txt,
)


def rgb_msg_timestamp(rgb_msg: CompressedImage) -> float:
    return rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9


def make_qos_best_effort(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class OdomTxtPublisher(Node):
    def __init__(self):
        super().__init__('odom_txt_publisher')

        self.declare_parameter('odom_file', '')
        self.declare_parameter('frame_id', DEFAULT_FRAME_ID)
        self.declare_parameter('child_frame_id', DEFAULT_CHILD_FRAME_ID)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('max_time_diff', DEFAULT_MAX_TIME_DIFF)
        self.declare_parameter('enable_timer_playback', False)
        self.declare_parameter('timer_loop', False)
        self.declare_parameter('rgb_topic', DEFAULT_RGB_TOPIC)
        odom_file   = self.get_parameter('odom_file').value
        self.frame_id       = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        publish_rate        = self.get_parameter('publish_rate_hz').value
        max_dt              = self.get_parameter('max_time_diff').value
        self.enable_timer    = self.get_parameter('enable_timer_playback').value
        self.timer_loop      = self.get_parameter('timer_loop').value
        rgb_topic           = self.get_parameter('rgb_topic').value

        if not odom_file:
            self.get_logger().error('odom_file parameter is required!')
            raise RuntimeError('odom_file not set')

        self.entries = parse_odom_txt(odom_file)
        self.matcher = OdomMatcher(self.entries, max_dt=max_dt)
        self._entry_idx = 0
        self._lock = threading.Lock()

        qos = make_qos_best_effort()

        self.odom_pub    = self.create_publisher(Odometry, '/odom_txt/odometry', 10)
        self.path_pub    = self.create_publisher(PathMsg, '/odom_txt/path', 10)
        self.matched_pub = self.create_publisher(Odometry, '/odom_txt/matched', 10)
        self.xy_yaw_pub  = self.create_publisher(Pose2D, '/odom_txt/xy_yaw', 10)
        self.xy_yaw_timer_pub = self.create_publisher(Pose2D, '/odom_txt/xy_yaw_timer', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.path_msg = self._build_path_msg()

        self.rgb_sub = self.create_subscription(CompressedImage, rgb_topic, self._rgb_callback, qos)
        self.get_logger().info(f'Subscribed to RGB: {rgb_topic}')

        self.timer = None
        if self.enable_timer:
            period = 1.0 / publish_rate
            self.timer = self.create_timer(period, self._timer_callback)
            self.get_logger().info(
                f'Timer playback enabled at {publish_rate:.1f} Hz '
                f'(loop={self.timer_loop})'
            )
        else:
            self.get_logger().info(
                'Timer playback disabled; /odom_txt/xy_yaw is RGB-matched only'
            )

        self._matched_count = 0
        self._missed_count  = 0

        self.get_logger().info(
            f'OdomTxtPublisher ready. '
            f'{len(self.entries)} entries loaded.')

    def _build_path_msg(self) -> PathMsg:
        msg = PathMsg()
        msg.header.frame_id = self.frame_id
        for entry in self.entries:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = entry.x
            pose.pose.position.y = entry.y
            pose.pose.position.z = entry.z
            pose.pose.orientation.x = entry.qx
            pose.pose.orientation.y = entry.qy
            pose.pose.orientation.z = entry.qz
            pose.pose.orientation.w = entry.qw
            msg.poses.append(pose)
        return msg

    def _entry_to_msg(self, entry: OdomEntry, stamp=None) -> Odometry:
        msg = Odometry()
        if stamp is None:
            stamp = self.get_clock().now().to_msg()
        msg.header.stamp    = stamp
        msg.header.frame_id = self.frame_id
        msg.child_frame_id  = self.child_frame_id

        msg.pose.pose.position.x = entry.x
        msg.pose.pose.position.y = entry.y
        msg.pose.pose.position.z = entry.z
        msg.pose.pose.orientation.x = entry.qx
        msg.pose.pose.orientation.y = entry.qy
        msg.pose.pose.orientation.z = entry.qz
        msg.pose.pose.orientation.w = entry.qw
        return msg

    def _broadcast_tf(self, entry: OdomEntry, stamp):
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = self.frame_id
        t.child_frame_id  = self.child_frame_id
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def _timer_callback(self):
        with self._lock:
            if self._entry_idx >= len(self.entries):
                if self.timer_loop:
                    self._entry_idx = 0
                else:
                    return

            entry = self.entries[self._entry_idx]
            self._entry_idx += 1

        stamp = self.get_clock().now().to_msg()

        odom_msg = self._entry_to_msg(entry, stamp)
        self.odom_pub.publish(odom_msg)

        self.xy_yaw_timer_pub.publish(self._entry_to_xyyaw(entry))

        self._broadcast_tf(entry, stamp)

        if self._entry_idx % 10 == 0:
            self.path_msg.header.stamp = stamp
            self.path_pub.publish(self.path_msg)

    def _rgb_callback(self, rgb_msg: CompressedImage):
        rgb_ts = rgb_msg_timestamp(rgb_msg)

        matched = self.matcher.find_closest(rgb_ts)

        if matched is None:
            self._missed_count += 1
            if self._missed_count % 10 == 1:
                self.get_logger().warn(
                    f'No odom match for RGB ts={rgb_ts:.3f} '
                    f'(missed={self._missed_count})'
                )
            return

        self._matched_count += 1

        stamp = rgb_msg.header.stamp

        odom_msg = self._entry_to_msg(matched, stamp)
        self.matched_pub.publish(odom_msg)

        xyyaw_msg = self._entry_to_xyyaw(matched)
        self.xy_yaw_pub.publish(xyyaw_msg)

        dt = abs(matched.timestamp - rgb_ts)

        if self._matched_count % 50 == 1:
            self.get_logger().info(
                f'[Match] rgb_ts={rgb_ts:.3f} | '
                f'odom_ts={matched.timestamp:.3f} | '
                f'dt={dt*1000:.1f}ms | '
                f'pos=({matched.x:.2f}, {matched.y:.2f}) | '
                f'yaw={math.degrees(matched.yaw):.1f}° | '
                f'matched={self._matched_count} '
                f'missed={self._missed_count}'
            )

    def _entry_to_xyyaw(self, entry: OdomEntry):
        msg = Pose2D()
        msg.x = float(entry.x)
        msg.y = float(entry.y)
        msg.theta = float(entry.yaw)
        return msg


class FloorTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("floor_trajectory_publisher")

        self.declare_parameter("floor_trajectory_file", "")
        self.declare_parameter("frame_id", DEFAULT_FRAME_ID)
        self.declare_parameter("child_frame_id", DEFAULT_CHILD_FRAME_ID)
        self.declare_parameter("max_time_diff", DEFAULT_MAX_TIME_DIFF)
        self.declare_parameter("rgb_topic", DEFAULT_RGB_TOPIC)

        floor_file = self.get_parameter("floor_trajectory_file").value
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        max_dt = self.get_parameter("max_time_diff").value
        rgb_topic = self.get_parameter("rgb_topic").value

        if not floor_file:
            raise RuntimeError("floor_trajectory_file parameter is required")
        if not Path(floor_file).is_file():
            raise FileNotFoundError(f"floor_trajectory_file not found: {floor_file}")

        self.entries = parse_floor_trajectory_txt(floor_file)
        self.matcher = FloorMatcher(self.entries, max_dt=max_dt)
        self._matched_count = 0
        self._missed_count = 0

        qos = make_qos_best_effort()

        self.matched_pub = self.create_publisher(Odometry, "/odom_txt/matched", 10)
        self.xy_yaw_pub = self.create_publisher(Pose2D, "/odom_txt/xy_yaw", 10)
        self.path_pub = self.create_publisher(PathMsg, "/odom_txt/path", 10)
        self.path_msg = self._build_path_msg()

        self.rgb_sub = self.create_subscription(
            CompressedImage, rgb_topic, self._rgb_callback, qos
        )

        self.get_logger().info(
            f"FloorTrajectoryPublisher ready: {len(self.entries)} entries, "
            f"no PCD estimation at startup."
        )

    @staticmethod
    def _entry_to_xyyaw(entry: FloorEntry) -> Pose2D:
        msg = Pose2D()
        msg.x = float(entry.x)
        msg.y = float(entry.y)
        msg.theta = float(entry.yaw)
        return msg

    def _entry_to_msg(self, entry: FloorEntry, stamp) -> Odometry:
        qx, qy, qz, qw = floor_xyyaw_to_quaternion(entry.yaw)
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.pose.pose.position.x = entry.x
        msg.pose.pose.position.y = entry.y
        msg.pose.pose.position.z = entry.z
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        return msg

    def _build_path_msg(self) -> PathMsg:
        msg = PathMsg()
        msg.header.frame_id = self.frame_id
        for entry in self.entries:
            qx, qy, qz, qw = floor_xyyaw_to_quaternion(entry.yaw)
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = entry.x
            pose.pose.position.y = entry.y
            pose.pose.position.z = entry.z
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            msg.poses.append(pose)
        return msg

    def _rgb_callback(self, rgb_msg: CompressedImage):
        rgb_ts = rgb_msg_timestamp(rgb_msg)
        matched = self.matcher.find_closest(rgb_ts)
        if matched is None:
            self._missed_count += 1
            if self._missed_count % 10 == 1:
                self.get_logger().warn(
                    f"No floor match for RGB ts={rgb_ts:.3f} (missed={self._missed_count})"
                )
            return

        self._matched_count += 1
        stamp = rgb_msg.header.stamp
        self.matched_pub.publish(self._entry_to_msg(matched, stamp))
        self.xy_yaw_pub.publish(self._entry_to_xyyaw(matched))

        if self._matched_count % 50 == 1:
            dt = abs(matched.timestamp - rgb_ts) * 1000
            self.get_logger().info(
                f"[Match] rgb_ts={rgb_ts:.3f} floor_ts={matched.timestamp:.3f} "
                f"dt={dt:.1f}ms pos=({matched.x:.2f},{matched.y:.2f}) "
                f"yaw={math.degrees(matched.yaw):.1f}° "
                f"matched={self._matched_count} missed={self._missed_count}"
            )


def main_floor():
    rclpy.init()
    try:
        node = FloorTrajectoryPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


def main_camera():
    rclpy.init()
    try:
        node = OdomTxtPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("floor", "camera"):
        print("Usage: python trajectory_publishers.py {floor|camera} [--ros-args ...]")
        sys.exit(1)
    mode = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if mode == "floor":
        main_floor()
    else:
        main_camera()
