import math
import threading
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path as PathMsg
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CompressedImage
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Pose2D
import tf2_ros
import bisect

@dataclass
class OdomEntry:
    timestamp: float         
    matrix: np.ndarray
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    yaw: float = 0.0


def parse_odom_txt(filepath: str) -> list[OdomEntry]:
    """
    Parse odom txt file with format:
    """
    entries = []
    lines = Path(filepath).read_text().strip().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            timestamp = float(line)
        except ValueError:
            i += 1
            continue

        matrix_lines = []
        for j in range(1, 5):
            if i + j < len(lines):
                matrix_lines.append(lines[i + j].strip())

        if len(matrix_lines) < 4:
            break

        try:
            matrix = np.array([
                [float(v) for v in row.split()]
                for row in matrix_lines
            ])
        except ValueError:
            i += 1
            continue

        if matrix.shape != (4, 4):
            i += 5
            continue

        # Extract translation + rotation
        tx, ty, tz = matrix[0, 3], matrix[1, 3], matrix[2, 3]
        rot_matrix = matrix[:3, :3]
        quat = Rotation.from_matrix(rot_matrix).as_quat()
        yaw = Rotation.from_matrix(rot_matrix).as_euler('xyz')[2]

        entry = OdomEntry(
            timestamp=timestamp,
            matrix=matrix,
            x=tx, y=ty, z=tz,
            qx=quat[0], qy=quat[1], qz=quat[2], qw=quat[3],
            yaw=yaw,
        )
        entries.append(entry)
        i += 6 

    print(f"[OdomParser] Loaded {len(entries)} entries "
          f"from {filepath}")
    print(f"  Time range: {entries[0].timestamp:.3f} → "
          f"{entries[-1].timestamp:.3f} "
          f"({entries[-1].timestamp - entries[0].timestamp:.1f}s)")
    return entries

class OdomMatcher:
    def __init__(self, entries: list[OdomEntry], max_dt: float = 0.5):
        """
        max_dt: max allowed time difference (seconds). If no match within this threshold, return None.
        """
        self.entries = entries
        self.timestamps = [e.timestamp for e in entries]
        self.max_dt = max_dt

    def find_closest(self, query_ts: float) -> OdomEntry | None:
        """Binary search for closest timestamp"""
        ts = self.timestamps
        idx = bisect.bisect_left(ts, query_ts)

        candidates = []
        if idx < len(ts):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        best_idx = min(candidates, key=lambda i: abs(ts[i] - query_ts))
        dt = abs(ts[best_idx] - query_ts)

        if dt > self.max_dt:
            return None

        return self.entries[best_idx]


class OdomTxtPublisher(Node):
    def __init__(self):
        super().__init__('odom_txt_publisher')

        # params
        self.declare_parameter('odom_file', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('max_time_diff', 0.05)
        self.declare_parameter('rgb_topic',
            '/camera/camera/color/image_raw/compressed')

        odom_file   = self.get_parameter('odom_file').value
        self.frame_id       = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        publish_rate        = self.get_parameter('publish_rate_hz').value
        max_dt              = self.get_parameter('max_time_diff').value
        rgb_topic           = self.get_parameter('rgb_topic').value

        if not odom_file:
            self.get_logger().error('odom_file parameter is required!')
            raise RuntimeError('odom_file not set')

        #load and parse
        self.entries = parse_odom_txt(odom_file)
        self.matcher = OdomMatcher(self.entries, max_dt=max_dt)
        self._entry_idx = 0          # for sequential publish
        self._lock = threading.Lock()

        # ── Publishers ───────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_pub   = self.create_publisher(Odometry, '/odom_txt/odometry', 10)
        self.path_pub   = self.create_publisher(PathMsg, '/odom_txt/path', 10)
        self.matched_pub = self.create_publisher(Odometry, '/odom_txt/matched', 10)
        self.xy_yaw_pub = self.create_publisher(Pose2D, '/odom_txt/xy_yaw', 10)
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Pre-build path message (static)
        self.path_msg = self._build_path_msg()

        # Subscriber: RGB for time-matched odom
        self.rgb_sub = self.create_subscription(CompressedImage, rgb_topic, self._rgb_callback, qos)
        self.get_logger().info(f'Subscribed to RGB: {rgb_topic}')

        # Timer for sequential playback
        period = 1.0 / publish_rate
        self.timer = self.create_timer(period, self._timer_callback)

        # Stats
        self._matched_count = 0
        self._missed_count  = 0

        self.get_logger().info(
            f'OdomTxtPublisher ready. '
            f'{len(self.entries)} entries loaded.')

    # build path message from all entries (static)
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

    # convert odom entry to Odometry message
    def _entry_to_msg(self, entry: OdomEntry,
                      stamp=None) -> Odometry:
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

    # Broadcast TF from entry
    def _broadcast_tf(self, entry: OdomEntry, stamp):
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = self.frame_id
        t.child_frame_id  = self.child_frame_id
        t.transform.translation.x = entry.x
        t.transform.translation.y = entry.y
        t.transform.translation.z = entry.z
        t.transform.rotation.x = entry.qx
        t.transform.rotation.y = entry.qy
        t.transform.rotation.z = entry.qz
        t.transform.rotation.w = entry.qw
        self.tf_broadcaster.sendTransform(t)

    # Timer callback: sequential playback   
    def _timer_callback(self):
        with self._lock:
            if self._entry_idx >= len(self.entries):
                self._entry_idx = 0  # loop

            entry = self.entries[self._entry_idx]
            self._entry_idx += 1

        stamp = self.get_clock().now().to_msg()

        # Publish Odometry
        odom_msg = self._entry_to_msg(entry, stamp)
        self.odom_pub.publish(odom_msg)

        # Publish x,y,yaw
        xyyaw_msg = self._entry_to_xyyaw(entry)
        self.xy_yaw_pub.publish(xyyaw_msg)

        # Publish TF
        self._broadcast_tf(entry, stamp)

        # Publish path periodically
        if self._entry_idx % 10 == 0:
            self.path_msg.header.stamp = stamp
            self.path_pub.publish(self.path_msg)

    # RGB callback
    def _rgb_callback(self, rgb_msg: CompressedImage):
        # extract timestamp
        rgb_ts = (
            rgb_msg.header.stamp.sec
            + rgb_msg.header.stamp.nanosec * 1e-9
        )

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

        # Publish matched odom
        odom_msg = self._entry_to_msg(matched, stamp)
        self.matched_pub.publish(odom_msg)

        # Publish matched x,y,yaw
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
        msg.theta = float(entry.yaw)   # radians
        return msg

if __name__ == '__main__':
    rclpy.init()
    try:
        node = OdomTxtPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()