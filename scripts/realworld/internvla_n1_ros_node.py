#!/usr/bin/env python3

import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray, MultiArrayDimension, String
from std_srvs.srv import Trigger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "third_party" / "diffusion-policy"):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent  # noqa: E402

@dataclass
class AgentArgs:
    device: str
    model_path: str
    resize_w: int
    resize_h: int
    num_history: int
    camera_intrinsic: np.ndarray
    plan_step_gap: int


def _resolve_repo_relative_path(raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)

    repo_path = PROJECT_ROOT / path
    if repo_path.exists():
        return str(repo_path)

    # Leave Hugging Face repo ids or not-yet-created paths unchanged.
    return raw_path


def _load_instruction(instruction: str, instruction_file: str) -> str:
    instruction = instruction.strip()
    instruction_file = instruction_file.strip()
    if not instruction_file:
        return instruction

    path = Path(instruction_file).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.read_text(encoding="utf-8").strip()


def _make_intrinsic(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 9:
        return values.reshape(3, 3)
    if values.size == 16:
        return values.reshape(4, 4)
    raise ValueError("camera_intrinsic must be a flat list with 9 or 16 values")


def _yaw_to_quaternion(yaw: float):
    half_yaw = 0.5 * yaw
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def _action_list(output_action) -> list:
    if output_action is None:
        return []
    return np.asarray(output_action, dtype=np.int32).reshape(-1).tolist()


class InternVLAN1RosNode(Node):
    def __init__(self):
        super().__init__("internvla_n1_ros_node")

        self._declare_parameters()
        self.bridge = CvBridge()
        self._busy = False
        self._last_infer_time = 0.0

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.instruction_topic = self.get_parameter("instruction_topic").value
        self.trajectory_frame_id = self.get_parameter("trajectory_frame_id").value
        self.depth_unit_scale = float(self.get_parameter("depth_unit_scale").value)
        self.float_depth_unit_scale = float(self.get_parameter("float_depth_unit_scale").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.invalid_depth_value_m = float(self.get_parameter("invalid_depth_value_m").value)
        self.max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self.rerun_on_look_down_action = bool(self.get_parameter("rerun_on_look_down_action").value)
        self.reset_on_instruction = bool(self.get_parameter("reset_on_instruction").value)

        self.camera_intrinsic = _make_intrinsic(self.get_parameter("camera_intrinsic").value)
        self.instruction = _load_instruction(
            self.get_parameter("instruction").value,
            self.get_parameter("instruction_file").value,
        )

        agent_args = AgentArgs(
            device=self.get_parameter("device").value,
            model_path=_resolve_repo_relative_path(self.get_parameter("model_path").value),
            resize_w=int(self.get_parameter("resize_w").value),
            resize_h=int(self.get_parameter("resize_h").value),
            num_history=int(self.get_parameter("num_history").value),
            camera_intrinsic=self.camera_intrinsic,
            plan_step_gap=int(self.get_parameter("plan_step_gap").value),
        )

        self.get_logger().info(f"Loading InternVLA-N1 model from {agent_args.model_path} on {agent_args.device}")
        self.agent = InternVLAN1AsyncAgent(agent_args)

        if bool(self.get_parameter("warmup").value):
            self._warmup()

        self.path_pub = self.create_publisher(PathMsg, self.get_parameter("trajectory_path_topic").value, 1)
        self.array_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("trajectory_array_topic").value, 1
        )
        self.action_pub = self.create_publisher(Int32MultiArray, self.get_parameter("discrete_action_topic").value, 1)
        self.pixel_pub = self.create_publisher(Int32MultiArray, self.get_parameter("pixel_goal_topic").value, 1)
        self.latency_pub = self.create_publisher(Float32, self.get_parameter("latency_topic").value, 1)
        self.status_pub = self.create_publisher(String, self.get_parameter("status_topic").value, 1)

        self.create_subscription(String, self.instruction_topic, self._instruction_callback, 10)
        self.create_service(Trigger, self.get_parameter("reset_service").value, self._reset_callback)

        qos_profile = self._image_qos()
        rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos_profile)
        depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile)
        self.synchronizer = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=int(self.get_parameter("sync_queue_size").value),
            slop=float(self.get_parameter("sync_slop_sec").value),
        )
        self.synchronizer.registerCallback(self._rgb_depth_callback)

        self.get_logger().info(
            "Ready. Subscribed to "
            f"rgb={self.rgb_topic}, depth={self.depth_topic}; publishing "
            f"path={self.get_parameter('trajectory_path_topic').value}"
        )

    def _declare_parameters(self):
        default_intrinsic = [
            386.5,
            0.0,
            328.9,
            0.0,
            0.0,
            386.5,
            244.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("model_path", "checkpoints/InternVLA-N1-DualVLN")
        self.declare_parameter("resize_w", 384)
        self.declare_parameter("resize_h", 384)
        self.declare_parameter("num_history", 8)
        self.declare_parameter("plan_step_gap", 4)
        self.declare_parameter("camera_intrinsic", default_intrinsic)
        self.declare_parameter("warmup", True)
        self.declare_parameter("warmup_instruction", "hello")

        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("instruction_topic", "/internvla_n1/instruction")
        self.declare_parameter("instruction", "")
        self.declare_parameter("instruction_file", "")
        self.declare_parameter("reset_on_instruction", True)

        self.declare_parameter("trajectory_path_topic", "/internvla_n1/trajectory_path")
        self.declare_parameter("trajectory_array_topic", "/internvla_n1/trajectory")
        self.declare_parameter("discrete_action_topic", "/internvla_n1/discrete_actions")
        self.declare_parameter("pixel_goal_topic", "/internvla_n1/pixel_goal")
        self.declare_parameter("latency_topic", "/internvla_n1/inference_latency_sec")
        self.declare_parameter("status_topic", "/internvla_n1/status")
        self.declare_parameter("reset_service", "/internvla_n1/reset")
        self.declare_parameter("trajectory_frame_id", "base_link")

        self.declare_parameter("sync_queue_size", 5)
        self.declare_parameter("sync_slop_sec", 0.1)
        self.declare_parameter("image_best_effort_qos", True)
        self.declare_parameter("max_rate_hz", 0.0)
        self.declare_parameter("rerun_on_look_down_action", True)

        self.declare_parameter("depth_unit_scale", 0.001)
        self.declare_parameter("float_depth_unit_scale", 1.0)
        self.declare_parameter("depth_min_m", 0.0)
        self.declare_parameter("depth_max_m", 10.0)
        self.declare_parameter("invalid_depth_value_m", 0.0)

    def _image_qos(self) -> QoSProfile:
        reliability = (
            ReliabilityPolicy.BEST_EFFORT
            if bool(self.get_parameter("image_best_effort_qos").value)
            else ReliabilityPolicy.RELIABLE
        )
        return QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=10)

    def _warmup(self):
        self.get_logger().info("Warming up InternVLA-N1")
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_depth = np.zeros((480, 640), dtype=np.float32)
        dummy_pose = np.eye(4, dtype=np.float32)
        warmup_instruction = self.get_parameter("warmup_instruction").value
        with torch.inference_mode():
            self.agent.reset()
            self.agent.step(
                dummy_rgb,
                dummy_depth,
                dummy_pose,
                warmup_instruction,
                intrinsic=self.camera_intrinsic,
                look_down=False,
            )
            self.agent.reset()
        self.get_logger().info("Warmup finished")

    def _instruction_callback(self, msg: String):
        instruction = msg.data.strip()
        if not instruction:
            self.get_logger().warn("Ignoring empty instruction message")
            return

        changed = instruction != self.instruction
        self.instruction = instruction
        self.get_logger().info("Updated navigation instruction")
        if changed and self.reset_on_instruction:
            self.agent.reset()
            self.get_logger().info("Agent history reset after instruction update")

    def _reset_callback(self, _request, response):
        self.agent.reset()
        self._last_infer_time = 0.0
        response.success = True
        response.message = "InternVLA-N1 agent reset"
        return response

    def _rgb_depth_callback(self, rgb_msg: Image, depth_msg: Image):
        if self._busy:
            self.get_logger().warn("Skipping frame because inference is still running", throttle_duration_sec=5.0)
            return

        if not self.instruction:
            self.get_logger().warn(
                "No instruction set. Provide parameter 'instruction', 'instruction_file', or publish std_msgs/String "
                f"to {self.instruction_topic}.",
                throttle_duration_sec=5.0,
            )
            return

        now = time.monotonic()
        if self.max_rate_hz > 0 and now - self._last_infer_time < 1.0 / self.max_rate_hz:
            return

        self._busy = True
        start_time = time.monotonic()
        try:
            rgb = self._rgb_msg_to_array(rgb_msg)
            depth = self._depth_msg_to_meters(depth_msg)
            pose = np.eye(4, dtype=np.float32)

            with torch.inference_mode():
                output = self.agent.step(
                    rgb,
                    depth,
                    pose,
                    self.instruction,
                    intrinsic=self.camera_intrinsic,
                    look_down=False,
                )
                if self.rerun_on_look_down_action and _action_list(output.output_action) == [5]:
                    output = self.agent.step(
                        rgb,
                        depth,
                        pose,
                        self.instruction,
                        intrinsic=self.camera_intrinsic,
                        look_down=True,
                    )

            latency_sec = time.monotonic() - start_time
            self._publish_output(output, rgb_msg.header, latency_sec)
            self._last_infer_time = time.monotonic()
        except Exception:
            self.get_logger().error("InternVLA-N1 inference failed:\n" + traceback.format_exc())
        finally:
            self._busy = False

    def _rgb_msg_to_array(self, rgb_msg: Image) -> np.ndarray:
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def _depth_msg_to_meters(self, depth_msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        if np.issubdtype(depth.dtype, np.integer):
            depth_m = depth.astype(np.float32) * self.depth_unit_scale
        else:
            depth_m = depth.astype(np.float32) * self.float_depth_unit_scale

        depth_m[~np.isfinite(depth_m)] = self.invalid_depth_value_m
        if self.depth_min_m > 0.0:
            depth_m[depth_m < self.depth_min_m] = self.invalid_depth_value_m
        if self.depth_max_m > 0.0:
            depth_m = np.clip(depth_m, 0.0, self.depth_max_m)
        return np.ascontiguousarray(depth_m, dtype=np.float32)

    def _publish_output(self, output, header, latency_sec: float):
        latency_msg = Float32()
        latency_msg.data = float(latency_sec)
        self.latency_pub.publish(latency_msg)

        if output.output_action is not None:
            action_msg = Int32MultiArray()
            action_msg.data = _action_list(output.output_action)
            self.action_pub.publish(action_msg)
            self._publish_status(f"discrete_action={action_msg.data} latency_sec={latency_sec:.3f}")
            return

        if output.output_trajectory is None:
            self._publish_status(f"no_output latency_sec={latency_sec:.3f}")
            return

        trajectory = np.asarray(output.output_trajectory, dtype=np.float32)
        if trajectory.ndim != 2 or trajectory.shape[1] < 2:
            self.get_logger().error(f"Invalid trajectory shape: {trajectory.shape}")
            return

        self.path_pub.publish(self._trajectory_to_path(trajectory, header))
        self.array_pub.publish(self._trajectory_to_array_msg(trajectory))

        if output.output_pixel is not None:
            pixel_msg = Int32MultiArray()
            pixel_msg.data = [int(v) for v in output.output_pixel]
            self.pixel_pub.publish(pixel_msg)

        self._publish_status(f"trajectory_points={trajectory.shape[0]} latency_sec={latency_sec:.3f}")

    def _trajectory_to_path(self, trajectory: np.ndarray, header) -> PathMsg:
        path_msg = PathMsg()
        path_msg.header.stamp = header.stamp
        path_msg.header.frame_id = self.trajectory_frame_id or header.frame_id

        last_yaw = 0.0
        for idx, point in enumerate(trajectory):
            pose_msg = PoseStamped()
            pose_msg.header = path_msg.header
            pose_msg.pose.position.x = float(point[0])
            pose_msg.pose.position.y = float(point[1])
            pose_msg.pose.position.z = 0.0

            if trajectory.shape[1] >= 3:
                yaw = float(point[2])
            else:
                yaw = self._yaw_from_neighbor_points(trajectory, idx, last_yaw)
            last_yaw = yaw

            qx, qy, qz, qw = _yaw_to_quaternion(yaw)
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            path_msg.poses.append(pose_msg)

        return path_msg

    def _yaw_from_neighbor_points(self, trajectory: np.ndarray, idx: int, fallback_yaw: float) -> float:
        if trajectory.shape[0] < 2:
            return fallback_yaw

        if idx < trajectory.shape[0] - 1:
            delta = trajectory[idx + 1, :2] - trajectory[idx, :2]
        else:
            delta = trajectory[idx, :2] - trajectory[idx - 1, :2]

        if float(np.linalg.norm(delta)) < 1e-6:
            return fallback_yaw
        return float(math.atan2(delta[1], delta[0]))

    def _trajectory_to_array_msg(self, trajectory: np.ndarray) -> Float32MultiArray:
        rows, cols = trajectory.shape
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="points", size=rows, stride=rows * cols),
            MultiArrayDimension(label="fields", size=cols, stride=cols),
        ]
        msg.data = trajectory.reshape(-1).astype(np.float32).tolist()
        return msg

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)


def main(args: Optional[list] = None):
    rclpy.init(args=args)
    node = InternVLAN1RosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
