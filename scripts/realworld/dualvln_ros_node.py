import copy
import io
import json
import math
import threading
import time
from collections import deque
from enum import Enum

import numpy as np
import rclpy
import requests
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image, CompressedImage

frame_data = {}
frame_idx = 0
# user-specific
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock

class DualVLNNode(Node):
    def __init__(self):
        # Name the node
        super().__init__('minimal_node')
        
        # --- Parameters ---
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

        # --- Publishers/Subscribers ---
        rgb_sub = Subscriber(self, Image, "/camera/camera/color/image_raw")
        depth_sub = Subscriber(self, Image, "/camera/camera/depth")

        self.synchronizer = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 1, 0.1)

        self.synchronizer.registerCallback(self.rgb_depth_callback)
        
        # --- Timers ---
        self.timer = self.create_timer(1.0, self.timer_callback) # 1Hz
        
        self.get_logger().info('Dual VLN Node has started.')

    def rgb_depth_callback(self, rgb_msg, depth_msg):
        compressed_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        print(compressed_image.shape)
        
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        print(depth_image.shape)


        

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
        self.declare_parameter("depth_topic", "/camera/camera/depth")
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


    def timer_callback(self):
        """Example periodic task."""
        # msg = String()
        # msg.data = 'Hello'
        # self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: {msg.data}')
        pass

    # def listener_callback(self, msg):
    #     """Example subscription handler."""
    #     self.get_logger().info(f'Received: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = DualVLNNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Graceful shutdown
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()