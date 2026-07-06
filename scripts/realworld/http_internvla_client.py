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
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from PIL import Image as PIL_Image, ImageDraw
from sensor_msgs.msg import Image

frame_data = {}
frame_idx = 0
# user-specific
from controllers import Mpc_controller, PID_controller
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from thread_utils import ReadWriteLock


class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


# global variable
policy_init = True
mpc = None
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
http_idx = -1
first_running_time = 0.0
last_pixel_goal = None
last_s2_step = -1
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None

desired_v, desired_w = 0.0, 0.0
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()

# offset_x = 0.3
offset_x = 0.0
offset_y = 0.0

def dual_sys_eval(image_bytes, depth_bytes, front_image_bytes, url='http://127.0.0.1:5801/eval_dual'):
    global policy_init, http_idx, first_running_time, last_pixel_goal
    data = {"reset": policy_init, "idx": http_idx}
    json_data = json.dumps(data)

    policy_init = False
    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
        'depth': ('depth_image', depth_bytes, 'image/png'),
    }
    start = time.time()
    response = requests.post(url, files=files, data={'json': json_data}, timeout=100)
    print(f"response {response.text}")
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"idx: {http_idx} after http {time.time() - start}")

    response_json = json.loads(response.text)
    if 'pixel_goal' in response_json:
        last_pixel_goal = response_json['pixel_goal']

    return response_json


def annotate_inference_image(image, pixel_goal=None, trajectory=None):
    if image is None:
        return None

    annotated = np.ascontiguousarray(np.asarray(image).copy())
    if annotated.ndim != 3 or annotated.shape[2] < 3:
        return None
    annotated = np.ascontiguousarray(annotated[:, :, :3])

    pil_image = PIL_Image.fromarray(annotated)
    draw = ImageDraw.Draw(pil_image)
    if trajectory is not None:
        draw_topdown_trajectory_label(draw, pil_image.size, trajectory)
    if pixel_goal is not None:
        draw_pixel_goal(draw, pil_image.size, pixel_goal)
    return np.ascontiguousarray(np.asarray(pil_image))


def draw_pixel_goal(draw, image_size, pixel_goal, radius=6):
    pixel_goal = np.asarray(pixel_goal, dtype=np.int32).reshape(-1)
    if pixel_goal.size < 2:
        return

    width, height = image_size
    y, x = int(pixel_goal[0]), int(pixel_goal[1])
    if x < 0 or x >= width or y < 0 or y >= height:
        return

    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0), outline=(255, 255, 255))
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 0, 0), width=2)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 0, 0), width=2)


def draw_topdown_trajectory_label(draw, image_size, trajectory, margin=12):
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        return

    points = trajectory[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return

    width, height = image_size
    inset_size = min(180, max(120, min(width, height) // 3), width - margin * 2, height - margin * 2)
    if inset_size < 80:
        return

    left, top = margin, margin
    right, bottom = left + inset_size, top + inset_size
    draw.rectangle((left, top, right, bottom), fill=(18, 24, 30), outline=(240, 240, 240), width=2)
    draw.text((left + 8, top + 5), "top-down traj", fill=(240, 240, 240))

    plot_left = left + 14
    plot_top = top + 26
    plot_right = right - 12
    plot_bottom = bottom - 12
    plot_width = max(1, plot_right - plot_left)
    plot_height = max(1, plot_bottom - plot_top)

    points = np.vstack((np.zeros((1, 2), dtype=np.float32), points))
    forward = points[:, 0]
    lateral = points[:, 1]
    min_forward, max_forward = min(0.0, float(np.min(forward))), max(0.0, float(np.max(forward)))
    min_lateral, max_lateral = min(0.0, float(np.min(lateral))), max(0.0, float(np.max(lateral)))

    forward_span = max(max_forward - min_forward, 0.5)
    lateral_span = max(max_lateral - min_lateral, 0.5)
    min_forward -= forward_span * 0.1
    max_forward += forward_span * 0.1
    min_lateral -= lateral_span * 0.1
    max_lateral += lateral_span * 0.1

    def to_plot(point):
        forward_m, lateral_m = float(point[0]), float(point[1])
        x = plot_left + (max_lateral - lateral_m) / (max_lateral - min_lateral) * plot_width
        y = plot_bottom - (forward_m - min_forward) / (max_forward - min_forward) * plot_height
        return int(round(x)), int(round(y))

    origin = to_plot((0.0, 0.0))
    draw.line((plot_left, origin[1], plot_right, origin[1]), fill=(70, 82, 92), width=1)
    draw.line((origin[0], plot_top, origin[0], plot_bottom), fill=(70, 82, 92), width=1)

    plot_points = [to_plot(point) for point in points]
    if len(plot_points) > 1:
        draw.line(plot_points, fill=(0, 220, 255), width=3)
    draw.ellipse((origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4), fill=(70, 255, 120))
    goal = plot_points[-1]
    draw.ellipse((goal[0] - 4, goal[1] - 4, goal[0] + 4, goal[1] + 4), fill=(255, 190, 40))
    draw.text((plot_left, plot_top), "front", fill=(180, 190, 200))


def response_trajectory_to_path_msg(trajectory, stamp, frame_id='egocentric_frame'):
    path_msg = PathMsg()
    path_msg.header.stamp = stamp
    path_msg.header.frame_id = frame_id

    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        return path_msg

    for point in trajectory:
        pose_msg = PoseStamped()
        pose_msg.header = path_msg.header
        pose_msg.pose.position.x = float(point[0])
        pose_msg.pose.position.y = float(point[1])
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        path_msg.poses.append(pose_msg)

    return path_msg


def control_thread():
    global desired_v, desired_w
    while True:
        global current_control_mode
        if current_control_mode == ControlMode.MPC_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            if mpc is not None and manager is not None and odom is not None:
                local_mpc = mpc
                opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]

                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)
        elif current_control_mode == ControlMode.PID_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0:
                    v = 0.0
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)

        time.sleep(0.1)


def planning_thread():
    global trajs_in_world

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.3
        time.sleep(0.05)

        if not manager.new_image_arrived:
            time.sleep(0.01)
            continue
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        infer_rgb = copy.deepcopy(manager.rgb_image)
        infer_depth = copy.deepcopy(manager.depth_image)
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        # time_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
                # time_diff = odom[0] - rgb_time
        # odom_time = manager.odom_timestamp
        odom_rw_lock.release_read()

        if odom_infer is not None and rgb_bytes is not None and depth_bytes is not None:
            global frame_data
            frame_data[http_idx] = {
                'infer_rgb': copy.deepcopy(infer_rgb),
                'infer_depth': copy.deepcopy(infer_depth),
                'infer_odom': copy.deepcopy(odom_infer),
            }
            if len(frame_data) > 100:
                del frame_data[min(frame_data.keys())]
            response = dual_sys_eval(rgb_bytes, depth_bytes, None)
            if manager.debug_publish_visualization and 'pixel_goal' in response:
                manager.publish_annotated_image(
                    infer_rgb,
                    pixel_goal=last_pixel_goal,
                    trajectory=response.get('trajectory'),
                )

            global current_control_mode
            traj_len = 0.0
            if 'trajectory' in response:
                trajectory = response['trajectory']
                if manager.debug_publish_visualization:
                    manager.publish_response_trajectory_path(trajectory)
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
                print(f"traj len {traj_len}")
                for i, traj in enumerate(trajectory):
                    if i < 3:
                        continue
                    x_, y_, yaw_ = odom[0], odom[1], odom[2]

                    w_T_b = np.array(
                        [
                            [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                            [np.sin(yaw_), np.cos(yaw_), 0, y_],
                            [0.0, 0.0, 1.0, 0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    )
                    w_P = (w_T_b @ (np.array([traj[0] + offset_x, traj[1] + offset_y, 0.0, 1.0])).T)[:2]
                    trajs_in_world.append(w_P)
                trajs_in_world = np.array(trajs_in_world)
                print(f"{time.time()} update traj")

                manager.last_trajs_in_world = trajs_in_world
                mpc_rw_lock.acquire_write()
                global mpc
                if mpc is None:
                    mpc = Mpc_controller(np.array(trajs_in_world))
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
                manager.request_cnt += 1
                mpc_rw_lock.release_write()
                current_control_mode = ControlMode.MPC_Mode
            elif 'discrete_action' in response:
                actions = response['discrete_action']
                print("actions log: ", actions)
                if actions != [5] and actions != [9]:
                    manager.incremental_change_goal(actions)
                    current_control_mode = ControlMode.PID_Mode
        else:
            print(
                f"skip planning. odom_infer: {odom_infer is not None} rgb_bytes: {rgb_bytes is not None} depth_bytes: {depth_bytes is not None}"
            )
            time.sleep(0.1)

        time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))


class Go2Manager(Node):
    def __init__(self):
        super().__init__('go2_manager')
        self.debug_publish_visualization = bool(self.declare_parameter('debug_publish_visualization', True).value)

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5, durability=DurabilityPolicy.VOLATILE)

        rgb_down_sub = Subscriber(self, Image, "/camera/waist_front_zed_stream/left/color/rect/image", qos_profile=qos_profile)
        depth_down_sub = Subscriber(self, Image, "/camera/waist_front_zed_stream/depth/depth_registered", qos_profile=qos_profile)

        
        self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        self.odom_sub = self.create_subscription(Odometry, "/graph_msf/opt_odometry_world_base_filtered", self.odom_callback, qos_profile)

        # publisher
        self.control_pub = self.create_publisher(Twist, '/cmd_vel/nav', 5)
        self.response_trajectory_path_pub = self.create_publisher(PathMsg, '/vln_path', 5)
        self.pixel_goal_image_pub = self.create_publisher(Image, '/internvla_n1/pixel_goal_image', 5)

        # class member variable
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.rgb_forward_image = None
        self.rgb_forward_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None

    def rgb_forward_callback(self, rgb_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_forward_image = raw_image
        image = PIL_Image.fromarray(self.rgb_forward_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        self.rgb_forward_bytes = image_bytes
        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_image = raw_image
        image = PIL_Image.fromarray(self.rgb_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
 
        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth_encoding = depth_msg.encoding.upper()
        if depth_encoding == '16UC1':
            # Integer depth images contain millimeters.
            self.depth_image = raw_depth.astype(np.float32) / 1000.0
        elif depth_encoding == '32FC1':
            # Floating-point depth images already contain meters.
            self.depth_image = raw_depth.astype(np.float32)
        else:
            self.get_logger().error(f"Unsupported depth encoding: {depth_msg.encoding}")
            return
 
        self.depth_image[~np.isfinite(self.depth_image)] = 0.0
        self.depth_image[self.depth_image < 0.0] = 0.0
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth = PIL_Image.fromarray(depth)
        depth_bytes = io.BytesIO()
        depth.save(depth_bytes, format='PNG')
        depth_bytes.seek(0)
 
        rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = image_bytes
 
        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        self.last_rgb_time = self.rgb_time
 
        self.depth_bytes = depth_bytes
        self.depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1.0e9
        self.last_depth_time = self.depth_time
 
        rgb_depth_rw_lock.release_write()
 
        self.new_vis_image_arrived = True
        self.new_image_arrived = True
 
 
    def odom_callback(self, msg):
        self.odom_cnt += 1
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        odom_rw_lock.release_write()

        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw

        self.control_pub.publish(request)

    def publish_response_trajectory_path(self, trajectory):
        path_msg = response_trajectory_to_path_msg(trajectory, self.get_clock().now().to_msg(), frame_id='egocentric_frame')
        if path_msg.poses:
            self.response_trajectory_path_pub.publish(path_msg)

    def publish_annotated_image(self, image, pixel_goal=None, trajectory=None):
        annotated = annotate_inference_image(image, pixel_goal=pixel_goal, trajectory=trajectory)
        if annotated is None:
            return

        image_msg = self.cv_bridge.cv2_to_imgmsg(annotated, encoding='rgb8')
        image_msg.header.stamp = self.get_clock().now().to_msg()
        self.pixel_goal_image_pub.publish(image_msg)


if __name__ == '__main__':
    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread)
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True
    rclpy.init()

    try:
        manager = Go2Manager()

        control_thread_instance.start()
        planning_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()
