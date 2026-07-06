import argparse
import json
import os
import time
from datetime import datetime
import torch

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''




@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    start_time = time.time()

    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}")

    camera_pose = np.eye(4, dtype=np.float32)
    instruction = args.instruction
    print(f"Instruction is {instruction}")
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    idx += 1

    look_down = False
    t0 = time.time()
    dual_sys_output = {}

    with torch.inference_mode():
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )
        print("dual_sys_output0", dual_sys_output, "\n")
        if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
            look_down = True
            dual_sys_output = agent.step(
                image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
            )
        print("dual_sys_output", dual_sys_output, "\n")

        json_output = {}
        if dual_sys_output.output_action is not None:
            json_output['discrete_action'] = dual_sys_output.output_action
        else:
            json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
            if dual_sys_output.output_pixel is not None:
                json_output['pixel_goal'] = dual_sys_output.output_pixel

    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


# 1. Go straight and stop at the green plant
# 2. Turn left. Stop when you are facing left
# 3. Move straight and stop at the second green plant
# /home/phuongnh/khang/InternNav/checkpoints/DualVLN-pixel-goal-v3

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model_path", type=str, default="checkpoints/dit/290626-only-s1/InternVLA-N1-DualVLN-office-rtx5090-v1")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    parser.add_argument(
        "--instruction",
        type=str,
        default=(
            "Go straight and turn left at the green plant on the floor. Stop after 2 meters"
        ),
    )
    args = parser.parse_args()

    args.camera_intrinsic = np.array(
        [[365.76409912109375, 0.0, 476.19769287109375, 0.0], [0.0, 365.76409912109375, 309.35333251953125, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    agent = InternVLAN1AsyncAgent(args)

    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.float32)
    dummy_pose = np.eye(4, dtype=np.float32)
    with torch.inference_mode():
        agent.reset()
        agent.step(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640), dtype=np.float32),
            np.eye(4, dtype=np.float32),
            "hello",
            intrinsic=args.camera_intrinsic
    )
    agent.reset()


    app.run(host='0.0.0.0', port=5801)
