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

DEFAULT_SUBGOALS = [
    "Go straight to the fire extinguisher. Turn right and go straight, stop at the green plant",
]

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''
subgoals = list(DEFAULT_SUBGOALS)
default_subgoals = list(DEFAULT_SUBGOALS)
current_subgoal_idx = 0
all_subgoals_done = False


def get_current_instruction():
    global subgoals, current_subgoal_idx, all_subgoals_done
    if all_subgoals_done:
        return None
    if subgoals:
        return subgoals[current_subgoal_idx]
    return args.instruction


def advance_to_next_subgoal():
    global current_subgoal_idx, all_subgoals_done
    if not subgoals:
        return False
    if current_subgoal_idx >= len(subgoals) - 1:
        all_subgoals_done = True
        print(f"All {len(subgoals)} subgoals completed.")
        return False
    current_subgoal_idx += 1
    print(
        f"Subgoal {current_subgoal_idx}/{len(subgoals)} finished. "
        f"Loading next: {subgoals[current_subgoal_idx]}"
    )
    agent.reset()
    return True




@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time, subgoals, current_subgoal_idx, all_subgoals_done
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
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        current_subgoal_idx = 0
        all_subgoals_done = False
        if data.get('subgoals'):
            subgoals = list(data['subgoals'])
            print(f"Loaded {len(subgoals)} subgoals from client")
        else:
            subgoals = list(default_subgoals)
            if subgoals:
                print(f"Using {len(subgoals)} default server subgoals")
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    instruction = get_current_instruction()
    if instruction is None:
        return jsonify({
            'discrete_action': [0],
            'all_subgoals_done': True,
            'subgoal_idx': current_subgoal_idx,
            'total_subgoals': len(subgoals),
        })

    print(
        f"Subgoal {current_subgoal_idx + 1}/{len(subgoals)}: {instruction}"
        if subgoals
        else f"Instruction is {instruction}"
    )

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

        json_output = {
            'subgoal_idx': current_subgoal_idx,
            'total_subgoals': len(subgoals),
            'instruction': instruction,
            'subgoal_finished': False,
            'all_subgoals_done': all_subgoals_done,
        }
        if dual_sys_output.output_action is not None:
            json_output['discrete_action'] = dual_sys_output.output_action
            if dual_sys_output.output_action == [0]:
                json_output['subgoal_finished'] = True
                if subgoals:
                    advance_to_next_subgoal()
                    json_output['all_subgoals_done'] = all_subgoals_done
                    if not all_subgoals_done:
                        json_output['next_instruction'] = subgoals[current_subgoal_idx]
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

# /home/phuongnh/khang/InternNav/checkpoints/DualVLN-pixel-goal-v1
# /home/phuongnh/khang/InternNav/checkpoints/dit/260626/only-s1/InternVLA-N1-DualVLN-office-rtx5090-v2
# /home/phuongnh/khang/InternNav/checkpoints/dit/290626-only-s1/InternVLA-N1-DualVLN-office-rtx5090-v1
# /home/phuongnh/khang/InternNav/checkpoints/navdp/240626-no-llm
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    ### note: if using pixel goal as condition, set use_pixel_goal_for_s1 = true
    parser.add_argument("--model_path", type=str, default="checkpoints/navdp/240626-no-llm")
    parser.add_argument(
        "--use_pixel_goal_for_s1",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ###
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
        help="Single instruction used with --no_subgoals.",
    )
    parser.add_argument(
        "--subgoals",
        nargs="+",
        default=DEFAULT_SUBGOALS,
        help="Ordered subgoal instructions. Advances to the next on STOP.",
    )
    parser.add_argument(
        "--no_subgoals",
        action="store_true",
        help="Disable subgoal mode and use --instruction only.",
    )
    parser.add_argument(
        "--model_path_original",
        default="checkpoints/InternVLA-N1-DualVLN",
    )
    
    args = parser.parse_args()

    if args.no_subgoals:
        default_subgoals = []
        subgoals = []
        print(f"Single-instruction mode: {args.instruction}")
    else:
        default_subgoals = list(args.subgoals)
        subgoals = list(default_subgoals)
        print(f"Subgoal mode: {len(subgoals)} subgoals (auto-advance on STOP)")
        for i, sg in enumerate(subgoals):
            print(f"  {i + 1}. {sg}")

    

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


# python scripts/realworld/http_internvla_server_new_subgoal.py \
#   --subgoals \
#     "Go straight and stop at the green plant" \
#     "Turn left. Stop when you are facing left" \
#     "Move straight and stop at the second green plant"
