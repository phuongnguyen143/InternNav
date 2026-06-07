"""
Inference wrapper for dualvln.

Architecture:
------------
- S2 (slow planner): VLM reads RGB (+ optional history) and the language instruction,
  then outputs either discrete actions (STOP / arrows) or pixel waypoint coordinates.
- S1 (fast executor): When S2 outputs a pixel goal, S1 turns the latent into a
  short trajectory using goal-frame + current-frame RGB-D.

Async pipeline: because S2 re-plans every `plan_step_gap` frames while S1 executes trajectories in between.

This fiel is used by scripts/realworld/http_internvla_server.py, internvla_n1_ros_node.py
"""

import copy
import itertools
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from collections import OrderedDict

from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions

DEFAULT_IMAGE_TOKEN = "<image>"


class InternVLAN1AsyncAgent:
    """
    Runs InternVLA-N1 on live RGB-D streams for robot deployment.

    Call `reset()` at the start of each navigation episode, then `step()` on every
    sensor tick. Returns an `S2Output` with one of:
      - output_action: discrete indices (0=STOP, 1=forward, 2/3=turn, 5=look down)
      - output_trajectory: XY waypoints from S1 (when S2 chose a pixel goal)
      - output_pixel: [row, col] image coordinates of the chosen waypoint
    """

    def __init__(self, args):
        self.device = torch.device(args.device)
        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"args.model_path{args.model_path}")
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.processor.tokenizer.padding_side = 'left'

        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history  # past frames sampled into the S2 prompt
        self.PLAN_STEP_GAP = args.plan_step_gap  # S2 re-plan interval (S1 runs in between)

        # Base prompt template; <instruction> is replaced per episode in step_s2().
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        # Natural-language prefixes before each <image> token in the prompt.
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        # Per-episode buffers
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_s2_idx = -100  # episode_idx when S2 last ran

        # Pending outputs from the most recent S2 call (consumed across step() calls)
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None  # RGB at the frame where S2 picked the pixel goal
        self.pixel_goal_depth = None

    def reset(self):
        """Clear episode state; call when starting a new navigation task."""
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None

        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

    def parse_actions(self, output):
        """Extract discrete action indices from VLM text (e.g. '↑' -> [1])."""
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        """Record a frame without model inference (between S2 re-plans)."""
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        self.episode_idx += 1

    def trajectory_tovw(self, trajectory, kp=1.0):
        """Convert trajectory subgoal to (linear_vel, angular_vel); unused by step()."""
        subgoal = trajectory[-1]
        linear_vel, angular_vel = kp * np.linalg.norm(subgoal[:2]), kp * subgoal[2]
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        return linear_vel, angular_vel

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """
        Main control loop entry point.

        Runs S2 when enough steps have passed, on the first call, or when look_down=True.
        Otherwise caches the frame and runs S1 if a pixel-goal latent is pending.

        pose/intrinsic are kept for API compatibility with sim/ROS wrappers; not used here.
        """
        dual_sys_output = S2Output()
        no_output_flag = self.output_action is None and self.output_latent is None

        if (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or look_down or no_output_flag:
            self.output_action, self.output_latent, self.output_pixel = self.step_s2(
                rgb, depth, pose, instruction, intrinsic, look_down
            )
            self.last_s2_idx = self.episode_idx
            dual_sys_output.output_pixel = self.output_pixel
            # Snapshot sensors at planning time; S1 compares goal frame vs current frame.
            self.pixel_goal_rgb = copy.deepcopy(rgb)
            self.pixel_goal_depth = copy.deepcopy(depth)
        else:
            self.step_no_infer(rgb, depth, pose)

        if self.output_action is not None:
            # Discrete actions are one-shot: return them and clear so S2 runs next cycle.
            dual_sys_output.output_action = copy.deepcopy(self.output_action)
            self.output_action = None
        elif self.output_latent is not None:
            # S1 expects two RGB-D pairs: [pixel-goal frame, current frame], 224x224.
            processed_pixel_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255
            processed_pixel_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224)))
            processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
            processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
            rgbs = (
                torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                .unsqueeze(0)
                .to(self.device)
            )
            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )
            trajectories = self.step_s1(self.output_latent, rgbs, depths)
            # Continuous XY path (not discretized to turn/forward symbols).
            dual_sys_output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)

        return dual_sys_output

    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """
        S2 planner: build a multi-image chat prompt and run VLM generation.

        Returns (action_seq, traj_latents, pixel_goal):
          - Digits in output -> pixel goal + latents for S1: (None, latents, [row, col])
          - Arrow/STOP tokens -> discrete actions: (action_list, None, None)

        look_down=True appends a downward-view frame as a follow-up turn (after action 5).
        """
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        else:
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}_look_down.jpg")

        if not look_down:
            # Fresh planning turn: reset chat history and inject instruction + frame history.
            self.conversation_history = []
            self.past_key_values = None

            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                # Evenly sample past frames so the VLM sees temporal context without all frames.
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            # Follow-up turn: keep prior assistant reply, add the look-down image only.
            self.input_images.append(image)
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        # Split prompt on <image> placeholders to interleave text and image objects.
        parts = split_and_clean(prompt_instruction)

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})

        self.conversation_history.append({'role': 'user', 'content': content})

        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                # use_cache=True,
                # past_key_values=self.past_key_values,
                return_dict_in_generate=True,
                # raw_input_ids=copy.deepcopy(inputs.input_ids),
            )
        output_ids = outputs.sequences

        t1 = time.time()
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        with open(f"{self.save_dir}/llm_output_{self.episode_idx: 04d}.txt", 'w') as f:
            f.write(self.llm_output)
        self.last_output_ids = copy.deepcopy(output_ids[0])
        self.past_key_values = copy.deepcopy(outputs.past_key_values)
        print(f"output {self.episode_idx}  {self.llm_output} cost: {t1 - t0}s")

        if bool(re.search(r'\d', self.llm_output)):
            # Numeric output -> pixel waypoint; swap to [row, col] for downstream drawing/control.
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            pixel_values = inputs.pixel_values
            t0 = time.time()
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
                return None, traj_latents, pixel_goal

        else:
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None

    def step_s1(self, latent, rgb, depth):
        """
        S1 inference:
        - Latent -> S1 inference -> trajectory
        - Trajectory -> action
        S1 executor: diffusion-style trajectory head conditioned on S2 latents + RGB-D pair.

        Returns raw trajectory deltas; step() converts them to an XY path via traj_to_actions().
        """
        all_trajs = self.model.generate_traj(latent, rgb, depth)
        return all_trajs
