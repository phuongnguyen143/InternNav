"""Deployment wrapper for InternVLA-N1 dual-system navigation.

System 2 (S2): Qwen2.5-VL generates high-level plans — either pixel-goal coordinates
in the image or discrete action symbols (↑, ←, →, STOP).

System 1 (S1): A diffusion trajectory head (NextDiT or NavDP) consumes VLM latents
and RGB(-D) observations to produce low-level waypoint sequences, then discrete actions.

Typical inference loop (see internvla_n1_agent.py):
  s2_step() → pixel goal + traj latents  OR  direct action symbols
  s1_step_latent() → continuous trajectory → up to 4 robot actions
"""
import copy
import itertools
import re
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    InternVLAN1ForCausalLM,
    InternVLAN1ModelConfig,
)
from internnav.model.utils.device import model_load_dtype, resolve_torch_device
from internnav.model.utils.vln_utils import (
    S1Output,
    S2Output,
    chunk_token,
    split_and_clean,
    traj_to_actions,
)


class InternVLAN1Net(PreTrainedModel):
    """High-level policy that orchestrates S2 (VLM) and S1 (trajectory) inference."""

    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Union[InternVLAN1ModelConfig, ModelCfg]):
        super().__init__(config)
        self.model_config = ModelCfg(**config.model_cfg['model'])

        device = resolve_torch_device(self.model_config.device)
        load_dtype = model_load_dtype(device)
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            self.model_config.model_path,
            torch_dtype=load_dtype,
            attn_implementation="sdpa",
            device_map={"": device},
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(self.model_config.model_path)
        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = 'left'

        self.init_prompts()

        self.num_frames = self.model_config.num_frames
        self.num_history = self.model_config.num_history
        self.num_future_steps = self.model_config.num_future_steps
        self.continuous_traj = self.model_config.continuous_traj
        self.resize_w = self.model_config.resize_w
        self.resize_h = self.model_config.resize_h

        # Episode state tracked across S2 calls within one navigation episode.
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0  # Index into rgb_list; only advances on normal (non look-down) steps
        self.conversation_history = []  # Multi-turn chat used only during look-down follow-ups
        self.llm_output = ""

    def init_prompts(self):
        self.DEFAULT_IMAGE_TOKEN = "<image>"
        # S2 prompt asks the VLM for the next waypoint pixel coords or STOP.
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        # Discrete action vocabulary the VLM may emit instead of pixel coordinates.
        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

    def reset(self):
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        """Buffer an observation without running inference (e.g. during S1 execution)."""
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        self.episode_idx += 1

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """System 2: VLM reasoning step.

        Returns S2Output with either:
          - output_pixel + output_latent (numeric coords → S1 trajectory head), or
          - output_action (discrete symbols like ↑/←/STOP, bypasses S1).

        look_down=True continues a multi-turn chat with a downward-facing camera view;
        those frames are intentionally excluded from rgb_list history.
        """
        # 1. Preprocess input
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:  # Don't add look_down images to rgb_list
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)

        # 2. Prepare input for the model
        if not look_down:
            # Clear conversation history when not looking down, provide normal image history and instruction
            self.conversation_history = []
            # 2.1 instruction
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            # 2.2 images
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                # Uniformly subsample past frames so the VLM sees a fixed num_history budget.
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (self.DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1  # Only increment when not looking down to maintain correspondence with rgb_list idx
        else:
            # Continue conversation based on previous when looking down
            self.input_images.append(image)  # This image should be the look_down image
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + self.DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
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

        # 3. Autoregressive text generation (Qwen2.5-VL)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"============ output {self.episode_idx}  {self.llm_output}")
        output = S2Output()

        # 4. Route VLM output: numeric coords -> S1 latents or pixel goal; symbols -> direct actions
        use_pixel_s1 = getattr(self.model.config, 'use_pixel_goal_for_s1', False) or (
            getattr(self.model.config, 's1_goal_conditioning', 'latent') == 'pixel'
        )
        if bool(re.search(r'\d', self.llm_output)):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            # VLM outputs (row, col) but downstream expects [y, x] image coordinates.
            pixel_goal = [int(coord[1]), int(coord[0])]
            output.output_pixel = np.array(pixel_goal)

            if not use_pixel_s1:
                # One extra forward pass: extract hidden states at learnable traj-query tokens.
                image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
                with torch.no_grad():
                    traj_latents = self.model.generate_latents(output_ids, inputs.pixel_values, image_grid_thw)
                output.output_latent = traj_latents

        else:
            action_seq = self.parse_actions(self.llm_output)
            output.output_action = action_seq

        return output

    def s1_step_latent(self, rgb, depth, latent=None, pixel_goal=None):
        """System 1: diffusion trajectory generation from S2 goal conditioning.

        Uses VLM latents (default) or pixel (x, y) when config.use_pixel_goal_for_s1=True.
        """
        with torch.no_grad():
            dp_actions = self.model.generate_traj(
                traj_latents=latent,
                images_dp=rgb,
                depths_dp=depth,
                pixel_goal=pixel_goal,
            )

        if self.continuous_traj:
            action_list = traj_to_actions(dp_actions)
        else:
            # NextDiT may sample multiple trajectories; pick one at random.
            random_choice = np.random.choice(dp_actions.shape[0])
            action_list = chunk_token(dp_actions[random_choice])

        action_list = [x for x in action_list if x != 0]  # drop padding / null actions

        output = S1Output(idx=action_list[:4])
        return output
