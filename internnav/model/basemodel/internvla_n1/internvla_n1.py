"""InternVLA-N1 core model: Qwen2.5-VL backbone + S1
"""
import inspect
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .internvla_n1_arch import InternVLAN1MetaForCausalLM, InternVLAN1MetaModel

TRAJ_TOKEN_INDEX = 151667   
IMAGE_TOKEN_INDEX = 151655  

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class InternVLAN1ModelConfig(Qwen2_5_VLConfig):
    """HuggingFace config for InternVLA-N1; inherits all Qwen2.5-VL settings."""

    model_type = "internvla_n1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Extra deployment/training settings (system1 type, n_query, etc.) live here.
        self.model_cfg = kwargs.get('model_cfg', None)
        self.use_pixel_goal_for_s1 = kwargs.get('use_pixel_goal_for_s1', False)
        self.s1_pixel_goal_norm_size = kwargs.get('s1_pixel_goal_norm_size', 224.0)

    # If transformer version is newer than 4.48.0, use the following code to expose the text_config attributes
    # def __post_init__(self, **kwargs):
    #     super().__post_init__(**kwargs)
    #     self._expose_text_config_attrs()

    # def _expose_text_config_attrs(self):
    #     """Mirror text_config on the root config for legacy InternVLA checkpoints."""
    #     text_config = getattr(self, "text_config", None)
    #     if text_config is None:
    #         return
    #     text_config_cls = self.sub_configs["text_config"]
    #     text_params = list(inspect.signature(text_config_cls.__init__).parameters.keys())
    #     text_params += ["rope_parameters", "rope_scaling", "rope_theta"]
    #     for key in text_params:
    #         if hasattr(text_config, key) and not hasattr(self, key):
    #             setattr(self, key, getattr(text_config, key))


class InternVLAN1Model(InternVLAN1MetaModel, Qwen2_5_VLModel):
    """System 2 backbone: Qwen2.5-VL transformer + vision encoder.

    InternVLAN1MetaModel adds System 1 side modules on top:
      latent_queries, traj_dit (or navdp), memory encoder, etc.
    """

    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(InternVLAN1Model, self).__init__(config)


class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM):
    """Top-level model used at inference and training time.
    """

    config_class = InternVLAN1ModelConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type == "internvla_n1"

        self.model = InternVLAN1Model(config)
        self.rope_deltas = None
        # Standard LM head: hidden state -> next-token logits (same as any causal LM).
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        # ImageNet mean/std for normalizing RGB before DepthAnythingV2 (async S1 only).
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        t_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        traj_images: Optional[torch.Tensor] = None,
        traj_depths: Optional[torch.Tensor] = None,
        video_frame_num: Optional[torch.Tensor] = None,
        traj_poses: Optional[torch.Tensor] = None,
        pixel_coords_gt: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ------------------------------------------------------------------
        #  1 Build the input embedding sequence (text + vision + traj).
        # Token IDs alone are not enough: images and traj queries need real vectors.
        # ------------------------------------------------------------------
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # Replace IMAGE_TOKEN_INDEX slots with patch embeddings from the vision encoder.
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Replace TRAJ_TOKEN_INDEX slots with learnable "latent_queries".
            # These are NOT text tokens — they are trainable vectors that absorb
            # navigation intent from the transformer and pass it to System 1.
            n_traj_tokens = (input_ids == TRAJ_TOKEN_INDEX).sum().item()
            traj_idx = input_ids == TRAJ_TOKEN_INDEX
            latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
            H = latent_queries.shape[-1]
            latent_queries = latent_queries.contiguous().view(-1, H)
            if n_traj_tokens != 0:
                inputs_embeds[traj_idx] = latent_queries

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # ------------------------------------------------------------------
        # 2 Compute position IDs 3D rotary embeddings
        # Image patches change sequence length, so positions are not simply 0..N-1.
        # ------------------------------------------------------------------
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ------------------------------------------------------------------
        # 3 Run the Qwen2.5-VL transformer and predict next-token logits
        # ------------------------------------------------------------------
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # ------------------------------------------------------------------
        # 4 (training) System 1 trajectory loss.
        # Requires labels + traj_images/traj_poses/t_s_pos (pixel_goal_only batches).
        # ------------------------------------------------------------------
        loss = None
        has_s1_supervision = (
            labels is not None
            and traj_poses is not None
            and traj_images is not None
            and t_s_pos is not None
        )
        if labels is not None and not has_s1_supervision:
            raise ValueError("Batch has labels but no S1 traj fields (t_s_pos, traj_images, traj_poses). ")
        if has_s1_supervision:
            # Pull hidden states at traj-query positions, these encode "where to go".
            # t_s_pos[b] is the start index of the n_query latent tokens for batch item b.
            
            traj_hidden_states = []
            for b in range(hidden_states.shape[0]):
                traj_hidden_states.append(hidden_states[b, t_s_pos[b] : t_s_pos[b] + self.config.n_query, :])

            traj_hidden_states = torch.stack(traj_hidden_states, dim=0)
            traj_hidden_states = traj_hidden_states.unsqueeze(1).repeat(1, traj_poses.size(1), 1, 1).flatten(0, 1)
            loss_mask = torch.arange(traj_images.size(1), device=self.device).expand(
                traj_images.size(0), traj_images.size(1)
            ) < video_frame_num.unsqueeze(1)

            per_frame_pixel = None
            pixel_valid_per_frame = None
            if (
                'nextdit' in self.get_system1_type()
                and self.uses_pixel_goal_for_s1()
                and pixel_coords_gt is not None
            ):
                per_frame_pixel = pixel_coords_gt.unsqueeze(1).repeat(1, traj_poses.size(1), 1).flatten(0, 1)
                pixel_valid = ~torch.isnan(pixel_coords_gt).any(dim=-1)
                pixel_valid_per_frame = pixel_valid.unsqueeze(1).repeat(1, traj_poses.size(1)).flatten(0, 1)

            run_s1_loss = True
            if (
                'nextdit' in self.get_system1_type()
                and self.uses_pixel_goal_for_s1()
                and (per_frame_pixel is None or not pixel_valid_per_frame.any())
            ):
                # No valid pixel goal in this batch, so we skip S1 loss (e.g. turn/stop-only batch).
                run_s1_loss = False

            # Two S1 backends: NextDiT (flow-matching DiT) or NavDP (DDPM policy).
            if run_s1_loss and 'nextdit' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    # "async" = robot keeps moving while S2 re-plans.
                    # S1 sees TWO frames: the frame where S2 picked the pixel goal,
                    # plus the robot's current view. DepthAnythingV2 encodes visual memory.
                    cur_images = traj_images.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    bsz = cur_images.size(0)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1).permute(0, 1, 4, 2, 3)


                    #####
                    # images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
                    rgb_model = self.get_model().rgb_model
                    rgb_model_dtype = next(rgb_model.parameters()).dtype
                    images_dp = images_dp.to(dtype=rgb_model_dtype)
                    images_dp_norm = (
                        images_dp - self._resnet_mean.to(device=images_dp.device, dtype=rgb_model_dtype)
                    ) / self._resnet_std.to(device=images_dp.device, dtype=rgb_model_dtype)
                    #####
                    
                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1))[0]
                        .unflatten(dim=0, sizes=(bsz, -1))
                    )

                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)

                    goal_tokens = self.build_s1_goal_tokens(traj_hidden_states, per_frame_pixel)
                    latents = torch.cat([memory_tokens, goal_tokens], dim=1)
                else:
                    goal_tokens = self.build_s1_goal_tokens(traj_hidden_states, per_frame_pixel)
                    latents = goal_tokens

                # Flow-matching training (like diffusion, but predicts velocity not noise):
                #   noisy_pose = (1 - sigma) * clean_pose + sigma * random_noise
                # The DiT learns to predict (noise - clean_pose), the "flow" direction.
                relative_poses = traj_poses.flatten(0, 1)
                bsz = relative_poses.shape[0]
                noise = torch.randn(relative_poses.shape, device=relative_poses.device, dtype=relative_poses.dtype)
                u = torch.rand(size=(bsz,), device="cpu")
                indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
                sigmas = self.get_sigmas(
                    timesteps, latents.device, n_dim=relative_poses.shape[-1], dtype=relative_poses.dtype
                )

                noisy_trajectory = (1 - sigmas) * relative_poses + sigmas * noise
                action_features = self.get_model().action_encoder(noisy_trajectory)
                pos_ids = torch.arange(relative_poses.shape[1]).reshape(1, -1).repeat(bsz, 1).to(relative_poses.device)
                pos_embed = self.get_model().pos_encoding(pos_ids)
                action_features += pos_embed

                noise_pred = self.get_model().traj_dit(
                    x=action_features,
                    timestep=timesteps,
                    z_latents=latents,
                )
                noise_pred = self.get_model().action_decoder(noise_pred)
                target = noise - relative_poses  # flow-matching velocity target
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                mask = loss_mask.flatten(0, 1)[:, None, None]
                if pixel_valid_per_frame is not None:
                    mask = mask & pixel_valid_per_frame[:, None, None]
                masked_loss = loss * mask
                if mask.sum() > 0:
                    loss = masked_loss.sum() / mask.sum() / (loss.shape[1] * loss.shape[2])
                else:
                    loss = None
            elif run_s1_loss and 'navdp' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    cur_images = traj_images.flatten(0, 1)
                    cur_depths = traj_depths.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    pix_goal_depths = traj_depths[:, 0:1].repeat(1, traj_depths.size(1), 1, 1).flatten(0, 1)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1)  # (bs*select_size, 2, 224, 224, 3)
                    depths_dp = torch.stack([pix_goal_depths, cur_depths], dim=1).unsqueeze(
                        -1
                    )  # (bs*select_size, 2, 224, 224, 1)
                    pred_pg, noise = self.model.navdp.forward_vlm_traj(
                        traj_hidden_states, images_dp, depths_dp, tensor_label_actions=traj_poses
                    )
                    pg_action_loss = (pred_pg - noise).square()
                    mask = loss_mask.flatten(0, 1)[:, None, None]
                    masked_loss = pg_action_loss * mask
                    loss = masked_loss.sum() / mask.sum() / (pg_action_loss.shape[1] * pg_action_loss.shape[2])

            elif run_s1_loss:
                raise NotImplementedError

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate_latents(self, input_ids, pixel_values, image_grid_thw):
        """Bridge System 2 n System 1 at inference time.

          1. Takes the full generated token sequence from S2.
          2. Appends N_QUERY special traj tokens to the end.
          3. Runs one more forward pass through the VLM.
          4. Returns the final-layer hidden states at those traj positions.

        Returns: [batch, n_query, hidden_size]  (typically hidden_size=3584)
        """
        input_ids.to(self.get_model().device)

        # Embed the generated text tokens.
        with torch.no_grad():
            text_embeds = self.get_model().embed_tokens(input_ids)

        # Prepare learnable query vectors (same ones used during training).
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
        image_idx = input_ids == IMAGE_TOKEN_INDEX
        N_QUERY = self.get_n_query()

        # Append traj placeholder tokens so the transformer can attend to them.
        input_ids = torch.cat([input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY]).to(input_ids.device)], dim=1)

        # Re-inject vision embeddings at image token positions.
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).unsqueeze(0)
        text_embeds[image_idx] = image_embeds.to(text_embeds.device)[: image_idx.sum(), :]

        # Concatenate: [prompt+generation embeddings | latent_query embeddings]
        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)

        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=text_embeds,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        # Keep only the last N_QUERY positions — these are the S1 conditioning vectors.
        hidden_states = outputs.hidden_states[-1][:, -N_QUERY:, :]
        return hidden_states

    def generate_traj(
        self,
        traj_latents,
        images_dp,
        depths_dp=None,
        pixel_goal=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
    ):
        """System 1 inference: turn S2 latents into a robot trajectory.
          Start from random noise and iteratively denoise into a clean path of
          predict_step_nums waypoints, each with (x, y, z) relative to the robot.

        Returns: [num_sample_trajs, predict_step_nums, 3]
        """
        if 'nextdit' in self.get_system1_type():
            scheduler = FlowMatchEulerDiscreteScheduler()
            # device = traj_latents.device
            # dtype = traj_latents.dtype

            # # Project VLM hidden dim (3584) → NextDiT conditioning dim (768).
            # traj_latents = self.get_model().cond_projector(traj_latents)
            if self.uses_pixel_goal_for_s1():
                if pixel_goal is None:
                    raise ValueError("pixel_goal is required when use_pixel_goal_for_s1=True")
                device = images_dp.device if images_dp is not None else self.get_model().device
                dtype = next(self.get_model().parameters()).dtype
                pixel_tensor = torch.as_tensor(pixel_goal, device=device, dtype=dtype)
                if pixel_tensor.dim() == 1:
                    pixel_tensor = pixel_tensor.unsqueeze(0)
                goal_tokens = self.build_s1_goal_tokens(pixel_coords_gt=pixel_tensor)
            else:
                if traj_latents is None:
                    raise ValueError("traj_latents is required when use_pixel_goal_for_s1=False")
                device = traj_latents.device
                dtype = traj_latents.dtype
                goal_tokens = self.build_s1_goal_tokens(traj_hidden_states=traj_latents)
            if 'async' in self.get_system1_type():
                with torch.no_grad():
                    images_dp = images_dp.permute(0, 1, 4, 2, 3)
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
                    self.get_model().rgb_model.to(dtype)
                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                        .unflatten(dim=0, sizes=(1, -1))
                    )
                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)
                hidden_states = torch.cat([memory_tokens, goal_tokens], dim=1)
            else:
                hidden_states = goal_tokens
            # Classifier-free guidance (CFG): run the DiT twice per step 
            # once with zero conditioning (unconditional) and once with real latents.
            # Final prediction = uncond + scale * (cond - uncond). Higher scale = stronger
            # adherence to the VLM's navigation intent.
            hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
            hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
            batch_size = goal_tokens.shape[0]
            latent_size = predict_step_nums       # 32 future waypoints
            latent_channels = 3                   # each waypoint is (x, y, z)

            # Start denoising from pure Gaussian noise.
            latents = randn_tensor(
                shape=(batch_size * num_sample_trajs, latent_size, latent_channels),
                generator=None,
                device=device,
                dtype=dtype,
            )

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

            hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

            # Iterative denoising loop: noise → clean trajectory over num_inference_steps.
            for t in scheduler.timesteps:
                # Encode current noisy waypoints + sinusoidal position along the path.
                latent_features = self.get_model().action_encoder(latents)
                pos_ids = (
                    torch.arange(latent_features.shape[1])
                    .reshape(1, -1)
                    .repeat(batch_size, 1)
                    .to(latent_features.device)
                )
                pos_embed = self.get_model().pos_encoding(pos_ids)
                latent_features += pos_embed  # [num_sample_trajs, t, 384]
                latent_model_input = latent_features.repeat(2, 1, 1)
                if hasattr(scheduler, "scale_model_input"):
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict noise model_output
                noise_pred = self.get_model().traj_dit(
                    x=latent_model_input,
                    timestep=t.unsqueeze(0)
                    .expand(latent_model_input.shape[0])
                    .to(latent_model_input.device, torch.long),
                    z_latents=hidden_states_input,
                )

                noise_pred = self.get_model().action_decoder(noise_pred)

                # CFG: blend unconditional and conditional noise predictions.
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents).prev_sample
            return latents

        elif 'navdp' in self.get_system1_type():
            # NavDP: pretrained DDPM policy — delegates entirely to the navdp module.
            if 'async' in self.get_system1_type():
                all_trajs = self.model.navdp.predict_pointgoal_action_async(
                    traj_latents.to(self.get_model().device), images_dp, depths_dp
                )
            else:
                all_trajs = self.model.navdp.predict_pointgoal_action(traj_latents.to(self.get_model().device))
            return all_trajs
