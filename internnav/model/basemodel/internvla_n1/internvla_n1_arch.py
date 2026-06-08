"""InternVLA-N1 architecture extensions mixed into the Qwen2.5-VL backbone.

InternVLAN1MetaModel adds System-1 modules on top of Qwen2_5_VLModel:
  - latent_queries: learnable tokens that bridge S2 (VLM) → S1 (trajectory head)
  - nextdit path:  traj_dit + flow-matching scheduler + action encoder/decoder
  - navdp path:    pretrained NavDP diffusion policy (async mode only)
  - async extras:  DepthAnythingV2 + MemoryEncoder + QFormer for visual memory

config.system1 selects the S1 backend, e.g. "nextdit", "nextdit_async", "navdp_async".
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

LatentEmbSize = 768  # Projected VLM hidden dim fed to NextDiT cross-attention
MODEL_PATH_TO = "checkpoints"


def build_navdp(navdp_cfg, memory_size):
    from .navdp import NavDP_Policy_DPT_CriticSum_DAT

    navdp = NavDP_Policy_DPT_CriticSum_DAT(memory_size=memory_size, navdp_version=0.1)
    navdp.load_model()
    return navdp


def build_traj_dit(config):
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    from .nextdit_crossattn_traj import NextDiTCrossAttn, NextDiTCrossAttnConfig

    dit = NextDiTCrossAttn(NextDiTCrossAttnConfig(latent_embedding_size=LatentEmbSize))
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    return dit, noise_scheduler


def build_depthanythingv2(config):
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import (
        DepthAnythingV2,
    )

    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
    DAv2_model = DepthAnythingV2(**model_configs['vits'])
    DAv2_model.load_state_dict(
        torch.load(f'{MODEL_PATH_TO}/depth_anything_v2_vits.pth', map_location="cpu")
    )  # download from https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
    rgb_model = DAv2_model.pretrained

    return rgb_model


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class MemoryEncoder(nn.Module):
    """
    Transformer encoder over spatial visual features for async S1 memory."""

    def __init__(self, hidden_size=384, num_heads=6, num_layers=3, max_len=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.memory_pos = nn.Parameter(torch.randn(max_len, hidden_size))

    def forward(self, memory, memory_mask=None):
        """
        memory: (B, N, C)
        memory_mask: (B, N)
        """
        B, N, C = memory.shape
        pos = self.memory_pos[:N, :].unsqueeze(0).expand(B, -1, -1)  # (B, N, C)
        memory = memory + pos
        encoded_memory = self.encoder(memory, src_key_padding_mask=memory_mask)
        return encoded_memory


class QFormer(nn.Module):
    """
    Compresses variable-length visual memory into a fixed set of query tokens."""

    def __init__(self, num_query=32, hidden_size=768, num_layers=3, num_heads=12):
        super().__init__()
        self.num_query = num_query
        self.hidden_size = hidden_size

        self.query_tokens = nn.Parameter(torch.randn(num_query, hidden_size))
        self.query_pos = nn.Parameter(torch.randn(num_query, hidden_size))

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.visual_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, visual_feats, visual_attn_mask=None):
        B = visual_feats.size(0)

        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        query_tokens = query_tokens + self.query_pos.unsqueeze(0)

        out = self.decoder(query_tokens, visual_feats, memory_key_padding_mask=visual_attn_mask)
        return out


class InternVLAN1MetaModel:
    """
    Mixin that attaches System-1 trajectory modules to the VLM backbone.
    """

    def __init__(self, config):
        super(InternVLAN1MetaModel, self).__init__(config)
        if hasattr(config, "system1"):
            # Learnable queries inserted at TRAJ_TOKEN_INDEX positions during forward().
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))

            if config.system1 in (None, "none", ""):
                pass
            elif 'nextdit' in config.system1:
                # NextDiT: flow-matching diffusion transformer with VLM cross-attention.
                self.traj_dit, self.noise_scheduler = build_traj_dit(config)
                self.action_encoder = nn.Linear(3, 384, bias=True)   # pose → feature
                self.pos_encoding = SinusoidalPositionalEncoding(384)
                self.action_decoder = nn.Linear(384, 3, bias=True)   # feature → pose
                # Project VLM hidden states (3584) down to NextDiT conditioning dim (768).
                self.cond_projector = nn.Sequential(
                    nn.Linear(3584, LatentEmbSize), nn.GELU(approximate="tanh"), nn.Linear(LatentEmbSize, LatentEmbSize)
                )

                if 'async' in config.system1:
                    # Async mode adds goal+current RGB memory alongside VLM latents.
                    self.rgb_model = build_depthanythingv2(config)
                    self.memory_encoder = MemoryEncoder()
                    self.rgb_resampler = QFormer()

            elif 'navdp' in config.system1:
                if 'async' in config.system1:
                    # NavDP: pretrained DDPM diffusion policy with RGB-D encoder.
                    self.navdp = build_navdp(config, memory_size=2)
            else:
                raise NotImplementedError

    def initialize_vision_modules(self, model_args):
        """
        Lazy-init S1 modules during training (S2 checkpoint may ship without S1 weights)."""
        if model_args.system1 in (None, "none", ""):
            return

        if 'nextdit' in model_args.system1:
            self.traj_dit, self.noise_scheduler = build_traj_dit(model_args)
            self.action_encoder = nn.Linear(3, 384, bias=True)
            self.pos_encoding = SinusoidalPositionalEncoding(384)
            self.action_decoder = nn.Linear(384, 3, bias=True)

            self.cond_projector = nn.Sequential(
                nn.Linear(3584, LatentEmbSize), nn.GELU(approximate="tanh"), nn.Linear(LatentEmbSize, LatentEmbSize)
            )

            if 'async' in model_args.system1:
                self.rgb_model = build_depthanythingv2(model_args)
                self.memory_encoder = MemoryEncoder()
                self.rgb_resampler = QFormer()
        elif 'navdp' in model_args.system1:
            if 'async' in model_args.system1:
                self.navdp = build_navdp(model_args, memory_size=2)
        else:
            raise NotImplementedError

        self.config.system1 = model_args.system1
        self.config.n_query = model_args.n_query
        if getattr(self, 'latent_queries', None) is None:
            print("random initiation the latent_queries !!!")
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))


class InternVLAN1MetaForCausalLM(ABC):
    """
    hared helpers for InternVLAN1ForCausalLM (system1 type, noise schedule, etc.)."""

    @abstractmethod
    def get_model(self):
        pass

    def get_mm_projector(self):
        return self.get_model().mm_projector

    def get_n_query(self):
        return self.get_model().config.n_query

    def get_system1_type(self):
        return self.get_model().config.system1

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        """
        Look up flow-matching noise level (sigma) for each training timestep."""
        sigmas = self.get_model().noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.get_model().noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
