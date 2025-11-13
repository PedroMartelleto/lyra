# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple
import math
import torch.nn.functional as F

# We reuse the core building blocks from Lyra's original model
from src.models.utils.model import get_model_blocks
from src.rendering.gs import GaussianRenderer

# TODO: You will need to install the diffusers library: pip install diffusers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DenoisingTransformer(nn.Module):
    """
    This network predicts the noise added to a noisy 3D Gaussian tensor.
    It reuses the core Transformer/Mamba architecture from Lyra.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gaussian_dim = self.opt.output_dims # e.g., 14 for pos, opacity, scale, rot, rgb

        # Conditioning projection for the input image embedding
        self.condition_proj = nn.Linear(opt.enc_embed_dim, opt.enc_embed_dim)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(opt.enc_embed_dim, opt.enc_embed_dim),
            nn.SiLU(),
            nn.Linear(opt.enc_embed_dim, opt.enc_embed_dim),
        )

        # Input projection to map gaussians to the model's dimension
        self.input_proj = nn.Linear(self.gaussian_dim, opt.enc_embed_dim)

        # The core denoising network (reusing Lyra's blocks)
        self.blocks = get_model_blocks(
            opt.enc_embed_dim,
            opt.enc_depth,
            opt.enc_num_heads,
            opt.mlp_ratio,
            opt.use_mamba,
            opt.llrm_7m1t,
            nn.LayerNorm,
            opt.use_qk_norm,
        )

        # Final projection to predict the noise
        self.output_proj = nn.Linear(opt.enc_embed_dim, self.gaussian_dim)

    def forward(self, noisy_gaussians: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor):
        """
        Args:
            noisy_gaussians (torch.Tensor): (B, N, 14) Noisy Gaussian parameters.
            timesteps (torch.Tensor): (B,) Diffusion timesteps.
            condition (torch.Tensor): (B, T_cond, C) Embedding of the input image.
        """
        # 1. Project inputs and embeddings
        gauss_embed = self.input_proj(noisy_gaussians)
        
        # Timestep embedding
        t_embed = self.time_embed(timestep_embedding(timesteps, self.opt.enc_embed_dim)) # (B, C)
        t_embed = t_embed.unsqueeze(1) # (B, 1, C)

        # Condition embedding (we can average the conditioning tokens)
        cond_embed = self.condition_proj(condition.mean(dim=1, keepdim=True)) # (B, 1, C)

        # 2. Combine embeddings and process through the transformer
        x = gauss_embed + t_embed + cond_embed # Broadcasting adds embeddings to each token
        
        for block in self.blocks:
            x = block(x)

        # 3. Project back to predict the noise
        predicted_noise = self.output_proj(x)
        return predicted_noise

class Generative3DGSModel(nn.Module):
    """
    A conditional diffusion model for generating 3D Gaussian Splatting scenes.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.denoiser = DenoisingTransformer(opt)
        self.gaussian_renderer = GaussianRenderer(opt)
        self.num_gaussians = opt.num_gaussians_generative # A new config parameter

        # TODO: Initialize a diffusion scheduler from the diffusers library
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=opt.diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2"
        )
        
        # We need a way to get the initial image embedding, reusing Lyra's VAE encoder logic
        # This part would be handled in the training script.

    def forward(self, noisy_gaussians: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor):
        return self.denoiser(noisy_gaussians, timesteps, condition)
        
    def render_views(self, gaussians: torch.Tensor, cam_view: torch.Tensor, intrinsics: torch.Tensor):
        """ Renders the generated gaussians from given camera poses. """
        # We need to activate the raw gaussian parameters before rendering
        # Reusing the activation functions from the original Lyra model
        scale_cap = self.opt.gaussian_scale_cap
        scale_shift = 1 - math.log(scale_cap)
        scale_act = lambda x: torch.minimum(torch.exp(x-scale_shift),torch.tensor([scale_cap],device=x.device,dtype=x.dtype))
        opacity_act = lambda x: torch.sigmoid(x-2.0)
        rot_act = lambda x: F.normalize(x, dim=-1)
        rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5

        pos, opacity, scale, rotation, rgb = gaussians.split([3, 1, 3, 4, 3], dim=-1)
        
        activated_gaussians = torch.cat([
            pos,
            opacity_act(opacity),
            scale_act(scale),
            rot_act(rotation),
            rgb_act(rgb)
        ], dim=-1)

        bg_color = torch.ones(3, device=gaussians.device, dtype=gaussians.dtype)
        render_output = self.gaussian_renderer.render(activated_gaussians.unsqueeze(0), cam_view.unsqueeze(0), bg_color, intrinsics.unsqueeze(0))
        return render_output