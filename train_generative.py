# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from lpips import LPIPS
from tqdm.auto import tqdm

from src.models.data.provider import Provider
from src.models.recon.generative_3dgs_model import Generative3DGSModel
from src.models.utils.misc import load_and_merge_configs, seed_everything
from src.models.utils.loss import compute_loss

# TODO: You need a VAE to encode the conditioning image.
# We can reuse the one from Lyra/Cosmos.
from src.models.utils.cosmos_1_tokenizer import load_cosmos_1_tokenizer
from src.models.utils.model import encode_multi_view_video

def main():
    # --- Setup ---
    config_path = "configs/training/generative_3dgs.yaml" # Your new config file
    opt = load_and_merge_configs([config_path])
    seed_everything(opt.seed)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        mixed_precision=opt.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(opt.output_dir, "logs")
    )
    
    # --- Models ---
    model = Generative3DGSModel(opt)
    # TODO: Load pre-trained VAE for image encoding
    vae, _ = load_cosmos_1_tokenizer(opt.vae_path, load_encoder=True, load_jit=False)
    vae.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    
    # --- Data ---
    train_dataset = Provider("lyra_real_data", opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
    )
    
    lpips_loss = LPIPS(net="vgg").to(accelerator.device)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    vae.to(accelerator.device)

    # --- Training Loop ---
    progress_bar = tqdm(range(opt.max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Training Generative 3DGS")

    for step, batch in enumerate(train_dataloader):
        if step >= opt.max_train_steps:
            break

        with accelerator.accumulate(model):
            # 1. Prepare Inputs
            # We use the first frame of the input sequence as our condition
            condition_image = batch['images_input_vae'][:, 0:1] # (B, 1, C, H, W)
            # Encode the image to get a condition embedding
            with torch.no_grad():
                condition_embed = encode_multi_view_video(vae.encoder, condition_image, 1, opt.vae_backbone)
                condition_embed = rearrange(condition_embed, 'b t c h w -> b (t h w) c')

            # The ground truth are the subsequent frames in the video
            gt_images = batch['images_output'] # (B, V, C, H, W)
            cam_view = batch['cam_view']       # (B, V, 4, 4)
            intrinsics = batch['intrinsics']   # (B, V, 4)
            B = gt_images.shape[0]

            # 2. Diffusion Forward Process
            # Create a "clean" target: a set of random gaussians.
            # In a more advanced setup, these would be from a pre-trained 3D VAE.
            # For simplicity, we start by predicting them from a fixed noise distribution.
            clean_gaussians = torch.randn(B, model.module.num_gaussians, model.module.denoiser.gaussian_dim, device=accelerator.device)
            
            # Sample random noise
            noise = torch.randn_like(clean_gaussians)
            
            # Sample random timesteps
            timesteps = torch.randint(0, model.module.noise_scheduler.config.num_train_timesteps, (B,), device=accelerator.device).long()
            
            # Add noise to the clean gaussians
            noisy_gaussians = model.module.noise_scheduler.add_noise(clean_gaussians, noise, timesteps)
            
            # 3. Predict the noise
            predicted_noise = model(noisy_gaussians, timesteps, condition_embed)
            
            # 4. Compute Loss on the noise prediction
            # This is the standard diffusion model loss.
            noise_loss = F.mse_loss(predicted_noise, noise)

            # 5. (Optional but Recommended) Add a View-Space Reconstruction Loss
            # This loss ensures the generated scene looks correct from the known views.
            # It requires a full denoising loop inside the training step, which is expensive.
            # A simpler alternative is to predict the final 'x0' and compute the loss on that.
            if opt.lambda_views > 0:
                # Predict the original clean gaussians ('x0') from the noise
                pred_clean_gaussians = model.module.noise_scheduler.step(predicted_noise, timesteps, noisy_gaussians).pred_original_sample
                
                # Render the predicted clean scene
                render_output = model.module.render_views(pred_clean_gaussians, cam_view, intrinsics)
                pred_images = render_output['images_pred']

                # Compute reconstruction loss
                recon_loss = F.l1_loss(pred_images, gt_images)
                lpips_val = lpips_loss(pred_images.view(-1, *pred_images.shape[2:]), gt_images.view(-1, *gt_images.shape[2:])).mean()
                view_loss = recon_loss + opt.lambda_lpips * lpips_val
                
                loss = noise_loss + opt.lambda_views * view_loss
            else:
                loss = noise_loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        if accelerator.is_main_process:
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            accelerator.log({"loss": loss.item()}, step=step)
            # TODO: Add saving checkpoints

if __name__ == "__main__":
    main()