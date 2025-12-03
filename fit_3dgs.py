import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio
import einops
from omegaconf import OmegaConf

from gsplat.rendering import rasterization
from src.models.utils.misc import load_and_merge_configs, seed_everything
from src.models.data.provider import Provider
from src.utils.visu import save_video

# --- Helper: Lift Depth to 3D (Initialization) ---
def lift_data_to_3dgs(rgbs, depths, intrinsics, c2ws, subsample_rate=1):
    """
    Converts RGB-D video into a point cloud of 3D Gaussians.
    """
    device = rgbs.device
    B, T, C, H, W = rgbs.shape
    
    means_list = []
    colors_list = []
    
    # Grid for projection
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x_grid = x_grid.flatten().float()
    y_grid = y_grid.flatten().float()

    print(f"Initializing 3DGS from {T} frames (subsample={subsample_rate})...")
    
    # Iterate over frames to accumulate points
    for t in range(0, T, subsample_rate):
        # Unpack params
        fx, fy, cx, cy = intrinsics[0, t] # Assuming batch 0
        c2w = c2ws[0, t]
        
        # Get Frame Data
        rgb = rgbs[0, t].permute(1, 2, 0).reshape(-1, 3)
        depth = depths[0, t, 0].flatten()
        
        # Filter valid depth
        mask = (depth > 0) & (depth < 100) # Simple validity check
        
        # Subsample pixels for initialization (prevent creating 100M points)
        # We only keep a % of pixels per frame to keep optimization fast
        pixel_keep_rate = 0.05 
        rand_mask = torch.rand_like(depth) < pixel_keep_rate
        mask = mask & rand_mask
        
        if mask.sum() == 0: continue

        z_c = depth[mask]
        x_c = (x_grid[mask] - cx) * z_c / fx
        y_c = (y_grid[mask] - cy) * z_c / fy
        
        xyz_cam = torch.stack([x_c, y_c, z_c], dim=-1)
        
        # Transform to World
        rot = c2w[:3, :3]
        trans = c2w[:3, 3]
        xyz_world = xyz_cam @ rot.T + trans
        
        means_list.append(xyz_world)
        colors_list.append(rgb[mask])

    if not means_list:
        raise ValueError("No valid points found to initialize 3DGS. Check depth maps.")

    means = torch.cat(means_list, dim=0)
    colors = torch.cat(colors_list, dim=0)
    
    num_points = means.shape[0]
    print(f"Initialized {num_points} Gaussians.")
    
    # Initialize Parameters
    # Scales: log space. Initialize small.
    scales = torch.ones((num_points, 3), device=device) * -4.0 # exp(-4) is small
    
    # Rotations: Identity quaternions (1, 0, 0, 0)
    quats = torch.zeros((num_points, 4), device=device)
    quats[:, 0] = 1.0
    
    # Opacities: Logit space. Initialize high (sigmoid(4) ~ 0.98)
    opacities = torch.ones((num_points), device=device) * 4.0
    
    # Colors: Logit space (inverse sigmoid)
    # clip to avoid inf
    colors = colors.clamp(0.01, 0.99)
    colors = torch.logit(colors)

    return means, scales, quats, opacities, colors

# --- Model Class ---
class SimpleGaussianModel(nn.Module):
    def __init__(self, means, scales, quats, opacities, colors):
        super().__init__()
        self.means = nn.Parameter(means)
        self.scales = nn.Parameter(scales)
        self.quats = nn.Parameter(quats)
        self.opacities = nn.Parameter(opacities)
        self.colors = nn.Parameter(colors) # SH degree 0
        
    def forward(self):
        # Activation functions
        active_scales = torch.exp(self.scales)
        active_opacities = torch.sigmoid(self.opacities)
        active_colors = torch.sigmoid(self.colors)
        active_quats = F.normalize(self.quats, dim=-1)
        return self.means, active_scales, active_quats, active_opacities, active_colors

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/demo/lyra_dynamic.yaml')
    parser.add_argument('--iterations', type=int, default=1000, help="Optimization steps")
    parser.add_argument('--output_dir', type=str, default='outputs/fitted_3dgs')
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    # 1. Config Loading Logic
    inference_config = load_and_merge_configs(['configs/inference/default.yaml', args.config])
    
    if 'config_path' in inference_config and inference_config.config_path is not None:
        if isinstance(inference_config.config_path, str):
            model_config = OmegaConf.load(inference_config.config_path)
        else:
            model_config = load_and_merge_configs(inference_config.config_path)
        config = OmegaConf.merge(model_config, inference_config)
    else:
        config = inference_config

    if config.get('subsample_data_train_val') is None:
        config.subsample_data_train_val = False
    
    seed_everything(42)
    device = torch.device('cuda')
    
    # --- FIX: Force 'fixed' sampling to prevent loading target views (6-11) ---
    config.num_input_multi_views = 1
    config.static_view_indices_fixed = ['0']
    config.static_view_indices_sampling = 'fixed' 
    
    print(f"Output Directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Data (Video + Depth)
    print("Loading Data...")
    provider = Provider(config.dataset_name, config, training=False)
    batch = provider.get_item(0) 
    
    # Unpack Batch
    rgbs = batch['images_output'].to(device).unsqueeze(0) # [1, T, 3, H, W]
    
    if 'depths_output' in batch:
        depths = batch['depths_output'].to(device).unsqueeze(0) # [1, T, 1, H, W]
    else:
        raise ValueError("Depth data missing. Ensure 'use_depth: true' in config or data provider.")

    c2ws = batch['c2ws_input'].to(device).unsqueeze(0) # [1, T, 4, 4]
    intrinsics = batch['intrinsics_input'].to(device).unsqueeze(0) # [1, T, 4]
    
    H, W = config.img_size
    T = rgbs.shape[1]
    
    print(f"Loaded Sequence: {T} frames, {H}x{W}")

    # 3. Initialize Gaussians (Lift Data)
    means, scales, quats, opacities, colors = lift_data_to_3dgs(rgbs, depths, intrinsics, c2ws)
    
    model = SimpleGaussianModel(means, scales, quats, opacities, colors).to(device)
    
    # 4. Optimization Setup
    optimizer = optim.Adam([
        {'params': [model.means], 'lr': 0.00016 * args.lr},
        {'params': [model.colors], 'lr': 0.0025 * args.lr},
        {'params': [model.opacities], 'lr': 0.05 * args.lr},
        {'params': [model.scales], 'lr': 0.005 * args.lr},
        {'params': [model.quats], 'lr': 0.001 * args.lr},
    ], lr=args.lr)

    # 5. Training Loop
    print("Starting Optimization...")
    pbar = tqdm(range(args.iterations))
    
    for step in pbar:
        # Sample random frame
        t = np.random.randint(0, T)
        
        # Prepare Render Inputs
        gt_image = rgbs[0, t].permute(1, 2, 0) # [H, W, 3]
        
        c2w = c2ws[0, t]
        viewmat = torch.inverse(c2w).transpose(0, 1).unsqueeze(0) # [1, 4, 4]
        
        K_vals = intrinsics[0, t]
        K = torch.tensor([
            [K_vals[0], 0, K_vals[2]],
            [0, K_vals[1], K_vals[3]],
            [0, 0, 1]
        ], device=device).unsqueeze(0) # [1, 3, 3]
        
        # Get Gaussian Params
        m, s, q, o, c = model()
        
        # Rasterize
        rendered_image, alpha, meta = rasterization(
            means=m,
            quats=q,
            scales=s,
            opacities=o,
            colors=c,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            backgrounds=torch.ones(1, 3, device=device),
            render_mode='RGB'
        )
        
        rendered_image = rendered_image[0] # [H, W, 3]
        
        # Loss
        l1_loss = F.l1_loss(rendered_image, gt_image)
        loss = l1_loss 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f} | Pts: {m.shape[0]}")

    # 6. Render Trajectory
    print("Rendering Final Trajectory...")
    frames = []
    
    with torch.no_grad():
        m, s, q, o, c = model()
        for t in tqdm(range(T)):
            c2w = c2ws[0, t]
            viewmat = torch.inverse(c2w).transpose(0, 1).unsqueeze(0)
            
            K_vals = intrinsics[0, t]
            K = torch.tensor([
                [K_vals[0], 0, K_vals[2]],
                [0, K_vals[1], K_vals[3]],
                [0, 0, 1]
            ], device=device).unsqueeze(0)

            image, _, _ = rasterization(
                means=m, quats=q, scales=s, opacities=o, colors=c,
                viewmats=viewmat, Ks=K, width=W, height=H,
                packed=False, backgrounds=torch.ones(1, 3, device=device),
                render_mode='RGB'
            )
            
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            frames.append(img_np)

    # Save Video
    save_path = os.path.join(args.output_dir, "fitted_trajectory.mp4")
    imageio.mimwrite(save_path, frames, fps=24, quality=8)
    print(f"Saved fitted video to {save_path}")

if __name__ == "__main__":
    main()