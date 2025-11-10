import torch
import os
import numpy as np

from src.models.data.spatialvid_provider import SpatialVidProvider
from src.rendering.gs import GaussianRenderer
from src.utils.visu import save_video, create_depth_visu


def run_data_pipeline_check():
    """
    Experiment 0: A sanity check for the entire data loading and rendering pipeline.

    This script performs the following steps:
    1. Loads a single video sequence from your SpatialVID dataset, including RGB frames,
       camera poses, intrinsics, and depth maps.
    2. Creates a simple, hard-coded 3D scene (a single red sphere at the origin).
    3. Renders this 3D scene using the camera poses from your dataset.
    4. Creates a side-by-side comparison video:
       [Ground Truth Video | Rendered Sphere Motion | Ground Truth Depth]
    5. Saves the final video to the 'sanity_checks' directory.

    What to look for in the output video 'exp0_pipeline_check.mp4':
      - The Ground Truth video should play normally.
      - The Rendered Sphere's movement should perfectly mirror the camera motion.
        If the camera in the GT video moves forward, the sphere should get bigger.
        If the camera pans left, the sphere should move to the right of its frame.
      - The Depth Video should show a coherent, moving depth map of the scene.
    
    If all three videos are aligned and play correctly, your data pipeline is working.
    """
    print("--- Running Experiment 0: Data Pipeline Sanity Check ---")

    # --- 1. Define Configuration and Load Data ---
    # This dummy class simulates the OmegaConf object used during training,
    # providing all necessary parameters for the data loader to function correctly.
    class Opt:
        # --- Data Loading Config ---
        data_mode = [['spatialvid', 1]] # Use the registry entry for your dataset
        batch_size = 1
        num_workers = 0
        img_size = [256, 256]

        # --- Frame Sampling Config ---
        # We'll load 1 input frame (placeholder) and 100 target frames for rendering.
        num_input_views = 1
        num_views = 101 # Total frames to load (input + target)
        static_frame_sampling = 'uniform' # Sample frames uniformly
        
        # --- Multi-View/Video Config ---
        is_static = True # Process as a collection of static scenes
        num_input_multi_views = 1
        sample_num_input_multi_views = False
        static_view_indices_sampling = 'fixed'
        static_view_indices_fixed = ['0'] # For datasets with one video per scene
        
        # --- Technical Flags ---
        subsample_data_train_val = False # Use the whole dataset
        load_latents = False # We are loading RGB images, not pre-computed latents
        use_depth = True # We want to load and verify depth maps
        relative_translation_scale = True # Normalize camera translation
        time_embedding = False # Not needed for this check
        
        # --- Add other default attributes to prevent errors ---
        def __getattr__(self, name):
             # Return a default value for any other requested attribute
            return None

    opt = Opt()
    print("Initializing dataset provider for 'spatialvid'...")
    dataset = SpatialVidProvider("spatialvid", opt, training=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    print("Loading one batch of data...")
    batch = next(iter(dataloader))
    print("\n--- BATCH LOADED SUCCESSFULLY ---")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  - Batch key '{key}': shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  - Batch key '{key}': value={value}")
    print("---------------------------------\n")
    
    print("Loading one batch of data...")
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("\nERROR: The dataloader is empty. This might be because:")
        print("1. The `root_path` in `src/models/data/registry.py` is incorrect.")
        print("2. The metadata CSV path is incorrect or the file is empty.")
        print("3. No video/annotation files were found in the specified directories.")
        return

    # Move necessary tensors to GPU
    cam_view = batch['cam_view'].cuda()     # (B, V, 4, 4)
    intrinsics = batch['intrinsics'].cuda() # (B, V, 4)
    gt_video = batch['images_output'].cuda()   # (B, V, C, H, W)
    gt_depth = batch['depths_output'].cuda()   # (B, V, 1, H, W)
    
    B, V, C, H, W = gt_video.shape
    print(f"Successfully loaded a batch of data: {V} frames of size {H}x{W}.")

    # --- 2. Create a Dummy 3DGS Scene ---
    # A single, large, red Gaussian sphere at the world origin.
    manual_gaussian = torch.tensor([[
        # Position (x, y, z) at origin
        0.0, 0.0, 0.0,
        # Opacity (fully opaque)
        1.0,
        # Scale (a sphere of radius ~0.5)
        0.5, 0.5, 0.5,
        # Rotation (identity quaternion)
        1.0, 0.0, 0.0, 0.0,
        # Color (red)
        1.0, 0.0, 0.0
    ]], device='cuda', dtype=torch.float32)

    # Note: These are ACTIVATED values, not raw model outputs.
    # The renderer expects them in this final format.
    gaussians = manual_gaussian.unsqueeze(0).expand(B, -1, -1) # (B, N, 14)

    # --- 3. Render the Scene ---
    class RenderOpt:
        img_size = opt.img_size
        znear = 0.1
        zfar = 500

    print("Rendering the dummy scene with loaded camera poses...")
    renderer = GaussianRenderer(RenderOpt())
    bg_color = torch.ones(3, device='cuda', dtype=torch.float32) # White background

    render_output = renderer.render(gaussians, cam_view, bg_color, intrinsics)
    rendered_video = render_output['images_pred']

    # --- 4. Create and Save Visualization ---
    output_dir = "sanity_checks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Colorize the ground truth depth map for visualization
    print("Creating depth visualization...")
    gt_depth_vis = create_depth_visu(gt_depth)

    # Combine the three videos into a single grid: [GT | Render | Depth]
    # Ensure all have the same channel count (3)
    # gt_depth_vis might be BGR, so we convert to RGB
    gt_depth_vis = gt_depth_vis[..., [2,1,0], :, :] # BGR to RGB
    
    # Clamp videos to valid range [0, 1] before saving
    gt_video = gt_video.clamp(0, 1)
    rendered_video = rendered_video.clamp(0, 1)
    gt_depth_vis = gt_depth_vis.clamp(0, 1)

    # Concatenate along the width dimension
    comparison_grid = torch.cat([gt_video, rendered_video, gt_depth_vis], dim=-1)

    # The save_video function expects (B, T, C, H, W)
    print(f"Saving comparison video to '{output_dir}/exp0_pipeline_check.mp4'...")
    save_video(comparison_grid, output_dir, name="exp0_pipeline_check", fps=30)
    
    print("\n--- Experiment 0 Successful! ---")
    print(f"Check the output video at: {os.path.abspath(os.path.join(output_dir, 'exp0_pipeline_check.mp4'))}")
    print("Verify that the sphere's motion matches the camera motion in the ground truth video.")


if __name__ == "__main__":
    run_data_pipeline_check()