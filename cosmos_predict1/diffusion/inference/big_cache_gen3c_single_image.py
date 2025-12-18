# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path
import cv2
from moge.model.v1 import MoGeModel
import torch
import random
import numpy as np
from typing import Dict, Any
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    check_input_frames,
    validate_args,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
import torch.nn.functional as F
torch.enable_grad(False)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    ) 
    parser.add_argument(
        "--input_image_path",
        type=str,
        help="Input image path for generating a single video",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        choices=[
            "left",
            "right",
            "up",
            "zoom_in",
            "zoom_out",
            "clockwise",
            "counterclockwise",
            "none",
        ],
        default="left",
        help="Select a trajectory type from the available options (default: original)",
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        default="center_facing",
        help="Controls camera rotation during movement: center_facing (rotate to look at center), no_rotation (keep orientation), or trajectory_aligned (rotate in the direction of movement)",
    )
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=0.3,
        help="Distance of the camera from the center of the scene",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Strength of noise augmentation on warped frames",
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="If set, save the warped images (buffer) side by side with the output video.",
    )
    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="If set, filter the points continuity of the warped images.",
    )
    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="If set, use foreground masking for the warped images.",
    )
    parser.add_argument(
        "--multi_trajectory",
        action="store_true",
        help="If set, do multi-trajectory generation used by the 3DGS decoder.",
    )
    parser.add_argument(
        "--sequential_trajectory",
        action="store_true",
        help="If set, runs a sequential trajectory generation where previous runs update the 3D cache.",
    )
    parser.add_argument(
        "--camera_gen_kwargs",
        type=Dict[str, Any],
        default={},
    )
    parser.add_argument(
        "--total_movement_distance_factor",
        type=float,
        default=1.0,
        help="Multiply multi trajectory setup with movement distance factor (larger means more movement but potentially more artifacts)",
    )
    return parser

def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args()

def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"

def _predict_moge_depth(current_image_path: str | np.ndarray,
                        target_h: int, target_w: int,
                        device: torch.device, moge_model: MoGeModel):
    """Handles MoGe depth prediction for a single image."""

    if isinstance(current_image_path, str):
        input_image_bgr = cv2.imread(current_image_path)
        if input_image_bgr is None:
            raise FileNotFoundError(f"Input image not found: {current_image_path}")
        input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_image_rgb = current_image_path
    
    # If using numpy array, ensure it's uint8 for opencv/moge resizing logic logic
    if input_image_rgb.dtype != np.uint8:
        input_image_rgb = (input_image_rgb).astype(np.uint8)

    depth_pred_h, depth_pred_w = 720, 1280

    input_image_for_depth_resized = cv2.resize(input_image_rgb, (depth_pred_w, depth_pred_h))
    input_image_for_depth_tensor_chw = torch.tensor(input_image_for_depth_resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    moge_output_full = moge_model.infer(input_image_for_depth_tensor_chw)
    moge_depth_hw_full = moge_output_full["depth"]
    moge_intrinsics_33_full_normalized = moge_output_full["intrinsics"]
    moge_mask_hw_full = moge_output_full["mask"]

    moge_depth_hw_full = torch.where(moge_mask_hw_full==0, torch.tensor(1000.0, device=moge_depth_hw_full.device), moge_depth_hw_full)
    moge_intrinsics_33_full_pixel = moge_intrinsics_33_full_normalized.clone()
    moge_intrinsics_33_full_pixel[0, 0] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 1] *= depth_pred_h
    moge_intrinsics_33_full_pixel[0, 2] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 2] *= depth_pred_h

    # Calculate scaling factor for height
    height_scale_factor = target_h / depth_pred_h
    width_scale_factor = target_w / depth_pred_w

    # Resize depth map, mask, and image tensor
    moge_depth_hw = F.interpolate(
        moge_depth_hw_full.unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    moge_mask_hw = F.interpolate(
        moge_mask_hw_full.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(target_h, target_w),
        mode='nearest', 
    ).squeeze(0).squeeze(0).to(torch.bool)

    input_image_tensor_chw_target_res = F.interpolate(
        input_image_for_depth_tensor_chw.unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    moge_image_b1chw_float = input_image_tensor_chw_target_res.unsqueeze(0).unsqueeze(1) * 2 - 1

    moge_intrinsics_33 = moge_intrinsics_33_full_pixel.clone()
    moge_intrinsics_33[1, 1] *= height_scale_factor  # fy
    moge_intrinsics_33[1, 2] *= height_scale_factor  # cy
    moge_intrinsics_33[0, 0] *= width_scale_factor  # fx
    moge_intrinsics_33[0, 2] *= width_scale_factor  # cx

    moge_depth_b11hw = moge_depth_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    moge_depth_b11hw = torch.nan_to_num(moge_depth_b11hw, nan=1e4)
    moge_depth_b11hw = torch.clamp(moge_depth_b11hw, min=0, max=1e4)
    moge_mask_b11hw = moge_mask_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    moge_intrinsics_b133 = moge_intrinsics_33.unsqueeze(0).unsqueeze(0)
    initial_w2c_44 = torch.eye(4, dtype=torch.float32, device=device)
    moge_initial_w2c_b144 = initial_w2c_44.unsqueeze(0).unsqueeze(0)

    return (
        moge_image_b1chw_float,
        moge_depth_b11hw,
        moge_mask_b11hw,
        moge_initial_w2c_b144,
        moge_intrinsics_b133,
    )

def _predict_moge_depth_from_tensor(
    image_tensor_chw_0_1: torch.Tensor, # Shape (C, H_input, W_input), range [0,1]
    moge_model: MoGeModel
):
    """Handles MoGe depth prediction from an image tensor."""
    moge_output_full = moge_model.infer(image_tensor_chw_0_1)
    moge_depth_hw_full = moge_output_full["depth"]      # (moge_inf_h, moge_inf_w)
    moge_mask_hw_full = moge_output_full["mask"]        # (moge_inf_h, moge_inf_w)

    moge_depth_11hw = moge_depth_hw_full.unsqueeze(0).unsqueeze(0)
    moge_depth_11hw = torch.nan_to_num(moge_depth_11hw, nan=1e4)
    moge_depth_11hw = torch.clamp(moge_depth_11hw, min=0, max=1e4)
    moge_mask_11hw = moge_mask_hw_full.unsqueeze(0).unsqueeze(0)
    moge_depth_11hw = torch.where(moge_mask_11hw==0, torch.tensor(1000.0, device=moge_depth_11hw.device), moge_depth_11hw)

    return moge_depth_11hw, moge_mask_11hw

trajectories_map = {
    "left": {"traj_idx": 0, "movement_distance_range": [0.2, 0.3]},
    "right": {"traj_idx": 1, "movement_distance_range": [0.2, 0.3]},
    "up": {"traj_idx": 2, "movement_distance_range": [0.1, 0.2]},
    "zoom_out": {"traj_idx": 3, "movement_distance_range": [0.3, 0.4]},
    "zoom_in": {"traj_idx": 4, "movement_distance_range": [0.3, 0.4]},
    "clockwise": {"traj_idx": 5, "movement_distance_range": [0.4, 0.6]},
}


def demo_sequential(args):
    """
    New function for continuous multi-trajectory generation with persistent 3D Cache.
    """
    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.num_gpus > 1:
        from megatron.core import parallel_state
        from cosmos_predict1.utils import distributed
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()

    # 1. Initialize models ONCE
    pipeline = Gen3cPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        disable_prompt_encoder=args.disable_prompt_encoder,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=121,
        seed=args.seed,
    )
    if args.num_gpus > 1:
        pipeline.model.net.enable_context_parallel(process_group)

    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sample_n_frames = pipeline.model.chunk_size

    # Define trajectory sequence
    trajectories = [
        "left",
        "right",
        "up",
        "zoom_in",
        "zoom_out",
        "clockwise"
    ]
    
    # 2. Load Initial Image and Setup Cache ONCE
    if not args.input_image_path:
         log.critical("Input image path is required for sequential mode.")
         return

    (
        moge_image_b1chw_float,
        moge_depth_b11hw,
        _, 
        moge_initial_w2c_b144,
        moge_intrinsics_b133,
    ) = _predict_moge_depth(
        args.input_image_path, args.height, args.width, device, moge_model
    )

    # Force a larger buffer to hold accumulated frames
    accumulated_frame_buffer_max = 20 
    
    cache = Cache3D_Buffer(
        frame_buffer_max=accumulated_frame_buffer_max, 
        generator=generator,
        noise_aug_strength=args.noise_aug_strength,
        input_image=moge_image_b1chw_float[:, 0].clone(), 
        input_depth=moge_depth_b11hw[:, 0],      
        input_w2c=moge_initial_w2c_b144[:, 0],
        input_intrinsics=moge_intrinsics_b133[:, 0],
        filter_points_threshold=args.filter_points_threshold,
        foreground_masking=args.foreground_masking,
    )

    # Initialize current state with input image pose [1, 4, 4] and [1, 3, 3]
    current_w2c = moge_initial_w2c_b144[:, 0]
    current_intrinsics = moge_intrinsics_b133[:, 0]
    
    # Keep track of the LAST generated frame for visual conditioning
    current_visual_input = moge_image_b1chw_float.permute(0, 2, 1, 3, 4)
    
    base_save_folder = args.video_save_folder
    os.makedirs(base_save_folder, exist_ok=True)

    # 3. Sequential Generation Loop
    for traj_idx, traj_type in enumerate(trajectories):
        log.info(f"--- Starting Trajectory {traj_idx}: {traj_type} ---")
        
        # A. Generate Camera Trajectory
        try:
            move_dist = random.uniform(
                trajectories_map[traj_type]["movement_distance_range"][0],
                trajectories_map[traj_type]["movement_distance_range"][1]
            ) * args.total_movement_distance_factor

            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=traj_type,
                initial_w2c=current_w2c[0],  # Pass [4, 4]
                initial_intrinsics=current_intrinsics[0], # Pass [3, 3]
                num_frames=args.num_video_frames,
                movement_distance=move_dist,
                camera_rotation=args.camera_rotation,
                center_depth=1.0, 
                device=device.type,
                **args.camera_gen_kwargs,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory {traj_type}: {e}")
            break

        # B. Render Cache with DEPTH
        render_w2cs = generated_w2cs[:, 0:sample_n_frames]
        render_intrinsics = generated_intrinsics[:, 0:sample_n_frames]
        
        log.info(f"Rendering cache for {traj_type} (with depth)...")
        # Request depth to perform z-buffering manually
        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            render_w2cs,
            render_intrinsics,
            render_depth=True
        )

        # 1. Render RGB
        rgb_warps, rgb_masks = cache.render_cache(render_w2cs, render_intrinsics, render_depth=False)
        # 2. Render Depth
        depth_warps, _ = cache.render_cache(render_w2cs, render_intrinsics, render_depth=True)
        # depth_warps is [B, T, N, H, W] (no channel dim, or squeeze it)

        # Handle Save Buffer (Visualization) - Use FULL cache for this
        buffer_vis_np = None
        if args.save_buffer:
            # rgb_warps: [B, T, N, C, H, W] -> Squeeze B -> [T, N, C, H, W]
            squeezed_warps = rgb_warps.detach().cpu().squeeze(0)
            T_dim, N_dim, C_dim, H_dim, W_dim = squeezed_warps.shape
            
            # Stack buffers horizontally: [T, C, H, N*W]
            buffer_video_TCHnW = squeezed_warps.permute(0, 2, 3, 1, 4) # [T, C, H, N, W]
            buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_dim, C_dim, H_dim, N_dim * W_dim)
            
            # Normalize to 0-255 uint8, [T, H, W_stacked, C]
            buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5).clamp(0, 1) * 255.0
            buffer_vis_np = buffer_video_TCHWstacked.permute(0, 2, 3, 1).numpy().astype(np.uint8)

        # rgb_warps: [B, T, N, 3, H, W]
        # depth_warps: [B, T, N, H, W] or [B, T, N, 1, H, W] check shape
        if depth_warps.dim() == 5:
            depth_warps = depth_warps.unsqueeze(3) # Ensure [B, T, N, 1, H, W]
        
        B_size, T_size, N_size, C_size, H_size, W_size = rgb_warps.shape
        
        # Slice 0: Recent frame (Always keep)
        recent_rgb = rgb_warps[:, :, 0:1] # [B, T, 1, 3, H, W]
        recent_mask = rgb_masks[:, :, 0:1]

        # Slice 1..N: History frames
        history_rgb = rgb_warps[:, :, 1:] # [B, T, N-1, 3, H, W]
        history_depth = depth_warps[:, :, 1:] # [B, T, N-1, 1, H, W]
        history_mask = rgb_masks[:, :, 1:] # [B, T, N-1, 1, H, W]

        # Handle case where history is empty (first iteration)
        if N_size > 1:
            # Mask out invalid depths (set to infinity)
            # Mask is 1 for valid, 0 for invalid.
            # depth > 0 check usually implies validity too, but let's use mask.
            # If mask is 0, set depth to INF
            
            # history_mask [B, T, N-1, 1, H, W]
            invalid_mask = (history_mask < 0.5) 
            history_depth_masked = history_depth.clone()
            history_depth_masked[invalid_mask] = float('inf')
            
            # Find min depth across N dimension (dim=2)
            # min_vals: [B, T, 1, H, W], min_indices: [B, T, 1, H, W]
            # Squeeze dim 3 (channel 1) for min()
            history_depth_squeezed = history_depth_masked.squeeze(3) # [B, T, N-1, H, W]
            min_depth_vals, min_indices = torch.min(history_depth_squeezed, dim=2, keepdim=True) # [B, T, 1, H, W]
            
            # Expand indices for gather: [B, T, 1, 3, H, W]
            min_indices_expanded = min_indices.unsqueeze(3).expand(-1, -1, -1, 3, -1, -1)
            
            # Gather RGB based on min depth indices
            # history_rgb: [B, T, N-1, 3, H, W]
            aggregated_history_rgb = torch.gather(history_rgb, 2, min_indices_expanded) # [B, T, 1, 3, H, W]
            
            # Save input depths for debugging as 0 -> black, 1 -> white & output for debugging
            def save_debug_depth(depth_tensor, save_path):
                depth_np = depth_tensor.squeeze().detach().cpu().contiguous().numpy() # [H, W]
                depth_np_clipped = np.clip(depth_np, 0, 10) # Clip for better visualization
                depth_np_normalized = (depth_np_clipped / 10.0 * 255.0).astype(np.uint8)
                # save with PIL
                from PIL import Image
                img = Image.fromarray(depth_np_normalized)
                img.save(save_path)
            
            def save_debug_rgb(rgb_tensor, save_path):
                rgb_np = rgb_tensor.detach().cpu().clone().contiguous().numpy() # [3, H, W]
                rgb_np = np.transpose(rgb_np, (1, 2, 0)) #
                rgb_np_clipped = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
                # save with PIL
                from PIL import Image
                img = Image.fromarray(rgb_np_clipped)
                img.save(save_path)
            
            # Compute aggregated mask (Union of masks or mask of chosen pixel?)
            # Valid if min_depth is not INF
            aggregated_history_mask = (min_depth_vals < 1e5).float().unsqueeze(3) # [B, T, 1, 1, H, W]
            # aggregated_history_rgb[~(aggregated_history_mask.bool()).expand_as(aggregated_history_rgb)] = 0

            debug_depth_save_folder = "debug/"
            os.makedirs(debug_depth_save_folder, exist_ok=True)
            save_debug_depth(min_depth_vals[0,0], os.path.join(debug_depth_save_folder, f"traj_{traj_idx}_{traj_type}_min_depth.png"))
            save_debug_depth(history_depth[0,0,0], os.path.join(debug_depth_save_folder, f"traj_{traj_idx}_{traj_type}_history_depth_0.png"))
            save_debug_rgb(aggregated_history_rgb[0, 0, 0], os.path.join(debug_depth_save_folder, f"traj_{traj_idx}_{traj_type}_aggregated_history_rgb_0.png"))
            save_debug_rgb(recent_rgb[0, 0, 0], os.path.join(debug_depth_save_folder, f"traj_{traj_idx}_{traj_type}_recent_rgb.png"))

            # Concatenate Recent + Aggregated History
            input_warps = torch.cat([recent_rgb, aggregated_history_rgb], dim=2)
            input_masks = torch.cat([recent_mask, aggregated_history_mask], dim=2)
        
        else:
            input_warps = recent_rgb
            input_masks = recent_mask
        
        input_warps = input_warps.to(device)
        input_masks = input_masks.to(device)

        # C. Generate Video
        log.info(f"Generating video for {traj_type}...")
        generated_output = pipeline.generate(
            prompt=args.prompt,
            image_path=current_visual_input,
            negative_prompt=args.negative_prompt,
            rendered_warp_images=input_warps, # [B, T, 2, 3, H, W]
            rendered_warp_masks=input_masks,  # [B, T, 2, 1, H, W]
            return_latents=True,
        )
        
        if generated_output is None:
            log.critical(f"Guardrail blocked generation for {traj_type}.")
            break
            
        video_frames, prompt_out, latents = generated_output
        # video_frames is [T, H, W, C] numpy uint8

        # Combine with buffer if requested
        final_video_to_save = video_frames
        if buffer_vis_np is not None:
             # Ensure dimensions match (they should if frames count matches)
             if buffer_vis_np.shape[0] == video_frames.shape[0]:
                 final_video_to_save = np.concatenate([buffer_vis_np, video_frames], axis=2)
                 log.info(f"Concatenated buffer visualization. Width: {final_video_to_save.shape[2]}")
        
        # D. Update 3D Cache
        # one third, two thirds, and last frame
        indices_to_update = [
            # len(video_frames) // 3,
            # 2 * len(video_frames) // 3,
            len(video_frames) - 1
        ]
        
        for i in indices_to_update:
            frame_np = video_frames[i]
            frame_tensor = torch.tensor(frame_np, device=device).permute(2, 0, 1).float() / 255.0
            
            pred_depth, _ = _predict_moge_depth_from_tensor(frame_tensor, moge_model)
            
            pose = render_w2cs[:, i]
            intr = render_intrinsics[:, i]
            
            img_input = (frame_tensor.unsqueeze(0) * 2.0) - 1.0
            
            cache.update_cache(
                new_image=img_input,
                new_depth=pred_depth,
                new_w2c=pose,
                new_intrinsics=intr
            )

        # E. Save Video
        save_path = os.path.join(base_save_folder, f"traj_{traj_idx}_{traj_type}.mp4")
        save_video(
            video=final_video_to_save,
            fps=args.fps,
            H=args.height,
            W=final_video_to_save.shape[2], # Use actual width (inc. buffer)
            video_save_quality=8,
            video_save_path=save_path,
        )
        log.info(f"Saved video to {save_path}")

        # F. Update State for next iteration
        origin_w2c = moge_initial_w2c_b144[:, 0]
        origin_intrinsics = moge_intrinsics_b133[:, 0]
        origin_visual_input = moge_image_b1chw_float.permute(0, 2, 1, 3, 4)

        current_w2c = origin_w2c
        current_intrinsics = origin_intrinsics
        current_visual_input = origin_visual_input
        
        # last_frame = video_frames[-1]
        # last_tensor = torch.tensor(last_frame, device=device).permute(2, 0, 1).float() / 255.0
        # current_visual_input = (last_tensor.unsqueeze(0).unsqueeze(2) * 2.0) - 1.0

    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    
    if args.sequential_trajectory:
        demo_sequential(args)