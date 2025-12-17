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

MAX_DEPTH = 80

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
            "down",
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

    moge_depth_hw_full = torch.where(moge_mask_hw_full==0, torch.tensor(MAX_DEPTH, device=moge_depth_hw_full.device), moge_depth_hw_full)
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
    moge_depth_b11hw = torch.nan_to_num(moge_depth_b11hw, nan=MAX_DEPTH)
    moge_depth_b11hw = torch.clamp(moge_depth_b11hw, min=0, max=MAX_DEPTH)
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
    moge_depth_11hw = torch.nan_to_num(moge_depth_11hw, nan=MAX_DEPTH)
    moge_depth_11hw = torch.clamp(moge_depth_11hw, min=0, max=MAX_DEPTH)
    moge_mask_11hw = moge_mask_hw_full.unsqueeze(0).unsqueeze(0)
    moge_depth_11hw = torch.where(moge_mask_11hw==0, torch.tensor(MAX_DEPTH, device=moge_depth_11hw.device), moge_depth_11hw)

    return moge_depth_11hw, moge_mask_11hw

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
    # This list can be modified to include whatever sequence is desired
    trajectories = [
        "left",
        "right",
        "up",
        "down",
        "zoom_in",
        "zoom_out"
    ]
    
    # 2. Load Initial Image and Setup Cache ONCE
    if not args.input_image_path:
         log.critical("Input image path is required for sequential mode.")
         return

    (
        moge_image_b1chw_float,
        moge_depth_b11hw,
        _, # mask not strictly used in basic cache init
        moge_initial_w2c_b144,
        moge_intrinsics_b133,
    ) = _predict_moge_depth(
        args.input_image_path, args.height, args.width, device, moge_model
    )

    # Force a larger buffer to hold accumulated frames (1 initial + 3 per trajectory)
    # 1 + 6 trajectories * 3 frames = 19 frames. Setting to 32 to be safe.
    accumulated_frame_buffer_max = 32 
    
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

    # Initialize current state with input image pose
    # Shapes: [1, 4, 4] and [1, 3, 3]
    current_w2c = moge_initial_w2c_b144[:, 0]
    current_intrinsics = moge_intrinsics_b133[:, 0]
    
    # Keep track of the LAST generated frame to serve as visual input for the pipeline's next generation
    # Initial visual input is the loaded image
    # moge_image_b1chw_float is [B, T, C, H, W] -> [1, 1, 3, H, W]
    # We need [B, C, T, H, W] -> [1, 3, 1, H, W]
    current_visual_input = moge_image_b1chw_float.permute(0, 2, 1, 3, 4)
    
    base_save_folder = args.video_save_folder
    os.makedirs(base_save_folder, exist_ok=True)

    # 3. Sequential Generation Loop
    for traj_idx, traj_type in enumerate(trajectories):
        log.info(f"--- Starting Trajectory {traj_idx}: {traj_type} ---")
        
        # A. Generate Camera Trajectory starting from current pose
        try:
            # We assume current_w2c is [1, 4, 4], remove batch dim for util if needed or handle inside
            # generate_camera_trajectory handles batch dim if passed correctly.
            # Passing [4, 4] by squeezing batch dim 0.
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=traj_type,
                initial_w2c=current_w2c[0], 
                initial_intrinsics=current_intrinsics[0],
                num_frames=args.num_video_frames,
                movement_distance=args.movement_distance,
                camera_rotation=args.camera_rotation,
                center_depth=1.0, 
                device=device.type,
                **args.camera_gen_kwargs,
            )
            # generate_camera_trajectory returns [1, T, 4, 4], [1, T, 3, 3]
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory {traj_type}: {e}")
            break

        # B. Render Cache (Project existing PC to new views)
        # We only generate the first chunk (sample_n_frames) for the main video generation
        # NOTE: If performing autoregressive generation *within* the trajectory, that loop happens inside here too
        # For simplicity, assuming 1 chunk per trajectory for this demo structure, 
        # or we just render the first chunk needed for the pipeline.
        
        # Using 0:sample_n_frames (e.g. 121 frames)
        render_w2cs = generated_w2cs[:, 0:sample_n_frames]
        render_intrinsics = generated_intrinsics[:, 0:sample_n_frames]
        
        log.info(f"Rendering cache for {traj_type}...")
        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            render_w2cs,
            render_intrinsics,
        )

        # C. Generate Video
        log.info(f"Generating video for {traj_type}...")

        # all_rendered_warps = []
        # if args.save_buffer:
        #     all_rendered_warps.append(rendered_warp_images.clone().cpu())
        
        # Use current_visual_input (last frame of previous run) as conditioning image
        generated_output = pipeline.generate(
            prompt=args.prompt, # Use valid prompt provided in args
            image_path=current_visual_input, # Pass tensor directly
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
            return_latents=True,
        )
        
        if generated_output is None:
            log.critical(f"Guardrail blocked generation for {traj_type}.")
            break
            
        video_frames, prompt_out, latents = generated_output
        # video_frames is numpy [T, H, W, C] in 0..255
        
        # D. Update 3D Cache
        # Select last 3 frames to update cache
        T = len(video_frames)
        # indices: 1/3 mark, 2/3 mark, and last frame
        indices_to_update = [T // 3, (T * 2) // 3, T - 1]
        
        log.info(f"Updating 3D cache with frames {indices_to_update} from {traj_type}...")

        print(f"\n[DEBUG] --- Trajectory: {traj_type} ---")
        print(f"[DEBUG] video_frames length (T): {T}")
        print(f"[DEBUG] indices_to_update: {indices_to_update}")
        print(f"[DEBUG] render_w2cs shape: {render_w2cs.shape}")
        print(f"[DEBUG] render_intrinsics shape: {render_intrinsics.shape}")
        
        # Check cache internal state
        if hasattr(cache, 'input_image'):
            print(f"[DEBUG] cache.input_image shape: {cache.input_image.shape}")
        
        # Check if indices exceed dimensions
        max_idx = max(indices_to_update)
        w2c_len = render_w2cs.shape[1]
        print(f"[DEBUG] Max index requested: {max_idx}, Available W2C frames: {w2c_len}")
        
        if max_idx >= w2c_len:
            print(f"[DEBUG] ⚠️ CRITICAL: Index {max_idx} is out of bounds for W2C (size {w2c_len})")
        
        for i in indices_to_update:
            # 1. Get RGB Frame -> Tensor 0..1 (C, H, W)
            frame_np = video_frames[i]
            frame_tensor = torch.tensor(frame_np, device=device).permute(2, 0, 1).float() / 255.0
            
            # 2. Predict Depth
            pred_depth, _ = _predict_moge_depth_from_tensor(frame_tensor, moge_model)
            
            # 3. Get Pose (Index in generated trajectory)
            # generated_w2cs is [1, T, 4, 4]. i is index in video_frames.
            # Assuming 1-to-1 mapping between generated video frames and trajectory frames
            # frame i corresponds to render_w2cs index i
            pose = render_w2cs[:, i] # [1, 4, 4]
            intr = render_intrinsics[:, i] # [1, 3, 3]

            # 4. Update Cache
            # new_image expects [B, C, H, W] range [-1, 1]
            # frame_tensor is [C, H, W] 0..1
            img_input = (frame_tensor.unsqueeze(0) * 2.0) - 1.0
            
            print(f"[DEBUG] pred_depth shape: {pred_depth.shape}, pose shape: {pose.shape}, intr shape: {intr.shape}")

            cache.update_cache(
                new_image=img_input,
                new_depth=pred_depth,
                new_w2c=pose,
                new_intrinsics=intr
            )

        # E. Save Video
        save_path = os.path.join(base_save_folder, f"traj_{traj_idx}_{traj_type}.mp4")
        save_video(
            video=video_frames,
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=8,
            video_save_path=save_path,
        )
        log.info(f"Saved video to {save_path}")

        # F. Update State for next iteration
        # New start pose is the END of the current trajectory
        # Actually, generated_w2cs contains the full path. The "end" is the last frame.
        current_w2c = generated_w2cs[:, -1] # [1, 4, 4]
        current_intrinsics = generated_intrinsics[:, -1] # [1, 3, 3]
        
        # New visual input is the LAST frame of the current video
        # Need to format as [B, C, T, H, W] range [-1, 1]
        last_frame = video_frames[-1] # [H, W, C]
        last_tensor = torch.tensor(last_frame, device=device).permute(2, 0, 1).float() / 255.0
        current_visual_input = (last_tensor.unsqueeze(0).unsqueeze(2) * 2.0) - 1.0

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
        # NEW Mode: Persistent cache sequential generation
        demo_sequential(args)
    else:
        raise NotImplementedError("This script currently only supports sequential_trajectory mode.")