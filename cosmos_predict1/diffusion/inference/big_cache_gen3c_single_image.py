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
import copy
from typing import Dict, Any, Tuple
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    check_input_frames,
    validate_args,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.big_cache_3d import Cache3D_Buffer
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
    # Added missing VIPE arguments
    parser.add_argument(
        "--vipe_path",
        type=str,
        default=None,
        help="Optional: path to VIPE clip root or the mp4 file under rgb/. If set, load VIPE-formatted data directly.",
    )
    parser.add_argument(
        "--vipe_starting_frame_idx",
        type=int,
        default=0,
        help="Starting frame index within the VIPE rgb mp4 to use as the reference frame.",
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
            "look_behind", 
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

def print_cache_stats(cache, tag=""):
    """Helper function to print point cloud cache statistics."""
    if not hasattr(cache, "input_points") or cache.input_points is None:
        log.info(f"[{tag}] Cache empty or not initialized.")
        return

    try:
        # Cache structure: input_points is typically [B, F, N, V, H, W, 3]
        pts = cache.input_points
        
        # Calculate approximate memory usage (GB)
        total_bytes = 0
        
        # Sum up size of main tensors stored in cache
        for attr in ["input_points", "input_image", "input_depth", "input_mask"]:
            tensor = getattr(cache, attr, None)
            if tensor is not None:
                total_bytes += tensor.numel() * tensor.element_size()
        
        size_gb = total_bytes / (1024 ** 3)
        num_points = pts.numel() // 3 # Divide by coordinate dim
        
        # Extract buffer size (N dimension)
        # Shape usually: [B, 1, N, 1, H, W, 3] or similar depending on Permutations
        # Cache3D_Base reshapes to B, F, N, V, H, W, 3
        buffer_size = pts.shape[2] if len(pts.shape) > 2 else 1
        
        log.info(f"[{tag}] Cache Stats | Buffer(N): {buffer_size} frames | Total Points: {num_points:,} | Approx Size: {size_gb:.2f} GB")
    except Exception as e:
        log.warning(f"[{tag}] Failed to calculate cache stats: {e}")

def _predict_moge_depth(current_image_path: str | np.ndarray,
                        target_h: int, target_w: int,
                        device: torch.device, moge_model: MoGeModel):
    """Handles MoGe depth prediction for a single image."""

    input_image_rgb = None
    if isinstance(current_image_path, str):
        if current_image_path.lower().endswith('.mp4'):
            # Handle MP4 input by reading the first frame
            cap = cv2.VideoCapture(current_image_path)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    input_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if input_image_rgb is None:
                raise ValueError(f"Could not read frame from video: {current_image_path}")
        else:
            # Handle standard image input
            input_image_bgr = cv2.imread(current_image_path)
            if input_image_bgr is None:
                raise FileNotFoundError(f"Input image not found: {current_image_path}")
            input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_image_rgb = current_image_path
    
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
    # Resizing depth: (H, W) -> (1, 1, H, W) for interpolate, then squeeze
    moge_depth_hw = F.interpolate(
        moge_depth_hw_full.unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    # Resizing mask: (H, W) -> (1, 1, H, W) for interpolate, then squeeze
    moge_mask_hw = F.interpolate(
        moge_mask_hw_full.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(target_h, target_w),
        mode='nearest',  # Using nearest neighbor for binary mask
    ).squeeze(0).squeeze(0).to(torch.bool)

    # Resizing image tensor: (C, H, W) -> (1, C, H, W) for interpolate, then squeeze
    input_image_tensor_chw_target_res = F.interpolate(
        input_image_for_depth_tensor_chw.unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    moge_image_b1chw_float = input_image_tensor_chw_target_res.unsqueeze(0).unsqueeze(1) * 2 - 1

    moge_intrinsics_33 = moge_intrinsics_33_full_pixel.clone()
    # Adjust intrinsics for resized height
    moge_intrinsics_33[1, 1] *= height_scale_factor  # fy
    moge_intrinsics_33[1, 2] *= height_scale_factor  # cy
    moge_intrinsics_33[0, 0] *= width_scale_factor  # fx
    moge_intrinsics_33[0, 2] *= width_scale_factor  # cx

    moge_depth_b11hw = moge_depth_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    moge_depth_b11hw = torch.nan_to_num(moge_depth_b11hw, nan=1e4)
    moge_depth_b11hw = torch.clamp(moge_depth_b11hw, min=0, max=1e4)
    moge_mask_b11hw = moge_mask_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # Prepare initial intrinsics [B, 1, 3, 3]
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

def get_look_behind_trajectory(
    initial_w2c: torch.Tensor, 
    num_frames: int, 
    movement_distance: float,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Generates a trajectory that rotates the camera 180 degrees around the Y-axis to look behind.
    
    Args:
        initial_w2c: Initial world-to-camera matrix (4x4).
        num_frames: Number of frames in the trajectory.
        movement_distance: Not used for pure rotation, but kept for signature consistency.
        device: Device to create tensors on.
        
    Returns:
        generated_w2cs: (1, num_frames, 4, 4)
    """
    # World-to-camera to Camera-to-world
    c2w = torch.inverse(initial_w2c)
    
    # We want to rotate 180 degrees around the Y axis of the camera frame over num_frames.
    # We'll create a sequence of relative rotations.
    angles = torch.linspace(0, np.pi, num_frames, device=device)
    
    w2cs = []
    for angle in angles:
        # Rotation around Y axis
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Relative rotation matrix (rotating the camera frame)
        # Assuming Y is up, X is right, Z is back (OpenGL convention, often used in diffusion models)
        # R_y(theta) = [cos  0  sin]
        #              [ 0   1   0 ]
        #              [-sin 0  cos]
        rel_rot = torch.tensor([
            [cos_a, 0, sin_a, 0],
            [0, 1, 0, 0],
            [-sin_a, 0, cos_a, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=initial_w2c.dtype)
        
        # Apply relative rotation to the initial c2w
        # New pose = Old pose * Relative Rotation
        new_c2w = c2w @ rel_rot
        
        # Back to w2c
        w2cs.append(torch.inverse(new_c2w))
        
    return torch.stack(w2cs).unsqueeze(0)


def generate_trajectory_wrapper(args, initial_w2c, initial_intrinsics, device):
    """Wrapper to handle standard trajectories and custom 'look_behind'."""
    if args.trajectory == "look_behind":
        generated_w2cs = get_look_behind_trajectory(
            initial_w2c, 
            args.num_video_frames, 
            args.movement_distance, 
            device=device
        )
        # Intrinsics remain constant
        generated_intrinsics = initial_intrinsics.unsqueeze(0).unsqueeze(0).repeat(1, args.num_video_frames, 1, 1)
    else:
        generated_w2cs, generated_intrinsics = generate_camera_trajectory(
            trajectory_type=args.trajectory,
            initial_w2c=initial_w2c,
            initial_intrinsics=initial_intrinsics,
            num_frames=args.num_video_frames,
            movement_distance=args.movement_distance,
            camera_rotation=args.camera_rotation,
            center_depth=1.0,
            device=device,
            **args.camera_gen_kwargs,
        )
    return generated_w2cs, generated_intrinsics


def run_single_generation(
    pipeline, 
    cache, 
    moge_model, 
    initial_image, # (1, C, H, W)
    initial_prompt, 
    args, 
    device, 
    clip_name_base, 
    save_idx,
    generated_w2cs,
    generated_intrinsics,
    save_outputs=True
):
    """
    Executes a single video generation pass for a given trajectory, updates the cache, and saves results.
    """
    sample_n_frames = pipeline.model.chunk_size
    
    log.info(f"Generating 0 - {sample_n_frames} frames for trajectory {args.trajectory}")
    
    # 1. Render from cache (provides alignment and context)
    rendered_warp_images, rendered_warp_masks = cache.render_cache(
        generated_w2cs[:, 0:sample_n_frames],
        generated_intrinsics[:, 0:sample_n_frames],
    )
    
    all_rendered_warps = []
    if args.save_buffer:
        all_rendered_warps.append(rendered_warp_images.clone().cpu())
        
    # 2. Generate video
    generated_output = pipeline.generate(
        prompt=initial_prompt,
        image_path=initial_image.unsqueeze(2), # (1, C, 1, H, W)
        negative_prompt=args.negative_prompt,
        rendered_warp_images=rendered_warp_images,
        rendered_warp_masks=rendered_warp_masks,
        return_latents=True,
    )
    
    if generated_output is None:
        log.critical("Guardrail blocked video2world generation.")
        return None, None
        
    video, prompt, latents = generated_output
    
    # 3. Auto-regressive generation loop
    num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
    
    for num_iter in range(1, num_ar_iterations):
        start_frame_idx = num_iter * (sample_n_frames - 1)
        end_frame_idx = start_frame_idx + sample_n_frames
        
        log.info(f"Generating {start_frame_idx} - {end_frame_idx} frames")
        
        last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)
        pred_image_for_depth_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1) / 255.0
        
        current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
        current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
        
        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            current_segment_w2cs,
            current_segment_intrinsics,
        )
        
        if args.save_buffer:
            all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())
            
        pred_image_for_depth_bcthw_minus1_1 = pred_image_for_depth_chw_0_1.unsqueeze(0).unsqueeze(2) * 2 - 1
        
        generated_output = pipeline.generate(
            prompt=initial_prompt,
            image_path=pred_image_for_depth_bcthw_minus1_1,
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
            return_latents=True,
        )
        
        video_new, prompt, latents_new = generated_output
        video = np.concatenate([video, video_new[1:]], axis=0)
        latents = torch.cat([latents, latents_new[1:]], axis=0)
        
    # 4. Update Cache with the full generated trajectory
    # We process in batches to avoid OOM
    log.info("Updating cache with generated trajectory...")
    batch_size = 5
    T_total = video.shape[0]
    
    for i in range(0, T_total, batch_size):
        # Prepare batch
        batch_video = video[i : min(i + batch_size, T_total)] # (B, H, W, C) numpy
        batch_w2cs = generated_w2cs[0, i : min(i + batch_size, T_total)] # (B, 4, 4)
        batch_intrinsics = generated_intrinsics[0, i : min(i + batch_size, T_total)] # (B, 3, 3)
        
        batch_tensor_01 = torch.from_numpy(batch_video).permute(0, 3, 1, 2).float().to(device) / 255.0 # (B, C, H, W)
        
        # Predict depth for batch
        for j in range(batch_tensor_01.shape[0]):
            img_tensor = batch_tensor_01[j] # (C, H, W)
            pred_depth, pred_mask = _predict_moge_depth_from_tensor(img_tensor, moge_model) # (1, 1, H, W)
            
            # Update cache
            cache.update_cache(
                new_image=(img_tensor.unsqueeze(0) * 2 - 1), # (1, C, H, W), [-1, 1]
                new_depth=pred_depth,
                new_w2c=batch_w2cs[j].unsqueeze(0),
                new_intrinsics=batch_intrinsics[j].unsqueeze(0),
                new_mask=pred_mask,
                depth_alignment=True, # Align new depth to existing cache!
                alignment_method="non_rigid"
            )
            
            # Print stats
            if j == batch_tensor_01.shape[0] - 1: # Print once per batch
                print_cache_stats(cache, tag=f"Traj {args.trajectory} Update {i+j}/{T_total}")

    # 5. Save outputs
    if save_outputs:
        final_video_to_save = video
        final_width = args.width
        
        # Handle save_buffer visualization
        if args.save_buffer and all_rendered_warps:
            squeezed_warps = [t.squeeze(0) for t in all_rendered_warps]
            if squeezed_warps:
                n_max = max(t.shape[1] for t in squeezed_warps)
                padded_t_list = []
                for sq_t in squeezed_warps:
                    current_n_i = sq_t.shape[1]
                    padding_needed_dim1 = n_max - current_n_i
                    pad_spec = (0,0, 0,0, 0,0, 0,padding_needed_dim1, 0,0)
                    padded_t = F.pad(sq_t, pad_spec, mode='constant', value=-1.0)
                    padded_t_list.append(padded_t)
                
                full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)
                T_total_warp, _, C_dim, H_dim, W_dim = full_rendered_warp_tensor.shape
                
                buffer_video_TCHnW = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
                buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_total_warp, C_dim, H_dim, n_max * W_dim)
                buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5) * 255.0
                buffer_numpy_TCHWstacked = buffer_video_TCHWstacked.cpu().numpy().astype(np.uint8)
                buffer_numpy_THWC = np.transpose(buffer_numpy_TCHWstacked, (0, 2, 3, 1))
                
                min_len = min(buffer_numpy_THWC.shape[0], final_video_to_save.shape[0])
                buffer_numpy_THWC = buffer_numpy_THWC[:min_len]
                final_video_to_save = final_video_to_save[:min_len]

                final_video_to_save = np.concatenate([buffer_numpy_THWC, final_video_to_save], axis=2)
                final_width = args.width * (1 + n_max)

        # File naming
        save_name = f"{clip_name_base}_{save_idx}"
        if args.trajectory != "none":
            save_name += f"_{args.trajectory}"
            
        # Save pose
        generated_c2ws = generated_w2cs.inverse()
        pose_save_path = os.path.join(args.video_save_folder, "pose", f"{save_name}.npz")
        os.makedirs(os.path.dirname(pose_save_path), exist_ok=True)
        pose_list = []
        for i in range(generated_c2ws.shape[1]):
            pose = generated_c2ws[0, i].cpu().numpy().reshape(4, 4)
            pose_list.append((i, pose))
        pose_data = np.stack([p for _, p in pose_list], axis=0)
        pose_inds = np.array([idx for idx, _ in pose_list])
        np.savez(pose_save_path, data=pose_data, inds=pose_inds)
        
        # Save intrinsics
        intrinsics_save_path = os.path.join(args.video_save_folder, "intrinsics", f"{save_name}.npz")
        os.makedirs(os.path.dirname(intrinsics_save_path), exist_ok=True)
        intrinsics_list = []
        for i in range(generated_intrinsics.shape[1]):
            k = generated_intrinsics[0, i].cpu().numpy()
            intrinsics_fxfycxcy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            intrinsics_list.append((i, intrinsics_fxfycxcy))
        intr_data = np.stack([p for _, p in intrinsics_list], axis=0)
        intr_inds = np.array([idx for idx, _ in intrinsics_list])
        np.savez(intrinsics_save_path, data=intr_data, inds=intr_inds)

        # Save latent
        latent_save_path = os.path.join(args.video_save_folder, "latent", f"{save_name}.pkl")
        os.makedirs(os.path.dirname(latent_save_path), exist_ok=True)
        video_latent = latents.detach().float().cpu().numpy()
        torch.save(video_latent, latent_save_path)
        
        # Save Video
        video_save_path = os.path.join(args.video_save_folder, "rgb", f"{save_name}.mp4")
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        save_video(
            video=final_video_to_save,
            fps=args.fps,
            H=args.height,
            W=final_width,
            video_save_quality=8,
            video_save_path=video_save_path,
        )
        log.info(f"Saved video to {video_save_path}")

    # Prepare last frame for potential next-stage seeding
    last_frame_tensor = torch.tensor(video[-1], device=device).permute(2, 0, 1).float() / 255.0
    last_frame_tensor = last_frame_tensor.unsqueeze(0) * 2 - 1 # (1, C, H, W) [-1, 1]
    
    return video, last_frame_tensor


def demo(args):
    """Run video-to-world generation demo with aggregated cache and 12-trajectory mode."""
    # --- 1. Distributed Setup & Device Binding ---
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        from megatron.core import parallel_state
        from cosmos_predict1.utils import distributed
        
        distributed.init()
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()
        rank = torch.distributed.get_rank()
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Deterministic Seeding ---
    misc.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- 3. Pipeline Initialization ---
    pipeline = Gen3cPipeline(
        inference_type="video2world",
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

    if args.num_gpus > 1:
        torch.distributed.barrier()

    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

    # --- 4. Load Inputs ---
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        visual_input_path = args.vipe_path if args.vipe_path is not None else args.input_image_path
        prompts = [{"prompt": args.prompt, "visual_input": visual_input_path}]

    if rank == 0:
        os.makedirs(os.path.dirname(args.video_save_folder), exist_ok=True)
    
    if args.num_gpus > 1:
        torch.distributed.barrier()

    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        current_video_path = input_dict.get("visual_input", None)
        
        if current_video_path is None: 
            continue
        if not check_input_frames(current_video_path, 1): 
            continue

        # Initial depth prediction (Run on all ranks)
        (
            moge_image_b1chw_float,
            moge_depth_b11hw,
            moge_mask_b11hw,
            moge_initial_w2c_b144,
            moge_intrinsics_b133,
        ) = _predict_moge_depth(
            current_video_path, args.height, args.width, device, moge_model
        )

        # Shared Cache
        # Increased buffer to handle accumulated points from 13+ trajectories
        cache = Cache3D_Buffer(
            frame_buffer_max=300, 
            generator=None,
            noise_aug_strength=args.noise_aug_strength,
            input_image=moge_image_b1chw_float[:, 0].clone(), 
            input_depth=moge_depth_b11hw[:, 0],       
            input_w2c=moge_initial_w2c_b144[:, 0],
            input_intrinsics=moge_intrinsics_b133[:, 0],
            filter_points_threshold=args.filter_points_threshold,
            foreground_masking=args.foreground_masking,
            input_format=["B", "C", "H", "W"]
        )
        print_cache_stats(cache, tag="Init")

        initial_cam_w2c_for_traj = moge_initial_w2c_b144[0, 0]
        initial_cam_intrinsics_for_traj = moge_intrinsics_b133[0, 0]
        clip_name_base = Path(current_video_path).stem

        if args.multi_trajectory:
            args.camera_gen_kwargs = {'radius_x_factor': 0.15, 'radius_y_factor': 0.10, 'num_circles': 2}
            
            # --- Front Trajectories ---
            front_trajectories = {
                "front_left": {"traj_idx": 0, "type": "left", "dist": [0.2, 0.3]},
                "front_right": {"traj_idx": 1, "type": "right", "dist": [0.2, 0.3]},
                "front_up": {"traj_idx": 2, "type": "up", "dist": [0.1, 0.2]},
                "front_zoom_out": {"traj_idx": 3, "type": "zoom_out", "dist": [0.3, 0.4]},
                "front_zoom_in": {"traj_idx": 4, "type": "zoom_in", "dist": [0.3, 0.4]},
                "front_clockwise": {"traj_idx": 5, "type": "clockwise", "dist": [0.4, 0.6]},
            }
            
            for name in sorted(front_trajectories.keys()):
                info = front_trajectories[name]
                args.trajectory = info["type"]
                
                random.seed(args.seed + info["traj_idx"])
                args.movement_distance = random.uniform(info["dist"][0], info["dist"][1]) * args.total_movement_distance_factor
                
                generated_w2cs, generated_intrinsics = generate_trajectory_wrapper(
                    args, initial_cam_w2c_for_traj, initial_cam_intrinsics_for_traj, device
                )
                
                run_single_generation(
                    pipeline, cache, moge_model, 
                    moge_image_b1chw_float[:, 0], current_prompt, 
                    args, device, clip_name_base, info["traj_idx"],
                    generated_w2cs, generated_intrinsics,
                    save_outputs=(rank == 0)
                )
            
            # --- Transition: Look Behind ---
            args.trajectory = "look_behind"
            args.movement_distance = 0.0
            
            generated_w2cs, generated_intrinsics = generate_trajectory_wrapper(
                args, initial_cam_w2c_for_traj, initial_cam_intrinsics_for_traj, device
            )
            
            # Run generation and capture the LAST frame/pose to seed back trajectories
            _, last_frame_tensor = run_single_generation(
                pipeline, cache, moge_model, 
                moge_image_b1chw_float[:, 0], current_prompt, 
                args, device, clip_name_base, 6,
                generated_w2cs, generated_intrinsics,
                save_outputs=(rank == 0)
            )
            
            back_start_w2c = generated_w2cs[0, -1] 
            
            # --- Back Trajectories ---
            back_trajectories = {
                "back_left": {"traj_idx": 7, "type": "left", "dist": [0.2, 0.3]},
                "back_right": {"traj_idx": 8, "type": "right", "dist": [0.2, 0.3]},
                "back_up": {"traj_idx": 9, "type": "up", "dist": [0.1, 0.2]},
                "back_zoom_out": {"traj_idx": 10, "type": "zoom_out", "dist": [0.3, 0.4]},
                "back_zoom_in": {"traj_idx": 11, "type": "zoom_in", "dist": [0.3, 0.4]},
                "back_clockwise": {"traj_idx": 12, "type": "clockwise", "dist": [0.4, 0.6]},
            }
            
            for name in sorted(back_trajectories.keys()):
                info = back_trajectories[name]
                args.trajectory = info["type"]
                
                random.seed(args.seed + info["traj_idx"])
                args.movement_distance = random.uniform(info["dist"][0], info["dist"][1]) * args.total_movement_distance_factor
                
                # Generate trajectories relative to the BACK START pose
                generated_w2cs, generated_intrinsics = generate_trajectory_wrapper(
                    args, back_start_w2c, initial_cam_intrinsics_for_traj, device
                )
                
                # Use the last frame of look_behind as the visual seed
                run_single_generation(
                    pipeline, cache, moge_model, 
                    last_frame_tensor, current_prompt, 
                    args, device, clip_name_base, info["traj_idx"],
                    generated_w2cs, generated_intrinsics,
                    save_outputs=(rank == 0)
                )

        else:
            # Single run mode
            generated_w2cs, generated_intrinsics = generate_trajectory_wrapper(
                args, initial_cam_w2c_for_traj, initial_cam_intrinsics_for_traj, device
            )
            run_single_generation(
                pipeline, cache, moge_model, 
                moge_image_b1chw_float[:, 0], current_prompt, 
                args, device, clip_name_base, 0,
                generated_w2cs, generated_intrinsics,
                save_outputs=(rank == 0)
            )

    if "RANK" in os.environ:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    demo(args)