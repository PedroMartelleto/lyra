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

import torch
from einops import rearrange
import math

from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
    forward_warp,
    reliable_depth_mask_range_batch,
    unproject_points,
)
from cosmos_predict1.diffusion.inference.camera_utils import align_depth

class Cache3D_Base:
    def __init__(
        self,
        input_image,
        input_depth,
        input_w2c,
        input_intrinsics,
        input_mask=None,
        input_format=None,
        input_points=None,
        weight_dtype=torch.float32,
        is_depth=True,
        device="cuda",
        filter_points_threshold=1.0,
        foreground_masking=False,
    ):
        """
        input_image: Tensor with varying dimensions.
        input_format: List of dimension labels corresponding to input_image's dimensions.
                      E.g., ['B', 'C', 'H', 'W'], ['B', 'F', 'C', 'H', 'W'], etc.
        """
        self.weight_dtype = weight_dtype
        self.is_depth = is_depth
        self.device = device
        self.filter_points_threshold = filter_points_threshold
        self.foreground_masking = foreground_masking
        if input_format is None:
            assert input_image.dim() == 4
            input_format = ["B", "C", "H", "W"]

        # Map dimension names to their indices in input_image
        format_to_indices = {dim: idx for idx, dim in enumerate(input_format)}
        input_shape = input_image.shape
        if input_mask is not None:
            input_image = torch.cat([input_image, input_mask], dim=format_to_indices.get("C"))

        # B (batch size), F (frame count), N dimensions: no aggregation during warping.
        # Only broadcasting over F to match the target w2c.
        # V: aggregate via concatenation or duster
        B = input_shape[format_to_indices.get("B", 0)] if "B" in format_to_indices else 1  # batch
        F = input_shape[format_to_indices.get("F", 0)] if "F" in format_to_indices else 1  # frame
        N = input_shape[format_to_indices.get("N", 0)] if "N" in format_to_indices else 1  # buffer
        V = input_shape[format_to_indices.get("V", 0)] if "V" in format_to_indices else 1  # view
        H = input_shape[format_to_indices.get("H", 0)] if "H" in format_to_indices else None
        W = input_shape[format_to_indices.get("W", 0)] if "W" in format_to_indices else None

        # Desired dimension order
        desired_dims = ["B", "F", "N", "V", "C", "H", "W"]

        # Build permute order based on input_format
        permute_order = []
        for dim in desired_dims:
            idx = format_to_indices.get(dim)
            if idx is not None:
                permute_order.append(idx)
            else:
                # Placeholder for dimensions to be added later
                permute_order.append(None)

        # Remove None values for permute operation
        permute_indices = [idx for idx in permute_order if idx is not None]
        input_image = input_image.permute(*permute_indices)

        # Insert dimensions of size 1 where necessary
        for i, idx in enumerate(permute_order):
            if idx is None:
                input_image = input_image.unsqueeze(i)

        # Now input_image has the shape B x F x N x V x C x H x W
        if input_mask is not None:
            self.input_image, self.input_mask = input_image[:, :, :, :, :3], input_image[:, :, :, :, 3:]
            self.input_mask = self.input_mask.to(self.device)
        else:
            self.input_mask = None
            self.input_image = input_image
        
        # EXPERIMENT: Keep entire cache on GPU to avoid CPU<->GPU transfer bottleneck
        self.input_image = self.input_image.to(weight_dtype).to(self.device)

        if input_points is not None:
            self.input_points = input_points.reshape(B, F, N, V, H, W, 3).to(self.device)
            self.input_depth = None
        else:
            input_depth = torch.nan_to_num(input_depth, nan=100)
            input_depth = torch.clamp(input_depth, min=0, max=100)
            if weight_dtype == torch.float16:
                input_depth = torch.clamp(input_depth, max=70)
            
            # Compute points on GPU
            self.input_points = (
                self._compute_input_points(
                    input_depth.reshape(-1, 1, H, W),
                    input_w2c.reshape(-1, 4, 4),
                    input_intrinsics.reshape(-1, 3, 3),
                )
                .to(weight_dtype)
                .reshape(B, F, N, V, H, W, 3)
                .to(self.device)
            )
            self.input_depth = input_depth.to(self.device)

        if self.filter_points_threshold < 1.0 and input_depth is not None:
            input_depth = input_depth.reshape(-1, 1, H, W)
            # Ensure calculations happen on device
            depth_mask = reliable_depth_mask_range_batch(input_depth.to(self.device), ratio_thresh=self.filter_points_threshold).reshape(B, F, N, V, 1, H, W)
            if self.input_mask is None:
                self.input_mask = depth_mask
            else:
                self.input_mask = self.input_mask * depth_mask
        
        self.boundary_mask = None
        if foreground_masking:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(input_depth.to(self.device))
            self.boundary_mask = (~depth_mask).reshape(B, F, N, V, 1, H, W).to(self.device)

    def _compute_input_points(self, input_depth, input_w2c, input_intrinsics):
        # Ensure computation happens on a capable device if possible, or CPU
        comp_device = input_depth.device if input_depth.device.type == "cuda" else self.device
        input_points = unproject_points(
            input_depth.to(comp_device),
            input_w2c.to(comp_device),
            input_intrinsics.to(comp_device),
            is_depth=self.is_depth,
        )
        return input_points

    def update_cache(self):
        raise NotImplementedError

    def input_frame_count(self) -> int:
        return self.input_image.shape[1]

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        # Optimized render_cache that avoids expanding the input tensor in memory
        bs, F_target, _, _ = target_w2cs.shape
        B, F_in, N, V, C, H, W = self.input_image.shape
        assert bs == B
        
        # Determine total number of warps: B * F_target * N
        total_items = B * F_target * N
        
        # Warp chunk size (how many image-pose pairs to process at once on GPU)
        warp_chunk_size = 8
        
        rendered_warp_images = []
        rendered_warp_masks = []
        rendered_warp_depth = []

        # Ensure targets are on device
        target_w2cs = target_w2cs.to(self.device)
        target_intrinsics = target_intrinsics.to(self.device)

        # Iterate in chunks
        for i in range(0, total_items, warp_chunk_size):
            actual_chunk_size = min(warp_chunk_size, total_items - i)
            indices = torch.arange(i, i + actual_chunk_size)
            
            # Map flat index back to [b, f_target, n]
            n_idx = indices % N
            rem = indices // N
            f_target_idx = rem % F_target
            b_idx = rem // F_target
            
            # Determine source frame index.
            if F_in == 1:
                f_src_idx = torch.zeros_like(indices) # Always 0
            else:
                f_src_idx = (f_target_idx + start_frame_idx).clamp(0, F_in - 1)

            # Gather source data (Input Images/Points) directly from GPU tensors
            # Slicing keeps them on GPU
            batch_imgs = self.input_image[b_idx, f_src_idx, n_idx]
            batch_pts = self.input_points[b_idx, f_src_idx, n_idx]
            
            batch_masks = None
            if self.input_mask is not None:
                batch_masks = self.input_mask[b_idx, f_src_idx, n_idx]
            
            batch_bmasks = None
            if self.boundary_mask is not None:
                if self.boundary_mask.shape[1] == 1: # F=1
                    batch_bmasks = self.boundary_mask[b_idx, 0, n_idx, :, 0, 0]
                else:
                    batch_bmasks = self.boundary_mask[b_idx, f_src_idx, n_idx, :, 0, 0]

            # Gather Targets
            chunk_target_w2c = target_w2cs[b_idx, f_target_idx]
            chunk_target_intr = target_intrinsics[b_idx, f_target_idx]

            # forward_warp expects [Chunk, C, H, W] if V=1 (squeezed)
            if V == 1:
                batch_imgs = batch_imgs.squeeze(1)
                batch_pts = batch_pts.squeeze(1)
                if batch_masks is not None: batch_masks = batch_masks.squeeze(1)
            
            with torch.no_grad():
                (
                    chunk_warped_img,
                    chunk_warped_mask,
                    chunk_warped_depth,
                    _,
                ) = forward_warp(
                    frame1=batch_imgs,
                    mask1=batch_masks,
                    depth1=None,
                    transformation1=None,
                    transformation2=chunk_target_w2c,
                    intrinsic1=chunk_target_intr,
                    intrinsic2=chunk_target_intr,
                    render_depth=render_depth,
                    world_points1=batch_pts,
                    foreground_masking=self.foreground_masking,
                    boundary_mask=batch_bmasks,
                )
                
                # Move chunks to CPU to accumulate final result (avoiding VRAM OOM for output tensor)
                # Although we keep points on GPU, the rendered 4D video volume is huge.
                rendered_warp_images.append(chunk_warped_img.to("cpu"))
                rendered_warp_masks.append(chunk_warped_mask.to("cpu"))
                if render_depth:
                    rendered_warp_depth.append(chunk_warped_depth.to("cpu"))

        # Concatenate all chunks (on CPU)
        pixels = torch.cat(rendered_warp_images, dim=0)
        masks = torch.cat(rendered_warp_masks, dim=0)
        
        # Reshape back to [B, F_target, N, C, H, W]
        pixels = rearrange(pixels, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
        masks = rearrange(masks, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
        
        if render_depth:
            depths = torch.cat(rendered_warp_depth, dim=0)
            pixels_depth = rearrange(depths, "(b f n) h w -> b f n h w", b=bs, f=F_target, n=N)
            return pixels_depth.to(self.device), masks.to(self.device)

        # Move result back to device for downstream model consumption
        # WARNING: If N and F are large, this line is the memory bottleneck.
        return pixels.to(self.device), masks.to(self.device)


class Cache3D_Buffer(Cache3D_Base):
    def __init__(self, frame_buffer_max=0, noise_aug_strength=0, generator=None, **kwargs):
        super().__init__(**kwargs)
        self.frame_buffer_max = frame_buffer_max
        self.noise_aug_strength = noise_aug_strength
        self.generator = generator

    def update_cache(self, new_image, new_depth, new_w2c, new_mask=None, new_intrinsics=None, depth_alignment=True, alignment_method="non_rigid", prune_threshold=0.05):  # 3D cache
        # Move inputs to device for alignment/computation
        new_image = new_image.to(self.weight_dtype).to(self.device)
        new_depth = new_depth.to(self.weight_dtype).to(self.device)
        new_w2c = new_w2c.to(self.weight_dtype).to(self.device)
        if new_intrinsics is not None:
            new_intrinsics = new_intrinsics.to(self.weight_dtype).to(self.device)

        new_depth = torch.nan_to_num(new_depth, nan=1e4)
        new_depth = torch.clamp(new_depth, min=0, max=1e4)

        if depth_alignment:
            target_depth, target_mask = self.render_cache(
                new_w2c.unsqueeze(1), new_intrinsics.unsqueeze(1), render_depth=True
            )
            # target_depth is [B, 1, N, H, W]
            # Flatten N (buffers) by taking min depth (closest surface)
            if target_depth.shape[2] > 0:
                # We want alignment against the best matching surface, usually min depth
                # Valid mask logic: must be valid in rendering
                valid_mask = target_mask[:, 0, :, 0] > 0.5 # B, N, H, W
                flat_depths = target_depth[:, 0] # B, N, H, W
                # Set invalid to inf for min
                flat_depths[~valid_mask] = 1e9
                min_depth, _ = torch.min(flat_depths, dim=1) # B, H, W
                
                # Use min_depth as target, and Union of masks as target_mask
                target_depth = min_depth.unsqueeze(0).unsqueeze(0) # 1, 1, H, W
                target_mask_any = valid_mask.any(dim=1).unsqueeze(0).unsqueeze(0).float() # 1, 1, H, W
                
                target_depth = target_depth.squeeze()
                target_mask = target_mask_any.squeeze()
            else:
                target_depth = target_depth[:, 0, 0] # likely empty or zeros
                target_mask = target_mask[:, 0, 0]

            if alignment_method == "rigid":
                new_depth = (
                    align_depth(
                        new_depth.squeeze(),
                        target_depth,
                        target_mask.bool(),
                    )
                    .reshape_as(new_depth)
                    .detach()
                )
            elif alignment_method == "non_rigid":
                with torch.enable_grad():
                    new_depth = (
                        align_depth(
                            new_depth.squeeze(),
                            target_depth,
                            target_mask.bool(),
                            k=new_intrinsics.squeeze(),
                            c2w=torch.inverse(new_w2c.squeeze()),
                            alignment_method="non_rigid",
                            num_iters=100,
                            lambda_arap=0.1,
                            smoothing_kernel_size=3,
                        )
                        .reshape_as(new_depth)
                        .detach()
                    )
            else:
                raise NotImplementedError
        
        # Basic Pruning: Check redundancy against existing buffer
        if prune_threshold > 0 and self.input_image.shape[2] > 0:
             with torch.no_grad():
                 rendered_depths, rendered_masks = self.render_cache(
                     new_w2c.unsqueeze(1), new_intrinsics.unsqueeze(1), render_depth=True
                 )
                 # rendered_depths: [B, 1, N, H, W]
                 # rendered_masks: [B, 1, N, 1, H, W]
                 
                 valid_mask = rendered_masks[:, 0, :, 0] > 0.5 # B, N, H, W
                 r_depth_vals = rendered_depths[:, 0] # B, N, H, W
                 r_depth_vals[~valid_mask] = 1e9
                 
                 min_dist, _ = torch.min(r_depth_vals, dim=1, keepdim=True) # B, 1, H, W
                 
                 # Check overlap
                 dist_diff = torch.abs(new_depth - min_dist)
                 is_redundant = (dist_diff < prune_threshold) & (min_dist < 1e8)
                 
                 if new_mask is None:
                     new_mask = torch.ones_like(new_depth)
                 
                 # Mask out redundant points in the new frame
                 # Fixed type error: use ~ on boolean tensor
                 new_mask = new_mask * (~is_redundant).float()
        
        new_points = unproject_points(new_depth, new_w2c, new_intrinsics, is_depth=self.is_depth)
        
        # EXPERIMENT: Keep all updates on GPU
        # new_image = new_image.cpu()
        # new_depth = new_depth.cpu()
        # new_points = new_points.cpu()
        # if new_mask is not None: new_mask = new_mask.cpu()

        if self.filter_points_threshold < 1.0:
            B, F, N, V, C, H, W = self.input_image.shape
            # Ensure on device
            new_depth_gpu = new_depth.to(self.device) 
            new_depth_gpu = new_depth_gpu.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(new_depth_gpu, ratio_thresh=self.filter_points_threshold).reshape(B, 1, H, W)
            # depth_mask = depth_mask.cpu()
            if new_mask is None:
                new_mask = depth_mask
            else:
                new_mask = new_mask * depth_mask
        
        if new_mask is not None:
             new_mask = new_mask.to(self.device)
            
        if self.frame_buffer_max > 1:  # newest frame first
            if self.input_image.shape[2] < self.frame_buffer_max:
                self.input_image = torch.cat([new_image[:, None, None, None], self.input_image], 2)
                self.input_points = torch.cat([new_points[:, None, None, None], self.input_points], 2)
                if self.input_mask is not None:
                    self.input_mask = torch.cat([new_mask[:, None, None, None], self.input_mask], 2)
            else:
                self.input_image[:, :, 0] = new_image[:, None, None]
                self.input_points[:, :, 0] = new_points[:, None, None]
                if self.input_mask is not None:
                    self.input_mask[:, :, 0] = new_mask[:, None, None]
        else:
            self.input_image = new_image[:, None, None, None]
            self.input_points = new_points[:, None, None, None]
            if new_mask is not None:
                if self.input_mask is None:
                     self.input_mask = new_mask[:, None, None, None]
                else:
                     self.input_mask[:, :, 0] = new_mask[:, None, None]


    def render_cache(
        self,
        target_w2cs,
        target_intrinsics,
        render_depth: bool = False,
        start_frame_idx: int = 0,  # For consistency with Cache4D
    ):
        assert start_frame_idx == 0, "start_frame_idx must be 0 for Cache3D_Buffer"

        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(self.device)
        pixels, masks = super().render_cache(
            target_w2cs, target_intrinsics, render_depth
        )
        pixels = pixels.to(output_device)
        masks = masks.to(output_device)
        if not render_depth:
            noise = torch.randn(pixels.shape, generator=self.generator, device=pixels.device, dtype=pixels.dtype)
            per_buffer_noise = (
                torch.arange(start=pixels.shape[2] - 1, end=-1, step=-1, device=pixels.device)
                * self.noise_aug_strength
            )
            pixels = pixels + noise * per_buffer_noise.reshape(1, 1, -1, 1, 1, 1)  # B, F, N, C, H, W
        return pixels, masks


class Cache3D_BufferSelector(Cache3D_Base):
    def __init__(self, frame_buffer_max=1, mask_for_max_buffer_model: bool = True, mask_full_threshold: float = 0.9, **kwargs):
        """A buffer that holds many initialization frames and selects top-K by overlap per target.

        This class does not support update_cache. It assumes multiple source frames are provided
        at initialization time via the 'N' (buffer) dimension.
        """
        super().__init__(**kwargs)
        self.frame_buffer_max = max(int(frame_buffer_max), 1)
        self.mask_for_max_buffer_model = bool(mask_for_max_buffer_model)
        self.mask_full_threshold = float(mask_full_threshold)

    def update_cache(self, *args, **kwargs): 
        raise NotImplementedError("Cache3D_BufferSelector does not support update_cache")

    def render_cache(
        self,
        target_w2cs,
        target_intrinsics,
        render_depth: bool = False,
        start_frame_idx: int = 0,
    ):
        # Warp from all buffer frames first
        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(self.device)

        pixels_all, masks_all = super().render_cache(
            target_w2cs, target_intrinsics, render_depth, start_frame_idx
        )  # shapes: [B, F, N, C, H, W] (pixels) and [B, F, N, 1, H, W] (masks)

        B, F, N = pixels_all.shape[0], pixels_all.shape[1], pixels_all.shape[2]
        if N <= self.frame_buffer_max:
            pixels_sel, masks_sel = pixels_all, masks_all
        else:
            # Compute per-buffer overlap score: sum over frames and pixels
            # masks_all: [B, F, N, 1, H, W]
            overlap_scores = masks_all.sum(dim=(1, 3, 4, 5))  # -> [B, N]

            # Select top-K for each batch independently
            k = min(self.frame_buffer_max, N)
            topk_indices = overlap_scores.topk(k=k, dim=1, largest=True, sorted=True).indices  # [B, k]

            # Gather along N dimension
            selected_pixels_list = []
            selected_masks_list = []
            for b in range(B):
                idx_b = topk_indices[b]  # [k]
                selected_pixels_list.append(pixels_all[b : b + 1, :, idx_b])  # [1, F, k, C, H, W]
                selected_masks_list.append(masks_all[b : b + 1, :, idx_b])    # [1, F, k, 1, H, W]

            pixels_sel = torch.cat(selected_pixels_list, dim=0)
            masks_sel = torch.cat(selected_masks_list, dim=0)
        if self.mask_for_max_buffer_model and not render_depth:
            # masks_sel: [B, F, k, 1, H, W]
            _masks = masks_sel.mean(dim=[3, 4, 5])  # -> [B, F, k]
            Bm, Fm, Nm = _masks.shape
            _masks_flat = rearrange(_masks, "b t n -> (b t) n")
            result_mask = torch.zeros_like(_masks_flat)
            near_full = _masks_flat >= self.mask_full_threshold
            has_near_full = near_full.any(dim=1)

            indices = near_full.float().argmax(dim=1)
            valid_rows = torch.arange(near_full.size(0), device=_masks_flat.device)[has_near_full]
            valid_indices = indices[has_near_full]
            result_mask[valid_rows, valid_indices] = 1

            invalid_rows = torch.arange(near_full.size(0), device=_masks_flat.device)[~has_near_full]
            if invalid_rows.numel() > 0:
                result_mask[invalid_rows] = 1
            result_mask = rearrange(result_mask, "(b t) n -> b t n", b=Bm, t=Fm)

            result_mask_expanded = result_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, F, k, 1, 1, 1]
            pixels_sel = (pixels_sel + 1) * result_mask_expanded - 1
            masks_sel = masks_sel * result_mask_expanded
        return pixels_sel.to(output_device), masks_sel.to(output_device)


class Cache4D(Cache3D_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_cache(self, **kwargs):
        raise NotImplementedError

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        rendered_warp_images, rendered_warp_masks = super().render_cache(target_w2cs, target_intrinsics, render_depth, start_frame_idx)
        return rendered_warp_images, rendered_warp_masks