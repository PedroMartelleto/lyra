# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from einops import repeat

from src.models.data.provider import Provider

class SpatialVidProvider(Provider):
    """
    INSTRUMENTED VERSION: A specialized Provider for the SpatialVid dataset.
    Includes detailed print statements to debug index generation.
    """

    def get_item(self, idx):
        """
        A robust, simplified override for fetching a data item.
        """
        # --- 1. Generate Indices ---
        total_num_frames = self.dataset.count_frames(idx)
        # print(f"[Provider: get_item] Video index {idx} has {total_num_frames} total frames.")
        
        if total_num_frames == 0:
            print(f"[Provider: get_item] ERROR: Video {idx} has 0 frames. Retrying with a new index.")
            raise RuntimeError(f"Video at index {idx} reports 0 frames.")

        num_input_frames = self.opt.num_input_views
        num_target_frames = self.opt.num_views - self.opt.num_input_views
        # print(f"[Provider: get_item] Requesting {num_input_frames} input frames and {num_target_frames} target frames.")

        available_indices = np.arange(total_num_frames)

        # Sample Input Frames
        replace_input = total_num_frames < num_input_frames
        input_indices = np.sort(np.random.choice(available_indices, num_input_frames, replace=replace_input))

        # Sample Target Frames
        replace_target = total_num_frames < num_target_frames
        target_indices = np.sort(np.random.choice(available_indices, num_target_frames, replace=replace_target))
        
        all_frame_indices = np.concatenate([input_indices, target_indices])
        all_view_indices = [0] * len(all_frame_indices) # SpatialVid is single-view

        # print(f"[Provider: get_item] Generated frame_indices (len={len(all_frame_indices)}): {all_frame_indices[:5]}...{all_frame_indices[-5:]}")
        # print(f"[Provider: get_item] Generated view_indices (len={len(all_view_indices)}): {all_view_indices[:5]}...")

        # --- 2. Load Data ---
        # print(f"[Provider: get_item] Calling self.dataset.get_data...")
        original_output_dict = self.dataset.get_data(
            idx,
            data_fields=self.data_fields,
            frame_indices=all_frame_indices.tolist(),
            view_indices=all_view_indices
        )
        # print(f"[Provider: get_item] self.dataset.get_data FINISHED.")

        # --- 3. Preprocess Data ---
        # print(f"[Provider: get_item] Starting preprocessing...")
        file_name = self.dataset.mp4_file_paths[idx].stem
        
        rgbs = original_output_dict[self.data_fields[0]]
        c2ws = original_output_dict[self.data_fields[1]]
        intrinsics = original_output_dict[self.data_fields[2]]
        depths = original_output_dict.get(self.data_fields[3], None)
        if depths is not None:
            depths = depths.unsqueeze(1)

        timesteps = torch.from_numpy(all_frame_indices).float()
        target_index_tensor = torch.from_numpy(target_indices)
        num_input_multi_views = 1

        preprocessed_batch = self._preprocess(
            file_name=file_name, rgbs=rgbs, c2ws=c2ws, intrinsics=intrinsics,
            depths=depths, timesteps=timesteps, latents=None,
            target_index=target_index_tensor, num_input_multi_views=num_input_multi_views
        )
        # print(f"[Provider: get_item] Preprocessing FINISHED.")
        return preprocessed_batch

    def __getitem__(self, idx):
        count = 0
        while True:
            try:
                results = self.get_item(idx)
                return results
            except Exception as e:
                count += 1
                if count > 20:
                    print(f"data loader error count {count}: {e}")
                idx = np.random.randint(0, len(self.dataset))