# src/models/data/difix_adapter.py

import os
import torch
import numpy as np
from decord import VideoReader, cpu
from typing import List, Optional, Any

from src.models.data.base import BaseDataset
from src.models.data.datafield import DataField

class DifixAdapter(BaseDataset):
    """
    A Dataset Adapter that wraps an existing dataset (e.g., Radym/Lyra) 
    but overrides the RGB loading to fetch restored videos from lets_difix.py output.
    """

    def __init__(self, original_dataset: BaseDataset, difix_output_dir: str):
        super().__init__()
        self.original_dataset = original_dataset
        self.difix_output_dir = difix_output_dir
        
        # Verify difix output exists
        if not os.path.exists(self.difix_output_dir):
            raise FileNotFoundError(f"Difix output directory not found: {self.difix_output_dir}")

    def __len__(self):
        return len(self.original_dataset)

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped dataset.
        This handles properties like 'is_static', 'mp4_file_paths', 'root_path', etc.
        required by the Provider.
        """
        return getattr(self.original_dataset, name)

    @property
    def sample_list(self):
        return self.original_dataset.sample_list

    @sample_list.setter
    def sample_list(self, value):
        self.original_dataset.sample_list = value

    def available_data_fields(self) -> List[DataField]:
        return self.original_dataset.available_data_fields()

    def num_videos(self) -> int:
        return self.original_dataset.num_videos()

    def num_views(self, video_idx: int) -> int:
        return self.original_dataset.num_views(video_idx)

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        return self.original_dataset.num_frames(video_idx, view_idx)

    def count_frames(self, video_idx: int):
        return self.num_frames(video_idx)

    def count_cameras(self, video_idx: int):
        return self.num_views(video_idx)

    def get_data(self, idx, data_fields, frame_indices=None, view_indices=None):
        return self._read_data(idx, frame_indices, view_indices, data_fields)

    def _get_difix_video_path(self, view_idx: int) -> str:
        filename = f"view_{view_idx}.mp4"
        return os.path.join(self.difix_output_dir, filename)

    def _read_data(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> dict[DataField, Any]:
        
        fields_to_load_from_orig = [f for f in data_fields if f != DataField.IMAGE_RGB]
        output_dict = {}

        # 1. Load Metadata from Original Dataset
        if fields_to_load_from_orig:
            if hasattr(self.original_dataset, 'get_data'):
                output_dict = self.original_dataset.get_data(
                    video_idx, fields_to_load_from_orig, frame_idxs, view_idxs
                )
            else:
                output_dict = self.original_dataset._read_data(
                    video_idx, frame_idxs, view_idxs, fields_to_load_from_orig
                )

        # 2. Load RGB from Difix Output
        if DataField.IMAGE_RGB in data_fields:
            # FIX: Handle numpy array truth value ambiguity
            if view_idxs is not None and len(view_idxs) > 0:
                current_view = view_idxs[0]
                # If view_idxs is a tensor or numpy array, ensure scalar
                if hasattr(current_view, 'item'):
                    current_view = current_view.item()
            else:
                current_view = 0
            
            video_path = self._get_difix_video_path(current_view)
            
            if not os.path.exists(video_path):
                # Try fallback to view 0 if specific view not found (common in monocular setups)
                # But warning: geometric consistency might break if view index is wrong
                video_path = self._get_difix_video_path(0)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Restored video not found: {video_path}")

            # Load Video
            vr = VideoReader(video_path, ctx=cpu(0))
            
            # Clamp frame indices
            safe_frame_idxs = [min(i, len(vr)-1) for i in frame_idxs]
            
            # Get Batch
            frames = vr.get_batch(safe_frame_idxs).asnumpy() # (T, H, W, C)
            
            # Process Side-by-Side (Left: Input, Right: Restored)
            h, w_combined, c = frames.shape[1:]
            w_single = w_combined // 2
            
            # Extract Right Half
            restored_frames = frames[:, :, w_single:, :] 
            
            # Normalize
            restored_frames = restored_frames.astype(np.float32) / 255.0
            
            # Convert to Torch: (T, C, H, W)
            rgb_torch = torch.from_numpy(restored_frames).permute(0, 3, 1, 2).contiguous()
            
            output_dict[DataField.IMAGE_RGB] = rgb_torch
            
            if "__key__" not in output_dict:
                output_dict["__key__"] = f"difix_restored_view_{current_view}"

        return output_dict