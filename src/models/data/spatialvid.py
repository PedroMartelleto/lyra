# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import zipfile
from pathlib import Path
from typing import List, Any
import io

import numpy as np
import torch
import pandas as pd
from lru import LRU
from decord import VideoReader
import OpenEXR
import Imath

from src.models.data.base import BaseDataset
from src.models.data.datafield import DataField
from src.models.utils.data import read_exr_depth_to_numpy


class SpatialVid(BaseDataset):
    """
    FINAL VERSION: A fully compliant and resilient dataset loader for SpatialVID.

    This version treats the length of the 'poses.npy' array as the source of
    truth for the number of frames, making it robust to inconsistencies in the
    metadata CSV file.
    """
    MAX_ZIP_DESCRIPTORS = 10

    def __init__(self, root_path: str, metadata_csv: str):
        super().__init__()
        self.root_path = Path(root_path)
        metadata_path = self.root_path / metadata_csv
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found at: {metadata_path}")
        self.metadata = pd.read_csv(metadata_path)
        self.zip_descriptors = LRU(self.MAX_ZIP_DESCRIPTORS, callback=lambda _, handle: handle.close())
        
        # Cache for the true number of frames to avoid re-reading .npy files
        self._frame_count_cache = {}

    def available_data_fields(self) -> List[DataField]:
        return [
            DataField.IMAGE_RGB,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.CAMERA_INTRINSICS,
            DataField.METRIC_DEPTH,
        ]

    def __len__(self):
        return self.num_videos()

    def num_videos(self) -> int:
        return len(self.metadata)

    def num_views(self, video_idx: int) -> int:
        return 1

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        """
        Returns the true number of frames by checking the length of the poses.npy file.
        This is the source of truth, not the CSV.
        """
        if video_idx in self._frame_count_cache:
            return self._frame_count_cache[video_idx]

        record = self.metadata.iloc[video_idx]
        poses_path = self.root_path / record['annotation path'] / "poses.npy"
        
        if not poses_path.exists():
            print(f"Warning: poses.npy not found for index {video_idx}. Returning 0 frames.")
            self._frame_count_cache[video_idx] = 0
            return 0
            
        try:
            with open(poses_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, version)
                frame_count = shape[0]
                self._frame_count_cache[video_idx] = frame_count
                return frame_count
        except Exception as e:
            print(f"Warning: Could not read header of {poses_path}. Loading full array. Error: {e}")
            # Fallback to loading the whole array if header reading fails
            all_poses = np.load(poses_path)
            frame_count = all_poses.shape[0]
            self._frame_count_cache[video_idx] = frame_count
            return frame_count


    def _read_data(self, video_idx: int, frame_idxs: List[int], view_idxs: List[int], data_fields: List[DataField]) -> dict[DataField, Any]:
        record = self.metadata.iloc[video_idx]
        output_dict = {"__key__": record['id']}

        # We can add a final safety check here
        max_requested_idx = max(frame_idxs)
        true_frame_count = self.num_frames(video_idx)
        if max_requested_idx >= true_frame_count:
            raise IndexError(f"FATAL in _read_data: Requested frame index {max_requested_idx} is out of bounds for video {video_idx} which has {true_frame_count} frames.")

        for data_field in data_fields:
            if data_field == DataField.IMAGE_RGB:
                video_path = self.root_path / record['video path']
                vr = VideoReader(str(video_path), num_threads=4)
                frames = vr.get_batch(frame_idxs).asnumpy()
                output_dict[data_field] = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            elif data_field == DataField.CAMERA_C2W_TRANSFORM:
                poses_path = self.root_path / record['annotation path'] / "poses.npy"
                all_poses = np.load(poses_path)
                output_dict[data_field] = torch.from_numpy(all_poses[frame_idxs]).float()
            elif data_field == DataField.CAMERA_INTRINSICS:
                intrinsics_path = self.root_path / record['annotation path'] / "intrinsics.npy"
                all_intrinsics = np.load(intrinsics_path)
                if all_intrinsics.ndim == 1:
                    output_dict[data_field] = torch.from_numpy(np.tile(all_intrinsics, (len(frame_idxs), 1))).float()
                else:
                    output_dict[data_field] = torch.from_numpy(all_intrinsics[frame_idxs]).float()
            elif data_field == DataField.METRIC_DEPTH:
                depth_zip = self._get_zip_handle(video_idx)
                depths = []
                for frame_idx in frame_idxs:
                    frame_name = f"{frame_idx:05d}.exr"
                    with depth_zip.open(frame_name, "r") as f:
                        in_memory_file = io.BytesIO(f.read())
                        exr_file = OpenEXR.InputFile(in_memory_file)
                        depths.append(read_exr_depth_to_numpy(exr_file))
                output_dict[data_field] = torch.from_numpy(np.stack(depths, axis=0).astype(np.float32))
            else:
                raise NotImplementedError(f"Data field '{data_field}' not supported.")
        return output_dict

    def _get_zip_handle(self, video_idx: int) -> zipfile.ZipFile:
        if video_idx in self.zip_descriptors:
            return self.zip_descriptors[video_idx]
        record = self.metadata.iloc[video_idx]
        depth_zip_path = self.root_path / "depths" / f"group_{str(record['group id']).zfill(4)}" / f"{record['id']}.zip"
        if not depth_zip_path.exists():
            raise FileNotFoundError(f"Depth zip not found: {depth_zip_path}")
        zip_handle = zipfile.ZipFile(depth_zip_path, "r")
        self.zip_descriptors[video_idx] = zip_handle
        return zip_handle