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

import einops


def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    """
    Convert 4-dimensional quaternions to 3x3 rotation matrices.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """

    # Order changed to match scipy format: (i, j, k, r)
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    # Construct rotation matrix elements using quaternion algebra
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),  # R[0,0]
            two_s * (i * j - k * r),  # R[0,1]
            two_s * (i * k + j * r),  # R[0,2]
            two_s * (i * j + k * r),  # R[1,0]
            1 - two_s * (i * i + k * k),  # R[1,1]
            two_s * (j * k - i * r),  # R[1,2]
            two_s * (i * k - j * r),  # R[2,0]
            two_s * (j * k + i * r),  # R[2,1]
            1 - two_s * (i * i + j * j),  # R[2,2]
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def pose_from_quaternion(pose):
    """
    Convert quaternion-based pose representation to 4x4 homogeneous transformation matrices.
    """
    # Convert numpy array to torch tensor if needed
    if type(pose) == np.ndarray:
        pose = torch.from_numpy(pose)
    # Add batch dimension if input is 1D
    if len(pose.shape) == 1:
        pose = pose[None]

    # Extract translation and quaternion components
    quat_t = pose[..., :3]  # Translation components [tx, ty, tz]
    quat_r = pose[..., 3:]  # Quaternion components [qi, qj, qk, qr]

    # Initialize world-to-camera transformation matrix
    w2c_matrix = torch.eye(4, device=pose.device).repeat(*pose.shape[:-1], 1, 1)
    w2c_matrix[..., :3, :3] = quaternion_to_matrix(quat_r)  # Set rotation
    w2c_matrix[..., :3, 3] = quat_t  # Set translation
    return w2c_matrix


class SpatialVid(BaseDataset):
    """
    A fully compliant and resilient dataset loader for SpatialVID.

    This version treats the length of the 'poses.npy' array as the source of
    truth for the number of frames, making it robust to inconsistencies in the
    metadata CSV file. It also handles downsampling of higher-framerate RGB
    videos to match the framerate of poses and depth maps.
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

    def get_video_info(self, video_idx: int) -> dict:
        record = self.metadata.iloc[video_idx]
        video_path = self.root_path / record['video path']
        return {
            'video_path': str(video_path),
            'id': record['id'],
            'group_id': record['group id'],
        }

    def read_poses_for_video(self, video_idx: int) -> torch.Tensor:
        record = self.metadata.iloc[video_idx]
        poses_path = self.root_path / record['annotation path'] / "poses.npy"
        all_poses = np.load(poses_path)
        all_poses = pose_from_quaternion(all_poses)
        return all_poses.float()

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
            self._frame_count_cache[video_idx] = 0
            return 0
            
        try:
            with open(poses_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, version)
                frame_count = shape[0]
                self._frame_count_cache[video_idx] = frame_count
                return frame_count
        except Exception:
            all_poses = np.load(poses_path)
            frame_count = all_poses.shape[0]
            self._frame_count_cache[video_idx] = frame_count
            return frame_count

    def _read_data(self, video_idx: int, frame_idxs: List[int], view_idxs: List[int], data_fields: List[DataField]) -> dict[DataField, Any]:
        record = self.metadata.iloc[video_idx]
        output_dict = {"__key__": record['id']}

        if not frame_idxs:
            return output_dict

        max_requested_idx = max(frame_idxs)
        true_frame_count = self.num_frames(video_idx)
        if max_requested_idx >= true_frame_count:
            raise IndexError(f"Requested frame index {max_requested_idx} is out of bounds for video {video_idx} which has {true_frame_count} frames.")

        for data_field in data_fields:
            if data_field == DataField.IMAGE_RGB:
                video_path = self.root_path / record['video path']
                vr = VideoReader(str(video_path), num_threads=4)
                
                num_rgb_frames = len(vr)
                num_pose_frames = self.num_frames(video_idx)
                
                if num_pose_frames == 0:
                    output_dict[data_field] = torch.empty((0, 3, 0, 0))
                    continue

                rgb_indices_map = np.linspace(0, num_rgb_frames - 1, num=num_pose_frames, dtype=int)
                selected_rgb_indices = rgb_indices_map[frame_idxs]
                
                frames = vr.get_batch(selected_rgb_indices).asnumpy()
                output_dict[data_field] = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0

            elif data_field == DataField.CAMERA_C2W_TRANSFORM:
                poses_path = self.root_path / record['annotation path'] / "poses.npy"
                all_poses = np.load(poses_path)
                selected_poses = all_poses[frame_idxs]
                output_dict[data_field] = pose_from_quaternion(selected_poses).float()

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
                    try:
                        with depth_zip.open(frame_name, "r") as f:
                            in_memory_file = io.BytesIO(f.read())
                            exr_file = OpenEXR.InputFile(in_memory_file)
                            depths.append(read_exr_depth_to_numpy(exr_file))
                    except KeyError:
                        if depths:
                            depths.append(np.zeros_like(depths[0]))
                
                if depths:
                    output_dict[data_field] = torch.from_numpy(np.stack(depths, axis=0).astype(np.float32))
                else:
                    output_dict[data_field] = torch.empty((len(frame_idxs), 0, 0))

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