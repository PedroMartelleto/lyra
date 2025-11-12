import torch
import os
import numpy as np
import random
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd
from decord import VideoReader

from src.models.data.spatialvid_provider import SpatialVidProvider
from src.models.data.spatialvid import SpatialVid

def analyze_dataset_statistics(num_videos_to_check=15):
    """
    Analyzes a random subset of videos from the SpatialVid dataset to gather
    key statistics about frame counts and effective frame rates.
    """
    print(f"--- Analyzing Dataset Statistics for {num_videos_to_check} Random Videos ---")

    # Minimal config to initialize the dataset. We don't need a dataloader.
    # We will be accessing the dataset object directly.
    config_dict = {
        'data_mode': [['spatialvid', 1]],
        'img_size': [256, 256], # Placeholder, not used for this analysis
        'num_views': 2, # Minimal load
        'num_input_views': 1,
        'use_depth': True,
        'subsample_data_train_val': False,
        'load_latents': False,
        # Add other necessary keys with default values to ensure provider initializes
        'is_static': True,
        'num_input_multi_views': 1,
        'sample_num_input_multi_views': False,
        'static_view_indices_sampling': 'fixed',
        'static_view_indices_fixed': ['0'],
        'use_plucker': False,
        'plucker_embedding_vae': False,
        'compute_plucker_cuda': False,
        'relative_translation_scale': True,
        'time_embedding': False,
    }
    opt = OmegaConf.create(config_dict)

    # We instantiate the underlying SpatialVid dataset directly, not the Provider
    dataset = SpatialVid(
        root_path="/iopsstor/scratch/cscs/pmartell/SpatialVid/HQ",  # Assuming 'spatialvid' maps to the root path
        metadata_csv=dataset_registry['spatialvid']['kwargs']['metadata_csv']
    )
    
    # --- 1. Select Random Videos ---
    num_total_videos = len(dataset)
    if num_total_videos < num_videos_to_check:
        print(f"Warning: Requested to check {num_videos_to_check} videos, but dataset only has {num_total_videos}. Checking all videos.")
        indices_to_check = list(range(num_total_videos))
    else:
        indices_to_check = random.sample(range(num_total_videos), num_videos_to_check)
    
    results = []
    
    print(f"Processing {len(indices_to_check)} videos...")
    for video_idx in tqdm(indices_to_check, desc="Analyzing Videos"):
        # --- 2. Get Total Frame Counts ---
        video_info = dataset.get_video_info(video_idx)
        video_path = video_info['video_path']
        
        # Original video FPS and frame count
        video_reader = VideoReader(video_path, num_threads=1)
        original_fps = video_reader.get_avg_fps()
        rgb_frame_count = len(video_reader)
        del video_reader # Close the reader
        
        depth_frame_count = dataset.num_frames(video_idx) # Assuming depth count matches rgb
        
        # Load all poses for this video
        all_poses = dataset.read_poses_for_video(video_idx)
        pose_frame_count = len(all_poses)
        
        # --- 3. Estimate Effective FPS by Checking for Duplicates ---
        # Load all depth frames to check for repeats
        all_depths = dataset.read(video_idx, data_fields=['metric_depth'])['metric_depth']

        # Calculate effective FPS
        # (Unique Frames / Total Frames) * Original FPS
        if rgb_frame_count > 0:
            effective_depth_fps = 1/(rgb_frame_count / depth_frame_count) * original_fps
            effective_pose_fps = 1/(rgb_frame_count / pose_frame_count) * original_fps
        else:
            effective_depth_fps = 0
            effective_pose_fps = 0

        results.append({
            "Video Index": video_idx,
            "File Name": os.path.basename(video_path),
            "Original FPS": f"{original_fps:.2f}",
            "RGB Frames": rgb_frame_count,
            "Depth Frames": depth_frame_count,
            "Pose Frames": pose_frame_count,
            "Est. Depth FPS": f"{effective_depth_fps:.2f}",
            "Est. Pose FPS": f"{effective_pose_fps:.2f}",
        })

    # --- 4. Print Summary Table ---
    if results:
        df = pd.DataFrame(results)
        df.set_index("Video Index", inplace=True)
        print("\n--- Dataset Statistics Summary ---")
        print(df.to_string())
    else:
        print("\nNo videos were processed.")

if __name__ == "__main__":
    # To run this script, you need to add the dataset registry definition
    # or ensure it's imported correctly.
    
    # Minimal registry definition for standalone execution
    dataset_registry = {
        'spatialvid': {
            'kwargs': {
                "root_path": "/iopsstor/scratch/cscs/pmartell/SpatialVid/HQ",
                "metadata_csv": "data/train/SpatialVID_HQ_metadata.csv",
            }
        }
    }
    
    # You might need to adjust the paths below to match your project structure
    # This assumes the script is in the root of your project.
    from src.models.data.spatialvid import SpatialVid
    
    analyze_dataset_statistics()