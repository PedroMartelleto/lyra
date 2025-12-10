import sys

local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Depth-Anything-3", "src"))

# Insert at index 0 to ensure it takes precedence over pip installed packages
if local_path not in sys.path:
    sys.path.insert(0, local_path)


import torch
import os
import numpy as np
from depth_anything_3.api import DepthAnything3
from decord import VideoReader, cpu
from omegaconf import OmegaConf

from src.models.data import get_multi_dataloader
from src.models.utils.misc import load_and_merge_configs
from src.models.data.registry import dataset_registry

def main():
    # -------------------------------------------------------------------------
    # 1. Load Configuration
    # -------------------------------------------------------------------------
    demo_config_path = "configs/demo/lyra_dynamic.yaml"
    print(f"--- Loading config from {demo_config_path} ---")
    
    demo_config = OmegaConf.load(demo_config_path)

    if 'config_path' in demo_config:
        base_config = load_and_merge_configs(demo_config.config_path)
        config = OmegaConf.merge(base_config, demo_config)
    else:
        config = demo_config

    # -------------------------------------------------------------------------
    # 2. Adjust Config & Patch Registry
    # -------------------------------------------------------------------------
    config.batch_size = 1
    config.num_workers = 0 
    config.subsample_data_train_val = False
    
    # Disable GT depth loading
    config.use_depth = False
    
    if config.dataset_name is not None:
        config.data_mode = [[config.dataset_name, 1]]

    # Patch registry for single view
    dataset_name = config.dataset_name
    if dataset_name in dataset_registry:
        print(f"--- Patching registry for {dataset_name} to force SINGLE VIEW (View 0) ---")
        registry_entry = dataset_registry[dataset_name]['kwargs']
        if 'sampling_buckets' in registry_entry:
            registry_entry['sampling_buckets'] = [['0']]
        registry_entry['start_view_idx'] = 0
        registry_entry['end_view_idx'] = 0
        registry_entry['start_view_target_idx'] = None
        registry_entry['end_view_target_idx'] = None
        config.num_input_multi_views = 1

    # -------------------------------------------------------------------------
    # 3. Extract Trajectory AND Intrinsics from Dataset
    # -------------------------------------------------------------------------
    print("Initializing Data Provider to retrieve Lyra trajectory...")
    _, test_loader = get_multi_dataloader(config)
    
    try:
        batch_test = next(iter(test_loader))
    except StopIteration:
        raise ValueError("Test dataloader is empty! Could not load trajectory.")
    except Exception as e:
        print(f"\nCRITICAL ERROR during data loading: {e}")
        return

    # --- Extrinsics ---
    c2ws_input = batch_test['c2ws_input']
    if c2ws_input.ndim == 4:
        c2ws_input = c2ws_input[0] # [T, 4, 4]

    w2cs = torch.inverse(c2ws_input).float()
    extrinsics_lyra = w2cs.cpu().numpy().astype(np.float32) # [T, 4, 4]

    # --- Intrinsics ---
    # Lyra loader provides intrinsics in [fx, fy, cx, cy] format
    intrinsics_vec = batch_test['intrinsics_input']
    if intrinsics_vec.ndim == 3:
        intrinsics_vec = intrinsics_vec[0] # [T, 4]
    
    # Convert [T, 4] -> [T, 3, 3] matrix
    T_len = intrinsics_vec.shape[0]
    intrinsics_lyra = np.zeros((T_len, 3, 3), dtype=np.float32)
    
    # Fill diagonal and offsets
    intrinsics_vec_np = intrinsics_vec.cpu().numpy()
    intrinsics_lyra[:, 0, 0] = intrinsics_vec_np[:, 0] # fx
    intrinsics_lyra[:, 1, 1] = intrinsics_vec_np[:, 1] # fy
    intrinsics_lyra[:, 0, 2] = intrinsics_vec_np[:, 2] # cx
    intrinsics_lyra[:, 1, 2] = intrinsics_vec_np[:, 3] # cy
    intrinsics_lyra[:, 2, 2] = 1.0

    print(f"Loaded trajectory (c2ws): {c2ws_input.shape}")
    print(f"Loaded intrinsics: {intrinsics_lyra.shape}")

    # -------------------------------------------------------------------------
    # 4. Load GEN3C Video
    # -------------------------------------------------------------------------
    input_video = "assets/demo/dynamic/diffusion_output_generated/0/rgb/781d986f-791d-52f9-801f-34d4a616e072.mp4"
    
    if not os.path.exists(input_video):
        # Fallback check
        input_video_demo2 = input_video.replace("demo", "demo2")
        if os.path.exists(input_video_demo2):
            input_video = input_video_demo2
        else:
            print(f"Warning: Video file not found at {input_video}")
            return

    print(f"Loading video: {input_video}")
    vr = VideoReader(input_video, ctx=cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy() # [T, H, W, C]
    images_list = [f for f in frames]
    
    # -------------------------------------------------------------------------
    # 5. Align Video and Trajectory Lengths
    # -------------------------------------------------------------------------
    n_frames_vid = len(images_list)
    n_frames_traj = len(extrinsics_lyra)
    min_len = min(n_frames_vid, n_frames_traj)
    
    final_images = images_list[:min_len]
    final_extrinsics = extrinsics_lyra[:min_len]
    final_intrinsics = intrinsics_lyra[:min_len] # Also align intrinsics
    
    print(f"Aligning frames: Video({n_frames_vid}) vs Trajectory({n_frames_traj}) -> Using {min_len} frames.")

    # -------------------------------------------------------------------------
    # 6. Run DepthAnything3 Inference
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading DepthAnything3 model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)

    print("Running inference with Lyra trajectory and intrinsics...")
    prediction = model.inference(
        final_images,
        extrinsics=final_extrinsics, 
        intrinsics=final_intrinsics,
        infer_gs=True,
        export_dir="output_render_da3",
        process_res=1024,
        num_max_points=10_000_000,
        export_format="gs_video",
        # trj_mode="original",
    )

    if hasattr(prediction, 'processed_images'):
        print(f"Processed images shape: {prediction.processed_images.shape}")
    
    print("Rendering complete. Results saved to 'output_render_da3'.")

if __name__ == "__main__":
    main()