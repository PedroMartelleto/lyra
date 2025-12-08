import os
import torch
import numpy as np
from decord import VideoReader, cpu
from depth_anything_3.api import DepthAnything3

# 1. Define paths
video_path = "assets/demo2/dynamic/diffusion_output_generated/0/rgb/815e2628-544a-50ea-b746-bef4b9e9b695.mp4"
output_dir = "output_results"

# 2. Load frames using decord
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
else:
    print(f"Loading video from {video_path}...")
    vr = VideoReader(video_path, ctx=cpu(0))
    # Get all frames as a numpy array (T, H, W, C)
    # You can sub-sample if the video is too long: range(0, len(vr), 2)
    frames_array = vr.get_batch(range(len(vr))).asnumpy()
    
    # Convert to list of numpy arrays for the API
    images = [frame for frame in frames_array]
    print(f"Loaded {len(images)} frames.")

    # 3. Initialize Model (Must use 'da3-giant' or 'da3nested-giant-large' for GS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3(model_name="da3nested-giant-large")
    model = model.to(device)

    # 4. Run Inference
    # We use ref_view_strategy="middle" which is optimized for video sequences
    print("Running inference...")
    prediction = model.inference(
        image=images,
        infer_gs=True,               # Required for Gaussian Splatting
        ref_view_strategy="middle",  # Best for temporal sequences
        
        # Automatically export to PLY
        export_dir=output_dir,
        export_format="gs_ply",
        
        # Optional: Align to input scale if you had GT poses (we don't here)
        align_to_input_ext_scale=False 
    )

    print(f"Processing complete. Results saved to {output_dir}")