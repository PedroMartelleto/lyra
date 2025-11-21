# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import os
import argparse
from gsplat.rendering import rasterization
from torchvision.utils import save_image

def run_look_at_origin_test():
    """
    Experiment A: An unambiguous 'look at' setup.
    Places a camera at (0, 0, 5) and looks at a sphere at the origin (0, 0, 0).
    This test is designed to be robust to coordinate system conventions.
    """
    print("\n--- Running Experiment A: Look At Origin Test ---")
    output_dir, H, W, device = "sanity_checks", 256, 256, "cuda"
    os.makedirs(output_dir, exist_ok=True)

    # --- Gaussian at the world origin ---
    means = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    scales = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    opacities = torch.tensor([0.99], dtype=torch.float32, device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    # --- Camera placed at (0, 0, 5) looking at the origin ---
    # This creates the world-to-camera matrix (view matrix).
    # We build the camera-to-world matrix first, as it's more intuitive.
    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[2, 3] = 5.0 # Move camera to z=5
    viewmat = torch.inverse(c2w).unsqueeze(0) # Invert to get world-to-camera, add batch dim

    focal = W * 1.5
    K = torch.tensor([
        [focal, 0, W/2], [0, focal, H/2], [0, 0, 1]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    try:
        image, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmat, Ks=K, width=W, height=H, render_mode='RGB'
        )
        save_image(image.squeeze(0).permute(2, 0, 1), os.path.join(output_dir, "expA_look_at_origin.png"))
        print("--- ✅ Experiment A Finished ---")
        print(f"Check the output image: {os.path.abspath(os.path.join(output_dir, 'expA_look_at_origin.png'))}")
    except Exception as e:
        print(f"--- ❌ Experiment A FAILED: {e} ---")

def run_opencv_convention_test():
    """
    Experiment B: Test the OpenCV/COLMAP camera convention.
    Applies a 180-degree rotation around the X-axis to flip the Y and Z axes.
    """
    print("\n--- Running Experiment B: OpenCV Convention Test ---")
    output_dir, H, W, device = "sanity_checks", 256, 256, "cuda"
    os.makedirs(output_dir, exist_ok=True)

    # --- Gaussian at (0, 0, -5), in front of camera at origin ---
    means = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float32, device=device)
    scales = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    opacities = torch.tensor([0.99], dtype=torch.float32, device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    # --- Camera at origin, but rotated to OpenCV convention ---
    # This matrix rotates the world by 180 degrees around the X-axis before applying the camera view.
    # It's equivalent to having a camera where +Y is down and +Z is forward.
    opencv_to_opengl = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    viewmat = opencv_to_opengl.unsqueeze(0) # Batch dimension

    focal = W * 1.5
    K = torch.tensor([
        [focal, 0, W/2], [0, focal, H/2], [0, 0, 1]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    try:
        image, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmat, Ks=K, width=W, height=H, render_mode='RGB'
        )
        save_image(image.squeeze(0).permute(2, 0, 1), os.path.join(output_dir, "expB_opencv_convention.png"))
        print("--- ✅ Experiment B Finished ---")
        print(f"Check the output image: {os.path.abspath(os.path.join(output_dir, 'expB_opencv_convention.png'))}")
    except Exception as e:
        print(f"--- ❌ Experiment B FAILED: {e} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run convention tests for the gsplat renderer.")
    parser.add_argument(
        "--test",
        type=str,
        choices=['A', 'B'],
        required=True,
        help="The test to run: 'A' for look-at-origin, 'B' for OpenCV convention."
    )
    args = parser.parse_args()

    if args.test == 'A':
        run_look_at_origin_test()
    elif args.test == 'B':
        run_opencv_convention_test()