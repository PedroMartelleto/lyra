# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law of a simple 3D object.
# limitations under the License.

import torch
import os
from gsplat.rendering import rasterization
from torchvision.utils import save_image

def print_tensor_info(name, tensor):
    """Helper function to print detailed information about a tensor."""
    print(f"--- Tensor: {name} ---")
    if tensor is None:
        print("  None")
        return
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  Has Inf: {torch.isinf(tensor).any().item()}")
    if torch.is_tensor(tensor) and tensor.numel() > 0:
        print(f"  Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Mean: {tensor.mean().item():.4f}")
    print("-" * (len(name) + 14))

def run_minimal_renderer_test():
    """
    Experiment 0 (Final): Minimal gsplat test, completely removing the 'backgrounds' argument.
    This isolates the issue to the background handling of the gsplat library.
    """
    print("\n--- Running Experiment 0 (Final): Minimal Renderer Test without Backgrounds ---")
    output_dir = "sanity_checks"
    os.makedirs(output_dir, exist_ok=True)
    
    H, W = 256, 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("WARNING: Running on CPU. gsplat is CUDA-only. This will fail.")
        return

    # --- 1. Define Gaussian Properties ---
    means = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32, device=device)
    scales = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    opacities = torch.tensor([0.99], dtype=torch.float32, device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    
    # --- 2. Define Camera Properties ---
    world_to_cam = torch.eye(4, dtype=torch.float32, device=device)
    viewmat = world_to_cam.unsqueeze(0)
    
    focal = W * 1.5
    K = torch.tensor([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    # --- 3. Print All Inputs for Debugging ---
    print("--- Inputs to gsplat.rasterization ---")
    print_tensor_info("Means (Position)", means)
    print("Backgrounds: None (will use gsplat default)")
    print(f"Image Size (H, W): ({H}, {W})")
    print("--------------------------------------\n")

    # --- 4. Call Rasterizer ---
    try:
        # **MODIFICATION**: The `backgrounds` argument is completely removed from this call.
        image, alpha, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            render_mode='RGB',
        )
        print("rasterization call successful.")
        
        # --- 5. Save Output ---
        output_image = image.squeeze(0).permute(2, 0, 1) # (C, H, W)
        output_path = os.path.join(output_dir, "exp0_minimal_renderer_output.png")
        save_image(output_image, output_path)

        print("\n--- ✅ Experiment 0 Finished ---")
        print(f"Check the output image: {os.path.abspath(output_path)}")
        print("Expected result: An image of a red sphere on a black background.")

    except Exception as e:
        print("\n--- ❌ Experiment 0 FAILED ---")
        print("The call to gsplat.rasterization raised an error.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_minimal_renderer_test()