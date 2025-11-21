import torch
from omegaconf import OmegaConf

from src.models.recon.model_latent_recon import LatentRecon
from src.models.utils.misc import load_and_merge_configs

def create_dummy_batch(config, batch_size, device, dtype):
    """Creates a dummy batch dictionary with tensors of the correct shape and type."""
    
    # Input view and frame counts
    num_input_multi_views = config.num_input_multi_views
    num_input_frames = config.num_input_views
    num_total_input_frames = num_input_multi_views * num_input_frames

    # Latent dimensions
    img_h, img_w = config.img_size
    latent_h = img_h // config.latent_spat_compression
    latent_w = img_w // config.latent_spat_compression
    
    # Target view count (for rendering output)
    num_target_frames = config.num_views - num_input_frames

    # Rays and plucker embeddings (spatially downsampled)
    rays_h = img_h // config.patch_size_out_factor[1]
    rays_w = img_w // config.patch_size_out_factor[2]
    
    batch = {
        'images_input_embed': torch.randn(
            batch_size, num_total_input_frames, config.num_latent_c, latent_h, latent_w, 
            device=device, dtype=dtype
        ),
        'plucker_embedding': torch.randn(
            batch_size, num_total_input_frames, 6, img_h, img_w, 
            device=device, dtype=dtype
        ),
        'rays_os': torch.randn(
            batch_size, num_total_input_frames, 3, rays_h, rays_w, 
            device=device, dtype=dtype
        ),
        'rays_ds': torch.randn(
            batch_size, num_total_input_frames, 3, rays_h, rays_w, 
            device=device, dtype=dtype
        ),
        'time_embeddings': torch.randn(
            batch_size, num_total_input_frames, config.time_embedding_dim, 
            device=device, dtype=dtype
        ) if config.time_embedding else False,
        'cam_view': torch.eye(4, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(
            batch_size, num_target_frames, 1, 1
        ),
        'intrinsics': torch.tensor(
            [img_w, img_h, img_w/2, img_h/2], device=device, dtype=dtype
        ).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_target_frames, 1),
        'num_input_multi_views': num_input_multi_views,
    }
    return batch

def check_model_forward_pass():
    """
    Initializes the LatentRecon model, creates a dummy input batch,
    and performs a single forward pass to check for structural errors.
    """
    print("--- Running Model Forward Pass Check ---")

    # --- 1. Load Configuration ---
    print("Loading model configuration...")
    # Use a lightweight config for the test
    config_paths = [
        'configs/training/default.yaml',
        'configs/training/3dgs_res_176_320_views_17.yaml'
    ]
    config = load_and_merge_configs(config_paths)
    
    # Use a small batch size for the test
    config.batch_size = 1
    # Disable deferred backpropagation for a simpler forward pass check
    config.deferred_bp = False
    
    # --- 2. Setup Model and Data ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")

    print("Initializing LatentRecon model...")
    model = LatentRecon(config).to(device).eval()
    
    print("Creating dummy input batch...")
    dummy_batch = create_dummy_batch(config, config.batch_size, device, dtype)
    
    # --- 3. Perform Forward Pass ---
    print("Executing forward pass...")
    try:
        with torch.no_grad():
            output = model(dummy_batch)
        print("Forward pass completed successfully.")
    except Exception as e:
        print(f"\n--- ❌ FORWARD PASS FAILED ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Verify Outputs ---
    print("Verifying output shapes and types...")
    
    B = config.batch_size
    V_out = config.num_views - config.num_input_views
    H, W = config.img_size
    num_gaussians = (
        (config.num_input_multi_views * config.num_input_views) * 
        (H // config.patch_size_out_factor[1]) * 
        (W // config.patch_size_out_factor[2])
    )

    expected_shapes = {
        'images_pred': (B, V_out, 3, H, W),
        'depths_pred': (B, V_out, 1, H, W),
        'alphas_pred': (B, V_out, 1, H, W),
        'gaussians': (B, num_gaussians, 14),
    }

    all_checks_passed = True
    for key, shape in expected_shapes.items():
        if key not in output:
            print(f"❌ FAILED: Output dictionary missing key '{key}'")
            all_checks_passed = False
            continue
            
        tensor = output[key]
        if tensor.shape != shape:
            print(f"❌ FAILED: Shape mismatch for '{key}'. Expected {shape}, but got {tensor.shape}.")
            all_checks_passed = False
        else:
            print(f"✅ PASSED: Shape for '{key}' is correct: {tensor.shape}")

        if tensor.dtype != dtype:
            print(f"❌ FAILED: Dtype mismatch for '{key}'. Expected {dtype}, but got {tensor.dtype}.")
            all_checks_passed = False
        else:
            print(f"✅ PASSED: Dtype for '{key}' is correct: {tensor.dtype}")
            
    if all_checks_passed:
        print("\n--- ✅ All checks passed. Model forward pass is structurally correct! ---")
    else:
        print("\n--- ❌ Some checks failed. Please review the errors above. ---")


if __name__ == '__main__':
    check_model_forward_pass()