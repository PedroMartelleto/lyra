# run_difix_3dgs.py

import os
import torch
import re
import argparse
from copy import deepcopy
from omegaconf import OmegaConf, ListConfig
from accelerate import Accelerator
from tqdm import tqdm
import einops
from src.models.utils.model import load_vae, encode_multi_view_video, encode_video, decode_multi_view_latents

# Import Lyra modules
from src.models.utils.misc import seed_everything, load_and_merge_configs, dtype_map
from src.models.utils.train import get_most_recent_checkpoint
from src.models.utils.model import load_vae, encode_multi_view_video, encode_latent_time_vae, encode_plucker_vae
from src.models.utils.render import get_plucker_embedding_and_rays, save_ply
from src.models.recon.model_latent_recon import LatentRecon
from src.models.data.provider import Provider
from src.models.data.registry import dataset_registry
from src.utils.visu import save_video, create_depth_visu

# Import our custom adapter
from src.models.data.difix_adapter import DifixAdapter

def main():
    # --- Configuration ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/demo/lyra_dynamic.yaml', help="Base config")
    parser.add_argument('--difix_output_dir', type=str, default='outputs/difix_2', help="Path to lets_difix.py output")
    parser.add_argument('--original_data_root', type=str, default='outputs/demo/lyra_dynamic_2/static_view_indices_fixed_5_0_1_2_3_4/lyra_dynamic_demo_generated/0', help="Path to original data (for poses/intrinsics)")
    parser.add_argument('--output_dir', type=str, default='outputs/difix_3dgs', help="Where to save 3DGS results")
    args = parser.parse_args()

    # 1. Load Inference Config
    config_inference = load_and_merge_configs(['configs/inference/default.yaml', args.config])

    # 2. Load Model Config
    if isinstance(config_inference.config_path, str):
        config_model = OmegaConf.load(config_inference.config_path)
    else:
        config_model = load_and_merge_configs(config_inference.config_path)

    # 3. Merge
    config = OmegaConf.merge(config_model, config_inference)
    
    # --- FIX 1: Force Single View Configuration ---
    # We are reconstructing a single video stream from difix.
    # Even if the model was trained with 6 views, at inference we provide 1 input view sequence.
    # We must align the config so Provider generates data for 1 view.
    
    # Overwrite fixed indices to just '0' (or the first one).
    config.static_view_indices_fixed = ['0'] 
    config.static_view_indices_sampling = 'fixed'
    config.num_input_multi_views = 1 

    # --- FIX 3: Disable Manual Target Index List ---
    config.target_index_manual = None
    config.set_manual_time_idx = True 

    # --- FIX 4: Disable Latent Loading ---
    config.load_latents = False

    # Setup Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    seed = config.get('seed', 42)
    seed_everything(seed)

    if accelerator.is_main_process:
        print(f"Running with seed: {seed}")
        print(f"Output Directory: {args.output_dir}")

    # --- 1. Load Models ---
    if accelerator.is_main_process:
        print(f"[{device}] Loading VAE ({config.vae_backbone})...")
    
    vae = load_vae(config.vae_backbone, config.vae_path)
    vae.to(device, dtype=weight_dtype)
    vae.eval()

    if accelerator.is_main_process:
        print(f"[{device}] Loading 3DGS Transformer...")
        
    ckpt_path = config.ckpt_path
    if ckpt_path is None:
        ckpt_name = get_most_recent_checkpoint(config.output_dir)
        if ckpt_name:
            ckpt_path = os.path.join(config.output_dir, ckpt_name, 'pytorch_model/mp_rank_00_model_states.pt')

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    if accelerator.is_main_process:
        print(f"Loading checkpoint from: {ckpt_path}")

    transformer = LatentRecon(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "module" in checkpoint:
        transformer.load_state_dict(checkpoint["module"])
    else:
        transformer.load_state_dict(checkpoint)
    
    transformer.to(device, dtype=weight_dtype)
    transformer.eval()

    # --- 2. Setup Data ---
    dataset_name = config.dataset_name
    if dataset_name not in dataset_registry:
        reg_entry = {
            'cls': dataset_registry['lyra_dynamic']['cls'], 
            'kwargs': dataset_registry['lyra_dynamic']['kwargs'].copy()
        }
    else:
        reg_entry = dataset_registry[dataset_name].copy()
        reg_entry['kwargs'] = dataset_registry[dataset_name]['kwargs'].copy()

    # Override path
    reg_entry['kwargs']['root_path'] = args.original_data_root

    # Remove Target View Indices to prevent lookup errors
    keys_to_remove = ["start_view_target_idx", "end_view_target_idx"]
    for k in keys_to_remove:
        if k in reg_entry['kwargs']:
            reg_entry['kwargs'].pop(k)

    # Prepare kwargs for Dataset Init
    dataset_kwargs = deepcopy(reg_entry['kwargs'])
    if 'default' in dataset_registry:
        for key in dataset_registry['default'].keys():
            if key in dataset_kwargs:
                dataset_kwargs.pop(key)

    if accelerator.is_main_process:
        print(f"[{device}] Initializing Base Dataset at {args.original_data_root}...")
    
    original_dataset = reg_entry['cls'](**dataset_kwargs)

    if accelerator.is_main_process:
        print(f"[{device}] Initializing Difix Adapter pointing to {args.difix_output_dir}...")
    
    dataset = DifixAdapter(original_dataset, args.difix_output_dir)

    # Register custom entry
    dataset_registry['difix_custom'] = reg_entry
        
    # Instantiate Provider and swap dataset
    provider = Provider('difix_custom', config, training=False)
    provider.dataset = dataset 
    provider.dataset.sample_list = dataset.original_dataset.sample_list

    dataloader = torch.utils.data.DataLoader(
        provider,
        batch_size=1, 
        shuffle=False,
        num_workers=4
    )

    # --- 3. Inference Loop ---
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[{device}] Starting Inference on {len(dataloader)} samples...")
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                if k not in ['intrinsics_input', 'c2ws_input', 'cam_view', 'intrinsics', 'file_name', 'target_index', 'flip_flag']:
                    batch[k] = batch[k].to(weight_dtype)

        with torch.no_grad():
            # 3.1 Plucker Embeddings
            if config.get('compute_plucker_cuda', True):
                batch['plucker_embedding'], batch['rays_os'], batch['rays_ds'] = get_plucker_embedding_and_rays(
                    batch['intrinsics_input'],
                    batch['c2ws_input'],
                    config.img_size,
                    config.patch_size_out_factor,
                    batch['flip_flag'],
                    get_batch_index=False,
                    dtype=dtype_map[config.compute_plucker_dtype] if config.get('compute_plucker_dtype') else weight_dtype,
                    out_dtype=weight_dtype
                )

            # 3.2 Encode Video to Latents
            video_input = batch['images_input_vae']
            num_input_multi_views = batch.get('num_input_multi_views', 1)
            if isinstance(num_input_multi_views, torch.Tensor):
                num_input_multi_views = num_input_multi_views.item()
            num_input_multi_views = int(num_input_multi_views)

            # Safe encode check: if video shape doesn't match views, force match
            b, frames, c, h, w = video_input.shape
            if num_input_multi_views > 1 and frames % num_input_multi_views != 0:
                print(f"WARNING: Frame count {frames} not divisible by num_input_multi_views {num_input_multi_views}. Forcing num_input_multi_views=1.")
                num_input_multi_views = 1

            latents = encode_multi_view_video(vae, video_input, num_input_multi_views, config.vae_backbone)
            batch['images_input_embed'] = latents

            # 3.3 Encode Time and Plucker with VAE
            if config.get('time_embedding_vae', False):
                 batch = encode_latent_time_vae(batch, lambda x: encode_video(vae, x, config.vae_backbone), config.img_size)
            if config.get('plucker_embedding_vae', False):
                batch = encode_plucker_vae(batch, lambda x: encode_multi_view_video(vae, x, num_input_multi_views, config.vae_backbone))

            # 3.4 Run 3DGS Transformer
            model_output = transformer(batch)

            # 3.5 Render/Process Outputs
            if accelerator.is_main_process:
                pred_images = model_output['images_pred']
                pred_depths = create_depth_visu(model_output['depths_pred'])
                
                save_name = f"difix_recon_scene_{i}"
                fps = config.get('out_fps', 24)
                
                save_video(pred_images.detach().cpu(), args.output_dir, name=save_name, fps=fps)
                save_video(pred_depths.detach().cpu(), args.output_dir, name=f"{save_name}_depth", fps=fps)

                if config.get('save_gaussians', False):
                    gauss_path = os.path.join(args.output_dir, f"{save_name}.ply")
                    save_ply(model_output['gaussians'].detach().cpu(), gauss_path)

                print(f"Saved {save_name} to {args.output_dir}")

    if accelerator.is_main_process:
        print("Inference Complete.")

if __name__ == "__main__":
    main()