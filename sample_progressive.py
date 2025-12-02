# sample_progressive.py

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Dict, List
from tqdm import tqdm
import os
import re
import torch
import torch.nn.functional as F
import numpy as np
from accelerate import PartialState
import einops
from omegaconf import OmegaConf
from accelerate.logging import get_logger

from src.models.recon.model_latent_recon import LatentRecon
from src.utils.visu import create_depth_visu, generate_wave_video, save_video
from src.models.data import get_multi_dataloader
from src.models.utils.model import encode_latent_time_vae, encode_plucker_vae
from src.models.utils.render import get_plucker_embedding_and_rays, save_ply, save_ply_orig
from src.models.utils.model import load_vae, encode_multi_view_video, encode_video, decode_multi_view_latents
from src.models.utils.data import write_dict_to_json
from src.models.utils.misc import dtype_map, seed_everything, load_and_merge_configs
from src.models.utils.train import get_most_recent_checkpoint

logger = get_logger(__name__, log_level="INFO")

def load_model(ckpt_path, config, weight_dtype):
    # Load model
    distributed_state = PartialState()
    device = distributed_state.device
    vae = load_vae(config.vae_backbone, config.vae_path)
    transformer = LatentRecon(
        config
    )

    # Load ckpt
    data = torch.load(ckpt_path)
    transformer.load_state_dict(data["module"])

    # Cast model
    transformer.to(device=device, dtype=weight_dtype)
    vae.to(device=device, dtype=weight_dtype)
    transformer.eval()
    vae.eval()
    return transformer, vae, distributed_state

def main(
    config,
    **kwargs
):
    # For dynamic scenes, loop over all target times
    target_index_manual = config.target_index_manual
    if target_index_manual is None and config.target_index_manual_start_idx is not None:
        target_index_manual = list(range(config.target_index_manual_start_idx, config.target_index_manual_start_idx + config.target_index_manual_num_idx, config.target_index_manual_stride))
    
    if target_index_manual is not None and not isinstance(target_index_manual, int):
        for target_index_manual_manual_i in target_index_manual:
            print(f"Bullet time {target_index_manual_manual_i}")
            config.target_index_manual = target_index_manual_manual_i
            transformer, vae, distributed_state, ckpt_path = main_single(config, **kwargs)
            kwargs['transformer'] = transformer
            kwargs['vae'] = vae
            kwargs['distributed_state'] = distributed_state
            kwargs['ckpt_path'] = ckpt_path
    else:
        main_single(config, **kwargs)

def main_single(
    config,
    seed: int = 0,
    transformer = None,
    vae = None,
    distributed_state = None,
    ckpt_path = None,
):
    weight_dtype = torch.bfloat16
    out_fps = config.out_fps
    g = torch.Generator()
    g.manual_seed(seed)
    seed_everything(seed)
    
    outdir = config.out_dir_inference + "_progressive"

    if isinstance(config.config_path, str):
        main_config = OmegaConf.load(config.config_path)
    else:
        main_config = load_and_merge_configs(config.config_path)

    # Get latest checkpoint
    ckpt_name = None
    if ckpt_path is None:
        if config.ckpt_path is None:
            ckpt_model_sub_path = 'pytorch_model/mp_rank_00_model_states.pt'
            ckpts_path = main_config.output_dir
            ckpt_name = config.ckpt_name
            if ckpt_name is None:
                ckpt_name = get_most_recent_checkpoint(ckpts_path)
            ckpt_path = os.path.join(ckpts_path, ckpt_name, ckpt_model_sub_path)
            
        else:
            ckpt_path = config.ckpt_path
    if ckpt_name is None:
        has_ckpt_name = re.search(r"(checkpoint-\d+)", ckpt_path)
        if has_ckpt_name:
            ckpt_name = has_ckpt_name.group(1)
    if ckpt_name is not None:
        outdir = os.path.join(outdir, ckpt_name)
    if os.path.isfile(ckpt_path):
        print(f"Found ckpt at path {ckpt_path}")
    else:
        raise ValueError(f"Could not find ckpt at path {ckpt_path}")
    
    if config.set_manual_time_idx:
        main_config.set_manual_time_idx = config.set_manual_time_idx
    
    # Fix keys
    OmegaConf.update(main_config, "target_index_manual", config.target_index_manual, force_add=True)
    if config.target_index_subsample is not None:
        OmegaConf.update(main_config, "target_index_subsample", config.target_index_subsample, force_add=True)
    else:
        OmegaConf.update(main_config, "target_index_subsample", None, force_add=True)

    # Set view indices
    if config.static_view_indices_fixed is not None:
        main_config.static_view_indices_fixed = config.static_view_indices_fixed
        outdir = os.path.join(outdir, f"static_view_indices_fixed_{'_'.join(config.static_view_indices_fixed)}")
        main_config.static_view_indices_sampling = 'fixed'
        main_config.num_input_multi_views = len(config.static_view_indices_fixed)
    
    orig_num_input_multi_views = main_config.num_input_multi_views
    frames_per_view_raw = main_config.num_input_views 

    main_config.batch_size = 1
    main_config.gs_view_chunk_size = 1
    main_config.num_train_images = 1

    if config.dataset_name is not None:
        main_config.data_mode = [[config.dataset_name, 1]]
        outdir = os.path.join(outdir, config.dataset_name)
    
    if config.target_index_manual is not None:
        outdir = os.path.join(outdir, str(config.target_index_manual))

    if config.num_test_images is not None:
        main_config.num_test_images = config.num_test_images
    
    main_config.use_depth = config.use_depth

    train_dataloader, test_dataloader = get_multi_dataloader(main_config)
    if transformer is None and vae is None and distributed_state is None:
        transformer, vae, distributed_state = load_model(ckpt_path, main_config, weight_dtype)
    
    # Output dirs
    outdir_raw = os.path.join(outdir, "raw")
    outdir_meta = os.path.join(outdir, "meta")
    outdir_full = os.path.join(outdir, "full_output")
    for d in [outdir, outdir_raw, outdir_meta, outdir_full]:
        os.makedirs(d, exist_ok=True)
    
    for idx, batch_test in tqdm(enumerate(test_dataloader)):
        batch_file_name = batch_test['file_name']
        meta_data_sample = {'file_name': batch_file_name}
        meta_data_out_path = os.path.join(outdir_meta, f'sample_{idx}.json')
        if os.path.isfile(meta_data_out_path) and config.skip_existing:
            tqdm.write(f"Skipping {batch_file_name} already exists")
            continue
        
        for batch_k, batch_v in batch_test.items():
            if not isinstance(batch_v, torch.Tensor):
                continue
            batch_test[batch_k] = batch_v.to(distributed_state.device)
            if batch_k not in ['intrinsics_input', 'c2ws_input', 'cam_view', 'intrinsics', 'file_name']:
                batch_test[batch_k] = batch_test[batch_k].to(weight_dtype)
        
        if main_config.compute_plucker_cuda:
            batch_test['plucker_embedding'], batch_test['rays_os'], batch_test['rays_ds'] = get_plucker_embedding_and_rays(
                batch_test['intrinsics_input'],
                batch_test['c2ws_input'],
                main_config.img_size,
                main_config.patch_size_out_factor,
                batch_test['flip_flag'],
                get_batch_index=False,
                dtype=dtype_map[main_config.compute_plucker_dtype],
                out_dtype=weight_dtype
            )

        # --- PROGRESSIVE SAMPLING LOGIC ---
        
        # Determine the number of views loaded
        # Note: We rely on the config to tell us how many views were requested (orig_num_input_multi_views)
        # to properly calculate the split size.
        
        # 1. Determine Slice Length for Input (Latents or RGB)
        slice_len_input = 0
        if 'rgb_latents' in batch_test:
            # Latent shape: [B, Total_Latent_Frames, ...]
            # We want Latent_Frames_Per_View
            total_latent_frames = batch_test['rgb_latents'].shape[1]
            slice_len_input = total_latent_frames // orig_num_input_multi_views
            batch_test['rgb_latents'] = batch_test['rgb_latents'][:, :slice_len_input]
        elif 'images_input_vae' in batch_test:
            # RGB shape: [B, Total_Raw_Frames, ...]
            total_raw_frames = batch_test['images_input_vae'].shape[1]
            slice_len_input = total_raw_frames // orig_num_input_multi_views
            batch_test['images_input_vae'] = batch_test['images_input_vae'][:, :slice_len_input]
        
        # 2. Slice Rays / Plucker (Dimensions usually match RGB frames or Latent frames depending on implementation)
        # Usually these match the input tensor dimension 1.
        for key in ['plucker_embedding', 'rays_os', 'rays_ds']:
            if key in batch_test:
                # Assuming these tensors have the same structure [B, T, ...]
                total_len = batch_test[key].shape[1]
                slice_len = total_len // orig_num_input_multi_views
                batch_test[key] = batch_test[key][:, :slice_len]

        # 3. Slice Time Embeddings (CRITICAL FIX)
        if 'time_embeddings' in batch_test and isinstance(batch_test['time_embeddings'], torch.Tensor):
             # time_embeddings structure: [B, (T_input * Num_Views) + 1_Target, ...]
             total_time_len = batch_test['time_embeddings'].shape[1]
             # The input portion is everything except the last element
             input_portion_len = total_time_len - 1
             # Per view input length
             time_slice_len = input_portion_len // orig_num_input_multi_views
             
             # Reconstruct: View 0 Input + Target
             batch_test['time_embeddings'] = torch.cat([
                 batch_test['time_embeddings'][:, :time_slice_len], # View 0
                 batch_test['time_embeddings'][:, -1:]              # Target
             ], dim=1)

        # 4. Construct Targets: Trajectory 0 (Input) AND Trajectory 1 (Next)
        # c2ws_input shape: [B, Total_Frames_Raw, 4, 4]
        total_raw_frames = batch_test['c2ws_input'].shape[1]
        raw_slice_len = total_raw_frames // orig_num_input_multi_views
        
        # Ensure we have enough data for 2 trajectories
        if total_raw_frames < 2 * raw_slice_len:
            print(f"Warning: Not enough frames for trajectory 1. Using trajectory 0 for both.")
            target_c2ws = torch.cat([
                batch_test['c2ws_input'][:, :raw_slice_len], 
                batch_test['c2ws_input'][:, :raw_slice_len]
            ], dim=1)
            target_intrinsics = torch.cat([
                batch_test['intrinsics_input'][:, :raw_slice_len], 
                batch_test['intrinsics_input'][:, :raw_slice_len]
            ], dim=1)
        else:
            # View 0 (0 to T) and View 1 (T to 2T)
            target_c2ws = torch.cat([
                batch_test['c2ws_input'][:, :raw_slice_len], 
                batch_test['c2ws_input'][:, raw_slice_len:2*raw_slice_len]
            ], dim=1)

            target_intrinsics = torch.cat([
                batch_test['intrinsics_input'][:, :raw_slice_len],
                batch_test['intrinsics_input'][:, raw_slice_len:2*raw_slice_len]
            ], dim=1)

        batch_test['cam_view'] = torch.inverse(target_c2ws).transpose(1, 2)
        batch_test['intrinsics'] = target_intrinsics

        # 5. Override Config for Model
        batch_test['num_input_multi_views'] = 1 
        num_input_multi_views = 1 

        # --- END PROGRESSIVE LOGIC ---

        # Encode video
        if 'rgb_latents' in batch_test:
            model_input = batch_test['rgb_latents'].to(weight_dtype) 
            batch_test['images_input_embed'] = model_input
            video = None
        else:
            video = batch_test['images_input_vae']
            if main_config.use_rgb_decoder:
                model_input = video
            else:
                model_input = encode_multi_view_video(vae, video, num_input_multi_views, main_config.vae_backbone)
            batch_test['images_input_embed'] = model_input

        if main_config.time_embedding_vae:
            batch_test = encode_latent_time_vae(batch_test, lambda x: encode_video(vae, x, main_config.vae_backbone), main_config.img_size)
        if main_config.plucker_embedding_vae:
            batch_test = encode_plucker_vae(batch_test, lambda x: encode_multi_view_video(vae, x, num_input_multi_views, main_config.vae_backbone))
        
        with torch.no_grad():
            model_output = transformer(batch_test)
        
        # Get RGB
        pred_images = model_output['images_pred'].cpu() # Shape: [B, 2*T, C, H, W]
        
        # Split output (T corresponds to frames_per_view_raw, which is Raw T)
        # Note: renderer outputs frames matching batch_test['cam_view'], which has 2 * raw_slice_len frames
        T_out = raw_slice_len
        
        view_0_pred = pred_images[:, :T_out]
        view_1_pred = pred_images[:, T_out:]

        save_video(view_0_pred, outdir_full, name=f'sample_{idx}_view_0', fps=out_fps)
        save_video(view_1_pred, outdir_full, name=f'sample_{idx}_view_1', fps=out_fps)

        write_dict_to_json(meta_data_sample, meta_data_out_path)

        tqdm.write(f"Saved batch index {idx} (View 0 & View 1) to {outdir}")
        
    tqdm.write(f"Saved all results to {outdir}")
    return transformer, vae, distributed_state, ckpt_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--config_default', type=str, default='configs/inference/default.yaml')
    args, unknown = parser.parse_known_args()
    config = load_and_merge_configs([args.config_default, args.config])
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    main(config)