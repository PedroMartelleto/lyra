import os
import sys
import argparse
import subprocess
import glob
import json
import shutil
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zipfile
from pathlib import Path
from tqdm import tqdm
from typing import NamedTuple, Optional, List, Dict

# Try to import gsplat
try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Error: 'gsplat' library not found. Please install it via 'pip install gsplat'.")
    sys.exit(1)

# ==============================================================================
# Utils: Math & Geometric
# ==============================================================================

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def build_rotation(r):
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] + r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

# ==============================================================================
# Model: 3D Gaussian Splatting Optimization
# ==============================================================================

class GaussianModel(nn.Module):
    def __init__(self, sh_degree: int = 3):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        
        # Parameters
        self._xyz = nn.Parameter(torch.empty(0))
        self._features_dc = nn.Parameter(torch.empty(0))
        self._features_rest = nn.Parameter(torch.empty(0))
        self._scaling = nn.Parameter(torch.empty(0))
        self._rotation = nn.Parameter(torch.empty(0))
        self._opacity = nn.Parameter(torch.empty(0))
        
        # Activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = F.normalize

        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        # Densification stats
        self.max_radii2D = torch.empty(0)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_sh_degree,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_sh_degree,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        
        # RGB to SH (approx)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = (fused_color - 0.5) / 0.28209479177387814
        
        print(f"Number of points at initialization : {fused_point_cloud.shape[0]}")

        # Initialize attributes
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_split(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(2, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(2, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(2, 1) / 1.6)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(2, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(n_init_points * 2, device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

# Helper for KNN
def distCUDA2(points):
    try:
        from simple_knn._C import distCUDA2
        return distCUDA2(points)
    except ImportError:
        # Fallback: simple heuristic, random 0.01 scale for everything
        return torch.ones(points.shape[0], device=points.device) * 0.01

# ==============================================================================
# ViPE & Data Handling
# ==============================================================================

def run_vipe(input_video_path, output_dir):
    """
    Runs ViPE to extract depth, poses, and intrinsics.
    """
    print(f"--- Running ViPE on {input_video_path} ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cmd = [
        "vipe", "infer", 
        str(input_video_path), 
        "-p", "lyra", 
        "--output", str(output_dir)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("--- ViPE inference completed ---")
    except subprocess.CalledProcessError as e:
        print(f"Error running ViPE: {e}")
        print("Ensure 'vipe' is installed and the environment is configured correctly.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'vipe' command not found. Please install ViPE.")
        sys.exit(1)

    return output_dir

class ViPEDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, device="cuda"):
        self.device = device
        self.rgb_dir = os.path.join(data_root, "rgb")
        self.depth_dir = os.path.join(data_root, "depth")
        self.pose_dir = os.path.join(data_root, "pose")
        self.intrinsics_dir = os.path.join(data_root, "intrinsics")

        # 0. Check and Extract ZIPs in all folders
        self._extract_zip_if_needed(self.rgb_dir)
        self._extract_zip_if_needed(self.depth_dir)
        self._extract_zip_if_needed(self.pose_dir)
        self._extract_zip_if_needed(self.intrinsics_dir)

        # 1. Load RGB Files (Recursive search)
        self.rgb_files = sorted(
            glob.glob(os.path.join(self.rgb_dir, "**/*.png"), recursive=True) + 
            glob.glob(os.path.join(self.rgb_dir, "**/*.jpg"), recursive=True)
        )
        if len(self.rgb_files) == 0:
            mp4_files = glob.glob(os.path.join(self.rgb_dir, "*.mp4"))
            if len(mp4_files) > 0:
                print("Found MP4 in RGB folder, extracting frames...")
                self._extract_frames(mp4_files[0], self.rgb_dir, ext="png")
                self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, "*.png")))
            else:
                raise FileNotFoundError(f"No RGB images or videos found in {self.rgb_dir}")

        num_frames = len(self.rgb_files)
        print(f"Found {num_frames} RGB frames.")

        # 2. Load Depth Files (Recursive search: png OR exr)
        self.depth_files = sorted(
            glob.glob(os.path.join(self.depth_dir, "**/*.png"), recursive=True) +
            glob.glob(os.path.join(self.depth_dir, "**/*.exr"), recursive=True)
        )
        self.depth_mode = "files"
        
        # Handle cases where depth files are missing or in a single npy
        if len(self.depth_files) == 0:
            # Check for mp4 depth
            mp4_depth = glob.glob(os.path.join(self.depth_dir, "*.mp4"))
            if len(mp4_depth) > 0:
                print("Found MP4 in Depth folder, extracting frames...")
                self._extract_frames(mp4_depth[0], self.depth_dir, ext="png")
                self.depth_files = sorted(glob.glob(os.path.join(self.depth_dir, "*.png")))
            else:
                # Check for single big .npy file
                npy_files = sorted(glob.glob(os.path.join(self.depth_dir, "**/*.npy"), recursive=True))
                if len(npy_files) > 0:
                    print(f"Loading depth from single NPY file: {npy_files[0]}")
                    self.depth_data = np.load(npy_files[0]) 
                    if self.depth_data.ndim == 3 and self.depth_data.shape[0] == num_frames:
                        self.depth_mode = "memory"
                    elif self.depth_data.ndim == 2:
                        print("Warning: Single depth frame found, broadcasting.")
                        self.depth_data = np.stack([self.depth_data] * num_frames)
                        self.depth_mode = "memory"
                    else:
                        # Maybe multiple npy files corresponding to frames
                        if len(npy_files) == num_frames:
                            self.depth_files = npy_files
                            self.depth_mode = "files"
        
        if self.depth_mode == "files" and len(self.depth_files) != num_frames:
            # Check for npy files if pngs/exr failed
            self.depth_files = sorted(glob.glob(os.path.join(self.depth_dir, "**/*.npy"), recursive=True))
            if len(self.depth_files) != num_frames:
                raise ValueError(f"Mismatch: {num_frames} RGB frames but {len(self.depth_files)} depth files found in {self.depth_dir}")

        # 3. Load Poses & Intrinsics
        # Prefer NPZ > NPY > TXT
        self.pose_files = sorted(
            glob.glob(os.path.join(self.pose_dir, "**/*.npz"), recursive=True) +
            glob.glob(os.path.join(self.pose_dir, "**/*.npy"), recursive=True) +
            glob.glob(os.path.join(self.pose_dir, "**/*.txt"), recursive=True)
        )
        self.intr_files = sorted(
            glob.glob(os.path.join(self.intrinsics_dir, "**/*.npz"), recursive=True) +
            glob.glob(os.path.join(self.intrinsics_dir, "**/*.npy"), recursive=True) +
            glob.glob(os.path.join(self.intrinsics_dir, "**/*.txt"), recursive=True)
        )

        self.pose_data = None
        self.intr_data = None
        
        # Load Intrinsics
        if len(self.intr_files) >= 1:
             # Prioritize first match (sorted puts npz first if structured correctly, 
             # but here we manually ordered the list construction above so [0] is best candidate)
             fpath = self.intr_files[0]
             print(f"Loading intrinsics from single file: {fpath}")
             
             if fpath.endswith('.npz'):
                 data = np.load(fpath)
                 if 'data' in data:
                     loaded_intr = data['data']
                 else:
                     loaded_intr = data[list(data.keys())[0]]
             elif fpath.endswith('.npy'):
                 loaded_intr = np.load(fpath)
             else:
                 # Text file with potential index prefix "0: ..."
                 try:
                     loaded_intr = np.loadtxt(fpath)
                 except ValueError:
                     # Parse lines like "0: fx fy cx cy"
                     intr_list = []
                     with open(fpath, 'r') as f:
                         for line in f:
                             line = line.strip()
                             if not line: continue
                             # Strip "0: " prefix if present
                             if ':' in line:
                                 vals = line.split(':')[-1].strip().split()
                             else:
                                 vals = line.split()
                             intr_list.append([float(v) for v in vals])
                     loaded_intr = np.array(intr_list)

             # Process dimensions
             if loaded_intr.ndim == 1 and loaded_intr.size == 4:
                 # Single [fx, fy, cx, cy] -> Broadcast
                 self.intr_data = np.stack([loaded_intr] * num_frames)
             elif loaded_intr.ndim == 2 and loaded_intr.shape[0] == num_frames:
                 self.intr_data = loaded_intr
             elif loaded_intr.ndim == 3 and loaded_intr.shape[0] == num_frames:
                 # already N x 3 x 3
                 self.intr_data = loaded_intr
             else:
                 # Attempt broadcast if shape doesn't match N
                 if loaded_intr.shape[0] == 1:
                     self.intr_data = np.stack([loaded_intr[0]] * num_frames)
                 else:
                     print(f"Warning: Intrinsics shape {loaded_intr.shape} doesn't match {num_frames} frames. Using first frame.")
                     self.intr_data = np.stack([loaded_intr[0]] * num_frames)

        # Load Poses
        if len(self.pose_files) >= 1:
             fpath = self.pose_files[0]
             print(f"Loading poses from single file: {fpath}")
             
             if fpath.endswith('.npz'):
                 data = np.load(fpath)
                 if 'data' in data:
                     self.pose_data = data['data']
                 else:
                     self.pose_data = data[list(data.keys())[0]]
             elif fpath.endswith('.npy'):
                 self.pose_data = np.load(fpath)
             else:
                 try:
                     self.pose_data = np.loadtxt(fpath)
                 except ValueError:
                     # Manual parse "0: r00 r01 ... t0 ..."
                     pose_list = []
                     with open(fpath, 'r') as f:
                         for line in f:
                             line = line.strip()
                             if not line: continue
                             if ':' in line:
                                 vals = line.split(':')[-1].strip().split()
                             else:
                                 vals = line.split()
                             pose_list.append([float(v) for v in vals])
                     self.pose_data = np.array(pose_list)
            
             # Reshape if flat
             if self.pose_data.ndim == 2 and self.pose_data.shape[1] == 12:
                 self.pose_data = self.pose_data.reshape(-1, 3, 4)
             elif self.pose_data.ndim == 2 and self.pose_data.shape[1] == 16:
                 self.pose_data = self.pose_data.reshape(-1, 4, 4)

    def _extract_zip_if_needed(self, dir_path):
        zips = glob.glob(os.path.join(dir_path, "*.zip"))
        if len(zips) > 0:
            print(f"Extracting {zips[0]} in {dir_path}...")
            with zipfile.ZipFile(zips[0], 'r') as zip_ref:
                zip_ref.extractall(dir_path)

    def _extract_frames(self, video_path, out_dir, ext="png"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imwrite(os.path.join(out_dir, f"{i:05d}.{ext}"), frame)
            i += 1
        cap.release()

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Read RGB
        rgb = cv2.imread(self.rgb_files[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        rgb = torch.from_numpy(rgb / 255.0).float().to(self.device).permute(2, 0, 1)

        # Read Depth
        if self.depth_mode == "memory":
            depth = self.depth_data[idx]
        else:
            fpath = self.depth_files[idx]
            if fpath.endswith('.npy'):
                depth = np.load(fpath)
            elif fpath.endswith('.exr'):
                # Read EXR using OpenCV with ANYDEPTH/UNCHANGED to keep float32
                depth = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                # ViPE output is metric depth in meters
            else:
                depth = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH)
                if depth is None:
                    # Fallback for 8-bit visual depth (bad but prevents crash)
                    depth = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    depth = depth.astype(np.float32) / 255.0 * 10.0 # arbitrary scale
                else:
                    depth = depth.astype(np.float32) / 1000.0 # mm to m
        
        # EXR might be 3 channel (R=G=B=Depth), take first channel
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        
        depth = torch.from_numpy(depth).float().to(self.device).unsqueeze(0)

        # Read Pose (c2w)
        if self.pose_data is not None:
            c2w = self.pose_data[idx]
        else:
            fpath = self.pose_files[idx]
            c2w = np.load(fpath) if fpath.endswith('.npy') else np.loadtxt(fpath)
            
        c2w = torch.from_numpy(c2w).float().to(self.device)
        if c2w.shape == (3, 4):
            c2w = torch.cat([c2w, torch.tensor([[0,0,0,1]], device=self.device, dtype=torch.float)], dim=0)

        # Read Intrinsics
        if self.intr_data is not None:
            K_arr = self.intr_data[idx]
        else:
            fpath = self.intr_files[idx]
            K_arr = np.load(fpath) if fpath.endswith('.npy') else np.loadtxt(fpath)
        
        if K_arr.size == 4:
            fx, fy, cx, cy = K_arr.flatten()
            K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=self.device, dtype=torch.float)
        else:
            K = torch.from_numpy(K_arr).float().to(self.device)

        return {
            'image': rgb,
            'depth': depth,
            'c2w': c2w,
            'K': K,
            'H': H, 
            'W': W
        }

    def generate_point_cloud(self):
        """ Unprojects all depth maps to form a combined point cloud for initialization. """
        print("Generating initial point cloud from ViPE depth...")
        all_points = []
        all_colors = []
        
        # Subsample frames for speed
        indices = range(0, len(self), max(1, len(self)//20))
        
        for idx in indices:
            data = self[idx]
            rgb = data['image'].permute(1, 2, 0) # H, W, 3
            depth = data['depth'].squeeze(0) # H, W
            c2w = data['c2w']
            K = data['K']
            H, W = data['H'], data['W']

            # Grid
            y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            
            # Unproject
            z = depth
            x3d = (x - K[0, 2]) * z / K[0, 0]
            y3d = (y - K[1, 2]) * z / K[1, 1]
            xyz = torch.stack([x3d, y3d, z], dim=-1) # H, W, 3
            
            # Mask invalid depth
            mask = (z > 0) & (z < 100)
            xyz = xyz[mask]
            col = rgb[mask]
            
            # Transform to World
            xyz_world = (c2w[:3, :3] @ xyz.T).T + c2w[:3, 3]
            
            all_points.append(xyz_world.cpu().numpy())
            all_colors.append(col.cpu().numpy())
            
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        
        # Random downsample if too large
        if all_points.shape[0] > 100_000:
            choice = np.random.choice(all_points.shape[0], 100_000, replace=False)
            all_points = all_points[choice]
            all_colors = all_colors[choice]
            
        return BasicPointCloud(points=all_points, colors=all_colors, normals=np.zeros_like(all_points))

# ==============================================================================
# Training Logic
# ==============================================================================

class TrainingArgs(NamedTuple):
    percent_dense: float = 0.01
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002

def train_3dgs(dataset, output_path, iterations=2000):
    print(f"--- Starting 3DGS Fitting for {iterations} iterations ---")
    
    # 1. Init Model
    pcd = dataset.generate_point_cloud()
    spatial_lr_scale = np.linalg.norm(pcd.points.max(0) - pcd.points.min(0))
    
    gaussians = GaussianModel()
    gaussians.create_from_pcd(pcd, spatial_lr_scale)
    
    opt_args = TrainingArgs()
    gaussians.training_setup(opt_args)
    
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pbar = tqdm(range(1, iterations + 1))
    
    for iteration in pbar:
        gaussians.update_learning_rate(iteration)
        
        # Pick random view
        view_idx = np.random.randint(0, len(dataset))
        view = dataset[view_idx]
        
        gt_image = view['image']
        gt_c2w = view['c2w']
        gt_K = view['K']
        H, W = view['H'], view['W']
        
        means3D = gaussians.get_xyz
        rotations = gaussians.get_rotation
        scales = gaussians.get_scaling
        opacity = gaussians.get_opacity
        shs = gaussians.get_features
        
        w2c = torch.linalg.inv(gt_c2w)
        
        rendered_image, alpha, meta = rasterization(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=shs, 
            viewmats=w2c.unsqueeze(0),
            Ks=gt_K.unsqueeze(0),
            width=W,
            height=H,
            packed=False,
            backgrounds=bg_color.unsqueeze(0)
        )
        
        pred_image = rendered_image[0].permute(2, 0, 1)
        
        l1_loss = F.l1_loss(pred_image, gt_image)
        total_loss = l1_loss 
        
        total_loss.backward()
        
        with torch.no_grad():
            if iteration < opt_args.densify_until_iter:
                gaussians.max_radii2D[meta['camera_ids']] = torch.max(
                    gaussians.max_radii2D[meta['camera_ids']],
                    meta['radii']
                )
                gaussians.xyz_gradient_accum[meta['camera_ids']] += torch.norm(
                    meta['means2d'].grad[meta['camera_ids'], :2], dim=-1, keepdim=True
                )
                gaussians.denom[meta['camera_ids']] += 1

                if iteration > opt_args.densify_from_iter and iteration % opt_args.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_args.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt_args.densify_grad_threshold, 0.005, spatial_lr_scale, size_threshold)
                
                if iteration % opt_args.opacity_reset_interval == 0 or (dataset.device == "cuda" and iteration == opt_args.densify_from_iter):
                    gaussians._opacity.data[:] = inverse_sigmoid(torch.min(gaussians.get_opacity, torch.ones_like(gaussians.get_opacity)*0.01))

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            
        if iteration % 500 == 0:
            pbar.set_description(f"Loss: {total_loss.item():.4f} | Pts: {gaussians.get_xyz.shape[0]}")

    print(f"Saving model to {output_path}...")
    torch.save(gaussians.capture(), os.path.join(output_path, "model.pt"))
    print("Fitting Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit 3DGS using ViPE depth priors.")
    parser.add_argument("input_video", type=str, help="Path to input MP4 video.")
    parser.add_argument("--output_dir", type=str, default="output_3dgs", help="Output directory.")
    parser.add_argument("--vipe_output_dir", type=str, default="vipe_out", help="Directory for ViPE intermediates.")
    parser.add_argument("--steps", type=int, default=2000, help="Optimization steps.")
    args = parser.parse_args()

    # 1. Run ViPE to get Depth/Poses
    vipe_data_dir = run_vipe(args.input_video, args.vipe_output_dir)
    
    # 2. Load the data
    dataset = ViPEDataset(vipe_data_dir)
    
    # 3. Fit 3DGS
    os.makedirs(args.output_dir, exist_ok=True)
    train_3dgs(dataset, args.output_dir, iterations=args.steps)