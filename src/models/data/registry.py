from src.models.data.radym_wrapper import RadymWrapper
from src.models.data.spatialvid_wrapper import SpatialVidWrapper
from src.models.data.spatialvid import SpatialVid

dataset_registry = {}

dataset_registry['spatialvid'] = {
    'cls': SpatialVidWrapper,
    'kwargs': {
        "cls": SpatialVid,
        "root_path": "/iopsstor/scratch/cscs/pmartell/SpatialVid/HQ",
        "metadata_csv": "data/train/SpatialVID_HQ_metadata.csv",
        "is_static": True,
        "is_multi_view": False,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
    'is_w2c': False,
}

dataset_registry['default'] = {
    "has_latents": False,
    "is_w2c": False,
    "is_generated_cosmos_latent": False,
    "sampling_buckets": None,
    "end_view_idx": None,
    "start_view_target_idx": None,
    "end_view_target_idx": None,
}

# Training static

dataset_registry['lyra_static'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "/path/to/static",
        "is_static": True,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Training dynamic

dataset_registry['lyra_dynamic'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "/path/to/dynamic",
        "is_static": False,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
        "end_view_idx": 5,
        "start_view_target_idx": 6,
        "end_view_target_idx": 11,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Static demo (pre-generated)

dataset_registry['lyra_static_demo'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "assets/demo/static/diffusion_output",
        "is_static": True,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Static demo (self-generated)

dataset_registry['lyra_static_demo_generated'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "assets/demo/static/diffusion_output_generated",
        "is_static": True,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Dynamic demo (pre-generated)

dataset_registry['lyra_dynamic_demo'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "assets/demo/dynamic/diffusion_output",
        "is_static": False,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
        "end_view_idx": 5,
        "start_view_target_idx": 6,
        "end_view_target_idx": 11,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}

# Dynamic demo (self-generated)

dataset_registry['lyra_dynamic_demo_generated'] = {
    'cls': RadymWrapper, 
    'kwargs': {
        "root_path": "assets/demo/dynamic/diffusion_output_generated",
        "is_static": False,
        "is_multi_view": True,
        "has_latents": True,
        "is_generated_cosmos_latent": True,
        "sampling_buckets": [['0'], ['1'], ['2'], ['3'], ['4'], ['5']],
        "start_view_idx": 0,
        "end_view_idx": 5,
        "start_view_target_idx": 6,
        "end_view_target_idx": 11,
    },
    'scene_scale': 1.,
    'max_gap': 121,
    'min_gap': 45,
}
