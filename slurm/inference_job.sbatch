#!/bin/bash
#SBATCH --account=a144
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --job-name=infer-gen3c_parallel
#SBATCH --output=infer_gen3c_parallel.out
#SBATCH --error=infer_gen3c_parallel.err

# export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)

# srun --environment=instant4d bash -c "
# cd /iopsstor/scratch/cscs/pmartell/lyra/
# python3 da3_inference.py
# "

# srun --environment=lyra bash -c "
# cd /iopsstor/scratch/cscs/pmartell/lyra/
# accelerate launch sample.py --config configs/demo/lyra_static.yaml
# "

# srun --environment=lyra bash -c "
# cd /iopsstor/scratch/cscs/pmartell/lyra/
# torchrun --nproc_per_node=4 cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
#     --checkpoint_dir checkpoints \
#     --num_gpus 4 \
#     --input_image_path assets/Garden.jpg \
#     --video_save_folder assets/demo/static/speed9 \
#     --foreground_masking \
#     --multi_trajectory \
#     --total_movement_distance_factor 9.0
# "

srun --environment=lyra bash -c "
cd /iopsstor/scratch/cscs/pmartell/lyra/
torchrun --nproc_per_node=1 cosmos_predict1/diffusion/inference/big_cache_gen3c_single_image.py \
    --checkpoint_dir checkpoints \
    --num_gpus 1 \
    --input_image_path assets/Garden.jpg \
    --video_save_folder assets/demo/static/bigCache_speed5_more_trajs \
    --foreground_masking \
    --multi_trajectory \
    --total_movement_distance_factor 5.0 \
    --save_buffer
"