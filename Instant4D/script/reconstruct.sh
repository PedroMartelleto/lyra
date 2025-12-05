# we apply this script to reconstruct with megaSAM
# run source recover.sh if you want to switch between conda environments
# conda activate instant4d

# directories to reconstruct
evalset=(
  panda
)

# replace with own paths
DATA_DIR=Instant4D/example
Depth_DIR=Instant4D/SLAM/medium
CKPT_PATH=checkpoints/megasam_final.pth # download the weight 
Anything_weight=checkpoints/depth_anything_vitl14.pth
raft_ckpt=checkpoints/raft-things.pth

# conda activate unidepth # in case you have another environment 
cd SLAM/mega-sam  

export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

for seq in ${evalset[@]}; do
    DATA_PATH=$DATA_DIR/$seq
    CUDA_VISIBLE_DEVICES=3 python UniDepth/scripts/demo_mega-sam.py \
    --scene-name $seq \
    --img-path $DATA_PATH \
    --outdir $Depth_DIR/UniDepth/
done



# conda activate instant4d
# Run DepthAnything
for seq in ${evalset[@]}; do
  DATA_PATH=$DATA_DIR/$seq/
  CUDA_VISIBLE_DEVICES=3 python Depth-Anything/run_videos.py --encoder vitl \
  --load-from $Anything_weight \
  --img-path $DATA_PATH \
  --outdir $Depth_DIR/Depth-Anything/$seq
done



for seq in ${evalset[@]}; do
    DATA_PATH=$DATA_DIR/$seq
    CUDA_VISIBLE_DEVICES=3 python3 camera_tracking_scripts/test_demo.py \
    --datapath=$DATA_PATH \
    --weights=$CKPT_PATH \
    --scene_name $seq \
    --mono_depth_path  $Depth_DIR/Depth-Anything \
    --metric_depth_path $Depth_DIR/UniDepth \
    --disable_vis $@
done


# Run Raft Optical Flows
for seq in ${evalset[@]}; do
  DATA_PATH=$DATA_DIR/$seq
  CUDA_VISIBLE_DEVICES=3 python3 cvd_opt/preprocess_flow.py \
  --datapath=$DATA_PATH \
  --model=$raft_ckpt \
  --scene_name $seq --mixed_precision
done

# Run CVD optmization
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=3 python3 cvd_opt/cvd_opt.py \
  --scene_name $seq \
  --w_grad 2.0 --w_normal 5.0
done

cd ../..