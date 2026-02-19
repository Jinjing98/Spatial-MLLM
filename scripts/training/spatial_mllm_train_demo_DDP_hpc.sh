#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1  #JJ was 2 for init 2 sft models
#SBATCH --gres=gpu:2           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --gpus-per-task=2 #JJ was 1 for init 2 sft models
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

set -euo pipefail

source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
module load CUDA/12.4.0
cd $SLURM_SUBMIT_DIR

# Set environment variables
# export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_PROJECT="Spatial-MLLM-SFT"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

PRETRAINED_CKPT_ROOT="/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM/"

# DATASET_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets/vsibench"  # Dataset root directory
# DATASETS="spatial_mllm_mix_10_dbg" # default "spatial_mllm_mix_133k,route_plan_scannet_2k"

DATASET_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets/SQA3D"  # Dataset root directory
DATASETS="sqa3d_filtered_40k" # default "sqa3d_filtered_40k,sqa3d_filtered_40k_small"

# DATASET_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets/vsibench"  # Dataset root directory
# Export DATASET_ROOT for Python scripts (__init__.py) to use for data loading
export DATASET_ROOT
# JJ Freq Edit
OUTPUT_ROOT="/data/horse/ws/jixu233b-metadata_ws/exps/train/spatialmllm"
TRAIN_EPOCHS=1 # default 1 
NUM_WORKERS=2 # default 8, set to 0 to avoid multiprocessing overhead
NPROC_PER_NODE=2 # default 6 
GRAD_ACCUM_STEPS=8 # JJ: reduced from 8 to match 4-sample debug dataset (4 samples / 2 GPUs = 2 per GPU)
BATCH_SIZE=1 # default 1 
VIDEO_MAX_FRAMES=16 # default 16
VIDEO_MIN_FRAMES=16 # default 16
VIDEO_FRAME_FPS=4 # default 4
GRADIENT_CHECKPOINTING=True # default False
MODEL_TYPE="spatial-mllm" #"custom-spatial-mllm" # spatial-mllm
PRETRAINED_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
RUN_NAME_APPENDIX="_2x8_hpc"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration (disabled for single GPU training)
# deepspeed=./scripts/training/zero3.json
USE_DEEPSPEED=False  # Set to True to enable DeepSpeed

# Model configuration
# model_type=spatial-mllm
# vggt_checkpoints_path=checkpoints/VGGT-1B/model.safetensors
vggt_checkpoints_path="${PRETRAINED_CKPT_ROOT}checkpoints/VGGT-1B/model.safetensors"
spatial_embeds_layer_idx=-1
connector_type=mlp_add 
# pretrained_model_name_or_path=Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=7e-6
mm_projector_lr=2e-5
weight_decay=0.1
max_grad_norm=1.0
# batch_size=1 
# grad_accum_steps=8

# Training entry point
entry_file=src/qwenvl/train/train_qwen.py

# Dataset configuration
# datasets="spatial_mllm_mix_133k,route_plan_scannet_2k"

# Data configuration
max_pixels=324576
min_pixels=293216
video_max_frame_pixels=324576
video_min_frame_pixels=293216
# video_max_frames=16
# video_min_frames=16
# video_frame_fps=4

# Output configuration
timestamp=$(date +'%Y%m%d_%H%M%S')
base_run_name="spatial-mllm-sft"
run_name="${timestamp}_${base_run_name}${RUN_NAME_APPENDIX}"
output_dir=${OUTPUT_ROOT}/${run_name}
mkdir -p ${output_dir}
logfile="${output_dir}/$(date +'%Y%m%d_%H%M%S')_train.log"

# Training arguments
# JJ : Removed --deepspeed for native PyTorch single GPU training
args="
    --model_type ${MODEL_TYPE} \
    --vggt_checkpoints_path ${vggt_checkpoints_path} \
    --spatial_embeds_layer_idx ${spatial_embeds_layer_idx} \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
    --dataset_use ${DATASETS} \
    --tune_mm_vision False \
    --tune_mm_spatial_encoder False \
    --tune_mm_connector True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --video_max_frame_pixels ${video_max_frame_pixels} \
    --video_min_frame_pixels ${video_min_frame_pixels} \
    --video_max_frames ${VIDEO_MAX_FRAMES} \
    --video_min_frames ${VIDEO_MIN_FRAMES} \
    --video_frame_fps ${VIDEO_FRAME_FPS} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.0625 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --weight_decay ${weight_decay} \
    --warmup_ratio 0.03 \
    --max_grad_norm ${max_grad_norm} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --dataloader_num_workers ${NUM_WORKERS} \
    --run_name ${run_name}"
    #  \
    # --report_to wandb"

# Launch training (native PyTorch without DeepSpeed)
# python ${entry_file} ${args} 2>&1 | tee -a "${logfile}"
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} 2>&1 | tee -a "${logfile}"