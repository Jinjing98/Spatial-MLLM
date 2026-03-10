#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1  #JJ was 2 for init 2 sft models
#SBATCH --gres=gpu:2           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --gpus-per-task=2 #JJ was 1 for init 2 sft models
#SBATCH --cpus-per-task=8
#SBATCH --time=60:00:00
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
export WANDB_PROJECT="Spatial-MLLM-SFT"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

PRETRAINED_CKPT_ROOT="/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM/"

# ============ Dataset ============
DATASET_ROOT="/home/jixu233b/Projects/VLM_3D/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/SPMLLM-DATA"
DATASETS="spatial_mllm_mix_133k,route_plan_scannet_2k"
# DATASET_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets/SQA3D"
# DATASETS="sqa3d_filtered_40k"
# DATASET_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets/ViCA-322K"
# DATASETS="vica_322k_base%50"
export DATASET_ROOT

# ============ Training basics ============
# JJ Freq Edit
OUTPUT_ROOT="/data/horse/ws/jixu233b-metadata_ws/exps/train/spatialmllm"
TRAIN_EPOCHS=1 # default 1
NUM_WORKERS=2 # default 8
NPROC_PER_NODE=2 # default 6
GRAD_ACCUM_STEPS=8
BATCH_SIZE=1
VIDEO_MAX_FRAMES=16
VIDEO_MIN_FRAMES=16
VIDEO_FRAME_FPS=4
# JJ : Temporal-merge-aware real-neighbour sampling
# Sample N/2 anchors uniformly, then add a real temporal neighbour for each anchor.
SAMPLING_ENFORCE_REAL_NEIGHBOUR=True  # set True to enable
NEIGHBOUR_MODE="after"                 # before | after | random
NEIGHBOUR_MAX_STEP=0                   # max frame-ID offset; actual step ~ randint(1, max)
GRADIENT_CHECKPOINTING=True
MODEL_TYPE="custom-spatial-mllm-lvsm" #"custom-spatial-mllm-lvsm" or "custom-spatial-mllm"
PRETRAINED_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
RUN_NAME_APPENDIX="_lvsm_2x8_hpc"

# ============ What to train (freeze / unfreeze) ============
# JJ : Base model components
TUNE_VISION=False                # ViT visual encoder
TUNE_SPATIAL_ENCODER=False       # VGGT spatial encoder
TUNE_CONNECTOR=False             # MLPAddConnector (visual.merger + connector)
TUNE_LLM=True                   # Qwen LLM backbone + lm_head
# JJ : LVSM-specific (only when MODEL_TYPE=custom-spatial-mllm-lvsm)
TUNE_CONNECTOR_LVSM=True        # connector_lvsm (FiLM adaptor)
TUNE_LVSM_DECODER=True          # lvsm_model transformer_blocks + image_token_decoder

# ============ Learning rates ============
lr=7e-6                          # base lr (LLM, spatial_encoder, lvsm_model if unfrozen)
mm_projector_lr=2e-5             # visual.merger + connector (MLPAddConnector)
# JJ : Separate lr for LVSM adaptor (connector_lvsm only; lvsm_model uses base lr)
# Linear scaling from LVSM original: 4e-4 * (BS_eff / 512)
# BS_eff=1 → ~8e-7, conservative start: 1e-7
LVSM_ADAPTOR_LR=1e-7
weight_decay=0.1
max_grad_norm=1.0

# ============ LVSM integration config (only for custom-spatial-mllm-lvsm) ============
ENFORCE_LVSM=True
LVSM_CHECKPOINT_PATH="submodule/LVSM/checkpoints/scene_decoder_only_256.pt"
LVSM_IMAGE_SIZE=256
# JJ : NVS_LOSS_WEIGHT tuning guide:
#   CE loss ~ 1-5, raw NVS loss (L2+Percep) ~ 5-50 initially
#   1.0 → NaN (NVS dominates, gradient explosion)
#   0.01 → safe start (NVS contributes ~0.05-0.5 vs CE ~2-5)
#   0.1  → moderate (try after stable training at 0.01)
# JJ : With residual context architecture, NVS starts stable (gate=0).
# 0.01 is a safe starting point.
NVS_LOSS_WEIGHT=0.1
NUM_TARGET_VIEWS=4
LVSM_L2_WEIGHT=1.0
LVSM_PERCEPTUAL_WEIGHT=0.5
LVSM_LPIPS_WEIGHT=0.0
VGG_WEIGHT_FILE="submodule/LVSM/metric_checkpoint/imagenet-vgg-verydeep-19.mat"
NVS_ENABLED=True         # JJ : Load novel target frames for NVS loss
# JJ : NVS target pool — 'nvs' (novel only), 'input' (reconstruction sanity check), 'all' (mixed)
NVS_TARGET_POOL="nvs"
NVS_IMG_LOG_INTERVAL=2  # JJ : Log rendered vs GT images to wandb every N steps (0=disable)
# JJ : Adapter types for LVSM ↔ QwenVL bridges (currently only "linear")
LVSM2QWEN_TYPE="linear"
LLM2LVSM_TYPE="linear"
# JJ : SDPA output gating (arxiv 2505.06708) — element-wise sigmoid gate after attention
ENABLE_SDPA_GATING=False

# ============ 4D Pose RoPE config (only for custom-spatial-mllm) ============
USE_POSE_ROPE=False  # Set to True to enable 4D Pose-aware RoPE
POSE_ENC_TYPE="PTHW"  # Pose encoding type ('PTHW', 'PHW', or 'THW')
MROPE_SECTION="8 8 24 24"  # Custom mrope_section (e.g., "16 24 24" for 3D or "8 8 24 24" for 4D). Leave empty for default.
                  # Examples: 
                  # - For PTHW (4D): MROPE_SECTION="8 8 24 24"
                  # - For PHW/THW (3D): MROPE_SECTION="16 24 24"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration (disabled for single GPU training)
USE_DEEPSPEED=False  # Set to True to enable DeepSpeed

# Model configuration
vggt_checkpoints_path="${PRETRAINED_CKPT_ROOT}checkpoints/VGGT-1B/model.safetensors"
spatial_embeds_layer_idx=-1
connector_type=mlp_add

# Training entry point
entry_file=src/qwenvl/train/train_qwen.py

# Data configuration
max_pixels=324576
min_pixels=293216
video_max_frame_pixels=324576
video_min_frame_pixels=293216

# Output configuration
timestamp=$(date +'%Y%m%d_%H%M%S')
base_run_name="spatial-mllm-sft"
run_name="${timestamp}_${base_run_name}${RUN_NAME_APPENDIX}"
output_dir=${OUTPUT_ROOT}/${run_name}
mkdir -p ${output_dir}
logfile="${output_dir}/$(date +'%Y%m%d_%H%M%S')_train.log"

# Training arguments
args="
    --model_type ${MODEL_TYPE} \
    --vggt_checkpoints_path ${vggt_checkpoints_path} \
    --spatial_embeds_layer_idx ${spatial_embeds_layer_idx} \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
    --dataset_use ${DATASETS} \
    --tune_mm_vision ${TUNE_VISION} \
    --tune_mm_spatial_encoder ${TUNE_SPATIAL_ENCODER} \
    --tune_mm_connector ${TUNE_CONNECTOR} \
    --tune_mm_llm ${TUNE_LLM} \
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
    --sampling_enforce_real_neighbour ${SAMPLING_ENFORCE_REAL_NEIGHBOUR} \
    --neighbour_mode ${NEIGHBOUR_MODE} \
    --neighbour_max_step ${NEIGHBOUR_MAX_STEP} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.0625 \
    --save_total_limit 2 \
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

# JJ : Add LVSM args if model type is custom-spatial-mllm-lvsm
if [ "$MODEL_TYPE" = "custom-spatial-mllm-lvsm" ]; then
    args="$args \
    --enforce_lvsm ${ENFORCE_LVSM} \
    --lvsm_checkpoint_path ${LVSM_CHECKPOINT_PATH} \
    --nvs_loss_weight ${NVS_LOSS_WEIGHT} \
    --num_target_views ${NUM_TARGET_VIEWS} \
    --lvsm_l2_weight ${LVSM_L2_WEIGHT} \
    --lvsm_perceptual_weight ${LVSM_PERCEPTUAL_WEIGHT} \
    --lvsm_lpips_weight ${LVSM_LPIPS_WEIGHT} \
    --vgg_weight_file ${VGG_WEIGHT_FILE} \
    --tune_mm_connector_lvsm ${TUNE_CONNECTOR_LVSM} \
    --tune_lvsm_decoder ${TUNE_LVSM_DECODER} \
    --nvs_enabled ${NVS_ENABLED} \
    --nvs_img_log_interval ${NVS_IMG_LOG_INTERVAL} \
    --lvsm_adaptor_lr ${LVSM_ADAPTOR_LR} \
    --nvs_target_pool ${NVS_TARGET_POOL} \
    --lvsm2qwen_type ${LVSM2QWEN_TYPE} \
    --llm2lvsm_type ${LLM2LVSM_TYPE} \
    --enable_sdpa_gating ${ENABLE_SDPA_GATING}"
    echo "[Training] LVSM integration enabled: ckpt=${LVSM_CHECKPOINT_PATH}, nvs_weight=${NVS_LOSS_WEIGHT}, lvsm_adaptor_lr=${LVSM_ADAPTOR_LR}"
    echo "[Training] SDPA gating: ${ENABLE_SDPA_GATING}"
fi

# JJ: Add Pose RoPE args if enabled (only for custom-spatial-mllm)
if [ "$USE_POSE_ROPE" = "True" ] || [ "$USE_POSE_ROPE" = "true" ]; then
    args="$args --use_pose_rope --pose_enc_type ${POSE_ENC_TYPE}"
    if [ -n "$MROPE_SECTION" ]; then
        args="$args --mrope_section ${MROPE_SECTION}"
        echo "[Training] 4D Pose RoPE enabled: pose_enc_type=${POSE_ENC_TYPE}, mrope_section=${MROPE_SECTION}"
    else
        echo "[Training] 4D Pose RoPE enabled: pose_enc_type=${POSE_ENC_TYPE} (using default mrope_section)"
    fi
fi

# JJ : Enable wandb reporting for LVSM diagnostics
args="$args --report_to wandb"

# Launch training (DDP with torchrun)
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} 2>&1 | tee -a "${logfile}"
