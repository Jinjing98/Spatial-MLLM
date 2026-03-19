#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 #2
#SBATCH --gpus-per-task=1 #2
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

# ============ GPU / Batch scaling — adjust these together ============
# JJ : effective_bs = N_GPU × BATCH_SIZE × GRAD_ACCUM_STEPS
#   2 GPU:  2 × 1 × 8 = 16   (--gres=gpu:2, --gpus-per-task=2, --mem=80G)
#   4 GPU:  4 × 1 × 4 = 16   (--gres=gpu:4, --gpus-per-task=4, --mem=160G)
#   8 GPU:  8 × 1 × 2 = 16   (--gres=gpu:8, --gpus-per-task=8, --mem=320G)
# When changing N_GPU: update SBATCH gpu/mem lines AND NPROC+GRAD_ACCUM below.
N_GPU=1
BATCH_SIZE=1
GRAD_ACCUM_STEPS=$((16 / N_GPU / BATCH_SIZE))  # auto-scale to keep effective_bs=16

set -euo pipefail

source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
module load release/24.04
module load CUDA/12.4.0
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Set environment variables
export WANDB_PROJECT="Spatial-MLLM-SFT"
WANDB_PROJECT_OVERFIT="Spatial-MLLM-Overfit"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# jj: Enable per-stage time diagnostics for LVSM path (set 0 to disable quickly).
export SPMLLM_LVSM_STAGE_TIME=1
export SPMLLM_LVSM_STAGE_TIME_INTERVAL=1

PRETRAINED_CKPT_ROOT="/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM/"

# ============ Dataset ============
DATASET_ROOT="/home/jixu233b/Projects/VLM_3D/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/SPMLLM-DATA"
DATASETS="spatial_mllm_mix_133k,route_plan_scannet_2k" # default "spatial_mllm_mix_133k,route_plan_scannet_2k"
DATASETS="spatial_mllm_mix_133k%1,route_plan_scannet_2k%1" # default "spatial_mllm_mix_133k,route_plan_scannet_2k"
# DATASET_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets/SQA3D"
# DATASETS="sqa3d_filtered_40k"
# DATASET_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets/ViCA-322K"
# DATASETS="vica_322k_base%50"
export DATASET_ROOT

# ============ Training basics ============
# JJ Freq Edit
OUTPUT_ROOT="/data/horse/ws/jixu233b-metadata_ws/exps/train/spatialmllm"
TRAIN_EPOCHS=1 # default 1
WARMUP_RATIO=0.03 # default 0.03
SAVE_STEPS=0.0625 # fraction of total steps between checkpoints (e.g. 0.125 -> ~8 saves)
SAVE_STRATEGY="steps" # default save strategy
SAVE_STEPS_ARG="--save_steps ${SAVE_STEPS}"
NUM_WORKERS=2 # default 8
NPROC_PER_NODE=${N_GPU}
VIDEO_MAX_FRAMES=16 # default 16
VIDEO_MIN_FRAMES=16 # default 16
VIDEO_FRAME_FPS=4 # default 4
# JJ : Temporal-merge-aware real-neighbour sampling
# Sample N/2 anchors uniformly, then add a real temporal neighbour for each anchor.
# SAMPLING_ENFORCE_REAL_NEIGHBOUR=False  # set True to enable
SAMPLING_ENFORCE_REAL_NEIGHBOUR=True  # set True to enable
# NEIGHBOUR_MODE="random"                 # before | after | random
# NEIGHBOUR_MAX_STEP=3                   # max frame-ID offset; actual step ~ randint(1, max)
# Update on 2024-10-04: 
NEIGHBOUR_MODE="after"                 # before | after | random
NEIGHBOUR_MAX_STEP=1                   # max frame-ID offset; actual step ~ randint(1, max)

GRADIENT_CHECKPOINTING=True # default False
# JJ : Model type — "custom-spatial-mllm-lvsm" for LVSM integration, "custom-spatial-mllm" for base
MODEL_TYPE="custom-spatial-mllm-lvsm"
PRETRAINED_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# RUN_NAME_APPENDIX="_8sa_knowview_nvsloss_decodefromllmMid"

# RUN_NAME_APPENDIX="_8sa_knowview_nvsloss_decodefromllm"
# RUN_NAME_APPENDIX="_8sa_knowview_nvsloss_decodefromrandom"
# RUN_NAME_APPENDIX="_8sa_knowview_nvsloss_decodefromclip"
# RUN_NAME_APPENDIX="_8sa_knowview_nvsloss_decodefromvggt"
RUN_NAME_APPENDIX="_8sa_knowview_nvsloss_celoss_lre-4_decodefromllm"

#tmux1 3173410  _8sa_knowview_nvsloss_decodefromrandom
#tmux0 3173413  _8sa_knowview_nvsloss_decodefromllm
#tmux2 3173413  _8sa_knowview_nvsloss_decodefromclip
#tmux4 3173414  _8sa_knowview_nvsloss_decodefromvggt

# RUN_NAME_APPENDIX="_8sa_knowview_nvsloss_decodefromclip_2contextview_2decoderview"
# RUN_NAME_APPENDIX="_8sa_knowview_nvsloss_decodefromvggt_2contextview_2decoderview"

# RUN_NAME_APPENDIX="_4x4_hpc_spmllm_enforceRealNbr"

# put the fix reagrding connector n spatial reasoner here.

# ============ Overfitting experiment switch (minimal and explicit) ============
# OVERFIT_MODE=false
OVERFIT_MODE=true
OVERFIT_MAX_TRAIN_SAMPLES=8
MAX_TRAIN_SAMPLES_ARG=""
LOGGING_STEPS=1
LOGGING_STEPS_OVERFIT=1
NVS_IMG_LOG_INTERVAL_OVERFIT=5
# wo warming up & constant high lr for of
LR_SCHEDULER_TYPE_OVERFIT="constant_with_warmup" # default "cosine" #
LR_OVERFIT=1e-5
WARMUP_RATIO_OVERFIT=0.0
EPOCHS_OVERFIT=10000

# ============ What to train (freeze / unfreeze) ============
# JJ : Base model components
TUNE_VISION=False                # ViT visual encoder
TUNE_SPATIAL_ENCODER=False       # VGGT spatial encoder
TUNE_CONNECTOR=False             # MLPAddConnector (visual.merger + connector)
TUNE_LLM=False                   # Qwen LLM backbone + lm_head
# JJ : LVSM-specific (only when MODEL_TYPE=custom-spatial-mllm-lvsm)
TUNE_CONNECTOR_LVSM=True        # connector_lvsm (FiLM adaptor)
TUNE_LVSM_DECODER=True          # lvsm_model transformer_blocks + image_token_decoder

# ============ Learning rates ============
LR_SCHEDULER_TYPE="cosine" # default "cosine" #constant_with_warmup
lr=7e-6                          # base lr (LLM, spatial_encoder, lvsm_model if unfrozen)
mm_projector_lr=2e-5             # visual.merger + connector (MLPAddConnector)
# JJ : Separate lr for LVSM adaptor (connector_lvsm only; lvsm_model uses base lr)
# Original LVSM: lr=4e-4, BS=64. Linear scaling: 4e-4 * (16/64) = 1e-4
# connector_lvsm is randomly initialised (zero-init gates, Xavier proj),
# so it needs a normal learning rate to train — NOT the tiny fine-tuning rate.
# Range: 1e-5 (conservative) ~ 5e-5 (aggressive). Match mm_projector_lr as baseline.
LVSM_ADAPTOR_LR=2e-5
weight_decay=0.1
max_grad_norm=1.0

# ============ LVSM integration config (only for custom-spatial-mllm-lvsm) ============
ENFORCE_LVSM=True
# ENFORCE_LVSM=False
LVSM_CHECKPOINT_PATH="/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM/checkpoints/scene_decoder_only_256.pt"
LVSM_IMAGE_SIZE=256
# JJ : NVS_LOSS_WEIGHT tuning guide:
#   CE loss ~ 1-5, raw NVS loss (L2+Percep) ~ 5-50 initially
#   1.0 → NaN (NVS dominates, gradient explosion)
#   0.01 → safe start (NVS contributes ~0.05-0.5 vs CE ~2-5)
#   0.1  → moderate (try after stable training at 0.01)
# JJ : With residual context architecture, NVS starts stable (gate=0).
# 0.01 is a safe starting point.
NVS_LOSS_WEIGHT=0.1
NUM_TARGET_VIEWS=4 # actual num of views for loss. We awlays try to provide actual NVS pool with n_in num.
# NUM_TARGET_VIEWS=2 # actual num of views for loss. We awlays try to provide actual NVS pool with n_in num.
LVSM_L2_WEIGHT=1.0
LVSM_PERCEPTUAL_WEIGHT=0.5 # can be costly
LVSM_LPIPS_WEIGHT=0.0
VGG_WEIGHT_FILE="/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM/checkpoints/imagenet-vgg-verydeep-19.mat"
NVS_ENABLED=True         # JJ : Load novel target frames for NVS loss

# jj: Precomputed pose package switch (script-controlled, not overridden by external env).
USE_PRE_COMPUTE_POSE=True
# USE_PRE_COMPUTE_POSE=False
PRECOMPUTE_POSE_SOURCE=vggt
PRECOMPUTE_POSE_ROOT=/home/jixu233b/Projects/VLM_3D/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/SPMLLM-DATA/precompute_pose_vggt_16_pa
# export DEBUG_COMPARE_PRECOMPUTED_POSE=1
export DEBUG_COMPARE_PRECOMPUTED_POSE=0 #recompute 

# JJ : NVS target pool — 'nvs' (novel only), 'input' (reconstruction sanity check), 'all' (mixed)
NVS_TARGET_POOL="input"
# NVS_TARGET_POOL="input"
# NVS_IMG_LOG_INTERVAL=135  # JJ : Log rendered vs GT images to wandb every N steps (0=disable)
NVS_IMG_LOG_INTERVAL=20  # JJ : Log rendered vs GT images to wandb every N steps (0=disable)
# JJ : Adapter types for LVSM ↔ QwenVL bridges (currently only "linear")
LVSM2QWEN_TYPE="linear"
LLM2LVSM_TYPE="linear"
# JJ : Phase-5 adaptation strategy: only 'patch_residual' is supported
VLM2CONTEXT_ADAPT_STRATEGY="patch_residual"
# JJ : SDPA output gating (arxiv 2505.06708) — element-wise sigmoid gate after attention
ENABLE_SDPA_GATING=False
# ENABLE_SDPA_GATING=True

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

if [ "${OVERFIT_MODE}" = "true" ] || [ "${OVERFIT_MODE}" = "True" ]; then
    export WANDB_PROJECT="${WANDB_PROJECT_OVERFIT}"
    SAVE_STRATEGY="no"
    SAVE_STEPS_ARG=""
    MAX_TRAIN_SAMPLES_ARG="--max_train_samples ${OVERFIT_MAX_TRAIN_SAMPLES}"
    LOGGING_STEPS=${LOGGING_STEPS_OVERFIT}
    NVS_IMG_LOG_INTERVAL=${NVS_IMG_LOG_INTERVAL_OVERFIT}
    LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE_OVERFIT}
    WARMUP_RATIO=${WARMUP_RATIO_OVERFIT}
    TRAIN_EPOCHS=${EPOCHS_OVERFIT}
    lr=${LR_OVERFIT}
    RUN_NAME_APPENDIX="${RUN_NAME_APPENDIX}_overfit20"
    echo "[Overfit] Enabled: max_train_samples=${OVERFIT_MAX_TRAIN_SAMPLES}, save_strategy=${SAVE_STRATEGY}, logging_steps=${LOGGING_STEPS}, nvs_img_log_interval=${NVS_IMG_LOG_INTERVAL}, wandb_project=${WANDB_PROJECT}"
fi

run_name="${timestamp}_${base_run_name}${RUN_NAME_APPENDIX}"
output_dir=${OUTPUT_ROOT}/${run_name}
mkdir -p ${output_dir}
logfile="${output_dir}/$(date +'%Y%m%d_%H%M%S')_train.log"

# --save_total_limit 2 \

# Training arguments
args="
    --model_type ${MODEL_TYPE} \
    --vggt_checkpoints_path ${vggt_checkpoints_path} \
    --spatial_embeds_layer_idx ${spatial_embeds_layer_idx} \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
    --dataset_use ${DATASETS} \
    ${MAX_TRAIN_SAMPLES_ARG} \
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
    --save_strategy ${SAVE_STRATEGY} \
    ${SAVE_STEPS_ARG} \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${WARMUP_RATIO} \
    --max_grad_norm ${max_grad_norm} \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --logging_steps ${LOGGING_STEPS} \
    --model_max_length 8192 \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --dataloader_num_workers ${NUM_WORKERS} \
    --run_name ${run_name} \
    --ddp_find_unused_parameters True"

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
    --use_pre_compute_pose ${USE_PRE_COMPUTE_POSE} \
    --precompute_pose_source ${PRECOMPUTE_POSE_SOURCE} \
    --precompute_pose_root ${PRECOMPUTE_POSE_ROOT} \
    --nvs_img_log_interval ${NVS_IMG_LOG_INTERVAL} \
    --lvsm_adaptor_lr ${LVSM_ADAPTOR_LR} \
    --nvs_target_pool ${NVS_TARGET_POOL} \
    --lvsm2qwen_type ${LVSM2QWEN_TYPE} \
    --llm2lvsm_type ${LLM2LVSM_TYPE} \
    --vlm2context_adapt_strategy ${VLM2CONTEXT_ADAPT_STRATEGY} \
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

# jj: Ensure torchrun sets rank env fresh; avoid accidental stale shell exports.
# unset RANK WORLD_SIZE LOCAL_RANK

python ${entry_file} ${args} 2>&1 | tee -a "${logfile}"
# # # Launch training (DDP with torchrun)
# torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} --no_python bash -lc '
#     set -euo pipefail
#     python "'"${entry_file}"'" '"${args}"' --local_rank ${LOCAL_RANK}
# ' 2>&1 | tee -a "${logfile}"
                        
