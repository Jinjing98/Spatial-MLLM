#!/bin/bash
set -euo pipefail

# jj: Ensure script always runs from repo root so `src/...` imports work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# jj: Dataset root can point to your official training data package.
DATASET_ROOT="${DATASET_ROOT:-/data/horse/ws/jixu233b-metadata_ws/datasets/spmllm}"
DATASETS="${DATASETS:-spatial_mllm_mix_133k%20,route_plan_scannet_2k%20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/horse/ws/jixu233b-metadata_ws/datasets/spmllm/precompute_pose_vggt_16_pa}"

PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
VGGT_CHECKPOINTS_PATH="${VGGT_CHECKPOINTS_PATH:-/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM/checkpoints/VGGT-1B/model.safetensors}"

VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-16}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-16}"
VIDEO_FRAME_FPS="${VIDEO_FRAME_FPS:-4}"
SAMPLING_ENFORCE_REAL_NEIGHBOUR="${SAMPLING_ENFORCE_REAL_NEIGHBOUR:-True}"
NEIGHBOUR_MODE="${NEIGHBOUR_MODE:-after}"
NEIGHBOUR_MAX_STEP="${NEIGHBOUR_MAX_STEP:-1}"

MAX_PIXELS="${MAX_PIXELS:-324576}"
MIN_PIXELS="${MIN_PIXELS:-293216}"
VIDEO_MAX_FRAME_PIXELS="${VIDEO_MAX_FRAME_PIXELS:-324576}"
VIDEO_MIN_FRAME_PIXELS="${VIDEO_MIN_FRAME_PIXELS:-293216}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-True}"

mkdir -p "${OUTPUT_ROOT}"
TS="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${OUTPUT_ROOT}/run_shard$(printf '%03d' "${SHARD_ID}")_of_$(printf '%03d' "${NUM_SHARDS}")_${TS}.log"

export DATASET_ROOT

echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"
echo "[INFO] DATASETS=${DATASETS}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] SHARD=${SHARD_ID}/${NUM_SHARDS}"
echo "[INFO] LOG_FILE=${LOG_FILE}"

python scripts/preprocess/precompute_cam_info_dataset_vggt.py \
  --dataset_use "${DATASETS}" \
  --output_root "${OUTPUT_ROOT}" \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
  --vggt_checkpoints_path "${VGGT_CHECKPOINTS_PATH}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --video_min_frames "${VIDEO_MIN_FRAMES}" \
  --video_max_frames "${VIDEO_MAX_FRAMES}" \
  --video_frame_fps "${VIDEO_FRAME_FPS}" \
  --sampling_enforce_real_neighbour "${SAMPLING_ENFORCE_REAL_NEIGHBOUR}" \
  --neighbour_mode "${NEIGHBOUR_MODE}" \
  --neighbour_max_step "${NEIGHBOUR_MAX_STEP}" \
  --max_pixels "${MAX_PIXELS}" \
  --min_pixels "${MIN_PIXELS}" \
  --video_max_frame_pixels "${VIDEO_MAX_FRAME_PIXELS}" \
  --video_min_frame_pixels "${VIDEO_MIN_FRAME_PIXELS}" \
  --limit "${LIMIT}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_id "${SHARD_ID}" \
  --skip_existing "${SKIP_EXISTING}" \
  2>&1 | tee "${LOG_FILE}"

echo "[DONE] Finished. See ${LOG_FILE}"
