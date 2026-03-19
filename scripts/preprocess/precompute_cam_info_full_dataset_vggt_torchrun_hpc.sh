#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --time=80:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

set -euo pipefail

source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
module load release/24.04
module load CUDA/12.4.0
export DS_BUILD_OPS=0  # suppress DeepSpeed nvcc probe (no compiled ops needed for inference)

REPO_ROOT="/home/jixu233b/Projects/VLM_3D/SpatialMllmHallucinate/third_party/Spatial-MLLM"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# jj: Core dataset/output settings (override via sbatch --export or env before sbatch).
export DATASET_ROOT="${DATASET_ROOT:-/data/horse/ws/jixu233b-metadata_ws/datasets/spmllm}"
# export DATASETS="${DATASETS:-spatial_mllm_mix_133k%1,route_plan_scannet_2k%1}"
export DATASETS="${DATASETS:-spatial_mllm_mix_133k,route_plan_scannet_2k}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data/horse/ws/jixu233b-metadata_ws/datasets/spmllm/precompute_pose_vggt_16_pa}"

# jj: Model / sampling defaults aligned with training script.
export VGGT_CHECKPOINTS_PATH="${VGGT_CHECKPOINTS_PATH:-/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM/checkpoints/VGGT-1B/model.safetensors}"
export PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
export VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-16}"
export VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-16}"
export VIDEO_FRAME_FPS="${VIDEO_FRAME_FPS:-4}"
export SAMPLING_ENFORCE_REAL_NEIGHBOUR="${SAMPLING_ENFORCE_REAL_NEIGHBOUR:-True}"
export NEIGHBOUR_MODE="${NEIGHBOUR_MODE:-after}"
export NEIGHBOUR_MAX_STEP="${NEIGHBOUR_MAX_STEP:-1}"
export MAX_PIXELS="${MAX_PIXELS:-324576}"
export MIN_PIXELS="${MIN_PIXELS:-293216}"
export VIDEO_MAX_FRAME_PIXELS="${VIDEO_MAX_FRAME_PIXELS:-324576}"
export VIDEO_MIN_FRAME_PIXELS="${VIDEO_MIN_FRAME_PIXELS:-293216}"
export SEED="${SEED:-42}"
export LIMIT="${LIMIT:-0}"
export SKIP_EXISTING="${SKIP_EXISTING:-True}"

# jj: Parallel workers = number of GPUs requested by this job.
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
# NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"
echo "[INFO] DATASETS=${DATASETS}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"

# jj: Each torchrun worker gets one shard and one physical GPU.
# We use --no_python + bash to set CUDA_VISIBLE_DEVICES per worker.
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --no_python bash -lc '
  set -euo pipefail
  export SHARD_ID=${LOCAL_RANK}
  export NUM_SHARDS=${WORLD_SIZE}
  export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
  export DEVICE=cuda

  echo "[WORKER] rank=${LOCAL_RANK} shard=${SHARD_ID}/${NUM_SHARDS} cuda_visible=${CUDA_VISIBLE_DEVICES}"
  bash scripts/preprocess/precompute_cam_info_full_dataset_vggt.sh
'

echo "[DONE] Full-dataset precompute finished. OUTPUT_ROOT=${OUTPUT_ROOT}"
