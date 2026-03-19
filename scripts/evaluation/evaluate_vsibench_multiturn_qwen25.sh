#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

set -euo pipefail

# Global
DATA_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets"
MODELS_ROOT="/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM"
EVAL_RESULTS_BASE="/data/horse/ws/jixu233b-metadata_ws/exps/stats/spatialmllm_results/results"

# activate conda
source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
module load release/24.04
module load CUDA/12.4.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
fi

export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
mkdir -p "$TRITON_CACHE_DIR"

pwd

OUTPUT_ROOT="${EVAL_RESULTS_BASE}/vsibench_multiturn"
mkdir -p "$OUTPUT_ROOT"

# ==================== Editable experiment config ====================
MODEL_TYPE="qwen2.5-vl"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_PATH="${MODELS_ROOT}/checkpoints/your_local_qwen25_checkpoint"

NUM_CONTEXT=2
RANDOM_SEED=0
DEBUG_LEAK_TARGET_CONTEXT="${DEBUG_LEAK_TARGET_CONTEXT:-0}"
CONTEXT_ANSWER_MODE="${CONTEXT_ANSWER_MODE:-letter_only}"
MODEL_NAME_SUFFIX="_multiturn_ctx${NUM_CONTEXT}_seed${RANDOM_SEED}"
MODEL_NAME="${MODEL_TYPE}${MODEL_NAME_SUFFIX}"

DATASET_LIST=(
    "arkitscenes"
    "scannet"
    "scannetpp"
)

QUESTION_TYPE_LIST=(
    "obj_appearance_order"
    "object_abs_distance"
    "object_counting"
    "object_rel_direction_easy"
    "object_rel_direction_hard"
    "object_rel_direction_medium"
    "object_rel_distance"
    "object_size_estimation"
    "room_size_estimation"
    "route_planning"
)

# Examples:
# DATASETS=("${DATASET_LIST[@]}")
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[@]}")
# SCENE_NAME_LIST=()

DATASETS=("${DATASET_LIST[0]}")
QUESTION_TYPES=("${QUESTION_TYPE_LIST[3]}")
SCENE_NAME_LIST=("42446103")

nframes=(16)
# nframes=(8 16 32)

# sample_fps="0.5"
sample_fps=""
# ================================================================

for nframe in "${nframes[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXP_DIR="${OUTPUT_ROOT}/${MODEL_NAME}-${nframe}f"
    LOG_FILE="${EXP_DIR}/run.log"

    mkdir -p "$EXP_DIR"

    echo "----------------------------------------------------------------"
    echo "Starting multi-turn sweep: [nframes=$nframe, num_context=$NUM_CONTEXT, seed=$RANDOM_SEED]"
    echo "Artifacts dir: $EXP_DIR"
    echo "----------------------------------------------------------------"

    {
        echo "================ EXPERIMENT INFO ================"
        echo "Time: $TIMESTAMP"
        echo "Model path: $MODEL_PATH"
        echo "Model type: $MODEL_TYPE"
        echo "Params: NFRAMES=$nframe NUM_CONTEXT=$NUM_CONTEXT RANDOM_SEED=$RANDOM_SEED"
        echo "Debug: LEAK_TARGET_CONTEXT=$DEBUG_LEAK_TARGET_CONTEXT CONTEXT_ANSWER_MODE=$CONTEXT_ANSWER_MODE"
        echo "Datasets: ${DATASETS[*]}"
        echo "Question types: ${QUESTION_TYPES[*]}"
        echo "Scene names: ${SCENE_NAME_LIST[*]:-ALL}"
        echo "Commit: $(git rev-parse HEAD)"
        echo "================================================="
    } > "$LOG_FILE"

    CMD=(
        python src/evaluation/vsibench/eval_vsibench_multiturn.py
        --model_path "$MODEL_PATH"
        --model_type "$MODEL_TYPE"
        --nframes "$nframe"
        --annotation_dir "${DATA_ROOT}/vsibench"
        --question_types "${QUESTION_TYPES[@]}"
        --datasets "${DATASETS[@]}"
        --video_dir "${DATA_ROOT}/vsibench"
        --batch_size 1
        --output_dir "$EXP_DIR"
        --output_name "eval_result"
        --num_context "$NUM_CONTEXT"
        --random_seed "$RANDOM_SEED"
    )

    if [[ -n "$sample_fps" ]]; then
        CMD+=(--sample_fps "$sample_fps")
    fi

    if (( ${#SCENE_NAME_LIST[@]} > 0 )); then
        CMD+=(--scene_names "${SCENE_NAME_LIST[@]}")
    fi

    if [[ "$DEBUG_LEAK_TARGET_CONTEXT" == "1" ]]; then
        CMD+=(--debug_leak_target_context)
    fi

    if [[ "$CONTEXT_ANSWER_MODE" != "letter_only" ]]; then
        CMD+=(--context_answer_mode "$CONTEXT_ANSWER_MODE")
    fi

    printf -v CMD_STR '%q ' "${CMD[@]}"
    echo "Running command: ${CMD_STR% }" | tee -a "$LOG_FILE"

    "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

    echo ">>> Multi-turn experiment finished. Results in $EXP_DIR"
done