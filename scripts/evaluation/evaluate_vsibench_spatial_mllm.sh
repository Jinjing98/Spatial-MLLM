#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1 #2
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

# Global
DATA_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets"
MODELS_ROOT="/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM"
RESULTS_SAVE_ROOT="/home/jixu233b/Projects/VLM_3D/SpatialMllmHallucinate/third_party/Spatial-MLLM"

# # tso
# DATA_ROOT="/mnt/nct-zfs/TCO-All/SharedDatasets"
# MODELS_ROOT="/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM"
# RESULTS_SAVE_ROOT="/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM"


# activate conda
source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
# conda activate /data/horse/ws/jixu233b-3d_ws/envs/transformers_v5
module load CUDA/12.4.0 # nvcc

cd "$(dirname "$0")"
cd ../..
cd "$SLURM_SUBMIT_DIR"


# This avoids NFS slowdowns.
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR

# Print current directory
pwd

OUTPUT_ROOT="${RESULTS_SAVE_ROOT}/results/vsibench"
mkdir -p "$OUTPUT_ROOT"

MODEL_PATH="${MODELS_ROOT}/checkpoints/Spatial-MLLM-v1.1-Instruct-135K"
MODEL_TYPE="spatial-mllm"
# MODEL_TYPE="custom-spatial-mllm"
# MODEL_TYPE="qwen2.5-vl"
# MODEL_PATH='Qwen/Qwen2.5-VL-3B-Instruct'
# MODEL_TYPE="qwen2.5-vl"
# MODEL_PATH='Qwen/Qwen2.5-VL-3B-Instruct'
# MODEL_TYPE="spatial-mllm"
# MODEL_TYPE="custom-spatial-mllm"
# JJ: Fixed default values (not overridable by env vars)
# MODEL_TYPE="qwen2.5-vl"
# MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# MODEL_TYPE="qwen3-vl"
# MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"

# MODEL_TYPE="spatial-mllm"

MODEL_TYPE="custom-spatial-mllm"
MODEL_PATH="Diankun/Spatial-MLLM-v1.1-Instruct-135K"
MODEL_NAME=$(echo "$MODEL_PATH" | cut -d'/' -f2)




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
    "route_planning" # missing in previous all
)
SCENE_NAME_LIST=()  # By default, empty array means all scenes will be evaluated
# SCENE_NAME_LIST=("42446103")  # By default, empty array means all scenes will be evaluated

# QUESTION_TYPES=("${QUESTION_TYPE_LIST[3]}" "${QUESTION_TYPE_LIST[4]}" "${QUESTION_TYPE_LIST[5]}") #ego. 
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[0]}" "${QUESTION_TYPE_LIST[1]}" "${QUESTION_TYPE_LIST[6]}") #allo.

# DATASETS=("${DATASET_LIST[0]}") #arkitscenes
DATASETS=("${DATASET_LIST[@]}") #all datasets
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[8]}") #semantic
QUESTION_TYPES=("${QUESTION_TYPE_LIST[@]}") #all cases
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[6]}") #allo.

# nframes=(None)
# nframes=(8)
nframes=(16)
# nframes=(32)
# sample_fps=(None)
# sample_fps=(1)

# JJ : Parse CLI args for skipping eval / metric phases
# Usage: bash script.sh [--skip_eval] [--skip_metric]
EXTRA_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --skip_eval)  EXTRA_ARGS+=" --skip_eval" ;;
        --skip_metric) EXTRA_ARGS+=" --skip_metric" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

for nframe in "${nframes[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXP_DIR="${OUTPUT_ROOT}/${MODEL_NAME}-${nframe}f"
    LOG_FILE="${EXP_DIR}/run.log"

    mkdir -p "$EXP_DIR"
    
    echo "----------------------------------------------------------------"
    echo "Starting Sweep: [nframes=$nframe]"
    echo "Artifacts dir: $EXP_DIR"
    echo "----------------------------------------------------------------"

    {
        echo "================ EXPERIMENT INFO ================"
        echo "Time: $TIMESTAMP"
        echo "Params: NFRAMES=$nframe"
        echo "Commit: $(git rev-parse HEAD)"
        echo "================================================="
    } > "$LOG_FILE"

    # --- run experiment ---
    # python src/evaluation/vsibench/eval_vsibench.py \
    python src/evaluation/vsibench/eval_vsibench.py \
        --model_path $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --nframes $nframe \
        --annotation_dir "${DATA_ROOT}/vsibench" \
        --question_types ${QUESTION_TYPES[@]} \
        --datasets ${DATASETS[@]} \
        --video_dir "${DATA_ROOT}/vsibench" \
        --batch_size 1 \
        --output_dir "$EXP_DIR" \
        --output_name "eval_result" \
        ${SCENE_NAME_LIST[@]:+--scene_names ${SCENE_NAME_LIST[@]}} \
        $EXTRA_ARGS \
        2>&1 | tee -a "$LOG_FILE"
        
        # --sample_fps 0.01 \

    echo ">>> Experiment Finished. Results in $EXP_DIR"
done