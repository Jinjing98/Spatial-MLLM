#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1 #2
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --gpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

# test on capella

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
# conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
conda activate /data/horse/ws/jixu233b-3d_ws/envs/transformers_v5
module load CUDA/12.4.0 # nvcc

cd "$(dirname "$0")"
cd ../..
cd "$SLURM_SUBMIT_DIR"

# This avoids NFS slowdowns.
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR

# Print current directory
pwd

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

MODEL_PATH="${MODELS_ROOT}/checkpoints/Spatial-MLLM-v1.1-Instruct-135K"
# MODEL_NAME=$(echo "$MODEL_PATH" | cut -d'/' -f2)
# MODEL_NAME=$(echo "$MODEL_PATH" | cut -d'/' -f9)
MODEL_TYPE="spatial-mllm"
MODEL_NAME_SUFFIX=""

MODEL_TYPE="custom-spatial-mllm"
MODEL_NAME_SUFFIX="adaptedPosID_RoPE"

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
MODEL_TYPE="qwen3-vl"
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
# MODEL_TYPE="spatial-mllm"
# MODEL_TYPE="custom-spatial-mllm"
# MODEL_PATH="Diankun/Spatial-MLLM-v1.1-Instruct-135K"

MODEL_NAME_SUFFIX=""
MODEL_NAME="${MODEL_TYPE}${MODEL_NAME_SUFFIX}"

# nframes=(None)
# nframes=(32)
# nframes=(16)
nframes=(8)

# sample_fps=(None)
# sample_fps=(1)

# QUESTION_TYPES=("${QUESTION_TYPE_LIST[3]}" "${QUESTION_TYPE_LIST[4]}" "${QUESTION_TYPE_LIST[5]}") #ego. 
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[0]}" "${QUESTION_TYPE_LIST[1]}" "${QUESTION_TYPE_LIST[6]}") #allo.

DATASETS=("${DATASET_LIST[@]}") #all datasets
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[6]}") #allo.
# DATASETS=("${DATASET_LIST[1]}") #arkitscenes
# DATASETS=("${DATASET_LIST[2]}") #arkitscenes
# DATASETS=("${DATASET_LIST[0]}") #arkitscenes
QUESTION_TYPES=("${QUESTION_TYPE_LIST[@]}") #all cases
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[6]}") #all cases

SCENE_NAME_LIST=()  # By default, empty array means all scenes will be evaluated
# SCENE_NAME_LIST=("42446103")  # Example: specify particular scenes to evaluate



for nframe in "${nframes[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # JJ
    SAMPLING='sa_sampling'
    MERGEAWARE_DETAILS=''
    # SAMPLING='uniform_sampling'
    # MERGEAWARE_DETAILS=''
    SAMPLING='mergeaware_uniform_sampling'
    MERGEAWARE_DETAILS='_rnd_fidss30'
    # SAMPLING='mergeaware_sa_sampling'
    # MERGEAWARE_DETAILS='_rnd_idxss1'


    OUTPUT_ROOT="${RESULTS_SAVE_ROOT}/results/vsibench_${SAMPLING}"
    mkdir -p "$OUTPUT_ROOT"
    VIDEO_DIR="${DATA_ROOT}/vsibench/${SAMPLING}_${nframe}f${MERGEAWARE_DETAILS}" 

    # Build dataset suffix
    DATASET_SUFFIX=""
    if [ ${#DATASETS[@]} -ne ${#DATASET_LIST[@]} ]; then
        DATASET_SUFFIX="_$(IFS=_; echo "${DATASETS[*]}")"
    fi
    
    # Build question type suffix
    QUESTION_SUFFIX=""
    if [ ${#QUESTION_TYPES[@]} -ne ${#QUESTION_TYPE_LIST[@]} ]; then
        QUESTION_SUFFIX="_$(IFS=_; echo "${QUESTION_TYPES[*]}")"
    fi
    
    EXP_DIR="${OUTPUT_ROOT}/${MODEL_NAME}-${nframe}f${DATASET_SUFFIX}${QUESTION_SUFFIX}"
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
    EXTRA_ARGS=""
    if [ ${#SCENE_NAME_LIST[@]} -gt 0 ]; then
        EXTRA_ARGS="--scene_names ${SCENE_NAME_LIST[@]}"
    fi
    
    python /home/jixu233b/Projects/VLM_3D/SpatialMllmHallucinate/third_party/Spatial-MLLM/src/evaluation/vsibench/eval_vsibench.py \
        --model_path $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --nframes $nframe \
        --annotation_dir "${DATA_ROOT}/vsibench" \
        --question_types ${QUESTION_TYPES[@]} \
        --datasets ${DATASETS[@]} \
        --video_dir "${VIDEO_DIR}" \
        --batch_size 1 \
        --output_dir "$EXP_DIR" \
        --output_name "eval_result" \
        $EXTRA_ARGS \
        2>&1 | tee -a "$LOG_FILE"
        
    echo ">>> Experiment Finished. Results in $EXP_DIR"
done