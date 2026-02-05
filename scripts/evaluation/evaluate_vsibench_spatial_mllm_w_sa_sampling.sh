#!/bin/bash
# Global
DATA_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets"
MODELS_ROOT="/data/horse/ws/jixu233b-metadata_ws/models/Spatial-MLLM"

# tso
DATA_ROOT="/mnt/nct-zfs/TCO-All/SharedDatasets"
MODELS_ROOT="/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM"

cd "$(dirname "$0")"
cd ../..

# This avoids NFS slowdowns.
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR

# Print current directory
pwd
OUTPUT_ROOT="results/vsibench-sa-sampling"
mkdir -p "$OUTPUT_ROOT"


MODEL_PATH="${MODELS_ROOT}/checkpoints/Spatial-MLLM-v1.1-Instruct-135K"
MODEL_NAME=$(echo "$MODEL_PATH" | cut -d'/' -f2)
MODEL_TYPE="spatial-mllm"
MODEL_TYPE="custom-spatial-mllm"


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
)
SCENE_NAME_LIST=()  # By default, empty array means all scenes will be evaluated
# SCENE_NAME_LIST=("42446103")  # By default, empty array means all scenes will be evaluated

# QUESTION_TYPES=("${QUESTION_TYPE_LIST[3]}" "${QUESTION_TYPE_LIST[4]}" "${QUESTION_TYPE_LIST[5]}") #ego. 
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[0]}" "${QUESTION_TYPE_LIST[1]}" "${QUESTION_TYPE_LIST[6]}") #allo.

DATASETS=("${DATASET_LIST[0]}") #arkitscenes
# DATASETS=("${DATASET_LIST[@]}") #all datasets
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[8]}") #semantic
QUESTION_TYPES=("${QUESTION_TYPE_LIST[@]}") #all cases
# QUESTION_TYPES=("${QUESTION_TYPE_LIST[6]}") #allo.

nframes=(16)

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
    python src/evaluation/vsibench/eval_vsibench.py \
        --model_path $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --nframes $nframe \
        --annotation_dir "${DATA_ROOT}/vsibench" \
        --question_types ${QUESTION_TYPES[@]} \
        --datasets ${DATASETS[@]} \
        --video_dir "${DATA_ROOT}/vsibench/sa_sampling_16f" \
        --batch_size 1 \
        --output_dir "$EXP_DIR" \
        --output_name "eval_result" \
        ${SCENE_NAME_LIST[@]:+--scene_names ${SCENE_NAME_LIST[@]}} \
        2>&1 | tee -a "$LOG_FILE"
        
    echo ">>> Experiment [$tag] Finished. Results in $EXP_DIR"
done