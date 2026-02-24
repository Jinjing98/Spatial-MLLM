#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

# ============================================================================
# FPS/Efficient Pose-Aware Sampling Script for Videos
# ============================================================================
# This script runs FPS or Efficient sampling on video files using
# pre-computed VGGT predictions (from sa_sampling.py output)
# ============================================================================

# Global paths
DATA_ROOT="/mnt/nct-zfs/TCO-All/SharedDatasets"
MODELS_ROOT="/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/"
RESULTS_SAVE_ROOT="/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets"

# This avoids NFS slowdowns.
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR

# Print current directory
pwd

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR="${RESULTS_SAVE_ROOT}/vsibench"

# Number of frames to sample
NUM_FRAMES="${NUM_FRAMES:-16}"
# NUM_FRAMES="${NUM_FRAMES:-8}"

# Sampling type: "fps", "efficient", "mergeaware_fps", or "mergeaware_efficient"
SAMPLING_TYPE="${SAMPLING_TYPE:-mergeaware_fps}"
# SAMPLING_TYPE="${SAMPLING_TYPE:-mergeaware_efficient}"

# Predictions root (must exist, from sa_sampling.py output)
# Example: sa_sampling_16f, sa_sampling_128f, etc.
# Can be overridden with environment variable PREDICTIONS_ROOT
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${BASE_DIR}/sa_sampling_16f_single_video}"

# ============================================================================
# Temporal Merge Aware Sampling Parameters
# ============================================================================
# JJ : For mergeaware_fps and mergeaware_efficient modes
# neighbor_mode: "before", "after" (default), or "random"
NEIGHBOR_MODE="${NEIGHBOR_MODE:-random}"
# index_step_size: Index step size for mergeaware modes (operates on 128-frame pool indices, default: 1)
INDEX_STEP_SIZE="${INDEX_STEP_SIZE:-1}"
# enforce_duplicate: Set to "--enforce_duplicate" to enable simple duplication (f1,f1,f2,f2,...)
# ENFORCE_DUPLICATE="${ENFORCE_DUPLICATE:---enforce_duplicate}"  # 启用
ENFORCE_DUPLICATE="${ENFORCE_DUPLICATE:-}"  # 禁用（默认）

# ============================================================================
# FPS Sampling Parameters
# ============================================================================
FPS_DISTANCE_MODE="${FPS_DISTANCE_MODE:-max_norm}"  # Options: l2, max_norm, fro, geodesic
FPS_STARTING_MODE="${FPS_STARTING_MODE:-medoid}"    # Options: medoid, random, first

# ============================================================================
# Efficient Sampling Parameters
# ============================================================================
EFFICIENT_SAMPLING_MODE="${EFFICIENT_SAMPLING_MODE:-hybrid}"     # Options: grid, hybrid, fps2d
EFFICIENT_NORMALIZATION="${EFFICIENT_NORMALIZATION:-std_norm}"    # Options: minmax, standard
EFFICIENT_DIAGONAL_PRIORITY="${EFFICIENT_DIAGONAL_PRIORITY:-0.0}"
EFFICIENT_STARTING_MODE="${EFFICIENT_STARTING_MODE:-medoid}"    # Options: medoid, random, first

# ============================================================================
# Visualization Options
# ============================================================================
# VISUALIZE_SAMPLING="${VISUALIZE_SAMPLING:-}"        # Set to "--visualize_sampling" to enable
VISUALIZE_SAMPLING="--visualize_sampling"        # Set to "--visualize_sampling" to enable
# PLOT_POSE_ANALYSIS="${PLOT_POSE_ANALYSIS:-}"       # Set to "--plot_pose_analysis" to enable
PLOT_POSE_ANALYSIS="--plot_pose_analysis"       # Set to "--plot_pose_analysis" to enable

# ============================================================================
# Dry Run Mode
# ============================================================================
DRY_RUN="${DRY_RUN:-}"                              # Set to "--dry_run" to enable

# ============================================================================
# Test Mode (single video)
# ============================================================================
# Set VIDEO_PATH to test on a single video
VIDEO_PATH="/mnt/nct-zfs/TCO-All/SharedDatasets/vsibench/arkitscenes/42446103.mp4"
# VIDEO_PATH="${VIDEO_PATH:-}"

# ============================================================================
echo "============================================"
echo "Pose-Aware Sampling: ${SAMPLING_TYPE^^}"
echo "============================================"
echo "Strategy:       ${SAMPLING_TYPE}"
echo "Num frames:     ${NUM_FRAMES}"
echo "Predictions:    ${PREDICTIONS_ROOT}"
if [[ "$SAMPLING_TYPE" == "fps" ]]; then
    echo "Distance mode:  ${FPS_DISTANCE_MODE}"
    echo "Starting mode:  ${FPS_STARTING_MODE}"
elif [[ "$SAMPLING_TYPE" == "efficient" ]]; then
    echo "Sampling mode:  ${EFFICIENT_SAMPLING_MODE}"
    echo "Normalization:  ${EFFICIENT_NORMALIZATION}"
    echo "Diagonal prio:  ${EFFICIENT_DIAGONAL_PRIORITY}"
    echo "Starting mode:  ${EFFICIENT_STARTING_MODE}"
elif [[ "$SAMPLING_TYPE" == "mergeaware_fps" ]]; then
    echo "Distance mode:  ${FPS_DISTANCE_MODE}"
    echo "Starting mode:  ${FPS_STARTING_MODE}"
    echo "Neighbor mode:  ${NEIGHBOR_MODE}"
    echo "Index step:     ${INDEX_STEP_SIZE}"
    echo "Enforce dup:    ${ENFORCE_DUPLICATE:-disabled}"
elif [[ "$SAMPLING_TYPE" == "mergeaware_efficient" ]]; then
    echo "Sampling mode:  ${EFFICIENT_SAMPLING_MODE}"
    echo "Normalization:  ${EFFICIENT_NORMALIZATION}"
    echo "Diagonal prio:  ${EFFICIENT_DIAGONAL_PRIORITY}"
    echo "Starting mode:  ${EFFICIENT_STARTING_MODE}"
    echo "Neighbor mode:  ${NEIGHBOR_MODE}"
    echo "Index step:     ${INDEX_STEP_SIZE}"
    echo "Enforce dup:    ${ENFORCE_DUPLICATE:-disabled}"
fi
echo "Visualization:  ${VISUALIZE_SAMPLING:-disabled}"
echo "Dry run:        ${DRY_RUN:-disabled}"
echo "============================================"

run_sampling() {
    dataset=$1
    
    echo ""
    echo "Processing ${SAMPLING_TYPE} sampling for ${dataset}..."
    
    # Direct execution with if/else (clearer than eval CMD)
    if [[ -n "$VIDEO_PATH" ]]; then
        # Single video mode
        echo "Single video mode: processing $VIDEO_PATH"
        
        if [[ "$SAMPLING_TYPE" == "fps" ]]; then
            python src/sampling/pa_sampling.py \
                --video_path "$VIDEO_PATH" \
                --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
                --output_folder "${BASE_DIR}/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f/single_video" \
                --num_frames $NUM_FRAMES \
                --sampling_type "$SAMPLING_TYPE" \
                --predictions_root "${PREDICTIONS_ROOT}" \
                --fps_distance_mode "$FPS_DISTANCE_MODE" \
                --fps_starting_mode "$FPS_STARTING_MODE" \
                $VISUALIZE_SAMPLING \
                $PLOT_POSE_ANALYSIS \
                $DRY_RUN
        elif [[ "$SAMPLING_TYPE" == "efficient" ]]; then
            python src/sampling/pa_sampling.py \
                --video_path "$VIDEO_PATH" \
                --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
                --output_folder "${BASE_DIR}/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f/single_video" \
                --num_frames $NUM_FRAMES \
                --sampling_type "$SAMPLING_TYPE" \
                --predictions_root "${PREDICTIONS_ROOT}" \
                --efficient_sampling_mode "$EFFICIENT_SAMPLING_MODE" \
                --efficient_normalization "$EFFICIENT_NORMALIZATION" \
                --efficient_diagonal_priority $EFFICIENT_DIAGONAL_PRIORITY \
                --efficient_starting_mode "$EFFICIENT_STARTING_MODE" \
                $VISUALIZE_SAMPLING \
                $PLOT_POSE_ANALYSIS \
                $DRY_RUN
        elif [[ "$SAMPLING_TYPE" == "mergeaware_fps" ]]; then
            python src/sampling/pa_sampling.py \
                --video_path "$VIDEO_PATH" \
                --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
                --output_folder "${BASE_DIR}/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f_single_video/${dataset}" \
                --num_frames $NUM_FRAMES \
                --sampling_type "$SAMPLING_TYPE" \
                --predictions_root "${PREDICTIONS_ROOT}" \
                --fps_distance_mode "$FPS_DISTANCE_MODE" \
                --fps_starting_mode "$FPS_STARTING_MODE" \
                --neighbor_mode "$NEIGHBOR_MODE" \
                --index_step_size $INDEX_STEP_SIZE \
                $ENFORCE_DUPLICATE \
                $VISUALIZE_SAMPLING \
                $PLOT_POSE_ANALYSIS \
                $DRY_RUN
        elif [[ "$SAMPLING_TYPE" == "mergeaware_efficient" ]]; then
            python src/sampling/pa_sampling.py \
                --video_path "$VIDEO_PATH" \
                --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
                --output_folder "${BASE_DIR}/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f_single_video/${dataset}" \
                --num_frames $NUM_FRAMES \
                --sampling_type "$SAMPLING_TYPE" \
                --predictions_root "${PREDICTIONS_ROOT}" \
                --efficient_sampling_mode "$EFFICIENT_SAMPLING_MODE" \
                --efficient_normalization "$EFFICIENT_NORMALIZATION" \
                --efficient_diagonal_priority $EFFICIENT_DIAGONAL_PRIORITY \
                --efficient_starting_mode "$EFFICIENT_STARTING_MODE" \
                --neighbor_mode "$NEIGHBOR_MODE" \
                --index_step_size $INDEX_STEP_SIZE \
                $ENFORCE_DUPLICATE \
                $VISUALIZE_SAMPLING \
                $PLOT_POSE_ANALYSIS \
                $DRY_RUN
        fi
    else
        # Batch mode
        echo "Batch mode: processing ${BASE_DIR}/${dataset}"
        
        if [[ "$SAMPLING_TYPE" == "fps" ]]; then
            python src/sampling/pa_sampling.py \
                --video_folder "${BASE_DIR}/${dataset}" \
                --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
                --output_folder "${BASE_DIR}/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f/${dataset}" \
                --num_frames $NUM_FRAMES \
                --sampling_type "$SAMPLING_TYPE" \
                --predictions_root "${PREDICTIONS_ROOT}/${dataset}" \
                --fps_distance_mode "$FPS_DISTANCE_MODE" \
                --fps_starting_mode "$FPS_STARTING_MODE" \
                $VISUALIZE_SAMPLING \
                $PLOT_POSE_ANALYSIS \
                $DRY_RUN
        elif [[ "$SAMPLING_TYPE" == "efficient" ]]; then
            python src/sampling/pa_sampling.py \
                --video_folder "${BASE_DIR}/${dataset}" \
                --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
                --output_folder "${BASE_DIR}/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f/${dataset}" \
                --num_frames $NUM_FRAMES \
                --sampling_type "$SAMPLING_TYPE" \
                --predictions_root "${PREDICTIONS_ROOT}/${dataset}" \
                --efficient_sampling_mode "$EFFICIENT_SAMPLING_MODE" \
                --efficient_normalization "$EFFICIENT_NORMALIZATION" \
                --efficient_diagonal_priority $EFFICIENT_DIAGONAL_PRIORITY \
                --efficient_starting_mode "$EFFICIENT_STARTING_MODE" \
                $VISUALIZE_SAMPLING \
                $PLOT_POSE_ANALYSIS \
                $DRY_RUN
        elif [[ "$SAMPLING_TYPE" == "mergeaware_fps" ]]; then
            python src/sampling/pa_sampling.py \
                --video_folder "${BASE_DIR}/${dataset}" \
                --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
                --output_folder "${BASE_DIR}/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f/${dataset}" \
                --num_frames $NUM_FRAMES \
                --sampling_type "$SAMPLING_TYPE" \
                --predictions_root "${PREDICTIONS_ROOT}/${dataset}" \
                --fps_distance_mode "$FPS_DISTANCE_MODE" \
                --fps_starting_mode "$FPS_STARTING_MODE" \
                --neighbor_mode "$NEIGHBOR_MODE" \
                --index_step_size $INDEX_STEP_SIZE \
                $ENFORCE_DUPLICATE \
                $VISUALIZE_SAMPLING \
                $PLOT_POSE_ANALYSIS \
                $DRY_RUN
        elif [[ "$SAMPLING_TYPE" == "mergeaware_efficient" ]]; then
            python src/sampling/pa_sampling.py \
                --video_folder "${BASE_DIR}/${dataset}" \
                --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
                --output_folder "${BASE_DIR}/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f/${dataset}" \
                --num_frames $NUM_FRAMES \
                --sampling_type "$SAMPLING_TYPE" \
                --predictions_root "${PREDICTIONS_ROOT}/${dataset}" \
                --efficient_sampling_mode "$EFFICIENT_SAMPLING_MODE" \
                --efficient_normalization "$EFFICIENT_NORMALIZATION" \
                --efficient_diagonal_priority $EFFICIENT_DIAGONAL_PRIORITY \
                --efficient_starting_mode "$EFFICIENT_STARTING_MODE" \
                --neighbor_mode "$NEIGHBOR_MODE" \
                --index_step_size $INDEX_STEP_SIZE \
                $ENFORCE_DUPLICATE \
                $VISUALIZE_SAMPLING \
                $PLOT_POSE_ANALYSIS \
                $DRY_RUN
        fi
    fi
    
    echo "Sampling complete for ${dataset}"
}

# ============================================================================
# Run sampling for different datasets
# ============================================================================

# Uncomment the dataset you want to process
# run_sampling "scannet"
# run_sampling "scannetpp"
run_sampling "arkitscenes"

echo ""
echo "============================================"
echo "✅ All sampling tasks completed!"
echo "============================================"
