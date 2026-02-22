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
# Global
DATA_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets"
MODELS_ROOT="/data/horse/ws/jixu233b-metadata_ws/models"
VGGT_PRECOMPUTED_DIR='sa_sampling_16f'

# activate conda
source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
module load CUDA/12.4.0 # nvcc

cd "$(dirname "$0")"
cd ../..
cd "$SLURM_SUBMIT_DIR"



# This avoids NFS slowdowns.
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR

# Print current directory
pwd

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR="${DATA_ROOT}/vsibench"

# Number of frames to sample
# NUM_FRAMES="${NUM_FRAMES:-64}"
NUM_FRAMES="${NUM_FRAMES:-32}"
# NUM_FRAMES="${NUM_FRAMES:-16}"
# NUM_FRAMES="${NUM_FRAMES:-8}"
# Sampling type: "fps" or "efficient"
SAMPLING_TYPE="${SAMPLING_TYPE:-fps}"
SAMPLING_TYPE_OUTPUT_FOLDER_NAME_APPENDIX="_stdnorm_medoid"

# SAMPLING_TYPE="${SAMPLING_TYPE:-efficient}"
# SAMPLING_TYPE_OUTPUT_FOLDER_NAME_APPENDIX=""

# JJ for detailed the hyper per strategy
SAMPLING_TYPE_OUTPUT_FOLDER_NAME="${SAMPLING_TYPE}${SAMPLING_TYPE_OUTPUT_FOLDER_NAME_APPENDIX}"

# Predictions root (must exist, from sa_sampling.py output)
# Example: sa_sampling_16f, sa_sampling_128f, etc.
# Can be overridden with environment variable PREDICTIONS_ROOT
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${DATA_ROOT}/vsibench/${VGGT_PRECOMPUTED_DIR}}"

# ============================================================================
# FPS Sampling Parameters
# ============================================================================
FPS_DISTANCE_MODE="${FPS_DISTANCE_MODE:-data_driven}"  # #v0:max_norm Options: data_driven, max_norm, fro, geodesic
FPS_STARTING_MODE="${FPS_STARTING_MODE:-medoid}"    # v0:first Options: medoid, random, first

# ============================================================================
# Efficient Sampling Parameters
# ============================================================================
EFFICIENT_SAMPLING_MODE="${EFFICIENT_SAMPLING_MODE:-hybrid}"    #hybrid # Options: grid, hybrid, fps2d
EFFICIENT_NORMALIZATION="${EFFICIENT_NORMALIZATION:-max_norm}"    # Options: max_norm, std_norm 
EFFICIENT_DIAGONAL_PRIORITY="${EFFICIENT_DIAGONAL_PRIORITY:-0.0}"
EFFICIENT_STARTING_MODE="${EFFICIENT_STARTING_MODE:-first}"    # Options: medoid, random, first

# ============================================================================
# Visualization Options
# ============================================================================
VISUALIZE_SAMPLING="${VISUALIZE_SAMPLING:-}"        # Set to "--visualize_sampling" to enable
PLOT_POSE_ANALYSIS="${PLOT_POSE_ANALYSIS:-}"       # Set to "--plot_pose_analysis" to enable
# VISUALIZE_SAMPLING="--visualize_sampling"        # Set to "--visualize_sampling" to enable
# PLOT_POSE_ANALYSIS="--plot_pose_analysis"       # Set to "--plot_pose_analysis" to enable

# ============================================================================
# Dry Run Mode
# ============================================================================
DRY_RUN="${DRY_RUN:-}"                              # Set to "--dry_run" to enable

# ============================================================================
# Test Mode (single video)
# ============================================================================
# Set VIDEO_PATH to test on a single video
# VIDEO_PATH="/mnt/nct-zfs/TCO-All/SharedDatasets/vsibench/arkitscenes/42446103.mp4"
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
fi
echo "Visualization:  ${VISUALIZE_SAMPLING:-disabled}"
echo "Dry run:        ${DRY_RUN:-disabled}"
echo "============================================"

run_sampling() {
    dataset=$1
    
    echo ""
    echo "Processing ${SAMPLING_TYPE} sampling for ${dataset}..."
    
    # Batch mode
    echo "Batch mode: processing ${BASE_DIR}/${dataset}"
    
    if [[ "$SAMPLING_TYPE" == "fps" ]]; then
        python src/sampling/pa_sampling.py \
            --video_folder "${BASE_DIR}/${dataset}" \
            --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
            --output_folder "${BASE_DIR}/${SAMPLING_TYPE_OUTPUT_FOLDER_NAME}_sampling_${NUM_FRAMES}f/${dataset}" \
            --num_frames $NUM_FRAMES \
            --sampling_type "$SAMPLING_TYPE" \
            --predictions_root "${PREDICTIONS_ROOT}/${dataset}" \
            --fps_distance_mode "$FPS_DISTANCE_MODE" \
            --fps_starting_mode "$FPS_STARTING_MODE" \
            $VISUALIZE_SAMPLING \
            $PLOT_POSE_ANALYSIS \
            $DRY_RUN
            # --video_path "$VIDEO_PATH" \
    elif [[ "$SAMPLING_TYPE" == "efficient" ]]; then
        python src/sampling/pa_sampling.py \
            --video_folder "${BASE_DIR}/${dataset}" \
            --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
            --output_folder "${BASE_DIR}/${SAMPLING_TYPE_OUTPUT_FOLDER_NAME}_sampling_${NUM_FRAMES}f/${dataset}" \
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
            # --video_path "$VIDEO_PATH" \
    fi
    
    echo "Sampling complete for ${dataset}"
}

# ============================================================================
# Run sampling for different datasets
# ============================================================================

# Uncomment the dataset you want to process
run_sampling "scannet"
run_sampling "scannetpp"
run_sampling "arkitscenes"

echo ""
echo "============================================"
echo "All sampling tasks completed!"
echo "============================================"
