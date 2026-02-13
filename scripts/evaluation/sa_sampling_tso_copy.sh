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

# Global
# DATA_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets"
# MODELS_ROOT="/data/horse/ws/jixu233b-metadata_ws/models"
# RESULTS_SAVE_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets"

# tso
DATA_ROOT="/mnt/nct-zfs/TCO-All/SharedDatasets"
MODELS_ROOT="/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/"
RESULTS_SAVE_ROOT="/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets"

    # parser.add_argument("--base_data_root", type=str, default="/data/horse/ws/jixu233b-metadata_ws/datasets/vsibench",


# activate conda
# source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
# conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
# module load CUDA/12.4.0 # nvcc

# cd "$(dirname "$0")"
# cd ../..
# cd "$SLURM_SUBMIT_DIR"

# This avoids NFS slowdowns.
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR

# Print current directory
pwd

# download VGGT-1B model checkpoint
# hf download facebook/VGGT-1B --local-dir ${MODELS_ROOT}/VGGT-1B

# run sampling
BASE_DIR="${RESULTS_SAVE_ROOT}/vsibench"

# Number of frames to sample
# NUM_FRAMES="${NUM_FRAMES:-4}"
NUM_FRAMES="${NUM_FRAMES:-8}"
# NUM_FRAMES="${NUM_FRAMES:-16}"
# NUM_FRAMES="${NUM_FRAMES:-32}"

# Sampling type: "both" (default), "sa", "uniform", "mergeaware_uniform", "mergeaware_sa"
# SAMPLING_TYPE="${SAMPLING_TYPE:-both}"
SAMPLING_TYPE="${SAMPLING_TYPE:-sa}"
# SAMPLING_TYPE="${SAMPLING_TYPE:-mergeaware_uniform}"
# SAMPLING_TYPE="${SAMPLING_TYPE:-uniform}"
# SAMPLING_TYPE="${SAMPLING_TYPE:-mergeaware_sa}"

# JJ : Temporal merge aware sampling parameters
# neighbor_mode: "before" (default), "after", or "random"
NEIGHBOR_MODE="${NEIGHBOR_MODE:-before}"
# step_size: step size for adding neighbor frames (default: 1)
STEP_SIZE="${STEP_SIZE:-1}"

# Dry run mode: set DRY_RUN="--dry_run" to only check anomalies without processing
# Usage: DRY_RUN="--dry_run" bash scripts/evaluation/sa_sampling.sh
DRY_RUN="${DRY_RUN:-}"
VIDEO_PATH="/mnt/nct-zfs/TCO-All/SharedDatasets/vsibench/arkitscenes/42446103.mp4"

if [[ -n "$DRY_RUN" ]]; then
    echo "============================================"
    echo "DRY RUN MODE: Only checking for anomalies"
    echo "No files will be created or modified"
    echo "============================================"
fi

run_sampling() {
    dataset=$1
    
    # JJ : Build output folder suffix with temporal merge aware parameters
    if [[ "$SAMPLING_TYPE" == "mergeaware_uniform" ]] || [[ "$SAMPLING_TYPE" == "mergeaware_sa" ]]; then
        # Abbreviate neighbor_mode: before->bef, after->aft, random->rnd
        case "$NEIGHBOR_MODE" in
            "before") NBR_ABBREV="bef" ;;
            "after")  NBR_ABBREV="aft" ;;
            "random") NBR_ABBREV="rnd" ;;
            *)        NBR_ABBREV="$NEIGHBOR_MODE" ;;
        esac
        SUFFIX="_nbr${NBR_ABBREV}_ss${STEP_SIZE}"
    else
        SUFFIX=""
    fi
    
    if [[ "$SAMPLING_TYPE" == "both" ]]; then
        # Process SA sampling
        echo "Processing SA sampling for ${dataset}..."
        python src/sampling/sa_sampling.py \
            --video_folder "${BASE_DIR}/${dataset}" \
            --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
            --output_folder "${BASE_DIR}/sa_sampling_${NUM_FRAMES}f/${dataset}" \
            --num_frames $NUM_FRAMES \
            --sampling_type "sa" \
            $DRY_RUN
        
        # Process uniform sampling
        echo "Processing uniform sampling for ${dataset}..."
        python src/sampling/sa_sampling.py \
            --video_folder "${BASE_DIR}/${dataset}" \
            --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
            --output_folder "${BASE_DIR}/uniform_sampling_${NUM_FRAMES}f/${dataset}" \
            --num_frames $NUM_FRAMES \
            --sampling_type "uniform" \
            $DRY_RUN
    else
        # Single sampling type
        echo "Processing ${SAMPLING_TYPE} sampling for ${dataset}..."
        python src/sampling/sa_sampling.py \
            --video_folder "${BASE_DIR}/${dataset}" \
            --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
            --output_folder "${BASE_DIR}/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f${SUFFIX}/${dataset}" \
            --num_frames $NUM_FRAMES \
            --sampling_type "$SAMPLING_TYPE" \
            --neighbor_mode "$NEIGHBOR_MODE" \
            --step_size $STEP_SIZE \
            --video_path "$VIDEO_PATH" \
            --save_extra \
            $DRY_RUN
    fi
    
    echo "Sampling complete for ${dataset}"
}

# run_sampling "scannet"
# run_sampling "scannetpp"
run_sampling "arkitscenes"