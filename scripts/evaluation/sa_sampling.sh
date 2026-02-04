#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

# Global
DATA_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets"
MODELS_ROOT="/data/horse/ws/jixu233b-metadata_ws/models"

# activate conda
source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
module load CUDA/12.4.0 # nvcc

cd "$(dirname "$0")"
cd ../..

# This avoids NFS slowdowns.
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR

# Print current directory
pwd

# download VGGT-1B model checkpoint
# hf download facebook/VGGT-1B --local-dir ${MODELS_ROOT}/VGGT-1B

# run sampling
BASE_DIR="${DATA_ROOT}/vsibench"

# Number of frames to sample
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_FRAMES="${NUM_FRAMES:-8}"

# Sampling type: "both" (default), "sa", or "uniform"
SAMPLING_TYPE="${SAMPLING_TYPE:-both}"
SAMPLING_TYPE="${SAMPLING_TYPE:-sa}"

if [[ "$SAMPLING_TYPE" == "both" || "$SAMPLING_TYPE" == "sa" ]]; then
    mkdir -p "$BASE_DIR/sa_sampling_${NUM_FRAMES}f"
fi
if [[ "$SAMPLING_TYPE" == "both" || "$SAMPLING_TYPE" == "uniform" ]]; then
    mkdir -p "$BASE_DIR/uniform_sampling_${NUM_FRAMES}f"
fi

run_sampling() {
    dataset=$1
    temp_dir="${BASE_DIR}/${dataset}_temp"
    
    python src/sampling/sa_sampling.py \
        --video_folder "${BASE_DIR}/${dataset}" \
        --model_path "${MODELS_ROOT}/Spatial-MLLM/checkpoints/VGGT-1B" \
        --output_folder "$temp_dir" \
        --num_frames $NUM_FRAMES \
        --sampling_type "$SAMPLING_TYPE"

    if [[ "$SAMPLING_TYPE" == "both" || "$SAMPLING_TYPE" == "sa" ]]; then
        mv "${temp_dir}/sa_sampling" "${BASE_DIR}/sa_sampling_${NUM_FRAMES}f/${dataset}"
    fi
    if [[ "$SAMPLING_TYPE" == "both" || "$SAMPLING_TYPE" == "uniform" ]]; then
        mv "${temp_dir}/uniform_sampling" "${BASE_DIR}/uniform_sampling_${NUM_FRAMES}f/${dataset}"
    fi
    rmdir "$temp_dir"
}

run_sampling "scannet"
run_sampling "scannetpp"
run_sampling "arkitscenes"