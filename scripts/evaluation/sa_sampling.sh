# !/bin/bash

cd "$(dirname "$0")"
cd ../..

# download VGGT-1B model checkpoint
# hf download facebook/VGGT-1B --local-dir checkpoints/VGGT-1B

# run sampling
# BASE_DIR="datasets/evaluation/vsibench"
# tso
BASE_DIR="/mnt/nct-zfs/TCO-All/SharedDatasets/vsibench"
# VIDEO_PATH="/mnt/nct-zfs/TCO-All/SharedDatasets/vsibench/arkitscenes/42446103.mp4"
# TMP_DIR="/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/tmp"

mkdir -p "$BASE_DIR/sa_sampling_16f"
mkdir -p "$BASE_DIR/uniform_sampling_16f"

run_sampling() {
    dataset=$1
    temp_dir="${BASE_DIR}/${dataset}_temp"
    # temp_dir="${TMP_DIR}/${dataset}_temp"
    
    python src/sampling/sa_sampling.py \
        --video_folder "${BASE_DIR}/${dataset}" \
        --model_path checkpoints/VGGT-1B \
        --output_folder "$temp_dir" \
        # --video_path "$VIDEO_PATH" \
        # --save_video \

    mv "${temp_dir}/sa_sampling" "${BASE_DIR}/sa_sampling_16f/${dataset}"
    mv "${temp_dir}/uniform_sampling" "${BASE_DIR}/uniform_sampling_16f/${dataset}"
    rmdir "$temp_dir"
}

# run_sampling "scannet"
# run_sampling "scannetpp"
run_sampling "arkitscenes"