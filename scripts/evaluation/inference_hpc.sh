#!/bin/bash
# JJ: Support customized parameters from command line or environment variables
# Usage:
#   1. From environment variables:
#      MODEL_PATH=path/to/model VIDEO_PATH=path/to/video bash inference.sh
#   2. From command line arguments:
#      bash inference.sh --model_path path/to/model --video_path path/to/video
#   3. Mix of both (command line args override env vars)

# JJ: Fixed default values (not overridable by env vars)
MODEL_TYPE="qwen2.5-vl"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# MODEL_TYPE="qwen3-vl"
# MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"

# MODEL_TYPE="spatial-mllm-qwen3"
# MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"


# MODEL_TYPE="spatial-mllm"
# MODEL_TYPE="custom-spatial-mllm"
# MODEL_PATH="Diankun/Spatial-MLLM-v1.1-Instruct-135K"

# MODEL_TYPE="spatial-mllm"
# MODEL_TYPE="custom-spatial-mllm"
# MODEL_PATH="Diankun/Spatial-MLLM-v1.1-Instruct-135K"
# # MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"


VIDEO_PATH="datasets/fool_mllm/42446103.mp4" # given video path
VIDEO_PATH="/home/jixu233b/Projects/VLM_3D/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/42446103.mp4" # given sampled video path
# VIDEO_PATH="datasets/fool_mllm/scannetpp_3f15a9266d.mp4" # given video path

# TEXT="How many chair(s) are in this room?\nPlease answer the question using a single word or phrase."
# TEXT="Describe exploration trajectory in the video in detailed 1000 words. Answer in chinese."
TEXT="Describe exploration trajectory in the video in detailed 10 words. Answer in chinese."

# JJ: Collect all command line args
EXTRA_ARGS=("$@")

# Build final ARGS array with all parameters
# e.g. bash inference.sh --mp4_nframes None --sample_fps 1.0 --use_visual False --use_geo True
# e.g. bash inference.sh --use_pose_rope  # Enable 4D Pose RoPE
ARGS=(
    --model_path "${MODEL_PATH}"
    --model_type "${MODEL_TYPE}"
    --video_path "${VIDEO_PATH}"
    --text "${TEXT}"
    "${EXTRA_ARGS[@]}"
    # --pose_enc_type PHW
    # --mrope_section 24 20 20
    # --mp4_nframes 32
    # --pose_enc_type PTHW
    # --mrope_section 8 8 24 24

    # --pose_enc_type PHW
    # --mrope_section 16 24 24
    # --use_pose_rope
)

python src/inference.py "${ARGS[@]}"

# Echo script execution info
echo ""
echo "================================"
echo "Script execution completed"
echo "Model type: ${MODEL_TYPE}"
echo "Model path (folder using all frames): ${MODEL_PATH}"
echo "Video path: ${VIDEO_PATH}"
# extra args
echo "Extra args: ${EXTRA_ARGS[@]}"
echo "================================"