#!/bin/bash

cd /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/previous_version/src

source activate /mnt/cluster/environments/jinjingxu/pkg/envs/transformers_latest

# ==============================================================================
# INTERACTIVE MODE: Launch GUI for uploading and visualizing
# ==============================================================================
python demo_gradio_flex.py


# ==============================================================================
# BATCH PROCESSING MODE: Process videos automatically without GUI
# ==============================================================================
# Use demo_gradio.py with --mp4_file_path and --save_root for batch processing
# 
# FRAME SAMPLING OPTIONS (choose one):
#   --total_frames N     : Extract exactly N frames uniformly from video
#   --sample_fps X       : Extract frames at X seconds interval (default: 1.0)
#
# SAVING OPTIONS (default: save all three):
#   By default: saves images/, predictions.npz, and .glb file
#   --disable_img_save           : Do NOT save video frames
#   --disable_predictions_save   : Do NOT save predictions.npz
#   --disable_glb_save          : Do NOT save GLB file
#
# OUTPUT DIRECTORY FORMAT:
#   {save_root}/{video_name}_{total_frames}_{sample_fps}_{MMDDHHMMSS}/


# ------------------------------------------------------------------------------
# Example 1: Process video with 30 frames, save all outputs
# ------------------------------------------------------------------------------
# python demo_gradio.py \
#     --mp4_file_path /path/to/video.mp4 \
#     --save_root /path/to/save_root \
#     --total_frames 30 \
#     --no_gui


# ------------------------------------------------------------------------------
# Example 2: Process video, only save GLB (minimal disk usage)
# ------------------------------------------------------------------------------
# python demo_gradio.py \
#     --mp4_file_path /path/to/video.mp4 \
#     --save_root /path/to/save_root \
#     --sample_fps 1.0 \
#     --disable_img_save \
#     --disable_predictions_save \
#     --no_gui


# ------------------------------------------------------------------------------
# Example 3: Large frame processing (500+ frames) on GPU node
# ------------------------------------------------------------------------------
# Step 1: Run on A100 GPU node to process video
# srun --gres=gpu:a100 --pty bash
# source activate /mnt/cluster/environments/jinjingxu/pkg/envs/transformers_latest
# python previous_version/src/demo_gradio.py \
#     --mp4_file_path /path/to/large_video.mp4 \
#     --save_root /path/to/save_root \
#     --total_frames 500 \
#     --no_gui
# 
# Step 2: Visualize results in GUI (on local workstation)
# source activate /mnt/cluster/environments/jinjingxu/pkg/envs/transformers_latest
# python previous_version/src/demo_gradio_flex.py
# # In the GUI, go to "Load Existing Results" and enter the output directory path


# ------------------------------------------------------------------------------
# Example 4: Sample at specific FPS interval
# ------------------------------------------------------------------------------
# python demo_gradio.py \
#     --mp4_file_path /path/to/video.mp4 \
#     --save_root /path/to/save_root \
#     --sample_fps 0.5 \
#     --no_gui
