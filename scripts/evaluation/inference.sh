#!/bin/bash

MODEL_PATH="Diankun/Spatial-MLLM-v1.1-Instruct-135K"
MODEL_TYPE="spatial-mllm"
TEXT="How many chair(s) are in this room?\nPlease answer the question using a single word or phrase."

python src/inference.py \
    --model_path Diankun/Spatial-MLLM-v1.1-Instruct-135K \
    --model_type spatial-mllm \
    --text "How many chair(s) are in this room?\nPlease answer the question using a single word or phrase."