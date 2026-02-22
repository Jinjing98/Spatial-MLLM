#!/bin/bash

# ## 参数说明

# - `--style {1,2,3}`: 输出格式
#   - Style 1: 简洁版 (RelDir 加权平均)
#   - Style 2: 完整版 (RelDir 三个难度 + MRA + ACC)
#   - Style 3: 同 Style 1
  
# - `--model-name`: 自定义模型名称（默认从文件名提取）
  
# - `--precision`: 小数位数（默认 2）
  
# - `--header`: 输出表格表头
  
# - `--hline`: 每行后添加 `\hline`


# # Style 1: 简洁版 (RelDir平均值)
# python src/evaluation/vsibench/format_metrics_latex.py \
#     results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl.json \
#     --style 1 --header

# # Style 2: 完整版 (RelDir分开 + MRA + ACC)
# python src/evaluation/vsibench/format_metrics_latex.py \
#     results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl.json \
#     --style 2 --model-name "Qwen3-VL (8f)" --header

# # Style 3: 同 Style 1
# python src/evaluation/vsibench/format_metrics_latex.py \
#     results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl.json \
#     --style 3 --header


cat > table.tex << 'EOF'
\begin{table}[H]
\centering
\setlength{\tabcolsep}{0.0pt}
\small
\begin{tabular}{l|ccccccccccccc}
\hline
EOF

# results/vsibench_efficient_sampling/qwen3-vl*/eval_result/metrics_qwen3-vl_*.json  \
# results/vsibench_efficient_sampling/spatial-mllm*/eval_result/metrics_spatial-mllm_*.json  \
# results/vsibench_sa_sampling/spatial-mllm*/eval_result/metrics_spatial-mllm_*.json  \
# results/vsibench/Qwen3-VL*f/eval_result/metrics_qwen3-vl_*.json  \
# results/vsibench_sa_sampling/spatial-mllmad*/eval_result/metrics*mllm.json  \
# results/vsibench_sa_sampling/spatial-mllm-*/eval_result/metrics*mllm.json  \
# results/vsibench_mergeaware_sa_sampling/qwen3*/eval_result/metrics*.json  \

MODEL_QUERY_NAME="qwen3-vl*" 
SAMPLING_QUERY_NAME='_efficient_sampling_grid' #_sa_sampling _efficient_sampling_grid
METRICS_FILE_FORMAT="metrics_${MODEL_QUERY_NAME}.json"
# METRICS_FILE_FORMAT="metrics_${MODEL_QUERY_NAME}_*.json"

MODEL_QUERY_NAME="qwen3-vl*" 
MODEL_QUERY_NAME_AVG_ONLY="qwen3-vl"
SAMPLING_QUERY_NAME='_efficient_sampling_grid' #_fps_sampling _efficient_sampling _sa_sampling _efficient_sampling_grid
METRICS_FILE_FORMAT="metrics_${MODEL_QUERY_NAME}.json"
METRICS_FILE_FORMAT="metrics_${MODEL_QUERY_NAME_AVG_ONLY}.json"

MODEL_QUERY_NAME="spatial-mllm*" 
MODEL_QUERY_NAME_AVG_ONLY="spatial-mllm"
SAMPLING_QUERY_NAME='_sa_sampling' #_efficient_sampling _sa_sampling _efficient_sampling_grid
METRICS_FILE_FORMAT="metrics_${MODEL_QUERY_NAME}.json"
METRICS_FILE_FORMAT="metrics_${MODEL_QUERY_NAME_AVG_ONLY}.json"

MODEL_QUERY_NAME="spatial-mllm*" 
MODEL_QUERY_NAME_AVG_ONLY="spatial-mllm"
SAMPLING_QUERY_NAME='_fps_stdnorm_medoid_sampling' #_fps_stdnorm_medoid_sampling _efficient_sampling_v0_hybrid _efficient_sampling _sa_sampling _efficient_sampling_grid
METRICS_FILE_FORMAT="metrics_${MODEL_QUERY_NAME}.json"
METRICS_FILE_FORMAT="metrics_${MODEL_QUERY_NAME_AVG_ONLY}.json"

QUERY_FORMATS=(results/vsibench${SAMPLING_QUERY_NAME}/${MODEL_QUERY_NAME}/eval_result/${METRICS_FILE_FORMAT})
echo "Query format: ${QUERY_FORMATS}"

python src/evaluation/vsibench/format_metrics_latex.py \
"${QUERY_FORMATS[@]}" \
--precision 1 \
--style 2 \
--header >> table.tex

echo '\hline' >> table.tex
echo '\end{tabular}' >> table.tex
echo '\caption{VSIBench Results}' >> table.tex
echo '\end{table}' >> table.tex