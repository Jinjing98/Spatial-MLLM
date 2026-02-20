# VSIBench Metrics LaTeX Formatter

将 VSIBench 的 metrics JSON 文件转换为 LaTeX 表格行。

## 快速使用

```bash
# Style 1: 简洁版 (RelDir平均值)
python src/evaluation/vsibench/format_metrics_latex.py \
    results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl.json \
    --style 1 --header

# Style 2: 完整版 (RelDir分开 + MRA + ACC)
python src/evaluation/vsibench/format_metrics_latex.py \
    results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl.json \
    --style 2 --model-name "Qwen3-VL (8f)" --header

# Style 3: 同 Style 1
python src/evaluation/vsibench/format_metrics_latex.py \
    results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl.json \
    --style 3 --header

# 批量处理多个 dataset
python src/evaluation/vsibench/format_metrics_latex.py \
    results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl_*.json \
    --style 1 --header --hline
```

## 输出格式

### Style 1 & 3 (简洁版)
```
Model & ObjCnt & AbsDist & ObjSz & RoomSz & RelDist & RelDir & RoutePl & ApprOrd & Overall \\
\hline
qwen3-vl & 0.39 & 0.37 & 0.73 & 0.55 & 0.40 & 0.44 & 0.38 & - & 0.49 \\
```

### Style 2 (完整版)
```
Model & ObjCnt & AbsDist & ObjSz & RoomSz & RelDist & RelDir-E & RelDir-M & RelDir-H & RoutePl & ApprOrd & MRA & ACC & Overall \\
\hline
Qwen3-VL & 0.39 & 0.37 & 0.73 & 0.55 & 0.40 & 0.49 & 0.45 & 0.37 & 0.38 & - & 0.54 & 0.41 & 0.49 \\
```

## 参数说明

- `--style {1,2,3}`: 输出格式
  - Style 1: 简洁版 (RelDir 加权平均)
  - Style 2: 完整版 (RelDir 三个难度 + MRA + ACC)
  - Style 3: 同 Style 1
  
- `--model-name`: 自定义模型名称（默认从文件名提取）
  
- `--precision`: 小数位数（默认 2）
  
- `--header`: 输出表格表头
  
- `--hline`: 每行后添加 `\hline`

## 自定义格式

编辑 `format_metrics_latex.py` 中的配置：

```python
STYLE_CONFIGS = {
    "style1": {
        "columns": [
            "ObjCnt", "AbsDist", "ObjSz", "RoomSz", "RelDist",
            "RelDir-Avg", "RoutePl", "ApprOrd", "all_micro"
        ],
        "reldir_mode": "avg",  # weighted_avg or direct_avg
        "show_mra_acc": False,
    },
    # ... 添加更多 style
}
```

## 高级用法

### 生成完整的 LaTeX 表格

```bash
# \begin{tabular}{l|ccccccccc}

cat > table.tex << 'EOF'
\begin{table}[h]
\centering
\begin{tabular}{l|ccccccccccc}
\hline
EOF

python src/evaluation/vsibench/format_metrics_latex.py     results/vsibench_efficient_sampling/qwen3-vl*/eval_result/metrics_qwen3-vl_*.json     --style 2 --header >> table.tex

# python src/evaluation/vsibench/format_metrics_latex.py \
#     results/*/eval_result/metrics_*.json \
#     --style 2 --header >> table.tex

echo '\hline' >> table.tex
echo '\end{tabular}' >> table.tex
echo '\caption{VSIBench Results}' >> table.tex
echo '\end{table}' >> table.tex
```

### 对比多个模型

```bash
python src/evaluation/vsibench/format_metrics_latex.py \
    results/vsibench_efficient_sampling/qwen2.5-vl-8f/eval_result/metrics_qwen2.5-vl.json \
    --style 1 --model-name "Qwen2.5-VL" --header

python src/evaluation/vsibench/format_metrics_latex.py \
    results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl.json \
    --style 1 --model-name "Qwen3-VL"
```

### 每个 dataset 单独一行

```bash
for dataset in arkitscenes scannet scannetpp; do
    python src/evaluation/vsibench/format_metrics_latex.py \
        results/vsibench_efficient_sampling/qwen3-vl-8f/eval_result/metrics_qwen3-vl_${dataset}.json \
        --style 1 --model-name "Qwen3-VL (${dataset})"
done
```

## 问题类型缩写

| 完整名称 | 缩写 |
|---------|------|
| object_counting | ObjCnt |
| object_abs_distance | AbsDist |
| object_size_estimation | ObjSz |
| room_size_estimation | RoomSz |
| object_rel_distance | RelDist |
| object_rel_direction_easy | RelDir-E |
| object_rel_direction_medium | RelDir-M |
| object_rel_direction_hard | RelDir-H |
| route_planning | RoutePl |
| obj_appearance_order | ApprOrd |

可在脚本中修改 `QUESTION_TYPE_ABBR` 字典来自定义缩写。
