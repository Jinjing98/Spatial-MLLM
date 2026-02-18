# pa_sampling_fps_efficient_tso.sh 快速使用指南

## 🎯 脚本用途
在 SLURM 上运行 FPS/Efficient 姿态感知采样，复用已有的 VGGT predictions。

## ⚠️ 前置条件
```bash
# 1. 先运行 SA 采样生成 predictions.pt
sbatch scripts/evaluation/pa_sampling_tso.sh

# 2. 确保 VGGT-1B 模型已下载
```

## 🚀 常用命令

### FPS 采样（默认）
```bash
sbatch scripts/evaluation/pa_sampling_fps_efficient_tso.sh
```

### Efficient 采样
```bash
SAMPLING_TYPE=efficient sbatch scripts/evaluation/pa_sampling_fps_efficient_tso.sh
```

### 带可视化
```bash
SAMPLING_TYPE=fps \
VISUALIZE_SAMPLING="--visualize_sampling" \
PLOT_POSE_ANALYSIS="--plot_pose_analysis" \
sbatch scripts/evaluation/pa_sampling_fps_efficient_tso.sh
```

### 测试单个视频
```bash
VIDEO_PATH="/mnt/nct-zfs/TCO-All/SharedDatasets/vsibench/arkitscenes/42446103.mp4" \
sbatch scripts/evaluation/pa_sampling_fps_efficient_tso.sh
```

## 🎛️ 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SAMPLING_TYPE` | `fps` | 采样类型: `fps` 或 `efficient` |
| `NUM_FRAMES` | `16` | 采样帧数 |
| `FPS_DISTANCE_MODE` | `max_norm` | FPS距离: `max_norm`, `data_driven` |
| `FPS_STARTING_MODE` | `medoid` | FPS起始: `medoid`, `first`, `rand` |
| `EFFICIENT_SAMPLING_MODE` | `hybrid` | Efficient模式: `grid`, `hybrid`, `fps_2d` |
| `EFFICIENT_NORMALIZATION` | `std_norm` | 归一化: `std_norm`, `max_norm` |
| `VISUALIZE_SAMPLING` | (空) | 设为 `"--visualize_sampling"` 启用 |
| `PLOT_POSE_ANALYSIS` | (空) | 设为 `"--plot_pose_analysis"` 启用 |

## 📂 输出位置
```
datasets/vsibench/${SAMPLING_TYPE}_sampling_${NUM_FRAMES}f/${dataset}/
├── video_name/
    ├── selected_frames.json          # Metadata
    ├── video_name_frame_*.png        # 选中的帧
    ├── sampling_quality_*.html       # 可视化（可选）
    └── pose_analysis.html            # 分析（可选）
```

## 📊 监控任务
```bash
# 查看任务状态
squeue -u $USER

# 查看输出日志
tail -f /data/horse/ws/jixu233b-metadata_ws/hpc_out/[JOB_ID].out

# 取消任务
scancel [JOB_ID]
```

## ✅ 推荐工作流

```bash
# Step 1: SA 采样生成 predictions（一次性）
sbatch scripts/evaluation/pa_sampling_tso.sh

# Step 2: FPS 采样（复用 predictions）
SAMPLING_TYPE=fps \
VISUALIZE_SAMPLING="--visualize_sampling" \
sbatch scripts/evaluation/pa_sampling_fps_efficient_tso.sh

# Step 3: Efficient 采样（复用 predictions）
SAMPLING_TYPE=efficient \
VISUALIZE_SAMPLING="--visualize_sampling" \
sbatch scripts/evaluation/pa_sampling_fps_efficient_tso.sh
```

**优势**: VGGT 推理只运行一次，节省计算资源！

## 🐛 常见问题

**Q: "predictions.pt not found"**  
A: 检查脚本 line 48 的 `PREDICTIONS_ROOT` 路径，确保先运行了 `sa_sampling.py`

**Q: "decord not installed"**  
A: `pip install decord`（采样仍会完成，只是不提取图像）

**Q: 如何修改处理的数据集？**  
A: 编辑脚本 line 149，将 `run_sampling "arkitscenes"` 改为其他数据集
