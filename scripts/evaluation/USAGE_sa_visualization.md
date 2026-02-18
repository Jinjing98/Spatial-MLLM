# SA Sampling Visualization 使用指南

## 🎯 功能概述

为 `sa_sampling.py` 添加了可视化功能，可以生成：
- **`sampling_quality.html`**: 3D轨迹、采样覆盖率、距离矩阵
- **`pose_analysis.html`**: Farness 分析、2D scatter plots

## 📋 支持的策略

| 策略 | 可视化支持 | 要求 |
|------|-----------|------|
| `sa` | ✅ 完整支持 | `--save_extra` (自动有 VGGT predictions) |
| `mergeaware_sa` | ✅ 完整支持 | `--save_extra` (自动有 VGGT predictions) |
| `mergeaware_uniform` | ✅ 完整支持 | `--save_extra` (需要运行 VGGT) |
| `uniform` | ❌ 跳过 | 默认无 pose 信息 |

## 🚀 使用方法

### 方法 1: 通过 Shell 脚本（推荐）

#### 启用可视化
```bash
# 启用 sampling_quality.html
VISUALIZE_SAMPLING="--visualize_sampling" bash scripts/evaluation/sa_sampling_tso.sh

# 启用两个可视化
VISUALIZE_SAMPLING="--visualize_sampling" \
PLOT_POSE_ANALYSIS="--plot_pose_analysis" \
bash scripts/evaluation/sa_sampling_tso.sh
```

#### 与 `pa_sampling_fps_efficient_tso.sh` 保持一致
```bash
# 编辑 sa_sampling_tso.sh 启用可视化（Line 76-79）
VISUALIZE_SAMPLING="--visualize_sampling"
PLOT_POSE_ANALYSIS="--plot_pose_analysis"
```

### 方法 2: 直接使用 Python

```bash
python src/sampling/sa_sampling.py \
    --video_folder /path/to/videos \
    --model_path /path/to/VGGT-1B \
    --output_folder /path/to/output \
    --num_frames 16 \
    --sampling_type sa \
    --save_extra \
    --visualize_sampling \
    --plot_pose_analysis
```

## 📊 输出文件

运行后，在输出目录中会生成：

```
output_dir/
├── video_name/
│   ├── selected_frames.json          # 元数据
│   ├── sa_predictions.pt             # VGGT predictions (需要 --save_extra)
│   ├── sampling_quality.html         # 可视化 1 (需要 --visualize_sampling)
│   └── pose_analysis.html            # 可视化 2 (需要 --plot_pose_analysis)
```

## 🔧 实现细节

### 复用的代码
- 完全复用 `src/utils/visualisation.py` 中的 `visualize_pose_sampling_results()`
- 与 `pa_sampling.py` 的 FPS/Efficient 策略使用相同的可视化函数

### 关键特性
1. **自动提取 poses**: 从 `predictions['extrinsic']` 提取 (128, 4, 4)
2. **优雅降级**: 可视化失败时不影响采样流程
3. **条件执行**: 只在 `--save_extra` 时运行（确保有 predictions）
4. **一致的 API**: 与 `pa_sampling.py` 保持相同的参数名称

## 📝 注意事项

1. **必须启用 `--save_extra`**: 
   - SA/MergeAware-SA 自动生成 predictions
   - MergeAware-Uniform 会运行 VGGT inference
   - 标准 Uniform 默认不生成（跳过可视化）

2. **输出路径**: 
   - 可视化 HTML 保存在与采样帧相同的目录
   - 不写入 `/tmp`，而是与数据保持一致

3. **性能影响**: 
   - 可视化计算时间 < 5秒（对于 128 帧）
   - 不影响 VGGT 推理或采样时间

## 🔍 与 FPS/Efficient 的对比

| 特性 | SA Sampling | FPS/Efficient Sampling |
|------|-------------|------------------------|
| 脚本 | `sa_sampling_tso.sh` | `pa_sampling_fps_efficient_tso.sh` |
| Python | `src/sampling/sa_sampling.py` | `src/sampling/pa_sampling.py` |
| Pose Source | VGGT (固定) | GT 或 VGGT (可选) |
| 可视化函数 | `visualize_pose_sampling_results()` | `visualize_pose_sampling_results()` |
| 参数 | `--visualize_sampling`, `--plot_pose_analysis` | 相同 |

## 📚 示例

### 单视频模式 + 完整可视化
```bash
VIDEO_PATH="/path/to/video.mp4" \
VISUALIZE_SAMPLING="--visualize_sampling" \
PLOT_POSE_ANALYSIS="--plot_pose_analysis" \
SAMPLING_TYPE="sa" \
NUM_FRAMES=16 \
bash scripts/evaluation/sa_sampling_tso.sh
```

### 批量模式 + 仅 sampling_quality
```bash
VISUALIZE_SAMPLING="--visualize_sampling" \
SAMPLING_TYPE="mergeaware_sa" \
NUM_FRAMES=32 \
bash scripts/evaluation/sa_sampling_tso.sh
```

## ✅ 验证

检查可视化是否生成：
```bash
# 检查输出目录
ls -lh /path/to/output/video_name/*.html

# 预期输出:
# sampling_quality.html  (如果 --visualize_sampling)
# pose_analysis.html     (如果 --plot_pose_analysis)
```

## 🐛 故障排除

### 问题 1: "Visualization skipped"
**原因**: 未启用 `--save_extra`  
**解决**: 在 shell 脚本中确保 `--save_extra` 参数存在

### 问题 2: "Visualization failed: ..."
**原因**: `visualisation.py` 导入失败或计算错误  
**解决**: 检查 Python 环境和依赖（plotly, numpy）

### 问题 3: Uniform 采样无可视化
**原因**: 标准 Uniform 采样不运行 VGGT（设计如此）  
**解决**: 使用 `mergeaware_uniform` 或手动启用 VGGT
