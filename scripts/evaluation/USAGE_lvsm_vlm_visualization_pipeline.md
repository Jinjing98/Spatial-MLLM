# LVSM + VLM 完整可视化 Pipeline 维护手册

本手册目标：把 `采样 -> 预计算 -> 训练 -> 推理 -> 评测` 的关键可视化链路固定下来，保证实验可复现、可对比、可排错。

---

## 1. 设计原则

- 最小改动优先：只在必要位置加可视化/日志，不改主训练语义。
- 产物先于图：先保证中间产物结构稳定，再做展示图。
- 不静默改行为：所有 debug 开关必须显式记录在实验名和日志里。
- 统一命名：同一视频在不同阶段使用同一个 `video_key`。

---

## 2. 全链路输入/输出契约

### 2.1 采样阶段（SA/FPS/Efficient）

输入：
- 原始视频目录
- 采样策略配置（`num_frames`, `sampling_type` 等）

输出（每个视频目录）：
- `selected_frames.json`
- `sampling_quality.html`（可选）
- `pose_analysis.html`（可选）
- `*_predictions.pt`（启用 `--save_extra` 时）

参考：
- `scripts/evaluation/USAGE_sa_visualization.md`
- `src/utils/visualisation.py`

### 2.2 预计算阶段（VGGT pose package）

输入：
- `dataset_use`
- `DATASET_ROOT`
- VGGT checkpoint

输出（按 `video_key` 对齐）：
- `<video_rel_path>.pt`

必含字段：
- `input_frame_indices`
- `novel_pool_indices`
- `nvs_is_input_mask`
- `input_extrinsics_w2c`
- `input_intrinsics`
- `target_extrinsics_w2c`
- `target_intrinsics`

参考脚本：
- `scripts/preprocess/precompute_cam_info_dataset_vggt.py`
- `scripts/preprocess/precompute_cam_info_full_dataset_vggt.sh`
- `scripts/preprocess/precompute_cam_info_full_dataset_vggt_torchrun_hpc.sh`

### 2.3 训练阶段（LVSM+VLM）

输入：
- 训练数据 + 可选预计算 pose 包
- 训练脚本配置（含 `nvs_enabled`、`use_pre_compute_pose`）

可视化/诊断输出：
- wandb：
  - `nvs/*`（loss/psnr）
  - `gates/*`（`nvs_gate`, `mod_gate`, `sdpa`）
  - `fuse/*`, `adapt/*`（delta ratio）
  - `grad/*`（bridge grad norm）
- 日志：
  - `[NVS-SKIP]` 条件分解
  - 可选 `[DEBUG_COMPARE_PRECOMPUTED_POSE]` 差异统计

关键脚本：
- `scripts/training/spatial_mllm_train_demo_DDP_LVSM_hpc.sh`
- `scripts/training/spatial_mllm_train_demo_DDP_LVSM_overfitting_hpc.sh`

### 2.4 推理阶段（单视频可视化验证）

输入：
- 模型 checkpoint
- 视频（或帧目录）
- 文本 query

输出：
- 终端响应 + 时延统计
- 若视频目录含 `selected_frames.json`，会自动加载 selected frame id（`custom-spatial-mllm` 路径）

脚本：
- `src/inference.py`
- `scripts/evaluation/inference_hpc.sh`

### 2.5 评测阶段（VSIBench）

输入：
- 评测 annotation/video
- 模型与采样配置

输出：
- `eval_result*.json` / 汇总指标
- 可选趋势图（checkpoint trend）

脚本：
- `src/evaluation/vsibench/eval_vsibench.py`
- `src/evaluation/vsibench/eval_vsibench_multiturn.py`
- `scripts/evaluation/evaluate_vsibench_spatial_mllm*.sh`
- `scripts/evaluation/evaluate_vsibench_multiturn_qwen25.sh`

---

## 3. 推荐最小维护流程（每次改动后）

1. 采样可视化冒烟：
   - 跑 1 个视频，生成 `sampling_quality.html` + `pose_analysis.html`。
2. 预计算一致性冒烟：
   - 对同一视频生成 `.pt`，检查 7 个必需字段。
3. 训练 20 样本 overfit：
   - 开启 `nvs_enabled`，确认 wandb 出现 `nvs/*`、`gates/*`、`grad/*`。
4. 单视频推理对比：
   - 同一输入对比 baseline 模型与 LVSM 模型响应。
5. 小规模 VSIBench：
   - 先单 dataset + 单 question type，确认输出格式稳定。

---

## 4. 关键开关矩阵（必须显式记录）

训练前请在 run name 或日志写清楚：

- 数据路径：
  - `nvs_enabled`
  - `use_pre_compute_pose`
  - `precompute_pose_root`
- 适配路径：
  - `vlm2context_adapt_strategy`
  - `lvsm2qwen_type`
  - `llm2lvsm_type`
- 调试路径（会改变训练行为）：
  - `disable_lvsm2llm_fusion`
  - `disable_llm2lvsm_fusion`
  - `nvs_loss_only`
  - `decoder_input_vggt_geo`

---

## 5. 目录与命名建议

建议统一按实验根目录保存：

```text
<exp_root>/
  sampling/
    <scene>/selected_frames.json
    <scene>/sampling_quality.html
    <scene>/pose_analysis.html
  precompute_pose_vggt/
    <video_rel>.pt
  train/
    <run_name>/train.log
  eval/
    vsibench/<run_name>/eval_result.json
    vsibench_multiturn/<run_name>/eval_result.json
```

命名建议：
- run 名必须包含：`model_type + nframes + sampling + precompute(on/off) + adapt_strategy`

---

## 6. 常见故障与定位

- 训练出现 `[NVS-SKIP]`：
  - 先看条件分解日志，重点检查 `nvs_target_tchw` 和 target cameras 是否都存在。
- precompute 路径报 missing key：
  - 检查 `.pt` 是否由当前 `precompute_cam_info_dataset_vggt.py` 生成。
- 推理结果异常但训练正常：
  - 检查推理侧是否沿用了训练同款开关（尤其 Pose RoPE / frame selection）。
- DDP 梯度问题：
  - 确认开启 `gradient_checkpointing` 时 `use_reentrant=False` 已生效。

---

## 7. 与 DESIGN 文档的关系

- 架构与模型内部逻辑：看 `src/custom_qwenvl/lvsm_utils/DESIGN.md`
- 操作与维护流程：看本文件

两者需同步维护：当模型逻辑改动时，先更新 `DESIGN.md`，再更新本手册中的“开关矩阵”和“最小维护流程”。
