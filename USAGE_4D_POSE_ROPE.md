# 4D Pose-aware RoPE Usage Guide

This document explains how to enable and use the 4D Pose-aware RoPE (PTHW) in inference.

## Quick Start

### Option 1: Using Shell Script with Flag

```bash
# Enable 4D Pose RoPE (P+T+H+W)
bash scripts/evaluation/inference_tso.sh --debug_pthw_rope True

# Use standard 3D mRoPE (T+H+W) - default
bash scripts/evaluation/inference_tso.sh --debug_pthw_rope False
# or simply omit the flag
bash scripts/evaluation/inference_tso.sh
```

### Option 2: Direct Python Call

```bash
# Enable 4D Pose RoPE
python src/inference.py \
    --model_type custom-spatial-mllm \
    --model_path Diankun/Spatial-MLLM-v1.1-Instruct-135K \
    --video_path datasets/vsibench/sa_sampling_16f/arkitscenes/41069025 \
    --debug_pthw_rope True

# Disable 4D Pose RoPE (use standard 3D mRoPE)
python src/inference.py \
    --model_type custom-spatial-mllm \
    --model_path Diankun/Spatial-MLLM-v1.1-Instruct-135K \
    --video_path datasets/vsibench/sa_sampling_16f/arkitscenes/41069025 \
    --debug_pthw_rope False
```

## Advanced Configuration

### Customize Pose RoPE Parameters

```bash
python src/inference.py \
    --model_type custom-spatial-mllm \
    --model_path Diankun/Spatial-MLLM-v1.1-Instruct-135K \
    --video_path datasets/vsibench/sa_sampling_16f/arkitscenes/41069025 \
    --debug_pthw_rope True \
    --pose_enc_type PTHW \
    --global_normalize_scale_factor 16.0 \
    --add_offset_in_pose_id False
```

### Parameter Descriptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug_pthw_rope` | bool | False | Enable 4D Pose-aware RoPE |
| `pose_enc_type` | str | "PTHW" | Pose encoding type (only 'PTHW' supported) |
| `global_normalize_scale_factor` | float | 16.0 | Scale factor for Pose normalization, maps Pose to [0, scale_factor] |
| `add_offset_in_pose_id` | bool | False | Whether to add sequential offset to Pose dimension |

## How It Works

### Standard 3D mRoPE (default)
- Position IDs shape: `(3, B, L)` - Temporal + Height + Width
- Dtype: `long`
- Encoding: `T, H, W` dimensions from video grid

### 4D Pose-aware RoPE (when enabled)
- Position IDs shape: `(4, B, L)` - **Pose** + Temporal + Height + Width
- Dtype: `float32` (to preserve Pose precision)
- Encoding:
  - **P dimension**: Camera trajectory encoded via Lie group scalar index
  - T dimension: Temporal position (same as 3D mRoPE)
  - H dimension: Height position
  - W dimension: Width position

### Under the Hood

When `--debug_pthw_rope True` is set:

1. **Model Loading** (lines 66-67 in inference.py):
   ```python
   model, processor = load_model_and_processor(model_type, model_path, ...)
   ```

2. **Monkey Patch Applied** (lines 69-83):
   ```python
   if model_type == "custom-spatial-mllm" and debug_pthw_rope:
       from src.custom_qwenvl.model.custom_spatial_mllm_pose_rope import patch_model_with_pose_rope
       model = patch_model_with_pose_rope(model, use_pose_rope=True, ...)
   ```

3. **During Forward Pass**:
   - Camera poses extracted from `extrinsics_w2c`
   - Converted to c2w format (4x4 homogeneous matrices)
   - Passed to `custom_get_pose_rope_index()` instead of `custom_get_rope_index()`
   - Pose dimension computed via Lie group scalar index
   - Result: 4D position_ids `(4, B, L)` instead of 3D `(3, B, L)`

## Expected Output

### With 4D Pose RoPE Enabled

```
[Inference] model_type=custom-spatial-mllm, mp4_nframes=16, sample_fps=None
[Inference] 🆕 4D Pose RoPE ENABLED: pose_enc_type=PTHW, scale_factor=16.0
[INFO] Model patched with Pose RoPE support.
[INFO] - use_pose_rope: True
[INFO] - pose_enc_type: PTHW
[INFO] - global_normalize_scale_factor: 16.0
[INFO] - add_offset_in_pose_id: False
[INFO] Model will use 4D Pose-aware RoPE (P+T+H+W)
[Inference] ✅ Monkey patch applied: Model now uses 4D Pose-aware RoPE (P+T+H+W)
```

### With 4D Pose RoPE Disabled (default)

```
[Inference] model_type=custom-spatial-mllm, mp4_nframes=16, sample_fps=None
[Inference] ℹ️  Using standard 3D mRoPE (T+H+W)
```

## Testing & Comparison

### A/B Testing: 3D vs 4D

```bash
# Test 1: Standard 3D mRoPE
bash scripts/evaluation/inference_tso.sh --debug_pthw_rope False > output_3d.txt

# Test 2: 4D Pose RoPE
bash scripts/evaluation/inference_tso.sh --debug_pthw_rope True > output_4d.txt

# Compare results
diff output_3d.txt output_4d.txt
```

### Verify Position IDs Dimension

The debug output during prefill will show:

**3D mRoPE**:
```
Prefill Position_ids:
torch.Size([3, 1, 3158])  # (T, H, W)
```

**4D Pose RoPE**:
```
Prefill Position_ids:
torch.Size([4, 1, 3158])  # (P, T, H, W)
Early Text Position_ids (4D: P,T,H,W):
P: tensor([0., 1., 2., ...])
T: tensor([0., 1., 2., ...])
```

## Requirements

- Model type must be `custom-spatial-mllm`
- Video must have camera poses (extracted by spatial encoder)
- Selected frames JSON should exist in video path (for mRoPE_readaptT)

## Notes

1. **Performance**: 4D Pose RoPE adds minimal computational overhead (~5% in position encoding)
2. **Memory**: Same memory footprint as 3D mRoPE (position_ids are float32 but small)
3. **Compatibility**: Only works with `custom-spatial-mllm` model type
4. **Experimental**: This is a research feature for comparing spatial encoding strategies

## Troubleshooting

### Issue: "AssertionError: selected_frames must be provided"
**Solution**: Ensure `selected_frames.json` exists in your video directory when using mRoPE_readaptT mode.

### Issue: "Camera poses not found"
**Solution**: Video must have associated camera information. Check spatial encoder output.

### Issue: Position IDs shape mismatch
**Solution**: Verify that CustomQwen2_5_VLModel supports 4D position_ids. Check decoder layer implementation.

## References

- Monkey patch implementation: `src/custom_qwenvl/model/custom_spatial_mllm_pose_rope.py`
- 4D RoPE function: `src/custom_qwenvl/model/custom_qwen2_5_VLRoPE.py::custom_get_pose_rope_index`
- Pose distance metrics: `src/utils/pose_distance_metrics.py::compute_lie_scalar_index_torch`
