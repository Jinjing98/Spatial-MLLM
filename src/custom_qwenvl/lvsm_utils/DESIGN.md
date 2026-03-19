# LVSM Integration Design (Current Staged Version)

**Model**: `CustomSpatialMLLMLVSMForConditionalGeneration`  
**Base**: `CustomSpatialMLLMForConditionalGeneration`  
**Scope**: Qwen2.5-VL-3B path (`hidden_size=2048`)

---

## 0) Pipeline Docs

- 训练/评测/可视化的端到端维护流程见：
  - `scripts/evaluation/USAGE_lvsm_vlm_visualization_pipeline.md`

## 1) What Changed (vs previous doc)

- Phase-5 context adaptation is now **patch residual**, not FiLM.
- Dataset supports **precomputed pose/index package** (`.pt`) and forwards camera tensors into model.
- NVS sampling logic updated to a **uniform interior novel-frame policy** with short-video fallback.
- Training observability expanded (bridge grad norm callback + gate/delta diagnostics + SDPA gate stats).
- Current staged code contains several **debug switches** that alter behavior and must be treated explicitly.

---

## 2) Complete LVSM+VLM Network Flow

### 2.1 Full Plot Pipeline (maintained view)

```
Input Frames [N_in, C, H, W]  +  Target Frames [N_tgt, C, H, W] (optional, for NVS)
        │                                  │
        ├──────────── interleaved (if NVS) ───────────┤
        │             [N_in + N_tgt, C, H, W]         │
        ▼                                              │
  ┌───────────────┐                                    │
  │ QwenVL ViT    │  (frozen)                          │
  │ input frames  │  -> video_embeds [L_vis, d_qwen]  │
  └──────┬────────┘                                    │
         │                                             │
  ┌──────▼────────────────┐                            │
  │ VGGT spatial encoder  │ (pose path)               │
  │ - online cam decode OR│                            │
  │ - precomputed cams    │                            │
  └──────┬────────────────┘                            │
         │ split by nvs_is_input_mask                 │
         ├──────────────────┐                         │
     input poses        target poses                  │
   [1,N_in,3,4]/[3,3] [1,N_tgt,3,4]/[3,3]            │
         │                                             │
  ┌──────▼──────────────────────────────────────────┐  │
  │ LVSM image_tokenizer path (prepare_lvsm_inputs) │  │
  │ frame resize + Plucker rays                     │  │
  │ -> lvsm_base_context [B, N_in*P, d_lvsm]        │  │
  └──────┬──────────────────────────────────────────┘  │
         │                                             │
  ┌──────▼──────────────────────────────────────────┐  │
  │ Phase-3: LVSM -> LLM fusion (connector_lvsm)    │  │
  │ fuse_for_llm(video_embeds, lvsm_tokens, grid)   │  │
  │ -> fused_embeds [L_vis, d_qwen]                 │  │
  └──────┬──────────────────────────────────────────┘  │
         │                                             │
  ┌──────▼────────────┐                                │
  │ Qwen LLM forward  │                                │
  │ -> hidden states  │                                │
  └──────┬────────────┘                                │
         │ visual token extraction                     │
         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ Phase-5: LLM -> LVSM context adaptation                      │
  │ source A: LLM hidden states                                  │
  │ source B: VGGT geo token bridge (debug/ablation)             │
  │                                                               │
  │ project_visual_tokens_to_lvsm                                 │
  │   -> llm_delta [B, L_vis, d_lvsm]                             │
  │ upsample_llm_delta_to_lvsm_grid                               │
  │   -> patch_delta [B, N_in*P, d_lvsm]                          │
  │ default: fuse with context tokens                             │
  │   lvsm_context = fuse_patch_residual_context(base, patch_delta)│
  │ debug: no context fusion                                      │
  │   lvsm_context = patch_delta                                   │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
  ┌──────────────────────────────▼───────────────────────────────┐
  │ LVSM decoder-only transformer                                │
  │ (frozen or partial unfreeze by tune_lvsm_decoder)            │
  │ context + target pose cond -> rendered [B, N_tgt, 3, H, W]   │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
  ┌──────────────────────────────▼───────────────────────────────┐
  │ NVS Loss (L2 + Perceptual + optional LPIPS)                  │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
               total_loss = CE + nvs_loss_weight * NVS
               (or NVS-only when debug switch is enabled)
```

### 2.2 Current fast summary (implementation view)

```
video_tchw + optional nvs_target_tchw (+ nvs_is_input_mask)
    -> Qwen visual encoder
    -> optional Phase-3 LVSM->LLM fusion
    -> Qwen LLM forward
    -> visual_hidden source (LLM or VGGT-geo bridge)
    -> Phase-5 patch residual adaptation
    -> LVSM decode
    -> NVS loss
    -> final loss
```

---

## 3) Core Modules

| Module | File | Trainable | Role |
|---|---|---|---|
| `LVSMConnector` | `src/custom_qwenvl/lvsm_utils/lvsm_connector.py` | Yes | Phase-3 LVSM->LLM fusion + Phase-5 patch residual adaptation |
| `lvsm_model` | `src/custom_qwenvl/lvsm_utils/lvsm_wrapper.py` | Partly (controlled by `tune_lvsm_decoder`) | Decoder-only LVSM for render |
| `NVSLoss` | `src/custom_qwenvl/lvsm_utils/nvs_loss.py` | No | Render supervision |
| Dataset + Collator | `src/qwenvl/data/data_qwen.py` | N/A | Frame sampling, NVS target building, precomputed camera forwarding |

---

## 4) LVSMConnector (Current Behavior)

### Phase-3: LVSM -> LLM

- `lvsm_norm` + `lvsm_proj` + residual add into visual tokens.
- gate: `nvs_gate` (current staged init is small non-zero, then `tanh` effective gate).
- `lvsm_delta_scale = 1/sqrt(d_qwen)` to control magnitude.

### Phase-5: LLM -> LVSM (patch residual)

- `project_visual_tokens_to_lvsm(visual_hidden)`:
  - fp32 path with `LayerNorm + Linear` per token.
- `_upsample_llm_delta_to_lvsm(...)`:
  - recover LVSM patch layout from merged visual token layout.
- `fuse_patch_residual_context(base, delta)`:
  - `ctx = base + tanh(mod_gate) * delta`.
- Runtime branch:
  - default behavior: fuse `patch_delta` with `lvsm_base_context` (context-token fusion enabled).
  - debug behavior (`disable_llm2lvsm_fusion=True`): bypass fusion and use `lvsm_context = patch_delta`.

`vlm2context_adapt_strategy` currently only supports `"patch_residual"` in staged model path.

---

## 5) Precomputed Pose Data Path

### Dataset side

When `use_pre_compute_pose=True`:

- Resolve `.pt` by `video` relative path under `precompute_pose_root`.
- Required keys:
  - `input_frame_indices`
  - `novel_pool_indices`
  - `nvs_is_input_mask`
  - `input_extrinsics_w2c`, `input_intrinsics`
  - `target_extrinsics_w2c`, `target_intrinsics`
- Use precomputed input indices directly in `process_video(..., frame_idx_override=...)`.
- Build NVS targets from `novel_pool_indices` instead of online random sampling.
- Forward camera tensors via collator fields:
  - `precomputed_input_extrinsics_w2c`
  - `precomputed_input_intrinsics`
  - `precomputed_target_extrinsics_w2c`
  - `precomputed_target_intrinsics`

### Model side

In `forward`:

- If precomputed camera tensors exist, they replace online pose decode path.
- Optional debug compare can run online VGGT and print max/mean abs diff.
- Includes fail-fast count consistency checks (input frames, target count, camera count).

---

## 6) NVS Target Sampling (Online Mode)

In non-precompute mode, `_get_nvs_target_frames` now:

- Samples novel frames from interior range `[first_input+1, last_input-1]`, excluding input indices.
- Targets `N_in` samples when possible; warns if fewer are available.
- If no valid novel frame exists, creates a **dummy target fallback** to avoid no-gradient steps for LVSM-related params under DDP.

Also, frame count is enforced divisible by `temporal_patch_size` after deduplication.

---

## 7) Training / Optimizer / Diagnostics

### Argument additions

- Model args:
  - `vlm2context_adapt_strategy`
- Data args:
  - `max_train_samples`
  - `use_pre_compute_pose`
  - `precompute_pose_source`
  - `precompute_pose_root`

### Optimizer grouping

`lvsm_adaptor_lr` bucket now includes:

- `connector_lvsm.*`
- optional `vggt_geo_norm.*`
- optional `vggt_geo_proj.*`

### Diagnostics

- `LVSMBridgeGradMonitorCallback` logs grad norms for:
  - `lvsm2llm`
  - `llm2lvsm`
  - `vggt2qwen` bridge (if enabled)
- Model logs:
  - raw gates (`nvs_gate`, `mod_gate`)
  - activation-space delta ratios (`lvsm2llm_delta_ratio`, `llm2lvsm_delta_ratio`)
  - SDPA gate aggregate stats (if layer has `sdpa_gate`)

### DDP compatibility

- When `gradient_checkpointing=True`, trainer sets:
  - `gradient_checkpointing_kwargs = {"use_reentrant": False}`

---

## 8) Current Debug/Ablation Switches In Staged Model

The following staged defaults materially change behavior and are **not neutral**:

- `disable_lvsm2llm_fusion = True`
- `disable_llm2lvsm_fusion = True`
- `nvs_loss_only = True`
- `decoder_input_vggt_geo = True`

Interpretation:

- Model may bypass intended fusion/adaptation path and optimize NVS-only objective.
- This is useful for targeted diagnosis, but should be explicitly toggled before formal runs.
- In particular, `disable_llm2lvsm_fusion=True` means Phase-5 no longer uses base LVSM context-token fusion; it directly feeds `patch_delta` as decoder context.

---

## 9) Known Risks / TODO

- Patch-residual branch currently relies on upsample quality; stronger refinement head is still TODO.
- `llm2lvsm_type='cross_attn'` is reserved but intentionally not implemented.
- Large latency share still concentrates in spatial encoder / NVS decode for multi-view settings.
- Keep precompute and online sampling configs aligned (frame count, neighbour policy, interval assumptions).

---

## 10) Related Scripts Added/Updated

- Precompute:
  - `scripts/preprocess/precompute_cam_info_dataset_vggt.py`
  - `scripts/preprocess/precompute_cam_info_full_dataset_vggt.sh`
  - `scripts/preprocess/precompute_cam_info_full_dataset_vggt_torchrun_hpc.sh`
- Training:
  - `scripts/training/spatial_mllm_train_demo_DDP_LVSM_hpc.sh`
  - `scripts/training/spatial_mllm_train_demo_DDP_LVSM_overfitting_hpc.sh`
- Evaluation:
  - `scripts/evaluation/evaluate_vsibench_multiturn_qwen25.sh`
  - `scripts/evaluation/inference_hpc.sh`

This document should be kept consistent with staged behavior first, then refined once debug switches are finalized.
