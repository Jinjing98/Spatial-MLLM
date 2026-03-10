# LVSM Integration Design

**Model:** `CustomSpatialMLLMLVSMForConditionalGeneration`  
**Extends:** `CustomSpatialMLLMForConditionalGeneration` (Qwen2.5-VL-3B only, `hidden_size=2048`)

---

## Pipeline

```
Input Frames [N_in, C, H, W]  +  Target Frames [N_tgt, C, H, W]  (from dataloader)
        │                                  │
        ├──────────── interleaved ──────────┤
        │         [N_in+N_tgt, C, H, W]    │
        ▼                                  │
  ┌─────────────┐                          │
  │  QwenVL ViT │  frozen                  │
  │  input only │  → video_embeds          │
  └──────┬──────┘    [L_vis, 2048]         │
         │                                  │
  ┌──────▼──────┐                          │
  │    VGGT     │  frozen                  │
  │  Spatial    │  ← interleaved frames    │
  │  Encoder    │  → poses for all frames  │
  └──────┬──────┘                          │
         │ split by nvs_is_input_mask       │
         ├────────────────┐                │
     input poses      target poses         │
     [1,N_in,3,4]    [1,N_tgt,3,4]         │
         │                                  │
  ┌──────▼──────────────────────────────┐  │
  │  LVSM image_tokenizer (frozen)      │  │
  │  frames ─►_resize_to_lvsm(256×256)  │  │
  │  posed = RGB(−1..1) + Plücker       │  │
  │  [B, N_in, 9, 256, 256]             │  │
  │  → lvsm_base_context                │  │
  │    [B, N_in*1024, 768]              │  │
  └──────┬──────────────────────────────┘  │
         │                                  │
  ┌──────▼──────────────────────────────┐  │
  │  [Phase 3] LVSMConnector.fuse_for_llm  │
  │  (learnable, GATED residual add)    │  │
  │  temporal pool → spatial pool →     │  │
  │  lvsm_norm + lvsm_proj (768→2048)   │  │
  │  × lvsm_delta_scale (1/√2048)       │  │
  │  + gate=tanh(nvs_gate) × projected  │  │
  │  → fused_embeds [L_vis, 2048]       │  │
  └──────┬──────────────────────────────┘  │
         │                                  │
  ┌──────▼──────┐                          │
  │  Qwen2.5    │  trainable               │
  │  LLM 3B     │  mRoPE                   │
  └──────┬──────┘                          │
         │ hidden_states → extract visual   │
         │ visual_hidden [L_vis, 2048]      │
         │                                  ▼
  ┌──────▼──────────────────────────────────────────────────────┐
  │  [Phase 5] Per-view FiLM Modulation (LVSMConnector)         │
  │                                                              │
  │  extract_per_view_feat(visual_hidden):                       │
  │    group tokens by video_grid_thw → spatial mean-pool        │
  │    → expand temporal merge → view_norm + view_proj (fp32)   │
  │    → per_view_feat [B, N_in, 768]                           │
  │                                                              │
  │  modulate_lvsm_context(lvsm_base_context, per_view_feat):    │
  │    gamma = gamma_head(per_view_feat)  [B, N_in, 768]        │
  │    beta  = beta_head(per_view_feat)   [B, N_in, 768]        │
  │    gate  = tanh(mod_gate)  (init=0 → identity)              │
  │    ctx = base × (1 + gate×gamma) + gate×beta                │
  │    → lvsm_context [B, N_in*1024, 768]                       │
  └──────┬───────────────────────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────────────────────┐
  │  LVSM Transformer (frozen or partly unfrozen)                │
  │  target: _resize_to_lvsm(256×256) + Plücker rays            │
  │  target_pose_cond [1, N_tgt, 6, 256, 256]  (bf16)          │
  │  context + target_tokens → 24-layer transformer             │
  │  → image_token_decoder → rendered [B, N_tgt, 3, 256, 256]  │
  └──────┬───────────────────────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────────────────────────┐
  │  NVS Loss (frozen, fp32)                                     │
  │  L2(rendered, target_gt) + w×Perceptual(VGG19)              │
  └──────┬───────────────────────────────────────────────────────┘
         │
  total_loss = CE_loss + nvs_loss_weight × NVS_loss
```

---

## Modules

| Module | File | Trainable | Role |
|--------|------|-----------|------|
| `LVSMConnector` | `lvsm_connector.py` | **YES** (only new learnable) | Phase-3 LVSM→LLM fusion + Phase-5 FiLM modulation |
| `LVSMDecoderOnly` | `lvsm_wrapper.py` | default frozen; `tune_lvsm_decoder` unfreezes transformer+decoder | 24-layer transformer, image_tokenizer, image_token_decoder |
| `NVSLoss` | `nvs_loss.py` | frozen | L2 + VGG19 perceptual, runs fp32 |
| `ray_utils` | `ray_utils.py` | N/A | w2c→c2w, intrinsics→fxfycxcy, Plücker rays, posed input |

### LVSMConnector internals

**Phase 3 — LVSM→LLM** (`fuse_for_llm`):
- `lvsm_norm`: `LayerNorm(768)`
- `lvsm_proj`: `Linear(768→2048)` — zero-init → identity at step 0
- `lvsm_delta_scale = 1/√2048` — prevents magnitude explosion
- `nvs_gate`: `nn.Parameter(zeros(1))`, applied as `tanh(nvs_gate)`, skipped when `|gate| < 1e-6`

**Phase 5 — LLM→LVSM** (`extract_per_view_feat` + `modulate_lvsm_context`):
- `view_norm`: `LayerNorm(2048)`
- `view_proj`: `Linear(2048→768)` — Xavier uniform init
- `gamma_head`: `Linear(768→768)` — zero-init
- `beta_head`: `Linear(768→768)` — zero-init
- `mod_gate`: `nn.Parameter(zeros(1))`, applied as `tanh(mod_gate)`, hard short-circuit when `< 1e-6`

---

## Key Design Decisions

**1. Per-view FiLM modulation (current) vs. patch-level residual (old)**  
Old: `lvsm_context = cat([base_context, gate * llm_delta])` — patch-level, unstable, hard to train.  
Current: `ctx = base × (1 + gate×γ) + gate×β` where γ, β come from spatially-pooled per-frame LLM features.  
Rationale: forces LLM to encode view-level awareness while keeping LVSM prior stable.

**2. Gate initialization → identity at step 0**  
Both `nvs_gate` and `mod_gate` start at 0 (→ `tanh(0)=0`).  
Phase 3: LVSM contributes 0 to LLM tokens until gate grows.  
Phase 5: FiLM is identity until gate grows. In both cases LVSM starts purely on pretrained context.  
Hard short-circuit (`|gate| < 1e-6`) avoids `0 × NaN = NaN` pathology.

**3. Weight loading order & re-initialization**  
`load_lvsm_checkpoint()` must be called **after** `from_pretrained()`.  
Reason: `post_init()` (called twice — parent + child) re-initializes ALL `nn.Linear` with random normal, corrupting gates and projection heads.  
After loading, explicit re-init:
- `lvsm_norm`, `view_norm` → `(1, 0)` (LayerNorm canonical)
- `lvsm_proj` → `zeros` (identity residual at step 0)
- `view_proj` → Xavier uniform (clean start)
- `gamma_head`, `beta_head` → `zeros` (FiLM identity)
- `nvs_gate`, `mod_gate` → `zeros`

**4. VGG float32 reload**  
`from_pretrained(torch_dtype=bf16)` converts ALL params including frozen VGG weights (float64 → bf16). The float64→bf16→float32 roundtrip causes NaN in deep VGG feature extraction.  
Fix: `reload_vgg_float32()` called inside `load_lvsm_checkpoint()` — reloads original `.mat` weights back to float32 after `to(bf16)`.

**5. Precision strategy**  
- LVSM transformer: **bf16** throughout (required by xformers flash-attn)  
- `extract_per_view_feat` + `modulate_lvsm_context`: **fp32** (functional path, weights not cast in-place)  
- `NVSLoss.forward` + `PerceptualLoss.forward`: **fp32** (`@autocast(enabled=False)`)  
- Target GT images: float32 for NVSLoss; cast to bf16 only for LVSM rays

**6. Image preprocessing: direct resize**  
Current (`_resize_to_lvsm`): direct bilinear resize `H×W → 256×256` with independent `scale_x`, `scale_y`; fx≠fy allowed for non-square inputs (expected for real datasets). Intrinsics adapted via explicit per-axis scaling.  
Old (`_prepare_lvsm_inputs_old`): also direct bilinear resize, but used `intrinsics_to_fxfycxcy()` API (less clean, kept for reference). Both allow fx≠fy; the difference is in API clarity, not behavior.  
Target frames go through the same `_resize_to_lvsm` before NVS loss.

**7. VGGT interleaving for target poses**  
Input + target frames interleaved → single VGGT forward pass for all poses.  
Split via `nvs_is_input_mask` (bool mask, True = input, False = target).

**8. `skip_connector = True` (always)**  
The existing `MLPAddConnector` (spatial_encoder → LLM fusion) is **always bypassed** in LVSM model. `skip_connector` is set to `True` after `post_init()`. The only connector active is `connector_lvsm` (Phase 3/5).

**9. `lvsm_base_context` is detached**  
In `_decode_nvs`, `lvsm_base_context.detach()` is passed to `modulate_lvsm_context`.  
Gradients flow only through the FiLM parameters (connector_lvsm) and back to the LLM — not through the LVSM image_tokenizer. This keeps the tokenizer stable and avoids double-gradient issues.

**10. NaN guards**  
- Context NaN → `assert False` (hard stop, training must not continue with NaN LVSM context)  
- Rendered NaN → `assert False`  
- `nan_to_num` applied defensively on `per_view_feat` before FiLM

---

## Default Hyperparameters

### LVSM Architecture
| Param | Default | Note |
|-------|---------|------|
| `lvsm_d` | 768 | transformer dim (matches `scene_decoder_only_256.pt`) |
| `lvsm_d_head` | 64 | attention head dim |
| `lvsm_n_layer` | 24 | transformer depth |
| `lvsm_use_qk_norm` | True | QK normalization |
| `lvsm_image_size` | 256 | input/output resolution |
| `lvsm_patch_size` | 8 | → 32×32 = 1024 patches/frame |

### Training
| Param | Default | Note |
|-------|---------|------|
| `nvs_loss_weight` | 0.1 | model code default=`1.0`; training script sets `0.1` (0.01→too slow, 1.0→NaN) |
| `num_target_views` | 8 | max novel views per step; more = more stable gradient |
| `lr` (LLM + lvsm_model) | 7e-6 | shared base lr for LLM backbone + LVSM decoder |
| `mm_projector_lr` | 2e-5 | ViT merger + MLPAddConnector |
| `lvsm_adaptor_lr` | 1e-7 | **connector_lvsm only** (FiLM adaptor); lvsm_model uses base lr |
| `lr_scheduler_type` | `"constant"` | no warmup (`warmup_ratio=0.0`) |
| `nvs_target_pool` | `"nvs"` | `"nvs"` novel only / `"input"` reconstruction check / `"all"` mixed |
| `lvsm2qwen_type` | `"linear"` | Phase-3 adapter type (currently only `"linear"`) |
| `llm2lvsm_type` | `"linear"` | Phase-5 adapter type (currently only `"linear"`) |
| `tune_lvsm_decoder` | True | unfreeze `transformer_blocks` + `image_token_decoder` |
| `lvsm_grad_checkpoint` | True | gradient checkpointing inside LVSM transformer |
| `lvsm_grad_checkpoint_every` | 1 | checkpoint every N layers |
| `nvs_img_log_interval` | 2 | steps between wandb rendered image logs |
| `debug_nvs_use_src_views` | False | legacy sanity-check flag (superseded by `nvs_target_pool`) |

### NVS Loss
| Param | Default | Note |
|-------|---------|------|
| `l2_loss_weight` | 1.0 | MSE |
| `perceptual_loss_weight` | 0.5 | VGG19 feature-level |
| `lpips_loss_weight` | 0.0 | disabled |
| `vgg_weight_file` | `submodule/LVSM/metric_checkpoint/imagenet-vgg-verydeep-19.mat` | required |

### Data
| Param | Default | Note |
|-------|---------|------|
| `video_max_frames` / `video_min_frames` | 16 | input views for QwenVL |
| `video_frame_fps` | 4 | |
| `nvs_enabled` | True | dataloader loads novel target frames |
| `sampling_enforce_real_neighbour` | False | temporal-merge-aware neighbour sampling (see below) |
| `neighbour_mode` | `"random"` | `"before"` / `"after"` / `"random"` |
| `neighbour_max_step` | 1 | max frame-ID offset for neighbour sampling |

---

## Data Pipeline (`data_qwen.py`)

### NVS Target Frame Sampling (`_get_nvs_target_frames`)

Called per sample when `nvs_enabled=True`. Re-opens the video to read intermediate frames.

```
video frames:  [f0 ---- f1 ---- f2 ---- f3]   (input, N_in = 4 from process_video)
                   ↑         ↑       ↑
              target_0   target_1  target_2    (1 per gap, randint(lo+1, hi-1))

nvs_is_input_mask: [T, F, T, F, T, F, T]     (True=input, False=target, in temporal order)
```

- Samples exactly one frame per consecutive input-frame gap
- Gaps with no space (`hi - lo < 2`) are skipped → `N_tgt` may be < `N_in - 1`
- Returns `None, None` if no valid gaps exist (sample skipped silently)
- Target frames resized to match input `(H, W)` so VGGT interleaving works

### Real-Neighbour Sampling (`_add_real_neighbours`)

Optional (`sampling_enforce_real_neighbour=True`). Ensures each temporal-merge pair contains genuinely adjacent frames (not just evenly-spaced linspace).

```
anchor_idx (N/2 frames, linspace) → pair each with a real neighbour
  - first anchor  → always "after"   (boundary guard)
  - last anchor   → always "before"
  - middle        → neighbour_mode (default: random)
  - step ~ randint(1, neighbour_max_step) per anchor
  - overlap guard: if neighbour hits adjacent anchor → duplicate anchor instead
```

### Collation (`DataCollatorForSupervisedDataset`)

NVS fields are collated as **lists** (not padded tensors) because `N_tgt` varies per sample:
```python
batch["nvs_target_tchw"]   = List[Tensor[N_tgt, C, H, W]]   # one per sample
batch["nvs_is_input_mask"] = List[Tensor[N_in + N_tgt]]      # bool, one per sample
```
Both fields are absent from `batch` if no sample in the batch has NVS targets.

---

## Optimizer Param Groups

| Group | Modules | LR |
|-------|---------|-----|
| LLM | `model.*` (Qwen2.5 decoder) | `7e-6` |
| Projector | `connector.*`, `lm_head.*` | `2e-5` (`mm_projector_lr`) |
| LVSM decoder | `lvsm_model.*` | `7e-6` (base `lr`) |
| LVSM adaptor | `connector_lvsm.*` | `1e-7` (`lvsm_adaptor_lr`) |

Key: `lvsm_model` (the frozen/partly-unfrozen LVSM transformer) uses the **same base lr** as LLM.  
`connector_lvsm` (FiLM adaptor, the only new learnable module) uses a much smaller `lvsm_adaptor_lr`.  
Trainable flags: `tune_mm_llm=True`, `tune_mm_connector_lvsm=True`, `tune_lvsm_decoder=True`.  
Frozen: `tune_mm_vision=False`, `tune_mm_spatial_encoder=False`, `skip_connector=True`.

---

## LLM Backbone — Custom Decoder Layer

The Qwen2.5 LLM backbone uses a custom decoder layer stack (`CustomQwen2_5_VLDecoderLayer`) with custom attention classes, enabling future extensibility hooks. Currently active features in LVSM training:

| Feature | Status | Config flag |
|---------|--------|-------------|
| Standard mRoPE (3D THW) | **Active** | `position_ids_compute_mode="mRoPE"` |
| PRoPE (Pose-aware RoPE) | Inactive | `RoPE_attn_mode='PRoPE4VisionToken'` |
| SDPA output gating | Inactive | `enable_sdpa_gating=False` |
| 4D mRoPE (PTHW) | Inactive | `USE_POSE_ROPE=False` |

**PRoPE** (when enabled): applies camera-pose transforms to Q/K/V after standard RoPE.  
`Q ← P^T Q`,  `K ← P^{-1} K`,  `V ← P^{-1} V`,  then after attention: `out ← P · out`.  
Effect: attention computed in world-coordinate space.

**SDPA gating** (arxiv 2505.06708): element-wise sigmoid gate after attention output, before `o_proj`.  
`attn_out = attn_out × sigmoid(sdpa_gate(attn_out))`.  
Init: `weight=0, bias=4.0` → `sigmoid(4) ≈ 0.98` → near-identity at step 0.

**Dynamic mRoPE dimensions**: `apply_multimodal_rotary_pos_emb` supports 3D (THW / PHW) and 4D (PTHW) via `len(mrope_section)`.

---

## File Layout
```
src/custom_qwenvl/
├── model/
│   ├── custom_spatial_mllm.py              # base model (no LVSM)
│   ├── custom_spatial_mllm_lvsm.py         # LVSM model (this design)
│   │   ├── _resize_to_lvsm()               # direct resize + intrinsic adapt
│   │   ├── _prepare_lvsm_inputs()          # frames + poses → LVSM tokens
│   │   ├── _upsample_llm_delta_to_lvsm()   # legacy, kept for reference
│   │   ├── _decode_nvs()                   # FiLM modulate → LVSM decode → loss
│   │   ├── load_lvsm_checkpoint()          # weights + re-init + VGG reload
│   │   └── forward()                       # 6-phase pipeline
│   ├── custom_qwen2_5_VLDecoderLayer.py    # custom attention + PRoPE + SDPA gate
│   └── custom_RoPE_utils.py                # apply_poseaware_rotary / output_transform
└── lvsm_utils/
    ├── lvsm_connector.py   # LVSMConnector (only new learnable module)
    ├── lvsm_wrapper.py     # LVSMDecoderOnly + build_lvsm_model()
    ├── nvs_loss.py         # NVSLoss + PerceptualLoss(VGG19) + reload_vgg_float32()
    └── ray_utils.py        # w2c_to_c2w, Plücker rays, posed input
```
