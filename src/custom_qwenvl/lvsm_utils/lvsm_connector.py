"""
JJ : LVSMConnector — the ONLY new learnable module for LVSM integration.

Bridges QwenVL visual tokens (d=2048) with LVSM tokens (d=768) in two directions:

  Phase-3  (LVSM → LLM):   fuse_for_llm()
  Phase-5  (LLM → LVSM):   per-view modulation via
                            extract_per_view_feat()  +  modulate_lvsm_context()

Design note (per-view modulation):
  Instead of mapping LLM hidden → patch-level LVSM delta (hard + unstable),
  we modulate the frozen lvsm_base_context *per view* with gamma/beta derived
  from spatially-pooled LLM features.  This forces the LLM to encode view-level
  awareness while keeping the LVSM prior stable.
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LVSMConnector(nn.Module):
    """
    JJ : Token fusion connector between QwenVL and LVSM.

    Sections:
      [A] Shared config / constants
      [B] Phase-3 modules  — LVSM → LLM  (fuse_for_llm)
      [C] Phase-5 modules  — LLM → LVSM  (per-view modulation)
    """

    def __init__(
        self,
        d_lvsm=768,
        d_qwen=2048,
        temporal_patch_size=2,
        spatial_merge_size=2,
        lvsm_spatial_size=32,  # 256 / 8 = 32 patches per spatial dim
        lvsm2qwen_type="linear",
        llm2lvsm_type="linear",
    ):
        super().__init__()

        # ============================================================
        # [A] Shared config
        # ============================================================
        self.d_lvsm = d_lvsm
        self.d_qwen = d_qwen
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.lvsm_spatial_size = lvsm_spatial_size
        self.lvsm_n_patches = lvsm_spatial_size ** 2  # 1024 patches per frame
        self.lvsm2qwen_type = lvsm2qwen_type
        self.llm2lvsm_type = llm2lvsm_type

        # ============================================================
        # [B] Phase-3: LVSM → QwenVL  (fuse_for_llm)
        # ============================================================
        if lvsm2qwen_type == "linear":
            self.lvsm_norm = nn.LayerNorm(d_lvsm)
            self.lvsm_proj = nn.Linear(d_lvsm, d_qwen)
        else:
            raise NotImplementedError(
                f"lvsm2qwen_type='{lvsm2qwen_type}' not supported. Choose from: 'linear'."
            )
        self.lvsm_delta_scale = 1.0 / math.sqrt(d_qwen)
        # JJ : gates LVSM→LLM contribution; 0-init → identity at start
        self.nvs_gate = nn.Parameter(torch.zeros(1))

        # ============================================================
        # [C] Phase-5: LLM → LVSM  (per-view FiLM modulation)
        # ============================================================
        if llm2lvsm_type == "linear":
            # C.1  Per-view feature extraction: pool + project visual_hidden → [B, N, d_lvsm]
            self.view_norm = nn.LayerNorm(d_qwen)
            self.view_proj = nn.Linear(d_qwen, d_lvsm)
        else:
            raise NotImplementedError(
                f"llm2lvsm_type='{llm2lvsm_type}' not supported. Choose from: 'linear'."
            )
        # C.2  FiLM heads: generate gamma / beta per view
        self.gamma_head = nn.Linear(d_lvsm, d_lvsm)
        self.beta_head = nn.Linear(d_lvsm, d_lvsm)
        # C.3  Modulation gate; 0-init → identity (base only) at start
        self.mod_gate = nn.Parameter(torch.zeros(1))

    # ==================================================================
    # Phase-3: LVSM → LLM
    # ==================================================================
    def fuse_for_llm(self, video_embeds, lvsm_tokens, video_grid_thw):
        """
        JJ : Fuse LVSM tokens into QwenVL visual tokens for LLM input.

        Pipeline:
          A) Temporal pooling  — mirrors QwenVL temporal merge
          B) 2D spatial pooling per grid entry
          C) Norm + project 768 → 2048
          D) Gated residual add to video_embeds

        Args:
            video_embeds:   [L_vis, d_qwen]           QwenVL visual tokens (flat)
            lvsm_tokens:    [B, N*n_patches, d_lvsm]  LVSM input tokens
            video_grid_thw: [num_grids, 3]             (t_merged, h_pre, w_pre)

        Returns:
            fused_embeds:   [L_vis, d_qwen]
        """
        B = lvsm_tokens.shape[0]
        N = lvsm_tokens.shape[1] // self.lvsm_n_patches

        # A: Temporal pooling
        lvsm_tokens = lvsm_tokens.view(B, N, self.lvsm_n_patches, self.d_lvsm)
        T_merged = N // self.temporal_patch_size
        lvsm_tokens = lvsm_tokens.view(
            B, T_merged, self.temporal_patch_size, self.lvsm_n_patches, self.d_lvsm
        ).mean(dim=2)  # [B, T_merged, 1024, 768]

        # B: 2D spatial adaptive pool per grid entry
        all_pooled = []
        t_offset = 0
        for grid_idx in range(video_grid_thw.shape[0]):
            t_grid, h_pre, w_pre = (int(x) for x in video_grid_thw[grid_idx].tolist())
            h_out = h_pre // self.spatial_merge_size
            w_out = w_pre // self.spatial_merge_size

            chunk = lvsm_tokens[:, t_offset:t_offset + t_grid]
            spatial = chunk.reshape(
                B * t_grid, self.lvsm_spatial_size, self.lvsm_spatial_size, self.d_lvsm
            ).permute(0, 3, 1, 2).contiguous()
            pooled = F.adaptive_avg_pool2d(spatial, (h_out, w_out))
            pooled = pooled.permute(0, 2, 3, 1).reshape(B, t_grid * h_out * w_out, self.d_lvsm)
            all_pooled.append(pooled)
            t_offset += t_grid

        lvsm_aligned = torch.cat(all_pooled, dim=1)  # [B, L_vis, 768]

        # C: Norm + project
        lvsm_projected = self.lvsm_proj(self.lvsm_norm(lvsm_aligned))
        lvsm_projected = (lvsm_projected * self.lvsm_delta_scale).reshape(-1, self.d_qwen)

        # D: Gated residual add
        gate = torch.nan_to_num(torch.tanh(self.nvs_gate), nan=0.0)
        if gate.abs() > 1e-6:
            fused_embeds = video_embeds + gate * lvsm_projected
        else:
            fused_embeds = video_embeds
        return fused_embeds

    # ==================================================================
    # Phase-5: LLM → LVSM  (per-view modulation)
    # ==================================================================
    @torch.amp.autocast('cuda', enabled=False)
    def extract_per_view_feat(self, visual_hidden, video_grid_thw):
        """
        JJ : Extract one feature vector per raw frame from LLM visual hidden states.

        Forced fp32 throughout to avoid bf16 LayerNorm / Linear instability.

        Steps:
          1. Group tokens by grid entry (respecting video_grid_thw)
          2. Spatial mean-pool each merged frame  → [B, T_merged, d_qwen]
          3. Expand temporal merge  → [B, N_raw, d_qwen]
          4. Norm + project  → [B, N_raw, d_lvsm]

        Args:
            visual_hidden:  [L_vis, d_qwen]  or  [B, L_vis, d_qwen]
            video_grid_thw: [num_grids, 3]   (t_merged, h_pre, w_pre)

        Returns:
            per_view_feat:  [B, N_raw, d_lvsm]  (fp32)
        """
        if visual_hidden.dim() == 2:
            visual_hidden = visual_hidden.unsqueeze(0)  # [1, L_vis, d_qwen]

        # JJ : Force fp32 for numerical stability in this small branch
        x = visual_hidden.float()
        B = x.shape[0]
        sp = self.spatial_merge_size
        tp = self.temporal_patch_size

        # Step 1-2: Group by grid entry, spatial mean-pool per merged frame
        per_merged_frame = []
        tok = 0
        for grid_idx in range(video_grid_thw.shape[0]):
            t_m, h_p, w_p = (int(v) for v in video_grid_thw[grid_idx].tolist())
            h_o, w_o = h_p // sp, w_p // sp
            n_tok = t_m * h_o * w_o

            chunk = x[:, tok:tok + n_tok]
            chunk = chunk.reshape(B, t_m, h_o * w_o, self.d_qwen)
            pooled = chunk.mean(dim=2)                            # [B, t_m, d_qwen]
            per_merged_frame.append(pooled)
            tok += n_tok

        merged_feat = torch.cat(per_merged_frame, dim=1)         # [B, T_merged_total, d_qwen]

        # Step 3: Expand temporal merge → [B, N_raw, d_qwen]
        raw_feat = merged_feat.unsqueeze(2).expand(-1, -1, tp, -1)
        raw_feat = raw_feat.reshape(B, -1, self.d_qwen)          # [B, N_raw, d_qwen]

        # Step 4: Norm + project → [B, N_raw, d_lvsm]  (fp32, functional)
        # JJ : Use F.layer_norm / F.linear with .float() on weights only,
        #      so module parameters stay in their original dtype (no in-place cast).
        _rf = raw_feat.detach()
        # logger.warning(
        #     f"[DIAG-PVF] raw_feat finite={torch.isfinite(_rf).all().item()} "
        #     f"min={_rf.min().item():.4f} max={_rf.max().item():.4f} "
        #     f"mean={_rf.mean().item():.4f} std={_rf.std().item():.4f}")

        w_ln = self.view_norm.weight.float()
        b_ln = self.view_norm.bias.float() if self.view_norm.bias is not None else None
        normed = F.layer_norm(
            raw_feat.float(),
            normalized_shape=(self.d_qwen,),
            weight=w_ln,
            bias=b_ln,
            eps=self.view_norm.eps,
        )
        _n = normed.detach()
        # logger.warning(
        #     f"[DIAG-PVF] after view_norm finite={torch.isfinite(_n).all().item()} "
        #     f"min={_n.min().item():.4f} max={_n.max().item():.4f} "
        #     f"view_norm.weight finite={torch.isfinite(self.view_norm.weight).all().item()} "
        #     f"view_norm.weight range=[{self.view_norm.weight.min().item():.4f}, {self.view_norm.weight.max().item():.4f}]")

        w_proj = self.view_proj.weight.float()
        b_proj = self.view_proj.bias.float() if self.view_proj.bias is not None else None
        per_view_feat = F.linear(normed, w_proj, b_proj)
        _p = per_view_feat.detach()
        _pf = torch.nan_to_num(_p)
        # logger.warning(
        #     f"[DIAG-PVF] after view_proj finite={torch.isfinite(_p).all().item()} "
        #     f"min={_pf.min().item():.4f} max={_pf.max().item():.4f} "
        #     f"view_proj.weight finite={torch.isfinite(self.view_proj.weight).all().item()} "
        #     f"view_proj.weight norm={self.view_proj.weight.norm().item():.4f}")

        return per_view_feat

    def modulate_lvsm_context(self, lvsm_base_context, per_view_feat):
        """
        JJ : FiLM-style per-view modulation of lvsm_base_context.

        ctx[b, n, p, :] = base[b, n, p, :] * (1 + gate * gamma[b, n, :])
                                             +      gate * beta[b, n, :]

        At init (gate=0): hard short-circuit → return base unchanged (true identity).
        As gate grows:  LLM increasingly shapes per-view statistics.

        Args:
            lvsm_base_context: [B, N*P, d_lvsm]   pretrained LVSM input tokens (detached)
            per_view_feat:     [B, N, d_lvsm]      from extract_per_view_feat()

        Returns:
            lvsm_context:      [B, N*P, d_lvsm]   modulated context for LVSM decoder
        """
        # JJ : Hard short-circuit when gate ≈ 0 — avoids 0 * NaN = NaN
        gate = torch.nan_to_num(torch.tanh(self.mod_gate.float()), nan=0.0)
        if gate.abs().item() < 1e-6:
            return lvsm_base_context

        B = lvsm_base_context.shape[0]
        P = self.lvsm_n_patches
        N = lvsm_base_context.shape[1] // P
        d = self.d_lvsm

        assert per_view_feat.shape == (B, N, d), (
            f"[modulate] per_view_feat shape mismatch: "
            f"got {per_view_feat.shape}, expected ({B}, {N}, {d})"
        )

        # JJ : Sanitize per_view_feat — NaN/Inf → 0 as last-resort guard
        feat = torch.nan_to_num(per_view_feat.float(), nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape base: [B, N, P, d]
        base = lvsm_base_context.float().view(B, N, P, d)

        # FiLM parameters: [B, N, 1, d]  (broadcast over P patches)
        # JJ : functional fp32 — no in-place module cast
        gamma = F.linear(feat, self.gamma_head.weight.float(),
                         self.gamma_head.bias.float()).unsqueeze(2)   # [B, N, 1, d]
        beta = F.linear(feat, self.beta_head.weight.float(),
                        self.beta_head.bias.float()).unsqueeze(2)     # [B, N, 1, d]

        # Gated modulation (fp32)
        ctx = base * (1.0 + gate * gamma) + gate * beta

        return ctx.reshape(B, N * P, d).to(lvsm_base_context.dtype)

    # ==================================================================
    # Utility
    # ==================================================================
    def print_trainable_parameters(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LVSMConnector] Total: {total:,}, Trainable: {trainable:,}")
