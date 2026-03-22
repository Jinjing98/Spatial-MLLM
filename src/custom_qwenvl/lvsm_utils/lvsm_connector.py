"""
JJ : LVSMConnector — the ONLY new learnable module for LVSM integration.

Bridges QwenVL visual tokens (d=2048) with LVSM tokens (d=768) in two directions:

  Phase-3  (LVSM → LLM):   fuse_for_llm()
  Phase-5  (LLM → LVSM):   patch residual via
                            project_visual_tokens_to_lvsm() + fuse_patch_residual_context()

Design note (patch residual):
  LLM visual tokens are projected to LVSM channels and upsampled to LVSM patch grid,
  then fused as gated residual on top of frozen lvsm_base_context.
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
        vlm2context_adapt_strategy="patch_residual",
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
        self.lvsm2llm_in_dim = self.d_lvsm
        self.lvsm2qwen_type = lvsm2qwen_type
        self.llm2lvsm_type = llm2lvsm_type
        self.vlm2context_adapt_strategy = vlm2context_adapt_strategy

        # ============================================================
        # [B] Phase-3: LVSM → QwenVL  (fuse_for_llm)
        # ============================================================
        self._init_lvsm2qwen_adapter()

        # ============================================================
        # [C] Phase-5: LLM → LVSM
        # ============================================================
        self._init_llm2lvsm_adapter()

    def _init_lvsm2qwen_adapter(self) -> None:
        if self.lvsm2qwen_type == "linear":
            self.lvsm_norm = nn.LayerNorm(self.d_lvsm)
            self.lvsm_proj = nn.Linear(self.d_lvsm, self.d_qwen)
        else:
            raise NotImplementedError(
                f"lvsm2qwen_type='{self.lvsm2qwen_type}' not supported. Choose from: 'linear'."
            )
        self.lvsm_delta_scale = 1.0 / math.sqrt(self.d_qwen)
        # JJ : Gates LVSM→LLM contribution; small init so gradient flows from step 0.
        self.nvs_gate = nn.Parameter(torch.full((1,), 0.1))

    def _init_llm2lvsm_adapter(self) -> None:
        # Keep attributes explicit for readability and checkpoint compatibility.
        self.view_norm = None
        self.view_proj = None
        self.gamma_head = None
        self.beta_head = None

        if self.llm2lvsm_type == "linear":
            # Per-view feature extraction: pool + project visual_hidden -> [B, N, d_lvsm]
            self.view_norm = nn.LayerNorm(self.d_qwen)
            self.view_proj = nn.Linear(self.d_qwen, self.d_lvsm)
        elif self.llm2lvsm_type == "cross_attn":
            self._init_cross_attn_placeholder()
        else:
            raise NotImplementedError(
                f"llm2lvsm_type='{self.llm2lvsm_type}' not supported. "
                f"Choose from: 'linear', 'cross_attn'."
            )

        if self.vlm2context_adapt_strategy != "patch_residual":
            raise NotImplementedError(
                f"vlm2context_adapt_strategy='{self.vlm2context_adapt_strategy}' not supported. "
                f"Only 'patch_residual' is supported."
            )

        # Shared modulation gate for Phase-5 adaptation branches.
        self.mod_gate = nn.Parameter(torch.full((1,), 0.1))

    def _init_cross_attn_placeholder(self) -> None:
        # Standalone cross-attention branch: placeholder only for now.
        self.cross_attn = None
        raise NotImplementedError(
            "llm2lvsm_type='cross_attn' is reserved as a standalone branch and not implemented yet."
        )

    @torch.amp.autocast('cuda', enabled=False)
    def project_visual_tokens_to_lvsm(self, visual_hidden):
        """
        Project flat LLM visual tokens to LVSM channel dimension without pooling.

        TODO: Stronger patch_residual ablation to try here:
        replace the current token-wise `LN + Linear(2048->768)` before spatial
        upsampling with `LN -> bilinear upsample -> projection`, so the branch
        restores 2D layout first and compresses channels afterward.

        Args:
            visual_hidden: [L_vis, d_qwen] or [B, L_vis, d_qwen]

        Returns:
            llm_delta: [B, L_vis, d_lvsm] (fp32)
        """
        if self.llm2lvsm_type != "linear" or self.view_norm is None or self.view_proj is None:
            raise RuntimeError(
                "project_visual_tokens_to_lvsm requires llm2lvsm_type='linear' with initialized view_norm/view_proj."
            )

        if visual_hidden.dim() == 2:
            visual_hidden = visual_hidden.unsqueeze(0)  # [1, L_vis, d_qwen]

        x = visual_hidden.float()
        w_ln = self.view_norm.weight.float()
        b_ln = self.view_norm.bias.float() if self.view_norm.bias is not None else None
        normed = F.layer_norm(
            x,
            normalized_shape=(self.d_qwen,),
            weight=w_ln,
            bias=b_ln,
            eps=self.view_norm.eps,
        )
        w_proj = self.view_proj.weight.float()
        b_proj = self.view_proj.bias.float() if self.view_proj.bias is not None else None
        llm_delta = F.linear(normed, w_proj, b_proj)
        return torch.nan_to_num(llm_delta, nan=0.0, posinf=0.0, neginf=0.0)

    def fuse_patch_residual_context(self, lvsm_base_context, patch_delta):
        """
        Patch-wise residual fusion:
            ctx = base + tanh(mod_gate) * patch_delta

        Args:
            lvsm_base_context: [B, N*P, d_lvsm]
            patch_delta:       [B, N*P, d_lvsm]

        Returns:
            lvsm_context:      [B, N*P, d_lvsm]
        """
        if lvsm_base_context.shape != patch_delta.shape:
            raise ValueError(
                f"[patch_residual] shape mismatch: base={tuple(lvsm_base_context.shape)} "
                f"delta={tuple(patch_delta.shape)}"
            )

        gate = torch.nan_to_num(torch.tanh(self.mod_gate.float()), nan=0.0)
        if gate.abs().item() < 1e-6:
            return lvsm_base_context

        base = lvsm_base_context.float()
        delta = torch.nan_to_num(patch_delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
        ctx = base + gate * delta
        return ctx.to(lvsm_base_context.dtype)

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
    # Utility
    # ==================================================================
    def print_trainable_parameters(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LVSMConnector] Total: {total:,}, Trainable: {trainable:,}")
