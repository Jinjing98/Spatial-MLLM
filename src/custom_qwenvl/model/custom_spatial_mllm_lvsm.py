"""
JJ : Custom Spatial MLLM with LVSM integration for Novel View Synthesis.

Extends CustomSpatialMLLMForConditionalGeneration with:
  - Frozen LVSM decoder-only model (pretrained)
  - Learnable connector_lvsm (ONLY new learnable module)
  - NVS loss (L2 + Perceptual) on rendered target views
  - enforce_LVSM flag for fallback to base model

Architecture flow:
  Phase 1: QwenVL ViT + VGGT spatial encoding (existing)
  Phase 2: LVSM input tokenization (Plücker rays + images → tokens)
  Phase 3: Token fusion via connector_lvsm (LVSM tokens → pool → project → add to QwenVL)
  Phase 4: LLM causal self-attention (existing, no modification)
  Phase 5: NVS decoding (LLM visual hidden → project → LVSM transformer → rendered images)
  Phase 6: Combined loss = CE_loss + nvs_loss_weight * NVS_loss

Only supports Qwen2.5-VL-3B. Other model sizes raise NotImplementedError.
"""

import logging
import os
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

logger = logging.getLogger(__name__)

from src.qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri

# JJ : Import base model
from src.custom_qwenvl.model.custom_spatial_mllm import (
    CustomSpatialMLLMConfig,
    CustomSpatialMLLMForConditionalGeneration,
)

# JJ : Import LVSM utilities
from src.custom_qwenvl.lvsm_utils.ray_utils import (
    w2c_to_c2w,
    intrinsics_to_fxfycxcy,
    compute_plucker_rays,
    get_posed_input,
)
from src.custom_qwenvl.lvsm_utils.lvsm_connector import LVSMConnector
from src.custom_qwenvl.lvsm_utils.lvsm_wrapper import build_lvsm_model
from src.custom_qwenvl.lvsm_utils.nvs_loss import NVSLoss

# JJ : Camera temporal downsampling (reuse existing)
from src.custom_qwenvl.model.camera_pose_temporal_merge import downsample_cams
from src.custom_qwenvl.model.custom_qwen2_5_VLRoPE import custom_get_rope_index


# ============================================================================
# Config
# ============================================================================
class CustomSpatialMLLMLVSMConfig(CustomSpatialMLLMConfig):
    """JJ : Config for LVSM-integrated Spatial MLLM."""
    model_type = "custom-spatial-mllm-lvsm"

    def __init__(self, spatial_config=None, connector_config=None, lvsm_config=None, **kwargs):
        super().__init__(spatial_config=spatial_config, connector_config=connector_config, **kwargs)
        self.lvsm_config = lvsm_config if lvsm_config is not None else {}


# ============================================================================
# Model
# ============================================================================
class CustomSpatialMLLMLVSMForConditionalGeneration(CustomSpatialMLLMForConditionalGeneration):
    """
    JJ : Spatial MLLM with LVSM integration for joint QA + NVS training.
    
    Inherits all functionality from CustomSpatialMLLMForConditionalGeneration,
    adds LVSM components and NVS loss pathway.
    """
    
    def __init__(self, config):
        super().__init__(config)        
        
        lvsm_cfg = getattr(config, 'lvsm_config', {})
        
        # JJ : Only 3B model supported
        if config.hidden_size != 2048:
            raise NotImplementedError(
                f"LVSM integration only supports Qwen2.5-VL-3B (hidden_size=2048). "
                f"Got hidden_size={config.hidden_size}."
            )
        
        # --- LVSM config (defaults match official LVSM decoder-only) ---
        self.enforce_LVSM = lvsm_cfg.get('enforce_LVSM', True)
        self.nvs_loss_weight = lvsm_cfg.get('nvs_loss_weight', 1.0)
        self.lvsm_image_size = lvsm_cfg.get('lvsm_image_size', 256)
        self.lvsm_patch_size = lvsm_cfg.get('lvsm_patch_size', 8)
        self.num_target_views = lvsm_cfg.get('num_target_views', 4)
        self.lvsm_grad_checkpoint = lvsm_cfg.get('lvsm_grad_checkpoint', True)
        self.lvsm_grad_checkpoint_every = lvsm_cfg.get('lvsm_grad_checkpoint_every', 1)
        self.lvsm_d = lvsm_cfg.get('lvsm_d', 768)
        self.nvs_img_log_interval = lvsm_cfg.get('nvs_img_log_interval', 50)
        # JJ : NVS target pool — 'nvs' (novel only), 'input' (reconstruction), 'all' (mixed)
        self.nvs_target_pool = lvsm_cfg.get('nvs_target_pool', 'nvs')
        # JJ : Adapter types for LVSM ↔ QwenVL bridges
        self.lvsm2qwen_type = lvsm_cfg.get('lvsm2qwen_type', 'linear')
        self.llm2lvsm_type = lvsm_cfg.get('llm2lvsm_type', 'linear')
        self.vlm2context_adapt_strategy = lvsm_cfg.get('vlm2context_adapt_strategy', 'patch_residual')
        if self.vlm2context_adapt_strategy != 'patch_residual':
            raise NotImplementedError(
                f"vlm2context_adapt_strategy='{self.vlm2context_adapt_strategy}' not supported. "
                "Only 'patch_residual' is supported."
            )
        
        # --- LVSM decoder-only model (structure only, NO checkpoint loading) ---
        # JJ : Checkpoint must be loaded AFTER from_pretrained returns, because
        #      post_init() calls _init_weights which re-initializes all nn.Linear
        #      modules with random values, destroying any weights loaded here.
        #      Same pattern as VGGT spatial_encoder weight loading.
        self._lvsm_checkpoint_path = lvsm_cfg.get('lvsm_checkpoint_path', None)
        self.lvsm_model = build_lvsm_model(
            checkpoint_path=None,  # JJ : Do NOT load here, post_init will overwrite
            d=self.lvsm_d,
            d_head=lvsm_cfg.get('lvsm_d_head', 64),
            n_layer=lvsm_cfg.get('lvsm_n_layer', 24),
            use_qk_norm=lvsm_cfg.get('lvsm_use_qk_norm', True),
            image_size=self.lvsm_image_size,
            patch_size=self.lvsm_patch_size,
        )
        print(f"[INFO] LVSM model structure built. enforce_LVSM={self.enforce_LVSM}, "
              f"d={self.lvsm_d}, image_size={self.lvsm_image_size}, patch_size={self.lvsm_patch_size}"
              f" (checkpoint will be loaded after from_pretrained)")
        
        # --- connector_lvsm (ONLY new learnable module) ---
        self.connector_lvsm = LVSMConnector(
            d_lvsm=self.lvsm_d,
            d_qwen=config.hidden_size,  # 2048 for 3B
            temporal_patch_size=config.vision_config.temporal_patch_size,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            lvsm_spatial_size=self.lvsm_image_size // self.lvsm_patch_size,
            lvsm2qwen_type=self.lvsm2qwen_type,
            llm2lvsm_type=self.llm2lvsm_type,
            vlm2context_adapt_strategy=self.vlm2context_adapt_strategy,
        )
        print(f"[INFO] LVSMConnector built (learnable). d_lvsm={self.lvsm_d}, d_qwen={config.hidden_size}")

        # --- NVS Loss (frozen) ---
        self.nvs_loss_fn = NVSLoss(
            l2_weight=lvsm_cfg.get('l2_loss_weight', 1.0),
            perceptual_weight=lvsm_cfg.get('perceptual_loss_weight', 0.5),
            lpips_weight=lvsm_cfg.get('lpips_loss_weight', 0.0),
            vgg_weight_file=lvsm_cfg.get('vgg_weight_file', './metric_checkpoint/imagenet-vgg-verydeep-19.mat'),
        )
        print(f"[INFO] NVS Loss built. l2={lvsm_cfg.get('l2_loss_weight', 1.0)}, "
              f"perceptual={lvsm_cfg.get('perceptual_loss_weight', 0.5)}, "
              f"lpips={lvsm_cfg.get('lpips_loss_weight', 0.0)}")

        # JJ : Log complete LVSM config summary for training monitoring
        logger.info(
            f"\n{'='*60}\n"
            f"[LVSM-Config] Settings summary:\n"
            f"  enforce_LVSM:           {self.enforce_LVSM}\n"
            f"  nvs_loss_weight:        {self.nvs_loss_weight}\n"
            f"  nvs_target_pool:        {self.nvs_target_pool}\n"
            f"  num_target_views:       {self.num_target_views}\n"
            f"  lvsm_image_size:        {self.lvsm_image_size}\n"
            f"  lvsm_patch_size:        {self.lvsm_patch_size}\n"
            f"  lvsm_d:                 {self.lvsm_d}\n"
            f"  lvsm_grad_checkpoint:   {self.lvsm_grad_checkpoint}\n"
            f"  lvsm_grad_ckpt_every:   {self.lvsm_grad_checkpoint_every}\n"
            f"  l2_weight:              {lvsm_cfg.get('l2_loss_weight', 1.0)}\n"
            f"  perceptual_weight:      {lvsm_cfg.get('perceptual_loss_weight', 0.5)}\n"
            f"  lpips_weight:           {lvsm_cfg.get('lpips_loss_weight', 0.0)}\n"
            f"  nvs_img_log_interval:   {self.nvs_img_log_interval}\n"
            f"  nvs_img_log_max_views:  4 (hardcoded)\n"
            f"  lvsm2qwen_type:         {self.lvsm2qwen_type}\n"
            f"  llm2lvsm_type:          {self.llm2lvsm_type}\n"
            f"  vlm2ctx_strategy:       {self.vlm2context_adapt_strategy}\n"
            f"  lvsm_checkpoint:        {self._lvsm_checkpoint_path}\n"
            f"{'='*60}"
        )

        self.post_init()

        self.skip_connector = True
        # jj: Cache minimal adapter diagnostics for wandb logging (updated in forward/decode).
        self._diag_lvsm2llm_delta_ratio = None
        self._diag_llm2lvsm_delta_ratio = None

        #/////debug
        #///used when inject vggt geo feat
        self.disable_lvsm2llm_fusion = False#True
        self.disable_llm2lvsm_fusion = False#True
        self.nvs_loss_only = False#True

        # JJ: mirror connector geometry attrs for decoder_input_vggt_geo path
        self.spatial_embeds_layer_idx = -1
        self.visual_temporal_merge_size = 2
        self.visual_spatial_merge_size = 2

        self.decoder_input_llm_layer = False
        self.random_reset_decoder_input_token =False
        self.decoder_input_vggt_geo = False#True
        if self.decoder_input_vggt_geo:
            # JJ: Bridge VGGT geometric packed tokens -> Qwen width so existing llm2lvsm projector can be reused.
            # Use for debugging how far the clip is from mvg vit.
            self.vggt_geo_in_dim = (
                config.hidden_size * self.visual_temporal_merge_size * (self.visual_spatial_merge_size ** 2)
            )
            # We use ln rather qwenrmsenorm, as we feed to lvsm direcly wo going through llm.
            self.vggt_geo_norm = nn.LayerNorm(self.vggt_geo_in_dim)
            self.vggt_geo_proj = nn.Linear(self.vggt_geo_in_dim, config.hidden_size)

        self.decoder_input_which_llm_layer = 0 # self.model.config.num_hidden_layers + 1 36+1
        # self.decoder_input_which_llm_layer = int((self.model.config.num_hidden_layers)//2) # self.model.config.num_hidden_layers + 1 36+1
        # self.decoder_input_which_llm_layer = -1 # self.model.config.num_hidden_layers + 1 36+1

    # ================================================================
    # JJ : Load LVSM pretrained weights (must be called AFTER from_pretrained)
    # ================================================================
    def _freeze_lvsm_after_load(self):
        # jj: clarity refactor only; behavior preserved.
        for p in self.lvsm_model.parameters():
            p.requires_grad = False
            if not p.data.is_contiguous():
                p.data = p.data.contiguous()
        for buf in self.lvsm_model.buffers():
            if not buf.data.is_contiguous():
                buf.data = buf.data.contiguous()
        self.lvsm_model.eval()

    def _reinit_connector_lvsm_post_post_init(self):
        # jj: clarity cleanup only; keep the same init policy/order and train behavior.
        # WHY: post_init/_init_weights may corrupt connector init; we restore a stable start:
        # linear -> Xavier+0, LayerNorm -> (1,0), gates -> small non-zero.
        conn = self.connector_lvsm

        def _init_linear_xavier(module):
            if module is None:
                return
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        def _init_layernorm(module):
            if module is None:
                return
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        _init_linear_xavier(conn.lvsm_proj)
        _init_layernorm(conn.lvsm_norm)
        _init_layernorm(conn.view_norm)
        _init_linear_xavier(conn.view_proj)
        _init_linear_xavier(conn.gamma_head)
        _init_linear_xavier(conn.beta_head)
        _init_layernorm(getattr(self, "vggt_geo_norm", None))
        _init_linear_xavier(getattr(self, "vggt_geo_proj", None))

        # Gates — small non-zero init so gradient flows from step 0
        with torch.no_grad():
            conn.nvs_gate.fill_(0.1)
            conn.mod_gate.fill_(0.1)

        with torch.no_grad():
            nvs_gate_raw = conn.nvs_gate.item()
            mod_gate_raw = conn.mod_gate.item()
            nvs_gate_eff = torch.tanh(conn.nvs_gate).item()
            mod_gate_eff = torch.tanh(conn.mod_gate).item()

            def _stat_str(tensor: torch.Tensor) -> str:
                t = tensor.detach().float()
                return (
                    f"mean={t.mean().item():.6f}, std={t.std().item():.6f}, "
                    f"min={t.min().item():.6f}, max={t.max().item():.6f}, norm={t.norm().item():.6f}"
                )

            stat_lines = [
                "[INFO] connector_lvsm init actual values:",
                f"  nvs_gate raw={nvs_gate_raw:.6f}, tanh={nvs_gate_eff:.6f}",
                f"  mod_gate raw={mod_gate_raw:.6f}, tanh={mod_gate_eff:.6f}",
                f"  lvsm_proj.weight:  {_stat_str(conn.lvsm_proj.weight)}",
            ]
            if conn.view_proj is not None:
                stat_lines.append(
                    f"  view_proj.weight:  {_stat_str(conn.view_proj.weight)}"
                )
            if conn.gamma_head is not None:
                stat_lines.append(
                    f"  gamma_head.weight: {_stat_str(conn.gamma_head.weight)}"
                )
            if conn.beta_head is not None:
                stat_lines.append(
                    f"  beta_head.weight:  {_stat_str(conn.beta_head.weight)}"
                )
            print("\n".join(stat_lines))

    def _reload_nvs_perceptual_vgg_if_needed(self):
        # JJ : Reload VGG perceptual loss weights in float32.
        # WHY: from_pretrained(torch_dtype=bf16) converts ALL params to bf16,
        #      including frozen VGG weights.  The float64→bf16→float32 roundtrip
        #      causes NaN in deep VGG blocks.  Reloading from .mat after to(bf16)
        #      restores original precision.
        if (self.nvs_loss_weight > 0
                and hasattr(self.nvs_loss_fn, 'perceptual_loss_module')):
            plm = self.nvs_loss_fn.perceptual_loss_module
            plm.reload_vgg_float32()
            # Move to same device as model
            device = next(self.parameters()).device
            plm.blocks.to(device)

    def load_lvsm_checkpoint(self, checkpoint_path=None):
        """
        JJ : Load LVSM pretrained weights after from_pretrained.
        
        Must be called AFTER from_pretrained() returns, because post_init()
        calls _init_weights which re-initializes all nn.Linear with random values.
        Same pattern as VGGT spatial_encoder.load_pretrained_weights().
        """
        ckpt_path = checkpoint_path or self._lvsm_checkpoint_path
        if ckpt_path is not None:
            self.lvsm_model.load_checkpoint(ckpt_path)
            # JJ : Re-freeze and ensure contiguous after loading
            self._freeze_lvsm_after_load()
            print(f"[INFO] LVSM checkpoint loaded and frozen: {ckpt_path}")
        else:
            print("[WARN] No LVSM checkpoint path provided. Using random weights.")

        # JJ : Post-load connector reinit (order-sensitive, no behavior change).
        self._reinit_connector_lvsm_post_post_init()
        self._reload_nvs_perceptual_vgg_if_needed()

        # JJ : Re-init SDPA gates after child post_init() corruption
        # WHY: parent __init__ calls post_init() + _init_sdpa_gates() correctly,
        #      but child __init__ calls post_init() AGAIN (line 150), which
        #      re-randomizes sdpa_gate weights via _init_weights.
        #      This final call restores weight=0, bias=4.0 → sigmoid≈0.98.
        if getattr(self.config, 'enable_sdpa_gating', False):
            self._init_sdpa_gates()

        # JJ : Sanity check — every param must be finite after re-init
        self._check_connector_params()

    def _check_connector_params(self):
        """JJ : Print finite status of every connector_lvsm parameter."""
        for name, p in self.connector_lvsm.named_parameters():
            finite = torch.isfinite(p).all().item()
            print(f"[CHECK-connector] {name:20s} finite={finite} "
                  f"shape={tuple(p.shape)}")

    @staticmethod
    def _safe_delta_ratio(updated: torch.Tensor, base: torch.Tensor, eps: float = 1e-12) -> float:
        # jj: Compute ||updated-base|| / ||base|| in fp32 for stable, comparable diagnostics.
        u = updated.detach().float()
        b = base.detach().float()
        return (torch.norm(u - b) / torch.norm(b).clamp_min(eps)).item()

    # ================================================================
    # JJ : Image pre-processing helper — direct resize to lvsm_image_size
    # ================================================================
    @staticmethod
    @torch.no_grad()
    def _resize_to_lvsm(frames, intrinsics_3x3, target_size):
        """
        JJ : Directly resize frames to target_size × target_size, adapt intrinsics.

        Uses direct bilinear resize (preserves full FOV at the cost of fx ≠ fy for
        non-square inputs).  target_size is driven by the LVSM checkpoint's image_size
        config (256 for scene_decoder_only_256.pt, 512 for future 512 checkpoints).

        Args:
            frames:          [N, C, H, W]  float32 tensor in [0, 1]
            intrinsics_3x3:  [B, N, 3, 3]  pixel-unit intrinsics at the SAME (H, W) as frames
            target_size:     int  — matches LVSM checkpoint's image_size

        Returns:
            resized:   [N, C, target_size, target_size]
            fxfycxcy:  [B, N, 4]  intrinsics scaled to target_size × target_size image
        """
        H, W = frames.shape[-2], frames.shape[-1]

        # Direct resize — keep full FOV
        resized = F.interpolate(
            frames, size=(target_size, target_size), mode='bilinear', align_corners=False
        )

        # Scale intrinsics independently for x and y axes
        scale_x = target_size / W
        scale_y = target_size / H
        K = intrinsics_3x3.float()
        fx = K[..., 0, 0] * scale_x
        fy = K[..., 1, 1] * scale_y
        cx = K[..., 0, 2] * scale_x
        cy = K[..., 1, 2] * scale_y
        fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1).to(intrinsics_3x3.dtype)
        return resized, fxfycxcy

    # ================================================================
    # JJ : LVSM input preparation (crop-first version)
    # ================================================================
    def _prepare_lvsm_inputs(self, video_tchw, extrinsics_w2c, intrinsics):
        """
        JJ : Prepare LVSM inputs — center-crop to square, resize 256×256, adapt intrinsics.

        Args:
            video_tchw:       list of [T, C, H, W] tensors — raw video frames
            extrinsics_w2c:   [B, S, 3, 4]  VGGT w2c extrinsics
            intrinsics:       [B, S, 3, 3]  VGGT pixel intrinsics at frame (H, W)

        Returns:
            lvsm_images:       [B, N, 3, lvsm_size, lvsm_size] in [0, 1]
            lvsm_input_tokens: [B, N*n_patches, d_lvsm]
            c2w:               [B, N, 4, 4]
            fxfycxcy:          [B, N, 4]  — square-corrected intrinsics
        """
        device = extrinsics_w2c.device
        dtype  = extrinsics_w2c.dtype
        lvsm_size = self.lvsm_image_size

        raw_frames = video_tchw[0]  # [T, C, H, W]
        if raw_frames.max() > 1.0:
            raw_frames = raw_frames / 255.0

        # JJ : Direct resize + adapt intrinsics (target_size = self.lvsm_image_size)
        lvsm_frames, fxfycxcy = self._resize_to_lvsm(
            raw_frames.float(), intrinsics, lvsm_size
        )  # lvsm_frames: [T, 3, lvsm_size, lvsm_size],  fxfycxcy: [B, T, 4]
        lvsm_images = lvsm_frames.unsqueeze(0)  # [1, T, 3, lvsm_size, lvsm_size]

        c2w = w2c_to_c2w(extrinsics_w2c)  # [B, S, 4, 4]

        # # JJ [DIAG] : Trace NaN source step-by-step
        # Compute Plücker rays
        ray_o, ray_d = compute_plucker_rays(c2w, fxfycxcy, h=lvsm_size, w=lvsm_size, device=device)


        # Get posed input (RGB + Plücker): [B, N, 9, lvsm_size, lvsm_size]
        posed_input = get_posed_input(
            images=lvsm_images.to(dtype), ray_o=ray_o.to(dtype), ray_d=ray_d.to(dtype)
        )


        # Tokenize via frozen LVSM image_tokenizer
        with torch.no_grad():
            lvsm_input_tokens = self.lvsm_model.image_tokenizer(posed_input)


        _, n_patches, d = lvsm_input_tokens.shape
        B = c2w.shape[0]
        N = raw_frames.shape[0]
        lvsm_input_tokens = lvsm_input_tokens.reshape(B, N * n_patches, d)

        return lvsm_images, lvsm_input_tokens, c2w, fxfycxcy

    def _prepare_lvsm_inputs_old(self, video_tchw, extrinsics_w2c, intrinsics):
        """
        JJ : [OLD - kept as fallback] Direct bilinear resize without center-crop.
        Non-square videos cause fx≠fy after resize, out-of-distribution for LVSM.
        Use _prepare_lvsm_inputs instead.
        
        Args:
            video_tchw: list of [T, C, H, W] tensors - raw video frames
            extrinsics_w2c: [B, S, 3, 4] - VGGT w2c extrinsics
            intrinsics: [B, S, 3, 3] - VGGT intrinsics
            
        Returns:
            lvsm_images: [B, N, 3, 256, 256] - resized images in [0,1]
            lvsm_input_tokens: [B, N*n_patches, d_lvsm] - LVSM input tokens
            c2w: [B, N, 4, 4] - camera-to-world matrices
            fxfycxcy: [B, N, 4] - rescaled intrinsics
        """
        device = extrinsics_w2c.device
        dtype = extrinsics_w2c.dtype
        
        # Get raw frames and resize to LVSM resolution
        # video_tchw[0] shape: [T, C, H, W]
        raw_frames = video_tchw[0]  # [T, C, H, W]
        T, C, H_orig, W_orig = raw_frames.shape
        
        # Normalize to [0, 1] if needed
        if raw_frames.max() > 1.0:
            raw_frames = raw_frames / 255.0
        
        # Resize to LVSM resolution: [T, 3, 256, 256]
        lvsm_size = self.lvsm_image_size
        lvsm_frames = F.interpolate(
            raw_frames, size=(lvsm_size, lvsm_size), mode='bilinear', align_corners=False
        )
        lvsm_images = lvsm_frames.unsqueeze(0)  # [1, T, 3, 256, 256]
        
        # Convert camera params
        c2w = w2c_to_c2w(extrinsics_w2c)  # [B, S, 4, 4]
        fxfycxcy = intrinsics_to_fxfycxcy(
            intrinsics, target_h=lvsm_size, target_w=lvsm_size,
            orig_hw=(H_orig, W_orig)
        )  # [B, S, 4]

        # Compute Plücker rays
        ray_o, ray_d = compute_plucker_rays(c2w, fxfycxcy, h=lvsm_size, w=lvsm_size, device=device)
        
        # Get posed input (RGB + Plücker): [B, N, 9, 256, 256]
        posed_input = get_posed_input(images=lvsm_images.to(dtype), ray_o=ray_o.to(dtype), ray_d=ray_d.to(dtype))

        
        # Tokenize via frozen LVSM image_tokenizer: [B*N, n_patches, d_lvsm]
        with torch.no_grad():
            lvsm_input_tokens = self.lvsm_model.image_tokenizer(posed_input)


        _, n_patches, d = lvsm_input_tokens.shape
        B = c2w.shape[0]
        N = T
        lvsm_input_tokens = lvsm_input_tokens.reshape(B, N * n_patches, d)  # [B, N*n_patches, d]
        
        return lvsm_images, lvsm_input_tokens, c2w, fxfycxcy

    # ================================================================
    # JJ : Helper — upsample LLM delta to LVSM patch resolution
    # ================================================================
    def _upsample_llm_delta_to_lvsm(self, llm_delta, video_grid_thw):
        """
        JJ : Undo QwenVL's temporal + spatial merge to produce a delta tensor
        with the same spatial-temporal layout as lvsm_base_context.

                TODO: If patch_residual needs more capacity, keep this resize path but
                add a small post-bilinear refinement head before fusion, instead of only
                relying on raw bilinear upsampling as the final patch-space transform.

        QwenVL compresses: N raw frames
          → T_merged = N//temporal_patch_size temporal groups
          → each group: h_out × w_out tokens  (h_out = h_pre//spatial_merge_size)

        We reverse this:
          1. Bilinear upsample (h_out, w_out) → (32, 32)  [LVSM patch grid]
          2. Repeat each temporal group temporal_patch_size times  [restore raw frames]

        Args:
            llm_delta:       [B, L_vis, d_lvsm]  — projected LLM delta
            video_grid_thw:  [num_grids, 3]       — QwenVL grid (t_merged, h_pre, w_pre)

        Returns:
            [B, N_lvsm * lvsm_n_patches, d_lvsm]  — aligned with lvsm_base_context
        """
        B, L_vis, d = llm_delta.shape
        tp  = self.connector_lvsm.temporal_patch_size   # 2
        sp  = self.connector_lvsm.spatial_merge_size    # 2
        H_l = self.connector_lvsm.lvsm_spatial_size     # 32
        P   = self.connector_lvsm.lvsm_n_patches         # 1024

        all_up = []
        t_tok = 0  # cursor into L_vis dimension
        for grid_idx in range(video_grid_thw.shape[0]):
            t_m, h_p, w_p = (int(x) for x in video_grid_thw[grid_idx].tolist())
            h_o, w_o = h_p // sp, w_p // sp
            n_tok = t_m * h_o * w_o

            assert t_tok + n_tok <= L_vis, (
                f"[_upsample_llm_delta_to_lvsm] token offset overflow: "
                f"t_tok={t_tok} + n_tok={n_tok} > L_vis={L_vis}"
            )

            # Extract grid-entry tokens: [B, t_m*h_o*w_o, d]
            chunk = llm_delta[:, t_tok:t_tok + n_tok]

            # Reshape to 2D spatial map and upsample to LVSM patch grid
            # [B*t_m, d, h_o, w_o] → [B*t_m, d, H_l, H_l]
            chunk = chunk.reshape(B * t_m, h_o, w_o, d).permute(0, 3, 1, 2).contiguous()
            chunk = F.interpolate(chunk, size=(H_l, H_l), mode='bilinear', align_corners=False)

            # [B*t_m, d, H_l, H_l] → [B, t_m, P, d]
            chunk = chunk.permute(0, 2, 3, 1).reshape(B, t_m, P, d)

            # Expand temporal: one QwenVL group → tp LVSM frames (repeat)
            # [B, t_m, P, d] → [B, t_m*tp, P, d] → [B, t_m*tp*P, d]
            chunk = chunk.unsqueeze(2).expand(-1, -1, tp, -1, -1).reshape(B, t_m * tp * P, d)

            all_up.append(chunk)
            t_tok += n_tok

        assert t_tok == L_vis, (
            f"[_upsample_llm_delta_to_lvsm] not all tokens consumed: {t_tok} != {L_vis}"
        )
        return torch.cat(all_up, dim=1)  # [B, N_lvsm*P, d_lvsm]

    # ================================================================
    # JJ : Helper - decode novel target views via LVSM and compute NVS loss
    # ================================================================
    def _decode_nvs(self, visual_hidden, nvs_target_frames,
                    target_extrinsics_w2c, target_intrinsics,
                    lvsm_base_context, video_grid_thw):
        """
        JJ : NVS decoding with configurable context adaptation.

        Pipeline:
          base_context    = image_tokenizer(input_frames)              [pretrained, detached]
          llm_delta       = project_visual_tokens_to_lvsm(visual_hidden)
          patch_delta     = upsample_to_lvsm_grid(llm_delta)
          lvsm_context    = base_context + gate * patch_delta
          rendered        = LVSM_transformer(lvsm_context, target_poses) → loss vs GT

        At small gate init: ctx is near base (stable warm start).

        NOTE: xformers flash-attn requires bf16/fp16 throughout.

        Args:
            visual_hidden:         [L_vis, d_qwen]     LLM output visual hidden states
            nvs_target_frames:     [N_tgt, C, H, W]    novel target GT frames in [0,1]
            target_extrinsics_w2c: [1, N_tgt, 3, 4]   target w2c extrinsics from VGGT
            target_intrinsics:     [1, N_tgt, 3, 3]    target intrinsics from VGGT
            lvsm_base_context:     [B, N*n_patches, d_lvsm]  pretrained LVSM tokens
            video_grid_thw:        [num_grids, 3]       QwenVL grid (t_merged, h_pre, w_pre)

        Returns:
            nvs_loss_metrics: edict with loss, l2_loss, perceptual_loss, etc.
        """
        device = visual_hidden.device
        dtype = visual_hidden.dtype  # bf16 from autocast — kept throughout for xformers
        lvsm_size = self.lvsm_image_size
        N_tgt = nvs_target_frames.shape[0]

        # JJ : Subsample if more targets than num_target_views
        num_targets = min(self.num_target_views, N_tgt)
        if num_targets < N_tgt:
            sel = sorted(random.sample(range(N_tgt), num_targets))
            nvs_target_frames = nvs_target_frames[sel]
            target_extrinsics_w2c = target_extrinsics_w2c[:, sel]
            target_intrinsics = target_intrinsics[:, sel]
            N_tgt = num_targets
        # JJ : nvs_target_pool ('input'/'nvs'/'all') is handled at the call site in forward()

        # ---- Patch-residual modulation ----

        base_context = lvsm_base_context.detach().to(dtype)
        # Step 1: project each visual token to LVSM channels (no pooling)
        # TODO: Main ablation candidate for a stronger patch branch:
        # compare current `projection -> bilinear upsample` against
        # `LayerNorm -> bilinear upsample -> projection`.
        llm_delta = self.connector_lvsm.project_visual_tokens_to_lvsm(visual_hidden)  # [B, L_vis, d_lvsm]
        # Step 2: recover patch-wise structure aligned to lvsm_base_context
        patch_delta = self._upsample_llm_delta_to_lvsm(llm_delta, video_grid_thw)  # [B, N*P, d_lvsm]

        if self.random_reset_decoder_input_token:
            # JJ : For a stronger patch_residual ablation, randomly re-initialize the patch_delta tokens at each forward pass.
            # This tests whether the patch_residual branch provides useful learning signal even without a stable, informative LLM delta input.
            patch_delta = torch.randn_like(patch_delta)

        if self.disable_llm2lvsm_fusion:
            # If fusion is disabled, skip adding the delta to the context.
            # This allows us to isolate the effect of the LLM-to-LVSM modulation during ablation.
            lvsm_context = patch_delta
        else:
            # Step 3: gated residual fusion on context
            lvsm_context = self.connector_lvsm.fuse_patch_residual_context(
                base_context, patch_delta.to(dtype)
            )  # [B, N*P, d_lvsm]
        # jj: Track actual LLM->LVSM modulation magnitude (not gate proxy) for diagnosis.
        self._diag_llm2lvsm_delta_ratio = self._safe_delta_ratio(lvsm_context, base_context)

        # JJ : NaN checkpoint — context
        if not torch.isfinite(lvsm_context).all():
            mod_gate = self.connector_lvsm.mod_gate.item()
            logger.warning(
                f"[NVS-NaN] lvsm_context has NaN/Inf! "
                f"base finite={torch.isfinite(lvsm_base_context).all()} "
                f"mod_gate={mod_gate:.4f}"
            )
            assert False, "lvsm_context contains nan"

        # JJ : Center-crop target GT frames to square + adapt intrinsics (same as input views)
        # NOTE: target_intrinsics are at VGGT processing resolution (= input frame H×W).
        # Assumes target frames share the same resolution; holds for same-video data.
        tgt_frames = nvs_target_frames.to(device=device, dtype=torch.float32)
        if tgt_frames.max() > 1.0:
            tgt_frames = tgt_frames / 255.0
        lvsm_tgt, target_fxfycxcy = self._resize_to_lvsm(
            tgt_frames, target_intrinsics, lvsm_size
        )
        target_gt = lvsm_tgt.unsqueeze(0)  # [1, N_tgt, 3, lvsm_size, lvsm_size] — float32 for NVSLoss

        # JJ : Convert target camera params for Plücker rays
        target_c2w = w2c_to_c2w(target_extrinsics_w2c)  # [1, N_tgt, 4, 4]

        # JJ : Compute target Plücker rays
        target_ray_o, target_ray_d = compute_plucker_rays(
            target_c2w, target_fxfycxcy, h=lvsm_size, w=lvsm_size, device=device
        )
        # Cast rays to model dtype (bf16) for LVSM transformer — xformers requires bf16/fp16
        target_pose_cond = get_posed_input(
            images=None, ray_o=target_ray_o.to(dtype), ray_d=target_ray_d.to(dtype)
        )  # [1, N_tgt, 6, 256, 256] in bf16

        # JJ : NaN checkpoint — target pose conditioning
        if not torch.isfinite(target_pose_cond).all():
            logger.warning(
                f"[NVS-NaN] target_pose_cond has NaN/Inf! "
                f"c2w finite={torch.isfinite(target_c2w).all()} "
                f"fxfycxcy finite={torch.isfinite(target_fxfycxcy).all()}")

        # JJ : Decode via LVSM transformer (bf16 — required by xformers flash-attn)
        rendered_images = self.lvsm_model.decode_target_views(
            lvsm_context=lvsm_context,
            target_pose_cond=target_pose_cond,
            gradient_checkpoint=self.lvsm_grad_checkpoint,
            checkpoint_every=self.lvsm_grad_checkpoint_every,
        )  # [1, N_tgt, 3, 256, 256] in bf16

        # JJ : NaN checkpoint — rendered output
        if not torch.isfinite(rendered_images).all():
            nan_frac = (~torch.isfinite(rendered_images)).float().mean().item()
            valid = rendered_images[torch.isfinite(rendered_images)]
            valid_range = f"[{valid.min():.4f}, {valid.max():.4f}]" if valid.numel() > 0 else "ALL_NaN"
            logger.warning(
                f"[NVS-NaN] rendered_images has NaN/Inf! bad_frac={nan_frac:.2%} "
                # f"gate={gate.item():.6f} | rendered valid range: {valid_range}"
                )
            assert False, "rendered_images contains nan"

        # JJ : Compute NVS loss (L2 + Perceptual)
        nvs_loss_metrics = self.nvs_loss_fn(rendered_images, target_gt)
        # JJ : Also return rendered + GT for wandb image logging
        nvs_loss_metrics.rendered = rendered_images.detach()  # [1, N_tgt, 3, H, W]
        nvs_loss_metrics.target_gt = target_gt.detach()       # [1, N_tgt, 3, H, W]
        return nvs_loss_metrics

    # ================================================================
    # JJ : wandb logging helpers (LVSM diagnostics)
    # ================================================================
    def _wandb_log_nvs_scalars(
        self, ce, nvs_raw, nvs_weighted, l2, percep, lpips_val, psnr, total
    ):
        """JJ : Log LVSM sub-losses as wandb scalars (every step)."""
        try:
            import wandb
            if wandb.run is None:
                return
            # JJ : commit=False → metrics buffered into Trainer's next wandb.log commit,
            # ensuring NVS metrics share the same x-axis step as train/loss etc.
            # JJ : Collect SDPA gate stats (weight norm + bias mean per layer)
            sdpa_w_norms = []
            sdpa_b_means = []
            for layer in self.model.layers:
                if hasattr(layer, 'sdpa_gate'):
                    sdpa_w_norms.append(layer.sdpa_gate.weight.norm().item())
                    sdpa_b_means.append(layer.sdpa_gate.bias.mean().item())

            gate_metrics = {
                "instruct_tune/ce_loss": ce,
                "nvs/l2_loss": l2,
                "nvs/perceptual_loss": percep,
                "nvs/lpips_loss": lpips_val,
                "nvs/psnr": psnr,
                "nvs/total_nvs_loss_weighted": nvs_weighted,
                # JJ : connector_lvsm gates (raw scalar params)
                "gates/nvs_gate_raw": self.connector_lvsm.nvs_gate.item(),
                "gates/mod_gate_raw": self.connector_lvsm.mod_gate.item(),
            }
            # jj: Minimal bridge diagnostics — actual activation-space injection/modulation magnitudes.
            if self._diag_lvsm2llm_delta_ratio is not None:
                gate_metrics["fuse/lvsm2llm_delta_ratio"] = self._diag_lvsm2llm_delta_ratio
            if self._diag_llm2lvsm_delta_ratio is not None:
                gate_metrics["adapt/llm2lvsm_delta_ratio"] = self._diag_llm2lvsm_delta_ratio
            # JJ : SDPA gate stats across all decoder layers
            if sdpa_w_norms:
                gate_metrics["gates/sdpa_weight_norm_mean"] = sum(sdpa_w_norms) / len(sdpa_w_norms)
                gate_metrics["gates/sdpa_weight_norm_max"] = max(sdpa_w_norms)
                gate_metrics["gates/sdpa_bias_mean"] = sum(sdpa_b_means) / len(sdpa_b_means)

            wandb.log(gate_metrics, commit=False)
        except ImportError:
            pass  # wandb not installed, skip

    @torch.no_grad()
    def _wandb_log_nvs_images(self, rendered, target_gt, max_views=4):
        """
        JJ : Log rendered vs GT image comparison to wandb (sparse interval).
        
        Args:
            rendered: [1, N_tgt, 3, H, W] in [0,1] (may contain NaN)
            target_gt: [1, N_tgt, 3, H, W] in [0,1]
            max_views: max number of views to log
        """
        try:
            import wandb
            if wandb.run is None:
                return
            
            N = min(rendered.shape[1], max_views)
            images = []
            for i in range(N):
                pred_img = rendered[0, i].clamp(0, 1).float().cpu()   # [3, H, W]
                gt_img = target_gt[0, i].clamp(0, 1).float().cpu()    # [3, H, W]
                # JJ : Replace NaN pixels with magenta for visibility
                nan_mask = torch.isnan(pred_img)
                if nan_mask.any():
                    pred_img = pred_img.clone()
                    pred_img[0][nan_mask[0]] = 1.0  # R
                    pred_img[1][nan_mask[1]] = 0.0  # G
                    pred_img[2][nan_mask[2]] = 1.0  # B
                # JJ : Side-by-side: [3, H, 2*W]
                pair = torch.cat([gt_img, pred_img], dim=2)
                # to HWC uint8 for wandb.Image
                pair_np = (pair.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
                images.append(
                    wandb.Image(pair_np, caption=f"view_{i} (left=GT, right=Rendered)")
                )
            wandb.log(
                {"nvs/rendered_vs_gt": images},
                commit=False,
            )
        except ImportError:
            pass

    # ================================================================
    # JJ : Main forward pass
    # ================================================================
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_tchw: Optional[List[torch.FloatTensor]] = None,
        video_tchw: Optional[List[torch.FloatTensor]] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        selected_frames: Optional[List[int]] = None,
        # JJ : NVS novel target views (from dataloader, None when nvs_enabled=False)
        nvs_target_tchw: Optional[List[torch.FloatTensor]] = None,
        nvs_is_input_mask: Optional[List[torch.Tensor]] = None,
        # jj: Optional precomputed VGGT camera tensors from dataset (used to replace online pose decode).
        precomputed_input_extrinsics_w2c: Optional[List[torch.Tensor]] = None,
        precomputed_input_intrinsics: Optional[List[torch.Tensor]] = None,
        precomputed_target_extrinsics_w2c: Optional[List[torch.Tensor]] = None,
        precomputed_target_intrinsics: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        # jj: Reset per-step adapter diagnostics to avoid stale values when branches are skipped.
        self._diag_lvsm2llm_delta_ratio = None
        self._diag_llm2lvsm_delta_ratio = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ============================================================
        # Phase 1: Visual Encoding (existing logic from parent)
        # ============================================================
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            if pixel_values is not None:
                assert False, '[LVSM] Image input not supported, use video input.'

            # JJ : Store LVSM intermediates
            _lvsm_images = None
            _lvsm_input_tokens = None
            _c2w = None
            _fxfycxcy = None
            _video_token_mask = None
            _target_extrinsics_w2c = None
            _target_intrinsics = None
            _nvs_target_frames = None  # JJ : novel target frames for NVS loss

            if pixel_values_videos is not None:
                assert video_tchw is not None
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_tchw = [v.type(self.visual.dtype) for v in video_tchw]

                # QwenVL visual encoding (input frames only, no targets)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: "
                        f"tokens: {n_video_tokens}, features {n_video_features}"
                    )

                # ============================================================
                # JJ : VGGT spatial encoding with interleaved novel targets
                # ============================================================
                _has_nvs_targets = (
                    self.enforce_LVSM
                    and nvs_target_tchw is not None
                    and nvs_is_input_mask is not None
                    and len(nvs_target_tchw) > 0
                    and len(nvs_is_input_mask) > 0
                )

                # JJ : enforce_LVSM=True requires NVS targets every step; raise if missing
                if self.enforce_LVSM and not _has_nvs_targets:
                    raise RuntimeError(
                        f"[NVS-FATAL] enforce_LVSM=True but _has_nvs_targets=False. "
                        f"nvs_target_tchw={'None' if nvs_target_tchw is None else f'len={len(nvs_target_tchw)}'}, "
                        f"nvs_is_input_mask={'None' if nvs_is_input_mask is None else f'len={len(nvs_is_input_mask)}'}. "
                        f"Check dataloader: nvs_enabled must be True and _get_nvs_target_frames must not return None."
                    )

                _actual_N_in = None  # JJ : Actual input frame count (for nvs_target_pool)
                if _has_nvs_targets:
                    # JJ : Interleave input + target frames for joint VGGT pose estimation
                    nvs_mask = nvs_is_input_mask[0]  # [N_in + N_tgt] bool
                    _actual_N_in = nvs_mask.sum().item()
                    # JJ : video_tchw[0] must have exactly N_in frames (no padding)
                    assert video_tchw[0].shape[0] == _actual_N_in, (
                        f"video_tchw frame count {video_tchw[0].shape[0]} != "
                        f"nvs_mask input count {_actual_N_in}"
                    )
                    input_frames = video_tchw[0]  # [N_in, C, H, W]
                    target_frames_raw = nvs_target_tchw[0].to(
                        device=input_frames.device, dtype=input_frames.dtype
                    )  # [N_tgt, C, H, W]

                    # JJ : Spatial dims must match (dataloader already resizes targets)
                    assert target_frames_raw.shape[2:] == input_frames.shape[2:], (
                        f"target spatial {target_frames_raw.shape[2:]} != "
                        f"input spatial {input_frames.shape[2:]}; "
                        f"dataloader _get_nvs_target_frames should have resized"
                    )

                    # JJ : Build interleaved tensor [N_in + N_tgt, C, H, W]
                    S_total = nvs_mask.shape[0]
                    interleaved = torch.zeros(
                        S_total, *input_frames.shape[1:],
                        device=input_frames.device, dtype=input_frames.dtype
                    )
                    interleaved[nvs_mask] = input_frames
                    interleaved[~nvs_mask] = target_frames_raw

                    vggt_tchw = [interleaved]
                    _nvs_target_frames = nvs_target_tchw[0]  # save raw targets for NVS loss
                else:
                    vggt_tchw = video_tchw
                    _actual_N_in = video_tchw[0].shape[0]  # all frames are input

                # jj: Use precomputed VGGT cameras when provided (replace online pose decode only).
                _has_precomputed_cams = (
                    precomputed_input_extrinsics_w2c is not None
                    and precomputed_input_intrinsics is not None
                    and len(precomputed_input_extrinsics_w2c) > 0
                    and len(precomputed_input_intrinsics) > 0
                )
                if _has_nvs_targets:
                    _has_precomputed_cams = (
                        _has_precomputed_cams
                        and precomputed_target_extrinsics_w2c is not None
                        and precomputed_target_intrinsics is not None
                        and len(precomputed_target_extrinsics_w2c) > 0
                        and len(precomputed_target_intrinsics) > 0
                    )

                spatial_embeds_list, patch_start_idx = None, None
                if _has_precomputed_cams:
                    # jj: Only compute spatial features when connector path is enabled.
                    # With precomputed cameras + skip_connector=True, skip redundant VGGT feature pass.
                    if not self.skip_connector:
                        spatial_embeds_list, patch_start_idx = self.spatial_encoder(
                            vggt_tchw, grid_thw=video_grid_thw, return_cam_enc=False
                        )
                    self.extrinsics_w2c = precomputed_input_extrinsics_w2c[0].to(device=vggt_tchw[0].device)
                    self.intrisics = precomputed_input_intrinsics[0].to(device=vggt_tchw[0].device)
                    if _has_nvs_targets:
                        _target_extrinsics_w2c = precomputed_target_extrinsics_w2c[0].to(device=vggt_tchw[0].device)
                        _target_intrinsics = precomputed_target_intrinsics[0].to(device=vggt_tchw[0].device)
                    # jj: Fail-fast consistency checks between dataloader frames and precomputed camera tensors.
                    if self.extrinsics_w2c.shape[1] != _actual_N_in or self.intrisics.shape[1] != _actual_N_in:
                        raise RuntimeError(
                            f"[precompute_pose] input camera count mismatch: "
                            f"cams={self.extrinsics_w2c.shape[1]}/{self.intrisics.shape[1]} vs input_frames={_actual_N_in}"
                        )
                    if _has_nvs_targets:
                        _n_tgt = _nvs_target_frames.shape[0]
                        if _target_extrinsics_w2c.shape[1] != _n_tgt or _target_intrinsics.shape[1] != _n_tgt:
                            raise RuntimeError(
                                f"[precompute_pose] target camera count mismatch: "
                                f"cams={_target_extrinsics_w2c.shape[1]}/{_target_intrinsics.shape[1]} vs targets={_n_tgt}"
                            )
                    # jj: Optional debug compare — verify precomputed cameras against online VGGT estimation.
                    if os.environ.get("DEBUG_COMPARE_PRECOMPUTED_POSE", "0").strip() in ("1", "true", "True"):
                        _, _, _cam_cmp = self.spatial_encoder(
                            vggt_tchw, grid_thw=video_grid_thw, return_cam_enc=True
                        )
                        _all_ext_cmp, _all_int_cmp = pose_encoding_to_extri_intri(
                            _cam_cmp[0][-1].unsqueeze(0), vggt_tchw[0][-1].shape[-2:]
                        )
                        if _has_nvs_targets:
                            _m = nvs_is_input_mask[0]
                            _in_ext_cmp = _all_ext_cmp[:, _m]
                            _in_int_cmp = _all_int_cmp[:, _m]
                            _tg_ext_cmp = _all_ext_cmp[:, ~_m]
                            _tg_int_cmp = _all_int_cmp[:, ~_m]
                            # JJ: Rich diagnostics for precomputed-vs-online pose mismatch (max/mean + R/t + fx/fy/cx/cy).
                            _in_ext_err = (self.extrinsics_w2c.float() - _in_ext_cmp.float()).abs()
                            _in_int_err = (self.intrisics.float() - _in_int_cmp.float()).abs()
                            _tg_ext_err = (_target_extrinsics_w2c.float() - _tg_ext_cmp.float()).abs()
                            _tg_int_err = (_target_intrinsics.float() - _tg_int_cmp.float()).abs()

                            _din_ext = _in_ext_err.max().item()
                            _din_int = _in_int_err.max().item()
                            _dtg_ext = _tg_ext_err.max().item()
                            _dtg_int = _tg_int_err.max().item()
                            logger.warning(
                                f"[DEBUG_COMPARE_PRECOMPUTED_POSE] max_abs_diff "
                                f"in_ext={_din_ext:.6e} in_int={_din_int:.6e} "
                                f"tgt_ext={_dtg_ext:.6e} tgt_int={_dtg_int:.6e}"
                            )
                            logger.warning(
                                f"[DEBUG_COMPARE_PRECOMPUTED_POSE] mean_abs_diff "
                                f"in_ext={_in_ext_err.mean().item():.6e} in_int={_in_int_err.mean().item():.6e} "
                                f"tgt_ext={_tg_ext_err.mean().item():.6e} tgt_int={_tg_int_err.mean().item():.6e}"
                            )
                            logger.warning(
                                f"[DEBUG_COMPARE_PRECOMPUTED_POSE] input_breakdown "
                                f"R_max={_in_ext_err[..., :3, :3].max().item():.6e} t_max={_in_ext_err[..., :3, 3].max().item():.6e} "
                                f"fx_max={_in_int_err[..., 0, 0].max().item():.6e} fy_max={_in_int_err[..., 1, 1].max().item():.6e} "
                                f"cx_max={_in_int_err[..., 0, 2].max().item():.6e} cy_max={_in_int_err[..., 1, 2].max().item():.6e}"
                            )
                            logger.warning(
                                f"[DEBUG_COMPARE_PRECOMPUTED_POSE] target_breakdown "
                                f"R_max={_tg_ext_err[..., :3, :3].max().item():.6e} t_max={_tg_ext_err[..., :3, 3].max().item():.6e} "
                                f"fx_max={_tg_int_err[..., 0, 0].max().item():.6e} fy_max={_tg_int_err[..., 1, 1].max().item():.6e} "
                                f"cx_max={_tg_int_err[..., 0, 2].max().item():.6e} cy_max={_tg_int_err[..., 1, 2].max().item():.6e}"
                            )
                else:
                    # JJ : Run VGGT on (interleaved or input-only) frames
                    spatial_embeds_list, patch_start_idx, camera_encs = self.spatial_encoder(
                        vggt_tchw, grid_thw=video_grid_thw, return_cam_enc=True
                    )
                    all_extrinsics_w2c, all_intrinsics = pose_encoding_to_extri_intri(
                        camera_encs[0][-1].unsqueeze(0), vggt_tchw[0][-1].shape[-2:]
                    )

                    # JJ : Split poses into input / target using mask
                    if _has_nvs_targets:
                        nvs_mask = nvs_is_input_mask[0]
                        self.extrinsics_w2c = all_extrinsics_w2c[:, nvs_mask]   # [1, N_in, 3, 4]
                        self.intrisics = all_intrinsics[:, nvs_mask]             # [1, N_in, 3, 3]
                        _target_extrinsics_w2c = all_extrinsics_w2c[:, ~nvs_mask]  # [1, N_tgt, 3, 4]
                        _target_intrinsics = all_intrinsics[:, ~nvs_mask]          # [1, N_tgt, 3, 3]
                    else:
                        self.extrinsics_w2c = all_extrinsics_w2c
                        self.intrisics = all_intrinsics

                # Existing connector fusion (skip_connector flag from parent)
                if self.skip_connector:
                    fused_embeds = video_embeds
                else:
                    fused_embeds, _, _ = self.connector(
                        video_embeds=video_embeds,
                        spatial_embeds_list=spatial_embeds_list,
                        patch_start_idx=patch_start_idx,
                        grid_thw=video_grid_thw,
                    )

                # ============================================================
                # Phase 2 & 3: LVSM token preparation & fusion
                # ============================================================
                if self.enforce_LVSM:
                    # Phase 2: Prepare LVSM inputs (input frames + input poses only)
                    _lvsm_images, _lvsm_input_tokens, _c2w, _fxfycxcy = self._prepare_lvsm_inputs(
                        video_tchw, self.extrinsics_w2c, self.intrisics
                    )
                    
                    # Phase 3: Fuse LVSM tokens into QwenVL tokens for LLM
                    if self.disable_lvsm2llm_fusion:                            
                        video_embeds_base = fused_embeds
                    else:  
                        video_embeds_base = fused_embeds
                        fused_embeds = self.connector_lvsm.fuse_for_llm(
                            fused_embeds, _lvsm_input_tokens, video_grid_thw
                        )
                    # jj: Track actual LVSM->LLM injection magnitude for diagnosis.
                    self._diag_lvsm2llm_delta_ratio = self._safe_delta_ratio(fused_embeds, video_embeds_base)
                    # # JJ [DIAG] : Confirm whether Phase 3 corrupts embeddings before LLM
                    # logger.warning(
                    #     f"[DIAG-Phase3] fused_embeds finite={torch.isfinite(fused_embeds).all().item()} "
                    #     f"min={fused_embeds.min().item():.3f} max={fused_embeds.max().item():.3f} "
                    #     f"lvsm_proj.weight norm={self.connector_lvsm.lvsm_proj.weight.norm().item():.6f}"
                    # )

                # Scatter fused visual tokens into input embeddings
                mask = input_ids == self.config.video_token_id
                _video_token_mask = mask.clone()  # JJ : Save for Phase 5 extraction
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, fused_embeds)

                # # JJ [DIAG] Check inputs_embeds right before LLM — remove once confirmed clean
                # logger.warning(
                #     f"[DIAG-PRE-LLM] inputs_embeds finite={torch.isfinite(inputs_embeds).all().item()} "
                #     f"min={inputs_embeds.min():.4f} max={inputs_embeds.max():.4f} "
                #     f"fused_embeds finite={torch.isfinite(fused_embeds).all().item()} "
                #     f"min={fused_embeds.min():.4f} max={fused_embeds.max():.4f}")
                assert torch.isfinite(inputs_embeds).all(), "inputs_embeds contains nan"

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # ============================================================
        # RoPE position ids (existing logic from parent)
        # ============================================================
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                assert self.position_ids_compute_mode in ["mRoPE_woT", "mRoPE", "mRoPE_readaptT"]
                if self.position_ids_compute_mode == "mRoPE_readaptT":
                    assert selected_frames is not None
                
                position_ids, rope_deltas, visual_token_mask = custom_get_rope_index(
                    self.config, input_ids, image_grid_thw, video_grid_thw,
                    second_per_grid_ts, attention_mask,
                    position_ids_compute_mode=self.position_ids_compute_mode,
                    selected_frames_id=selected_frames,
                    temporal_patch_size=self.model.config.vision_config.temporal_patch_size,
                    temporal_readapted_merge_strategy=self.temporal_readapted_merge_strategy,
                    temporal_readapted_use_dynamic_scale_factor=self.temporal_readapted_use_dynamic_scale_factor,
                    temporal_readapted_scale_factor=self.temporal_readapted_scale_factor,
                )
                self.visual_token_mask = visual_token_mask
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                self.visual_token_mask = torch.zeros_like(position_ids)[0]

        # Camera downsampling (existing)
        intrisics_down, extrinsics_w2c_down = downsample_cams(
            self.intrisics, self.extrinsics_w2c,
            temporal_patch_size=2, extrinsics_sample_strategy="mean",
                                        )
        self.intrisics_down = intrisics_down
        self.extrinsics_w2c_down = extrinsics_w2c_down

        # ============================================================
        # Phase 4: LLM Forward (existing, no modification)
        # ============================================================
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            # JJ： lvsm can use other layers visual info
            output_hidden_states=output_hidden_states or self.decoder_input_llm_layer,
            return_dict=return_dict,
            cache_position=cache_position,
            RoPE_attn_mode=self.RoPE_attn_mode,
            visual_token_mask=self.visual_token_mask,
            intrisics=self.intrisics_down,
            extrinsics_w2c=self.extrinsics_w2c_down,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        # ============================================================
        # Phase 5 & 6: CE Loss + NVS Loss
        # ============================================================
        loss = None
        if labels is not None:
            # CE loss (existing)
            logits_float = logits.float()
            shift_logits = logits_float[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            ce_loss = loss_fct(shift_logits, shift_labels)
            
            loss = ce_loss

            # Phase 5: NVS decoding & loss
            # JJ : Guard — 'input' mode only needs LVSM tokens; 'nvs'/'all' also need novel targets
            # JJ : Build _can_nvs with per-condition tracking for diagnostics
            _cond_enforce = self.enforce_LVSM
            _cond_weight = self.nvs_loss_weight > 0
            _cond_vmask = _video_token_mask is not None
            _cond_lvsm_tok = _lvsm_input_tokens is not None
            _can_nvs = _cond_enforce and _cond_weight and _cond_vmask and _cond_lvsm_tok

            if self.nvs_target_pool in ("nvs", "all"):
                _cond_tgt_frames = _nvs_target_frames is not None
                _cond_tgt_ext = _target_extrinsics_w2c is not None
                _can_nvs = _can_nvs and _cond_tgt_frames and _cond_tgt_ext
            else:
                _cond_tgt_frames = True  # not required for 'input' mode
                _cond_tgt_ext = True

            # JJ : Warn when NVS path is skipped — DDP requires all trainable params to participate
            if not _can_nvs:
                logger.warning(
                    f"[NVS-SKIP] NVS decode path skipped. Condition breakdown: "
                    f"enforce_LVSM={_cond_enforce}, "
                    f"nvs_loss_weight={self.nvs_loss_weight}(>0:{_cond_weight}), "
                    f"video_token_mask={_cond_vmask}, "
                    f"lvsm_input_tokens={_cond_lvsm_tok}, "
                    f"nvs_target_pool='{self.nvs_target_pool}', "
                    f"nvs_target_frames={_cond_tgt_frames}, "
                    f"target_extrinsics_w2c={_cond_tgt_ext}. "
                    f"LVSM trainable params will have no gradient this step."
                )

            if _can_nvs:
                # JJ : Extract visual hidden states from LLM output
                # cp from spmllm: hidden dim is perfectly 1024*2, aligns with qwen2.5 hidden so that we can resue our llm2lvsm projector
                def _preprocess_spatial_embeds(
                    spatial_embeds_list: List[List[torch.Tensor]],
                    patch_start_idx: List[int],
                    grid_thw: torch.Tensor,
                ) -> torch.Tensor:
                    all_spatial_embeds = []
                    grid_idx = 0

                    for i, spatial_embeds_item in enumerate(spatial_embeds_list):

                        # spatial_embeds_list: List[List[Float[Tensor, "S+5,P,2D"]]]
                        spatial_embeds = spatial_embeds_item[self.spatial_embeds_layer_idx].unsqueeze(0)
                        # spatial_embeds: Float[Tensor, "B,S,P,2D"]
                        spatial_embeds = spatial_embeds[:, :, patch_start_idx[i]:]

                        B, S, P, DD = spatial_embeds.shape
                        assert B == 1, "batch size should be 1"

                        # Find corresponding grid_thw rows
                        accumulated_t = 0

                        if grid_idx >= len(grid_thw):
                            raise ValueError(f"Not enough grid_thw rows for spatial_embeds {i}")

                        while accumulated_t * self.visual_temporal_merge_size < S:
                            if grid_idx >= len(grid_thw):
                                raise ValueError(
                                    f"Not enough grid_thw rows for spatial_embeds {i}. Accumulated T={accumulated_t}, Target S={S}"
                                )

                            t, h, w = grid_thw[grid_idx].tolist()

                            if accumulated_t == 0:
                                npatch_h, npatch_w = h, w
                            else:
                                assert h == npatch_h and w == npatch_w, f"Spatial dimensions mismatch within video {i}"

                            accumulated_t += t
                            grid_idx += 1

                        npatch_t = accumulated_t

                        assert P == npatch_h * npatch_w, "patch number mismatch"
                        assert npatch_t == S // self.visual_temporal_merge_size, "temporal patch number mismatch"

                        # reshape spatial embeddings to 2D grid
                        spatial_embeds = (
                            spatial_embeds.view(B, S, npatch_h, npatch_w, DD).permute(0, 1, 4, 2, 3).contiguous()
                        )  # [B, S, DD, np_h, np_w]

                        spatial_embeds = (
                            spatial_embeds.view(
                                B,
                                npatch_t,
                                self.visual_temporal_merge_size,
                                DD,
                                npatch_h // self.visual_spatial_merge_size,
                                self.visual_spatial_merge_size,
                                npatch_w // self.visual_spatial_merge_size,
                                self.visual_spatial_merge_size,
                            )
                            .permute(0, 1, 4, 6, 5, 7, 3, 2)
                            .contiguous()
                        )

                        spatial_embeds = spatial_embeds.reshape(
                            B * npatch_t * npatch_h * npatch_w, DD * self.visual_temporal_merge_size
                        )
                        spatial_embeds = spatial_embeds.view(
                            -1, DD * self.visual_temporal_merge_size * self.visual_spatial_merge_size**2
                        )
                        all_spatial_embeds.append(spatial_embeds)

                    all_spatial_embeds_concated = torch.cat(all_spatial_embeds, dim=0)
                    return all_spatial_embeds_concated

                assert not (self.decoder_input_vggt_geo and self.decoder_input_llm_layer), "decoder_input_vggt_geo and decoder_input_llm_layer cannot both be True; choose one source for NVS decoder input"
                if self.decoder_input_vggt_geo:
                    spatial_embeds_list, patch_start_idx, camera_encs = self.spatial_encoder(
                        video_tchw, grid_thw=video_grid_thw, return_cam_enc=True
                    )
                    spatial_embeds = _preprocess_spatial_embeds(spatial_embeds_list, patch_start_idx, video_grid_thw)
                    # JJ: Adapt packed VGGT geo tokens (merged_dim) back to Qwen width before llm2lvsm projection.
                    if spatial_embeds.shape[-1] != self.vggt_geo_in_dim:
                        raise RuntimeError(
                            f"Unexpected VGGT geo dim: got {spatial_embeds.shape[-1]}, expected {self.vggt_geo_in_dim}"
                        )
                    visual_hidden = self.vggt_geo_proj(self.vggt_geo_norm(spatial_embeds))
                else:
                    if self.decoder_input_llm_layer:
                        try:
                            hidden_states = outputs.hidden_states[self.decoder_input_which_llm_layer]
                        except Exception as e:
                            logger.error(f"Failed to extract hidden states from layer {self.decoder_input_which_llm_layer}: {e}")
                            raise
                    else:
                        # Path of previous trained LVSM vlm
                        hidden_states = outputs[0]

                    visual_hidden = hidden_states[_video_token_mask]  # [L_vis, d_qwen]

                # JJ : Select NVS target pool
                if self.nvs_target_pool == "input":
                    # Reconstruction: use input frames as targets
                    _nvs_tgt = video_tchw[0]  # [N_in, C, H, W] (already == _actual_N_in)
                    _tgt_ext = self.extrinsics_w2c
                    _tgt_int = self.intrisics
                elif self.nvs_target_pool == "nvs":
                    # Novel only: use gap-sampled novel frames
                    _nvs_tgt = _nvs_target_frames
                    _tgt_ext = _target_extrinsics_w2c
                    _tgt_int = _target_intrinsics
                elif self.nvs_target_pool == "all":
                    # Mixed: input + novel frames as targets
                    _input_tgt = video_tchw[0].to(  # [N_in, C, H, W] (already == _actual_N_in)
                        device=_nvs_target_frames.device, dtype=_nvs_target_frames.dtype
                    )
                    _nvs_tgt = torch.cat([_input_tgt, _nvs_target_frames], dim=0)
                    _tgt_ext = torch.cat([self.extrinsics_w2c, _target_extrinsics_w2c], dim=1)
                    _tgt_int = torch.cat([self.intrisics, _target_intrinsics], dim=1)
                else:
                    raise NotImplementedError(
                        f"nvs_target_pool='{self.nvs_target_pool}' not supported. "
                        f"Choose from: 'nvs', 'input', 'all'."
                    )

                # JJ : NVS decode with residual context (base = pretrained LVSM tokens)
                nvs_loss_metrics = self._decode_nvs(
                    visual_hidden,
                    _nvs_tgt,
                    _tgt_ext,
                    _tgt_int,
                    lvsm_base_context=_lvsm_input_tokens,
                    video_grid_thw=video_grid_thw,
                )

                # Phase 6: Combined loss
                nvs_loss = nvs_loss_metrics.loss
                weighted_nvs = self.nvs_loss_weight * nvs_loss

                # JJ : NaN/Inf guard — skip NVS term if not finite to avoid poisoning CE
                if self.nvs_loss_only:
                    loss = weighted_nvs
                else:
                    if not torch.isfinite(weighted_nvs):
                        logger.warning(
                            f"[NVS] NVS loss not finite! Skipping NVS term. "
                            f"nvs={nvs_loss.item():.6f} L2={nvs_loss_metrics.l2_loss.item():.6f} "
                            f"Percep={nvs_loss_metrics.perceptual_loss.item():.6f} "
                            f"LPIPS={nvs_loss_metrics.lpips_loss.item():.6f} "
                            f"gate={self.connector_lvsm.nvs_gate.item():.6f}")
                        loss = ce_loss
                    else:
                        loss = ce_loss + weighted_nvs

                # JJ : Log LVSM sub-losses to wandb (every step)
                _nvs_val = nvs_loss.item()
                _l2_val = nvs_loss_metrics.l2_loss.item()
                _percep_val = nvs_loss_metrics.perceptual_loss.item()
                _lpips_val = nvs_loss_metrics.lpips_loss.item()
                _psnr_val = nvs_loss_metrics.psnr.item()
                self._wandb_log_nvs_scalars(
                    ce_loss.item(), _nvs_val, weighted_nvs.item(),
                    _l2_val, _percep_val, _lpips_val, _psnr_val, loss.item(),
                )

                # JJ : Periodically log rendered vs GT images to wandb.
                # Use wandb.run.step (Trainer's step) to align with scalar logs.
                try:
                    import wandb as _wb
                    _cur_step = _wb.run.step if (_wb.run is not None) else 0
                except ImportError:
                    _cur_step = 0
                if (
                    self.nvs_img_log_interval > 0
                    and _cur_step % self.nvs_img_log_interval == 0
                ):
                    self._wandb_log_nvs_images(
                        nvs_loss_metrics.rendered, nvs_loss_metrics.target_gt
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
