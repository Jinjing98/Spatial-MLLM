"""
JJ : LVSM Decoder-Only model wrapper for integration with Spatial-MLLM.

Recreates the LVSM decoder-only architecture (Images2LatentScene) locally,
avoiding import path issues with the LVSM submodule.
Loads pretrained weights from LVSM checkpoint and freezes all parameters.

Architecture (from LVSM ICLR 2025):
  - image_tokenizer: Rearrange + Linear (in_channels * patch_size^2 → d)
  - target_pose_tokenizer: Rearrange + Linear (6 * patch_size^2 → d)
  - transformer_blocks: n_layer × QK_Norm_TransformerBlock
  - transformer_input_layernorm: LayerNorm
  - image_token_decoder: LayerNorm + Linear + Sigmoid
"""

import os
import sys
import importlib.util

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


# ============================================================================
# JJ : Import LVSM transformer block from submodule via importlib
# ============================================================================
def _get_lvsm_transformer_block_class():
    """
    JJ : Dynamically import QK_Norm_TransformerBlock from LVSM submodule.
    Uses importlib to avoid sys.path pollution and __init__.py requirements.
    """
    lvsm_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'submodule', 'LVSM')
    lvsm_root = os.path.abspath(lvsm_root)
    transformer_path = os.path.join(lvsm_root, 'model', 'transformer.py')
    
    if not os.path.exists(transformer_path):
        raise FileNotFoundError(
            f"[LVSMWrapper] LVSM transformer.py not found at: {transformer_path}\n"
            f"Ensure the LVSM submodule is properly set up."
        )
    
    spec = importlib.util.spec_from_file_location("_lvsm_transformer", transformer_path)
    module = importlib.util.module_from_spec(spec)
    # JJ : Register in sys.modules to prevent re-import issues
    sys.modules["_lvsm_transformer"] = module
    spec.loader.exec_module(module)
    
    return module.QK_Norm_TransformerBlock, module.init_weights


# ============================================================================
# JJ : LVSM Decoder-Only Model (recreated for clean integration)
# ============================================================================
class LVSMDecoderOnly(nn.Module):
    """
    JJ : LVSM decoder-only architecture for Novel View Synthesis.
    
    Recreates Images2LatentScene without requiring LVSM package imports.
    All parameters are frozen after loading pretrained weights.
    
    Only used for Qwen2.5-VL-3B. Other model sizes raise NotImplementedError.
    """
    
    def __init__(
        self,
        d=768,
        d_head=64,
        n_layer=24,
        use_qk_norm=True,
        image_size=256,
        patch_size=8,
        image_in_channels=9,   # 3 RGB + 6 Plücker
        target_in_channels=6,  # 6 Plücker (no RGB)
    ):
        super().__init__()
        self.d = d
        self.d_head = d_head
        self.n_layer = n_layer
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2  # 1024
        self.spatial_size = image_size // patch_size       # 32
        
        # JJ : Import transformer block class
        QK_Norm_TransformerBlock, init_weights_fn = _get_lvsm_transformer_block_class()
        
        # --- Tokenizers ---
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size, pw=patch_size,
            ),
            nn.Linear(image_in_channels * (patch_size ** 2), d, bias=False),
        )
        
        self.target_pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size, pw=patch_size,
            ),
            nn.Linear(target_in_channels * (patch_size ** 2), d, bias=False),
        )
        
        # --- Image token decoder ---
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(d, bias=False),
            nn.Linear(d, (patch_size ** 2) * 3, bias=False),
            nn.Sigmoid(),
        )
        
        # --- Transformer ---
        self.transformer_blocks = nn.ModuleList([
            QK_Norm_TransformerBlock(d, d_head, use_qk_norm=use_qk_norm)
            for _ in range(n_layer)
        ])
        self.transformer_input_layernorm = nn.LayerNorm(d, bias=False)
    
    def pass_layers(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """
        JJ : Forward through all transformer blocks with optional gradient checkpointing.
        
        Args:
            input_tokens: [batch, seq_len, d]
            gradient_checkpoint: bool - use gradient checkpointing for memory saving
            checkpoint_every: int - group N layers per checkpoint segment
            
        Returns:
            output_tokens: [batch, seq_len, d]
        """
        if not gradient_checkpoint:
            for layer in self.transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
        
        def _process_layer_group(tokens, start_idx, end_idx):
            for idx in range(start_idx, end_idx):
                tokens = self.transformer_blocks[idx](tokens)
            return tokens
        
        num_layers = len(self.transformer_blocks)
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group, input_tokens, start_idx, end_idx,
                use_reentrant=False,
            )
        return input_tokens
    
    def decode_target_views(
        self,
        lvsm_context,
        target_pose_cond,
        gradient_checkpoint=False,
        checkpoint_every=1,
    ):
        """
        JJ : Decode target views from LLM-processed context + target Plücker poses.
        
        This is the NVS decoding pipeline (Phase 5 in the architecture):
        1. Tokenize target poses
        2. For each target view: concat context + target_tokens → transformer → decode
        
        Args:
            lvsm_context: [B, L_ctx, d] - LLM visual hidden states projected to d=768
            target_pose_cond: [B, V_target, 6, H, W] - target Plücker rays
            gradient_checkpoint: bool
            checkpoint_every: int
            
        Returns:
            rendered_images: [B, V_target, 3, H, W] - rendered images in [0,1]
        """
        B, V_target = target_pose_cond.shape[:2]
        L_ctx = lvsm_context.shape[1]
        
        # Tokenize target poses: [B*V_target, n_patches, d]
        target_tokens = self.target_pose_tokenizer(target_pose_cond)
        
        # Repeat context for each target view: [B, L_ctx, d] → [B*V_target, L_ctx, d]
        repeated_context = repeat(
            lvsm_context, 'b l d -> (b v) l d', v=V_target
        )
        
        # Concat context + target tokens: [B*V_target, L_ctx + n_patches, d]
        transformer_input = torch.cat([repeated_context, target_tokens], dim=1)
        transformer_input = self.transformer_input_layernorm(transformer_input)
        
        # Forward through transformer (frozen)
        output_tokens = self.pass_layers(
            transformer_input,
            gradient_checkpoint=gradient_checkpoint,
            checkpoint_every=checkpoint_every,
        )
        
        # Split and take target tokens only
        _, pred_target_tokens = output_tokens.split([L_ctx, self.n_patches], dim=1)
        
        # Decode to images: [B*V_target, n_patches, patch_size^2 * 3]
        rendered_flat = self.image_token_decoder(pred_target_tokens)
        
        # Rearrange to image format: [B, V_target, 3, H, W]
        rendered_images = rearrange(
            rendered_flat,
            "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            b=B, v=V_target,
            h=self.spatial_size, w=self.spatial_size,
            p1=self.patch_size, p2=self.patch_size, c=3,
        )
        
        return rendered_images
    
    def load_checkpoint(self, ckpt_path):
        """
        JJ : Load pretrained LVSM weights from checkpoint.
        
        Filters out loss_computer and process_data keys (we use our own loss).
        
        Args:
            ckpt_path: str - path to LVSM .pt checkpoint file
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[LVSMWrapper] Checkpoint not found: {ckpt_path}\n"
                f"Please provide a valid LVSM pretrained checkpoint path."
            )
        
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        # Filter out keys we don't need
        filtered = {
            k: v for k, v in state_dict.items()
            if not k.startswith("loss_computer") and not k.startswith("process_data")
        }
        
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[LVSMWrapper] Missing keys (expected for loss/data modules): {missing}")
        if unexpected:
            print(f"[LVSMWrapper] Unexpected keys: {unexpected}")
        
        print(f"[LVSMWrapper] Loaded LVSM checkpoint from: {ckpt_path}")


def build_lvsm_model(
    checkpoint_path=None,
    d=768, d_head=64, n_layer=24, use_qk_norm=True,
    image_size=256, patch_size=8,
    device=None, dtype=None,
):
    """
    JJ : Factory function to create and initialize frozen LVSM model.
    
    Only supports Qwen2.5-VL-3B integration. Other model sizes: NotImplementedError.
    
    Args:
        checkpoint_path: str or None - path to pretrained LVSM .pt checkpoint
        d, d_head, n_layer, use_qk_norm: LVSM transformer config
        image_size, patch_size: LVSM image tokenization config
        device: torch.device
        dtype: torch.dtype
        
    Returns:
        model: LVSMDecoderOnly (frozen)
    """
    model = LVSMDecoderOnly(
        d=d, d_head=d_head, n_layer=n_layer, use_qk_norm=use_qk_norm,
        image_size=image_size, patch_size=patch_size,
    )
    
    if checkpoint_path is not None:
        model.load_checkpoint(checkpoint_path)
    else:
        print("[LVSMWrapper] No checkpoint provided. LVSM uses random weights.")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    
    return model
