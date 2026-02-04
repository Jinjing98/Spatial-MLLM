import torch
import math
from typing import Optional, Tuple
from transformers import Qwen2_5_VLConfig
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import is_flash_attn_greater_or_equal_2_10, logging
import torch.nn as nn

# Import pose-aware rotary embedding utilities
from src.custom_qwenvl.model.custom_RoPE_utils import apply_poseaware_rotary, apply_poseaware_output_transform

logger = logging.get_logger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

class CustomQwen2_5_VLDecoderLayer(nn.Module):
    """
    Custom Qwen2_5_VL Decoder Layer for Spatial MLLM.
    Only differs from Qwen2_5_VLDecoderLayer in that it uses custom attention classes.
    """
    
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        
        # ============ KEY DIFFERENCE: Use custom attention classes ============
        self.self_attn = CUSTOM_QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        # =======================================================================

        # Import MLP and RMSNorm from transformers (these remain unchanged)
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
        
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # Added to pass spatial arguments to custom attention
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model. Also used to pass spatial-specific arguments to custom attention.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention - passes **kwargs to custom attention for spatial processing
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,  # Pass spatial arguments to custom attention
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors."""
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class CustomQwen2_5_VLAttention(nn.Module):
    """
    Custom Multi-headed attention for Spatial MLLM.
    Based on Qwen2_5_VLAttention with spatial reasoning modifications.
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # TODO: Initialize rotary_emb if needed
        # self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        
        # ============ CUSTOMIZATION POINT 1: Add spatial components ============
        # Add your spatial attention modules here, e.g.:
        # self.spatial_encoder = SpatialEncoder(...)
        # self.spatial_gate = nn.Linear(...)
        # =======================================================================

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # Added to accept additional spatial arguments
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # ============ CUSTOMIZATION POINT 2: Pre-process hidden states ============
        # Add spatial preprocessing here if needed
        # hidden_states = self.apply_spatial_encoding(hidden_states, **kwargs)
        # ===========================================================================

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # ============ CUSTOMIZATION POINT 3: Modify Q/K/V with spatial info ============
        # Apply spatial transformations to query, key, or value states
        # query_states = self.apply_spatial_bias_to_queries(query_states, **kwargs)
        # ===============================================================================

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # ============ CUSTOMIZATION POINT 4: Post-RoPE PRoPE transformations ============
        # Apply PRoPE-style pose-aware transformations: Q=P^T, K=P^(-1), V=P^(-1)
        # This must be done BEFORE KV cache update to ensure cached values are transformed
        pose_info = kwargs.get('pose_info', None)
        if pose_info is not None:
            query_states, key_states, value_states = apply_poseaware_rotary(
                query_states, key_states, value_states, pose_info
            )
        # ================================================================================

        if past_key_value is not None:
            # KV cache: stores already-transformed K and V (with P^(-1) applied)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # ============ CUSTOMIZATION POINT 5: Modify attention weights with spatial bias ============
        # Add spatial attention bias or mask based on spatial relationships
        # spatial_bias = self.compute_spatial_attention_bias(**kwargs)
        # attn_weights = attn_weights + spatial_bias
        # ===========================================================================================

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # ============ CUSTOMIZATION POINT 6: Post-softmax attention weight modification ============
        # Apply spatial gating or re-weighting after softmax
        # attn_weights = self.apply_spatial_gating(attn_weights, **kwargs)
        # ===========================================================================================
        
        attn_output = torch.matmul(attn_weights, value_states)

        # ============ CUSTOMIZATION POINT 6.5: PRoPE output transformation ============
        # Apply P transformation to convert from world coordinates back to camera coordinates
        # This completes the PRoPE cycle: camera (P^T, P^(-1)) -> world (attention) -> camera (P)
        if pose_info is not None:
            attn_output = apply_poseaware_output_transform(attn_output, pose_info)
        # ==============================================================================

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # ============ CUSTOMIZATION POINT 7: Pre-projection spatial fusion ============
        # Optional: Fuse spatial features before final projection (if needed)
        # attn_output = self.fuse_spatial_features(attn_output, **kwargs)
        # ==============================================================================

        attn_output = self.o_proj(attn_output)

        # ============ CUSTOMIZATION POINT 8: Post-projection spatial enhancement ============
        # Add residual spatial features or apply final spatial transformations
        # attn_output = attn_output + self.spatial_residual(**kwargs)
        # ====================================================================================

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CustomQwen2_5_VLFlashAttention2(CustomQwen2_5_VLAttention):
    """
    Custom Qwen2_5_VL flash attention module for Spatial MLLM.
    This module inherits from CustomQwen2_5_VLAttention.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # Added to accept additional spatial arguments
    ):
        bsz, q_len, _ = hidden_states.size()

        # ============ CUSTOMIZATION POINT 1: Pre-process hidden states ============
        # Add spatial preprocessing here if needed
        # hidden_states = self.apply_spatial_encoding(hidden_states, **kwargs)
        # ===========================================================================

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # ============ CUSTOMIZATION POINT 2: Modify Q/K/V with spatial info ============
        # Apply spatial transformations to query, key, or value states
        # query_states = self.apply_spatial_bias_to_queries(query_states, **kwargs)
        # ===============================================================================

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # ============ CUSTOMIZATION POINT 3: Post-RoPE PRoPE transformations ============
        # Apply PRoPE-style pose-aware transformations: Q=P^T, K=P^(-1), V=P^(-1)
        # This must be done BEFORE KV cache update to ensure cached values are transformed
        pose_info = kwargs.get('pose_info', None)
        if pose_info is not None:
            query_states, key_states, value_states = apply_poseaware_rotary(
                query_states, key_states, value_states, pose_info
            )
        # ================================================================================

        if past_key_value is not None:
            # KV cache: stores already-transformed K and V (with P^(-1) applied)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # ============ CUSTOMIZATION POINT 4: Pre-flash-attention modifications ============
        # Prepare spatial information for flash attention if needed
        # You might need to modify attention_mask based on spatial relationships
        # attention_mask = self.apply_spatial_mask(attention_mask, **kwargs)
        # ===================================================================================

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        # ============ CUSTOMIZATION POINT 5: PRoPE output transformation ============
        # Apply P transformation to convert from world coordinates back to camera coordinates
        # Flash attention output shape: [B, seq_len, num_heads, head_dim]
        # PRoPE transform expects: [B, num_heads, seq_len, head_dim]
        if pose_info is not None:
            attn_output = attn_output.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
            attn_output = apply_poseaware_output_transform(attn_output, pose_info)
            attn_output = attn_output.transpose(1, 2)  # [B, seq_len, num_heads, head_dim]
        # ============================================================================

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        
        # ============ CUSTOMIZATION POINT 6: Pre-projection spatial fusion ============
        # Optional: Fuse spatial features before final projection (if needed)
        # attn_output = self.fuse_spatial_features(attn_output, **kwargs)
        # ==============================================================================
        
        attn_output = self.o_proj(attn_output)

        # ============ CUSTOMIZATION POINT 7: Post-projection spatial enhancement ============
        # Add residual spatial features or apply final spatial transformations
        # attn_output = attn_output + self.spatial_residual(**kwargs)
        # ====================================================================================

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ============================================================================
# Custom Attention Classes Mapping
# ============================================================================

CUSTOM_QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": CustomQwen2_5_VLAttention,
    "flash_attention_2": CustomQwen2_5_VLFlashAttention2,
    "sdpa": CustomQwen2_5_VLAttention,  # Use eager for SDPA, or implement CustomQwen2_5_VLSdpaAttention if needed
}


# ============================================================================
# Custom Decoder Layer
# ============================================================================
