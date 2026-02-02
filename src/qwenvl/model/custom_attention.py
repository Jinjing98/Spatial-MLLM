"""
Custom Attention Mechanisms for Spatial MLLM.

This module provides custom attention layers that you can modify to inject
spatial information, use custom attention patterns, or add cross-view attention.
"""

from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2MLP,
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from typing_extensions import Unpack


class CustomQwen2Attention(Qwen2Attention):
    """
    Custom Qwen2 Attention with modifications for spatial reasoning.
    
    You can override this to:
    1. Add spatial biases to attention scores
    2. Implement cross-view attention
    3. Inject depth/pose information into attention
    4. Use custom attention patterns
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # ============================================================================
        # ADD YOUR CUSTOM LAYERS HERE
        # ============================================================================
        # Example: Add a spatial bias projection
        # self.spatial_bias_proj = nn.Linear(spatial_dim, config.num_attention_heads)
        
        # Example: Add learnable attention temperature
        # self.attention_temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        Custom attention forward pass.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) for RoPE
            attention_mask: Attention mask
            past_key_value: KV cache
            cache_position: Cache position
            **kwargs: Additional arguments (can include custom spatial info)
        
        Returns:
            attn_output: [batch, seq_len, hidden_size]
            attn_weights: Attention weights (if output_attentions=True)
        """
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # ============================================================================
        # STANDARD ATTENTION COMPUTATION (from original Qwen2Attention)
        # ============================================================================
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Update KV cache if exists
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # ============================================================================
        # OPTION 1: USE CUSTOM ATTENTION COMPUTATION
        # ============================================================================
        # Uncomment this block if you want full control over attention computation
        
        # # Compute attention scores manually
        # attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        # 
        # # YOUR CUSTOM CODE: Add spatial bias to attention scores
        # # Example: if you have spatial_info in kwargs
        # if "spatial_bias" in kwargs:
        #     spatial_bias = kwargs["spatial_bias"]  # [batch, num_heads, seq_len, seq_len]
        #     attn_scores = attn_scores + spatial_bias
        # 
        # # Apply attention mask
        # if attention_mask is not None:
        #     attn_scores = attn_scores + attention_mask
        # 
        # # Softmax to get attention weights
        # attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # 
        # # Compute attention output
        # attn_output = torch.matmul(attn_weights, value_states)
        # attn_output = attn_output.transpose(1, 2).contiguous()
        
        # ============================================================================
        # OPTION 2: USE STANDARD ATTENTION WITH OPTIONAL MODIFICATIONS
        # ============================================================================
        # This uses the optimized attention implementations (flash attention, sdpa, etc.)
        
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        # YOUR CUSTOM CODE: Modify kwargs before attention
        # Example: Add custom scaling
        # custom_scaling = self.scaling * self.attention_temperature
        custom_scaling = self.scaling
        
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=custom_scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        
        # ============================================================================
        # YOUR CUSTOM CODE: Post-process attention output
        # ============================================================================
        # Example: Add residual spatial information
        # if "spatial_features" in kwargs:
        #     spatial_features = kwargs["spatial_features"]
        #     attn_output = attn_output + self.spatial_proj(spatial_features)
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights


class CustomQwen2DecoderLayer(Qwen2DecoderLayer):
    """
    Custom Qwen2 Decoder Layer that uses CustomQwen2Attention.
    
    You can also modify the entire layer computation here, including:
    1. Different normalization strategies
    2. Custom MLP modifications
    3. Additional sub-layers
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        # Don't call super().__init__() directly, instead manually initialize
        nn.Module.__init__(self)  # Initialize GradientCheckpointingLayer's parent
        
        self.hidden_size = config.hidden_size
        
        # Use custom attention instead of standard Qwen2Attention
        self.self_attn = CustomQwen2Attention(config=config, layer_idx=layer_idx)
        
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        
        # ============================================================================
        # ADD YOUR CUSTOM LAYERS HERE
        # ============================================================================
        # Example: Add a spatial adapter
        # self.spatial_adapter = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size // 4),
        #     nn.GELU(),
        #     nn.Linear(config.hidden_size // 4, config.hidden_size)
        # )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Custom decoder layer forward pass.
        
        You can pass custom information through **kwargs, such as:
        - spatial_bias: Spatial attention bias
        - spatial_features: Additional spatial features to inject
        - view_info: Multi-view information
        """
        
        # ============================================================================
        # YOUR CUSTOM CODE: Pre-process hidden states
        # ============================================================================
        # Example: Print layer statistics
        # if self.attention_type == "sliding_attention":
        #     print(f"[DEBUG] Layer with sliding attention, hidden mean: {hidden_states.mean().item():.4f}")
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # ============================================================================
        # SELF ATTENTION with custom modifications
        # ============================================================================
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,  # Pass through custom kwargs to attention
        )
        
        # ============================================================================
        # YOUR CUSTOM CODE: Post-attention modifications
        # ============================================================================
        # Example: Add spatial adapter after attention
        # if hasattr(self, 'spatial_adapter'):
        #     hidden_states = hidden_states + 0.1 * self.spatial_adapter(hidden_states)
        
        hidden_states = residual + hidden_states
        
        # ============================================================================
        # MLP (Feed-Forward Network)
        # ============================================================================
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # ============================================================================
        # YOUR CUSTOM CODE: Post-layer modifications
        # ============================================================================
        # Example: Apply layer-specific processing
        # hidden_states = hidden_states * some_custom_gate
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs


# ============================================================================
# EXAMPLE: Cross-View Attention Layer
# ============================================================================

class CrossViewAttention(nn.Module):
    """
    Example custom attention layer for cross-view reasoning.
    
    This demonstrates how you might implement attention between multiple views
    or inject spatial relationships between tokens.
    """
    
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.scaling = self.head_dim ** -0.5
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        view_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention between main hidden states and view states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size] - queries
            view_states: [batch, num_views, hidden_size] - keys and values
            attention_mask: Optional mask
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Queries from hidden states
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # Keys and values from view states
        K = self.k_proj(view_states).view(batch_size, -1, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)  # [batch, num_heads, num_views, head_dim]
        
        V = self.v_proj(view_states).view(batch_size, -1, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)  # [batch, num_heads, num_views, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        output = self.o_proj(attn_output)
        
        return output


# ============================================================================
# HELPER: Example usage in decoder layer with cross-view attention
# ============================================================================

class DecoderLayerWithCrossView(CustomQwen2DecoderLayer):
    """
    Example decoder layer that adds cross-view attention.
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Add cross-view attention
        self.cross_view_attn = CrossViewAttention(config)
        self.cross_view_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # Standard self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states
        
        # Cross-view attention (if view states are provided)
        if "view_states" in kwargs and kwargs["view_states"] is not None:
            residual = hidden_states
            hidden_states = self.cross_view_layernorm(hidden_states)
            cross_view_output = self.cross_view_attn(
                hidden_states=hidden_states,
                view_states=kwargs["view_states"],
                attention_mask=kwargs.get("cross_view_mask", None),
            )
            hidden_states = residual + cross_view_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs
