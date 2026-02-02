"""
Custom Qwen2_5_VLModel (Text Decoder) with customization points.

This is the base text model that sits inside Qwen2_5_VLForConditionalGeneration.
Architecture: Qwen2_5_VLForConditionalGeneration -> Qwen2_5_VLModel -> language_model -> model

You can customize:
1. Decoder layers (replace with custom attention)
2. Rotary embeddings (replace with PRoPE or custom RoPE)
3. Forward pass logic
4. Attention mask creation
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLConfig,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2RMSNorm,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache, SlidingWindowCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CustomQwen2_5_VLModel(Qwen2_5_VLModel):
    """
    Custom Qwen2_5_VLModel with marked customization points.
    
    This is the text decoder model. You can customize:
    - Decoder layers (for custom attention mechanisms)
    - Rotary embeddings (for PRoPE or other position encodings)
    - Forward pass (for custom processing logic)
    """
    
    def __init__(self, config: Qwen2_5_VLConfig):
        # Call grandparent __init__ to skip parent's __init__
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # ============================================================================
        # CUSTOMIZATION POINT 1: Replace Decoder Layers
        # ============================================================================
        # Option A: Use standard layers (default)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Option B: Use custom decoder layers (uncomment to use)
        # from src.qwenvl.model.custom_attention import CustomQwen2DecoderLayer
        # self.layers = nn.ModuleList(
        #     [CustomQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        # )
        # print(f"[INFO] Using CustomQwen2DecoderLayer for {len(self.layers)} layers")
        
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # ============================================================================
        # CUSTOMIZATION POINT 2: Replace Rotary Embeddings
        # ============================================================================
        # Option A: Use standard RoPE (default)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        
        # Option B: Use PRoPE or custom rotary embeddings (uncomment to use)
        # from src.qwenvl.model.custom_qwen2_5_VLPRoPE import Qwen2_5_VLPRotaryEmbedding
        # self.rotary_emb = Qwen2_5_VLPRotaryEmbedding(config=config)
        # print("[INFO] Using custom PRoPE rotary embeddings")

        self.gradient_checkpointing = False
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # ============================================================================
        # CUSTOMIZATION POINT 3: Add Custom Forward Arguments
        # ============================================================================
        # Add any custom arguments you need to pass through
        # Example: Ks and viewmats for PRoPE
        **kwargs,  # Catch-all for custom arguments
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Custom forward pass for Qwen2_5_VLModel.
        
        Args:
            Standard transformer arguments (input_ids, attention_mask, etc.)
            **kwargs: Custom arguments (e.g., Ks, viewmats for PRoPE)
        
        Returns:
            BaseModelOutputWithPast
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # ============================================================================
        # CUSTOMIZATION POINT 4: Custom Position IDs Processing
        # ============================================================================
        # The hard coded `3` is for temporal, height and width dimensions
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        
        # YOUR CUSTOM CODE: Modify position_ids if needed
        # Example: Add spatial biases, modify based on camera parameters, etc.

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # ============================================================================
        # CUSTOMIZATION POINT 5: Custom Position Embeddings
        # ============================================================================
        # Create position embeddings to be shared across the decoder layers
        
        # Option A: Standard RoPE (default)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Option B: PRoPE with camera parameters (uncomment if using PRoPE)
        # if 'Ks' in kwargs and 'viewmats' in kwargs:
        #     position_embeddings = self.rotary_emb(
        #         hidden_states, 
        #         position_ids,
        #         Ks=kwargs['Ks'],
        #         viewmats=kwargs['viewmats']
        #     )
        # else:
        #     position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # ============================================================================
        # CUSTOMIZATION POINT 6: Pre-Decoder Processing
        # ============================================================================
        # YOUR CUSTOM CODE: Add any preprocessing before decoder layers
        # Example: Inject spatial features, add custom embeddings, etc.
        # hidden_states = hidden_states + self.custom_spatial_embedding(...)

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # ============================================================================
            # CUSTOMIZATION POINT 7: Per-Layer Custom Processing
            # ============================================================================
            # YOUR CUSTOM CODE: Add custom processing per layer
            # Example: Layer-specific modifications, skip connections, etc.
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    # Pass through custom kwargs to decoder layers
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # ============================================================================
        # CUSTOMIZATION POINT 8: Post-Decoder Processing
        # ============================================================================
        # YOUR CUSTOM CODE: Add any postprocessing after all decoder layers
        # Example: Apply final transformations, inject additional information, etc.
        # hidden_states = self.custom_post_processing(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        """
        Update causal mask for attention computation.
        
        ============================================================================
        CUSTOMIZATION POINT 9: Custom Attention Mask
        ============================================================================
        You can override this method to create custom attention patterns:
        - Sparse attention
        - Local attention windows
        - Custom causal patterns
        - Spatial-aware attention masks
        """
        
        # Standard implementation (unchanged)
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2_5_VL. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5_VLConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape (batch_size, 1, query_length, key_value_length).
        
        ============================================================================
        CUSTOMIZATION POINT 10: Custom 4D Attention Mask Creation
        ============================================================================
        You can customize this to create:
        - Different causal patterns
        - Sliding window with custom logic
        - Spatial-aware masking patterns
        """
        
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            
            if config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        
        return causal_mask


# ============================================================================
# SUMMARY OF CUSTOMIZATION POINTS
# ============================================================================
"""
1. __init__ line ~51: Replace decoder layers with custom attention layers
2. __init__ line ~64: Replace rotary embeddings with PRoPE or custom RoPE
3. forward line ~105: Add custom forward arguments (e.g., Ks, viewmats)
4. forward line ~161: Custom position IDs processing
5. forward line ~175: Custom position embeddings (PRoPE integration point)
6. forward line ~186: Pre-decoder processing (inject spatial features)
7. forward line ~200: Per-layer custom processing
8. forward line ~230: Post-decoder processing
9. _update_causal_mask line ~246: Custom attention mask patterns
10. _prepare_4d_causal_attention_mask line ~327: Custom 4D mask creation

USAGE EXAMPLE:
--------------
from src.qwenvl.model.custom_qwen2_5_VLM import CustomQwen2_5_VLModel

# In your CustomSpatialMLLM __init__:
if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
    original_decoder = self.model.language_model.model
    custom_decoder = CustomQwen2_5_VLModel(original_decoder.config)
    custom_decoder.load_state_dict(original_decoder.state_dict(), strict=False)
    self.model.language_model.model = custom_decoder
"""
