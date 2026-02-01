"""
Custom Qwen2Model wrapper with modified forward method.
This allows you to override the base model's forward pass while maintaining clean code structure.
"""

from typing import Optional
import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model, 
    Qwen2PreTrainedModel,
    BaseModelOutputWithPast,
    Cache,
    DynamicCache,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging
from typing_extensions import Unpack

logger = logging.get_logger(__name__)


class CustomQwen2Model(Qwen2Model):
    """
    Custom Qwen2Model that overrides the forward method.
    
    This class inherits from Qwen2Model and allows you to modify the forward pass
    while keeping the rest of the model architecture intact.
    """
    
    def __init__(self, config):
        super().__init__(config)
        # Add any custom layers or modifications here if needed
        # self.custom_layer = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        """
        Custom forward method for Qwen2Model.
        
        This is a copy of the original Qwen2Model.forward() method from transformers library.
        You can modify this method to add custom behavior.
        
        Original source: transformers/models/qwen2/modeling_qwen2.py (line ~353)
        """
        
        # ===================================================================================
        # ORIGINAL CODE FROM transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward()
        # ===================================================================================
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # ===================================================================================
        # YOUR CUSTOM CODE CAN GO HERE (BEFORE ATTENTION MASK CREATION)
        # ===================================================================================
        # Example: Modify inputs_embeds
        # inputs_embeds = self.custom_layer(inputs_embeds)
        
        # It may already have been prepared by e.g. `generate`
        # Import the mask creation functions locally to avoid circular imports
        from transformers.modeling_attn_mask_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask
        )
        
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # ===================================================================================
        # YOUR CUSTOM CODE CAN GO HERE (BEFORE DECODER LAYERS)
        # ===================================================================================
        # Example: Add custom processing before decoder layers
        # print(f"[DEBUG] Hidden states shape before layers: {hidden_states.shape}")
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # ===================================================================================
            # YOUR CUSTOM CODE CAN GO HERE (INSIDE DECODER LOOP)
            # ===================================================================================
            # Example: Add custom processing per layer
            # if layer_idx == 0:
            #     print(f"[DEBUG] First layer input: {hidden_states.mean().item()}")
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # ===================================================================================
        # YOUR CUSTOM CODE CAN GO HERE (AFTER DECODER LAYERS)
        # ===================================================================================
        # Example: Add custom processing after all layers
        # print(f"[DEBUG] Final hidden states mean: {hidden_states.mean().item()}")
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
