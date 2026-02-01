"""
EXAMPLE: How to use CustomQwen2Model in SpatialMLLMForConditionalGeneration

This is an EXAMPLE file showing how to integrate the custom model forward.
DO NOT USE THIS FILE DIRECTLY - it's just for reference.

To use this approach:
1. Modify custom_qwen2_model.py with your custom logic
2. Update your spatial_mllm.py __init__ method as shown below
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

from src.qwenvl.model.connector import get_connector
from src.qwenvl.model.spatial_encoder import VGGTSpatialEncoderConfig, VGGTSpatialEncoderPreTrainedModel
from src.qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri
# Import your custom model
from src.qwenvl.model.custom_qwen2_model import CustomQwen2Model


class SpatialMLLMConfig(Qwen2_5_VLConfig):
    model_type = "spatial-mllm"

    def __init__(self, spatial_config=None, connector_config=None, **kwargs):
        super().__init__(**kwargs)
        self.sub_configs["spatial_config"] = VGGTSpatialEncoderConfig
        if isinstance(spatial_config, dict):
            self.spatial_config = self.sub_configs["spatial_config"](**spatial_config)
        elif spatial_config is None:
            self.spatial_config = self.sub_configs["spatial_config"]()

        self.connector_config = connector_config if connector_config is not None else {}


class SpatialMLLMForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    EXAMPLE showing how to integrate CustomQwen2Model.
    
    This replaces the base Qwen2Model decoder with your custom version
    that has the modified forward method.
    """
    
    def __init__(self, config):
        # Initialize parent class first
        super().__init__(config)
        
        # ==============================================================================
        # OPTION 1: Replace the decoder model with custom one (RECOMMENDED)
        # ==============================================================================
        # The architecture is:
        #   self.model = Qwen2_5VLModel
        #     └── self.model.model = Qwen2Model (this is what we want to replace)
        
        # Check if we're using Qwen2_5VL architecture
        if hasattr(self.model, 'model'):
            print("[INFO] Replacing base Qwen2Model with CustomQwen2Model...")
            
            # Store reference to original model
            original_decoder = self.model.model
            
            # Create custom model with same config
            custom_decoder = CustomQwen2Model(original_decoder.config)
            
            # Copy all weights from original to custom model
            # This ensures your custom model starts with the pretrained weights
            custom_decoder.load_state_dict(original_decoder.state_dict(), strict=False)
            
            # Replace the decoder
            self.model.model = custom_decoder
            
            print("[INFO] Successfully replaced decoder with CustomQwen2Model")
        else:
            print("[WARNING] Could not find self.model.model - architecture may have changed")
        
        # ==============================================================================
        # OPTION 2: Monkey patching (alternative, less clean)
        # ==============================================================================
        # Uncomment this if you prefer monkey patching instead:
        #
        # import types
        # self._original_forward = self.model.model.forward
        # self.model.model.forward = types.MethodType(self._custom_forward_wrapper, self.model.model)
        
        # Add your spatial components
        self.spatial_encoder = VGGTSpatialEncoderPreTrainedModel(config.spatial_config)
        self.connector = get_connector(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    # ==============================================================================
    # If using Option 2 (monkey patching), implement this method:
    # ==============================================================================
    def _custom_forward_wrapper(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **flash_attn_kwargs,
    ):
        """
        Wrapper for custom forward - only needed if using monkey patching (Option 2).
        Note: 'self' here is the Qwen2Model instance, not SpatialMLLMForConditionalGeneration.
        """
        # Add your custom logic here
        print(f"[DEBUG] Custom forward called!")
        
        # Call original forward
        return self._original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

    # Your original forward method stays the same
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
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        # ... rest of your original forward method stays the same ...
        # When you call self.model(...) at lines 212-223, it will now use
        # your CustomQwen2Model.forward() instead of the original
        
        pass  # Replace with your actual implementation


# ==============================================================================
# TESTING YOUR CHANGES
# ==============================================================================
if __name__ == "__main__":
    """
    Quick test to verify the custom model is being used.
    """
    from transformers import Qwen2_5_VLConfig
    
    # Create a small test config
    config = SpatialMLLMConfig()
    
    # Initialize model
    print("Initializing model...")
    model = SpatialMLLMForConditionalGeneration(config)
    
    # Check if custom model is being used
    if hasattr(model.model, 'model'):
        print(f"Decoder model type: {type(model.model.model)}")
        print(f"Is CustomQwen2Model: {type(model.model.model).__name__ == 'CustomQwen2Model'}")
    
    print("Test complete!")
