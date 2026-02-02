"""
Custom Spatial MLLM with CustomQwen2Model decoder.

This version integrates CustomQwen2Model into SpatialMLLMForConditionalGeneration,
replacing the base Qwen2Model decoder with your custom version.
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


class CustomSpatialMLLMConfig(Qwen2_5_VLConfig):
    model_type = "custom-spatial-mllm"

    def __init__(self, spatial_config=None, connector_config=None, **kwargs):
        super().__init__(**kwargs)
        self.sub_configs["spatial_config"] = VGGTSpatialEncoderConfig
        if isinstance(spatial_config, dict):
            self.spatial_config = self.sub_configs["spatial_config"](**spatial_config)
        elif spatial_config is None:
            self.spatial_config = self.sub_configs["spatial_config"]()

        self.connector_config = connector_config if connector_config is not None else {}


class CustomSpatialMLLMForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    Custom Spatial MLLM that uses CustomQwen2.5VLModel as the decoder.
    
    This replaces the base Qwen2.5VLModel decoder with CustomQwen2.5VLModel
    to enable custom forward logic in the language model.
    """
    
    def __init__(self, config):
        # Initialize parent class first
        super().__init__(config)
        
        # Replace the entire VL model with custom one
        print("[INFO] Replacing Qwen2_5_VLModel with CustomQwen2_5_VLModel...")
        
        from src.qwenvl.model.custom_qwen2_5_VLM import CustomQwen2_5_VLModel
        
        # Store reference to original VL model
        original_vl_model = self.model
        
        # Create custom VL model with same config
        custom_vl_model = CustomQwen2_5_VLModel(original_vl_model.config)
        
        # Copy all weights from original to custom model
        # This preserves the pretrained vision encoder and language model weights
        custom_vl_model.load_state_dict(original_vl_model.state_dict(), strict=False)
        
        # Replace self.model with custom version
        self.model = custom_vl_model
        
        print("[INFO] Successfully replaced Qwen2_5_VLModel with CustomQwen2_5_VLModel")
        print(f"[INFO] Model type: {type(self.model).__name__}")
        
        # Add spatial components
        self.spatial_encoder = VGGTSpatialEncoderPreTrainedModel(config.spatial_config)
        self.connector = get_connector(config)

        # Initialize weights and apply final processing
        self.post_init()

        # JJ: extend; notice only work under MODEL_TYPE="custom-spatial-mllm"
        # RoPE compute mode
        self.position_ids_compute_mode = "mRoPE"
        self.position_ids_compute_mode = "mRoPE_woT"
        assert self.position_ids_compute_mode in ["mRoPE_woT", "mRoPE"]
        self.RoPE_attn_mode = 'default' # use position_ids
        self.RoPE_attn_mode = 'PRoPE4VisionToken' # use position_ids
        assert self.RoPE_attn_mode in ['default', 'PRoPE4VisionToken']
        # quick Debug Some Function
        self.offline_debug = False
        self.model.offline_debug = False
        self.offline_debug = True
        self.model.offline_debug = True

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
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                assert 0, 'Not tested'
                assert image_tchw is not None, "`image_tchw` must be provided when `pixel_values` is not None."
                pixel_values = pixel_values.type(self.visual.dtype)
                image_tchw = [image_tchw_i.type(self.visual.dtype) for image_tchw_i in image_tchw]

                # get image embeddings
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                # get spatial embeddings
                spatial_embeds_list, patch_start_idx = self.spatial_encoder(image_tchw)

                # fuse video and spatial embeddings
                fused_embeds = self.connector(
                    image_embeds=image_embeds,
                    spatial_embeds_list=spatial_embeds_list,
                    patch_start_idx=patch_start_idx,
                    grid_thw=image_grid_thw,
                )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, fused_embeds)

            if pixel_values_videos is not None:
                assert video_tchw is not None, "`video_tchw` must be provided when `pixel_values_videos` is not None."
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_tchw = [video_tchw_i.type(self.visual.dtype) for video_tchw_i in video_tchw]
                # get video embeddings
                # video_tchw:16 3 644 476
                # video_grid_thw: 8 46 34 (use the vl2.5 vision encoder merge 2 temporal as one)
                # 12512(8*46*34) 1176 -> 3128 2048
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                # get spatial embeddings
                spatial_embeds_list, patch_start_idx, camera_encs = self.spatial_encoder(video_tchw, grid_thw=video_grid_thw, return_cam_enc=True)
                # abtain the pose from B S 9 to B S 3 4 and B S 4 4
                assert len(camera_encs) == 1 and len(camera_encs[0]) == 4, "camera_encs must have only one element and the last element must be a 9D pose encoding"
                assert len(camera_encs[0][-1])== 2 * video_grid_thw[0][0]
                extrinsics_w2c, intrisics = pose_encoding_to_extri_intri(camera_encs[0][-1].unsqueeze(0), video_tchw[0][-1].shape[-2:])
                print('Camera_encs:')
                print(f"{extrinsics_w2c.shape} {intrisics.shape}") # 1 4 4 16 4 4
                print(f"{len(camera_encs)} {len(camera_encs[0])} {camera_encs[0][0].shape}") # 16 9

                # fuse video and spatial embeddings
                # JJ: TODO: there should be better way?
                # here to reuse qv2.5 vision encoder (merge 2 temporal frame always)
                # 3d feature from VGGT is also rearraged in a similar manner (which is unnatually)
                # 'visual_temporal_merge_size' in 2D;
                fused_embeds = self.connector(
                    video_embeds=video_embeds,
                    spatial_embeds_list=spatial_embeds_list,
                    patch_start_idx=patch_start_idx,
                    grid_thw=video_grid_thw,
                )

                # JJ: TODO: does the above fusion make sense?
                if self.offline_debug:
                    pass
                    # fused_embeds = video_embeds


                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, fused_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                if self.position_ids_compute_mode == "mRoPE":
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                elif self.position_ids_compute_mode == "mRoPE_woT":
                    from src.qwenvl.get_rope_index_varients import get_rope_index_mRoPE_woT
                    # internally will set the T as 0 for all
                    position_ids, rope_deltas = get_rope_index_mRoPE_woT(
                        self.config,
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                else:
                    raise ValueError(f"Invalid RoPE compute mode: {self.position_ids_compute_mode}")

                self.rope_deltas = rope_deltas
                print(f"Details:")
                print(f"image_grid_thw: {image_grid_thw}") # None
                print(f"video_grid_thw: {video_grid_thw}") # 8 46 34
                print(f"second_per_grid_ts: {second_per_grid_ts}")
                print(f"Prefill Position_ids:") # 3, batch_size, seq_length
                print(f"{position_ids.shape}") # 3, batch_size, seq_length e.g. 3,1,3210

            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                assert seq_length == 1, "seq_length must be 1 for generation"
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)# purely text, therefore naive repeat along 3 dims
                print(f"Next Position_ids:") # 3, batch_size, 1
                print(f"{position_ids.shape}") # 3, batch_size, 1

            print(f"Early Position_ids:")
            print(f"({position_ids[:,0,:10]})") # 3, batch_size, seq_length
            print(f"Middle Position_ids:")
            #for example: there are (15*25)=375 tokens per img; 375*8=3000, 3210-3000=210 prompt token
            print(f"({position_ids[:,0,1500:2000:25]})") # 3, batch_size, seq_length
            print(f"Late Position_ids:")
            print(f"({position_ids[:,0,-10:]})") # 3, batch_size, seq_length


        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            # RoPE_attn_mode=self.RoPE_attn_mode,
            # intrisics=intrisics[:,::2], #B S 3 3
            # extrinsics_w2c=extrinsics_w2c[:,::2], #B S 3 4
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_tchw"] = None
            model_inputs["video_tchw"] = None

        return model_inputs


if __name__ == "__main__":
    """
    Quick test to verify the custom model is being used.
    """
    print("Testing CustomSpatialMLLMForConditionalGeneration...")
    
    config = CustomSpatialMLLMConfig()
    model = CustomSpatialMLLMForConditionalGeneration(config)
    
    # Check if custom model is being used
    if hasattr(model.model, 'model'):
        print(f"Decoder model type: {type(model.model.model)}")
        print(f"Is CustomQwen2Model: {type(model.model.model).__name__ == 'CustomQwen2Model'}")
    
    print("Test complete!")
