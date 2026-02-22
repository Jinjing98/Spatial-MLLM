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


# JJ: Custom modules
from src.custom_qwenvl.model.camera_pose_temporal_merge import downsample_cams
from src.custom_qwenvl.model.custom_qwen2_5_VLM import CustomQwen2_5_VLModel
from src.custom_qwenvl.model.custom_qwen2_5_VLRoPE import custom_get_rope_index

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
        super().__init__(config)        
        # Store reference to original VL model
        original_vl_model = self.model
        
        # JJ: 
        # Create custom VL model with same config
        print("[INFO] Create CustomQwen2_5_VLModel with default Qwen25 config...")
        print("[INFO] Default Qwen25 temporal patch size: ", original_vl_model.config.vision_config.temporal_patch_size)
        custom_vl_model = CustomQwen2_5_VLModel(original_vl_model.config)
        # original_vl_model.config.vision_config.temporal_patch_size = 1 # HACK: JJ
        custom_vl_model.load_state_dict(original_vl_model.state_dict(), strict=True)        
        print("[INFO] Load Qwen2_5 VLModel weights in CustomQwen2_5_VLModel successfully.")
        print("[INFO] CustomQwen2_5_VLModel temporal patch size:", custom_vl_model.config.vision_config.temporal_patch_size)
        self.model = custom_vl_model
        
        # Add spatial components
        self.spatial_encoder = VGGTSpatialEncoderPreTrainedModel(config.spatial_config)
        print("[INFO] Init SpatialEncoder successfully.")
        self.connector = get_connector(config)
        print(f"[INFO] Init Connector with temporal patch size {config.vision_config.temporal_patch_size} successfully.")

        # Initialize weights and apply final processing
        self.post_init()

        # NOTE JJ
        # RoPE pose id compute mode + THW dim T HACK
        self.position_ids_compute_mode = "mRoPE_readaptT" # "mRoPE_readaptT" "mRoPE_woT"
        assert self.position_ids_compute_mode in ["mRoPE_woT", "mRoPE", "mRoPE_readaptT"]
        # RoPE attention in custom decoder layer
        self.RoPE_attn_mode = 'default' # 'PRoPE4VisionToken' # use position_ids
        # self.RoPE_attn_mode = 'PRoPE4VisionToken' # 'PRoPE4VisionToken' # use position_ids
        assert self.RoPE_attn_mode in ['default', 'PRoPE4VisionToken']
        # Used to indenty visual tokens for PRoPE
        self.visual_token_mask = None # directly aligned with position_ids len
        self.intrisics = None
        self.intrisics_down = None
        self.extrinsics_w2c = None
        self.extrinsics_w2c_down = None

        self.offline_debug = False
        self.model.offline_debug = False
        # JJ: Temporarily disabled for cleaner loss debugging
        # self.offline_debug = True
        # self.model.offline_debug = True
        
        # JJ: Track training step for NaN detection
        self.global_step = 0
        self.current_epoch = 0
    
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
        selected_frames: Optional[List[int]] = None, # JJ: used for mRoPE_readaptT under SA strategy
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
            if pixel_values is not None: # will not run through here 
                assert False, 'Should not reach here...'
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
                fused_embeds,_, _ = self.connector(
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
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                # get spatial embeddings
                spatial_embeds_list, patch_start_idx, camera_encs = self.spatial_encoder(video_tchw, grid_thw=video_grid_thw, return_cam_enc=True)
                self.extrinsics_w2c, self.intrisics = pose_encoding_to_extri_intri(camera_encs[0][-1].unsqueeze(0), video_tchw[0][-1].shape[-2:])
                assert len(camera_encs) == 1 and len(camera_encs[0]) == 4, "camera_encs must have only one element and the last element must be a 9D pose encoding"
                assert len(camera_encs[0][-1])== 2 * video_grid_thw[0][0]

                # JJ: TODO: there should be better way?
                # fuse video and spatial embeddings
                # Reuse qv2.5 vision encoder (merge 2 temporal frame via tublar)
                # 3d feature from VGGT (visual_temporal_merge_size) is also rearraged below with 2 tem frames as one.
                # TODO: adjust fusion; effectiveness of fusion
                # fused_embeds = self.connector(
                fused_embeds, _, _ = self.connector(
                    video_embeds=video_embeds,
                    spatial_embeds_list=spatial_embeds_list,
                    patch_start_idx=patch_start_idx,
                    grid_thw=video_grid_thw,
                )

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
                # JJ
                assert self.position_ids_compute_mode in ["mRoPE_woT", "mRoPE", "mRoPE_readaptT"]
                if self.position_ids_compute_mode == "mRoPE_readaptT":
                    assert selected_frames is not None, "`selected_frames` must be provided when `position_ids_compute_mode` is `mRoPE_readaptT`"
                
                position_ids, rope_deltas, visual_token_mask = custom_get_rope_index(
                    self.config,
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                    # JJ extend
                    position_ids_compute_mode=self.position_ids_compute_mode,
                    selected_frames_id=selected_frames,
                    temporal_patch_size=self.model.config.vision_config.temporal_patch_size,
                )
                self.visual_token_mask = visual_token_mask # JJ. Indicate in current tokens, what are the vision ones.
                self.rope_deltas = rope_deltas

                if self.offline_debug:
                    print(f"*"*20)
                    print(f"Details During Prefill:")
                    print(f"image_grid_thw: {image_grid_thw}") # None
                    print(f"video_grid_thw: {video_grid_thw}") # 8 46 34
                    print(f"second_per_grid_ts: {second_per_grid_ts}")
                    print(f"Prefill Position_ids:") # 3, batch_size, seq_length
                    print(f"{position_ids.shape}") # 3, batch_size, seq_length e.g. 3,1,3210

                    print(f"Early Text Position_ids:")
                    print(f"({position_ids[:,0,:18]})") # 3, batch_size, seq_length
                    # print(f"Visual token mask: {self.visual_token_mask[0,:100]}")
                    print(f"Middle (vision) Position_ids:")
                    #for example: there are (15*25)=375 tokens per img; 375*8=3000, 3210-3000=210 prompt token
                    print(f"({position_ids[:,0,15:3128+15:391]})") # 3, batch_size, seq_length
                    # print(f"Visual token mask: {self.visual_token_mask[0,1500:3000:125]}")
                    print(f"Later (Most) Position_ids:")
                    print(f"({position_ids[:,0,30:]})") # 3, batch_size, seq_length
                    # print(f"Visual token mask: {self.visual_token_mask[0, -10:]}")

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

                # JJ FIXME
                self.visual_token_mask = torch.zeros_like(position_ids)[0]
                print(f"Next Position_ids...") # 3, batch_size, 1

        # JJ: Track training progress (only increment during training with labels)
        if labels is not None:
            self.global_step += 1
            # Print progress every step (only during training)
            if self.global_step % 1 == 0:  # Can change to % 10 for less verbose
                print(f"[Step {self.global_step}] Forward pass...")
        
        # JJ: Debug - Check camera parameters and inputs_embeds BEFORE model forward
        if inputs_embeds is not None and torch.isnan(inputs_embeds).any():
            print(f"\n{'='*60}")
            print(f"[JJ-CRITICAL-ERROR] *** NaN DETECTED at Step {self.global_step} ***")
            print(f"[JJ-ERROR] inputs_embeds contains NaN BEFORE model forward!")
            print(f"[JJ-ERROR] NaN count in inputs_embeds: {torch.isnan(inputs_embeds).sum()}")
            if self.intrisics is not None:
                print(f"[JJ-DEBUG-CAM] intrisics has NaN: {torch.isnan(self.intrisics).any().item()}")
            if self.extrinsics_w2c is not None:
                print(f"[JJ-DEBUG-CAM] extrinsics_w2c has NaN: {torch.isnan(self.extrinsics_w2c).any().item()}")
            print(f"{'='*60}\n")
            raise RuntimeError(
                f"[NaN DETECTED] inputs_embeds contains NaN at global_step={self.global_step}. "
                f"This indicates the problem occurred in video/spatial encoding or connector fusion. "
                f"Check spatial_encoder or connector outputs."
            )
        
        intrisics_down, extrinsics_w2c_down = downsample_cams(self.intrisics, self.extrinsics_w2c, 
                                        temporal_patch_size=2, 
                                        extrinsics_sample_strategy="mean",
                                        )
        
        # JJ: Debug - Check after downsampling
        if intrisics_down is not None and torch.isnan(intrisics_down).any():
            print(f"[JJ-ERROR] *** intrisics_down contains NaN AFTER downsampling! ***")
        if extrinsics_w2c_down is not None and torch.isnan(extrinsics_w2c_down).any():
            print(f"[JJ-ERROR] *** extrinsics_w2c_down contains NaN AFTER downsampling! ***")
        
        self.intrisics_down = intrisics_down
        self.extrinsics_w2c_down = extrinsics_w2c_down

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
            RoPE_attn_mode=self.RoPE_attn_mode,
            visual_token_mask=self.visual_token_mask,
            intrisics = self.intrisics_down,#B S 3 3
            extrinsics_w2c=self.extrinsics_w2c_down, #B S 3 4
        )

        hidden_states = outputs[0]
        
        # JJ: Debug - Check hidden_states for NaN/Inf
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            print(f"\n{'='*60}")
            print(f"[JJ-CRITICAL-ERROR] NaN DETECTED at Step {self.global_step}")
            print(f"[JJ-ERROR] hidden_states contains NaN or Inf AFTER model forward!")
            print(f"NaN count: {torch.isnan(hidden_states).sum()}")
            print(f"Inf count: {torch.isinf(hidden_states).sum()}")
            print(f"{'='*60}\n")
            raise RuntimeError(
                f"[NaN DETECTED] hidden_states contains NaN at global_step={self.global_step}. "
                f"This indicates the problem occurred INSIDE self.model() forward pass. "
                f"Likely in PRoPE or custom decoder layers."
            )
        
        logits = self.lm_head(hidden_states)
        
        # JJ: Debug - Check logits for NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"\n{'='*60}")
            print(f"[JJ-CRITICAL-ERROR] NaN DETECTED at Step {self.global_step}")
            print(f"[JJ-ERROR] logits contains NaN or Inf!")
            print(f"NaN count: {torch.isnan(logits).sum()}")
            print(f"Inf count: {torch.isinf(logits).sum()}")
            print(f"{'='*60}\n")
            raise RuntimeError(
                f"[NaN DETECTED] logits contains NaN at global_step={self.global_step}. "
                f"This is propagated from hidden_states."
            )

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
            
            # JJ: Check loss value and raise error if NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n{'='*60}")
                print(f"[JJ-CRITICAL-ERROR] NaN DETECTED at Step {self.global_step}")
                print(f"[JJ-ERROR] Loss is NaN or Inf!")
                print(f"[JJ-DEBUG-LOSS] labels != -100 count: {(labels != -100).sum().item()}")
                print(f"shift_logits stats - min: {shift_logits.min()}, max: {shift_logits.max()}")
                print(f"shift_logits NaN count: {torch.isnan(shift_logits).sum()}")
                print(f"{'='*60}\n")
                raise RuntimeError(
                    f"[NaN DETECTED] Loss is NaN at global_step={self.global_step}. "
                    f"This is the final manifestation of earlier NaN propagation."
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
