"""
Monkey Patch for CustomSpatialMLLMForConditionalGeneration to support 4D Pose-aware RoPE.

This module provides a drop-in replacement forward method that adds 4D PTHW RoPE support
while keeping the original 3D mRoPE functionality intact.

Usage:
    from src.custom_qwenvl.model.custom_spatial_mllm_pose_rope import patch_model_with_pose_rope
    
    model = CustomSpatialMLLMForConditionalGeneration(config)
    patch_model_with_pose_rope(model)  # Apply monkey patch
"""

from typing import List, Optional, Tuple, Union
import torch
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

# 🆕 NEW: Import 4D Pose RoPE function
from src.custom_qwenvl.model.custom_qwen2_5_VLRoPE import (
    custom_get_rope_index,           # Original 3D mRoPE
    custom_get_pose_rope_index       # New 4D Pose RoPE
)
from src.qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.custom_qwenvl.model.camera_pose_temporal_merge import downsample_cams


def forward_with_pose_rope(
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
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    """
    Forward pass with 4D Pose-aware RoPE support.
    
    This is a monkey-patched version of the original forward method.
    Changes are marked with:
        # 🆕 NEW: Newly added code
        # ✏️ MODIFIED: Modified from original
        # (unmarked code is unchanged from original)
    """
    
    # ========================================
    # JJ: Debug flag - Comment/uncomment to enable/disable NaN detection
    # ========================================
    ENABLE_POSE_ROPE_NAN_DEBUG = False  # Set to True to enable detailed NaN detection at 6 checkpoints
    # ENABLE_POSE_ROPE_NAN_DEBUG = True  # Uncomment this line to enable debug mode
    
    # ========================================
    # Phase 1: Setup (UNCHANGED)
    # ========================================
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        
        # ========== DEBUG: Check text embeddings ==========
        if ENABLE_POSE_ROPE_NAN_DEBUG:
            print(f"\n[DEBUG-NaN-0] === Checking text embeddings (after embed_tokens) ===")
            print(f"[DEBUG-NaN-0] inputs_embeds shape: {inputs_embeds.shape}")
            print(f"[DEBUG-NaN-0] inputs_embeds dtype: {inputs_embeds.dtype}")
            print(f"[DEBUG-NaN-0] inputs_embeds has NaN: {torch.isnan(inputs_embeds).any()}")
            print(f"[DEBUG-NaN-0] inputs_embeds has Inf: {torch.isinf(inputs_embeds).any()}")
            if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
                print(f"[ERROR] Text embeddings contain NaN/Inf!")
                raise RuntimeError("NaN/Inf detected in text embeddings!")
            print(f"[DEBUG-NaN-0] ========================================\n")
        
        # ========================================
        # Phase 2: Image Processing (UNCHANGED)
        # ========================================
        if pixel_values is not None:
            assert False, 'Should not reach here...'
            assert image_tchw is not None, "`image_tchw` must be provided when `pixel_values` is not None."
            pixel_values = pixel_values.type(self.visual.dtype)
            image_tchw = [image_tchw_i.type(self.visual.dtype) for image_tchw_i in image_tchw]

            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            spatial_embeds_list, patch_start_idx = self.spatial_encoder(image_tchw)

            # JJ: Conditional fusion based on skip_connector flag
            if self.skip_connector:
                # Skip connector fusion, use visual embeddings only
                fused_embeds = image_embeds
            else:
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

        # ========================================
        # Phase 3: Video Processing
        # ========================================
        if pixel_values_videos is not None:
            assert video_tchw is not None, "`video_tchw` must be provided when `pixel_values_videos` is not None."
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_tchw = [video_tchw_i.type(self.visual.dtype) for video_tchw_i in video_tchw]
            
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            
            # ========== DEBUG: Check video embeddings ==========
            if ENABLE_POSE_ROPE_NAN_DEBUG:
                print(f"\n[DEBUG-NaN-3] === Checking video embeddings (after visual encoder) ===")
                print(f"[DEBUG-NaN-3] video_embeds shape: {video_embeds.shape}")
                print(f"[DEBUG-NaN-3] video_embeds dtype: {video_embeds.dtype}")
                print(f"[DEBUG-NaN-3] video_embeds has NaN: {torch.isnan(video_embeds).any()}")
                print(f"[DEBUG-NaN-3] video_embeds has Inf: {torch.isinf(video_embeds).any()}")
                if torch.isnan(video_embeds).any() or torch.isinf(video_embeds).any():
                    print(f"[ERROR] Video embeddings contain NaN/Inf!")
                    raise RuntimeError("NaN/Inf detected in video_embeds!")
                print(f"[DEBUG-NaN-3] ========================================\n")
            
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            spatial_embeds_list, patch_start_idx, camera_encs = self.spatial_encoder(
                video_tchw, grid_thw=video_grid_thw, return_cam_enc=True
            )
            
            # ========== DEBUG: Check spatial embeddings ==========
            if ENABLE_POSE_ROPE_NAN_DEBUG:
                print(f"\n[DEBUG-NaN-4] === Checking spatial embeddings (after spatial encoder) ===")
                print(f"[DEBUG-NaN-4] spatial_embeds_list length: {len(spatial_embeds_list)}")
                for i, sp_emb in enumerate(spatial_embeds_list):
                    print(f"[DEBUG-NaN-4] spatial_embeds_list[{i}] shape: {sp_emb.shape}")
                    print(f"[DEBUG-NaN-4] spatial_embeds_list[{i}] has NaN: {torch.isnan(sp_emb).any()}")
                    print(f"[DEBUG-NaN-4] spatial_embeds_list[{i}] has Inf: {torch.isinf(sp_emb).any()}")
                    if torch.isnan(sp_emb).any() or torch.isinf(sp_emb).any():
                        print(f"[ERROR] Spatial embeddings contain NaN/Inf at index {i}!")
                        raise RuntimeError(f"NaN/Inf detected in spatial_embeds_list[{i}]!")
                print(f"[DEBUG-NaN-4] ========================================\n")
            
            # Extract camera parameters (UNCHANGED)
            self.extrinsics_w2c, self.intrisics = pose_encoding_to_extri_intri(
                camera_encs[0][-1].unsqueeze(0), video_tchw[0][-1].shape[-2:]
            )
            assert len(camera_encs) == 1 and len(camera_encs[0]) == 4, "camera_encs must have only one element and the last element must be a 9D pose encoding"
            assert len(camera_encs[0][-1])== 2 * video_grid_thw[0][0]

            # 🆕 NEW: Extract camera poses for Pose RoPE
            if self.use_pose_rope:
                # Convert w2c to c2w poses
                # extrinsics_w2c: (B=1, S, 3, 4) where S is number of frames
                # ✏️ FIXED: Convert to float32 to avoid BFloat16 SVD issues
                w2c_matrices = self.extrinsics_w2c[0].float()  # (S, 3, 4), convert to float32
                S = w2c_matrices.shape[0]
                
                # Build full 4x4 homogeneous transformation matrices
                c2w_poses = torch.zeros(S, 4, 4, device=w2c_matrices.device, dtype=torch.float32)
                
                # w2c -> c2w: need to invert
                # w2c = [R|t] -> c2w = [R^T | -R^T * t]
                R_w2c = w2c_matrices[:, :3, :3]  # (S, 3, 3)
                t_w2c = w2c_matrices[:, :3, 3:4]  # (S, 3, 1)
                
                R_c2w = R_w2c.transpose(-2, -1)  # (S, 3, 3)
                t_c2w = -torch.bmm(R_c2w, t_w2c)  # (S, 3, 1)
                
                c2w_poses[:, :3, :3] = R_c2w
                c2w_poses[:, :3, 3:4] = t_c2w
                c2w_poses[:, 3, 3] = 1.0
                
                # Store for use in position_ids computation
                self.selected_frames_poses = c2w_poses.detach()  # (S, 4, 4) in float32

            # JJ: Conditional fusion based on skip_connector flag
            if self.skip_connector:
                # Skip connector fusion, use visual embeddings only
                fused_embeds = video_embeds
            else:
                # Connector fusion (UNCHANGED)
                fused_embeds, _, _ = self.connector(
                    video_embeds=video_embeds,
                    spatial_embeds_list=spatial_embeds_list,
                    patch_start_idx=patch_start_idx,
                    grid_thw=video_grid_thw,
                )
            
            # ========== DEBUG: Check fused embeddings ==========
            if ENABLE_POSE_ROPE_NAN_DEBUG:
                print(f"\n[DEBUG-NaN-5] === Checking fused embeddings (after connector) ===")
                print(f"[DEBUG-NaN-5] fused_embeds shape: {fused_embeds.shape}")
                print(f"[DEBUG-NaN-5] fused_embeds dtype: {fused_embeds.dtype}")
                print(f"[DEBUG-NaN-5] fused_embeds has NaN: {torch.isnan(fused_embeds).any()}")
                print(f"[DEBUG-NaN-5] fused_embeds has Inf: {torch.isinf(fused_embeds).any()}")
                if torch.isnan(fused_embeds).any() or torch.isinf(fused_embeds).any():
                    print(f"[ERROR] Fused embeddings contain NaN/Inf!")
                    raise RuntimeError("NaN/Inf detected in fused_embeds!")
                print(f"[DEBUG-NaN-5] ========================================\n")

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, fused_embeds)
            
            # ========== DEBUG: Check final inputs_embeds ==========
            if ENABLE_POSE_ROPE_NAN_DEBUG:
                print(f"\n[DEBUG-NaN-6] === Checking final inputs_embeds (after fusion) ===")
                print(f"[DEBUG-NaN-6] inputs_embeds shape: {inputs_embeds.shape}")
                print(f"[DEBUG-NaN-6] inputs_embeds dtype: {inputs_embeds.dtype}")
                print(f"[DEBUG-NaN-6] inputs_embeds has NaN: {torch.isnan(inputs_embeds).any()}")
                print(f"[DEBUG-NaN-6] inputs_embeds has Inf: {torch.isinf(inputs_embeds).any()}")
                if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
                    nan_count = torch.isnan(inputs_embeds).sum().item()
                    print(f"[ERROR] Final inputs_embeds contain NaN/Inf! NaN count: {nan_count}")
                    raise RuntimeError("NaN/Inf detected in final inputs_embeds after fusion!")
                print(f"[DEBUG-NaN-6] ========================================\n")

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # ========================================
    # Phase 4: Position IDs Computation
    # ========================================
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # Prefill stage: calculate RoPE index once
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            # ✏️ MODIFIED: Bug fix - check for mRoPE_readaptT instead of mRoPE
            assert self.position_ids_compute_mode in ["mRoPE_woT", "mRoPE", "mRoPE_readaptT"]
            if self.position_ids_compute_mode == "mRoPE_readaptT":  # Fixed from "mRoPE"
                assert selected_frames is not None, "`selected_frames` must be provided when `position_ids_compute_mode` is `mRoPE_readaptT`"
            
            # 🆕 NEW: Branch based on use_pose_rope flag
            if self.use_pose_rope:
                # ========== DEBUG: Check poses before position_ids computation ==========
                if ENABLE_POSE_ROPE_NAN_DEBUG:
                    print(f"\n[DEBUG-NaN-1] === Checking poses before position_ids computation ===")
                    if self.selected_frames_poses is not None:
                        print(f"[DEBUG-NaN-1] selected_frames_poses shape: {self.selected_frames_poses.shape}")
                        print(f"[DEBUG-NaN-1] selected_frames_poses has NaN: {torch.isnan(self.selected_frames_poses).any()}")
                        print(f"[DEBUG-NaN-1] selected_frames_poses has Inf: {torch.isinf(self.selected_frames_poses).any()}")
                        if torch.isnan(self.selected_frames_poses).any() or torch.isinf(self.selected_frames_poses).any():
                            print(f"[ERROR] Poses contain NaN/Inf before position_ids computation!")
                            print(f"[ERROR] Poses:\n{self.selected_frames_poses}")
                            raise RuntimeError("NaN/Inf detected in selected_frames_poses!")
                    else:
                        print(f"[DEBUG-NaN-1] selected_frames_poses is None")
                    print(f"[DEBUG-NaN-1] ========================================\n")
                
                # Use 4D Pose-aware RoPE
                position_ids, rope_deltas, visual_token_mask = custom_get_pose_rope_index(
                    config=self.config,
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    selected_frames_poses=self.selected_frames_poses,  # 🆕 NEW: Camera poses (S, 4, 4)
                    selected_frames_id=selected_frames,  # 🆕 NEW: Frame indices for temporal adjustment
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                    pose_enc_type=self.pose_enc_type,  # 🆕 NEW: 'PTHW'
                    temporal_patch_size=self.model.config.vision_config.temporal_patch_size,
                    # Pose parameters
                    pose_scale_factor=self.pose_scale_factor,  # 🆕 NEW: Renamed
                    pose_merge_strategy=self.pose_merge_strategy,  # 🆕 NEW
                    pose_use_dynamic_scale_factor=self.pose_use_dynamic_scale_factor,  # 🆕 NEW
                    pose_anchor_rereference_strategy=self.pose_anchor_rereference_strategy,  # 🆕 NEW
                    pose_id_scalar_lambda_trans=self.pose_id_scalar_lambda_trans,  # 🆕 NEW
                    hard_reset_reference_after_pose_merge=self.hard_reset_reference_after_pose_merge,  # 🆕 NEW
                    do_offset_in_pose_pos_id=self.do_offset_in_pose_pos_id,  # 🆕 NEW: Renamed
                    # Temporal parameters
                    THW_position_ids_compute_mode=self.position_ids_compute_mode,  # 🆕 NEW: Renamed parameter
                    temporal_readapted_merge_strategy=self.temporal_readapted_merge_strategy,  # 🆕 NEW: Renamed
                    temporal_readapted_use_dynamic_scale_factor=self.temporal_readapted_use_dynamic_scale_factor,  # 🆕 NEW: Renamed
                    temporal_readapted_scale_factor=self.temporal_readapted_scale_factor,  # 🆕 NEW: Renamed
                )
                # position_ids: (4, B, L) with dtype=float32, includes Pose dimension
                
                # ========== DEBUG: Check position_ids after computation ==========
                if ENABLE_POSE_ROPE_NAN_DEBUG:
                    print(f"\n[DEBUG-NaN-2] === Checking position_ids after 4D RoPE computation ===")
                    print(f"[DEBUG-NaN-2] position_ids shape: {position_ids.shape}")
                    print(f"[DEBUG-NaN-2] position_ids dtype: {position_ids.dtype}")
                    print(f"[DEBUG-NaN-2] position_ids has NaN: {torch.isnan(position_ids).any()}")
                    print(f"[DEBUG-NaN-2] position_ids has Inf: {torch.isinf(position_ids).any()}")
                    if torch.isnan(position_ids).any() or torch.isinf(position_ids).any():
                        print(f"[ERROR] position_ids contain NaN/Inf!")
                        print(f"[ERROR] P min/max: {position_ids[0].min()}/{position_ids[0].max()}")
                        print(f"[ERROR] T min/max: {position_ids[1].min()}/{position_ids[1].max()}")
                        print(f"[ERROR] H min/max: {position_ids[2].min()}/{position_ids[2].max()}")
                        print(f"[ERROR] W min/max: {position_ids[3].min()}/{position_ids[3].max()}")
                        raise RuntimeError("NaN/Inf detected in position_ids after 4D RoPE!")
                    print(f"[DEBUG-NaN-2] position_ids P min/max: {position_ids[0].min()}/{position_ids[0].max()}")
                    print(f"[DEBUG-NaN-2] position_ids T min/max: {position_ids[1].min()}/{position_ids[1].max()}")
                    print(f"[DEBUG-NaN-2] ========================================\n")
            else:
                # Use original 3D mRoPE (UNCHANGED)
                position_ids, rope_deltas, visual_token_mask = custom_get_rope_index(
                    self.config,
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                    position_ids_compute_mode=self.position_ids_compute_mode,
                    selected_frames_id=selected_frames,
                    temporal_patch_size=self.model.config.vision_config.temporal_patch_size,
                    temporal_readapted_merge_strategy=self.temporal_readapted_merge_strategy,  # Renamed
                    temporal_readapted_use_dynamic_scale_factor=self.temporal_readapted_use_dynamic_scale_factor,  # Renamed
                    temporal_readapted_scale_factor=self.temporal_readapted_scale_factor,  # Renamed
                )
                # position_ids: (3, B, L) with dtype=long, T/H/W dimensions
            
            self.visual_token_mask = visual_token_mask
            self.rope_deltas = rope_deltas

            # ✏️ MODIFIED: Debug output adjusted for 4D support
            if self.offline_debug:
                print(f"*"*20)
                print(f"Details During Prefill:")
                print(f"video_grid_thw: {video_grid_thw}")
                num_visual_tokens = video_grid_thw[0][0] * video_grid_thw[0][1] * video_grid_thw[0][2] // (2*2)
                per_image_token_num=video_grid_thw[0][1] * video_grid_thw[0][2] // (2*2)
                num_pre_text = 15
                num_post_text = position_ids.shape[2] - num_visual_tokens - num_pre_text
                inspect_range = num_visual_tokens + num_pre_text + 5

                print(f"second_per_grid_ts: {second_per_grid_ts}")
                print(f"Prefill Position_ids:")
                print(f"{position_ids.shape}")  # (4, B, L) or (3, B, L)

                if self.use_pose_rope:
                    # 🆕 NEW: 4D position_ids debug output
                    print(f"Early Text Position_ids (4D: P,T,H,W):")
                    print(f"P: {position_ids[0,0,:num_pre_text+3]}")
                    print(f"T: {position_ids[1,0,:num_pre_text+3]}")
                    print(f"H: {position_ids[2,0,:num_pre_text+3]}")
                    print(f"W: {position_ids[3,0,:num_pre_text+3]}")
                    
                    print(f"Middle (vision) Position_ids:")
                    print(f"P: {position_ids[0,0,num_pre_text:inspect_range:per_image_token_num//2]}")
                    print(f"T: {position_ids[1,0,num_pre_text:inspect_range:per_image_token_num//2]}")
                    
                    print(f"Later (Most) Position_ids:")
                    print(f"P: {position_ids[0,0,-(num_post_text+5):]}")
                    print(f"T: {position_ids[1,0,-(num_post_text+5):]}")
                else:
                    # Original 3D position_ids debug output (UNCHANGED)
                    print(f"Early Text Position_ids:")
                    print(f"({position_ids[:,0,:num_pre_text+3]})")
                    print(f"Middle (vision) Position_ids:")
                    print(f"({position_ids[:,0,num_pre_text:inspect_range:per_image_token_num//2]})")
                    print(f"Later (Most) Position_ids:")
                    print(f"({position_ids[:,0,-(num_post_text+5):]})")

        # Generation stage: use pre-calculated rope-deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            assert seq_length == 1, "seq_length must be 1 for generation"
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
            )
            if self.offline_debug:
                print('cache position[0]: ', cache_position[0])
                print('rope deltas: ', self.rope_deltas)
                print('delta: ', delta)
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            
            # ✏️ MODIFIED: Adjust dimensions based on use_pose_rope
            if self.use_pose_rope:
                position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)  # 🆕 NEW: 4D for generation
            else:
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # Original 3D

            self.visual_token_mask = torch.zeros_like(position_ids)[0]

    # ========================================
    # Phase 5: Training Progress Tracking (UNCHANGED)
    # ========================================
    if labels is not None:
        self.global_step += 1
        if self.global_step % 1 == 0:
            print(f"[Step {self.global_step}] Forward pass...")
    
    # ========================================
    # Phase 6: NaN Detection - inputs_embeds (UNCHANGED)
    # ========================================
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
    
    # ========================================
    # Phase 7: Camera Parameter Downsampling (UNCHANGED)
    # ========================================
    intrisics_down, extrinsics_w2c_down = downsample_cams(
        self.intrisics, self.extrinsics_w2c, 
        temporal_patch_size=2, 
        extrinsics_sample_strategy="mean"
    )
    
    if intrisics_down is not None and torch.isnan(intrisics_down).any():
        print(f"[JJ-ERROR] *** intrisics_down contains NaN AFTER downsampling! ***")
    if extrinsics_w2c_down is not None and torch.isnan(extrinsics_w2c_down).any():
        print(f"[JJ-ERROR] *** extrinsics_w2c_down contains NaN AFTER downsampling! ***")
    
    self.intrisics_down = intrisics_down
    self.extrinsics_w2c_down = extrinsics_w2c_down

    # ========================================
    # Phase 8: Model Forward (UNCHANGED)
    # ========================================
    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,  # Now can be (4, B, L) or (3, B, L)
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
        intrisics=self.intrisics_down,
        extrinsics_w2c=self.extrinsics_w2c_down,
    )

    hidden_states = outputs[0]
    
    # ========================================
    # Phase 9: NaN Detection - hidden_states (UNCHANGED)
    # ========================================
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
    
    # ========================================
    # Phase 10: NaN Detection - logits (UNCHANGED)
    # ========================================
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

    # ========================================
    # Phase 11: Loss Computation (UNCHANGED)
    # ========================================
    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
        # NaN detection for loss
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


def patch_model_with_pose_rope(
    model,
    use_pose_rope=True,
    pose_enc_type="PTHW",
):
    """
    Apply monkey patch to enable 4D Pose-aware RoPE.
    
    Args:
        model: CustomSpatialMLLMForConditionalGeneration instance
        use_pose_rope: Whether to use 4D Pose RoPE (True) or 3D mRoPE (False)
        pose_enc_type: Pose encoding type, currently only 'PTHW' supported
        
    Note:
        All RoPE parameters are inherited from model's existing attributes:
        
        Temporal Parameters (defined in model.__init__):
        - model.temporal_readapted_merge_strategy
        - model.temporal_readapted_use_dynamic_scale_factor
        - model.temporal_readapted_scale_factor
        
        Pose Parameters (defined in model.__init__):
        - model.pose_scale_factor
        - model.pose_merge_strategy
        - model.pose_use_dynamic_scale_factor
        - model.pose_anchor_rereference_strategy
        - model.hard_reset_reference_after_pose_merge
        - model.do_offset_in_pose_pos_id
        
        Debug Control:
        - To enable NaN detection, edit ENABLE_POSE_ROPE_NAN_DEBUG at the top of forward_with_pose_rope()
    
    Example:
        >>> model = CustomSpatialMLLMForConditionalGeneration(config)
        >>> patch_model_with_pose_rope(model, use_pose_rope=True)
        >>> # Now model.forward uses 4D Pose RoPE
    """
    # 🆕 NEW: Add Pose RoPE configuration attributes
    model.use_pose_rope = use_pose_rope
    model.pose_enc_type = pose_enc_type
    
    # Verify that model has all required Temporal parameters (should be defined in __init__)
    assert hasattr(model, 'temporal_readapted_merge_strategy'), \
        "Model must have temporal_readapted_merge_strategy attribute (defined in model.__init__)"
    assert hasattr(model, 'temporal_readapted_use_dynamic_scale_factor'), \
        "Model must have temporal_readapted_use_dynamic_scale_factor attribute (defined in model.__init__)"
    assert hasattr(model, 'temporal_readapted_scale_factor'), \
        "Model must have temporal_readapted_scale_factor attribute (defined in model.__init__)"
    
    # Verify that model has all required Pose parameters (should be defined in __init__)
    assert hasattr(model, 'pose_scale_factor'), \
        "Model must have pose_scale_factor attribute (defined in model.__init__)"
    assert hasattr(model, 'pose_merge_strategy'), \
        "Model must have pose_merge_strategy attribute (defined in model.__init__)"
    assert hasattr(model, 'pose_use_dynamic_scale_factor'), \
        "Model must have pose_use_dynamic_scale_factor attribute (defined in model.__init__)"
    assert hasattr(model, 'pose_anchor_rereference_strategy'), \
        "Model must have pose_anchor_rereference_strategy attribute (defined in model.__init__)"
    assert hasattr(model, 'pose_id_scalar_lambda_trans'), \
        "Model must have pose_id_scalar_lambda_trans attribute (defined in model.__init__)"
    assert hasattr(model, 'hard_reset_reference_after_pose_merge'), \
        "Model must have hard_reset_reference_after_pose_merge attribute (defined in model.__init__)"
    assert hasattr(model, 'do_offset_in_pose_pos_id'), \
        "Model must have do_offset_in_pose_pos_id attribute (defined in model.__init__)"
    
    # Runtime state
    model.selected_frames_poses = None  # Will be populated during forward
    
    # Replace forward method
    import types
    model.forward = types.MethodType(forward_with_pose_rope, model)
    
    # Print configuration
    print(f"[INFO] ========================================")
    print(f"[INFO] Model patched with Pose RoPE support.")
    print(f"[INFO] ========================================")
    print(f"[INFO] - use_pose_rope: {use_pose_rope}")
    print(f"[INFO] - pose_enc_type: {pose_enc_type}")
    print(f"[INFO]")
    print(f"[INFO] Temporal Parameters (inherited from model.__init__):")
    print(f"[INFO]   - temporal_readapted_merge_strategy: {model.temporal_readapted_merge_strategy}")
    print(f"[INFO]   - temporal_readapted_use_dynamic_scale_factor: {model.temporal_readapted_use_dynamic_scale_factor}")
    print(f"[INFO]   - temporal_readapted_scale_factor: {model.temporal_readapted_scale_factor}")
    print(f"[INFO]")
    print(f"[INFO] Pose Parameters (inherited from model.__init__):")
    print(f"[INFO]   - pose_scale_factor: {model.pose_scale_factor}")
    print(f"[INFO]   - pose_merge_strategy: {model.pose_merge_strategy}")
    print(f"[INFO]   - pose_use_dynamic_scale_factor: {model.pose_use_dynamic_scale_factor}")
    print(f"[INFO]   - pose_anchor_rereference_strategy: {model.pose_anchor_rereference_strategy}")
    print(f"[INFO]   - pose_id_scalar_lambda_trans: {model.pose_id_scalar_lambda_trans}")
    print(f"[INFO]   - hard_reset_reference_after_pose_merge: {model.hard_reset_reference_after_pose_merge}")
    print(f"[INFO]   - do_offset_in_pose_pos_id: {model.do_offset_in_pose_pos_id}")
    print(f"[INFO]")
    print(f"[INFO] 💡 To enable NaN detection, edit ENABLE_POSE_ROPE_NAN_DEBUG in forward_with_pose_rope()")
    print(f"[INFO] ========================================")
    
    return model

