"""
Spatial MLLM based on Qwen3-VL with VGGT for Pose Estimation.

This model extends Qwen3VLForConditionalGeneration with:
1. VGGT Spatial Encoder for camera pose estimation (Pose RoPE only)
2. Support for Pose-aware RoPE (PHW mode via monkey patching)
3. Preserves all Qwen3 original features (Deepstack, etc.)

Key Design:
- VGGT is ONLY used for pose estimation (no feature fusion)
- All vision processing is handled by Qwen3 (including Deepstack)
- Pose is injected via Monkey Patch on get_rope_index
- Much simpler than Qwen2.5 version (no connector, no custom decoder)

Key differences from Qwen2.5 version:
- No spatial feature fusion (only pose estimation)
- No connector needed
- Qwen3 uses textual timestamps for temporal encoding
- Qwen3 uses 3D mRoPE (T+H+W) with interleaved layout
- Qwen3's get_rope_index returns (position_ids, rope_deltas)
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig

from src.qwenvl.model.spatial_encoder import VGGTSpatialEncoderConfig, VGGTSpatialEncoderPreTrainedModel
from src.qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri

# JJ: Camera pose utilities
from src.custom_qwenvl.model.camera_pose_temporal_merge import downsample_cams


class SpatialMLLMQwen3Config(Qwen3VLConfig):
    """Configuration for Spatial MLLM Qwen3."""
    model_type = "spatial-mllm-qwen3"

    def __init__(self, spatial_config=None, **kwargs):
        super().__init__(**kwargs)
        self.sub_configs["spatial_config"] = VGGTSpatialEncoderConfig
        
        if isinstance(spatial_config, dict):
            self.spatial_config = self.sub_configs["spatial_config"](**spatial_config)
        elif spatial_config is None:
            self.spatial_config = self.sub_configs["spatial_config"]()
        
        # JJ: Pose RoPE configuration (will be set by monkey patch)
        self.pose_rope_config = {
            "use_pose_rope": False,
            "pose_enc_type": "THW",  # Default: no Pose dimension
            "mrope_section": None,   # Will use Qwen3 default [24, 20, 20]
        }


class SpatialMLLMQwen3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    Spatial MLLM based on Qwen3-VL (Pose Estimation Only).
    
    Architecture:
        - Vision Encoder: Qwen3VLVisionModel (from transformers, unchanged)
        - Spatial Encoder: VGGTSpatialEncoder (VGGT for camera pose estimation ONLY)
        - Language Model: Qwen3 LLM with 3D mRoPE (unchanged)
        - ❌ No Connector: We don't fuse spatial features with vision features
        - ✅ Deepstack: Preserved (handled by parent Qwen3VLForConditionalGeneration)
    
    Key Design Principle:
        - VGGT is ONLY used for estimating camera poses
        - Poses are stored in self.selected_frames_poses
        - Monkey patch on get_rope_index reads poses and injects into position_ids
        - All vision processing is handled by parent Qwen3 (including Deepstack)
    
    Usage:
        >>> from src.custom_qwen3vl.model import SpatialMLLMQwen3ForConditionalGeneration
        >>> from src.custom_qwen3vl.model import patch_qwen3_with_pose_rope
        >>> 
        >>> model = SpatialMLLMQwen3ForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-7B")
        >>> model = patch_qwen3_with_pose_rope(model, pose_enc_type="PHW")
    """
    
    config_class = SpatialMLLMQwen3Config
    
    def __init__(self, config: SpatialMLLMQwen3Config):
        # Initialize parent Qwen3-VL model
        super().__init__(config)
        
        # JJ: CRITICAL FIX for meta tensor issue
        # Save spatial config but DON'T initialize VGGT yet (it contains .item() calls)
        # VGGT will be initialized in _post_load_hook() after model is on a real device
        self._spatial_config = config.spatial_config
        self.spatial_encoder = None  # Placeholder
        
        # ❌ No Connector needed (we don't fuse features)
        # self.connector = ...
        
        # ==================== Pose RoPE Parameters ====================
        # These parameters will be used by the monkey patch
        # (see spatial_mllm_qwen3_pose_rope.py)
        
        # Pose encoding strategy (will be set by monkey patch)
        self.pose_enc_type = "THW"  # Default: no Pose (options: "PHW", "THW")
        
        # Pose dimension parameters (same as Qwen2.5 version for consistency)
        self.pose_scale_factor = 16.0
        self.pose_merge_strategy = "mean"  # Options: 'mean', 'first', 'last', 'median'
        self.pose_use_dynamic_scale_factor = True
        self.pose_anchor_rereference_strategy = 'first'  # Options: 'first', 'medoid'
        self.pose_id_scalar_lambda_trans = 1.0
        self.hard_reset_reference_after_pose_merge = True
        self.do_offset_in_pose_pos_id = True
        
        # Storage for runtime pose data (used by monkey patch)
        self.selected_frames_poses = None  # Will be set during forward
        self.current_video_grid_thw = None
        
        # Camera parameters (for debugging/logging if needed)
        self.intrisics = None
        self.extrinsics_w2c = None
        
        # Debug flags
        self.offline_debug = False
        self.global_step = 0
        self.current_epoch = 0
        
        # Initialize weights and apply final processing
        self.post_init()
        
        print(f"[INFO] SpatialMLLMQwen3 initialized (VGGT will be initialized after load):")
        print(f"       - VGGT for pose estimation: ⏳ (pending)")
        print(f"       - Spatial feature fusion: ❌ (not needed)")
        print(f"       - Qwen3 Deepstack: ✅ (preserved)")
        print(f"       - Pose RoPE: will be enabled via monkey patch")
    
    def init_spatial_encoder(self):
        """
        Initialize VGGT Spatial Encoder after model is loaded to a real device.
        
        This MUST be called AFTER from_pretrained() completes, because VGGT's
        vision_transformer.py uses .item() which is incompatible with meta tensors.
        """
        if self.spatial_encoder is not None:
            print("[WARNING] Spatial encoder already initialized, skipping.")
            return
        
        print("[INFO] Initializing VGGT Spatial Encoder on real device...")
        self.spatial_encoder = VGGTSpatialEncoderPreTrainedModel(self._spatial_config)
        
        # Move to the same device as the main model
        device = next(self.parameters()).device
        self.spatial_encoder = self.spatial_encoder.to(device)
        
        print(f"[INFO] ✅ VGGT initialized on device: {device}")

    
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
        cache_position: Optional[torch.LongTensor] = None,
        # JJ: Additional parameters for spatial encoding (pose estimation ONLY)
        image_tchw: Optional[List[torch.FloatTensor]] = None,
        video_tchw: Optional[List[torch.FloatTensor]] = None,
        # ⚠️ NOTE: second_per_grid_ts is NOT used in Qwen3 (uses textual timestamps instead)
        # Kept here for API compatibility but will be ignored
        # second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with camera pose estimation for Pose RoPE.
        
        Key workflow:
        1. If video_tchw provided: estimate camera poses with VGGT
        2. Store poses in self.selected_frames_poses
        3. Call parent's forward → Qwen3 handles all vision processing (including Deepstack)
        4. Monkey patch on get_rope_index will read self.selected_frames_poses and inject into position_ids
        
        Args:
            image_tchw: List of image tensors (T, C, H, W) for pose estimation
            video_tchw: List of video tensors (T, C, H, W) for pose estimation
            
        Note:
            - We don't modify inputs_embeds (no feature fusion)
            - All vision processing is handled by parent Qwen3VLForConditionalGeneration
            - Pose is injected via monkey patch, not here
        """
        
        # JJ: Track training progress (only increment during training with labels)
        if labels is not None:
            self.global_step += 1
            if self.offline_debug and self.global_step % 10 == 0:
                print(f"[Step {self.global_step}] Forward pass...")
        
        # ==================== VGGT Pose Estimation ====================
        # JJ: Estimate camera poses for Pose RoPE (NO feature fusion)
        
        if pixel_values_videos is not None and video_tchw is not None:
            # Check if VGGT is initialized
            if self.spatial_encoder is None:
                raise RuntimeError(
                    "[ERROR] VGGT spatial encoder is not initialized!\n"
                    "Please call model.init_spatial_encoder() after from_pretrained()."
                )
            
            # Get camera pose encodings from VGGT (we ONLY need pose, not features)
            _, _, camera_encs = self.spatial_encoder(
                video_tchw, 
                grid_thw=video_grid_thw, 
                return_cam_enc=True
            )
            
            # Sanity check
            assert len(camera_encs) == 1 and len(camera_encs[0]) == 4, \
                "camera_encs must have only one element and the last element must be a 9D pose encoding"
            assert len(camera_encs[0][-1]) == 2 * video_grid_thw[0][0], \
                f"Expected {2 * video_grid_thw[0][0]} camera poses, got {len(camera_encs[0][-1])}"
            
            # Extract camera extrinsics (w2c format)
            self.extrinsics_w2c, self.intrisics = pose_encoding_to_extri_intri(
                camera_encs[0][-1].unsqueeze(0), 
                video_tchw[0][-1].shape[-2:]
            )
            
            # JJ: Convert w2c to c2w poses (SAME as Qwen2.5 implementation)
            # extrinsics_w2c: (B=1, S, 3, 4) where S is number of frames
            # Convert to float32 to avoid BFloat16 SVD issues
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
            
            # JJ: Store c2w poses for Pose RoPE (NO downsampling, SAME as Qwen2.5)
            self.selected_frames_poses = c2w_poses.detach()  # (S, 4, 4) in float32
            self.current_video_grid_thw = video_grid_thw
            
            if self.offline_debug:
                print(f"[Pose Estimation] Extracted {S} camera poses (c2w format)")
                print(f"[Pose Estimation] Stored in self.selected_frames_poses for Monkey Patch")
        
        elif pixel_values is not None and image_tchw is not None:
            # Image pose estimation (if needed in the future)
            _, _, camera_encs = self.spatial_encoder(
                image_tchw, 
                return_cam_enc=True
            )
            self.extrinsics_w2c, self.intrisics = pose_encoding_to_extri_intri(
                camera_encs[0][-1].unsqueeze(0), 
                image_tchw[0][-1].shape[-2:]
            )
            # For images, no temporal downsampling needed
            self.selected_frames_poses = self.extrinsics_w2c
        
        # ==================== Call Parent Forward ====================
        # JJ: Let Qwen3 handle everything:
        #     - Vision encoding (including Deepstack)
        #     - Position IDs computation (will use our monkey-patched get_rope_index)
        #     - Language modeling
        #     - Loss computation
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )
        
        # JJ: Debug - Check for NaN in loss
        if labels is not None and outputs.loss is not None:
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                print(f"\n{'='*60}")
                print(f"[JJ-CRITICAL-ERROR] NaN/Inf DETECTED at Step {self.global_step}")
                print(f"[JJ-ERROR] Loss is NaN or Inf!")
                print(f"{'='*60}\n")
                raise RuntimeError(
                    f"[NaN DETECTED] Loss is NaN at global_step={self.global_step}."
                )
            elif self.offline_debug and self.global_step % 10 == 0:
                print(f"[Step {self.global_step}] Loss: {outputs.loss.item():.6f}")
        
        return outputs
    
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
        is_first_iteration=False,
        **kwargs,
    ):
        """
        JJ: Override to compute poses BEFORE calling super().prepare_inputs_for_generation.
        This ensures model.selected_frames_poses is populated before get_rope_index is called.
        
        Same mechanism as Qwen2.5 version.
        """
        
        # JJ: CRITICAL - Compute poses BEFORE calling super() on first iteration
        # Because super().prepare_inputs_for_generation will call model.get_rope_index,
        # which requires model.selected_frames_poses to be set
        if is_first_iteration and "video_tchw" in kwargs and kwargs["video_tchw"] is not None:
            video_tchw = kwargs["video_tchw"]
            
            if self.offline_debug:
                print(f"[Prepare Inputs] 🔄 Computing poses for Prefill stage...")
                print(f"[Prepare Inputs] video_tchw[0] shape: {video_tchw[0].shape}")
            
            # Initialize spatial encoder if needed
            if self.spatial_encoder is None and self._spatial_config is not None:
                self._init_spatial_encoder()
            
            # Compute poses using VGGT
            with torch.no_grad():
                _, _, camera_encs = self.spatial_encoder(
                    video_tchw,
                    grid_thw=video_grid_thw,
                    return_cam_enc=True
                )
                
                from src.qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri
                
                # Extract camera extrinsics (w2c format)
                extrinsics_w2c, _ = pose_encoding_to_extri_intri(
                    camera_encs[0][-1].unsqueeze(0),
                    video_tchw[0][-1].shape[-2:]
                )
                
                # Convert w2c to c2w poses (SAME as Qwen2.5)
                w2c_matrices = extrinsics_w2c[0].float()  # (S, 3, 4)
                S = w2c_matrices.shape[0]
                
                c2w_poses = torch.zeros(S, 4, 4, device=w2c_matrices.device, dtype=torch.float32)
                R_w2c = w2c_matrices[:, :3, :3]
                t_w2c = w2c_matrices[:, :3, 3:4]
                R_c2w = R_w2c.transpose(-2, -1)
                t_c2w = -torch.bmm(R_c2w, t_w2c)
                c2w_poses[:, :3, :3] = R_c2w
                c2w_poses[:, :3, 3:4] = t_c2w
                c2w_poses[:, 3, 3] = 1.0
                
                # Store c2w poses (NO downsampling)
                self.selected_frames_poses = c2w_poses.detach()
                self.current_video_grid_thw = video_grid_thw
                
                if self.offline_debug:
                    print(f"[Prepare Inputs] ✅ Poses computed: {S} frames")
        
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
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        
        # JJ: Clear pixel inputs after prefill (same as original Qwen3 and Qwen2.5)
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            # Also clear video_tchw to avoid reprocessing
            if "video_tchw" in model_inputs:
                model_inputs["video_tchw"] = None
            if "image_tchw" in model_inputs:
                model_inputs["image_tchw"] = None
        
        return model_inputs


if __name__ == "__main__":
    """
    Quick test to verify the model initialization.
    """
    print("Testing SpatialMLLMQwen3ForConditionalGeneration...")
    
    config = SpatialMLLMQwen3Config()
    model = SpatialMLLMQwen3ForConditionalGeneration(config)
    
    print(f"Model type: {type(model)}")
    print(f"Has spatial_encoder: {hasattr(model, 'spatial_encoder')}")
    print(f"Has connector: {hasattr(model, 'connector')}")  # Should be False now
    print(f"Has selected_frames_poses: {hasattr(model, 'selected_frames_poses')}")
    print("Test complete!")
