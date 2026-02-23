"""
Monkey Patch for Qwen3-VL to support Pose-aware RoPE (PHW mode).

Key differences from Qwen2.5 version:
1. Patches Qwen3's model.get_rope_index instead of the whole forward
2. Handles Qwen3's return signature: (position_ids, rope_deltas)
3. Replaces T dimension (which is fake=1 in Qwen3) with Pose dimension
4. Default mrope_section = [24, 20, 20] (Qwen3 default)
5. Adapts to Qwen3's interleaved mRoPE layout

This implementation reuses core logic from custom_qwen2_5_VLRoPE.py for:
- Lie scalar distance computation
- Pose aggregation and scaling
- Camera pose preprocessing
"""

from ast import Pass
from typing import Optional, Tuple
import torch
import numpy as np

# JJ: Reuse Pose computation from Qwen2.5 implementation
try:
    from src.utils.pose_distance_metrics import compute_lie_scalar_index_torch
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    root_dir = Path(__file__).resolve().parents[3]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.utils.pose_distance_metrics import compute_lie_scalar_index_torch


def patch_qwen3_with_pose_rope(
    model,
    use_pose_rope: bool = True,
    pose_enc_type: str = "PHW",
    mrope_section: Optional[list] = None,
    **kwargs,
):
    """
    Apply Pose RoPE monkey patch to SpatialMLLMQwen3.
    
    Args:
        model: SpatialMLLMQwen3ForConditionalGeneration instance
        use_pose_rope: Enable Pose RoPE (default: True)
        pose_enc_type: "PHW" (Pose+H+W) or "THW" (keep original Qwen3)
        mrope_section: Custom mrope_section (default: [24, 20, 20] for Qwen3)
    
    Strategy:
        1. Save original model.model.get_rope_index method
        2. Replace with get_rope_index_with_pose that:
           a. Calls original get_rope_index → get (3D position_ids, deltas)
           b. If pose_enc_type == "PHW": replace T dimension with Pose
           c. Return modified (position_ids, deltas)
        3. Update model.config.text_config.rope_parameters["mrope_section"] if provided
    
    Compatibility:
        - Works with Qwen3's textual timestamp encoding (timestamps are in text tokens)
        - Pose dimension replaces the fake T=1 in video frames' RoPE
        - Text tokens keep original 3D (T+H+W) encoding
    """
    
    if not use_pose_rope:
        print("[WARNING] use_pose_rope=False, skipping Pose RoPE patch.")
        return model
    
    # Validate pose_enc_type
    if pose_enc_type not in ["PHW", "THW"]:
        raise ValueError(f"Unsupported pose_enc_type for Qwen3: {pose_enc_type}. Only 'PHW' and 'THW' are supported.")
    
    if pose_enc_type == "THW":
        print("[INFO] pose_enc_type='THW' (no Pose dimension), using original Qwen3 RoPE.")
        return model
    
    # ==================== mrope_section Sanity Check ====================
    # Qwen3 default: [24, 20, 20] for head_dim=128
    #   - P/T: 24 dims
    #   - H: 20 dims
    #   - W: 20 dims
    #   - Total: 64 dims (half of head_dim=128)
    
    head_dim = model.config.text_config.hidden_size // model.config.text_config.num_attention_heads
    
    # ⚠️ TODO: Custom mrope_section is NOT supported yet!
    # Qwen3's mrope_section is saved as instance variable during __init__:
    #   self.mrope_section = config.rope_parameters.get("mrope_section", [24, 20, 20])
    # Modifying config AFTER initialization has NO EFFECT.
    # 
    # To support custom mrope_section, we need to:
    #   for layer in model.model.layers:
    #       layer.self_attn.rotary_emb.mrope_section = custom_mrope_section
    
    qwen3_default_mrope_section = [24, 20, 20]
    
    if mrope_section is not None and mrope_section != qwen3_default_mrope_section:
        raise NotImplementedError("Custom mrope_section is NOT YET SUPPORTED for Qwen3!")
        print(f"\n{'='*70}")
        print(f"⚠️  WARNING: Custom mrope_section is NOT YET SUPPORTED for Qwen3!")
        print(f"{'='*70}")
        print(f"  Requested: {mrope_section}")
        print(f"  Default:   {qwen3_default_mrope_section}")
        print(f"")
        print(f"  Reason: Qwen3's RoPE layer caches mrope_section during __init__.")
        print(f"          Modifying config after model loading has no effect.")
        print(f"")
        print(f"  Workaround: We will use the default value {qwen3_default_mrope_section}.")
        print(f"")
        print(f"  TODO: To support custom mrope_section, need to patch all layers:")
        print(f"        for layer in model.model.layers:")
        print(f"            layer.self_attn.rotary_emb.mrope_section = custom_value")
        print(f"{'='*70}\n")
        
        mrope_section = qwen3_default_mrope_section  # Force to use default
    elif mrope_section is None:
        mrope_section = qwen3_default_mrope_section
        print(f"[INFO] Using Qwen3-VL default mrope_section: {mrope_section}")
    else:
        # User explicitly provided default value
        print(f"[INFO] Using Qwen3-VL default mrope_section: {mrope_section}")
    
    # Validate default mrope_section
    expected_sum = head_dim // 2
    actual_sum = sum(mrope_section)
    if actual_sum != expected_sum:
        raise ValueError(
            f"Qwen3 default mrope_section sum ({actual_sum}) != head_dim/2 ({expected_sum}). "
            f"This indicates a model architecture mismatch!"
        )
    
    # Update model config (for documentation only, won't affect runtime)
    model.config.text_config.rope_parameters["mrope_section"] = mrope_section
    model.pose_enc_type = pose_enc_type
    
    # Store config for checkpoint saving
    model.config.pose_rope_config = {
        "use_pose_rope": True,
        "pose_enc_type": pose_enc_type,
        "mrope_section": mrope_section,
    }
    
    print(f"[INFO] ⚠️  NOTE: mrope_section cannot be changed after model initialization.")
    print(f"[INFO]    Currently using: {mrope_section} (Qwen3 default)")
    
    # ==================== Save Original Method ====================
    original_get_rope_index = model.model.get_rope_index
    
    # ==================== Define Patched Method ====================
    def get_rope_index_with_pose(
        input_ids: torch.LongTensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        JJ: Patched get_rope_index for PHW Pose RoPE.
        
        Critical Rules (SAME as Qwen2.5 version):
        1. ONLY works during Prefill (when video_tchw is passed via prepare_inputs_for_generation)
        2. Strictly REQUIRES model.selected_frames_poses (raises error if not available)
        3. Replaces T dimension with Pose distance computed from camera extrinsics
        
        Workflow:
        1. Check prerequisites (video_grid_thw, model.selected_frames_poses)
        2. Call original Qwen3's get_rope_index → (position_ids, deltas)
        3. Compute Pose distance from model.selected_frames_poses
        4. Replace position_ids[0] (T dimension) with Pose for video tokens
        5. Return modified (position_ids, deltas)
        """
        
        # ==================== Step 0: Prerequisites Check ====================
        # JJ: Strict error handling - no poses = raise error (as requested by user)
        if video_grid_thw is not None and model.selected_frames_poses is None:
            raise RuntimeError(
                "[Pose RoPE ERROR] video_grid_thw is provided but model.selected_frames_poses is None!\n"
                "This means VGGT pose computation was not triggered during forward().\n"
                "Possible causes:\n"
                "1. video_tchw was not passed to forward() (check prepare_inputs_for_generation)\n"
                "2. VGGT failed to initialize (check init_spatial_encoder)\n"
                "3. GPU memory is insufficient for VGGT (requires ~2GB during prefill)\n"
                "\n"
                "Solution: Ensure video_tchw is passed during prefill stage."
            )
        
        # Step 1: Call original Qwen3's get_rope_index
        position_ids, mrope_position_deltas = original_get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            **kwargs,
        )

        # JJ: Debug output for native THW RoPE BEFORE Pose modification (only during prefill)
        # Check if this is prefill stage: rope_deltas is None or position_ids has large sequence length
        is_prefill = (mrope_position_deltas is None or position_ids.shape[2] > 10)
        
        # if (model.offline_debug or True) and is_prefill and video_grid_thw is not None:
        if (model.offline_debug) and is_prefill and video_grid_thw is not None:
            print(f"\n" + "="*60)
            print(f"[Native THW RoPE BEFORE Pose] Original Qwen3 position_ids")
            print(f"video_grid_thw: {video_grid_thw}")
            print(f"position_ids shape: {position_ids.shape}")  # (3, B, L) for THW
            
            # Calculate ranges (SAME as PHW debug output)
            spatial_merge_size = model.config.vision_config.spatial_merge_size
            num_visual_tokens = video_grid_thw[0][0] * video_grid_thw[0][1] * video_grid_thw[0][2] // (spatial_merge_size * spatial_merge_size)
            textual_time_token_per_image = 9  # Qwen3's temporal text tokens between frames
            per_image_token = textual_time_token_per_image + video_grid_thw[0][1] * video_grid_thw[0][2] // (spatial_merge_size * spatial_merge_size)
            num_pre_text = 10
            num_post_text = position_ids.shape[2] - per_image_token * video_grid_thw[0][0] - num_pre_text
            inspect_range = per_image_token * video_grid_thw[0][0] + num_pre_text + 5
            
            print(f"\n[Native THW] Early Text Position_ids (3D: T,H,W):")
            print(f"T: {position_ids[0,0,:num_pre_text+3]}")
            print(f"H: {position_ids[1,0,:num_pre_text+3]}")
            print(f"W: {position_ids[2,0,:num_pre_text+3]}")
            
            print(f"Debug 300:350:")
            print(f"P: {position_ids[0,0,300:350]}")
            print(f"H: {position_ids[1,0,300:350]}")
            print(f"W: {position_ids[2,0,300:350]}")
            
            print(f"\n[Native THW] Middle (vision+text) Position_ids:")
            print(f"T: {position_ids[0,0,num_pre_text+1:inspect_range:per_image_token//4]}")
            print(f"H: {position_ids[1,0,num_pre_text+1:inspect_range:per_image_token//4]}")
            print(f"W: {position_ids[2,0,num_pre_text+1:inspect_range:per_image_token//4]}")
            
            print(f"\n[Native THW] Later (text) Position_ids:")
            print(f"T: {position_ids[0,0,-(num_post_text+5):]}")
            print(f"H: {position_ids[1,0,-(num_post_text+5):]}")
            print(f"W: {position_ids[2,0,-(num_post_text+5):]}")
            
            print(f"\n[Native THW] T dimension range: [{position_ids[0].min()}, {position_ids[0].max()}]")
            print(f"[Native THW] H dimension range: [{position_ids[1].min()}, {position_ids[1].max()}]")
            print(f"[Native THW] W dimension range: [{position_ids[2].min()}, {position_ids[2].max()}]")
            print(f"="*60 + "\n")

        
        # ==================== Debug: Print rope_deltas ====================
        if model.offline_debug:
            print(f"[Pose RoPE DEBUG] mrope_position_deltas shape: {mrope_position_deltas.shape}")
            print(f"[Pose RoPE DEBUG] mrope_position_deltas: {mrope_position_deltas}")
        
        # Step 2: Check if we need to add Pose dimension
        if video_grid_thw is None or model.selected_frames_poses is None:
            # No video or no pose data, return original

            return position_ids, mrope_position_deltas
        
        # ==================== Step 3: Compute Pose Distance ====================
        # JJ: Reuse logic from Qwen2.5 implementation
        #converst postion_ids from long to float, later can be merged wiht float pose rope
        position_ids = position_ids.float()
        # Extract extrinsics from selected_frames_poses (shape: [B, N_frames, 4, 4] or [N_frames, 4, 4])
        if model.selected_frames_poses.ndim == 4:
            assert model.selected_frames_poses.shape[0] == 1, "Qwen3 only supports one batch"
            extrinsics = model.selected_frames_poses[0]  # Take first batch
        else:
            extrinsics = model.selected_frames_poses
        
        # ==================== Step 3.1: Determine Reference Frame (SAME as Qwen2.5) ====================
        # JJ: Support both 'first' and 'medoid' strategies
        if model.pose_anchor_rereference_strategy == 'first':
            # Use first frame as anchor
            ref_frame_idx = 0
        elif model.pose_anchor_rereference_strategy == 'medoid':
            # Use medoid frame as anchor (frame with minimum sum of distances to all others)
            N = extrinsics.shape[0]
            device = extrinsics.device
            
            # Compute distance matrix: D[i,j] = distance between pose i and pose j
            distance_matrix = torch.zeros(N, N, device=device)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        # Compute relative pose: T_j_to_i = inv(T_i_to_w) @ T_j_to_w
                        pose_i_w2c = torch.linalg.inv(extrinsics[i])
                        pose_rel = pose_i_w2c @ extrinsics[j]
                        
                        # Compute Lie scalar distance (rotation angle + weighted translation)
                        R_rel = pose_rel[:3, :3]
                        t_rel = pose_rel[:3, 3]
                        
                        # Rotation angle from trace(R) = 1 + 2*cos(theta)
                        trace = torch.diagonal(R_rel).sum()
                        cos_theta = (trace - 1.0) / 2.0
                        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                        theta = torch.acos(cos_theta)
                        
                        # Translation distance
                        d_trans = torch.norm(t_rel)
                        
                        # Combined distance with pose_id_scalar_lambda_trans weighting
                        distance_matrix[i, j] = torch.sqrt(
                            theta**2 + (model.pose_id_scalar_lambda_trans**2) * d_trans**2
                        )
            
            # Find medoid: frame with minimum sum of distances
            dist_sums = distance_matrix.sum(dim=1)  # (N,)
            ref_frame_idx = torch.argmin(dist_sums).item()
            
            if model.offline_debug:
                print(f'[Pose RoPE] Reference frame index (medoid): {ref_frame_idx}')
        else:
            raise NotImplementedError(
                f"pose_anchor_rereference_strategy='{model.pose_anchor_rereference_strategy}' not supported. "
                f"Only 'first' and 'medoid' are implemented."
            )
        
        if model.offline_debug:
            print(f'[Pose RoPE] Reference frame index: {ref_frame_idx}')
        
        # Compute Lie scalar distance with explicit parameters (SAME as Qwen2.5)
        # JJ: All parameters explicitly specified for consistency and clarity
        pose_distances = compute_lie_scalar_index_torch(
            extrinsics,
            pose_id_scalar_lambda_trans=model.pose_id_scalar_lambda_trans,
            traj_scale_norm=True,        # Normalize translation per trajectory to [0,1]
            global_normalize=True,       # Normalize final P to [0,1]
            reorth_rot=True,             # Re-orthogonalize rotation matrices for numerical stability
            reference_frame_id=ref_frame_idx  # Use determined reference frame (SAME as Qwen2.5)
        )  # Shape: (N_frames,), range: [0, 1]
        print('Pose_distance_shape:',pose_distances.shape)
        print('Pose_distance:',pose_distances)
        
        # Sanity check (SAME as Qwen2.5)
        if pose_distances.min() < 0 or pose_distances.max() == 0:
            raise ValueError(
                f"[Pose RoPE ERROR] Invalid pose_distances: "
                f"min={pose_distances.min()}, max={pose_distances.max()}"
            )
        
        # ==================== Step 4: Aggregate and Scale Pose Distances ====================
        # JJ: Match video_grid_thw to aggregate same-frame poses
        # Qwen3 uses video_grid_thw[:, 0] for temporal dimension (after temporal downsampling)
        # Our poses may have multiple poses per frame (need to aggregate based on temporal_patch_size)
        
        # Get temporal_patch_size from vision config (SAME as Qwen2.5)
        temporal_patch_size = model.config.vision_config.temporal_patch_size  # Default: 2 for Qwen3
        num_video_frames = video_grid_thw[0][0].item()  # Number of video tokens after temporal downsampling (llm_grid_t)
        
        if model.offline_debug:
            print(f'[Pose RoPE] temporal_patch_size: {temporal_patch_size}')
            print(f'[Pose RoPE] num_video_frames (llm_grid_t): {num_video_frames}')
            print(f'[Pose RoPE] len(pose_distances): {len(pose_distances)}')
        
        # Sanity check: P length should match llm_grid_t * temporal_patch_size (SAME as Qwen2.5)
        expected_len = num_video_frames * temporal_patch_size
        if len(pose_distances) != expected_len:
            raise ValueError(
                f"[Pose RoPE ERROR] Expected P length {expected_len} "
                f"(num_video_frames={num_video_frames} * temporal_patch_size={temporal_patch_size}), "
                f"but got {len(pose_distances)}. "
                f"This indicates a mismatch between input frames and temporal downsampling."
            )
        
        # Aggregate P based on temporal_patch_size (SAME as Qwen2.5)
        selected_frames_tensor = pose_distances.view(num_video_frames, temporal_patch_size)
        
        if model.pose_merge_strategy == 'mean':
            pose_range_tensor = selected_frames_tensor.mean(dim=1)
        elif model.pose_merge_strategy == 'first':
            pose_range_tensor = selected_frames_tensor[:, 0]
        elif model.pose_merge_strategy == 'last':
            pose_range_tensor = selected_frames_tensor[:, -1]
        elif model.pose_merge_strategy == 'median':
            pose_range_tensor = selected_frames_tensor.median(dim=1)[0]
        else:
            raise ValueError(
                f"Unknown pose_merge_strategy: {model.pose_merge_strategy}. "
                f"Supported strategies: ['mean', 'first', 'last', 'median']"
            )
        
        if model.offline_debug:
            print(f'[Pose RoPE] After aggregation:')
            print(f'[Pose RoPE]   shape: {pose_range_tensor.shape}')
            print(f'[Pose RoPE]   values: {pose_range_tensor}')
        
        # Warning if pose and temporal strategies differ (SAME as Qwen2.5)
        # Note: temporal_readapted_merge_strategy is used for mRoPE_readaptT mode
        # For consistency, both should typically use the same strategy
        if hasattr(model, 'temporal_readapted_merge_strategy') and \
           model.pose_merge_strategy != model.temporal_readapted_merge_strategy:
            print(f"[WARNING] pose_merge_strategy ('{model.pose_merge_strategy}') != "
                  f"temporal_readapted_merge_strategy ('{model.temporal_readapted_merge_strategy}')")
        
        # Re-normalize after aggregation if configured (SAME as Qwen2.5)
        # Motivation: Aggregation (especially mean) hides the zero pose due to re-reference,
        # losing "reference frame" semantics. Re-norm restores this and ensures consistent
        # range utilization.
        if model.hard_reset_reference_after_pose_merge:
            pose_min = pose_range_tensor.min()
            pose_max = pose_range_tensor.max()
            pose_range = pose_max - pose_min
            if pose_range > 0:
                pose_range_tensor = (pose_range_tensor - pose_min) / pose_range  # Re-norm to [0, 1]
                if model.offline_debug:
                    print(f'[Pose RoPE] After re-normalization (hard_reset_reference):')
                    print(f'[Pose RoPE]   values: {pose_range_tensor}')
                    print(f'[Pose RoPE]   range: [{pose_range_tensor.min():.4f}, {pose_range_tensor.max():.4f}]')
        
        # Scale (re-)normalized pose to target range (SAME as Qwen2.5)
        if model.pose_use_dynamic_scale_factor:
            # Dynamic: scale to [0, len(selected_frames_poses) - 1]
            target_max_p_pos = len(model.selected_frames_poses) - 1
            pose_range_tensor = pose_range_tensor * target_max_p_pos
            if model.offline_debug:
                print(f'[Pose RoPE] Dynamic scaling: multiply by {target_max_p_pos}')
        else:
            # Fixed: scale to [0, pose_scale_factor]
            pose_range_tensor = pose_range_tensor * model.pose_scale_factor
            if model.offline_debug:
                print(f'[Pose RoPE] Fixed scaling: multiply by {model.pose_scale_factor}')
        
        print(f'[Pose RoPE] pose_use_dynamic_scale_factor: {model.pose_use_dynamic_scale_factor}')
        print(f'[Pose RoPE] pose_range_tensor after scale: {pose_range_tensor}')
        
        # JJ: Note - do NOT add offset here!
        # In Qwen2.5, offset (text_len + st_idx) is added when assembling vision_pos
        # In Qwen3 monkey patch, we handle offset during position_ids replacement (Step 5)
        
        # Final result (without offset yet)
        pose_distances = pose_range_tensor
        
        # ==================== Step 5: Replace T Dimension with Pose ====================
        # Find video token positions in position_ids
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Locate video tokens
        video_token_id = model.config.video_token_id
        
        for batch_idx in range(batch_size):
            # Find all video token positions in this batch
            video_token_positions = (input_ids[batch_idx] == video_token_id).nonzero(as_tuple=True)[0]
            
            if len(video_token_positions) == 0:
                continue
            
            # Qwen3 processes each frame separately with vision_start/end tokens
            # We need to map pose_distances to the actual vision feature tokens (not the placeholder)
            
            # For simplicity, we assume video tokens are contiguous and match the number of frames
            # This may need refinement based on actual Qwen3 tokenization
            
            # Calculate tokens per frame
            spatial_merge_size = model.config.vision_config.spatial_merge_size
            h, w = video_grid_thw[0][1].item(), video_grid_thw[0][2].item()
            tokens_per_frame = (h // spatial_merge_size) * (w // spatial_merge_size)
            
            # Replace T dimension (position_ids[0]) with Pose for video tokens
            for frame_idx, pose_dist in enumerate(pose_distances):
                start_idx = frame_idx * tokens_per_frame
                end_idx = start_idx + tokens_per_frame
                if start_idx < len(video_token_positions):
                    actual_positions = video_token_positions[start_idx:min(end_idx, len(video_token_positions))]
                    
                    # JJ: Apply offset logic (SAME as Qwen2.5 line 936)
                    if model.do_offset_in_pose_pos_id:
                        # Get the base offset from the first token's original position_ids[0]
                        # This contains text_len + st_idx (global offset from original Qwen3)
                        base_offset = position_ids[0, batch_idx, actual_positions[0]].item()
                        
                        # In Qwen3, video_grid_thw[:, 0] represents temporal dimension
                        # For each frame, original T is typically 0 or frame_idx
                        # We need to subtract the original T value to get pure offset
                        # Then add our pose_dist
                        
                        # Simpler approach: replace with (base_offset - original_T + pose_dist)
                        # But we don't know original_T exactly...
                        
                        # Even simpler: use the base_offset from first token of first frame
                        # All frames share similar structure, so we can compute relative offset
                        if frame_idx == 0:
                            # Store the base offset from first frame
                            first_frame_base = base_offset
                        
                        # For Qwen3 with "fake T=1", each frame has similar offset pattern
                        # Replace: original position_ids[0] → base_offset + pose_dist
                        # This ensures pose_dist is added on top of the global offset
                        pose_with_offset = first_frame_base + pose_dist
                        position_ids[0, batch_idx, actual_positions] = pose_with_offset#.long()
                    else:
                        # No offset: just use pose_dist directly (stay in normalized range)
                        position_ids[0, batch_idx, actual_positions] = pose_dist#.long()
        
        # ==================== Debug: Detailed Position IDs Printout ====================
        if model.offline_debug or True:
        # if model.offline_debug:
            print(f"*"*20)
            print(f"Details During Prefill (Qwen3 PHW Pose RoPE):")
            print(f"video_grid_thw: {video_grid_thw}")
            
            # Calculate ranges (following Qwen2.5 pattern)
            num_visual_tokens = video_grid_thw[0][0] * video_grid_thw[0][1] * video_grid_thw[0][2] // (spatial_merge_size * spatial_merge_size)
            textual_time_token_per_image = 9
            per_image_token = textual_time_token_per_image + video_grid_thw[0][1] * video_grid_thw[0][2] // (spatial_merge_size * spatial_merge_size)
            num_pre_text = 10
            num_post_text = position_ids.shape[2] - per_image_token*video_grid_thw[0][0] - num_pre_text
            num_visual_tokens_with_textual_time = per_image_token*video_grid_thw[0][0]
            inspect_range = num_visual_tokens_with_textual_time + num_pre_text + 5
            
            print(f"Prefill Position_ids:")
            print(f"{position_ids.shape}")  # (3, B, L) for PHW
            
            print(f"Pose_enc_type: {model.pose_enc_type}")
            
            # PHW mode: Print P, H, W dimensions separately
            print(f"Early Text Position_ids (3D: P,H,W):")
            print(f"P: {position_ids[0,0,:num_pre_text+3]}")
            print(f"H: {position_ids[1,0,:num_pre_text+3]}")
            print(f"W: {position_ids[2,0,:num_pre_text+3]}")
            
            print(f"Debug 300:330:")
            print(f"P: {position_ids[0,0,300:330]}")
            print(f"H: {position_ids[1,0,300:330]}")
            print(f"W: {position_ids[2,0,300:330]}")
            print(f"Debug 600:630:")
            print(f"P: {position_ids[0,0,600:630]}")
            print(f"H: {position_ids[1,0,600:630]}")
            print(f"W: {position_ids[2,0,600:630]}")
            print(f"Middle (vision) Position_ids:")
            print(f"P: {position_ids[0,0,num_pre_text+1:inspect_range:per_image_token//4]}")
            print(f"H: {position_ids[1,0,num_pre_text+1:inspect_range:per_image_token//4]}")
            print(f"W: {position_ids[2,0,num_pre_text+1:inspect_range:per_image_token//4]}")
            
            print(f"Later (Most) Position_ids:")
            print(f"P: {position_ids[0,0,-(num_post_text+5):]}")
            print(f"H: {position_ids[1,0,-(num_post_text+5):]}")
            print(f"W: {position_ids[2,0,-(num_post_text+5):]}")
            
            print(f"[Pose RoPE DEBUG] Modified position_ids[0] (Pose dim) range: "
                  f"[{position_ids[0].min():.4f}, {position_ids[0].max():.4f}]")
        
        # ==================== Sanity Check: rope_deltas ====================
        if model.offline_debug:
            print(f"[Pose RoPE SANITY CHECK] position_ids shape: {position_ids.shape}")
            print(f"[Pose RoPE SANITY CHECK] position_ids[0] (P/T) - min: {position_ids[0].min()}, max: {position_ids[0].max()}")
            print(f"[Pose RoPE SANITY CHECK] position_ids[1] (H) - min: {position_ids[1].min()}, max: {position_ids[1].max()}")
            print(f"[Pose RoPE SANITY CHECK] position_ids[2] (W) - min: {position_ids[2].min()}, max: {position_ids[2].max()}")
            print(f"[Pose RoPE SANITY CHECK] mrope_position_deltas: {mrope_position_deltas}")
        
        return position_ids, mrope_position_deltas
    
    # ==================== Apply Monkey Patch ====================
    original_get_rope_index = model.model.get_rope_index  # Save reference BEFORE patching
    model.model.get_rope_index = get_rope_index_with_pose
    
    print(f"[SUCCESS] Applied Pose RoPE monkey patch to Qwen3-VL:")
    print(f"          - Mode: {pose_enc_type} (Pose+H+W)")
    print(f"          - mrope_section: {mrope_section}")
    print(f"          - head_dim: {head_dim}")
    print(f"          - Pose scale factor: {model.pose_scale_factor}")
    print(f"          - Pose merge strategy: {model.pose_merge_strategy}")
    print(f"          - Pose anchor strategy: {model.pose_anchor_rereference_strategy}")
    print(f"          - offline_debug: {model.offline_debug}")
    print(f"[INFO] 🔍 Monkey patch will be triggered during model.generate()")
    print(f"[INFO] 🔍 To see debug output, ensure:")
    print(f"       1. video_tchw is passed to model.forward()")
    print(f"       2. model.selected_frames_poses is populated")
    print(f"       3. model.generate() is called with video inputs")
    
    return model


# ==================== Helper Functions (Reused from Qwen2.5) ====================

def aggregate_same_frame_poses_to_same_pos_id(
    pose_indices: torch.Tensor,
    video_grid_thw: torch.LongTensor,
    temporal_patch_size: int,
    merge_strategy: str = "mean",
) -> torch.Tensor:
    """
    Aggregate poses that belong to the same temporal frame.
    
    Args:
        pose_indices: (N,) tensor of pose indices
        video_grid_thw: (1, 3) tensor of [T, H, W]
        temporal_patch_size: Temporal patch size (e.g., 2 for Qwen3)
        merge_strategy: 'mean', 'first', 'last', or 'median'
    
    Returns:
        aggregated_indices: (T,) tensor where T = video_grid_thw[0, 0]
    """
    T = video_grid_thw[0, 0].item()
    N = len(pose_indices)
    
    # If N == T, no aggregation needed
    if N == T:
        return pose_indices
    
    # Compute how many original frames map to each temporal token
    frames_per_token = N // T
    
    if merge_strategy == "mean":
        aggregated = pose_indices.view(T, frames_per_token).mean(dim=1)
    elif merge_strategy == "first":
        aggregated = pose_indices[::frames_per_token][:T]
    elif merge_strategy == "last":
        aggregated = pose_indices[frames_per_token-1::frames_per_token][:T]
    elif merge_strategy == "median":
        aggregated = pose_indices.view(T, frames_per_token).median(dim=1).values
    else:
        raise ValueError(f"Unknown merge_strategy: {merge_strategy}")
    
    return aggregated
