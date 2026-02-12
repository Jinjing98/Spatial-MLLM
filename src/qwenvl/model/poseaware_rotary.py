"""
Pose-Aware Rotary Embedding for Multi-Camera Vision-Language Models

This module implements pose-aware transformations that apply camera extrinsics
as block-diagonal matrices to query and key embeddings in attention.

Key concepts:
- Camera extrinsics (4x4 pose matrices) are applied to visual token embeddings
- head_dim is treated as groups of 4D vectors (requires head_dim % 4 == 0)
- Query uses extrinsics, Key uses inverse extrinsics (ensures relative pose invariance)
- All attention heads and decoder layers share the same transformation

Design philosophy inspired by PRoPE (Perspective-aware Rotary Position Embedding),
but simplified to only use pose transformations without spatial RoPE.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


def extend_extrinsics_to_4x4(extrinsics_3x4: torch.Tensor) -> torch.Tensor:
    """
    Extend [B, num_cams, 3, 4] extrinsics to [B, num_cams, 4, 4] by adding [0,0,0,1] row.
    
    Args:
        extrinsics_3x4: Camera extrinsics [B, num_cams, 3, 4]
    
    Returns:
        extrinsics_4x4: Camera extrinsics [B, num_cams, 4, 4]
    """
    B, num_cams, _, _ = extrinsics_3x4.shape
    assert extrinsics_3x4.shape[2:] == (3, 4), f"Expected shape [..., 3, 4], got {extrinsics_3x4.shape}"
    
    # Create bottom row [0, 0, 0, 1]
    bottom_row = torch.zeros(B, num_cams, 1, 4, device=extrinsics_3x4.device, dtype=extrinsics_3x4.dtype)
    bottom_row[:, :, :, 3] = 1.0
    
    # Concatenate
    extrinsics_4x4 = torch.cat([extrinsics_3x4, bottom_row], dim=2)
    
    return extrinsics_4x4


def compute_camera_token_assignment(
    visual_token_mask: torch.Tensor,
    num_cams: int,
) -> torch.Tensor:
    """
    Compute which camera each visual token belongs to.
    
    Assumes visual tokens are temporally ordered and aligned with cameras:
    - First 1/num_cams visual tokens -> camera 0
    - Second 1/num_cams visual tokens -> camera 1
    - etc.
    
    Args:
        visual_token_mask: [B, seq_len] boolean mask for visual tokens
        num_cams: Number of cameras
    
    Returns:
        camera_indices: [B, seq_len] integer tensor, -1 for non-visual tokens, [0, num_cams-1] for visual tokens
    """
    B, seq_len = visual_token_mask.shape
    camera_indices = torch.full((B, seq_len), -1, dtype=torch.long, device=visual_token_mask.device)
    
    for b in range(B):
        visual_positions = torch.where(visual_token_mask[b])[0]
        num_visual_tokens = len(visual_positions)
        
        if num_visual_tokens == 0:
            continue
        
        # Divide visual tokens equally among cameras
        tokens_per_cam = num_visual_tokens // num_cams
        assert num_visual_tokens % num_cams == 0, \
            f"num_visual_tokens={num_visual_tokens} must be divisible by num_cams={num_cams}"
        
        for cam_idx in range(num_cams):
            start_idx = cam_idx * tokens_per_cam
            end_idx = (cam_idx + 1) * tokens_per_cam
            token_positions = visual_positions[start_idx:end_idx]
            camera_indices[b, token_positions] = cam_idx
    
    return camera_indices


def precompute_pose_transform_info(
    extrinsics_w2c: torch.Tensor,
    visual_token_mask: torch.Tensor,
    head_dim: int,
    scope_range: tuple = (0.0, 1.0),
) -> dict:
    """
    Precompute pose transformation information for efficient application in attention layers.
    
    This function:
    1. Extends 3x4 extrinsics to 4x4 (if needed)
    2. Computes inverse extrinsics for keys
    3. Assigns each visual token to its camera
    4. Validates head_dim is divisible by 4
    5. Computes the dimension range to apply PRoPE transformation
    
    Args:
        extrinsics_w2c: Camera extrinsics [B, num_cams, 3, 4] or [B, num_cams, 4, 4]
        visual_token_mask: [B, seq_len] boolean mask for visual tokens
        head_dim: Dimension of each attention head
        scope_range: Tuple (start_ratio, end_ratio) defining the portion of head_dim to transform.
                     E.g., (0.0, 0.25) applies PRoPE only to the first 25% of dimensions.
                     Default (0.0, 1.0) applies to all dimensions.
    
    Returns:
        pose_info: Dictionary containing:
            - 'extrinsics_4x4': [B, num_cams, 4, 4] pose matrices for queries
            - 'extrinsics_transpose': [B, num_cams, 4, 4] transpose of pose matrices
            - 'extrinsics_inv_4x4': [B, num_cams, 4, 4] inverse pose matrices for keys
            - 'camera_indices': [B, seq_len] camera assignment for each token
            - 'head_dim': head dimension
            - 'num_4d_groups': number of 4D groups in head_dim
            - 'scope_start': starting dimension index for PRoPE
            - 'scope_end': ending dimension index for PRoPE
    """
    # Validate head_dim
    assert head_dim % 4 == 0, f"head_dim={head_dim} must be divisible by 4 for 4x4 pose transformation"
    num_4d_groups = head_dim // 4
    
    # Compute scope dimensions (must be divisible by 4)
    start_ratio, end_ratio = scope_range
    assert 0.0 <= start_ratio < end_ratio <= 1.0, f"Invalid scope_range={scope_range}"
    
    scope_start = int(head_dim * start_ratio)
    scope_end = int(head_dim * end_ratio)
    
    # Round to nearest multiple of 4
    scope_start = (scope_start // 4) * 4
    scope_end = ((scope_end + 3) // 4) * 4  # Round up
    scope_end = min(scope_end, head_dim)  # Ensure within bounds
    
    # Ensure non-empty scope
    if scope_end <= scope_start:
        scope_end = scope_start + 4
    
    num_4d_groups_in_scope = (scope_end - scope_start) // 4
    
    # Extend extrinsics to 4x4 if needed
    if extrinsics_w2c.shape[-2:] == (3, 4):
        extrinsics_4x4 = extend_extrinsics_to_4x4(extrinsics_w2c)
    elif extrinsics_w2c.shape[-2:] == (4, 4):
        extrinsics_4x4 = extrinsics_w2c
    else:
        raise ValueError(f"extrinsics_w2c must have shape [..., 3, 4] or [..., 4, 4], got {extrinsics_w2c.shape}")
    
    B, num_cams, _, _ = extrinsics_4x4.shape
    
    # JJ: FIXME May need to norm tran across video.
    # set the translation part to 0
    # extrinsics_4x4[:,:,:3,-1] = 0.0

    # matrix_eye_dbg = torch.eye(4).expand(extrinsics_w2c.shape[0], 
                                        #  extrinsics_w2c.shape[1], 4, 4).to(extrinsics_w2c.device).to(extrinsics_w2c.dtype)
    # extrinsics_4x4 = matrix_eye_dbg
    # extrinsics_inv_4x4 = matrix_eye_dbg
    # extrinsics_transpose = matrix_eye_dbg

    # Compute transpose and inverse for PRoPE-style transformations
    def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
        """Invert a 4x4 SE(3) matrix."""
        assert transforms.shape[-2:] == (4, 4)
        Rinv = transforms[..., :3, :3].transpose(-1, -2)
        out = torch.zeros_like(transforms)
        out[..., :3, :3] = Rinv
        out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
        out[..., 3, 3] = 1.0
        return out

    extrinsics_transpose = extrinsics_4x4.transpose(-1, -2)  # P^T
    extrinsics_inv_4x4 = _invert_SE3(extrinsics_4x4)

    # Assign cameras to tokens
    camera_indices = compute_camera_token_assignment(visual_token_mask, num_cams)
    
    pose_info = {
        'extrinsics_4x4': extrinsics_4x4,          # P: camera<-world [B, num_cams, 4, 4]
        'extrinsics_transpose': extrinsics_transpose,  # P^T [B, num_cams, 4, 4]
        'extrinsics_inv_4x4': extrinsics_inv_4x4,  # P^(-1): world<-camera [B, num_cams, 4, 4]
        'camera_indices': camera_indices,          # [B, seq_len]
        'head_dim': head_dim,
        'num_4d_groups': num_4d_groups,
        'num_4d_groups_in_scope': num_4d_groups_in_scope,
        'num_cams': num_cams,
        'scope_start': scope_start,  # Starting dimension for PRoPE
        'scope_end': scope_end,      # Ending dimension for PRoPE
        'scope_range': scope_range,  # Original scope range tuple
    }
    
    return pose_info


def apply_pose_transform_to_embeddings(
    embeddings: torch.Tensor,
    pose_matrices: torch.Tensor,
    camera_indices: torch.Tensor,
    scope_start: int = 0,
    scope_end: int = None,
) -> torch.Tensor:
    """
    Apply 4x4 pose transformation to embeddings using camera assignments.
    
    The head_dim is treated as num_4d_groups groups of 4D vectors.
    Each 4D vector is transformed by the corresponding camera's 4x4 pose matrix.
    Only dimensions in [scope_start, scope_end) are transformed; others remain unchanged.
    
    Efficient implementation: assumes visual tokens are sequentially ordered by camera.
    
    Args:
        embeddings: [B, num_heads, seq_len, head_dim] attention states
        pose_matrices: [B, num_cams, 4, 4] pose transformation matrices
        camera_indices: [B, seq_len] camera assignment (-1 for non-visual tokens)
        scope_start: Starting dimension index for transformation (default: 0)
        scope_end: Ending dimension index for transformation (default: head_dim)
    
    Returns:
        transformed_embeddings: [B, num_heads, seq_len, head_dim] with pose applied to visual tokens
    """
    B, num_heads, seq_len, head_dim = embeddings.shape
    num_cams = pose_matrices.shape[1]
    
    # Set default scope_end
    if scope_end is None:
        scope_end = head_dim
    
    # Validate scope
    assert 0 <= scope_start < scope_end <= head_dim, f"Invalid scope [{scope_start}, {scope_end}) for head_dim={head_dim}"
    assert scope_start % 4 == 0 and scope_end % 4 == 0, f"Scope must be divisible by 4, got [{scope_start}, {scope_end})"
    
    scope_dim = scope_end - scope_start
    num_4d_groups_in_scope = scope_dim // 4
    
    assert head_dim % 4 == 0, f"head_dim={head_dim} must be divisible by 4"
    num_4d_groups = head_dim // 4
    
    # Clone to avoid in-place modification
    output = embeddings.clone()
    
    # Find visual token range for each batch
    for b in range(B):
        visual_mask = camera_indices[b] >= 0
        if not visual_mask.any():
            continue
        
        # Get start and end indices of visual tokens (they should be contiguous)
        visual_indices = torch.where(visual_mask)[0]
        start_idx = visual_indices[0].item()
        end_idx = visual_indices[-1].item() + 1
        num_visual_tokens = end_idx - start_idx
        
        # Validate that visual tokens are contiguous
        assert len(visual_indices) == num_visual_tokens, \
            "Visual tokens must be contiguous for efficient implementation"
        
        # Validate equal division among cameras
        assert num_visual_tokens % num_cams == 0, \
            f"num_visual_tokens={num_visual_tokens} must be divisible by num_cams={num_cams}"
        tokens_per_cam = num_visual_tokens // num_cams
        
        # Extract visual token embeddings: [num_heads, num_visual_tokens, head_dim]
        visual_embs = embeddings[b, :, start_idx:end_idx, :]
        
        # Extract only the scope dimensions: [num_heads, num_visual_tokens, scope_dim]
        visual_embs_scope = visual_embs[:, :, scope_start:scope_end]
        
        # Reshape to group by cameras: [num_heads, num_cams, tokens_per_cam, num_4d_groups_in_scope, 4]
        visual_embs_scope = visual_embs_scope.view(num_heads, num_cams, tokens_per_cam, num_4d_groups_in_scope, 4).contiguous()
        
        # Reshape for batched matmul: [num_heads, num_cams, tokens_per_cam * num_4d_groups_in_scope, 4]
        visual_embs_flat = visual_embs_scope.view(num_heads, num_cams, tokens_per_cam * num_4d_groups_in_scope, 4)
        
        # Get pose matrices for this batch: [num_cams, 4, 4]
        pose_batch = pose_matrices[b]  # [num_cams, 4, 4]
        
        # Expand for each head: [num_heads, num_cams, 4, 4]
        pose_expanded = pose_batch.unsqueeze(0).expand(num_heads, num_cams, 4, 4)


        # Batched matrix multiplication: 
        # [num_heads, num_cams, tokens_per_cam * num_4d_groups_in_scope, 4] @ [num_heads, num_cams, 4, 4]^T
        # Result: [num_heads, num_cams, tokens_per_cam * num_4d_groups_in_scope, 4]
        transformed = torch.matmul(visual_embs_flat, pose_expanded.transpose(-2, -1))
        
        # Reshape back: [num_heads, num_cams, tokens_per_cam, num_4d_groups_in_scope, 4]
        transformed = transformed.view(num_heads, num_cams, tokens_per_cam, num_4d_groups_in_scope, 4)
        
        # Reshape to scope shape: [num_heads, num_visual_tokens, scope_dim]
        transformed = transformed.view(num_heads, num_visual_tokens, scope_dim)
        
        # Write back only the transformed scope dimensions
        output[b, :, start_idx:end_idx, scope_start:scope_end] = transformed
    
    return output


def apply_poseaware_rotary(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    pose_info: Optional[dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply PRoPE-style pose-aware transformations to Q, K, V.
    
    Following the PRoPE design:
    - Query: P^T (transpose of camera extrinsics)
    - Key:   P^(-1) (inverse of camera extrinsics)
    - Value: P^(-1) (inverse of camera extrinsics)
    
    Note: Output transformation (P) is applied after attention computation.
    
    This design transforms visual tokens into a canonical world coordinate system
    where attention is computed, achieving camera pose equivariance.
    
    Args:
        query_states: [B, num_heads, seq_len, head_dim]
        key_states: [B, num_heads, seq_len, head_dim]
        value_states: [B, num_heads, seq_len, head_dim]
        pose_info: Dictionary from precompute_pose_transform_info(), or None to skip
    
    Returns:
        query_states_transformed: [B, num_heads, seq_len, head_dim]
        key_states_transformed: [B, num_heads, seq_len, head_dim]
        value_states_transformed: [B, num_heads, seq_len, head_dim]
    """
    if pose_info is None:
        # No pose transformation, return as is
        return query_states, key_states, value_states
    
    # Extract pose info
    extrinsics_4x4 = pose_info['extrinsics_4x4']           # P: camera<-world
    extrinsics_transpose = pose_info['extrinsics_transpose']  # P^T
    extrinsics_inv_4x4 = pose_info['extrinsics_inv_4x4']  # P^(-1): world<-camera
    camera_indices = pose_info['camera_indices']
    scope_start = pose_info.get('scope_start', 0)
    scope_end = pose_info.get('scope_end', None)
    
    # PRoPE transformations: convert to canonical world coordinate system
    # Only apply to dimensions in [scope_start, scope_end)
    query_states_transformed = apply_pose_transform_to_embeddings(
        query_states, extrinsics_transpose, camera_indices, scope_start, scope_end
    )
    
    key_states_transformed = apply_pose_transform_to_embeddings(
        key_states, extrinsics_inv_4x4, camera_indices, scope_start, scope_end
    )
    
    value_states_transformed = apply_pose_transform_to_embeddings(
        value_states, extrinsics_inv_4x4, camera_indices, scope_start, scope_end
    )
    
    return query_states_transformed, key_states_transformed, value_states_transformed


def apply_poseaware_output_transform(
    attn_output: torch.Tensor,
    pose_info: Optional[dict],
) -> torch.Tensor:
    """
    Apply PRoPE-style output transformation (P) after attention computation.
    
    This transforms the attention output from world coordinates back to camera coordinates,
    completing the PRoPE transformation cycle: camera -> world (compute attention) -> camera.
    
    Args:
        attn_output: [B, num_heads, seq_len, head_dim] attention output
        pose_info: Dictionary from precompute_pose_transform_info(), or None to skip
    
    Returns:
        attn_output_transformed: [B, num_heads, seq_len, head_dim]
    """
    if pose_info is None:
        return attn_output
    
    # Extract pose info
    extrinsics_4x4 = pose_info['extrinsics_4x4']  # P: camera<-world
    camera_indices = pose_info['camera_indices']
    scope_start = pose_info.get('scope_start', 0)
    scope_end = pose_info.get('scope_end', None)
    
    # Transform back to camera coordinates (only on scope dimensions)
    attn_output_transformed = apply_pose_transform_to_embeddings(
        attn_output, extrinsics_4x4, camera_indices, scope_start, scope_end
    )
    
    return attn_output_transformed

