"""
JJ : Camera conversion and Plücker ray utilities for LVSM integration.

Handles:
- VGGT w2c → c2w conversion
- Intrinsics 3x3 → fxfycxcy format with resolution rescaling
- Plücker ray computation (compatible with LVSM's compute_rays)
- Posed input construction (RGB + Plücker rays)
"""

import torch
from einops import rearrange


@torch.no_grad()
def w2c_to_c2w(extrinsics_w2c):
    """
    JJ : Convert VGGT w2c extrinsics to c2w format for LVSM.
    
    VGGT outputs [R|t] in w2c (world-to-camera) convention.
    LVSM expects c2w (camera-to-world) as 4x4 matrices.
    
    c2w = [R^T | -R^T @ t]
          [0   |    1     ]
    
    Args:
        extrinsics_w2c: [B, S, 3, 4] - VGGT w2c extrinsics
        
    Returns:
        c2w: [B, S, 4, 4] - camera-to-world matrices
    """
    B, S = extrinsics_w2c.shape[:2]
    device = extrinsics_w2c.device
    dtype = extrinsics_w2c.dtype
    
    R = extrinsics_w2c[..., :3, :3]  # [B, S, 3, 3]
    t = extrinsics_w2c[..., :3, 3:]  # [B, S, 3, 1]
    
    R_T = R.transpose(-1, -2)  # [B, S, 3, 3]
    t_c2w = -torch.matmul(R_T, t)  # [B, S, 3, 1]
    
    # Build 4x4 c2w matrix
    c2w = torch.zeros(B, S, 4, 4, device=device, dtype=dtype)
    c2w[..., :3, :3] = R_T
    c2w[..., :3, 3:] = t_c2w
    c2w[..., 3, 3] = 1.0
    
    return c2w


@torch.no_grad()
def intrinsics_to_fxfycxcy(intrinsics, target_h=256, target_w=256, orig_hw=None):
    """
    JJ : Convert 3x3 intrinsics to fxfycxcy format, rescaled for target resolution.
    
    Args:
        intrinsics: [B, S, 3, 3] - camera intrinsic matrices
        target_h: int - target image height (for LVSM, typically 256)
        target_w: int - target image width (for LVSM, typically 256)
        orig_hw: tuple (H_orig, W_orig) - original image resolution.
                 If None, estimated from intrinsics (cx*2, cy*2).
    
    Returns:
        fxfycxcy: [B, S, 4] - (fx, fy, cx, cy) rescaled for target resolution
    """
    fx = intrinsics[..., 0, 0]  # [B, S]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    
    if orig_hw is not None:
        H_orig, W_orig = orig_hw
    else:
        # Estimate original resolution from principal point (assumes center principal point)
        H_orig = float(2 * cy.max().item())
        W_orig = float(2 * cx.max().item())
    
    # Rescale intrinsics for target resolution
    scale_w = target_w / W_orig
    scale_h = target_h / H_orig
    
    fx = fx * scale_w
    fy = fy * scale_h
    cx = cx * scale_w
    cy = cy * scale_h
    
    fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1)  # [B, S, 4]
    return fxfycxcy


@torch.no_grad()
def compute_plucker_rays(c2w, fxfycxcy, h, w, device=None):
    """
    JJ : Compute ray origins and directions from camera parameters.
    
    Adapted from LVSM's ProcessData.compute_rays with bug fix for intrinsic rescaling.
    
    Args:
        c2w: [B, V, 4, 4] - camera-to-world matrices
        fxfycxcy: [B, V, 4] - (fx, fy, cx, cy) already at target resolution
        h: int - image height
        w: int - image width
        device: torch.device (optional, defaults to c2w.device)
        
    Returns:
        ray_o: [B, V, 3, h, w] - ray origins
        ray_d: [B, V, 3, h, w] - ray directions (normalized)
    """
    if device is None:
        device = c2w.device
        
    b, v = c2w.size()[:2]
    c2w_flat = c2w.reshape(b * v, 4, 4).to(device)
    fxfycxcy_flat = fxfycxcy.reshape(b * v, 4).to(device)
    
    # Create pixel grid
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    xx = xx[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1).float()
    yy = yy[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1).float()
    
    # Unproject to camera space (using rescaled intrinsics directly)
    x_cam = (xx + 0.5 - fxfycxcy_flat[:, 2:3]) / fxfycxcy_flat[:, 0:1]
    y_cam = (yy + 0.5 - fxfycxcy_flat[:, 3:4]) / fxfycxcy_flat[:, 1:2]
    z_cam = torch.ones_like(x_cam)
    
    # Ray direction in camera space → world space
    ray_d = torch.stack([x_cam, y_cam, z_cam], dim=2)  # [b*v, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w_flat[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # normalize
    
    # Ray origin = camera position in world space
    ray_o = c2w_flat[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]
    
    # Reshape to spatial layout
    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    
    return ray_o, ray_d


def get_posed_input(images=None, ray_o=None, ray_d=None):
    """
    JJ : Construct posed input by concatenating RGB images with Plücker ray conditioning.
    
    Uses default Plücker parameterization: [cross(o, d), d]
    
    Args:
        images: [B, V, 3, H, W] in [0, 1] range, or None for target pose only
        ray_o: [B, V, 3, H, W] - ray origins
        ray_d: [B, V, 3, H, W] - ray directions
        
    Returns:
        If images provided: [B, V, 9, H, W] (normalized_RGB + Plücker)
        If images is None:  [B, V, 6, H, W] (Plücker only, for target poses)
    """
    o_cross_d = torch.cross(ray_o, ray_d, dim=2)  # [B, V, 3, H, W]
    pose_cond = torch.cat([o_cross_d, ray_d], dim=2)  # [B, V, 6, H, W]
    
    if images is None:
        return pose_cond
    else:
        # LVSM convention: normalize images from [0,1] to [-1,1]
        return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)  # [B, V, 9, H, W]
