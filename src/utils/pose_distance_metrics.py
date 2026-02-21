# JJ: Compute farness between camera poses
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Union
from scipy.spatial.transform import Rotation as R

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_pose_farness(
    poses_c2w: Union[List[np.ndarray], torch.Tensor],
    trans_metric_mode: str = 'euclidean',
    rot_metric_mode: str = 'angle_axis',
    reorth_rot: bool = True,
    translation_scale: Union[None, float] = 1.0,
    reference_frame_id: int = 0
) -> Tuple[List[float], List[float]]:
    """
    Compute farness (distance) between camera poses relative to a reference pose.
    Supports both NumPy arrays and PyTorch tensors.
    
    Args:
        poses_c2w: Camera-to-world poses. List of 4x4 transformation matrices (NumPy) 
                   or tensor of shape (N, 4, 4) (PyTorch)
        trans_metric_mode: Translation distance metric, options:
            - 'euclidean': Euclidean distance ||t1 - t2||
        rot_metric_mode: Rotation distance metric, options:
            - 'angle_axis': Angle from angle-axis representation of relative rotation
        reorth_rot: Before computing rotation metric, re-orthogonalize rotations using SVD (default: True)
        translation_scale: Scale factor for translation distances (default: 1.0)
            - None: Raw distances (no scaling)
            - 1.0: Normalize so max translation distance = 1.0
        reference_frame_id: Index of the reference frame (default: 0)
    
    Returns:
        farness_trans: List of translation distances (same length as poses_c2w)
        farness_rot: List of rotation distances in radians (same length as poses_c2w)
    """
    # Sanity check: poses should not be empty
    if TORCH_AVAILABLE and isinstance(poses_c2w, torch.Tensor):
        if poses_c2w.shape[0] == 0:
            raise ValueError("Input poses_c2w is empty")
        if poses_c2w.ndim != 3 or poses_c2w.shape[1] != 4 or poses_c2w.shape[2] != 4:
            raise ValueError(f"Expected poses_c2w shape (N, 4, 4), got {poses_c2w.shape}")
        return compute_pose_farness_torch(poses_c2w, trans_metric_mode, rot_metric_mode, reorth_rot, translation_scale, reference_frame_id)
    else:
        if not isinstance(poses_c2w, list) or len(poses_c2w) == 0:
            raise ValueError("Input poses_c2w must be a non-empty list")
        return compute_pose_farness_numpy(poses_c2w, trans_metric_mode, rot_metric_mode, reorth_rot, translation_scale, reference_frame_id)


def compute_pose_farness_numpy(
    poses_c2w: List[np.ndarray],
    trans_metric_mode: str = 'euclidean',
    rot_metric_mode: str = 'angle_axis',
    reorth_rot: bool = True,
    translation_scale: Union[None, float] = 1.0,
    reference_frame_id: int = 0
) -> Tuple[List[float], List[float]]:
    """
    NumPy implementation of pose farness computation.
    
    Args:
        poses_c2w: List of 4x4 camera-to-world transformation matrices
        trans_metric_mode: Translation distance metric
        rot_metric_mode: Rotation distance metric
        reorth_rot: Before computing rotation metric, re-orthogonalize rotations using SVD (default: True)
        translation_scale: Scale factor for translation distances (default: 1.0)
        reference_frame_id: Index of the reference frame (default: 0)
    
    Returns:
        farness_trans: List of translation distances
        farness_rot: List of rotation distances in radians
    """
    if len(poses_c2w) == 0:
        return [], []
    
    # Validate reference_frame_id
    if not (0 <= reference_frame_id < len(poses_c2w)):
        raise ValueError(
            f"reference_frame_id={reference_frame_id} out of range [0, {len(poses_c2w)})"
        )
    
    # Reference pose is the specified one
    ref_c2w = np.array(poses_c2w[reference_frame_id])
    ref_t = ref_c2w[:3, 3]
    ref_R = ref_c2w[:3, :3]
    
    # Re-orthogonalize reference rotation if requested
    if reorth_rot:
        ref_R = _reorthogonalize_rotation_numpy(ref_R)
    
    farness_trans = []
    farness_rot = []
    
    for pose_c2w in poses_c2w:
        pose_c2w = np.array(pose_c2w)
        t = pose_c2w[:3, 3]
        R = pose_c2w[:3, :3]
        
        # Re-orthogonalize current rotation if requested
        if reorth_rot:
            R = _reorthogonalize_rotation_numpy(R)
        
        # Compute translation distance
        trans_dist = _compute_translation_distance_numpy(ref_t, t, trans_metric_mode)
        farness_trans.append(trans_dist)
        
        # Compute rotation distance
        rot_dist = _compute_rotation_distance_numpy(ref_R, R, rot_metric_mode)
        farness_rot.append(rot_dist)
    
    # Apply translation scaling if requested
    if translation_scale is not None:
        max_trans = max(farness_trans) if farness_trans else 1.0
        if max_trans > 0:
            scale_factor = translation_scale / max_trans
            farness_trans = [t * scale_factor for t in farness_trans]
    
    return farness_trans, farness_rot


def compute_pose_farness_torch(
    poses_c2w: torch.Tensor,
    trans_metric_mode: str = 'euclidean',
    rot_metric_mode: str = 'angle_axis',
    reorth_rot: bool = True,
    translation_scale: Union[None, float] = 1.0,
    reference_frame_id: int = 0
) -> Tuple[List[float], List[float]]:
    """
    PyTorch implementation of pose farness computation.
    
    Args:
        poses_c2w: Tensor of shape (N, 4, 4) containing camera-to-world transformation matrices
        trans_metric_mode: Translation distance metric
        rot_metric_mode: Rotation distance metric
        reorth_rot: Before computing rotation metric, re-orthogonalize rotations using SVD (default: True)
        translation_scale: Scale factor for translation distances (default: 1.0)
        reference_frame_id: Index of the reference frame (default: 0)
    
    Returns:
        farness_trans: List of translation distances
        farness_rot: List of rotation distances in radians
    """
    if poses_c2w.shape[0] == 0:
        return [], []
    
    # Validate reference_frame_id
    if not (0 <= reference_frame_id < poses_c2w.shape[0]):
        raise ValueError(
            f"reference_frame_id={reference_frame_id} out of range [0, {poses_c2w.shape[0]})"
        )
    
    # Reference pose is the specified one
    ref_c2w = poses_c2w[reference_frame_id]
    ref_t = ref_c2w[:3, 3]
    ref_R = ref_c2w[:3, :3]
    
    # Re-orthogonalize reference rotation if requested
    if reorth_rot:
        ref_R = _reorthogonalize_rotation_torch(ref_R)
    
    farness_trans = []
    farness_rot = []
    
    for i in range(poses_c2w.shape[0]):
        pose_c2w = poses_c2w[i]
        t = pose_c2w[:3, 3]
        R = pose_c2w[:3, :3]
        
        # Re-orthogonalize current rotation if requested
        if reorth_rot:
            R = _reorthogonalize_rotation_torch(R)
        
        # Compute translation distance
        trans_dist = _compute_translation_distance_torch(ref_t, t, trans_metric_mode)
        farness_trans.append(trans_dist)
        
        # Compute rotation distance
        rot_dist = _compute_rotation_distance_torch(ref_R, R, rot_metric_mode)
        farness_rot.append(rot_dist)
    
    # Apply translation scaling if requested
    if translation_scale is not None:
        max_trans = max(farness_trans) if farness_trans else 1.0
        if max_trans > 0:
            scale_factor = translation_scale / max_trans
            farness_trans = [t * scale_factor for t in farness_trans]
    
    return farness_trans, farness_rot

def _compute_translation_distance_numpy(t1: np.ndarray, t2: np.ndarray, mode: str) -> float:
    """
    Compute translation distance between two translation vectors (NumPy).
    
    Args:
        t1: Translation vector 1 (3,)
        t2: Translation vector 2 (3,)
        mode: Distance metric mode
    
    Returns:
        Translation distance
    """
    if mode == 'euclidean':
        return float(np.linalg.norm(t2 - t1))
    else:
        raise NotImplementedError(f"Translation metric mode '{mode}' not implemented")


def _reorthogonalize_rotation_numpy(R: np.ndarray) -> np.ndarray:
    """
    Re-orthogonalize a rotation matrix using SVD (NumPy).
    
    Args:
        R: Rotation matrix (3x3) that may have numerical errors
    
    Returns:
        Re-orthogonalized rotation matrix (3x3)
    """
    U, _, Vt = np.linalg.svd(R)
    R_orth = U @ Vt
    # Ensure proper rotation (det = 1, not reflection with det = -1)
    if np.linalg.det(R_orth) < 0:
        U[:, -1] *= -1
        R_orth = U @ Vt
    return R_orth


def _compute_rotation_distance_numpy(R1: np.ndarray, R2: np.ndarray, mode: str) -> float:
    """
    Compute rotation distance between two rotation matrices (NumPy).
    
    Args:
        R1: Rotation matrix 1 (3x3)
        R2: Rotation matrix 2 (3x3)
        mode: Distance metric mode
    
    Returns:
        Rotation distance in radians
    """
    if mode == 'angle_axis':
        # Convert relative rotation to angle-axis and extract angle
        R_rel = R1.T @ R2
        rot = R.from_matrix(R_rel)
        rotvec = rot.as_rotvec()
        angle = np.linalg.norm(rotvec)
        return float(angle)
    else:
        raise NotImplementedError(f"Rotation metric mode '{mode}' not implemented")


def _compute_translation_distance_torch(t1: torch.Tensor, t2: torch.Tensor, mode: str) -> float:
    """
    Compute translation distance between two translation vectors (PyTorch).
    
    Args:
        t1: Translation vector 1 (3,)
        t2: Translation vector 2 (3,)
        mode: Distance metric mode
    
    Returns:
        Translation distance
    """
    if mode == 'euclidean':
        return float(torch.norm(t2 - t1).item())
    else:
        raise NotImplementedError(f"Translation metric mode '{mode}' not implemented")


def _reorthogonalize_rotation_torch(R: torch.Tensor) -> torch.Tensor:
    """
    Re-orthogonalize a rotation matrix using SVD (PyTorch).
    
    Args:
        R: Rotation matrix (3x3) that may have numerical errors
    
    Returns:
        Re-orthogonalized rotation matrix (3x3)
    
    Note:
        SVD and det() are not implemented for BFloat16, so we temporarily convert to float32
        and explicitly disable autocast to prevent automatic conversion back to BF16.
        This ensures compatibility with mixed precision training while maintaining
        numerical accuracy for the orthogonalization process.
    """
    # JJ: Save original dtype
    original_dtype = R.dtype
    
    # JJ: Explicitly disable autocast context to prevent BF16 conversion
    with torch.cuda.amp.autocast(enabled=False):
        R_fp32 = R.float()  # Convert to float32 for numerical operations
        
        U, _, Vt = torch.linalg.svd(R_fp32)
        R_orth = U @ Vt
        
        # Ensure proper rotation (det = 1, not reflection with det = -1)
        # Must check det in FP32 before converting back
        if torch.det(R_orth) < 0:
            U[:, -1] *= -1
            R_orth = U @ Vt
    
    # JJ: Convert back to original dtype only after all operations
    return R_orth.to(original_dtype)


def _compute_rotation_distance_torch(R1: torch.Tensor, R2: torch.Tensor, mode: str) -> float:
    """
    Compute rotation distance between two rotation matrices (PyTorch).
    
    Args:
        R1: Rotation matrix 1 (3x3)
        R2: Rotation matrix 2 (3x3)
        mode: Distance metric mode
    
    Returns:
        Rotation distance in radians
    """
    if mode == 'angle_axis':
        # Convert relative rotation to angle-axis and extract angle
        R_rel = R1.T @ R2
        # Use matrix_to_axis_angle implementation
        angle = _rotation_matrix_to_angle_torch(R_rel)
        return float(angle.item())
    else:
        raise NotImplementedError(f"Rotation metric mode '{mode}' not implemented")


def _rotation_matrix_to_angle_torch(R: torch.Tensor) -> torch.Tensor:
    """
    Extract rotation angle from rotation matrix (PyTorch).
    
    Args:
        R: Rotation matrix (3x3)
    
    Returns:
        Rotation angle in radians
    """
    # Compute angle from rotation matrix using trace
    trace = torch.trace(R)
    cos_angle = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    return angle


def compute_lie_scalar_index_torch(
    poses_c2w: torch.Tensor,
    pose_id_scalar_lambda_trans: float = 1.0,  # JJ: Balance weight between rotation and translation
    traj_scale_norm: bool = True,
    global_normalize: bool = True,
    reorth_rot: bool = True,
    reference_frame_id: int = 0
) -> torch.Tensor:
    """
    Compute Lie-style scalar index for each pose relative to a reference frame.

    P_t = sqrt( theta_t^2 + (pose_id_scalar_lambda_trans^2) * d_t^2 )

    Notes:
        - traj_scale_norm: normalize translation per trajectory to [0,1] to
          make pose_id_scalar_lambda_trans weighting consistent across trajectories.
        - global_normalize: optional, normalize final P to [0,1] for phase modulation.
        - translation_scale inside compute_pose_farness_torch is set to None
          to disable internal auto-scaling; we handle normalization explicitly.

    Args:
        poses_c2w: (N,4,4) camera-to-world poses
        pose_id_scalar_lambda_trans: balance weight between rotation and translation
        traj_scale_norm: whether to normalize translation per trajectory (internal)
        global_normalize: optional normalize final P to [0,1] after lambda weighting
        reorth_rot: whether to re-orthogonalize rotations
        reference_frame_id: index of the reference frame (default: 0)

    Returns:
        P: (N,) tensor, scalar index per frame, normalized to [0, 1] if global_normalize=True
    """

    # ------------------ REUSE ------------------
    # Use existing farness computation, disable internal translation scaling
    farness_trans, farness_rot = compute_pose_farness_torch(
        poses_c2w,
        trans_metric_mode='euclidean',
        rot_metric_mode='angle_axis',
        reorth_rot=reorth_rot,
        translation_scale=None,  # disable auto-scaling inside function n doing externllay
        reference_frame_id=reference_frame_id  # 🆕 NEW: pass reference frame id
    )
    # ✏️ FIXED: Ensure tensors are created on the same device as input poses_c2w
    device = poses_c2w.device
    trans = torch.tensor(farness_trans, dtype=torch.float32, device=device)
    rot = torch.tensor(farness_rot, dtype=torch.float32, device=device)
    # ------------------ END REUSE ------------------

    # ------------------ NEW: per-trajectory translation normalization ------------------
    if traj_scale_norm:
        max_trans = trans.max()
        if max_trans > 0:
            trans = trans / max_trans  # unify translation scale across trajectory
    # ------------------ END NEW ------------------

    # ------------------ NEW: fuse rotation and translation into scalar P ------------------
    P = torch.sqrt(rot**2 + (pose_id_scalar_lambda_trans**2) * trans**2)
    # ------------------ END NEW ------------------

    # ------------------ NEW: optional global normalization for phase modulation ------------------
    if global_normalize:
        max_val = P.max()
        if max_val > 0:
            P = P / max_val  # Normalize to [0, 1]
    # ------------------ END NEW ------------------

    return P