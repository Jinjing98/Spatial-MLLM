# JJ: Pose-aware Farthest Point Sampling (FPS) for frame selection
import numpy as np
from typing import List, Tuple, Union, Optional
from scipy.spatial.transform import Rotation as R


def compute_pairwise_pose_distance(
    poses: Union[List[np.ndarray], np.ndarray],
    distance_mode: str = 'max_norm',
    reorth_rot: bool = True
) -> np.ndarray:
    """
    Compute pairwise SE(3) distance matrix between all poses.
    
    This is the key function for pose-space coverage: it measures how different
    two camera poses are by combining translation distance and rotation distance.
    
    Args:
        poses: List of 4x4 transformation matrices or array of shape (N, 4, 4)
        distance_mode: How to combine translation and rotation distances
            - 'max_norm': Normalize by max, then d = alpha * trans + beta * rot
                         with alpha=1.0, beta=1.0
            - 'data_driven': Compute weights based on variance to balance contributions
        reorth_rot: Re-orthogonalize rotation matrices using SVD before computing distances
    
    Returns:
        distance_matrix: (N, N) symmetric matrix where [i,j] = SE(3) distance between pose i and j
    """
    # Convert to numpy array if needed
    if isinstance(poses, list):
        poses = np.array(poses)
    
    N = poses.shape[0]
    
    # Extract translations and rotations
    translations = poses[:, :3, 3]  # (N, 3)
    rotations = poses[:, :3, :3]    # (N, 3, 3)
    
    # Re-orthogonalize rotations if requested
    if reorth_rot:
        rotations = np.array([_reorthogonalize_rotation(R) for R in rotations])
    
    # Compute pairwise translation distances: ||t_i - t_j||
    trans_dist = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(translations[i] - translations[j])
            trans_dist[i, j] = dist
            trans_dist[j, i] = dist
    
    # Compute pairwise rotation distances: geodesic angle between R_i and R_j
    rot_dist = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist = _compute_rotation_angle(rotations[i], rotations[j])
            rot_dist[i, j] = dist
            rot_dist[j, i] = dist
    
    # Combine translation and rotation distances based on mode
    if distance_mode == 'max_norm':
        # Normalize by maximum value, then combine with equal weights
        trans_max = trans_dist.max()
        rot_max = rot_dist.max()
        
        # Avoid division by zero
        trans_norm = trans_dist / trans_max if trans_max > 0 else trans_dist
        rot_norm = rot_dist / rot_max if rot_max > 0 else rot_dist
        
        alpha, beta = 1.0, 1.0
        distance_matrix = alpha * trans_norm  + beta * rot_norm
        
    elif distance_mode == 'data_driven':
        # Compute weights based on variance to balance contributions
        # Goal: make both terms contribute equally to total variance
        trans_var = np.var(trans_dist[np.triu_indices(N, k=1)])  # Only upper triangle
        rot_var = np.var(rot_dist[np.triu_indices(N, k=1)])
        
        # If one has zero variance, use equal weights
        if trans_var > 0 and rot_var > 0:
            # Weight inversely proportional to std to balance contributions
            trans_std = np.sqrt(trans_var)
            rot_std = np.sqrt(rot_var)
            alpha = 1.0 / trans_std
            beta = 1.0 / rot_std
            # Normalize so alpha + beta = 2 (for scale consistency)
            total = alpha + beta
            alpha = 2.0 * alpha / total
            beta = 2.0 * beta / total
        else:
            alpha, beta = 1.0, 1.0
        
        distance_matrix = alpha * trans_dist + beta * rot_dist
        
    else:
        raise NotImplementedError(f"Distance mode '{distance_mode}' not implemented")
    
    return distance_matrix


def farthest_point_sampling(
    poses: Union[List[np.ndarray], np.ndarray],
    num_samples: int,
    distance_mode: str = 'max_norm',
    starting_mode: str = 'medoid',
    reorth_rot: bool = True,
    distance_matrix: Optional[np.ndarray] = None
) -> List[int]:
    """
    Select a subset of frames using Farthest Point Sampling (FPS) in SE(3) space.
    
    FPS greedily selects frames that are maximally distant from already selected frames,
    ensuring good coverage of the pose space (both translation and rotation).
    
    Algorithm:
        1. Select starting frame based on starting_mode
        2. Iteratively select the frame that maximizes the minimum distance to all selected frames
        3. Repeat until num_samples frames are selected
    
    Args:
        poses: List of 4x4 transformation matrices or array of shape (N, 4, 4)
        num_samples: Number of frames to select (m)
        distance_mode: 'max_norm' or 'data_driven' (see compute_pairwise_pose_distance)
        starting_mode: How to choose the first frame
            - 'first': Use the first frame (index 0)
            - 'rand': Random selection
            - 'medoid': Frame with minimum sum of distances to all others (geometric center)
        reorth_rot: Re-orthogonalize rotation matrices
        distance_matrix: Pre-computed distance matrix (if None, will compute)
    
    Returns:
        selected_indices: List of selected frame indices (length = num_samples)
    """
    # Convert to numpy array if needed
    if isinstance(poses, list):
        poses = np.array(poses)
    
    N = poses.shape[0]
    
    if num_samples > N:
        raise ValueError(f"Cannot sample {num_samples} frames from {N} total frames")
    
    # Compute distance matrix if not provided
    if distance_matrix is None:
        distance_matrix = compute_pairwise_pose_distance(poses, distance_mode, reorth_rot)
    
    # Select starting frame
    if starting_mode == 'first':
        start_idx = 0
    elif starting_mode == 'rand':
        start_idx = np.random.randint(N)
    elif starting_mode == 'medoid':
        # Medoid: frame with minimum sum of distances to all others
        dist_sums = distance_matrix.sum(axis=1)
        start_idx = int(np.argmin(dist_sums))
    else:
        raise ValueError(f"Unknown starting_mode: {starting_mode}")
    
    # Initialize
    selected = [start_idx]
    remaining = set(range(N)) - {start_idx}
    
    # Track minimum distance from each remaining frame to selected set
    min_distances = distance_matrix[start_idx].copy()
    
    # Greedily select farthest points
    for _ in range(num_samples - 1):
        # Find the frame with maximum minimum distance to selected set
        # Only consider remaining frames
        max_dist = -1
        farthest_idx = -1
        for idx in remaining:
            if min_distances[idx] > max_dist:
                max_dist = min_distances[idx]
                farthest_idx = idx
        
        if farthest_idx == -1:
            break
        
        # Add to selected set
        selected.append(farthest_idx)
        remaining.remove(farthest_idx)
        
        # Update minimum distances: for each remaining frame,
        # its new min distance is min(old_min, dist_to_new_frame)
        for idx in remaining:
            min_distances[idx] = min(min_distances[idx], distance_matrix[farthest_idx, idx])
    
    return selected


def _reorthogonalize_rotation(R: np.ndarray) -> np.ndarray:
    """
    Re-orthogonalize a rotation matrix using SVD.
    
    Due to numerical errors from file I/O or computation, rotation matrices
    may not be perfectly orthogonal. This function projects them back to SO(3).
    
    Args:
        R: Rotation matrix (3x3) that may have numerical errors
    
    Returns:
        Re-orthogonalized rotation matrix (3x3) with det = 1
    """
    U, _, Vt = np.linalg.svd(R)
    R_orth = U @ Vt
    # Ensure proper rotation (det = 1, not reflection with det = -1)
    if np.linalg.det(R_orth) < 0:
        U[:, -1] *= -1
        R_orth = U @ Vt
    return R_orth


def _compute_rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute geodesic angle between two rotation matrices.
    
    This is the natural distance metric on SO(3) manifold:
    angle = arccos((trace(R1^T @ R2) - 1) / 2)
    
    Equivalent to the angle of the rotation that takes R1 to R2.
    
    Args:
        R1: Rotation matrix 1 (3x3)
        R2: Rotation matrix 2 (3x3)
    
    Returns:
        Rotation angle in radians [0, pi]
    """
    R_rel = R1.T @ R2
    trace = np.trace(R_rel)
    # Clamp to [-1, 1] to handle numerical errors in arccos
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return float(angle)


def analyze_pose_coverage(
    poses: Union[List[np.ndarray], np.ndarray],
    selected_indices: List[int],
    distance_mode: str = 'max_norm',
    reorth_rot: bool = True
) -> dict:
    """
    Analyze how well the selected frames cover the pose space.
    
    Computes statistics about the coverage quality by measuring how far
    each non-selected frame is from the nearest selected frame.
    
    Args:
        poses: All poses (N, 4, 4)
        selected_indices: Indices of selected frames
        distance_mode: Distance computation mode
        reorth_rot: Re-orthogonalize rotations
    
    Returns:
        Dictionary containing:
            - 'min_distances': For each frame, distance to nearest selected frame
            - 'mean_min_distance': Average of minimum distances
            - 'max_min_distance': Maximum of minimum distances (worst coverage)
            - 'coverage_uniformity': Std of min distances (lower = more uniform)
    """
    if isinstance(poses, list):
        poses = np.array(poses)
    
    N = poses.shape[0]
    selected_set = set(selected_indices)
    
    # Compute distance matrix
    distance_matrix = compute_pairwise_pose_distance(poses, distance_mode, reorth_rot)
    
    # For each frame, find distance to nearest selected frame
    min_distances = np.full(N, np.inf)
    for i in range(N):
        for j in selected_indices:
            min_distances[i] = min(min_distances[i], distance_matrix[i, j])
    
    # Compute statistics
    non_selected_distances = [min_distances[i] for i in range(N) if i not in selected_set]
    
    return {
        'min_distances': min_distances,
        'mean_min_distance': np.mean(non_selected_distances) if non_selected_distances else 0.0,
        'max_min_distance': np.max(non_selected_distances) if non_selected_distances else 0.0,
        'coverage_uniformity': np.std(min_distances),
        'num_selected': len(selected_indices),
        'num_total': N
    }


def main():
    """
    Test FPS with real data and visualize results.
    """
    import json
    from pathlib import Path
    import torch
    
    # JJ: Configuration for visualization
    PLOT_POSE_ANALYSIS = True  # Set to True to also generate pose_analysis.html with all frames
    
    # JJ: Configuration for pose source
    USE_VGGT_POSES = False  # True: use VGGT predicted poses, False: use GT poses from JSON
    
    # Paths
    predictions_path = "/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/vsibench/sa_sampling_16f_single_video/arkitscenes/00777c41d4/sa_predictions.pt"
    json_path = "/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/scannetpp/pa_sampling_good/00777c41d4/selected_frames.json"
    # json_path = "/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/scannetpp/pa_sampling_good/020312de8d/selected_frames.json"
    
    # Load poses based on configuration
    if USE_VGGT_POSES:
        print("Loading VGGT predicted poses from:", predictions_path)
        
        # Import extract function from flexiable_sa_sampling
        try:
            from flexiable_sa_sampling import extract_poses_from_predictions
        except ImportError:
            # Define locally if import fails
            def extract_poses_from_predictions(predictions: dict) -> np.ndarray:
                """Extract camera poses from VGGT predictions."""
                extrinsics = predictions['extrinsic']  # Shape: (B, T, 4, 4)
                if extrinsics.dim() == 4:
                    extrinsics = extrinsics[0]  # Remove batch dimension
                poses = extrinsics.cpu().numpy()  # (T, 4, 4)
                return poses
        
        predictions = torch.load(predictions_path, weights_only=False)
        poses = extract_poses_from_predictions(predictions)
        pose_source = "VGGT Predicted"
        
        print(f"\nLoaded {len(poses)} VGGT predicted poses")
    else:
        print("Loading GT poses from:", json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract pose matrices
        pose_matrices = [frame['pose_matrix'] for frame in data['transform_matrix']]
        poses = np.array(pose_matrices)
        pose_source = "Ground Truth"
        
        print(f"\nLoaded {len(poses)} GT poses")
        print(f"Scene: {data['scene_name']}")
    
    # Test different configurations
    num_samples = 20
    print(f"\n{'='*60}")
    print(f"Testing FPS to select {num_samples} frames")
    print(f"{'='*60}")
    print(f"Pose source: {pose_source}")
    print(f"Visualization: 1x2 layout (2 figures)")
    print(f"Generate pose_analysis.html: {'Yes' if PLOT_POSE_ANALYSIS else 'No'}")
    
    # Pre-compute distance matrix for efficiency
    print("\nPre-computing distance matrix...")
    distance_matrix = compute_pairwise_pose_distance(poses, distance_mode='max_norm')
    
    # Import visualization
    try:
        from visualisation import visualize_sampled_pose
        has_vis = True
    except ImportError:
        print("Warning: visualisation module not found, skipping visualization")
        has_vis = False
    
    configs_to_visualize = [
        ('max_norm', 'first'),
        ('max_norm', 'medoid'),
        ('max_norm', 'rand'),
        ('data_driven', 'rand'),
        ('data_driven', 'first'),
        ('data_driven', 'medoid'),
    ]
    
    # for distance_mode in ['max_norm', 'data_driven']:
    for distance_mode in ['max_norm']:
        # for starting_mode in ['first', 'medoid', 'rand']:
        for starting_mode in ['first']:
            print(f"\n[{distance_mode} | {starting_mode}]")
            
            # Compute distance matrix for this mode
            if distance_mode == 'data_driven':
                dist_matrix = compute_pairwise_pose_distance(poses, distance_mode='data_driven')
            else:
                dist_matrix = distance_matrix  # Use pre-computed
            
            selected = farthest_point_sampling(
                poses,
                num_samples=num_samples,
                distance_mode=distance_mode,
                starting_mode=starting_mode,
                reorth_rot=True,
                distance_matrix=dist_matrix
            )
            
            print(f"  Selected indices (FPS order): {selected[:15]}..." if len(selected) > 10 else f"  Selected indices (FPS order): {selected}")
            selected_sorted = sorted(selected)
            print(f"  Selected indices (sorted): {selected_sorted[:15]}..." if len(selected_sorted) > 10 else f"  Selected indices (sorted): {selected_sorted}")
            
            # Analyze coverage
            coverage = analyze_pose_coverage(poses, selected, distance_mode)
            print(f"  Mean min distance: {coverage['mean_min_distance']:.4f}")
            print(f"  Max min distance: {coverage['max_min_distance']:.4f}")
            print(f"  Coverage uniformity (std): {coverage['coverage_uniformity']:.4f}")
            
            # Visualize selected configurations
            if has_vis and (distance_mode, starting_mode) in configs_to_visualize:
                print(f"  → Generating visualization...")
                output_name = f"tmp/fps_quality_{distance_mode}_{starting_mode}.html"
                visualize_sampled_pose(
                    pose_analysis_poses=poses,
                    selected_indices=selected,
                    min_distances=coverage['min_distances'],
                    distance_matrix=dist_matrix,
                    output_path=output_name,
                    show=False,
                    method_name=f'FPS ({distance_mode}, {starting_mode})',
                    plot_pose_analysis=PLOT_POSE_ANALYSIS,  # JJ: Control pose_analysis.html generation
                    pose_source=pose_source  # JJ: Pass pose source for annotation
                )
            
            # JJ: Test Lie scalar index computation for selected frames
            print(f"\n  Testing Lie Scalar Index for selected {num_samples} frames:")
            print(f"  {'-'*50}")
            
            # Import the function from pose_distance_metrics
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from pose_distance_metrics import compute_lie_scalar_index_torch
            
            # Extract selected poses and convert to torch tensor
            selected_poses = poses[selected]
            selected_poses_tensor = torch.from_numpy(selected_poses).float()
            
            # Test with different lambda_trans values
            for lambda_trans in [0.1, 0.5, 1.0, 2.0, 5.0]:
                P = compute_lie_scalar_index_torch(
                    poses_c2w=selected_poses_tensor,
                    lambda_trans=lambda_trans,
                    traj_scale_norm=True,
                    global_normalize=False,
                    reorth_rot=True
                )
                
                print(f"\n    lambda_trans = {lambda_trans}:")
                print(f"      P shape: {P.shape}")
                print(f"      P range: [{P.min().item():.4f}, {P.max().item():.4f}]")
                print(f"      P mean: {P.mean().item():.4f}, std: {P.std().item():.4f}")
                print(f"      Values: {[f'{v:.4f}' for v in P.tolist()]}")
    
    print("\n" + "="*60)
    print("✓ FPS testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
