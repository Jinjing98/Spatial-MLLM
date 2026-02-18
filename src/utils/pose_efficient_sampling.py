# JJ: Efficient Pose Sampling using 2D (translation, rotation) space
import numpy as np
import time
from typing import List, Tuple, Union, Optional
from collections import defaultdict

# Import from pose_distance_metrics for farness computation
try:
    from .pose_distance_metrics import compute_pose_farness
except ImportError:
    # When running as script directly
    from pose_distance_metrics import compute_pose_farness


def efficient_pose_sampling(
    poses: Union[List[np.ndarray], np.ndarray],
    num_samples: int,
    sampling_mode: str = 'hybrid',
    normalization: str = 'std_norm',
    diagonal_priority: float = 0.0,  # JJ: Should always be 0.0 for optimal coverage
    starting_mode: str = 'farthest',
    skip_first_frame: bool = False,
    reorth_rot: bool = True,
    grid_density: float = 1.5,
    verbose: bool = False
) -> List[int]:
    """
    Efficient frame sampling using 2D (translation, rotation) farness space.
    
    Main API entry point - compatible with pose_fps_sampling.py interface.
    
    Args:
        poses: List of 4x4 transformation matrices or array of shape (N, 4, 4)
        num_samples: Number of frames to select (m)
        sampling_mode: Sampling strategy
            - 'grid': Pure grid-based sampling, O(N), fastest
            - 'hybrid': Grid prefilter + 2D FPS, O(N + m²), balanced (default)
            - 'fps_2d': Pure 2D FPS, O(N·m), highest quality
        normalization: How to normalize translation and rotation to 2D space
            - 'std_norm': Normalize by standard deviation (adaptive, recommended)
            - 'max_norm': Normalize by maximum value
        diagonal_priority: Weight for diagonal (farness from frame0) in [0, 1]
            - **Recommended: 0.0** (default) - Pure FPS for optimal mutual coverage
            - 0.0: Select frames that are maximally distant from each other (best coverage)
            - >0.0: Add bias towards frames far from frame0 (NOT recommended for general use)
            - Note: This parameter exists for edge cases but should almost always be 0.0
        starting_mode: How to choose the first frame (for FPS-based modes)
            - 'farthest': Start from point farthest from origin (frame0)
            - 'medoid': Start from 2D medoid
            - 'first': Start from first frame
        skip_first_frame: If True, exclude frame 0 from selection
        reorth_rot: Re-orthogonalize rotation matrices before computing farness
        grid_density: For grid/hybrid modes, grid_cells = grid_density * num_samples
        verbose: Print timing and debug information
    
    Returns:
        selected_indices: List of selected frame indices (length = num_samples)
    """
    start_time = time.time()
    
    # Convert to numpy array if needed
    if isinstance(poses, list):
        poses = np.array(poses)
    
    N = poses.shape[0]
    
    if num_samples > N:
        raise ValueError(f"Cannot sample {num_samples} frames from {N} total frames")
    
    if num_samples == N:
        return list(range(N))
    
    # Step 1: Compute farness (distance to frame0) - O(N)
    t0 = time.time()
    farness_trans, farness_rot = compute_pose_farness(
        poses if isinstance(poses, list) else poses.tolist(),
        trans_metric_mode='euclidean',
        rot_metric_mode='angle_axis',
        reorth_rot=reorth_rot,
        translation_scale=None  # Don't scale yet, will normalize in 2D
    )
    t1 = time.time()
    if verbose:
        print(f"  Farness computation: {(t1-t0)*1000:.2f} ms")
    
    # Step 2: Normalize to 2D space
    t0 = time.time()
    points_2d, diagonal_scores = _normalize_to_2d(
        farness_trans, farness_rot, normalization
    )
    t1 = time.time()
    if verbose:
        print(f"  2D normalization: {(t1-t0)*1000:.2f} ms")
    
    # Handle skip_first_frame
    valid_indices = list(range(1 if skip_first_frame else 0, N))
    
    # Step 3: Select sampling strategy
    t0 = time.time()
    if sampling_mode == 'grid':
        selected = _grid_sampling(
            points_2d, diagonal_scores, num_samples, 
            valid_indices, grid_density, diagonal_priority
        )
    elif sampling_mode == 'hybrid':
        selected = _hybrid_sampling(
            points_2d, diagonal_scores, num_samples,
            valid_indices, grid_density, diagonal_priority, starting_mode
        )
    elif sampling_mode == 'fps_2d':
        selected = _fps_2d_sampling(
            points_2d, diagonal_scores, num_samples,
            valid_indices, diagonal_priority, starting_mode
        )
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
    
    t1 = time.time()
    
    total_time = time.time() - start_time
    if verbose:
        print(f"  Sampling ({sampling_mode}): {(t1-t0)*1000:.2f} ms")
        print(f"  Total time: {total_time*1000:.2f} ms")
    
    return selected


def _normalize_to_2d(
    farness_trans: List[float],
    farness_rot: List[float],
    mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize translation and rotation farness to 2D space.
    
    Returns:
        points_2d: (N, 2) array of normalized (trans, rot) coordinates
        diagonal_scores: (N,) array of distances to origin
    """
    trans = np.array(farness_trans)
    rot = np.array(farness_rot)
    
    if mode == 'std_norm':
        # Normalize by standard deviation
        trans_std = np.std(trans)
        rot_std = np.std(rot)
        
        trans_norm = trans / trans_std if trans_std > 1e-6 else trans
        rot_norm = rot / rot_std if rot_std > 1e-6 else rot
        
    elif mode == 'max_norm':
        # Normalize by maximum value
        trans_max = np.max(trans)
        rot_max = np.max(rot)
        
        trans_norm = trans / trans_max if trans_max > 1e-6 else trans
        rot_norm = rot / rot_max if rot_max > 1e-6 else rot
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    
    # Stack to (N, 2) array
    points_2d = np.column_stack([trans_norm, rot_norm])
    
    # Compute diagonal scores (distance to origin)
    diagonal_scores = np.linalg.norm(points_2d, axis=1)
    
    return points_2d, diagonal_scores


def _grid_sampling(
    points_2d: np.ndarray,
    diagonal_scores: np.ndarray,
    num_samples: int,
    valid_indices: List[int],
    grid_density: float,
    diagonal_priority: float
) -> List[int]:
    """
    Pure grid-based sampling - O(N).
    
    Divides 2D space into grid cells and selects one point per cell.
    """
    N = len(points_2d)
    
    # Determine grid size
    num_cells = int(grid_density * num_samples)
    grid_size = int(np.ceil(np.sqrt(num_cells)))
    
    # Compute grid bounds
    mins = points_2d[valid_indices].min(axis=0)
    maxs = points_2d[valid_indices].max(axis=0)
    ranges = maxs - mins
    
    # Avoid division by zero
    ranges = np.where(ranges < 1e-6, 1.0, ranges)
    
    # Assign points to grid cells
    grid = defaultdict(list)
    for idx in valid_indices:
        point = points_2d[idx]
        # Compute grid coordinates
        grid_x = int((point[0] - mins[0]) / ranges[0] * (grid_size - 1e-6))
        grid_y = int((point[1] - mins[1]) / ranges[1] * (grid_size - 1e-6))
        grid_x = np.clip(grid_x, 0, grid_size - 1)
        grid_y = np.clip(grid_y, 0, grid_size - 1)
        grid[(grid_x, grid_y)].append(idx)
    
    # Sort cells by diagonal score (prioritize outer cells)
    cell_scores = {}
    for cell, indices in grid.items():
        # Cell score = average diagonal score of points in cell
        cell_scores[cell] = np.mean([diagonal_scores[i] for i in indices])
    
    sorted_cells = sorted(grid.keys(), key=lambda c: cell_scores[c], reverse=True)
    
    # Select one point from each cell
    selected = []
    for cell in sorted_cells:
        if len(selected) >= num_samples:
            break
        
        indices = grid[cell]
        if len(indices) == 0:
            continue
        
        # Select best point in cell based on diagonal_priority
        if diagonal_priority > 0.5:
            # Prioritize point farthest from origin
            best_idx = max(indices, key=lambda i: diagonal_scores[i])
        else:
            # Select point closest to cell center
            cell_center = np.array(cell) + 0.5
            distances = [np.linalg.norm(
                (points_2d[i] - mins) / ranges * grid_size - cell_center
            ) for i in indices]
            best_idx = indices[np.argmin(distances)]
        
        selected.append(best_idx)
    
    # If not enough cells, fill with remaining points by diagonal score
    if len(selected) < num_samples:
        remaining = [i for i in valid_indices if i not in selected]
        remaining_sorted = sorted(remaining, key=lambda i: diagonal_scores[i], reverse=True)
        selected.extend(remaining_sorted[:num_samples - len(selected)])
    
    return selected[:num_samples]


def _hybrid_sampling(
    points_2d: np.ndarray,
    diagonal_scores: np.ndarray,
    num_samples: int,
    valid_indices: List[int],
    grid_density: float,
    diagonal_priority: float,
    starting_mode: str
) -> List[int]:
    """
    Hybrid sampling: Grid prefilter + 2D FPS - O(N + m²).
    
    1. Use grid to select 3*num_samples candidates (ensures coverage)
    2. Run FPS on candidates to select final num_samples
    """
    # Step 1: Grid prefilter to get candidates
    num_candidates = min(int(3 * num_samples), len(valid_indices))
    candidates = _grid_sampling(
        points_2d, diagonal_scores, num_candidates,
        valid_indices, grid_density, diagonal_priority
    )
    
    # Step 2: FPS on candidates
    if len(candidates) <= num_samples:
        return candidates
    
    selected = _fps_2d_sampling(
        points_2d, diagonal_scores, num_samples,
        candidates, diagonal_priority, starting_mode
    )
    
    return selected


def _fps_2d_sampling(
    points_2d: np.ndarray,
    diagonal_scores: np.ndarray,
    num_samples: int,
    valid_indices: List[int],
    diagonal_priority: float,
    starting_mode: str
) -> List[int]:
    """
    Pure 2D FPS sampling - O(N·m).
    
    Farthest Point Sampling in 2D (trans, rot) space.
    """
    if len(valid_indices) <= num_samples:
        return valid_indices
    
    # Select starting point
    if starting_mode == 'farthest':
        # Start from point farthest from origin
        start_idx = max(valid_indices, key=lambda i: diagonal_scores[i])
    elif starting_mode == 'medoid':
        # Start from 2D medoid (point with minimum sum of distances)
        dist_sums = np.zeros(len(valid_indices))
        for i, idx_i in enumerate(valid_indices):
            for idx_j in valid_indices:
                dist_sums[i] += np.linalg.norm(points_2d[idx_i] - points_2d[idx_j])
        start_idx = valid_indices[np.argmin(dist_sums)]
    elif starting_mode == 'first':
        start_idx = valid_indices[0]
    else:
        raise ValueError(f"Unknown starting_mode: {starting_mode}")
    
    # Initialize
    selected = [start_idx]
    remaining = set(valid_indices) - {start_idx}
    
    # Track minimum distance from each point to selected set
    min_distances = np.full(len(points_2d), np.inf)
    for idx in remaining:
        min_distances[idx] = np.linalg.norm(points_2d[idx] - points_2d[start_idx])
    
    # Greedy FPS with optional diagonal weighting
    for _ in range(num_samples - 1):
        if len(remaining) == 0:
            break
        
        # Compute scores: weighted combination of distance and diagonal
        scores = np.full(len(points_2d), -np.inf)
        for idx in remaining:
            # Base score: distance to nearest selected point (sparsity)
            sparsity_score = min_distances[idx]
            
            # Bonus score: distance to origin (diagonal priority)
            diagonal_bonus = diagonal_scores[idx]
            
            # Combine with weights
            scores[idx] = (1 - diagonal_priority) * sparsity_score + \
                          diagonal_priority * diagonal_bonus
        
        # Select point with highest score
        farthest_idx = int(np.argmax(scores))
        
        if farthest_idx not in remaining:
            break
        
        # Add to selected
        selected.append(farthest_idx)
        remaining.remove(farthest_idx)
        
        # Update min_distances
        for idx in remaining:
            dist = np.linalg.norm(points_2d[idx] - points_2d[farthest_idx])
            min_distances[idx] = min(min_distances[idx], dist)
    
    return selected


def analyze_2d_coverage(
    poses: Union[List[np.ndarray], np.ndarray],
    selected_indices: List[int],
    normalization: str = 'std_norm',
    reorth_rot: bool = True
) -> dict:
    """
    Analyze coverage quality in 2D (translation, rotation) space.
    
    Similar to analyze_pose_coverage in pose_fps_sampling.py but operates
    in 2D farness space instead of full SE(3) space.
    
    Args:
        poses: All poses (N, 4, 4)
        selected_indices: Indices of selected frames
        normalization: Normalization mode for 2D space
        reorth_rot: Re-orthogonalize rotations
    
    Returns:
        Dictionary containing coverage statistics
    """
    if isinstance(poses, list):
        poses = np.array(poses)
    
    N = poses.shape[0]
    selected_set = set(selected_indices)
    
    # Compute 2D points
    farness_trans, farness_rot = compute_pose_farness(
        poses if isinstance(poses, list) else poses.tolist(),
        reorth_rot=reorth_rot,
        translation_scale=None
    )
    points_2d, diagonal_scores = _normalize_to_2d(
        farness_trans, farness_rot, normalization
    )
    
    # For each frame, find distance to nearest selected frame in 2D space
    min_distances = np.full(N, np.inf)
    for i in range(N):
        for j in selected_indices:
            dist_2d = np.linalg.norm(points_2d[i] - points_2d[j])
            min_distances[i] = min(min_distances[i], dist_2d)
    
    # Compute statistics
    non_selected_distances = [min_distances[i] for i in range(N) if i not in selected_set]
    
    return {
        'min_distances': min_distances,
        'mean_min_distance': np.mean(non_selected_distances) if non_selected_distances else 0.0,
        'max_min_distance': np.max(non_selected_distances) if non_selected_distances else 0.0,
        'coverage_uniformity': np.std(min_distances),
        'num_selected': len(selected_indices),
        'num_total': N,
        'points_2d': points_2d,
        'diagonal_scores': diagonal_scores
    }


def main():
    """
    Test efficient sampling with real data and compare strategies.
    """
    import json
    from pathlib import Path
    import torch
    
    # JJ: Configuration for visualization
    PLOT_POSE_ANALYSIS = True  # Set to True to also generate pose_analysis.html with all frames
    GENERATE_VISUALIZATIONS = True  # Set to True to generate HTML visualizations
    
    # JJ: Configuration for pose source
    USE_SYNTHETIC_POSES = True  # True: use synthetic trajectories, False: use real poses
    USE_VGGT_POSES = True  # If USE_SYNTHETIC_POSES=False, True: use VGGT predicted poses, False: use GT poses
    SYNTHETIC_TRAJ_TYPE = 'traj1'  # Options: 'traj1', 'traj2', 'traj3'
    SYNTHETIC_N_FRAMES = 100  # Number of frames for synthetic trajectory
    
    # Paths
    predictions_path = "/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/vsibench/sa_sampling_16f_single_video/arkitscenes/00777c41d4/sa_predictions.pt"
    json_path = "/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/scannetpp/pa_sampling_good/00777c41d4/selected_frames.json"
    # json_path = "/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/scannetpp/pa_sampling_good/020312de8d/selected_frames.json"
    
    # Load poses based on configuration
    if USE_SYNTHETIC_POSES:
        # JJ: Load synthetic trajectory
        try:
            from .traj_generator import (
                generate_traj1_loop_closure,
                generate_traj2_linear_translation,
                generate_traj3_rotation_then_translation
            )
        except ImportError:
            from traj_generator import (
                generate_traj1_loop_closure,
                generate_traj2_linear_translation,
                generate_traj3_rotation_then_translation
            )
        
        print(f"Generating synthetic poses: {SYNTHETIC_TRAJ_TYPE} with {SYNTHETIC_N_FRAMES} frames")
        
        traj_generators = {
            'traj1': generate_traj1_loop_closure,
            'traj2': generate_traj2_linear_translation,
            'traj3': generate_traj3_rotation_then_translation
        }
        
        pose_list = traj_generators[SYNTHETIC_TRAJ_TYPE](n_frames=SYNTHETIC_N_FRAMES)
        poses = np.array(pose_list)
        pose_source = f"Synthetic ({SYNTHETIC_TRAJ_TYPE})"
        
        print(f"\nGenerated {len(poses)} synthetic poses")
        print(f"Trajectory type: {SYNTHETIC_TRAJ_TYPE}")
    elif USE_VGGT_POSES:
        # JJ: Load VGGT predicted poses
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
        # JJ: Load real GT poses from JSON
        print("Loading GT poses from:", json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        pose_matrices = [frame['pose_matrix'] for frame in data['transform_matrix']]
        poses = np.array(pose_matrices)
        pose_source = "Ground Truth"
        
        print(f"\nLoaded {len(poses)} GT poses")
        print(f"Scene: {data['scene_name']}")
    
    num_samples = 50
    print(f"\n{'='*70}")
    print(f"Testing Efficient Sampling - selecting {num_samples} frames")
    print(f"{'='*70}")
    print(f"Pose source: {pose_source}")
    print(f"Generate visualizations: {'Yes' if GENERATE_VISUALIZATIONS else 'No'}")
    print(f"Generate pose_analysis.html: {'Yes' if PLOT_POSE_ANALYSIS else 'No'}")
    
    # Import visualization
    has_vis = False
    if GENERATE_VISUALIZATIONS:
        try:
            # Try relative import first (when run as module)
            try:
                from .visualisation import visualize_sampled_pose
            except ImportError:
                # Fall back to direct import (when run as script in src/utils/)
                from visualisation import visualize_sampled_pose
            has_vis = True
        except ImportError:
            print("Warning: visualisation module not found, skipping visualization")
    
    # Pre-compute SE(3) distance matrix for visualization (if needed)
    distance_matrix_se3 = None
    if has_vis:
        try:
            from .pose_fps_sampling import compute_pairwise_pose_distance
        except ImportError:
            from pose_fps_sampling import compute_pairwise_pose_distance
        
        try:
            print("\nPre-computing SE(3) distance matrix for visualization...")
            distance_matrix_se3 = compute_pairwise_pose_distance(poses, distance_mode='max_norm')
        except Exception as e:
            print(f"Warning: Could not compute SE(3) distance matrix: {e}")
    
    # Test all strategies
    strategies = [
        ('grid', 'Grid-based (O(N))'),
        # ('hybrid', 'Hybrid: Grid + FPS (O(N + m²))'),
        # ('fps_2d', '2D FPS (O(N·m))'),
    ]
    
    # Configurations to visualize
    configs_to_visualize = ['grid', 'hybrid', 'fps_2d']
    
    results = {}
    
    for mode, description in strategies:
        print(f"\n[{description}]")
        print(f"  Mode: {mode}")
        
        # Run multiple times for timing
        times = []
        for _ in range(5):
            start = time.time()
            selected = efficient_pose_sampling(
                poses,
                num_samples=num_samples,
                sampling_mode=mode,
                normalization='std_norm',
                diagonal_priority=0.0,  # Pure FPS for optimal coverage
                starting_mode='farthest',
                skip_first_frame=False,
                verbose=False
            )
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        
        # Analyze coverage in 2D space
        coverage_2d = analyze_2d_coverage(poses, selected)
        
        print(f"  Time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Selected indices (sorted): {sorted(selected)[:10]}{'...' if len(selected) > 10 else ''}")
        print(f"  Mean min distance (2D): {coverage_2d['mean_min_distance']:.4f}")
        print(f"  Max min distance (2D): {coverage_2d['max_min_distance']:.4f}")
        print(f"  Coverage uniformity (std): {coverage_2d['coverage_uniformity']:.4f}")
        
        results[mode] = {
            'time': avg_time,
            'selected': selected,
            'coverage_2d': coverage_2d
        }
        
        # Visualize selected configurations
        if has_vis and mode in configs_to_visualize and distance_matrix_se3 is not None:
            print(f"  → Generating visualization...")
            output_name = f"tmp/efficient_sampling_{mode}.html"
            
            # Compute SE(3) coverage for visualization
            try:
                try:
                    from .pose_fps_sampling import analyze_pose_coverage
                except ImportError:
                    from pose_fps_sampling import analyze_pose_coverage
                coverage_se3 = analyze_pose_coverage(poses, selected, 'max_norm', True)
                
                visualize_sampled_pose(
                    pose_analysis_poses=poses,
                    selected_indices=selected,
                    min_distances=coverage_se3['min_distances'],
                    distance_matrix=distance_matrix_se3,
                    output_path=output_name,
                    show=False,
                    method_name=f'Efficient Sampling ({mode})',
                    plot_pose_analysis=PLOT_POSE_ANALYSIS,
                    pose_source=pose_source  # JJ: Pass pose source for annotation
                )
            except Exception as e:
                print(f"  Warning: Visualization failed: {e}")
    
    # Compare with original FPS
    print(f"\n{'='*70}")
    print("Comparison with Original FPS (SE(3) space)")
    print(f"{'='*70}")
    
    try:
        try:
            from .pose_fps_sampling import farthest_point_sampling, analyze_pose_coverage
        except ImportError:
            from pose_fps_sampling import farthest_point_sampling, analyze_pose_coverage
        
        # Time original FPS
        times = []
        for _ in range(5):
            start = time.time()
            selected_fps = farthest_point_sampling(
                poses,
                num_samples=num_samples,
                distance_mode='max_norm',
                starting_mode='medoid',
                reorth_rot=True
            )
            times.append(time.time() - start)
        
        fps_time = np.mean(times) * 1000
        fps_std = np.std(times) * 1000
        
        coverage_fps = analyze_pose_coverage(poses, selected_fps, 'max_norm', True)
        
        print(f"\n[Original FPS (SE(3) space)]")
        print(f"  Time: {fps_time:.2f} ± {fps_std:.2f} ms")
        print(f"  Selected indices (sorted): {sorted(selected_fps)[:10]}{'...' if len(selected_fps) > 10 else ''}")
        print(f"  Mean min distance (SE3): {coverage_fps['mean_min_distance']:.4f}")
        print(f"  Max min distance (SE3): {coverage_fps['max_min_distance']:.4f}")
        print(f"  Coverage uniformity (std): {coverage_fps['coverage_uniformity']:.4f}")
        
        # Visualize FPS for comparison
        if has_vis and distance_matrix_se3 is not None:
            print(f"  → Generating visualization...")
            output_name = "tmp/efficient_sampling_fps_baseline.html"
            visualize_sampled_pose(
                pose_analysis_poses=poses,
                selected_indices=selected_fps,
                min_distances=coverage_fps['min_distances'],
                distance_matrix=distance_matrix_se3,
                output_path=output_name,
                show=False,
                method_name='FPS Baseline (SE3)',
                plot_pose_analysis=PLOT_POSE_ANALYSIS,
                pose_source=pose_source  # JJ: Pass pose source for annotation
            )
        
        # Speedup comparison
        print(f"\n{'='*70}")
        print("Speedup Summary")
        print(f"{'='*70}")
        for mode, desc in strategies:
            speedup = fps_time / results[mode]['time']
            print(f"  {mode:12s}: {speedup:5.1f}x faster than original FPS")
        
    except ImportError:
        print("  (Original FPS not available for comparison)")
    
    print(f"\n{'='*70}")
    print("✓ Efficient sampling testing complete!")
    if has_vis:
        print("✓ Visualizations saved to current directory")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
