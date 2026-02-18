# JJ: Flexible SA (Space-Aware) Sampling - unified API
import numpy as np
import torch
import time
from typing import List, Dict, Optional, Union
from pathlib import Path

# Import from sa_sampling.py for voxel computation and greedy sampling
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sampling.sa_sampling import compute_voxel_sets, maximum_coverage_sampling


def sa_sampling(
    predictions: Dict[str, torch.Tensor],
    num_samples: int,
    voxel_density: float = 20.0,
    conf_percentile: float = 50.0,
    conf_threshold: float = 0.1,
    verbose: bool = False
) -> List[int]:
    """
    Space-aware frame sampling using VGGT predictions.
    
    Main API entry point - compatible with pose_fps_sampling.py and pose_efficient_sampling.py interface.
    
    This function performs greedy maximum coverage sampling in 3D voxel space:
    1. Compute voxel sets for each frame based on world_points
    2. Greedily select frames that maximize new voxel coverage
    
    Args:
        predictions: VGGT predictions dict containing:
            - 'world_points': Tensor of shape (1, T, H, W, 3) - 3D coordinates
            - 'world_points_conf': Tensor of shape (1, T, H, W) - confidence scores
            where T is the number of frames (typically 128)
        num_samples: Number of frames to select (m)
        voxel_density: Voxel grid density (voxels per scene dimension), controls voxel_size
        conf_percentile: Percentile for confidence threshold (0-100)
        conf_threshold: Minimum confidence value for valid points
        verbose: Print timing and debug information
    
    Returns:
        selected_indices: List of selected frame indices (length = num_samples)
                         Indices are in range [0, T-1] where T is number of frames in predictions
    
    Note:
        - The returned indices refer to positions in the predictions dict (0-127 for 128 frames)
        - These may need to be mapped to original video frame IDs depending on context
    """
    start_time = time.time()
    
    # Step 1: Sanity check and extract data
    if verbose:
        print("="*70)
        print("SA Sampling - Sanity Checks")
        print("="*70)
    
    required_keys = ['world_points', 'world_points_conf']
    for key in required_keys:
        if key not in predictions:
            raise ValueError(f"Missing required key '{key}' in predictions dict")
    
    world_points = predictions['world_points']  # (1, T, H, W, 3)
    world_points_conf = predictions['world_points_conf']  # (1, T, H, W)
    
    # Validate shapes
    if world_points.dim() != 5:
        raise ValueError(f"Expected world_points to be 5D (1, T, H, W, 3), got {world_points.dim()}D")
    if world_points_conf.dim() != 4:
        raise ValueError(f"Expected world_points_conf to be 4D (1, T, H, W), got {world_points_conf.dim()}D")
    
    batch_size, T, H, W, xyz = world_points.shape
    if batch_size != 1:
        raise ValueError(f"Expected batch_size=1, got {batch_size}")
    if xyz != 3:
        raise ValueError(f"Expected world_points last dim=3, got {xyz}")
    
    if world_points_conf.shape != (1, T, H, W):
        raise ValueError(
            f"Shape mismatch: world_points {world_points.shape} vs "
            f"world_points_conf {world_points_conf.shape}"
        )
    
    if num_samples > T:
        raise ValueError(f"Cannot sample {num_samples} frames from {T} total frames")
    
    if verbose:
        print(f"  ✓ world_points shape: {tuple(world_points.shape)}")
        print(f"  ✓ world_points_conf shape: {tuple(world_points_conf.shape)}")
        print(f"  ✓ Total frames (T): {T}")
        print(f"  ✓ Sampling {num_samples} frames")
        print(f"  ✓ Image resolution: {H} x {W}")
    
    if num_samples == T:
        return list(range(T))
    
    # Step 2: Compute confidence mask
    t0 = time.time()
    world_points_flat = world_points.reshape(-1, 3)  # (B, 3)
    world_points_conf_flat = world_points_conf.reshape(-1)  # (B)
    
    # Compute confidence threshold
    init_threshold_val = float(np.percentile(world_points_conf_flat.cpu().numpy(), conf_percentile))
    world_points_conf_mask = (world_points_conf >= init_threshold_val) & (world_points_conf > conf_threshold)
    world_points_conf_flat_mask = (world_points_conf_flat >= init_threshold_val) & (world_points_conf_flat > conf_threshold)
    
    valid_count = world_points_conf_flat_mask.sum().item()
    total_count = world_points_conf_flat.numel()
    
    if verbose:
        print(f"\n  Confidence filtering:")
        print(f"    Percentile threshold ({conf_percentile}%): {init_threshold_val:.4f}")
        print(f"    Min confidence: {conf_threshold:.4f}")
        print(f"    Valid points: {valid_count:,} / {total_count:,} ({100*valid_count/total_count:.1f}%)")
    
    if valid_count == 0:
        raise ValueError("No valid points after confidence filtering")
    
    t1 = time.time()
    if verbose:
        print(f"  Time: {(t1-t0)*1000:.2f} ms")
    
    # Step 3: Compute bounding box and voxel size
    t0 = time.time()
    valid_points = world_points_flat[world_points_conf_flat_mask]  # (N_valid, 3)
    x_min, y_min, z_min = valid_points.min(dim=0)[0]
    x_max, y_max, z_max = valid_points.max(dim=0)[0]
    
    scene_range = min(x_max - x_min, y_max - y_min, z_max - z_min)
    voxel_size = scene_range / voxel_density
    
    if verbose:
        print(f"\n  Scene bounding box:")
        print(f"    X: [{x_min:.2f}, {x_max:.2f}] range={x_max-x_min:.2f}")
        print(f"    Y: [{y_min:.2f}, {y_max:.2f}] range={y_max-y_min:.2f}")
        print(f"    Z: [{z_min:.2f}, {z_max:.2f}] range={z_max-z_min:.2f}")
        print(f"    Voxel size: {voxel_size:.4f} (density={voxel_density})")
    
    t1 = time.time()
    if verbose:
        print(f"  Time: {(t1-t0)*1000:.2f} ms")
    
    # Step 4: Compute voxel sets for each frame
    t0 = time.time()
    if verbose:
        print(f"\n  Computing voxel sets for {T} frames...")
    
    voxel_sets = compute_voxel_sets(
        world_points=world_points,
        world_points_conf_mask=world_points_conf_mask,
        x_min=x_min.item(),
        y_min=y_min.item(),
        z_min=z_min.item(),
        voxel_size=voxel_size.item()
    )
    
    # Compute statistics
    voxel_counts = [len(vset) for vset in voxel_sets]
    total_unique_voxels = len(set().union(*voxel_sets))
    
    if verbose:
        print(f"  Voxel set statistics:")
        print(f"    Per-frame voxels: min={min(voxel_counts)}, max={max(voxel_counts)}, "
              f"mean={np.mean(voxel_counts):.1f}, median={np.median(voxel_counts):.1f}")
        print(f"    Total unique voxels (union): {total_unique_voxels:,}")
    
    t1 = time.time()
    if verbose:
        print(f"  Time: {(t1-t0)*1000:.2f} ms")
    
    # Step 5: Greedy maximum coverage sampling
    t0 = time.time()
    if verbose:
        print(f"\n  Running greedy maximum coverage sampling...")
    
    selected_indices = maximum_coverage_sampling(voxel_sets, num_samples)
    
    t1 = time.time()
    if verbose:
        print(f"  Time: {(t1-t0)*1000:.2f} ms")
    
    # Step 6: Analyze coverage
    if verbose:
        covered_voxels = set().union(*[voxel_sets[i] for i in selected_indices])
        coverage_ratio = len(covered_voxels) / total_unique_voxels if total_unique_voxels > 0 else 0
        
        print(f"\n  Sampling results:")
        print(f"    Selected {len(selected_indices)} frames")
        print(f"    Selected indices (sorted): {sorted(selected_indices)}")
        print(f"    Covered voxels: {len(covered_voxels):,} / {total_unique_voxels:,} ({coverage_ratio*100:.1f}%)")
    
    total_time = time.time() - start_time
    if verbose:
        print(f"\n  Total time: {total_time*1000:.2f} ms")
        print("="*70)
    
    # Return sorted indices (for consistency with other methods)
    return sorted(selected_indices)


def analyze_sa_coverage(
    predictions: Dict[str, torch.Tensor],
    selected_indices: List[int],
    voxel_density: float = 20.0,
    conf_percentile: float = 50.0,
    conf_threshold: float = 0.1
) -> dict:
    """
    Analyze coverage quality in 3D voxel space.
    
    Similar to analyze_pose_coverage in pose_fps_sampling.py but operates
    in 3D voxel space instead of SE(3) space.
    
    Args:
        predictions: VGGT predictions dict
        selected_indices: Indices of selected frames
        voxel_density: Voxel grid density
        conf_percentile: Percentile for confidence threshold
        conf_threshold: Minimum confidence value
    
    Returns:
        Dictionary containing coverage statistics
    """
    world_points = predictions['world_points']
    world_points_conf = predictions['world_points_conf']
    
    T = world_points.shape[1]
    selected_set = set(selected_indices)
    
    # Compute voxel sets (reuse preprocessing from sa_sampling)
    world_points_flat = world_points.reshape(-1, 3)
    world_points_conf_flat = world_points_conf.reshape(-1)
    
    init_threshold_val = float(np.percentile(world_points_conf_flat.cpu().numpy(), conf_percentile))
    world_points_conf_mask = (world_points_conf >= init_threshold_val) & (world_points_conf > conf_threshold)
    world_points_conf_flat_mask = (world_points_conf_flat >= init_threshold_val) & (world_points_conf_flat > conf_threshold)
    
    valid_points = world_points_flat[world_points_conf_flat_mask]
    x_min, y_min, z_min = valid_points.min(dim=0)[0]
    x_max, y_max, z_max = valid_points.max(dim=0)[0]
    
    scene_range = min(x_max - x_min, y_max - y_min, z_max - z_min)
    voxel_size = scene_range / voxel_density
    
    voxel_sets = compute_voxel_sets(
        world_points=world_points,
        world_points_conf_mask=world_points_conf_mask,
        x_min=x_min.item(),
        y_min=y_min.item(),
        z_min=z_min.item(),
        voxel_size=voxel_size.item()
    )
    
    # Compute coverage for selected frames
    selected_voxels = set().union(*[voxel_sets[i] for i in selected_indices])
    total_unique_voxels = len(set().union(*voxel_sets))
    
    # For each frame, compute distance to nearest selected frame (in voxel space)
    # Distance = Jaccard distance: 1 - |A ∩ B| / |A ∪ B|
    min_distances = np.full(T, np.inf)
    for i in range(T):
        for j in selected_indices:
            if len(voxel_sets[i]) == 0 or len(voxel_sets[j]) == 0:
                dist = 1.0
            else:
                intersection = len(voxel_sets[i] & voxel_sets[j])
                union = len(voxel_sets[i] | voxel_sets[j])
                dist = 1.0 - (intersection / union if union > 0 else 0)
            min_distances[i] = min(min_distances[i], dist)
    
    # Compute statistics
    non_selected_distances = [min_distances[i] for i in range(T) if i not in selected_set]
    
    return {
        'min_distances': min_distances,
        'mean_min_distance': np.mean(non_selected_distances) if non_selected_distances else 0.0,
        'max_min_distance': np.max(non_selected_distances) if non_selected_distances else 0.0,
        'coverage_uniformity': np.std(min_distances),
        'num_selected': len(selected_indices),
        'num_total': T,
        'covered_voxels': len(selected_voxels),
        'total_unique_voxels': total_unique_voxels,
        'coverage_ratio': len(selected_voxels) / total_unique_voxels if total_unique_voxels > 0 else 0.0,
        'voxel_sets': voxel_sets  # For further analysis
    }


def extract_poses_from_predictions(predictions: Dict[str, torch.Tensor]) -> np.ndarray:
    """
    Extract 4x4 pose matrices from VGGT predictions.
    
    Args:
        predictions: VGGT predictions dict containing 'extrinsic' (1, T, 3, 4)
    
    Returns:
        poses: (T, 4, 4) array of pose matrices
    """
    if 'extrinsic' not in predictions:
        raise ValueError("No 'extrinsic' found in predictions dict")
    
    extrinsic = predictions['extrinsic']  # (1, T, 3, 4)
    
    if extrinsic.dim() != 4 or extrinsic.shape[0] != 1 or extrinsic.shape[2:] != (3, 4):
        raise ValueError(f"Expected extrinsic shape (1, T, 3, 4), got {tuple(extrinsic.shape)}")
    
    T = extrinsic.shape[1]
    extrinsic = extrinsic[0]  # (T, 3, 4)
    
    # Convert to 4x4 by adding [0, 0, 0, 1] row
    poses = np.zeros((T, 4, 4), dtype=np.float32)
    poses[:, :3, :] = extrinsic.cpu().numpy()
    poses[:, 3, 3] = 1.0
    
    return poses


def load_gt_poses_from_json(
    json_path: str,
    frame_indices: Optional[List[int]] = None,
    num_frames_expected: Optional[int] = None
) -> np.ndarray:
    """
    Load ground truth poses from ScanNet++ transforms JSON file.
    
    This loads GT camera poses from the original dataset, typically from:
    - ScanNet++: data/{scene}/dslr/transforms_undistorted.json
    - Processed outputs: selected_frames.json with 'transform_matrix' field
    
    IMPORTANT: When using with VGGT predictions, you must provide frame_indices
    to extract the subset of GT poses corresponding to the frames used in VGGT input.
    
    Workflow:
        1. VGGT input: Video frames are uniformly sampled to 128 frames
        2. VGGT predictions: Contains data for these 128 frames
        3. GT poses: Must be subsampled using the same frame_indices
    
    Args:
        json_path: Path to JSON file containing transform matrices
        frame_indices: Optional list of frame indices to extract (for subsampling)
                      Example: [0, 2, 4, ...] for every 2nd frame
                      If None, load all frames
        num_frames_expected: Expected number of frames after subsampling (for validation)
    
    Returns:
        poses: (T, 4, 4) array of GT pose matrices
    """
    import json
    from pathlib import Path
    
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"GT poses JSON not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if 'frames' in data:
        # ScanNet++ format: transforms_undistorted.json
        frames = data['frames']
        pose_matrices = [frame['transform_matrix'] for frame in frames]
    elif 'transform_matrix' in data:
        # Processed format: selected_frames.json (from process_and_sample_scannetpp.py)
        pose_matrices = [frame['pose_matrix'] for frame in data['transform_matrix']]
    else:
        raise ValueError(f"Unknown JSON format. Expected 'frames' or 'transform_matrix' key. Found keys: {list(data.keys())}")
    
    # Convert to numpy array
    poses = np.array(pose_matrices, dtype=np.float32)
    
    print(f"  Loaded {len(poses)} GT poses from JSON")
    
    # Subsample if frame_indices provided
    if frame_indices is not None:
        if max(frame_indices) >= len(poses):
            raise ValueError(
                f"frame_indices contains index {max(frame_indices)} but GT poses only has {len(poses)} frames"
            )
        poses = poses[frame_indices]
        print(f"  Subsampled to {len(poses)} poses using frame_indices")
    
    # Validate expected count
    if num_frames_expected is not None and len(poses) != num_frames_expected:
        raise ValueError(
            f"Expected {num_frames_expected} poses but got {len(poses)} after subsampling. "
            f"Check that frame_indices matches the VGGT input sampling."
        )
    
    # Validate shape
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shape (T, 4, 4), got {poses.shape}")
    
    return poses


def compute_frame_indices_for_vggt(total_frames: int, sample_count: int = 128) -> np.ndarray:
    """
    Compute frame indices used for VGGT input sampling.
    
    JJ: This replicates the sampling logic from sa_sampling.py:extract_initial_frames_from_video (line 537)
    to determine which frames from the original video were used as VGGT input.
    
    The logic is:
        sample_count = min(sample_count, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int)
    
    Args:
        total_frames: Total number of frames in original video/dataset
        sample_count: Number of frames to sample for VGGT (default: 128)
    
    Returns:
        frame_indices: Array of frame indices used for VGGT input
    """
    sample_count = min(sample_count, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int)
    return frame_indices


def main():
    """
    Test SA sampling with saved VGGT predictions and visualize results.
    """
    import json
    
    # JJ: Configuration
    GENERATE_VISUALIZATIONS = True  # Set to True to generate HTML visualizations
    PLOT_POSE_ANALYSIS = True  # Set to True to also generate pose_analysis.html with all frames
    USE_GT_POSES = False  # Set to True to use GT poses instead of VGGT predicted poses
    
    # Paths
    predictions_path = "/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/vsibench/sa_sampling_16f_single_video/arkitscenes/00777c41d4/sa_predictions.pt"
    gt_poses_path = "/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/scannetpp/pa_sampling_all/00777c41d4/selected_frames.json"
    
    # Load predictions from saved file
    print(f"Loading predictions from:")
    print(f"  {predictions_path}")
    
    predictions = torch.load(predictions_path, weights_only=False)
    
    print(f"\nPredictions dict keys: {list(predictions.keys())}")
    print(f"Shapes:")
    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {tuple(val.shape)}")
    
    # Get number of frames in predictions
    T_pred = predictions['world_points'].shape[1]  # Should be 128
    print(f"\nNumber of frames in predictions: {T_pred}")
    
    # Extract poses for visualization
    if USE_GT_POSES:
        print(f"\n{'='*70}")
        print("Loading Ground Truth Poses")
        print(f"{'='*70}")
        print(f"  GT poses path: {gt_poses_path}")
        
        # Load GT poses JSON to get total frame count
        with open(gt_poses_path, 'r') as f:
            gt_data = json.load(f)
        
        # Get total frames from GT
        if 'transform_matrix' in gt_data:
            total_frames_gt = len(gt_data['transform_matrix'])
            print(f"  Total frames in GT: {total_frames_gt}")
        else:
            raise ValueError("GT JSON does not contain 'transform_matrix'")
        
        # Compute frame indices used for VGGT input (replicates sa_sampling workflow)
        print(f"\n  Computing frame indices for VGGT input sampling...")
        print(f"    VGGT samples {T_pred} frames uniformly from {total_frames_gt} total frames")
        
        frame_indices = compute_frame_indices_for_vggt(
            total_frames=total_frames_gt,
            sample_count=T_pred  # Should be 128
        )
        
        print(f"    Frame indices: {frame_indices[:10]}... (first 10)")
        print(f"    Frame index range: [{frame_indices[0]}, {frame_indices[-1]}]")
        print(f"    Frame index step (approx): {(frame_indices[-1] - frame_indices[0]) / (len(frame_indices) - 1):.2f}")
        
        # Load GT poses with subsampling
        poses = load_gt_poses_from_json(
            gt_poses_path,
            frame_indices=frame_indices.tolist(),
            num_frames_expected=T_pred
        )
        
        pose_source = "Ground Truth (subsampled)"
    else:
        poses = extract_poses_from_predictions(predictions)
        pose_source = "VGGT Predicted"
    
    print(f"\n  Final poses shape: {poses.shape}")
    print(f"  Pose source: {pose_source}")
    
    # Validate poses match predictions
    if poses.shape[0] != T_pred:
        raise ValueError(
            f"Mismatch: poses has {poses.shape[0]} frames but predictions has {T_pred} frames"
        )
    
    num_samples = 16
    print(f"\n{'='*70}")
    print(f"Testing SA Sampling - selecting {num_samples} frames")
    print(f"{'='*70}")
    print(f"Generate visualizations: {'Yes' if GENERATE_VISUALIZATIONS else 'No'}")
    print(f"Generate pose_analysis.html: {'Yes' if PLOT_POSE_ANALYSIS else 'No'}")
    print(f"Pose source for visualization: {pose_source}")
    
    num_samples = 16
    print(f"\n{'='*70}")
    print(f"Testing SA Sampling - selecting {num_samples} frames")
    print(f"{'='*70}")
    print(f"Generate visualizations: {'Yes' if GENERATE_VISUALIZATIONS else 'No'}")
    print(f"Generate pose_analysis.html: {'Yes' if PLOT_POSE_ANALYSIS else 'No'}")
    
    # Import visualization
    has_vis = False
    if GENERATE_VISUALIZATIONS:
        try:
            from visualisation import visualize_sampled_pose
            has_vis = True
        except ImportError:
            print("Warning: visualisation module not found, skipping visualization")
    
    # Pre-compute SE(3) distance matrix for visualization (if needed)
    distance_matrix_se3 = None
    if has_vis:
        try:
            from pose_fps_sampling import compute_pairwise_pose_distance
            print("\nPre-computing SE(3) distance matrix for visualization...")
            distance_matrix_se3 = compute_pairwise_pose_distance(poses, distance_mode='max_norm')
            print(f"  Distance matrix shape: {distance_matrix_se3.shape}")
        except Exception as e:
            print(f"Warning: Could not compute SE(3) distance matrix: {e}")
    
    # Run SA sampling
    print(f"\n{'='*70}")
    print("Running SA Sampling")
    print(f"{'='*70}")
    
    selected = sa_sampling(
        predictions=predictions,
        num_samples=num_samples,
        voxel_density=20.0,
        conf_percentile=50.0,
        conf_threshold=0.1,
        verbose=True
    )
    
    print(f"\n{'='*70}")
    print(f"Selected {len(selected)} frames: {selected}")
    print(f"{'='*70}")
    
    # Analyze coverage
    print(f"\n{'='*70}")
    print("Coverage Analysis")
    print(f"{'='*70}")
    
    coverage = analyze_sa_coverage(predictions, selected)
    print(f"  Coverage ratio: {coverage['coverage_ratio']*100:.1f}%")
    print(f"  Covered voxels: {coverage['covered_voxels']:,} / {coverage['total_unique_voxels']:,}")
    print(f"  Mean min distance (Jaccard): {coverage['mean_min_distance']:.4f}")
    print(f"  Max min distance (Jaccard): {coverage['max_min_distance']:.4f}")
    print(f"  Coverage uniformity (std): {coverage['coverage_uniformity']:.4f}")
    
    # Visualize results
    if has_vis and distance_matrix_se3 is not None:
        print(f"\n{'='*70}")
        print("Generating Visualization")
        print(f"{'='*70}")
        
        try:
            from pose_fps_sampling import analyze_pose_coverage
            
            # Compute SE(3) coverage for visualization
            coverage_se3 = analyze_pose_coverage(poses, selected, 'max_norm', True)
            
            print(f"  SE(3) coverage statistics:")
            print(f"    Mean min distance: {coverage_se3['mean_min_distance']:.4f}")
            print(f"    Max min distance: {coverage_se3['max_min_distance']:.4f}")
            print(f"    Coverage uniformity (std): {coverage_se3['coverage_uniformity']:.4f}")
            
            output_name = "tmp/sa_sampling_quality.html"
            print(f"  → Saving to {output_name}...")
            
            visualize_sampled_pose(
                pose_analysis_poses=poses,
                selected_indices=selected,
                min_distances=coverage_se3['min_distances'],
                distance_matrix=distance_matrix_se3,
                output_path=output_name,
                show=False,
                method_name='SA Sampling (Space-Aware)',
                plot_pose_analysis=PLOT_POSE_ANALYSIS,
                pose_source=pose_source  # JJ: Pass pose source for annotation
            )
            
            print(f"  ✓ Visualization saved to: {output_name}")
            if PLOT_POSE_ANALYSIS:
                print(f"  ✓ Pose analysis saved to: pose_analysis.html")
        
        except Exception as e:
            print(f"  Warning: Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✓ SA sampling testing complete!")
    if has_vis:
        print("✓ Visualizations saved to current directory")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
