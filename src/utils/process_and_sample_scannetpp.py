# JJ: Process ScanNet++ DSLR images for pose-aware sampling
import os
import json
import shutil
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Literal, Optional, Dict, Any

# Handle both relative import (when used as module) and absolute import (when run as script)
try:
    from . import pose_fps_sampling
    from . import pose_efficient_sampling
    from . import flexiable_sa_sampling
    from .visualisation import visualize_sampled_pose
    # JJ: Import Block 2 sampling functions from pa_sampling
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1] / 'sampling'))
    from src.sampling import pa_sampling
except ImportError:
    import pose_fps_sampling
    import pose_efficient_sampling
    import flexiable_sa_sampling
    try:
        from visualisation import visualize_sampled_pose
    except ImportError:
        visualize_sampled_pose = None
    # JJ: Import Block 2 sampling functions from pa_sampling
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1] / 'sampling'))
    from src.sampling import pa_sampling


def process_dslr(
    dataset_root: str,
    scene_name: str,
    target_size: int = 640,
    full_video_fps: int = 30,
    sample_video_fps: int = 10,
    pre_filter_strategy: Literal['all', 'good'] = 'good',
    pose_aware_strategy: Literal['None', 'fps', 'uniform', 'efficient', 'sa'] = 'None',
    sample_pose_source: Literal['gt', 'vggt_max_128'] = 'gt',  # JJ: Pose source for sampling (FPS/Efficient)
    vis_pose_source: Literal['gt', 'vggt_max_128'] = 'gt',  # JJ: Pose source for visualization (all strategies)
    predictions_path: Optional[str] = None,  # JJ: Path to VGGT predictions (required when using vggt_max_128)
    pa_fps_hyper_setting_dict: Optional[Dict[str, Any]] = None,
    uniform_hyper_setting_dict: Optional[Dict[str, Any]] = None,
    efficient_hyper_setting_dict: Optional[Dict[str, Any]] = None,
    sa_hyper_setting_dict: Optional[Dict[str, Any]] = None,
    save_full_video: bool = True,
    construct_sample_video: bool = False,
    visualize_sampling: bool = True,
    plot_pose_analysis: bool = False,  # JJ: Control whether to generate pose_analysis.html
    pose_analysis_target: Literal['raw_frames', 'pre_filtered', 'sampled'] = 'pre_filtered'  # JJ: Control which frames to analyze
):
    """
    Process ScanNet++ DSLR images: filter good poses, apply pose-aware sampling, copy images, generate video and metadata.
    
    Args:
        dataset_root: Path to scannetpp dataset root
        scene_name: Scene name (e.g., '00777c41d4')
        target_size: Maximum dimension for video frames (default: 640)
        full_video_fps: Frame rate for full video (default: 30)
        sample_video_fps: Frame rate for sample video (default: 10)
        pre_filter_strategy: Pre-filtering based on pose quality
            - 'all': Use all frames
            - 'good': Filter out frames with is_bad=True (default)
        pose_aware_strategy: Pose-aware sampling strategy applied after pre-filtering
            - 'None': Use all pre-filtered frames (default)
            - 'fps': Farthest Point Sampling in SE(3) space
            - 'uniform': Uniform sampling (select every k-th frame)
            - 'efficient': Efficient sampling in 2D (translation, rotation) space
            - 'sa': Space-Aware sampling using 3D voxel coverage (requires VGGT predictions)
        pa_fps_hyper_setting_dict: Hyperparameters for FPS sampling, e.g.:
            {
                'num_samples': 20,
                'distance_mode': 'max_norm',  # or 'data_driven'
                'starting_mode': 'medoid',    # 'first', 'rand', 'medoid'
                'reorth_rot': True
            }
        uniform_hyper_setting_dict: Hyperparameters for uniform sampling, e.g.:
            {
                'num_samples': 20
            }
        efficient_hyper_setting_dict: Hyperparameters for efficient sampling, e.g.:
            {
                'num_samples': 20,
                'sampling_mode': 'hybrid',      # 'grid', 'hybrid', 'fps_2d'
                'normalization': 'std_norm',    # 'std_norm', 'max_norm'
                'diagonal_priority': 0.0,       # Recommended: 0.0 for optimal mutual coverage
                'starting_mode': 'farthest',    # 'farthest', 'medoid', 'first'
                'grid_density': 1.5
            }
        sa_hyper_setting_dict: Hyperparameters for SA sampling, e.g.:
            {
                'num_samples': 20,
                'predictions_path': 'path/to/sa_predictions.pt',  # Required: VGGT predictions
                'voxel_density': 20.0,          # Voxels per scene dimension
                'conf_percentile': 50.0,        # Confidence threshold percentile
                'conf_threshold': 0.1,          # Minimum confidence value
                'use_gt_poses': True,           # Use GT poses for visualization (not sampling)
            }
            Note: SA sampling requires pre-computed VGGT predictions (.pt file)
        save_full_video: Whether to generate full video from all images (default: True)
        construct_sample_video: Whether to generate video from sampled images only (default: False)
        visualize_sampling: Whether to generate sampling quality visualization with 2 figures (default: True)
        plot_pose_analysis: If True, also generate pose_analysis.html with farness analysis. (default: False)
        pose_analysis_target: Which frame set to analyze in pose_analysis.html:
            - 'raw_frames': All original frames (no filtering)
            - 'pre_filtered': Frames after pre-filtering (default)
            - 'sampled': Final sampled frames only
    """
    if pre_filter_strategy not in ['all', 'good']:
        raise NotImplementedError(f"pre_filter_strategy '{pre_filter_strategy}' not implemented")
    
    if pose_aware_strategy not in ['None', 'fps', 'uniform', 'efficient', 'sa']:
        raise NotImplementedError(f"pose_aware_strategy '{pose_aware_strategy}' not implemented")
    
    # JJ: Validation - Prohibit dangerous pose source combination
    # When sampling from GT (244 frames) but visualizing with VGGT (128 frames),
    # selected frame indices may not exist in VGGT's 128 uniformly sampled frames
    if pose_aware_strategy in ['fps', 'efficient', 'uniform', 'None']:
        if sample_pose_source == 'gt' and vis_pose_source == 'vggt_max_128':
            raise ValueError(
                "\n" + "="*80 + "\n"
                "❌ ERROR: Invalid pose source combination\n"
                "="*80 + "\n"
                f"  Strategy:           {pose_aware_strategy}\n"
                f"  sample_pose_source: '{sample_pose_source}'\n"
                f"  vis_pose_source:    '{vis_pose_source}'\n"
                "\n"
                "  ❌ REASON:\n"
                "    Frames sampled from GT poses (e.g., 244 frames) may not exist in\n"
                "    VGGT predictions (max 128 uniformly sampled frames).\n"
                "\n"
                "    Example: If GT has 244 frames and sampling selects frame indices\n"
                "             [5, 87, 156, 230, ...], but VGGT only has 128 frames\n"
                "             [0, 2, 4, 6, ...], then indices 156, 230, ... don't exist\n"
                "             in VGGT predictions, causing visualization to fail.\n"
                "\n"
                "  ✅ VALID COMBINATIONS:\n"
                "    1. sample_pose_source='gt'          + vis_pose_source='gt'\n"
                "    2. sample_pose_source='vggt_max_128' + vis_pose_source='vggt_max_128'\n"
                "    3. sample_pose_source='vggt_max_128' + vis_pose_source='gt'  (requires mapping)\n"
                "\n"
                "  Note: SA strategy allows gt+vggt because sampling uses voxels, not poses.\n"
                "="*80
            )
    
    # Default FPS hyperparameters
    if pa_fps_hyper_setting_dict is None:
        pa_fps_hyper_setting_dict = {
            'num_samples': 20,
            'distance_mode': 'max_norm',
            'starting_mode': 'medoid',
            'reorth_rot': True
        }
    
    # Default uniform hyperparameters
    if uniform_hyper_setting_dict is None:
        uniform_hyper_setting_dict = {
            'num_samples': 20
        }
    
    # Default efficient hyperparameters
    if efficient_hyper_setting_dict is None:
        efficient_hyper_setting_dict = {
            'num_samples': 20,
            'sampling_mode': 'hybrid',
            'normalization': 'std_norm',
            'diagonal_priority': 0.0,  # Pure FPS for optimal coverage
            'starting_mode': 'farthest',
            'grid_density': 1.5
        }
    
    # Default SA hyperparameters
    if sa_hyper_setting_dict is None:
        sa_hyper_setting_dict = {
            'num_samples': 20,
            'predictions_path': None,  # Must be provided by user
            'voxel_density': 20.0,
            'conf_percentile': 50.0,
            'conf_threshold': 0.1,
            'vis_pose_source': 'gt'  # JJ: Only vis_pose_source for SA (sampling uses voxels)
        }
    
    # JJ: Warn if pose sources differ for FPS/Efficient strategies
    if pose_aware_strategy in ['fps', 'efficient']:
        if sample_pose_source != vis_pose_source:
            print("\n" + "=" * 80)
            print("⚠️  WARNING: Inconsistent Pose Sources")
            print("=" * 80)
            print(f"Strategy: {pose_aware_strategy.upper()}")
            print(f"  sample_pose_source: '{sample_pose_source}'")
            print(f"  vis_pose_source:    '{vis_pose_source}'")
            print()
            print(f"⚠️  {pose_aware_strategy.upper()} sampling is directly based on pose distances.")
            print("   Using different pose sources for sampling and visualization may lead to")
            print("   confusing results, as the visualization will not accurately reflect")
            print("   the pose-space coverage achieved by the sampling.")
            print()
            print("   Recommendation: Use the same pose source for both sampling and visualization.")
            print("=" * 80 + "\n")
    
    # Setup paths
    data_dir = Path(dataset_root) / "data" / scene_name
    transforms_path = data_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    images_dir = data_dir / "dslr" / "resized_undistorted_images"
    
    # Construct output directory name based on strategies
    if pose_aware_strategy == 'None':
        output_subdir = f"pa_sampling_{pre_filter_strategy}"
    elif pose_aware_strategy == 'fps':
        num_samples = pa_fps_hyper_setting_dict.get('num_samples', 20)
        output_subdir = f"pa_sampling_{pre_filter_strategy}_fps_{num_samples}"
    elif pose_aware_strategy == 'uniform':
        num_samples = uniform_hyper_setting_dict.get('num_samples', 20)
        output_subdir = f"pa_sampling_{pre_filter_strategy}_uniform_{num_samples}"
    elif pose_aware_strategy == 'efficient':
        num_samples = efficient_hyper_setting_dict.get('num_samples', 20)
        sampling_mode = efficient_hyper_setting_dict.get('sampling_mode', 'hybrid')
        output_subdir = f"pa_sampling_{pre_filter_strategy}_efficient_{sampling_mode}_{num_samples}"
    elif pose_aware_strategy == 'sa':
        num_samples = sa_hyper_setting_dict.get('num_samples', 20)
        output_subdir = f"pa_sampling_{pre_filter_strategy}_sa_{num_samples}"
    else:
        raise NotImplementedError(f"pose_aware_strategy '{pose_aware_strategy}' not implemented")
        output_subdir = f"pa_sampling_{pre_filter_strategy}_{pose_aware_strategy}"
    
    output_dir = Path(dataset_root) / output_subdir / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_dir = Path(dataset_root) / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{scene_name}.mp4"
    
    # Load transforms
    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)
    
    frames = transforms_data['frames']
    
    # Step 1: Pre-filter frames based on strategy
    pre_filtered_indices = []
    pre_filtered_frames = []
    
    for idx, frame in enumerate(frames):
        if pre_filter_strategy == 'good' and frame.get('is_bad', False):
            continue
        
        pre_filtered_indices.append(idx)
        pre_filtered_frames.append(frame)
    
    print(f"Pre-filtering ({pre_filter_strategy}): {len(pre_filtered_indices)}/{len(frames)} frames")
    
    # JJ: Load VGGT predictions if sample_pose_source is vggt_max_128 (for FPS/Efficient)
    vggt_predictions = None
    vggt_poses_for_sampling = None
    vggt_frame_indices = None
    
    if sample_pose_source == 'vggt_max_128' and pose_aware_strategy in ['fps', 'efficient', 'uniform']:
        if predictions_path is None:
            raise ValueError(
                f"sample_pose_source='vggt_max_128' requires --predictions_path for {pose_aware_strategy} strategy"
            )
        
        import torch
        print(f"\n  Loading VGGT predictions for sampling from: {predictions_path}")
        vggt_predictions = torch.load(predictions_path, weights_only=False)
        
        # Extract poses from VGGT predictions
        vggt_poses_for_sampling = flexiable_sa_sampling.extract_poses_from_predictions(vggt_predictions)
        T_vggt = vggt_poses_for_sampling.shape[0]
        print(f"  ✓ Loaded {T_vggt} VGGT predicted poses for sampling")
        
        # Sanity check shape
        if T_vggt > 128:
            print(f"  ⚠️  Warning: VGGT predictions have {T_vggt} frames (expected max 128)")
        
        # Compute which original frame indices correspond to these VGGT frames
        # Reuses sa_sampling.py:extract_initial_frames_from_video logic (line 537)
        vggt_frame_indices = flexiable_sa_sampling.compute_frame_indices_for_vggt(
            total_frames=len(frames),
            sample_count=T_vggt
        )
        print(f"  VGGT frames correspond to original indices: {vggt_frame_indices[:10]}... (showing first 10)")
    
    # Step 2: Apply pose-aware sampling strategy on pre-filtered frames
    sampling_time_ms = 0.0  # Track core sampling time
    
    if pose_aware_strategy == 'None':
        # Use all pre-filtered frames
        final_indices = pre_filtered_indices
        sampling_time_ms = 0.0
        
    elif pose_aware_strategy == 'fps':
        # Apply Farthest Point Sampling
        print(f"Applying FPS with settings: {pa_fps_hyper_setting_dict}")
        print(f"  Pose source for sampling: {sample_pose_source}")
        
        # JJ: Choose pose source for sampling
        if sample_pose_source == 'vggt_max_128':
            # Use VGGT predicted poses (already loaded)
            poses = vggt_poses_for_sampling
            sampling_pool_frame_indices = vggt_frame_indices
            print(f"  Using VGGT predicted poses: {poses.shape[0]} frames")
        else:  # 'gt'
            # Extract GT poses from pre-filtered frames
            poses = np.array([frame['transform_matrix'] for frame in pre_filtered_frames])
            sampling_pool_frame_indices = np.array(pre_filtered_indices)
            print(f"  Using GT poses: {poses.shape[0]} frames")
        
        # JJ: Use Block 2 sampling function from pa_sampling
        start_time = time.time()
        fps_indices = pa_sampling.run_fps_sampling(
            poses=poses,
            num_samples=pa_fps_hyper_setting_dict['num_samples'],
            distance_mode=pa_fps_hyper_setting_dict['distance_mode'],
            starting_mode=pa_fps_hyper_setting_dict['starting_mode'],
            reorth_rot=pa_fps_hyper_setting_dict['reorth_rot'],
            verbose=False
        )
        sampling_time_ms = (time.time() - start_time) * 1000
        
        # Map back to original frame indices
        final_indices = [int(sampling_pool_frame_indices[i]) for i in fps_indices]
        
        print(f"FPS selected {len(final_indices)} frames from {len(poses)} poses")
        print(f"  ⏱ Sampling time: {sampling_time_ms:.2f} ms")
        
    elif pose_aware_strategy == 'uniform':
        # Apply Uniform Sampling (select every k-th frame)
        num_samples = uniform_hyper_setting_dict['num_samples']
        
        print(f"Applying Uniform Sampling with num_samples={num_samples}")
        print(f"  Pose source for sampling: {sample_pose_source}")
        
        # JJ: Choose pose source for sampling
        if sample_pose_source == 'vggt_max_128':
            # Use VGGT frame pool
            n_frames = len(vggt_frame_indices)
            sampling_pool_frame_indices = vggt_frame_indices
            print(f"  Using VGGT frame pool: {n_frames} frames")
        else:  # 'gt'
            # Use pre-filtered frames
            n_frames = len(pre_filtered_indices)
            sampling_pool_frame_indices = np.array(pre_filtered_indices)
            print(f"  Using GT frame pool: {n_frames} frames")
        
        # Time the core sampling computation
        start_time = time.time()
        if num_samples >= n_frames:
            # If requested samples >= available frames, use all
            uniform_indices = list(range(n_frames))
        else:
            # Calculate step size to uniformly sample num_samples frames
            # Use linspace to get evenly spaced indices
            uniform_indices = np.linspace(0, n_frames - 1, num_samples, dtype=int).tolist()
        sampling_time_ms = (time.time() - start_time) * 1000
        
        # Map back to original frame indices
        final_indices = [int(sampling_pool_frame_indices[i]) for i in uniform_indices]
        
        print(f"Uniform sampling selected {len(final_indices)} frames from {n_frames} frames")
        print(f"  ⏱ Sampling time: {sampling_time_ms:.2f} ms")
        
    elif pose_aware_strategy == 'efficient':
        # Apply Efficient Sampling in 2D (translation, rotation) space
        print(f"Applying Efficient Sampling with settings: {efficient_hyper_setting_dict}")
        print(f"  Pose source for sampling: {sample_pose_source}")
        
        # JJ: Choose pose source for sampling
        if sample_pose_source == 'vggt_max_128':
            # Use VGGT predicted poses (already loaded)
            poses = vggt_poses_for_sampling
            sampling_pool_frame_indices = vggt_frame_indices
            print(f"  Using VGGT predicted poses: {poses.shape[0]} frames")
        else:  # 'gt'
            # Extract GT poses from pre-filtered frames
            poses = np.array([frame['transform_matrix'] for frame in pre_filtered_frames])
            sampling_pool_frame_indices = np.array(pre_filtered_indices)
            print(f"  Using GT poses: {poses.shape[0]} frames")
        
        # JJ: Use Block 2 sampling function from pa_sampling
        start_time = time.time()
        efficient_indices = pa_sampling.run_efficient_sampling(
            poses=poses,
            num_samples=efficient_hyper_setting_dict['num_samples'],
            sampling_mode=efficient_hyper_setting_dict.get('sampling_mode', 'hybrid'),
            normalization=efficient_hyper_setting_dict.get('normalization', 'std_norm'),
            diagonal_priority=efficient_hyper_setting_dict.get('diagonal_priority', 0.3),
            starting_mode=efficient_hyper_setting_dict.get('starting_mode', 'farthest'),
            skip_first_frame=efficient_hyper_setting_dict.get('skip_first_frame', False),
            reorth_rot=efficient_hyper_setting_dict.get('reorth_rot', True),
            grid_density=efficient_hyper_setting_dict.get('grid_density', 1.5),
            verbose=False
        )
        sampling_time_ms = (time.time() - start_time) * 1000
        
        # Map back to original frame indices
        final_indices = [int(sampling_pool_frame_indices[i]) for i in efficient_indices]
        
        print(f"Efficient sampling selected {len(final_indices)} frames from {len(poses)} poses")
        print(f"  ⏱ Sampling time: {sampling_time_ms:.2f} ms")
        
    elif pose_aware_strategy == 'sa':
        # Apply Space-Aware Sampling using VGGT predictions
        print(f"Applying Space-Aware Sampling with settings: {sa_hyper_setting_dict}")
        
        # Validate predictions_path
        predictions_path = sa_hyper_setting_dict.get('predictions_path')
        if predictions_path is None:
            raise ValueError(
                "SA sampling requires 'predictions_path' in sa_hyper_setting_dict. "
                "Please provide path to pre-computed VGGT predictions (.pt file)."
            )
        
        # Load VGGT predictions
        import torch
        print(f"  Loading VGGT predictions from: {predictions_path}")
        predictions = torch.load(predictions_path, weights_only=False)
        
        # Get number of frames in predictions
        T_pred = predictions['world_points'].shape[1]
        print(f"  Predictions contain {T_pred} frames")
        
        # Validate: predictions should correspond to pre-filtered frames
        if T_pred != len(pre_filtered_indices):
            print(f"  Warning: Predictions have {T_pred} frames but {len(pre_filtered_indices)} pre-filtered frames")
            print(f"  Assuming predictions correspond to first {T_pred} pre-filtered frames")
        
        # Time the core sampling function
        start_time = time.time()
        sa_indices = flexiable_sa_sampling.sa_sampling(
            predictions=predictions,
            num_samples=sa_hyper_setting_dict['num_samples'],
            voxel_density=sa_hyper_setting_dict.get('voxel_density', 20.0),
            conf_percentile=sa_hyper_setting_dict.get('conf_percentile', 50.0),
            conf_threshold=sa_hyper_setting_dict.get('conf_threshold', 0.1),
            verbose=True
        )
        sampling_time_ms = (time.time() - start_time) * 1000
        
        # Map back to original frame indices
        # sa_indices are in range [0, T_pred-1], map to pre_filtered_indices
        final_indices = [pre_filtered_indices[i] for i in sa_indices]
        
        print(f"SA sampling selected {len(final_indices)} frames from {T_pred} frames in predictions")
        print(f"  ⏱ Sampling time: {sampling_time_ms:.2f} ms")
        
    else:
        raise NotImplementedError(f"pose_aware_strategy '{pose_aware_strategy}' not implemented")
    
    # check the len of final_indices; print the frist 10 indices
    print(f"final_indices: {len(final_indices)}")
    print(f"first 10 indices: {final_indices[:10]}")
#     # JJ HACK : replace with SA indices (make sure prefilter is all for fairness)

    # Step 2.5: Generate sampling quality visualization (if applicable)
    if visualize_sampling and pose_aware_strategy in ['fps', 'uniform', 'efficient', 'sa', 'None'] and visualize_sampled_pose is not None:
        print(f"\nGenerating sampling quality visualization...")
        
        # For SA sampling, handle vis pose source
        if pose_aware_strategy == 'sa':
            sa_vis_pose_source = sa_hyper_setting_dict.get('vis_pose_source', 'gt')
            
            if sa_vis_pose_source == 'gt':
                # Load ALL GT poses (244) for visualization
                print(f"  Loading GT poses for visualization...")
                total_frames_gt = len(frames)
                
                # Load ALL GT pose matrices (not subsampled)
                all_gt_pose_matrices = [frame['transform_matrix'] for frame in frames]
                all_poses = np.array(all_gt_pose_matrices)
                print(f"  Loaded {len(all_poses)} GT poses (all original frames)")
                
                # JJ: Compute frame indices used for VGGT input (for mapping selected indices)
                # Reuses the same linspace logic from sa_sampling.py:extract_initial_frames_from_video (line 537)
                frame_indices_for_vggt = flexiable_sa_sampling.compute_frame_indices_for_vggt(
                    total_frames=total_frames_gt,
                    sample_count=T_pred
                )
                
                # Map sa_indices (in prediction space [0, T_pred-1]) to original frame indices
                # sa_indices are indices in the 128-frame pool, we need to map to original 244 frames
                selected_pre_filtered_indices = [int(frame_indices_for_vggt[i]) for i in sa_indices]
                print(f"  Mapped {len(sa_indices)} selected indices from prediction space to original frame indices")
                print(f"  Selected original frame indices: {selected_pre_filtered_indices[:10]}..." if len(selected_pre_filtered_indices) > 10 else f"  Selected original frame indices: {selected_pre_filtered_indices}")
                
                pose_source_label = "Ground Truth"
            elif sa_vis_pose_source == 'vggt_max_128':
                # Use VGGT predicted poses (uniformly sampled, max 128 frames)
                all_poses = flexiable_sa_sampling.extract_poses_from_predictions(predictions)
                selected_pre_filtered_indices = sa_indices
                pose_source_label = "VGGT Predicted (max 128)"
                
                # JJ: Sanity check shape
                print(f"  ✓ Sanity check: VGGT poses shape = {all_poses.shape}")
                if all_poses.shape[0] != T_pred:
                    raise ValueError(
                        f"Shape mismatch: extracted {all_poses.shape[0]} poses but predictions has {T_pred} frames"
                    )
                if all_poses.shape[0] > 128:
                    print(f"  ⚠️  Warning: VGGT poses have {all_poses.shape[0]} frames (expected max 128)")
            else:
                raise ValueError(f"Invalid sa_vis_pose_source: {sa_vis_pose_source}. Must be 'gt' or 'vggt_max_128'")
            
            distance_mode = 'max_norm'
            method_name = f"SA Sampling (Space-Aware) - Vis Poses: {pose_source_label}"
            
        else:
            # JJ: For FPS/Efficient/Uniform/None strategies, handle vis_pose_source
            if vis_pose_source == 'vggt_max_128':
                # Load VGGT predictions if not already loaded
                if vggt_predictions is None:
                    if predictions_path is None:
                        raise ValueError(
                            f"vis_pose_source='vggt_max_128' requires --predictions_path"
                        )
                    import torch
                    print(f"  Loading VGGT predictions for visualization from: {predictions_path}")
                    vggt_predictions = torch.load(predictions_path, weights_only=False)
                
                # Extract VGGT predicted poses
                all_poses = flexiable_sa_sampling.extract_poses_from_predictions(vggt_predictions)
                T_vggt_vis = all_poses.shape[0]
                pose_source_label = f"VGGT Predicted (max 128)"
                
                # Sanity check shape
                print(f"  ✓ Sanity check: VGGT poses for visualization shape = {all_poses.shape}")
                if T_vggt_vis > 128:
                    print(f"  ⚠️  Warning: VGGT poses have {T_vggt_vis} frames (expected max 128)")
                
                # Map final_indices to VGGT prediction space
                # final_indices are in original frame space, need to map to VGGT indices [0, 127]
                vggt_frame_indices_vis = flexiable_sa_sampling.compute_frame_indices_for_vggt(
                    total_frames=len(frames),
                    sample_count=T_vggt_vis
                )
                
                # Find which VGGT indices correspond to final_indices
                selected_pre_filtered_indices = []
                for final_idx in final_indices:
                    # Find the index in vggt_frame_indices_vis that matches final_idx
                    try:
                        vggt_idx = list(vggt_frame_indices_vis).index(final_idx)
                        selected_pre_filtered_indices.append(vggt_idx)
                    except ValueError:
                        raise ValueError(
                            f"Frame {final_idx} not found in VGGT predictions.\n"
                            f"This should not happen if sample_pose_source == vis_pose_source.\n"
                            f"VGGT frame indices: {list(vggt_frame_indices_vis)[:10]}..."
                        )
            else:  # 'gt'
                # Extract GT poses from pre-filtered frames
                all_poses = np.array([frame['transform_matrix'] for frame in pre_filtered_frames])
                pose_source_label = "Ground Truth"
                
                # Map final_indices back to pre-filtered indices
                selected_pre_filtered_indices = []
                for final_idx in final_indices:
                    pre_filtered_idx = pre_filtered_indices.index(final_idx)
                    selected_pre_filtered_indices.append(pre_filtered_idx)
            
            # Determine method name and distance mode
            if pose_aware_strategy == 'fps':
                distance_mode = pa_fps_hyper_setting_dict.get('distance_mode', 'max_norm')
                method_name = f"FPS ({distance_mode}) - Vis Poses: {pose_source_label}"
            elif pose_aware_strategy == 'uniform':
                distance_mode = 'max_norm'
                method_name = f"Uniform Sampling - Vis Poses: {pose_source_label}"
            elif pose_aware_strategy == 'efficient':
                distance_mode = 'max_norm'
                sampling_mode = efficient_hyper_setting_dict.get('sampling_mode', 'hybrid')
                method_name = f"Efficient Sampling ({sampling_mode}) - Vis Poses: {pose_source_label}"
            else:  # None
                distance_mode = 'max_norm'
                method_name = f"No Sampling (All Frames) - Vis Poses: {pose_source_label}"
        
        # Compute distance matrix
        distance_matrix = pose_fps_sampling.compute_pairwise_pose_distance(
            all_poses, 
            distance_mode=distance_mode,
            reorth_rot=True
        )
        
        # Compute coverage statistics
        coverage = pose_fps_sampling.analyze_pose_coverage(
            all_poses,
            selected_pre_filtered_indices,
            distance_mode=distance_mode,
            reorth_rot=True
        )
        
        # JJ: Prepare data for pose_analysis based on pose_analysis_target
        if plot_pose_analysis:
            # Special handling for SA strategy with VGGT poses
            if pose_aware_strategy == 'sa' and sa_hyper_setting_dict.get('vis_pose_source', 'gt') == 'vggt_max_128':
                # For SA with VGGT poses: all_poses is 128 frames (VGGT predicted)
                # selected_pre_filtered_indices are indices within [0, 127]
                if pose_analysis_target == 'raw_frames':
                    # Cannot use raw frames when using VGGT poses
                    print("  Warning: pose_analysis_target='raw_frames' not supported with VGGT poses, using 'pre_filtered' instead")
                    pose_analysis_target = 'pre_filtered'
                
                if pose_analysis_target == 'pre_filtered':
                    # Use all 128 VGGT predicted poses
                    analysis_poses = all_poses
                    analysis_selected_indices = selected_pre_filtered_indices
                    # Frame IDs are just 0-127 (indices in VGGT prediction space)
                    analysis_frame_ids = list(range(len(all_poses)))
                elif pose_analysis_target == 'sampled':
                    # Use only sampled frames from VGGT predictions
                    sorted_indices = sorted(range(len(selected_pre_filtered_indices)), 
                                          key=lambda i: selected_pre_filtered_indices[i])
                    analysis_poses = all_poses[selected_pre_filtered_indices][sorted_indices]
                    analysis_frame_ids = [selected_pre_filtered_indices[i] for i in sorted_indices]
                    analysis_selected_indices = list(range(len(selected_pre_filtered_indices)))
                else:
                    raise ValueError(f"Invalid pose_analysis_target: {pose_analysis_target}")
            else:
                # Standard handling for other strategies (FPS/Efficient/Uniform/None)
                # JJ: Need to handle vis_pose_source='vggt_max_128' case separately
                if vis_pose_source == 'vggt_max_128':
                    # Using VGGT poses for visualization
                    if pose_analysis_target == 'raw_frames':
                        # Cannot use raw frames when using VGGT poses
                        print("  Warning: pose_analysis_target='raw_frames' not supported with VGGT poses, using 'pre_filtered' instead")
                        pose_analysis_target = 'pre_filtered'
                    
                    if pose_analysis_target == 'pre_filtered':
                        # Use all VGGT predicted poses (128 frames)
                        analysis_poses = all_poses
                        analysis_selected_indices = selected_pre_filtered_indices
                        # Frame IDs are just 0-127 (indices in VGGT prediction space)
                        analysis_frame_ids = list(range(len(all_poses)))
                    elif pose_analysis_target == 'sampled':
                        # Use only sampled frames from VGGT predictions
                        sorted_indices = sorted(range(len(selected_pre_filtered_indices)), 
                                              key=lambda i: selected_pre_filtered_indices[i])
                        analysis_poses = all_poses[selected_pre_filtered_indices][sorted_indices]
                        analysis_frame_ids = [selected_pre_filtered_indices[i] for i in sorted_indices]
                        analysis_selected_indices = list(range(len(selected_pre_filtered_indices)))
                    else:
                        raise ValueError(f"Invalid pose_analysis_target: {pose_analysis_target}")
                else:
                    # Using GT poses for visualization
                    if pose_analysis_target == 'raw_frames':
                        # Use all raw frames (before filtering)
                        raw_poses = np.array([frame['transform_matrix'] for frame in frames])
                        analysis_poses = raw_poses
                        # Map final_indices to raw frame indices
                        analysis_selected_indices = final_indices
                        # Frame IDs are just 0, 1, 2, ..., len(frames)-1
                        analysis_frame_ids = list(range(len(frames)))
                    elif pose_analysis_target == 'pre_filtered':
                        # Use pre-filtered frames (current default)
                        analysis_poses = all_poses
                        analysis_selected_indices = selected_pre_filtered_indices
                        # Frame IDs are the original indices in frames
                        analysis_frame_ids = pre_filtered_indices
                    elif pose_analysis_target == 'sampled':
                        # Use only sampled frames, sorted by original frame ID for chronological order
                        # Get original frame IDs for sampled frames
                        sampled_frame_ids = [pre_filtered_indices[i] for i in selected_pre_filtered_indices]
                        
                        # Sort by frame ID to maintain temporal order (important for FPS where selection order != temporal order)
                        sorted_indices = sorted(range(len(sampled_frame_ids)), key=lambda i: sampled_frame_ids[i])
                        
                        # Reorder poses and frame IDs according to temporal order
                        analysis_poses = all_poses[selected_pre_filtered_indices][sorted_indices]
                        analysis_frame_ids = [sampled_frame_ids[i] for i in sorted_indices]
                        
                        # All frames are "selected" since we only show sampled frames
                        analysis_selected_indices = list(range(len(selected_pre_filtered_indices)))
                    else:
                        raise ValueError(f"Invalid pose_analysis_target: {pose_analysis_target}")
        else:
            # If not plotting pose_analysis, use default (doesn't matter)
            analysis_poses = all_poses
            analysis_selected_indices = selected_pre_filtered_indices
            analysis_frame_ids = None
        
        # Generate visualization
        strategy_name = pose_aware_strategy if pose_aware_strategy != 'None' else 'none'
        vis_output_path = output_dir / f"sampling_quality_{strategy_name}.html"
        try:
            visualize_sampled_pose(
                pose_analysis_poses=all_poses,  # JJ: Always use pre_filtered for sampling_quality.html
                selected_indices=selected_pre_filtered_indices,
                min_distances=coverage['min_distances'],
                distance_matrix=distance_matrix,
                output_path=str(vis_output_path),
                show=False,
                method_name=method_name,
                plot_pose_analysis=plot_pose_analysis,
                pose_analysis_selected=analysis_selected_indices if plot_pose_analysis else None,
                pose_analysis_target_poses=analysis_poses if plot_pose_analysis else None,  # JJ: Different poses for pose_analysis.html based on target
                pose_analysis_frame_ids=analysis_frame_ids if plot_pose_analysis else None,  # JJ: Actual frame IDs for x-axis
                pose_source=pose_source_label  # JJ: Pass pose source for annotation in pose_analysis.html
            )
            # JJ: Removed duplicate print - visualize_sampled_pose already prints the path
        except Exception as e:
            print(f"  Warning: Failed to generate visualization: {e}")
    
    # Step 3: Prepare transform data for selected frames
    selected_indices = final_indices
    transform_matrix_data = []
    
    for idx in selected_indices:
        frame = frames[idx]
        
        # Extract frame id from original filename (e.g., DSC00820.JPG -> 00820)
        original_filename = frame['file_path']
        frame_id = ''.join(filter(str.isdigit, Path(original_filename).stem))
        
        # Generate new filename: {scene_name}_frame_{original_frame_id}.png
        new_filename = f"{scene_name}_frame_{frame_id}.png"
        
        transform_matrix_data.append({
            "index": idx,
            "file_path": new_filename,
            "pose_matrix": frame['transform_matrix']
        })
    
    # Copy selected images with new naming format
    for frame_info in transform_matrix_data:
        original_filename = frames[frame_info['index']]['file_path']
        src_path = images_dir / original_filename
        
        if src_path.exists():
            # Read and save as PNG with new name
            img = cv2.imread(str(src_path))
            if img is not None:
                dst_path = output_dir / frame_info['file_path']
                cv2.imwrite(str(dst_path), img)
            else:
                print(f"Warning: Failed to read image: {src_path}")
        else:
            print(f"Warning: Image not found: {src_path}")
    
    # Generate video from resized_undistorted_images
    if save_full_video:
        # Get all image files and sort them
        image_files = sorted(images_dir.glob("*.JPG")) + sorted(images_dir.glob("*.jpg")) + \
                      sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))
        
        if image_files:
            # Read first image to get dimensions
            first_img = cv2.imread(str(image_files[0]))
            h, w = first_img.shape[:2]
            
            # Calculate new size maintaining aspect ratio
            if max(h, w) > target_size:
                if h > w:
                    new_h = target_size
                    new_w = int(w * target_size / h)
                else:
                    new_w = target_size
                    new_h = int(h * target_size / w)
            else:
                new_h, new_w = h, w
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, full_video_fps, (new_w, new_h))
            
            # Write frames to video
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    video_writer.write(resized)
            
            video_writer.release()
            print(f"Full video saved to: {video_path}")
        else:
            print(f"Warning: No images found in {images_dir}")
    
    # Generate sample video from selected frames
    if construct_sample_video:
        sample_video_path = output_dir / "sample_video.mp4"
        sample_images = sorted(output_dir.glob("*.png"))
        
        if sample_images:
            # Read first sample image to get dimensions
            first_img = cv2.imread(str(sample_images[0]))
            h, w = first_img.shape[:2]
            
            # Calculate new size maintaining aspect ratio
            if max(h, w) > target_size:
                if h > w:
                    new_h = target_size
                    new_w = int(w * target_size / h)
                else:
                    new_w = target_size
                    new_h = int(h * target_size / w)
            else:
                new_h, new_w = h, w
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(sample_video_path), fourcc, sample_video_fps, (new_w, new_h))
            
            # Write frames to video
            for img_path in sample_images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    video_writer.write(resized)
            
            video_writer.release()
            print(f"Sample video saved to: {sample_video_path}")
        else:
            print(f"Warning: No sample images found in {output_dir}")
    
    # Generate selected_frames.json
    output_json = {
        "scene_name": scene_name,
        "pre_filter_strategy": pre_filter_strategy,
        "pose_aware_strategy": pose_aware_strategy,
        "pa_fps_hyper_setting": pa_fps_hyper_setting_dict if pose_aware_strategy == 'fps' else None,
        "uniform_hyper_setting": uniform_hyper_setting_dict if pose_aware_strategy == 'uniform' else None,
        "efficient_hyper_setting": efficient_hyper_setting_dict if pose_aware_strategy == 'efficient' else None,
        "sampling_time_ms": sampling_time_ms,
        "selected_frames": selected_indices,
        "num_frames": len(selected_indices),
        "total_frames": len(frames),
        "transform_matrix": transform_matrix_data
    }
    
    json_path = output_dir / "selected_frames.json"
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Processed scene {scene_name}:")
    print(f"  - Pre-filter strategy: {pre_filter_strategy}")
    print(f"  - Pose-aware strategy: {pose_aware_strategy}")
    if pose_aware_strategy == 'fps':
        print(f"  - FPS settings: {pa_fps_hyper_setting_dict}")
    elif pose_aware_strategy == 'uniform':
        print(f"  - Uniform settings: {uniform_hyper_setting_dict}")
    elif pose_aware_strategy == 'efficient':
        print(f"  - Efficient settings: {efficient_hyper_setting_dict}")
    print(f"  - Selected {len(selected_indices)} frames from {len(frames)} total")
    print(f"  - Core sampling time: {sampling_time_ms:.2f} ms")
    print(f"  - Images copied to: {output_dir}")
    print(f"  - Metadata saved to: {json_path}")
    if pose_aware_strategy == 'sa':
        print(f"  - SA settings: {sa_hyper_setting_dict}")
    print(f"{'='*80}")


def main():
    import argparse
    
    # JJ: Parse command line arguments
    parser = argparse.ArgumentParser(description='Process and sample ScanNet++ DSLR images')
    
    # Data paths
    parser.add_argument('--data_root', type=str, 
                        default='/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/scannetpp',
                        help='Root directory of ScanNet++ dataset')
    parser.add_argument('--scene_name', type=str, 
                        default='00777c41d4',
                        choices=['00777c41d4', '020312de8d'],
                        help='Scene name to process')
    
    # Filtering and sampling strategies
    parser.add_argument('--pre_filter_strategy', type=str, 
                        default='all', choices=['all', 'good'],
                        help='Pre-filtering strategy: all=use all frames, good=filter out bad poses')
    parser.add_argument('--pose_aware_strategy', type=str, 
                        default='uniform', choices=['None', 'fps', 'uniform', 'efficient', 'sa'],
                        help='Pose-aware sampling strategy')
    
    # General pose source (applies to all strategies)
    parser.add_argument('--sample_pose_source', type=str, default='gt', 
                        choices=['gt', 'vggt_max_128'],
                        help='Pose source for sampling (FPS/Efficient only): gt=Ground Truth poses, vggt_max_128=VGGT predicted poses (uniformly sampled, max 128 frames); Notice this param will not affect the sampling results of strategy flexiable SA.')
    parser.add_argument('--vis_pose_source', type=str, default='gt', 
                        choices=['gt', 'vggt_max_128'],
                        help='Pose source for visualization (all strategies): gt=Ground Truth poses, vggt_max_128=VGGT predicted poses (uniformly sampled, max 128 frames)')
    
    # Predictions path (required when using vggt_max_128)
    parser.add_argument('--predictions_path', type=str, 
                        default=None,
                        help='Path to VGGT predictions (.pt file). Required when --sample_pose_source=vggt_max_128 or --vis_pose_source=vggt_max_128')
    
    # FPS specific parameters
    parser.add_argument('--fps_distance_mode', type=str, 
                        default='max_norm', choices=['max_norm', 'data_driven'],
                        help='Distance mode for FPS sampling')
    parser.add_argument('--fps_starting_mode', type=str, 
                        default='first', choices=['first', 'rand', 'medoid'],
                        help='Starting point selection mode for FPS')
    
    # Efficient sampling specific parameters
    parser.add_argument('--efficient_sampling_mode', type=str, 
                        default='hybrid', choices=['grid', 'hybrid', 'fps_2d'],
                        help='Efficient sampling mode: grid=O(N), hybrid=O(N+m²), fps_2d=O(N·m)')
    parser.add_argument('--efficient_normalization', type=str, 
                        default='std_norm', choices=['std_norm', 'max_norm'],
                        help='Normalization mode for efficient sampling')
    parser.add_argument('--efficient_diagonal_priority', type=float, 
                        default=0.0,
                        help='Diagonal priority weight [0.0-1.0] for efficient sampling (Recommended: 0.0)')
    parser.add_argument('--efficient_starting_mode', type=str, 
                        default='farthest', choices=['farthest', 'medoid', 'first'],
                        help='Starting point selection mode for efficient sampling')
    
    # SA sampling specific parameters
    parser.add_argument('--sa_voxel_density', type=float, 
                        default=20.0,
                        help='Voxel density for SA sampling (voxels per scene dimension)')
    parser.add_argument('--sa_conf_percentile', type=float, 
                        default=50.0,
                        help='Confidence threshold percentile for SA sampling [0-100]')
    parser.add_argument('--sa_conf_threshold', type=float, 
                        default=0.1,
                        help='Minimum confidence value for SA sampling')
    
    # Sampling parameters
    parser.add_argument('--num_samples', type=int, 
                        default=20,
                        help='Number of frames to sample')
    
    # Visualization parameters
    parser.add_argument('--visualize_sampling', action='store_true', default=True,
                        help='Generate sampling_quality.html visualization')
    parser.add_argument('--no_visualize_sampling', action='store_false', dest='visualize_sampling',
                        help='Skip sampling_quality.html visualization')
    parser.add_argument('--plot_pose_analysis', action='store_true', default=True,
                        help='Generate pose_analysis.html with farness analysis')
    parser.add_argument('--no_plot_pose_analysis', action='store_false', dest='plot_pose_analysis',
                        help='Skip pose_analysis.html generation')
    parser.add_argument('--pose_analysis_target', type=str, 
                        default='pre_filtered', choices=['raw_frames', 'pre_filtered', 'sampled'],
                        help='Which frame set to analyze in pose_analysis.html')
    
    # Video generation parameters
    parser.add_argument('--save_full_video', action='store_true', default=False,
                        help='Generate full video from all images')
    parser.add_argument('--construct_sample_video', action='store_true', default=False,
                        help='Generate video from sampled images only')
    
    args = parser.parse_args()
    
    # Prepare hyperparameter dictionaries based on strategy
    if args.pose_aware_strategy == 'fps':
        pa_fps_hyper_setting_dict = {
            'num_samples': args.num_samples,
            'distance_mode': args.fps_distance_mode,
            'starting_mode': args.fps_starting_mode,
            'reorth_rot': True
        }
        uniform_hyper_setting_dict = None
        efficient_hyper_setting_dict = None
        sa_hyper_setting_dict = None
    elif args.pose_aware_strategy == 'uniform':
        pa_fps_hyper_setting_dict = None
        uniform_hyper_setting_dict = {
            'num_samples': args.num_samples
        }
        efficient_hyper_setting_dict = None
        sa_hyper_setting_dict = None
    elif args.pose_aware_strategy == 'efficient':
        pa_fps_hyper_setting_dict = None
        uniform_hyper_setting_dict = None
        efficient_hyper_setting_dict = {
            'num_samples': args.num_samples,
            'sampling_mode': args.efficient_sampling_mode,
            'normalization': args.efficient_normalization,
            'diagonal_priority': args.efficient_diagonal_priority,
            'starting_mode': args.efficient_starting_mode,
            'grid_density': 1.5,
            'reorth_rot': True
        }
        sa_hyper_setting_dict = None
    elif args.pose_aware_strategy == 'sa':
        pa_fps_hyper_setting_dict = None
        uniform_hyper_setting_dict = None
        efficient_hyper_setting_dict = None
        sa_hyper_setting_dict = {
            'num_samples': args.num_samples,
            'predictions_path': args.predictions_path,  # JJ: Use general predictions_path
            'voxel_density': args.sa_voxel_density,
            'conf_percentile': args.sa_conf_percentile,
            'conf_threshold': args.sa_conf_threshold,
            'vis_pose_source': args.vis_pose_source  # JJ: SA only uses vis_pose_source
        }
    else:  # 'None'
        pa_fps_hyper_setting_dict = None
        uniform_hyper_setting_dict = None
        efficient_hyper_setting_dict = None
        sa_hyper_setting_dict = None
    
    # Run processing
    print("\n" + "="*80)
    print(f"Processing scene: {args.scene_name}")
    print(f"Strategy: {args.pre_filter_strategy} + {args.pose_aware_strategy}")
    print("="*80)
    
    process_dslr(
        dataset_root=args.data_root,
        scene_name=args.scene_name,
        target_size=640,
        full_video_fps=30,
        sample_video_fps=10,
        pre_filter_strategy=args.pre_filter_strategy,
        pose_aware_strategy=args.pose_aware_strategy,
        sample_pose_source=args.sample_pose_source,  # JJ: Pose source for sampling
        vis_pose_source=args.vis_pose_source,  # JJ: Pose source for visualization
        predictions_path=args.predictions_path,  # JJ: General predictions path
        pa_fps_hyper_setting_dict=pa_fps_hyper_setting_dict,
        uniform_hyper_setting_dict=uniform_hyper_setting_dict,
        efficient_hyper_setting_dict=efficient_hyper_setting_dict,
        sa_hyper_setting_dict=sa_hyper_setting_dict,
        save_full_video=args.save_full_video,
        construct_sample_video=args.construct_sample_video,
        visualize_sampling=args.visualize_sampling,
        plot_pose_analysis=args.plot_pose_analysis,
        pose_analysis_target=args.pose_analysis_target,
    )
    
if __name__ == "__main__":
    main()