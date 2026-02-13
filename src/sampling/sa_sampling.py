import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

import json
import shutil
import tempfile
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch import cuda
from torchvision import transforms as TF
import subprocess

try:
    from decord import VideoReader, cpu  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    VideoReader = None
    cpu = None
    
from src.qwenvl.external.vggt.models.vggt import VGGT  # type: ignore
from src.qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore


def compute_voxel_sets(world_points, world_points_conf_mask, x_min, y_min, z_min, voxel_size):
    """
    Compute the voxel set covered by each frame.

    Args:
        world_points: Tensor of shape (1, T, H, W, 3) containing 3D coordinates.
        world_points_conf_mask: Boolean tensor of shape (1, T, H, W) indicating valid points.
        x_min, y_min, z_min: Minimum scene coordinates.
        voxel_size: Size of each voxel.

    Returns:
        List[Set[Tuple[int]]]: Voxel coordinate set for each frame.
    """
    device = world_points.device
    T = world_points.shape[1]
    voxel_sets = []
    
    # Ensure coordinate parameters are tensors on the correct device
    x_min = torch.tensor(x_min, device=device) if not isinstance(x_min, torch.Tensor) else x_min.to(device)
    y_min = torch.tensor(y_min, device=device) if not isinstance(y_min, torch.Tensor) else y_min.to(device)
    z_min = torch.tensor(z_min, device=device) if not isinstance(z_min, torch.Tensor) else z_min.to(device)
    voxel_size = torch.tensor(voxel_size, device=device) if not isinstance(voxel_size, torch.Tensor) else voxel_size.to(device)

    for t in range(T):
        # Retrieve valid points for the current frame
        mask = world_points_conf_mask[0, t].flatten()  # (H*W,)
        points = world_points[0, t].reshape(-1, 3)    # (H*W, 3)
        valid_points = points[mask]  # (N_valid, 3)
        
        if valid_points.size(0) == 0:
            voxel_sets.append(set())
            continue
            
        # Compute voxel coordinates
        offset = torch.tensor([x_min, y_min, z_min], device=device)
        voxel_coords = ((valid_points - offset) / voxel_size).floor().long()
        
        # Remove duplicates and convert to a CPU set
        unique_voxels = torch.unique(voxel_coords, dim=0)
        voxel_set = set(map(tuple, unique_voxels.cpu().numpy().tolist()))
        
        voxel_sets.append(voxel_set)
    
    return voxel_sets

def maximum_coverage_sampling(voxel_sets, K):
    """
    Greedy maximum-coverage sampling.

    Args:
        voxel_sets: List of voxel sets for each frame.
        K: Maximum number of frames to select.

    Returns:
        List[int]: Indices of selected frames.
    """
    selected = []
    covered = set()
    remaining_frames = set(range(len(voxel_sets)))
    
    for _ in range(K):
        if not remaining_frames:
            break
            
        max_gain = -1
        best_frame = None
        
        # Find the frame with the maximum marginal gain
        for frame in remaining_frames:
            gain = len(voxel_sets[frame] - covered)
            if gain > max_gain:
                max_gain = gain
                best_frame = frame
                
        if best_frame is None or max_gain <= 0:
            break  # No more new coverage
            
        selected.append(best_frame)
        covered.update(voxel_sets[best_frame])
        remaining_frames.remove(best_frame)
    
    return selected

# JJ : Run VGGT inference and return predictions dict with extrinsic/intrinsic
def run_vggt_inference(vggt, images, dtype):
    """
    Run VGGT model inference and compute extrinsic/intrinsic matrices.

    Args:
        vggt: Pretrained VGGT model.
        images: Tensor of shape (T, 3, H, W) representing the video frames.
        dtype: Data type for mixed precision inference.
    Returns:
        dict: Predictions dict with all VGGT outputs plus extrinsic/intrinsic.
    """
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            predictions = vggt(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    return predictions


def space_aware_frame_sampling(vggt, images, K, dtype):
    """
    Perform space-aware frame sampling on a video tensor.

    Args:
        vggt: Pretrained VGGT model.
        images: Tensor of shape (T, 3, H, W) representing the video frames.
        K: Number of frames to sample.
    Returns:
        List[int]: Indices of selected frames.
    """
    # JJ : Reuse run_vggt_inference helper
    predictions = run_vggt_inference(vggt, images, dtype)


    world_points = predictions['world_points']  # shape (1, T, H, W, 3)
    world_points_flat = world_points.reshape(-1, 3)  # shape (B, 3)
    world_points_conf = predictions['world_points_conf']  # shape (1, T, H, W)
    world_points_conf_flat = world_points_conf.reshape(-1)  # shape (B)

    init_threshold_val = np.percentile(world_points_conf_flat.cpu().numpy(), 50)
    world_points_conf_mask = (world_points_conf >= init_threshold_val) & (world_points_conf > 0.1)
    world_points_conf_flat_mask = (world_points_conf_flat >= init_threshold_val) & (world_points_conf_flat > 0.1)

    # get bounding box of world_points
    x_min, y_min, z_min = world_points_flat[world_points_conf_flat_mask].min(dim=0)[0]
    x_max, y_max, z_max = world_points_flat[world_points_conf_flat_mask].max(dim=0)[0]
    # print(x_min, y_min, z_min, x_max, y_max, z_max)

    voxel_size = min(x_max - x_min, y_max - y_min, z_max - z_min) / 20
    # print(voxel_size)

    voxel_sets = compute_voxel_sets(
        world_points=world_points,
        world_points_conf_mask=world_points_conf_mask,
        x_min=x_min.item(),
        y_min=y_min.item(),
        z_min=z_min.item(),
        voxel_size=voxel_size.item()
    )

    selected_frames = sorted(maximum_coverage_sampling(voxel_sets, K))

    return selected_frames, predictions


# JJ : Temporal merge aware sampling helper
def add_neighbor_frames(sampled_indices, neighbor_mode, step_size, total_frames, raise_on_overlap=True):
    """
    Add neighbor frames to sampled indices based on temporal merge aware strategy.
    
    Special boundary handling:
    - First frame: always uses 'after' mode (to avoid negative indices)
    - Last frame: always uses 'before' mode (to avoid exceeding total_frames)
    - Middle frames: use the specified neighbor_mode
    
    Args:
        sampled_indices: List of initially sampled frame indices (sorted, incremental).
        neighbor_mode: 'before', 'after', or 'random' (applies to middle frames).
        step_size: Step size for neighbor frame offset.
        total_frames: Total number of frames in the video/pool.
        raise_on_overlap: Deprecated, always raises on overlap.
    
    Returns:
        List[int]: Final frame indices (sorted, incremental) with neighbors added.
    
    Raises:
        ValueError: If neighbor frame is out of bounds or overlaps with adjacent frames.
    """
    import random
    
    final_indices = []
    sampled_indices = sorted(sampled_indices)
    num_sampled = len(sampled_indices)
    
    for i, frame_idx in enumerate(sampled_indices):
        # Determine neighbor position with boundary handling
        is_first = (i == 0)
        is_last = (i == num_sampled - 1)
        
        if is_first:
            # First frame: force 'after' to avoid negative indices
            mode = "after"
        elif is_last:
            # Last frame: force 'before' to avoid exceeding total_frames
            mode = "before"
        else:
            # Middle frames: use specified neighbor_mode
            if neighbor_mode == "random":
                mode = random.choice(["before", "after"])
            else:
                mode = neighbor_mode
        
        # Calculate neighbor frame index
        if mode == "before":
            neighbor_idx = frame_idx - step_size
            
            # Check overflow: out of bounds or overlaps with previous sampled frame
            if neighbor_idx < 0:
                raise ValueError(f"Neighbor frame {neighbor_idx} < 0 for frame {frame_idx}. "
                               f"Cannot add neighbor with step_size={step_size} in 'before' mode.")
            elif i > 0 and neighbor_idx <= sampled_indices[i-1]:
                raise ValueError(f"Neighbor frame {neighbor_idx} overlaps with previous frame {sampled_indices[i-1]} "
                               f"for frame {frame_idx}. Increase frame spacing or decrease step_size.")
            
            # Add in order: [neighbor, frame]
            final_indices.extend([neighbor_idx, frame_idx])
        
        else:  # mode == "after"
            neighbor_idx = frame_idx + step_size
            
            # Check overflow: out of bounds or overlaps with next sampled frame
            if neighbor_idx >= total_frames:
                raise ValueError(f"Neighbor frame {neighbor_idx} >= total_frames {total_frames} for frame {frame_idx}. "
                               f"Cannot add neighbor with step_size={step_size} in 'after' mode.")
            elif i < len(sampled_indices) - 1 and neighbor_idx >= sampled_indices[i+1]:
                raise ValueError(f"Neighbor frame {neighbor_idx} overlaps with next frame {sampled_indices[i+1]} "
                               f"for frame {frame_idx}. Increase frame spacing or decrease step_size.")
            
            # Add in order: [frame, neighbor]
            final_indices.extend([frame_idx, neighbor_idx])
    
    # Sort and ensure incremental (should already be incremental by construction)
    # Convert all to Python int to avoid numpy.int64 serialization issues
    final_indices = sorted([int(x) for x in final_indices])
    return final_indices


def load_and_preprocess_images(image_path_list):
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    for image_path in image_path_list:
        img = Image.open(image_path)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        width, height = img.size
        if height > width:
            img = img.rotate(-90, expand=True)

        width, height = img.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        print(f"Warning: Images have varying shapes {shapes}, padding to the largest size.")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)

    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images


def create_video_from_frames(frame_dir, output_video_path, fps=1):
    """
    Create a video from frames in a directory using ffmpeg.
    
    Args:
        frame_dir: Directory containing frame images.
        output_video_path: Path to save the output video.
        fps: Frames per second for the output video.
    """
    frame_files = sorted(glob(os.path.join(frame_dir, "*.png")))
    if not frame_files:
        print(f"No frames found in {frame_dir}, skipping video creation.")
        return
    
    # Create a temporary file list for ffmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        file_list_path = f.name
        for frame_file in frame_files:
            f.write(f"file '{os.path.abspath(frame_file)}'\n")
    
    try:
        # Try multiple encoders in order of preference
        encoders = [
            ('libopenh264', 'yuv420p'),  # Open H.264 (non-GPL)
            ('mpeg4', 'yuv420p'),         # MPEG-4 (widely compatible)
            ('libvpx-vp9', 'yuv420p'),    # VP9 (modern, efficient)
        ]
        
        video_created = False
        for encoder, pix_fmt in encoders:
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-r', str(fps),
                '-i', file_list_path,
                '-c:v', encoder,
                '-pix_fmt', pix_fmt,
                '-y',  # Overwrite output file if exists
                output_video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Video saved to {output_video_path} using {encoder} encoder")
                video_created = True
                break
        
        if not video_created:
            print(f"Error creating video: Could not find a compatible encoder")
            print(f"Last error: {result.stderr}")
    finally:
        # Clean up the temporary file list
        if os.path.exists(file_list_path):
            os.remove(file_list_path)


# JJ : Helper functions for process_videos_on_device refactoring
def check_video_completion(video_name, args):
    """
    Check if a video has already been processed completely.
    
    Returns:
        tuple: (skip_video, anomalies) where skip_video is bool and anomalies is list of str
    """
    skip_video = True
    anomalies = []
    
    # Check space-aware sampling
    if args.sampling_type in ["both", "sa"]:
        sa_dir = os.path.join(args.output_folder, video_name)
        metadata_path = os.path.join(sa_dir, "selected_frames.json")
        
        if os.path.exists(sa_dir):
            png_files = glob(os.path.join(sa_dir, "*.png"))
            png_count = len(png_files)
            
            metadata_valid = False
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get("num_frames") == args.num_frames:
                        metadata_valid = True
                except (json.JSONDecodeError, KeyError):
                    pass
            
            if png_count != args.num_frames:
                anomalies.append(f"SA: PNG count {png_count}/{args.num_frames}")
                skip_video = False
            elif not metadata_valid:
                anomalies.append(f"SA: metadata invalid")
                skip_video = False
            
            if args.save_extra and not os.path.exists(os.path.join(sa_dir, "sa_predictions.pt")):
                anomalies.append(f"SA: predictions.pt missing")
                skip_video = False
        else:
            anomalies.append(f"SA: dir not found")
            skip_video = False
    
    # Check uniform sampling
    if args.sampling_type in ["both", "uniform"]:
        uniform_dir = os.path.join(args.output_folder, video_name)
        
        if os.path.exists(uniform_dir):
            png_files = glob(os.path.join(uniform_dir, "*.png"))
            png_count = len(png_files)
            if png_count != args.num_frames:
                anomalies.append(f"Uniform: PNG count {png_count}/{args.num_frames}")
                skip_video = False
            
            if args.save_extra and not os.path.exists(os.path.join(uniform_dir, "uniform_predictions.pt")):
                anomalies.append(f"Uniform: predictions.pt missing")
                skip_video = False
        else:
            anomalies.append(f"Uniform: dir not found")
            skip_video = False
    
    # Check temporal merge aware uniform sampling
    if args.sampling_type == "mergeaware_uniform":
        uniform_dir = os.path.join(args.output_folder, video_name)
        metadata_path = os.path.join(uniform_dir, "selected_frames.json")
        
        if os.path.exists(uniform_dir):
            png_files = glob(os.path.join(uniform_dir, "*.png"))
            png_count = len(png_files)
            if png_count != args.num_frames:
                anomalies.append(f"TempMergeUniform: PNG count {png_count}/{args.num_frames}")
                skip_video = False
            
            metadata_valid = False
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get("num_frames") == args.num_frames:
                        metadata_valid = True
                except (json.JSONDecodeError, KeyError):
                    pass
            
            if not metadata_valid:
                anomalies.append(f"TempMergeUniform: metadata invalid")
                skip_video = False
            
            if args.save_extra and not os.path.exists(os.path.join(uniform_dir, "uniform_predictions.pt")):
                anomalies.append(f"TempMergeUniform: predictions.pt missing")
                skip_video = False
        else:
            anomalies.append(f"TempMergeUniform: dir not found")
            skip_video = False
    
    # Check temporal merge aware SA sampling
    if args.sampling_type == "mergeaware_sa":
        sa_dir = os.path.join(args.output_folder, video_name)
        metadata_path = os.path.join(sa_dir, "selected_frames.json")
        
        if os.path.exists(sa_dir):
            png_files = glob(os.path.join(sa_dir, "*.png"))
            png_count = len(png_files)
            if png_count != args.num_frames:
                anomalies.append(f"TempMergeSA: PNG count {png_count}/{args.num_frames}")
                skip_video = False
            
            metadata_valid = False
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get("num_frames") == args.num_frames:
                        metadata_valid = True
                except (json.JSONDecodeError, KeyError):
                    pass
            
            if not metadata_valid:
                anomalies.append(f"TempMergeSA: metadata invalid")
                skip_video = False
            
            if args.save_extra and not os.path.exists(os.path.join(sa_dir, "sa_predictions.pt")):
                anomalies.append(f"TempMergeSA: predictions.pt missing")
                skip_video = False
        else:
            anomalies.append(f"TempMergeSA: dir not found")
            skip_video = False
    
    return skip_video, anomalies


def extract_initial_frames_from_video(vr, tmp_dir, sample_count=128):
    """
    Extract initial frames uniformly from video.
    
    Returns:
        tuple: (frame_indices, num_frames) where frame_indices is the array of extracted frame indices
    """
    num_frames = len(vr)
    if num_frames == 0:
        raise ValueError(f"No frames found in video")
    
    sample_count = min(sample_count, num_frames)
    frame_indices = np.linspace(0, num_frames - 1, num=sample_count, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        frame = vr[frame_idx].asnumpy()
        image = Image.fromarray(frame)
        frame_path = tmp_dir / f"frame_{frame_idx:04d}.png"
        image.save(frame_path)
    
    return frame_indices, num_frames


def process_sa_sampling(device_id, video_name, tmp_dir, frame_indices, model, device, dtype, args):
    """Process standard SA sampling."""
    frame_image_paths = sorted(glob(str(tmp_dir / "*.png")))
    images = load_and_preprocess_images(frame_image_paths).to(device, dtype=dtype)
    K = min(args.num_frames, images.shape[0])
    selected_frames, predictions = space_aware_frame_sampling(model, images, K, dtype)
    print(f"[GPU {device_id}] Selected frames: {selected_frames}")

    selected_original_indices = [int(frame_indices[idx]) for idx in selected_frames]

    sa_dir = os.path.join(args.output_folder, video_name)
    os.makedirs(sa_dir, exist_ok=True)
    saved_count = 0
    for orig_idx in selected_original_indices:
        src_path = tmp_dir / f"frame_{orig_idx:04d}.png"
        if not src_path.exists():
            raise FileNotFoundError(
                f"Frame {orig_idx} not found at {src_path}. "
                f"Should exist in 128-frame pool for SA sampling."
            )
        dst_name = f"{video_name}_frame_{orig_idx:06d}.png"
        dst_path = os.path.join(sa_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        saved_count += 1

    print(f"[GPU {device_id}] Saved {saved_count} selected frames to {sa_dir}")
    
    # Save metadata
    metadata = {
        "scene_name": video_name,
        "selected_frames": [int(x) for x in selected_original_indices],
        "selected_prediction_indices": [int(idx) for idx in selected_frames],
        "num_frames": len(selected_original_indices)
    }
    metadata_path = os.path.join(sa_dir, f"selected_frames.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save predictions if requested
    if args.save_extra:
        sa_predictions_filtered = {k: predictions[k] for k in args.extra_list if k in predictions}
        sa_predictions_path = os.path.join(sa_dir, f"sa_predictions.pt")
        torch.save(sa_predictions_filtered, sa_predictions_path)
        print(f"[GPU {device_id}] Saved SA predictions ({list(sa_predictions_filtered.keys())}) to {sa_predictions_path}")

    # Create video if requested
    if args.save_video:
        sa_video_path = os.path.join(sa_dir, f"{video_name}_sa_sampling.mp4")
        create_video_from_frames(sa_dir, sa_video_path, fps=1)


def process_mergeaware_sa(device_id, video_name, tmp_dir, frame_indices, model, device, dtype, args):
    """Process temporal merge aware SA sampling."""
    frame_image_paths = sorted(glob(str(tmp_dir / "*.png")))
    images = load_and_preprocess_images(frame_image_paths).to(device, dtype=dtype)
    
    # First do SA sampling with n_prime = n // 2
    n_prime = args.num_frames // 2
    K = min(n_prime, images.shape[0])
    initial_selected_frames, predictions = space_aware_frame_sampling(model, images, K, dtype)
    print(f"[GPU {device_id}] Initial SA selected frame indices in 128-pool: {initial_selected_frames}")
    
    # Add neighbor frames in index space (0-127)
    selected_pool_indices = add_neighbor_frames(
        initial_selected_frames,
        args.neighbor_mode,
        args.index_step_size,
        len(frame_indices)
    )
    
    print(f"[GPU {device_id}] Temporal merge aware SA: {len(initial_selected_frames)} initial pool indices -> {len(selected_pool_indices)} final pool indices")
    print(f"[GPU {device_id}] Final pool indices: {selected_pool_indices}")
    
    # Map pool indices back to original video frame IDs
    selected_original_indices = [int(frame_indices[idx]) for idx in selected_pool_indices]
    initial_selected_original = [int(frame_indices[idx]) for idx in initial_selected_frames]
    
    print(f"[GPU {device_id}] Mapped to original frame IDs: {selected_original_indices}")

    sa_dir = os.path.join(args.output_folder, video_name)
    os.makedirs(sa_dir, exist_ok=True)
    
    # Save all frames
    saved_count = 0
    for orig_idx in selected_original_indices:
        frame_path = tmp_dir / f"frame_{orig_idx:04d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(
                f"Frame {orig_idx} not found at {frame_path}. "
                f"All frames should exist in 128-frame pool for mergeaware_sa mode. "
                f"Pool indices: {selected_pool_indices}, Frame IDs: {selected_original_indices}"
            )
        
        dst_name = f"{video_name}_frame_{orig_idx:06d}.png"
        dst_path = os.path.join(sa_dir, dst_name)
        shutil.copy2(frame_path, dst_path)
        saved_count += 1

    print(f"[GPU {device_id}] Saved {saved_count} temporal merge aware SA sampled frames to {sa_dir}")
    
    # Save metadata
    metadata = {
        "scene_name": video_name,
        "selected_frames": [int(x) for x in selected_original_indices],
        "initial_selected_frames": [int(x) for x in initial_selected_original],
        "selected_pool_indices": [int(idx) for idx in selected_pool_indices],
        "initial_pool_indices": [int(idx) for idx in initial_selected_frames],
        "num_frames": len(selected_original_indices),
        "sampling_type": "mergeaware_sa",
        "neighbor_mode": args.neighbor_mode,
        "index_step_size": args.index_step_size
    }
    metadata_path = os.path.join(sa_dir, f"selected_frames.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save predictions if requested
    if args.save_extra:
        sa_predictions_filtered = {k: predictions[k] for k in args.extra_list if k in predictions}
        sa_predictions_path = os.path.join(sa_dir, f"sa_predictions.pt")
        torch.save(sa_predictions_filtered, sa_predictions_path)
        print(f"[GPU {device_id}] Saved SA predictions ({list(sa_predictions_filtered.keys())}) to {sa_predictions_path}")
    
    # Create video if requested
    if args.save_video:
        sa_video_path = os.path.join(sa_dir, f"{video_name}_mergeaware_sa_sampling.mp4")
        create_video_from_frames(sa_dir, sa_video_path, fps=1)


def process_uniform_sampling(device_id, video_name, tmp_dir, frame_indices, args):
    """Process standard uniform sampling."""
    uniform_dir = os.path.join(args.output_folder, video_name)
    os.makedirs(uniform_dir, exist_ok=True)
    if len(frame_indices) <= args.num_frames:
        sampled_indices = frame_indices
    else:
        sampled_indices = np.linspace(0, len(frame_indices) - 1, num=args.num_frames, dtype=int)
        sampled_indices = frame_indices[sampled_indices]
    
    saved_count = 0
    for orig_idx in sampled_indices:
        src_path = tmp_dir / f"frame_{orig_idx:04d}.png"
        if not src_path.exists():
            raise FileNotFoundError(
                f"Frame {orig_idx} not found at {src_path}. "
                f"Should exist in 128-frame pool for uniform sampling."
            )
        dst_name = f"{video_name}_frame_{orig_idx:06d}.png"
        dst_path = os.path.join(uniform_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        saved_count += 1

    print(f"[GPU {device_id}] Saved {saved_count} uniform sampled frames to {uniform_dir}")


def process_mergeaware_uniform(device_id, video_name, tmp_dir, frame_indices, vr, num_frames, model, device, dtype, args):
    """Process temporal merge aware uniform sampling."""
    uniform_dir = os.path.join(args.output_folder, video_name)
    os.makedirs(uniform_dir, exist_ok=True)
    
    # First do uniform sampling with n_prime = n // 2
    n_prime = args.num_frames // 2
    if len(frame_indices) <= n_prime:
        initial_sampled = frame_indices
    else:
        initial_sampled_local = np.linspace(0, len(frame_indices) - 1, num=n_prime, dtype=int)
        initial_sampled = frame_indices[initial_sampled_local]
    
    # Add neighbor frames in frame ID space
    sampled_indices = add_neighbor_frames(
        initial_sampled, 
        args.neighbor_mode, 
        args.fid_step_size, 
        num_frames
    )
    
    print(f"[GPU {device_id}] Temporal merge aware uniform: {n_prime} initial frames -> {len(sampled_indices)} final frames")
    print(f"[GPU {device_id}] Final sampled indices (frame IDs): {sampled_indices}")
    
    # Extract all needed frames from video
    print(f"[GPU {device_id}] Extracting {len(sampled_indices)} frames from video...")
    for orig_idx in sampled_indices:
        frame_path = tmp_dir / f"frame_{orig_idx:04d}.png"
        if not frame_path.exists():
            frame = vr[orig_idx].asnumpy()
            image = Image.fromarray(frame)
            image.save(frame_path)
    
    # Copy all frames to output directory
    saved_count = 0
    for orig_idx in sampled_indices:
        frame_path = tmp_dir / f"frame_{orig_idx:04d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame {orig_idx} not found at {frame_path} after extraction.")
        
        dst_name = f"{video_name}_frame_{orig_idx:06d}.png"
        dst_path = os.path.join(uniform_dir, dst_name)
        shutil.copy2(frame_path, dst_path)
        saved_count += 1

    print(f"[GPU {device_id}] Saved {saved_count} temporal merge aware uniform sampled frames to {uniform_dir}")
    
    # Save metadata
    metadata = {
        "scene_name": video_name,
        "selected_frames": [int(x) for x in sampled_indices],
        "initial_selected_frames": [int(x) for x in initial_sampled],
        "num_frames": len(sampled_indices),
        "sampling_type": "mergeaware_uniform",
        "neighbor_mode": args.neighbor_mode,
        "fid_step_size": args.fid_step_size
    }
    metadata_path = os.path.join(uniform_dir, f"selected_frames.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Run VGGT inference and save predictions if requested
    if args.save_extra:
        uniform_frame_paths = sorted(glob(os.path.join(uniform_dir, "*.png")))
        uniform_images = load_and_preprocess_images(uniform_frame_paths).to(device, dtype=dtype)
        uniform_predictions = run_vggt_inference(model, uniform_images, dtype)
        uniform_predictions_filtered = {k: uniform_predictions[k] for k in args.extra_list if k in uniform_predictions}
        uniform_predictions_path = os.path.join(uniform_dir, f"uniform_predictions.pt")
        torch.save(uniform_predictions_filtered, uniform_predictions_path)
        print(f"[GPU {device_id}] Saved uniform predictions ({list(uniform_predictions_filtered.keys())}) to {uniform_predictions_path}")

    # Create video if requested
    if args.save_video:
        uniform_video_path = os.path.join(uniform_dir, f"{video_name}_mergeaware_uniform_sampling.mp4")
        create_video_from_frames(uniform_dir, uniform_video_path, fps=1)


def process_videos_on_device(device_id, video_paths, args):
    """
    Process videos on a specific GPU device.
    
    Refactored for clarity: main logic delegates to helper functions.
    """
    if not video_paths:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    # Clean up stale temporary directories
    tmp_base = tempfile.gettempdir()
    stale_dirs = glob(os.path.join(tmp_base, "sw_sampling_*"))
    if stale_dirs:
        print(f"[GPU {device_id}] Cleaning up {len(stale_dirs)} stale temporary directories...")
        for stale_dir in stale_dirs:
            try:
                if os.path.isdir(stale_dir):
                    shutil.rmtree(stale_dir)
            except Exception as e:
                print(f"[GPU {device_id}] Warning: Could not remove {stale_dir}: {e}")

    # Load model if needed
    device = "cuda"
    dtype = torch.bfloat16
    need_vggt = args.sampling_type in ["both", "sa", "mergeaware_sa"] or args.save_extra
    if not args.dry_run and need_vggt:
        model = VGGT.from_pretrained(args.model_path).to(device)
    else:
        model = None

    # Statistics tracking
    total_videos = len(video_paths)
    skipped_videos = 0
    process_videos = 0

    for video_path in tqdm(video_paths, desc=f"GPU {device_id} processing videos"):
        # Check if video has already been processed
        video_name = Path(video_path).stem
        skip_video, anomalies = check_video_completion(video_name, args)
        
        if skip_video:
            skipped_videos += 1
            continue
        else:
            process_videos += 1
            print(f"[GPU {device_id}] ✗ PROCESS: {video_name} - {', '.join(anomalies)}")
            
            # JJ : Clean up incomplete output directory to ensure fresh start
            output_dir = os.path.join(args.output_folder, video_name)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                print(f"[GPU {device_id}] Removed incomplete output: {output_dir}")
        
        # Dry run mode: skip all actual processing
        if args.dry_run:
            continue
        
        # Only anomaly videos reach here - start actual inference
        print(f"[GPU {device_id}] Starting inference for {video_name}...")
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sw_sampling_{video_name}_gpu{device_id}_"))
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            frame_indices, num_frames = extract_initial_frames_from_video(vr, tmp_dir, sample_count=128)
            print(f"[GPU {device_id}] Saved {len(frame_indices)} frames to {tmp_dir}")


            # Process sampling based on type
            if args.sampling_type in ["both", "sa"]:
                process_sa_sampling(device_id, video_name, tmp_dir, frame_indices, model, device, dtype, args)
            
            elif args.sampling_type == "mergeaware_sa":
                process_mergeaware_sa(device_id, video_name, tmp_dir, frame_indices, model, device, dtype, args)

            # Uniform sampling
            if args.sampling_type in ["both", "uniform"]:
                process_uniform_sampling(device_id, video_name, tmp_dir, frame_indices, args)
            
            # Temporal merge aware uniform sampling
            elif args.sampling_type == "mergeaware_uniform":
                process_mergeaware_uniform(device_id, video_name, tmp_dir, frame_indices, vr, num_frames, model, device, dtype, args)
        
        except Exception as e:
            print(f"[GPU {device_id}] Error processing {video_name}: {e}")
            raise
        finally:
            # Always clean up temporary directory
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"[GPU {device_id}] SUMMARY:")
    print(f"  Total videos: {total_videos}")
    print(f"  Skipped (already complete): {skipped_videos}")
    print(f"  Need processing: {process_videos}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Space-Aware Frame Sampling")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the input video folder.")
    parser.add_argument("--video_path", type=str, default=None, help="Path to the input video file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained VGGT model.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder for selected frames.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample using space-aware sampling.")
    parser.add_argument("--save_video", action="store_true", help="Save a video with fps=1 from the sampled frames.")
    parser.add_argument("--sampling_type", type=str, default="both", 
                        choices=["both", "sa", "uniform", "mergeaware_uniform", "mergeaware_sa"], 
                        help="Type of sampling to perform: 'both' (default), 'sa' (space-aware only), 'uniform' (uniform only), "
                             "'mergeaware_uniform', or 'mergeaware_sa'.")
    parser.add_argument("--save_extra", action="store_true", help="Save VGGT predictions dict as .pt alongside sampled frames.")
    # JJ : Specify which prediction keys to save
    parser.add_argument("--extra_list", type=str, nargs="+",
                        default=["depth", "depth_conf", "world_points", "world_points_conf", "extrinsic", "intrinsic"],
                        help="List of prediction keys to save when --save_extra is enabled.")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode: only check and report anomalies, no actual processing.")
    # JJ : Temporal merge aware sampling parameters
    parser.add_argument("--neighbor_mode", type=str, default="after", choices=["before", "after", "random"],
                        help="How to add neighbor frames: 'before' (frame-step), 'after' (frame+step), 'random' (randomly choose).")
    parser.add_argument("--fid_step_size", type=int, default=30,
                        help="Frame ID step size for mergeaware_uniform (operates on original video frame IDs).")
    parser.add_argument("--index_step_size", type=int, default=1,
                        help="Index step size for mergeaware_sa (operates on 128-frame pool indices).")
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()

    # Parse CUDA_VISIBLE_DEVICES to handle specific GPU selection
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
    else:
        gpu_ids = [str(i) for i in range(n_gpu)]


    if args.video_path != None:
        assert os.path.exists(args.video_path), f"Video path {args.video_path} does not exist."
        all_videos = [args.video_path]
                
        # Extract num_frames from output_folder if it contains pattern like "8f" or "16f"
        import re
        frames_match = re.search(r'(\d+)f', args.output_folder)
        if frames_match:
            num_frames_str = frames_match.group(0)  # e.g., "8f"
        else:
            num_frames_str = f"{args.num_frames}f"
        
        # JJ : Build sampling directory name with temporal merge aware parameters
        # Determine sampling type from output_folder or use args.sampling_type
        # Check more specific patterns first (mergeaware before sa/uniform)
        if "mergeaware_sa_sampling" in args.output_folder or args.sampling_type == "mergeaware_sa":
            sampling_dir = f"mergeaware_sa_sampling_{num_frames_str}"
        elif "mergeaware_uniform_sampling" in args.output_folder or args.sampling_type == "mergeaware_uniform":
            sampling_dir = f"mergeaware_uniform_sampling_{num_frames_str}"
        elif "sa_sampling" in args.output_folder:
            sampling_dir = f"sa_sampling_{num_frames_str}"
        elif "uniform_sampling" in args.output_folder:
            sampling_dir = f"uniform_sampling_{num_frames_str}"
        else:
            sampling_dir = f"{args.sampling_type}_sampling_{num_frames_str}"
        
        # Add neighbor_mode and step_size suffix for temporal merge aware modes
        if args.sampling_type == "mergeaware_uniform":
            # Abbreviate neighbor_mode: before->bef, after->aft, random->rnd
            nbr_abbrev = {"before": "bef", "after": "aft", "random": "rnd"}
            nbr_str = nbr_abbrev.get(args.neighbor_mode, args.neighbor_mode)
            sampling_dir = f"{sampling_dir}_nbr{nbr_str}_fidss{args.fid_step_size}"
        elif args.sampling_type == "mergeaware_sa":
            nbr_abbrev = {"before": "bef", "after": "aft", "random": "rnd"}
            nbr_str = nbr_abbrev.get(args.neighbor_mode, args.neighbor_mode)
            sampling_dir = f"{sampling_dir}_nbr{nbr_str}_idxss{args.index_step_size}"

        # udpate the single video output folder
        dir_sampling_dir, dataset_name = os.path.dirname(args.output_folder), os.path.basename(args.output_folder)
        dir_sampling_dir_updated = os.path.join(os.path.dirname(dir_sampling_dir), sampling_dir+'_single_video')
        args.output_folder = os.path.join(dir_sampling_dir_updated, dataset_name)
        print(f"Single video mode: Output redirected to {args.output_folder}")
    else:
        all_videos = sorted(glob(os.path.join(args.video_folder, "*.mp4")))
    print('video path', args.video_path, args.video_folder)
    if not all_videos:
        print("No videos found to process.")
        sys.exit(0)

    num_gpus = min(len(gpu_ids), len(all_videos))
    video_splits = [list(split) for split in np.array_split(all_videos, num_gpus) if len(split) > 0]

    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN MODE: Only checking for anomalies")
        print("No files will be created or modified")
        print("="*80 + "\n")

    processes = []
    for idx, video_subset in enumerate(video_splits):
        device_id = gpu_ids[idx]
        process = mp.Process(target=process_videos_on_device, args=(device_id, video_subset, args))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
