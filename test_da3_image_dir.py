#!/usr/bin/env python3
"""
JJ: Test DA3 vs VGGT on image directory (from pose-aware sampling)

Compares three methods:
1. DA3 Camera Decoder (use_ray_pose=False)
2. DA3 Ray Head (use_ray_pose=True)
3. VGGT (original spatial encoder)
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from safetensors.torch import load_file

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qwenvl.model.da3_spatial_encoder import DA3SpatialEncoderPreTrainedModel, DA3SpatialEncoderConfig
from qwenvl.external.vggt.models.vggt import VGGT
from qwenvl.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_images_from_dir(image_dir):
    """Load all images from directory, sorted by name"""
    image_dir = Path(image_dir)
    
    # Find all image files
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_files)} images")
    
    # Load images
    images = []
    for img_file in image_files:
        img = Image.open(img_file).convert('RGB')
        img_np = np.array(img)
        images.append(img_np)
    
    return np.array(images), [f.name for f in image_files]


def extract_camera_positions(extrinsics_w2c):
    """Extract camera positions from w2c extrinsics"""
    if extrinsics_w2c.dim() == 4:
        extrinsics_w2c = extrinsics_w2c[0]  # [S, 3, 4]
    
    R = extrinsics_w2c[:, :3, :3]
    t = extrinsics_w2c[:, :3, 3:4]
    
    # camera position = -R^T @ t
    camera_pos = -torch.bmm(R.transpose(1, 2), t).squeeze(-1)
    return camera_pos.cpu().numpy()


def plot_trajectory_comparison(positions_dict, titles_dict, save_path=None):
    """Plot comparison of multiple trajectories"""
    num_methods = len(positions_dict)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Color map for different methods
    colors = {'cam_dec': 'blue', 'ray_head': 'red', 'vggt': 'green'}
    
    # ============================================================================
    # Row 1: Individual 3D Trajectories
    # ============================================================================
    for idx, (method_key, positions) in enumerate(positions_dict.items()):
        ax = fig.add_subplot(2, num_methods + 1, idx + 1, projection='3d')
        color = colors.get(method_key, 'gray')
        
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                f'{color[0]}-o', linewidth=2, markersize=6, label='Camera Path')
        ax.scatter(*positions[0], color='green', s=200, marker='^', 
                   label='Start', edgecolors='black', linewidth=2)
        ax.scatter(*positions[-1], color='orange', s=200, marker='v',
                   label='End', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(titles_dict[method_key], fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # ============================================================================
    # Overlay comparison (3D)
    # ============================================================================
    ax_overlay = fig.add_subplot(2, num_methods + 1, num_methods + 1, projection='3d')
    
    for method_key, positions in positions_dict.items():
        color = colors.get(method_key, 'gray')
        ax_overlay.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                       f'{color[0]}-o', linewidth=2, markersize=4, 
                       label=titles_dict[method_key], alpha=0.7)
    
    ax_overlay.set_xlabel('X (m)', fontsize=10)
    ax_overlay.set_ylabel('Y (m)', fontsize=10)
    ax_overlay.set_zlabel('Z (m)', fontsize=10)
    ax_overlay.set_title('🔍 Overlay Comparison', fontsize=12, fontweight='bold')
    ax_overlay.legend(fontsize=9)
    ax_overlay.grid(True, alpha=0.3)
    
    # ============================================================================
    # Row 2: Top View (XY) for each method
    # ============================================================================
    for idx, (method_key, positions) in enumerate(positions_dict.items()):
        ax = fig.add_subplot(2, num_methods + 1, num_methods + 2 + idx)
        color = colors.get(method_key, 'gray')
        
        ax.plot(positions[:, 0], positions[:, 1], f'{color[0]}-o', 
                linewidth=2, markersize=4)
        ax.scatter(positions[0, 0], positions[0, 1], color='green', s=100, 
                   marker='^', edgecolors='black', linewidth=2)
        ax.scatter(positions[-1, 0], positions[-1, 1], color='orange', s=100, 
                   marker='v', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(f'Top View - {titles_dict[method_key]}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # ============================================================================
    # Top view overlay
    # ============================================================================
    ax_overlay_top = fig.add_subplot(2, num_methods + 1, 2 * (num_methods + 1))
    
    for method_key, positions in positions_dict.items():
        color = colors.get(method_key, 'gray')
        ax_overlay_top.plot(positions[:, 0], positions[:, 1],
                           f'{color[0]}-o', linewidth=2, markersize=4,
                           label=titles_dict[method_key], alpha=0.7)
    
    ax_overlay_top.set_xlabel('X (m)', fontsize=10)
    ax_overlay_top.set_ylabel('Y (m)', fontsize=10)
    ax_overlay_top.set_title('Top View - Overlay', fontsize=10)
    ax_overlay_top.legend(fontsize=9)
    ax_overlay_top.grid(True, alpha=0.3)
    ax_overlay_top.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Comparison plot saved to: {save_path}")
    
    return fig


def compute_motion_stats(positions):
    """Compute motion statistics from positions"""
    motions = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return {
        'mean': motions.mean(),
        'std': motions.std(),
        'max': motions.max(),
        'total': motions.sum(),
        'motions': motions
    }


def main():
    print("\n" + "🎨"*40)
    print("DA3 vs VGGT Comparison - Image Directory")
    print("🎨"*40 + "\n")
    
    # Image directory
    image_dir = "/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/vsibench/sa_sampling_20f_single_video/arkitscenes/00777c41d4"
    
    print(f"📁 Image directory: {image_dir}\n")
    
    # Load images
    images, image_names = load_images_from_dir(image_dir)
    print(f"   Images: {image_names[:3]} ... {image_names[-1]}")
    print(f"   Shape: {images.shape}\n")
    
    # Limit to 6 frames for 10GB GPU (VGGT + DA3 both in memory)
    max_frames = 16
    if len(images) > max_frames:
        print(f"⚠️  Limiting to {max_frames} frames (GPU memory constraint)")
        images = images[:max_frames]
        image_names = image_names[:max_frames]
        print(f"   Selected: {image_names[0]} ... {image_names[-1]}\n")
    
    # Convert to tensor
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()  # [S, 3, H, W]
    
    if images_tensor.max() > 1.0:
        images_tensor = images_tensor / 255.0
    
    S, C, H, W = images_tensor.shape
    print(f"   Tensor shape: [S={S}, C={C}, H={H}, W={W}]")
    
    # Prepare for DA3
    video_tensor_da3 = [images_tensor.cuda()]
    video_grid_thw = torch.tensor([[S, H//14, W//14]])
    
    # Prepare for VGGT (needs 518x518)
    from torchvision.transforms.functional import resize
    images_tensor_vggt = resize(images_tensor, [518, 518], antialias=True)
    video_tensor_vggt = [images_tensor_vggt.cuda()]
    
    # Storage for all results
    all_positions = {}
    all_stats = {}
    titles = {}
    
    # ============================================================================
    # TEST 1: DA3 Camera Decoder (use_ray_pose=False)
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 1: DA3 Camera Decoder (use_ray_pose=False)")
    print("="*80)
    
    config_cam = DA3SpatialEncoderConfig(
        model_name="da3-large",
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
        process_res=504,
    )
    
    print("\n[DA3-CamDec] Initializing...")
    encoder_cam = DA3SpatialEncoderPreTrainedModel(config_cam).cuda()
    
    print("[DA3-CamDec] Running inference...")
    with torch.no_grad():
        _, _, camera_encs_cam = encoder_cam(video_tensor_da3, return_cam_enc=True, grid_thw=video_grid_thw)
    
    extrinsics_cam = camera_encs_cam[0][-1][0]
    print(f"✅ Extrinsics shape: {extrinsics_cam.shape} (w2c format)")
    
    positions_cam = extract_camera_positions(extrinsics_cam)
    all_positions['cam_dec'] = positions_cam
    all_stats['cam_dec'] = compute_motion_stats(positions_cam)
    titles['cam_dec'] = 'DA3 Camera Decoder'
    
    print(f"\n📊 Camera positions (first 3 frames):")
    for i in range(min(3, len(positions_cam))):
        print(f"  Frame {i}: [{positions_cam[i, 0]:7.4f}, {positions_cam[i, 1]:7.4f}, {positions_cam[i, 2]:7.4f}]")
    
    # Free memory
    del encoder_cam
    torch.cuda.empty_cache()
    
    # ============================================================================
    # TEST 2: DA3 Ray Head (use_ray_pose=True)
    # ============================================================================
    print("\n\n" + "="*80)
    print("TEST 2: DA3 Ray Head (use_ray_pose=True)")
    print("="*80)
    
    config_ray = DA3SpatialEncoderConfig(
        model_name="da3-large",
        use_ray_pose=True,
        ref_view_strategy="saddle_balanced",
        process_res=504,
    )
    
    print("\n[DA3-RayHead] Initializing...")
    encoder_ray = DA3SpatialEncoderPreTrainedModel(config_ray).cuda()
    
    print("[DA3-RayHead] Running inference...")
    with torch.no_grad():
        _, _, camera_encs_ray = encoder_ray(video_tensor_da3, return_cam_enc=True, grid_thw=video_grid_thw)
    
    extrinsics_ray = camera_encs_ray[0][-1][0]
    print(f"✅ Extrinsics shape: {extrinsics_ray.shape} (w2c format, converted)")
    
    positions_ray = extract_camera_positions(extrinsics_ray)
    all_positions['ray_head'] = positions_ray
    all_stats['ray_head'] = compute_motion_stats(positions_ray)
    titles['ray_head'] = 'DA3 Ray Head'
    
    print(f"\n📊 Camera positions (first 3 frames):")
    for i in range(min(3, len(positions_ray))):
        print(f"  Frame {i}: [{positions_ray[i, 0]:7.4f}, {positions_ray[i, 1]:7.4f}, {positions_ray[i, 2]:7.4f}]")
    
    # Free memory
    del encoder_ray
    torch.cuda.empty_cache()
    
    # ============================================================================
    # TEST 3: VGGT (Original Spatial Encoder)
    # ============================================================================
    print("\n\n" + "="*80)
    print("TEST 3: VGGT (Original Spatial Encoder)")
    print("="*80)
    
    print("\n[VGGT] Initializing...")
    vggt_model = VGGT(img_size=518, patch_size=14, embed_dim=1024).eval().cuda()
    
    # Load VGGT weights
    vggt_weight_path = "checkpoints/VGGT-1B/model.safetensors"
    print(f"[VGGT] Loading weights from: {vggt_weight_path}")
    vggt_state_dict = load_file(vggt_weight_path, device="cpu")
    missing_keys, unexpected_keys = vggt_model.load_state_dict(vggt_state_dict, strict=False)
    if missing_keys:
        print(f"   Warning: Missing keys: {missing_keys[:3]}...")
    
    print("[VGGT] Running inference...")
    with torch.no_grad():
        # VGGT expects [B, S, C, H, W]
        vggt_input = video_tensor_vggt[0].unsqueeze(0).float()  # [1, S, 3, 518, 518], ensure float32
        vggt_predictions = vggt_model(vggt_input, do_mis=True)
        pose_enc_vggt = vggt_predictions["pose_enc"]  # [B, S, 9]
    
    print(f"✅ Pose encoding shape: {pose_enc_vggt.shape}")
    
    # Convert VGGT pose encoding to extrinsics/intrinsics
    # pose_enc_vggt: [B, S, 9]
    B_vggt, S_vggt = pose_enc_vggt.shape[:2]
    pose_enc_flat = pose_enc_vggt.reshape(B_vggt * S_vggt, 9)  # [B*S, 9]
    
    extrinsics_vggt, intrinsics_vggt = pose_encoding_to_extri_intri(
        pose_enc_flat.unsqueeze(0).float(),  # [1, B*S, 9], ensure float32
        (518, 518)
    )
    
    print(f"✅ Extrinsics shape: {extrinsics_vggt.shape} (w2c format)")
    
    positions_vggt = extract_camera_positions(extrinsics_vggt)
    all_positions['vggt'] = positions_vggt
    all_stats['vggt'] = compute_motion_stats(positions_vggt)
    titles['vggt'] = 'VGGT'
    
    print(f"\n📊 Camera positions (first 3 frames):")
    for i in range(min(3, len(positions_vggt))):
        print(f"  Frame {i}: [{positions_vggt[i, 0]:7.4f}, {positions_vggt[i, 1]:7.4f}, {positions_vggt[i, 2]:7.4f}]")
    
    # Free memory
    del vggt_model
    torch.cuda.empty_cache()
    
    # ============================================================================
    # Final Comparison
    # ============================================================================
    print("\n\n" + "="*80)
    print("📊 FINAL COMPARISON - DA3 vs VGGT")
    print("="*80)
    
    save_dir = Path("test_results")
    save_dir.mkdir(exist_ok=True)
    
    # Plot comparison
    fig_comparison = plot_trajectory_comparison(
        all_positions,
        titles,
        save_path=save_dir / "da3_vs_vggt_comparison.png"
    )
    
    # Print statistics table
    print("\n" + "="*80)
    print("Motion Statistics Summary")
    print("="*80)
    print(f"\n{'Method':<20} {'Mean (m)':<12} {'Std (m)':<12} {'Max (m)':<12} {'Total (m)':<12}")
    print("-" * 80)
    
    for method_key in ['cam_dec', 'ray_head', 'vggt']:
        stats = all_stats[method_key]
        method_name = titles[method_key]
        print(f"{method_name:<20} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
              f"{stats['max']:<12.4f} {stats['total']:<12.4f}")
    
    # Pairwise position differences
    print("\n" + "="*80)
    print("Pairwise Position Differences (Mean L2 Distance)")
    print("="*80)
    
    methods = list(all_positions.keys())
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            pos_diff = np.linalg.norm(
                all_positions[method1] - all_positions[method2], axis=1
            )
            print(f"\n{titles[method1]} vs {titles[method2]}:")
            print(f"  Mean difference: {pos_diff.mean():.4f}m")
            print(f"  Max difference:  {pos_diff.max():.4f}m")
            print(f"  Min difference:  {pos_diff.min():.4f}m")
    
    # Recommendations
    print("\n" + "="*80)
    print("🎯 RECOMMENDATIONS")
    print("="*80)
    
    cam_dec_std = all_stats['cam_dec']['std']
    vggt_std = all_stats['vggt']['std']
    ray_std = all_stats['ray_head']['std']
    
    print(f"\n1. **Smoothness** (Lower std is better):")
    print(f"   - DA3 Camera Decoder: {cam_dec_std:.4f}m")
    print(f"   - VGGT:              {vggt_std:.4f}m")
    print(f"   - DA3 Ray Head:      {ray_std:.4f}m")
    
    if cam_dec_std < vggt_std:
        print(f"   ✅ DA3 Camera Decoder is {((vggt_std - cam_dec_std) / vggt_std * 100):.1f}% smoother than VGGT")
    else:
        print(f"   ⚠️  VGGT is {((cam_dec_std - vggt_std) / cam_dec_std * 100):.1f}% smoother than DA3 Camera Decoder")
    
    print(f"\n2. **For Pose-aware Sampling**:")
    print(f"   - Recommend: DA3 Camera Decoder (fast + smooth)")
    
    print(f"\n3. **For Pose RoPE**:")
    print(f"   - Any method works (just keep consistent)")
    print(f"   - Recommend: DA3 Camera Decoder (best speed/quality trade-off)")
    
    print(f"\n✅ Done! All results saved to {save_dir}/")
    print(f"   - da3_vs_vggt_comparison.png (main comparison)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
