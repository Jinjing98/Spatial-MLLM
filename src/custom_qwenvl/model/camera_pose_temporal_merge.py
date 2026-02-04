import torch
from typing import Tuple

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    Args:
        R: (..., 3, 3) rotation matrices
    Returns:
        q: (..., 4) quaternions
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    q = torch.zeros(R_flat.shape[0], 4, device=R.device, dtype=R.dtype)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s
    q[mask1, 1] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s
    q[mask1, 2] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s
    q[mask1, 3] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s
    
    # Case 2: R[0,0] is the largest diagonal
    mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    s = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2
    q[mask2, 0] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s
    q[mask2, 3] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s
    
    # Case 3: R[1,1] is the largest diagonal
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    s = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2
    q[mask3, 0] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s
    q[mask3, 1] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s
    q[mask3, 2] = 0.25 * s
    q[mask3, 3] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s
    
    # Case 4: R[2,2] is the largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2
    q[mask4, 0] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s
    q[mask4, 1] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s
    q[mask4, 2] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s
    q[mask4, 3] = 0.25 * s
    
    return q.reshape(*batch_shape, 4)

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to rotation matrix.
    Args:
        q: (..., 4) quaternions
    Returns:
        R: (..., 3, 3) rotation matrices
    """
    # Normalize quaternion
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.zeros(*q.shape[:-1], 3, 3, device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2*y*y - 2*z*z
    R[..., 0, 1] = 2*x*y - 2*z*w
    R[..., 0, 2] = 2*x*z + 2*y*w
    R[..., 1, 0] = 2*x*y + 2*z*w
    R[..., 1, 1] = 1 - 2*x*x - 2*z*z
    R[..., 1, 2] = 2*y*z - 2*x*w
    R[..., 2, 0] = 2*x*z - 2*y*w
    R[..., 2, 1] = 2*y*z + 2*x*w
    R[..., 2, 2] = 1 - 2*x*x - 2*y*y
    
    return R

def slerp_quaternion(q1: torch.Tensor, q2: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    """
    Spherical linear interpolation between two quaternions.
    Args:
        q1: (..., 4) first quaternion
        q2: (..., 4) second quaternion  
        t: interpolation parameter (0.5 for mean)
    Returns:
        q: (..., 4) interpolated quaternion
    """
    # Normalize quaternions
    q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)
    q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + 1e-8)
    
    # Compute dot product
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    
    # If dot < 0, negate q2 to take shorter path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.where(dot < 0, -dot, dot)
    
    # Clamp dot to avoid numerical issues with acos
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # If quaternions are very close, use linear interpolation
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # Use linear interpolation when sin_theta is small
    use_linear = (sin_theta.abs() < 1e-6).squeeze(-1)
    
    # SLERP formula
    w1 = torch.sin((1 - t) * theta) / (sin_theta + 1e-8)
    w2 = torch.sin(t * theta) / (sin_theta + 1e-8)
    q_slerp = w1 * q1 + w2 * q2
    
    # Linear interpolation fallback
    q_linear = (1 - t) * q1 + t * q2
    q_linear = q_linear / (torch.norm(q_linear, dim=-1, keepdim=True) + 1e-8)
    
    # Select based on condition
    result = torch.where(use_linear.unsqueeze(-1), q_linear, q_slerp)
    
    return result

def downsample_cams(intrisics,extrinsics_w2c, temporal_patch_size: int = 2, extrinsics_sample_strategy: str = "1st") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample the camera parameters by a temporal_patch_size of 2.
    Args:
        extrinsics_w2c: (B, S, 3, 4)
        intrisics: (B, S, 3, 3)
        temporal_patch_size: int
        extrinsics_sample_strategy: str
    Returns:
        intrisics_aligned: (B, S, 3, 3)
        extrinsics_aligned: (B, S, 3, 4)
    """
    intrisics_aligned = intrisics[:, ::temporal_patch_size]
    if extrinsics_sample_strategy == "1st":
        extrinsics_aligned = extrinsics_w2c[:, ::temporal_patch_size]
    elif extrinsics_sample_strategy == "mean":
        assert temporal_patch_size == 2, "temporal_patch_size must be 2 for mean strategy"
        # interpolate an intermediate pose given consecutive pair of pose1 and pose 2
        B, S, _, _ = extrinsics_w2c.shape
        # Get pairs: [0,1], [2,3], [4,5], ...
        extrinsics_even = extrinsics_w2c[:, 0::2]  # (B, S//2, 3, 4)
        extrinsics_odd = extrinsics_w2c[:, 1::2]   # (B, S//2, 3, 4)
        
        # Extract rotation and translation
        R1 = extrinsics_even[..., :3, :3]  # (B, S//2, 3, 3)
        t1 = extrinsics_even[..., :3, 3]   # (B, S//2, 3)
        R2 = extrinsics_odd[..., :3, :3]
        t2 = extrinsics_odd[..., :3, 3]
        
        # Interpolate rotation using SLERP
        q1 = rotation_matrix_to_quaternion(R1)  # (B, S//2, 4)
        q2 = rotation_matrix_to_quaternion(R2)
        q_mean = slerp_quaternion(q1, q2, t=0.5)
        R_mean = quaternion_to_rotation_matrix(q_mean)  # (B, S//2, 3, 3)
        
        # # Interpolate translation linearly
        # t_mean = (t1 + t2) / 2.0  # (B, S//2, 3)

        # C = -R^T @ t
        C1 = -torch.matmul(R1.transpose(-1, -2), t1.unsqueeze(-1)).squeeze(-1)
        C2 = -torch.matmul(R2.transpose(-1, -2), t2.unsqueeze(-1)).squeeze(-1)
        C_mean = (C1 + C2) / 2.0
        # Convert back to W2C translation: t = -R @ C
        t_mean = -torch.matmul(R_mean, C_mean.unsqueeze(-1)).squeeze(-1)

        # Combine into extrinsics matrix
        extrinsics_aligned = torch.cat([R_mean, t_mean.unsqueeze(-1)], dim=-1)  # (B, S//2, 3, 4)
    else:
        raise ValueError(f"Invalid extrinsics_sample_strategy: {extrinsics_sample_strategy}")
    return intrisics_aligned, extrinsics_aligned

def draw_camera_frustum(ax, R, t, color='blue', label='', scale=0.3):
    """
    Draw a camera frustum in 3D space.
    Args:
        ax: matplotlib 3D axis
        R: (3, 3) rotation matrix (W2C)
        t: (3,) translation vector (W2C)
        color: frustum color
        label: camera label
        scale: frustum size
    """
    # Camera center in world coordinates: C = -R^T @ t
    C = -R.T @ t
    
    # Camera coordinate axes in world frame
    # R is W2C, so R^T gives C2W rotation (camera axes in world frame)
    R_c2w = R.T
    
    # Define frustum corners in camera coordinates
    frustum_depth = scale
    frustum_width = scale * 0.6
    frustum_height = scale * 0.4
    
    # Frustum corners in camera frame (right-handed, z forward)
    corners_cam = torch.tensor([
        [0, 0, 0],  # apex (camera center)
        [-frustum_width, -frustum_height, frustum_depth],  # bottom-left
        [frustum_width, -frustum_height, frustum_depth],   # bottom-right
        [frustum_width, frustum_height, frustum_depth],    # top-right
        [-frustum_width, frustum_height, frustum_depth],   # top-left
    ], dtype=torch.float32)
    
    # Transform to world coordinates
    corners_world = (R_c2w @ corners_cam.T).T + C
    
    # Draw frustum edges
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # apex to corners
        (1, 2), (2, 3), (3, 4), (4, 1),  # far plane rectangle
    ]
    
    for i, j in edges:
        ax.plot3D(*zip(corners_world[i].numpy(), corners_world[j].numpy()), 
                  color=color, linewidth=1.5, alpha=0.7)
    
    # Draw camera center
    ax.scatter(*C.numpy(), color=color, s=100, marker='o', label=label)
    
    # Draw coordinate axes
    axis_length = scale * 0.5
    axes_colors = ['red', 'green', 'blue']  # X, Y, Z
    axes_labels = ['X', 'Y', 'Z']
    
    for i, (col, lab) in enumerate(zip(axes_colors, axes_labels)):
        axis_end = C + R_c2w[:, i] * axis_length
        ax.plot3D(*zip(C.numpy(), axis_end.numpy()), 
                  color=col, linewidth=2, alpha=0.8)

def visualize_camera_interpolation(R1, t1, R2, t2, R_mean, t_mean, save_path='camera_interpolation.png'):
    """
    Visualize camera interpolation in 3D space.
    Args:
        R1, t1: first camera (W2C)
        R2, t2: second camera (W2C)
        R_mean, t_mean: interpolated camera (W2C)
        save_path: output PNG path
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw cameras
    draw_camera_frustum(ax, R1, t1, color='blue', label='Camera 1', scale=0.5)
    draw_camera_frustum(ax, R2, t2, color='red', label='Camera 2', scale=0.5)
    draw_camera_frustum(ax, R_mean, t_mean, color='green', label='Interpolated (Mean)', scale=0.5)
    
    # Draw line connecting camera centers
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    C_mean = -R_mean.T @ t_mean
    
    ax.plot3D(*zip(C1.numpy(), C2.numpy()), 'k--', linewidth=1, alpha=0.5, label='Camera path')
    
    # Set labels and title
    ax.set_xlabel('X (World)', fontsize=10)
    ax.set_ylabel('Y (World)', fontsize=10)
    ax.set_zlabel('Z (World)', fontsize=10)
    ax.set_title('Camera Pose Interpolation Visualization', fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    all_centers = torch.stack([C1, C2, C_mean])
    max_range = (all_centers.max(0)[0] - all_centers.min(0)[0]).max() / 2.0
    mid = all_centers.mean(0)
    ax.set_xlim(mid[0] - max_range * 1.2, mid[0] + max_range * 1.2)
    ax.set_ylim(mid[1] - max_range * 1.2, mid[1] + max_range * 1.2)
    ax.set_zlim(mid[2] - max_range * 1.2, mid[2] + max_range * 1.2)
    
    # Add legend
    ax.legend(fontsize=10, loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Create proper rotation matrices (orthogonal) for testing
    def random_rotation_matrix(batch_size, num_frames):
        """Generate random valid rotation matrices."""
        # Generate random rotation angles
        angles = torch.randn(batch_size, num_frames, 3) * 0.5
        R = torch.zeros(batch_size, num_frames, 3, 3)
        
        for b in range(batch_size):
            for f in range(num_frames):
                # Simple rotation around z-axis for testing
                theta = angles[b, f, 2]
                R[b, f, 0, 0] = torch.cos(theta)
                R[b, f, 0, 1] = -torch.sin(theta)
                R[b, f, 1, 0] = torch.sin(theta)
                R[b, f, 1, 1] = torch.cos(theta)
                R[b, f, 2, 2] = 1.0
        return R
    
    print("Testing camera downsampling...")
    B, S = 2, 16
    intrisics = torch.randn(B, S, 3, 3)
    
    # Create valid extrinsics with proper rotation matrices
    R = random_rotation_matrix(B, S)
    t = torch.randn(B, S, 3, 1)
    extrinsics_w2c = torch.cat([R, t], dim=-1)  # (B, S, 3, 4)
    
    # Test strategy "1st"
    print("\n=== Testing strategy '1st' ===")
    intrisics_aligned_1st, extrinsics_w2c_aligned_1st = downsample_cams(
        intrisics, extrinsics_w2c, temporal_patch_size=2, extrinsics_sample_strategy="1st"
    )
    print(f"Intrinsics shape: {intrisics_aligned_1st.shape} (expected: ({B}, {S//2}, 3, 3))")
    print(f"Extrinsics shape: {extrinsics_w2c_aligned_1st.shape} (expected: ({B}, {S//2}, 3, 4))")
    
    # Visualize '1st' strategy
    print("\n=== Generating 3D visualization for '1st' strategy ===")
    R1_vis_1st = R[0, 0].clone()
    t1_vis_1st = t[0, 0, :, 0].clone()
    R2_vis_1st = R[0, 1].clone()
    t2_vis_1st = t[0, 1, :, 0].clone()
    R_1st_vis = extrinsics_w2c_aligned_1st[0, 0, :3, :3].clone()
    t_1st_vis = extrinsics_w2c_aligned_1st[0, 0, :3, 3].clone()
    
    visualize_camera_interpolation(
        R1_vis_1st, t1_vis_1st, R2_vis_1st, t2_vis_1st,
        R_1st_vis, t_1st_vis,
        save_path='camera_interpolation_1st.png'
    )
    
    # Test strategy "mean"
    print("\n=== Testing strategy 'mean' ===")
    intrisics_aligned_mean, extrinsics_w2c_aligned_mean = downsample_cams(
        intrisics, extrinsics_w2c, temporal_patch_size=2, extrinsics_sample_strategy="mean"
    )
    print(f"Intrinsics shape: {intrisics_aligned_mean.shape} (expected: ({B}, {S//2}, 3, 3))")
    print(f"Extrinsics shape: {extrinsics_w2c_aligned_mean.shape} (expected: ({B}, {S//2}, 3, 4))")
    
    # Verify rotation matrices are still valid (orthogonal)
    R_mean = extrinsics_w2c_aligned_mean[..., :3, :3]
    RRT = torch.matmul(R_mean, R_mean.transpose(-1, -2))
    I = torch.eye(3).unsqueeze(0).unsqueeze(0).expand_as(RRT)
    orthogonality_error = torch.abs(RRT - I).max()
    print(f"\nOrthogonality check (max error): {orthogonality_error.item():.6f} (should be < 1e-5)")
    
    # Check determinant is 1 (proper rotation)
    det = torch.linalg.det(R_mean)
    det_error = torch.abs(det - 1.0).max()
    print(f"Determinant check (max error): {det_error.item():.6f} (should be < 1e-5)")
    
    # Verify interpolation: check that mean translation is between the two original translations
    t_mean = extrinsics_w2c_aligned_mean[0, 0, :, 3]
    t0 = extrinsics_w2c[0, 0, :, 3]
    t1 = extrinsics_w2c[0, 1, :, 3]
    t_expected = (t0 + t1) / 2.0
    translation_error = torch.abs(t_mean - t_expected).max()
    print(f"Translation interpolation check (max error): {translation_error.item():.6f} (should be < 1e-5)")
    
    print("\n✓ All tests passed!")
    
    # Visualize 'mean' strategy
    print("\n=== Generating 3D visualization for 'mean' strategy ===")
    # Use first sample from batch for visualization
    R1_vis = R[0, 0].clone()
    t1_vis = t[0, 0, :, 0].clone()
    R2_vis = R[0, 1].clone()
    t2_vis = t[0, 1, :, 0].clone()
    
    # Get interpolated camera
    R_mean_vis = extrinsics_w2c_aligned_mean[0, 0, :3, :3].clone()
    t_mean_vis = extrinsics_w2c_aligned_mean[0, 0, :3, 3].clone()
    
    visualize_camera_interpolation(
        R1_vis, t1_vis, R2_vis, t2_vis, 
        R_mean_vis, t_mean_vis,
        save_path='camera_interpolation_mean.png'
    )


