# JJ: Visualization utilities for camera poses and farness scores
import numpy as np
from typing import List, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_camera_trajectory_3d(
    poses: Union[List[np.ndarray], np.ndarray],
    farness_trans: List[float],
    farness_rot: List[float],
    color_by: str = 'trans',
    output_path: str = None,
    show: bool = True
):
    """
    Visualize 3D camera trajectory with farness scores.
    
    Args:
        poses: List of 4x4 transformation matrices or (N, 4, 4) array
        farness_trans: Translation farness scores
        farness_rot: Rotation farness scores (in radians)
        color_by: 'trans' or 'rot' - which metric to use for coloring
        output_path: Path to save HTML file (optional)
        show: Whether to show the plot
    """
    if isinstance(poses, list):
        poses = np.stack([np.array(p) for p in poses])
    
    # Extract camera positions
    positions = poses[:, :3, 3]
    
    # Extract camera orientations (forward direction)
    forward_dirs = poses[:, :3, 2]  # Z-axis of camera frame
    
    # Choose color metric
    if color_by == 'trans':
        colors = farness_trans
        color_label = 'Translation Distance (m)'
    else:
        colors = [np.degrees(r) for r in farness_rot]
        color_label = 'Rotation Distance (°)'
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add camera positions
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers+lines',
        marker=dict(
            size=6,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_label),
            line=dict(width=0.5, color='white')
        ),
        line=dict(color='gray', width=2),
        text=[f"Frame {i}<br>Trans: {farness_trans[i]:.3f}m<br>Rot: {np.degrees(farness_rot[i]):.1f}°" 
              for i in range(len(positions))],
        hoverinfo='text',
        name='Camera Trajectory'
    ))
    
    # Highlight reference camera (first one)
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]],
        y=[positions[0, 1]],
        z=[positions[0, 2]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        name='Reference Camera',
        hovertext='Reference (Frame 0)'
    ))
    
    # Add camera orientation vectors (every 10th frame to avoid clutter)
    arrow_indices = list(range(0, len(positions), max(1, len(positions) // 20)))
    for idx in arrow_indices:
        start = positions[idx]
        end = start + forward_dirs[idx] * 0.2  # Scale factor for visibility
        
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f'3D Camera Trajectory (colored by {color_by})',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        width=1000,
        height=800
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"3D trajectory saved to: {output_path}")
    
    if show:
        fig.show()
    
    return fig


def visualize_farness_scatter(
    farness_trans: List[float],
    farness_rot: List[float],
    output_path: str = None,
    show: bool = True
):
    """
    Visualize 2D scatter plot of translation vs rotation farness.
    
    Args:
        farness_trans: Translation farness scores
        farness_rot: Rotation farness scores (in radians)
        output_path: Path to save HTML file (optional)
        show: Whether to show the plot
    """
    # Convert rotation to degrees for better readability
    farness_rot_deg = [np.degrees(r) for r in farness_rot]
    
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=farness_trans,
        y=farness_rot_deg,
        mode='markers',
        marker=dict(
            size=8,
            color=list(range(len(farness_trans))),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Frame Index'),
            line=dict(width=0.5, color='white')
        ),
        text=[f"Frame {i}<br>Trans: {farness_trans[i]:.3f}m<br>Rot: {farness_rot_deg[i]:.1f}°" 
              for i in range(len(farness_trans))],
        hoverinfo='text',
        name='Frames'
    ))
    
    # Highlight reference frame
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond', line=dict(width=2, color='white')),
        name='Reference',
        hovertext='Reference (Frame 0)'
    ))
    
    fig.update_layout(
        title='Translation vs Rotation Farness',
        xaxis_title='Translation Distance (m)',
        yaxis_title='Rotation Distance (°)',
        width=900,
        height=700,
        hovermode='closest'
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"Scatter plot saved to: {output_path}")
    
    if show:
        fig.show()
    
    return fig


def visualize_farness_timeseries(
    farness_trans: List[float],
    farness_rot: List[float],
    frame_indices: List[int] = None,
    output_path: str = None,
    show: bool = True
):
    """
    Visualize time series of farness scores.
    
    Args:
        farness_trans: Translation farness scores
        farness_rot: Rotation farness scores (in radians)
        frame_indices: Optional list of actual frame indices (if not sequential)
        output_path: Path to save HTML file (optional)
        show: Whether to show the plot
    """
    if frame_indices is None:
        frame_indices = list(range(len(farness_trans)))
    
    # Convert rotation to degrees
    farness_rot_deg = [np.degrees(r) for r in farness_rot]
    
    # Create subplot with two y-axes
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Translation Farness', 'Rotation Farness'),
        vertical_spacing=0.12
    )
    
    # Translation subplot
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=farness_trans,
            mode='lines+markers',
            name='Translation',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            hovertemplate='Frame %{x}<br>Trans: %{y:.3f}m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Rotation subplot
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=farness_rot_deg,
            mode='lines+markers',
            name='Rotation',
            line=dict(color='red', width=2),
            marker=dict(size=4),
            hovertemplate='Frame %{x}<br>Rot: %{y:.1f}°<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Frame Index", row=2, col=1)
    fig.update_yaxes(title_text="Distance (m)", row=1, col=1)
    fig.update_yaxes(title_text="Angle (°)", row=2, col=1)
    
    fig.update_layout(
        title='Farness Time Series',
        width=1200,
        height=800,
        showlegend=True
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"Time series plot saved to: {output_path}")
    
    if show:
        fig.show()
    
    return fig


def visualize_distance_matrix(
    poses: Union[List[np.ndarray], np.ndarray],
    metric_mode: str = 'trans',
    output_path: str = None,
    show: bool = True
):
    """
    Visualize pairwise distance matrix between all camera poses.
    
    Args:
        poses: List of 4x4 transformation matrices or (N, 4, 4) array
        metric_mode: 'trans' for translation, 'rot' for rotation
        output_path: Path to save HTML file (optional)
        show: Whether to show the plot
    """
    if isinstance(poses, list):
        poses = np.stack([np.array(p) for p in poses])
    
    n_poses = len(poses)
    distance_matrix = np.zeros((n_poses, n_poses))
    
    # Compute pairwise distances
    for i in range(n_poses):
        for j in range(n_poses):
            if metric_mode == 'trans':
                # Translation distance
                t_i = poses[i, :3, 3]
                t_j = poses[j, :3, 3]
                distance_matrix[i, j] = np.linalg.norm(t_i - t_j)
            else:
                # Rotation distance (geodesic)
                R_i = poses[i, :3, :3]
                R_j = poses[j, :3, :3]
                R_rel = R_i.T @ R_j
                trace = np.trace(R_rel)
                cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                distance_matrix[i, j] = np.degrees(angle)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=distance_matrix,
        x=list(range(n_poses)),
        y=list(range(n_poses)),
        colorscale='Viridis',
        colorbar=dict(
            title='Distance (m)' if metric_mode == 'trans' else 'Angle (°)'
        ),
        hovertemplate='Frame %{x} to Frame %{y}<br>Distance: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Pairwise {"Translation" if metric_mode == "trans" else "Rotation"} Distance Matrix',
        xaxis_title='Frame Index',
        yaxis_title='Frame Index',
        width=900,
        height=800
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"Distance matrix saved to: {output_path}")
    
    if show:
        fig.show()
    
    return fig


def visualize_all(
    poses_c2w: Union[List[np.ndarray], np.ndarray],
    farness_trans: List[float],
    farness_rot: List[float],
    frame_indices: List[int] = None,
    selected_indices: List[int] = None,
    output_path: str = None,
    show: bool = True,
    translation_unit: str = 'normalized',
    plot_selected_distance_matrix: bool = True,  # JJ: plot distance matrix for selected frames only
    pose_source: str = None,  # JJ: Source of poses ('GT', 'VGGT Estimated', 'Other')
):
    """
    Generate all visualizations in a single HTML file.
    
    Args:
        poses_c2w: Camera-to-world poses. List of 4x4 transformation matrices or (N, 4, 4) array
        farness_trans: Translation farness scores
        farness_rot: Rotation farness scores (in radians)
        frame_indices: Optional list of actual frame indices
        selected_indices: Optional list of selected frame indices to highlight in red
        output_path: Path to save HTML file (optional, default: 'pose_analysis.html')
        show: Whether to show the plot
        translation_unit: Unit label for translation ('normalized' or 'm')
        plot_selected_distance_matrix: If True and selected_indices is provided, plot distance matrices for selected frames only
        pose_source: Source of poses - will add annotation if using estimated poses (e.g., 'VGGT Estimated', 'VGGT Predicted')
    """
    from pathlib import Path
    
    # Sanity checks
    if isinstance(poses_c2w, list):
        if len(poses_c2w) == 0:
            raise ValueError("Input poses_c2w is empty")
        poses_c2w = np.stack([np.array(p) for p in poses_c2w])
    
    if len(farness_trans) != len(farness_rot):
        raise ValueError(f"Length mismatch: farness_trans ({len(farness_trans)}) != farness_rot ({len(farness_rot)})")
    
    if len(farness_trans) != len(poses_c2w):
        raise ValueError(f"Length mismatch: farness scores ({len(farness_trans)}) != poses_c2w ({len(poses_c2w)})")
    
    if frame_indices is None:
        frame_indices = list(range(len(farness_trans)))
    elif len(frame_indices) != len(farness_trans):
        raise ValueError(f"Length mismatch: frame_indices ({len(frame_indices)}) != farness scores ({len(farness_trans)})")
    
    farness_rot_deg = [np.degrees(r) for r in farness_rot]
    n_poses = len(poses_c2w)
    
    print("\n" + "=" * 60)
    print("Generating combined visualization:")
    print("=" * 60)
    
    # JJ: Dynamic layout based on plot_selected_distance_matrix flag
    has_selected = plot_selected_distance_matrix and selected_indices is not None and len(selected_indices) > 0
    
    if has_selected:
        # 4-row layout with selected frames distance matrices
        n_rows = 4
        subplot_titles = (
            'Translation Distance Matrix (All Frames)',
            'Rotation Distance Matrix (All Frames)',
            'Translation Distance Matrix (Selected Frames)',
            'Rotation Distance Matrix (Selected Frames)',
            'Translation Farness Time Series',
            'Rotation Farness Time Series',
            'Translation vs Rotation Scatter',
            ''  # Empty placeholder
        )
        specs = [
            [{"type": "heatmap"}, {"type": "heatmap"}],      # Row 1: All frames dist matrices
            [{"type": "heatmap"}, {"type": "heatmap"}],      # Row 2: Selected frames dist matrices
            [{"type": "scatter"}, {"type": "scatter"}],      # Row 3: Time series
            [{"type": "scatter"}, None]                       # Row 4: Scatter + empty
        ]
        row_heights = [0.25, 0.25, 0.25, 0.25]
    else:
        # 3-row layout (original)
        n_rows = 3
        subplot_titles = (
            'Translation Distance Matrix (All Frames)',
            'Rotation Distance Matrix (All Frames)',
            'Translation Farness Time Series',
            'Rotation Farness Time Series',
            'Translation vs Rotation Scatter',
            ''  # Empty placeholder
        )
        specs = [
            [{"type": "heatmap"}, {"type": "heatmap"}],      # Row 1: All frames dist matrices
            [{"type": "scatter"}, {"type": "scatter"}],      # Row 2: Time series
            [{"type": "scatter"}, None]                       # Row 3: Scatter + empty
        ]
        row_heights = [0.33, 0.33, 0.33]
    
    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
        row_heights=row_heights
    )
    
    trans_unit_label = translation_unit
    
    # JJ: Define dynamic row positions based on layout
    if has_selected:
        row_time_series = 3  # Time series on row 3
        row_scatter = 4      # Scatter on row 4
    else:
        row_time_series = 2  # Time series on row 2
        row_scatter = 3      # Scatter on row 3
    
    # 1. Translation Distance Matrix (vectorized)
    print("[1/4] Computing translation distance matrix...")
    T_c2w = poses_c2w[:, :3, 3]  # (N, 3)
    # Vectorized: ||T_i - T_j|| for all i, j
    trans_dist_matrix = np.linalg.norm(T_c2w[:, None] - T_c2w[None, :], axis=-1)  # (N, N)
    
    fig.add_trace(
        go.Heatmap(
            z=trans_dist_matrix,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=f'Distance ({trans_unit_label})',
                x=0.46,
                len=0.28,
                y=0.84
            ),
            hovertemplate='Frame %{x} to %{y}<br>Dist: %{z:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Rotation Distance Matrix (vectorized)
    print("[2/4] Computing rotation distance matrix...")
    R_c2w = poses_c2w[:, :3, :3]  # (N, 3, 3)
    
    # Vectorized rotation distance computation
    # R_rel[i, j] = R_i^T @ R_j for all i, j
    # Use broadcasting: R_i^T @ R_j = (N, 3, 3)^T @ (N, 3, 3)
    R_i_T = np.transpose(R_c2w, (0, 2, 1))  # (N, 3, 3) transposed -> (N, 3, 3)
    R_j = R_c2w  # (N, 3, 3)
    
    # Compute all pairwise products: R_i_T[i] @ R_j[j]
    # Use einsum: 'ikl,jlm->ijkm' then trace over k=m
    R_rel = np.einsum('ikl,jlm->ijkm', R_i_T, R_j)  # (N, N, 3, 3)
    traces = np.einsum('iijj->ij', R_rel)  # Trace over last two dimensions -> (N, N)
    # Actually we want diagonal of 3x3, so:
    traces = np.trace(R_rel, axis1=2, axis2=3)  # (N, N)
    cos_angles = np.clip((traces - 1) / 2, -1.0, 1.0)
    rot_dist_matrix = np.degrees(np.arccos(cos_angles))  # (N, N)
    
    fig.add_trace(
        go.Heatmap(
            z=rot_dist_matrix,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Angle (°)',
                x=1.02,
                len=0.28,
                y=0.84
            ),
            hovertemplate='Frame %{x} to %{y}<br>Angle: %{z:.1f}°<extra></extra>'
        ),
        row=1, col=2
    )
    
    # JJ: 2b. Distance matrices for selected frames (if enabled)
    if has_selected:
        print(f"[2b] Computing distance matrices for {len(selected_indices)} selected frames...")
        
        # Extract selected poses
        selected_poses = poses_c2w[selected_indices]
        
        # Translation distance matrix for selected frames
        T_selected = selected_poses[:, :3, 3]
        trans_dist_matrix_selected = np.linalg.norm(T_selected[:, None] - T_selected[None, :], axis=-1)
        
        fig.add_trace(
            go.Heatmap(
                z=trans_dist_matrix_selected,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=f'Distance ({trans_unit_label})',
                    x=0.46,
                    len=0.20,
                    y=0.59
                ),
                hovertemplate='Selected %{x} to %{y}<br>Dist: %{z:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Rotation distance matrix for selected frames
        R_selected = selected_poses[:, :3, :3]
        R_sel_i_T = np.transpose(R_selected, (0, 2, 1))
        R_sel_j = R_selected
        R_rel_selected = np.einsum('ikl,jlm->ijkm', R_sel_i_T, R_sel_j)
        traces_selected = np.trace(R_rel_selected, axis1=2, axis2=3)
        cos_angles_selected = np.clip((traces_selected - 1) / 2, -1.0, 1.0)
        rot_dist_matrix_selected = np.degrees(np.arccos(cos_angles_selected))
        
        fig.add_trace(
            go.Heatmap(
                z=rot_dist_matrix_selected,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Angle (°)',
                    x=1.02,
                    len=0.20,
                    y=0.59
                ),
                hovertemplate='Selected %{x} to %{y}<br>Angle: %{z:.1f}°<extra></extra>'
            ),
            row=2, col=2
        )
    
    # 3. Scatter Plot (Translation vs Rotation)
    print("[3/4] Generating scatter plot...")
    # JJ: Color selected frames differently if provided
    selected_set = set(selected_indices) if selected_indices else set()
    
    if selected_indices:
        # Plot non-selected frames
        non_selected = [i for i in range(len(farness_trans)) if i not in selected_set]
        if non_selected:
            fig.add_trace(
                go.Scatter(
                    x=[farness_trans[i] for i in non_selected],
                    y=[farness_rot_deg[i] for i in non_selected],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=non_selected,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title='Frame Index',
                            x=0.46,
                            len=0.28,
                            y=0.51
                        ),
                        line=dict(width=0.5, color='white')
                    ),
                    text=[f"Frame {i}<br>Trans: {farness_trans[i]:.3f}<br>Rot: {farness_rot_deg[i]:.1f}°" 
                          for i in non_selected],
                hoverinfo='text',
                showlegend=False,
                name='All Frames'
            ),
            row=row_scatter, col=1
        )
        
        # Plot selected frames in red
        fig.add_trace(
            go.Scatter(
                x=[farness_trans[i] for i in selected_indices],
                y=[farness_rot_deg[i] for i in selected_indices],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    line=dict(width=1, color='white')
                ),
                text=[f"Frame {i} (SELECTED)<br>Trans: {farness_trans[i]:.3f}<br>Rot: {farness_rot_deg[i]:.1f}°" 
                      for i in selected_indices],
                hoverinfo='text',
                showlegend=False,
                name='Selected Frames'
            ),
            row=row_scatter, col=1
        )
    else:
        # Original behavior when no selection
        fig.add_trace(
            go.Scatter(
                x=farness_trans,
                y=farness_rot_deg,
                mode='markers',
                marker=dict(
                    size=6,
                    color=list(range(len(farness_trans))),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title='Frame Index',
                        x=0.46,
                        len=0.28,
                        y=0.51
                    ),
                    line=dict(width=0.5, color='white')
                ),
                text=[f"Frame {i}<br>Trans: {farness_trans[i]:.3f}<br>Rot: {farness_rot_deg[i]:.1f}°" 
                      for i in range(len(farness_trans))],
                hoverinfo='text',
                showlegend=False
            ),
            row=row_scatter, col=1
        )
    
    # Add reference frame to scatter
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='white')),
            showlegend=False,
            hovertext='Reference (Frame 0)'
        ),
        row=row_scatter, col=1
    )
    
    # 4. Time Series - Translation
    print("[4/4] Generating time series...")
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=farness_trans,
            mode='lines+markers',
            name='Translation',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            hovertemplate=f'Frame %{{x}}<br>Trans: %{{y:.3f}} {trans_unit_label}<extra></extra>',
            showlegend=True
        ),
        row=row_time_series, col=1
    )
    
    # JJ: Highlight selected frames on translation time series
    if selected_indices:
        fig.add_trace(
            go.Scatter(
                x=[frame_indices[i] for i in selected_indices],
                y=[farness_trans[i] for i in selected_indices],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                name='Selected',
                hovertemplate=f'Frame %{{x}} (SELECTED)<br>Trans: %{{y:.3f}} {trans_unit_label}<extra></extra>',
                showlegend=False
            ),
            row=row_time_series, col=1
        )
    
    # 5. Time Series - Rotation
    fig.add_trace(
        go.Scatter(
            x=frame_indices,
            y=farness_rot_deg,
            mode='lines+markers',
            name='Rotation',
            line=dict(color='red', width=2),
            marker=dict(size=4),
            hovertemplate='Frame %{x}<br>Rot: %{y:.1f}°<extra></extra>',
            showlegend=True
        ),
        row=row_time_series, col=2
    )
    
    # JJ: Highlight selected frames on rotation time series
    if selected_indices:
        fig.add_trace(
            go.Scatter(
                x=[frame_indices[i] for i in selected_indices],
                y=[farness_rot_deg[i] for i in selected_indices],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                name='Selected',
                hovertemplate='Frame %{x} (SELECTED)<br>Rot: %{y:.1f}°<extra></extra>',
                showlegend=False
            ),
            row=row_time_series, col=2
        )
    
    # Update axes labels
    # Row 1: All frames distance matrices
    fig.update_xaxes(title_text="Frame Index", showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Frame Index", showgrid=False, row=1, col=1)
    fig.update_xaxes(title_text="Frame Index", showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text="Frame Index", showgrid=False, row=1, col=2)
    
    # Ensure heatmaps are square (equal aspect ratio)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    
    # JJ: Row 2: Selected frames distance matrices (if enabled)
    if has_selected:
        fig.update_xaxes(title_text="Selected Frame Index", showgrid=False, row=2, col=1)
        fig.update_yaxes(title_text="Selected Frame Index", showgrid=False, row=2, col=1)
        fig.update_xaxes(title_text="Selected Frame Index", showgrid=False, row=2, col=2)
        fig.update_yaxes(title_text="Selected Frame Index", showgrid=False, row=2, col=2)
        fig.update_yaxes(scaleanchor="x3", scaleratio=1, row=2, col=1)
        fig.update_yaxes(scaleanchor="x4", scaleratio=1, row=2, col=2)
    
    # Time series row (dynamic)
    fig.update_xaxes(title_text="Frame Index", row=row_time_series, col=1)
    fig.update_yaxes(title_text=f"Translation ({trans_unit_label})", row=row_time_series, col=1)
    fig.update_xaxes(title_text="Frame Index", row=row_time_series, col=2)
    fig.update_yaxes(title_text="Rotation (°)", row=row_time_series, col=2)
    
    # Scatter plot row (dynamic)
    fig.update_xaxes(title_text=f"Translation Distance ({trans_unit_label})", row=row_scatter, col=1)
    fig.update_yaxes(title_text="Rotation Distance (°)", row=row_scatter, col=1)
    
    # Update overall layout
    layout_height = 1600 if has_selected else 1400  # JJ: Taller when showing selected frames
    
    # JJ: Determine title based on pose source
    title_suffix = ''
    if pose_source and pose_source.lower() not in ['gt', 'ground truth']:
        title_suffix = f' <span style="font-size:14px; color:orange;">[Using {pose_source} Poses]</span>'
    
    fig.update_layout(
        title_text=f'Camera Pose Farness Analysis{title_suffix}',
        width=1400,
        height=layout_height,
        showlegend=True,
        legend=dict(x=0.75, y=0.35),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    if output_path is None:
        output_path = 'tmp/pose_analysis.html'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JJ: Write HTML with optional appendix for estimated poses
    html_str = fig.to_html()
    
    # JJ: Add appendix note if using estimated poses
    if pose_source and pose_source.lower() not in ['gt', 'ground truth']:
        appendix_note = f"""
        <div style="margin: 20px; padding: 15px; background-color: #fff3cd; border-left: 5px solid #ff9800; border-radius: 5px;">
            <h3 style="margin-top: 0; color: #ff9800;">⚠ Note: Using Estimated Poses</h3>
            <p style="margin-bottom: 0;">
                This visualization is based on <strong>{pose_source}</strong> rather than ground truth poses. 
                The farness analysis and distance metrics reflect the quality of the estimated pose trajectory. 
                Discrepancies may exist between estimated and ground truth camera positions.
            </p>
        </div>
        </body>
        """
        html_str = html_str.replace('</body>', appendix_note)
    
    with open(output_path, 'w') as f:
        f.write(html_str)
    
    # JJ: Print absolute path for easy clicking
    abs_path = output_path.resolve()
    print(f"\n✓ Combined visualization saved to: {abs_path}")
    
    if show:
        fig.show()
    
    return fig


def visualize_sampled_pose(
    pose_analysis_poses: Union[List[np.ndarray], np.ndarray],  # JJ: Poses for sampling_quality.html
    selected_indices: List[int],
    min_distances: np.ndarray = None,
    distance_matrix: np.ndarray = None,
    farness_trans: List[float] = None,
    farness_rot: List[float] = None,
    output_path: str = None,
    show: bool = True,
    method_name: str = 'FPS',
    plot_pose_analysis: bool = False,
    pose_analysis_selected: List[int] = None,  # JJ: Selected indices for pose_analysis
    pose_analysis_target_poses: Union[List[np.ndarray], np.ndarray] = None,  # JJ: Different poses for pose_analysis if needed
    pose_analysis_frame_ids: List[int] = None,  # JJ: Actual frame IDs for pose_analysis x-axis
    pose_source: str = None  # JJ: Source of poses for annotation in pose_analysis.html
):
    """
    Visualize the quality of sampled pose selection with 1x2 layout (2 figures).
    
    Generates 3D trajectory and coverage quality visualization.
    
    Args:
        pose_analysis_poses: Camera poses for sampling_quality.html (N, 4, 4)
        selected_indices: Indices of selected frames for sampling quality
        min_distances: For each frame, distance to nearest selected frame (optional)
        distance_matrix: Full (N, N) distance matrix (optional, will compute if None)
        farness_trans: Translation farness scores (optional, for pose_analysis)
        farness_rot: Rotation farness scores in radians (optional, for pose_analysis)
        output_path: Path to save HTML file
        show: Whether to show the plot
        method_name: Name of the sampling method (for title)
        plot_pose_analysis: If True, also generate pose_analysis.html with farness analysis.
        pose_analysis_selected: Selected indices in pose_analysis.html (if None, use selected_indices)
        pose_analysis_target_poses: Poses for pose_analysis.html (if None, use pose_analysis_poses)
        pose_analysis_frame_ids: Actual frame IDs for pose_analysis x-axis (if None, use sequential indices)
        pose_source: Source of poses (e.g., 'GT', 'VGGT Estimated') - adds annotation in pose_analysis.html if not GT
    """
    from pathlib import Path
    
    if isinstance(pose_analysis_poses, list):
        pose_analysis_poses = np.stack([np.array(p) for p in pose_analysis_poses])
    
    poses = pose_analysis_poses  # JJ: Use unified poses for sampling_quality visualization
    N = len(poses)
    m = len(selected_indices)
    selected_set = set(selected_indices)
    
    # Compute min_distances if not provided
    if min_distances is None:
        if distance_matrix is None:
            # Compute simple distance matrix
            from pose_fps_sampling import compute_pairwise_pose_distance
            distance_matrix = compute_pairwise_pose_distance(poses)
        
        min_distances = np.full(N, np.inf)
        for i in range(N):
            for j in selected_indices:
                min_distances[i] = min(min_distances[i], distance_matrix[i, j])
    
    # JJ: If plot_pose_analysis=True, call visualize_all
    if plot_pose_analysis:
        # Use pose_analysis_target_poses if provided, otherwise use poses
        if pose_analysis_target_poses is not None:
            if isinstance(pose_analysis_target_poses, list):
                analysis_poses = np.stack([np.array(p) for p in pose_analysis_target_poses])
            else:
                analysis_poses = pose_analysis_target_poses
        else:
            analysis_poses = poses
        
        analysis_selected = pose_analysis_selected if pose_analysis_selected is not None else selected_indices
        N_analysis = len(analysis_poses)
        
        # Compute farness for analysis poses if needed
        if farness_trans is None or farness_rot is None:
            print("  → Computing farness scores for pose_analysis...")
            try:
                from .pose_distance_metrics import compute_pose_farness
            except ImportError:
                from pose_distance_metrics import compute_pose_farness
            poses_list = [analysis_poses[i] for i in range(N_analysis)]
            farness_trans_analysis, farness_rot_analysis = compute_pose_farness(poses_list, translation_scale=None)
            print(f"  → Farness trans range: [{min(farness_trans_analysis):.4f}, {max(farness_trans_analysis):.4f}] m")
            print(f"  → Farness rot range: [{np.degrees(min(farness_rot_analysis)):.2f}°, {np.degrees(max(farness_rot_analysis)):.2f}°]")
        else:
            farness_trans_analysis = farness_trans
            farness_rot_analysis = farness_rot
        
        print("\n" + "=" * 60)
        print("Generating pose_analysis.html with selected points marked...")
        print("=" * 60)
        
        # Determine output path for visualize_all
        if output_path:
            # Extract method suffix from main output path
            # e.g., "efficient_sampling_grid.html" -> "pose_analysis_grid.html"
            output_stem = Path(output_path).stem  # e.g., "efficient_sampling_grid"
            if '_' in output_stem:
                # Extract suffix after last underscore
                suffix = output_stem.split('_')[-1]  # e.g., "grid"
                vis_all_path = Path(output_path).parent / f"pose_analysis_{suffix}.html"
            else:
                vis_all_path = Path(output_path).parent / "pose_analysis.html"
        else:
            vis_all_path = "tmp/pose_analysis.html"
        
        # JJ: Use actual frame IDs for x-axis if provided, otherwise use sequential indices
        frame_ids_for_plot = pose_analysis_frame_ids if pose_analysis_frame_ids is not None else list(range(N_analysis))
        
        visualize_all(
            poses_c2w=analysis_poses,
            farness_trans=farness_trans_analysis,
            farness_rot=farness_rot_analysis,
            frame_indices=frame_ids_for_plot,
            selected_indices=analysis_selected,
            output_path=str(vis_all_path),
            show=False,
            translation_unit='m',
            pose_source=pose_source  # JJ: Pass pose source for annotation
        )
    
    # JJ: Create 1x2 subplot layout (only 3D traj + coverage quality)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '3D Camera Trajectory (Selected Frames)',
            'Coverage Quality (Min Distance per Frame)'
        ),
        specs=[
            [{"type": "scatter3d"}, {"type": "bar"}]
        ],
        horizontal_spacing=0.10
    )
    
    positions = poses[:, :3, 3]
    
    # 1. 3D Trajectory with selected frames highlighted
    print("[1/2] Plotting 3D trajectory...")
    
    # Downsample trajectory for large N to reduce file size
    max_trajectory_points = 2000
    if N > max_trajectory_points:
        print(f"  → Downsampling 3D trajectory from {N} to {max_trajectory_points} points...")
        traj_step = max(1, N // max_trajectory_points)
        traj_indices = list(range(0, N, traj_step))
        traj_positions = positions[traj_indices]
    else:
        traj_positions = positions
    
    # All frames (gray line)
    fig.add_trace(
        go.Scatter3d(
            x=traj_positions[:, 0],
            y=traj_positions[:, 1],
            z=traj_positions[:, 2],
            mode='lines+markers',
            line=dict(color='lightgray', width=2),
            marker=dict(size=2, color='lightgray'),
            name='All Frames',
            showlegend=True,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Selected frames (colored by selection order)
    selected_positions = positions[selected_indices]
    fig.add_trace(
        go.Scatter3d(
            x=selected_positions[:, 0],
            y=selected_positions[:, 1],
            z=selected_positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=list(range(m)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Selection Order', x=0.46, y=0.85, len=0.3),
                line=dict(width=1, color='white')
            ),
            text=[f"Frame {idx}<br>Order: {i+1}/{m}" for i, idx in enumerate(selected_indices)],
            hoverinfo='text',
            name='Selected Frames',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # First selected frame (star marker)
    fig.add_trace(
        go.Scatter3d(
            x=[positions[selected_indices[0], 0]],
            y=[positions[selected_indices[0], 1]],
            z=[positions[selected_indices[0], 2]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            name='Start Frame',
            hovertext=f'Start: Frame {selected_indices[0]}',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # 2. Coverage quality bar chart (downsampled for large N)
    print("[2/2] Plotting coverage quality...")
    non_selected_indices = [i for i in range(N) if i not in selected_set]
    
    # Downsample bar chart for large N to reduce file size
    max_bars = 1000  # Maximum number of bars to display
    if N > max_bars:
        print(f"  → Downsampling coverage bar chart from {N} to {max_bars} bars...")
        # Keep all selected frames + sample non-selected frames
        display_indices = list(selected_indices)
        non_selected_sample = [i for i in range(0, N, max(1, N // (max_bars - len(selected_indices)))) 
                               if i not in selected_set]
        display_indices.extend(non_selected_sample[:max_bars - len(selected_indices)])
        display_indices = sorted(display_indices)
        
        colors = ['red' if i in selected_set else 'steelblue' for i in display_indices]
        fig.add_trace(
            go.Bar(
                x=display_indices,
                y=[min_distances[i] for i in display_indices],
                marker=dict(color=colors),
                hovertemplate='Frame %{x}<br>Min Dist: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
    else:
        colors = ['red' if i in selected_set else 'steelblue' for i in range(N)]
        fig.add_trace(
            go.Bar(
                x=list(range(N)),
                y=min_distances,
                marker=dict(color=colors),
                hovertemplate='Frame %{x}<br>Min Dist: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update axes
    fig.update_scenes(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data',
        row=1, col=1
    )
    
    fig.update_xaxes(title_text="Frame Index", row=1, col=2)
    fig.update_yaxes(title_text="Min Distance", row=1, col=2)
    
    # Compute and display statistics
    non_selected_min_dist = [min_distances[i] for i in non_selected_indices]
    mean_min_dist = np.mean(non_selected_min_dist) if non_selected_min_dist else 0
    max_min_dist = np.max(non_selected_min_dist) if non_selected_min_dist else 0
    std_min_dist = np.std(min_distances)
    
    stats_text = (
        f"<b>Coverage Statistics</b><br>"
        f"Selected: {m}/{N} frames ({100*m/N:.1f}%)<br>"
        f"Mean min distance: {mean_min_dist:.4f}<br>"
        f"Max min distance: {max_min_dist:.4f}<br>"
        f"Coverage uniformity (std): {std_min_dist:.4f}"
    )
    
    # Overall layout
    # Position stats_text on subplot (1,2) - Coverage Quality
    fig.update_layout(
        title_text=f'{method_name} Sampling Quality Visualization',
        width=1600,
        height=600,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        annotations=[
            dict(
                text=stats_text,
                xref="x2 domain", yref="y2 domain",  # Use domain coordinates (0-1) for subplot (1,2)
                xanchor='left', yanchor='top',
                x=0.02,  # 2% from left edge of subplot
                y=0.98,  # 98% from bottom (near top) of subplot
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=11)
            )
        ]
    )
    
    if output_path is None:
        output_path = 'tmp/fps_selection_quality.html'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(str(output_path))
    # JJ: Print absolute path for easy clicking
    abs_path = output_path.resolve()
    print(f"\n✓ FPS quality visualization saved to: {abs_path}")
    
    if show:
        fig.show()
    
    return fig


# ============================================================================
# Block 3: High-Level Visualization Wrapper (for pa_sampling.py)
# ============================================================================

def visualize_pose_sampling_results(
    all_poses: np.ndarray,
    selected_indices: List[int],
    output_dir: str,
    strategy_name: str = 'fps',
    method_name: str = 'FPS',
    distance_mode: str = 'max_norm',
    plot_pose_analysis: bool = False,
    pose_analysis_target: str = 'all',
    pose_source: str = 'VGGT Predicted',
    verbose: bool = False
):
    """
    High-level wrapper for visualizing pose sampling results.
    
    This function wraps the visualization logic from process_and_sample_scannetpp.py
    for reuse in pa_sampling.py.
    
    Args:
        all_poses: All available poses (N, 4, 4)
        selected_indices: Indices of selected frames [0, N-1]
        output_dir: Directory to save visualization files
        strategy_name: Strategy name for file naming ('fps', 'efficient', etc.)
        method_name: Method name for plot titles
        distance_mode: Distance mode for coverage analysis
        plot_pose_analysis: Whether to generate pose_analysis.html
        pose_analysis_target: 'all' or 'sampled'
        pose_source: Source of poses (e.g., 'VGGT Predicted', 'Ground Truth')
        verbose: Print progress
    
    Returns:
        dict with keys:
            - 'sampling_quality_path': path to sampling_quality.html
            - 'pose_analysis_path': path to pose_analysis.html (if generated)
            - 'coverage': coverage statistics dict
    """
    import sys
    from pathlib import Path
    
    # Lazy import to avoid circular dependency
    sys.path.append(str(Path(__file__).resolve().parents[1] / 'utils'))
    try:
        from src.utils import pose_fps_sampling
    except ImportError:
        import pose_fps_sampling
    
    if verbose:
        print(f"\nGenerating visualization...")
        print(f"  all_poses: {all_poses.shape}")
        print(f"  selected_indices: {len(selected_indices)} frames")
    
    # Compute distance matrix
    distance_matrix = pose_fps_sampling.compute_pairwise_pose_distance(
        all_poses,
        distance_mode=distance_mode,
        reorth_rot=True
    )
    
    # Compute coverage statistics
    coverage = pose_fps_sampling.analyze_pose_coverage(
        all_poses,
        selected_indices,
        distance_mode=distance_mode,
        reorth_rot=True
    )
    
    # Prepare data for pose_analysis
    if plot_pose_analysis:
        if pose_analysis_target == 'all':
            # Use all poses
            analysis_poses = all_poses
            analysis_selected_indices = selected_indices
            analysis_frame_ids = list(range(len(all_poses)))
        elif pose_analysis_target == 'sampled':
            # Use only sampled frames, sorted by index
            sorted_order = sorted(range(len(selected_indices)), 
                                key=lambda i: selected_indices[i])
            analysis_poses = all_poses[selected_indices][sorted_order]
            analysis_frame_ids = [selected_indices[i] for i in sorted_order]
            analysis_selected_indices = list(range(len(selected_indices)))
        else:
            raise ValueError(f"Invalid pose_analysis_target: {pose_analysis_target}")
    else:
        analysis_poses = None
        analysis_selected_indices = None
        analysis_frame_ids = None
    
    # Generate visualization
    output_path = Path(output_dir) / f"sampling_quality_{strategy_name}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        visualize_sampled_pose(
            pose_analysis_poses=all_poses,
            selected_indices=selected_indices,
            min_distances=coverage['min_distances'],
            distance_matrix=distance_matrix,
            output_path=str(output_path),
            show=False,
            method_name=method_name,
            plot_pose_analysis=plot_pose_analysis,
            pose_analysis_selected=analysis_selected_indices,
            pose_analysis_target_poses=analysis_poses,
            pose_analysis_frame_ids=analysis_frame_ids,
            pose_source=pose_source
        )
        
        result = {
            'sampling_quality_path': str(output_path),
            'coverage': coverage
        }
        
        if plot_pose_analysis:
            pose_analysis_path = output_path.parent / 'pose_analysis.html'
            result['pose_analysis_path'] = str(pose_analysis_path)
        
        if verbose:
            print(f"  ✓ Visualization saved to: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"  Warning: Failed to generate visualization: {e}")
        return None


# JJ: Test trajectories generation and visualization
if __name__ == '__main__':
    from scipy.spatial.transform import Rotation as R
    import sys
    import os
    import json
    import argparse
    
    # Import from same directory
    try:
        from .pose_distance_metrics import compute_pose_farness
    except ImportError:
        from pose_distance_metrics import compute_pose_farness
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test visualization with synthetic or real trajectories")
    parser.add_argument('--mode', type=str, choices=['synthetic', 'real'], default='synthetic',
                      help="Test mode: 'synthetic' for generated trajectories, 'real' for ScanNet++ data")
    parser.add_argument('--json_path', type=str, 
                      default="/mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialMllmHallucinate/third_party/Spatial-MLLM/datasets/scannetpp/pa_sampling_good/00777c41d4/selected_frames.json",
                      help="Path to ScanNet++ selected_frames.json for real data testing")
    args = parser.parse_args()
    
    if args.mode == 'real':
        # JJ: Test with real ScanNet++ data
        print("\n" + "=" * 80)
        print("Testing Visualization with Real ScanNet++ Data")
        print("=" * 80)
        
        print(f"\nLoading poses from: {args.json_path}")
        with open(args.json_path, 'r') as f:
            data = json.load(f)
        
        # Extract pose matrices
        pose_matrices = [frame['pose_matrix'] for frame in data['transform_matrix']]
        poses_numpy = [np.array(pm) for pm in pose_matrices]
        frame_indices = data['selected_frames']
        
        print(f"Loaded {len(poses_numpy)} poses")
        print(f"Scene: {data['scene_name']}")
        print(f"Selected frames: {data['num_frames']}/{data['total_frames']}")
        
        # Compute farness scores
        print("\nComputing farness scores...")
        farness_trans, farness_rot = compute_pose_farness(
            poses_c2w=poses_numpy,
            trans_metric_mode='euclidean',
            rot_metric_mode='angle_axis',
            reorth_rot=True,
            translation_scale=1.0
        )
        
        print(f"\nStatistics:")
        print(f"  Translation (normalized) - min: {min(farness_trans):.4f}, max: {max(farness_trans):.4f}, mean: {np.mean(farness_trans):.4f}")
        print(f"  Rotation - min: {np.degrees(min(farness_rot)):.2f}°, max: {np.degrees(max(farness_rot)):.2f}°, mean: {np.degrees(np.mean(farness_rot)):.2f}°")
        
        # Generate visualization
        from pathlib import Path
        output_path = Path(args.json_path).parent / "pose_analysis.html"
        
        visualize_all(
            poses_c2w=poses_numpy,
            farness_trans=farness_trans,
            farness_rot=farness_rot,
            frame_indices=frame_indices,
            output_path=str(output_path),
            show=False,
            translation_unit='normalized'
        )
        
        # JJ: Test Lie scalar index computation
        print("\n" + "=" * 80)
        print("Testing Lie Scalar Index Computation")
        print("=" * 80)
        
        import torch
        from pose_distance_metrics import compute_lie_scalar_index_torch
        
        # Convert to torch tensor
        poses_array = np.array(poses_numpy)
        poses_tensor = torch.from_numpy(poses_array).float()
        
        # Test with different pose_id_scalar_lambda_trans values
        for pose_id_scalar_lambda_trans in [0.1, 0.5, 1.0, 2.0, 5.0]:
            P = compute_lie_scalar_index_torch(
                poses_c2w=poses_tensor,
                pose_id_scalar_lambda_trans=pose_id_scalar_lambda_trans,
                traj_scale_norm=True,
                global_normalize=True,
                reorth_rot=True
            )
            
            print(f"\npose_id_scalar_lambda_trans = {pose_id_scalar_lambda_trans}:")
            print(f"  P shape: {P.shape}")
            print(f"  P range: [{P.min().item():.4f}, {P.max().item():.4f}]")
            print(f"  P mean: {P.mean().item():.4f}, std: {P.std().item():.4f}")
            print(f"  Values: {[f'{v:.4f}' for v in P.tolist()]}")
        
        print("\n" + "=" * 80)
        print("✓ Real data visualization completed!")
        print("=" * 80)
        
    else:
        # JJ: Test with synthetic trajectories
        print("\n" + "=" * 80)
        print("Generating Test Trajectories")
        print("=" * 80)
        
        # Import trajectory generators
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
        
        # Generate trajectories
        print("\n[1/3] Generating Trajectory 1: Loop Closure...")
        traj1_poses = generate_traj1_loop_closure(100)
        farness_trans_1, farness_rot_1 = compute_pose_farness(
            traj1_poses, 
            translation_scale=None  # Use raw distances
        )
        print(f"  → Generated {len(traj1_poses)} frames")
        print(f"  → Trans range: [{min(farness_trans_1):.3f}, {max(farness_trans_1):.3f}]")
        print(f"  → Rot range: [{np.degrees(min(farness_rot_1)):.1f}°, {np.degrees(max(farness_rot_1)):.1f}°]")
        
        print("\n[2/3] Generating Trajectory 2: Linear Translation with Minor Rotation...")
        traj2_poses = generate_traj2_linear_translation(100)
        farness_trans_2, farness_rot_2 = compute_pose_farness(
            traj2_poses,
            translation_scale=None
        )
        print(f"  → Generated {len(traj2_poses)} frames")
        print(f"  → Trans range: [{min(farness_trans_2):.3f}, {max(farness_trans_2):.3f}]")
        print(f"  → Rot range: [{np.degrees(min(farness_rot_2)):.1f}°, {np.degrees(max(farness_rot_2)):.1f}°]")
        
        print("\n[3/3] Generating Trajectory 3: Rotation then Translation...")
        traj3_poses = generate_traj3_rotation_then_translation(100)
        farness_trans_3, farness_rot_3 = compute_pose_farness(
            traj3_poses,
            translation_scale=None
        )
        print(f"  → Generated {len(traj3_poses)} frames")
        print(f"  → Trans range: [{min(farness_trans_3):.3f}, {max(farness_trans_3):.3f}]")
        print(f"  → Rot range: [{np.degrees(min(farness_rot_3)):.1f}°, {np.degrees(max(farness_rot_3)):.1f}°]")
        
        # Visualize and save
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)
        
        visualize_all(
            traj1_poses,
            farness_trans_1,
            farness_rot_1,
            output_path='tmp/traj1_loop_closure.html',
            show=False,
            translation_unit='m'
        )
        
        visualize_all(
            traj2_poses,
            farness_trans_2,
            farness_rot_2,
            output_path='tmp/traj2_linear_trans.html',
            show=False,
            translation_unit='m'
        )
        
        visualize_all(
            traj3_poses,
            farness_trans_3,
            farness_rot_3,
            output_path='tmp/traj3_fixed_rot.html',
            show=False,
            translation_unit='m'
        )
        
        # JJ: Test Lie scalar index computation for synthetic trajectories
        print("\n" + "=" * 80)
        print("Testing Lie Scalar Index for Synthetic Trajectories")
        print("=" * 80)
        
        import torch
        from pose_distance_metrics import compute_lie_scalar_index_torch
        
        # Test all 3 trajectories
        trajectories = [
            ("Trajectory 1: Loop Closure", traj1_poses),
            ("Trajectory 2: Linear Translation", traj2_poses),
            ("Trajectory 3: Rotation then Translation", traj3_poses)
        ]
        
        for traj_name, traj_poses in trajectories:
            print(f"\n[{traj_name}]")
            traj_array = np.array(traj_poses)
            traj_tensor = torch.from_numpy(traj_array).float()
            
            for pose_id_scalar_lambda_trans in [0.1, 0.5, 1.0, 2.0, 5.0]:
                P = compute_lie_scalar_index_torch(
                    poses_c2w=traj_tensor,
                    pose_id_scalar_lambda_trans=pose_id_scalar_lambda_trans,
                    traj_scale_norm=True,
                    global_normalize=True,
                    reorth_rot=True
                )
                print(f"\n  pose_id_scalar_lambda_trans = {pose_id_scalar_lambda_trans}:")
                print(f"    P range: [{P.min().item():.4f}, {P.max().item():.4f}]")
                print(f"    P mean: {P.mean().item():.4f}, std: {P.std().item():.4f}")
                print(f"    First 10: {[f'{v:.4f}' for v in P[:].tolist()]}")
        
        print("\n" + "=" * 80)
        print("✓ All synthetic trajectories generated successfully!")
        print("=" * 80)
        print("\nSaved files:")
        print("  - traj1_loop_closure.html")
        print("  - traj2_linear_trans.html")
        print("  - traj3_fixed_rot.html")
        print()
