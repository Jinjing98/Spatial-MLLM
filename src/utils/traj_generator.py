# JJ: Synthetic trajectory generators for testing pose sampling algorithms
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List


def generate_traj1_loop_closure(n_frames=100) -> List[np.ndarray]:
    """
    Trajectory 1: Realistic loop closure with multiple revisits
    - 0-30: Move in +X direction
    - 30-50: Turn and move in +Y direction  
    - 50-65: First loop closure - return near start (with offset)
    - 65-85: Move to new area (-X direction)
    - 85-100: Second loop closure - revisit middle section
    
    Returns:
        List of 4x4 transformation matrices (camera-to-world)
    """
    poses = []
    waypoints = []
    
    # Define waypoints for the trajectory
    # Segment 1: frames 0-30, move along +X
    for i in range(31):
        t = i / 30
        x = t * 4.0  # Move 4 units in X
        y = np.random.uniform(-0.2, 0.2)  # Small random drift
        z = 1.5 + np.random.uniform(-0.1, 0.1)
        waypoints.append(np.array([x, y, z]))
    
    # Segment 2: frames 30-50, turn and move in +Y
    for i in range(20):
        t = i / 19
        x = 4.0 + np.random.uniform(-0.2, 0.2)
        y = t * 3.0  # Move 3 units in Y
        z = 1.5 + np.random.uniform(-0.1, 0.1)
        waypoints.append(np.array([x, y, z]))
    
    # Segment 3: frames 50-65, FIRST LOOP CLOSURE - return near start
    # Not perfect closure, offset by 0.4-0.6m
    for i in range(15):
        t = i / 14
        # Interpolate back towards start with offset
        x = 4.0 * (1 - t) + 0.5  # Return to X≈0.5 (not perfect 0)
        y = 3.0 * (1 - t) + 0.4  # Return to Y≈0.4 (not perfect 0)
        z = 1.5 + np.random.uniform(-0.1, 0.1)
        waypoints.append(np.array([x, y, z]))
    
    # Segment 4: frames 65-85, move to new area in -X direction
    for i in range(20):
        t = i / 19
        x = 0.5 - t * 3.5  # Move to X≈-3
        y = 0.4 + t * 2.0  # Also move in Y
        z = 1.5 + np.random.uniform(-0.1, 0.1)
        waypoints.append(np.array([x, y, z]))
    
    # Segment 5: frames 85-100, SECOND LOOP CLOSURE - revisit middle section
    # Return near frames 30-40 region (around X=4, Y=1)
    for i in range(15):
        t = i / 14
        x = -3.0 + t * 6.5  # Move back to X≈3.5
        y = 2.4 - t * 1.2   # Move to Y≈1.2
        z = 1.5 + np.random.uniform(-0.1, 0.1)
        waypoints.append(np.array([x, y, z]))
    
    # Generate poses from waypoints
    for i, camera_pos in enumerate(waypoints):
        # Generate rotation with some randomness
        if i == 0:
            # First frame: look forward
            yaw = 0
        else:
            # Look in the direction of motion with some noise
            motion = waypoints[i] - waypoints[i-1]
            if np.linalg.norm(motion[:2]) > 0.01:
                yaw = np.arctan2(motion[1], motion[0])
                yaw += np.random.uniform(-0.3, 0.3)  # Add noise
            else:
                # Keep previous yaw
                prev_rot = R.from_matrix(poses[-1][:3, :3])
                prev_euler = prev_rot.as_euler('zyx', degrees=True)
                yaw = np.radians(prev_euler[0])
        
        # Small pitch and roll variations (hand-held camera)
        pitch = np.random.uniform(-5, 5)
        roll = np.random.uniform(-3, 3)
        
        rotation = R.from_euler('zyx', [np.degrees(yaw), pitch, roll], degrees=True)
        R_mat = rotation.as_matrix()
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = camera_pos
        
        poses.append(T)
    
    return poses


def generate_traj2_linear_translation(n_frames=100) -> List[np.ndarray]:
    """
    Trajectory 2: Linear translation with minor rotation
    - Translation: move far linearly along one direction
    - Rotation: only small random variations
    
    Returns:
        List of 4x4 transformation matrices (camera-to-world)
    """
    poses = []
    
    # Base rotation with small random variations
    base_rotation = R.from_euler('xyz', [0, 0, 0], degrees=True)
    
    for i in range(n_frames):
        # Linear translation along x-axis
        t = i * 0.15  # Move 0.15 units per frame
        translation = np.array([t, 0, 1.5])
        
        # Small random rotation variations (±5 degrees)
        random_angles = np.random.uniform(-5, 5, 3)
        rotation = R.from_euler('xyz', random_angles, degrees=True) * base_rotation
        R_mat = rotation.as_matrix()
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = translation
        
        poses.append(T)
    
    return poses


def generate_traj3_rotation_then_translation(n_frames=100) -> List[np.ndarray]:
    """
    Trajectory 3: Fixed position with large rotation, then translation
    - First 70 frames: stay in small area, do large hand-held rotations
    - Last 30 frames: move linearly far away
    
    Returns:
        List of 4x4 transformation matrices (camera-to-world)
    """
    poses = []
    
    # Small area center
    base_pos = np.array([0, 0, 1.5])
    
    for i in range(n_frames):
        if i < 70:
            # First 70 frames: stay in small area with large rotation variations
            # Small random position jitter (±0.1 units)
            pos_jitter = np.random.uniform(-0.1, 0.1, 3)
            translation = base_pos + pos_jitter
            
            # Large random rotations (hand-held camera motion)
            # Each frame gets significantly different rotation
            pitch = np.random.uniform(-30, 30)  # Look up/down
            yaw = np.random.uniform(-45, 45)    # Look left/right
            roll = np.random.uniform(-15, 15)   # Tilt
            
            rotation = R.from_euler('xyz', [pitch, yaw, roll], degrees=True)
            R_mat = rotation.as_matrix()
        else:
            # Last 30 frames: move far linearly
            progress = (i - 70) / 30
            translation = base_pos + np.array([progress * 10, progress * 5, 0])
            
            # Keep rotation more stable during translation
            rotation = R.from_euler('xyz', [0, 0, 0], degrees=True)
            R_mat = rotation.as_matrix()
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = translation
        
        poses.append(T)
    
    return poses
