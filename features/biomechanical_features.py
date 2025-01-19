import torch
from typing import Dict, List, Tuple

def calculate_vector(point1, point2):
    """
    Calculate vector between two 3D points
    """
    return point2 - point1

def calculate_angle(vector1, vector2):
    """
    Calculate angle between two 3D vectors using dot product
    Returns angle in degrees
    """
    dot_product = torch.dot(vector1, vector2)
    norms = torch.norm(vector1) * torch.norm(vector2)
    
    # Handle numerical stability
    cos_angle = torch.clamp(dot_product / norms, -1.0, 1.0)
    angle_rad = torch.acos(cos_angle)
    return torch.rad2deg(angle_rad)

def calculate_projected_angle(vector1, vector2, normal):
    """
    Calculate angle between two vectors when projected onto a plane
    defined by its normal vector
    """
    # Project vectors onto the plane
    proj1 = vector1 - torch.dot(vector1, normal) * normal
    proj2 = vector2 - torch.dot(vector2, normal) * normal
    
    return calculate_angle(proj1, proj2)

def calculate_joint_angles(poses):
    """
    Calculate relevant joint angles from pose data
    Input: poses - torch tensor of shape (frames, 33, 3)
    Output: dictionary of joint angles
    """
    joint_configs = {
        # Upper body
        'right_shoulder': {
            'joints': (13, 11, 23),  # right_elbow, right_shoulder, right_hip
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'left_shoulder': {
            'joints': (14, 12, 24),  # left_elbow, left_shoulder, left_hip
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'right_elbow': {
            'joints': (11, 13, 15),  # right_shoulder, right_elbow, right_wrist
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'left_elbow': {
            'joints': (12, 14, 16),  # left_shoulder, left_elbow, left_wrist
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        
        # Lower body
        'right_hip': {
            'joints': (11, 23, 25),  # right_shoulder, right_hip, right_knee
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'left_hip': {
            'joints': (12, 24, 26),  # left_shoulder, left_hip, left_knee
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'right_knee': {
            'joints': (23, 25, 27),  # right_hip, right_knee, right_ankle
            'planes': ['sagittal', 'transverse', 'frontal']  # Now including transverse
        },
        'left_knee': {
            'joints': (24, 26, 28),  # left_hip, left_knee, left_ankle
            'planes': ['sagittal', 'transverse', 'frontal']  # Now including transverse
        },
        'right_ankle': {
            'joints': (25, 27, 31),  # right_knee, right_ankle, right_foot_index
            'planes': ['sagittal', 'transverse', 'frontal']
        },
        'left_ankle': {
            'joints': (26, 28, 32),  # left_knee, left_ankle, left_foot_index
            'planes': ['sagittal', 'transverse', 'frontal']
        }
    }
    num_frames = poses.shape[0]
    angles = {}
    angles_tensor = torch.zeros((num_frames,len(joint_configs),1 ))
    # Define joint triplets for angle calculation
    
    # Define anatomical planes using normal vectors
    planes = {
        'sagittal': torch.tensor([1, 0, 0]),  # Left-right axis (flexion/extension)
        'frontal': torch.tensor([0, 0, 1]),   # Forward-backward axis (abduction/adduction)
        'transverse': torch.tensor([0, 1, 0])  # Up-down axis (internal/external rotation)
    }
    
    # Calculate angles for each frame
    for frame in range(num_frames):
        frame_angles = {}
        
        for j, (joint_name, config) in enumerate(joint_configs.items()):
            j1, j2, j3 = config['joints']
            
            # Calculate vectors
            vector1 = calculate_vector(poses[frame, j2], poses[frame, j1])
            vector2 = calculate_vector(poses[frame, j2], poses[frame, j3])
            
            # Calculate 3D angle
            computed_angle = calculate_angle(vector1, vector2)
            frame_angles[f"{joint_name}_3d"] = computed_angle 
            angles_tensor[frame, j,0] =  computed_angle
            # # Calculate projected angles on anatomical planes
            # for plane_name in config['planes']:
            #     normal = planes[plane_name]
            #     projected_angle = calculate_projected_angle(vector1, vector2, normal)
            #     frame_angles[f"{joint_name}_{plane_name}"] = projected_angle
        
        angles[frame] = frame_angles
    
    return angles, angles_tensor


class BiomechanicalFeatureExtractor:
    def __init__(self):
        """
        Initialize the feature extractor.
        Note: Simplified version using frame-by-frame differences
        """
        pass
    
    def compute_velocities(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute joint velocities from positions using simple frame differences.
        
        Args:
            joint_positions: Tensor of shape (frames, num_joints, 3)
            
        Returns:
            Tensor of shape (frames, num_joints, 3) containing velocities
        """
        # Initialize velocities tensor with zeros
        velocities = torch.zeros_like(joint_positions)
        
        # Compute velocities as simple differences between consecutive frames
        velocities[:-1] = joint_positions[1:] - joint_positions[:-1]
        
        # For the last frame, use the same velocity as the second-to-last frame
        velocities[-1] = velocities[-2]
        
        return velocities
    
    def compute_accelerations(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Compute joint accelerations from velocities using simple frame differences.
        
        Args:
            velocities: Tensor of shape (frames, num_joints, 3)
            
        Returns:
            Tensor of shape (frames, num_joints, 3) containing accelerations
        """
        # Initialize accelerations tensor with zeros
        accelerations = torch.zeros_like(velocities)
        
        # Compute accelerations as simple differences between consecutive velocities
        accelerations[:-1] = velocities[1:] - velocities[:-1]
        
        # For the last frame, use the same acceleration as the second-to-last frame
        accelerations[-1] = accelerations[-2]
        
        return accelerations
    

    def compute_joint_angles(
        self, 
        joint_positions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        joint_angles, angles_tensor = calculate_joint_angles(joint_positions)
        return joint_angles, angles_tensor
    
    def extract_features(
        self, 
        joint_positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all biomechanical features from joint positions.
        
        Args:
            joint_positions: Tensor of shape (frames, num_joints, 3)
        Returns:
            Dictionary containing all computed features
        """
        velocities = self.compute_velocities(joint_positions)
        accelerations = self.compute_accelerations(velocities)
        angles_dict, angles = self.compute_joint_angles(joint_positions)# Access Angles Dict if you need to know which scalar value corresponds to which angle

        return {
            "Joint Position": joint_positions,
            "Joint Angles": angles,
            "Joint Velocity": velocities,
            "Joint Acceleration": accelerations
        }
    
    @staticmethod
    def save_features(features: Dict[str, torch.Tensor], filename: str):
        """Save features to a .pt file."""
        torch.save(features, filename)
        print(f"saving {filename}")
    
    @staticmethod
    def load_features(filename: str) -> Dict[str, torch.Tensor]:
        """Load features from a .pt file."""
        return torch.load(f"{filename}.pt")
