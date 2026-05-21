import numpy as np
from typing import List, Dict, Tuple, Optional


class HandKeypointProcessor:
    def __init__(self, num_keypoints: int = 21):
        self.num_keypoints = num_keypoints
        
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        if len(keypoints.shape) == 2:
            return self._normalize_single_frame(keypoints)
        elif len(keypoints.shape) == 3:
            return np.array([self._normalize_single_frame(frame) for frame in keypoints])
        else:
            raise ValueError(f"Expected 2D or 3D array, got {len(keypoints.shape)}D")

    def _normalize_single_frame(self, frame: np.ndarray) -> np.ndarray:
        wrist = frame[0]
        
        centered = frame - wrist
        
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered
        
        return normalized

    def extract_features(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        if len(keypoints_sequence.shape) == 2:
            keypoints_sequence = keypoints_sequence[np.newaxis, ...]
        
        normalized = self.normalize_keypoints(keypoints_sequence)
        
        features_list = []
        for frame in normalized:
            frame_features = self._extract_frame_features(frame)
            features_list.append(frame_features)
        
        return np.array(features_list)

    def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        features = []
        
        features.extend(frame.flatten())
        
        distances = self._calculate_inter_keypoint_distances(frame)
        features.extend(distances)
        
        angles = self._calculate_joint_angles(frame)
        features.extend(angles)
        
        return np.array(features)

    def _calculate_inter_keypoint_distances(self, frame: np.ndarray) -> List[float]:
        distances = []
        for i in range(self.num_keypoints):
            for j in range(i + 1, self.num_keypoints):
                dist = np.linalg.norm(frame[i] - frame[j])
                distances.append(dist)
        return distances

    def _calculate_joint_angles(self, frame: np.ndarray) -> List[float]:
        angles = []
        
        finger_joints = [
            [0, 1, 2], [1, 2, 3], [2, 3, 4],
            [0, 5, 6], [5, 6, 7], [6, 7, 8],
            [0, 9, 10], [9, 10, 11], [10, 11, 12],
            [0, 13, 14], [13, 14, 15], [14, 15, 16],
            [0, 17, 18], [17, 18, 19], [18, 19, 20]
        ]
        
        for joint in finger_joints:
            p1, p2, p3 = frame[joint]
            v1 = p1 - p2
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
            else:
                angle = 0.0
            
            angles.append(angle)
        
        return angles

    def compute_velocity(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        if len(keypoints_sequence) < 2:
            return np.zeros_like(keypoints_sequence)
        
        velocities = np.diff(keypoints_sequence, axis=0)
        velocities = np.vstack([velocities, velocities[-1]])
        return velocities

    def compute_acceleration(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        velocities = self.compute_velocity(keypoints_sequence)
        accelerations = np.diff(velocities, axis=0)
        accelerations = np.vstack([accelerations, accelerations[-1]])
        return accelerations

    def standardize_features(self, features: np.ndarray, 
                            mean: Optional[np.ndarray] = None, 
                            std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mean is None or std is None:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std = np.where(std == 0, 1.0, std)
        
        standardized = (features - mean) / std
        return standardized, mean, std

    def calculate_keypoint_weights(self, 
                                   keypoints_sequence: np.ndarray,
                                   weight_type: str = 'variance') -> np.ndarray:
        n_frames = len(keypoints_sequence)
        n_keypoints = self.num_keypoints
        
        variances = np.zeros(n_keypoints)
        for k in range(n_keypoints):
            keypoint_positions = keypoints_sequence[:, k, :]
            variances[k] = np.mean(np.var(keypoint_positions, axis=0))
        
        if weight_type == 'variance':
            weights = variances / np.sum(variances)
        elif weight_type == 'inverse_variance':
            weights = 1.0 / (variances + 1e-8)
            weights = weights / np.sum(weights)
        elif weight_type == 'uniform':
            weights = np.ones(n_keypoints) / n_keypoints
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
        
        return weights

    def calculate_feature_dim_weights(self, features: np.ndarray,
                                     weight_type: str = 'variance') -> np.ndarray:
        if weight_type == 'variance':
            variances = np.var(features, axis=0)
            weights = variances / np.sum(variances)
        elif weight_type == 'inverse_variance':
            variances = np.var(features, axis=0)
            weights = 1.0 / (variances + 1e-8)
            weights = weights / np.sum(weights)
        elif weight_type == 'uniform':
            weights = np.ones(features.shape[1]) / features.shape[1]
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
        
        return weights

    def expand_keypoint_weights_to_features(self, keypoint_weights: np.ndarray,
                                          feature_dim: int) -> np.ndarray:
        n_keypoints = len(keypoint_weights)
        coords_per_keypoint = feature_dim // n_keypoints
        
        expanded_weights = np.repeat(keypoint_weights, coords_per_keypoint)
        
        if len(expanded_weights) < feature_dim:
            remaining = feature_dim - len(expanded_weights)
            avg_weight = np.mean(keypoint_weights)
            expanded_weights = np.pad(expanded_weights, (0, remaining), 
                                    mode='constant', constant_values=avg_weight)
        
        return expanded_weights


class GestureDataAugmenter:
    def __init__(self):
        pass

    def time_warp(self, sequence: np.ndarray, warp_factor: float = 0.2) -> np.ndarray:
        n_frames = len(sequence)
        n_features = sequence.shape[1]
        
        warp_amount = int(n_frames * warp_factor)
        if warp_amount == 0:
            return sequence
        
        indices = np.arange(n_frames)
        
        for _ in range(warp_amount):
            if np.random.random() > 0.5:
                insert_idx = np.random.randint(1, n_frames - 1)
                indices = np.insert(indices, insert_idx, insert_idx)
            else:
                remove_idx = np.random.randint(1, n_frames - 1)
                indices = np.delete(indices, remove_idx)
        
        if len(indices) != n_frames:
            indices = np.linspace(0, len(indices) - 1, n_frames, dtype=int)
        
        return sequence[indices]

    def add_noise(self, sequence: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise

    def random_scale(self, sequence: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        scale_factor = np.random.uniform(*scale_range)
        return sequence * scale_factor

    def random_rotation(self, sequence: np.ndarray, max_angle: float = np.pi / 6) -> np.ndarray:
        angle = np.random.uniform(-max_angle, max_angle)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        rotated = np.dot(sequence, rotation_matrix)
        return rotated

    def augment_sequence(self, sequence: np.ndarray, 
                         num_augments: int = 5) -> List[np.ndarray]:
        augmented_sequences = [sequence]
        
        for _ in range(num_augments):
            aug_seq = sequence.copy()
            
            if np.random.random() > 0.5:
                aug_seq = self.time_warp(aug_seq)
            if np.random.random() > 0.5:
                aug_seq = self.add_noise(aug_seq)
            if np.random.random() > 0.5:
                aug_seq = self.random_scale(aug_seq)
            if np.random.random() > 0.5 and aug_seq.shape[-1] == 3:
                aug_seq = self.random_rotation(aug_seq)
            
            augmented_sequences.append(aug_seq)
        
        return augmented_sequences


def smooth_sequence(sequence: np.ndarray, window_size: int = 3) -> np.ndarray:
    if window_size < 2 or len(sequence) < window_size:
        return sequence
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.zeros_like(sequence)
    
    for i in range(sequence.shape[1]):
        smoothed[:, i] = np.convolve(sequence[:, i], kernel, mode='same')
    
    return smoothed


def sample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return sequence[indices]