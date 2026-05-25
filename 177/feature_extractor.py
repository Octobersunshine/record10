import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from trajectory_manager import Trajectory3D, TrajectoryPoint
from config import Config


@dataclass
class TrajectoryFeatures:
    track_id: int
    timestamps: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    speeds: np.ndarray
    accelerations: np.ndarray
    accelerations_magnitude: np.ndarray
    curvatures: np.ndarray
    turning_angles: np.ndarray
    turning_rates: np.ndarray
    angular_velocities: np.ndarray
    path_length: float
    displacement: float
    straightness: float
    mean_speed: float
    max_speed: float
    mean_acceleration: float
    max_acceleration: float
    mean_turning_rate: float
    max_turning_rate: float
    feature_matrix: np.ndarray


class FeatureExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.smoothing_window = config.get('SMOOTH_WINDOW_SIZE', 5)
        self.min_trajectory_length = config.get('MIN_TRAJECTORY_LENGTH', 10)
        
    def extract_features(self, trajectory: Trajectory3D) -> Optional[TrajectoryFeatures]:
        if len(trajectory.points) < self.min_trajectory_length:
            return None
        
        positions = trajectory.get_position_array()
        times = trajectory.get_time_array()
        
        if len(positions) < 3:
            return None
        
        positions = self._smooth_positions(positions)
        
        velocities = self._compute_velocities(positions, times)
        speeds = np.linalg.norm(velocities, axis=1)
        
        accelerations = self._compute_accelerations(velocities, times)
        accelerations_magnitude = np.linalg.norm(accelerations, axis=1)
        
        curvatures = self._compute_curvatures(positions)
        
        turning_angles = self._compute_turning_angles(positions)
        
        turning_rates = self._compute_turning_rates(turning_angles, times)
        
        angular_velocities = self._compute_angular_velocities(velocities, times)
        
        path_length = self._compute_path_length(positions)
        
        displacement = np.linalg.norm(positions[-1] - positions[0])
        
        straightness = displacement / path_length if path_length > 0 else 0
        
        feature_matrix = self._build_feature_matrix(
            speeds, accelerations_magnitude, curvatures, 
            turning_rates, angular_velocities
        )
        
        return TrajectoryFeatures(
            track_id=trajectory.track_id,
            timestamps=times,
            positions=positions,
            velocities=velocities,
            speeds=speeds,
            accelerations=accelerations,
            accelerations_magnitude=accelerations_magnitude,
            curvatures=curvatures,
            turning_angles=turning_angles,
            turning_rates=turning_rates,
            angular_velocities=angular_velocities,
            path_length=path_length,
            displacement=displacement,
            straightness=straightness,
            mean_speed=np.mean(speeds),
            max_speed=np.max(speeds),
            mean_acceleration=np.mean(accelerations_magnitude),
            max_acceleration=np.max(accelerations_magnitude),
            mean_turning_rate=np.mean(turning_rates),
            max_turning_rate=np.max(turning_rates),
            feature_matrix=feature_matrix
        )
    
    def _smooth_positions(self, positions: np.ndarray) -> np.ndarray:
        if self.smoothing_window <= 1:
            return positions
        
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        
        smoothed = np.zeros_like(positions)
        for i in range(3):
            smoothed[:, i] = np.convolve(positions[:, i], kernel, mode='same')
        
        return smoothed
    
    def _compute_velocities(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        velocities = np.zeros_like(positions)
        
        for i in range(1, len(positions)):
            dt = times[i] - times[i-1]
            if dt > 0:
                velocities[i] = (positions[i] - positions[i-1]) / dt
        
        velocities[0] = velocities[1] if len(velocities) > 1 else 0
        
        return velocities
    
    def _compute_accelerations(self, velocities: np.ndarray, times: np.ndarray) -> np.ndarray:
        accelerations = np.zeros_like(velocities)
        
        for i in range(1, len(velocities)):
            dt = times[i] - times[i-1]
            if dt > 0:
                accelerations[i] = (velocities[i] - velocities[i-1]) / dt
        
        accelerations[0] = accelerations[1] if len(accelerations) > 1 else 0
        
        return accelerations
    
    def _compute_curvatures(self, positions: np.ndarray) -> np.ndarray:
        n = len(positions)
        curvatures = np.zeros(n)
        
        for i in range(1, n - 1):
            p1 = positions[i-1]
            p2 = positions[i]
            p3 = positions[i+1]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            cross = np.cross(v1, v2)
            cross_mag = np.linalg.norm(cross)
            
            v1_mag = np.linalg.norm(v1)
            v2_mag = np.linalg.norm(v2)
            
            denom = v1_mag * v2_mag * (v1_mag + v2_mag)
            
            if denom > 1e-10:
                curvatures[i] = 2 * cross_mag / denom
        
        curvatures[0] = curvatures[1] if n > 1 else 0
        curvatures[-1] = curvatures[-2] if n > 1 else 0
        
        return curvatures
    
    def _compute_turning_angles(self, positions: np.ndarray) -> np.ndarray:
        n = len(positions)
        angles = np.zeros(n)
        
        for i in range(1, n - 1):
            v1 = positions[i] - positions[i-1]
            v2 = positions[i+1] - positions[i]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-10 and v2_norm > 1e-10:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angles[i] = np.arccos(cos_angle)
        
        angles[0] = angles[1] if n > 1 else 0
        angles[-1] = angles[-2] if n > 1 else 0
        
        return angles
    
    def _compute_turning_rates(self, angles: np.ndarray, times: np.ndarray) -> np.ndarray:
        n = len(angles)
        rates = np.zeros(n)
        
        for i in range(1, n):
            dt = times[i] - times[i-1]
            if dt > 0:
                rates[i] = abs(angles[i] - angles[i-1]) / dt
        
        rates[0] = rates[1] if n > 1 else 0
        
        return rates
    
    def _compute_angular_velocities(self, velocities: np.ndarray, 
                                     times: np.ndarray) -> np.ndarray:
        n = len(velocities)
        angular_vel = np.zeros(n)
        
        for i in range(1, n):
            v1 = velocities[i-1]
            v2 = velocities[i]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-10 and v2_norm > 1e-10:
                cross = np.cross(v1, v2)
                dot = np.dot(v1, v2)
                angle = np.arctan2(np.linalg.norm(cross), dot)
                
                dt = times[i] - times[i-1]
                if dt > 0:
                    angular_vel[i] = angle / dt
        
        angular_vel[0] = angular_vel[1] if n > 1 else 0
        
        return angular_vel
    
    def _compute_path_length(self, positions: np.ndarray) -> float:
        if len(positions) < 2:
            return 0.0
        
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)
    
    def _build_feature_matrix(self, speeds: np.ndarray, 
                              accelerations: np.ndarray,
                              curvatures: np.ndarray, 
                              turning_rates: np.ndarray,
                              angular_velocities: np.ndarray) -> np.ndarray:
        return np.column_stack([
            speeds,
            accelerations,
            curvatures,
            turning_rates,
            angular_velocities
        ])
    
    def extract_window_features(self, trajectory: Trajectory3D, 
                                window_size: int = 30,
                                step_size: int = 10) -> List[np.ndarray]:
        if len(trajectory.points) < window_size:
            features = self.extract_features(trajectory)
            return [features.feature_matrix] if features else []
        
        window_features = []
        positions = trajectory.get_position_array()
        times = trajectory.get_time_array()
        
        for start in range(0, len(positions) - window_size + 1, step_size):
            end = start + window_size
            
            window_positions = positions[start:end]
            window_times = times[start:end]
            
            velocities = self._compute_velocities(window_positions, window_times)
            speeds = np.linalg.norm(velocities, axis=1)
            
            accelerations = self._compute_accelerations(velocities, window_times)
            acc_magnitude = np.linalg.norm(accelerations, axis=1)
            
            curvatures = self._compute_curvatures(window_positions)
            turning_angles = self._compute_turning_angles(window_positions)
            turning_rates = self._compute_turning_rates(turning_angles, window_times)
            angular_velocities = self._compute_angular_velocities(velocities, window_times)
            
            feature_matrix = np.column_stack([
                speeds,
                acc_magnitude,
                curvatures,
                turning_rates,
                angular_velocities
            ])
            
            window_features.append(feature_matrix)
        
        return window_features
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std < 1e-10] = 1
        
        return (features - mean) / std
    
    def get_feature_names(self) -> List[str]:
        return [
            'speed',
            'acceleration',
            'curvature',
            'turning_rate',
            'angular_velocity'
        ]
