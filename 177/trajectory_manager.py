import numpy as np
import csv
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from stereo_matcher import StereoMatch
from config import Config


@dataclass
class TrajectoryPoint:
    timestamp: float
    frame_index: int
    track_id: int
    x: float
    y: float
    z: float
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None
    speed: Optional[float] = None
    acceleration: Optional[float] = None


@dataclass
class Trajectory3D:
    track_id: int
    points: List[TrajectoryPoint] = field(default_factory=list)
    
    def add_point(self, point: TrajectoryPoint):
        self.points.append(point)
    
    def compute_velocities(self):
        if len(self.points) < 2:
            return
        
        for i in range(1, len(self.points)):
            dt = self.points[i].timestamp - self.points[i-1].timestamp
            if dt > 0:
                self.points[i].vx = (self.points[i].x - self.points[i-1].x) / dt
                self.points[i].vy = (self.points[i].y - self.points[i-1].y) / dt
                self.points[i].vz = (self.points[i].z - self.points[i-1].z) / dt
                self.points[i].speed = np.sqrt(
                    self.points[i].vx ** 2 + 
                    self.points[i].vy ** 2 + 
                    self.points[i].vz ** 2
                )
    
    def compute_accelerations(self):
        if len(self.points) < 3:
            return
        
        for i in range(2, len(self.points)):
            if self.points[i].speed is not None and self.points[i-1].speed is not None:
                dt = self.points[i].timestamp - self.points[i-1].timestamp
                if dt > 0:
                    self.points[i].acceleration = (
                        self.points[i].speed - self.points[i-1].speed
                    ) / dt
    
    def get_position_array(self) -> np.ndarray:
        return np.array([[p.x, p.y, p.z] for p in self.points])
    
    def get_time_array(self) -> np.ndarray:
        return np.array([p.timestamp for p in self.points])
    
    def smooth_trajectory(self, window_size: int = 3):
        if len(self.points) < window_size:
            return
        
        positions = self.get_position_array()
        kernel = np.ones(window_size) / window_size
        
        smoothed_x = np.convolve(positions[:, 0], kernel, mode='same')
        smoothed_y = np.convolve(positions[:, 1], kernel, mode='same')
        smoothed_z = np.convolve(positions[:, 2], kernel, mode='same')
        
        for i, point in enumerate(self.points):
            point.x = smoothed_x[i]
            point.y = smoothed_y[i]
            point.z = smoothed_z[i]


class TrajectoryManager:
    def __init__(self, config: Config):
        self.config = config
        self.trajectories: Dict[int, Trajectory3D] = {}
        self.frame_index = 0
        self.start_time = datetime.now()
    
    def add_stereo_match(self, stereo_match: StereoMatch, frame_index: int):
        timestamp = (datetime.now() - self.start_time).total_seconds()
        
        track_id = stereo_match.track_id if stereo_match.track_id is not None else -1
        
        point = TrajectoryPoint(
            timestamp=timestamp,
            frame_index=frame_index,
            track_id=track_id,
            x=stereo_match.point_3d[0],
            y=stereo_match.point_3d[1],
            z=stereo_match.point_3d[2]
        )
        
        if track_id not in self.trajectories:
            self.trajectories[track_id] = Trajectory3D(track_id=track_id)
        
        self.trajectories[track_id].add_point(point)
    
    def add_3d_point(self, track_id: int, point_3d: Tuple[float, float, float], frame_index: int):
        timestamp = (datetime.now() - self.start_time).total_seconds()
        
        point = TrajectoryPoint(
            timestamp=timestamp,
            frame_index=frame_index,
            track_id=track_id,
            x=point_3d[0],
            y=point_3d[1],
            z=point_3d[2]
        )
        
        if track_id not in self.trajectories:
            self.trajectories[track_id] = Trajectory3D(track_id=track_id)
        
        self.trajectories[track_id].add_point(point)
    
    def finalize_trajectories(self, smooth: bool = True):
        for traj in self.trajectories.values():
            if smooth:
                traj.smooth_trajectory()
            traj.compute_velocities()
            traj.compute_accelerations()
    
    def save_to_csv(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = self.config.TRAJECTORY_OUTPUT_PATH
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.finalize_trajectories()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'track_id', 'frame_index', 'timestamp',
                'x_mm', 'y_mm', 'z_mm',
                'vx_mm/s', 'vy_mm/s', 'vz_mm/s',
                'speed_mm/s', 'acceleration_mm/s2'
            ])
            
            for traj in sorted(self.trajectories.values(), key=lambda t: t.track_id):
                for point in traj.points:
                    writer.writerow([
                        point.track_id,
                        point.frame_index,
                        f"{point.timestamp:.6f}",
                        f"{point.x:.4f}",
                        f"{point.y:.4f}",
                        f"{point.z:.4f}",
                        f"{point.vx:.4f}" if point.vx is not None else "",
                        f"{point.vy:.4f}" if point.vy is not None else "",
                        f"{point.vz:.4f}" if point.vz is not None else "",
                        f"{point.speed:.4f}" if point.speed is not None else "",
                        f"{point.acceleration:.4f}" if point.acceleration is not None else ""
                    ])
        
        print(f"Trajectory saved to {filepath}")
        return filepath
    
    def load_from_csv(self, filepath: str):
        self.trajectories.clear()
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                track_id = int(row['track_id'])
                
                point = TrajectoryPoint(
                    timestamp=float(row['timestamp']),
                    frame_index=int(row['frame_index']),
                    track_id=track_id,
                    x=float(row['x_mm']),
                    y=float(row['y_mm']),
                    z=float(row['z_mm']),
                    vx=float(row['vx_mm/s']) if row['vx_mm/s'] else None,
                    vy=float(row['vy_mm/s']) if row['vy_mm/s'] else None,
                    vz=float(row['vz_mm/s']) if row['vz_mm/s'] else None,
                    speed=float(row['speed_mm/s']) if row['speed_mm/s'] else None,
                    acceleration=float(row['acceleration_mm/s2']) if row['acceleration_mm/s2'] else None
                )
                
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = Trajectory3D(track_id=track_id)
                
                self.trajectories[track_id].add_point(point)
    
    def get_trajectory_statistics(self) -> Dict:
        stats = {}
        
        for track_id, traj in self.trajectories.items():
            if len(traj.points) < 2:
                continue
            
            positions = traj.get_position_array()
            times = traj.get_time_array()
            
            stats[track_id] = {
                'num_points': len(traj.points),
                'duration': times[-1] - times[0],
                'total_distance': self._compute_total_distance(positions),
                'max_speed': max([p.speed for p in traj.points if p.speed is not None], default=0),
                'avg_speed': np.mean([p.speed for p in traj.points if p.speed is not None]),
                'bounding_box': {
                    'x_min': positions[:, 0].min(),
                    'x_max': positions[:, 0].max(),
                    'y_min': positions[:, 1].min(),
                    'y_max': positions[:, 1].max(),
                    'z_min': positions[:, 2].min(),
                    'z_max': positions[:, 2].max(),
                }
            }
        
        return stats
    
    def _compute_total_distance(self, positions: np.ndarray) -> float:
        if len(positions) < 2:
            return 0.0
        return np.sum(np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1)))
    
    def get_active_trajectories(self) -> List[Trajectory3D]:
        return [t for t in self.trajectories.values() if len(t.points) >= 3]
