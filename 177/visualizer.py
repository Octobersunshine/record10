import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple
from trajectory_manager import Trajectory3D, TrajectoryManager
from stereo_matcher import StereoMatch
from config import Config


class Visualizer:
    def __init__(self, config: Config):
        self.config = config
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (0, 255, 128),
            (255, 0, 128)
        ]
        
    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        return self.colors[track_id % len(self.colors)]
    
    def _get_color_float(self, track_id: int) -> Tuple[float, float, float]:
        color = self._get_color(track_id)
        return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    
    def draw_2d_tracking(self, frame: np.ndarray, 
                         left_tracks: List, 
                         right_tracks: List,
                         stereo_matches: Dict[int, StereoMatch]) -> np.ndarray:
        h, w = frame.shape[:2]
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
        vis[:, :w] = frame.copy()
        vis[:, w:] = frame.copy()
        
        for track in left_tracks:
            color = self._get_color(track.track_id)
            cx, cy = track.predicted_position
            
            if len(track.track_history) > 1:
                for i in range(1, len(track.track_history)):
                    pt1 = (int(track.track_history[i-1][0]), int(track.track_history[i-1][1]))
                    pt2 = (int(track.track_history[i][0]), int(track.track_history[i][1]))
                    cv2.line(vis, pt1, pt2, color, 1)
            
            cv2.circle(vis, (cx, cy), 5, color, -1)
            cv2.putText(vis, f"ID:{track.track_id}", (cx + 8, cy - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        for track in right_tracks:
            color = self._get_color(track.track_id)
            cx, cy = track.predicted_position
            cx += w
            
            if len(track.track_history) > 1:
                for i in range(1, len(track.track_history)):
                    pt1 = (int(track.track_history[i-1][0]) + w, int(track.track_history[i-1][1]))
                    pt2 = (int(track.track_history[i][0]) + w, int(track.track_history[i][1]))
                    cv2.line(vis, pt1, pt2, color, 1)
            
            cv2.circle(vis, (cx, cy), 5, color, -1)
            cv2.putText(vis, f"ID:{track.track_id}", (cx + 8, cy - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        for track_id, match in stereo_matches.items():
            color = self._get_color(track_id)
            left_pt = match.left_point
            right_pt = (match.right_point[0] + w, match.right_point[1])
            cv2.line(vis, left_pt, right_pt, color, 1)
        
        cv2.putText(vis, "Left Camera", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, "Right Camera", (w + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def draw_3d_positions(self, frame: np.ndarray, 
                          stereo_matches: Dict[int, StereoMatch]) -> np.ndarray:
        h, w = frame.shape[:2]
        vis = frame.copy()
        
        for track_id, match in stereo_matches.items():
            color = self._get_color(track_id)
            x, y, z = match.point_3d
            
            cx, cy = match.left_point
            
            cv2.circle(vis, (cx, cy), 6, color, -1)
            cv2.putText(vis, f"X:{x:.1f}", (cx + 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(vis, f"Y:{y:.1f}", (cx + 10, cy + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(vis, f"Z:{z:.1f}mm", (cx + 10, cy + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis
    
    def plot_3d_trajectory(self, trajectory_manager: TrajectoryManager, 
                          save_path: Optional[str] = None,
                          show_plot: bool = True):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        trajectories = trajectory_manager.get_active_trajectories()
        
        if not trajectories:
            print("No trajectories to plot.")
            return
        
        for traj in trajectories:
            if len(traj.points) < 2:
                continue
                
            color = self._get_color_float(traj.track_id)
            positions = traj.get_position_array()
            
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                   color=color, linewidth=1.5, label=f'Track {traj.track_id}')
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                      color=color, s=50, marker='o', edgecolors='black')
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                      color=color, s=50, marker='s', edgecolors='black')
            
            for i, point in enumerate(traj.points):
                if i % max(1, len(traj.points) // 10) == 0:
                    ax.text(point.x, point.y, point.z, f"{i}",
                           fontsize=8, color=color)
        
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title('3D Flight Trajectory of Insects', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        all_positions = np.vstack([t.get_position_array() for t in trajectories if len(t.points) >= 2])
        if len(all_positions) > 0:
            max_range = np.array([
                all_positions[:, 0].max() - all_positions[:, 0].min(),
                all_positions[:, 1].max() - all_positions[:, 1].min(),
                all_positions[:, 2].max() - all_positions[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
            mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
            mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D trajectory plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def plot_trajectory_projections(self, trajectory_manager: TrajectoryManager,
                                    save_path: Optional[str] = None,
                                    show_plot: bool = True):
        trajectories = trajectory_manager.get_active_trajectories()
        
        if not trajectories:
            print("No trajectories to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for traj in trajectories:
            if len(traj.points) < 2:
                continue
                
            color = self._get_color_float(traj.track_id)
            positions = traj.get_position_array()
            times = traj.get_time_array()
            
            axes[0, 0].plot(positions[:, 0], positions[:, 1], color=color,
                           linewidth=1.5, label=f'Track {traj.track_id}')
            axes[0, 0].set_xlabel('X (mm)', fontsize=10)
            axes[0, 0].set_ylabel('Y (mm)', fontsize=10)
            axes[0, 0].set_title('X-Y Projection (Top View)', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(positions[:, 0], positions[:, 2], color=color,
                           linewidth=1.5, label=f'Track {traj.track_id}')
            axes[0, 1].set_xlabel('X (mm)', fontsize=10)
            axes[0, 1].set_ylabel('Z (mm)', fontsize=10)
            axes[0, 1].set_title('X-Z Projection (Side View)', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(positions[:, 1], positions[:, 2], color=color,
                           linewidth=1.5, label=f'Track {traj.track_id}')
            axes[1, 0].set_xlabel('Y (mm)', fontsize=10)
            axes[1, 0].set_ylabel('Z (mm)', fontsize=10)
            axes[1, 0].set_title('Y-Z Projection (Front View)', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            
            speed_data = [p.speed for p in traj.points if p.speed is not None]
            if speed_data:
                axes[1, 1].plot(times[-len(speed_data):], speed_data, color=color,
                               linewidth=1.5, label=f'Track {traj.track_id}')
        
        axes[1, 1].set_xlabel('Time (s)', fontsize=10)
        axes[1, 1].set_ylabel('Speed (mm/s)', fontsize=10)
        axes[1, 1].set_title('Speed vs Time', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory projections saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def plot_trajectory_statistics(self, trajectory_manager: TrajectoryManager,
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True):
        stats = trajectory_manager.get_trajectory_statistics()
        
        if not stats:
            print("No statistics to plot.")
            return
        
        track_ids = list(stats.keys())
        num_tracks = len(track_ids)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x = np.arange(num_tracks)
        colors = [self._get_color_float(tid) for tid in track_ids]
        
        durations = [stats[tid]['duration'] for tid in track_ids]
        axes[0, 0].bar(x, durations, color=colors, edgecolor='black')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'T{tid}' for tid in track_ids])
        axes[0, 0].set_ylabel('Duration (s)', fontsize=10)
        axes[0, 0].set_title('Track Duration', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        distances = [stats[tid]['total_distance'] for tid in track_ids]
        axes[0, 1].bar(x, distances, color=colors, edgecolor='black')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'T{tid}' for tid in track_ids])
        axes[0, 1].set_ylabel('Total Distance (mm)', fontsize=10)
        axes[0, 1].set_title('Total Travel Distance', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        avg_speeds = [stats[tid]['avg_speed'] for tid in track_ids]
        max_speeds = [stats[tid]['max_speed'] for tid in track_ids]
        width = 0.35
        axes[1, 0].bar(x - width/2, avg_speeds, width, label='Avg Speed',
                      color='lightblue', edgecolor='black')
        axes[1, 0].bar(x + width/2, max_speeds, width, label='Max Speed',
                      color='salmon', edgecolor='black')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'T{tid}' for tid in track_ids])
        axes[1, 0].set_ylabel('Speed (mm/s)', fontsize=10)
        axes[1, 0].set_title('Average and Maximum Speed', fontsize=12)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        num_points = [stats[tid]['num_points'] for tid in track_ids]
        axes[1, 1].bar(x, num_points, color=colors, edgecolor='black')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'T{tid}' for tid in track_ids])
        axes[1, 1].set_ylabel('Number of Points', fontsize=10)
        axes[1, 1].set_title('Track Length (Data Points)', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def create_composite_view(self, left_frame: np.ndarray, right_frame: np.ndarray,
                              disparity_map: np.ndarray,
                              left_tracks: List, right_tracks: List,
                              stereo_matches: Dict[int, StereoMatch]) -> np.ndarray:
        h, w = left_frame.shape[:2]
        
        tracking_view = self.draw_2d_tracking(left_frame, left_tracks, right_tracks, stereo_matches)
        
        disparity_vis = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disparity_vis = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
        disparity_vis = cv2.resize(disparity_vis, (w, h))
        
        pos_view = self.draw_3d_positions(left_frame, stereo_matches)
        
        top_row = tracking_view
        bottom_row = np.hstack([disparity_vis, pos_view])
        
        if top_row.shape[1] != bottom_row.shape[1]:
            target_w = max(top_row.shape[1], bottom_row.shape[1])
            top_row = cv2.copyMakeBorder(top_row, 0, 0, 0, target_w - top_row.shape[1],
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            bottom_row = cv2.copyMakeBorder(bottom_row, 0, 0, 0, target_w - bottom_row.shape[1],
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        composite = np.vstack([top_row, bottom_row])
        
        return composite
