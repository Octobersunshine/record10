import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from insect_detector import Detection
from config import Config


@dataclass
class Track:
    track_id: int
    detection: Detection
    kalman_filter: KalmanFilter
    age: int = 0
    hits: int = 0
    misses: int = 0
    track_history: List[Tuple[int, int]] = field(default_factory=list)
    
    @property
    def predicted_position(self) -> Tuple[int, int]:
        return (int(self.kalman_filter.x[0]), int(self.kalman_filter.x[1]))


class InsectTracker:
    def __init__(self, config: Config):
        self.config = config
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.max_misses = 10
        self.min_hits = 3
        
    def _create_kalman_filter(self, x: int, y: int) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        kf.x = np.array([[x], [y], [0], [0]])
        kf.P *= 1000
        kf.R = np.array([[10, 0], [0, 10]])
        kf.Q = np.eye(4) * 0.1
        return kf
    
    def _compute_iou(self, det1: Detection, det2: Detection) -> float:
        x1 = max(det1.x, det2.x)
        y1 = max(det1.y, det2.y)
        x2 = min(det1.x + det1.width, det2.x + det2.width)
        y2 = min(det1.y + det1.height, det2.y + det2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = det1.width * det1.height
        area2 = det2.width * det2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_cost_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            pred_x, pred_y = track.predicted_position
            for j, det in enumerate(detections):
                det_x, det_y = det.centroid
                
                distance = np.sqrt((pred_x - det_x) ** 2 + (pred_y - det_y) ** 2)
                iou = self._compute_iou(track.detection, det)
                
                cost_matrix[i, j] = distance * (1.0 - iou)
        
        return cost_matrix
    
    def _assign_detections_to_tracks(self, cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
            
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = []
        unmatched_detections = []
        
        matched_tracks = set()
        matched_detections = set()
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 100:
                matches.append((i, j))
                matched_tracks.add(i)
                matched_detections.add(j)
        
        for i in range(cost_matrix.shape[0]):
            if i not in matched_tracks:
                unmatched_tracks.append(i)
        
        for j in range(cost_matrix.shape[1]):
            if j not in matched_detections:
                unmatched_detections.append(j)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def update(self, detections: List[Detection]) -> List[Track]:
        for track in self.tracks.values():
            track.kalman_filter.predict()
        
        track_list = list(self.tracks.values())
        cost_matrix = self._compute_cost_matrix(track_list, detections)
        matches, unmatched_tracks, unmatched_detections = self._assign_detections_to_tracks(cost_matrix)
        
        for track_idx, det_idx in matches:
            track = track_list[track_idx]
            det = detections[det_idx]
            
            track.kalman_filter.update(np.array([[det.centroid[0]], [det.centroid[1]]]))
            track.detection = det
            track.hits += 1
            track.misses = 0
            track.age += 1
            track.track_history.append(det.centroid)
        
        for track_idx in unmatched_tracks:
            track = track_list[track_idx]
            track.misses += 1
            track.age += 1
        
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            kf = self._create_kalman_filter(det.centroid[0], det.centroid[1])
            new_track = Track(
                track_id=self.next_id,
                detection=det,
                kalman_filter=kf,
                age=1,
                hits=1,
                misses=0,
                track_history=[det.centroid]
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        active_tracks = {}
        for track_id, track in self.tracks.items():
            if track.misses <= self.max_misses:
                active_tracks[track_id] = track
        self.tracks = active_tracks
        
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]
    
    def get_visible_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.hits >= self.min_hits and t.misses == 0]
    
    def visualize_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        vis_frame = frame.copy()
        
        for track in tracks:
            color = self._get_track_color(track.track_id)
            
            if len(track.track_history) > 1:
                for i in range(1, len(track.track_history)):
                    cv2.line(vis_frame, track.track_history[i-1], track.track_history[i], color, 1)
            
            x, y = track.predicted_position
            det = track.detection
            cv2.rectangle(vis_frame, (det.x, det.y), 
                         (det.x + det.width, det.y + det.height), color, 2)
            cv2.circle(vis_frame, (x, y), 4, color, -1)
            cv2.putText(vis_frame, f"ID:{track.track_id}", 
                       (det.x, det.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (0, 255, 128),
            (255, 0, 128)
        ]
        return colors[track_id % len(colors)]
