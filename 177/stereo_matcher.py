import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from insect_detector import Detection
from insect_tracker import Track
from camera_calibration import StereoCalibration
from config import Config


@dataclass
class StereoMatch:
    left_point: Tuple[int, int]
    right_point: Tuple[int, int]
    disparity: float
    point_3d: Tuple[float, float, float]
    track_id: Optional[int] = None
    confidence: float = 0.0
    epipolar_error: float = 0.0


@dataclass
class MatchCandidate:
    left_idx: int
    right_idx: int
    cost: float
    epipolar_error: float
    disparity: float
    left_point: Tuple[int, int]
    right_point: Tuple[int, int]


class StereoMatcher:
    def __init__(self, config: Config, calibration: StereoCalibration):
        self.config = config
        self.calibration = calibration
        
        self.stereo_bm = self._create_stereo_matcher()
        self.stereo_sgbm = self._create_sgbm_matcher()
        self.wls_filter = self._create_wls_filter()
        
        self.raft_model = None
        self._init_raft_if_available()
        
    def _create_stereo_matcher(self) -> cv2.StereoBM:
        min_disp = self.config.STEREO_MIN_DISPARITY
        num_disp = self.config.STEREO_NUM_DISPARITIES
        block_size = self.config.STEREO_BLOCK_SIZE
        
        stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
        stereo.setMinDisparity(min_disp)
        stereo.setNumDisparities(num_disp)
        stereo.setBlockSize(block_size)
        stereo.setDisp12MaxDiff(1)
        stereo.setUniquenessRatio(15)
        stereo.setSpeckleRange(32)
        stereo.setSpeckleWindowSize(100)
        stereo.setPreFilterSize(5)
        stereo.setPreFilterCap(31)
        
        return stereo
    
    def _create_sgbm_matcher(self) -> cv2.StereoSGBM:
        min_disp = self.config.STEREO_MIN_DISPARITY
        num_disp = self.config.STEREO_NUM_DISPARITIES
        block_size = self.config.STEREO_BLOCK_SIZE
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        return stereo
    
    def _create_wls_filter(self) -> cv2.ximgproc_DisparityWLSFilter:
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_sgbm)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(1.5)
        return wls_filter
    
    def _init_raft_if_available(self):
        try:
            import torch
            from raft_stereo import RAFTStereo
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model_path = getattr(self.config, 'RAFT_MODEL_PATH', None)
            if model_path and os.path.exists(model_path):
                self.raft_model = torch.nn.DataParallel(RAFTStereo(self.config))
                self.raft_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.raft_model.to(self.device)
                self.raft_model.eval()
                print("RAFT-Stereo model loaded successfully.")
        except ImportError:
            self.raft_model = None
        except Exception as e:
            print(f"RAFT-Stereo initialization skipped: {e}")
            self.raft_model = None
    
    def compute_epipolar_line(self, point: Tuple[int, int], 
                              which: str = 'left') -> np.ndarray:
        if self.calibration.F is None:
            F = self._compute_fundamental_matrix()
        else:
            F = self.calibration.F
        
        pt = np.array([point[0], point[1], 1.0])
        
        if which == 'left':
            line = F @ pt
        else:
            line = F.T @ pt
        
        return line
    
    def point_to_epipolar_distance(self, point: Tuple[int, int], 
                                    epipolar_line: np.ndarray) -> float:
        a, b, c = epipolar_line
        x, y = point
        
        numerator = abs(a * x + b * y + c)
        denominator = np.sqrt(a ** 2 + b ** 2)
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
    
    def _compute_fundamental_matrix(self) -> np.ndarray:
        K1 = self.calibration.left_camera_matrix
        K2 = self.calibration.right_camera_matrix
        R = self.calibration.R
        T = self.calibration.T
        
        E = np.cross(T, np.eye(3) @ R.T, axis=0) if T.ndim == 1 else np.cross(T.flatten(), np.eye(3), axis=0)
        E = np.array([
            [0, -T[2], T[1]],
            [T[2], 0, -T[0]],
            [-T[1], T[0], 0]
        ]) @ R
        
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        
        return F
    
    def compute_disparity_map(self, left_rect: np.ndarray, right_rect: np.ndarray,
                              use_sgbm: bool = True) -> np.ndarray:
        if len(left_rect.shape) == 3:
            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_rect
            right_gray = right_rect
        
        left_gray = cv2.equalizeHist(left_gray)
        right_gray = cv2.equalizeHist(right_gray)
        
        matcher = self.stereo_sgbm if use_sgbm else self.stereo_bm
        
        disparity_left = matcher.compute(left_gray, right_gray)
        
        stereo_right = cv2.ximgproc.createRightMatcher(matcher)
        disparity_right = stereo_right.compute(right_gray, left_gray)
        
        disparity_left = disparity_left.astype(np.float32) / 16.0
        disparity_right = disparity_right.astype(np.float32) / 16.0
        
        filtered_disp = self.wls_filter.filter(disparity_left, left_gray, 
                                               disparity_map_right=disparity_right)
        
        return filtered_disp
    
    def compute_disparity_raft(self, left_rect: np.ndarray, 
                               right_rect: np.ndarray) -> Optional[np.ndarray]:
        if self.raft_model is None:
            return None
        
        try:
            import torch
            
            left_tensor = self._prepare_tensor_raft(left_rect)
            right_tensor = self._prepare_tensor_raft(right_rect)
            
            with torch.no_grad():
                disparity = self.raft_model(left_tensor, right_tensor)
                
            disparity_np = disparity.squeeze().cpu().numpy()
            
            return disparity_np
        except Exception as e:
            print(f"RAFT-Stereo inference failed: {e}")
            return None
    
    def _prepare_tensor_raft(self, frame: np.ndarray) -> 'torch.Tensor':
        import torch
        
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.resize(frame, (960, 540))
        
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def match_tracks_by_epipolar(self, left_tracks: List[Track], 
                                 right_tracks: List[Track],
                                 epipolar_threshold: float = 3.0) -> Dict[int, StereoMatch]:
        if not left_tracks or not right_tracks:
            return {}
        
        candidates = self._generate_match_candidates(left_tracks, right_tracks, 
                                                     epipolar_threshold)
        
        if not candidates:
            return {}
        
        matches = self._solve_matching_problem(candidates, left_tracks, right_tracks)
        
        return matches
    
    def _generate_match_candidates(self, left_tracks: List[Track], 
                                   right_tracks: List[Track],
                                   epipolar_threshold: float) -> List[MatchCandidate]:
        candidates = []
        
        F = self._compute_fundamental_matrix()
        
        for i, left_track in enumerate(left_tracks):
            left_point = left_track.predicted_position
            
            epipolar_line = F @ np.array([left_point[0], left_point[1], 1.0])
            
            for j, right_track in enumerate(right_tracks):
                right_point = right_track.predicted_position
                
                epipolar_error = self.point_to_epipolar_distance(right_point, epipolar_line)
                
                if epipolar_error > epipolar_threshold:
                    continue
                
                x_diff = left_point[0] - right_point[0]
                if x_diff <= 0 or x_diff > self.config.STEREO_NUM_DISPARITIES:
                    continue
                
                y_diff = abs(left_point[1] - right_point[1])
                if y_diff > epipolar_threshold * 2:
                    continue
                
                distance = np.sqrt(x_diff ** 2 + y_diff ** 2)
                
                appearance_cost = self._compute_appearance_cost(left_track, right_track)
                
                cost = distance * 0.4 + epipolar_error * 0.4 + appearance_cost * 0.2
                
                candidates.append(MatchCandidate(
                    left_idx=i,
                    right_idx=j,
                    cost=cost,
                    epipolar_error=epipolar_error,
                    disparity=x_diff,
                    left_point=left_point,
                    right_point=right_point
                ))
        
        return candidates
    
    def _compute_appearance_cost(self, left_track: Track, right_track: Track) -> float:
        left_det = left_track.detection
        right_det = right_track.detection
        
        if left_det is None or right_det is None:
            return 1.0
        
        size_ratio = min(left_det.width * left_det.height, right_det.width * right_det.height) / \
                    max(left_det.width * left_det.height, right_det.width * right_det.height)
        
        aspect_ratio_diff = abs(
            left_det.width / max(left_det.height, 1) - right_det.width / max(right_det.height, 1)
        )
        
        cost = (1.0 - size_ratio) + aspect_ratio_diff * 0.5
        
        return min(cost, 2.0)
    
    def _solve_matching_problem(self, candidates: List[MatchCandidate],
                                left_tracks: List[Track], 
                                right_tracks: List[Track]) -> Dict[int, StereoMatch]:
        matches = {}
        
        if not candidates:
            return matches
        
        n_left = len(left_tracks)
        n_right = len(right_tracks)
        
        cost_matrix = np.full((n_left, n_right), 1e6)
        
        for cand in candidates:
            cost_matrix[cand.left_idx, cand.right_idx] = cand.cost
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_right = set()
        
        for left_i, right_i in zip(row_ind, col_ind):
            if cost_matrix[left_i, right_i] >= 1e6:
                continue
            
            left_track = left_tracks[left_i]
            right_track = right_tracks[right_i]
            
            left_point = left_track.predicted_position
            right_point = right_track.predicted_position
            
            if right_i in matched_right:
                continue
            
            disparity = left_point[0] - right_point[0]
            
            if disparity <= 0 or disparity > self.config.STEREO_NUM_DISPARITIES:
                continue
            
            point_3d = self._triangulate_point(left_point, right_point)
            
            cand = next((c for c in candidates 
                        if c.left_idx == left_i and c.right_idx == right_i), None)
            confidence = self._compute_match_confidence(cand) if cand else 0.0
            
            if confidence < 0.3:
                continue
            
            stereo_match = StereoMatch(
                left_point=left_point,
                right_point=right_point,
                disparity=disparity,
                point_3d=point_3d,
                track_id=left_track.track_id,
                confidence=confidence,
                epipolar_error=cand.epipolar_error if cand else 0.0
            )
            
            matches[left_track.track_id] = stereo_match
            matched_right.add(right_i)
        
        return matches
    
    def _compute_match_confidence(self, candidate: MatchCandidate) -> float:
        if candidate is None:
            return 0.0
        
        epi_conf = max(0, 1.0 - candidate.epipolar_error / 10.0)
        
        cost_conf = max(0, 1.0 - candidate.cost / 100.0)
        
        disp_conf = max(0, 1.0 - abs(candidate.disparity - 
                       self.config.STEREO_NUM_DISPARITIES / 2) / 
                       (self.config.STEREO_NUM_DISPARITIES / 2))
        
        confidence = epi_conf * 0.4 + cost_conf * 0.3 + disp_conf * 0.3
        
        return min(max(confidence, 0.0), 1.0)
    
    def match_detections_by_epipolar(self, left_detections: List[Detection],
                                     right_detections: List[Detection],
                                     epipolar_threshold: float = 3.0) -> List[StereoMatch]:
        if not left_detections or not right_detections:
            return []
        
        F = self._compute_fundamental_matrix()
        
        candidates = []
        
        for i, left_det in enumerate(left_detections):
            left_point = left_det.centroid
            
            epipolar_line = F @ np.array([left_point[0], left_point[1], 1.0])
            
            best_cand = None
            best_cost = float('inf')
            
            for j, right_det in enumerate(right_detections):
                right_point = right_det.centroid
                
                epipolar_error = self.point_to_epipolar_distance(right_point, epipolar_line)
                
                if epipolar_error > epipolar_threshold:
                    continue
                
                x_diff = left_point[0] - right_point[0]
                if x_diff <= 0 or x_diff > self.config.STEREO_NUM_DISPARITIES:
                    continue
                
                y_diff = abs(left_point[1] - right_point[1])
                if y_diff > epipolar_threshold * 2:
                    continue
                
                distance = np.sqrt(x_diff ** 2 + y_diff ** 2)
                
                size_ratio = min(left_det.width * left_det.height, 
                                right_det.width * right_det.height) / \
                            max(left_det.width * left_det.height, 
                                right_det.width * right_det.height)
                
                cost = distance * 0.5 + epipolar_error * 0.3 + (1.0 - size_ratio) * 20
                
                if cost < best_cost:
                    best_cost = cost
                    best_cand = MatchCandidate(
                        left_idx=i,
                        right_idx=j,
                        cost=cost,
                        epipolar_error=epipolar_error,
                        disparity=x_diff,
                        left_point=left_point,
                        right_point=right_point
                    )
            
            if best_cand is not None:
                candidates.append(best_cand)
        
        matches = []
        matched_right = set()
        
        candidates.sort(key=lambda c: c.cost)
        
        for cand in candidates:
            if cand.right_idx in matched_right:
                continue
            
            point_3d = self._triangulate_point(cand.left_point, cand.right_point)
            confidence = self._compute_match_confidence(cand)
            
            if confidence < 0.3:
                continue
            
            stereo_match = StereoMatch(
                left_point=cand.left_point,
                right_point=cand.right_point,
                disparity=cand.disparity,
                point_3d=point_3d,
                confidence=confidence,
                epipolar_error=cand.epipolar_error
            )
            
            matches.append(stereo_match)
            matched_right.add(cand.right_idx)
        
        return matches
    
    def match_with_disparity_map(self, left_tracks: List[Track],
                                 disparity_map: np.ndarray) -> Dict[int, StereoMatch]:
        matches = {}
        
        for left_track in left_tracks:
            left_point = left_track.predicted_position
            x, y = int(left_point[0]), int(left_point[1])
            
            if 0 <= y < disparity_map.shape[0] and 0 <= x < disparity_map.shape[1]:
                disparity = disparity_map[y, x]
                
                if disparity > 0:
                    right_point = (int(x - disparity), y)
                    
                    point_3d = self.compute_3d_from_disparity(disparity_map, left_point)
                    
                    if point_3d is not None:
                        stereo_match = StereoMatch(
                            left_point=left_point,
                            right_point=right_point,
                            disparity=disparity,
                            point_3d=point_3d,
                            track_id=left_track.track_id,
                            confidence=0.7
                        )
                        matches[left_track.track_id] = stereo_match
        
        return matches
    
    def _triangulate_point(self, left_point: Tuple[int, int], 
                          right_point: Tuple[int, int]) -> Tuple[float, float, float]:
        if self.calibration.Q is not None:
            disparity = left_point[0] - right_point[0]
            point_homogeneous = cv2.perspectiveTransform(
                np.array([[[left_point[0], left_point[1], disparity]]], dtype=np.float32),
                self.calibration.Q
            )
            x = point_homogeneous[0, 0, 0]
            y = point_homogeneous[0, 0, 1]
            z = point_homogeneous[0, 0, 2]
        else:
            points_4d = cv2.triangulatePoints(
                self.calibration.P1, self.calibration.P2,
                np.array([left_point], dtype=np.float32).T,
                np.array([right_point], dtype=np.float32).T
            )
            points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
            x = points_3d[0, 0, 0]
            y = points_3d[0, 0, 1]
            z = points_3d[0, 0, 2]
        
        return (x * self.config.PIXEL_TO_MM, y * self.config.PIXEL_TO_MM, z * self.config.PIXEL_TO_MM)
    
    def compute_3d_from_disparity(self, disparity_map: np.ndarray, 
                                  left_point: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
        x, y = int(left_point[0]), int(left_point[1])
        
        if 0 <= y < disparity_map.shape[0] and 0 <= x < disparity_map.shape[1]:
            disparity = disparity_map[y, x]
            if disparity > 0:
                point_homogeneous = cv2.perspectiveTransform(
                    np.array([[[x, y, disparity]]], dtype=np.float32),
                    self.calibration.Q
                )
                x_3d = point_homogeneous[0, 0, 0] * self.config.PIXEL_TO_MM
                y_3d = point_homogeneous[0, 0, 1] * self.config.PIXEL_TO_MM
                z_3d = point_homogeneous[0, 0, 2] * self.config.PIXEL_TO_MM
                return (x_3d, y_3d, z_3d)
        
        return None
    
    def visualize_disparity(self, disparity_map: np.ndarray) -> np.ndarray:
        disp_visual = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_visual = cv2.applyColorMap(disp_visual, cv2.COLORMAP_JET)
        return disp_visual
    
    def visualize_matches(self, left_frame: np.ndarray, right_frame: np.ndarray,
                         matches: List[StereoMatch]) -> np.ndarray:
        h_left, w_left = left_frame.shape[:2]
        h_right, w_right = right_frame.shape[:2]
        max_h = max(h_left, h_right)
        
        vis_left = left_frame.copy()
        vis_right = right_frame.copy()
        
        if h_left < max_h:
            vis_left = cv2.copyMakeBorder(vis_left, 0, max_h - h_left, 0, 0, cv2.BORDER_CONSTANT)
        if h_right < max_h:
            vis_right = cv2.copyMakeBorder(vis_right, 0, max_h - h_right, 0, 0, cv2.BORDER_CONSTANT)
        
        combined = np.hstack([vis_left, vis_right])
        
        for match in matches:
            left_pt = match.left_point
            right_pt = (match.right_point[0] + w_left, match.right_point[1])
            
            color = self._get_confidence_color(match.confidence)
            
            cv2.circle(combined, left_pt, 5, color, -1)
            cv2.circle(combined, right_pt, 5, color, -1)
            cv2.line(combined, left_pt, right_pt, color, 1)
            
            cv2.putText(combined, f"d:{match.disparity:.1f}", 
                       (left_pt[0] + 10, left_pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(combined, f"z:{match.point_3d[2]:.1f}mm", 
                       (left_pt[0] + 10, left_pt[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(combined, f"c:{match.confidence:.2f}", 
                       (left_pt[0] + 10, left_pt[1] + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return combined
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        if confidence >= 0.7:
            return (0, 255, 0)
        elif confidence >= 0.5:
            return (255, 255, 0)
        else:
            return (0, 0, 255)
    
    def draw_epipolar_lines(self, left_frame: np.ndarray, right_frame: np.ndarray,
                            left_points: List[Tuple[int, int]]) -> np.ndarray:
        h_left, w_left = left_frame.shape[:2]
        
        vis_left = left_frame.copy()
        vis_right = right_frame.copy()
        
        F = self._compute_fundamental_matrix()
        
        for pt in left_points:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            cv2.circle(vis_left, pt, 5, color, -1)
            
            epipolar_line = F @ np.array([pt[0], pt[1], 1.0])
            a, b, c = epipolar_line
            
            y1 = 0
            x1 = int(-(b * y1 + c) / a) if a != 0 else 0
            
            y2 = vis_right.shape[0] - 1
            x2 = int(-(b * y2 + c) / a) if a != 0 else 0
            
            x1 = max(0, min(x1, vis_right.shape[1] - 1))
            x2 = max(0, min(x2, vis_right.shape[1] - 1))
            
            cv2.line(vis_right, (x1, y1), (x2, y2), color, 1)
        
        combined = np.hstack([vis_left, vis_right])
        
        return combined
