import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, convolve
from typing import List, Tuple, Optional, Dict


class EnhancedRadarExtrapolator:
    def __init__(self, flow_method: str = 'farneback', 
                 use_mass_conservation: bool = True,
                 use_intensity_correction: bool = True,
                 use_adaptive_weighting: bool = True,
                 **kwargs):
        self.flow_method = flow_method.lower()
        self.use_mass_conservation = use_mass_conservation
        self.use_intensity_correction = use_intensity_correction
        self.use_adaptive_weighting = use_adaptive_weighting
        self.flow_params = kwargs
        self._init_flow_params()
        
        self.intensity_model = None
        self.growth_rate_map = None
        self.divergence_map = None

    def _init_flow_params(self):
        if self.flow_method == 'farneback':
            self.flow_params.setdefault('pyr_scale', 0.5)
            self.flow_params.setdefault('levels', 3)
            self.flow_params.setdefault('winsize', 15)
            self.flow_params.setdefault('iterations', 3)
            self.flow_params.setdefault('poly_n', 5)
            self.flow_params.setdefault('poly_sigma', 1.2)
            self.flow_params.setdefault('flags', 0)
        elif self.flow_method == 'dis':
            self.flow_params.setdefault('preset', cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        elif self.flow_method == 'lucas_kanade':
            self.flow_params.setdefault('winSize', (15, 15))
            self.flow_params.setdefault('maxLevel', 2)
            self.flow_params.setdefault('criteria', (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def _compute_flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        prev_norm = self._normalize_image(prev)
        curr_norm = self._normalize_image(curr)

        if self.flow_method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                prev_norm, curr_norm, None,
                self.flow_params['pyr_scale'],
                self.flow_params['levels'],
                self.flow_params['winsize'],
                self.flow_params['iterations'],
                self.flow_params['poly_n'],
                self.flow_params['poly_sigma'],
                self.flow_params['flags']
            )
        elif self.flow_method == 'dis':
            dis = cv2.DISOpticalFlow_create(self.flow_params['preset'])
            flow = dis.calc(prev_norm, curr_norm, None)
        elif self.flow_method == 'lucas_kanade':
            flow = self._lucas_kanade_flow(prev_norm, curr_norm)
        else:
            raise ValueError(f"Unknown method: {self.flow_method}")

        return flow

    def _lucas_kanade_flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        h, w = prev.shape
        flow = np.zeros((h, w, 2), dtype=np.float32)

        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        if p0 is None:
            return flow

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev, curr, p0, None,
            winSize=self.flow_params['winSize'],
            maxLevel=self.flow_params['maxLevel'],
            criteria=self.flow_params['criteria']
        )

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_old) > 0:
            points = good_old.astype(np.int32)
            displacements = good_new - good_old

            for i, (x, y) in enumerate(points):
                if 0 <= y < h and 0 <= x < w:
                    flow[y, x] = displacements[i]

            flow = self._interpolate_flow(flow, points)

        return flow

    def _interpolate_flow(self, flow: np.ndarray, points: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack((x_coords.ravel(), y_coords.ravel()))

        valid_mask = (flow[:, :, 0] != 0) | (flow[:, :, 1] != 0)
        if not np.any(valid_mask):
            return flow

        known_x = x_coords[valid_mask]
        known_y = y_coords[valid_mask]
        known_u = flow[valid_mask, 0]
        known_v = flow[valid_mask, 1]
        known_points = np.column_stack((known_x, known_y))

        if len(known_points) < 4:
            return flow

        u_interp = griddata(known_points, known_u, grid_points, method='linear', fill_value=0)
        v_interp = griddata(known_points, known_v, grid_points, method='linear', fill_value=0)

        flow[:, :, 0] = u_interp.reshape(h, w)
        flow[:, :, 1] = v_interp.reshape(h, w)

        return flow

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            normalized = (img - img_min) / (img_max - img_min) * 255
        else:
            normalized = np.zeros_like(img)
        return normalized.astype(np.uint8)

    def _compute_divergence(self, flow: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
        
        du_dx = convolve(u, sobel_x)
        dv_dy = convolve(v, sobel_y)
        
        divergence = du_dx + dv_dy
        return divergence

    def compute_intensity_trend(self, images: List[np.ndarray], 
                                smooth_sigma: float = 2.0) -> np.ndarray:
        if len(images) < 3:
            raise ValueError("At least 3 images are required for intensity trend analysis")
        
        h, w = images[0].shape
        num_frames = len(images)
        
        images_smoothed = [gaussian_filter(img, sigma=smooth_sigma) for img in images]
        
        X = np.arange(num_frames)
        X_mean = np.mean(X)
        X_var = np.sum((X - X_mean) ** 2)
        
        slope_map = np.zeros((h, w), dtype=np.float32)
        intercept_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                y_vals = np.array([img[y, x] for img in images_smoothed])
                y_mean = np.mean(y_vals)
                
                if X_var > 0:
                    slope = np.sum((X - X_mean) * (y_vals - y_mean)) / X_var
                else:
                    slope = 0
                
                slope_map[y, x] = slope
                intercept_map[y, x] = y_mean - slope * X_mean
        
        self.intensity_model = {
            'slope': slope_map,
            'intercept': intercept_map,
            'num_frames': num_frames
        }
        
        self.growth_rate_map = slope_map / (np.mean(images_smoothed, axis=0) + 1e-6)
        
        return self.growth_rate_map

    def detect_growth_decay_regions(self, images: List[np.ndarray], 
                                    threshold: float = 0.5,
                                    min_intensity: float = 5.0) -> Dict[str, np.ndarray]:
        if len(images) < 3:
            self.compute_intensity_trend(images)
        
        h, w = images[0].shape
        
        growth_region = (self.growth_rate_map > threshold) & (images[-1] > min_intensity)
        decay_region = (self.growth_rate_map < -threshold) & (images[-1] > min_intensity)
        stable_region = ~growth_region & ~decay_region & (images[-1] > min_intensity)
        
        growth_region = gaussian_filter(growth_region.astype(np.float32), sigma=1) > 0.5
        decay_region = gaussian_filter(decay_region.astype(np.float32), sigma=1) > 0.5
        
        return {
            'growth': growth_region,
            'decay': decay_region,
            'stable': stable_region
        }

    def compute_adaptive_weights(self, images: List[np.ndarray],
                                 regions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        h, w = images[0].shape
        
        motion_confidence = np.ones((h, w), dtype=np.float32)
        if len(images) >= 3:
            flows = []
            for i in range(len(images) - 1):
                flow = self._compute_flow(images[i], images[i + 1])
                flows.append(flow)
            
            flow_std = np.std(flows, axis=0)
            motion_confidence = np.exp(-np.sqrt(flow_std[:, :, 0]**2 + flow_std[:, :, 1]**2) / 2)
        
        intensity_trust = np.ones((h, w), dtype=np.float32)
        if self.intensity_model is not None:
            images_array = np.array(images)
            predicted = np.zeros_like(images_array)
            for t in range(len(images)):
                predicted[t] = self.intensity_model['slope'] * t + self.intensity_model['intercept']
            
            residuals = np.mean((images_array - predicted) ** 2, axis=0)
            intensity_trust = np.exp(-residuals / 50)
        
        weights = {
            'motion': motion_confidence,
            'intensity': intensity_trust,
            'growth_weight': regions['growth'].astype(np.float32) * 0.5 + 0.5,
            'decay_weight': regions['decay'].astype(np.float32) * 0.5 + 0.5
        }
        
        return weights

    def semi_lagrangian_mass_conserving(self, image: np.ndarray, flow: np.ndarray,
                                        divergence: np.ndarray,
                                        dt: float = 1.0) -> np.ndarray:
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        src_x = x_coords - flow[:, :, 0] * dt
        src_y = y_coords - flow[:, :, 1] * dt
        
        src_x_clipped = np.clip(src_x, 0, w - 1)
        src_y_clipped = np.clip(src_y, 0, h - 1)
        
        x0 = np.floor(src_x_clipped).astype(np.int32)
        y0 = np.floor(src_y_clipped).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        
        fx = src_x_clipped - x0
        fy = src_y_clipped - y0
        
        v00 = image[y0, x0]
        v01 = image[y0, x1]
        v10 = image[y1, x0]
        v11 = image[y1, x1]
        
        top = v00 * (1 - fx) + v01 * fx
        bottom = v10 * (1 - fx) + v11 * fx
        advected = top * (1 - fy) + bottom * fy
        
        if self.use_mass_conservation:
            mass_correction = np.exp(-divergence * dt)
            advected = advected * mass_correction
        
        boundary_mask = (src_x < 0) | (src_x >= w) | (src_y < 0) | (src_y >= h)
        advected[boundary_mask] = 0
        
        return advected

    def intensity_correction(self, base_image: np.ndarray, step: int,
                            regions: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.use_intensity_correction or self.intensity_model is None:
            return base_image
        
        h, w = base_image.shape
        corrected = base_image.copy()
        
        growth_correction = np.exp(self.growth_rate_map * step * 0.5)
        decay_correction = np.exp(self.growth_rate_map * step * 0.5)
        
        growth_mask = regions['growth']
        decay_mask = regions['decay']
        
        corrected[growth_mask] = base_image[growth_mask] * growth_correction[growth_mask]
        corrected[decay_mask] = base_image[decay_mask] * decay_correction[decay_mask]
        
        corrected = np.clip(corrected, 0, None)
        
        return corrected

    def extrapolate_enhanced(self, images: List[np.ndarray], steps: int = 1,
                            time_interval: float = 6.0) -> List[np.ndarray]:
        if len(images) < 2:
            raise ValueError("At least 2 images are required for extrapolation")
        
        h, w = images[0].shape
        last_image = images[-1]
        
        flows = []
        divergences = []
        for i in range(len(images) - 1):
            flow = self._compute_flow(images[i], images[i + 1])
            flows.append(flow)
            div = self._compute_divergence(flow)
            divergences.append(div)
        
        avg_flow = np.mean(flows, axis=0)
        avg_divergence = np.mean(divergences, axis=0)
        self.divergence_map = avg_divergence
        
        if len(images) >= 3 and self.use_intensity_correction:
            self.compute_intensity_trend(images)
            regions = self.detect_growth_decay_regions(images)
            weights = self.compute_adaptive_weights(images, regions)
        else:
            regions = {'growth': np.zeros((h, w), dtype=bool),
                      'decay': np.zeros((h, w), dtype=bool),
                      'stable': np.ones((h, w), dtype=bool)}
            weights = None
        
        extrapolated = []
        current = last_image.copy()
        
        for step in range(1, steps + 1):
            advected = self.semi_lagrangian_mass_conserving(
                last_image, avg_flow, avg_divergence, dt=step
            )
            
            if self.use_intensity_correction and self.intensity_model is not None:
                corrected = self.intensity_correction(advected, step, regions)
                
                if self.use_adaptive_weighting and weights is not None:
                    alpha = weights['intensity'] * weights['growth_weight']
                    alpha = np.clip(alpha, 0.2, 0.8)
                    final = advected * (1 - alpha) + corrected * alpha
                else:
                    final = corrected
            else:
                final = advected
            
            extrapolated.append(final)
        
        return extrapolated

    def compute_average_flow(self, images: List[np.ndarray]) -> np.ndarray:
        if len(images) < 2:
            raise ValueError("At least 2 images are required to compute optical flow")

        flows = []
        for i in range(len(images) - 1):
            flow = self._compute_flow(images[i], images[i + 1])
            flows.append(flow)

        avg_flow = np.mean(flows, axis=0)
        return avg_flow


class NowcastingPipeline:
    def __init__(self, num_history_frames: int = 6, 
                 max_lead_time: int = 120,
                 time_interval: int = 6,
                 **kwargs):
        self.num_history_frames = num_history_frames
        self.max_lead_time = max_lead_time
        self.time_interval = time_interval
        self.extrapolator = EnhancedRadarExtrapolator(**kwargs)
        
    def predict(self, history_images: List[np.ndarray], 
                lead_times: Optional[List[int]] = None) -> Dict:
        if lead_times is None:
            lead_times = list(range(self.time_interval, 
                                   self.max_lead_time + 1, 
                                   self.time_interval))
        
        num_steps = max(lead_times) // self.time_interval
        
        extrapolated = self.extrapolator.extrapolate_enhanced(
            history_images, steps=num_steps, time_interval=self.time_interval
        )
        
        predictions = {}
        for lt in lead_times:
            step_idx = (lt // self.time_interval) - 1
            if step_idx < len(extrapolated):
                predictions[lt] = extrapolated[step_idx]
            else:
                predictions[lt] = extrapolated[-1]
        
        diagnostics = {
            'flow_field': self.extrapolator.compute_average_flow(history_images),
            'growth_rate': self.extrapolator.growth_rate_map,
            'divergence': self.extrapolator.divergence_map,
            'lead_times': lead_times
        }
        
        return {
            'predictions': predictions,
            'diagnostics': diagnostics
        }


def calculate_csi(observed: np.ndarray, predicted: np.ndarray, threshold: float = 10) -> float:
    obs_mask = observed >= threshold
    pred_mask = predicted >= threshold

    hits = np.sum(obs_mask & pred_mask)
    misses = np.sum(obs_mask & ~pred_mask)
    false_alarms = np.sum(~obs_mask & pred_mask)

    total = hits + misses + false_alarms
    if total == 0:
        return 1.0

    csi = hits / total
    return csi


def calculate_pod(observed: np.ndarray, predicted: np.ndarray, threshold: float = 10) -> float:
    obs_mask = observed >= threshold
    pred_mask = predicted >= threshold

    hits = np.sum(obs_mask & pred_mask)
    misses = np.sum(obs_mask & ~pred_mask)

    total = hits + misses
    if total == 0:
        return 1.0

    pod = hits / total
    return pod


def calculate_far(observed: np.ndarray, predicted: np.ndarray, threshold: float = 10) -> float:
    obs_mask = observed >= threshold
    pred_mask = predicted >= threshold

    hits = np.sum(obs_mask & pred_mask)
    false_alarms = np.sum(~obs_mask & pred_mask)

    total = hits + false_alarms
    if total == 0:
        return 0.0

    far = false_alarms / total
    return far


def calculate_bias(observed: np.ndarray, predicted: np.ndarray, threshold: float = 10) -> float:
    obs_mask = observed >= threshold
    pred_mask = predicted >= threshold
    
    obs_count = np.sum(obs_mask)
    pred_count = np.sum(pred_mask)
    
    if obs_count == 0:
        return 1.0 if pred_count == 0 else float('inf')
    
    bias = pred_count / obs_count
    return bias
