import numpy as np
import cv2
from scipy.interpolate import griddata
from typing import List, Tuple, Optional


class RadarExtrapolator:
    def __init__(self, method: str = 'farneback', **kwargs):
        self.method = method.lower()
        self.flow_params = kwargs
        self._init_flow_params()

    def _init_flow_params(self):
        if self.method == 'farneback':
            self.flow_params.setdefault('pyr_scale', 0.5)
            self.flow_params.setdefault('levels', 3)
            self.flow_params.setdefault('winsize', 15)
            self.flow_params.setdefault('iterations', 3)
            self.flow_params.setdefault('poly_n', 5)
            self.flow_params.setdefault('poly_sigma', 1.2)
            self.flow_params.setdefault('flags', 0)
        elif self.method == 'dis':
            self.flow_params.setdefault('preset', cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        elif self.method == 'lucas_kanade':
            self.flow_params.setdefault('winSize', (15, 15))
            self.flow_params.setdefault('maxLevel', 2)
            self.flow_params.setdefault('criteria', (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def _compute_flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        prev_norm = self._normalize_image(prev)
        curr_norm = self._normalize_image(curr)

        if self.method == 'farneback':
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
        elif self.method == 'dis':
            dis = cv2.DISOpticalFlow_create(self.flow_params['preset'])
            flow = dis.calc(prev_norm, curr_norm, None)
        elif self.method == 'lucas_kanade':
            flow = self._lucas_kanade_flow(prev_norm, curr_norm)
        else:
            raise ValueError(f"Unknown method: {self.method}")

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

    def compute_average_flow(self, images: List[np.ndarray]) -> np.ndarray:
        if len(images) < 2:
            raise ValueError("At least 2 images are required to compute optical flow")

        flows = []
        for i in range(len(images) - 1):
            flow = self._compute_flow(images[i], images[i + 1])
            flows.append(flow)

        avg_flow = np.mean(flows, axis=0)
        return avg_flow

    def extrapolate(self, image: np.ndarray, flow: np.ndarray, steps: int = 1, 
                    method: str = 'warp') -> List[np.ndarray]:
        extrapolated = []
        h, w = image.shape[:2]

        for step in range(1, steps + 1):
            scale_factor = step
            scaled_flow = flow * scale_factor

            if method == 'warp':
                ext_image = self._warp_image(image, scaled_flow)
            elif method == 'advection':
                ext_image = self._advect_image(image, scaled_flow)
            else:
                raise ValueError(f"Unknown extrapolation method: {method}")

            extrapolated.append(ext_image)

        return extrapolated

    def _warp_image(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        map_x = (x_coords + flow[:, :, 0]).astype(np.float32)
        map_y = (y_coords + flow[:, :, 1]).astype(np.float32)

        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped

    def _advect_image(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        src_x = x_coords - flow[:, :, 0]
        src_y = y_coords - flow[:, :, 1]

        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)

        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        fx = src_x - x0
        fy = src_y - y0

        fx = np.expand_dims(fx, axis=-1) if image.ndim == 3 else fx
        fy = np.expand_dims(fy, axis=-1) if image.ndim == 3 else fy

        v00 = image[y0, x0]
        v01 = image[y0, x1]
        v10 = image[y1, x0]
        v11 = image[y1, x1]

        top = v00 * (1 - fx) + v01 * fx
        bottom = v10 * (1 - fx) + v11 * fx
        result = top * (1 - fy) + bottom * fy

        return result

    def extrapolate_sequence(self, images: List[np.ndarray], lead_times: List[int],
                            time_interval: int = 6) -> List[np.ndarray]:
        if len(images) < 2:
            raise ValueError("At least 2 historical images are required")

        avg_flow = self.compute_average_flow(images)
        last_image = images[-1]

        max_lead = max(lead_times)
        num_steps = max_lead // time_interval

        all_extrapolated = self.extrapolate(last_image, avg_flow, steps=num_steps)

        result = []
        for lt in lead_times:
            step_idx = (lt // time_interval) - 1
            if step_idx < len(all_extrapolated):
                result.append(all_extrapolated[step_idx])
            else:
                result.append(all_extrapolated[-1])

        return result


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
