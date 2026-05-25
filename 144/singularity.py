import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def unwrap_angle(angles):
    unwrapped = np.unwrap(2 * angles) / 2
    return unwrapped


def calculate_local_coherence(theta_field, x, y, radius=5):
    h, w = theta_field.shape
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
    
    patch = theta_field[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    
    sin_2theta = np.sin(2 * patch)
    cos_2theta = np.cos(2 * patch)
    coherence = np.sqrt(np.mean(sin_2theta)**2 + np.mean(cos_2theta)**2)
    
    return coherence


def calculate_poincare_index(theta_field, center_x, center_y, radius=3):
    h, w = theta_field.shape
    
    if (center_x - radius < 0 or center_x + radius >= w or
        center_y - radius < 0 or center_y + radius >= h):
        return 0.0
    
    num_points = 16
    angles = []
    
    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        x = int(center_x + radius * np.cos(t))
        y = int(center_y + radius * np.sin(t))
        
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        angles.append(theta_field[y, x])
    
    angles = np.array(angles)
    unwrapped = unwrap_angle(angles)
    
    total_change = 0.0
    for i in range(num_points):
        j = (i + 1) % num_points
        diff = unwrapped[j] - unwrapped[i]
        
        while diff > np.pi:
            diff -= np.pi
        while diff < -np.pi:
            diff += np.pi
        
        total_change += diff
    
    poincare_index = total_change / np.pi
    
    return poincare_index


def calculate_singularity_confidence(theta_field, x, y, poincare_idx, coherence_map=None):
    h, w = theta_field.shape
    
    local_coherence = calculate_local_coherence(theta_field, x, y, radius=8)
    
    expected_idx = 1.0 if poincare_idx > 0 else -1.0
    idx_error = abs(poincare_idx - expected_idx)
    idx_confidence = max(0, 1 - idx_error / 0.5)
    
    if coherence_map is not None:
        y_clamped = np.clip(y, 0, h - 1)
        x_clamped = np.clip(x, 0, w - 1)
        global_coherence = coherence_map[y_clamped, x_clamped]
        coherence_score = (local_coherence + global_coherence) / 2
    else:
        coherence_score = local_coherence
    
    confidence = idx_confidence * coherence_score
    
    return confidence, local_coherence, idx_confidence


def detect_singularities_robust(theta_field, coherence_map=None, 
                                 min_distance=20, 
                                 poincare_threshold=0.6,
                                 confidence_threshold=0.3,
                                 coherence_threshold=0.2):
    h, w = theta_field.shape
    poincare_map = np.zeros((h, w), dtype=np.float64)
    confidence_map = np.zeros((h, w), dtype=np.float64)
    
    radius = 5
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            if coherence_map is not None and coherence_map[y, x] < coherence_threshold:
                continue
            
            poincare_map[y, x] = calculate_poincare_index(theta_field, x, y, radius)
    
    core_candidates = []
    delta_candidates = []
    
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            idx = poincare_map[y, x]
            
            if abs(idx) < poincare_threshold:
                continue
            
            if coherence_map is not None and coherence_map[y, x] < coherence_threshold:
                continue
            
            confidence, local_coh, idx_conf = calculate_singularity_confidence(
                theta_field, x, y, idx, coherence_map
            )
            confidence_map[y, x] = confidence
            
            if confidence < confidence_threshold:
                continue
            
            if idx > poincare_threshold:
                core_candidates.append((x, y, idx, confidence, local_coh))
            elif idx < -poincare_threshold:
                delta_candidates.append((x, y, idx, confidence, local_coh))
    
    cores = non_maximum_suppression_with_confidence(core_candidates, min_distance, is_core=True)
    deltas = non_maximum_suppression_with_confidence(delta_candidates, min_distance, is_core=False)
    
    return cores, deltas, poincare_map, confidence_map


def non_maximum_suppression_with_confidence(candidates, min_distance, is_core=True):
    if not candidates:
        return []
    
    candidates.sort(key=lambda p: -p[3])
    
    selected = []
    
    for cand in candidates:
        x, y, idx, conf, coh = cand
        keep = True
        
        for sel in selected:
            sx, sy, _, _, _ = sel
            dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            if dist < min_distance:
                keep = False
                break
        
        if keep:
            selected.append(cand)
    
    return selected


def non_maximum_suppression(candidates, min_distance, is_core=True):
    if not candidates:
        return []
    
    if is_core:
        candidates.sort(key=lambda p: -p[2])
    else:
        candidates.sort(key=lambda p: p[2])
    
    selected = []
    
    for cand in candidates:
        x, y, idx = cand
        keep = True
        
        for sel in selected:
            sx, sy, _ = sel
            dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            if dist < min_distance:
                keep = False
                break
        
        if keep:
            selected.append(cand)
    
    return selected


def multiscale_singularity_detection(image, theta_field_func, scales=[1.0, 0.75, 0.5],
                                      min_distance=20, poincare_threshold=0.6,
                                      confidence_threshold=0.3, coherence_threshold=0.2):
    h, w = image.shape[:2]
    
    all_cores = []
    all_deltas = []
    scale_weights = []
    
    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        
        if scale < 1.0:
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scaled_image = image
        
        scaled_theta, scaled_coherence = theta_field_func(scaled_image)
        
        cores, deltas, poincare_map, conf_map = detect_singularities_robust(
            scaled_theta, scaled_coherence,
            min_distance=int(min_distance * scale),
            poincare_threshold=poincare_threshold,
            confidence_threshold=confidence_threshold,
            coherence_threshold=coherence_threshold
        )
        
        for core in cores:
            x, y, idx, conf, coh = core
            orig_x = x / scale
            orig_y = y / scale
            all_cores.append((orig_x, orig_y, idx, conf * scale, coh, scale))
        
        for delta in deltas:
            x, y, idx, conf, coh = delta
            orig_x = x / scale
            orig_y = y / scale
            all_deltas.append((orig_x, orig_y, idx, conf * scale, coh, scale))
        
        scale_weights.append(scale)
    
    final_cores = cluster_and_validate_singularities(all_cores, min_distance, is_core=True)
    final_deltas = cluster_and_validate_singularities(all_deltas, min_distance, is_core=False)
    
    return final_cores, final_deltas


def cluster_and_validate_singularities(candidates, min_distance, is_core=True):
    if not candidates:
        return []
    
    candidates.sort(key=lambda p: -p[3])
    
    clusters = []
    
    for cand in candidates:
        x, y, idx, conf, coh, scale = cand
        assigned = False
        
        for cluster in clusters:
            cx, cy = cluster['center']
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_distance:
                cluster['points'].append(cand)
                cluster['center'] = (
                    np.mean([p[0] for p in cluster['points']]),
                    np.mean([p[1] for p in cluster['points']])
                )
                assigned = True
                break
        
        if not assigned:
            clusters.append({
                'center': (x, y),
                'points': [cand]
            })
    
    result = []
    for cluster in clusters:
        points = cluster['points']
        num_scales = len(set(p[5] for p in points))
        
        if num_scales >= 2:
            cx, cy = cluster['center']
            avg_idx = np.mean([p[2] for p in points])
            avg_conf = np.mean([p[3] for p in points])
            avg_coh = np.mean([p[4] for p in points])
            result.append((int(cx), int(cy), avg_idx, avg_conf, avg_coh, num_scales))
    
    return result


def detect_complex_filter(theta_field, block_size=16):
    h, w = theta_field.shape
    
    vx = np.cos(2 * theta_field)
    vy = np.sin(2 * theta_field)
    
    kernel_size = block_size * 2 + 1
    y, x = np.mgrid[-block_size:block_size+1, -block_size:block_size+1]
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    core_filter = np.exp(1j * theta) * np.exp(-r**2 / (2 * (block_size/2)**2))
    delta_filter = np.exp(-1j * theta) * np.exp(-r**2 / (2 * (block_size/2)**2))
    
    complex_field = vx + 1j * vy
    
    from scipy.signal import convolve2d
    core_response = np.abs(convolve2d(complex_field, core_filter, mode='same'))
    delta_response = np.abs(convolve2d(complex_field, delta_filter, mode='same'))
    
    return core_response, delta_response


def extract_peak_points(response_map, threshold=0.5, min_distance=15, coherence_map=None, coherence_threshold=0.2):
    h, w = response_map.shape
    max_val = response_map.max()
    if max_val == 0:
        return []
    
    threshold = threshold * max_val
    candidates = []
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if coherence_map is not None and coherence_map[y, x] < coherence_threshold:
                continue
            
            val = response_map[y, x]
            if val > threshold:
                neighbors = response_map[y-1:y+2, x-1:x+2]
                if val == neighbors.max():
                    if coherence_map is not None:
                        candidates.append((x, y, val, coherence_map[y, x]))
                    else:
                        candidates.append((x, y, val, 1.0))
    
    candidates.sort(key=lambda p: -p[2])
    selected = []
    
    for cand in candidates:
        x, y, val, coh = cand
        keep = True
        
        for sel in selected:
            sx, sy, _, _ = sel
            dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            if dist < min_distance:
                keep = False
                break
        
        if keep:
            selected.append(cand)
    
    return selected


def visualize_singularities(image, cores, deltas, scale=1.0, show_confidence=False):
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    core_color = (0, 0, 255)
    delta_color = (255, 0, 0)
    
    for core in cores:
        if len(core) >= 4:
            x, y, idx, conf = core[:4]
        else:
            x, y, idx = core[:3]
            conf = 1.0
        
        x_scaled = int(x * scale)
        y_scaled = int(y * scale)
        
        radius = int(8 * conf + 4)
        cv2.circle(vis, (x_scaled, y_scaled), radius, core_color, 2)
        
        if show_confidence:
            cv2.putText(vis, f'C({conf:.2f})', (x_scaled + 10, y_scaled), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, core_color, 1)
        else:
            cv2.putText(vis, 'C', (x_scaled + 10, y_scaled), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, core_color, 2)
    
    for delta in deltas:
        if len(delta) >= 4:
            x, y, idx, conf = delta[:4]
        else:
            x, y, idx = delta[:3]
            conf = 1.0
        
        x_scaled = int(x * scale)
        y_scaled = int(y * scale)
        
        radius = int(8 * conf + 4)
        cv2.circle(vis, (x_scaled, y_scaled), radius, delta_color, 2)
        
        if show_confidence:
            cv2.putText(vis, f'D({conf:.2f})', (x_scaled + 10, y_scaled), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, delta_color, 1)
        else:
            cv2.putText(vis, 'D', (x_scaled + 10, y_scaled), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, delta_color, 2)
    
    return vis


def visualize_poincare_map(poincare_map):
    normalized = cv2.normalize(poincare_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return colored


def visualize_confidence_map(confidence_map):
    normalized = cv2.normalize(confidence_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
    return colored
