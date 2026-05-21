import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from dtw import DTW, FastDTW, OnlineDTW
from gesture_processor import HandKeypointProcessor, smooth_sequence, sample_sequence


class StreamingGestureRecognizer:
    def __init__(self, 
                 base_recognizer: 'GestureRecognizer',
                 confidence_threshold: float = 0.5,
                 early_stop_threshold: float = float('inf'),
                 min_frames_for_detection: int = 5,
                 max_frames_for_detection: int = 120,
                 smoothing_window: int = 3):
        self.base_recognizer = base_recognizer
        self.confidence_threshold = confidence_threshold
        self.early_stop_threshold = early_stop_threshold
        self.min_frames_for_detection = min_frames_for_detection
        self.max_frames_for_detection = max_frames_for_detection
        self.smoothing_window = smoothing_window
        
        self.online_matchers: Dict[str, List[OnlineDTW]] = {}
        self.current_frame_buffer: List[np.ndarray] = []
        self.detection_history: List[Dict] = []
        self.best_gesture_so_far: Optional[Tuple[str, float]] = None
        self.frame_count = 0
        
        self._init_online_matchers()
    
    def _init_online_matchers(self) -> None:
        norm_params = (self.base_recognizer.feature_mean, self.base_recognizer.feature_std) \
                     if self.base_recognizer.feature_mean is not None else None
        
        for gesture_name, template in self.base_recognizer.templates.items():
            self.online_matchers[gesture_name] = []
            for template_seq in template.sequences:
                matcher = OnlineDTW(
                    template=template_seq,
                    distance_func=self.base_recognizer.distance_func,
                    use_normalization=self.base_recognizer.use_dim_normalization,
                    weights=self.base_recognizer.feature_weights,
                    norm_params=norm_params,
                    early_stop_threshold=self.early_stop_threshold,
                    min_frames_for_early_stop=self.min_frames_for_detection
                )
                self.online_matchers[gesture_name].append(matcher)
    
    def reset(self) -> None:
        for gesture_matchers in self.online_matchers.values():
            for matcher in gesture_matchers:
                matcher.reset()
        
        self.current_frame_buffer = []
        self.detection_history = []
        self.best_gesture_so_far = None
        self.frame_count = 0
    
    def _extract_features(self, keypoints: np.ndarray) -> np.ndarray:
        if len(keypoints.shape) == 2:
            keypoints = keypoints[np.newaxis, ...]
        
        features = self.base_recognizer._extract_features(keypoints)
        return features.reshape(features.shape[0], -1)[0]
    
    def update(self, keypoints: np.ndarray) -> Tuple[Optional[str], float, Dict[str, Any]]:
        self.frame_count += 1
        
        features = self._extract_features(keypoints)
        self.current_frame_buffer.append(features)
        
        if len(self.current_frame_buffer) > self.max_frames_for_detection:
            self.current_frame_buffer.pop(0)
        
        gesture_distances = {}
        all_completed = True
        
        for gesture_name, matchers in self.online_matchers.items():
            distances = []
            gesture_completed = True
            
            for matcher in matchers:
                if not matcher.is_complete:
                    dist, completed, status = matcher.update(features)
                    distances.append(dist)
                    if not completed:
                        gesture_completed = False
                else:
                    distances.append(matcher.final_distance)
            
            gesture_distances[gesture_name] = min(distances) if distances else float('inf')
            
            if not gesture_completed:
                all_completed = False
        
        sorted_gestures = sorted(gesture_distances.items(), key=lambda x: x[1])
        best_gesture, best_distance = sorted_gestures[0]
        
        second_best_distance = sorted_gestures[1][1] if len(sorted_gestures) > 1 else float('inf')
        confidence = self._calculate_confidence(best_distance, second_best_distance)
        
        detection_result = {
            'frame': self.frame_count,
            'best_gesture': best_gesture,
            'best_distance': best_distance,
            'confidence': confidence,
            'all_distances': gesture_distances,
            'buffer_length': len(self.current_frame_buffer),
            'all_completed': all_completed
        }
        self.detection_history.append(detection_result)
        
        if confidence > self.best_gesture_so_far[1] if self.best_gesture_so_far else -1:
            self.best_gesture_so_far = (best_gesture, confidence)
        
        should_detect = (
            confidence >= self.confidence_threshold and
            len(self.current_frame_buffer) >= self.min_frames_for_detection
        )
        
        if should_detect:
            return best_gesture, confidence, detection_result
        else:
            return None, confidence, detection_result
    
    def _calculate_confidence(self, best_distance: float, second_best_distance: float) -> float:
        if best_distance == float('inf'):
            return 0.0
        
        margin = second_best_distance - best_distance if second_best_distance != float('inf') else 1.0
        
        distance_score = 1.0 / (1.0 + best_distance)
        
        margin_score = min(1.0, margin / max(best_distance, 0.001))
        
        confidence = 0.7 * distance_score + 0.3 * margin_score
        
        return max(0.0, min(1.0, confidence))
    
    def get_smoothed_confidence(self, window_size: Optional[int] = None) -> Dict[str, float]:
        if window_size is None:
            window_size = self.smoothing_window
        
        if len(self.detection_history) < window_size:
            return {}
        
        recent_history = self.detection_history[-window_size:]
        gesture_confidences = {}
        
        for entry in recent_history:
            gesture = entry['best_gesture']
            conf = entry['confidence']
            if gesture not in gesture_confidences:
                gesture_confidences[gesture] = []
            gesture_confidences[gesture].append(conf)
        
        smoothed = {g: np.mean(confs) for g, confs in gesture_confidences.items()}
        return smoothed
    
    def get_progress(self) -> Dict[str, float]:
        progress = {}
        for gesture_name, matchers in self.online_matchers.items():
            gesture_progress = []
            for matcher in matchers:
                status = matcher._get_status()
                gesture_progress.append(status['progress'])
            progress[gesture_name] = np.mean(gesture_progress)
        return progress
    
    def force_detection(self) -> Tuple[str, float]:
        if not self.detection_history:
            return "unknown", 0.0
        
        if self.best_gesture_so_far:
            return self.best_gesture_so_far
        
        best_entry = max(self.detection_history, key=lambda x: x['confidence'])
        return best_entry['best_gesture'], best_entry['confidence']
    
    def get_matching_status(self) -> Dict[str, Dict]:
        status = {}
        for gesture_name, matchers in self.online_matchers.items():
            matcher_statuses = []
            for matcher in matchers:
                matcher_statuses.append({
                    'is_complete': matcher.is_complete,
                    'final_distance': matcher.final_distance,
                    'progress': matcher._get_status()['progress']
                })
            status[gesture_name] = matcher_statuses
        return status


class GestureTemplate:
    def __init__(self, name: str, sequences: List[np.ndarray]):
        self.name = name
        self.sequences = sequences
        self.mean_sequence = None
        self.std_sequence = None
        
        if sequences:
            self._compute_statistics()

    def _compute_statistics(self):
        max_len = max(len(seq) for seq in self.sequences)
        aligned_sequences = []
        
        for seq in self.sequences:
            aligned = sample_sequence(seq, max_len)
            aligned_sequences.append(aligned)
        
        aligned_sequences = np.array(aligned_sequences)
        self.mean_sequence = np.mean(aligned_sequences, axis=0)
        self.std_sequence = np.std(aligned_sequences, axis=0)


class GestureRecognizer:
    def __init__(self, 
                 distance_func: str = 'euclidean',
                 use_fastdtw: bool = False,
                 threshold: float = 1.0,
                 feature_type: str = 'keypoints',
                 use_dim_normalization: bool = True,
                 use_dim_weights: bool = True,
                 weight_type: str = 'inverse_variance'):
        self.templates: Dict[str, GestureTemplate] = {}
        self.processor = HandKeypointProcessor()
        self.threshold = threshold
        self.feature_type = feature_type
        self.use_dim_normalization = use_dim_normalization
        self.use_dim_weights = use_dim_weights
        self.weight_type = weight_type
        self.distance_func = distance_func
        
        self.feature_mean = None
        self.feature_std = None
        self.feature_weights = None
        
        if use_fastdtw:
            self.dtw = FastDTW(
                distance_func=distance_func,
                use_normalization=use_dim_normalization
            )
        else:
            self.dtw = DTW(
                distance_func=distance_func,
                use_normalization=use_dim_normalization
            )

    def _extract_features(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        if self.feature_type == 'keypoints':
            return self.processor.normalize_keypoints(keypoints_sequence)
        elif self.feature_type == 'full':
            return self.processor.extract_features(keypoints_sequence)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _compute_normalization_params(self, all_sequences: List[np.ndarray]) -> None:
        if not all_sequences:
            return
        
        all_features = np.vstack(all_sequences)
        
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0)
        self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)
        
        if self.use_dim_weights:
            variances = np.var(all_features, axis=0)
            if self.weight_type == 'inverse_variance':
                weights = 1.0 / (variances + 1e-8)
            elif self.weight_type == 'variance':
                weights = variances
            else:
                weights = np.ones_like(variances)
            
            self.feature_weights = weights / np.sum(weights) * len(weights)
        
        self.dtw.mean = self.feature_mean
        self.dtw.std = self.feature_std
        self.dtw.weights = self.feature_weights

    def add_template(self, gesture_name: str, 
                     keypoints_sequences: List[np.ndarray],
                     augment: bool = True,
                     num_augments: int = 3) -> None:
        from gesture_processor import GestureDataAugmenter
        
        processed_sequences = []
        
        for seq in keypoints_sequences:
            features = self._extract_features(seq)
            features_flat = features.reshape(len(features), -1)
            processed_sequences.append(features_flat)
            
            if augment:
                augmenter = GestureDataAugmenter()
                augmented = augmenter.augment_sequence(features_flat, num_augments)
                processed_sequences.extend(augmented[1:])
        
        self.templates[gesture_name] = GestureTemplate(gesture_name, processed_sequences)
        
        all_sequences = []
        for template in self.templates.values():
            all_sequences.extend(template.sequences)
        self._compute_normalization_params(all_sequences)

    def get_dimension_stats(self) -> Dict[str, np.ndarray]:
        stats = {
            'mean': self.feature_mean,
            'std': self.feature_std,
            'weights': self.feature_weights
        }
        return stats

    def analyze_dimension_importance(self) -> Dict[str, np.ndarray]:
        all_sequences = []
        for template in self.templates.values():
            all_sequences.extend(template.sequences)
        
        if not all_sequences:
            return {}
        
        all_features = np.vstack(all_sequences)
        variances = np.var(all_features, axis=0)
        means = np.mean(all_features, axis=0)
        ranges = np.max(all_features, axis=0) - np.min(all_features, axis=0)
        
        importance = variances / np.sum(variances)
        
        return {
            'variance': variances,
            'mean': means,
            'range': ranges,
            'importance': importance
        }

    def recognize(self, keypoints_sequence: np.ndarray, 
                  top_k: int = 3) -> List[Tuple[str, float]]:
        if not self.templates:
            return []
        
        features = self._extract_features(keypoints_sequence)
        features_flat = features.reshape(len(features), -1)
        
        results = []
        
        for gesture_name, template in self.templates.items():
            min_distance = float('inf')
            
            for template_seq in template.sequences:
                if hasattr(self.dtw, 'normalized_distance'):
                    distance = self.dtw.normalized_distance(features_flat, template_seq)
                else:
                    distance, _ = self.dtw.compute(features_flat, template_seq)
                    distance = distance / max(len(features_flat), len(template_seq))
                
                if distance < min_distance:
                    min_distance = distance
            
            results.append((gesture_name, min_distance))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def recognize_with_threshold(self, keypoints_sequence: np.ndarray) -> Optional[Tuple[str, float]]:
        results = self.recognize(keypoints_sequence, top_k=1)
        
        if results and results[0][1] <= self.threshold:
            return results[0]
        return None

    def get_recognition_score(self, keypoints_sequence: np.ndarray, 
                              gesture_name: str) -> float:
        if gesture_name not in self.templates:
            return float('inf')
        
        features = self._extract_features(keypoints_sequence)
        features_flat = features.reshape(len(features), -1)
        
        template = self.templates[gesture_name]
        distances = []
        
        for template_seq in template.sequences:
            if hasattr(self.dtw, 'normalized_distance'):
                distance = self.dtw.normalized_distance(features_flat, template_seq)
            else:
                distance, _ = self.dtw.compute(features_flat, template_seq)
                distance = distance / max(len(features_flat), len(template_seq))
            distances.append(distance)
        
        return np.mean(distances)

    def save_templates(self, filepath: str) -> None:
        templates_data = {}
        
        for name, template in self.templates.items():
            templates_data[name] = {
                'sequences': [seq.tolist() for seq in template.sequences],
                'mean_sequence': template.mean_sequence.tolist() if template.mean_sequence is not None else None,
                'std_sequence': template.std_sequence.tolist() if template.std_sequence is not None else None
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'templates': templates_data,
                'threshold': self.threshold,
                'feature_type': self.feature_type,
                'use_dim_normalization': self.use_dim_normalization,
                'use_dim_weights': self.use_dim_weights,
                'weight_type': self.weight_type,
                'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
                'feature_std': self.feature_std.tolist() if self.feature_std is not None else None,
                'feature_weights': self.feature_weights.tolist() if self.feature_weights is not None else None
            }, f)

    def load_templates(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.threshold = data.get('threshold', 1.0)
        self.feature_type = data.get('feature_type', 'keypoints')
        self.use_dim_normalization = data.get('use_dim_normalization', True)
        self.use_dim_weights = data.get('use_dim_weights', True)
        self.weight_type = data.get('weight_type', 'inverse_variance')
        
        if data.get('feature_mean') is not None:
            self.feature_mean = np.array(data['feature_mean'])
        if data.get('feature_std') is not None:
            self.feature_std = np.array(data['feature_std'])
        if data.get('feature_weights') is not None:
            self.feature_weights = np.array(data['feature_weights'])
        
        self.dtw.mean = self.feature_mean
        self.dtw.std = self.feature_std
        self.dtw.weights = self.feature_weights
        
        for name, template_data in data['templates'].items():
            sequences = [np.array(seq) for seq in template_data['sequences']]
            template = GestureTemplate(name, sequences)
            if template_data['mean_sequence'] is not None:
                template.mean_sequence = np.array(template_data['mean_sequence'])
            if template_data['std_sequence'] is not None:
                template.std_sequence = np.array(template_data['std_sequence'])
            self.templates[name] = template

    def list_gestures(self) -> List[str]:
        return list(self.templates.keys())

    def remove_template(self, gesture_name: str) -> None:
        if gesture_name in self.templates:
            del self.templates[gesture_name]

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold


class RealTimeGestureRecognizer:
    def __init__(self, recognizer: GestureRecognizer, 
                 min_sequence_length: int = 10,
                 max_sequence_length: int = 60,
                 smoothing_window: int = 3):
        self.recognizer = recognizer
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.smoothing_window = smoothing_window
        
        self.current_sequence: List[np.ndarray] = []
        self.is_recording: bool = False

    def start_recording(self) -> None:
        self.current_sequence = []
        self.is_recording = True

    def stop_recording(self) -> Optional[List[Tuple[str, float]]]:
        self.is_recording = False
        
        if len(self.current_sequence) >= self.min_sequence_length:
            keypoints_sequence = np.array(self.current_sequence)
            
            if self.smoothing_window > 1:
                smoothed_sequence = smooth_sequence(
                    keypoints_sequence.reshape(len(keypoints_sequence), -1),
                    self.smoothing_window
                )
                keypoints_sequence = smoothed_sequence.reshape(keypoints_sequence.shape)
            
            results = self.recognizer.recognize(keypoints_sequence)
            return results
        
        return None

    def add_frame(self, keypoints: np.ndarray) -> Optional[List[Tuple[str, float]]]:
        if not self.is_recording:
            return None
        
        self.current_sequence.append(keypoints.copy())
        
        if len(self.current_sequence) > self.max_sequence_length:
            self.current_sequence.pop(0)
        
        if len(self.current_sequence) >= self.min_sequence_length:
            keypoints_sequence = np.array(self.current_sequence)
            
            if self.smoothing_window > 1:
                smoothed_sequence = smooth_sequence(
                    keypoints_sequence.reshape(len(keypoints_sequence), -1),
                    self.smoothing_window
                )
                keypoints_sequence = smoothed_sequence.reshape(keypoints_sequence.shape)
            
            results = self.recognizer.recognize(keypoints_sequence)
            return results
        
        return None

    def get_current_sequence_length(self) -> int:
        return len(self.current_sequence)

    def clear_sequence(self) -> None:
        self.current_sequence = []


def create_sample_gestures() -> Dict[str, List[np.ndarray]]:
    gestures = {}
    
    n_frames_swipe = 30
    base_keypoints = np.random.rand(21, 3) * 0.5
    
    swipe_right = []
    for i in range(n_frames_swipe):
        kp = base_keypoints.copy()
        kp[:, 0] += i / n_frames_swipe
        swipe_right.append(kp)
    gestures['swipe_right'] = [np.array(swipe_right)]
    
    swipe_left = []
    for i in range(n_frames_swipe):
        kp = base_keypoints.copy()
        kp[:, 0] -= i / n_frames_swipe
        swipe_left.append(kp)
    gestures['swipe_left'] = [np.array(swipe_left)]
    
    n_frames_click = 15
    click_down = []
    for i in range(n_frames_click):
        kp = base_keypoints.copy()
        if i < n_frames_click // 2:
            kp[8, 1] -= i / (n_frames_click // 2) * 0.3
        else:
            kp[8, 1] -= (n_frames_click - i) / (n_frames_click // 2) * 0.3
        click_down.append(kp)
    gestures['click'] = [np.array(click_down)]
    
    return gestures