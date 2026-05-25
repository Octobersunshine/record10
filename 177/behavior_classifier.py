import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from feature_extractor import TrajectoryFeatures
from config import Config


class BehaviorType(Enum):
    RESTING = "resting"
    WALKING = "walking"
    FLYING = "flying"
    ESCAPING = "escaping"
    COURTING = "courting"
    FORAGING = "foraging"
    AGGRESSIVE = "aggressive"
    UNKNOWN = "unknown"


BEHAVIOR_COLORS = {
    BehaviorType.RESTING: (100, 100, 100),
    BehaviorType.WALKING: (0, 255, 0),
    BehaviorType.FLYING: (0, 0, 255),
    BehaviorType.ESCAPING: (255, 0, 0),
    BehaviorType.COURTING: (255, 0, 255),
    BehaviorType.FORAGING: (0, 255, 255),
    BehaviorType.AGGRESSIVE: (255, 128, 0),
    BehaviorType.UNKNOWN: (128, 128, 128)
}


@dataclass
class BehaviorSegment:
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    behavior: BehaviorType
    confidence: float
    feature_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class BehaviorClassification:
    track_id: int
    segments: List[BehaviorSegment] = field(default_factory=list)
    dominant_behavior: BehaviorType = BehaviorType.UNKNOWN
    behavior_distribution: Dict[BehaviorType, float] = field(default_factory=dict)


class HMMClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.n_states = len(BehaviorType) - 1
        self.n_features = 5
        
        self.transition_matrix = None
        self.emission_means = None
        self.emission_covs = None
        self.initial_dist = None
        
        self._initialize_default_model()
        
    def _initialize_default_model(self):
        self.initial_dist = np.array([
            0.15, 0.15, 0.15, 0.1, 0.15, 0.15, 0.15
        ])
        
        self.transition_matrix = np.array([
            [0.5, 0.15, 0.05, 0.05, 0.1, 0.1, 0.05],
            [0.15, 0.4, 0.15, 0.05, 0.1, 0.1, 0.05],
            [0.05, 0.1, 0.5, 0.15, 0.05, 0.1, 0.05],
            [0.05, 0.05, 0.15, 0.5, 0.05, 0.05, 0.15],
            [0.1, 0.1, 0.05, 0.05, 0.5, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.05, 0.1, 0.45, 0.1],
            [0.05, 0.05, 0.05, 0.15, 0.1, 0.1, 0.5]
        ])
        
        self.emission_means = np.array([
            [5.0, 10.0, 0.01, 0.05, 0.05],
            [20.0, 30.0, 0.05, 0.3, 0.2],
            [100.0, 50.0, 0.1, 0.5, 1.0],
            [150.0, 200.0, 0.15, 2.0, 2.0],
            [30.0, 20.0, 0.3, 0.8, 0.5],
            [50.0, 40.0, 0.2, 0.4, 0.3],
            [80.0, 100.0, 0.25, 1.5, 1.5]
        ])
        
        self.emission_covs = np.array([
            np.eye(self.n_features) * 5.0 for _ in range(self.n_states)
        ])
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        n = len(x)
        diff = x - mean
        
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        
        if cov_det < 1e-10:
            cov_det = 1e-10
        
        exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        norm = 1.0 / (np.sqrt((2 * np.pi) ** n * cov_det))
        
        return norm * np.exp(exponent)
    
    def _compute_emission_prob(self, observation: np.ndarray) -> np.ndarray:
        probs = np.zeros(self.n_states)
        
        for i in range(self.n_states):
            probs[i] = self._gaussian_pdf(
                observation, 
                self.emission_means[i], 
                self.emission_covs[i]
            )
        
        if np.sum(probs) < 1e-10:
            return np.ones(self.n_states) / self.n_states
        
        return probs / np.sum(probs)
    
    def forward_backward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(observations)
        
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))
        
        alpha[0] = self.initial_dist * self._compute_emission_prob(observations[0])
        alpha[0] /= np.sum(alpha[0])
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = self._compute_emission_prob(observations[t])[j] * \
                             np.sum(alpha[t-1] * self.transition_matrix[:, j])
            alpha[t] /= np.sum(alpha[t])
        
        beta[-1] = np.ones(self.n_states)
        
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix[i, :] * 
                    self._compute_emission_prob(observations[t+1]) * 
                    beta[t+1]
                )
            beta[t] /= np.sum(beta[t])
        
        return alpha, beta
    
    def viterbi(self, observations: np.ndarray) -> Tuple[List[int], np.ndarray]:
        T = len(observations)
        
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = np.log(self.initial_dist + 1e-10) + \
                   np.log(self._compute_emission_prob(observations[0]) + 1e-10)
        
        for t in range(1, T):
            probs = self._compute_emission_prob(observations[t])
            
            for j in range(self.n_states):
                trans_probs = delta[t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = trans_probs[psi[t, j]] + np.log(probs[j] + 1e-10)
        
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        
        probs = np.exp(delta - np.max(delta, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        return path.tolist(), probs
    
    def baum_welch(self, observations_list: List[np.ndarray], 
                    n_iterations: int = 20, 
                    tolerance: float = 1e-4):
        if not observations_list:
            return
        
        n_sequences = len(observations_list)
        
        for iteration in range(n_iterations):
            xi_sum = np.zeros((self.n_states, self.n_states))
            gamma_sum = np.zeros(self.n_states)
            means_num = np.zeros((self.n_states, self.n_features))
            covs_num = np.zeros((self.n_states, self.n_features, self.n_features))
            
            for observations in observations_list:
                T = len(observations)
                
                alpha, beta = self.forward_backward(observations)
                
                gamma = alpha * beta
                gamma /= np.sum(gamma, axis=1, keepdims=True)
                
                xi = np.zeros((T - 1, self.n_states, self.n_states))
                for t in range(T - 1):
                    probs = self._compute_emission_prob(observations[t + 1])
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi[t, i, j] = alpha[t, i] * \
                                         self.transition_matrix[i, j] * \
                                         probs[j] * beta[t + 1, j]
                    xi[t] /= np.sum(xi[t])
                
                xi_sum += np.sum(xi, axis=0)
                gamma_sum += np.sum(gamma[:-1], axis=0)
                
                for i in range(self.n_states):
                    for t in range(T):
                        means_num[i] += gamma[t, i] * observations[t]
                        diff = observations[t] - self.emission_means[i]
                        covs_num[i] += gamma[t, i] * np.outer(diff, diff)
            
            old_means = self.emission_means.copy()
            
            self.transition_matrix = xi_sum / (gamma_sum[:, np.newaxis] + 1e-10)
            self.transition_matrix /= np.sum(self.transition_matrix, axis=1, keepdims=True)
            
            for i in range(self.n_states):
                total_gamma = np.sum(gamma_sum)
                self.emission_means[i] = means_num[i] / (total_gamma + 1e-10)
                self.emission_covs[i] = covs_num[i] / (total_gamma + 1e-10)
                self.emission_covs[i] += np.eye(self.n_features) * 1e-6
            
            mean_change = np.max(np.abs(self.emission_means - old_means))
            if mean_change < tolerance:
                break
    
    def classify(self, features: TrajectoryFeatures) -> BehaviorClassification:
        if features.feature_matrix.shape[0] < 3:
            return BehaviorClassification(
                track_id=features.track_id,
                dominant_behavior=BehaviorType.UNKNOWN
            )
        
        observations = features.feature_matrix
        
        path, state_probs = self.viterbi(observations)
        
        behavior_types = [
            BehaviorType.RESTING,
            BehaviorType.WALKING,
            BehaviorType.FLYING,
            BehaviorType.ESCAPING,
            BehaviorType.COURTING,
            BehaviorType.FORAGING,
            BehaviorType.AGGRESSIVE
        ]
        
        segments = self._merge_segments(
            path, state_probs, features.timestamps, 
            features.positions[:, 0], features.positions[:, 1]
        )
        
        behavior_counts = {}
        for seg in segments:
            behavior_counts[seg.behavior] = behavior_counts.get(seg.behavior, 0) + 1
        
        total = sum(behavior_counts.values())
        behavior_dist = {b: c/total for b, c in behavior_counts.items()}
        
        dominant = max(behavior_counts, key=behavior_counts.get) if behavior_counts else BehaviorType.UNKNOWN
        
        return BehaviorClassification(
            track_id=features.track_id,
            segments=segments,
            dominant_behavior=dominant,
            behavior_distribution=behavior_dist
        )
    
    def _merge_segments(self, path: List[int], 
                       state_probs: np.ndarray,
                       timestamps: np.ndarray,
                       x_positions: np.ndarray,
                       y_positions: np.ndarray) -> List[BehaviorSegment]:
        behavior_types = [
            BehaviorType.RESTING,
            BehaviorType.WALKING,
            BehaviorType.FLYING,
            BehaviorType.ESCAPING,
            BehaviorType.COURTING,
            BehaviorType.FORAGING,
            BehaviorType.AGGRESSIVE
        ]
        
        segments = []
        current_state = path[0]
        start_idx = 0
        
        for i in range(1, len(path)):
            if path[i] != current_state:
                segment = BehaviorSegment(
                    start_time=timestamps[start_idx],
                    end_time=timestamps[i-1],
                    start_frame=int(start_idx),
                    end_frame=int(i-1),
                    behavior=behavior_types[current_state],
                    confidence=np.mean(state_probs[start_idx:i, current_state])
                )
                segments.append(segment)
                
                current_state = path[i]
                start_idx = i
        
        if start_idx < len(path):
            segment = BehaviorSegment(
                start_time=timestamps[start_idx],
                end_time=timestamps[-1],
                start_frame=int(start_idx),
                end_frame=int(len(path)-1),
                behavior=behavior_types[current_state],
                confidence=np.mean(state_probs[start_idx:, current_state])
            )
            segments.append(segment)
        
        return self._merge_short_segments(segments, min_duration=0.5)
    
    def _merge_short_segments(self, segments: List[BehaviorSegment], 
                              min_duration: float = 0.5) -> List[BehaviorSegment]:
        if len(segments) <= 1:
            return segments
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            duration = seg.end_time - seg.start_time
            
            if duration < min_duration and merged:
                merged[-1].end_time = seg.end_time
                merged[-1].end_frame = seg.end_frame
                merged[-1].confidence = (merged[-1].confidence + seg.confidence) / 2
            else:
                merged.append(seg)
        
        return merged
    
    def train(self, training_data: List[Tuple[np.ndarray, List[int]]]):
        observations_list = [data[0] for data in training_data]
        self.baum_welch(observations_list, n_iterations=50)


class LSTMClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.device = None
        self.is_available = False
        
        self._try_initialize()
        
    def _try_initialize(self):
        try:
            import torch
            import torch.nn as nn
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            class BehaviorLSTM(nn.Module):
                def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=7):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                       batch_first=True, dropout=0.2)
                    self.fc1 = nn.Linear(hidden_size, 32)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.3)
                    self.fc2 = nn.Linear(32, num_classes)
                    
                def forward(self, x):
                    batch_size = x.size(0)
                    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
                    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
                    
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc1(out[:, -1, :])
                    out = self.relu(out)
                    out = self.dropout(out)
                    out = self.fc2(out)
                    
                    return out
            
            self.model = BehaviorLSTM()
            self.is_available = True
            
        except ImportError:
            self.is_available = False
            print("PyTorch not available. LSTM classifier disabled.")
    
    def load_model(self, model_path: str):
        if not self.is_available:
            return
        
        import torch
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"LSTM model loaded from {model_path}")
    
    def classify(self, features: TrajectoryFeatures) -> BehaviorClassification:
        if not self.is_available:
            return BehaviorClassification(
                track_id=features.track_id,
                dominant_behavior=BehaviorType.UNKNOWN
            )
        
        import torch
        
        observations = features.feature_matrix
        
        if len(observations) < 10:
            return BehaviorClassification(
                track_id=features.track_id,
                dominant_behavior=BehaviorType.UNKNOWN
            )
        
        seq_length = min(len(observations), 100)
        observations = observations[-seq_length:]
        
        with torch.no_grad():
            inputs = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        behavior_types = [
            BehaviorType.RESTING,
            BehaviorType.WALKING,
            BehaviorType.FLYING,
            BehaviorType.ESCAPING,
            BehaviorType.COURTING,
            BehaviorType.FORAGING,
            BehaviorType.AGGRESSIVE
        ]
        
        behavior_idx = predicted.item()
        dominant = behavior_types[behavior_idx]
        
        behavior_dist = {b: float(probabilities[i]) for i, b in enumerate(behavior_types)}
        
        segment = BehaviorSegment(
            start_time=features.timestamps[0],
            end_time=features.timestamps[-1],
            start_frame=0,
            end_frame=len(features.timestamps) - 1,
            behavior=dominant,
            confidence=float(probabilities[behavior_idx])
        )
        
        return BehaviorClassification(
            track_id=features.track_id,
            segments=[segment],
            dominant_behavior=dominant,
            behavior_distribution=behavior_dist
        )


class BehaviorClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.hmm_classifier = HMMClassifier(config)
        self.lstm_classifier = LSTMClassifier(config)
        self.use_lstm = config.get('USE_LSTM_BEHAVIOR', False)
        
        self.classifications: Dict[int, BehaviorClassification] = {}
        
    def classify_trajectory(self, features: TrajectoryFeatures) -> BehaviorClassification:
        if self.use_lstm and self.lstm_classifier.is_available:
            classification = self.lstm_classifier.classify(features)
        else:
            classification = self.hmm_classifier.classify(features)
        
        self.classifications[features.track_id] = classification
        
        return classification
    
    def classify_all(self, features_list: List[TrajectoryFeatures]) -> Dict[int, BehaviorClassification]:
        for features in features_list:
            self.classify_trajectory(features)
        
        return self.classifications
    
    def get_behavior_label(self, behavior: BehaviorType) -> str:
        labels = {
            BehaviorType.RESTING: "静止",
            BehaviorType.WALKING: "爬行",
            BehaviorType.FLYING: "飞行",
            BehaviorType.ESCAPING: "逃逸",
            BehaviorType.COURTING: "求偶",
            BehaviorType.FORAGING: "觅食",
            BehaviorType.AGGRESSIVE: "攻击",
            BehaviorType.UNKNOWN: "未知"
        }
        return labels.get(behavior, "未知")
    
    def visualize_behavior(self, frame: np.ndarray, 
                          classification: BehaviorClassification,
                          position: Tuple[int, int]) -> np.ndarray:
        vis_frame = frame.copy()
        
        x, y = position
        color = BEHAVIOR_COLORS.get(classification.dominant_behavior, (128, 128, 128))
        
        label = self.get_behavior_label(classification.dominant_behavior)
        
        cv2.rectangle(vis_frame, (x - 40, y - 25), (x + 40, y - 5), color, -1)
        cv2.putText(vis_frame, label, (x - 35, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def generate_behavior_report(self) -> str:
        report_lines = ["=" * 60, "行为分类报告", "=" * 60]
        
        for track_id, classification in sorted(self.classifications.items()):
            report_lines.append(f"\n轨迹 {track_id}:")
            report_lines.append(f"  主要行为: {self.get_behavior_label(classification.dominant_behavior)}")
            report_lines.append(f"  行为分布:")
            
            for behavior, prob in sorted(classification.behavior_distribution.items(), 
                                         key=lambda x: -x[1]):
                label = self.get_behavior_label(behavior)
                report_lines.append(f"    {label}: {prob:.2%}")
            
            report_lines.append(f"  行为片段数: {len(classification.segments)}")
            
            for i, seg in enumerate(classification.segments[:5]):
                label = self.get_behavior_label(seg.behavior)
                duration = seg.end_time - seg.start_time
                report_lines.append(f"    片段 {i+1}: {label} ({duration:.2f}s, 置信度: {seg.confidence:.2f})")
        
        return "\n".join(report_lines)
    
    def save_classifications(self, filepath: str):
        import csv
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'track_id', 'dominant_behavior', 'dominant_behavior_label',
                'resting_prob', 'walking_prob', 'flying_prob',
                'escaping_prob', 'courting_prob', 'foraging_prob',
                'aggressive_prob', 'num_segments'
            ])
            
            for track_id, classification in sorted(self.classifications.items()):
                dist = classification.behavior_distribution
                
                writer.writerow([
                    track_id,
                    classification.dominant_behavior.value,
                    self.get_behavior_label(classification.dominant_behavior),
                    f"{dist.get(BehaviorType.RESTING, 0):.4f}",
                    f"{dist.get(BehaviorType.WALKING, 0):.4f}",
                    f"{dist.get(BehaviorType.FLYING, 0):.4f}",
                    f"{dist.get(BehaviorType.ESCAPING, 0):.4f}",
                    f"{dist.get(BehaviorType.COURTING, 0):.4f}",
                    f"{dist.get(BehaviorType.FORAGING, 0):.4f}",
                    f"{dist.get(BehaviorType.AGGRESSIVE, 0):.4f}",
                    len(classification.segments)
                ])
        
        print(f"Behavior classifications saved to {filepath}")
