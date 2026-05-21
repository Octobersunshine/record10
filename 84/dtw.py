import numpy as np
from typing import Tuple, Optional, Union, List, Dict


class OnlineDTW:
    def __init__(self, 
                 template: np.ndarray,
                 distance_func: str = 'euclidean',
                 use_normalization: bool = True,
                 weights: Optional[np.ndarray] = None,
                 norm_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 beam_width: int = 5,
                 early_stop_threshold: float = float('inf'),
                 min_frames_for_early_stop: int = 10):
        self.template = template
        self.template_len = len(template)
        self.distance_func = distance_func
        self.use_normalization = use_normalization
        self.weights = weights
        self.mean = norm_params[0] if norm_params is not None else None
        self.std = norm_params[1] if norm_params is not None else None
        self.beam_width = beam_width
        self.early_stop_threshold = early_stop_threshold
        self.min_frames_for_early_stop = min_frames_for_early_stop
        
        self.reset()
    
    def reset(self) -> None:
        self.current_sequence: List[np.ndarray] = []
        self.cost_matrix: List[np.ndarray] = []
        self.best_paths: List[List[int]] = []
        self.current_min_cost = float('inf')
        self.best_path = []
        self.is_complete = False
        self.final_distance = None
    
    def _calculate_point_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.use_normalization and self.mean is not None and self.std is not None:
            x_norm = (x - self.mean) / self.std
            y_norm = (y - self.mean) / self.std
        else:
            x_norm = x
            y_norm = y
        
        diff = x_norm - y_norm
        if self.weights is not None:
            diff = diff * self.weights
        
        if self.distance_func == 'euclidean':
            return np.sqrt(np.sum(diff ** 2))
        elif self.distance_func == 'manhattan':
            return np.sum(np.abs(diff))
        elif self.distance_func == 'cosine':
            dot_product = np.dot(x_norm, y_norm)
            norm_x = np.linalg.norm(x_norm)
            norm_y = np.linalg.norm(y_norm)
            if norm_x == 0 or norm_y == 0:
                return 1.0
            return 1 - (dot_product / (norm_x * norm_y))
        else:
            raise ValueError(f"Unknown distance function: {self.distance_func}")
    
    def update(self, new_frame: np.ndarray) -> Tuple[float, bool, Dict]:
        if self.is_complete:
            return self.final_distance, True, self._get_status()
        
        self.current_sequence.append(new_frame)
        i = len(self.current_sequence) - 1
        
        if i == 0:
            cost_row = np.zeros(self.template_len)
            cost_row[0] = self._calculate_point_distance(new_frame, self.template[0])
            for j in range(1, self.template_len):
                cost_row[j] = cost_row[j-1] + self._calculate_point_distance(new_frame, self.template[j])
            self.cost_matrix.append(cost_row)
            self.best_paths.append([0] * self.template_len)
        else:
            prev_cost = self.cost_matrix[-1]
            curr_cost = np.zeros(self.template_len)
            curr_path = np.zeros(self.template_len, dtype=int)
            
            curr_cost[0] = prev_cost[0] + self._calculate_point_distance(new_frame, self.template[0])
            curr_path[0] = 0
            
            for j in range(1, self.template_len):
                match_cost = prev_cost[j-1]
                insert_cost = prev_cost[j]
                delete_cost = curr_cost[j-1]
                
                min_cost = min(match_cost, insert_cost, delete_cost)
                curr_cost[j] = min_cost + self._calculate_point_distance(new_frame, self.template[j])
                
                if min_cost == match_cost:
                    curr_path[j] = 0
                elif min_cost == insert_cost:
                    curr_path[j] = 1
                else:
                    curr_path[j] = 2
            
            self.cost_matrix.append(curr_cost)
            self.best_paths.append(curr_path.tolist())
        
        self.current_min_cost = self.cost_matrix[-1][-1]
        
        should_stop = self._check_early_stop()
        
        if len(self.current_sequence) >= self.template_len or should_stop:
            self._finalize()
        
        return self.current_min_cost / len(self.current_sequence), self.is_complete, self._get_status()
    
    def _check_early_stop(self) -> bool:
        if len(self.current_sequence) < self.min_frames_for_early_stop:
            return False
        
        if self.current_min_cost > self.early_stop_threshold * len(self.current_sequence):
            return True
        
        return False
    
    def _finalize(self) -> None:
        self.is_complete = True
        self.final_distance = self.current_min_cost / len(self.current_sequence)
        
        i = len(self.current_sequence) - 1
        j = self.template_len - 1
        path = []
        
        while i >= 0 and j >= 0:
            path.append((i, j))
            if i == 0 and j == 0:
                break
            elif i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                step = self.best_paths[i][j]
                if step == 0:
                    i -= 1
                    j -= 1
                elif step == 1:
                    i -= 1
                else:
                    j -= 1
        
        self.best_path = path[::-1]
    
    def _get_status(self) -> Dict:
        progress = len(self.current_sequence) / max(self.template_len, 1) * 100
        return {
            'current_length': len(self.current_sequence),
            'template_length': self.template_len,
            'progress': progress,
            'current_min_cost': self.current_min_cost,
            'normalized_distance': self.current_min_cost / len(self.current_sequence) if self.current_sequence else 0.0
        }
    
    def get_alignment_path(self) -> List[Tuple[int, int]]:
        return self.best_path
    
    def get_partial_alignment(self) -> List[Tuple[int, int]]:
        if not self.best_path:
            return []
        return self.best_path


class DTW:
    def __init__(self, 
                 distance_func: str = 'euclidean',
                 use_normalization: bool = True,
                 weights: Optional[np.ndarray] = None,
                 norm_params: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.distance_func = distance_func
        self.use_normalization = use_normalization
        self.weights = weights
        self.mean = norm_params[0] if norm_params is not None else None
        self.std = norm_params[1] if norm_params is not None else None

    def _normalize_vector(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_normalization:
            return x, y
        
        if self.mean is None or self.std is None:
            combined = np.vstack([x, y])
            mean = np.mean(combined, axis=0)
            std = np.std(combined, axis=0)
            std = np.where(std == 0, 1.0, std)
        else:
            mean = self.mean
            std = self.std
        
        x_norm = (x - mean) / std
        y_norm = (y - mean) / std
        
        return x_norm, y_norm

    def _apply_weights(self, diff: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return diff
        return diff * self.weights

    def _calculate_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        x_norm, y_norm = self._normalize_vector(x, y)
        
        diff = x_norm - y_norm
        diff = self._apply_weights(diff)
        
        if self.distance_func == 'euclidean':
            return np.sqrt(np.sum(diff ** 2))
        elif self.distance_func == 'manhattan':
            return np.sum(np.abs(diff))
        elif self.distance_func == 'cosine':
            dot_product = np.dot(x_norm, y_norm)
            norm_x = np.linalg.norm(x_norm)
            norm_y = np.linalg.norm(y_norm)
            if norm_x == 0 or norm_y == 0:
                return 1.0
            return 1 - (dot_product / (norm_x * norm_y))
        elif self.distance_func == 'mahalanobis':
            return np.sqrt(np.sum(diff ** 2))
        elif self.distance_func == 'weighted_euclidean':
            return np.sqrt(np.sum(diff ** 2))
        else:
            raise ValueError(f"Unknown distance function: {self.distance_func}")

    def fit_normalization(self, sequences: list) -> None:
        all_data = np.vstack(sequences)
        self.mean = np.mean(all_data, axis=0)
        self.std = np.std(all_data, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)

    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = weights

    def calculate_dimension_importance(self, sequences: list) -> np.ndarray:
        all_data = np.vstack(sequences)
        variances = np.var(all_data, axis=0)
        importance = variances / np.sum(variances)
        return importance

    def compute(self, seq1: np.ndarray, seq2: np.ndarray, 
                window: Optional[int] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        n, m = len(seq1), len(seq2)
        
        if window is None:
            window = max(n, m)
        
        window = max(window, abs(n - m))
        
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            start_j = max(1, i - window)
            end_j = min(m + 1, i + window + 1)
            for j in range(start_j, end_j):
                cost = self._calculate_distance(seq1[i - 1], seq2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )

        path = self._backtrack(dtw_matrix)
        
        return dtw_matrix[n, m], dtw_matrix, path

    def _backtrack(self, dtw_matrix: np.ndarray) -> np.ndarray:
        i, j = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
        path = []
        
        while i > 0 or j > 0:
            path.append((i - 1, j - 1))
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                min_prev = min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )
                if min_prev == dtw_matrix[i - 1, j - 1]:
                    i -= 1
                    j -= 1
                elif min_prev == dtw_matrix[i - 1, j]:
                    i -= 1
                else:
                    j -= 1
        
        path.append((0, 0))
        return np.array(path[::-1])

    def normalized_distance(self, seq1: np.ndarray, seq2: np.ndarray,
                            window: Optional[int] = None) -> float:
        distance, _, _ = self.compute(seq1, seq2, window)
        max_len = max(len(seq1), len(seq2))
        return distance / max_len


class FastDTW:
    def __init__(self, 
                 distance_func: str = 'euclidean', 
                 radius: int = 1,
                 use_normalization: bool = True,
                 weights: Optional[np.ndarray] = None,
                 norm_params: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.distance_func = distance_func
        self.radius = radius
        self.use_normalization = use_normalization
        self.weights = weights
        self.norm_params = norm_params
        self.base_dtw = DTW(
            distance_func=distance_func,
            use_normalization=use_normalization,
            weights=weights,
            norm_params=norm_params
        )

    def fit_normalization(self, sequences: list) -> None:
        self.base_dtw.fit_normalization(sequences)

    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = weights
        self.base_dtw.set_weights(weights)

    def calculate_dimension_importance(self, sequences: list) -> np.ndarray:
        return self.base_dtw.calculate_dimension_importance(sequences)

    def _reduce_by_2(self, seq: np.ndarray) -> np.ndarray:
        if len(seq) <= 2:
            return seq
        reduced = []
        for i in range(0, len(seq) - 1, 2):
            reduced.append((seq[i] + seq[i + 1]) / 2.0)
        if len(seq) % 2 == 1:
            reduced.append(seq[-1])
        return np.array(reduced)

    def _expand_window(self, path: np.ndarray, n: int, m: int) -> np.ndarray:
        window = set()
        radius = self.radius
        
        for i, j in path:
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = i * 2 + di, j * 2 + dj
                    if 0 <= ni < n and 0 <= nj < m:
                        window.add((ni, nj))
        
        return np.array(list(window))

    def compute(self, seq1: np.ndarray, seq2: np.ndarray,
                min_size: int = 32) -> Tuple[float, np.ndarray]:
        n, m = len(seq1), len(seq2)
        
        if min(n, m) <= min_size:
            dist, _, path = self.base_dtw.compute(seq1, seq2)
            return dist, path
        
        reduced_seq1 = self._reduce_by_2(seq1)
        reduced_seq2 = self._reduce_by_2(seq2)
        
        _, low_res_path = self.compute(reduced_seq1, reduced_seq2, min_size)
        
        window = self._expand_window(low_res_path, n, m)
        
        return self._dtw_with_window(seq1, seq2, window)

    def _dtw_with_window(self, seq1: np.ndarray, seq2: np.ndarray,
                         window: np.ndarray) -> Tuple[float, np.ndarray]:
        n, m = len(seq1), len(seq2)
        
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        window_set = set((i + 1, j + 1) for i, j in window)
        window_set.add((1, 1))
        
        sorted_cells = sorted(window_set, key=lambda x: (x[0], x[1]))
        
        for i, j in sorted_cells:
            cost = self.base_dtw._calculate_distance(seq1[i - 1], seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
        
        path = self.base_dtw._backtrack(dtw_matrix)
        
        return dtw_matrix[n, m], path