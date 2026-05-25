import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    feature_type: str
    l1_reg: float
    l2_reg: float
    aicc: float
    n_params: int
    log_likelihood: float


class MaxEntModel:
    def __init__(self, 
                 l1_reg: float = 0.0, 
                 l2_reg: float = 1.0, 
                 feature_type: str = 'quadratic'):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.feature_type = feature_type
        self.weights = None
        self.feature_means = None
        self.feature_stds = None
        self.n_samples_presence = None
        self.n_samples_background = None
        self._presence_features = None
        self._background_features = None
        
    def set_params(self, **params):
        if 'l1_reg' in params:
            self.l1_reg = params['l1_reg']
        if 'l2_reg' in params:
            self.l2_reg = params['l2_reg']
        if 'feature_type' in params:
            self.feature_type = params['feature_type']
        return self

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_means is None or self.feature_stds is None:
            self.feature_means = np.mean(features, axis=0)
            self.feature_stds = np.std(features, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0
        return (features - self.feature_means) / self.feature_stds

    def _compute_raw_features(self, features: np.ndarray) -> np.ndarray:
        n_samples = features.shape[0]
        feature_list = [np.ones((n_samples, 1))]
        
        if self.feature_type in ['linear', 'quadratic', 'product']:
            feature_list.append(features)
        
        if self.feature_type in ['quadratic', 'product']:
            quad_features = features ** 2
            feature_list.append(quad_features)
        
        if self.feature_type == 'product':
            product_features = []
            for i in range(features.shape[1]):
                for j in range(i + 1, features.shape[1]):
                    product_features.append(features[:, i] * features[:, j])
            if product_features:
                product_features = np.array(product_features).T
                feature_list.append(product_features)
        
        return np.hstack(feature_list)

    def _compute_probabilities(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        raw_scores = np.dot(features, weights)
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        return exp_scores / np.sum(exp_scores)

    def _objective_function(self, weights: np.ndarray, 
                           presence_features: np.ndarray,
                           background_features: np.ndarray) -> float:
        presence_probs = self._compute_probabilities(presence_features, weights)
        background_probs = self._compute_probabilities(background_features, weights)
        
        log_presence = np.sum(np.log(presence_probs + 1e-10))
        
        entropy = -np.sum(background_probs * np.log(background_probs + 1e-10))
        
        l2_term = 0.5 * self.l2_reg * np.sum(weights ** 2)
        l1_term = self.l1_reg * np.sum(np.abs(weights))
        
        return -(log_presence + entropy - l2_term - l1_term)

    def _gradient(self, weights: np.ndarray,
                  presence_features: np.ndarray,
                  background_features: np.ndarray) -> np.ndarray:
        presence_probs = self._compute_probabilities(presence_features, weights)
        background_probs = self._compute_probabilities(background_features, weights)
        
        presence_expectation = np.dot(presence_probs, presence_features)
        background_expectation = np.dot(background_probs, background_features)
        
        feature_expectation_diff = presence_expectation - background_expectation
        
        l2_grad = self.l2_reg * weights
        l1_grad = self.l1_reg * np.sign(weights)
        
        return -(feature_expectation_diff - l2_grad - l1_grad)

    def fit(self, presence_data: np.ndarray, background_data: np.ndarray,
            max_iter: int = 1000, tolerance: float = 1e-8) -> 'MaxEntModel':
        self.n_samples_presence = presence_data.shape[0]
        self.n_samples_background = background_data.shape[0]
        
        presence_normalized = self._normalize_features(presence_data)
        background_normalized = self._normalize_features(background_data)
        
        self._presence_features = self._compute_raw_features(presence_normalized)
        self._background_features = self._compute_raw_features(background_normalized)
        
        n_features = self._presence_features.shape[1]
        initial_weights = np.zeros(n_features)
        
        result = minimize(
            fun=self._objective_function,
            x0=initial_weights,
            jac=self._gradient,
            args=(self._presence_features, self._background_features),
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': tolerance,
                'disp': False
            }
        )
        
        self.weights = result.x
        return self

    def compute_log_likelihood(self) -> float:
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        
        presence_probs = self._compute_probabilities(self._presence_features, self.weights)
        return np.sum(np.log(presence_probs + 1e-10))

    def compute_aicc(self) -> float:
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        
        n = self.n_samples_presence
        k = np.sum(np.abs(self.weights) > 1e-6)
        
        log_likelihood = self.compute_log_likelihood()
        
        aic = 2 * k - 2 * log_likelihood
        
        if n - k - 1 > 0:
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        else:
            aicc = float('inf')
        
        return aicc

    def get_effective_params(self) -> int:
        if self.weights is None:
            return 0
        return np.sum(np.abs(self.weights) > 1e-6)

    def predict(self, env_data: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        normalized = self._normalize_features(env_data)
        features = self._compute_raw_features(normalized)
        
        raw_scores = np.dot(features, self.weights)
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        suitability = exp_scores / np.max(exp_scores)
        
        return suitability


class MaxEntModelSelector:
    def __init__(self):
        self.best_model = None
        self.best_config = None
        self.all_results = []
    
    def grid_search(self, 
                    presence_data: np.ndarray,
                    background_data: np.ndarray,
                    feature_types: List[str] = ['linear', 'quadratic'],
                    l1_reg_values: List[float] = [0.0, 0.1, 0.5, 1.0],
                    l2_reg_values: List[float] = [0.1, 0.5, 1.0, 2.0]) -> Dict:
        
        best_aicc = float('inf')
        best_model = None
        best_params = None
        
        for feature_type in feature_types:
            for l1_reg in l1_reg_values:
                for l2_reg in l2_reg_values:
                    print(f"  Testing: features={feature_type}, L1={l1_reg}, L2={l2_reg}...", end=' ')
                    
                    model = MaxEntModel(
                        l1_reg=l1_reg, 
                        l2_reg=l2_reg, 
                        feature_type=feature_type
                    )
                    model.fit(presence_data, background_data)
                    
                    aicc = model.compute_aicc()
                    log_likelihood = model.compute_log_likelihood()
                    n_params = model.get_effective_params()
                    
                    config = ModelConfig(
                        feature_type=feature_type,
                        l1_reg=l1_reg,
                        l2_reg=l2_reg,
                        aicc=aicc,
                        n_params=n_params,
                        log_likelihood=log_likelihood
                    )
                    
                    self.all_results.append(config)
                    
                    print(f"AICc={aicc:.2f}, params={n_params}")
                    
                    if aicc < best_aicc:
                        best_aicc = aicc
                        best_model = model
                        best_params = {
                            'feature_type': feature_type,
                            'l1_reg': l1_reg,
                            'l2_reg': l2_reg,
                            'aicc': aicc,
                            'log_likelihood': log_likelihood,
                            'n_params': n_params
                        }
        
        self.best_model = best_model
        self.best_config = best_params
        
        return best_params
    
    def get_results_summary(self) -> List[Dict]:
        summary = []
        for result in self.all_results:
            summary.append({
                'feature_type': result.feature_type,
                'l1_reg': result.l1_reg,
                'l2_reg': result.l2_reg,
                'aicc': result.aicc,
                'n_params': result.n_params,
                'log_likelihood': result.log_likelihood
            })
        
        summary.sort(key=lambda x: x['aicc'])
        return summary
