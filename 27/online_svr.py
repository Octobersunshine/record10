import numpy as np
import joblib
from sklearn.svm import SVR, NuSVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import os
from collections import deque
from datetime import datetime, timedelta


class OnlineSVRModel:
    def __init__(self, 
                 max_buffer_size=1000,
                 retrain_threshold=50,
                 retrain_time_interval=3600,
                 use_robust_scaler=True,
                 preserve_support_vectors=True,
                 sv_preservation_ratio=1.0,
                 use_sample_weights=True,
                 new_sample_weight=2.0):
        self.model = None
        self.scaler_X = RobustScaler() if use_robust_scaler else StandardScaler()
        self.scaler_y = RobustScaler() if use_robust_scaler else StandardScaler()
        
        self.max_buffer_size = max_buffer_size
        self.retrain_threshold = retrain_threshold
        self.retrain_time_interval = retrain_time_interval
        
        self.preserve_support_vectors = preserve_support_vectors
        self.sv_preservation_ratio = sv_preservation_ratio
        self.use_sample_weights = use_sample_weights
        self.new_sample_weight = new_sample_weight
        
        self.X_buffer = []
        self.y_buffer = []
        self.new_since_last_train = 0
        self.last_train_time = None
        
        self.X_history = None
        self.y_history = None
        self.is_trained = False
        
        self.model_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }
        
        self.training_stats = {
            'total_samples': 0,
            'total_train_calls': 0,
            'total_partial_updates': 0,
            'last_train_duration': 0
        }

    def _merge_with_support_vectors(self, X_new, y_new):
        if not self.is_trained or not self.preserve_support_vectors or self.model.support_vectors_ is None:
            return X_new, y_new
        
        X_sv = self.model.support_vectors_
        n_sv = len(X_sv)
        n_select = int(n_sv * self.sv_preservation_ratio)
        
        if n_select < n_sv:
            indices = np.random.choice(n_sv, n_select, replace=False)
            X_sv_selected = X_sv[indices]
        else:
            X_sv_selected = X_sv
        
        X_sv_original = self.scaler_X.inverse_transform(X_sv_selected)
        
        y_sv_scaled = self.model.dual_coef_.ravel()[:n_select]
        y_sv_original = self.scaler_y.inverse_transform(y_sv_scaled.reshape(-1, 1)).ravel()
        
        X_combined = np.vstack([X_sv_original, X_new])
        y_combined = np.hstack([y_sv_original, y_new])
        
        return X_combined, y_combined

    def _compute_sample_weights(self, n_total, n_new):
        if not self.use_sample_weights:
            return None
        
        weights = np.ones(n_total)
        if n_new > 0:
            weights[-n_new:] = self.new_sample_weight
        
        return weights

    def initial_train(self, X, y, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        start_time = datetime.now()
        
        X = np.array(X)
        y = np.array(y).ravel()
        
        self.model_params = {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma
        }
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.model.fit(X_scaled, y_scaled)
        
        self.X_history = X.copy()
        self.y_history = y.copy()
        self.is_trained = True
        self.last_train_time = datetime.now()
        self.new_since_last_train = 0
        
        duration = (datetime.now() - start_time).total_seconds()
        self.training_stats['total_samples'] = len(y)
        self.training_stats['total_train_calls'] = 1
        self.training_stats['last_train_duration'] = duration
        
        return {
            'status': 'initial_train_complete',
            'n_samples': len(y),
            'n_support_vectors': len(self.model.support_),
            'train_duration': duration,
            'model_params': self.model_params
        }

    def add_sample(self, X, y=None, auto_retrain=True):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if y is not None:
            y = np.array(y).ravel()
            if y.ndim == 0:
                y = y.reshape(1)
        
        results = []
        
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i] if y is not None else None
            
            self.X_buffer.append(x_i)
            if y_i is not None:
                self.y_buffer.append(y_i)
            
            self.new_since_last_train += 1
            
            if len(self.X_buffer) > self.max_buffer_size:
                self.X_buffer.pop(0)
                if self.y_buffer:
                    self.y_buffer.pop(0)
            
            result = {
                'sample_index': i,
                'added': True,
                'buffer_size': len(self.X_buffer),
                'new_since_last_train': self.new_since_last_train
            }
            
            if y_i is not None and self.is_trained:
                X_scaled = self.scaler_X.transform(x_i.reshape(1, -1))
                y_pred_scaled = self.model.predict(X_scaled)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
                result['prediction'] = y_pred
                result['actual'] = y_i
                result['error'] = abs(y_pred - y_i)
            
            results.append(result)
        
        retrain_result = None
        if auto_retrain and y is not None:
            retrain_result = self._check_and_trigger_retrain()
        
        return {
            'samples_added': len(X),
            'results': results,
            'retrain_triggered': retrain_result is not None,
            'retrain_result': retrain_result
        }

    def _check_and_trigger_retrain(self):
        if not self.is_trained:
            return None
        
        should_retrain = False
        reason = []
        
        if self.new_since_last_train >= self.retrain_threshold:
            should_retrain = True
            reason.append(f'new_samples={self.new_since_last_train} >= threshold={self.retrain_threshold}')
        
        if self.last_train_time is not None:
            time_since_last = (datetime.now() - self.last_train_time).total_seconds()
            if time_since_last >= self.retrain_time_interval:
                should_retrain = True
                reason.append(f'time_since_last={time_since_last:.1f}s >= interval={self.retrain_time_interval}s')
        
        if should_retrain and len(self.y_buffer) >= 10:
            return self.retrain(reason=', '.join(reason))
        
        return None

    def retrain(self, reason='manual', warm_start=True):
        start_time = datetime.now()
        
        if len(self.y_buffer) == 0:
            return {'status': 'no_new_samples', 'message': '缓存中无新样本'}
        
        X_new = np.array(self.X_buffer[-len(self.y_buffer):])
        y_new = np.array(self.y_buffer)
        
        n_new = len(y_new)
        
        if warm_start and self.is_trained and self.preserve_support_vectors:
            X_combined, y_combined = self._merge_with_support_vectors(X_new, y_new)
            sample_weights = self._compute_sample_weights(len(y_combined), n_new)
        else:
            X_combined = X_new
            y_combined = y_new
            sample_weights = None
        
        X_scaled = self.scaler_X.fit_transform(X_combined)
        y_scaled = self.scaler_y.fit_transform(y_combined.reshape(-1, 1)).ravel()
        
        self.model = SVR(**self.model_params)
        
        if sample_weights is not None:
            self.model.fit(X_scaled, y_scaled, sample_weight=sample_weights)
        else:
            self.model.fit(X_scaled, y_scaled)
        
        if self.X_history is not None:
            self.X_history = np.vstack([self.X_history, X_new])
            self.y_history = np.hstack([self.y_history, y_new])
        else:
            self.X_history = X_new
            self.y_history = y_new
        
        self.last_train_time = datetime.now()
        self.new_since_last_train = 0
        
        duration = (datetime.now() - start_time).total_seconds()
        self.training_stats['total_train_calls'] += 1
        self.training_stats['total_partial_updates'] += 1
        self.training_stats['last_train_duration'] = duration
        
        return {
            'status': 'retrain_complete',
            'reason': reason,
            'n_new_samples': n_new,
            'n_total_samples': len(self.y_history) if self.y_history is not None else n_new,
            'n_support_vectors': len(self.model.support_),
            'train_duration': duration,
            'used_support_vectors': warm_start and self.preserve_support_vectors,
            'sample_weights_applied': sample_weights is not None
        }

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("模型尚未初始化训练，请先调用 initial_train()")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return y_pred.tolist()

    def predict_with_confidence(self, X, n_neighbors=5):
        predictions = self.predict(X)
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        confidences = []
        if self.X_history is not None and len(self.X_history) > 0:
            from sklearn.metrics.pairwise import euclidean_distances
            X_scaled = self.scaler_X.transform(X)
            X_history_scaled = self.scaler_X.transform(self.X_history)
            
            for x in X_scaled:
                distances = euclidean_distances(x.reshape(1, -1), X_history_scaled).ravel()
                nearest_dist = np.sort(distances)[:n_neighbors].mean()
                confidence = 1.0 / (1.0 + nearest_dist)
                confidences.append(confidence)
        else:
            confidences = [0.5] * len(predictions)
        
        return list(zip(predictions, confidences))

    def save_model(self, filepath):
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'model_params': self.model_params,
            'X_history': self.X_history,
            'y_history': self.y_history,
            'X_buffer': np.array(self.X_buffer),
            'y_buffer': np.array(self.y_buffer),
            'new_since_last_train': self.new_since_last_train,
            'last_train_time': self.last_train_time,
            'training_stats': self.training_stats,
            'config': {
                'max_buffer_size': self.max_buffer_size,
                'retrain_threshold': self.retrain_threshold,
                'retrain_time_interval': self.retrain_time_interval,
                'use_robust_scaler': isinstance(self.scaler_X, RobustScaler),
                'preserve_support_vectors': self.preserve_support_vectors,
                'sv_preservation_ratio': self.sv_preservation_ratio,
                'use_sample_weights': self.use_sample_weights,
                'new_sample_weight': self.new_sample_weight
            }
        }
        joblib.dump(model_data, filepath)
        return f"在线SVR模型已保存至: {filepath}"

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.model_params = model_data['model_params']
        self.X_history = model_data['X_history']
        self.y_history = model_data['y_history']
        self.X_buffer = model_data['X_buffer'].tolist() if model_data['X_buffer'] is not None else []
        self.y_buffer = model_data['y_buffer'].tolist() if model_data['y_buffer'] is not None else []
        self.new_since_last_train = model_data['new_since_last_train']
        self.last_train_time = model_data['last_train_time']
        self.training_stats = model_data['training_stats']
        
        config = model_data.get('config', {})
        self.max_buffer_size = config.get('max_buffer_size', 1000)
        self.retrain_threshold = config.get('retrain_threshold', 50)
        self.retrain_time_interval = config.get('retrain_time_interval', 3600)
        self.preserve_support_vectors = config.get('preserve_support_vectors', True)
        self.sv_preservation_ratio = config.get('sv_preservation_ratio', 1.0)
        self.use_sample_weights = config.get('use_sample_weights', True)
        self.new_sample_weight = config.get('new_sample_weight', 2.0)
        
        self.is_trained = True
        return f"在线SVR模型已从 {filepath} 加载"

    def get_model_info(self):
        info = {
            'status': 'trained' if self.is_trained else 'not_trained',
            'model_params': self.model_params,
            'buffer_size': len(self.X_buffer),
            'new_since_last_train': self.new_since_last_train,
            'last_train_time': self.last_train_time.isoformat() if self.last_train_time else None,
            'history_size': len(self.X_history) if self.X_history is not None else 0,
            'training_stats': self.training_stats
        }
        
        if self.is_trained:
            info['n_support_vectors'] = len(self.model.support_)
        
        return info

    def get_buffer_info(self):
        if len(self.y_buffer) == 0:
            return {'message': '缓存为空'}
        
        y_buffer = np.array(self.y_buffer)
        return {
            'buffer_size': len(self.X_buffer),
            'labeled_samples': len(self.y_buffer),
            'y_stats': {
                'mean': float(np.mean(y_buffer)),
                'std': float(np.std(y_buffer)),
                'min': float(np.min(y_buffer)),
                'max': float(np.max(y_buffer))
            },
            'new_since_last_train': self.new_since_last_train,
            'retrain_readiness': self.new_since_last_train / self.retrain_threshold if self.retrain_threshold > 0 else 0
        }

    def clear_buffer(self):
        n_cleared = len(self.X_buffer)
        self.X_buffer = []
        self.y_buffer = []
        return {'cleared_samples': n_cleared, 'message': '缓存已清空'}

    def update_params(self, **kwargs):
        if 'C' in kwargs:
            self.model_params['C'] = kwargs['C']
        if 'epsilon' in kwargs:
            self.model_params['epsilon'] = kwargs['epsilon']
        if 'gamma' in kwargs:
            self.model_params['gamma'] = kwargs['gamma']
        if 'retrain_threshold' in kwargs:
            self.retrain_threshold = kwargs['retrain_threshold']
        if 'new_sample_weight' in kwargs:
            self.new_sample_weight = kwargs['new_sample_weight']
        
        return {'status': 'params_updated', 'model_params': self.model_params}
