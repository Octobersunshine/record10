import numpy as np
import joblib
from sklearn.svm import SVR, NuSVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from scipy import stats


class OutlierDetector:
    @staticmethod
    def z_score(y, threshold=3.0):
        z_scores = np.abs(stats.zscore(y))
        return z_scores < threshold

    @staticmethod
    def iqr(y, factor=1.5):
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return (y >= lower_bound) & (y <= upper_bound)

    @staticmethod
    def modified_z_score(y, threshold=3.5):
        median_y = np.median(y)
        mad = np.median(np.abs(y - median_y))
        if mad == 0:
            mad = np.std(y)
        modified_z = 0.6745 * np.abs(y - median_y) / mad
        return modified_z < threshold

    @staticmethod
    def cook_distance(X, y, threshold=None):
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        try:
            hat_matrix = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
            residuals = y - X_with_intercept @ (np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y)
            mse = np.sum(residuals ** 2) / (len(y) - X_with_intercept.shape[1])
            cook_d = (residuals ** 2 / (X_with_intercept.shape[1] * mse)) * (np.diag(hat_matrix) / (1 - np.diag(hat_matrix)) ** 2)
            
            if threshold is None:
                threshold = 4 / len(y)
            return cook_d < threshold
        except:
            return np.ones(len(y), dtype=bool)


class RobustSVRModel:
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_trained = False
        self.outlier_mask = None
        self.X_clean = None
        self.y_clean = None

    def detect_outliers(self, X, y, method='combined', **kwargs):
        y = np.array(y).ravel()
        masks = []

        if method in ['z_score', 'combined']:
            threshold = kwargs.get('z_threshold', 3.0)
            masks.append(OutlierDetector.z_score(y, threshold))

        if method in ['iqr', 'combined']:
            factor = kwargs.get('iqr_factor', 1.5)
            masks.append(OutlierDetector.iqr(y, factor))

        if method in ['modified_z', 'combined']:
            threshold = kwargs.get('mod_z_threshold', 3.5)
            masks.append(OutlierDetector.modified_z_score(y, threshold))

        if method in ['cook', 'combined']:
            threshold = kwargs.get('cook_threshold', None)
            masks.append(OutlierDetector.cook_distance(X, y, threshold))

        if masks:
            combined_mask = np.all(masks, axis=0)
        else:
            combined_mask = np.ones(len(y), dtype=bool)

        return combined_mask

    def train(self, X, y, 
              kernel='rbf', 
              C=1.0, 
              epsilon=0.1, 
              gamma='scale',
              use_nusvr=False,
              nu=0.5,
              test_size=0.2, 
              random_state=42,
              remove_outliers=True,
              outlier_method='combined',
              use_robust_scaler=True,
              max_sv_ratio=None,
              auto_tune=False,
              **outlier_kwargs):
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        original_count = len(y)

        if remove_outliers:
            self.outlier_mask = self.detect_outliers(X, y, method=outlier_method, **outlier_kwargs)
            X_clean = X[self.outlier_mask]
            y_clean = y[self.outlier_mask]
            removed_count = original_count - np.sum(self.outlier_mask)
            print(f"检测到 {removed_count} 个异常值，已移除")
        else:
            X_clean = X
            y_clean = y
            self.outlier_mask = np.ones(len(y), dtype=bool)

        self.X_clean = X_clean
        self.y_clean = y_clean

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=random_state
        )

        if use_robust_scaler:
            self.scaler_X = RobustScaler()
            self.scaler_y = RobustScaler()
        else:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train).ravel()

        if auto_tune:
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'epsilon': [0.01, 0.1, 0.5, 1.0],
                'gamma': ['scale', 'auto', 0.1, 1.0]
            }
            
            if use_nusvr:
                param_grid['nu'] = [0.1, 0.25, 0.5, 0.75]
                base_model = NuSVR(kernel=kernel)
            else:
                base_model = SVR(kernel=kernel)

            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train_scaled)
            self.model = grid_search.best_estimator_
            print(f"自动调优最佳参数: {grid_search.best_params_}")
        else:
            if use_nusvr:
                self.model = NuSVR(kernel=kernel, C=C, nu=nu, gamma=gamma)
            else:
                self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
            
            self.model.fit(X_train_scaled, y_train_scaled)

        if max_sv_ratio is not None and not use_nusvr:
            n_samples = len(X_train_scaled)
            max_sv = int(n_samples * max_sv_ratio)
            current_sv = len(self.model.support_)
            
            if current_sv > max_sv:
                print(f"警告: 支持向量数量 ({current_sv}) 超过限制 ({max_sv})，尝试调整参数...")
                adjusted_epsilon = epsilon * 2
                while current_sv > max_sv and adjusted_epsilon < 10:
                    self.model = SVR(kernel=kernel, C=C, epsilon=adjusted_epsilon, gamma=gamma)
                    self.model.fit(X_train_scaled, y_train_scaled)
                    current_sv = len(self.model.support_)
                    adjusted_epsilon *= 1.5

        X_test_scaled = self.scaler_X.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test = y_test.ravel()

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.is_trained = True

        return {
            'original_samples': original_count,
            'clean_samples': len(y_clean),
            'outliers_removed': original_count - len(y_clean),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'support_vectors': len(self.model.support_),
            'sv_ratio': len(self.model.support_) / len(y_clean),
            'model_params': {
                'kernel': kernel,
                'C': self.model.C,
                'epsilon': getattr(self.model, 'epsilon', None),
                'nu': getattr(self.model, 'nu', None),
                'gamma': self.model.gamma,
                'use_nusvr': use_nusvr,
                'remove_outliers': remove_outliers,
                'use_robust_scaler': use_robust_scaler
            }
        }

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")

        X = np.array(X)
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        return y_pred.tolist()

    def save_model(self, filepath):
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")

        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'outlier_mask': self.outlier_mask
        }
        joblib.dump(model_data, filepath)
        return f"模型已保存至: {filepath}"

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.outlier_mask = model_data.get('outlier_mask')
        self.is_trained = True
        return f"模型已从 {filepath} 加载"

    def get_model_info(self):
        if not self.is_trained:
            return {"status": "模型尚未训练"}

        return {
            "status": "已训练",
            "model_type": type(self.model).__name__,
            "kernel": self.model.kernel,
            "C": self.model.C,
            "epsilon": getattr(self.model, 'epsilon', None),
            "nu": getattr(self.model, 'nu', None),
            "support_vectors": len(self.model.support_),
            "n_features": self.model.n_features_in_,
            "scaler_type": type(self.scaler_X).__name__
        }

    def get_outlier_info(self):
        if self.outlier_mask is None:
            return {"message": "未进行异常值检测"}
        
        outliers = np.where(~self.outlier_mask)[0]
        return {
            "total_samples": len(self.outlier_mask),
            "outlier_count": len(outliers),
            "outlier_indices": outliers.tolist(),
            "outlier_percentage": len(outliers) / len(self.outlier_mask) * 100
        }
