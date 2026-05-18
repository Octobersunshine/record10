import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

class SVRModel:
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False

    def train(self, X, y, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', test_size=0.2, random_state=42):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train).ravel()

        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.model.fit(X_train_scaled, y_train_scaled)

        X_test_scaled = self.scaler_X.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test = y_test.ravel()

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        self.is_trained = True

        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'support_vectors': len(self.model.support_),
            'model_params': {
                'kernel': kernel,
                'C': C,
                'epsilon': epsilon,
                'gamma': gamma
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
            'scaler_y': self.scaler_y
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
        self.is_trained = True
        return f"模型已从 {filepath} 加载"

    def get_model_info(self):
        if not self.is_trained:
            return {"status": "模型尚未训练"}

        return {
            "status": "已训练",
            "kernel": self.model.kernel,
            "C": self.model.C,
            "epsilon": self.model.epsilon,
            "support_vectors": len(self.model.support_),
            "n_features": self.model.n_features_in_
        }
