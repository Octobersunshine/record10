import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple, Optional, Union, List


class PLSRegressor:
    def __init__(self, n_components: int = 2, scale: bool = True, 
                 fix_sign: bool = True, sign_criterion: str = "first_element"):
        self.n_components = n_components
        self.scale = scale
        self.fix_sign = fix_sign
        self.sign_criterion = sign_criterion
        self.model = PLSRegression(n_components=n_components, scale=False)
        self.scaler_X = StandardScaler() if scale else None
        self.scaler_Y = StandardScaler() if scale else None
        self._is_trained = False
        self._signs = None

    def _resolve_sign_ambiguity(self, X: np.ndarray, Y: np.ndarray) -> None:
        x_weights = self.model.x_weights_
        n_components = x_weights.shape[1]
        signs = np.ones(n_components)
        
        for i in range(n_components):
            if self.sign_criterion == "first_element":
                first_nonzero_idx = np.where(x_weights[:, i] != 0)[0]
                if len(first_nonzero_idx) > 0 and x_weights[first_nonzero_idx[0], i] < 0:
                    signs[i] = -1
            elif self.sign_criterion == "sum":
                if np.sum(x_weights[:, i]) < 0:
                    signs[i] = -1
            elif self.sign_criterion == "max_abs":
                max_abs_idx = np.argmax(np.abs(x_weights[:, i]))
                if x_weights[max_abs_idx, i] < 0:
                    signs[i] = -1
            elif self.sign_criterion == "loading_sum":
                if np.sum(self.model.x_loadings_[:, i]) < 0:
                    signs[i] = -1
        
        self._signs = signs
        
        self.model.x_weights_ = x_weights * signs
        self.model.y_weights_ = self.model.y_weights_ * signs
        self.model.x_loadings_ = self.model.x_loadings_ * signs
        self.model.y_loadings_ = self.model.y_loadings_ * signs
        self.model.x_scores_ = self.model.x_scores_ * signs
        self.model.y_scores_ = self.model.y_scores_ * signs
        
        self.model.coef_ = self.model.x_weights_ @ np.linalg.pinv(self.model.x_loadings_.T @ self.model.x_weights_) @ self.model.x_loadings_.T @ self.model.y_loadings_.T

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'PLSRegressor':
        if self.scale:
            X_scaled = self.scaler_X.fit_transform(X)
            Y_scaled = self.scaler_Y.fit_transform(Y)
            self.model.fit(X_scaled, Y_scaled)
        else:
            self.model.fit(X, Y)
        
        if self.fix_sign:
            self._resolve_sign_ambiguity(X, Y)
        
        self._is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        if self.scale:
            X_scaled = self.scaler_X.transform(X)
            Y_pred_scaled = self.model.predict(X_scaled)
            Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        else:
            Y_pred = self.model.predict(X)
        return Y_pred

    def get_coefficients(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        if self.scale:
            coef = self.model.coef_ * (self.scaler_Y.scale_ / self.scaler_X.scale_)
        else:
            coef = self.model.coef_
        return coef

    def get_intercept(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        if self.scale:
            intercept = (self.scaler_Y.mean_ - 
                         np.dot(self.scaler_X.mean_, self.get_coefficients().T))
        else:
            intercept = self.model.y_mean_ - np.dot(self.model.x_mean_, self.model.coef_.T)
        return intercept

    def get_x_weights(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.x_weights_

    def get_y_weights(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.y_weights_

    def get_x_loadings(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.x_loadings_

    def get_y_loadings(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.y_loadings_

    def get_x_scores(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.x_scores_

    def get_y_scores(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.y_scores_

    def get_signs(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self._signs if self._signs is not None else np.ones(self.n_components)

    def get_feature_importance(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        x_weights = self.get_x_weights()
        x_scores = self.get_x_scores()
        importance = np.sum((x_scores.T @ x_scores) * (x_weights.T @ x_weights), axis=0)
        return importance / np.sum(importance)

    def get_vip_scores(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        _, n_features = self.model.x_weights_.shape
        x_weights = self.model.x_weights_
        x_scores = self.model.x_scores_
        
        ssy = np.sum(x_scores ** 2, axis=0)
        importance = np.zeros(n_features)
        
        for j in range(n_features):
            sum_vip = 0.0
            for a in range(self.n_components):
                sum_vip += (ssy[a] * (x_weights[j, a] ** 2) / 
                           np.sum(x_weights[:, a] ** 2))
            importance[j] = np.sqrt(self.n_components * sum_vip / np.sum(ssy))
        
        return importance


def train_pls_model(X: np.ndarray, Y: np.ndarray, n_components: int = 2, 
                    scale: bool = True, fix_sign: bool = True,
                    sign_criterion: str = "first_element") -> Tuple[PLSRegressor, np.ndarray, np.ndarray]:
    model = PLSRegressor(n_components=n_components, scale=scale, 
                        fix_sign=fix_sign, sign_criterion=sign_criterion)
    model.fit(X, Y)
    coefficients = model.get_coefficients()
    intercept = model.get_intercept()
    return model, coefficients, intercept


def predict_pls_model(model: PLSRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


class SparsePLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components: int = 2, scale: bool = True,
                 eta: float = 0.5, max_iter: int = 500, tol: float = 1e-6,
                 fix_sign: bool = True, sign_criterion: str = "first_element"):
        self.n_components = n_components
        self.scale = scale
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.fix_sign = fix_sign
        self.sign_criterion = sign_criterion
        self.scaler_X = StandardScaler() if scale else None
        self.scaler_Y = StandardScaler() if scale else None
        self._is_trained = False
        self._signs = None
        
        self.x_weights_ = None
        self.y_weights_ = None
        self.x_loadings_ = None
        self.y_loadings_ = None
        self.x_scores_ = None
        self.y_scores_ = None
        self.coef_ = None
        self.x_mean_ = None
        self.y_mean_ = None

    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _resolve_sign_ambiguity(self) -> None:
        n_components = self.x_weights_.shape[1]
        signs = np.ones(n_components)
        
        for i in range(n_components):
            if self.sign_criterion == "first_element":
                first_nonzero_idx = np.where(self.x_weights_[:, i] != 0)[0]
                if len(first_nonzero_idx) > 0 and self.x_weights_[first_nonzero_idx[0], i] < 0:
                    signs[i] = -1
            elif self.sign_criterion == "sum":
                if np.sum(self.x_weights_[:, i]) < 0:
                    signs[i] = -1
            elif self.sign_criterion == "max_abs":
                max_abs_idx = np.argmax(np.abs(self.x_weights_[:, i]))
                if self.x_weights_[max_abs_idx, i] < 0:
                    signs[i] = -1
            elif self.sign_criterion == "loading_sum":
                if np.sum(self.x_loadings_[:, i]) < 0:
                    signs[i] = -1
        
        self._signs = signs
        
        self.x_weights_ = self.x_weights_ * signs
        self.y_weights_ = self.y_weights_ * signs
        self.x_loadings_ = self.x_loadings_ * signs
        self.y_loadings_ = self.y_loadings_ * signs
        self.x_scores_ = self.x_scores_ * signs
        self.y_scores_ = self.y_scores_ * signs

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'SparsePLS':
        n_samples, n_features = X.shape
        n_targets = Y.shape[1] if Y.ndim > 1 else 1
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        if self.scale:
            X_scaled = self.scaler_X.fit_transform(X)
            Y_scaled = self.scaler_Y.fit_transform(Y)
        else:
            X_scaled = X.copy()
            Y_scaled = Y.copy()
        
        self.x_mean_ = X_scaled.mean(axis=0)
        self.y_mean_ = Y_scaled.mean(axis=0)
        
        X_centered = X_scaled - self.x_mean_
        Y_centered = Y_scaled - self.y_mean_
        
        self.x_weights_ = np.zeros((n_features, self.n_components))
        self.y_weights_ = np.zeros((n_targets, self.n_components))
        self.x_loadings_ = np.zeros((n_features, self.n_components))
        self.y_loadings_ = np.zeros((n_targets, self.n_components))
        self.x_scores_ = np.zeros((n_samples, self.n_components))
        self.y_scores_ = np.zeros((n_samples, self.n_components))
        
        X_residual = X_centered.copy()
        Y_residual = Y_centered.copy()
        
        for k in range(self.n_components):
            w = np.random.randn(n_features)
            w = w / np.linalg.norm(w)
            c = np.random.randn(n_targets)
            c = c / np.linalg.norm(c)
            
            for iter_idx in range(self.max_iter):
                t = X_residual @ w
                t = t / np.linalg.norm(t)
                
                u = Y_residual @ c
                u = u / np.linalg.norm(u)
                
                w_new = X_residual.T @ u
                w_new = self._soft_threshold(w_new, self.eta * np.max(np.abs(w_new)))
                
                if np.linalg.norm(w_new) > 0:
                    w_new = w_new / np.linalg.norm(w_new)
                else:
                    w_new = np.zeros_like(w_new)
                
                c_new = Y_residual.T @ t
                c_new = c_new / np.linalg.norm(c_new)
                
                if (np.linalg.norm(w_new - w) < self.tol and 
                    np.linalg.norm(c_new - c) < self.tol):
                    break
                
                w, c = w_new, c_new
            
            t = X_residual @ w
            if np.linalg.norm(t) > 0:
                t = t / np.linalg.norm(t)
            
            p = X_residual.T @ t
            q = Y_residual.T @ t
            
            self.x_weights_[:, k] = w
            self.y_weights_[:, k] = c
            self.x_loadings_[:, k] = p
            self.y_loadings_[:, k] = q
            self.x_scores_[:, k] = t
            self.y_scores_[:, k] = Y_residual @ c
            
            X_residual = X_residual - np.outer(t, p)
            Y_residual = Y_residual - np.outer(t, q)
        
        self.coef_ = self.x_weights_ @ np.linalg.pinv(self.x_loadings_.T @ self.x_weights_) @ self.x_loadings_.T @ self.y_loadings_.T
        
        if self.fix_sign:
            self._resolve_sign_ambiguity()
        
        self._is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.scale:
            X_scaled = self.scaler_X.transform(X)
        else:
            X_scaled = X.copy()
        
        X_centered = X_scaled - self.x_mean_
        Y_pred_centered = X_centered @ self.coef_
        Y_pred = Y_pred_centered + self.y_mean_
        
        if self.scale:
            Y_pred = self.scaler_Y.inverse_transform(Y_pred)
        
        return Y_pred

    def get_coefficients(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        if self.scale:
            coef = self.coef_ * (self.scaler_Y.scale_ / self.scaler_X.scale_)
        else:
            coef = self.coef_
        return coef

    def get_intercept(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        if self.scale:
            intercept = (self.scaler_Y.mean_ - 
                         np.dot(self.scaler_X.mean_, self.get_coefficients().T))
        else:
            intercept = self.y_mean_ - np.dot(self.x_mean_, self.coef_.T)
        return intercept

    def get_x_weights(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.x_weights_

    def get_y_weights(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.y_weights_

    def get_x_loadings(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.x_loadings_

    def get_y_loadings(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.y_loadings_

    def get_x_scores(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.x_scores_

    def get_y_scores(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self.y_scores_

    def get_signs(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        return self._signs if self._signs is not None else np.ones(self.n_components)

    def get_selected_features(self, tol: float = 1e-10) -> List[int]:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        coef = self.get_coefficients()
        if coef.ndim == 2:
            feature_mask = np.any(np.abs(coef) > tol, axis=1)
        else:
            feature_mask = np.abs(coef) > tol
        return np.where(feature_mask)[0].tolist()

    def get_sparsity_ratio(self, tol: float = 1e-10) -> float:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        coef = self.get_coefficients()
        n_zero = np.sum(np.abs(coef) <= tol)
        return n_zero / coef.size

    def get_vip_scores(self) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        n_features = self.x_weights_.shape[0]
        
        ssy = np.sum(self.x_scores_ ** 2, axis=0)
        importance = np.zeros(n_features)
        
        for j in range(n_features):
            sum_vip = 0.0
            for a in range(self.n_components):
                weight_norm = np.sum(self.x_weights_[:, a] ** 2)
                if weight_norm > 0:
                    sum_vip += (ssy[a] * (self.x_weights_[j, a] ** 2) / weight_norm)
            importance[j] = np.sqrt(self.n_components * sum_vip / np.sum(ssy))
        
        return importance


def train_spls_model(X: np.ndarray, Y: np.ndarray, n_components: int = 2,
                     scale: bool = True, eta: float = 0.5,
                     fix_sign: bool = True, sign_criterion: str = "first_element"
                     ) -> Tuple[SparsePLS, np.ndarray, np.ndarray]:
    model = SparsePLS(n_components=n_components, scale=scale, eta=eta,
                      fix_sign=fix_sign, sign_criterion=sign_criterion)
    model.fit(X, Y)
    coefficients = model.get_coefficients()
    intercept = model.get_intercept()
    return model, coefficients, intercept


def predict_spls_model(model: SparsePLS, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def test_sign_consistency():
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_targets = 2
    
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features, n_targets)
    Y = X @ true_coef + 0.1 * np.random.randn(n_samples, n_targets)
    
    print("=" * 60)
    print("测试PLS符号一致性修复")
    print("=" * 60)
    
    print("\n1. 测试带符号修复的模型:")
    weights_list = []
    coef_list = []
    pred_list = []
    
    for i in range(3):
        model_fixed, coef_fixed, intercept_fixed = train_pls_model(
            X, Y, n_components=3, scale=True, fix_sign=True
        )
        weights_fixed = model_fixed.get_x_weights()
        weights_list.append(weights_fixed)
        coef_list.append(coef_fixed)
        
        X_new = np.random.randn(5, n_features)
        Y_pred_fixed = predict_pls_model(model_fixed, X_new)
        pred_list.append(Y_pred_fixed)
        
        print(f"  运行 {i+1}: 第一个权重向量首元素符号 = {np.sign(weights_fixed[0, 0]):+.0f}")
    
    sign_consistent = all(np.allclose(weights_list[0], w) for w in weights_list[1:])
    coef_consistent = all(np.allclose(coef_list[0], c) for c in coef_list[1:])
    print(f"\n  权重向量符号一致: {sign_consistent}")
    print(f"  系数一致: {coef_consistent}")
    
    print("\n2. 对比带修复 vs 不带修复:")
    model_no_fix, coef_no_fix, intercept_no_fix = train_pls_model(
        X, Y, n_components=3, scale=True, fix_sign=False
    )
    weights_no_fix = model_no_fix.get_x_weights()
    
    model_with_fix, coef_with_fix, intercept_with_fix = train_pls_model(
        X, Y, n_components=3, scale=True, fix_sign=True
    )
    weights_with_fix = model_with_fix.get_x_weights()
    
    print(f"  不带修复 - 权重符号: {np.sign(weights_no_fix[0, :])}")
    print(f"  带修复   - 权重符号: {np.sign(weights_with_fix[0, :])}")
    
    Y_pred_no_fix = predict_pls_model(model_no_fix, X_new)
    Y_pred_with_fix = predict_pls_model(model_with_fix, X_new)
    pred_diff = np.max(np.abs(Y_pred_no_fix - Y_pred_with_fix))
    print(f"  预测结果最大差异: {pred_diff:.2e} (应该接近0)")
    
    print("\n3. 不同的符号判定准则测试:")
    criteria = ["first_element", "sum", "max_abs", "loading_sum"]
    for criterion in criteria:
        model, _, _ = train_pls_model(
            X, Y, n_components=3, scale=True, fix_sign=True, sign_criterion=criterion
        )
        w = model.get_x_weights()
        print(f"  {criterion:15s} - 权重符号: {np.sign(w[0, :])}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return sign_consistent and coef_consistent


def test_sparse_pls():
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    n_targets = 2
    
    n_relevant = 5
    true_coef = np.zeros((n_features, n_targets))
    true_coef[:n_relevant, :] = np.random.randn(n_relevant, n_targets)
    
    X = np.random.randn(n_samples, n_features)
    Y = X @ true_coef + 0.1 * np.random.randn(n_samples, n_targets)
    
    print("=" * 60)
    print("测试稀疏PLS (SPLS)")
    print("=" * 60)
    print(f"\n数据: {n_samples}样本, {n_features}特征, {n_targets}响应变量")
    print(f"真实相关特征: 前{n_relevant}个特征")
    
    print("\n1. 不同稀疏度对比:")
    etas = [0.0, 0.3, 0.5, 0.7]
    for eta in etas:
        model, coef, intercept = train_spls_model(
            X, Y, n_components=3, eta=eta, scale=True, fix_sign=True
        )
        sparsity = model.get_sparsity_ratio()
        selected = model.get_selected_features()
        n_selected = len(selected)
        
        print(f"  eta={eta:.1f}: 稀疏度={sparsity:.2%}, 选中特征数={n_selected}")
        if n_selected <= 10:
            print(f"    选中特征索引: {selected}")
    
    print("\n2. 稀疏系数矩阵示例 (eta=0.5):")
    model, coef, intercept = train_spls_model(
        X, Y, n_components=3, eta=0.5, scale=True, fix_sign=True
    )
    selected = model.get_selected_features()
    
    print(f"  系数矩阵形状: {coef.shape}")
    print(f"  非零系数数量: {np.sum(np.abs(coef) > 1e-10)} / {coef.size}")
    print(f"\n  稀疏系数 (仅显示非零行):")
    for i in range(n_features):
        if np.any(np.abs(coef[i, :]) > 1e-10):
            row_str = "  ".join([f"{coef[i, j]:8.4f}" for j in range(n_targets)])
            print(f"  特征{i:2d}: [{row_str}] {' *' if i < n_relevant else ''}")
    
    print("\n3. 预测性能对比:")
    X_test = np.random.randn(50, n_features)
    Y_test = X_test @ true_coef
    
    for eta in [0.0, 0.3, 0.5]:
        model, _, _ = train_spls_model(X, Y, n_components=3, eta=eta)
        Y_pred = predict_spls_model(model, X_test)
        mse = np.mean((Y_test - Y_pred) ** 2)
        n_selected = len(model.get_selected_features())
        print(f"  eta={eta:.1f}: MSE={mse:.6f}, 选中特征={n_selected}")
    
    print("\n4. VIP特征重要性:")
    vip = model.get_vip_scores()
    top_indices = np.argsort(vip)[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"  第{i+1}名: 特征{idx:2d}, VIP={vip[idx]:.4f}{' *' if idx < n_relevant else ''}")
    
    print("\n" + "=" * 60)
    print("SPLS测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_sign_consistency()
    print("\n" + "=" * 60 + "\n")
    test_sparse_pls()
