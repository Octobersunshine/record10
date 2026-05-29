import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from scipy import stats

MIN_BANDWIDTH = 0.1


class LOWESSResult:
    def __init__(self, x: np.ndarray, y_fitted: np.ndarray,
                 ci_lower: np.ndarray, ci_upper: np.ndarray,
                 residuals: np.ndarray, robust_weights: np.ndarray):
        self.x = x
        self.y_fitted = y_fitted
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.residuals = residuals
        self.robust_weights = robust_weights

    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y_fitted': self.y_fitted,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'residuals': self.residuals,
            'robust_weights': self.robust_weights,
        }


class LOWESS:
    def __init__(self, bandwidth: float = 0.2, kernel: str = 'tricubic',
                 robust_iters: int = 3, confidence: float = 0.95):
        self.bandwidth = max(bandwidth, MIN_BANDWIDTH)
        self.kernel = kernel
        self.robust_iters = robust_iters
        self.confidence = confidence
        self.x = None
        self.y = None
        self.best_bandwidth_ = None
        self._robust_weights = None
        self._fitted_values = None

    def _tricubic_kernel(self, distances: np.ndarray) -> np.ndarray:
        mask = np.abs(distances) <= 1
        weights = np.zeros_like(distances)
        weights[mask] = (1 - np.abs(distances[mask]) ** 3) ** 3
        return weights

    def _bisquare_kernel(self, distances: np.ndarray) -> np.ndarray:
        mask = np.abs(distances) <= 1
        weights = np.zeros_like(distances)
        weights[mask] = (1 - distances[mask] ** 2) ** 2
        return weights

    def _get_weights(self, x_query: float, x_data: np.ndarray,
                     exclude_idx: Optional[int] = None) -> np.ndarray:
        n = len(x_data)
        k = max(3, int(self.bandwidth * n))
        distances = np.abs(x_data - x_query)
        sorted_indices = np.argsort(distances)
        k = min(k, n)
        k_nearest_indices = sorted_indices[:k]
        max_distance = distances[k_nearest_indices[-1]]
        if max_distance == 0:
            max_distance = 1.0
        normalized_distances = distances / max_distance
        weights = self._tricubic_kernel(normalized_distances)
        if self._robust_weights is not None:
            weights = weights * self._robust_weights
        if exclude_idx is not None:
            weights[exclude_idx] = 0.0
        return weights

    def _weighted_linear_regression(self, x: np.ndarray, y: np.ndarray,
                                   weights: np.ndarray) -> Tuple[float, float]:
        W = np.diag(weights)
        X = np.column_stack([np.ones_like(x), x])
        XTWX = X.T @ W @ X
        XTWy = X.T @ W @ y
        try:
            beta = np.linalg.solve(XTWX, XTWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return beta[0], beta[1]

    def _compute_local_fit(self, xq: float, x_data: np.ndarray,
                           y_data: np.ndarray,
                           weights: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        W = np.diag(weights)
        X = np.column_stack([np.ones_like(x_data), x_data])
        XTWX = X.T @ W @ X
        XTWy = X.T @ W @ y_data
        try:
            beta = np.linalg.solve(XTWX, XTWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X, y_data, rcond=None)[0]
        intercept, slope = beta[0], beta[1]
        xq_vec = np.array([1.0, xq])
        try:
            XTWX_inv = np.linalg.inv(XTWX)
        except np.linalg.LinAlgError:
            XTWX_inv = np.linalg.pinv(XTWX)
        leverage = xq_vec @ XTWX_inv @ XTWX @ XTWX_inv @ xq_vec
        hat_row = xq_vec @ XTWX_inv @ X.T @ W
        return intercept, slope, leverage, hat_row

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'LOWESS':
        self.x = np.asarray(x).flatten()
        self.y = np.asarray(y).flatten()
        self._robust_weights = None
        self._fitted_values = self._robust_lowess_fit()
        return self

    def _robust_lowess_fit(self) -> np.ndarray:
        n = len(self.x)
        self._robust_weights = np.ones(n)
        fitted = np.zeros(n)
        for iteration in range(self.robust_iters):
            for i in range(n):
                weights = self._get_weights(self.x[i], self.x, exclude_idx=None)
                intercept, slope = self._weighted_linear_regression(self.x, self.y, weights)
                fitted[i] = intercept + slope * self.x[i]
            if iteration < self.robust_iters - 1:
                residuals = self.y - fitted
                median_abs_res = np.median(np.abs(residuals))
                if median_abs_res < 1e-10:
                    break
                u = residuals / (6.0 * median_abs_res)
                self._robust_weights = self._bisquare_kernel(u)
        return fitted

    def auto_fit(self, x: np.ndarray, y: np.ndarray,
                 bw_candidates: Optional[np.ndarray] = None) -> 'LOWESS':
        self.x = np.asarray(x).flatten()
        self.y = np.asarray(y).flatten()
        if bw_candidates is None:
            bw_candidates = np.linspace(MIN_BANDWIDTH, 0.9, 17)
        best_score = np.inf
        best_bw = MIN_BANDWIDTH
        for bw in bw_candidates:
            score = self._loocv_score(bw)
            if score < best_score:
                best_score = score
                best_bw = bw
        self.bandwidth = best_bw
        self.best_bandwidth_ = best_bw
        self._robust_weights = None
        self._fitted_values = self._robust_lowess_fit()
        return self

    def _loocv_score(self, bandwidth: float) -> float:
        original_bw = self.bandwidth
        original_rw = self._robust_weights
        self.bandwidth = max(bandwidth, MIN_BANDWIDTH)
        self._robust_weights = None
        n = len(self.x)
        residuals = np.zeros(n)
        for i in range(n):
            weights = self._get_weights(self.x[i], self.x, exclude_idx=i)
            if np.sum(weights) < 1e-10:
                residuals[i] = self.y[i]
                continue
            intercept, slope = self._weighted_linear_regression(self.x, self.y, weights)
            y_pred = intercept + slope * self.x[i]
            residuals[i] = (self.y[i] - y_pred) ** 2
        self.bandwidth = original_bw
        self._robust_weights = original_rw
        return np.mean(residuals)

    def predict(self, x_query: np.ndarray) -> np.ndarray:
        if self.x is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")
        x_query = np.asarray(x_query).flatten()
        predictions = np.zeros_like(x_query, dtype=float)
        x_min, x_max = self.x.min(), self.x.max()
        for i, xq in enumerate(x_query):
            if xq < x_min:
                idx = np.argmin(np.abs(self.x - x_min))
                weights = self._get_weights(x_min, self.x)
                intercept, slope = self._weighted_linear_regression(self.x, self.y, weights)
                predictions[i] = intercept + slope * xq
            elif xq > x_max:
                idx = np.argmin(np.abs(self.x - x_max))
                weights = self._get_weights(x_max, self.x)
                intercept, slope = self._weighted_linear_regression(self.x, self.y, weights)
                predictions[i] = intercept + slope * xq
            else:
                weights = self._get_weights(xq, self.x)
                intercept, slope = self._weighted_linear_regression(self.x, self.y, weights)
                predictions[i] = intercept + slope * xq
        return predictions

    def fit_result(self, x_eval: Optional[np.ndarray] = None) -> LOWESSResult:
        if self.x is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")
        if x_eval is None:
            x_eval = np.sort(self.x)
        else:
            x_eval = np.sort(np.asarray(x_eval).flatten())
        n_eval = len(x_eval)
        y_fitted = np.zeros(n_eval)
        ci_lower = np.zeros(n_eval)
        ci_upper = np.zeros(n_eval)
        train_fitted = self._fitted_values if self._fitted_values is not None else self.predict(self.x)
        train_residuals = self.y - train_fitted
        n_train = len(self.x)
        nonzero_mask = np.abs(train_residuals) > 1e-10
        if np.sum(nonzero_mask) > 2:
            residual_var = np.sum(train_residuals[nonzero_mask] ** 2) / (np.sum(nonzero_mask) - 2)
        else:
            residual_var = np.var(train_residuals)
        z_val = stats.norm.ppf(1 - (1 - self.confidence) / 2)
        x_min, x_max = self.x.min(), self.x.max()
        for i, xq in enumerate(x_eval):
            if xq < x_min:
                boundary = x_min
            elif xq > x_max:
                boundary = x_max
            else:
                boundary = xq
            weights = self._get_weights(boundary, self.x)
            intercept, slope, leverage, hat_row = self._compute_local_fit(
                boundary, self.x, self.y, weights)
            y_fitted[i] = intercept + slope * xq
            std_err = np.sqrt(residual_var * max(leverage, 0))
            half_width = z_val * std_err
            ci_lower[i] = y_fitted[i] - half_width
            ci_upper[i] = y_fitted[i] + half_width
        robust_w = self._robust_weights.copy() if self._robust_weights is not None else np.ones(n_eval)
        return LOWESSResult(
            x=x_eval,
            y_fitted=y_fitted,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            residuals=train_residuals,
            robust_weights=robust_w,
        )


def main():
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 2 * np.pi, n)
    y_true = np.sin(x)
    noise = np.random.normal(0, 0.3, size=n)
    y = y_true + noise

    outlier_idx = np.random.choice(n, 8, replace=False)
    y[outlier_idx] += np.random.choice([-1, 1], size=8) * np.random.uniform(1.5, 2.5, size=8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- Panel 1: Robust LOWESS vs Non-robust ---
    ax = axes[0, 0]
    ax.scatter(x, y, alpha=0.5, color='gray', label='Data (with outliers)')
    ax.plot(x, y_true, 'k--', label='True sin(x)', linewidth=1.5)
    model_nr = LOWESS(bandwidth=0.3, robust_iters=1)
    model_nr.fit(x, y)
    y_nr = model_nr.predict(x)
    ax.plot(x, y_nr, 'blue', linewidth=2, label='Non-robust (1 iter)')
    model_r = LOWESS(bandwidth=0.3, robust_iters=4)
    model_r.fit(x, y)
    y_r = model_r.predict(x)
    ax.plot(x, y_r, 'red', linewidth=2, label='Robust (4 iters)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Robust LOWESS vs Non-Robust')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Extrapolation ---
    ax = axes[0, 1]
    x_extrap = np.linspace(-1, 2 * np.pi + 1, 150)
    model_ext = LOWESS(bandwidth=0.3, robust_iters=3)
    model_ext.fit(x, y)
    y_ext = model_ext.predict(x_extrap)
    ax.scatter(x, y, alpha=0.5, color='gray', label='Training data')
    ax.plot(x, y_true, 'k--', label='True sin(x)', linewidth=1.5)
    ax.axvspan(-1, 0, alpha=0.1, color='orange', label='Extrapolation zone')
    ax.axvspan(2 * np.pi, 2 * np.pi + 1, alpha=0.1, color='orange')
    ax.plot(x_extrap, y_ext, 'red', linewidth=2, label='LOWESS (with extrapolation)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Extrapolation Beyond Training Range')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Confidence Interval ---
    ax = axes[1, 0]
    model_ci = LOWESS(bandwidth=0.3, robust_iters=3, confidence=0.95)
    model_ci.fit(x, y)
    result = model_ci.fit_result()
    ax.scatter(x, y, alpha=0.4, color='gray', label='Data')
    ax.plot(result.x, result.y_fitted, 'blue', linewidth=2, label='Fitted curve')
    ax.fill_between(result.x, result.ci_lower, result.ci_upper,
                    alpha=0.2, color='blue', label='95% CI')
    ax.plot(x, y_true, 'k--', label='True sin(x)', linewidth=1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('95% Pointwise Confidence Interval')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Robust weights ---
    ax = axes[1, 1]
    ax.scatter(x, y, alpha=0.5, color='gray', label='Data')
    outlier_mask = np.zeros(n, dtype=bool)
    outlier_mask[outlier_idx] = True
    normal_idx = np.where(~outlier_mask)[0]
    outlier_idx_sorted = np.sort(outlier_idx)
    ax.scatter(x[normal_idx], y[normal_idx], alpha=0.5, color='gray')
    ax.scatter(x[outlier_idx_sorted], y[outlier_idx_sorted], alpha=0.7,
               color='red', s=60, marker='x', label='Outliers', zorder=5)
    rw = result.robust_weights
    ax2 = ax.twinx()
    ax2.plot(x, rw, 'g-', alpha=0.7, linewidth=1.5, label='Robust weights')
    ax2.set_ylabel('Robust weight', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Robust Weights (outliers down-weighted)')
    ax.legend(fontsize=8, loc='upper left')
    ax2.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lowess_demo.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'lowess_demo.png'")

    print("\n=== Robust LOWESS ===")
    mse_nr = np.mean((y_nr - y_true) ** 2)
    mse_r = np.mean((y_r - y_true) ** 2)
    print(f"Non-robust MSE: {mse_nr:.4f}")
    print(f"Robust MSE:     {mse_r:.4f}")

    print("\n=== Extrapolation ===")
    x_test_ext = np.array([-0.5, 0.0, np.pi, 2 * np.pi, 2 * np.pi + 0.5])
    y_ext_test = model_ext.predict(x_test_ext)
    for xt, yp in zip(x_test_ext, y_ext_test):
        status = "extrapolated" if (xt < x.min() or xt > x.max()) else "interpolated"
        print(f"  x={xt:.2f}, predicted={yp:.4f} ({status})")

    print("\n=== Confidence Interval (sample points) ===")
    sample_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    for idx in sample_idx:
        print(f"  x={result.x[idx]:.4f}, fitted={result.y_fitted[idx]:.4f}, "
              f"CI=[{result.ci_lower[idx]:.4f}, {result.ci_upper[idx]:.4f}]")

    print("\n=== fit_result dict keys ===")
    rd = result.to_dict()
    for k, v in rd.items():
        print(f"  {k}: shape={v.shape}")


if __name__ == "__main__":
    main()
