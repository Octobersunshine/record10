import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


VALID_KERNELS = ["gaussian", "epanechnikov", "uniform", "triangular"]


class NadarayaWatsonRegressor:
    def __init__(self, bandwidth=1.0, kernel="gaussian", adaptive=False, k_neighbors=10):
        self.bandwidth = bandwidth
        self.kernel = kernel.lower()
        if self.kernel not in VALID_KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Use one of: {VALID_KERNELS}"
            )
        self.adaptive = adaptive
        self.k_neighbors = k_neighbors
        self.X_train = None
        self.y_train = None
        self._local_bandwidths = None
        self._residual_variance = None
        self._fitted_values = None

    def _kernel(self, u):
        if self.kernel == "gaussian":
            return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        elif self.kernel == "epanechnikov":
            return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2), 0.0)
        elif self.kernel == "uniform":
            return np.where(np.abs(u) <= 1.0, 0.5, 0.0)
        elif self.kernel == "triangular":
            return np.where(np.abs(u) <= 1.0, (1.0 - np.abs(u)), 0.0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        if self.X_train.ndim == 1:
            self.X_train = self.X_train[:, np.newaxis]

        if self.adaptive:
            self._compute_local_bandwidths()

        self._fitted_values = self.predict(self.X_train)
        residuals = self.y_train - self._fitted_values
        n = len(self.y_train)
        eff_df = self._effective_degrees_of_freedom()
        self._residual_variance = np.sum(residuals ** 2) / max(n - eff_df, 1.0)

        return self

    def _compute_local_bandwidths(self):
        n = self.X_train.shape[0]
        k = min(self.k_neighbors, n - 1)
        local_h = np.zeros(n)

        for i in range(n):
            diffs = self.X_train[i] - self.X_train
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))
            sorted_dist = np.sort(distances)
            kth_dist = sorted_dist[k] if k < len(sorted_dist) else sorted_dist[-1]
            local_h[i] = kth_dist

        geom_mean = np.exp(np.mean(np.log(local_h + 1e-12)))
        lambda_i = (local_h / geom_mean) ** 0.5
        self._local_bandwidths = self.bandwidth * lambda_i

    def _effective_degrees_of_freedom(self):
        n = self.X_train.shape[0]
        if self.adaptive:
            trace = 0.0
            for i in range(n):
                diffs = self.X_train[i] - self.X_train
                distances = np.sqrt(np.sum(diffs ** 2, axis=1))
                h = self._local_bandwidths
                u = distances / h
                weights = self._kernel(u)
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    trace += weights[i] / weight_sum
            return trace
        else:
            diffs = self.X_train[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diffs ** 2, axis=2))
            u = distances / self.bandwidth
            weights = self._kernel(u)
            weight_sums = np.sum(weights, axis=1, keepdims=True)
            weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
            hat_matrix_diag = np.diag(weights / weight_sums)
            return np.sum(hat_matrix_diag)

    def predict(self, X, return_ci=False, alpha=0.05):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        n_test = X.shape[0]
        predictions = np.zeros(n_test)
        variances = np.zeros(n_test)

        for i in range(n_test):
            diffs = X[i] - self.X_train
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))

            if self.adaptive:
                h = self._local_bandwidths
            else:
                h = self.bandwidth

            u = distances / h
            weights = self._kernel(u)
            weight_sum = np.sum(weights)

            if weight_sum == 0.0:
                predictions[i] = np.mean(self.y_train)
                if self._residual_variance is not None:
                    variances[i] = np.var(self.y_train)
            else:
                w = weights / weight_sum
                predictions[i] = np.sum(w * self.y_train)
                if self._residual_variance is not None:
                    variances[i] = self._residual_variance * np.sum(w ** 2)

        if return_ci:
            z = norm.ppf(1 - alpha / 2)
            se = np.sqrt(variances)
            lower = predictions - z * se
            upper = predictions + z * se
            return predictions, lower, upper
        else:
            return predictions

    def _silverman_bandwidth(self):
        n = self.X_train.shape[0]
        d = self.X_train.shape[1]
        std = np.std(self.X_train, axis=0)
        sigma = np.mean(std)
        iqr = np.mean(
            np.percentile(self.X_train, 75, axis=0)
            - np.percentile(self.X_train, 25, axis=0)
        ) / 1.34
        s = min(sigma, iqr) if iqr > 0 else sigma
        if s == 0:
            s = 1.0
        h = 0.9 * s * n ** (-1.0 / (d + 4))
        return h

    def _loocv_score(self, bandwidth):
        original_bw = self.bandwidth
        self.bandwidth = bandwidth

        if self.adaptive:
            self._compute_local_bandwidths()

        n = self.X_train.shape[0]
        sq_errors = np.zeros(n)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_loo = self.X_train[mask]
            y_loo = self.y_train[mask]

            diffs = self.X_train[i] - X_loo
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))

            if self.adaptive:
                h = self._local_bandwidths[mask]
            else:
                h = bandwidth

            u = distances / h
            weights = self._kernel(u)
            weight_sum = np.sum(weights)

            if weight_sum == 0.0:
                sq_errors[i] = (self.y_train[i] - np.mean(y_loo)) ** 2
            else:
                y_pred_i = np.sum(weights * y_loo) / weight_sum
                sq_errors[i] = (self.y_train[i] - y_pred_i) ** 2

        self.bandwidth = original_bw
        if self.adaptive:
            self._compute_local_bandwidths()

        return np.mean(sq_errors)

    def fit_auto_bandwidth(self, X, y, method="cv", n_bandwidths=50, bw_range=None):
        self.fit(X, y)

        if method == "silverman":
            h_opt = self._silverman_bandwidth()
        elif method == "cv":
            h_silverman = self._silverman_bandwidth()
            if bw_range is None:
                h_min = max(h_silverman * 0.05, 1e-6)
                h_max = h_silverman * 5.0
            else:
                h_min, h_max = bw_range

            result = minimize_scalar(
                self._loocv_score,
                bounds=(h_min, h_max),
                method="bounded",
                options={"xatol": 1e-4, "maxiter": n_bandwidths},
            )
            h_opt = result.x
        else:
            raise ValueError(f"Unknown method: {method}. Use 'silverman' or 'cv'.")

        self.bandwidth = h_opt
        self.fit(self.X_train, self.y_train)
        return h_opt

    @property
    def recommended_bandwidth(self):
        return self.bandwidth

    @property
    def residual_variance(self):
        return self._residual_variance


if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    X_train = np.sort(np.random.uniform(-3, 3, n))
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, n)

    X_test = np.linspace(-3, 3, 200)
    y_true = np.sin(X_test)

    print("=== 核函数对比 (固定带宽 h=0.3) ===")
    for kernel in VALID_KERNELS:
        model = NadarayaWatsonRegressor(bandwidth=0.3, kernel=kernel)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        print(f"{kernel:<14}  RMSE={rmse:.4f}")

    print()
    print("=== 自适应带宽 vs 固定带宽 (高斯核) ===")
    model_fixed = NadarayaWatsonRegressor(bandwidth=0.3, kernel="gaussian")
    model_fixed.fit(X_train, y_train)
    y_pred_fixed = model_fixed.predict(X_test)
    rmse_fixed = np.sqrt(np.mean((y_pred_fixed - y_true) ** 2))
    print(f"固定带宽 h=0.3      RMSE={rmse_fixed:.4f}")

    model_adaptive = NadarayaWatsonRegressor(
        bandwidth=0.3, kernel="gaussian", adaptive=True, k_neighbors=10
    )
    model_adaptive.fit(X_train, y_train)
    y_pred_adapt = model_adaptive.predict(X_test)
    rmse_adapt = np.sqrt(np.mean((y_pred_adapt - y_true) ** 2))
    print(f"自适应带宽 (k=10)  RMSE={rmse_adapt:.4f}")

    print()
    print("=== 置信带演示 (高斯核 + CV带宽) ===")
    model = NadarayaWatsonRegressor(kernel="gaussian")
    h_cv = model.fit_auto_bandwidth(X_train, y_train, method="cv")
    y_pred, y_lower, y_upper = model.predict(X_test, return_ci=True, alpha=0.05)
    rmse_cv = np.sqrt(np.mean((y_pred - y_true) ** 2))
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    print(f"CV 带宽: {h_cv:.4f}")
    print(f"RMSE:    {rmse_cv:.4f}")
    print(f"95%置信带覆盖率: {coverage:.2%}")
    print(f"残差方差估计: {model.residual_variance:.4f}")
