import warnings
import numpy as np


class StreamingCovariance:
    def __init__(self, n_features):
        self.n_features = n_features
        self.n_samples = 0
        self.mean = np.zeros(n_features, dtype=float)
        self.cov_matrix = np.zeros((n_features, n_features), dtype=float)
        self.M2 = np.zeros((n_features, n_features), dtype=float)

    def update(self, x):
        x = np.array(x, dtype=float)
        self.n_samples += 1
        n = self.n_samples
        delta = x - self.mean
        self.mean = self.mean + delta / n
        delta2 = x - self.mean
        self.M2 = self.M2 + np.outer(delta, delta2)
        if n > 1:
            self.cov_matrix = self.M2 / (n - 1)

    def batch_update(self, samples):
        for x in samples:
            self.update(x)

    def get_covariance(self):
        if self.n_samples <= 1:
            return np.eye(self.n_features) * 1e-10
        return self.cov_matrix.copy()

    def get_mean(self):
        return self.mean.copy()

    def reset(self):
        self.n_samples = 0
        self.mean = np.zeros(self.n_features, dtype=float)
        self.cov_matrix = np.zeros((self.n_features, self.n_features), dtype=float)
        self.M2 = np.zeros((self.n_features, self.n_features), dtype=float)


def standardized_euclidean_distance(samples, p1, p2):
    samples = np.array(samples, dtype=float)
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    stds = np.std(samples, axis=0, ddof=0)
    stds[stds == 0] = 1.0
    diff = p1 - p2
    return float(np.sqrt(np.sum((diff / stds) ** 2)))


def weighted_euclidean_distance(p1, p2, weights=None):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    d = p1.shape[0]

    if weights is None:
        weights = np.ones(d, dtype=float)
    else:
        weights = np.array(weights, dtype=float)
        if weights.shape[0] != d:
            raise ValueError(f"Weights dimension {weights.shape[0]} does not match points dimension {d}")
        if np.any(weights < 0):
            raise ValueError("Weights cannot be negative")

    diff = p1 - p2
    weighted_diff_sq = (weights ** 0.5) * diff
    distance = float(np.sqrt(np.sum(weighted_diff_sq ** 2)))

    per_dim_contribution = (weighted_diff_sq ** 2) / (distance ** 2 + 1e-15)
    per_dim_contribution = per_dim_contribution / np.sum(per_dim_contribution)

    return distance, per_dim_contribution, weights


def mahalanobis_distance(samples, p1, p2, regularization=1e-6, cond_warning_threshold=1e10):
    samples = np.array(samples, dtype=float)
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    cov = np.cov(samples, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])

    d = cov.shape[0]
    cond_num = np.linalg.cond(cov)
    warnings_list = []

    if cond_num > cond_warning_threshold:
        warnings_list.append(
            f"Covariance matrix is ill-conditioned (cond = {cond_num:.2e})."
        )

    cov_reg = cov + regularization * np.eye(d)
    diff = p1 - p2

    try:
        cov_inv = np.linalg.inv(cov_reg)
        method_used = "inverse"
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)
        method_used = "pseudo-inverse"
        warnings_list.append("Matrix is singular, pseudo-inverse is used.")

    if regularization > 0:
        warnings_list.append(
            f"Regularization delta={regularization:.2e} added to diagonal."
        )

    warnings_list.append(f"Condition number: {cond_num:.2e}, Method: {method_used}")

    dist_sq = diff @ cov_inv @ diff
    distance = float(np.sqrt(dist_sq))

    L = np.linalg.cholesky(cov_reg)
    L_inv = np.linalg.inv(L)
    transformed_diff = L_inv @ diff

    per_dim_contribution = (transformed_diff ** 2) / (dist_sq + 1e-15)
    per_dim_contribution = per_dim_contribution / np.sum(per_dim_contribution)

    return distance, warnings_list, per_dim_contribution


def streaming_mahalanobis_distance(streaming_cov, p1, p2, regularization=1e-6, cond_warning_threshold=1e10):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    cov = streaming_cov.get_covariance()
    d = cov.shape[0]

    cond_num = np.linalg.cond(cov)
    warnings_list = []

    if cond_num > cond_warning_threshold:
        warnings_list.append(
            f"Covariance matrix is ill-conditioned (cond = {cond_num:.2e})."
        )

    cov_reg = cov + regularization * np.eye(d)
    diff = p1 - p2

    try:
        cov_inv = np.linalg.inv(cov_reg)
        method_used = "inverse"
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)
        method_used = "pseudo-inverse"
        warnings_list.append("Matrix is singular, pseudo-inverse is used.")

    if regularization > 0:
        warnings_list.append(
            f"Regularization delta={regularization:.2e} added to diagonal."
        )

    warnings_list.append(f"Streaming samples: {streaming_cov.n_samples}, Condition number: {cond_num:.2e}, Method: {method_used}")

    dist_sq = diff @ cov_inv @ diff
    distance = float(np.sqrt(dist_sq))

    L = np.linalg.cholesky(cov_reg)
    L_inv = np.linalg.inv(L)
    transformed_diff = L_inv @ diff

    per_dim_contribution = (transformed_diff ** 2) / (dist_sq + 1e-15)
    per_dim_contribution = per_dim_contribution / np.sum(per_dim_contribution)

    return distance, warnings_list, per_dim_contribution


def compute_distances(samples, point1, point2, weights=None, regularization=1e-6, cond_warning_threshold=1e10):
    se_dist = standardized_euclidean_distance(samples, point1, point2)
    we_dist, we_contrib, weights_used = weighted_euclidean_distance(point1, point2, weights)
    ma_dist, ma_warnings, ma_contrib = mahalanobis_distance(
        samples, point1, point2, regularization, cond_warning_threshold
    )
    return {
        "standardized_euclidean": se_dist,
        "weighted_euclidean": {
            "distance": we_dist,
            "contribution": we_contrib,
            "weights": weights_used
        },
        "mahalanobis": {
            "distance": ma_dist,
            "warnings": ma_warnings,
            "contribution": ma_contrib
        }
    }


if __name__ == "__main__":
    samples_full_rank = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 3.5],
        [3.0, 3.0, 5.5],
        [4.0, 5.0, 4.0],
        [5.0, 6.0, 7.5],
    ]
    p1 = [1.0, 2.0, 3.0]
    p2 = [4.0, 5.0, 6.0]
    weights = [1.0, 2.0, 0.5]

    print("=== Full-rank covariance matrix test ===")
    result = compute_distances(samples_full_rank, p1, p2, weights=weights)
    print(f"Standardized Euclidean Distance: {result['standardized_euclidean']:.6f}")
    print(f"Weighted Euclidean Distance:     {result['weighted_euclidean']['distance']:.6f}")
    print(f"  Weights used: {result['weighted_euclidean']['weights']}")
    print(f"  Dim contribution (%): {[f'{c*100:.1f}%' for c in result['weighted_euclidean']['contribution']]}")
    print(f"Mahalanobis Distance:            {result['mahalanobis']['distance']:.6f}")
    print(f"  Dim contribution (%): {[f'{c*100:.1f}%' for c in result['mahalanobis']['contribution']]}")
    for w in result['mahalanobis']['warnings']:
        print(f"  [INFO] {w}")

    print()
    print("=== Singular covariance matrix test (rank-deficient) ===")
    samples_singular = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = compute_distances(samples_singular, p1, p2)
    print(f"Standardized Euclidean Distance: {result['standardized_euclidean']:.6f}")
    print(f"Weighted Euclidean Distance:     {result['weighted_euclidean']['distance']:.6f}")
    print(f"Mahalanobis Distance:            {result['mahalanobis']['distance']:.6f}")
    for w in result['mahalanobis']['warnings']:
        print(f"  [WARN/INFO] {w}")

    print()
    print("=== Streaming / Incremental update test ===")
    sc = StreamingCovariance(n_features=3)
    for i, x in enumerate(samples_full_rank):
        sc.update(x)
        ma_dist, ma_warns, ma_contrib = streaming_mahalanobis_distance(sc, p1, p2)
        print(f"  After {i+1} samples: Mahalanobis = {ma_dist:.6f}, n={sc.n_samples}")
    print(f"  Final streaming mean: {sc.get_mean()}")
    print(f"  Batch cov == Streaming cov: {np.allclose(np.cov(samples_full_rank, rowvar=False), sc.get_covariance())}")
