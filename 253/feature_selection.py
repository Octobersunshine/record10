import warnings
from collections import Counter

import numpy as np


def variance_threshold_selection(X, threshold=0.0):
    variances = np.var(X, axis=0)

    near_one = np.all(np.abs(variances - 1.0) < 0.05)
    near_zero_mean = np.all(np.abs(np.mean(X, axis=0)) < 0.05)
    if near_one and near_zero_mean:
        warnings.warn(
            "检测到数据可能经过标准化（各特征方差≈1，均值≈0），"
            "方差阈值法将失效（所有特征方差相同无法区分）。"
            "建议使用未标准化数据，或改用相关系数等特征选择方法。",
            UserWarning,
            stacklevel=2,
        )

    retained_indices = np.where(variances >= threshold)[0]
    return retained_indices.tolist()


def correlation_selection(X, y, threshold=0.0):
    n_features = X.shape[1]
    correlations = np.zeros(n_features)
    y_mean = np.mean(y)
    y_centered = y - y_mean
    y_std = np.std(y)

    for i in range(n_features):
        col = X[:, i]
        col_mean = np.mean(col)
        col_centered = col - col_mean
        col_std = np.std(col)
        if col_std == 0 or y_std == 0:
            correlations[i] = 0.0
        else:
            correlations[i] = np.abs(np.dot(col_centered, y_centered) / (len(y) * col_std * y_std))

    retained_indices = np.where(correlations >= threshold)[0]
    return retained_indices.tolist()


def chi2_selection(X, y, n_features=None, p_value_threshold=None):
    n_samples, n_features_total = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    chi2_scores = np.zeros(n_features_total)

    for i in range(n_features_total):
        col = X[:, i]
        col_bins = np.unique(col)
        observed = np.zeros((len(col_bins), n_classes))

        for row_idx, val in enumerate(col):
            bin_idx = np.where(col_bins == val)[0][0]
            class_idx = np.where(classes == y[row_idx])[0][0]
            observed[bin_idx, class_idx] += 1

        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        total = observed.sum()

        if total == 0:
            chi2_scores[i] = 0.0
            continue

        expected = (row_totals @ col_totals) / total
        expected[expected == 0] = 1e-10

        chi2 = ((observed - expected) ** 2 / expected).sum()
        chi2_scores[i] = chi2

    sorted_indices = np.argsort(chi2_scores)[::-1]

    if n_features is not None:
        k = min(n_features, n_features_total)
        return sorted_indices[:k].tolist(), chi2_scores[sorted_indices[:k]]

    if p_value_threshold is not None:
        from scipy.stats import chi2 as chi2_dist
        p_values = 1 - chi2_dist.cdf(chi2_scores, (len(classes) - 1))
        retained = np.where(p_values < p_value_threshold)[0]
        return retained[np.argsort(chi2_scores[retained])[::-1]].tolist(), chi2_scores[retained]

    return sorted_indices.tolist(), chi2_scores


def mutual_info_selection(X, y, n_features=None, discrete_features=True):
    n_samples, n_features_total = X.shape
    y_counts = Counter(y)
    y_probs = {k: v / n_samples for k, v in y_counts.items()}

    mi_scores = np.zeros(n_features_total)

    for i in range(n_features_total):
        col = X[:, i]
        if not discrete_features:
            col = np.digitize(col, bins=np.percentile(col, [25, 50, 75]))

        xy_joint = Counter(zip(col, y))
        x_counts = Counter(col)
        x_probs = {k: v / n_samples for k, v in x_counts.items()}

        mi = 0.0
        for (x_val, y_val), count in xy_joint.items():
            p_xy = count / n_samples
            p_x = x_probs[x_val]
            p_y = y_probs[y_val]
            if p_x > 0 and p_y > 0:
                mi += p_xy * np.log2(p_xy / (p_x * p_y))

        mi_scores[i] = mi

    sorted_indices = np.argsort(mi_scores)[::-1]

    if n_features is not None:
        k = min(n_features, n_features_total)
        return sorted_indices[:k].tolist(), mi_scores[sorted_indices[:k]]

    return sorted_indices.tolist(), mi_scores


def find_elbow_point(scores):
    scores_sorted = np.sort(scores)[::-1]
    n = len(scores_sorted)

    if n <= 2:
        return n

    x0, y0 = 0, scores_sorted[0]
    x1, y1 = n - 1, scores_sorted[-1]

    line_len = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    if line_len == 0:
        return n

    max_dist = -1
    elbow_idx = 0

    for i in range(1, n - 1):
        px, py = i, scores_sorted[i]
        dist = abs((y1 - y0) * px - (x1 - x0) * py + x1 * y0 - y1 * x0) / line_len
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    return elbow_idx + 1


def auto_feature_selection(X, y, method='mutual_info', max_features=None):
    if max_features is None:
        max_features = X.shape[1]

    if method == 'chi2':
        _, scores = chi2_selection(X, y, n_features=None)
    elif method == 'mutual_info':
        _, scores = mutual_info_selection(X, y, n_features=None)
    elif method == 'correlation':
        n_features = X.shape[1]
        y_mean = np.mean(y)
        y_centered = y - y_mean
        y_std = np.std(y)
        scores = np.zeros(n_features)
        for i in range(n_features):
            col = X[:, i]
            col_mean = np.mean(col)
            col_centered = col - col_mean
            col_std = np.std(col)
            if col_std == 0 or y_std == 0:
                scores[i] = 0.0
            else:
                scores[i] = np.abs(np.dot(col_centered, y_centered) / (len(y) * col_std * y_std))
    else:
        raise ValueError(f"不支持的方法: {method}，可选: 'chi2', 'mutual_info', 'correlation'")

    optimal_k = find_elbow_point(scores)
    optimal_k = min(optimal_k, max_features)
    optimal_k = max(1, optimal_k)

    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices[:optimal_k].tolist(), scores[sorted_indices[:optimal_k]], optimal_k


if __name__ == "__main__":
    X = np.array([
        [1, 2, 3, 0],
        [1, 4, 1, 0],
        [1, 6, 2, 0],
        [1, 8, 4, 0],
    ], dtype=float)
    y = np.array([1, 2, 3, 4], dtype=float)

    var_indices = variance_threshold_selection(X, threshold=0.5)
    print(f"方差选择保留的特征索引: {var_indices}")

    corr_indices = correlation_selection(X, y, threshold=0.5)
    print(f"相关系数选择保留的特征索引: {corr_indices}")

    X_clean = np.array([
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 50.0],
        [3.0, 30.0, 80.0],
        [4.0, 40.0, 60.0],
        [5.0, 50.0, 70.0],
    ])
    X_std = (X_clean - X_clean.mean(axis=0)) / X_clean.std(axis=0)
    print("\n--- 标准化数据测试 ---")
    var_indices_std = variance_threshold_selection(X_std, threshold=0.5)
    print(f"标准化数据方差选择保留的特征索引: {var_indices_std}")

    print("\n" + "="*50)
    print("卡方检验 & 互信息 & 肘部法则测试")
    print("="*50)

    np.random.seed(42)
    n_samples = 100
    X_cls = np.zeros((n_samples, 6))
    y_cls = np.random.randint(0, 2, n_samples)

    X_cls[:, 0] = y_cls * 3 + np.random.randn(n_samples) * 0.5
    X_cls[:, 1] = y_cls * 2 + np.random.randn(n_samples) * 0.8
    X_cls[:, 2] = np.random.randn(n_samples) * 2
    X_cls[:, 3] = (y_cls == 0) * 5 + np.random.randn(n_samples) * 0.3
    X_cls[:, 4] = np.random.randn(n_samples)
    X_cls[:, 5] = np.random.randn(n_samples) * 0.1

    X_cls_disc = np.round(X_cls).astype(int)

    chi2_indices, chi2_scores = chi2_selection(X_cls_disc, y_cls, n_features=3)
    print(f"\n卡方检验 Top3 特征索引: {chi2_indices}")
    print(f"卡方检验分数: {chi2_scores.round(3)}")

    mi_indices, mi_scores = mutual_info_selection(X_cls_disc, y_cls, n_features=3)
    print(f"\n互信息 Top3 特征索引: {mi_indices}")
    print(f"互信息分数: {mi_scores.round(4)}")

    auto_indices, auto_scores, best_k = auto_feature_selection(X_cls_disc, y_cls, method='mutual_info')
    print(f"\n肘部法则自动选择 - 最佳特征数: {best_k}")
    print(f"自动选择的特征索引: {auto_indices}")
    print(f"对应互信息分数: {auto_scores.round(4)}")

    auto_chi2_indices, auto_chi2_scores, best_k_chi2 = auto_feature_selection(X_cls_disc, y_cls, method='chi2')
    print(f"\n卡方+肘部法则 - 最佳特征数: {best_k_chi2}")
    print(f"自动选择的特征索引: {auto_chi2_indices}")
