import numpy as np
from collections import defaultdict


def variance_threshold_selection(X, threshold=0.0, tol=1e-2):
    variances = np.var(X, axis=0)

    non_zero_var = variances[variances > 0]
    if len(non_zero_var) > 0:
        mean_var = np.mean(non_zero_var)
        var_std = np.std(non_zero_var)
        is_standardized = (
            abs(mean_var - 1.0) < tol
            and var_std < tol
        )
    else:
        is_standardized = False

    if is_standardized:
        print(
            "WARNING: Detected data may be standardized (all non-zero variances ≈ 1.0). "
            "Variance threshold selection is ineffective on standardized data. "
            "Please use unstandardized raw data, or consider using correlation_selection() instead."
        )

    mask = variances >= threshold
    retained_indices = np.where(mask)[0]
    scores = variances[mask]
    return retained_indices, scores


def correlation_selection(X, y, threshold=0.0, top_k=None):
    y_mean = np.mean(y)
    y_centered = y - y_mean
    y_std = np.std(y)
    n_features = X.shape[1]
    correlations = np.zeros(n_features)

    for i in range(n_features):
        col = X[:, i]
        col_mean = np.mean(col)
        col_centered = col - col_mean
        col_std = np.std(col)
        if col_std == 0 or y_std == 0:
            correlations[i] = 0.0
        else:
            correlations[i] = np.abs(
                np.dot(col_centered, y_centered) / (len(y) * col_std * y_std)
            )

    if top_k is not None:
        sorted_indices = np.argsort(correlations)[::-1]
        retained_indices = sorted_indices[:top_k]
        retained_indices = np.sort(retained_indices)
    else:
        retained_indices = np.where(correlations >= threshold)[0]

    scores = correlations[retained_indices]
    return retained_indices, scores


def chi2_selection(X, y, threshold=0.0, top_k=None):
    n_samples, n_features = X.shape
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    n_classes = len(classes)
    chi2_scores = np.zeros(n_features)

    for i in range(n_features):
        col = X[:, i].astype(int)
        feature_vals = np.unique(col)
        observed = np.zeros((len(feature_vals), n_classes))

        for f_idx, f_val in enumerate(feature_vals):
            for c_idx, c_val in enumerate(classes):
                observed[f_idx, c_idx] = np.sum((col == f_val) & (y == c_val))

        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        total = observed.sum()

        if total == 0:
            chi2_scores[i] = 0.0
            continue

        expected = row_totals @ col_totals / total
        expected[expected == 0] = 1e-10

        chi2 = np.sum((observed - expected) ** 2 / expected)
        chi2_scores[i] = chi2

    if top_k is not None:
        sorted_indices = np.argsort(chi2_scores)[::-1]
        retained_indices = sorted_indices[:top_k]
        retained_indices = np.sort(retained_indices)
    else:
        retained_indices = np.where(chi2_scores >= threshold)[0]

    scores = chi2_scores[retained_indices]
    return retained_indices, scores


def _entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-10))


def _mutual_info_discrete(x, y):
    n = len(y)
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    mi = 0.0

    for xv in x_unique:
        for yv in y_unique:
            p_xy = np.sum((x == xv) & (y == yv)) / n
            if p_xy > 0:
                p_x = np.sum(x == xv) / n
                p_y = np.sum(y == yv) / n
                mi += p_xy * np.log2(p_xy / (p_x * p_y + 1e-10))

    return mi


def _mutual_info_continuous(x, y, n_bins=10):
    x_bins = np.digitize(x, np.linspace(x.min(), x.max(), n_bins + 1))
    return _mutual_info_discrete(x_bins, y)


def mutual_info_selection(X, y, discrete_features="auto", threshold=0.0, top_k=None, n_bins=10):
    n_samples, n_features = X.shape
    y = np.asarray(y)

    if y.dtype == np.float64 and len(np.unique(y)) > 10:
        y_bins = np.digitize(y, np.linspace(y.min(), y.max(), n_bins + 1))
    else:
        y = y.astype(int)
        y_bins = y

    mi_scores = np.zeros(n_features)

    if discrete_features == "auto":
        discrete_mask = np.array([len(np.unique(X[:, i])) <= 10 for i in range(n_features)])
    elif isinstance(discrete_features, list):
        discrete_mask = np.array([i in discrete_features for i in range(n_features)])
    else:
        discrete_mask = np.zeros(n_features, dtype=bool)

    for i in range(n_features):
        if discrete_mask[i]:
            x_col = X[:, i].astype(int)
            mi_scores[i] = _mutual_info_discrete(x_col, y_bins)
        else:
            x_col = X[:, i].astype(float)
            mi_scores[i] = _mutual_info_continuous(x_col, y_bins, n_bins)

    if top_k is not None:
        sorted_indices = np.argsort(mi_scores)[::-1]
        retained_indices = sorted_indices[:top_k]
        retained_indices = np.sort(retained_indices)
    else:
        retained_indices = np.where(mi_scores >= threshold)[0]

    scores = mi_scores[retained_indices]
    return retained_indices, scores


def _simple_classifier_predict(X_train, y_train, X_test):
    predictions = []
    for x in X_test:
        distances = np.sum(np.abs(X_train - x), axis=1)
        nearest_idx = np.argmin(distances)
        predictions.append(y_train[nearest_idx])
    return np.array(predictions)


def _simple_regressor_predict(X_train, y_train, X_test):
    predictions = []
    for x in X_test:
        distances = np.sum(np.abs(X_train - x), axis=1)
        nearest_idx = np.argmin(distances)
        predictions.append(y_train[nearest_idx])
    return np.array(predictions)


def select_optimal_k_cv(
    X,
    y,
    selector_func,
    k_candidates=None,
    cv_folds=5,
    is_classification=True,
    **selector_kwargs
):
    n_samples = X.shape[0]
    max_possible_k = X.shape[1]

    if k_candidates is None:
        k_candidates = []
        k = 1
        while k <= max_possible_k:
            k_candidates.append(k)
            if k < 5:
                k += 1
            elif k < 20:
                k += 2
            else:
                k += 5
        k_candidates = [k for k in k_candidates if k <= max_possible_k]

    fold_size = n_samples // cv_folds
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_scores = defaultdict(list)

    for fold in range(cv_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < cv_folds - 1 else n_samples

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for k in k_candidates:
            if k > X_train.shape[1]:
                continue

            selector_kwargs["top_k"] = k
            selected_indices, _ = selector_func(X_train, y_train, **selector_kwargs)

            if len(selected_indices) == 0:
                fold_scores[k].append(0.0 if is_classification else float("inf"))
                continue

            X_train_selected = X_train[:, selected_indices]
            X_val_selected = X_val[:, selected_indices]

            if is_classification:
                y_pred = _simple_classifier_predict(X_train_selected, y_train, X_val_selected)
                acc = np.mean(y_pred == y_val)
                fold_scores[k].append(acc)
            else:
                y_pred = _simple_regressor_predict(X_train_selected, y_train, X_val_selected)
                mse = np.mean((y_pred - y_val) ** 2)
                fold_scores[k].append(mse)

    avg_scores = {}
    for k in k_candidates:
        if k in fold_scores and len(fold_scores[k]) > 0:
            avg_scores[k] = np.mean(fold_scores[k])

    if not avg_scores:
        return max_possible_k, avg_scores

    if is_classification:
        best_k = max(avg_scores.keys(), key=lambda k: avg_scores[k])
    else:
        best_k = min(avg_scores.keys(), key=lambda k: avg_scores[k])

    return best_k, avg_scores


if __name__ == "__main__":
    np.random.seed(42)
    X = np.array(
        [
            [1, 1, 1, 0, 10],
            [1, 1, 1, 1, 20],
            [1, 1, 1, 0, 30],
            [1, 1, 1, 1, 40],
            [1, 1, 1, 0, 50],
        ]
    )
    y = np.array([10, 25, 12, 28, 55])

    indices, scores = variance_threshold_selection(X, threshold=0.0)
    print("=== Variance Threshold Selection ===")
    print(f"Retained feature indices: {indices}")
    print(f"Variance scores:         {scores}")

    indices, scores = correlation_selection(X, y, threshold=0.3)
    print("\n=== Correlation Selection ===")
    print(f"Retained feature indices:    {indices}")
    print(f"Absolute correlation scores: {scores}")

    indices, scores = correlation_selection(X, y, top_k=2)
    print("\n=== Correlation Selection (top_k=2) ===")
    print(f"Retained feature indices:    {indices}")
    print(f"Absolute correlation scores: {scores}")

    print("\n=== Variance Threshold on Standardized Data ===")
    X_non_const = X[:, 3:]
    X_std = (X_non_const - np.mean(X_non_const, axis=0)) / np.std(X_non_const, axis=0)
    print(f"Standardized data:\n{X_std}")
    print(f"Variances: {np.var(X_std, axis=0)}")
    indices, scores = variance_threshold_selection(X_std, threshold=0.5)
    print(f"Retained feature indices: {indices}")
    print(f"Variance scores:         {scores}")

    print("\n=== Chi-square Selection (Classification) ===")
    y_cls = np.array([0, 1, 0, 1, 1])
    X_discrete = np.array([
        [1, 0, 2, 0, 1],
        [1, 1, 0, 1, 2],
        [0, 0, 1, 0, 0],
        [1, 1, 2, 1, 1],
        [0, 0, 0, 0, 2],
    ])
    indices, scores = chi2_selection(X_discrete, y_cls, top_k=3)
    print(f"Retained feature indices:    {indices}")
    print(f"Chi-square scores:           {scores}")

    print("\n=== Mutual Information Selection ===")
    indices, scores = mutual_info_selection(X, y, top_k=3, n_bins=5)
    print(f"Retained feature indices:    {indices}")
    print(f"Mutual information scores:   {scores}")

    print("\n=== Mutual Information (Classification) ===")
    indices, scores = mutual_info_selection(X_discrete, y_cls, discrete_features="auto", top_k=3)
    print(f"Retained feature indices:    {indices}")
    print(f"Mutual information scores:   {scores}")

    print("\n=== CV Optimal K Selection (Classification) ===")
    np.random.seed(123)
    X_large = np.random.randint(0, 3, size=(50, 8))
    y_large = np.random.randint(0, 2, size=50)
    best_k, cv_scores = select_optimal_k_cv(
        X_large, y_large,
        selector_func=chi2_selection,
        cv_folds=5,
        is_classification=True
    )
    print(f"Best K: {best_k}")
    print(f"CV Scores (k: accuracy):")
    for k, acc in sorted(cv_scores.items()):
        print(f"  k={k}: {acc:.4f}")

    print("\n=== CV Optimal K Selection (Regression) ===")
    np.random.seed(456)
    X_reg = np.random.rand(60, 6)
    y_reg = X_reg[:, 0] * 10 + X_reg[:, 1] * 5 + np.random.randn(60) * 0.5
    best_k, cv_scores = select_optimal_k_cv(
        X_reg, y_reg,
        selector_func=mutual_info_selection,
        cv_folds=5,
        is_classification=False,
        discrete_features=[],
        n_bins=10
    )
    print(f"Best K: {best_k}")
    print(f"CV Scores (k: MSE):")
    for k, mse in sorted(cv_scores.items()):
        print(f"  k={k}: {mse:.4f}")
