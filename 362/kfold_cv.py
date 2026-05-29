import numpy as np
from itertools import combinations
from collections import defaultdict


def kfold_split(n_samples, k=5, shuffle=True, random_state=None):
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size
    return folds


def stratified_kfold_split(y, k=5, shuffle=True, random_state=None):
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    unique_classes, y_counts = np.unique(y, return_counts=True)

    if np.any(y_counts < k):
        min_count = y_counts.min()
        raise ValueError(
            f"Class '{unique_classes[np.argmin(y_counts)]}' has only {min_count} "
            f"samples, which is less than k={k}. Cannot stratify."
        )

    folds = [[] for _ in range(k)]
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if shuffle:
            rng.shuffle(cls_indices)
        cls_fold_sizes = np.full(k, len(cls_indices) // k, dtype=int)
        cls_fold_sizes[:len(cls_indices) % k] += 1
        current = 0
        for fold_idx in range(k):
            start = current
            end = current + cls_fold_sizes[fold_idx]
            folds[fold_idx].extend(cls_indices[start:end])
            current = end

    folds = [np.array(fold) for fold in folds]
    for i in range(k):
        if shuffle:
            rng.shuffle(folds[i])
    return folds


def loo_split(n_samples):
    indices = np.arange(n_samples)
    splits = []
    for i in range(n_samples):
        val_idx = np.array([i])
        train_idx = np.concatenate([indices[:i], indices[i + 1:]])
        splits.append((train_idx, val_idx))
    return splits


def lpo_split(n_samples, p=2):
    if p < 1 or p >= n_samples:
        raise ValueError(f"p must be in [1, n_samples-1], got p={p}")
    indices = np.arange(n_samples)
    n_combos = 1
    for i in range(p):
        n_combos = n_combos * (n_samples - i) // (i + 1)
    if n_combos > 10000:
        print(f"Warning: Leave-P-Out with p={p} generates {n_combos} splits. "
              f"Consider using smaller p or K-Fold instead.")
    splits = []
    for val_tuple in combinations(range(n_samples), p):
        val_idx = np.array(val_tuple)
        mask = np.ones(n_samples, dtype=bool)
        mask[val_idx] = False
        train_idx = indices[mask]
        splits.append((train_idx, val_idx))
    return splits


def repeated_kfold_split(n_samples, k=5, n_repeats=3, random_state=None):
    rng = np.random.RandomState(random_state)
    all_splits = []
    for repeat_idx in range(n_repeats):
        seed = rng.randint(0, 2 ** 31)
        folds = kfold_split(n_samples, k, shuffle=True, random_state=seed)
        for fold_idx in range(k):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])
            all_splits.append((train_idx, val_idx, repeat_idx, fold_idx))
    return all_splits


def repeated_stratified_kfold_split(y, k=5, n_repeats=3, random_state=None):
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    unique_classes, y_counts = np.unique(y, return_counts=True)
    if np.any(y_counts < k):
        min_count = y_counts.min()
        raise ValueError(
            f"Class '{unique_classes[np.argmin(y_counts)]}' has only {min_count} "
            f"samples, which is less than k={k}. Cannot stratify."
        )
    all_splits = []
    for repeat_idx in range(n_repeats):
        seed = rng.randint(0, 2 ** 31)
        folds = stratified_kfold_split(y, k, shuffle=True, random_state=seed)
        for fold_idx in range(k):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])
            all_splits.append((train_idx, val_idx, repeat_idx, fold_idx))
    return all_splits


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, pos_label=1):
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred, pos_label=1):
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, pos_label=1):
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mae_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse_score(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse_score(y_true, y_pred):
    return np.sqrt(mse_score(y_true, y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def _detect_task_type(y):
    y = np.asarray(y)
    if y.dtype.kind in ('f', 'c'):
        if np.all(y == y.astype(int)):
            n_unique = len(np.unique(y))
            return 'classification' if n_unique <= 20 else 'regression'
        return 'regression'
    n_unique = len(np.unique(y))
    return 'classification' if n_unique <= 20 else 'regression'


def _compute_metrics(y_true, y_pred, task_type, pos_label):
    if task_type == 'classification':
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label),
            'recall': recall_score(y_true, y_pred, pos_label),
            'f1': f1_score(y_true, y_pred, pos_label),
        }
    else:
        return {
            'mae': mae_score(y_true, y_pred),
            'mse': mse_score(y_true, y_pred),
            'rmse': rmse_score(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }


def _print_fold_result(fold_label, metrics, task_type):
    if task_type == 'classification':
        print(f"{fold_label}  "
              f"Acc={metrics['accuracy']:.4f}  "
              f"P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  "
              f"F1={metrics['f1']:.4f}")
    else:
        print(f"{fold_label}  "
              f"MAE={metrics['mae']:.4f}  "
              f"MSE={metrics['mse']:.4f}  "
              f"RMSE={metrics['rmse']:.4f}  "
              f"R²={metrics['r2']:.4f}")


def _print_summary(metrics, task_type):
    metric_names = {
        'classification': ['accuracy', 'precision', 'recall', 'f1'],
        'regression': ['mae', 'mse', 'rmse', 'r2']
    }[task_type]
    print("\n" + "=" * 60)
    print(f"{'Metric':<12}{'Mean':>10}{'Std':>12}{'Min':>10}{'Max':>10}")
    print("-" * 60)
    for name in metric_names:
        values = metrics[name]
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        print(f"{name:<12}{mean:>10.4f}{std:>12.4f}{np.min(values):>10.4f}{np.max(values):>10.4f}")
    print("=" * 60)


def _print_predictions(predictions, task_type, max_display=20):
    print("\n--- Detailed Predictions (per fold) ---")
    n_folds = len(predictions)
    show_limit = min(n_folds, max_display)
    for i in range(show_limit):
        p = predictions[i]
        y_true = p['y_true']
        y_pred = p['y_pred']
        label = p.get('label', f'Fold {i + 1}')
        if task_type == 'classification':
            correct = y_true == y_pred
            details = ' '.join(
                f"{'✓' if c else '✗'}" for c in correct[:30]
            )
            n_correct = np.sum(correct)
            print(f"  {label} [{n_correct}/{len(y_true)} correct]: {details}")
        else:
            errors = np.abs(y_true - y_pred)
            detail_str = ' '.join(f"{e:.2f}" for e in errors[:20])
            print(f"  {label} [MAE={np.mean(errors):.4f}]: |err|=[{detail_str}]")
        if len(y_true) > 30 and task_type == 'classification':
            print(f"    ... ({len(y_true) - 30} more samples)")
        elif len(y_true) > 20 and task_type != 'classification':
            print(f"    ... ({len(y_true) - 20} more samples)")
    if n_folds > show_limit:
        print(f"  ... ({n_folds - show_limit} more folds not shown)")


def kfold_cross_validation(model, X, y, k=5, task_type='auto', stratified=None,
                           shuffle=True, random_state=None, pos_label=1,
                           verbose=True, return_predictions=False):
    if task_type == 'auto':
        task_type = _detect_task_type(y)
    if task_type not in ('classification', 'regression'):
        raise ValueError("task_type must be 'auto', 'classification', or 'regression'")

    if stratified is None:
        stratified = (task_type == 'classification')

    X = np.asarray(X)
    y = np.asarray(y)

    if stratified:
        folds = stratified_kfold_split(y, k, shuffle, random_state)
        split_strategy = 'Stratified K-Fold'
    else:
        folds = kfold_split(len(X), k, shuffle, random_state)
        split_strategy = 'Standard K-Fold'

    if verbose:
        print(f"\n[{task_type.capitalize()}] {split_strategy}, k={k}")
        print("-" * 60)

    metrics = defaultdict(list)
    predictions = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])

        fold_metrics = _compute_metrics(y[val_idx], y_pred, task_type, pos_label)
        for name, value in fold_metrics.items():
            metrics[name].append(value)

        predictions.append({
            'label': f'Fold {i + 1}/{k}',
            'val_indices': val_idx,
            'y_true': y[val_idx].copy(),
            'y_pred': y_pred.copy(),
        })

        if verbose:
            _print_fold_result(f'Fold {i + 1}/{k}', fold_metrics, task_type)

    if verbose:
        _print_summary(metrics, task_type)
        if return_predictions:
            _print_predictions(predictions, task_type)

    result = {
        'metrics': dict(metrics),
        'predictions': predictions,
        'split_strategy': split_strategy,
        'task_type': task_type,
        'k': k,
    }
    return result


def loocv(model, X, y, task_type='auto', pos_label=1, verbose=True,
          return_predictions=False):
    if task_type == 'auto':
        task_type = _detect_task_type(y)

    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)
    splits = loo_split(n_samples)

    if verbose:
        print(f"\n[{task_type.capitalize()}] Leave-One-Out CV, n={n_samples}")
        print("-" * 60)

    metrics = defaultdict(list)
    predictions = []

    for i, (train_idx, val_idx) in enumerate(splits):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])

        fold_metrics = _compute_metrics(y[val_idx], y_pred, task_type, pos_label)
        for name, value in fold_metrics.items():
            metrics[name].append(value)

        predictions.append({
            'label': f'Sample {i}',
            'val_indices': val_idx,
            'y_true': y[val_idx].copy(),
            'y_pred': y_pred.copy(),
        })

        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            running_means = {n: np.mean(v) for n, v in metrics.items()}
            _print_fold_result(f'Progress {i + 1}/{n_samples}', running_means, task_type)

    if verbose:
        _print_summary(metrics, task_type)
        if return_predictions:
            _print_predictions(predictions, task_type)

    return {
        'metrics': dict(metrics),
        'predictions': predictions,
        'split_strategy': 'Leave-One-Out',
        'task_type': task_type,
        'n_splits': n_samples,
    }


def lpocv(model, X, y, p=2, task_type='auto', pos_label=1, verbose=True,
           return_predictions=False):
    if task_type == 'auto':
        task_type = _detect_task_type(y)

    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)
    splits = lpo_split(n_samples, p)
    n_splits = len(splits)

    if verbose:
        print(f"\n[{task_type.capitalize()}] Leave-{p}-Out CV, "
              f"n={n_samples}, total_splits={n_splits}")
        print("-" * 60)

    metrics = defaultdict(list)
    predictions = []

    for i, (train_idx, val_idx) in enumerate(splits):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])

        fold_metrics = _compute_metrics(y[val_idx], y_pred, task_type, pos_label)
        for name, value in fold_metrics.items():
            metrics[name].append(value)

        predictions.append({
            'label': f'LPO split {i + 1}',
            'val_indices': val_idx,
            'y_true': y[val_idx].copy(),
            'y_pred': y_pred.copy(),
        })

        if verbose and (i + 1) % max(1, n_splits // 10) == 0:
            running_means = {n: np.mean(v) for n, v in metrics.items()}
            _print_fold_result(f'Progress {i + 1}/{n_splits}', running_means, task_type)

    if verbose:
        _print_summary(metrics, task_type)
        if return_predictions:
            _print_predictions(predictions, task_type)

    return {
        'metrics': dict(metrics),
        'predictions': predictions,
        'split_strategy': f'Leave-{p}-Out',
        'task_type': task_type,
        'n_splits': n_splits,
        'p': p,
    }


def repeated_kfold_cv(model, X, y, k=5, n_repeats=3, task_type='auto',
                      stratified=None, random_state=None, pos_label=1,
                      verbose=True, return_predictions=False):
    if task_type == 'auto':
        task_type = _detect_task_type(y)
    if task_type not in ('classification', 'regression'):
        raise ValueError("task_type must be 'auto', 'classification', or 'regression'")

    if stratified is None:
        stratified = (task_type == 'classification')

    X = np.asarray(X)
    y = np.asarray(y)

    if stratified:
        all_splits = repeated_stratified_kfold_split(y, k, n_repeats, random_state)
        split_strategy = f'Repeated Stratified K-Fold'
    else:
        all_splits = repeated_kfold_split(len(X), k, n_repeats, random_state)
        split_strategy = f'Repeated K-Fold'

    if verbose:
        print(f"\n[{task_type.capitalize()}] {split_strategy}, k={k}, repeats={n_repeats}")
        print("-" * 60)

    metrics = defaultdict(list)
    predictions = []
    repeat_metrics = defaultdict(lambda: defaultdict(list))

    for train_idx, val_idx, repeat_idx, fold_idx in all_splits:
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])

        fold_metrics = _compute_metrics(y[val_idx], y_pred, task_type, pos_label)
        for name, value in fold_metrics.items():
            metrics[name].append(value)
            repeat_metrics[repeat_idx][name].append(value)

        predictions.append({
            'label': f'Repeat {repeat_idx + 1} Fold {fold_idx + 1}/{k}',
            'val_indices': val_idx,
            'y_true': y[val_idx].copy(),
            'y_pred': y_pred.copy(),
            'repeat': repeat_idx,
            'fold': fold_idx,
        })

        if verbose:
            _print_fold_result(
                f'Repeat {repeat_idx + 1}/{n_repeats} Fold {fold_idx + 1}/{k}',
                fold_metrics, task_type
            )

    if verbose:
        _print_summary(metrics, task_type)

        print(f"\n--- Per-Repeat Stability Analysis ---")
        metric_names = {
            'classification': ['accuracy', 'f1'],
            'regression': ['mae', 'r2']
        }[task_type]
        print(f"{'Repeat':<10}", end='')
        for name in metric_names:
            print(f"{name + ' mean':>12}{name + ' std':>12}", end='')
        print()
        print("-" * (10 + 24 * len(metric_names)))
        for rep in range(n_repeats):
            print(f"{'Rep ' + str(rep + 1):<10}", end='')
            for name in metric_names:
                vals = repeat_metrics[rep][name]
                print(f"{np.mean(vals):>12.4f}{np.std(vals, ddof=1):>12.4f}", end='')
            print()

        across_repeats = {}
        for name in metric_names:
            rep_means = [np.mean(repeat_metrics[rep][name]) for rep in range(n_repeats)]
            across_repeats[name] = {
                'mean_of_means': np.mean(rep_means),
                'std_of_means': np.std(rep_means, ddof=1) if n_repeats > 1 else 0.0,
            }
        print(f"\nStability (std of per-repeat means across {n_repeats} repeats):")
        for name in metric_names:
            s = across_repeats[name]
            print(f"  {name}: grand_mean={s['mean_of_means']:.4f}, "
                  f"across_repeat_std={s['std_of_means']:.4f}")

        if return_predictions:
            _print_predictions(predictions, task_type)

    return {
        'metrics': dict(metrics),
        'predictions': predictions,
        'split_strategy': split_strategy,
        'task_type': task_type,
        'k': k,
        'n_repeats': n_repeats,
        'repeat_metrics': {rep: dict(vals) for rep, vals in repeat_metrics.items()},
    }


class SimpleKNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest = np.argsort(dists)[:self.k]
            labels = self.y_train[nearest]
            preds.append(np.bincount(labels.astype(int)).argmax())
        return np.array(preds)


class SimpleKNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest = np.argsort(dists)[:self.k]
            preds.append(np.mean(self.y_train[nearest]))
        return np.array(preds)


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("DEMO 1: Standard K-Fold (Classification with Stratified)")
    print("=" * 60)
    n_samples = 150
    n_features = 4
    X_cls = np.random.randn(n_samples, n_features)
    y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)
    print(f"Class distribution: {np.bincount(y_cls)}")

    model_cls = SimpleKNNClassifier(k=5)
    res1 = kfold_cross_validation(model_cls, X_cls, y_cls, k=5,
                                  random_state=42, return_predictions=True)

    print("\n\n" + "=" * 60)
    print("DEMO 2: Leave-One-Out CV (LOOCV)")
    print("=" * 60)
    n_loo = 30
    X_loo = np.random.randn(n_loo, 2)
    y_loo = (X_loo[:, 0] > 0).astype(int)
    res2 = loocv(SimpleKNNClassifier(k=3), X_loo, y_loo,
                 return_predictions=True)

    print("\n\n" + "=" * 60)
    print("DEMO 3: Leave-P-Out CV (LPOCV, p=2)")
    print("=" * 60)
    n_lpo = 20
    X_lpo = np.random.randn(n_lpo, 2)
    y_lpo = (X_lpo[:, 0] > 0).astype(int)
    res3 = lpocv(SimpleKNNClassifier(k=3), X_lpo, y_lpo, p=2,
                 return_predictions=True)

    print("\n\n" + "=" * 60)
    print("DEMO 4: Repeated K-Fold CV (3 repeats)")
    print("=" * 60)
    X_rep = np.random.randn(100, 3)
    true_coeff = np.array([1.5, -2.0, 0.8])
    y_rep = X_rep @ true_coeff + 0.5 * np.random.randn(100)

    model_reg = SimpleKNNRegressor(k=5)
    res4 = repeated_kfold_cv(model_reg, X_rep, y_rep, k=5, n_repeats=3,
                             random_state=42, return_predictions=True)

    print("\n\n" + "=" * 60)
    print("DEMO 5: Repeated Stratified K-Fold CV (Classification)")
    print("=" * 60)
    X_rcls = np.random.randn(120, 3)
    y_rcls = (X_rcls[:, 0] + X_rcls[:, 1] > 0).astype(int)
    res5 = repeated_kfold_cv(SimpleKNNClassifier(k=5), X_rcls, y_rcls,
                             k=5, n_repeats=3, random_state=42,
                             return_predictions=True)

    print("\n\n" + "=" * 60)
    print("DEMO 6: LOOCV for Regression")
    print("=" * 60)
    n_loo_reg = 25
    X_loo_reg = np.random.randn(n_loo_reg, 2)
    y_loo_reg = 2 * X_loo_reg[:, 0] - X_loo_reg[:, 1] + 0.3 * np.random.randn(n_loo_reg)
    res6 = loocv(SimpleKNNRegressor(k=3), X_loo_reg, y_loo_reg,
                 return_predictions=True)

    print("\n\n" + "=" * 60)
    print("SUMMARY: Accessing prediction details programmatically")
    print("=" * 60)
    print(f"\nDEMO 1 result keys: {list(res1.keys())}")
    print(f"Number of prediction folds: {len(res1['predictions'])}")
    fold0 = res1['predictions'][0]
    print(f"Fold 0 prediction keys: {list(fold0.keys())}")
    print(f"Fold 0 y_true: {fold0['y_true'][:10]}")
    print(f"Fold 0 y_pred: {fold0['y_pred'][:10]}")
    print(f"Fold 0 val_indices: {fold0['val_indices'][:10]}")

    print(f"\nDEMO 4 repeat_metrics: Repeat 0 accuracy={np.mean(res4['repeat_metrics'][0]['r2']):.4f}")
