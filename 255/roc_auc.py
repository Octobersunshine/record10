import warnings
from itertools import combinations

import numpy as np


def _clean_y_prob(y_prob):
    y_prob = np.asarray(y_prob, dtype=np.float64)
    nan_mask = np.isnan(y_prob)
    inf_mask = np.isinf(y_prob)

    if not nan_mask.any() and not inf_mask.any():
        return y_prob

    valid_mask = ~(nan_mask | inf_mask)
    y_prob_clean = y_prob.copy()

    if valid_mask.any():
        median_val = np.median(y_prob[valid_mask])
        finite_min = np.min(y_prob[valid_mask])
        finite_max = np.max(y_prob[valid_mask])
    else:
        median_val = 0.0
        finite_min = 0.0
        finite_max = 1.0

    if nan_mask.any():
        y_prob_clean[nan_mask] = median_val
        warnings.warn(
            f"Found {nan_mask.sum()} NaN value(s) in y_prob, "
            f"replaced with median {median_val:.4f}"
        )

    if inf_mask.any():
        pos_inf_mask = y_prob == np.inf
        neg_inf_mask = y_prob == -np.inf
        if pos_inf_mask.any():
            y_prob_clean[pos_inf_mask] = finite_max
        if neg_inf_mask.any():
            y_prob_clean[neg_inf_mask] = finite_min
        warnings.warn(
            f"Found {inf_mask.sum()} Inf value(s) in y_prob, "
            f"clamped to finite range [{finite_min:.4f}, {finite_max:.4f}]"
        )

    return y_prob_clean


def _sample_thresholds(y_prob, n_thresholds):
    if n_thresholds is None:
        return None
    n_thresholds = int(n_thresholds)
    if n_thresholds <= 1:
        raise ValueError("n_thresholds must be >= 2 or None")
    thresholds = np.linspace(1.0, 0.0, n_thresholds)
    return thresholds


def _compute_roc_at_thresholds(y_true, y_prob, thresholds):
    tpr_list = [0.0]
    fpr_list = [0.0]

    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    if total_pos == 0 or total_neg == 0:
        if total_pos == 0 and total_neg == 0:
            return np.array([0.0]), np.array([0.0])
        tpr_val = 1.0 if total_pos > 0 else 0.0
        return np.array([0.0, 1.0]), np.array([0.0, tpr_val])

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    sort_idx = np.lexsort((tpr_arr, fpr_arr))
    fpr_arr = fpr_arr[sort_idx]
    tpr_arr = tpr_arr[sort_idx]

    return fpr_arr, tpr_arr


def roc_curve_and_auc(y_true, y_prob, n_thresholds=None):
    y_true = np.asarray(y_true)
    y_prob = _clean_y_prob(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_prob has {len(y_prob)} elements"
        )

    if n_thresholds is not None:
        thresholds = _sample_thresholds(y_prob, n_thresholds)
        fpr, tpr = _compute_roc_at_thresholds(y_true, y_prob, thresholds)
    else:
        desc_score_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        y_prob_sorted = y_prob[desc_score_indices]

        distinct_value_indices = np.where(np.diff(y_prob_sorted))[0]
        threshold_indices = np.append(distinct_value_indices, len(y_prob_sorted) - 1)

        tps = np.cumsum(y_true_sorted)[threshold_indices]
        fps = threshold_indices + 1 - tps

        tpr = np.concatenate(([0.0], tps / tps[-1]))
        fpr = np.concatenate(([0.0], fps / fps[-1]))

    auc = np.trapezoid(tpr, fpr)

    return fpr, tpr, auc


def precision_recall_curve_and_auc(y_true, y_prob, n_thresholds=None):
    y_true = np.asarray(y_true)
    y_prob = _clean_y_prob(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_prob has {len(y_prob)} elements"
        )

    total_pos = np.sum(y_true == 1)
    if total_pos == 0:
        warnings.warn("No positive samples in y_true, PR-AUC is undefined, returning 0.0")
        return np.array([0.0]), np.array([0.0]), 0.0

    if n_thresholds is not None:
        thresholds = _sample_thresholds(y_prob, n_thresholds)
        precision_list = [1.0]
        recall_list = [0.0]
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / total_pos
            precision_list.append(prec)
            recall_list.append(rec)
        precision_arr = np.array(precision_list)
        recall_arr = np.array(recall_list)
    else:
        desc_score_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[desc_score_indices]

        tps = np.cumsum(y_true_sorted)
        fps = np.arange(1, len(y_true_sorted) + 1) - tps

        precision_arr = tps / (tps + fps)
        recall_arr = tps / total_pos

        precision_arr = np.concatenate(([1.0], precision_arr))
        recall_arr = np.concatenate(([0.0], recall_arr))

    sort_idx = np.argsort(recall_arr)
    recall_sorted = recall_arr[sort_idx]
    precision_sorted = precision_arr[sort_idx]

    auc = np.trapezoid(precision_sorted, recall_sorted)

    return recall_sorted, precision_sorted, auc


def _clean_y_prob_matrix(y_prob_matrix):
    y_prob_matrix = np.asarray(y_prob_matrix, dtype=np.float64)
    for col in range(y_prob_matrix.shape[1]):
        y_prob_matrix[:, col] = _clean_y_prob(y_prob_matrix[:, col])
    return y_prob_matrix


def roc_curve_ovr(y_true, y_prob_matrix, n_thresholds=None):
    y_true = np.asarray(y_true)
    y_prob_matrix = _clean_y_prob_matrix(y_prob_matrix)

    classes = np.unique(y_true)
    n_classes = len(classes)

    if y_prob_matrix.ndim != 2:
        raise ValueError("y_prob_matrix must be 2-dimensional (n_samples, n_classes)")
    if y_prob_matrix.shape[0] != len(y_true):
        raise ValueError(
            f"Row mismatch: y_true has {len(y_true)} samples, "
            f"y_prob_matrix has {y_prob_matrix.shape[0]} rows"
        )
    if y_prob_matrix.shape[1] != n_classes:
        warnings.warn(
            f"y_prob_matrix has {y_prob_matrix.shape[1]} columns but "
            f"{n_classes} unique classes found; using first {n_classes} columns"
        )

    results = {}
    for i, cls in enumerate(classes):
        y_true_bin = (y_true == cls).astype(int)
        y_prob_cls = y_prob_matrix[:, i]
        fpr, tpr, auc = roc_curve_and_auc(y_true_bin, y_prob_cls, n_thresholds=n_thresholds)
        results[cls] = {"fpr": fpr, "tpr": tpr, "auc": auc}

    all_fpr = np.unique(np.concatenate([results[cls]["fpr"] for cls in classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for cls in classes:
        mean_tpr += np.interp(all_fpr, results[cls]["fpr"], results[cls]["tpr"])
    mean_tpr /= n_classes
    macro_auc = np.trapezoid(mean_tpr, all_fpr)

    return results, all_fpr, mean_tpr, macro_auc


def roc_curve_ovo(y_true, y_prob_matrix, n_thresholds=None):
    y_true = np.asarray(y_true)
    y_prob_matrix = _clean_y_prob_matrix(y_prob_matrix)

    classes = np.unique(y_true)
    n_classes = len(classes)

    if y_prob_matrix.ndim != 2:
        raise ValueError("y_prob_matrix must be 2-dimensional (n_samples, n_classes)")
    if y_prob_matrix.shape[0] != len(y_true):
        raise ValueError(
            f"Row mismatch: y_true has {len(y_true)} samples, "
            f"y_prob_matrix has {y_prob_matrix.shape[0]} rows"
        )

    pair_results = {}
    pair_aucs = []

    for ci, cj in combinations(classes, 2):
        mask = (y_true == ci) | (y_true == cj)
        y_true_pair = y_true[mask]
        y_prob_pair = y_prob_matrix[mask]

        y_true_bin_ci = (y_true_pair == ci).astype(int)

        prob_ci = y_prob_pair[:, ci] / (y_prob_pair[:, ci] + y_prob_pair[:, cj] + 1e-15)

        fpr, tpr, auc = roc_curve_and_auc(y_true_bin_ci, prob_ci, n_thresholds=n_thresholds)
        pair_results[(ci, cj)] = {"fpr": fpr, "tpr": tpr, "auc": auc}
        pair_aucs.append(auc)

    macro_auc = np.mean(pair_aucs) if pair_aucs else 0.0

    class_aucs = {cls: [] for cls in classes}
    for (ci, cj), res in pair_results.items():
        class_aucs[ci].append(res["auc"])
        class_aucs[cj].append(res["auc"])

    weighted_auc = 0.0
    total = 0
    for cls in classes:
        n_cls = np.sum(y_true == cls)
        avg = np.mean(class_aucs[cls]) if class_aucs[cls] else 0.0
        weighted_auc += n_cls * avg
        total += n_cls
    weighted_auc = weighted_auc / total if total > 0 else 0.0

    return pair_results, macro_auc, weighted_auc


if __name__ == "__main__":
    print("=" * 60)
    print("1. Binary ROC (with threshold sampling)")
    print("=" * 60)
    y_true_bin = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    y_prob_bin = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.3, 0.6, 0.75, 0.45]

    fpr, tpr, auc = roc_curve_and_auc(y_true_bin, y_prob_bin)
    print(f"Full thresholds -> AUC: {auc:.4f}, points: {len(fpr)}")

    fpr_s, tpr_s, auc_s = roc_curve_and_auc(y_true_bin, y_prob_bin, n_thresholds=5)
    print(f"n_thresholds=5 -> AUC: {auc_s:.4f}, points: {len(fpr_s)}")

    print("\n" + "=" * 60)
    print("2. PR Curve (imbalanced data)")
    print("=" * 60)
    np.random.seed(42)
    n_total = 1000
    n_pos = 50
    y_true_imb = np.array([1] * n_pos + [0] * (n_total - n_pos))
    y_prob_imb = np.concatenate([
        np.random.uniform(0.5, 1.0, n_pos),
        np.random.uniform(0.0, 0.5, n_total - n_pos),
    ])

    recall, precision, pr_auc = precision_recall_curve_and_auc(y_true_imb, y_prob_imb)
    print(f"Full thresholds -> PR-AUC: {pr_auc:.4f}, points: {len(recall)}")

    recall_s, precision_s, pr_auc_s = precision_recall_curve_and_auc(y_true_imb, y_prob_imb, n_thresholds=10)
    print(f"n_thresholds=10 -> PR-AUC: {pr_auc_s:.4f}, points: {len(recall_s)}")

    print("\n" + "=" * 60)
    print("3. Multi-class ROC - OvR")
    print("=" * 60)
    np.random.seed(42)
    n_samples = 150
    y_true_multi = np.array([0] * 50 + [1] * 50 + [2] * 50)
    y_prob_multi = np.zeros((n_samples, 3))
    for i in range(n_samples):
        cls = y_true_multi[i]
        y_prob_multi[i, cls] = np.random.uniform(0.6, 0.95)
        other = [c for c in range(3) if c != cls]
        for c in other:
            y_prob_multi[i, c] = np.random.uniform(0.02, 0.35)
    row_sums = y_prob_multi.sum(axis=1, keepdims=True)
    y_prob_multi = y_prob_multi / row_sums

    ovr_results, macro_fpr, macro_tpr, macro_auc = roc_curve_ovr(y_true_multi, y_prob_multi)
    for cls, res in ovr_results.items():
        print(f"  Class {cls}: AUC = {res['auc']:.4f}")
    print(f"  Macro-average AUC = {macro_auc:.4f}")

    ovr_results_s, _, _, macro_auc_s = roc_curve_ovr(y_true_multi, y_prob_multi, n_thresholds=20)
    print(f"  Macro-average AUC (n_thresholds=20) = {macro_auc_s:.4f}")

    print("\n" + "=" * 60)
    print("4. Multi-class ROC - OvO")
    print("=" * 60)
    pair_results, ovo_macro_auc, ovo_weighted_auc = roc_curve_ovo(y_true_multi, y_prob_multi)
    for pair, res in pair_results.items():
        print(f"  Pair {pair}: AUC = {res['auc']:.4f}")
    print(f"  OvO Macro-average AUC    = {ovo_macro_auc:.4f}")
    print(f"  OvO Weighted-average AUC = {ovo_weighted_auc:.4f}")
