import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import (
    learning_curve,
    RepeatedStratifiedKFold,
    RepeatedKFold,
)

_CLASSIFICATION_SCORING = {
    "accuracy",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision",
    "recall",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "average_precision",
    "balanced_accuracy",
    "top_k_accuracy",
}


def _is_classification(scoring, y):
    if isinstance(scoring, str) and scoring in _CLASSIFICATION_SCORING:
        return True
    if isinstance(scoring, str) and scoring.startswith("neg_"):
        return False
    unique = np.unique(y)
    if len(unique) <= 20 and len(unique) / len(y) < 0.05:
        return True
    return False


def _build_cv(cv, n_repeats, is_classification, random_state):
    if cv is not None:
        return cv
    if is_classification:
        return RepeatedStratifiedKFold(
            n_splits=5, n_repeats=n_repeats, random_state=random_state
        )
    return RepeatedKFold(
        n_splits=5, n_repeats=n_repeats, random_state=random_state
    )


def _confidence_interval(scores_2d, confidence=0.95):
    n = scores_2d.shape[1]
    mean = np.mean(scores_2d, axis=1)
    se = stats.sem(scores_2d, axis=1)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se
    return mean, mean - margin, mean + margin


def _is_greater(a, b, scoring):
    if isinstance(scoring, str) and scoring.startswith("neg_"):
        return a < b
    return a > b


def _is_higher_score(a, scoring):
    return _is_greater(a, 0, scoring)


def _diagnose_fit(
    train_mean,
    val_mean,
    train_sizes,
    scoring="accuracy",
    baseline_score=None,
    gap_threshold=0.10,
    low_score_threshold=None,
    tail_points=3,
):
    train_mean_arr = np.asarray(train_mean)
    val_mean_arr = np.asarray(val_mean)
    sizes_arr = np.asarray(train_sizes)

    n = len(train_mean_arr)
    tail_start = max(0, n - tail_points)

    final_train = np.mean(train_mean_arr[tail_start:])
    final_val = np.mean(val_mean_arr[tail_start:])
    final_gap = abs(final_train - final_val)

    val_tail = val_mean_arr[tail_start:]
    size_tail = sizes_arr[tail_start:]
    if len(val_tail) >= 2:
        val_slope, _ = np.polyfit(size_tail, val_tail, 1)
        val_slope_per_100 = val_slope * 100
    else:
        val_slope_per_100 = 0.0

    first_half_val = np.mean(val_mean_arr[: n // 2])
    last_half_val = np.mean(val_mean_arr[n // 2 :])
    val_improving = _is_greater(last_half_val, first_half_val, scoring)

    is_error_metric = isinstance(scoring, str) and scoring.startswith("neg_")

    if low_score_threshold is None:
        if is_error_metric:
            low_score_threshold = -0.5
        else:
            low_score_threshold = 0.7

    train_low = not _is_greater(final_train, low_score_threshold, scoring)
    val_low = not _is_greater(final_val, low_score_threshold, scoring)

    gap_large = final_gap > gap_threshold

    if baseline_score is not None:
        val_vs_baseline = _is_greater(final_val, baseline_score, scoring)
    else:
        val_vs_baseline = None

    converged = abs(val_slope_per_100) < 0.005

    diagnosis = "good_fit"
    severity = "low"
    signals = {}

    if train_low and val_low and not gap_large:
        diagnosis = "underfitting"
        severity = "high" if not val_improving else "medium"
    elif gap_large and _is_greater(final_train, final_val, scoring):
        if train_low and val_low:
            diagnosis = "underfitting_with_noise"
            severity = "high"
        else:
            diagnosis = "overfitting"
            if val_improving and not converged:
                severity = "medium"
            else:
                severity = "high"
    elif not val_improving and val_low:
        diagnosis = "underfitting"
        severity = "high"
    elif gap_large and not val_improving:
        diagnosis = "overfitting"
        severity = "high"
    elif val_improving and not converged:
        diagnosis = "more_data_needed"
        severity = "low"

    signals = {
        "final_train_score": float(final_train),
        "final_val_score": float(final_val),
        "final_gap": float(final_gap),
        "val_slope_per_100_samples": float(val_slope_per_100),
        "val_is_improving": bool(val_improving),
        "converged": bool(converged),
        "train_score_low": bool(train_low),
        "val_score_low": bool(val_low),
        "gap_is_large": bool(gap_large),
        "baseline_met": val_vs_baseline,
    }

    return {
        "diagnosis": diagnosis,
        "severity": severity,
        "signals": signals,
        "scoring": scoring,
        "is_error_metric": is_error_metric,
    }


def recommend_actions(diagnosis_result, is_classification=True):
    diag = diagnosis_result["diagnosis"]
    severity = diagnosis_result["severity"]
    signals = diagnosis_result["signals"]
    is_err = diagnosis_result["is_error_metric"]

    label = "error" if is_err else "score"
    gap_label = f"train-val {label} gap"

    recommendations = []

    if diag == "underfitting":
        recommendations.append(
            {
                "priority": 1,
                "action": "Increase model complexity",
                "details": "Both training and validation scores are low with small gap. "
                "Model is too simple to capture patterns. Try: deeper tree/layers, "
                "more hidden units, reduce regularization strength, use a more powerful model family.",
            }
        )
        recommendations.append(
            {
                "priority": 2,
                "action": "Add more informative features",
                "details": "Feature engineering: create interaction terms, polynomial features, "
                "domain-specific features. Consider dimensionality reduction only if features are noisy.",
            }
        )
        recommendations.append(
            {
                "priority": 3,
                "action": "Tune hyperparameters",
                "details": "Perform systematic hyperparameter search (GridSearchCV, Optuna) "
                "focused on model capacity parameters.",
            }
        )
    elif diag == "overfitting":
        if signals["val_is_improving"] and not signals["converged"]:
            recommendations.append(
                {
                    "priority": 1,
                    "action": "Add more training data",
                    "details": f"Validation {label} is still improving with more samples. "
                    "Collecting more data is likely to reduce overfitting and improve generalization.",
                }
            )
        recommendations.append(
            {
                "priority": 2,
            "action": "Increase regularization",
                "details": f"Large {gap_label} indicates model memorizes noise. Try: "
                "stronger L1/L2 regularization, add dropout layers (NN), increase min_samples_leaf (tree), "
                "reduce max_depth, use early stopping.",
            }
        )
        recommendations.append(
            {
                "priority": 3,
                "action": "Reduce model complexity",
                "details": "Simplify the model architecture: fewer layers/trees, smaller hidden dimensions, "
                "feature selection to remove noisy features.",
            }
        )
        recommendations.append(
            {
                "priority": 4,
                "action": "Data augmentation / cleaning",
                "details": "Add augmentation to increase effective dataset size, remove noisy/outlier samples, "
                "improve data quality.",
            }
        )
    elif diag == "underfitting_with_noise":
        recommendations.append(
            {
                "priority": 1,
                "action": "Increase model complexity and add regularization",
                "details": "Model is both underfitting (low scores) and showing signs of overfitting (large gap). "
                "Increase capacity while adding regularization to balance bias-variance tradeoff.",
            }
        )
        recommendations.append(
            {
                "priority": 2,
                "action": "Improve data quality",
                "details": "Examine dataset for label noise, outliers, or inconsistent annotations. "
                "Cleaning data can help both bias and variance simultaneously.",
            }
        )
    elif diag == "more_data_needed":
        recommendations.append(
            {
                "priority": 1,
                "action": "Collect more training data",
                "details": f"Validation {label} has not plateaued. Adding more samples "
                "will likely improve performance. Consider data augmentation if collection is expensive.",
            }
        )
        recommendations.append(
            {
                "priority": 2,
                "action": "Experiment with learning rate scheduling",
                "details": "If using iterative optimizers, try learning rate decay or cosine annealing "
                "to help convergence with more data.",
            }
        )
    elif diag == "good_fit":
        recommendations.append(
            {
                "priority": 1,
                "action": "Maintain current setup",
                "details": f"Training and validation {label}s are both good with small gap. "
                "Model is well-balanced.",
            }
        )
        recommendations.append(
            {
                "priority": 2,
                "action": "Marginal improvements",
                "details": "For further gains: try ensemble methods, fine-grained hyperparameter tuning, "
                "or experiment with small architecture tweaks.",
            }
        )
        if signals["val_is_improving"] and not signals["converged"]:
            recommendations.append(
                {
                    "priority": 3,
                    "action": "Add more data for further gains",
                    "details": "Validation score still shows slight improvement with more samples. "
                    "Additional data may yield marginal improvements.",
                }
            )

    severity_note = {
        "low": "Monitoring suggested. Current trajectory is healthy.",
        "medium": "Take action in next iteration. Performance may plateau suboptimally.",
        "high": "Address immediately. Model is not performing acceptably.",
    }

    return {
        "diagnosis": diag,
        "severity": severity,
        "severity_note": severity_note.get(severity, ""),
        "recommendations": recommendations,
        "signals_summary": {
            "Final train score": f"{signals['final_train_score']:.4f}",
            "Final val score": f"{signals['final_val_score']:.4f}",
            "Train-val gap": f"{signals['final_gap']:.4f}",
            "Val slope / 100 samples": f"{signals['val_slope_per_100_samples']:.5f}",
            "Val improving": "Yes" if signals["val_is_improving"] else "No",
            "Converged": "Yes" if signals["converged"] else "No",
        },
    }


def plot_learning_curve(
    estimator,
    X,
    y,
    cv=None,
    n_repeats=10,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    confidence=0.95,
    title="Learning Curve",
    figsize=(8, 6),
    random_state=42,
    diagnose=True,
    baseline_score=None,
    gap_threshold=0.10,
    low_score_threshold=None,
    tail_points=3,
):
    is_clf = _is_classification(scoring, y)
    cv = _build_cv(cv, n_repeats, is_clf, random_state)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1
    )

    n_folds = train_scores.shape[1]

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    _, train_ci_lo, train_ci_hi = _confidence_interval(train_scores, confidence)
    _, val_ci_lo, val_ci_hi = _confidence_interval(val_scores, confidence)

    diagnosis_raw = None
    diagnosis_report = None
    if diagnose:
        diagnosis_raw = _diagnose_fit(
            train_mean,
            val_mean,
            train_sizes_abs,
            scoring=scoring,
            baseline_score=baseline_score,
            gap_threshold=gap_threshold,
            low_score_threshold=low_score_threshold,
            tail_points=tail_points,
        )
        diagnosis_report = recommend_actions(diagnosis_raw, is_classification=is_clf)

    fig, ax = plt.subplots(figsize=figsize)
    subtitle = f"({int(confidence * 100)}% CI, {n_folds} folds"
    if diagnosis_report is not None:
        diag_label = diagnosis_report["diagnosis"].replace("_", " ").title()
        sev_label = diagnosis_report["severity"].title()
        subtitle += f" — {diag_label}, {sev_label})"
    else:
        subtitle += ")"
    ax.set_title(f"{title}  {subtitle}")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel(scoring)
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.fill_between(
        train_sizes_abs,
        train_ci_lo,
        train_ci_hi,
        alpha=0.18,
        color="steelblue",
        label=f"Train {int(confidence * 100)}% CI",
    )
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.08,
        color="steelblue",
    )

    ax.fill_between(
        train_sizes_abs,
        val_ci_lo,
        val_ci_hi,
        alpha=0.18,
        color="darkorange",
        label=f"Val {int(confidence * 100)}% CI",
    )
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.08,
        color="darkorange",
    )

    ax.plot(
        train_sizes_abs, train_mean, "o-", color="steelblue", label="Training Score"
    )
    ax.plot(
        train_sizes_abs, val_mean, "o-", color="darkorange", label="Validation Score"
    )

    ax.legend(loc="best")
    fig.tight_layout()

    result = {
        "train_sizes": train_sizes_abs.tolist(),
        "n_folds": int(n_folds),
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "train_ci_lo": train_ci_lo.tolist(),
        "train_ci_hi": train_ci_hi.tolist(),
        "val_mean": val_mean.tolist(),
        "val_std": val_std.tolist(),
        "val_ci_lo": val_ci_lo.tolist(),
        "val_ci_hi": val_ci_hi.tolist(),
        "diagnosis_raw": diagnosis_raw,
        "diagnosis_report": diagnosis_report,
        "is_classification": is_clf,
        "fig": fig,
    }
    return result
