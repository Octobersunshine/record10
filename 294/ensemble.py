import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class StackingResult:
    meta_model_coef_: Optional[np.ndarray] = None
    meta_model_intercept_: Optional[float] = None
    cv_scores_: Dict[str, List[float]] = field(default_factory=dict)
    oof_predictions_: Optional[np.ndarray] = None
    base_model_scores_: Dict[str, float] = field(default_factory=dict)
    ensemble_score_: Optional[float] = None
    feature_importance_: Optional[Dict[str, float]] = None
    n_folds_: Optional[int] = None
    random_state_: Optional[int] = None


def _kfold_split(n_samples: int, n_folds: int = 5,
                 shuffle: bool = True,
                 random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    fold_sizes = (n_samples // n_folds) * np.ones(n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, test_idx))
        current = end
    return folds


def _scorer_reg(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def _scorer_cls(y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    scores = {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1}

    if y_proba is not None and y_proba.ndim >= 1:
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            scores["AUC"] = roc_auc_score(y_true, y_proba)
        except Exception:
            pass

    return scores


class VotingClassifier:
    def __init__(self, mode: str = "hard", weights: Optional[List[float]] = None,
                 tie_break: str = "first"):
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        if tie_break not in ("first", "confidence", "prob_sum", "positive"):
            raise ValueError("tie_break must be one of: 'first', 'confidence', 'prob_sum', 'positive'")
        self.mode = mode
        self.weights = weights
        self.tie_break = tie_break

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]],
                probas: Optional[np.ndarray] = None) -> np.ndarray:
        predictions = np.array(predictions)
        if predictions.ndim != 2:
            raise ValueError("predictions shape should be (n_models, n_samples)")

        if self.mode == "hard":
            return self._hard_vote(predictions, probas)
        else:
            return self._soft_vote(predictions)

    def _hard_vote(self, predictions: np.ndarray,
                   probas: Optional[np.ndarray]) -> np.ndarray:
        n_samples = predictions.shape[1]
        result = np.empty(n_samples, dtype=predictions.dtype)

        for i in range(n_samples):
            votes = predictions[:, i]
            if self.weights is not None:
                w = np.array(self.weights)
                class_counts = {}
                for vote, weight in zip(votes, w):
                    class_counts[vote] = class_counts.get(vote, 0.0) + weight
            else:
                vals, counts = np.unique(votes, return_counts=True)
                class_counts = dict(zip(vals, counts.astype(float)))

            max_count = max(class_counts.values())
            tied_classes = [cls for cls, cnt in class_counts.items()
                            if abs(cnt - max_count) < 1e-12]

            if len(tied_classes) == 1:
                result[i] = tied_classes[0]
            else:
                result[i] = self._break_tie(tied_classes, i, predictions, probas)

        return result

    def _break_tie(self, tied_classes: list, sample_idx: int,
                   predictions: np.ndarray,
                   probas: Optional[np.ndarray]) -> int:
        if self.tie_break == "first":
            return min(tied_classes)

        if self.tie_break == "positive":
            return 1 if 1 in tied_classes else min(tied_classes)

        if probas is None:
            return min(tied_classes)

        probas_2d = np.atleast_2d(probas)
        if probas_2d.shape == predictions.shape:
            class_probs = {}
            for cls in tied_classes:
                mask = predictions[:, sample_idx] == cls
                if self.weights is not None:
                    w = np.array(self.weights)
                    class_probs[cls] = np.sum(probas_2d[:, sample_idx][mask] * w[mask])
                else:
                    class_probs[cls] = np.sum(probas_2d[:, sample_idx][mask])
        else:
            class_probs = {}
            for cls in tied_classes:
                if self.weights is not None:
                    w = np.array(self.weights).reshape(-1, 1)
                    weighted_p = probas_2d[:, sample_idx] * w
                    class_probs[cls] = np.sum(weighted_p[:, cls])
                else:
                    class_probs[cls] = np.sum(probas_2d[:, sample_idx, cls])

        if self.tie_break == "confidence":
            return max(class_probs, key=class_probs.get)
        elif self.tie_break == "prob_sum":
            return max(class_probs, key=class_probs.get)
        else:
            return min(tied_classes)

    def _soft_vote(self, probas: np.ndarray) -> np.ndarray:
        if self.weights is not None:
            w = np.array(self.weights)
            avg_proba = np.average(probas, axis=0, weights=w)
        else:
            avg_proba = np.mean(probas, axis=0)

        result = np.empty(len(avg_proba), dtype=int)
        for i, p in enumerate(avg_proba):
            if abs(p - 0.5) < 1e-12:
                if self.tie_break == "positive":
                    result[i] = 1
                elif self.tie_break == "confidence" or self.tie_break == "prob_sum":
                    prob_sum = np.sum(probas[:, i])
                    if prob_sum >= probas.shape[0] * 0.5:
                        result[i] = 1
                    else:
                        result[i] = 0
                else:
                    result[i] = 0
            else:
                result[i] = 1 if p > 0.5 else 0
        return result

    def get_decision_rules(self) -> dict:
        rules = {
            "mode": self.mode,
            "weights": self.weights,
            "tie_break": self.tie_break,
            "description": self._get_rule_description(),
            "tie_break_options": {
                "first": "出现平票时选择类别编号较小的类别（默认，确定性但偏向保守）",
                "positive": "出现平票时优先选择正类（1），适合偏好奇的场景",
                "confidence": "出现平票时选择预测置信度最高的类别（需要传入probas）",
                "prob_sum": "出现平票时选择所有模型预测概率之和最高的类别（需要传入probas）",
            }
        }
        return rules

    def _get_rule_description(self) -> str:
        if self.mode == "hard":
            if self.weights is not None:
                base = "加权硬投票：每个模型按权重对预测类别投票"
            else:
                base = "简单硬投票：每个模型对预测类别投一票"
            if self.tie_break == "first":
                tie = "平票时选择类别编号较小者"
            elif self.tie_break == "positive":
                tie = "平票时优先选择正类(1)"
            elif self.tie_break == "confidence":
                tie = "平票时选择置信度最高的类别（需probas）"
            else:
                tie = "平票时选择概率和最大的类别（需probas）"
            return f"{base}；{tie}"
        else:
            if self.weights is not None:
                base = "加权软投票：对各模型预测概率做加权平均"
            else:
                base = "简单软投票：对各模型预测概率取算术平均"
            if self.tie_break == "first":
                tie = "均值=0.5时判为负类(0)"
            elif self.tie_break == "positive":
                tie = "均值=0.5时判为正类(1)"
            else:
                tie = "均值=0.5时按概率总和方向决定"
            return f"{base}，以0.5为阈值二分类；{tie}"


class VotingClassifierMulticlass:
    def __init__(self, n_classes: int, mode: str = "hard",
                 weights: Optional[List[float]] = None,
                 tie_break: str = "first"):
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        if tie_break not in ("first", "confidence", "prob_sum"):
            raise ValueError("tie_break must be one of: 'first', 'confidence', 'prob_sum'")
        self.mode = mode
        self.weights = weights
        self.n_classes = n_classes
        self.tie_break = tie_break

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]],
                probas: Optional[np.ndarray] = None) -> np.ndarray:
        predictions = np.array(predictions)
        if predictions.ndim != 2:
            raise ValueError("predictions shape should be (n_models, n_samples)")

        if self.mode == "hard":
            return self._hard_vote(predictions, probas)
        else:
            raise ValueError("For soft vote with multiclass, use predict_from_proba with probability matrix")

    def _hard_vote(self, predictions: np.ndarray,
                   probas: Optional[np.ndarray]) -> np.ndarray:
        n_samples = predictions.shape[1]
        result = np.empty(n_samples, dtype=predictions.dtype)

        for i in range(n_samples):
            votes = predictions[:, i]
            if self.weights is not None:
                w = np.array(self.weights)
                class_counts = {}
                for vote, weight in zip(votes, w):
                    class_counts[vote] = class_counts.get(vote, 0.0) + weight
            else:
                vals, counts = np.unique(votes, return_counts=True)
                class_counts = dict(zip(vals, counts.astype(float)))

            max_count = max(class_counts.values())
            tied_classes = [cls for cls, cnt in class_counts.items()
                            if abs(cnt - max_count) < 1e-12]

            if len(tied_classes) == 1:
                result[i] = tied_classes[0]
            else:
                result[i] = self._break_tie(tied_classes, i, predictions, probas)

        return result

    def _break_tie(self, tied_classes: list, sample_idx: int,
                   predictions: np.ndarray,
                   probas: Optional[np.ndarray]) -> int:
        if self.tie_break == "first":
            return min(tied_classes)

        if probas is None:
            return min(tied_classes)

        probas_3d = np.array(probas)
        if probas_3d.ndim != 3:
            return min(tied_classes)

        class_probs = {}
        for cls in tied_classes:
            if self.weights is not None:
                w = np.array(self.weights).reshape(-1, 1)
                weighted_p = probas_3d[:, sample_idx] * w
                class_probs[cls] = np.sum(weighted_p[:, cls])
            else:
                class_probs[cls] = np.sum(probas_3d[:, sample_idx, cls])

        return max(class_probs, key=class_probs.get)

    def predict_from_proba(self, probas: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        probas = np.array(probas)
        if probas.ndim != 3:
            raise ValueError("probas shape should be (n_models, n_samples, n_classes)")

        if self.weights is not None:
            w = np.array(self.weights)
            avg_proba = np.zeros((probas.shape[1], probas.shape[2]))
            w_sum = np.sum(w)
            for i in range(probas.shape[0]):
                avg_proba += w[i] * probas[i]
            avg_proba /= w_sum
        else:
            avg_proba = np.mean(probas, axis=0)

        result = np.empty(probas.shape[1], dtype=int)
        for i in range(probas.shape[1]):
            probs = avg_proba[i]
            max_p = np.max(probs)
            tied = np.where(np.abs(probs - max_p) < 1e-12)[0]
            if len(tied) == 1:
                result[i] = tied[0]
            else:
                if self.tie_break == "first":
                    result[i] = tied[0]
                else:
                    tied_probs = {}
                    for cls in tied:
                        if self.weights is not None:
                            w = np.array(self.weights)
                            tied_probs[cls] = np.sum(probas[:, i, cls] * w)
                        else:
                            tied_probs[cls] = np.sum(probas[:, i, cls])
                    result[i] = max(tied_probs, key=tied_probs.get)
        return result

    def get_decision_rules(self) -> dict:
        rules = {
            "mode": self.mode,
            "n_classes": self.n_classes,
            "weights": self.weights,
            "tie_break": self.tie_break,
            "description": self._get_rule_description(),
            "tie_break_options": {
                "first": "出现平票时选择类别编号较小的类别（默认，确定性）",
                "confidence": "出现平票时选择平均置信度最高的类别（软投票）或概率和最高（硬投票，需probas）",
                "prob_sum": "出现平票时选择所有模型预测概率之和最高的类别（需要传入probas）",
            }
        }
        return rules

    def _get_rule_description(self) -> str:
        if self.mode == "hard":
            if self.weights is not None:
                base = f"{self.n_classes}类加权硬投票：每个模型按权重对预测类别投票"
            else:
                base = f"{self.n_classes}类简单硬投票：每个模型对预测类别投一票"
            if self.tie_break == "first":
                tie = "平票时选择类别编号较小者"
            else:
                tie = "平票时选择概率和最大的类别（需probas）"
            return f"{base}；{tie}"
        else:
            if self.weights is not None:
                base = f"{self.n_classes}类加权软投票：对各模型预测概率做加权平均"
            else:
                base = f"{self.n_classes}类简单软投票：对各模型预测概率取算术平均"
            if self.tie_break == "first":
                tie = "最大概率平票时选类别编号较小者"
            else:
                tie = "最大概率平票时选概率总和最大者"
            return f"{base}，取概率最大类别；{tie}"


class AveragingRegressor:
    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        predictions = np.array(predictions)
        if predictions.ndim != 2:
            raise ValueError("predictions shape should be (n_models, n_samples)")

        if self.weights is not None:
            w = np.array(self.weights)
            return np.average(predictions, axis=0, weights=w)
        else:
            return np.mean(predictions, axis=0)


class StackingRegressor:
    def __init__(self, meta_model=None, n_folds: int = 5,
                 shuffle: bool = True, random_state: Optional[int] = None):
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False
        self.result_: Optional[StackingResult] = None

    def fit(self, X: Optional[np.ndarray], y: np.ndarray,
            base_models: Optional[List[Any]] = None,
            base_model_predict_func: Optional[Callable] = None,
            base_predictions: Optional[np.ndarray] = None,
            X_eval: Optional[np.ndarray] = None,
            y_eval: Optional[np.ndarray] = None) -> "StackingRegressor":
        if self.meta_model is None:
            raise ValueError("meta_model must be provided")

        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y) if np.std(y) > 0 else 1.0

        if base_predictions is not None:
            return self._fit_from_predictions(base_predictions, y, X_eval, y_eval)
        elif base_models is not None and X is not None:
            return self._fit_with_cv(X, y, base_models, base_model_predict_func,
                                     X_eval, y_eval)
        else:
            raise ValueError("Either base_predictions or (base_models and X) must be provided")

    def _fit_from_predictions(self, base_predictions: np.ndarray,
                              y: np.ndarray,
                              X_eval: Optional[np.ndarray],
                              y_eval: Optional[np.ndarray]) -> "StackingRegressor":
        base_predictions = np.array(base_predictions)
        if base_predictions.ndim != 2:
            raise ValueError("base_predictions shape should be (n_models, n_samples)")

        self.base_models_ = None
        self._fit_meta_model(base_predictions.T, y)

        self.result_ = StackingResult(
            n_folds_=None,
            random_state_=self.random_state,
        )
        self._extract_meta_model_params()
        self._fitted = True
        self._compute_scores(base_predictions, y, X_eval, y_eval)

        return self

    def _fit_with_cv(self, X: np.ndarray, y: np.ndarray,
                     base_models: List[Any],
                     base_model_predict_func: Optional[Callable],
                     X_eval: Optional[np.ndarray],
                     y_eval: Optional[np.ndarray]) -> "StackingRegressor":
        n_samples = len(y)
        n_models = len(base_models)

        folds = _kfold_split(n_samples, self.n_folds, self.shuffle, self.random_state)

        oof_preds = np.zeros((n_models, n_samples))
        self.base_models_ = []
        cv_scores = {f"Model_{i}": [] for i in range(n_models)}

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            fold_models = []
            for i, model in enumerate(base_models):
                model_clone = self._clone_model(model)
                model_clone.fit(X_train, y_train)

                if base_model_predict_func is not None:
                    val_pred = base_model_predict_func(model_clone, X_val)
                else:
                    val_pred = model_clone.predict(X_val)

                oof_preds[i, val_idx] = val_pred
                fold_models.append(model_clone)

                fold_score = _scorer_reg(y_val, val_pred)["R2"]
                cv_scores[f"Model_{i}"].append(fold_score)

            print(f"  Fold {fold_idx + 1}/{self.n_folds} complete")

        for i, model in enumerate(base_models):
            final_model = self._clone_model(model)
            final_model.fit(X, y)
            self.base_models_.append(final_model)

        self._fit_meta_model(oof_preds.T, y)

        self.result_ = StackingResult(
            cv_scores_=cv_scores,
            oof_predictions_=oof_preds,
            n_folds_=self.n_folds,
            random_state_=self.random_state,
        )
        self._extract_meta_model_params()
        self._fitted = True
        self._compute_scores(oof_preds, y, X_eval, y_eval)

        return self

    def _clone_model(self, model: Any) -> Any:
        try:
            from sklearn.base import clone
            return clone(model)
        except Exception:
            import copy
            return copy.deepcopy(model)

    def _fit_meta_model(self, meta_features: np.ndarray, y: np.ndarray) -> None:
        self.meta_model.fit(meta_features, y)

    def _extract_meta_model_params(self) -> None:
        if self.result_ is None:
            return

        if hasattr(self.meta_model, "coef_"):
            self.result_.meta_model_coef_ = np.array(self.meta_model.coef_).flatten()
            if hasattr(self.meta_model, "intercept_"):
                intercept = self.meta_model.intercept_
                self.result_.meta_model_intercept_ = float(intercept) if np.isscalar(intercept) else float(intercept[0])

            if self.result_.meta_model_coef_ is not None:
                n_models = len(self.result_.meta_model_coef_)
                self.result_.feature_importance_ = {
                    f"BaseModel_{i}": float(self.result_.meta_model_coef_[i])
                    for i in range(n_models)
                }

    def _compute_scores(self, base_preds: np.ndarray, y: np.ndarray,
                        X_eval: Optional[np.ndarray],
                        y_eval: Optional[np.ndarray]) -> None:
        if self.result_ is None:
            return

        eval_preds = self.predict(X_eval) if (X_eval is not None and self.base_models_ is not None) else None
        y_true_eval = y_eval if y_eval is not None else y
        base_preds_eval = base_preds if eval_preds is None else self._predict_base(X_eval)

        stack_pred = self.predict(X_eval) if (X_eval is not None and self.base_models_ is not None) \
            else self.meta_model.predict(base_preds.T)

        for i in range(base_preds_eval.shape[0]):
            scores = _scorer_reg(y_true_eval, base_preds_eval[i])
            self.result_.base_model_scores_[f"BaseModel_{i}"] = float(scores["R2"])

        stack_scores = _scorer_reg(y_true_eval, stack_pred)
        self.result_.ensemble_score_ = float(stack_scores["R2"])
        self.result_.ensemble_metrics_ = stack_scores

    def _predict_base(self, X: np.ndarray) -> np.ndarray:
        if self.base_models_ is None:
            raise ValueError("No base models fitted; use fit with base_models parameter")
        preds = []
        for model in self.base_models_:
            preds.append(model.predict(X))
        return np.array(preds)

    def predict(self, X: Optional[np.ndarray] = None,
                base_predictions: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("StackingRegressor must be fitted before predict")

        if base_predictions is not None:
            base_predictions = np.array(base_predictions)
            return self.meta_model.predict(base_predictions.T)
        elif X is not None and self.base_models_ is not None:
            base_preds = self._predict_base(X)
            return self.meta_model.predict(base_preds.T)
        else:
            raise ValueError("Either base_predictions or (X and fitted base_models) must be provided")

    def get_result(self) -> Optional[StackingResult]:
        return self.result_

    def evaluate(self, X: Optional[np.ndarray] = None, y_true: np.ndarray = None,
                 base_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y_pred = self.predict(X, base_predictions)
        metrics = _scorer_reg(y_true, y_pred)

        base_metrics = {}
        if base_predictions is not None:
            base_preds = np.array(base_predictions)
        elif X is not None and self.base_models_ is not None:
            base_preds = self._predict_base(X)
        else:
            base_preds = None

        if base_preds is not None:
            for i in range(base_preds.shape[0]):
                base_metrics[f"BaseModel_{i}"] = _scorer_reg(y_true, base_preds[i])

        return {
            "ensemble_metrics": metrics,
            "base_model_metrics": base_metrics,
            "improvement": {
                key: metrics[key] - np.mean([m[key] for m in base_metrics.values()])
                for key in metrics.keys()
            } if base_metrics else None
        }


class StackingClassifier:
    def __init__(self, meta_model=None, n_folds: int = 5,
                 shuffle: bool = True, random_state: Optional[int] = None,
                 use_proba: bool = True):
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.use_proba = use_proba
        self._fitted = False
        self.result_: Optional[StackingResult] = None

    def fit(self, X: Optional[np.ndarray], y: np.ndarray,
            base_models: Optional[List[Any]] = None,
            base_predictions: Optional[np.ndarray] = None,
            X_eval: Optional[np.ndarray] = None,
            y_eval: Optional[np.ndarray] = None) -> "StackingClassifier":
        if self.meta_model is None:
            raise ValueError("meta_model must be provided")

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if base_predictions is not None:
            return self._fit_from_predictions(base_predictions, y, X_eval, y_eval)
        elif base_models is not None and X is not None:
            return self._fit_with_cv(X, y, base_models, X_eval, y_eval)
        else:
            raise ValueError("Either base_predictions or (base_models and X) must be provided")

    def _fit_from_predictions(self, base_predictions: np.ndarray,
                              y: np.ndarray,
                              X_eval: Optional[np.ndarray],
                              y_eval: Optional[np.ndarray]) -> "StackingClassifier":
        base_predictions = np.array(base_predictions)
        self.base_models_ = None
        self._fit_meta_model(base_predictions.T, y)

        self.result_ = StackingResult(
            n_folds_=None,
            random_state_=self.random_state,
        )
        self._extract_meta_model_params()
        self._fitted = True
        self._compute_scores(base_predictions, y, X_eval, y_eval)

        return self

    def _fit_with_cv(self, X: np.ndarray, y: np.ndarray,
                     base_models: List[Any],
                     X_eval: Optional[np.ndarray],
                     y_eval: Optional[np.ndarray]) -> "StackingClassifier":
        n_samples = len(y)
        n_models = len(base_models)

        folds = _kfold_split(n_samples, self.n_folds, self.shuffle, self.random_state)

        if self.use_proba:
            n_features_per_model = self.n_classes_ if self.n_classes_ > 2 else 1
        else:
            n_features_per_model = 1

        oof_preds = np.zeros((n_models * n_features_per_model, n_samples))
        self.base_models_ = []
        cv_scores = {f"Model_{i}": [] for i in range(n_models)}

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            for i, model in enumerate(base_models):
                model_clone = self._clone_model(model)
                model_clone.fit(X_train, y_train)

                if self.use_proba:
                    proba = model_clone.predict_proba(X_val)
                    if self.n_classes_ == 2:
                        val_pred = proba[:, 1]
                        oof_preds[i, val_idx] = val_pred
                    else:
                        start_idx = i * self.n_classes_
                        oof_preds[start_idx:start_idx + self.n_classes_, val_idx] = proba.T
                else:
                    val_pred = model_clone.predict(X_val)
                    oof_preds[i, val_idx] = val_pred

                acc = np.mean(y_val == model_clone.predict(X_val))
                cv_scores[f"Model_{i}"].append(acc)

            print(f"  Fold {fold_idx + 1}/{self.n_folds} complete")

        for i, model in enumerate(base_models):
            final_model = self._clone_model(model)
            final_model.fit(X, y)
            self.base_models_.append(final_model)

        self._fit_meta_model(oof_preds.T, y)

        self.result_ = StackingResult(
            cv_scores_=cv_scores,
            oof_predictions_=oof_preds,
            n_folds_=self.n_folds,
            random_state_=self.random_state,
        )
        self._extract_meta_model_params()
        self._fitted = True
        self._compute_scores(oof_preds, y, X_eval, y_eval)

        return self

    def _clone_model(self, model: Any) -> Any:
        try:
            from sklearn.base import clone
            return clone(model)
        except Exception:
            import copy
            return copy.deepcopy(model)

    def _fit_meta_model(self, meta_features: np.ndarray, y: np.ndarray) -> None:
        self.meta_model.fit(meta_features, y)

    def _extract_meta_model_params(self) -> None:
        if self.result_ is None:
            return

        if hasattr(self.meta_model, "coef_"):
            coef = np.array(self.meta_model.coef_)
            if coef.ndim > 1:
                coef = coef[0]
            self.result_.meta_model_coef_ = coef.flatten()

            if hasattr(self.meta_model, "intercept_"):
                intercept = self.meta_model.intercept_
                self.result_.meta_model_intercept_ = float(intercept) if np.isscalar(intercept) else float(intercept[0])

            if self.result_.meta_model_coef_ is not None:
                self.result_.feature_importance_ = {
                    f"MetaFeature_{i}": float(self.result_.meta_model_coef_[i])
                    for i in range(len(self.result_.meta_model_coef_))
                }

    def _compute_scores(self, base_preds: np.ndarray, y: np.ndarray,
                        X_eval: Optional[np.ndarray],
                        y_eval: Optional[np.ndarray]) -> None:
        if self.result_ is None:
            return

        if X_eval is not None and self.base_models_ is not None:
            base_preds_eval = self._predict_base(X_eval)
            stack_pred = self.predict(X_eval)
            stack_proba = self.predict_proba(X_eval)
            y_true_eval = y_eval if y_eval is not None else y
        else:
            base_preds_eval = base_preds
            stack_pred = self.meta_model.predict(base_preds.T)
            stack_proba = self.meta_model.predict_proba(base_preds.T) if hasattr(self.meta_model, "predict_proba") else None
            y_true_eval = y

        if self.use_proba and self.n_classes_ > 2:
            n_features = self.n_classes_
            for i in range(base_preds_eval.shape[0] // n_features):
                start_idx = i * n_features
                end_idx = start_idx + n_features
                proba = base_preds_eval[start_idx:end_idx].T
                pred = np.argmax(proba, axis=1)
                scores = _scorer_cls(y_true_eval, pred, proba[:, 1] if self.n_classes_ == 2 else proba)
                self.result_.base_model_scores_[f"BaseModel_{i}"] = float(scores["Accuracy"])
        else:
            for i in range(base_preds_eval.shape[0]):
                if self.use_proba and self.n_classes_ == 2:
                    pred = (base_preds_eval[i] >= 0.5).astype(int)
                    proba = base_preds_eval[i]
                else:
                    pred = base_preds_eval[i].astype(int)
                    proba = None
                scores = _scorer_cls(y_true_eval, pred, proba)
                self.result_.base_model_scores_[f"BaseModel_{i}"] = float(scores["Accuracy"])

        stack_scores = _scorer_cls(y_true_eval, stack_pred,
                                   stack_proba[:, 1] if stack_proba is not None and stack_proba.ndim == 2 else None)
        self.result_.ensemble_score_ = float(stack_scores["Accuracy"])
        self.result_.ensemble_metrics_ = stack_scores

    def _predict_base(self, X: np.ndarray) -> np.ndarray:
        if self.base_models_ is None:
            raise ValueError("No base models fitted; use fit with base_models parameter")
        preds = []
        for model in self.base_models_:
            if self.use_proba:
                proba = model.predict_proba(X)
                if self.n_classes_ == 2:
                    preds.append(proba[:, 1])
                else:
                    preds.extend(proba.T)
            else:
                preds.append(model.predict(X))
        return np.array(preds)

    def predict(self, X: Optional[np.ndarray] = None,
                base_predictions: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("StackingClassifier must be fitted before predict")

        if base_predictions is not None:
            base_predictions = np.array(base_predictions)
            return self.meta_model.predict(base_predictions.T)
        elif X is not None and self.base_models_ is not None:
            base_preds = self._predict_base(X)
            return self.meta_model.predict(base_preds.T)
        else:
            raise ValueError("Either base_predictions or (X and fitted base_models) must be provided")

    def predict_proba(self, X: Optional[np.ndarray] = None,
                      base_predictions: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("StackingClassifier must be fitted before predict")

        if not hasattr(self.meta_model, "predict_proba"):
            raise RuntimeError("meta_model does not support predict_proba")

        if base_predictions is not None:
            base_predictions = np.array(base_predictions)
            return self.meta_model.predict_proba(base_predictions.T)
        elif X is not None and self.base_models_ is not None:
            base_preds = self._predict_base(X)
            return self.meta_model.predict_proba(base_preds.T)
        else:
            raise ValueError("Either base_predictions or (X and fitted base_models) must be provided")

    def get_result(self) -> Optional[StackingResult]:
        return self.result_

    def evaluate(self, X: Optional[np.ndarray] = None, y_true: np.ndarray = None,
                 base_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y_pred = self.predict(X, base_predictions)
        y_proba = self.predict_proba(X, base_predictions) if hasattr(self.meta_model, "predict_proba") else None
        metrics = _scorer_cls(y_true, y_pred, y_proba[:, 1] if y_proba is not None and y_proba.ndim == 2 else None)

        base_metrics = {}
        if base_predictions is not None:
            base_preds = np.array(base_predictions)
        elif X is not None and self.base_models_ is not None:
            base_preds = self._predict_base(X)
        else:
            base_preds = None

        if base_preds is not None:
            if self.use_proba and self.n_classes_ > 2:
                n_features = self.n_classes_
                for i in range(base_preds.shape[0] // n_features):
                    start_idx = i * n_features
                    end_idx = start_idx + n_features
                    proba = base_preds[start_idx:end_idx].T
                    pred = np.argmax(proba, axis=1)
                    base_metrics[f"BaseModel_{i}"] = _scorer_cls(
                        y_true, pred,
                        proba[:, 1] if self.n_classes_ == 2 else proba
                    )
            else:
                for i in range(base_preds.shape[0]):
                    if self.use_proba and self.n_classes_ == 2:
                        pred = (base_preds[i] >= 0.5).astype(int)
                        proba = base_preds[i]
                    else:
                        pred = base_preds[i].astype(int)
                        proba = None
                    base_metrics[f"BaseModel_{i}"] = _scorer_cls(y_true, pred, proba)

        return {
            "ensemble_metrics": metrics,
            "base_model_metrics": base_metrics,
            "improvement": {
                key: metrics[key] - np.mean([m[key] for m in base_metrics.values()])
                for key in metrics.keys()
            } if base_metrics else None
        }


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("1. VotingClassifier - Binary Classification")
    print("=" * 60)

    model_preds = np.array([
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
    ])

    hard_voter = VotingClassifier(mode="hard")
    print(f"Hard voting result: {hard_voter.predict(model_preds)}")

    weighted_hard = VotingClassifier(mode="hard", weights=[3, 2, 1])
    print(f"Weighted hard voting: {weighted_hard.predict(model_preds)}")

    model_probas = np.array([
        [0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.1, 0.9, 0.75, 0.4],
        [0.8, 0.6, 0.7, 0.4, 0.35, 0.9, 0.2, 0.8, 0.3, 0.45],
        [0.4, 0.3, 0.65, 0.8, 0.7, 0.75, 0.15, 0.4, 0.8, 0.35],
    ])

    soft_voter = VotingClassifier(mode="soft")
    print(f"Soft voting result:  {soft_voter.predict(model_probas)}")

    weighted_soft = VotingClassifier(mode="soft", weights=[3, 2, 1])
    print(f"Weighted soft voting:{weighted_soft.predict(model_probas)}")

    print()
    print("=" * 60)
    print("2. VotingClassifierMulticlass - Multi-class Classification")
    print("=" * 60)

    mc_probas = np.array([
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]],
        [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
        [[0.6, 0.1, 0.3], [0.15, 0.7, 0.15], [0.1, 0.4, 0.5]],
    ])

    mc_voter = VotingClassifierMulticlass(n_classes=3, mode="soft")
    print(f"Multiclass soft vote: {mc_voter.predict_from_proba(mc_probas)}")

    weighted_mc = VotingClassifierMulticlass(n_classes=3, mode="soft", weights=[2, 1, 1])
    print(f"Weighted multiclass:  {weighted_mc.predict_from_proba(mc_probas)}")

    print()
    print("=" * 60)
    print("3. AveragingRegressor - Regression")
    print("=" * 60)

    reg_preds = np.array([
        [10.2, 20.5, 30.1, 40.8, 50.3],
        [9.8, 21.0, 29.5, 41.2, 49.8],
        [10.5, 19.8, 30.8, 39.9, 50.7],
    ])

    avg_regressor = AveragingRegressor()
    print(f"Simple averaging: {avg_regressor.predict(reg_preds)}")

    weighted_avg = AveragingRegressor(weights=[0.5, 0.3, 0.2])
    print(f"Weighted averaging:{weighted_avg.predict(reg_preds)}")

    print()
    print("=" * 60)
    print("4. Stacking with Cross-Validation (堆叠集成 + 交叉验证)")
    print("=" * 60)

    try:
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.datasets import make_regression, make_classification
        from sklearn.model_selection import train_test_split

        print("\n--- 4.1 Stacking Regressor with CV ---")
        X_reg, y_reg = make_regression(n_samples=500, n_features=20, n_informative=15,
                                       noise=15, random_state=42)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        base_models_reg = [
            LinearRegression(),
            Ridge(alpha=1.0),
            Lasso(alpha=0.5),
            RandomForestRegressor(n_estimators=50, random_state=42),
        ]

        stack_reg_cv = StackingRegressor(
            meta_model=LinearRegression(),
            n_folds=5,
            shuffle=True,
            random_state=42
        )

        print("  Training base models with 5-fold CV...")
        stack_reg_cv.fit(
            X=X_train_reg,
            y=y_train_reg,
            base_models=base_models_reg,
            X_eval=X_test_reg,
            y_eval=y_test_reg
        )

        result_reg = stack_reg_cv.get_result()
        print(f"\n  Cross-validation fold scores (R2):")
        for model_name, scores in result_reg.cv_scores_.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"    {model_name}: {mean_score:.4f} ± {std_score:.4f}")

        print(f"\n  Meta-model coefficients:")
        if result_reg.meta_model_coef_ is not None:
            for i, coef in enumerate(result_reg.meta_model_coef_):
                print(f"    BaseModel_{i}: {coef:.4f}")
            print(f"    Intercept: {result_reg.meta_model_intercept_:.4f}")

        if result_reg.feature_importance_ is not None:
            print(f"\n  Feature importance (meta-model coefs):")
            sorted_feats = sorted(result_reg.feature_importance_.items(),
                                  key=lambda x: abs(x[1]), reverse=True)
            for name, imp in sorted_feats:
                print(f"    {name}: {imp:.4f}")

        eval_reg = stack_reg_cv.evaluate(X=X_test_reg, y_true=y_test_reg)
        print(f"\n  Test set performance:")
        print(f"    {'Model':<20} {'R2':>8} {'RMSE':>8} {'MAE':>8}")
        print(f"    {'-' * 50}")
        for model_name, metrics in eval_reg["base_model_metrics"].items():
            print(f"    {model_name:<20} {metrics['R2']:>8.4f} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f}")
        ens = eval_reg["ensemble_metrics"]
        print(f"    {'Stacking Ensemble':<20} {ens['R2']:>8.4f} {ens['RMSE']:>8.4f} {ens['MAE']:>8.4f}")
        print(f"    {'-' * 50}")
        imp = eval_reg["improvement"]
        print(f"    {'Improvement (avg)':<20} {imp['R2']:>+8.4f} {imp['RMSE']:>+8.4f} {imp['MAE']:>+8.4f}")

        print("\n--- 4.2 Stacking Classifier with CV ---")
        X_cls, y_cls = make_classification(n_samples=500, n_features=20, n_informative=12,
                                           n_redundant=3, n_classes=2,
                                           weights=[0.6, 0.4], random_state=42)
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42
        )

        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        base_models_cls = [
            LogisticRegression(max_iter=1000),
            KNeighborsClassifier(n_neighbors=5),
            RandomForestClassifier(n_estimators=50, random_state=42),
            GradientBoostingClassifier(n_estimators=50, random_state=42),
        ]

        stack_cls_cv = StackingClassifier(
            meta_model=LogisticRegression(max_iter=1000),
            n_folds=5,
            shuffle=True,
            random_state=42,
            use_proba=True
        )

        print("  Training base models with 5-fold CV (using probabilities)...")
        stack_cls_cv.fit(
            X=X_train_cls,
            y=y_train_cls,
            base_models=base_models_cls,
            X_eval=X_test_cls,
            y_eval=y_test_cls
        )

        result_cls = stack_cls_cv.get_result()
        print(f"\n  Cross-validation fold scores (Accuracy):")
        for model_name, scores in result_cls.cv_scores_.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"    {model_name}: {mean_score:.4f} ± {std_score:.4f}")

        print(f"\n  Meta-model coefficients:")
        if result_cls.meta_model_coef_ is not None:
            for i, coef in enumerate(result_cls.meta_model_coef_):
                print(f"    MetaFeature_{i}: {coef:.4f}")
            if result_cls.meta_model_intercept_ is not None:
                print(f"    Intercept: {result_cls.meta_model_intercept_:.4f}")

        eval_cls = stack_cls_cv.evaluate(X=X_test_cls, y_true=y_test_cls)
        print(f"\n  Test set performance:")
        print(f"    {'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
        print(f"    {'-' * 55}")
        for model_name, metrics in eval_cls["base_model_metrics"].items():
            auc = metrics.get('AUC', None)
            auc_str = f"{auc:>7.4f}" if auc is not None else "     N/A"
            print(f"    {model_name:<20} {metrics['Accuracy']:>7.4f} {metrics['Precision']:>7.4f} "
                  f"{metrics['Recall']:>7.4f} {metrics['F1']:>7.4f} {auc_str}")
        ens_cls = eval_cls["ensemble_metrics"]
        auc_ens = ens_cls.get('AUC', None)
        auc_ens_str = f"{auc_ens:>7.4f}" if auc_ens is not None else "     N/A"
        print(f"    {'Stacking Ensemble':<20} {ens_cls['Accuracy']:>7.4f} {ens_cls['Precision']:>7.4f} "
              f"{ens_cls['Recall']:>7.4f} {ens_cls['F1']:>7.4f} {auc_ens_str}")
        print(f"    {'-' * 55}")
        imp_cls = eval_cls["improvement"]
        if imp_cls:
            print(f"    {'Improvement (avg)':<20} {imp_cls['Accuracy']:>+7.4f} {imp_cls['Precision']:>+7.4f} "
                  f"{imp_cls['Recall']:>+7.4f} {imp_cls['F1']:>+7.4f}")

        print("\n--- 4.3 Stacking from pre-computed predictions ---")
        n_test = len(y_test_reg)
        base_preds_reg = np.array([
            stack_reg_cv.base_models_[i].predict(X_test_reg) + np.random.randn(n_test) * 2
            for i in range(len(base_models_reg))
        ])

        stack_reg_simple = StackingRegressor(meta_model=Ridge(alpha=1.0))
        stack_reg_simple.fit(
            X=None,
            y=y_test_reg,
            base_predictions=base_preds_reg
        )

        eval_simple = stack_reg_simple.evaluate(y_true=y_test_reg,
                                                base_predictions=base_preds_reg)
        print(f"  Stacking from pre-computed predictions:")
        print(f"    Ensemble R2: {eval_simple['ensemble_metrics']['R2']:.4f}")
        print(f"    Avg base R2: {np.mean([m['R2'] for m in eval_simple['base_model_metrics'].values()]):.4f}")

    except ImportError as e:
        print(f"  sklearn not installed or missing component, skipping stacking demo: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"  Error in stacking demo: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("5. Performance comparison on synthetic data")
    print("=" * 60)

    n_samples = 1000
    y_binary = np.random.randint(0, 2, n_samples)

    p1 = np.clip(y_binary * 0.8 + (1 - y_binary) * 0.2 + np.random.randn(n_samples) * 0.1, 0, 1)
    p2 = np.clip(y_binary * 0.75 + (1 - y_binary) * 0.25 + np.random.randn(n_samples) * 0.12, 0, 1)
    p3 = np.clip(y_binary * 0.7 + (1 - y_binary) * 0.3 + np.random.randn(n_samples) * 0.15, 0, 1)

    probas = np.array([p1, p2, p3])
    hard_preds = (probas >= 0.5).astype(int)

    hard_result = VotingClassifier(mode="hard").predict(hard_preds)
    soft_result = VotingClassifier(mode="soft").predict(probas)
    wsoft_result = VotingClassifier(mode="soft", weights=[0.5, 0.3, 0.2]).predict(probas)

    hard_acc = np.mean(hard_result == y_binary)
    soft_acc = np.mean(soft_result == y_binary)
    wsoft_acc = np.mean(wsoft_result == y_binary)
    single_accs = [np.mean((p >= 0.5).astype(int) == y_binary) for p in [p1, p2, p3]]

    print(f"Model 1 accuracy:          {single_accs[0]:.4f}")
    print(f"Model 2 accuracy:          {single_accs[1]:.4f}")
    print(f"Model 3 accuracy:          {single_accs[2]:.4f}")
    print(f"Hard voting accuracy:      {hard_acc:.4f}")
    print(f"Soft voting accuracy:      {soft_acc:.4f}")
    print(f"Weighted soft accuracy:    {wsoft_acc:.4f}")

    print()
    print("=" * 60)
    print("6. Tie-break Scenarios (平票场景测试)")
    print("=" * 60)

    print("\n--- 二分类硬投票 平票 (2:2) ---")
    tie_preds = np.array([
        [1, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
    ])
    tie_probas = np.array([
        [0.9, 0.1, 0.6],
        [0.8, 0.2, 0.55],
        [0.3, 0.7, 0.65],
        [0.4, 0.6, 0.45],
    ])

    for strategy in ["first", "positive", "confidence", "prob_sum"]:
        voter = VotingClassifier(mode="hard", tie_break=strategy)
        if strategy in ["confidence", "prob_sum"]:
            pred = voter.predict(tie_preds, probas=tie_probas)
        else:
            pred = voter.predict(tie_preds)
        print(f"  tie_break={strategy:12s}: {pred}   (平票样本: 第0,1,2个样本)")

    print("\n--- 二分类软投票 平票 (均值=0.5) ---")
    tie_soft_probas = np.array([
        [0.6, 0.5, 0.5],
        [0.4, 0.5, 0.5],
    ])

    for strategy in ["first", "positive", "confidence"]:
        voter = VotingClassifier(mode="soft", tie_break=strategy)
        pred = voter.predict(tie_soft_probas)
        print(f"  tie_break={strategy:12s}: {pred}   (平票样本: 第0,1个样本均值=0.5)")

    print("\n--- 三分类硬投票 平票 (1:1:1) ---")
    mc_tie_preds = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
    ])
    mc_tie_probas = np.array([
        [[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]],
        [[0.1, 0.7, 0.2], [0.1, 0.2, 0.7]],
        [[0.2, 0.1, 0.7], [0.6, 0.2, 0.2]],
    ])

    for strategy in ["first", "confidence", "prob_sum"]:
        voter = VotingClassifierMulticlass(n_classes=3, mode="hard", tie_break=strategy)
        if strategy in ["confidence", "prob_sum"]:
            pred = voter.predict(mc_tie_preds, probas=mc_tie_probas)
        else:
            pred = voter.predict(mc_tie_preds)
        print(f"  tie_break={strategy:12s}: {pred}   (两个样本均为平票)")

    print()
    print("=" * 60)
    print("7. Decision Rules (决策规则说明)")
    print("=" * 60)

    demo_voter = VotingClassifier(mode="hard", weights=[2, 1, 1], tie_break="confidence")
    rules = demo_voter.get_decision_rules()
    print(f"\n  模式: {rules['mode']}")
    print(f"  权重: {rules['weights']}")
    print(f"  平票策略: {rules['tie_break']}")
    print(f"  完整规则: {rules['description']}")
    print(f"\n  所有平票策略说明:")
    for key, desc in rules["tie_break_options"].items():
        print(f"    {key:12s}: {desc}")

    print()
    demo_voter2 = VotingClassifierMulticlass(n_classes=3, mode="soft", tie_break="prob_sum")
    rules2 = demo_voter2.get_decision_rules()
    print(f"  多分类模式: {rules2['mode']}")
    print(f"  类别数: {rules2['n_classes']}")
    print(f"  平票策略: {rules2['tie_break']}")
    print(f"  完整规则: {rules2['description']}")

    print()
    print("=" * 60)
    print("平票处理原理说明:")
    print("=" * 60)
    print("""
  问题: 原实现中 np.argmax 在遇到多个最大值时只会返回第一个,
        这取决于 np.unique 的返回顺序,行为不确定。

  修复:
  1. 显式检测平票 (多个类别票数/概率相等)
  2. 提供 4 种确定性平票策略:
     - first:     选类别编号最小的 (确定性,默认)
     - positive:  优先选正类 (1),适合宁可误报也不漏报的场景
     - confidence:选置信度最高的 (需传入概率矩阵)
     - prob_sum:  选所有模型概率之和最大的 (需传入概率矩阵)

  3. 提供 get_decision_rules() 方法返回完整决策规则说明
""")
