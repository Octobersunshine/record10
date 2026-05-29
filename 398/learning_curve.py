import copy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


def generate_learning_curve(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    task_type="classification",
    train_sizes=None,
    metric=None,
    n_repeats=5,
    random_state=42,
    return_diagnosis=False,
    higher_is_better=None,
    diagnosis_thresholds=None,
):
    """
    生成模型的学习曲线数据

    Parameters:
    -----------
    model : sklearn风格的模型对象
        需要实现fit()和predict()方法
    X_train : array-like, shape (n_samples, n_features)
        训练集特征
    y_train : array-like, shape (n_samples,)
        训练集标签
    X_val : array-like, shape (n_samples, n_features)
        验证集特征
    y_val : array-like, shape (n_samples,)
        验证集标签
    task_type : str, optional (default="classification")
        任务类型，"classification"或"regression"
    train_sizes : array-like, optional
        训练样本量比例，默认从10%到100%，步长10%
    metric : callable, optional
        自定义评估指标函数，签名为 metric(y_true, y_pred)
    n_repeats : int, optional (default=5)
        每个样本量下重复采样的次数，取平均以减少小样本时验证指标的波动
    random_state : int, optional (default=42)
        随机种子，用于每次采样训练数据时的可重复性
    return_diagnosis : bool, optional (default=False)
        是否同时返回欠拟合/过拟合诊断报告
    higher_is_better : bool, optional
        指标是否越高越好，仅在 return_diagnosis=True 时使用。默认根据 metric_name 自动推断
    diagnosis_thresholds : dict, optional
        自定义诊断阈值，仅在 return_diagnosis=True 时使用

    Returns:
    --------
    dict or (dict, dict)
        return_diagnosis=False 时仅返回学习曲线数据字典：
        - train_sizes: 训练样本量比例数组
        - train_samples: 每个比例对应的实际训练样本数
        - train_scores_mean: 训练集性能指标的均值
        - train_scores_std: 训练集性能指标的标准差
        - val_scores_mean: 验证集性能指标的均值
        - val_scores_std: 验证集性能指标的标准差
        - metric_name: 使用的评估指标名称

        return_diagnosis=True 时返回 (lc_data, diagnosis_report) 元组
    """
    if train_sizes is None:
        train_sizes = np.arange(0.1, 1.01, 0.1)

    if metric is None:
        if task_type == "classification":
            metric = accuracy_score
            metric_name = "accuracy"
        elif task_type == "regression":
            metric = lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
            metric_name = "neg_rmse"
        else:
            raise ValueError("task_type 必须是 'classification' 或 'regression'")
    else:
        metric_name = metric.__name__

    rng = np.random.RandomState(random_state)

    X_train_arr = np.asarray(X_train)
    y_train_arr = np.asarray(y_train)
    total_samples = len(X_train_arr)

    train_scores_all = []
    val_scores_all = []
    train_sample_counts = []

    for size in train_sizes:
        n_samples = int(total_samples * size)
        if n_samples < 1:
            n_samples = 1

        repeat_train_scores = []
        repeat_val_scores = []

        for _ in range(n_repeats):
            indices = rng.choice(total_samples, n_samples, replace=False)
            X_subset = X_train_arr[indices]
            y_subset = y_train_arr[indices]

            cloned_model = copy.deepcopy(model)
            cloned_model.fit(X_subset, y_subset)

            y_train_pred = cloned_model.predict(X_subset)
            y_val_pred = cloned_model.predict(X_val)

            repeat_train_scores.append(metric(y_subset, y_train_pred))
            repeat_val_scores.append(metric(y_val, y_val_pred))

        train_scores_all.append(repeat_train_scores)
        val_scores_all.append(repeat_val_scores)
        train_sample_counts.append(n_samples)

    train_scores_all = np.array(train_scores_all)
    val_scores_all = np.array(val_scores_all)

    lc_data = {
        "train_sizes": np.array(train_sizes),
        "train_samples": np.array(train_sample_counts),
        "train_scores_mean": train_scores_all.mean(axis=1),
        "train_scores_std": train_scores_all.std(axis=1),
        "val_scores_mean": val_scores_all.mean(axis=1),
        "val_scores_std": val_scores_all.std(axis=1),
        "metric_name": metric_name,
    }

    if return_diagnosis:
        diagnosis_report = diagnose_learning_curve(
            lc_data,
            higher_is_better=higher_is_better,
            thresholds=diagnosis_thresholds,
        )
        return lc_data, diagnosis_report

    return lc_data


def plot_learning_curve(learning_curve_data, title="Learning Curve"):
    """
    绘制学习曲线（含标准差阴影区域）

    Parameters:
    -----------
    learning_curve_data : dict
        generate_learning_curve函数返回的字典
    title : str, optional
        图表标题
    """
    import matplotlib.pyplot as plt

    train_sizes = learning_curve_data["train_sizes"]
    train_mean = learning_curve_data["train_scores_mean"]
    train_std = learning_curve_data["train_scores_std"]
    val_mean = learning_curve_data["val_scores_mean"]
    val_std = learning_curve_data["val_scores_std"]
    metric_name = learning_curve_data["metric_name"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x_axis = train_sizes * 100

    ax.plot(x_axis, train_mean, "o-", label="Training", color="#1f77b4", linewidth=2)
    ax.fill_between(
        x_axis,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="#1f77b4",
    )

    ax.plot(x_axis, val_mean, "o-", label="Validation", color="#ff7f0e", linewidth=2)
    ax.fill_between(
        x_axis,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
        color="#ff7f0e",
    )

    ax.set_xlabel("Training Set Size (%)")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(10, 101, 10))

    if metric_name == "neg_rmse":
        ax.set_ylabel("Negative RMSE")

    fig.tight_layout()
    return plt


def diagnose_learning_curve(learning_curve_data, higher_is_better=None, thresholds=None):
    """
    根据学习曲线诊断模型的欠拟合/过拟合状态，并给出改进建议

    Parameters:
    -----------
    learning_curve_data : dict
        generate_learning_curve函数返回的字典
    higher_is_better : bool, optional
        指标是否越高越好。默认根据 metric_name 自动推断：
        - accuracy, f1, precision, recall, roc_auc: True
        - rmse, mse, mae, neg_rmse: False
    thresholds : dict, optional
        自定义诊断阈值，包含：
        - gap_thresh: 训练集与验证集性能差距的阈值（默认0.1）
        - train_low_thresh: 训练集性能"差"的阈值（默认0.7，越高越好时）
        - train_high_thresh: 训练集性能"好"的阈值（默认0.9，越高越好时）
        - converge_thresh: 曲线收敛判定阈值（最后3个点的性能变化小于该值则认为收敛）

    Returns:
    --------
    dict
        诊断报告，包含以下键：
        - diagnosis: 诊断结论（'underfitting', 'overfitting', 'good_fit', 'needs_more_data'）
        - diagnosis_cn: 中文诊断结论
        - train_score_final: 训练集最终性能（最后一个点的均值）
        - val_score_final: 验证集最终性能（最后一个点的均值）
        - final_gap: 最终的训练集与验证集性能差距
        - is_converged: 曲线是否已收敛
        - trend: 验证集曲线趋势（'rising', 'stable', 'falling'）
        - recommendations: 改进建议列表
        - details: 详细诊断信息
    """
    train_mean = learning_curve_data["train_scores_mean"]
    val_mean = learning_curve_data["val_scores_mean"]
    val_std = learning_curve_data.get("val_scores_std", np.zeros_like(val_mean))
    metric_name = learning_curve_data.get("metric_name", "")

    if higher_is_better is None:
        higher_is_better = _infer_higher_is_better(metric_name)

    if thresholds is None:
        thresholds = {}

    gap_thresh = thresholds.get("gap_thresh", 0.1)
    train_low_thresh = thresholds.get("train_low_thresh", 0.7)
    train_high_thresh = thresholds.get("train_high_thresh", 0.9)
    converge_thresh = thresholds.get("converge_thresh", 0.02)

    if not higher_is_better:
        train_low_thresh, train_high_thresh = train_high_thresh, train_low_thresh
        gap_thresh = -gap_thresh

    train_final = train_mean[-1]
    val_final = val_mean[-1]

    if higher_is_better:
        final_gap = train_final - val_final
    else:
        final_gap = val_final - train_final

    n_points = len(val_mean)
    if n_points >= 3:
        last_three = val_mean[-3:]
        max_change = np.max(last_three) - np.min(last_three) if higher_is_better else np.min(last_three) - np.max(last_three)
        is_converged = abs(max_change) < converge_thresh
    else:
        is_converged = False

    if n_points >= 3:
        first_half = val_mean[:n_points // 2]
        second_half = val_mean[n_points // 2:]
        if higher_is_better:
            if np.mean(second_half) - np.mean(first_half) > converge_thresh:
                trend = "rising"
            elif abs(np.mean(second_half) - np.mean(first_half)) < converge_thresh:
                trend = "stable"
            else:
                trend = "falling"
        else:
            if np.mean(first_half) - np.mean(second_half) > converge_thresh:
                trend = "rising"
            elif abs(np.mean(first_half) - np.mean(second_half)) < converge_thresh:
                trend = "stable"
            else:
                trend = "falling"
    else:
        trend = "unknown"

    diagnosis = None
    diagnosis_cn = None
    recommendations = []

    if higher_is_better:
        train_is_low = train_final < train_low_thresh
        train_is_high = train_final > train_high_thresh
        gap_is_large = final_gap > abs(gap_thresh)
    else:
        train_is_low = train_final > train_low_thresh
        train_is_high = train_final < train_high_thresh
        gap_is_large = final_gap > abs(gap_thresh)

    if train_is_low and not gap_is_large:
        diagnosis = "underfitting"
        diagnosis_cn = "欠拟合（高偏差）"
        recommendations.extend([
            "增加模型复杂度（如使用更复杂的模型、增加树深度/层数）",
            "减少正则化强度（如减小L2正则化系数）",
            "增加有用的特征或进行特征工程",
            "延长训练时间（如增加迭代次数）",
        ])
    elif train_is_high and gap_is_large:
        diagnosis = "overfitting"
        diagnosis_cn = "过拟合（高方差）"
        recommendations.extend([
            "增加训练样本量（当前曲线仍有上升空间）",
            "增加正则化（如L1/L2正则化、Dropout）",
            "降低模型复杂度（如减小树深度、减少层数）",
            "进行特征选择，减少噪声特征",
            "增加数据增强",
        ])
    elif train_is_high and not gap_is_large and is_converged:
        diagnosis = "good_fit"
        diagnosis_cn = "拟合良好"
        recommendations.extend([
            "当前模型状态良好，可考虑：",
            "在测试集上进行最终验证",
            "尝试集成学习进一步提升性能",
        ])
    elif not is_converged and trend == "rising":
        diagnosis = "needs_more_data"
        diagnosis_cn = "需要更多训练数据"
        recommendations.extend([
            "验证曲线仍在上升，增加训练样本可能进一步提升性能",
            "如果样本量受限，可尝试数据增强或半监督学习",
        ])
    elif train_is_low and gap_is_large:
        diagnosis = "underfitting_with_high_variance"
        diagnosis_cn = "欠拟合且伴有高方差"
        recommendations.extend([
            "首先增加模型复杂度以解决欠拟合问题",
            "同时考虑增加正则化控制方差",
            "进行特征工程提升数据质量",
        ])
    else:
        diagnosis = "intermediate"
        diagnosis_cn = "中间状态，需进一步观察"
        recommendations.extend([
            "建议增加训练样本量继续观察曲线趋势",
            "可微调模型超参数",
        ])

    details = {
        "higher_is_better": higher_is_better,
        "train_final": train_final,
        "val_final": val_final,
        "final_gap": final_gap,
        "is_converged": is_converged,
        "trend": trend,
        "val_std_final": val_std[-1],
        "gap_threshold_used": gap_thresh,
        "train_low_threshold": train_low_thresh,
        "train_high_threshold": train_high_thresh,
    }

    report = {
        "diagnosis": diagnosis,
        "diagnosis_cn": diagnosis_cn,
        "train_score_final": train_final,
        "val_score_final": val_final,
        "final_gap": final_gap,
        "is_converged": is_converged,
        "trend": trend,
        "recommendations": recommendations,
        "details": details,
    }

    return report


def format_diagnosis_report(report, metric_name="score"):
    """
    将诊断报告格式化为易读的字符串

    Parameters:
    -----------
    report : dict
        diagnose_learning_curve 返回的诊断报告
    metric_name : str, optional
        指标名称，用于显示

    Returns:
    --------
    str
        格式化的诊断报告字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("               模型学习曲线诊断报告")
    lines.append("=" * 60)
    lines.append(f"诊断结论: {report['diagnosis_cn']}")
    lines.append(f"英文标识: {report['diagnosis']}")
    lines.append("-" * 60)
    lines.append(f"训练集最终{metric_name}: {report['train_score_final']:.4f}")
    lines.append(f"验证集最终{metric_name}: {report['val_score_final']:.4f}")
    lines.append(f"两曲线最终差距: {report['final_gap']:.4f}")
    lines.append(f"曲线是否收敛: {'是' if report['is_converged'] else '否'}")
    lines.append(f"验证曲线趋势: {_trend_cn(report['trend'])}")
    lines.append("-" * 60)
    lines.append("改进建议:")
    for i, rec in enumerate(report["recommendations"], 1):
        lines.append(f"  {i}. {rec}")
    lines.append("-" * 60)
    lines.append("详细信息:")
    d = report["details"]
    lines.append(f"  指标方向: {'越高越好' if d['higher_is_better'] else '越低越好'}")
    lines.append(f"  验证集标准差(最终): {d['val_std_final']:.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _infer_higher_is_better(metric_name):
    """根据指标名称推断指标方向"""
    name = metric_name.lower()
    higher_better_names = ["accuracy", "acc", "f1", "precision", "recall", "roc_auc", "r2", "score"]
    lower_better_names = ["rmse", "mse", "mae", "loss", "error", "neg_"]

    for h in higher_better_names:
        if h in name:
            return True
    for l in lower_better_names:
        if l in name:
            return False
    return True


def _trend_cn(trend):
    """趋势英文转中文"""
    mapping = {
        "rising": "上升（性能仍在提升）",
        "stable": "平稳（已收敛）",
        "falling": "下降（性能恶化）",
        "unknown": "未知",
    }
    return mapping.get(trend, trend)


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    print("=== 分类任务示例 ===")
    X_cls, y_cls = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=42
    )
    X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )

    model_cls = RandomForestClassifier(n_estimators=100, random_state=42)
    lc_data_cls, report_cls = generate_learning_curve(
        model_cls,
        X_train_cls,
        y_train_cls,
        X_val_cls,
        y_val_cls,
        task_type="classification",
        n_repeats=5,
        return_diagnosis=True,
    )

    print("训练样本比例:", lc_data_cls["train_sizes"])
    print("训练样本数:", lc_data_cls["train_samples"])
    print("训练准确率(均值):", lc_data_cls["train_scores_mean"])
    print("训练准确率(标准差):", lc_data_cls["train_scores_std"])
    print("验证准确率(均值):", lc_data_cls["val_scores_mean"])
    print("验证准确率(标准差):", lc_data_cls["val_scores_std"])
    print("指标名称:", lc_data_cls["metric_name"])
    print("\n")
    print(format_diagnosis_report(report_cls, "准确率"))

    print("\n=== 回归任务示例 ===")
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=20, n_informative=10, noise=0.1, random_state=42
    )
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    lc_data_reg, report_reg = generate_learning_curve(
        model_reg,
        X_train_reg,
        y_train_reg,
        X_val_reg,
        y_val_reg,
        task_type="regression",
        metric=rmse,
        n_repeats=5,
        return_diagnosis=True,
        higher_is_better=False,
    )

    print("训练样本比例:", lc_data_reg["train_sizes"])
    print("训练样本数:", lc_data_reg["train_samples"])
    print("训练RMSE(均值):", lc_data_reg["train_scores_mean"])
    print("训练RMSE(标准差):", lc_data_reg["train_scores_std"])
    print("验证RMSE(均值):", lc_data_reg["val_scores_mean"])
    print("验证RMSE(标准差):", lc_data_reg["val_scores_std"])
    print("指标名称:", lc_data_reg["metric_name"])
    print("\n")
    print(format_diagnosis_report(report_reg, "RMSE"))

    print("\n=== 小样本测试（20个样本） ===")
    X_tiny, y_tiny = make_classification(
        n_samples=20, n_features=5, n_informative=3, random_state=42
    )
    X_tr_tiny, X_val_tiny, y_tr_tiny, y_val_tiny = train_test_split(
        X_tiny, y_tiny, test_size=0.3, random_state=42
    )
    from sklearn.tree import DecisionTreeClassifier

    model_tiny = DecisionTreeClassifier(random_state=42)
    lc_data_tiny, report_tiny = generate_learning_curve(
        model_tiny,
        X_tr_tiny,
        y_tr_tiny,
        X_val_tiny,
        y_val_tiny,
        task_type="classification",
        n_repeats=10,
        return_diagnosis=True,
    )
    print("训练样本数:", lc_data_tiny["train_samples"])
    print("验证准确率(均值):", lc_data_tiny["val_scores_mean"])
    print("验证准确率(标准差):", lc_data_tiny["val_scores_std"])
    print("\n")
    print(format_diagnosis_report(report_tiny, "准确率"))

    print("\n学习曲线数据生成完成！")
