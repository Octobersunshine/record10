import numpy as np
from typing import List, Tuple, Dict, Union

def calculate_pr_curve(y_true: List[int], y_prob: List[float]) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    计算二分类的精确率-召回率曲线（PR曲线）和平均精度（AP）

    对相同概率值的样本组内部进行细粒度采样：
    - 优先排列正例再排列负例，确保曲线平滑（避免垂直/水平线段）
    - 在组内均匀插值阈值，使曲线点列连续过渡

    Args:
        y_true: 真实标签列表，取值为0或1
        y_prob: 预测概率列表，取值范围[0,1]

    Returns:
        pr_points: 列表，每个元素为(阈值, 精确率, 召回率)的元组
        ap: 平均精度（Average Precision）
    """
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    n = len(y_true)
    total_positives = int(np.sum(y_true == 1))

    if total_positives == 0:
        return [(1.0, 1.0, 0.0)], 0.0

    sort_key = np.lexsort((-y_true, -y_prob))
    y_true_sorted = y_true[sort_key]
    y_prob_sorted = y_prob[sort_key]

    groups = []
    i = 0
    while i < n:
        j = i
        while j < n and y_prob_sorted[j] == y_prob_sorted[i]:
            j += 1
        groups.append((i, j))
        i = j

    unique_probs = [y_prob_sorted[g[0]] for g in groups]

    pr_points = []
    precisions_for_ap = []
    recalls_for_ap = []

    tp = 0
    fp = 0

    for group_idx, (start, end) in enumerate(groups):
        group_size = end - start
        group_prob = unique_probs[group_idx]
        next_prob = unique_probs[group_idx + 1] if group_idx + 1 < len(groups) else 0.0

        for step in range(group_size):
            k = start + step
            if y_true_sorted[k] == 1:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp)
            recall = tp / total_positives

            if group_size > 1:
                threshold = group_prob - (group_prob - next_prob) * (step + 1) / (group_size + 1)
            else:
                threshold = group_prob

            pr_points.append((float(threshold), float(precision), float(recall)))
            precisions_for_ap.append(precision)
            recalls_for_ap.append(recall)

    precisions_for_ap = np.array(precisions_for_ap)
    recalls_for_ap = np.array(recalls_for_ap)

    for i in range(len(precisions_for_ap) - 2, -1, -1):
        precisions_for_ap[i] = max(precisions_for_ap[i], precisions_for_ap[i + 1])

    ap = 0.0
    for i in range(1, len(recalls_for_ap)):
        ap += precisions_for_ap[i] * (recalls_for_ap[i] - recalls_for_ap[i - 1])

    return pr_points, float(ap)


def calculate_f1_curve(y_true: List[int], y_prob: List[float]) -> List[Tuple[float, float, float, float]]:
    """
    计算二分类的F1分数随阈值变化的曲线

    Args:
        y_true: 真实标签列表，取值为0或1
        y_prob: 预测概率列表，取值范围[0,1]

    Returns:
        f1_points: 列表，每个元素为(阈值, 精确率, 召回率, F1分数)的元组
    """
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    n = len(y_true)
    total_positives = int(np.sum(y_true == 1))

    if total_positives == 0:
        return [(1.0, 1.0, 0.0, 0.0)]

    sort_key = np.lexsort((-y_true, -y_prob))
    y_true_sorted = y_true[sort_key]
    y_prob_sorted = y_prob[sort_key]

    groups = []
    i = 0
    while i < n:
        j = i
        while j < n and y_prob_sorted[j] == y_prob_sorted[i]:
            j += 1
        groups.append((i, j))
        i = j

    unique_probs = [y_prob_sorted[g[0]] for g in groups]

    f1_points = []
    tp = 0
    fp = 0

    for group_idx, (start, end) in enumerate(groups):
        group_size = end - start
        group_prob = unique_probs[group_idx]
        next_prob = unique_probs[group_idx + 1] if group_idx + 1 < len(groups) else 0.0

        for step in range(group_size):
            k = start + step
            if y_true_sorted[k] == 1:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp)
            recall = tp / total_positives
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if group_size > 1:
                threshold = group_prob - (group_prob - next_prob) * (step + 1) / (group_size + 1)
            else:
                threshold = group_prob

            f1_points.append((float(threshold), float(precision), float(recall), float(f1)))

    return f1_points


def calculate_multiclass_pr_curve(
    y_true: List[int],
    y_probs: List[List[float]],
    average: str = "macro"
) -> Union[Dict[int, Tuple[List[Tuple[float, float, float]], float]], Tuple[List[Tuple[float, float, float]], float]]:
    """
    计算多分类的精确率-召回率曲线（PR曲线）和平均精度（mAP）

    使用一对多（One-vs-Rest）策略，支持微平均和宏平均：
    - macro: 逐类别计算PR曲线，然后取平均
    - micro: 合并所有类别计算全局PR曲线
    - None: 返回每个类别的PR曲线字典

    Args:
        y_true: 真实标签列表，取值为0, 1, 2, ..., n_classes-1
        y_probs: 预测概率矩阵，形状为 [n_samples, n_classes]
        average: 平均策略，可选 "macro", "micro", None

    Returns:
        若 average=None: 字典 {class_id: (pr_points, ap)}
        若 average="macro" 或 "micro": (pr_points, mAP)
    """
    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)

    n_classes = y_probs.shape[1]
    classes = np.unique(y_true)

    if average is None:
        results = {}
        for c in range(n_classes):
            y_true_binary = (y_true == c).astype(int)
            y_prob_binary = y_probs[:, c]
            pr_points, ap = calculate_pr_curve(y_true_binary, y_prob_binary)
            results[c] = (pr_points, ap)
        return results

    elif average == "macro":
        class_pr_curves = []
        class_aps = []
        all_recalls = set()

        for c in range(n_classes):
            y_true_binary = (y_true == c).astype(int)
            y_prob_binary = y_probs[:, c]
            pr_points, ap = calculate_pr_curve(y_true_binary, y_prob_binary)
            class_pr_curves.append(pr_points)
            class_aps.append(ap)
            for _, _, recall in pr_points:
                all_recalls.add(recall)

        sorted_recalls = sorted(all_recalls)
        macro_precisions = []

        for recall_target in sorted_recalls:
            precisions_at_recall = []
            for pr_points in class_pr_curves:
                precision_at_recall = 0.0
                for _, precision, recall in pr_points:
                    if recall >= recall_target:
                        precision_at_recall = max(precision_at_recall, precision)
                precisions_at_recall.append(precision_at_recall)
            macro_precisions.append(np.mean(precisions_at_recall))

        pr_points = [(1.0 - i / len(sorted_recalls), p, r) for i, (p, r) in enumerate(zip(macro_precisions, sorted_recalls))]
        map_score = np.mean(class_aps)

        for i in range(len(macro_precisions) - 2, -1, -1):
            macro_precisions[i] = max(macro_precisions[i], macro_precisions[i + 1])

        ap = 0.0
        for i in range(1, len(sorted_recalls)):
            ap += macro_precisions[i] * (sorted_recalls[i] - sorted_recalls[i - 1])

        return pr_points, float(ap)

    elif average == "micro":
        all_tp = []
        all_fp = []
        all_thresholds = []

        for c in range(n_classes):
            y_true_binary = (y_true == c).astype(int)
            y_prob_binary = y_probs[:, c]

            sort_key = np.lexsort((-y_true_binary, -y_prob_binary))
            yt_sorted = y_true_binary[sort_key]
            yp_sorted = y_prob_binary[sort_key]

            tp = 0
            fp = 0
            for i in range(len(yt_sorted)):
                if yt_sorted[i] == 1:
                    tp += 1
                else:
                    fp += 1
                all_tp.append(tp)
                all_fp.append(fp)
                all_thresholds.append(yp_sorted[i])

        sort_idx = np.argsort(-np.array(all_thresholds))
        all_tp = np.array(all_tp)[sort_idx]
        all_fp = np.array(all_fp)[sort_idx]
        all_thresholds = np.array(all_thresholds)[sort_idx]

        total_positives = np.sum([np.sum(y_true == c) for c in range(n_classes)])

        pr_points = []
        precisions_for_ap = []
        recalls_for_ap = []

        for i in range(len(all_tp)):
            precision = all_tp[i] / (all_tp[i] + all_fp[i]) if (all_tp[i] + all_fp[i]) > 0 else 1.0
            recall = all_tp[i] / total_positives
            pr_points.append((float(all_thresholds[i]), float(precision), float(recall)))
            precisions_for_ap.append(precision)
            recalls_for_ap.append(recall)

        precisions_for_ap = np.array(precisions_for_ap)
        recalls_for_ap = np.array(recalls_for_ap)

        for i in range(len(precisions_for_ap) - 2, -1, -1):
            precisions_for_ap[i] = max(precisions_for_ap[i], precisions_for_ap[i + 1])

        ap = 0.0
        for i in range(1, len(recalls_for_ap)):
            ap += precisions_for_ap[i] * (recalls_for_ap[i] - recalls_for_ap[i - 1])

        return pr_points, float(ap)

    else:
        raise ValueError(f"Unknown average strategy: {average}")


def calculate_multiclass_f1_curve(
    y_true: List[int],
    y_probs: List[List[float]],
    average: str = "macro"
) -> Union[Dict[int, List[Tuple[float, float, float, float]]], List[Tuple[float, float, float, float]]]:
    """
    计算多分类的F1分数随阈值变化的曲线

    Args:
        y_true: 真实标签列表
        y_probs: 预测概率矩阵
        average: 平均策略，可选 "macro", "micro", None

    Returns:
        F1曲线数据
    """
    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    n_classes = y_probs.shape[1]

    if average is None:
        results = {}
        for c in range(n_classes):
            y_true_binary = (y_true == c).astype(int)
            y_prob_binary = y_probs[:, c]
            f1_points = calculate_f1_curve(y_true_binary, y_prob_binary)
            results[c] = f1_points
        return results

    elif average == "macro":
        all_thresholds = set()
        class_f1_curves = {}

        for c in range(n_classes):
            y_true_binary = (y_true == c).astype(int)
            y_prob_binary = y_probs[:, c]
            f1_points = calculate_f1_curve(y_true_binary, y_prob_binary)
            class_f1_curves[c] = f1_points
            for t, _, _, _ in f1_points:
                all_thresholds.add(t)

        sorted_thresholds = sorted(all_thresholds, reverse=True)
        macro_f1_points = []

        for threshold in sorted_thresholds:
            f1s = []
            for c in range(n_classes):
                f1_at_t = 0.0
                for t, _, _, f1 in class_f1_curves[c]:
                    if t <= threshold:
                        f1_at_t = f1
                        break
                f1s.append(f1_at_t)
            avg_f1 = np.mean(f1s)
            macro_f1_points.append((threshold, 0.0, 0.0, float(avg_f1)))

        return macro_f1_points

    elif average == "micro":
        all_y_true_binary = []
        all_y_prob_binary = []

        for c in range(n_classes):
            all_y_true_binary.extend((y_true == c).astype(int).tolist())
            all_y_prob_binary.extend(y_probs[:, c].tolist())

        return calculate_f1_curve(all_y_true_binary, all_y_prob_binary)

    else:
        raise ValueError(f"Unknown average strategy: {average}")


def main():
    print("=" * 60)
    print("测试1：二分类 - 无重复概率值")
    print("=" * 60)
    y_true_1 = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_prob_1 = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3, 0.2]
    _print_pr_curve(y_true_1, y_prob_1)

    print()
    print("=" * 60)
    print("测试2：二分类 - 含重复概率值（0.7出现3次，0.4出现2次）")
    print("=" * 60)
    y_true_2 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y_prob_2 = [0.9, 0.7, 0.7, 0.7, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1]
    _print_pr_curve(y_true_2, y_prob_2)

    print()
    print("=" * 60)
    print("测试3：二分类 - F1分数曲线")
    print("=" * 60)
    _print_f1_curve(y_true_1, y_prob_1)

    print()
    print("=" * 60)
    print("测试4：多分类（3类）- 逐类别PR曲线")
    print("=" * 60)
    y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_probs_multi = [
        [0.9, 0.08, 0.02],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1],
        [0.3, 0.6, 0.1],
        [0.1, 0.1, 0.8],
        [0.5, 0.3, 0.2],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ]
    _print_multiclass_pr_curve(y_true_multi, y_probs_multi, average=None)

    print()
    print("=" * 60)
    print("测试5：多分类 - Macro平均PR曲线")
    print("=" * 60)
    _print_multiclass_pr_curve(y_true_multi, y_probs_multi, average="macro")

    print()
    print("=" * 60)
    print("测试6：多分类 - Micro平均PR曲线")
    print("=" * 60)
    _print_multiclass_pr_curve(y_true_multi, y_probs_multi, average="micro")

    print()
    print("=" * 60)
    print("测试7：多分类 - F1分数曲线（Macro平均）")
    print("=" * 60)
    _print_multiclass_f1_curve(y_true_multi, y_probs_multi, average="macro")


def _print_pr_curve(y_true, y_prob):
    pr_points, ap = calculate_pr_curve(y_true, y_prob)
    print(f"{'阈值':>10} {'精确率':>10} {'召回率':>10}")
    print("-" * 60)
    for threshold, precision, recall in pr_points:
        print(f"{threshold:>10.4f} {precision:>10.4f} {recall:>10.4f}")
    print(f"平均精度 (AP): {ap:.4f}")


def _print_f1_curve(y_true, y_prob):
    f1_points = calculate_f1_curve(y_true, y_prob)
    print(f"{'阈值':>10} {'精确率':>10} {'召回率':>10} {'F1分数':>10}")
    print("-" * 70)
    for threshold, precision, recall, f1 in f1_points:
        print(f"{threshold:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
    best_f1_idx = np.argmax([p[3] for p in f1_points])
    best = f1_points[best_f1_idx]
    print(f"最佳F1: {best[3]:.4f} (阈值={best[0]:.4f}, 精确率={best[1]:.4f}, 召回率={best[2]:.4f})")


def _print_multiclass_pr_curve(y_true, y_probs, average):
    result = calculate_multiclass_pr_curve(y_true, y_probs, average=average)

    if average is None:
        for class_id, (pr_points, ap) in result.items():
            print(f"\n类别 {class_id}:")
            print(f"{'阈值':>10} {'精确率':>10} {'召回率':>10}")
            print("-" * 60)
            for threshold, precision, recall in pr_points:
                print(f"{threshold:>10.4f} {precision:>10.4f} {recall:>10.4f}")
            print(f"类别 {class_id} AP: {ap:.4f}")
    else:
        pr_points, map_score = result
        print(f"{'阈值':>10} {'精确率':>10} {'召回率':>10}")
        print("-" * 60)
        for i, (threshold, precision, recall) in enumerate(pr_points[:10]):
            print(f"{threshold:>10.4f} {precision:>10.4f} {recall:>10.4f}")
        if len(pr_points) > 10:
            print(f"... 共 {len(pr_points)} 个点")
        print(f"{'macro' if average == 'macro' else 'micro'}-mAP: {map_score:.4f}")


def _print_multiclass_f1_curve(y_true, y_probs, average):
    result = calculate_multiclass_f1_curve(y_true, y_probs, average=average)

    if average is None:
        for class_id, f1_points in result.items():
            print(f"\n类别 {class_id}:")
            print(f"{'阈值':>10} {'精确率':>10} {'召回率':>10} {'F1分数':>10}")
            print("-" * 70)
            for threshold, precision, recall, f1 in f1_points:
                print(f"{threshold:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
    else:
        print(f"{'阈值':>10} {'F1分数':>10}")
        print("-" * 30)
        for i, (threshold, _, _, f1) in enumerate(result[:10]):
            print(f"{threshold:>10.4f} {f1:>10.4f}")
        if len(result) > 10:
            print(f"... 共 {len(result)} 个点")
        best_f1_idx = np.argmax([p[3] for p in result])
        print(f"最佳F1: {result[best_f1_idx][3]:.4f} (阈值={result[best_f1_idx][0]:.4f})")


if __name__ == "__main__":
    main()
