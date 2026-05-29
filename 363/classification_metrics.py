import warnings
import math
from collections import defaultdict


def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for true, pred in zip(y_true, y_pred):
        i = label_to_idx[true]
        j = label_to_idx[pred]
        cm[i][j] += 1
    return cm, labels


def binary_confusion_matrix(y_true, y_pred, pos_label=1):
    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == pos_label and pred == pos_label:
            tp += 1
        elif true != pos_label and pred != pos_label:
            tn += 1
        elif true != pos_label and pred == pos_label:
            fp += 1
        elif true == pos_label and pred != pos_label:
            fn += 1
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def _safe_divide(numerator, denominator, metric_name, zero_division=0.0):
    if denominator == 0:
        if numerator == 0:
            if zero_division == 'warn':
                warnings.warn(
                    f"{metric_name} is ill-defined and being set to 0.0 "
                    f"due to no predictions.",
                    UserWarning,
                    stacklevel=2
                )
                return 0.0
            elif zero_division == 'nan':
                return float('nan')
            else:
                return zero_division
        else:
            return 0.0
    return numerator / denominator


def _compute_binary_metrics(tp, tn, fp, fn, zero_division=0.0):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = _safe_divide(tp, tp + fp, 'Precision', zero_division)
    recall = _safe_divide(tp, tp + fn, 'Recall', zero_division)
    f1 = _safe_divide(2 * precision * recall, precision + recall, 'F1', zero_division) if (precision + recall) > 0 else 0.0
    specificity = _safe_divide(tn, tn + fp, 'Specificity', zero_division)
    return accuracy, precision, recall, f1, specificity


def _compute_metrics_per_class(cm, labels, zero_division=0.0):
    n = len(labels)
    metrics_per_class = []
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        tn = sum(sum(row) for row in cm) - tp - fp - fn
        _, precision, recall, f1, specificity = _compute_binary_metrics(
            tp, tn, fp, fn, zero_division
        )
        metrics_per_class.append({
            'label': labels[i],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'support': sum(cm[i])
        })
    return metrics_per_class


def _micro_average(cm, zero_division=0.0):
    n = len(cm)
    tp_sum = sum(cm[i][i] for i in range(n))
    fp_sum = sum(sum(cm[j][i] for j in range(n)) - cm[i][i] for i in range(n))
    fn_sum = sum(sum(cm[i][j] for j in range(n)) - cm[i][i] for i in range(n))
    total = sum(sum(row) for row in cm)
    
    accuracy = tp_sum / total if total > 0 else 0.0
    precision = _safe_divide(tp_sum, tp_sum + fp_sum, 'Micro Precision', zero_division)
    recall = _safe_divide(tp_sum, tp_sum + fn_sum, 'Micro Recall', zero_division)
    f1 = _safe_divide(2 * precision * recall, precision + recall, 'Micro F1', zero_division) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1


def _macro_average(metrics_per_class):
    n = len(metrics_per_class)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    def safe_avg(values):
        valid = [v for v in values if not math.isnan(v)]
        return sum(valid) / len(valid) if valid else float('nan')
    
    precision = safe_avg([m['precision'] for m in metrics_per_class])
    recall = safe_avg([m['recall'] for m in metrics_per_class])
    f1 = safe_avg([m['f1'] for m in metrics_per_class])
    
    return precision, recall, f1


def _weighted_average(metrics_per_class):
    total_support = sum(m['support'] for m in metrics_per_class)
    if total_support == 0:
        return 0.0, 0.0, 0.0
    
    def safe_weighted_avg(values, weights):
        total_w = 0.0
        total_vw = 0.0
        for v, w in zip(values, weights):
            if not math.isnan(v):
                total_vw += v * w
                total_w += w
        return total_vw / total_w if total_w > 0 else float('nan')
    
    precisions = [m['precision'] for m in metrics_per_class]
    recalls = [m['recall'] for m in metrics_per_class]
    f1s = [m['f1'] for m in metrics_per_class]
    supports = [m['support'] for m in metrics_per_class]
    
    precision = safe_weighted_avg(precisions, supports)
    recall = safe_weighted_avg(recalls, supports)
    f1 = safe_weighted_avg(f1s, supports)
    
    return precision, recall, f1


def classification_metrics(y_true, y_pred, average='binary', pos_label=1, 
                           labels=None, zero_division=0.0):
    if average not in ['binary', 'micro', 'macro', 'weighted', None]:
        raise ValueError(
            f"average must be one of 'binary', 'micro', 'macro', 'weighted', None, "
            f"got {average}"
        )
    
    if zero_division not in [0.0, 1.0, 'warn', 'nan']:
        raise ValueError(
            f"zero_division must be one of 0.0, 1.0, 'warn', 'nan', got {zero_division}"
        )
    
    if average == 'binary':
        cm = binary_confusion_matrix(y_true, y_pred, pos_label)
        tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
        accuracy, precision, recall, f1, specificity = _compute_binary_metrics(
            tp, tn, fp, fn, zero_division
        )
        return {
            'confusion_matrix': cm,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity
            }
        }
    else:
        cm, labels = confusion_matrix(y_true, y_pred, labels)
        total = sum(sum(row) for row in cm)
        accuracy = sum(cm[i][i] for i in range(len(cm))) / total if total > 0 else 0.0
        
        metrics_per_class = _compute_metrics_per_class(cm, labels, zero_division)
        
        if average == 'micro':
            _, precision, recall, f1 = _micro_average(cm, zero_division)
            specificity = float('nan')
        elif average == 'macro':
            precision, recall, f1 = _macro_average(metrics_per_class)
            specificity = float('nan')
        elif average == 'weighted':
            precision, recall, f1 = _weighted_average(metrics_per_class)
            specificity = float('nan')
        else:
            precision = recall = f1 = specificity = float('nan')
        
        result = {
            'confusion_matrix': cm,
            'labels': labels,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity
            }
        }
        if average is None:
            result['metrics_per_class'] = metrics_per_class
        
        return result


def roc_curve(y_true, y_score, pos_label=1):
    desc_score_indices = sorted(range(len(y_score)), key=lambda k: y_score[k], reverse=True)
    y_true_sorted = [y_true[i] for i in desc_score_indices]
    y_score_sorted = [y_score[i] for i in desc_score_indices]
    
    distinct_value_indices = []
    prev = None
    for i, score in enumerate(y_score_sorted):
        if score != prev:
            distinct_value_indices.append(i)
            prev = score
    distinct_value_indices.append(len(y_score_sorted))
    
    tps = [0]
    fps = [0]
    for i in range(len(distinct_value_indices) - 1):
        start = distinct_value_indices[i]
        end = distinct_value_indices[i + 1]
        tp = sum(1 for j in range(start, end) if y_true_sorted[j] == pos_label)
        fp = (end - start) - tp
        tps.append(tps[-1] + tp)
        fps.append(fps[-1] + fp)
    
    if len(tps) < 2:
        tps = [0, tps[-1]]
        fps = [0, fps[-1]]
    
    p = sum(1 for y in y_true if y == pos_label)
    n = len(y_true) - p
    
    if p == 0 or n == 0:
        warnings.warn(
            "ROC curve is not defined when there are no positive or negative samples.",
            UserWarning,
            stacklevel=2
        )
        return [0.0, 1.0], [0.0, 1.0]
    
    fpr = [fp / n for fp in fps]
    tpr = [tp / p for tp in tps]
    
    last_fpr = fpr[-1]
    last_tpr = tpr[-1]
    if last_fpr < 1.0:
        fpr.append(1.0)
        tpr.append(last_tpr)
    
    return fpr, tpr


def auc_score(fpr, tpr):
    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have the same length")
    if len(fpr) < 2:
        return 0.0
    
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (tpr[i] + tpr[i - 1]) * (fpr[i] - fpr[i - 1]) / 2.0
    return max(0.0, min(1.0, auc))


def plot_confusion_matrix(cm, labels=None, normalize=False, title='Confusion Matrix'):
    if isinstance(cm, dict):
        cm_arr = [[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]]
        if labels is None:
            labels = ['0', '1']
    else:
        cm_arr = [row[:] for row in cm]
    
    if labels is None:
        labels = [str(i) for i in range(len(cm_arr))]
    
    n = len(cm_arr)
    
    if normalize:
        row_sums = [sum(row) for row in cm_arr]
        cm_norm = []
        for i in range(n):
            if row_sums[i] == 0:
                cm_norm.append([0.0] * n)
            else:
                cm_norm.append([cm_arr[i][j] / row_sums[i] for j in range(n)])
        display_cm = cm_norm
        fmt = '.2f'
    else:
        display_cm = cm_arr
        fmt = 'd'
    
    heatmap_chars = ' ░▒▓█'
    
    max_val = max(max(row) for row in display_cm) if n > 0 else 1.0
    if max_val == 0:
        max_val = 1.0
    
    label_width = max(len(str(l)) for l in labels)
    cell_width = max(6, label_width + 2)
    
    lines = []
    lines.append(f"{' ' * (label_width + 2)}{title}")
    if normalize:
        lines.append(f"{' ' * (label_width + 2)}(Normalized)")
    
    header = ' ' * (label_width + 1)
    for label in labels:
        header += f"{str(label):>{cell_width}s}"
    lines.append(header)
    lines.append(' ' * (label_width + 1) + '-' * (cell_width * n))
    
    for i in range(n):
        row_str = f"{str(labels[i]):>{label_width}s} |"
        for j in range(n):
            val = display_cm[i][j]
            if normalize:
                val_str = f"{val:{fmt}}"
            else:
                val_str = f"{int(val):{fmt}}"
            
            intensity = int(val / max_val * (len(heatmap_chars) - 1))
            block = heatmap_chars[intensity] * 2
            row_str += f" {block} {val_str:>{cell_width - 4}s}"
        lines.append(row_str)
    
    return '\n'.join(lines)


def classification_report(y_true, y_pred, labels=None, target_names=None, 
                          zero_division=0.0, digits=4):
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    metrics_per_class = _compute_metrics_per_class(cm, labels, zero_division)
    
    if target_names is None:
        target_names = [str(l) for l in labels]
    
    if len(target_names) != len(labels):
        raise ValueError("target_names length must match number of labels")
    
    header = f"{'':<12s}{'precision':>{digits + 6}s}{'recall':>{digits + 6}s}" \
             f"{'f1':>{digits + 6}s}{'support':>{digits + 6}s}"
    
    lines = []
    lines.append(header)
    lines.append('-' * len(header))
    
    total_support = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    weighted_f1_sum = 0.0
    correct = 0
    
    for i, m in enumerate(metrics_per_class):
        name = target_names[i]
        prec = m['precision']
        rec = m['recall']
        f1 = m['f1']
        sup = m['support']
        
        total_support += sup
        correct += cm[i][i]
        
        if not math.isnan(prec):
            precision_sum += prec
            weighted_precision_sum += prec * sup
        if not math.isnan(rec):
            recall_sum += rec
            weighted_recall_sum += rec * sup
        if not math.isnan(f1):
            f1_sum += f1
            weighted_f1_sum += f1 * sup
        
        prec_str = f"{prec:.{digits}f}" if not math.isnan(prec) else f"{'nan':>{digits + 2}s}"
        rec_str = f"{rec:.{digits}f}" if not math.isnan(rec) else f"{'nan':>{digits + 2}s}"
        f1_str = f"{f1:.{digits}f}" if not math.isnan(f1) else f"{'nan':>{digits + 2}s}"
        
        lines.append(f"{name:<12s}{prec_str:>{digits + 6}s}{rec_str:>{digits + 6}s}"
                     f"{f1_str:>{digits + 6}s}{sup:>{digits + 6}d}")
    
    lines.append('-' * len(header))
    
    accuracy = correct / total_support if total_support > 0 else 0.0
    n_classes = len(metrics_per_class)
    
    macro_prec = precision_sum / n_classes if n_classes > 0 else 0.0
    macro_rec = recall_sum / n_classes if n_classes > 0 else 0.0
    macro_f1 = f1_sum / n_classes if n_classes > 0 else 0.0
    
    weighted_prec = weighted_precision_sum / total_support if total_support > 0 else 0.0
    weighted_rec = weighted_recall_sum / total_support if total_support > 0 else 0.0
    weighted_f1 = weighted_f1_sum / total_support if total_support > 0 else 0.0
    
    acc_str = f"{accuracy:.{digits}f}"
    lines.append(f"{'accuracy':<12s}{'':>{digits + 6}s}{'':>{digits + 6}s}"
                 f"{acc_str:>{digits + 6}s}{total_support:>{digits + 6}d}")
    
    macro_prec_str = f"{macro_prec:.{digits}f}"
    macro_rec_str = f"{macro_rec:.{digits}f}"
    macro_f1_str = f"{macro_f1:.{digits}f}"
    lines.append(f"{'macro avg':<12s}{macro_prec_str:>{digits + 6}s}"
                 f"{macro_rec_str:>{digits + 6}s}"
                 f"{macro_f1_str:>{digits + 6}s}{total_support:>{digits + 6}d}")
    
    weighted_prec_str = f"{weighted_prec:.{digits}f}"
    weighted_rec_str = f"{weighted_rec:.{digits}f}"
    weighted_f1_str = f"{weighted_f1:.{digits}f}"
    lines.append(f"{'weighted avg':<12s}{weighted_prec_str:>{digits + 6}s}"
                 f"{weighted_rec_str:>{digits + 6}s}"
                 f"{weighted_f1_str:>{digits + 6}s}{total_support:>{digits + 6}d}")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    print("=" * 60)
    print("测试1: 普通二分类")
    print("=" * 60)
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    result = classification_metrics(y_true, y_pred, average='binary')
    print("混淆矩阵:", result['confusion_matrix'])
    for k, v in result['metrics'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "=" * 60)
    print("测试2: 除零情况 - 没有正类预测 (TP+FP=0)")
    print("=" * 60)
    y_true2 = [1, 1, 1, 1]
    y_pred2 = [0, 0, 0, 0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result2 = classification_metrics(y_true2, y_pred2, average='binary', zero_division='warn')
        if w:
            print(f"警告: {w[-1].message}")
    print("混淆矩阵:", result2['confusion_matrix'])
    for k, v in result2['metrics'].items():
        print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: NaN")
    
    print("\n" + "=" * 60)
    print("测试3: zero_division='nan' 返回NaN")
    print("=" * 60)
    result3 = classification_metrics(y_true2, y_pred2, average='binary', zero_division='nan')
    for k, v in result3['metrics'].items():
        print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: NaN")
    
    print("\n" + "=" * 60)
    print("测试4: 多分类 - micro average")
    print("=" * 60)
    y_true3 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred3 = [0, 0, 2, 0, 2, 1, 0, 1, 2]
    result4 = classification_metrics(y_true3, y_pred3, average='micro')
    print("混淆矩阵:", result4['confusion_matrix'])
    print("标签:", result4['labels'])
    for k, v in result4['metrics'].items():
        print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: NaN")
    
    print("\n" + "=" * 60)
    print("测试5: 多分类 - macro average")
    print("=" * 60)
    result5 = classification_metrics(y_true3, y_pred3, average='macro')
    for k, v in result5['metrics'].items():
        print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: NaN")
    
    print("\n" + "=" * 60)
    print("测试6: 多分类 - weighted average")
    print("=" * 60)
    result6 = classification_metrics(y_true3, y_pred3, average='weighted')
    for k, v in result6['metrics'].items():
        print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: NaN")
    
    print("\n" + "=" * 60)
    print("测试7: 多分类 - average=None (逐类指标)")
    print("=" * 60)
    result7 = classification_metrics(y_true3, y_pred3, average=None)
    for m in result7['metrics_per_class']:
        print(f"类别 {m['label']}:")
        for k, v in m.items():
            if k != 'label':
                print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: NaN")
    
    print("\n" + "=" * 60)
    print("测试8: 除零 - 某类完全没有预测 (macro)")
    print("=" * 60)
    y_true4 = [0, 0, 1, 1]
    y_pred4 = [0, 0, 0, 0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result8 = classification_metrics(y_true4, y_pred4, average='macro', zero_division='warn')
        if w:
            for warning in w:
                print(f"警告: {warning.message}")
    for k, v in result8['metrics'].items():
        print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: NaN")
    
    print("\n" + "=" * 60)
    print("测试9: ROC曲线和AUC计算")
    print("=" * 60)
    y_true9 = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_score9 = [0.9, 0.1, 0.8, 0.3, 0.2, 0.7, 0.6, 0.4, 0.85, 0.05]
    fpr, tpr = roc_curve(y_true9, y_score9)
    auc_val = auc_score(fpr, tpr)
    print("FPR  TPR")
    for f, t in zip(fpr, tpr):
        print(f"{f:.4f} {t:.4f}")
    print(f"\nAUC: {auc_val:.4f}")
    
    print("\n" + "=" * 60)
    print("测试10: 二分类混淆矩阵可视化 (原始值)")
    print("=" * 60)
    cm_binary = binary_confusion_matrix(y_true, y_pred)
    print(plot_confusion_matrix(cm_binary, labels=['Negative', 'Positive']))
    
    print("\n" + "=" * 60)
    print("测试11: 二分类混淆矩阵可视化 (归一化)")
    print("=" * 60)
    print(plot_confusion_matrix(cm_binary, labels=['Negative', 'Positive'], normalize=True))
    
    print("\n" + "=" * 60)
    print("测试12: 多分类混淆矩阵可视化 (原始值)")
    print("=" * 60)
    cm_multi, labels_multi = confusion_matrix(y_true3, y_pred3)
    print(plot_confusion_matrix(cm_multi, labels=['Class 0', 'Class 1', 'Class 2']))
    
    print("\n" + "=" * 60)
    print("测试13: 多分类混淆矩阵可视化 (归一化)")
    print("=" * 60)
    print(plot_confusion_matrix(cm_multi, labels=['Class 0', 'Class 1', 'Class 2'], normalize=True))
    
    print("\n" + "=" * 60)
    print("测试14: 分类报告 (二分类)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    print("\n" + "=" * 60)
    print("测试15: 分类报告 (多分类)")
    print("=" * 60)
    print(classification_report(y_true3, y_pred3, 
                          target_names=['Class A', 'Class B', 'Class C']))
    
    print("\n" + "=" * 60)
    print("测试16: ROC边界情况 - 完美分类")
    print("=" * 60)
    y_true10 = [1, 1, 1, 0, 0, 0]
    y_score10 = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    fpr10, tpr10 = roc_curve(y_true10, y_score10)
    auc10 = auc_score(fpr10, tpr10)
    print("完美分类 AUC:", auc10)
    
    print("\n" + "=" * 60)
    print("测试17: ROC边界情况 - 随机分类")
    print("=" * 60)
    y_true11 = [1, 1, 0, 0]
    y_score11 = [0.6, 0.4, 0.5, 0.3]
    fpr11, tpr11 = roc_curve(y_true11, y_score11)
    auc11 = auc_score(fpr11, tpr11)
    print("随机分类 AUC:", auc11)
