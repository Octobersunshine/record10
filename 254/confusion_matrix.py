import math


def get_classes(y_true, y_pred):
    return sorted(set(y_true) | set(y_pred))


def confusion_matrix_multiclass(y_true, y_pred, labels=None):
    if labels is None:
        labels = get_classes(y_true, y_pred)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t]][label_to_idx[p]] += 1
    return cm, labels


def tp_fp_tn_fn_from_cm(cm, class_idx):
    n = len(cm)
    tp = cm[class_idx][class_idx]
    fp = sum(cm[i][class_idx] for i in range(n)) - tp
    fn = sum(cm[class_idx][j] for j in range(n)) - tp
    tn = sum(cm[i][j] for i in range(n) for j in range(n)) - tp - fp - fn
    return tp, fp, tn, fn


def safe_divide(a, b, zero_value=0.0):
    if b == 0:
        return zero_value
    return a / b


def precision_recall_f1_multiclass(y_true, y_pred, average='macro', zero_division=0.0):
    cm, labels = confusion_matrix_multiclass(y_true, y_pred)
    n = len(labels)
    total = sum(sum(row) for row in cm)
    
    if average == 'micro':
        tp_total = sum(cm[i][i] for i in range(n))
        fp_total = sum(sum(cm[i][j] for i in range(n)) - cm[j][j] for j in range(n))
        fn_total = sum(sum(cm[j][i] for i in range(n)) - cm[j][j] for j in range(n))
        precision = safe_divide(tp_total, tp_total + fp_total, zero_division)
        recall = safe_divide(tp_total, tp_total + fn_total, zero_division)
        f1 = safe_divide(2 * precision * recall, precision + recall, zero_division)
        accuracy = tp_total / total if total else zero_division
        return accuracy, precision, recall, f1, None, None
    
    per_class = []
    support = []
    for i in range(n):
        tp, fp, tn, fn = tp_fp_tn_fn_from_cm(cm, i)
        p = safe_divide(tp, tp + fp, zero_division)
        r = safe_divide(tp, tp + fn, zero_division)
        f = safe_divide(2 * p * r, p + r, zero_division)
        s = sum(cm[i])
        per_class.append((p, r, f))
        support.append(s)
    
    accuracy = sum(cm[i][i] for i in range(n)) / total if total else zero_division
    
    if average == 'macro':
        precision = sum(p for p, r, f in per_class) / n
        recall = sum(r for p, r, f in per_class) / n
        f1 = sum(f for p, r, f in per_class) / n
    elif average == 'weighted':
        precision = sum(p * s for (p, r, f), s in zip(per_class, support)) / total if total else zero_division
        recall = sum(r * s for (p, r, f), s in zip(per_class, support)) / total if total else zero_division
        f1 = sum(f * s for (p, r, f), s in zip(per_class, support)) / total if total else zero_division
    else:
        raise ValueError(f"Unsupported average: {average}")
    
    return accuracy, precision, recall, f1, per_class, support


def classification_report(y_true, y_pred, zero_division=0.0, digits=4):
    cm, labels = confusion_matrix_multiclass(y_true, y_pred)
    n = len(labels)
    total = sum(sum(row) for row in cm)
    
    per_class = []
    support = []
    for i in range(n):
        tp, fp, tn, fn = tp_fp_tn_fn_from_cm(cm, i)
        p = safe_divide(tp, tp + fp, zero_division)
        r = safe_divide(tp, tp + fn, zero_division)
        f = safe_divide(2 * p * r, p + r, zero_division)
        s = sum(cm[i])
        per_class.append((p, r, f))
        support.append(s)
    
    accuracy = sum(cm[i][i] for i in range(n)) / total if total else zero_division
    
    macro_p = sum(p for p, r, f in per_class) / n
    macro_r = sum(r for p, r, f in per_class) / n
    macro_f = sum(f for p, r, f in per_class) / n
    
    weighted_p = sum(p * s for (p, r, f), s in zip(per_class, support)) / total if total else zero_division
    weighted_r = sum(r * s for (p, r, f), s in zip(per_class, support)) / total if total else zero_division
    weighted_f = sum(f * s for (p, r, f), s in zip(per_class, support)) / total if total else zero_division
    
    tp_total = sum(cm[i][i] for i in range(n))
    fp_total = sum(sum(cm[i][j] for i in range(n)) - cm[j][j] for j in range(n))
    fn_total = sum(sum(cm[j][i] for i in range(n)) - cm[j][j] for j in range(n))
    micro_p = safe_divide(tp_total, tp_total + fp_total, zero_division)
    micro_r = safe_divide(tp_total, tp_total + fn_total, zero_division)
    micro_f = safe_divide(2 * micro_p * micro_r, micro_p + micro_r, zero_division)
    
    header = f"{'类别':>8} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'支持数':>10}"
    lines = [header, "-" * len(header)]
    for i, label in enumerate(labels):
        p, r, f = per_class[i]
        s = support[i]
        lines.append(f"{str(label):>8} {p:>{digits+6}.{digits}f} {r:>{digits+6}.{digits}f} {f:>{digits+6}.{digits}f} {s:>10}")
    lines.append("")
    lines.append(f"{'准确率':>8} {'':>10} {'':>10} {accuracy:>{digits+6}.{digits}f} {total:>10}")
    lines.append(f"{'宏平均':>8} {macro_p:>{digits+6}.{digits}f} {macro_r:>{digits+6}.{digits}f} {macro_f:>{digits+6}.{digits}f} {total:>10}")
    lines.append(f"{'微平均':>8} {micro_p:>{digits+6}.{digits}f} {micro_r:>{digits+6}.{digits}f} {micro_f:>{digits+6}.{digits}f} {total:>10}")
    lines.append(f"{'加权平均':>8} {weighted_p:>{digits+6}.{digits}f} {weighted_r:>{digits+6}.{digits}f} {weighted_f:>{digits+6}.{digits}f} {total:>10}")
    
    report_text = "\n".join(lines)
    
    report_dict = {
        str(label): {
            "precision": per_class[i][0],
            "recall": per_class[i][1],
            "f1-score": per_class[i][2],
            "support": support[i],
        }
        for i, label in enumerate(labels)
    }
    report_dict["accuracy"] = accuracy
    report_dict["macro avg"] = {
        "precision": macro_p, "recall": macro_r, "f1-score": macro_f, "support": total,
    }
    report_dict["micro avg"] = {
        "precision": micro_p, "recall": micro_r, "f1-score": micro_f, "support": total,
    }
    report_dict["weighted avg"] = {
        "precision": weighted_p, "recall": weighted_r, "f1-score": weighted_f, "support": total,
    }
    
    return report_text, report_dict


def confusion_matrix_visualization(cm, labels):
    n = len(labels)
    total = sum(sum(row) for row in cm)
    cm_normalized = [
        [cell / sum(row) if sum(row) else 0.0 for cell in row]
        for row in cm
    ]
    return {
        "matrix": cm,
        "matrix_normalized": cm_normalized,
        "labels": labels,
        "xticklabels": [str(l) for l in labels],
        "yticklabels": [str(l) for l in labels],
        "title": "Confusion Matrix",
        "xlabel": "Predicted Label",
        "ylabel": "True Label",
        "total_samples": total,
        "annotations": [
            [f"{cm[i][j]}\n({cm_normalized[i][j]:.2%})" for j in range(n)]
            for i in range(n)
        ],
    }


def evaluate(y_true, y_pred, average='macro', zero_division=0.0):
    cm, labels = confusion_matrix_multiclass(y_true, y_pred)
    n = len(labels)
    
    print("混淆矩阵 (行=真实, 列=预测):")
    print(f"  标签: {labels}")
    for i, row in enumerate(cm):
        print(f"  类{labels[i]}: {row}")
    
    if n == 2:
        tp, fp, tn, fn = tp_fp_tn_fn_from_cm(cm, 1)
        specificity = safe_divide(tn, tn + fp, zero_division)
        print(f"\n二分类混淆矩阵(以类{labels[1]}为正类):")
        print(f"  TP = {tp}  FP = {fp}")
        print(f"  FN = {fn}  TN = {tn}")
        print(f"  特异度 (Specificity) = {specificity:.4f}")
    
    accuracy, precision, recall, f1, per_class, support = precision_recall_f1_multiclass(
        y_true, y_pred, average=average, zero_division=zero_division
    )
    
    print(f"\n衍生指标 (average={average}):")
    print(f"  准确率 (Accuracy)    = {accuracy:.4f}")
    print(f"  精确率 (Precision)   = {precision:.4f}")
    print(f"  召回率 (Recall)      = {recall:.4f}")
    print(f"  F1分数 (F1 Score)    = {f1:.4f}")
    
    if per_class:
        print(f"\n各类别指标:")
        for i, label in enumerate(labels):
            p, r, f = per_class[i]
            print(f"  类{label}: P={p:.4f}, R={r:.4f}, F1={f:.4f}, 支持数={support[i]}")
    
    report_text, report_dict = classification_report(y_true, y_pred, zero_division=zero_division)
    print(f"\n分类报告:")
    print(report_text)
    
    viz = confusion_matrix_visualization(cm, labels)
    
    return {
        "confusion_matrix": cm,
        "labels": labels,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
        "support": support,
        "classification_report": report_dict,
        "visualization": viz,
    }


if __name__ == "__main__":
    print("=" * 50)
    print("测试1: 二分类")
    print("=" * 50)
    y_true1 = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred1 = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    result = evaluate(y_true1, y_pred1, average='macro')
    print(f"\n可视化数据结构 keys: {list(result['visualization'].keys())}")
    
    print("\n" + "=" * 50)
    print("测试2: 除零场景 (某类完全未预测到)")
    print("=" * 50)
    y_true2 = [0, 0, 0, 0]
    y_pred2 = [0, 0, 0, 0]
    evaluate(y_true2, y_pred2, average='macro')
    
    print("\n" + "=" * 50)
    print("测试3: 多分类 (3类), macro 平均")
    print("=" * 50)
    y_true3 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred3 = [0, 2, 1, 0, 0, 2, 0, 1, 2]
    result3 = evaluate(y_true3, y_pred3, average='macro')
    
    print("\n" + "=" * 50)
    print("测试4: 多分类 weighted 平均")
    print("=" * 50)
    evaluate(y_true3, y_pred3, average='weighted')
    
    print("\n" + "=" * 50)
    print("测试5: 多分类 micro 平均")
    print("=" * 50)
    evaluate(y_true3, y_pred3, average='micro')
    
    print("\n" + "=" * 50)
    print("测试6: 直接调用 classification_report")
    print("=" * 50)
    report_text, report_dict = classification_report(y_true3, y_pred3)
    print(report_text)
    
    print("\n" + "=" * 50)
    print("测试7: zero_division=nan")
    print("=" * 50)
    y_true4 = [1, 1, 1, 1]
    y_pred4 = [0, 0, 0, 0]
    evaluate(y_true4, y_pred4, zero_division=float('nan'))
