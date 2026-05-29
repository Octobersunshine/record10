import numpy as np


def label_encode(data, mapping=None, handle_unknown="error"):
    if mapping is None:
        unique_categories = sorted(set(data))
        mapping = {cat: i for i, cat in enumerate(unique_categories)}
    
    encoded = []
    for cat in data:
        if cat in mapping:
            encoded.append(mapping[cat])
        else:
            if handle_unknown == "error":
                raise ValueError(f"未知类别: {cat}")
            elif handle_unknown == "ignore":
                encoded.append(-1)
            elif handle_unknown == "unknown":
                if "___UNKNOWN___" not in mapping:
                    mapping["___UNKNOWN___"] = len(mapping)
                encoded.append(mapping["___UNKNOWN___"])
            else:
                raise ValueError(f"handle_unknown 必须是 'error', 'ignore', 'unknown' 之一")
    
    return np.array(encoded), mapping


def one_hot_encode(data, categories=None, handle_unknown="error"):
    if categories is None:
        categories = sorted(set(data))
    
    category_to_int = {cat: i for i, cat in enumerate(categories)}
    n_samples = len(data)
    n_categories = len(categories)
    
    if handle_unknown == "unknown":
        n_categories += 1
        category_to_int["___UNKNOWN___"] = n_categories - 1
        categories = categories + ["___UNKNOWN___"]
    
    encoded = np.zeros((n_samples, n_categories), dtype=int)
    
    for i, cat in enumerate(data):
        if cat in category_to_int:
            encoded[i, category_to_int[cat]] = 1
        else:
            if handle_unknown == "error":
                raise ValueError(f"未知类别: {cat}")
            elif handle_unknown == "ignore":
                pass
            elif handle_unknown == "unknown":
                encoded[i, category_to_int["___UNKNOWN___"]] = 1
            else:
                raise ValueError(f"handle_unknown 必须是 'error', 'ignore', 'unknown' 之一")
    
    return encoded, categories


def frequency_encode(data, mapping=None, handle_unknown="error"):
    if mapping is None:
        total = len(data)
        counts = {}
        for cat in data:
            counts[cat] = counts.get(cat, 0) + 1
        mapping = {cat: counts[cat] / total for cat in sorted(counts)}
    
    global_mean = sum(mapping.values()) / len(mapping) if mapping else 0.0
    encoded = []
    for cat in data:
        if cat in mapping:
            encoded.append(mapping[cat])
        else:
            if handle_unknown == "error":
                raise ValueError(f"未知类别: {cat}")
            elif handle_unknown == "ignore":
                encoded.append(global_mean)
            elif handle_unknown == "unknown":
                encoded.append(global_mean)
            else:
                raise ValueError(f"handle_unknown 必须是 'error', 'ignore', 'unknown' 之一")
    
    return np.array(encoded), mapping


def target_encode(data, target, mapping=None, handle_unknown="error", smoothing=0):
    if mapping is None:
        category_stats = {}
        for cat, y in zip(data, target):
            if cat not in category_stats:
                category_stats[cat] = {"sum": 0.0, "count": 0}
            category_stats[cat]["sum"] += y
            category_stats[cat]["count"] += 1
        
        global_mean = sum(s["sum"] for s in category_stats.values()) / sum(s["count"] for s in category_stats.values())
        
        mapping = {}
        for cat in sorted(category_stats):
            cat_sum = category_stats[cat]["sum"]
            cat_count = category_stats[cat]["count"]
            cat_mean = cat_sum / cat_count
            if smoothing > 0:
                smoothed = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
                mapping[cat] = smoothed
            else:
                mapping[cat] = cat_mean
    
    global_mean = sum(mapping.values()) / len(mapping) if mapping else 0.0
    
    encoded = []
    for cat in data:
        if cat in mapping:
            encoded.append(mapping[cat])
        else:
            if handle_unknown == "error":
                raise ValueError(f"未知类别: {cat}")
            elif handle_unknown == "ignore":
                encoded.append(global_mean)
            elif handle_unknown == "unknown":
                encoded.append(global_mean)
            else:
                raise ValueError(f"handle_unknown 必须是 'error', 'ignore', 'unknown' 之一")
    
    return np.array(encoded), mapping


if __name__ == "__main__":
    train_data = ["猫", "狗", "鸟", "猫", "鸟", "狗"]
    test_data = ["猫", "狗", "老虎", "鸟", "狮子"]
    
    print("训练集:", train_data)
    print("测试集:", test_data)
    
    print("\n=== 1. 在训练集上 fit ===")
    train_onehot, train_categories = one_hot_encode(train_data)
    print("训练集独热编码:\n", train_onehot)
    print("列名:", train_categories)
    
    print("\n=== 2. handle_unknown='error' (默认，测试集有未知类别时报错) ===")
    try:
        test_onehot, _ = one_hot_encode(test_data, categories=train_categories, handle_unknown="error")
    except ValueError as e:
        print("报错:", e)
    
    print("\n=== 3. handle_unknown='ignore' (未知类别全为0) ===")
    test_onehot_ignore, _ = one_hot_encode(test_data, categories=train_categories, handle_unknown="ignore")
    print("测试集独热编码:\n", test_onehot_ignore)
    print("说明: '老虎'和'狮子'行全为0")
    
    print("\n=== 4. handle_unknown='unknown' (新增未知类别列) ===")
    test_onehot_unknown, new_categories = one_hot_encode(test_data, categories=train_categories, handle_unknown="unknown")
    print("测试集独热编码:\n", test_onehot_unknown)
    print("列名:", new_categories)
    print("说明: '老虎'和'狮子'在最后一列('___UNKNOWN___')为1")

    print("\n" + "=" * 50)
    print("频数编码 & 目标编码 演示")
    print("=" * 50)

    data = ["猫", "狗", "鸟", "猫", "鸟", "狗", "猫", "狗", "狗", "鸟"]
    target = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]

    print("\n数据:", data)
    print("目标:", target)

    print("\n=== 5. 频数编码 ===")
    freq_result, freq_mapping = frequency_encode(data)
    print("编码结果:", freq_result)
    print("映射字典:", freq_mapping)

    print("\n=== 6. 频数编码 - 测试集含未知类别 ===")
    test_data2 = ["猫", "老虎", "鸟"]
    freq_test, _ = frequency_encode(test_data2, mapping=freq_mapping, handle_unknown="ignore")
    print("测试数据:", test_data2)
    print("编码结果:", freq_test)
    print("说明: '老虎'用全局均值填充")

    print("\n=== 7. 目标编码 ===")
    target_result, target_mapping = target_encode(data, target)
    print("编码结果:", target_result)
    print("映射字典:", target_mapping)

    print("\n=== 8. 目标编码 - 带平滑(smoothing=1) ===")
    target_smooth, smooth_mapping = target_encode(data, target, smoothing=1)
    print("编码结果:", target_smooth)
    print("映射字典:", smooth_mapping)
    print("说明: 小样本类别向全局均值收缩，减少过拟合")

    print("\n=== 9. 目标编码 - 测试集含未知类别 ===")
    target_test, _ = target_encode(test_data2, target=[1, 0, 1], mapping=target_mapping, handle_unknown="ignore")
    print("测试数据:", test_data2)
    print("编码结果:", target_test)
