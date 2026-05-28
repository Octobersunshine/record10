import numpy as np
from typing import Optional, Tuple, Union, List


def shuffle_data(data: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(data.shape[0])
    return data[indices]


def shuffle_arrays(*arrays: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    if len(arrays) == 0:
        raise ValueError("至少需要提供一个数组")
    
    n = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n:
            raise ValueError(f"数组 {i} 的样本数 ({arr.shape[0]}) 与第一个数组 ({n}) 不一致")
    
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(n)
    
    return tuple(arr[indices] for arr in arrays)


def random_sample(
    data: np.ndarray,
    size: Optional[Union[int, float]] = None,
    replace: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    n = data.shape[0]
    
    if size is None:
        size = n
    elif isinstance(size, float):
        if not (0.0 < size <= 1.0):
            raise ValueError("比例参数必须在 (0, 1] 范围内")
        size = int(n * size)
    elif isinstance(size, int):
        if size <= 0:
            raise ValueError("抽样数量必须大于 0")
        if not replace and size > n:
            raise ValueError("无放回抽样时数量不能超过样本总数")
    else:
        raise TypeError("size 必须是 int 或 float 类型")
    
    indices = np.random.choice(n, size=size, replace=replace)
    return data[indices]


def random_sample_arrays(
    *arrays: np.ndarray,
    size: Optional[Union[int, float]] = None,
    replace: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    if len(arrays) == 0:
        raise ValueError("至少需要提供一个数组")
    
    n = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n:
            raise ValueError(f"数组 {i} 的样本数 ({arr.shape[0]}) 与第一个数组 ({n}) 不一致")
    
    if seed is not None:
        np.random.seed(seed)
    
    if size is None:
        size = n
    elif isinstance(size, float):
        if not (0.0 < size <= 1.0):
            raise ValueError("比例参数必须在 (0, 1] 范围内")
        size = int(n * size)
    elif isinstance(size, int):
        if size <= 0:
            raise ValueError("抽样数量必须大于 0")
        if not replace and size > n:
            raise ValueError("无放回抽样时数量不能超过样本总数")
    else:
        raise TypeError("size 必须是 int 或 float 类型")
    
    indices = np.random.choice(n, size=size, replace=replace)
    
    return tuple(arr[indices] for arr in arrays)


def _resolve_size(n: int, size: Optional[Union[int, float]]) -> int:
    if size is None:
        return n
    if isinstance(size, float):
        if not (0.0 < size <= 1.0):
            raise ValueError("比例参数必须在 (0, 1] 范围内")
        return int(n * size)
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("抽样数量必须大于 0")
        return size
    raise TypeError("size 必须是 int 或 float 类型")


def stratified_sample(
    *arrays: np.ndarray,
    labels: np.ndarray,
    size: Optional[Union[int, float]] = None,
    replace: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    if len(arrays) == 0:
        raise ValueError("至少需要提供一个数组")

    n = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n:
            raise ValueError(f"数组 {i} 的样本数 ({arr.shape[0]}) 与第一个数组 ({n}) 不一致")
    if labels.shape[0] != n:
        raise ValueError(f"标签数组样本数 ({labels.shape[0]}) 与数据数组 ({n}) 不一致")

    sample_size = _resolve_size(n, size)
    if not replace and sample_size > n:
        raise ValueError("无放回抽样时数量不能超过样本总数")

    if seed is not None:
        np.random.seed(seed)

    unique_classes, class_counts = np.unique(labels, return_counts=True)
    class_ratios = class_counts / n
    per_class_counts = np.round(class_ratios * sample_size).astype(int)

    diff = sample_size - per_class_counts.sum()
    if diff != 0:
        order = np.argsort(-class_ratios)
        for i in range(abs(diff)):
            idx = order[i % len(order)]
            per_class_counts[idx] += 1 if diff > 0 else -1
        per_class_counts = np.maximum(per_class_counts, 0)

    if not replace:
        for i, cnt in enumerate(per_class_counts):
            if cnt > class_counts[i]:
                raise ValueError(
                    f"类别 '{unique_classes[i]}' 需要抽样 {cnt} 个，"
                    f"但仅有 {class_counts[i]} 个样本（无放回）"
                )

    selected_indices = []
    for cls, cnt in zip(unique_classes, per_class_counts):
        cls_indices = np.where(labels == cls)[0]
        chosen = np.random.choice(cls_indices, size=cnt, replace=replace)
        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)

    return tuple(arr[selected_indices] for arr in arrays)


def weighted_sample(
    *arrays: np.ndarray,
    weights: np.ndarray,
    size: Optional[Union[int, float]] = None,
    replace: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    if len(arrays) == 0:
        raise ValueError("至少需要提供一个数组")

    n = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n:
            raise ValueError(f"数组 {i} 的样本数 ({arr.shape[0]}) 与第一个数组 ({n}) 不一致")
    if weights.shape[0] != n:
        raise ValueError(f"权重数组长度 ({weights.shape[0]}) 与数据数组 ({n}) 不一致")

    sample_size = _resolve_size(n, size)
    if not replace and sample_size > n:
        raise ValueError("无放回抽样时数量不能超过样本总数")

    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("权重总和必须大于 0")
    probs = weights.astype(np.float64) / weight_sum

    if seed is not None:
        np.random.seed(seed)

    indices = np.random.choice(n, size=sample_size, replace=replace, p=probs)

    return tuple(arr[indices] for arr in arrays)


def stratified_train_test_split(
    *arrays: np.ndarray,
    labels: np.ndarray,
    test_size: Union[int, float] = 0.2,
    seed: Optional[int] = None,
) -> Tuple:
    if len(arrays) == 0:
        raise ValueError("至少需要提供一个数组")

    n = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n:
            raise ValueError(f"数组 {i} 的样本数 ({arr.shape[0]}) 与第一个数组 ({n}) 不一致")
    if labels.shape[0] != n:
        raise ValueError(f"标签数组样本数 ({labels.shape[0]}) 与数据数组 ({n}) 不一致")

    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("测试集比例必须在 (0, 1) 范围内")
        test_count = int(n * test_size)
    elif isinstance(test_size, int):
        if test_size <= 0 or test_size >= n:
            raise ValueError("测试集数量必须在 (0, 样本总数) 范围内")
        test_count = test_size
    else:
        raise TypeError("test_size 必须是 int 或 float 类型")

    if seed is not None:
        np.random.seed(seed)

    unique_classes, class_counts = np.unique(labels, return_counts=True)
    class_ratios = class_counts / n
    test_per_class = np.round(class_ratios * test_count).astype(int)

    diff = test_count - test_per_class.sum()
    if diff != 0:
        order = np.argsort(-class_ratios)
        for i in range(abs(diff)):
            idx = order[i % len(order)]
            test_per_class[idx] += 1 if diff > 0 else -1
        test_per_class = np.maximum(test_per_class, 0)

    train_indices = []
    test_indices = []
    for cls, cnt in zip(unique_classes, test_per_class):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        test_indices.extend(cls_indices[:cnt])
        train_indices.extend(cls_indices[cnt:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    result = []
    for arr in arrays:
        result.append(arr[train_indices])
        result.append(arr[test_indices])

    return tuple(result)


def shuffle_and_sample(
    data: np.ndarray,
    size: Optional[Union[int, float]] = None,
    replace: bool = False,
    shuffle_first: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    if shuffle_first:
        data = shuffle_data(data, seed=seed)
    return random_sample(data, size=size, replace=replace, seed=seed)


def train_test_split(
    *arrays: np.ndarray,
    test_size: Union[int, float] = 0.2,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple:
    if len(arrays) == 0:
        raise ValueError("至少需要提供一个数组")
    
    n = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n:
            raise ValueError(f"数组 {i} 的样本数 ({arr.shape[0]}) 与第一个数组 ({n}) 不一致")
    
    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("测试集比例必须在 (0, 1) 范围内")
        test_count = int(n * test_size)
    elif isinstance(test_size, int):
        if test_size <= 0 or test_size >= n:
            raise ValueError("测试集数量必须在 (0, 样本总数) 范围内")
        test_count = test_size
    else:
        raise TypeError("test_size 必须是 int 或 float 类型")
    
    if shuffle:
        arrays = shuffle_arrays(*arrays, seed=seed)
    
    result = []
    for arr in arrays:
        train_data = arr[:-test_count]
        test_data = arr[-test_count:]
        result.extend([train_data, test_data])
    
    return tuple(result)


if __name__ == "__main__":
    data = np.arange(20).reshape(10, 2)
    print("原始数据:")
    print(data)
    print()
    
    print("=" * 60)
    print("1. 单数组随机洗牌:")
    shuffled = shuffle_data(data, seed=42)
    print(shuffled)
    print()
    
    print("=" * 60)
    print("2. 无放回抽样 (数量=5):")
    sample_no_replace = random_sample(data, size=5, replace=False, seed=42)
    print(sample_no_replace)
    print()
    
    print("=" * 60)
    print("3. 无放回抽样 (比例=0.3):")
    sample_ratio = random_sample(data, size=0.3, replace=False, seed=42)
    print(sample_ratio)
    print()
    
    print("=" * 60)
    print("4. 有放回抽样 (数量=15):")
    sample_replace = random_sample(data, size=15, replace=True, seed=42)
    print(sample_replace)
    print()
    
    print("=" * 60)
    print("5. 先洗牌再抽样:")
    result = shuffle_and_sample(data, size=5, replace=False, shuffle_first=True, seed=42)
    print(result)
    print()
    
    print("=" * 60)
    print("6. 多数组联合洗牌 (特征+标签):")
    features = np.arange(20).reshape(10, 2)
    labels = np.arange(10)
    print("原始特征:")
    print(features)
    print("原始标签:")
    print(labels)
    shuffled_feat, shuffled_label = shuffle_arrays(features, labels, seed=42)
    print("洗牌后特征:")
    print(shuffled_feat)
    print("洗牌后标签:")
    print(shuffled_label)
    print("验证对应关系: 特征每行第一个数//2 == 标签:", np.all(shuffled_feat[:, 0] // 2 == shuffled_label))
    print()
    
    print("=" * 60)
    print("7. 多数组联合抽样 (特征+标签, 无放回, 数量=5):")
    sampled_feat, sampled_label = random_sample_arrays(features, labels, size=5, replace=False, seed=42)
    print("抽样后特征:")
    print(sampled_feat)
    print("抽样后标签:")
    print(sampled_label)
    print("验证对应关系: 特征每行第一个数//2 == 标签:", np.all(sampled_feat[:, 0] // 2 == sampled_label))
    print()
    
    print("=" * 60)
    print("8. 多数组训练集/测试集划分 (特征+标签, 80%/20%):")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, seed=42)
    print("训练集特征:", X_train.shape)
    print(X_train)
    print("训练集标签:", y_train.shape)
    print(y_train)
    print("测试集特征:", X_test.shape)
    print(X_test)
    print("测试集标签:", y_test.shape)
    print(y_test)
    print("验证训练集对应关系:", np.all(X_train[:, 0] // 2 == y_train))
    print("验证测试集对应关系:", np.all(X_test[:, 0] // 2 == y_test))
    print()

    print("=" * 60)
    print("9. 分层抽样 (保持各类别比例):")
    features = np.arange(30).reshape(15, 2)
    labels = features[:, 0] // 10
    print("原始标签:", labels)
    print("各类别数量:", dict(zip(*np.unique(labels, return_counts=True))))
    sampled_feat, sampled_lab = stratified_sample(features, labels, labels=labels, size=6, replace=False, seed=42)
    print("抽样后标签:", sampled_lab)
    print("抽样后各类别数量:", dict(zip(*np.unique(sampled_lab, return_counts=True))))
    print("验证对应关系:", np.all(sampled_feat[:, 0] // 10 == sampled_lab))
    print()

    print("=" * 60)
    print("10. 分层抽样 (按比例0.5):")
    sampled_feat2, sampled_lab2 = stratified_sample(features, labels, labels=labels, size=0.5, replace=False, seed=42)
    print("抽样后标签:", sampled_lab2)
    print("抽样后各类别数量:", dict(zip(*np.unique(sampled_lab2, return_counts=True))))
    print()

    print("=" * 60)
    print("11. 加权抽样 (自定义概率):")
    features = np.arange(10).reshape(5, 2)
    labels_w = np.arange(5)
    weights = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
    print("原始数据:")
    print(features)
    print("样本权重:", weights)
    sampled_feat3, sampled_lab3 = weighted_sample(features, labels_w, weights=weights, size=10, replace=True, seed=42)
    print("有放回加权抽样后 (数量=10):")
    print(sampled_feat3)
    print("抽样后标签:", sampled_lab3)
    unique, counts = np.unique(sampled_lab3, return_counts=True)
    print("各样本被抽中次数:", dict(zip(unique, counts)))
    print()

    print("=" * 60)
    print("12. 加权抽样 (无放回, 数量=3):")
    sampled_feat4, sampled_lab4 = weighted_sample(features, labels_w, weights=weights, size=3, replace=False, seed=42)
    print("抽样后特征:", sampled_feat4)
    print("抽样后标签:", sampled_lab4)
    print()

    print("=" * 60)
    print("13. 分层训练集/测试集划分:")
    features = np.arange(30).reshape(15, 2)
    labels_s = features[:, 0] // 10
    print("原始标签:", labels_s)
    print("原始各类别数量:", dict(zip(*np.unique(labels_s, return_counts=True))))
    X_train_s, X_test_s, y_train_s, y_test_s = stratified_train_test_split(
        features, labels_s, labels=labels_s, test_size=0.2, seed=42
    )
    print("训练集标签:", y_train_s, "各类别数量:", dict(zip(*np.unique(y_train_s, return_counts=True))))
    print("测试集标签:", y_test_s, "各类别数量:", dict(zip(*np.unique(y_test_s, return_counts=True))))
    print("验证训练集对应关系:", np.all(X_train_s[:, 0] // 10 == y_train_s))
    print("验证测试集对应关系:", np.all(X_test_s[:, 0] // 10 == y_test_s))
