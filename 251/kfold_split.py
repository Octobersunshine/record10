import random
from itertools import combinations
from typing import List, Tuple, Optional
from collections import Counter


def k_fold_split(n_samples: int, k: int, shuffle: bool = True, random_state: int = None) -> List[Tuple[List[int], List[int]]]:
    if k <= 1:
        raise ValueError("k must be greater than 1")
    if k > n_samples:
        raise ValueError("k cannot be greater than the number of samples")
    
    indices = list(range(n_samples))
    
    if shuffle:
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(indices)
    
    folds = [[] for _ in range(k)]
    for idx, sample_idx in enumerate(indices):
        fold_idx = idx % k
        folds[fold_idx].append(sample_idx)
    
    splits = []
    for i in range(k):
        val_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])
        splits.append((train_indices, val_indices))
    
    return splits


def stratified_k_fold_split(labels: List[int], k: int, shuffle: bool = True, random_state: int = None) -> List[Tuple[List[int], List[int]]]:
    n_samples = len(labels)
    
    if k <= 1:
        raise ValueError("k must be greater than 1")
    if k > n_samples:
        raise ValueError("k cannot be greater than the number of samples")
    
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        if count < k:
            raise ValueError(f"Class {label} has only {count} samples, which is less than k={k}")
    
    indices_by_label = {}
    for idx, label in enumerate(labels):
        if label not in indices_by_label:
            indices_by_label[label] = []
        indices_by_label[label].append(idx)
    
    if shuffle:
        if random_state is not None:
            random.seed(random_state)
        for label in indices_by_label:
            random.shuffle(indices_by_label[label])
    
    folds = [[] for _ in range(k)]
    
    for label, indices in indices_by_label.items():
        for idx, sample_idx in enumerate(indices):
            fold_idx = idx % k
            folds[fold_idx].append(sample_idx)
    
    if shuffle:
        if random_state is not None:
            random.seed(random_state + 1 if random_state else None)
        for fold in folds:
            random.shuffle(fold)
    
    splits = []
    for i in range(k):
        val_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])
        splits.append((train_indices, val_indices))
    
    return splits


def loo_split(n_samples: int) -> List[Tuple[List[int], List[int]]]:
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    
    splits = []
    for i in range(n_samples):
        train_indices = list(range(i)) + list(range(i + 1, n_samples))
        val_indices = [i]
        splits.append((train_indices, val_indices))
    
    return splits


def lpo_split(n_samples: int, p: int) -> List[Tuple[List[int], List[int]]]:
    if p < 1:
        raise ValueError("p must be at least 1")
    if p >= n_samples:
        raise ValueError("p must be less than n_samples")
    
    indices = list(range(n_samples))
    splits = []
    
    for val_indices_tuple in combinations(indices, p):
        val_indices = list(val_indices_tuple)
        train_indices = [i for i in indices if i not in val_indices]
        splits.append((train_indices, val_indices))
    
    return splits


def repeated_k_fold_split(n_samples: int, k: int, n_repeats: int, random_state: int = None) -> List[Tuple[List[int], List[int]]]:
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    
    all_splits = []
    
    for repeat_idx in range(n_repeats):
        if random_state is not None:
            current_seed = random_state + repeat_idx * 1000
        else:
            current_seed = None
        
        splits = k_fold_split(n_samples, k, shuffle=True, random_state=current_seed)
        all_splits.extend(splits)
    
    return all_splits


def repeated_stratified_k_fold_split(labels: List[int], k: int, n_repeats: int, random_state: int = None) -> List[Tuple[List[int], List[int]]]:
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    
    all_splits = []
    
    for repeat_idx in range(n_repeats):
        if random_state is not None:
            current_seed = random_state + repeat_idx * 1000
        else:
            current_seed = None
        
        splits = stratified_k_fold_split(labels, k, shuffle=True, random_state=current_seed)
        all_splits.extend(splits)
    
    return all_splits


if __name__ == "__main__":
    print("=" * 60)
    print("测试1: 标准K折交叉验证（10个样本，3折）")
    print("=" * 60)
    n_samples = 10
    k = 3
    
    print(f"样本总数: {n_samples}")
    print(f"K值: {k}")
    print(f"\n{k}折交叉验证划分结果:\n")
    
    splits = k_fold_split(n_samples, k, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"第 {fold_idx} 折:")
        print(f"  训练集索引: {sorted(train_idx)}")
        print(f"  验证集索引: {sorted(val_idx)}")
        print(f"  训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
        print()
    
    print("=" * 60)
    print("测试2: 分层K折交叉验证（分类问题）")
    print("=" * 60)
    
    labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    k = 3
    
    print(f"样本总数: {len(labels)}")
    print(f"标签分布: {Counter(labels)}")
    print(f"K值: {k}")
    print(f"\n分层{k}折交叉验证划分结果:\n")
    
    splits = stratified_k_fold_split(labels, k, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        print(f"第 {fold_idx} 折:")
        print(f"  训练集索引: {sorted(train_idx)}")
        print(f"  验证集索引: {sorted(val_idx)}")
        print(f"  训练集标签分布: {Counter(train_labels)}")
        print(f"  验证集标签分布: {Counter(val_labels)}")
        print()
    
    print("=" * 60)
    print("测试3: 验证各折大小均匀性")
    print("=" * 60)
    
    n_samples = 17
    k = 5
    splits = k_fold_split(n_samples, k, shuffle=False)
    
    print(f"样本总数: {n_samples}")
    print(f"K值: {k}")
    print(f"各折验证集大小: {[len(val) for _, val in splits]}")
    print(f"最大/最小折大小差: {max(len(val) for _, val in splits) - min(len(val) for _, val in splits)}")
    print()
    
    print("=" * 60)
    print("测试4: 留一法交叉验证（LOOCV）")
    print("=" * 60)
    
    n_samples = 5
    splits = loo_split(n_samples)
    
    print(f"样本总数: {n_samples}")
    print(f"划分总数: {len(splits)}")
    print()
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"第 {fold_idx} 折:")
        print(f"  训练集索引: {train_idx}")
        print(f"  验证集索引: {val_idx}")
        print()
    
    print("=" * 60)
    print("测试5: 留P法交叉验证（Leave-P-Out）")
    print("=" * 60)
    
    n_samples = 5
    p = 2
    splits = lpo_split(n_samples, p)
    
    print(f"样本总数: {n_samples}")
    print(f"P值: {p}")
    print(f"划分总数: {len(splits)} (C({n_samples},{p}) = {len(splits)})")
    print()
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"第 {fold_idx} 折:")
        print(f"  训练集索引: {train_idx}")
        print(f"  验证集索引: {val_idx}")
        print()
    
    print("=" * 60)
    print("测试6: 重复K折交叉验证")
    print("=" * 60)
    
    n_samples = 8
    k = 4
    n_repeats = 2
    splits = repeated_k_fold_split(n_samples, k, n_repeats, random_state=42)
    
    print(f"样本总数: {n_samples}")
    print(f"K值: {k}")
    print(f"重复次数: {n_repeats}")
    print(f"总划分数: {len(splits)} ({n_repeats} × {k})")
    print()
    
    for repeat in range(n_repeats):
        print(f"第 {repeat + 1} 次重复:")
        for fold in range(k):
            idx = repeat * k + fold
            train_idx, val_idx = splits[idx]
            print(f"  第 {fold + 1} 折 - 验证集: {sorted(val_idx)}")
        print()
    
    print("=" * 60)
    print("测试7: 验证随机种子可复现性")
    print("=" * 60)
    
    splits1 = repeated_k_fold_split(10, 3, 2, random_state=42)
    splits2 = repeated_k_fold_split(10, 3, 2, random_state=42)
    
    print(f"两次使用相同种子的划分是否一致: {splits1 == splits2}")
    print(f"第一次划分第1折验证集: {sorted(splits1[0][1])}")
    print(f"第二次划分第1折验证集: {sorted(splits2[0][1])}")
