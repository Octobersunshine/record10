import numpy as np
from collections import Counter


def _find_k_neighbors_all(X, k):
    n_samples = X.shape[0]
    neighbors = np.zeros((n_samples, k), dtype=int)
    for i in range(n_samples):
        diff = X - X[i]
        distances = np.einsum('ij,ij->i', diff, diff)
        distances[i] = np.inf
        neighbors[i] = np.argsort(distances)[:k]
    return neighbors


def _find_k_neighbors(X, k, rng):
    n_samples = X.shape[0]
    neighbors = np.zeros((n_samples, k), dtype=int)
    for i in range(n_samples):
        diff = X - X[i]
        distances = np.einsum('ij,ij->i', diff, diff)
        distances[i] = np.inf
        neighbors[i] = np.argsort(distances)[:k]
    return neighbors


def _smote_generate(X, n_samples, k, rng):
    n = X.shape[0]
    neighbors = _find_k_neighbors(X, k, rng)
    synthetic = np.zeros((n_samples, X.shape[1]))
    for i in range(n_samples):
        idx = rng.randint(0, n)
        neighbor_idx = rng.choice(neighbors[idx])
        diff = X[neighbor_idx] - X[idx]
        gap = rng.random()
        synthetic[i] = X[idx] + gap * diff
    return synthetic


def smote_oversample(X, y, k_neighbors=5, random_state=None):
    rng = np.random.RandomState(random_state)
    counts = Counter(y)
    max_count = max(counts.values())
    X_resampled, y_resampled = [], []
    for label in counts:
        mask = y == label
        X_class = X[mask]
        n = counts[label]
        X_resampled.append(X_class)
        y_resampled.append(np.full(n, label))
        if n < max_count:
            n_synthetic = max_count - n
            synthetic = _smote_generate(X_class, n_synthetic, min(k_neighbors, n - 1), rng)
            X_resampled.append(synthetic)
            y_resampled.append(np.full(n_synthetic, label))
    return np.concatenate(X_resampled), np.concatenate(y_resampled)


def random_oversample(X, y, random_state=None):
    rng = np.random.RandomState(random_state)
    counts = Counter(y)
    max_count = max(counts.values())
    X_resampled, y_resampled = [], []
    for label in counts:
        mask = y == label
        X_class = X[mask]
        n = counts[label]
        indices = np.arange(n)
        if n < max_count:
            extra = rng.choice(indices, size=max_count - n, replace=True)
            indices = np.concatenate([indices, extra])
        X_resampled.append(X_class[indices])
        y_resampled.append(np.full(len(indices), label))
    return np.concatenate(X_resampled), np.concatenate(y_resampled)


def undersample(X, y, random_state=None):
    rng = np.random.RandomState(random_state)
    counts = Counter(y)
    min_count = min(counts.values())
    X_resampled, y_resampled = [], []
    for label in counts:
        mask = y == label
        X_class = X[mask]
        indices = rng.choice(np.arange(counts[label]), size=min_count, replace=False)
        X_resampled.append(X_class[indices])
        y_resampled.append(np.full(min_count, label))
    return np.concatenate(X_resampled), np.concatenate(y_resampled)


def _tomek_links_clean(X, y):
    n = len(y)
    neighbors = _find_k_neighbors_all(X, 1)
    tomek_pairs = set()
    for i in range(n):
        nn = neighbors[i, 0]
        if neighbors[nn, 0] == i and y[i] != y[nn]:
            tomek_pairs.add(i)
            tomek_pairs.add(nn)
    mask = np.ones(n, dtype=bool)
    mask[list(tomek_pairs)] = False
    return X[mask], y[mask]


def smote_tomek(X, y, k_neighbors=5, random_state=None):
    X_smote, y_smote = smote_oversample(X, y, k_neighbors, random_state)
    return _tomek_links_clean(X_smote, y_smote)


def _enn_clean(X, y, k_neighbors=3):
    n = len(y)
    neighbors = _find_k_neighbors_all(X, k_neighbors)
    to_remove = []
    for i in range(n):
        nn_labels = y[neighbors[i]]
        pred = Counter(nn_labels).most_common(1)[0][0]
        if pred != y[i]:
            to_remove.append(i)
    mask = np.ones(n, dtype=bool)
    mask[to_remove] = False
    return X[mask], y[mask]


def smote_enn(X, y, k_neighbors=5, random_state=None):
    X_smote, y_smote = smote_oversample(X, y, k_neighbors, random_state)
    return _enn_clean(X_smote, y_smote, k_neighbors=3)


def easy_ensemble(X, y, n_estimators=5, random_state=None):
    rng = np.random.RandomState(random_state)
    counts = Counter(y)
    labels = list(counts.keys())
    minority_label = min(counts, key=counts.get)
    majority_label = max(counts, key=counts.get)
    n_minority = counts[minority_label]
    X_min = X[y == minority_label]
    X_maj = X[y == majority_label]
    subsets = []
    for _ in range(n_estimators):
        maj_indices = rng.choice(len(X_maj), size=n_minority, replace=False)
        X_sub = np.vstack([X_min, X_maj[maj_indices]])
        y_sub = np.concatenate([np.full(n_minority, minority_label),
                                 np.full(n_minority, majority_label)])
        shuffle_idx = rng.permutation(len(X_sub))
        subsets.append((X_sub[shuffle_idx], y_sub[shuffle_idx]))
    return subsets


def balance_cascade(X, y, n_estimators=5, random_state=None):
    rng = np.random.RandomState(random_state)
    counts = Counter(y)
    minority_label = min(counts, key=counts.get)
    majority_label = max(counts, key=counts.get)
    n_minority = counts[minority_label]
    X_min = X[y == minority_label]
    X_maj = X[y == majority_label]
    y_min = np.full(n_minority, minority_label)
    subsets = []
    maj_indices = np.arange(len(X_maj))
    for _ in range(n_estimators):
        if len(maj_indices) <= n_minority:
            selected = maj_indices
        else:
            selected = rng.choice(maj_indices, size=n_minority, replace=False)
        X_sub = np.vstack([X_min, X_maj[selected]])
        y_sub = np.concatenate([y_min, np.full(len(selected), majority_label)])
        shuffle_idx = rng.permutation(len(X_sub))
        subsets.append((X_sub[shuffle_idx], y_sub[shuffle_idx]))
        if len(maj_indices) > n_minority:
            maj_indices = np.array([i for i in maj_indices if i not in selected])
    return subsets


class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        return self
    
    def predict(self, X):
        predictions = []
        for x in X:
            diff = self.X - x
            distances = np.einsum('ij,ij->i', diff, diff)
            idx = np.argsort(distances)[:self.k]
            pred = Counter(self.y[idx]).most_common(1)[0][0]
            predictions.append(pred)
        return np.array(predictions)


def evaluate_model(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def train_test_split(X, y, test_size=0.3, random_state=None):
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]


def compare_resampling_methods(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    methods = {
        '原始数据': (X_train, y_train),
        '随机过采样': random_oversample(X_train, y_train, random_state=random_state),
        'SMOTE过采样': smote_oversample(X_train, y_train, random_state=random_state),
        '欠采样': undersample(X_train, y_train, random_state=random_state),
        'SMOTE+Tomek': smote_tomek(X_train, y_train, random_state=random_state),
        'SMOTE+ENN': smote_enn(X_train, y_train, random_state=random_state),
    }
    results = {}
    for name, (X_tr, y_tr) in methods.items():
        model = SimpleKNN(k=3)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_test)
        results[name] = evaluate_model(y_test, y_pred)
    return results


if __name__ == "__main__":
    np.random.seed(42)
    X0 = np.random.randn(450, 2) + np.array([2, 2])
    X1 = np.random.randn(50, 2) + np.array([-1, -1])
    X = np.vstack([X0, X1])
    y = np.array([0] * 450 + [1] * 50)

    print("=" * 60)
    print("【数据分布】")
    print(f"原始分布: {Counter(y)}")
    print("=" * 60)

    print("\n【1. 基础采样方法】")
    X_rand, y_rand = random_oversample(X, y, random_state=42)
    print(f"随机过采样: {Counter(y_rand)}")

    X_smote, y_smote = smote_oversample(X, y, random_state=42)
    print(f"SMOTE过采样: {Counter(y_smote)}")

    X_under, y_under = undersample(X, y, random_state=42)
    print(f"欠采样: {Counter(y_under)}")

    print("\n【2. 综合采样方法 (过采样+欠采样结合)】")
    X_st, y_st = smote_tomek(X, y, random_state=42)
    print(f"SMOTE+TomekLinks: {Counter(y_st)} (移除边界噪声样本)")

    X_se, y_se = smote_enn(X, y, random_state=42)
    print(f"SMOTE+ENN: {Counter(y_se)} (KNN过滤不一致样本)")

    print("\n【3. 集成采样方法】")
    ee_subsets = easy_ensemble(X, y, n_estimators=5, random_state=42)
    print(f"EasyEnsemble: 生成{len(ee_subsets)}个平衡子集，每个子集: {Counter(ee_subsets[0][1])}")

    bc_subsets = balance_cascade(X, y, n_estimators=5, random_state=42)
    print(f"BalanceCascade: 生成{len(bc_subsets)}个平衡子集，每个子集: {Counter(bc_subsets[0][1])}")

    print("\n" + "=" * 60)
    print("【分类性能对比 (KNN分类器)】")
    print("=" * 60)
    results = compare_resampling_methods(X, y, random_state=42)
    print(f"{'方法':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1']:<10.3f}")
    print("=" * 60)
    print("注: 召回率提升说明对少数类的识别能力增强，F1为综合指标")
