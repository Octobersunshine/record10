import numpy as np
from collections import Counter, defaultdict
import random


class KNNClassifier:
    def __init__(self, k=5, tie_break='nearest', random_state=None,
                 weights='uniform', metric='euclidean', p=2):
        self.k = k
        self.tie_break = tie_break
        self.random_state = random_state
        self.weights = weights
        self.metric = metric
        self.p = p
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)

    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1.0 / self.p)
        else:
            raise ValueError(f"不支持的距离度量: {self.metric}，可选: 'euclidean', 'manhattan', 'minkowski'")

    def _compute_weight(self, distance):
        if self.weights == 'uniform':
            return 1.0
        elif self.weights == 'distance':
            return 1.0 / (distance + 1e-10)
        else:
            raise ValueError(f"不支持的权重模式: {self.weights}，可选: 'uniform', 'distance'")

    def _predict_single(self, x):
        distances = np.array([self._compute_distance(x, x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)[:self.k]
        k_neighbors = self.y_train[k_indices]
        k_distances = distances[k_indices]
        k_weights = np.array([self._compute_weight(d) for d in k_distances])

        if self.weights == 'uniform':
            vote_counts = Counter(k_neighbors)
        else:
            vote_counts = defaultdict(float)
            for label, w in zip(k_neighbors, k_weights):
                vote_counts[label] += w

        max_votes = max(vote_counts.values())
        top_candidates = [label for label, count in vote_counts.items() if count == max_votes]

        is_tie = len(top_candidates) > 1

        if is_tie:
            if self.tie_break == 'nearest':
                candidate_min_distances = {}
                for label in top_candidates:
                    label_indices = [i for i, neighbor in enumerate(k_neighbors) if neighbor == label]
                    min_distance = min(k_distances[i] for i in label_indices)
                    candidate_min_distances[label] = min_distance
                predicted_label = min(candidate_min_distances, key=candidate_min_distances.get)
                tie_break_reason = f"选择最近邻居(距离: {candidate_min_distances[predicted_label]:.4f})"
            elif self.tie_break == 'random':
                predicted_label = random.choice(top_candidates)
                tie_break_reason = "随机选择"
            elif self.tie_break == 'weighted':
                candidate_weights = {}
                for label in top_candidates:
                    label_indices = [i for i, neighbor in enumerate(k_neighbors) if neighbor == label]
                    weight = sum(1.0 / (k_distances[i] + 1e-10) for i in label_indices)
                    candidate_weights[label] = weight
                predicted_label = max(candidate_weights, key=candidate_weights.get)
                tie_break_reason = f"距离加权投票(权重: {candidate_weights[predicted_label]:.4f})"
            else:
                predicted_label = top_candidates[0]
                tie_break_reason = "选择首个候选"
        else:
            predicted_label = top_candidates[0]
            tie_break_reason = "多数投票" if self.weights == 'uniform' else "加权投票"

        neighbor_details = [
            {
                'index': int(idx),
                'label': int(label),
                'distance': float(dist),
                'weight': float(w),
            }
            for idx, label, dist, w in zip(k_indices, k_neighbors, k_distances, k_weights)
        ]

        all_distances = [
            {
                'index': int(i),
                'distance': float(distances[i]),
            }
            for i in range(len(distances))
        ]

        vote_details = {
            'votes': {int(k): (float(v) if self.weights != 'uniform' else int(v)) for k, v in vote_counts.items()},
            'is_tie': is_tie,
            'tie_candidates': [int(c) for c in top_candidates] if is_tie else None,
            'tie_break_reason': tie_break_reason,
            'weight_mode': self.weights,
        }

        return int(predicted_label), neighbor_details, vote_details, all_distances

    def predict(self, X, return_details=False):
        X = np.array(X, dtype=float)
        results = [self._predict_single(x) for x in X]
        predictions = np.array([r[0] for r in results])

        if return_details:
            details = [
                {
                    'neighbors': r[1],
                    'vote_details': r[2],
                    'all_distances': r[3],
                }
                for r in results
            ]
            return predictions, details
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == np.array(y))


def _print_detail(detail, sample_idx, x=None):
    prefix = f"  样本{sample_idx}" if x is None else f"  样本{sample_idx}: {x}"
    print(f"{prefix} -> 预测标签={detail['vote_details']['_pred']}")
    print(f"    邻居详情 (标签, 距离, 权重):")
    for n in detail['neighbors']:
        print(f"      标签={n['label']}, 距离={n['distance']:.4f}, 权重={n['weight']:.4f}")
    votes = detail['vote_details']['votes']
    is_tie = detail['vote_details']['is_tie']
    reason = detail['vote_details']['tie_break_reason']
    weight_mode = detail['vote_details']['weight_mode']
    print(f"    投票统计({weight_mode}): {votes}")
    if is_tie:
        print(f"    [平局] 候选: {detail['vote_details']['tie_candidates']}, 决策: {reason}")
    else:
        print(f"    决策: {reason}")


if __name__ == "__main__":
    print("=" * 60)
    print("测试1: 不同距离度量对比")
    print("=" * 60)

    X_train_demo = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [3.0, 3.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [4.0, 4.0],
    ])
    y_train_demo = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X_test_demo = np.array([[1.5, 1.5]])

    for metric_name in ['euclidean', 'manhattan', 'minkowski']:
        p_val = 3 if metric_name == 'minkowski' else 2
        knn = KNNClassifier(k=3, metric=metric_name, p=p_val, weights='uniform')
        knn.fit(X_train_demo, y_train_demo)
        y_pred, details = knn.predict(X_test_demo, return_details=True)
        detail = details[0]
        detail['vote_details']['_pred'] = int(y_pred[0])

        print(f"\n--- {metric_name}" + (f" (p={p_val})" if metric_name == 'minkowski' else "") + " ---")
        print(f"  预测标签: {y_pred[0]}")
        labels = [n['label'] for n in detail['neighbors']]
        dists = [f"{n['distance']:.4f}" for n in detail['neighbors']]
        print(f"  邻居标签: {labels}")
        print(f"  邻居距离: {dists}")
        print(f"  投票: {detail['vote_details']['votes']}")

    print("\n" + "=" * 60)
    print("测试2: 加权KNN vs 均匀KNN")
    print("=" * 60)

    X_train_w = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [2.0, 2.0],
        [2.5, 2.5],
    ])
    y_train_w = np.array([0, 1, 1, 1])
    X_test_w = np.array([[0.3, 0.3]])

    for weight_mode in ['uniform', 'distance']:
        knn_w = KNNClassifier(k=4, weights=weight_mode, metric='euclidean')
        knn_w.fit(X_train_w, y_train_w)
        y_pred_w, details_w = knn_w.predict(X_test_w, return_details=True)
        detail_w = details_w[0]
        detail_w['vote_details']['_pred'] = int(y_pred_w[0])

        print(f"\n--- 权重模式: {weight_mode} ---")
        for n in detail_w['neighbors']:
            print(f"  标签={n['label']}, 距离={n['distance']:.4f}, 权重={n['weight']:.4f}")
        print(f"  投票: {detail_w['vote_details']['votes']}")
        print(f"  预测: {y_pred_w[0]}, 原因: {detail_w['vote_details']['tie_break_reason']}")

    print("\n" + "=" * 60)
    print("测试3: 全距离列表返回")
    print("=" * 60)

    knn_all = KNNClassifier(k=3, metric='euclidean')
    knn_all.fit(X_train_w, y_train_w)
    _, details_all = knn_all.predict(X_test_w, return_details=True)
    all_dists = details_all[0]['all_distances']
    print(f"训练样本数: {len(all_dists)}")
    for d in all_dists:
        print(f"  训练样本{d['index']}: 距离={d['distance']:.4f}")

    print("\n" + "=" * 60)
    print("测试4: Iris数据集 - 综合对比")
    print("=" * 60)
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    configs = [
        {'k': 3, 'weights': 'uniform', 'metric': 'euclidean'},
        {'k': 3, 'weights': 'distance', 'metric': 'euclidean'},
        {'k': 3, 'weights': 'uniform', 'metric': 'manhattan'},
        {'k': 3, 'weights': 'distance', 'metric': 'manhattan'},
        {'k': 5, 'weights': 'uniform', 'metric': 'minkowski', 'p': 3},
        {'k': 5, 'weights': 'distance', 'metric': 'minkowski', 'p': 3},
    ]

    print(f"{'k':>3} | {'权重':>8} | {'度量':>12} | {'p':>3} | {'准确率':>8}")
    print("-" * 52)
    for cfg in configs:
        knn_cfg = KNNClassifier(
            k=cfg['k'], weights=cfg['weights'], metric=cfg['metric'],
            p=cfg.get('p', 2), tie_break='nearest'
        )
        knn_cfg.fit(X_train, y_train)
        acc = knn_cfg.score(X_test, y_test)
        p_str = str(cfg.get('p', 2))
        print(f"{cfg['k']:>3} | {cfg['weights']:>8} | {cfg['metric']:>12} | {p_str:>3} | {acc:>8.4f}")
