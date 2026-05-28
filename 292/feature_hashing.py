import hashlib
import pickle
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, vstack


def _hash_index(category, output_dim, hash_func="md5"):
    """计算哈希索引"""
    if category is None:
        category = ""
    if hash_func == "md5":
        hash_value = int(hashlib.md5(category.encode("utf-8")).hexdigest(), 16)
    elif hash_func == "sha1":
        hash_value = int(hashlib.sha1(category.encode("utf-8")).hexdigest(), 16)
    elif hash_func == "sha256":
        hash_value = int(hashlib.sha256(category.encode("utf-8")).hexdigest(), 16)
    else:
        raise ValueError(f"Unknown hash function: {hash_func}")
    return hash_value % output_dim


def _hash_sign(category):
    """计算符号哈希，返回 +1 或 -1"""
    if category is None:
        category = ""
    sign_hash = int(hashlib.sha1(("sign_" + category).encode("utf-8")).hexdigest(), 16)
    return 1 if (sign_hash % 2 == 0) else -1


def _parse_feature(feature):
    """
    解析特征，支持多种格式：
        - str: 纯类别特征，权重默认为 1.0
        - tuple/list: (category, weight)
        - str with ":": "category:weight" 格式
    """
    if isinstance(feature, (tuple, list)):
        if len(feature) == 2:
            return str(feature[0]), float(feature[1])
        elif len(feature) == 1:
            return str(feature[0]), 1.0
        else:
            raise ValueError(f"Invalid feature tuple: {feature}")
    elif isinstance(feature, str):
        if ":" in feature:
            parts = feature.rsplit(":", 1)
            try:
                return parts[0], float(parts[1])
            except ValueError:
                return feature, 1.0
        else:
            return feature, 1.0
    else:
        return str(feature), 1.0


def _resolve_conflict(values, strategy):
    """根据策略解决冲突值"""
    if strategy == "sum":
        return sum(v for v, _ in values)
    elif strategy == "average":
        total = sum(v for v, _ in values)
        return total / len(values) if values else 0
    elif strategy == "max":
        if not values:
            return 0
        max_idx = max(range(len(values)), key=lambda i: abs(values[i][0]))
        return values[max_idx][0]
    elif strategy == "weighted_sum":
        return sum(v * w for v, w in values)
    elif strategy == "weighted_average":
        total_weight = sum(w for _, w in values)
        if total_weight == 0:
            return 0
        return sum(v * w for v, w in values) / total_weight
    else:
        raise ValueError(f"Unknown conflict strategy: {strategy}")


def feature_hashing(
    categories,
    output_dim,
    conflict_strategy="weighted_average",
    use_signed_hash=True,
):
    """
    特征哈希（Feature Hashing）：将高基数类别特征映射到固定维度的稀疏向量。
    支持带权重的特征输入，包含完善的哈希冲突处理机制。

    参数:
        categories: 支持多种输入格式:
            - list of str: 纯类别特征
            - list of tuple: [(category, weight), ...]
            - list of str with ":": ["category:weight", ...]
            - list of list: 每行多个特征
        output_dim: int, 输出向量的维度
        conflict_strategy: str, 冲突处理策略
            - "sum": 冲突值相加
            - "average": 冲突值取平均
            - "max": 冲突值取绝对值最大的
            - "weighted_sum": 带权重求和（默认）
            - "weighted_average": 带权重平均
        use_signed_hash: bool, 是否使用符号哈希

    返回:
        scipy.sparse.csr_matrix, 形状为 (n_samples, output_dim) 的稀疏特征矩阵
        dict, 冲突统计信息
    """
    if isinstance(categories[0], (list, tuple, set)) and not (
        len(categories[0]) == 2 and isinstance(categories[0][1], (int, float))
    ):
        n_samples = len(categories)
    else:
        categories = [[c] for c in categories]
        n_samples = len(categories)

    indices = []
    indptr = [0]
    data = []

    collision_stats = {
        "total_collisions": 0,
        "max_collisions_per_bin": 0,
        "collision_bins": defaultdict(int),
        "sign_cancellation_count": 0,
    }

    for row_cats in categories:
        bin_values = defaultdict(list)

        for feature in row_cats:
            category, weight = _parse_feature(feature)
            col_index = _hash_index(category, output_dim, hash_func="md5")

            if use_signed_hash:
                base_value = _hash_sign(category)
            else:
                base_value = 1

            value = base_value * weight
            bin_values[col_index].append((value, weight))

        for col_index, values in bin_values.items():
            n_conflict = len(values)
            if n_conflict > 1:
                collision_stats["total_collisions"] += (n_conflict - 1)
                collision_stats["collision_bins"][col_index] += (n_conflict - 1)
                collision_stats["max_collisions_per_bin"] = max(
                    collision_stats["max_collisions_per_bin"], n_conflict
                )

            final_value = _resolve_conflict(values, conflict_strategy)

            if use_signed_hash and n_conflict > 1 and abs(final_value) < 1e-10:
                collision_stats["sign_cancellation_count"] += 1

            if abs(final_value) > 1e-10:
                indices.append(col_index)
                data.append(final_value)

        indptr.append(len(indices))

    feature_matrix = csr_matrix(
        (data, indices, indptr), shape=(n_samples, output_dim), dtype=np.float64
    )

    collision_stats["total_bins_used"] = len(set(indices))
    collision_stats["collision_rate"] = (
        collision_stats["total_collisions"] / max(feature_matrix.nnz, 1)
    )

    return feature_matrix, collision_stats


def feature_cross(feature_a, feature_b):
    """
    特征交叉（Feature Crossing）：用于 CTR 等场景的特征组合
    例如: 特征A="北京", 特征B="male" → 交叉特征="北京_x_male"
    """
    cat_a, w_a = _parse_feature(feature_a)
    cat_b, w_b = _parse_feature(feature_b)
    cross_cat = f"{cat_a}_x_{cat_b}"
    cross_weight = w_a * w_b
    return (cross_cat, cross_weight)


class FeatureHasher:
    """
    支持在线学习的特征哈希器，适用于 CTR 预估等大规模在线学习场景。

    特性:
    - 增量学习：新类别无需重新训练，直接调用 partial_fit
    - 状态持久化：支持 save/load 保存和恢复模型状态
    - 统计追踪：记录各哈希桶的点击率、出现次数等统计量
    - 带权重特征：支持 (category, weight) 格式输入
    """

    def __init__(
        self,
        output_dim=2**18,
        conflict_strategy="weighted_average",
        use_signed_hash=True,
    ):
        self.output_dim = output_dim
        self.conflict_strategy = conflict_strategy
        self.use_signed_hash = use_signed_hash

        self.n_samples_seen = 0
        self.n_features_seen = 0

        self.bin_counts = np.zeros(output_dim, dtype=np.int64)
        self.bin_click_counts = np.zeros(output_dim, dtype=np.int64)
        self.bin_sum_weights = np.zeros(output_dim, dtype=np.float64)
        self.bin_sum_values = np.zeros(output_dim, dtype=np.float64)

        self.category_frequency = defaultdict(int)
        self.last_updated = None

        self.collision_stats = {
            "total_collisions": 0,
            "max_collisions_per_bin": 0,
            "sign_cancellation_count": 0,
            "total_partial_fit_calls": 0,
        }

    def _get_bin_index(self, category):
        return _hash_index(category, self.output_dim, hash_func="md5")

    def _get_sign(self, category):
        return _hash_sign(category) if self.use_signed_hash else 1

    def fit(self, X, y=None):
        """
        批量拟合（重置状态）。

        参数:
            X: 特征列表，格式同 feature_hashing 函数
            y: 可选，标签列表（1=点击, 0=未点击），用于 CTR 统计
        """
        self._reset_stats()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """
        增量拟合，支持新类别流式处理，无需重新训练。

        参数:
            X: 特征列表，格式同 feature_hashing 函数
            y: 可选，标签列表（1=点击, 0=未点击），用于 CTR 统计

        返回:
            self
        """
        import time

        self.last_updated = time.time()
        self.collision_stats["total_partial_fit_calls"] += 1

        if not isinstance(X[0], (list, tuple, set)) or (
            len(X[0]) == 2 and isinstance(X[0][1], (int, float))
        ):
            X = [[x] for x in X]

        n_batch = len(X)
        self.n_samples_seen += n_batch

        for i, row_cats in enumerate(X):
            bin_values = defaultdict(list)
            affected_bins = set()

            for feature in row_cats:
                category, weight = _parse_feature(feature)
                self.category_frequency[category] += 1
                self.n_features_seen += 1

                col_index = self._get_bin_index(category)
                sign = self._get_sign(category)
                value = sign * weight

                bin_values[col_index].append((value, weight))
                affected_bins.add(col_index)

            for col_index, values in bin_values.items():
                n_conflict = len(values)
                if n_conflict > 1:
                    self.collision_stats["total_collisions"] += (n_conflict - 1)
                    self.collision_stats["max_collisions_per_bin"] = max(
                        self.collision_stats["max_collisions_per_bin"], n_conflict
                    )

                final_value = _resolve_conflict(values, self.conflict_strategy)

                if self.use_signed_hash and n_conflict > 1 and abs(final_value) < 1e-10:
                    self.collision_stats["sign_cancellation_count"] += 1

                self.bin_counts[col_index] += 1
                self.bin_sum_values[col_index] += final_value
                self.bin_sum_weights[col_index] += sum(w for _, w in values)

                if y is not None and i < len(y):
                    if y[i] == 1:
                        self.bin_click_counts[col_index] += 1

        return self

    def transform(self, X):
        """
        将特征转换为稀疏向量。

        参数:
            X: 特征列表

        返回:
            scipy.sparse.csr_matrix, 形状为 (n_samples, output_dim)
        """
        matrix, _ = feature_hashing(
            X,
            output_dim=self.output_dim,
            conflict_strategy=self.conflict_strategy,
            use_signed_hash=self.use_signed_hash,
        )
        return matrix

    def fit_transform(self, X, y=None):
        """拟合并转换"""
        return self.fit(X, y).transform(X)

    def partial_fit_transform(self, X, y=None):
        """增量拟合并转换"""
        return self.partial_fit(X, y).transform(X)

    def get_ctr(self, category=None, bin_index=None):
        """
        获取 CTR（点击率）统计。

        参数:
            category: str, 类别名称
            bin_index: int, 哈希桶索引

        返回:
            dict: 包含 count, clicks, ctr
        """
        if category is not None:
            bin_index = self._get_bin_index(category)

        if bin_index is None:
            raise ValueError("Either category or bin_index must be provided")

        count = self.bin_counts[bin_index]
        clicks = self.bin_click_counts[bin_index]
        ctr = clicks / count if count > 0 else 0.0

        return {
            "bin_index": bin_index,
            "count": count,
            "clicks": clicks,
            "ctr": ctr,
            "avg_value": self.bin_sum_values[bin_index] / max(count, 1),
        }

    def get_bin_stats(self, top_k=10, sort_by="count"):
        """
        获取活跃度最高的哈希桶统计。

        参数:
            top_k: int, 返回前 K 个
            sort_by: str, 排序方式 ('count', 'clicks', 'ctr')

        返回:
            list of dict
        """
        stats = []
        for i in range(self.output_dim):
            if self.bin_counts[i] > 0:
                count = self.bin_counts[i]
                clicks = self.bin_click_counts[i]
                stats.append({
                    "bin_index": i,
                    "count": count,
                    "clicks": clicks,
                    "ctr": clicks / count if count > 0 else 0.0,
                    "avg_value": self.bin_sum_values[i] / count,
                })

        stats.sort(key=lambda x: x[sort_by], reverse=True)
        return stats[:top_k]

    def get_collision_rate(self):
        """获取整体冲突率"""
        total_bins = np.count_nonzero(self.bin_counts)
        if total_bins == 0:
            return 0.0
        return self.collision_stats["total_collisions"] / max(total_bins, 1)

    def _reset_stats(self):
        """重置所有统计状态"""
        self.n_samples_seen = 0
        self.n_features_seen = 0
        self.bin_counts = np.zeros(self.output_dim, dtype=np.int64)
        self.bin_click_counts = np.zeros(self.output_dim, dtype=np.int64)
        self.bin_sum_weights = np.zeros(self.output_dim, dtype=np.float64)
        self.bin_sum_values = np.zeros(self.output_dim, dtype=np.float64)
        self.category_frequency = defaultdict(int)
        self.collision_stats = {
            "total_collisions": 0,
            "max_collisions_per_bin": 0,
            "sign_cancellation_count": 0,
            "total_partial_fit_calls": 0,
        }

    def save(self, filepath):
        """保存模型状态到文件"""
        state = {
            "output_dim": self.output_dim,
            "conflict_strategy": self.conflict_strategy,
            "use_signed_hash": self.use_signed_hash,
            "n_samples_seen": self.n_samples_seen,
            "n_features_seen": self.n_features_seen,
            "bin_counts": self.bin_counts,
            "bin_click_counts": self.bin_click_counts,
            "bin_sum_weights": self.bin_sum_weights,
            "bin_sum_values": self.bin_sum_values,
            "category_frequency": dict(self.category_frequency),
            "collision_stats": self.collision_stats,
            "last_updated": self.last_updated,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath):
        """从文件加载模型状态"""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        hasher = cls(
            output_dim=state["output_dim"],
            conflict_strategy=state["conflict_strategy"],
            use_signed_hash=state["use_signed_hash"],
        )
        hasher.n_samples_seen = state["n_samples_seen"]
        hasher.n_features_seen = state["n_features_seen"]
        hasher.bin_counts = state["bin_counts"]
        hasher.bin_click_counts = state["bin_click_counts"]
        hasher.bin_sum_weights = state["bin_sum_weights"]
        hasher.bin_sum_values = state["bin_sum_values"]
        hasher.category_frequency = defaultdict(int, state["category_frequency"])
        hasher.collision_stats = state["collision_stats"]
        hasher.last_updated = state["last_updated"]
        return hasher

    def __repr__(self):
        return (
            f"FeatureHasher(output_dim={self.output_dim}, "
            f"conflict_strategy='{self.conflict_strategy}', "
            f"use_signed_hash={self.use_signed_hash}, "
            f"n_samples_seen={self.n_samples_seen})"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("测试 1: 带权重的特征哈希 - 多种输入格式")
    print("=" * 70)

    categories_weighted = [
        [("user_12345", 0.8), ("北京", 1.0), ("age:25", 0.5)],
        [("user_67890", 0.9), "上海", "gender:female"],
        [("user_11111", 0.7), ("北京", 1.0), "gender:male"],
    ]

    output_dim = 16
    matrix_weighted, stats_weighted = feature_hashing(
        categories_weighted, output_dim, conflict_strategy="weighted_average"
    )

    print("输入带权特征:")
    for row in categories_weighted:
        print(f"  {row}")
    print("\n稠密表示 (带权平均哈希):")
    np.set_printoptions(precision=3, suppress=True)
    print(matrix_weighted.toarray())
    print(f"\n冲突率: {stats_weighted['collision_rate']:.2%}")
    print(f"非零元素: {matrix_weighted.nnz}")

    print("\n" + "=" * 70)
    print("测试 2: 特征交叉 (Feature Crossing) - CTR 场景常用")
    print("=" * 70)

    feat1 = ("北京", 1.0)
    feat2 = ("male", 1.0)
    cross = feature_cross(feat1, feat2)
    print(f"特征A: {feat1}")
    print(f"特征B: {feat2}")
    print(f"交叉特征: {cross}")

    cross_categories = [
        [feature_cross(("北京", 1.0), ("male", 1.0)), ("user_12345", 0.8)],
        [feature_cross(("上海", 1.0), ("female", 1.0)), ("user_67890", 0.9)],
    ]
    matrix_cross, _ = feature_hashing(cross_categories, 16)
    print(f"\n交叉特征矩阵:\n{matrix_cross.toarray()}")

    print("\n" + "=" * 70)
    print("测试 3: FeatureHasher 在线学习 - 流式数据处理")
    print("=" * 70)

    hasher = FeatureHasher(
        output_dim=16,
        conflict_strategy="weighted_average",
        use_signed_hash=True,
    )
    print(f"初始化: {hasher}")

    batch_1 = [
        [("user_001", 0.8), ("北京", 1.0), ("male", 1.0)],
        [("user_002", 0.7), ("上海", 1.0), ("female", 1.0)],
    ]
    y_1 = [1, 0]

    hasher.partial_fit(batch_1, y_1)
    print(f"\n第1批 partial_fit 后:")
    print(f"  已处理样本数: {hasher.n_samples_seen}")
    print(f"  已处理特征数: {hasher.n_features_seen}")
    print(f"  总冲突次数: {hasher.collision_stats['total_collisions']}")

    print("\n  变换结果:")
    X1 = hasher.transform(batch_1)
    print(X1.toarray())

    batch_2 = [
        [("user_003", 0.9), ("广州", 1.0), ("male", 1.0)],
        [("user_001", 0.8), ("北京", 1.0), ("male", 1.0)],
        [("user_NEW", 0.6), ("深圳", 1.0), ("female", 1.0)],
    ]
    y_2 = [0, 1, 1]

    hasher.partial_fit(batch_2, y_2)
    print(f"\n第2批 partial_fit 后 (包含新类别 user_NEW, 深圳):")
    print(f"  已处理样本数: {hasher.n_samples_seen}")
    print(f"  已处理特征数: {hasher.n_features_seen}")
    print(f"  已见类别数: {len(hasher.category_frequency)}")
    print(f"  partial_fit 调用次数: {hasher.collision_stats['total_partial_fit_calls']}")

    print("\n  Top 活跃桶统计:")
    top_bins = hasher.get_bin_stats(top_k=5, sort_by="count")
    for b in top_bins:
        print(f"    桶{b['bin_index']}: 次数={b['count']}, 点击={b['clicks']}, CTR={b['ctr']:.2%}")

    print(f"\n  '北京' 的 CTR 统计: {hasher.get_ctr(category='北京')}")
    print(f"  'user_001' 的 CTR 统计: {hasher.get_ctr(category='user_001')}")

    print("\n" + "=" * 70)
    print("测试 4: 模型保存与加载 - 状态持久化")
    print("=" * 70)

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        model_path = f.name

    hasher.save(model_path)
    print(f"模型已保存到: {model_path}")
    print(f"保存前 - 样本数: {hasher.n_samples_seen}, 特征数: {hasher.n_features_seen}")

    loaded_hasher = FeatureHasher.load(model_path)
    print(f"\n加载后 - 样本数: {loaded_hasher.n_samples_seen}, 特征数: {loaded_hasher.n_features_seen}")
    print(f"配置一致: {loaded_hasher.output_dim == hasher.output_dim}")

    batch_3 = [[("user_100", 0.5), ("杭州", 1.0)]]
    loaded_hasher.partial_fit(batch_3, [0])
    print(f"\n加载后继续 partial_fit:")
    print(f"  样本数: {loaded_hasher.n_samples_seen}")
    print(f"  类别数: {len(loaded_hasher.category_frequency)}")

    os.unlink(model_path)

    print("\n" + "=" * 70)
    print("测试 5: CTR 预估场景完整模拟")
    print("=" * 70)

    ctr_hasher = FeatureHasher(output_dim=64, conflict_strategy="weighted_sum")

    print("模拟 10 批流式广告点击数据...")
    import random

    random.seed(42)
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "南京", "武汉"]
    genders = ["male", "female"]

    for batch_idx in range(10):
        batch = []
        labels = []
        for _ in range(20):
            user_id = f"user_{random.randint(1, 100)}"
            city = random.choice(cities)
            gender = random.choice(genders)
            city_weight = random.uniform(0.5, 1.5)
            user_weight = random.uniform(0.3, 1.0)

            features = [
                (user_id, user_weight),
                feature_cross((city, city_weight), (gender, 1.0)),
            ]
            batch.append(features)

            ctr = 0.3 if city in ["北京", "上海"] else 0.1
            label = 1 if random.random() < ctr else 0
            labels.append(label)

        ctr_hasher.partial_fit(batch, labels)

        if (batch_idx + 1) % 5 == 0:
            print(f"  批 {batch_idx + 1}: 样本={ctr_hasher.n_samples_seen}, "
                  f"类别={len(ctr_hasher.category_frequency)}, "
                  f"冲突率={ctr_hasher.get_collision_rate():.2%}")

    print("\nTop 5 点击率最高的桶:")
    top_ctr = ctr_hasher.get_bin_stats(top_k=5, sort_by="ctr")
    for b in top_ctr:
        if b["count"] >= 3:
            print(f"  桶{b['bin_index']}: CTR={b['ctr']:.2%} (样本={b['count']})")

    print("\n各城市 CTR:")
    for city in ["北京", "上海", "广州", "成都"]:
        stats = ctr_hasher.get_ctr(category=city)
        print(f"  {city}: CTR={stats['ctr']:.2%} (样本={stats['count']})")

    print("\n" + "=" * 70)
    print("测试 6: 对比 - 有无符号哈希对冲突抵消的影响")
    print("=" * 70)

    conflict_demo = [
        ["apple", "banana", "cherry", "date"],
    ]

    mat_signed, stats_signed = feature_hashing(
        conflict_demo, output_dim=4, conflict_strategy="sum", use_signed_hash=True
    )
    mat_unsigned, stats_unsigned = feature_hashing(
        conflict_demo, output_dim=4, conflict_strategy="sum", use_signed_hash=False
    )

    print(f"输入: {conflict_demo[0]}")
    print(f"输出维度: 4 (严重冲突场景)")
    print(f"\n符号哈希 (有正负抵消):")
    print(f"  {mat_signed.toarray()[0]}")
    print(f"  符号抵消次数: {stats_signed['sign_cancellation_count']}")
    print(f"\n无符号哈希 (全部为正):")
    print(f"  {mat_unsigned.toarray()[0]}")
    print(f"  符号抵消次数: {stats_unsigned['sign_cancellation_count']}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
