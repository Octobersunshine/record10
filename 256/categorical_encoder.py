from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from scipy.sparse import csr_matrix, vstack


class LabelEncoder:
    def __init__(self):
        self.classes_: Optional[List[str]] = None
        self._mapping: Optional[Dict[str, int]] = None

    def fit(self, y: List[str]) -> 'LabelEncoder':
        unique_classes = sorted(set(y))
        self.classes_ = list(unique_classes)
        self._mapping = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y: List[str]) -> np.ndarray:
        if self._mapping is None:
            raise ValueError("LabelEncoder has not been fitted yet. Call fit() first.")
        return np.array([self._mapping[cls] for cls in y], dtype=np.int64)

    def fit_transform(self, y: List[str]) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: Union[np.ndarray, List[int]]) -> List[str]:
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet. Call fit() first.")
        return [self.classes_[int(idx)] for idx in y]


class OneHotEncoder:
    VALID_HANDLE_UNKNOWN = {'error', 'ignore', 'unknown'}

    def __init__(self, sparse_output: bool = True, handle_unknown: str = 'error'):
        if handle_unknown not in self.VALID_HANDLE_UNKNOWN:
            raise ValueError(
                f"handle_unknown must be one of {self.VALID_HANDLE_UNKNOWN}, "
                f"got '{handle_unknown}'."
            )
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.categories_: Optional[List[List[str]]] = None
        self._n_features: Optional[int] = None

    def fit(self, X: List[str]) -> 'OneHotEncoder':
        unique_categories = sorted(set(X))
        self.categories_ = [list(unique_categories)]
        if self.handle_unknown == 'unknown':
            self._n_features = len(self.categories_[0]) + 1
        else:
            self._n_features = len(self.categories_[0])
        return self

    def transform(self, X: List[str]) -> Union[np.ndarray, csr_matrix, Tuple[Union[np.ndarray, csr_matrix], np.ndarray]]:
        if self.categories_ is None:
            raise ValueError("OneHotEncoder has not been fitted yet. Call fit() first.")

        n_samples = len(X)
        n_features = self._n_features
        category_to_idx = {cat: idx for idx, cat in enumerate(self.categories_[0])}

        row_indices = []
        col_indices = []
        data = []
        unknown_rows = []

        for i, val in enumerate(X):
            if val not in category_to_idx:
                if self.handle_unknown == 'error':
                    raise ValueError(
                        f"Unknown category '{val}' encountered during transform."
                    )
                elif self.handle_unknown == 'ignore':
                    unknown_rows.append(i)
                    continue
                elif self.handle_unknown == 'unknown':
                    unknown_rows.append(i)
                    row_indices.append(i)
                    col_indices.append(n_features - 1)
                    data.append(1)
                    continue
            row_indices.append(i)
            col_indices.append(category_to_idx[val])
            data.append(1)

        sparse_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_samples, n_features),
            dtype=np.float64
        )

        if self.sparse_output:
            result = sparse_matrix
        else:
            result = sparse_matrix.toarray()

        if self.handle_unknown == 'ignore' and unknown_rows:
            valid_mask = np.ones(n_samples, dtype=bool)
            valid_mask[unknown_rows] = False
            if isinstance(result, csr_matrix):
                result = result[valid_mask]
            else:
                result = result[valid_mask]
            return result, np.where(valid_mask)[0]

        return result

    def fit_transform(self, X: List[str]) -> Union[np.ndarray, csr_matrix, Tuple[Union[np.ndarray, csr_matrix], np.ndarray]]:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Union[np.ndarray, csr_matrix]) -> List[Optional[str]]:
        if self.categories_ is None:
            raise ValueError("OneHotEncoder has not been fitted yet. Call fit() first.")

        if hasattr(X, 'toarray'):
            X = X.toarray()

        X = np.asarray(X)

        if self.handle_unknown == 'unknown':
            known_cols = X[:, :-1]
            unknown_col = X[:, -1]
            indices = np.argmax(known_cols, axis=1)
            has_unknown = unknown_col.astype(bool)
            result = []
            for row_i, (idx, is_unknown) in enumerate(zip(indices, has_unknown)):
                if is_unknown:
                    result.append("unknown")
                elif known_cols[row_i].sum() == 0:
                    result.append(None)
                else:
                    result.append(self.categories_[0][int(idx)])
            return result

        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {X.shape[1]}.")

        indices = np.argmax(X, axis=1)
        result = []
        for i, idx in enumerate(indices):
            if X[i].sum() == 0:
                result.append(None)
            else:
                result.append(self.categories_[0][int(idx)])
        return result

    def get_feature_names_out(self) -> np.ndarray:
        if self.categories_ is None:
            raise ValueError("OneHotEncoder has not been fitted yet. Call fit() first.")
        names = [f"x0_{cat}" for cat in self.categories_[0]]
        if self.handle_unknown == 'unknown':
            names.append("x0_unknown")
        return np.array(names)


class TargetEncoder:
    VALID_HANDLE_UNKNOWN = {'error', 'value'}

    def __init__(self, smoothing: float = 1.0, handle_unknown: str = 'value', min_samples_leaf: int = 1):
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        if handle_unknown not in self.VALID_HANDLE_UNKNOWN:
            raise ValueError(
                f"handle_unknown must be one of {self.VALID_HANDLE_UNKNOWN}, "
                f"got '{handle_unknown}'."
            )
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown
        self.min_samples_leaf = min_samples_leaf
        self._encoding: Optional[Dict[str, float]] = None
        self._global_mean: Optional[float] = None

    def fit(self, X: List[str], y: Union[List[float], np.ndarray]) -> 'TargetEncoder':
        y = np.asarray(y, dtype=np.float64)
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got {len(X)} and {len(y)}."
            )

        self._global_mean = float(np.mean(y))

        category_stats: Dict[str, Tuple[int, float]] = {}
        for xi, yi in zip(X, y):
            if xi not in category_stats:
                category_stats[xi] = (0, 0.0)
            count, sum_target = category_stats[xi]
            category_stats[xi] = (count + 1, sum_target + yi)

        self._encoding = {}
        for category, (count, sum_target) in category_stats.items():
            cat_mean = sum_target / count
            if self.smoothing == 0:
                smoothed_mean = cat_mean
            else:
                smoothing_factor = 1 / (1 + np.exp(-(count - self.min_samples_leaf) / self.smoothing))
                smoothed_mean = smoothing_factor * cat_mean + (1 - smoothing_factor) * self._global_mean
            self._encoding[category] = float(smoothed_mean)

        return self

    def transform(self, X: List[str]) -> np.ndarray:
        if self._encoding is None or self._global_mean is None:
            raise ValueError("TargetEncoder has not been fitted yet. Call fit() first.")

        result = []
        for xi in X:
            if xi in self._encoding:
                result.append(self._encoding[xi])
            else:
                if self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{xi}' encountered during transform.")
                elif self.handle_unknown == 'value':
                    result.append(self._global_mean)

        return np.array(result, dtype=np.float64)

    def fit_transform(self, X: List[str], y: Union[List[float], np.ndarray]) -> np.ndarray:
        return self.fit(X, y).transform(X)


class FrequencyEncoder:
    VALID_HANDLE_UNKNOWN = {'error', 'value'}

    def __init__(self, handle_unknown: str = 'value', normalize: bool = True):
        if handle_unknown not in self.VALID_HANDLE_UNKNOWN:
            raise ValueError(
                f"handle_unknown must be one of {self.VALID_HANDLE_UNKNOWN}, "
                f"got '{handle_unknown}'."
            )
        self.handle_unknown = handle_unknown
        self.normalize = normalize
        self._encoding: Optional[Dict[str, float]] = None
        self._default_value: Optional[float] = None

    def fit(self, X: List[str]) -> 'FrequencyEncoder':
        n_samples = len(X)
        frequency: Dict[str, int] = {}
        for xi in X:
            frequency[xi] = frequency.get(xi, 0) + 1

        self._encoding = {}
        for category, count in frequency.items():
            if self.normalize:
                self._encoding[category] = count / n_samples
            else:
                self._encoding[category] = float(count)

        self._default_value = 0.0 if self.normalize else 0.0

        return self

    def transform(self, X: List[str]) -> np.ndarray:
        if self._encoding is None or self._default_value is None:
            raise ValueError("FrequencyEncoder has not been fitted yet. Call fit() first.")

        result = []
        for xi in X:
            if xi in self._encoding:
                result.append(self._encoding[xi])
            else:
                if self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{xi}' encountered during transform.")
                elif self.handle_unknown == 'value':
                    result.append(self._default_value)

        return np.array(result, dtype=np.float64)

    def fit_transform(self, X: List[str]) -> np.ndarray:
        return self.fit(X).transform(X)


if __name__ == "__main__":
    train_labels = ["apple", "banana", "orange", "apple", "orange", "banana", "apple"]
    test_labels = ["apple", "grape", "banana", "mango", "orange"]

    print("=" * 60)
    print("LabelEncoder 测试")
    print("=" * 60)
    le = LabelEncoder()
    le_labels = le.fit_transform(train_labels)
    print(f"原始数据: {train_labels}")
    print(f"类别映射: {le._mapping}")
    print(f"编码结果: {le_labels}")
    print(f"逆变换: {le.inverse_transform(le_labels)}")
    print()

    print("=" * 60)
    print("OneHotEncoder — handle_unknown='error' (默认)")
    print("=" * 60)
    ohe_error = OneHotEncoder(sparse_output=False, handle_unknown='error')
    ohe_error.fit(train_labels)
    print(f"训练集编码:\n{ohe_error.fit_transform(train_labels)}")
    try:
        ohe_error.transform(test_labels)
    except ValueError as e:
        print(f"测试集含未知类别时报错: {e}")
    print()

    print("=" * 60)
    print("OneHotEncoder — handle_unknown='ignore'")
    print("=" * 60)
    ohe_ignore = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_ignore.fit(train_labels)
    result_ignore, valid_idx = ohe_ignore.transform(test_labels)
    print(f"测试集: {test_labels}")
    print(f"有效样本索引: {valid_idx}")
    print(f"有效样本: {[test_labels[i] for i in valid_idx]}")
    print(f"编码结果 (忽略未知样本):\n{result_ignore}")
    print(f"特征名称: {ohe_ignore.get_feature_names_out()}")
    print(f"逆变换: {ohe_ignore.inverse_transform(result_ignore)}")
    print()

    ohe_ignore_sparse = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    ohe_ignore_sparse.fit(train_labels)
    result_sparse, valid_idx2 = ohe_ignore_sparse.transform(test_labels)
    print(f"稀疏模式 — 有效样本索引: {valid_idx2}")
    print(f"稀疏矩阵:\n{result_sparse}")
    print()

    print("=" * 60)
    print("OneHotEncoder — handle_unknown='unknown'")
    print("=" * 60)
    ohe_unknown = OneHotEncoder(sparse_output=False, handle_unknown='unknown')
    ohe_unknown.fit(train_labels)
    print(f"测试集: {test_labels}")
    print(f"特征名称: {ohe_unknown.get_feature_names_out()}")
    result_unknown = ohe_unknown.transform(test_labels)
    print(f"编码结果 (未知类别归入unknown列):\n{result_unknown}")
    print(f"逆变换: {ohe_unknown.inverse_transform(result_unknown)}")
    print()

    ohe_unknown_sparse = OneHotEncoder(sparse_output=True, handle_unknown='unknown')
    ohe_unknown_sparse.fit(train_labels)
    result_unknown_sparse = ohe_unknown_sparse.transform(test_labels)
    print(f"稀疏模式:\n{result_unknown_sparse}")
    print(f"转换为稠密:\n{result_unknown_sparse.toarray()}")
    print(f"逆变换: {ohe_unknown_sparse.inverse_transform(result_unknown_sparse)}")
    print()

    print("=" * 60)
    print("无效 handle_unknown 参数测试")
    print("=" * 60)
    try:
        OneHotEncoder(handle_unknown='invalid')
    except ValueError as e:
        print(f"正确捕获异常: {e}")
    print()

    print("=" * 60)
    print("TargetEncoder 目标编码测试")
    print("=" * 60)
    train_target = [1.0, 0.0, 0.5, 1.0, 0.6, 0.2, 0.9]
    print(f"训练类别: {train_labels}")
    print(f"目标值: {train_target}")
    print(f"全局均值: {np.mean(train_target):.4f}")
    print()

    te_smooth = TargetEncoder(smoothing=1.0)
    te_result = te_smooth.fit_transform(train_labels, train_target)
    print(f"平滑参数 smoothing=1.0 编码映射:")
    for cat, val in sorted(te_smooth._encoding.items()):
        print(f"  {cat}: {val:.4f}")
    print(f"编码结果: {te_result.round(4)}")
    print(f"测试集编码 (含未知类别 grape):")
    te_test = te_smooth.transform(test_labels)
    print(f"  {test_labels} -> {te_test.round(4)}")
    print()

    te_no_smooth = TargetEncoder(smoothing=0.0)
    te_no_smooth.fit(train_labels, train_target)
    print(f"无平滑 smoothing=0 编码映射:")
    for cat, val in sorted(te_no_smooth._encoding.items()):
        print(f"  {cat}: {val:.4f}")
    print()

    print("=" * 60)
    print("FrequencyEncoder 频率编码测试")
    print("=" * 60)
    fe_norm = FrequencyEncoder(normalize=True)
    fe_norm_result = fe_norm.fit_transform(train_labels)
    print(f"训练类别: {train_labels}")
    print(f"归一化频率编码映射:")
    for cat, val in sorted(fe_norm._encoding.items()):
        print(f"  {cat}: {val:.4f}")
    print(f"编码结果: {fe_norm_result.round(4)}")
    print()

    fe_count = FrequencyEncoder(normalize=False)
    fe_count_result = fe_count.fit_transform(train_labels)
    print(f"计数频率编码映射:")
    for cat, val in sorted(fe_count._encoding.items()):
        print(f"  {cat}: {val:.0f}")
    print(f"编码结果: {fe_count_result.astype(int)}")
    print()

    print(f"测试集编码 (含未知类别 grape, mango):")
    fe_test = fe_norm.transform(test_labels)
    print(f"  {test_labels} -> {fe_test.round(4)}")
    print()

    print("=" * 60)
    print("高基数场景对比: OneHot vs Target vs Frequency")
    print("=" * 60)
    high_cardinality = ["user_" + str(i) for i in range(10)]
    y_dummy = np.random.randn(10)
    print(f"模拟高基数类别 (10个唯一值): {high_cardinality[:5]}...")
    print(f"OneHotEncoder 输出维度: 10")
    print(f"TargetEncoder 输出维度: 1")
    print(f"FrequencyEncoder 输出维度: 1")
    te_high = TargetEncoder().fit_transform(high_cardinality, y_dummy)
    print(f"TargetEncoder 结果示例 (前5个): {te_high[:5].round(4)}")
    fe_high = FrequencyEncoder().fit_transform(high_cardinality)
    print(f"FrequencyEncoder 结果示例 (前5个): {fe_high[:5].round(4)}")
