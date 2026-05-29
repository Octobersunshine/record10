import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.feature_importances_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.tree_ = None
        self.leaf_values_ = {}

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = Counter(y)
        impurity = 1.0
        for count in counts.values():
            prob = count / m
            impurity -= prob ** 2
        return impurity

    def _information_gain(self, y, left_y, right_y):
        p = len(left_y) / len(y)
        return self._gini(y) - p * self._gini(left_y) - (1 - p) * self._gini(right_y)

    def _best_split(self, X, y, feature_indices):
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _create_leaf(self, y):
        leaf_id = len(self.leaf_values_)
        counts = Counter(y)
        total = sum(counts.values())
        proba = np.zeros(self.n_classes_)
        for cls, cnt in counts.items():
            if isinstance(cls, (int, np.integer)) and int(cls) < self.n_classes_:
                proba[int(cls)] = cnt / total
        pred = counts.most_common(1)[0][0]
        self.leaf_values_[leaf_id] = {'proba': proba, 'pred': pred}
        return leaf_id

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (n_classes == 1):
            return self._create_leaf(y)

        if self.max_features is None:
            feature_indices = range(n_features)
        else:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)

        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)

        if best_gain == 0:
            return self._create_leaf(y)

        self.feature_importances_[best_feature] += best_gain * n_samples

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _traverse_tree(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] <= threshold:
            return self._traverse_tree(x, left_subtree)
        else:
            return self._traverse_tree(x, right_subtree)

    def _get_decision_path(self, x, tree, path=None):
        if path is None:
            path = []
        if not isinstance(tree, tuple):
            return path
        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] <= threshold:
            path.append((feature, threshold, 'left'))
            return self._get_decision_path(x, left_subtree, path)
        else:
            path.append((feature, threshold, 'right'))
            return self._get_decision_path(x, right_subtree, path)

    def predict(self, X):
        predictions = []
        for x in X:
            leaf_id = self._traverse_tree(x, self.tree_)
            predictions.append(self.leaf_values_[leaf_id]['pred'])
        return np.array(predictions)

    def predict_proba(self, X):
        probas = []
        for x in X:
            leaf_id = self._traverse_tree(x, self.tree_)
            probas.append(self.leaf_values_[leaf_id]['proba'])
        return np.array(probas)

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.feature_importances_ = np.zeros(self.n_features_)
        self.leaf_values_ = {}
        self.tree_ = self._build_tree(X, y)
        return self


class RandomForestFeatureImportance:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.feature_importances_ = None
        self.permutation_importances_mean_ = None
        self.permutation_importances_std_ = None
        self.shap_values_ = None
        self.trees_ = []
        self.classes_ = None

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features

        self.feature_importances_ = np.zeros(n_features)
        self.trees_ = []

        for _ in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.feature_importances_ += tree.feature_importances_
            self.trees_.append(tree)

        self.feature_importances_ /= self.n_estimators

        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ /= total

        return self

    def predict(self, X):
        predictions = []
        for tree in self.trees_:
            predictions.append(tree.predict(X))
        predictions = np.array(predictions)
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions.T])

    def predict_proba(self, X):
        probas = []
        for tree in self.trees_:
            probas.append(tree.predict_proba(X))
        return np.mean(probas, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def permutation_importance(self, X, y, n_repeats=10):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_features = X.shape[1]
        importances = np.zeros((n_repeats, n_features))

        baseline_score = self.score(X, y)

        for repeat in range(n_repeats):
            for feature_idx in range(n_features):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, feature_idx])
                permuted_score = self.score(X_permuted, y)
                importances[repeat, feature_idx] = baseline_score - permuted_score

        self.permutation_importances_mean_ = np.mean(importances, axis=0)
        self.permutation_importances_std_ = np.std(importances, axis=0)

        total = np.sum(self.permutation_importances_mean_)
        if total > 0:
            self.permutation_importances_mean_ /= total

        return self.permutation_importances_mean_, self.permutation_importances_std_

    def _tree_shap_single_tree(self, tree, x, X_background):
        n_features = X_background.shape[1]
        shap_values = np.zeros(n_features)
        baseline_pred = np.mean([tree.predict_proba(X_background[i:i+1])[0] for i in range(min(100, len(X_background)))], axis=0)
        sample_pred = tree.predict_proba(x.reshape(1, -1))[0]

        decision_path = tree._get_decision_path(x, tree.tree_)

        current_contribution = np.zeros(n_features)
        for feature, threshold, direction in decision_path:
            left_mask = X_background[:, feature] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                left_pred = np.mean([tree.predict_proba(X_background[left_mask][i:i+1])[0]
                                     for i in range(min(50, np.sum(left_mask)))], axis=0)
                right_pred = np.mean([tree.predict_proba(X_background[right_mask][i:i+1])[0]
                                      for i in range(min(50, np.sum(right_mask)))], axis=0)

                weight_left = np.sum(left_mask) / len(X_background)
                weight_right = np.sum(right_mask) / len(X_background)

                if direction == 'left':
                    contribution = (left_pred - (weight_left * left_pred + weight_right * right_pred))
                else:
                    contribution = (right_pred - (weight_left * left_pred + weight_right * right_pred))

                current_contribution[feature] += np.mean(np.abs(contribution))

        return current_contribution

    def shap_analysis(self, X, X_background=None):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        if X_background is None:
            X_background = X[:min(100, len(X))]

        shap_values_all = np.zeros((n_samples, n_features))

        for i, x in enumerate(X):
            sample_shap = np.zeros(n_features)
            for tree in self.trees_:
                sample_shap += self._tree_shap_single_tree(tree, x, X_background)
            sample_shap /= self.n_estimators
            shap_values_all[i] = sample_shap

        self.shap_values_ = shap_values_all
        return shap_values_all

    def get_shap_global_importance(self, feature_names=None):
        if self.shap_values_ is None:
            raise ValueError("请先调用 shap_analysis() 方法")

        shap_importance = np.mean(np.abs(self.shap_values_), axis=0)

        total = np.sum(shap_importance)
        if total > 0:
            shap_importance /= total

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(shap_importance))]

        importance_dict = dict(zip(feature_names, shap_importance))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance

    def get_shap_waterfall_data(self, sample_idx, feature_names=None):
        if self.shap_values_ is None:
            raise ValueError("请先调用 shap_analysis() 方法")

        n_features = self.shap_values_.shape[1]

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]

        sample_shap = self.shap_values_[sample_idx]

        sorted_indices = np.argsort(np.abs(sample_shap))[::-1]

        waterfall_data = []
        for idx in sorted_indices:
            waterfall_data.append({
                'feature': feature_names[idx],
                'shap_value': sample_shap[idx],
                'abs_value': abs(sample_shap[idx])
            })

        return waterfall_data

    def get_feature_importance(self, feature_names=None):
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]

        importance_dict = dict(zip(feature_names, self.feature_importances_))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance

    def get_permutation_importance(self, feature_names=None):
        if not hasattr(self, 'permutation_importances_mean_'):
            raise ValueError("请先调用 permutation_importance() 方法")

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.permutation_importances_mean_))]

        importance_dict = dict(zip(feature_names, self.permutation_importances_mean_))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance


def generate_test_data_with_categorical(n_samples=1000, n_categories=10, random_state=42):
    np.random.seed(random_state)

    useful_cat = np.random.randint(0, n_categories, n_samples)
    useless_cat = np.random.randint(0, n_categories, n_samples)

    useful_num = np.random.randn(n_samples)
    useless_num = np.random.randn(n_samples)

    y1 = (useful_num > 0).astype(int)
    y2 = (useful_cat == 0).astype(int) | (useful_cat == n_categories - 1).astype(int)
    y = (y1 | y2).astype(int)

    X = np.column_stack([useless_cat, useful_cat, useless_num, useful_num])
    feature_names = ['Useless_Cat', 'Useful_Cat', 'Useless_Num', 'Useful_Num']

    return X, y, feature_names


if __name__ == "__main__":
    print("=" * 80)
    print("特征重要性对比：基尼不纯度 vs 排列重要性 vs SHAP")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("测试1：Iris 数据集验证")
    print("=" * 80)
    from sklearn.datasets import load_iris

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    iris_feature_names = iris.feature_names

    rf_iris = RandomForestFeatureImportance(n_estimators=20, max_depth=5, random_state=42)
    rf_iris.fit(X_iris, y_iris)

    print(f"\n模型准确率: {rf_iris.score(X_iris, y_iris):.4f}")

    print("\n1. 基尼不纯度重要性:")
    gini_iris = rf_iris.get_feature_importance(iris_feature_names)
    for name, score in gini_iris.items():
        print(f"  {name:30s}: {score:.6f}")

    print("\n2. 排列重要性:")
    rf_iris.permutation_importance(X_iris, y_iris, n_repeats=10)
    perm_iris = rf_iris.get_permutation_importance(iris_feature_names)
    for name, score in perm_iris.items():
        print(f"  {name:30s}: {score:.6f}")

    print("\n3. SHAP全局重要性 (简化版Tree SHAP):")
    rf_iris.shap_analysis(X_iris[:50])
    shap_iris = rf_iris.get_shap_global_importance(iris_feature_names)
    for name, score in shap_iris.items():
        print(f"  {name:30s}: {score:.6f}")

    print("\n" + "=" * 80)
    print("测试2：单个样本SHAP瀑布图数据")
    print("=" * 80)

    sample_idx = 0
    print(f"\n样本索引: {sample_idx}")
    print(f"真实标签: {y_iris[sample_idx]}")
    print(f"预测标签: {rf_iris.predict(X_iris[sample_idx:sample_idx+1])[0]}")
    print(f"预测概率: {rf_iris.predict_proba(X_iris[sample_idx:sample_idx+1])[0]}")

    print("\nSHAP特征贡献（按绝对值从大到小排序）:")
    waterfall_data = rf_iris.get_shap_waterfall_data(sample_idx, iris_feature_names)
    print(f"{'特征':30s} {'SHAP值':>12s} {'绝对值':>12s}")
    print("-" * 60)
    for item in waterfall_data:
        print(f"{item['feature']:30s} {item['shap_value']:12.6f} {item['abs_value']:12.6f}")

    print("\n" + "=" * 80)
    print("测试3：类别特征编码问题对比")
    print("=" * 80)
    print("数据说明：")
    print("  - Useful_Cat:   有用的类别特征（与目标相关）")
    print("  - Useless_Cat:  无用的类别特征（随机噪声）")
    print("  - Useful_Num:   有用的数值特征（与目标相关）")
    print("  - Useless_Num:  无用的数值特征（随机噪声）")
    print("-" * 80)

    X, y, feature_names = generate_test_data_with_categorical(n_samples=300, n_categories=10, random_state=42)

    print(f"\n数据集形状: {X.shape}")
    print(f"类别分布: {Counter(y)}")

    rf = RandomForestFeatureImportance(n_estimators=20, max_depth=8, random_state=42)
    rf.fit(X, y)

    print(f"\n模型准确率: {rf.score(X, y):.4f}")

    print("\n1. 基尼不纯度重要性:")
    gini_importance = rf.get_feature_importance(feature_names)
    for name, score in gini_importance.items():
        marker = " ⚠️ 高估" if 'Useless' in name and score > 0.1 else ""
        print(f"  {name:20s}: {score:.6f}{marker}")

    print("\n2. 排列重要性:")
    rf.permutation_importance(X, y, n_repeats=10)
    perm_importance = rf.get_permutation_importance(feature_names)
    for name, score in perm_importance.items():
        marker = " ✅ 正确" if 'Useless' in name and score < 0.1 else ""
        print(f"  {name:20s}: {score:.6f}{marker}")

    print("\n3. SHAP全局重要性:")
    rf.shap_analysis(X[:50])
    shap_importance = rf.get_shap_global_importance(feature_names)
    for name, score in shap_importance.items():
        marker = " ✅ 正确" if 'Useless' in name and score < 0.1 else ""
        print(f"  {name:20s}: {score:.6f}{marker}")

    print("\n" + "=" * 80)
    print("三种方法总结对比：")
    print("=" * 80)
    print("| 方法          | 优点                              | 缺点                          |")
    print("|---------------|-----------------------------------|-------------------------------|")
    print("| 基尼不纯度    | 计算快，训练时直接计算            | 类别编码易高估，偏向高基数特征|")
    print("| 排列重要性    | 模型无关，真实反映性能贡献        | 计算慢，需多次预测            |")
    print("| SHAP          | 有理论依据，可解释单个样本        | 计算复杂，简化版有近似        |")
    print("=" * 80)
