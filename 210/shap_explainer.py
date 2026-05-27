import itertools
import math
import re
import numpy as np
from typing import Any, Callable, Optional, List, Tuple, Dict, Union


# ======================================================================
# 通用 SHAP 解释器（模型无关，排列采样法 — Kernel SHAP 思想）
# ======================================================================
class ShapExplainer:
    """基于 Shapley 值的模型可解释性工具（使用模型无关的排列采样法）。

    核心思想：对每个特征，枚举它在所有可能的特征子集 S 中作为最后一个
    加入者所带来的边际贡献，并按组合权重加权平均，得到该特征的 Shapley 值。

    对于树模型（XGBoost/LightGBM/CatBoost），可以通过传入一个背景数据集
    来估计缺失特征时的条件期望，从而得到更稳定、更接近真实的贡献。
    """

    def __init__(
        self,
        model: Any,
        background: Optional[np.ndarray] = None,
        predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_background: int = 100,
        feature_names: Optional[List[str]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        """
        参数:
            model:         训练好的黑箱模型（支持 .predict 的 sklearn 风格模型）。
            background:    背景数据集，用于近似特征缺失时的期望输出（形状 [N, F]）。
                           若为 None，则对缺失特征使用 0 填充，得到的是对 0 的偏离。
            predict_fn:    自定义预测函数，签名为 predict_fn(X) -> y。
                           若为 None，则使用 model.predict(X)。
            n_background:  从背景数据中随机抽取的样本数，用于加速。
            feature_names: 可选的特征名称列表。
            seed:          随机种子，保证可复现。
        """
        self.model = model
        self.feature_names = feature_names

        if predict_fn is None:
            if not hasattr(model, "predict"):
                raise ValueError("model 必须实现 predict 方法，或者提供 predict_fn。")
            self.predict_fn = lambda X: np.asarray(model.predict(X))
        else:
            self.predict_fn = predict_fn

        self.rng = np.random.default_rng(seed)

        if background is not None:
            background = np.asarray(background)
            if background.ndim != 2:
                raise ValueError("background 必须是二维数组 [N, F]。")
            n = min(n_background, len(background))
            idx = self.rng.choice(len(background), size=n, replace=False)
            self.background = background[idx]
        else:
            self.background = None

    # ------------------------------------------------------------------
    # 1. 暴力枚举：仅适用于特征数较少的情况（F <= 12 左右）
    # ------------------------------------------------------------------
    def explain_exact(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """对每个样本精确计算 Shapley 值（时间复杂度 O(F! * F)）。"""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n_samples, n_features = X.shape

        if self.background is not None and self.background.shape[1] != n_features:
            raise ValueError("背景数据的特征维度与输入不一致。")

        shap_values = np.zeros((n_samples, n_features), dtype=float)
        baseline = self._baseline()

        fact = math.factorial
        total = fact(n_features)

        for i in range(n_samples):
            x = X[i]
            for feature in range(n_features):
                phi = 0.0
                other_features = [j for j in range(n_features) if j != feature]
                for subset in itertools.chain.from_iterable(
                    itertools.combinations(other_features, r)
                    for r in range(n_features)
                ):
                    S = set(subset)
                    weight = fact(len(S)) * fact(n_features - len(S) - 1) / total
                    v_with = self._value(x, S | {feature})
                    v_without = self._value(x, S)
                    phi += weight * (v_with - v_without)
                shap_values[i, feature] = phi

        return shap_values, baseline

    # ------------------------------------------------------------------
    # 2. 采样近似：适用于高维特征（推荐默认方法）
    # ------------------------------------------------------------------
    def explain(
        self,
        X: np.ndarray,
        n_samples_per_feature: int = 200,
    ) -> Tuple[np.ndarray, float]:
        """通过随机排列采样近似 Shapley 值。

        对每个样本、每个特征，随机抽取若干次特征排列，
        计算该特征在当前排列中的边际贡献，并取平均。
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n_rows, n_features = X.shape

        if self.background is not None and self.background.shape[1] != n_features:
            raise ValueError("背景数据的特征维度与输入不一致。")

        shap_values = np.zeros((n_rows, n_features), dtype=float)
        baseline = self._baseline()

        for i in range(n_rows):
            x = X[i]
            for _ in range(n_samples_per_feature):
                perm = self.rng.permutation(n_features)
                current_S: set = set()
                prev_v = baseline
                for feat in perm:
                    current_S.add(feat)
                    cur_v = self._value(x, current_S)
                    shap_values[i, feat] += (cur_v - prev_v)
                    prev_v = cur_v

        shap_values /= n_samples_per_feature
        return shap_values, baseline

    # ------------------------------------------------------------------
    # 3. 辅助方法
    # ------------------------------------------------------------------
    def _baseline(self) -> float:
        if self.background is not None:
            preds = self.predict_fn(self.background)
            return float(np.mean(preds))
        return 0.0

    def _value(self, x: np.ndarray, S: set) -> float:
        n_features = len(x)
        S_sorted = sorted(S)

        if self.background is not None:
            X_in = self.background.copy()
            for s in S_sorted:
                X_in[:, s] = x[s]
            preds = self.predict_fn(X_in)
            return float(np.mean(preds))

        vec = np.zeros(n_features, dtype=float)
        for s in S_sorted:
            vec[s] = x[s]
        return float(self.predict_fn(vec.reshape(1, -1))[0])

    # ------------------------------------------------------------------
    # 4. 结果展示
    # ------------------------------------------------------------------
    def summary(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        baseline: float,
        top_k: Optional[int] = None,
    ) -> None:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]

        for i in range(len(X)):
            print(f"\n==== 样本 {i} ====")
            print(f"  基准值 E[f]       = {baseline:.6f}")
            preds = self.predict_fn(X[i : i + 1])
            print(f"  模型预测 f(x)     = {float(preds[0]):.6f}")
            total = baseline + float(np.sum(shap_values[i]))
            print(f"  基准 + ΣSHAP      = {total:.6f}  (应与预测值一致)")

            order = np.argsort(-np.abs(shap_values[i]))
            if top_k is not None:
                order = order[:top_k]
            print(f"  {'特征':<14}{'取值':>12}{'SHAP贡献':>14}")
            for idx in order:
                print(
                    f"  {names[idx]:<14}{X[i, idx]:>12.4f}{shap_values[i, idx]:>14.6f}"
                )


# ======================================================================
# 树结构定义
# ======================================================================
class TreeNode:
    """树节点。"""

    __slots__ = (
        "is_leaf",
        "feature",
        "threshold",
        "left",
        "right",
        "value",
        "n_samples",
        "left_n",
        "right_n",
        "default_left",
    )

    def __init__(self) -> None:
        self.is_leaf = False
        self.feature = -1
        self.threshold = 0.0
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None
        self.value = 0.0
        self.n_samples = 0
        self.left_n = 0
        self.right_n = 0
        self.default_left = True


class Tree:
    """单棵决策树。"""

    __slots__ = ("root", "max_depth", "n_leaves")

    def __init__(self, root: TreeNode, max_depth: int, n_leaves: int) -> None:
        self.root = root
        self.max_depth = max_depth
        self.n_leaves = n_leaves


# ======================================================================
# 树模型结构提取
# ======================================================================
def _extract_xgb_trees(model: Any) -> List[Tree]:
    """从 XGBoost 模型中提取树结构。"""
    booster = model.get_booster()
    dump = booster.get_dump(with_stats=True)
    trees: List[Tree] = []

    for tree_text in dump:
        nodes: Dict[int, TreeNode] = {}
        root_node: Optional[TreeNode] = None

        for line in tree_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"(\d+):\[([fF]\d+)<([^\]]+)\]", line)
            if m:
                node_id = int(m.group(1))
                feat_name = m.group(2)
                threshold = float(m.group(3))
                feat_idx = int(re.findall(r"\d+", feat_name)[0])

                stats_m = re.search(r"yes=(\d+),no=(\d+),missing=(\d+),gain=([^,]+),cover=([\d.eE+-]+)", line)
                if stats_m:
                    yes_id = int(stats_m.group(1))
                    no_id = int(stats_m.group(2))
                    missing_id = int(stats_m.group(3))
                    cover = float(stats_m.group(5))

                    if node_id not in nodes:
                        nodes[node_id] = TreeNode()
                    node = nodes[node_id]
                    node.is_leaf = False
                    node.feature = feat_idx
                    node.threshold = threshold
                    node.n_samples = cover
                    node.default_left = missing_id == yes_id

                    for child_id in (yes_id, no_id):
                        if child_id not in nodes:
                            nodes[child_id] = TreeNode()

                    node.left = nodes[yes_id]
                    node.right = nodes[no_id]

                    if root_node is None:
                        root_node = node
                continue

            m = re.match(r"(\d+):leaf=([^,]+),cover=([\d.eE+-]+)", line)
            if m:
                node_id = int(m.group(1))
                leaf_value = float(m.group(2))
                cover = float(m.group(3))

                if node_id not in nodes:
                    nodes[node_id] = TreeNode()
                node = nodes[node_id]
                node.is_leaf = True
                node.value = leaf_value
                node.n_samples = cover

        if root_node is None:
            continue

        def _fill_counts(node: TreeNode) -> None:
            if node.is_leaf:
                return
            if node.left is not None:
                _fill_counts(node.left)
                node.left_n = node.left.n_samples
            if node.right is not None:
                _fill_counts(node.right)
                node.right_n = node.right.n_samples

        _fill_counts(root_node)

        max_depth = 0
        n_leaves = 0

        def _stats(node: TreeNode, depth: int) -> None:
            nonlocal max_depth, n_leaves
            if depth > max_depth:
                max_depth = depth
            if node.is_leaf:
                n_leaves += 1
                return
            if node.left is not None:
                _stats(node.left, depth + 1)
            if node.right is not None:
                _stats(node.right, depth + 1)

        _stats(root_node, 1)
        trees.append(Tree(root_node, max_depth, n_leaves))

    return trees


def _extract_sklearn_trees(model: Any) -> List[Tree]:
    """从 sklearn 集成模型中提取树结构。"""
    trees: List[Tree] = []

    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return trees

    if hasattr(model, "n_estimators") and hasattr(estimators, "ndim"):
        est_list = estimators.ravel()
    else:
        est_list = list(estimators)

    for est in est_list:
        sk_tree = getattr(est, "tree_", None)
        if sk_tree is None:
            continue
        root = _build_sklearn_node(sk_tree, 0)
        max_depth = int(sk_tree.max_depth) if hasattr(sk_tree, "max_depth") else 0
        n_leaves = int(sk_tree.n_leaves) if hasattr(sk_tree, "n_leaves") else 0
        trees.append(Tree(root, max_depth, n_leaves))

    return trees


def _build_sklearn_node(sk_tree: Any, node_id: int) -> TreeNode:
    node = TreeNode()
    children_left = sk_tree.children_left
    children_right = sk_tree.children_right
    feature = sk_tree.feature
    threshold = sk_tree.threshold
    value = sk_tree.value
    n_node_samples = sk_tree.n_node_samples

    node.n_samples = float(n_node_samples[node_id])

    if children_left[node_id] == children_right[node_id]:
        node.is_leaf = True
        val = value[node_id]
        if val.ndim == 3:
            node.value = float(val[0, 0, 0])
        elif val.ndim == 2:
            node.value = float(val[0, 0])
        else:
            node.value = float(val.flat[0])
        return node

    node.is_leaf = False
    node.feature = int(feature[node_id])
    node.threshold = float(threshold[node_id])

    left_id = children_left[node_id]
    right_id = children_right[node_id]

    node.left = _build_sklearn_node(sk_tree, left_id)
    node.right = _build_sklearn_node(sk_tree, right_id)

    node.left_n = float(n_node_samples[left_id])
    node.right_n = float(n_node_samples[right_id])
    node.default_left = True

    return node


def extract_trees(model: Any) -> List[Tree]:
    """从模型中提取树结构列表，返回空列表表示不是树模型。"""
    if hasattr(model, "get_booster"):
        try:
            return _extract_xgb_trees(model)
        except Exception:
            pass
    if hasattr(model, "estimators_"):
        try:
            return _extract_sklearn_trees(model)
        except Exception:
            pass
    return []


# ======================================================================
# TreeSHAP —— O(T * L * D²) 多项式时间树模型解释器
# ======================================================================
class TreeShapExplainer:
    """TreeSHAP 解释器。

    基于 Lundberg 等人 (2020) "From local explanations to global understanding
    with explainable AI for trees" 提出的高效 TreeSHAP 算法。

    核心原理：
        对每棵树，枚举样本路径上特征的所有子集（O(D * 2^D)，D 为树深度），
        通过条件期望计算每个特征的 Shapley 贡献。相比暴力枚举 O(2^M)，
        将指数从总特征数 M 降到树深度 D（通常 D <= 12），避免指数爆炸。

    时间复杂度：O(T * D * 2^D)
        T = 树的数量, D = 树的路径深度（通常 <= 12）

    Interventional 模式（interventional=True）：
        不使用树内部的样本计数，而用背景数据重新估计每个节点的
        左右子样本比例，消除特征相关性偏差。
    """

    def __init__(
        self,
        model: Any,
        background: Optional[np.ndarray] = None,
        predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        interventional: bool = False,
        seed: Optional[int] = 42,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.interventional = interventional
        self.rng = np.random.default_rng(seed)
        self._trees = extract_trees(model)

        if predict_fn is None:
            if not hasattr(model, "predict"):
                raise ValueError("model 必须实现 predict 方法，或者提供 predict_fn。")
            self.predict_fn = lambda X: np.asarray(model.predict(X))
        else:
            self.predict_fn = predict_fn

        if background is not None:
            background = np.asarray(background)
            if background.ndim != 2:
                raise ValueError("background 必须是二维数组 [N, F]。")
            self.background = background
        else:
            self.background = None

        if not self._trees:
            raise ValueError(
                "无法从模型中提取树结构。仅支持 XGBoost 和 sklearn 集成模型。"
            )

        self._n_features = self._infer_n_features()

        if interventional and background is None:
            raise ValueError(
                "interventional=True 时必须提供 background 数据。"
            )

        self._node_counts_cache: Dict[int, Tuple[float, float]] = {}
        if interventional and self.background is not None:
            self._precompute_interventional_counts()

        self._tree_scale = self._detect_tree_scale()

    def _detect_tree_scale(self) -> float:
        if hasattr(self.model, "get_booster"):
            return float(getattr(self.model, "learning_rate", 0.3))
        if hasattr(self.model, "estimators_"):
            n_est = len(self.model.estimators_)
            if hasattr(self.model, "learning_rate"):
                return float(self.model.learning_rate)
            return 1.0 / n_est
        return 1.0

    def _infer_n_features(self) -> int:
        n_feat = 0

        def _walk(node: TreeNode) -> None:
            nonlocal n_feat
            if node.is_leaf:
                return
            if node.feature >= 0:
                n_feat = max(n_feat, node.feature + 1)
            if node.left:
                _walk(node.left)
            if node.right:
                _walk(node.right)

        for tree in self._trees:
            _walk(tree.root)
        return n_feat

    def _precompute_interventional_counts(self) -> None:
        assert self.background is not None
        self._node_counts_cache.clear()
        for tree in self._trees:
            self._compute_node_counts(tree.root)

    def _compute_node_counts(self, node: TreeNode) -> None:
        if node.is_leaf:
            self._node_counts_cache[id(node)] = (float(len(self.background)), 0.0)
            return
        assert self.background is not None
        feat_vals = self.background[:, node.feature]
        n_left = float(np.sum(feat_vals <= node.threshold))
        n_right = float(len(self.background) - n_left)
        self._node_counts_cache[id(node)] = (n_left, n_right)
        if node.left is not None:
            self._compute_node_counts(node.left)
        if node.right is not None:
            self._compute_node_counts(node.right)

    def _get_node_counts(
        self, node: TreeNode, tree_idx: int
    ) -> Tuple[float, float]:
        if node.is_leaf:
            n_bg = float(len(self.background)) if self.background is not None else 1.0
            return (n_bg, 0.0)
        if self.interventional:
            return self._node_counts_cache.get(id(node), (node.left_n, node.right_n))
        return (node.left_n, node.right_n)

    # ------------------------------------------------------------------
    # 核心：单棵树的 SHAP 贡献计算（正确算法）
    # ------------------------------------------------------------------
    def _tree_shap(
        self, tree: Tree, x: np.ndarray, tree_idx: int
    ) -> np.ndarray:
        """对单棵树计算各特征对样本 x 的 SHAP 贡献。

        正确算法（O(T * D * 2^D)，其中 D 为树深度，远小于 M）：
            1. 找到样本 x 在树中的路径（root -> leaf）
            2. 对路径上每个特征 f_j，枚举路径上其他特征的所有子集 T
            3. 计算 v(T) 和 v(T ∪ {f_j})（条件期望）
            4. 使用调整后的 Shapley 权重累加贡献

        条件期望 v(S)：从根节点开始递归，对每个节点：
            - 若划分特征 ∈ S，按 x 的值确定性走左/右子树
            - 否则按左右子样本比例加权平均
        """
        shap = np.zeros(self._n_features, dtype=float)
        M = self._n_features

        path_internal = []
        node = tree.root
        while not node.is_leaf:
            f = node.feature
            go_left = x[f] <= node.threshold
            if node.default_left and np.isnan(x[f]):
                go_left = True
            path_internal.append((node, f, go_left))
            node = node.left if go_left else node.right

        unique_path_features = list(dict.fromkeys(f for _, f, _ in path_internal))
        D = len(unique_path_features)
        K = M - D

        W = np.zeros(D, dtype=float)
        for t in range(D):
            total = 0.0
            for k in range(K + 1):
                comb = math.comb(K, k) if K > 0 else 1
                num = math.factorial(t + k) * math.factorial(M - t - k - 1)
                total += comb * num
            W[t] = total / math.factorial(M)

        cache: Dict[Tuple[int, frozenset], float] = {}

        def expected_value(node: TreeNode, known: frozenset) -> float:
            if node.is_leaf:
                return node.value
            key = (id(node), known)
            if key in cache:
                return cache[key]

            f = node.feature
            n_left, n_right = self._get_node_counts(node, tree_idx)
            n_total = n_left + n_right
            if n_total < 1e-12:
                n_total = 1.0
            a = n_left / n_total

            if f in known:
                go_left = x[f] <= node.threshold
                if node.default_left and np.isnan(x[f]):
                    go_left = True
                child = node.left if go_left else node.right
                result = expected_value(child, known)
            else:
                result = a * expected_value(node.left, known) + (1.0 - a) * expected_value(node.right, known)

            cache[key] = result
            return result

        for j, f_j in enumerate(unique_path_features):
            other_features = [unique_path_features[k] for k in range(D) if k != j]

            for r in range(len(other_features) + 1):
                for combo in itertools.combinations(other_features, r):
                    T = frozenset(combo)
                    t = len(T)
                    v_T = expected_value(tree.root, T)
                    v_T_with_f = expected_value(tree.root, T | frozenset([f_j]))
                    shap[f_j] += W[t] * (v_T_with_f - v_T)

        return shap

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """对所有样本计算 SHAP 值。

        返回：
            shap_values: (n_samples, n_features) 的 SHAP 贡献数组
            baseline:    模型的期望预测值 E[f]
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n_samples = X.shape[0]

        if X.shape[1] != self._n_features:
            raise ValueError(
                f"输入特征维度 {X.shape[1]} 与模型特征维度 {self._n_features} 不一致。"
            )

        shap_values = np.zeros((n_samples, self._n_features), dtype=float)

        for i in range(n_samples):
            x = X[i]
            for t_idx, tree in enumerate(self._trees):
                phi = self._tree_shap(tree, x, t_idx)
                shap_values[i] += self._tree_scale * phi

        baseline = self._baseline()
        return shap_values, baseline

    def _baseline(self) -> float:
        total = 0.0
        n_trees = len(self._trees)
        if n_trees == 0:
            return 0.0
        for tree in self._trees:
            total += self._tree_expected(tree.root)
        return self._tree_scale * total

    def _tree_expected(self, node: TreeNode) -> float:
        if node.is_leaf:
            return node.value
        assert node.left is not None and node.right is not None
        n_left, n_right = self._get_node_counts(node, 0)
        total = n_left + n_right
        if total < 1e-12:
            return 0.0
        return (n_left / total) * self._tree_expected(
            node.left
        ) + (n_right / total) * self._tree_expected(node.right)

    def summary(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        baseline: float,
        top_k: Optional[int] = None,
    ) -> None:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        names = self.feature_names or [f"f{i}" for i in range(self._n_features)]

        for i in range(len(X)):
            print(f"\n==== 样本 {i} ====")
            print(f"  基准值 E[f]       = {baseline:.6f}")
            preds = self.predict_fn(X[i : i + 1])
            print(f"  模型预测 f(x)     = {float(preds[0]):.6f}")
            total = baseline + float(np.sum(shap_values[i]))
            print(f"  基准 + ΣSHAP      = {total:.6f}  (应与预测值一致)")

            order = np.argsort(-np.abs(shap_values[i]))
            if top_k is not None:
                order = order[:top_k]
            print(f"  {'特征':<14}{'取值':>12}{'SHAP贡献':>14}")
            for idx in order:
                print(
                    f"  {names[idx]:<14}{X[i, idx]:>12.4f}{shap_values[i, idx]:>14.6f}"
                )


# ======================================================================
# 便捷入口：自动选择最优解释器
# ======================================================================
def auto_explain(
    model: Any,
    X: np.ndarray,
    background: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    interventional: bool = False,
    n_samples_per_feature: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, float, str]:
    """自动检测模型类型并选择最优的 SHAP 计算方法。

    返回：
        shap_values: SHAP 贡献数组
        baseline:    基准值
        method:      使用的方法名（"TreeSHAP" 或 "KernelSHAP"）
    """
    trees = extract_trees(model)
    if trees:
        explainer = TreeShapExplainer(
            model=model,
            background=background,
            feature_names=feature_names,
            interventional=interventional,
            seed=seed,
        )
        sv, bl = explainer.explain(X)
        return sv, bl, "TreeSHAP"
    else:
        explainer = ShapExplainer(
            model=model,
            background=background,
            feature_names=feature_names,
            seed=seed,
        )
        sv, bl = explainer.explain(
            X, n_samples_per_feature=n_samples_per_feature
        )
        return sv, bl, "KernelSHAP"


# ======================================================================
# SHAP 全局分析与交互检测（用于合规性验证）
# ======================================================================
class ShapAnalyzer:
    """SHAP 全局分析器：特征重要性、交互作用、依赖关系。

    用于医学诊断/信用评分等受监管领域的模型合规性验证：
    - 识别关键特征及其全局影响
    - 检测特征间的交互作用（潜在偏差来源）
    - 分析特征依赖关系（单调性、阈值效应）
    """

    def __init__(
        self,
        model: Any,
        X: np.ndarray,
        background: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        shap_values: Optional[np.ndarray] = None,
        baseline: Optional[float] = None,
        interventional: bool = False,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.X = np.atleast_2d(np.asarray(X, dtype=float))
        self.n_samples, self.n_features = self.X.shape
        self.feature_names = feature_names or [f"f{i}" for i in range(self.n_features)]
        self.rng = np.random.default_rng(seed)

        if shap_values is not None and baseline is not None:
            self.shap_values = np.asarray(shap_values)
            self.baseline = float(baseline)
        else:
            result = auto_explain(
                model=model,
                X=self.X,
                background=background,
                feature_names=self.feature_names,
                interventional=interventional,
                seed=seed,
            )
            self.shap_values = result[0]
            self.baseline = float(result[1])

        self.predictions = model.predict(self.X)
        self._interaction_matrix: Optional[np.ndarray] = None
        self._interaction_pairs: Optional[List[Tuple[int, int, float]]] = None

    # ------------------------------------------------------------------
    # 1. 全局特征重要性
    # ------------------------------------------------------------------
    def global_importance(self) -> np.ndarray:
        """计算全局特征重要性（|SHAP|均值）。

        返回：
            importance: 长度为 n_features 的数组，值越大表示特征越重要
        """
        return np.mean(np.abs(self.shap_values), axis=0)

    def global_importance_rank(self) -> List[Tuple[str, float, int]]:
        """返回排序后的特征重要性列表。

        返回：
            [(feature_name, importance, feature_index), ...] 按重要性降序排列
        """
        importance = self.global_importance()
        indices = np.argsort(-importance)
        return [
            (self.feature_names[i], float(importance[i]), int(i))
            for i in indices
        ]

    # ------------------------------------------------------------------
    # 2. 特征交互作用检测
    # ------------------------------------------------------------------
    def compute_interactions(
        self,
        n_samples: Optional[int] = None,
        method: str = "auto",
    ) -> np.ndarray:
        """计算 SHAP 交互值矩阵（n_features × n_features）。

        交互值 Φ_{ij} 表示特征 i 和 j 之间的交互作用强度：
            - Φ_{ii} = 特征 i 的主效应
            - Φ_{ij} = 特征 i 与 j 的交互效应（i ≠ j）
            - Σ_j Φ_{ij} ≈ SHAP_i（交互效应之和近似等于主效应）

        参数:
            n_samples: 用于交互估计的样本数（None 表示使用全部）
            method: "tree" | "sampling" | "auto"
                    tree - 使用树结构快速估计（仅支持树模型）
                    sampling - 通用采样方法
                    auto - 自动选择

        返回：
            interaction_matrix: (n_features, n_features) 矩阵
        """
        n = n_samples or min(self.n_samples, 100)
        indices = self.rng.choice(self.n_samples, size=n, replace=False)
        X_sub = self.X[indices]

        if method == "auto":
            trees = extract_trees(self.model)
            method = "tree" if trees else "sampling"

        if method == "tree":
            return self._tree_interactions(X_sub)
        else:
            return self._sampling_interactions(X_sub)

    def _tree_interactions(self, X_sub: np.ndarray) -> np.ndarray:
        """基于树结构的交互作用估计。

        原理：如果两个特征经常出现在同一条路径上（祖先-后代关系），
        则它们之间存在交互作用。
        """
        trees = extract_trees(self.model)
        if not trees:
            return self._sampling_interactions(X_sub)

        n = len(X_sub)
        interactions = np.zeros((self.n_features, self.n_features), dtype=float)

        for idx in range(n):
            x = X_sub[idx]
            for tree in trees:
                path_features = []
                node = tree.root
                while not node.is_leaf:
                    f = node.feature
                    go_left = x[f] <= node.threshold
                    path_features.append(f)
                    node = node.left if go_left else node.right

                unique_path = list(dict.fromkeys(path_features))
                for i, fi in enumerate(unique_path):
                    for j, fj in enumerate(unique_path):
                        interactions[fi, fj] += 1.0

        for i in range(self.n_features):
            row_sum = np.sum(interactions[i])
            if row_sum > 0:
                interactions[i] /= row_sum

        return interactions

    def _sampling_interactions(self, X_sub: np.ndarray) -> np.ndarray:
        """基于采样的通用交互作用估计方法。

        使用 H-statistic 思想：比较联合效应与边际效应之和的差异。
        """
        n = len(X_sub)
        interactions = np.zeros((self.n_features, self.n_features), dtype=float)

        if self.background is None:
            for i in range(self.n_features):
                interactions[i, i] = np.mean(
                    np.abs(self.shap_values[indices[:n], i])
                ) if hasattr(self, 'indices') else np.mean(
                    np.abs(self.shap_values[:n, i])
                )
            return interactions

        bg = self.background[: min(len(self.background), 50)]

        for i in range(self.n_features):
            for j in range(self.n_features):
                if i == j:
                    interactions[i, j] = np.mean(
                        np.abs(self.shap_values[:n, i])
                    )
                else:
                    inter_strength = 0.0
                    for idx in range(min(n, 20)):
                        x = X_sub[idx].copy()
                        x_both = x.copy()
                        x_only_i = x.copy()
                        x_only_j = x.copy()
                        x_none = x.copy()

                        x_only_i[j] = bg[0, j]
                        x_only_j[i] = bg[0, i]
                        x_none[i] = bg[0, i]
                        x_none[j] = bg[0, j]

                        pred_both = self.model.predict(x_both.reshape(1, -1))[0]
                        pred_only_i = self.model.predict(x_only_i.reshape(1, -1))[0]
                        pred_only_j = self.model.predict(x_only_j.reshape(1, -1))[0]
                        pred_none = self.model.predict(x_none.reshape(1, -1))[0]

                        joint = pred_both - pred_none
                        marginal = (pred_only_i - pred_none) + (pred_only_j - pred_none)
                        inter_strength += abs(joint - marginal)

                    interactions[i, j] = inter_strength / min(n, 20)

        return interactions

    def top_interactions(
        self, n_pairs: int = 10, recompute: bool = False
    ) -> List[Tuple[str, str, float]]:
        """返回交互作用最强的特征对。

        参数:
            n_pairs: 返回的交互对数量
            recompute: 是否重新计算交互矩阵

        返回：
            [(feature_i, feature_j, interaction_strength), ...]
        """
        if self._interaction_matrix is None or recompute:
            self._interaction_matrix = self.compute_interactions()

        pairs = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                strength = (
                    self._interaction_matrix[i, j] + self._interaction_matrix[j, i]
                ) / 2
                pairs.append((i, j, float(strength)))

        pairs.sort(key=lambda x: x[2], reverse=True)
        self._interaction_pairs = pairs

        return [
            (self.feature_names[i], self.feature_names[j], s)
            for i, j, s in pairs[:n_pairs]
        ]

    # ------------------------------------------------------------------
    # 3. 特征依赖图数据
    # ------------------------------------------------------------------
    def dependence_data(
        self, feature: Union[int, str], interaction_feature: Optional[Union[int, str]] = None
    ) -> Dict[str, np.ndarray]:
        """生成特征依赖图的数据。

        参数:
            feature: 目标特征（索引或名称）
            interaction_feature: 可选的交互特征（用于着色）

        返回：
            {"feature_values": ..., "shap_values": ..., "interaction_values": ...}
        """
        if isinstance(feature, str):
            feat_idx = self.feature_names.index(feature)
        else:
            feat_idx = feature

        result: Dict[str, np.ndarray] = {
            "feature_values": self.X[:, feat_idx],
            "shap_values": self.shap_values[:, feat_idx],
        }

        if interaction_feature is not None:
            if isinstance(interaction_feature, str):
                inter_idx = self.feature_names.index(interaction_feature)
            else:
                inter_idx = interaction_feature
            result["interaction_values"] = self.X[:, inter_idx]

        return result

    def find_best_interaction(self, feature: Union[int, str]) -> Tuple[str, float]:
        """找到与给定特征交互最强的特征。

        返回：
            (best_interaction_feature_name, interaction_strength)
        """
        if isinstance(feature, str):
            feat_idx = self.feature_names.index(feature)
        else:
            feat_idx = feature

        if self._interaction_matrix is None:
            self._interaction_matrix = self.compute_interactions()

        interactions = self._interaction_matrix[feat_idx].copy()
        interactions[feat_idx] = 0
        best_idx = int(np.argmax(interactions))

        return self.feature_names[best_idx], float(interactions[best_idx])

    # ------------------------------------------------------------------
    # 4. 单调性检测（合规性验证）
    # ------------------------------------------------------------------
    def check_monotonicity(
        self, feature: Union[int, str], n_bins: int = 10
    ) -> Dict[str, Any]:
        """检测特征对预测的影响是否单调（合规性要求）。

        在信用评分中，某些特征（如收入）应该与评分呈单调关系。
        """
        if isinstance(feature, str):
            feat_idx = self.feature_names.index(feature)
        else:
            feat_idx = feature

        feat_vals = self.X[:, feat_idx]
        shap_vals = self.shap_values[:, feat_idx]

        bins = np.linspace(feat_vals.min(), feat_vals.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (feat_vals >= bins[i]) & (feat_vals < bins[i + 1])
            if i == n_bins - 1:
                mask = (feat_vals >= bins[i]) & (feat_vals <= bins[i + 1])
            if np.sum(mask) > 0:
                bin_means[i] = np.mean(shap_vals[mask])
                bin_counts[i] = np.sum(mask)

        valid = bin_counts > 0
        if np.sum(valid) >= 3:
            x_valid = bin_centers[valid]
            y_valid = bin_means[valid]
            diffs = np.diff(y_valid)
            increasing = np.all(diffs >= -1e-10)
            decreasing = np.all(diffs <= 1e-10)

            if increasing:
                direction = "increasing"
            elif decreasing:
                direction = "decreasing"
            else:
                direction = "non-monotonic"
        else:
            direction = "insufficient_data"

        return {
            "feature": self.feature_names[feat_idx],
            "direction": direction,
            "bin_centers": bin_centers,
            "bin_means": bin_means,
            "bin_counts": bin_counts,
        }

    # ------------------------------------------------------------------
    # 5. 合规性报告（医学诊断/信用评分）
    # ------------------------------------------------------------------
    def compliance_report(
        self,
        threshold_features: Optional[List[Union[int, str]]] = None,
        expected_directions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """生成合规性验证报告。

        参数:
            threshold_features: 需要检查单调性的关键特征列表
            expected_directions: 期望的单调性方向 {"feature": "increasing"/"decreasing"}

        返回：
            合规性报告字典
        """
        report: Dict[str, Any] = {
            "model_summary": {
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "baseline_prediction": self.baseline,
                "mean_prediction": float(np.mean(self.predictions)),
                "min_prediction": float(np.min(self.predictions)),
                "max_prediction": float(np.max(self.predictions)),
            },
            "feature_importance": self.global_importance_rank()[:10],
            "top_interactions": self.top_interactions(10),
            "monotonicity_checks": [],
            "concerns": [],
        }

        if threshold_features:
            for feat in threshold_features:
                if isinstance(feat, int):
                    feat_name = self.feature_names[feat]
                else:
                    feat_name = feat

                mono_result = self.check_monotonicity(feat)
                report["monotonicity_checks"].append(mono_result)

                if expected_directions and feat_name in expected_directions:
                    expected = expected_directions[feat_name]
                    actual = mono_result["direction"]
                    if actual != expected and actual != "insufficient_data":
                        report["concerns"].append(
                            f"特征 '{feat_name}' 期望方向 {expected}，实际为 {actual}"
                        )

        importance = self.global_importance()
        zero_importance = [
            self.feature_names[i]
            for i in range(self.n_features)
            if importance[i] < 1e-10
        ]
        if zero_importance:
            report["concerns"].append(
                f"以下特征对模型无贡献: {zero_importance}"
            )

        report["is_compliant"] = len(report["concerns"]) == 0

        return report

    def print_report(self, report: Optional[Dict[str, Any]] = None) -> None:
        """打印合规性报告到控制台。"""
        if report is None:
            report = self.compliance_report()

        print("\n" + "=" * 70)
        print("  SHAP 合规性验证报告")
        print("=" * 70)

        s = report["model_summary"]
        print(f"\n  样本数: {s['n_samples']}, 特征数: {s['n_features']}")
        print(f"  基准预测: {s['baseline_prediction']:.4f}")
        print(f"  预测范围: [{s['min_prediction']:.4f}, {s['max_prediction']:.4f}]")
        print(f"  平均预测: {s['mean_prediction']:.4f}")

        print("\n  --- 特征重要性 Top 10 ---")
        for name, imp, idx in report["feature_importance"]:
            bar = "█" * int(imp * 50 / max(1, report["feature_importance"][0][1]))
            print(f"  {name:<12} {imp:.6f}  {bar}")

        print("\n  --- 交互作用 Top 10 ---")
        for f1, f2, strength in report["top_interactions"]:
            bar = "█" * int(strength * 30) if strength > 0 else ""
            print(f"  {f1} ↔ {f2:<12} {strength:.6f}  {bar}")

        if report["monotonicity_checks"]:
            print("\n  --- 单调性检查 ---")
            for mc in report["monotonicity_checks"]:
                print(f"  {mc['feature']:<12}: {mc['direction']}")

        if report["concerns"]:
            print("\n  ⚠ 合规性警告:")
            for concern in report["concerns"]:
                print(f"    - {concern}")
        else:
            print("\n  ✓ 未发现合规性问题")

        status = "✓ 合规" if report["is_compliant"] else "✗ 不合规"
        print(f"\n  总体评估: {status}")
        print("=" * 70 + "\n")


# ======================================================================
# 可视化函数（可选依赖 matplotlib）
# ======================================================================
def plot_feature_importance(
    analyzer: ShapAnalyzer,
    max_display: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Any:
    """绘制全局特征重要性柱状图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib: pip install matplotlib")
        return None

    importance = analyzer.global_importance_rank()[:max_display]
    names = [item[0] for item in importance]
    values = [item[1] for item in importance]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    y_pos = range(len(names))
    ax.barh(y_pos, values, color="#2196F3")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_dependence(
    analyzer: ShapAnalyzer,
    feature: Union[int, str],
    interaction_feature: Optional[Union[int, str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Any:
    """绘制特征依赖图（SHAP值 vs 特征值）。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib: pip install matplotlib")
        return None

    data = analyzer.dependence_data(feature, interaction_feature)

    fig, ax = plt.subplots(figsize=(10, 6))

    if interaction_feature and "interaction_values" in data:
        scatter = ax.scatter(
            data["feature_values"],
            data["shap_values"],
            c=data["interaction_values"],
            cmap="coolwarm",
            alpha=0.6,
            s=20,
        )
        plt.colorbar(scatter, label=f"{interaction_feature} value")
    else:
        ax.scatter(
            data["feature_values"],
            data["shap_values"],
            alpha=0.6,
            s=20,
            color="#2196F3",
        )

    feature_name = feature if isinstance(feature, str) else analyzer.feature_names[feature]
    ax.set_xlabel(f"{feature_name} value")
    ax.set_ylabel(f"SHAP value of {feature_name}")
    ax.set_title(f"Dependence Plot: {feature_name}")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_interaction_heatmap(
    analyzer: ShapAnalyzer,
    save_path: Optional[str] = None,
    show: bool = True,
    max_features: int = 15,
) -> Any:
    """绘制特征交互热力图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib: pip install matplotlib")
        return None

    if analyzer._interaction_matrix is None:
        analyzer.compute_interactions()

    interaction = analyzer._interaction_matrix.copy()
    np.fill_diagonal(interaction, 0)

    if interaction.shape[0] > max_features:
        importance = analyzer.global_importance()
        top_indices = np.argsort(-importance)[:max_features]
        interaction = interaction[np.ix_(top_indices, top_indices)]
        names = [analyzer.feature_names[i] for i in top_indices]
    else:
        names = analyzer.feature_names

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), max(6, len(names) * 0.4)))
    im = ax.imshow(interaction, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Feature Interaction Heatmap")
    plt.colorbar(im, ax=ax, label="Interaction strength")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_summary_beeswarm(
    analyzer: ShapAnalyzer,
    max_display: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Any:
    """绘制 SHAP 摘要图（beeswarm 风格）。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib: pip install matplotlib")
        return None

    importance = analyzer.global_importance()
    top_indices = np.argsort(-importance)[:max_display]

    fig, ax = plt.subplots(figsize=(12, max(6, max_display * 0.4)))

    for pos, feat_idx in enumerate(top_indices):
        shap_vals = analyzer.shap_values[:, feat_idx]
        feat_vals = analyzer.X[:, feat_idx]

        feat_vals_norm = (feat_vals - feat_vals.min()) / (
            feat_vals.max() - feat_vals.min() + 1e-10
        )

        jitter = np.random.normal(0, 0.04, len(shap_vals))
        colors = plt.cm.coolwarm(feat_vals_norm)

        ax.scatter(
            shap_vals,
            np.full(len(shap_vals), pos) + jitter,
            c=colors,
            alpha=0.5,
            s=15,
        )

    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([analyzer.feature_names[i] for i in top_indices])
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("SHAP Summary Plot")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ======================================================================
# 使用示例
# ======================================================================
if __name__ == "__main__":
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None

    rng = np.random.default_rng(0)

    N, F = 500, 5
    X_all = rng.normal(size=(N, F))
    y_all = (
        2.0 * X_all[:, 0]
        - 1.5 * X_all[:, 1]
        + 0.8 * X_all[:, 2] * X_all[:, 3]
        + 0.3 * X_all[:, 4]
        + rng.normal(scale=0.1, size=N)
    )

    split = int(0.8 * N)
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    feature_names = ["f0", "f1", "f2", "f3", "f4"]

    if xgb is not None:
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=0,
        )
        model.fit(X_train, y_train)
    else:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=50, random_state=0)
        model.fit(X_train, y_train)

    X_sample = X_test[:3]

    print("=" * 60)
    print("  使用 TreeShapExplainer（O(T*L*D²)，避免指数爆炸）")
    print("=" * 60)

    tree_explainer = TreeShapExplainer(
        model=model,
        background=X_train[:100],
        feature_names=feature_names,
        interventional=False,
        seed=42,
    )
    shap_tree, base_tree = tree_explainer.explain(X_sample)
    tree_explainer.summary(X_sample, shap_tree, base_tree)

    print("\n" + "=" * 60)
    print("  使用 ShapExplainer 采样近似（Kernel SHAP）")
    print("=" * 60)

    kernel_explainer = ShapExplainer(
        model=model,
        background=X_train[:100],
        feature_names=feature_names,
        seed=42,
    )
    shap_kernel, base_kernel = kernel_explainer.explain(
        X_sample, n_samples_per_feature=500
    )
    kernel_explainer.summary(X_sample, shap_kernel, base_kernel)

    print("\n" + "=" * 60)
    print("  使用 auto_explain 自动选择方法")
    print("=" * 60)

    sv_auto, bl_auto, method = auto_explain(
        model=model,
        X=X_sample,
        background=X_train[:100],
        feature_names=feature_names,
        seed=42,
    )
    print(f"自动选择方法: {method}")
    print(f"基准值: {bl_auto:.6f}")
    for i in range(len(X_sample)):
        print(f"样本 {i}: SHAP sum = {np.sum(sv_auto[i]):.6f}, "
              f"pred = {model.predict(X_sample[i:i+1])[0]:.6f}, "
              f"baseline + sum = {bl_auto + np.sum(sv_auto[i]):.6f}")

    # ======================================================================
    # 新增功能：全局分析、交互检测、合规性验证
    # ======================================================================
    print("\n" + "=" * 70)
    print("  使用 ShapAnalyzer 进行全局分析与合规性验证")
    print("=" * 70)

    analyzer = ShapAnalyzer(
        model=model,
        X=X_test,
        background=X_train[:100],
        feature_names=feature_names,
        seed=42,
    )

    # 1. 全局特征重要性
    print("\n" + "-" * 50)
    print("  全局特征重要性")
    print("-" * 50)
    importance = analyzer.global_importance_rank()
    for name, imp, idx in importance:
        print(f"  {name:<12}: {imp:.6f}")

    # 2. 特征交互作用
    print("\n" + "-" * 50)
    print("  Top 10 特征交互作用")
    print("-" * 50)
    top_interactions = analyzer.top_interactions(10)
    for f1, f2, strength in top_interactions:
        if strength > 0.01:
            print(f"  {f1:<6} ↔ {f2:<6}: {strength:.4f}")

    # 3. 依赖图数据示例
    print("\n" + "-" * 50)
    print("  特征依赖图数据（f0）")
    print("-" * 50)
    dep_data = analyzer.dependence_data("f0")
    print(f"  特征值范围: [{dep_data['feature_values'].min():.4f}, {dep_data['feature_values'].max():.4f}]")
    print(f"  SHAP值范围: [{dep_data['shap_values'].min():.4f}, {dep_data['shap_values'].max():.4f}]")

    # 4. 单调性检测
    print("\n" + "-" * 50)
    print("  单调性检测（合规性验证）")
    print("-" * 50)
    for feat in feature_names:
        mono = analyzer.check_monotonicity(feat)
        print(f"  {feat:<12}: {mono['direction']}")

    # 5. 合规性报告
    print("\n" + "-" * 50)
    print("  合规性报告")
    print("-" * 50)
    report = analyzer.compliance_report(
        threshold_features=["f0", "f1"],
        expected_directions={"f0": "increasing", "f1": "decreasing"},
    )
    analyzer.print_report(report)

    # 6. 可视化（如果 matplotlib 可用）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\n" + "-" * 50)
        print("  生成可视化图表")
        print("-" * 50)

        plot_feature_importance(analyzer, save_path="e:/temp/record10/210/feature_importance.png", show=False)
        print("  ✓ 特征重要性图已保存: feature_importance.png")

        plot_dependence(analyzer, "f0", interaction_feature="f1",
                        save_path="e:/temp/record10/210/dependence_f0.png", show=False)
        print("  ✓ 依赖图已保存: dependence_f0.png")

        plot_interaction_heatmap(analyzer,
                                  save_path="e:/temp/record10/210/interaction_heatmap.png", show=False)
        print("  ✓ 交互热力图已保存: interaction_heatmap.png")

        plot_summary_beeswarm(analyzer,
                               save_path="e:/temp/record10/210/summary_beeswarm.png", show=False)
        print("  ✓ 摘要图已保存: summary_beeswarm.png")

        plt.close("all")
    except ImportError:
        print("\n  matplotlib 未安装，跳过可视化")
    except Exception as e:
        print(f"\n  可视化出错: {e}")

    print("\n" + "=" * 70)
    print("  所有分析完成")
    print("=" * 70)
