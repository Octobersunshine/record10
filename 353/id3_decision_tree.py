import math
import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class DecisionTree:
    def __init__(self, algorithm="c4.5"):
        if algorithm not in ("id3", "c4.5"):
            raise ValueError("algorithm must be 'id3' or 'c4.5'")
        self.algorithm = algorithm
        self.tree = None
        self.feature_names = None
        self.feature_types = None

    def _entropy(self, data):
        labels = [row[-1] for row in data]
        label_counts = Counter(labels)
        entropy = 0.0
        total = len(data)
        for count in label_counts.values():
            prob = count / total
            entropy -= prob * math.log2(prob)
        return entropy

    def _split_data_discrete(self, data, feature_index, value):
        return [row for row in data if row[feature_index] == value]

    def _split_data_continuous(self, data, feature_index, threshold, direction):
        if direction == "le":
            return [row for row in data if row[feature_index] <= threshold]
        else:
            return [row for row in data if row[feature_index] > threshold]

    def _information_gain_discrete(self, data, feature_index):
        base_entropy = self._entropy(data)
        feature_values = set(row[feature_index] for row in data)
        new_entropy = 0.0
        total = len(data)
        for value in feature_values:
            sub_data = self._split_data_discrete(data, feature_index, value)
            prob = len(sub_data) / total
            new_entropy += prob * self._entropy(sub_data)
        return base_entropy - new_entropy

    def _split_info_discrete(self, data, feature_index):
        feature_values = set(row[feature_index] for row in data)
        split_info = 0.0
        total = len(data)
        for value in feature_values:
            sub_data = self._split_data_discrete(data, feature_index, value)
            prob = len(sub_data) / total
            if prob > 0:
                split_info -= prob * math.log2(prob)
        return split_info

    def _gain_ratio_discrete(self, data, feature_index):
        gain = self._information_gain_discrete(data, feature_index)
        split_info = self._split_info_discrete(data, feature_index)
        if split_info == 0:
            return 0.0
        return gain / split_info

    def _information_gain_continuous(self, data, feature_index):
        base_entropy = self._entropy(data)
        values = sorted(set(row[feature_index] for row in data))
        total = len(data)
        best_gain = 0.0
        best_threshold = None
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2.0
            sub_le = self._split_data_continuous(data, feature_index, threshold, "le")
            sub_gt = self._split_data_continuous(data, feature_index, threshold, "gt")
            if not sub_le or not sub_gt:
                continue
            new_entropy = (
                len(sub_le) / total * self._entropy(sub_le)
                + len(sub_gt) / total * self._entropy(sub_gt)
            )
            gain = base_entropy - new_entropy
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        return best_gain, best_threshold

    def _split_info_continuous(self, data, feature_index, threshold):
        total = len(data)
        sub_le = self._split_data_continuous(data, feature_index, threshold, "le")
        sub_gt = self._split_data_continuous(data, feature_index, threshold, "gt")
        split_info = 0.0
        for sub in (sub_le, sub_gt):
            prob = len(sub) / total
            if prob > 0:
                split_info -= prob * math.log2(prob)
        return split_info

    def _gain_ratio_continuous(self, data, feature_index):
        gain, threshold = self._information_gain_continuous(data, feature_index)
        if threshold is None:
            return 0.0, None
        split_info = self._split_info_continuous(data, feature_index, threshold)
        if split_info == 0:
            return 0.0, threshold
        return gain / split_info, threshold

    def _choose_best_feature(self, data, feature_indices):
        best_score = -1
        best_feature = -1
        best_threshold = None
        best_type = None
        for i in feature_indices:
            ftype = self.feature_types[i]
            if ftype == "discrete":
                if self.algorithm == "id3":
                    score = self._information_gain_discrete(data, i)
                else:
                    score = self._gain_ratio_discrete(data, i)
                threshold = None
            else:
                if self.algorithm == "id3":
                    score, threshold = self._information_gain_continuous(data, i)
                else:
                    score, threshold = self._gain_ratio_continuous(data, i)
                if threshold is None:
                    continue
            if score > best_score:
                best_score = score
                best_feature = i
                best_threshold = threshold
                best_type = ftype
        return best_feature, best_type, best_threshold

    def _majority_label(self, data):
        labels = [row[-1] for row in data]
        return Counter(labels).most_common(1)[0][0]

    def _build_tree(self, data, feature_indices):
        labels = [row[-1] for row in data]
        majority = Counter(labels).most_common(1)[0][0]
        if len(set(labels)) == 1:
            return {"__label__": labels[0], "__majority__": majority}
        if not feature_indices:
            return {"__label__": majority, "__majority__": majority}
        best_feature, best_type, best_threshold = self._choose_best_feature(
            data, feature_indices
        )
        if best_feature == -1:
            return {"__label__": majority, "__majority__": majority}
        best_feature_name = self.feature_names[best_feature]
        tree = {
            "__feature__": best_feature_name,
            "__majority__": majority,
            "__type__": best_type,
        }
        if best_type == "continuous":
            tree["__threshold__"] = best_threshold
            remaining_features = feature_indices
            sub_le = self._split_data_continuous(data, best_feature, best_threshold, "le")
            sub_gt = self._split_data_continuous(data, best_feature, best_threshold, "gt")
            key_le = f"<={best_threshold}"
            key_gt = f">{best_threshold}"
            if not sub_le:
                tree[key_le] = {"__label__": majority, "__majority__": majority}
            else:
                tree[key_le] = self._build_tree(sub_le, remaining_features)
            if not sub_gt:
                tree[key_gt] = {"__label__": majority, "__majority__": majority}
            else:
                tree[key_gt] = self._build_tree(sub_gt, remaining_features)
        else:
            remaining_features = [f for f in feature_indices if f != best_feature]
            feature_values = set(row[best_feature] for row in data)
            for value in feature_values:
                sub_data = self._split_data_discrete(data, best_feature, value)
                if not sub_data:
                    tree[value] = {"__label__": majority, "__majority__": majority}
                else:
                    tree[value] = self._build_tree(sub_data, remaining_features)
        return tree

    def _infer_feature_types(self, X):
        types = []
        for col in zip(*X):
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in col):
                types.append("continuous")
            else:
                types.append("discrete")
        return types

    def fit(self, X, y, feature_names, feature_types=None):
        self.feature_names = feature_names
        if feature_types is None:
            self.feature_types = self._infer_feature_types(X)
        else:
            if len(feature_types) != len(feature_names):
                raise ValueError("feature_types length must match feature_names")
            self.feature_types = feature_types
        data = [list(x) + [label] for x, label in zip(X, y)]
        feature_indices = list(range(len(feature_names)))
        self.tree = self._build_tree(data, feature_indices)
        return self

    def _predict_single(self, sample, tree):
        if "__label__" in tree:
            return tree["__label__"]
        feature_name = tree["__feature__"]
        feature_index = self.feature_names.index(feature_name)
        feature_value = sample[feature_index]
        ftype = tree["__type__"]
        if ftype == "continuous":
            threshold = tree["__threshold__"]
            if feature_value <= threshold:
                key = f"<={threshold}"
            else:
                key = f">{threshold}"
            if key not in tree:
                logger.warning(
                    "连续特征 '%s' 的分支 '%s' 不存在，"
                    "回退到当前节点多数类别 '%s'",
                    feature_name, key, tree["__majority__"]
                )
                return tree["__majority__"]
            return self._predict_single(sample, tree[key])
        else:
            if feature_value not in tree:
                logger.warning(
                    "离散特征 '%s' 的值 '%s' 未在训练集中出现过，"
                    "回退到当前节点多数类别 '%s'",
                    feature_name, feature_value, tree["__majority__"]
                )
                return tree["__majority__"]
            return self._predict_single(sample, tree[feature_value])

    def predict(self, X):
        if self.tree is None:
            raise ValueError("Model has not been trained yet.")
        return [self._predict_single(x, self.tree) for x in X]

    def get_tree_json(self):
        return json.dumps(self.tree, ensure_ascii=False, indent=2)

    def _visualize_tree(self, tree, prefix="", is_last=True, is_root=True):
        lines = []
        if "__label__" in tree:
            connector = "── " if is_root else ("└── " if is_last else "├── ")
            lines.append(f"{prefix}{connector}类别: {tree['__label__']}")
            return lines
        feature_name = tree["__feature__"]
        ftype = tree["__type__"]
        connector = "── " if is_root else ("└── " if is_last else "├── ")
        if ftype == "continuous":
            threshold = tree["__threshold__"]
            lines.append(f"{prefix}{connector}[{feature_name}]")
            children = []
            for key in tree:
                if key.startswith("<=") or key.startswith(">"):
                    children.append(key)
            for i, child_key in enumerate(children):
                is_child_last = i == len(children) - 1
                extension = "" if is_root else ("    " if is_last else "│   ")
                child_prefix = prefix + extension + ("" if is_root else "")
                child_connector = "└── " if is_child_last else "├── "
                lines.append(f"{child_prefix}{'    ' if is_root else ''}{child_connector}{child_key}:")
                sub_prefix = child_prefix + ("    " if is_root else "") + ("    " if is_child_last else "│   ")
                lines.extend(
                    self._visualize_tree(
                        tree[child_key], sub_prefix, is_child_last, is_root=False
                    )
                )
        else:
            lines.append(f"{prefix}{connector}[{feature_name}]")
            children = [k for k in tree if not k.startswith("__")]
            for i, child_key in enumerate(children):
                is_child_last = i == len(children) - 1
                extension = "" if is_root else ("    " if is_last else "│   ")
                child_prefix = prefix + extension + ("" if is_root else "")
                child_connector = "└── " if is_child_last else "├── "
                lines.append(f"{child_prefix}{'    ' if is_root else ''}{child_connector}= {child_key}:")
                sub_prefix = child_prefix + ("    " if is_root else "") + ("    " if is_child_last else "│   ")
                lines.extend(
                    self._visualize_tree(
                        tree[child_key], sub_prefix, is_child_last, is_root=False
                    )
                )
        return lines

    def visualize_tree(self):
        if self.tree is None:
            raise ValueError("Model has not been trained yet.")
        return "\n".join(self._visualize_tree(self.tree))

    def print_tree(self):
        print(self.visualize_tree())


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("测试 1: ID3 算法 + 离散特征")
    print("=" * 60)
    X_train1 = [
        ["青年", "否", "否", "一般"],
        ["青年", "否", "否", "好"],
        ["青年", "是", "否", "好"],
        ["青年", "是", "是", "一般"],
        ["青年", "否", "否", "一般"],
        ["中年", "否", "否", "一般"],
        ["中年", "否", "否", "好"],
        ["中年", "是", "是", "好"],
        ["中年", "否", "是", "非常好"],
        ["中年", "否", "是", "非常好"],
        ["老年", "否", "是", "非常好"],
        ["老年", "否", "是", "好"],
        ["老年", "是", "否", "好"],
        ["老年", "是", "否", "非常好"],
        ["老年", "否", "否", "一般"],
    ]
    y_train1 = ["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"]
    feature_names1 = ["年龄", "有工作", "有自己的房子", "信贷情况"]

    dt_id3 = DecisionTree(algorithm="id3")
    dt_id3.fit(X_train1, y_train1, feature_names1)
    print("\n决策树结构:")
    dt_id3.print_tree()

    X_test1 = [
        ["青年", "否", "是", "一般"],
        ["中年", "是", "否", "好"],
        ["老年", "否", "否", "非常好"],
        ["青年", "否", "未知", "好"],
    ]
    predictions1 = dt_id3.predict(X_test1)
    print("\n测试样本预测结果:")
    for i, (sample, pred) in enumerate(zip(X_test1, predictions1)):
        print(f"样本 {i + 1}: {sample} -> 预测类别: {pred}")

    print("\n" + "=" * 60)
    print("测试 2: C4.5 算法 + 混合特征（连续 + 离散）")
    print("=" * 60)
    X_train2 = [
        ["青年", "否", 60, "一般"],
        ["青年", "否", 70, "好"],
        ["青年", "是", 80, "好"],
        ["青年", "是", 75, "一般"],
        ["青年", "否", 55, "一般"],
        ["中年", "否", 65, "一般"],
        ["中年", "否", 72, "好"],
        ["中年", "是", 85, "好"],
        ["中年", "否", 90, "非常好"],
        ["中年", "否", 92, "非常好"],
        ["老年", "否", 88, "非常好"],
        ["老年", "否", 78, "好"],
        ["老年", "是", 82, "好"],
        ["老年", "是", 95, "非常好"],
        ["老年", "否", 50, "一般"],
    ]
    y_train2 = ["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"]
    feature_names2 = ["年龄", "有工作", "收入", "信贷情况"]

    dt_c45 = DecisionTree(algorithm="c4.5")
    dt_c45.fit(X_train2, y_train2, feature_names2)
    print("\n决策树结构:")
    dt_c45.print_tree()

    X_test2 = [
        ["青年", "否", 88, "一般"],
        ["中年", "是", 60, "好"],
        ["老年", "否", 45, "非常好"],
        ["少年", "否", 75, "好"],
    ]
    predictions2 = dt_c45.predict(X_test2)
    print("\n测试样本预测结果:")
    for i, (sample, pred) in enumerate(zip(X_test2, predictions2)):
        print(f"样本 {i + 1}: {sample} -> 预测类别: {pred}")

    print("\n" + "=" * 60)
    print("决策树 JSON 结构（C4.5）:")
    print("=" * 60)
    print(dt_c45.get_tree_json())
