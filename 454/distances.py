import math
from typing import List, Tuple, Union, Literal, Dict, Any


Vector = List[float]
Vectors = List[Vector]
NaNStrategy = Literal["ignore", "mean_fill", "error"]


def _has_nan(vec: Vector) -> bool:
    return any(math.isnan(x) for x in vec)


def _handle_nan_ignore(vec1: Vector, vec2: Vector) -> Tuple[Vector, Vector, List[int]]:
    valid_pairs = []
    valid_indices = []
    for i, (a, b) in enumerate(zip(vec1, vec2)):
        if not math.isnan(a) and not math.isnan(b):
            valid_pairs.append((a, b))
            valid_indices.append(i)
    if not valid_pairs:
        return [], [], []
    v1_clean, v2_clean = zip(*valid_pairs)
    return list(v1_clean), list(v2_clean), valid_indices


def _handle_nan_mean_fill(vec1: Vector, vec2: Vector) -> Tuple[Vector, Vector]:
    v1_valid = [x for x in vec1 if not math.isnan(x)]
    v2_valid = [x for x in vec2 if not math.isnan(x)]
    mean1 = sum(v1_valid) / len(v1_valid) if v1_valid else 0.0
    mean2 = sum(v2_valid) / len(v2_valid) if v2_valid else 0.0
    v1_filled = [x if not math.isnan(x) else mean1 for x in vec1]
    v2_filled = [x if not math.isnan(x) else mean2 for x in vec2]
    return v1_filled, v2_filled


def _prepare_vectors(
    vec1: Vector,
    vec2: Vector,
    nan_strategy: NaNStrategy
) -> Tuple[Vector, Vector, List[int]]:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    valid_indices = list(range(len(vec1)))
    has_nan = _has_nan(vec1) or _has_nan(vec2)
    if has_nan:
        if nan_strategy == "error":
            raise ValueError("Vectors contain NaN values. "
                           "Use nan_strategy='ignore' or 'mean_fill' to handle NaNs.")
        elif nan_strategy == "ignore":
            vec1, vec2, valid_indices = _handle_nan_ignore(vec1, vec2)
        elif nan_strategy == "mean_fill":
            vec1, vec2 = _handle_nan_mean_fill(vec1, vec2)

    return vec1, vec2, valid_indices


def manhattan_distance(
    vec1: Vector,
    vec2: Vector,
    nan_strategy: NaNStrategy = "error"
) -> float:
    vec1, vec2, _ = _prepare_vectors(vec1, vec2, nan_strategy)
    if not vec1:
        return 0.0
    return sum(abs(a - b) for a, b in zip(vec1, vec2))


def euclidean_distance(
    vec1: Vector,
    vec2: Vector,
    nan_strategy: NaNStrategy = "error"
) -> float:
    vec1, vec2, _ = _prepare_vectors(vec1, vec2, nan_strategy)
    if not vec1:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def chebyshev_distance(
    vec1: Vector,
    vec2: Vector,
    nan_strategy: NaNStrategy = "error"
) -> float:
    vec1, vec2, _ = _prepare_vectors(vec1, vec2, nan_strategy)
    if not vec1:
        return 0.0
    return max(abs(a - b) for a, b in zip(vec1, vec2))


def minkowski_distance(
    vec1: Vector,
    vec2: Vector,
    p: float = 2.0,
    nan_strategy: NaNStrategy = "error"
) -> float:
    if p <= 0:
        raise ValueError("p must be greater than 0")
    vec1, vec2, _ = _prepare_vectors(vec1, vec2, nan_strategy)
    if not vec1:
        return 0.0
    if p == float('inf'):
        return max(abs(a - b) for a, b in zip(vec1, vec2))
    return sum(abs(a - b) ** p for a, b in zip(vec1, vec2)) ** (1.0 / p)


def dimension_analysis(
    vec1: Vector,
    vec2: Vector,
    nan_strategy: NaNStrategy = "error"
) -> Dict[str, Any]:
    vec1_orig, vec2_orig = vec1.copy(), vec2.copy()
    vec1, vec2, valid_indices = _prepare_vectors(vec1, vec2, nan_strategy)

    if not vec1:
        return {
            "abs_diffs": [],
            "sorted_dims": [],
            "max_diff_dim": None,
            "max_diff_value": 0.0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "valid_indices": []
        }

    abs_diffs = [abs(a - b) for a, b in zip(vec1, vec2)]
    indexed_diffs = [(valid_indices[i], abs_diffs[i]) for i in range(len(abs_diffs))]
    sorted_dims = sorted(indexed_diffs, key=lambda x: x[1], reverse=True)
    max_diff_idx, max_diff_val = sorted_dims[0] if sorted_dims else (None, 0.0)

    mean_diff = sum(abs_diffs) / len(abs_diffs)
    variance = sum((d - mean_diff) ** 2 for d in abs_diffs) / len(abs_diffs)
    std_diff = math.sqrt(variance)

    return {
        "abs_diffs": abs_diffs,
        "sorted_dims": sorted_dims,
        "max_diff_dim": max_diff_idx,
        "max_diff_value": max_diff_val,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "valid_indices": valid_indices
    }


def compute_distances(
    vec1: Union[Vector, Vectors],
    vec2: Union[Vector, Vectors],
    p: float = 2.0,
    nan_strategy: NaNStrategy = "error",
    return_analysis: bool = False
) -> Dict[str, Any]:
    if not vec1 or not vec2:
        raise ValueError("Input vectors cannot be empty")

    is_batch = isinstance(vec1[0], (list, tuple)) and isinstance(vec2[0], (list, tuple))

    if is_batch:
        if len(vec1) != len(vec2):
            raise ValueError("Batch sizes must be equal")
        l1_list, l2_list, linf_list, lp_list = [], [], [], []
        analysis_list = []
        for v1, v2 in zip(vec1, vec2):
            l1_list.append(manhattan_distance(v1, v2, nan_strategy))
            l2_list.append(euclidean_distance(v1, v2, nan_strategy))
            linf_list.append(chebyshev_distance(v1, v2, nan_strategy))
            lp_list.append(minkowski_distance(v1, v2, p, nan_strategy))
            if return_analysis:
                analysis_list.append(dimension_analysis(v1, v2, nan_strategy))

        result = {
            "l1": l1_list,
            "l2": l2_list,
            "linf": linf_list,
            "lp": lp_list,
            "p_value": p
        }
        if return_analysis:
            result["analysis"] = analysis_list
        return result
    else:
        result = {
            "l1": manhattan_distance(vec1, vec2, nan_strategy),
            "l2": euclidean_distance(vec1, vec2, nan_strategy),
            "linf": chebyshev_distance(vec1, vec2, nan_strategy),
            "lp": minkowski_distance(vec1, vec2, p, nan_strategy),
            "p_value": p
        }
        if return_analysis:
            result["analysis"] = dimension_analysis(vec1, vec2, nan_strategy)
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("=== 1. 多种距离计算测试 ===")
    print("=" * 60)
    v1 = [1, 2, 3, 10]
    v2 = [4, 6, 8, 7]
    print(f"向量1: {v1}")
    print(f"向量2: {v2}")

    result = compute_distances(v1, v2, p=3.0, return_analysis=True)
    print(f"\n曼哈顿距离 (L1): {result['l1']}")
    print(f"欧氏距离 (L2): {result['l2']:.6f}")
    print(f"切比雪夫距离 (L∞): {result['linf']}")
    print(f"闵可夫斯基距离 (p={result['p_value']}): {result['lp']:.6f}")

    print("\n=== 维度差值分析 ===")
    analysis = result["analysis"]
    print(f"各维度绝对差值: {analysis['abs_diffs']}")
    print(f"按贡献排序 (维度索引, 差值): {analysis['sorted_dims']}")
    print(f"最大贡献维度: 第{analysis['max_diff_dim']}维 (差值={analysis['max_diff_value']})")
    print(f"平均差值: {analysis['mean_diff']:.4f}")
    print(f"差值标准差: {analysis['std_diff']:.4f}")

    print("\n" + "=" * 60)
    print("=== 2. 闵可夫斯基距离不同p值测试 ===")
    print("=" * 60)
    for p_test in [1.0, 2.0, 3.0, 10.0, float('inf')]:
        lp = minkowski_distance(v1, v2, p=p_test)
        p_str = "∞" if p_test == float('inf') else str(p_test)
        print(f"p={p_str:4}: {lp:.6f}")

    print("\n" + "=" * 60)
    print("=== 3. 含NaN的距离计算 + 分析 ===")
    print("=" * 60)
    v1_nan = [1, float('nan'), 3, 10, float('nan')]
    v2_nan = [4, 6, float('nan'), 7, 15]
    print(f"向量1: {v1_nan}")
    print(f"向量2: {v2_nan}")

    result_nan = compute_distances(v1_nan, v2_nan, nan_strategy="ignore", return_analysis=True)
    print(f"\n曼哈顿距离 (L1): {result_nan['l1']}")
    print(f"欧氏距离 (L2): {result_nan['l2']:.6f}")
    print(f"切比雪夫距离 (L∞): {result_nan['linf']}")

    analysis_nan = result_nan["analysis"]
    print(f"\n有效维度索引: {analysis_nan['valid_indices']}")
    print(f"各有效维度绝对差值: {analysis_nan['abs_diffs']}")
    print(f"按贡献排序: {analysis_nan['sorted_dims']}")

    print("\n" + "=" * 60)
    print("=== 4. 批量计算（含分析） ===")
    print("=" * 60)
    batch1 = [[1, 2], [3, 4], [5, 12]]
    batch2 = [[4, 6], [7, 8], [9, 10]]
    print(f"批量1: {batch1}")
    print(f"批量2: {batch2}")

    batch_result = compute_distances(batch1, batch2, return_analysis=True)
    for i in range(len(batch1)):
        print(f"\n  点对 {i+1}: {batch1[i]} <-> {batch2[i]}")
        print(f"    L1: {batch_result['l1'][i]}, L2: {batch_result['l2'][i]:.4f}, "
              f"L∞: {batch_result['linf'][i]}")
        print(f"    最大贡献维度: 第{batch_result['analysis'][i]['max_diff_dim']}维")
