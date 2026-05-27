import math
import warnings
from typing import List, Tuple, Dict, Union

Number = Union[int, float]


def zscore_normalize(data: List[Number]) -> Tuple[List[float], Dict[str, float]]:
    n = len(data)
    if n == 0:
        return [], {"mean": 0.0, "std": 0.0, "warning": ""}
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(variance)
    if std < 1e-10:
        warning_msg = f"标准差接近零({std:.2e})，数据无有效变异，返回全零数组"
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
        return [0.0] * n, {"mean": mean, "std": std, "warning": warning_msg}
    normalized = [(x - mean) / std for x in data]
    return normalized, {"mean": mean, "std": std, "warning": ""}


def minmax_normalize(data: List[Number]) -> Tuple[List[float], Dict[str, float]]:
    n = len(data)
    if n == 0:
        return [], {"min": 0.0, "max": 0.0}
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val
    if range_val == 0:
        return [0.0] * n, {"min": min_val, "max": max_val}
    normalized = [(x - min_val) / range_val for x in data]
    return normalized, {"min": min_val, "max": max_val}


def maxabs_normalize(data: List[Number]) -> Tuple[List[float], Dict[str, float]]:
    n = len(data)
    if n == 0:
        return [], {"max_abs": 0.0}
    max_abs = max(abs(x) for x in data)
    if max_abs == 0:
        return [0.0] * n, {"max_abs": max_abs}
    normalized = [x / max_abs for x in data]
    return normalized, {"max_abs": max_abs}


def _median(sorted_data: List[Number]) -> float:
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_data[mid])
    return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0


def _quartiles(data: List[Number]) -> Tuple[float, float, float]:
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    q2 = _median(sorted_data)
    if n % 2 == 1:
        q1 = _median(sorted_data[:mid])
        q3 = _median(sorted_data[mid + 1:])
    else:
        q1 = _median(sorted_data[:mid])
        q3 = _median(sorted_data[mid:])
    return q1, q2, q3


def robust_normalize(data: List[Number]) -> Tuple[List[float], Dict[str, float]]:
    n = len(data)
    if n == 0:
        return [], {"median": 0.0, "iqr": 0.0, "q1": 0.0, "q3": 0.0}
    q1, median, q3 = _quartiles(data)
    iqr = q3 - q1
    if iqr < 1e-10:
        warning_msg = f"IQR接近零({iqr:.2e})，数据无有效变异，返回全零数组"
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
        return [0.0] * n, {"median": median, "iqr": iqr, "q1": q1, "q3": q3, "warning": warning_msg}
    normalized = [(x - median) / iqr for x in data]
    return normalized, {"median": median, "iqr": iqr, "q1": q1, "q3": q3, "warning": ""}


def robust_inverse_transform(normalized_data: List[float], params: Dict[str, float]) -> List[float]:
    median = params["median"]
    iqr = params["iqr"]
    return [x * iqr + median for x in normalized_data]


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]

    z_result, z_params = zscore_normalize(data)
    print("Z-score 标准化:")
    print(f"  结果: {z_result}")
    print(f"  参数: {z_params}")

    mm_result, mm_params = minmax_normalize(data)
    print("\nMin-Max 归一化:")
    print(f"  结果: {mm_result}")
    print(f"  参数: {mm_params}")

    ma_result, ma_params = maxabs_normalize(data)
    print("\nMaxAbs 归一化:")
    print(f"  结果: {ma_result}")
    print(f"  参数: {ma_params}")

    data_with_neg = [-5, -2, 0, 3, 10]
    print("\n--- 含负值数据 ---")
    print(f"输入: {data_with_neg}")

    z_result2, z_params2 = zscore_normalize(data_with_neg)
    print(f"\nZ-score: {z_result2}")
    print(f"  参数: {z_params2}")

    mm_result2, mm_params2 = minmax_normalize(data_with_neg)
    print(f"\nMin-Max: {mm_result2}")
    print(f"  参数: {mm_params2}")

    ma_result2, ma_params2 = maxabs_normalize(data_with_neg)
    print(f"\nMaxAbs: {ma_result2}")
    print(f"  参数: {ma_params2}")

    constant_data = [7, 7, 7, 7, 7]
    print("\n--- 常量数据（方差为零）---")
    print(f"输入: {constant_data}")
    z_result3, z_params3 = zscore_normalize(constant_data)
    print(f"\nZ-score: {z_result3}")
    print(f"  参数: {z_params3}")

    print("\n--- 鲁棒标准化（含异常值对比）---")
    data_normal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data_with_outlier = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]

    print(f"正常数据: {data_normal}")
    r_result1, r_params1 = robust_normalize(data_normal)
    print(f"  鲁棒标准化: {r_result1}")
    print(f"  参数: median={r_params1['median']:.2f}, IQR={r_params1['iqr']:.2f}, Q1={r_params1['q1']:.2f}, Q3={r_params1['q3']:.2f}")

    z_result1, _ = zscore_normalize(data_normal)
    print(f"  Z-score 标准化: {z_result1}")

    print(f"\n含异常值数据: {data_with_outlier}")
    r_result2, r_params2 = robust_normalize(data_with_outlier)
    print(f"  鲁棒标准化: {r_result2}")
    print(f"  参数: median={r_params2['median']:.2f}, IQR={r_params2['iqr']:.2f}, Q1={r_params2['q1']:.2f}, Q3={r_params2['q3']:.2f}")

    z_result2, _ = zscore_normalize(data_with_outlier)
    print(f"  Z-score 标准化: {z_result2}")

    print("\n--- 鲁棒标准化逆变换验证 ---")
    original_data = [10, 20, 30, 40, 50]
    print(f"原始数据: {original_data}")
    normalized, params = robust_normalize(original_data)
    print(f"标准化后: {normalized}")
    restored = robust_inverse_transform(normalized, params)
    print(f"逆变换还原: {restored}")
    print(f"还原误差: {[abs(restored[i] - original_data[i]) for i in range(len(original_data))]}")
