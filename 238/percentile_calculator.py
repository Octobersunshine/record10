import random
import math


def _validate_inputs(data, percentiles, method):
    if not data:
        raise ValueError("数据列表不能为空")
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("数据列表必须包含数值")
    if not percentiles:
        raise ValueError("百分位列表不能为空")
    if not all(isinstance(p, (int, float)) and 0 <= p <= 100 for p in percentiles):
        raise ValueError("百分位值必须在0到100之间")
    if method not in ["type5", "type7"]:
        raise ValueError("method参数必须为'type5'或'type7'")


def _validate_weighted_inputs(data, weights, percentiles):
    if not data:
        raise ValueError("数据列表不能为空")
    if not weights:
        raise ValueError("权重列表不能为空")
    if len(data) != len(weights):
        raise ValueError("数据列表和权重列表长度必须相同")
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("数据列表必须包含数值")
    if not all(isinstance(w, (int, float)) and w >= 0 for w in weights):
        raise ValueError("权重必须为非负数")
    if sum(weights) <= 0:
        raise ValueError("权重之和必须大于0")
    if not percentiles:
        raise ValueError("百分位列表不能为空")
    if not all(isinstance(p, (int, float)) and 0 <= p <= 100 for p in percentiles):
        raise ValueError("百分位值必须在0到100之间")


def _calculate_rank(n, percentile, method):
    p = percentile / 100
    if method == "type7":
        return (n - 1) * p
    elif method == "type5":
        return (n - 1) * p + 0.5


def _linear_interpolation(sorted_data, percentile, method):
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]

    if percentile <= 0:
        return sorted_data[0]
    if percentile >= 100:
        return sorted_data[-1]

    rank = _calculate_rank(n, percentile, method)
    
    if rank <= 0:
        return sorted_data[0]
    if rank >= n - 1:
        return sorted_data[-1]

    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = rank - lower_idx

    result = sorted_data[lower_idx] + fraction * (sorted_data[upper_idx] - sorted_data[lower_idx])
    return round(result, 10)


def _weighted_linear_interpolation(sorted_data, sorted_weights, percentile):
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]

    if percentile <= 0:
        return sorted_data[0]
    if percentile >= 100:
        return sorted_data[-1]

    p = percentile / 100
    total_weight = sum(sorted_weights)

    cumulative_weights = []
    cum_sum = 0
    for w in sorted_weights:
        cum_sum += w
        cumulative_weights.append(cum_sum)

    adjusted_cumulative = [0] * n
    adjusted_cumulative[0] = sorted_weights[0] / 2
    for i in range(1, n):
        adjusted_cumulative[i] = adjusted_cumulative[i - 1] + \
            (sorted_weights[i - 1] + sorted_weights[i]) / 2

    total_adjusted = adjusted_cumulative[-1] + sorted_weights[-1] / 2
    target = p * (total_adjusted - sorted_weights[0] / 2 - sorted_weights[-1] / 2) + \
        sorted_weights[0] / 2

    for i in range(n):
        if adjusted_cumulative[i] >= target:
            if i == 0:
                return sorted_data[0]
            prev_adj = adjusted_cumulative[i - 1]
            curr_adj = adjusted_cumulative[i]
            if curr_adj == prev_adj:
                return sorted_data[i]
            fraction = (target - prev_adj) / (curr_adj - prev_adj)
            result = sorted_data[i - 1] + fraction * (sorted_data[i] - sorted_data[i - 1])
            return round(result, 10)

    return sorted_data[-1]


def calculate_percentiles(data, percentiles, method="type7"):
    _validate_inputs(data, percentiles, method)
    sorted_data = sorted(data)
    results = {}
    for p in percentiles:
        results[p] = _linear_interpolation(sorted_data, p, method)
    return results


def calculate_weighted_percentiles(data, weights, percentiles):
    _validate_weighted_inputs(data, weights, percentiles)
    paired = sorted(zip(data, weights), key=lambda x: x[0])
    sorted_data = [x[0] for x in paired]
    sorted_weights = [x[1] for x in paired]

    n = len(sorted_data)
    all_equal = all(w == sorted_weights[0] for w in sorted_weights)

    if all_equal:
        results = calculate_percentiles(data, percentiles, method="type7")
        return results

    results = {}
    for p in percentiles:
        results[p] = _weighted_linear_interpolation(sorted_data, sorted_weights, p)
    return results


def calculate_weighted_quartiles(data, weights):
    quartile_percentiles = [25, 50, 75]
    results = calculate_weighted_percentiles(data, weights, quartile_percentiles)
    return {
        "Q1": results[25],
        "Q2/中位数": results[50],
        "Q3": results[75]
    }


def calculate_quartiles(data, method="type7"):
    quartile_percentiles = [25, 50, 75]
    results = calculate_percentiles(data, quartile_percentiles, method)
    return {
        "Q1": results[25],
        "Q2/中位数": results[50],
        "Q3": results[75]
    }


def bootstrap_percentile_ci(data, percentile, n_bootstrap=1000, confidence_level=0.95,
                            method="type7", random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    n = len(data)
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        sample = random.choices(data, k=n)
        est = calculate_percentiles(sample, [percentile], method)[percentile]
        bootstrap_estimates.append(est)

    bootstrap_estimates.sort()
    alpha = 1 - confidence_level
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)
    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    point_estimate = calculate_percentiles(data, [percentile], method)[percentile]

    return {
        "point_estimate": point_estimate,
        "ci_lower": round(bootstrap_estimates[lower_idx], 10),
        "ci_upper": round(bootstrap_estimates[upper_idx], 10),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "bootstrap_std": round(
            sum((x - sum(bootstrap_estimates) / n_bootstrap) ** 2 for x in bootstrap_estimates) / n_bootstrap,
            10
        ) ** 0.5
    }


def normal_approx_percentile_ci(data, percentile, confidence_level=0.95, method="type7"):
    n = len(data)
    if n < 2:
        raise ValueError("样本量至少为2才能使用正态近似")

    sorted_data = sorted(data)
    point_estimate = calculate_percentiles(data, [percentile], method)[percentile]

    p = percentile / 100

    rank = _calculate_rank(n, percentile, method)
    lower_idx = max(0, int(math.floor(rank)))
    upper_idx = min(n - 1, lower_idx + 1)

    if lower_idx == upper_idx:
        density = 1.0
    else:
        density = 1.0 / (sorted_data[upper_idx] - sorted_data[lower_idx])

    variance = (p * (1 - p)) / (n * density ** 2)
    std_error = math.sqrt(variance)

    z = 1.96 if confidence_level == 0.95 else \
        {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)

    margin = z * std_error

    return {
        "point_estimate": point_estimate,
        "ci_lower": round(max(sorted_data[0], point_estimate - margin), 10),
        "ci_upper": round(min(sorted_data[-1], point_estimate + margin), 10),
        "confidence_level": confidence_level,
        "std_error": round(std_error, 10),
        "method": "normal_approximation"
    }


def calculate_percentile_ci(data, percentile, ci_method="bootstrap", **kwargs):
    if ci_method == "bootstrap":
        return bootstrap_percentile_ci(data, percentile, **kwargs)
    elif ci_method == "normal":
        return normal_approx_percentile_ci(data, percentile, **kwargs)
    else:
        raise ValueError("ci_method必须为'bootstrap'或'normal'")


if __name__ == "__main__":
    data = [1, 2, 5, 6, 7, 9, 12, 15, 18, 20]
    weights = [1, 1, 2, 1, 3, 2, 1, 2, 1, 1]
    target_percentiles = [25, 50, 75, 90, 95]

    print("=" * 60)
    print("分位数计算器")
    print("=" * 60)
    print("数据列表:", data)
    print()

    print("-" * 60)
    print("方法: Type 7 (默认, 与numpy/R一致)")
    print("-" * 60)
    print("\n四分位数计算结果:")
    quartiles = calculate_quartiles(data, method="type7")
    for key, value in quartiles.items():
        print(f"  {key}: {value}")

    print(f"\n指定百分位数计算结果:")
    percentile_results = calculate_percentiles(data, target_percentiles, method="type7")
    for p, value in percentile_results.items():
        print(f"  {p}% 分位数: {value}")

    print()
    print("-" * 60)
    print("方法: Type 5")
    print("-" * 60)
    print("\n四分位数计算结果:")
    quartiles = calculate_quartiles(data, method="type5")
    for key, value in quartiles.items():
        print(f"  {key}: {value}")

    print(f"\n指定百分位数计算结果:")
    percentile_results = calculate_percentiles(data, target_percentiles, method="type5")
    for p, value in percentile_results.items():
        print(f"  {p}% 分位数: {value}")

    print()
    print("=" * 60)
    print("加权分位数计算")
    print("=" * 60)
    print("数据列表:", data)
    print("权重列表:", weights)
    print("\n加权四分位数计算结果:")
    weighted_quartiles = calculate_weighted_quartiles(data, weights)
    for key, value in weighted_quartiles.items():
        print(f"  {key}: {value}")

    print(f"\n加权指定百分位数计算结果:")
    weighted_results = calculate_weighted_percentiles(data, weights, target_percentiles)
    for p, value in weighted_results.items():
        print(f"  {p}% 分位数: {value}")

    print()
    print("=" * 60)
    print("分位数区间估计 (Bootstrap方法)")
    print("=" * 60)
    target_percentile = 90
    ci_result = calculate_percentile_ci(
        data, target_percentile, ci_method="bootstrap",
        n_bootstrap=1000, confidence_level=0.95, random_seed=42
    )
    print(f"{target_percentile}% 分位数点估计: {ci_result['point_estimate']}")
    print(f"{int(ci_result['confidence_level'] * 100)}% 置信区间: "
          f"[{ci_result['ci_lower']}, {ci_result['ci_upper']}]")
    print(f"Bootstrap 标准差: {round(ci_result['bootstrap_std'], 4)}")
    print(f"Bootstrap 次数: {ci_result['n_bootstrap']}")

    print()
    print("=" * 60)
    print("分位数区间估计 (正态近似方法)")
    print("=" * 60)
    ci_result_normal = calculate_percentile_ci(
        data, target_percentile, ci_method="normal", confidence_level=0.95
    )
    print(f"{target_percentile}% 分位数点估计: {ci_result_normal['point_estimate']}")
    print(f"{int(ci_result_normal['confidence_level'] * 100)}% 置信区间: "
          f"[{ci_result_normal['ci_lower']}, {ci_result_normal['ci_upper']}]")
    print(f"标准误: {ci_result_normal['std_error']}")

    print()
    print("=" * 60)
    print("小样本测试 (n=3, data=[1, 2, 3])")
    print("=" * 60)
    small_data = [1, 2, 3]
    print(f"Type 7 - 100%分位数: {calculate_percentiles(small_data, [100], method='type7')[100]}")
    print(f"Type 7 - 0%分位数: {calculate_percentiles(small_data, [0], method='type7')[0]}")
    print(f"Type 5 - 100%分位数: {calculate_percentiles(small_data, [100], method='type5')[100]}")
    print(f"Type 5 - 0%分位数: {calculate_percentiles(small_data, [0], method='type5')[0]}")
    print(f"最大值: {max(small_data)}, 最小值: {min(small_data)}")
