import math
from typing import Union, List, Dict, Any


Number = Union[int, float]
NumberOrList = Union[Number, List[Number]]


def _log_sum_exp(log_values: List[float]) -> float:
    if not log_values:
        return -math.inf
    max_val = max(log_values)
    if max_val == -math.inf:
        return -math.inf
    return max_val + math.log(sum(math.exp(v - max_val) for v in log_values))


def _to_list(x: NumberOrList) -> List[Number]:
    return x if isinstance(x, list) else [x]


def _format_result(k_list: List[Number], pmf_list: List[float], cdf_list: List[float]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    results = [{"k": k, "pmf": pmf, "cdf": cdf} for k, pmf, cdf in zip(k_list, pmf_list, cdf_list)]
    return results[0] if len(results) == 1 else results


def binomial_log_pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return -math.inf
    if p == 0.0:
        return 0.0 if k == 0 else -math.inf
    if p == 1.0:
        return 0.0 if k == n else -math.inf
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_comb + k * math.log(p) + (n - k) * math.log(1 - p)


def binomial_pmf(n: int, k: NumberOrList, p: float) -> Union[float, List[float]]:
    k_list = _to_list(k)
    return [math.exp(binomial_log_pmf(n, int(ki), p)) for ki in k_list] if len(k_list) > 1 else math.exp(binomial_log_pmf(n, int(k_list[0]), p))


def binomial_cdf(n: int, k: NumberOrList, p: float) -> Union[float, List[float]]:
    k_list = _to_list(k)
    results = []
    for ki in k_list:
        ki_int = int(ki)
        if ki_int < 0:
            results.append(0.0)
        elif ki_int >= n:
            results.append(1.0)
        else:
            log_probs = [binomial_log_pmf(n, i, p) for i in range(ki_int + 1)]
            results.append(math.exp(_log_sum_exp(log_probs)))
    return results[0] if len(results) == 1 else results


def binomial_stats(n: int, p: float) -> Dict[str, float]:
    return {
        "mean": n * p,
        "variance": n * p * (1 - p),
        "std": math.sqrt(n * p * (1 - p)),
        "mode": math.floor((n + 1) * p) if (n + 1) * p % 1 != 0 else [(n + 1) * p - 1, (n + 1) * p]
    }


def binomial_calculate(n: int, k: NumberOrList, p: float) -> Dict[str, Any]:
    k_list = _to_list(k)
    pmf_list = [math.exp(binomial_log_pmf(n, int(ki), p)) for ki in k_list]
    cdf_list = []
    for ki in k_list:
        ki_int = int(ki)
        if ki_int < 0:
            cdf_list.append(0.0)
        elif ki_int >= n:
            cdf_list.append(1.0)
        else:
            log_probs = [binomial_log_pmf(n, i, p) for i in range(ki_int + 1)]
            cdf_list.append(math.exp(_log_sum_exp(log_probs)))
    return {
        "distribution": "binomial",
        "parameters": {"n": n, "p": p},
        "statistics": binomial_stats(n, p),
        "results": _format_result(k_list, pmf_list, cdf_list)
    }


def poisson_log_pmf(k: int, lam: float) -> float:
    if k < 0:
        return -math.inf
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if lam == 0.0:
        return 0.0 if k == 0 else -math.inf
    return -lam + k * math.log(lam) - math.lgamma(k + 1)


def poisson_pmf(k: NumberOrList, lam: float) -> Union[float, List[float]]:
    k_list = _to_list(k)
    return [math.exp(poisson_log_pmf(int(ki), lam)) for ki in k_list] if len(k_list) > 1 else math.exp(poisson_log_pmf(int(k_list[0]), lam))


def poisson_cdf(k: NumberOrList, lam: float) -> Union[float, List[float]]:
    k_list = _to_list(k)
    results = []
    for ki in k_list:
        ki_int = int(ki)
        if ki_int < 0:
            results.append(0.0)
        else:
            log_probs = [poisson_log_pmf(i, lam) for i in range(ki_int + 1)]
            results.append(math.exp(_log_sum_exp(log_probs)))
    return results[0] if len(results) == 1 else results


def poisson_stats(lam: float) -> Dict[str, float]:
    return {
        "mean": lam,
        "variance": lam,
        "std": math.sqrt(lam),
        "mode": math.floor(lam) if lam >= 1 else 0
    }


def poisson_calculate(k: NumberOrList, lam: float) -> Dict[str, Any]:
    k_list = _to_list(k)
    pmf_list = [math.exp(poisson_log_pmf(int(ki), lam)) for ki in k_list]
    cdf_list = []
    for ki in k_list:
        ki_int = int(ki)
        if ki_int < 0:
            cdf_list.append(0.0)
        else:
            log_probs = [poisson_log_pmf(i, lam) for i in range(ki_int + 1)]
            cdf_list.append(math.exp(_log_sum_exp(log_probs)))
    return {
        "distribution": "poisson",
        "parameters": {"lambda": lam},
        "statistics": poisson_stats(lam),
        "results": _format_result(k_list, pmf_list, cdf_list)
    }


def negative_binomial_log_pmf(r: int, k: int, p: float) -> float:
    if k < 0 or r <= 0:
        return -math.inf
    if p <= 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    log_comb = math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
    return log_comb + r * math.log(p) + k * math.log(1 - p)


def negative_binomial_pmf(r: int, k: NumberOrList, p: float) -> Union[float, List[float]]:
    k_list = _to_list(k)
    return [math.exp(negative_binomial_log_pmf(r, int(ki), p)) for ki in k_list] if len(k_list) > 1 else math.exp(negative_binomial_log_pmf(r, int(k_list[0]), p))


def negative_binomial_cdf(r: int, k: NumberOrList, p: float) -> Union[float, List[float]]:
    k_list = _to_list(k)
    results = []
    for ki in k_list:
        ki_int = int(ki)
        if ki_int < 0:
            results.append(0.0)
        else:
            log_probs = [negative_binomial_log_pmf(r, i, p) for i in range(ki_int + 1)]
            results.append(math.exp(_log_sum_exp(log_probs)))
    return results[0] if len(results) == 1 else results


def negative_binomial_stats(r: int, p: float) -> Dict[str, float]:
    return {
        "mean": r * (1 - p) / p,
        "variance": r * (1 - p) / (p ** 2),
        "std": math.sqrt(r * (1 - p)) / p,
        "mode": math.floor((r - 1) * (1 - p) / p) if r > 1 else 0
    }


def negative_binomial_calculate(r: int, k: NumberOrList, p: float) -> Dict[str, Any]:
    k_list = _to_list(k)
    pmf_list = [math.exp(negative_binomial_log_pmf(r, int(ki), p)) for ki in k_list]
    cdf_list = []
    for ki in k_list:
        ki_int = int(ki)
        if ki_int < 0:
            cdf_list.append(0.0)
        else:
            log_probs = [negative_binomial_log_pmf(r, i, p) for i in range(ki_int + 1)]
            cdf_list.append(math.exp(_log_sum_exp(log_probs)))
    return {
        "distribution": "negative_binomial",
        "parameters": {"r": r, "p": p},
        "statistics": negative_binomial_stats(r, p),
        "results": _format_result(k_list, pmf_list, cdf_list)
    }


def geometric_log_pmf(k: int, p: float) -> float:
    if k < 1:
        return -math.inf
    if p <= 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    return math.log(p) + (k - 1) * math.log(1 - p)


def geometric_pmf(k: NumberOrList, p: float) -> Union[float, List[float]]:
    k_list = _to_list(k)
    return [math.exp(geometric_log_pmf(int(ki), p)) for ki in k_list] if len(k_list) > 1 else math.exp(geometric_log_pmf(int(k_list[0]), p))


def geometric_cdf(k: NumberOrList, p: float) -> Union[float, List[float]]:
    k_list = _to_list(k)
    results = []
    for ki in k_list:
        ki_int = int(ki)
        if ki_int < 1:
            results.append(0.0)
        else:
            log_probs = [geometric_log_pmf(i, p) for i in range(1, ki_int + 1)]
            results.append(math.exp(_log_sum_exp(log_probs)))
    return results[0] if len(results) == 1 else results


def geometric_stats(p: float) -> Dict[str, float]:
    return {
        "mean": 1 / p,
        "variance": (1 - p) / (p ** 2),
        "std": math.sqrt(1 - p) / p,
        "mode": 1
    }


def geometric_calculate(k: NumberOrList, p: float) -> Dict[str, Any]:
    k_list = _to_list(k)
    pmf_list = [math.exp(geometric_log_pmf(int(ki), p)) for ki in k_list]
    cdf_list = []
    for ki in k_list:
        ki_int = int(ki)
        if ki_int < 1:
            cdf_list.append(0.0)
        else:
            log_probs = [geometric_log_pmf(i, p) for i in range(1, ki_int + 1)]
            cdf_list.append(math.exp(_log_sum_exp(log_probs)))
    return {
        "distribution": "geometric",
        "parameters": {"p": p},
        "statistics": geometric_stats(p),
        "results": _format_result(k_list, pmf_list, cdf_list)
    }


def main():
    print("=" * 60)
    print("1. 二项分布 (批量计算)")
    print("=" * 60)
    result = binomial_calculate(10, [3, 5, 7], 0.5)
    print(f"参数: n={result['parameters']['n']}, p={result['parameters']['p']}")
    print(f"统计: 均值={result['statistics']['mean']:.2f}, 方差={result['statistics']['variance']:.2f}")
    for r in result['results']:
        print(f"  k={r['k']}: PMF={r['pmf']:.6f}, CDF={r['cdf']:.6f}")
    print()

    print("=" * 60)
    print("2. 泊松分布 (批量计算)")
    print("=" * 60)
    result = poisson_calculate([2, 3, 5], 3.0)
    print(f"参数: lambda={result['parameters']['lambda']}")
    print(f"统计: 均值={result['statistics']['mean']:.2f}, 方差={result['statistics']['variance']:.2f}")
    for r in result['results']:
        print(f"  k={r['k']}: PMF={r['pmf']:.6f}, CDF={r['cdf']:.6f}")
    print()

    print("=" * 60)
    print("3. 负二项分布 (批量计算)")
    print("   (第r次成功前的失败次数)")
    print("=" * 60)
    result = negative_binomial_calculate(3, [2, 4, 6], 0.4)
    print(f"参数: r={result['parameters']['r']}, p={result['parameters']['p']}")
    print(f"统计: 均值={result['statistics']['mean']:.2f}, 方差={result['statistics']['variance']:.2f}")
    for r in result['results']:
        print(f"  k={r['k']}次失败: PMF={r['pmf']:.6f}, CDF={r['cdf']:.6f}")
    print()

    print("=" * 60)
    print("4. 几何分布 (批量计算)")
    print("   (首次成功前的试验次数)")
    print("=" * 60)
    result = geometric_calculate([1, 3, 5], 0.3)
    print(f"参数: p={result['parameters']['p']}")
    print(f"统计: 均值={result['statistics']['mean']:.2f}, 方差={result['statistics']['variance']:.2f}")
    for r in result['results']:
        print(f"  第{r['k']}次试验首次成功: PMF={r['pmf']:.6f}, CDF={r['cdf']:.6f}")
    print()

    print("=" * 60)
    print("5. 大参数测试 (lgamma防溢出验证)")
    print("=" * 60)
    result = negative_binomial_calculate(100, 150, 0.4)
    print(f"负二项分布: r={result['parameters']['r']}, k={result['results']['k']}, p={result['parameters']['p']}")
    print(f"  PMF: {result['results']['pmf']:.6e}")
    print(f"  CDF: {result['results']['cdf']:.6f}")


if __name__ == "__main__":
    main()
