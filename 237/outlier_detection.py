import math
from typing import List, Dict, Any, Union


def _percentile(sorted_data: List[float], p: float) -> float:
    n = len(sorted_data)
    pos = (n - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_data[lo]
    return sorted_data[lo] * (hi - pos) + sorted_data[hi] * (pos - lo)


def _median(sorted_data: List[float]) -> float:
    return _percentile(sorted_data, 0.5)


def _log_gamma(x: float) -> float:
    cof = [
        76.18009172947146, -86.50532032941677,
        24.01409824083091, -1.231739572450155,
        0.1208650973866179e-2, -0.5395239384953e-5,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1.0
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    lbeta = _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
    front = math.exp(
        math.log(x) * a + math.log(1.0 - x) * b - lbeta
    )

    use_continued_fraction = x < (a + 1.0) / (a + b + 2.0)

    if use_continued_fraction:
        f = _beta_cf(x, a, b)
        return front * f / a
    else:
        f = _beta_cf(1.0 - x, b, a)
        return 1.0 - front * f / b


def _beta_cf(x: float, a: float, b: float) -> float:
    max_iter = 200
    eps = 3.0e-12
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return h


def _t_cdf(t_val: float, df: int) -> float:
    x = df / (df + t_val * t_val)
    ib = _regularized_incomplete_beta(x, df / 2.0, 0.5)
    if t_val >= 0:
        return 1.0 - 0.5 * ib
    else:
        return 0.5 * ib


def _t_ppf(p: float, df: int) -> float:
    if p <= 0.0:
        return -1e308
    if p >= 1.0:
        return 1e308

    if abs(p - 0.5) < 1e-16:
        return 0.0

    if p > 0.5:
        return -_t_ppf(1.0 - p, df)

    lo, hi = -100.0, 0.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2.0


def zscore_outliers(
    data: List[float],
    threshold: float = 3.0,
) -> Dict[str, Any]:
    n = len(data)
    if n == 0:
        return {"method": "z-score", "threshold": threshold, "mean": None, "std": None, "outliers": []}

    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = math.sqrt(variance)

    if std == 0:
        return {"method": "z-score", "threshold": threshold, "mean": mean, "std": 0.0, "outliers": []}

    outliers = []
    for i, x in enumerate(data):
        z = (x - mean) / std
        if abs(z) > threshold:
            outliers.append({"index": i, "value": x, "z_score": z})

    return {
        "method": "z-score",
        "threshold": threshold,
        "mean": mean,
        "std": std,
        "outliers": outliers,
    }


def iqr_outliers(
    data: List[float],
    k: float = 1.5,
) -> Dict[str, Any]:
    n = len(data)
    if n == 0:
        return {"method": "IQR", "k": k, "q1": None, "q3": None, "iqr": None, "lower_bound": None, "upper_bound": None, "outliers": []}

    sorted_data = sorted(data)
    q1 = _percentile(sorted_data, 0.25)
    q3 = _percentile(sorted_data, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outliers = []
    for i, x in enumerate(data):
        if x < lower_bound or x > upper_bound:
            outliers.append({"index": i, "value": x})

    return {
        "method": "IQR",
        "k": k,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers": outliers,
    }


def iqr_adjusted_outliers(
    data: List[float],
    k: float = 1.5,
) -> Dict[str, Any]:
    n = len(data)
    if n == 0:
        return {
            "method": "IQR-Adjusted", "k": k, "skewness": None,
            "q1": None, "q3": None, "iqr": None,
            "lower_bound": None, "upper_bound": None, "outliers": [],
        }

    sorted_data = sorted(data)
    q1 = _percentile(sorted_data, 0.25)
    q2 = _percentile(sorted_data, 0.50)
    q3 = _percentile(sorted_data, 0.75)
    iqr = q3 - q1

    if iqr == 0:
        return {
            "method": "IQR-Adjusted", "k": k, "skewness": 0.0,
            "q1": q1, "q3": q3, "iqr": 0.0,
            "lower_bound": q1, "upper_bound": q3, "outliers": [],
        }

    bowley_skewness = (q3 - 2 * q2 + q1) / iqr

    if bowley_skewness >= 0:
        lower_bound = q1 - k * math.exp(-4 * bowley_skewness) * iqr
        upper_bound = q3 + k * math.exp(3 * bowley_skewness) * iqr
    else:
        lower_bound = q1 - k * math.exp(-3 * bowley_skewness) * iqr
        upper_bound = q3 + k * math.exp(4 * bowley_skewness) * iqr

    outliers = []
    for i, x in enumerate(data):
        if x < lower_bound or x > upper_bound:
            outliers.append({"index": i, "value": x})

    return {
        "method": "IQR-Adjusted",
        "k": k,
        "skewness": bowley_skewness,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers": outliers,
    }


def mad_outliers(
    data: List[float],
    threshold: float = 3.5,
) -> Dict[str, Any]:
    n = len(data)
    if n == 0:
        return {"method": "MAD", "threshold": threshold, "median": None, "mad": None, "outliers": []}

    sorted_data = sorted(data)
    median = _median(sorted_data)

    abs_devs = sorted(abs(x - median) for x in data)
    mad = _median(abs_devs)

    if mad == 0:
        abs_devs_nonzero = [abs(x - median) for x in data if x != median]
        if abs_devs_nonzero:
            mad = min(abs_devs_nonzero)
        else:
            return {
                "method": "MAD", "threshold": threshold,
                "median": median, "mad": 0.0, "outliers": [],
            }

    outliers = []
    for i, x in enumerate(data):
        modified_z = 0.6745 * (x - median) / mad
        if abs(modified_z) > threshold:
            outliers.append({"index": i, "value": x, "modified_z": modified_z})

    return {
        "method": "MAD",
        "threshold": threshold,
        "median": median,
        "mad": mad,
        "outliers": outliers,
    }


def grubbs_outliers(
    data: List[float],
    alpha: float = 0.05,
    max_iter: int = 50,
) -> Dict[str, Any]:
    n = len(data)
    if n < 3:
        return {
            "method": "Grubbs", "alpha": alpha, "iterations": 0,
            "note": "Grubbs test requires at least 3 data points", "outliers": [],
        }

    remaining = list(data)
    original_indices = list(range(n))
    all_outliers = []
    iteration = 0

    for iteration in range(1, max_iter + 1):
        m = len(remaining)
        if m < 3:
            break

        mean = sum(remaining) / m
        variance = sum((x - mean) ** 2 for x in remaining) / m
        std = math.sqrt(variance)

        if std == 0:
            break

        deviations = [abs(x - mean) for x in remaining]
        max_dev_idx = deviations.index(max(deviations))
        g_stat = deviations[max_dev_idx] / std

        df = m - 2
        t_crit = _t_ppf(alpha / (2 * m), df)
        g_crit = ((m - 1) / math.sqrt(m)) * math.sqrt(
            t_crit ** 2 / (df + t_crit ** 2)
        )

        if g_stat > g_crit:
            outlier_original_idx = original_indices[max_dev_idx]
            outlier_value = remaining[max_dev_idx]
            all_outliers.append({
                "index": outlier_original_idx,
                "value": outlier_value,
                "g_stat": g_stat,
                "g_crit": g_crit,
                "iteration": iteration,
            })
            remaining.pop(max_dev_idx)
            original_indices.pop(max_dev_idx)
        else:
            break

    return {
        "method": "Grubbs",
        "alpha": alpha,
        "iterations": iteration if all_outliers else 0,
        "outliers": all_outliers,
    }


Point = Union[float, List[float]]


def _euclidean_distance(a: Point, b: Point) -> float:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b)
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def dbscan_outliers(
    data: List[Point],
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict[str, Any]:
    n = len(data)
    if n == 0:
        return {
            "method": "DBSCAN", "eps": eps, "min_samples": min_samples,
            "n_clusters": 0, "outliers": [],
        }

    neighbors: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean_distance(data[i], data[j])
            if d <= eps:
                neighbors[i].append(j)
                neighbors[j].append(i)

    labels = [-1] * n
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        if len(neighbors[i]) < min_samples:
            continue

        labels[i] = cluster_id
        queue = list(neighbors[i])

        while queue:
            q = queue.pop(0)
            if labels[q] != -1:
                continue
            labels[q] = cluster_id

            if len(neighbors[q]) >= min_samples:
                for nb in neighbors[q]:
                    if labels[nb] == -1:
                        queue.append(nb)

        cluster_id += 1

    outliers = []
    for i in range(n):
        if labels[i] == -1:
            outliers.append({"index": i, "value": data[i]})

    return {
        "method": "DBSCAN",
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": cluster_id,
        "outliers": outliers,
    }


def detect_outliers(
    data: List[float],
    zscore_threshold: float = 3.0,
    iqr_k: float = 1.5,
    mad_threshold: float = 3.5,
    grubbs_alpha: float = 0.05,
) -> Dict[str, Any]:
    return {
        "zscore": zscore_outliers(data, threshold=zscore_threshold),
        "iqr": iqr_outliers(data, k=iqr_k),
        "iqr_adjusted": iqr_adjusted_outliers(data, k=iqr_k),
        "mad": mad_outliers(data, threshold=mad_threshold),
        "grubbs": grubbs_outliers(data, alpha=grubbs_alpha),
    }


def _print_result(label: str, info: Dict[str, Any], data: list) -> None:
    print(f"\n【{label}】")
    method = info["method"]
    if method == "z-score":
        print(f"  阈值 (threshold): {info['threshold']}")
        print(f"  均值 (mean): {info['mean']:.4f}")
        print(f"  标准差 (std):  {info['std']:.4f}")
    elif method == "IQR":
        print(f"  系数 (k):       {info['k']}")
        print(f"  Q1:             {info['q1']:.4f}")
        print(f"  Q3:             {info['q3']:.4f}")
        print(f"  IQR:            {info['iqr']:.4f}")
        print(f"  下界:           {info['lower_bound']:.4f}")
        print(f"  上界:           {info['upper_bound']:.4f}")
    elif method == "IQR-Adjusted":
        print(f"  系数 (k):       {info['k']}")
        print(f"  Bowley偏度:     {info['skewness']:.4f}")
        print(f"  Q1:             {info['q1']:.4f}")
        print(f"  Q3:             {info['q3']:.4f}")
        print(f"  IQR:            {info['iqr']:.4f}")
        print(f"  校正下界:       {info['lower_bound']:.4f}")
        print(f"  校正上界:       {info['upper_bound']:.4f}")
    elif method == "MAD":
        print(f"  阈值 (threshold): {info['threshold']}")
        print(f"  中位数 (median):  {info['median']:.4f}")
        print(f"  MAD:              {info['mad']:.4f}")
    elif method == "Grubbs":
        print(f"  显著性水平 (alpha): {info['alpha']}")
        print(f"  迭代次数:          {info.get('iterations', 0)}")
        if "note" in info:
            print(f"  注意: {info['note']}")
    elif method == "DBSCAN":
        print(f"  eps:            {info['eps']}")
        print(f"  min_samples:    {info['min_samples']}")
        print(f"  聚类数:         {info['n_clusters']}")

    print(f"  异常值数量: {len(info['outliers'])}")
    for o in info["outliers"]:
        parts = [f"索引={o['index']}", f"数值={o['value']}"]
        if "z_score" in o:
            parts.append(f"Z分数={o['z_score']:.4f}")
        if "modified_z" in o:
            parts.append(f"修正Z={o['modified_z']:.4f}")
        if "g_stat" in o:
            parts.append(f"G统计量={o['g_stat']:.4f}")
            parts.append(f"G临界值={o['g_crit']:.4f}")
            parts.append(f"迭代={o['iteration']}")
        print(f"    {', '.join(parts)}")


def _generate_lognormal(n: int, seed: int = 42) -> List[float]:
    import random
    rng = random.Random(seed)
    mu, sigma = 0.0, 0.8
    return [math.exp(rng.gauss(mu, sigma)) for _ in range(n)]


def _generate_clusters_2d(n_per_cluster: int = 50, seed: int = 42) -> List[List[float]]:
    import random
    rng = random.Random(seed)
    centers = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    points: List[List[float]] = []
    for cx, cy in centers:
        for _ in range(n_per_cluster):
            x = cx + rng.gauss(0, 0.5)
            y = cy + rng.gauss(0, 0.5)
            points.append([x, y])
    points.append([25.0, 25.0])
    points.append([-10.0, 15.0])
    points.append([50.0, -5.0])
    return points


if __name__ == "__main__":
    print("=" * 70)
    print("测试1: 一维数据 — 普通数据含明显异常值")
    print("=" * 70)
    sample = [10, 12, 13, 11, 14, 12, 15, 10, 13, 12, 100, 11, 13, 14, 12, -50, 13, 11, 12, 14]
    print(f"输入数据: {sample}")

    result = detect_outliers(sample)
    _print_result("Z-score", result["zscore"], sample)
    _print_result("IQR（经典）", result["iqr"], sample)
    _print_result("IQR（偏度校正）", result["iqr_adjusted"], sample)
    _print_result("MAD", result["mad"], sample)
    _print_result("Grubbs", result["grubbs"], sample)

    print("\n\n" + "=" * 70)
    print("测试2: 一维数据 — 对数正态分布（右偏）验证偏态校正效果")
    print("=" * 70)
    lognorm_data = _generate_lognormal(200)
    lognorm_data.append(25.0)
    lognorm_data.append(30.0)

    result2 = detect_outliers(lognorm_data, mad_threshold=3.5)

    print(f"数据量: {len(lognorm_data)}，注入极端值: 25.0, 30.0")
    _print_result("IQR（经典）", result2["iqr"], lognorm_data)
    _print_result("IQR（偏度校正）", result2["iqr_adjusted"], lognorm_data)
    _print_result("MAD", result2["mad"], lognorm_data)
    _print_result("Grubbs", result2["grubbs"], lognorm_data)

    iqr_count = len(result2["iqr"]["outliers"])
    adj_count = len(result2["iqr_adjusted"]["outliers"])
    mad_count = len(result2["mad"]["outliers"])
    grubbs_count = len(result2["grubbs"]["outliers"])
    print(f"\n  >>> 经典IQR 标记了 {iqr_count} 个异常值")
    print(f"  >>> 偏度校正IQR 标记了 {adj_count} 个异常值")
    print(f"  >>> MAD 标记了 {mad_count} 个异常值")
    print(f"  >>> Grubbs 标记了 {grubbs_count} 个异常值")

    print("\n\n" + "=" * 70)
    print("测试3: 多维数据 — DBSCAN密度聚类异常检测（2D点集）")
    print("=" * 70)
    points_2d = _generate_clusters_2d(50, seed=42)
    print(f"数据量: {len(points_2d)} (3个聚类各50点 + 3个远离点)")

    db_result = dbscan_outliers(points_2d, eps=1.5, min_samples=5)
    _print_result("DBSCAN", db_result, points_2d)

    print("\n\n" + "=" * 70)
    print("测试4: 一维数据 — DBSCAN异常检测")
    print("=" * 70)
    sample_1d = [1.0, 1.1, 1.2, 0.9, 1.0, 1.1, 1.0, 0.9, 1.2, 1.1,
                 5.0, 5.1, 5.2, 4.9, 5.0, 5.1, 4.8, 5.2, 5.0, 5.1,
                 100.0, -20.0]
    print(f"输入数据: {sample_1d}")

    db_1d = dbscan_outliers(sample_1d, eps=0.5, min_samples=3)
    _print_result("DBSCAN (1D)", db_1d, sample_1d)
