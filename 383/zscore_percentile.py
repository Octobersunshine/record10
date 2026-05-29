import math
import warnings


def _median(sorted_data):
    n = len(sorted_data)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return sorted_data[mid]
    return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0


def calculate_z_score(data, query_point):
    n = len(data)
    if n < 2:
        return None
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std_dev = math.sqrt(variance)
    if std_dev == 0:
        warnings.warn("数据集标准差为0，所有值相同，Z-score定义为0")
        return 0.0
    return (query_point - mean) / std_dev


def calculate_modified_z_score(data, query_point):
    n = len(data)
    if n < 2:
        return None
    sorted_data = sorted(data)
    median_val = _median(sorted_data)
    deviations = sorted(abs(x - median_val) for x in sorted_data)
    mad = _median(deviations)
    if mad == 0:
        warnings.warn("MAD为0，修正Z-score定义为0")
        return 0.0
    return 0.6745 * (query_point - median_val) / mad


def calculate_percentile_rank(data, query_point):
    n = len(data)
    if n == 0:
        return None
    less_count = sum(1 for x in data if x < query_point)
    equal_count = sum(1 for x in data if x == query_point)
    if equal_count == n:
        warnings.warn("数据集所有值相同，百分位排名定义为50%")
        return 50.0
    return (less_count + 0.5 * equal_count) / n * 100


def calculate_cdf_data(data):
    if not data:
        return []
    sorted_data = sorted(data)
    n = len(sorted_data)
    cdf = []
    for i, val in enumerate(sorted_data):
        cdf.append((val, (i + 1) / n * 100))
    return cdf


def calculate_metrics(data, query_points):
    if not isinstance(query_points, (list, tuple)):
        query_points = [query_points]

    sorted_data = sorted(data) if data else []
    cdf_data = calculate_cdf_data(data)
    results = []
    for qp in query_points:
        z = calculate_z_score(data, qp)
        mz = calculate_modified_z_score(data, qp)
        pr = calculate_percentile_rank(data, qp)
        results.append({
            'query_point': qp,
            'z_score': z,
            'modified_z_score': mz,
            'percentile_rank': pr
        })
    return {
        'sorted_data': sorted_data,
        'cdf_data': cdf_data,
        'results': results
    }


if __name__ == '__main__':
    dataset = [12, 15, 18, 20, 22, 25, 28, 30, 32, 35]
    queries = [22, 28, 35]

    result = calculate_metrics(dataset, queries)

    print(f"原始数据集: {dataset}")
    print(f"排序数据集: {result['sorted_data']}")
    print(f"\n累积分布图数据 (值 -> 累积百分比):")
    for val, cum_pct in result['cdf_data']:
        print(f"  {val:>4} -> {cum_pct:>6.1f}%")

    print(f"\n--- 批量查询结果 ---")
    for r in result['results']:
        qp = r['query_point']
        print(f"\n查询点: {qp}")
        print(f"  Z-score:          {r['z_score']:.4f}" if r['z_score'] is not None else "  Z-score:          无法计算")
        print(f"  修正Z-score:      {r['modified_z_score']:.4f}" if r['modified_z_score'] is not None else "  修正Z-score:      无法计算")
        print(f"  百分位排名:        {r['percentile_rank']:.2f}%" if r['percentile_rank'] is not None else "  百分位排名:        无法计算")

    print("\n--- 含异常值的数据集测试 ---")
    outlier_dataset = [10, 12, 13, 14, 15, 16, 17, 18, 100]
    outlier_queries = [15, 100]
    outlier_result = calculate_metrics(outlier_dataset, outlier_queries)
    print(f"数据集: {outlier_dataset}")
    print(f"排序后: {outlier_result['sorted_data']}")
    for r in outlier_result['results']:
        qp = r['query_point']
        print(f"\n查询点: {qp}")
        print(f"  Z-score:          {r['z_score']:.4f}" if r['z_score'] is not None else "  Z-score:          无法计算")
        print(f"  修正Z-score:      {r['modified_z_score']:.4f}" if r['modified_z_score'] is not None else "  修正Z-score:      无法计算")
        print(f"  百分位排名:        {r['percentile_rank']:.2f}%" if r['percentile_rank'] is not None else "  百分位排名:        无法计算")

    print("\n--- 标准差为0的边界测试 ---")
    zero_dataset = [5, 5, 5, 5, 5]
    zero_result = calculate_metrics(zero_dataset, 5)
    print(f"数据集: {zero_dataset}")
    r = zero_result['results'][0]
    print(f"查询点: {r['query_point']}")
    print(f"Z-score:          {r['z_score']:.4f}")
    print(f"修正Z-score:      {r['modified_z_score']:.4f}")
    print(f"百分位排名:        {r['percentile_rank']:.2f}%")
