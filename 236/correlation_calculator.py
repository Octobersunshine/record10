import math
import warnings


def _is_missing(val):
    return val is None or (isinstance(val, float) and math.isnan(val))

def handle_missing_values(x, y, method='listwise'):
    if len(x) != len(y):
        raise ValueError("两个列表长度必须相等")
    
    method = method.lower()
    if method not in ['listwise', 'pairwise']:
        raise ValueError("method参数必须是 'listwise' 或 'pairwise'")
    
    if method == 'listwise':
        valid_pairs = [(xi, yi) for xi, yi in zip(x, y) 
                      if not _is_missing(xi) and not _is_missing(yi)]
        x_clean = [p[0] for p in valid_pairs]
        y_clean = [p[1] for p in valid_pairs]
        return x_clean, y_clean
    else:
        return x, y


def calculate_covariance(x, y):
    if len(x) != len(y):
        raise ValueError("两个列表长度必须相等")
    
    n = len(x)
    if n < 2:
        raise ValueError("列表长度至少为2")
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)
    
    return covariance


def calculate_pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("两个列表长度必须相等")
    
    n = len(x)
    if n < 2:
        raise ValueError("列表长度至少为2")
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    variance_x = sum((xi - mean_x) ** 2 for xi in x)
    variance_y = sum((yi - mean_y) ** 2 for yi in y)
    
    if variance_x == 0 or variance_y == 0:
        if variance_x == 0 and variance_y == 0:
            msg = "两个变量的方差均为零，皮尔逊相关系数无定义"
        elif variance_x == 0:
            msg = "变量 x 的方差为零（所有值相同），皮尔逊相关系数无定义"
        else:
            msg = "变量 y 的方差为零（所有值相同），皮尔逊相关系数无定义"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return float('nan')
    
    correlation = numerator / math.sqrt(variance_x * variance_y)
    
    return correlation


def calculate_correlation_and_covariance(x, y):
    covariance = calculate_covariance(x, y)
    correlation = calculate_pearson_correlation(x, y)
    
    return {
        'pearson_correlation': correlation,
        'covariance': covariance
    }


def _rank_data(data):
    n = len(data)
    indexed = sorted(range(n), key=lambda i: data[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and data[indexed[j + 1]] == data[indexed[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = rank
        i = j + 1
    return ranks


def calculate_spearman_correlation(x, y, missing='listwise'):
    x_clean, y_clean = handle_missing_values(x, y, method=missing)
    n = len(x_clean)
    if n < 2:
        if n == 0:
            warnings.warn("没有有效数据对，斯皮尔曼秩相关系数无定义", RuntimeWarning, stacklevel=2)
        else:
            warnings.warn("有效数据对少于2个，斯皮尔曼秩相关系数无定义", RuntimeWarning, stacklevel=2)
        return float('nan')
    
    ranks_x = _rank_data(x_clean)
    ranks_y = _rank_data(y_clean)
    
    d_squared_sum = sum((rx - ry) ** 2 for rx, ry in zip(ranks_x, ranks_y))
    
    var_rx = sum((r - sum(ranks_x) / n) ** 2 for r in ranks_x)
    var_ry = sum((r - sum(ranks_y) / n) ** 2 for r in ranks_y)
    
    if var_rx == 0 or var_ry == 0:
        if var_rx == 0 and var_ry == 0:
            msg = "两个变量的秩方差均为零，斯皮尔曼秩相关系数无定义"
        elif var_rx == 0:
            msg = "变量 x 的所有值相同，斯皮尔曼秩相关系数无定义"
        else:
            msg = "变量 y 的所有值相同，斯皮尔曼秩相关系数无定义"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return float('nan')
    
    rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))
    
    return rho


def calculate_kendall_tau(x, y, missing='listwise'):
    x_clean, y_clean = handle_missing_values(x, y, method=missing)
    n = len(x_clean)
    if n < 2:
        if n == 0:
            warnings.warn("没有有效数据对，肯德尔tau系数无定义", RuntimeWarning, stacklevel=2)
        else:
            warnings.warn("有效数据对少于2个，肯德尔tau系数无定义", RuntimeWarning, stacklevel=2)
        return float('nan')
    
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x_clean[i] - x_clean[j]
            dy = y_clean[i] - y_clean[j]
            
            if dx == 0 and dy == 0:
                continue
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    
    denom_x = concordant + discordant + ties_x
    denom_y = concordant + discordant + ties_y
    
    if denom_x == 0 or denom_y == 0:
        if denom_x == 0 and denom_y == 0:
            msg = "两个变量的所有值均相同，肯德尔tau系数无定义"
        elif denom_x == 0:
            msg = "变量 x 的所有值均相同，肯德尔tau系数无定义"
        else:
            msg = "变量 y 的所有值均相同，肯德尔tau系数无定义"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return float('nan')
    
    tau = (concordant - discordant) / math.sqrt(denom_x * denom_y)
    
    return tau


def calculate_all_correlations(x, y, missing='listwise'):
    x_clean, y_clean = handle_missing_values(x, y, method=missing)
    
    pearson = calculate_pearson_correlation(x_clean, y_clean) if len(x_clean) >= 2 else float('nan')
    spearman = calculate_spearman_correlation(x_clean, y_clean, missing='listwise')
    kendall = calculate_kendall_tau(x_clean, y_clean, missing='listwise')
    covariance = calculate_covariance(x_clean, y_clean) if len(x_clean) >= 2 else float('nan')
    
    return {
        'pearson_correlation': pearson,
        'spearman_correlation': spearman,
        'kendall_tau': kendall,
        'covariance': covariance,
        'valid_pairs': len(x_clean)
    }


if __name__ == "__main__":
    import warnings as _w
    _w.simplefilter("always")
    
    def _format_corr(val):
        return "NaN" if math.isnan(val) else f"{val:.4f}"

    print("=" * 70)
    print("测试1: 基础相关性对比")
    print("=" * 70)
    x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1_linear = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    y1_mono = [1, 3, 5, 7, 9, 12, 15, 19, 24, 30]
    
    for name, y in [("线性关系", y1_linear), ("单调非线性", y1_mono)]:
        result = calculate_all_correlations(x1, y)
        print(f"\n  {name}:")
        print(f"    皮尔逊: {_format_corr(result['pearson_correlation'])}")
        print(f"    斯皮尔曼: {_format_corr(result['spearman_correlation'])}")
        print(f"    肯德尔tau: {_format_corr(result['kendall_tau'])}")
        print(f"    协方差: {_format_corr(result['covariance'])}")

    print("\n" + "=" * 70)
    print("测试2: 缺失值处理 (listwise 删除)")
    print("=" * 70)
    x2 = [1, 2, None, 4, 5, float('nan'), 7, 8]
    y2 = [2, 4, 6, None, 10, 12, 14, 16]
    result = calculate_all_correlations(x2, y2, missing='listwise')
    print(f"  x: {x2}")
    print(f"  y: {y2}")
    print(f"  有效数据对: {result['valid_pairs']}")
    print(f"  皮尔逊: {_format_corr(result['pearson_correlation'])}")
    print(f"  斯皮尔曼: {_format_corr(result['spearman_correlation'])}")
    print(f"  肯德尔tau: {_format_corr(result['kendall_tau'])}")

    print("\n" + "=" * 70)
    print("测试3: 方差为零的情况")
    print("=" * 70)
    x3 = [5, 5, 5, 5, 5]
    y3 = [1, 2, 3, 4, 5]
    result = calculate_all_correlations(x3, y3)
    print(f"  x (全相同): {x3}")
    print(f"  y: {y3}")
    print(f"  皮尔逊: {_format_corr(result['pearson_correlation'])}")
    print(f"  斯皮尔曼: {_format_corr(result['spearman_correlation'])}")
    print(f"  肯德尔tau: {_format_corr(result['kendall_tau'])}")

    print("\n" + "=" * 70)
    print("测试4: 等级数据")
    print("=" * 70)
    x4 = [10, 20, 30, 40, 50]
    y4 = [5, 3, 4, 2, 1]
    result = calculate_all_correlations(x4, y4)
    print(f"  x: {x4}")
    print(f"  y: {y4}")
    print(f"  皮尔逊: {_format_corr(result['pearson_correlation'])}")
    print(f"  斯皮尔曼: {_format_corr(result['spearman_correlation'])}")
    print(f"  肯德尔tau: {_format_corr(result['kendall_tau'])}")
