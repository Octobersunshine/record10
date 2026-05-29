import numpy as np


def get_method_info():
    """
    返回计算方法的详细说明
    
    返回:
        包含方法说明的字典
    """
    return {
        'method_name': '类型7分位点算法 (Type 7, Hyndman & Fan)',
        'compatible_with': 'R语言默认 (quantile(x, type=7))',
        'formula': '索引位置 i = 1 + (n-1) * p',
        'description': '使用线性插值法，基于加权平均的样本分位数估计。对于排序后的数据 x_1 ≤ x_2 ≤ ... ≤ x_n，索引位置 i = 1 + (n-1)*p，其中 p 为分位概率。令 k = floor(i)，d = i - k，则分位数 Q(p) = x_k + d * (x_{k+1} - x_k)。该方法对小样本（n<5）也能给出稳定且合理的估计值，是统计软件中最常用的分位数计算方法之一。',
        'advantages': [
            '小样本（n<5）时结果稳定连续，无跳跃',
            '与R语言默认行为完全一致，结果可复现',
            '连续单调函数，p从0到1时结果平滑变化',
            '符合大多数统计软件（R、Python pandas）的默认行为'
        ]
    }


def type7_quantile(sorted_data, p, return_intermediate=False):
    """
    实现类型7分位点算法（与R语言quantile(x, type=7)完全一致）
    
    参数:
        sorted_data: 已排序的数据集（从小到大）
        p: 分位概率，0 ≤ p ≤ 1
        return_intermediate: 是否返回中间计算过程值
    
    返回:
        如果 return_intermediate=False，返回分位数值
        如果 return_intermediate=True，返回 (分位数值, 中间值字典)
    """
    n = len(sorted_data)
    
    intermediate = {
        'n': n,
        'p': p,
        'sorted_data': list(sorted_data) if n <= 20 else f"[...{n}个元素...]"
    }
    
    if n == 0:
        result = np.nan
        intermediate['note'] = '空数据集'
        if return_intermediate:
            return result, intermediate
        return result
    if n == 1:
        result = sorted_data[0]
        intermediate['note'] = '单个数据点'
        if return_intermediate:
            return result, intermediate
        return result
    
    i = 1 + (n - 1) * p
    k = int(np.floor(i))
    d = i - k
    
    intermediate['index_i'] = i
    intermediate['k'] = k
    intermediate['d'] = d
    
    if k >= n:
        result = sorted_data[-1]
        intermediate['note'] = '索引超出上界，取最大值'
        intermediate['x_k'] = sorted_data[-1]
    elif k <= 0:
        result = sorted_data[0]
        intermediate['note'] = '索引超出下界，取最小值'
        intermediate['x_k'] = sorted_data[0]
    else:
        x_k = sorted_data[k - 1]
        x_k1 = sorted_data[k]
        result = x_k + d * (x_k1 - x_k)
        intermediate['x_k'] = x_k
        intermediate['x_k+1'] = x_k1
        intermediate['calculation'] = f"{x_k} + {d:.4f} * ({x_k1} - {x_k}) = {result:.4f}"
    
    if return_intermediate:
        return result, intermediate
    return result


def weighted_type7_quantile(data, weights, p, return_intermediate=False):
    """
    加权类型7分位点算法
    
    参数:
        data: 数据集
        weights: 权重列表（与data一一对应）
        p: 分位概率，0 ≤ p ≤ 1
        return_intermediate: 是否返回中间计算过程值
    
    返回:
        如果 return_intermediate=False，返回加权分位数值
        如果 return_intermediate=True，返回 (分位数值, 中间值字典)
    """
    data_array = np.array(data)
    weights_array = np.array(weights, dtype=float)
    
    if len(data_array) != len(weights_array):
        raise ValueError("数据和权重的长度必须相同")
    
    if len(data_array) == 0:
        if return_intermediate:
            return np.nan, {'note': '空数据集'}
        return np.nan
    
    sort_idx = np.argsort(data_array)
    sorted_data = data_array[sort_idx]
    sorted_weights = weights_array[sort_idx]
    
    total_weight = np.sum(sorted_weights)
    normalized_weights = sorted_weights / total_weight
    
    cumulative_weights = np.cumsum(normalized_weights)
    
    intermediate = {
        'n': len(data_array),
        'p': p,
        'sorted_data': list(sorted_data) if len(sorted_data) <= 20 else f"[...{len(sorted_data)}个元素...]",
        'sorted_weights': list(sorted_weights) if len(sorted_weights) <= 20 else f"[...{len(sorted_weights)}个元素...]",
        'total_weight': total_weight,
        'normalized_weights': list(normalized_weights) if len(normalized_weights) <= 20 else f"[...{len(normalized_weights)}个元素...]",
        'cumulative_weights': list(cumulative_weights) if len(cumulative_weights) <= 20 else f"[...{len(cumulative_weights)}个元素...]"
    }
    
    if p <= cumulative_weights[0]:
        result = sorted_data[0]
        intermediate['note'] = '分位概率在第一个数据点之前'
        if return_intermediate:
            return result, intermediate
        return result
    
    if p >= cumulative_weights[-1]:
        result = sorted_data[-1]
        intermediate['note'] = '分位概率在最后一个数据点之后'
        if return_intermediate:
            return result, intermediate
        return result
    
    k = np.searchsorted(cumulative_weights, p) - 1
    
    if k >= len(cumulative_weights) - 1:
        k = len(cumulative_weights) - 2
    
    w_k = cumulative_weights[k]
    w_k1 = cumulative_weights[k + 1]
    
    d = (p - w_k) / (w_k1 - w_k) if (w_k1 - w_k) > 0 else 0
    
    x_k = sorted_data[k]
    x_k1 = sorted_data[k + 1]
    result = x_k + d * (x_k1 - x_k)
    
    intermediate['k'] = int(k)
    intermediate['w_k'] = w_k
    intermediate['w_k+1'] = w_k1
    intermediate['d'] = d
    intermediate['x_k'] = x_k
    intermediate['x_k+1'] = x_k1
    intermediate['calculation'] = f"{x_k} + {d:.4f} * ({x_k1} - {x_k}) = {result:.4f}"
    
    if return_intermediate:
        return result, intermediate
    return result


def calculate_quantiles(data, quantiles=None, include_method_info=False, 
                         return_intermediate=False, weights=None):
    """
    计算数据集的分位数，使用类型7分位点算法（与R语言默认一致）
    
    参数:
        data: 数据集（列表或数组）
        quantiles: 要计算的分位数列表，如 [0.25, 0.5, 0.75]
        include_method_info: 是否包含计算方法说明
        return_intermediate: 是否返回每个分位数的中间计算过程
        weights: 可选，权重列表（与data一一对应），用于加权分位数计算
    
    返回:
        基本返回: 分位数字典 {分位数值: 计算结果}
        如果 include_method_info=True: (分位数字典, 方法说明字典)
        如果 return_intermediate=True: (分位数字典, 中间值字典)
        如果两者都为True: (分位数字典, 方法说明字典, 中间值字典)
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    
    data_array = np.array(data)
    results = {}
    intermediates = {}
    
    if weights is not None:
        for q in quantiles:
            if return_intermediate:
                results[q], intermediates[q] = weighted_type7_quantile(data_array, weights, q, return_intermediate=True)
            else:
                results[q] = weighted_type7_quantile(data_array, weights, q)
    else:
        sorted_data = sorted(data_array)
        for q in quantiles:
            if return_intermediate:
                results[q], intermediates[q] = type7_quantile(sorted_data, q, return_intermediate=True)
            else:
                results[q] = type7_quantile(sorted_data, q)
    
    result_list = [results]
    
    if include_method_info:
        result_list.append(get_method_info())
    
    if return_intermediate:
        result_list.append(intermediates)
    
    if len(result_list) == 1:
        return result_list[0]
    return tuple(result_list)


def calculate_quartiles_and_iqr(data, include_method_info=False, 
                                 return_intermediate=False, weights=None):
    """
    计算四分位数（Q1, Q2/中位数, Q3）和四分位距IQR
    
    参数:
        data: 数据集（列表或数组）
        include_method_info: 是否包含计算方法说明
        return_intermediate: 是否返回每个分位数的中间计算过程
        weights: 可选，权重列表（与data一一对应），用于加权分位数计算
    
    返回:
        基本返回: 包含Q1, Q2, Q3, IQR的字典
        如果 include_method_info=True: (结果字典, 方法说明字典)
        如果 return_intermediate=True: (结果字典, 中间值字典)
        如果两者都为True: (结果字典, 方法说明字典, 中间值字典)
    """
    quartile_result = calculate_quantiles(
        data, [0.25, 0.5, 0.75], 
        include_method_info=False, 
        return_intermediate=return_intermediate,
        weights=weights
    )
    
    if return_intermediate:
        quartiles, intermediates = quartile_result
    else:
        quartiles = quartile_result
        intermediates = None
    
    q1 = quartiles[0.25]
    q2 = quartiles[0.5]
    q3 = quartiles[0.75]
    iqr = q3 - q1
    
    result = {
        'Q1': q1,
        'Q2 (中位数)': q2,
        'Q3': q3,
        'IQR': iqr
    }
    
    result_list = [result]
    
    if include_method_info:
        result_list.append(get_method_info())
    
    if return_intermediate:
        result_list.append(intermediates)
    
    if len(result_list) == 1:
        return result_list[0]
    return tuple(result_list)


def calculate_boxplot_stats(data, weights=None, iqr_factor=1.5):
    """
    计算箱线图统计数据
    
    参数:
        data: 数据集（列表或数组）
        weights: 可选，权重列表，用于加权分位数计算
        iqr_factor: IQR因子，用于计算异常值边界（默认1.5）
    
    返回:
        包含箱线图所有统计量的字典
    """
    data_array = np.array(data)
    
    quartile_result = calculate_quantiles(
        data, [0.25, 0.5, 0.75], 
        include_method_info=False,
        return_intermediate=False,
        weights=weights
    )
    
    q1 = quartile_result[0.25]
    q2 = quartile_result[0.5]
    q3 = quartile_result[0.75]
    iqr = q3 - q1
    
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    
    sorted_data = np.sort(data_array)
    
    lower_whisker_candidates = sorted_data[sorted_data >= lower_bound]
    if len(lower_whisker_candidates) > 0:
        lower_whisker = lower_whisker_candidates[0]
    else:
        lower_whisker = q1
    
    upper_whisker_candidates = sorted_data[sorted_data <= upper_bound]
    if len(upper_whisker_candidates) > 0:
        upper_whisker = upper_whisker_candidates[-1]
    else:
        upper_whisker = q3
    
    outliers_lower = sorted_data[sorted_data < lower_whisker].tolist()
    outliers_upper = sorted_data[sorted_data > upper_whisker].tolist()
    outliers = outliers_lower + outliers_upper
    
    return {
        'Q1': q1,
        'Q2 (中位数)': q2,
        'Q3': q3,
        'IQR': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'lower_whisker': lower_whisker,
        'upper_whisker': upper_whisker,
        'outliers_lower': outliers_lower,
        'outliers_upper': outliers_upper,
        'outliers': outliers,
        'n_outliers': len(outliers),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'mean': float(np.mean(data_array)),
        'n': len(data_array),
        'iqr_factor': iqr_factor
    }


def print_results(data, custom_quantiles=None, weights=None):
    """
    打印分位数计算结果
    
    参数:
        data: 数据集
        custom_quantiles: 自定义分位数列表
        weights: 可选，权重列表
    """
    method_info = get_method_info()
    
    print("=" * 60)
    print("数据集:", data)
    print("数据排序后:", sorted(data))
    print("样本量 n =", len(data))
    if weights is not None:
        print("权重:", weights)
    print("=" * 60)
    
    quartile_results = calculate_quartiles_and_iqr(data, weights=weights)
    weight_str = "（加权）" if weights is not None else ""
    print(f"\n四分位数结果{weight_str}（{method_info['method_name']}）:")
    print("-" * 40)
    for key, value in quartile_results.items():
        print(f"  {key}: {value:.4f}")
    
    if custom_quantiles:
        custom_results = calculate_quantiles(data, custom_quantiles, weights=weights)
        print(f"\n自定义分位数结果{weight_str}:")
        print("-" * 40)
        for q, value in custom_results.items():
            print(f"  {int(q*100)}% 分位数 (Q{q}): {value:.4f}")
    
    print("\n" + "=" * 60)


def print_boxplot_stats(data, weights=None, iqr_factor=1.5):
    """
    打印箱线图统计数据
    
    参数:
        data: 数据集
        weights: 可选，权重列表
        iqr_factor: IQR因子
    """
    stats = calculate_boxplot_stats(data, weights=weights, iqr_factor=iqr_factor)
    
    print("=" * 60)
    print("箱线图统计数据")
    print("=" * 60)
    print(f"样本量: {stats['n']}")
    print(f"最小值: {stats['min']:.4f}")
    print(f"最大值: {stats['max']:.4f}")
    print(f"均值: {stats['mean']:.4f}")
    print("-" * 40)
    print(f"Q1 (25%分位数): {stats['Q1']:.4f}")
    print(f"Q2 (中位数): {stats['Q2 (中位数)']:.4f}")
    print(f"Q3 (75%分位数): {stats['Q3']:.4f}")
    print(f"IQR (四分位距): {stats['IQR']:.4f}")
    print("-" * 40)
    print(f"异常值边界 (IQR × {stats['iqr_factor']}):")
    print(f"  下界: {stats['lower_bound']:.4f}")
    print(f"  上界: {stats['upper_bound']:.4f}")
    print("-" * 40)
    print(f"下须端点: {stats['lower_whisker']:.4f}")
    print(f"上须端点: {stats['upper_whisker']:.4f}")
    print("-" * 40)
    print(f"异常值数量: {stats['n_outliers']}")
    if stats['outliers_lower']:
        print(f"  低端异常值: {stats['outliers_lower']}")
    if stats['outliers_upper']:
        print(f"  高端异常值: {stats['outliers_upper']}")
    if not stats['outliers']:
        print("  无异常值")
    print("=" * 60)


def print_intermediate_values(intermediates):
    """
    打印中间计算过程值
    
    参数:
        intermediates: 中间值字典
    """
    print("=" * 60)
    print("分位数计算中间过程")
    print("=" * 60)
    for q, intermediate in intermediates.items():
        print(f"\n{int(q*100)}% 分位数 (Q{q}):")
        print("-" * 40)
        for key, value in intermediate.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    print("=" * 60)


def print_method_info():
    """
    打印计算方法说明
    """
    info = get_method_info()
    print("=" * 60)
    print("计算方法说明")
    print("=" * 60)
    print(f"方法名称: {info['method_name']}")
    print(f"兼容软件: {info['compatible_with']}")
    print(f"计算公式: {info['formula']}")
    print("-" * 40)
    print(f"详细说明: {info['description']}")
    print("-" * 40)
    print("方法优势:")
    for i, adv in enumerate(info['advantages'], 1):
        print(f"  {i}. {adv}")
    print("=" * 60)


if __name__ == "__main__":
    print_method_info()
    print("\n")
    
    sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sample_data2 = [3, 7, 8, 5, 12, 14, 21, 15, 18, 14]
    sample_data3 = [1, 3, 5, 7, 9, 11, 13]
    sample_data_with_outliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    
    small_sample_n2 = [5, 10]
    small_sample_n3 = [2, 5, 8]
    small_sample_n4 = [1, 4, 6, 9]
    
    print("示例 1 (n=10):")
    print_results(sample_data, custom_quantiles=[0.1, 0.9])
    
    print("\n\n示例 2 (n=10):")
    print_results(sample_data2, custom_quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    
    print("\n\n示例 3 (n=7, 奇数个数据):")
    print_results(sample_data3, custom_quantiles=[0.05, 0.95])
    
    print("\n\n示例 4 (小样本 n=2):")
    print_results(small_sample_n2, custom_quantiles=[0.25, 0.5, 0.75])
    
    print("\n\n示例 5 (小样本 n=3):")
    print_results(small_sample_n3, custom_quantiles=[0.1, 0.5, 0.9])
    
    print("\n\n示例 6 (小样本 n=4):")
    print_results(small_sample_n4, custom_quantiles=[0.2, 0.4, 0.6, 0.8])
    
    print("\n\n示例 7 (加权分位数):")
    weighted_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 5]
    print_results(weighted_data, custom_quantiles=[0.25, 0.5, 0.75], weights=weights)
    
    print("\n\n示例 8 (箱线图统计 - 含异常值):")
    print_boxplot_stats(sample_data_with_outliers)
    
    print("\n\n示例 9 (计算过程中间值):")
    _, intermediates = calculate_quantiles(sample_data, [0.25, 0.5, 0.75], return_intermediate=True)
    print_intermediate_values(intermediates)
    
    print("\n\n" + "=" * 60)
    print("API 使用示例:")
    print("=" * 60)
    
    data = [5, 10]
    print("\n数据:", data)
    
    result, method_info = calculate_quartiles_and_iqr(data, include_method_info=True)
    print("\n计算结果:", result)
    print("\n方法说明:")
    print(f"  方法: {method_info['method_name']}")
    print(f"  公式: {method_info['formula']}")
    print(f"  兼容: {method_info['compatible_with']}")
    
    quantiles, method_info2 = calculate_quantiles(data, [0.1, 0.9], include_method_info=True)
    print("\n自定义分位数:", quantiles)
    
    print("\n\n箱线图API示例:")
    box_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50]
    box_stats = calculate_boxplot_stats(box_data)
    print(f"异常值列表: {box_stats['outliers']}")
    print(f"下须: {box_stats['lower_whisker']}, 上须: {box_stats['upper_whisker']}")
    
    print("\n\n加权分位数API示例:")
    w_data = [1, 2, 3, 4, 5]
    w_weights = [1, 1, 1, 1, 10]
    w_result = calculate_quantiles(w_data, [0.25, 0.5, 0.75], weights=w_weights)
    print(f"加权分位数结果:", w_result)
    
    print("\n\n获取中间值示例:")
    data = [1, 2, 3, 4, 5, 6, 7,8,9,10]
    q_results, _, inters = calculate_quantiles(data, [0.25, 0.75], include_method_info=True, return_intermediate=True)
    print(f"Q1计算过程: {inters[0.25]['calculation']}")
    print(f"Q3计算过程: {inters[0.75]['calculation']}")
    print("=" * 60)
