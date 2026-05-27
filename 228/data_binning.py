import numpy as np
import pandas as pd
from math import log2, ceil


def sturges_rule(data):
    """
    Sturges公式: bins = ceil(1 + log2(n))
    适用于近似正态分布的数据，对大样本偏少
    """
    n = len(data)
    if n == 0:
        return 1
    return max(1, ceil(1 + log2(n)))


def freedman_diaconis_rule(data):
    """
    Freedman-Diaconis规则: bin_width = 2 * IQR / n^(1/3), bins = range / bin_width
    对异常值鲁棒，适用于偏态分布
    """
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return sturges_rule(data)
    bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
    data_range = data.max() - data.min()
    if data_range == 0:
        return 1
    return max(1, ceil(data_range / bin_width))


def sqrt_rule(data):
    """
    平方根法则: bins = ceil(sqrt(n))
    简单直观，适用于中等样本量
    """
    n = len(data)
    if n == 0:
        return 1
    return max(1, ceil(n ** 0.5))


def auto_bins(data, rule='fd'):
    """
    自动选择最优箱数

    参数:
        data: 数组类数据
        rule: 箱数选择规则
            'sturges' - Sturges公式
            'fd'      - Freedman-Diaconis规则（默认）
            'sqrt'    - 平方根法则

    返回:
        推荐的箱数
    """
    data = np.asarray(data)
    rules = {
        'sturges': sturges_rule,
        'fd': freedman_diaconis_rule,
        'sqrt': sqrt_rule,
    }
    if rule not in rules:
        raise ValueError(f"rule 必须是 {list(rules.keys())} 之一，得到: {rule}")
    return rules[rule](data)


def equal_width_binning(data, bins, min_val=None, max_val=None):
    """
    等宽分箱：将数据区间分成宽度相等的区间
    """
    data = np.asarray(data)
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_edges[-1] = max_val + 1e-10
    bin_indices = np.digitize(data, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    
    return bin_indices, bin_edges


def equal_freq_binning(data, bins, tol=None):
    """
    等频分箱：每个区间包含大致相等的样本数
    修复相同值跨箱问题：基于唯一值确定分箱边界，确保相同值在同一箱子
    
    参数:
        data: 数组类数据
        bins: 区间数量
        tol: 容差，当两个值的差小于等于tol时视为相同值，默认为None（精确匹配）
    """
    data = np.asarray(data)
    sorted_data = np.sort(data)
    n = len(data)
    
    if n == 0:
        return np.array([]), np.array([0.0, 1e-10])
    
    unique_vals, unique_indices = np.unique(sorted_data, return_index=True)
    unique_counts = np.diff(np.append(unique_indices, n))
    
    if len(unique_vals) <= bins:
        bin_edges = np.append(unique_vals, unique_vals[-1] + 1e-10)
        bin_indices = np.digitize(data, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(unique_vals) - 1)
        return bin_indices, bin_edges
    
    target_count = n / bins
    bin_edges = [sorted_data[0]]
    current_count = 0
    current_bin_idx = 1
    
    for val, count in zip(unique_vals, unique_counts):
        current_count += count
        
        if current_bin_idx < bins and current_count >= target_count * current_bin_idx:
            if tol is None:
                bin_edges.append(val)
            else:
                if len(bin_edges) > 0 and abs(val - bin_edges[-1]) > tol:
                    bin_edges.append(val)
                elif len(bin_edges) == 0:
                    bin_edges.append(val)
            current_bin_idx += 1
    
    bin_edges = np.unique(bin_edges)
    
    while len(bin_edges) > bins + 1:
        gaps = np.diff(bin_edges)
        merge_idx = np.argmin(gaps)
        bin_edges = np.delete(bin_edges, merge_idx + 1)
    
    bin_edges[-1] = sorted_data[-1] + 1e-10
    
    bin_indices = np.digitize(data, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    
    return bin_indices, bin_edges


def calculate_binning_stats(data, bin_indices, bin_edges, bins, weights=None):
    """
    计算每个区间的频数、频率和累积频率

    参数:
        data: 原始数据
        bin_indices: 每个数据点所属区间索引
        bin_edges: 区间边界
        bins: 区间数量
        weights: 每个数据点的权重，None表示等权重
    """
    data = np.asarray(data)
    n = len(data)
    
    if weights is not None:
        weights = np.asarray(weights)
        if len(weights) != n:
            raise ValueError(f"权重长度({len(weights)})与数据长度({n})不匹配")
        total_weight = weights.sum()
        if total_weight == 0:
            raise ValueError("权重总和不能为零")
        weighted_counts = np.zeros(bins)
        for i in range(n):
            weighted_counts[bin_indices[i]] += weights[i]
        counts = weighted_counts
        frequencies = counts / total_weight
    else:
        counts = np.bincount(bin_indices, minlength=bins).astype(float)
        frequencies = counts / n
    
    cumulative_frequencies = np.cumsum(frequencies)
    
    bin_labels = []
    for i in range(bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == bins - 1:
            right = bin_edges[i + 1] - 1e-10
            label = f'[{left:.4f}, {right:.4f}]'
        else:
            label = f'[{left:.4f}, {right:.4f})'
        bin_labels.append(label)
    
    if weights is not None:
        raw_counts = np.bincount(bin_indices, minlength=bins).astype(float)
        result = pd.DataFrame({
            '区间': bin_labels,
            '左边界': bin_edges[:-1],
            '右边界': bin_edges[1:],
            '频数': raw_counts,
            '加权频数': counts,
            '频率': frequencies,
            '累积频率': cumulative_frequencies
        })
    else:
        result = pd.DataFrame({
            '区间': bin_labels,
            '左边界': bin_edges[:-1],
            '右边界': bin_edges[1:],
            '频数': counts,
            '频率': frequencies,
            '累积频率': cumulative_frequencies
        })
    
    histogram = {
        'bin_edges': bin_edges.tolist(),
        'counts': counts.tolist()
    }
    
    return result, histogram


def bin_data(data, bins='auto', method='equal_width', min_val=None, max_val=None,
             tol=None, weights=None, rule='fd'):
    """
    数据分箱主函数

    参数:
        data: 数组类数据
        bins: 区间数量，整数或 'auto'（自动选择最优箱数）
        method: 'equal_width'（等宽）或 'equal_freq'（等频）
        min_val: 最小值（仅等宽分箱可用）
        max_val: 最大值（仅等宽分箱可用）
        tol: 容差（仅等频分箱可用），小于等于tol的值视为相同
        weights: 每个数据点的权重数组，None表示等权重
        rule: 自动选箱规则，'sturges'/'fd'/'sqrt'，默认'fd'

    返回:
        stats_df: 包含频数、频率、累积频率的DataFrame
        histogram: 包含区间边界和频数的字典
        bin_indices: 每个数据点所属的区间索引
    """
    data = np.asarray(data)
    
    if isinstance(bins, str):
        if bins != 'auto':
            raise ValueError("bins 为字符串时仅支持 'auto'")
        bins = auto_bins(data, rule)
    
    if method == 'equal_width':
        bin_indices, bin_edges = equal_width_binning(data, bins, min_val, max_val)
    elif method == 'equal_freq':
        bin_indices, bin_edges = equal_freq_binning(data, bins, tol)
    else:
        raise ValueError("method 必须是 'equal_width' 或 'equal_freq'")
    
    actual_bins = len(bin_edges) - 1
    stats_df, histogram = calculate_binning_stats(
        data, bin_indices, bin_edges, actual_bins, weights
    )
    
    return stats_df, histogram, bin_indices


def check_binning_consistency(data, bin_indices):
    """检查相同值是否被分到相同的箱子"""
    data = np.asarray(data)
    bin_indices = np.asarray(bin_indices)
    
    value_to_bins = {}
    for val, bin_idx in zip(data, bin_indices):
        if val not in value_to_bins:
            value_to_bins[val] = set()
        value_to_bins[val].add(bin_idx)
    
    inconsistent = {k: v for k, v in value_to_bins.items() if len(v) > 1}
    return inconsistent


if __name__ == '__main__':
    np.random.seed(42)
    
    print("=" * 60)
    print("测试1：自动选择最优箱数")
    print("=" * 60)
    data = np.random.normal(100, 20, 200)
    print(f"\n数据量: {len(data)}")
    
    for rule_name in ['sturges', 'fd', 'sqrt']:
        nbins = auto_bins(data, rule=rule_name)
        print(f"  {rule_name:10s} -> 推荐箱数: {nbins}")
    
    stats, hist, _ = bin_data(data, bins='auto', method='equal_width', rule='fd')
    print(f"\n使用FD规则自动分箱结果（{len(hist['bin_edges'])-1}箱）：")
    print(stats.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("测试2：加权直方图")
    print("=" * 60)
    data_small = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    weights = np.array([1, 1, 1, 1, 1, 10, 10, 10, 10, 10])
    print(f"\n数据:   {data_small}")
    print(f"权重:   {weights}")
    print("(后半部分权重远大于前半部分)")
    
    stats_unweighted, _, _ = bin_data(data_small, bins=5, method='equal_width')
    stats_weighted, hist_w, _ = bin_data(data_small, bins=5, method='equal_width', weights=weights)
    
    print("\n--- 无权直方图 ---")
    print(stats_unweighted.to_string(index=False))
    print("\n--- 加权直方图 ---")
    print(stats_weighted.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("测试3：含大量重复值的等频分箱一致性检查")
    print("=" * 60)
    data_with_duplicates = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8])
    
    stats, hist, indices = bin_data(data_with_duplicates, bins=4, method='equal_freq')
    print("\n分箱统计结果：")
    print(stats.to_string(index=False))
    
    inconsistent = check_binning_consistency(data_with_duplicates, indices)
    if inconsistent:
        print(f"\n❌ 发现不一致的分箱: {inconsistent}")
    else:
        print("\n✅ 分箱一致性验证通过：相同值在同一箱子")
    
    print("\n" + "=" * 60)
    print("测试4：自动选箱 + 加权直方图组合")
    print("=" * 60)
    data_combo = np.concatenate([
        np.random.normal(50, 5, 50),
        np.random.normal(80, 10, 50)
    ])
    weights_combo = np.concatenate([
        np.ones(50),
        np.ones(50) * 3
    ])
    
    stats_combo, hist_combo, _ = bin_data(
        data_combo, bins='auto', method='equal_width',
        weights=weights_combo, rule='fd'
    )
    print(f"\n双峰数据（n=100），第二簇权重×3，FD规则自动选箱:")
    print(stats_combo.to_string(index=False))
    print(f"\n直方图数据：区间边界数={len(hist_combo['bin_edges'])}, 箱数={len(hist_combo['bin_edges'])-1}")
