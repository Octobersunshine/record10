import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def calculate_covariance_matrix(data):
    """
    计算协方差矩阵

    参数:
        data: 输入数据矩阵，每行一个样本，每列一个特征

    返回:
        cov_matrix: 协方差矩阵
        stats: 协方差矩阵的统计信息
    """
    data_array = np.asarray(data, dtype=np.float64)

    n_samples = data_array.shape[0]
    n_features = data_array.shape[1]

    means = np.mean(data_array, axis=0)
    centered_data = data_array - means

    cov_matrix = np.dot(centered_data.T, centered_data) / (n_samples - 1)

    stats = {
        'shape': cov_matrix.shape,
        'mean': np.mean(cov_matrix),
        'std': np.std(cov_matrix),
        'min': np.min(cov_matrix),
        'max': np.max(cov_matrix),
        'diagonal': np.diag(cov_matrix),
        'diagonal_mean': np.mean(np.diag(cov_matrix)),
        'trace': np.trace(cov_matrix),
        'determinant': np.linalg.det(cov_matrix) if n_features <= 10 else None,
        'rank': np.linalg.matrix_rank(cov_matrix)
    }

    return cov_matrix, stats


def calculate_pearson_correlation(data, warn_zero_variance=True, fill_value=0.0):
    """
    计算皮尔逊相关系数矩阵

    参数:
        data: 输入数据矩阵，每行一个样本，每列一个特征
        warn_zero_variance: 是否对零方差特征发出警告
        fill_value: 当涉及零方差特征时，用什么值填充相关系数（默认0.0）

    返回:
        corr_matrix: 皮尔逊相关系数矩阵
        stats: 相关系数矩阵的统计信息
    """
    data_array = np.asarray(data, dtype=np.float64)

    n_features = data_array.shape[1]

    stds = np.std(data_array, axis=0, ddof=1)
    cov_matrix, _ = calculate_covariance_matrix(data_array)

    zero_variance_indices = np.where(stds <= 1e-10)[0]
    has_zero_variance = len(zero_variance_indices) > 0

    if has_zero_variance and warn_zero_variance:
        if len(zero_variance_indices) == 1:
            warnings.warn(
                f"特征 {zero_variance_indices[0]} 的方差为0（常数特征），"
                f"无法计算有效的皮尔逊相关系数。建议剔除该常数列后再进行计算。",
                UserWarning
            )
        else:
            warnings.warn(
                f"特征 {list(zero_variance_indices)} 的方差为0（常数特征），"
                f"无法计算有效的皮尔逊相关系数。建议剔除这些常数列后再进行计算。",
                UserWarning
            )

    corr_matrix = np.zeros_like(cov_matrix)
    for i in range(n_features):
        for j in range(n_features):
            if stds[i] > 1e-10 and stds[j] > 1e-10:
                corr_matrix[i, j] = cov_matrix[i, j] / (stds[i] * stds[j])
            else:
                corr_matrix[i, j] = fill_value

    np.fill_diagonal(corr_matrix, 1.0)
    for idx in zero_variance_indices:
        corr_matrix[idx, idx] = np.nan

    off_diagonal_mask = ~np.eye(n_features, dtype=bool)
    valid_mask = ~np.isnan(corr_matrix)
    valid_off_diagonal = corr_matrix[off_diagonal_mask & valid_mask]

    stats = {
        'shape': corr_matrix.shape,
        'mean': np.nanmean(corr_matrix),
        'std': np.nanstd(corr_matrix),
        'min': np.nanmin(corr_matrix),
        'max': np.nanmax(corr_matrix),
        'off_diagonal_mean': np.mean(valid_off_diagonal) if len(valid_off_diagonal) > 0 else np.nan,
        'off_diagonal_min': np.min(valid_off_diagonal) if len(valid_off_diagonal) > 0 else np.nan,
        'off_diagonal_max': np.max(valid_off_diagonal) if len(valid_off_diagonal) > 0 else np.nan,
        'high_positive_count': int(np.sum((valid_off_diagonal > 0.7) & (valid_off_diagonal < 1.0))) if len(valid_off_diagonal) > 0 else 0,
        'high_negative_count': int(np.sum(valid_off_diagonal < -0.7)) if len(valid_off_diagonal) > 0 else 0,
        'weak_correlation_count': int(np.sum(np.abs(valid_off_diagonal) < 0.3)) if len(valid_off_diagonal) > 0 else 0,
        'zero_variance_indices': zero_variance_indices,
        'zero_variance_count': len(zero_variance_indices),
        'fill_value': fill_value
    }

    return corr_matrix, stats


def calculate_spearman_correlation(data, warn_zero_variance=True, fill_value=0.0):
    """
    计算斯皮尔曼秩相关系数矩阵（用于检测非线性单调关系）

    参数:
        data: 输入数据矩阵，每行一个样本，每列一个特征
        warn_zero_variance: 是否对零方差特征发出警告
        fill_value: 当涉及零方差特征时，用什么值填充相关系数（默认0.0）

    返回:
        spearman_matrix: 斯皮尔曼秩相关系数矩阵
        stats: 统计信息
    """
    data_array = np.asarray(data, dtype=np.float64)
    n_features = data_array.shape[1]

    stds = np.std(data_array, axis=0, ddof=1)
    zero_variance_indices = np.where(stds <= 1e-10)[0]
    has_zero_variance = len(zero_variance_indices) > 0

    if has_zero_variance and warn_zero_variance:
        if len(zero_variance_indices) == 1:
            warnings.warn(
                f"特征 {zero_variance_indices[0]} 的方差为0（常数特征），"
                f"无法计算有效的斯皮尔曼相关系数。",
                UserWarning
            )
        else:
            warnings.warn(
                f"特征 {list(zero_variance_indices)} 的方差为0（常数特征），"
                f"无法计算有效的斯皮尔曼相关系数。",
                UserWarning
            )

    spearman_matrix = np.zeros((n_features, n_features))
    p_values = np.zeros((n_features, n_features))

    valid_indices = [i for i in range(n_features) if stds[i] > 1e-10]
    for i in range(n_features):
        for j in range(n_features):
            if i in zero_variance_indices or j in zero_variance_indices:
                spearman_matrix[i, j] = fill_value
                p_values[i, j] = np.nan
            else:
                corr, p_val = spearmanr(data_array[:, i], data_array[:, j])
                spearman_matrix[i, j] = corr
                p_values[i, j] = p_val

    for idx in zero_variance_indices:
        spearman_matrix[idx, idx] = np.nan

    off_diagonal_mask = ~np.eye(n_features, dtype=bool)
    valid_mask = ~np.isnan(spearman_matrix)
    valid_off_diagonal = spearman_matrix[off_diagonal_mask & valid_mask]

    stats = {
        'shape': spearman_matrix.shape,
        'mean': np.nanmean(spearman_matrix),
        'std': np.nanstd(spearman_matrix),
        'min': np.nanmin(spearman_matrix),
        'max': np.nanmax(spearman_matrix),
        'off_diagonal_mean': np.mean(valid_off_diagonal) if len(valid_off_diagonal) > 0 else np.nan,
        'off_diagonal_min': np.min(valid_off_diagonal) if len(valid_off_diagonal) > 0 else np.nan,
        'off_diagonal_max': np.max(valid_off_diagonal) if len(valid_off_diagonal) > 0 else np.nan,
        'high_positive_count': int(np.sum((valid_off_diagonal > 0.7) & (valid_off_diagonal < 1.0))) if len(valid_off_diagonal) > 0 else 0,
        'high_negative_count': int(np.sum(valid_off_diagonal < -0.7)) if len(valid_off_diagonal) > 0 else 0,
        'zero_variance_indices': zero_variance_indices,
        'zero_variance_count': len(zero_variance_indices),
        'p_values': p_values
    }

    return spearman_matrix, stats


def calculate_kendall_tau(data, warn_zero_variance=True, fill_value=0.0):
    """
    计算肯德尔tau系数矩阵（等级相关，用于测量有序变量之间的关联）

    参数:
        data: 输入数据矩阵，每行一个样本，每列一个特征
        warn_zero_variance: 是否对零方差特征发出警告
        fill_value: 当涉及零方差特征时，用什么值填充相关系数（默认0.0）

    返回:
        kendall_matrix: 肯德尔tau系数矩阵
        stats: 统计信息
    """
    data_array = np.asarray(data, dtype=np.float64)
    n_features = data_array.shape[1]

    stds = np.std(data_array, axis=0, ddof=1)
    zero_variance_indices = np.where(stds <= 1e-10)[0]
    has_zero_variance = len(zero_variance_indices) > 0

    if has_zero_variance and warn_zero_variance:
        if len(zero_variance_indices) == 1:
            warnings.warn(
                f"特征 {zero_variance_indices[0]} 的方差为0（常数特征），"
                f"无法计算有效的肯德尔tau系数。",
                UserWarning
            )
        else:
            warnings.warn(
                f"特征 {list(zero_variance_indices)} 的方差为0（常数特征），"
                f"无法计算有效的肯德尔tau系数。",
                UserWarning
            )

    kendall_matrix = np.zeros((n_features, n_features))
    p_values = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if i in zero_variance_indices or j in zero_variance_indices:
                kendall_matrix[i, j] = fill_value
                p_values[i, j] = np.nan
            else:
                tau, p_val = kendalltau(data_array[:, i], data_array[:, j])
                kendall_matrix[i, j] = tau
                p_values[i, j] = p_val

    for idx in zero_variance_indices:
        kendall_matrix[idx, idx] = np.nan

    off_diagonal_mask = ~np.eye(n_features, dtype=bool)
    valid_mask = ~np.isnan(kendall_matrix)
    valid_off_diagonal = kendall_matrix[off_diagonal_mask & valid_mask]

    stats = {
        'shape': kendall_matrix.shape,
        'mean': np.nanmean(kendall_matrix),
        'std': np.nanstd(kendall_matrix),
        'min': np.nanmin(kendall_matrix),
        'max': np.nanmax(kendall_matrix),
        'off_diagonal_mean': np.mean(valid_off_diagonal) if len(valid_off_diagonal) > 0 else np.nan,
        'off_diagonal_min': np.min(valid_off_diagonal) if len(valid_off_diagonal) > 0 else np.nan,
        'off_diagonal_max': np.max(valid_off_diagonal) if len(valid_off_diagonal) > 0 else np.nan,
        'high_positive_count': int(np.sum((valid_off_diagonal > 0.7) & (valid_off_diagonal < 1.0))) if len(valid_off_diagonal) > 0 else 0,
        'high_negative_count': int(np.sum(valid_off_diagonal < -0.7)) if len(valid_off_diagonal) > 0 else 0,
        'zero_variance_indices': zero_variance_indices,
        'zero_variance_count': len(zero_variance_indices),
        'p_values': p_values
    }

    return kendall_matrix, stats


def generate_heatmap_data(matrix, feature_names=None, matrix_type='correlation'):
    """
    生成前端可直接渲染的热力图数据

    参数:
        matrix: 相关系数或协方差矩阵
        feature_names: 特征名称列表
        matrix_type: 矩阵类型，用于配置颜色映射范围
            - 'correlation': 相关系数，范围 [-1, 1]
            - 'covariance': 协方差，自适应范围

    返回:
        dict: 包含热力图数据的字典，可直接序列化为JSON传给前端
    """
    matrix_array = np.asarray(matrix)
    n_features = matrix_array.shape[0]

    if feature_names is None:
        feature_names = [f'Feature_{i + 1}' for i in range(n_features)]

    if matrix_type == 'correlation':
        vmin = -1.0
        vmax = 1.0
        colormap = 'RdBu_r'
    else:
        vmin = float(np.nanmin(matrix_array))
        vmax = float(np.nanmax(matrix_array))
        colormap = 'viridis'

    data = []
    for i in range(n_features):
        for j in range(n_features):
            value = matrix_array[i, j]
            if not np.isnan(value):
                data.append({
                    'x': j,
                    'y': i,
                    'value': float(value),
                    'x_label': feature_names[j],
                    'y_label': feature_names[i]
                })

    heatmap_data = {
        'matrix': matrix_array.tolist(),
        'data': data,
        'x_labels': feature_names,
        'y_labels': feature_names,
        'vmin': vmin,
        'vmax': vmax,
        'colormap': colormap,
        'n_features': n_features
    }

    return heatmap_data


def remove_constant_columns(data, feature_names=None):
    """
    剔除数据中的常数（零方差）列

    参数:
        data: 输入数据矩阵，每行一个样本，每列一个特征
        feature_names: 可选，特征名称列表

    返回:
        cleaned_data: 剔除常数列后的数据矩阵
        kept_indices: 保留的列索引
        removed_indices: 被移除的列索引
        new_feature_names: 新的特征名称列表（如果提供了 feature_names）
    """
    data_array = np.asarray(data, dtype=np.float64)
    stds = np.std(data_array, axis=0, ddof=1)

    kept_indices = np.where(stds > 1e-10)[0]
    removed_indices = np.where(stds <= 1e-10)[0]

    cleaned_data = data_array[:, kept_indices]

    new_feature_names = None
    if feature_names is not None:
        new_feature_names = [feature_names[i] for i in kept_indices]

    return cleaned_data, kept_indices, removed_indices, new_feature_names


def analyze_data_matrix(data, warn_zero_variance=True, corr_fill_value=0.0,
                        include_spearman=True, include_kendall=True,
                        generate_heatmaps=True, feature_names=None):
    """
    综合分析：计算协方差矩阵、多种相关系数矩阵及各自的统计信息

    参数:
        data: 输入数据矩阵，每行一个样本，每列一个特征
        warn_zero_variance: 是否对零方差特征发出警告
        corr_fill_value: 当涉及零方差特征时，相关系数的填充值
        include_spearman: 是否计算斯皮尔曼秩相关系数
        include_kendall: 是否计算肯德尔tau系数
        generate_heatmaps: 是否生成热力图数据
        feature_names: 特征名称列表（用于热力图）

    返回:
        dict: 包含各种矩阵、统计信息及热力图数据的字典
    """
    data_array = np.asarray(data, dtype=np.float64)

    if feature_names is None:
        n_features = data_array.shape[1]
        feature_names = [f'Feature_{i + 1}' for i in range(n_features)]

    data_stats = {
        'n_samples': data_array.shape[0],
        'n_features': data_array.shape[1],
        'feature_means': np.mean(data_array, axis=0),
        'feature_stds': np.std(data_array, axis=0, ddof=1),
        'feature_mins': np.min(data_array, axis=0),
        'feature_maxs': np.max(data_array, axis=0)
    }

    cov_matrix, cov_stats = calculate_covariance_matrix(data_array)
    pearson_matrix, pearson_stats = calculate_pearson_correlation(
        data_array,
        warn_zero_variance=warn_zero_variance,
        fill_value=corr_fill_value
    )

    results = {
        'data_stats': data_stats,
        'feature_names': feature_names,
        'covariance_matrix': cov_matrix,
        'covariance_stats': cov_stats,
        'pearson_matrix': pearson_matrix,
        'pearson_stats': pearson_stats,
        'correlation_matrix': pearson_matrix,
        'correlation_stats': pearson_stats
    }

    if include_spearman:
        spearman_matrix, spearman_stats = calculate_spearman_correlation(
            data_array,
            warn_zero_variance=warn_zero_variance,
            fill_value=corr_fill_value
        )
        results['spearman_matrix'] = spearman_matrix
        results['spearman_stats'] = spearman_stats

    if include_kendall:
        kendall_matrix, kendall_stats = calculate_kendall_tau(
            data_array,
            warn_zero_variance=warn_zero_variance,
            fill_value=corr_fill_value
        )
        results['kendall_matrix'] = kendall_matrix
        results['kendall_stats'] = kendall_stats

    if generate_heatmaps:
        results['heatmaps'] = {
            'covariance': generate_heatmap_data(cov_matrix, feature_names, matrix_type='covariance'),
            'pearson': generate_heatmap_data(pearson_matrix, feature_names, matrix_type='correlation')
        }
        if include_spearman:
            results['heatmaps']['spearman'] = generate_heatmap_data(
                results['spearman_matrix'], feature_names, matrix_type='correlation'
            )
        if include_kendall:
            results['heatmaps']['kendall'] = generate_heatmap_data(
                results['kendall_matrix'], feature_names, matrix_type='correlation'
            )

    return results


def print_correlation_summary(title, matrix, stats, feature_names, zero_variance_indices):
    """
    打印相关系数矩阵摘要信息
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    df = pd.DataFrame(matrix, index=feature_names, columns=feature_names)
    print(df.round(4))

    print(f"\n{title}统计信息:")
    print(f"  形状: {stats['shape']}")
    print(f"  整体均值: {stats['mean']:.4f}")
    print(f"  整体标准差: {stats['std']:.4f}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
    if 'off_diagonal_mean' in stats:
        print(f"  非对角线均值: {stats['off_diagonal_mean']:.4f}")
        print(f"  非对角线范围: [{stats['off_diagonal_min']:.4f}, {stats['off_diagonal_max']:.4f}]")
    if 'high_positive_count' in stats:
        print(f"  强正相关对数(|r| > 0.7): {stats['high_positive_count']}")
        print(f"  强负相关对数(r < -0.7): {stats['high_negative_count']}")

    print(f"\n{title} 高相关性特征对 (|r| > 0.7):")
    found = False
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if i in zero_variance_indices or j in zero_variance_indices:
                continue
            r = matrix[i, j]
            if abs(r) > 0.7:
                found = True
                strength = "极强" if abs(r) > 0.9 else "强"
                direction = "正" if r > 0 else "负"
                p_str = ""
                if 'p_values' in stats:
                    p_val = stats['p_values'][i, j]
                    p_str = f", p = {p_val:.4e}"
                print(f"  {feature_names[i]} ↔ {feature_names[j]}: r = {r:.4f} ({strength}{direction}相关{p_str})")
    if not found:
        print("  未发现高相关性特征对")


def print_results(results, feature_names=None):
    """
    格式化输出结果
    """
    if feature_names is None:
        feature_names = results.get('feature_names')
    if feature_names is None:
        n_features = results['data_stats']['n_features']
        feature_names = [f'Feature_{i + 1}' for i in range(n_features)]

    print("=" * 80)
    print("数据统计信息")
    print("=" * 80)
    print(f"样本数量: {results['data_stats']['n_samples']}")
    print(f"特征数量: {results['data_stats']['n_features']}")
    print("\n各特征统计:")
    for i, name in enumerate(feature_names):
        print(f"  {name}:")
        print(f"    均值 = {results['data_stats']['feature_means'][i]:.4f}")
        print(f"    标准差 = {results['data_stats']['feature_stds'][i]:.4f}")
        print(f"    范围 = [{results['data_stats']['feature_mins'][i]:.4f}, {results['data_stats']['feature_maxs'][i]:.4f}]")

    zero_variance_indices = results.get('pearson_stats', {}).get('zero_variance_indices', [])
    if len(zero_variance_indices) > 0:
        print("\n" + "!" * 80)
        print("警告：检测到零方差（常数）特征")
        print("!" * 80)
        for idx in zero_variance_indices:
            name = feature_names[idx] if idx < len(feature_names) else f'Feature_{idx + 1}'
            print(f"  - {name} (索引: {idx})")
        print("\n建议：")
        print("  1. 常数特征无法提供有效的相关性信息")
        print("  2. 建议剔除这些常数列后再进行相关性分析")
        print("  3. 涉及这些特征的相关系数已填充为: "
              f"{results.get('pearson_stats', {}).get('fill_value', 0.0)}")
        print("!" * 80)

    print("\n" + "=" * 80)
    print("协方差矩阵")
    print("=" * 80)
    df_cov = pd.DataFrame(results['covariance_matrix'],
                          index=feature_names,
                          columns=feature_names)
    print(df_cov.round(4))

    print("\n协方差矩阵统计信息:")
    cs = results['covariance_stats']
    print(f"  形状: {cs['shape']}")
    print(f"  整体均值: {cs['mean']:.4f}")
    print(f"  整体标准差: {cs['std']:.4f}")
    print(f"  最小值: {cs['min']:.4f}")
    print(f"  最大值: {cs['max']:.4f}")
    print(f"  迹(对角线和): {cs['trace']:.4f}")
    if cs['determinant'] is not None:
        print(f"  行列式: {cs['determinant']:.4e}")
    print(f"  秩: {cs['rank']}")
    print(f"  对角线元素(各特征方差): {np.round(cs['diagonal'], 4)}")

    print_correlation_summary(
        "皮尔逊相关系数矩阵（线性相关）",
        results.get('pearson_matrix', results.get('correlation_matrix')),
        results.get('pearson_stats', results.get('correlation_stats', {})),
        feature_names,
        zero_variance_indices
    )

    if 'spearman_matrix' in results:
        print_correlation_summary(
            "斯皮尔曼秩相关系数矩阵（非线性单调关系）",
            results['spearman_matrix'],
            results['spearman_stats'],
            feature_names,
            results['spearman_stats'].get('zero_variance_indices', [])
        )

    if 'kendall_matrix' in results:
        print_correlation_summary(
            "肯德尔tau系数矩阵（等级相关）",
            results['kendall_matrix'],
            results['kendall_stats'],
            feature_names,
            results['kendall_stats'].get('zero_variance_indices', [])
        )

    if 'heatmaps' in results:
        print("\n" + "=" * 80)
        print("热力图数据")
        print("=" * 80)
        for key in results['heatmaps']:
            hm = results['heatmaps'][key]
            print(f"  {key}: {hm['n_features']}x{hm['n_features']}, "
                  f"颜色范围 [{hm['vmin']:.2f}, {hm['vmax']:.2f}], "
                  f"色图: {hm['colormap']}")
        print("\n热力图数据结构说明:")
        print("  - matrix: 二维数组，可直接用于前端渲染")
        print("  - data: 点数据列表，每个点包含 x, y, value 及标签")
        print("  - x_labels / y_labels: 坐标轴标签")
        print("  - vmin / vmax: 颜色映射范围")
        print("  - colormap: 推荐的颜色映射名称")


if __name__ == "__main__":
    import json
    np.random.seed(42)

    n_samples = 100
    X1 = np.random.normal(0, 1, n_samples)
    X2 = 0.8 * X1 + np.random.normal(0, 0.3, n_samples)
    X3 = -0.6 * X1 + np.random.normal(0, 0.4, n_samples)
    X4 = np.random.normal(5, 2, n_samples)
    X5 = X2 + X4 + np.random.normal(0, 0.2, n_samples)
    X_const1 = np.full(n_samples, 5.0)
    X_const2 = np.full(n_samples, 10.0)

    data_matrix = np.column_stack([X1, X2, X3, X4, X5, X_const1, X_const2])
    feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'Const1', 'Const2']

    print("=" * 80)
    print("测试1：完整分析（包含皮尔逊、斯皮尔曼、肯德尔、热力图）")
    print("=" * 80)
    results = analyze_data_matrix(
        data_matrix,
        include_spearman=True,
        include_kendall=True,
        generate_heatmaps=True,
        feature_names=feature_names,
        warn_zero_variance=False
    )
    print_results(results, feature_names)

    print("\n\n" + "=" * 80)
    print("测试2：先剔除常数列，再进行分析")
    print("=" * 80)
    cleaned_data, kept_idx, removed_idx, new_names = remove_constant_columns(
        data_matrix, feature_names
    )
    print(f"原特征数: {data_matrix.shape[1]}, 剔除后特征数: {cleaned_data.shape[1]}")
    print(f"被移除的常数列索引: {list(removed_idx)}")
    if len(removed_idx) > 0:
        print(f"被移除的常数列名称: {[feature_names[i] for i in removed_idx]}")

    results_clean = analyze_data_matrix(
        cleaned_data,
        include_spearman=True,
        include_kendall=True,
        generate_heatmaps=True,
        feature_names=new_names,
        warn_zero_variance=True
    )
    print_results(results_clean, new_names)

    print("\n\n" + "=" * 80)
    print("热力图JSON数据示例（前3条数据）")
    print("=" * 80)
    pearson_heatmap = results_clean['heatmaps']['pearson']
    print("Pearson热力图前3条数据:")
    for item in pearson_heatmap['data'][:3]:
        print(f"  {item}")
    print(f"\n可通过 json.dumps(results['heatmaps']['pearson']) 传给前端渲染")
