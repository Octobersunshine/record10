import numpy as np
from scipy import stats


def confidence_interval_from_data(data, confidence_level=0.95, distribution='auto'):
    """
    从原始样本数据计算均值的置信区间
    
    参数:
        data: 样本数据（可迭代对象）
        confidence_level: 置信水平，默认0.95（95%）
        distribution: 分布类型，'z'（正态）、't'（t分布）或'auto'（自动选择）
                    auto模式: n < 30 使用t分布，n >= 30 使用正态分布(Z分布)
        
    返回:
        dict: 包含计算结果的字典
            - 'mean': 样本均值
            - 'std': 样本标准差
            - 'n': 样本量
            - 'df': 自由度(t分布时为n-1，正态分布时为None)
            - 'confidence_level': 置信水平
            - 'lower_bound': 置信区间下界
            - 'upper_bound': 置信区间上界
            - 'margin_of_error': 边际误差
            - 'distribution': 使用的分布类型
            - 'critical_value': 临界值
    """
    data = np.array(data)
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # 样本标准差
    
    return _calculate_ci(sample_mean, sample_std, n, confidence_level, distribution)


def confidence_interval_from_stats(mean, std, n, confidence_level=0.95, distribution='auto'):
    """
    从统计量（均值、标准差、样本量）计算均值的置信区间
    
    参数:
        mean: 样本均值
        std: 样本标准差
        n: 样本量
        confidence_level: 置信水平，默认0.95（95%）
        distribution: 分布类型，'z'（正态）、't'（t分布）或'auto'（自动选择）
                    auto模式: n < 30 使用t分布，n >= 30 使用正态分布(Z分布)
        
    返回:
        dict: 包含计算结果的字典
            - 'mean': 样本均值
            - 'std': 样本标准差
            - 'n': 样本量
            - 'df': 自由度(t分布时为n-1，正态分布时为None)
            - 'confidence_level': 置信水平
            - 'lower_bound': 置信区间下界
            - 'upper_bound': 置信区间上界
            - 'margin_of_error': 边际误差
            - 'distribution': 使用的分布类型
            - 'critical_value': 临界值
    """
    return _calculate_ci(mean, std, n, confidence_level, distribution)


def _calculate_ci(mean, std, n, confidence_level, distribution):
    """
    内部函数：计算置信区间
    """
    if n < 2:
        raise ValueError("样本量必须大于1")
    
    if std <= 0:
        raise ValueError("标准差必须大于0")
    
    if not (0 < confidence_level < 1):
        raise ValueError("置信水平必须在0和1之间（如0.95表示95%）")
    
    alpha = 1 - confidence_level
    
    if distribution == 'auto':
        distribution = 't' if n < 30 else 'z'
    elif distribution not in ['z', 't']:
        raise ValueError("distribution参数必须是'z'、't'或'auto'")
    
    df = None
    if distribution == 'z':
        critical_value = stats.norm.ppf(1 - alpha / 2)
        dist_used = '正态分布(Z分布)'
    else:
        df = n - 1
        critical_value = stats.t.ppf(1 - alpha / 2, df)
        dist_used = f't分布(自由度={df})'
    
    standard_error = std / np.sqrt(n)
    margin_of_error = critical_value * standard_error
    
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return {
        'mean': mean,
        'std': std,
        'n': n,
        'df': df,
        'confidence_level': confidence_level,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'margin_of_error': margin_of_error,
        'distribution': dist_used,
        'critical_value': critical_value,
        'standard_error': standard_error
    }


def proportion_confidence_interval(successes, n, confidence_level=0.95, method='wilson'):
    """
    计算比例p的置信区间
    
    参数:
        successes: 成功次数（或正例数量）
        n: 样本量
        confidence_level: 置信水平，默认0.95（95%）
        method: 计算方法
            - 'wilson': Wilson得分法（推荐，尤其适合小样本）
            - 'normal': 正态近似法
        
    返回:
        dict: 包含计算结果的字典
    """
    if n < 1:
        raise ValueError("样本量必须大于0")
    if successes < 0 or successes > n:
        raise ValueError("成功次数必须在0和n之间")
    if not (0 < confidence_level < 1):
        raise ValueError("置信水平必须在0和1之间")
    
    p_hat = successes / n
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)
    
    if method == 'normal':
        if n * p_hat < 5 or n * (1 - p_hat) < 5:
            import warnings
            warnings.warn("正态近似法建议np和n(1-p)都≥5，考虑使用Wilson得分法")
        
        standard_error = np.sqrt(p_hat * (1 - p_hat) / n)
        margin_of_error = z * standard_error
        lower_bound = p_hat - margin_of_error
        upper_bound = p_hat + margin_of_error
        method_name = '正态近似法'
        
    elif method == 'wilson':
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
        lower_bound = center - margin
        upper_bound = center + margin
        standard_error = np.sqrt(p_hat * (1 - p_hat) / n)
        margin_of_error = upper_bound - p_hat
        method_name = 'Wilson得分法'
    else:
        raise ValueError("method参数必须是'wilson'或'normal'")
    
    lower_bound = max(0, lower_bound)
    upper_bound = min(1, upper_bound)
    
    return {
        'type': 'proportion',
        'p_hat': p_hat,
        'successes': successes,
        'n': n,
        'confidence_level': confidence_level,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'margin_of_error': margin_of_error,
        'standard_error': standard_error,
        'critical_value': z,
        'method': method_name
    }


def variance_confidence_interval(data=None, var=None, n=None, confidence_level=0.95):
    """
    计算方差σ²的置信区间（卡方分布）
    
    参数:
        data: 样本数据（可迭代对象），如果提供则忽略var和n
        var: 样本方差
        n: 样本量
        confidence_level: 置信水平，默认0.95（95%）
        
    返回:
        dict: 包含计算结果的字典
    """
    if data is not None:
        data = np.array(data)
        n = len(data)
        sample_var = np.var(data, ddof=1)
    elif var is not None and n is not None:
        sample_var = var
    else:
        raise ValueError("必须提供data，或者同时提供var和n")
    
    if n < 2:
        raise ValueError("样本量必须大于1")
    if sample_var <= 0:
        raise ValueError("方差必须大于0")
    if not (0 < confidence_level < 1):
        raise ValueError("置信水平必须在0和1之间")
    
    alpha = 1 - confidence_level
    df = n - 1
    
    chi2_lower = stats.chi2.ppf(1 - alpha / 2, df)
    chi2_upper = stats.chi2.ppf(alpha / 2, df)
    
    lower_bound = (df * sample_var) / chi2_lower
    upper_bound = (df * sample_var) / chi2_upper
    
    return {
        'type': 'variance',
        'variance': sample_var,
        'std': np.sqrt(sample_var),
        'n': n,
        'df': df,
        'confidence_level': confidence_level,
        'lower_bound_var': lower_bound,
        'upper_bound_var': upper_bound,
        'lower_bound_std': np.sqrt(lower_bound),
        'upper_bound_std': np.sqrt(upper_bound),
        'chi2_lower': chi2_lower,
        'chi2_upper': chi2_upper,
        'distribution': '卡方分布(χ²)'
    }


def bootstrap_confidence_interval(data, confidence_level=0.95, n_bootstrap=10000, 
                                  statistic=np.mean, method='percentile', random_state=None):
    """
    自助法(Bootstrap)置信区间（非参数）
    
    参数:
        data: 样本数据（可迭代对象）
        confidence_level: 置信水平，默认0.95（95%）
        n_bootstrap: 自助抽样次数，默认10000
        statistic: 要计算的统计量函数，默认np.mean（均值）
        method: 区间计算方法
            - 'percentile': 百分位数法
            - 'normal': 正态近似法
            - 'basic': 基本法（反转法）
        random_state: 随机种子
        
    返回:
        dict: 包含计算结果的字典
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    data = np.array(data)
    n = len(data)
    
    if n < 2:
        raise ValueError("样本量必须大于1")
    if not (0 < confidence_level < 1):
        raise ValueError("置信水平必须在0和1之间")
    
    original_stat = statistic(data)
    
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    if method == 'percentile':
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        method_name = '百分位数法'
    elif method == 'normal':
        boot_mean = np.mean(bootstrap_stats)
        boot_std = np.std(bootstrap_stats, ddof=1)
        z = stats.norm.ppf(1 - alpha / 2)
        lower_bound = original_stat - z * boot_std
        upper_bound = original_stat + z * boot_std
        method_name = '正态近似法'
    elif method == 'basic':
        lower_boot = np.percentile(bootstrap_stats, lower_percentile)
        upper_boot = np.percentile(bootstrap_stats, upper_percentile)
        lower_bound = 2 * original_stat - upper_boot
        upper_bound = 2 * original_stat - lower_boot
        method_name = '基本法(反转法)'
    else:
        raise ValueError("method参数必须是'percentile'、'normal'或'basic'")
    
    return {
        'type': 'bootstrap',
        'statistic_name': statistic.__name__ if hasattr(statistic, '__name__') else 'custom',
        'original_stat': original_stat,
        'n': n,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'bootstrap_mean': np.mean(bootstrap_stats),
        'bootstrap_std': np.std(bootstrap_stats, ddof=1),
        'bootstrap_median': np.median(bootstrap_stats),
        'method': method_name,
        'bootstrap_distribution': bootstrap_stats
    }


def print_result(result):
    """
    格式化输出结果（支持均值、比例、方差、自助法）
    """
    ci_percent = result['confidence_level'] * 100
    
    print("=" * 60)
    print("置信区间计算结果")
    print("=" * 60)
    
    result_type = result.get('type', 'mean')
    
    if result_type == 'mean':
        print(f"样本均值: {result['mean']:.4f}")
        print(f"样本标准差: {result['std']:.4f}")
        print(f"样本量: {result['n']}")
        if result['df'] is not None:
            print(f"自由度(df): {result['df']}")
        print(f"使用分布: {result['distribution']}")
        print(f"置信水平: {ci_percent:.0f}%")
        print(f"标准误: {result['standard_error']:.4f}")
        print(f"临界值: {result['critical_value']:.4f}")
        print(f"边际误差: {result['margin_of_error']:.4f}")
        print("-" * 60)
        print(f"{ci_percent:.0f}% 置信区间: ({result['lower_bound']:.4f}, {result['upper_bound']:.4f})")
        print(f"区间宽度: {result['upper_bound'] - result['lower_bound']:.4f}")
    
    elif result_type == 'proportion':
        print(f"统计类型: 比例置信区间")
        print(f"计算方法: {result['method']}")
        print(f"样本比例: {result['p_hat']:.4f} ({result['successes']}/{result['n']})")
        print(f"样本量: {result['n']}")
        print(f"置信水平: {ci_percent:.0f}%")
        print(f"标准误: {result['standard_error']:.4f}")
        print(f"临界值(Z): {result['critical_value']:.4f}")
        print(f"边际误差: {result['margin_of_error']:.4f}")
        print("-" * 60)
        print(f"{ci_percent:.0f}% 置信区间: ({result['lower_bound']:.4f}, {result['upper_bound']:.4f})")
        print(f"区间宽度: {result['upper_bound'] - result['lower_bound']:.4f}")
    
    elif result_type == 'variance':
        print(f"统计类型: 方差/标准差置信区间")
        print(f"使用分布: {result['distribution']}")
        print(f"样本方差: {result['variance']:.4f}")
        print(f"样本标准差: {result['std']:.4f}")
        print(f"样本量: {result['n']}")
        print(f"自由度(df): {result['df']}")
        print(f"置信水平: {ci_percent:.0f}%")
        print(f"卡方临界值(上): {result['chi2_upper']:.4f}")
        print(f"卡方临界值(下): {result['chi2_lower']:.4f}")
        print("-" * 60)
        print(f"{ci_percent:.0f}% 方差置信区间: ({result['lower_bound_var']:.4f}, {result['upper_bound_var']:.4f})")
        print(f"{ci_percent:.0f}% 标准差置信区间: ({result['lower_bound_std']:.4f}, {result['upper_bound_std']:.4f})")
    
    elif result_type == 'bootstrap':
        print(f"统计类型: 自助法(Bootstrap)置信区间")
        print(f"计算方法: {result['method']}")
        print(f"统计量: {result['statistic_name']}")
        print(f"原始统计值: {result['original_stat']:.4f}")
        print(f"样本量: {result['n']}")
        print(f"自助抽样次数: {result['n_bootstrap']}")
        print(f"置信水平: {ci_percent:.0f}%")
        print(f"自助分布均值: {result['bootstrap_mean']:.4f}")
        print(f"自助分布标准差: {result['bootstrap_std']:.4f}")
        print(f"自助分布中位数: {result['bootstrap_median']:.4f}")
        print("-" * 60)
        print(f"{ci_percent:.0f}% 置信区间: ({result['lower_bound']:.4f}, {result['upper_bound']:.4f})")
        print(f"区间宽度: {result['upper_bound'] - result['lower_bound']:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("=== 第一部分: 均值置信区间 ===")
    print("示例1: 小样本(n=10) - 自动选择t分布")
    data = [23, 25, 28, 22, 26, 24, 27, 21, 25, 23]
    result = confidence_interval_from_data(data, confidence_level=0.95)
    print_result(result)
    
    print("\n" + "=" * 60 + "\n")
    
    print("示例2: 大样本(n=100) - 自动选择Z分布")
    result2 = confidence_interval_from_stats(mean=65, std=12, n=100, confidence_level=0.95)
    print_result(result2)
    
    print("\n" + "=" * 60 + "\n")
    
    print("=== 第二部分: 比例置信区间 ===")
    print("示例3: 比例置信区间 - Wilson得分法（推荐）")
    result3 = proportion_confidence_interval(successes=15, n=50, confidence_level=0.95, method='wilson')
    print_result(result3)
    
    print("\n" + "=" * 60 + "\n")
    
    print("示例4: 比例置信区间 - 正态近似法")
    result4 = proportion_confidence_interval(successes=15, n=50, confidence_level=0.95, method='normal')
    print_result(result4)
    
    print("\n" + "=" * 60 + "\n")
    
    print("=== 第三部分: 方差/标准差置信区间 ===")
    print("示例5: 方差置信区间 - 从原始数据计算")
    data_var = [12, 15, 14, 16, 13, 15, 14, 17, 15, 14]
    result5 = variance_confidence_interval(data=data_var, confidence_level=0.95)
    print_result(result5)
    
    print("\n" + "=" * 60 + "\n")
    
    print("示例6: 方差置信区间 - 从统计量计算")
    result6 = variance_confidence_interval(var=25, n=30, confidence_level=0.95)
    print_result(result6)
    
    print("\n" + "=" * 60 + "\n")
    
    print("=== 第四部分: 自助法(Bootstrap)置信区间 ===")
    print("示例7: 自助法均值置信区间 - 百分位数法")
    np.random.seed(42)
    data_boot = np.random.normal(loc=50, scale=10, size=30)
    result7 = bootstrap_confidence_interval(data_boot, confidence_level=0.95, 
                                            n_bootstrap=5000, statistic=np.mean, 
                                            method='percentile', random_state=42)
    print_result(result7)
    
    print("\n" + "=" * 60 + "\n")
    
    print("示例8: 自助法中位数置信区间 - 正态近似法")
    result8 = bootstrap_confidence_interval(data_boot, confidence_level=0.95, 
                                            n_bootstrap=5000, statistic=np.median, 
                                            method='normal', random_state=42)
    print_result(result8)
    
    print("\n" + "=" * 60 + "\n")
    
    print("示例9: 自助法标准差置信区间 - 基本法")
    result9 = bootstrap_confidence_interval(data_boot, confidence_level=0.95, 
                                            n_bootstrap=5000, statistic=np.std, 
                                            method='basic', random_state=42)
    print_result(result9)
