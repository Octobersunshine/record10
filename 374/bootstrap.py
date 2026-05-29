import numpy as np
from scipy.stats import norm, expon, gamma, lognorm, beta
from typing import Callable, Optional, Tuple, Dict, Any, Union


def _compute_bias_correction(
    bootstrap_stats: np.ndarray,
    original_stat: float
) -> float:
    prop_less = np.mean(bootstrap_stats < original_stat)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = norm.ppf(prop_less)
    return z0


def _compute_acceleration(
    sample: np.ndarray,
    stat_func: Callable[[np.ndarray], float]
) -> float:
    n = len(sample)
    jackknife_stats = np.empty(n)
    
    for i in range(n):
        jackknife_sample = np.delete(sample, i)
        jackknife_stats[i] = stat_func(jackknife_sample)
    
    jackknife_mean = np.mean(jackknife_stats)
    numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
    denominator = 6.0 * (np.sum((jackknife_mean - jackknife_stats) ** 2)) ** (3/2)
    
    if denominator < 1e-10:
        return 0.0
    
    a = numerator / denominator
    return a


def _bca_percentiles(
    bootstrap_stats: np.ndarray,
    original_stat: float,
    sample: np.ndarray,
    stat_func: Callable[[np.ndarray], float],
    confidence_level: float
) -> Tuple[float, float]:
    z0 = _compute_bias_correction(bootstrap_stats, original_stat)
    a = _compute_acceleration(sample, stat_func)
    
    alpha = 1 - confidence_level
    alpha1 = alpha / 2
    alpha2 = 1 - alpha / 2
    
    z_alpha1 = norm.ppf(alpha1)
    z_alpha2 = norm.ppf(alpha2)
    
    def adjust_z(z, z0, a):
        numerator = z0 + z
        denominator = 1 - a * (z0 + z)
        if abs(denominator) < 1e-10:
            return z0
        return z0 + numerator / denominator
    
    z1_adj = adjust_z(z_alpha1, z0, a)
    z2_adj = adjust_z(z_alpha2, z0, a)
    
    p1 = norm.cdf(z1_adj) * 100
    p2 = norm.cdf(z2_adj) * 100
    
    p1 = np.clip(p1, 0, 100)
    p2 = np.clip(p2, 0, 100)
    
    ci_lower = np.percentile(bootstrap_stats, p1)
    ci_upper = np.percentile(bootstrap_stats, p2)
    
    return ci_lower, ci_upper


def _fit_distribution(
    sample: np.ndarray,
    dist_type: str
) -> Tuple[Any, np.ndarray]:
    if dist_type == 'normal':
        mu, std = np.mean(sample), np.std(sample, ddof=1)
        dist = norm(loc=mu, scale=std)
        params = np.array([mu, std])
    elif dist_type == 'exponential':
        loc, scale = expon.fit(sample)
        dist = expon(loc=loc, scale=scale)
        params = np.array([loc, scale])
    elif dist_type == 'gamma':
        a, loc, scale = gamma.fit(sample)
        dist = gamma(a=a, loc=loc, scale=scale)
        params = np.array([a, loc, scale])
    elif dist_type == 'lognormal':
        shape, loc, scale = lognorm.fit(sample)
        dist = lognorm(s=shape, loc=loc, scale=scale)
        params = np.array([shape, loc, scale])
    elif dist_type == 'beta':
        a, b, loc, scale = beta.fit(sample)
        dist = beta(a=a, b=b, loc=loc, scale=scale)
        params = np.array([a, b, loc, scale])
    else:
        raise ValueError(
            f"Unknown distribution type: {dist_type}. "
            f"Use 'normal', 'exponential', 'gamma', 'lognormal', or 'beta'."
        )
    return dist, params


def _parametric_resample(
    sample: np.ndarray,
    n_bootstraps: int,
    dist_type: str
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = len(sample)
    dist, params = _fit_distribution(sample, dist_type)
    bootstrap_samples = dist.rvs(size=(n_bootstraps, n))
    info = {
        'distribution': dist_type,
        'params': params,
        'params_names': _get_param_names(dist_type)
    }
    return bootstrap_samples, info


def _get_param_names(dist_type: str) -> list:
    names = {
        'normal': ['mu', 'sigma'],
        'exponential': ['loc', 'scale'],
        'gamma': ['shape', 'loc', 'scale'],
        'lognormal': ['shape', 'loc', 'scale'],
        'beta': ['a', 'b', 'loc', 'scale']
    }
    return names.get(dist_type, [])


def _block_resample(
    sample: np.ndarray,
    n_bootstraps: int,
    block_size: Optional[int] = None,
    block_type: str = 'moving'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = len(sample)
    
    if block_size is None:
        block_size = max(2, int(np.sqrt(n)))
    
    if block_size >= n:
        raise ValueError(f"block_size ({block_size}) must be less than sample size ({n})")
    
    if block_type == 'moving':
        n_blocks = n - block_size + 1
        blocks = np.array([sample[i:i+block_size] for i in range(n_blocks)])
    elif block_type == 'fixed':
        n_complete_blocks = n // block_size
        blocks = np.array([
            sample[i*block_size:(i+1)*block_size] 
            for i in range(n_complete_blocks)
        ])
        if n % block_size != 0:
            last_block = sample[n_complete_blocks*block_size:]
            blocks = np.vstack([blocks, np.pad(
                last_block, 
                (0, block_size - len(last_block)),
                mode='constant',
                constant_values=np.nan
            )])
    else:
        raise ValueError(
            f"Unknown block_type: {block_type}. Use 'moving' or 'fixed'."
        )
    
    n_blocks_needed = int(np.ceil(n / block_size))
    bootstrap_samples = np.empty((n_bootstraps, n))
    
    for i in range(n_bootstraps):
        block_indices = np.random.choice(len(blocks), size=n_blocks_needed, replace=True)
        selected_blocks = blocks[block_indices].flatten()
        bootstrap_samples[i] = selected_blocks[:n]
    
    info = {
        'block_size': block_size,
        'block_type': block_type,
        'n_blocks': len(blocks)
    }
    return bootstrap_samples, info


def _nonparametric_resample(
    sample: np.ndarray,
    n_bootstraps: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = len(sample)
    indices = np.random.choice(n, size=(n_bootstraps, n), replace=True)
    bootstrap_samples = sample[indices]
    info = {'resample_method': 'nonparametric'}
    return bootstrap_samples, info


def _compute_histogram(
    data: np.ndarray,
    bins: Optional[int] = None
) -> Dict[str, Any]:
    if bins is None:
        bins = min(50, max(10, int(np.sqrt(len(data)))))
    
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    return {
        'counts': counts,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'bin_widths': bin_widths,
        'density': counts / (len(data) * bin_widths),
        'n_bins': bins
    }


def bootstrap_confidence_interval(
    sample: np.ndarray,
    stat_func: Callable[[np.ndarray], float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    method: str = 'auto',
    resample_method: str = 'nonparametric',
    parametric_dist: str = 'normal',
    block_size: Optional[int] = None,
    block_type: str = 'moving',
    return_histogram: bool = False,
    histogram_bins: Optional[int] = None,
    return_bootstrap_stats: bool = False,
    random_state: Optional[int] = None
) -> Union[Tuple[float, float, float], Dict[str, Any]]:
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(sample)
    original_stat = stat_func(sample)
    
    if resample_method == 'nonparametric':
        bootstrap_samples, resample_info = _nonparametric_resample(
            sample, n_bootstraps
        )
    elif resample_method == 'parametric':
        bootstrap_samples, resample_info = _parametric_resample(
            sample, n_bootstraps, parametric_dist
        )
    elif resample_method == 'block':
        bootstrap_samples, resample_info = _block_resample(
            sample, n_bootstraps, block_size, block_type
        )
    else:
        raise ValueError(
            f"Unknown resample_method: {resample_method}. "
            f"Use 'nonparametric', 'parametric', or 'block'."
        )
    
    bootstrap_stats = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        if resample_method == 'block' and block_type == 'fixed' and n % block_size != 0:
            valid_data = bootstrap_samples[i][~np.isnan(bootstrap_samples[i])]
            bootstrap_stats[i] = stat_func(valid_data)
        else:
            bootstrap_stats[i] = stat_func(bootstrap_samples[i])
    
    if method == 'auto':
        method = 'bca' if n < 30 else 'percentile'
    
    if method == 'percentile':
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    elif method == 'bca':
        ci_lower, ci_upper = _bca_percentiles(
            bootstrap_stats, original_stat, sample, stat_func, confidence_level
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'percentile', 'bca', or 'auto'."
        )
    
    if not return_histogram and not return_bootstrap_stats:
        return original_stat, ci_lower, ci_upper
    
    result = {
        'original_stat': original_stat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'method': method,
        'resample_method': resample_method,
        'resample_info': resample_info,
        'n_bootstraps': n_bootstraps,
        'sample_size': n
    }
    
    if return_histogram:
        result['histogram'] = _compute_histogram(bootstrap_stats, histogram_bins)
    
    if return_bootstrap_stats:
        result['bootstrap_stats'] = bootstrap_stats
    
    return result


def mean_ci(
    sample: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    method: str = 'auto',
    resample_method: str = 'nonparametric',
    parametric_dist: str = 'normal',
    block_size: Optional[int] = None,
    block_type: str = 'moving',
    return_histogram: bool = False,
    histogram_bins: Optional[int] = None,
    return_bootstrap_stats: bool = False,
    random_state: Optional[int] = None
) -> Union[Tuple[float, float, float], Dict[str, Any]]:
    return bootstrap_confidence_interval(
        sample, np.mean, n_bootstraps, confidence_level, method,
        resample_method, parametric_dist, block_size, block_type,
        return_histogram, histogram_bins, return_bootstrap_stats, random_state
    )


def median_ci(
    sample: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    method: str = 'auto',
    resample_method: str = 'nonparametric',
    parametric_dist: str = 'normal',
    block_size: Optional[int] = None,
    block_type: str = 'moving',
    return_histogram: bool = False,
    histogram_bins: Optional[int] = None,
    return_bootstrap_stats: bool = False,
    random_state: Optional[int] = None
) -> Union[Tuple[float, float, float], Dict[str, Any]]:
    return bootstrap_confidence_interval(
        sample, np.median, n_bootstraps, confidence_level, method,
        resample_method, parametric_dist, block_size, block_type,
        return_histogram, histogram_bins, return_bootstrap_stats, random_state
    )


def std_ci(
    sample: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    method: str = 'auto',
    resample_method: str = 'nonparametric',
    parametric_dist: str = 'normal',
    block_size: Optional[int] = None,
    block_type: str = 'moving',
    return_histogram: bool = False,
    histogram_bins: Optional[int] = None,
    return_bootstrap_stats: bool = False,
    random_state: Optional[int] = None
) -> Union[Tuple[float, float, float], Dict[str, Any]]:
    return bootstrap_confidence_interval(
        sample, np.std, n_bootstraps, confidence_level, method,
        resample_method, parametric_dist, block_size, block_type,
        return_histogram, histogram_bins, return_bootstrap_stats, random_state
    )


if __name__ == "__main__":
    print("=" * 70)
    print("测试1: 非参数自助法 vs BCa (小样本 n=8)")
    print("=" * 70)
    np.random.seed(123)
    sample_small = np.random.normal(loc=50, scale=10, size=8)
    
    print(f"样本量: {len(sample_small)}")
    print(f"真实均值: 50")
    print(f"样本均值: {np.mean(sample_small):.2f}")
    print()
    
    mean_val_p, mean_lower_p, mean_upper_p = mean_ci(
        sample_small, n_bootstraps=5000, method='percentile', 
        resample_method='nonparametric', random_state=42
    )
    mean_val_b, mean_lower_b, mean_upper_b = mean_ci(
        sample_small, n_bootstraps=5000, method='bca', 
        resample_method='nonparametric', random_state=42
    )
    
    print(f"均值置信区间对比 (95%):")
    print(f"  百分位数法: ({mean_lower_p:.2f}, {mean_upper_p:.2f}), 宽度: {mean_upper_p - mean_lower_p:.2f}")
    print(f"  BCa方法:    ({mean_lower_b:.2f}, {mean_upper_b:.2f}), 宽度: {mean_upper_b - mean_lower_b:.2f}")
    print()
    
    print("=" * 70)
    print("测试2: 参数自助法 (正态分布假设)")
    print("=" * 70)
    np.random.seed(456)
    sample_norm = np.random.normal(loc=50, scale=10, size=15)
    
    print(f"样本量: {len(sample_norm)}")
    print(f"样本均值: {np.mean(sample_norm):.2f}, 样本标准差: {np.std(sample_norm, ddof=1):.2f}")
    print()
    
    result_para = mean_ci(
        sample_norm, n_bootstraps=5000, 
        resample_method='parametric', parametric_dist='normal',
        return_histogram=True, return_bootstrap_stats=True,
        random_state=42
    )
    
    print("参数自助法结果:")
    print(f"  分布: {result_para['resample_info']['distribution']}")
    dist_params = result_para['resample_info']['params']
    param_names = result_para['resample_info']['params_names']
    for name, val in zip(param_names, dist_params):
        print(f"  拟合参数 {name}: {val:.4f}")
    print(f"  均值置信区间: ({result_para['ci_lower']:.2f}, {result_para['ci_upper']:.2f})")
    print(f"  区间宽度: {result_para['ci_upper'] - result_para['ci_lower']:.2f}")
    print(f"  直方图箱数: {result_para['histogram']['n_bins']}")
    print(f"  直方图计数范围: [{result_para['histogram']['counts'].min()}, {result_para['histogram']['counts'].max()}]")
    print(f"  Bootstrap统计量均值: {result_para['bootstrap_stats'].mean():.2f}")
    print()
    
    print("=" * 70)
    print("测试3: 参数自助法 - 不同分布对比")
    print("=" * 70)
    np.random.seed(789)
    sample_exp = np.random.exponential(scale=10, size=20)
    
    print(f"指数分布样本 (真实均值=10)")
    print(f"样本量: {len(sample_exp)}")
    print(f"样本均值: {np.mean(sample_exp):.2f}")
    print()
    
    for dist in ['normal', 'exponential', 'gamma']:
        val, lower, upper = mean_ci(
            sample_exp, n_bootstraps=3000,
            resample_method='parametric', parametric_dist=dist,
            method='percentile', random_state=123
        )
        print(f"  {dist:12s}: CI: ({lower:.2f}, {upper:.2f}), 宽度: {upper-lower:.2f}")
    print()
    
    print("=" * 70)
    print("测试4: 块自助法 (时间序列数据)")
    print("=" * 70)
    np.random.seed(321)
    
    def generate_ar1(n, phi=0.7, mu=100, sigma=5):
        x = np.zeros(n)
        x[0] = mu
        for i in range(1, n):
            x[i] = mu + phi * (x[i-1] - mu) + np.random.normal(0, sigma)
        return x
    
    ts_data = generate_ar1(100, phi=0.7)
    
    print(f"AR(1)时间序列 (n={len(ts_data)}, phi=0.7)")
    print(f"样本均值: {np.mean(ts_data):.2f}")
    print(f"滞后1阶自相关: {np.corrcoef(ts_data[:-1], ts_data[1:])[0,1]:.3f}")
    print()
    
    block_sizes = [5, 10, 15]
    for bs in block_sizes:
        val, lower, upper = mean_ci(
            ts_data, n_bootstraps=3000,
            resample_method='block', block_size=bs, block_type='moving',
            method='percentile', random_state=42
        )
        print(f"  块长={bs:2d}: CI=({lower:.2f}, {upper:.2f}), 宽度={upper-lower:.2f}")
    
    print()
    val_np, lower_np, upper_np = mean_ci(
        ts_data, n_bootstraps=3000,
        resample_method='nonparametric',
        method='percentile', random_state=42
    )
    print(f"  普通自助法: CI=({lower_np:.2f}, {upper_np:.2f}), 宽度={upper_np-lower_np:.2f}")
    print()
    print("说明: 块自助法保持了时间序列的自相关性，")
    print("      普通自助法会破坏自相关性，通常导致区间偏窄")
    print()
    
    print("=" * 70)
    print("测试5: 固定块 vs 移动块对比")
    print("=" * 70)
    for btype in ['moving', 'fixed']:
        result_block = mean_ci(
            ts_data, n_bootstraps=2000,
            resample_method='block', block_size=8, block_type=btype,
            method='percentile', return_histogram=True,
            random_state=42
        )
        info = result_block['resample_info']
        print(f"{btype}块自助法:")
        print(f"  块大小: {info['block_size']}")
        print(f"  块数量: {info['n_blocks']}")
        print(f"  CI: ({result_block['ci_lower']:.2f}, {result_block['ci_upper']:.2f})")
        print(f"  直方图箱数: {result_block['histogram']['n_bins']}")
        print()
    
    print("=" * 70)
    print("测试6: 直方图数据结构示例")
    print("=" * 70)
    result_hist = median_ci(
        sample_small, n_bootstraps=5000,
        resample_method='nonparametric', method='bca',
        return_histogram=True, histogram_bins=20,
        random_state=42
    )
    
    hist = result_hist['histogram']
    print(f"中位数的Bootstrap分布:")
    print(f"  箱数: {hist['n_bins']}")
    print(f"  箱范围: [{hist['bin_edges'][0]:.2f} 到 {hist['bin_edges'][-1]:.2f}]")
    print(f"  最高计数: {hist['counts'].max()} (在 {hist['bin_centers'][hist['counts'].argmax()]:.2f} 附近)")
    print(f"  前5个箱中心: {np.round(hist['bin_centers'][:5], 2)}")
    print(f"  前5个计数: {hist['counts'][:5]}")
    print()
    
    print("=" * 70)
    print("测试7: 覆盖率模拟 - 参数 vs 非参数")
    print("=" * 70)
    print("100次模拟，比较不同重采样方法的覆盖率...")
    print()
    
    n_simulations = 100
    true_mean = 50
    
    coverage_nonparam = 0
    coverage_param = 0
    
    np.random.seed(999)
    for sim in range(n_simulations):
        sample = np.random.normal(loc=true_mean, scale=10, size=10)
        
        _, lower_np, upper_np = mean_ci(
            sample, n_bootstraps=2000, method='bca',
            resample_method='nonparametric', random_state=sim
        )
        _, lower_p, upper_p = mean_ci(
            sample, n_bootstraps=2000, method='percentile',
            resample_method='parametric', parametric_dist='normal', random_state=sim
        )
        
        if lower_np <= true_mean <= upper_np:
            coverage_nonparam += 1
        if lower_p <= true_mean <= upper_p:
            coverage_param += 1
    
    print(f"样本量 n=10, 目标覆盖率 95%")
    print(f"非参数BCa: {coverage_nonparam/n_simulations*100:.1f}%")
    print(f"参数自助法(正态): {coverage_param/n_simulations*100:.1f}%")
    print()
    print("完成所有测试!")
