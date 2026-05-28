import numpy as np
from typing import Callable, Tuple, Optional, Union, Dict, Any
import multiprocessing as mp
from functools import partial

try:
    from scipy.stats import norm
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def norm_ppf(p):
        return np.sqrt(2) * np.arctanh(2 * p - 1)
    
    def norm_cdf(x):
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))
    
    class _Norm:
        @staticmethod
        def ppf(p):
            return norm_ppf(p)
        
        @staticmethod
        def cdf(x):
            return norm_cdf(x)
    
    norm = _Norm()


def bootstrap_resample(data: np.ndarray, B: int) -> np.ndarray:
    n = len(data)
    indices = np.random.randint(0, n, size=(B, n))
    return data[indices]


def compute_statistic(data: np.ndarray, stat_func: Callable) -> float:
    return stat_func(data)


def percentile_ci(boot_stats: np.ndarray, alpha: float) -> Tuple[float, float]:
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lower, upper


def jackknife_values(data: np.ndarray, stat_func: Callable) -> np.ndarray:
    n = len(data)
    jackknife_stats = np.zeros(n)
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats[i] = stat_func(jackknife_sample)
    return jackknife_stats


def bca_ci(
    data: np.ndarray,
    boot_stats: np.ndarray,
    stat_func: Callable,
    alpha: float
) -> Tuple[float, float]:
    n = len(data)
    B = len(boot_stats)

    original_stat = stat_func(data)

    z0 = norm.ppf(np.mean(boot_stats < original_stat))

    jackknife_stats = jackknife_values(data, stat_func)
    jack_mean = np.mean(jackknife_stats)
    numerator = np.sum((jack_mean - jackknife_stats) ** 3)
    denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** (3 / 2)
    a = numerator / denominator if denominator != 0 else 0.0

    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)

    alpha1 = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    alpha2 = norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

    lower = np.percentile(boot_stats, 100 * alpha1)
    upper = np.percentile(boot_stats, 100 * alpha2)

    return lower, upper


def fit_distribution(
    data: np.ndarray,
    dist_type: str = 'norm'
) -> Dict[str, Any]:
    data = np.asarray(data)

    if dist_type == 'norm':
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)
        return {'dist': 'norm', 'mu': mu, 'sigma': sigma}
    elif dist_type == 'expon':
        lambda_est = 1.0 / np.mean(data) if np.mean(data) > 0 else 1.0
        return {'dist': 'expon', 'lambda': lambda_est}
    elif dist_type == 'poisson':
        lambda_est = np.mean(data)
        return {'dist': 'poisson', 'lambda': lambda_est}
    elif dist_type == 'binomial':
        p = np.mean(data)
        return {'dist': 'binomial', 'p': p, 'n': 1}
    elif dist_type == 'gamma':
        if HAS_SCIPY:
            shape, loc, scale = scipy_stats.gamma.fit(data, floc=0)
            return {'dist': 'gamma', 'shape': shape, 'scale': scale}
        else:
            mean = np.mean(data)
            var = np.var(data, ddof=1)
            shape = mean ** 2 / var if var > 0 else 1.0
            scale = var / mean if mean > 0 else 1.0
            return {'dist': 'gamma', 'shape': shape, 'scale': scale}
    elif dist_type == 't':
        if HAS_SCIPY:
            df, loc, scale = scipy_stats.t.fit(data)
            return {'dist': 't', 'df': df, 'loc': loc, 'scale': scale}
        else:
            return {'dist': 'norm', 'mu': np.mean(data), 'sigma': np.std(data, ddof=1)}
    else:
        raise ValueError(f"Unsupported distribution: {dist_type}")


def parametric_resample(
    data: np.ndarray,
    B: int,
    dist_type: str = 'norm',
    dist_params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    n = len(data)

    if dist_params is None:
        dist_params = fit_distribution(data, dist_type)

    samples = np.zeros((B, n))

    if dist_params['dist'] == 'norm':
        mu = dist_params['mu']
        sigma = dist_params['sigma']
        samples = np.random.normal(mu, sigma, size=(B, n))
    elif dist_params['dist'] == 'expon':
        lambda_est = dist_params['lambda']
        samples = np.random.exponential(1.0 / lambda_est, size=(B, n))
    elif dist_params['dist'] == 'poisson':
        lambda_est = dist_params['lambda']
        samples = np.random.poisson(lambda_est, size=(B, n))
    elif dist_params['dist'] == 'binomial':
        p = dist_params['p']
        n_trials = dist_params['n']
        samples = np.random.binomial(n_trials, p, size=(B, n))
    elif dist_params['dist'] == 'gamma':
        shape = dist_params['shape']
        scale = dist_params['scale']
        samples = np.random.gamma(shape, scale, size=(B, n))
    elif dist_params['dist'] == 't':
        if HAS_SCIPY:
            df = dist_params['df']
            loc = dist_params['loc']
            scale = dist_params['scale']
            samples = scipy_stats.t.rvs(df, loc, scale, size=(B, n))
        else:
            mu = dist_params.get('mu', np.mean(data))
            sigma = dist_params.get('sigma', np.std(data, ddof=1))
            samples = np.random.normal(mu, sigma, size=(B, n))
    else:
        raise ValueError(f"Unsupported distribution: {dist_params['dist']}")

    return samples


def parametric_ci(
    data: np.ndarray,
    stat_func: Callable,
    B: int = 1000,
    alpha: float = 0.05,
    dist_type: str = 'norm'
) -> Tuple[float, float]:
    boot_samples = parametric_resample(data, B, dist_type)
    boot_stats = np.array([stat_func(sample) for sample in boot_samples])
    return percentile_ci(boot_stats, alpha)


def block_resample(
    data: np.ndarray,
    B: int,
    block_size: int,
    method: str = 'moving'
) -> np.ndarray:
    n = len(data)
    if block_size >= n:
        return np.tile(data, (B, 1))

    boot_samples = np.zeros((B, n))

    if method == 'fixed':
        n_blocks = (n + block_size - 1) // block_size
        for b in range(B):
            pos = 0
            for _ in range(n_blocks):
                if pos >= n:
                    break
                block_idx = np.random.randint(0, n - block_size + 1)
                block = data[block_idx:block_idx + block_size]
                take = min(block_size, n - pos)
                boot_samples[b, pos:pos + take] = block[:take]
                pos += take
    elif method == 'moving':
        for b in range(B):
            pos = 0
            while pos < n:
                block_idx = np.random.randint(0, n - block_size + 1)
                block = data[block_idx:block_idx + block_size]
                take = min(block_size, n - pos)
                boot_samples[b, pos:pos + take] = block[:take]
                pos += take
    elif method == 'circular':
        extended_data = np.concatenate([data, data])
        for b in range(B):
            pos = 0
            while pos < n:
                start = np.random.randint(0, n)
                block = extended_data[start:start + block_size]
                take = min(block_size, n - pos)
                boot_samples[b, pos:pos + take] = block[:take]
                pos += take
    else:
        raise ValueError(f"Unknown block method: {method}. Use 'fixed', 'moving', or 'circular'.")

    return boot_samples


def block_bootstrap_ci(
    data: np.ndarray,
    stat_func: Callable,
    B: int = 1000,
    alpha: float = 0.05,
    block_size: Optional[int] = None,
    block_method: str = 'moving'
) -> Tuple[float, float]:
    n = len(data)

    if block_size is None:
        block_size = max(2, int(np.sqrt(n)))

    boot_samples = block_resample(data, B, block_size, block_method)
    boot_stats = np.array([stat_func(sample) for sample in boot_samples])
    return percentile_ci(boot_stats, alpha)


def _compute_single_bootstrap(
    data: np.ndarray,
    stat_func: Callable,
    seed: int,
    resample_type: str = 'nonparametric',
    **kwargs
) -> float:
    np.random.seed(seed)

    if resample_type == 'nonparametric':
        n = len(data)
        indices = np.random.randint(0, n, size=n)
        sample = data[indices]
    elif resample_type == 'parametric':
        dist_type = kwargs.get('dist_type', 'norm')
        dist_params = kwargs.get('dist_params')
        sample = parametric_resample(data, 1, dist_type, dist_params)[0]
    elif resample_type == 'block':
        block_size = kwargs.get('block_size', 5)
        block_method = kwargs.get('block_method', 'moving')
        sample = block_resample(data, 1, block_size, block_method)[0]
    else:
        raise ValueError(f"Unknown resample type: {resample_type}")

    return stat_func(sample)


def _compute_batch_bootstrap(
    data: np.ndarray,
    stat_func: Callable,
    seeds: np.ndarray,
    resample_type: str = 'nonparametric',
    **kwargs
) -> np.ndarray:
    results = np.zeros(len(seeds))
    for i, seed in enumerate(seeds):
        results[i] = _compute_single_bootstrap(data, stat_func, seed, resample_type, **kwargs)
    return results


def parallel_bootstrap_stats(
    data: np.ndarray,
    stat_func: Callable,
    B: int = 1000,
    n_jobs: int = -1,
    resample_type: str = 'nonparametric',
    random_state: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    n_jobs = max(1, min(n_jobs, mp.cpu_count()))

    if random_state is not None:
        np.random.seed(random_state)
    seeds = np.random.randint(0, 2**31 - 1, size=B)

    min_B_for_parallel = 5000 if resample_type == 'nonparametric' else 2000
    if n_jobs == 1 or B < min_B_for_parallel:
        boot_stats = np.zeros(B)
        for i in range(B):
            boot_stats[i] = _compute_single_bootstrap(
                data, stat_func, seeds[i], resample_type, **kwargs
            )
        return boot_stats

    batch_size = max(1, B // (n_jobs * 4))
    seed_batches = [seeds[i:i + batch_size] for i in range(0, B, batch_size)]

    func = partial(
        _compute_batch_bootstrap,
        data, stat_func,
        resample_type=resample_type,
        **kwargs
    )

    try:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(func, seed_batches)
        return np.concatenate(results)
    except Exception:
        boot_stats = np.zeros(B)
        for i in range(B):
            boot_stats[i] = _compute_single_bootstrap(
                data, stat_func, seeds[i], resample_type, **kwargs
            )
        return boot_stats


def bootstrap_se(
    data: np.ndarray,
    stat_func: Callable,
    B: int = 100
) -> float:
    boot_samples = bootstrap_resample(data, B)
    boot_stats = np.array([stat_func(sample) for sample in boot_samples])
    return np.std(boot_stats, ddof=1)


def studentized_ci(
    data: np.ndarray,
    stat_func: Callable,
    B: int = 1000,
    B_inner: int = 100,
    alpha: float = 0.05
) -> Tuple[float, float]:
    n = len(data)
    original_stat = stat_func(data)
    original_se = bootstrap_se(data, stat_func, B=B_inner)

    boot_samples = bootstrap_resample(data, B)
    t_stats = np.zeros(B)

    for i in range(B):
        boot_sample = boot_samples[i]
        boot_stat = stat_func(boot_sample)
        boot_se = bootstrap_se(boot_sample, stat_func, B=B_inner)
        if boot_se > 0:
            t_stats[i] = (boot_stat - original_stat) / boot_se
        else:
            t_stats[i] = 0.0

    t_lower = np.percentile(t_stats, 100 * (1 - alpha / 2))
    t_upper = np.percentile(t_stats, 100 * alpha / 2)

    lower = original_stat - t_lower * original_se
    upper = original_stat - t_upper * original_se

    return lower, upper


def double_bootstrap_ci(
    data: np.ndarray,
    stat_func: Callable,
    B1: int = 500,
    B2: int = 200,
    alpha: float = 0.05
) -> Tuple[float, float]:
    n = len(data)
    original_stat = stat_func(data)

    boot_samples1 = bootstrap_resample(data, B1)
    boot_stats1 = np.array([stat_func(sample) for sample in boot_samples1])

    p_values = np.zeros(B1)
    for i in range(B1):
        boot_sample1 = boot_samples1[i]
        boot_stat1 = boot_stats1[i]
        boot_samples2 = bootstrap_resample(boot_sample1, B2)
        boot_stats2 = np.array([stat_func(sample) for sample in boot_samples2])
        p_values[i] = np.mean(boot_stats2 <= boot_stat1)

    sorted_p = np.sort(p_values)
    lower_idx = int(alpha / 2 * B1)
    upper_idx = int((1 - alpha / 2) * B1)
    lower_idx = max(0, min(lower_idx, B1 - 1))
    upper_idx = max(0, min(upper_idx, B1 - 1))

    alpha1_calibrated_lower = sorted_p[lower_idx]
    alpha1_calibrated_upper = sorted_p[upper_idx]

    lower = np.percentile(boot_stats1, 100 * alpha1_calibrated_lower)
    upper = np.percentile(boot_stats1, 100 * alpha1_calibrated_upper)

    if upper <= lower or (upper - lower) < 0.1 * np.abs(original_stat):
        boot_se = np.std(boot_stats1, ddof=1)
        lower = original_stat - norm.ppf(1 - alpha / 2) * boot_se
        upper = original_stat + norm.ppf(1 - alpha / 2) * boot_se

    return lower, upper


def bootstrap_ci(
    data: np.ndarray,
    stat_func: Callable,
    B: int = 1000,
    alpha: float = 0.05,
    method: str = 'percentile',
    random_state: Optional[int] = None,
    B_inner: int = 100,
    B2: int = 200,
    n_jobs: int = 1,
    dist_type: str = 'norm',
    dist_params: Optional[Dict[str, Any]] = None,
    block_size: Optional[int] = None,
    block_method: str = 'moving'
) -> Tuple[float, float, float, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data)
    original_stat = stat_func(data)
    boot_stats = None
    ci = None

    if method == 'percentile':
        if n_jobs == 1:
            boot_samples = bootstrap_resample(data, B)
            boot_stats = np.array([stat_func(sample) for sample in boot_samples])
        else:
            boot_stats = parallel_bootstrap_stats(
                data, stat_func, B, n_jobs, 'nonparametric', random_state
            )
        ci = percentile_ci(boot_stats, alpha)
    elif method == 'bca':
        if n_jobs == 1:
            boot_samples = bootstrap_resample(data, B)
            boot_stats = np.array([stat_func(sample) for sample in boot_samples])
        else:
            boot_stats = parallel_bootstrap_stats(
                data, stat_func, B, n_jobs, 'nonparametric', random_state
            )
        ci = bca_ci(data, boot_stats, stat_func, alpha)
    elif method == 'studentized':
        if n_jobs == 1:
            boot_samples = bootstrap_resample(data, B)
            boot_stats = np.array([stat_func(sample) for sample in boot_samples])
        else:
            boot_stats = parallel_bootstrap_stats(
                data, stat_func, B, n_jobs, 'nonparametric', random_state
            )
        ci = studentized_ci(data, stat_func, B, B_inner, alpha)
    elif method == 'double':
        if n_jobs == 1:
            boot_samples = bootstrap_resample(data, B)
            boot_stats = np.array([stat_func(sample) for sample in boot_samples])
        else:
            boot_stats = parallel_bootstrap_stats(
                data, stat_func, B, n_jobs, 'nonparametric', random_state
            )
        ci = double_bootstrap_ci(data, stat_func, B, B2, alpha)
    elif method == 'parametric':
        if n_jobs == 1:
            boot_samples = parametric_resample(data, B, dist_type, dist_params)
            boot_stats = np.array([stat_func(sample) for sample in boot_samples])
        else:
            if dist_params is None:
                dist_params = fit_distribution(data, dist_type)
            boot_stats = parallel_bootstrap_stats(
                data, stat_func, B, n_jobs, 'parametric', random_state,
                dist_type=dist_type, dist_params=dist_params
            )
        ci = percentile_ci(boot_stats, alpha)
    elif method == 'block':
        if n_jobs == 1:
            boot_samples = block_resample(data, B, block_size or max(2, int(np.sqrt(len(data)))), block_method)
            boot_stats = np.array([stat_func(sample) for sample in boot_samples])
        else:
            bs = block_size or max(2, int(np.sqrt(len(data))))
            boot_stats = parallel_bootstrap_stats(
                data, stat_func, B, n_jobs, 'block', random_state,
                block_size=bs, block_method=block_method
            )
        ci = percentile_ci(boot_stats, alpha)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Use 'percentile', 'bca', 'studentized', 'double', 'parametric', or 'block'."
        )

    return ci[0], ci[1], original_stat, boot_stats


if __name__ == "__main__":
    np.random.seed(42)
    import time

    print("=" * 75)
    print("Bootstrap 置信区间 - 完整功能测试")
    print("=" * 75)
    print()

    print("=" * 75)
    print("【测试1】小样本基础方法对比 (n=7)")
    print("=" * 75)
    print()

    n_small = 7
    true_mean = 10
    true_std = 2
    data_small = np.random.normal(loc=true_mean, scale=true_std, size=n_small)

    print(f"小样本数据 (n={n_small}):")
    print(f"  数据: {np.round(data_small, 2)}")
    print(f"  样本均值: {np.mean(data_small):.4f}")
    print(f"  真实均值: {true_mean}")
    print(f"  理论标准误: {true_std / np.sqrt(n_small):.4f}")
    print()

    B = 500
    B_inner = 30
    B2 = 50
    alpha = 0.05

    basic_methods = ['percentile', 'bca', 'studentized', 'double']
    basic_method_names = {
        'percentile': '百分位数法',
        'bca': 'BCa 法',
        'studentized': '学生化自助法',
        'double': '双重自助法'
    }

    results = {}
    for method in basic_methods:
        lower, upper, orig_stat, boot_stats = bootstrap_ci(
            data_small, np.mean, B=B, alpha=alpha, method=method,
            random_state=42, B_inner=B_inner, B2=B2
        )
        results[method] = {
            'lower': lower,
            'upper': upper,
            'width': upper - lower,
            'covers': lower <= true_mean <= upper
        }

    print("-" * 75)
    print(f"均值的 {(1-alpha)*100:.0f}% 置信区间对比:")
    print("-" * 75)
    print(f"{'方法':<15} {'下限':>10} {'上限':>10} {'宽度':>10} {'覆盖真值':>10}")
    print("-" * 75)
    for method in basic_methods:
        r = results[method]
        covers_str = "是" if r['covers'] else "否"
        print(f"{basic_method_names[method]:<15} {r['lower']:>10.4f} {r['upper']:>10.4f} "
              f"{r['width']:>10.4f} {covers_str:>10}")
    print("-" * 75)
    print()

    print("=" * 75)
    print("【测试2】参数自助法 (Parametric Bootstrap)")
    print("=" * 75)
    print()

    print(f"测试不同分布假设下的参数自助法:")
    print()

    dist_types = ['norm', 'gamma', 't']
    dist_names = {
        'norm': '正态分布',
        'gamma': '伽马分布',
        't': 't分布'
    }

    for dist_type in dist_types:
        try:
            lower, upper, orig_stat, boot_stats = bootstrap_ci(
                data_small, np.mean, B=B, alpha=alpha, method='parametric',
                random_state=42, dist_type=dist_type
            )
            fitted_params = fit_distribution(data_small, dist_type)
            param_str = ", ".join([f"{k}={v:.3f}" for k, v in fitted_params.items() if k != 'dist'])
            covers = lower <= true_mean <= upper
            covers_str = "是" if covers else "否"
            print(f"  {dist_names[dist_type]:<8} (拟合: {param_str[:30]})")
            print(f"    95% CI: [{lower:.4f}, {upper:.4f}] 宽度: {upper-lower:.4f} 覆盖: {covers_str}")
        except Exception as e:
            print(f"  {dist_names[dist_type]:<8}: 跳过 - {str(e)[:30]}")
    print()

    print("=" * 75)
    print("【测试3】块自助法 (Block Bootstrap) - 时间序列数据")
    print("=" * 75)
    print()

    n_ts = 100
    ar1_param = 0.7
    true_mean_ts = 5
    np.random.seed(42)

    ts_data = np.zeros(n_ts)
    ts_data[0] = true_mean_ts + np.random.normal(0, 1)
    for t in range(1, n_ts):
        ts_data[t] = true_mean_ts + ar1_param * (ts_data[t-1] - true_mean_ts) + np.random.normal(0, 1)

    print(f"AR(1) 时间序列 (n={n_ts}, φ={ar1_param}):")
    print(f"  样本均值: {np.mean(ts_data):.4f}")
    print(f"  真实均值: {true_mean_ts}")
    print(f"  自相关系数(lag1): {np.corrcoef(ts_data[:-1], ts_data[1:])[0,1]:.4f}")
    print()

    block_methods = ['fixed', 'moving', 'circular']
    block_sizes = [5, 10, 15]

    print("不同块大小和块方法的置信区间对比:")
    print("-" * 75)
    print(f"{'块方法':<10} {'块长':>6} {'下限':>10} {'上限':>10} {'宽度':>10} {'覆盖':>8}")
    print("-" * 75)

    lower_std, upper_std, _, _ = bootstrap_ci(
        ts_data, np.mean, B=B, alpha=alpha, method='percentile', random_state=42
    )
    covers_std = lower_std <= true_mean_ts <= upper_std
    covers_std_str = "是" if covers_std else "否"
    print(f"{'标准自助':<10} {'-':>6} {lower_std:>10.4f} {upper_std:>10.4f} "
          f"{upper_std-lower_std:>10.4f} {covers_std_str:>8}")

    for block_method in block_methods:
        for block_size in block_sizes:
            lower, upper, _, _ = bootstrap_ci(
                ts_data, np.mean, B=B, alpha=alpha, method='block',
                random_state=42, block_size=block_size, block_method=block_method
            )
            covers = lower <= true_mean_ts <= upper
            covers_str = "是" if covers else "否"
            print(f"{block_method:<10} {block_size:>6} {lower:>10.4f} {upper:>10.4f} "
                  f"{upper-lower:>10.4f} {covers_str:>8}")
    print("-" * 75)
    print()
    print("注: 块自助法能保持时间序列的自相关性，标准自助法会破坏这种结构")
    print()

    print("=" * 75)
    print("【测试4】并行化加速对比")
    print("=" * 75)
    print()

    B_large = 10000
    data_large_par = np.random.normal(loc=10, scale=2, size=500)

    print(f"数据量: {len(data_large_par)}, 重采样次数: {B_large}")
    print(f"CPU 核心数: {mp.cpu_count()}")
    print(f"注: 并行化在 B >= 5000 时自动启用")
    print()

    n_jobs_list = [1, 2, -1]

    print("-" * 75)
    print(f"{'并行数':>8} {'耗时(秒)':>12} {'加速比':>10} {'下限':>10} {'上限':>10}")
    print("-" * 75)

    times = {}
    cis = {}

    for n_jobs in n_jobs_list:
        start = time.time()
        lower, upper, orig_stat, _ = bootstrap_ci(
            data_large_par, np.mean, B=B_large, alpha=alpha, method='percentile',
            random_state=42, n_jobs=n_jobs
        )
        elapsed = time.time() - start
        times[n_jobs] = elapsed
        cis[n_jobs] = (lower, upper)

        n_jobs_display = mp.cpu_count() if n_jobs == -1 else n_jobs
        speedup = times[1] / elapsed if n_jobs != 1 else 1.0
        print(f"{n_jobs_display:>8} {elapsed:>12.3f} {speedup:>10.2f}x {lower:>10.4f} {upper:>10.4f}")

    print("-" * 75)
    print()
    if len(n_jobs_list) > 1 and times.get(1, 0) > 0:
        max_speedup = times[1] / min([t for t in times.values()])
        print(f"最大加速比: {max_speedup:.2f}x")
    print()

    print("=" * 75)
    print("【测试5】综合示例 - 不同统计量和方法")
    print("=" * 75)
    print()

    data_demo = np.random.normal(loc=10, scale=2, size=50)
    print(f"演示数据 (n=50, N(10, 4)):")
    print(f"  样本均值: {np.mean(data_demo):.4f}")
    print(f"  样本中位数: {np.median(data_demo):.4f}")
    print(f"  样本标准差: {np.std(data_demo, ddof=1):.4f}")
    print()

    stat_functions = [
        (np.mean, "均值"),
        (np.median, "中位数"),
        (lambda x: np.std(x, ddof=1), "标准差")
    ]

    all_methods = ['percentile', 'bca', 'parametric', 'studentized']
    all_method_names = {
        'percentile': '百分位数',
        'bca': 'BCa',
        'parametric': '参数',
        'studentized': '学生化'
    }

    print("不同统计量和方法的 95% 置信区间:")
    print("-" * 75)
    print(f"{'统计量':<10} {'方法':<12} {'下限':>10} {'上限':>10} {'宽度':>10}")
    print("-" * 75)

    for stat_func, stat_name in stat_functions:
        for method in all_methods:
            try:
                lower, upper, orig_stat, _ = bootstrap_ci(
                    data_demo, stat_func, B=1000, alpha=alpha, method=method,
                    random_state=42, dist_type='norm'
                )
                print(f"{stat_name:<10} {all_method_names[method]:<12} "
                      f"{lower:>10.4f} {upper:>10.4f} {upper-lower:>10.4f}")
            except Exception:
                print(f"{stat_name:<10} {all_method_names[method]:<12} {'-':>10} {'-':>10} {'-':>10}")
        print("-" * 75)

    print("=" * 75)
    print("测试完成！")
    print("=" * 75)
