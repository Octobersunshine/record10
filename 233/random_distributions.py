import numpy as np
from scipy import stats


_DISTRIBUTION_OFFSETS = {
    'uniform': 1001,
    'normal': 1002,
    'exponential': 1003,
    'poisson': 1004,
    'binomial': 1005,
    'gamma': 1006,
    'beta': 1007,
}

_DISTRIBUTION_TYPES = {
    'uniform': 'continuous',
    'normal': 'continuous',
    'exponential': 'continuous',
    'poisson': 'discrete',
    'binomial': 'discrete',
    'gamma': 'continuous',
    'beta': 'continuous',
}


def _get_scipy_distribution(dist_type, **params):
    dist_type = dist_type.lower()

    if dist_type == 'uniform':
        low = params.get('low', 0.0)
        high = params.get('high', 1.0)
        return stats.uniform(loc=low, scale=high - low)

    elif dist_type == 'normal':
        loc = params.get('loc', 0.0)
        scale = params.get('scale', 1.0)
        return stats.norm(loc=loc, scale=scale)

    elif dist_type == 'exponential':
        scale = params.get('scale', 1.0)
        return stats.expon(scale=scale)

    elif dist_type == 'poisson':
        lam = params.get('lam', 1.0)
        return stats.poisson(mu=lam)

    elif dist_type == 'binomial':
        n = params.get('n', 10)
        p = params.get('p', 0.5)
        return stats.binom(n=n, p=p)

    elif dist_type == 'gamma':
        shape = params.get('shape', 2.0)
        scale = params.get('scale', 1.0)
        return stats.gamma(a=shape, scale=scale)

    elif dist_type == 'beta':
        a = params.get('a', 2.0)
        b = params.get('b', 2.0)
        return stats.beta(a=a, b=b)

    else:
        raise ValueError(
            f"不支持的分布类型: {dist_type}。"
            f"支持的类型: {', '.join(_DISTRIBUTION_OFFSETS.keys())}"
        )


def generate_random_distribution(dist_type, sample_size, seed=None, **params):
    """
    生成常见分布的随机数

    参数:
        dist_type (str): 分布类型
        sample_size (int): 样本量
        seed (int, optional): 随机种子，用于结果可复现。相同seed下不同分布使用独立随机流。
        **params: 分布参数

    返回:
        np.ndarray: 随机数组

    分布参数说明:
        - uniform (均匀分布): low=最小值, high=最大值 (默认 low=0, high=1)
        - normal (正态分布): loc=均值, scale=标准差 (默认 loc=0, scale=1)
        - exponential (指数分布): scale=1/λ (默认 scale=1, 即 λ=1)
        - poisson (泊松分布): lam=λ (默认 lam=1)
        - binomial (二项分布): n=试验次数, p=成功概率 (默认 n=10, p=0.5)
        - gamma (伽马分布): shape=形状参数α, scale=尺度参数θ (默认 shape=2, scale=1)
        - beta (贝塔分布): a=α, b=β (默认 a=2, b=2)
    """
    dist_type = dist_type.lower()

    if dist_type not in _DISTRIBUTION_OFFSETS:
        raise ValueError(
            f"不支持的分布类型: {dist_type}。"
            f"支持的类型: {', '.join(_DISTRIBUTION_OFFSETS.keys())}"
        )

    if seed is not None:
        base_seed = int(seed)
        offset = _DISTRIBUTION_OFFSETS[dist_type]
        seed_seq = np.random.SeedSequence(base_seed).spawn(offset + 1)[offset]
        rng = np.random.default_rng(seed_seq)
    else:
        rng = np.random.default_rng()

    if dist_type == 'uniform':
        low = params.get('low', 0.0)
        high = params.get('high', 1.0)
        return rng.uniform(low=low, high=high, size=sample_size)

    elif dist_type == 'normal':
        loc = params.get('loc', 0.0)
        scale = params.get('scale', 1.0)
        return rng.normal(loc=loc, scale=scale, size=sample_size)

    elif dist_type == 'exponential':
        scale = params.get('scale', 1.0)
        return rng.exponential(scale=scale, size=sample_size)

    elif dist_type == 'poisson':
        lam = params.get('lam', 1.0)
        return rng.poisson(lam=lam, size=sample_size)

    elif dist_type == 'binomial':
        n = params.get('n', 10)
        p = params.get('p', 0.5)
        return rng.binomial(n=n, p=p, size=sample_size)

    elif dist_type == 'gamma':
        shape = params.get('shape', 2.0)
        scale = params.get('scale', 1.0)
        return rng.gamma(shape=shape, scale=scale, size=sample_size)

    elif dist_type == 'beta':
        a = params.get('a', 2.0)
        b = params.get('b', 2.0)
        return rng.beta(a=a, b=b, size=sample_size)


def calculate_pdf(dist_type, x, **params):
    """
    计算概率密度函数 (PDF) 或 概率质量函数 (PMF)

    参数:
        dist_type (str): 分布类型
        x (float or array_like): 计算点
        **params: 分布参数

    返回:
        float or np.ndarray: PDF/PMF值
    """
    dist = _get_scipy_distribution(dist_type, **params)
    dist_type_lower = dist_type.lower()

    if _DISTRIBUTION_TYPES[dist_type_lower] == 'discrete':
        x = np.asarray(x, dtype=int)
        return dist.pmf(x)
    else:
        return dist.pdf(x)


def calculate_cdf(dist_type, x, **params):
    """
    计算累积分布函数 (CDF)

    参数:
        dist_type (str): 分布类型
        x (float or array_like): 计算点
        **params: 分布参数

    返回:
        float or np.ndarray: CDF值
    """
    dist = _get_scipy_distribution(dist_type, **params)
    return dist.cdf(x)


if __name__ == '__main__':
    TEST_SEED = 42

    print("=" * 60)
    print("验证1: 新增分布随机数生成 (二项、伽马、贝塔)")
    print("=" * 60)
    binom_data = generate_random_distribution('binomial', 8, seed=TEST_SEED, n=10, p=0.5)
    gamma_data = generate_random_distribution('gamma', 5, seed=TEST_SEED, shape=2, scale=1)
    beta_data = generate_random_distribution('beta', 5, seed=TEST_SEED, a=2, b=5)
    print(f"二项分布 B(n=10, p=0.5): {binom_data}")
    print(f"  样本均值: {binom_data.mean():.4f}, 理论均值: {10 * 0.5:.1f}")
    print(f"伽马分布 Γ(α=2, θ=1):     {gamma_data}")
    print(f"  样本均值: {gamma_data.mean():.4f}, 理论均值: {2 * 1:.1f}")
    print(f"贝塔分布 Beta(α=2, β=5):  {beta_data}")
    print(f"  样本均值: {beta_data.mean():.4f}, 理论均值: {2 / (2 + 5):.4f}\n")

    print("=" * 60)
    print("验证2: PDF/PMF 计算")
    print("=" * 60)
    x_vals = np.array([-2, -1, 0, 1, 2])
    norm_pdf = calculate_pdf('normal', x_vals, loc=0, scale=1)
    for xi, pi in zip(x_vals, norm_pdf):
        print(f"  正态分布 N(0,1) PDF({xi:>2}) = {pi:.6f}")

    k_vals = np.arange(0, 6)
    poisson_pmf = calculate_pdf('poisson', k_vals, lam=2)
    for ki, pi in zip(k_vals, poisson_pmf):
        print(f"  泊松分布 P(λ=2) PMF({ki:>2}) = {pi:.6f}")

    binom_pmf = calculate_pdf('binomial', k_vals, n=10, p=0.3)
    for ki, pi in zip(k_vals, binom_pmf):
        print(f"  二项分布 B(10,0.3) PMF({ki:>2}) = {pi:.6f}")

    beta_pdf = calculate_pdf('beta', [0.25, 0.5, 0.75], a=2, b=5)
    for xi, pi in zip([0.25, 0.5, 0.75], beta_pdf):
        print(f"  贝塔分布 B(2,5) PDF({xi:.2f}) = {pi:.6f}")
    print()

    print("=" * 60)
    print("验证3: CDF 计算")
    print("=" * 60)
    cdf_x = np.array([-1, 0, 1, 2])
    norm_cdf = calculate_cdf('normal', cdf_x, loc=0, scale=1)
    for xi, ci in zip(cdf_x, norm_cdf):
        print(f"  正态分布 N(0,1) CDF({xi:>2}) = {ci:.6f}")

    exp_cdf = calculate_cdf('exponential', [0.5, 1, 2], scale=1)
    for xi, ci in zip([0.5, 1, 2], exp_cdf):
        print(f"  指数分布 Exp(λ=1) CDF({xi:>2}) = {ci:.6f}")

    binom_cdf = calculate_cdf('binomial', [3, 5, 7], n=10, p=0.5)
    for xi, ci in zip([3, 5, 7], binom_cdf):
        print(f"  二项分布 B(10,0.5) CDF({xi:>2}) = {ci:.6f}")

    beta_cdf = calculate_cdf('beta', [0.3, 0.5, 0.7], a=2, b=2)
    for xi, ci in zip([0.3, 0.5, 0.7], beta_cdf):
        print(f"  贝塔分布 B(2,2) CDF({xi:.1f}) = {ci:.6f}")
    print()

    print("=" * 60)
    print("验证4: CDF 数值积分验证 (对比解析解)")
    print("=" * 60)
    from scipy.integrate import quad

    norm_dist = _get_scipy_distribution('normal', loc=0, scale=1)
    x_point = 1.5
    cdf_analytical = calculate_cdf('normal', x_point, loc=0, scale=1)
    cdf_integral, _ = quad(norm_dist.pdf, -np.inf, x_point)
    print(f"  正态分布 CDF(1.5) 解析值: {cdf_analytical:.8f}")
    print(f"  正态分布 CDF(1.5) 积分值: {cdf_integral:.8f}")
    print(f"  两者一致: {np.isclose(cdf_analytical, cdf_integral)}")

    gamma_dist = _get_scipy_distribution('gamma', shape=2, scale=1)
    x_gamma = 3.0
    cdf_analytical_g = calculate_cdf('gamma', x_gamma, shape=2, scale=1)
    cdf_integral_g, _ = quad(gamma_dist.pdf, 0, x_gamma)
    print(f"\n  伽马分布 CDF(3.0) 解析值: {cdf_analytical_g:.8f}")
    print(f"  伽马分布 CDF(3.0) 积分值: {cdf_integral_g:.8f}")
    print(f"  两者一致: {np.isclose(cdf_analytical_g, cdf_integral_g)}")
    print()

    print("=" * 60)
    print("验证5: 相同seed + 不同分布独立性验证 (新增分布)")
    print("=" * 60)
    norm_b = generate_random_distribution('normal', 5, seed=TEST_SEED, loc=0, scale=1)
    binom_b = generate_random_distribution('binomial', 5, seed=TEST_SEED, n=10, p=0.5)
    gamma_b = generate_random_distribution('gamma', 5, seed=TEST_SEED, shape=2, scale=1)
    beta_b = generate_random_distribution('beta', 5, seed=TEST_SEED, a=2, b=2)
    print(f"  正态: {norm_b}")
    print(f"  二项: {binom_b}")
    print(f"  伽马: {gamma_b}")
    print(f"  贝塔: {beta_b}")
    print(f"  正态≠二项: {not np.allclose(norm_b, binom_b.astype(float))}")
    print(f"  伽马≠贝塔: {not np.allclose(gamma_b, beta_b)}")
    print()

    print("=" * 60)
    print("验证6: 大样本统计特性验证 (样本量=100000)")
    print("=" * 60)
    big_binom = generate_random_distribution('binomial', 100000, seed=TEST_SEED, n=20, p=0.4)
    print(f"二项分布 B(20, 0.4): 实际均值={big_binom.mean():.4f} (理论 {20 * 0.4:.1f})")

    big_gamma = generate_random_distribution('gamma', 100000, seed=TEST_SEED, shape=3, scale=2)
    print(f"伽马分布 Γ(3, 2): 实际均值={big_gamma.mean():.4f} (理论 {3 * 2:.1f})")

    big_beta = generate_random_distribution('beta', 100000, seed=TEST_SEED, a=5, b=2)
    print(f"贝塔分布 Beta(5, 2): 实际均值={big_beta.mean():.4f} (理论 {5 / (5 + 2):.4f})")
