import math
import random
from scipy import stats
from scipy.special import comb
import numpy as np


def _runs_exact_pmf(n1, n2, k):
    """
    计算游程数为k的精确概率质量函数
    n1: 正值个数, n2: 负值个数, k: 游程数
    """
    if k < 2 or k > n1 + n2:
        return 0.0
    
    total = comb(n1 + n2, n1)
    
    if k % 2 == 0:
        m = k // 2
        prob = 2 * comb(n1 - 1, m - 1) * comb(n2 - 1, m - 1) / total
    else:
        m = (k - 1) // 2
        prob = (comb(n1 - 1, m) * comb(n2 - 1, m - 1) + comb(n1 - 1, m - 1) * comb(n2 - 1, m)) / total
    
    return prob


def _runs_exact_pvalue(n1, n2, observed_runs):
    """
    计算游程测试的精确双侧p值
    
    双侧p值 = 2 * min(左尾概率, 右尾概率)
    左尾概率: P(R <= observed_runs)
    右尾概率: P(R >= observed_runs)
    """
    min_runs = 2
    max_runs = n1 + n2
    
    p_le = 0.0
    for k in range(min_runs, observed_runs + 1):
        p_le += _runs_exact_pmf(n1, n2, k)
    
    p_ge = 0.0
    for k in range(observed_runs, max_runs + 1):
        p_ge += _runs_exact_pmf(n1, n2, k)
    
    p_value = min(1.0, 2 * min(p_le, p_ge))
    
    return p_value


def _runs_monte_carlo_pvalue(n1, n2, observed_runs, n_simulations=10000, seed=None):
    """
    使用蒙特卡洛模拟计算游程测试的p值
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    sequence = [1] * n1 + [-1] * n2
    
    extreme_count = 0
    mean_runs = (2 * n1 * n2) / (n1 + n2) + 1
    
    obs_deviation = abs(observed_runs - mean_runs)
    
    for _ in range(n_simulations):
        rng.shuffle(sequence)
        
        sim_runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i - 1]:
                sim_runs += 1
        
        sim_deviation = abs(sim_runs - mean_runs)
        if sim_deviation >= obs_deviation:
            extreme_count += 1
    
    p_value = (extreme_count + 1) / (n_simulations + 1)
    
    return p_value


def frequency_test(sequence, num_bins=10):
    """
    频数测试（卡方检验）
    检验随机数在各个区间的分布是否均匀
    """
    n = len(sequence)
    min_val, max_val = min(sequence), max(sequence)
    
    observed = [0] * num_bins
    for x in sequence:
        bin_idx = int((x - min_val) / (max_val - min_val + 1e-10) * num_bins)
        bin_idx = min(bin_idx, num_bins - 1)
        observed[bin_idx] += 1
    
    expected = n / num_bins
    
    chi_square = sum((obs - expected) ** 2 / expected for obs in observed)
    p_value = 1 - stats.chi2.cdf(chi_square, num_bins - 1)
    
    return {
        'test_name': '频数测试 (卡方检验)',
        'statistic': chi_square,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'df': num_bins - 1,
        'observed': observed,
        'expected': expected
    }


def runs_test(sequence, method='auto', n_simulations=10000, seed=42):
    """
    游程测试
    检验序列中连续高于/低于中位数的游程数量是否合理
    
    参数:
        sequence: 待测试的随机数序列
        method: 计算p值的方法
            'auto': 自动选择 (n<20用精确分布, 20<=n<50用蒙特卡洛, n>=50用正态近似)
            'exact': 使用精确分布计算
            'monte_carlo': 使用蒙特卡洛模拟
            'normal': 使用正态近似
        n_simulations: 蒙特卡洛模拟次数
        seed: 蒙特卡洛模拟的随机种子
    """
    n = len(sequence)
    median = np.median(sequence)
    
    signs = []
    for x in sequence:
        if x > median:
            signs.append(1)
        elif x < median:
            signs.append(-1)
    
    n_pos = sum(1 for s in signs if s == 1)
    n_neg = sum(1 for s in signs if s == -1)
    n_effective = n_pos + n_neg
    
    num_runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            num_runs += 1
    
    n1, n2 = n_pos, n_neg
    
    if n1 == 0 or n2 == 0:
        return {
            'test_name': '游程测试',
            'statistic': float('inf'),
            'p_value': 0.0,
            'passed': False,
            'num_runs': num_runs,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'method': 'exact'
        }
    
    if method == 'auto':
        if n_effective < 20:
            method = 'exact'
        elif n_effective < 50:
            method = 'monte_carlo'
        else:
            method = 'normal'
    
    mean_runs = (2 * n1 * n2) / (n1 + n2) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    
    if var_runs == 0:
        z = 0
    else:
        z = (num_runs - mean_runs) / math.sqrt(var_runs)
    
    if method == 'exact':
        try:
            p_value = _runs_exact_pvalue(n1, n2, num_runs)
        except (OverflowError, ValueError):
            p_value = _runs_monte_carlo_pvalue(n1, n2, num_runs, n_simulations, seed)
            method = 'monte_carlo (fallback)'
    elif method == 'monte_carlo':
        p_value = _runs_monte_carlo_pvalue(n1, n2, num_runs, n_simulations, seed)
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'test_name': '游程测试',
        'statistic': z,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'num_runs': num_runs,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'mean_runs': mean_runs,
        'method': method,
        'n_effective': n_effective
    }


def serial_correlation_test(sequence, lag=1):
    """
    序列相关性测试
    检验序列与其滞后版本的相关系数
    """
    n = len(sequence)
    if n <= lag:
        return {
            'test_name': f'序列相关性测试 (lag={lag})',
            'statistic': 0.0,
            'p_value': 1.0,
            'passed': True,
            'correlation': 0.0,
            'lag': lag
        }
    
    mean = sum(sequence) / n
    
    numerator = sum((sequence[i] - mean) * (sequence[i + lag] - mean) for i in range(n - lag))
    denominator = sum((x - mean) ** 2 for x in sequence)
    
    if denominator == 0:
        correlation = 0.0
    else:
        correlation = numerator / denominator
    
    z = correlation * math.sqrt(n - lag - 1) / math.sqrt(1 - correlation ** 2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'test_name': f'序列相关性测试 (lag={lag})',
        'statistic': z,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'correlation': correlation,
        'lag': lag
    }


def run_all_tests(sequence, num_bins=10, lags=[1, 2, 3]):
    """
    运行所有随机数质量测试
    """
    results = []
    
    results.append(frequency_test(sequence, num_bins))
    results.append(runs_test(sequence))
    
    for lag in lags:
        results.append(serial_correlation_test(sequence, lag))
    
    return results


def print_results(results):
    """
    打印测试结果
    """
    print("=" * 70)
    print("随机数生成器质量测试报告")
    print("=" * 70)
    print(f"{'测试项目':<30} {'统计量':<15} {'p值':<15} {'是否通过':<10}")
    print("-" * 70)
    
    for res in results:
        status = "通过 [OK]" if res['passed'] else "未通过 [FAIL]"
        print(f"{res['test_name']:<30} {res['statistic']:<15.6f} {res['p_value']:<15.6f} {status:<10}")
    
    print("=" * 70)
    print("注: p值 > 0.05 表示通过测试 (95%置信水平)")
    print("=" * 70)
    
    for res in results:
        print(f"\n{res['test_name']} 详细信息:")
        if 'observed' in res:
            print(f"  自由度: {res['df']}")
            print(f"  观测频数: {res['observed']}")
            print(f"  期望频数: {res['expected']:.2f}")
        if 'num_runs' in res:
            print(f"  游程数: {res['num_runs']}")
            print(f"  高于中位数: {res['n_pos']}")
            print(f"  低于中位数: {res['n_neg']}")
            print(f"  期望游程数: {res['mean_runs']:.2f}")
            if 'method' in res:
                print(f"  计算方法: {res['method']}")
            if 'n_effective' in res:
                print(f"  有效样本量: {res['n_effective']}")
        if 'correlation' in res:
            print(f"  滞后阶数: {res['lag']}")
            print(f"  相关系数: {res['correlation']:.6f}")


def compare_runs_methods(sequence, description):
    """
    比较游程测试不同方法的p值
    """
    print(f"\n{description}")
    print(f"序列长度: {len(sequence)}")
    print("-" * 70)
    print(f"{'方法':<25} {'p值':<15} {'是否通过':<10}")
    print("-" * 70)
    
    methods = ['exact', 'monte_carlo', 'normal']
    for method in methods:
        res = runs_test(sequence, method=method, n_simulations=10000, seed=42)
        status = "通过 [OK]" if res['passed'] else "未通过 [FAIL]"
        print(f"{method:<25} {res['p_value']:<15.6f} {status:<10}")
    
    print("-" * 70)


class Xorshift128Plus:
    """
    Xorshift128+ 随机数生成器
    """
    def __init__(self, seed=42):
        self.state = [seed & 0xFFFFFFFFFFFFFFFF, (seed + 1) & 0xFFFFFFFFFFFFFFFF]
    
    def random(self):
        s1 = self.state[0]
        s0 = self.state[1]
        result = (s0 + s1) & 0xFFFFFFFFFFFFFFFF
        
        self.state[0] = s0
        s1 ^= (s1 << 23) & 0xFFFFFFFFFFFFFFFF
        self.state[1] = (s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5)) & 0xFFFFFFFFFFFFFFFF
        
        return result / 0xFFFFFFFFFFFFFFFF
    
    def generate_n(self, n):
        return [self.random() for _ in range(n)]


class PCG32:
    """
    PCG32 随机数生成器
    """
    def __init__(self, seed=42):
        self.state = 0
        self.inc = (seed << 1) | 1
        self.state = (self.state * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        self.state = (self.state + seed) & 0xFFFFFFFFFFFFFFFF
    
    def random(self):
        oldstate = self.state
        self.state = (oldstate * 6364136223846793005 + self.inc) & 0xFFFFFFFFFFFFFFFF
        xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
        rot = oldstate >> 59
        result = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF
        return result / 0xFFFFFFFF
    
    def generate_n(self, n):
        return [self.random() for _ in range(n)]


class MT19937:
    """
    MT19937 梅森旋转随机数生成器 (简化实现)
    """
    def __init__(self, seed=42):
        self.w, self.n, self.m, self.r = 32, 624, 397, 31
        self.a = 0x9908B0DF
        self.u, self.d = 11, 0xFFFFFFFF
        self.s, self.b = 7, 0x9D2C5680
        self.t, self.c = 15, 0xEFC60000
        self.l = 18
        self.f = 1812433253
        
        self.MT = [0] * self.n
        self.index = self.n
        self.lower_mask = (1 << self.r) - 1
        self.upper_mask = 0xFFFFFFFF & (~self.lower_mask)
        
        self.MT[0] = seed
        for i in range(1, self.n):
            self.MT[i] = (self.f * (self.MT[i-1] ^ (self.MT[i-1] >> (self.w-2))) + i) & 0xFFFFFFFF
    
    def _twist(self):
        for i in range(self.n):
            x = (self.MT[i] & self.upper_mask) + (self.MT[(i+1) % self.n] & self.lower_mask)
            xA = x >> 1
            if x % 2 != 0:
                xA = xA ^ self.a
            self.MT[i] = self.MT[(i + self.m) % self.n] ^ xA
        self.index = 0
    
    def random(self):
        if self.index >= self.n:
            self._twist()
        
        y = self.MT[self.index]
        y = y ^ ((y >> self.u) & self.d)
        y = y ^ ((y << self.s) & self.b)
        y = y ^ ((y << self.t) & self.c)
        y = y ^ (y >> self.l)
        
        self.index += 1
        return (y & 0xFFFFFFFF) / 0xFFFFFFFF
    
    def generate_n(self, n):
        return [self.random() for _ in range(n)]


def birthday_spacings_test(sequence, n_samples=256, n_bits=24):
    """
    生日悖论测试 (Dieharder测试子集)
    检验随机数生成的间距分布是否均匀
    选择较小的位宽使得期望碰撞数在合理范围内
    """
    n = len(sequence)
    
    int_sequence = [int(x * (1 << n_bits)) for x in sequence]
    int_sequence = sorted(int_sequence[:n_samples])
    
    spacings = []
    for i in range(1, len(int_sequence)):
        spacings.append(int_sequence[i] - int_sequence[i-1])
    
    spacing_counts = {}
    for s in spacings:
        if s in spacing_counts:
            spacing_counts[s] += 1
        else:
            spacing_counts[s] = 1
    
    collisions = sum(count - 1 for count in spacing_counts.values())
    
    lambda_exp = n_samples * (n_samples - 1) / (2 * (1 << n_bits))
    
    p_value = 1.0 - stats.poisson.cdf(collisions, lambda_exp)
    
    statistic = collisions
    
    return {
        'test_name': '生日悖论测试',
        'statistic': statistic,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'expected_lambda': lambda_exp,
        'collisions': collisions,
        'n_samples': n_samples,
        'n_bits': n_bits
    }


def overlapping_permutations_test(sequence, n_tuples=5):
    """
    重叠排列测试 (Dieharder测试子集)
    检验连续5个元素的排列分布是否均匀
    """
    n = len(sequence)
    
    perm_counts = {}
    total_perms = 0
    
    for i in range(n - n_tuples + 1):
        window = sequence[i:i + n_tuples]
        rank = sorted(range(n_tuples), key=lambda k: window[k])
        rank_tuple = tuple(rank)
        
        if rank_tuple in perm_counts:
            perm_counts[rank_tuple] += 1
        else:
            perm_counts[rank_tuple] = 1
        total_perms += 1
    
    n_perms = math.factorial(n_tuples)
    expected = total_perms / n_perms
    
    observed = list(perm_counts.values())
    while len(observed) < n_perms:
        observed.append(0)
    
    chi_square = sum((obs - expected) ** 2 / expected for obs in observed)
    p_value = 1 - stats.chi2.cdf(chi_square, n_perms - 1)
    
    return {
        'test_name': f'重叠排列测试 ({n_tuples}-tuple)',
        'statistic': chi_square,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'df': n_perms - 1,
        'n_permutations': n_perms
    }


def spectral_test(sequence, n_fft_points=None):
    """
    频谱测试 (FFT峰值测试)
    检验随机序列的频域特性 - 基于NIST SP 800-22标准
    
    对于白噪声，FFT变换后的实部和虚部服从正态分布，
    因此幅度的平方服从指数分布
    """
    n = len(sequence)
    
    if n_fft_points is None:
        n_fft_points = min(n, 10000)
    
    data = np.array(sequence[:n_fft_points]) - 0.5
    
    fft_result = np.fft.fft(data)
    
    half_len = n_fft_points // 2
    magnitudes = np.abs(fft_result[1:half_len])
    
    mag_squared = magnitudes ** 2
    
    T = math.sqrt(-2 * math.log(0.05 / half_len)) * math.sqrt(n_fft_points / 4)
    N1 = half_len * 0.95
    N_above = np.sum(mag_squared > n_fft_points / 4)
    
    d = (N_above - N1) / math.sqrt(half_len * 0.95 * 0.05)
    p_value_peak = 2 * (1 - stats.norm.cdf(abs(d)))
    
    n_bins = 10
    max_mag_sq = np.max(mag_squared)
    bins = np.linspace(0, max_mag_sq * 1.1, n_bins + 1)
    
    observed, _ = np.histogram(mag_squared, bins=bins)
    
    expected = []
    for i in range(n_bins):
        prob_low = 1 - math.exp(-bins[i] * 4 / n_fft_points)
        prob_high = 1 - math.exp(-bins[i+1] * 4 / n_fft_points)
        expected.append((prob_high - prob_low) * len(mag_squared))
    
    expected = np.array(expected)
    mask = expected >= 5
    chi_square = 0.0
    if np.sum(mask) > 1:
        observed = observed[mask]
        expected = expected[mask]
        chi_square = np.sum((observed - expected) ** 2 / expected)
        p_value_chi = 1 - stats.chi2.cdf(chi_square, len(observed) - 1)
    else:
        p_value_chi = 0.5
    
    max_peak = np.max(magnitudes)
    
    return {
        'test_name': '频谱测试 (FFT)',
        'statistic': max_peak,
        'p_value': p_value_peak,
        'passed': p_value_peak > 0.05 and p_value_chi > 0.05,
        'max_peak': max_peak,
        'chi_square': chi_square,
        'p_value_chi': p_value_chi,
        'N_above': N_above,
        'N1_expected': N1,
        'd_statistic': d
    }


def run_all_advanced_tests(sequence, n_samples=512, n_tuples=5):
    """
    运行所有高级测试
    """
    results = []
    
    results.append(birthday_spacings_test(sequence, n_samples))
    results.append(overlapping_permutations_test(sequence, n_tuples))
    results.append(spectral_test(sequence))
    
    return results


def print_advanced_results(results):
    """
    打印高级测试结果
    """
    print("=" * 70)
    print("高级随机数质量测试报告 (Dieharder子集 + 频谱测试)")
    print("=" * 70)
    print(f"{'测试项目':<25} {'统计量':<15} {'p值':<15} {'是否通过':<10}")
    print("-" * 70)
    
    for res in results:
        status = "通过 [OK]" if res['passed'] else "未通过 [FAIL]"
        print(f"{res['test_name']:<25} {res['statistic']:<15.6f} {res['p_value']:<15.6f} {status:<10}")
    
    print("=" * 70)
    
    for res in results:
        print(f"\n{res['test_name']} 详细信息:")
        if 'collisions' in res:
            print(f"  碰撞数: {res['collisions']}")
            print(f"  期望lambda: {res['expected_lambda']:.6f}")
            if 'n_samples' in res:
                print(f"  样本数: {res['n_samples']}")
                print(f"  位宽: {res['n_bits']} bits")
        if 'n_permutations' in res:
            print(f"  排列数: {res['n_permutations']}")
            print(f"  自由度: {res['df']}")
        if 'max_peak' in res:
            print(f"  最大峰值: {res['max_peak']:.4f}")
            print(f"  卡方统计量: {res['chi_square']:.4f}")
            print(f"  分布p值: {res['p_value_chi']:.6f}")
            if 'N_above' in res:
                print(f"  超出阈值的数量: {res['N_above']}")
                print(f"  期望数量: {res['N1_expected']:.1f}")
                print(f"  D统计量: {res['d_statistic']:.4f}")


def compare_rngs(seed=42, n=10000):
    """
    比较不同随机数生成器的测试结果
    """
    print("\n" * 2)
    print("=" * 70)
    print(f"随机数生成器质量对比测试 (n={n})")
    print("=" * 70)
    
    random.seed(seed)
    rngs = [
        ("Python random", [random.random() for _ in range(n)]),
        ("MT19937", MT19937(seed).generate_n(n)),
        ("PCG32", PCG32(seed).generate_n(n)),
        ("Xorshift128+", Xorshift128Plus(seed).generate_n(n)),
    ]
    
    test_names = ['频数测试', '游程测试', '生日悖论', '重叠排列', '频谱测试']
    
    print(f"\n{'生成器':<20} {'频数测试':<12} {'游程测试':<12} {'生日悖论':<12} {'重叠排列':<12} {'频谱测试':<12}")
    print("-" * 80)
    
    for name, seq in rngs:
        freq_res = frequency_test(seq)
        runs_res = runs_test(seq)
        bday_res = birthday_spacings_test(seq)
        perm_res = overlapping_permutations_test(seq)
        spec_res = spectral_test(seq)
        
        fmt = lambda r: "[OK]" if r['passed'] else "[F]"
        print(f"{name:<20} {fmt(freq_res):<12} {fmt(runs_res):<12} {fmt(bday_res):<12} {fmt(perm_res):<12} {fmt(spec_res):<12}")
    
    print("-" * 80)
    print(f"{'生成器':<20} {'频数测试':<12} {'游程测试':<12} {'生日悖论':<12} {'重叠排列':<12} {'频谱测试':<12}")
    print("-" * 80)
    
    for name, seq in rngs:
        freq_res = frequency_test(seq)
        runs_res = runs_test(seq)
        bday_res = birthday_spacings_test(seq)
        perm_res = overlapping_permutations_test(seq)
        spec_res = spectral_test(seq)
        
        fmt_p = lambda r: f"{r['p_value']:.4f}"
        print(f"{name:<20} {fmt_p(freq_res):<12} {fmt_p(runs_res):<12} {fmt_p(bday_res):<12} {fmt_p(perm_res):<12} {fmt_p(spec_res):<12}")
    
    print("=" * 80)
    
    return rngs


if __name__ == "__main__":
    print("测试1: 使用Python内置random生成的随机数 (期望通过所有测试)")
    print("-" * 70)
    random.seed(42)
    good_sequence = [random.random() for _ in range(1000)]
    results1 = run_all_tests(good_sequence)
    print_results(results1)
    
    print("\n" * 2)
    
    print("测试2: 使用非随机序列 (期望未通过测试)")
    print("-" * 70)
    bad_sequence = sorted([random.random() for _ in range(1000)])
    results2 = run_all_tests(bad_sequence)
    print_results(results2)
    
    print("\n" * 2)
    
    print("测试3: 交替序列 (期望未通过游程和相关性测试)")
    print("-" * 70)
    alt_sequence = []
    for i in range(1000):
        if i % 2 == 0:
            alt_sequence.append(0.1)
        else:
            alt_sequence.append(0.9)
    results3 = run_all_tests(alt_sequence)
    print_results(results3)
    
    print("\n" * 2)
    
    print("=" * 70)
    print("小样本测试: 游程测试不同方法对比")
    print("=" * 70)
    
    print("\n测试4: 小样本随机序列 (n=12)")
    print("-" * 70)
    random.seed(123)
    small_random = [random.random() for _ in range(12)]
    print(f"序列: {[round(x, 3) for x in small_random]}")
    compare_runs_methods(small_random, "小样本随机序列对比")
    
    print("\n测试5: 小样本排序序列 (n=12, 极端情况)")
    print("-" * 70)
    small_sorted = sorted([random.random() for _ in range(12)])
    print(f"序列: {[round(x, 3) for x in small_sorted]}")
    compare_runs_methods(small_sorted, "小样本排序序列对比")
    
    print("\n测试6: 极小样本交替序列 (n=10)")
    print("-" * 70)
    tiny_alt = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    print(f"序列: {tiny_alt}")
    compare_runs_methods(tiny_alt, "极小样本交替序列对比")
    
    print("\n测试7: 中等样本 (n=30, 使用蒙特卡洛)")
    print("-" * 70)
    medium_seq = [random.random() for _ in range(30)]
    auto_result = runs_test(medium_seq, method='auto')
    print(f"自动选择方法: {auto_result['method']}")
    print(f"有效样本量: {auto_result['n_effective']}")
    print(f"p值: {auto_result['p_value']:.6f}")
    
    print("\n" + "=" * 70)
    print("自动选择策略说明:")
    print("  n < 20  → 使用精确分布 (组合数学计算)")
    print("  20 ≤ n < 50 → 使用蒙特卡洛模拟 (10000次)")
    print("  n ≥ 50 → 使用正态近似")
    print("=" * 70)
    
    print("\n" * 2)
    
    print("=" * 70)
    print("高级测试: Dieharder子集 + 频谱测试")
    print("=" * 70)
    
    print("\n测试8: Python random 高级测试 (n=10000)")
    print("-" * 70)
    random.seed(42)
    adv_seq = [random.random() for _ in range(10000)]
    adv_results = run_all_advanced_tests(adv_seq)
    print_advanced_results(adv_results)
    
    print("\n" * 2)
    
    print("测试9: 排序序列高级测试 (n=10000, 期望未通过)")
    print("-" * 70)
    bad_adv_seq = sorted([random.random() for _ in range(10000)])
    bad_adv_results = run_all_advanced_tests(bad_adv_seq)
    print_advanced_results(bad_adv_results)
    
    print("\n" * 2)
    
    rng_list = compare_rngs(seed=42, n=10000)
