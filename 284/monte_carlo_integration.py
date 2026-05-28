import random
import math


class ImportanceSampler:
    """
    基于 |f(x)| 构造分段线性重要性采样分布，支持逆变换采样。
    
    原理：
      1. 在 [a,b] 上取 M 个等距节点，计算 |f(x_i)| 构造分片线性 PDF
      2. 对每个分段计算梯形面积得到归一化 CDF
      3. 采样时用逆变换法：生成 U~Uniform(0,1)，由 CDF 反函数得到采样点
      4. 同时提供 p(x) 供重要性权重计算
    """

    def __init__(self, f, a, b, num_bins=1000):
        self.a = a
        self.b = b
        self.num_bins = num_bins
        self.dx = (b - a) / num_bins

        self._nodes_x = []
        self._nodes_y = []
        for i in range(num_bins + 1):
            x = a + i * self.dx
            y = abs(f(x))
            y = max(y, 1e-15)
            self._nodes_x.append(x)
            self._nodes_y.append(y)

        self._segment_areas = []
        for i in range(num_bins):
            area = 0.5 * (self._nodes_y[i] + self._nodes_y[i + 1]) * self.dx
            self._segment_areas.append(area)

        self._total_area = sum(self._segment_areas)
        if self._total_area < 1e-30:
            self._total_area = 1.0
            self._segment_areas = [1.0 / num_bins] * num_bins

        self._cdf = [0.0]
        for area in self._segment_areas:
            self._cdf.append(self._cdf[-1] + area / self._total_area)
        self._cdf[-1] = 1.0

    def sample(self):
        u = random.random()
        idx = self._find_segment(u)
        x_left = self._nodes_x[idx]
        y_left = self._nodes_y[idx]
        y_right = self._nodes_y[idx + 1]
        seg_area = self._segment_areas[idx]
        seg_cdf_start = self._cdf[idx]
        frac = (u - seg_cdf_start) / (self._cdf[idx + 1] - seg_cdf_start) if (self._cdf[idx + 1] - seg_cdf_start) > 0 else 0.5

        if abs(y_left - y_right) < 1e-30:
            t = frac
        else:
            a_coeff = 0.5 * (y_right - y_left) / self.dx
            b_coeff = y_left
            c_coeff = -frac * seg_area / self.dx
            if abs(a_coeff) < 1e-30:
                t = frac
            else:
                disc = b_coeff ** 2 - 4 * a_coeff * c_coeff
                disc = max(disc, 0.0)
                sqrt_disc = math.sqrt(disc)
                t1 = (-b_coeff + sqrt_disc) / (2 * a_coeff)
                t2 = (-b_coeff - sqrt_disc) / (2 * a_coeff)
                t = t1 if 0.0 <= t1 <= 1.0 else t2
                t = max(0.0, min(1.0, t))

        return x_left + t * self.dx

    def pdf(self, x):
        if x < self.a or x > self.b:
            return 0.0
        idx = int((x - self.a) / self.dx)
        idx = min(idx, self.num_bins - 1)
        x_left = self._nodes_x[idx]
        y_left = self._nodes_y[idx]
        y_right = self._nodes_y[idx + 1]
        t = (x - x_left) / self.dx
        local_val = y_left + t * (y_right - y_left)
        return local_val / self._total_area

    def _find_segment(self, u):
        lo, hi = 0, len(self._cdf) - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cdf[mid + 1] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo


def monte_carlo_integral(f, a, b, N, importance_sampling=True, num_bins=1000):
    """
    一维蒙特卡洛积分（支持重要性采样）
    
    参数:
        f: 被积函数，如 lambda x: x**2
        a: 积分下限
        b: 积分上限
        N: 采样点数
        importance_sampling: 是否启用重要性采样（默认True）
        num_bins: 构造重要性采样PDF时的分片数（默认1000）
    
    返回:
        (积分估计值, 估计误差(标准差))
    """
    if N <= 0:
        raise ValueError("采样点数N必须大于0")

    if not importance_sampling:
        samples = []
        for _ in range(N):
            x = random.uniform(a, b)
            samples.append(f(x))
        mean = sum(samples) / N
        integral = (b - a) * mean
        variance = sum((y - mean) ** 2 for y in samples) / N
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        error = (b - a) * std_dev / math.sqrt(N)
        return integral, error

    sampler = ImportanceSampler(f, a, b, num_bins=num_bins)

    samples = []
    for _ in range(N):
        x = sampler.sample()
        p = sampler.pdf(x)
        if p > 1e-30:
            samples.append(f(x) / p)
        else:
            samples.append(0.0)

    mean = sum(samples) / N
    integral = mean
    variance = sum((y - mean) ** 2 for y in samples) / N
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    error = std_dev / math.sqrt(N)
    return integral, error


class HaltonSequence:
    """
    Halton 低差异序列生成器。
    
    每个维度使用不同的素数基，通过 radical inverse 构造。
    收敛速度 O(N^{-1}(log N)^d)，优于纯随机的 O(N^{-1/2})。
    """

    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
              109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173]

    def __init__(self, dim):
        if dim > len(self.PRIMES):
            raise ValueError(f"Halton序列最多支持 {len(self.PRIMES)} 维，当前请求 {dim} 维")
        self.dim = dim
        self.bases = self.PRIMES[:dim]
        self._index = 0

    def _radical_inverse(self, n, base):
        result = 0.0
        f = 1.0 / base
        i = n
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    def next_point(self):
        point = []
        for base in self.bases:
            point.append(self._radical_inverse(self._index, base))
        self._index += 1
        return point

    def generate(self, n, skip=0):
        self._index = skip
        points = []
        for _ in range(n):
            points.append(self.next_point())
        return points


class SobolSequence:
    """
    Sobol 低差异序列生成器。
    
    使用原始多项式和方向数递推构造，收敛速度优于 Halton（尤其是高维）。
    基于 Bratley & Fox (1988) 算法实现，支持最多 20 维。
    
    每个维度存储 (多项式度数 s, 多项式系数 a, 初始方向数 m[0..s-1])，
    通过递推公式生成完整的 32 位方向数表。
    """

    MAX_BITS = 32

    _DIM_DATA = [
        (0, 0, [1]),
        (1, 0, [1]),
        (2, 1, [1, 1]),
        (3, 1, [1, 3, 7]),
        (3, 2, [1, 3, 5]),
        (4, 1, [1, 1, 5]),
        (4, 4, [1, 3, 7]),
        (5, 2, [1, 7, 11, 13]),
        (5, 4, [1, 1, 7, 11]),
        (5, 7, [1, 9, 13, 15]),
        (5, 11, [1, 1, 3, 5]),
        (5, 13, [1, 3, 5, 15]),
        (5, 14, [1, 7, 11, 13]),
        (6, 1, [1, 1, 1, 3, 9]),
        (6, 11, [1, 3, 5, 9, 15]),
        (6, 13, [1, 1, 5, 11, 13]),
        (6, 19, [1, 7, 9, 13, 15]),
        (7, 1, [1, 1, 3, 7, 15, 31]),
        (7, 11, [1, 3, 7, 9, 15, 21]),
        (7, 16, [1, 1, 5, 9, 15, 23]),
    ]

    def __init__(self, dim):
        if dim > len(self._DIM_DATA):
            raise ValueError(f"Sobol序列内置最多 {len(self._DIM_DATA)} 维数据，当前请求 {dim} 维")
        self.dim = dim
        self._v = [self._generate_direction_numbers(d) for d in range(dim)]
        self._index = 0
        self._x = [0] * dim

    def _generate_direction_numbers(self, d):
        if d == 0:
            return [1 << (self.MAX_BITS - 1 - i) for i in range(self.MAX_BITS)]
        s, a_poly, m_init = self._DIM_DATA[d]
        m = list(m_init) + [0] * (self.MAX_BITS - len(m_init))
        for k in range(s, self.MAX_BITS):
            m[k] = m[k - s] << s
            for j in range(1, s):
                if (a_poly >> (j - 1)) & 1:
                    m[k] ^= (m[k - (s - j)] << j)
            m[k] ^= m[k - s]
        v = [0] * self.MAX_BITS
        for i in range(self.MAX_BITS):
            v[i] = m[i] << (self.MAX_BITS - 1 - i)
        return v

    def next_point(self):
        if self._index == 0:
            self._index = 1
            return [0.0] * self.dim
        c = 0
        val = self._index
        while val & 1:
            c += 1
            val >>= 1
        point = []
        for d in range(self.dim):
            self._x[d] ^= self._v[d][c]
            point.append(self._x[d] / (2 ** self.MAX_BITS))
        self._index += 1
        return point

    def generate(self, n, skip=0):
        self._index = 0
        self._x = [0] * self.dim
        points = []
        for i in range(skip):
            self.next_point()
        for _ in range(n):
            points.append(self.next_point())
        return points


def _t_critical(df, alpha=0.05):
    """
    近似计算 t 分布双侧临界值 t_{1-alpha/2, df}。
    当 df >= 30 时用正态近似；df < 30 时用查表法。
    """
    TABLE = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
    }
    if df >= 120:
        return 1.960
    if df in TABLE:
        return TABLE[df]
    lower = max(k for k in TABLE if k <= df)
    upper = min(k for k in TABLE if k >= df)
    if lower == upper:
        return TABLE[lower]
    w = (df - lower) / (upper - lower)
    return TABLE[lower] * (1 - w) + TABLE[upper] * w


def mc_integral_nd(f, bounds, N, method='sobol', num_randomizations=30, confidence=0.95):
    """
    多维蒙特卡洛积分，支持超立方体区域、低差异序列和置信区间。
    
    参数:
        f: 被积函数，接受 d 维列表/元组参数，返回浮点数。
           例如 f(x) 其中 x = [x1, x2, ..., xd]
        bounds: 积分区域边界，格式为 [(a1,b1), (a2,b2), ..., (ad,bd)]
                表示超立方体 [a1,b1] x [a2,b2] x ... x [ad,bd]
        N: 每次随机化使用的采样点数
        method: 采样方法，可选：
            'random'  - 纯随机采样
            'halton'  - Halton 低差异序列 + 随机移位
            'sobol'   - Sobol 低差异序列 + 随机移位（默认）
        num_randomizations: 随机移位次数（用于误差估计），默认30
        confidence: 置信水平，默认0.95（即95%置信区间）
    
    返回:
        dict: {
            'integral': 积分估计值,
            'std_error': 标准误差,
            'confidence_interval': (下界, 上界),
            'confidence_level': 置信水平,
            'method': 使用的采样方法,
            'num_randomizations': 随机化次数,
            'points_per_randomization': 每次随机化采样点数
        }
    """
    dim = len(bounds)
    volume = 1.0
    for (a, b) in bounds:
        if b <= a:
            raise ValueError(f"积分区间上界({b})必须大于下界({a})")
        volume *= (b - a)

    if method == 'random':
        estimates = []
        for _ in range(num_randomizations):
            total = 0.0
            for _ in range(N):
                point = []
                for (a, b) in bounds:
                    point.append(random.uniform(a, b))
                total += f(point)
            estimates.append(volume * total / N)
    elif method == 'halton':
        seq = HaltonSequence(dim)
        base_points = seq.generate(N, skip=0)
        estimates = []
        for _ in range(num_randomizations):
            shift = [random.random() for _ in range(dim)]
            total = 0.0
            for pt in base_points:
                shifted = []
                for k in range(dim):
                    val = (pt[k] + shift[k]) % 1.0
                    shifted.append(bounds[k][0] + val * (bounds[k][1] - bounds[k][0]))
                total += f(shifted)
            estimates.append(volume * total / N)
    elif method == 'sobol':
        seq = SobolSequence(dim)
        base_points = seq.generate(N, skip=0)
        estimates = []
        for _ in range(num_randomizations):
            shift = [random.random() for _ in range(dim)]
            total = 0.0
            for pt in base_points:
                shifted = []
                for k in range(dim):
                    val = (pt[k] + shift[k]) % 1.0
                    shifted.append(bounds[k][0] + val * (bounds[k][1] - bounds[k][0]))
                total += f(shifted)
            estimates.append(volume * total / N)
    else:
        raise ValueError(f"不支持的采样方法: {method}，可选: 'random', 'halton', 'sobol'")

    mean_est = sum(estimates) / len(estimates)
    var_est = sum((e - mean_est) ** 2 for e in estimates) / (len(estimates) - 1)
    std_error = math.sqrt(var_est / len(estimates))

    alpha = 1.0 - confidence
    t_val = _t_critical(num_randomizations - 1, alpha)
    half_width = t_val * std_error
    ci = (mean_est - half_width, mean_est + half_width)

    return {
        'integral': mean_est,
        'std_error': std_error,
        'confidence_interval': ci,
        'confidence_level': confidence,
        'method': method,
        'num_randomizations': num_randomizations,
        'points_per_randomization': N,
    }


if __name__ == "__main__":
    random.seed(42)

    def compare_test(label, f, a, b, N, true_value):
        r_uniform, e_uniform = monte_carlo_integral(f, a, b, N, importance_sampling=False)
        r_is, e_is = monte_carlo_integral(f, a, b, N, importance_sampling=True)
        print(f"\n{label}")
        print(f"  真实值:       {true_value:.6f}")
        print(f"  均匀采样:     估计={r_uniform:.6f}  误差={e_uniform:.6f}  偏差={abs(r_uniform - true_value):.6f}")
        print(f"  重要性采样:   估计={r_is:.6f}  误差={e_is:.6f}  偏差={abs(r_is - true_value):.6f}")
        if e_uniform > 0:
            print(f"  误差降低:     {(1 - e_is / e_uniform) * 100:.1f}%")

    print("=" * 70)
    print("一维蒙特卡洛积分测试（重要性采样 vs 均匀采样）")
    print("=" * 70)

    N = 100000

    compare_test("测试1: x^2 从 0 到 1", lambda x: x**2, 0, 1, N, 1.0 / 3)
    compare_test("测试2: sin(x) 从 0 到 π", lambda x: math.sin(x), 0, math.pi, N, 2.0)
    compare_test("测试3: e^x 从 0 到 1", lambda x: math.exp(x), 0, 1, N, math.e - 1)
    compare_test("测试4: 10*exp(-10*x) 从 0 到 1 (尖峰函数)", lambda x: 10 * math.exp(-10 * x), 0, 1, N, 1.0 - math.exp(-10))
    compare_test("测试5: 1/(1+100*(x-0.8)^2) 从 0 到 1 (窄峰)", lambda x: 1.0 / (1 + 100 * (x - 0.8) ** 2), 0, 1, N, (math.atan(2) + math.atan(8)) / 10)
    compare_test("测试6: exp(-x^2) 从 -5 到 5 (高斯型)", lambda x: math.exp(-x ** 2), -5, 5, N, math.sqrt(math.pi))

    print("\n\n" + "=" * 70)
    print("多维蒙特卡洛积分测试（Sobol vs Halton vs 纯随机）")
    print("=" * 70)

    def compare_nd_test(label, f, bounds, N, true_value, num_rand=30):
        print(f"\n{label}")
        print(f"  真实值: {true_value:.6f}")
        print(f"  维度: {len(bounds)}, 每次随机化采样: {N}, 随机化次数: {num_rand}")
        for method in ['random', 'halton', 'sobol']:
            result = mc_integral_nd(f, bounds, N, method=method, num_randomizations=num_rand)
            ci = result['confidence_interval']
            in_ci = ci[0] <= true_value <= ci[1]
            print(f"  {method:8s}: 估计={result['integral']:.6f}  标准误差={result['std_error']:.6f}  "
                  f"95%CI=[{ci[0]:.6f}, {ci[1]:.6f}]  偏差={abs(result['integral'] - true_value):.6f}  "
                  f"CI包含真值: {'是' if in_ci else '否'}")

    print("\n--- 2维测试 ---")

    compare_nd_test(
        "2D: ∫∫ x1*x2 dA, [0,1]x[0,1]",
        lambda x: x[0] * x[1],
        [(0, 1), (0, 1)],
        10000, 0.25
    )

    compare_nd_test(
        "2D: ∫∫ sin(x1)*cos(x2) dA, [0,π]x[0,π/2]",
        lambda x: math.sin(x[0]) * math.cos(x[1]),
        [(0, math.pi), (0, math.pi / 2)],
        10000, 2.0
    )

    print("\n--- 3维测试 ---")

    compare_nd_test(
        "3D: ∫∫∫ exp(-(x1²+x2²+x3²)) dV, [-1,1]³",
        lambda x: math.exp(-(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)),
        [(-1, 1), (-1, 1), (-1, 1)],
        20000,
        math.pi ** 1.5 * math.erf(1) ** 3,
        num_rand=30
    )

    print("\n--- 5维测试 ---")

    def f_5d_sphere(x):
        r2 = sum(xi ** 2 for xi in x)
        return 1.0 if r2 <= 1.0 else 0.0

    compare_nd_test(
        "5D: 单位球体积指示函数, [-1,1]⁵",
        f_5d_sphere,
        [(-1, 1)] * 5,
        50000,
        math.pi ** 2.5 / math.gamma(2.5 + 1),
        num_rand=30
    )

    print("\n--- 10维测试 ---")

    def f_10d_gaussian(x):
        return math.exp(-sum(xi ** 2 for xi in x))

    true_10d = math.pi ** 5
    compare_nd_test(
        "10D: ∫...∫ exp(-||x||²) dV, [-3,3]¹⁰",
        f_10d_gaussian,
        [(-3, 3)] * 10,
        100000,
        true_10d,
        num_rand=30
    )

    print("\n\n" + "=" * 70)
    print("收敛速度对比测试：不同N下的偏差")
    print("=" * 70)

    f_test = lambda x: x[0] * x[1]
    bounds_test = [(0, 1), (0, 1)]
    true_test = 0.25
    print(f"\n函数: x1*x2, 区域: [0,1]², 真实值: {true_test}")
    print(f"{'N':>10s}  {'random偏差':>12s}  {'halton偏差':>12s}  {'sobol偏差':>12s}")
    for N_val in [100, 500, 1000, 5000, 10000, 50000]:
        results = {}
        for method in ['random', 'halton', 'sobol']:
            r = mc_integral_nd(f_test, bounds_test, N_val, method=method, num_randomizations=20)
            results[method] = abs(r['integral'] - true_test)
        print(f"{N_val:>10d}  {results['random']:>12.6f}  {results['halton']:>12.6f}  {results['sobol']:>12.6f}")
