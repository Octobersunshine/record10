import math
import numpy as np
from scipy import stats
from scipy.stats import qmc


def monte_carlo_integrate(f, a, b, N, confidence=0.95, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, N)
    y = f(x)
    width = b - a
    mean_y = np.mean(y)
    integral = mean_y * width
    std_y = np.std(y, ddof=1)
    se = std_y * width / np.sqrt(N)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_low = integral - z * se
    ci_high = integral + z * se
    return integral, se, (ci_low, ci_high)


class ImportanceSampler:
    def __init__(self, f, a, b, num_bins=1000):
        self.a = a
        self.b = b
        self.num_bins = num_bins
        self.edges = np.linspace(a, b, num_bins + 1)
        self.dx = self.edges[1] - self.edges[0]
        h = np.abs(f(self.edges))
        h = np.maximum(h, 1e-15)
        self.h = h
        self.areas = (h[:-1] + h[1:]) * self.dx / 2.0
        self.total_area = np.sum(self.areas)
        self.probs = self.areas / self.total_area
        self.cum_probs = np.concatenate([[0.0], np.cumsum(self.probs)])

    def sample(self, rng, N):
        u = rng.uniform(0.0, 1.0, N)
        bin_indices = np.searchsorted(self.cum_probs, u, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        u_local = (u - self.cum_probs[bin_indices]) / self.probs[bin_indices]
        u_local = np.clip(u_local, 0.0, 1.0)
        h_left = self.h[bin_indices]
        h_right = self.h[bin_indices + 1]
        dh = h_right - h_left
        near_flat = np.abs(dh) < 1e-12 * np.maximum(h_left, 1e-15)
        t = np.empty(N)
        t[near_flat] = u_local[near_flat]
        active = ~near_flat
        if np.any(active):
            ha = h_left[active]
            dha = dh[active]
            ua = u_local[active]
            discriminant = ha * ha + 2.0 * dha * ua * (ha + dha / 2.0)
            discriminant = np.maximum(discriminant, 0.0)
            t[active] = (-ha + np.sqrt(discriminant)) / dha
        t = np.clip(t, 0.0, 1.0)
        x = self.edges[bin_indices] + t * self.dx
        return x

    def pdf(self, x):
        bin_indices = np.clip(((x - self.a) / self.dx).astype(int), 0, self.num_bins - 1)
        t = (x - self.edges[bin_indices]) / self.dx
        h_at_x = self.h[bin_indices] + t * (self.h[bin_indices + 1] - self.h[bin_indices])
        return h_at_x / self.total_area


def monte_carlo_integrate_is(f, a, b, N, num_bins=1000, confidence=0.95, seed=None):
    sampler = ImportanceSampler(f, a, b, num_bins)
    rng = np.random.default_rng(seed)
    x = sampler.sample(rng, N)
    y = f(x)
    p = sampler.pdf(x)
    weights = y / p
    integral = np.mean(weights)
    se = np.std(weights, ddof=1) / np.sqrt(N)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_low = integral - z * se
    ci_high = integral + z * se
    return integral, se, (ci_low, ci_high)


def monte_carlo_integrate_nd(f, a, b, N, confidence=0.95, seed=None):
    """
    多维蒙特卡洛积分（均匀随机采样），支持超立方体区域 [a, b]。

    参数:
        f: 被积函数，接受形状 (N, d) 的输入，返回形状 (N,) 的输出
        a: 数组，各维度下界，shape (d,)
        b: 数组，各维度上界，shape (d,)
        N: 采样点数
        confidence: 置信水平 (默认 0.95)
        seed: 随机种子

    返回:
        integral: 积分近似值
        se: 标准误差
        (ci_low, ci_high): 置信区间
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a.shape[0]
    if b.shape[0] != d:
        raise ValueError("a 和 b 必须有相同的维度")

    rng = np.random.default_rng(seed)
    x = rng.uniform(low=a, high=b, size=(N, d))
    y = f(x)

    volume = np.prod(b - a)
    mean_y = np.mean(y)
    integral = mean_y * volume
    std_y = np.std(y, ddof=1)
    se = std_y * volume / np.sqrt(N)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_low = integral - z * se
    ci_high = integral + z * se
    return integral, se, (ci_low, ci_high)


def monte_carlo_integrate_nd_sobol(f, a, b, N, confidence=0.95, seed=None):
    """
    多维蒙特卡洛积分（Sobol 低差异序列），支持超立方体区域 [a, b]。

    Sobol 序列的差异度 D_N = O((log N)^d / N)，比纯随机的 O(1/√N) 收敛更快，
    尤其在高维问题中优势明显。

    参数:
        f: 被积函数，接受形状 (N, d) 的输入，返回形状 (N,) 的输出
        a: 数组，各维度下界，shape (d,)
        b: 数组，各维度上界，shape (d,)
        N: 采样点数，建议为 2 的幂次以获得最佳均匀性
        confidence: 置信水平 (默认 0.95)
        seed: 随机种子（用于 Sobol 序列的 scramble）

    返回:
        integral: 积分近似值
        se: 标准误差（基于样本标准差）
        (ci_low, ci_high): 置信区间
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a.shape[0]
    if b.shape[0] != d:
        raise ValueError("a 和 b 必须有相同的维度")

    rng = np.random.default_rng(seed)
    sobol_sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = sobol_sampler.random(N)
    x = a + u * (b - a)
    y = f(x)

    volume = np.prod(b - a)
    mean_y = np.mean(y)
    integral = mean_y * volume
    std_y = np.std(y, ddof=1)
    se = std_y * volume / np.sqrt(N)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_low = integral - z * se
    ci_high = integral + z * se
    return integral, se, (ci_low, ci_high)


if __name__ == "__main__":
    N = 100000

    print("=" * 70)
    print("测试1: sin(x) on [0, pi], 全正函数, 真实值 = 2.0")
    print("=" * 70)

    f1 = np.sin
    a1, b1 = 0.0, np.pi

    approx_u, se_u, ci_u = monte_carlo_integrate(f1, a1, b1, N, seed=42)
    approx_is, se_is, ci_is = monte_carlo_integrate_is(f1, a1, b1, N, seed=42)

    print(f"  均匀采样:   近似值 = {approx_u:.6f}, SE = {se_u:.6f}, 95% CI = [{ci_u[0]:.6f}, {ci_u[1]:.6f}]")
    print(f"  重要性采样: 近似值 = {approx_is:.6f}, SE = {se_is:.6e}, 95% CI = [{ci_is[0]:.6f}, {ci_is[1]:.6f}]")
    print(f"  真实值 = 2.0, 在 95% CI 内? 均匀: {ci_u[0] <= 2.0 <= ci_u[1]}, IS: {ci_is[0] <= 2.0 <= ci_is[1]}")

    print()
    print("=" * 70)
    print("测试2: sin(10x)*exp(-x) on [0, pi], 振荡衰减函数")
    print("       (剧烈振荡 + 衰减, 真实值 = 10/101*(1-e^(-pi)))")
    print("=" * 70)

    f2 = lambda x: np.sin(10.0 * x) * np.exp(-x)
    a2, b2 = 0.0, np.pi
    exact2 = 10.0 / 101.0 * (1.0 - np.exp(-np.pi))

    approx_u2, se_u2, ci_u2 = monte_carlo_integrate(f2, a2, b2, N, seed=42)
    approx_is2, se_is2, ci_is2 = monte_carlo_integrate_is(f2, a2, b2, N, seed=42)

    print(f"  均匀采样:   近似值 = {approx_u2:.6f}, SE = {se_u2:.6f}, 95% CI = [{ci_u2[0]:.6f}, {ci_u2[1]:.6f}]")
    print(f"  重要性采样: 近似值 = {approx_is2:.6f}, SE = {se_is2:.6f}, 95% CI = [{ci_is2[0]:.6f}, {ci_is2[1]:.6f}]")
    print(f"  真实值 = {exact2:.6f}, 在 95% CI 内? 均匀: {ci_u2[0] <= exact2 <= ci_u2[1]}, IS: {ci_is2[0] <= exact2 <= ci_is2[1]}")

    print()
    print("=" * 70)
    print("测试3: exp(-50*(x-0.5)^2) on [0,1], 极窄高斯峰")
    print("       (峰值在 x=0.5, 均匀采样难以覆盖)")
    print("=" * 70)

    f3 = lambda x: np.exp(-50.0 * (x - 0.5) ** 2)
    a3, b3 = 0.0, 1.0
    exact3 = np.sqrt(np.pi / 50.0) * 0.5 * (math.erf(np.sqrt(50.0) * 0.5) - math.erf(-np.sqrt(50.0) * 0.5))

    approx_u3, se_u3, ci_u3 = monte_carlo_integrate(f3, a3, b3, N, seed=42)
    approx_is3, se_is3, ci_is3 = monte_carlo_integrate_is(f3, a3, b3, N, seed=42)

    print(f"  均匀采样:   近似值 = {approx_u3:.6f}, SE = {se_u3:.6f}, 95% CI = [{ci_u3[0]:.6f}, {ci_u3[1]:.6f}]")
    print(f"  重要性采样: 近似值 = {approx_is3:.6f}, SE = {se_is3:.6e}, 95% CI = [{ci_is3[0]:.6f}, {ci_is3[1]:.6f}]")
    print(f"  真实值 = {exact3:.6f}, 在 95% CI 内? 均匀: {ci_u3[0] <= exact3 <= ci_u3[1]}, IS: {ci_is3[0] <= exact3 <= ci_is3[1]}")

    print()
    print("=" * 70)
    print("测试4: 1/(1+1000*(x-0.3)^2) on [0,1], 窄洛伦兹峰")
    print("=" * 70)

    f4 = lambda x: 1.0 / (1.0 + 1000.0 * (x - 0.3) ** 2)
    a4, b4 = 0.0, 1.0
    gamma = np.sqrt(1000.0)
    exact4 = (1.0 / gamma) * (math.atan(gamma * (1.0 - 0.3)) - math.atan(gamma * (0.0 - 0.3)))

    approx_u4, se_u4, ci_u4 = monte_carlo_integrate(f4, a4, b4, N, seed=42)
    approx_is4, se_is4, ci_is4 = monte_carlo_integrate_is(f4, a4, b4, N, seed=42)

    print(f"  均匀采样:   近似值 = {approx_u4:.6f}, SE = {se_u4:.6f}, 95% CI = [{ci_u4[0]:.6f}, {ci_u4[1]:.6f}]")
    print(f"  重要性采样: 近似值 = {approx_is4:.6f}, SE = {se_is4:.6e}, 95% CI = [{ci_is4[0]:.6f}, {ci_is4[1]:.6f}]")
    print(f"  真实值 = {exact4:.6f}, 在 95% CI 内? 均匀: {ci_u4[0] <= exact4 <= ci_u4[1]}, IS: {ci_is4[0] <= exact4 <= ci_is4[1]}")

    print()
    print("=" * 70)
    print("误差缩小倍数 (均匀采样误差 / IS采样误差):")
    print("=" * 70)
    ratios = []
    for label, eu, eis in [
        ("sin(x)          ", se_u, se_is),
        ("sin(10x)*exp(-x)", se_u2, se_is2),
        ("窄高斯峰        ", se_u3, se_is3),
        ("窄洛伦兹峰      ", se_u4, se_is4),
    ]:
        r = eu / eis if eis > 1e-15 else float('inf')
        ratios.append(r)
        print(f"  {label}: {r:.1f}x")

    print()
    print("=" * 70)
    print("测试5: 6维高斯积分 ∫_{[-1,1]^6} exp(-|x|^2) dx")
    print("       (d=6维, 真实值 = (√π erf(1))^6)")
    print("=" * 70)

    d = 6
    a_nd = [-1.0] * d
    b_nd = [1.0] * d

    f_nd = lambda x: np.exp(-np.sum(x ** 2, axis=1))
    exact_nd = (np.sqrt(np.pi) * math.erf(1.0)) ** d

    N_nd = 2 ** 16

    approx_nd, se_nd, ci_nd = monte_carlo_integrate_nd(f_nd, a_nd, b_nd, N_nd, seed=42)
    approx_sobol, se_sobol, ci_sobol = monte_carlo_integrate_nd_sobol(f_nd, a_nd, b_nd, N_nd, seed=42)

    print(f"  维度 d = {d}, 采样点数 N = {N_nd}")
    print(f"  均匀随机: 近似值 = {approx_nd:.6f}, SE = {se_nd:.6f}, 95% CI = [{ci_nd[0]:.6f}, {ci_nd[1]:.6f}]")
    print(f"  Sobol:    近似值 = {approx_sobol:.6f}, SE = {se_sobol:.6f}, 95% CI = [{ci_sobol[0]:.6f}, {ci_sobol[1]:.6f}]")
    print(f"  真实值 = {exact_nd:.6f}")
    print(f"  真实值在 95% CI 内? 均匀: {ci_nd[0] <= exact_nd <= ci_nd[1]}, Sobol: {ci_sobol[0] <= exact_nd <= ci_sobol[1]}")
    print(f"  实际偏差: 均匀 = {abs(approx_nd - exact_nd):.6f}, Sobol = {abs(approx_sobol - exact_nd):.6f}")
    print(f"  Sobol 相对均匀的偏差比: 均匀/Sobol = {abs(approx_nd - exact_nd) / max(abs(approx_sobol - exact_nd), 1e-15):.2f}x")

    print()
    print("=" * 70)
    print("测试6: 10维单位球体积 ∫_{[-1,1]^10} 1_{|x|<=1} dx")
    print("       (d=10维, 真实值 = π^(5/2) / Γ(5/2) = π^5/120 ≈ 2.55016)")
    print("=" * 70)

    d2 = 10
    a_nd2 = [-1.0] * d2
    b_nd2 = [1.0] * d2

    f_nd2 = lambda x: (np.sum(x ** 2, axis=1) <= 1.0).astype(float)
    exact_nd2 = np.pi ** (d2 / 2) / math.gamma(d2 / 2 + 1)

    N_nd2 = 2 ** 17

    approx_nd2, se_nd2, ci_nd2 = monte_carlo_integrate_nd(f_nd2, a_nd2, b_nd2, N_nd2, seed=42)
    approx_sobol2, se_sobol2, ci_sobol2 = monte_carlo_integrate_nd_sobol(f_nd2, a_nd2, b_nd2, N_nd2, seed=42)

    print(f"  维度 d = {d2}, 采样点数 N = {N_nd2}")
    print(f"  均匀随机: 近似值 = {approx_nd2:.6f}, SE = {se_nd2:.6f}, 95% CI = [{ci_nd2[0]:.6f}, {ci_nd2[1]:.6f}]")
    print(f"  Sobol:    近似值 = {approx_sobol2:.6f}, SE = {se_sobol2:.6f}, 95% CI = [{ci_sobol2[0]:.6f}, {ci_sobol2[1]:.6f}]")
    print(f"  真实值 = {exact_nd2:.6f}")
    print(f"  真实值在 95% CI 内? 均匀: {ci_nd2[0] <= exact_nd2 <= ci_nd2[1]}, Sobol: {ci_sobol2[0] <= exact_nd2 <= ci_sobol2[1]}")
    print(f"  实际偏差: 均匀 = {abs(approx_nd2 - exact_nd2):.6f}, Sobol = {abs(approx_sobol2 - exact_nd2):.6f}")
    print(f"  Sobol 相对均匀的偏差比: 均匀/Sobol = {abs(approx_nd2 - exact_nd2) / max(abs(approx_sobol2 - exact_nd2), 1e-15):.2f}x")

    print()
    print("=" * 70)
    print("测试7: 收敛速度对比 (d=3维高斯积分)")
    print("       比较不同 N 下均匀随机 vs Sobol 的实际偏差")
    print("=" * 70)

    d3 = 3
    a_nd3 = [-1.0] * d3
    b_nd3 = [1.0] * d3
    f_nd3 = lambda x: np.exp(-np.sum(x ** 2, axis=1))
    exact_nd3 = (np.sqrt(np.pi) * math.erf(1.0)) ** d3

    print(f"  真实值 = {exact_nd3:.6f}")
    print(f"  {'N':>10} {'均匀偏差':>12} {'Sobol偏差':>12} {'偏差比(均/Sob)':>14}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*14}")
    for k in range(8, 18):
        N_conv = 2 ** k
        app_u, _, _ = monte_carlo_integrate_nd(f_nd3, a_nd3, b_nd3, N_conv, seed=42)
        app_s, _, _ = monte_carlo_integrate_nd_sobol(f_nd3, a_nd3, b_nd3, N_conv, seed=42)
        err_u = abs(app_u - exact_nd3)
        err_s = abs(app_s - exact_nd3)
        ratio = err_u / max(err_s, 1e-15)
        print(f"  {N_conv:>10} {err_u:>12.6f} {err_s:>12.6f} {ratio:>14.2f}")
