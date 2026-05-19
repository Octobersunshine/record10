import numpy as np


def grunwald_coefficients(alpha, n, shifted=True):
    coeffs = np.zeros(n + 1)
    coeffs[0] = 1.0
    for k in range(1, n + 1):
        if shifted:
            coeffs[k] = coeffs[k - 1] * (1 - (alpha + 1) / k)
        else:
            coeffs[k] = coeffs[k - 1] * (1 - alpha / k)
    return coeffs


def solve_relaxation_explicit(alpha, t_final, h, f0=1.0, lambda_val=1.0):
    N = int(t_final / h)
    t = np.linspace(0, t_final, N + 1)
    f = np.zeros(N + 1)
    f[0] = f0
    coeffs = grunwald_coefficients(alpha, N, shifted=False)

    for n in range(1, N + 1):
        sum_val = 0.0
        for k in range(1, n + 1):
            sum_val += coeffs[k] * f[n - k]
        f[n] = -lambda_val * (h ** alpha) * f[n - 1] - sum_val

    return t, f


def solve_relaxation_implicit(alpha, t_final, h, f0=1.0, lambda_val=1.0):
    N = int(t_final / h)
    t = np.linspace(0, t_final, N + 1)
    f = np.zeros(N + 1)
    f[0] = f0
    coeffs = grunwald_coefficients(alpha, N, shifted=True)

    for n in range(1, N + 1):
        sum_val = 0.0
        for k in range(1, n):
            sum_val += coeffs[k] * f[n - k]
        denom = 1.0 + lambda_val * (h ** alpha) - coeffs[n]
        f[n] = (-sum_val) / denom

    return t, f


def check_oscillation(f):
    df = np.diff(f)
    sign_changes = np.sum(np.diff(np.sign(df)) != 0)
    return sign_changes > 0, sign_changes


def main():
    print("分数阶弛豫方程数值稳定性测试")
    print("=" * 60)
    print()
    print("测试α接近1时的数值稳定性...")
    print()

    alpha = 0.95
    t_final = 5.0
    h = 0.01

    print(f"参数: α={alpha}, t_final={t_final}, h={h}")
    print()

    t_exp, f_exp = solve_relaxation_explicit(alpha, t_final, h)
    t_impl, f_impl = solve_relaxation_implicit(alpha, t_final, h)

    has_osc_exp, num_osc_exp = check_oscillation(f_exp)
    has_osc_impl, num_osc_impl = check_oscillation(f_impl)

    print("=" * 60)
    print("方法对比:")
    print("-" * 60)
    print(f"{'方法':<15} {'t=5.0处的值':<15} {'是否震荡':<12} {'震荡次数':<10}")
    print("-" * 60)
    print(f"{'显式格式':<15} {f_exp[-1]:<15.6f} {'是 ⚠️' if has_osc_exp else '否 ✓':<12} {num_osc_exp:<10}")
    print(f"{'隐式格式':<15} {f_impl[-1]:<15.6f} {'是 ⚠️' if has_osc_impl else '否 ✓':<12} {num_osc_impl:<10}")
    print("-" * 60)
    print()

    print("改进效果:")
    print("=" * 60)
    if has_osc_exp and not has_osc_impl:
        print("✓ 成功消除了数值震荡!")
        print(f"  显式格式: 出现 {num_osc_exp} 次震荡")
        print(f"  隐式格式: 无震荡")
    elif has_osc_exp:
        print("⚠️  两种方法都有震荡，建议使用更小的时间步长")
    else:
        print("✓ 两种方法都稳定")
    print()

    print("不同α值的稳定性测试:")
    print("-" * 60)
    alphas = [0.8, 0.9, 0.95, 0.99]
    print(f"{'α':<8} {'显式格式':<15} {'隐式格式':<15}")
    print("-" * 60)
    for a in alphas:
        _, f_e = solve_relaxation_explicit(a, 3.0, 0.01)
        _, f_i = solve_relaxation_implicit(a, 3.0, 0.01)
        osc_e, _ = check_oscillation(f_e)
        osc_i, _ = check_oscillation(f_i)
        status_e = "震荡 ⚠️" if osc_e else "稳定 ✓"
        status_i = "震荡 ⚠️" if osc_i else "稳定 ✓"
        print(f"{a:<8.2f} {status_e:<15} {status_i:<15}")
    print()

    print("=" * 60)
    print("稳定性改进原理:")
    print("-" * 60)
    print("1. 隐式格式将最新时间步的未知项移到等式左边")
    print("2. 使用移位的Grünwald-Letnikov系数改善边界处理")
    print("3. 无条件稳定，无需对时间步长施加严格限制")
    print()
    print("测试完成!")


if __name__ == "__main__":
    main()
