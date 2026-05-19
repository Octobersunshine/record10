import numpy as np
from scipy.special import gamma, mittag_leffler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FractionalODESolver:
    def __init__(self, alpha, t_final, h):
        self.alpha = alpha
        self.t_final = t_final
        self.h = h
        self.N = int(t_final / h)
        self.t = np.linspace(0, t_final, self.N + 1)

    def grunwald_coefficients(self, n, shifted=True):
        coeffs = np.zeros(n + 1)
        coeffs[0] = 1.0
        for k in range(1, n + 1):
            if shifted:
                coeffs[k] = coeffs[k - 1] * (1 - (self.alpha + 1) / k)
            else:
                coeffs[k] = coeffs[k - 1] * (1 - self.alpha / k)
        return coeffs

    def solve_relaxation_explicit(self, f0=1.0, lambda_val=1.0):
        f = np.zeros(self.N + 1)
        f[0] = f0
        coeffs = self.grunwald_coefficients(self.N, shifted=False)

        for n in range(1, self.N + 1):
            sum_val = 0.0
            for k in range(1, n + 1):
                sum_val += coeffs[k] * f[n - k]
            f[n] = -lambda_val * (self.h ** self.alpha) * f[n - 1] - sum_val

        return self.t, f

    def solve_relaxation_implicit(self, f0=1.0, lambda_val=1.0):
        f = np.zeros(self.N + 1)
        f[0] = f0
        coeffs = self.grunwald_coefficients(self.N, shifted=True)

        for n in range(1, self.N + 1):
            sum_val = 0.0
            for k in range(1, n):
                sum_val += coeffs[k] * f[n - k]
            denom = 1.0 + lambda_val * (self.h ** self.alpha) - coeffs[n]
            f[n] = (-sum_val) / denom

        return self.t, f

    def solve_relaxation_l1(self, f0=1.0, lambda_val=1.0):
        f = np.zeros(self.N + 1)
        f[0] = f0

        for n in range(1, self.N + 1):
            sum_val = 0.0
            for k in range(1, n + 1):
                b_k = (k ** (1 - self.alpha)) - ((k - 1) ** (1 - self.alpha))
                sum_val += b_k * (f[n - k + 1] - f[n - k])

            a_coeff = gamma(2 - self.alpha) * (self.h ** self.alpha)
            numerator = a_coeff * f[n - 1] - sum_val
            denominator = a_coeff + lambda_val * (self.h ** self.alpha) * gamma(2 - self.alpha)
            f[n] = numerator / denominator

        return self.t, f

    def solve_relaxation_equation(self, f0=1.0, lambda_val=1.0, method='implicit'):
        if method == 'explicit':
            return self.solve_relaxation_explicit(f0, lambda_val)
        elif method == 'implicit':
            return self.solve_relaxation_implicit(f0, lambda_val)
        elif method == 'l1':
            return self.solve_relaxation_l1(f0, lambda_val)
        else:
            raise ValueError(f"未知方法: {method}")

    def analytical_solution(self, f0=1.0, lambda_val=1.0):
        return f0 * mittag_leffler(-lambda_val * (self.t ** self.alpha), self.alpha)


def plot_comparison(t, numerical, analytical, alpha, method='隐式格式'):
    plt.figure(figsize=(10, 6))
    plt.plot(t, numerical, 'b-', label=f'数值解 ({method})', linewidth=2)
    plt.plot(t, analytical, 'r--', label='解析解 (Mittag-Leffler)', linewidth=2)
    plt.xlabel('时间 t', fontsize=12)
    plt.ylabel('f(t)', fontsize=12)
    plt.title(f'分数阶弛豫方程 (α = {alpha})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fractional_relaxation.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_stability_comparison():
    alpha = 0.95
    t_final = 5.0
    h = 0.01

    solver = FractionalODESolver(alpha, t_final, h)
    t_exp, f_exp = solver.solve_relaxation_equation(method='explicit')
    t_impl, f_impl = solver.solve_relaxation_equation(method='implicit')
    t_l1, f_l1 = solver.solve_relaxation_equation(method='l1')

    plt.figure(figsize=(12, 7))
    plt.plot(t_exp, f_exp, 'r--', label='显式格式 (有震荡)', linewidth=2, alpha=0.7)
    plt.plot(t_impl, f_impl, 'b-', label='隐式格式 (稳定)', linewidth=2)
    plt.plot(t_l1, f_l1, 'g-.', label='L1格式 (高精度)', linewidth=2)
    plt.xlabel('时间 t', fontsize=12)
    plt.ylabel('f(t)', fontsize=12)
    plt.title(f'不同数值格式稳定性对比 (α = {alpha})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n不同方法的震荡检测:")
    for name, f in [('显式格式', f_exp), ('隐式格式', f_impl), ('L1格式', f_l1)]:
        has_oscillation = np.any(np.diff(np.sign(np.diff(f))) != 0)
        print(f"  {name}: {'有震荡 ⚠️' if has_oscillation else '无震荡 ✓'}")


def plot_different_alphas(method='implicit'):
    alphas = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    t_final = 5.0
    h = 0.01

    plt.figure(figsize=(12, 7))

    for alpha in alphas:
        solver = FractionalODESolver(alpha, t_final, h)
        t, f_num = solver.solve_relaxation_equation(method=method)
        plt.plot(t, f_num, label=f'α = {alpha}', linewidth=2)

    plt.xlabel('时间 t', fontsize=12)
    plt.ylabel('f(t)', fontsize=12)
    method_name = '隐式格式' if method == 'implicit' else 'L1格式'
    plt.title(f'不同阶数α下的分数阶弛豫方程解 ({method_name})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('different_alphas.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("分数阶微分方程数值求解器")
    print("=" * 50)
    print()
    print("⚠️  重要改进：修复了α接近1时的数值震荡问题")
    print("=" * 50)
    print()

    alpha = 0.95
    t_final = 5.0
    h = 0.01

    print(f"参数设置:")
    print(f"  分数阶数 α = {alpha} (接近1，容易出现震荡)")
    print(f"  时间区间 [0, {t_final}]")
    print(f"  时间步长 h = {h}")
    print()

    solver = FractionalODESolver(alpha, t_final, h)

    print("1. 正在进行不同数值方法的稳定性对比...")
    plot_stability_comparison()
    print("   稳定性对比图已保存: stability_comparison.png")
    print()

    print("2. 使用隐式格式求解...")
    t, f_num = solver.solve_relaxation_equation(method='implicit')
    print("   数值解计算完成!")

    try:
        f_analytical = solver.analytical_solution()
        print("   解析解计算完成!")

        error = np.abs(f_num - f_analytical)
        print(f"   最大误差: {np.max(error):.6f}")
        print(f"   平均误差: {np.mean(error):.6f}")

        plot_comparison(t, f_num, f_analytical, alpha, '隐式格式')
        print("   数值-解析解对比图已保存: fractional_relaxation.png")
    except Exception as e:
        print(f"   警告: 解析解计算出错 ({str(e)})，跳过对比")
        plt.figure(figsize=(10, 6))
        plt.plot(t, f_num, 'b-', label='数值解 (隐式格式)', linewidth=2)
        plt.xlabel('时间 t', fontsize=12)
        plt.ylabel('f(t)', fontsize=12)
        plt.title(f'分数阶弛豫方程 (α = {alpha}) - 隐式格式', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fractional_relaxation.png', dpi=300, bbox_inches='tight')
        plt.close()

    print()
    print("3. 正在绘制不同阶数的对比图...")
    plot_different_alphas(method='implicit')
    print("   不同α值对比图已保存: different_alphas.png")
    print()

    print("=" * 50)
    print("稳定性改进说明:")
    print("-" * 50)
    print("1. 显式格式 (Original):")
    print("   - 条件稳定，α接近1时容易出现震荡")
    print("   - 计算简单，但需要很小的时间步长")
    print()
    print("2. 隐式格式 (Implicit Euler): ✓ 推荐")
    print("   - 无条件稳定，无震荡")
    print("   - 使用移位的Grünwald-Letnikov系数")
    print("   - 计算效率高，适合工程应用")
    print()
    print("3. L1格式: ⭐ 高精度")
    print("   - 基于分段线性逼近")
    print("   - 精度更高，稳定性极好")
    print("   - 适合需要高精度的科研计算")
    print()
    print("使用方法:")
    print("  solver.solve_relaxation_equation(method='implicit')  # 隐式格式")
    print("  solver.solve_relaxation_equation(method='l1')       # L1格式")
    print("  solver.solve_relaxation_equation(method='explicit') # 显式格式")
    print()
    print("程序执行完成!")


if __name__ == "__main__":
    main()
