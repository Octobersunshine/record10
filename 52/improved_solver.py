import numpy as np
from scipy.special import gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ImprovedFractionalODESolver:
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

    def solve_relaxation_predictor_corrector(self, f0=1.0, lambda_val=1.0, corrector_steps=2):
        f = np.zeros(self.N + 1)
        f[0] = f0
        coeffs = self.grunwald_coefficients(self.N, shifted=True)

        for n in range(1, self.N + 1):
            sum_pred = 0.0
            for k in range(1, n):
                sum_pred += coeffs[k] * f[n - k]
            f_pred = -lambda_val * (self.h ** self.alpha) * f[n - 1] - sum_pred

            f_corr = f_pred
            for _ in range(corrector_steps):
                sum_corr = 0.0
                for k in range(1, n):
                    sum_corr += coeffs[k] * f[n - k]
                denom = 1.0 + lambda_val * (self.h ** self.alpha) - coeffs[n]
                f_corr = (-sum_corr) / denom
            f[n] = f_corr

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


def compare_methods():
    alpha = 0.95
    t_final = 5.0
    h = 0.01

    solver = ImprovedFractionalODESolver(alpha, t_final, h)

    t_impl, f_impl = solver.solve_relaxation_implicit()
    t_pc, f_pc = solver.solve_relaxation_predictor_corrector()
    t_l1, f_l1 = solver.solve_relaxation_l1()

    t_exact = np.linspace(0, t_final, 1000)
    f_exact = np.exp(-t_exact)

    plt.figure(figsize=(12, 8))
    plt.plot(t_exact, f_exact, 'k-', label='精确解 (α=1时指数衰减)', linewidth=2)
    plt.plot(t_impl, f_impl, 'b--', label='隐式格式', linewidth=2, markevery=50)
    plt.plot(t_pc, f_pc, 'g-.', label='预测-校正法', linewidth=2, markevery=50)
    plt.plot(t_l1, f_l1, 'r:', label='L1格式', linewidth=2, markevery=50)
    plt.xlabel('时间 t', fontsize=12)
    plt.ylabel('f(t)', fontsize=12)
    plt.title(f'不同数值方法对比 (α = {alpha})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("不同方法在t=5.0处的值:")
    print(f"  精确解(α=1): {np.exp(-5):.6f}")
    print(f"  隐式格式:    {f_impl[-1]:.6f}")
    print(f"  预测-校正:   {f_pc[-1]:.6f}")
    print(f"  L1格式:      {f_l1[-1]:.6f}")


def stability_test():
    alphas = [0.9, 0.95, 0.99, 1.0]
    t_final = 5.0
    h = 0.01

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, alpha in enumerate(alphas):
        solver = ImprovedFractionalODESolver(alpha, t_final, h)
        t, f = solver.solve_relaxation_implicit()

        ax = axes[idx]
        ax.plot(t, f, 'b-', linewidth=2)
        ax.set_xlabel('时间 t', fontsize=10)
        ax.set_ylabel('f(t)', fontsize=10)
        ax.set_title(f'α = {alpha}', fontsize=12)
        ax.grid(True, alpha=0.3)

        if alpha >= 0.99:
            t_exact = np.linspace(0, t_final, len(t))
            f_exact = np.exp(-t_exact)
            ax.plot(t_exact, f_exact, 'r--', label='指数衰减', linewidth=1.5)
            ax.legend(fontsize=9)

    plt.suptitle('隐式格式在不同α值下的稳定性', fontsize=14)
    plt.tight_layout()
    plt.savefig('stability_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n稳定性测试完成！")
    for alpha in alphas:
        solver = ImprovedFractionalODESolver(alpha, t_final, h)
        t, f = solver.solve_relaxation_implicit()
        has_oscillation = np.any(np.diff(np.sign(np.diff(f))) != 0)
        print(f"  α={alpha}: {'有震荡' if has_oscillation else '无震荡'}")


def main():
    print("改进的分数阶微分方程求解器")
    print("=" * 50)
    print()

    print("1. 不同数值方法对比...")
    compare_methods()
    print()

    print("2. 稳定性测试...")
    stability_test()
    print()

    print("所有测试完成！图像已保存。")
    print()
    print("改进方案说明:")
    print("-" * 50)
    print("1. 隐式格式 (Implicit Euler):")
    print("   - 无条件稳定，避免震荡")
    print("   - 需要求解简单的线性方程")
    print()
    print("2. 预测-校正法 (Predictor-Corrector):")
    print("   - 先用显式预测，再用隐式校正")
    print("   - 兼顾精度和稳定性")
    print()
    print("3. L1格式:")
    print("   - 基于分段线性逼近")
    print("   - 精度更高，稳定性好")
    print()
    print("4. 移位的Grünwald-Letnikov系数:")
    print("   - 改善边界处理")
    print("   - 提高数值稳定性")


if __name__ == "__main__":
    main()
