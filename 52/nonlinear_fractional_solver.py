import numpy as np
from scipy.special import gamma, binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


class NonlinearFractionalSolver:
    def __init__(self, alpha, t_final, h, dim=1):
        self.alpha = alpha if np.isscalar(alpha) else np.array(alpha)
        self.t_final = t_final
        self.h = h
        self.N = int(t_final / h)
        self.t = np.linspace(0, t_final, self.N + 1)
        self.dim = dim if np.isscalar(alpha) else len(alpha)

    def binomial_coeffs(self, n):
        if np.isscalar(self.alpha):
            alpha = self.alpha
            b = np.zeros(n + 1)
            b[0] = 1.0
            for j in range(1, n + 1):
                b[j] = b[j - 1] * (1 - (alpha + 1) / j)
            return b
        else:
            coeffs = []
            for a in self.alpha:
                b = np.zeros(n + 1)
                b[0] = 1.0
                for j in range(1, n + 1):
                    b[j] = b[j - 1] * (1 - (a + 1) / j)
                coeffs.append(b)
            return coeffs

    def predictor_corrector(self, f, x0, params=None):
        x = np.zeros((self.N + 1, self.dim))
        x[0] = x0

        if np.isscalar(self.alpha):
            alpha = self.alpha
            mu = gamma(2 - alpha) * (self.h ** alpha)

            for n in range(self.N):
                sum_pred = 0.0
                for j in range(n + 1):
                    b_j = (n - j + 1) ** (1 - alpha) - (n - j) ** (1 - alpha)
                    sum_pred += b_j * f(self.t[j], x[j], params)
                x_pred = x0 + sum_pred / gamma(2 - alpha)

                sum_corr = 0.0
                for j in range(1, n + 1):
                    a_j = (n - j + 1) ** (2 - alpha) - 2 * (n - j) ** (2 - alpha) + (n - j - 1) ** (2 - alpha)
                    sum_corr += a_j * f(self.t[j], x[j], params)

                x[n + 1] = x0 + (self.h ** alpha) / gamma(alpha + 2) * (
                    f(self.t[n + 1], x_pred, params) + sum_corr
                )
        else:
            for i in range(self.dim):
                alpha_i = self.alpha[i]
                mu_i = gamma(2 - alpha_i) * (self.h ** alpha_i)

                for n in range(self.N):
                    sum_pred = 0.0
                    for j in range(n + 1):
                        b_j = (n - j + 1) ** (1 - alpha_i) - (n - j) ** (1 - alpha_i)
                        sum_pred += b_j * f(self.t[j], x[j], params)[i]
                    x_pred_i = x0[i] + sum_pred / gamma(2 - alpha_i)

                    sum_corr = 0.0
                    for j in range(1, n + 1):
                        a_j = (n - j + 1) ** (2 - alpha_i) - 2 * (n - j) ** (2 - alpha_i) + (n - j - 1) ** (2 - alpha_i)
                        sum_corr += a_j * f(self.t[j], x[j], params)[i]

                    x[n + 1, i] = x0[i] + (self.h ** alpha_i) / gamma(alpha_i + 2) * (
                        f(self.t[n + 1], x_pred_i, params)[i] + sum_corr
                    )

        return self.t, x

    def adams_bashforth_moulton(self, f, x0, params=None):
        x = np.zeros((self.N + 1, self.dim))
        x[0] = x0

        if np.isscalar(self.alpha):
            alpha = self.alpha
            coeffs = self.binomial_coeffs(self.N)

            for n in range(self.N):
                sum_hist = 0.0
                for j in range(n + 1):
                    sum_hist += coeffs[n - j] * f(self.t[j], x[j], params)

                x_pred = x0 + (self.h ** alpha) * sum_hist

                sum_corr = coeffs[0] * f(self.t[n + 1], x_pred, params)
                for j in range(1, n + 1):
                    sum_corr += coeffs[n + 1 - j] * f(self.t[j], x[j], params)

                x[n + 1] = x0 + (self.h ** alpha) * sum_corr
        else:
            coeffs_list = self.binomial_coeffs(self.N)

            for i in range(self.dim):
                alpha_i = self.alpha[i]
                coeffs = coeffs_list[i]

                for n in range(self.N):
                    sum_hist = 0.0
                    for j in range(n + 1):
                        sum_hist += coeffs[n - j] * f(self.t[j], x[j], params)[i]

                    x_pred_i = x0[i] + (self.h ** alpha_i) * sum_hist

                    sum_corr = coeffs[0] * f(self.t[n + 1], x_pred_i, params)[i]
                    for j in range(1, n + 1):
                        sum_corr += coeffs[n + 1 - j] * f(self.t[j], x[j], params)[i]

                    x[n + 1, i] = x0[i] + (self.h ** alpha_i) * sum_corr

        return self.t, x


def lotka_volterra(t, x, params):
    a, b, c, d = params
    dx = np.zeros_like(x)
    dx[0] = a * x[0] - b * x[0] * x[1]
    dx[1] = -c * x[1] + d * x[0] * x[1]
    return dx


def fractional_lotka_volterra():
    print("=" * 60)
    print("分数阶Lotka-Volterra捕食者-食饵模型")
    print("=" * 60)
    print()

    alpha = [0.9, 0.9]
    t_final = 50.0
    h = 0.01

    print(f"参数设置:")
    print(f"  分数阶数: α1 = {alpha[0]}, α2 = {alpha[1]}")
    print(f"  时间区间: [0, {t_final}]")
    print(f"  时间步长: h = {h}")
    print()

    params = [1.0, 1.0, 1.0, 1.0]
    x0 = [0.5, 0.5]

    print(f"Lotka-Volterra参数:")
    print(f"  食饵增长率: a = {params[0]}")
    print(f"  捕食率:      b = {params[1]}")
    print(f"  捕食者死亡率: c = {params[2]}")
    print(f"  转化率:      d = {params[3]}")
    print(f"  初始条件:     x0 = {x0}")
    print()

    solver = NonlinearFractionalSolver(alpha, t_final, h, dim=2)

    print("正在使用预测-校正方法求解...")
    t, x = solver.predictor_corrector(lotka_volterra, x0, params)
    print("求解完成!")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(t, x[:, 0], 'b-', label='食饵 (x)', linewidth=2)
    axes[0].plot(t, x[:, 1], 'r-', label='捕食者 (y)', linewidth=2)
    axes[0].set_xlabel('时间 t', fontsize=12)
    axes[0].set_ylabel('种群数量', fontsize=12)
    axes[0].set_title(f'时间演化 (α1={alpha[0]}, α2={alpha[1]})', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x[:, 0], x[:, 1], 'g-', linewidth=1.5, alpha=0.8)
    axes[1].plot(x[0, 0], x[0, 1], 'go', markersize=8, label='初始点')
    axes[1].set_xlabel('食饵数量 x', fontsize=12)
    axes[1].set_ylabel('捕食者数量 y', fontsize=12)
    axes[1].set_title('相空间轨迹', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lotka_volterra.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("结果图已保存: lotka_volterra.png")
    print()

    print("种群数量统计:")
    print(f"  食饵最大值: {np.max(x[:, 0]):.4f}")
    print(f"  食饵最小值: {np.min(x[:, 0]):.4f}")
    print(f"  捕食者最大值: {np.max(x[:, 1]):.4f}")
    print(f"  捕食者最小值: {np.min(x[:, 1]):.4f}")
    print()

    return t, x


def compare_different_alphas():
    print("=" * 60)
    print("不同分数阶数的对比")
    print("=" * 60)
    print()

    alphas_list = [0.7, 0.8, 0.9, 1.0]
    t_final = 30.0
    h = 0.01
    params = [1.0, 1.0, 1.0, 1.0]
    x0 = [0.5, 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, alpha_val in enumerate(alphas_list):
        alpha = [alpha_val, alpha_val]
        solver = NonlinearFractionalSolver(alpha, t_final, h, dim=2)
        t, x = solver.predictor_corrector(lotka_volterra, x0, params)

        ax = axes[idx]
        ax.plot(x[:, 0], x[:, 1], color=colors[idx], linewidth=1.5, label=f'α = {alpha_val}')
        ax.plot(x[0, 0], x[0, 1], 'o', color=colors[idx], markersize=6)
        ax.set_xlabel('食饵 x', fontsize=10)
        ax.set_ylabel('捕食者 y', fontsize=10)
        ax.set_title(f'α = {alpha_val}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle('不同分数阶数的相空间对比', fontsize=14)
    plt.tight_layout()
    plt.savefig('lv_different_alphas.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("不同α值对比图已保存: lv_different_alphas.png")
    print()


def fractional_vs_integer_order():
    print("=" * 60)
    print("分数阶 vs 整数阶对比")
    print("=" * 60)
    print()

    t_final = 40.0
    h = 0.01
    params = [1.0, 1.0, 1.0, 1.0]
    x0 = [0.5, 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    alpha_frac = [0.85, 0.85]
    solver_frac = NonlinearFractionalSolver(alpha_frac, t_final, h, dim=2)
    t_frac, x_frac = solver_frac.predictor_corrector(lotka_volterra, x0, params)

    alpha_int = [1.0, 1.0]
    solver_int = NonlinearFractionalSolver(alpha_int, t_final, h, dim=2)
    t_int, x_int = solver_int.predictor_corrector(lotka_volterra, x0, params)

    axes[0].plot(t_frac, x_frac[:, 0], 'b-', label='分数阶食饵', linewidth=2)
    axes[0].plot(t_frac, x_frac[:, 1], 'r-', label='分数阶捕食者', linewidth=2)
    axes[0].plot(t_int, x_int[:, 0], 'b--', label='整数阶食饵', linewidth=2, alpha=0.7)
    axes[0].plot(t_int, x_int[:, 1], 'r--', label='整数阶捕食者', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('时间 t', fontsize=12)
    axes[0].set_ylabel('种群数量', fontsize=12)
    axes[0].set_title('时间演化对比', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_frac[:, 0], x_frac[:, 1], 'g-', label='分数阶 (α=0.85)', linewidth=2)
    axes[1].plot(x_int[:, 0], x_int[:, 1], 'm--', label='整数阶 (α=1.0)', linewidth=2, alpha=0.7)
    axes[1].plot(x_frac[0, 0], x_frac[0, 1], 'go', markersize=8)
    axes[1].set_xlabel('食饵 x', fontsize=12)
    axes[1].set_ylabel('捕食者 y', fontsize=12)
    axes[1].set_title('相空间对比', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lv_frac_vs_int.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("分数阶-整数阶对比图已保存: lv_frac_vs_int.png")
    print()

    print("动力学行为差异:")
    print(f"  分数阶振荡周期更长，衰减更慢")
    print(f"  整数阶保持恒定振幅周期振荡")
    print()


def van_der_pol(t, x, params):
    mu, = params
    dx = np.zeros_like(x)
    dx[0] = x[1]
    dx[1] = mu * (1 - x[0]**2) * x[1] - x[0]
    return dx


def fractional_van_der_pol():
    print("=" * 60)
    print("分数阶Van der Pol振子")
    print("=" * 60)
    print()

    alpha = [0.95, 0.95]
    t_final = 100.0
    h = 0.01

    print(f"参数设置:")
    print(f"  分数阶数: α1 = {alpha[0]}, α2 = {alpha[1]}")
    print(f"  时间区间: [0, {t_final}]")
    print(f"  时间步长: h = {h}")
    print()

    params = [1.0]
    x0 = [0.1, 0.0]

    print(f"Van der Pol参数:")
    print(f"  非线性系数: μ = {params[0]}")
    print(f"  初始条件:     x0 = {x0}")
    print()

    solver = NonlinearFractionalSolver(alpha, t_final, h, dim=2)

    print("正在使用预测-校正方法求解...")
    t, x = solver.predictor_corrector(van_der_pol, x0, params)
    print("求解完成!")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(t, x[:, 0], 'b-', label='位移 x', linewidth=1.5)
    axes[0].set_xlabel('时间 t', fontsize=12)
    axes[0].set_ylabel('位移', fontsize=12)
    axes[0].set_title(f'时间演化 (α={alpha[0]})', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x[:, 0], x[:, 1], 'r-', linewidth=1, alpha=0.8)
    axes[1].plot(x[0, 0], x[0, 1], 'ro', markersize=8, label='初始点')
    axes[1].set_xlabel('位移 x', fontsize=12)
    axes[1].set_ylabel('速度 dx/dt', fontsize=12)
    axes[1].set_title('相空间轨迹 (极限环)', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('van_der_pol.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("结果图已保存: van_der_pol.png")
    print()

    return t, x


def lorenz(t, x, params):
    sigma, rho, beta = params
    dx = np.zeros_like(x)
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = x[0] * (rho - x[2]) - x[1]
    dx[2] = x[0] * x[1] - beta * x[2]
    return dx


def fractional_lorenz():
    print("=" * 60)
    print("分数阶Lorenz混沌系统")
    print("=" * 60)
    print()

    alpha = [0.99, 0.99, 0.99]
    t_final = 50.0
    h = 0.01

    print(f"参数设置:")
    print(f"  分数阶数: α = {alpha}")
    print(f"  时间区间: [0, {t_final}]")
    print(f"  时间步长: h = {h}")
    print()

    params = [10.0, 28.0, 8.0/3.0]
    x0 = [1.0, 1.0, 1.0]

    print(f"Lorenz参数:")
    print(f"  σ = {params[0]}, ρ = {params[1]}, β = {params[2]:.3f}")
    print(f"  初始条件: x0 = {x0}")
    print()

    solver = NonlinearFractionalSolver(alpha, t_final, h, dim=3)

    print("正在使用预测-校正方法求解...")
    t, x = solver.predictor_corrector(lorenz, x0, params)
    print("求解完成!")
    print()

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131)
    ax1.plot(x[:, 0], x[:, 1], 'b-', linewidth=0.7, alpha=0.8)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('x-y 平面', fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(132)
    ax2.plot(x[:, 1], x[:, 2], 'r-', linewidth=0.7, alpha=0.8)
    ax2.set_xlabel('y', fontsize=11)
    ax2.set_ylabel('z', fontsize=11)
    ax2.set_title('y-z 平面', fontsize=13)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(x[:, 0], x[:, 1], x[:, 2], 'g-', linewidth=0.5, alpha=0.8)
    ax3.set_xlabel('x', fontsize=10)
    ax3.set_ylabel('y', fontsize=10)
    ax3.set_zlabel('z', fontsize=10)
    ax3.set_title('3D 吸引子', fontsize=13)

    plt.tight_layout()
    plt.savefig('lorenz.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("结果图已保存: lorenz.png")
    print()

    return t, x


def main():
    print("非线性分数阶微分方程求解器")
    print("=" * 60)
    print()
    print("求解方法: 预测-校正法 (Predictor-Corrector)")
    print("  - Adams-Bashforth预测步")
    print("  - Adams-Moulton校正步")
    print()

    print("\n" + "=" * 60)
    print("示例1: 分数阶Lotka-Volterra捕食者-食饵模型")
    print("=" * 60 + "\n")
    fractional_lotka_volterra()
    compare_different_alphas()
    fractional_vs_integer_order()

    print("\n" + "=" * 60)
    print("示例2: 分数阶Van der Pol振子")
    print("=" * 60 + "\n")
    fractional_van_der_pol()

    print("\n" + "=" * 60)
    print("示例3: 分数阶Lorenz混沌系统")
    print("=" * 60 + "\n")
    fractional_lorenz()

    print("=" * 60)
    print("所有示例求解完成!")
    print("=" * 60)
    print()
    print("生成的图像文件:")
    print("  Lotka-Volterra模型:")
    print("    - lotka_volterra.png      - 基本结果")
    print("    - lv_different_alphas.png - 不同α值对比")
    print("    - lv_frac_vs_int.png       - 分数阶与整数阶对比")
    print()
    print("  Van der Pol振子:")
    print("    - van_der_pol.png          - 极限环振荡")
    print()
    print("  Lorenz混沌系统:")
    print("    - lorenz.png               - 混沌吸引子")
    print()


if __name__ == "__main__":
    main()
