import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar, minimize
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_turbulent_data(n_points=150, u_tau=0.05, nu=1e-6, kappa=0.41, B=5.0, noise=0.02, seed=42):
    """生成湍流边界层速度剖面示例数据"""
    np.random.seed(seed)
    y = np.logspace(-7, -2, n_points)
    
    y_plus = y * u_tau / nu
    U_plus = spalding_formula(y_plus, kappa, B)
    U_plus += np.random.normal(0, noise, n_points)
    U = U_plus * u_tau
    
    return y, U


def spalding_formula(y_plus, kappa=0.41, B=5.0, n_terms=5):
    """
    Spalding复合壁面律公式 - 描述从粘性子层到对数律区的全域
    
    公式: κU+ = ln(1 + κy+) + Σ [e^(-κn y+) * (1 - e^(-κn y+)) / n]  for n=1,2,...
    
    参数:
    y_plus: 无量纲壁面距离
    kappa: 卡门常数
    B: 积分常数
    n_terms: 级数展开项数
    
    返回:
    U_plus: 无量纲速度
    """
    y_plus = np.asarray(y_plus, dtype=np.float64)
    kappa = float(kappa)
    
    ln_term = np.log(1.0 + kappa * y_plus)
    
    sum_term = np.zeros_like(y_plus)
    for n in range(1, n_terms + 1):
        arg = kappa * n * y_plus
        sum_term += np.exp(-arg) * (1.0 - np.exp(-arg)) / n
    
    kappa_U_plus = ln_term + kappa * B + sum_term
    U_plus = kappa_U_plus / kappa
    
    return U_plus


def spalding_formula_simple(y_plus, kappa=0.41, B=5.0, A=26.0):
    """
    简化版Spalding公式 - 使用混合函数平滑过渡
    
    参数:
    A: 过渡区域形状参数
    """
    y_plus = np.asarray(y_plus, dtype=np.float64)
    
    Gamma = y_plus / A
    
    viscous_term = y_plus
    log_term = (1.0/kappa) * np.log(1.0 + kappa * y_plus) + B
    
    blend = 1.0 - np.exp(-Gamma) - Gamma * np.exp(-Gamma)
    
    U_plus = viscous_term * (1.0 - blend) + log_term * blend
    
    return U_plus


def spalding_roughness_corrected(y_plus, kappa=0.41, B=5.0, delta_roughness_plus=0.0):
    """
    考虑粗糙度修正的Spalding公式
    
    参数:
    delta_roughness_plus: 等效粗糙度位移 (y+单位)
    """
    y_plus_eff = np.maximum(y_plus - delta_roughness_plus, 1e-6)
    return spalding_formula(y_plus_eff, kappa, B)


def iterative_spalding_fit(y, U, nu, u_tau_init=None, max_iter=20, tol=1e-6):
    """
    迭代拟合Spalding公式 - 同时优化u_tau, kappa, B
    
    算法:
    1. 初始估计u_tau
    2. 固定u_tau, 拟合kappa和B
    3. 固定kappa和B, 优化u_tau
    4. 重复直到收敛
    """
    y = np.asarray(y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    
    if u_tau_init is None:
        u_tau_init = estimate_u_tau_viscous(y, U, nu)
    
    u_tau = u_tau_init
    kappa = 0.41
    B = 5.0
    
    print(f"\n迭代拟合Spalding公式:")
    print(f"{'迭代':>4} {'u_tau':>12} {'κ':>10} {'B':>10} {'残差':>12}")
    print("-" * 55)
    
    prev_residual = np.inf
    
    for iteration in range(max_iter):
        y_plus = y * u_tau / nu
        U_plus = U / u_tau
        
        def fit_kappa_B(params):
            k, b = params
            U_plus_pred = spalding_formula(y_plus, k, b)
            return np.sum((U_plus - U_plus_pred)**2)
        
        res1 = minimize(fit_kappa_B, [kappa, B], bounds=[(0.3, 0.5), (3.0, 7.0)], method='L-BFGS-B')
        kappa, B = res1.x
        
        def fit_u_tau(ut):
            yp = y * ut / nu
            Up = U / ut
            Up_pred = spalding_formula(yp, kappa, B)
            return np.sum((Up - Up_pred)**2)
        
        res2 = minimize(fit_u_tau, [u_tau], bounds=[(1e-4, 1.0)], method='L-BFGS-B')
        u_tau = res2.x[0]
        
        y_plus = y * u_tau / nu
        U_plus = U / u_tau
        U_plus_pred = spalding_formula(y_plus, kappa, B)
        residual = np.sqrt(np.mean((U_plus - U_plus_pred)**2))
        
        print(f"{iteration:>4} {u_tau:>12.6f} {kappa:>10.4f} {B:>10.4f} {residual:>12.6f}")
        
        if abs(prev_residual - residual) < tol:
            print(f"\n收敛于 {iteration + 1} 次迭代")
            break
        prev_residual = residual
    
    return u_tau, kappa, B


def estimate_u_tau_viscous(y, U, nu):
    """通过粘性子层线性关系估计初始u_tau"""
    mask = y <= np.percentile(y, 20)
    y_sub = y[mask]
    U_sub = U[mask]
    
    if len(y_sub) < 3:
        mask = np.arange(len(y))[:len(y)//4]
        y_sub = y[mask]
        U_sub = U[mask]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_sub, U_sub)
    u_tau = np.sqrt(slope * nu)
    
    return max(u_tau, 1e-4)


def traditional_segmented_fit(y, U, nu, y_plus_log_min=30):
    """传统分段拟合方法 - 用于对比"""
    u_tau = estimate_u_tau_viscous(y, U, nu)
    
    y_plus = y * u_tau / nu
    U_plus = U / u_tau
    
    log_mask = y_plus >= y_plus_log_min
    if np.sum(log_mask) < 5:
        log_mask = y >= np.percentile(y, 30)
    
    y_plus_log = y_plus[log_mask]
    U_plus_log = U_plus[log_mask]
    
    ln_y_plus = np.log(y_plus_log)
    slope, intercept, r_value, p_value, std_err = stats.linregress(ln_y_plus, U_plus_log)
    
    kappa = 1.0 / slope
    B = intercept
    
    return u_tau, kappa, B, log_mask


def fit_with_roughness_iterative(y, U, nu, max_iter=15):
    """考虑粗糙度的迭代拟合"""
    u_tau, kappa, B = iterative_spalding_fit(y, U, nu, max_iter=10)
    
    y_plus = y * u_tau / nu
    U_plus = U / u_tau
    
    def fit_roughness(params):
        k, b, dr = params
        U_plus_pred = spalding_roughness_corrected(y_plus, k, b, dr)
        return np.sum((U_plus - U_plus_pred)**2)
    
    res = minimize(fit_roughness, [kappa, B, 0.0], 
                   bounds=[(0.3, 0.5), (3.0, 7.0), (-10, 10)], 
                   method='L-BFGS-B')
    
    kappa, B, delta_roughness_plus = res.x
    
    return u_tau, kappa, B, delta_roughness_plus


def plot_comparison(y, U, nu, u_tau_spalding, kappa_spalding, B_spalding,
                    u_tau_trad, kappa_trad, B_trad, log_mask_trad):
    """绘制Spalding拟合与传统分段拟合的对比图"""
    
    y_plus_s = y * u_tau_spalding / nu
    U_plus_s = U / u_tau_spalding
    
    y_plus_t = y * u_tau_trad / nu
    U_plus_t = U / u_tau_trad
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx(y_plus_s, U_plus_s, 'o', label='数据点', alpha=0.6, markersize=5, color='gray')
    
    y_plus_smooth = np.logspace(0, 3.5, 200)
    
    ax1.semilogx(y_plus_smooth, spalding_formula(y_plus_smooth, kappa_spalding, B_spalding),
                 'r-', linewidth=2.5, label=f'Spalding拟合 (κ={kappa_spalding:.3f}, B={B_spalding:.3f})')
    
    ax1.semilogx(y_plus_smooth, y_plus_smooth, 'k:', alpha=0.5, label='粘性子层 U+ = y+')
    ax1.semilogx(y_plus_smooth, (1/kappa_spalding)*np.log(y_plus_smooth) + B_spalding,
                 'k--', alpha=0.5, label='对数律渐近线')
    
    ax1.set_xlabel('y+')
    ax1.set_ylabel('U+')
    ax1.set_title(f'Spalding全域拟合 (u_tau={u_tau_spalding:.6f} m/s)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, 3000])
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx(y_plus_t, U_plus_t, 'o', alpha=0.6, markersize=5, color='gray', label='数据点')
    ax2.semilogx(y_plus_t[log_mask_trad], U_plus_t[log_mask_trad], 'o', 
                 alpha=0.8, markersize=6, color='blue', label='对数区拟合点')
    
    ax2.semilogx(y_plus_smooth, y_plus_smooth, 'k:', alpha=0.5, label='粘性子层')
    ax2.semilogx(y_plus_smooth, (1/kappa_trad)*np.log(y_plus_smooth) + B_trad,
                 'g-', linewidth=2.5, label=f'对数律拟合 (κ={kappa_trad:.3f}, B={B_trad:.3f})')
    
    ax2.axvline(30, color='r', linestyle='--', alpha=0.7, label='y+ = 30 (主观阈值)')
    
    ax2.set_xlabel('y+')
    ax2.set_ylabel('U+')
    ax2.set_title(f'传统分段拟合 (u_tau={u_tau_trad:.6f} m/s)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, 3000])
    
    ax3 = fig.add_subplot(gs[0, 2])
    U_plus_pred_s = spalding_formula(y_plus_s, kappa_spalding, B_spalding)
    residuals_s = U_plus_s - U_plus_pred_s
    
    U_plus_pred_t = (1/kappa_trad) * np.log(y_plus_t) + B_trad
    residuals_t = U_plus_t - U_plus_pred_t
    residuals_t[~log_mask_trad] = np.nan
    
    ax3.semilogx(y_plus_s, residuals_s, 'o', alpha=0.6, markersize=5, color='red', 
                 label=f'Spalding全域 (RMSE={np.sqrt(np.nanmean(residuals_s**2)):.3f})')
    ax3.semilogx(y_plus_t, residuals_t, 'o', alpha=0.6, markersize=5, color='green',
                 label=f'对数律分段 (RMSE={np.sqrt(np.nanmean(residuals_t**2)):.3f})')
    
    ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(5, color='k', linestyle=':', alpha=0.5)
    ax3.axvline(30, color='r', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('y+')
    ax3.set_ylabel('残差 U+')
    ax3.set_title('拟合残差对比')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([1, 3000])
    
    ax4 = fig.add_subplot(gs[1, 0:2])
    
    y_plus_thresholds = [15, 20, 25, 30, 40, 50, 70, 100]
    kappas_threshold = []
    Bs_threshold = []
    
    for yp_min in y_plus_thresholds:
        _, k, b, _ = traditional_segmented_fit(y, U, nu, y_plus_log_min=yp_min)
        kappas_threshold.append(k)
        Bs_threshold.append(b)
    
    ax4.plot(y_plus_thresholds, kappas_threshold, 'o-', color='blue', linewidth=2, label='κ (卡门常数)')
    ax4.axhline(kappa_spalding, color='red', linestyle='--', linewidth=2, 
                label=f'κ (Spalding全域) = {kappa_spalding:.4f}')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(y_plus_thresholds, Bs_threshold, 's-', color='green', linewidth=2, label='B (积分常数)')
    ax4_twin.axhline(B_spalding, color='red', linestyle=':', linewidth=2,
                     label=f'B (Spalding全域) = {B_spalding:.4f}')
    
    ax4.set_xlabel('对数律起始 y+ 阈值')
    ax4.set_ylabel('κ', color='blue')
    ax4_twin.set_ylabel('B', color='green')
    ax4.set_title('传统分段拟合: 拟合参数对y+阈值的敏感性')
    ax4.grid(True, alpha=0.3)
    
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)
    
    ax5 = fig.add_subplot(gs[1, 2])
    
    u_tau_values = np.linspace(u_tau_spalding * 0.8, u_tau_spalding * 1.2, 50)
    rmse_values = []
    
    for ut in u_tau_values:
        yp = y * ut / nu
        Up = U / ut
        Up_pred = spalding_formula(yp, kappa_spalding, B_spalding)
        rmse = np.sqrt(np.mean((Up - Up_pred)**2))
        rmse_values.append(rmse)
    
    ax5.plot(u_tau_values, rmse_values, 'b-', linewidth=2)
    ax5.axvline(u_tau_spalding, color='r', linestyle='--', 
                label=f'最优 u_tau = {u_tau_spalding:.6f}')
    ax5.set_xlabel('摩擦速度 u_tau (m/s)')
    ax5.set_ylabel('RMSE (U+)')
    ax5.set_title('Spalding拟合: u_tau优化曲线')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('spalding_vs_traditional_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比图已保存到: spalding_vs_traditional_comparison.png")
    
    return fig


def print_comparison_results(u_tau_true, kappa_true, B_true,
                             u_tau_spalding, kappa_spalding, B_spalding,
                             u_tau_trad, kappa_trad, B_trad):
    """打印拟合结果对比"""
    
    print("\n" + "=" * 80)
    print("拟合结果对比".center(80))
    print("=" * 80)
    print(f"{'参数':<15} {'真实值':<15} {'Spalding全域':<20} {'传统分段':<20}")
    print("-" * 80)
    
    print(f"{'摩擦速度 u_tau':<15} {u_tau_true:<15.6f} "
          f"{u_tau_spalding:<20.6f} {u_tau_trad:<20.6f}")
    print(f"{'卡门常数 κ':<15} {kappa_true:<15.4f} "
          f"{kappa_spalding:<20.4f} {kappa_trad:<20.4f}")
    print(f"{'积分常数 B':<15} {B_true:<15.4f} "
          f"{B_spalding:<20.4f} {B_trad:<20.4f}")
    
    print("-" * 80)
    print(f"{'相对误差(κ)':<15} {'-':<15} "
          f"{abs(kappa_spalding-kappa_true)/kappa_true*100:<19.2f}% "
          f"{abs(kappa_trad-kappa_true)/kappa_true*100:<19.2f}%")
    print(f"{'相对误差(B)':<15} {'-':<15} "
          f"{abs(B_spalding-B_true)/B_true*100:<19.2f}% "
          f"{abs(B_trad-B_true)/B_true*100:<19.2f}%")
    print("=" * 80)


def main():
    np.random.seed(42)
    
    print("=" * 80)
    print("Spalding复合曲线 - 湍流边界层全域拟合".center(80))
    print("=" * 80)
    
    u_tau_true = 0.05
    nu = 1e-6
    kappa_true = 0.41
    B_true = 5.0
    
    print(f"\n真实参数:")
    print(f"  摩擦速度 u_tau = {u_tau_true} m/s")
    print(f"  卡门常数 κ = {kappa_true}")
    print(f"  积分常数 B = {B_true}")
    
    y, U = generate_turbulent_data(n_points=150, u_tau=u_tau_true, nu=nu,
                                   kappa=kappa_true, B=B_true, noise=0.025)
    
    print(f"\n生成数据点: {len(y)} 个")
    print(f"y 范围: [{y.min():.2e}, {y.max():.2e}] m")
    
    print("\n" + "-" * 80)
    print("方法1: Spalding全域迭代拟合 (无主观阈值)")
    print("-" * 80)
    u_tau_spalding, kappa_spalding, B_spalding = iterative_spalding_fit(y, U, nu)
    
    print("\n" + "-" * 80)
    print("方法2: 传统分段拟合 (需要主观选择y+阈值)")
    print("-" * 80)
    u_tau_trad, kappa_trad, B_trad, log_mask_trad = traditional_segmented_fit(y, U, nu)
    
    print_comparison_results(u_tau_true, kappa_true, B_true,
                             u_tau_spalding, kappa_spalding, B_spalding,
                             u_tau_trad, kappa_trad, B_trad)
    
    print("\n" + "=" * 80)
    print("粗糙度分析 (模拟粗糙壁面)".center(80))
    print("=" * 80)
    
    y_rough, U_rough = generate_turbulent_data(n_points=150, u_tau=u_tau_true, nu=nu,
                                               kappa=kappa_true, B=B_true - 1.0, noise=0.025)
    
    print(f"\n生成粗糙壁面数据 (B减小以模拟粗糙度效应)")
    u_tau_r, kappa_r, B_r, delta_rough = fit_with_roughness_iterative(y_rough, U_rough, nu)
    
    print(f"\n粗糙壁面拟合结果:")
    print(f"  u_tau = {u_tau_r:.6f} m/s")
    print(f"  κ = {kappa_r:.4f}")
    print(f"  B = {B_r:.4f}")
    print(f"  等效粗糙度 Δy+ = {delta_rough:.4f}")
    if delta_rough > 0.5:
        print(f"  → 检测到显著壁面粗糙度效应")
    else:
        print(f"  → 壁面接近水力光滑")
    
    print("\n" + "-" * 80)
    print("生成对比图表...")
    fig = plot_comparison(y, U, nu,
                          u_tau_spalding, kappa_spalding, B_spalding,
                          u_tau_trad, kappa_trad, B_trad, log_mask_trad)
    
    print("\n" + "=" * 80)
    print("核心优势总结".center(80))
    print("=" * 80)
    print("Spalding复合曲线拟合:")
    print("  ✓ 无需主观选择y+阈值")
    print("  ✓ 利用整个边界层数据点")
    print("  ✓ 同时优化u_tau, κ, B")
    print("  ✓ 结果更稳定，对数据噪声鲁棒性更强")
    print("  ✓ 便于扩展到粗糙壁面等复杂情况")
    print("=" * 80)
    
    plt.show()


if __name__ == "__main__":
    main()
