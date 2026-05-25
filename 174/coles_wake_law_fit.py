import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy import stats
from scipy.integrate import quad

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def spalding_formula(y_plus, kappa=0.41, B=5.0, n_terms=5):
    """
    Spalding复合壁面律公式 - 描述内层（粘性+对数区）
    
    公式: κU+ = ln(1 + κy+) + κB + Σ [e^(-κn y+)·(1-e^(-κn y+))/n]
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


def coles_wake_law(eta, Pi=0.0):
    """
    Coles尾迹律函数 - 描述外层速度亏损
    
    参数:
    eta: y/δ - 无量纲距离
    Pi: 尾迹强度参数 (与压力梯度相关)
    
    返回:
    W(η): 尾迹函数值
    """
    return 2.0 * Pi * np.sin(np.pi * eta / 2.0)**2


def extended_spalding_coles(y_plus, eta, kappa=0.41, B=5.0, Pi=0.0):
    """
    扩展的Spalding-Coles公式 - 整合内层和外层
    
    U+ = Spalding内层 + (1/κ)·Coles尾迹律
    
    参数:
    y_plus: 无量纲壁面距离 (内层变量)
    eta: y/δ (外层变量)
    kappa: 卡门常数
    B: 积分常数
    Pi: 尾迹强度参数
    """
    U_inner = spalding_formula(y_plus, kappa, B)
    U_wake = (1.0 / kappa) * coles_wake_law(eta, Pi)
    
    return U_inner + U_wake


def velocity_defect_law(eta, Pi=0.0):
    """
    速度亏损律 - 用于描述边界层外层
    
    (U_e - U)/u_tau = -(1/κ)·ln(η) + Π·W(η)/κ
    
    参数:
    eta: y/δ
    Pi: 尾迹强度参数
    """
    kappa = 0.41
    
    U_defect = -(1.0/kappa) * np.log(eta) + (1.0/kappa) * coles_wake_law(eta, Pi)
    
    return U_defect


def generate_boundary_layer_with_pg(y, delta, u_tau, nu, kappa=0.41, B=5.0, 
                                     Pi=0.0, U_e=None, noise=0.02, seed=42):
    """
    生成带压力梯度效应的湍流边界层速度剖面
    
    参数:
    y: 法向距离数组
    delta: 边界层厚度
    u_tau: 摩擦速度
    nu: 运动粘度
    Pi: 尾迹强度参数 (正=逆压梯度, 负=顺压梯度)
    U_e: 外流速度 (如果为None，自动计算)
    """
    np.random.seed(seed)
    
    y_plus = y * u_tau / nu
    eta = y / delta
    eta = np.minimum(eta, 0.999)
    
    if U_e is None:
        eta_e = 0.999
        y_plus_e = eta_e * delta * u_tau / nu
        U_e_plus = extended_spalding_coles(y_plus_e, eta_e, kappa, B, Pi)
        U_e = U_e_plus * u_tau
    
    U_plus = extended_spalding_coles(y_plus, eta, kappa, B, Pi)
    U_plus += np.random.normal(0, noise, len(y))
    U = U_plus * u_tau
    
    return U, U_e


def estimate_boundary_layer_params(y, U, u_tau, nu, kappa=0.41, B=5.0):
    """
    估计边界层特征参数
    
    返回:
    delta: 边界层厚度 (U = 0.99U_e)
    delta_star: 位移厚度
    theta: 动量厚度
    H: 形状因子
    U_e: 外流速度
    Pi: 尾迹强度参数
    """
    U_e = np.max(U)
    U_e_idx = np.argmax(U)
    
    delta_idx = np.where(U >= 0.99 * U_e)[0]
    if len(delta_idx) > 0:
        delta = y[delta_idx[0]]
    else:
        delta = y[-1]
    
    y_norm = y / delta
    U_norm = U / U_e
    
    integrand_d_star = 1.0 - U_norm
    delta_star = np.trapz(integrand_d_star, y)
    
    integrand_theta = U_norm * (1.0 - U_norm)
    theta = np.trapz(integrand_theta, y)
    
    H = delta_star / theta
    
    eta = y / delta
    U_plus = U / u_tau
    U_inner_plus = spalding_formula(y * u_tau / nu, kappa, B)
    
    W_eta = coles_wake_law(np.minimum(eta, 0.999), 1.0)
    
    outer_mask = eta > 0.2
    if np.sum(outer_mask) > 5:
        U_wake = (U_plus - U_inner_plus) * kappa
        W_eta_masked = W_eta[outer_mask]
        U_wake_masked = U_wake[outer_mask]
        
        valid = np.abs(W_eta_masked) > 0.01
        if np.sum(valid) > 3:
            Pi = np.mean(U_wake_masked[valid] / W_eta_masked[valid])
        else:
            Pi = 0.0
    else:
        Pi = 0.0
    
    return {
        'delta': delta,
        'delta_star': delta_star,
        'theta': theta,
        'H': H,
        'U_e': U_e,
        'Pi': Pi
    }


def fit_with_pressure_gradient(y, U, nu, max_iter=20, tol=1e-6):
    """
    带压力梯度效应的边界层拟合
    
    同时优化: u_tau, kappa, B, Pi, delta
    """
    y = np.asarray(y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    
    U_e_est = np.max(U)
    
    mask_viscous = y <= np.percentile(y, 15)
    if len(y[mask_viscous]) >= 3:
        slope, _, _, _, _ = stats.linregress(y[mask_viscous], U[mask_viscous])
        u_tau_init = np.sqrt(slope * nu)
    else:
        u_tau_init = 0.001 * U_e_est
    u_tau_init = max(u_tau_init, 1e-5)
    
    delta_99_idx = np.where(U >= 0.99 * U_e_est)[0]
    delta_init = y[delta_99_idx[0]] if len(delta_99_idx) > 0 else y[-1]
    
    kappa = 0.41
    B = 5.0
    Pi = 0.0
    u_tau = u_tau_init
    delta = delta_init
    
    print(f"\n带压力梯度的迭代拟合:")
    print(f"{'迭代':>4} {'u_tau':>12} {'κ':>10} {'B':>10} {'Π':>10} {'δ(m)':>12} {'残差':>12}")
    print("-" * 75)
    
    prev_residual = np.inf
    
    for iteration in range(max_iter):
        y_plus = y * u_tau / nu
        eta = y / delta
        eta = np.minimum(eta, 0.999)
        
        def fit_params(params):
            k, b, p = params
            U_pred = extended_spalding_coles(y_plus, eta, k, b, p)
            U_data = U / u_tau
            return np.sum((U_data - U_pred)**2)
        
        res1 = minimize(fit_params, [kappa, B, Pi], 
                        bounds=[(0.3, 0.5), (3.0, 7.0), (-2.0, 5.0)],
                        method='L-BFGS-B')
        kappa, B, Pi = res1.x
        
        def fit_u_tau_delta(params):
            ut, d = params
            yp = y * ut / nu
            et = np.minimum(y / d, 0.999)
            Up_pred = extended_spalding_coles(yp, et, kappa, B, Pi)
            Up_data = U / ut
            return np.sum((Up_data - Up_pred)**2)
        
        res2 = minimize(fit_u_tau_delta, [u_tau, delta],
                        bounds=[(1e-5, 1.0), (y[1], y[-1] * 1.5)],
                        method='L-BFGS-B')
        u_tau, delta = res2.x
        
        y_plus = y * u_tau / nu
        eta = np.minimum(y / delta, 0.999)
        U_plus_pred = extended_spalding_coles(y_plus, eta, kappa, B, Pi)
        U_plus_data = U / u_tau
        residual = np.sqrt(np.mean((U_plus_data - U_plus_pred)**2))
        
        print(f"{iteration:>4} {u_tau:>12.6f} {kappa:>10.4f} {B:>10.4f} {Pi:>10.4f} {delta:>12.6f} {residual:>12.6f}")
        
        if abs(prev_residual - residual) < tol:
            print(f"\n收敛于 {iteration + 1} 次迭代")
            break
        prev_residual = residual
    
    params = estimate_boundary_layer_params(y, U, u_tau, nu, kappa, B)
    params['u_tau'] = u_tau
    params['kappa'] = kappa
    params['B_fit'] = B
    params['Pi_fit'] = Pi
    params['delta_fit'] = delta
    
    return params


def classify_pressure_gradient(Pi, H):
    """
    根据尾迹强度Π和形状因子H判断压力梯度类型
    
    Clauser平衡边界层判据:
    - 顺压梯度 (FPG): Π ≈ 0 (或负值), H < 1.4
    - 零压梯度 (ZPG): Π ≈ 0.55, H ≈ 1.4
    - 逆压梯度 (APG): Π > 0.6, H > 1.5
    - 强逆压梯度: Π > 1.0, H > 2.0
    """
    if Pi < 0.2 and H < 1.4:
        return "顺压梯度 (Favorable Pressure Gradient)", "green"
    elif Pi < 0.7 and H < 1.5:
        return "零压梯度/弱顺压 (Zero/Weak FPG)", "blue"
    elif Pi < 1.2 and H < 2.0:
        return "逆压梯度 (Adverse Pressure Gradient)", "orange"
    else:
        return "强逆压梯度 (Strong APG - 接近分离)", "red"


def plot_airfoil_boundary_layer(y_list, U_list, params_list, labels, nu):
    """
    绘制翼型表面不同位置的边界层速度剖面对比
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    ax = axes[0, 0]
    for i, (y, U, label) in enumerate(zip(y_list, U_list, labels)):
        color = ['blue', 'green', 'orange', 'red'][min(i, 3)]
        ax.plot(y * 1000, U, '-', label=label, color=color, linewidth=2)
    ax.set_xlabel('距离壁面 y (mm)')
    ax.set_ylabel('平均速度 U (m/s)')
    ax.set_title('翼型表面不同位置的速度剖面')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for i, (y, U, params, label) in enumerate(zip(y_list, U_list, params_list, labels)):
        u_tau = params['u_tau']
        y_plus = y * u_tau / nu
        U_plus = U / u_tau
        eta = y / params['delta']
        
        color = ['blue', 'green', 'orange', 'red'][min(i, 3)]
        ax.semilogx(y_plus, U_plus, '-', label=f"{label} (Π={params['Pi_fit']:.2f})", 
                    color=color, linewidth=1.5, alpha=0.8)
        
        y_plus_smooth = np.logspace(0, 3, 200)
        eta_smooth = y_plus_smooth * nu / (u_tau * params['delta'])
        eta_smooth = np.minimum(eta_smooth, 0.999)
        U_plus_fit = extended_spalding_coles(y_plus_smooth, eta_smooth, 
                                             params['kappa'], params['B_fit'], params['Pi_fit'])
        ax.semilogx(y_plus_smooth, U_plus_fit, '--', color=color, alpha=0.5)
    
    ax.semilogx(y_plus_smooth, spalding_formula(y_plus_smooth, 0.41, 5.0), 
                'k-', linewidth=2, label='内层律 (Π=0)')
    ax.set_xlabel('y+')
    ax.set_ylabel('U+')
    ax.set_title('壁面律表示 (半对数坐标)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 2000])
    
    ax = axes[0, 2]
    for i, (y, U, params, label) in enumerate(zip(y_list, U_list, params_list, labels)):
        eta = y / params['delta']
        U_defect = (params['U_e'] - U) / params['u_tau']
        
        color = ['blue', 'green', 'orange', 'red'][min(i, 3)]
        ax.plot(eta, U_defect, '-', label=f"{label} (H={params['H']:.2f})", 
                color=color, linewidth=1.5, alpha=0.8)
        
        eta_smooth = np.linspace(0.001, 0.999, 200)
        U_defect_fit = velocity_defect_law(eta_smooth, params['Pi_fit'])
        ax.plot(eta_smooth, U_defect_fit, '--', color=color, alpha=0.5)
    
    ax.set_xlabel('η = y/δ')
    ax.set_ylabel('(U_e - U)/u_τ')
    ax.set_title('速度亏损律')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    positions = np.arange(len(params_list))
    Pi_values = [p['Pi_fit'] for p in params_list]
    H_values = [p['H'] for p in params_list]
    
    ax.bar(positions - 0.2, Pi_values, 0.4, label='尾迹强度 Π', color='steelblue', alpha=0.8)
    ax.bar(positions + 0.2, [h/2 for h in H_values], 0.4, label='形状因子 H/2', color='coral', alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, fontsize=9)
    ax.set_ylabel('参数值')
    ax.set_title('边界层特征参数沿翼型表面的变化')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    ax.axhline(y=0.55, color='blue', linestyle='--', alpha=0.5, label='ZPG基准 (Π≈0.55)')
    ax.axhline(y=1.4, color='blue', linestyle=':', alpha=0.5, label='ZPG基准 (H≈1.4)')
    
    ax.fill_between([-1, 5], 0, 0.3, alpha=0.1, color='green', label='FPG区')
    ax.fill_between([-1, 5], 0.3, 0.8, alpha=0.1, color='yellow', label='ZPG区')
    ax.fill_between([-1, 5], 0.8, 3.0, alpha=0.1, color='red', label='APG区')
    
    for i, (params, label) in enumerate(zip(params_list, labels)):
        pg_type, color = classify_pressure_gradient(params['Pi_fit'], params['H'])
        ax.plot(i, params['Pi_fit'], 'o', color=color, markersize=15, label=f'{label}')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, fontsize=9)
    ax.set_ylabel('尾迹强度 Π')
    ax.set_title('压力梯度类型判断')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.5, len(params_list) - 0.5])
    ax.set_ylim([-0.5, 2.5])
    
    ax = axes[1, 2]
    for i, (y, U, params, label) in enumerate(zip(y_list, U_list, params_list, labels)):
        u_tau = params['u_tau']
        y_plus = y * u_tau / nu
        eta = y / params['delta']
        
        U_plus_inner = spalding_formula(y_plus, params['kappa'], params['B_fit'])
        U_plus_wake = (1.0/params['kappa']) * coles_wake_law(np.minimum(eta, 0.999), params['Pi_fit'])
        U_plus_total = U_plus_inner + U_plus_wake
        
        residual = U / u_tau - U_plus_total
        
        color = ['blue', 'green', 'orange', 'red'][min(i, 3)]
        ax.semilogx(y_plus, residual, '-', label=label, color=color, linewidth=1.5, alpha=0.7)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=5, color='k', linestyle=':', alpha=0.5, label='y+=5')
    ax.axvline(x=30, color='k', linestyle='--', alpha=0.5, label='y+=30')
    ax.set_xlabel('y+')
    ax.set_ylabel('残差 (U+ - 拟合值)')
    ax.set_title('Spalding-Coles拟合残差')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 2000])
    
    plt.tight_layout()
    plt.savefig('airfoil_boundary_layer_analysis.png', dpi=150, bbox_inches='tight')
    print("\n翼型边界层分析图已保存到: airfoil_boundary_layer_analysis.png")
    
    return fig


def main():
    np.random.seed(42)
    
    print("=" * 80)
    print("Coles尾迹律 - 带压力梯度的湍流边界层拟合".center(80))
    print("=" * 80)
    
    nu = 1.5e-5
    n_points = 200
    y = np.logspace(-6, -3, n_points)
    
    print("\n" + "=" * 80)
    print("模拟翼型表面不同位置的边界层".center(80))
    print("=" * 80)
    
    configs = [
        {'name': '前缘驻点', 'u_tau': 0.08, 'delta': 0.005, 'Pi': -0.3, 'kappa': 0.41, 'B': 5.0},
        {'name': '翼型前部 (FPG)', 'u_tau': 0.06, 'delta': 0.008, 'Pi': 0.1, 'kappa': 0.41, 'B': 5.0},
        {'name': '翼型中部 (ZPG)', 'u_tau': 0.05, 'delta': 0.012, 'Pi': 0.55, 'kappa': 0.41, 'B': 5.0},
        {'name': '翼型后部 (APG)', 'u_tau': 0.035, 'delta': 0.020, 'Pi': 1.2, 'kappa': 0.41, 'B': 5.0},
    ]
    
    y_list = []
    U_list = []
    params_list = []
    
    for i, config in enumerate(configs):
        U, U_e = generate_boundary_layer_with_pg(
            y, config['delta'], config['u_tau'], nu,
            kappa=config['kappa'], B=config['B'], Pi=config['Pi'],
            noise=0.02
        )
        
        y_list.append(y)
        U_list.append(U)
        
        print(f"\n{'='*60}")
        print(f"{config['name']}:")
        print(f"{'='*60}")
        print(f"  真实参数: u_tau={config['u_tau']:.4f}, δ={config['delta']:.4f}, Π={config['Pi']:.2f}")
        
        params = fit_with_pressure_gradient(y, U, nu, max_iter=15)
        params_list.append(params)
        
        pg_type, _ = classify_pressure_gradient(params['Pi_fit'], params['H'])
        
        print(f"\n  拟合结果:")
        print(f"    u_tau = {params['u_tau']:.6f} m/s (误差: {abs(params['u_tau']-config['u_tau'])/config['u_tau']*100:.2f}%)")
        print(f"    κ = {params['kappa']:.4f}")
        print(f"    B = {params['B_fit']:.4f}")
        print(f"    Π = {params['Pi_fit']:.4f} (真实值: {config['Pi']})")
        print(f"    δ = {params['delta_fit']:.6f} m (真实值: {config['delta']})")
        print(f"    δ* = {params['delta_star']:.6f} m")
        print(f"    θ = {params['theta']:.6f} m")
        print(f"    H = {params['H']:.4f}")
        print(f"    U_e = {params['U_e']:.4f} m/s")
        print(f"    压力梯度类型: {pg_type}")
    
    print("\n" + "=" * 80)
    print("生成翼型边界层分析图表...")
    
    labels = [c['name'] for c in configs]
    fig = plot_airfoil_boundary_layer(y_list, U_list, params_list, labels, nu)
    
    print("\n" + "=" * 80)
    print("Coles尾迹律理论说明".center(80))
    print("=" * 80)
    print("""
Coles尾迹律公式:
  U+ = (1/κ)·ln(y+) + B + (2Π/κ)·sin²(πη/2)
  
  其中:
    - 内层: Spalding公式 (描述粘性子层和对数律区)
    - 外层: Coles尾迹律 (描述边界层外缘的速度亏损)
    - Π: 尾迹强度参数，与压力梯度相关
    - η = y/δ: 外层无量纲坐标

压力梯度与边界层特征的关系:
  - 顺压梯度 (FPG): Π < 0.3, H < 1.4 (加速流动)
  - 零压梯度 (ZPG): Π ≈ 0.55, H ≈ 1.4 (标准平板)
  - 逆压梯度 (APG): Π > 0.6, H > 1.5 (减速流动)
  - 强逆压梯度: Π > 1.0, H > 2.0 (接近分离)

应用场景:
  - 翼型表面边界层分析
  - 叶片通道内的湍流流动
  - 管道扩压器设计
  - 飞行器气动设计
""")
    
    plt.show()


if __name__ == "__main__":
    main()
