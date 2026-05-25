import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit


def generate_soft_tissue_data(K=5000.0, B=50.0, n=1.5, m=0.5, 
                               noise_level=0.5, num_points=100,
                               outlier_ratio=0.05):
    """
    生成软组织交互的模拟数据（力-位移-速度）
    使用Hunt-Crossley模型: F = K*x^n + B*x^m*v
    
    参数:
        K: 非线性刚度系数 (N/m^n)
        B: 阻尼系数 (N·s/m^(m+1))
        n: 刚度非线性指数 (通常1.0-2.0，软组织约1.5)
        m: 阻尼非线性指数 (通常 = n-1)
        noise_level: 噪声水平 (N)
        num_points: 数据点数量
        outlier_ratio: 离群值比例
    
    返回:
        displacement: 位移数组 (m)
        velocity: 速度数组 (m/s)
        force: 力数组 (N)
        true_params: 真实参数字典
    """
    t = np.linspace(0, 2.0, num_points)
    
    freq = 0.5
    amplitude = 0.01
    displacement = amplitude * (1 - np.cos(2 * np.pi * freq * t))
    velocity = amplitude * 2 * np.pi * freq * np.sin(2 * np.pi * freq * t)
    
    displacement = np.maximum(displacement, 0)
    
    force_elastic = K * np.power(displacement, n)
    force_damping = B * np.power(np.maximum(displacement, 1e-10), m) * velocity
    force_true = force_elastic + force_damping
    
    noise = np.random.normal(0, noise_level, num_points)
    force = force_true + noise
    
    num_outliers = int(num_points * outlier_ratio)
    if num_outliers > 0:
        outlier_idx = np.random.choice(num_points, num_outliers, replace=False)
        force[outlier_idx] += np.random.uniform(-5*noise_level, 5*noise_level, num_outliers)
    
    true_params = {'K': K, 'B': B, 'n': n, 'm': m}
    
    return displacement, velocity, force, true_params


def hunt_crossley_force(x, v, K, B, n, m):
    """
    Hunt-Crossley接触力模型
    
    F = K*x^n + B*x^m*v
    
    参数:
        x: 位移 (m)
        v: 速度 (m/s)
        K: 刚度系数
        B: 阻尼系数
        n: 刚度非线性指数
        m: 阻尼非线性指数
    
    返回:
        接触力 (N)
    """
    x_pos = np.maximum(x, 0)
    force_elastic = K * np.power(x_pos, n)
    force_damping = B * np.power(x_pos, m) * v
    return force_elastic + force_damping


def linear_force(x, v, K, B):
    """
    线性Kelvin-Voigt模型
    
    F = K*x + B*v
    
    参数:
        x: 位移 (m)
        v: 速度 (m/s)
        K: 刚度系数 (N/m)
        B: 阻尼系数 (N·s/m)
    
    返回:
        接触力 (N)
    """
    return K * x + B * v


def residual_linear(params, x, v, F):
    """线性模型残差"""
    K, B = params
    return F - linear_force(x, v, K, B)


def residual_hunt_crossley(params, x, v, F):
    """Hunt-Crossley模型残差"""
    K, B, n, m = params
    return F - hunt_crossley_force(x, v, K, B, n, m)


def residual_hunt_crossley_fixed_m(params, x, v, F, m_fixed):
    """固定m值的Hunt-Crossley模型残差"""
    K, B, n = params
    return F - hunt_crossley_force(x, v, K, B, n, m_fixed)


def identify_linear_model(x, v, F):
    """
    辨识线性Kelvin-Voigt模型: F = K*x + B*v
    
    参数:
        x: 位移数组 (m)
        v: 速度数组 (m/s)
        F: 力数组 (N)
    
    返回:
        K: 刚度系数 (N/m)
        B: 阻尼系数 (N·s/m)
        r_squared: 决定系数
        F_pred: 预测力
    """
    X = np.column_stack([x, v])
    params, residuals, rank, s = np.linalg.lstsq(X, F, rcond=None)
    K, B = params
    
    F_pred = linear_force(x, v, K, B)
    ss_res = np.sum((F - F_pred) ** 2)
    ss_tot = np.sum((F - np.mean(F)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    return K, B, r_squared, F_pred


def identify_hunt_crossley(x, v, F, fix_m=True, m_fixed=None, 
                           K0=1000.0, B0=10.0, n0=1.5, m0=0.5):
    """
    辨识Hunt-Crossley非线性模型
    
    参数:
        x: 位移数组 (m)
        v: 速度数组 (m/s)
        F: 力数组 (N)
        fix_m: 是否固定阻尼指数m (通常m = n-1)
        m_fixed: 固定的m值（如果fix_m=True），默认使用n-1
        K0, B0, n0, m0: 初始猜测值
    
    返回:
        params: 辨识得到的参数 [K, B, n, m]
        r_squared: 决定系数
        F_pred: 预测力
    """
    if fix_m:
        def model_wrapper(xv, K, B, n):
            x, v = xv
            m = n - 1 if m_fixed is None else m_fixed
            return hunt_crossley_force(x, v, K, B, n, m)
        
        xv_data = np.vstack([x, v])
        try:
            popt, pcov = curve_fit(model_wrapper, xv_data, F, 
                                   p0=[K0, B0, n0], 
                                   bounds=([100, 0, 0.5], [1e6, 1000, 3.0]))
            K, B, n = popt
            m = n - 1 if m_fixed is None else m_fixed
            params = [K, B, n, m]
        except:
            params = [K0, B0, n0, n0-1 if m_fixed is None else m_fixed]
    else:
        def model_wrapper(xv, K, B, n, m):
            x, v = xv
            return hunt_crossley_force(x, v, K, B, n, m)
        
        xv_data = np.vstack([x, v])
        try:
            popt, pcov = curve_fit(model_wrapper, xv_data, F, 
                                   p0=[K0, B0, n0, m0],
                                   bounds=([100, 0, 0.5, 0], [1e6, 1000, 3.0, 2.0]))
            params = popt
        except:
            params = [K0, B0, n0, m0]
    
    K, B, n, m = params
    F_pred = hunt_crossley_force(x, v, K, B, n, m)
    ss_res = np.sum((F - F_pred) ** 2)
    ss_tot = np.sum((F - np.mean(F)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    return params, r_squared, F_pred


def identify_robust_hunt_crossley(x, v, F, fix_m=True, m_fixed=None,
                                   K0=1000.0, B0=10.0, n0=1.5, m0=0.5,
                                   residual_threshold=1.0, max_trials=200):
    """
    使用RANSAC鲁棒辨识Hunt-Crossley模型
    
    参数:
        x, v, F: 位移、速度、力数组
        fix_m: 是否固定m = n-1
        residual_threshold: 残差阈值 (N)
        max_trials: 最大迭代次数
    
    返回:
        params: 辨识参数 [K, B, n, m]
        r_squared: 决定系数
        F_pred: 预测力
        inlier_mask: 内点掩码
    """
    n_points = len(x)
    min_samples = 10
    best_inlier_count = 0
    best_params = None
    best_inlier_mask = np.ones(n_points, dtype=bool)
    
    for _ in range(max_trials):
        sample_idx = np.random.choice(n_points, min_samples, replace=False)
        x_sample = x[sample_idx]
        v_sample = v[sample_idx]
        F_sample = F[sample_idx]
        
        try:
            if fix_m:
                def model_wrapper(xv, K, B, n):
                    x, v = xv
                    m = n - 1 if m_fixed is None else m_fixed
                    return hunt_crossley_force(x, v, K, B, n, m)
                
                xv_data = np.vstack([x_sample, v_sample])
                popt, _ = curve_fit(model_wrapper, xv_data, F_sample,
                                   p0=[K0, B0, n0],
                                   bounds=([100, 0, 0.5], [1e6, 1000, 3.0]))
                K, B, n = popt
                m = n - 1 if m_fixed is None else m_fixed
                params = [K, B, n, m]
            else:
                params, _, _ = identify_hunt_crossley(
                    x_sample, v_sample, F_sample, fix_m=False, 
                    K0=K0, B0=B0, n0=n0, m0=m0)
            
            K, B, n, m = params
            F_pred_all = hunt_crossley_force(x, v, K, B, n, m)
            residuals = np.abs(F - F_pred_all)
            inlier_mask = residuals <= residual_threshold
            inlier_count = np.sum(inlier_mask)
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask = inlier_mask.copy()
                best_params = params
        except:
            continue
    
    if best_params is None:
        return identify_hunt_crossley(x, v, F, fix_m, m_fixed, K0, B0, n0, m0) + (np.ones(n_points, dtype=bool),)
    
    x_in = x[best_inlier_mask]
    v_in = v[best_inlier_mask]
    F_in = F[best_inlier_mask]
    
    K, B, n, m = best_params
    try:
        if fix_m:
            def model_wrapper(xv, K_final, B_final, n_final):
                x, v = xv
                m_final = n_final - 1 if m_fixed is None else m_fixed
                return hunt_crossley_force(x, v, K_final, B_final, n_final, m_final)
            
            xv_data = np.vstack([x_in, v_in])
            popt, _ = curve_fit(model_wrapper, xv_data, F_in,
                               p0=[K, B, n],
                               bounds=([100, 0, 0.5], [1e6, 1000, 3.0]))
            K, B, n = popt
            m = n - 1 if m_fixed is None else m_fixed
            params = [K, B, n, m]
        else:
            params, _, _ = identify_hunt_crossley(x_in, v_in, F_in, fix_m=False)
    except:
        params = best_params
    
    K, B, n, m = params
    F_pred = hunt_crossley_force(x, v, K, B, n, m)
    ss_res = np.sum((F[best_inlier_mask] - F_pred[best_inlier_mask]) ** 2)
    ss_tot = np.sum((F[best_inlier_mask] - np.mean(F[best_inlier_mask])) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    return params, r_squared, F_pred, best_inlier_mask


def plot_identification_results(x, v, F, linear_results, hc_results, 
                                robust_results=None, true_params=None):
    """
    绘制辨识结果对比
    
    参数:
        x, v, F: 位移、速度、力数据
        linear_results: 线性模型结果 (K, B, r2, F_pred)
        hc_results: Hunt-Crossley结果 (params, r2, F_pred)
        robust_results: 鲁棒Hunt-Crossley结果 (可选)
        true_params: 真实参数 (可选)
    """
    fig = plt.figure(figsize=(16, 10))
    
    gs = fig.add_gridspec(2, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x * 1000, F, c='blue', alpha=0.5, s=20, label='测量数据')
    
    K_lin, B_lin, r2_lin, F_lin = linear_results
    idx_sort = np.argsort(x)
    ax1.plot(x[idx_sort] * 1000, F_lin[idx_sort], 'r-', linewidth=2, 
             label=f'线性模型\nK={K_lin:.1f}, B={B_lin:.2f}\nR²={r2_lin:.4f}')
    
    params_hc, r2_hc, F_hc = hc_results
    K_hc, B_hc, n_hc, m_hc = params_hc
    ax1.plot(x[idx_sort] * 1000, F_hc[idx_sort], 'g-', linewidth=2,
             label=f'Hunt-Crossley\nK={K_hc:.1f}, B={B_hc:.2f}\nn={n_hc:.3f}, R²={r2_hc:.4f}')
    
    if true_params is not None:
        F_true = hunt_crossley_force(x, v, **true_params)
        ax1.plot(x[idx_sort] * 1000, F_true[idx_sort], 'k--', linewidth=2, alpha=0.7,
                 label=f'真实模型\nK={true_params["K"]:.1f}, n={true_params["n"]:.1f}')
    
    ax1.set_xlabel('位移 x (mm)', fontsize=11)
    ax1.set_ylabel('力 F (N)', fontsize=11)
    ax1.set_title('力-位移曲线', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(v, F, c='blue', alpha=0.5, s=20, label='测量数据')
    ax2.set_xlabel('速度 v (m/s)', fontsize=11)
    ax2.set_ylabel('力 F (N)', fontsize=11)
    ax2.set_title('力-速度曲线', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    ax3 = fig.add_subplot(gs[0, 2])
    models = ['线性模型', 'Hunt-Crossley']
    r2_values = [r2_lin, r2_hc]
    colors = ['red', 'green']
    
    if robust_results is not None:
        params_rob, r2_rob, F_rob, mask_rob = robust_results
        models.append('鲁棒H-C')
        r2_values.append(r2_rob)
        colors.append('purple')
    
    bars = ax3.bar(models, r2_values, color=colors, alpha=0.7)
    ax3.set_ylabel('决定系数 R²', fontsize=11)
    ax3.set_title('模型拟合优度对比', fontsize=13, fontweight='bold')
    ax3.set_ylim([max(0, min(r2_values) - 0.05), 1.01])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, r2 in zip(bars, r2_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax4 = fig.add_subplot(gs[1, 0])
    residual_lin = F - F_lin
    residual_hc = F - F_hc
    ax4.scatter(x * 1000, residual_lin, c='red', alpha=0.4, s=20, label=f'线性 (±{np.std(residual_lin):.3f})')
    ax4.scatter(x * 1000, residual_hc, c='green', alpha=0.4, s=20, label=f'H-C (±{np.std(residual_hc):.3f})')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('位移 x (mm)', fontsize=11)
    ax4.set_ylabel('残差 (N)', fontsize=11)
    ax4.set_title('残差分析', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    ax5 = fig.add_subplot(gs[1, 1])
    time = np.arange(len(F))
    ax5.plot(time, F, 'b-', alpha=0.5, label='测量力')
    ax5.plot(time, F_lin, 'r-', alpha=0.8, label='线性预测')
    ax5.plot(time, F_hc, 'g-', alpha=0.8, label='H-C预测')
    ax5.set_xlabel('采样点', fontsize=11)
    ax5.set_ylabel('力 F (N)', fontsize=11)
    ax5.set_title('力随时间变化', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)
    
    ax6 = fig.add_subplot(gs[1, 2])
    if robust_results is not None:
        params_rob, r2_rob, F_rob, mask_rob = robust_results
        K_rob, B_rob, n_rob, m_rob = params_rob
        
        ax6.scatter(x[mask_rob] * 1000, F[mask_rob], c='blue', alpha=0.5, s=20, label='内点')
        ax6.scatter(x[~mask_rob] * 1000, F[~mask_rob], c='red', alpha=0.7, s=40, marker='x', label='剔除异常点')
        ax6.plot(x[idx_sort] * 1000, F_rob[idx_sort], 'purple', linewidth=2,
                 label=f'鲁棒H-C\nK={K_rob:.1f}, n={n_rob:.3f}\nR²={r2_rob:.4f}')
        ax6.set_xlabel('位移 x (mm)', fontsize=11)
        ax6.set_ylabel('力 F (N)', fontsize=11)
        ax6.set_title('RANSAC鲁棒辨识', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=9)
    else:
        ax6.text(0.5, 0.5, '无鲁棒辨识结果', ha='center', va='center', 
                 transform=ax6.transAxes, fontsize=12)
        ax6.axis('off')
    
    plt.tight_layout()
    plt.show()


def print_comparison(true_params, linear_results, hc_results, robust_results=None):
    """
    打印辨识结果对比
    """
    print('=' * 75)
    print(f"{'参数':<12} {'真实值':<15} {'线性模型':<18} {'Hunt-Crossley':<20}", end='')
    if robust_results:
        print(f" {'鲁棒H-C':<15}")
    else:
        print()
    print('=' * 75)
    
    K_lin, B_lin, r2_lin, _ = linear_results
    params_hc, r2_hc, _ = hc_results
    K_hc, B_hc, n_hc, m_hc = params_hc
    
    print(f"{'K':<12} {true_params['K']:<15.2f} {K_lin:<18.2f} {K_hc:<20.2f}", end='')
    if robust_results:
        print(f" {robust_results[0][0]:<15.2f}")
    else:
        print()
    
    print(f"{'B':<12} {true_params['B']:<15.2f} {B_lin:<18.2f} {B_hc:<20.2f}", end='')
    if robust_results:
        print(f" {robust_results[0][1]:<15.2f}")
    else:
        print()
    
    print(f"{'n':<12} {true_params['n']:<15.2f} {'-':<18} {n_hc:<20.3f}", end='')
    if robust_results:
        print(f" {robust_results[0][2]:<15.3f}")
    else:
        print()
    
    print(f"{'m':<12} {true_params['m']:<15.2f} {'-':<18} {m_hc:<20.3f}", end='')
    if robust_results:
        print(f" {robust_results[0][3]:<15.3f}")
    else:
        print()
    
    print(f"{'R²':<12} {'1.0000':<15} {r2_lin:<18.4f} {r2_hc:<20.4f}", end='')
    if robust_results:
        print(f" {robust_results[1]:<15.4f}")
    else:
        print()
    
    print('=' * 75)


def main():
    np.random.seed(42)
    
    print('=' * 70)
    print('软组织接触力学参数辨识系统 - 医疗机器人专用')
    print('=' * 70)
    
    print('\n生成软组织交互模拟数据...')
    true_params = {
        'K': 8000.0,
        'B': 80.0,
        'n': 1.6,
        'm': 0.6
    }
    
    displacement, velocity, force, _ = generate_soft_tissue_data(
        K=true_params['K'], B=true_params['B'], 
        n=true_params['n'], m=true_params['m'],
        noise_level=0.8, num_points=150, outlier_ratio=0.08
    )
    
    print(f"数据点数: {len(force)}")
    print(f"位移范围: {np.min(displacement)*1000:.2f} - {np.max(displacement)*1000:.2f} mm")
    print(f"力范围: {np.min(force):.2f} - {np.max(force):.2f} N")
    print(f"真实参数: K={true_params['K']}, B={true_params['B']}, n={true_params['n']}, m={true_params['m']}")
    
    print('\n' + '=' * 70)
    print('开始辨识...')
    print('=' * 70)
    
    print('\n【1】线性Kelvin-Voigt模型 (F = Kx + Bv)')
    linear_results = identify_linear_model(displacement, velocity, force)
    K_lin, B_lin, r2_lin, _ = linear_results
    print(f"  刚度 K = {K_lin:.2f} N/m")
    print(f"  阻尼 B = {B_lin:.2f} N·s/m")
    print(f"  R² = {r2_lin:.6f}")
    
    print('\n【2】Hunt-Crossley非线性模型 (F = Kx^n + Bx^m·v)')
    hc_results = identify_hunt_crossley(displacement, velocity, force, fix_m=True)
    params_hc, r2_hc, _ = hc_results
    K_hc, B_hc, n_hc, m_hc = params_hc
    print(f"  刚度系数 K = {K_hc:.2f} N/m^{n_hc:.3f}")
    print(f"  阻尼系数 B = {B_hc:.2f} N·s/m^{m_hc+1:.3f}")
    print(f"  非线性指数 n = {n_hc:.4f}")
    print(f"  阻尼指数 m = {m_hc:.4f}")
    print(f"  R² = {r2_hc:.6f}")
    
    print('\n【3】RANSAC鲁棒Hunt-Crossley模型')
    robust_results = identify_robust_hunt_crossley(
        displacement, velocity, force, fix_m=True,
        residual_threshold=1.5, max_trials=300
    )
    params_rob, r2_rob, _, mask_rob = robust_results
    K_rob, B_rob, n_rob, m_rob = params_rob
    print(f"  刚度系数 K = {K_rob:.2f} N/m^{n_rob:.3f}")
    print(f"  阻尼系数 B = {B_rob:.2f} N·s/m^{m_rob+1:.3f}")
    print(f"  非线性指数 n = {n_rob:.4f}")
    print(f"  有效数据点: {np.sum(mask_rob)}/{len(mask_rob)}")
    print(f"  R² = {r2_rob:.6f}")
    
    print('\n' + '=' * 70)
    print('结果对比:')
    print('=' * 70)
    print_comparison(true_params, linear_results, hc_results, robust_results)
    
    improvement = (r2_hc - r2_lin) / (1 - r2_lin) * 100 if r2_lin < 1 else 0
    print(f"\n非线性模型相对线性模型的拟合提升: {improvement:.2f}%")
    
    plot_identification_results(
        displacement, velocity, force,
        linear_results, hc_results, robust_results,
        true_params
    )


if __name__ == '__main__':
    main()
