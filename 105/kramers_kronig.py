import numpy as np
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def drude_model_real(omega, omega_p, gamma):
    eps_inf = 1.0
    return eps_inf - omega_p ** 2 / (omega ** 2 + gamma ** 2)


def drude_model_imag(omega, omega_p, gamma):
    return omega_p ** 2 * gamma / (omega * (omega ** 2 + gamma ** 2))


def lorentz_tail_real(omega, A, n):
    return 1.0 + A / (omega ** n)


def lorentz_tail_imag(omega, B, m):
    return B / (omega ** m)


def fit_low_freq_drude(omega, eps_data, fit_type='real'):
    low_freq_mask = omega < np.percentile(omega, 30)
    omega_fit = omega[low_freq_mask]
    eps_fit = eps_data[low_freq_mask]
    
    if fit_type == 'real':
        model = drude_model_real
        p0 = [1.0, 0.1]
    else:
        model = drude_model_imag
        p0 = [1.0, 0.1]
    
    try:
        popt, _ = curve_fit(model, omega_fit, eps_fit, p0=p0, maxfev=10000)
        return popt
    except:
        return p0


def fit_high_freq_lorentz(omega, eps_data, fit_type='real'):
    high_freq_mask = omega > np.percentile(omega, 70)
    omega_fit = omega[high_freq_mask]
    eps_fit = eps_data[high_freq_mask]
    
    if fit_type == 'real':
        model = lorentz_tail_real
        p0 = [1.0, 2.0]
    else:
        model = lorentz_tail_imag
        p0 = [1.0, 2.0]
    
    try:
        popt, _ = curve_fit(model, omega_fit, eps_fit, p0=p0, maxfev=10000)
        return popt
    except:
        return p0


def extrapolate_data(omega, eps_data, fit_type='real', num_extend_low=500, num_extend_high=500):
    omega_min = omega[0]
    omega_max = omega[-1]
    
    if omega_min > 0.01:
        omega_low = np.logspace(np.log10(0.001), np.log10(omega_min * 0.99), num_extend_low)
    else:
        omega_low = np.logspace(np.log10(0.0001), np.log10(omega_min * 0.99), num_extend_low)
    
    omega_high = np.logspace(np.log10(omega_max * 1.01), np.log10(omega_max * 100), num_extend_high)
    
    if fit_type == 'real':
        drude_params = fit_low_freq_drude(omega, eps_data, 'real')
        lorentz_params = fit_high_freq_lorentz(omega, eps_data, 'real')
        eps_low = drude_model_real(omega_low, *drude_params)
        eps_high = lorentz_tail_real(omega_high, *lorentz_params)
    else:
        drude_params = fit_low_freq_drude(omega, eps_data, 'imag')
        lorentz_params = fit_high_freq_lorentz(omega, eps_data, 'imag')
        eps_low = drude_model_imag(omega_low, *drude_params)
        eps_high = lorentz_tail_imag(omega_high, *lorentz_params)
    
    omega_full = np.concatenate([omega_low, omega, omega_high])
    eps_full = np.concatenate([eps_low, eps_data, eps_high])
    
    sort_idx = np.argsort(omega_full)
    omega_full = omega_full[sort_idx]
    eps_full = eps_full[sort_idx]
    
    return omega_full, eps_full


def estimate_noise_level(data, window_size=20):
    if len(data) < 2 * window_size:
        window_size = len(data) // 4
    if window_size < 5:
        window_size = 5
    
    diff = np.diff(data)
    noise_std = np.std(diff) / np.sqrt(2)
    
    return noise_std


def build_kk_matrix(n, transform_type='real_to_imag'):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                H[i, j] = 1.0 / (j - i)
    
    H = H / np.pi
    
    if transform_type == 'imag_to_real':
        H = -H
    
    return H


def tikhonov_regularized_kk(data, alpha=0.01, transform_type='real_to_imag'):
    n = len(data)
    H = build_kk_matrix(n, transform_type)
    
    L = np.eye(n)
    
    A = H.T @ H + alpha * L.T @ L
    b = H.T @ data
    
    result = np.linalg.solve(A, b)
    
    return result


def find_optimal_alpha_lcurve(data, alphas=None, transform_type='real_to_imag'):
    if alphas is None:
        alphas = np.logspace(-8, 2, 50)
    
    residuals = []
    solutions = []
    
    n = len(data)
    H = build_kk_matrix(n, transform_type)
    
    for alpha in alphas:
        L = np.eye(n)
        A = H.T @ H + alpha * L.T @ L
        b = H.T @ data
        x = np.linalg.solve(A, b)
        
        residual = np.linalg.norm(H @ x - data)
        solution_norm = np.linalg.norm(x)
        
        residuals.append(residual)
        solutions.append(solution_norm)
    
    residuals = np.array(residuals)
    solutions = np.array(solutions)
    
    log_res = np.log10(residuals)
    log_sol = np.log10(solutions)
    
    curvature = np.zeros(len(alphas))
    for i in range(1, len(alphas) - 1):
        d1x = (log_res[i+1] - log_res[i-1]) / 2
        d1y = (log_sol[i+1] - log_sol[i-1]) / 2
        d2x = log_res[i+1] - 2*log_res[i] + log_res[i-1]
        d2y = log_sol[i+1] - 2*log_sol[i] + log_sol[i-1]
        curvature[i] = (d1x * d2y - d1y * d2x) / (d1x**2 + d1y**2)**(1.5)
    
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    
    idx = np.argmax(curvature)
    
    return alphas[idx], alphas, residuals, solutions


def kramers_kronig_tikhonov(omega, data, transform_type='real_to_imag', 
                             alpha=None, use_extrapolation=True, 
                             noise_level=None):
    if noise_level is None:
        noise_level = estimate_noise_level(data)
    
    if use_extrapolation:
        fit_type = 'real' if transform_type == 'real_to_imag' else 'imag'
        omega_full, data_full = extrapolate_data(omega, data, fit_type)
    else:
        omega_full = omega
        data_full = data
    
    if alpha is None or alpha == 'auto':
        alpha, _, _, _ = find_optimal_alpha_lcurve(data_full, transform_type=transform_type)
    
    result_full = tikhonov_regularized_kk(data_full, alpha=alpha, transform_type=transform_type)
    
    if use_extrapolation:
        omega_set = set(omega)
        indices = [i for i, w in enumerate(omega_full) if w in omega_set]
        result = result_full[indices]
    else:
        result = result_full
    
    return result, alpha, noise_level


def monte_carlo_uncertainty(omega, data, transform_type='real_to_imag',
                           alpha=None, use_extrapolation=True,
                           noise_level=None, n_samples=100, 
                           confidence_level=0.95):
    if noise_level is None:
        noise_level = estimate_noise_level(data)
    
    results = []
    
    for _ in range(n_samples):
        noisy_data = data + np.random.normal(0, noise_level, size=len(data))
        
        result, _, _ = kramers_kronig_tikhonov(
            omega, noisy_data, transform_type=transform_type,
            alpha=alpha, use_extrapolation=use_extrapolation,
            noise_level=noise_level
        )
        results.append(result)
    
    results = np.array(results)
    
    mean_result = np.mean(results, axis=0)
    std_result = np.std(results, axis=0)
    
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    lower_bound = np.percentile(results, lower_percentile, axis=0)
    upper_bound = np.percentile(results, upper_percentile, axis=0)
    
    return {
        'mean': mean_result,
        'std': std_result,
        'lower': lower_bound,
        'upper': upper_bound,
        'confidence_level': confidence_level,
        'n_samples': n_samples,
        'noise_level': noise_level
    }


def kramers_kronig_real_to_imag(omega, epsilon_real, use_extrapolation=True,
                               method='hilbert', alpha=None, 
                               return_uncertainty=False, **kwargs):
    if method == 'hilbert':
        if use_extrapolation:
            omega_full, eps_real_full = extrapolate_data(omega, epsilon_real, 'real')
            eps_imag_full = np.imag(hilbert(eps_real_full))
            omega_set = set(omega)
            indices = [i for i, w in enumerate(omega_full) if w in omega_set]
            epsilon_imag = eps_imag_full[indices]
        else:
            epsilon_imag = np.imag(hilbert(epsilon_real))
        
        if return_uncertainty:
            unc = monte_carlo_uncertainty(omega, epsilon_real, 'real_to_imag',
                                         use_extrapolation=use_extrapolation, **kwargs)
            return epsilon_imag, unc
        return epsilon_imag
    
    elif method == 'tikhonov':
        result, alpha_opt, noise = kramers_kronig_tikhonov(
            omega, epsilon_real, 'real_to_imag', alpha=alpha,
            use_extrapolation=use_extrapolation
        )
        if return_uncertainty:
            unc = monte_carlo_uncertainty(omega, epsilon_real, 'real_to_imag',
                                         alpha=alpha_opt, use_extrapolation=use_extrapolation,
                                         noise_level=noise, **kwargs)
            return result, {'alpha': alpha_opt, 'noise_level': noise, **unc}
        return result
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hilbert' or 'tikhonov'")


def kramers_kronig_imag_to_real(omega, epsilon_imag, use_extrapolation=True,
                               method='hilbert', alpha=None,
                               return_uncertainty=False, **kwargs):
    if method == 'hilbert':
        if use_extrapolation:
            omega_full, eps_imag_full = extrapolate_data(omega, epsilon_imag, 'imag')
            eps_real_full = -np.imag(hilbert(eps_imag_full))
            omega_set = set(omega)
            indices = [i for i, w in enumerate(omega_full) if w in omega_set]
            epsilon_real = eps_real_full[indices]
        else:
            epsilon_real = -np.imag(hilbert(epsilon_imag))
        
        if return_uncertainty:
            unc = monte_carlo_uncertainty(omega, epsilon_imag, 'imag_to_real',
                                         use_extrapolation=use_extrapolation, **kwargs)
            return epsilon_real, unc
        return epsilon_real
    
    elif method == 'tikhonov':
        result, alpha_opt, noise = kramers_kronig_tikhonov(
            omega, epsilon_imag, 'imag_to_real', alpha=alpha,
            use_extrapolation=use_extrapolation
        )
        if return_uncertainty:
            unc = monte_carlo_uncertainty(omega, epsilon_imag, 'imag_to_real',
                                         alpha=alpha_opt, use_extrapolation=use_extrapolation,
                                         noise_level=noise, **kwargs)
            return result, {'alpha': alpha_opt, 'noise_level': noise, **unc}
        return result
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hilbert' or 'tikhonov'")


def lorentz_oscillator(omega, omega0, gamma, A):
    denominator = (omega0 ** 2 - omega ** 2) ** 2 + (gamma * omega) ** 2
    epsilon_real = 1 + A * (omega0 ** 2 - omega ** 2) / denominator
    epsilon_imag = A * gamma * omega / denominator
    return epsilon_real, epsilon_imag


def test_noise_robustness():
    omega_full_range = np.linspace(0.1, 20, 500)
    omega0 = 3.0
    gamma = 0.5
    A = 2.0
    
    epsilon_real_true, epsilon_imag_true = lorentz_oscillator(omega_full_range, omega0, gamma, A)
    
    limited_mask = (omega_full_range >= 1) & (omega_full_range <= 8)
    omega_limited = omega_full_range[limited_mask]
    epsilon_real_clean = epsilon_real_true[limited_mask]
    epsilon_imag_clean = epsilon_imag_true[limited_mask]
    
    noise_level = 0.02 * np.max(epsilon_imag_clean)
    np.random.seed(42)
    epsilon_real_noisy = epsilon_real_clean + np.random.normal(0, noise_level, size=len(epsilon_real_clean))
    epsilon_imag_noisy = epsilon_imag_clean + np.random.normal(0, noise_level, size=len(epsilon_imag_clean))
    
    imag_hilbert, unc_hilbert = kramers_kronig_real_to_imag(
        omega_limited, epsilon_real_noisy,
        use_extrapolation=True, method='hilbert',
        return_uncertainty=True, n_samples=50
    )
    
    imag_tikhonov, unc_tikhonov = kramers_kronig_real_to_imag(
        omega_limited, epsilon_real_noisy,
        use_extrapolation=True, method='tikhonov',
        return_uncertainty=True, n_samples=50
    )
    
    epsilon_imag_true_limited = epsilon_imag_true[limited_mask]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(omega_limited, epsilon_real_clean, 'k-', label='无噪声', linewidth=2)
    ax1.plot(omega_limited, epsilon_real_noisy, 'b.', label='含噪声', alpha=0.5)
    ax1.set_xlabel('角频率 ω')
    ax1.set_ylabel("ε'(ω)")
    ax1.legend()
    ax1.set_title('输入：含噪声的实部数据')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(omega_limited, epsilon_imag_true_limited, 'k-', label='真实值', linewidth=2)
    ax2.plot(omega_limited, imag_hilbert, 'r--', label='Hilbert变换', alpha=0.7)
    ax2.fill_between(omega_limited, unc_hilbert['lower'], unc_hilbert['upper'], 
                    color='r', alpha=0.2, label='95%置信区间')
    ax2.set_xlabel('角频率 ω')
    ax2.set_ylabel('ε"(ω)')
    ax2.legend()
    ax2.set_title('Hilbert变换结果（无正则化）')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(omega_limited, epsilon_imag_true_limited, 'k-', label='真实值', linewidth=2)
    ax3.plot(omega_limited, imag_tikhonov, 'g--', label='Tikhonov正则化', alpha=0.7)
    ax3.fill_between(omega_limited, unc_tikhonov['lower'], unc_tikhonov['upper'], 
                    color='g', alpha=0.2, label='95%置信区间')
    ax3.set_xlabel('角频率 ω')
    ax3.set_ylabel('ε"(ω)')
    ax3.legend()
    ax3.set_title(f'Tikhonov正则化（α={unc_tikhonov["alpha"]:.2e}）')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    error_hilbert = np.abs(epsilon_imag_true_limited - imag_hilbert)
    error_tikhonov = np.abs(epsilon_imag_true_limited - imag_tikhonov)
    ax4.semilogy(omega_limited, error_hilbert, 'r-', label='Hilbert误差', alpha=0.7)
    ax4.semilogy(omega_limited, error_tikhonov, 'g-', label='Tikhonov误差', alpha=0.7)
    ax4.set_xlabel('角频率 ω')
    ax4.set_ylabel('绝对误差')
    ax4.legend()
    ax4.set_title('误差对比（对数坐标）')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, :])
    omega_plot = np.arange(len(omega_limited))
    ax5.errorbar(omega_plot, imag_hilbert, yerr=unc_hilbert['std'], 
                fmt='r.', label='Hilbert（±1σ）', alpha=0.5)
    ax5.errorbar(omega_plot + 0.2, imag_tikhonov, yerr=unc_tikhonov['std'], 
                fmt='g.', label='Tikhonov（±1σ）', alpha=0.5)
    ax5.plot(omega_plot + 0.1, epsilon_imag_true_limited, 'k-', label='真实值', linewidth=2)
    ax5.set_xlabel('频率点索引')
    ax5.set_ylabel('ε"(ω)')
    ax5.legend()
    ax5.set_title('不确定性区间对比')
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('kk_noise_robustness.png', dpi=150, bbox_inches='tight')
    print("结果图已保存为 kk_noise_robustness.png")
    
    mean_error_hilbert = np.mean(error_hilbert)
    mean_error_tikhonov = np.mean(error_tikhonov)
    mean_unc_hilbert = np.mean(unc_hilbert['std'])
    mean_unc_tikhonov = np.mean(unc_tikhonov['std'])
    
    improvement = (mean_error_hilbert - mean_error_tikhonov) / mean_error_hilbert * 100
    
    print("\n" + "="*70)
    print("噪声鲁棒性测试结果（噪声水平: {:.4f}）".format(noise_level))
    print("="*70)
    print(f"Hilbert变换平均误差: {mean_error_hilbert:.6f}")
    print(f"Hilbert变换平均不确定度: {mean_unc_hilbert:.6f}")
    print("-"*70)
    print(f"Tikhonov正则化平均误差: {mean_error_tikhonov:.6f}")
    print(f"Tikhonov正则化平均不确定度: {mean_unc_tikhonov:.6f}")
    print(f"最优正则化参数 α: {unc_tikhonov['alpha']:.2e}")
    print("-"*70)
    print(f"误差减少率: {improvement:.1f}%")
    print("="*70)
    
    if improvement > 30:
        print("\n✓ Tikhonov正则化显著提高了噪声鲁棒性！")
    else:
        print("\n正则化有一定效果。")
    
    return


def test_causality():
    omega_full_range = np.linspace(0.1, 20, 1000)
    omega0 = 3.0
    gamma = 0.5
    A = 2.0
    
    epsilon_real_true_full, epsilon_imag_true_full = lorentz_oscillator(omega_full_range, omega0, gamma, A)
    
    limited_mask = (omega_full_range >= 1) & (omega_full_range <= 8)
    omega_limited = omega_full_range[limited_mask]
    epsilon_real_limited = epsilon_real_true_full[limited_mask]
    epsilon_imag_limited = epsilon_imag_true_full[limited_mask]
    
    epsilon_imag_no_extra = kramers_kronig_real_to_imag(omega_limited, epsilon_real_limited, use_extrapolation=False)
    epsilon_real_no_extra = kramers_kronig_imag_to_real(omega_limited, epsilon_imag_limited, use_extrapolation=False)
    
    epsilon_imag_with_extra = kramers_kronig_real_to_imag(omega_limited, epsilon_real_limited, use_extrapolation=True)
    epsilon_real_with_extra = kramers_kronig_imag_to_real(omega_limited, epsilon_imag_limited, use_extrapolation=True)
    
    epsilon_real_true_limited = epsilon_real_true_full[limited_mask]
    epsilon_imag_true_limited = epsilon_imag_true_full[limited_mask]
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(omega_limited, epsilon_real_true_limited, 'k-', label='真实值', linewidth=2)
    ax1.plot(omega_limited, epsilon_real_no_extra, 'r--', label='无外推', alpha=0.7)
    ax1.plot(omega_limited, epsilon_real_with_extra, 'b-.', label='有外推', alpha=0.7)
    ax1.set_xlabel('角频率 ω')
    ax1.set_ylabel("ε'(ω)")
    ax1.legend()
    ax1.set_title('实部对比（从虚部计算）')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(omega_limited, epsilon_imag_true_limited, 'k-', label='真实值', linewidth=2)
    ax2.plot(omega_limited, epsilon_imag_no_extra, 'r--', label='无外推', alpha=0.7)
    ax2.plot(omega_limited, epsilon_imag_with_extra, 'b-.', label='有外推', alpha=0.7)
    ax2.set_xlabel('角频率 ω')
    ax2.set_ylabel('ε"(ω)')
    ax2.legend()
    ax2.set_title('虚部对比（从实部计算）')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    error_no_extra_real = np.abs(epsilon_real_true_limited - epsilon_real_no_extra)
    error_with_extra_real = np.abs(epsilon_real_true_limited - epsilon_real_with_extra)
    ax3.semilogy(omega_limited, error_no_extra_real, 'r-', label='无外推', alpha=0.7)
    ax3.semilogy(omega_limited, error_with_extra_real, 'b-', label='有外推', alpha=0.7)
    ax3.set_xlabel('角频率 ω')
    ax3.set_ylabel('绝对误差')
    ax3.legend()
    ax3.set_title('实部误差（对数坐标）')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    error_no_extra_imag = np.abs(epsilon_imag_true_limited - epsilon_imag_no_extra)
    error_with_extra_imag = np.abs(epsilon_imag_true_limited - epsilon_imag_with_extra)
    ax4.semilogy(omega_limited, error_no_extra_imag, 'r-', label='无外推', alpha=0.7)
    ax4.semilogy(omega_limited, error_with_extra_imag, 'b-', label='有外推', alpha=0.7)
    ax4.set_xlabel('角频率 ω')
    ax4.set_ylabel('绝对误差')
    ax4.legend()
    ax4.set_title('虚部误差（对数坐标）')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, :])
    omega_full_plot, eps_real_extra_plot = extrapolate_data(omega_limited, epsilon_real_limited, 'real')
    ax5.plot(omega_full_range, epsilon_real_true_full, 'k-', label='真实全频段', alpha=0.5)
    ax5.plot(omega_limited, epsilon_real_limited, 'bo-', label='测量频段', markersize=3, alpha=0.7)
    ax5.plot(omega_full_plot, eps_real_extra_plot, 'g--', label='外推后数据', alpha=0.7)
    ax5.axvspan(1, 8, alpha=0.2, color='yellow', label='测量范围')
    ax5.set_xlabel('角频率 ω')
    ax5.set_ylabel("ε'(ω)")
    ax5.legend()
    ax5.set_title('外推示意图（低频Drude模型，高频Lorentz尾部）')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('kramers_kronig_extrapolation_comparison.png', dpi=150, bbox_inches='tight')
    print("结果图已保存为 kramers_kronig_extrapolation_comparison.png")
    
    mean_error_real_no = np.mean(error_no_extra_real)
    mean_error_real_with = np.mean(error_with_extra_real)
    mean_error_imag_no = np.mean(error_no_extra_imag)
    mean_error_imag_with = np.mean(error_with_extra_imag)
    
    improvement_real = (mean_error_real_no - mean_error_real_with) / mean_error_real_no * 100
    improvement_imag = (mean_error_imag_no - mean_error_imag_with) / mean_error_imag_no * 100
    
    print("\n" + "="*60)
    print("外推法误差对比")
    print("="*60)
    print(f"实部平均误差（无外推）: {mean_error_real_no:.6f}")
    print(f"实部平均误差（有外推）: {mean_error_real_with:.6f}")
    print(f"实部误差减少: {improvement_real:.1f}%")
    print("-"*60)
    print(f"虚部平均误差（无外推）: {mean_error_imag_no:.6f}")
    print(f"虚部平均误差（有外推）: {mean_error_imag_with:.6f}")
    print(f"虚部误差减少: {improvement_imag:.1f}%")
    print("="*60)


if __name__ == "__main__":
    print("运行外推法测试...")
    test_causality()
    print("\n" + "="*70)
    print("运行噪声鲁棒性测试...")
    test_noise_robustness()
