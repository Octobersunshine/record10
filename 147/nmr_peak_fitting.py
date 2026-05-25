import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from itertools import combinations


def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen) ** 2 / (2 * sigma ** 2))


def lorentzian(x, amp, cen, gamma):
    return amp * (gamma ** 2) / ((x - cen) ** 2 + gamma ** 2)


def voigt(x, amp, cen, sigma, gamma):
    from scipy.special import wofz
    z = (x - cen + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * wofz(z).real / (sigma * np.sqrt(2 * np.pi))


def multi_peak(x, *params, peak_type='gaussian'):
    y = np.zeros_like(x)
    n = len(params)
    if peak_type == 'gaussian':
        for i in range(0, n, 3):
            if i + 2 < n:
                y += gaussian(x, params[i], params[i + 1], params[i + 2])
    elif peak_type == 'lorentzian':
        for i in range(0, n, 3):
            if i + 2 < n:
                y += lorentzian(x, params[i], params[i + 1], params[i + 2])
    elif peak_type == 'voigt':
        for i in range(0, n, 4):
            if i + 3 < n:
                y += voigt(x, params[i], params[i + 1], params[i + 2], params[i + 3])
    return y


def estimate_noise(y, window_size=50):
    noise_est = []
    for i in range(len(y) - window_size):
        segment = y[i:i + window_size]
        noise_est.append(np.std(segment))
    return np.median(noise_est)


def calculate_aic(y, y_fit, n_params):
    n = len(y)
    rss = np.sum((y - y_fit) ** 2)
    if rss <= 0:
        return np.inf
    aic = 2 * n_params + n * np.log(rss / n)
    return aic


def calculate_bic(y, y_fit, n_params):
    n = len(y)
    rss = np.sum((y - y_fit) ** 2)
    if rss <= 0:
        return np.inf
    bic = n_params * np.log(n) + n * np.log(rss / n)
    return bic


def detect_candidate_peaks(x, y, height_factor=0.5, distance=3, min_prominence=0.1):
    noise = estimate_noise(y)
    height_threshold = height_factor * noise
    
    peaks, properties = find_peaks(
        y, 
        height=height_threshold, 
        distance=distance,
        prominence=min_prominence,
        width=(1, None)
    )
    
    candidates = []
    for i, peak_idx in enumerate(peaks):
        candidates.append({
            'index': peak_idx,
            'x': x[peak_idx],
            'y': y[peak_idx],
            'prominence': properties['prominences'][i] if 'prominences' in properties else 0,
            'width': properties['widths'][i] * (x[1] - x[0]) if 'widths' in properties else 0.1
        })
    
    candidates.sort(key=lambda c: c['prominence'], reverse=True)
    
    return candidates


def build_params_from_candidates(candidates, selected_indices, peak_type, x_min, x_max):
    init_params = []
    bounds_lower = []
    bounds_upper = []
    
    n_params_per_peak = 3 if peak_type in ['gaussian', 'lorentzian'] else 4
    
    for idx in selected_indices:
        cand = candidates[idx]
        amp = cand['y']
        cen = cand['x']
        width = max(cand['width'], 0.01)
        
        init_params.extend([amp, cen])
        bounds_lower.extend([0.01 * amp, x_min])
        bounds_upper.extend([5 * amp, x_max])
        
        if peak_type in ['gaussian', 'lorentzian']:
            init_params.append(width)
            bounds_lower.append(0.01 * width)
            bounds_upper.append(10 * width)
        elif peak_type == 'voigt':
            init_params.extend([width * 0.5, width * 0.5])
            bounds_lower.extend([0.005 * width, 0.005 * width])
            bounds_upper.extend([5 * width, 5 * width])
    
    return init_params, bounds_lower, bounds_upper


def fit_with_n_peaks(x, y, candidates, n_peaks, peak_type):
    if n_peaks == 0 or n_peaks > len(candidates):
        return None, np.inf, np.inf
    
    best_aic = np.inf
    best_bic = np.inf
    best_popt = None
    best_selected = None
    
    n_candidates = len(candidates)
    n_params_per_peak = 3 if peak_type in ['gaussian', 'lorentzian'] else 4
    
    if n_peaks <= 8 and n_candidates <= 15:
        combo_iter = combinations(range(n_candidates), n_peaks)
        max_combos = 100
        combos_tested = 0
        
        for selected_indices in combo_iter:
            if combos_tested >= max_combos:
                break
            combos_tested += 1
            
            init_params, bounds_lower, bounds_upper = build_params_from_candidates(
                candidates, selected_indices, peak_type, x.min(), x.max()
            )
            
            def fit_func(x, *params):
                return multi_peak(x, *params, peak_type=peak_type)
            
            try:
                popt, _ = curve_fit(
                    fit_func, x, y, p0=init_params,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=5000
                )
                
                y_fit = fit_func(x, *popt)
                n_params = len(popt)
                aic = calculate_aic(y, y_fit, n_params)
                bic = calculate_bic(y, y_fit, n_params)
                
                if aic < best_aic:
                    best_aic = aic
                    best_bic = bic
                    best_popt = popt
                    best_selected = selected_indices
                    
            except (RuntimeError, ValueError):
                continue
    else:
        selected_indices = list(range(n_peaks))
        init_params, bounds_lower, bounds_upper = build_params_from_candidates(
            candidates, selected_indices, peak_type, x.min(), x.max()
        )
        
        def fit_func(x, *params):
            return multi_peak(x, *params, peak_type=peak_type)
        
        try:
            popt, _ = curve_fit(
                fit_func, x, y, p0=init_params,
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000
            )
            
            y_fit = fit_func(x, *popt)
            n_params = len(popt)
            best_aic = calculate_aic(y, y_fit, n_params)
            best_bic = calculate_bic(y, y_fit, n_params)
            best_popt = popt
            best_selected = selected_indices
            
        except (RuntimeError, ValueError):
            pass
    
    return best_popt, best_aic, best_bic, best_selected


def auto_peak_fit_with_ic(x, y, peak_type='gaussian', criterion='bic', 
                          max_peaks=15, min_peaks=1, plot=True, verbose=True):
    noise = estimate_noise(y)
    if verbose:
        print(f"估计噪声水平: {noise:.6f}")
    
    candidates = detect_candidate_peaks(
        x, y, 
        height_factor=0.3,
        distance=2,
        min_prominence=0.05
    )
    
    if len(candidates) == 0:
        print("未检测到候选峰")
        return None, None, None
    
    if verbose:
        print(f"检测到 {len(candidates)} 个候选峰")
    
    max_peaks = min(max_peaks, len(candidates))
    min_peaks = max(min_peaks, 1)
    
    results_by_n = []
    aic_values = []
    bic_values = []
    n_peaks_range = list(range(min_peaks, max_peaks + 1))
    
    for n_peaks in n_peaks_range:
        if verbose:
            print(f"测试 {n_peaks} 个峰的模型...", end=' ')
        
        popt, aic, bic, selected = fit_with_n_peaks(x, y, candidates, n_peaks, peak_type)
        
        if popt is not None:
            results_by_n.append({
                'n_peaks': n_peaks,
                'popt': popt,
                'aic': aic,
                'bic': bic,
                'selected': selected
            })
            aic_values.append(aic)
            bic_values.append(bic)
            if verbose:
                print(f"AIC={aic:.2f}, BIC={bic:.2f}")
        else:
            aic_values.append(np.inf)
            bic_values.append(np.inf)
            if verbose:
                print("拟合失败")
    
    if len(results_by_n) == 0:
        print("所有模型拟合失败")
        return None, None, None
    
    if criterion.lower() == 'aic':
        best_idx = np.argmin([r['aic'] for r in results_by_n])
    elif criterion.lower() == 'bic':
        best_idx = np.argmin([r['bic'] for r in results_by_n])
    else:
        raise ValueError("criterion 必须是 'aic' 或 'bic'")
    
    best_result = results_by_n[best_idx]
    best_n = best_result['n_peaks']
    best_popt = best_result['popt']
    
    if verbose:
        print(f"\n根据{criterion.upper()}选择的最优峰数: {best_n}")
        print(f"最优模型 AIC: {best_result['aic']:.2f}, BIC: {best_result['bic']:.2f}")
    
    n_params_per_peak = 3 if peak_type in ['gaussian', 'lorentzian'] else 4
    
    peak_results = []
    for i in range(best_n):
        idx = i * n_params_per_peak
        amp = best_popt[idx]
        cen = best_popt[idx + 1]
        
        if peak_type == 'gaussian':
            sigma = best_popt[idx + 2]
            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            area = amp * sigma * np.sqrt(2 * np.pi)
        elif peak_type == 'lorentzian':
            gamma = best_popt[idx + 2]
            fwhm = 2 * gamma
            area = amp * gamma * np.pi
        elif peak_type == 'voigt':
            sigma = best_popt[idx + 2]
            gamma = best_popt[idx + 3]
            fwhm_g = 2 * np.sqrt(2 * np.log(2)) * sigma
            fwhm_l = 2 * gamma
            fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l ** 2 + fwhm_g ** 2)
            area = amp
        
        peak_results.append({
            'peak_index': i + 1,
            'chemical_shift': cen,
            'amplitude': amp,
            'fwhm': fwhm,
            'area': area
        })
    
    ic_analysis = {
        'n_peaks_range': n_peaks_range,
        'aic_values': aic_values,
        'bic_values': bic_values,
        'best_n': best_n,
        'criterion': criterion
    }
    
    if plot:
        plot_fitting_with_ic(x, y, best_popt, peak_type, peak_results, ic_analysis)
    
    return peak_results, best_popt, ic_analysis


def plot_fitting_with_ic(x, y, popt, peak_type, results, ic_analysis):
    fig = plt.figure(figsize=(14, 10))
    
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, y, 'b-', label='原始谱图', linewidth=1.5)
    
    y_fit = multi_peak(x, *popt, peak_type=peak_type)
    ax1.plot(x, y_fit, 'r--', label='拟合曲线', linewidth=1.5)
    
    n_params_per_peak = 3 if peak_type in ['gaussian', 'lorentzian'] else 4
    n_peaks = len(popt) // n_params_per_peak
    
    for i in range(n_peaks):
        idx = i * n_params_per_peak
        if peak_type == 'gaussian':
            y_peak = gaussian(x, popt[idx], popt[idx + 1], popt[idx + 2])
        elif peak_type == 'lorentzian':
            y_peak = lorentzian(x, popt[idx], popt[idx + 1], popt[idx + 2])
        elif peak_type == 'voigt':
            y_peak = voigt(x, popt[idx], popt[idx + 1], popt[idx + 2], popt[idx + 3])
        ax1.fill_between(x, y_peak, alpha=0.3, label=f'峰 {i + 1}')
    
    ax1.set_xlabel('化学位移 (ppm)')
    ax1.set_ylabel('强度')
    ax1.set_title(f'NMR谱峰拟合 - {peak_type}线形 (最优峰数: {n_peaks})')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    ax2 = fig.add_subplot(gs[1, :])
    residual = y - y_fit
    ax2.plot(x, residual, 'g-', label='残差', linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('化学位移 (ppm)')
    ax2.set_ylabel('残差')
    ax2.set_title('拟合残差')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    ax3 = fig.add_subplot(gs[2, 0])
    valid_aic = np.isfinite(ic_analysis['aic_values'])
    if np.any(valid_aic):
        ax3.plot(
            np.array(ic_analysis['n_peaks_range'])[valid_aic],
            np.array(ic_analysis['aic_values'])[valid_aic],
            'bo-', label='AIC', linewidth=1.5, markersize=6
        )
        best_idx = ic_analysis['n_peaks_range'].index(ic_analysis['best_n'])
        ax3.scatter(ic_analysis['best_n'], ic_analysis['aic_values'][best_idx],
                   c='red', s=100, zorder=5, label='最优模型')
    ax3.set_xlabel('峰数')
    ax3.set_ylabel('AIC')
    ax3.set_title('赤池信息准则 (AIC)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, 1])
    valid_bic = np.isfinite(ic_analysis['bic_values'])
    if np.any(valid_bic):
        ax4.plot(
            np.array(ic_analysis['n_peaks_range'])[valid_bic],
            np.array(ic_analysis['bic_values'])[valid_bic],
            'mo-', label='BIC', linewidth=1.5, markersize=6
        )
        best_idx = ic_analysis['n_peaks_range'].index(ic_analysis['best_n'])
        ax4.scatter(ic_analysis['best_n'], ic_analysis['bic_values'][best_idx],
                   c='red', s=100, zorder=5, label='最优模型')
    ax4.set_xlabel('峰数')
    ax4.set_ylabel('BIC')
    ax4.set_title('贝叶斯信息准则 (BIC)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nmr_peak_fitting_ic_result.png', dpi=300, bbox_inches='tight')
    plt.show()


def auto_peak_fit(x, y, peak_type='gaussian', height_threshold=None, distance=5, 
                  prominence=0.5, width_range=(1, 20), plot=True):
    if height_threshold is None:
        noise = estimate_noise(y)
        height_threshold = 3 * noise
    
    peaks, properties = find_peaks(y, height=height_threshold, distance=distance,
                                   prominence=prominence, width=width_range)
    
    if len(peaks) == 0:
        print("未检测到峰，请调整阈值参数")
        return None, None, None
    
    print(f"检测到 {len(peaks)} 个峰")
    
    init_params = []
    bounds_lower = []
    bounds_upper = []
    
    for i, peak_idx in enumerate(peaks):
        amp = y[peak_idx]
        cen = x[peak_idx]
        
        if peak_type in ['gaussian', 'lorentzian']:
            width = properties['widths'][i] * (x[1] - x[0])
            init_params.extend([amp, cen, width])
            bounds_lower.extend([0.1 * amp, x.min(), 0.1 * width])
            bounds_upper.extend([2 * amp, x.max(), 10 * width])
        elif peak_type == 'voigt':
            width = properties['widths'][i] * (x[1] - x[0])
            init_params.extend([amp, cen, width * 0.5, width * 0.5])
            bounds_lower.extend([0.1 * amp, x.min(), 0.01 * width, 0.01 * width])
            bounds_upper.extend([2 * amp, x.max(), 10 * width, 10 * width])
    
    def fit_func(x, *params):
        return multi_peak(x, *params, peak_type=peak_type)
    
    try:
        popt, pcov = curve_fit(fit_func, x, y, p0=init_params, 
                               bounds=(bounds_lower, bounds_upper),
                               maxfev=10000)
    except RuntimeError as e:
        print(f"拟合失败: {e}")
        return None, peaks, properties
    
    results = []
    n_params = 3 if peak_type in ['gaussian', 'lorentzian'] else 4
    
    for i in range(len(peaks)):
        idx = i * n_params
        amp = popt[idx]
        cen = popt[idx + 1]
        
        if peak_type == 'gaussian':
            sigma = popt[idx + 2]
            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            area = amp * sigma * np.sqrt(2 * np.pi)
        elif peak_type == 'lorentzian':
            gamma = popt[idx + 2]
            fwhm = 2 * gamma
            area = amp * gamma * np.pi
        elif peak_type == 'voigt':
            sigma = popt[idx + 2]
            gamma = popt[idx + 3]
            fwhm_g = 2 * np.sqrt(2 * np.log(2)) * sigma
            fwhm_l = 2 * gamma
            fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l ** 2 + fwhm_g ** 2)
            area = amp
        
        results.append({
            'peak_index': i + 1,
            'chemical_shift': cen,
            'amplitude': amp,
            'fwhm': fwhm,
            'area': area
        })
    
    if plot:
        plot_fitting(x, y, popt, peaks, properties, peak_type, results)
    
    return results, popt, properties


def plot_fitting(x, y, popt, peaks, properties, peak_type, results):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'b-', label='原始谱图', linewidth=1.5)
    
    y_fit = multi_peak(x, *popt, peak_type=peak_type)
    plt.plot(x, y_fit, 'r--', label='拟合曲线', linewidth=1.5)
    
    n_params = 3 if peak_type in ['gaussian', 'lorentzian'] else 4
    for i in range(len(peaks)):
        idx = i * n_params
        if peak_type == 'gaussian':
            y_peak = gaussian(x, popt[idx], popt[idx + 1], popt[idx + 2])
        elif peak_type == 'lorentzian':
            y_peak = lorentzian(x, popt[idx], popt[idx + 1], popt[idx + 2])
        elif peak_type == 'voigt':
            y_peak = voigt(x, popt[idx], popt[idx + 1], popt[idx + 2], popt[idx + 3])
        plt.fill_between(x, y_peak, alpha=0.3, label=f'峰 {i + 1}')
    
    plt.xlabel('化学位移 (ppm)')
    plt.ylabel('强度')
    plt.title(f'NMR谱峰拟合 - {peak_type}线形')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    
    plt.subplot(2, 1, 2)
    residual = y - y_fit
    plt.plot(x, residual, 'g-', label='残差', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('化学位移 (ppm)')
    plt.ylabel('残差')
    plt.title('拟合残差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('nmr_peak_fitting_result.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_results(results):
    if results is None:
        return
    
    print("\n" + "=" * 80)
    print("峰拟合结果")
    print("=" * 80)
    print(f"{'峰号':^6} {'化学位移 (ppm)':^18} {'振幅':^12} {'半高宽 (ppm)':^16} {'积分面积':^14}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['peak_index']:^6d} {r['chemical_shift']:^18.4f} {r['amplitude']:^12.4f} "
              f"{r['fwhm']:^16.4f} {r['area']:^14.4f}")
    
    print("=" * 80)


def generate_overlapping_peaks_data(n_true_peaks=3, noise_level=0.03, overlap=True):
    x = np.linspace(0, 10, 1500)
    y = np.zeros_like(x)
    
    true_params = []
    
    centers = np.linspace(3, 7, n_true_peaks)
    if overlap:
        centers += np.random.normal(0, 0.1, n_true_peaks)
    
    for i in range(n_true_peaks):
        amp = np.random.uniform(0.8, 1.5)
        cen = centers[i]
        sigma = np.random.uniform(0.2, 0.4) if overlap else np.random.uniform(0.1, 0.2)
        
        y += gaussian(x, amp, cen, sigma)
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        area = amp * sigma * np.sqrt(2 * np.pi)
        
        true_params.append({
            'peak_index': i + 1,
            'chemical_shift': cen,
            'amplitude': amp,
            'fwhm': fwhm,
            'area': area
        })
    
    y += np.random.normal(0, noise_level, len(y))
    
    return x, y, true_params


def generate_test_data(n_peaks=3, noise_level=0.05):
    x = np.linspace(0, 10, 1000)
    y = np.zeros_like(x)
    
    true_params = []
    
    for i in range(n_peaks):
        amp = np.random.uniform(0.5, 2.0)
        cen = np.random.uniform(1, 9)
        sigma = np.random.uniform(0.1, 0.3)
        
        y += gaussian(x, amp, cen, sigma)
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        area = amp * sigma * np.sqrt(2 * np.pi)
        
        true_params.append({
            'peak_index': i + 1,
            'chemical_shift': cen,
            'amplitude': amp,
            'fwhm': fwhm,
            'area': area
        })
    
    y += np.random.normal(0, noise_level, len(y))
    
    return x, y, true_params


if __name__ == "__main__":
    print("NMR谱自动峰拟合工具 (含AIC/BIC自动选峰)")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("测试1: 重叠峰的AIC/BIC自动选峰")
    print("=" * 70)
    
    x, y, true_params = generate_overlapping_peaks_data(n_true_peaks=3, noise_level=0.03, overlap=True)
    
    print("\n真实峰参数:")
    print_results(true_params)
    
    print("\n使用BIC自动选择最优峰数:")
    results, popt, ic_analysis = auto_peak_fit_with_ic(
        x, y, 
        peak_type='gaussian',
        criterion='bic',
        max_peaks=8,
        min_peaks=1,
        plot=True,
        verbose=True
    )
    print_results(results)
    
    print("\n" + "=" * 70)
    print("测试2: 使用AIC准则")
    print("=" * 70)
    
    results_aic, popt_aic, ic_analysis_aic = auto_peak_fit_with_ic(
        x, y, 
        peak_type='gaussian',
        criterion='aic',
        max_peaks=8,
        min_peaks=1,
        plot=False,
        verbose=True
    )
    print_results(results_aic)
