import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def gaussian_2d(x, y, amp, x0, y0, sigma_x, sigma_y, theta=0):
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    return amp * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))


def lorentzian_2d(x, y, amp, x0, y0, gamma_x, gamma_y, theta=0):
    dx = x - x0
    dy = y - y0
    
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    dx_rot = dx * cos_theta - dy * sin_theta
    dy_rot = dx * sin_theta + dy * cos_theta
    
    return amp / (1 + (dx_rot / gamma_x) ** 2 + (dy_rot / gamma_y) ** 2)


def voigt_2d(x, y, amp, x0, y0, sigma_x, sigma_y, gamma_x, gamma_y, theta=0):
    dx = x - x0
    dy = y - y0
    
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    dx_rot = dx * cos_theta - dy * sin_theta
    dy_rot = dx * sin_theta + dy * cos_theta
    
    from scipy.special import wofz
    
    z_x = (dx_rot + 1j * gamma_x) / (sigma_x * np.sqrt(2))
    z_y = (dy_rot + 1j * gamma_y) / (sigma_y * np.sqrt(2))
    
    v_x = wofz(z_x).real
    v_y = wofz(z_y).real
    
    return amp * v_x * v_y


def multi_peak_2d(xy, *params, peak_type='gaussian'):
    x, y = xy
    z = np.zeros_like(x)
    
    if peak_type == 'gaussian':
        n_params_per_peak = 6
        for i in range(0, len(params), n_params_per_peak):
            if i + 5 < len(params):
                z += gaussian_2d(x, y, params[i], params[i+1], params[i+2], 
                                 params[i+3], params[i+4], params[i+5])
    elif peak_type == 'lorentzian':
        n_params_per_peak = 6
        for i in range(0, len(params), n_params_per_peak):
            if i + 5 < len(params):
                z += lorentzian_2d(x, y, params[i], params[i+1], params[i+2],
                                   params[i+3], params[i+4], params[i+5])
    elif peak_type == 'voigt':
        n_params_per_peak = 8
        for i in range(0, len(params), n_params_per_peak):
            if i + 7 < len(params):
                z += voigt_2d(x, y, params[i], params[i+1], params[i+2],
                              params[i+3], params[i+4], params[i+5], params[i+6], params[i+7])
    
    return z.ravel()


def detect_peaks_2d(z, threshold=3, min_distance=5, neighborhood_size=3):
    noise = np.std(z[z < np.percentile(z, 25)])
    threshold_abs = threshold * noise
    
    z_max = ndimage.maximum_filter(z, size=neighborhood_size)
    maxima = (z == z_max)
    
    z_min = ndimage.minimum_filter(z, size=neighborhood_size)
    delta = ((z_max - z_min) > threshold_abs)
    
    maxima[delta == 0] = 0
    
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    peaks = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2
        intensity = z[int(y_center), int(x_center)]
        
        if intensity > threshold_abs:
            peaks.append({
                'x': x_center,
                'y': y_center,
                'intensity': intensity
            })
    
    peaks.sort(key=lambda p: p['intensity'], reverse=True)
    
    if min_distance > 1:
        filtered_peaks = []
        for peak in peaks:
            too_close = False
            for fpeak in filtered_peaks:
                dist = np.sqrt((peak['x'] - fpeak['x']) ** 2 + (peak['y'] - fpeak['y']) ** 2)
                if dist < min_distance:
                    too_close = True
                    if peak['intensity'] > fpeak['intensity']:
                        fpeak.update(peak)
                    break
            if not too_close:
                filtered_peaks.append(peak)
        peaks = filtered_peaks
    
    return peaks


def estimate_peak_widths_2d(z, peak_x, peak_y, max_radius=20):
    h, w = z.shape
    peak_val = z[int(peak_y), int(peak_x)]
    half_max = peak_val / 2
    
    angles = np.linspace(0, 2 * np.pi, 36)
    widths = []
    
    for angle in angles:
        for r in range(1, max_radius):
            xi = int(peak_x + r * np.cos(angle))
            yi = int(peak_y + r * np.sin(angle))
            
            if 0 <= xi < w and 0 <= yi < h:
                if z[yi, xi] <= half_max:
                    widths.append(r)
                    break
    
    if len(widths) == 0:
        return 3.0, 3.0, 0.0
    
    widths = np.array(widths)
    sigma_x = np.mean(widths[0::2]) if len(widths[0::2]) > 0 else np.mean(widths)
    sigma_y = np.mean(widths[1::2]) if len(widths[1::2]) > 0 else np.mean(widths)
    
    return max(sigma_x, 1.0), max(sigma_y, 1.0), 0.0


def simulated_annealing_2d(xy, z_data, init_params, peak_type, 
                           n_iter=1000, initial_temp=10.0, final_temp=0.01,
                           param_scales=None):
    x, y = xy
    z_data_flat = z_data.ravel()
    
    n_params = len(init_params)
    current_params = np.array(init_params, dtype=float)
    
    if param_scales is None:
        param_scales = np.abs(current_params) * 0.1 + 0.01
    
    def objective(params):
        z_fit = multi_peak_2d((x, y), *params, peak_type=peak_type)
        return np.sum((z_fit - z_data_flat) ** 2)
    
    current_cost = objective(current_params)
    best_params = current_params.copy()
    best_cost = current_cost
    
    cost_history = []
    temp_history = []
    
    cooling_rate = (final_temp / initial_temp) ** (1 / n_iter)
    temp = initial_temp
    
    for i in range(n_iter):
        new_params = current_params + np.random.randn(n_params) * param_scales
        
        new_cost = objective(new_params)
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / temp):
            current_params = new_params
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_params = current_params.copy()
                best_cost = current_cost
        
        temp *= cooling_rate
        
        cost_history.append(current_cost)
        temp_history.append(temp)
        
        if i % 100 == 0 and i > 0:
            param_scales *= 0.9
    
    return best_params, best_cost, {'cost_history': cost_history, 'temp_history': temp_history}


def auto_peak_fit_2d(x, y, z, peak_type='gaussian', threshold=3, 
                     min_distance=5, max_peaks=20, use_sa=True,
                     sa_iter=500, plot=True, verbose=True):
    if verbose:
        print("检测二维峰...")
    
    peaks = detect_peaks_2d(z, threshold=threshold, min_distance=min_distance)
    
    if len(peaks) == 0:
        print("未检测到峰")
        return None, None, None
    
    if len(peaks) > max_peaks:
        peaks = peaks[:max_peaks]
    
    if verbose:
        print(f"检测到 {len(peaks)} 个峰")
    
    n_params_per_peak = 6 if peak_type in ['gaussian', 'lorentzian'] else 8
    init_params = []
    bounds_lower = []
    bounds_upper = []
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    for peak in peaks:
        x_idx = int(peak['x'])
        y_idx = int(peak['y'])
        
        x_pos = x[y_idx, x_idx] if x.ndim == 2 else x[x_idx]
        y_pos = y[y_idx, x_idx] if y.ndim == 2 else y[y_idx]
        amp = peak['intensity']
        
        sigma_x, sigma_y, theta = estimate_peak_widths_2d(z, peak['x'], peak['y'])
        
        if peak_type == 'gaussian':
            init_params.extend([amp, x_pos, y_pos, sigma_x * (x[1] - x[0]), sigma_y * (y[1] - y[0]), theta])
            bounds_lower.extend([0.1 * amp, x_min, y_min, 0.1, 0.1, -np.pi/4])
            bounds_upper.extend([3 * amp, x_max, y_max, 10, 10, np.pi/4])
        elif peak_type == 'lorentzian':
            init_params.extend([amp, x_pos, y_pos, sigma_x * (x[1] - x[0]), sigma_y * (y[1] - y[0]), theta])
            bounds_lower.extend([0.1 * amp, x_min, y_min, 0.05, 0.05, -np.pi/4])
            bounds_upper.extend([3 * amp, x_max, y_max, 5, 5, np.pi/4])
        elif peak_type == 'voigt':
            init_params.extend([amp, x_pos, y_pos, 
                               sigma_x * (x[1] - x[0]) * 0.5, sigma_y * (y[1] - y[0]) * 0.5,
                               sigma_x * (x[1] - x[0]) * 0.5, sigma_y * (y[1] - y[0]) * 0.5,
                               theta])
            bounds_lower.extend([0.1 * amp, x_min, y_min, 0.05, 0.05, 0.05, 0.05, -np.pi/4])
            bounds_upper.extend([3 * amp, x_max, y_max, 3, 3, 3, 3, np.pi/4])
    
    xy_mesh = np.meshgrid(x, y) if x.ndim == 1 else (x, y)
    
    if verbose:
        print("执行局部优化...")
    
    def fit_func(xy_flat, *params):
        x_2d = xy_flat[:len(xy_flat)//2].reshape(z.shape)
        y_2d = xy_flat[len(xy_flat)//2:].reshape(z.shape)
        return multi_peak_2d((x_2d, y_2d), *params, peak_type=peak_type)
    
    xy_flat = np.concatenate([xy_mesh[0].ravel(), xy_mesh[1].ravel()])
    
    try:
        popt, _ = curve_fit(
            fit_func, xy_flat, z.ravel(),
            p0=init_params,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
    except (RuntimeError, ValueError) as e:
        if verbose:
            print(f"局部优化失败，使用初始参数: {e}")
        popt = np.array(init_params)
    
    if use_sa:
        if verbose:
            print("执行模拟退火全局优化...")
        
        param_scales = np.abs(popt) * 0.05 + 0.01
        
        popt_sa, best_cost, sa_info = simulated_annealing_2d(
            xy_mesh, z, popt, peak_type,
            n_iter=sa_iter,
            param_scales=param_scales
        )
        
        z_fit_sa = multi_peak_2d(xy_mesh, *popt_sa, peak_type=peak_type).reshape(z.shape)
        z_fit_local = multi_peak_2d(xy_mesh, *popt, peak_type=peak_type).reshape(z.shape)
        
        mse_sa = np.mean((z_fit_sa - z) ** 2)
        mse_local = np.mean((z_fit_local - z) ** 2)
        
        if mse_sa < mse_local:
            if verbose:
                print(f"模拟退火改进了拟合 (MSE: {mse_local:.6f} -> {mse_sa:.6f})")
            popt = popt_sa
        else:
            if verbose:
                print(f"局部优化结果更好 (MSE: {mse_local:.6f})")
    
    results = []
    for i, peak in enumerate(peaks):
        idx = i * n_params_per_peak
        amp = popt[idx]
        x0 = popt[idx + 1]
        y0 = popt[idx + 2]
        
        if peak_type == 'gaussian':
            sigma_x = popt[idx + 3]
            sigma_y = popt[idx + 4]
            theta = popt[idx + 5]
            fwhm_x = 2 * np.sqrt(2 * np.log(2)) * sigma_x
            fwhm_y = 2 * np.sqrt(2 * np.log(2)) * sigma_y
            volume = 2 * np.pi * amp * sigma_x * sigma_y
        elif peak_type == 'lorentzian':
            gamma_x = popt[idx + 3]
            gamma_y = popt[idx + 4]
            theta = popt[idx + 5]
            fwhm_x = 2 * gamma_x
            fwhm_y = 2 * gamma_y
            volume = np.pi * amp * gamma_x * gamma_y
        elif peak_type == 'voigt':
            sigma_x = popt[idx + 3]
            sigma_y = popt[idx + 4]
            gamma_x = popt[idx + 5]
            gamma_y = popt[idx + 6]
            theta = popt[idx + 7]
            fwhm_gx = 2 * np.sqrt(2 * np.log(2)) * sigma_x
            fwhm_gy = 2 * np.sqrt(2 * np.log(2)) * sigma_y
            fwhm_lx = 2 * gamma_x
            fwhm_ly = 2 * gamma_y
            fwhm_x = 0.5346 * fwhm_lx + np.sqrt(0.2166 * fwhm_lx ** 2 + fwhm_gx ** 2)
            fwhm_y = 0.5346 * fwhm_ly + np.sqrt(0.2166 * fwhm_ly ** 2 + fwhm_gy ** 2)
            volume = amp
        
        results.append({
            'peak_index': i + 1,
            'amplitude': amp,
            'x_position': x0,
            'y_position': y0,
            'fwhm_x': fwhm_x,
            'fwhm_y': fwhm_y,
            'theta': theta,
            'volume': volume
        })
    
    if plot:
        plot_2d_fitting(x, y, z, popt, peak_type, results)
    
    return results, popt, peaks


def plot_2d_fitting(x, y, z, popt, peak_type, results):
    xy_mesh = np.meshgrid(x, y) if x.ndim == 1 else (x, y)
    z_fit = multi_peak_2d(xy_mesh, *popt, peak_type=peak_type).reshape(z.shape)
    z_residual = z - z_fit
    
    fig = plt.figure(figsize=(16, 12))
    
    vmax = np.percentile(z, 99)
    vmin = np.percentile(z, 1)
    
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(z, extent=[x.min(), x.max(), y.min(), y.max()], 
                     origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_xlabel('F2 (ppm)')
    ax1.set_ylabel('F1 (ppm)')
    ax1.set_title('原始 2D NMR 谱')
    plt.colorbar(im1, ax=ax1)
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(z_fit, extent=[x.min(), x.max(), y.min(), y.max()],
                     origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_xlabel('F2 (ppm)')
    ax2.set_ylabel('F1 (ppm)')
    ax2.set_title('拟合 2D NMR 谱')
    plt.colorbar(im2, ax=ax2)
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(z_residual, extent=[x.min(), x.max(), y.min(), y.max()],
                     origin='lower', aspect='auto', cmap='coolwarm',
                     vmin=-np.max(np.abs(z_residual)), vmax=np.max(np.abs(z_residual)))
    ax3.set_xlabel('F2 (ppm)')
    ax3.set_ylabel('F1 (ppm)')
    ax3.set_title('残差')
    plt.colorbar(im3, ax=ax3)
    ax3.invert_xaxis()
    ax3.invert_yaxis()
    
    ax4 = plt.subplot(2, 3, 4)
    levels = np.logspace(np.log10(vmax * 0.05), np.log10(vmax), 10)
    ax4.contour(x, y, z, levels=levels, colors='blue', linewidths=0.8, alpha=0.7)
    ax4.contour(x, y, z_fit, levels=levels, colors='red', linewidths=0.8, linestyles='--', alpha=0.7)
    ax4.set_xlabel('F2 (ppm)')
    ax4.set_ylabel('F1 (ppm)')
    ax4.set_title('等高线对比 (蓝:原始, 红:拟合)')
    ax4.invert_xaxis()
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(2, 3, 5)
    n_params_per_peak = 6 if peak_type in ['gaussian', 'lorentzian'] else 8
    for i in range(len(results)):
        idx = i * n_params_per_peak
        if peak_type == 'gaussian':
            peak_data = gaussian_2d(xy_mesh[0], xy_mesh[1], 
                                    popt[idx], popt[idx+1], popt[idx+2],
                                    popt[idx+3], popt[idx+4], popt[idx+5])
        elif peak_type == 'lorentzian':
            peak_data = lorentzian_2d(xy_mesh[0], xy_mesh[1],
                                      popt[idx], popt[idx+1], popt[idx+2],
                                      popt[idx+3], popt[idx+4], popt[idx+5])
        else:
            peak_data = voigt_2d(xy_mesh[0], xy_mesh[1],
                                 popt[idx], popt[idx+1], popt[idx+2],
                                 popt[idx+3], popt[idx+4], popt[idx+5], popt[idx+6], popt[idx+7])
        ax5.contour(x, y, peak_data, levels=levels, cmap='Set1', linewidths=1)
        ax5.text(results[i]['x_position'], results[i]['y_position'], 
                 str(i+1), fontsize=10, fontweight='bold',
                 ha='center', va='center')
    ax5.set_xlabel('F2 (ppm)')
    ax5.set_ylabel('F1 (ppm)')
    ax5.set_title('各分峰展示')
    ax5.invert_xaxis()
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    table_data = []
    for r in results[:10]:
        table_data.append([
            f"{r['peak_index']}",
            f"{r['x_position']:.2f}",
            f"{r['y_position']:.2f}",
            f"{r['amplitude']:.2f}",
            f"{r['volume']:.2f}"
        ])
    
    if len(results) > 10:
        table_data.append(['...', '...', '...', '...', '...'])
    
    table = ax6.table(cellText=table_data,
                      colLabels=['峰号', 'F2 (ppm)', 'F1 (ppm)', '振幅', '体积'],
                      cellLoc='center',
                      loc='center',
                      fontsize=8)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax6.set_title('峰拟合结果摘要', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig('nmr_2d_fitting_result.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_2d_results(results):
    if results is None:
        return
    
    print("\n" + "=" * 100)
    print("2D NMR 峰拟合结果")
    print("=" * 100)
    print(f"{'峰号':^6} {'F2 (ppm)':^12} {'F1 (ppm)':^12} {'振幅':^12} {'FWHM_x':^10} {'FWHM_y':^10} {'角度(°)':^10} {'体积':^12}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['peak_index']:^6d} {r['x_position']:^12.3f} {r['y_position']:^12.3f} "
              f"{r['amplitude']:^12.3f} {r['fwhm_x']:^10.3f} {r['fwhm_y']:^10.3f} "
              f"{np.degrees(r['theta']):^10.2f} {r['volume']:^12.3f}")
    
    print("=" * 100)


def generate_2d_test_data(n_peaks=5, noise_level=0.05, peak_type='gaussian', 
                          x_range=(0, 10), y_range=(0, 10), size=128):
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], size)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    
    true_params = []
    
    for i in range(n_peaks):
        amp = np.random.uniform(0.5, 2.0)
        x0 = np.random.uniform(x_range[0] + 1, x_range[1] - 1)
        y0 = np.random.uniform(y_range[0] + 1, y_range[1] - 1)
        sigma_x = np.random.uniform(0.1, 0.3)
        sigma_y = np.random.uniform(0.1, 0.3)
        theta = np.random.uniform(-np.pi/6, np.pi/6)
        
        if peak_type == 'gaussian':
            Z += gaussian_2d(X, Y, amp, x0, y0, sigma_x, sigma_y, theta)
            volume = 2 * np.pi * amp * sigma_x * sigma_y
        elif peak_type == 'lorentzian':
            Z += lorentzian_2d(X, Y, amp, x0, y0, sigma_x, sigma_y, theta)
            volume = np.pi * amp * sigma_x * sigma_y
        else:
            Z += voigt_2d(X, Y, amp, x0, y0, sigma_x * 0.5, sigma_y * 0.5, sigma_x * 0.5, sigma_y * 0.5, theta)
            volume = amp
        
        true_params.append({
            'peak_index': i + 1,
            'amplitude': amp,
            'x_position': x0,
            'y_position': y0,
            'fwhm_x': 2 * np.sqrt(2 * np.log(2)) * sigma_x,
            'fwhm_y': 2 * np.sqrt(2 * np.log(2)) * sigma_y,
            'theta': theta,
            'volume': volume
        })
    
    Z += np.random.normal(0, noise_level * np.max(Z), Z.shape)
    
    return x, y, Z, true_params


def generate_hsqc_style_data(n_peaks=8, noise_level=0.03):
    x = np.linspace(10, 180, 256)
    y = np.linspace(0, 10, 128)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    
    true_params = []
    
    backbone_shifts = [
        (175, 8.2), (125, 7.5), (55, 4.5), (45, 4.0),
        (35, 2.0), (25, 1.5), (60, 4.8), (130, 7.0)
    ]
    
    for i, (cx, hx) in enumerate(backbone_shifts[:n_peaks]):
        amp = np.random.uniform(0.6, 1.5)
        sigma_x = np.random.uniform(2, 4)
        sigma_y = np.random.uniform(0.08, 0.15)
        theta = 0
        
        Z += gaussian_2d(X, Y, amp, cx, hx, sigma_x, sigma_y, theta)
        volume = 2 * np.pi * amp * sigma_x * sigma_y
        
        true_params.append({
            'peak_index': i + 1,
            'amplitude': amp,
            'x_position': cx,
            'y_position': hx,
            'fwhm_x': 2 * np.sqrt(2 * np.log(2)) * sigma_x,
            'fwhm_y': 2 * np.sqrt(2 * np.log(2)) * sigma_y,
            'theta': theta,
            'volume': volume
        })
    
    Z += np.random.normal(0, noise_level * np.max(Z), Z.shape)
    
    return x, y, Z, true_params


def generate_cosy_style_data(n_peaks=6, noise_level=0.04):
    x = np.linspace(0, 10, 256)
    y = np.linspace(0, 10, 256)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    
    true_params = []
    
    diagonal_shifts = [1.2, 2.3, 3.5, 4.1, 7.2, 8.5]
    
    for i, shift in enumerate(diagonal_shifts[:n_peaks]):
        amp_diag = np.random.uniform(1.0, 2.0)
        sigma_diag = np.random.uniform(0.08, 0.15)
        theta_diag = np.pi / 4
        
        Z += gaussian_2d(X, Y, amp_diag, shift, shift, sigma_diag, sigma_diag, theta_diag)
        volume = 2 * np.pi * amp_diag * sigma_diag * sigma_diag
        
        true_params.append({
            'peak_index': len(true_params) + 1,
            'amplitude': amp_diag,
            'x_position': shift,
            'y_position': shift,
            'fwhm_x': 2 * np.sqrt(2 * np.log(2)) * sigma_diag,
            'fwhm_y': 2 * np.sqrt(2 * np.log(2)) * sigma_diag,
            'theta': theta_diag,
            'volume': volume
        })
        
        if i > 0:
            prev_shift = diagonal_shifts[i - 1]
            amp_cross = np.random.uniform(0.3, 0.7)
            sigma_cross = np.random.uniform(0.06, 0.12)
            
            Z += gaussian_2d(X, Y, amp_cross, shift, prev_shift, sigma_cross, sigma_cross, 0)
            Z += gaussian_2d(X, Y, amp_cross, prev_shift, shift, sigma_cross, sigma_cross, 0)
            
            volume_cross = 2 * np.pi * amp_cross * sigma_cross * sigma_cross
            true_params.append({
                'peak_index': len(true_params) + 1,
                'amplitude': amp_cross,
                'x_position': shift,
                'y_position': prev_shift,
                'fwhm_x': 2 * np.sqrt(2 * np.log(2)) * sigma_cross,
                'fwhm_y': 2 * np.sqrt(2 * np.log(2)) * sigma_cross,
                'theta': 0,
                'volume': volume_cross
            })
            true_params.append({
                'peak_index': len(true_params) + 1,
                'amplitude': amp_cross,
                'x_position': prev_shift,
                'y_position': shift,
                'fwhm_x': 2 * np.sqrt(2 * np.log(2)) * sigma_cross,
                'fwhm_y': 2 * np.sqrt(2 * np.log(2)) * sigma_cross,
                'theta': 0,
                'volume': volume_cross
            })
    
    Z += np.random.normal(0, noise_level * np.max(Z), Z.shape)
    
    return x, y, Z, true_params


if __name__ == "__main__":
    print("2D NMR 谱自动峰拟合工具")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("测试1: HSQC 风格数据拟合")
    print("=" * 70)
    
    x, y, Z, true_params = generate_hsqc_style_data(n_peaks=6, noise_level=0.05)
    
    print(f"\n真实峰数: {len(true_params)}")
    print("真实峰位置 (F2, F1):")
    for p in true_params:
        print(f"  峰 {p['peak_index']}: ({p['x_position']:.2f}, {p['y_position']:.2f})")
    
    print("\n开始拟合...")
    results, popt, peaks = auto_peak_fit_2d(
        x, y, Z,
        peak_type='gaussian',
        threshold=3,
        min_distance=8,
        max_peaks=15,
        use_sa=True,
        sa_iter=300,
        plot=True,
        verbose=True
    )
    
    print_2d_results(results)
    
    print("\n" + "=" * 70)
    print("测试2: COSY 风格数据拟合")
    print("=" * 70)
    
    x_cosy, y_cosy, Z_cosy, true_cosy = generate_cosy_style_data(n_peaks=4, noise_level=0.04)
    
    print(f"\n真实峰数: {len(true_cosy)}")
    
    results_cosy, popt_cosy, peaks_cosy = auto_peak_fit_2d(
        x_cosy, y_cosy, Z_cosy,
        peak_type='gaussian',
        threshold=3,
        min_distance=10,
        max_peaks=20,
        use_sa=True,
        sa_iter=300,
        plot=True,
        verbose=True
    )
    
    print_2d_results(results_cosy)
