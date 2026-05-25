import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import solve


def randles_impedance(f, Rs, Rct, Cdl):
    omega = 2 * np.pi * f
    Z_cdl = 1 / (1j * omega * Cdl)
    Z_parallel = (Rct * Z_cdl) / (Rct + Z_cdl)
    Z_total = Rs + Z_parallel
    return Z_total


def randles_cpe_impedance(f, Rs, Rct, Q, n):
    omega = 2 * np.pi * f
    Z_cpe = 1 / (Q * (1j * omega) ** n)
    Z_parallel = (Rct * Z_cpe) / (Rct + Z_cpe)
    Z_total = Rs + Z_parallel
    return Z_total


def compute_weights(Z_exp, weighting='modulus', sigma_real=None, sigma_imag=None):
    if weighting is None or weighting == 'none':
        weights = np.ones_like(Z_exp.real)
    elif weighting == 'modulus':
        weights = 1.0 / (np.abs(Z_exp) ** 2)
    elif weighting == 'unit':
        weights = 1.0 / (np.abs(Z_exp) ** 2)
    elif weighting == 'proportional':
        weights = 1.0 / (np.abs(Z_exp) ** 2)
    elif weighting == 'variance':
        if sigma_real is None or sigma_imag is None:
            raise ValueError("sigma_real and sigma_imag must be provided for variance weighting")
        weights_real = 1.0 / (sigma_real ** 2)
        weights_imag = 1.0 / (sigma_imag ** 2)
        weights = (weights_real + weights_imag) / 2
    else:
        raise ValueError(f"Unknown weighting method: {weighting}")
    
    weights = np.clip(weights, 1e-10, 1e10)
    return weights


def residuals_randles(params, f, Z_exp, weights=None):
    Rs, Rct, Cdl = params
    Z_model = randles_impedance(f, Rs, Rct, Cdl)
    res_real = Z_model.real - Z_exp.real
    res_imag = Z_model.imag - Z_exp.imag
    
    if weights is not None:
        res_real = res_real * np.sqrt(weights)
        res_imag = res_imag * np.sqrt(weights)
    
    return np.concatenate([res_real, res_imag])


def residuals_randles_cpe(params, f, Z_exp, weights=None):
    Rs, Rct, Q, n = params
    Z_model = randles_cpe_impedance(f, Rs, Rct, Q, n)
    res_real = Z_model.real - Z_exp.real
    res_imag = Z_model.imag - Z_exp.imag
    
    if weights is not None:
        res_real = res_real * np.sqrt(weights)
        res_imag = res_imag * np.sqrt(weights)
    
    return np.concatenate([res_real, res_imag])


def fit_eis(f, Z_exp, model='randles', initial_guess=None, bounds=None, 
            weighting='modulus', sigma_real=None, sigma_imag=None):
    weights = compute_weights(Z_exp, weighting, sigma_real, sigma_imag)
    
    if model == 'randles':
        if initial_guess is None:
            initial_guess = estimate_initial_parameters(f, Z_exp)
        if bounds is None:
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        residual_func = residuals_randles
        n_params = 3
    elif model == 'randles_cpe':
        if initial_guess is None:
            Rs_est, Rct_est, Cdl_est = estimate_initial_parameters(f, Z_exp)
            initial_guess = [Rs_est, Rct_est, Cdl_est, 0.9]
        if bounds is None:
            bounds = ([0, 0, 0, 0.5], [np.inf, np.inf, np.inf, 1.0])
        residual_func = residuals_randles_cpe
        n_params = 4
    else:
        raise ValueError(f"Unknown model: {model}")
    
    result = least_squares(
        residual_func,
        initial_guess,
        bounds=bounds,
        args=(f, Z_exp, weights),
        method='trf',
        max_nfev=10000,
        ftol=1e-12,
        xtol=1e-12
    )
    
    result.weights = weights
    result.weighting = weighting
    result.n_params = n_params
    
    return result


def estimate_initial_parameters(f, Z_exp):
    high_freq_idx = np.argmax(f)
    Rs_est = Z_exp[high_freq_idx].real
    low_freq_idx = np.argmin(f)
    Rct_est = Z_exp[low_freq_idx].real - Rs_est
    if Rct_est <= 0:
        Rct_est = np.max(Z_exp.real) - Rs_est
        if Rct_est <= 0:
            Rct_est = 100
    f_max_idx = np.argmax(-Z_exp.imag)
    f_max = f[f_max_idx]
    omega_max = 2 * np.pi * f_max
    Cdl_est = 1 / (omega_max * Rct_est)
    return [Rs_est, Rct_est, Cdl_est]


def generate_sample_data(f, Rs_true=10, Rct_true=150, Cdl_true=2e-6, noise_level=0.02, 
                         model='randles', Q_true=None, n_true=None, frequency_dependent_noise=False):
    if model == 'randles':
        Z_true = randles_impedance(f, Rs_true, Rct_true, Cdl_true)
    elif model == 'randles_cpe':
        Z_true = randles_cpe_impedance(f, Rs_true, Rct_true, Q_true, n_true)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    if frequency_dependent_noise:
        noise_scale = noise_level * (1 + 0.1 * np.log10(f.max() / f))
    else:
        noise_scale = noise_level
    
    real_noise = np.random.normal(0, noise_scale * np.abs(Z_true), Z_true.shape)
    imag_noise = np.random.normal(0, noise_scale * np.abs(Z_true), Z_true.shape)
    Z_noisy = Z_true + real_noise + 1j * imag_noise
    
    sigma_real = noise_scale * np.abs(Z_true)
    sigma_imag = noise_scale * np.abs(Z_true)
    
    return Z_noisy, Z_true, sigma_real, sigma_imag


def calculate_statistics(Z_exp, Z_fit, n_params, weights=None):
    residuals = Z_exp - Z_fit
    chi_squared = np.sum(residuals.real**2 + residuals.imag**2)
    
    if weights is not None:
        weighted_chi_squared = np.sum(weights * (residuals.real**2 + residuals.imag**2))
    else:
        weighted_chi_squared = chi_squared
    
    n_points = len(Z_exp) * 2
    dof = n_points - n_params
    chi_squared_red = chi_squared / dof if dof > 0 else np.inf
    weighted_chi_squared_red = weighted_chi_squared / dof if dof > 0 else np.inf
    rmse = np.sqrt(chi_squared / n_points)
    
    return chi_squared, chi_squared_red, weighted_chi_squared, weighted_chi_squared_red, rmse, dof


def plot_nyquist(Z_exp, Z_fit=None, title='Nyquist Plot', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(Z_exp.real, -Z_exp.imag, 'o', label='Experimental Data', markersize=6, zorder=5)
    if Z_fit is not None:
        ax.plot(Z_fit.real, -Z_fit.imag, '-', label='Fitted Model', linewidth=2, zorder=3)
    
    ax.set_xlabel('Z_real (Ω)', fontsize=12)
    ax.set_ylabel('-Z_imag (Ω)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    return ax


def plot_bode(f, Z_exp, Z_fit=None, title='Bode Plot', fig=None):
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    else:
        ax1, ax2 = fig.axes
    
    mag_exp = np.abs(Z_exp)
    phase_exp = np.angle(Z_exp, deg=True)
    
    ax1.semilogx(f, mag_exp, 'o', label='Experimental Data', markersize=6, zorder=5)
    if Z_fit is not None:
        mag_fit = np.abs(Z_fit)
        ax1.semilogx(f, mag_fit, '-', label='Fitted Model', linewidth=2, zorder=3)
    ax1.set_ylabel('|Z| (Ω)', fontsize=12)
    ax1.set_title(title + ' - Magnitude', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2.semilogx(f, phase_exp, 'o', label='Experimental Data', markersize=6, zorder=5)
    if Z_fit is not None:
        phase_fit = np.angle(Z_fit, deg=True)
        ax2.semilogx(f, phase_fit, '-', label='Fitted Model', linewidth=2, zorder=3)
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Phase (degrees)', fontsize=12)
    ax2.set_title(title + ' - Phase', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def plot_residuals(f, Z_exp, Z_fit, title='Residuals Plot', weights=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    res_real = (Z_fit.real - Z_exp.real) / np.abs(Z_exp) * 100
    res_imag = (Z_fit.imag - Z_exp.imag) / np.abs(Z_exp) * 100
    
    ax1.semilogx(f, res_real, 'o-', markersize=6, linewidth=1, label='Relative Residual')
    if weights is not None:
        ax1_twin = ax1.twinx()
        ax1_twin.semilogx(f, weights, 'r--', alpha=0.5, label='Weight')
        ax1_twin.set_ylabel('Weight', fontsize=12, color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Real Residual (%)', fontsize=12)
    ax1.set_title(title + ' - Real Part', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2.semilogx(f, res_imag, 'o-', markersize=6, linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Imaginary Residual (%)', fontsize=12)
    ax2.set_title(title + ' - Imaginary Part', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def plot_weight_comparison(f, Z_exp, results_dict, Z_true=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['b', 'r', 'g', 'm']
    markers = ['-', '--', '-.', ':']
    
    axes[0].plot(Z_exp.real, -Z_exp.imag, 'ko', label='Data', markersize=5, alpha=0.5)
    if Z_true is not None:
        axes[0].plot(Z_true.real, -Z_true.imag, 'k-', label='True', linewidth=1, alpha=0.3)
    
    for i, (name, result) in enumerate(results_dict.items()):
        if result['model'] == 'randles':
            Z_fit = randles_impedance(f, *result['params'])
        else:
            Z_fit = randles_cpe_impedance(f, *result['params'])
        axes[0].plot(Z_fit.real, -Z_fit.imag, colors[i] + markers[i], 
                     label=f'{name}', linewidth=2, zorder=5-i)
    axes[0].set_xlabel('Z_real (Ω)', fontsize=12)
    axes[0].set_ylabel('-Z_imag (Ω)', fontsize=12)
    axes[0].set_title('Nyquist Plot Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    method_names = list(results_dict.keys())
    Rs_errors = [abs(r['params'][0] - r['true_params'][0]) / r['true_params'][0] * 100 
                 for r in results_dict.values()]
    Rct_errors = [abs(r['params'][1] - r['true_params'][1]) / r['true_params'][1] * 100 
                  for r in results_dict.values()]
    
    x = np.arange(len(method_names))
    width = 0.35
    
    axes[1].bar(x - width/2, Rs_errors, width, label='Rs error', alpha=0.7)
    axes[1].bar(x + width/2, Rct_errors, width, label='Rct error', alpha=0.7)
    axes[1].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1].set_title('Parameter Error Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_names, rotation=45, ha='right')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    chi_sq_red = [r['chi_sq_red'] for r in results_dict.values()]
    axes[2].bar(method_names, chi_sq_red, alpha=0.7, color=colors[:len(method_names)])
    axes[2].set_ylabel('Reduced χ²', fontsize=12)
    axes[2].set_title('Goodness of Fit', fontsize=14, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def load_eis_from_csv(filepath, delimiter=',', f_col=0, z_real_col=1, z_imag_col=2, imag_sign='negative'):
    data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1)
    f = data[:, f_col]
    z_real = data[:, z_real_col]
    z_imag = data[:, z_imag_col]
    
    if imag_sign == 'negative':
        z_imag = -z_imag
    
    Z = z_real + 1j * z_imag
    return f, Z


def compute_tau_axis(f, num_points=100, tau_range_factor=10):
    f_min, f_max = np.min(f), np.max(f)
    tau_min = 1 / (2 * np.pi * f_max) / tau_range_factor
    tau_max = 1 / (2 * np.pi * f_min) * tau_range_factor
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), num_points)
    return tau


def drt_impedance_matrix(f, tau):
    omega = 2 * np.pi * f
    omega_tau = omega[:, np.newaxis] * tau[np.newaxis, :]
    
    A_real = 1.0 / (1.0 + omega_tau**2)
    A_imag = -omega_tau / (1.0 + omega_tau**2)
    
    return A_real, A_imag


def compute_drt(f, Z_exp, tau=None, lambda_reg=1e-3, method='real_imag'):
    if tau is None:
        tau = compute_tau_axis(f)
    
    n_tau = len(tau)
    A_real, A_imag = drt_impedance_matrix(f, tau)
    
    high_freq_idx = np.argmax(f)
    Rs_est = Z_exp[high_freq_idx].real
    Z_polarization = Z_exp - Rs_est
    
    if method == 'real_imag':
        A = np.vstack([A_real, A_imag])
        b = np.concatenate([Z_polarization.real, Z_polarization.imag])
        
        ATA = A.T @ A
        L = np.eye(n_tau)
        L[1:-1, 1:-1] = np.diag(np.ones(n_tau-2), 0) - 0.5 * np.diag(np.ones(n_tau-3), 1) - 0.5 * np.diag(np.ones(n_tau-3), -1)
        LTL = L.T @ L
        
        gamma = solve(ATA + lambda_reg * LTL, A.T @ b)
        
    elif method == 'imag_only':
        A = A_imag
        b = Z_polarization.imag
        
        ATA = A.T @ A
        L = np.eye(n_tau)
        L[1:-1, 1:-1] = np.diag(np.ones(n_tau-2), 0) - 0.5 * np.diag(np.ones(n_tau-3), 1) - 0.5 * np.diag(np.ones(n_tau-3), -1)
        LTL = L.T @ L
        
        gamma = solve(ATA + lambda_reg * LTL, A.T @ b)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    gamma = np.maximum(gamma, 0)
    
    Z_reconstructed = Rs_est + A_real @ gamma + 1j * (A_imag @ gamma)
    
    result = {
        'tau': tau,
        'gamma': gamma,
        'Rs': Rs_est,
        'Z_reconstructed': Z_reconstructed,
        'lambda_reg': lambda_reg,
        'method': method,
        'R_total': np.trapz(gamma, np.log(tau))
    }
    
    return result


def find_drt_peaks(tau, gamma, min_height_ratio=0.05, min_distance=5):
    gamma_normalized = gamma / np.max(gamma)
    
    peaks = []
    for i in range(1, len(gamma) - 1):
        if gamma[i] > gamma[i-1] and gamma[i] > gamma[i+1]:
            if gamma_normalized[i] >= min_height_ratio:
                peaks.append(i)
    
    if len(peaks) > 1:
        peaks = sorted(peaks, key=lambda x: gamma[x], reverse=True)
        selected_peaks = [peaks[0]]
        for p in peaks[1:]:
            if all(abs(p - sp) >= min_distance for sp in selected_peaks):
                selected_peaks.append(p)
        peaks = sorted(selected_peaks)
    
    peak_info = []
    for p in peaks:
        left = p
        while left > 0 and gamma[left-1] < gamma[left]:
            left -= 1
        right = p
        while right < len(gamma) - 1 and gamma[right+1] < gamma[right]:
            right += 1
        
        peak_area = np.trapz(gamma[left:right+1], np.log(tau[left:right+1]))
        
        peak_info.append({
            'index': p,
            'tau': tau[p],
            'gamma': gamma[p],
            'area': peak_area,
            'f_peak': 1 / (2 * np.pi * tau[p])
        })
    
    return peak_info


def optimize_lambda_cv(f, Z_exp, tau=None, lambda_values=None, cv_folds=5):
    if tau is None:
        tau = compute_tau_axis(f)
    
    if lambda_values is None:
        lambda_values = np.logspace(-6, 0, 20)
    
    n_points = len(f)
    indices = np.arange(n_points)
    np.random.shuffle(indices)
    
    cv_errors = []
    
    for lambda_reg in lambda_values:
        fold_errors = []
        
        for fold in range(cv_folds):
            val_mask = np.zeros(n_points, dtype=bool)
            val_mask[indices[fold::cv_folds]] = True
            train_mask = ~val_mask
            
            drt_train = compute_drt(f[train_mask], Z_exp[train_mask], tau, lambda_reg)
            
            A_real, A_imag = drt_impedance_matrix(f[val_mask], tau)
            Z_pred = drt_train['Rs'] + A_real @ drt_train['gamma'] + 1j * (A_imag @ drt_train['gamma'])
            
            error = np.mean(np.abs(Z_pred - Z_exp[val_mask])**2)
            fold_errors.append(error)
        
        cv_errors.append(np.mean(fold_errors))
    
    best_idx = np.argmin(cv_errors)
    best_lambda = lambda_values[best_idx]
    
    return best_lambda, lambda_values, cv_errors


def plot_drt(drt_result, title='Distribution of Relaxation Times', ax=None, show_peaks=True):
    tau = drt_result['tau']
    gamma = drt_result['gamma']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(tau, gamma, 'b-', linewidth=2, label='DRT')
    ax.fill_between(tau, gamma, alpha=0.3, color='skyblue')
    
    if show_peaks:
        peaks = find_drt_peaks(tau, gamma)
        for i, peak in enumerate(peaks):
            ax.plot(peak['tau'], peak['gamma'], 'ro', markersize=8, zorder=5)
            ax.annotate(f"Peak {i+1}\nτ={peak['tau']:.2e}s\nf={peak['f_peak']:.1f}Hz\nArea={peak['area']:.2f}Ω",
                       xy=(peak['tau'], peak['gamma']),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       fontsize=8)
    
    ax.set_xlabel('Relaxation Time τ (s)', fontsize=12)
    ax.set_ylabel('γ(lnτ) (Ω)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    return ax


def plot_drt_comparison(f, Z_exp, drt_result, title='DRT Reconstruction Comparison'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    Z_recon = drt_result['Z_reconstructed']
    
    axes[0].plot(Z_exp.real, -Z_exp.imag, 'o', label='Experimental Data', markersize=6, zorder=5)
    axes[0].plot(Z_recon.real, -Z_recon.imag, 'r-', label='DRT Reconstructed', linewidth=2, zorder=3)
    axes[0].set_xlabel('Z_real (Ω)', fontsize=12)
    axes[0].set_ylabel('-Z_imag (Ω)', fontsize=12)
    axes[0].set_title('Nyquist Plot', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    plot_drt(drt_result, title='DRT Spectrum', ax=axes[1])
    
    plt.tight_layout()
    return fig


def generate_multi_relaxation_data(f, Rs, R_list, tau_list, noise_level=0.02):
    omega = 2 * np.pi * f
    Z = Rs + 0j
    
    for R, tau in zip(R_list, tau_list):
        Z += R / (1 + 1j * omega * tau)
    
    real_noise = np.random.normal(0, noise_level * np.abs(Z), Z.shape)
    imag_noise = np.random.normal(0, noise_level * np.abs(Z), Z.shape)
    Z_noisy = Z + real_noise + 1j * imag_noise
    
    return Z_noisy, Z


def main():
    np.random.seed(42)
    
    f = np.logspace(5, -1, 50)
    
    print("=" * 80)
    print("EIS Fitting with Weighted Complex Nonlinear Least Squares (CNLS)")
    print("=" * 80)
    
    Rs_true, Rct_true, Cdl_true = 10, 150, 2e-6
    Z_noisy, Z_true, sigma_real, sigma_imag = generate_sample_data(
        f, Rs_true, Rct_true, Cdl_true, noise_level=0.02, frequency_dependent_noise=True
    )
    
    print(f"\nTrue Parameters:")
    print(f"  Rs  = {Rs_true} Ω (Solution resistance)")
    print(f"  Rct = {Rct_true} Ω (Charge transfer resistance)")
    print(f"  Cdl = {Cdl_true} F (Double layer capacitance)")
    
    print(f"\n{'=' * 80}")
    print("COMPARISON OF DIFFERENT WEIGHTING METHODS")
    print(f"{'=' * 80}")
    
    weighting_methods = [
        ('None (Equal)', 'none'),
        ('Modulus (1/|Z|²)', 'modulus'),
        ('Variance-based', 'variance')
    ]
    
    results_dict = {}
    
    for method_name, weighting in weighting_methods:
        print(f"\n--- {method_name} Weighting ---")
        
        if weighting == 'variance':
            result = fit_eis(f, Z_noisy, model='randles', weighting=weighting,
                           sigma_real=sigma_real, sigma_imag=sigma_imag)
        else:
            result = fit_eis(f, Z_noisy, model='randles', weighting=weighting)
        
        Rs_fit, Rct_fit, Cdl_fit = result.x
        Z_fit = randles_impedance(f, Rs_fit, Rct_fit, Cdl_fit)
        
        chi_sq, chi_sq_red, w_chi_sq, w_chi_sq_red, rmse, dof = calculate_statistics(
            Z_noisy, Z_fit, 3, result.weights
        )
        
        print(f"  Fitted Parameters:")
        print(f"    Rs  = {Rs_fit:.4f} Ω (error: {abs((Rs_fit - Rs_true)/Rs_true*100):.2f}%)")
        print(f"    Rct = {Rct_fit:.4f} Ω (error: {abs((Rct_fit - Rct_true)/Rct_true*100):.2f}%)")
        print(f"    Cdl = {Cdl_fit:.4e} F (error: {abs((Cdl_fit - Cdl_true)/Cdl_true*100):.2f}%)")
        print(f"  Goodness of Fit:")
        print(f"    Reduced χ² = {chi_sq_red:.6f}")
        print(f"    Weighted χ² = {w_chi_sq_red:.6f}")
        
        results_dict[method_name] = {
            'params': result.x,
            'true_params': [Rs_true, Rct_true, Cdl_true],
            'model': 'randles',
            'chi_sq_red': chi_sq_red,
            'result': result
        }
    
    print(f"\n{'=' * 80}")
    print("WEIGHTING METHODS EXPLANATION")
    print(f"{'=' * 80}")
    print(f"""
  1. None (Equal Weighting):
     - All data points have equal weight
     - Minimizes: Σ[(Z_model.real - Z_exp.real)² + (Z_model.imag - Z_exp.imag)²]
     - Issue: Large |Z| values dominate the fit due to larger absolute residuals

  2. Modulus Weighting (1/|Z|²) - RECOMMENDED:
     - Weights are proportional to 1/|Z|²
     - Minimizes: Σ[((Z_model.real - Z_exp.real)² + (Z_model.imag - Z_exp.imag)²) / |Z|²]
     - Ensures equal importance to all data points regardless of impedance magnitude
     - Standard method in EIS literature

  3. Variance-based Weighting:
     - Uses actual measurement variances (σ_real², σ_imag²)
     - Minimizes: Σ[(Z_model.real - Z_exp.real)²/σ_real² + (Z_model.imag - Z_exp.imag)²/σ_imag²]
     - Optimal if noise levels are known
     - Requires noise estimation from repeated measurements
    """)
    
    print(f"\n{'=' * 80}")
    print("GENERATING COMPARISON PLOTS...")
    print(f"{'=' * 80}")
    
    fig_compare = plot_weight_comparison(f, Z_noisy, results_dict, Z_true)
    fig_compare.savefig('weighting_comparison.png', dpi=300, bbox_inches='tight')
    
    print(f"\nGenerating detailed plots for Modulus weighting (recommended)...")
    
    result_best = results_dict['Modulus (1/|Z|²)']['result']
    Rs_fit_best, Rct_fit_best, Cdl_fit_best = result_best.x
    Z_fit_best = randles_impedance(f, Rs_fit_best, Rct_fit_best, Cdl_fit_best)
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_nyquist(Z_noisy, Z_fit_best, title='Nyquist Plot - Modulus Weighting', ax=ax1)
    fig1.savefig('nyquist_plot_weighted.png', dpi=300, bbox_inches='tight')
    
    fig2 = plot_bode(f, Z_noisy, Z_fit_best, title='Bode Plot - Modulus Weighting')
    fig2.savefig('bode_plot_weighted.png', dpi=300, bbox_inches='tight')
    
    fig3 = plot_residuals(f, Z_noisy, Z_fit_best, title='Residuals - Modulus Weighting', 
                          weights=result_best.weights)
    fig3.savefig('residuals_plot_weighted.png', dpi=300, bbox_inches='tight')
    
    print(f"\nPlots saved:")
    print(f"  - weighting_comparison.png")
    print(f"  - nyquist_plot_weighted.png")
    print(f"  - bode_plot_weighted.png")
    print(f"  - residuals_plot_weighted.png")
    
    print(f"\n{'=' * 80}")
    print("DEMONSTRATION WITH CPE MODEL (Modulus Weighting)")
    print(f"{'=' * 80}")
    
    Q_true, n_true = 2e-6, 0.85
    Z_noisy_cpe, Z_true_cpe, _, _ = generate_sample_data(
        f, Rs_true, Rct_true, model='randles_cpe', Q_true=Q_true, n_true=n_true, noise_level=0.02
    )
    
    result_cpe = fit_eis(f, Z_noisy_cpe, model='randles_cpe', weighting='modulus')
    Rs_fit_cpe, Rct_fit_cpe, Q_fit, n_fit = result_cpe.x
    Z_fit_cpe = randles_cpe_impedance(f, Rs_fit_cpe, Rct_fit_cpe, Q_fit, n_fit)
    
    print(f"\nTrue CPE Parameters:")
    print(f"  Rs  = {Rs_true} Ω")
    print(f"  Rct = {Rct_true} Ω")
    print(f"  Q   = {Q_true} F·s^(n-1)")
    print(f"  n   = {n_true}")
    
    print(f"\nFitted CPE Parameters:")
    print(f"  Rs  = {Rs_fit_cpe:.4f} Ω (error: {abs((Rs_fit_cpe - Rs_true)/Rs_true*100):.2f}%)")
    print(f"  Rct = {Rct_fit_cpe:.4f} Ω (error: {abs((Rct_fit_cpe - Rct_true)/Rct_true*100):.2f}%)")
    print(f"  Q   = {Q_fit:.4e} F·s^(n-1) (error: {abs((Q_fit - Q_true)/Q_true*100):.2f}%)")
    print(f"  n   = {n_fit:.4f} (error: {abs((n_fit - n_true)/n_true*100):.2f}%)")
    
    chi_sq_cpe, chi_sq_red_cpe, _, _, _, _ = calculate_statistics(Z_noisy_cpe, Z_fit_cpe, 4)
    print(f"\nGoodness of Fit:")
    print(f"  Reduced χ² = {chi_sq_red_cpe:.6f}")
    
    print(f"\n{'=' * 80}")
    print("DISTRIBUTION OF RELAXATION TIMES (DRT) ANALYSIS")
    print(f"{'=' * 80}")
    print(f"""
DRT Analysis Overview:
- No equivalent circuit assumption required
- Directly extracts relaxation time distribution from impedance data
- Ideal for identifying multi-step electrochemical processes
- Uses Tikhonov regularization for stable inversion
    """)
    
    print(f"\n--- Demo 1: DRT on single relaxation (Randles circuit) ---")
    drt_result_single = compute_drt(f, Z_noisy, lambda_reg=1e-3)
    
    print(f"  Estimated Rs = {drt_result_single['Rs']:.4f} Ω")
    print(f"  Total polarization resistance = {drt_result_single['R_total']:.4f} Ω")
    print(f"  (True Rct = {Rct_true} Ω)")
    
    peaks_single = find_drt_peaks(drt_result_single['tau'], drt_result_single['gamma'])
    print(f"  Found {len(peaks_single)} relaxation process(es):")
    for i, peak in enumerate(peaks_single):
        print(f"    Peak {i+1}: τ={peak['tau']:.2e}s, f={peak['f_peak']:.1f}Hz, Area={peak['area']:.2f}Ω")
    
    fig_drt1 = plot_drt_comparison(f, Z_noisy, drt_result_single, 
                                   title='DRT Analysis - Single Relaxation Process')
    fig_drt1.savefig('drt_single_relaxation.png', dpi=300, bbox_inches='tight')
    
    print(f"\n--- Demo 2: DRT on multi-relaxation process (2 time constants) ---")
    Rs_multi = 10
    R_list = [80, 120]
    tau_list = [1e-3, 1e-1]
    Z_multi_noisy, Z_multi_true = generate_multi_relaxation_data(
        f, Rs_multi, R_list, tau_list, noise_level=0.02
    )
    
    print(f"  True parameters:")
    print(f"    Rs = {Rs_multi} Ω")
    for i, (R, tau) in enumerate(zip(R_list, tau_list)):
        print(f"    Process {i+1}: R={R}Ω, τ={tau:.2e}s, f={1/(2*np.pi*tau):.1f}Hz")
    
    print(f"\n  Optimizing regularization parameter λ using cross-validation...")
    best_lambda, lambda_values, cv_errors = optimize_lambda_cv(f, Z_multi_noisy, cv_folds=5)
    print(f"  Optimal λ = {best_lambda:.2e}")
    
    drt_result_multi = compute_drt(f, Z_multi_noisy, lambda_reg=best_lambda)
    
    print(f"\n  DRT Results:")
    print(f"  Estimated Rs = {drt_result_multi['Rs']:.4f} Ω")
    print(f"  Total polarization resistance = {drt_result_multi['R_total']:.4f} Ω")
    print(f"  (True total Rp = {sum(R_list)} Ω)")
    
    peaks_multi = find_drt_peaks(drt_result_multi['tau'], drt_result_multi['gamma'])
    print(f"  Found {len(peaks_multi)} relaxation process(es):")
    for i, peak in enumerate(peaks_multi):
        print(f"    Peak {i+1}: τ={peak['tau']:.2e}s, f={peak['f_peak']:.1f}Hz, Area={peak['area']:.2f}Ω")
    
    fig_drt2 = plot_drt_comparison(f, Z_multi_noisy, drt_result_multi,
                                   title='DRT Analysis - Multi-Relaxation Process')
    fig_drt2.savefig('drt_multi_relaxation.png', dpi=300, bbox_inches='tight')
    
    fig_cv, ax_cv = plt.subplots(figsize=(8, 5))
    ax_cv.semilogx(lambda_values, cv_errors, 'o-', linewidth=2)
    ax_cv.axvline(x=best_lambda, color='r', linestyle='--', label=f'Optimal λ = {best_lambda:.2e}')
    ax_cv.set_xlabel('Regularization Parameter λ', fontsize=12)
    ax_cv.set_ylabel('Cross-Validation Error', fontsize=12)
    ax_cv.set_title('λ Optimization via Cross-Validation', fontsize=14, fontweight='bold')
    ax_cv.legend(fontsize=10)
    ax_cv.grid(True, alpha=0.3, which='both')
    fig_cv.savefig('drt_cv_optimization.png', dpi=300, bbox_inches='tight')
    
    print(f"\nDRT plots saved:")
    print(f"  - drt_single_relaxation.png")
    print(f"  - drt_multi_relaxation.png")
    print(f"  - drt_cv_optimization.png")
    
    print(f"\n{'=' * 80}")
    print("FITTING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"""
=== Equivalent Circuit Fitting ===
  result = fit_eis(f, Z, model='randles', weighting='modulus')

  Available weighting methods:
  - 'none'      : Equal weighting (not recommended)
  - 'modulus'   : 1/|Z|² weighting (RECOMMENDED, default)
  - 'variance'  : Based on measurement noise (requires sigma_real, sigma_imag)

=== DRT Analysis (No Circuit Assumption) ===
  drt_result = compute_drt(f, Z, lambda_reg=1e-3)
  
  Auto-optimize λ:
  best_lambda, _, _ = optimize_lambda_cv(f, Z)
  drt_result = compute_drt(f, Z, lambda_reg=best_lambda)
  
  Find relaxation processes:
  peaks = find_drt_peaks(drt_result['tau'], drt_result['gamma'])
    """)
    
    plt.close('all')


if __name__ == "__main__":
    main()
