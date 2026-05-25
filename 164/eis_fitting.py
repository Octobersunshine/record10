import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def randles_impedance(f, Rs, Rct, Cdl):
    omega = 2 * np.pi * f
    Z_cdl = 1 / (1j * omega * Cdl)
    Z_parallel = (Rct * Z_cdl) / (Rct + Z_cdl)
    Z_total = Rs + Z_parallel
    return Z_total


def residuals(params, f, Z_exp):
    Rs, Rct, Cdl = params
    Z_model = randles_impedance(f, Rs, Rct, Cdl)
    res_real = Z_model.real - Z_exp.real
    res_imag = Z_model.imag - Z_exp.imag
    return np.concatenate([res_real, res_imag])


def fit_eis(f, Z_exp, initial_guess=None, bounds=None):
    if initial_guess is None:
        initial_guess = [10, 100, 1e-6]
    
    if bounds is None:
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    
    result = least_squares(
        residuals,
        initial_guess,
        bounds=bounds,
        args=(f, Z_exp),
        method='trf',
        max_nfev=10000
    )
    
    return result


def generate_sample_data(f, Rs_true=10, Rct_true=150, Cdl_true=2e-6, noise_level=0.02):
    Z_true = randles_impedance(f, Rs_true, Rct_true, Cdl_true)
    real_noise = np.random.normal(0, noise_level * np.abs(Z_true), Z_true.shape)
    imag_noise = np.random.normal(0, noise_level * np.abs(Z_true), Z_true.shape)
    Z_noisy = Z_true + real_noise + 1j * imag_noise
    return Z_noisy, Z_true


def plot_nyquist(Z_exp, Z_fit=None, title='Nyquist Plot'):
    plt.figure(figsize=(8, 6))
    plt.plot(Z_exp.real, -Z_exp.imag, 'o', label='Experimental Data', markersize=6)
    if Z_fit is not None:
        plt.plot(Z_fit.real, -Z_fit.imag, '-', label='Fitted Model', linewidth=2)
    plt.xlabel('Z_real (Ω)')
    plt.ylabel('-Z_imag (Ω)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_bode(f, Z_exp, Z_fit=None, title='Bode Plot'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    mag_exp = np.abs(Z_exp)
    phase_exp = np.angle(Z_exp, deg=True)
    
    ax1.semilogx(f, mag_exp, 'o', label='Experimental Data', markersize=6)
    if Z_fit is not None:
        mag_fit = np.abs(Z_fit)
        ax1.semilogx(f, mag_fit, '-', label='Fitted Model', linewidth=2)
    ax1.set_ylabel('|Z| (Ω)')
    ax1.set_title(title + ' - Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogx(f, phase_exp, 'o', label='Experimental Data', markersize=6)
    if Z_fit is not None:
        phase_fit = np.angle(Z_fit, deg=True)
        ax2.semilogx(f, phase_fit, '-', label='Fitted Model', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title(title + ' - Phase')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(42)
    
    f = np.logspace(5, -1, 50)
    
    Rs_true, Rct_true, Cdl_true = 10, 150, 2e-6
    Z_noisy, Z_true = generate_sample_data(f, Rs_true, Rct_true, Cdl_true, noise_level=0.02)
    
    print("=" * 60)
    print("EIS Fitting to Randles Circuit: Rs + (Rct || Cdl)")
    print("=" * 60)
    print(f"\nTrue parameters:")
    print(f"  Rs  = {Rs_true} Ω")
    print(f"  Rct = {Rct_true} Ω")
    print(f"  Cdl = {Cdl_true} F")
    
    initial_guess = [5, 100, 1e-6]
    print(f"\nInitial guess:")
    print(f"  Rs  = {initial_guess[0]} Ω")
    print(f"  Rct = {initial_guess[1]} Ω")
    print(f"  Cdl = {initial_guess[2]} F")
    
    result = fit_eis(f, Z_noisy, initial_guess=initial_guess)
    
    Rs_fit, Rct_fit, Cdl_fit = result.x
    
    print(f"\nFitted parameters:")
    print(f"  Rs  = {Rs_fit:.4f} Ω")
    print(f"  Rct = {Rct_fit:.4f} Ω")
    print(f"  Cdl = {Cdl_fit:.4e} F")
    
    print(f"\nRelative errors:")
    print(f"  Rs  = {abs((Rs_fit - Rs_true) / Rs_true * 100):.2f}%")
    print(f"  Rct = {abs((Rct_fit - Rct_true) / Rct_true * 100):.2f}%")
    print(f"  Cdl = {abs((Cdl_fit - Cdl_true) / Cdl_true * 100):.2f}%")
    
    print(f"\nOptimization status: {result.message}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Final cost: {result.cost:.6f}")
    
    Z_fit = randles_impedance(f, Rs_fit, Rct_fit, Cdl_fit)
    
    chi_squared = np.sum((Z_noisy - Z_fit).real**2 + (Z_noisy - Z_fit).imag**2)
    print(f"Chi-squared: {chi_squared:.6f}")
    
    plot_nyquist(Z_noisy, Z_fit, title='Nyquist Plot - Randles Circuit Fitting')
    plot_bode(f, Z_noisy, Z_fit, title='Bode Plot - Randles Circuit Fitting')
    
    print("\n" + "=" * 60)
    print("Fitting complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
