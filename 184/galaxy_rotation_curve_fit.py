import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


def nfw_v_circular(r, rho_s, r_s):
    G = 4.30091e-6
    r_s_kpc = r_s
    x = r / r_s_kpc
    rho_s_converted = rho_s * 1e9
    M_enclosed = 4 * np.pi * rho_s_converted * r_s_kpc**3 * (np.log(1 + x) - x / (1 + x))
    v_circular = np.sqrt(G * M_enclosed / r)
    return v_circular


def burkert_v_circular(r, rho_0, r_0):
    G = 4.30091e-6
    r_0_kpc = r_0
    x = r / r_0_kpc
    rho_0_converted = rho_0 * 1e9
    term1 = np.log(1 + x)
    term2 = -0.5 * np.log(1 + x**2)
    term3 = -np.arctan(x)
    M_enclosed = 4 * np.pi * rho_0_converted * r_0_kpc**3 * (term1 + term2 + term3)
    M_enclosed = np.maximum(M_enclosed, 1e-10)
    v_circular = np.sqrt(G * M_enclosed / r)
    return v_circular


def generate_sample_data(r_min=1, r_max=30, n_points=20, noise=5):
    r = np.linspace(r_min, r_max, n_points)
    true_rho_s = 5.0
    true_r_s = 15.0
    v_true = nfw_v_circular(r, true_rho_s, true_r_s)
    v_obs = v_true + np.random.normal(0, noise, n_points)
    v_err = np.full_like(v_obs, noise)
    return r, v_obs, v_err, (true_rho_s, true_r_s)


def fit_rotation_curve(r, v, v_err, model='nfw'):
    if model == 'nfw':
        func = nfw_v_circular
        p0 = [5.0, 15.0]
        bounds = ([0.01, 1.0], [100.0, 100.0])
    elif model == 'burkert':
        func = burkert_v_circular
        p0 = [5.0, 15.0]
        bounds = ([0.01, 1.0], [100.0, 100.0])
    else:
        raise ValueError("Model must be 'nfw' or 'burkert'")
    
    popt, pcov = curve_fit(func, r, v, p0=p0, sigma=v_err, 
                          bounds=bounds, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr, pcov


def calculate_goodness_of_fit(r, v, v_err, popt, model):
    if model == 'nfw':
        v_pred = nfw_v_circular(r, *popt)
    else:
        v_pred = burkert_v_circular(r, *popt)
    
    chi2 = np.sum(((v - v_pred) / v_err)**2)
    dof = len(r) - len(popt)
    reduced_chi2 = chi2 / dof
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    
    return chi2, reduced_chi2, p_value


def plot_fit_results(r, v, v_err, popt_nfw, popt_burkert):
    plt.figure(figsize=(12, 8))
    
    r_fine = np.linspace(min(r), max(r), 1000)
    
    plt.errorbar(r, v, yerr=v_err, fmt='o', color='black', 
                 label='Observations', capsize=5, markersize=6)
    
    v_nfw = nfw_v_circular(r_fine, *popt_nfw)
    plt.plot(r_fine, v_nfw, 'r-', linewidth=2, 
             label=f'NFW fit\n$\\rho_s$={popt_nfw[0]:.2f} $\\times 10^{-9}$ M$_\\odot$/pc$^3$\n$r_s$={popt_nfw[1]:.2f} kpc')
    
    v_burkert = burkert_v_circular(r_fine, *popt_burkert)
    plt.plot(r_fine, v_burkert, 'b--', linewidth=2, 
             label=f'Burkert fit\n$\\rho_0$={popt_burkert[0]:.2f} $\\times 10^{-9}$ M$_\\odot$/pc$^3$\n$r_0$={popt_burkert[1]:.2f} kpc')
    
    plt.xlabel('Radius r (kpc)', fontsize=14)
    plt.ylabel('Rotation Velocity v (km/s)', fontsize=14)
    plt.title('Galaxy Rotation Curve Fitting', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('rotation_curve_fit.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    np.random.seed(42)
    
    print("=" * 60)
    print("Galaxy Rotation Curve Fitting with Dark Matter Halo Models")
    print("=" * 60)
    
    print("\nGenerating sample observation data...")
    r, v_obs, v_err, true_params = generate_sample_data()
    
    print(f"True NFW parameters used to generate data:")
    print(f"  rho_s = {true_params[0]:.2f} x 10^-9 M_sun/pc^3")
    print(f"  r_s = {true_params[1]:.2f} kpc")
    
    print("\n" + "-" * 60)
    print("Fitting NFW profile...")
    popt_nfw, perr_nfw, _ = fit_rotation_curve(r, v_obs, v_err, model='nfw')
    chi2_nfw, rchi2_nfw, pval_nfw = calculate_goodness_of_fit(r, v_obs, v_err, popt_nfw, 'nfw')
    
    print(f"Best-fit parameters:")
    print(f"  rho_s = {popt_nfw[0]:.2f} +/- {perr_nfw[0]:.2f} x 10^-9 M_sun/pc^3")
    print(f"  r_s = {popt_nfw[1]:.2f} +/- {perr_nfw[1]:.2f} kpc")
    print(f"Goodness of fit:")
    print(f"  chi^2 = {chi2_nfw:.2f}")
    print(f"  reduced chi^2 = {rchi2_nfw:.2f}")
    print(f"  p-value = {pval_nfw:.4f}")
    
    print("\n" + "-" * 60)
    print("Fitting Burkert profile...")
    popt_burkert, perr_burkert, _ = fit_rotation_curve(r, v_obs, v_err, model='burkert')
    chi2_burkert, rchi2_burkert, pval_burkert = calculate_goodness_of_fit(r, v_obs, v_err, popt_burkert, 'burkert')
    
    print(f"Best-fit parameters:")
    print(f"  rho_0 = {popt_burkert[0]:.2f} +/- {perr_burkert[0]:.2f} x 10^-9 M_sun/pc^3")
    print(f"  r_0 = {popt_burkert[1]:.2f} +/- {perr_burkert[1]:.2f} kpc")
    print(f"Goodness of fit:")
    print(f"  chi^2 = {chi2_burkert:.2f}")
    print(f"  reduced chi^2 = {rchi2_burkert:.2f}")
    print(f"  p-value = {pval_burkert:.4f}")
    
    print("\n" + "-" * 60)
    print("Model comparison:")
    if rchi2_nfw < rchi2_burkert:
        print(f"NFW model provides better fit (reduced chi^2 = {rchi2_nfw:.2f} vs {rchi2_burkert:.2f})")
    else:
        print(f"Burkert model provides better fit (reduced chi^2 = {rchi2_burkert:.2f} vs {rchi2_nfw:.2f})")
    
    print("\n" + "=" * 60)
    print("Generating plot...")
    plot_fit_results(r, v_obs, v_err, popt_nfw, popt_burkert)
    print("Plot saved as 'rotation_curve_fit.png'")
    print("=" * 60)


if __name__ == "__main__":
    main()
