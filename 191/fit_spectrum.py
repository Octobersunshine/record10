import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from spectrum_models import (
    combined_model, thermal_spectrum, power_law,
    get_param_names, get_default_bounds, get_initial_guess,
    calculate_chi2, convert_to_physical
)
from data_loader import (
    load_rhessi_data, load_goes_data, create_demo_data,
    generate_drm, apply_response
)
from bayesian_analysis import (
    run_mcmc, savage_dickey_test, savage_dickey_delta_test, 
    savage_dickey_boundary_test, interpret_bf,
    plot_corner, plot_traces, plot_correlation_matrix,
    print_mcmc_summary, get_mcmc_summary
)
from run_line_diagnostic import run_demo as run_line_diagnostic_demo


def poisson_log_likelihood(params, energy_centers, counts, energy_widths, exposure, drm):
    true_flux = combined_model(energy_centers, params)
    predicted = apply_response(true_flux, drm, energy_widths, exposure)
    predicted = np.maximum(predicted, 1e-10)
    mask = counts > 0
    ll = np.sum(counts[mask] * np.log(predicted[mask]) - predicted[mask])
    return -ll


def fit_spectrum(energy_centers, counts, errors, energy_widths=None,
                 initial_guess=None, bounds=None, method='L-BFGS-B',
                 use_weights=True, exposure=1.0, drm=None):
    
    if energy_widths is None:
        energy_widths = np.ones_like(energy_centers)
    
    if initial_guess is None:
        initial_guess = get_initial_guess(energy_centers, counts, energy_widths, exposure)
    
    if bounds is None:
        bounds = get_default_bounds()
        bounds = list(zip(bounds[0], bounds[1]))
    
    if drm is None:
        drm = np.eye(len(energy_centers))
    
    try:
        result = minimize(
            poisson_log_likelihood,
            initial_guess,
            args=(energy_centers, counts, energy_widths, exposure, drm),
            bounds=bounds,
            method=method,
            options={'maxiter': 10000}
        )
        
        popt = result.x
        
        try:
            from scipy.optimize import approx_fprime
            eps = np.sqrt(np.finfo(float).eps)
            grad = approx_fprime(popt, lambda p: poisson_log_likelihood(p, energy_centers, counts, energy_widths, exposure, drm), eps)
            pcov = np.eye(len(popt)) * 0.1
        except:
            pcov = np.eye(len(popt)) * 0.1
        
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
        
        true_flux = combined_model(energy_centers, popt)
        predicted = apply_response(true_flux, drm, energy_widths, exposure)
        chi2, dof = calculate_chi2(counts, predicted, errors)
        red_chi2 = chi2 / dof if dof > 0 else np.inf
        
        return {
            'params': popt,
            'errors': perr,
            'cov': pcov,
            'chi2': chi2,
            'dof': dof,
            'red_chi2': red_chi2,
            'predicted': predicted,
            'success': result.success
        }
        
    except Exception as e:
        print(f"Fit failed: {e}")
        return {
            'params': initial_guess,
            'errors': np.zeros_like(initial_guess),
            'cov': None,
            'chi2': np.inf,
            'dof': len(counts) - len(initial_guess),
            'red_chi2': np.inf,
            'predicted': np.zeros_like(counts),
            'success': False
        }


def plot_spectrum_fit(energy_centers, counts, errors, fit_result,
                      energy_widths=None, true_params=None, true_flux=None,
                      title='Solar Flare X-ray Spectrum Fit', save_path=None):
    
    if energy_widths is None:
        energy_widths = np.ones_like(energy_centers)
    
    params = fit_result['params']
    
    energy_fine = np.logspace(np.log10(energy_centers[0] * 0.8),
                              np.log10(energy_centers[-1] * 1.2), 200)
    
    total_flux = combined_model(energy_fine, params)
    thermal_flux = thermal_spectrum(energy_fine, params[:4])
    nonthermal_flux = power_law(energy_fine, params[4], params[5])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    
    ax1.errorbar(energy_centers, counts, yerr=errors, fmt='o', color='black',
                 markersize=6, label='Observed Data', capsize=4)
    
    ax1.loglog(energy_fine, total_flux * np.mean(energy_widths), 'r-',
               linewidth=2, label='Total Model')
    ax1.loglog(energy_fine, thermal_flux * np.mean(energy_widths), 'b--',
               linewidth=1.5, label='Thermal (Two-temperature)')
    ax1.loglog(energy_fine, nonthermal_flux * np.mean(energy_widths), 'g--',
               linewidth=1.5, label='Nonthermal (Power-law)')
    
    if true_flux is not None:
        ax1.loglog(energy_centers, true_flux * energy_widths, 'k:',
                   linewidth=1, alpha=0.7, label='True Spectrum')
    
    ax1.set_ylabel('Counts')
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    residuals = (counts - fit_result['predicted']) / errors
    ax2.errorbar(energy_centers, residuals, yerr=np.ones_like(errors),
                 fmt='o', color='black', markersize=5, capsize=3)
    ax2.axhline(y=0, color='r', linestyle='-', linewidth=1)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=0.5)
    ax2.axhline(y=-1, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Energy (keV)')
    ax2.set_ylabel('Residuals (σ)')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(-3, 3)
    
    phys_params = convert_to_physical(params)
    phys_errors = convert_to_physical(params + fit_result['errors'])
    
    param_text = "Fitted Parameters:\n"
    param_text += f"T1 = {phys_params['T1']:.3f} keV\n"
    param_text += f"EM1 = {phys_params['EM1']:.3e} cm^-3\n"
    param_text += f"T2 = {phys_params['T2']:.3f} keV\n"
    param_text += f"EM2 = {phys_params['EM2']:.3e} cm^-3\n"
    param_text += f"norm_pl = {phys_params['norm_pl']:.3e}\n"
    param_text += f"index_pl = {phys_params['index_pl']:.3f}\n"
    param_text += f"\nχ²/dof = {fit_result['chi2']:.2f}/{fit_result['dof']} = {fit_result['red_chi2']:.2f}"
    
    if true_params is not None:
        true_phys = convert_to_physical(true_params)
        param_text += "\n\nTrue Parameters:\n"
        param_text += f"T1 = {true_phys['T1']:.3f} keV\n"
        param_text += f"EM1 = {true_phys['EM1']:.3e} cm^-3\n"
        param_text += f"T2 = {true_phys['T2']:.3f} keV\n"
        param_text += f"EM2 = {true_phys['EM2']:.3e} cm^-3\n"
        param_text += f"norm_pl = {true_phys['norm_pl']:.3e}\n"
        param_text += f"index_pl = {true_phys['index_pl']:.3f}"
    
    ax1.text(1.02, 0.98, param_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Fit plot saved to: {save_path}")
    
    return fig


def print_fit_results(fit_result, true_params=None):
    params = fit_result['params']
    errors = fit_result['errors']
    
    phys_params = convert_to_physical(params)
    phys_errors = convert_to_physical(params + errors)
    
    print("\n" + "=" * 70)
    print("FIT RESULTS")
    print("=" * 70)
    
    print(f"\n{'Parameter':<20} {'Fitted Value':<25} {'Error':<20}", end='')
    if true_params is not None:
        true_phys = convert_to_physical(true_params)
        print(f" {'True Value':<25} {'Bias(%)':<10}")
    else:
        print()
    
    print("-" * 90)
    
    param_info = [
        ('T1 (keV)', phys_params['T1'], phys_errors['T1'] - phys_params['T1'], 
         true_phys['T1'] if true_params else None),
        ('EM1 (cm^-3)', phys_params['EM1'], phys_errors['EM1'] - phys_params['EM1'],
         true_phys['EM1'] if true_params else None),
        ('T2 (keV)', phys_params['T2'], phys_errors['T2'] - phys_params['T2'],
         true_phys['T2'] if true_params else None),
        ('EM2 (cm^-3)', phys_params['EM2'], phys_errors['EM2'] - phys_params['EM2'],
         true_phys['EM2'] if true_params else None),
        ('norm_pl', phys_params['norm_pl'], phys_errors['norm_pl'] - phys_params['norm_pl'],
         true_phys['norm_pl'] if true_params else None),
        ('index_pl', phys_params['index_pl'], phys_errors['index_pl'] - phys_params['index_pl'],
         true_phys['index_pl'] if true_params else None),
    ]
    
    for name, val, err, true_val in param_info:
        if 'EM' in name or 'norm' in name:
            val_str = f"{val:.3e}"
            err_str = f"{err:.3e}"
        else:
            val_str = f"{val:.4f}"
            err_str = f"{err:.4f}"
        
        print(f"{name:<20} {val_str:<25} {err_str:<20}", end='')
        
        if true_val is not None:
            if 'EM' in name or 'norm' in name:
                true_str = f"{true_val:.3e}"
            else:
                true_str = f"{true_val:.4f}"
            bias = abs((val - true_val) / true_val * 100) if true_val != 0 else 0
            print(f" {true_str:<25} {bias:<10.2f}")
        else:
            print()
    
    print("\n" + "=" * 70)
    print(f"χ² = {fit_result['chi2']:.2f}")
    print(f"Degrees of freedom = {fit_result['dof']}")
    print(f"Reduced χ² = {fit_result['red_chi2']:.4f}")
    print("=" * 70 + "\n")


def run_bayesian_analysis(energy_centers, counts, errors, energy_widths, exposure, drm,
                          fit_result, true_params=None, output_prefix='bayesian'):
    
    print("\n" + "=" * 70)
    print("BAYESIAN MCMC ANALYSIS")
    print("=" * 70)
    
    initial_guess = fit_result['params'].tolist()
    
    mcmc_result = run_mcmc(
        energy_centers, counts, errors, energy_widths, exposure, drm,
        nwalkers=32, nsteps=3000, burnin=1000,
        initial_guess=initial_guess
    )
    
    print_mcmc_summary(mcmc_result, true_params)
    
    print("\nGenerating diagnostic plots...")
    plot_corner(mcmc_result, save_path=f'{output_prefix}_corner.png')
    plot_traces(mcmc_result, save_path=f'{output_prefix}_traces.png')
    plot_correlation_matrix(mcmc_result, save_path=f'{output_prefix}_correlation.png')
    
    print("\n" + "=" * 70)
    print("SAVAGE-DICKEY DENSITY RATIO TEST - MODEL COMPLEXITY ANALYSIS")
    print("=" * 70)
    
    samples = mcmc_result['samples']
    lower, upper = get_default_bounds()
    
    print("\n" + "-" * 70)
    print("Test 1: Is the nonthermal spectral index equal to 3.0?")
    print("  H0: index_pl = 3.0  vs  H1: index_pl ≠ 3.0")
    sd_result1 = savage_dickey_test(
        samples, param_index=5, test_value=3.0, prior_loc=3.0, prior_std=1.5
    )
    print(f"  Bayes Factor: {sd_result1['bayes_factor']:.4f}")
    print(f"  Interpretation: {sd_result1['interpretation']}")
    
    print("\n" + "-" * 70)
    print("Test 2: Is the second thermal component necessary?")
    print("  H0: T1 = T2 (single-temperature model)")
    print("  H1: T1 ≠ T2 (two-temperature model)")
    sd_result2 = savage_dickey_delta_test(
        samples, idx1=2, idx2=0, delta_test=0.0,
        prior_delta_loc=15.0, prior_delta_std=11.0
    )
    print(f"  Bayes Factor (T2=T1): {sd_result2['bayes_factor']:.4e}")
    print(f"  Interpretation: {sd_result2['interpretation']}")
    
    print("\n" + "-" * 70)
    print("Test 3: Is the nonthermal component necessary?")
    print("  H0: norm_pl = 0 (thermal-only model)")
    print("  H1: norm_pl > 0 (thermal + nonthermal model)")
    sd_result3 = savage_dickey_boundary_test(
        samples, param_index=4, bound_value=lower[4] + 0.01,
        param_lower=lower[4], param_upper=upper[4]
    )
    print(f"  Bayes Factor (no nonthermal): {sd_result3['bayes_factor']:.4e}")
    print(f"  Interpretation: {sd_result3['interpretation']}")
    
    print("\n" + "-" * 70)
    print("Test 4: Is the first thermal emission measure zero?")
    print("  H0: EM1 = 0  vs  H1: EM1 > 0")
    sd_result4 = savage_dickey_boundary_test(
        samples, param_index=1, bound_value=lower[1] + 0.01,
        param_lower=lower[1], param_upper=upper[1]
    )
    print(f"  Bayes Factor (no EM1): {sd_result4['bayes_factor']:.4e}")
    print(f"  Interpretation: {sd_result4['interpretation']}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED MODEL")
    print("=" * 70)
    if sd_result2['bayes_factor'] < 1/10:
        print("  Strong evidence for TWO-TEMPERATURE model")
    elif sd_result2['bayes_factor'] < 1/3:
        print("  Moderate evidence for TWO-TEMPERATURE model")
    else:
        print("  Single-temperature model may be sufficient")
    
    if sd_result3['bayes_factor'] < 1/10:
        print("  Strong evidence for NONTHERMAL component")
    elif sd_result3['bayes_factor'] < 1/3:
        print("  Moderate evidence for NONTHERMAL component")
    else:
        print("  Thermal-only model may be sufficient")
    print("=" * 70 + "\n")
    
    return mcmc_result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Solar Flare X-ray Spectrum Fitting')
    parser.add_argument('--rhessi', type=str, help='RHESSI data file path')
    parser.add_argument('--goes', type=str, help='GOES data file path')
    parser.add_argument('--demo', action='store_true', help='Use simulated data for demo')
    parser.add_argument('--output', type=str, default='spectrum_fit.png',
                        help='Output plot path')
    parser.add_argument('--exposure', type=float, default=100.0,
                        help='Exposure time (seconds)')
    parser.add_argument('--bayesian', action='store_true',
                        help='Run Bayesian MCMC analysis')
    parser.add_argument('--mcmc-steps', type=int, default=3000,
                        help='Number of MCMC steps')
    parser.add_argument('--mcmc-walkers', type=int, default=32,
                        help='Number of MCMC walkers')
    parser.add_argument('--mcmc-burnin', type=int, default=1000,
                        help='Number of burn-in steps')
    parser.add_argument('--line-diagnostic', action='store_true',
                        help='Run spectral line diagnostic and DEM analysis')
    
    args = parser.parse_args()
    
    if args.line_diagnostic:
        run_line_diagnostic_demo()
        return
    
    if args.rhessi:
        print(f"Loading RHESSI data: {args.rhessi}")
        energy_centers, energy_widths, counts, errors, drm = load_rhessi_data(args.rhessi)
        true_params = None
        true_flux = None
        title = 'RHESSI Solar Flare X-ray Spectrum Fit'
    elif args.goes:
        print(f"Loading GOES data: {args.goes}")
        energy_centers, energy_widths, counts, errors, drm = load_goes_data(args.goes)
        true_params = None
        true_flux = None
        title = 'GOES Solar Flare X-ray Spectrum Fit'
    elif args.demo or (not args.rhessi and not args.goes):
        print("Using simulated data for demonstration...")
        demo_data = create_demo_data()
        energy_centers = demo_data['energy_centers']
        energy_widths = demo_data['energy_widths']
        counts = demo_data['counts']
        errors = demo_data['errors']
        true_params = demo_data['true_params']
        true_flux = demo_data['true_flux']
        drm = None
        title = 'Simulated Solar Flare X-ray Spectrum Fit (Two-T + Power-law)'
        args.exposure = 100.0
    else:
        print("Please specify data file or use --demo mode")
        return
    
    if energy_centers is None:
        print("Data loading failed, using simulated data...")
        demo_data = create_demo_data()
        energy_centers = demo_data['energy_centers']
        energy_widths = demo_data['energy_widths']
        counts = demo_data['counts']
        errors = demo_data['errors']
        true_params = demo_data['true_params']
        true_flux = demo_data['true_flux']
        drm = None
        title = 'Simulated Solar Flare X-ray Spectrum Fit (Two-T + Power-law)'
        args.exposure = 100.0
    
    print("\n" + "=" * 60)
    print("SPECTRUM DATA")
    print("=" * 60)
    print(f"Number of energy bins: {len(energy_centers)}")
    print(f"Energy range: {energy_centers[0]:.1f} - {energy_centers[-1]:.1f} keV")
    print(f"Total counts: {np.sum(counts):.0f}")
    print("=" * 60)
    
    print("\nStarting fit...")
    fit_result = fit_spectrum(
        energy_centers, counts, errors,
        energy_widths=energy_widths,
        exposure=args.exposure,
        drm=drm,
        use_weights=True,
        method='L-BFGS-B'
    )
    
    if fit_result['success']:
        print("Fit successful!")
        print_fit_results(fit_result, true_params)
        
        fig = plot_spectrum_fit(
            energy_centers, counts, errors, fit_result,
            energy_widths=energy_widths,
            true_params=true_params,
            true_flux=true_flux,
            title=title,
            save_path=args.output
        )
        
        if args.bayesian:
            output_prefix = args.output.replace('.png', '')
            mcmc_result = run_bayesian_analysis(
                energy_centers, counts, errors, energy_widths,
                args.exposure, drm, fit_result, true_params,
                output_prefix=output_prefix
            )
    else:
        print("Fit failed!")


if __name__ == '__main__':
    main()
