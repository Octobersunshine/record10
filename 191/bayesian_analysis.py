import numpy as np
import emcee
import corner
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from spectrum_models import (
    combined_model, convert_to_physical, get_default_bounds, get_initial_guess
)
from data_loader import apply_response


def log_prior(params):
    lower, upper = get_default_bounds()
    
    for i, (p, lo, hi) in enumerate(zip(params, lower, upper)):
        if p < lo or p > hi:
            return -np.inf
    
    T1, log_EM1, T2, log_EM2, log_norm_pl, index_pl = params
    
    lp = 0.0
    
    lp += norm.logpdf(T1, loc=10.0, scale=5.0)
    lp += norm.logpdf(T2, loc=25.0, scale=10.0)
    
    lp += norm.logpdf(log_EM1, loc=45.0, scale=1.0)
    lp += norm.logpdf(log_EM2, loc=44.0, scale=1.0)
    
    lp += norm.logpdf(log_norm_pl, loc=-35.0, scale=5.0)
    
    lp += norm.logpdf(index_pl, loc=3.0, scale=1.5)
    
    return lp


def log_likelihood(params, energy_centers, counts, energy_widths, exposure, drm):
    true_flux = combined_model(energy_centers, params)
    predicted = apply_response(true_flux, drm, energy_widths, exposure)
    predicted = np.maximum(predicted, 1e-10)
    
    mask = counts > 0
    ll = np.sum(counts[mask] * np.log(predicted[mask]) - predicted[mask])
    
    if not np.isfinite(ll):
        return -np.inf
    
    return ll


def log_posterior(params, energy_centers, counts, energy_widths, exposure, drm):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood(params, energy_centers, counts, energy_widths, exposure, drm)
    if not np.isfinite(ll):
        return -np.inf
    
    return lp + ll


def run_mcmc(energy_centers, counts, errors, energy_widths, exposure, drm=None,
             nwalkers=32, nsteps=5000, burnin=1000, initial_guess=None):
    
    if drm is None:
        drm = np.eye(len(energy_centers))
    
    if initial_guess is None:
        initial_guess = get_initial_guess(energy_centers, counts, energy_widths, exposure)
    
    ndim = len(initial_guess)
    
    lower, upper = get_default_bounds()
    pos = []
    for _ in range(nwalkers):
        p = np.array(initial_guess) + np.random.randn(ndim) * np.array([0.1, 0.01, 0.5, 0.01, 0.1, 0.05])
        for i in range(ndim):
            p[i] = np.clip(p[i], lower[i] + 0.01 * (upper[i] - lower[i]), 
                          upper[i] - 0.01 * (upper[i] - lower[i]))
        pos.append(p)
    pos = np.array(pos)
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=(energy_centers, counts, energy_widths, exposure, drm)
    )
    
    print(f"Running MCMC: {nwalkers} walkers, {nsteps} steps...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    samples = sampler.get_chain(discard=burnin, flat=True)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True)
    
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    tau = sampler.get_autocorr_time(tol=0)
    print(f"Autocorrelation times: {tau}")
    
    return {
        'sampler': sampler,
        'samples': samples,
        'log_prob': log_prob_samples,
        'ndim': ndim,
        'nwalkers': nwalkers,
        'nsteps': nsteps,
        'burnin': burnin,
        'initial_guess': initial_guess
    }


def savage_dickey_test(samples_full, param_index, test_value, prior_loc=0.0, prior_std=1.0):
    samples_param = samples_full[:, param_index]
    
    kernel = gaussian_kde(samples_param, bw_method='scott')
    posterior_at_test = kernel.evaluate([test_value])[0]
    
    prior_at_test = norm.pdf(test_value, loc=prior_loc, scale=prior_std)
    
    bf = posterior_at_test / prior_at_test if prior_at_test > 0 else np.inf
    
    return {
        'bayes_factor': bf,
        'posterior_at_test': posterior_at_test,
        'prior_at_test': prior_at_test,
        'interpretation': interpret_bf(bf)
    }


def savage_dickey_delta_test(samples_full, idx1, idx2, delta_test=0.0, 
                            prior_delta_loc=15.0, prior_delta_std=11.0):
    samples1 = samples_full[:, idx1]
    samples2 = samples_full[:, idx2]
    delta_samples = samples1 - samples2
    
    kernel = gaussian_kde(delta_samples, bw_method='scott')
    posterior_at_test = kernel.evaluate([delta_test])[0]
    
    prior_at_test = norm.pdf(delta_test, loc=prior_delta_loc, scale=prior_delta_std)
    
    bf = posterior_at_test / prior_at_test if prior_at_test > 0 else np.inf
    
    return {
        'bayes_factor': bf,
        'posterior_at_test': posterior_at_test,
        'prior_at_test': prior_at_test,
        'delta_samples': delta_samples,
        'interpretation': interpret_bf(bf)
    }


def savage_dickey_boundary_test(samples_full, param_index, bound_value, 
                                param_lower, param_upper, prior_width=None):
    samples_param = samples_full[:, param_index]
    
    kernel = gaussian_kde(samples_param, bw_method='scott')
    posterior_at_bound = kernel.evaluate([bound_value])[0]
    
    if prior_width is None:
        prior_width = param_upper - param_lower
    prior_at_bound = 1.0 / prior_width
    
    bf = posterior_at_bound / prior_at_bound if prior_at_bound > 0 else np.inf
    
    return {
        'bayes_factor': bf,
        'posterior_at_bound': posterior_at_bound,
        'prior_at_bound': prior_at_bound,
        'interpretation': interpret_bf(bf)
    }


def interpret_bf(bf):
    if bf > 100:
        return "Strong evidence for H0"
    elif bf > 10:
        return "Moderate evidence for H0"
    elif bf > 3:
        return "Weak evidence for H0"
    elif bf > 1/3:
        return "Inconclusive"
    elif bf > 1/10:
        return "Weak evidence for H1"
    elif bf > 1/100:
        return "Moderate evidence for H1"
    else:
        return "Strong evidence for H1"


def compare_models(mcmc_result_full, mcmc_result_restricted):
    log_prob_full = mcmc_result_full['log_prob']
    log_prob_restricted = mcmc_result_restricted['log_prob']
    
    max_log_prob_full = np.max(log_prob_full)
    max_log_prob_restricted = np.max(log_prob_restricted)
    
    bf_approx = np.exp(max_log_prob_full - max_log_prob_restricted)
    
    return {
        'bayes_factor': bf_approx,
        'max_log_prob_full': max_log_prob_full,
        'max_log_prob_restricted': max_log_prob_restricted,
        'interpretation': interpret_bf(bf_approx)
    }


def plot_corner(mcmc_result, param_names=None, save_path='corner_plot.png'):
    samples = mcmc_result['samples']
    
    if param_names is None:
        param_names = [
            'T1 (keV)', 'log$_{10}$(EM1)', 'T2 (keV)', 
            'log$_{10}$(EM2)', 'log$_{10}$(norm$_{pl}$)', 'index$_{pl}$'
        ]
    
    fig = corner.corner(
        samples,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        levels=[0.393, 0.865, 0.989],
        color='steelblue',
        title_fmt='.3f'
    )
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Corner plot saved to: {save_path}")
    
    return fig


def plot_traces(mcmc_result, param_names=None, save_path='trace_plot.png'):
    sampler = mcmc_result['sampler']
    nsteps = mcmc_result['nsteps']
    ndim = mcmc_result['ndim']
    
    if param_names is None:
        param_names = [
            'T1 (keV)', 'log$_{10}$(EM1)', 'T2 (keV)', 
            'log$_{10}$(EM2)', 'log$_{10}$(norm$_{pl}$)', 'index$_{pl}$'
        ]
    
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 3 * ndim), sharex=True)
    
    for i in range(ndim):
        ax = axes[i]
        chain = sampler.get_chain()[:, :, i]
        ax.plot(chain, "k", alpha=0.3, linewidth=0.5)
        ax.set_ylabel(param_names[i])
        ax.axvline(mcmc_result['burnin'], color='r', linestyle='--', label='Burn-in')
        
        if i == 0:
            ax.legend()
    
    axes[-1].set_xlabel("Step number")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trace plot saved to: {save_path}")
    
    return fig


def plot_correlation_matrix(mcmc_result, param_names=None, save_path='correlation_matrix.png'):
    samples = mcmc_result['samples']
    corr = np.corrcoef(samples.T)
    
    if param_names is None:
        param_names = [
            'T1', 'log(EM1)', 'T2', 
            'log(EM2)', 'log(norm$_{pl}$)', 'index$_{pl}$'
        ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{corr[i, j]:.2f}',
                          ha="center", va="center", color="w", fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation coefficient')
    ax.set_title('Parameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix saved to: {save_path}")
    
    return fig


def get_mcmc_summary(mcmc_result, param_names=None):
    samples = mcmc_result['samples']
    
    if param_names is None:
        param_names = [
            'T1 (keV)', 'log10(EM1)', 'T2 (keV)', 
            'log10(EM2)', 'log10(norm_pl)', 'index_pl'
        ]
    
    summary = {}
    for i, name in enumerate(param_names):
        s = samples[:, i]
        summary[name] = {
            'mean': np.mean(s),
            'median': np.median(s),
            'std': np.std(s),
            '16th': np.percentile(s, 16),
            '84th': np.percentile(s, 84),
            '2.5th': np.percentile(s, 2.5),
            '97.5th': np.percentile(s, 97.5)
        }
    
    return summary


def print_mcmc_summary(mcmc_result, true_params=None):
    samples = mcmc_result['samples']
    
    param_names = [
        'T1 (keV)', 'log10(EM1)', 'T2 (keV)', 
        'log10(EM2)', 'log10(norm_pl)', 'index_pl'
    ]
    
    phys_param_names = [
        'T1 (keV)', 'EM1 (cm^-3)', 'T2 (keV)', 
        'EM2 (cm^-3)', 'norm_pl', 'index_pl'
    ]
    
    print("\n" + "=" * 90)
    print("MCMC PARAMETER SUMMARY")
    print("=" * 90)
    
    print(f"\n{'Parameter':<20} {'Mean':<15} {'Median':<15} {'Std':<15} {'95% Credible Interval':<30}", end='')
    if true_params is not None:
        true_phys = convert_to_physical(true_params)
        print(f" {'True Value':<15}")
    else:
        print()
    
    print("-" * 120)
    
    for i in range(6):
        s = samples[:, i]
        mean = np.mean(s)
        median = np.median(s)
        std = np.std(s)
        ci_low = np.percentile(s, 2.5)
        ci_high = np.percentile(s, 97.5)
        
        if i in [0, 2, 5]:
            print(f"{phys_param_names[i]:<20} {mean:<15.4f} {median:<15.4f} {std:<15.4f} [{ci_low:<8.3f}, {ci_high:<8.3f}]", end='')
        elif i == 1:
            phys_val = 10 ** mean
            phys_low = 10 ** ci_low
            phys_high = 10 ** ci_high
            print(f"{phys_param_names[i]:<20} {phys_val:<15.3e} {10**median:<15.3e} {10**std:<15.3e} [{phys_low:<8.2e}, {phys_high:<8.2e}]", end='')
        elif i == 3:
            phys_val = 10 ** mean
            phys_low = 10 ** ci_low
            phys_high = 10 ** ci_high
            print(f"{phys_param_names[i]:<20} {phys_val:<15.3e} {10**median:<15.3e} {10**std:<15.3e} [{phys_low:<8.2e}, {phys_high:<8.2e}]", end='')
        elif i == 4:
            phys_val = 10 ** mean
            phys_low = 10 ** ci_low
            phys_high = 10 ** ci_high
            print(f"{phys_param_names[i]:<20} {phys_val:<15.3e} {10**median:<15.3e} {10**std:<15.3e} [{phys_low:<8.2e}, {phys_high:<8.2e}]", end='')
        
        if true_params is not None:
            true_phys = convert_to_physical(true_params)
            keys = ['T1', 'EM1', 'T2', 'EM2', 'norm_pl', 'index_pl']
            true_val = true_phys[keys[i]]
            if i in [0, 2, 5]:
                print(f" {true_val:<15.4f}")
            else:
                print(f" {true_val:<15.3e}")
        else:
            print()
    
    print("=" * 90)
    
    corr = np.corrcoef(samples.T)
    print("\nKey Parameter Correlations:")
    print(f"  T1 vs index_pl: {corr[0, 5]:.3f}")
    print(f"  T2 vs index_pl: {corr[2, 5]:.3f}")
    print(f"  T1 vs T2: {corr[0, 2]:.3f}")
    print(f"  log(EM1) vs log(EM2): {corr[1, 3]:.3f}")
    print(f"  log(norm_pl) vs index_pl: {corr[4, 5]:.3f}")
    print("=" * 90 + "\n")
