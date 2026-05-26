from mcmc_rotation_curve_fitter import MCMCRotationCurveFitter
from rotation_curve_fitter import GalaxyRotationCurveFitter
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

print("=" * 75)
print("COMPREHENSIVE MCMC ANALYSIS DEMONSTRATION")
print("=" * 75)

print("\n" + "=" * 75)
print("PART 1: DEMONSTRATING PARAMETER CORRELATION PROBLEM")
print("=" * 75)

fitter_ml = GalaxyRotationCurveFitter()
fitter_ml.generate_sample_data(r_min=2, r_max=35, n_points=25, noise=8, 
                               model='nfw', true_params=(5.0, 15.0))
fitter_ml.fit_both()

print("\nTraditional Least-Squares Results:")
print("-" * 75)
print(f"NFW: rho_s = {fitter_ml.nfw_params[0]:.4f} +/- {fitter_ml.nfw_errors[0]:.4f}")
print(f"     r_s   = {fitter_ml.nfw_params[1]:.4f} +/- {fitter_ml.nfw_errors[1]:.4f}")
print(f"     These errors IGNORE parameter correlations!")

print("\n" + "=" * 75)
print("PART 2: FULL BAYESIAN MCMC ANALYSIS")
print("=" * 75)

fitter_mcmc = MCMCRotationCurveFitter()
fitter_mcmc.load_data(fitter_ml.r, fitter_ml.v, fitter_ml.v_err)

fitter_mcmc.find_maximum_likelihood('nfw')
print(f"\nMaximum Likelihood: rho_s = {fitter_mcmc.ml_params[0]:.4f}, r_s = {fitter_mcmc.ml_params[1]:.4f}")

fitter_mcmc.run_mcmc('nfw', n_walkers=32, n_steps=5000, n_burn=1000)

print("\n" + "=" * 75)
print("PART 3: COMPARISON - ML vs Bayesian")
print("=" * 75)

print(f"\n{'Method':<20} {'rho_s':<20} {'r_s':<20}")
print("-" * 60)
print(f"{'True values':<20} {5.0000:<20.4f} {15.0000:<20.4f}")
print(f"{'Least-squares':<20} {fitter_ml.nfw_params[0]:<20.4f} {fitter_ml.nfw_params[1]:<20.4f}")
print(f"{'MCMC median':<20} {fitter_mcmc.best_fit_params[0]:<20.4f} {fitter_mcmc.best_fit_params[1]:<20.4f}")

print("\n" + "=" * 75)
print("PART 4: PARAMETER UNCERTAINTY COMPARISON")
print("=" * 75)

print(f"\n{'Method':<20} {'rho_s error':<15} {'r_s error':<15}")
print("-" * 50)
print(f"{'Least-squares':<20} {fitter_ml.nfw_errors[0]:<15.4f} {fitter_ml.nfw_errors[1]:<15.4f}")
print(f"{'MCMC (1 sigma)':<20} {fitter_mcmc.param_uncertainties[0]:<15.4f} {fitter_mcmc.param_uncertainties[1]:<15.4f}")

corr = fitter_mcmc.get_correlation_matrix()
print(f"\nParameter correlation: {corr[0,1]:.4f}")
print(f"{'='*75}")
print(f"IMPORTANT: The errors from least-squares are often UNDERESTIMATED")
print(f"because they don't account for parameter degeneracies.")
print(f"MCMC provides correct marginalized uncertainties.")
print(f"{'='*75}")

print("\n" + "=" * 75)
print("PART 5: CORRELATED PARAMETER PAIRS")
print("=" * 75)

samples = fitter_mcmc.flat_samples
print(f"\nExploring the (rho_s, r_s) degeneracy:")
print(f"  When rho_s INCREASES, r_s must DECREASE to maintain similar rotation curves")
print(f"  This is the classic mass-concentration degeneracy")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax1 = axes[0, 0]
ax1.scatter(samples[::10, 0], samples[::10, 1], alpha=0.1, s=1, color='steelblue')
ax1.set_xlabel(r'$\rho_s$ [$\times 10^{-9}$ M$_\odot$/pc$^3$]', fontsize=12)
ax1.set_ylabel(r'$r_s$ [kpc]', fontsize=12)
ax1.set_title('Parameter Degeneracy', fontsize=14, fontweight='bold')
ax1.axvline(x=5.0, color='red', linestyle='--', alpha=0.7, label='True value')
ax1.axhline(y=15.0, color='red', linestyle='--', alpha=0.7)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.hist(samples[:, 0], bins=50, density=True, color='steelblue', alpha=0.7)
ax2.axvline(x=5.0, color='red', linestyle='--', label='True value')
ax2.axvline(x=np.median(samples[:, 0]), color='black', linestyle='-', label='Median')
ax2.set_xlabel(r'$\rho_s$ [$\times 10^{-9}$ M$_\odot$/pc$^3$]', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.set_title(r'Posterior for $\rho_s$', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.hist(samples[:, 1], bins=50, density=True, color='steelblue', alpha=0.7)
ax3.axvline(x=15.0, color='red', linestyle='--', label='True value')
ax3.axvline(x=np.median(samples[:, 1]), color='black', linestyle='-', label='Median')
ax3.set_xlabel(r'$r_s$ [kpc]', fontsize=12)
ax3.set_ylabel('Probability Density', fontsize=12)
ax3.set_title(r'Posterior for $r_s$', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
r_fine = np.linspace(2, 35, 200)
n_plot = min(200, len(samples))
indices = np.random.choice(len(samples), n_plot, replace=False)

for idx in indices:
    v_pred = fitter_mcmc.nfw_v_circular(r_fine, *samples[idx])
    ax4.plot(r_fine, v_pred, color='steelblue', alpha=0.05)

v_true = fitter_mcmc.nfw_v_circular(r_fine, 5.0, 15.0)
ax4.plot(r_fine, v_true, color='red', linewidth=2, label='True model')

v_best = fitter_mcmc.nfw_v_circular(r_fine, *fitter_mcmc.best_fit_params)
ax4.plot(r_fine, v_best, color='black', linewidth=2, label='Best fit')

ax4.errorbar(fitter_mcmc.r, fitter_mcmc.v, yerr=fitter_mcmc.v_err, 
             fmt='o', color='black', capsize=3, markersize=4, label='Data')
ax4.set_xlabel('Radius r (kpc)', fontsize=12)
ax4.set_ylabel('v (km/s)', fontsize=12)
ax4.set_title('Posterior Rotation Curves', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved comprehensive analysis plot: comprehensive_analysis.png")

print("\n" + "=" * 75)
print("PART 6: USING REAL OBSERVATION DATA")
print("=" * 75)

fitter_real = MCMCRotationCurveFitter()
fitter_real.load_from_file('sample_observation_data.csv', delimiter=',', skiprows=2)

print(f"\nLoaded real observation data: {len(fitter_real.r)} points")
print(f"Radius: {fitter_real.r.min():.1f} - {fitter_real.r.max():.1f} kpc")

fitter_real.find_maximum_likelihood('nfw')
fitter_real.run_mcmc('nfw', n_walkers=32, n_steps=5000, n_burn=1000)

print("\nReal Data MCMC Results:")
print("-" * 75)
print(f"  rho_s = {fitter_real.best_fit_params[0]:.4f} +/- {fitter_real.param_uncertainties[0]:.4f}")
print(f"  r_s   = {fitter_real.best_fit_params[1]:.4f} +/- {fitter_real.param_uncertainties[1]:.4f}")

corr_real = fitter_real.get_correlation_matrix()
print(f"\n  Parameter correlation: {corr_real[0,1]:.4f}")

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)
print("""
Key findings:
1. NFW halo parameters (rho_s, r_s) are HIGHLY correlated (-0.99)
2. This degeneracy arises because rotation curves are mostly flat
3. Traditional least-squares errors can be misleading
4. MCMC correctly accounts for parameter correlations
5. The posterior is elongated in the correlated direction

Recommended workflow:
1. Start with maximum likelihood to find a good starting point
2. Run MCMC with multiple walkers
3. Check convergence using autocorrelation time
4. Examine corner plots for parameter degeneracies
5. Report median +/- 1 sigma from the posterior
""")

print("Generated files:")
print("  nfw_trace.png - MCMC chain convergence")
print("  nfw_corner.png - Posterior distribution")
print("  nfw_fit_uncertainty.png - Fit with uncertainty bands")
print("  comprehensive_analysis.png - Full comparison analysis")
print("=" * 75)
