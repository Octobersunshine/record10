from comprehensive_rotation_curve import ComprehensiveRotationCurveFitter
from baryon_models import generate_sample_rotation_curve, BaryonicModels
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

print("=" * 80)
print("COMPREHENSIVE GALAXY ROTATION CURVE ANALYSIS")
print("=" * 80)
print("\nThis demonstration includes:")
print("  1. Baryonic mass models (stars, gas with population synthesis)")
print("  2. Dark matter halo (NFW)")
print("  3. Modified Newtonian Dynamics (MOND) as alternative")
print("  4. Maximum likelihood fitting")
print("  5. Model comparison using AIC/BIC")

print("\n" + "=" * 80)
print("PART 1: STAR FORMATION SYNTHESIS - MASS-TO-LIGHT RATIOS")
print("=" * 80)

baryon_model = BaryonicModels()

print("\nStellar Population Mass-to-Light Ratios:")
print("-" * 80)

colors = [0.5, 0.7, 0.9, 1.1]
for color in colors:
    for population in ['old', 'young']:
        ML = baryon_model.compute_mass_to_light_ratio(color, population)
        print(f"  B-V = {color:.1f}, {population:>5} population: "
              f"M/L_B = {ML['B']:.2f}, M/L_V = {ML['V']:.2f}")

print("\n" + "=" * 80)
print("PART 2: GENERATING SAMPLE GALAXY DATA")
print("=" * 80)

r, v_obs, v_err, true_params = generate_sample_rotation_curve(
    r_min=2, r_max=30, n_points=25, noise=5)

print(f"\nGenerated galaxy with:")
print(f"  {len(r)} radial points from {r.min():.1f} to {r.max():.1f} kpc")
print(f"  Velocity range: {v_obs.min():.1f} to {v_obs.max():.1f} km/s")

print(f"\nTrue parameters:")
print(f"  Baryons: M_disk={true_params['baryons'][0]:.1f}, R_disk={true_params['baryons'][1]:.1f}")
print(f"           M_bulge={true_params['baryons'][2]:.1f}, r_bulge={true_params['baryons'][3]:.1f}")
print(f"           M_HI={true_params['baryons'][4]:.1f}, R_HI={true_params['baryons'][5]:.1f}")
print(f"           M_H2={true_params['baryons'][6]:.1f}, R_H2={true_params['baryons'][7]:.1f}")
print(f"  Dark Matter: rho_s={true_params['dm'][0]:.1f}, r_s={true_params['dm'][1]:.1f}")

print("\n" + "=" * 80)
print("PART 3: FITTING DARK MATTER MODEL (NFW + Baryons)")
print("=" * 80)

fitter = ComprehensiveRotationCurveFitter()
fitter.load_data(r, v_obs, v_err)

print("\nFitting maximum likelihood...")
ml_params_dm = fitter.fit_maximum_likelihood(model='dm')

print("\nDark Matter Model Fit Results:")
print("-" * 80)
print(f"  M_disk  = {fitter.ml_baryon_params[0]:.2f} ± ... (10^10 M☉)")
print(f"  R_disk  = {fitter.ml_baryon_params[1]:.2f} kpc")
print(f"  M_bulge = {fitter.ml_baryon_params[2]:.2f} (10^10 M☉)")
print(f"  r_bulge = {fitter.ml_baryon_params[3]:.2f} kpc")
print(f"  M_HI    = {fitter.ml_baryon_params[4]:.2f} (10^10 M☉)")
print(f"  R_HI    = {fitter.ml_baryon_params[5]:.2f} kpc")
print(f"  M_H2    = {fitter.ml_baryon_params[6]:.2f} (10^10 M☉)")
print(f"  R_H2    = {fitter.ml_baryon_params[7]:.2f} kpc")
print(f"  rho_s   = {fitter.ml_dm_params[0]:.2f} (10^-9 M☉/pc³)")
print(f"  r_s     = {fitter.ml_dm_params[1]:.2f} kpc")

gof_dm = fitter.compute_goodness_of_fit('dm')
print(f"\n  χ² = {gof_dm['chi2']:.2f}, χ²_red = {gof_dm['reduced_chi2']:.2f}")
print(f"  AIC = {gof_dm['aic']:.2f}, BIC = {gof_dm['bic']:.2f}")
print(f"  p-value = {gof_dm['p_value']:.4f}")

print("\n" + "=" * 80)
print("PART 4: FITTING MOND MODEL (No Dark Matter)")
print("=" * 80)

print("\nFitting MOND...")
ml_params_mond = fitter.fit_maximum_likelihood(model='mond')

print("\nMOND Model Fit Results:")
print("-" * 80)
print(f"  M_disk  = {fitter.ml_mond_params[0]:.2f} (10^10 M☉)")
print(f"  R_disk  = {fitter.ml_mond_params[1]:.2f} kpc")
print(f"  M_bulge = {fitter.ml_mond_params[2]:.2f} (10^10 M☉)")
print(f"  r_bulge = {fitter.ml_mond_params[3]:.2f} kpc")
print(f"  M_HI    = {fitter.ml_mond_params[4]:.2f} (10^10 M☉)")
print(f"  R_HI    = {fitter.ml_mond_params[5]:.2f} kpc")
print(f"  M_H2    = {fitter.ml_mond_params[6]:.2f} (10^10 M☉)")
print(f"  R_H2    = {fitter.ml_mond_params[7]:.2f} kpc")
print(f"\n  MOND acceleration scale a0 = {fitter.mond_model.a0:.1e} m/s²")

gof_mond = fitter.compute_goodness_of_fit('mond')
print(f"\n  χ² = {gof_mond['chi2']:.2f}, χ²_red = {gof_mond['reduced_chi2']:.2f}")
print(f"  AIC = {gof_mond['aic']:.2f}, BIC = {gof_mond['bic']:.2f}")
print(f"  p-value = {gof_mond['p_value']:.4f}")

print("\n" + "=" * 80)
print("PART 5: MODEL COMPARISON")
print("=" * 80)

delta_aic = gof_dm['aic'] - gof_mond['aic']
delta_bic = gof_dm['bic'] - gof_mond['bic']

print(f"\n{'Metric':<15} {'Dark Matter':<15} {'MOND':<15} {'Difference':<15}")
print("-" * 60)
print(f"{'χ²_red':<15} {gof_dm['reduced_chi2']:<15.2f} {gof_mond['reduced_chi2']:<15.2f} "
      f"{gof_dm['reduced_chi2']-gof_mond['reduced_chi2']:<15.2f}")
print(f"{'AIC':<15} {gof_dm['aic']:<15.2f} {gof_mond['aic']:<15.2f} {delta_aic:<15.2f}")
print(f"{'BIC':<15} {gof_dm['bic']:<15.2f} {gof_mond['bic']:<15.2f} {delta_bic:<15.2f}")

print(f"\nInterpretation:")
if abs(delta_aic) < 2:
    print("  ΔAIC < 2: Models are essentially equivalent")
elif abs(delta_aic) < 10:
    preferred = 'Dark Matter' if delta_aic < 0 else 'MOND'
    print(f"  ΔAIC = {delta_aic:.1f}: Weak evidence for {preferred}")
else:
    preferred = 'Dark Matter' if delta_aic < 0 else 'MOND'
    print(f"  ΔAIC = {delta_aic:.1f}: Strong evidence for {preferred}")

print(f"\n{'='*80}")
print("PHYSICAL INTERPRETATION")
print(f"{'='*80}")

total_baryons_dm = (fitter.ml_baryon_params[0] + fitter.ml_baryon_params[2] + 
                    fitter.ml_baryon_params[4] + fitter.ml_baryon_params[6])
total_baryons_mond = (fitter.ml_mond_params[0] + fitter.ml_mond_params[2] + 
                      fitter.ml_mond_params[4] + fitter.ml_mond_params[6])

print(f"\nDark Matter scenario:")
print(f"  Total baryonic mass: {total_baryons_dm:.2f} × 10^10 M☉")
print(f"  Dark matter contribution dominates at large radii")
print(f"  Baryonic Tully-Fisher relation may hold")

print(f"\nMOND scenario:")
print(f"  Total baryonic mass: {total_baryons_mond:.2f} × 10^10 M☉")
print(f"  No dark matter needed - gravity modified at low accelerations")
print(f"  a << a0 regime: v⁴ ∝ M_baryon (Baryonic Tully-Fisher)")

print("\n" + "=" * 80)
print("PART 6: GENERATING VISUALIZATIONS")
print("=" * 80)

fitter.plot_decomposition('dm', 'dm_decomposition.png')
fitter.plot_decomposition('mond', 'mond_decomposition.png')
fitter.compare_models('model_comparison.png')

print("\n" + "=" * 80)
print("PART 7: MCMC ANALYSIS FOR DARK MATTER")
print("=" * 80)

print("\nRunning MCMC for dark matter model (this may take a moment)...")
dm_samples, dm_median, dm_std = fitter.run_mcmc(
    model='dm', n_walkers=32, n_steps=3000, n_burn=1000)

print(f"\nMCMC Results (median ± std):")
print(f"  M_disk  = {dm_median[0]:.3f} ± {dm_std[0]:.3f}")
print(f"  R_disk  = {dm_median[1]:.3f} ± {dm_std[1]:.3f}")
print(f"  M_bulge = {dm_median[2]:.3f} ± {dm_std[2]:.3f}")
print(f"  r_bulge = {dm_median[3]:.3f} ± {dm_std[3]:.3f}")
print(f"  rho_s   = {dm_median[8]:.3f} ± {dm_std[8]:.3f}")
print(f"  r_s     = {dm_median[9]:.3f} ± {dm_std[9]:.3f}")

print("\n" + "=" * 80)
print("PART 8: MCMC ANALYSIS FOR MOND")
print("=" * 80)

print("\nRunning MCMC for MOND model...")
mond_samples, mond_median, mond_std = fitter.run_mcmc(
    model='mond', n_walkers=32, n_steps=3000, n_burn=1000)

print(f"\nMCMC Results (median ± std):")
print(f"  M_disk  = {mond_median[0]:.3f} ± {mond_std[0]:.3f}")
print(f"  R_disk  = {mond_median[1]:.3f} ± {mond_std[1]:.3f}")
print(f"  M_bulge = {mond_median[2]:.3f} ± {mond_std[2]:.3f}")
print(f"  r_bulge = {mond_median[3]:.3f} ± {mond_std[3]:.3f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key components implemented:

1. BARYONIC MODELS
   - Stellar disk (exponential profile)
   - Bulge (Plummer/Hernquist profile)
   - HI gas disk (exponential)
   - Molecular gas (H2) disk
   - Population synthesis for mass-to-light ratios

2. DARK MATTER MODEL
   - NFW (Navarro-Frenk-White) profile
   - ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]

3. MOND MODEL (Modified Newtonian Dynamics)
   - Standard interpolating function
   - Acceleration scale a0 = 1.2 × 10⁻¹⁰ m/s²
   - μ(a/a0) = (a/a0) / sqrt(1 + (a/a0)²)

4. STATISTICAL METHODS
   - Maximum likelihood estimation
   - MCMC sampling (emcee)
   - Model comparison (AIC, BIC, χ²)
""")

print("\nGenerated output files:")
print("  dm_decomposition.png - Dark matter rotation curve breakdown")
print("  mond_decomposition.png - MOND rotation curve breakdown")
print("  model_comparison.png - DM vs MOND comparison")

print("\n" + "=" * 80)
