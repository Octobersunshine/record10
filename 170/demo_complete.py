print("=" * 70)
print("AFM Analysis Toolkit - Complete Demo")
print("=" * 70)
print("\nThis toolkit includes:")
print("  1. Elastic contact models (DMT, JKR, MD) - for hard materials")
print("  2. Viscoelastic models (SLS, relaxation spectrum) - for biomaterials")
print()

print("\n" + "=" * 70)
print("PART 1: Elastic Contact Models (Hard Materials)")
print("=" * 70)

try:
    from afm_fitting import AFMCurveFitter
    
    fitter = AFMCurveFitter(radius=10e-9, nu=0.5, z0=0.3e-9)
    
    E_true = 5e9
    F_ad_true = 5e-9
    
    print(f"\nGenerating test data:")
    print(f"  True Young's modulus E: {E_true:.2e} Pa")
    print(f"  True adhesion force F_ad: {F_ad_true:.2e} N")
    print(f"  Tip radius: {fitter.R * 1e9:.1f} nm")
    
    delta, F_noisy = fitter.generate_synthetic_data(
        model='MD', E_true=E_true, F_ad_true=F_ad_true, noise=0.03, lambda_param=1.0
    )
    
    print("\nFitting with MD model (Maugis-Dugdale transition model):")
    fitter.fit(delta, F_noisy, model='MD', p0=[1e9, 1e-9])
    fitter.print_results()
    
    print("Model comparison (R² values):")
    models = ['DMT', 'JKR', 'MD']
    for model in models:
        fitter_test = AFMCurveFitter(radius=10e-9, nu=0.5)
        fitter_test.fit(delta, F_noisy, model=model)
        r2 = fitter_test.calculate_goodness_of_fit(delta, F_noisy)
        print(f"  {model}: {r2:.6f}")
    
    print("\n✓ Elastic models demo completed!")
    
except Exception as e:
    print(f"\n✗ Elastic models demo skipped: {e}")

print("\n" + "=" * 70)
print("PART 2: Viscoelastic Models (Biomaterials)")
print("=" * 70)

try:
    from afm_viscoelastic import ViscoelasticAFM
    
    ve_afm = ViscoelasticAFM(radius=30e-9, nu=0.5)
    
    print("\nCell-like material parameters:")
    E0_true = 1500
    Einf_true = 600
    tau_true = 1.5
    delta0_true = 80e-9
    
    print(f"  Instantaneous modulus E₀: {E0_true} Pa")
    print(f"  Equilibrium modulus E∞: {Einf_true} Pa")
    print(f"  Relaxation time τ: {tau_true} s")
    print(f"  Elastic ratio E∞/E₀: {Einf_true/E0_true:.1%}")
    
    t, delta_noisy, delta_true = ve_afm.generate_creep_data(
        t_total=30.0, n_points=400,
        E0=E0_true, Einf=Einf_true, tau=tau_true, delta0=delta0_true,
        noise=0.02
    )
    
    print("\nFitting SLS (Standard Linear Solid) model:")
    ve_afm.fit_sls_creep(t, delta_noisy, p0=[1000, 400, 1.0, 60e-9])
    ve_afm.print_viscoelastic_results()
    
    print("\nExtracting relaxation time spectrum (3 modes):")
    G_t_synth = (E0_true + Einf_true) * (delta0_true / delta_true) ** 1.5
    tau_spec, G_spec, Ge = ve_afm.calculate_stress_relaxation_spectrum(
        t, G_t_synth, n_modes=3
    )
    
    if tau_spec is not None:
        print("  Discrete relaxation modes:")
        for i, (tau, G) in enumerate(zip(tau_spec, G_spec)):
            print(f"    Mode {i+1}: τ = {tau:.3f} s, G_i = {G:.1f} Pa")
    
    print("\n✓ Viscoelastic models demo completed!")
    
except Exception as e:
    print(f"\n✗ Viscoelastic models demo skipped: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nAvailable files:")
print("  Core modules:")
print("    - afm_fitting.py        : Elastic contact models (DMT, JKR, MD)")
print("    - afm_viscoelastic.py   : Viscoelastic models (SLS, relaxation spectrum)")
print()
print("  Example scripts:")
print("    - example_usage.py      : Elastic model examples")
print("    - example_viscoelastic.py: Viscoelastic examples")
print("    - fit_real_data.py      : Real data fitting utility")
print()
print("  Documentation:")
print("    - MD_MODEL_README.md    : Maugis-Dugdale model guide")
print("    - VISCOELASTIC_README.md: Viscoelastic analysis guide")
print()
print("  Sample data:")
print("    - sample_afm_data.csv   : Sample force-distance data")
print()
print("Quick start commands:")
print("  python afm_fitting.py          # Run elastic model demo")
print("  python afm_viscoelastic.py     # Run viscoelastic demo")
print("  python example_usage.py        # Elastic model examples")
print("  python example_viscoelastic.py # Viscoelastic examples")
print("  python fit_real_data.py        # Fit real CSV data")
print()
print("=" * 70)
print("Demo completed successfully!")
print("=" * 70)
