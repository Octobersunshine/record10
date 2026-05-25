from afm_fitting import AFMCurveFitter, load_afm_data
import numpy as np


def example_1_synthetic_md():
    print("=" * 60)
    print("Example 1: Fitting synthetic Maugis-Dugdale data")
    print("=" * 60)
    
    fitter = AFMCurveFitter(radius=5e-9, nu=0.3, z0=0.3e-9)
    
    E_true = 2.5e9
    F_ad_true = 8e-9
    
    delta, F_noisy = fitter.generate_synthetic_data(
        model='MD', E_true=E_true, F_ad_true=F_ad_true, noise=0.04, 
        n_points=300, lambda_param=1.0
    )
    
    popt, pcov = fitter.fit(delta, F_noisy, model='MD', p0=[1e9, 1e-9])
    
    fitter.print_results()
    
    r2 = fitter.calculate_goodness_of_fit(delta, F_noisy)
    print(f"\nR-squared: {r2:.4f}")
    
    fitter.plot_results(delta, F_noisy, save_path='example_md_fit.png')
    print("Plot saved as 'example_md_fit.png'")
    print()


def example_2_small_radius_stability():
    print("=" * 60)
    print("Example 2: MD model stability with small contact radius")
    print("=" * 60)
    
    radii = [2e-9, 5e-9, 10e-9]
    models = ['DMT', 'JKR', 'MD']
    
    for R in radii:
        print(f"\n--- Tip radius: {R*1e9:.1f} nm ---")
        fitter = AFMCurveFitter(radius=R, nu=0.5)
        
        E_true = 5e9
        F_ad_true = 5e-9
        
        delta, F_noisy = fitter.generate_synthetic_data(
            model='MD', E_true=E_true, F_ad_true=F_ad_true, noise=0.02
        )
        
        r2_values = {}
        for model in models:
            try:
                fitter.fit(delta, F_noisy, model=model)
                r2 = fitter.calculate_goodness_of_fit(delta, F_noisy)
                r2_values[model] = r2
                print(f"  {model}: R² = {r2:.6f}")
            except Exception as e:
                print(f"  {model}: Failed - {e}")
        
        best_model = max(r2_values, key=r2_values.get)
        print(f"  Best model: {best_model}")
    print()


def example_3_real_data_simulation():
    print("=" * 60)
    print("Example 3: Simulating real AFM data loading with MD")
    print("=" * 60)
    
    sample_data = np.array([
        [0.0e-9, -2.0e-9],
        [0.2e-9, -1.5e-9],
        [0.4e-9, -0.8e-9],
        [0.6e-9, 0.2e-9],
        [0.8e-9, 1.5e-9],
        [1.0e-9, 3.0e-9],
        [1.5e-9, 7.5e-9],
        [2.0e-9, 13.5e-9],
        [2.5e-9, 21.0e-9],
        [3.0e-9, 30.0e-9],
        [4.0e-9, 52.0e-9],
        [5.0e-9, 79.0e-9],
    ])
    
    delta = sample_data[:, 0]
    force = sample_data[:, 1]
    
    fitter = AFMCurveFitter(radius=5e-9, nu=0.5, z0=0.3e-9)
    fitter.fit(delta, force, model='MD', p0=[1e9, 5e-9])
    
    print("Sample data (indentation, force):")
    for d, f in zip(delta, force):
        print(f"  {d*1e9:.1f} nm, {f*1e9:.2f} nN")
    print()
    
    fitter.print_results()
    print()


def example_4_compare_all_models():
    print("=" * 60)
    print("Example 4: Comparing DMT, JKR, and MD models")
    print("=" * 60)
    
    fitter_md = AFMCurveFitter(radius=8e-9, nu=0.5, z0=0.3e-9)
    fitter_dmt = AFMCurveFitter(radius=8e-9, nu=0.5)
    fitter_jkr = AFMCurveFitter(radius=8e-9, nu=0.5)
    
    E_true = 4e9
    F_ad_true = 6e-9
    
    delta, F_noisy = fitter_md.generate_synthetic_data(
        model='MD', E_true=E_true, F_ad_true=F_ad_true, noise=0.03, lambda_param=1.0
    )
    
    fitter_md.fit(delta, F_noisy, model='MD')
    fitter_dmt.fit(delta, F_noisy, model='DMT')
    fitter_jkr.fit(delta, F_noisy, model='JKR')
    
    print(f"True values: E = {E_true:.2e} Pa, F_ad = {F_ad_true:.2e} N")
    print()
    
    print("MD Model Results:")
    print(f"  E: {fitter_md.E:.2e} Pa (error: {abs(fitter_md.E-E_true)/E_true*100:.1f}%)")
    print(f"  F_ad: {fitter_md.F_ad:.2e} N (error: {abs(fitter_md.F_ad-F_ad_true)/F_ad_true*100:.1f}%)")
    print(f"  R²: {fitter_md.calculate_goodness_of_fit(delta, F_noisy):.6f}")
    print()
    
    print("DMT Model Results:")
    print(f"  E: {fitter_dmt.E:.2e} Pa (error: {abs(fitter_dmt.E-E_true)/E_true*100:.1f}%)")
    print(f"  F_ad: {fitter_dmt.F_ad:.2e} N (error: {abs(fitter_dmt.F_ad-F_ad_true)/F_ad_true*100:.1f}%)")
    print(f"  R²: {fitter_dmt.calculate_goodness_of_fit(delta, F_noisy):.6f}")
    print()
    
    print("JKR Model Results:")
    print(f"  E: {fitter_jkr.E:.2e} Pa (error: {abs(fitter_jkr.E-E_true)/E_true*100:.1f}%)")
    print(f"  F_ad: {fitter_jkr.F_ad:.2e} N (error: {abs(fitter_jkr.F_ad-F_ad_true)/F_ad_true*100:.1f}%)")
    print(f"  R²: {fitter_jkr.calculate_goodness_of_fit(delta, F_noisy):.6f}")
    print()


if __name__ == "__main__":
    example_1_synthetic_md()
    example_2_small_radius_stability()
    example_3_real_data_simulation()
    example_4_compare_all_models()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
