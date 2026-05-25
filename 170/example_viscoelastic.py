from afm_viscoelastic import ViscoelasticAFM, load_creep_data
import numpy as np


def example_1_sls_creep_fitting():
    print("=" * 60)
    print("Example 1: SLS Model Fitting on Synthetic Creep Data")
    print("=" * 60)
    
    ve_afm = ViscoelasticAFM(radius=30e-9, nu=0.5)
    
    E0_true = 1500
    Einf_true = 600
    tau_true = 1.5
    delta0_true = 80e-9
    
    print(f"\nTrue parameters (cell-like material):")
    print(f"  Instantaneous modulus E₀: {E0_true} Pa")
    print(f"  Equilibrium modulus E∞: {Einf_true} Pa")
    print(f"  Relaxation time τ: {tau_true} s")
    print(f"  Initial indentation δ₀: {delta0_true*1e9:.1f} nm")
    
    t, delta_noisy, delta_true = ve_afm.generate_creep_data(
        t_total=30.0, n_points=500,
        E0=E0_true, Einf=Einf_true, tau=tau_true, delta0=delta0_true,
        noise=0.015
    )
    
    print("\nFitting SLS model...")
    popt, pcov = ve_afm.fit_sls_creep(t, delta_noisy, p0=[1000, 400, 1.0, 60e-9])
    
    ve_afm.print_viscoelastic_results()
    
    if popt is not None:
        E0_fit, Einf_fit, tau_fit, delta0_fit = popt
        print(f"Fitting errors:")
        print(f"  ΔE₀: {abs(E0_fit - E0_true)/E0_true*100:.2f}%")
        print(f"  ΔE∞: {abs(Einf_fit - Einf_true)/Einf_true*100:.2f}%")
        print(f"  Δτ: {abs(tau_fit - tau_true)/tau_true*100:.2f}%")
    
    ve_afm.plot_creep_curve(t, delta_noisy, delta_true, save_path='example1_creep.png')
    print("\nPlot saved as 'example1_creep.png'")
    print()


def example_2_hydrogel_viscoelasticity():
    print("=" * 60)
    print("Example 2: Hydrogel-like Material with Multiple Relaxation Modes")
    print("=" * 60)
    
    ve_afm = ViscoelasticAFM(radius=25e-9, nu=0.45)
    
    print("\nGenerating data for hydrogel with 3 relaxation modes...")
    t_total = 60.0
    t = np.linspace(0, t_total, 600)
    
    def multi_mode_creep(t, delta0, Ge, tau_list, G_list):
        G_t = Ge
        for tau, Gi in zip(tau_list, G_list):
            G_t += Gi * np.exp(-t / tau)
        
        G0 = Ge + sum(G_list)
        delta_t = delta0 * (G0 / G_t) ** (2/3)
        return delta_t
    
    Ge_true = 300
    tau_list_true = [0.1, 2.0, 15.0]
    G_list_true = [400, 300, 200]
    delta0_true = 100e-9
    G0_true = Ge_true + sum(G_list_true)
    
    delta_true = multi_mode_creep(t, delta0_true, Ge_true, tau_list_true, G_list_true)
    noise_amp = 0.01 * np.abs(delta_true).mean()
    delta_noisy = delta_true + noise_amp * np.random.randn(len(delta_true))
    
    print(f"True parameters:")
    print(f"  Equilibrium modulus Ge: {Ge_true} Pa")
    print(f"  Instantaneous modulus G0: {G0_true} Pa")
    print(f"  Relaxation times: {tau_list_true} s")
    print(f"  Relaxation moduli: {G_list_true} Pa")
    
    print("\nFitting with SLS model (single mode)...")
    ve_afm.fit_sls_creep(t, delta_noisy, p0=[1000, 300, 2.0, 80e-9])
    ve_afm.print_viscoelastic_results()
    
    print("\nExtracting relaxation spectrum (3 modes)...")
    G_t_synth = G0_true * (delta0_true / delta_true) ** 1.5
    tau_fit, Gi_fit, Ge_fit = ve_afm.calculate_stress_relaxation_spectrum(
        t, G_t_synth, n_modes=3, tau_min=1e-2, tau_max=1e2
    )
    
    if tau_fit is not None:
        print(f"Extracted spectrum:")
        for i, (tau, Gi) in enumerate(zip(tau_fit, Gi_fit)):
            print(f"  Mode {i+1}: τ = {tau:.3f} s, G_i = {Gi:.1f} Pa")
        print(f"  Equilibrium modulus Ge: {Ge_fit:.1f} Pa")
    
    ve_afm.plot_relaxation_spectrum(save_path='example2_spectrum.png')
    print("\nRelaxation spectrum plot saved as 'example2_spectrum.png'")
    print()


def example_3_cell_mechanics_comparison():
    print("=" * 60)
    print("Example 3: Comparing Different Cell Types")
    print("=" * 60)
    
    cell_types = {
        'Soft cell (e.g., neuron)': {'E0': 500, 'Einf': 200, 'tau': 3.0},
        'Medium cell (e.g., fibroblast)': {'E0': 2000, 'Einf': 800, 'tau': 1.0},
        'Stiff cell (e.g., bone cell)': {'E0': 8000, 'Einf': 4000, 'tau': 0.3}
    }
    
    ve_afm = ViscoelasticAFM(radius=20e-9, nu=0.5)
    results = {}
    
    for cell_type, params in cell_types.items():
        print(f"\n{cell_type}:")
        t, delta_noisy, _ = ve_afm.generate_creep_data(
            t_total=15.0, n_points=300,
            E0=params['E0'], Einf=params['Einf'], tau=params['tau'],
            delta0=50e-9, noise=0.01
        )
        
        ve_afm.fit_sls_creep(t, delta_noisy, p0=[params['E0']*0.7, params['Einf']*0.7, 1.0, 40e-9])
        
        if ve_afm.fit_params is not None:
            E0, Einf, tau, _ = ve_afm.fit_params
            results[cell_type] = {
                'E0': E0, 'Einf': Einf, 'tau': tau,
                'ratio': Einf/E0
            }
            print(f"  E₀ = {E0:.0f} Pa, E∞ = {Einf:.0f} Pa")
            print(f"  τ = {tau:.2f} s, E∞/E₀ = {Einf/E0:.2%}")
    
    print("\nSummary:")
    print("-" * 60)
    print(f"{'Cell Type':<30} {'E₀ (Pa)':>10} {'E∞/E₀':>8} {'τ (s)':>8}")
    print("-" * 60)
    for cell_type, res in results.items():
        print(f"{cell_type:<30} {res['E0']:>10.0f} {res['ratio']:>8.1%} {res['tau']:>8.2f}")
    print()


def example_4_real_data_format():
    print("=" * 60)
    print("Example 4: Loading and Analyzing Real Experimental Data")
    print("=" * 60)
    
    sample_data = np.array([
        [0.0, 50.0e-9],
        [0.1, 52.3e-9],
        [0.5, 58.7e-9],
        [1.0, 63.2e-9],
        [2.0, 68.5e-9],
        [5.0, 74.1e-9],
        [10.0, 76.8e-9],
        [20.0, 78.2e-9],
        [30.0, 78.8e-9],
    ])
    
    print("\nSample experimental data format (CSV):")
    print("time_s,indentation_m")
    for row in sample_data:
        print(f"{row[0]:.1f},{row[1]:.6e}")
    
    np.savetxt('sample_creep_data.csv', sample_data, delimiter=',', 
               header='time_s,indentation_m', comments='', fmt='%.6e')
    print("\nSample data saved to 'sample_creep_data.csv'")
    
    print("\nLoading and fitting...")
    t, delta = load_creep_data('sample_creep_data.csv')
    
    if t is not None:
        ve_afm = ViscoelasticAFM(radius=20e-9, nu=0.5)
        ve_afm.fit_sls_creep(t, delta, p0=[1000, 500, 1.0, 40e-9])
        ve_afm.print_viscoelastic_results()
        
        ve_afm.plot_creep_curve(t, delta, save_path='example4_real_data.png')
        print("Plot saved as 'example4_real_data.png'")
    print()


if __name__ == "__main__":
    example_1_sls_creep_fitting()
    example_2_hydrogel_viscoelasticity()
    example_3_cell_mechanics_comparison()
    example_4_real_data_format()
    
    print("=" * 60)
    print("All viscoelastic examples completed!")
    print("=" * 60)
