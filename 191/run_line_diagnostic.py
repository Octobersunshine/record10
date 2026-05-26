import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from line_diagnostic import (
    spectral_lines, abundance_models, DEMInverter,
    synthesize_spectrum, add_continuum, calculate_line_intensity,
    generate_demo_spectrum, identify_lines, fit_abundances,
    get_abundance_factor
)


def plot_synthetic_spectrum(wavelengths, spectrum, identified_lines=None, 
                            save_path='synthetic_spectrum.png', title='Synthetic X-ray Spectrum'):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(wavelengths, spectrum, 'b-', linewidth=1, label='Synthetic Spectrum')
    
    if identified_lines:
        for line_info in identified_lines:
            line = line_info['line']
            wave = line['wavelength']
            idx = np.argmin(np.abs(wavelengths - wave))
            if idx < len(spectrum):
                peak_intensity = spectrum[idx]
                ax.annotate(line['ion'], xy=(wave, peak_intensity),
                           xytext=(wave + 0.3, peak_intensity * 1.2),
                           fontsize=8, rotation=90,
                           arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    
    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Intensity (phot cm$^{-2}$ s$^{-1}$)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Spectrum plot saved to: {save_path}")
    
    return fig


def plot_abundance_comparison(abundance_factors, save_path='abundance_comparison.png'):
    elements = list(abundance_factors.keys())
    factors = [abundance_factors[el] for el in elements]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(elements))
    bars = ax.bar(x_pos, factors, color='steelblue', alpha=0.7)
    
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='Photospheric (FIP=1)')
    ax.axhline(y=4.0, color='g', linestyle='--', linewidth=1, label='Coronal (FIP=4)')
    
    ax.set_xlabel('Element', fontsize=12)
    ax.set_ylabel('Abundance Enhancement Factor', fontsize=12)
    ax.set_title('Element Abundance Enhancement (relative to photospheric)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(elements, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    ax.set_yscale('log')
    
    for bar, factor in zip(bars, factors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
               f'{factor:.1f}x', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Abundance comparison saved to: {save_path}")
    
    return fig


def plot_emissivity_curves(selected_lines=None, save_path='emissivity_curves.png'):
    if selected_lines is None:
        selected_lines = [0, 3, 5, 7, 8, 10]
    
    logT_array = np.linspace(5.5, 8.0, 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_lines)))
    
    for idx, (line_idx, color) in enumerate(zip(selected_lines, colors)):
        line = spectral_lines[line_idx]
        emiss = np.array([line_emissivity(logT, line) for logT in logT_array])
        ax.plot(logT_array, emiss / emiss.max(), color=color, 
               label=f"{line['ion']} ({line['wavelength']} Å)", linewidth=2)
        
        ax.axvline(line['logT_max'], color=color, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('log Temperature (K)', fontsize=12)
    ax.set_ylabel('Normalized Emissivity', fontsize=12)
    ax.set_title('Line Emissivity as Function of Temperature', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Emissivity curves saved to: {save_path}")
    
    return fig


def line_emissivity(logT, line):
    from line_diagnostic import line_emissivity as le
    return le(logT, line)


def run_demo():
    print("=" * 70)
    print("SOLAR FLARE SPECTRAL LINE DIAGNOSTIC DEMO")
    print("=" * 70)
    
    print("\n1. Generating synthetic X-ray spectrum...")
    demo_data = generate_demo_spectrum()
    wavelengths = demo_data['wavelengths']
    spectrum = demo_data['spectrum']
    true_logT = demo_data['logT_centers']
    true_DEM = demo_data['DEM']
    
    noise = np.random.normal(0, 0.05 * spectrum.max(), size=len(spectrum))
    spectrum_obs = spectrum + noise
    spectrum_obs = np.maximum(spectrum_obs, 0)
    
    plot_synthetic_spectrum(wavelengths, spectrum_obs, None,
                           save_path='synthetic_spectrum.png',
                           title='Synthetic X-ray Spectrum (With Noise)')
    
    print("\n2. Identifying spectral lines...")
    identified_lines = identify_lines(wavelengths, spectrum_obs, threshold=0.1)
    print(f"   Identified {len(identified_lines)} spectral lines:")
    for line_info in identified_lines[:8]:
        print(f"     - {line_info['line']['ion']:10s}: {line_info['line']['wavelength']:.2f} Å, "
              f"I = {line_info['intensity']:.2e}")
    
    plot_synthetic_spectrum(wavelengths, spectrum_obs, identified_lines,
                           save_path='spectrum_with_lines.png',
                           title='Identified Spectral Lines')
    
    print("\n3. Constructing response matrix for DEM inversion...")
    selected_line_indices = [0, 3, 5, 7, 8, 10, 13, 15]
    selected_lines = [spectral_lines[i] for i in selected_line_indices]
    
    inverter = DEMInverter(logT_min=6.0, logT_max=7.8, n_bins=15)
    RM = inverter.construct_response_matrix(selected_lines, abundance_model='coronal')
    print(f"   Response matrix shape: {RM.shape}")
    
    print("\n4. Extracting line intensities...")
    line_intensities = []
    for line in selected_lines:
        idx = np.argmin(np.abs(wavelengths - line['wavelength']))
        line_intensities.append(max(spectrum_obs[idx], 1e-5))
    line_errors = np.array(line_intensities) * 0.1 + 1e-6
    
    print("   Selected lines and intensities:")
    for line, intensity in zip(selected_lines, line_intensities):
        print(f"     {line['ion']:10s}: {intensity:.2e}")
    
    print("\n5. Running regularized DEM inversion...")
    DEM_reg, chi2 = inverter.invert_regularized(
        line_intensities, line_errors, lambda_reg=0.01, method='positivity'
    )
    print(f"   χ² = {chi2:.2f}")
    print(f"   Total EM = {inverter.calculate_total_em():.2e} cm⁻³")
    
    inverter.plot_DEM(save_path='dem_regularized.png')
    
    print("\n6. Running MCMC DEM inversion (this may take a minute)...")
    try:
        DEM_med, DEM_low, DEM_high = inverter.invert_mcmc(
            line_intensities, line_errors,
            n_walkers=50, n_steps=1500, burnin=500
        )
        inverter.plot_DEM(save_path='dem_mcmc.png')
    except Exception as e:
        print(f"   MCMC skipped (error: {e})")
    
    print("\n7. Comparing DEM results with true input...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(true_logT, true_DEM, 'go-', linewidth=2, markersize=8, label='True DEM')
    ax.plot(inverter.logT_centers, DEM_reg, 'rs-', linewidth=2, markersize=5, label='Regularized Inversion')
    if hasattr(inverter, 'DEM_median'):
        ax.fill_between(inverter.logT_centers, DEM_low, DEM_high,
                       alpha=0.3, color='blue', label='MCMC 68% CI')
        ax.plot(inverter.logT_centers, inverter.DEM_median, 'b-', linewidth=2, label='MCMC Median')
    ax.set_xlabel('log Temperature (K)', fontsize=12)
    ax.set_ylabel('DEM (cm⁻⁵ K⁻¹)', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('DEM Comparison: True vs Inferred', fontsize=14)
    plt.tight_layout()
    plt.savefig('dem_comparison.png', dpi=150)
    plt.close()
    print("   DEM comparison saved to: dem_comparison.png")
    
    print("\n8. Testing abundance variations...")
    print("   Testing different abundance models:")
    for model_name in ['photospheric', 'coronal', 'flare', 'flaring']:
        spec = synthesize_spectrum(wavelengths, true_logT, true_DEM, abundance_model=model_name)
        print(f"     {model_name:15s}: total flux = {np.sum(spec):.2e}")
    
    abundance_enhancements = {
        'Fe': 4.0, 'Ca': 4.0, 'O': 1.0, 'Mg': 4.0,
        'Si': 4.0, 'S': 4.0, 'Ar': 4.0
    }
    plot_abundance_comparison(abundance_enhancements, save_path='abundance_plot.png')
    
    print("\n9. Plotting emissivity curves...")
    plot_emissivity_curves(save_path='emissivity_curves.png')
    
    print("\n10. Fitting element abundances from line intensities...")
    fitted_abundances = fit_abundances(
        line_intensities, selected_lines, DEM_reg,
        inverter.logT_centers, reference_model='photospheric'
    )
    print("    Fitted abundance enhancements:")
    for element, factor in fitted_abundances.items():
        print(f"      {element:5s}: {factor:.2f}x photospheric")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE - Results saved as PNG files")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - synthetic_spectrum.png")
    print("  - spectrum_with_lines.png")
    print("  - dem_regularized.png")
    print("  - dem_mcmc.png")
    print("  - dem_comparison.png")
    print("  - abundance_plot.png")
    print("  - emissivity_curves.png")


def line_emissivity(logT, line):
    logT0 = line['logT_max']
    sigma_T = 0.25
    emiss = np.exp(-(logT - logT0)**2 / (2 * sigma_T**2))
    return emiss


if __name__ == '__main__':
    run_demo()
