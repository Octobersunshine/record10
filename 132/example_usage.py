import numpy as np
from zernike_analysis import ZernikeAnalyzer, generate_test_wavefront


def example_1_psf_strehl_basic():
    print("=" * 70)
    print("EXAMPLE 1: Basic PSF and Strehl Ratio Analysis")
    print("=" * 70)
    
    analyzer = ZernikeAnalyzer(max_order=6)
    
    coeffs = np.zeros(28)
    coeffs[3] = 0.1  
    coeffs[6] = 0.05  
    coeffs[10] = 0.03  
    
    analyzer.print_psf_analysis(coeffs, wavelength=1.0, grid_size=256)
    print()


def example_2_aberration_effect_comparison():
    print("=" * 70)
    print("EXAMPLE 2: Aberration Type Effect Comparison")
    print("=" * 70)
    
    analyzer = ZernikeAnalyzer(max_order=6)
    wavelength = 1.0
    
    aberration_tests = [
        ("Ideal (No Aberration)", lambda c: c),
        ("Defocus (0.1λ)", lambda c: (c[3] == 0.1)),
        ("Astigmatism (0.1λ)", lambda c: (c[4] == 0.1)),
        ("Coma (0.1λ)", lambda c: (c[6] == 0.1)),
        ("Spherical (0.1λ)", lambda c: (c[10] == 0.1)),
    ]
    
    n_terms = len(analyzer.noll_indices)
    coeffs_list = []
    labels = []
    
    for label, _ in aberration_tests:
        coeffs = np.zeros(n_terms)
        if "Defocus" in label:
            coeffs[3] = 0.1
        elif "Astigmatism" in label:
            coeffs[4] = 0.1
        elif "Coma" in label:
            coeffs[6] = 0.1
        elif "Spherical" in label:
            coeffs[10] = 0.1
        coeffs_list.append(coeffs)
        labels.append(label)
    
    print(f"\n{'Aberration Type':<25} {'Strehl Ratio':>15} {'FWHM (λ/D)':>15} {'EE50 (λ/D)':>15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    
    for coeffs, label in zip(coeffs_list, labels):
        strehl = analyzer.compute_strehl_ratio(coeffs, wavelength, method='exact')
        fx, fy, psf = analyzer.compute_psf(coeffs, 128, wavelength)
        psf_info = analyzer.analyze_psf(fx, fy, psf)
        print(f"{label:<25} {strehl:>15.4f} {psf_info['fwhm_avg']:>15.4f} {psf_info['ee50_radius']:>15.4f}")
    
    print(f"\nNote: All aberrations have same RMS amplitude (0.1λ)")
    print(f"      but different impact on image quality!")
    print()


def example_3_strehl_vs_wfe():
    print("=" * 70)
    print("EXAMPLE 3: Strehl Ratio vs Wavefront Error")
    print("=" * 70)
    
    analyzer = ZernikeAnalyzer(max_order=6)
    wavelength = 1.0
    
    wfe_values = [0.0, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    print(f"\n{'RMS WFE (λ)':<15} {'Strehl (Exact)':>15} {'Strehl (Exp)':>15} {'Strehl (Marechal)':>18}")
    print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*18}")
    
    for wfe in wfe_values:
        coeffs = np.zeros(len(analyzer.noll_indices))
        coeffs[3] = wfe
        
        strehl_exact = analyzer.compute_strehl_ratio(coeffs, wavelength, method='exact')
        strehl_approx = analyzer.compute_strehl_ratio(coeffs, wavelength, method='approximate')
        strehl_marechal = analyzer.compute_strehl_ratio(coeffs, wavelength, method='marechal')
        
        print(f"{wfe:<15.3f} {strehl_exact:>15.4f} {strehl_approx:>15.4f} {strehl_marechal:>18.4f}")
    
    print(f"\nKey thresholds:")
    print(f"  Marechal criterion: RMS WFE < λ/14 ≈ {1/14:.4f}λ → Strehl > 0.8")
    print(f"  λ/4 rule:           RMS WFE < 0.25λ → Moderate quality")
    print()


def example_4_adaptive_optics_correction():
    print("=" * 70)
    print("EXAMPLE 4: Adaptive Optics Correction Simulation")
    print("=" * 70)
    
    analyzer = ZernikeAnalyzer(max_order=8)
    wavelength = 0.5e-6  
    
    n_terms = len(analyzer.noll_indices)
    np.random.seed(42)
    coeffs_aberrated = np.random.randn(n_terms) * 0.15
    coeffs_aberrated[0] = 0  
    coeffs_aberrated[1:3] *= 0.5  
    
    print(f"\n--- BEFORE CORRECTION (Atmospheric Turbulence) ---")
    analyzer.print_psf_analysis(coeffs_aberrated, wavelength=1.0, grid_size=128)
    
    correction_orders = [2, 4, 6, 8]
    
    print(f"\n--- ADAPTIVE OPTICS CORRECTION PROGRESS ---")
    print(f"{'Correction Order':<20} {'Strehl Ratio':>15} {'Improvement':>15}")
    print(f"{'-'*20} {'-'*15} {'-'*15}")
    
    strehl_before = analyzer.compute_strehl_ratio(coeffs_aberrated, 1.0, method='exact')
    
    for order in correction_orders:
        coeffs_corrected = coeffs_aberrated.copy()
        n_correct = sum([(n + 1) for n in range(order + 1)])
        coeffs_corrected[:n_correct] = 0
        
        strehl_after = analyzer.compute_strehl_ratio(coeffs_corrected, 1.0, method='exact')
        improvement = strehl_after / strehl_before if strehl_before > 0 else 0
        
        print(f"Zernike order {order:<10} {strehl_after:>15.4f} {improvement:>15.2f}x")
    
    print(f"\nInterpretation:")
    print(f"  - Order 2: Tip/tilt correction (basic tracking)")
    print(f"  - Order 4: Defocus + astigmatism correction")
    print(f"  - Order 6: Coma + trefoil + spherical correction")
    print(f"  - Order 8: High-order correction")
    print()


def example_5_mtf_calculation():
    print("=" * 70)
    print("EXAMPLE 5: Modulation Transfer Function (MTF)")
    print("=" * 70)
    
    analyzer = ZernikeAnalyzer(max_order=6)
    wavelength = 1.0
    
    coeffs_ideal = np.zeros(28)
    coeffs_aberrated = np.zeros(28)
    coeffs_aberrated[3] = 0.1  
    coeffs_aberrated[10] = 0.05  
    
    fx_ideal, fy_ideal, mtf_ideal = analyzer.compute_mtf(coeffs_ideal, 128, wavelength)
    fx_ab, fy_ab, mtf_ab = analyzer.compute_mtf(coeffs_aberrated, 128, wavelength)
    
    center = mtf_ideal.shape[0] // 2
    mtf_ideal_slice = mtf_ideal[center, center:]
    mtf_ab_slice = mtf_ab[center, center:]
    freq_axis = fx_ideal[center, center:]
    
    cutoff_idx = np.argmin(np.abs(mtf_ideal_slice - 0.01))
    cutoff_freq = freq_axis[cutoff_idx] if cutoff_idx > 0 else freq_axis[-1]
    
    print(f"\nMTF Analysis:")
    print(f"  Diffraction-limited MTF cutoff: ~{cutoff_freq:.2f} cycles/λ/D")
    print(f"\nMTF at selected spatial frequencies:")
    print(f"  {'Frequency':<15} {'MTF (Ideal)':>15} {'MTF (Aberrated)':>18} {'Ratio':>12}")
    print(f"  {'-'*15} {'-'*15} {'-'*18} {'-'*12}")
    
    for target_freq in [0.1, 0.2, 0.3, 0.4, 0.5]:
        idx = np.argmin(np.abs(freq_axis - target_freq))
        if idx < len(mtf_ideal_slice):
            mtf_i = mtf_ideal_slice[idx]
            mtf_a = mtf_ab_slice[idx]
            ratio = mtf_a / mtf_i if mtf_i > 0 else 0
            print(f"  {freq_axis[idx]:<15.3f} {mtf_i:>15.3f} {mtf_a:>18.3f} {ratio:>12.1%}")
    
    print(f"\nConclusion: Aberrations reduce MTF contrast, especially at mid-high frequencies")
    print()


def example_6_optical_system_spec():
    print("=" * 70)
    print("EXAMPLE 6: Optical System Design Specification")
    print("=" * 70)
    
    analyzer = ZernikeAnalyzer(max_order=6)
    
    system_specs = {
        'Diameter': 0.1,  
        'Focal_length': 0.5,  
        'Wavelength': 0.55e-6,  
        'Field_of_view': 1.0,  
    }
    
    print(f"\nSystem Specifications:")
    for key, val in system_specs.items():
        print(f"  {key:<25}: {val}")
    
    f_number = system_specs['Focal_length'] / system_specs['Diameter']
    airy_disk_radius = 1.22 * system_specs['Wavelength'] * f_number / system_specs['Diameter'] * 1e6
    print(f"\nDerived Parameters:")
    print(f"  F-number:                {f_number:.1f}")
    print(f"  Airy disk radius:        {airy_disk_radius:.2f} μm")
    
    max_allowed_wfe = system_specs['Wavelength'] / 14
    print(f"\nMarechal Criterion (Diffraction-limited):")
    print(f"  Max allowed RMS WFE:     {max_allowed_wfe * 1e9:.1f} nm")
    print(f"  Corresponding Strehl:    0.8")
    
    coeffs_diffraction = np.zeros(28)
    coeffs_diffraction[1:28] = max_allowed_wfe / system_specs['Wavelength'] * 0.1
    
    strehl_diffraction = analyzer.compute_strehl_ratio(coeffs_diffraction, 
                                                       wavelength=1.0, 
                                                       method='exact')
    
    print(f"\nVerification with small aberrations:")
    print(f"  Strehl ratio achieved:   {strehl_diffraction:.4f}")
    print(f"  Meets diffraction limit: {'YES' if strehl_diffraction >= 0.8 else 'NO'}")
    
    print()


def example_7_combined_analysis():
    print("=" * 70)
    print("EXAMPLE 7: Complete Wavefront → PSF Analysis Workflow")
    print("=" * 70)
    
    analyzer = ZernikeAnalyzer(max_order=6, fit_method='svd')
    
    print(f"\nStep 1: Generate test wavefront with known aberrations")
    X, Y, W = generate_test_wavefront('mixed', grid_size=50, noise=0.02)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = W.flatten()
    mask = (x_flat ** 2 + y_flat ** 2) <= 1.0
    
    print(f"Step 2: Fit Zernike coefficients using SVD")
    coeffs, metrics = analyzer.fit_wavefront(x_flat, y_flat, z_flat, mask=mask)
    analyzer.print_analysis(coeffs, metrics)
    
    print(f"\nStep 3: Compute PSF and Strehl ratio")
    analyzer.print_psf_analysis(coeffs, wavelength=1.0, grid_size=128)
    
    print(f"\nStep 4: System assessment")
    strehl = analyzer.compute_strehl_ratio(coeffs, method='exact')
    if strehl >= 0.8:
        print("  → System is DIFFRACTION LIMITED")
    elif strehl >= 0.5:
        print("  → System has GOOD IMAGE QUALITY")
    else:
        print("  → System NEEDS CORRECTION")
    print()


if __name__ == "__main__":
    example_1_psf_strehl_basic()
    example_2_aberration_effect_comparison()
    example_3_strehl_vs_wfe()
    example_4_adaptive_optics_correction()
    example_5_mtf_calculation()
    example_6_optical_system_spec()
    example_7_combined_analysis()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
