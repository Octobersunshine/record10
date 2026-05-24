import numpy as np
import sys
sys.path.insert(0, '.')
from waveguide import RectangularWaveguide, Polarization, ExcitationType


def demo_square_waveguide():
    print("=" * 70)
    print("Square Waveguide Degenerate Mode Demonstration")
    print("=" * 70)
    
    a = 20e-3
    b = 20e-3
    frequency = 15e9
    
    wg = RectangularWaveguide(a, b)
    
    print(f"\nSquare Waveguide: {a*1000:.1f} mm x {b*1000:.1f} mm")
    print(f"Operating Frequency: {frequency/1e9:.2f} GHz")
    
    print("\n--- Finding All Degenerate Mode Groups ---")
    groups = wg.find_all_degenerate_groups(max_m=3, max_n=3)
    
    for i, group in enumerate(groups):
        if len(group) > 1:
            print(f"\nDegenerate Group {i+1} (f_c = {group[0][3]/1e9:.4f} GHz):")
            for mode_info in group:
                print(f"  - {mode_info[0]}_{mode_info[1]}{mode_info[2]}")
    
    print("\n--- TE_11 Mode Degeneracy Analysis ---")
    wg.print_mode_info(1, 1, frequency, 'TE')
    
    print("\n--- Symmetry-based Excitation Analysis ---")
    print("\nX-directed electric dipole excitation:")
    modes_x = wg.extract_modes_by_symmetry(frequency, ExcitationType.ELECTRIC_DIPOLE_X, max_m=2, max_n=2)
    for mode_type, m, n, coeff in modes_x:
        print(f"  {mode_type}_{m}{n}: coefficient = {coeff:.4f}")
    
    print("\nY-directed electric dipole excitation:")
    modes_y = wg.extract_modes_by_symmetry(frequency, ExcitationType.ELECTRIC_DIPOLE_Y, max_m=2, max_n=2)
    for mode_type, m, n, coeff in modes_y:
        print(f"  {mode_type}_{m}{n}: coefficient = {coeff:.4f}")
    
    print("\nZ-directed electric dipole excitation (TM modes only):")
    modes_z = wg.extract_modes_by_symmetry(frequency, ExcitationType.ELECTRIC_DIPOLE_Z, max_m=2, max_n=2)
    for mode_type, m, n, coeff in modes_z:
        print(f"  {mode_type}_{m}{n}: coefficient = {coeff:.4f}")
    
    print("\n--- Polarization Control Demonstration ---")
    resolution = 50
    x = np.linspace(0, a, resolution)
    y = np.linspace(0, b, resolution)
    X, Y = np.meshgrid(x, y)
    
    try:
        if wg.is_propagating(1, 1, frequency, 'TE'):
            print(f"\nGenerating polarized TE_11 modes...")
            
            pol_x = Polarization.X_POLARIZED
            fields_x = wg.generate_polarized_mode(1, 1, frequency, pol_x, X, Y)
            Ex_max = np.max(np.abs(fields_x['Ex']))
            Ey_max = np.max(np.abs(fields_x['Ey']))
            print(f"  X-polarized: max(|Ex|) = {Ex_max:.4e}, max(|Ey|) = {Ey_max:.4e}")
            
            pol_y = Polarization.Y_POLARIZED
            fields_y = wg.generate_polarized_mode(1, 1, frequency, pol_y, X, Y)
            Ex_max = np.max(np.abs(fields_y['Ex']))
            Ey_max = np.max(np.abs(fields_y['Ey']))
            print(f"  Y-polarized: max(|Ex|) = {Ex_max:.4e}, max(|Ey|) = {Ey_max:.4e}")
            
            pol_rc = Polarization.CIRCULAR_RIGHT
            fields_rc = wg.generate_polarized_mode(1, 1, frequency, pol_rc, X, Y)
            Ex_max = np.max(np.abs(fields_rc['Ex']))
            Ey_max = np.max(np.abs(fields_rc['Ey']))
            ratio = Ey_max / Ex_max if Ex_max > 0 else 0
            print(f"  Right circular: max(|Ex|) = {Ex_max:.4e}, max(|Ey|) = {Ey_max:.4e}, ratio = {ratio:.4f}")
            
    except Exception as e:
        print(f"  Note: {e}")
    
    print("\n" + "=" * 70)
    print("Degenerate Mode Demo Complete!")
    print("=" * 70)


def demo_mode_matching():
    print("\n" + "=" * 70)
    print("Mode Matching (Orthogonal Projection) Demonstration")
    print("=" * 70)
    
    a = 22.86e-3
    b = 10.16e-3
    frequency = 10e9
    
    wg = RectangularWaveguide(a, b)
    
    resolution = 60
    x = np.linspace(0, a, resolution)
    y = np.linspace(0, b, resolution)
    X, Y = np.meshgrid(x, y)
    
    print(f"\nWaveguide: {a*1000:.2f} mm x {b*1000:.2f} mm")
    print(f"Frequency: {frequency/1e9:.2f} GHz")
    
    print("\n--- Test 1: Project TE10 mode onto itself ---")
    source_fields = wg.get_fields_complex(1, 0, frequency, X, Y, mode='TE', amplitude=2.0)
    
    candidates = [('TE', 1, 0), ('TE', 2, 0), ('TE', 0, 1), ('TE', 1, 1), ('TM', 1, 1)]
    amplitudes = wg.mode_matching(source_fields, frequency, candidates, resolution)
    
    print("  Expected: TE10 amplitude ~ 2.0, others ~ 0")
    for (mode_type, m, n), amp in amplitudes.items():
        print(f"  {mode_type}_{m}{n}: |amp| = {abs(amp):.6f}")
    
    print("\n--- Test 2: Hybrid source field ---")
    source_hybrid = {'Ex': 0j, 'Ey': 0j, 'Ez': 0j}
    
    fields_te10 = wg.get_fields_complex(1, 0, frequency, X, Y, mode='TE', amplitude=1.0)
    for comp in source_hybrid:
        source_hybrid[comp] += fields_te10[comp]
    
    try:
        fields_te20 = wg.get_fields_complex(2, 0, frequency, X, Y, mode='TE', amplitude=0.5)
        for comp in source_hybrid:
            source_hybrid[comp] += fields_te20[comp]
        
        amplitudes2 = wg.mode_matching(source_hybrid, frequency, candidates, resolution)
        print("  Expected: TE10 ~ 1.0, TE20 ~ 0.5, others ~ 0")
        for (mode_type, m, n), amp in amplitudes2.items():
            print(f"  {mode_type}_{m}{n}: |amp| = {abs(amp):.6f}")
    except ValueError as e:
        print(f"  TE20 not propagating: {e}")
    
    print("\n" + "=" * 70)
    print("Mode Matching Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_square_waveguide()
    demo_mode_matching()
