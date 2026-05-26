import numpy as np
from fsi_solver import FSISolver


def quick_test():
    print("\n" + "=" * 70)
    print("Quick FSI Test - Flexible Wing Under Aerodynamic Load")
    print("=" * 70)
    
    config = {
        'n_panels': 40,
        'n_struct_points': 20,
        'V_inf': 10.0,
        'chord': 1.0,
        'rho': 1.225,
        't_final': 0.2,
        'dt': 0.01,
        'alpha_mean': 3.0,
        'alpha_amplitude': 1.0,
        'pitch_frequency': 2 * np.pi,
        'structural_model': 'spring',
        'stiffness': 500.0,
        'damping': 50.0,
        'mass_per_length': 10.0,
        'max_fsi_iter': 30,
        'fsi_tol': 1e-6,
        'relaxation': 0.5,
        'use_aitken': True
    }
    
    solver = FSISolver(config)
    results = solver.solve()
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Final time: {results['times'][-1]:.4f} s")
    print(f"  Mean Cl: {np.mean(results['Cl']):.4f}")
    print(f"  Peak Cl: {np.max(results['Cl']):.4f}")
    print(f"  Max tip deflection: {np.max(results['w_tip'])*1000:.3f} mm")
    print(f"  Mean FSI iterations: {np.mean(results['iterations']):.1f}")
    print(f"  Converged steps: {sum(results['converged'])}/{len(results['converged'])}")
    
    solver.plot_results(results, 'quick_test')
    solver.plot_deformed_shape(results, 'quick_test')
    
    print("\nQuick test completed successfully!")
    print("Files generated:")
    print("  - fsi_results_quick_test.png")
    print("  - deformed_shape_quick_test.png")
    
    return results


if __name__ == "__main__":
    quick_test()
