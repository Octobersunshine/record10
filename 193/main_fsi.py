import numpy as np
from fsi_solver import FSISolver, compare_fsi_vs_rigid
from structural_model import StructuralDynamics


def run_fsi_analysis():
    print("\n" + "#" * 70)
    print("#  Flexible Wing Design - Fluid-Structure Interaction")
    print("#  Using Strong Coupling (Staggered Method with Aitken Relaxation)")
    print("#" * 70)
    
    config = {
        'n_panels': 80,
        'n_struct_points': 50,
        'V_inf': 10.0,
        'chord': 1.0,
        'rho': 1.225,
        't_final': 0.5,
        'dt': 0.005,
        'alpha_mean': 5.0,
        'alpha_amplitude': 2.0,
        'pitch_frequency': 2 * np.pi,
        'structural_model': 'spring',
        'stiffness': 500.0,
        'damping': 50.0,
        'mass_per_length': 10.0,
        'max_fsi_iter': 100,
        'fsi_tol': 1e-8,
        'relaxation': 0.5,
        'use_aitken': True
    }
    
    solver = FSISolver(config)
    results = solver.solve()
    
    solver.plot_results(results, 'flexible_wing')
    solver.plot_deformed_shape(results, 'flexible_wing')
    
    return results


def parametric_study():
    print("\n" + "#" * 70)
    print("#  Parametric Study: Effect of Structural Stiffness")
    print("#" * 70)
    
    stiffness_values = [100, 500, 1000, 5000]
    results_list = []
    
    for stiffness in stiffness_values:
        print(f"\nStiffness = {stiffness} N/m")
        print("-" * 50)
        
        config = {
            'n_panels': 60,
            'n_struct_points': 30,
            'V_inf': 10.0,
            'chord': 1.0,
            'rho': 1.225,
            't_final': 0.3,
            'dt': 0.01,
            'alpha_mean': 5.0,
            'alpha_amplitude': 1.0,
            'pitch_frequency': 2 * np.pi,
            'stiffness': stiffness,
            'damping': stiffness * 0.1,
            'max_fsi_iter': 100,
            'fsi_tol': 1e-8
        }
        
        solver = FSISolver(config)
        results = solver.solve()
        results_list.append((stiffness, results))
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for stiffness, results in results_list:
        axes[0, 0].plot(results['times'], results['Cl'], 
                       linewidth=2, label=f'k={stiffness} N/m')
        axes[0, 1].plot(results['times'], np.array(results['w_tip']) * 1000, 
                       linewidth=2, label=f'k={stiffness} N/m')
    
    axes[0, 0].set_ylabel('$C_l$')
    axes[0, 0].set_title('Lift Coefficient vs Stiffness')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_ylabel('Tip Deflection (mm)')
    axes[0, 1].set_title('Tip Deflection vs Stiffness')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    stiffness_arr = [s for s, _ in results_list]
    mean_Cl = [np.mean(r['Cl']) for _, r in results_list]
    max_deflection = [np.max(r['w_max']) * 1000 for _, r in results_list]
    
    axes[1, 0].plot(stiffness_arr, mean_Cl, 'bo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Stiffness (N/m)')
    axes[1, 0].set_ylabel('Mean $C_l$')
    axes[1, 0].set_title('Mean Lift vs Stiffness')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(stiffness_arr, max_deflection, 'rs-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Stiffness (N/m)')
    axes[1, 1].set_ylabel('Max Deflection (mm)')
    axes[1, 1].set_title('Max Deflection vs Stiffness')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parametric_study.png', dpi=150)
    plt.close()
    print("\nParametric study plot saved: parametric_study.png")
    
    return results_list


def test_convergence():
    print("\n" + "#" * 70)
    print("#  Convergence Study: Effect of Tolerance and Relaxation")
    print("#" * 70)
    
    tolerances = [1e-4, 1e-6, 1e-8]
    relaxations = [0.3, 0.5, 0.7]
    
    results_conv = []
    
    for tol in tolerances:
        for relax in relaxations:
            config = {
                'n_panels': 40,
                'n_struct_points': 20,
                'V_inf': 10.0,
                'chord': 1.0,
                't_final': 0.1,
                'dt': 0.01,
                'alpha_mean': 3.0,
                'alpha_amplitude': 0.0,
                'stiffness': 500.0,
                'damping': 50.0,
                'max_fsi_iter': 100,
                'fsi_tol': tol,
                'relaxation': relax,
                'use_aitken': False
            }
            
            solver = FSISolver(config)
            results = solver.solve()
            results_conv.append((tol, relax, results))
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for tol in tolerances:
        for relax in relaxations:
            label = f'tol={tol:.0e}, ω={relax}'
            mean_iter = np.mean([
                r['iterations'] for t, w, r in results_conv 
                if t == tol and w == relax
            ])
            ax.bar(label, mean_iter, alpha=0.7)
    
    ax.set_ylabel('Mean FSI Iterations')
    ax.set_title('Convergence: Iterations vs Tolerance and Relaxation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('convergence_study.png', dpi=150)
    plt.close()
    print("Convergence study plot saved: convergence_study.png")
    
    return results_conv


def main():
    print("\n" + "#" * 70)
    print("#  Flexible Wing Design - FSI Analysis Suite")
    print("#  Strong Coupling with Aitken Relaxation")
    print("#" * 70)
    
    results_rigid, results_flexible = compare_fsi_vs_rigid()
    
    results_flexible = run_fsi_analysis()
    
    parametric_results = parametric_study()
    
    convergence_results = test_convergence()
    
    print("\n" + "#" * 70)
    print("#  FSI Analysis Complete!")
    print("#" * 70)
    print("\nKey Features:")
    print("  ✓ Fluid-Structure Interaction (Strong Coupling)")
    print("  ✓ Staggered Method with Aitken Relaxation")
    print("  ✓ Aerodynamic loads from Improved VLM")
    print("  ✓ Structural dynamics (spring/beam models)")
    print("  ✓ Convergence monitoring and parametric studies")
    print("\nOutput Files:")
    print("  - fsi_results_rigid.png")
    print("  - fsi_results_flexible.png")
    print("  - fsi_results_flexible_wing.png")
    print("  - deformed_shape_flexible_wing.png")
    print("  - fsi_comparison.png")
    print("  - parametric_study.png")
    print("  - convergence_study.png")
    print("#" * 70)


if __name__ == "__main__":
    main()
