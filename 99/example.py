import numpy as np
import matplotlib.pyplot as plt
from sn_solver import DiscreteOrdinatesSolver
from quadrature import get_quadrature, verify_quadrature, list_available_quadratures


def example1_uniform_source():
    print("=" * 60)
    print("Example 1: Uniform Source in Slab")
    print("=" * 60)
    
    geometry = {
        'N': 100,
        'L': 10.0
    }
    
    N = geometry['N']
    sigma_t = 1.0 * np.ones(N)
    sigma_s = 0.5 * np.ones(N)
    Q = 1.0 * np.ones(N)
    
    cross_sections = {
        'sigma_t': sigma_t,
        'sigma_s': sigma_s,
        'Q': Q
    }
    
    quadrature = get_quadrature(8, quadrature_type='legendre')
    verify_quadrature(quadrature['mu'], quadrature['w'])
    
    boundary_conditions = {
        'type': 'vacuum',
        'left': 0.0,
        'right': 0.0
    }
    
    solver = DiscreteOrdinatesSolver(geometry, cross_sections, quadrature, boundary_conditions)
    converged, iterations, residual = solver.solve(tol=1e-10, max_iter=5000)
    
    print(f"\nConverged: {converged}")
    print(f"Iterations: {iterations}")
    print(f"Final residual: {residual:.2e}")
    
    phi = solver.get_flux()
    J = solver.get_current()
    x = solver.x
    
    print(f"\nCenter flux: {phi[N//2]:.6f}")
    print(f"Edge flux (left): {phi[0]:.6f}")
    print(f"Edge flux (right): {phi[-1]:.6f}")
    
    return x, phi, J


def example2_two_region():
    print("\n" + "=" * 60)
    print("Example 2: Two-Region Slab")
    print("=" * 60)
    
    geometry = {
        'N': 200,
        'L': 20.0
    }
    
    N = geometry['N']
    sigma_t = np.ones(N)
    sigma_s = np.ones(N)
    Q = np.zeros(N)
    
    mid = N // 2
    sigma_t[:mid] = 1.0
    sigma_s[:mid] = 0.8
    sigma_t[mid:] = 5.0
    sigma_s[mid:] = 0.1
    Q[:mid] = 2.0
    
    cross_sections = {
        'sigma_t': sigma_t,
        'sigma_s': sigma_s,
        'Q': Q
    }
    
    quadrature = get_quadrature(16, quadrature_type='legendre')
    
    boundary_conditions = {
        'type': 'vacuum',
        'left': 0.0,
        'right': 0.0
    }
    
    solver = DiscreteOrdinatesSolver(geometry, cross_sections, quadrature, boundary_conditions)
    converged, iterations, residual = solver.solve(tol=1e-10, max_iter=5000)
    
    print(f"Converged: {converged}")
    print(f"Iterations: {iterations}")
    print(f"Final residual: {residual:.2e}")
    
    phi = solver.get_flux()
    J = solver.get_current()
    x = solver.x
    
    return x, phi, J


def example3_incident_current():
    print("\n" + "=" * 60)
    print("Example 3: Incident Current on Left Boundary")
    print("=" * 60)
    
    geometry = {
        'N': 100,
        'L': 5.0
    }
    
    N = geometry['N']
    sigma_t = 2.0 * np.ones(N)
    sigma_s = 1.5 * np.ones(N)
    Q = np.zeros(N)
    
    cross_sections = {
        'sigma_t': sigma_t,
        'sigma_s': sigma_s,
        'Q': Q
    }
    
    quadrature = get_quadrature(8, quadrature_type='level_symmetric')
    
    boundary_conditions = {
        'type': 'incident',
        'left': 1.0,
        'right': 0.0
    }
    
    solver = DiscreteOrdinatesSolver(geometry, cross_sections, quadrature, boundary_conditions)
    converged, iterations, residual = solver.solve(tol=1e-10, max_iter=5000)
    
    print(f"Converged: {converged}")
    print(f"Iterations: {iterations}")
    print(f"Final residual: {residual:.2e}")
    
    phi = solver.get_flux()
    J = solver.get_current()
    x = solver.x
    
    print(f"\nLeft boundary flux: {phi[0]:.6f}")
    print(f"Right boundary flux: {phi[-1]:.6f}")
    
    return x, phi, J


def example4_reflective_boundary():
    print("\n" + "=" * 60)
    print("Example 4: Reflective Boundary Conditions")
    print("=" * 60)
    
    geometry = {
        'N': 100,
        'L': 10.0
    }
    
    N = geometry['N']
    sigma_t = 1.0 * np.ones(N)
    sigma_s = 0.9 * np.ones(N)
    Q = 1.0 * np.ones(N)
    
    cross_sections = {
        'sigma_t': sigma_t,
        'sigma_s': sigma_s,
        'Q': Q
    }
    
    quadrature = get_quadrature(8, quadrature_type='legendre')
    
    boundary_conditions = {
        'type': 'reflective',
        'left': 0.0,
        'right': 0.0
    }
    
    solver = DiscreteOrdinatesSolver(geometry, cross_sections, quadrature, boundary_conditions)
    converged, iterations, residual = solver.solve(tol=1e-10, max_iter=5000)
    
    print(f"Converged: {converged}")
    print(f"Iterations: {iterations}")
    print(f"Final residual: {residual:.2e}")
    
    phi = solver.get_flux()
    J = solver.get_current()
    x = solver.x
    
    print(f"\nFlux is nearly flat due to reflection")
    print(f"Min flux: {np.min(phi):.6f}")
    print(f"Max flux: {np.max(phi):.6f}")
    print(f"Current near zero: {np.max(np.abs(J)):.2e}")
    
    return x, phi, J


def example5_dsa_comparison():
    print("\n" + "=" * 60)
    print("Example 5: DSA Acceleration Comparison")
    print("=" * 60)
    
    geometry = {
        'N': 100,
        'L': 10.0
    }
    
    N = geometry['N']
    sigma_t = 1.0 * np.ones(N)
    sigma_s = 0.99 * np.ones(N)
    Q = 1.0 * np.ones(N)
    
    cross_sections = {
        'sigma_t': sigma_t,
        'sigma_s': sigma_s,
        'Q': Q
    }
    
    quadrature = get_quadrature(8, quadrature_type='legendre')
    
    boundary_conditions = {
        'type': 'vacuum',
        'left': 0.0,
        'right': 0.0
    }
    
    print("\nWithout DSA:")
    solver1 = DiscreteOrdinatesSolver(geometry, cross_sections, quadrature, boundary_conditions)
    converged1, iterations1, residual1 = solver1.solve(tol=1e-8, max_iter=5000, use_dsa=False)
    print(f"  Converged: {converged1}")
    print(f"  Iterations: {iterations1}")
    print(f"  Final residual: {residual1:.2e}")
    
    print("\nWith DSA:")
    solver2 = DiscreteOrdinatesSolver(geometry, cross_sections, quadrature, boundary_conditions)
    converged2, iterations2, residual2 = solver2.solve(tol=1e-8, max_iter=5000, use_dsa=True)
    print(f"  Converged: {converged2}")
    print(f"  Iterations: {iterations2}")
    print(f"  Final residual: {residual2:.2e}")
    
    if iterations1 > 0:
        speedup = iterations1 / iterations2
        print(f"\nDSA speedup: {speedup:.1f}x")
    
    phi = solver2.get_flux()
    J = solver2.get_current()
    x = solver2.x
    
    return x, phi, J


def example6_quadrature_comparison():
    print("\n" + "=" * 60)
    print("Example 6: Quadrature Comparison")
    print("=" * 60)
    
    geometry = {
        'N': 50,
        'L': 5.0
    }
    
    N = geometry['N']
    sigma_t = 1.0 * np.ones(N)
    sigma_s = 0.5 * np.ones(N)
    Q = 1.0 * np.ones(N)
    
    cross_sections = {
        'sigma_t': sigma_t,
        'sigma_s': sigma_s,
        'Q': Q
    }
    
    boundary_conditions = {
        'type': 'vacuum',
        'left': 0.0,
        'right': 0.0
    }
    
    quadrature_types = ['legendre', 'level_symmetric', 'chebyshev']
    
    results = []
    for qtype in quadrature_types:
        try:
            quadrature = get_quadrature(8, quadrature_type=qtype)
            solver = DiscreteOrdinatesSolver(geometry, cross_sections, quadrature, boundary_conditions)
            converged, iterations, residual = solver.solve(tol=1e-8, max_iter=1000)
            phi = solver.get_flux()
            flux_check = solver.check_negative_flux()
            results.append((qtype, phi[N//2], iterations, flux_check['has_negative']))
            print(f"{qtype:15s}: center flux = {phi[N//2]:.8f}, iterations = {iterations}, neg_flux = {flux_check['has_negative']}")
        except Exception as e:
            print(f"{qtype:15s}: Error: {e}")
    
    return solver.x, solver.get_flux(), solver.get_current()


def example7_neg_flux_fix_verification():
    print("\n" + "=" * 60)
    print("Example 7: Negative Flux Fix Verification")
    print("=" * 60)
    
    geometry = {
        'N': 60,
        'L': 10.0
    }
    
    N = geometry['N']
    sigma_t = 1.0 * np.ones(N)
    sigma_s = 0.95 * np.ones(N)
    Q = 1.0 * np.ones(N)
    
    cross_sections = {
        'sigma_t': sigma_t,
        'sigma_s': sigma_s,
        'Q': Q
    }
    
    quadrature = get_quadrature(8, quadrature_type='level_symmetric')
    
    boundary_conditions = {
        'type': 'vacuum',
        'left': 0.0,
        'right': 0.0
    }
    
    print("\n--- Without positivity fix ---")
    solver1 = DiscreteOrdinatesSolver(
        geometry, cross_sections, quadrature, boundary_conditions,
        use_positivity_fix=False
    )
    solver1.solve(tol=1e-6, max_iter=500)
    check1 = solver1.check_negative_flux()
    print(f"  Negative angular cells: {check1['negative_angular_cells']}")
    print(f"  Negative scalar cells: {check1['negative_scalar_cells']}")
    print(f"  Min angular flux: {check1['min_angular_flux']:.6e}")
    print(f"  Min scalar flux: {check1['min_scalar_flux']:.6e}")
    
    print("\n--- With positivity fix ---")
    solver2 = DiscreteOrdinatesSolver(
        geometry, cross_sections, quadrature, boundary_conditions,
        use_positivity_fix=True,
        positivity_method='set_to_zero'
    )
    solver2.solve(tol=1e-6, max_iter=500)
    check2 = solver2.check_negative_flux()
    print(f"  Negative angular cells: {check2['negative_angular_cells']}")
    print(f"  Negative scalar cells: {check2['negative_scalar_cells']}")
    print(f"  Min angular flux: {check2['min_angular_flux']:.6e}")
    print(f"  Min scalar flux: {check2['min_scalar_flux']:.6e}")
    
    if not check2['has_negative']:
        print("\n  SUCCESS: Negative flux problem fixed!")
    
    return solver2.x, solver2.get_flux(), solver2.get_current()


def plot_results(x_list, phi_list, J_list, titles):
    n = len(x_list)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        axes[i, 0].plot(x_list[i], phi_list[i], 'b-', linewidth=2)
        axes[i, 0].set_xlabel('Position (cm)')
        axes[i, 0].set_ylabel('Scalar Flux')
        axes[i, 0].set_title(titles[i] + ' - Scalar Flux')
        axes[i, 0].grid(True)
        
        axes[i, 1].plot(x_list[i], J_list[i], 'r-', linewidth=2)
        axes[i, 1].set_xlabel('Position (cm)')
        axes[i, 1].set_ylabel('Current')
        axes[i, 1].set_title(titles[i] + ' - Current')
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('sn_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'sn_results.png'")


def main():
    print("\nDiscrete Ordinates (SN) Neutron Transport Solver")
    print("=" * 60)
    print("\nAvailable quadrature sets:")
    list_available_quadratures()
    
    x1, phi1, J1 = example1_uniform_source()
    x2, phi2, J2 = example2_two_region()
    x3, phi3, J3 = example3_incident_current()
    x4, phi4, J4 = example4_reflective_boundary()
    x5, phi5, J5 = example5_dsa_comparison()
    x6, phi6, J6 = example6_quadrature_comparison()
    x7, phi7, J7 = example7_neg_flux_fix_verification()
    
    x_list = [x1, x2, x3, x4, x5, x7]
    phi_list = [phi1, phi2, phi3, phi4, phi5, phi7]
    J_list = [J1, J2, J3, J4, J5, J7]
    titles = ['Uniform Source', 'Two-Region', 'Incident Current', 
              'Reflective BC', 'DSA Test', 'Neg Flux Fix']
    
    try:
        plot_results(x_list, phi_list, J_list, titles)
    except Exception as e:
        print(f"\nWarning: Could not generate plot: {e}")
        print("Results computed successfully.")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
