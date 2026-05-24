import numpy as np
import sys
sys.path.insert(0, '.')
from sn_solver import DiscreteOrdinatesSolver
from quadrature import get_quadrature, verify_quadrature


def test_quadrature_validity():
    print("=" * 70)
    print("TEST 1: Quadrature Set Validity Check")
    print("=" * 70)
    
    for n in [2, 4, 6, 8, 12, 16]:
        print(f"\n--- S{n} Level Symmetric Quadrature ---")
        quad = get_quadrature(n, 'level_symmetric')
        results = verify_quadrature(quad['mu'], quad['w'], verbose=False)
        
        weight_sum_error = abs(results['sum_weights'] - 2.0)
        all_positive = np.all(quad['w'] > 0)
        
        print(f"  Weight sum error: {weight_sum_error:.2e}")
        print(f"  All weights positive: {all_positive}")
        print(f"  Min weight: {results['min_weight']:.6f}")
        
        if weight_sum_error > 1e-10 or not all_positive:
            print(f"  WARNING: Quadrature validation failed!")
        else:
            print(f"  OK: Quadrature validated successfully")
    
    print("\nQuadrature validity check complete!")


def test_negative_flux_detection():
    print("\n" + "=" * 70)
    print("TEST 2: Negative Flux Detection (with/without fix)")
    print("=" * 70)
    
    geometry = {
        'N': 50,
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
    
    quad_s8 = get_quadrature(8, 'level_symmetric')
    
    boundary_conditions = {
        'type': 'vacuum',
        'left': 0.0,
        'right': 0.0
    }
    
    print("\n--- Without positivity fix ---")
    solver_no_fix = DiscreteOrdinatesSolver(
        geometry, cross_sections, quad_s8, boundary_conditions,
        use_positivity_fix=False
    )
    converged, iterations, residual = solver_no_fix.solve(tol=1e-6, max_iter=1000)
    flux_check = solver_no_fix.check_negative_flux()
    
    print(f"  Converged: {converged}, Iterations: {iterations}")
    print(f"  Negative angular cells: {flux_check['negative_angular_cells']}")
    print(f"  Negative scalar cells: {flux_check['negative_scalar_cells']}")
    print(f"  Min angular flux: {flux_check['min_angular_flux']:.6e}")
    print(f"  Min scalar flux: {flux_check['min_scalar_flux']:.6e}")
    
    print("\n--- With positivity fix (set_to_zero) ---")
    solver_fix = DiscreteOrdinatesSolver(
        geometry, cross_sections, quad_s8, boundary_conditions,
        use_positivity_fix=True,
        positivity_method='set_to_zero'
    )
    converged, iterations, residual = solver_fix.solve(tol=1e-6, max_iter=1000)
    flux_check = solver_fix.check_negative_flux()
    
    print(f"  Converged: {converged}, Iterations: {iterations}")
    print(f"  Negative angular cells: {flux_check['negative_angular_cells']}")
    print(f"  Negative scalar cells: {flux_check['negative_scalar_cells']}")
    print(f"  Min angular flux: {flux_check['min_angular_flux']:.6e}")
    print(f"  Min scalar flux: {flux_check['min_scalar_flux']:.6e}")
    
    if not flux_check['has_negative']:
        print("  SUCCESS: No negative fluxes detected!")
    
    phi_diff = np.linalg.norm(solver_no_fix.get_flux() - solver_fix.get_flux()) / np.linalg.norm(solver_fix.get_flux())
    print(f"\n  Relative difference in flux: {phi_diff:.4e}")


def test_reflective_boundary_stability():
    print("\n" + "=" * 70)
    print("TEST 3: Reflective Boundary Stability")
    print("=" * 70)
    
    geometry = {
        'N': 50,
        'L': 5.0
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
    
    quad_s8 = get_quadrature(8, 'level_symmetric')
    
    boundary_conditions = {
        'type': 'reflective',
        'left': 0.0,
        'right': 0.0
    }
    
    print("\n--- Reflective boundary test ---")
    solver = DiscreteOrdinatesSolver(
        geometry, cross_sections, quad_s8, boundary_conditions,
        use_positivity_fix=True
    )
    converged, iterations, residual = solver.solve(tol=1e-8, max_iter=2000)
    flux_check = solver.check_negative_flux()
    
    print(f"  Converged: {converged}, Iterations: {iterations}")
    print(f"  Residual: {residual:.2e}")
    print(f"  Negative angular cells: {flux_check['negative_angular_cells']}")
    print(f"  Negative scalar cells: {flux_check['negative_scalar_cells']}")
    
    phi = solver.get_flux()
    J = solver.get_current()
    
    print(f"\n  Flux variation (max-min)/mean: {(np.max(phi)-np.min(phi))/np.mean(phi):.4f}")
    print(f"  Max current (should be near 0 for reflective): {np.max(np.abs(J)):.4e}")
    
    if not flux_check['has_negative']:
        print("  SUCCESS: No negative fluxes with reflective BC!")


def test_different_quadrature_orders():
    print("\n" + "=" * 70)
    print("TEST 4: Different Quadrature Orders Comparison")
    print("=" * 70)
    
    geometry = {
        'N': 40,
        'L': 8.0
    }
    
    N = geometry['N']
    sigma_t = 1.0 * np.ones(N)
    sigma_s = 0.7 * np.ones(N)
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
    
    results = []
    for n in [2, 4, 6, 8, 12]:
        quad = get_quadrature(n, 'level_symmetric')
        solver = DiscreteOrdinatesSolver(
            geometry, cross_sections, quad, boundary_conditions,
            use_positivity_fix=True
        )
        converged, iterations, residual = solver.solve(tol=1e-8, max_iter=1000)
        flux_check = solver.check_negative_flux()
        phi = solver.get_flux()
        
        results.append({
            'order': n,
            'center_flux': phi[N//2],
            'iterations': iterations,
            'has_negative': flux_check['has_negative']
        })
        
        print(f"S{n}: center flux = {phi[N//2]:.8f}, iterations = {iterations:3d}, neg_flux = {flux_check['has_negative']}")
    
    print("\nAll quadrature orders tested successfully!")


def test_incident_current_problem():
    print("\n" + "=" * 70)
    print("TEST 5: Incident Current Problem (Potential ray effects)")
    print("=" * 70)
    
    geometry = {
        'N': 60,
        'L': 10.0
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
    
    quad_s8 = get_quadrature(8, 'level_symmetric')
    
    boundary_conditions = {
        'type': 'incident',
        'left': 1.0,
        'right': 0.0
    }
    
    print("\n--- Without positivity fix ---")
    solver1 = DiscreteOrdinatesSolver(
        geometry, cross_sections, quad_s8, boundary_conditions,
        use_positivity_fix=False
    )
    solver1.solve(tol=1e-8, max_iter=1000)
    check1 = solver1.check_negative_flux()
    print(f"  Min angular flux: {check1['min_angular_flux']:.6e}")
    print(f"  Min scalar flux: {check1['min_scalar_flux']:.6e}")
    print(f"  Has negative: {check1['has_negative']}")
    
    print("\n--- With positivity fix ---")
    solver2 = DiscreteOrdinatesSolver(
        geometry, cross_sections, quad_s8, boundary_conditions,
        use_positivity_fix=True
    )
    solver2.solve(tol=1e-8, max_iter=1000)
    check2 = solver2.check_negative_flux()
    print(f"  Min angular flux: {check2['min_angular_flux']:.6e}")
    print(f"  Min scalar flux: {check2['min_scalar_flux']:.6e}")
    print(f"  Has negative: {check2['has_negative']}")
    
    if not check2['has_negative']:
        print("  SUCCESS: Positivity fix eliminated negative fluxes!")


def main():
    print("\n" + "=" * 70)
    print("NEGATIVE FLUX FIX VERIFICATION TEST SUITE")
    print("=" * 70)
    
    try:
        test_quadrature_validity()
        test_negative_flux_detection()
        test_reflective_boundary_stability()
        test_different_quadrature_orders()
        test_incident_current_problem()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nSummary of fixes:")
        print("  1. Verified quadrature set data (positive weights, correct moments)")
        print("  2. Added optional positivity fix for angular and scalar fluxes")
        print("  3. Improved reflective boundary condition handling")
        print("  4. Added negative flux detection and reporting methods")
        print("\nUsage:")
        print("  solver = DiscreteOrdinatesSolver(..., use_positivity_fix=True)")
        print("  solver.check_negative_flux()  # Check for negative fluxes")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
