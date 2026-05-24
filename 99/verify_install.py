import numpy as np
import sys
sys.path.insert(0, '.')

print("Testing imports...")
try:
    from sn_solver import DiscreteOrdinatesSolver
    print("  ✓ sn_solver imported successfully")
except Exception as e:
    print(f"  ✗ sn_solver import error: {e}")

try:
    from monte_carlo import MonteCarloSolver, MCSNCoupling, DeepPenetrationBenchmark
    print("  ✓ monte_carlo imported successfully")
except Exception as e:
    print(f"  ✗ monte_carlo import error: {e}")

try:
    from quadrature import get_quadrature
    print("  ✓ quadrature imported successfully")
except Exception as e:
    print(f"  ✗ quadrature import error: {e}")

print("\nTesting SN solver (basic)...")
try:
    geometry = {'N': 10, 'L': 5.0}
    sigma_t = 1.0 * np.ones(10)
    sigma_s = 0.5 * np.ones(10)
    Q = 1.0 * np.ones(10)
    
    cross_sections = {'sigma_t': sigma_t, 'sigma_s': sigma_s, 'Q': Q}
    quadrature = get_quadrature(4, 'level_symmetric')
    bc = {'type': 'vacuum', 'left': 0.0, 'right': 0.0}
    
    solver = DiscreteOrdinatesSolver(geometry, cross_sections, quadrature, bc, use_positivity_fix=True)
    converged, iters, res = solver.solve()
    
    phi = solver.get_flux()
    check = solver.check_negative_flux()
    
    print(f"  ✓ SN solver converged: {converged}, iters: {iters}")
    print(f"  ✓ Max flux: {np.max(phi):.4f}")
    print(f"  ✓ Has negative flux: {check['has_negative']}")
except Exception as e:
    print(f"  ✗ SN solver error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting MC solver (basic)...")
try:
    geometry = {'N': 20, 'L': 10.0}
    sigma_t = 1.0 * np.ones(20)
    sigma_s = 0.9 * np.ones(20)
    Q = np.zeros(20)
    Q[0] = 10.0 / 0.5
    
    cross_sections = {'sigma_t': sigma_t, 'sigma_s': sigma_s, 'Q': Q}
    
    mc = MonteCarloSolver(geometry, cross_sections, seed=42)
    phi = mc.run_fixed_source(n_particles=1000)
    
    print(f"  ✓ MC solver completed")
    print(f"  ✓ Max flux: {np.max(phi):.4e}")
    print(f"  ✓ Particles tracked successfully")
except Exception as e:
    print(f"  ✗ MC solver error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Basic functionality test completed!")
print("="*60)
print("\nTo run full examples:")
print("  python example.py                 - SN examples")
print("  python test_negative_flux.py      - Negative flux fixes")
print("  python example_mc_sn_coupling.py  - MC-SN coupling demos")
print("\nFile structure:")
print("  sn_solver.py       - Discrete Ordinates (SN) solver")
print("  monte_carlo.py     - Monte Carlo solver + coupling")
print("  quadrature.py      - Angle quadrature sets")
print("  example.py         - SN examples")
print("  test_negative_flux.py - Negative flux verification")
print("  example_mc_sn_coupling.py - MC-SN coupling demos")
