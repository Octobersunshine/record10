import numpy as np
import sys
sys.path.insert(0, '.')

from sn_solver import DiscreteOrdinatesSolver
from monte_carlo import MonteCarloSolver, MCSNCoupling, DeepPenetrationBenchmark
from quadrature import get_quadrature


def example_deep_penetration_sn_only():
    print("=" * 70)
    print("EXAMPLE: Deep Penetration - SN Only (Reference)")
    print("=" * 70)
    
    benchmark = DeepPenetrationBenchmark(L=15.0, sigma_t=1.0, sigma_s_ratio=0.95, N=150)
    geometry = benchmark.get_geometry()
    cross_sections = benchmark.get_cross_sections()
    
    quadrature = get_quadrature(8, 'level_symmetric')
    
    boundary_conditions = {
        'type': 'vacuum',
        'left': 0.0,
        'right': 0.0
    }
    
    solver = DiscreteOrdinatesSolver(
        geometry, cross_sections, quadrature, boundary_conditions,
        use_positivity_fix=True
    )
    converged, iterations, residual = solver.solve(tol=1e-8, max_iter=2000)
    
    phi_sn = solver.get_flux()
    x = solver.x
    
    flux_check = solver.check_negative_flux()
    
    print(f"\nConverged: {converged}, Iterations: {iterations}")
    print(f"Max flux: {np.max(phi_sn):.4e} at x = {x[np.argmax(phi_sn)]:.2f}")
    print(f"Flux at 5 mean free paths: {phi_sn[50]:.4e}")
    print(f"Flux at 10 mean free paths: {phi_sn[100]:.4e}")
    print(f"Has negative flux: {flux_check['has_negative']}")
    
    if flux_check['has_negative']:
        print(f"  Min angular flux: {flux_check['min_angular_flux']:.4e}")
        print(f"  Min scalar flux: {flux_check['min_scalar_flux']:.4e}")
    
    return x, phi_sn, flux_check


def example_deep_penetration_mc_only():
    print("\n" + "=" * 70)
    print("EXAMPLE: Deep Penetration - MC Only (Reference)")
    print("=" * 70)
    
    benchmark = DeepPenetrationBenchmark(L=15.0, sigma_t=1.0, sigma_s_ratio=0.95, N=150)
    geometry = benchmark.get_geometry()
    cross_sections = benchmark.get_cross_sections()
    
    mc_solver = MonteCarloSolver(geometry, cross_sections, seed=42)
    phi_mc = mc_solver.run_fixed_source(n_particles=50000)
    x = mc_solver.x_edges[:-1] + mc_solver.dx/2
    
    phi_uncertainty = mc_solver.get_flux_uncertainty()
    leakage = mc_solver.get_leakage()
    
    print(f"\nTotal particles: 50000")
    print(f"Max flux: {np.max(phi_mc):.4e}")
    print(f"Mean relative uncertainty: {np.mean(phi_uncertainty / (phi_mc + 1e-10)):.2%}")
    print(f"Left leakage: {leakage[0]:.4e}, Right leakage: {leakage[1]:.4e}")
    print(f"Flux at 5 mean free paths: {phi_mc[50]:.4e}")
    print(f"Flux at 10 mean free paths: {phi_mc[100]:.4e}")
    
    return x, phi_mc, phi_uncertainty


def example_deep_penetration_coupled():
    print("\n" + "=" * 70)
    print("EXAMPLE: Deep Penetration - MC-SN Coupled")
    print("=" * 70)
    
    benchmark = DeepPenetrationBenchmark(L=15.0, sigma_t=1.0, sigma_s_ratio=0.95, N=150)
    geometry = benchmark.get_geometry()
    cross_sections = benchmark.get_cross_sections()
    
    quadrature = get_quadrature(8, 'level_symmetric')
    
    boundary_conditions = {
        'type': 'vacuum',
        'left': 0.0,
        'right': 0.0
    }
    
    mc_regions = [(7.5, 15.0)]
    
    print(f"\nMC regions: {mc_regions}")
    print(f"SN region: [0, 7.5)")
    print(f"MC region: [7.5, 15.0] (deep penetration zone)")
    
    coupled_solver = MCSNCoupling(
        geometry, cross_sections, mc_regions, n_particles=30000
    )
    
    phi_coupled = coupled_solver.solve_coupled(
        DiscreteOrdinatesSolver, quadrature, boundary_conditions,
        use_positivity_fix=True, max_iter=3, tol=1e-2
    )
    
    x = coupled_solver.x
    mc_mask = coupled_solver.get_mc_mask()
    
    print(f"\n=== Coupled Solution Summary ===")
    print(f"SN region ({np.sum(~mc_mask)} cells): max flux = {np.max(phi_coupled[~mc_mask]):.4e}")
    print(f"MC region ({np.sum(mc_mask)} cells): max flux = {np.max(phi_coupled[mc_mask]):.4e}")
    print(f"Flux at interface (x=7.5): {phi_coupled[75]:.4e}")
    print(f"Flux at 10 mean free paths: {phi_coupled[100]:.4e}")
    
    return x, phi_coupled, mc_mask


def compare_methods():
    print("\n" + "=" * 70)
    print("COMPARISON: SN vs MC vs Coupled")
    print("=" * 70)
    
    x_sn, phi_sn, flux_check_sn = example_deep_penetration_sn_only()
    x_mc, phi_mc, unc_mc = example_deep_penetration_mc_only()
    x_coupled, phi_coupled, mc_mask = example_deep_penetration_coupled()
    
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    
    positions = [25, 50, 75, 100, 125]
    print(f"{'Position (x)':>15} {'SN':>12} {'MC':>12} {'Coupled':>12} {'SN/Coupled':>12}")
    print("-" * 70)
    
    for idx in positions:
        x_pos = x_sn[idx]
        sn_val = phi_sn[idx]
        mc_val = phi_mc[idx] if idx < len(phi_mc) else 0.0
        coupled_val = phi_coupled[idx]
        ratio = sn_val / coupled_val if coupled_val > 0 else 0.0
        print(f"{x_pos:>15.2f} {sn_val:>12.4e} {mc_val:>12.4e} {coupled_val:>12.4e} {ratio:>12.4f}")
    
    print("\nKey observations:")
    print("  1. SN may have ray effects/oscillations in deep penetration regions")
    print("  2. MC has statistical noise but no ray effects")
    print("  3. Coupled method combines advantages:")
    print("     - High flux region: SN (fast, accurate)")
    print("     - Low flux region: MC (no ray effects)")
    
    return {
        'x': x_sn,
        'sn': phi_sn,
        'mc': phi_mc,
        'coupled': phi_coupled,
        'mc_mask': mc_mask
    }


def example_shielding_problem():
    print("\n" + "=" * 70)
    print("EXAMPLE: Multi-Layer Shielding Problem")
    print("=" * 70)
    
    L = 10.0
    N = 100
    dx = L / N
    
    geometry = {'N': N, 'L': L}
    
    sigma_t = np.ones(N)
    sigma_s = np.ones(N)
    Q = np.zeros(N)
    
    for i in range(N):
        x = i * dx
        if x < 2.0:
            sigma_t[i] = 1.0
            sigma_s[i] = 0.9
            Q[i] = 5.0 / dx if i == 0 else 0.0
        elif x < 5.0:
            sigma_t[i] = 5.0
            sigma_s[i] = 1.0
        else:
            sigma_t[i] = 10.0
            sigma_s[i] = 0.5
    
    cross_sections = {
        'sigma_t': sigma_t,
        'sigma_s': sigma_s,
        'Q': Q
    }
    
    quadrature = get_quadrature(16, 'level_symmetric')
    
    boundary_conditions = {
        'type': 'vacuum',
        'left': 0.0,
        'right': 0.0
    }
    
    print("\n--- SN Solution ---")
    solver_sn = DiscreteOrdinatesSolver(
        geometry, cross_sections, quadrature, boundary_conditions,
        use_positivity_fix=True
    )
    solver_sn.solve(tol=1e-8, max_iter=1000)
    phi_sn = solver_sn.get_flux()
    check_sn = solver_sn.check_negative_flux()
    
    print(f"Max flux: {np.max(phi_sn):.4e}")
    print(f"Has negative flux: {check_sn['has_negative']}")
    
    print("\n--- Coupled Solution ---")
    mc_regions = [(5.0, 10.0)]
    
    coupled_solver = MCSNCoupling(
        geometry, cross_sections, mc_regions, n_particles=20000
    )
    phi_coupled = coupled_solver.solve_coupled(
        DiscreteOrdinatesSolver, quadrature, boundary_conditions,
        use_positivity_fix=True, max_iter=3
    )
    mc_mask = coupled_solver.get_mc_mask()
    
    print(f"\nShielding performance:")
    print(f"  Source region (0-2cm): max flux = {np.max(phi_sn[:20]):.4e}")
    print(f"  High-Z shield (2-5cm): max flux = {np.max(phi_sn[20:50]):.4e}")
    print(f"  Dense shield (5-10cm) [MC]: max flux = {np.max(phi_coupled[mc_mask]):.4e}")
    
    x = np.linspace(dx/2, L - dx/2, N)
    return x, phi_sn, phi_coupled, mc_mask


def main():
    print("\n" + "=" * 70)
    print("MC-SN COUPLING DEMONSTRATION FOR DEEP PENETRATION")
    print("=" * 70)
    
    try:
        results = compare_methods()
        
        example_shielding_problem()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 70)
        
        print("\nUsage Guide:")
        print("-" * 50)
        print("1. Basic MC solver:")
        print("   mc = MonteCarloSolver(geometry, cross_sections)")
        print("   mc.run_fixed_source(n_particles=10000)")
        
        print("\n2. MC-SN coupled solver:")
        print("   coupled = MCSNCoupling(geometry, cross_sections, mc_regions)")
        print("   phi = coupled.solve_coupled(DiscreteOrdinatesSolver, quad, bc)")
        
        print("\n3. Deep penetration benchmark:")
        print("   benchmark = DeepPenetrationBenchmark(L=20, sigma_t=1.0)")
        print("   geometry = benchmark.get_geometry()")
        print("   mc_regions = benchmark.get_mc_regions_deep(threshold_x=10.0)")
        
        print("\nKey benefits of MC-SN coupling:")
        print("  - Eliminates ray effects in deep penetration regions")
        print("  - More accurate than pure SN for shielding problems")
        print("  - Faster than pure MC for high-flux regions")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
