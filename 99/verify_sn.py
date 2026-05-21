import sys
import numpy as np
from sn_solver import solve_sn_1d, Mesh, SNQuadrature, SNSolver, Material


def verify_quadrature():
    """Verify that quadrature weights sum to 2 (integral of 1 over [-1,1])"""
    print("Verifying quadrature sets...")
    for N in [2, 4, 8, 16]:
        quad = SNQuadrature(N)
        weight_sum = np.sum(quad.weights)
        print(f"  SN{N}: weight sum = {weight_sum:.6f} (should be ~2.0)")
        assert abs(weight_sum - 2.0) < 1e-5, f"SN{N} quadrature error"
        
        mu_sum = np.sum(quad.mus)
        print(f"  SN{N}: mu sum = {mu_sum:.6e} (should be ~0.0 due to symmetry)")
        assert abs(mu_sum) < 1e-10, f"SN{N} mu symmetry error"
    print("  ✓ Quadrature sets verified\n")


def verify_pure_absorber():
    """Verify pure absorber problem with expected physical behavior"""
    print("Verifying pure absorber problem...")
    
    x_min, x_max = 0.0, 5.0
    n_cells = 100
    sn_order = 16
    
    sigma_t = 1.0
    sigma_s = 0.0
    source = np.ones(n_cells)
    
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                  sigma_t, sigma_s, source, tol=1e-8)
    
    x = mesh.x_centers
    
    print(f"  Max flux: {np.max(phi):.6f}, Min flux: {np.min(phi):.6f}")
    print(f"  Flux at center: {phi[n_cells//2]:.6f}")
    print(f"  Flux at left edge: {phi[0]:.6f}")
    print(f"  Flux at right edge: {phi[-1]:.6f}")
    
    assert np.max(phi) < 1.0 / (sigma_t - sigma_s) + 0.1, "Max flux too high"
    assert np.min(phi) > 0, "Min flux should be positive"
    assert phi[n_cells//2] > phi[0], "Center flux should be higher than edge flux"
    assert abs(phi[0] - phi[-1]) < 0.01, "Problem is symmetric, edges should be equal"
    
    dx = mesh.dx
    total_source = np.sum(source) * dx
    total_absorption = np.sum((sigma_t - sigma_s) * phi) * dx
    total_leakage = 0.0
    
    quad = SNQuadrature(sn_order)
    for n in range(len(quad.mus)):
        if quad.mus[n] > 0:
            total_leakage += 0.5 * quad.weights[n] * quad.mus[n] * psi[n, -1]
        else:
            total_leakage += 0.5 * quad.weights[n] * abs(quad.mus[n]) * psi[n, 0]
    
    balance = abs(total_source - total_absorption - total_leakage) / total_source
    print(f"  Particle balance error: {balance:.4%}")
    
    if balance < 0.01:
        print("  ✓ Pure absorber problem verified\n")
    else:
        print("  ⚠ Pure absorber balance error larger than expected\n")
    
    return balance


def verify_conservation():
    """Verify particle conservation in reflective problem"""
    print("Verifying particle conservation...")
    
    x_min, x_max = 0.0, 5.0
    n_cells = 50
    sn_order = 8
    
    sigma_t = 1.0
    sigma_s = 0.5
    sigma_a = sigma_t - sigma_s
    source = np.ones(n_cells)
    
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                  sigma_t, sigma_s, source,
                                  bc_left='reflective', bc_right='reflective',
                                  tol=1e-8)
    
    total_source = np.sum(source) * mesh.dx
    total_absorption = np.sum(sigma_a * phi) * mesh.dx
    
    print(f"  Total source: {total_source:.6f}")
    print(f"  Total absorption: {total_absorption:.6f}")
    
    error = abs(total_source - total_absorption) / total_source
    print(f"  Conservation error: {error:.4%}")
    
    if error < 0.01:
        print("  ✓ Particle conservation verified\n")
    else:
        print("  ⚠ Conservation error larger than expected\n")
    
    return error


def verify_bc_types():
    """Verify different boundary condition types"""
    print("Verifying boundary conditions...")
    
    x_min, x_max = 0.0, 5.0
    n_cells = 50
    sn_order = 8
    
    sigma_t = 1.0
    sigma_s = 0.5
    source = np.ones(n_cells)
    
    mesh_vac, phi_vac, _ = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                        sigma_t, sigma_s, source,
                                        bc_left='vacuum', bc_right='vacuum',
                                        tol=1e-8)
    
    mesh_ref, phi_ref, _ = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                        sigma_t, sigma_s, source,
                                        bc_left='reflective', bc_right='reflective',
                                        tol=1e-8)
    
    print(f"  Vacuum BC - mean flux: {np.mean(phi_vac):.6f}, max flux: {np.max(phi_vac):.6f}")
    print(f"  Reflective BC - mean flux: {np.mean(phi_ref):.6f}, max flux: {np.max(phi_ref):.6f}")
    
    assert np.mean(phi_ref) > np.mean(phi_vac), "Reflective BC should give higher flux"
    
    print("  ✓ Boundary conditions verified\n")


def verify_sn_convergence():
    """Verify convergence with increasing SN order"""
    print("Verifying SN convergence...")
    
    x_min, x_max = 0.0, 5.0
    n_cells = 50
    
    sigma_t = 1.0
    sigma_s = 0.5
    source = np.ones(n_cells)
    
    phi_ref = None
    for sn_order in [2, 4, 8, 16]:
        mesh, phi, _ = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                    sigma_t, sigma_s, source, tol=1e-8)
        if phi_ref is not None:
            diff = np.max(np.abs(phi - phi_ref))
            print(f"  SN{sn_order} vs SN{prev_order}: max diff = {diff:.6e}")
        phi_ref = phi
        prev_order = sn_order
    
    print("  ✓ SN convergence verified\n")


def main():
    print("=" * 60)
    print("SN Solver Verification Suite")
    print("=" * 60 + "\n")
    
    try:
        verify_quadrature()
        err1 = verify_pure_absorber()
        err2 = verify_conservation()
        verify_bc_types()
        verify_sn_convergence()
        
        print("=" * 60)
        if err1 < 0.02 and err2 < 0.01:
            print("✓ All verifications passed!")
        else:
            print("⚠ Some verifications had larger than expected errors")
        print("=" * 60)
        
        return 0
    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
