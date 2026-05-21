import numpy as np
from sn_solver import solve_sn_1d, Mesh, SNQuadrature, SNSolver, Material


def test_strong_absorber():
    """Test strong absorber - prone to negative fluxes with standard DD"""
    print("Testing strong absorber problem...")
    print("=" * 60)
    
    x_min, x_max = 0.0, 10.0
    n_cells = 20
    sn_order = 8
    
    sigma_t = 5.0
    sigma_s = 0.1
    source = np.ones(n_cells) * 10.0
    
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                  sigma_t, sigma_s, source, tol=1e-8)
    
    print(f"Material: sigma_t={sigma_t}, sigma_s={sigma_s}, c={sigma_s/sigma_t:.3f}")
    print(f"Cell size: dx={mesh.dx:.3f}, Optical thickness per cell: {sigma_t*mesh.dx:.3f}")
    print()
    
    min_phi = np.min(phi)
    min_psi = np.min(psi)
    
    print(f"Min scalar flux: {min_phi:.6e}")
    print(f"Min angular flux: {min_psi:.6e}")
    print()
    
    if min_phi < 0 or min_psi < 0:
        print("⚠ NEGATIVE FLUX DETECTED!")
        if min_phi < 0:
            neg_cells = np.sum(phi < 0)
            print(f"  Negative scalar flux in {neg_cells} cells")
            print(f"  Most negative: {min_phi:.6e} at x={mesh.x_centers[np.argmin(phi)]:.3f}")
        if min_psi < 0:
            neg_angles = np.sum(psi < 0)
            print(f"  Negative angular flux in {neg_angles} locations")
            idx = np.unravel_index(np.argmin(psi), psi.shape)
            quad = SNQuadrature(sn_order)
            print(f"  Most negative: {min_psi:.6e} at angle mu={quad.mus[idx[0]]:.4f}, x={mesh.x_edges[idx[1]]:.3f}")
    else:
        print("✓ No negative fluxes detected")
    
    print()
    return min_phi, min_psi


def test_large_optical_thickness():
    """Test problem with large optical thickness per cell"""
    print("\nTesting large optical thickness per cell...")
    print("=" * 60)
    
    x_min, x_max = 0.0, 5.0
    n_cells = 10
    sn_order = 8
    
    sigma_t = 10.0
    sigma_s = 0.5
    source = np.ones(n_cells)
    
    print(f"Optical thickness per cell: {sigma_t * (x_max-x_min)/n_cells:.1f}")
    print(f"Cell size: {(x_max-x_min)/n_cells:.3f}")
    print()
    
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                  sigma_t, sigma_s, source, tol=1e-8)
    
    min_phi = np.min(phi)
    min_psi = np.min(psi)
    
    print(f"Min scalar flux: {min_phi:.6e}")
    print(f"Min angular flux: {min_psi:.6e}")
    
    if min_phi < 0 or min_psi < 0:
        print("⚠ NEGATIVE FLUX DETECTED!")
    else:
        print("✓ No negative fluxes detected")
    
    return min_phi, min_psi


def test_boundary_layer():
    """Test problem with sharp boundary layer"""
    print("\nTesting sharp boundary layer problem...")
    print("=" * 60)
    
    x_min, x_max = 0.0, 3.0
    n_cells = 15
    sn_order = 16
    
    sigma_t = 8.0
    sigma_s = 4.0
    source = np.ones(n_cells) * 5.0
    
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                  sigma_t, sigma_s, source, tol=1e-8)
    
    min_phi = np.min(phi)
    min_psi = np.min(psi)
    
    print(f"Min scalar flux: {min_phi:.6e}")
    print(f"Min angular flux: {min_psi:.6e}")
    
    if min_phi < 0 or min_psi < 0:
        print("⚠ NEGATIVE FLUX DETECTED!")
    else:
        print("✓ No negative fluxes detected")
    
    return min_phi, min_psi


if __name__ == "__main__":
    print("Negative Flux Detection Test Suite")
    print("=" * 60)
    print()
    
    results = []
    results.append(test_strong_absorber())
    results.append(test_large_optical_thickness())
    results.append(test_boundary_layer())
    
    print("\n" + "=" * 60)
    print("Summary:")
    has_negative = any(min(p) < 0 for p in results)
    if has_negative:
        print("⚠ Negative fluxes detected - need positive-preserving method")
    else:
        print("✓ All tests passed - no negative fluxes")
    print("=" * 60)
