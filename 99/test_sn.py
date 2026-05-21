import numpy as np
import matplotlib.pyplot as plt
from sn_solver import solve_sn_1d, Mesh, SNQuadrature, SNSolver, Material


def test_source_problem():
    print("=" * 60)
    print("Test 1: Fixed Source Problem (Pure Absorber)")
    print("=" * 60)
    
    x_min, x_max = 0.0, 5.0
    n_cells = 50
    sn_order = 8
    
    sigma_t = 1.0
    sigma_s = 0.0
    source = np.ones(n_cells)
    
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                  sigma_t, sigma_s, source)
    
    x = mesh.x_centers
    dx = mesh.dx
    sigma_a = sigma_t - sigma_s
    phi_analytic = 1.0 / sigma_a * (1 - np.exp(-sigma_a * x) - np.exp(-sigma_a * (x_max - x)) + np.exp(-sigma_a * x_max))
    phi_analytic = 1.0 / sigma_a * (1 - (np.sinh(sigma_a * (x_max - x)) + np.sinh(sigma_a * x)) / np.sinh(sigma_a * x_max))
    
    print(f"Max scalar flux: {np.max(phi):.6f}")
    print(f"Min scalar flux: {np.min(phi):.6f}")
    print(f"Mean scalar flux: {np.mean(phi):.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, phi, 'b-', linewidth=2, label=f'SN{sn_order}')
    plt.xlabel('x (cm)')
    plt.ylabel('Scalar Flux')
    plt.title('Fixed Source Problem - Pure Absorber')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test1_pure_absorber.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to test1_pure_absorber.png")
    return phi


def test_scattering_problem():
    print("\n" + "=" * 60)
    print("Test 2: Scattering Medium")
    print("=" * 60)
    
    x_min, x_max = 0.0, 10.0
    n_cells = 100
    sn_order = 8
    
    sigma_t = 1.0
    sigma_s = 0.8
    source = np.ones(n_cells)
    
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                  sigma_t, sigma_s, source)
    
    x = mesh.x_centers
    
    print(f"Max scalar flux: {np.max(phi):.6f}")
    print(f"Min scalar flux: {np.min(phi):.6f}")
    print(f"Mean scalar flux: {np.mean(phi):.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, phi, 'r-', linewidth=2, label=f'SN{sn_order}, c={sigma_s/sigma_t:.2f}')
    plt.xlabel('x (cm)')
    plt.ylabel('Scalar Flux')
    plt.title('Fixed Source Problem - Scattering Medium')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test2_scattering.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to test2_scattering.png")
    return phi


def test_sn_convergence():
    print("\n" + "=" * 60)
    print("Test 3: SN Order Convergence")
    print("=" * 60)
    
    x_min, x_max = 0.0, 5.0
    n_cells = 50
    
    sigma_t = 1.0
    sigma_s = 0.5
    source = np.ones(n_cells)
    
    sn_orders = [2, 4, 8, 16]
    colors = ['b', 'g', 'r', 'm']
    
    plt.figure(figsize=(10, 6))
    
    for sn_order, color in zip(sn_orders, colors):
        mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                      sigma_t, sigma_s, source, tol=1e-8)
        x = mesh.x_centers
        plt.plot(x, phi, f'{color}-', linewidth=2, label=f'SN{sn_order}')
        print(f"SN{sn_order}: max flux = {np.max(phi):.6f}")
    
    plt.xlabel('x (cm)')
    plt.ylabel('Scalar Flux')
    plt.title('SN Order Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test3_sn_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to test3_sn_convergence.png")


def test_two_material_problem():
    print("\n" + "=" * 60)
    print("Test 4: Two-Material Problem")
    print("=" * 60)
    
    x_min, x_max = 0.0, 10.0
    n_cells = 100
    sn_order = 8
    
    mesh = Mesh(x_min, x_max, n_cells)
    quad = SNQuadrature(sn_order)
    
    materials = []
    source = np.zeros(n_cells)
    
    for i in range(n_cells):
        if mesh.x_centers[i] < 5.0:
            materials.append(Material(sigma_t=2.0, sigma_s=1.5))
        else:
            materials.append(Material(sigma_t=0.5, sigma_s=0.1))
        source[i] = 1.0 if mesh.x_centers[i] < 5.0 else 0.0
    
    solver = SNSolver(mesh, quad, materials, bc_left='vacuum', bc_right='vacuum')
    phi, psi = solver.solve(source)
    
    x = mesh.x_centers
    
    print(f"Max scalar flux: {np.max(phi):.6f}")
    print(f"Min scalar flux: {np.min(phi):.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, phi, 'b-', linewidth=2)
    plt.axvline(x=5.0, color='k', linestyle='--', alpha=0.5, label='Material Interface')
    plt.xlabel('x (cm)')
    plt.ylabel('Scalar Flux')
    plt.title('Two-Material Problem')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test4_two_material.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to test4_two_material.png")
    return phi


def test_reflective_bc():
    print("\n" + "=" * 60)
    print("Test 5: Reflective Boundary Conditions")
    print("=" * 60)
    
    x_min, x_max = 0.0, 5.0
    n_cells = 50
    sn_order = 8
    
    sigma_t = 1.0
    sigma_s = 0.5
    source = np.ones(n_cells)
    
    print("Vacuum BCs:")
    mesh_vac, phi_vac, _ = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                        sigma_t, sigma_s, source,
                                        bc_left='vacuum', bc_right='vacuum')
    
    print("\nReflective BCs:")
    mesh_ref, phi_ref, _ = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                        sigma_t, sigma_s, source,
                                        bc_left='reflective', bc_right='reflective')
    
    x = mesh_vac.x_centers
    
    print(f"\nVacuum BC - max flux: {np.max(phi_vac):.6f}, mean flux: {np.mean(phi_vac):.6f}")
    print(f"Reflective BC - max flux: {np.max(phi_ref):.6f}, mean flux: {np.mean(phi_ref):.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, phi_vac, 'b-', linewidth=2, label='Vacuum BCs')
    plt.plot(x, phi_ref, 'r--', linewidth=2, label='Reflective BCs')
    plt.xlabel('x (cm)')
    plt.ylabel('Scalar Flux')
    plt.title('Boundary Condition Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test5_bc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to test5_bc_comparison.png")


def test_angular_flux():
    print("\n" + "=" * 60)
    print("Test 6: Angular Flux Visualization")
    print("=" * 60)
    
    x_min, x_max = 0.0, 5.0
    n_cells = 50
    sn_order = 8
    
    sigma_t = 1.0
    sigma_s = 0.3
    source = np.ones(n_cells)
    
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                  sigma_t, sigma_s, source)
    
    x_edges = mesh.x_edges
    mus = SNQuadrature(sn_order).mus
    
    plt.figure(figsize=(12, 8))
    
    for n in range(len(mus)):
        if mus[n] > 0:
            plt.plot(x_edges, psi[n, :], 'b-', linewidth=1.5, alpha=0.7)
        else:
            plt.plot(x_edges, psi[n, :], 'r--', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('x (cm)')
    plt.ylabel('Angular Flux')
    plt.title('Angular Flux Distribution (blue: μ>0, red: μ<0)')
    plt.grid(True, alpha=0.3)
    plt.savefig('test6_angular_flux.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to test6_angular_flux.png")


def main():
    print("SN Solver Test Suite")
    print("Testing 1D Discrete Ordinates Method for Neutron Transport")
    
    test_source_problem()
    test_scattering_problem()
    test_sn_convergence()
    test_two_material_problem()
    test_reflective_bc()
    test_angular_flux()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
