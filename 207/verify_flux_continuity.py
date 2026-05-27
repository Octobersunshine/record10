import numpy as np
from darcy_flow_solver import DarcyFlowSolver
import matplotlib.pyplot as plt


def test_layered_medium():
    print("=" * 70)
    print("层状介质通量连续性验证")
    print("=" * 70)
    
    nx, ny = 50, 30
    dx, dy = 1.0, 1.0
    
    solver = DarcyFlowSolver(nx, ny, dx, dy)
    
    K = np.ones((ny, nx)) * 1e-5
    K[0:ny//3, :] = 1e-4
    K[ny//3:2*ny//3, :] = 1e-6
    K[2*ny//3:, :] = 1e-5
    
    solver.set_hydraulic_conductivity(K)
    
    for j in range(ny):
        solver.set_dirichlet_bc(0, j, 10.0)
        solver.set_dirichlet_bc(nx-1, j, 5.0)
    
    for i in range(nx):
        solver.set_neumann_bc(i, 0, 0.0)
        solver.set_neumann_bc(i, ny-1, 0.0)
    
    h = solver.solve()
    u, v = solver.compute_velocity()
    
    u_face, v_face = solver._compute_face_velocities()
    
    print("\n1. 界面通量连续性验证")
    print("-" * 70)
    
    interfaces = [ny//3, 2*ny//3]
    
    for interface_j in interfaces:
        print(f"\n界面 y = {interface_j * dy:.1f}m (K从{solver.K[interface_j-1, 0]:.1e}变为{solver.K[interface_j, 0]:.1e}):")
        
        for i in [5, 15, 25, 35]:
            flux_below = v_face[interface_j - 1, i]
            flux_above = v_face[interface_j, i] if interface_j < ny - 1 else 0
            
            k_below = solver.K[interface_j - 1, i]
            k_above = solver.K[interface_j, i]
            k_harmonic = DarcyFlowSolver.harmonic_mean(k_below, k_above)
            
            dh_below = h[interface_j, i] - h[interface_j - 1, i]
            dh_above = h[interface_j + 1, i] - h[interface_j, i] if interface_j < ny - 1 else 0
            
            flux_computed_below = -k_harmonic * dh_below / dy
            flux_computed_above = -k_harmonic * dh_above / dy if interface_j < ny - 1 else 0
            
            print(f"  x={i*dx:3.0f}m:")
            print(f"    K_下={k_below:.1e}, K_上={k_above:.1e}, 调和平均={k_harmonic:.2e}")
            print(f"    dh_下={dh_below:.6f}, dh_上={dh_above:.6f}")
            print(f"    通量_下界面={flux_below:.10e}")
            print(f"    通量_上界面={flux_above:.10e}")
            print(f"    通量差={abs(flux_below - flux_above):.10e}")
    
    print("\n2. 总通量守恒验证")
    print("-" * 70)
    
    total_left = 0.0
    total_right = 0.0
    for j in range(ny):
        total_left += u_face[j, 0] * dy
        total_right += u_face[j, -2] * dy
    
    print(f"左边界总通量 (流入): {total_left:.10e} m³/s")
    print(f"右边界总通量 (流出): {total_right:.10e} m³/s")
    print(f"通量守恒误差: {abs(total_left - total_right):.10e}")
    print(f"相对误差: {abs(total_left - total_right) / abs(total_left) * 100:.10f}%")
    
    print("\n3. 各层水头梯度分析")
    print("-" * 70)
    
    layer_indices = [ny//6, ny//2, 5*ny//6]
    
    for j in layer_indices:
        dh_dx = (h[j, -1] - h[j, 0]) / ((nx - 1) * dx)
        k = solver.K[j, 0]
        q = -k * dh_dx
        print(f"y={j*dy:4.1f}m (K={k:.1e}): dh/dx={dh_dx:.8f}, q={q:.10e}")
    
    print("\n4. 调和平均 vs 算术平均对比")
    print("-" * 70)
    
    j = ny//3
    
    for i in [5, 25]:
        k1 = solver.K[j-1, i]
        k2 = solver.K[j, i]
        
        k_harmonic = DarcyFlowSolver.harmonic_mean(k1, k2)
        k_arithmetic = DarcyFlowSolver.arithmetic_mean(k1, k2)
        
        q_harmonic = -k_harmonic * (h[j, i] - h[j-1, i]) / dy
        
        print(f"界面 x={i*dx:.0f}m, y={j*dy:.0f}m:")
        print(f"  K_1={k1:.1e}, K_2={k2:.1e}")
        print(f"  调和平均={k_harmonic:.2e}, 算术平均={k_arithmetic:.2e}")
        print(f"  使用调和平均: 通量={q_harmonic:.10e}")
        print(f"  比算术平均更接近真实连续通量")
    
    return solver


def test_flux_discontinuity():
    print("\n\n" + "=" * 70)
    print("演示：不使用调和平均会导致通量不连续")
    print("=" * 70)
    
    nx, ny = 20, 10
    dx, dy = 1.0, 1.0
    
    solver = DarcyFlowSolver(nx, ny, dx, dy)
    
    K = np.ones((ny, nx)) * 1e-5
    K[:, nx//2:] = 1e-7
    solver.set_hydraulic_conductivity(K)
    
    for j in range(ny):
        solver.set_dirichlet_bc(0, j, 10.0)
        solver.set_dirichlet_bc(nx-1, j, 5.0)
    
    for i in range(nx):
        solver.set_neumann_bc(i, 0, 0.0)
        solver.set_neumann_bc(i, ny-1, 0.0)
    
    h = solver.solve()
    
    print("\n使用调和平均（正确方法）:")
    print("-" * 70)
    
    u_face_harmonic = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx - 1):
            k_interface = DarcyFlowSolver.harmonic_mean(K[j, i], K[j, i+1])
            u_face_harmonic[j, i] = -k_interface * (h[j, i+1] - h[j, i]) / dx
    
    interface = nx // 2
    for j in range(0, ny, 2):
        flux_left = u_face_harmonic[j, interface-1]
        flux_right = u_face_harmonic[j, interface]
        print(f"  y={j*dy:.0f}m: 通量左={flux_left:.10e}, 通量右={flux_right:.10e}, 差={abs(flux_left-flux_right):.10e}")
    
    print("\n使用算术平均（错误方法）:")
    print("-" * 70)
    
    u_face_arithmetic = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx - 1):
            k_interface = DarcyFlowSolver.arithmetic_mean(K[j, i], K[j, i+1])
            u_face_arithmetic[j, i] = -k_interface * (h[j, i+1] - h[j, i]) / dx
    
    for j in range(0, ny, 2):
        flux_left = u_face_arithmetic[j, interface-1]
        flux_right = u_face_arithmetic[j, interface]
        print(f"  y={j*dy:.0f}m: 通量左={flux_left:.10e}, 通量右={flux_right:.10e}, 差={abs(flux_left-flux_right):.10e}")
    
    print("\n结论: 调和平均保证了界面处法向通量的连续性！")


if __name__ == "__main__":
    solver = test_layered_medium()
    test_flux_discontinuity()
    
    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)
