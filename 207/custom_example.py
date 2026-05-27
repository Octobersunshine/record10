import numpy as np
from darcy_flow_solver import DarcyFlowSolver


def custom_problem():
    print("Custom Darcy Flow Problem")
    print("=" * 50)
    
    nx, ny = 40, 30
    dx, dy = 2.0, 2.0
    
    solver = DarcyFlowSolver(nx, ny, dx, dy)
    
    K = np.ones((ny, nx)) * 1e-5
    K[10:20, 5:15] = 1e-7
    solver.set_hydraulic_conductivity(K)
    
    for j in range(ny):
        solver.set_dirichlet_bc(0, j, 12.0)
        solver.set_dirichlet_bc(nx-1, j, 6.0)
    
    for i in range(nx):
        solver.set_neumann_bc(i, 0, 0.0)
        solver.set_neumann_bc(i, ny-1, 0.0)
    
    h = solver.solve()
    u, v = solver.compute_velocity()
    
    print(f"Domain: {nx*dx}m x {ny*dy}m")
    print(f"Grid: {nx} x {ny}")
    print(f"Head range: {h.min():.4f} m to {h.max():.4f} m")
    
    speed = np.sqrt(u**2 + v**2)
    print(f"Max velocity: {speed.max():.6e} m/s")
    print(f"Average velocity: {speed.mean():.6e} m/s")
    
    total_flow_left = np.sum(u[:, 0]) * dy
    total_flow_right = np.sum(u[:, -1]) * dy
    print(f"Total inflow (left): {total_flow_left:.6e} m³/s")
    print(f"Total outflow (right): {total_flow_right:.6e} m³/s")
    
    mid_h = h[ny//2, :]
    print(f"\nHead at midline (y={ny//2*dy}m):")
    for i in range(0, nx, 5):
        print(f"  x={i*dx:5.1f}m: h={mid_h[i]:.4f}m")
    
    return solver


if __name__ == "__main__":
    solver = custom_problem()
    print("\nSolution complete!")
