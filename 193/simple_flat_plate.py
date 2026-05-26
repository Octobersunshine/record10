import numpy as np


def vlm_flat_plate(alpha_deg, n_panels=40, V_inf=10.0):
    alpha = np.radians(alpha_deg)
    
    beta = np.linspace(0, np.pi, n_panels+1)
    x = (1 - np.cos(beta)) / 2
    
    x_v = 0.25 * x[:-1] + 0.75 * x[1:]
    y_v = np.zeros(n_panels)
    
    x_c = 0.75 * x[:-1] + 0.25 * x[1:]
    y_c = np.zeros(n_panels)
    
    nx = np.zeros(n_panels)
    ny = np.ones(n_panels)
    
    A = np.zeros((n_panels, n_panels))
    for i in range(n_panels):
        for j in range(n_panels):
            dx = x_c[i] - x_v[j]
            dy = y_c[i] - y_v[j]
            r2 = dx**2 + dy**2 + 1e-10
            
            u = -dy / (2 * np.pi * r2)
            v = dx / (2 * np.pi * r2)
            
            A[i, j] = u * nx[i] + v * ny[i]
    
    RHS = -(V_inf * np.cos(alpha) * nx + V_inf * np.sin(alpha) * ny)
    
    gamma = np.linalg.solve(A, RHS)
    
    total_gamma = np.sum(gamma)
    Cl = 2 * total_gamma / V_inf
    
    Cl_theory = 2 * np.pi * alpha
    
    print(f"Alpha: {alpha_deg} deg")
    print(f"  Theoretical Cl: {Cl_theory:.4f}")
    print(f"  Computed Cl: {Cl:.4f}")
    print(f"  Error: {abs(Cl - Cl_theory)/Cl_theory*100:.2f}%")
    
    return Cl, gamma, x_v


if __name__ == "__main__":
    print("Testing Flat Plate VLM")
    print("=" * 50)
    
    for alpha in [1, 3, 5, 8, 10]:
        vlm_flat_plate(alpha, n_panels=80)
        print()
