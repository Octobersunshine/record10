import numpy as np
import matplotlib.pyplot as plt


class SimpleVLM:
    def __init__(self):
        pass
    
    def vortex_2d(self, xv, yv, x, y, gamma=1.0):
        dx = x - xv
        dy = y - yv
        r2 = dx**2 + dy**2 + 1e-10
        u = -gamma * dy / (2 * np.pi * r2)
        v = gamma * dx / (2 * np.pi * r2)
        return u, v
    
    def solve_static(self, alpha_deg=5.0):
        alpha = np.radians(alpha_deg)
        V_inf = 10.0
        
        n = 40
        x = np.cos(np.linspace(np.pi, 0, n+1)) / 2 + 0.5
        
        x_c = 0.75 * x[:-1] + 0.25 * x[1:]
        y_c = np.zeros(n)
        
        x_v = 0.25 * x[:-1] + 0.75 * x[1:]
        y_v = np.zeros(n)
        
        dx = x[1:] - x[:-1]
        nx = np.zeros(n)
        ny = np.ones(n)
        
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                u, v = self.vortex_2d(x_v[j], y_v[j], x_c[i], y_c[i])
                A[i, j] = u * nx[i] + v * ny[i]
        
        RHS = -(V_inf * np.cos(alpha) * nx + V_inf * np.sin(alpha) * ny)
        
        gamma = np.linalg.solve(A, RHS)
        
        total_gamma = np.sum(gamma)
        Cl = total_gamma / (0.5 * V_inf)
        
        print(f"Alpha: {alpha_deg} deg")
        print(f"Theoretical Cl: {2*np.pi*alpha:.4f}")
        print(f"Computed Cl: {Cl:.4f}")
        print(f"Total gamma: {total_gamma:.4f}")
        
        return gamma, x, x_c, x_v


if __name__ == "__main__":
    vlm = SimpleVLM()
    gamma, x, x_c, x_v = vlm.solve_static(alpha_deg=5.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_v, gamma, 'bo-', label='Vortex distribution')
    plt.xlabel('x/c')
    plt.ylabel('Circulation')
    plt.title('Vortex Distribution (Flat Plate)')
    plt.grid(True)
    plt.legend()
    plt.savefig('debug_vortex_dist.png')
    plt.close()
    print("Saved debug_vortex_dist.png")
