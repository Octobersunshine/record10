import numpy as np
from thin_airfoil_vlm import ThinAirfoilVLM


def test_static():
    solver = ThinAirfoilVLM(n_panels=80)
    
    config = {
        'naca': '0012',
        'alpha_mean': 5.0,
        'alpha_amplitude': 0.0,
        'pitch_frequency': 0.0
    }
    
    x, y, alpha = solver.generate_camber_line(config, time=0.0)
    
    print("x range:", x[0], x[-1])
    print("y range:", y[0], y[-1])
    print("alpha:", alpha)
    
    x_c, y_c, x_v, y_v, nx, ny, length = solver.get_panel_geometry(x, y)
    
    print("\nFirst 5 panels:")
    for i in range(5):
        print(f"  Panel {i}: x_c={x_c[i]:.4f}, x_v={x_v[i]:.4f}, nx={nx[i]:.4f}, ny={ny[i]:.4f}")
    
    V_inf = 10.0
    gamma = solver.solve(x_c, y_c, x_v, y_v, nx, ny, V_inf, alpha)
    
    print("\nFirst 5 gamma values:", gamma[:5])
    print("Last 5 gamma values:", gamma[-5:])
    print("Sum gamma:", np.sum(gamma))
    
    Cl = solver.compute_lift_coefficient(V_inf)
    Cl_theory = 2 * np.pi * alpha
    
    print(f"\nAlpha: {np.degrees(alpha):.2f} deg")
    print(f"Theoretical Cl: {Cl_theory:.4f}")
    print(f"Computed Cl: {Cl:.4f}")


if __name__ == "__main__":
    test_static()
