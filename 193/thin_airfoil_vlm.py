import numpy as np
import matplotlib.pyplot as plt


class ThinAirfoilVLM:
    def __init__(self, n_panels=80):
        self.n_panels = n_panels
        self.gamma = None
        self.wake_vortices = []
        self.wake_strengths = []
        
    def vortex_2d(self, xv, yv, x, y, gamma=1.0):
        dx = x - xv
        dy = y - yv
        r2 = dx**2 + dy**2 + 1e-10
        u = -gamma * dy / (2 * np.pi * r2)
        v = gamma * dx / (2 * np.pi * r2)
        return u, v
    
    def generate_camber_line(self, config, time=0.0):
        n = self.n_panels
        beta = np.linspace(0, np.pi, n+1)
        x = (1 - np.cos(beta)) / 2
        
        y_c = np.zeros_like(x)
        
        naca = config.get('naca', '0012')
        m = int(naca[0]) / 100.0
        p = int(naca[1]) / 10.0
        
        if m > 0 and p > 0:
            mask = x < p
            y_c[mask] = m / p**2 * (2 * p * x[mask] - x[mask]**2)
            y_c[~mask] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[~mask] - x[~mask]**2)
        
        if 'camber_amplitude' in config:
            camber_amp = config['camber_amplitude']
            camber_freq = config.get('camber_frequency', 2.0)
            y_c += camber_amp * np.sin(2 * np.pi * camber_freq * x) * np.sin(time)
        
        alpha_mean = config.get('alpha_mean', 0.0)
        alpha_amp = config.get('alpha_amplitude', 0.0)
        pitch_freq = config.get('pitch_frequency', 0.0)
        
        alpha = np.radians(alpha_mean + alpha_amp * np.sin(pitch_freq * time))
        
        return x, y_c, alpha
    
    def get_panel_geometry(self, x, y):
        n = len(x) - 1
        
        x_c = np.zeros(n)
        y_c = np.zeros(n)
        x_v = np.zeros(n)
        y_v = np.zeros(n)
        nx = np.zeros(n)
        ny = np.zeros(n)
        length = np.zeros(n)
        
        for i in range(n):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            length[i] = np.sqrt(dx**2 + dy**2)
            
            x_c[i] = 0.75 * x[i] + 0.25 * x[i+1]
            y_c[i] = 0.75 * y[i] + 0.25 * y[i+1]
            x_v[i] = 0.25 * x[i] + 0.75 * x[i+1]
            y_v[i] = 0.25 * y[i] + 0.75 * y[i+1]
            
            nx[i] = -dy / length[i]
            ny[i] = dx / length[i]
        
        return x_c, y_c, x_v, y_v, nx, ny, length
    
    def build_influence_matrix(self, x_c, y_c, x_v, y_v, nx, ny):
        n = len(x_c)
        A = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                u, v = self.vortex_2d(x_v[j], y_v[j], x_c[i], y_c[i])
                A[i, j] = u * nx[i] + v * ny[i]
        
        return A
    
    def build_wake_influence(self, x_c, y_c, nx, ny):
        n = len(x_c)
        wake_influence = np.zeros(n)
        
        for wake_idx, (xw, yw, strength) in enumerate(zip(self.wake_vortices[::2], 
                                                          self.wake_vortices[1::2], 
                                                          self.wake_strengths)):
            for i in range(n):
                u, v = self.vortex_2d(xw, yw, x_c[i], y_c[i], strength)
                wake_influence[i] += u * nx[i] + v * ny[i]
        
        return wake_influence
    
    def solve(self, x_c, y_c, x_v, y_v, nx, ny, V_inf, alpha, panel_velocities=None):
        n = len(x_c)
        
        A = self.build_influence_matrix(x_c, y_c, x_v, y_v, nx, ny)
        
        U_inf = V_inf * np.cos(alpha)
        W_inf = V_inf * np.sin(alpha)
        
        RHS = -(U_inf * nx + W_inf * ny)
        
        if panel_velocities is not None:
            RHS -= panel_velocities[:, 0] * nx + panel_velocities[:, 1] * ny
        
        wake_influence = self.build_wake_influence(x_c, y_c, nx, ny)
        RHS -= wake_influence
        
        self.gamma = np.linalg.solve(A, RHS)
        
        return self.gamma
    
    def compute_lift_coefficient(self, V_inf):
        if self.gamma is None:
            return 0.0
        total_gamma = np.sum(self.gamma)
        Cl = 2 * total_gamma / V_inf
        return Cl
    
    def compute_pressure_coefficient(self, x_c, y_c, x_v, y_v, V_inf, alpha):
        u, v = np.zeros_like(x_c), np.zeros_like(x_c)
        for j in range(len(self.gamma)):
            u_j, v_j = self.vortex_2d(x_v[j], y_v[j], x_c, y_c, self.gamma[j])
            u += u_j
            v += v_j
        
        U_inf = V_inf * np.cos(alpha)
        W_inf = V_inf * np.sin(alpha)
        
        V_mag = np.sqrt((u + U_inf)**2 + (v + W_inf)**2)
        Cp = 1.0 - (V_mag / V_inf)**2
        
        return Cp
    
    def shed_wake(self, x_te, y_te, dt, V_inf, alpha):
        if self.gamma is not None:
            gamma_te = self.gamma[-1]
            
            x_wake = x_te + V_inf * np.cos(alpha) * dt
            y_wake = y_te + V_inf * np.sin(alpha) * dt
            
            self.wake_vortices.extend([x_wake, y_wake])
            self.wake_strengths.append(-gamma_te)
    
    def convect_wake(self, dt, V_inf, alpha, convection_factor=1.0):
        if len(self.wake_vortices) == 0:
            return
        
        wake_array = np.array(self.wake_vortices).reshape(-1, 2)
        
        u_conv = V_inf * np.cos(alpha) * convection_factor
        v_conv = V_inf * np.sin(alpha) * convection_factor
        
        wake_array[:, 0] += u_conv * dt
        wake_array[:, 1] += v_conv * dt
        
        self.wake_vortices = wake_array.flatten().tolist()
    
    def reset_wake(self):
        self.wake_vortices = []
        self.wake_strengths = []
        self.gamma = None


def static_validation():
    print("Static Validation: Thin Airfoil Theory")
    print("=" * 50)
    
    solver = ThinAirfoilVLM(n_panels=80)
    
    config = {
        'naca': '0012',
        'alpha_mean': 5.0,
        'alpha_amplitude': 0.0,
        'pitch_frequency': 0.0
    }
    
    x, y, alpha = solver.generate_camber_line(config, time=0.0)
    x_c, y_c, x_v, y_v, nx, ny, length = solver.get_panel_geometry(x, y)
    
    V_inf = 10.0
    gamma = solver.solve(x_c, y_c, x_v, y_v, nx, ny, V_inf, alpha)
    
    Cl = solver.compute_lift_coefficient(V_inf)
    Cl_theory = 2 * np.pi * alpha
    
    print(f"Angle of Attack: {np.degrees(alpha):.2f} deg")
    print(f"Theoretical Cl: {Cl_theory:.4f}")
    print(f"Computed Cl: {Cl:.4f}")
    print(f"Relative Error: {abs(Cl - Cl_theory)/Cl_theory*100:.2f}%")
    
    return Cl, Cl_theory


def unsteady_simulation(config):
    print("\nUnsteady Simulation")
    print("=" * 50)
    
    V_inf = config.get('V_inf', 10.0)
    t_final = config.get('t_final', 1.0)
    dt = config.get('dt', 0.01)
    
    n_steps = int(t_final / dt)
    times = np.linspace(0, t_final, n_steps)
    
    solver = ThinAirfoilVLM(n_panels=config.get('n_panels', 80))
    solver.reset_wake()
    
    Cl_history = np.zeros(n_steps)
    Cd_history = np.zeros(n_steps)
    alpha_history = np.zeros(n_steps)
    
    x_prev, y_prev = None, None
    
    for step, t in enumerate(times):
        x, y, alpha = solver.generate_camber_line(config, t)
        x_c, y_c, x_v, y_v, nx, ny, length = solver.get_panel_geometry(x, y)
        
        panel_velocities = None
        if x_prev is not None:
            panel_velocities = np.zeros((len(x_c), 2))
            for i in range(len(x_c)):
                x_mid_prev = 0.5 * (x_prev[i] + x_prev[i+1])
                y_mid_prev = 0.5 * (y_prev[i] + y_prev[i+1])
                x_mid_curr = 0.5 * (x[i] + x[i+1])
                y_mid_curr = 0.5 * (y[i] + y[i+1])
                panel_velocities[i, 0] = (x_mid_curr - x_mid_prev) / dt
                panel_velocities[i, 1] = (y_mid_curr - y_mid_prev) / dt
        
        gamma = solver.solve(x_c, y_c, x_v, y_v, nx, ny, V_inf, alpha, panel_velocities)
        
        Cl = solver.compute_lift_coefficient(V_inf)
        
        rho = 1.225
        Cp = solver.compute_pressure_coefficient(x_c, y_c, x_v, y_v, V_inf, alpha)
        
        force_x = 0.0
        force_y = 0.0
        q = 0.5 * rho * V_inf**2
        for i in range(len(Cp)):
            force_x += q * Cp[i] * length[i] * nx[i]
            force_y += q * Cp[i] * length[i] * ny[i]
        
        qS = q * 1.0
        Cd = (force_x * np.cos(alpha) + force_y * np.sin(alpha)) / qS
        
        Cl_history[step] = Cl
        Cd_history[step] = Cd
        alpha_history[step] = np.degrees(alpha)
        
        solver.shed_wake(1.0, 0.0, dt, V_inf, alpha)
        solver.convect_wake(dt, V_inf, alpha, convection_factor=0.9)
        
        x_prev, y_prev = x.copy(), y.copy()
    
    results = {
        'times': times,
        'Cl': Cl_history,
        'Cd': Cd_history,
        'alpha': alpha_history,
        'wake_vortices': solver.wake_vortices,
        'wake_strengths': solver.wake_strengths
    }
    
    print(f"Mean Cl: {np.mean(Cl_history):.4f}")
    print(f"Peak Cl: {np.max(Cl_history):.4f}")
    print(f"Min Cl: {np.min(Cl_history):.4f}")
    print(f"Wake vortices: {len(solver.wake_strengths)}")
    
    return results


def plot_results(results, title, save_suffix):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    axes[0].plot(results['times'], results['Cl'], 'b-', linewidth=2)
    axes[0].set_ylabel('$C_l$', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Lift Coefficient', fontsize=11)
    
    axes[1].plot(results['alpha'], results['Cl'], 'r-', linewidth=2)
    axes[1].set_xlabel('Angle of Attack (deg)', fontsize=12)
    axes[1].set_ylabel('$C_l$', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('$C_l$ - $\\alpha$ Hysteresis', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'results_{save_suffix}.png', dpi=150)
    plt.close()
    print(f"Plot saved: results_{save_suffix}.png")


if __name__ == "__main__":
    static_validation()
    
    config_pitch = {
        'naca': '0012',
        'V_inf': 10.0,
        't_final': 1.0,
        'dt': 0.005,
        'alpha_mean': 5.0,
        'alpha_amplitude': 3.0,
        'pitch_frequency': 2 * np.pi,
        'n_panels': 80
    }
    results_pitch = unsteady_simulation(config_pitch)
    plot_results(results_pitch, "Dynamic Pitch Motion", "pitch")
    
    config_camber = {
        'naca': '0012',
        'V_inf': 10.0,
        't_final': 1.0,
        'dt': 0.005,
        'alpha_mean': 3.0,
        'alpha_amplitude': 0.0,
        'pitch_frequency': 0.0,
        'camber_amplitude': 0.015,
        'camber_frequency': 2.0,
        'n_panels': 80
    }
    results_camber = unsteady_simulation(config_camber)
    plot_results(results_camber, "Dynamic Camber Deformation", "camber")
    
    config_combined = {
        'naca': '0012',
        'V_inf': 10.0,
        't_final': 1.0,
        'dt': 0.005,
        'alpha_mean': 4.0,
        'alpha_amplitude': 2.0,
        'pitch_frequency': 2 * np.pi,
        'camber_amplitude': 0.01,
        'camber_frequency': 3.0,
        'n_panels': 80
    }
    results_combined = unsteady_simulation(config_combined)
    plot_results(results_combined, "Combined Pitch + Camber", "combined")
    
    print("\nAll simulations completed!")
