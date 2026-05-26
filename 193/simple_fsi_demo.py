import numpy as np
from structural_model import SimpleSpringModel, EulerBernoulliBeam


class SimpleFSI:
    def __init__(self, config):
        self.config = config
        self.n_panels = config.get('n_panels', 40)
        self.n_struct = config.get('n_struct_points', 20)
        self.chord = config.get('chord', 1.0)
        
        self.struct = SimpleSpringModel(
            n_points=self.n_struct,
            chord=self.chord,
            stiffness=config.get('stiffness', 1000.0),
            damping=config.get('damping', 10.0),
            mass_per_length=config.get('mass_per_length', 10.0)
        )
        
        self.w_history = []
        self.Cl_history = []
        self.Cd_history = []
        self.iterations_history = []
        
    def compute_aero_forces_simple(self, w_deformed, alpha_rad, V_inf=10.0):
        rho = 1.225
        q = 0.5 * rho * V_inf**2
        
        x_vlm = (1 - np.cos(np.linspace(0, np.pi, self.n_panels + 1))) / 2
        
        y_c = np.interp(x_vlm, self.struct.x, w_deformed)
        
        dy_dx = np.gradient(y_c, x_vlm)
        
        Cp = 2 * np.sin(alpha_rad + dy_dx)
        
        Cl = np.trapezoid(Cp, x_vlm)
        Cd = np.trapezoid(Cp * dy_dx, x_vlm)
        
        x_mid = (x_vlm[:-1] + x_vlm[1:]) / 2
        Cp_mid = 0.5 * (Cp[:-1] + Cp[1:])
        dx = x_vlm[1:] - x_vlm[:-1]
        
        forces_lift = q * Cp_mid * dx
        forces_drag = q * Cp_mid * dy_dx[:-1] * dx
        
        forces_struct = np.interp(self.struct.x, x_mid, forces_lift)
        
        return x_mid, forces_struct, Cl, Cd
        
    def solve_fsi(self, alpha_rad, V_inf=10.0, max_iter=100, tol=1e-6, 
                  relaxation=0.3, load_factor=0.0):
        self.struct.reset()
        
        w_current = np.zeros(self.n_struct)
        
        for iteration in range(max_iter):
            alpha_eff = alpha_rad * (1.0 + load_factor * 0.0)
            
            x_mid, forces_struct, Cl, Cd = self.compute_aero_forces_simple(
                w_current, alpha_rad, V_inf
            )
            
            forces_struct *= (1.0 + load_factor * 0.0)
            
            w_new = self.struct.compute_deflection(forces_struct)
            
            residual = np.linalg.norm(w_new - w_current) / (np.linalg.norm(w_new) + 1e-10)
            
            w_current = w_current + relaxation * (w_new - w_current)
            
            if residual < tol:
                return {
                    'converged': True,
                    'iterations': iteration + 1,
                    'residual': residual,
                    'Cl': Cl,
                    'Cd': Cd,
                    'w': w_current.copy(),
                    'forces': forces_struct.copy()
                }
        
        return {
            'converged': False,
            'iterations': iteration + 1,
            'residual': residual,
            'Cl': Cl,
            'Cd': Cd,
            'w': w_current.copy(),
            'forces': forces_struct.copy()
        }
        
    def time_marching(self, alpha_series, V_inf=10.0, dt=0.01, 
                      max_fsi_iter=50, tol=1e-6, relaxation=0.3):
        n_steps = len(alpha_series)
        times = np.arange(n_steps) * dt
        
        results = {
            'times': times,
            'Cl': [],
            'Cd': [],
            'w_tip': [],
            'w_max': [],
            'iterations': [],
            'converged': []
        }
        
        w_prev = np.zeros(self.n_struct)
        
        print("\n" + "=" * 70)
        print("Simple FSI Time Marching")
        print("=" * 70)
        
        for step, alpha in enumerate(alpha_series):
            fsi_result = self.solve_fsi(
                alpha, V_inf, max_iter=max_fsi_iter, 
                tol=tol, relaxation=relaxation
            )
            
            self.struct.w = fsi_result['w'].copy()
            
            results['Cl'].append(fsi_result['Cl'])
            results['Cd'].append(fsi_result['Cd'])
            results['w_tip'].append(fsi_result['w'][-1])
            results['w_max'].append(np.max(np.abs(fsi_result['w'])))
            results['iterations'].append(fsi_result['iterations'])
            results['converged'].append(fsi_result['converged'])
            
            if step % 5 == 0 or step == n_steps - 1:
                status = "✓" if fsi_result['converged'] else "✗"
                print(f"  Step {step:4d}/{n_steps} | t={times[step]:.4f}s | "
                      f"α={np.degrees(alpha):.2f}° | Cl={fsi_result['Cl']:.4f} | "
                      f"w_tip={fsi_result['w'][-1]*1000:.3f}mm | "
                      f"Iter={fsi_result['iterations']:2d} {status}")
        
        print("\nFSI Time Marching Complete!")
        
        return results


def test_simple_fsi():
    print("\n" + "#" * 70)
    print("#  Simple FSI Demo - Flexible Wing Under Dynamic Load")
    print("#" * 70)
    
    config = {
        'n_panels': 40,
        'n_struct_points': 20,
        'chord': 1.0,
        'stiffness': 1000.0,
        'damping': 10.0,
        'mass_per_length': 10.0
    }
    
    fsi = SimpleFSI(config)
    
    t_final = 0.5
    dt = 0.02
    n_steps = int(t_final / dt)
    times = np.arange(n_steps) * dt
    
    alpha_mean = np.radians(3.0)
    alpha_amp = np.radians(2.0)
    freq = 2 * np.pi
    alpha_series = alpha_mean + alpha_amp * np.sin(freq * times)
    
    results = fsi.time_marching(
        alpha_series, V_inf=10.0, dt=dt,
        max_fsi_iter=50, tol=1e-6, relaxation=0.3
    )
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Final time: {results['times'][-1]:.4f} s")
    print(f"  Mean Cl: {np.mean(results['Cl']):.4f}")
    print(f"  Peak Cl: {np.max(results['Cl']):.4f}")
    print(f"  Max tip deflection: {np.max(results['w_tip'])*1000:.3f} mm")
    print(f"  Mean FSI iterations: {np.mean(results['iterations']):.1f}")
    print(f"  Converged steps: {sum(results['converged'])}/{len(results['converged'])}")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    axes[0].plot(results['times'], results['Cl'], 'b-', linewidth=2)
    axes[0].set_ylabel('$C_l$')
    axes[0].set_title('Lift Coefficient')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(results['times'], np.array(results['w_tip']) * 1000, 'g-', linewidth=2)
    axes[1].set_ylabel('Tip Deflection (mm)')
    axes[1].set_title('Wing Tip Deflection')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(results['times'], results['iterations'], 'k-', linewidth=1, marker='o', markersize=3)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('FSI Iterations')
    axes[2].set_title('FSI Iterations per Time Step')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_fsi_results.png', dpi=150)
    plt.close()
    print("\nPlot saved: simple_fsi_results.png")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fsi.struct.x, fsi.struct.w * 1000, 'b-', linewidth=2)
    ax.set_xlabel('Chordwise Position (m)')
    ax.set_ylabel('Deflection (mm)')
    ax.set_title('Final Deformed Shape')
    ax.grid(True, alpha=0.3)
    ax.fill_between(fsi.struct.x, 0, fsi.struct.w * 1000, alpha=0.2, color='blue')
    plt.tight_layout()
    plt.savefig('simple_fsi_deformed.png', dpi=150)
    plt.close()
    print("Plot saved: simple_fsi_deformed.png")
    
    return results


if __name__ == "__main__":
    test_simple_fsi()
