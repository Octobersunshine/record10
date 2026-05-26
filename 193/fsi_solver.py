import numpy as np
from structural_model import StructuralDynamics
from improved_vlm import ImprovedVLM


class FSISolver:
    def __init__(self, config):
        self.config = config
        self.n_panels = config.get('n_panels', 80)
        self.n_struct = config.get('n_struct_points', 50)
        self.chord = config.get('chord', 1.0)
        
        self.vlm = ImprovedVLM(n_panels=self.n_panels)
        self.struct = StructuralDynamics(
            model_type=config.get('structural_model', 'spring'),
            n_points=self.n_struct,
            chord=self.chord,
            stiffness=config.get('stiffness', 1000.0),
            damping=config.get('damping', 10.0),
            mass_per_length=config.get('mass_per_length', 10.0)
        )
        
        self.w_history = []
        self.Cl_history = []
        self.Cd_history = []
        self.convergence_history = []
        
    def interpolate_forces_to_structure(self, x_vlm, forces_vlm, x_struct):
        forces_struct = np.interp(x_struct, x_vlm, forces_vlm)
        return forces_struct
        
    def interpolate_deformation_to_aero(self, x_struct, w_struct, x_vlm):
        w_vlm = np.interp(x_vlm, x_struct, w_struct)
        return w_vlm
        
    def compute_aero_forces(self, w_deformed, time, dt):
        V_inf = self.config.get('V_inf', 10.0)
        
        x_vlm = np.linspace(0, self.chord, self.n_panels + 1)
        x_vlm = (1 - np.cos(np.linspace(0, np.pi, self.n_panels + 1))) / 2
        
        y_c = w_deformed
        
        alpha = self.config.get('alpha_mean', 5.0)
        alpha_amp = self.config.get('alpha_amplitude', 0.0)
        pitch_freq = self.config.get('pitch_frequency', 0.0)
        alpha = np.radians(alpha + alpha_amp * np.sin(pitch_freq * time))
        
        x_c, y_c, x_v, y_v, nx, ny, length = self.vlm.get_panel_geometry(x_vlm, y_c)
        
        self.vlm.time_marching_step(
            x_vlm, y_c, x_c, y_c, x_v, y_v, nx, ny, 
            V_inf, alpha, dt, None
        )
        
        Cp = self.vlm.compute_pressure_coefficient_unsteady(
            x_c, y_c, x_v, y_v, V_inf, alpha, dt, None
        )
        
        rho = self.config.get('rho', 1.225)
        q = 0.5 * rho * V_inf**2
        
        forces = np.zeros(self.n_panels)
        for i in range(self.n_panels):
            forces[i] = q * Cp[i] * length[i] * ny[i]
        
        Cl = self.vlm.compute_lift_coefficient(V_inf)
        Cd = np.sum(q * Cp * length * nx) / q
        
        return x_c, forces, Cl, Cd
        
    def strong_coupling_step(self, time, dt, max_iter=50, tol=1e-6, 
                             relaxation=0.5, use_aitken=True):
        V_inf = self.config.get('V_inf', 10.0)
        rho = self.config.get('rho', 1.225)
        
        x_vlm = np.linspace(0, self.chord, self.n_panels + 1)
        x_vlm = (1 - np.cos(np.linspace(0, np.pi, self.n_panels + 1))) / 2
        
        x_struct = self.struct.x
        
        w_prev = self.struct.model.w.copy()
        
        w_current = w_prev.copy()
        
        residual_prev = None
        relaxation_factor = relaxation
        
        iteration_converged = False
        
        for iteration in range(max_iter):
            w_vlm = self.interpolate_deformation_to_aero(x_struct, w_current, x_vlm)
            
            x_c, forces_vlm, Cl, Cd = self.compute_aero_forces(w_vlm, time, dt)
            
            forces_struct = self.interpolate_forces_to_structure(x_c, forces_vlm, x_struct)
            
            x_struct_new, w_struct_new = self.struct.compute_deformation(forces_struct)
            
            residual = np.linalg.norm(w_struct_new - w_current) / (np.linalg.norm(w_struct_new) + 1e-10)
            
            if use_aitken and iteration > 0 and residual_prev is not None:
                relaxation_factor = relaxation_factor * (1 - residual / residual_prev)
                relaxation_factor = np.clip(relaxation_factor, 0.1, 0.9)
            
            w_current = w_current + relaxation_factor * (w_struct_new - w_current)
            
            residual_prev = residual
            
            if residual < tol:
                iteration_converged = True
                break
        
        self.struct.model.w = w_current.copy()
        
        return {
            'converged': iteration_converged,
            'iterations': iteration + 1,
            'final_residual': residual,
            'Cl': Cl,
            'Cd': Cd,
            'w': w_current.copy(),
            'forces': forces_struct.copy(),
            'relaxation': relaxation_factor
        }
        
    def solve(self):
        t_final = self.config.get('t_final', 1.0)
        dt = self.config.get('dt', 0.01)
        
        n_steps = int(t_final / dt)
        times = np.linspace(0, t_final, n_steps)
        
        self.vlm.reset_wake()
        self.struct.reset()
        
        results = {
            'times': [],
            'Cl': [],
            'Cd': [],
            'w_tip': [],
            'w_max': [],
            'iterations': [],
            'converged': [],
            'residuals': []
        }
        
        print("\n" + "=" * 70)
        print("Fluid-Structure Interaction - Strong Coupling (Staggered Method)")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Structural model: {self.config.get('structural_model', 'spring')}")
        print(f"  Stiffness: {self.config.get('stiffness', 1000.0)} N/m")
        print(f"  Damping: {self.config.get('damping', 10.0)} Ns/m")
        print(f"  Max iterations per step: {self.config.get('max_fsi_iter', 50)}")
        print(f"  Convergence tolerance: {self.config.get('fsi_tol', 1e-6)}")
        print(f"  Time steps: {n_steps}")
        print("=" * 70)
        
        for step, t in enumerate(times):
            fsi_result = self.strong_coupling_step(
                t, dt,
                max_iter=self.config.get('max_fsi_iter', 50),
                tol=self.config.get('fsi_tol', 1e-6),
                relaxation=self.config.get('relaxation', 0.5),
                use_aitken=self.config.get('use_aitken', True)
            )
            
            self.struct.time_step(fsi_result['forces'], dt)
            
            results['times'].append(t)
            results['Cl'].append(fsi_result['Cl'])
            results['Cd'].append(fsi_result['Cd'])
            results['w_tip'].append(fsi_result['w'][-1])
            results['w_max'].append(np.max(np.abs(fsi_result['w'])))
            results['iterations'].append(fsi_result['iterations'])
            results['converged'].append(fsi_result['converged'])
            results['residuals'].append(fsi_result['final_residual'])
            
            if step % 20 == 0 or step == n_steps - 1:
                status = "✓" if fsi_result['converged'] else "✗"
                print(f"  Step {step:4d}/{n_steps} | t={t:.4f}s | "
                      f"Cl={fsi_result['Cl']:.4f} | w_tip={fsi_result['w'][-1]*1000:.3f}mm | "
                      f"Iter={fsi_result['iterations']:2d} {status} | "
                      f"Res={fsi_result['final_residual']:.2e}")
        
        print("\nFSI Solution Complete!")
        print(f"  Converged steps: {sum(results['converged'])}/{n_steps}")
        print(f"  Mean iterations: {np.mean(results['iterations']):.1f}")
        print(f"  Max deflection: {np.max(results['w_max'])*1000:.3f} mm")
        
        return results
        
    def plot_results(self, results, save_suffix='fsi'):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Fluid-Structure Interaction Results', fontsize=14, fontweight='bold')
        
        times = results['times']
        
        axes[0, 0].plot(times, results['Cl'], 'b-', linewidth=2)
        axes[0, 0].set_ylabel('$C_l$')
        axes[0, 0].set_title('Lift Coefficient')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(times, results['Cd'], 'r-', linewidth=2)
        axes[0, 1].set_ylabel('$C_d$')
        axes[0, 1].set_title('Drag Coefficient')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(times, np.array(results['w_tip']) * 1000, 'g-', linewidth=2)
        axes[1, 0].set_ylabel('Tip Deflection (mm)')
        axes[1, 0].set_title('Wing Tip Deflection')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(times, results['iterations'], 'k-', linewidth=1, marker='o', markersize=3)
        axes[1, 1].set_ylabel('Iterations')
        axes[1, 1].set_title('FSI Iterations per Step')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 0].semilogy(times, np.maximum(results['residuals'], 1e-15), 'm-', linewidth=2)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Residual')
        axes[2, 0].set_title('Convergence History')
        axes[2, 0].grid(True, alpha=0.3)
        
        converged = np.array(results['converged']).astype(int)
        axes[2, 1].plot(times, converged, 'ko', markersize=2)
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Converged')
        axes[2, 1].set_title('Convergence Status (1=Converged, 0=Not)')
        axes[2, 1].set_ylim(-0.1, 1.1)
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'fsi_results_{save_suffix}.png', dpi=150)
        plt.close()
        print(f"\nPlot saved: fsi_results_{save_suffix}.png")
        
    def plot_deformed_shape(self, results, save_suffix='fsi'):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = self.struct.x
        
        w_final = self.struct.model.w
        ax.plot(x, w_final * 1000, 'b-', linewidth=2, label='Final Shape')
        ax.plot(x, np.zeros_like(x), 'k--', linewidth=1, label='Original')
        
        ax.fill_between(x, 0, w_final * 1000, alpha=0.3, color='blue')
        
        ax.set_xlabel('Chordwise Position (m)')
        ax.set_ylabel('Deflection (mm)')
        ax.set_title('Final Deformed Shape')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'deformed_shape_{save_suffix}.png', dpi=150)
        plt.close()
        print(f"Plot saved: deformed_shape_{save_suffix}.png")


def compare_fsi_vs_rigid():
    print("\n" + "#" * 70)
    print("#  Comparison: FSI vs Rigid Airfoil")
    print("#" * 70)
    
    config_rigid = {
        'n_panels': 80,
        'V_inf': 10.0,
        'chord': 1.0,
        'rho': 1.225,
        't_final': 0.5,
        'dt': 0.01,
        'alpha_mean': 5.0,
        'alpha_amplitude': 2.0,
        'pitch_frequency': 2 * np.pi,
        'stiffness': 1e10,
        'damping': 1000.0,
        'max_fsi_iter': 50,
        'fsi_tol': 1e-6
    }
    
    config_flexible = {
        'n_panels': 80,
        'V_inf': 10.0,
        'chord': 1.0,
        'rho': 1.225,
        't_final': 0.5,
        'dt': 0.01,
        'alpha_mean': 5.0,
        'alpha_amplitude': 2.0,
        'pitch_frequency': 2 * np.pi,
        'stiffness': 500.0,
        'damping': 50.0,
        'max_fsi_iter': 50,
        'fsi_tol': 1e-6
    }
    
    print("\nSolving Rigid Airfoil...")
    solver_rigid = FSISolver(config_rigid)
    results_rigid = solver_rigid.solve()
    solver_rigid.plot_results(results_rigid, 'rigid')
    
    print("\nSolving Flexible Airfoil...")
    solver_flexible = FSISolver(config_flexible)
    results_flexible = solver_flexible.solve()
    solver_flexible.plot_results(results_flexible, 'flexible')
    solver_flexible.plot_deformed_shape(results_flexible, 'flexible')
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].plot(results_rigid['times'], results_rigid['Cl'], 'b-', 
                linewidth=2, label='Rigid')
    axes[0].plot(results_flexible['times'], results_flexible['Cl'], 'r--', 
                linewidth=2, label='Flexible')
    axes[0].set_ylabel('$C_l$')
    axes[0].set_title('Lift Coefficient: Rigid vs Flexible')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(results_flexible['times'], 
                np.array(results_flexible['w_tip']) * 1000, 'g-', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Tip Deflection (mm)')
    axes[1].set_title('Flexible Wing Tip Deflection')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fsi_comparison.png', dpi=150)
    plt.close()
    print("\nComparison plot saved: fsi_comparison.png")
    
    print("\n" + "=" * 70)
    print("Comparison Summary:")
    print(f"  Rigid - Mean Cl: {np.mean(results_rigid['Cl']):.4f}")
    print(f"  Flexible - Mean Cl: {np.mean(results_flexible['Cl']):.4f}")
    print(f"  Max tip deflection: {np.max(results_flexible['w_tip'])*1000:.3f} mm")
    print("=" * 70)
    
    return results_rigid, results_flexible


if __name__ == "__main__":
    compare_fsi_vs_rigid()
