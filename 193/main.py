import numpy as np
import matplotlib.pyplot as plt
from unsteady_solver import UnsteadySolver


def example_pitch_motion():
    print("=" * 60)
    print("Example 1: Dynamic Pitch Motion")
    print("=" * 60)
    
    config = {
        'naca': '0012',
        'V_inf': 10.0,
        'chord': 1.0,
        'rho': 1.225,
        't_final': 1.0,
        'dt': 0.005,
        'alpha_mean': 5.0,
        'alpha_amplitude': 3.0,
        'pitch_frequency': 2 * np.pi,
        'pitch_pivot': 0.25
    }
    
    solver = UnsteadySolver(n_panels=80, n_points=50)
    results = solver.solve_unsteady(config)
    
    print_results_summary(results, config)
    plot_results(results, "Dynamic Pitch Motion", save_suffix="pitch")
    
    return results


def example_camber_deformation():
    print("\n" + "=" * 60)
    print("Example 2: Dynamic Camber Deformation")
    print("=" * 60)
    
    config = {
        'naca': '0012',
        'V_inf': 10.0,
        'chord': 1.0,
        'rho': 1.225,
        't_final': 1.0,
        'dt': 0.005,
        'alpha_mean': 3.0,
        'alpha_amplitude': 0.0,
        'pitch_frequency': 0.0,
        'camber_amplitude': 0.015,
        'camber_frequency': 2.0,
        'pitch_pivot': 0.25
    }
    
    solver = UnsteadySolver(n_panels=80, n_points=50)
    results = solver.solve_unsteady(config)
    
    print_results_summary(results, config)
    plot_results(results, "Dynamic Camber Deformation", save_suffix="camber")
    
    return results


def example_combined_motion():
    print("\n" + "=" * 60)
    print("Example 3: Combined Pitch + Camber Deformation")
    print("=" * 60)
    
    config = {
        'naca': '0012',
        'V_inf': 10.0,
        'chord': 1.0,
        'rho': 1.225,
        't_final': 1.0,
        'dt': 0.005,
        'alpha_mean': 4.0,
        'alpha_amplitude': 2.0,
        'pitch_frequency': 2 * np.pi,
        'camber_amplitude': 0.01,
        'camber_frequency': 3.0,
        'pitch_pivot': 0.25
    }
    
    solver = UnsteadySolver(n_panels=80, n_points=50)
    results = solver.solve_unsteady(config)
    
    print_results_summary(results, config)
    plot_results(results, "Combined Pitch + Camber Motion", save_suffix="combined")
    
    return results


def print_results_summary(results, config):
    print(f"\nConfiguration:")
    print(f"  Airfoil: NACA {config.get('naca', '0012')}")
    print(f"  V_inf: {config.get('V_inf', 10.0)} m/s")
    print(f"  Chord: {config.get('chord', 1.0)} m")
    
    if config.get('alpha_amplitude', 0) > 0:
        k = config.get('pitch_frequency', 0) * config.get('chord', 1.0) / (2 * config.get('V_inf', 10.0))
        print(f"  Reduced frequency k: {k:.4f}")
    
    print(f"\nResults Summary:")
    print(f"  Time steps: {len(results['times'])}")
    print(f"  Mean Cl: {np.mean(results['Cl']):.4f}")
    print(f"  Peak Cl: {np.max(results['Cl']):.4f}")
    print(f"  Min Cl: {np.min(results['Cl']):.4f}")
    print(f"  Mean Cd: {np.mean(results['Cd']):.4f}")
    print(f"  Mean Cm: {np.mean(results['Cm']):.4f}")
    print(f"  Wake vortices: {len(results['wake_strengths'])}")


def plot_results(results, title, save_suffix=""):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    times = results['times']
    
    axes[0].plot(times, results['Cl'], 'b-', linewidth=2)
    axes[0].set_ylabel('Lift Coefficient $C_l$', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Lift Coefficient vs Time', fontsize=11)
    
    axes[1].plot(times, results['Cd'], 'r-', linewidth=2)
    axes[1].set_ylabel('Drag Coefficient $C_d$', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Drag Coefficient vs Time', fontsize=11)
    
    axes[2].plot(times, results['Cm'], 'g-', linewidth=2)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Moment Coefficient $C_m$', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Moment Coefficient vs Time', fontsize=11)
    
    plt.tight_layout()
    
    if save_suffix:
        plt.savefig(f'aerodynamic_forces_{save_suffix}.png', dpi=150, bbox_inches='tight')
    
    plt.close()
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(results['alpha'], results['Cl'], 'b-', linewidth=2)
    ax2.set_xlabel('Angle of Attack (deg)', fontsize=12)
    ax2.set_ylabel('Lift Coefficient $C_l$', fontsize=12)
    ax2.set_title('$C_l$ - $\\alpha$ Hysteresis Loop', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    if save_suffix:
        plt.savefig(f'hysteresis_{save_suffix}.png', dpi=150, bbox_inches='tight')
    
    plt.close()
    
    print(f"  Plots saved: aerodynamic_forces_{save_suffix}.png, hysteresis_{save_suffix}.png")


def plot_wake(results, config, save_suffix="wake"):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    wake_x = results['wake_vortices'][::2]
    wake_y = results['wake_vortices'][1::2]
    strengths = results['wake_strengths']
    
    if len(wake_x) > 0:
        strength_norm = np.abs(strengths) / np.max(np.abs(strengths))
        scatter = ax.scatter(wake_x, wake_y, c=strengths, s=30 * strength_norm + 5, 
                            cmap='coolwarm', alpha=0.7, edgecolors='k', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='Vortex Strength')
    
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel('y/c', fontsize=12)
    ax.set_title('Wake Vortex Distribution', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'wake_vortices_{save_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Wake plot saved: wake_vortices_{save_suffix}.png")


def run_static_validation():
    print("\n" + "=" * 60)
    print("Static Validation: Thin Airfoil Theory Comparison")
    print("=" * 60)
    
    alpha_deg = 5.0
    alpha_rad = np.radians(alpha_deg)
    
    config = {
        'naca': '0012',
        'V_inf': 10.0,
        'chord': 1.0,
        'rho': 1.225,
        't_final': 0.1,
        'dt': 0.02,
        'alpha_mean': alpha_deg,
        'alpha_amplitude': 0.0,
        'pitch_frequency': 0.0,
        'pitch_pivot': 0.25
    }
    
    solver = UnsteadySolver(n_panels=100, n_points=60)
    results = solver.solve_unsteady(config)
    
    Cl_theory = 2 * np.pi * alpha_rad
    Cl_computed = np.mean(results['Cl'][-10:])
    
    print(f"\n  Angle of Attack: {alpha_deg} deg")
    print(f"  Theoretical Cl (thin airfoil): {Cl_theory:.4f}")
    print(f"  Computed Cl: {Cl_computed:.4f}")
    print(f"  Relative error: {abs(Cl_computed - Cl_theory) / Cl_theory * 100:.2f}%")
    
    return Cl_computed, Cl_theory


def main():
    print("\n" + "#" * 60)
    print("#  Unsteady Aerodynamic Force Calculation for")
    print("#  Flexible Airfoil using Vortex Lattice Method")
    print("#" * 60)
    
    run_static_validation()
    
    results_pitch = example_pitch_motion()
    plot_wake(results_pitch, {}, "pitch")
    
    results_camber = example_camber_deformation()
    plot_wake(results_camber, {}, "camber")
    
    results_combined = example_combined_motion()
    plot_wake(results_combined, {}, "combined")
    
    print("\n" + "=" * 60)
    print("All calculations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
