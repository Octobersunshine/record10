import numpy as np
import sys
sys.path.insert(0, '.')

from kdv_internal_wave import KdVSolver
from kdv_improved import VariableCoefficientKdV
from nonhydrostatic_model import NonhydrostaticModel
import matplotlib.pyplot as plt


def compare_all_models_1D():
    print("=" * 80)
    print("COMPARISON: KdV vs vKdV vs Boussinesq vs Nonhydrostatic")
    print("=" * 80)
    
    L = 80
    N = 256
    dt = 0.002
    T_max = 4
    
    x = np.linspace(-L/2, L/2, N)
    
    def terrain(x):
        h1, h2 = 1.0, 0.6
        transition = 0.5 * (1 + np.tanh(x / 5))
        return h1 * (1 - transition) + h2 * transition
    
    h = terrain(x)
    amp = 0.08
    u0 = amp * (1 / np.cosh((x + 25) / 5)) ** 2
    
    results = {}
    
    print("\n1. Standard KdV (original)...")
    solver_kdv = KdVSolver(L=L, N=N, dt=dt, T_max=T_max)
    solver_kdv.terrain_function = lambda *args, **kwargs: h
    u_kdv, _ = solver_kdv.solve(u0, terrain_type='flat')
    results['KdV'] = u_kdv
    
    print("2. Variable-coefficient KdV...")
    solver_vkdv = VariableCoefficientKdV(L=L, N=N, dt=dt, T_max=T_max, equation_type='vkdv')
    solver_vkdv.terrain_function = lambda *args, **kwargs: h
    u_vkdv, _ = solver_vkdv.solve(u0, terrain_type='flat')
    results['vKdV'] = u_vkdv
    
    print("3. Boussinesq...")
    solver_bouss = VariableCoefficientKdV(L=L, N=N, dt=dt, T_max=T_max, equation_type='boussinesq')
    solver_bouss.terrain_function = lambda *args, **kwargs: h
    u_bouss, _ = solver_bouss.solve(u0, terrain_type='flat')
    results['Boussinesq'] = u_bouss
    
    print("\nPlotting comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_indices = [0, int(T_max/dt/4), int(T_max/dt/2), int(T_max/dt)]
    
    for i, (name, u_hist) in enumerate(results.items()):
        ax = axes[i//2, i%2]
        for t_idx in t_indices:
            ax.plot(x, u_hist[t_idx], label=f't={t_idx*dt:.2f}')
        ax.fill_between(x, -0.01, -0.01*h/h.max(), alpha=0.3, color='gray', label='Terrain')
        ax.set_title(f'{name} Model')
        ax.set_xlabel('x')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_1D.png', dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    t_idx = -1
    ax.plot(x, results['KdV'][t_idx], 'r--', label='Standard KdV', linewidth=2)
    ax.plot(x, results['vKdV'][t_idx], 'b-', label='vKdV (conservative)', linewidth=2)
    ax.plot(x, results['Boussinesq'][t_idx], 'g-', label='Boussinesq', linewidth=2, alpha=0.7)
    ax.fill_between(x, -0.01, -0.01*h/h.max(), alpha=0.3, color='gray', label='Terrain')
    ax.set_title(f'Comparison at t={t_idx*dt:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison_final.png', dpi=150)
    plt.close()
    
    print("\nSaved: model_comparison_1D.png, model_comparison_final.png")
    
    return results


def mixing_rate_analysis():
    print("\n" + "=" * 80)
    print("DETAILED MIXING RATE ANALYSIS")
    print("=" * 80)
    
    model = NonhydrostaticModel(
        Lx=100, Lz=30, Nx=128, Nz=32, 
        dt=0.015, T_max=25,
        nu=2e-4, kappa=1e-4
    )
    
    print("\nSetting up simulation...")
    model.set_terrain('ridge', height=12.0, x0=50.0, width=20.0)
    model.set_density_profile('two_layer', rho1=1000.0, rho2=1002.0, 
                              z_interface=18.0, thickness=1.2)
    model.add_internal_wave('mode1', x0=15.0, amp=1.8, width=10.0)
    
    print("Running nonhydrostatic simulation...")
    u_hist, w_hist, rho_hist = model.solve(save_interval=15)
    
    print("\nComputing mixing diagnostics...")
    mix_stats = model.plot_mixing_diagnostics('mixing_analysis_detailed.png')
    
    times = [d['time'] for d in model.mixing_history]
    KE = np.array([d['KE'] for d in model.mixing_history])
    APE = np.array([d['available_PE'] for d in model.mixing_history])
    mix_eff = np.array([d['mixing_efficiency'] for d in model.mixing_history])
    
    dKE_dt = np.gradient(KE, times)
    dAPE_dt = np.gradient(APE, times)
    
    mixing_rate = -dKE_dt - dAPE_dt
    mixing_rate = np.maximum(mixing_rate, 0)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(times, mixing_rate, 'r-', linewidth=2)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mixing Rate (dE/dt)')
    axes[0].set_title('Temporal Mixing Rate')
    axes[0].grid(True, alpha=0.3)
    
    cumulative_mixing = np.cumsum(mixing_rate) * (times[1] - times[0])
    axes[1].plot(times, cumulative_mixing, 'b-', linewidth=2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Cumulative Mixing')
    axes[1].set_title('Cumulative Mixing')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mixing_rate_timeseries.png', dpi=150)
    plt.close()
    
    t_half = len(times) // 2
    rho_initial = rho_hist[0]
    rho_final = rho_hist[-1]
    
    gradient_initial = np.sqrt(np.gradient(rho_initial, axis=0)**2 + np.gradient(rho_initial, axis=1)**2)
    gradient_final = np.sqrt(np.gradient(rho_final, axis=0)**2 + np.gradient(rho_final, axis=1)**2)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    im0 = axes[0, 0].pcolormesh(model.X, model.Z, rho_initial, cmap='viridis', shading='auto')
    axes[0, 0].contour(model.X, model.Z, rho_initial, levels=12, colors='w', alpha=0.5)
    axes[0, 0].fill_between(model.x, 0, model.terrain, color='gray', alpha=0.5)
    axes[0, 0].set_title('Initial Density')
    axes[0, 0].set_ylabel('z')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].pcolormesh(model.X, model.Z, rho_final, cmap='viridis', shading='auto')
    axes[0, 1].contour(model.X, model.Z, rho_final, levels=12, colors='w', alpha=0.5)
    axes[0, 1].fill_between(model.x, 0, model.terrain, color='gray', alpha=0.5)
    axes[0, 1].set_title('Final Density (After Breaking)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].pcolormesh(model.X, model.Z, gradient_initial, cmap='hot', shading='auto')
    axes[1, 0].fill_between(model.x, 0, model.terrain, color='gray', alpha=0.5)
    axes[1, 0].set_title('Initial Density Gradient')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('z')
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].pcolormesh(model.X, model.Z, gradient_final, cmap='hot', shading='auto')
    axes[1, 1].fill_between(model.x, 0, model.terrain, color='gray', alpha=0.5)
    axes[1, 1].set_title('Final Density Gradient')
    axes[1, 1].set_xlabel('x')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('density_evolution_mixing.png', dpi=150)
    plt.close()
    
    model.plot_snapshot(u_hist[-1], w_hist[-1], rho_hist[-1], 'nh_final_snapshot.png')
    
    print("\n" + "=" * 80)
    print("MIXING ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"  Peak Mixing Rate: {np.max(mixing_rate):.2e}")
    print(f"  Total Cumulative Mixing: {cumulative_mixing[-1]:.2e}")
    print(f"  Final Mixing Efficiency: {mix_stats['final_mixing_efficiency']:.4f}")
    print(f"  Max Mixing Efficiency: {mix_stats['max_mixing_efficiency']:.4f}")
    print(f"  Energy Dissipation: {mix_stats['total_energy_dissipation']:.2f}%")
    print("=" * 80)
    
    print("\nSaved: mixing_analysis_detailed.png, mixing_rate_timeseries.png")
    print("       density_evolution_mixing.png, nh_final_snapshot.png")
    
    return model, mix_stats


def parameter_sweep_mixing():
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP: Mixing vs Wave Amplitude")
    print("=" * 80)
    
    amplitudes = [0.8, 1.2, 1.6, 2.0]
    mixing_results = []
    
    for amp in amplitudes:
        print(f"\nRunning simulation with amplitude = {amp}...")
        
        model = NonhydrostaticModel(
            Lx=80, Lz=25, Nx=96, Nz=24, 
            dt=0.02, T_max=20,
            nu=1e-4, kappa=1e-4
        )
        
        model.set_terrain('ridge', height=10.0, x0=40.0, width=15.0)
        model.set_density_profile('two_layer', rho1=1000.0, rho2=1002.0, 
                                  z_interface=15.0, thickness=1.0)
        model.add_internal_wave('mode1', x0=10.0, amp=amp, width=8.0)
        
        u_hist, w_hist, rho_hist = model.solve(save_interval=20)
        mix_stats = model.plot_mixing_diagnostics(f'mixing_amp_{amp}.png')
        
        mixing_results.append({
            'amplitude': amp,
            'max_efficiency': mix_stats['max_mixing_efficiency'],
            'final_efficiency': mix_stats['final_mixing_efficiency'],
            'dissipation': mix_stats['total_energy_dissipation']
        })
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    amps = [r['amplitude'] for r in mixing_results]
    max_effs = [r['max_efficiency'] for r in mixing_results]
    diss = [r['dissipation'] for r in mixing_results]
    
    axes[0].plot(amps, max_effs, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Wave Amplitude')
    axes[0].set_ylabel('Max Mixing Efficiency')
    axes[0].set_title('Mixing Efficiency vs Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(amps, diss, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Wave Amplitude')
    axes[1].set_ylabel('Energy Dissipation (%)')
    axes[1].set_title('Energy Dissipation vs Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_mixing.png', dpi=150)
    plt.close()
    
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 80)
    for r in mixing_results:
        print(f"  Amp={r['amplitude']:4.1f}: MaxEff={r['max_efficiency']:.4f}, "
              f"Dissipation={r['dissipation']:.2f}%")
    print("=" * 80)
    
    print("\nSaved: parameter_sweep_mixing.png")
    
    return mixing_results


def main():
    print("\n" + "#" * 80)
    print("# COMPREHENSIVE DEMO: Wave Models and Mixing Analysis")
    print("#" * 80 + "\n")
    
    compare_all_models_1D()
    
    model, mix_stats = mixing_rate_analysis()
    
    parameter_sweep_mixing()
    
    print("\n" + "#" * 80)
    print("# DEMO COMPLETED SUCCESSFULLY")
    print("#" * 80)
    print("\nGenerated output files:")
    print("  - model_comparison_1D.png")
    print("  - model_comparison_final.png")
    print("  - mixing_analysis_detailed.png")
    print("  - mixing_rate_timeseries.png")
    print("  - density_evolution_mixing.png")
    print("  - nh_final_snapshot.png")
    print("  - parameter_sweep_mixing.png")
    print("  - mixing_amp_*.png (for each amplitude)")
    print("\nKey features demonstrated:")
    print("  ✓ 1D model comparison (KdV, vKdV, Boussinesq)")
    print("  ✓ 2D nonhydrostatic Euler equations")
    print("  ✓ Internal wave breaking simulation")
    print("  ✓ Mixing rate calculation")
    print("  ✓ Mixing efficiency analysis")
    print("  ✓ Parameter sensitivity study")
    print("#" * 80)


if __name__ == "__main__":
    main()
