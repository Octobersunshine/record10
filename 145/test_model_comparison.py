import numpy as np
import sys
sys.path.insert(0, '.')
from kdv_internal_wave import KdVSolver
from kdv_improved import VariableCoefficientKdV
import matplotlib.pyplot as plt


def test_energy_conservation():
    print("=" * 80)
    print("TEST 1: Energy Conservation Comparison")
    print("=" * 80)
    
    L = 80
    N = 512
    dt = 0.001
    T_max = 3
    
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]
    
    h1, h2 = 1.0, 0.7
    x_trans, width = 0, 5
    transition = 0.5 * (1 + np.tanh((x - x_trans) / width))
    h = h1 * (1 - transition) + h2 * transition
    
    amp = 0.1
    h0 = 1.0
    u0 = amp * (1 / np.cosh((x + 25) / np.sqrt(4 * h0**3 / (3 * amp)))) ** 2
    
    print("\nRunning OLD KdV model (standard KdV with ad-hoc terrain modification)...")
    old_solver = KdVSolver(L=L, N=N, dt=dt, T_max=T_max)
    
    def old_terrain(x, **kwargs):
        return h
    old_solver.terrain_function = lambda *args, **kwargs: old_terrain(x)
    
    u_old, h_old = old_solver.solve(u0, terrain_type='flat')
    
    energy_old = []
    for i in range(len(u_old)):
        energy = 0.5 * np.sum(h * u_old[i]**2) * dx
        energy_old.append(energy)
    energy_old = np.array(energy_old)
    error_old = np.max(np.abs((energy_old - energy_old[0]) / energy_old[0] * 100))
    print(f"  OLD model max energy error: {error_old:.4f}%")
    
    print("\nRunning NEW vKdV model (conservation form)...")
    new_solver = VariableCoefficientKdV(L=L, N=N, dt=dt, T_max=T_max, equation_type='vkdv')
    
    def new_terrain(x, **kwargs):
        return h
    new_solver.terrain_function = new_terrain
    
    u_new, h_new = new_solver.solve(u0, terrain_type='flat')
    error_new = new_solver.plot_diagnostics('vkdv_conservation_test.png')['max_energy_error']
    print(f"  NEW model max energy error: {error_new:.4f}%")
    
    print("\nRunning Boussinesq model...")
    bouss_solver = VariableCoefficientKdV(L=L, N=N, dt=dt, T_max=T_max, equation_type='boussinesq')
    bouss_solver.terrain_function = new_terrain
    u_bouss, h_bouss = bouss_solver.solve(u0, terrain_type='flat')
    error_bouss = bouss_solver.plot_diagnostics('boussinesq_conservation_test.png')['max_energy_error']
    print(f"  Boussinesq model max energy error: {error_bouss:.4f}%")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    t = np.arange(len(u_old)) * dt
    
    axes[0].plot(t, (energy_old - energy_old[0]) / energy_old[0] * 100, 
                 'r--', label=f'Old KdV (error: {error_old:.2f}%)', linewidth=2)
    axes[0].plot(t, (np.array(new_solver.energy_history) - new_solver.energy_history[0]) / 
                 new_solver.energy_history[0] * 100, 
                 'b-', label=f'New vKdV (error: {error_new:.2f}%)', linewidth=2)
    axes[0].plot(t, (np.array(bouss_solver.energy_history) - bouss_solver.energy_history[0]) / 
                 bouss_solver.energy_history[0] * 100, 
                 'g-', label=f'Boussinesq (error: {error_bouss:.2f}%)', linewidth=2)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Relative Energy Error (%)')
    axes[0].set_title('Energy Conservation Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    t_idx = -1
    axes[1].plot(x, u_old[t_idx], 'r--', label='Old KdV', alpha=0.7)
    axes[1].plot(x, u_new[t_idx], 'b-', label='New vKdV', alpha=0.7)
    axes[1].plot(x, u_bouss[t_idx], 'g-', label='Boussinesq', alpha=0.7)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Waveform at t={t[t_idx]:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_comparison_detailed.png', dpi=150)
    plt.close()
    
    print(f"\n  Improvement (OLD vs NEW vKdV): {error_old/error_new:.1f}x better")
    print("\nSaved: energy_comparison_detailed.png")
    
    return {
        'old_error': error_old,
        'new_error': error_new,
        'bouss_error': error_bouss
    }


def test_reflection():
    print("\n" + "=" * 80)
    print("TEST 2: Reflection Analysis")
    print("=" * 80)
    
    L = 100
    N = 1024
    dt = 0.0005
    T_max = 5
    
    x = np.linspace(-L/2, L/2, N)
    
    h1, h2 = 1.0, 0.5
    x_trans, width = 0, 3
    transition = 0.5 * (1 + np.tanh((x - x_trans) / width))
    h = h1 * (1 - transition) + h2 * transition
    
    amp = 0.08
    h0 = 1.0
    u0 = amp * (1 / np.cosh((x + 35) / np.sqrt(4 * h0**3 / (3 * amp)))) ** 2
    
    def terrain_func(x, **kwargs):
        return h
    
    print("\nRunning OLD model...")
    old_solver = KdVSolver(L=L, N=N, dt=dt, T_max=T_max)
    old_solver.terrain_function = lambda *args, **kwargs: terrain_func(x)
    u_old, _ = old_solver.solve(u0, terrain_type='flat')
    
    print("Running NEW vKdV model...")
    new_solver = VariableCoefficientKdV(L=L, N=N, dt=dt, T_max=T_max, equation_type='vkdv')
    new_solver.terrain_function = terrain_func
    u_new, _ = new_solver.solve(u0, terrain_type='flat')
    
    print("Running Boussinesq model...")
    bouss_solver = VariableCoefficientKdV(L=L, N=N, dt=dt, T_max=T_max, equation_type='boussinesq')
    bouss_solver.terrain_function = terrain_func
    u_bouss, _ = bouss_solver.solve(u0, terrain_type='flat')
    
    t_idx = int(T_max / dt / 2)
    
    left_mask = x < x_trans - 2*width
    right_mask = x > x_trans + 2*width
    
    def measure_reflection(u, mask):
        reflected_energy = np.sum(u[mask]**2)
        return reflected_energy
    
    old_refl = measure_reflection(u_old[t_idx], left_mask)
    new_refl = measure_reflection(u_new[t_idx], left_mask)
    bouss_refl = measure_reflection(u_bouss[t_idx], left_mask)
    
    old_trans = measure_reflection(u_old[t_idx], right_mask)
    new_trans = measure_reflection(u_new[t_idx], right_mask)
    bouss_trans = measure_reflection(u_bouss[t_idx], right_mask)
    
    print(f"\nAt t={t_idx*dt:.2f}:")
    print(f"  OLD KdV:    Reflected energy = {old_refl:.6f}, Transmitted = {old_trans:.6f}")
    print(f"  NEW vKdV:   Reflected energy = {new_refl:.6f}, Transmitted = {new_trans:.6f}")
    print(f"  Boussinesq: Reflected energy = {bouss_refl:.6f}, Transmitted = {bouss_trans:.6f}")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    models = [
        ('OLD KdV', u_old, 'r--'),
        ('NEW vKdV', u_new, 'b-'),
        ('Boussinesq', u_bouss, 'g-')
    ]
    
    for i, (name, u, style) in enumerate(models):
        for t_show in [0, t_idx//2, t_idx]:
            axes[i].plot(x, u[t_show], style, label=f't={t_show*dt:.2f}', alpha=0.8 if t_show > 0 else 1.0)
        axes[i].axvline(x=x_trans, color='k', linestyle=':', alpha=0.5, label='Terrain transition')
        axes[i].fill_between(x, -0.01, -0.01*(h/h.max()), alpha=0.2, color='gray')
        axes[i].set_title(f'{name} - Reflection Analysis')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-0.02, amp*1.5)
    
    axes[-1].set_xlabel('x')
    plt.tight_layout()
    plt.savefig('reflection_comparison.png', dpi=150)
    plt.close()
    
    print("\nSaved: reflection_comparison.png")
    
    return {
        'old_refl': old_refl,
        'new_refl': new_refl,
        'bouss_refl': bouss_refl
    }


def test_convergence():
    print("\n" + "=" * 80)
    print("TEST 3: Numerical Convergence")
    print("=" * 80)
    
    L = 50
    T_max = 2
    amp = 0.1
    h0 = 1.0
    
    N_values = [128, 256, 512, 1024]
    dt_values = [0.002, 0.001, 0.0005, 0.00025]
    
    errors_vkdv = []
    errors_bouss = []
    
    x_fine = np.linspace(-L/2, L/2, 2048)
    u0_fine = amp * (1 / np.cosh((x_fine + 15) / np.sqrt(4 * h0**3 / (3 * amp)))) ** 2
    
    solver_ref = VariableCoefficientKdV(L=L, N=2048, dt=0.0001, T_max=T_max, equation_type='vkdv')
    u_ref, _ = solver_ref.solve(u0_fine, terrain_type='flat')
    u_ref_final = u_ref[-1]
    
    for N, dt in zip(N_values, dt_values):
        print(f"\nTesting N={N}, dt={dt}...")
        
        x = np.linspace(-L/2, L/2, N)
        u0 = amp * (1 / np.cosh((x + 15) / np.sqrt(4 * h0**3 / (3 * amp)))) ** 2
        
        solver = VariableCoefficientKdV(L=L, N=N, dt=dt, T_max=T_max, equation_type='vkdv')
        u_hist, _ = solver.solve(u0, terrain_type='flat')
        
        u_final = u_hist[-1]
        u_ref_interp = np.interp(x, x_fine, u_ref_final)
        
        error = np.sqrt(np.sum((u_final - u_ref_interp)**2) * (L/N))
        errors_vkdv.append(error)
        print(f"  vKdV L2 error: {error:.2e}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(N_values, errors_vkdv, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('N (grid points)')
    ax.set_ylabel('L2 Error')
    ax.set_title('Numerical Convergence - vKdV Model')
    ax.grid(True, alpha=0.3, which='both')
    
    for i in range(len(N_values)-1):
        rate = np.log(errors_vkdv[i]/errors_vkdv[i+1]) / np.log(2)
        ax.text(N_values[i], errors_vkdv[i]*1.2, f'rate={rate:.1f}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('convergence_test.png', dpi=150)
    plt.close()
    
    print("\nSaved: convergence_test.png")
    return errors_vkdv


def demo_realistic_scenario():
    print("\n" + "=" * 80)
    print("DEMO: Realistic Oceanic Internal Wave Scenario")
    print("=" * 80)
    
    L = 200
    N = 1024
    dt = 0.001
    T_max = 10
    
    x = np.linspace(-L/2, L/2, N)
    
    def realistic_terrain(x):
        h = 200 * np.ones_like(x)
        
        h -= 100 * np.exp(-((x - 20) ** 2) / (2 * 15 ** 2))
        
        shelf_start = 40
        h = np.where(x > shelf_start, 
                     200 - 120 * (1 - np.exp(-(x - shelf_start) / 10)),
                     h)
        return h / 200
    
    h = realistic_terrain(x)
    
    amp = 0.05
    h0 = 1.0
    u0 = amp * (1 / np.cosh((x + 60) / np.sqrt(4 * h0**3 / (3 * amp)))) ** 2
    
    print("\nRunning vKdV model on realistic terrain...")
    solver = VariableCoefficientKdV(L=L, N=N, dt=dt, T_max=T_max, equation_type='vkdv')
    solver.terrain_function = lambda x, **kwargs: h
    u_history, _ = solver.solve(u0, terrain_type='flat')
    
    diag = solver.plot_diagnostics('realistic_diagnostics.png')
    print(f"  Max energy error: {diag['max_energy_error']:.4f}%")
    
    solver.plot_evolution(u_history, h, 'realistic_evolution.png')
    solver.plot_spacetime(u_history, 'realistic_spacetime.png')
    
    step = max(1, len(u_history) // 100)
    solver.create_animation(u_history[::step], h, 'realistic_animation.gif', fps=20)
    
    print("Saved: realistic_diagnostics.png, realistic_evolution.png")
    print("       realistic_spacetime.png, realistic_animation.gif")


def main():
    print("\n" + "#" * 80)
    print("# IMPROVED KdV MODELS - VALIDATION TEST SUITE")
    print("#" * 80 + "\n")
    
    errors = test_energy_conservation()
    
    reflections = test_reflection()
    
    convergence = test_convergence()
    
    demo_realistic_scenario()
    
    print("\n" + "=" * 80)
    print("SUMMARY OF IMPROVEMENTS")
    print("=" * 80)
    print(f"\nEnergy Conservation:")
    print(f"  OLD model:    {errors['old_error']:.4f}% error")
    print(f"  NEW vKdV:     {errors['new_error']:.4f}% error ({errors['old_error']/errors['new_error']:.1f}x improvement)")
    print(f"  Boussinesq:   {errors['bouss_error']:.4f}% error")
    
    print(f"\nKey Improvements:")
    print("  ✓ Conservation-form variable-coefficient KdV equation")
    print("  ✓ De-aliasing filter (2/3 rule) for numerical stability")
    print("  ✓ Boussinesq-type model as alternative")
    print("  ✓ Built-in energy/mass conservation diagnostics")
    print("  ✓ Physically consistent terrain interaction")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
