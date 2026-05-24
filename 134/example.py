import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spin_glass import run_adaptive_pt_simulation, plot_results, plot_adaptation_history

print("="*70)
print("Edwards-Anderson Spin Glass Simulation - Adaptive PT")
print("="*70)

L = 8
print(f"\nLattice size: {L}x{L} = {L*L} spins")
print("Model: Edwards-Anderson (±J bonds)")
print("Method: Adaptive Parallel Tempering Monte Carlo")
print("Target exchange rate: 20%-60%")

result, pt, history = run_adaptive_pt_simulation(
    L=L,
    T_min=0.3,
    T_max=2.0,
    n_replicas_initial=10,
    n_steps=3000,
    n_equil=1000,
    n_adapt_cycles=4,
    min_exchange_rate=0.2,
    max_exchange_rate=0.6,
    seed=42
)

print("\n" + "="*70)
print("Final Results")
print("="*70)
print(f"{'T':>8} {'q':>10} {'Cv':>10} {'ExRate':>10}")
print("-"*70)
for i, (T, q, cv) in enumerate(zip(result.temperatures, result.q_mean, result.specific_heat)):
    ex_rate = result.exchange_rates[i-1] if i > 0 else np.nan
    print(f"{T:8.3f} {q:10.4f} {cv:10.4f} {ex_rate:10.2f}")

plot_adaptation_history(history, save_path='adaptation_history.png')
T_c_est = plot_results(result, L, save_path=f'spinglass_L{L}.png')

print(f"\nEstimated glass transition temperature: T_g ≈ {T_c_est:.3f}")
print(f"\nPlots saved:")
print(f"  - adaptation_history.png")
print(f"  - spinglass_L{L}.png")
print("="*70)
