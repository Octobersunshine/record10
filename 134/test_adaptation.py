import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spin_glass import EdwardsAnderson, AdaptiveParallelTempering

def compare_adaptation():
    print("="*70)
    print("Comparing Fixed vs Adaptive Temperature Spacing")
    print("="*70)
    
    L = 6
    T_min, T_max = 0.4, 1.8
    n_replicas = 10
    
    model = EdwardsAnderson(L=L, seed=42)
    Ts_fixed = np.geomspace(T_min, T_max, n_replicas)
    
    print(f"\nLattice: {L}x{L}, Temperature range: [{T_min}, {T_max}]")
    print(f"Number of replicas: {n_replicas}")
    print(f"Target exchange rate: 20%-60%")
    
    pt_fixed = AdaptiveParallelTempering(model, Ts_fixed.copy(), seed=123)
    pt_adaptive = AdaptiveParallelTempering(
        model, Ts_fixed.copy(),
        min_exchange_rate=0.2,
        max_exchange_rate=0.6,
        seed=123
    )
    
    print("\n" + "-"*70)
    print("Fixed temperature spacing (before adaptation)")
    print("-"*70)
    
    for _ in range(500):
        pt_fixed.step()
    
    rates_fixed = pt_fixed.exchange_rates()
    print(f"Fixed Ts: {Ts_fixed}")
    print(f"Exchange rates: {rates_fixed}")
    print(f"Mean exchange rate: {np.mean(rates_fixed):.3f}")
    print(f"Within 20%-60% range: {np.sum((rates_fixed >= 0.2) & (rates_fixed <= 0.6))}/{len(rates_fixed)}")
    
    print("\n" + "-"*70)
    print("Running adaptive optimization...")
    print("-"*70)
    
    Ts_opt, history = pt_adaptive.optimize_temperatures(n_cycles=4)
    
    print("\n" + "-"*70)
    print("After adaptive optimization")
    print("-"*70)
    
    pt_adaptive.reset_statistics()
    for _ in range(500):
        pt_adaptive.step()
    
    rates_adaptive = pt_adaptive.exchange_rates()
    print(f"Optimized Ts: {Ts_opt}")
    print(f"Exchange rates: {rates_adaptive}")
    print(f"Mean exchange rate: {np.mean(rates_adaptive):.3f}")
    print(f"Within 20%-60% range: {np.sum((rates_adaptive >= 0.2) & (rates_adaptive <= 0.6))}/{len(rates_adaptive)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(Ts_fixed, 'o-', label='Fixed (geometric)')
    ax.plot(Ts_opt, 's-', label='Adaptive')
    ax.set_xlabel('Replica index')
    ax.set_ylabel('Temperature T')
    ax.set_title('Temperature Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(np.diff(Ts_fixed), 'o-', label='Fixed spacing')
    ax.plot(np.diff(Ts_opt), 's-', label='Adaptive spacing')
    ax.set_xlabel('Gap index')
    ax.set_ylabel('ΔT')
    ax.set_title('Temperature Spacing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    T_mid_fixed = (Ts_fixed[:-1] + Ts_fixed[1:]) / 2
    T_mid_opt = (Ts_opt[:-1] + Ts_opt[1:]) / 2
    ax.plot(T_mid_fixed, rates_fixed, 'o-', label='Fixed')
    ax.plot(T_mid_opt, rates_adaptive, 's-', label='Adaptive')
    ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0.6, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Exchange Rate')
    ax.set_title('Exchange Rates vs T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    ax = axes[1, 1]
    Cv_fixed = pt_fixed.get_specific_heat()
    Cv_opt = pt_adaptive.get_specific_heat()
    ax.plot(Ts_fixed, Cv_fixed, 'o-', label='Fixed')
    ax.plot(Ts_opt, Cv_opt, 's-', label='Adaptive')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Specific Heat Cv')
    ax.set_title('Specific Heat')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptation_comparison.png', dpi=150)
    print(f"\nComparison plot saved to: adaptation_comparison.png")
    
    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    print(f"Mean exchange rate: {np.mean(rates_fixed):.3f} → {np.mean(rates_adaptive):.3f}")
    print(f"Rates in range: {np.sum((rates_fixed >= 0.2) & (rates_fixed <= 0.6))}/{len(rates_fixed)} → {np.sum((rates_adaptive >= 0.2) & (rates_adaptive <= 0.6))}/{len(rates_adaptive)}")
    print("="*70)


if __name__ == "__main__":
    compare_adaptation()
