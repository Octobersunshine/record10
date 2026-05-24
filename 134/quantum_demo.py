import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quantum_spin_glass import (
    TransverseFieldSK,
    QuantumAnnealing,
    compute_quantum_phase_diagram,
    plot_quantum_phase_diagram,
    plot_quantum_annealing_results
)


def demo_dmft_solution():
    print("="*70)
    print("DEMO 1: DMFT Solution of Quantum Spin Glass")
    print("="*70)
    
    model = TransverseFieldSK(J0=0.0, J=1.0, seed=42)
    
    T_vals = [0.1, 0.5, 1.0, 1.5]
    Gamma_vals = np.linspace(0, 2.5, 12)
    
    results_T = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for T in T_vals:
        m_list, q_list, X_list = [], [], []
        for Gamma in Gamma_vals:
            result = model.solve_dmft(T, Gamma)
            m_list.append(result.m)
            q_list.append(result.q)
            X_list.append(result.X)
        
        results_T[T] = {'m': m_list, 'q': q_list, 'X': X_list}
        
        axes[0].plot(Gamma_vals, m_list, 'o-', label=f'T={T}', markersize=4)
        axes[1].plot(Gamma_vals, q_list, 'o-', label=f'T={T}', markersize=4)
        axes[2].plot(Gamma_vals, X_list, 'o-', label=f'T={T}', markersize=4)
    
    axes[0].set_xlabel('Transverse Field Γ')
    axes[0].set_ylabel('Magnetization m')
    axes[0].set_title('m vs Γ')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Transverse Field Γ')
    axes[1].set_ylabel('Glass Order q')
    axes[1].set_title('q vs Γ')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Transverse Field Γ')
    axes[2].set_ylabel('Susceptibility χ')
    axes[2].set_title('χ vs Γ')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dmft_order_params.png', dpi=150)
    print("\nSaved: dmft_order_params.png")
    
    print("\nKey observations:")
    print("  - At low T and small Γ: q ≈ 1 (spin glass phase)")
    print("  - At critical Γ: q drops to 0 (quantum paramagnet)")
    print("  - Susceptibility shows peak at quantum phase transition")


def demo_quantum_phase_diagram():
    print("\n" + "="*70)
    print("DEMO 2: Quantum Phase Diagram (T-Γ plane)")
    print("="*70)
    
    print("\nComputing phase diagram (this may take a minute)...")
    Ts, Gammas, m_map, q_map = compute_quantum_phase_diagram(
        T_min=0.05, T_max=2.0, n_T=12,
        Gamma_min=0.0, Gamma_max=2.5, n_Gamma=12,
        J0=0.0, J=1.0
    )
    
    plot_quantum_phase_diagram(Ts, Gammas, m_map, q_map, 
                                save_path='quantum_phase_diagram.png')
    print("Saved: quantum_phase_diagram.png")
    
    print("\nPhase diagram features:")
    print("  - Spin glass phase: low T, small Γ")
    print("  - Quantum paramagnetic phase: large Γ")
    print("  - Classical paramagnetic phase: high T")
    print("  - Quantum critical point at T→0")


def demo_quantum_annealing():
    print("\n" + "="*70)
    print("DEMO 3: Quantum Annealing for Optimization")
    print("="*70)
    
    N = 25
    print(f"\nCreating spin glass problem with N={N} spins...")
    
    qa = QuantumAnnealing(N=N, seed=123)
    qa.set_spinglass(J_std=1.0)
    
    print("Comparing 3 annealing schedules (3 runs each):")
    print("  1. Linear: Γ(t) = Γ0 - (Γ0-Γ1)*t")
    print("  2. Exponential: Γ(t) = Γ0*(Γ1/Γ0)^t")
    print("  3. Cosine: Γ(t) = slow at start, fast in middle")
    
    results = qa.compare_annealing_schedules(n_runs=3)
    
    plot_quantum_annealing_results(qa, results, save_path='quantum_annealing.png')
    print("Saved: quantum_annealing.png")
    
    print("\nSchedule performance comparison:")
    for schedule in ['linear', 'exponential', 'cosine']:
        final_E = results[schedule]['mean'][-1]
        print(f"  {schedule}: final E = {final_E:.2f}")


def demo_quantum_critical_point():
    print("\n" + "="*70)
    print("DEMO 4: Quantum Critical Point Estimation")
    print("="*70)
    
    model = TransverseFieldSK(J0=0.0, J=1.0, seed=42)
    
    T_low = 0.1
    Gamma_c = model.find_quantum_critical_point(T=T_low)
    
    print(f"\nAt T = {T_low}:")
    print(f"  Estimated quantum critical point: Γ_c ≈ {Gamma_c:.3f}")
    
    Gamma_scan = np.linspace(0, 2, 20)
    q_scan = [model.solve_dmft(T_low, G).q for G in Gamma_scan]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(Gamma_scan, q_scan, 'o-', linewidth=2)
    ax.axvline(x=Gamma_c, color='r', linestyle='--', label=f'Γ_c ≈ {Gamma_c:.2f}')
    ax.set_xlabel('Transverse Field Γ')
    ax.set_ylabel('Order Parameter q')
    ax.set_title(f'Quantum Phase Transition at T = {T_low}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quantum_critical.png', dpi=150)
    print("Saved: quantum_critical.png")


def main():
    print("\n" + "="*70)
    print("QUANTUM SPIN GLASS SIMULATION SUITE")
    print("Dynamical Mean-Field Theory + Quantum Annealing")
    print("="*70)
    
    try:
        demo_dmft_solution()
        demo_quantum_phase_diagram()
        demo_quantum_annealing()
        demo_quantum_critical_point()
        
        print("\n" + "="*70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("  - dmft_order_params.png")
        print("  - quantum_phase_diagram.png")
        print("  - quantum_annealing.png")
        print("  - quantum_critical.png")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
