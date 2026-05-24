import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.integrate import quad
from scipy.optimize import root_scalar
import pickle


@dataclass
class DMFTResult:
    T: float
    Gamma: float
    m: float
    q: float
    X: float
    converged: bool
    n_iter: int


class TransverseFieldSK:
    def __init__(self, J0: float = 1.0, J: float = 1.0, seed: Optional[int] = None):
        self.J0 = J0
        self.J = J
        self.rng = np.random.default_rng(seed)
    
    def free_energy_quantum(self, m: float, q: float, T: float, Gamma: float) -> float:
        if T <= 0:
            T = 1e-6
        
        integrand = lambda x: np.log(2 * np.cosh(np.sqrt((self.J0 * m + self.J * np.sqrt(q) * x)**2 + Gamma**2) / T))
        integral, _ = quad(integrand, -10, 10)
        gaussian_int = integral / np.sqrt(2 * np.pi)
        
        F = -T * gaussian_int - 0.5 * self.J0 * m**2 + 0.5 * self.J**2 * q
        return F
    
    def saddle_point_equations(self, vars: np.ndarray, T: float, Gamma: float) -> np.ndarray:
        m, q = vars
        if T <= 0:
            T = 1e-6
        
        def integrand_m(x):
            arg = np.sqrt((self.J0 * m + self.J * np.sqrt(q) * x)**2 + Gamma**2)
            tanh_term = np.tanh(arg / T)
            numerator = self.J0 * m + self.J * np.sqrt(q) * x
            return tanh_term * numerator / arg
        
        def integrand_q(x):
            arg = np.sqrt((self.J0 * m + self.J * np.sqrt(q) * x)**2 + Gamma**2)
            tanh_term = np.tanh(arg / T)
            numerator = self.J0 * m + self.J * np.sqrt(q) * x
            return (tanh_term * numerator / arg)**2
        
        int_m, _ = quad(integrand_m, -10, 10)
        int_q, _ = quad(integrand_q, -10, 10)
        
        int_m /= np.sqrt(2 * np.pi)
        int_q /= np.sqrt(2 * np.pi)
        
        dm = int_m - m
        dq = int_q - q
        
        return np.array([dm, dq])
    
    def solve_dmft(self, T: float, Gamma: float, 
                   m_init: float = 0.5, q_init: float = 0.5,
                   max_iter: int = 100, tol: float = 1e-6,
                   alpha: float = 0.3) -> DMFTResult:
        
        m, q = m_init, q_init
        converged = False
        
        for iteration in range(max_iter):
            if T <= 0:
                T = 1e-6
            
            def integrand_m(x):
                arg = np.sqrt((self.J0 * m + self.J * np.sqrt(max(q, 1e-10)) * x)**2 + Gamma**2)
                tanh_term = np.tanh(arg / T)
                numerator = self.J0 * m + self.J * np.sqrt(max(q, 1e-10)) * x
                return tanh_term * numerator / max(arg, 1e-10)
            
            def integrand_q(x):
                arg = np.sqrt((self.J0 * m + self.J * np.sqrt(max(q, 1e-10)) * x)**2 + Gamma**2)
                tanh_term = np.tanh(arg / T)
                numerator = self.J0 * m + self.J * np.sqrt(max(q, 1e-10)) * x
                return (tanh_term * numerator / max(arg, 1e-10))**2
            
            int_m, _ = quad(integrand_m, -10, 10)
            int_q, _ = quad(integrand_q, -10, 10)
            
            int_m /= np.sqrt(2 * np.pi)
            int_q /= np.sqrt(2 * np.pi)
            
            m_new = (1 - alpha) * m + alpha * int_m
            q_new = (1 - alpha) * q + alpha * int_q
            
            q_new = max(q_new, 0)
            
            delta = np.sqrt((m_new - m)**2 + (q_new - q)**2)
            
            m, q = m_new, q_new
            
            if delta < tol:
                converged = True
                break
        
        X = (q - m**2) / T if T > 0 else 0
        
        return DMFTResult(
            T=T, Gamma=Gamma,
            m=abs(m), q=q, X=X,
            converged=converged,
            n_iter=iteration + 1
        )
    
    def find_quantum_critical_point(self, T: float = 0.1) -> float:
        def critical_measure(Gamma):
            result = self.solve_dmft(T, Gamma)
            return result.q
        
        Gamma_vals = np.linspace(0, 3, 30)
        q_vals = [critical_measure(G) for G in Gamma_vals]
        
        dq_dGamma = np.gradient(q_vals, Gamma_vals)
        idx = np.argmin(np.abs(dq_dGamma))
        
        return Gamma_vals[idx]


class QuantumAnnealing:
    def __init__(self, N: int, J: Optional[np.ndarray] = None, seed: Optional[int] = None):
        self.N = N
        self.rng = np.random.default_rng(seed)
        
        if J is None:
            self.J = self.rng.standard_normal((N, N)) / np.sqrt(N)
            self.J = (self.J + self.J.T) / 2
            np.fill_diagonal(self.J, 0)
        else:
            self.J = J
        
        self.h = np.zeros(N)
    
    def set_ferromagnetic(self, J0: float = 1.0):
        self.J = -J0 * np.ones((self.N, self.N)) / self.N
        np.fill_diagonal(self.J, 0)
    
    def set_spinglass(self, J_std: float = 1.0):
        self.J = self.rng.standard_normal((self.N, self.N)) * J_std / np.sqrt(self.N)
        self.J = (self.J + self.J.T) / 2
        np.fill_diagonal(self.J, 0)
    
    def classical_energy(self, spins: np.ndarray) -> float:
        return -0.5 * spins @ self.J @ spins - self.h @ spins
    
    def quantum_energy(self, spins: np.ndarray, Gamma: float) -> float:
        E_class = self.classical_energy(spins)
        E_quantum = -Gamma * self.N
        return E_class + E_quantum
    
    def quantum_monte_carlo_step(self, spins: np.ndarray, Gamma: float, T: float, 
                                  n_trotter: int = 4) -> np.ndarray:
        beta = 1.0 / T if T > 0 else 1e10
        delta_tau = beta / n_trotter
        
        spins_trotter = np.tile(spins, (n_trotter, 1))
        
        for tau in range(n_trotter):
            for i in range(self.N):
                delta_E = 2 * spins_trotter[tau, i] * (self.J[i] @ spins_trotter[tau])
                
                tau_prev = (tau - 1) % n_trotter
                tau_next = (tau + 1) % n_trotter
                delta_E += -Gamma * delta_tau * (spins_trotter[tau_prev, i] + spins_trotter[tau_next, i])
                
                if delta_E <= 0 or self.rng.random() < np.exp(-delta_tau * delta_E):
                    spins_trotter[tau, i] *= -1
        
        return spins_trotter[0].copy()
    
    def quantum_annealing_schedule(self, Gamma_start: float, Gamma_end: float, 
                                   n_steps: int, schedule_type: str = 'linear') -> np.ndarray:
        if schedule_type == 'linear':
            return np.linspace(Gamma_start, Gamma_end, n_steps)
        elif schedule_type == 'exponential':
            return Gamma_start * (Gamma_end / Gamma_start) ** np.linspace(0, 1, n_steps)
        elif schedule_type == 'cosine':
            t = np.linspace(0, 1, n_steps)
            return 0.5 * (Gamma_start + Gamma_end) + 0.5 * (Gamma_start - Gamma_end) * np.cos(np.pi * t)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def run_quantum_annealing(self, Gamma_start: float = 5.0, Gamma_end: float = 0.01,
                              n_steps: int = 100, T: float = 0.1, n_trotter: int = 4,
                              n_equil: int = 10, schedule_type: str = 'linear',
                              spins_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float], np.ndarray]:
        
        if spins_init is None:
            spins = self.rng.choice([-1, 1], size=self.N)
        else:
            spins = spins_init.copy()
        
        Gamma_schedule = self.quantum_annealing_schedule(Gamma_start, Gamma_end, n_steps, schedule_type)
        
        energies = []
        all_spins = []
        
        for step, Gamma in enumerate(Gamma_schedule):
            for _ in range(n_equil):
                spins = self.quantum_monte_carlo_step(spins, Gamma, T, n_trotter)
            
            E = self.classical_energy(spins)
            energies.append(E)
            all_spins.append(spins.copy())
        
        return spins, energies, Gamma_schedule
    
    def compare_annealing_schedules(self, n_runs: int = 5) -> dict:
        results = {}
        schedules = ['linear', 'exponential', 'cosine']
        
        for schedule in schedules:
            energies_all = []
            for run in range(n_runs):
                _, energies, _ = self.run_quantum_annealing(
                    Gamma_start=5.0, Gamma_end=0.01,
                    n_steps=50, T=0.1, n_trotter=4,
                    n_equil=5, schedule_type=schedule
                )
                energies_all.append(energies)
            
            results[schedule] = {
                'mean': np.mean(energies_all, axis=0),
                'std': np.std(energies_all, axis=0),
                'all': energies_all
            }
        
        return results


def compute_quantum_phase_diagram(
    T_min: float = 0.01,
    T_max: float = 2.0,
    n_T: int = 20,
    Gamma_min: float = 0.0,
    Gamma_max: float = 3.0,
    n_Gamma: int = 20,
    J0: float = 0.0,
    J: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    model = TransverseFieldSK(J0=J0, J=J)
    
    Ts = np.linspace(T_min, T_max, n_T)
    Gammas = np.linspace(Gamma_min, Gamma_max, n_Gamma)
    
    m_map = np.zeros((n_T, n_Gamma))
    q_map = np.zeros((n_T, n_Gamma))
    
    for i, T in enumerate(Ts):
        for j, Gamma in enumerate(Gammas):
            result = model.solve_dmft(T, Gamma)
            m_map[i, j] = result.m
            q_map[i, j] = result.q
    
    return Ts, Gammas, m_map, q_map


def plot_quantum_phase_diagram(Ts: np.ndarray, Gammas: np.ndarray, 
                                m_map: np.ndarray, q_map: np.ndarray,
                                save_path: Optional[str] = None):
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    im = ax.pcolormesh(Gammas, Ts, m_map, shading='auto', cmap='viridis')
    ax.set_xlabel('Transverse Field Γ')
    ax.set_ylabel('Temperature T')
    ax.set_title('Magnetization m')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.pcolormesh(Gammas, Ts, q_map, shading='auto', cmap='plasma')
    ax.set_xlabel('Transverse Field Γ')
    ax.set_ylabel('Temperature T')
    ax.set_title('Glass Order Parameter q')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 0]
    for i in range(0, len(Ts), max(1, len(Ts)//5)):
        ax.plot(Gammas, m_map[i], 'o-', label=f'T={Ts[i]:.2f}')
    ax.set_xlabel('Transverse Field Γ')
    ax.set_ylabel('Magnetization m')
    ax.set_title('m vs Γ for fixed T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for i in range(0, len(Gammas), max(1, len(Gammas)//5)):
        ax.plot(Ts, q_map[:, i], 'o-', label=f'Γ={Gammas[i]:.2f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Order Parameter q')
    ax.set_title('q vs T for fixed Γ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_quantum_annealing_results(qa: QuantumAnnealing, results: dict, 
                                    save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    n_steps = len(results['linear']['mean'])
    steps = np.arange(n_steps)
    
    for schedule, data in results.items():
        ax.errorbar(steps, data['mean'], yerr=data['std'], 
                    fmt='o-', capsize=3, label=schedule)
    
    ax.set_xlabel('Annealing Step')
    ax.set_ylabel('Classical Energy')
    ax.set_title(f'Quantum Annealing Schedules Comparison (N={qa.N})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    final_energies = [data['mean'][-1] for data in results.values()]
    schedules = list(results.keys())
    ax.bar(schedules, final_energies, alpha=0.7)
    ax.set_ylabel('Final Energy')
    ax.set_title('Final Energy by Schedule')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def run_dmft_example():
    print("="*70)
    print("Quantum Spin Glass - DMFT Example")
    print("="*70)
    
    model = TransverseFieldSK(J0=0.0, J=1.0, seed=42)
    
    print("\nSolving DMFT at various points...")
    
    points = [
        (0.2, 0.0, "Low T, zero field"),
        (0.2, 1.0, "Low T, medium field"),
        (1.0, 0.0, "High T, zero field"),
        (0.2, 2.0, "Low T, strong field"),
    ]
    
    print(f"\n{'T':>8} {'Γ':>8} {'m':>10} {'q':>10} {'χ':>10} {'Conv':>6}")
    print("-"*60)
    
    for T, Gamma, desc in points:
        result = model.solve_dmft(T, Gamma)
        conv_str = "✓" if result.converged else "✗"
        print(f"{T:8.2f} {Gamma:8.2f} {result.m:10.4f} {result.q:10.4f} {result.X:10.4f} {conv_str:>6}")
    
    print("\nComputing quantum phase diagram...")
    Ts, Gammas, m_map, q_map = compute_quantum_phase_diagram(
        T_min=0.1, T_max=2.0, n_T=15,
        Gamma_min=0.0, Gamma_max=2.5, n_Gamma=15,
        J0=0.0, J=1.0
    )
    
    plot_quantum_phase_diagram(Ts, Gammas, m_map, q_map, 
                                save_path='quantum_phase_diagram.png')
    print("Phase diagram saved to: quantum_phase_diagram.png")
    
    print("\n" + "="*70)
    print("Quantum Annealing Example")
    print("="*70)
    
    N = 20
    print(f"\nCreating quantum annealer with N={N} spins...")
    qa = QuantumAnnealing(N=N, seed=42)
    qa.set_spinglass(J_std=1.0)
    
    print("Comparing annealing schedules (5 runs each)...")
    results = qa.compare_annealing_schedules(n_runs=3)
    
    plot_quantum_annealing_results(qa, results, save_path='quantum_annealing.png')
    print("Annealing comparison saved to: quantum_annealing.png")
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    run_dmft_example()
