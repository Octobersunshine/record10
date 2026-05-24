import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pickle


@dataclass
class SimResult:
    temperatures: np.ndarray
    q_mean: np.ndarray
    q_std: np.ndarray
    energy_mean: np.ndarray
    energy_std: np.ndarray
    specific_heat: np.ndarray
    susceptibility: np.ndarray
    exchange_rates: np.ndarray


class EdwardsAnderson:
    def __init__(self, L: int, seed: Optional[int] = None):
        self.L = L
        self.N = L * L
        self.rng = np.random.default_rng(seed)
        self._init_bonds()
        
    def _init_bonds(self):
        self.bonds = {}
        for i in range(self.L):
            for j in range(self.L):
                neighbors = []
                if i > 0:
                    neighbors.append(((i-1, j), self.rng.choice([-1, 1])))
                if i < self.L - 1:
                    neighbors.append(((i+1, j), self.rng.choice([-1, 1])))
                if j > 0:
                    neighbors.append(((i, j-1), self.rng.choice([-1, 1])))
                if j < self.L - 1:
                    neighbors.append(((i, j+1), self.rng.choice([-1, 1])))
                self.bonds[(i, j)] = neighbors
    
    def random_spins(self) -> np.ndarray:
        return self.rng.choice([-1, 1], size=(self.L, self.L))
    
    def energy(self, spins: np.ndarray) -> float:
        E = 0.0
        for (i, j), neighbors in self.bonds.items():
            for (ni, nj), J in neighbors:
                if (ni, nj) > (i, j):
                    E -= J * spins[i, j] * spins[ni, nj]
        return E
    
    def delta_energy(self, spins: np.ndarray, i: int, j: int) -> float:
        dE = 0.0
        for (ni, nj), J in self.bonds[(i, j)]:
            dE += 2 * J * spins[i, j] * spins[ni, nj]
        return dE
    
    def metropolis_step(self, spins: np.ndarray, T: float) -> np.ndarray:
        for _ in range(self.N):
            i = self.rng.integers(0, self.L)
            j = self.rng.integers(0, self.L)
            dE = self.delta_energy(spins, i, j)
            if dE <= 0 or self.rng.random() < np.exp(-dE / T):
                spins[i, j] *= -1
        return spins


class AdaptiveParallelTempering:
    def __init__(self, model: EdwardsAnderson, temperatures: np.ndarray, 
                 min_exchange_rate: float = 0.2, max_exchange_rate: float = 0.6,
                 seed: Optional[int] = None):
        self.model = model
        self.Ts = temperatures.copy()
        self.n_replicas = len(temperatures)
        self.min_exchange_rate = min_exchange_rate
        self.max_exchange_rate = max_exchange_rate
        self.rng = np.random.default_rng(seed)
        self.replicas = [model.random_spins() for _ in range(self.n_replicas)]
        
        self.exchange_accept = np.zeros(self.n_replicas - 1)
        self.exchange_total = np.zeros(self.n_replicas - 1)
        
        self.energy_sum = np.zeros(self.n_replicas)
        self.energy_sq_sum = np.zeros(self.n_replicas)
        self.sample_count = 0
        
        self.adaptation_history = []
    
    def step(self):
        for i in range(self.n_replicas):
            self.replicas[i] = self.model.metropolis_step(self.replicas[i], self.Ts[i])
        
        for i in range(self.n_replicas - 1):
            self.exchange_total[i] += 1
            E1 = self.model.energy(self.replicas[i])
            E2 = self.model.energy(self.replicas[i+1])
            T1, T2 = self.Ts[i], self.Ts[i+1]
            delta = (E1 - E2) * (1/T1 - 1/T2)
            
            if delta <= 0 or self.rng.random() < np.exp(-delta):
                self.replicas[i], self.replicas[i+1] = self.replicas[i+1], self.replicas[i]
                self.exchange_accept[i] += 1
        
        energies = self.get_energies()
        self.energy_sum += energies
        self.energy_sq_sum += energies**2
        self.sample_count += 1
    
    def get_energies(self) -> np.ndarray:
        return np.array([self.model.energy(r) for r in self.replicas])
    
    def get_spins(self) -> List[np.ndarray]:
        return [r.copy() for r in self.replicas]
    
    def exchange_rates(self) -> np.ndarray:
        return self.exchange_accept / np.maximum(self.exchange_total, 1)
    
    def reset_statistics(self):
        self.exchange_accept = np.zeros(self.n_replicas - 1)
        self.exchange_total = np.zeros(self.n_replicas - 1)
        self.energy_sum = np.zeros(self.n_replicas)
        self.energy_sq_sum = np.zeros(self.n_replicas)
        self.sample_count = 0
    
    def get_specific_heat(self) -> np.ndarray:
        if self.sample_count == 0:
            return np.zeros(self.n_replicas)
        E_mean = self.energy_sum / self.sample_count
        E2_mean = self.energy_sq_sum / self.sample_count
        Cv = (E2_mean - E_mean**2) / (self.Ts**2 * self.model.N)
        return Cv
    
    def get_energy_mean(self) -> np.ndarray:
        if self.sample_count == 0:
            return np.zeros(self.n_replicas)
        return self.energy_sum / self.sample_count / self.model.N
    
    def optimize_temperatures(self, n_cycles: int = 5) -> Tuple[np.ndarray, List[dict]]:
        print(f"\n{'='*60}")
        print(f"Starting adaptive temperature optimization ({n_cycles} cycles)")
        print(f"{'='*60}")
        
        history = []
        
        for cycle in range(n_cycles):
            self.reset_statistics()
            
            n_collect = max(500, 1000 // (cycle + 1))
            print(f"\nCycle {cycle+1}/{n_cycles}: Collecting statistics for {n_collect} steps...")
            
            for _ in range(n_collect):
                self.step()
            
            rates = self.exchange_rates()
            Cv = self.get_specific_heat()
            
            print(f"  Exchange rates: [{', '.join([f'{r:.2f}' for r in rates])}]")
            print(f"  Cv peak at T={self.Ts[np.argmax(Cv)]:.3f}, max Cv={np.max(Cv):.3f}")
            
            low_rates = np.where(rates < self.min_exchange_rate)[0]
            high_rates = np.where(rates > self.max_exchange_rate)[0]
            
            history.append({
                'cycle': cycle,
                'temperatures': self.Ts.copy(),
                'exchange_rates': rates.copy(),
                'specific_heat': Cv.copy()
            })
            
            if len(low_rates) == 0 and len(high_rates) == 0:
                print(f"  ✓ All exchange rates within target range ({self.min_exchange_rate:.0%}-{self.max_exchange_rate:.0%})")
                break
            
            if len(low_rates) > 0:
                print(f"  ! {len(low_rates)} gaps with exchange rate < {self.min_exchange_rate:.0%}")
            if len(high_rates) > 0:
                print(f"  ! {len(high_rates)} gaps with exchange rate > {self.max_exchange_rate:.0%}")
            
            self._adapt_temperatures(rates, Cv)
        
        self.adaptation_history = history
        return self.Ts, history
    
    def _adapt_temperatures(self, rates: np.ndarray, Cv: np.ndarray):
        T_new = self.Ts.copy()
        adjusted = False
        
        for i in range(len(rates)):
            if rates[i] < self.min_exchange_rate:
                factor = 0.85
                mid_T = self.Ts[i] * (self.Ts[i+1] / self.Ts[i]) ** 0.5
                
                if i == 0:
                    T_new[i+1] = mid_T
                elif i == len(rates) - 1:
                    T_new[i] = mid_T
                else:
                    if Cv[i] > Cv[i+1]:
                        T_new[i] = self.Ts[i] * (self.Ts[i+1] / self.Ts[i]) ** 0.3
                    else:
                        T_new[i+1] = self.Ts[i] * (self.Ts[i+1] / self.Ts[i]) ** 0.7
                adjusted = True
            
            elif rates[i] > self.max_exchange_rate:
                factor = 1.15
                if i == 0:
                    T_new[i+1] = min(T_new[i+1] * factor, self.Ts[-1])
                elif i == len(rates) - 1:
                    T_new[i] = max(T_new[i] / factor, self.Ts[0])
                else:
                    T_new[i] = max(T_new[i] / factor**0.5, self.Ts[0])
                    T_new[i+1] = min(T_new[i+1] * factor**0.5, self.Ts[-1])
                adjusted = True
        
        if adjusted:
            T_new = np.clip(T_new, self.Ts[0], self.Ts[-1])
            self.Ts = T_new
            print(f"  → Adjusted temperatures")
    
    def refine_around_peak(self, Cv: np.ndarray, n_add: int = 3):
        peak_idx = np.argmax(Cv)
        T_peak = self.Ts[peak_idx]
        
        T_left = self.Ts[max(0, peak_idx-1)]
        T_right = self.Ts[min(len(self.Ts)-1, peak_idx+1)]
        
        new_Ts = np.linspace(T_left, T_right, n_add + 2)[1:-1]
        
        Ts_combined = np.concatenate([self.Ts, new_Ts])
        Ts_combined.sort()
        
        self.Ts = Ts_combined
        self.n_replicas = len(self.Ts)
        
        n_new = self.n_replicas - len(self.replicas)
        for _ in range(n_new):
            self.replicas.append(self.model.random_spins())
        
        self.exchange_accept = np.zeros(self.n_replicas - 1)
        self.exchange_total = np.zeros(self.n_replicas - 1)
        
        print(f"  → Added {n_new} temperatures around peak at T={T_peak:.3f}")


def edwards_anderson_q(configs1: np.ndarray, configs2: Optional[np.ndarray] = None) -> float:
    if configs2 is None:
        n = len(configs1)
        if n < 2:
            return 0.0
        q_sum = 0.0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                q_sum += np.mean(configs1[i] * configs1[j])
                count += 1
        return q_sum / count if count > 0 else 0.0
    else:
        qs = [np.mean(c1 * c2) for c1, c2 in zip(configs1, configs2)]
        return np.mean(qs)


def run_adaptive_pt_simulation(
    L: int,
    T_min: float,
    T_max: float,
    n_replicas_initial: int,
    n_steps: int,
    n_equil: int,
    n_adapt_cycles: int = 5,
    sample_interval: int = 10,
    min_exchange_rate: float = 0.2,
    max_exchange_rate: float = 0.6,
    seed: Optional[int] = None
) -> Tuple[SimResult, AdaptiveParallelTempering, List[dict]]:
    
    model = EdwardsAnderson(L, seed=seed)
    
    Ts_initial = np.geomspace(T_min, T_max, n_replicas_initial)
    print(f"Initial temperature range: [{T_min:.2f}, {T_max:.2f}] with {n_replicas_initial} replicas")
    print(f"Target exchange rate range: [{min_exchange_rate:.0%}, {max_exchange_rate:.0%}]")
    
    pt = AdaptiveParallelTempering(
        model, Ts_initial,
        min_exchange_rate=min_exchange_rate,
        max_exchange_rate=max_exchange_rate,
        seed=seed+1 if seed else None
    )
    
    Ts_opt, adapt_history = pt.optimize_temperatures(n_cycles=n_adapt_cycles)
    n_replicas_final = len(Ts_opt)
    
    print(f"\n{'='*60}")
    print(f"Temperature optimization complete")
    print(f"  Final replicas: {n_replicas_final}")
    print(f"  Final temperatures: {Ts_opt}")
    print(f"{'='*60}")
    
    pt.reset_statistics()
    
    print(f"\nEquilibrating for {n_equil} steps...")
    for _ in range(n_equil):
        pt.step()
    
    pt.reset_statistics()
    
    print(f"Production run for {n_steps} steps...")
    all_energies = [[] for _ in range(n_replicas_final)]
    all_configs = [[] for _ in range(n_replicas_final)]
    
    for step in range(n_steps):
        pt.step()
        if step % sample_interval == 0:
            energies = pt.get_energies()
            configs = pt.get_spins()
            for i in range(n_replicas_final):
                all_energies[i].append(energies[i])
                all_configs[i].append(configs[i])
    
    final_exchange_rates = pt.exchange_rates()
    print(f"\nFinal exchange rates: [{', '.join([f'{r:.2f}' for r in final_exchange_rates])}]")
    
    q_mean = np.zeros(n_replicas_final)
    q_std = np.zeros(n_replicas_final)
    energy_mean = np.zeros(n_replicas_final)
    energy_std = np.zeros(n_replicas_final)
    specific_heat = np.zeros(n_replicas_final)
    
    for i in range(n_replicas_final):
        configs = np.array(all_configs[i])
        energies = np.array(all_energies[i])
        
        q_vals = []
        n_samples = len(configs)
        for a in range(0, n_samples, 2):
            if a + 1 < n_samples:
                q_vals.append(np.mean(configs[a] * configs[a+1]))
        
        q_mean[i] = np.mean(q_vals) if q_vals else 0
        q_std[i] = np.std(q_vals) if q_vals else 0
        energy_mean[i] = np.mean(energies) / model.N
        energy_std[i] = np.std(energies) / model.N
        
        E2_mean = np.mean(energies**2)
        E_mean = np.mean(energies)
        specific_heat[i] = (E2_mean - E_mean**2) / (Ts_opt[i]**2 * model.N)
    
    susceptibility = (q_mean - q_mean**2) / Ts_opt
    
    result = SimResult(
        temperatures=Ts_opt,
        q_mean=q_mean,
        q_std=q_std,
        energy_mean=energy_mean,
        energy_std=energy_std,
        specific_heat=specific_heat,
        susceptibility=susceptibility,
        exchange_rates=final_exchange_rates
    )
    
    return result, pt, adapt_history


def plot_results(result: SimResult, L: int, save_path: Optional[str] = None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    ax.errorbar(result.temperatures, result.q_mean, yerr=result.q_std, fmt='o-', capsize=5)
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Edwards-Anderson q')
    ax.set_title(f'Order Parameter (L={L})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.errorbar(result.temperatures, result.energy_mean, yerr=result.energy_std, fmt='o-', capsize=5)
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Energy per spin')
    ax.set_title(f'Energy (L={L})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(result.temperatures, result.specific_heat, 'o-')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Specific Heat Cv')
    ax.set_title(f'Specific Heat (L={L})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(result.temperatures, result.susceptibility, 'o-')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Susceptibility χ')
    ax.set_title(f'Susceptibility (L={L})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    T_mid = (result.temperatures[:-1] + result.temperatures[1:]) / 2
    ax.plot(T_mid, result.exchange_rates, 'o-', color='green')
    ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Min (20%)')
    ax.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='Max (60%)')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Exchange Rate')
    ax.set_title('Exchange Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    ax = axes[1, 2]
    ax.plot(result.temperatures[:-1], np.diff(result.temperatures), 'o-', color='purple')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('ΔT')
    ax.set_title('Temperature Spacing')
    ax.grid(True, alpha=0.3)
    
    peak_idx = np.argmax(result.susceptibility)
    T_c_est = result.temperatures[peak_idx]
    print(f"\nEstimated transition temperature from susceptibility peak: T_c ≈ {T_c_est:.3f}")
    
    cv_idx = np.argmax(result.specific_heat)
    print(f"Specific heat peak at T ≈ {result.temperatures[cv_idx]:.3f}")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    
    return T_c_est


def plot_adaptation_history(history: List[dict], save_path: Optional[str] = None):
    n_cycles = len(history)
    fig, axes = plt.subplots(n_cycles, 2, figsize=(12, 4*n_cycles))
    
    if n_cycles == 1:
        axes = axes.reshape(1, -1)
    
    for idx, data in enumerate(history):
        cycle = data['cycle']
        Ts = data['temperatures']
        rates = data['exchange_rates']
        Cv = data['specific_heat']
        
        ax = axes[idx, 0]
        T_mid = (Ts[:-1] + Ts[1:]) / 2
        ax.plot(T_mid, rates, 'o-', color='green')
        ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=0.6, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('T')
        ax.set_ylabel('Exchange Rate')
        ax.set_title(f'Cycle {cycle+1} - Exchange Rates')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        ax = axes[idx, 1]
        ax.plot(Ts, Cv, 'o-', color='blue')
        ax.set_xlabel('T')
        ax.set_ylabel('Cv')
        ax.set_title(f'Cycle {cycle+1} - Specific Heat')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def find_phase_transition(
    L_list: List[int],
    T_min: float = 0.5,
    T_max: float = 2.0,
    n_replicas_initial: int = 16,
    n_steps: int = 5000,
    n_equil: int = 2000,
    n_adapt_cycles: int = 5,
    base_seed: int = 42
) -> dict:
    results = {}
    for L in L_list:
        print(f"\n{'='*60}")
        print(f"Running simulation for L={L}")
        print(f"{'='*60}")
        result, pt, history = run_adaptive_pt_simulation(
            L=L,
            T_min=T_min,
            T_max=T_max,
            n_replicas_initial=n_replicas_initial,
            n_steps=n_steps,
            n_equil=n_equil,
            n_adapt_cycles=n_adapt_cycles,
            seed=base_seed + L
        )
        results[L] = {'result': result, 'pt': pt, 'history': history}
        
        plot_adaptation_history(history, save_path=f'adaptation_L{L}.png')
        T_c = plot_results(result, L, save_path=f'spinglass_L{L}.png')
        results[L]['T_c_est'] = T_c
    
    return results


if __name__ == "__main__":
    L_list = [8]
    results = find_phase_transition(
        L_list=L_list,
        T_min=0.3,
        T_max=2.0,
        n_replicas_initial=12,
        n_steps=5000,
        n_equil=2000,
        n_adapt_cycles=4
    )
    
    with open('spinglass_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nSimulation complete! Results saved.")
