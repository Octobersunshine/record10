import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional


class Reaction:
    def __init__(self, propensity: Callable, stoichiometry: np.ndarray, name: str = ""):
        self.propensity = propensity
        self.stoichiometry = stoichiometry
        self.name = name


class TauLeaping:
    def __init__(self, species_names: List[str], epsilon: float = 0.03, nc: int = 10):
        self.species_names = species_names
        self.reactions: List[Reaction] = []
        self.time_points: List[float] = []
        self.state_history: List[np.ndarray] = []
        self.epsilon = epsilon
        self.nc = nc
        self.negative_count = 0
        self.correction_count = 0
    
    def add_reaction(self, reaction: Reaction):
        self.reactions.append(reaction)
    
    def _compute_propensities(self, state: np.ndarray, t: float) -> np.ndarray:
        return np.array([rxn.propensity(state, t) for rxn in self.reactions])
    
    def _select_tau(self, propensities: np.ndarray, state: np.ndarray) -> float:
        total_propensity = propensities.sum()
        if total_propensity <= 0:
            return float('inf')
        
        num_species = len(state)
        num_reactions = len(self.reactions)
        
        mu = np.zeros(num_species)
        sigma_sq = np.zeros(num_species)
        
        for j in range(num_reactions):
            for i in range(num_species):
                mu[i] += self.reactions[j].stoichiometry[i] * propensities[j]
                sigma_sq[i] += (self.reactions[j].stoichiometry[i] ** 2) * propensities[j]
        
        tau_candidates = []
        for i in range(num_species):
            if abs(mu[i]) > 0 or sigma_sq[i] > 0:
                numerator = max(self.epsilon * state[i], 1.0)
                if abs(mu[i]) > 0:
                    tau1 = numerator / abs(mu[i])
                    tau_candidates.append(tau1)
                if sigma_sq[i] > 0:
                    tau2 = (numerator ** 2) / sigma_sq[i]
                    tau_candidates.append(tau2)
        
        if len(tau_candidates) == 0:
            return 1.0 / total_propensity
        
        tau = min(tau_candidates)
        
        tau_max = 10.0 / total_propensity
        tau = min(tau, tau_max)
        
        return tau
    
    def _compute_max_reactions(self, state: np.ndarray, tau: float) -> np.ndarray:
        num_reactions = len(self.reactions)
        max_k = np.full(num_reactions, float('inf'))
        
        for j in range(num_reactions):
            stoich = self.reactions[j].stoichiometry
            for i in range(len(state)):
                if stoich[i] < 0:
                    max_possible = state[i] // abs(stoich[i]) if state[i] > 0 else 0
                    max_k[j] = min(max_k[j], max_possible)
        
        return max_k
    
    def _sample_reaction_counts(self, propensities: np.ndarray, tau: float, 
                                 state: np.ndarray, max_k: np.ndarray) -> np.ndarray:
        num_reactions = len(self.reactions)
        k = np.zeros(num_reactions, dtype=int)
        
        for j in range(num_reactions):
            if propensities[j] <= 0:
                k[j] = 0
                continue
            
            lambda_j = propensities[j] * tau
            
            if lambda_j > max_k[j] and max_k[j] < self.nc:
                if max_k[j] <= 0:
                    k[j] = 0
                else:
                    p = lambda_j / (lambda_j + max_k[j]) if (lambda_j + max_k[j]) > 0 else 0
                    p = min(max(p, 0), 1)
                    k[j] = np.random.binomial(int(max_k[j]), p)
            else:
                k[j] = np.random.poisson(lambda_j)
                k[j] = int(min(k[j], max_k[j]))
        
        return k
    
    def _apply_state_update(self, state: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, bool]:
        new_state = state.copy()
        had_negative = False
        
        for j in range(len(self.reactions)):
            if k[j] > 0:
                new_state += k[j] * self.reactions[j].stoichiometry
        
        negative_mask = new_state < 0
        if negative_mask.any():
            had_negative = True
            self.negative_count += 1
            
            for j in range(len(self.reactions)):
                stoich = self.reactions[j].stoichiometry
                for i in range(len(new_state)):
                    if stoich[i] < 0 and new_state[i] < 0:
                        over_consumed = abs(new_state[i])
                        max_reduce = over_consumed // abs(stoich[i])
                        if max_reduce > 0:
                            k[j] = max(0, k[j] - max_reduce)
                            self.correction_count += 1
            
            new_state = state.copy()
            for j in range(len(self.reactions)):
                if k[j] > 0:
                    new_state += k[j] * self.reactions[j].stoichiometry
        
        new_state = np.maximum(new_state, 0)
        return new_state, had_negative
    
    def simulate(self, initial_state: np.ndarray, t_max: float, 
                 max_steps: int = 100000, use_correction: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        state = initial_state.copy().astype(float)
        t = 0.0
        self.time_points = [t]
        self.state_history = [state.copy()]
        self.negative_count = 0
        self.correction_count = 0
        
        step = 0
        while t < t_max and step < max_steps:
            propensities = self._compute_propensities(state, t)
            total_propensity = propensities.sum()
            
            if total_propensity <= 0:
                break
            
            tau = self._select_tau(propensities, state)
            
            if t + tau > t_max:
                tau = t_max - t
            
            if tau * total_propensity < 1.0:
                tau = np.random.exponential(1.0 / total_propensity)
                if t + tau > t_max:
                    break
                reaction_index = np.random.choice(len(self.reactions), p=propensities / total_propensity)
                state += self.reactions[reaction_index].stoichiometry
                state = np.maximum(state, 0)
            else:
                max_k = self._compute_max_reactions(state, tau)
                k = self._sample_reaction_counts(propensities, tau, state, max_k)
                
                if use_correction:
                    state, had_negative = self._apply_state_update(state, k)
                else:
                    for j in range(len(self.reactions)):
                        if k[j] > 0:
                            state += k[j] * self.reactions[j].stoichiometry
                    state = np.maximum(state, 0)
            
            t += tau
            self.time_points.append(t)
            self.state_history.append(state.copy())
            step += 1
        
        return np.array(self.time_points), np.array(self.state_history)
    
    def simulate_midpoint(self, initial_state: np.ndarray, t_max: float, 
                          max_steps: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        state = initial_state.copy().astype(float)
        t = 0.0
        self.time_points = [t]
        self.state_history = [state.copy()]
        self.negative_count = 0
        self.correction_count = 0
        
        step = 0
        while t < t_max and step < max_steps:
            propensities = self._compute_propensities(state, t)
            total_propensity = propensities.sum()
            
            if total_propensity <= 0:
                break
            
            tau = self._select_tau(propensities, state)
            
            if t + tau > t_max:
                tau = t_max - t
            
            if tau * total_propensity < 1.0:
                tau = np.random.exponential(1.0 / total_propensity)
                if t + tau > t_max:
                    break
                reaction_index = np.random.choice(len(self.reactions), p=propensities / total_propensity)
                state += self.reactions[reaction_index].stoichiometry
                state = np.maximum(state, 0)
            else:
                tau_half = tau / 2
                
                max_k_half = self._compute_max_reactions(state, tau_half)
                k_half = self._sample_reaction_counts(propensities, tau_half, state, max_k_half)
                mid_state, _ = self._apply_state_update(state, k_half)
                mid_state = np.maximum(mid_state, 0)
                
                mid_propensities = self._compute_propensities(mid_state, t + tau_half)
                
                max_k = self._compute_max_reactions(state, tau)
                k = self._sample_reaction_counts(mid_propensities, tau, state, max_k)
                state, _ = self._apply_state_update(state, k)
                state = np.maximum(state, 0)
            
            t += tau
            self.time_points.append(t)
            self.state_history.append(state.copy())
            step += 1
        
        return np.array(self.time_points), np.array(self.state_history)
    
    def plot_results(self, figsize: Tuple[int, int] = (12, 8), title: str = None):
        time_array = np.array(self.time_points)
        state_array = np.array(self.state_history)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, name in enumerate(self.species_names):
            ax.step(time_array, state_array[:, i], where='post', label=name, linewidth=2)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Molecule Count', fontsize=12)
        if title is None:
            title = 'Gene Regulation Network - Tau-Leaping Simulation'
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_simple_degradation_network() -> TauLeaping:
    species_names = ['A', 'B']
    simulator = TauLeaping(species_names, epsilon=0.05)
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 1.0 * s[0],
        stoichiometry=np.array([-1, 1]),
        name='A → B'
    ))
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.1 * s[1],
        stoichiometry=np.array([0, -1]),
        name='B → ∅'
    ))
    
    return simulator


def create_gene_expression_network() -> TauLeaping:
    species_names = ['DNA', 'mRNA', 'Protein', 'Protein_Dimer']
    simulator = TauLeaping(species_names, epsilon=0.03)
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.5 * s[0],
        stoichiometry=np.array([0, 1, 0, 0]),
        name='Transcription'
    ))
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.1 * s[1],
        stoichiometry=np.array([0, -1, 0, 0]),
        name='mRNA degradation'
    ))
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.5 * s[1],
        stoichiometry=np.array([0, 0, 1, 0]),
        name='Translation'
    ))
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.05 * s[2],
        stoichiometry=np.array([0, 0, -1, 0]),
        name='Protein degradation'
    ))
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.001 * s[2] * (s[2] - 1) if s[2] >= 2 else 0,
        stoichiometry=np.array([0, 0, -2, 1]),
        name='Dimerization'
    ))
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.01 * s[3],
        stoichiometry=np.array([0, 0, 2, -1]),
        name='Dimer dissociation'
    ))
    
    return simulator


def main():
    print("Tau-Leaping Simulation with Negative Population Correction")
    print("=" * 70)
    
    print("\n1. Simple Degradation Network - Testing Corrections")
    print("-" * 50)
    simulator1 = create_simple_degradation_network()
    initial_state1 = np.array([1000, 0])
    t_max1 = 50.0
    
    print("\nWithout correction (may produce negatives):")
    times1_no_corr, states1_no_corr = simulator1.simulate(initial_state1, t_max1, use_correction=False)
    print(f"Steps: {len(times1_no_corr)}, Negatives detected: {simulator1.negative_count}")
    print(f"Final state: A={states1_no_corr[-1,0]}, B={states1_no_corr[-1,1]}")
    
    print("\nWith negative correction:")
    times1_corr, states1_corr = simulator1.simulate(initial_state1, t_max1, use_correction=True)
    print(f"Steps: {len(times1_corr)}, Negatives detected: {simulator1.negative_count}")
    print(f"Corrections applied: {simulator1.correction_count}")
    print(f"Final state: A={states1_corr[-1,0]}, B={states1_corr[-1,1]}")
    
    print("\n2. Gene Expression Network")
    print("-" * 50)
    simulator2 = create_gene_expression_network()
    initial_state2 = np.array([1, 0, 0, 0])
    t_max2 = 100.0
    
    times2, states2 = simulator2.simulate(initial_state2, t_max2)
    print(f"Simulation completed: {len(times2)} steps")
    print(f"Negatives detected: {simulator2.negative_count}")
    print(f"Corrections applied: {simulator2.correction_count}")
    print(f"Final state: DNA={states2[-1,0]}, mRNA={states2[-1,1]}, "
          f"Protein={states2[-1,2]}, Dimer={states2[-1,3]}")
    
    print("\n3. Midpoint Tau-Leaping (more accurate)")
    print("-" * 50)
    times3, states3 = simulator2.simulate_midpoint(initial_state2, t_max2)
    print(f"Simulation completed: {len(times3)} steps")
    print(f"Final state: DNA={states3[-1,0]}, mRNA={states3[-1,1]}, "
          f"Protein={states3[-1,2]}, Dimer={states3[-1,3]}")
    
    print("\n" + "=" * 70)
    print("Correction Strategies Implemented:")
    print("1. Poisson boundary checking - limit reactions to prevent over-consumption")
    print("2. Binomial sampling for reactions near depletion")
    print("3. Post-leap correction - detect and fix negative populations")
    print("4. Automatic fallback to SSA for small total propensity")
    print("5. Midpoint method - improved accuracy with intermediate evaluation")
    print("=" * 70)


if __name__ == "__main__":
    main()
