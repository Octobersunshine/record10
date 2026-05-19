import numpy as np
import matplotlib.pyplot as plt
from tau_leaping import TauLeaping, Reaction


def create_negative_prone_network() -> Tuple[TauLeaping, np.ndarray]:
    species_names = ['Substrate', 'Product', 'Enzyme']
    simulator = TauLeaping(species_names, epsilon=0.2)
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.5 * s[0] * s[2],
        stoichiometry=np.array([-1, 1, 0]),
        name='S + E → P + E'
    ))
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.01 * s[1],
        stoichiometry=np.array([1, -1, 0]),
        name='P → S'
    ))
    
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.1 * s[0],
        stoichiometry=np.array([-1, 0, 0]),
        name='S degradation'
    ))
    
    initial_state = np.array([50, 0, 5])
    
    return simulator, initial_state


class NaiveTauLeaping:
    def __init__(self, species_names, tau_fixed=0.1):
        self.species_names = species_names
        self.reactions = []
        self.time_points = []
        self.state_history = []
        self.tau_fixed = tau_fixed
    
    def add_reaction(self, reaction):
        self.reactions.append(reaction)
    
    def simulate(self, initial_state, t_max):
        state = initial_state.copy().astype(float)
        t = 0.0
        self.time_points = [t]
        self.state_history = [state.copy()]
        
        negative_events = []
        
        while t < t_max:
            propensities = np.array([rxn.propensity(state, t) for rxn in self.reactions])
            total_propensity = propensities.sum()
            
            if total_propensity <= 0:
                break
            
            k = np.random.poisson(propensities * self.tau_fixed)
            
            for j in range(len(self.reactions)):
                if k[j] > 0:
                    state += k[j] * self.reactions[j].stoichiometry
            
            negative_mask = state < 0
            if negative_mask.any():
                negative_events.append((t, state.copy()))
            
            state = np.maximum(state, 0)
            
            t += self.tau_fixed
            self.time_points.append(t)
            self.state_history.append(state.copy())
        
        return np.array(self.time_points), np.array(self.state_history), negative_events


def demo_naive_vs_corrected():
    print("=" * 70)
    print("Demonstration: Negative Population in Tau-Leaping")
    print("=" * 70)
    
    species_names = ['A', 'B']
    initial_state = np.array([10, 0])
    
    naive_sim = NaiveTauLeaping(species_names, tau_fixed=2.0)
    naive_sim.add_reaction(Reaction(
        propensity=lambda s, t: 1.0 * s[0],
        stoichiometry=np.array([-1, 1]),
        name='A → B'
    ))
    
    print("\n1. Naive Tau-Leaping (large fixed step)")
    print("-" * 50)
    times_naive, states_naive, negatives = naive_sim.simulate(initial_state, t_max=20.0)
    print(f"Simulation steps: {len(times_naive)}")
    print(f"Negative population events: {len(negatives)}")
    for t, state in negatives[:5]:
        print(f"  t={t:.2f}: A={state[0]:.1f}, B={state[1]:.1f}")
    
    simulator, _ = create_negative_prone_network()
    simulator.reactions = []
    simulator.species_names = ['A', 'B']
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 1.0 * s[0],
        stoichiometry=np.array([-1, 1]),
        name='A → B'
    ))
    
    print("\n2. Corrected Tau-Leaping (adaptive step + corrections)")
    print("-" * 50)
    times_corr, states_corr = simulator.simulate(initial_state, t_max=20.0)
    print(f"Simulation steps: {len(times_corr)}")
    print(f"Negative population detected: {simulator.negative_count}")
    print(f"Corrections applied: {simulator.correction_count}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].step(times_naive, states_naive[:, 0], where='post', label='A', linewidth=2)
    axes[0].step(times_naive, states_naive[:, 1], where='post', label='B', linewidth=2)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero line')
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Population', fontsize=12)
    axes[0].set_title('Naive Tau-Leaping (Large Fixed Step)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].step(times_corr, states_corr[:, 0], where='post', label='A', linewidth=2)
    axes[1].step(times_corr, states_corr[:, 1], where='post', label='B', linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero line')
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Population', fontsize=12)
    axes[1].set_title('Corrected Tau-Leaping', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tau_leaping_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to 'tau_leaping_comparison.png'")


def demo_correction_strategies():
    print("\n" + "=" * 70)
    print("Detailed Comparison of Correction Strategies")
    print("=" * 70)
    
    species_names = ['Substrate', 'Product', 'Enzyme']
    initial_state = np.array([20, 0, 2])
    
    simulator = TauLeaping(species_names, epsilon=0.1)
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.3 * s[0] * s[2],
        stoichiometry=np.array([-1, 1, 0]),
        name='S + E → P + E'
    ))
    simulator.add_reaction(Reaction(
        propensity=lambda s, t: 0.05 * s[1],
        stoichiometry=np.array([1, -1, 0]),
        name='P → S'
    ))
    
    strategies = [
        ('Without correction', False),
        ('With boundary checking', True),
        ('Midpoint method (most accurate)', None)
    ]
    
    results = []
    
    for name, use_correction in strategies:
        print(f"\n{name}:")
        print("-" * 40)
        
        if name == 'Midpoint method (most accurate)':
            times, states = simulator.simulate_midpoint(initial_state, t_max=50.0)
        else:
            times, states = simulator.simulate(initial_state, t_max=50.0, use_correction=use_correction)
        
        min_state = states.min(axis=0)
        print(f"  Steps: {len(times)}")
        print(f"  Min populations: S={min_state[0]:.1f}, P={min_state[1]:.1f}, E={min_state[2]:.1f}")
        print(f"  Final state: S={states[-1,0]:.1f}, P={states[-1,1]:.1f}, E={states[-1,2]:.1f}")
        print(f"  Negatives detected: {simulator.negative_count}")
        print(f"  Corrections applied: {simulator.correction_count}")
        
        results.append((name, times, states))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, times, states) in enumerate(results):
        axes[idx].step(times, states[:, 0], where='post', label='Substrate', linewidth=2)
        axes[idx].step(times, states[:, 1], where='post', label='Product', linewidth=2)
        axes[idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[idx].set_xlabel('Time', fontsize=10)
        axes[idx].set_ylabel('Population', fontsize=10)
        axes[idx].set_title(name, fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correction_strategies.png', dpi=150, bbox_inches='tight')
    print("\nStrategy comparison plot saved to 'correction_strategies.png'")


def explain_correction_methods():
    print("\n" + "=" * 70)
    print("Explanation of Correction Methods")
    print("=" * 70)
    
    print("""
1. POISSON BOUNDARY CHECKING
   Problem: Poisson sampling can produce reaction counts that consume
            more molecules than available.
   Solution: Calculate maximum possible reactions for each species:
             max_k[j] = floor(state[i] / |stoichiometry[i]|)
             Then cap Poisson samples: k[j] = min(poisson, max_k[j])

2. BINOMIAL SAMPLING FOR NEAR-DEPLETION REACTIONS
   Problem: When lambda > max_k[j], Poisson is inappropriate.
   Solution: Switch to binomial sampling when max_k[j] < nc:
             k ~ Binomial(max_k[j], p)
             where p = lambda / (lambda + max_k[j])

3. POST-LEAP CORRECTION
   Problem: Even with precautions, negatives can occur (combined effects).
   Solution: Detect negatives after leap, then back-calculate how many
             reactions need to be reduced to eliminate negatives.

4. ADAPTIVE STEP SIZE SELECTION
   Problem: Fixed large tau is the main cause of negatives.
   Solution: Use Gillespie's tau-selection formula:
             tau = min_i( epsilon * x[i] / |mu[i]|, (epsilon * x[i])^2 / sigma[i]^2 )
             This ensures state changes are bounded by epsilon fraction.

5. AUTOMATIC SSA FALLBACK
   Problem: When expected reactions < 1, tau-leaping is wasteful and inaccurate.
   Solution: Fall back to exact SSA when tau * total_propensity < 1

6. MIDPOINT METHOD
   Problem: Propensities change during the leap, introducing error.
   Solution: Evaluate propensities at midpoint:
             1. Take half-leap to get intermediate state
             2. Evaluate propensities at midpoint
             3. Use midpoint propensities for the full leap
    """)


if __name__ == "__main__":
    demo_naive_vs_corrected()
    demo_correction_strategies()
    explain_correction_methods()
