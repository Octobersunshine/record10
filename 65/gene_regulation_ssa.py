import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable


class Reaction:
    def __init__(self, propensity: Callable, stoichiometry: np.ndarray, name: str = ""):
        self.propensity = propensity
        self.stoichiometry = stoichiometry
        self.name = name


class GillespieSSA:
    def __init__(self, species_names: List[str]):
        self.species_names = species_names
        self.reactions: List[Reaction] = []
        self.time_points: List[float] = []
        self.state_history: List[np.ndarray] = []
    
    def add_reaction(self, reaction: Reaction):
        self.reactions.append(reaction)
    
    def simulate(self, initial_state: np.ndarray, t_max: float, max_steps: int = 1000000) -> Tuple[np.ndarray, np.ndarray]:
        state = initial_state.copy()
        t = 0.0
        self.time_points = [t]
        self.state_history = [state.copy()]
        
        step = 0
        while t < t_max and step < max_steps:
            propensities = np.array([rxn.propensity(state, t) for rxn in self.reactions])
            total_propensity = propensities.sum()
            
            if total_propensity <= 0:
                break
            
            tau = np.random.exponential(1.0 / total_propensity)
            t += tau
            
            if t > t_max:
                break
            
            reaction_index = np.random.choice(len(self.reactions), p=propensities / total_propensity)
            state += self.reactions[reaction_index].stoichiometry
            state = np.maximum(state, 0)
            
            self.time_points.append(t)
            self.state_history.append(state.copy())
            step += 1
        
        return np.array(self.time_points), np.array(self.state_history)
    
    def plot_results(self, figsize: Tuple[int, int] = (12, 8)):
        time_array = np.array(self.time_points)
        state_array = np.array(self.state_history)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, name in enumerate(self.species_names):
            ax.step(time_array, state_array[:, i], where='post', label=name, linewidth=2)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Molecule Count', fontsize=12)
        ax.set_title('Gene Regulation Network - Gillespie SSA Simulation', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_gene_expression_network(k0: float = 0.5, k1: float = 0.005, k2: float = 0.1, k3: float = 0.01) -> GillespieSSA:
    species_names = ['DNA', 'mRNA', 'Protein', 'Protein_Dimer']
    ssa = GillespieSSA(species_names)
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: k0 * s[0],
        stoichiometry=np.array([0, 1, 0, 0]),
        name='Transcription'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: k2 * s[1],
        stoichiometry=np.array([0, -1, 0, 0]),
        name='mRNA degradation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: k1 * s[1],
        stoichiometry=np.array([0, 0, 1, 0]),
        name='Translation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: k3 * s[2],
        stoichiometry=np.array([0, 0, -1, 0]),
        name='Protein degradation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 0.001 * s[2] * (s[2] - 1) if s[2] >= 2 else 0,
        stoichiometry=np.array([0, 0, -2, 1]),
        name='Dimerization'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 0.01 * s[3],
        stoichiometry=np.array([0, 0, 2, -1]),
        name='Dimer dissociation'
    ))
    
    return ssa


def create_repressilator_network() -> GillespieSSA:
    species_names = ['X', 'Y', 'Z', 'X_mRNA', 'Y_mRNA', 'Z_mRNA']
    ssa = GillespieSSA(species_names)
    
    alpha = 10.0
    alpha0 = 0.5
    beta = 1.0
    n = 2
    kd = 1.0
    
    def hill_repression(repressor: int, alpha: float, alpha0: float, n: int, kd: float) -> float:
        return alpha0 + (alpha - alpha0) / (1 + (repressor / kd) ** n)
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: hill_repression(s[2], alpha, alpha0, n, kd),
        stoichiometry=np.array([0, 0, 0, 1, 0, 0]),
        name='X transcription'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: hill_repression(s[0], alpha, alpha0, n, kd),
        stoichiometry=np.array([0, 0, 0, 0, 1, 0]),
        name='Y transcription'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: hill_repression(s[1], alpha, alpha0, n, kd),
        stoichiometry=np.array([0, 0, 0, 0, 0, 1]),
        name='Z transcription'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: beta * s[3],
        stoichiometry=np.array([0, 0, 0, -1, 0, 0]),
        name='X mRNA degradation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: beta * s[4],
        stoichiometry=np.array([0, 0, 0, 0, -1, 0]),
        name='Y mRNA degradation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: beta * s[5],
        stoichiometry=np.array([0, 0, 0, 0, 0, -1]),
        name='Z mRNA degradation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 1.0 * s[3],
        stoichiometry=np.array([1, 0, 0, 0, 0, 0]),
        name='X translation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 1.0 * s[4],
        stoichiometry=np.array([0, 1, 0, 0, 0, 0]),
        name='Y translation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 1.0 * s[5],
        stoichiometry=np.array([0, 0, 1, 0, 0, 0]),
        name='Z translation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 0.1 * s[0],
        stoichiometry=np.array([-1, 0, 0, 0, 0, 0]),
        name='X degradation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 0.1 * s[1],
        stoichiometry=np.array([0, -1, 0, 0, 0, 0]),
        name='Y degradation'
    ))
    
    ssa.add_reaction(Reaction(
        propensity=lambda s, t: 0.1 * s[2],
        stoichiometry=np.array([0, 0, -1, 0, 0, 0]),
        name='Z degradation'
    ))
    
    return ssa


def main():
    print("Gene Regulation Network Simulation using Gillespie SSA")
    print("=" * 60)
    
    print("\n1. Basic Gene Expression Network")
    print("-" * 40)
    ssa1 = create_gene_expression_network()
    initial_state1 = np.array([1, 0, 0, 0])
    t_max1 = 100.0
    times1, states1 = ssa1.simulate(initial_state1, t_max1)
    print(f"Simulation completed: {len(times1)} reactions")
    print(f"Final state: DNA={states1[-1,0]}, mRNA={states1[-1,1]}, Protein={states1[-1,2]}, Dimer={states1[-1,3]}")
    ssa1.plot_results()
    
    print("\n2. Repressilator Network")
    print("-" * 40)
    ssa2 = create_repressilator_network()
    initial_state2 = np.array([0, 2, 0, 10, 0, 10])
    t_max2 = 200.0
    times2, states2 = ssa2.simulate(initial_state2, t_max2)
    print(f"Simulation completed: {len(times2)} reactions")
    print(f"Final state: X={states2[-1,0]}, Y={states2[-1,1]}, Z={states2[-1,2]}")
    ssa2.plot_results()


if __name__ == "__main__":
    main()
