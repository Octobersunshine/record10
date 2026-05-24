import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from go_model import GoModel
from amber_forcefield import GoAmberHybrid


class Replica:
    def __init__(self, 
                 replica_id: int, 
                 temperature: float,
                 simulation: GoModel):
        self.id = replica_id
        self.T = temperature
        self.sim = simulation
        self.sim.T = temperature
        self.current_energy = self.compute_energy()
        self.Q_history = []
        self.energy_history = []
    
    def compute_energy(self):
        if hasattr(self.sim, 'compute_energy'):
            return self.sim.compute_energy()
        else:
            return self._compute_go_energy()
    
    def _compute_go_energy(self):
        energy = 0.0
        for idx, (i, j) in enumerate(self.sim.native_contacts):
            vec = self.sim.positions[j] - self.sim.positions[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                continue
            r0 = self.sim.native_distances[idx]
            sig = r0 / (2 ** (1/6))
            r6 = (sig / dist) ** 6
            r12 = r6 * r6
            energy += 4 * self.sim.epsilon * (r12 - r6)
        return energy
    
    def run_md(self, n_steps: int):
        for _ in range(n_steps):
            self.sim.langevin_step()
        self.current_energy = self.compute_energy()
        self.Q_history.append(self.sim.compute_native_contact_fraction())
        self.energy_history.append(self.current_energy)
    
    def get_state(self):
        return {
            'positions': self.sim.positions.copy(),
            'velocities': self.sim.velocities.copy(),
            'energy': self.current_energy
        }
    
    def set_state(self, state: dict):
        self.sim.positions = state['positions'].copy()
        self.sim.velocities = state['velocities'].copy()
        self.current_energy = state['energy']


class REMD:
    def __init__(self,
                 num_replicas: int,
                 T_min: float,
                 T_max: float,
                 num_beads: int,
                 native_contacts: np.ndarray,
                 native_distances: np.ndarray,
                 native_positions: np.ndarray = None,
                 use_hybrid_ff: bool = False,
                 scheduler: str = 'geometric'):
        
        self.num_replicas = num_replicas
        self.T_min = T_min
        self.T_max = T_max
        
        if scheduler == 'geometric':
            self.temperatures = np.exp(
                np.linspace(np.log(T_min), np.log(T_max), num_replicas)
            )
        elif scheduler == 'linear':
            self.temperatures = np.linspace(T_min, T_max, num_replicas)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        
        self.replicas = []
        for i in range(num_replicas):
            if use_hybrid_ff and native_positions is not None:
                sim = GoAmberHybrid(
                    num_beads=num_beads,
                    native_contacts=native_contacts,
                    native_distances=native_distances,
                    native_positions=native_positions,
                    temperature=self.temperatures[i]
                )
            else:
                sim = GoModel(
                    num_beads=num_beads,
                    native_contacts=native_contacts,
                    native_distances=native_distances,
                    temperature=self.temperatures[i],
                    auto_gamma=True,
                    gamma_model='stokes_einstein'
                )
            
            replica = Replica(i, self.temperatures[i], sim)
            self.replicas.append(replica)
        
        self.exchange_attempts = 0
        self.exchange_accepts = np.zeros((num_replicas, num_replicas))
        self.trajectories = [[] for _ in range(num_replicas)]
        self.Q_trajectories = [[] for _ in range(num_replicas)]
        self.energy_trajectories = [[] for _ in range(num_replicas)]
        self.replica_positions = np.arange(num_replicas)
        
    def attempt_exchange(self, i: int, j: int) -> bool:
        Ti = self.replicas[i].T
        Tj = self.replicas[j].T
        Ei = self.replicas[i].current_energy
        Ej = self.replicas[j].current_energy
        
        beta_i = 1.0 / Ti
        beta_j = 1.0 / Tj
        
        delta = (beta_j - beta_i) * (Ei - Ej)
        
        if delta <= 0 or np.random.random() < np.exp(-delta):
            state_i = self.replicas[i].get_state()
            state_j = self.replicas[j].get_state()
            self.replicas[i].set_state(state_j)
            self.replicas[j].set_state(state_i)
            
            self.exchange_accepts[i, j] += 1
            self.exchange_accepts[j, i] += 1
            
            self.replica_positions[i], self.replica_positions[j] = \
                self.replica_positions[j], self.replica_positions[i]
            
            return True
        return False
    
    def run_remd(self,
                 n_cycles: int,
                 n_steps_per_cycle: int,
                 exchange_interval: int = 1,
                 record_interval: int = 10,
                 verbose: bool = True) -> dict:
        
        if verbose:
            print(f"\n开始REMD模拟:")
            print(f"  副本数: {self.num_replicas}")
            print(f"  温度范围: {self.T_min:.2f} - {self.T_max:.2f}")
            print(f"  循环数: {n_cycles}")
            print(f"  每循环步数: {n_steps_per_cycle}")
            print(f"  温度调度: {self.temperatures}")
        
        for cycle in tqdm(range(n_cycles), disable=not verbose, desc="REMD Cycles"):
            for rep in self.replicas:
                rep.run_md(n_steps_per_cycle)
            
            if cycle % exchange_interval == 0:
                self.exchange_attempts += 1
                for i in range(0, self.num_replicas - 1, 2):
                    self.attempt_exchange(i, i + 1)
                for i in range(1, self.num_replicas - 1, 2):
                    self.attempt_exchange(i, i + 1)
            
            if cycle % record_interval == 0:
                for i, rep in enumerate(self.replicas):
                    self.trajectories[i].append(rep.sim.positions.copy())
                    self.Q_trajectories[i].append(rep.Q_history[-1])
                    self.energy_trajectories[i].append(rep.energy_history[-1])
        
        return self.analyze_results()
    
    def analyze_results(self) -> dict:
        results = {
            'temperatures': self.temperatures,
            'exchange_rates': self.exchange_accepts / self.exchange_attempts if self.exchange_attempts > 0 else 0,
            'Q_trajectories': self.Q_trajectories,
            'energy_trajectories': self.energy_trajectories,
            'final_Q': [Q[-1] if Q else 0.0 for Q in self.Q_trajectories],
            'final_energy': [E[-1] if E else 0.0 for E in self.energy_trajectories]
        }
        
        avg_Q = []
        std_Q = []
        for Q_traj in self.Q_trajectories:
            if len(Q_traj) > 0:
                avg_Q.append(np.mean(Q_traj[-len(Q_traj)//2:]))
                std_Q.append(np.std(Q_traj[-len(Q_traj)//2:]))
            else:
                avg_Q.append(0.0)
                std_Q.append(0.0)
        
        results['avg_Q'] = avg_Q
        results['std_Q'] = std_Q
        
        return results
    
    def plot_results(self, save_path: str = 'remd_results.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for i, Q_traj in enumerate(self.Q_trajectories):
            if len(Q_traj) > 0:
                axes[0, 0].plot(Q_traj, label=f'T={self.temperatures[i]:.2f}', alpha=0.7)
        axes[0, 0].set_xlabel('Cycle')
        axes[0, 0].set_ylabel('Q')
        axes[0, 0].set_title('Q vs Cycle')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.temperatures, 
                       [np.mean(Q) if len(Q) > 0 else 0 for Q in self.Q_trajectories], 
                       'o-', linewidth=2)
        axes[0, 1].fill_between(self.temperatures,
                               [np.mean(Q) - np.std(Q) if len(Q) > 0 else 0 for Q in self.Q_trajectories],
                               [np.mean(Q) + np.std(Q) if len(Q) > 0 else 0 for Q in self.Q_trajectories],
                               alpha=0.3)
        axes[0, 1].set_xlabel('Temperature')
        axes[0, 1].set_ylabel('Average Q')
        axes[0, 1].set_title('Melting Curve')
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, E_traj in enumerate(self.energy_trajectories):
            if len(E_traj) > 0:
                axes[0, 2].plot(E_traj, label=f'T={self.temperatures[i]:.2f}', alpha=0.7)
        axes[0, 2].set_xlabel('Cycle')
        axes[0, 2].set_ylabel('Energy')
        axes[0, 2].set_title('Energy vs Cycle')
        axes[0, 2].grid(True, alpha=0.3)
        
        exchange_rates = self.exchange_accepts.diagonal(offset=1) / self.exchange_attempts if self.exchange_attempts > 0 else np.zeros(self.num_replicas-1)
        axes[1, 0].bar(range(self.num_replicas - 1), exchange_rates)
        axes[1, 0].set_xlabel('Replica Pair')
        axes[1, 0].set_ylabel('Exchange Rate')
        axes[1, 0].set_title('Exchange Acceptance Rates')
        axes[1, 0].axhline(y=0.2, color='r', linestyle='--', label='Target (20%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        for i in range(min(3, self.num_replicas)):
            if len(self.Q_trajectories[i]) > 0:
                axes[1, 1].hist(self.Q_trajectories[i], bins=30, alpha=0.5, 
                               label=f'T={self.temperatures[i]:.2f}', density=True)
        axes[1, 1].set_xlabel('Q')
        axes[1, 1].set_ylabel('Probability Density')
        axes[1, 1].set_title('Q Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        T_mid = self.temperatures[len(self.temperatures)//2]
        mid_idx = len(self.temperatures) // 2
        if len(self.Q_trajectories[mid_idx]) > 0:
            axes[1, 2].scatter(self.energy_trajectories[mid_idx], 
                              self.Q_trajectories[mid_idx], 
                              alpha=0.5, s=10)
        axes[1, 2].set_xlabel('Energy')
        axes[1, 2].set_ylabel('Q')
        axes[1, 2].set_title(f'Energy vs Q (T={T_mid:.2f})')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class FoldingPathAnalyzer:
    def __init__(self, trajectory: np.ndarray, native_positions: np.ndarray):
        self.trajectory = trajectory
        self.native_positions = native_positions
        self.n_frames = len(trajectory)
        self.n_beads = len(native_positions)
        
        self.Q_values = None
        self.rmsd_values = None
        self.contact_formation_order = None
        
    def compute_order_parameters(self):
        self.Q_values = []
        self.rmsd_values = []
        
        for frame in self.trajectory:
            Q = self._compute_Q(frame)
            rmsd = self._compute_rmsd(frame)
            self.Q_values.append(Q)
            self.rmsd_values.append(rmsd)
        
        self.Q_values = np.array(self.Q_values)
        self.rmsd_values = np.array(self.rmsd_values)
        
        return {
            'Q': self.Q_values,
            'RMSD': self.rmsd_values
        }
    
    def _compute_Q(self, positions: np.ndarray) -> float:
        distances = squareform(pdist(positions))
        native_distances = squareform(pdist(self.native_positions))
        
        contacts = np.where((native_distances < 8.0) & 
                           (np.abs(np.arange(self.n_beads)[:, None] - 
                                   np.arange(self.n_beads)[None, :]) >= 3))
        
        formed = np.sum(distances[contacts] < 1.2 * native_distances[contacts])
        total = len(contacts[0])
        
        return formed / total if total > 0 else 0.0
    
    def _compute_rmsd(self, positions: np.ndarray) -> float:
        centered_current = positions - np.mean(positions, axis=0)
        centered_native = self.native_positions - np.mean(self.native_positions, axis=0)
        
        H = centered_current.T @ centered_native
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        aligned = centered_current @ R
        diff = aligned - centered_native
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    def find_folding_transitions(self, Q_threshold: float = 0.8) -> list:
        transitions = []
        in_folded = False
        
        for i, Q in enumerate(self.Q_values):
            if Q >= Q_threshold and not in_folded:
                transitions.append(('fold', i))
                in_folded = True
            elif Q < Q_threshold * 0.5 and in_folded:
                transitions.append(('unfold', i))
                in_folded = False
        
        return transitions
    
    def compute_contact_order(self, native_contacts: np.ndarray) -> dict:
        contact_times = {}
        
        for idx, (i, j) in enumerate(native_contacts):
            r0 = np.linalg.norm(self.native_positions[i] - self.native_positions[j])
            first_formed = None
            
            for t, frame in enumerate(self.trajectory):
                dist = np.linalg.norm(frame[j] - frame[i])
                if dist < 1.2 * r0:
                    first_formed = t
                    break
            
            contact_times[(i, j)] = first_formed
        
        sorted_contacts = sorted(contact_times.items(), key=lambda x: (x[1] is None, x[1]))
        
        return {
            'contact_times': contact_times,
            'sorted_contacts': sorted_contacts,
            'early_contacts': [c[0] for c in sorted_contacts[:len(sorted_contacts)//3] if c[1] is not None],
            'late_contacts': [c[0] for c in sorted_contacts[-len(sorted_contacts)//3:] if c[1] is not None]
        }
    
    def cluster_states(self, n_clusters: int = 5) -> dict:
        flattened = self.trajectory.reshape(self.n_frames, -1)
        Z = linkage(flattened, method='ward')
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        
        centers = []
        for c in range(1, n_clusters + 1):
            mask = labels == c
            if np.any(mask):
                center_idx = np.argmin(np.sum((flattened[mask] - np.mean(flattened[mask], axis=0))**2, axis=1))
                centers.append(self.trajectory[mask][center_idx])
        
        return {
            'labels': labels,
            'centers': centers,
            'populations': np.bincount(labels)[1:] / len(labels)
        }
    
    def plot_folding_path(self, save_path: str = 'folding_path.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(self.Q_values, self.rmsd_values, alpha=0.5, linewidth=1)
        sc = axes[0, 0].scatter(self.Q_values, self.rmsd_values, 
                               c=np.arange(self.n_frames), 
                               cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(sc, ax=axes[0, 0], label='Time (frame)')
        axes[0, 0].set_xlabel('Q')
        axes[0, 0].set_ylabel('RMSD')
        axes[0, 0].set_title('Folding Free Energy Surface')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.Q_values, linewidth=2)
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Q')
        axes[0, 1].set_title('Folding Kinetics')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist2d(self.Q_values, self.rmsd_values, bins=30, cmap='Blues')
        axes[1, 0].set_xlabel('Q')
        axes[1, 0].set_ylabel('RMSD')
        axes[1, 0].set_title('Population Distribution')
        
        axes[1, 1].plot(self.rmsd_values, linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('RMSD')
        axes[1, 1].set_title('RMSD vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def compare_sampling_efficiency(normal_traj_Q: list, remd_traj_Q: list) -> dict:
    normal_Q = np.array(normal_traj_Q)
    remd_Q = np.array(remd_traj_Q)
    
    normal_var = np.var(normal_Q)
    remd_var = np.var(remd_Q)
    
    normal_range = np.max(normal_Q) - np.min(normal_Q)
    remd_range = np.max(remd_Q) - np.min(remd_Q)
    
    normal_folded = np.sum(normal_Q > 0.8) / len(normal_Q)
    remd_folded = np.sum(remd_Q > 0.8) / len(remd_Q) if len(remd_Q) > 0 else 0
    
    return {
        'normal_variance': normal_var,
        'remd_variance': remd_var,
        'variance_ratio': remd_var / normal_var if normal_var > 0 else float('inf'),
        'normal_Q_range': normal_range,
        'remd_Q_range': remd_range,
        'normal_folded_fraction': normal_folded,
        'remd_folded_fraction': remd_folded,
        'efficiency_gain': remd_folded / normal_folded if normal_folded > 0 else float('inf')
    }
