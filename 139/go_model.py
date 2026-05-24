import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from units import UnitConverter, get_optimal_gamma, folding_rate_from_mfpt


class GoModel:
    def __init__(self, 
                 num_beads: int, 
                 native_contacts: np.ndarray, 
                 native_distances: np.ndarray,
                 temperature: float = 1.0,
                 epsilon: float = 1.0,
                 sigma: float = 1.0,
                 gamma: float = None,
                 dt: float = 0.005,
                 box_size: float = 100.0,
                 auto_gamma: bool = True,
                 gamma_model: str = 'stokes_einstein'):
        
        self.N = num_beads
        self.native_contacts = native_contacts
        self.native_distances = native_distances
        self.T = temperature
        self.epsilon = epsilon
        self.sigma = sigma
        self.dt = dt
        self.box_size = box_size
        
        self.unit_converter = UnitConverter()
        
        if gamma is None and auto_gamma:
            temp_K = temperature * 300.0
            self.gamma = get_optimal_gamma(num_beads, temp_K, gamma_model)
        else:
            self.gamma = gamma if gamma is not None else 0.5
        
        self.k_bond = 100.0
        self.r0_bond = 1.0
        
        self.positions = None
        self.velocities = None
        
        self.initialize_positions()
        
    def initialize_positions(self):
        self.positions = np.zeros((self.N, 3))
        for i in range(1, self.N):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            self.positions[i] = self.positions[i-1] + np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ]) * self.r0_bond
        
        self.velocities = np.random.randn(self.N, 3) * np.sqrt(self.T)
        
    def compute_bond_forces(self):
        forces = np.zeros_like(self.positions)
        for i in range(self.N - 1):
            vec = self.positions[i+1] - self.positions[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                continue
            force_mag = -self.k_bond * (dist - self.r0_bond)
            force = force_mag * vec / dist
            forces[i] -= force
            forces[i+1] += force
        return forces
    
    def compute_native_forces(self):
        forces = np.zeros_like(self.positions)
        
        for idx, (i, j) in enumerate(self.native_contacts):
            vec = self.positions[j] - self.positions[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                continue
            
            r0 = self.native_distances[idx]
            sig = r0 / (2 ** (1/6))
            
            r6 = (sig / dist) ** 6
            r12 = r6 * r6
            
            force_mag = 12 * self.epsilon * (r12 - r6) / dist
            force = force_mag * vec / dist
            
            forces[i] -= force
            forces[j] += force
        
        return forces
    
    def compute_non_native_repulsion(self):
        forces = np.zeros_like(self.positions)
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if j == i + 1:
                    continue
                
                is_native = False
                for (ni, nj) in self.native_contacts:
                    if (ni == i and nj == j) or (ni == j and nj == i):
                        is_native = True
                        break
                
                if not is_native:
                    vec = self.positions[j] - self.positions[i]
                    dist = np.linalg.norm(vec)
                    if dist < 1e-8:
                        continue
                    
                    sig = 0.8
                    if dist < sig * (2 ** (1/6)):
                        r6 = (sig / dist) ** 6
                        r12 = r6 * r6
                        force_mag = 12 * self.epsilon * r12 / dist
                        force = force_mag * vec / dist
                        forces[i] -= force
                        forces[j] += force
        
        return forces
    
    def compute_forces(self):
        bond_forces = self.compute_bond_forces()
        native_forces = self.compute_native_forces()
        repulsion_forces = self.compute_non_native_repulsion()
        return bond_forces + native_forces + repulsion_forces
    
    def langevin_step(self):
        forces = self.compute_forces()
        
        noise = np.random.randn(self.N, 3) * np.sqrt(2 * self.gamma * self.T / self.dt)
        
        self.velocities += (forces - self.gamma * self.velocities + noise) * self.dt
        self.positions += self.velocities * self.dt
        
        self.positions = np.clip(self.positions, -self.box_size/2, self.box_size/2)
    
    def compute_native_contact_fraction(self) -> float:
        if len(self.native_contacts) == 0:
            return 0.0
        
        formed = 0
        for idx, (i, j) in enumerate(self.native_contacts):
            vec = self.positions[j] - self.positions[i]
            dist = np.linalg.norm(vec)
            r0 = self.native_distances[idx]
            if dist < 1.2 * r0:
                formed += 1
        
        return formed / len(self.native_contacts)
    
    def compute_rmsd(self, native_positions: np.ndarray) -> float:
        centered_current = self.positions - np.mean(self.positions, axis=0)
        centered_native = native_positions - np.mean(native_positions, axis=0)
        
        H = centered_current.T @ centered_native
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        aligned = centered_current @ R
        diff = aligned - centered_native
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    def simulate(self, n_steps: int, record_interval: int = 100):
        trajectory = []
        Q_values = []
        
        for step in tqdm(range(n_steps), desc="Simulating"):
            self.langevin_step()
            
            if step % record_interval == 0:
                trajectory.append(self.positions.copy())
                Q_values.append(self.compute_native_contact_fraction())
        
        return np.array(trajectory), np.array(Q_values)


def generate_native_structure(num_beads: int, structure_type: str = 'helix') -> np.ndarray:
    positions = np.zeros((num_beads, 3))
    
    if structure_type == 'helix':
        rise = 0.5
        radius = 0.8
        theta_increment = 2 * np.pi / 3.6
        
        for i in range(num_beads):
            theta = i * theta_increment
            positions[i] = [
                radius * np.cos(theta),
                radius * np.sin(theta),
                i * rise
            ]
    elif structure_type == 'hairpin':
        for i in range(num_beads // 2):
            positions[i] = [0, 0, i * 1.0]
        for i in range(num_beads // 2, num_beads):
            positions[i] = [1.5, 0, (num_beads - i - 1) * 1.0]
    else:
        for i in range(num_beads):
            positions[i] = [0, 0, i * 1.0]
    
    return positions


def generate_contact_map(native_positions: np.ndarray, 
                         cutoff: float = 8.0, 
                         min_sequence_separation: int = 3) -> tuple:
    num_beads = len(native_positions)
    distances = squareform(pdist(native_positions))
    
    contacts = []
    contact_distances = []
    
    for i in range(num_beads):
        for j in range(i + min_sequence_separation, num_beads):
            if distances[i, j] < cutoff:
                contacts.append((i, j))
                contact_distances.append(distances[i, j])
    
    return np.array(contacts), np.array(contact_distances)


def calculate_mfpt(num_trajectories: int, 
                   simulation: GoModel, 
                   native_positions: np.ndarray,
                   Q_threshold: float = 0.8,
                   max_steps: int = 500000) -> tuple:
    folding_times = []
    Q_trajectories = []
    
    for traj_idx in tqdm(range(num_trajectories), desc="MFPT Calculation"):
        simulation.initialize_positions()
        
        Q = 0.0
        folded_step = None
        Q_history = []
        
        for step in range(max_steps):
            simulation.langevin_step()
            
            if step % 10 == 0:
                Q = simulation.compute_native_contact_fraction()
                Q_history.append(Q)
                
                if Q >= Q_threshold and folded_step is None:
                    folded_step = step
                    break
        
        folding_times.append(folded_step if folded_step is not None else max_steps)
        Q_trajectories.append(Q_history)
    
    mfpt = np.mean(folding_times)
    mfpt_error = np.std(folding_times) / np.sqrt(num_trajectories)
    
    return mfpt, mfpt_error, folding_times, Q_trajectories


def plot_results(Q_values: np.ndarray, trajectory: np.ndarray, 
                 contacts: np.ndarray = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(Q_values)
    axes[0].set_xlabel('Time (steps x record_interval)')
    axes[0].set_ylabel('Native Contact Fraction Q')
    axes[0].set_title('Folding Kinetics')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(Q_values, bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Q')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Q Distribution')
    axes[1].grid(True, alpha=0.3)
    
    if len(trajectory) > 0:
        final_pos = trajectory[-1]
        axes[2].plot(final_pos[:, 0], final_pos[:, 1], 'o-', linewidth=2, markersize=6)
        
        if contacts is not None:
            for (i, j) in contacts[:20]:
                axes[2].plot([final_pos[i, 0], final_pos[j, 0]],
                           [final_pos[i, 1], final_pos[j, 1]],
                           'r--', alpha=0.3)
        
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        axes[2].set_title('Final Structure')
        axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('go_model_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_mfpt_results(folding_times: list, Q_trajectories: list, mfpt: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(folding_times, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(mfpt, color='red', linestyle='--', label=f'MFPT = {mfpt:.0f} steps')
    axes[0].set_xlabel('Folding Time (steps)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Folding Time Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for i, Q in enumerate(Q_trajectories[:10]):
        axes[1].plot(np.arange(len(Q)) * 10, Q, alpha=0.6, linewidth=1)
    axes[1].set_xlabel('Time (steps)')
    axes[1].set_ylabel('Q')
    axes[1].set_title('Folding Trajectories')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mfpt_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def calibrate_friction_coefficient(num_beads: int,
                                    contacts: np.ndarray,
                                    contact_distances: np.ndarray,
                                    target_kf: float = None,
                                    num_calibration_trajs: int = 10,
                                    max_calibration_steps: int = 100000,
                                    temperature: float = 1.0) -> dict:
    from units import FoldingRateCalibrator
    
    print("\n" + "="*60)
    print("开始摩擦系数校准")
    print("="*60)
    
    calibrator = FoldingRateCalibrator(num_beads)
    
    if target_kf is None:
        target_kf = calibrator.estimate_kf_from_size()
        print(f"\n根据蛋白质大小估计目标折叠速率: kf = {target_kf:.2e} s^-1")
    else:
        print(f"\n使用用户指定的目标折叠速率: kf = {target_kf:.2e} s^-1")
    
    print(f"\n初步模拟 (γ = 自动选择)...")
    sim_prelim = GoModel(
        num_beads=num_beads,
        native_contacts=contacts,
        native_distances=contact_distances,
        temperature=temperature,
        auto_gamma=True,
        gamma_model='stokes_einstein'
    )
    
    mfpt_prelim, _, _, _ = calculate_mfpt(
        num_trajectories=num_calibration_trajs,
        simulation=sim_prelim,
        native_positions=None,
        Q_threshold=0.75,
        max_steps=max_calibration_steps
    )
    
    print(f"\n初步模拟MFPT: {mfpt_prelim:.0f} 步")
    
    results = calibrator.get_calibrated_gamma(
        simulated_mfpt=mfpt_prelim,
        dt=sim_prelim.dt,
        target_kf=target_kf
    )
    
    calibrator.print_calibration_report(results)
    
    return results


def create_calibrated_simulation(num_beads: int,
                                  contacts: np.ndarray,
                                  contact_distances: np.ndarray,
                                  temperature: float = 1.0,
                                  target_kf: float = None) -> tuple:
    
    calibration = calibrate_friction_coefficient(
        num_beads=num_beads,
        contacts=contacts,
        contact_distances=contact_distances,
        target_kf=target_kf,
        temperature=temperature
    )
    
    sim = GoModel(
        num_beads=num_beads,
        native_contacts=contacts,
        native_distances=contact_distances,
        temperature=temperature,
        gamma=calibration['corrected_gamma'],
        auto_gamma=False
    )
    
    return sim, calibration


def compare_gamma_models(num_beads: int, temperature: float = 0.9):
    from units import UnitConverter, get_optimal_gamma
    
    print("\n" + "="*60)
    print("不同摩擦系数模型对比")
    print("="*60)
    
    models = ['stokes_einstein', 'empirical', 'literature']
    temp_K = temperature * 300.0
    
    print(f"\n蛋白质大小: {num_beads} 个残基")
    print(f"温度: {temp_K:.0f} K")
    
    converter = UnitConverter()
    print(f"时间尺度: τ = {converter.time_ps_per_tau:.3f} ps")
    print(f"估计半径: {converter.estimate_protein_radius(num_beads):.2f} Å")
    
    print(f"\n{'模型':<20} {'γ':<10} {'相对扩散':<15}")
    print("-" * 45)
    
    for model in models:
        gamma = get_optimal_gamma(num_beads, temp_K, model)
        relative_diffusion = 1.0 / gamma
        print(f"{model:<20} {gamma:<10.4f} {relative_diffusion:<15.2f}")
    
    print("="*60 + "\n")

