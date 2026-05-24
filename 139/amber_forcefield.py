import numpy as np
from scipy.spatial.distance import pdist, squareform


class AmberForceField:
    def __init__(self, use_ca_model: bool = True):
        self.use_ca_model = use_ca_model
        
        self.k_bond = 450.0
        self.r0_bond = 1.526
        
        self.k_angle = 50.0
        self.theta0 = np.deg2rad(116.0)
        
        self.k_dihedral = 2.5
        self.n_dihedral = 2
        self.delta_dihedral = np.pi
        
        self.epsilon = 0.2
        self.sigma = 4.0
        
        self.elec_epsilon = 1.0
        self.epsilon_r = 80.0
        self.k_elec = 332.0
        
        self.native_contacts = None
        self.native_distances = None
        self.native_bonds = None
        self.native_angles = None
        
    def set_native_structure(self, positions: np.ndarray, contacts: np.ndarray = None):
        self.native_positions = positions.copy()
        
        if contacts is not None:
            self.native_contacts = contacts
            self.native_distances = np.array([
                np.linalg.norm(positions[i] - positions[j]) for i, j in contacts
            ])
        
        n = len(positions)
        self.native_bonds = [(i, i+1) for i in range(n-1)]
        
        if n >= 3:
            self.native_angles = [(i, i+1, i+2) for i in range(n-2)]
        
        if n >= 4:
            self.native_dihedrals = [(i, i+1, i+2, i+3) for i in range(n-4)]
    
    def compute_bond_energy(self, positions: np.ndarray) -> tuple:
        energy = 0.0
        forces = np.zeros_like(positions)
        
        for i, j in self.native_bonds:
            vec = positions[j] - positions[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                continue
            
            dr = dist - self.r0_bond
            energy += 0.5 * self.k_bond * dr * dr
            
            force_mag = -self.k_bond * dr
            force = force_mag * vec / dist
            forces[i] -= force
            forces[j] += force
        
        return energy, forces
    
    def compute_angle_energy(self, positions: np.ndarray) -> tuple:
        energy = 0.0
        forces = np.zeros_like(positions)
        
        if len(self.native_angles) == 0:
            return energy, forces
        
        for i, j, k in self.native_angles:
            vec1 = positions[i] - positions[j]
            vec2 = positions[k] - positions[j]
            
            r1 = np.linalg.norm(vec1)
            r2 = np.linalg.norm(vec2)
            
            if r1 < 1e-8 or r2 < 1e-8:
                continue
            
            cos_theta = np.dot(vec1, vec2) / (r1 * r2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            dtheta = theta - self.theta0
            energy += 0.5 * self.k_angle * dtheta * dtheta
            
            force_mag = -self.k_angle * dtheta
            sin_theta = np.sqrt(1 - cos_theta * cos_theta)
            if sin_theta < 1e-8:
                sin_theta = 1e-8
            
            d_cos_d_theta = -sin_theta
            factor = force_mag / d_cos_d_theta
            
            forces[i] += factor * (vec2 / (r1 * r2) - cos_theta * vec1 / (r1 * r1))
            forces[k] += factor * (vec1 / (r1 * r2) - cos_theta * vec2 / (r2 * r2))
            forces[j] -= forces[i] + forces[k]
        
        return energy, forces
    
    def compute_dihedral_energy(self, positions: np.ndarray) -> tuple:
        energy = 0.0
        forces = np.zeros_like(positions)
        
        if not hasattr(self, 'native_dihedrals') or len(self.native_dihedrals) == 0:
            return energy, forces
        
        for i, j, k, l in self.native_dihedrals:
            r1 = positions[j] - positions[i]
            r2 = positions[k] - positions[j]
            r3 = positions[l] - positions[k]
            
            n1 = np.cross(r1, r2)
            n2 = np.cross(r2, r3)
            
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            
            if n1_norm < 1e-8 or n2_norm < 1e-8:
                continue
            
            n1_normalized = n1 / n1_norm
            n2_normalized = n2 / n2_norm
            
            cos_phi = np.dot(n1_normalized, n2_normalized)
            cos_phi = np.clip(cos_phi, -1.0, 1.0)
            phi = np.arccos(cos_phi)
            
            if np.dot(np.cross(n1, n2), r2) < 0:
                phi = -phi
            
            energy += self.k_dihedral * (1 + np.cos(self.n_dihedral * phi - self.delta_dihedral))
            
            force_mag = -self.k_dihedral * self.n_dihedral * np.sin(self.n_dihedral * phi - self.delta_dihedral)
            
            r2_norm = np.linalg.norm(r2)
            f1 = force_mag * r2_norm / (n1_norm * n1_norm) * np.cross(r2, n1)
            f4 = force_mag * r2_norm / (n2_norm * n2_norm) * np.cross(n2, r2)
            
            A = np.dot(r1, r2) / (r2_norm * r2_norm)
            B = np.dot(r3, r2) / (r2_norm * r2_norm)
            
            f2 = A * f1 - f4 - force_mag / r2_norm * np.cross(r3, r1)
            f3 = B * f4 - f1 + force_mag / r2_norm * np.cross(r3, r1)
            
            forces[i] += f1
            forces[j] += f2
            forces[k] += f3
            forces[l] += f4
        
        return energy, forces
    
    def compute_lj_energy(self, positions: np.ndarray) -> tuple:
        energy = 0.0
        forces = np.zeros_like(positions)
        n = len(positions)
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(i - j) <= 3:
                    continue
                
                vec = positions[j] - positions[i]
                dist = np.linalg.norm(vec)
                if dist < 1e-8:
                    continue
                
                is_native = False
                if self.native_contacts is not None:
                    for (ni, nj) in self.native_contacts:
                        if (ni == i and nj == j) or (ni == j and nj == i):
                            is_native = True
                            break
                
                if is_native:
                    sig = self.sigma
                    r6 = (sig / dist) ** 6
                    r12 = r6 * r6
                    energy += 4 * self.epsilon * (r12 - r6)
                    
                    force_mag = 24 * self.epsilon * (2 * r12 - r6) / dist
                    force = force_mag * vec / dist
                    forces[i] -= force
                    forces[j] += force
                else:
                    sig = 3.5
                    if dist < sig * (2 ** (1/6)):
                        r6 = (sig / dist) ** 6
                        r12 = r6 * r6
                        energy += 4 * self.epsilon * r12
                        
                        force_mag = 24 * self.epsilon * 2 * r12 / dist
                        force = force_mag * vec / dist
                        forces[i] -= force
                        forces[j] += force
        
        return energy, forces
    
    def compute_total_energy(self, positions: np.ndarray) -> tuple:
        e_bond, f_bond = self.compute_bond_energy(positions)
        e_angle, f_angle = self.compute_angle_energy(positions)
        e_dihedral, f_dihedral = self.compute_dihedral_energy(positions)
        e_lj, f_lj = self.compute_lj_energy(positions)
        
        total_energy = e_bond + e_angle + e_dihedral + e_lj
        total_forces = f_bond + f_angle + f_dihedral + f_lj
        
        energy_components = {
            'bond': e_bond,
            'angle': e_angle,
            'dihedral': e_dihedral,
            'lj': e_lj,
            'total': total_energy
        }
        
        return total_energy, total_forces, energy_components


class GoAmberHybrid:
    def __init__(self, 
                 num_beads: int,
                 native_contacts: np.ndarray,
                 native_distances: np.ndarray,
                 native_positions: np.ndarray,
                 temperature: float = 1.0,
                 epsilon_contact: float = 1.0,
                 weight_amber: float = 0.3,
                 weight_go: float = 0.7):
        
        self.N = num_beads
        self.native_contacts = native_contacts
        self.native_distances = native_distances
        self.T = temperature
        self.epsilon_contact = epsilon_contact
        self.weight_amber = weight_amber
        self.weight_go = weight_go
        
        self.amber_ff = AmberForceField(use_ca_model=True)
        self.amber_ff.set_native_structure(native_positions, native_contacts)
        
        self.positions = None
        self.velocities = None
        self.gamma = 0.15
        self.dt = 0.005
        
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
            ]) * 1.0
        
        self.velocities = np.random.randn(self.N, 3) * np.sqrt(self.T)
    
    def compute_go_forces(self):
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
            
            force_mag = 12 * self.epsilon_contact * (r12 - r6) / dist
            force = force_mag * vec / dist
            
            forces[i] -= force
            forces[j] += force
        
        return forces
    
    def compute_hybrid_forces(self):
        _, amber_forces, _ = self.amber_ff.compute_total_energy(self.positions)
        go_forces = self.compute_go_forces()
        
        bond_forces = np.zeros_like(self.positions)
        for i in range(self.N - 1):
            vec = self.positions[i+1] - self.positions[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                continue
            force_mag = -100.0 * (dist - 1.0)
            force = force_mag * vec / dist
            bond_forces[i] -= force
            bond_forces[i+1] += force
        
        total_forces = (self.weight_amber * amber_forces + 
                       self.weight_go * go_forces + 
                       bond_forces)
        
        return total_forces
    
    def compute_energy(self):
        amber_energy, _, _ = self.amber_ff.compute_total_energy(self.positions)
        
        go_energy = 0.0
        for idx, (i, j) in enumerate(self.native_contacts):
            vec = self.positions[j] - self.positions[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                continue
            r0 = self.native_distances[idx]
            sig = r0 / (2 ** (1/6))
            r6 = (sig / dist) ** 6
            r12 = r6 * r6
            go_energy += 4 * self.epsilon_contact * (r12 - r6)
        
        return self.weight_amber * amber_energy + self.weight_go * go_energy
    
    def compute_native_contact_fraction(self):
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
    
    def langevin_step(self):
        forces = self.compute_hybrid_forces()
        
        noise = np.random.randn(self.N, 3) * np.sqrt(2 * self.gamma * self.T / self.dt)
        
        self.velocities += (forces - self.gamma * self.velocities + noise) * self.dt
        self.positions += self.velocities * self.dt
    
    def simulate(self, n_steps: int, record_interval: int = 100):
        trajectory = []
        energies = []
        Q_values = []
        
        for step in range(n_steps):
            self.langevin_step()
            
            if step % record_interval == 0:
                trajectory.append(self.positions.copy())
                energies.append(self.compute_energy())
                Q_values.append(self.compute_native_contact_fraction())
        
        return np.array(trajectory), np.array(energies), np.array(Q_values)
