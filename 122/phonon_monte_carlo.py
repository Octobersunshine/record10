import numpy as np
from scipy.constants import hbar, k, pi
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class PhononPacket:
    phonon_id: int
    branch: str
    omega: float
    k_magnitude: float
    position: np.ndarray
    velocity: np.ndarray
    energy: float
    active: bool = True
    time_to_scatter: float = 0.0
    scatter_count: int = 0


@dataclass
class Layer:
    thickness: float
    material: str
    interface_transmission: float = 0.8


class Superlattice:
    def __init__(self, layers: List[Layer], periodic: bool = True):
        self.layers = layers
        self.periodic = periodic
        self.total_thickness = sum(layer.thickness for layer in layers)
        
        self._build_layer_boundaries()
    
    def _build_layer_boundaries(self):
        self.boundaries = []
        z = 0.0
        for layer in self.layers:
            self.boundaries.append((z, z + layer.thickness, layer))
            z += layer.thickness
    
    def get_layer_at(self, z: float) -> Tuple[int, Layer]:
        if self.periodic:
            z = z % self.total_thickness
        
        for i, (z_start, z_end, layer) in enumerate(self.boundaries):
            if z_start <= z < z_end:
                return i, layer
        
        return len(self.layers) - 1, self.layers[-1]
    
    def check_interface_crossing(self, z_old: float, z_new: float) -> Optional[Tuple[int, float, Layer]]:
        z_min = min(z_old, z_new)
        z_max = max(z_old, z_new)
        
        for i, (z_start, z_end, layer) in enumerate(self.boundaries):
            if z_min < z_start < z_max:
                return i, z_start, layer
        
        return None


class PhononMonteCarlo:
    def __init__(self, material='Si', structure='bulk', **kwargs):
        self.material = material
        self.structure = structure
        self.L = kwargs.get('L', None)
        self.T = kwargs.get('T', 300)
        self.dT = kwargs.get('dT', 10)
        self._setup_material_params()
        
        if structure == 'superlattice':
            self.superlattice = kwargs.get('superlattice', None)
            if self.superlattice is None:
                self._create_default_superlattice()
        else:
            self.superlattice = None
        
        self.rng = np.random.RandomState(kwargs.get('seed', 42))
        self.phonon_packets: List[PhononPacket] = []
        self.heat_flux_history = []
    
    def _setup_material_params(self):
        if self.material == 'Si':
            self.rho = 2330.0
            self.n_atoms = 5e28
            self.theta_D = 640.0
            self.v_long = 8433.0
            self.v_trans = 5845.0
            self.B_U_LA = 1.4e-19
            self.B_U_TA = 3.2e-18
            self.B_N_LA = 2.0e-20
            self.B_N_TA = 5.0e-19
            self.A_iso = 1.08e-43
            self.omega_max_LA = 9.6e13
            self.omega_max_TA = 7.6e13
        elif self.material == 'Ge':
            self.rho = 5323.0
            self.n_atoms = 4.42e28
            self.theta_D = 374.0
            self.v_long = 5410.0
            self.v_trans = 3350.0
            self.B_U_LA = 2.0e-19
            self.B_U_TA = 5.0e-18
            self.B_N_LA = 3.0e-20
            self.B_N_TA = 8.0e-19
            self.A_iso = 8.0e-43
            self.omega_max_LA = 6.0e13
            self.omega_max_TA = 4.5e13
        elif self.material == 'Si/Ge':
            self.rho = 3826.5
            self.n_atoms = 4.71e28
            self.theta_D = 507.0
            self.v_long = 6921.5
            self.v_trans = 4597.5
            self.B_U_LA = 1.7e-19
            self.B_U_TA = 4.1e-18
            self.B_N_LA = 2.5e-20
            self.B_N_TA = 6.5e-19
            self.A_iso = 5.0e-42
            self.omega_max_LA = 7.8e13
            self.omega_max_TA = 6.0e13
        else:
            self.rho = 2330.0
            self.n_atoms = 5e28
            self.theta_D = 500.0
            self.v_long = 6000.0
            self.v_trans = 4000.0
            self.B_U_LA = 2.0e-19
            self.B_U_TA = 4.0e-18
            self.B_N_LA = 3.0e-20
            self.B_N_TA = 6.0e-19
            self.A_iso = 1.0e-42
            self.omega_max_LA = 7.0e13
            self.omega_max_TA = 5.5e13
        
        self.k_D = (6 * pi**2 * self.n_atoms)**(1/3)
    
    def _create_default_superlattice(self):
        layer_thickness = 5e-9
        layers = [
            Layer(layer_thickness, 'Si', 0.75),
            Layer(layer_thickness, 'Ge', 0.75),
        ]
        self.superlattice = Superlattice(layers, periodic=True)
    
    def debye_dos(self, omega, branch='TA'):
        if branch == 'LA':
            v = self.v_long
            omega_max = self.omega_max_LA
        else:
            v = self.v_trans
            omega_max = self.omega_max_TA
        
        if omega > omega_max:
            return 0.0
        
        return omega**2 / (2 * pi**2 * v**3)
    
    def sample_phonon_frequency(self, branch='TA'):
        f_max = self.theta_D * k / hbar
        while True:
            omega = self.rng.uniform(0, f_max)
            p_accept = (omega / f_max)**2
            if self.rng.random() < p_accept:
                return omega
    
    def sample_branch(self):
        branches = ['LA', 'TA1', 'TA2']
        weights = [1.0, 1.0, 1.0]
        total = sum(weights)
        r = self.rng.random() * total
        cumulative = 0
        for branch, w in zip(branches, weights):
            cumulative += w
            if r <= cumulative:
                return branch
    
    def bose_einstein(self, omega, T):
        if T == 0 or omega == 0:
            return 0.0
        x = hbar * omega / (k * T)
        if x > 500:
            return 0.0
        return 1.0 / (np.exp(x) - 1.0)
    
    def bose_einstein_derivative(self, omega, T):
        if T == 0 or omega == 0:
            return 0.0
        x = hbar * omega / (k * T)
        if x > 500:
            return 0.0
        exp_x = np.exp(x)
        return -exp_x * x / (k * T * (exp_x - 1)**2)
    
    def relaxation_time(self, omega, T, branch='LA'):
        if branch == 'LA':
            B_U = self.B_U_LA
            B_N = self.B_N_LA
        else:
            B_U = self.B_U_TA
            B_N = self.B_N_TA
        
        if T < self.theta_D / 4:
            tau_u = 1.0 / (B_U * omega**2 * T * np.exp(-self.theta_D / (3 * T)))
        else:
            tau_u = 1.0 / (B_U * omega**2 * T)
        
        tau_n = 1.0 / (B_N * omega**2 * T**3) if omega > 0 else np.inf
        tau_iso = 1.0 / (self.A_iso * omega**4) if omega > 0 else np.inf
        
        tau_inv = 1.0/tau_u + 1.0/tau_n + 1.0/tau_iso
        
        if self.L is not None and self.structure == 'thin_film':
            if branch == 'LA':
                v = self.v_long
            else:
                v = self.v_trans
            tau_inv += v / self.L
        
        return 1.0 / tau_inv
    
    def mean_free_path(self, omega, T, branch='LA'):
        tau = self.relaxation_time(omega, T, branch)
        if branch == 'LA':
            v = self.v_long
        else:
            v = self.v_trans
        return v * tau
    
    def initialize_phonons(self, n_phonons, T_hot, T_cold):
        self.phonon_packets = []
        T_avg = (T_hot + T_cold) / 2
        
        for i in range(n_phonons):
            branch = self.sample_branch()
            omega = self.sample_phonon_frequency(branch)
            
            if branch == 'LA':
                v_mag = self.v_long
                k_mag = omega / self.v_long
            else:
                v_mag = self.v_trans
                k_mag = omega / self.v_trans
            
            theta = np.arccos(self.rng.uniform(-1, 1))
            phi = self.rng.uniform(0, 2 * pi)
            
            vx = v_mag * np.sin(theta) * np.cos(phi)
            vy = v_mag * np.sin(theta) * np.sin(phi)
            vz = v_mag * np.cos(theta)
            velocity = np.array([vx, vy, vz])
            
            if self.structure == 'bulk' or self.L is None:
                z = self.rng.uniform(-1e-6, 1e-6)
            else:
                z = self.rng.uniform(-self.L/2, self.L/2)
            
            position = np.array([0.0, 0.0, z])
            
            energy = hbar * omega * self.bose_einstein(omega, T_avg)
            
            tau = self.relaxation_time(omega, T_avg, branch)
            time_to_scatter = -np.log(self.rng.random()) * tau
            
            packet = PhononPacket(
                phonon_id=i,
                branch=branch,
                omega=omega,
                k_magnitude=k_mag,
                position=position,
                velocity=velocity,
                energy=energy,
                active=True,
                time_to_scatter=time_to_scatter,
                scatter_count=0
            )
            self.phonon_packets.append(packet)
    
    def scatter_phonon(self, packet: PhononPacket, T: float):
        branch = packet.branch
        
        tau_u = self.relaxation_time(packet.omega, T, branch)
        
        r = self.rng.random()
        
        if r < 0.7:
            if branch == 'LA':
                v_mag = self.v_long
            else:
                v_mag = self.v_trans
            
            theta = np.arccos(self.rng.uniform(-1, 1))
            phi = self.rng.uniform(0, 2 * pi)
            
            vx = v_mag * np.sin(theta) * np.cos(phi)
            vy = v_mag * np.sin(theta) * np.sin(phi)
            vz = v_mag * np.cos(theta)
            packet.velocity = np.array([vx, vy, vz])
        
        packet.scatter_count += 1
        
        tau = self.relaxation_time(packet.omega, T, branch)
        packet.time_to_scatter = -np.log(self.rng.random()) * tau
    
    def handle_boundary(self, packet: PhononPacket, T_hot: float, T_cold: float):
        z = packet.position[2]
        
        if self.structure == 'bulk' or self.L is None:
            return True
        
        L_half = self.L / 2
        
        if z >= L_half:
            packet.position[2] = 2 * L_half - z
            packet.velocity[2] *= -1
            return self.rng.random() < 0.9
            
        elif z <= -L_half:
            packet.position[2] = -2 * L_half - z
            packet.velocity[2] *= -1
            return self.rng.random() < 0.9
        
        return True
    
    def handle_superlattice_interface(self, packet: PhononPacket, z_old: float, z_new: float):
        if self.superlattice is None:
            return True
        
        crossing = self.superlattice.check_interface_crossing(z_old, z_new)
        if crossing is None:
            return True
        
        layer_idx, z_interface, layer = crossing
        
        transmission = layer.interface_transmission
        
        if self.rng.random() < transmission:
            return True
        else:
            packet.position[2] = 2 * z_interface - z_new
            packet.velocity[2] *= -1
            return False
    
    def run_simulation(self, n_phonons=10000, n_steps=1000, dt=1e-13, 
                       T_hot=None, T_cold=None):
        if T_hot is None:
            T_hot = self.T + self.dT
        if T_cold is None:
            T_cold = self.T - self.dT
        
        T_avg = (T_hot + T_cold) / 2
        
        print(f"初始化 {n_phonons} 个声子...")
        self.initialize_phonons(n_phonons, T_hot, T_cold)
        
        print(f"开始模拟，共 {n_steps} 步...")
        start_time = time.time()
        
        energy_exchange_hot = 0.0
        energy_exchange_cold = 0.0
        
        for step in range(n_steps):
            for packet in self.phonon_packets:
                if not packet.active:
                    continue
                
                z_old = packet.position[2]
                packet.position += packet.velocity * dt
                z_new = packet.position[2]
                
                if self.superlattice is not None:
                    transmitted = self.handle_superlattice_interface(packet, z_old, z_new)
                    if not transmitted:
                        continue
                
                at_boundary = not self.handle_boundary(packet, T_hot, T_cold)
                if at_boundary:
                    if packet.velocity[2] < 0:
                        energy_exchange_hot += packet.energy
                    else:
                        energy_exchange_cold += packet.energy
                
                packet.time_to_scatter -= dt
                
                if packet.time_to_scatter <= 0:
                    self.scatter_phonon(packet, T_avg)
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{n_steps}, 已用时 {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"模拟完成，总耗时: {total_time:.2f}s")
        
        simulation_time = n_steps * dt
        volume = 1e-6 * 1e-6 * self.L if self.L is not None else 1e-18
        heat_flux = (energy_exchange_hot - energy_exchange_cold) / (simulation_time * volume)
        thermal_cond = heat_flux * self.L / (2 * self.dT) if self.L else heat_flux / (2 * self.dT / 1e-6)
        
        return {
            'thermal_conductivity': thermal_cond,
            'heat_flux': heat_flux,
            'simulation_time_s': total_time,
            'n_phonons': n_phonons,
            'n_steps': n_steps
        }
    
    def thermal_conductivity_mc(self, T=300, n_phonons=5000, n_steps=500, L=None):
        if L is not None:
            self.L = L
        
        self.T = T
        self.dT = max(1, T * 0.02)
        
        result = self.run_simulation(
            n_phonons=n_phonons,
            n_steps=n_steps,
            T_hot=T + self.dT,
            T_cold=T - self.dT
        )
        
        return result['thermal_conductivity']
    
    def size_effect_analysis(self, L_array, T=300, n_phonons=3000, n_steps=400):
        kappas = []
        kappa_bulk = None
        
        print("\n" + "="*60)
        print("尺寸效应分析")
        print("="*60)
        
        for i, L in enumerate(L_array):
            print(f"\n[{i+1}/{len(L_array)}] L = {L*1e9:.1f} nm")
            
            if L > 1e-5:
                kappa = self.thermal_conductivity_analytical(T)
                if kappa_bulk is None:
                    kappa_bulk = kappa
            else:
                self.structure = 'thin_film'
                self.L = L
                kappa = self.thermal_conductivity_mc(T, n_phonons, n_steps, L)
            
            kappas.append(kappa)
            print(f"  κ = {kappa:.2f} W/mK")
        
        return np.array(kappas), kappa_bulk
    
    def superlattice_thermal_conductivity(self, period_array, T=300, 
                                           n_phonons=4000, n_steps=500):
        kappas = []
        
        print("\n" + "="*60)
        print("超晶格热导率分析")
        print("="*60)
        
        for i, period in enumerate(period_array):
            print(f"\n[{i+1}/{len(period_array)}] 周期 = {period*1e9:.1f} nm")
            
            layer_thickness = period / 2
            layers = [
                Layer(layer_thickness, 'Si', 0.8),
                Layer(layer_thickness, 'Ge', 0.8),
            ]
            self.superlattice = Superlattice(layers)
            self.structure = 'superlattice'
            self.L = period * 40
            self.material = 'Si/Ge'
            self._setup_material_params()
            
            kappa = self.thermal_conductivity_mc(T, n_phonons, n_steps, period * 40)
            kappas.append(kappa)
            print(f"  κ = {kappa:.2f} W/mK")
        
        return np.array(kappas)
    
    def thermal_conductivity_analytical(self, T):
        from phonon_bte_enhanced import EnhancedPhononBTE
        bte = EnhancedPhononBTE(material=self.material, L=None)
        return bte.thermal_conductivity(T)
    
    def get_phonon_statistics(self):
        if not self.phonon_packets:
            return None
        
        active = [p for p in self.phonon_packets if p.active]
        n_active = len(active)
        
        if n_active == 0:
            return None
        
        avg_energy = np.mean([p.energy for p in active])
        avg_scatters = np.mean([p.scatter_count for p in active])
        
        branches = {}
        for p in active:
            branches[p.branch] = branches.get(p.branch, 0) + 1
        
        return {
            'n_active': n_active,
            'avg_energy': avg_energy,
            'avg_scatters': avg_scatters,
            'branch_distribution': branches
        }
