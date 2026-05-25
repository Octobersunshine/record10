import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


MU0 = 4 * np.pi * 1e-7


@dataclass
class Layer:
    thickness: float
    resistivity: float


@dataclass
class MTData:
    frequencies: np.ndarray
    apparent_resistivity: np.ndarray
    phase: np.ndarray


@dataclass
class Substation:
    id: int
    name: str
    x: float
    y: float
    grounding_resistance: float


@dataclass
class TransmissionLine:
    id: int
    from_substation: int
    to_substation: int
    length: float
    resistance: float


class LayeredEarthModel:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.n_layers = len(layers)

    def compute_impedance(self, frequency: float) -> complex:
        omega = 2 * np.pi * frequency
        depths = np.cumsum([l.thickness for l in self.layers[:-1]])
        
        k = np.sqrt(1j * omega * MU0 / np.array([l.resistivity for l in self.layers]))
        
        Z = np.zeros(self.n_layers, dtype=complex)
        Z[-1] = omega * MU0 / k[-1]
        
        for i in range(self.n_layers - 2, -1, -1):
            h = self.layers[i].thickness
            Z[i] = (omega * MU0 / k[i]) * (Z[i + 1] + (omega * MU0 / k[i]) * np.tanh(1j * k[i] * h)) / \
                   ((omega * MU0 / k[i]) + Z[i + 1] * np.tanh(1j * k[i] * h))
        
        return Z[0]

    def compute_apparent_resistivity(self, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rho_app = np.zeros_like(frequencies)
        phase = np.zeros_like(frequencies)
        
        for i, f in enumerate(frequencies):
            Z = self.compute_impedance(f)
            rho_app[i] = (np.abs(Z) ** 2) / (MU0 * 2 * np.pi * f)
            phase[i] = np.angle(Z, deg=True)
        
        return rho_app, phase

    def compute_electric_field_depth(self, frequency: float, 
                                      B_amplitude: float,
                                      depths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        omega = 2 * np.pi * frequency
        k = np.sqrt(1j * omega * MU0 / np.array([l.resistivity for l in self.layers]))
        
        layer_depths = np.cumsum([0] + [l.thickness for l in self.layers[:-1]])
        
        A = np.zeros(self.n_layers, dtype=complex)
        B = np.zeros(self.n_layers, dtype=complex)
        
        B[-1] = 0.0
        A[-1] = B_amplitude
        
        for i in range(self.n_layers - 2, -1, -1):
            h = self.layers[i].thickness
            exp_term = np.exp(1j * k[i] * h)
            exp_term_neg = np.exp(-1j * k[i] * h)
            
            A[i] = (A[i + 1] * (1 + k[i + 1] / k[i]) * exp_term + 
                    A[i + 1] * (1 - k[i + 1] / k[i]) * exp_term_neg) / 2
            B[i] = (A[i + 1] * (1 - k[i + 1] / k[i]) * exp_term + 
                    A[i + 1] * (1 + k[i + 1] / k[i]) * exp_term_neg) / 2
        
        E = np.zeros_like(depths, dtype=complex)
        
        for d_idx, z in enumerate(depths):
            layer_idx = 0
            for i in range(len(layer_depths) - 1):
                if layer_depths[i] <= z < layer_depths[i + 1]:
                    layer_idx = i
                    break
            else:
                layer_idx = self.n_layers - 1
            
            z_in_layer = z - layer_depths[layer_idx]
            E[d_idx] = (A[layer_idx] * np.exp(1j * k[layer_idx] * z_in_layer) + 
                        B[layer_idx] * np.exp(-1j * k[layer_idx] * z_in_layer))
        
        return E, depths

    def compute_surface_impedance_tensor(self, frequencies: np.ndarray, 
                                          polarization: str = 'TE') -> np.ndarray:
        Z = np.zeros(len(frequencies), dtype=complex)
        
        for i, f in enumerate(frequencies):
            Z[i] = self.compute_impedance(f)
        
        return Z


class FEMEarthModel:
    def __init__(self, x_min: float, x_max: float, z_min: float, z_max: float,
                 nx: int, nz: int, layers: List[Layer]):
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max
        self.nx = nx
        self.nz = nz
        self.layers = layers
        
        self.x_nodes = np.linspace(x_min, x_max, nx)
        self.z_nodes = np.linspace(z_min, z_max, nz)
        
        self.n_nodes = nx * nz
        self.elements = []
        self._generate_mesh()
        
        self.sigma = np.zeros(self.n_nodes)
        self._assign_conductivity()

    def _generate_mesh(self):
        for i in range(self.nz - 1):
            for j in range(self.nx - 1):
                n1 = i * self.nx + j
                n2 = i * self.nx + j + 1
                n3 = (i + 1) * self.nx + j + 1
                n4 = (i + 1) * self.nx + j
                self.elements.append([n1, n2, n3, n4])

    def _assign_conductivity(self):
        layer_depths = np.cumsum([0] + [l.thickness for l in self.layers[:-1]])
        
        for i, z in enumerate(self.z_nodes):
            layer_idx = 0
            for li in range(len(layer_depths)):
                if li == len(layer_depths) - 1:
                    layer_idx = li
                    break
                if layer_depths[li] <= abs(z) < layer_depths[li + 1]:
                    layer_idx = li
                    break
            
            sigma_val = 1.0 / self.layers[layer_idx].resistivity
            for j in range(self.nx):
                self.sigma[i * self.nx + j] = sigma_val

    def _shape_functions(self, xi, eta):
        N = np.array([
            (1 - xi) * (1 - eta) / 4,
            (1 + xi) * (1 - eta) / 4,
            (1 + xi) * (1 + eta) / 4,
            (1 - xi) * (1 + eta) / 4
        ])
        dN_dxi = np.array([
            -(1 - eta) / 4,
            (1 - eta) / 4,
            (1 + eta) / 4,
            -(1 + eta) / 4
        ])
        dN_deta = np.array([
            -(1 - xi) / 4,
            -(1 + xi) / 4,
            (1 + xi) / 4,
            (1 - xi) / 4
        ])
        return N, dN_dxi, dN_deta

    def solve(self, omega: float, E0: float) -> np.ndarray:
        rows = []
        cols = []
        data = []
        b = np.zeros(self.n_nodes, dtype=complex)
        
        gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        gw = np.array([1, 1])
        
        for elem in self.elements:
            x_coords = np.array([self.x_nodes[n % self.nx] for n in elem])
            z_coords = np.array([self.z_nodes[n // self.nx] for n in elem])
            
            elem_sigma = np.mean([self.sigma[n] for n in elem])
            
            ke = np.zeros((4, 4), dtype=complex)
            
            for i, xi in enumerate(gp):
                for j, eta in enumerate(gp):
                    N, dN_dxi, dN_deta = self._shape_functions(xi, eta)
                    
                    dx_dxi = np.sum(dN_dxi * x_coords)
                    dx_deta = np.sum(dN_deta * x_coords)
                    dz_dxi = np.sum(dN_dxi * z_coords)
                    dz_deta = np.sum(dN_deta * z_coords)
                    
                    det_J = dx_dxi * dz_deta - dx_deta * dz_dxi
                    
                    inv_J = np.array([
                        [dz_deta, -dx_deta],
                        [-dz_dxi, dx_dxi]
                    ]) / det_J
                    
                    dN_dx = inv_J[0, 0] * dN_dxi + inv_J[0, 1] * dN_deta
                    dN_dz = inv_J[1, 0] * dN_dxi + inv_J[1, 1] * dN_deta
                    
                    for ni in range(4):
                        for nj in range(4):
                            ke[ni, nj] += (dN_dx[ni] * dN_dx[nj] + 
                                           dN_dz[ni] * dN_dz[nj] - 
                                           1j * omega * MU0 * elem_sigma * N[ni] * N[nj]) * \
                                          det_J * gw[i] * gw[j]
            
            for ni in range(4):
                for nj in range(4):
                    rows.append(elem[ni])
                    cols.append(elem[nj])
                    data.append(ke[ni, nj])
        
        K = csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        
        surface_nodes = [j for j in range(self.nx)]
        for n in surface_nodes:
            K[n, :] = 0
            K[n, n] = 1
            b[n] = E0
        
        E_field = spsolve(K, b)
        return E_field

    def get_surface_electric_field(self) -> np.ndarray:
        return np.array([self.sigma[j] for j in range(self.nx)])


class AdvancedGICCalculator:
    def __init__(self, earth_model: LayeredEarthModel):
        self.substations: Dict[int, Substation] = {}
        self.lines: List[TransmissionLine] = []
        self.earth_model = earth_model
        self.fem_model = None

    def add_substation(self, substation: Substation):
        self.substations[substation.id] = substation

    def add_line(self, line: TransmissionLine):
        self.lines.append(line)

    def setup_fem_model(self, x_range: float = 500000, depth: float = 100000,
                        nx: int = 50, nz: int = 30):
        self.fem_model = FEMEarthModel(
            x_min=-x_range/2, x_max=x_range/2,
            z_min=-depth, z_max=0,
            nx=nx, nz=nz,
            layers=self.earth_model.layers
        )

    def calculate_surface_electric_field(self, frequencies: np.ndarray,
                                          B_spectrum: np.ndarray,
                                          use_fem: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        n_freq = len(frequencies)
        E_surface = np.zeros(n_freq, dtype=complex)
        
        if use_fem and self.fem_model is not None:
            for i, (f, B_amp) in enumerate(zip(frequencies, B_spectrum)):
                omega = 2 * np.pi * f
                E0 = B_amp * omega * 1000
                E_field = self.fem_model.solve(omega, E0)
                E_surface[i] = np.mean(E_field[:self.fem_model.nx])
        else:
            for i, (f, B_amp) in enumerate(zip(frequencies, B_spectrum)):
                Z = self.earth_model.compute_impedance(f)
                E_surface[i] = Z * B_amp
        
        return frequencies, E_surface

    def calculate_time_domain_electric_field(self, time: np.ndarray,
                                              dB_dt: np.ndarray,
                                              use_fem: bool = False) -> np.ndarray:
        n = len(time)
        dt = time[1] - time[0]
        
        freqs = np.fft.fftfreq(n, dt)
        positive_mask = freqs >= 0
        
        dB_dt_fft = np.fft.fft(dB_dt)
        
        B_fft = np.zeros_like(dB_dt_fft, dtype=complex)
        B_fft[positive_mask] = dB_dt_fft[positive_mask] / (2j * np.pi * freqs[positive_mask] + 1e-10)
        B_fft[~positive_mask] = np.conj(B_fft[positive_mask][::-1])
        
        freqs_pos = freqs[positive_mask]
        B_pos = B_fft[positive_mask]
        
        _, E_spectrum = self.calculate_surface_electric_field(
            np.abs(freqs_pos), np.abs(B_pos), use_fem
        )
        
        E_fft = np.zeros_like(B_fft, dtype=complex)
        E_fft[positive_mask] = E_spectrum * np.exp(1j * np.angle(B_pos))
        E_fft[~positive_mask] = np.conj(E_fft[positive_mask][::-1])
        
        E_time = np.fft.ifft(E_fft).real
        
        return E_time

    def calculate_line_voltage_fd(self, line: TransmissionLine,
                                   frequencies: np.ndarray,
                                   E_spectrum: np.ndarray,
                                   e_field_angle: float = 0.0) -> np.ndarray:
        angle_rad = math.radians(e_field_angle)
        sub1 = self.substations[line.from_substation]
        sub2 = self.substations[line.to_substation]
        
        dx = sub2.x - sub1.x
        dy = sub2.y - sub1.y
        line_length = math.sqrt(dx**2 + dy**2)
        
        if line_length > 0:
            cos_theta = dx / line_length
            sin_theta = dy / line_length
        else:
            cos_theta = 1.0
            sin_theta = 0.0
        
        E_eff = E_spectrum * (cos_theta * math.cos(angle_rad) + sin_theta * math.sin(angle_rad))
        V = E_eff * line.length * 1000.0
        
        return V

    def build_nodal_matrix(self) -> Tuple[np.ndarray, List[int]]:
        n = len(self.substations)
        node_ids = sorted(self.substations.keys())
        node_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        Y = np.zeros((n, n))
        
        for line in self.lines:
            i = node_idx[line.from_substation]
            j = node_idx[line.to_substation]
            y = 1.0 / line.resistance
            
            Y[i, i] += y
            Y[j, j] += y
            Y[i, j] -= y
            Y[j, i] -= y
        
        for nid in node_ids:
            i = node_idx[nid]
            yg = 1.0 / self.substations[nid].grounding_resistance
            Y[i, i] += yg
        
        return Y, node_ids

    def calculate_gic(self, time: np.ndarray, dB_dt: np.ndarray,
                       e_field_angle: float = 0.0,
                       use_fem: bool = False) -> Dict[int, np.ndarray]:
        E_time = self.calculate_time_domain_electric_field(time, dB_dt, use_fem)
        
        n = len(self.substations)
        node_ids = sorted(self.substations.keys())
        node_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        Y, _ = self.build_nodal_matrix()
        
        n_times = len(time)
        V = np.zeros((n, n_times))
        J = np.zeros((n, n_times))
        
        for t in range(n_times):
            I = np.zeros(n)
            for line in self.lines:
                i = node_idx[line.from_substation]
                j = node_idx[line.to_substation]
                
                sub1 = self.substations[line.from_substation]
                sub2 = self.substations[line.to_substation]
                dx = sub2.x - sub1.x
                dy = sub2.y - sub1.y
                line_len = math.sqrt(dx**2 + dy**2)
                
                if line_len > 0:
                    cos_theta = dx / line_len
                    sin_theta = dy / line_len
                else:
                    cos_theta = 1.0
                    sin_theta = 0.0
                
                angle_rad = math.radians(e_field_angle)
                E_eff = E_time[t] * (cos_theta * math.cos(angle_rad) + sin_theta * math.sin(angle_rad))
                V_line = E_eff * line.length * 1000.0
                
                y = 1.0 / line.resistance
                I[i] += V_line * y
                I[j] -= V_line * y
            
            try:
                V[:, t] = np.linalg.solve(Y, I)
            except np.linalg.LinAlgError:
                V[:, t] = np.linalg.lstsq(Y, I, rcond=None)[0]
        
        for nid in node_ids:
            i = node_idx[nid]
            J[i, :] = V[i, :] / self.substations[nid].grounding_resistance
        
        gic_results = {}
        for nid in node_ids:
            i = node_idx[nid]
            gic_results[nid] = J[i, :]
        
        return gic_results, E_time


def create_sample_layered_model() -> LayeredEarthModel:
    layers = [
        Layer(thickness=500, resistivity=100),
        Layer(thickness=5000, resistivity=500),
        Layer(thickness=20000, resistivity=10),
        Layer(thickness=float('inf'), resistivity=1000)
    ]
    return LayeredEarthModel(layers)


def create_sample_grid() -> AdvancedGICCalculator:
    earth_model = create_sample_layered_model()
    calc = AdvancedGICCalculator(earth_model)
    
    calc.add_substation(Substation(1, "变电站A", 0, 0, 0.5))
    calc.add_substation(Substation(2, "变电站B", 100, 0, 0.3))
    calc.add_substation(Substation(3, "变电站C", 50, 80, 0.4))
    calc.add_substation(Substation(4, "变电站D", 150, 60, 0.6))
    
    calc.add_line(TransmissionLine(1, 1, 2, 100, 0.05))
    calc.add_line(TransmissionLine(2, 1, 3, 95, 0.048))
    calc.add_line(TransmissionLine(3, 2, 3, 90, 0.045))
    calc.add_line(TransmissionLine(4, 2, 4, 70, 0.035))
    calc.add_line(TransmissionLine(5, 3, 4, 100, 0.05))
    
    return calc


def generate_storm_profile(duration_hours: float = 24.0,
                             dt_minutes: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    n_points = int(duration_hours * 60 / dt_minutes)
    t = np.linspace(0, duration_hours * 3600, n_points)
    t_hours = t / 3600
    
    main_phase = 6.0 * 3600
    peak_time = 12.0 * 3600
    
    dB = np.zeros_like(t)
    for i, ti in enumerate(t_hours):
        if ti < 6.0:
            dB[i] = -200 * (1 - np.exp(-ti / 1.5))
        else:
            decay = ti - 6.0
            dB[i] = -200 * np.exp(-decay / 8.0) + \
                    50 * np.sin(2 * np.pi * (ti - 12.0) / 3.0) * np.exp(-decay / 6.0)
    
    dB_dt = np.gradient(dB, t / 60.0)
    
    return t, dB_dt


def main():
    print("=" * 70)
    print("多层地电阻率模型GIC计算程序")
    print("=" * 70)
    
    calc = create_sample_grid()
    
    print("\n地电阻率模型参数:")
    for i, layer in enumerate(calc.earth_model.layers):
        if layer.thickness == float('inf'):
            print(f"  层{i+1}: 半空间, 电阻率={layer.resistivity} Ω·m")
        else:
            print(f"  层{i+1}: 厚度={layer.thickness}m, 电阻率={layer.resistivity} Ω·m")
    
    print("\n计算MT响应...")
    frequencies = np.logspace(-4, 2, 100)
    rho_app, phase = calc.earth_model.compute_apparent_resistivity(frequencies)
    
    print("生成磁暴剖面...")
    t, dB_dt = generate_storm_profile(duration_hours=24.0, dt_minutes=5.0)
    
    print("计算GIC (使用分层介质解析解)...")
    gic_results, E_time = calc.calculate_gic(t, dB_dt, e_field_angle=45.0, use_fem=False)
    
    print("\n" + "=" * 70)
    print("GIC计算结果")
    print("=" * 70)
    
    max_gic = {}
    for sid, gic in gic_results.items():
        sub = calc.substations[sid]
        max_gic[sid] = np.max(np.abs(gic))
        print(f"\n{sub.name}:")
        print(f"  最大GIC: {max_gic[sid]:.2f} A")
        print(f"  平均GIC: {np.mean(np.abs(gic)):.2f} A")
        print(f"  均方根GIC: {np.sqrt(np.mean(gic**2)):.2f} A")
    
    print("\n" + "=" * 70)
    print("电网拓扑信息:")
    print("=" * 70)
    for sub in calc.substations.values():
        print(f"\n{sub.name}:")
        print(f"  接地电阻: {sub.grounding_resistance} Ω")
    
    print(f"\n输电线路数量: {len(calc.lines)}")
    print(f"变电站数量: {len(calc.substations)}")
    
    fig = plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 2, 1)
    plt.loglog(frequencies, rho_app, 'b-', linewidth=1.5)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('视电阻率 (Ω·m)')
    plt.title('MT视电阻率曲线')
    plt.grid(True, alpha=0.3, which='both')
    
    plt.subplot(3, 2, 2)
    plt.semilogx(frequencies, phase, 'r-', linewidth=1.5)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('相位 (度)')
    plt.title('MT相位曲线')
    plt.grid(True, alpha=0.3, which='both')
    
    plt.subplot(3, 2, 3)
    plt.plot(t / 3600, dB_dt, 'b-', linewidth=1.5)
    plt.xlabel('时间 (小时)')
    plt.ylabel('dB/dt (nT/min)')
    plt.title('地磁场变化率')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    plt.plot(t / 3600, E_time * 1000, 'g-', linewidth=1.5)
    plt.xlabel('时间 (小时)')
    plt.ylabel('地表电场 (mV/km)')
    plt.title('感应地表电场')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    for sid, gic in gic_results.items():
        sub = calc.substations[sid]
        plt.plot(t / 3600, gic, label=f'{sub.name}', linewidth=1.5)
    plt.xlabel('时间 (小时)')
    plt.ylabel('GIC (A)')
    plt.title('各变电站中性点GIC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_gic_results.png', dpi=150, bbox_inches='tight')
    print("\n结果图已保存为: advanced_gic_results.png")
    
    depths = np.linspace(0, 50000, 200)
    E_depth, _ = calc.earth_model.compute_electric_field_depth(
        1e-4, 1e-9, depths
    )
    
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(np.abs(E_depth), depths / 1000, 'b-', linewidth=1.5)
    plt.xlabel('电场幅值 (V/m)')
    plt.ylabel('深度 (km)')
    plt.title('电场随深度分布 (f=0.0001 Hz)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.savefig('e_field_depth.png', dpi=150, bbox_inches='tight')
    print("电场深度分布图已保存为: e_field_depth.png")


if __name__ == "__main__":
    main()
