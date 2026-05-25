import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math


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


class GICCalculator:
    def __init__(self):
        self.substations: Dict[int, Substation] = {}
        self.lines: List[TransmissionLine] = []
        self.conductivity = 0.01

    def add_substation(self, substation: Substation):
        self.substations[substation.id] = substation

    def add_line(self, line: TransmissionLine):
        self.lines.append(line)

    def calculate_electric_field(self, dB_dt: np.ndarray, 
                                   lat: float = 40.0) -> np.ndarray:
        earth_radius = 6371000.0
        lat_rad = math.radians(lat)
        E = 2 * math.pi * earth_radius * np.cos(lat_rad) * dB_dt / 86400.0
        return E * 1e-9

    def calculate_line_voltage(self, line: TransmissionLine, 
                                 E: np.ndarray, angle: float = 0.0) -> np.ndarray:
        angle_rad = math.radians(angle)
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
        
        E_eff = E * (cos_theta * math.cos(angle_rad) + sin_theta * math.sin(angle_rad))
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

    def calculate_gic(self, dB_dt: np.ndarray, 
                        e_field_angle: float = 0.0,
                        lat: float = 40.0) -> Dict[int, np.ndarray]:
        E = self.calculate_electric_field(dB_dt, lat)
        
        n = len(self.substations)
        node_ids = sorted(self.substations.keys())
        node_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        Y, _ = self.build_nodal_matrix()
        
        n_times = len(dB_dt) if hasattr(dB_dt, '__len__') else 1
        V = np.zeros((n, n_times))
        J = np.zeros((n, n_times))
        
        for t in range(n_times):
            I = np.zeros(n)
            for line in self.lines:
                i = node_idx[line.from_substation]
                j = node_idx[line.to_substation]
                V_line = self.calculate_line_voltage(line, E[t], e_field_angle)
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
        
        return gic_results


def create_sample_grid() -> GICCalculator:
    calc = GICCalculator()
    
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
    t = np.linspace(0, duration_hours, n_points)
    
    main_phase = 6.0
    peak_time = 12.0
    
    dB = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < main_phase:
            dB[i] = -200 * (1 - np.exp(-ti / 1.5))
        else:
            decay = ti - main_phase
            dB[i] = -200 * np.exp(-decay / 8.0) + 50 * np.sin(2 * np.pi * (ti - peak_time) / 3.0) * np.exp(-decay / 6.0)
    
    dB_dt = np.gradient(dB, t / 60.0)
    
    return t, dB_dt


def main():
    calc = create_sample_grid()
    t, dB_dt = generate_storm_profile(duration_hours=24.0, dt_minutes=5.0)
    
    gic_results = calc.calculate_gic(dB_dt, e_field_angle=45.0, lat=40.0)
    
    print("=" * 60)
    print("地磁感应电流(GIC)计算结果")
    print("=" * 60)
    
    max_gic = {}
    for sid, gic in gic_results.items():
        sub = calc.substations[sid]
        max_gic[sid] = np.max(np.abs(gic))
        print(f"\n{sub.name}:")
        print(f"  最大GIC: {max_gic[sid]:.2f} A")
        print(f"  平均GIC: {np.mean(np.abs(gic)):.2f} A")
    
    print("\n" + "=" * 60)
    print("电网拓扑信息:")
    print("=" * 60)
    for sub in calc.substations.values():
        print(f"\n{sub.name}:")
        print(f"  接地电阻: {sub.grounding_resistance} Ω")
    
    print(f"\n输电线路数量: {len(calc.lines)}")
    print(f"变电站数量: {len(calc.substations)}")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, dB_dt, 'b-', linewidth=1.5)
    plt.xlabel('时间 (小时)')
    plt.ylabel('地磁场变化率 dB/dt (nT/min)')
    plt.title('磁暴期间地磁场变化率')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    for sid, gic in gic_results.items():
        sub = calc.substations[sid]
        plt.plot(t, gic, label=f'{sub.name}', linewidth=1.5)
    plt.xlabel('时间 (小时)')
    plt.ylabel('GIC (A)')
    plt.title('各变电站中性点GIC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gic_results.png', dpi=150)
    print("\n结果图已保存为: gic_results.png")


if __name__ == "__main__":
    main()
