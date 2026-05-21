import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import warnings

@dataclass
class Bus:
    bus_num: int
    bus_type: str
    V: float = 1.0
    theta: float = 0.0
    P_load: float = 0.0
    Q_load: float = 0.0
    P_gen: float = 0.0
    Q_gen: float = 0.0
    Q_max: float = 999.0
    Q_min: float = -999.0

@dataclass
class Line:
    from_bus: int
    to_bus: int
    R: float
    X: float
    B: float = 0.0
    tap: float = 1.0

@dataclass
class Generator:
    bus_num: int
    P_gen: float
    V_set: float
    Q_max: float = 999.0
    Q_min: float = -999.0

class NewtonRaphsonPowerFlow:
    def __init__(self, buses: List[Bus], lines: List[Line], generators: List[Generator]):
        self.buses = buses
        self.lines = lines
        self.generators = generators
        self.n_buses = len(buses)
        self.bus_idx = {bus.bus_num: i for i, bus in enumerate(buses)}
        self.Y_bus = None
        self._original_bus_types = {bus.bus_num: bus.bus_type for bus in buses}
        self._original_voltages = {bus.bus_num: bus.V for bus in buses}
        self._build_Y_bus()
        self._apply_generator_data()

    def _build_Y_bus(self):
        self.Y_bus = np.zeros((self.n_buses, self.n_buses), dtype=complex)
        for line in self.lines:
            i = self.bus_idx[line.from_bus]
            j = self.bus_idx[line.to_bus]
            Y_series = 1.0 / (line.R + 1j * line.X)
            Y_shunt = 1j * line.B / 2.0
            tap_ratio = line.tap
            self.Y_bus[i, i] += Y_series / (tap_ratio ** 2) + Y_shunt
            self.Y_bus[j, j] += Y_series + Y_shunt
            self.Y_bus[i, j] -= Y_series / tap_ratio
            self.Y_bus[j, i] -= Y_series / tap_ratio

    def _apply_generator_data(self):
        for gen in self.generators:
            idx = self.bus_idx[gen.bus_num]
            self.buses[idx].P_gen = gen.P_gen
            self.buses[idx].V = gen.V_set
            self.buses[idx].Q_max = gen.Q_max
            self.buses[idx].Q_min = gen.Q_min

    def reset_solution(self):
        for bus in self.buses:
            bus.bus_type = self._original_bus_types[bus.bus_num]
            bus.V = self._original_voltages[bus.bus_num]
            bus.theta = 0.0

    def _get_pq_pv_buses(self):
        pq_buses = []
        pv_buses = []
        slack_idx = None
        for i, bus in enumerate(self.buses):
            if bus.bus_type == 'PQ':
                pq_buses.append(i)
            elif bus.bus_type == 'PV':
                pv_buses.append(i)
            elif bus.bus_type == 'Slack':
                slack_idx = i
        return pq_buses, pv_buses, slack_idx

    def _calculate_power_mismatch(self, V, theta, pq_buses, pv_buses, slack_idx):
        n = self.n_buses
        P_calc = np.zeros(n)
        Q_calc = np.zeros(n)
        for i in range(n):
            for k in range(n):
                P_calc[i] += V[i] * V[k] * (self.Y_bus[i, k].real * np.cos(theta[i] - theta[k]) + 
                                             self.Y_bus[i, k].imag * np.sin(theta[i] - theta[k]))
                Q_calc[i] += V[i] * V[k] * (self.Y_bus[i, k].real * np.sin(theta[i] - theta[k]) - 
                                             self.Y_bus[i, k].imag * np.cos(theta[i] - theta[k]))
        P_spec = np.array([bus.P_gen - bus.P_load for bus in self.buses])
        Q_spec = np.array([bus.Q_gen - bus.Q_load for bus in self.buses])
        delta_P = P_spec - P_calc
        delta_Q = Q_spec - Q_calc
        mismatch = []
        for i in range(n):
            if i != slack_idx:
                mismatch.append(delta_P[i])
        for i in pq_buses:
            mismatch.append(delta_Q[i])
        return np.array(mismatch), P_calc, Q_calc

    def _build_jacobian(self, V, theta, pq_buses, pv_buses, slack_idx):
        n = self.n_buses
        n_pq = len(pq_buses)
        n_pv = len(pv_buses)
        n_state = (n - 1) + n_pq
        J = np.zeros((n_state, n_state))
        state_buses = [i for i in range(n) if i != slack_idx]
        theta_idx_map = {i: idx for idx, i in enumerate(state_buses)}
        V_idx_map = {i: idx + len(state_buses) for idx, i in enumerate(pq_buses)}
        for i in state_buses:
            for k in state_buses:
                if i != k:
                    J[theta_idx_map[i], theta_idx_map[k]] = V[i] * V[k] * (
                        self.Y_bus[i, k].real * np.sin(theta[i] - theta[k]) -
                        self.Y_bus[i, k].imag * np.cos(theta[i] - theta[k])
                    )
                else:
                    for m in range(n):
                        if m != i:
                            J[theta_idx_map[i], theta_idx_map[i]] += V[i] * V[m] * (
                                -self.Y_bus[i, m].real * np.sin(theta[i] - theta[m]) +
                                self.Y_bus[i, m].imag * np.cos(theta[i] - theta[m])
                            )
            if i in pq_buses:
                for k in pq_buses:
                    if i != k:
                        J[theta_idx_map[i], V_idx_map[k]] = V[i] * (
                            self.Y_bus[i, k].real * np.cos(theta[i] - theta[k]) +
                            self.Y_bus[i, k].imag * np.sin(theta[i] - theta[k])
                        )
                    else:
                        J[theta_idx_map[i], V_idx_map[i]] = 2 * V[i] * self.Y_bus[i, i].real
                        for m in range(n):
                            if m != i:
                                J[theta_idx_map[i], V_idx_map[i]] += V[m] * (
                                    self.Y_bus[i, m].real * np.cos(theta[i] - theta[m]) +
                                    self.Y_bus[i, m].imag * np.sin(theta[i] - theta[m])
                                )
        for i in pq_buses:
            for k in state_buses:
                if i != k:
                    J[V_idx_map[i], theta_idx_map[k]] = -V[i] * V[k] * (
                        self.Y_bus[i, k].real * np.cos(theta[i] - theta[k]) +
                        self.Y_bus[i, k].imag * np.sin(theta[i] - theta[k])
                    )
                else:
                    for m in range(n):
                        if m != i:
                            J[V_idx_map[i], V_idx_map[i]] += V[i] * V[m] * (
                                self.Y_bus[i, m].real * np.cos(theta[i] - theta[m]) +
                                self.Y_bus[i, m].imag * np.sin(theta[i] - theta[m])
                            )
            for k in pq_buses:
                if i != k:
                    J[V_idx_map[i], V_idx_map[k]] = V[i] * (
                        self.Y_bus[i, k].real * np.sin(theta[i] - theta[k]) -
                        self.Y_bus[i, k].imag * np.cos(theta[i] - theta[k])
                    )
                else:
                    J[V_idx_map[i], V_idx_map[i]] = -2 * V[i] * self.Y_bus[i, i].imag
                    for m in range(n):
                        if m != i:
                            J[V_idx_map[i], V_idx_map[i]] += V[m] * (
                                self.Y_bus[i, m].real * np.sin(theta[i] - theta[m]) -
                                self.Y_bus[i, m].imag * np.cos(theta[i] - theta[m])
                            )
        return J

    def solve(self, max_iter: int = 100, tolerance: float = 1e-6, verbose: bool = False, max_q_corrections: int = 10):
        V = np.array([bus.V for bus in self.buses], dtype=float)
        theta = np.array([bus.theta for bus in self.buses], dtype=float)
        pv_converted_to_pq = []
        for q_correction in range(max_q_corrections):
            pq_buses, pv_buses, slack_idx = self._get_pq_pv_buses()
            state_buses = [i for i in range(self.n_buses) if i != slack_idx]
            for iteration in range(max_iter):
                mismatch, P_calc, Q_calc = self._calculate_power_mismatch(V, theta, pq_buses, pv_buses, slack_idx)
                max_mismatch = np.max(np.abs(mismatch))
                if verbose:
                    print(f"Q-correction {q_correction + 1}, Iteration {iteration + 1}: Max mismatch = {max_mismatch:.2e}")
                if max_mismatch < tolerance:
                    if verbose:
                        print(f"Converged in {iteration + 1} iterations!")
                    break
                J = self._build_jacobian(V, theta, pq_buses, pv_buses, slack_idx)
                try:
                    dx = np.linalg.solve(J, mismatch)
                except np.linalg.LinAlgError:
                    raise RuntimeError("Jacobian matrix is singular!")
                for idx, i in enumerate(state_buses):
                    theta[i] += dx[idx]
                for idx, i in enumerate(pq_buses):
                    V[i] += dx[len(state_buses) + idx]
            else:
                raise RuntimeError(f"Did not converge in {max_iter} iterations!")
            _, _, Q_calc_final = self._calculate_power_mismatch(V, theta, pq_buses, pv_buses, slack_idx)
            q_violation = False
            for i, bus in enumerate(self.buses):
                if bus.bus_type == 'PV':
                    Q_gen = Q_calc_final[i] + bus.Q_load
                    if Q_gen > bus.Q_max + 1e-10:
                        if verbose:
                            print(f"  -> Bus {bus.bus_num} Q_gen = {Q_gen:.4f} > Q_max = {bus.Q_max:.4f}, convert to PQ, fix Q_gen = {bus.Q_max:.4f}")
                        bus.bus_type = 'PQ'
                        bus.Q_gen = bus.Q_max
                        pv_converted_to_pq.append(bus.bus_num)
                        q_violation = True
                    elif Q_gen < bus.Q_min - 1e-10:
                        if verbose:
                            print(f"  -> Bus {bus.bus_num} Q_gen = {Q_gen:.4f} < Q_min = {bus.Q_min:.4f}, convert to PQ, fix Q_gen = {bus.Q_min:.4f}")
                        bus.bus_type = 'PQ'
                        bus.Q_gen = bus.Q_min
                        pv_converted_to_pq.append(bus.bus_num)
                        q_violation = True
            if not q_violation:
                if verbose and len(pv_converted_to_pq) > 0:
                    print(f"Converted PV buses to PQ: {pv_converted_to_pq}")
                if verbose:
                    print("No Q violations, solution is final!")
                break
        else:
            if verbose:
                print(f"Reached maximum Q corrections ({max_q_corrections}), proceeding with current solution")
        _, P_calc_final, Q_calc_final = self._calculate_power_mismatch(V, theta, *self._get_pq_pv_buses())
        for i, bus in enumerate(self.buses):
            bus.V = V[i]
            bus.theta = theta[i]
            bus.P_gen = P_calc_final[i] + bus.P_load if bus.bus_type == 'Slack' else bus.P_gen
            bus.Q_gen = Q_calc_final[i] + bus.Q_load if bus.bus_type in ['Slack', 'PV'] else bus.Q_gen
        results = self._get_results()
        results['pv_converted_to_pq'] = pv_converted_to_pq
        return results

    def _get_results(self):
        results = {
            'buses': [],
            'line_flows': []
        }
        for bus in self.buses:
            results['buses'].append({
                'bus_num': bus.bus_num,
                'bus_type': bus.bus_type,
                'V_mag': bus.V,
                'V_angle_deg': np.degrees(bus.theta),
                'P_gen': bus.P_gen,
                'Q_gen': bus.Q_gen,
                'P_load': bus.P_load,
                'Q_load': bus.Q_load
            })
        for line in self.lines:
            i = self.bus_idx[line.from_bus]
            j = self.bus_idx[line.to_bus]
            V_i = self.buses[i].V * np.exp(1j * self.buses[i].theta)
            V_j = self.buses[j].V * np.exp(1j * self.buses[j].theta)
            Y_series = 1.0 / (line.R + 1j * line.X)
            Y_shunt = 1j * line.B / 2.0
            tap_ratio = line.tap
            I_ij = (V_i / tap_ratio - V_j) * Y_series + V_i * Y_shunt / (tap_ratio ** 2)
            I_ji = (V_j - V_i / tap_ratio) * Y_series + V_j * Y_shunt
            S_ij = V_i * np.conj(I_ij)
            S_ji = V_j * np.conj(I_ji)
            results['line_flows'].append({
                'from_bus': line.from_bus,
                'to_bus': line.to_bus,
                'P_from': S_ij.real,
                'Q_from': S_ij.imag,
                'P_to': S_ji.real,
                'Q_to': S_ji.imag,
                'P_loss': S_ij.real + S_ji.real,
                'Q_loss': S_ij.imag + S_ji.imag
            })
        return results

def example_3bus_system():
    buses = [
        Bus(bus_num=1, bus_type='Slack', V=1.04, theta=0.0),
        Bus(bus_num=2, bus_type='PV', P_load=0.0, Q_load=0.0, P_gen=0.5),
        Bus(bus_num=3, bus_type='PQ', P_load=1.2, Q_load=0.6)
    ]
    lines = [
        Line(from_bus=1, to_bus=2, R=0.02, X=0.06, B=0.03),
        Line(from_bus=1, to_bus=3, R=0.08, X=0.24, B=0.02),
        Line(from_bus=2, to_bus=3, R=0.06, X=0.18, B=0.02)
    ]
    generators = [
        Generator(bus_num=1, P_gen=0.0, V_set=1.04),
        Generator(bus_num=2, P_gen=0.5, V_set=1.02)
    ]
    return buses, lines, generators

def example_ieee14bus_system():
    buses = [
        Bus(bus_num=1, bus_type='Slack', V=1.06, theta=0.0),
        Bus(bus_num=2, bus_type='PV', P_load=0.217, Q_load=0.127, P_gen=0.4),
        Bus(bus_num=3, bus_type='PV', P_load=0.942, Q_load=0.19, P_gen=0.0),
        Bus(bus_num=4, bus_type='PQ', P_load=0.478, Q_load=0.039),
        Bus(bus_num=5, bus_type='PQ', P_load=0.076, Q_load=0.016),
        Bus(bus_num=6, bus_type='PV', P_load=0.112, Q_load=0.075, P_gen=0.0),
        Bus(bus_num=7, bus_type='PQ', P_load=0.0, Q_load=0.0),
        Bus(bus_num=8, bus_type='PV', P_load=0.0, Q_load=0.0, P_gen=0.0),
        Bus(bus_num=9, bus_type='PQ', P_load=0.295, Q_load=0.166),
        Bus(bus_num=10, bus_type='PQ', P_load=0.09, Q_load=0.058),
        Bus(bus_num=11, bus_type='PQ', P_load=0.035, Q_load=0.018),
        Bus(bus_num=12, bus_type='PQ', P_load=0.061, Q_load=0.016),
        Bus(bus_num=13, bus_type='PQ', P_load=0.135, Q_load=0.058),
        Bus(bus_num=14, bus_type='PQ', P_load=0.149, Q_load=0.05)
    ]
    lines = [
        Line(from_bus=1, to_bus=2, R=0.01938, X=0.05917, B=0.0528),
        Line(from_bus=1, to_bus=5, R=0.05403, X=0.22304, B=0.0492),
        Line(from_bus=2, to_bus=3, R=0.04699, X=0.19797, B=0.0438),
        Line(from_bus=2, to_bus=4, R=0.05811, X=0.17632, B=0.034),
        Line(from_bus=2, to_bus=5, R=0.05695, X=0.17388, B=0.0346),
        Line(from_bus=3, to_bus=4, R=0.06701, X=0.17103, B=0.0128),
        Line(from_bus=4, to_bus=5, R=0.01335, X=0.04211, B=0.0),
        Line(from_bus=4, to_bus=7, R=0.0, X=0.20912, B=0.0, tap=0.978),
        Line(from_bus=4, to_bus=9, R=0.0, X=0.55618, B=0.0, tap=0.969),
        Line(from_bus=5, to_bus=6, R=0.0, X=0.25202, B=0.0, tap=0.932),
        Line(from_bus=6, to_bus=11, R=0.09498, X=0.1989, B=0.0),
        Line(from_bus=6, to_bus=12, R=0.12291, X=0.25581, B=0.0),
        Line(from_bus=6, to_bus=13, R=0.06615, X=0.13027, B=0.0),
        Line(from_bus=7, to_bus=8, R=0.0, X=0.17615, B=0.0),
        Line(from_bus=7, to_bus=9, R=0.0, X=0.11001, B=0.0),
        Line(from_bus=9, to_bus=10, R=0.03181, X=0.0845, B=0.0),
        Line(from_bus=9, to_bus=14, R=0.12711, X=0.27038, B=0.0),
        Line(from_bus=10, to_bus=11, R=0.08205, X=0.19207, B=0.0),
        Line(from_bus=12, to_bus=13, R=0.22092, X=0.19988, B=0.0),
        Line(from_bus=13, to_bus=14, R=0.17093, X=0.34802, B=0.0)
    ]
    generators = [
        Generator(bus_num=1, P_gen=0.0, V_set=1.06),
        Generator(bus_num=2, P_gen=0.4, V_set=1.045),
        Generator(bus_num=3, P_gen=0.0, V_set=1.01),
        Generator(bus_num=6, P_gen=0.0, V_set=1.07),
        Generator(bus_num=8, P_gen=0.0, V_set=1.09)
    ]
    return buses, lines, generators

def print_results(results):
    if 'pv_converted_to_pq' in results and len(results['pv_converted_to_pq']):
        print("\n" + "="*80)
        print(f"PV BUSES CONVERTED TO PQ DUE TO Q LIMITS: " + str(results['pv_converted_to_pq']))
        print("="*80)
    print("\n" + "="*80)
    print("BUS RESULTS")
    print("="*80)
    print(f"{'Bus':>4} {'Type':>6} {'V(pu)':>8} {'Angle(deg)':>12} {'P_gen(pu)':>10} {'Q_gen(pu)':>10} {'P_load(pu)':>10} {'Q_load(pu)':>10}")
    print("-"*80)
    for bus in results['buses']:
        bus_type_str = bus['bus_type']
        if 'pv_converted_to_pq' in results and bus['bus_num'] in results['pv_converted_to_pq']:
            bus_type_str = 'PQ*'
        print(f"{bus['bus_num']:4d} {bus_type_str:>6} {bus['V_mag']:8.4f} {bus['V_angle_deg']:12.4f} {bus['P_gen']:10.4f} {bus['Q_gen']:10.4f} {bus['P_load']:10.4f} {bus['Q_load']:10.4f}")
    print("\n" + "="*80)
    print("LINE FLOW RESULTS")
    print("="*80)
    print(f"{'From':>4} {'To':>4} {'P_from(pu)':>12} {'Q_from(pu)':>12} {'P_to(pu)':>12} {'Q_to(pu)':>12} {'P_loss(pu)':>12} {'Q_loss(pu)':>12}")
    print("-"*80)
    for flow in results['line_flows']:
        print(f"{flow['from_bus']:4d} {flow['to_bus']:4d} {flow['P_from']:12.4f} {flow['Q_from']:12.4f} {flow['P_to']:12.4f} {flow['Q_to']:12.4f} {flow['P_loss']:12.4f} {flow['Q_loss']:12.4f}")

def example_qviolation_test():
    buses = [
        Bus(bus_num=1, bus_type='Slack', V=1.04, theta=0.0),
        Bus(bus_num=2, bus_type='PV', P_load=0.0, Q_load=0.0, P_gen=0.8, Q_max=0.2, Q_min=-0.1),
        Bus(bus_num=3, bus_type='PQ', P_load=1.5, Q_load=0.8)
    ]
    lines = [
        Line(from_bus=1, to_bus=2, R=0.02, X=0.06, B=0.03),
        Line(from_bus=1, to_bus=3, R=0.08, X=0.24, B=0.02),
        Line(from_bus=2, to_bus=3, R=0.06, X=0.18, B=0.02)
    ]
    generators = [
        Generator(bus_num=1, P_gen=0.0, V_set=1.04),
        Generator(bus_num=2, P_gen=0.8, V_set=1.02, Q_max=0.2, Q_min=-0.1)
    ]
    return buses, lines, generators

@dataclass
class WindFarm:
    bus_num: int
    capacity: float
    mean_speed: float = 8.5
    std_speed: float = 2.5
    cut_in: float = 3.0
    rated_speed: float = 12.0
    cut_out: float = 25.0
    power_factor: float = 0.95

    def sample_power(self, n_samples: int = 1) -> np.ndarray:
        speeds = np.random.normal(self.mean_speed, self.std_speed, n_samples)
        powers = np.zeros(n_samples)
        for i, v in enumerate(speeds):
            if v < self.cut_in or v > self.cut_out:
                powers[i] = 0
            elif v >= self.rated_speed:
                powers[i] = self.capacity
            else:
                powers[i] = self.capacity * ((v - self.cut_in) / (self.rated_speed - self.cut_in)) ** 3
        return np.clip(powers, 0, self.capacity)

@dataclass
class SolarFarm:
    bus_num: int
    capacity: float
    mean_irradiance: float = 600
    std_irradiance: float = 250
    std_temp: float = 5.0
    power_factor: float = 0.98

    def sample_power(self, n_samples: int = 1) -> np.ndarray:
        irradiance = np.random.normal(self.mean_irradiance, self.std_irradiance, n_samples)
        irradiance = np.clip(irradiance, 0, 1000)
        powers = self.capacity * (irradiance / 1000)
        return np.clip(powers, 0, self.capacity)

@dataclass
class UncertainLoad:
    bus_num: int
    base_P: float
    base_Q: float
    std_factor: float = 0.1

    def sample_load(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        P_factor = np.random.normal(1.0, self.std_factor, n_samples)
        Q_factor = np.random.normal(1.0, self.std_factor, n_samples)
        return self.base_P * P_factor, self.base_Q * Q_factor

class ProbabilisticPowerFlow:
    def __init__(self, base_buses: List[Bus], base_lines: List[Line], 
                 base_generators: List[Generator]):
        self.base_buses = base_buses
        self.base_lines = base_lines
        self.base_generators = base_generators
        self.wind_farms: List[WindFarm] = []
        self.solar_farms: List[SolarFarm] = []
        self.uncertain_loads: List[UncertainLoad] = []
        self.results = None

    def add_wind_farm(self, wind_farm: WindFarm):
        self.wind_farms.append(wind_farm)

    def add_solar_farm(self, solar_farm: SolarFarm):
        self.solar_farms.append(solar_farm)

    def add_uncertain_load(self, load: UncertainLoad):
        self.uncertain_loads.append(load)

    def _create_scenario(self, wind_powers: Dict[int, float], 
                         solar_powers: Dict[int, float],
                         load_pq: Dict[int, Tuple[float, float]]) -> Tuple[List[Bus], List[Generator]]:
        buses = []
        generators = []
        bus_gen_map = {gen.bus_num: gen for gen in self.base_generators}
        for bus in self.base_buses:
            new_bus = Bus(
                bus_num=bus.bus_num,
                bus_type=bus.bus_type,
                V=bus.V,
                theta=bus.theta,
                P_load=bus.P_load,
                Q_load=bus.Q_load,
                P_gen=bus.P_gen,
                Q_gen=bus.Q_gen,
                Q_max=bus.Q_max,
                Q_min=bus.Q_min
            )
            if bus.bus_num in load_pq:
                new_bus.P_load, new_bus.Q_load = load_pq[bus.bus_num]
            total_renewable_P = 0
            total_renewable_Q = 0
            if bus.bus_num in wind_powers:
                total_renewable_P += wind_powers[bus.bus_num]
                total_renewable_Q += wind_powers[bus.bus_num] * np.tan(np.arccos(0.95))
            if bus.bus_num in solar_powers:
                total_renewable_P += solar_powers[bus.bus_num]
                total_renewable_Q += solar_powers[bus.bus_num] * np.tan(np.arccos(0.98))
            if bus.bus_type == 'PQ':
                new_bus.P_gen += total_renewable_P
                new_bus.Q_gen += total_renewable_Q
            buses.append(new_bus)
        for gen in self.base_generators:
            generators.append(Generator(
                bus_num=gen.bus_num,
                P_gen=gen.P_gen,
                V_set=gen.V_set,
                Q_max=gen.Q_max,
                Q_min=gen.Q_min
            ))
        return buses, generators

    def run_monte_carlo(self, n_samples: int = 1000, verbose: bool = True, 
                         progress_interval: int = 100) -> Dict:
        if verbose:
            print(f"\nStarting Monte Carlo simulation with {n_samples} samples...")
            print(f"  Wind farms: {len(self.wind_farms)}")
            print(f"  Solar farms: {len(self.solar_farms)}")
            print(f"  Uncertain loads: {len(self.uncertain_loads)}")
        wind_samples = {}
        for wf in self.wind_farms:
            wind_samples[wf.bus_num] = wf.sample_power(n_samples)
        solar_samples = {}
        for sf in self.solar_farms:
            solar_samples[sf.bus_num] = sf.sample_power(n_samples)
        load_samples = {}
        for ul in self.uncertain_loads:
            load_samples[ul.bus_num] = ul.sample_load(n_samples)
        n_buses = len(self.base_buses)
        V_magnitudes = np.zeros((n_samples, n_buses))
        V_angles = np.zeros((n_samples, n_buses))
        line_flows = []
        successful = 0
        failed = 0
        for i in range(n_samples):
            if verbose and (i + 1) % progress_interval == 0:
                print(f"  Progress: {i + 1}/{n_samples} samples completed...")
            wind_p = {wf.bus_num: wind_samples[wf.bus_num][i] for wf in self.wind_farms}
            solar_p = {sf.bus_num: solar_samples[sf.bus_num][i] for sf in self.solar_farms}
            load_pq = {ul.bus_num: (load_samples[ul.bus_num][0][i], load_samples[ul.bus_num][1][i]) 
                       for ul in self.uncertain_loads}
            scenario_buses, scenario_generators = self._create_scenario(wind_p, solar_p, load_pq)
            try:
                pf = NewtonRaphsonPowerFlow(scenario_buses, self.base_lines, scenario_generators)
                result = pf.solve(verbose=False)
                for j, bus_result in enumerate(result['buses']):
                    V_magnitudes[i, j] = bus_result['V_mag']
                    V_angles[i, j] = bus_result['V_angle_deg']
                successful += 1
            except Exception as e:
                failed += 1
                if verbose and failed <= 5:
                    print(f"  Warning: Sample {i + 1} failed to converge: {str(e)}")
        if verbose:
            print(f"\nMonte Carlo simulation completed:")
            print(f"  Successful: {successful}/{n_samples} ({100*successful/n_samples:.1f}%)")
            print(f"  Failed: {failed}/{n_samples} ({100*failed/n_samples:.1f}%)")
        valid_idx = V_magnitudes[:, 0] != 0
        self.results = {
            'V_magnitudes': V_magnitudes[valid_idx],
            'V_angles': V_angles[valid_idx],
            'bus_numbers': [bus.bus_num for bus in self.base_buses],
            'n_samples': successful,
            'n_failed': failed,
            'wind_samples': wind_samples,
            'solar_samples': solar_samples
        }
        return self.results

    def compute_statistics(self) -> Dict:
        if self.results is None:
            raise RuntimeError("Run Monte Carlo simulation first!")
        stats = {}
        for idx, bus_num in enumerate(self.results['bus_numbers']):
            V = self.results['V_magnitudes'][:, idx]
            stats[bus_num] = {
                'mean': np.mean(V),
                'std': np.std(V),
                'min': np.min(V),
                'max': np.max(V),
                'median': np.median(V),
                '5th_percentile': np.percentile(V, 5),
                '95th_percentile': np.percentile(V, 95),
                'violation_094': np.sum(V < 0.94) / len(V),
                'violation_106': np.sum(V > 1.06) / len(V),
            }
        return stats

    def print_statistics(self):
        stats = self.compute_statistics()
        print("\n" + "="*100)
        print("PROBABILISTIC POWER FLOW RESULTS - VOLTAGE STATISTICS")
        print("="*100)
        print(f"{'Bus':>4} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'5th %':>10} {'95th %':>10} {'V<0.94':>10} {'V>1.06':>10}")
        print("-"*100)
        for bus_num in sorted(stats.keys()):
            s = stats[bus_num]
            print(f"{bus_num:4d} {s['mean']:10.4f} {s['std']:10.4f} {s['min']:10.4f} {s['max']:10.4f} "
                  f"{s['5th_percentile']:10.4f} {s['95th_percentile']:10.4f} "
                  f"{s['violation_094']:10.1%} {s['violation_106']:10.1%}")
        print("="*100)

    def compute_pdf(self, bus_num: int, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if self.results is None:
            raise RuntimeError("Run Monte Carlo simulation first!")
        idx = self.results['bus_numbers'].index(bus_num)
        V = self.results['V_magnitudes'][:, idx]
        hist, bin_edges = np.histogram(V, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist

def example_probabilistic_system():
    buses = [
        Bus(bus_num=1, bus_type='Slack', V=1.05, theta=0.0),
        Bus(bus_num=2, bus_type='PV', P_load=0.2, Q_load=0.1, P_gen=1.0, Q_max=1.5, Q_min=-0.5),
        Bus(bus_num=3, bus_type='PQ', P_load=1.2, Q_load=0.6),
        Bus(bus_num=4, bus_type='PQ', P_load=0.8, Q_load=0.4),
        Bus(bus_num=5, bus_type='PQ', P_load=0.5, Q_load=0.25)
    ]
    lines = [
        Line(from_bus=1, to_bus=2, R=0.01, X=0.05, B=0.02),
        Line(from_bus=1, to_bus=3, R=0.02, X=0.08, B=0.01),
        Line(from_bus=2, to_bus=3, R=0.01, X=0.04, B=0.015),
        Line(from_bus=2, to_bus=4, R=0.015, X=0.06, B=0.01),
        Line(from_bus=3, to_bus=5, R=0.02, X=0.07, B=0.01),
        Line(from_bus=4, to_bus=5, R=0.015, X=0.05, B=0.01)
    ]
    generators = [
        Generator(bus_num=1, P_gen=0.0, V_set=1.05),
        Generator(bus_num=2, P_gen=1.0, V_set=1.03)
    ]
    return buses, lines, generators

if __name__ == "__main__":
    print("="*80)
    print("TEST 1: 3-bus system (no Q violation)")
    print("="*80)
    buses, lines, generators = example_3bus_system()
    pf = NewtonRaphsonPowerFlow(buses, lines, generators)
    results = pf.solve(verbose=True)
    print_results(results)
    
    print("\n\n" + "="*80)
    print("TEST 2: PV node Q violation test (Bus 2 will hit Q limit and convert to PQ)")
    print("="*80)
    buses3, lines3, generators3 = example_qviolation_test()
    pf3 = NewtonRaphsonPowerFlow(buses3, lines3, generators3)
    results3 = pf3.solve(verbose=True)
    print_results(results3)
    
    print("\n\n" + "="*80)
    print("TEST 3: IEEE 14-bus system")
    print("="*80)
    buses2, lines2, generators2 = example_ieee14bus_system()
    pf2 = NewtonRaphsonPowerFlow(buses2, lines2, generators2)
    results2 = pf2.solve(verbose=True)
    print_results(results2)

    print("\n\n" + "="*100)
    print("TEST 4: Probabilistic Power Flow (Monte Carlo Simulation with Wind/Solar)")
    print("="*100)
    buses_pf, lines_pf, generators_pf = example_probabilistic_system()
    ppf = ProbabilisticPowerFlow(buses_pf, lines_pf, generators_pf)
    wind1 = WindFarm(bus_num=4, capacity=1.0, mean_speed=9.0, std_speed=3.0)
    wind2 = WindFarm(bus_num=5, capacity=0.8, mean_speed=8.0, std_speed=2.5)
    solar1 = SolarFarm(bus_num=3, capacity=0.5, mean_irradiance=550, std_irradiance=200)
    load1 = UncertainLoad(bus_num=3, base_P=1.2, base_Q=0.6, std_factor=0.15)
    load2 = UncertainLoad(bus_num=4, base_P=0.8, base_Q=0.4, std_factor=0.1)
    ppf.add_wind_farm(wind1)
    ppf.add_wind_farm(wind2)
    ppf.add_solar_farm(solar1)
    ppf.add_uncertain_load(load1)
    ppf.add_uncertain_load(load2)
    np.random.seed(42)
    results_ppf = ppf.run_monte_carlo(n_samples=500, progress_interval=100)
    ppf.print_statistics()
    print("\nVoltage PDF for Bus 3:")
    v_values, pdf_values = ppf.compute_pdf(bus_num=3, n_bins=20)
    print(f"{'Voltage (pu)':>15} {'PDF':>10}")
    print("-" * 25)
    for v, pdf in zip(v_values, pdf_values):
        if pdf > 0.01:
            print(f"{v:15.4f} {pdf:10.3f}")
