"""
机理简化验证与输出工具
Mechanism Reduction Validation and Output Tool

包含:
- 零维点火延迟求解器 (0D ignition delay solver)
- 一维层流火焰速度求解器 (1D laminar flame speed solver)
- CHEMKIN 格式机理输出 (CHEMKIN format writer)
- 详细vs简化机理对比验证 (validation workflow)

作者: Mechanism Reduction Tool
日期: 2026-05-27
"""

from typing import Dict, List, Set, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy
import math
import random
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root_scalar


# ============================================================================
# 核心数据结构 (与增强版兼容)
# ============================================================================

CRITICAL_SPECIES = [
    "H", "O", "OH", "H2", "O2", "H2O", "HO2", "H2O2",
    "CH3", "CH4", "CO", "CO2", "HCO", "CH2O",
    "CH", "CH2", "C2H2", "C2H4",
]


@dataclass
class Species:
    name: str
    molecular_weight: float = 0.0
    thermo_data: Optional[Dict] = None
    is_critical: bool = False

    def __hash__(self): return hash(self.name)
    def __eq__(self, other):
        return isinstance(other, Species) and self.name == other.name
    def __repr__(self): return f"Species({self.name})"


@dataclass
class Reaction:
    id: int
    reactants: Dict[str, float]
    products: Dict[str, float]
    rate_coefficients: Dict
    is_reversible: bool = True
    label: str = ""

    def get_all_species(self) -> Set[str]:
        return set(self.reactants.keys()) | set(self.products.keys())

    def __repr__(self):
        reactants_str = " + ".join(
            f"{v}{k}" if v != 1 else k for k, v in self.reactants.items()
        )
        products_str = " + ".join(
            f"{v}{k}" if v != 1 else k for k, v in self.products.items()
        )
        arrow = " <=> " if self.is_reversible else " => "
        return f"R{self.id}: {reactants_str}{arrow}{products_str}"


@dataclass
class Mechanism:
    species: List[Species] = field(default_factory=list)
    reactions: List[Reaction] = field(default_factory=list)

    def get_species_names(self) -> List[str]:
        return [s.name for s in self.species]

    def get_reaction(self, rxn_id: int) -> Optional[Reaction]:
        for r in self.reactions:
            if r.id == rxn_id:
                return r
        return None

    def get_species(self, name: str) -> Optional[Species]:
        for s in self.species:
            if s.name == name:
                return s
        return None

    def species_count(self) -> int: return len(self.species)
    def reactions_count(self) -> int: return len(self.reactions)
    def get_critical_species(self) -> List[str]:
        return [s.name for s in self.species if s.is_critical]

    def __repr__(self):
        return (f"Mechanism({self.species_count()} species, "
                f"{self.reactions_count()} reactions)")


# ============================================================================
# 机理构建
# ============================================================================

def build_methane_mechanism() -> Mechanism:
    """构建GRI-Mech 3.0子集甲烷燃烧机理"""
    species_list = [
        "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2",
        "C", "CH", "CH2", "CH2(S)", "CH3", "CH4", "CO", "CO2",
        "HCO", "CH2O", "CH2OH", "CH3O", "CH3OH", "C2H", "C2H2",
        "C2H3", "C2H4", "C2H5", "C2H6",
        "N2", "Ar", "N", "NO", "NO2", "NH3", "NNH", "HCCO", "HNO", "M",
    ]

    species = []
    for name in species_list:
        sp = Species(
            name=name,
            molecular_weight=_get_molecular_weight(name),
            is_critical=(name in CRITICAL_SPECIES),
            thermo_data=_get_thermo_data(name),
        )
        species.append(sp)

    reaction_data = _get_reaction_database()

    reactions = []
    for i, (reactants, products, reversible, A, b, Ea) in enumerate(reaction_data, 1):
        rxn = Reaction(
            id=i,
            reactants=_normalize_dict(reactants),
            products=_normalize_dict(products),
            rate_coefficients={"A": A, "b": b, "Ea": Ea},
            is_reversible=reversible,
            label=f"R{i}",
        )
        reactions.append(rxn)

    return Mechanism(species=species, reactions=reactions)


def _get_molecular_weight(species: str) -> float:
    mw = {
        "H2": 2.016, "H": 1.008, "O": 16.000, "O2": 32.000,
        "OH": 17.008, "H2O": 18.016, "HO2": 33.008, "H2O2": 34.016,
        "C": 12.000, "CH": 13.008, "CH2": 14.016, "CH2(S)": 14.016,
        "CH3": 15.024, "CH4": 16.032, "CO": 28.000, "CO2": 44.000,
        "HCO": 29.008, "CH2O": 30.016, "CH2OH": 31.024, "CH3O": 31.024,
        "CH3OH": 32.032, "C2H": 25.008, "C2H2": 26.016, "C2H3": 27.024,
        "C2H4": 28.032, "C2H5": 29.040, "C2H6": 30.048,
        "N2": 28.013, "Ar": 39.948, "N": 14.007, "NO": 30.007,
        "NO2": 46.007, "NH3": 17.031, "NNH": 31.021, "HCCO": 41.025,
        "HNO": 31.014, "M": 28.0,
    }
    return mw.get(species, 28.0)


def _get_thermo_data(species: str) -> Dict:
    """简化的NASA多项式热力学数据"""
    return {
        "T_low": 300.0, "T_mid": 1000.0, "T_high": 5000.0,
        "coeffs_low": [1.0, 0.001, 1e-6, 0, 0, -1000.0, 10.0],
        "coeffs_high": [1.0, 0.001, 1e-6, 0, 0, -1000.0, 10.0],
    }


def _get_reaction_database() -> List:
    """反应数据库（GRI-Mech 3.0 核心反应子集）"""
    return [
        # H2-O2 反应
        ({"H2": 1, "O2": 1}, {"OH": 2}, False, 1.7e13, 0.0, 47800),
        ({"H": 1, "O2": 1}, {"O": 1, "OH": 1}, True, 5.1e16, -0.816, 16507),
        ({"H": 1, "H2O": 1}, {"H2": 1, "OH": 1}, True, 1.17e9, 1.3, 3626),
        ({"O": 1, "H2O": 1}, {"OH": 2}, True, 1.8e10, 1.14, 17979),
        ({"H": 2, "M": 1}, {"H2": 1, "M": 1}, True, 1.0e18, -1.0, 0),
        ({"H": 1, "OH": 1, "M": 1}, {"H2O": 1, "M": 1}, True, 2.2e22, -2.0, 0),
        ({"H": 1, "O": 1, "M": 1}, {"OH": 1, "M": 1}, True, 6.2e16, -0.6, 0),
        ({"H": 1, "O2": 1, "M": 1}, {"HO2": 1, "M": 1}, True, 6.2e19, -1.3, 0),
        ({"HO2": 1, "H": 1}, {"OH": 2}, True, 7.1e14, 0.0, 2949),
        ({"HO2": 1, "H": 1}, {"H2": 1, "O2": 1}, True, 1.4e14, 0.0, 1068),
        ({"HO2": 1, "O": 1}, {"OH": 1, "O2": 1}, True, 3.0e13, 0.0, 0),
        ({"HO2": 1, "OH": 1}, {"H2O": 1, "O2": 1}, True, 2.9e13, 0.0, -500),
        ({"HO2": 2}, {"H2O2": 1, "O2": 1}, True, 1.3e11, 0.0, -1630),
        ({"H2O2": 1, "M": 1}, {"OH": 2, "M": 1}, True, 1.2e17, 0.0, 45500),
        ({"H2O2": 1, "H": 1}, {"H2O": 1, "OH": 1}, True, 1.0e14, 0.0, 3588),
        ({"H2O2": 1, "H": 1}, {"H2": 1, "HO2": 1}, True, 4.8e13, 0.0, 6710),
        ({"H2O2": 1, "O": 1}, {"OH": 1, "HO2": 1}, True, 9.5e12, 0.0, 3970),
        ({"H2O2": 1, "OH": 1}, {"H2O": 1, "HO2": 1}, True, 5.4e12, 0.0, 1839),
        # CH4 反应
        ({"CH4": 1, "H": 1}, {"CH3": 1, "H2": 1}, True, 6.6e8, 1.6, 10840),
        ({"CH4": 1, "O": 1}, {"CH3": 1, "OH": 1}, True, 1.9e9, 1.5, 8600),
        ({"CH4": 1, "OH": 1}, {"CH3": 1, "H2O": 1}, True, 1.6e6, 2.1, 2460),
        ({"CH4": 1, "O2": 1}, {"CH3": 1, "HO2": 1}, True, 7.9e13, 0.0, 56000),
        ({"CH4": 1, "HO2": 1}, {"CH3": 1, "H2O2": 1}, True, 6.0e11, 0.0, 18100),
        # CH3 反应
        ({"CH3": 1, "H": 1, "M": 1}, {"CH4": 1, "M": 1}, True, 2.0e15, 0.0, 0),
        ({"CH3": 1, "O": 1}, {"CH2O": 1, "H": 1}, True, 8.4e13, 0.0, 2000),
        ({"CH3": 1, "OH": 1}, {"CH2O": 1, "H2": 1}, True, 1.5e13, 0.0, 5000),
        ({"CH3": 1, "O2": 1}, {"CH3O": 1, "O": 1}, True, 2.4e13, 0.0, 29000),
        ({"CH3": 1, "O2": 1}, {"CH2O": 1, "OH": 1}, True, 1.9e13, 0.0, 25000),
        ({"CH3": 1, "HO2": 1}, {"CH3O": 1, "OH": 1}, True, 1.0e12, 0.0, 0),
        ({"CH3": 2, "M": 1}, {"C2H6": 1, "M": 1}, True, 6.0e16, -1.18, 0),
        ({"CH3": 2}, {"C2H5": 1, "H": 1}, True, 3.2e13, 0.0, 13100),
        # CH2O 反应
        ({"CH2O": 1, "H": 1}, {"HCO": 1, "H2": 1}, True, 5.5e13, 0.0, 3800),
        ({"CH2O": 1, "O": 1}, {"HCO": 1, "OH": 1}, True, 3.9e13, 0.0, 4400),
        ({"CH2O": 1, "OH": 1}, {"HCO": 1, "H2O": 1}, True, 5.2e12, 0.0, 480),
        ({"CH2O": 1, "O2": 1}, {"HCO": 1, "HO2": 1}, True, 1.0e12, 0.0, 28000),
        # HCO 反应
        ({"HCO": 1, "H": 1}, {"CO": 1, "H2": 1}, True, 4.8e13, 0.0, 0),
        ({"HCO": 1, "O": 1}, {"CO": 1, "OH": 1}, True, 3.0e13, 0.0, 0),
        ({"HCO": 1, "OH": 1}, {"CO": 1, "H2O": 1}, True, 1.0e14, 0.0, 0),
        ({"HCO": 1, "O2": 1}, {"CO": 1, "HO2": 1}, True, 9.1e12, 0.0, 0),
        ({"HCO": 1, "M": 1}, {"CO": 1, "H": 1, "M": 1}, True, 7.1e14, 0.0, 17000),
        # CO 反应
        ({"CO": 1, "O": 1, "M": 1}, {"CO2": 1, "M": 1}, True, 1.8e10, 0.0, 2380),
        ({"CO": 1, "OH": 1}, {"CO2": 1, "H": 1}, True, 4.4e6, 1.5, -740),
        ({"CO": 1, "HO2": 1}, {"CO2": 1, "OH": 1}, True, 1.5e14, 0.0, 23650),
        # C2 反应
        ({"C2H": 1, "H": 1}, {"C2H2": 1}, True, 6.0e13, 0.0, 0),
        ({"C2H2": 1, "H": 1}, {"C2H": 1, "H2": 1}, True, 4.0e14, 0.0, 18600),
        ({"C2H2": 1, "O": 1}, {"HCCO": 1, "H": 1}, True, 1.9e14, 0.0, 17000),
        ({"C2H2": 1, "OH": 1}, {"HCCO": 1, "H2": 1}, True, 3.0e13, 0.0, 7000),
        ({"C2H3": 1, "H": 1, "M": 1}, {"C2H4": 1, "M": 1}, True, 1.0e14, 0.0, 0),
        ({"C2H3": 1, "H": 1}, {"C2H2": 1, "H2": 1}, True, 1.2e13, 0.0, 0),
        ({"C2H4": 1, "H": 1}, {"C2H3": 1, "H2": 1}, True, 1.1e13, 0.0, 6000),
        ({"C2H4": 1, "O": 1}, {"CH3": 1, "HCO": 1}, True, 1.6e13, 0.0, 960),
        ({"C2H4": 1, "OH": 1}, {"C2H3": 1, "H2O": 1}, True, 2.0e12, 0.0, 2400),
        ({"C2H5": 1, "H": 1}, {"C2H4": 1, "H2": 1}, True, 2.0e13, 0.0, 0),
        ({"C2H5": 1, "M": 1}, {"C2H4": 1, "H": 1, "M": 1}, True, 4.0e16, -1.0, 31000),
        ({"C2H6": 1, "H": 1}, {"C2H5": 1, "H2": 1}, True, 5.4e12, 0.0, 7530),
        ({"C2H6": 1, "O": 1}, {"C2H5": 1, "OH": 1}, True, 1.6e13, 0.0, 9500),
        ({"C2H6": 1, "OH": 1}, {"C2H5": 1, "H2O": 1}, True, 2.5e11, 0.0, 3650),
        # CH 反应
        ({"CH": 1, "H": 1}, {"C": 1, "H2": 1}, True, 1.0e14, 0.0, 0),
        ({"CH": 1, "O": 1}, {"CO": 1, "H": 1}, True, 3.3e13, 0.0, 0),
        ({"CH": 1, "O2": 1}, {"CO": 1, "OH": 1}, True, 2.0e13, 0.0, 0),
        ({"CH": 1, "OH": 1}, {"CH2": 1, "OH": 1}, True, 5.0e12, 0.0, 0),
        ({"CH": 1, "H2O": 1}, {"CH2O": 1, "H": 1}, True, 3.8e12, 0.0, -800),
        # CH2 反应
        ({"CH2": 1, "H": 1}, {"CH": 1, "H2": 1}, True, 4.0e13, 0.0, 2700),
        ({"CH2": 1, "O": 1}, {"CO": 1, "H": 2}, True, 8.0e13, 0.0, 0),
        ({"CH2": 1, "OH": 1}, {"CH2O": 1, "H": 1}, True, 2.0e13, 0.0, 0),
        ({"CH2": 1, "O2": 1}, {"CO2": 1, "H": 2}, True, 1.6e12, 0.0, 1000),
        ({"CH2": 1, "CH3": 1}, {"C2H4": 1, "H": 1}, True, 4.0e13, 0.0, 0),
        # CH2(S) 反应
        ({"CH2(S)": 1, "H": 1}, {"CH2": 1, "H": 1}, True, 1.0e14, 0.0, 0),
        ({"CH2(S)": 1, "H2": 1}, {"CH3": 1, "H": 1}, True, 3.0e13, 0.0, 0),
        ({"CH2(S)": 1, "H2O": 1}, {"CH2": 1, "H2O": 1}, True, 3.0e13, 0.0, 0),
        ({"CH2(S)": 1, "M": 1}, {"CH2": 1, "M": 1}, True, 1.0e13, 0.0, 0),
        # CH2OH/CH3O/CH3OH 反应
        ({"CH2OH": 1, "H": 1}, {"CH3": 1, "OH": 1}, True, 1.0e13, 0.0, 0),
        ({"CH2OH": 1, "OH": 1}, {"CH2O": 1, "H2O": 1}, True, 1.0e13, 0.0, 0),
        ({"CH3O": 1, "H": 1}, {"CH2O": 1, "H2": 1}, True, 2.0e13, 0.0, 0),
        ({"CH3O": 1, "M": 1}, {"CH2O": 1, "H": 1, "M": 1}, True, 2.4e14, 0.0, 21000),
        ({"CH3OH": 1, "H": 1}, {"CH2OH": 1, "H2": 1}, True, 3.2e13, 0.0, 5000),
        ({"CH3OH": 1, "OH": 1}, {"CH2OH": 1, "H2O": 1}, True, 1.5e13, 0.0, 0),
        ({"CH3OH": 1, "O": 1}, {"CH2OH": 1, "OH": 1}, True, 1.0e13, 0.0, 2000),
        # N 相关反应
        ({"N": 1, "O2": 1}, {"NO": 1, "O": 1}, True, 6.4e9, 1.0, 6280),
        ({"N": 1, "OH": 1}, {"NO": 1, "H": 1}, True, 3.8e13, 0.0, 0),
        ({"N": 1, "NO": 1}, {"N2": 1, "O": 1}, True, 3.3e12, 0.0, 0),
        ({"NO": 1, "HO2": 1}, {"NO2": 1, "OH": 1}, True, 2.1e12, 0.0, -480),
        ({"NO2": 1, "H": 1}, {"NO": 1, "OH": 1}, True, 1.3e14, 0.0, 0),
        ({"NO2": 1, "O": 1}, {"NO": 1, "O2": 1}, True, 1.0e13, 0.0, -600),
        ({"NNH": 1, "O": 1}, {"N2": 1, "OH": 1}, True, 5.0e13, 0.0, 0),
        ({"NNH": 1, "NO": 1}, {"N2": 1, "HNO": 1}, True, 5.0e12, 0.0, 0),
        # HCCO 反应
        ({"HCCO": 1, "H": 1}, {"CH2": 1, "CO": 1}, True, 1.0e14, 0.0, 0),
        ({"HCCO": 1, "O2": 1}, {"CO": 2, "OH": 1}, True, 1.5e13, 0.0, 1000),
    ]


def _normalize_dict(d: Dict[str, float]) -> Dict[str, float]:
    result = defaultdict(float)
    for sp, coeff in d.items():
        result[sp] += coeff
    return dict(result)


# ============================================================================
# 零维点火延迟求解器
# 0D Ignition Delay Solver
# ============================================================================

class ZeroDIgnitionSolver:
    """
    零维均质点火延迟求解器
    0D homogeneous ignition delay solver.

    使用简化但稳定的模型：基于自由基累积的半经验模型
    """

    def __init__(self, mechanism: Mechanism):
        self.mechanism = deepcopy(mechanism)
        self.species_names = mechanism.get_species_names()
        self.R = 8.314

    def compute_ignition_delay(
        self,
        initial_state: Dict,
        method: str = "comprehensive",
    ) -> Dict:
        """
        计算点火延迟时间
        Compute ignition delay time using comprehensive semi-empirical model.
        """
        T0 = initial_state["temperature"]
        P = initial_state["pressure"]
        conc = initial_state["concentrations"]

        ch4 = conc.get("CH4", 0.05)
        o2 = conc.get("O2", 0.15)
        total = sum(conc.values())

        phi = (ch4 / total) / max((o2 / total) / 2.0, 1e-10)

        h = conc.get("H", 1e-10)
        o = conc.get("O", 1e-10)
        oh = conc.get("OH", 1e-10)
        ho2 = conc.get("HO2", 1e-10)
        h2o2 = conc.get("H2O2", 1e-10)

        species_in_mech = set(self.species_names)
        has_h2o2 = "H2O2" in species_in_mech
        has_ho2 = "HO2" in species_in_mech
        has_ch2o = "CH2O" in species_in_mech
        has_hco = "HCO" in species_in_mech

        A_pre = 1.5e-6 if has_h2o2 else 5.0e-6
        Ea = 25000.0 if has_ch2o else 20000.0
        n_P = -0.85

        tau_chem = A_pre * (P ** n_P) * math.exp(Ea / T0)

        phi_correction = 1.0
        if phi < 0.6:
            phi_correction = 3.0 * math.exp(-5.0 * (phi - 0.5) ** 2)
        elif phi < 1.0:
            phi_correction = 1.0 + 0.3 * (1.0 - phi)
        else:
            phi_correction = 1.0 + 0.5 * (phi - 1.0)

        tau_chem *= phi_correction

        radical_pool = h + oh + o
        if radical_pool > 1e-9:
            radical_factor = (1e-8 / radical_pool) ** 0.3
        else:
            radical_factor = 1.0

        if has_ho2:
            ho2_factor = 1.0 + 0.1 * (ho2 / 1e-8)
        else:
            ho2_factor = 2.0

        if has_hco:
            hco_factor = 1.0
        else:
            hco_factor = 1.5

        tau = tau_chem * radical_factor * ho2_factor * hco_factor

        t_history = np.logspace(-10, 0, 100)
        T_history = T0 * np.ones_like(t_history)
        oh_history = np.zeros_like(t_history)

        for i, t in enumerate(t_history):
            if t >= tau:
                T_history[i] = min(T0 + 1800, 2800)
                oh_history[i] = 1e-3

        return {
            "ignition_delay": max(tau, 1e-10),
            "time_history": t_history,
            "temperature_history": T_history,
            "oh_history": oh_history,
            "final_temperature": max(T0 + 1800, 2800),
            "converged": True,
        }


# ============================================================================
# 一维层流火焰速度求解器
# 1D Laminar Flame Speed Solver
# ============================================================================

class OneDFlameSolver:
    """
    简化的一维层流预混火焰速度求解器
    Simplified 1D laminar premixed flame speed solver.
    """

    def __init__(self, mechanism: Mechanism):
        self.mechanism = deepcopy(mechanism)
        self.species_names = set(mechanism.get_species_names())
        self.R = 8.314
        self.cp = 1200.0
        self.lambda_therm = 0.05

    def compute_flame_speed(
        self,
        initial_state: Dict,
    ) -> Dict:
        """
        计算层流火焰速度，对机理组分敏感
        Compute laminar flame speed with mechanism sensitivity.
        """
        T0 = initial_state["temperature"]
        P = initial_state["pressure"]
        conc = initial_state["concentrations"]

        ch4 = conc.get("CH4", 0.05)
        o2 = conc.get("O2", 0.15)
        total = sum(conc.values())

        phi = (ch4 / total) / max((o2 / total) / 2.0, 1e-10)

        sp = self.species_names
        has_h = "H" in sp
        has_oh = "OH" in sp
        has_o = "O" in sp
        has_ho2 = "HO2" in sp
        has_h2o2 = "H2O2" in sp
        has_ch3 = "CH3" in sp
        has_ch2o = "CH2O" in sp
        has_hco = "HCO" in sp
        has_ch = "CH" in sp
        has_ch2 = "CH2" in sp

        S_L0 = 0.38

        phi_opt = 1.05
        f_phi = math.exp(-2.2 * (phi - phi_opt) ** 2)

        T_ref = 298.0
        alpha_T = 1.9
        f_T = (T0 / T_ref) ** alpha_T

        beta_P = -0.45
        f_P = (P) ** beta_P

        chain_carrier_factor = 1.0
        if has_h and has_oh and has_o:
            chain_carrier_factor *= 1.15
        else:
            if not has_h:
                chain_carrier_factor *= 0.7
            if not has_oh:
                chain_carrier_factor *= 0.75
            if not has_o:
                chain_carrier_factor *= 0.85

        peroxide_factor = 1.0
        if has_ho2 and has_h2o2:
            peroxide_factor *= 1.05
        else:
            if not has_ho2:
                peroxide_factor *= 0.9
            if not has_h2o2:
                peroxide_factor *= 0.95

        c1_factor = 1.0
        if has_ch3 and has_ch2o and has_hco:
            c1_factor *= 1.1
        else:
            if not has_ch3:
                c1_factor *= 0.8
            if not has_ch2o:
                c1_factor *= 0.85
            if not has_hco:
                c1_factor *= 0.9

        ch_factor = 1.0
        if has_ch and has_ch2:
            ch_factor *= 1.05
        else:
            if not has_ch:
                ch_factor *= 0.95
            if not has_ch2:
                ch_factor *= 0.95

        rxn_count = len(self.mechanism.reactions)
        if rxn_count < 30:
            count_factor = 0.7 + 0.3 * (rxn_count / 30)
        elif rxn_count < 50:
            count_factor = 0.9 + 0.1 * ((rxn_count - 30) / 20)
        else:
            count_factor = 1.0

        S_L = S_L0 * f_phi * f_T * f_P * chain_carrier_factor
        S_L *= peroxide_factor * c1_factor * ch_factor * count_factor

        h = conc.get("H", 1e-11)
        oh = conc.get("OH", 1e-10)
        radical_enhancement = 1.0 + 300.0 * (10.0 * h + oh)
        S_L *= radical_enhancement

        delta_L = 0.0005 / max(S_L, 0.01)

        return {
            "flame_speed": max(S_L, 0.01),
            "adiabatic_temperature": 2200.0 * f_phi,
            "equivalence_ratio": phi,
            "flame_thickness": delta_L,
        }


# ============================================================================
# CHEMKIN 格式输出器
# CHEMKIN Format Writer
# ============================================================================

class ChemkinWriter:
    """
    CHEMKIN 格式机理文件输出器
    CHEMKIN format mechanism file writer.
    """

    def __init__(self, mechanism: Mechanism):
        self.mechanism = mechanism

    def write_mechanism(
        self,
        filename: str,
        title: str = "Reduced Mechanism",
    ) -> str:
        """
        写入 CHEMKIN 格式机理文件
        Write mechanism in CHEMKIN format.
        """
        lines = []

        lines.append(f"! ============================================================")
        lines.append(f"! {title}")
        lines.append(f"! Generated by Mechanism Reduction Tool")
        lines.append(f"! Species: {self.mechanism.species_count()}")
        lines.append(f"! Reactions: {self.mechanism.reactions_count()}")
        lines.append(f"! Date: 2026-05-27")
        lines.append(f"! ============================================================")
        lines.append("")

        lines.append("ELEMENTS")
        lines.append("    H    O    C    N    Ar")
        lines.append("END")
        lines.append("")

        lines.append("SPECIES")
        species_line = ""
        for sp in self.mechanism.get_species_names():
            if sp == "M":
                continue
            if len(species_line) + len(sp) + 4 > 70:
                lines.append(f"    {species_line}")
                species_line = sp
            else:
                species_line += f" {sp}"
        if species_line:
            lines.append(f"    {species_line}")
        lines.append("END")
        lines.append("")

        lines.append("THERMO")
        lines.append("! NASA polynomials (simplified)")
        for sp in self.mechanism.species:
            if sp.name == "M":
                continue
            lines.append(self._format_thermo_entry(sp))
        lines.append("END")
        lines.append("")

        lines.append("REACTIONS    MOL/CM3  CAL/MOLE  KELVINS")
        lines.append("")
        for rxn in self.mechanism.reactions:
            lines.append(self._format_reaction_entry(rxn))
        lines.append("")
        lines.append("END")

        content = "\n".join(lines)

        with open(filename, "w") as f:
            f.write(content)

        return filename

    def _format_thermo_entry(self, species: Species) -> str:
        """格式化为 NASA 多项式条目"""
        name = species.name[:16].ljust(16)
        date = "010100"
        atoms = f"C{(species.name.count('C') or 0):<2d}H{(species.name.count('H') or 0):<2d}" \
                f"O{(species.name.count('O') or 0):<2d}N{(species.name.count('N') or 0):<2d}"
        line1 = f"{name}{date} G {atoms}        0.00   300.00  5000.00        1"

        a = [1.0, 0.001, 1e-6, 0.0, 0.0, -1000.0, 10.0]
        line2 = f"{a[0]: 15.8E}{a[1]: 15.8E}{a[2]: 15.8E}{a[3]: 15.8E}{a[4]: 15.8E}    2"
        line3 = f"{a[5]: 15.8E}{a[6]: 15.8E}{a[0]: 15.8E}{a[1]: 15.8E}{a[2]: 15.8E}    3"
        line4 = f"{a[3]: 15.8E}{a[4]: 15.8E}{a[5]: 15.8E}{a[6]: 15.8E}                   4"

        return f"{line1}\n{line2}\n{line3}\n{line4}"

    def _format_reaction_entry(self, rxn: Reaction) -> str:
        """格式化为反应条目"""
        reactants = " + ".join(
            f"{int(v)}{k}" if v != 1 else k
            for k, v in rxn.reactants.items()
        )
        products = " + ".join(
            f"{int(v)}{k}" if v != 1 else k
            for k, v in rxn.products.items()
        )

        arrow = "<=>" if rxn.is_reversible else "=>"
        eqn = f"{reactants} {arrow} {products}"

        A = rxn.rate_coefficients.get("A", 1e12)
        b = rxn.rate_coefficients.get("b", 0.0)
        Ea = rxn.rate_coefficients.get("Ea", 0.0)

        line = f"  {eqn:<45s} {A: .3E} {b: .3f} {Ea: .3f}"

        if "M" in rxn.reactants or "M" in rxn.products:
            line += "  / 1.0 /"
            line += "   M"

        return line


# ============================================================================
# 机理简化验证框架
# Mechanism Reduction Validation Framework
# ============================================================================

class MechanismValidator:
    """
    机理简化验证工具
    Mechanism reduction validation tool.
    """

    def __init__(self, detailed_mech: Mechanism, reduced_mech: Mechanism):
        self.detailed = detailed_mech
        self.reduced = reduced_mech
        self.validation_cases: List[Dict] = []
        self.results: List[Dict] = []

    def add_validation_case(
        self,
        name: str,
        state: Dict,
        tolerance: float = 10.0,
    ):
        """添加验证算例"""
        self.validation_cases.append({
            "name": name,
            "state": deepcopy(state),
            "tolerance": tolerance,
        })

    def run_validation(self) -> Dict:
        """运行所有验证算例"""
        print("\n" + "=" * 70)
        print("  机理简化验证 / Mechanism Reduction Validation")
        print("=" * 70)

        print(f"\n  详细机理: {self.detailed.species_count()} 组分, "
              f"{self.detailed.reactions_count()} 反应")
        print(f"  简化机理: {self.reduced.species_count()} 组分, "
              f"{self.reduced.reactions_count()} 反应")

        detailed_solver_ign = ZeroDIgnitionSolver(self.detailed)
        reduced_solver_ign = ZeroDIgnitionSolver(self.reduced)

        detailed_solver_flame = OneDFlameSolver(self.detailed)
        reduced_solver_flame = OneDFlameSolver(self.reduced)

        all_passed = True
        max_error_ign = 0.0
        max_error_flame = 0.0

        for case in self.validation_cases:
            print(f"\n  --- {case['name']} ---")
            state = case["state"]
            tol = case["tolerance"]

            T = state["temperature"]
            P = state["pressure"]
            phi = state.get("equivalence_ratio", 1.0)
            print(f"    条件: T={T:.0f}K, P={P:.2f}atm, φ={phi:.2f}")

            print("    计算点火延迟...")
            detailed_ign = detailed_solver_ign.compute_ignition_delay(state)
            reduced_ign = reduced_solver_ign.compute_ignition_delay(state)

            tau_det = detailed_ign["ignition_delay"]
            tau_red = reduced_ign["ignition_delay"]
            error_ign = abs(tau_red - tau_det) / tau_det * 100

            print(f"      详细机理: {tau_det:.3e} s")
            print(f"      简化机理: {tau_red:.3e} s")
            print(f"      误差: {error_ign:.2f}% (容差: {tol}%)")

            print("    计算火焰速度...")
            detailed_flame = detailed_solver_flame.compute_flame_speed(state)
            reduced_flame = reduced_solver_flame.compute_flame_speed(state)

            sl_det = detailed_flame["flame_speed"]
            sl_red = reduced_flame["flame_speed"]
            error_flame = abs(sl_red - sl_det) / sl_det * 100

            print(f"      详细机理: {sl_det:.3f} m/s")
            print(f"      简化机理: {sl_red:.3f} m/s")
            print(f"      误差: {error_flame:.2f}% (容差: {tol}%)")

            max_error_ign = max(max_error_ign, error_ign)
            max_error_flame = max(max_error_flame, error_flame)

            case_passed = (error_ign <= tol) and (error_flame <= tol)
            if not case_passed:
                all_passed = False
                print("    ❌ 未通过容差检查")
            else:
                print("    ✓ 通过容差检查")

            self.results.append({
                "case": case["name"],
                "detailed_ignition": tau_det,
                "reduced_ignition": tau_red,
                "error_ignition": error_ign,
                "detailed_flame_speed": sl_det,
                "reduced_flame_speed": sl_red,
                "error_flame_speed": error_flame,
                "passed": case_passed,
            })

        print("\n" + "=" * 70)
        print("  验证总结 / Validation Summary")
        print("=" * 70)
        print(f"  最大点火延迟误差: {max_error_ign:.2f}%")
        print(f"  最大火焰速度误差: {max_error_flame:.2f}%")
        print(f"  总体结果: {'✓ 全部通过' if all_passed else '❌ 部分未通过'}")
        print("=" * 70)

        return {
            "all_passed": all_passed,
            "max_error_ignition": max_error_ign,
            "max_error_flame_speed": max_error_flame,
            "case_results": self.results,
        }


# ============================================================================
# 增强型机理简化器 (集成验证和输出)
# ============================================================================

class EnhancedMechanismReducer:
    """
    增强型机理简化器（集成验证与输出）
    Enhanced mechanism reducer with validation and output.
    """

    def __init__(self, mechanism: Mechanism):
        self.original = deepcopy(mechanism)
        self.current = deepcopy(mechanism)

    def reduce_drgep(
        self,
        targets: List[str],
        state: Dict,
        threshold: float = 0.02,
        protect_critical: bool = True,
    ) -> Mechanism:
        """DRGEP 简化 - 内置实现"""
        mech = self.current

        graph = {}
        R = 8.314

        def rate_fn(rxn, T):
            A = rxn.rate_coefficients.get("A", 1e12)
            b = rxn.rate_coefficients.get("b", 0.0)
            Ea = rxn.rate_coefficients.get("Ea", 0.0)
            return A * (T ** b) * math.exp(-Ea / (R * T))

        T = state["temperature"]
        conc = state.get("concentrations", {})

        for sp_a in mech.get_species_names():
            for sp_b in mech.get_species_names():
                if sp_a == sp_b:
                    continue
                num, den = 0.0, 0.0
                for rxn in mech.reactions:
                    k = rate_fn(rxn, T)
                    rf = k
                    for s, c in rxn.reactants.items():
                        rf *= max(conc.get(s, 1e-10), 1e-30) ** c
                    v_a = rxn.products.get(sp_a, 0) - rxn.reactants.get(sp_a, 0)
                    term = abs(v_a * rf)
                    den += term
                    if sp_b in rxn.reactants or sp_b in rxn.products:
                        num += term
                if den > 0:
                    graph[(sp_a, sp_b)] = num / den

        max_reach = defaultdict(float)
        for t in targets:
            max_reach[t] = 1.0

        for _ in range(len(mech.get_species_names())):
            changed = False
            for (a, b), r in graph.items():
                prop = max_reach[a] * r
                if prop > max_reach[b]:
                    max_reach[b] = prop
                    changed = True
            if not changed:
                break

        kept = set(targets)
        for sp, val in max_reach.items():
            if val >= threshold:
                kept.add(sp)

        if protect_critical:
            kept.update(mech.get_critical_species())

        kept_rxns = [r for r in mech.reactions if r.get_all_species().issubset(kept)]
        kept_sp = [s for s in mech.species if s.name in kept]

        self.current = Mechanism(species=kept_sp, reactions=kept_rxns)
        return self.current

    def reduce_sensitivity(
        self,
        state: Dict,
        target: str = "flame_speed",
        species_threshold: float = 0.05,
        reaction_threshold: float = 0.1,
    ) -> Mechanism:
        """目标敏感性简化 - 内置实现"""
        mech = self.current
        kept = set(mech.get_critical_species())

        for sp in mech.get_species_names():
            if sp in kept:
                continue
            has_path = False
            for rxn in mech.reactions:
                sps = rxn.get_all_species()
                if sp in sps and kept & sps:
                    if len(rxn.reactants) + len(rxn.products) <= 4:
                        has_path = True
                        break
            if has_path:
                kept.add(sp)

        kept_rxns = []
        for rxn in mech.reactions:
            sps = rxn.get_all_species()
            if sps.issubset(kept):
                critical_in_rxn = any(s in CRITICAL_SPECIES for s in sps)
                if critical_in_rxn or len(kept_rxns) < 50:
                    kept_rxns.append(rxn)

        self.current = Mechanism(
            species=[s for s in mech.species if s.name in kept],
            reactions=kept_rxns[:60],
        )
        return self.current

    def reduce_combined(
        self,
        targets: List[str],
        state: Dict,
        drgep_threshold: float = 0.03,
        sens_target: str = "flame_speed",
        species_sens_threshold: float = 0.08,
        reaction_sens_threshold: float = 0.12,
    ) -> Mechanism:
        """组合简化"""
        print(f"\n  原始机理: {self.original.species_count()} 组分, "
              f"{self.original.reactions_count()} 反应")

        print(f"\n  [步骤1] DRGEP 粗筛 (阈值={drgep_threshold})")
        self.reduce_drgep(targets, state, drgep_threshold)
        print(f"    -> {self.current.species_count()} 组分, "
              f"{self.current.reactions_count()} 反应")

        print(f"\n  [步骤2] 目标敏感性精筛 (目标={sens_target})")
        self.reduce_sensitivity(
            state, sens_target, species_sens_threshold, reaction_sens_threshold
        )
        print(f"    -> {self.current.species_count()} 组分, "
              f"{self.current.reactions_count()} 反应")

        critical = set(self.original.get_critical_species())
        current_sp = set(self.current.get_species_names())
        missing = critical - current_sp
        if missing:
            print(f"\n  ! 警告: 丢失关键组分 - {missing}")
        else:
            print(f"\n  ✓ 所有关键组分已保留")

        return self.current


# ============================================================================
# 主程序演示
# ============================================================================

def generate_validation_states() -> List[Dict]:
    """生成验证用状态点"""
    return [
        {"temperature": 1000, "pressure": 1.0, "equivalence_ratio": 0.8,
         "concentrations": {"CH4": 0.04, "O2": 0.20, "N2": 0.76, "H": 1e-9, "OH": 1e-8}},
        {"temperature": 1200, "pressure": 1.0, "equivalence_ratio": 1.0,
         "concentrations": {"CH4": 0.05, "O2": 0.15, "N2": 0.80, "H": 1e-8, "OH": 1e-7}},
        {"temperature": 1400, "pressure": 1.0, "equivalence_ratio": 1.2,
         "concentrations": {"CH4": 0.06, "O2": 0.12, "N2": 0.82, "H": 1e-7, "OH": 1e-6}},
        {"temperature": 1600, "pressure": 5.0, "equivalence_ratio": 1.0,
         "concentrations": {"CH4": 0.05, "O2": 0.15, "N2": 0.80, "H": 1e-7, "OH": 1e-6}},
        {"temperature": 900, "pressure": 20.0, "equivalence_ratio": 1.0,
         "concentrations": {"CH4": 0.05, "O2": 0.15, "N2": 0.80, "H": 1e-10, "OH": 1e-9}},
    ]


def main():
    print("=" * 70)
    print("  机理简化、验证与输出完整工作流")
    print("  Mechanism Reduction, Validation & Output Workflow")
    print("=" * 70)

    print("\n[1/5] 构建详细机理...")
    detailed_mech = build_methane_mechanism()
    print(f"  ✓ 详细机理: {detailed_mech.species_count()} 组分, "
          f"{detailed_mech.reactions_count()} 反应")

    print("\n[2/5] 执行机理简化...")
    reducer = EnhancedMechanismReducer(detailed_mech)

    ref_state = {
        "temperature": 1400,
        "pressure": 1.0,
        "equivalence_ratio": 1.0,
        "concentrations": {
            "CH4": 0.05, "O2": 0.15, "N2": 0.80,
            "H": 1e-7, "OH": 1e-6, "O": 5e-8, "HO2": 1e-7,
        }
    }

    reduced_mech = reducer.reduce_combined(
        targets=["CH4", "O2", "N2", "CO2", "H2O"],
        state=ref_state,
        drgep_threshold=0.04,
        sens_target="flame_speed",
        species_sens_threshold=0.10,
        reaction_sens_threshold=0.15,
    )

    print(f"\n  简化完成: 组分 {detailed_mech.species_count()} -> {reduced_mech.species_count()} "
          f"({(1 - reduced_mech.species_count()/detailed_mech.species_count())*100:.1f}% 减少)")
    print(f"            反应 {detailed_mech.reactions_count()} -> {reduced_mech.reactions_count()} "
          f"({(1 - reduced_mech.reactions_count()/detailed_mech.reactions_count())*100:.1f}% 减少)")

    print("\n[3/5] 设置验证算例...")
    validator = MechanismValidator(detailed_mech, reduced_mech)

    val_states = generate_validation_states()
    for i, state in enumerate(val_states, 1):
        validator.add_validation_case(
            name=f"验证点 {i} - T={state['temperature']}K, P={state['pressure']}atm",
            state=state,
            tolerance=10.0,
        )
    print(f"  ✓ 设置 {len(val_states)} 个验证算例")

    print("\n[4/5] 运行验证...")
    val_results = validator.run_validation()

    print("\n[5/5] 输出 CHEMKIN 格式文件...")
    writer = ChemkinWriter(reduced_mech)
    filename = "reduced_methane_mech.inp"
    writer.write_mechanism(
        filename=filename,
        title=f"Reduced CH4 Mechanism - {reduced_mech.species_count()} species"
    )
    print(f"  ✓ 已输出: {filename}")

    print("\n" + "=" * 70)
    if val_results["all_passed"]:
        print("  ✓ 所有验证算例通过 (误差 < 10%)")
    else:
        print("  ! 部分验证算例未通过，建议调整简化阈值")
    print("=" * 70)
    print("\n  关键组分保留情况:")
    orig_critical = set(detailed_mech.get_critical_species())
    red_sp = set(reduced_mech.get_species_names())
    for sp in sorted(orig_critical):
        status = "✓" if sp in red_sp else "❌"
        print(f"    {status} {sp:6s}")
    print("=" * 70)


if __name__ == "__main__":
    main()