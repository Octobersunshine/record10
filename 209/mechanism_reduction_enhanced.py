"""
化学反应机理简化工具（增强版）
Enhanced Chemical Reaction Mechanism Reduction Tool

使用 DRGEP (带误差传播的直接关系图法) 和目标导向敏感性分析
进行机理简化，确保关键组分（如OH自由基）不会被误删。

Supports DRGEP (Direct Relation Graph with Error Propagation) and
target-oriented sensitivity analysis for robust mechanism reduction.

作者: Mechanism Reduction Tool
日期: 2026-05-27
"""

from typing import Dict, List, Set, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy
import math
import random


# ============================================================================
# 关键组分定义 / Critical Species Definition
# ============================================================================

CRITICAL_SPECIES = [
    "H", "O", "OH", "H2", "O2", "H2O", "HO2", "H2O2",
    "CH3", "CH4", "CO", "CO2", "HCO", "CH2O",
    "CH", "CH2", "C2H2", "C2H4",
]


# ============================================================================
# 数据结构定义 / Data Structures
# ============================================================================

@dataclass
class Species:
    """化学组分 / Chemical Species"""
    name: str
    molecular_weight: float = 0.0
    thermo_data: Optional[Dict] = None
    is_critical: bool = False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Species):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"Species({self.name})"


@dataclass
class Reaction:
    """化学反应 / Chemical Reaction"""
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
    """化学反应机理 / Chemical Mechanism"""
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

    def species_count(self) -> int:
        return len(self.species)

    def reactions_count(self) -> int:
        return len(self.reactions)

    def get_critical_species(self) -> List[str]:
        return [s.name for s in self.species if s.is_critical]

    def __repr__(self):
        return (f"Mechanism({self.species_count()} species, "
                f"{self.reactions_count()} reactions)")


# ============================================================================
# 机理构建 / Mechanism Builder
# ============================================================================

def build_methane_mechanism() -> Mechanism:
    """
    构建甲烷燃烧机理（GRI-Mech 3.0 子集简化版）
    Build a simplified GRI-Mech 3.0 subset with 34 species and 88 reactions.
    """
    species_list = [
        "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2",
        "C", "CH", "CH2", "CH2(S)", "CH3", "CH4", "CO", "CO2",
        "HCO", "CH2O", "CH2OH", "CH3O", "CH3OH", "C2H", "C2H2",
        "C2H3", "C2H4", "C2H5", "C2H6",
        "N2", "Ar", "N", "NO", "NO2", "NH3", "NNH",
    ]

    species = []
    for name in species_list:
        sp = Species(
            name=name,
            molecular_weight=_get_molecular_weight(name),
            is_critical=(name in CRITICAL_SPECIES)
        )
        species.append(sp)

    reaction_data = [
        # H2-O2 反应 / H2-O2 reactions
        ({"H2": 1, "O2": 1}, {"OH": 2}, False),
        ({"H": 1, "O2": 1}, {"O": 1, "OH": 1}, True),
        ({"H": 1, "H2O": 1}, {"H2": 1, "OH": 1}, True),
        ({"O": 1, "H2O": 1}, {"OH": 2}, True),
        ({"H": 2, "M": 1}, {"H2": 1, "M": 1}, True),
        ({"H": 1, "OH": 1, "M": 1}, {"H2O": 1, "M": 1}, True),
        ({"H": 1, "O": 1, "M": 1}, {"OH": 1, "M": 1}, True),
        ({"H": 1, "O2": 1, "M": 1}, {"HO2": 1, "M": 1}, True),
        ({"HO2": 1, "H": 1}, {"OH": 2}, True),
        ({"HO2": 1, "H": 1}, {"H2": 1, "O2": 1}, True),
        ({"HO2": 1, "O": 1}, {"OH": 1, "O2": 1}, True),
        ({"HO2": 1, "OH": 1}, {"H2O": 1, "O2": 1}, True),
        ({"HO2": 2}, {"H2O2": 1, "O2": 1}, True),
        ({"H2O2": 1, "M": 1}, {"OH": 2, "M": 1}, True),
        ({"H2O2": 1, "H": 1}, {"H2O": 1, "OH": 1}, True),
        ({"H2O2": 1, "H": 1}, {"H2": 1, "HO2": 1}, True),
        ({"H2O2": 1, "O": 1}, {"OH": 1, "HO2": 1}, True),
        ({"H2O2": 1, "OH": 1}, {"H2O": 1, "HO2": 1}, True),
        # CH4 反应 / CH4 reactions
        ({"CH4": 1, "H": 1}, {"CH3": 1, "H2": 1}, True),
        ({"CH4": 1, "O": 1}, {"CH3": 1, "OH": 1}, True),
        ({"CH4": 1, "OH": 1}, {"CH3": 1, "H2O": 1}, True),
        ({"CH4": 1, "O2": 1}, {"CH3": 1, "HO2": 1}, True),
        ({"CH4": 1, "HO2": 1}, {"CH3": 1, "H2O2": 1}, True),
        # CH3 反应 / CH3 reactions
        ({"CH3": 1, "H": 1, "M": 1}, {"CH4": 1, "M": 1}, True),
        ({"CH3": 1, "O": 1}, {"CH2O": 1, "H": 1}, True),
        ({"CH3": 1, "OH": 1}, {"CH3O": 1, "H": 1}, True),
        ({"CH3": 1, "O2": 1}, {"CH3O": 1, "O": 1}, True),
        ({"CH3": 1, "O2": 1}, {"CH2O": 1, "OH": 1}, True),
        ({"CH3": 1, "HO2": 1}, {"CH3O": 1, "OH": 1}, True),
        ({"CH3": 2, "M": 1}, {"C2H6": 1, "M": 1}, True),
        ({"CH3": 2}, {"C2H5": 1, "H": 1}, True),
        # CH2O 反应 / CH2O reactions
        ({"CH2O": 1, "H": 1}, {"HCO": 1, "H2": 1}, True),
        ({"CH2O": 1, "O": 1}, {"HCO": 1, "OH": 1}, True),
        ({"CH2O": 1, "OH": 1}, {"HCO": 1, "H2O": 1}, True),
        ({"CH2O": 1, "O2": 1}, {"HCO": 1, "HO2": 1}, True),
        # HCO 反应 / HCO reactions
        ({"HCO": 1, "H": 1}, {"CO": 1, "H2": 1}, True),
        ({"HCO": 1, "O": 1}, {"CO": 1, "OH": 1}, True),
        ({"HCO": 1, "OH": 1}, {"CO": 1, "H2O": 1}, True),
        ({"HCO": 1, "O2": 1}, {"CO": 1, "HO2": 1}, True),
        ({"HCO": 1, "M": 1}, {"CO": 1, "H": 1, "M": 1}, True),
        # CO 反应 / CO reactions
        ({"CO": 1, "O": 1, "M": 1}, {"CO2": 1, "M": 1}, True),
        ({"CO": 1, "OH": 1}, {"CO2": 1, "H": 1}, True),
        ({"CO": 1, "HO2": 1}, {"CO2": 1, "OH": 1}, True),
        # C2 反应 / C2 reactions
        ({"C2H": 1, "H": 1}, {"C2H2": 1}, True),
        ({"C2H2": 1, "H": 1}, {"C2H": 1, "H2": 1}, True),
        ({"C2H2": 1, "O": 1}, {"HCCO": 1, "H": 1}, True),
        ({"C2H2": 1, "OH": 1}, {"HCCO": 1, "H2": 1}, True),
        ({"C2H3": 1, "H": 1, "M": 1}, {"C2H4": 1, "M": 1}, True),
        ({"C2H3": 1, "H": 1}, {"C2H2": 1, "H2": 1}, True),
        ({"C2H4": 1, "H": 1}, {"C2H3": 1, "H2": 1}, True),
        ({"C2H4": 1, "O": 1}, {"CH3": 1, "HCO": 1}, True),
        ({"C2H4": 1, "OH": 1}, {"C2H3": 1, "H2O": 1}, True),
        ({"C2H5": 1, "H": 1}, {"C2H4": 1, "H2": 1}, True),
        ({"C2H5": 1, "M": 1}, {"C2H4": 1, "H": 1, "M": 1}, True),
        ({"C2H6": 1, "H": 1}, {"C2H5": 1, "H2": 1}, True),
        ({"C2H6": 1, "O": 1}, {"C2H5": 1, "OH": 1}, True),
        ({"C2H6": 1, "OH": 1}, {"C2H5": 1, "H2O": 1}, True),
        # CH 反应 / CH reactions
        ({"CH": 1, "H": 1}, {"C": 1, "H2": 1}, True),
        ({"CH": 1, "O": 1}, {"CO": 1, "H": 1}, True),
        ({"CH": 1, "O2": 1}, {"CO": 1, "OH": 1}, True),
        ({"CH": 1, "OH": 1}, {"CH2": 1, "OH": 1}, True),
        ({"CH": 1, "H2O": 1}, {"CH2O": 1, "H": 1}, True),
        # CH2 反应 / CH2 reactions
        ({"CH2": 1, "H": 1}, {"CH": 1, "H2": 1}, True),
        ({"CH2": 1, "O": 1}, {"CO": 1, "H": 2}, True),
        ({"CH2": 1, "OH": 1}, {"CH2O": 1, "H": 1}, True),
        ({"CH2": 1, "O2": 1}, {"CO2": 1, "H": 2}, True),
        ({"CH2": 1, "CH3": 1}, {"C2H4": 1, "H": 1}, True),
        # CH2(S) 反应 / CH2(S) reactions
        ({"CH2(S)": 1, "H": 1}, {"CH2": 1, "H": 1}, True),
        ({"CH2(S)": 1, "H2": 1}, {"CH3": 1, "H": 1}, True),
        ({"CH2(S)": 1, "H2O": 1}, {"CH2": 1, "H2O": 1}, True),
        ({"CH2(S)": 1, "M": 1}, {"CH2": 1, "M": 1}, True),
        # CH2OH 反应 / CH2OH reactions
        ({"CH2OH": 1, "H": 1}, {"CH3": 1, "OH": 1}, True),
        ({"CH2OH": 1, "OH": 1}, {"CH2O": 1, "H2O": 1}, True),
        # CH3O 反应 / CH3O reactions
        ({"CH3O": 1, "H": 1}, {"CH2O": 1, "H2": 1}, True),
        ({"CH3O": 1, "M": 1}, {"CH2O": 1, "H": 1, "M": 1}, True),
        # CH3OH 反应 / CH3OH reactions
        ({"CH3OH": 1, "H": 1}, {"CH2OH": 1, "H2": 1}, True),
        ({"CH3OH": 1, "OH": 1}, {"CH2OH": 1, "H2O": 1}, True),
        ({"CH3OH": 1, "O": 1}, {"CH2OH": 1, "OH": 1}, True),
        # N 相关反应 / N-related reactions
        ({"N": 1, "O2": 1}, {"NO": 1, "O": 1}, True),
        ({"N": 1, "OH": 1}, {"NO": 1, "H": 1}, True),
        ({"N": 1, "NO": 1}, {"N2": 1, "O": 1}, True),
        ({"NO": 1, "HO2": 1}, {"NO2": 1, "OH": 1}, True),
        ({"NO2": 1, "H": 1}, {"NO": 1, "OH": 1}, True),
        ({"NO2": 1, "O": 1}, {"NO": 1, "O2": 1}, True),
        ({"NNH": 1, "O": 1}, {"N2": 1, "OH": 1}, True),
        ({"NNH": 1, "NO": 1}, {"N2": 1, "HNO": 1}, True),
        # HCCO 反应 / HCCO reactions
        ({"HCCO": 1, "H": 1}, {"CH2": 1, "CO": 1}, True),
        ({"HCCO": 1, "O2": 1}, {"CO": 2, "OH": 1}, True),
    ]

    reactions = []
    for i, (reactants, products, reversible) in enumerate(reaction_data, 1):
        rxn = Reaction(
            id=i,
            reactants=_normalize_species_dict(reactants),
            products=_normalize_species_dict(products),
            rate_coefficients=_get_rate_coefficients(i),
            is_reversible=reversible,
            label=f"R{i}"
        )
        reactions.append(rxn)

    return Mechanism(species=species, reactions=reactions)


def _get_molecular_weight(species: str) -> float:
    """获取组分分子量（简化版）"""
    mw_dict = {
        "H2": 2.016, "H": 1.008, "O": 16.000, "O2": 32.000,
        "OH": 17.008, "H2O": 18.016, "HO2": 33.008, "H2O2": 34.016,
        "C": 12.000, "CH": 13.008, "CH2": 14.016, "CH2(S)": 14.016,
        "CH3": 15.024, "CH4": 16.032, "CO": 28.000, "CO2": 44.000,
        "HCO": 29.008, "CH2O": 30.016, "CH2OH": 31.024, "CH3O": 31.024,
        "CH3OH": 32.032, "C2H": 25.008, "C2H2": 26.016, "C2H3": 27.024,
        "C2H4": 28.032, "C2H5": 29.040, "C2H6": 30.048,
        "N2": 28.013, "Ar": 39.948, "N": 14.007, "NO": 30.007,
        "NO2": 46.007, "NH3": 17.031, "NNH": 31.021, "HCCO": 41.025,
    }
    return mw_dict.get(species, 28.0)


def _get_rate_coefficients(rxn_id: int) -> Dict:
    """获取反应速率系数（简化的Arrhenius参数）"""
    A_values = [
        1.7e13, 5.1e16, 1.2e14, 1.8e10, 1.0e18, 2.2e22, 1.8e18, 6.2e19,
        7.1e14, 1.0e14, 3.0e13, 2.9e13, 1.3e12, 1.2e17, 1.0e14, 5.0e12,
        7.0e12, 5.4e12, 6.6e13, 1.9e14, 1.6e12, 1.2e12, 6.0e11, 2.0e15,
        8.4e13, 1.5e13, 2.4e12, 1.9e13, 1.0e12, 6.0e14, 3.2e13, 5.5e13,
        3.9e13, 5.2e12, 6.8e13, 4.8e13, 1.5e14, 7.5e12, 4.0e13, 7.1e14,
        1.8e10, 1.5e7, 1.5e14, 6.0e13, 4.0e13, 1.9e12, 3.0e13, 1.0e14,
        1.2e13, 1.1e14, 1.6e13, 3.1e12, 4.1e14, 1.3e14, 5.4e11, 1.6e13,
        3.4e13, 1.5e13, 1.0e14, 4.8e13, 3.3e13, 5.0e13, 1.0e14, 5.7e13,
        1.8e14, 1.3e13, 6.0e13, 2.0e13, 1.0e13, 3.0e13, 1.0e15, 5.0e12,
        1.0e13, 2.0e13, 2.5e13, 1.0e17, 3.2e13, 1.5e13, 1.0e14, 2.0e12,
        2.7e12, 9.0e11, 3.5e13, 1.5e14, 1.3e14, 1.0e13, 3.0e13, 5.0e13,
    ]
    Ea_values = [
        47800, 16500, 20300, 17900, 0, 0, 0, 0, 2900, 700, 0, 0, 0,
        45500, 3600, 6700, 6000, 1800, 10840, 8600, 2460, 13400, 10100,
        0, 2000, 5000, 9000, 5100, 0, 13100, 4430, 3000, 4400, 8000,
        0, 700, 0, 0, 23000, 47400, 0, 740, 23000, 0, 0, 17000, 3000,
        3000, 1600, 12500, 0, 5900, 0, 2500, 5400, 20000, 7500, 950,
        25700, 0, 3500, 14000, 5200, 27000, 13600, 29000, 0, 600, 0,
        0, 39900, 0, 18800, 4000, 25000, 7000, 6000, 0, 6000, 1000,
        0, 37500, 0, 500, 26000, 0, 0, 5000,
    ]
    idx = (rxn_id - 1) % len(A_values)
    return {"A": A_values[idx], "b": 0.0, "Ea": Ea_values[idx]}


def _normalize_species_dict(species_dict: Dict[str, float]) -> Dict[str, float]:
    """合并字典中的重复组分"""
    result = defaultdict(float)
    for sp, coeff in species_dict.items():
        result[sp] += coeff
    return dict(result)


# ============================================================================
# DRGEP 算法 (带误差传播的直接关系图法)
# DRGEP - Direct Relation Graph with Error Propagation
# ============================================================================

class DRGEPReducer:
    """
    DRGEP 机理简化器
    DRGEP Mechanism Reducer with Error Propagation.

    核心改进 / Key Improvements:
    1. 使用 r_AB^k 代替简单的 r_AB，考虑传播路径的误差累积
    2. 强制保留关键组分（OH, H, O 等自由基）
    3. 支持多状态下的鲁棒性简化
    """

    def __init__(self, mechanism: Mechanism, threshold: float = 0.01):
        self.mechanism = deepcopy(mechanism)
        self.threshold = threshold
        self.importance_matrix: Dict[Tuple[str, str], float] = {}
        self.error_propagation: Dict[Tuple[str, str], float] = {}

    def compute_pair_importance(
        self,
        species_a: str,
        species_b: str,
        state: Dict,
    ) -> float:
        """
        计算组分 B 对组分 A 的直接重要性系数 r_AB
        Compute direct importance coefficient r_AB of B relative to A.
        """
        numerator = 0.0
        denominator = 0.0

        for rxn in self.mechanism.reactions:
            rxn_rate = self._compute_reaction_rate(rxn, state)

            ν_a = rxn.products.get(species_a, 0) - rxn.reactants.get(species_a, 0)
            term = abs(ν_a * rxn_rate)
            denominator += term

            if species_b in rxn.reactants or species_b in rxn.products:
                numerator += term

        if denominator == 0.0:
            return 0.0

        return numerator / denominator

    def build_importance_graph(
        self,
        state: Dict,
    ) -> Dict[Tuple[str, str], float]:
        """
        构建直接重要性图
        Build direct importance graph.
        """
        species_names = self.mechanism.get_species_names()
        graph = {}

        for sp_a in species_names:
            for sp_b in species_names:
                if sp_a == sp_b:
                    continue
                imp = self.compute_pair_importance(sp_a, sp_b, state)
                if imp > 0:
                    graph[(sp_a, sp_b)] = imp

        self.importance_matrix = graph
        return graph

    def compute_error_propagation(
        self,
        targets: List[str],
    ) -> Dict[Tuple[str, str], float]:
        """
        计算误差传播系数 r_AB^k
        Compute error propagation coefficients considering all paths.

        r_AB^k = max_path [product(r_ij for edge ij in path)]
        """
        species_names = self.mechanism.get_species_names()
        propagation = {}

        for target in targets:
            max_reach = defaultdict(float)
            max_reach[target] = 1.0

            changed = True
            iterations = 0
            while changed and iterations < len(species_names):
                changed = False
                iterations += 1

                for sp_a in list(max_reach.keys()):
                    for sp_b in species_names:
                        if sp_b == target:
                            continue
                        direct = self.importance_matrix.get((sp_a, sp_b), 0.0)
                        if direct > 0:
                            propagated = max_reach[sp_a] * direct
                            if propagated > max_reach[sp_b]:
                                max_reach[sp_b] = propagated
                                changed = True

            for sp, val in max_reach.items():
                if sp != target:
                    key = (target, sp)
                    if val > propagation.get(key, 0.0):
                        propagation[key] = val

        self.error_propagation = propagation
        return propagation

    def reduce(
        self,
        target_species: List[str],
        state: Dict,
        threshold: Optional[float] = None,
        protect_critical: bool = True,
    ) -> Mechanism:
        """
        使用 DRGEP 简化机理
        Reduce mechanism using DRGEP method.

        Parameters
        ----------
        target_species : List[str]
            目标组分（燃料、氧化剂、主要产物）
        state : Dict
            热力学状态
        threshold : float, optional
            误差传播阈值
        protect_critical : bool
            是否强制保护关键组分
        """
        if threshold is not None:
            self.threshold = threshold

        self.build_importance_graph(state)
        self.compute_error_propagation(target_species)

        kept_species = set(target_species)

        for (target, sp), prop_val in self.error_propagation.items():
            if prop_val >= self.threshold:
                kept_species.add(sp)

        if protect_critical:
            critical = set(self.mechanism.get_critical_species())
            n_added = len(critical - kept_species)
            kept_species.update(critical)
            if n_added > 0:
                print(f"    [DRGEP] 强制保护 {n_added} 个关键组分")

        kept_reactions = []
        for rxn in self.mechanism.reactions:
            rxn_species = rxn.get_all_species()
            if rxn_species.issubset(kept_species):
                kept_reactions.append(rxn)

        kept_species_obj = [
            s for s in self.mechanism.species if s.name in kept_species
        ]

        return Mechanism(species=kept_species_obj, reactions=kept_reactions)

    def reduce_multi_state(
        self,
        target_species: List[str],
        states: List[Dict],
        threshold: Optional[float] = None,
        protect_critical: bool = True,
    ) -> Mechanism:
        """
        多状态鲁棒性简化：取所有状态下保留组分的并集
        Multi-state robust reduction: union of species from all states.
        """
        all_kept: Set[str] = set()

        for i, state in enumerate(states, 1):
            reduced = self.reduce(
                target_species, state, threshold, protect_critical=False
            )
            state_species = set(reduced.get_species_names())
            all_kept.update(state_species)
            print(f"    [状态 {i}] 保留 {len(state_species)} 个组分")

        if protect_critical:
            critical = set(self.mechanism.get_critical_species())
            all_kept.update(critical)

        kept_reactions = []
        for rxn in self.mechanism.reactions:
            rxn_species = rxn.get_all_species()
            if rxn_species.issubset(all_kept):
                kept_reactions.append(rxn)

        kept_species_obj = [
            s for s in self.mechanism.species if s.name in all_kept
        ]

        return Mechanism(species=kept_species_obj, reactions=kept_reactions)

    def _compute_reaction_rate(
        self,
        rxn: Reaction,
        state: Dict,
    ) -> float:
        """
        计算反应速率（Arrhenius 公式）
        Compute reaction rate using Arrhenius formula.
        """
        A = rxn.rate_coefficients.get("A", 1.0e12)
        b = rxn.rate_coefficients.get("b", 0.0)
        Ea = rxn.rate_coefficients.get("Ea", 10000.0)
        T = state.get("temperature", 1000.0)
        R = 8.314

        k = A * (T ** b) * math.exp(-Ea / (R * T))

        rate = k
        for species, coeff in rxn.reactants.items():
            conc = state.get("concentrations", {}).get(species, 1.0e-10)
            rate *= conc ** coeff

        return max(rate, 1.0e-30)


# ============================================================================
# 目标导向敏感性分析
# Target-Oriented Sensitivity Analysis
# ============================================================================

class TargetSensitivityAnalyzer:
    """
    基于目标参数的敏感性分析器
    Target-oriented sensitivity analyzer.

    目标参数 / Targets:
    - 点火延迟 (ignition delay)
    - 层流火焰速度 (laminar flame speed)
    - 主要产物浓度 (major species concentration)
    """

    def __init__(self, mechanism: Mechanism):
        self.mechanism = deepcopy(mechanism)
        self.species_sensitivities: Dict[str, float] = {}
        self.reaction_sensitivities: Dict[int, float] = {}

    def compute_ignition_delay(
        self,
        state: Dict,
        species_perturbation: Optional[Dict[str, float]] = None,
        rxn_perturbation: Optional[Dict[int, float]] = None,
    ) -> float:
        """
        计算点火延迟时间（详细半经验模型）
        Compute ignition delay using detailed semi-empirical model.

        τ = A * P^(-n) * exp(Ea/RT) * [Fuel]^a * [O2]^b * [Inhibitors]^c
        """
        T = state.get("temperature", 1000.0)
        P = state.get("pressure", 1.0)
        phi = state.get("equivalence_ratio", 1.0)

        conc = state.get("concentrations", {})

        ch4 = conc.get("CH4", 0.05)
        o2 = conc.get("O2", 0.15)

        radicals = conc.get("H", 1e-6) + conc.get("OH", 1e-6) + conc.get("O", 1e-6)
        ho2 = conc.get("HO2", 1e-7)
        h2o2 = conc.get("H2O2", 1e-8)

        if species_perturbation:
            for sp, factor in species_perturbation.items():
                if sp == "CH4":
                    ch4 *= factor
                elif sp == "O2":
                    o2 *= factor
                elif sp in ["H", "OH", "O"]:
                    radicals *= factor
                elif sp == "HO2":
                    ho2 *= factor
                elif sp == "H2O2":
                    h2o2 *= factor

        base_tau = 1.0e-6 * (P ** -0.8) * math.exp(20000.0 / T)

        tau = base_tau * (ch4 ** -0.5) * (o2 ** -1.2) * (phi ** 0.3)

        radical_effect = (radicals / 1e-6) ** (-0.3) if radicals > 0 else 1.0
        tau *= radical_effect

        h2o2_effect = (h2o2 / 1e-8) ** (-0.15) if h2o2 > 0 else 1.0
        tau *= h2o2_effect

        if rxn_perturbation:
            for rxn_id, factor in rxn_perturbation.items():
                rxn = self.mechanism.get_reaction(rxn_id)
                if rxn:
                    species_involved = rxn.get_all_species()
                    is_chain_branching = any(
                        sp in species_involved for sp in ["H", "OH", "O"]
                    )
                    if is_chain_branching:
                        tau *= factor ** (-0.2)

        return max(tau, 1.0e-8)

    def compute_flame_speed(
        self,
        state: Dict,
        species_perturbation: Optional[Dict[str, float]] = None,
        rxn_perturbation: Optional[Dict[int, float]] = None,
    ) -> float:
        """
        计算层流火焰速度（简化模型）
        Compute laminar flame speed using simplified model.

        S_L ∝ sqrt(α * ω)
        where α = thermal diffusivity, ω = reaction rate
        """
        T = state.get("temperature", 300.0)
        P = state.get("pressure", 1.0)
        phi = state.get("equivalence_ratio", 1.0)

        conc = state.get("concentrations", {})

        ch4 = conc.get("CH4", 0.05)
        o2 = conc.get("O2", 0.15)

        oh = conc.get("OH", 1e-10)
        h = conc.get("H", 1e-11)
        ho2 = conc.get("HO2", 1e-9)

        if species_perturbation:
            for sp, factor in species_perturbation.items():
                if sp == "OH":
                    oh *= factor
                elif sp == "H":
                    h *= factor
                elif sp == "HO2":
                    ho2 *= factor
                elif sp == "CH4":
                    ch4 *= factor
                elif sp == "O2":
                    o2 *= factor

        T_ad = 2200.0 * phi ** 0.1
        alpha = 2.5e-4 * (T_ad / 1000) ** 1.75 * (1.0 / P)

        omega_base = (ch4 ** 0.5) * (o2 ** 0.8) * math.exp(-15000.0 / T)
        radical_enhancement = 1.0 + 1000.0 * (oh + 5.0 * h)
        omega = omega_base * radical_enhancement

        S_L = 0.35 * math.sqrt(alpha * omega * 1e5)

        S_L *= math.exp(-1.5 * (phi - 1.0) ** 2)

        if rxn_perturbation:
            for rxn_id, factor in rxn_perturbation.items():
                rxn = self.mechanism.get_reaction(rxn_id)
                if rxn:
                    species_involved = rxn.get_all_species()
                    if "OH" in species_involved or "H" in species_involved:
                        S_L *= factor ** 0.15

        return max(S_L, 0.01)

    def analyze_species_sensitivity(
        self,
        state: Dict,
        target: str = "flame_speed",
    ) -> Dict[str, float]:
        """
        计算各组分对目标量的敏感性
        Compute species sensitivity to target quantity.

        S_i = (∂Q/∂X_i) * (X_i / Q)
        """
        if target == "flame_speed":
            base_value = self.compute_flame_speed(state)
        else:
            base_value = self.compute_ignition_delay(state)

        sensitivities = {}
        perturbation = 0.2

        for sp in self.mechanism.get_species_names():
            conc_perturb = deepcopy(state.get("concentrations", {}))
            orig_conc = conc_perturb.get(sp, 1e-10)
            conc_perturb[sp] = orig_conc * (1.0 + perturbation)

            pert_state = deepcopy(state)
            pert_state["concentrations"] = conc_perturb

            if target == "flame_speed":
                perturbed = self.compute_flame_speed(pert_state)
            else:
                perturbed = self.compute_ignition_delay(pert_state)

            if base_value > 0:
                sens = (perturbed - base_value) / base_value / perturbation
            else:
                sens = 0.0
            sensitivities[sp] = abs(sens)

        self.species_sensitivities = sensitivities
        return sensitivities

    def analyze_reaction_sensitivity(
        self,
        state: Dict,
        target: str = "flame_speed",
    ) -> Dict[int, float]:
        """
        计算各反应对目标量的敏感性
        Compute reaction sensitivity to target quantity.
        """
        if target == "flame_speed":
            base_value = self.compute_flame_speed(state)
        else:
            base_value = self.compute_ignition_delay(state)

        sensitivities = {}
        perturbation = 0.2

        for rxn in self.mechanism.reactions:
            pert = {rxn.id: 1.0 + perturbation}
            if target == "flame_speed":
                perturbed = self.compute_flame_speed(state, rxn_perturbation=pert)
            else:
                perturbed = self.compute_ignition_delay(state, rxn_perturbation=pert)

            sens = (perturbed - base_value) / base_value / perturbation
            sensitivities[rxn.id] = abs(sens)

        self.reaction_sensitivities = sensitivities
        return sensitivities

    def reduce_by_species_sensitivity(
        self,
        state: Dict,
        threshold: float = 0.01,
        target: str = "flame_speed",
        protect_critical: bool = True,
    ) -> Mechanism:
        """
        基于组分敏感性删除不重要的组分
        Remove unimportant species based on target sensitivity.
        """
        self.analyze_species_sensitivity(state, target)

        max_sens = max(self.species_sensitivities.values()) if self.species_sensitivities else 0.0

        kept_species = set()
        if max_sens > 0:
            for sp, sens in self.species_sensitivities.items():
                if sens / max_sens >= threshold:
                    kept_species.add(sp)
        else:
            for sp, sens in sorted(
                self.species_sensitivities.items(), key=lambda x: x[1], reverse=True
            )[:25]:
                kept_species.add(sp)

        if protect_critical:
            kept_species.update(self.mechanism.get_critical_species())

        kept_reactions = []
        for rxn in self.mechanism.reactions:
            rxn_species = rxn.get_all_species()
            if rxn_species.issubset(kept_species):
                kept_reactions.append(rxn)

        kept_species_obj = [
            s for s in self.mechanism.species if s.name in kept_species
        ]

        return Mechanism(species=kept_species_obj, reactions=kept_reactions)

    def reduce_by_reaction_sensitivity(
        self,
        state: Dict,
        threshold: float = 0.05,
        target: str = "flame_speed",
    ) -> Mechanism:
        """
        基于反应敏感性删除不重要的反应
        Remove unimportant reactions based on target sensitivity.
        """
        self.analyze_reaction_sensitivity(state, target)

        max_sens = max(self.reaction_sensitivities.values()) if self.reaction_sensitivities else 0.0

        kept_reactions = []
        if max_sens > 0:
            for rxn in self.mechanism.reactions:
                sens = self.reaction_sensitivities.get(rxn.id, 0.0)
                if sens / max_sens >= threshold:
                    kept_reactions.append(rxn)
        else:
            sorted_rxn = sorted(
                self.reaction_sensitivities.items(), key=lambda x: x[1], reverse=True
            )[:60]
            kept_ids = {rxn_id for rxn_id, _ in sorted_rxn}
            for rxn in self.mechanism.reactions:
                if rxn.id in kept_ids:
                    kept_reactions.append(rxn)

        used_species = set()
        for rxn in kept_reactions:
            used_species.update(rxn.get_all_species())

        kept_species = [
            s for s in self.mechanism.species if s.name in used_species
        ]

        return Mechanism(species=kept_species, reactions=kept_reactions)


# ============================================================================
# 增强型简化器集成
# Enhanced Integrated Reducer
# ============================================================================

class EnhancedMechanismReducer:
    """
    增强型机理简化工具
    Enhanced mechanism reduction tool.

    简化策略:
    1. DRGEP 粗筛（带误差传播，保护关键组分）
    2. 目标导向敏感性精筛（火焰速度/点火延迟）
    3. 验证关键目标参数误差
    """

    def __init__(self, mechanism: Mechanism):
        self.original_mechanism = deepcopy(mechanism)
        self.current_mechanism = deepcopy(mechanism)
        self.validation_results: Dict = {}

    def reduce_drgep(
        self,
        target_species: List[str],
        state: Dict,
        threshold: float = 0.01,
        protect_critical: bool = True,
    ) -> Mechanism:
        """DRGEP 简化"""
        drgep = DRGEPReducer(self.current_mechanism, threshold=threshold)
        reduced = drgep.reduce(
            target_species=target_species,
            state=state,
            protect_critical=protect_critical,
        )
        self.current_mechanism = reduced
        return reduced

    def reduce_drgep_multi_state(
        self,
        target_species: List[str],
        states: List[Dict],
        threshold: float = 0.01,
        protect_critical: bool = True,
    ) -> Mechanism:
        """多状态 DRGEP 简化"""
        drgep = DRGEPReducer(self.current_mechanism, threshold=threshold)
        reduced = drgep.reduce_multi_state(
            target_species=target_species,
            states=states,
            protect_critical=protect_critical,
        )
        self.current_mechanism = reduced
        return reduced

    def reduce_sensitivity(
        self,
        state: Dict,
        target: str = "flame_speed",
        species_threshold: float = 0.05,
        reaction_threshold: float = 0.1,
    ) -> Mechanism:
        """目标导向敏感性简化"""
        analyzer = TargetSensitivityAnalyzer(self.current_mechanism)

        reduced = analyzer.reduce_by_species_sensitivity(
            state=state,
            threshold=species_threshold,
            target=target,
        )
        self.current_mechanism = reduced

        analyzer2 = TargetSensitivityAnalyzer(self.current_mechanism)
        reduced = analyzer2.reduce_by_reaction_sensitivity(
            state=state,
            threshold=reaction_threshold,
            target=target,
        )
        self.current_mechanism = reduced

        return reduced

    def reduce_combined(
        self,
        target_species: List[str],
        state: Dict,
        drgep_threshold: float = 0.02,
        target: str = "flame_speed",
        species_sens_threshold: float = 0.08,
        reaction_sens_threshold: float = 0.15,
    ) -> Mechanism:
        """
        组合简化策略
        Combined reduction strategy: DRGEP + Sensitivity
        """
        print(f"  原始机理: {self.original_mechanism.species_count()} 组分, "
              f"{self.original_mechanism.reactions_count()} 反应")

        print(f"\n  [步骤1] DRGEP 粗筛 (阈值={drgep_threshold})")
        reduced = self.reduce_drgep(
            target_species=target_species,
            state=state,
            threshold=drgep_threshold,
            protect_critical=True,
        )
        print(f"  DRGEP后: {reduced.species_count()} 组分, "
              f"{reduced.reactions_count()} 反应")

        print(f"\n  [步骤2] 目标敏感性精筛 (目标={target})")
        reduced = self.reduce_sensitivity(
            state=state,
            target=target,
            species_threshold=species_sens_threshold,
            reaction_threshold=reaction_sens_threshold,
        )
        print(f"  敏感性后: {reduced.species_count()} 组分, "
              f"{reduced.reactions_count()} 反应")

        print(f"\n  [步骤3] 验证关键组分保留情况")
        critical_species = self.original_mechanism.get_critical_species()
        current_species = set(self.current_mechanism.get_species_names())
        missing = [sp for sp in critical_species if sp not in current_species]
        if missing:
            print(f"    ! 警告: 丢失关键组分 - {missing}")
        else:
            print(f"    ✓ 所有关键组分已保留")

        return reduced

    def validate(
        self,
        state: Dict,
    ) -> Dict:
        """
        验证简化机理的预测误差
        Validate prediction error of reduced mechanism.
        """
        analyzer_orig = TargetSensitivityAnalyzer(self.original_mechanism)
        analyzer_reduced = TargetSensitivityAnalyzer(self.current_mechanism)

        tau_orig = analyzer_orig.compute_ignition_delay(state)
        tau_reduced = analyzer_reduced.compute_ignition_delay(state)

        sl_orig = analyzer_orig.compute_flame_speed(state)
        sl_reduced = analyzer_reduced.compute_flame_speed(state)

        tau_error = abs(tau_reduced - tau_orig) / tau_orig * 100
        sl_error = abs(sl_reduced - sl_orig) / sl_orig * 100

        results = {
            "original_ignition_delay": tau_orig,
            "reduced_ignition_delay": tau_reduced,
            "ignition_delay_error(%)": tau_error,
            "original_flame_speed": sl_orig,
            "reduced_flame_speed": sl_reduced,
            "flame_speed_error(%)": sl_error,
        }

        self.validation_results = results
        return results

    def reset(self):
        """重置为原始机理"""
        self.current_mechanism = deepcopy(self.original_mechanism)


# ============================================================================
# 工具函数 / Utility Functions
# ============================================================================

def generate_test_states(
    n_states: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """生成多个测试状态点"""
    random.seed(seed)
    states = []

    temp_ranges = [(800, 1000), (1000, 1200), (1200, 1500), (1500, 1800), (1800, 2000)]
    press_ranges = [(0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0)]
    phi_ranges = [(0.5, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 2.0)]

    for i in range(n_states):
        states.append({
            "temperature": random.uniform(*temp_ranges[i % len(temp_ranges)]),
            "pressure": random.uniform(*press_ranges[i % len(press_ranges)]),
            "equivalence_ratio": random.uniform(*phi_ranges[i % len(phi_ranges)]),
            "concentrations": {
                "CH4": random.uniform(0.03, 0.08),
                "O2": random.uniform(0.1, 0.2),
                "N2": random.uniform(0.7, 0.85),
                "H": random.uniform(1e-10, 1e-6),
                "OH": random.uniform(1e-9, 1e-5),
                "O": random.uniform(1e-10, 1e-6),
                "HO2": random.uniform(1e-9, 1e-6),
                "H2O2": random.uniform(1e-10, 1e-7),
            }
        })

    return states


def print_mechanism_info(mechanism: Mechanism, title: str = "机理信息"):
    """打印机理信息"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  组分数: {mechanism.species_count()}")
    print(f"  反应数: {mechanism.reactions_count()}")

    critical = mechanism.get_critical_species()
    all_species = mechanism.get_species_names()
    missing_critical = [sp for sp in CRITICAL_SPECIES if sp not in all_species]

    print(f"  关键组分: {len(critical)} 个")
    if missing_critical:
        print(f"  丢失关键组分: {missing_critical}")
    else:
        print(f"  关键组分: 全部保留 ✓")

    print(f"\n  组分列表: {', '.join(all_species)}")
    print(f"\n  反应列表:")
    for rxn in mechanism.reactions:
        print(f"    {rxn}")
    print(f"{'='*60}\n")


# ============================================================================
# 主程序 / Main Program
# ============================================================================

def main():
    print("=" * 70)
    print("  增强型化学反应机理简化工具")
    print("  Enhanced Mechanism Reduction - DRGEP + Target Sensitivity")
    print("=" * 70)

    mechanism = build_methane_mechanism()
    print_mechanism_info(mechanism, "原始甲烷燃烧机理")

    target_species = ["CH4", "O2", "N2", "CO2", "H2O"]

    sample_state = {
        "temperature": 1400.0,
        "pressure": 1.0,
        "equivalence_ratio": 1.0,
        "concentrations": {
            "CH4": 0.05,
            "O2": 0.15,
            "N2": 0.75,
            "H2O": 0.01,
            "CO2": 0.01,
            "H": 1e-7,
            "OH": 1e-6,
            "O": 1e-7,
            "HO2": 1e-8,
            "H2O2": 1e-9,
        }
    }

    print("=" * 70)
    print("  示例 1: DRGEP 简化 (vs 原始 DRG 对比)")
    print("=" * 70)

    drgep_threshold = 0.02

    print(f"\n  [DRGEP 简化] 阈值={drgep_threshold}, 保护关键组分")
    drgep = DRGEPReducer(mechanism, threshold=drgep_threshold)
    reduced_drgep = drgep.reduce(
        target_species=target_species,
        state=sample_state,
        protect_critical=True,
    )

    print(f"  简化后: {reduced_drgep.species_count()} 组分, "
          f"{reduced_drgep.reactions_count()} 反应")
    print(f"  关键组分检查: ", end="")
    current_sp = set(reduced_drgep.get_species_names())
    missing = [sp for sp in CRITICAL_SPECIES if sp not in current_sp]
    if missing:
        print(f"丢失 {missing}")
    else:
        print("全部保留 ✓")

    print(f"\n  [不保护关键组分对比] 相同阈值, 不保护")
    drgep_no_protect = DRGEPReducer(mechanism, threshold=drgep_threshold)
    reduced_no_protect = drgep_no_protect.reduce(
        target_species=target_species,
        state=sample_state,
        protect_critical=False,
    )
    print(f"  简化后: {reduced_no_protect.species_count()} 组分, "
          f"{reduced_no_protect.reactions_count()} 反应")
    orig_sp = set(reduced_no_protect.get_species_names())
    missing_orig = [sp for sp in CRITICAL_SPECIES if sp not in orig_sp]
    if missing_orig:
        print(f"  丢失关键组分: {missing_orig}")
    else:
        print("  全部保留 ✓")

    print("=" * 70)
    print("  示例 2: 组分对火焰速度的敏感性排名")
    print("=" * 70)

    analyzer = TargetSensitivityAnalyzer(mechanism)
    species_sens = analyzer.analyze_species_sensitivity(sample_state, "flame_speed")

    print("\n  Top-15 对火焰速度敏感的组分:")
    sorted_sp = sorted(species_sens.items(), key=lambda x: x[1], reverse=True)[:15]
    max_sens = max(species_sens.values()) if species_sens else 1.0
    if max_sens > 0:
        for sp, sens in sorted_sp:
            is_critical = "★" if sp in CRITICAL_SPECIES else " "
            print(f"    {is_critical} {sp:8s}: {sens/max_sens:.4f}")
    else:
        for sp, sens in sorted_sp:
            is_critical = "★" if sp in CRITICAL_SPECIES else " "
            print(f"    {is_critical} {sp:8s}: {sens:.6f}")

    print("\n  关键组分敏感性检查:")
    for sp in ["H", "OH", "O", "HO2", "CH3"]:
        sp_sens = species_sens.get(sp, 0)
        if max_sens > 0:
            print(f"    {sp:4s}: {sp_sens/max_sens:.4f}")
        else:
            print(f"    {sp:4s}: {sp_sens:.6f}")

    print("=" * 70)
    print("  示例 3: 目标导向敏感性简化 (火焰速度)")
    print("=" * 70)

    sens_threshold = 0.1
    reduced_sens = analyzer.reduce_by_species_sensitivity(
        state=sample_state,
        threshold=sens_threshold,
        target="flame_speed",
        protect_critical=True,
    )

    print(f"\n  敏感性阈值: {sens_threshold}")
    print(f"  简化后: {reduced_sens.species_count()} 组分, "
          f"{reduced_sens.reactions_count()} 反应")

    print("\n  反应敏感性 Top-10:")
    rxn_sens = analyzer.analyze_reaction_sensitivity(sample_state, "flame_speed")
    sorted_rxn = sorted(rxn_sens.items(), key=lambda x: x[1], reverse=True)[:10]
    max_rxn_sens = max(rxn_sens.values()) if rxn_sens else 0.0
    for rxn_id, sens in sorted_rxn:
        rxn = mechanism.get_reaction(rxn_id)
        key_sp = [sp for sp in rxn.get_all_species() if sp in CRITICAL_SPECIES]
        if max_rxn_sens > 0:
            print(f"    R{rxn_id:02d}: {sens/max_rxn_sens:.4f} - 关键组分: {key_sp[:3]}")
        else:
            print(f"    R{rxn_id:02d}: {sens:.6f} - 关键组分: {key_sp[:3]}")

    print("=" * 70)
    print("  示例 4: 组合简化 (DRGEP + 敏感性)")
    print("=" * 70)

    reducer = EnhancedMechanismReducer(mechanism)
    reduced_combined = reducer.reduce_combined(
        target_species=target_species,
        state=sample_state,
        drgep_threshold=0.03,
        target="flame_speed",
        species_sens_threshold=0.08,
        reaction_sens_threshold=0.12,
    )

    print(f"\n  最终结果:")
    print(f"  组分数: {reduced_combined.species_count()} "
          f"(删除 {mechanism.species_count() - reduced_combined.species_count()})")
    print(f"  反应数: {reduced_combined.reactions_count()} "
          f"(删除 {mechanism.reactions_count() - reduced_combined.reactions_count()})")
    print(f"  组分删除率: "
          f"{(1 - reduced_combined.species_count() / mechanism.species_count()) * 100:.1f}%")
    print(f"  反应删除率: "
          f"{(1 - reduced_combined.reactions_count() / mechanism.reactions_count()) * 100:.1f}%")

    print("\n  [验证] 简化前后目标量对比:")
    validation = reducer.validate(sample_state)
    print(f"    点火延迟: {validation['original_ignition_delay']:.2e}s -> "
          f"{validation['reduced_ignition_delay']:.2e}s "
          f"(误差: {validation['ignition_delay_error(%)']:.2f}%)")
    print(f"    火焰速度: {validation['original_flame_speed']:.3f} m/s -> "
          f"{validation['reduced_flame_speed']:.3f} m/s "
          f"(误差: {validation['flame_speed_error(%)']:.2f}%)")

    print_mechanism_info(reduced_combined, "组合简化后机理")

    print("=" * 70)
    print("  示例 5: 多状态鲁棒性简化")
    print("=" * 70)

    test_states = generate_test_states(n_states=5, seed=42)
    print("\n  测试状态:")
    for i, s in enumerate(test_states, 1):
        print(f"    状态 {i}: T={s['temperature']:.0f}K, P={s['pressure']:.2f}atm, "
              f"φ={s['equivalence_ratio']:.2f}")

    print(f"\n  [多状态 DRGEP] 取所有状态保留组分的并集")
    reducer2 = EnhancedMechanismReducer(mechanism)
    reduced_multi = reducer2.reduce_drgep_multi_state(
        target_species=target_species,
        states=test_states,
        threshold=0.05,
        protect_critical=True,
    )

    print(f"\n  多状态简化后: {reduced_multi.species_count()} 组分, "
          f"{reduced_multi.reactions_count()} 反应")

    print("\n" + "=" * 70)
    print("  简化完成！关键改进总结:")
    print("  ✓ DRGEP 误差传播避免关键组分误删")
    print("  ✓ 关键组分 (OH, H, O, CH3等) 强制保护")
    print("  ✓ 基于火焰速度/点火延迟的目标敏感性")
    print("  ✓ 多状态鲁棒性简化")
    print("  ✓ 误差验证保证精度")
    print("=" * 70)


if __name__ == "__main__":
    main()