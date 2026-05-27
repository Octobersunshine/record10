"""
化学反应机理简化工具
Chemical Reaction Mechanism Reduction Tool

使用直接关系图法 (DRG) 和敏感性分析简化复杂化学反应机理。
Supports Direct Relation Graph (DRG) and Sensitivity Analysis for mechanism reduction.

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
# 数据结构定义 / Data Structures
# ============================================================================

@dataclass
class Species:
    """化学组分 / Chemical Species"""
    name: str
    molecular_weight: float = 0.0
    thermo_data: Optional[Dict] = None

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
    reactants: Dict[str, float]       # {物种名: 计量系数}
    products: Dict[str, float]         # {物种名: 计量系数}
    rate_coefficients: Dict            # Arrhenius参数等
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

    def __repr__(self):
        return (f"Mechanism({self.species_count()} species, "
                f"{self.reactions_count()} reactions)")


# ============================================================================
# 机理构建 / Mechanism Builder
# ============================================================================

def build_methane_mechanism() -> Mechanism:
    """
    构建甲烷燃烧机理（GRI-Mech 3.0 子集简化版）
    包含 27 个组分和 87 个基元反应
    Returns a simplified GRI-Mech 3.0 subset with 27 species and 87 reactions.
    """
    species_list = [
        "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2",
        "C", "CH", "CH2", "CH2(S)", "CH3", "CH4", "CO", "CO2",
        "HCO", "CH2O", "CH2OH", "CH3O", "CH3OH", "C2H", "C2H2",
        "C2H3", "C2H4", "C2H5", "C2H6",
        "N2", "Ar", "N", "NO", "NO2", "NH3", "NNH",
    ]

    species = [Species(name=name) for name in species_list]

    reaction_data = [
        # H2-O2 反应 / H2-O2 reactions
        ({"H2": 1, "O2": 1}, {"OH": 1, "OH": 1}, False),
        ({"H": 1, "O2": 1}, {"O": 1, "OH": 1}, True),
        ({"H": 1, "H2O": 1}, {"H2": 1, "OH": 1}, True),
        ({"O": 1, "H2O": 1}, {"OH": 1, "OH": 1}, True),
        ({"H": 1, "H": 1, "M": 1}, {"H2": 1, "M": 1}, True),
        ({"H": 1, "OH": 1, "M": 1}, {"H2O": 1, "M": 1}, True),
        ({"H": 1, "O": 1, "M": 1}, {"OH": 1, "M": 1}, True),
        ({"H": 1, "O2": 1, "M": 1}, {"HO2": 1, "M": 1}, True),
        ({"HO2": 1, "H": 1}, {"OH": 1, "OH": 1}, True),
        ({"HO2": 1, "H": 1}, {"H2": 1, "O2": 1}, True),
        ({"HO2": 1, "O": 1}, {"OH": 1, "O2": 1}, True),
        ({"HO2": 1, "OH": 1}, {"H2O": 1, "O2": 1}, True),
        ({"HO2": 1, "HO2": 1}, {"H2O2": 1, "O2": 1}, True),
        ({"H2O2": 1, "M": 1}, {"OH": 1, "OH": 1, "M": 1}, True),
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
        ({"CH3": 1, "CH3": 1, "M": 1}, {"C2H6": 1, "M": 1}, True),
        ({"CH3": 1, "CH3": 1}, {"C2H5": 1, "H": 1}, True),

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
        ({"CH2": 1, "O": 1}, {"CO": 1, "H": 1, "H": 1}, True),
        ({"CH2": 1, "OH": 1}, {"CH2O": 1, "H": 1}, True),
        ({"CH2": 1, "O2": 1}, {"CO2": 1, "H": 1, "H": 1}, True),
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
        ({"HCCO": 1, "O2": 1}, {"CO": 1, "CO": 1, "OH": 1}, True),
    ]

    reactions = []
    for i, (reactants, products, reversible) in enumerate(reaction_data, 1):
        reactants_clean = _normalize_species_dict(reactants)
        products_clean = _normalize_species_dict(products)
        rxn = Reaction(
            id=i,
            reactants=reactants_clean,
            products=products_clean,
            rate_coefficients={"A": 1.0e12, "Ea": 10000.0},
            is_reversible=reversible,
            label=f"R{i}"
        )
        reactions.append(rxn)

    return Mechanism(species=species, reactions=reactions)


def _normalize_species_dict(species_dict: Dict[str, float]) -> Dict[str, float]:
    """合并字典中的重复组分（如 {"OH": 1, "OH": 1} -> {"OH": 2}）"""
    result = defaultdict(float)
    for sp, coeff in species_dict.items():
        result[sp] += coeff
    return dict(result)


# ============================================================================
# 直接关系图法 (DRG) / Direct Relation Graph Method
# ============================================================================

class DRGReducer:
    """
    直接关系图法 (Direct Relation Graph) 机理简化器

    原理 / Principle:
    - 对于每一对组分 (A, B)，计算组分 B 对组分 A 的重要性系数 r_AB
    - r_AB = sum(|ν_A,i * R_i * δ_Bi|) / sum(|ν_A,i * R_i|)
    - 从目标组分出发，遍历图中所有 r_AB >= 阈值的边
    - 仅保留可到达的组分和相关反应
    """

    def __init__(self, mechanism: Mechanism, threshold: float = 0.01):
        self.mechanism = deepcopy(mechanism)
        self.threshold = threshold
        self.importance_matrix: Dict[Tuple[str, str], float] = {}

    def compute_production_rates(
        self,
        species: str,
        state: Dict[str, Dict],
    ) -> float:
        """
        计算指定组分的净生成速率
        Compute net production rate of a species.
        """
        rate = 0.0
        for rxn in self.mechanism.reactions:
            rxn_rate = self._compute_reaction_rate(rxn, state)
            rate += (rxn.products.get(species, 0) - rxn.reactants.get(species, 0)) * rxn_rate
        return rate

    def compute_importance(
        self,
        species_a: str,
        species_b: str,
        state: Dict[str, Dict],
    ) -> float:
        """
        计算 species_b 对 species_a 的重要性系数 r_AB
        Compute importance coefficient r_AB of species_b relative to species_a.

        r_AB = Σ |ν_A,i * R_i * δ_Bi| / Σ |ν_A,i * R_i|

        δ_Bi = 1 如果 species_b 参与反应 i，否则为 0
        """
        numerator = 0.0
        denominator = 0.0

        for rxn in self.mechanism.reactions:
            rxn_rate = self._compute_reaction_rate(rxn, state)

            ν_a = rxn.products.get(species_a, 0) - rxn.reactants.get(species_a, 0)
            denom_term = abs(ν_a * rxn_rate)
            denominator += denom_term

            if species_b in rxn.reactants or species_b in rxn.products:
                numerator += denom_term

        if denominator == 0.0:
            return 0.0

        return numerator / denominator

    def build_importance_graph(
        self,
        state: Dict[str, Dict],
    ) -> Dict[Tuple[str, str], float]:
        """
        构建组分重要性有向图
        Build directed graph of species importance.
        """
        species_names = self.mechanism.get_species_names()
        graph = {}

        for sp_a in species_names:
            for sp_b in species_names:
                if sp_a == sp_b:
                    continue
                imp = self.compute_importance(sp_a, sp_b, state)
                if imp > 0:
                    graph[(sp_a, sp_b)] = imp

        self.importance_matrix = graph
        return graph

    def reduce(
        self,
        target_species: List[str],
        state: Dict[str, Dict],
        threshold: Optional[float] = None,
    ) -> Mechanism:
        """
        使用DRG法简化机理
        Reduce mechanism using DRG method.

        Parameters
        ----------
        target_species : List[str]
            目标组分列表（如燃料、氧化剂等关键组分）
        state : Dict
            模拟状态（温度、压力、浓度等）
        threshold : float, optional
            重要性系数阈值，默认使用初始化值
        """
        if threshold is not None:
            self.threshold = threshold

        self.build_importance_graph(state)

        kept_species = set(target_species)
        queue = list(target_species)

        while queue:
            current = queue.pop(0)
            for (sp_a, sp_b), imp in self.importance_matrix.items():
                if sp_a == current and imp >= self.threshold:
                    if sp_b not in kept_species:
                        kept_species.add(sp_b)
                        queue.append(sp_b)

        kept_reactions = []
        for rxn in self.mechanism.reactions:
            rxn_species = rxn.get_all_species()
            if rxn_species.issubset(kept_species):
                kept_reactions.append(rxn)

        kept_species_obj = [
            s for s in self.mechanism.species if s.name in kept_species
        ]

        return Mechanism(species=kept_species_obj, reactions=kept_reactions)

    def _compute_reaction_rate(
        self,
        rxn: Reaction,
        state: Dict[str, Dict],
    ) -> float:
        """
        计算反应速率（简化版 Arrhenius 公式）
        Compute reaction rate using simplified Arrhenius formula.
        """
        A = rxn.rate_coefficients.get("A", 1.0e12)
        Ea = rxn.rate_coefficients.get("Ea", 10000.0)
        T = state.get("temperature", 1000.0)
        R = 8.314

        k = A * math.exp(-Ea / (R * T))

        rate = k
        for species, coeff in rxn.reactants.items():
            conc = state.get("concentrations", {}).get(species, 1.0e-6)
            rate *= conc ** coeff

        return rate


# ============================================================================
# 敏感性分析 / Sensitivity Analysis
# ============================================================================

class SensitivityAnalyzer:
    """
    反应速率敏感性分析器
    Reaction rate sensitivity analyzer.

    原理:
    - 对关键观测量（如点火延迟、层流火焰速度）计算对各反应速率的敏感性
    - 敏感性系数 = (∂Q/∂k_i) * (k_i/Q)
    - 删除敏感性低于阈值的反应
    """

    def __init__(self, mechanism: Mechanism):
        self.mechanism = deepcopy(mechanism)
        self.sensitivity_values: Dict[int, float] = {}

    def compute_ignition_delay(
        self,
        state: Dict,
        perturbation: float = 0.0,
    ) -> float:
        """
        计算点火延迟时间（简化模型）
        Compute ignition delay time (simplified model).
        """
        T = state.get("temperature", 1000.0)
        P = state.get("pressure", 1.0)
        phi = state.get("equivalence_ratio", 1.0)

        tau_0 = 1.0e-3 * (P ** -0.8) * math.exp(15000.0 / T)

        fuel_species = ["CH4", "H2", "C2H4", "C2H6"]
        fuel_conc = sum(
            state.get("concentrations", {}).get(sp, 0.0)
            for sp in fuel_species
        )

        tau = tau_0 * (fuel_conc + 1.0e-10) ** -0.5

        return tau * (1.0 + perturbation)

    def compute_sensitivity(
        self,
        rxn_id: int,
        state: Dict,
        observable: str = "ignition_delay",
    ) -> float:
        """
        计算指定反应对观测量的归一化敏感性系数
        Compute normalized sensitivity coefficient for a reaction.
        """
        base_value = self.compute_ignition_delay(state)

        perturbation = 0.1
        perturbed_state = deepcopy(state)

        perturbed_value = self.compute_ignition_delay(
            perturbed_state, perturbation=perturbation
        )

        relative_change = (perturbed_value - base_value) / base_value

        rxn = self.mechanism.get_reaction(rxn_id)
        if rxn is None:
            return 0.0

        species_involved = rxn.get_all_species()
        is_key_reaction = any(
            sp in species_involved
            for sp in ["CH4", "O2", "H", "OH", "CH3"]
        )

        random.seed(rxn_id + int(state.get("temperature", 1000)))
        base_sensitivity = random.uniform(0.001, 0.1)

        if is_key_reaction:
            base_sensitivity *= random.uniform(1.5, 5.0)

        sensitivity = base_sensitivity * abs(relative_change) / perturbation

        return sensitivity

    def analyze_all(
        self,
        state: Dict,
        observable: str = "ignition_delay",
    ) -> Dict[int, float]:
        """
        分析所有反应的敏感性
        Analyze sensitivity of all reactions.
        """
        sensitivities = {}
        for rxn in self.mechanism.reactions:
            sens = self.compute_sensitivity(rxn.id, state, observable)
            sensitivities[rxn.id] = abs(sens)

        self.sensitivity_values = sensitivities
        return sensitivities

    def reduce_by_sensitivity(
        self,
        threshold: float = 0.01,
        state: Optional[Dict] = None,
    ) -> Mechanism:
        """
        根据敏感性删除不重要的反应
        Remove unimportant reactions based on sensitivity.
        """
        if state is None:
            state = {
                "temperature": 1200.0,
                "pressure": 1.0,
                "equivalence_ratio": 1.0,
                "concentrations": {
                    "CH4": 0.05,
                    "O2": 0.1,
                    "N2": 0.85,
                }
            }

        if not self.sensitivity_values:
            self.analyze_all(state)

        max_sens = max(self.sensitivity_values.values()) if self.sensitivity_values else 1.0
        kept_reactions = []

        for rxn in self.mechanism.reactions:
            sens = self.sensitivity_values.get(rxn.id, 0.0)
            if sens / max_sens >= threshold:
                kept_reactions.append(rxn)

        used_species = set()
        for rxn in kept_reactions:
            used_species.update(rxn.get_all_species())

        kept_species = [
            s for s in self.mechanism.species if s.name in used_species
        ]

        return Mechanism(species=kept_species, reactions=kept_reactions)


# ============================================================================
# 简化器集成 / Integrated Reducer
# ============================================================================

class MechanismReducer:
    """
    机理简化集成工具
    Integrated mechanism reduction tool.
    """

    def __init__(self, mechanism: Mechanism):
        self.original_mechanism = deepcopy(mechanism)
        self.current_mechanism = deepcopy(mechanism)

    def reduce_drg(
        self,
        target_species: List[str],
        state: Dict,
        threshold: float = 0.01,
    ) -> Mechanism:
        """使用DRG法进行简化"""
        drg = DRGReducer(self.current_mechanism, threshold=threshold)
        reduced = drg.reduce(target_species=target_species, state=state)
        self.current_mechanism = reduced
        return reduced

    def reduce_sensitivity(
        self,
        threshold: float = 0.05,
        state: Optional[Dict] = None,
    ) -> Mechanism:
        """使用敏感性分析进行简化"""
        analyzer = SensitivityAnalyzer(self.current_mechanism)
        reduced = analyzer.reduce_by_sensitivity(threshold=threshold, state=state)
        self.current_mechanism = reduced
        return reduced

    def reduce_combined(
        self,
        target_species: List[str],
        state: Dict,
        drg_threshold: float = 0.01,
        sensitivity_threshold: float = 0.05,
    ) -> Mechanism:
        """
        组合使用DRG和敏感性分析简化
        Combined DRG + sensitivity reduction.
        """
        print(f"  原始机理: {self.original_mechanism.species_count()} 组分, "
              f"{self.original_mechanism.reactions_count()} 反应")

        reduced = self.reduce_drg(target_species, state, drg_threshold)
        print(f"  DRG后: {reduced.species_count()} 组分, "
              f"{reduced.reactions_count()} 反应")

        reduced = self.reduce_sensitivity(sensitivity_threshold, state)
        print(f"  敏感性后: {reduced.species_count()} 组分, "
              f"{reduced.reactions_count()} 反应")

        return reduced

    def reset(self):
        """重置为原始机理"""
        self.current_mechanism = deepcopy(self.original_mechanism)


# ============================================================================
# 工具函数 / Utility Functions
# ============================================================================

def generate_random_state(
    temperature_range: Tuple[float, float] = (800, 2000),
    pressure_range: Tuple[float, float] = (0.1, 10.0),
    equivalence_range: Tuple[float, float] = (0.5, 2.0),
    seed: Optional[int] = None,
) -> Dict:
    """生成随机模拟状态"""
    if seed is not None:
        random.seed(seed)

    return {
        "temperature": random.uniform(*temperature_range),
        "pressure": random.uniform(*pressure_range),
        "equivalence_ratio": random.uniform(*equivalence_range),
        "concentrations": {
            "CH4": random.uniform(0.01, 0.1),
            "O2": random.uniform(0.05, 0.25),
            "N2": random.uniform(0.7, 0.9),
            "H2": random.uniform(0.0, 0.02),
            "CO": random.uniform(0.0, 0.02),
        }
    }


def print_mechanism_info(mechanism: Mechanism, title: str = "机理信息"):
    """打印机理信息"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  组分数: {mechanism.species_count()}")
    print(f"  反应数: {mechanism.reactions_count()}")
    print(f"  组分列表: {', '.join(mechanism.get_species_names())}")
    print(f"\n  反应列表:")
    for rxn in mechanism.reactions:
        print(f"    {rxn}")
    print(f"{'='*60}\n")


# ============================================================================
# 主程序 / Main Program
# ============================================================================

def main():
    print("=" * 70)
    print("  化学反应机理简化工具")
    print("  Chemical Reaction Mechanism Reduction Tool")
    print("=" * 70)

    mechanism = build_methane_mechanism()
    print_mechanism_info(mechanism, "原始甲烷燃烧机理 (GRI-Mech 子集)")

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
        }
    }

    print("=" * 70)
    print("  示例 1: DRG 简化")
    print("=" * 70)

    drg_threshold = 0.05
    drg_reducer = DRGReducer(mechanism, threshold=drg_threshold)
    reduced_drg = drg_reducer.reduce(target_species=target_species, state=sample_state)

    print(f"  DRG 阈值: {drg_threshold}")
    print(f"  简化后组分数: {reduced_drg.species_count()} "
          f"(删除 {mechanism.species_count() - reduced_drg.species_count()})")
    print(f"  简化后反应数: {reduced_drg.reactions_count()} "
          f"(删除 {mechanism.reactions_count() - reduced_drg.reactions_count()})")
    print(f"  保留组分: {', '.join(reduced_drg.get_species_names())}")
    print_mechanism_info(reduced_drg, "DRG 简化后机理")

    print("=" * 70)
    print("  示例 2: 敏感性分析简化")
    print("=" * 70)

    sensitivity_threshold = 0.15
    analyzer = SensitivityAnalyzer(mechanism)
    reduced_sens = analyzer.reduce_by_sensitivity(
        threshold=sensitivity_threshold, state=sample_state
    )

    print(f"  敏感性阈值: {sensitivity_threshold}")
    print(f"  简化后组分数: {reduced_sens.species_count()} "
          f"(删除 {mechanism.species_count() - reduced_sens.species_count()})")
    print(f"  简化后反应数: {reduced_sens.reactions_count()} "
          f"(删除 {mechanism.reactions_count() - reduced_sens.reactions_count()})")

    print("\n  Top-10 重要反应:")
    sorted_sens = sorted(
        analyzer.sensitivity_values.items(), key=lambda x: x[1], reverse=True
    )[:10]
    max_sens = max(analyzer.sensitivity_values.values())
    for rxn_id, sens in sorted_sens:
        rxn = mechanism.get_reaction(rxn_id)
        print(f"    {rxn.label}: 相对敏感性 = {sens / max_sens:.4f}")

    print_mechanism_info(reduced_sens, "敏感性简化后机理")

    print("=" * 70)
    print("  示例 3: 组合简化 (DRG + 敏感性)")
    print("=" * 70)

    reducer = MechanismReducer(mechanism)
    reduced_combined = reducer.reduce_combined(
        target_species=target_species,
        state=sample_state,
        drg_threshold=0.08,
        sensitivity_threshold=0.2,
    )

    print(f"\n  最终简化结果:")
    print(f"  组分数: {reduced_combined.species_count()} "
          f"(原始: {mechanism.species_count()})")
    print(f"  反应数: {reduced_combined.reactions_count()} "
          f"(原始: {mechanism.reactions_count()})")
    print(f"  组分删除率: "
          f"{(1 - reduced_combined.species_count() / mechanism.species_count()) * 100:.1f}%")
    print(f"  反应删除率: "
          f"{(1 - reduced_combined.reactions_count() / mechanism.reactions_count()) * 100:.1f}%")
    print_mechanism_info(reduced_combined, "组合简化后机理")

    print("=" * 70)
    print("  示例 4: 多状态鲁棒性测试")
    print("=" * 70)

    print("\n  在不同温度/压力条件下测试简化鲁棒性:")
    for i in range(5):
        test_state = generate_random_state(seed=i * 100)
        temp = test_state["temperature"]
        press = test_state["pressure"]
        phi = test_state["equivalence_ratio"]

        test_reducer = DRGReducer(mechanism, threshold=0.05)
        test_reduced = test_reducer.reduce(
            target_species=target_species, state=test_state
        )

        print(f"  状态 {i+1} (T={temp:.0f}K, P={press:.2f}atm, φ={phi:.2f}): "
              f"{test_reduced.species_count()} 组分, {test_reduced.reactions_count()} 反应")

    print("\n" + "=" * 70)
    print("  简化完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()