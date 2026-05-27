"""
胶体颗粒絮凝过程模拟 —— 基于群体平衡方程 (PBE)
================================================

本模块使用多种数值方法求解Smoluchowski方程（含聚集和破裂），
模拟胶体颗粒在布朗运动和剪切流作用下的聚集-破裂动力学。

数学模型:
    ∂n(v,t)/∂t = 聚集源项 + 破裂源项

聚集源项:
    1/2 ∫₀ᵛ β(v', v-v') n(v',t) n(v-v',t) dv' - n(v,t) ∫₀^∞ β(v,v') n(v',t) dv'

破裂源项:
    ∫ᵛ^∞ b(v,v') Γ(v') n(v',t) dv' - Γ(v) n(v,t)

碰撞核:
    - 布朗运动核 (Smoluchowski, 1917)
    - 剪切流聚集核 (Saffman & Turner, 1956)

破裂核:
    - 剪切流破裂核 (Luo & Svendsen, 1996)
    - 二进制破裂子项分布

数值方法:
    - 固定支点离散分组法 (Fixed-Pivot Technique, Kumar & Ramkrishna, 1996)
    - 矩量法 (QMOM, Quadrature Method of Moments, McGraw, 1997)
    - 蒙特卡洛方法 (Monte Carlo, Time-Driven)

数值弥散问题:
    离散分组法存在数值弥散(虚假扩散)，导致粒径分布峰出现人为拖尾。
    QMOM和蒙特卡洛方法可以有效抑制这一问题。

动态平衡:
    当聚集速率与破裂速率达到平衡时，粒径分布趋于稳态。
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# =============================================================================
# 物理参数
# =============================================================================
@dataclass
class PhysicalParams:
    """物理参数集合"""
    temperature: float = 298.15
    viscosity: float = 0.001
    kB: float = 1.380649e-23
    shear_rate: float = 10.0
    enable_brownian: bool = True
    enable_shear_aggregation: bool = True
    enable_breakage: bool = True
    breakage_rate: float = 0.01
    surface_tension: float = 0.072
    critical_stress: float = 1.0


# =============================================================================
# 离散分组设置
# =============================================================================
@dataclass
class Discretization:
    """离散分组设置"""
    v_min: float = 1e-27
    v_max: float = 1e-18
    n_sections: int = 40
    geometric_ratio: float = 2.0

    def __post_init__(self):
        self.volumes, self.boundaries = self._generate_volumes()
        self.section_widths = np.diff(self.boundaries)

    def _generate_volumes(self) -> Tuple[np.ndarray, np.ndarray]:
        boundaries = np.zeros(self.n_sections + 1)
        boundaries[0] = self.v_min
        for i in range(1, self.n_sections + 1):
            boundaries[i] = boundaries[i - 1] * self.geometric_ratio
        volumes = np.sqrt(boundaries[:-1] * boundaries[1:])
        return volumes, boundaries

    @property
    def diameters(self) -> np.ndarray:
        return (6.0 * self.volumes / np.pi) ** (1.0 / 3.0)


# =============================================================================
# 碰撞核（聚集）
# =============================================================================
class CollisionKernel:
    """聚集碰撞核计算器"""

    def __init__(self, params: PhysicalParams):
        self.params = params

    def brownian_kernel(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        布朗运动碰撞核 (Smoluchowski)

        β_B(v1, v2) = 2k_BT/(3μ) * (1/v1^(1/3) + 1/v2^(1/3)) * (v1^(1/3) + v2^(1/3))
        """
        a1 = (3.0 * v1 / (4.0 * np.pi)) ** (1.0 / 3.0)
        a2 = (3.0 * v2 / (4.0 * np.pi)) ** (1.0 / 3.0)

        D1 = self.params.kB * self.params.temperature / (6.0 * np.pi * self.params.viscosity * a1)
        D2 = self.params.kB * self.params.temperature / (6.0 * np.pi * self.params.viscosity * a2)

        W = (a1 + a2) * (D1 + D2)
        beta = 2.0 * np.pi * W
        return beta

    def shear_kernel(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        剪切流聚集核 (Saffman & Turner)

        β_S(v1, v2) = 4/3 * γ * (a1 + a2)^3
        """
        a1 = (3.0 * v1 / (4.0 * np.pi)) ** (1.0 / 3.0)
        a2 = (3.0 * v2 / (4.0 * np.pi)) ** (1.0 / 3.0)

        beta = (4.0 / 3.0) * self.params.shear_rate * (a1 + a2) ** 3
        return beta

    def total_kernel(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """总聚集核"""
        beta = np.zeros_like(v1)
        if self.params.enable_brownian:
            beta += self.brownian_kernel(v1, v2)
        if self.params.enable_shear_aggregation:
            beta += self.shear_kernel(v1, v2)
        return beta


# =============================================================================
# 破裂核
# =============================================================================
class BreakageKernel:
    """
    破裂核计算器

    包含:
    - 破裂速率 Γ(v): 体积为v的絮团单位时间破裂次数
    - 子项分布 b(v', v): 体积为v的絮团破裂后产生体积为v'子絮团的概率
    """

    def __init__(self, params: PhysicalParams):
        self.params = params

    def breakage_rate(self, v: np.ndarray) -> np.ndarray:
        """
        剪切流破裂速率 (Luo & Svendsen, 1996)

        Γ(v) = k * γ * (v / v_min)^(1/3) * exp(-σ_c / (μγv^(2/3)))

        大絮团更容易破裂
        """
        v_min = 1e-27
        a = (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0)

        shear_stress = self.params.viscosity * self.params.shear_rate
        breakage_prob = np.exp(-self.params.critical_stress / np.maximum(shear_stress * (a * 1e6) ** 2, 1e-10))

        rate = self.params.breakage_rate * self.params.shear_rate * (v / v_min) ** (1/3) * breakage_prob

        rate = np.minimum(rate, 1e6)
        return rate

    def daughter_distribution(self, v_parent: np.ndarray, v_daughter: np.ndarray) -> np.ndarray:
        """
        二进制破裂子项分布

        等体积破裂: 一个絮团破裂成两个等体积的子絮团
        b(v', v) = 2 * δ(v' - v/2)
        """
        v_expected = v_parent / 2.0
        sigma = 0.2 * v_expected

        b = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * \
            np.exp(-0.5 * ((v_daughter - v_expected) / sigma) ** 2)

        b = np.where(v_daughter < v_parent, b, 0)
        b = np.where(v_daughter > 0, b, 0)

        return b

    def binary_breakage_volumes(self, v: float) -> Tuple[float, float]:
        """二进制破裂：生成两个子絮团的体积"""
        ratio = 0.5 + 0.1 * np.random.randn()
        ratio = np.clip(ratio, 0.2, 0.8)
        return v * ratio, v * (1.0 - ratio)


# =============================================================================
# 初始分布
# =============================================================================
def initial_distribution(
    discret: Discretization,
    total_concentration: float = 1e18,
    geometric_mean_volume: float = 1e-24,
    geometric_std: float = 1.5
) -> np.ndarray:
    """
    生成对数正态初始粒径分布

    Parameters
    ----------
    total_concentration : 总颗粒数浓度 [m^-3]
    geometric_mean_volume : 几何平均体积 [m^3]
    geometric_std : 几何标准差
    """
    d = discret.diameters
    d_mean = (6.0 * geometric_mean_volume / np.pi) ** (1.0 / 3.0)

    log_d = np.log(d)
    log_d_mean = np.log(d_mean)
    log_sigma = np.log(geometric_std)

    weights = np.exp(-0.5 * ((log_d - log_d_mean) / log_sigma) ** 2)
    weights = weights / (log_sigma * d * np.sqrt(2.0 * np.pi))
    weights = weights / np.sum(weights)

    N = total_concentration * weights
    return N


# =============================================================================
# PBE 求解器 (离散分组法)
# =============================================================================
class PBESolver:
    """
    群体平衡方程求解器 (聚集+破裂)

    使用固定支点离散分组法 (Fixed-Pivot Technique, Kumar & Ramkrishna, 1996)
    优化: 预计算所有聚集事件的目标区间和权重，实现向量化运算
    """

    def __init__(
        self,
        discret: Discretization,
        aggregation_kernel: Optional[CollisionKernel] = None,
        breakage_kernel: Optional[BreakageKernel] = None,
        initial_N: Optional[np.ndarray] = None,
        enable_aggregation: bool = True,
        enable_breakage: bool = True
    ):
        self.discret = discret
        self.aggregation_kernel = aggregation_kernel
        self.breakage_kernel = breakage_kernel
        self.enable_aggregation = enable_aggregation
        self.enable_breakage = enable_breakage
        self.N = initial_N if initial_N is not None else np.zeros(discret.n_sections)

        if self.enable_aggregation and self.aggregation_kernel is not None:
            self._precompute_aggregation_tables()
        if self.enable_breakage and self.breakage_kernel is not None:
            self._precompute_breakage_tables()

    def _precompute_aggregation_tables(self):
        """预计算聚集相关表"""
        V = self.discret.volumes
        n = self.discret.n_sections
        v1, v2 = np.meshgrid(V, V, indexing="ij")
        v1_flat = v1.ravel()
        v2_flat = v2.ravel()
        beta_flat = self.aggregation_kernel.total_kernel(v1_flat, v2_flat)
        self.beta_matrix = beta_flat.reshape(n, n)

        v_sum = V[:, np.newaxis] + V[np.newaxis, :]
        k_low = np.searchsorted(V, v_sum) - 1
        k_low = np.clip(k_low, 0, n - 2)
        k_high = k_low + 1

        v_low = V[k_low]
        v_high = V[k_high]
        dv = v_high - v_low

        eta_high = np.where(dv > 0, (v_sum - v_low) / dv, 1.0)
        eta_low = np.where(dv > 0, (v_high - v_sum) / dv, 0.0)

        mask_same = k_low == k_high
        eta_low = np.where(mask_same, 1.0, eta_low)
        eta_high = np.where(mask_same, 0.0, eta_high)

        self.agg_k_low = k_low
        self.agg_k_high = k_high
        self.agg_eta_low = eta_low
        self.agg_eta_high = eta_high

        self.factor_matrix = 0.5 * np.ones((n, n))

    def _precompute_breakage_tables(self):
        """预计算破裂相关表 - 二进制等体积破裂"""
        V = self.discret.volumes
        n = self.discret.n_sections

        self.gamma = self.breakage_kernel.breakage_rate(V)

        self.breakage_matrix = np.zeros((n, n))
        for i in range(n):
            v_parent = V[i]
            v_daughter = v_parent / 2.0

            idx = np.searchsorted(V, v_daughter) - 1
            idx = np.clip(idx, 0, n - 2)

            v_low = V[idx]
            v_high = V[idx + 1]
            dv = v_high - v_low

            if dv > 0:
                weight_high = (v_daughter - v_low) / dv
                weight_low = 1.0 - weight_high

                self.breakage_matrix[idx, i] = weight_low
                self.breakage_matrix[idx + 1, i] = weight_high
            else:
                self.breakage_matrix[idx, i] = 1.0

            total_number = np.sum(self.breakage_matrix[:, i])
            if total_number > 0:
                self.breakage_matrix[:, i] *= 2.0 / total_number

            total_volume = np.sum(V * self.breakage_matrix[:, i])
            if total_volume > 0:
                self.breakage_matrix[:, i] *= v_parent / total_volume

    def aggregation_source_term(self, N: np.ndarray) -> np.ndarray:
        """计算聚集源项 dN/dt"""
        if not self.enable_aggregation:
            return np.zeros_like(N)

        n = self.discret.n_sections

        sink = N * (self.beta_matrix @ N)

        NN = np.outer(N, N)
        weighted = self.beta_matrix * self.factor_matrix * NN

        source = np.zeros(n)
        for k in range(n):
            mask_low = self.agg_k_low == k
            mask_high = self.agg_k_high == k
            source[k] = np.sum(self.agg_eta_low[mask_low] * weighted[mask_low])
            source[k] += np.sum(self.agg_eta_high[mask_high] * weighted[mask_high])

        return source - sink

    def breakage_source_term(self, N: np.ndarray) -> np.ndarray:
        """
        计算破裂源项 dN/dt

        破裂源项 = 产生项 - 消耗项
        产生项: ∫ᵛ^∞ b(v,v') Γ(v') n(v') dv'
        消耗项: -Γ(v) n(v)
        """
        if not self.enable_breakage:
            return np.zeros_like(N)

        sink = -self.gamma * N

        source = self.breakage_matrix @ (self.gamma * N)

        return source + sink

    def _ode_rhs(self, t: float, N: np.ndarray) -> np.ndarray:
        """ODE右端函数"""
        N_pos = np.maximum(N, 0.0)
        dNdt = np.zeros_like(N_pos)

        if self.enable_aggregation:
            dNdt += self.aggregation_source_term(N_pos)
        if self.enable_breakage:
            dNdt += self.breakage_source_term(N_pos)

        dNdt = np.clip(dNdt, -1e20, 1e20)
        return dNdt

    def solve(
        self,
        t_span: Tuple[float, float],
        n_times: int = 100,
        method: str = "LSODA",
        rtol: float = 1e-6,
        atol: float = 1e-30
    ) -> dict:
        """求解PBE"""
        t_eval = np.linspace(t_span[0], t_span[1], n_times)

        sol = solve_ivp(
            self._ode_rhs,
            t_span,
            self.N,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=(t_span[1] - t_span[0]) / 50.0
        )

        return {
            "t": sol.t,
            "N": sol.y,
            "success": sol.success,
            "message": sol.message
        }


# =============================================================================
# PBE 求解器 (矩量法 QMOM)
# =============================================================================
class QMOMSolver:
    """
    矩量法 (Quadrature Method of Moments, McGraw 1997)

    通过跟踪分布的矩 (m0, m1, ..., m_{2N-1}) 来描述粒径分布。
    使用乘积差分算法 (Product-Difference Algorithm) 从矩中
    重建求积节点 (abscissas) 和权重 (weights)。

    支持聚集和破裂过程。
    """

    def __init__(
        self,
        aggregation_kernel: Optional[CollisionKernel] = None,
        breakage_kernel: Optional[BreakageKernel] = None,
        initial_moments: Optional[np.ndarray] = None,
        n_nodes: int = 3,
        enable_aggregation: bool = True,
        enable_breakage: bool = True
    ):
        self.aggregation_kernel = aggregation_kernel
        self.breakage_kernel = breakage_kernel
        self.n_nodes = n_nodes
        self.n_moments = 2 * n_nodes
        self.enable_aggregation = enable_aggregation
        self.enable_breakage = enable_breakage

        if initial_moments is not None:
            self.moments = initial_moments
        else:
            self.moments = np.zeros(self.n_moments)

    @staticmethod
    def product_difference_algorithm(moments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        乘积差分算法 (Gordon 1968)

        从矩序列 {m_k, k=0,...,2N-1} 计算求积节点和权重。
        """
        n = len(moments) // 2

        P = np.zeros((2 * n, 2 * n))

        for i in range(2 * n):
            P[i, 0] = moments[i]

        for j in range(1, 2 * n):
            for i in range(2 * n - j):
                P[i, j] = P[0, j - 1] * P[i + 1, j - 2] - P[0, j - 2] * P[i + 1, j - 1]
                if j > 2 and P[0, j - 2] != 0:
                    P[i, j] /= P[0, j - 2]

        sigma = np.zeros(2 * n + 1)
        sigma[0] = 0.0
        for i in range(2 * n):
            if i < n:
                sigma[i + 1] = P[0, 2 * i + 1] / P[0, 2 * i] if P[0, 2 * i] != 0 else 0.0
            else:
                sigma[i + 1] = P[i - n + 1, 2 * n - i] / P[i - n, 2 * n - i + 1] \
                    if P[i - n, 2 * n - i + 1] != 0 else 0.0

        J = np.zeros((n, n))
        for i in range(n):
            J[i, i] = sigma[2 * i] + sigma[2 * i + 1]
            if i < n - 1:
                J[i + 1, i] = -np.sqrt(max(sigma[2 * i + 2] * sigma[2 * i + 1], 0))
                J[i, i + 1] = J[i + 1, i]

        eigenvalues, eigenvectors = np.linalg.eigh(J)

        abscissas = np.sort(np.maximum(eigenvalues, 0))

        weights = P[0, 0] * eigenvectors[0, :] ** 2
        weights = np.sum(weights) * weights / np.sum(weights)

        return abscissas, weights

    @staticmethod
    def moments_from_distribution(
        volumes: np.ndarray,
        N: np.ndarray,
        n_moments: int
    ) -> np.ndarray:
        """从离散分布计算矩"""
        moments = np.zeros(n_moments)
        for k in range(n_moments):
            moments[k] = np.sum(volumes ** k * N)
        return moments

    def _aggregation_moment_source(self, abscissas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """计算聚集过程的矩源项"""
        n = self.n_nodes
        w = np.maximum(weights, 0)
        x = np.maximum(abscissas, 1e-30)

        beta_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                beta_matrix[i, j] = self.aggregation_kernel.total_kernel(
                    np.array([x[i]]), np.array([x[j]])
                )[0]

        dmdt = np.zeros(self.n_moments)
        for k in range(self.n_moments):
            gain = 0.0
            loss = 0.0
            for i in range(n):
                for j in range(n):
                    gain += 0.5 * w[i] * w[j] * (x[i] + x[j]) ** k * beta_matrix[i, j]
                    loss += w[i] * w[j] * x[i] ** k * beta_matrix[i, j]
            dmdt[k] = gain - loss

        return dmdt

    def _breakage_moment_source(self, abscissas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        计算破裂过程的矩源项

        对于二进制等体积破裂:
        d(m_k)/dt = ∫ Γ(v) [2*(v/2)^k - v^k] n(v) dv
        """
        x = np.maximum(abscissas, 1e-30)
        w = np.maximum(weights, 0)

        gamma = self.breakage_kernel.breakage_rate(x)

        dmdt = np.zeros(self.n_moments)
        for k in range(self.n_moments):
            dmdt[k] = np.sum(gamma * w * (2.0 * (x / 2.0) ** k - x ** k))

        return dmdt

    def _ode_rhs(self, t: float, m: np.ndarray) -> np.ndarray:
        """ODE右端函数"""
        m_pos = np.maximum(m, 0)

        try:
            abscissas, weights = self.product_difference_algorithm(m_pos)
        except Exception:
            return np.zeros(self.n_moments)

        dmdt = np.zeros(self.n_moments)

        if self.enable_aggregation and self.aggregation_kernel is not None:
            dmdt += self._aggregation_moment_source(abscissas, weights)

        if self.enable_breakage and self.breakage_kernel is not None:
            dmdt += self._breakage_moment_source(abscissas, weights)

        return dmdt

    def solve(
        self,
        t_span: Tuple[float, float],
        n_times: int = 100,
        method: str = "LSODA",
        rtol: float = 1e-6,
        atol: float = 1e-30
    ) -> dict:
        """求解PBE的矩方程"""
        t_eval = np.linspace(t_span[0], t_span[1], n_times)

        sol = solve_ivp(
            self._ode_rhs,
            t_span,
            self.moments,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=(t_span[1] - t_span[0]) / 50.0
        )

        n_times = len(sol.t)
        distributions = np.zeros((self.n_nodes, n_times))
        abscissas_history = np.zeros((self.n_nodes, n_times))

        for i in range(n_times):
            m = np.maximum(sol.y[:, i], 0)
            try:
                abscissas, weights = self.product_difference_algorithm(m)
                distributions[:, i] = weights
                abscissas_history[:, i] = abscissas
            except Exception:
                distributions[:, i] = 0
                abscissas_history[:, i] = 0

        return {
            "t": sol.t,
            "moments": sol.y,
            "abscissas": abscissas_history,
            "weights": distributions,
            "success": sol.success,
            "message": sol.message
        }


# =============================================================================
# PBE 求解器 (蒙特卡洛方法)
# =============================================================================
class MonteCarloSolver:
    """
    蒙特卡洛方法 (Time-Driven Monte Carlo)

    基于随机采样模拟颗粒聚集和破裂过程。每个计算颗粒代表一定数量的
    真实颗粒，通过随机碰撞和破裂事件来模拟动力学。

    支持聚集和破裂过程。
    """

    def __init__(
        self,
        aggregation_kernel: Optional[CollisionKernel] = None,
        breakage_kernel: Optional[BreakageKernel] = None,
        volumes: np.ndarray = None,
        initial_N: np.ndarray = None,
        n_computational_particles: int = 10000,
        seed: int = 42,
        enable_aggregation: bool = True,
        enable_breakage: bool = True
    ):
        self.aggregation_kernel = aggregation_kernel
        self.breakage_kernel = breakage_kernel
        self.volumes = volumes
        self.n_volumes = len(volumes) if volumes is not None else 0
        self.n_comp_particles = n_computational_particles
        self.rng = np.random.RandomState(seed)
        self.enable_aggregation = enable_aggregation
        self.enable_breakage = enable_breakage

        if initial_N is not None and volumes is not None:
            self.total_real_particles = np.sum(initial_N)
            self.weight_per_comp = self.total_real_particles / n_computational_particles
            self.computational_particles = self._initialize_particles(initial_N)
        else:
            self.total_real_particles = 0
            self.weight_per_comp = 0
            self.computational_particles = np.array([])

    def _initialize_particles(self, initial_N: np.ndarray) -> np.ndarray:
        """初始化计算颗粒"""
        particles = np.zeros(self.n_comp_particles)
        probs = initial_N / np.sum(initial_N)

        n_assigned = 0
        for i in range(self.n_volumes):
            n_i = int(round(probs[i] * self.n_comp_particles))
            n_i = min(n_i, self.n_comp_particles - n_assigned)
            particles[n_assigned:n_assigned + n_i] = self.volumes[i]
            n_assigned += n_i

        while n_assigned < self.n_comp_particles:
            idx = self.rng.choice(self.n_volumes, p=probs)
            particles[n_assigned] = self.volumes[idx]
            n_assigned += 1

        return particles[:n_assigned]

    def _compute_collision_probability(self, v1: float, v2: float, dt: float) -> float:
        """计算两个颗粒的碰撞概率"""
        beta = self.aggregation_kernel.total_kernel(
            np.array([v1]), np.array([v2])
        )[0]
        return beta * self.weight_per_comp * dt

    def _compute_breakage_probability(self, v: float, dt: float) -> float:
        """计算颗粒的破裂概率"""
        gamma = self.breakage_kernel.breakage_rate(np.array([v]))[0]
        return 1.0 - np.exp(-gamma * dt)

    def solve(
        self,
        t_span: Tuple[float, float],
        n_times: int = 100,
        dt: Optional[float] = None
    ) -> dict:
        """
        蒙特卡洛模拟 (聚集+破裂)
        """
        if dt is None:
            dt = (t_span[1] - t_span[0]) / 1000.0

        t_eval = np.linspace(t_span[0], t_span[1], n_times)
        n_output = len(t_eval)

        N_history = np.zeros((self.n_volumes, n_output))
        N_history[:, 0] = self._compute_number_distribution()

        t = t_span[0]
        output_idx = 1

        while t < t_span[1] and len(self.computational_particles) > 1:
            particles = self.computational_particles
            n_particles = len(particles)

            if n_particles < 1:
                break

            # ===== 聚集过程 =====
            if self.enable_aggregation and self.aggregation_kernel is not None and n_particles >= 2:
                n_pairs = n_particles // 2
                if n_pairs > 0:
                    indices = self.rng.permutation(n_particles)[:2 * n_pairs].reshape(n_pairs, 2)
                    v1 = particles[indices[:, 0]]
                    v2 = particles[indices[:, 1]]

                    beta = self.aggregation_kernel.total_kernel(v1, v2)
                    probs = beta * self.weight_per_comp * dt
                    probs = np.minimum(probs, 1.0)

                    collision_mask = self.rng.random(n_pairs) < probs

                    if np.any(collision_mask):
                        collision_indices = indices[collision_mask]
                        particles[collision_indices[:, 0]] = v1[collision_mask] + v2[collision_mask]
                        particles[collision_indices[:, 1]] = 0

                    particles = particles[particles > 0]

            # ===== 破裂过程 =====
            if self.enable_breakage and self.breakage_kernel is not None and len(particles) > 0:
                gamma = self.breakage_kernel.breakage_rate(particles)
                breakage_probs = 1.0 - np.exp(-gamma * dt)
                breakage_mask = self.rng.random(len(particles)) < breakage_probs

                if np.any(breakage_mask):
                    breakage_v = particles[breakage_mask]
                    new_particles = []
                    for v in breakage_v:
                        v1, v2 = self.breakage_kernel.binary_breakage_volumes(v)
                        new_particles.extend([v1, v2])

                    particles = particles[~breakage_mask]
                    particles = np.concatenate([particles, np.array(new_particles)])

            self.computational_particles = particles

            t += dt

            if output_idx < n_output and t >= t_eval[output_idx]:
                N_history[:, output_idx] = self._compute_number_distribution()
                output_idx += 1

        for idx in range(output_idx, n_output):
            N_history[:, idx] = self._compute_number_distribution()

        return {
            "t": t_eval,
            "N": N_history,
            "n_particles": len(self.computational_particles),
            "success": True,
            "message": "Monte Carlo simulation completed"
        }

    def _compute_number_distribution(self) -> np.ndarray:
        """从计算颗粒重构数浓度分布"""
        N = np.zeros(self.n_volumes)

        for v in self.computational_particles:
            idx = np.searchsorted(self.volumes, v, side='right') - 1
            idx = np.clip(idx, 0, self.n_volumes - 1)
            N[idx] += 1

        return N * self.weight_per_comp


# =============================================================================
# 方法对比结果
# =============================================================================
@dataclass
class ComparisonResult:
    """多方法对比结果"""
    t_sectional: np.ndarray
    N_sectional: np.ndarray
    t_qmom: np.ndarray
    moments_qmom: np.ndarray
    abscissas_qmom: np.ndarray
    weights_qmom: np.ndarray
    t_mc: np.ndarray
    N_mc: np.ndarray
    discret: Discretization


# =============================================================================
# 动态平衡分析
# =============================================================================
class SteadyStateAnalyzer:
    """动态平衡分析器"""

    @staticmethod
    def check_steady_state(
        t: np.ndarray,
        moments: np.ndarray,
        window_size: int = 10,
        rtol: float = 1e-3
    ) -> Tuple[bool, float]:
        """
        检测是否达到动态平衡

        Parameters
        ----------
        moments : 矩序列 (n_moments, n_times)
        window_size : 用于检测平衡的时间窗口大小
        rtol : 相对变化率容限

        Returns
        -------
        is_steady : 是否达到稳态
        steady_time : 达到稳态的时间
        """
        if len(t) < 2 * window_size:
            return False, np.inf

        m0 = moments[0, :]

        for i in range(window_size, len(t)):
            if i - window_size >= 0:
                window_mean = np.mean(m0[i - window_size:i])
                rel_change = np.abs(m0[i] - window_mean) / np.maximum(np.abs(window_mean), 1e-30)
                if rel_change < rtol:
                    return True, t[i]

        return False, np.inf

    @staticmethod
    def compute_steady_state_distribution(
        N: np.ndarray,
        t: np.ndarray,
        steady_start_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算稳态分布的时间平均"""
        if steady_start_idx >= len(t) - 1:
            return N[:, -1], np.zeros_like(N[:, -1])

        steady_N = N[:, steady_start_idx:]
        mean_dist = np.mean(steady_N, axis=1)
        std_dist = np.std(steady_N, axis=1)

        return mean_dist, std_dist


# =============================================================================
# 结果分析
# =============================================================================
@dataclass
class FlocculationResult:
    """絮凝结果分析"""
    t: np.ndarray
    N: np.ndarray
    discret: Discretization

    @property
    def total_concentration(self) -> np.ndarray:
        return np.sum(self.N, axis=0)

    @property
    def total_volume_concentration(self) -> np.ndarray:
        V = self.discret.volumes
        return np.sum(V[:, np.newaxis] * self.N, axis=0)

    @property
    def number_mean_diameter(self) -> np.ndarray:
        d = self.discret.diameters
        N_total = self.total_concentration
        return np.sum(d[:, np.newaxis] * self.N, axis=0) / np.maximum(N_total, 1e-30)

    @property
    def volume_mean_diameter(self) -> np.ndarray:
        d = self.discret.diameters
        V = self.discret.volumes
        V_total = self.total_volume_concentration
        return np.sum(d[:, np.newaxis] * V[:, np.newaxis] * self.N, axis=0) / np.maximum(V_total, 1e-30)

    @property
    def polydispersity_index(self) -> np.ndarray:
        d = self.discret.diameters
        N_total = self.total_concentration
        d_mean = self.number_mean_diameter
        d2_mean = np.sum(d[:, np.newaxis] ** 2 * self.N, axis=0) / np.maximum(N_total, 1e-30)
        return d2_mean / np.maximum(d_mean ** 2, 1e-30)


# =============================================================================
# 可视化
# =============================================================================
class Visualizer:
    """絮凝过程可视化"""

    @staticmethod
    def plot_size_distribution(
        result: FlocculationResult,
        time_indices: Optional[list] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Axes:
        """绘制粒径分布随时间的演化"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        d = result.discret.diameters
        t = result.t
        N = result.N

        if time_indices is None:
            n_times = len(t)
            time_indices = [0, n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]

        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(time_indices)))

        for idx, color in zip(time_indices, colors):
            label = f"t = {t[idx]:.2e} s"
            ax.plot(d * 1e6, N[:, idx], color=color, label=label, linewidth=2)

        ax.set_xlabel("粒径 d [μm]", fontsize=12)
        ax.set_ylabel("数浓度 n(d) [m⁻³]", fontsize=12)
        ax.set_title("粒径分布随时间演化 (离散分组法)", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_steady_state_distribution(
        d: np.ndarray,
        N_steady_mean: np.ndarray,
        N_steady_std: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """绘制稳态粒径分布"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(d * 1e6, N_steady_mean, "k-", linewidth=2, label="稳态分布 (时间平均)")

        if N_steady_std is not None:
            ax.fill_between(
                d * 1e6,
                np.maximum(N_steady_mean - N_steady_std, 1),
                N_steady_mean + N_steady_std,
                alpha=0.3, color="gray", label="±1σ 波动范围"
            )

        ax.set_xlabel("粒径 d [μm]", fontsize=12)
        ax.set_ylabel("数浓度 n(d) [m⁻³]", fontsize=12)
        ax.set_title("动态平衡下的稳态粒径分布", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_approach_to_steady(
        t: np.ndarray,
        N_total: np.ndarray,
        d_mean: np.ndarray,
        steady_time: Optional[float] = None,
        ax1: Optional[plt.Axes] = None,
        ax2: Optional[plt.Axes] = None
    ) -> Tuple[plt.Axes, plt.Axes]:
        """绘制向稳态收敛的过程"""
        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(t, N_total, "b-", linewidth=2)
        if steady_time is not None and np.isfinite(steady_time):
            ax1.axvline(steady_time, color="r", linestyle="--", label=f"稳态建立: t={steady_time:.1f}s")
            ax1.legend(fontsize=10)
        ax1.set_xlabel("时间 [s]", fontsize=12)
        ax1.set_ylabel("总颗粒数浓度 [m⁻³]", fontsize=12)
        ax1.set_title("总浓度向稳态收敛", fontsize=14)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, d_mean * 1e6, "r-", linewidth=2)
        if steady_time is not None and np.isfinite(steady_time):
            ax2.axvline(steady_time, color="r", linestyle="--")
        ax2.set_xlabel("时间 [s]", fontsize=12)
        ax2.set_ylabel("数均粒径 [μm]", fontsize=12)
        ax2.set_title("平均粒径向稳态收敛", fontsize=14)
        ax2.set_xscale("log")
        ax2.grid(True, alpha=0.3)

        return ax1, ax2

    @staticmethod
    def plot_breakage_rate(
        breakage_kernel: BreakageKernel,
        v_range: Tuple[float, float] = (1e-27, 1e-18),
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """绘制破裂速率随粒径的变化"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        v_arr = np.logspace(np.log10(v_range[0]), np.log10(v_range[1]), 100)
        gamma = breakage_kernel.breakage_rate(v_arr)
        d_arr = (6.0 * v_arr / np.pi) ** (1.0 / 3.0) * 1e6

        ax.plot(d_arr, gamma, "k-", linewidth=2)
        ax.set_xlabel("粒径 d [μm]", fontsize=12)
        ax.set_ylabel("破裂速率 Γ(v) [s⁻¹]", fontsize=12)
        ax.set_title("破裂速率随粒径变化", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_method_comparison(
        comparison: ComparisonResult,
        time_idx: int = -1,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """绘制三种方法的粒径分布对比"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        d = comparison.discret.diameters

        ax.plot(d * 1e6, comparison.N_sectional[:, time_idx],
                "b-", linewidth=2, label="离散分组法 (存在数值弥散)", alpha=0.8)

        ax.plot(d * 1e6, comparison.N_mc[:, time_idx],
                "g-", linewidth=2, label="蒙特卡洛方法", alpha=0.6)

        if hasattr(comparison, 'abscissas_qmom'):
            d_qmom = (6.0 * comparison.abscissas_qmom[:, time_idx] / np.pi) ** (1.0 / 3.0)
            w_qmom = comparison.weights_qmom[:, time_idx]
            ax.scatter(d_qmom * 1e6, w_qmom, color="r", s=80,
                      marker="s", label="QMOM (求积节点)", zorder=5)

        ax.set_xlabel("粒径 d [μm]", fontsize=12)
        ax.set_ylabel("数浓度 n(d) [m⁻³]", fontsize=12)
        ax.set_title(f"数值方法对比 (t = {comparison.t_sectional[time_idx]:.2e} s)", fontsize=14)
        ax.legend(fontsize=10, loc="lower left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_moment_comparison(
        comparison: ComparisonResult,
        ax1: Optional[plt.Axes] = None,
        ax2: Optional[plt.Axes] = None
    ) -> Tuple[plt.Axes, plt.Axes]:
        """绘制三种方法的统计矩对比"""
        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 数均粒径对比
        d = comparison.discret.diameters
        V = comparison.discret.volumes

        # 离散分组法
        N_sec = comparison.N_sectional
        N_total_sec = np.sum(N_sec, axis=0)
        d_mean_sec = np.sum(d[:, np.newaxis] * N_sec, axis=0) / np.maximum(N_total_sec, 1e-30)

        # 蒙特卡洛
        N_mc = comparison.N_mc
        N_total_mc = np.sum(N_mc, axis=0)
        d_mean_mc = np.sum(d[:, np.newaxis] * N_mc, axis=0) / np.maximum(N_total_mc, 1e-30)

        # QMOM
        d_qmom = (6.0 * comparison.abscissas_qmom / np.pi) ** (1.0 / 3.0)
        w_qmom = comparison.weights_qmom
        N_total_qmom = np.sum(w_qmom, axis=0)
        d_mean_qmom = np.sum(d_qmom[:, np.newaxis] * w_qmom, axis=0) / np.maximum(N_total_qmom, 1e-30)

        ax1.plot(comparison.t_sectional, d_mean_sec * 1e6, "b-",
                 linewidth=2, label="离散分组法")
        ax1.plot(comparison.t_mc, d_mean_mc * 1e6, "g--",
                 linewidth=2, label="蒙特卡洛")
        ax1.plot(comparison.t_qmom, d_mean_qmom * 1e6, "r-.",
                 linewidth=2, label="QMOM")
        ax1.set_xlabel("时间 [s]", fontsize=12)
        ax1.set_ylabel("数均粒径 [μm]", fontsize=12)
        ax1.set_title("数均粒径对比", fontsize=14)
        ax1.legend(fontsize=10)
        ax1.set_xscale("log")
        ax1.grid(True, alpha=0.3)

        # 总浓度对比
        ax2.plot(comparison.t_sectional, N_total_sec, "b-",
                 linewidth=2, label="离散分组法")
        ax2.plot(comparison.t_mc, N_total_mc, "g--",
                 linewidth=2, label="蒙特卡洛")
        ax2.plot(comparison.t_qmom, N_total_qmom, "r-.",
                 linewidth=2, label="QMOM")
        ax2.set_xlabel("时间 [s]", fontsize=12)
        ax2.set_ylabel("总数浓度 [m⁻³]", fontsize=12)
        ax2.set_title("总颗粒数浓度对比", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        return ax1, ax2

    @staticmethod
    def plot_numerical_diffusion(
        comparison: ComparisonResult,
        time_idx: int = -1,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """绘制数值弥散效果的放大图"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        d = comparison.discret.diameters

        N_sec = comparison.N_sectional[:, time_idx]
        N_mc = comparison.N_mc[:, time_idx]

        idx_valid = (N_sec > 1e10) & (N_mc > 1e10)

        if np.any(idx_valid):
            ax.plot(d[idx_valid] * 1e6, N_sec[idx_valid], "bo-",
                    linewidth=2, markersize=4, label="离散分组法 (拖尾明显)")
            ax.plot(d[idx_valid] * 1e6, N_mc[idx_valid], "gs-",
                    linewidth=2, markersize=4, label="蒙特卡洛 (无虚假扩散)")

            ax.set_xlabel("粒径 d [μm]", fontsize=12)
            ax.set_ylabel("数浓度 n(d) [m⁻³]", fontsize=12)
            ax.set_title("数值弥散对比 (放大图)", fontsize=14)
            ax.legend(fontsize=10)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_moments(
        result: FlocculationResult,
        ax1: Optional[plt.Axes] = None,
        ax2: Optional[plt.Axes] = None
    ) -> Tuple[plt.Axes, plt.Axes]:
        """绘制统计矩随时间的变化"""
        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        t = result.t

        ax1.plot(t, result.number_mean_diameter * 1e6, "b-", linewidth=2, label="数均粒径")
        ax1.plot(t, result.volume_mean_diameter * 1e6, "r--", linewidth=2, label="体均粒径")
        ax1.set_xlabel("时间 [s]", fontsize=12)
        ax1.set_ylabel("平均粒径 [μm]", fontsize=12)
        ax1.set_title("平均粒径随时间变化", fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        ax2.plot(t, result.total_concentration, "k-", linewidth=2, label="总数浓度")
        ax2.set_xlabel("时间 [s]", fontsize=12)
        ax2.set_ylabel("总数浓度 [m⁻³]", fontsize=12)
        ax2.set_title("总颗粒数浓度随时间变化", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")
        ax2.set_yscale("log")

        return ax1, ax2

    @staticmethod
    def plot_kernel_comparison(
        kernel: CollisionKernel,
        v_fixed: float = 1e-24,
        v_range: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """绘制碰撞核随粒径的变化"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if v_range is None:
            v_range = (1e-27, 1e-21)

        v_arr = np.logspace(
            np.log10(v_range[0]),
            np.log10(v_range[1]),
            200
        )
        v1 = np.full_like(v_arr, v_fixed)

        beta_brownian = kernel.brownian_kernel(v1, v_arr)
        beta_shear = kernel.shear_kernel(v1, v_arr)
        beta_total = kernel.total_kernel(v1, v_arr)

        d_arr = (6.0 * v_arr / np.pi) ** (1.0 / 3.0) * 1e6

        ax.plot(d_arr, beta_brownian, "b--", linewidth=2, label="布朗运动核")
        ax.plot(d_arr, beta_shear, "r-.", linewidth=2, label="剪切流核")
        ax.plot(d_arr, beta_total, "k-", linewidth=2, label="总核")

        ax.set_xlabel("粒径 d [μm]", fontsize=12)
        ax.set_ylabel("碰撞核 β [m³/s]", fontsize=12)
        ax.set_title(f"碰撞核对比 (固定粒径={(6*v_fixed/np.pi)**(1/3)*1e6:.2f} μm)", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_polydispersity(result: FlocculationResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """绘制多分散性指数随时间变化"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        t = result.t
        ax.plot(t, result.polydispersity_index, "m-", linewidth=2)
        ax.set_xlabel("时间 [s]", fontsize=12)
        ax.set_ylabel("多分散性指数 PI [-]", fontsize=12)
        ax.set_title("多分散性指数随时间变化", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        return ax


# =============================================================================
# 主函数
# =============================================================================
def main():
    """运行完整的聚集-破裂平衡模拟"""
    print("=" * 70)
    print("  胶体颗粒聚集-破裂过程模拟 —— 动态平衡分析")
    print("=" * 70)

    # ---- 1. 物理参数 ----
    params = PhysicalParams(
        temperature=298.15,
        viscosity=0.001,
        shear_rate=50.0,
        enable_brownian=True,
        enable_shear_aggregation=True,
        enable_breakage=True,
        breakage_rate=0.05,
        critical_stress=0.5
    )
    print(f"\n[物理参数]")
    print(f"  温度 T = {params.temperature} K")
    print(f"  粘度 μ = {params.viscosity} Pa·s")
    print(f"  剪切速率 γ = {params.shear_rate} 1/s")
    print(f"  破裂速率系数 = {params.breakage_rate}")
    print(f"  临界应力 = {params.critical_stress} Pa")

    # ---- 2. 离散分组 ----
    discret = Discretization(
        v_min=1e-27,
        v_max=1e-18,
        n_sections=40,
        geometric_ratio=1.5
    )
    print(f"\n[离散分组]")
    print(f"  分组数: {discret.n_sections}")
    print(f"  粒径范围: {discret.diameters[0]*1e6:.4f} ~ {discret.diameters[-1]*1e6:.2f} um")

    # ---- 3. 核函数 ----
    agg_kernel = CollisionKernel(params)
    brk_kernel = BreakageKernel(params)
    print(f"\n[核函数已初始化]")

    # ---- 4. 初始分布 ----
    N0 = initial_distribution(
        discret,
        total_concentration=1e15,
        geometric_mean_volume=1e-24,
        geometric_std=1.3
    )
    print(f"\n[初始分布]")
    print(f"  总颗粒数浓度: {np.sum(N0):.3e} 1/m^3")
    print(f"  初始数均粒径: {np.sum(discret.diameters * N0) / np.sum(N0) * 1e6:.4f} um")

    # ---- 5. 时间设置 ----
    t_span = (0.0, 200.0)
    n_times = 100

    # =======================================================================
    # 离散分组法 (聚集+破裂)
    # =======================================================================
    print(f"\n{'─' * 70}")
    print("[方法] 离散分组法 (聚集+破裂)")
    print(f"{'─' * 70}")

    solver_sectional = PBESolver(
        discret,
        aggregation_kernel=agg_kernel,
        breakage_kernel=brk_kernel,
        initial_N=N0.copy(),
        enable_aggregation=True,
        enable_breakage=True
    )
    print("  开始求解 ...")

    result_sectional = solver_sectional.solve(
        t_span=t_span, n_times=n_times, method="LSODA"
    )

    if result_sectional["success"]:
        print(f"  [OK] 求解成功")
    else:
        print(f"  [WARN] 警告: {result_sectional['message']}")

    # =======================================================================
    # 动态平衡分析
    # =======================================================================
    print(f"\n{'─' * 70}")
    print("[动态平衡分析]")
    print(f"{'─' * 70}")

    d = discret.diameters
    N = result_sectional["N"]
    t = result_sectional["t"]

    moments_sec = np.zeros((6, n_times))
    for k in range(6):
        moments_sec[k, :] = np.sum(discret.volumes[:, np.newaxis] ** k * N, axis=0)

    is_steady, steady_time = SteadyStateAnalyzer.check_steady_state(
        t, moments_sec, window_size=15, rtol=5e-3
    )

    if is_steady:
        print(f"  [OK] 达到动态平衡，稳态建立时间: {steady_time:.1f} s")
        steady_idx = np.argmin(np.abs(t - steady_time))
        N_steady_mean, N_steady_std = SteadyStateAnalyzer.compute_steady_state_distribution(
            N, t, steady_idx
        )
    else:
        print("  [WARN] 未达到明显稳态，延长模拟时间可能需要")
        N_steady_mean = N[:, -1]
        N_steady_std = np.zeros_like(N[:, -1])
        steady_idx = n_times - 1

    # =======================================================================
    # 可视化
    # =======================================================================
    print(f"\n[生成可视化图表 ...]")
    viz = Visualizer()

    # 图1: 向稳态收敛过程
    N_total = np.sum(N, axis=0)
    d_mean = np.sum(d[:, np.newaxis] * N, axis=0) / np.maximum(N_total, 1e-30)

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
    viz.plot_approach_to_steady(
        t, N_total, d_mean, steady_time if is_steady else None, ax1=ax1a, ax2=ax1b
    )
    fig1.tight_layout()
    fig1.savefig("e:\\temp\\record10\\211\\approach_to_steady.png", dpi=150)
    print("  [OK] approach_to_steady.png 已保存")

    # 图2: 稳态粒径分布
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    viz.plot_steady_state_distribution(d, N_steady_mean, N_steady_std, ax=ax2)
    fig2.tight_layout()
    fig2.savefig("e:\\temp\\record10\\211\\steady_distribution.png", dpi=150)
    print("  [OK] steady_distribution.png 已保存")

    # 图3: 粒径分布演化过程
    result_sec = FlocculationResult(t=t, N=N, discret=discret)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    viz.plot_size_distribution(result_sec, ax=ax3)
    fig3.tight_layout()
    fig3.savefig("e:\\temp\\record10\\211\\distribution_evolution.png", dpi=150)
    print("  [OK] distribution_evolution.png 已保存")

    # 图4: 破裂速率曲线
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    viz.plot_breakage_rate(brk_kernel, v_range=(discret.v_min, discret.v_max), ax=ax4)
    fig4.tight_layout()
    fig4.savefig("e:\\temp\\record10\\211\\breakage_rate.png", dpi=150)
    print("  [OK] breakage_rate.png 已保存")

    # 图5: 聚集核对比
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    viz.plot_kernel_comparison(agg_kernel, v_fixed=1e-24, ax=ax5)
    fig5.tight_layout()
    fig5.savefig("e:\\temp\\record10\\211\\aggregation_kernel.png", dpi=150)
    print("  [OK] aggregation_kernel.png 已保存")

    plt.close("all")

    # =======================================================================
    # 结果摘要
    # =======================================================================
    print(f"\n{'=' * 70}")
    print("[结果摘要]")
    print(f"{'=' * 70}")
    print(f"\n  初始状态:")
    print(f"    总颗粒数浓度: {np.sum(N0):.4e} 1/m^3")
    print(f"    数均粒径: {np.sum(d * N0) / np.sum(N0) * 1e6:.4f} um")

    print(f"\n  稳态状态 (t ≈ {steady_time if is_steady else t[-1]:.1f} s):")
    print(f"    总颗粒数浓度: {np.sum(N_steady_mean):.4e} 1/m^3")
    print(f"    数均粒径: {np.sum(d * N_steady_mean) / np.sum(N_steady_mean) * 1e6:.4f} um")

    V0 = np.sum(discret.volumes * N0)
    V_steady = np.sum(discret.volumes * N_steady_mean)
    print(f"\n  体积守恒检查:")
    print(f"    初始总体积: {V0:.4e} m^3/m^3")
    print(f"    稳态总体积: {V_steady:.4e} m^3/m^3")
    print(f"    相对误差: {abs(V_steady - V0) / V0 * 100:.2f}%")

    print(f"\n{'=' * 70}")
    print("  模拟完成!")
    print(f"{'=' * 70}")
    print(f"\n[关键发现]")
    print(f"  聚集过程使颗粒数减少、粒径增大")
    print(f"  破裂过程使颗粒数增加、粒径减小")
    print(f"  当聚集速率与破裂速率平衡时，分布趋于稳态")
    print(f"  破裂速率随粒径增大而显著增加，限制了最大絮团尺寸")

    return result_sectional, N_steady_mean


if __name__ == "__main__":
    main()
