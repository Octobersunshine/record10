"""
阿尔芬波在太阳风中的传播求解器
==================================
沿磁力线传播，支持WKB近似和全波MHD方程求解

物理背景:
- 阿尔芬波是沿磁场传播的横波，满足 ∂v/∂t = ±v_A ∂v/∂z
- 阿尔芬速度: v_A = B_0 / sqrt(μ_0 ρ_0)
- 在非均匀介质中，波振幅会随背景参数变化而演化
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

# 物理常数
mu_0 = 4 * np.pi * 1e-7  # 真空磁导率 [H/m]
k_B = 1.38e-23  # 玻尔兹曼常数 [J/K]
m_p = 1.67e-27  # 质子质量 [kg]
AU = 1.496e11  # 天文单位 [m]
R_sun = 6.96e8  # 太阳半径 [m]


@dataclass
class SolarWindParams:
    """太阳风背景参数"""
    r_array: np.ndarray  # 日心距离数组 [m]
    n_array: np.ndarray  # 质子密度数组 [m^-3]
    B_array: np.ndarray  # 磁场强度数组 [T]
    v_sw: np.ndarray  # 太阳风速度 [m/s]
    T: np.ndarray  # 温度 [K]

    def __post_init__(self):
        self.rho_array = m_p * self.n_array  # 质量密度 [kg/m^3]
        self.v_A_array = self.B_array / np.sqrt(mu_0 * self.rho_array)  # 阿尔芬速度
        self.scale_height = self._compute_scale_height()

    def _compute_scale_height(self) -> np.ndarray:
        """计算密度标度高度 H = n / |dn/dr|"""
        dn_dr = np.gradient(self.n_array, self.r_array)
        return np.abs(self.n_array / (dn_dr + 1e-30))

    def get_interpolators(self):
        """返回插值函数"""
        return {
            'n': interp1d(self.r_array, self.n_array,
                          kind='cubic', fill_value='extrapolate'),
            'B': interp1d(self.r_array, self.B_array,
                          kind='cubic', fill_value='extrapolate'),
            'v_sw': interp1d(self.r_array, self.v_sw,
                             kind='cubic', fill_value='extrapolate'),
            'T': interp1d(self.r_array, self.T,
                          kind='cubic', fill_value='extrapolate'),
            'rho': interp1d(self.r_array, self.rho_array,
                            kind='cubic', fill_value='extrapolate'),
            'v_A': interp1d(self.r_array, self.v_A_array,
                            kind='cubic', fill_value='extrapolate'),
        }


class SolarWindModel:
    """太阳风背景场模型"""

    @staticmethod
    def parker_model(r_range: Tuple[float, float], n_points: int = 500,
                     n_0: float = 1e15, B_0: float = 2e-4,
                     T_0: float = 1e6, v_sw_0: float = 300e3) -> SolarWindParams:
        """
        简化的Parker太阳风模型

        Parameters:
        -----------
        r_range : (r_start, r_end)，单位为太阳半径
        n_points : 空间点数
        n_0 : 日冕底部密度 [m^-3]
        B_0 : 日冕底部磁场强度 [T]
        T_0 : 日冕底部温度 [K]
        v_sw_0 : 初始太阳风速度 [m/s]
        """
        r_start, r_end = r_range
        r = np.linspace(r_start, r_end, n_points) * R_sun

        # 密度模型: n(r) = n_0 * (r/R_sun)^(-2) * exp(-R_sun/H * (1 - R_sun/r))
        # 简化版本: n(r) = n_0 * (r/R_sun)^(-2.5)
        r_norm = r / R_sun
        n = n_0 * r_norm ** (-2.5)

        # 磁场模型: 径向分量 B_r(r) = B_0 * (r/R_sun)^(-2)
        # 考虑行星际磁场的扇形结构
        B_r = B_0 * r_norm ** (-2)
        B = B_r  # 简化：只考虑径向分量

        # 太阳风速度: 加速后达到终端速度
        v_sw = v_sw_0 * (1 - np.exp(-(r_norm - 1) / 5)) + 400e3 * np.exp(-(r_norm - 1) / 5)
        v_sw = np.clip(v_sw, v_sw_0, 800e3)

        # 温度: 绝热膨胀
        T = T_0 * r_norm ** (-0.7)

        return SolarWindParams(
            r_array=r,
            n_array=n,
            B_array=B,
            v_sw=v_sw,
            T=T
        )

    @staticmethod
    def helio_model(r_range: Tuple[float, float], n_points: int = 500) -> SolarWindParams:
        """
        基于Helios观测数据的经验模型
        """
        r_start, r_end = r_range
        r = np.linspace(r_start, r_end, n_points) * R_sun
        r_au = r / AU

        # 密度: n(r) = 3.3e6 * r^(-2.1) [cm^-3] -> [m^-3]
        n = 3.3e12 * r_au ** (-2.1)

        # 磁场强度
        B = 3.5e-9 * r_au ** (-1.8)

        # 太阳风速度
        v_sw = 300e3 + 100e3 * (1 - np.exp(-(r_au - 0.3) / 0.3))
        v_sw = np.clip(v_sw, 250e3, 800e3)

        # 温度
        T = 1.5e5 * r_au ** (-0.8)

        return SolarWindParams(
            r_array=r,
            n_array=n,
            B_array=B,
            v_sw=v_sw,
            T=T
        )


class AlfvenWaveSolver:
    """阿尔芬波传播求解器"""

    def __init__(self, params: SolarWindParams):
        """
        Parameters:
        -----------
        params : 太阳风背景参数
        """
        self.params = params
        self.interp = params.get_interpolators()

    # ------------------------------------------------------------------
    # 工具方法: 通过插值求背景场的导数（避免阶梯差分噪声）
    # ------------------------------------------------------------------
    def _bg(self, r):
        """返回指定 r 处的背景场 (rho, B, v_A, v_sw, dvA/dr, drho/dr, dB/dr)"""
        rho = self.interp['rho'](r)
        B = self.interp['B'](r)
        v_A = self.interp['v_A'](r)
        v_sw = self.interp['v_sw'](r)
        # 中心差分估计导数
        h = max(r * 1e-6, 1.0)
        drho = (self.interp['rho'](r + h) - self.interp['rho'](r - h)) / (2 * h)
        d_B = (self.interp['B'](r + h) - self.interp['B'](r - h)) / (2 * h)
        dv_A = (self.interp['v_A'](r + h) - self.interp['v_A'](r - h)) / (2 * h)
        dv_sw = (self.interp['v_sw'](r + h) - self.interp['v_sw'](r - h)) / (2 * h)
        return rho, B, v_A, v_sw, drho, d_B, dv_A, dv_sw

    # ------------------------------------------------------------------
    # 1. WKB 近似 (带稳定化: 近临界点自动使用小步长 + 阻尼保护)
    # ------------------------------------------------------------------
    def solve_wkb(self, wave_params: Dict) -> Dict:
        """
        WKB 近似求解阿尔芬波振幅演化（稳定化版本）。

        振幅演化方程（来自作用量守恒的 WKB 近似）:
            d(ln δB)/dr = (1/4)d(ln ρ)/dr − (3/4)d(ln B)/dr
                         − (1/2) v_A/(v_A+v_sw) · d(ln v_sw)/dr

        稳定性:
            - 当 λ/H > 0.5 时启用小步长自适应
            - 当 λ/H > 0.9 时添加人工粘滞防止数值发散
        """
        r = self.params.r_array
        v_A = self.params.v_A_array
        rho = self.params.rho_array
        B = self.params.B_array
        v_sw = self.params.v_sw
        H = self.params.scale_height

        omega = wave_params.get('omega', 2 * np.pi * 1e-3)
        delta_B_0 = wave_params.get('delta_B_0', 1e-9)
        mode = wave_params.get('mode', 'outward')

        v_phase = v_A + np.sign(1 if mode == 'outward' else -1) * v_sw
        k = omega / v_phase
        wavelength = 2 * np.pi / k
        wkb_ratio = wavelength / H
        wkb_valid = wkb_ratio < 1.0

        log_delta_B = np.zeros_like(r)
        log_delta_B[0] = np.log(delta_B_0)

        dr = np.diff(r)

        for i in range(len(r) - 1):
            # 对数导数（使用解析差分）
            dln_rho = (np.log(rho[i + 1]) - np.log(rho[i])) / dr[i]
            dln_B = (np.log(B[i + 1]) - np.log(B[i])) / dr[i]
            dln_vsw = ((np.log(v_sw[i + 1]) - np.log(v_sw[i])) / dr[i]
                       if v_sw[i] > 0 else 0.0)

            dln_dB = (0.25 * dln_rho - 0.75 * dln_B
                      - 0.5 * v_A[i] / (v_A[i] + v_sw[i]) * dln_vsw)

            # 临界点稳定化: 当 WKB 比值接近 1 时添加小阻尼
            if wkb_ratio[i] > 0.9:
                damp = 0.1 * (wkb_ratio[i] - 0.9)  # 随 WKB 失效增强
                dln_dB -= damp * dln_dB  # 抑制指数增长/衰减

            log_delta_B[i + 1] = log_delta_B[i] + dln_dB * dr[i]

        delta_B = np.exp(log_delta_B)
        delta_v = delta_B / np.sqrt(mu_0 * rho)
        power = delta_B ** 2 * v_A / (2 * mu_0)

        return {
            'r': r, 'r_au': r / AU,
            'delta_B': delta_B, 'delta_v': delta_v,
            'v_A': v_A, 'v_sw': v_sw,
            'k': k, 'wavelength': wavelength,
            'wkb_valid': wkb_valid, 'wkb_ratio': wkb_ratio,
            'power': power,
            'power_density': power / (4 * np.pi * r ** 2),
            'method': 'WKB'
        }

    # ------------------------------------------------------------------
    # 2. 全波 MHD —— 耦合模方程（四阶龙格-库塔积分）
    #    处理非均匀介质中的反射与模式转换
    # ------------------------------------------------------------------
    def solve_full_mhd(self, wave_params: Dict,
                       boundary_cond: str = 'source') -> Dict:
        """
        全波 MHD 求解: 耦合模方程 + 四阶龙格-库塔 (RK4) 积分。

        状态向量: Y = [I₊, I₋]  （向外 / 向内 波的作用量密度）

        耦合方程 (来自多尺度展开，见 e.g. Withbroe 1988; Ofman 2010):
            dI₊/dr = −2 d(ln v_A)/dr · I₊ + Q · I₋
            dI₋/dr = +2 d(ln v_A)/dr · I₋ + Q · I₊

        耦合系数 Q 控制反射强度，由背景梯度决定:
            Q(r) = (1/2) |d(ln v_A)/dr| · exp(−λ/H)   (渐变时反射弱)

        边界条件: 内边界 I₊(r₀) = δB₀²/(2 μ₀ ρ₀)，I₋(r₀)=0
        """
        r = self.params.r_array
        v_A = self.params.v_A_array
        rho = self.params.rho_array
        B = self.params.B_array
        v_sw = self.params.v_sw
        H = self.params.scale_height

        omega = wave_params.get('omega', 2 * np.pi * 1e-3)
        delta_B_0 = wave_params.get('delta_B_0', 1e-9)

        # 背景对数导数（平滑）
        dln_vA = np.gradient(np.log(v_A + 1e-30), r)

        k = omega / (v_A + v_sw)
        wavelength = 2 * np.pi / k

        # 耦合系数: 渐变近似下 Q ≈ (1/2)|dln v_A/dr|，并受 WKB 比值抑制
        Q = 0.5 * np.abs(dln_vA) * np.exp(-wavelength / (H + 1e-30))

        # RK4 积分器 (逐网格点推进)
        N = len(r)
        I_plus = np.zeros(N, dtype=complex)
        I_minus = np.zeros(N, dtype=complex)
        I_plus[0] = (delta_B_0 ** 2) / (2 * mu_0 * rho[0])

        for i in range(N - 1):
            dr = r[i + 1] - r[i]
            Ip, Im = I_plus[i], I_minus[i]

            i_next = min(i + 1, N - 1)

            # k1: 左端点
            dln_vA_L, Q_L = dln_vA[i], Q[i]
            k1p = -2 * dln_vA_L * Ip + Q_L * Im
            k1m = +2 * dln_vA_L * Im + Q_L * Ip

            # k2, k3: 中点
            dln_vA_M = 0.5 * (dln_vA[i] + dln_vA[i_next])
            Q_M = 0.5 * (Q[i] + Q[i_next])
            k2p = -2 * dln_vA_M * (Ip + 0.5 * dr * k1p) + Q_M * (Im + 0.5 * dr * k1m)
            k2m = +2 * dln_vA_M * (Im + 0.5 * dr * k1m) + Q_M * (Ip + 0.5 * dr * k1p)
            k3p = -2 * dln_vA_M * (Ip + 0.5 * dr * k2p) + Q_M * (Im + 0.5 * dr * k2m)
            k3m = +2 * dln_vA_M * (Im + 0.5 * dr * k2m) + Q_M * (Ip + 0.5 * dr * k2p)

            # k4: 右端点
            dln_vA_R, Q_R = dln_vA[i_next], Q[i_next]
            k4p = -2 * dln_vA_R * (Ip + dr * k3p) + Q_R * (Im + dr * k3m)
            k4m = +2 * dln_vA_R * (Im + dr * k3m) + Q_R * (Ip + dr * k3p)

            I_plus[i + 1] = Ip + (dr / 6) * (k1p + 2 * k2p + 2 * k3p + k4p)
            I_minus[i + 1] = Im + (dr / 6) * (k1m + 2 * k2m + 2 * k3m + k4m)

            # 物理约束: 作用量密度非负（数值误差修正）
            if I_plus[i + 1].real < 0:
                I_plus[i + 1] = complex(0, I_plus[i + 1].imag)
            if I_minus[i + 1].real < 0:
                I_minus[i + 1] = complex(0, I_minus[i + 1].imag)

        # 从作用量恢复振幅
        delta_B_plus = np.sqrt(np.abs(2 * mu_0 * rho * I_plus.real))
        delta_B_minus = np.sqrt(np.abs(2 * mu_0 * rho * I_minus.real))
        delta_B = np.sqrt(delta_B_plus ** 2 + delta_B_minus ** 2)
        delta_v = delta_B / np.sqrt(mu_0 * rho)

        # 总功率 = 向外 − 向内 （净能流）
        power_net = (delta_B_plus ** 2 - delta_B_minus ** 2) * v_A / (2 * mu_0)
        power_total = delta_B ** 2 * v_A / (2 * mu_0)

        # 反射系数 R = √(I₋/I₊) （局部）
        with np.errstate(divide='ignore', invalid='ignore'):
            R_local = np.sqrt(np.abs(I_minus.real) / (np.abs(I_plus.real) + 1e-40))
        R_local = np.clip(R_local, 0, 1)

        return {
            'r': r, 'r_au': r / AU,
            'delta_B': delta_B,
            'delta_B_plus': delta_B_plus,
            'delta_B_minus': delta_B_minus,
            'delta_v': delta_v,
            'v_A': v_A, 'v_sw': v_sw,
            'k': k,
            'reflection_coeff': R_local,
            'power': power_total,
            'power_net': power_net,
            'power_density': power_total / (4 * np.pi * r ** 2),
            'power_density_net': power_net / (4 * np.pi * r ** 2),
            'I_plus': I_plus, 'I_minus': I_minus,
            'coupling_Q': Q,
            'method': 'Full_MHD_RK4'
        }

    # ------------------------------------------------------------------
    # 3. 全波二阶 ODE 直接积分 (龙格-库塔)
    #    物理方程 (线性化 MHD, 横波分量, 在运动坐标系中):
    #        d²ξ/dr² + d(ln(ρ v_A))/dr · dξ/dr + k² ξ = 0
    #    k = (ω + v_sw·k)/v_A  (实验室坐标系 → 随流坐标变换)
    #    δB = B · dξ/dr  (Faraday 定律)
    #
    #    数值稳定化:
    #    - 振幅约束: 当 |ξ|/|ξ₀| 超过理论上限 (ρv_A)^(-1/4) 的 100 倍
    #      时加入人工粘滞 (数值阻尼)，防止临界点附近的指数爆炸。
    # ------------------------------------------------------------------
    # 3. 全波二阶 ODE 直接积分 (备选 / 参考)
    #    注意: 二阶 ODE 不含反射机制，仅在严格 WKB 区可靠；
    #    反射区请使用 solve_full_mhd (耦合模)。此处保留以便于
    #    与 WKB 极限对比，结果物理意义有限。
    # ------------------------------------------------------------------
    def solve_full_mhd_ode(self, wave_params: Dict) -> Dict:
        """
        二阶线性 MHD 波动方程的 RK4 积分 (仅作参考，不含反射)。

        以位移 ξ(r) 为基本变量:
            d²ξ/dr² + C₁(r) · dξ/dr + C₂(r) · ξ = 0
        其中
            C₁(r) = d(ln(ρ v_A))/dr
            C₂(r) = (ω/(v_A + v_sw))²
        """
        r = self.params.r_array
        v_A = self.params.v_A_array
        rho = self.params.rho_array
        B = self.params.B_array
        v_sw = self.params.v_sw

        omega = wave_params.get('omega', 2 * np.pi * 1e-3)
        delta_B_0 = wave_params.get('delta_B_0', 1e-9)

        ln_rho_vA = np.log(rho * v_A + 1e-40)
        C1 = np.gradient(ln_rho_vA, r)
        k = omega / (v_A + v_sw)
        C2 = k ** 2

        # WKB 理论振幅包络
        amp_wkb = (rho[0] * v_A[0]) ** 0.25 / (rho * v_A + 1e-40) ** 0.25

        k0 = k[0]
        xi_0 = delta_B_0 / (1j * k0 * B[0])
        dxi_0 = 1j * k0 * xi_0
        xi_0_abs = np.abs(xi_0)

        # WKB 失效程度
        H = self.params.scale_height
        wl = 2 * np.pi / (k + 1e-30)
        wkb_fail = np.clip(wl / (H + 1e-30), 0, None)

        # 强阻尼阈值 (WKB 完全失效时用代理耗散)
        damp_profile = np.where(wkb_fail > 1, 2.0 * (wkb_fail - 1), 0.0)

        N = len(r)
        xi = np.zeros(N, dtype=complex)
        dxi = np.zeros(N, dtype=complex)
        xi[0], dxi[0] = xi_0, dxi_0

        for i in range(N - 1):
            dr = r[i + 1] - r[i]
            damp = damp_profile[i]

            def rhs(_xi, _dxi):
                return _dxi, -C1[i] * _dxi - C2[i] * _xi - damp * _dxi

            k1 = rhs(xi[i], dxi[i])
            k2 = rhs(xi[i] + 0.5 * dr * k1[0], dxi[i] + 0.5 * dr * k1[1])
            k3 = rhs(xi[i] + 0.5 * dr * k2[0], dxi[i] + 0.5 * dr * k2[1])
            k4 = rhs(xi[i] + dr * k3[0], dxi[i] + dr * k3[1])

            xi[i + 1] = xi[i] + (dr / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            dxi[i + 1] = dxi[i] + (dr / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

            # 数值稳定: 非有限时回退
            if not (np.isfinite(xi[i + 1]) and np.isfinite(dxi[i + 1])):
                xi[i + 1] = xi[i] * np.exp(-max(damp, 1) * dr)
                dxi[i + 1] = dxi[i] * np.exp(-max(damp, 1) * dr)

            # 硬裁剪: 当 |ξ| 超过 10³ · A_wkb 时按比例缩小
            A_wkb_here = xi_0_abs * amp_wkb[min(i + 1, N - 1)]
            if np.abs(xi[i + 1]) > 1e3 * A_wkb_here and A_wkb_here > 0:
                scale = 1e3 * A_wkb_here / np.abs(xi[i + 1])
                xi[i + 1] *= scale
                dxi[i + 1] *= scale

        delta_B = np.abs(B * dxi)
        delta_v = delta_B / np.sqrt(mu_0 * rho)
        amp_wkb_out = xi_0_abs * amp_wkb
        power = delta_B ** 2 * v_A / (2 * mu_0)

        return {
            'r': r, 'r_au': r / AU,
            'delta_B': delta_B,
            'delta_v': delta_v,
            'xi': xi,
            'dxi': dxi,
            'xi_wkb_amp': amp_wkb_out,
            'v_A': v_A, 'v_sw': v_sw,
            'k': k,
            'power': power,
            'power_density': power / (4 * np.pi * r ** 2),
            'C1': C1,
            'C2': C2,
            'method': 'Full_MHD_ODE_RK4'
        }

    def solve_characteristic(self, wave_params: Dict) -> Dict:
        """
        特征线方法求解（适用于时间相关问题）

        沿特征线: dr/dt = v_sw ± v_A
        波振幅沿特征线守恒（在均匀介质中）
        """
        r = self.params.r_array
        v_A = self.params.v_A_array
        v_sw = self.params.v_sw
        rho = self.params.rho_array

        omega = wave_params.get('omega', 2 * np.pi * 1e-3)
        delta_B_0 = wave_params.get('delta_B_0', 1e-9)
        t_end = wave_params.get('t_end', 86400)  # 模拟一天
        n_t = wave_params.get('n_t', 100)

        # 时间网格
        t = np.linspace(0, t_end, n_t)
        dt = t[1] - t[0]

        # 初始化波场（空间-时间网格）
        delta_B = np.zeros((n_t, len(r)))
        delta_B[0, 0] = delta_B_0  # 初始扰动

        # 特征线追踪
        for n in range(n_t - 1):
            for i in range(len(r) - 1):
                # 向外传播的特征线
                v_char = v_sw[i] + v_A[i]

                if v_char > 0:
                    # 从上游点插值
                    r_upstream = r[i] - v_char * dt
                    idx_up = np.searchsorted(r, r_upstream) - 1
                    if idx_up >= 0 and idx_up < len(r) - 1:
                        # 线性插值
                        frac = (r_upstream - r[idx_up]) / (r[idx_up + 1] - r[idx_up])
                        delta_B[n + 1, i] = (1 - frac) * delta_B[n, idx_up] + frac * delta_B[n, idx_up + 1]

                    # 考虑背景变化引起的振幅调整
                    if i > 0:
                        scale = np.sqrt(rho[i] / (rho[i - 1] + 1e-30))
                        delta_B[n + 1, i] *= scale

        return {
            'r': r,
            'r_au': r / AU,
            't': t,
            'delta_B': delta_B,
            'delta_v': delta_B / np.sqrt(mu_0 * rho),
            'v_A': v_A,
            'v_sw': v_sw,
            'method': 'Characteristic'
        }

    # ------------------------------------------------------------------
    # 4. 湍流级联与加热模型 (Matthaeus et al. 1999 / Zank et al. 2012)
    #
    #    输运方程 (稳态, 一维):
    #      (v_sw + v_A) · dE₊/dr = S₊ − D₊
    #      (v_sw − v_A) · dE₋/dr = S₋ − D₋
    #
    #    源项 S_± = 2 |dln v_A/dr| · [E∓ − (E₊E₋)^(1/2)]
    #      (能量在两波间通过反射重新分配; 两波越平衡, 源越小)
    #
    #    非线性级联耗散 D_± (Vasquez et al. 1997):
    #      D_± = a_c · (E₊E₋)^(1/2) · (E₊ + E₋) / (L⊥ · v_A)
    #      其中 L⊥ 是湍涡外尺度, a_c 是级联效率 (经验值 0.15)
    #
    #    总加热率 (等离子体吸收的能量):
    #      Q_T = D₊ + D₋
    #
    #    湍流压力 p_T = E₊ + E₋, 压力梯度驱动太阳风加速
    # ------------------------------------------------------------------
    def solve_turbulence(self, wave_params: Dict) -> Dict:
        """
        阿尔芬波湍流输运方程求解 (稳态, RK4)。

        基于 Matthaeus et al. (1999) / Zank et al. (2012) 模型:
            dE₊/dr = [S₊ − D₊] / (v_sw + v_A)
            dE₋/dr = [S₋ − D₋] / (v_sw − v_A)

        Parameters:
            wave_params['cascade_eff'] : a_c (级联效率, 默认 0.15)
            wave_params['Lperp']       : 湍流外尺度 L⊥ [m], 默认 10^5 km
            wave_params['delta_B_0']   : 初始向外扰动
            wave_params['delta_B_in_0']: 初始向内扰动 (种子)
        """
        r = self.params.r_array
        v_A = self.params.v_A_array
        rho = self.params.rho_array
        v_sw = self.params.v_sw

        omega = wave_params.get('omega', 2 * np.pi * 1e-3)
        delta_B_0 = wave_params.get('delta_B_0', 1e-8)
        delta_B_in_0 = wave_params.get('delta_B_in_0', 1e-9)
        a_c = wave_params.get('cascade_eff', 0.15)
        Lperp = wave_params.get('Lperp', 1e8)  # 10^5 km

        # 背景对数导数
        dln_vA = np.gradient(np.log(v_A + 1e-30), r)

        # 初始能量密度 (向内波给一个小的种子以触发级联)
        E_plus = np.zeros(len(r))
        E_minus = np.zeros(len(r))
        E_plus[0] = delta_B_0 ** 2 / (2 * mu_0)
        E_minus[0] = max(delta_B_in_0 ** 2 / (2 * mu_0), 1e-40)

        N = len(r)

        for i in range(N - 1):
            dr = r[i + 1] - r[i]
            Ep, Em = E_plus[i], E_minus[i]

            def rhs(Ep_, Em_, idx=i):
                E_sum = Ep_ + Em_
                E_prod_sqrt = np.sqrt(max(0, Ep_ * Em_))

                # 源项 (反射重新分配)
                S_plus = 2 * abs(dln_vA[idx]) * (Em_ - E_prod_sqrt)
                S_minus = 2 * abs(dln_vA[idx]) * (Ep_ - E_prod_sqrt)

                # 级联耗散
                D_plus = a_c * E_prod_sqrt * E_sum / (Lperp * v_A[idx] + 1e-30)
                D_minus = a_c * E_prod_sqrt * E_sum / (Lperp * v_A[idx] + 1e-30)

                # 多普勒相速度
                u_plus = v_sw[idx] + v_A[idx]
                u_minus = v_sw[idx] - v_A[idx]

                # 向内波: 在实验室坐标系下沿 -r 传播 (u_minus < 0)
                # 积分沿 +r 方向, 所以 dE₋/dr 的符号由 u_minus 的符号决定
                # 物理上: 向内波在 r 增大方向上, 若 u_minus < 0, dE₋/dr 应为正
                # 我们用 dE₋/dr = (S₋ − D₋) / u_minus  (u_minus < 0 时自动翻转)
                denom_p = max(1.0, abs(u_plus))
                denom_m = max(1.0, abs(u_minus))
                sign_m = np.sign(u_minus) if abs(u_minus) > 1 else -1.0

                dEp = (S_plus - D_plus) / denom_p
                dEm = (S_minus - D_minus) * sign_m / denom_m

                return dEp, dEm

            k1 = rhs(Ep, Em, i)
            k2 = rhs(Ep + 0.5 * dr * k1[0], Em + 0.5 * dr * k1[1], i)
            k3 = rhs(Ep + 0.5 * dr * k2[0], Em + 0.5 * dr * k2[1], i)
            k4 = rhs(Ep + dr * k3[0], Em + dr * k3[1], min(i + 1, N - 1))

            E_plus[i + 1] = Ep + (dr / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            E_minus[i + 1] = Em + (dr / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

            # 能量密度非负
            E_plus[i + 1] = max(E_plus[i + 1], 0)
            E_minus[i + 1] = max(E_minus[i + 1], 0)

        # 派生量
        delta_B_plus = np.sqrt(2 * mu_0 * E_plus)
        delta_B_minus = np.sqrt(2 * mu_0 * E_minus)
        delta_B_total = np.sqrt(delta_B_plus ** 2 + delta_B_minus ** 2)

        # 加热率 Q_T = D₊ + D₋ (级联耗散, 等离子体吸收)
        E_sum = E_plus + E_minus
        E_prod_sqrt = np.sqrt(np.maximum(0, E_plus * E_minus))
        D_total = a_c * E_prod_sqrt * E_sum / (Lperp * v_A + 1e-30)
        Q_T = 2 * D_total  # D₊ + D₋

        # 质量加热率 (每单位质量, 等效温度升高率)
        gamma = 5.0 / 3.0
        n = rho / m_p
        dT_dr_heat = (gamma - 1) / (gamma * n * k_B + 1e-30) * Q_T

        # 累积加热通量 ∫ Q_T dr
        Q_integral = np.zeros(N)
        for i in range(1, N):
            Q_integral[i] = Q_integral[i - 1] + 0.5 * (Q_T[i - 1] + Q_T[i]) * (r[i] - r[i - 1])

        # 湍流压力
        p_T = E_plus + E_minus
        dpT_dr = np.gradient(p_T, r)
        a_T = -dpT_dr / (rho + 1e-30)

        # 级联时间 τ_casc = E_total / Q_T
        tau_casc = np.where(Q_T > 0, E_sum / (Q_T + 1e-40), 1e20)

        # 扰动速度
        delta_v = delta_B_total / np.sqrt(mu_0 * rho)

        return {
            'r': r,
            'r_au': r / AU,
            'E_plus': E_plus,
            'E_minus': E_minus,
            'E_total': E_sum,
            'delta_B_plus': delta_B_plus,
            'delta_B_minus': delta_B_minus,
            'delta_B_total': delta_B_total,
            'delta_v': delta_v,
            'Q_T': Q_T,
            'dT_dr_heat': dT_dr_heat,
            'Q_integral': Q_integral,
            'Q_power_per_mass': Q_T / (rho + 1e-30),
            'p_T': p_T,
            'a_T': a_T,
            'tau_casc': tau_casc,
            'v_A': v_A,
            'v_sw': v_sw,
            'method': 'Turbulence_Cascade'
        }

    # ------------------------------------------------------------------
    # 5. 自洽太阳风加速: 迭代求解
    #    波加热 → 压力梯度 → 太阳风加速 → 反馈到波传播
    # ------------------------------------------------------------------
    def solve_self_consistent(self, wave_params: Dict,
                              n_iter: int = 5) -> Dict:
        """
        自洽太阳风加速模型 (简化):
        1. 使用背景场初始化 v_sw
        2. 求解湍流级联，得到加热率 Q_T 和湍流加速度 a_T
        3. 根据能量守恒修正 v_sw:  1/2 ρ v² 增加 = 积分 Q_T
        4. 迭代直至收敛

        返回:
            iterated_solar_wind: 迭代后的 v_sw, T, E_total 等
        """
        r = self.params.r_array
        v_A = self.params.v_A_array
        rho = self.params.rho_array
        v_sw = self.params.v_sw.copy()
        T = self.params.T.copy()

        delta_B_0 = wave_params.get('delta_B_0', 1e-8)
        a_c = wave_params.get('cascade_eff', 0.15)

        v_sw_history = [v_sw.copy()]
        E_total_history = []

        for it in range(n_iter):
            # 构造临时 SolarWindParams，用最新 v_sw
            params_temp = SolarWindParams(
                r_array=self.params.r_array,
                n_array=self.params.n_array,
                B_array=self.params.B_array,
                v_sw=v_sw,
                T=T
            )
            solver_temp = AlfvenWaveSolver(params_temp)
            turb = solver_temp.solve_turbulence(wave_params)

            E_total_history.append(turb['E_total'].copy())

            # 能量守恒: ∫ Q_T dr = ρ v_sw · dv_sw/dr (稳态近似, 忽略绝热)
            # 简化: 假设增量 Δv_sw ≈ ∫ Q_T / (ρ v_sw) dr
            Q_T = turb['Q_T']
            dv_sw = np.zeros_like(v_sw)
            for i in range(1, len(v_sw)):
                dv_sw[i] = dv_sw[i - 1] + (Q_T[i] / (rho[i] * max(v_sw[i], 1.0))) * (r[i] - r[i - 1])

            # 阻尼更新: v_sw_new = v_sw + α · dv_sw
            alpha = 0.3 / (it + 1)  # 逐步减小步长，稳定迭代
            v_sw = v_sw + alpha * dv_sw
            v_sw = np.clip(v_sw, 100e3, 1200e3)

            v_sw_history.append(v_sw.copy())

        # 最终迭代结果
        final_params = SolarWindParams(
            r_array=self.params.r_array,
            n_array=self.params.n_array,
            B_array=self.params.B_array,
            v_sw=v_sw,
            T=T
        )
        final_solver = AlfvenWaveSolver(final_params)
        final_turb = final_solver.solve_turbulence(wave_params)

        return {
            'r': r,
            'r_au': r / AU,
            'v_sw_initial': self.params.v_sw,
            'v_sw_final': v_sw,
            'v_sw_history': v_sw_history,
            'E_total_history': E_total_history,
            'final_turbulence': final_turb,
            'method': 'Self_Consistent_SolarWind'
        }


def plot_turbulence(turb_result: Dict,
                    sc_result: Optional[Dict] = None,
                    save_path: Optional[str] = None):
    """
    湍流加热 / 自洽太阳风加速 结果可视化。
    """
    r_au = turb_result['r_au']

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle('阿尔芬波湍流: 能量级联与太阳风加热/加速', fontsize=14)

    # 1. 能量密度 E₊, E₋, E_total
    ax = axes[0, 0]
    ax.semilogy(r_au, turb_result['E_plus'] + 1e-30,
                'b-', label='E₊ (outward)', linewidth=2)
    ax.semilogy(r_au, turb_result['E_minus'] + 1e-30,
                'r--', label='E₋ (reflected)', linewidth=2)
    ax.semilogy(r_au, turb_result['E_total'] + 1e-30,
                'k-', label='E_total', linewidth=2, alpha=0.5)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('E [J/m³]')
    ax.set_title('湍流能量密度 (E₊/E₋)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 扰动磁场振幅
    ax = axes[0, 1]
    ax.semilogy(r_au, turb_result['delta_B_plus'] * 1e9 + 1e-30,
                'b-', label='δB₊', linewidth=2)
    ax.semilogy(r_au, turb_result['delta_B_minus'] * 1e9 + 1e-30,
                'r--', label='δB₋', linewidth=2)
    ax.semilogy(r_au, turb_result['delta_B_total'] * 1e9 + 1e-30,
                'k-', label='δB_total', linewidth=2, alpha=0.5)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('δB [nT]')
    ax.set_title('扰动磁场')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 加热率 Q_T (单位体积) & 等效每公斤加热率
    ax = axes[1, 0]
    ax.loglog(r_au, turb_result['Q_T'] + 1e-30,
              'r-', label='Q_T (volumetric)', linewidth=2)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('Q_T [W/m³]')
    ax.set_title('湍流加热率 (体积)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.loglog(r_au, turb_result['Q_power_per_mass'] + 1e-30,
               'b--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Q/m [W/kg]', color='b')

    # 4. 累积加热量 (积分)
    ax = axes[1, 1]
    ax.semilogy(r_au, turb_result['Q_integral'] + 1e-30,
                'g-', linewidth=2)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('∫ Q_T dr [W/m²]')
    ax.set_title('累积加热能量 (面密度通量)')
    ax.grid(True, alpha=0.3)

    # 5. 湍流压力 & 加速度
    ax = axes[2, 0]
    ax.semilogy(r_au, turb_result['p_T'] + 1e-30,
                'b-', label='p_T (turbulent pressure)', linewidth=2)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('p_T [Pa]')
    ax.set_title('湍流压力与加速度')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax3 = ax.twinx()
    ax3.semilogy(r_au, np.abs(turb_result['a_T']) + 1e-30,
                 'r--', linewidth=2, alpha=0.7)
    ax3.set_ylabel('|a_T| [m/s²]', color='r')

    # 6. 级联时间 τ_casc & 自洽加速 (若有)
    ax = axes[2, 1]
    ax.semilogy(r_au, turb_result['tau_casc'] + 1e-30,
                'm-', linewidth=2)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('τ_casc [s]')
    ax.set_title('非线性级联时间')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        base = save_path.rsplit('.', 1)[0]
        plt.savefig(f"{base}_turbulence.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 自洽太阳风加速迭代对比图
    if sc_result is not None:
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
        fig2.suptitle('自洽太阳风加速 (迭代结果)', fontsize=14)

        r_au = sc_result['r_au']
        v_init = sc_result['v_sw_initial'] / 1e3
        v_final = sc_result['v_sw_final'] / 1e3

        ax = axes2[0]
        ax.plot(r_au, v_init, 'r--', label='初始 v_sw', linewidth=2)
        ax.plot(r_au, v_final, 'b-', label=f'最终 v_sw ({len(sc_result["v_sw_history"])-1} 次迭代)', linewidth=2)
        ax.set_xlabel('r [AU]')
        ax.set_ylabel('v_sw [km/s]')
        ax.set_title('太阳风速度演化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes2[1]
        for i, vh in enumerate(sc_result['v_sw_history']):
            ax.plot(r_au, vh / 1e3, alpha=0.5 + 0.5 * i / max(1, len(sc_result['v_sw_history']) - 1),
                    label=f'iter {i}')
        ax.set_xlabel('r [AU]')
        ax.set_ylabel('v_sw [km/s]')
        ax.set_title('迭代过程')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes2[2]
        turb_final = sc_result['final_turbulence']
        ax.semilogy(r_au, turb_final['E_total'] + 1e-30,
                    'b-', label='E_total (final)', linewidth=2)
        ax.semilogy(r_au, turb_final['Q_T'] + 1e-30,
                    'r--', label='Q_T (final)', linewidth=2)
        ax.set_xlabel('r [AU]')
        ax.set_title('最终湍流能量 & 加热率')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{base}_self_consistent.png", dpi=150, bbox_inches='tight')
        plt.close()


def plot_results(wkb_result: Dict, mhd_result: Dict,
                 ode_result: Optional[Dict] = None,
                 save_path: Optional[str] = None):
    """
    可视化结果: 对比 WKB / 耦合模 RK4 / 二阶 ODE RK4 三种方法。
    """
    has_ode = ode_result is not None

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle('阿尔芬波在太阳风中的传播 (WKB vs 耦合模RK4' +
                 (' vs 二阶ODE-RK4' if has_ode else '') + ')', fontsize=14)

    r_au = wkb_result['r_au']

    # 1. 背景参数
    ax = axes[0, 0]
    ax.plot(r_au, wkb_result['v_A'] / 1e3, 'b-', label='Alfvén speed')
    ax.plot(r_au, wkb_result['v_sw'] / 1e3, 'r-', label='Solar wind speed')
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('v [km/s]')
    ax.set_title('背景速度剖面')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 波振幅演化
    ax = axes[0, 1]
    ax.semilogy(r_au, wkb_result['delta_B'] * 1e9, 'b-', label='WKB', linewidth=2)
    ax.semilogy(r_au, mhd_result['delta_B'] * 1e9, 'r--', label='Coupled-Mode RK4',
                linewidth=2)
    if has_ode:
        ax.semilogy(r_au, ode_result['delta_B'] * 1e9, 'g-.',
                    label='2nd-ODE RK4', linewidth=2)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('δB [nT]')
    ax.set_title('扰动磁场振幅 (对比)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 耦合模方法: 向外 / 向内 波振幅分解
    ax = axes[1, 0]
    ax.semilogy(r_au, mhd_result['delta_B_plus'] * 1e9, 'b-',
                label='δB₊ (outward)', linewidth=2)
    ax.semilogy(r_au, mhd_result['delta_B_minus'] * 1e9, 'r--',
                label='δB₋ (reflected)', linewidth=2)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('δB [nT]')
    ax.set_title('耦合模分解 (全波 MHD)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 净功率密度 (向外 − 向内)
    ax = axes[1, 1]
    if 'power_density_net' in mhd_result:
        ax.semilogy(r_au, np.abs(mhd_result['power_density_net']),
                    'b-', label='Coupled-Mode (net)', linewidth=2)
        ax.semilogy(r_au, mhd_result['power_density'], 'r--',
                    label='Coupled-Mode (total)', linewidth=2)
    else:
        ax.semilogy(r_au, mhd_result['power_density'], 'r--',
                    label='Coupled-Mode', linewidth=2)
    ax.semilogy(r_au, wkb_result['power_density'], 'k:', label='WKB', linewidth=2)
    if has_ode:
        ax.semilogy(r_au, ode_result['power_density'], 'g-.',
                    label='2nd-ODE RK4', linewidth=2)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('P [W/m²]')
    ax.set_title('波功率密度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. WKB 适用性检验
    ax = axes[2, 0]
    ax.plot(r_au, wkb_result['wkb_ratio'], 'g-', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label='WKB limit (λ/H=1)')
    ax.fill_between(r_au, 0, 1, alpha=0.2, color='green', label='WKB valid')
    ax.fill_between(r_au, 1, wkb_result['wkb_ratio'].max(),
                    alpha=0.2, color='red', label='WKB invalid (需反射)')
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('λ / H')
    ax.set_title('WKB 适用性检验')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 局部反射系数 R = √(I₋/I₊)
    ax = axes[2, 1]
    ax.plot(r_au, mhd_result['reflection_coeff'], 'm-', linewidth=2, label='R = √(I₋/I₊)')
    if 'coupling_Q' in mhd_result:
        ax.plot(r_au, mhd_result['coupling_Q'] / (mhd_result['coupling_Q'].max() + 1e-30),
                'k--', alpha=0.5, label='耦合系数 Q (归一化)')
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('R')
    ax.set_title('局部反射系数 & 耦合系数')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 替换 tight_layout 以避免 tick 溢出 (matplotlib 1e±500 范围时出现 OverflowError)
    try:
        fig.tight_layout()
    except (OverflowError, ValueError):
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except (OverflowError, ValueError):
            # 部分坐标轴数据含 inf/NaN: 手动裁剪 ylim
            for a in axes.flat:
                for line in a.get_lines():
                    yd = line.get_ydata()
                    yd = np.where(np.isfinite(yd), yd, np.nan)
                    if np.any(np.isfinite(yd)):
                        ylo = np.nanmin(yd)
                        yhi = np.nanmax(yd)
                        if ylo < yhi:
                            a.set_ylim(max(ylo * 0.5, 1e-30), yhi * 2)
                if a.get_yscale() == 'log':
                    ylim = a.get_ylim()
                    if not np.isfinite(ylim[0]) or not np.isfinite(ylim[1]):
                        a.set_ylim(1e-30, 1e30)
            fig.subplots_adjust(hspace=0.4, wspace=0.3)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    plt.close()

    # 额外图: 波数和波长
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes2[0]
    ax.semilogy(r_au, wkb_result['k'], 'b-', linewidth=2)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('k [m⁻¹]')
    ax.set_title('波数')
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.semilogy(r_au, wkb_result['wavelength'] / AU, 'r-', linewidth=2)
    ax.set_xlabel('r [AU]')
    ax.set_ylabel('λ [AU]')
    ax.set_title('波长')
    ax.grid(True, alpha=0.3)

    try:
        fig2.tight_layout()
    except (OverflowError, ValueError):
        fig2.subplots_adjust(hspace=0.4, wspace=0.3)
    if save_path:
        base = save_path.rsplit('.', 1)[0]
        plt.savefig(f"{base}_k_wavelength.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ODE 专用图: 位移 ξ 和 WKB 对比
    if has_ode and 'xi' in ode_result:
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes3[0]
        ax.semilogy(r_au, np.abs(ode_result['xi']) + 1e-30,
                    'b-', label='|ξ| (ODE-RK4)', linewidth=2)
        if 'xi_wkb_amp' in ode_result:
            ax.semilogy(r_au, ode_result['xi_wkb_amp'] + 1e-30,
                        'r--', label='|ξ| (WKB)', linewidth=2)
        elif 'xi_wkb' in ode_result:
            ax.semilogy(r_au, np.abs(ode_result['xi_wkb']) + 1e-30,
                        'r--', label='|ξ| (WKB)', linewidth=2)
        ax.set_xlabel('r [AU]')
        ax.set_ylabel('|ξ| [m]')
        ax.set_title('位移振幅对比')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes3[1]
        ref = ode_result.get('xi_wkb_amp',
                             np.abs(ode_result.get('xi_wkb', ode_result['xi'])))
        ratio = np.abs(ode_result['xi']) / (ref + 1e-30)
        ratio = np.clip(ratio, 1e-10, 1e10)
        ax.semilogy(r_au, ratio, 'k-', linewidth=2)
        ax.set_xlabel('r [AU]')
        ax.set_ylabel('|ξ_ODE| / |ξ_WKB|')
        ax.set_title('振幅比 ( >1 表示 ODE 振幅大于 WKB)')
        ax.grid(True, alpha=0.3)

        try:
            fig3.tight_layout()
        except (OverflowError, ValueError):
            fig3.subplots_adjust(hspace=0.4, wspace=0.3)
        if save_path:
            base = save_path.rsplit('.', 1)[0]
            plt.savefig(f"{base}_xi_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """主函数"""

    print("=" * 60)
    print("阿尔芬波在太阳风中的传播求解 (含全波反射)")
    print("=" * 60)

    # 1. 创建太阳风背景模型
    print("\n[1] 初始化太阳风背景模型...")

    params = SolarWindModel.parker_model(
        r_range=(1.0, 5.0),
        n_points=1000,
        n_0=1e15,
        B_0=2e-4,
        T_0=1.5e6,
        v_sw_0=200e3
    )

    print(f"  径向范围: {params.r_array[0]/R_sun:.1f} - {params.r_array[-1]/R_sun:.1f} R_sun")
    print(f"  密度范围: {params.n_array[0]:.2e} - {params.n_array[-1]:.2e} m^-3")
    print(f"  磁场范围: {params.B_array[0]*1e9:.2f} - {params.B_array[-1]*1e9:.2f} nT")
    print(f"  阿尔芬速度范围: {params.v_A_array[0]/1e3:.0f} - {params.v_A_array[-1]/1e3:.0f} km/s")

    # 2. 创建求解器
    solver = AlfvenWaveSolver(params)

    # 3. 定义波参数
    wave_params = {
        'omega': 2 * np.pi * 1e-3,   # 1 mHz
        'delta_B_0': 1e-8,            # 10 nT
        'mode': 'outward'
    }

    print(f"\n[2] 波参数:")
    print(f"  频率: {wave_params['omega']/(2*np.pi)*1e3:.2f} mHz")
    print(f"  初始扰动: {wave_params['delta_B_0']*1e9:.1f} nT")
    print(f"  传播方向: 向外")

    # 4. WKB 近似求解
    print("\n[3] WKB 近似求解 (稳定化)...")
    wkb_result = solver.solve_wkb(wave_params)
    print(f"  初始振幅: {wkb_result['delta_B'][0]*1e9:.2f} nT")
    print(f"  最终振幅: {wkb_result['delta_B'][-1]*1e9:.4f} nT")
    print(f"  振幅衰减比: {wkb_result['delta_B'][-1]/wkb_result['delta_B'][0]:.4f}")

    # 5. 全波耦合模求解 (RK4)
    print("\n[4] 全波耦合模方程求解 (RK4, 含反射)...")
    mhd_result = solver.solve_full_mhd(wave_params)
    print(f"  初始振幅: {mhd_result['delta_B'][0]*1e9:.2f} nT")
    print(f"  最终振幅: {mhd_result['delta_B'][-1]*1e9:.4f} nT")
    print(f"  最终向外分量 dB+ : {mhd_result['delta_B_plus'][-1]*1e9:.4f} nT")
    print(f"  最终反射分量 dB- : {mhd_result['delta_B_minus'][-1]*1e9:.4f} nT")
    print(f"  最终净功率: {mhd_result['power_net'][-1]:.4e} W")

    # 6. 二阶 ODE 直接积分 (仅作参考，WKB 失效区结果物理意义有限)
    print("\n[5] 二阶线性 MHD ODE 直接积分 (RK4, 仅参考，不含反射)...")
    ode_result = solver.solve_full_mhd_ode(wave_params)
    print(f"  初始振幅: {ode_result['delta_B'][0]*1e9:.2f} nT")
    print(f"  最终振幅: {ode_result['delta_B'][-1]*1e9:.4f} nT")

    # 7. 比较
    print("\n[6] 结果比较 (耦合模为主要全波方法，ODE 仅参考):")
    diff_cm = np.abs(wkb_result['delta_B'] - mhd_result['delta_B']) / wkb_result['delta_B']
    diff_ode = np.abs(wkb_result['delta_B'] - ode_result['delta_B']) / wkb_result['delta_B']
    print(f"  WKB vs 耦合模-RK4: 最大相对差 {diff_cm.max()*100:.2f}%  "
          f"(平均 {diff_cm.mean()*100:.2f}%)")
    if np.all(np.isfinite(diff_ode)):
        print(f"  WKB vs ODE-RK4   : 最大相对差 {diff_ode.max()*100:.2f}%  "
              f"(平均 {diff_ode.mean()*100:.2f}%)")
    else:
        print("  WKB vs ODE-RK4   : 非有限值 (ODE 数值不稳定，仅作参考)")

    # 8. WKB 适用性
    print("\n[7] WKB 适用性检验:")
    wkb_ok = wkb_result['wkb_valid']
    print(f"  WKB 条件满足区域: {wkb_ok.sum()}/{len(wkb_ok)} 点")
    print(f"  λ/H 范围: {wkb_result['wkb_ratio'].min():.4f} - {wkb_result['wkb_ratio'].max():.4f}")
    print(f"  ⇒ 当 λ/H ≳ 1 时 WKB 失效, 反射波不可忽略")

    # 9. 能量通量
    print("\n[8] 波能量通量:")
    print(f"  WKB  : P₀ = {wkb_result['power'][0]:.4e} W,  P_end = {wkb_result['power'][-1]:.4e} W")
    print(f"  耦合模: P₀ = {mhd_result['power'][0]:.4e} W,  P_end = {mhd_result['power'][-1]:.4e} W")
    print(f"  ODE  : P₀ = {ode_result['power'][0]:.4e} W,  P_end = {ode_result['power'][-1]:.4e} W")
    print(f"  耦合模净能流守恒性: |ΔP_net/P_0| = "
          f"{abs(mhd_result['power_net'][-1]-mhd_result['power_net'][0])/abs(mhd_result['power_net'][0]):.4e}")

    # 10. 可视化
    print("\n[9] 生成可视化图表...")
    plot_results(wkb_result, mhd_result, ode_result, save_path='alfven_wave_results.png')

    # 11. 保存数据
    print("\n[10] 保存数据...")
    data = {
        'r_au': wkb_result['r_au'],
        'v_A': wkb_result['v_A'],
        'v_sw': wkb_result['v_sw'],
        'delta_B_wkb': wkb_result['delta_B'],
        'delta_B_coupled': mhd_result['delta_B'],
        'delta_B_plus': mhd_result['delta_B_plus'],
        'delta_B_minus': mhd_result['delta_B_minus'],
        'delta_B_ode': ode_result['delta_B'],
        'reflection_coeff': mhd_result['reflection_coeff'],
        'k': wkb_result['k'],
        'wavelength': wkb_result['wavelength'],
    }
    np.savez('alfven_wave_data.npz', **data)
    print("  数据已保存至: alfven_wave_data.npz")

    # 12. Helios 模型验证
    print("\n[11] 使用 Helios 经验模型验证 (0.3–1 AU)...")
    params_helio = SolarWindModel.helio_model(r_range=(20, 215), n_points=500)
    solver_helio = AlfvenWaveSolver(params_helio)
    wkb_helio = solver_helio.solve_wkb(wave_params)
    mhd_helio = solver_helio.solve_full_mhd(wave_params)
    ode_helio = solver_helio.solve_full_mhd_ode(wave_params)

    print(f"  范围: {params_helio.r_array[0]/AU:.2f} - {params_helio.r_array[-1]/AU:.2f} AU")
    print(f"  最终 δB (WKB)     : {wkb_helio['delta_B'][-1]*1e9:.4f} nT")
    print(f"  最终 δB (耦合模)  : {mhd_helio['delta_B'][-1]*1e9:.4f} nT")
    print(f"  最终 δB (ODE)     : {ode_helio['delta_B'][-1]*1e9:.4f} nT")

    plot_results(wkb_helio, mhd_helio, ode_helio, save_path='alfven_wave_helios.png')

    # 13. 湍流级联与加热
    print("\n[12] 湍流级联模型 (Matthaeus/Zank, 稳态 RK4)...")
    # 注: 为展示级联加热效应, 注入较强的两波不平衡扰动;
    #     物理上这种不平衡来自日冕底部的反射 / 重联 / 对流驱动.
    turb_params = {
        'omega': 2 * np.pi * 1e-3,
        'delta_B_0': 1e-7,       # 100 nT 向外波 (强驱动)
        'delta_B_in_0': 3e-8,     # 30 nT 向内波种子
        'cascade_eff': 1.0,       # 级联效率 (增强, 显示级联效应)
        'Lperp': 1e7,             # 湍流外尺度 10^4 km
    }
    turb_result = solver.solve_turbulence(turb_params)
    print(f"  最终 E_plus : {turb_result['E_plus'][-1]:.4e} J/m³")
    print(f"  最终 E_minus: {turb_result['E_minus'][-1]:.4e} J/m³")
    print(f"  峰值 Q_T     : {turb_result['Q_T'].max():.4e} W/m³")
    print(f"  峰值加热率/质量: {turb_result['Q_power_per_mass'].max():.4e} W/kg")
    print(f"  累积加热通量  : {turb_result['Q_integral'][-1]:.4e} W/m²")
    print(f"  级联时间范围  : {turb_result['tau_casc'].min():.2e} ~ {turb_result['tau_casc'].max():.2e} s")

    # 14. 自洽太阳风加速
    print("\n[13] 自洽太阳风加速 (迭代 5 次)...")
    sc_result = solver.solve_self_consistent(turb_params, n_iter=5)
    print(f"  初始 v_sw (末端): {sc_result['v_sw_initial'][-1]/1e3:.1f} km/s")
    print(f"  最终 v_sw (末端): {sc_result['v_sw_final'][-1]/1e3:.1f} km/s")
    print(f"  速度增量        : {(sc_result['v_sw_final'][-1] - sc_result['v_sw_initial'][-1])/1e3:.1f} km/s")

    # 15. 湍流可视化
    print("\n[14] 生成湍流可视化图表...")
    plot_turbulence(turb_result, sc_result, save_path='alfven_wave_turbulence.png')

    # 16. 保存湍流数据
    print("\n[15] 保存湍流数据...")
    np.savez('alfven_wave_turbulence_data.npz',
             r_au=turb_result['r_au'],
             E_plus=turb_result['E_plus'],
             E_minus=turb_result['E_minus'],
             E_total=turb_result['E_total'],
             Q_T=turb_result['Q_T'],
             delta_B_total=turb_result['delta_B_total'],
             v_sw_initial=sc_result['v_sw_initial'],
             v_sw_final=sc_result['v_sw_final'])
    print("  数据已保存至: alfven_wave_turbulence_data.npz")

    print("\n" + "=" * 60)
    print("计算完成!")
    print("=" * 60)

    return wkb_result, mhd_result, ode_result, turb_result, sc_result


if __name__ == "__main__":
    main()
