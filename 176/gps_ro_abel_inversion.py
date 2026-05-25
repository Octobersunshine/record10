"""
GPS掩星观测数据反演 —— 中性大气 + 电离层
============================================================

物理背景：
    GPS掩星（Radio Occultation, RO）利用低轨卫星（LEO）接收GPS卫星信号，
    当信号穿过大气层时发生折射，通过测量信号的弯曲角 α(a)（随碰撞参数a变化），
    可反演出大气折射率剖面 N(h) 和电离层电子密度剖面 Ne(h)。

核心公式（Abel变换）：
    碰撞参数  a = r * n(r)        (r: 地心距离, n: 折射率)
    反演公式：
        ln(n(r)) = (1/π) ∫_r^∞ α(a) / √(a² - r²)  da

    折射率 N 与 n 的关系：N = (n - 1) × 10⁶

电离层反演：
    双频GPS观测可以分离电离层延迟：
        L1 (1575.42 MHz) 和 L2 (1227.60 MHz) 信号的延迟差异
        电离层折射率：N_iono = -40.3 * Ne / f²
        TEC (总电子含量)：∫ Ne dh

本脚本实现：
    中性大气部分：
        1. 模拟数据生成（指数大气模型）
        2. Abel积分正演（由折射率剖面计算弯曲角）
        3. Abel积分反演（由弯曲角反演折射率剖面）
        4. 1DVar变分同化分离干/湿项
    电离层部分：
        5. 电离层电子密度剖面模拟（Chapman函数）
        6. 双频电离层延迟计算
        7. Abel积分反演电子密度剖面
        8. TEC计算与空间天气监测指标
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 第 1 部分：物理常数与模拟参数
# ============================================================
R_EARTH = 6371.0            # 地球平均半径 (km)
H_TOP = 60.0                 # 反演顶层高度 (km)
H_BOTTOM = 0.0               # 近地面高度 (km)
N_POINTS = 400               # 高度采样点数

# 模拟大气参数（指数模型）
H_SCALE = 7.5                # 标高 (km)
N_SURFACE = 350.0            # 地面折射率 N 单位 (N-units)


# ============================================================
# 第 2 部分：模拟数据生成
# ============================================================
def generate_refractivity_profile(h_km):
    """
    指数大气模型：生成折射率 N(h) 剖面。
    实际应用中此部分由观测数据或模型替代。

    参数:
        h_km: 高度数组 (km)
    返回:
        N: 折射率 N (N-units)
    """
    return N_SURFACE * np.exp(-h_km / H_SCALE)


def n_from_N(N):
    """折射率 N (N-units) 转换为折射指数 n"""
    return 1.0 + N * 1e-6


def forward_bending_angle(r_array, n_array):
    """
    Abel积分正演：由折射指数剖面 n(r) 计算弯曲角 α(a)。

    物理公式:
        α(a) = (1/π) ∫_a^∞ [d(ln(n))/dr] / √(r² - a²) * r * dr

    其中碰撞参数 a = r * n(r)，在正演中对每个 r 计算对应的 a。

    参数:
        r_array: 地心距离数组 (km)
        n_array: 折射指数数组
    返回:
        a_array: 碰撞参数 (km)
        alpha_array: 弯曲角 (弧度)
    """
    r_km = r_array
    n_val = n_array

    # 计算 d(ln(n))/dr
    ln_n = np.log(n_val)
    d_ln_n_dr = np.gradient(ln_n, r_km)

    # 碰撞参数 a = r * n(r)
    a_array = r_km * n_val

    # 对每个碰撞参数 a，通过数值积分计算弯曲角 α(a)
    alpha_array = np.zeros_like(a_array)
    a_top = a_array[-1]

    for i, a_val in enumerate(a_array):
        # 仅积分 r >= a_val 的区域
        mask = r_km >= a_val * 0.999999
        r_integrand = r_km[mask]
        dlnn = d_ln_n_dr[mask]

        if len(r_integrand) < 3:
            alpha_array[i] = 0.0
            continue

        # 被积函数: d(ln n)/dr * r / √(r² - a²)
        denominator = np.sqrt(np.maximum(r_integrand**2 - a_val**2, 1e-20))
        integrand = dlnn * r_integrand / denominator

        # 梯形积分
        alpha_array[i] = np.trapz(integrand, r_integrand)

    return a_array, alpha_array


# ============================================================
# 第 3 部分：Abel积分反演
# ============================================================
def invert_bending_angle(a_array, alpha_array, r_output):
    """
    Abel积分反演：由弯曲角 α(a) 反演折射指数剖面 n(r)。

    反演公式:
        ln(n(r)) = (1/π) ∫_r^∞ α(a) / √(a² - r²)  da

    参数:
        a_array:   碰撞参数数组 (km), 需单调递增
        alpha_array: 弯曲角数组 (弧度)
        r_output:  输出的地心距离数组 (km)
    返回:
        n_output:  反演的折射指数数组
        N_output:  反演的折射率 (N-units)
    """
    n_output = np.ones_like(r_output)
    a_top = a_array[-1]

    for i, r_val in enumerate(r_output):
        if r_val >= a_top - 1e-6:
            n_output[i] = 1.0
            continue

        # 仅积分 a >= r_val 的区域
        mask = a_array >= r_val
        a_integ = a_array[mask]
        alpha_integ = alpha_array[mask]

        if len(a_integ) < 3:
            n_output[i] = 1.0
            continue

        # 被积函数: α(a) / √(a² - r²)
        denominator = np.sqrt(np.maximum(a_integ**2 - r_val**2, 1e-20))
        integrand = alpha_integ / denominator

        # 梯形积分
        integral = np.trapz(integrand, a_integ)
        ln_n = integral / np.pi

        n_output[i] = np.exp(ln_n)

    N_output = (n_output - 1.0) * 1e6

    return n_output, N_output


# ============================================================
# 第 4 部分：高精度反演（使用样条插值 + 自适应积分）
# ============================================================
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.linalg import inv, diag

# ============================================================
# 第 4.5 部分：物理常数与温湿关系
# ============================================================
R_DRY = 287.05                # 干空气气体常数 (J/(kg·K))
R_VAPOR = 461.52              # 水汽气体常数 (J/(kg·K))
EPSILON = R_DRY / R_VAPOR     # 水汽与干空气气体常数比 ≈ 0.622
KAPPA = R_DRY / 1004.67       # R_d / Cp ≈ 0.286
GRAVITY = 9.80665             # 重力加速度 (m/s²)


def refractivity_from_tpwv(T, P, q):
    """
    Smith-Weintraub 公式：由温度、气压和比湿计算折射率。

    N = N_dry + N_wet
    N_dry = 77.6 * P / T
    N_wet = 3.73e5 * (P * q / (EPSILON + q)) / T²

    参数:
        T: 温度 (K)
        P: 气压 (hPa)
        q: 比湿 (kg/kg)
    返回:
        N: 折射率 (N-units)
    """
    e = P * q / (EPSILON + q)   # 水汽压 (hPa)
    N_dry = 77.6 * P / T
    N_wet = 3.73e5 * e / T**2
    return N_dry + N_wet, N_dry, N_wet


def compute_pressure_profile(P_sfc, T_profile, h_profile):
    """
    静力平衡积分：由地面气压和温度剖面计算气压剖面。

    dP/dz = -ρ * g = -P * g / (R_d * T_v)

    参数:
        P_sfc: 地面气压 (hPa)
        T_profile: 温度剖面 (K)
        h_profile: 高度剖面 (m)
    返回:
        P_profile: 气压剖面 (hPa)
    """
    n_levels = len(h_profile)
    P_profile = np.zeros(n_levels)
    P_profile[0] = P_sfc

    for i in range(1, n_levels):
        dh = h_profile[i] - h_profile[i-1]
        T_mid = 0.5 * (T_profile[i] + T_profile[i-1])
        P_profile[i] = P_profile[i-1] * np.exp(-GRAVITY * dh / (R_DRY * T_mid))

    return P_profile


# ============================================================
# 第 4.6 部分：背景场模拟（预报模式廓线）
# ============================================================
def generate_background_profiles(h_km):
    """
    生成模拟的背景场温湿廓线（类似ECMWF预报模式）。

    使用简化的标准大气 + 湿层结构：
    - 温度：对流层递减率 ~6.5 K/km，平流层 ~2 K/km
    - 比湿：近地面饱和，随高度指数衰减

    参数:
        h_km: 高度数组 (km)
    返回:
        T_b: 背景温度 (K)
        q_b: 背景比湿 (kg/kg)
        P_b: 背景气压 (hPa)
    """
    h_m = h_km * 1000.0

    # 温度剖面（标准大气 + 对流层顶逆温）
    T_sfc = 288.15               # 地面温度 (K)
    T_tropo = 216.65             # 对流层顶温度 (K)
    h_tropo = 11.0               # 对流层顶高度 (km)

    T_b = np.zeros_like(h_km)
    for i, h in enumerate(h_km):
        if h <= h_tropo:
            T_b[i] = T_sfc - 6.5 * h
        else:
            T_b[i] = T_tropo + 2.0 * (h - h_tropo)

    # 气压剖面（静力平衡）
    P_sfc = 1013.25              # 地面标准气压 (hPa)
    P_b = compute_pressure_profile(P_sfc, T_b, h_m)

    # 比湿剖面（近地面饱和，随高度衰减）
    q_sfc = 0.015                # 地面比湿 15 g/kg
    H_q = 3.0                    # 水汽标高 (km)

    # 对流层内有一湿层结构
    q_b = q_sfc * np.exp(-h_km / H_q)
    # 在 1-3km 增加一个边界层湿层
    pbl_factor = 1.0 + 0.5 * np.exp(-((h_km - 2.0) / 1.5)**2)
    q_b = q_b * pbl_factor
    q_b = np.clip(q_b, 0, 0.03)  # 限制最大比湿

    return T_b, q_b, P_b


# ============================================================
# 第 4.7 部分：1DVar 变分同化 —— 分离干/湿折射率
# ============================================================
def one_dimensional_variational(N_obs, h_km, T_b, q_b, P_b,
                                obs_error=0.5, bg_error_T=2.0, bg_error_q=0.002):
    """
    一维变分同化（1DVar）：结合观测和背景场分离温湿贡献。

    目标函数:
        J(x) = (x - x_b)^T B^{-1} (x - x_b) + (Hx - y)^T R^{-1} (Hx - y)

    其中:
        x = [T, q]  状态向量（温度、比湿）
        y = N_obs   观测（Abel反演的折射率）
        H: 观测算子（Smith-Weintraub公式）
        B: 背景误差协方差
        R: 观测误差协方差

    参数:
        N_obs: 观测折射率剖面 (N-units)
        h_km: 高度 (km)
        T_b: 背景温度 (K)
        q_b: 背景比湿 (kg/kg)
        P_b: 背景气压 (hPa)
        obs_error: 观测折射率误差 (N-units)
        bg_error_T: 背景温度误差 (K)
        bg_error_q: 背景比湿误差 (kg/kg)
    返回:
        T_ana: 分析温度 (K)
        q_ana: 分析比湿 (kg/kg)
        N_ana: 分析折射率 (N-units)
        N_dry_ana: 分析干项折射率 (N-units)
        N_wet_ana: 分析湿项折射率 (N-units)
    """
    n_levels = len(h_km)

    # 状态向量: x = [T, q] (每个高度两个变量)
    x_b = np.concatenate([T_b, q_b])

    # 背景误差协方差 B (对角矩阵)
    sigma_T = bg_error_T * np.ones(n_levels)
    sigma_q = bg_error_q * np.ones(n_levels)
    sigma_x = np.concatenate([sigma_T, sigma_q])
    B_inv = diag(1.0 / sigma_x**2)

    # 观测误差协方差 R
    R_inv = diag(1.0 / (obs_error * np.ones(n_levels))**2)

    def cost_function(x):
        T = x[:n_levels]
        q = x[n_levels:]

        # 计算折射率
        N_calc, _, _ = refractivity_from_tpwv(T, P_b, q)

        # 背景项
        dx = x - x_b
        J_b = 0.5 * np.dot(dx, B_inv @ dx)

        # 观测项
        dy = N_calc - N_obs
        J_o = 0.5 * np.dot(dy, R_inv @ dy)

        return J_b + J_o

    def gradient(x):
        T = x[:n_levels]
        q = x[n_levels:]

        N_calc, _, _ = refractivity_from_tpwv(T, P_b, q)
        dy = N_calc - N_obs

        # 观测算子对T的导数: dN/dT = -77.6*P/T² - 7.46e5*e/T³
        e = P_b * q / (EPSILON + q)
        dN_dT = -77.6 * P_b / T**2 - 7.46e5 * e / T**3

        # 观测算子对q的导数: dN/dq = 3.73e5 * P * EPSILON / (EPSILON + q)² / T²
        dN_dq = 3.73e5 * P_b * EPSILON / (EPSILON + q)**2 / T**2

        # 梯度
        grad_bg = B_inv @ (x - x_b)

        # 观测项梯度: ∇J_o = H^T R^{-1} (Hx - y)
        grad_obs_T = dN_dT * (R_inv @ dy)
        grad_obs_q = dN_dq * (R_inv @ dy)
        grad_obs = np.concatenate([grad_obs_T, grad_obs_q])

        return grad_bg + grad_obs

    # 约束条件
    bounds_T = [(T_b[i] - 10, T_b[i] + 10) for i in range(n_levels)]
    bounds_q = [(max(0, q_b[i] * 0.1), q_b[i] * 3.0) for i in range(n_levels)]
    bounds = bounds_T + bounds_q

    # L-BFGS-B 优化
    result = minimize(cost_function, x_b, method='L-BFGS-B',
                      jac=gradient, bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-12})

    x_ana = result.x
    T_ana = x_ana[:n_levels]
    q_ana = x_ana[n_levels:]

    # 计算分析场折射率
    N_ana, N_dry_ana, N_wet_ana = refractivity_from_tpwv(T_ana, P_b, q_ana)

    return T_ana, q_ana, N_ana, N_dry_ana, N_wet_ana


# ============================================================
# 第 4.8 部分：简化版1DVar（逐高度独立反演）
# ============================================================
def one_dimensional_variational_simple(N_obs, h_km, T_b, q_b, P_b,
                                       obs_error=0.5, bg_error_ratio=0.3):
    """
    简化版1DVar：逐高度独立反演，计算效率更高。

    对每个高度层独立求解：
        x_a = x_b + K (y - Hx_b)
        K = B H^T (H B H^T + R)^{-1}

    参数:
        N_obs: 观测折射率剖面 (N-units)
        h_km: 高度 (km)
        T_b: 背景温度 (K)
        q_b: 背景比湿 (kg/kg)
        P_b: 背景气压 (hPa)
        obs_error: 观测折射率误差 (N-units)
        bg_error_ratio: 背景误差与观测误差比
    返回:
        T_ana: 分析温度 (K)
        q_ana: 分析比湿 (kg/kg)
        N_ana: 分析折射率 (N-units)
        N_dry_ana: 分析干项折射率 (N-units)
        N_wet_ana: 分析湿项折射率 (N-units)
    """
    n_levels = len(h_km)

    T_ana = np.zeros(n_levels)
    q_ana = np.zeros(n_levels)

    for i in range(n_levels):
        # 背景值
        T_bi = T_b[i]
        q_bi = q_b[i]
        P_bi = P_b[i]

        # 背景场计算的折射率
        N_bg, N_dry_bg, N_wet_bg = refractivity_from_tpwv(T_bi, P_bi, q_bi)

        # 观测增量
        dy = N_obs[i] - N_bg

        # 计算雅可比矩阵 H = [dN/dT, dN/dq]
        e_bi = P_bi * q_bi / (EPSILON + q_bi)
        H_T = -77.6 * P_bi / T_bi**2 - 7.46e5 * e_bi / T_bi**3
        H_q = 3.73e5 * P_bi * EPSILON / (EPSILON + q_bi)**2 / T_bi**2
        H = np.array([H_T, H_q])

        # 背景误差
        sigma_T_bg = 2.0
        sigma_q_bg = 0.002
        B = np.diag([sigma_T_bg**2, sigma_q_bg**2])

        # 观测误差
        R = obs_error**2

        # 卡尔曼增益: K = B H^T (H B H^T + R)^{-1}
        S = H @ B @ H.T + R
        K = B @ H.T / S

        # 状态增量
        dx = K * dy

        # 分析场
        T_ana[i] = np.clip(T_bi + dx[0], T_bi - 10, T_bi + 10)
        q_ana[i] = np.clip(q_bi + dx[1], 1e-8, q_bi * 3.0)

    # 计算分析场折射率
    N_ana, N_dry_ana, N_wet_ana = refractivity_from_tpwv(T_ana, P_b, q_ana)

    return T_ana, q_ana, N_ana, N_dry_ana, N_wet_ana


# ============================================================
# 第 4.9 部分：统计平衡约束（可选进阶功能）
# ============================================================
def apply_balance_constraint(T_profile, q_profile, P_profile, h_km):
    """
    应用统计平衡约束，使温湿廓线满足静力平衡和经验关系。

    约束条件：
    1. 温度递减率约束
    2. 比湿不超过饱和
    3. 水汽随高度递减

    参数:
        T_profile: 温度剖面 (K)
        q_profile: 比湿剖面 (kg/kg)
        P_profile: 气压剖面 (hPa)
        h_km: 高度 (km)
    返回:
        T_bal: 约束后温度 (K)
        q_bal: 约束后比湿 (kg/kg)
    """
    n_levels = len(h_km)
    T_bal = T_profile.copy()
    q_bal = q_profile.copy()

    # 约束1: 相对湿度不超过100%
    for i in range(n_levels):
        # 饱和水汽压 (Magnus公式)
        T_c = T_bal[i] - 273.15
        e_sat = 6.112 * np.exp(17.67 * T_c / (T_c + 243.5))  # hPa

        # 实际水汽压
        e_act = P_profile[i] * q_bal[i] / (EPSILON + q_bal[i])

        # 若超过饱和，限制比湿
        if e_act > e_sat:
            q_bal[i] = EPSILON * e_sat / (P_profile[i] - e_sat)

    # 约束2: 比湿随高度递减（对流层内）
    troposphere = h_km <= 15.0
    for i in range(1, n_levels):
        if troposphere[i] and q_bal[i] > q_bal[i-1] * 1.2:
            q_bal[i] = q_bal[i-1] * 1.2

    # 约束3: 温度递减率限制 (4-9 K/km)
    for i in range(1, n_levels):
        if h_km[i] <= 15.0:
            dT_dz = (T_bal[i-1] - T_bal[i]) / (h_km[i] - h_km[i-1])
            if dT_dz < 4.0:
                T_bal[i] = T_bal[i-1] - 4.0 * (h_km[i] - h_km[i-1])
            elif dT_dz > 9.0:
                T_bal[i] = T_bal[i-1] - 9.0 * (h_km[i] - h_km[i-1])

    return T_bal, q_bal


def invert_bending_angle_spline(a_array, alpha_array, r_output):
    """
    高精度 Abel 反演：样条插值 + 自适应高斯积分。

    通过对 α(a) 进行三次样条插值，再用 scipy.integrate.quad 进行
    自适应积分，比简单梯形法更精确，尤其在数据稀疏时。

    参数:
        a_array:   碰撞参数数组 (km)
        alpha_array: 弯曲角数组 (弧度)
        r_output:  输出的地心距离数组 (km)
    返回:
        n_output:  反演的折射指数数组
        N_output:  反演的折射率 (N-units)
    """
    # 对 α(a) 做三次样条插值
    alpha_spline = CubicSpline(a_array, alpha_array, extrapolate=False)
    a_top = a_array[-1]

    n_output = np.ones_like(r_output)

    for i, r_val in enumerate(r_output):
        if r_val >= a_top - 1e-6:
            n_output[i] = 1.0
            continue

        # 被积函数（用于自适应积分）
        def integrand(a):
            return alpha_spline(a) / np.sqrt(a**2 - r_val**2)

        # 自适应积分：处理 a→r 时的奇点
        # 拆分积分：[r, r+delta] 和 [r+delta, a_top]
        delta = max(0.01, r_val * 1e-4)
        a_mid = min(r_val + delta, a_top)

        try:
            if a_mid < a_top:
                part1, _ = quad(integrand, r_val, a_mid,
                                points=[r_val], limit=200)
                part2, _ = quad(integrand, a_mid, a_top, limit=200)
                integral = part1 + part2
            else:
                integral, _ = quad(integrand, r_val, a_top,
                                   points=[r_val], limit=200)
        except Exception:
            # 退化到梯形法
            mask = a_array >= r_val
            a_integ = a_array[mask]
            alpha_integ = alpha_array[mask]
            if len(a_integ) >= 3:
                denom = np.sqrt(np.maximum(a_integ**2 - r_val**2, 1e-20))
                integral = np.trapz(alpha_integ / denom, a_integ)
            else:
                integral = 0.0

        ln_n = integral / np.pi
        n_output[i] = np.exp(ln_n)

    N_output = (n_output - 1.0) * 1e6

    return n_output, N_output


# ============================================================
# 第 5 部分：主程序 —— 模拟数据 + 反演 + 可视化
# ============================================================
def main():
    print("=" * 60)
    print("  GPS掩星 Abel积分反演大气折射率剖面")
    print("=" * 60)

    # ---- 5.1 构造高度剖面 ----
    h_km = np.linspace(H_BOTTOM, H_TOP, N_POINTS)
    r_km = R_EARTH + h_km

    print(f"\n[1] 生成模拟折射率剖面 (指数模型, Hs={H_SCALE}km, N0={N_SURFACE})")
    N_true = generate_refractivity_profile(h_km)
    n_true = n_from_N(N_true)

    # ---- 5.2 Abel正演：由折射率计算弯曲角 ----
    print("[2] Abel正演：由 n(r) 计算弯曲角 α(a) ...")
    a_array, alpha_rad = forward_bending_angle(r_km, n_true)
    alpha_mrad = alpha_rad * 1e3   # 转换为毫弧度
    print(f"    碰撞参数范围: {a_array[0]:.2f} ~ {a_array[-1]:.2f} km")
    print(f"    弯曲角范围:   {alpha_mrad[0]:.4f} ~ {alpha_mrad[-1]:.6f} mrad")

    # ---- 5.3 Abel反演：由弯曲角恢复折射率 ----
    print("[3] Abel反演（梯形法）：由 α(a) 反演 n(r) ...")
    n_inv_trap, N_inv_trap = invert_bending_angle(a_array, alpha_rad, r_km)

    print("[4] Abel反演（样条+自适应积分）：由 α(a) 反演 n(r) ...")
    n_inv_spline, N_inv_spline = invert_bending_angle_spline(a_array, alpha_rad, r_km)

    # ---- 5.4 误差分析 ----
    print("[5] 误差分析 ...")
    rel_err_trap = np.abs(N_inv_trap - N_true) / np.maximum(N_true, 1e-10) * 100
    rel_err_spline = np.abs(N_inv_spline - N_true) / np.maximum(N_true, 1e-10) * 100

    print(f"\n    梯形法反演:")
    print(f"      最大相对误差: {np.max(rel_err_trap):.4f} %")
    print(f"      平均相对误差: {np.mean(rel_err_trap):.6f} %")
    print(f"\n    样条+自适应积分反演:")
    print(f"      最大相对误差: {np.max(rel_err_spline):.4f} %")
    print(f"      平均相对误差: {np.mean(rel_err_spline):.6f} %")

    # ---- 5.5 可视化 ----
    print("[6] 绘图 ...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (1) 真实折射率剖面
    ax = axes[0, 0]
    ax.plot(N_true, h_km, 'b-', linewidth=2)
    ax.set_xlabel('折射率 N (N-units)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('真实折射率剖面 N(h)')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (2) 弯曲角剖面
    ax = axes[0, 1]
    ax.plot(alpha_mrad, a_array - R_EARTH, 'r-', linewidth=2)
    ax.set_xlabel('弯曲角 α (mrad)')
    ax.set_ylabel('碰撞参数高度 (km)')
    ax.set_title('弯曲角剖面 α(a)')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (3) 反演结果对比
    ax = axes[0, 2]
    ax.semilogx(N_true, h_km, 'k-', linewidth=2.5, label='真实剖面')
    ax.semilogx(N_inv_trap, h_km, 'r--', linewidth=1.5, label='梯形法反演')
    ax.semilogx(N_inv_spline, h_km, 'b.', linewidth=1, markersize=2, label='样条+自适应')
    ax.set_xlabel('折射率 N (N-units)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('反演结果对比（对数坐标）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (4) 相对误差（线性高度范围）
    ax = axes[1, 0]
    ax.plot(rel_err_trap, h_km, 'r-', linewidth=1.5, label='梯形法')
    ax.plot(rel_err_spline, h_km, 'b-', linewidth=1.5, label='样条+自适应')
    ax.set_xlabel('相对误差 (%)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('反演相对误差')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (5) 对数误差
    ax = axes[1, 1]
    ax.semilogx(rel_err_trap + 1e-10, h_km, 'r-', linewidth=1.5, label='梯形法')
    ax.semilogx(rel_err_spline + 1e-10, h_km, 'b-', linewidth=1.5, label='样条+自适应')
    ax.set_xlabel('相对误差 (%)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('反演相对误差（对数坐标）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (6) 近地面误差放大
    ax = axes[1, 2]
    h_low = h_km < 20
    ax.plot(N_true[h_low], h_km[h_low], 'k-', linewidth=2.5, label='真实')
    ax.plot(N_inv_spline[h_low], h_km[h_low], 'b.', linewidth=1, markersize=3, label='反演')
    ax.set_xlabel('折射率 N (N-units)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('0-20km 近地面对比')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    output_fig = 'gps_ro_abel_inversion.png'
    plt.savefig(output_fig, dpi=150, bbox_inches='tight')
    print(f"    图像已保存: {output_fig}")
    plt.close()

    # ---- 5.6 打印关键数值 ----
    print("\n" + "=" * 60)
    print("  关键高度层反演结果对比")
    print("=" * 60)
    key_heights = [0, 5, 10, 15, 20, 30, 40, 50]
    print(f"{'高度(km)':>10s} {'N_true':>10s} {'N_inv':>10s} {'误差(%)':>10s}")
    print("-" * 50)
    for hk in key_heights:
        idx = np.argmin(np.abs(h_km - hk))
        print(f"{hk:10.1f} {N_true[idx]:10.2f} {N_inv_spline[idx]:10.2f} "
              f"{rel_err_spline[idx]:10.4f}")

    print("\n" + "=" * 60)
    print("  反演完成！")
    print("=" * 60)

    return N_true, N_inv_spline, h_km, a_array, alpha_rad


# ============================================================
# 第 6 部分：实际数据读取接口（供真实数据使用）
# ============================================================
def read_bending_angle_data(filename):
    """
    读取实际弯曲角观测数据。

    文件格式（文本，两列）：
        # 碰撞参数(km)  弯曲角(弧度)
        6371.000   0.025
        6372.000   0.023
        ...

    参数:
        filename: 数据文件路径
    返回:
        a_array:   碰撞参数 (km)
        alpha_array: 弯曲角 (弧度)
    """
    data = np.loadtxt(filename, comments='#')
    a_array = data[:, 0]
    alpha_array = data[:, 1]

    # 确保单调递增
    sort_idx = np.argsort(a_array)
    a_array = a_array[sort_idx]
    alpha_array = alpha_array[sort_idx]

    return a_array, alpha_array


def save_refractivity_profile(filename, h_km, N_profile):
    """
    保存反演的折射率剖面到文件。

    参数:
        filename:  输出文件路径
        h_km:      高度数组 (km)
        N_profile: 折射率数组 (N-units)
    """
    header = f"# GPS掩星 Abel积分反演结果\n# 高度(km)  折射率N(N-units)\n"
    data = np.column_stack([h_km, N_profile])
    np.savetxt(filename, data, header=header, fmt='%.6f', comments='')
    print(f"反演结果已保存: {filename}")


# ============================================================
# 第 6.5 部分：1DVar 主程序 —— 水汽模糊分离演示
# ============================================================
def main_with_1dvar():
    """
    主程序：演示1DVar变分同化分离干/湿折射率贡献。

    流程:
    1. 生成模拟温湿廓线（真实状态）
    2. 由温湿计算折射率（Smith-Weintraub）
    3. Abel正演生成弯曲角
    4. Abel反演恢复折射率（仅得到总N）
    5. 1DVar变分同化分离干/湿项
    6. 结果可视化
    """
    print("=" * 70)
    print("  GPS掩星 1DVar变分同化：低对流层水汽模糊分离")
    print("=" * 70)

    # ---- 6.5.1 构造高度剖面 ----
    h_km = np.linspace(H_BOTTOM, H_TOP, N_POINTS)
    r_km = R_EARTH + h_km
    h_m = h_km * 1000.0

    print(f"\n[1] 生成模拟温湿廓线（真实状态）")
    # 真实温湿廓线（添加一些扰动使其与背景场不同）
    T_true, q_true, P_true = generate_background_profiles(h_km)

    # 添加扰动模拟真实大气与预报模式的差异
    np.random.seed(42)
    T_perturb = np.zeros_like(h_km)
    q_perturb = np.zeros_like(h_km)

    # 对流层内添加结构
    for i, h in enumerate(h_km):
        if h < 15:
            # 温度扰动（逆温层结构）
            T_perturb[i] = 2.0 * np.exp(-((h - 5.0) / 3.0)**2)
            # 湿度扰动（边界层湿层）
            q_perturb[i] = 0.005 * np.exp(-((h - 1.5) / 1.0)**2)

    T_true = T_true + T_perturb
    q_true = np.maximum(q_true + q_perturb, 1e-8)

    # 重新计算气压（静力平衡）
    P_true = compute_pressure_profile(1013.25, T_true, h_m)

    # 计算真实折射率（含干/湿项）
    N_true, N_dry_true, N_wet_true = refractivity_from_tpwv(T_true, P_true, q_true)
    n_true = n_from_N(N_true)

    print(f"    地面温度: {T_true[0]:.1f} K")
    print(f"    地面比湿: {q_true[0]*1000:.1f} g/kg")
    print(f"    地面折射率: {N_true[0]:.1f} N-units")
    print(f"    地面干项: {N_dry_true[0]:.1f}, 湿项: {N_wet_true[0]:.1f}")

    # ---- 6.5.2 Abel正演 ----
    print("\n[2] Abel正演：由温湿计算弯曲角 ...")
    a_array, alpha_rad = forward_bending_angle(r_km, n_true)
    alpha_mrad = alpha_rad * 1e3
    print(f"    弯曲角范围: {alpha_mrad[0]:.4f} ~ {alpha_mrad[-1]:.6f} mrad")

    # ---- 6.5.3 Abel反演 ----
    print("[3] Abel反演：由弯曲角反演总折射率 ...")
    n_inv, N_inv = invert_bending_angle_spline(a_array, alpha_rad, r_km)

    # ---- 6.5.4 生成背景场（模拟预报模式） ----
    print("[4] 生成背景场（模拟ECMWF预报模式）...")
    T_b, q_b, P_b = generate_background_profiles(h_km)
    N_b, N_dry_b, N_wet_b = refractivity_from_tpwv(T_b, P_b, q_b)

    print(f"    背景地面温度: {T_b[0]:.1f} K")
    print(f"    背景地面比湿: {q_b[0]*1000:.1f} g/kg")

    # ---- 6.5.5 1DVar变分同化 ----
    print("\n[5] 1DVar变分同化：分离干/湿折射率贡献 ...")

    # 简化版1DVar（逐高度独立）
    T_ana, q_ana, N_ana, N_dry_ana, N_wet_ana = \
        one_dimensional_variational_simple(N_inv, h_km, T_b, q_b, P_b,
                                           obs_error=0.5, bg_error_ratio=0.3)

    # 应用平衡约束
    T_ana_bal, q_ana_bal = apply_balance_constraint(T_ana, q_ana, P_b, h_km)
    N_ana_bal, N_dry_bal, N_wet_bal = refractivity_from_tpwv(T_ana_bal, P_b, q_ana_bal)

    # ---- 6.5.6 误差分析 ----
    print("\n[6] 误差分析 ...")

    # 折射率误差
    err_N = np.abs(N_ana_bal - N_true) / np.maximum(N_true, 1e-10) * 100

    # 干项误差
    err_Ndry = np.abs(N_dry_bal - N_dry_true) / np.maximum(N_dry_true, 1e-10) * 100

    # 湿项误差
    mask_wet = N_wet_true > 1.0  # 只在有意义的湿区统计
    if np.any(mask_wet):
        err_Nwet = np.abs(N_wet_bal[mask_wet] - N_wet_true[mask_wet]) / \
                   np.maximum(N_wet_true[mask_wet], 1e-10) * 100
        print(f"\n    湿项反演误差（仅湿区）:")
        print(f"      最大相对误差: {np.max(err_Nwet):.2f} %")
        print(f"      平均相对误差: {np.mean(err_Nwet):.4f} %")

    print(f"\n    总折射率误差:")
    print(f"      最大相对误差: {np.max(err_N):.4f} %")
    print(f"      平均相对误差: {np.mean(err_N):.6f} %")

    print(f"\n    干项折射率误差:")
    print(f"      最大相对误差: {np.max(err_Ndry):.4f} %")
    print(f"      平均相对误差: {np.mean(err_Ndry):.6f} %")

    # ---- 6.5.7 可视化 ----
    print("\n[7] 绘图 ...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # (1) 温度剖面
    ax = axes[0, 0]
    ax.plot(T_true - 273.15, h_km, 'r-', linewidth=2, label='真实温度')
    ax.plot(T_b - 273.15, h_km, 'b--', linewidth=1.5, label='背景温度')
    ax.plot(T_ana_bal - 273.15, h_km, 'g-', linewidth=1.5, label='1DVar分析')
    ax.set_xlabel('温度 (°C)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('温度剖面对比')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (2) 比湿剖面
    ax = axes[0, 1]
    ax.semilogx(q_true * 1000, h_km, 'r-', linewidth=2, label='真实比湿')
    ax.semilogx(q_b * 1000, h_km, 'b--', linewidth=1.5, label='背景比湿')
    ax.semilogx(q_ana_bal * 1000, h_km, 'g-', linewidth=1.5, label='1DVar分析')
    ax.set_xlabel('比湿 (g/kg)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('比湿剖面对比（对数坐标）')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (3) 折射率剖面
    ax = axes[0, 2]
    ax.semilogx(N_true, h_km, 'k-', linewidth=2.5, label='真实折射率')
    ax.semilogx(N_inv, h_km, 'b.', linewidth=1, markersize=2, label='Abel反演')
    ax.semilogx(N_ana_bal, h_km, 'g-', linewidth=1.5, label='1DVar分析')
    ax.set_xlabel('折射率 N (N-units)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('总折射率对比')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (4) 干项折射率
    ax = axes[1, 0]
    ax.plot(N_dry_true, h_km, 'k-', linewidth=2, label='真实干项')
    ax.plot(N_dry_b, h_km, 'b--', linewidth=1.5, label='背景干项')
    ax.plot(N_dry_bal, h_km, 'g-', linewidth=1.5, label='1DVar干项')
    ax.set_xlabel('干项折射率 N_dry')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('干项折射率对比')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (5) 湿项折射率
    ax = axes[1, 1]
    ax.plot(N_wet_true, h_km, 'r-', linewidth=2, label='真实湿项')
    ax.plot(N_wet_b, h_km, 'b--', linewidth=1.5, label='背景湿项')
    ax.plot(N_wet_bal, h_km, 'g-', linewidth=1.5, label='1DVar湿项')
    ax.set_xlabel('湿项折射率 N_wet')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('湿项折射率对比')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (6) 湿项占比
    ax = axes[1, 2]
    wet_frac_true = N_wet_true / np.maximum(N_true, 1e-10) * 100
    wet_frac_ana = N_wet_bal / np.maximum(N_ana_bal, 1e-10) * 100
    ax.plot(wet_frac_true, h_km, 'r-', linewidth=2, label='真实湿项占比')
    ax.plot(wet_frac_ana, h_km, 'g-', linewidth=1.5, label='1DVar湿项占比')
    ax.set_xlabel('湿项占比 (%)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('湿项占比（N_wet/N_total）')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (7) 折射率误差
    ax = axes[2, 0]
    ax.plot(err_N, h_km, 'b-', linewidth=1.5, label='总折射率误差')
    ax.plot(err_Ndry, h_km, 'r-', linewidth=1.5, label='干项误差')
    if np.any(mask_wet):
        err_wet_full = np.full_like(h_km, np.nan)
        err_wet_full[mask_wet] = err_Nwet
        ax.plot(err_wet_full, h_km, 'g-', linewidth=1.5, label='湿项误差')
    ax.set_xlabel('相对误差 (%)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('1DVar反演误差')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (8) 温湿增量（分析-背景）
    ax = axes[2, 1]
    ax.plot(T_ana_bal - T_b, h_km, 'r-', linewidth=1.5, label='温度增量')
    ax2 = ax.twiny()
    ax2.plot((q_ana_bal - q_b) * 1000, h_km, 'b-', linewidth=1.5, label='比湿增量')
    ax.set_xlabel('温度增量 (K)')
    ax2.set_xlabel('比湿增量 (g/kg)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('分析增量（Analysis - Background）')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (9) 低对流层放大图
    ax = axes[2, 2]
    h_low = h_km < 10
    ax.plot(N_wet_true[h_low], h_km[h_low], 'r-', linewidth=2.5, label='真实湿项')
    ax.plot(N_wet_b[h_low], h_km[h_low], 'b--', linewidth=1.5, label='背景湿项')
    ax.plot(N_wet_bal[h_low], h_km[h_low], 'g-', linewidth=2, label='1DVar湿项')
    ax.set_xlabel('湿项折射率 N_wet')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('0-10km 低对流层湿项对比')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    output_fig = 'gps_ro_1dvar_moisture_separation.png'
    plt.savefig(output_fig, dpi=150, bbox_inches='tight')
    print(f"    图像已保存: {output_fig}")
    plt.close()

    # ---- 6.5.8 打印关键高度结果 ----
    print("\n" + "=" * 70)
    print("  关键高度层温湿反演结果")
    print("=" * 70)
    key_heights = [0, 2, 5, 10, 15, 20, 30]
    print(f"{'高度':>6s} {'T_true':>8s} {'T_ana':>8s} {'q_true':>8s} {'q_ana':>8s} "
          f"{'Nwet_t':>8s} {'Nwet_a':>8s}")
    print("-" * 60)
    for hk in key_heights:
        idx = np.argmin(np.abs(h_km - hk))
        print(f"{hk:6.1f} {T_true[idx]-273.15:8.1f} {T_ana_bal[idx]-273.15:8.1f} "
              f"{q_true[idx]*1000:8.2f} {q_ana_bal[idx]*1000:8.2f} "
              f"{N_wet_true[idx]:8.1f} {N_wet_bal[idx]:8.1f}")

    print("\n" + "=" * 70)
    print("  1DVar变分同化完成！")
    print("=" * 70)

    return T_true, q_true, T_ana_bal, q_ana_bal, h_km


# ============================================================
# 第 8 部分：电离层物理常数
# ============================================================
K_IONO = 40.308              # 电离层常数 (m³/s²)
F_L1 = 1575.42e6             # GPS L1 频率 (Hz)
F_L2 = 1227.60e6             # GPS L2 频率 (Hz)
C_LIGHT = 299792458.0        # 光速 (m/s)

# 电离层反演高度范围
H_IONO_TOP = 2000.0          # 电离层反演顶层 (km)
H_IONO_BOTTOM = 60.0         # 电离层反演底层 (km)
N_IONO_POINTS = 500          # 电离层采样点数


# ============================================================
# 第 9 部分：电离层电子密度剖面模拟（Chapman函数）
# ============================================================
def chapman_function(h_km, Nm, hm, H):
    """
    Chapman函数：描述电离层电子密度剖面。

    Ne(h) = Nm * exp(0.5 * (1 - z - exp(-z)))
    z = (h - hm) / H

    参数:
        h_km: 高度 (km)
        Nm: 峰值电子密度 (el/m³)
        hm: 峰值高度 (km)
        H: 标高 (km)
    返回:
        Ne: 电子密度 (el/m³)
    """
    z = (h_km - hm) / H
    return Nm * np.exp(0.5 * (1.0 - z - np.exp(-z)))


def generate_ionosphere_profile(h_km, solar_activity='medium'):
    """
    生成模拟电离层电子密度剖面（F2层为主）。

    基于国际参考电离层（IRI）的简化模型：
    - E层：~100-150 km, 峰值~1e11 el/m³
    - F1层：~180-220 km, 峰值~2e11 el/m³
    - F2层：~250-400 km, 峰值随太阳活动变化

    参数:
        h_km: 高度数组 (km)
        solar_activity: 太阳活动水平 ('low', 'medium', 'high')
    返回:
        Ne: 电子密度剖面 (el/m³)
    """
    Ne = np.zeros_like(h_km)

    # 太阳活动参数
    if solar_activity == 'low':
        Nm_F2 = 5e11              # 低太阳活动
        hm_F2 = 320.0
    elif solar_activity == 'high':
        Nm_F2 = 3e12              # 高太阳活动
        hm_F2 = 400.0
    else:  # medium
        Nm_F2 = 1.5e12            # 中等太阳活动
        hm_F2 = 350.0

    H_F2 = 60.0                   # F2层标高

    # F2层
    Ne += chapman_function(h_km, Nm_F2, hm_F2, H_F2)

    # F1层
    Nm_F1 = 2e11
    hm_F1 = 200.0
    H_F1 = 40.0
    Ne += chapman_function(h_km, Nm_F1, hm_F1, H_F1)

    # E层
    Nm_E = 1e11
    hm_E = 110.0
    H_E = 20.0
    Ne += chapman_function(h_km, Nm_E, hm_E, H_E)

    # 顶部等离子体层（>1000 km）
    plasmasphere = np.zeros_like(h_km)
    mask_top = h_km > 800.0
    plasmasphere[mask_top] = 1e10 * np.exp(-(h_km[mask_top] - 800.0) / 1000.0)
    Ne += plasmasphere

    return np.maximum(Ne, 1e6)  # 最小电子密度


# ============================================================
# 第 10 部分：电离层折射率与延迟计算
# ============================================================
def ionospheric_refractivity(Ne, freq):
    """
    计算电离层折射率（N-units）。

    N_iono = -40.3 * Ne / f²

    参数:
        Ne: 电子密度 (el/m³)
        freq: 信号频率 (Hz)
    返回:
        N_iono: 电离层折射率 (N-units)
    """
    return -K_IONO * Ne / freq**2 * 1e6


def ionospheric_delay(Ne, freq, h_profile):
    """
    计算电离层引起的信号延迟（距离延迟）。

    延迟 = (40.3 / f²) * ∫ Ne dh

    参数:
        Ne: 电子密度剖面 (el/m³)
        freq: 信号频率 (Hz)
        h_profile: 高度剖面 (m)
    返回:
        delay: 距离延迟 (m)
    """
    # TEC = ∫ Ne dh (el/m²)
    TEC = np.trapz(Ne, h_profile)
    # 延迟 = K_IONO * TEC / f²
    delay = K_IONO * TEC / freq**2
    return delay


def compute_tec(Ne, h_profile_km):
    """
    计算总电子含量（TEC）。

    TEC = ∫ Ne dh (单位：TECU, 1 TECU = 10^16 el/m²)

    参数:
        Ne: 电子密度剖面 (el/m³)
        h_profile_km: 高度剖面 (km)
    返回:
        TEC: 总电子含量 (TECU)
    """
    h_m = h_profile_km * 1000.0
    TEC = np.trapz(Ne, h_m)  # el/m²
    return TEC / 1e16        # TECU


def dual_frequency_iono_correction(delay_L1, delay_L2):
    """
    双频电离层改正：估计电离层延迟。

    一阶近似：
        I_iono = (f1² / (f1² - f2²)) * (delay_L2 - delay_L1)

    参数:
        delay_L1: L1频率延迟 (m)
        delay_L2: L2频率延迟 (m)
    返回:
        I_iono: 电离层延迟估计 (m)
    """
    factor = F_L1**2 / (F_L1**2 - F_L2**2)
    return factor * (delay_L2 - delay_L1)


# ============================================================
# 第 11 部分：电离层Abel积分反演
# ============================================================
def forward_ionosphere_bending(Ne_profile, h_km, freq=F_L1):
    """
    电离层Abel正演：由电子密度剖面计算弯曲角。

    电离层折射指数：n = 1 - 40.3 * Ne / (2 * f²)
    弯曲角：α(a) = (1/π) ∫_a^∞ [d(ln n)/dr] * r / √(r² - a²) dr

    参数:
        Ne_profile: 电子密度剖面 (el/m³)
        h_km: 高度 (km)
        freq: 信号频率 (Hz)
    返回:
        a_array: 碰撞参数 (km)
        alpha_array: 弯曲角 (弧度)
    """
    r_km = R_EARTH + h_km

    # 电离层折射指数（注意：n < 1，这里用 1 + N_iono*1e-6）
    N_iono = ionospheric_refractivity(Ne_profile, freq)
    n_iono = 1.0 + N_iono * 1e-6

    # 碰撞参数
    a_array = r_km * n_iono

    # 计算 d(ln n)/dr
    ln_n = np.log(n_iono)
    d_ln_n_dr = np.gradient(ln_n, r_km)

    # 计算弯曲角
    alpha_array = np.zeros_like(a_array)

    for i, a_val in enumerate(a_array):
        mask = r_km >= a_val * 0.999999
        r_integ = r_km[mask]
        dlnn = d_ln_n_dr[mask]

        if len(r_integ) < 3:
            alpha_array[i] = 0.0
            continue

        denominator = np.sqrt(np.maximum(r_integ**2 - a_val**2, 1e-20))
        integrand = dlnn * r_integ / denominator
        alpha_array[i] = np.trapz(integrand, r_integ)

    return a_array, alpha_array


def invert_ionosphere_abel(a_array, alpha_array, r_output, freq=F_L1):
    """
    电离层Abel反演：由弯曲角反演电子密度剖面。

    ln n(r) = (1/π) ∫_r^∞ α(a) / √(a² - r²) da
    Ne = -2 * f² * (n - 1) / 40.3

    参数:
        a_array: 碰撞参数 (km)
        alpha_array: 弯曲角 (弧度)
        r_output: 输出地心距离 (km)
        freq: 信号频率 (Hz)
    返回:
        Ne_output: 电子密度 (el/m³)
        n_output: 折射指数
    """
    from scipy.interpolate import CubicSpline

    alpha_spline = CubicSpline(a_array, alpha_array, extrapolate=False)
    a_top = a_array[-1]

    n_output = np.ones_like(r_output)

    for i, r_val in enumerate(r_output):
        if r_val >= a_top - 1e-6:
            n_output[i] = 1.0
            continue

        def integrand(a):
            return alpha_spline(a) / np.sqrt(a**2 - r_val**2)

        delta = max(0.01, r_val * 1e-4)
        a_mid = min(r_val + delta, a_top)

        try:
            if a_mid < a_top:
                part1, _ = quad(integrand, r_val, a_mid,
                                points=[r_val], limit=200)
                part2, _ = quad(integrand, a_mid, a_top, limit=200)
                integral = part1 + part2
            else:
                integral, _ = quad(integrand, r_val, a_top,
                                   points=[r_val], limit=200)
        except Exception:
            mask = a_array >= r_val
            a_integ = a_array[mask]
            alpha_integ = alpha_array[mask]
            if len(a_integ) >= 3:
                denom = np.sqrt(np.maximum(a_integ**2 - r_val**2, 1e-20))
                integral = np.trapz(alpha_integ / denom, a_integ)
            else:
                integral = 0.0

        ln_n = integral / np.pi
        n_output[i] = np.exp(ln_n)

    # 由折射指数计算电子密度
    # N_iono = (n - 1) * 1e6 = -40.3 * Ne / f² * 1e6
    # Ne = -(n - 1) * f² / 40.3
    Ne_output = -(n_output - 1.0) * freq**2 / K_IONO

    return Ne_output, n_output


# ============================================================
# 第 12 部分：电离层双频反演（消除一阶电离层延迟）
# ============================================================
def dual_frequency_ionosphere_inversion(a_L1, alpha_L1, a_L2, alpha_L2,
                                        r_output):
    """
    双频电离层反演：利用L1/L2频率差异消除一阶电离层影响。

    原理：
        中性大气折射率与频率无关
        电离层折射率 ∝ 1/f²
        通过双频观测可以分离电离层和中性大气贡献

    参数:
        a_L1: L1频率的碰撞参数 (km)
        alpha_L1: L1频率的弯曲角 (弧度)
        a_L2: L2频率的碰撞参数 (km)
        alpha_L2: L2频率的弯曲角 (弧度)
        r_output: 输出地心距离 (km)
    返回:
        Ne_dual: 双频反演的电子密度 (el/m³)
        N_neutral: 中性大气折射率 (N-units)
    """
    # 分别反演L1和L2的总折射率
    Ne_L1, n_L1 = invert_ionosphere_abel(a_L1, alpha_L1, r_output, F_L1)
    Ne_L2, n_L2 = invert_ionosphere_abel(a_L2, alpha_L2, r_output, F_L2)

    # 总折射率（包含中性大气 + 电离层）
    N_L1 = (n_L1 - 1.0) * 1e6
    N_L2 = (n_L2 - 1.0) * 1e6

    # 双频组合消除一阶电离层影响
    # N_neutral = (f1² * N_L1 - f2² * N_L2) / (f1² - f2²)
    N_neutral = (F_L1**2 * N_L1 - F_L2**2 * N_L2) / (F_L1**2 - F_L2**2)

    # 电离层贡献
    # N_iono_L1 = N_L1 - N_neutral = -40.3 * Ne / f1² * 1e6
    N_iono_L1 = N_L1 - N_neutral
    Ne_dual = -N_iono_L1 * F_L1**2 / (K_IONO * 1e6)

    return Ne_dual, N_neutral


# ============================================================
# 第 13 部分：空间天气监测指标
# ============================================================
def compute_ionospheric_indices(Ne_profile, h_km):
    """
    计算电离层空间天气监测指标。

    指标包括：
    1. foF2: F2层临界频率 (MHz)
    2. hmF2: F2层峰值高度 (km)
    3. TEC: 总电子含量 (TECU)
    4. 电离层梯度指数

    参数:
        Ne_profile: 电子密度剖面 (el/m³)
        h_km: 高度 (km)
    返回:
        indices: 电离层指标字典
    """
    indices = {}

    # F2层峰值
    idx_peak = np.argmax(Ne_profile)
    indices['NmF2'] = Ne_profile[idx_peak]      # 峰值密度 (el/m³)
    indices['hmF2'] = h_km[idx_peak]            # 峰值高度 (km)

    # F2层临界频率: foF2 = 9 * sqrt(NmF2)  (MHz)
    indices['foF2'] = 9.0 * np.sqrt(indices['NmF2'] / 1e12)

    # TEC
    indices['TEC'] = compute_tec(Ne_profile, h_km)

    # E层特征 (100-150 km)
    mask_E = (h_km >= 90) & (h_km <= 160)
    if np.any(mask_E):
        idx_E = np.argmax(Ne_profile[mask_E])
        indices['NmE'] = Ne_profile[mask_E][idx_E]
        indices['hmE'] = h_km[mask_E][idx_E]

    # 底部厚度 (从100 km到hmF2)
    mask_bottom = h_km <= indices['hmF2']
    if np.any(mask_bottom):
        h_bottom = h_km[mask_bottom]
        Ne_bottom = Ne_profile[mask_bottom]
        if len(h_bottom) > 1:
            indices['bottom_thickness'] = h_bottom[-1] - h_bottom[0]

    # 高度梯度
    dNe_dh = np.gradient(Ne_profile, h_km)
    indices['max_gradient'] = np.max(np.abs(dNe_dh))

    return indices


# ============================================================
# 第 14 部分：电离层主程序
# ============================================================
def main_ionosphere():
    """
    电离层电子密度反演主程序。

    流程:
    1. 生成模拟电离层剖面（Chapman函数）
    2. 计算双频电离层延迟
    3. Abel正演生成双频弯曲角
    4. Abel反演恢复电子密度
    5. 双频组合消除电离层影响
    6. 计算空间天气指标
    7. 结果可视化
    """
    print("=" * 70)
    print("  GPS掩星电离层电子密度剖面反演")
    print("  空间天气监测与电离层延迟改正")
    print("=" * 70)

    # ---- 14.1 构造高度剖面 ----
    h_km = np.linspace(H_IONO_BOTTOM, H_IONO_TOP, N_IONO_POINTS)
    r_km = R_EARTH + h_km
    h_m = h_km * 1000.0

    print(f"\n[1] 生成模拟电离层剖面（中等太阳活动）")
    Ne_true = generate_ionosphere_profile(h_km, solar_activity='medium')

    # 计算关键指标
    idx_peak = np.argmax(Ne_true)
    NmF2 = Ne_true[idx_peak]
    hmF2 = h_km[idx_peak]
    TEC_true = compute_tec(Ne_true, h_km)

    print(f"    F2层峰值密度: {NmF2:.2e} el/m³")
    print(f"    F2层峰值高度: {hmF2:.1f} km")
    print(f"    总电子含量TEC: {TEC_true:.2f} TECU")

    # ---- 14.2 计算双频电离层延迟 ----
    print("\n[2] 计算双频电离层延迟 ...")
    delay_L1 = ionospheric_delay(Ne_true, F_L1, h_m)
    delay_L2 = ionospheric_delay(Ne_true, F_L2, h_m)
    delay_iono_est = dual_frequency_iono_correction(delay_L1, delay_L2)

    print(f"    L1频率延迟: {delay_L1*100:.4f} cm")
    print(f"    L2频率延迟: {delay_L2*100:.4f} cm")
    print(f"    估计电离层延迟: {delay_iono_est*100:.4f} cm")
    print(f"    一阶改正精度: {np.abs(delay_iono_est - delay_L1)/delay_L1*100:.4f} %")

    # ---- 14.3 Abel正演：生成双频弯曲角 ----
    print("\n[3] Abel正演：计算双频弯曲角 ...")
    a_L1, alpha_L1 = forward_ionosphere_bending(Ne_true, h_km, F_L1)
    a_L2, alpha_L2 = forward_ionosphere_bending(Ne_true, h_km, F_L2)

    print(f"    L1弯曲角范围: {np.min(alpha_L1)*1e6:.2f} ~ {np.max(alpha_L1)*1e6:.2f} μrad")
    print(f"    L2弯曲角范围: {np.min(alpha_L2)*1e6:.2f} ~ {np.max(alpha_L2)*1e6:.2f} μrad")

    # ---- 14.4 Abel反演：单频反演电子密度 ----
    print("\n[4] Abel反演：单频反演电子密度 ...")
    Ne_inv_L1, _ = invert_ionosphere_abel(a_L1, alpha_L1, r_km, F_L1)
    Ne_inv_L2, _ = invert_ionosphere_abel(a_L2, alpha_L2, r_km, F_L2)

    # 相对误差（限制在合理范围）
    mask_valid = Ne_true > 1e8
    err_L1 = np.abs(Ne_inv_L1[mask_valid] - Ne_true[mask_valid]) / Ne_true[mask_valid] * 100
    err_L2 = np.abs(Ne_inv_L2[mask_valid] - Ne_true[mask_valid]) / Ne_true[mask_valid] * 100

    print(f"    L1单频反演误差: 最大={np.max(err_L1):.2f}%, 平均={np.mean(err_L1):.4f}%")
    print(f"    L2单频反演误差: 最大={np.max(err_L2):.2f}%, 平均={np.mean(err_L2):.4f}%")

    # ---- 14.5 双频组合反演 ----
    print("\n[5] 双频组合反演：消除一阶电离层影响 ...")
    Ne_dual, N_neutral = dual_frequency_ionosphere_inversion(
        a_L1, alpha_L1, a_L2, alpha_L2, r_km)

    err_dual = np.abs(Ne_dual[mask_valid] - Ne_true[mask_valid]) / Ne_true[mask_valid] * 100
    print(f"    双频反演误差: 最大={np.max(err_dual):.2f}%, 平均={np.mean(err_dual):.4f}%")

    # ---- 14.6 空间天气指标 ----
    print("\n[6] 计算空间天气监测指标 ...")
    indices_true = compute_ionospheric_indices(Ne_true, h_km)
    indices_inv = compute_ionospheric_indices(Ne_dual, h_km)

    print(f"\n    真实指标:")
    print(f"      foF2 = {indices_true['foF2']:.2f} MHz")
    print(f"      hmF2 = {indices_true['hmF2']:.1f} km")
    print(f"      NmF2 = {indices_true['NmF2']:.2e} el/m³")
    print(f"      TEC  = {indices_true['TEC']:.2f} TECU")

    print(f"\n    反演指标:")
    print(f"      foF2 = {indices_inv['foF2']:.2f} MHz")
    print(f"      hmF2 = {indices_inv['hmF2']:.1f} km")
    print(f"      NmF2 = {indices_inv['NmF2']:.2e} el/m³")
    print(f"      TEC  = {indices_inv['TEC']:.2f} TECU")

    # ---- 14.7 可视化 ----
    print("\n[7] 绘图 ...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # (1) 电子密度剖面（对数坐标）
    ax = axes[0, 0]
    ax.semilogx(Ne_true, h_km, 'r-', linewidth=2, label='真实剖面')
    ax.semilogx(Ne_inv_L1, h_km, 'b--', linewidth=1.5, label='L1单频反演')
    ax.semilogx(Ne_dual, h_km, 'g-', linewidth=1.5, label='双频反演')
    ax.set_xlabel('电子密度 Ne (el/m³)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('电子密度剖面对比')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (2) F2层附近放大图
    ax = axes[0, 1]
    h_f2 = (h_km >= 200) & (h_km <= 500)
    ax.semilogx(Ne_true[h_f2], h_km[h_f2], 'r-', linewidth=2, label='真实')
    ax.semilogx(Ne_inv_L1[h_f2], h_km[h_f2], 'b--', linewidth=1.5, label='L1反演')
    ax.semilogx(Ne_dual[h_f2], h_km[h_f2], 'g-', linewidth=1.5, label='双频反演')
    ax.set_xlabel('电子密度 Ne (el/m³)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('F2层区域 (200-500 km)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (3) 电离层折射率
    ax = axes[0, 2]
    N_iono_L1 = ionospheric_refractivity(Ne_true, F_L1)
    N_iono_L2 = ionospheric_refractivity(Ne_true, F_L2)
    ax.plot(N_iono_L1, h_km, 'r-', linewidth=2, label='L1电离层折射率')
    ax.plot(N_iono_L2, h_km, 'b-', linewidth=2, label='L2电离层折射率')
    ax.set_xlabel('电离层折射率 (N-units)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('双频电离层折射率')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (4) 反演误差
    ax = axes[1, 0]
    ax.semilogx(err_L1 + 1e-10, h_km[mask_valid], 'b-', linewidth=1.5, label='L1单频误差')
    ax.semilogx(err_dual + 1e-10, h_km[mask_valid], 'g-', linewidth=1.5, label='双频误差')
    ax.set_xlabel('相对误差 (%)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('电子密度反演误差')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (5) 弯曲角剖面
    ax = axes[1, 1]
    ax.plot(alpha_L1 * 1e6, a_L1 - R_EARTH, 'r-', linewidth=2, label='L1弯曲角')
    ax.plot(alpha_L2 * 1e6, a_L2 - R_EARTH, 'b-', linewidth=2, label='L2弯曲角')
    ax.set_xlabel('弯曲角 α (μrad)')
    ax.set_ylabel('碰撞参数高度 (km)')
    ax.set_title('双频弯曲角剖面')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (6) TEC高度累积
    ax = axes[1, 2]
    TEC_cumulative = np.array([compute_tec(Ne_true[:i], h_km[:i])
                               for i in range(1, len(h_km))])
    ax.plot(TEC_cumulative, h_km[1:], 'r-', linewidth=2)
    ax.axvline(x=TEC_true, color='k', linestyle='--', label=f'TEC={TEC_true:.1f} TECU')
    ax.set_xlabel('累积TEC (TECU)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('TEC高度累积')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (7) 不同太阳活动对比
    ax = axes[2, 0]
    for act, color in [('low', 'b'), ('medium', 'g'), ('high', 'r')]:
        Ne_act = generate_ionosphere_profile(h_km, solar_activity=act)
        TEC_act = compute_tec(Ne_act, h_km)
        ax.semilogx(Ne_act, h_km, f'{color}-', linewidth=2,
                    label=f'{act} (TEC={TEC_act:.1f})')
    ax.set_xlabel('电子密度 Ne (el/m³)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('不同太阳活动水平')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # (8) 一阶电离层延迟改正效果
    ax = axes[2, 1]
    h_range = h_km < 2000
    ax.plot(h_km[h_range], Ne_true[h_range] * K_IONO / F_L1**2 * 100,
            'r-', linewidth=2, label='L1延迟 (cm/km)')
    ax.plot(h_km[h_range], Ne_true[h_range] * K_IONO / F_L2**2 * 100,
            'b-', linewidth=2, label='L2延迟 (cm/km)')
    ax.set_xlabel('高度 h (km)')
    ax.set_ylabel('单位高度延迟 (cm/km)')
    ax.set_title('双频单位高度延迟')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (9) 中性大气折射率估计
    ax = axes[2, 2]
    ax.plot(N_neutral, h_km, 'g-', linewidth=2)
    ax.set_xlabel('中性大气折射率 (N-units)')
    ax.set_ylabel('高度 h (km)')
    ax.set_title('双频估计的中性大气折射率')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    output_fig = 'gps_ro_ionosphere_inversion.png'
    plt.savefig(output_fig, dpi=150, bbox_inches='tight')
    print(f"    图像已保存: {output_fig}")
    plt.close()

    # ---- 14.8 打印关键高度结果 ----
    print("\n" + "=" * 70)
    print("  关键高度层电子密度反演结果")
    print("=" * 70)
    key_heights = [100, 150, 200, 250, 300, 350, 400, 500, 1000]
    print(f"{'高度(km)':>10s} {'Ne_true':>12s} {'Ne_L1':>12s} {'Ne_dual':>12s} {'误差(%)':>8s}")
    print("-" * 60)
    for hk in key_heights:
        idx = np.argmin(np.abs(h_km - hk))
        err = np.abs(Ne_dual[idx] - Ne_true[idx]) / Ne_true[idx] * 100
        print(f"{hk:10.1f} {Ne_true[idx]:12.3e} {Ne_inv_L1[idx]:12.3e} "
              f"{Ne_dual[idx]:12.3e} {err:8.2f}")

    print("\n" + "=" * 70)
    print("  电离层反演完成！")
    print("=" * 70)

    return Ne_true, Ne_dual, h_km, indices_true, indices_inv


# ============================================================
# 第 7 部分：程序入口
# ============================================================
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '1dvar':
            # 运行1DVar变分同化演示
            main_with_1dvar()
        elif sys.argv[1] == 'iono':
            # 运行电离层反演演示
            main_ionosphere()
        elif sys.argv[1] == 'all':
            # 运行全部演示
            print("\n" + "=" * 70)
            print("  运行全部演示")
            print("=" * 70)
            N_true, N_inv, h_km, a_array, alpha_rad = main()
            main_with_1dvar()
            main_ionosphere()
    else:
        # 运行基础Abel反演演示
        N_true, N_inv, h_km, a_array, alpha_rad = main()

    # 可选：保存反演结果
    # save_refractivity_profile('refractivity_profile.dat', h_km, N_inv)
