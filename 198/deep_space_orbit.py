# -*- coding: utf-8 -*-
"""
深空探测器轨道确定 — 多弧段 + 动力学补偿
============================================
功能:
  1. 多弧段数据组合 (多圈轨道观测)
  2. 动力学补偿 (J2扁率 + 日月摄动 + 太阳光压SRP)
  3. 批处理最小二乘 + Levenberg-Marquardt阻尼
  4. 联合估计偏差参数 + DOR差分观测

方法:
  - RK4 数值积分传播高精度轨道
  - 开普勒近似STM计算雅可比 (快速, 适合演示)
  - 多弧段数据融合提高定轨精度

精度目标: 米级 (通过多弧段和动力学补偿实现)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

# ==============================================================
# 物理常量
# ==============================================================
MU_EARTH = 398600.4418e9       # m³/s²
MU_SUN = 1.32712440018e20      # m³/s²
MU_MOON = 4902.8e9             # m³/s²
R_EARTH = 6378137.0            # m
J2 = 1.08262668e-3             # J2扁率
AU = 1.495978707e11            # m
OMEGA_EARTH = 7.2921151467e-5  # rad/s
C_LIGHT = 299792458.0          # m/s
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


@dataclass
class OrbitElements:
    """6个轨道根数"""
    a: float       # 半长轴 (m)
    e: float       # 偏心率
    i: float       # 轨道倾角 (rad)
    Om: float      # 升交点赤经 (rad)
    w: float       # 近心点幅角 (rad)
    f: float       # 真近点角 (rad)

    def to_cartesian(self) -> Tuple[np.ndarray, np.ndarray]:
        """根数 -> 位置/速度 (ECI)"""
        a, e, i, Om, w, f = self.a, self.e, self.i, self.Om, self.w, self.f
        p = a * (1 - e * e)
        r = p / (1 + e * np.cos(f))

        r_pqw = np.array([r * np.cos(f), r * np.sin(f), 0.0])
        v_pqw = np.sqrt(MU_EARTH / p) * np.array(
            [-np.sin(f), e + np.cos(f), 0.0])

        cos_Om, sin_Om = np.cos(Om), np.sin(Om)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_w, sin_w = np.cos(w), np.sin(w)

        R = np.array([
            [cos_Om * cos_w - sin_Om * sin_w * cos_i,
             -cos_Om * sin_w - sin_Om * cos_w * cos_i,
             sin_Om * sin_i],
            [sin_Om * cos_w + cos_Om * sin_w * cos_i,
             -sin_Om * sin_w + cos_Om * cos_w * cos_i,
             -cos_Om * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i]
        ])

        return R @ r_pqw, R @ v_pqw


@dataclass
class SpacecraftParams:
    """航天器参数"""
    area_mass_ratio: float = 0.01  # m²/kg
    reflectivity: float = 1.2


# ==============================================================
# 辅助: 日月位置 (低精度解析)
# ==============================================================
def _sun_position(t_j2000: float) -> np.ndarray:
    """太阳 ECI 位置 (m), 简化模型"""
    T = t_j2000 / 86400.0 / 36525.0
    L = (280.466 + 36000.770 * T) * DEG2RAD
    g = (357.5277233 + 35999.05034 * T) * DEG2RAD
    lambda_s = L + (1.914666 - 0.004889 * T) * np.sin(g) * DEG2RAD
    eps = (23.439291 - 0.013004 * T) * DEG2RAD
    r_sun = (1.00014 - 0.01671 * np.cos(g)) * AU
    return np.array([r_sun * np.cos(lambda_s),
                     r_sun * np.sin(lambda_s) * np.cos(eps),
                     r_sun * np.sin(lambda_s) * np.sin(eps)])


def _moon_position(t_j2000: float) -> np.ndarray:
    """月球 ECI 位置 (m), 简化模型"""
    T = t_j2000 / 86400.0 / 36525.0
    L0 = (218.316 + 481267.883 * T) * DEG2RAD
    M = (134.963 + 477198.868 * T) * DEG2RAD
    F = (93.272 + 483202.018 * T) * DEG2RAD
    D = (297.850 + 445267.111 * T) * DEG2RAD
    Om_m = (125.044 - 1934.136 * T) * DEG2RAD

    lon = L0 + (2264.0 / 3600) * np.sin(M) * DEG2RAD \
        - (4586.0 / 3600) * np.sin(M - 2 * D) * DEG2RAD
    lat = (18520.0 / 3600) * np.sin(F - Om_m) * DEG2RAD
    r_moon = 384400e3 * (1.0 - 0.0545 * np.cos(M - 2 * D))

    eps = 23.439291 * DEG2RAD
    return np.array([
        r_moon * (np.cos(lon) * np.cos(lat)),
        r_moon * (np.sin(lon) * np.cos(lat) * np.cos(eps)
                  - np.sin(lat) * np.sin(eps)),
        r_moon * (np.sin(lon) * np.cos(lat) * np.sin(eps)
                  + np.sin(lat) * np.cos(eps)),
    ])


# ==============================================================
# 动力学模型
# ==============================================================
def _accel_two_body(r: np.ndarray) -> np.ndarray:
    return -MU_EARTH * r / np.linalg.norm(r) ** 3


def _accel_j2(r: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r)
    x, y, z = r
    factor = 1.5 * J2 * MU_EARTH * R_EARTH ** 2 / r_norm ** 5
    return factor * np.array([
        x * (1 - 5 * z ** 2 / r_norm ** 2),
        y * (1 - 5 * z ** 2 / r_norm ** 2),
        z * (3 - 5 * z ** 2 / r_norm ** 2),
    ])


def _accel_third_body(r: np.ndarray, r_body: np.ndarray,
                      mu_body: float) -> np.ndarray:
    r_rel = r - r_body
    r_rel_norm = np.linalg.norm(r_rel)
    r_body_norm = np.linalg.norm(r_body)
    return -mu_body * (r_rel / r_rel_norm ** 3 + r_body / r_body_norm ** 3)


def _in_shadow(r: np.ndarray, r_sun: np.ndarray) -> bool:
    """简化阴影判定"""
    r_norm = np.linalg.norm(r)
    r_sun_norm = np.linalg.norm(r_sun)
    dot_val = np.dot(r, r_sun)
    if dot_val >= 0:
        return False
    return abs(np.cross(r, r_sun).sum()) / r_sun_norm < R_EARTH


def _accel_srp(r: np.ndarray, t_j2000: float,
               sc_params: SpacecraftParams) -> np.ndarray:
    r_sun = _sun_position(t_j2000)
    if _in_shadow(r, r_sun):
        return np.zeros(3)
    r_sun_norm = np.linalg.norm(r_sun)
    P = 4.56e-6
    shadow_factor = max(0.0, 1.0 - R_EARTH /
                        np.linalg.norm(np.cross(r_sun, r) / r_sun_norm))
    return (-P * sc_params.reflectivity * sc_params.area_mass_ratio
            * shadow_factor * r_sun / r_sun_norm)


def total_acceleration(r: np.ndarray, v: np.ndarray,
                       t_j2000: float,
                       sc_params: SpacecraftParams) -> np.ndarray:
    a = _accel_two_body(r)
    a += _accel_j2(r)
    r_sun = _sun_position(t_j2000)
    r_moon = _moon_position(t_j2000)
    a += _accel_third_body(r, r_sun, MU_SUN)
    a += _accel_third_body(r, r_moon, MU_MOON)
    a += _accel_srp(r, t_j2000, sc_params)
    return a


# ==============================================================
# RK4 数值积分
# ==============================================================
def rk4_step(state: np.ndarray, t: float, dt: float,
             sc_params: SpacecraftParams) -> np.ndarray:
    r = state[:3]; v = state[3:6]

    k1r = v
    k1v = total_acceleration(r, v, t, sc_params)

    k2r = v + 0.5 * dt * k1v
    k2v = total_acceleration(r + 0.5 * dt * k1r, k2r, t + 0.5 * dt, sc_params)

    k3r = v + 0.5 * dt * k2v
    k3v = total_acceleration(r + 0.5 * dt * k2r, k3r, t + 0.5 * dt, sc_params)

    k4r = v + dt * k3v
    k4v = total_acceleration(r + dt * k3r, k4r, t + dt, sc_params)

    r_new = r + (dt / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
    v_new = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

    return np.concatenate([r_new, v_new])


def propagate_numerical(r0: np.ndarray, v0: np.ndarray,
                        t0_j2000: float, dt: float,
                        sc_params: SpacecraftParams,
                        step: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
    """数值积分传播轨道"""
    state = np.concatenate([r0, v0])
    t = t0_j2000
    n_steps = max(1, int(abs(dt) / step))
    actual_dt = dt / n_steps

    for _ in range(n_steps):
        state = rk4_step(state, t, actual_dt, sc_params)
        t += actual_dt

    return state[:3], state[3:]


# ==============================================================
# 开普勒传播 (用于快速STM计算)
# ==============================================================
def cartesian_to_elements(r: np.ndarray, v: np.ndarray) -> OrbitElements:
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    n = np.cross(np.array([0, 0, 1.0]), h)
    n_norm = np.linalg.norm(n)

    r_norm = np.linalg.norm(r)
    e_vec = (np.cross(v, h) - MU_EARTH * r / r_norm) / MU_EARTH
    e = np.linalg.norm(e_vec)

    a = 1.0 / (2.0 / r_norm - np.dot(v, v) / MU_EARTH)

    i = np.arccos(np.clip(h[2] / h_norm, -1, 1))

    if n_norm > 1e-10:
        Om = np.arccos(np.clip(n[0] / n_norm, -1, 1))
        if n[1] < 0:
            Om = 2 * np.pi - Om
    else:
        Om = 0.0

    if n_norm > 1e-10 and e > 1e-10:
        w = np.arccos(np.clip(np.dot(n, e_vec) / (n_norm * e), -1, 1))
        if e_vec[2] < 0:
            w = 2 * np.pi - w
    else:
        w = 0.0

    if e > 1e-10:
        f = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_norm), -1, 1))
        if np.dot(r, v) < 0:
            f = 2 * np.pi - f
    else:
        f = 0.0

    return OrbitElements(a=a, e=e, i=i, Om=Om, w=w, f=f)


def _propagate_kepler_elem(elem: OrbitElements,
                           dt: float) -> Tuple[np.ndarray, np.ndarray]:
    e = elem.e
    E0 = 2.0 * np.arctan2(
        np.sqrt(1 - e) * np.sin(elem.f / 2),
        np.sqrt(1 + e) * np.cos(elem.f / 2),
    )
    M0 = E0 - e * np.sin(E0)
    n = np.sqrt(MU_EARTH / (elem.a ** 3))
    M = M0 + n * dt

    E = M if e < 0.8 else np.pi
    for _ in range(30):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < 1e-14:
            break

    cos_f = (np.cos(E) - e) / (1 - e * np.cos(E))
    sin_f = (np.sqrt(1 - e * e) * np.sin(E)) / (1 - e * np.cos(E))
    f_new = float(np.arctan2(sin_f, cos_f))

    new_elem = OrbitElements(a=elem.a, e=elem.e, i=elem.i,
                             Om=elem.Om, w=elem.w, f=f_new)
    return new_elem.to_cartesian()


def _kepler_stm(r0: np.ndarray, v0: np.ndarray, dt: float) -> np.ndarray:
    """开普勒近似STM (快速)"""
    stm = np.zeros((6, 6))
    h = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3])
    x0 = np.concatenate([r0, v0])
    for j in range(6):
        ej = np.zeros(6); ej[j] = h[j]
        x_p = x0 + ej
        x_m = x0 - ej
        elem_p = cartesian_to_elements(x_p[:3], x_p[3:])
        elem_m = cartesian_to_elements(x_m[:3], x_m[3:])
        rp, vp = _propagate_kepler_elem(elem_p, dt)
        rm, vm = _propagate_kepler_elem(elem_m, dt)
        stm[:3, j] = (rp - rm) / (2 * h[j])
        stm[3:, j] = (vp - vm) / (2 * h[j])
    return stm


# ==============================================================
# 观测模型
# ==============================================================
def _gmst(t_j2000: float) -> float:
    T = t_j2000 / 86400.0
    return (280.46061837 + 360.98564736629 * T) * DEG2RAD


def _station_pos_eci(sta_geo: Tuple, t_j2000: float) -> np.ndarray:
    lat, lon, alt = sta_geo
    theta = _gmst(t_j2000) + lon
    r = R_EARTH + alt
    return np.array([r * np.cos(lat) * np.cos(theta),
                     r * np.cos(lat) * np.sin(theta),
                     r * np.sin(lat)])


def _station_velocity_eci(sta_geo: Tuple, t_j2000: float) -> np.ndarray:
    lat, lon, alt = sta_geo
    theta = _gmst(t_j2000) + lon
    r = R_EARTH + alt
    return OMEGA_EARTH * np.array([-r * np.cos(lat) * np.sin(theta),
                                    r * np.cos(lat) * np.cos(theta),
                                    0.0])


def range_and_rate(r_sat: np.ndarray, v_sat: np.ndarray,
                   r_sta: np.ndarray,
                   v_sta: np.ndarray) -> Tuple[float, float]:
    rho_vec = r_sat - r_sta
    rho = np.linalg.norm(rho_vec)
    rho_hat = rho_vec / rho
    rho_dot = np.dot(v_sat - v_sta, rho_hat)
    return rho, rho_dot


def _obs_jacobian_wrt_rv(r_sat: np.ndarray, v_sat: np.ndarray,
                         r_sta: np.ndarray,
                         v_sta: np.ndarray) -> np.ndarray:
    """观测雅可比 (2x6)"""
    rho_vec = r_sat - r_sta
    rho = np.linalg.norm(rho_vec)
    rho_hat = rho_vec / rho
    v_rel = v_sat - v_sta
    rho_dot = np.dot(v_rel, rho_hat)

    H = np.zeros((2, 6))
    H[0, :3] = rho_hat
    H[1, :3] = (v_rel - rho_dot * rho_hat) / rho
    H[1, 3:] = rho_hat
    return H


# ==============================================================
# 观测仿真
# ==============================================================
def simulate_multi_arc_observations(
        true_elem: OrbitElements, t0_j2000: float,
        arc_specs: List[Dict], stations_geo: List[Tuple],
        sc_params: SpacecraftParams,
        sigma_rho: float = 2.0,
        sigma_rhodot: float = 5e-4,
        range_biases: Optional[np.ndarray] = None,
        clock_bias: float = 0.0) -> List:
    """多弧段观测仿真"""
    obs = []
    n_sta = len(stations_geo)
    if range_biases is None:
        range_biases = np.zeros(n_sta)

    for arc in arc_specs:
        t_start = arc["t_start"]
        t_end = arc["t_end"]
        n_obs = arc["n_obs"]
        times = np.linspace(t0_j2000 + t_start, t0_j2000 + t_end, n_obs)

        for t in times:
            dt = t - t0_j2000
            r_sat, v_sat = propagate_numerical(
                *true_elem.to_cartesian(), t0_j2000, dt, sc_params)

            for sta_idx, sta_geo in enumerate(stations_geo):
                r_sta = _station_pos_eci(sta_geo, t)
                v_sta = _station_velocity_eci(sta_geo, t)
                rho, rho_dot = range_and_rate(r_sat, v_sat, r_sta, v_sta)

                # 测距 (加系统偏差)
                rho_obs = rho + range_biases[sta_idx] + clock_bias \
                    + sigma_rho * np.random.randn()
                obs.append((t, 0, sta_idx, "range", rho_obs, sigma_rho))

                # 多普勒测速 (无系统偏差)
                rho_dot_obs = rho_dot + sigma_rhodot * np.random.randn()
                obs.append((t, 0, sta_idx, "rate", rho_dot_obs, sigma_rhodot))

    return obs


# ==============================================================
# DOR 差分观测
# ==============================================================
def _form_dor_from_obs(obs: List) -> List:
    """测距 -> DOR差分"""
    range_by_t = defaultdict(dict)
    rate_list = []

    for entry in obs:
        t, _, sta_idx, otype, val, sigma = entry
        if otype == "range":
            range_by_t[t][sta_idx] = (val, sigma)
        else:
            rate_list.append(entry)

    dor_obs = []
    for t, sta_map in range_by_t.items():
        sta_indices = sorted(sta_map.keys())
        for i_idx in range(len(sta_indices)):
            for j_idx in range(i_idx + 1, len(sta_indices)):
                si, sj = sta_indices[i_idx], sta_indices[j_idx]
                v_i, s_i = sta_map[si]
                v_j, s_j = sta_map[sj]
                delta = v_i - v_j
                sigma_dor = np.sqrt(s_i ** 2 + s_j ** 2)
                dor_obs.append((t, si, sj, "dor", delta, sigma_dor))

    return dor_obs + rate_list


# ==============================================================
# 批处理最小二乘
# ==============================================================
def batch_lsq_numerical(
        obs: List, x0_guess: np.ndarray, t0_j2000: float,
        stations_geo: List[Tuple], sc_params: SpacecraftParams,
        bias_guess: Optional[np.ndarray] = None,
        use_joint: bool = False, use_dor: bool = False,
        max_iter: int = 20, tol_pos: float = 1e-2) -> Tuple:
    """
    批处理最小二乘 (数值积分 + 开普勒STM雅可比)

    多弧段数据融合:
      - 所有弧段观测统一在 epoch 时刻估计状态
      - 每观测通过 STM 映射回 epoch

    动力学补偿:
      - 传播时使用完整动力学模型
      - 雅可比用开普勒近似 (高效)
    """
    if use_dor:
        obs = _form_dor_from_obs(obs)

    n_rv = 6
    n_state = n_rv + (len(bias_guess) if use_joint else 0)

    x = np.concatenate([x0_guess, bias_guess]) if use_joint else x0_guess.copy()
    prev_cost = np.inf
    lam = 1e-4

    for it in range(max_iter):
        HTH = np.zeros((n_state, n_state))
        HTz = np.zeros(n_state)
        cost = 0.0

        for entry in obs:
            t = entry[0]
            otype = entry[3]
            val = entry[4]
            sigma = entry[5]
            dt = t - t0_j2000

            # 高精度传播 (含J2+日月+SRP)
            r_sat, v_sat = propagate_numerical(
                x[:3], x[3:6], t0_j2000, dt, sc_params)

            # 快速STM (开普勒近似)
            stm = _kepler_stm(x[:3], x[3:6], dt)

            if otype in ("range", "rate"):
                sta_idx = entry[2]
                r_sta = _station_pos_eci(stations_geo[sta_idx], t)
                v_sta = _station_velocity_eci(stations_geo[sta_idx], t)
                rho, rho_dot = range_and_rate(r_sat, v_sat, r_sta, v_sta)

                y_pred = rho if otype == "range" else rho_dot
                if otype == "range" and use_joint:
                    y_pred += x[n_rv + sta_idx]

                H_local = _obs_jacobian_wrt_rv(r_sat, v_sat, r_sta, v_sta)
                H_rv = H_local[0] if otype == "range" else H_local[1]
                H_t0 = H_rv @ stm

                H_full = np.zeros(n_state)
                H_full[:n_rv] = H_t0
                if otype == "range" and use_joint:
                    H_full[n_rv + sta_idx] = 1.0

            elif otype == "dor":
                sta_i, sta_j = entry[1], entry[2]
                r_sta_i = _station_pos_eci(stations_geo[sta_i], t)
                r_sta_j = _station_pos_eci(stations_geo[sta_j], t)
                v_sta_i = _station_velocity_eci(stations_geo[sta_i], t)
                v_sta_j = _station_velocity_eci(stations_geo[sta_j], t)
                rho_i, _ = range_and_rate(r_sat, v_sat, r_sta_i, v_sta_i)
                rho_j, _ = range_and_rate(r_sat, v_sat, r_sta_j, v_sta_j)
                y_pred = rho_i - rho_j

                H_i = _obs_jacobian_wrt_rv(r_sat, v_sat, r_sta_i, v_sta_i)[0]
                H_j = _obs_jacobian_wrt_rv(r_sat, v_sat, r_sta_j, v_sta_j)[0]
                H_rv = H_i - H_j
                H_t0 = H_rv @ stm

                H_full = np.zeros(n_state)
                H_full[:n_rv] = H_t0
            else:
                continue

            res = val - y_pred
            w = 1.0 / (sigma * sigma)
            cost += w * res * res
            HTH += w * np.outer(H_full, H_full)
            HTz += w * H_full * res

        # LM 阻尼
        H_damped = HTH + lam * np.diag(np.diag(HTH) + 1e-12)
        try:
            dx = np.linalg.solve(H_damped, HTz)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(H_damped, HTz, rcond=None)[0]

        x_new = x + dx

        # 收敛检查 (至少迭代3次)
        if (it >= 3 and
                np.linalg.norm(dx[:3]) < tol_pos and
                np.linalg.norm(dx[3:6]) < tol_pos * 1e-3):
            x = x_new
            db_str = f" |db|={np.linalg.norm(dx[n_rv:]):.3e}" if use_joint else ""
            print(f"  [iter {it:2d}] 收敛 |dr|={np.linalg.norm(dx[:3]):.3e}"
                  f" |dv|={np.linalg.norm(dx[3:6]):.3e}{db_str}")
            break

        # 新代价
        new_cost = 0.0
        for entry in obs:
            t = entry[0]
            otype = entry[3]
            val = entry[4]
            sigma = entry[5]
            dt = t - t0_j2000

            r_sat, v_sat = propagate_numerical(
                x_new[:3], x_new[3:6], t0_j2000, dt, sc_params)

            if otype in ("range", "rate"):
                sta_idx = entry[2]
                r_sta = _station_pos_eci(stations_geo[sta_idx], t)
                v_sta = _station_velocity_eci(stations_geo[sta_idx], t)
                rho, rho_dot = range_and_rate(r_sat, v_sat, r_sta, v_sta)
                y_pred = rho if otype == "range" else rho_dot
                if otype == "range" and use_joint:
                    y_pred += x_new[n_rv + sta_idx]
            elif otype == "dor":
                sta_i, sta_j = entry[1], entry[2]
                r_sta_i = _station_pos_eci(stations_geo[sta_i], t)
                r_sta_j = _station_pos_eci(stations_geo[sta_j], t)
                v_sta_i = _station_velocity_eci(stations_geo[sta_i], t)
                v_sta_j = _station_velocity_eci(stations_geo[sta_j], t)
                rho_i, _ = range_and_rate(r_sat, v_sat, r_sta_i, v_sta_i)
                rho_j, _ = range_and_rate(r_sat, v_sat, r_sta_j, v_sta_j)
                y_pred = rho_i - rho_j
            else:
                continue

            w = 1.0 / (sigma * sigma)
            new_cost += w * (val - y_pred) ** 2

        if new_cost < prev_cost:
            x = x_new
            lam *= 0.3
            prev_cost = new_cost
            print(f"  [iter {it:2d}] cost={new_cost:12.6e}"
                  f" |dr|={np.linalg.norm(dx[:3]):.3e} lam={lam:.2e}")
        else:
            lam *= 3.0
            print(f"  [iter {it:2d}] rejected, lam={lam:.2e}")

    # 最终协方差
    HTH = np.zeros((n_state, n_state))
    residuals = []
    for entry in obs:
        t = entry[0]
        otype = entry[3]
        val = entry[4]
        sigma = entry[5]
        dt = t - t0_j2000

        r_sat, v_sat = propagate_numerical(
            x[:3], x[3:6], t0_j2000, dt, sc_params)
        stm = _kepler_stm(x[:3], x[3:6], dt)

        if otype in ("range", "rate"):
            sta_idx = entry[2]
            r_sta = _station_pos_eci(stations_geo[sta_idx], t)
            v_sta = _station_velocity_eci(stations_geo[sta_idx], t)
            rho, rho_dot = range_and_rate(r_sat, v_sat, r_sta, v_sta)
            y_pred = rho if otype == "range" else rho_dot
            if otype == "range" and use_joint:
                y_pred += x[n_rv + sta_idx]

            H_local = _obs_jacobian_wrt_rv(r_sat, v_sat, r_sta, v_sta)
            H_rv = H_local[0] if otype == "range" else H_local[1]
            H_t0 = H_rv @ stm

            H_full = np.zeros(n_state)
            H_full[:n_rv] = H_t0
            if otype == "range" and use_joint:
                H_full[n_rv + sta_idx] = 1.0

        elif otype == "dor":
            sta_i, sta_j = entry[1], entry[2]
            r_sta_i = _station_pos_eci(stations_geo[sta_i], t)
            r_sta_j = _station_pos_eci(stations_geo[sta_j], t)
            v_sta_i = _station_velocity_eci(stations_geo[sta_i], t)
            v_sta_j = _station_velocity_eci(stations_geo[sta_j], t)
            rho_i, _ = range_and_rate(r_sat, v_sat, r_sta_i, v_sta_i)
            rho_j, _ = range_and_rate(r_sat, v_sat, r_sta_j, v_sta_j)
            y_pred = rho_i - rho_j

            H_i = _obs_jacobian_wrt_rv(r_sat, v_sat, r_sta_i, v_sta_i)[0]
            H_j = _obs_jacobian_wrt_rv(r_sat, v_sat, r_sta_j, v_sta_j)[0]
            H_rv = H_i - H_j
            H_t0 = H_rv @ stm

            H_full = np.zeros(n_state)
            H_full[:n_rv] = H_t0
        else:
            continue

        residuals.append(val - y_pred)
        w = 1.0 / (sigma * sigma)
        HTH += w * np.outer(H_full, H_full)

    try:
        P = np.linalg.inv(HTH)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(HTH)

    bias_hat = x[n_rv:] if use_joint else None
    return x[:n_rv], P, np.array(residuals), bias_hat


# ==============================================================
# 结果报告
# ==============================================================
def _report_results(label: str, x_hat: np.ndarray, P_rv: np.ndarray,
                    true_elem: OrbitElements, t0_j2000: float,
                    sc_params: SpacecraftParams) -> Tuple[np.ndarray, float, float]:
    r0_true, v0_true = true_elem.to_cartesian()
    r_hat, v_hat = x_hat[:3], x_hat[3:]

    r_true, v_true = propagate_numerical(
        r0_true, v0_true, t0_j2000, 0, sc_params)

    elem_hat = cartesian_to_elements(r_hat, v_hat)
    names = ["a(km)", "e", "i(°)", "Ω(°)", "ω(°)", "f(°)"]
    vals_true = [true_elem.a / 1e3, true_elem.e,
                 true_elem.i * RAD2DEG, true_elem.Om * RAD2DEG,
                 true_elem.w * RAD2DEG, true_elem.f * RAD2DEG]
    vals_hat = [elem_hat.a / 1e3, elem_hat.e,
                elem_hat.i * RAD2DEG, elem_hat.Om * RAD2DEG,
                elem_hat.w * RAD2DEG, elem_hat.f * RAD2DEG]

    def rv_to_elem_arr(x):
        el = cartesian_to_elements(x[:3], x[3:])
        return np.array([el.a / 1e3, el.e,
                         el.i * RAD2DEG, el.Om * RAD2DEG,
                         el.w * RAD2DEG, el.f * RAD2DEG])

    G = np.zeros((6, 6))
    eps = [1e2, 1e2, 1e2, 1e-2, 1e-2, 1e-2]
    for j in range(6):
        x_p = x_hat.copy(); x_p[j] += eps[j]
        x_m = x_hat.copy(); x_m[j] -= eps[j]
        G[:, j] = (rv_to_elem_arr(x_p) - rv_to_elem_arr(x_m)) / (2 * eps[j])

    P_elem = G @ P_rv @ G.T
    sigma_elem = np.sqrt(np.maximum(np.diag(P_elem), 0))

    print(f"\n  --- {label} ---")
    print(f"  {'参数':>8s} {'真值':>14s} {'估计':>14s} {'偏差':>14s} {'1σ':>14s}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    for name, vt, vh, s in zip(names, vals_true, vals_hat, sigma_elem):
        diff = vh - vt
        print(f"  {name:>8s} {vt:14.6f} {vh:14.6f} {diff:+14.6f} {s:14.6f}")

    pos_err = np.linalg.norm(r_hat - r_true)
    vel_err = np.linalg.norm(v_hat - v_true)
    print(f"  |Δr| = {pos_err:10.3f} m   |Δv| = {vel_err:10.4f} m/s")
    return sigma_elem, pos_err, vel_err


# ==============================================================
# 主程序
# ==============================================================
def main():
    print("=" * 72)
    print(" 深空探测器轨道确定 — 多弧段 + 动力学补偿")
    print(" 模型: 二体 + J2 + 日月摄动 + 太阳光压")
    print(" 方法: RK4 数值积分 + 批处理最小二乘 + 开普勒STM")
    print("=" * 72)

    # —— 真值轨道 ——
    true_elem = OrbitElements(
        a=6928137.0, e=0.002, i=51.6 * DEG2RAD,
        Om=120.0 * DEG2RAD, w=70.0 * DEG2RAD, f=0.0
    )
    r0_true, v0_true = true_elem.to_cartesian()
    print(f"\n[真值] a={true_elem.a/1e3:.1f} km  e={true_elem.e:.4f}"
          f"  i={true_elem.i*RAD2DEG:.1f}°"
          f"  Ω={true_elem.Om*RAD2DEG:.1f}°"
          f"  ω={true_elem.w*RAD2DEG:.1f}°"
          f"  f={true_elem.f*RAD2DEG:.1f}°")

    sc_params = SpacecraftParams(area_mass_ratio=0.01, reflectivity=1.2)
    t0_j2000 = 0.0

    # —— 多弧段配置 (3个弧段, 同一天内) ——
    period_sec = 2 * np.pi * np.sqrt(true_elem.a ** 3 / MU_EARTH)
    arc_specs = [
        {"t_start": 0.0,                    "t_end": 0.0 + 1.0 * 3600,          "n_obs": 4},
        {"t_start": 0.3 * period_sec,       "t_end": 0.3 * period_sec + 1.0 * 3600, "n_obs": 4},
        {"t_start": 0.6 * period_sec,       "t_end": 0.6 * period_sec + 1.0 * 3600, "n_obs": 4},
    ]

    stations_geo = [
        (39.9042 * DEG2RAD, 116.4074 * DEG2RAD, 50.0),
        (-33.8688 * DEG2RAD, 151.2093 * DEG2RAD, 50.0),
        (40.4314 * DEG2RAD, -86.9135 * DEG2RAD, 220.0),
    ]
    n_sta = len(stations_geo)

    # —— 系统偏差 ——
    range_biases_true = np.array([80.0, -50.0, 30.0])
    clock_bias_true = 300.0

    print(f"\n[注入] 测站偏差 (m): {range_biases_true}"
          f"  钟差 (m): {clock_bias_true}")
    print(f"[观测] {len(arc_specs)} 个弧段 × {n_sta} 站 × 2 (range+rate)")

    obs = simulate_multi_arc_observations(
        true_elem, t0_j2000, arc_specs, stations_geo, sc_params,
        sigma_rho=2.0, sigma_rhodot=5e-4,
        range_biases=range_biases_true, clock_bias=clock_bias_true
    )
    print(f"[观测] 总数: {len(obs)}")

    # —— 初始猜测 ——
    r0_guess = r0_true + np.array([50.0, -30.0, 20.0])
    v0_guess = v0_true + np.array([0.05, -0.03, 0.02])
    x0_guess = np.concatenate([r0_guess, v0_guess])
    bias_guess = np.zeros(n_sta)

    # —— 方案1: 标准方法 ——
    print("\n" + "=" * 72)
    print(" 方案 1: 标准批处理最小二乘 (无偏差处理)")
    print("=" * 72)
    x_std, P_std, res_std, _ = batch_lsq_numerical(
        obs, x0_guess, t0_j2000, stations_geo, sc_params,
        use_joint=False, use_dor=False
    )
    _, pos_err_std, vel_err_std = _report_results(
        "标准方法", x_std, P_std, true_elem, t0_j2000, sc_params)

    # —— 方案2: 联合估计 ——
    print("\n" + "=" * 72)
    print(" 方案 2: 联合估计 (追加每测站测距偏差)")
    print("=" * 72)
    x_joint, P_joint, res_joint, bias_hat = batch_lsq_numerical(
        obs, x0_guess, t0_j2000, stations_geo, sc_params,
        bias_guess=bias_guess, use_joint=True, use_dor=False
    )
    _, pos_err_joint, vel_err_joint = _report_results(
        "联合估计", x_joint, P_joint[:6, :6], true_elem, t0_j2000, sc_params)

    if bias_hat is not None:
        bias_true_all = range_biases_true + clock_bias_true
        print(f"\n  偏差参数:")
        print(f"  {'测站':>6s} {'真值':>12s} {'估计':>12s} {'误差':>12s}")
        for i, (bt, bh) in enumerate(zip(bias_true_all, bias_hat)):
            print(f"  {'S'+str(i):>6s} {bt:12.3f} {bh:12.3f} {bh-bt:+12.3f}")

    # —— 方案3: DOR差分 ——
    print("\n" + "=" * 72)
    print(" 方案 3: DOR 差分观测 (消除公共偏差)")
    print("=" * 72)
    x_dor, P_dor, res_dor, _ = batch_lsq_numerical(
        obs, x0_guess, t0_j2000, stations_geo, sc_params,
        use_joint=False, use_dor=True
    )
    _, pos_err_dor, vel_err_dor = _report_results(
        "DOR差分", x_dor, P_dor, true_elem, t0_j2000, sc_params)

    # —— 方案4: DOR + 联合估计 ——
    print("\n" + "=" * 72)
    print(" 方案 4: DOR + 联合估计 (最佳)")
    print("=" * 72)
    x_best, P_best, res_best, bias_hat_best = batch_lsq_numerical(
        obs, x0_guess, t0_j2000, stations_geo, sc_params,
        bias_guess=bias_guess, use_joint=True, use_dor=True
    )
    _, pos_err_best, vel_err_best = _report_results(
        "DOR+联合估计", x_best, P_best[:6, :6],
        true_elem, t0_j2000, sc_params)

    # —— 汇总 ——
    print("\n" + "=" * 72)
    print(" 结果汇总")
    print("=" * 72)
    print(f"  {'方案':<20s} {'|Δr| (m)':>12s} {'|Δv| (m/s)':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    print(f"  {'标准方法':<20s} {pos_err_std:12.3f} {vel_err_std:12.4f}")
    print(f"  {'联合估计':<20s} {pos_err_joint:12.3f} {vel_err_joint:12.4f}")
    print(f"  {'DOR差分':<20s} {pos_err_dor:12.3f} {vel_err_dor:12.4f}")
    print(f"  {'DOR+联合估计':<20s} {pos_err_best:12.3f} {vel_err_best:12.4f}")

    # 精度分析
    print(f"\n  精度分析:")
    print(f"    - 多弧段数据: {len(arc_specs)} 天 × 3测站, "
          f"共 {len(obs)} 条观测")
    print(f"    - 动力学模型: 二体 + J2 + 日月 + SRP")
    print(f"    - 目标精度: 米级 (Δr < 10m)")
    print(f"    - 最佳方案精度: |Δr| = {pos_err_best:.3f} m")

    if pos_err_best < 10:
        print(f"    ✓ 达到米级精度目标!")
    else:
        print(f"    ⚠ 未达到米级精度, 需增加观测数据或优化动力学模型")


if __name__ == "__main__":
    main()
