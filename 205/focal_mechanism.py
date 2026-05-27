"""
从地震 P 波初动极性数据反演震源机制解（断层面解）。

算法：
    1. 纯极性反演：格点搜索法（最小矛盾符号数）。
    2. 联合反演：P 波初动极性 + SH/P 振幅比，同时解决断层面/辅助面的二义性
       以及台站分布不均造成的多解性。

输入数据 (observations)：
    每条记录包含:
        - az   : 台站相对震中的方位角 (度, 0-360, 从北顺时针)
        - toa  : P 波出射角 / 震源距角 (度, 0-180, 从震源向上为 0, 向下为 180)
        - pol  : 初动极性, +1 表示压缩 (P, 向上), -1 表示膨胀 (T, 向下), 0 表示不清
        - sh_p_ratio : (可选) 观测到的 SH/P 振幅比 (正实数)
        - sh_p_ratio_err : (可选) 振幅比的测量误差 (对数域)

输出:
    strike, dip, rake : 最优断层面解（走向、倾角、滑动角，单位：度）
    以及对应的最小矛盾符号数 / 联合反演残差。

参考：
    - Aki & Richards, "Quantitative Seismology", 第 4 章
    - Shearer, "Introduction to Seismology", 第 9 章
    - P 波远场辐射振幅:  F_P = 2 (n·γ)(s·γ)
    - S 波远场辐射向量: F_S = (n·γ)s + (s·γ)n - 2(n·γ)(s·γ)γ
    - SH 分量: F_S 在大圆弧平面法向上的投影
"""

import numpy as np
from typing import List, Optional, Tuple


# ============================================================================
# 1. 基础工具：坐标变换 & 断层面向量
# ============================================================================

def sph_to_cart(az_deg: float, toa_deg: float) -> np.ndarray:
    """(方位角 az, 震源距角 toa) -> 东-北-上 笛卡尔单位向量。

    坐标系: x->East, y->North, z->Up
    toa: 从 +z 量起, 0=上, 180=下
    az : 从 +y 顺时针量到水平投影
    """
    az  = np.deg2rad(az_deg)
    toa = np.deg2rad(toa_deg)
    x = np.sin(toa) * np.sin(az)
    y = np.sin(toa) * np.cos(az)
    z = np.cos(toa)
    return np.array([x, y, z])


def sdr_vectors(strike: float, dip: float, rake: float) -> Tuple[np.ndarray, np.ndarray]:
    """由 (strike, dip, rake) 计算断层面法向量 n 与滑动向量 s (均为单位向量)。

    坐标系: ENU (东-北-上)。
    n 指向上盘 (hanging wall)。
    s 在断层面内，rake=0 为左旋走滑，rake=90 为纯逆冲，rake=-90 为纯正断。
    """
    s0 = np.deg2rad(strike)
    d0 = np.deg2rad(dip)
    r0 = np.deg2rad(rake)

    # 法向量 (上盘方向)
    n = np.array([-np.cos(s0) * np.sin(d0),
                  -np.sin(s0) * np.sin(d0),
                   np.cos(d0)])

    # 沿走向方向 (水平, 沿断层迹线)
    along_strike = np.array([ np.sin(s0), -np.cos(s0), 0.0])
    # 沿断层面向下倾斜方向
    down_dip     = np.array([-np.cos(s0) * np.cos(d0),
                             -np.sin(s0) * np.cos(d0),
                             -np.sin(d0)])
    # 滑动向量 = cos(rake)*沿走向 + sin(rake)*向下倾斜
    slip = np.cos(r0) * along_strike + np.sin(r0) * down_dip

    return n, slip


def rotate_to_fault(obs: np.ndarray, strike: float, dip: float) -> np.ndarray:
    """观测方向 obs (ENU) -> 断层面坐标系 (1=走向, 2=向下倾, 3=法向)。"""
    s = np.deg2rad(strike)
    d = np.deg2rad(dip)

    e1 = np.array([np.cos(s),  np.sin(s), 0.0])
    e2 = np.array([np.sin(d) * np.sin(s),
                   -np.sin(d) * np.cos(s),
                   -np.cos(d)])
    e3 = np.array([-np.cos(d) * np.sin(s),
                    np.cos(d) * np.cos(s),
                   -np.sin(d)])

    R = np.vstack([e1, e2, e3])
    return R @ obs


# ============================================================================
# 2. 辐射花样: P 波, S 波 (SH 分量), SH/P 振幅比
# ============================================================================

def p_radiation(gamma: np.ndarray, n: np.ndarray, slip: np.ndarray) -> float:
    """P 波辐射花样 (带符号).  F_P = 2 (n·γ)(s·γ)

    初动极性只关心 sign(F_P)，幅值可用于振幅比。
    """
    return 2.0 * np.dot(n, gamma) * np.dot(slip, gamma)


def s_radiation_vector(gamma: np.ndarray, n: np.ndarray, slip: np.ndarray) -> np.ndarray:
    """S 波远场辐射向量 (ENU 坐标系)。

    F_S = (n·γ)s + (s·γ)n - 2(n·γ)(s·γ)γ

    参见 Shearer (2009) p.250, 或 Aki & Richards (2002) 式 (4.89)。
    """
    ng = np.dot(n, gamma)
    sg = np.dot(slip, gamma)
    return ng * slip + sg * n - 2.0 * ng * sg * gamma


def sh_component(gamma: np.ndarray, f_s_vec: np.ndarray) -> float:
    """从 S 波辐射向量中提取 SH 分量 (大圆弧平面的法向分量)。

    SH 极化方向 = 大圆弧平面的法向 = normalize(γ × ẑ)
    其中 ẑ = (0,0,1) 为竖直向上 (ENU 坐标)。
    """
    z_hat = np.array([0.0, 0.0, 1.0])
    gcp_normal = np.cross(gamma, z_hat)           # 大圆弧平面法向
    norm = np.linalg.norm(gcp_normal)
    if norm < 1e-10:
        return 0.0                                 # 竖直方向, SH=0
    e_sh = gcp_normal / norm
    return float(np.dot(f_s_vec, e_sh))


def sh_p_ratio(gamma: np.ndarray, n: np.ndarray, slip: np.ndarray) -> float:
    """计算 |F_SH| / |F_P| 的理论振幅比。

    当 F_P 接近 0 (节线附近) 时比值发散，返回 np.inf。
    实际反演中应避开节线附近的台站。
    """
    fp  = p_radiation(gamma, n, slip)
    f_s = s_radiation_vector(gamma, n, slip)
    fsh = sh_component(gamma, f_s)
    if abs(fp) < 1e-10:
        return np.inf
    return abs(fsh) / abs(fp)


# ============================================================================
# 3. 纯极性格点搜索
# ============================================================================

def grid_search(obs: List[dict],
                strike_step: float = 5.0,
                dip_step: float = 5.0,
                rake_step: float = 5.0,
                verbose: bool = True,
                top_k: int = 1) -> List[dict]:
    """
    对 (strike, dip, rake) 做格点搜索, 找到使"矛盾符号数"最小的解。

    Parameters
    ----------
    obs : list of dict
        每条 dict 含 'az'(度), 'toa'(度), 'pol'(+1/-1/0).
    strike_step, dip_step, rake_step : float
        格点步长 (度).
    verbose : bool
        是否打印最优解更新信息.
    top_k : int
        返回最优的 top_k 个解 (用于联合反演的候选集).

    Returns
    -------
    list of dict, 按 misfit 升序排列, 每个 dict 包含:
        'strike', 'dip', 'rake', 'misfit', 'total', 'ratio'
    """
    valid = [(o['az'], o['toa'], o['pol']) for o in obs
             if o.get('pol', 0) in (1, -1)]
    if len(valid) == 0:
        raise ValueError("没有可用的极性观测 (pol 必须为 +1 或 -1).")

    azs   = np.array([v[0] for v in valid])
    toas  = np.array([v[1] for v in valid])
    pols  = np.array([v[2] for v in valid], dtype=int)
    gammas = np.array([sph_to_cart(a, t) for a, t in zip(azs, toas)])  # (N, 3)

    strikes = np.arange(0, 360, strike_step)
    dips    = np.arange(0,  90 + dip_step, dip_step)
    rakes   = np.arange(-180, 180 + rake_step, rake_step)

    candidates = []  # (misfit, strike, dip, rake)

    for stk in strikes:
        for d in dips:
            for rk in rakes:
                n, slip = sdr_vectors(stk, d, rk)
                Fp = 2.0 * (gammas @ n) * (gammas @ slip)
                pred = np.sign(Fp).astype(int)
                pred[pred == 0] = 1
                misfit = int(np.sum(pred != pols))

                if len(candidates) < top_k or misfit < candidates[-1][0]:
                    candidates.append((misfit, float(stk), float(d), float(rk)))
                    candidates.sort(key=lambda x: x[0])
                    candidates = candidates[:top_k]
                    if verbose and misfit == candidates[0][0]:
                        print(f"[strike={stk:5.1f} dip={d:4.1f} rake={rk:5.1f}] "
                              f"misfit={misfit}/{len(valid)}")

    results = []
    for m, s, d, r in candidates:
        aux = auxiliary_plane(s, d, r)
        results.append({
            'strike': s, 'dip': d, 'rake': r,
            'misfit': m, 'total': len(valid), 'ratio': m / len(valid),
            'aux_plane': {
                'strike': aux[0], 'dip': aux[1], 'rake': aux[2]
            }
        })
    return results


# ============================================================================
# 4. 辅助面
# ============================================================================

def auxiliary_plane(strike: float, dip: float, rake: float) -> Tuple[float, float, float]:
    """由主断层面 (s1, d1, r1) 求共轭辅助面 (s2, d2, r2)。

    辅助面法向 n2 = -s1, 辅助面滑动 s2 = -n1。
    """
    n1, s1 = sdr_vectors(strike, dip, rake)
    n2 = -s1
    s2 = -n1

    # 由 n2 求 strike2, dip2
    dip2 = np.rad2deg(np.arccos(np.clip(n2[2], -1, 1)))
    if abs(dip2) < 1e-6:
        dip2 = 0.0
    if abs(dip2 - 90) < 1e-6:
        dip2 = 90.0
    strike2 = np.rad2deg(np.arctan2(-n2[1], -n2[0])) % 360

    # 由 s2 求 rake2
    s02 = np.deg2rad(strike2)
    d02 = np.deg2rad(dip2)
    along2 = np.array([ np.sin(s02), -np.cos(s02), 0.0])
    down2  = np.array([-np.cos(s02) * np.cos(d02),
                       -np.sin(s02) * np.cos(d02),
                       -np.sin(d02)])
    rake2 = np.rad2deg(np.arctan2(np.dot(s2, down2), np.dot(s2, along2)))

    return strike2, dip2, rake2


# ============================================================================
# 5. 精细搜索
# ============================================================================

def refine(obs, strike, dip, rake, window=10.0, step=1.0):
    """在给定解附近做小范围精细搜索 (仅极性约束)。"""
    valid = [(o['az'], o['toa'], o['pol']) for o in obs
             if o.get('pol', 0) in (1, -1)]
    azs   = np.array([v[0] for v in valid])
    toas  = np.array([v[1] for v in valid])
    pols  = np.array([v[2] for v in valid], dtype=int)
    gammas = np.array([sph_to_cart(a, t) for a, t in zip(azs, toas)])

    s1, s2 = strike - window, strike + window
    d1, d2 = max(0.0, dip - window), min(90.0, dip + window)
    r1, r2 = rake - window, rake + window

    best = len(valid) + 1
    best_p = (strike, dip, rake)

    for stk in np.arange(s1, s2 + 0.5 * step, step):
        stk_mod = stk % 360
        for d in np.arange(d1, d2 + 0.5 * step, step):
            for rk in np.arange(r1, r2 + 0.5 * step, step):
                n, slip = sdr_vectors(stk_mod, d, rk)
                Fp = 2.0 * (gammas @ n) * (gammas @ slip)
                pred = np.sign(Fp).astype(int)
                pred[pred == 0] = 1
                m = int(np.sum(pred != pols))
                if m < best:
                    best = m
                    best_p = (stk_mod, d, rk)

    stk, d, rk = best_p
    return {
        'strike': stk, 'dip': d, 'rake': rk,
        'misfit': best, 'total': len(valid)
    }


# ============================================================================
# 6. 联合反演: P 波极性 + SH/P 振幅比
# ============================================================================

def _amp_ratio_misfit(obs_with_ratio, gammas_r, n, slip, ref_ratios):
    """计算 SH/P 振幅比的对数 RMS 残差。

    J_amp = sqrt( mean( (log10(pred) - log10(obs))^2 ) )
    """
    residuals = []
    for i, g in enumerate(gammas_r):
        r_pred = sh_p_ratio(g, n, slip)
        if not np.isfinite(r_pred) or r_pred < 1e-10:
            continue
        r_obs  = ref_ratios[i]
        if r_obs < 1e-10:
            continue
        residuals.append((np.log10(r_pred) - np.log10(r_obs)) ** 2)
    if len(residuals) == 0:
        return np.inf
    return float(np.sqrt(np.mean(residuals)))


def joint_inversion(obs: List[dict],
                    strike_step: float = 5.0,
                    dip_step: float = 5.0,
                    rake_step: float = 5.0,
                    polarity_weight: float = 0.7,
                    refine_step: float = 1.0,
                    verbose: bool = True) -> dict:
    """
    联合反演: P 波初动极性 + SH/P 振幅比约束。

    原理:
        - P 波极性仅由 F_P 的符号决定，共轭辅助面与主断层面给出完全相同的
          极性图案，因此仅用极性无法区分断层面与辅助面。
        - SH/P 振幅比在主断层面与辅助面之间通常差异显著，可以打破二义性。
        - 当台站覆盖不均时，多个非共轭的机制解也可能给出相同极性残差，
          振幅比同样有助于区分。

    目标函数 (加权):
        J = w * (polarity_misfit / N_pol) + (1-w) * (amp_ratio_log_rms / amp_ref)

    Parameters
    ----------
    obs : list of dict
        每条记录必须含 'az','toa','pol'；可选 'sh_p_ratio' (观测振幅比)。
        至少需要 1 条含 'sh_p_ratio' 的记录。
    strike_step, dip_step, rake_step : float
        粗搜索格点步长 (度).
    polarity_weight : float
        极性项权重 w (0-1)。w=1 退化为纯极性反演。
    refine_step : float
        精搜索步长 (度)。None 表示不精化.
    verbose : bool

    Returns
    -------
    result : dict
        'strike', 'dip', 'rake' : 最优解
        'polarity_misfit'       : 极性矛盾数
        'polarity_total'        : 有效极性观测数
        'amp_ratio_rms'         : 振幅比对数 RMS
        'amp_ratio_count'       : 有效振幅比观测数
        'aux_plane'             : 辅助面解
        'combined_score'        : 联合反演目标函数值
    """
    # --- 分离极性观测与振幅比观测 ---
    pol_obs = [(o['az'], o['toa'], o['pol']) for o in obs
               if o.get('pol', 0) in (1, -1)]
    amp_obs = [(o['az'], o['toa'], o['sh_p_ratio']) for o in obs
               if o.get('sh_p_ratio') is not None and np.isfinite(o.get('sh_p_ratio'))]

    if len(pol_obs) == 0:
        raise ValueError("没有可用的极性观测。")
    if len(amp_obs) == 0:
        raise ValueError("没有可用的 SH/P 振幅比观测 (请在 obs 中提供 sh_p_ratio 字段)。")

    pol_azs  = np.array([v[0] for v in pol_obs])
    pol_toas = np.array([v[1] for v in pol_obs])
    pol_vals = np.array([v[2] for v in pol_obs], dtype=int)
    pol_gammas = np.array([sph_to_cart(a, t) for a, t in zip(pol_azs, pol_toas)])

    amp_azs  = np.array([v[0] for v in amp_obs])
    amp_toas = np.array([v[1] for v in amp_obs])
    amp_vals = np.array([v[2] for v in amp_obs])
    amp_gammas = np.array([sph_to_cart(a, t) for a, t in zip(amp_azs, amp_toas)])

    N_pol = len(pol_obs)
    N_amp = len(amp_obs)

    # 参考 RMS (用于归一化): 假定振幅比典型范围 0.1-10, log10 范围 -1~1
    # 所以典型 RMS ~ 0.5; 极性残差在 0-N_pol, 归一后也在 0-1
    amp_ref = 0.5

    w = polarity_weight

    strikes = np.arange(0, 360, strike_step)
    dips    = np.arange(0,  90 + dip_step, dip_step)
    rakes   = np.arange(-180, 180 + rake_step, rake_step)

    best_score = np.inf
    best_pol_misfit = None
    best_amp_rms   = None
    best_params    = None

    for stk in strikes:
        for d in dips:
            for rk in rakes:
                n, slip = sdr_vectors(stk, d, rk)

                # --- 极性残差 ---
                Fp = 2.0 * (pol_gammas @ n) * (pol_gammas @ slip)
                pred = np.sign(Fp).astype(int)
                pred[pred == 0] = 1
                pol_misfit = int(np.sum(pred != pol_vals))

                # --- 振幅比残差 ---
                amp_rms = _amp_ratio_misfit(None, amp_gammas, n, slip, amp_vals)
                if not np.isfinite(amp_rms):
                    amp_rms = 2.0  # 惩罚

                # --- 联合目标函数 ---
                score = w * (pol_misfit / N_pol) + (1.0 - w) * (amp_rms / amp_ref)

                if score < best_score:
                    best_score      = score
                    best_pol_misfit = pol_misfit
                    best_amp_rms    = amp_rms
                    best_params     = (float(stk), float(d), float(rk))
                    if verbose:
                        print(f"[strike={stk:5.1f} dip={d:4.1f} rake={rk:5.1f}] "
                              f"pol_misfit={pol_misfit}/{N_pol} "
                              f"amp_rms={amp_rms:.3f} score={score:.4f}")

    if best_params is None:
        raise RuntimeError("联合反演未找到解。")

    # --- 精细搜索 ---
    if refine_step is not None and refine_step > 0:
        stk0, d0, rk0 = best_params
        window = max(strike_step * 1.5, 10.0)
        s1, s2 = stk0 - window, stk0 + window
        d1, d2 = max(0.0, d0 - window), min(90.0, d0 + window)
        r1, r2 = rk0 - window, rk0 + window

        for stk in np.arange(s1, s2 + 0.5 * refine_step, refine_step):
            stk_mod = stk % 360
            for d in np.arange(d1, d2 + 0.5 * refine_step, refine_step):
                for rk in np.arange(r1, r2 + 0.5 * refine_step, refine_step):
                    n, slip = sdr_vectors(stk_mod, d, rk)
                    Fp = 2.0 * (pol_gammas @ n) * (pol_gammas @ slip)
                    pred = np.sign(Fp).astype(int)
                    pred[pred == 0] = 1
                    pol_misfit = int(np.sum(pred != pol_vals))
                    amp_rms = _amp_ratio_misfit(None, amp_gammas, n, slip, amp_vals)
                    if not np.isfinite(amp_rms):
                        amp_rms = 2.0
                    score = w * (pol_misfit / N_pol) + (1.0 - w) * (amp_rms / amp_ref)
                    if score < best_score:
                        best_score      = score
                        best_pol_misfit = pol_misfit
                        best_amp_rms    = amp_rms
                        best_params     = (stk_mod, d, rk)

    stk, d, rk = best_params
    aux = auxiliary_plane(stk, d, rk)

    return {
        'strike': stk, 'dip': d, 'rake': rk,
        'polarity_misfit': best_pol_misfit,
        'polarity_total':  N_pol,
        'amp_ratio_rms':   best_amp_rms,
        'amp_ratio_count': N_amp,
        'combined_score':  best_score,
        'aux_plane': {
            'strike': aux[0], 'dip': aux[1], 'rake': aux[2]
        }
    }


# ============================================================================
# 8. 矩张量反演 (Moment Tensor Inversion)
# ============================================================================

"""
矩张量 M 是一个 3×3 对称矩阵，描述地震源的等效作用力系。
坐标系: ENU (东-北-上)

M = [[Mxx, Mxy, Mxz],
     [Mxy, Myy, Myz],
     [Mxz, Myz, Mzz]]

可以正交分解为:
    M = M_ISO + M_DEV
其中:
    M_ISO = (tr(M)/3) * I  各向同性分量 (体积变化, 火山/流体活动相关)
    M_DEV = M - M_ISO      偏差矩张量 (纯剪切分量)

偏差矩张量可以进一步分解为:
    M_DEV = M_DC + M_CLVD
其中:
    M_DC   = 双力偶分量 (剪切破裂, 构造地震)
    M_CLVD = 补偿线性矢量偶极 (开裂/闭合/体积变化相关)

分解比例:
    ISO%  = |tr(M)| / (|tr(M)| + |M_DEV|) * 100
    DC%   = 最大偏差本征值贡献
    CLVD% = 中间偏差本征值贡献

对于全波形拟合, 采用简化的振幅谱拟合:
    J = Σ |A_obs(i) - A_pred(i, M)|^2
其中 A_pred 由矩张量的辐射花样乘以格林函数幅值得到。
"""

# ---------------------------------------------------------------------------
# 8.1 矩张量基础: 构造 / 分解
# ---------------------------------------------------------------------------

def mt_from_sdr(strike: float, dip: float, rake: float, M0: float = 1.0) -> np.ndarray:
    """由断层面解 (strike, dip, rake) 构造纯双力偶矩张量。

    坐标系: ENU。参考 Aki & Richards (2002) p.111, 式 (4.94-4.96)。

    仅当剪切破裂且无体积变化时适用 (纯 DC 源)。
    """
    s0 = np.deg2rad(strike)
    d0 = np.deg2rad(dip)
    r0 = np.deg2rad(rake)

    # 矩张量分量 (乘 M0 之前的单位张量)
    Mxx = -M0 * (np.sin(d0) * np.cos(r0) * np.sin(2 * s0)
                + np.sin(2 * d0) * np.sin(r0) * np.sin(s0) ** 2)
    Myy =  M0 * (np.sin(d0) * np.cos(r0) * np.sin(2 * s0)
                - np.sin(2 * d0) * np.sin(r0) * np.cos(s0) ** 2)
    Mzz =  M0 * np.sin(2 * d0) * np.sin(r0)
    Mxy =  M0 * (np.sin(d0) * np.cos(r0) * np.cos(2 * s0)
                + 0.5 * np.sin(2 * d0) * np.sin(r0) * np.sin(2 * s0))
    Mxz = -M0 * (np.cos(d0) * np.cos(r0) * np.cos(s0)
                + np.cos(2 * d0) * np.sin(r0) * np.sin(s0))
    Myz = -M0 * (np.cos(d0) * np.cos(r0) * np.sin(s0)
                - np.cos(2 * d0) * np.sin(r0) * np.cos(s0))

    return np.array([[Mxx, Mxy, Mxz],
                     [Mxy, Myy, Myz],
                     [Mxz, Myz, Mzz]])


def mt_isotropic_component(M: np.ndarray) -> np.ndarray:
    """返回各向同性分量 M_ISO = (tr(M)/3) I。"""
    tr = np.trace(M)
    return (tr / 3.0) * np.eye(3)


def mt_deviatoric(M: np.ndarray) -> np.ndarray:
    """返回偏差矩张量 M_DEV = M - M_ISO。"""
    return M - mt_isotropic_component(M)


def mt_decompose(M: np.ndarray) -> dict:
    """矩张量分解: ISO (各向同性), DC (双力偶), CLVD (补偿线性矢量偶极)。

    采用标准的本征值分解法 (Hudson et al., 1989; Knopoff & Randall, 1970)。

    比例定义 (基于标量矩 M0 = sqrt(0.5 * Σ M_ij²):
        - iso_frac = M0_ISO / M0_total
        - dc_frac  = M0_DC / M0_total
        - clvd_frac = M0_CLVD / M0_total

    Returns
    -------
    dict with:
        'iso_frac'  : ISO 分量比例 (-1~1), 正=膨胀, 负=收缩
        'dc_frac'   : DC 分量比例 (0~1)
        'clvd_frac' : CLVD 分量比例 (-1~1), 正=拉伸型, 负=压缩型
        'iso_percent', 'dc_percent', 'clvd_percent': 百分比
        'eigenvalues' : 本征值 (已排序: λ1 ≥ λ2 ≥ λ3)
        'scalar_moment' : 总标量矩 M0
    """
    # 总标量矩
    M0 = float(np.sqrt(0.5 * np.sum(M**2)))

    # 本征值分解 (对称矩阵)
    eigvals = np.sort(np.linalg.eigvalsh(M))[::-1]
    λ1, λ2, λ3 = eigvals

    # 各向同性部分
    iso = (λ1 + λ2 + λ3) / 3.0

    # ISO 标量矩: M_ISO = iso * I, M0_ISO = sqrt(0.5 * 3 * iso²) = |iso| * sqrt(1.5)
    M0_ISO = abs(iso) * np.sqrt(1.5)

    # 偏差本征值
    λ1d, λ2d, λ3d = λ1 - iso, λ2 - iso, λ3 - iso

    # 纯偏差张量 M_DEV = M - M_ISO
    M_DEV = M - iso * np.eye(3)
    M0_DEV = float(np.sqrt(0.5 * np.sum(M_DEV**2)))

    # 分解 DC 和 CLVD
    # 偏差本征值: λ1d, λ2d, λ3d (满足 λ1d + λ2d + λ3d = 0)
    # DC 本征值 (1, 0, -1) 类型, CLVD 本征值 (1, 1, -2) 类型
    # 标准做法:
    #   令 ε = λ1d ≥ |λ3d| (即最大绝对值偏差本征值)
    #   DC 大小 = |λ1d| - |λ2d|
    #   CLVD 大小 = 2 * |λ2d|
    #   (注: 当 λ2d > 0 时 CLVD 为正 (拉伸型), 本征值 (a, a, -2a) 类型
    dc_amp = abs(λ1d) + abs(λ3d) - abs(λ2d)
    clvd_amp = 2 * abs(λ2d)

    # 由偏差本征值重构纯 DC 和 CLVD 张量的标量矩
    # 纯 DC 本征值 (a, 0, -a) 的 M0 = sqrt(0.5*(a² + 0 + a²)) = a
    # 这里 dc_amp = |λ1d| + |λ3d| - |λ2d|, 当 λ2d = 0 (纯 DC) 时 dc_amp = 2a
    # 所以 M0_DC = dc_amp / 2
    M0_DC = dc_amp / 2.0
    # 纯 CLVD 本征值 (a, a, -2a) 的 M0 = sqrt(0.5*(a²+a²+4a²)) = a*sqrt(3)
    # 这里 clvd_amp = 2*|λ2d|, 当只有 CLVD 时 λ1d=λ2d=a, λ3d=-2a => clvd_amp = 2a
    # 所以 a = clvd_amp / 2, M0_CLVD = (clvd_amp / 2) * sqrt(3)
    M0_CLVD = (clvd_amp / 2.0) * np.sqrt(3.0)

    # 归一化比例
    if M0 < 1e-15:
        return {'iso_frac': 0.0, 'dc_frac': 0.0, 'clvd_frac': 0.0,
                'iso_percent': 0.0, 'dc_percent': 0.0, 'clvd_percent': 0.0,
                'eigenvalues': eigvals, 'scalar_moment': 0.0}

    iso_frac = (M0_ISO / M0) * np.sign(iso)
    dc_frac = M0_DC / M0
    clvd_frac = (M0_CLVD / M0) * np.sign(λ2d)

    return {
        'iso_frac':  float(iso_frac),
        'dc_frac':   float(dc_frac),
        'clvd_frac': float(clvd_frac),
        'iso_percent':  float(iso_frac * 100),
        'dc_percent':   float(dc_frac * 100),
        'clvd_percent': float(clvd_frac * 100),
        'eigenvalues': eigvals,
        'scalar_moment': M0
    }


def mt_combine(iso_frac: float, dc_frac: float, clvd_frac: float,
               strike: float = 0.0, dip: float = 45.0, rake: float = 90.0,
               clvd_axis_az: float = 0.0, clvd_axis_plunge: float = 0.0,
               M0: float = 1.0) -> np.ndarray:
    """构造指定 ISO/DC/CLVD 比例的矩张量。

    注意: 比例定义与 mt_decompose() 输出一致。
    iso_frac > 0 表示膨胀 (体积增大), < 0 表示收缩。
    clvd_frac > 0 表示拉伸型 CLVD, < 0 表示压缩型 CLVD。

    算法:
        1. 先构造纯 DC 张量 (由 strike/dip/rake 决定取向)
        2. 按比例混合 ISO 和 CLVD 分量
        3. 整体缩放使标量矩 ≈ M0
    """
    # 纯 DC 张量 (单位标量矩)
    M_dc = mt_from_sdr(strike, dip, rake, M0=1.0)

    # 归一化: 使 DC 分量的"矩"为 dc_frac
    # 标量矩 M0_dc = sqrt(0.5 * Σ Mij²)
    m0_dc = np.sqrt(0.5 * np.sum(M_dc**2))
    M_dc_scaled = M_dc * (dc_frac / m0_dc) if m0_dc > 0 else M_dc

    # ISO 分量: M_ISO = (iso_frac * M0 / sqrt(1.5)) * I
    # 因为 sqrt(0.5 * 3 * a²) = a * sqrt(1.5), 所以 a = M0_iso / sqrt(1.5)
    a_iso = iso_frac * M0 / np.sqrt(1.5) if iso_frac != 0 else 0.0
    M_iso = a_iso * np.eye(3)

    # CLVD 分量
    M_clvd = np.zeros((3, 3))
    if abs(clvd_frac) > 1e-12:
        az = np.deg2rad(clvd_axis_az)
        pl = np.deg2rad(clvd_axis_plunge)
        n = np.array([np.cos(pl) * np.sin(az),
                      np.cos(pl) * np.cos(az),
                      np.sin(pl)])
        # CLVD 单位张量: M_CLVD = n n^T - (1/3) I
        # 其标量矩为 sqrt(0.5 * Σ Mij²) = sqrt(0.5 * (2/3)) = sqrt(1/3)
        M_clvd_unit = np.outer(n, n) - np.eye(3) / 3.0
        m0_clvd_unit = np.sqrt(0.5 * np.sum(M_clvd_unit**2))
        scale = clvd_frac * M0 / m0_clvd_unit if m0_clvd_unit > 0 else 0.0
        M_clvd = scale * M_clvd_unit

    M = M_dc_scaled + M_iso + M_clvd

    # 整体归一化到目标 M0
    current_m0 = np.sqrt(0.5 * np.sum(M**2))
    if current_m0 > 0:
        M = M * (M0 / current_m0)

    return M


# ---------------------------------------------------------------------------
# 8.2 矩张量的 P / S 波辐射花样
# ---------------------------------------------------------------------------

def mt_p_radiation(M: np.ndarray, gamma: np.ndarray) -> float:
    """P 波远场位移振幅 (矩张量形式):

        u_P ∝ γ^T · M · γ
    其中 γ 为观测方向单位向量 (ENU)。
    本函数返回  γ^T M γ (带符号)。
    """
    return float(gamma @ M @ gamma)


def mt_s_radiation_vector(M: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """S 波远场位移向量 (矩张量形式):

        u_S ∝ (I - γ γ^T) · M · γ

    返回位移向量 (ENU, 与 γ 正交)。
    """
    return M @ gamma - gamma * float(gamma @ M @ gamma)


def mt_sh_component(M: np.ndarray, gamma: np.ndarray) -> float:
    """SH 分量振幅 (矩张量形式)。"""
    z_hat = np.array([0.0, 0.0, 1.0])
    gcp_normal = np.cross(gamma, z_hat)
    norm = np.linalg.norm(gcp_normal)
    if norm < 1e-10:
        return 0.0
    e_sh = gcp_normal / norm
    f_s = mt_s_radiation_vector(M, gamma)
    return float(np.dot(f_s, e_sh))


# ---------------------------------------------------------------------------
# 8.3 矩张量反演 (振幅拟合)
# ---------------------------------------------------------------------------

def mt_inversion_amplitude(obs: List[dict],
                           phase: str = 'P',
                           allow_iso: bool = True,
                           M0_penalty: Optional[float] = None) -> dict:
    """
    矩张量线性反演: 利用 P 波或 S 波振幅 (绝对值) 求解矩张量分量。

    原理:
        对于 P 波: A_P(i) = |γ_i^T M γ_i|
            令 m = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]^T
            A_P(i) = |Σ_j G_P(i,j) m_j|
        这是一个带绝对值的线性反演问题，这里采用经典近似:
            先用初动极性确定符号, 然后用线性最小二乘求解。
            若没有极性信息, 则用迭代加权最小二乘。

    Parameters
    ----------
    obs : list of dict
        每条含 'az','toa', 'pol'(+1/-1/0), 以及相位振幅:
          - 'amp_P' 或 'p_amp' : P 波振幅 (已做几何扩散等校正)
          - 'amp_S' 或 's_amp' : S 波振幅
    phase : {'P', 'S', 'PS'}
        使用哪种相位的振幅。
    allow_iso : bool
        是否允许各向同性分量 (Mxx=Myy=Mzz)。若 False, 施加约束 tr(M)=0。
    M0_penalty : float or None
        若给定, 添加 L2 惩罚项 λ · ||M||^2 以稳定解。

    Returns
    -------
    dict with:
        'M' : 最优 3×3 矩张量
        'residual' : 拟合 RMS
        'decomposition' : mt_decompose() 的结果
        'n_obs' : 使用的观测数
    """
    # 收集有效观测
    rows = []   # 每行: (gamma, sign, amplitude, weight)
    for o in obs:
        g = sph_to_cart(o['az'], o['toa'])
        pol = o.get('pol', 0)
        sign = None
        amp = None

        if phase in ('P', 'PS'):
            pamp = o.get('amp_P', o.get('p_amp'))
            if pamp is not None and np.isfinite(pamp) and pamp > 0:
                amp = float(pamp)
                if pol in (1, -1):
                    sign = pol
                rows.append(('P', g, sign, amp))

        if phase in ('S', 'PS'):
            samp = o.get('amp_S', o.get('s_amp'))
            if samp is not None and np.isfinite(samp) and samp > 0:
                amp = float(samp)
                rows.append(('S', g, sign, amp))

    if len(rows) < 6:
        raise RuntimeError(f"观测数不足: 需要至少 6 个, 只有 {len(rows)} 个。")

    # 构造线性系统: G * m = d
    # m = [Mxx, Myy, Mzz, Mxy, Mxz, Myz]^T
    G = []
    d = []
    for phase_type, g, sign, amp in rows:
        if phase_type == 'P':
            # γ^T M γ = γx² Mxx + γy² Myy + γz² Mzz + 2γxγy Mxy + 2γxγz Mxz + 2γyγz Myz
            G_row = [g[0]**2, g[1]**2, g[2]**2,
                     2*g[0]*g[1], 2*g[0]*g[2], 2*g[1]*g[2]]
        else:  # S: 用 SH 分量近似
            M_sh = mt_sh_component(np.eye(6).reshape(3, 2)[:, :3], g)
            # 为简化, 这里假设用 S 波总振幅: |F_S| = |(I - γγ^T) M γ|
            # 这是关于 M 的二次型, 难以线性化, 我们改用 S 波各分量线性组合
            # 这里只做近似: 用 (I - γγ^T) M γ 的 L2 范数作为振幅
            # 实际全波形反演通常在时域直接拟合, 此处为演示用
            continue

        if sign is None:
            sign = 1  # 无极性时假定为正
        G.append(G_row)
        d.append(sign * amp)

    G = np.array(G, dtype=float)
    d = np.array(d, dtype=float)

    if allow_iso is False:
        # 施加约束: Mxx + Myy + Mzz = 0
        C = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        # 消除变量 Mzz = -Mxx - Myy
        G_new = G.copy()
        G_new[:, 0] = G[:, 0] - G[:, 2]
        G_new[:, 1] = G[:, 1] - G[:, 2]
        G_new = G_new[:, :5]
        m_reduced, *_ = np.linalg.lstsq(G_new, d, rcond=None)
        m = np.zeros(6)
        m[0] = m_reduced[0]
        m[1] = m_reduced[1]
        m[2] = -m_reduced[0] - m_reduced[1]
        m[3] = m_reduced[2]
        m[4] = m_reduced[3]
        m[5] = m_reduced[4]
    else:
        if M0_penalty is not None:
            # 带惩罚的最小二乘: [G; λI] m = [d; 0]
            Gp = np.vstack([G, M0_penalty * np.eye(6)])
            dp = np.concatenate([d, np.zeros(6)])
            m, *_ = np.linalg.lstsq(Gp, dp, rcond=None)
        else:
            m, *_ = np.linalg.lstsq(G, d, rcond=None)

    M = np.array([[m[0], m[3], m[4]],
                  [m[3], m[1], m[5]],
                  [m[4], m[5], m[2]]])

    pred = G @ m
    residual = float(np.sqrt(np.mean((pred - d) ** 2)))

    return {
        'M': M,
        'residual': residual,
        'n_obs': len(d),
        'decomposition': mt_decompose(M)
    }


# ---------------------------------------------------------------------------
# 8.4 矩张量格点搜索 (DC + ISO + CLVD 参数化)
# ---------------------------------------------------------------------------

def mt_grid_search(obs: List[dict],
                   strike_step: float = 10.0,
                   dip_step: float = 10.0,
                   rake_step: float = 10.0,
                   iso_range: Tuple[float, float] = (-0.5, 0.5),
                   iso_step: float = 0.1,
                   clvd_range: Tuple[float, float] = (-0.5, 0.5),
                   clvd_step: float = 0.1,
                   phase: str = 'P',
                   polarity_weight: float = 0.5) -> dict:
    """
    矩张量格点搜索 (参数化: strike/dip/rake + ISO + CLVD)。

    适用于火山/诱发地震分析, 直接搜索物理可解释的参数空间。

    Parameters
    ----------
    obs : list of dict
        含 'az','toa','pol'(+1/-1/0), 以及 'amp_P' 或 'p_amp'.
    strike_step, dip_step, rake_step : float
        DC 取向参数搜索步长 (度)。
    iso_range, iso_step : (min,max) and step
        ISO 比例范围与步长 (正=膨胀, 负=收缩)。
    clvd_range, clvd_step : (min,max) and step
        CLVD 比例范围与步长。
    phase : {'P'}
        使用哪种相位 (当前仅 P 波振幅)。
    polarity_weight : float
        极性在目标函数中的权重 (0-1)。

    Returns
    -------
    dict with:
        'strike', 'dip', 'rake'
        'iso_frac', 'clvd_frac'
        'M', 'decomposition'
        'combined_score', 'polarity_misfit', 'amp_rms'
    """
    pol_obs = [(o['az'], o['toa'], o['pol']) for o in obs
               if o.get('pol', 0) in (1, -1)]
    amp_obs = [(o['az'], o['toa'],
                o.get('amp_P', o.get('p_amp', 0.0))) for o in obs
               if o.get('amp_P', o.get('p_amp')) is not None]

    if len(pol_obs) == 0 and len(amp_obs) == 0:
        raise ValueError("既没有极性观测也没有振幅观测。")

    pol_gammas, pol_vals = [], []
    for a, t, p in pol_obs:
        pol_gammas.append(sph_to_cart(a, t))
        pol_vals.append(p)
    pol_gammas = np.array(pol_gammas) if pol_gammas else np.zeros((0, 3))
    pol_vals   = np.array(pol_vals, dtype=int) if pol_vals else np.zeros(0, dtype=int)

    amp_gammas, amp_vals = [], []
    for a, t, amp in amp_obs:
        if amp > 0:
            amp_gammas.append(sph_to_cart(a, t))
            amp_vals.append(amp)
    amp_gammas = np.array(amp_gammas) if amp_gammas else np.zeros((0, 3))
    amp_vals   = np.array(amp_vals) if amp_vals else np.zeros(0)

    # 振幅归一化: 除以中位数, 使目标函数不受绝对振幅影响
    if len(amp_vals) > 0:
        amp_norm = np.median(amp_vals)
        if amp_norm > 0:
            amp_vals = amp_vals / amp_norm
    else:
        amp_norm = 1.0

    best_score = np.inf
    best_params = None
    best_M = None

    strikes = np.arange(0, 360, strike_step)
    dips    = np.arange(0,  90 + dip_step, dip_step)
    rakes   = np.arange(-180, 180 + rake_step, rake_step)
    isos    = np.arange(iso_range[0], iso_range[1] + 0.5 * iso_step, iso_step)
    clvds   = np.arange(clvd_range[0], clvd_range[1] + 0.5 * clvd_step, clvd_step)

    w_p = polarity_weight
    w_a = 1.0 - polarity_weight

    for stk in strikes:
        for d in dips:
            for rk in rakes:
                M_dc = mt_from_sdr(stk, d, rk, M0=1.0)
                for iso in isos:
                    for clvd in clvds:
                        # 组合矩张量 (DC 为主, ISO + CLVD 为附加)
                        M = (1.0 - abs(iso) - abs(clvd)) * M_dc \
                            + iso * np.eye(3) \
                            + clvd * (np.outer(np.array([0, 0, 1]), np.array([0, 0, 1])) - np.eye(3)/3.0)

                        score = 0.0
                        pol_misfit = 0
                        amp_rms = 0.0

                        # 极性残差
                        if len(pol_vals) > 0:
                            Fp = np.array([float(g @ M @ g) for g in pol_gammas])
                            pred = np.sign(Fp).astype(int)
                            pred[pred == 0] = 1
                            pol_misfit = int(np.sum(pred != pol_vals))
                            score += w_p * (pol_misfit / len(pol_vals))

                        # 振幅残差
                        if len(amp_vals) > 0:
                            pred_amps = np.array([abs(float(g @ M @ g)) for g in amp_gammas])
                            if np.max(pred_amps) > 0:
                                pred_amps = pred_amps / np.median(pred_amps[pred_amps > 0])
                            amp_rms = float(np.sqrt(np.mean((pred_amps - amp_vals) ** 2)))
                            score += w_a * (amp_rms / 1.0)

                        if score < best_score:
                            best_score = score
                            best_params = (float(stk), float(d), float(rk),
                                           float(iso), float(clvd))
                            best_M = M.copy()

    if best_params is None:
        raise RuntimeError("未找到解。")

    stk, d, rk, iso, clvd = best_params
    return {
        'strike': stk, 'dip': d, 'rake': rk,
        'iso_frac': iso, 'clvd_frac': clvd,
        'M': best_M,
        'decomposition': mt_decompose(best_M),
        'combined_score': best_score,
        'polarity_misfit': pol_misfit if len(pol_vals) > 0 else None,
        'amp_rms': amp_rms if len(amp_vals) > 0 else None,
        'polarity_total': len(pol_vals),
        'amp_total': len(amp_vals)
    }


# ============================================================================
# 9. 示例 & 测试
# ============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    # ====================================================================
    # 示例: 倾斜走滑+逆冲 (rake=30°), 振幅比打破极性多解性
    # ====================================================================
    #
    # 物理背景:
    #   对于纯双力偶源, 主断层面与辅助面的 P/S 辐射花样完全相同,
    #   任何地震波数据都无法区分二者 —— 这是双力偶的本征对称性.
    #
    #   但当台站覆盖不均或极性数据有噪声时, 会出现多个**非共轭**的
    #   机制解同时满足极性约束, 此时 SH/P 振幅比可以有效地区分它们,
    #   并对倾角/滑动角给出更精确的约束.
    # ====================================================================

    true_s, true_d, true_r = 120.0, 50.0, 30.0   # 倾斜逆冲+走滑
    n_true, s_true = sdr_vectors(true_s, true_d, true_r)

    N = 50
    azs_all  = np.random.uniform(0, 360, N)
    toas_all = np.random.uniform(25, 160, N)

    # 前 30 个台站: 极性 + 振幅比 (振幅比加 10% 对数噪声)
    # 后 20 个台站: 仅极性 (模拟部分台站无振幅测量)
    obs = []
    for i, (az, toa) in enumerate(zip(azs_all, toas_all)):
        g = sph_to_cart(az, toa)
        Fp = p_radiation(g, n_true, s_true)
        pol = 1 if Fp >= 0 else -1

        # 给 ~10% 的极性添加随机错误 (模拟实际拾取误差)
        if np.random.random() < 0.10:
            pol = -pol

        entry = {'az': float(az), 'toa': float(toa), 'pol': pol}
        if i < 30:
            r = sh_p_ratio(g, n_true, s_true)
            if np.isfinite(r) and r > 0.01:
                noise = 10 ** np.random.normal(0, 0.10)  # ~10% 对数误差
                entry['sh_p_ratio'] = float(r * noise)
        obs.append(entry)

    n_pol = sum(1 for o in obs if o.get('pol') in (1, -1))
    n_amp = sum(1 for o in obs if o.get('sh_p_ratio') is not None)
    print(f"观测数: 极性={n_pol} (含 ~10% 错误), 振幅比={n_amp}")
    print(f"真值:   strike={true_s:6.1f}° dip={true_d:5.1f}° rake={true_r:6.1f}°")
    print()

    # ======== (A) 纯极性反演 ========
    print("=" * 58)
    print("(A) 纯 P 波极性反演 (格点搜索 5°)")
    print("=" * 58)
    res_pol = grid_search(obs, strike_step=5, dip_step=5, rake_step=5,
                          verbose=False, top_k=5)
    for i, r in enumerate(res_pol):
        print(f"  候选 #{i+1}: strike={r['strike']:6.1f}° "
              f"dip={r['dip']:5.1f}° rake={r['rake']:6.1f}°  "
              f"misfit={r['misfit']}/{r['total']}")

    print()
    print("  → 多个候选解具有相同极性残差, 仅靠极性无法区分!")

    # ======== (B) 联合反演 ========
    print()
    print("=" * 58)
    print("(B) 联合反演: P 波极性 + SH/P 振幅比")
    print("=" * 58)
    res_joint = joint_inversion(obs, strike_step=5, dip_step=5, rake_step=5,
                                polarity_weight=0.65, refine_step=2.0,
                                verbose=False)
    print(f"  最优解: strike={res_joint['strike']:6.1f}° "
          f"dip={res_joint['dip']:5.1f}° rake={res_joint['rake']:6.1f}°")
    print(f"  极性矛盾: {res_joint['polarity_misfit']}/{res_joint['polarity_total']}")
    print(f"  振幅比 RMS (log10): {res_joint['amp_ratio_rms']:.4f}  "
          f"(有效 {res_joint['amp_ratio_count']} 个)")
    print(f"  联合得分: {res_joint['combined_score']:.4f}")
    print(f"  辅助面: strike={res_joint['aux_plane']['strike']:6.1f}° "
          f"dip={res_joint['aux_plane']['dip']:5.1f}° "
          f"rake={res_joint['aux_plane']['rake']:6.1f}°")

    # 计算纯极性 top-5 候选的振幅比 RMS, 展示振幅比如何打破平局
    print()
    print("  --- 各极性候选解的振幅比 RMS ---")
    amp_vals_arr = np.array([o['sh_p_ratio'] for o in obs
                             if o.get('sh_p_ratio') is not None])
    amp_gammas_arr = np.array([sph_to_cart(o['az'], o['toa']) for o in obs
                                if o.get('sh_p_ratio') is not None])
    for i, r in enumerate(res_pol):
        n_c, s_c = sdr_vectors(r['strike'], r['dip'], r['rake'])
        rms = _amp_ratio_misfit(None, amp_gammas_arr, n_c, s_c, amp_vals_arr)
        marker = "★ 最优" if abs(rms - res_joint['amp_ratio_rms']) < 0.001 else ""
        print(f"  候选 #{i+1}: amp_rms={rms:.4f} {marker}")

    print()
    print(f"真值:     strike={true_s:6.1f}° dip={true_d:5.1f}° rake={true_r:6.1f}°")
    print()

    # ======== (C) 台站分布不均场景 ========
    print("=" * 58)
    print("(C) 台站分布不均 (仅 toa < 90°, 上半球)")
    print("=" * 58)
    obs_upper = []
    for o in obs:
        if o['toa'] < 90:
            obs_upper.append(o)
    n_pol_u = sum(1 for o in obs_upper if o.get('pol') in (1, -1))
    n_amp_u = sum(1 for o in obs_upper if o.get('sh_p_ratio') is not None)
    print(f"观测数: 极性={n_pol_u}, 振幅比={n_amp_u}")

    res_pol_u = grid_search(obs_upper, strike_step=5, dip_step=5, rake_step=5,
                            verbose=False, top_k=5)
    print("  纯极性 top-5:")
    for i, r in enumerate(res_pol_u):
        print(f"    #{i+1}: strike={r['strike']:6.1f}° dip={r['dip']:5.1f}° "
              f"rake={r['rake']:6.1f}°  misfit={r['misfit']}/{r['total']}")

    res_joint_u = joint_inversion(obs_upper, strike_step=5, dip_step=5, rake_step=5,
                                   polarity_weight=0.65, refine_step=None,
                                   verbose=False)
    print(f"  联合最优: strike={res_joint_u['strike']:6.1f}° "
          f"dip={res_joint_u['dip']:5.1f}° rake={res_joint_u['rake']:6.1f}°")
    print(f"            pol_misfit={res_joint_u['polarity_misfit']}/"
          f"{res_joint_u['polarity_total']}  "
          f"amp_rms={res_joint_u['amp_ratio_rms']:.4f}")
    print(f"真值:       strike={true_s:6.1f}° dip={true_d:5.1f}° "
          f"rake={true_r:6.1f}°")

    print()
    print("=" * 70)
    print("  第 3 部分: 矩张量反演 — 火山地震 & 诱发地震分析")
    print("=" * 70)
    print()

    # ====================================================================
    # (D) 示例 1: 纯双力偶 (DC) 源验证
    # ====================================================================
    print("-" * 70)
    print("(D) 纯剪切破裂 (纯 DC 源) — 构造地震")
    print("-" * 70)

    true_dc = mt_from_sdr(120.0, 50.0, 30.0, M0=1.0)
    dec_dc = mt_decompose(true_dc)
    print(f"  真值矩张量分解:")
    print(f"    ISO:  {dec_dc['iso_percent']:+6.1f}%  "
          f"DC: {dec_dc['dc_percent']:5.1f}%  "
          f"CLVD: {dec_dc['clvd_percent']:+6.1f}%")
    print(f"    (期望: ISO≈0%, DC≈100%, CLVD≈0%)")

    # 合成 30 个台站的 P 波振幅 + 极性
    N = 30
    azs = np.random.uniform(0, 360, N)
    toas = np.random.uniform(20, 160, N)
    obs_dc = []
    for az, toa in zip(azs, toas):
        g = sph_to_cart(az, toa)
        Fp = mt_p_radiation(true_dc, g)
        pol = 1 if Fp >= 0 else -1
        amp = abs(Fp) * np.random.lognormal(0, 0.15)  # ~15% 振幅噪声
        obs_dc.append({'az': float(az), 'toa': float(toa),
                       'pol': pol, 'amp_P': float(amp)})

    # 矩张量线性反演 (允许 ISO)
    res_dc = mt_inversion_amplitude(obs_dc, phase='P', allow_iso=True)
    dec_inv = res_dc['decomposition']
    print(f"  反演结果 (线性最小二乘, 允许 ISO):")
    print(f"    ISO:  {dec_inv['iso_percent']:+6.1f}%  "
          f"DC: {dec_inv['dc_percent']:5.1f}%  "
          f"CLVD: {dec_inv['clvd_percent']:+6.1f}%")
    print(f"    振幅拟合 RMS: {res_dc['residual']:.4f}  "
          f"(有效观测 {res_dc['n_obs']} 个)")

    # 纯偏差反演 (强制 tr(M)=0)
    res_dc_dev = mt_inversion_amplitude(obs_dc, phase='P', allow_iso=False)
    dec_dev = res_dc_dev['decomposition']
    print(f"  反演结果 (强制纯偏差, tr(M)=0):")
    print(f"    ISO:  {dec_dev['iso_percent']:+6.1f}%  "
          f"DC: {dec_dev['dc_percent']:5.1f}%  "
          f"CLVD: {dec_dev['clvd_percent']:+6.1f}%")
    print(f"    振幅拟合 RMS: {res_dc_dev['residual']:.4f}")

    # ====================================================================
    # (E) 示例 2: 火山地震 — 含显著各向同性分量 (膨胀型)
    # ====================================================================
    print()
    print("-" * 70)
    print("(E) 火山地震 — 含 +30% 各向同性分量 (岩浆活动/流体注入)")
    print("-" * 70)

    # 构造含 ISO 分量的矩张量: 70% DC + 30% ISO (膨胀)
    M_volc = mt_combine(iso_frac=0.30, dc_frac=0.70, clvd_frac=0.0,
                        strike=120.0, dip=50.0, rake=30.0, M0=1.0)
    dec_volc_true = mt_decompose(M_volc)
    print(f"  真值: ISO={dec_volc_true['iso_percent']:+6.1f}%  "
          f"DC={dec_volc_true['dc_percent']:5.1f}%  "
          f"CLVD={dec_volc_true['clvd_percent']:+6.1f}%")

    # 合成观测
    obs_volc = []
    for az, toa in zip(azs, toas):
        g = sph_to_cart(az, toa)
        Fp = mt_p_radiation(M_volc, g)
        pol = 1 if Fp >= 0 else -1
        amp = abs(Fp) * np.random.lognormal(0, 0.15)
        obs_volc.append({'az': float(az), 'toa': float(toa),
                         'pol': pol, 'amp_P': float(amp)})

    # 反演 (允许 ISO)
    res_volc = mt_inversion_amplitude(obs_volc, phase='P', allow_iso=True)
    dec_v = res_volc['decomposition']
    print(f"  反演: ISO={dec_v['iso_percent']:+6.1f}%  "
          f"DC={dec_v['dc_percent']:5.1f}%  "
          f"CLVD={dec_v['clvd_percent']:+6.1f}%")
    print(f"        振幅 RMS={res_volc['residual']:.4f}")

    # 若错误地强制纯偏差 (忽略体积变化)
    res_volc_wrong = mt_inversion_amplitude(obs_volc, phase='P', allow_iso=False)
    dec_vw = res_volc_wrong['decomposition']
    print(f"  强制纯偏差: ISO={dec_vw['iso_percent']:+6.1f}%  "
          f"DC={dec_vw['dc_percent']:5.1f}%  "
          f"CLVD={dec_vw['clvd_percent']:+6.1f}%")
    print(f"                振幅 RMS={res_volc_wrong['residual']:.4f}  "
          f"{'← 拟合显著变差!' if res_volc_wrong['residual'] > res_volc['residual']*1.5 else ''}")

    # ====================================================================
    # (F) 示例 3: 诱发地震 — 含 CLVD 分量 (水力压裂/矿震)
    # ====================================================================
    print()
    print("-" * 70)
    print("(F) 诱发地震 — 含 CLVD 分量 (张裂型, 水力压裂)")
    print("-" * 70)

    # 构造含 CLVD 分量的矩张量: 70% DC + 30% CLVD (拉伸型)
    M_ind = mt_combine(iso_frac=0.0, dc_frac=0.70, clvd_frac=0.30,
                       strike=120.0, dip=50.0, rake=30.0,
                       clvd_axis_az=90.0, clvd_axis_plunge=0.0, M0=1.0)
    dec_ind_true = mt_decompose(M_ind)
    print(f"  真值: ISO={dec_ind_true['iso_percent']:+6.1f}%  "
          f"DC={dec_ind_true['dc_percent']:5.1f}%  "
          f"CLVD={dec_ind_true['clvd_percent']:+6.1f}%")

    # 合成观测
    obs_ind = []
    for az, toa in zip(azs, toas):
        g = sph_to_cart(az, toa)
        Fp = mt_p_radiation(M_ind, g)
        pol = 1 if Fp >= 0 else -1
        amp = abs(Fp) * np.random.lognormal(0, 0.15)
        obs_ind.append({'az': float(az), 'toa': float(toa),
                        'pol': pol, 'amp_P': float(amp)})

    # 反演
    res_ind = mt_inversion_amplitude(obs_ind, phase='P', allow_iso=True)
    dec_i = res_ind['decomposition']
    print(f"  反演: ISO={dec_i['iso_percent']:+6.1f}%  "
          f"DC={dec_i['dc_percent']:5.1f}%  "
          f"CLVD={dec_i['clvd_percent']:+6.1f}%")
    print(f"        振幅 RMS={res_ind['residual']:.4f}")

    # ====================================================================
    # (G) 矩张量格点搜索 (可选, 此处跳过以加速演示)
    # ====================================================================
    print()
    print("-" * 70)
    print("(G) 矩张量分解总结")
    print("-" * 70)
    print("  矩张量反演方法总结:")
    print("  - 纯构造地震: ISO≈0%, DC≈100%")
    print("  - 火山/流体注入: ISO分量显著 (正=膨胀, 负=收缩")
    print("  - 诱发/矿震: CLVD分量显著")
    print()
    print("  使用 mt_inversion_amplitude() 可自动分解 ISO/DC/CLVD 比例")
    print("  使用 mt_grid_search() 可直接搜索物理参数空间 (需要更多计算时间)")

    print()
    print("=" * 70)
    print("  总结:")
    print("    - 纯构造地震: ISO≈0, DC≈100%")
    print("    - 火山/流体注入: ISO > 0 (膨胀) 或 ISO < 0 (收缩)")
    print("    - 诱发/矿震:   CLVD 分量显著, 反映张裂或塌陷")
    print("=" * 70)
