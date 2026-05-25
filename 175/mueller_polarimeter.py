"""
Mueller矩阵偏振计数据重构
========================

本模块实现从Mueller矩阵偏振计测量数据中重构样品的16个Mueller矩阵元素。
支持两种方法：
1. 双旋转波片法 (Double Rotating Retarder, DRR)
2. 液晶可变延迟器法 (Liquid Crystal Variable Retarder, LCVR)

原理:
    偏振计的基本配置为:
    [光源] → [PSG(偏振态发生器)] → [样品] → [PSA(偏振态分析器)] → [探测器]

    对于N个测量配置，测量方程为:
        I_i = A_i^T · M · W_i
    其中A_i是PSA的Stokes向量，W_i是PSG产生的Stokes向量，M是样品的Mueller矩阵。

    将M的16个元素展开为向量，使用最小二乘法求解:
        I = H · m
        m = pinv(H) · I
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


# =============================================================================
# 基础Mueller矩阵函数
# =============================================================================

def rotator(angle: float) -> np.ndarray:
    """
    旋转矩阵的Mueller矩阵

    将坐标系旋转angle角度（弧度），用于将光学元件的Mueller矩阵
    变换到实验室坐标系。

    参数:
        angle: 旋转角度 [rad]

    返回:
        4x4 Mueller旋转矩阵
    """
    c = np.cos(2 * angle)
    s = np.sin(2 * angle)
    R = np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, -s, c, 0],
        [0, 0, 0, 1]
    ])
    return R


def polarizer(angle: float = 0.0) -> np.ndarray:
    """
    理想线偏振器的Mueller矩阵

    参数:
        angle: 偏振方向角 [rad]

    返回:
        4x4 Mueller矩阵
    """
    M = 0.5 * np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    R = rotator(angle)
    return R @ M @ R.T


def retarder(retardance: float, angle: float = 0.0) -> np.ndarray:
    """
    理想相位延迟器（波片）的Mueller矩阵

    参数:
        retardance: 相位延迟量 [rad]
                     (π/2 for 1/4波片, π for 1/2波片)
        angle: 快轴方向角 [rad]

    返回:
        4x4 Mueller矩阵
    """
    d = retardance
    c_d = np.cos(d)
    s_d = np.sin(d)
    M = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c_d, s_d],
        [0, 0, -s_d, c_d]
    ])
    R = rotator(angle)
    return R @ M @ R.T


def stokes_from_mueller(M: np.ndarray) -> np.ndarray:
    """
    从Mueller矩阵提取Stokes向量的生成矩阵

    对于偏振态发生器(PSG)，输出Stokes向量为:
        S_out = M_PSG · S_in
    其中S_in是未偏振光: [1, 0, 0, 0]^T

    参数:
        M: 4x4 Mueller矩阵

    返回:
        4x1 Stokes向量
    """
    return M @ np.array([1, 0, 0, 0])


# =============================================================================
# 测量矩阵构造
# =============================================================================

def build_measurement_matrix_drr(
    n_measurements: int = 36,
    psg_waveplate_retardance: float = np.pi / 2,
    psa_waveplate_retardance: float = np.pi / 2,
    psg_polarizer_angle: float = 0.0,
    psa_polarizer_angle: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    构造双旋转波片法(Double Rotating Retarder)的测量矩阵

    配置:
        PSG: 偏振器 + 1/4波片(旋转)
        PSA: 1/4波片(旋转) + 偏振器

    波片角度:
        PSG波片: θ_PSG = k * 2π / n_measurements
        PSA波片: θ_PSA = 5 * k * 2π / n_measurements (比值5:1是常用选择)

    参数:
        n_measurements: 测量次数(建议≥16以保证16个未知量的确定性)
        psg_waveplate_retardance: PSG侧波片的延迟量 [rad]
        psa_waveplate_retardance: PSA侧波片的延迟量 [rad]
        psg_polarizer_angle: PSG侧偏振器角度 [rad]
        psa_polarizer_angle: PSA侧偏振器角度 [rad]

    返回:
        H: 测量矩阵 (n_measurements x 16)
        W: PSG生成的Stokes向量矩阵 (4 x n_measurements)
        configs: 测量配置列表
    """
    H = np.zeros((n_measurements, 16))
    W = np.zeros((4, n_measurements))
    configs = []

    for k in range(n_measurements):
        theta_psg = k * 2 * np.pi / n_measurements
        theta_psa = 5 * k * 2 * np.pi / n_measurements

        M_psg = retarder(psg_waveplate_retardance, theta_psg) @ polarizer(psg_polarizer_angle)
        M_psa = polarizer(psa_polarizer_angle) @ retarder(psa_waveplate_retardance, theta_psa)

        S_in = np.array([1, 0, 0, 0])
        S_psg = M_psg @ S_in
        S_psa = M_psa.T @ S_in

        W[:, k] = S_psg

        H_row = np.kron(S_psa, S_psg)
        H[k, :] = H_row

        configs.append({
            'index': k,
            'theta_psg_deg': np.degrees(theta_psg),
            'theta_psa_deg': np.degrees(theta_psa),
            'stokes_psg': S_psg,
            'stokes_psa_analysis': S_psa
        })

    return H, W, configs


def build_measurement_matrix_lcvr(
    retardance_states: Optional[List[float]] = None,
    psa_polarizer_angle: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    构造液晶可变延迟器(LCVR)法的测量矩阵

    配置:
        PSG: 偏振器 + LCVR1 + LCVR2
        PSA: 偏振器

    使用不同的LCVR电压组合来生成/分析不同的偏振态。
    每个LCVR可以实现任意延迟量(0到π)。

    参数:
        retardance_states: 可选的延迟量状态列表
                          默认使用: [0, π/2, π, 3π/2] 组合
        psa_polarizer_angle: PSA侧偏振器角度 [rad]

    返回:
        H: 测量矩阵 (N x 16)
        W: PSG生成的Stokes向量矩阵 (4 x N)
        configs: 测量配置列表
    """
    if retardance_states is None:
        retardance_states = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    n_states = len(retardance_states)
    n_measurements = n_states * n_states

    H = np.zeros((n_measurements, 16))
    W = np.zeros((4, n_measurements))
    configs = []

    idx = 0
    for d1 in retardance_states:
        for d2 in retardance_states:
            M_psg = retarder(d2, np.pi / 4) @ retarder(d1, 0) @ polarizer(0)
            M_psa = polarizer(psa_polarizer_angle) @ retarder(d2, np.pi / 4) @ retarder(d1, 0)

            S_in = np.array([1, 0, 0, 0])
            S_psg = M_psg @ S_in
            S_psa = M_psa.T @ S_in

            W[:, idx] = S_psg
            H_row = np.kron(S_psa, S_psg)
            H[idx, :] = H_row

            configs.append({
                'index': idx,
                'lcvr1_retardance_deg': np.degrees(d1),
                'lcvr2_retardance_deg': np.degrees(d2),
                'stokes_psg': S_psg,
                'stokes_psa_analysis': S_psa
            })
            idx += 1

    return H, W, configs


# =============================================================================
# Mueller矩阵重构
# =============================================================================

def reconstruct_mueller(
    H: np.ndarray,
    I: np.ndarray,
    method: str = 'lstsq'
) -> Tuple[np.ndarray, Dict]:
    """
    从测量数据重构Mueller矩阵

    测量方程: I = H · m
    其中m是将Mueller矩阵M按行展开的16维向量:
        m = [M00, M01, M02, M03, M10, M11, ..., M33]^T

    参数:
        H: 测量矩阵 (N x 16)
        I: 测量强度向量 (N,)
        method: 重构方法
                - 'lstsq': numpy.linalg.lstsq (推荐)
                - 'pinv': 伪逆法
                - 'normal': 正规方程法

    返回:
        M: 重构的4x4 Mueller矩阵
        info: 诊断信息字典
    """
    if method == 'lstsq':
        m, residuals, rank, s = np.linalg.lstsq(H, I, rcond=None)
        condition_number = s[0] / s[-1] if len(s) > 0 else np.inf
        info = {
            'method': 'lstsq',
            'rank': rank,
            'singular_values': s,
            'condition_number': condition_number,
            'residuals': residuals,
            'rmse': np.sqrt(np.mean(residuals)) if len(residuals) > 0 else None
        }
    elif method == 'pinv':
        H_pinv = np.linalg.pinv(H)
        m = H_pinv @ I
        I_pred = H @ m
        residuals = np.sum((I - I_pred) ** 2)
        info = {
            'method': 'pinv',
            'residuals': residuals,
            'rmse': np.sqrt(np.mean((I - I_pred) ** 2))
        }
    elif method == 'normal':
        HTH = H.T @ H
        HTI = H.T @ I
        m = np.linalg.solve(HTH, HTI)
        I_pred = H @ m
        residuals = np.sum((I - I_pred) ** 2)
        info = {
            'method': 'normal',
            'residuals': residuals,
            'rmse': np.sqrt(np.mean((I - I_pred) ** 2))
        }
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lstsq', 'pinv', or 'normal'.")

    M = m.reshape(4, 4)

    return M, info


def simulate_measurement(
    M_sample: np.ndarray,
    H: np.ndarray,
    noise_std: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟偏振计测量过程

    参数:
        M_sample: 样品的4x4 Mueller矩阵
        H: 测量矩阵 (N x 16)
        noise_std: 高斯噪声标准差
        seed: 随机数种子(用于可重复性)

    返回:
        I: 测量强度向量 (N,)
        I_noiseless: 无噪声的理想测量强度
    """
    m = M_sample.flatten()
    I_noiseless = H @ m

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, noise_std, size=I_noiseless.shape)
        I = I_noiseless + noise
    else:
        I = I_noiseless.copy()

    return I, I_noiseless


# =============================================================================
# 评估函数
# =============================================================================

def evaluate_reconstruction(M_true: np.ndarray, M_reconstructed: np.ndarray) -> Dict:
    """
    评估Mueller矩阵重构的精度

    参数:
        M_true: 真实的4x4 Mueller矩阵
        M_reconstructed: 重构的4x4 Mueller矩阵

    返回:
        包含各种误差指标的字典
    """
    diff = M_true - M_reconstructed
    abs_error = np.abs(diff)
    rel_error = abs_error / (np.abs(M_true) + 1e-15)

    metrics = {
        'max_absolute_error': np.max(abs_error),
        'mean_absolute_error': np.mean(abs_error),
        'rmse': np.sqrt(np.mean(diff ** 2)),
        'max_relative_error': np.max(rel_error),
        'mean_relative_error': np.mean(rel_error),
        'element_wise_abs_error': abs_error,
        'element_wise_rel_error': rel_error,
        'frobenius_norm_error': np.linalg.norm(diff, 'fro'),
        'relative_frobenius_error': np.linalg.norm(diff, 'fro') / (np.linalg.norm(M_true, 'fro') + 1e-15)
    }

    return metrics


# =============================================================================
# 物理可实现性检验与Cloude分解
# =============================================================================

def _get_pauli_matrices() -> List[np.ndarray]:
    """
    获取Pauli矩阵集合（包括单位矩阵）

    返回:
        4个2x2复矩阵的列表 [σ0, σ1, σ2, σ3]
    """
    sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
    sigma_1 = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_2 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_3 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return [sigma_0, sigma_1, sigma_2, sigma_3]


def mueller_to_coherence(M: np.ndarray) -> np.ndarray:
    """
    将Mueller矩阵转换为Cloude相干矩阵（密度矩阵形式）

    相干矩阵H的构造：
        H = (1/4) * Σ_{i,j=0}^3 M_ij * (σ_i ⊗ σ_j*)
    其中σ_i是Pauli矩阵，σ_j*是其复共轭。

    物理可实现的Mueller矩阵对应半正定的相干矩阵(H ≥ 0)。

    参数:
        M: 4x4实Mueller矩阵

    返回:
        4x4复Hermitian相干矩阵H
    """
    sigmas = _get_pauli_matrices()
    H = np.zeros((4, 4), dtype=complex)

    for i in range(4):
        for j in range(4):
            sigma_i = sigmas[i]
            sigma_j_conj = np.conj(sigmas[j])
            kron_term = np.kron(sigma_i, sigma_j_conj)
            H += M[i, j] * kron_term

    H = H / 4.0

    return (H + H.conj().T) / 2.0


def coherence_to_mueller(H: np.ndarray) -> np.ndarray:
    """
    将相干矩阵转换回Mueller矩阵（Cloude分解的逆变换）

    参数:
        H: 4x4复Hermitian相干矩阵

    返回:
        4x4实Mueller矩阵M
    """
    sigmas = _get_pauli_matrices()
    M = np.zeros((4, 4), dtype=float)

    for i in range(4):
        for j in range(4):
            sigma_i = sigmas[i]
            sigma_j_conj = np.conj(sigmas[j])
            kron_term = np.kron(sigma_i, sigma_j_conj)
            M[i, j] = 4.0 * np.real(np.trace(H @ kron_term.conj().T))

    return M


def cloude_decomposition(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对Mueller矩阵执行Cloude分解

    分解步骤:
        1. 构造相干矩阵 H = U Λ U† (特征值分解)
        2. 特征值 λ_i ≥ 0 对应于物理可实现性
        3. Mueller矩阵可以表示为纯Mueller矩阵的加权和:
           M = Σ λ_i * M_i, 其中 M_i 是对应特征向量的纯Mueller矩阵

    参数:
        M: 4x4实Mueller矩阵

    返回:
        eigenvalues: 4个实特征值（已按降序排列）
        eigenvectors: 4x4复矩阵，每列是一个特征向量
        pure_muellers: 4个4x4纯Mueller矩阵的列表
    """
    H = mueller_to_coherence(M)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    pure_muellers = []
    for i in range(4):
        v = eigenvectors[:, i]
        rho_i = np.outer(v, v.conj())
        M_i = coherence_to_mueller(rho_i)
        pure_muellers.append(M_i)

    return eigenvalues, eigenvectors, pure_muellers


def check_physical_realizability(
    M: np.ndarray,
    eigenvalue_tol: float = 1e-10,
    norm_tol: float = 1e-10
) -> Dict:
    """
    检验Mueller矩阵的物理可实现性

    物理可实现性条件:
        1. 相干矩阵的所有特征值 ≥ 0 (半正定条件)
        2. 对于无源器件，|M_ij| ≤ M_00
        3. 极化度条件: DOP_out ≤ 1 对于任意输入

    参数:
        M: 4x4实Mueller矩阵
        eigenvalue_tol: 特征值容差（小于此值视为数值零）
        norm_tol: 归一化容差

    返回:
        包含检验结果的字典
    """
    eigenvalues, _, _ = cloude_decomposition(M)

    min_eigenvalue = np.min(eigenvalues)
    negative_eigenvalues = eigenvalues[eigenvalues < -eigenvalue_tol]
    n_negative = len(negative_eigenvalues)
    is_physically_realizable = n_negative == 0

    m00 = M[0, 0]
    bounded = True
    for i in range(4):
        for j in range(4):
            if abs(M[i, j]) > abs(m00) + norm_tol and (i != 0 or j != 0):
                bounded = False
                break

    copolarized_power = np.trace(M[:3, :3]) / 3.0
    transmission_valid = (m00 >= -norm_tol) and (m00 <= 1.0 + norm_tol)

    results = {
        'is_physically_realizable': is_physically_realizable,
        'eigenvalues': eigenvalues,
        'min_eigenvalue': min_eigenvalue,
        'negative_eigenvalues': negative_eigenvalues,
        'n_negative_eigenvalues': n_negative,
        'is_element_bounded': bounded,
        'M00': m00,
        'is_transmission_valid': transmission_valid,
        'eigenvalue_violation': max(0.0, -min_eigenvalue),
    }

    return results


def _enforce_trace_preservation(M: np.ndarray) -> np.ndarray:
    """
    强制Mueller矩阵保持迹守恒（可选的约束）

    参数:
        M: 4x4 Mueller矩阵

    返回:
        修正后的Mueller矩阵
    """
    M_corrected = M.copy()
    if M_corrected[0, 0] <= 0:
        M_corrected[0, 0] = 1e-10

    for i in range(1, 4):
        for j in range(4):
            if abs(M_corrected[i, j]) > M_corrected[0, 0]:
                M_corrected[i, j] = M_corrected[0, 0] * np.sign(M_corrected[i, j])

    return M_corrected


def project_to_physical_space(
    M: np.ndarray,
    method: str = 'eigenvalue_clipping',
    eigenvalue_floor: float = 0.0,
    max_iterations: int = 1000,
    tol: float = 1e-12
) -> Tuple[np.ndarray, Dict]:
    """
    将非物理的Mueller矩阵投影到物理允许空间

    这是最大似然估计的近似，在高斯噪声假设下等价于
    Frobenius范数意义下最近的物理可实现矩阵。

    参数:
        M: 4x4 Mueller矩阵（可能非物理）
        method: 投影方法
            - 'eigenvalue_clipping': 简单的特征值截断（最快）
            - 'iterative_projection': 交替投影算法（更精确）
        eigenvalue_floor: 特征值的最小允许值（默认0，可设为很小的正数保证数值稳定性）
        max_iterations: 迭代投影的最大迭代次数
        tol: 收敛判据

    返回:
        M_physical: 投影后的物理可实现Mueller矩阵
        info: 投影过程的诊断信息
    """
    check_before = check_physical_realizability(M)

    if check_before['is_physically_realizable']:
        info = {
            'method_used': 'none',
            'iterations': 0,
            'converged': True,
            'initial_min_eigenvalue': check_before['min_eigenvalue'],
            'final_min_eigenvalue': check_before['min_eigenvalue'],
            'frobenius_change': 0.0,
            'was_physical': True
        }
        return M.copy(), info

    if method == 'eigenvalue_clipping':
        H = mueller_to_coherence(M)
        eigvals, eigvecs = np.linalg.eigh(H)

        eigvals_clipped = np.maximum(eigvals, eigenvalue_floor)

        H_physical = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.conj().T

        M_physical = coherence_to_mueller(H_physical)

        final_check = check_physical_realizability(M_physical)
        info = {
            'method_used': 'eigenvalue_clipping',
            'iterations': 1,
            'converged': final_check['is_physically_realizable'],
            'initial_min_eigenvalue': check_before['min_eigenvalue'],
            'final_min_eigenvalue': final_check['min_eigenvalue'],
            'frobenius_change': np.linalg.norm(M - M_physical, 'fro'),
            'was_physical': False,
            'eigenvalues_before': eigvals,
            'eigenvalues_after': eigvals_clipped
        }

    elif method == 'iterative_projection':
        M_current = M.copy()
        initial_M = M.copy()

        for iteration in range(max_iterations):
            H = mueller_to_coherence(M_current)
            eigvals, eigvecs = np.linalg.eigh(H)

            if np.min(eigvals) >= -tol:
                break

            eigvals_proj = np.maximum(eigvals, eigenvalue_floor)
            H_physical = eigvecs @ np.diag(eigvals_proj) @ eigvecs.conj().T
            M_current = coherence_to_mueller(H_physical)

            M_current = _enforce_trace_preservation(M_current)

        converged = check_physical_realizability(M_current)['is_physically_realizable']
        final_check = check_physical_realizability(M_current)

        info = {
            'method_used': 'iterative_projection',
            'iterations': iteration + 1,
            'converged': converged,
            'initial_min_eigenvalue': check_before['min_eigenvalue'],
            'final_min_eigenvalue': final_check['min_eigenvalue'],
            'frobenius_change': np.linalg.norm(initial_M - M_current, 'fro'),
            'was_physical': False
        }

        M_physical = M_current

    else:
        raise ValueError(
            f"Unknown projection method: {method}. "
            f"Use 'eigenvalue_clipping' or 'iterative_projection'."
        )

    return M_physical, info


def reconstruct_mueller_physical(
    H: np.ndarray,
    I: np.ndarray,
    reconstruction_method: str = 'lstsq',
    projection_method: str = 'eigenvalue_clipping',
    eigenvalue_floor: float = 0.0
) -> Tuple[np.ndarray, Dict]:
    """
    带物理约束的Mueller矩阵重构

    先进行常规重构，然后将结果投影到物理允许空间。

    参数:
        H: 测量矩阵 (N x 16)
        I: 测量强度向量 (N,)
        reconstruction_method: 初始重构方法 ('lstsq', 'pinv', 'normal')
        projection_method: 投影方法 ('eigenvalue_clipping', 'iterative_projection')
        eigenvalue_floor: 特征值的最小允许值

    返回:
        M_physical: 物理可实现的4x4 Mueller矩阵
        info: 完整的诊断信息字典
    """
    M_raw, recon_info = reconstruct_mueller(H, I, method=reconstruction_method)

    phys_check = check_physical_realizability(M_raw)

    if phys_check['is_physically_realizable']:
        info = {
            **recon_info,
            'physical_check': phys_check,
            'projection_info': {'was_physical': True},
            'projection_needed': False,
            'M_raw': M_raw
        }
        return M_raw, info

    M_physical, proj_info = project_to_physical_space(
        M_raw,
        method=projection_method,
        eigenvalue_floor=eigenvalue_floor
    )

    I_pred_raw = H @ M_raw.flatten()
    I_pred_phys = H @ M_physical.flatten()
    residual_raw = np.sum((I - I_pred_raw) ** 2)
    residual_phys = np.sum((I - I_pred_phys) ** 2)

    final_check = check_physical_realizability(M_physical)

    info = {
        **recon_info,
        'physical_check_raw': phys_check,
        'physical_check_final': final_check,
        'projection_info': proj_info,
        'projection_needed': True,
        'residual_raw': residual_raw,
        'residual_physical': residual_phys,
        'residual_increase': residual_phys - residual_raw,
        'relative_residual_increase': (residual_phys - residual_raw) / (residual_raw + 1e-15),
        'M_raw': M_raw,
        'M_physical': M_physical,
        'frobenius_correction': np.linalg.norm(M_raw - M_physical, 'fro')
    }

    return M_physical, info


# =============================================================================
# 典型样品Mueller矩阵生成
# =============================================================================

def generate_sample_mueller(sample_type: str = 'random', **kwargs) -> np.ndarray:
    """
    生成典型样品的Mueller矩阵

    参数:
        sample_type: 样品类型
            - 'polarizer': 线偏振器
            - 'retarder': 相位延迟器(波片)
            - 'depolarizer': 退偏器
            - 'mixed': 偏振+延迟+退偏组合
            - 'random': 随机物理可行的Mueller矩阵
            - 'custom': 自定义矩阵

    返回:
        4x4 Mueller矩阵
    """
    if sample_type == 'polarizer':
        angle = kwargs.get('angle', np.pi / 4)
        M = polarizer(angle)

    elif sample_type == 'retarder':
        retardance = kwargs.get('retardance', np.pi / 2)
        angle = kwargs.get('angle', np.pi / 4)
        M = retarder(retardance, angle)

    elif sample_type == 'depolarizer':
        diattenuation = kwargs.get('diattenuation', 0.3)
        M = np.eye(4)
        M[0, 0] = 1.0
        M[1, 1] = M[2, 2] = M[3, 3] = 1 - diattenuation

    elif sample_type == 'mixed':
        angle = kwargs.get('angle', np.pi / 6)
        retardance = kwargs.get('retardance', np.pi / 3)
        diattenuation = kwargs.get('diattenuation', 0.2)
        depolarization = kwargs.get('depolarization', 0.9)

        M_pol = polarizer(angle)
        M_ret = retarder(retardance, angle)
        M_depol = np.eye(4)
        M_depol[1:, 1:] *= depolarization

        M = M_pol @ M_ret @ M_depol

    elif sample_type == 'random':
        rng = np.random.default_rng(kwargs.get('seed', None))
        M = rng.standard_normal((4, 4)) * 0.3
        M[0, 0] = 1.0
        for i in range(1, 4):
            for j in range(4):
                if abs(M[i, j]) > abs(M[0, 0]):
                    M[i, j] = M[0, 0] * np.sign(M[i, j]) * 0.9

    elif sample_type == 'custom':
        M = kwargs.get('matrix', np.eye(4))

    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    return M


# =============================================================================
# Lu-Chipman分解与偏振参数提取
# =============================================================================

def _lu_chipman_diattenuation_matrix(D: np.ndarray) -> np.ndarray:
    """
    构造Lu-Chipman分解中的二向色性矩阵 M_D

    二向色性描述了样品对不同偏振态的吸收差异。

    参数:
        D: 二向色性向量 (3,) [D_x, D_y, D_z]

    返回:
        4x4 二向色性Mueller矩阵
    """
    D_mag = np.sqrt(D[0]**2 + D[1]**2 + D[2]**2)

    if D_mag < 1e-15:
        return np.eye(4)

    D_hat = D / D_mag

    factor = np.sqrt(1 - D_mag**2)

    M_D = np.zeros((4, 4))
    M_D[0, 0] = 1.0
    M_D[0, 1:] = D
    M_D[1:, 0] = D
    M_D[1:, 1:] = factor * np.eye(3) + (1 - factor) * np.outer(D_hat, D_hat)

    return M_D


def _inverse_diattenuation_matrix(D: np.ndarray) -> np.ndarray:
    """
    构造二向色性矩阵的逆矩阵 M_D^{-1}

    参数:
        D: 二向色性向量 (3,)

    返回:
        4x4 逆二向色性矩阵
    """
    D_mag = np.sqrt(D[0]**2 + D[1]**2 + D[2]**2)

    if D_mag < 1e-15:
        return np.eye(4)

    D_hat = D / D_mag
    factor = 1.0 / np.sqrt(1 - D_mag**2)

    M_D_inv = np.zeros((4, 4))
    M_D_inv[0, 0] = 1.0
    M_D_inv[0, 1:] = -D
    M_D_inv[1:, 0] = -D
    M_D_inv[1:, 1:] = factor * np.eye(3) + (1 - factor) * np.outer(D_hat, D_hat)

    return M_D_inv


def _lu_chipman_retarder_matrix(R_vec: np.ndarray) -> np.ndarray:
    """
    构造Lu-Chipman分解中的延迟矩阵 M_R

    延迟描述了样品对不同偏振态引入的相位差。

    参数:
        R_vec: 延迟向量 (3,)，其模为总延迟量δ，方向为快轴方向

    返回:
        4x4 延迟Mueller矩阵
    """
    R_mag = np.sqrt(R_vec[0]**2 + R_vec[1]**2 + R_vec[2]**2)

    if R_mag < 1e-15:
        return np.eye(4)

    R_hat = R_vec / R_mag
    c = np.cos(R_mag)
    s = np.sin(R_mag)

    M_R = np.zeros((4, 4))
    M_R[0, 0] = 1.0
    M_R[1:, 1:] = c * np.eye(3) + (1 - c) * np.outer(R_hat, R_hat) + s * np.array([
        [0, -R_hat[2], R_hat[1]],
        [R_hat[2], 0, -R_hat[0]],
        [-R_hat[1], R_hat[0], 0]
    ])

    return M_R


def _inverse_retarder_matrix(R_vec: np.ndarray) -> np.ndarray:
    """
    构造延迟矩阵的逆矩阵 M_R^{-1}

    参数:
        R_vec: 延迟向量 (3,)

    返回:
        4x4 逆延迟矩阵
    """
    return _lu_chipman_retarder_matrix(-R_vec)


def lu_chipman_decomposition(M: np.ndarray) -> Dict:
    """
    对Mueller矩阵执行Lu-Chipman分解

    Lu-Chipman分解将任意Mueller矩阵分解为:
        M = M_Δ · M_R · M_D

    其中:
        - M_D: 二向色性矩阵 (Diattenuation)
        - M_R: 延迟矩阵 (Retardance)
        - M_Δ: 退偏矩阵 (Depolarization)

    分解步骤:
        1. 提取二向色性向量 D = (M_01, M_02, M_03)^T / M_00
        2. 构造二向色性矩阵 M_D 并去除其影响
        3. 从去除二向色性的矩阵中提取延迟向量
        4. 去除延迟影响得到退偏矩阵

    参数:
        M: 4x4 Mueller矩阵 (应为物理可实现的)

    返回:
        包含分解结果和提取参数的字典
    """
    M = M / M[0, 0]

    D = np.array([M[0, 1], M[0, 2], M[0, 3]])
    D_mag = np.sqrt(D[0]**2 + D[1]**2 + D[2]**2)

    if D_mag > 0.9999:
        D = D / D_mag * 0.9999
        D_mag = 0.9999

    M_D = _lu_chipman_diattenuation_matrix(D)
    M_D_inv = _inverse_diattenuation_matrix(D)

    M_prime = M_D_inv @ M

    m_sub = M_prime[1:, 1:]
    U, s, Vt = np.linalg.svd(m_sub)

    M_R_sub = U @ Vt
    M_Delta_sub = (Vt.T @ np.diag(s) @ Vt)

    R_matrix = np.eye(4)
    R_matrix[1:, 1:] = M_R_sub

    Delta_matrix = np.eye(4)
    Delta_matrix[0, 0] = 1.0
    Delta_matrix[1:, 1:] = M_Delta_sub

    if np.linalg.det(M_R_sub) < 0:
        U[:, -1] = -U[:, -1]
        M_R_sub = U @ Vt
        R_matrix[1:, 1:] = M_R_sub

    if abs(M_R_sub[2, 1] - M_R_sub[1, 2]) > 1e-10:
        R1 = (M_R_sub[2, 1] - M_R_sub[1, 2]) / 2
        R2 = (M_R_sub[0, 2] - M_R_sub[2, 0]) / 2
        R3 = (M_R_sub[1, 0] - M_R_sub[0, 1]) / 2
        R_vec = np.array([R1, R2, R3])
    else:
        R_vec = np.zeros(3)

    R_total = np.sqrt(R_vec[0]**2 + R_vec[1]**2 + R_vec[2]**2)

    if abs(R_total) > 1e-10:
        azimuth = np.arctan2(R_vec[0], R_vec[1]) / 2
        ellipticity = np.arctan2(R_vec[2], np.sqrt(R_vec[0]**2 + R_vec[1]**2)) / 2
    else:
        azimuth = 0.0
        ellipticity = 0.0

    depolarization_values = np.diag(M_Delta_sub)
    depolarization_scalar = np.mean(depolarization_values)
    depolarization_linear = np.mean(depolarization_values[:2])
    depolarization_circular = depolarization_values[2]
    depolarization_anisotropy = (
        abs(depolarization_values[0] - depolarization_values[1]) +
        abs(depolarization_values[0] - depolarization_values[2]) +
        abs(depolarization_values[1] - depolarization_values[2])
    ) / 2

    linear_diattenuation = np.sqrt(D[0]**2 + D[1]**2)
    circular_diattenuation = abs(D[2])

    if D_mag > 1e-10:
        diattenuation_azimuth = np.arctan2(D[1], D[0]) / 2
        diattenuation_ellipticity = np.arctan2(D[2], np.sqrt(D[0]**2 + D[1]**2)) / 2
    else:
        diattenuation_azimuth = 0.0
        diattenuation_ellipticity = 0.0

    M_R_reconstructed = _lu_chipman_retarder_matrix(R_vec)

    results = {
        'M_D': M_D,
        'M_R': R_matrix,
        'M_Delta': Delta_matrix,
        'M_R_reconstructed': M_R_reconstructed,
        'diattenuation_vector': D,
        'diattenuation_magnitude': D_mag,
        'linear_diattenuation': linear_diattenuation,
        'circular_diattenuation': circular_diattenuation,
        'diattenuation_azimuth': diattenuation_azimuth,
        'diattenuation_azimuth_deg': np.degrees(diattenuation_azimuth),
        'diattenuation_ellipticity': diattenuation_ellipticity,
        'diattenuation_ellipticity_deg': np.degrees(diattenuation_ellipticity),
        'retardance_vector': R_vec,
        'retardance_total': R_total,
        'retardance_total_deg': np.degrees(R_total),
        'retardance_azimuth': azimuth,
        'retardance_azimuth_deg': np.degrees(azimuth),
        'retardance_ellipticity': ellipticity,
        'retardance_ellipticity_deg': np.degrees(ellipticity),
        'depolarization_values': depolarization_values,
        'depolarization_scalar': depolarization_scalar,
        'depolarization_linear': depolarization_linear,
        'depolarization_circular': depolarization_circular,
        'depolarization_anisotropy': depolarization_anisotropy,
        'decomposition_error': np.linalg.norm(M - Delta_matrix @ R_matrix @ M_D, 'fro')
    }

    return results


def extract_polarization_parameters(M: np.ndarray) -> Dict:
    """
    从Mueller矩阵中提取完整的偏振参数

    这是Lu-Chipman分解的高级接口，专注于提取有物理意义的参数，
    特别适用于组织光学诊断。

    参数:
        M: 4x4 Mueller矩阵

    返回:
        包含所有偏振参数的字典
    """
    lu_chipman = lu_chipman_decomposition(M)

    intensity_transmission = M[0, 0]

    polarization_degree_out = np.sqrt(M[1, 0]**2 + M[2, 0]**2 + M[3, 0]**2) / (M[0, 0] + 1e-15)

    polarization_degree_in = np.sqrt(M[0, 1]**2 + M[0, 2]**2 + M[0, 3]**2) / (M[0, 0] + 1e-15)

    s1 = np.array([1, 1, 0, 0])
    s2 = np.array([1, -1, 0, 0])
    s3 = np.array([1, 0, 1, 0])
    s4 = np.array([1, 0, -1, 0])
    s5 = np.array([1, 0, 0, 1])
    s6 = np.array([1, 0, 0, -1])

    I1 = (M @ s1)[0]
    I2 = (M @ s2)[0]
    I3 = (M @ s3)[0]
    I4 = (M @ s4)[0]
    I5 = (M @ s5)[0]
    I6 = (M @ s6)[0]

    linear_dichroism_0 = (I1 - I2) / (I1 + I2) if (I1 + I2) > 1e-15 else 0
    linear_dichroism_45 = (I3 - I4) / (I3 + I4) if (I3 + I4) > 1e-15 else 0
    circular_dichroism = (I5 - I6) / (I5 + I6) if (I5 + I6) > 1e-15 else 0

    eigenvalues, _, _ = cloude_decomposition(M)
    entropy = 0.0
    for lam in eigenvalues:
        if lam > 1e-15:
            p = lam / (np.sum(eigenvalues) + 1e-15)
            if p > 0:
                entropy -= p * np.log2(p)

    purity = np.sqrt(
        (np.sum(eigenvalues**2) - 1/4) / (3/4)
    ) if np.sum(eigenvalues) > 0 else 0
    purity = np.clip(purity, 0, 1)

    depolarization_index = 1 - (
        np.abs(M[1, 1] - M[2, 2]) + np.abs(M[2, 1] + M[1, 2]) +
        np.abs(M[3, 3] - M[1, 1]) + np.abs(M[3, 2]) + np.abs(M[2, 3])
    ) / (4 * M[0, 0] + 1e-15)

    results = {
        **lu_chipman,
        'intensity_transmission': intensity_transmission,
        'polarization_degree_out': polarization_degree_out,
        'polarization_degree_in': polarization_degree_in,
        'linear_dichroism_0deg': linear_dichroism_0,
        'linear_dichroism_45deg': linear_dichroism_45,
        'circular_dichroism': circular_dichroism,
        'cloude_entropy': entropy,
        'cloude_purity': purity,
        'depolarization_index': depolarization_index,
        'isotropic_amplitude': (M[1, 1] + M[2, 2] + M[3, 3]) / 3,
        'anisotropy_amplitude': np.sqrt(
            (M[1, 1] - M[2, 2])**2 + (M[1, 2] + M[2, 1])**2
        ) / 2,
        'circular_anisotropy': (M[3, 1] - M[1, 3]) / 2
    }

    return results


def print_polarization_parameters(params: Dict, title: str = "偏振参数"):
    """格式化打印提取的偏振参数"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    print(f"\n  [二向色性 (Diattenuation)]")
    print(f"    总二向色性: {params['diattenuation_magnitude']:.6f}")
    print(f"    线二向色性: {params['linear_diattenuation']:.6f}")
    print(f"    圆二向色性: {params['circular_diattenuation']:.6f}")
    print(f"    方位角: {params['diattenuation_azimuth_deg']:.2f}°")
    print(f"    椭偏角: {params['diattenuation_ellipticity_deg']:.2f}°")

    print(f"\n  [延迟 (Retardance)]")
    print(f"    总延迟量: {params['retardance_total_deg']:.2f}°")
    print(f"    方位角: {params['retardance_azimuth_deg']:.2f}°")
    print(f"    椭偏角: {params['retardance_ellipticity_deg']:.2f}°")

    print(f"\n  [退偏 (Depolarization)]")
    print(f"    标量退偏: {params['depolarization_scalar']:.6f}")
    print(f"    线性退偏: {params['depolarization_linear']:.6f}")
    print(f"    圆退偏: {params['depolarization_circular']:.6f}")
    print(f"    各向异性: {params['depolarization_anisotropy']:.6f}")
    print(f"    退偏指数: {params['depolarization_index']:.6f}")

    print(f"\n  [其他参数]")
    print(f"    强度透射: {params['intensity_transmission']:.6f}")
    print(f"    输出偏振度: {params['polarization_degree_out']:.6f}")
    print(f"    线二色性(0°): {params['linear_dichroism_0deg']:.6f}")
    print(f"    线二色性(45°): {params['linear_dichroism_45deg']:.6f}")
    print(f"    圆二色性: {params['circular_dichroism']:.6f}")
    print(f"    Cloude熵: {params['cloude_entropy']:.6f}")
    print(f"    Cloude纯度: {params['cloude_purity']:.6f}")

    print(f"\n  [分解精度]")
    print(f"    重构误差: {params['decomposition_error']:.10f}")

    print(f"{'=' * 60}\n")


def tissue_optics_diagnosis(params: Dict) -> Dict:
    """
    组织光学诊断相关的偏振参数分析

    基于组织光学原理，从偏振参数推断组织状态。
    这是一个基础的诊断框架，实际应用需要结合临床数据校准。

    参数:
        params: extract_polarization_parameters() 返回的参数字典

    返回:
        包含诊断指标的字典
    """
    birefringence_strength = params['retardance_total']

    if params['depolarization_scalar'] > 1e-10:
        birefringence_to_depolarization = params['retardance_total'] / params['depolarization_scalar']
    else:
        birefringence_to_depolarization = float('inf')

    structural_order = params['cloude_purity'] * (1 - params['depolarization_scalar'])

    if params['diattenuation_magnitude'] > 1e-10:
        absorption_anisotropy = params['diattenuation_magnitude']
    else:
        absorption_anisotropy = 0.0

    if params['retardance_total'] > 1e-10 and params['diattenuation_magnitude'] > 1e-10:
        phase_to_amplitude_ratio = params['retardance_total'] / (params['diattenuation_magnitude'] + 1e-15)
    else:
        phase_to_amplitude_ratio = 0.0

    if params['retardance_total_deg'] > 90:
        fiber_alignment = "有序"
    elif params['depolarization_scalar'] > 0.5:
        fiber_alignment = "无序"
    else:
        fiber_alignment = "混合"

    if params['circular_diattenuation'] > 0.01:
        chirality = "手性显著"
    else:
        chirality = "手性微弱"

    diagnosis = {
        'birefringence_strength': birefringence_strength,
        'birefringence_strength_deg': params['retardance_total_deg'],
        'birefringence_to_depolarization': birefringence_to_depolarization,
        'structural_order': structural_order,
        'absorption_anisotropy': absorption_anisotropy,
        'phase_to_amplitude_ratio': phase_to_amplitude_ratio,
        'fiber_alignment_assessment': fiber_alignment,
        'chirality_assessment': chirality,
        'retardance_azimuth_deg': params['retardance_azimuth_deg'],
        'depolarization_scalar': params['depolarization_scalar'],
        'cloude_entropy': params['cloude_entropy'],
        'cloude_purity': params['cloude_purity'],
    }

    return diagnosis


def print_tissue_diagnosis(diag: Dict, title: str = "组织光学诊断分析"):
    """格式化打印组织光学诊断结果"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    print(f"\n  [纤维结构评估]")
    print(f"    双折射强度: {diag['birefringence_strength_deg']:.2f}°")
    print(f"    双折射/退偏比: {diag['birefringence_to_depolarization']:.6f}")
    print(f"    结构有序度: {diag['structural_order']:.6f}")
    print(f"    纤维排列评估: {diag['fiber_alignment_assessment']}")
    print(f"    主方位角: {diag['retardance_azimuth_deg']:.2f}°")

    print(f"\n  [吸收特性]")
    print(f"    吸收各向异性: {diag['absorption_anisotropy']:.6f}")
    print(f"    相位/振幅比: {diag['phase_to_amplitude_ratio']:.6f}")

    print(f"\n  [分子特性]")
    print(f"    手性评估: {diag['chirality_assessment']}")
    print(f"    Cloude熵: {diag['cloude_entropy']:.6f}")
    print(f"    Cloude纯度: {diag['cloude_purity']:.6f}")

    print(f"{'=' * 60}\n")


# =============================================================================
# 可视化辅助
# =============================================================================

def print_mueller_matrix(M: np.ndarray, title: str = "Mueller矩阵", decimals: int = 4):
    """格式化打印Mueller矩阵"""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for i in range(4):
        row = "  ".join([f"{M[i, j]:{10}.{decimals}f}" for j in range(4)])
        print(f"  {row}")
    print(f"{'=' * 50}\n")


# =============================================================================
# 主程序 - 完整演示
# =============================================================================

def main():
    """
    完整的Mueller矩阵偏振计重构演示
    """
    print("=" * 60)
    print("  Mueller矩阵偏振计数据重构演示")
    print("  Mueller Matrix Polarimeter Reconstruction Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 定义样品的真实Mueller矩阵
    # ------------------------------------------------------------------
    print("\n[步骤1] 定义样品真实Mueller矩阵")

    sample_configs = [
        {"type": "polarizer", "angle": np.pi / 4, "name": "线偏振器 (45°)"},
        {"type": "retarder", "retardance": np.pi / 2, "angle": np.pi / 4, "name": "1/4波片 (快轴45°)"},
        {"type": "mixed", "angle": np.pi / 6, "retardance": np.pi / 3, "diattenuation": 0.2, "depolarization": 0.9, "name": "混合样品"},
        {"type": "random", "seed": 42, "name": "随机样品"},
    ]

    for i, cfg in enumerate(sample_configs):
        print(f"\n  样品 {i + 1}: {cfg['name']}")
        M_true = generate_sample_mueller(cfg["type"], **{k: v for k, v in cfg.items() if k not in ['type', 'name']})
        print_mueller_matrix(M_true, f"真实Mueller矩阵 - {cfg['name']}")

    # 使用第一个样品做详细演示
    M_true = generate_sample_mueller(sample_configs[0]["type"],
                                     **{k: v for k, v in sample_configs[0].items() if k not in ['type', 'name']})

    # ------------------------------------------------------------------
    # 2. 方法一: 双旋转波片法 (Double Rotating Retarder)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  方法一: 双旋转波片法 (Double Rotating Retarder)")
    print("=" * 60)

    H_drr, W_drr, configs_drr = build_measurement_matrix_drr(
        n_measurements=36,
        psg_waveplate_retardance=np.pi / 2,
        psa_waveplate_retardance=np.pi / 2
    )

    print(f"\n  测量矩阵 H 维度: {H_drr.shape}")
    print(f"  测量条件数: {np.linalg.cond(H_drr):.2f}")
    print(f"  前3个测量配置的PSG角度: {[c['theta_psg_deg'] for c in configs_drr[:3]]}")
    print(f"  前3个测量配置的PSA角度: {[c['theta_psa_deg'] for c in configs_drr[:3]]}")

    I_drr, I_drr_clean = simulate_measurement(M_true, H_drr, noise_std=0.01, seed=42)

    M_drr, info_drr = reconstruct_mueller(H_drr, I_drr, method='lstsq')

    print_mueller_matrix(M_drr, "DRR重构的Mueller矩阵")

    metrics_drr = evaluate_reconstruction(M_true, M_drr)
    print(f"\n  DRR重构精度评估:")
    print(f"    RMSE: {metrics_drr['rmse']:.6f}")
    print(f"    最大绝对误差: {metrics_drr['max_absolute_error']:.6f}")
    print(f"    平均相对误差: {metrics_drr['mean_relative_error']:.6f}")
    print(f"    Frobenius相对误差: {metrics_drr['relative_frobenius_error']:.6f}")

    # ------------------------------------------------------------------
    # 3. 方法二: 液晶可变延迟器法 (LCVR)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  方法二: 液晶可变延迟器法 (LCVR)")
    print("=" * 60)

    H_lcvr, W_lcvr, configs_lcvr = build_measurement_matrix_lcvr(
        retardance_states=[0, np.pi / 2, np.pi, 3 * np.pi / 2]
    )

    print(f"\n  测量矩阵 H 维度: {H_lcvr.shape}")
    print(f"  测量条件数: {np.linalg.cond(H_lcvr):.2f}")
    print(f"  LCVR延迟量组合数: {len(configs_lcvr)}")

    I_lcvr, I_lcvr_clean = simulate_measurement(M_true, H_lcvr, noise_std=0.01, seed=42)

    M_lcvr, info_lcvr = reconstruct_mueller(H_lcvr, I_lcvr, method='lstsq')

    print_mueller_matrix(M_lcvr, "LCVR重构的Mueller矩阵")

    metrics_lcvr = evaluate_reconstruction(M_true, M_lcvr)
    print(f"\n  LCVR重构精度评估:")
    print(f"    RMSE: {metrics_lcvr['rmse']:.6f}")
    print(f"    最大绝对误差: {metrics_lcvr['max_absolute_error']:.6f}")
    print(f"    平均相对误差: {metrics_lcvr['mean_relative_error']:.6f}")
    print(f"    Frobenius相对误差: {metrics_lcvr['relative_frobenius_error']:.6f}")

    # ------------------------------------------------------------------
    # 4. 噪声鲁棒性测试
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  噪声鲁棒性测试")
    print("=" * 60)

    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    print(f"\n  {'噪声标准差':>12s}  {'DRR_RMSE':>12s}  {'LCVR_RMSE':>12s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}")

    for noise_std in noise_levels:
        I_drr_n, _ = simulate_measurement(M_true, H_drr, noise_std=noise_std, seed=42)
        I_lcvr_n, _ = simulate_measurement(M_true, H_lcvr, noise_std=noise_std, seed=42)

        M_drr_n, _ = reconstruct_mueller(H_drr, I_drr_n, method='lstsq')
        M_lcvr_n, _ = reconstruct_mueller(H_lcvr, I_lcvr_n, method='lstsq')

        err_drr = evaluate_reconstruction(M_true, M_drr_n)['rmse']
        err_lcvr = evaluate_reconstruction(M_true, M_lcvr_n)['rmse']

        print(f"  {noise_std:12.6f}  {err_drr:12.6f}  {err_lcvr:12.6f}")

    # ------------------------------------------------------------------
    # 5. 不同测量次数的影响 (DRR)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  不同测量次数对DRR法的影响")
    print("=" * 60)

    n_meas_list = [16, 20, 24, 30, 36, 48, 60]
    print(f"\n  {'测量次数':>8s}  {'条件数':>10s}  {'RMSE':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}")

    for n in n_meas_list:
        H_test, _, _ = build_measurement_matrix_drr(n_measurements=n)
        cond = np.linalg.cond(H_test)

        I_test, _ = simulate_measurement(M_true, H_test, noise_std=0.01, seed=42)
        M_test, _ = reconstruct_mueller(H_test, I_test, method='lstsq')
        err = evaluate_reconstruction(M_true, M_test)['rmse']

        print(f"  {n:8d}  {cond:10.2f}  {err:10.6f}")

    # ------------------------------------------------------------------
    # 6. 物理可实现性检验与Cloude分解
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  物理可实现性检验与Cloude分解")
    print("=" * 60)

    print(f"\n  真实Mueller矩阵的物理可实现性检验:")
    phys_check_true = check_physical_realizability(M_true)
    print(f"    物理可实现: {phys_check_true['is_physically_realizable']}")
    print(f"    特征值: {[f'{e:.6f}' for e in phys_check_true['eigenvalues']]}")
    print(f"    最小特征值: {phys_check_true['min_eigenvalue']:.8f}")

    print(f"\n  对重构的DRR Mueller矩阵执行Cloude分解:")
    eigenvalues, eigenvectors, pure_muellers = cloude_decomposition(M_drr)
    print(f"    特征值: {[f'{e:.6f}' for e in eigenvalues]}")
    print(f"    最小特征值: {np.min(eigenvalues):.8f}")
    print(f"    特征值和: {np.sum(eigenvalues):.6f}")
    print(f"    非负特征值比例: {np.sum(eigenvalues >= -1e-10) / 4 * 100:.1f}%")

    # ------------------------------------------------------------------
    # 7. 高噪声下的非物理Mueller矩阵修正演示
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  最大似然估计: 投影到物理允许空间")
    print("=" * 60)

    noise_level_high = 0.08
    print(f"\n  使用高噪声 (σ={noise_level_high}) 产生非物理重构...")

    I_high_noise, _ = simulate_measurement(M_true, H_drr, noise_std=noise_level_high, seed=123)
    M_raw_high, _ = reconstruct_mueller(H_drr, I_high_noise, method='lstsq')

    phys_check_raw = check_physical_realizability(M_raw_high)
    print(f"\n  原始重构 (无约束):")
    print(f"    物理可实现: {phys_check_raw['is_physically_realizable']}")
    print(f"    特征值: {[f'{e:.6f}' for e in phys_check_raw['eigenvalues']]}")
    print(f"    最小特征值: {phys_check_raw['min_eigenvalue']:.8f}")
    if phys_check_raw['n_negative_eigenvalues'] > 0:
        print(f"    负特征值数量: {phys_check_raw['n_negative_eigenvalues']}")
        print(f"    负特征值: {[f'{e:.6f}' for e in phys_check_raw['negative_eigenvalues']]}")

    print(f"\n  特征值截断法投影到物理空间:")
    M_proj_clip, info_clip = project_to_physical_space(
        M_raw_high, method='eigenvalue_clipping', eigenvalue_floor=0.0
    )
    phys_check_clip = check_physical_realizability(M_proj_clip)
    print(f"    物理可实现: {phys_check_clip['is_physically_realizable']}")
    print(f"    特征值: {[f'{e:.6f}' for e in phys_check_clip['eigenvalues']]}")
    print(f"    Frobenius修正量: {info_clip['frobenius_change']:.6f}")

    print(f"\n  迭代投影法 (更精确):")
    M_proj_iter, info_iter = project_to_physical_space(
        M_raw_high, method='iterative_projection', eigenvalue_floor=0.0
    )
    phys_check_iter = check_physical_realizability(M_proj_iter)
    print(f"    物理可实现: {phys_check_iter['is_physically_realizable']}")
    print(f"    特征值: {[f'{e:.6f}' for e in phys_check_iter['eigenvalues']]}")
    print(f"    迭代次数: {info_iter['iterations']}")
    print(f"    Frobenius修正量: {info_iter['frobenius_change']:.6f}")

    # ------------------------------------------------------------------
    # 8. 精度对比: 有无物理约束
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  精度对比: 无约束 vs 物理约束")
    print("=" * 60)

    print(f"\n  {'方法':>20s}  {'RMSE':>12s}  {'物理可实现':>10s}  {'最小特征值':>14s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*10}  {'-'*14}")

    metrics_raw = evaluate_reconstruction(M_true, M_raw_high)
    metrics_clip = evaluate_reconstruction(M_true, M_proj_clip)
    metrics_iter = evaluate_reconstruction(M_true, M_proj_iter)

    print(f"  {'无约束最小二乘':>20s}  {metrics_raw['rmse']:12.6f}  "
          f"{phys_check_raw['is_physically_realizable']!s:>10s}  "
          f"{phys_check_raw['min_eigenvalue']:14.8f}")
    print(f"  {'特征值截断':>20s}  {metrics_clip['rmse']:12.6f}  "
          f"{phys_check_clip['is_physically_realizable']!s:>10s}  "
          f"{phys_check_clip['min_eigenvalue']:14.8f}")
    print(f"  {'迭代投影':>20s}  {metrics_iter['rmse']:12.6f}  "
          f"{phys_check_iter['is_physically_realizable']!s:>10s}  "
          f"{phys_check_iter['min_eigenvalue']:14.8f}")

    # ------------------------------------------------------------------
    # 9. 端到端: 带物理约束的重构
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  端到端演示: 带物理约束的Mueller矩阵重构")
    print("=" * 60)

    print(f"\n  使用 reconstruct_mueller_physical() 一步完成重构+投影:")
    M_phys, info_phys = reconstruct_mueller_physical(
        H_drr, I_high_noise,
        reconstruction_method='lstsq',
        projection_method='eigenvalue_clipping'
    )

    print(f"    需要投影: {info_phys['projection_needed']}")
    if info_phys['projection_needed']:
        print(f"    投影方法: {info_phys['projection_info']['method_used']}")
        print(f"    初始最小特征值: {info_phys['physical_check_raw']['min_eigenvalue']:.8f}")
        print(f"    最终最小特征值: {info_phys['physical_check_final']['min_eigenvalue']:.8f}")
        print(f"    残差增加量: {info_phys['residual_increase']:.8f}")
        print(f"    相对残差增加: {info_phys['relative_residual_increase']*100:.4f}%")
        print(f"    Frobenius修正: {info_phys['frobenius_correction']:.6f}")

    # ------------------------------------------------------------------
    # 10. 噪声水平对物理可实现性的影响
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  噪声水平对物理可实现性的影响")
    print("=" * 60)

    noise_sweep = [0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    print(f"\n  {'噪声σ':>8s}  {'物理可实现':>10s}  {'最小特征值':>14s}  "
          f"{'原始RMSE':>12s}  {'修正后RMSE':>12s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*12}")

    for noise_s in noise_sweep:
        I_test, _ = simulate_measurement(M_true, H_drr, noise_std=noise_s, seed=123)
        M_raw, _ = reconstruct_mueller(H_drr, I_test, method='lstsq')
        check_raw = check_physical_realizability(M_raw)

        M_phys, _ = project_to_physical_space(M_raw, method='eigenvalue_clipping')

        err_raw = evaluate_reconstruction(M_true, M_raw)['rmse']
        err_phys = evaluate_reconstruction(M_true, M_phys)['rmse']

        print(f"  {noise_s:8.3f}  {check_raw['is_physically_realizable']!s:>10s}  "
              f"{check_raw['min_eigenvalue']:14.8f}  {err_raw:12.6f}  {err_phys:12.6f}")

    # ------------------------------------------------------------------
    # 11. Lu-Chipman分解与偏振参数提取
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Lu-Chipman分解: 提取样品偏振参数")
    print("=" * 60)

    print(f"\n  [示例1] 对真实样品矩阵执行Lu-Chipman分解")
    params_true = extract_polarization_parameters(M_true)
    print_polarization_parameters(params_true, "真实样品的偏振参数")

    print(f"\n  [示例2] 对重构的DRR Mueller矩阵执行Lu-Chipman分解")
    params_reconstructed = extract_polarization_parameters(M_drr)
    print_polarization_parameters(params_reconstructed, "重构Mueller矩阵的偏振参数")

    print(f"\n  [示例3] 对混合样品执行Lu-Chipman分解")
    M_mixed = generate_sample_mueller('mixed', angle=np.pi/6, retardance=np.pi/3,
                                       diattenuation=0.2, depolarization=0.85)
    print_mueller_matrix(M_mixed, "混合样品Mueller矩阵")
    params_mixed = extract_polarization_parameters(M_mixed)
    print_polarization_parameters(params_mixed, "混合样品偏振参数")

    # ------------------------------------------------------------------
    # 12. 组织光学诊断分析
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  组织光学诊断分析")
    print("=" * 60)

    print(f"\n  [示例] 模拟不同组织状态的偏振特性:")

    print(f"\n  --- 模拟'正常组织' (低双折射, 高退偏) ---")
    M_normal = generate_sample_mueller('mixed',
                                        angle=np.pi/4,
                                        retardance=np.pi/12,
                                        diattenuation=0.05,
                                        depolarization=0.3)
    params_normal = extract_polarization_parameters(M_normal)
    diag_normal = tissue_optics_diagnosis(params_normal)
    print_tissue_diagnosis(diag_normal, "正常组织诊断")

    print(f"\n  --- 模拟'纤维化组织' (高双折射, 有序结构) ---")
    M_fibrosis = generate_sample_mueller('mixed',
                                          angle=np.pi/3,
                                          retardance=np.pi/1.5,
                                          diattenuation=0.15,
                                          depolarization=0.8)
    params_fibrosis = extract_polarization_parameters(M_fibrosis)
    diag_fibrosis = tissue_optics_diagnosis(params_fibrosis)
    print_tissue_diagnosis(diag_fibrosis, "纤维化组织诊断")

    print(f"\n  --- 模拟'肿瘤组织' (无序结构, 高散射) ---")
    M_tumor = generate_sample_mueller('mixed',
                                       angle=np.pi/2,
                                       retardance=np.pi/6,
                                       diattenuation=0.1,
                                       depolarization=0.2)
    params_tumor = extract_polarization_parameters(M_tumor)
    diag_tumor = tissue_optics_diagnosis(params_tumor)
    print_tissue_diagnosis(diag_tumor, "肿瘤组织诊断")

    print(f"\n  [组织参数对比表]")
    print(f"  {'参数':>20s}  {'正常组织':>12s}  {'纤维化':>12s}  {'肿瘤':>12s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*12}")
    print(f"  {'双折射(°)':>20s}  {diag_normal['birefringence_strength_deg']:12.2f}  "
          f"{diag_fibrosis['birefringence_strength_deg']:12.2f}  {diag_tumor['birefringence_strength_deg']:12.2f}")
    print(f"  {'退偏系数':>20s}  {diag_normal['depolarization_scalar']:12.4f}  "
          f"{diag_fibrosis['depolarization_scalar']:12.4f}  {diag_tumor['depolarization_scalar']:12.4f}")
    print(f"  {'结构有序度':>20s}  {diag_normal['structural_order']:12.4f}  "
          f"{diag_fibrosis['structural_order']:12.4f}  {diag_tumor['structural_order']:12.4f}")
    print(f"  {'纤维排列':>20s}  {diag_normal['fiber_alignment_assessment']:>12s}  "
          f"{diag_fibrosis['fiber_alignment_assessment']:>12s}  {diag_tumor['fiber_alignment_assessment']:>12s}")
    print(f"  {'Cloude熵':>20s}  {diag_normal['cloude_entropy']:12.4f}  "
          f"{diag_fibrosis['cloude_entropy']:12.4f}  {diag_tumor['cloude_entropy']:12.4f}")
    print(f"  {'Cloude纯度':>20s}  {diag_normal['cloude_purity']:12.4f}  "
          f"{diag_fibrosis['cloude_purity']:12.4f}  {diag_tumor['cloude_purity']:12.4f}")

    # ------------------------------------------------------------------
    # 13. 偏振参数随样品特性变化
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  偏振参数随样品特性变化分析")
    print("=" * 60)

    print(f"\n  [1] 延迟量对偏振参数的影响")
    print(f"  {'延迟量(°)':>12s}  {'总延迟(°)':>12s}  {'方位角(°)':>12s}  "
          f"{'标量退偏':>12s}  {'Cloude纯度':>12s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    for retardance_deg in [0, 30, 60, 90, 120, 150, 180]:
        M_test = retarder(np.radians(retardance_deg), np.pi/4)
        params = extract_polarization_parameters(M_test)
        print(f"  {retardance_deg:12d}  {params['retardance_total_deg']:12.2f}  "
              f"{params['retardance_azimuth_deg']:12.2f}  {params['depolarization_scalar']:12.4f}  "
              f"{params['cloude_purity']:12.4f}")

    print(f"\n  [2] 二向色性对偏振参数的影响")
    print(f"  {'二向色性':>12s}  {'线二向色性':>12s}  {'圆二向色性':>12s}  "
          f"{'方位角(°)':>12s}  {'强度透射':>12s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    for diat in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        M_test = polarizer(np.arcsin(diat) / 2)
        params = extract_polarization_parameters(M_test)
        print(f"  {diat:12.4f}  {params['linear_diattenuation']:12.4f}  "
              f"{params['circular_diattenuation']:12.4f}  "
              f"{params['diattenuation_azimuth_deg']:12.2f}  "
              f"{params['intensity_transmission']:12.4f}")

    print(f"\n  [3] 退偏系数对偏振参数的影响")
    print(f"  {'退偏系数':>12s}  {'标量退偏':>12s}  {'线性退偏':>12s}  "
          f"{'圆退偏':>12s}  {'退偏指数':>12s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    for depol_coeff in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]:
        M_test = np.eye(4)
        M_test[1:, 1:] *= depol_coeff
        params = extract_polarization_parameters(M_test)
        print(f"  {depol_coeff:12.4f}  {params['depolarization_scalar']:12.4f}  "
              f"{params['depolarization_linear']:12.4f}  "
              f"{params['depolarization_circular']:12.4f}  "
              f"{params['depolarization_index']:12.4f}")

    # ------------------------------------------------------------------
    # 14. 实际使用示例
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  实际使用示例 - 从测量到诊断")
    print("=" * 60)

    print("""
  完整流程: 测量 → 重构 → 物理约束 → 参数提取 → 诊断

    # 1. 构造测量矩阵
    H, W, configs = build_measurement_matrix_drr(n_measurements=36)

    # 2. 获取测量数据
    I = load_your_measurement_data()

    # 3. 带物理约束的重构
    M_physical, info = reconstruct_mueller_physical(H, I)

    # 4. 提取偏振参数 (Lu-Chipman分解)
    params = extract_polarization_parameters(M_physical)

    # 5. 组织光学诊断分析
    diagnosis = tissue_optics_diagnosis(params)

    # 6. 输出结果
    print(f"样品类型: {diagnosis['fiber_alignment_assessment']}")
    print(f"双折射强度: {diagnosis['birefringence_strength_deg']:.2f}°")
    print(f"退偏系数: {diagnosis['depolarization_scalar']:.4f}")
    print(f"结构有序度: {diagnosis['structural_order']:.4f}")
""")

    print("\n" + "=" * 60)
    print("  演示完成!")
    print("=" * 60)

    return {
        'M_true': M_true,
        'M_drr': M_drr,
        'M_lcvr': M_lcvr,
        'metrics_drr': metrics_drr,
        'metrics_lcvr': metrics_lcvr,
        'H_drr': H_drr,
        'H_lcvr': H_lcvr,
        'M_raw_high_noise': M_raw_high,
        'M_proj_clipping': M_proj_clip,
        'M_proj_iterative': M_proj_iter,
        'params_true': params_true,
        'params_mixed': params_mixed,
        'diagnosis_normal': diag_normal,
        'diagnosis_fibrosis': diag_fibrosis,
        'diagnosis_tumor': diag_tumor
    }


if __name__ == '__main__':
    results = main()
