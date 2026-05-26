import numpy as np
from scipy.optimize import minimize


def bell_state(name: str) -> np.ndarray:
    name = name.strip().upper()
    if name in ("PHI+", "B00", "00"):
        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    elif name in ("PHI-", "B01", "01"):
        psi = np.array([1, 0, 0, -1]) / np.sqrt(2)
    elif name in ("PSI+", "B10", "10"):
        psi = np.array([0, 1, 1, 0]) / np.sqrt(2)
    elif name in ("PSI-", "B11", "11"):
        psi = np.array([0, 1, -1, 0]) / np.sqrt(2)
    else:
        raise ValueError(
            f"未知的 Bell 态: {name}。可选: Phi+, Phi-, Psi-, Psi-"
        )
    return psi


def ghz_state(n_qubits: int = 3) -> np.ndarray:
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    psi[-1] = 1.0
    psi /= np.sqrt(2)
    return psi


def w_state(n_qubits: int = 3) -> np.ndarray:
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    for i in range(n_qubits):
        idx = 1 << (n_qubits - 1 - i)
        psi[idx] = 1.0
    psi /= np.sqrt(n_qubits)
    return psi


def ideal_state_density_matrix(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).ravel()
    return np.outer(psi, psi.conj())


def _projector_overlap(rho: np.ndarray, psi: np.ndarray) -> complex:
    psi = np.asarray(psi, dtype=complex).ravel()
    return float(np.real(psi.conj() @ rho @ psi))


def fidelity(rho, target="phi+") -> float:
    """
    计算实验重构密度矩阵 rho 与理想 Bell 态之间的保真度 F = <psi|rho|psi>。

    Parameters
    ----------
    rho : array_like
        4x4 实验密度矩阵。
    target : str or np.ndarray
        目标 Bell 态名称（"phi+", "phi-", "psi+", "psi-"）或任意纯态向量。

    Returns
    -------
    float
        保真度，范围在 [0, 1]。
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("rho 必须是 4x4 密度矩阵")

    if isinstance(target, str):
        psi = bell_state(target)
    else:
        psi = np.asarray(target, dtype=complex).ravel()
        if psi.size != 4:
            raise ValueError("target 向量长度必须为 4")
        psi = psi / np.linalg.norm(psi)

    f = _projector_overlap(rho, psi)
    return float(np.clip(f, 0.0, 1.0))


def density_matrix_from_measurements(pauli_expectations: dict) -> np.ndarray:
    """
    从两比特 Pauli 期望值重构密度矩阵。
    pauli_expectations: dict，键为 'II','IX','IY','IZ','XI',...,'ZZ'，
    值为期望值 <A⊗B>（II 必须为 1）。
    """
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = {"I": I, "X": X, "Y": Y, "Z": Z}

    rho = np.zeros((4, 4), dtype=complex)
    for a in "IXYZ":
        for b in "IXYZ":
            key = a + b
            exp = pauli_expectations.get(key, 0.0)
            op = np.kron(paulis[a], paulis[b])
            rho += exp * op
    rho /= 4.0
    return rho


def _triu_params_to_matrix(params: np.ndarray, dim: int) -> np.ndarray:
    """
    将实参数向量转换为 dim×dim 上三角复矩阵 T。
    参数顺序：先所有实部，再所有虚部（仅上三角元素）。
    """
    n_triu = dim * (dim + 1) // 2
    real_parts = params[:n_triu]
    imag_parts = params[n_triu:]

    T = np.zeros((dim, dim), dtype=complex)
    idx = np.triu_indices(dim)
    T[idx] = real_parts + 1j * imag_parts
    return T


def _matrix_to_triu_params(T: np.ndarray) -> np.ndarray:
    """
    将上三角复矩阵 T 展平为实参数向量。
    """
    dim = T.shape[0]
    idx = np.triu_indices(dim)
    real_parts = T[idx].real
    imag_parts = T[idx].imag
    return np.concatenate([real_parts, imag_parts])


def _cholesky_mle_chi2(params: np.ndarray, rho_target: np.ndarray) -> float:
    """
    目标函数：最小化 ||ρ(T) - ρ_target||_F^2
    其中 ρ(T) = (T†T) / tr(T†T)
    """
    dim = rho_target.shape[0]
    T = _triu_params_to_matrix(params, dim)
    rho_est = T.conj().T @ T
    rho_est /= np.trace(rho_est)
    diff = rho_est - rho_target
    return float(np.real(np.vdot(diff, diff)))


def repair_density_matrix(rho: np.ndarray, method: str = "cholesky") -> np.ndarray:
    """
    修复非物理密度矩阵（存在负本征值），投影到半正定、迹为1、厄米的物理空间。

    Parameters
    ----------
    rho : array_like
        待修复的密度矩阵
    method : str
        修复方法：
        - "cholesky": 使用 Cholesky 重新参数化的最大似然估计（推荐，数值更稳定）
        - "spectral": 谱截断法（快速，将负本征值置零后归一化）

    Returns
    -------
    np.ndarray
        修复后的物理密度矩阵
    """
    rho = np.asarray(rho, dtype=complex)
    dim = rho.shape[0]

    rho = (rho + rho.conj().T) / 2.0
    rho = rho / np.real(np.trace(rho))

    if method == "spectral":
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals, 0.0)
        eigvals /= eigvals.sum()
        rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        return rho_fixed

    elif method == "cholesky":
        try:
            L = np.linalg.cholesky(rho + 1e-3 * np.eye(dim))
            T0 = np.triu(L.T)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(rho)
            eigvals = np.maximum(eigvals, 1e-6)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
            T0 = np.triu(L)

        x0 = _matrix_to_triu_params(T0)

        result = minimize(
            _cholesky_mle_chi2,
            x0,
            args=(rho,),
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        T_opt = _triu_params_to_matrix(result.x, dim)
        rho_opt = T_opt.conj().T @ T_opt
        rho_opt /= np.trace(rho_opt)
        rho_opt = (rho_opt + rho_opt.conj().T) / 2.0
        return rho_opt

    else:
        raise ValueError(f"未知方法: {method}，可选: cholesky, spectral")


def check_physicality(rho: np.ndarray) -> dict:
    """
    检查密度矩阵的物理性（厄米性、迹为1、半正定性）。
    """
    rho = np.asarray(rho, dtype=complex)
    dim = rho.shape[0]

    is_hermitian = np.allclose(rho, rho.conj().T)
    trace = np.real(np.trace(rho))
    trace_ok = np.isclose(trace, 1.0)

    eigvals = np.linalg.eigvalsh(rho)
    min_eig = eigvals.min()
    is_psd = min_eig >= -1e-10

    return {
        "is_hermitian": is_hermitian,
        "trace": trace,
        "trace_ok": trace_ok,
        "min_eigenvalue": min_eig,
        "is_psd": is_psd,
        "is_physical": is_hermitian and trace_ok and is_psd,
    }


def partial_transpose(rho: np.ndarray, dims: tuple, subsystem: int = 0) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    d1, d2 = dims
    total_dim = d1 * d2
    if rho.shape != (total_dim, total_dim):
        raise ValueError(f"rho 形状 {rho.shape} 与 dims={dims} 不匹配")

    rho_tensor = rho.reshape(d1, d2, d1, d2)
    if subsystem == 0:
        rho_pt = rho_tensor.transpose(0, 3, 2, 1)
    else:
        rho_pt = rho_tensor.transpose(2, 1, 0, 3)
    return rho_pt.reshape(total_dim, total_dim)


def negativity(rho: np.ndarray, dims: tuple = None, subsystem: int = 0) -> dict:
    """
    计算负性 (Negativity) 和对数负性 (Logarithmic Negativity)。

    N(ρ) = (||ρ^{T_A}||_1 - 1) / 2
    E_N(ρ) = log₂(||ρ^{T_A}||_1)

    Parameters
    ----------
    rho : array_like
        密度矩阵
    dims : tuple, optional
        两体维度 (d1, d2)，默认为 (2, 2)
    subsystem : int
        对哪个子系统进行部分转置 (0 或 1)

    Returns
    -------
    dict
        包含 'negativity' 和 'log_negativity'
    """
    rho = np.asarray(rho, dtype=complex)
    if dims is None:
        n = int(np.log2(rho.shape[0]))
        dims = (2, 2 ** (n - 1))

    rho_pt = partial_transpose(rho, dims, subsystem)
    trace_norm = np.sum(np.abs(np.linalg.eigvalsh(rho_pt)))
    neg = (trace_norm - 1) / 2
    log_neg = np.log2(trace_norm) if trace_norm > 0 else -np.inf

    return {
        "negativity": float(max(0, neg)),
        "log_negativity": float(max(0, log_neg)),
        "trace_norm_pt": float(trace_norm),
    }


def witness_operator(state_name: str, n_qubits: int = 2) -> np.ndarray:
    """
    生成纠缠目击算子 W。

    对于目标纯态 |ψ⟩，目击算子为 W = αI - |ψ⟩⟨ψ|，
    其中 α 是 |ψ⟩ 在可分态中的最大重叠。

    Bell 态: α = 1/2，W = I/2 - |Φ+⟩⟨Φ+|
    GHZ 态 (3 qubit): α = 1/2，W = I/2 - |GHZ⟩⟨GHZ|
    W 态 (3 qubit): α = 2/3，W = 2I/3 - |W⟩⟨W|
    """
    state_name = state_name.strip().upper()

    if state_name.startswith("PHI") or state_name.startswith("PSI") or state_name.startswith("B"):
        psi = bell_state(state_name)
        alpha = 0.5
    elif state_name == "GHZ":
        psi = ghz_state(n_qubits)
        alpha = 0.5
    elif state_name == "W":
        psi = w_state(n_qubits)
        if n_qubits == 3:
            alpha = 2.0 / 3.0
        else:
            alpha = 0.5
    else:
        raise ValueError(f"未知的态类型: {state_name}")

    dim = len(psi)
    W = alpha * np.eye(dim) - np.outer(psi, psi.conj())
    return W


def entanglement_witness(rho: np.ndarray, state_name: str, n_qubits: int = None) -> dict:
    """
    计算纠缠目击期望值 ⟨W⟩ = tr(Wρ)。

    若 ⟨W⟩ < 0，则 ρ 是纠缠态（对于该目击而言）。

    Parameters
    ----------
    rho : array_like
        密度矩阵
    state_name : str
        目标态名称："phi+", "phi-", "psi+", "psi-", "GHZ", "W"
    n_qubits : int, optional
        比特数（仅对 GHZ 和 W 态需要）

    Returns
    -------
    dict
        包含 'expectation_value' 和 'is_entangled'
    """
    rho = np.asarray(rho, dtype=complex)
    state_name_upper = state_name.strip().upper()

    if n_qubits is None:
        if state_name_upper in ("GHZ", "W"):
            n_qubits = int(np.log2(rho.shape[0]))
        else:
            n_qubits = 2

    W = witness_operator(state_name_upper, n_qubits)
    expectation = float(np.real(np.trace(W @ rho)))

    return {
        "expectation_value": expectation,
        "is_entangled": expectation < 0,
        "witness": state_name_upper,
        "threshold": 0.0,
    }


def concurrence(rho: np.ndarray) -> float:
    """
    计算两比特纠缠的 Concurrence。

    C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    其中 λᵢ 是 √(√ρ ρ̃ √ρ) 的本征值（降序排列），
    ρ̃ = (σ_y⊗σ_y)ρ*(σ_y⊗σ_y)
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("Concurrence 仅支持两比特系统 (4x4)")

    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Y_Y = np.kron(sigma_y, sigma_y)
    rho_tilde = Y_Y @ rho.conj() @ Y_Y

    sqrt_rho = _sqrt_matrix(rho)
    M = sqrt_rho @ rho_tilde @ sqrt_rho
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.sort(np.abs(eigvals))[::-1]

    C = max(0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3])
    return float(C)


def _sqrt_matrix(M: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def entanglement_of_formation(rho: np.ndarray) -> float:
    """
    计算两比特纠缠的形成纠缠 (Entanglement of Formation)。

    E_f(C) = H((1 + √(1 - C²)) / 2)
    """
    C = concurrence(rho)
    if C == 0:
        return 0.0

    x = (1 + np.sqrt(1 - C ** 2)) / 2
    x = np.clip(x, 0, 1)

    if x == 0 or x == 1:
        return 0.0

    entropy = -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    return float(entropy)


if __name__ == "__main__":
    print("=" * 70)
    print("1. Bell 态保真度测试")
    print("=" * 70)
    rho_ideal = ideal_state_density_matrix(bell_state("phi+"))
    print("理想 Φ+ 密度矩阵:")
    print(rho_ideal)
    print("保真度 F(Φ+, ρ_理想) =", fidelity(rho_ideal, "phi+"))

    noise = 0.1
    rho_noisy = (1 - noise) * rho_ideal + noise * np.eye(4) / 4
    print("\n含 10% 白噪声的 ρ:")
    print(rho_noisy)
    print("保真度 F(Φ+, ρ_noisy) =", fidelity(rho_noisy, "phi+"))

    exps = {
        "II": 1.0, "IZ": 0.0, "ZI": 0.0, "ZZ": 1.0,
        "XX": 1.0, "YY": -1.0,
    }
    rho_rec = density_matrix_from_measurements(exps)
    print("\n由 Pauli 期望值重构的 ρ:")
    print(rho_rec)
    print("保真度 F(Φ+, ρ_rec) =", fidelity(rho_rec, "phi+"))

    print("\n" + "=" * 70)
    print("2. 密度矩阵物理性修复测试")
    print("=" * 70)

    delta = np.zeros((4, 4), dtype=complex)
    delta[0, 0] = -0.15
    delta[3, 3] = 0.15
    rho_bad = rho_ideal + delta

    print("\n非物理 ρ（人为注入负本征值）:")
    print(rho_bad)
    phys_check = check_physicality(rho_bad)
    print(f"物理性检查: {phys_check}")

    rho_cholesky = repair_density_matrix(rho_bad, method="cholesky")
    print("\n=== Cholesky MLE 修复后 ===")
    print(rho_cholesky)
    phys_chol = check_physicality(rho_cholesky)
    print(f"物理性检查: {phys_chol}")

    print("\n" + "=" * 70)
    print("3. 两比特纠缠度量：Negativity, Concurrence, EoF")
    print("=" * 70)

    rho_bell = ideal_state_density_matrix(bell_state("phi+"))
    print("\n理想 Bell 态 |Φ+⟩:")
    print(f"  Negativity: {negativity(rho_bell)}")
    print(f"  Concurrence: {concurrence(rho_bell)}")
    print(f"  Entanglement of Formation: {entanglement_of_formation(rho_bell)}")

    rho_separable = np.eye(4) / 4
    print("\n完全可分态 (I/4):")
    print(f"  Negativity: {negativity(rho_separable)}")
    print(f"  Concurrence: {concurrence(rho_separable)}")
    print(f"  Entanglement of Formation: {entanglement_of_formation(rho_separable)}")

    rho_werner = 0.7 * rho_bell + 0.3 * np.eye(4) / 4
    print("\nWerner 态 (70% |Φ+⟩ + 30% 噪声):")
    print(f"  Negativity: {negativity(rho_werner)}")
    print(f"  Concurrence: {concurrence(rho_werner)}")
    print(f"  Entanglement of Formation: {entanglement_of_formation(rho_werner)}")

    print("\n" + "=" * 70)
    print("4. 纠缠目击 (Entanglement Witness) 测试")
    print("=" * 70)

    print("\nBell 态 |Φ+⟩ 的目击检测:")
    w_result = entanglement_witness(rho_bell, "phi+")
    print(f"  {w_result}")

    print("\nWerner 态的目击检测:")
    w_result = entanglement_witness(rho_werner, "phi+")
    print(f"  {w_result}")

    print("\n可分态 I/4 的目击检测 (期望非纠缠):")
    w_result = entanglement_witness(rho_separable, "phi+")
    print(f"  {w_result}")

    print("\n" + "=" * 70)
    print("5. 三量子比特 GHZ 态和 W 态测试")
    print("=" * 70)

    rho_ghz = ideal_state_density_matrix(ghz_state(3))
    print("\n三比特 GHZ 态 |GHZ⟩ = (|000⟩+|111⟩)/√2:")
    print(f"  部分转置 Negativity (A|BC): {negativity(rho_ghz, dims=(2, 4), subsystem=0)}")
    print(f"  GHZ 目击: {entanglement_witness(rho_ghz, 'GHZ', n_qubits=3)}")

    rho_w = ideal_state_density_matrix(w_state(3))
    print("\n三比特 W 态 |W⟩ = (|001⟩+|010⟩+|100⟩)/√3:")
    print(f"  部分转置 Negativity (A|BC): {negativity(rho_w, dims=(2, 4), subsystem=0)}")
    print(f"  W 目击: {entanglement_witness(rho_w, 'W', n_qubits=3)}")

    rho_ghz_noisy = 0.8 * rho_ghz + 0.2 * np.eye(8) / 8
    print("\n含 20% 噪声的 GHZ 态:")
    print(f"  GHZ 目击: {entanglement_witness(rho_ghz_noisy, 'GHZ', n_qubits=3)}")
    print(f"  Negativity (A|BC): {negativity(rho_ghz_noisy, dims=(2, 4), subsystem=0)}")

    print("\n" + "=" * 70)
    print("6. 实验数据示例：检测纠缠是否存在")
    print("=" * 70)

    exps_experiment = {
        "II": 1.0, "IZ": 0.05, "ZI": -0.03, "ZZ": 0.92,
        "XX": 0.88, "YY": -0.85,
    }
    rho_exp = density_matrix_from_measurements(exps_experiment)
    rho_exp_fixed = repair_density_matrix(rho_exp)

    print("\n实验重构密度矩阵 (修复后):")
    print(rho_exp_fixed)
    print(f"\n物理性检查: {check_physicality(rho_exp_fixed)}")
    print(f"与 |Φ+⟩ 的保真度: {fidelity(rho_exp_fixed, 'phi+'):.6f}")
    print(f"Negativity: {negativity(rho_exp_fixed)}")
    print(f"Concurrence: {concurrence(rho_exp_fixed):.6f}")
    print(f"纠缠目击 (Φ+): {entanglement_witness(rho_exp_fixed, 'phi+')}")

    if entanglement_witness(rho_exp_fixed, 'phi+')['is_entangled']:
        print("\n✅ 结论：实验数据证实存在纠缠！")
    else:
        print("\n⚠️ 结论：未能检测到纠缠。")
