import numpy as np


def select_k_by_energy(S, energy_ratio=0.95):
    """
    基于能量比例自动选择保留的奇异值个数k
    
    保留前k个奇异值，使其平方和占全部奇异值平方和的比例 >= energy_ratio
    
    参数:
        S: 奇异值数组（降序排列）
        energy_ratio: 目标能量保留比例，范围(0, 1]，默认0.95
    
    返回:
        k: 满足能量比例的最小奇异值个数
    """
    total_energy = np.sum(S ** 2)
    if total_energy == 0:
        return len(S)
    cumulative_energy = np.cumsum(S ** 2)
    ratios = cumulative_energy / total_energy
    k = int(np.searchsorted(ratios, energy_ratio) + 1)
    return min(k, len(S))


def select_k_by_elbow(S):
    """
    基于奇异值曲线拐点检测自适应选择k（Knee Point Detection）
    
    使用最大曲率法：将奇异值曲线归一化后，找到曲率最大的点作为拐点。
    曲率公式: κ = |y''| / (1 + y'^2)^(3/2)
    
    参数:
        S: 奇异值数组（降序排列）
    
    返回:
        k: 拐点处的奇异值索引（从1开始计数）
    """
    n = len(S)
    if n <= 2:
        return n

    y = S / S[0] if S[0] > 0 else S
    x = np.arange(n, dtype=float) / (n - 1)

    chord_x = x[-1] - x[0]
    chord_y = y[-1] - y[0]
    chord_len = np.sqrt(chord_x ** 2 + chord_y ** 2)

    if chord_len == 0:
        return n

    distances = np.abs(chord_y * x - chord_x * y + x[-1] * y[0] - y[-1] * x[0]) / chord_len

    k = int(np.argmax(distances) + 1)
    return min(k, n)


def randomized_svd(A, k, n_oversample=10, n_iter=2):
    """
    随机化SVD（Randomized SVD）- 加速大型矩阵的低秩近似
    
    算法:
    1. 生成随机高斯投影矩阵 Ω (n × (k + n_oversample))
    2. 计算 Y = A Ω，找到A的列空间近似
    3. 对Y进行QR分解得到Q（正交基）
    4. 计算小矩阵 B = Q^T A
    5. 对B做标准SVD: B = U_small Σ V^T
    6. 恢复 U = Q U_small
    
    参数:
        A: 输入矩阵 (m × n)
        k: 目标秩
        n_oversample: 过采样参数，增加稳定性，默认10
        n_iter: 幂迭代次数，提高精度，默认2
    
    返回:
        U: 左奇异向量 (m × k)
        S: 奇异值 (k)
        Vt: 右奇异向量转置 (k × n)
    """
    m, n = A.shape
    k_total = min(k + n_oversample, m, n)

    Omega = np.random.randn(n, k_total)
    Y = A @ Omega

    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    Q, _ = np.linalg.qr(Y, mode='reduced')
    B = Q.T @ A
    U_small, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_small

    return U[:, :k], S[:k], Vt[:k, :]


def svd_update_rank1(U, S, Vt, a, axis='row'):
    """
    秩1更新SVD：添加新行或新列后更新SVD分解
    
    当添加新行 a (1 × n)，新矩阵 = [[A], [a]]
    当添加新列 a (m × 1)，新矩阵 = [A, a]
    
    使用Brand算法的简化版本：
    将新增部分投影到现有奇异空间 + 正交分量，然后对角化小矩阵更新。
    
    参数:
        U, S, Vt: 原SVD分解，A = U diag(S) Vt，U∈R^{m×k}, S∈R^k, Vt∈R^{k×n}
        a: 新增行向量(1×n)或列向量(m×1)
        axis: 'row'添加行，'col'添加列
    
    返回:
        U_new, S_new, Vt_new: 更新后的SVD分解
    """
    k = len(S)
    V = Vt.T

    if axis == 'row':
        a = a.reshape(1, -1)
        c = a @ V
        d = a.T - V @ c.T
        r = np.linalg.norm(d)

        if r > 1e-12:
            d_normalized = d / r
            V_extended = np.hstack([V, d_normalized])
        else:
            V_extended = np.hstack([V, np.zeros_like(d)])
            r = 0

        K = np.hstack([np.diag(S), c.T])
        K = np.vstack([K, np.zeros((1, k + 1))])
        K[k, k] = r

        m_new = U.shape[0] + 1
        U_extended = np.zeros((m_new, k + 1))
        U_extended[:U.shape[0], :k] = U
        U_extended[U.shape[0], k] = 1

        Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

        U_new = U_extended @ Uk
        S_new = Sk
        Vt_new = Vkt @ V_extended.T

    elif axis == 'col':
        a = a.reshape(-1, 1)
        c = U.T @ a
        d = a - U @ c
        r = np.linalg.norm(d)

        if r > 1e-12:
            d_normalized = d / r
            U_extended = np.hstack([U, d_normalized])
        else:
            U_extended = np.hstack([U, np.zeros_like(d)])
            r = 0

        K = np.vstack([np.diag(S), np.zeros((1, k))])
        K = np.hstack([K, np.zeros((k + 1, 1))])
        K[:k, k] = c.flatten()
        K[k, k] = r

        n_new = V.shape[0] + 1
        V_extended = np.zeros((n_new, k + 1))
        V_extended[:V.shape[0], :k] = V
        V_extended[V.shape[0], k] = 1

        Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

        U_new = U_extended @ Uk
        S_new = Sk
        Vt_new = Vkt @ V_extended.T

    else:
        raise ValueError("axis must be 'row' or 'col'")

    return U_new, S_new, Vt_new


def svd_truncation(matrix, k=None, energy_ratio=0.95, method='energy', 
                   use_rsvd=False, n_oversample=10, n_iter=2):
    """
    对矩阵进行SVD分解并截断，保留前k个最大奇异值
    
    参数:
        matrix: 输入矩阵 (m x n)
        k: 目标秩，保留前k个最大奇异值。
           若为None，则根据method自动选择k。
        energy_ratio: 能量保留比例，仅在method='energy'且k=None时使用，默认0.95
        method: 自动选择k的方法，'energy'或'elbow'，仅在k=None时使用
        use_rsvd: 是否使用随机化SVD加速，默认False
        n_oversample: RSVD过采样参数，默认10
        n_iter: RSVD幂迭代次数，默认2
    
    返回:
        low_rank_matrix: 低秩近似矩阵 (m x n)
        truncation_error: 截断误差（Frobenius范数）
        k_actual: 实际保留的奇异值个数
    """
    m, n = matrix.shape
    max_k = min(m, n)

    if k is None:
        if use_rsvd:
            k_est = min(100, max_k)
            _, S_est, _ = randomized_svd(matrix, k_est, n_oversample, n_iter)
            if method == 'elbow':
                k = select_k_by_elbow(S_est)
            else:
                k = select_k_by_energy(S_est, energy_ratio)
        else:
            S_full = np.linalg.svd(matrix, compute_uv=False)
            if method == 'elbow':
                k = select_k_by_elbow(S_full)
            else:
                k = select_k_by_energy(S_full, energy_ratio)

    k = min(k, max_k)

    if use_rsvd:
        U, S, Vt = randomized_svd(matrix, k, n_oversample, n_iter)
    else:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        U, S, Vt = U[:, :k], S[:k], Vt[:k, :]

    low_rank_matrix = U @ np.diag(S) @ Vt

    if not use_rsvd:
        S_full = np.linalg.svd(matrix, compute_uv=False)
        if len(S_full) > k:
            truncation_error = np.sqrt(np.sum(S_full[k:] ** 2))
        else:
            truncation_error = 0.0
    else:
        truncation_error = np.linalg.norm(matrix - low_rank_matrix, 'fro')

    return low_rank_matrix, truncation_error, k


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("测试1: 随机化SVD vs 标准SVD 精度与速度对比")
    print("=" * 70)
    m, n, true_rank = 500, 300, 10
    U_true = np.random.randn(m, true_rank)
    V_true = np.random.randn(true_rank, n)
    large_matrix = U_true @ V_true + 0.01 * np.random.randn(m, n)
    k = 10

    import time
    start = time.time()
    approx_full, err_full, _ = svd_truncation(large_matrix, k=k, use_rsvd=False)
    time_full = time.time() - start

    start = time.time()
    approx_rand, err_rand, _ = svd_truncation(large_matrix, k=k, use_rsvd=True, n_oversample=15)
    time_rand = time.time() - start

    print(f"矩阵大小: {m} × {n}")
    print(f"标准SVD  耗时: {time_full:.4f}s, 误差: {err_full:.6f}")
    print(f"随机SVD  耗时: {time_rand:.4f}s, 误差: {err_rand:.6f}")
    print(f"加速比: {time_full / time_rand:.2f}x")
    print(f"近似矩阵差异 (Fro): {np.linalg.norm(approx_full - approx_rand, 'fro'):.6f}")

    print()
    print("=" * 70)
    print("测试2: 在线SVD更新 - 添加新行")
    print("=" * 70)
    m_small, n_small, k_small = 50, 40, 5
    A = np.random.randn(m_small, n_small)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U, S, Vt = U[:, :k_small], S[:k_small], Vt[:k_small, :]

    new_row = np.random.randn(1, n_small)
    A_with_new_row = np.vstack([A, new_row])

    U_row, S_row, Vt_row = svd_update_rank1(U, S, Vt, new_row, axis='row')
    approx_updated = U_row @ np.diag(S_row) @ Vt_row

    U_full, S_full, Vt_full = np.linalg.svd(A_with_new_row, full_matrices=False)
    approx_full = U_full[:, :k_small+1] @ np.diag(S_full[:k_small+1]) @ Vt_full[:k_small+1, :]

    print(f"原矩阵大小: {A.shape}")
    print(f"新增行后大小: {A_with_new_row.shape}")
    print(f"更新后SVD误差: {np.linalg.norm(A_with_new_row - approx_updated, 'fro'):.6f}")
    print(f"完整SVD误差: {np.linalg.norm(A_with_new_row - approx_full, 'fro'):.6f}")
    print(f"两种方法近似差异: {np.linalg.norm(approx_updated - approx_full, 'fro'):.6f}")

    print()
    print("=" * 70)
    print("测试3: 在线SVD更新 - 添加新列")
    print("=" * 70)
    new_col = np.random.randn(m_small, 1)
    A_with_new_col = np.hstack([A, new_col])

    U_col, S_col, Vt_col = svd_update_rank1(U, S, Vt, new_col, axis='col')
    approx_updated_col = U_col @ np.diag(S_col) @ Vt_col

    U_full_c, S_full_c, Vt_full_c = np.linalg.svd(A_with_new_col, full_matrices=False)
    approx_full_c = U_full_c[:, :k_small+1] @ np.diag(S_full_c[:k_small+1]) @ Vt_full_c[:k_small+1, :]

    print(f"原矩阵大小: {A.shape}")
    print(f"新增列后大小: {A_with_new_col.shape}")
    print(f"更新后SVD误差: {np.linalg.norm(A_with_new_col - approx_updated_col, 'fro'):.6f}")
    print(f"完整SVD误差: {np.linalg.norm(A_with_new_col - approx_full_c, 'fro'):.6f}")

    print()
    print("=" * 70)
    print("测试4: 随机化SVD + 自动选择k (95%能量)")
    print("=" * 70)
    approx, err, k_selected = svd_truncation(
        large_matrix, k=None, energy_ratio=0.95, method='energy', use_rsvd=True
    )
    print(f"随机化SVD自动选择 k={k_selected}")
    print(f"截断误差: {err:.6f}")
    approx_full, err_full, k_full = svd_truncation(
        large_matrix, k=None, energy_ratio=0.95, method='energy', use_rsvd=False
    )
    print(f"标准SVD自动选择 k={k_full}")
    print(f"截断误差: {err_full:.6f}")
