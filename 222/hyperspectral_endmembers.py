import numpy as np
from scipy.linalg import orth
from typing import Tuple


def vca(data: np.ndarray, p: int, snr_input: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vertex Component Analysis (VCA) 端元提取算法

    参数:
        data: 高光谱数据矩阵，形状 (n_bands, n_pixels)
        p: 端元数目
        snr_input: 信噪比（可选），如果为None则自动估计

    返回:
        E: 端元光谱矩阵，形状 (n_bands, p)
        indices: 提取的端元在原始数据中的索引
        C: 投影后的数据
    """
    n_bands, n_pixels = data.shape

    r_m = np.mean(data, axis=1, keepdims=True)
    R_o = data - r_m

    U, S, _ = np.linalg.svd(R_o @ R_o.T / n_pixels, full_matrices=True)
    U = U[:, :p]

    if snr_input is None:
        snr = estimate_snr(data, R_o, U, p)
    else:
        snr = snr_input

    if snr is None or snr > 15 + 10 * np.log10(p):
        d = p - 1
        proj = U[:, :d]
        X_p = proj.T @ R_o
        c = np.max(np.sum(X_p ** 2, axis=0)) ** 0.5
        Y = np.vstack([X_p, c * np.ones((1, n_pixels))])
        dim_Y = p
    else:
        d = p - 1
        proj = np.hstack([U[:, :d], r_m / np.linalg.norm(r_m)])
        X_p = proj.T @ R_o
        c = np.max(np.sum(X_p ** 2, axis=0)) ** 0.5
        Y = np.vstack([X_p, c * np.ones((1, n_pixels))])
        dim_Y = p + 1

    e = np.zeros((dim_Y, dim_Y))
    e[-1, 0] = 1

    indices = np.zeros(p, dtype=int)
    A = np.zeros((dim_Y, p))

    for i in range(p):
        w = np.random.randn(dim_Y, 1)
        f = w - A @ np.linalg.pinv(A) @ w
        f = f / np.linalg.norm(f)

        v = f.T @ Y
        indices[i] = np.argmax(np.abs(v))
        A[:, i] = Y[:, indices[i]]

    C = U[:, :p]
    E = data[:, indices]

    return E, indices, C


def estimate_snr(data: np.ndarray, R_o: np.ndarray, U: np.ndarray, p: int) -> float:
    """
    估计高光谱数据的信噪比
    """
    n_bands, n_pixels = data.shape

    if n_bands < p:
        return None

    Ud = U[:, :p]
    Rp = Ud @ Ud.T @ R_o
    noise = R_o - Rp

    signal_power = np.mean(Rp ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return np.inf

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def estimate_noise_covariance(data: np.ndarray) -> np.ndarray:
    """
    估计噪声协方差矩阵（使用相邻像素差分法）

    参数:
        data: 高光谱数据，形状 (n_bands, n_pixels)

    返回:
        noise_cov: 噪声协方差矩阵 (n_bands, n_bands)
    """
    n_bands, n_pixels = data.shape

    if n_pixels < 2:
        return np.eye(n_bands) * 1e-6

    diff = data[:, 1:] - data[:, :-1]
    noise_cov = (diff @ diff.T) / (2 * (n_pixels - 1))

    reg = 1e-6 * np.eye(n_bands)
    noise_cov = noise_cov + reg

    return noise_cov


def mnf_denoise(data: np.ndarray, n_components: int = None,
                variance_ratio: float = 0.99) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    最小噪声分离（Minimum Noise Fraction, MNF）降噪

    参数:
        data: 高光谱数据矩阵，形状 (n_bands, n_pixels)
        n_components: 保留的MNF成分数，如为None则根据variance_ratio自动确定
        variance_ratio: 方差解释率阈值，用于自动确定成分数

    返回:
        denoised_data: 降噪后的数据 (n_bands, n_pixels)
        eigenvalues: MNF特征值
        n_components: 实际使用的成分数
    """
    n_bands, n_pixels = data.shape

    data_mean = np.mean(data, axis=1, keepdims=True)
    data_centered = data - data_mean

    signal_cov = (data_centered @ data_centered.T) / (n_pixels - 1)
    noise_cov = estimate_noise_covariance(data)

    try:
        L = np.linalg.cholesky(noise_cov)
        L_inv = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        eigvals_noise, eigvecs_noise = np.linalg.eigh(noise_cov)
        eigvals_noise = np.clip(eigvals_noise, 1e-8, None)
        L_inv = eigvecs_noise @ np.diag(1.0 / np.sqrt(eigvals_noise)) @ eigvecs_noise.T

    whitened_cov = L_inv @ signal_cov @ L_inv.T

    eigenvalues, eigenvectors = np.linalg.eigh(whitened_cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    mnf_transform = L_inv.T @ eigenvectors

    eigenvalues = np.maximum(eigenvalues, 1e-10)

    if n_components is None:
        valid_eigs = eigenvalues[eigenvalues > 1e-6]
        total_variance = np.sum(valid_eigs)
        if total_variance > 0:
            cumulative_variance = np.cumsum(valid_eigs) / total_variance
            n_components = np.searchsorted(cumulative_variance, variance_ratio) + 1
            n_components = min(n_components, len(valid_eigs))
        else:
            n_components = min(10, n_bands)
        n_components = max(2, min(n_components, n_bands))

    projection = mnf_transform[:, :n_components]
    mnf_components = projection.T @ data_centered
    denoised_centered = projection @ mnf_components

    denoised_data = denoised_centered + data_mean

    return denoised_data, eigenvalues, n_components


def pca_denoise(data: np.ndarray, n_components: int = None,
                variance_ratio: float = 0.99) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    主成分分析（PCA）降噪

    参数:
        data: 高光谱数据矩阵，形状 (n_bands, n_pixels)
        n_components: 保留的主成分数，如为None则根据variance_ratio自动确定
        variance_ratio: 方差解释率阈值，用于自动确定成分数

    返回:
        denoised_data: 降噪后的数据 (n_bands, n_pixels)
        eigenvalues: PCA特征值
        n_components: 实际使用的成分数
    """
    n_bands, n_pixels = data.shape

    data_mean = np.mean(data, axis=1, keepdims=True)
    data_centered = data - data_mean

    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    eigenvalues = S ** 2 / (n_pixels - 1)

    if n_components is None:
        total_variance = np.sum(eigenvalues)
        cumulative_variance = np.cumsum(eigenvalues) / total_variance
        n_components = np.searchsorted(cumulative_variance, variance_ratio) + 1
        n_components = min(n_components, n_bands)

    U_reduced = U[:, :n_components]
    S_reduced = S[:n_components]
    Vt_reduced = Vt[:n_components, :]

    denoised_centered = U_reduced @ np.diag(S_reduced) @ Vt_reduced
    denoised_data = denoised_centered + data_mean

    return denoised_data, eigenvalues, n_components


def hfc_endmember_count(data: np.ndarray, alpha: float = 0.99,
                        max_p: int = 30,
                        use_improved: bool = True) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Harsanyi-Farrand-Chang (HFC) 方法自动估计端元数量

    参数:
        data: 高光谱数据矩阵，形状 (n_bands, n_pixels)
        alpha: 显著性水平，默认0.99（降低以提高检测率）
        max_p: 最大估计的端元数
        use_improved: 是否使用改进的HFC方法

    返回:
        p: 估计的端元数量
        eigenvalues: 协方差矩阵的特征值
        thresholds: 每个特征值对应的HFC阈值
    """
    n_bands, n_pixels = data.shape

    data_mean = np.mean(data, axis=1, keepdims=True)
    data_centered = data - data_mean

    cov_matrix = (data_centered @ data_centered.T) / (n_pixels - 1)

    eigenvalues_cov, _ = np.linalg.eigh(cov_matrix)
    sort_idx = np.argsort(eigenvalues_cov)[::-1]
    eigenvalues_cov = eigenvalues_cov[sort_idx]

    from scipy.stats import norm
    z_alpha = norm.ppf(alpha)

    if use_improved:
        eigenvalues_log = np.log(np.maximum(eigenvalues_cov, 1e-10))
        diff_log = -np.diff(eigenvalues_log)

        noise_floor = np.median(eigenvalues_cov[-max(10, n_bands // 5):])
        adjusted_eigenvalues = np.maximum(eigenvalues_cov - noise_floor * 0.5, 1e-10)

        thresholds = np.zeros(n_bands)
        for i in range(n_bands):
            if i == 0:
                thresholds[i] = np.inf
            else:
                sigma_sq = 2 * adjusted_eigenvalues[i] ** 2 / n_pixels
                thresholds[i] = z_alpha * np.sqrt(sigma_sq)

        p = 0
        for i in range(1, min(max_p + 1, n_bands)):
            diff = adjusted_eigenvalues[i - 1] - adjusted_eigenvalues[i]
            if diff > thresholds[i]:
                p = i
            else:
                if p > 0:
                    break

        if p == 0:
            max_drop_idx = np.argmax(diff_log[:min(max_p, len(diff_log))])
            p = max_drop_idx + 1

    else:
        correlation_matrix = np.corrcoef(data)
        eigenvalues_corr, _ = np.linalg.eigh(correlation_matrix)
        sort_idx = np.argsort(eigenvalues_corr)[::-1]
        eigenvalues_corr = eigenvalues_corr[sort_idx]

        thresholds = np.zeros(n_bands)
        for i in range(n_bands):
            if i == 0:
                thresholds[i] = np.inf
            else:
                term1 = 2 * eigenvalues_cov[i] * eigenvalues_corr[i]
                term2 = z_alpha * np.sqrt(2 * (eigenvalues_cov[i] ** 2) * (eigenvalues_corr[i] ** 2) / n_pixels)
                thresholds[i] = term1 + term2

        p = 0
        for i in range(1, min(max_p, n_bands)):
            diff = eigenvalues_cov[i - 1] - eigenvalues_cov[i]
            if diff > thresholds[i]:
                p = i + 1
            else:
                break

    p = max(1, min(p, max_p))

    return p, eigenvalues_cov, thresholds


def osp(data: np.ndarray, p: int, max_iter: int = 10, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal Subspace Projection (OSP) 端元提取算法

    参数:
        data: 高光谱数据矩阵，形状 (n_bands, n_pixels)
        p: 端元数目
        max_iter: 最大迭代次数
        tol: 收敛阈值

    返回:
        E: 端元光谱矩阵，形状 (n_bands, p)
        indices: 提取的端元在原始数据中的索引
    """
    n_bands, n_pixels = data.shape

    data_mean = np.mean(data, axis=1, keepdims=True)
    data_centered = data - data_mean

    U, _, _ = np.linalg.svd(data_centered, full_matrices=False)
    U = U[:, :p-1]

    P_U_orth = np.eye(n_bands) - U @ U.T

    indices = np.zeros(p, dtype=int)
    E = np.zeros((n_bands, p))

    first_idx = np.argmax(np.linalg.norm(P_U_orth @ data, axis=0))
    indices[0] = first_idx
    E[:, 0] = data[:, first_idx]

    for i in range(1, p):
        E_sub = E[:, :i]
        E_orth = orth(E_sub)
        P_orth = np.eye(n_bands) - E_orth @ E_orth.T

        scores = np.linalg.norm(P_orth @ data, axis=0)
        for j in range(i):
            scores[indices[j]] = -np.inf

        new_idx = np.argmax(scores)
        indices[i] = new_idx
        E[:, i] = data[:, new_idx]

    E, indices = refine_endmembers_osp(data, E, indices, max_iter, tol)

    return E, indices


def refine_endmembers_osp(data: np.ndarray, E: np.ndarray, indices: np.ndarray,
                          max_iter: int, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用OSP迭代优化端元
    """
    n_bands, n_pixels = data.shape
    p = E.shape[1]

    prev_volume = 0

    for iteration in range(max_iter):
        for i in range(p):
            E_sub = np.delete(E, i, axis=1)
            E_orth = orth(E_sub)
            P_orth = np.eye(n_bands) - E_orth @ E_orth.T

            scores = np.linalg.norm(P_orth @ data, axis=0)
            new_idx = np.argmax(scores)
            indices[i] = new_idx
            E[:, i] = data[:, new_idx]

        current_volume = abs(np.linalg.det(E.T @ E))
        if abs(current_volume - prev_volume) < tol * prev_volume:
            break
        prev_volume = current_volume

    return E, indices


def extract_endmembers(data: np.ndarray, p: int = None, method: str = 'vca',
                       snr_input: float = None, denoise: str = None,
                       denoise_components: int = None,
                       denoise_variance_ratio: float = 0.99,
                       hfc_alpha: float = 0.99, hfc_max_p: int = 30,
                       hfc_improved: bool = True
                       ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    端元提取主函数（支持降噪和端元数量自动估计）

    参数:
        data: 高光谱数据矩阵，形状 (n_bands, n_pixels) 或 (n_rows, n_cols, n_bands)
        p: 端元数目，如为None则使用HFC方法自动估计
        method: 'vca' 或 'osp'
        snr_input: 信噪比（仅VCA使用）
        denoise: 降噪方法，None/'mnf'/'pca'
        denoise_components: 降噪保留的成分数，如为None则根据方差比例自动确定
        denoise_variance_ratio: 降噪时的方差解释率阈值（0-1），默认0.99
        hfc_alpha: HFC方法的显著性水平，默认0.99
        hfc_max_p: HFC方法估计的最大端元数，默认30
        hfc_improved: 是否使用改进的HFC方法，默认True

    返回:
        E: 端元光谱矩阵，形状 (n_bands, p)
        indices: 提取的端元在原始数据中的索引（展平后）
        info: 包含额外信息的字典
            - 'p_estimated': 实际使用的端元数
            - 'p_hfc': HFC估计的端元数（如果使用）
            - 'denoise_method': 使用的降噪方法
            - 'denoise_components': 降噪保留的成分数
            - 'eigenvalues': 降噪或HFC的特征值
    """
    original_shape = data.shape
    if data.ndim == 3:
        n_rows, n_cols, n_bands = data.shape
        data_2d = data.reshape(-1, n_bands).T
    elif data.ndim == 2:
        if data.shape[0] > data.shape[1]:
            data_2d = data.T
        else:
            data_2d = data
    else:
        raise ValueError("数据维度必须是2或3")

    info = {}

    denoised_data = data_2d
    denoise_eigenvalues = None
    n_denoise_components = None

    if denoise is not None:
        denoise = denoise.lower()
        info['denoise_method'] = denoise

        if denoise == 'mnf':
            print(f"  进行MNF降噪 (方差比例: {denoise_variance_ratio})...")
            denoised_data, denoise_eigenvalues, n_denoise_components = mnf_denoise(
                data_2d, denoise_components, denoise_variance_ratio
            )
            info['denoise_components'] = n_denoise_components
            info['eigenvalues'] = denoise_eigenvalues
            print(f"  MNF降噪完成，保留 {n_denoise_components} 个成分")

        elif denoise == 'pca':
            print(f"  进行PCA降噪 (方差比例: {denoise_variance_ratio})...")
            denoised_data, denoise_eigenvalues, n_denoise_components = pca_denoise(
                data_2d, denoise_components, denoise_variance_ratio
            )
            info['denoise_components'] = n_denoise_components
            info['eigenvalues'] = denoise_eigenvalues
            print(f"  PCA降噪完成，保留 {n_denoise_components} 个成分")

        else:
            raise ValueError(f"不支持的降噪方法: {denoise}，请使用 'mnf' 或 'pca'")
    else:
        info['denoise_method'] = None
        info['denoise_components'] = None

    if p is None:
        print(f"  使用HFC方法自动估计端元数量 (alpha={hfc_alpha}, improved={hfc_improved})...")
        p_hfc, hfc_eigenvalues, hfc_thresholds = hfc_endmember_count(
            denoised_data, hfc_alpha, hfc_max_p, hfc_improved
        )
        p = p_hfc
        info['p_hfc'] = p_hfc
        info['hfc_eigenvalues'] = hfc_eigenvalues
        info['hfc_thresholds'] = hfc_thresholds
        print(f"  HFC估计端元数量: {p}")
    else:
        info['p_hfc'] = None

    info['p_estimated'] = p

    method = method.lower()
    if method == 'vca':
        E, indices, _ = vca(denoised_data, p, snr_input)
    elif method == 'osp':
        E, indices = osp(denoised_data, p)
    else:
        raise ValueError(f"不支持的方法: {method}，请使用 'vca' 或 'osp'")

    return E, indices, info


def fcls_unmixing(data: np.ndarray, endmembers: np.ndarray,
                  max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """
    全约束最小二乘（Fully Constrained Least Squares, FCLS）丰度反演

    参数:
        data: 高光谱数据，形状 (n_bands, n_pixels) 或 (n_rows, n_cols, n_bands)
        endmembers: 端元光谱矩阵，形状 (n_bands, n_endmembers)
        max_iter: 最大迭代次数
        tol: 收敛阈值

    返回:
        abundances: 丰度矩阵，形状 (n_endmembers, n_pixels) 或 (n_rows, n_cols, n_endmembers)
    """
    original_shape = data.shape
    n_endmembers = endmembers.shape[1]

    if data.ndim == 3:
        n_rows, n_cols, n_bands = data.shape
        data_2d = data.reshape(-1, n_bands).T
    elif data.ndim == 2:
        if data.shape[0] > data.shape[1]:
            data_2d = data.T
        else:
            data_2d = data
    else:
        raise ValueError("数据维度必须是2或3")

    n_bands, n_pixels = data_2d.shape

    E = endmembers
    ETE = E.T @ E
    ETy = E.T @ data_2d

    try:
        ETE_inv = np.linalg.inv(ETE + 1e-8 * np.eye(n_endmembers))
        abundances_unconstrained = ETE_inv @ ETy
    except np.linalg.LinAlgError:
        abundances_unconstrained = np.linalg.lstsq(ETE, ETy, rcond=None)[0]

    abundances = np.zeros_like(abundances_unconstrained)

    for p in range(n_pixels):
        S = []
        w = np.ones(n_endmembers, dtype=bool)
        y = data_2d[:, p:p+1]
        alpha = abundances_unconstrained[:, p:p+1].copy()

        for iteration in range(max_iter):
            alpha_neg = alpha.flatten() < -tol
            if not np.any(alpha_neg) or len(S) >= n_endmembers:
                break

            w[alpha_neg] = False
            S = np.where(w)[0].tolist()

            if len(S) == 0:
                break

            E_S = E[:, S]
            try:
                alpha_S = np.linalg.lstsq(E_S, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            alpha_new = np.zeros((n_endmembers, 1))
            alpha_new[S] = alpha_S
            alpha_new[~w] = 0

            alpha = alpha_new

            if np.all(alpha >= -tol):
                break

        alpha = np.maximum(alpha, 0)
        sum_alpha = np.sum(alpha)
        if sum_alpha > tol:
            alpha = alpha / sum_alpha

        abundances[:, p] = alpha.flatten()

    if len(original_shape) == 3:
        n_rows, n_cols, n_bands = original_shape
        abundances = abundances.T.reshape(n_rows, n_cols, n_endmembers)

    return abundances


def classify_subpixel(abundances: np.ndarray,
                      threshold: float = 0.1,
                      endmember_names: list = None) -> np.ndarray:
    """
    亚像元分类 - 根据丰度分配每个像素的主导端元

    参数:
        abundances: 丰度矩阵，形状 (n_endmembers, n_pixels) 或 (n_rows, n_cols, n_endmembers)
        threshold: 丰度阈值，低于此值的端元被忽略
        endmember_names: 端元名称列表，可选

    返回:
        classification: 分类结果，每个像素的主导端元索引
        class_map: 如果输入是3D，则返回分类图
    """
    if abundances.ndim == 3:
        n_rows, n_cols, n_endmembers = abundances.shape
        abun_2d = abundances.reshape(-1, n_endmembers).T
        output_3d = True
    else:
        n_endmembers, n_pixels = abundances.shape
        abun_2d = abundances
        output_3d = False

    abun_filtered = np.where(abun_2d >= threshold, abun_2d, 0)
    classification = np.argmax(abun_filtered, axis=0)
    total_abundance = np.sum(abun_filtered, axis=0)
    classification[total_abundance < threshold] = -1

    if output_3d:
        classification = classification.reshape(n_rows, n_cols)

    return classification


def generate_abundance_maps(abundances: np.ndarray,
                            endmember_names: list = None) -> dict:
    """
    生成每个端元的丰度图

    参数:
        abundances: 丰度矩阵，形状 (n_rows, n_cols, n_endmembers)
        endmember_names: 端元名称列表

    返回:
        abundance_maps: 字典，键为端元名称，值为丰度图 (n_rows, n_cols)
    """
    if abundances.ndim != 3:
        raise ValueError("丰度数据必须是3D数组 (n_rows, n_cols, n_endmembers)")

    n_rows, n_cols, n_endmembers = abundances.shape

    if endmember_names is None:
        endmember_names = [f'端元_{i+1}' for i in range(n_endmembers)]

    abundance_maps = {}
    for i, name in enumerate(endmember_names):
        abundance_maps[name] = abundances[:, :, i]

    return abundance_maps


def full_pipeline(data: np.ndarray,
                  method: str = 'vca',
                  denoise: str = 'pca',
                  denoise_variance_ratio: float = 0.99,
                  p: int = None,
                  hfc_alpha: float = 0.99,
                  fcls_max_iter: int = 1000,
                  classify_threshold: float = 0.1,
                  endmember_names: list = None) -> dict:
    """
    完整高光谱分析流程：降噪 -> 端元提取 -> 丰度反演 -> 分类

    参数:
        data: 高光谱数据，形状 (n_bands, n_pixels) 或 (n_rows, n_cols, n_bands)
        method: 端元提取方法，'vca' 或 'osp'
        denoise: 降噪方法，None/'pca'/'mnf'
        denoise_variance_ratio: 降噪方差保留比例
        p: 端元数，如为None则自动估计
        hfc_alpha: HFC显著性水平
        fcls_max_iter: FCLS最大迭代次数
        classify_threshold: 分类阈值
        endmember_names: 端元名称列表

    返回:
        results: 包含所有结果的字典
            - 'endmembers': 端元光谱矩阵 (n_bands, n_endmembers)
            - 'endmember_indices': 端元像素索引
            - 'info': 端元提取信息
            - 'abundances': 丰度矩阵
            - 'classification': 分类结果
            - 'abundance_maps': 丰度图字典（如果输入是3D）
    """
    print("=" * 60)
    print("高光谱完整分析流程")
    print("=" * 60)

    is_3d = data.ndim == 3
    if is_3d:
        n_rows, n_cols, n_bands = data.shape
        print(f"\n输入数据: {n_rows}x{n_cols}x{n_bands} (三维图像)")
    else:
        n_bands, n_pixels = data.shape if data.shape[0] < data.shape[1] else (data.shape[1], data.shape[0])
        print(f"\n输入数据: {n_bands}x{n_pixels} (二维矩阵)")

    print(f"\n步骤1: 端元提取 ({method.upper()})")
    if denoise:
        print(f"  预处理: {denoise.upper()} 降噪 (保留 {denoise_variance_ratio*100:.0f}% 方差)")
    if p is None:
        print(f"  端元数: 自动估计 (HFC, alpha={hfc_alpha})")
    else:
        print(f"  端元数: {p}")

    endmembers, endmember_indices, info = extract_endmembers(
        data, p=p, method=method,
        denoise=denoise, denoise_variance_ratio=denoise_variance_ratio,
        hfc_alpha=hfc_alpha
    )
    n_endmembers = endmembers.shape[1]
    print(f"  提取到 {n_endmembers} 个端元")

    if endmember_names is None:
        endmember_names = [f'端元_{i+1}' for i in range(n_endmembers)]

    print(f"\n步骤2: 丰度反演 (FCLS)")
    abundances = fcls_unmixing(data, endmembers, max_iter=fcls_max_iter)
    print(f"  丰度矩阵形状: {abundances.shape}")

    print(f"\n步骤3: 亚像元分类 (阈值={classify_threshold})")
    classification = classify_subpixel(abundances, threshold=classify_threshold)
    print(f"  分类完成")

    results = {
        'endmembers': endmembers,
        'endmember_indices': endmember_indices,
        'endmember_names': endmember_names,
        'info': info,
        'abundances': abundances,
        'classification': classification,
    }

    if is_3d:
        print(f"\n步骤4: 生成丰度图")
        abundance_maps = generate_abundance_maps(abundances, endmember_names)
        results['abundance_maps'] = abundance_maps
        print(f"  生成 {len(abundance_maps)} 个丰度图")

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

    return results


def mineral_mapping_example():
    """
    矿物填图应用示例
    """
    print("\n" + "=" * 60)
    print("应用示例1: 矿物填图")
    print("=" * 60)

    n_rows, n_cols, n_bands = 100, 100, 150
    data = np.zeros((n_rows, n_cols, n_bands))

    mineral_spectra = []
    for i in range(4):
        peak = np.exp(-(np.arange(n_bands) - (i + 1) * n_bands / 5) ** 2 / 200)
        mineral_spectra.append(peak)

    regions = [
        (slice(0, 50), slice(0, 50)),
        (slice(0, 50), slice(50, 100)),
        (slice(50, 100), slice(0, 50)),
        (slice(50, 100), slice(50, 100)),
    ]

    for i, region in enumerate(regions):
        for r in range(*region[0].indices(n_rows)):
            for c in range(*region[1].indices(n_cols)):
                mix = 0.7 * mineral_spectra[i]
                other_idx = (i + 1) % 4
                mix += 0.3 * mineral_spectra[other_idx]
                data[r, c, :] = mix + 0.02 * np.random.randn(n_bands)

    data = np.clip(data, 0, None)

    mineral_names = ['石英', '长石', '方解石', '黏土']

    results = full_pipeline(
        data,
        method='vca',
        denoise='pca',
        denoise_variance_ratio=0.98,
        p=4,
        endmember_names=mineral_names
    )

    print("\n分类统计:")
    for i, name in enumerate(mineral_names):
        count = np.sum(results['classification'] == i)
        print(f"  {name}: {count} 像素")

    unclassified = np.sum(results['classification'] == -1)
    print(f"  未分类: {unclassified} 像素")

    return results


def vegetation_stress_detection_example():
    """
    植被胁迫检测应用示例
    """
    print("\n" + "=" * 60)
    print("应用示例2: 植被胁迫检测")
    print("=" * 60)

    n_rows, n_cols, n_bands = 80, 80, 200
    data = np.zeros((n_rows, n_cols, n_bands))

    wavelengths = np.linspace(400, 900, n_bands)

    def healthy_veg(wl):
        red_edge = 1 / (1 + np.exp(-(wl - 700) / 10))
        return 0.1 + 0.5 * red_edge

    def stressed_veg(wl):
        red_edge = 1 / (1 + np.exp(-(wl - 680) / 15))
        return 0.15 + 0.3 * red_edge

    def soil(wl):
        return 0.2 + 0.1 * np.exp(-(wl - 500) ** 2 / 50000)

    healthy = healthy_veg(wavelengths)
    stressed = stressed_veg(wavelengths)
    soil_spec = soil(wavelengths)

    np.random.seed(42)
    for r in range(n_rows):
        for c in range(n_cols):
            dist_from_center = np.sqrt((r - 40) ** 2 + (c - 40) ** 2)

            if dist_from_center < 15:
                mix = 0.8 * stressed + 0.2 * soil_spec
            elif dist_from_center < 30:
                mix = 0.5 * healthy + 0.3 * stressed + 0.2 * soil_spec
            else:
                mix = 0.9 * healthy + 0.1 * soil_spec

            data[r, c, :] = mix + 0.01 * np.random.randn(n_bands)

    data = np.clip(data, 0, None)

    veg_names = ['健康植被', '胁迫植被', '土壤']

    results = full_pipeline(
        data,
        method='vca',
        denoise='pca',
        denoise_variance_ratio=0.99,
        p=3,
        endmember_names=veg_names
    )

    print("\n植被胁迫分析:")
    stress_idx = veg_names.index('胁迫植被')
    stress_map = results['abundance_maps']['胁迫植被']
    print(f"  平均胁迫丰度: {np.mean(stress_map):.3f}")
    print(f"  高胁迫区域(>0.5): {np.sum(stress_map > 0.5)} 像素")

    return results


def water_quality_monitoring_example():
    """
    水质监测应用示例
    """
    print("\n" + "=" * 60)
    print("应用示例3: 水质监测")
    print("=" * 60)

    n_rows, n_cols, n_bands = 60, 60, 180
    data = np.zeros((n_rows, n_cols, n_bands))

    wavelengths = np.linspace(400, 800, n_bands)

    def clear_water(wl):
        return np.exp(-(wl - 500) ** 2 / 10000) * 0.3

    def algae_water(wl):
        peak = np.exp(-(wl - 680) ** 2 / 500) * 0.5
        return clear_water(wl) + peak

    def sediment_water(wl):
        return 0.2 + 0.1 * np.exp(-(wl - 600) ** 2 / 20000)

    clear = clear_water(wavelengths)
    algae = algae_water(wavelengths)
    sediment = sediment_water(wavelengths)

    np.random.seed(123)
    for r in range(n_rows):
        for c in range(n_cols):
            if c < 20:
                mix = 0.9 * clear + 0.1 * sediment
            elif c < 40:
                mix = 0.5 * clear + 0.4 * algae + 0.1 * sediment
            else:
                mix = 0.2 * clear + 0.6 * algae + 0.2 * sediment

            data[r, c, :] = mix + 0.008 * np.random.randn(n_bands)

    data = np.clip(data, 0, None)

    water_names = ['清水', '藻类', '沉积物']

    results = full_pipeline(
        data,
        method='osp',
        denoise='pca',
        denoise_variance_ratio=0.99,
        p=3,
        endmember_names=water_names
    )

    print("\n水质分析:")
    algae_map = results['abundance_maps']['藻类']
    sediment_map = results['abundance_maps']['沉积物']
    print(f"  平均藻类丰度: {np.mean(algae_map):.3f}")
    print(f"  平均沉积物丰度: {np.mean(sediment_map):.3f}")
    print(f"  富营养化区域(藻类>0.5): {np.sum(algae_map > 0.5)} 像素")

    return results


def generate_synthetic_data(n_pixels: int = 1000, n_bands: int = 200,
                            n_endmembers: int = 5,
                            noise_level: float = 0.01,
                            seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成模拟高光谱数据用于测试

    参数:
        n_pixels: 像素数
        n_bands: 波段数
        n_endmembers: 端元数
        noise_level: 噪声水平，默认0.01（低噪声），高噪声可设为0.1
        seed: 随机种子

    返回:
        data: 模拟数据 (n_bands, n_pixels)
        E_true: 真实端元 (n_bands, n_endmembers)
        abundances: 丰度矩阵 (n_endmembers, n_pixels)
    """
    np.random.seed(seed)

    E_true = np.zeros((n_bands, n_endmembers))
    for i in range(n_endmembers):
        peak_pos = np.linspace(20, n_bands - 20, n_endmembers)[i]
        x = np.arange(n_bands)
        E_true[:, i] = np.exp(-(x - peak_pos) ** 2 / (2 * (n_bands / 10) ** 2))
        E_true[:, i] += 0.3 * np.random.randn(n_bands) * 0.1
        E_true[:, i] = np.clip(E_true[:, i], 0, None)
        E_true[:, i] = E_true[:, i] / np.max(E_true[:, i])

    abundances = np.random.dirichlet(alpha=np.ones(n_endmembers), size=n_pixels).T
    data = E_true @ abundances
    noise = noise_level * np.random.randn(n_bands, n_pixels)
    data = data + noise
    data = np.clip(data, 0, None)

    return data, E_true, abundances


if __name__ == "__main__":
    print("=" * 60)
    print("高光谱端元提取测试（增强版）")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("测试1: 基础功能测试（低噪声数据）")
    print("=" * 60)

    n_bands = 200
    n_pixels = 1000
    n_endmembers = 5

    print(f"\n生成模拟高光谱数据 (噪声水平: 0.01)...")
    data, E_true, abundances = generate_synthetic_data(n_pixels, n_bands, n_endmembers, noise_level=0.01)
    print(f"   数据形状: {data.shape} (波段数 x 像素数)")
    print(f"   真实端元数目: {n_endmembers}")

    print(f"\n使用 VCA 算法提取端元...")
    E_vca, idx_vca, info_vca = extract_endmembers(data, n_endmembers, method='vca')
    print(f"   提取的端元矩阵形状: {E_vca.shape}")
    print(f"   端元索引: {idx_vca}")

    print(f"\n使用 OSP 算法提取端元...")
    E_osp, idx_osp, info_osp = extract_endmembers(data, n_endmembers, method='osp')
    print(f"   提取的端元矩阵形状: {E_osp.shape}")
    print(f"   端元索引: {idx_osp}")

    print("\n" + "=" * 60)
    print("测试2: 高噪声数据 + 降噪 + 自动估计端元数")
    print("=" * 60)

    print(f"\n生成高噪声模拟数据 (噪声水平: 0.1)...")
    data_noisy, E_true_noisy, _ = generate_synthetic_data(n_pixels, n_bands, n_endmembers, noise_level=0.1, seed=123)
    print(f"   数据形状: {data_noisy.shape}")
    print(f"   真实端元数目: {n_endmembers}")

    print(f"\n测试2a: 原始VCA（无降噪，手动指定端元数）...")
    E_raw, idx_raw, info_raw = extract_endmembers(data_noisy, p=n_endmembers, method='vca')
    print(f"   提取端元数: {info_raw['p_estimated']}")
    print(f"   端元索引: {idx_raw}")

    print(f"\n测试2b: PCA降噪 + VCA + 自动估计端元数...")
    E_pca, idx_pca, info_pca = extract_endmembers(
        data_noisy, p=None, method='vca', denoise='pca',
        denoise_variance_ratio=0.99, hfc_alpha=0.99
    )
    print(f"   PCA保留成分数: {info_pca['denoise_components']}")
    print(f"   HFC估计端元数: {info_pca['p_hfc']}")
    print(f"   实际提取端元数: {info_pca['p_estimated']}")
    print(f"   端元索引: {idx_pca}")

    print(f"\n测试2c: MNF降噪 + VCA + 自动估计端元数...")
    E_mnf, idx_mnf, info_mnf = extract_endmembers(
        data_noisy, p=None, method='vca', denoise='mnf',
        denoise_variance_ratio=0.99, hfc_alpha=0.99
    )
    print(f"   MNF保留成分数: {info_mnf['denoise_components']}")
    print(f"   HFC估计端元数: {info_mnf['p_hfc']}")
    print(f"   实际提取端元数: {info_mnf['p_estimated']}")
    print(f"   端元索引: {idx_mnf}")

    print(f"\n测试2d: MNF降噪 + OSP + 自动估计端元数...")
    E_osp_mnf, idx_osp_mnf, info_osp_mnf = extract_endmembers(
        data_noisy, p=None, method='osp', denoise='mnf',
        denoise_variance_ratio=0.99, hfc_alpha=0.99
    )
    print(f"   MNF保留成分数: {info_osp_mnf['denoise_components']}")
    print(f"   HFC估计端元数: {info_osp_mnf['p_hfc']}")
    print(f"   实际提取端元数: {info_osp_mnf['p_estimated']}")
    print(f"   端元索引: {idx_osp_mnf}")

    print("\n" + "=" * 60)
    print("结果对比（提取的第一个端元与真实端元的相关系数）:")
    print("=" * 60)

    def spectral_similarity(spec1, spec2):
        return np.corrcoef(spec1, spec2)[0, 1]

    for i in range(min(n_endmembers, E_raw.shape[1])):
        sim_raw = spectral_similarity(E_true_noisy[:, i], E_raw[:, i])
        sim_pca = spectral_similarity(E_true_noisy[:, i], E_pca[:, i]) if i < E_pca.shape[1] else None
        sim_mnf = spectral_similarity(E_true_noisy[:, i], E_mnf[:, i]) if i < E_mnf.shape[1] else None

        print(f"\n端元 {i+1}:")
        print(f"  无降噪VCA:   {sim_raw:.4f}")
        if sim_pca is not None:
            print(f"  PCA降噪VCA:  {sim_pca:.4f}")
        if sim_mnf is not None:
            print(f"  MNF降噪VCA:  {sim_mnf:.4f}")

    print("\n" + "=" * 60)
    print("增强功能使用说明:")
    print("=" * 60)
    print("""
  # 1. 自动估计端元数量（HFC方法）
  E, indices, info = extract_endmembers(data, p=None, method='vca')
  print(f"估计的端元数: {info['p_hfc']}")

  # 2. 先降噪再提取端元（PCA或MNF）
  E, indices, info = extract_endmembers(data, p=5, method='vca', denoise='pca')
  E, indices, info = extract_endmembers(data, p=5, method='vca', denoise='mnf')

  # 3. 丰度反演（FCLS全约束最小二乘）
  from hyperspectral_endmembers import fcls_unmixing
  abundances = fcls_unmixing(data, E)  # 形状: (n_rows, n_cols, n_endmembers)

  # 4. 亚像元分类
  from hyperspectral_endmembers import classify_subpixel, generate_abundance_maps
  classification = classify_subpixel(abundances, threshold=0.1)
  abundance_maps = generate_abundance_maps(abundances, ['石英', '长石', '方解石'])

  # 5. 完整分析流程：降噪 + 端元提取 + 丰度反演 + 分类
  from hyperspectral_endmembers import full_pipeline
  results = full_pipeline(
      data, method='vca', denoise='pca', p=5,
      endmember_names=['石英', '长石', '方解石', '黏土']
  )
  # results包含: endmembers, abundances, classification, abundance_maps等
""")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("运行应用示例:")
    print("=" * 60)
    try:
        mineral_mapping_example()
        vegetation_stress_detection_example()
        water_quality_monitoring_example()
    except Exception as e:
        print(f"应用示例运行提示: {e}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
