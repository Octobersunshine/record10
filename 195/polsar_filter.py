import numpy as np
from scipy import ndimage


def create_symmetric_covariance_matrix(shh, svv, shv_real, shv_imag):
    """
    创建2x2极化协方差矩阵C2
    
    C2 = [[<|Shh|^2>,  <Shh*Svh*>],
          [<Svh*Shh*>, <|Svh|^2>]]
    
    对于互易介质: Shv = Svh
    
    参数:
        shh: HH极化通道幅度 (M, N)
        svv: VV极化通道幅度 (M, N)
        shv_real: HV极化通道实部 (M, N)
        shv_imag: HV极化通道虚部 (M, N)
    
    返回:
        C2: 2x2协方差矩阵 (2, 2, M, N)
    """
    M, N = shh.shape
    
    shh2 = np.abs(shh) ** 2
    svv2 = np.abs(svv) ** 2
    
    shh_svh = shh * (shv_real - 1j * shv_imag)
    svh_shh = np.conj(shh_svh)
    
    C2 = np.zeros((2, 2, M, N), dtype=np.complex128)
    C2[0, 0, :, :] = shh2
    C2[0, 1, :, :] = shh_svh
    C2[1, 0, :, :] = svh_shh
    C2[1, 1, :, :] = svv2
    
    return C2


def create_covariance_matrix_3x3(shh, svv, shv_real, shv_imag):
    """
    创建3x3极化协方差矩阵C3（Lexel基）
    
    参数:
        shh: HH极化通道幅度
        svv: VV极化通道幅度
        shv_real: HV极化通道实部
        shv_imag: HV极化通道虚部
    
    返回:
        C3: 3x3协方差矩阵 (3, 3, M, N)
    """
    M, N = shh.shape
    
    C3 = np.zeros((3, 3, M, N), dtype=np.complex128)
    
    C3[0, 0, :, :] = np.abs(shh) ** 2
    C3[1, 1, :, :] = np.abs(svv) ** 2
    C3[2, 2, :, :] = shv_real ** 2 + shv_imag ** 2
    
    C3[0, 1, :, :] = shh * np.conj(svv)
    C3[1, 0, :, :] = np.conj(C3[0, 1, :, :])
    
    C3[0, 2, :, :] = shh * (shv_real - 1j * shv_imag)
    C3[2, 0, :, :] = np.conj(C3[0, 2, :, :])
    
    C3[1, 2, :, :] = svv * (shv_real - 1j * shv_imag)
    C3[2, 1, :, :] = np.conj(C3[1, 2, :, :])
    
    return C3


def create_coherency_matrix_3x3(shh, svv, shv_real, shv_imag):
    """
    创建3x3极化相干矩阵T3（Pauli基）
    
    k = [Shh+Svv, Shh-Svv, 2*Shv]^T / sqrt(2)
    
    参数:
        shh: HH极化通道幅度
        svv: VV极化通道幅度
        shv_real: HV极化通道实部
        shv_imag: HV极化通道虚部
    
    返回:
        T3: 3x3相干矩阵 (3, 3, M, N)
    """
    M, N = shh.shape
    
    k1 = (shh + svv) / np.sqrt(2)
    k2 = (shh - svv) / np.sqrt(2)
    k3 = (2 * (shv_real + 1j * shv_imag)) / np.sqrt(2)
    
    k = np.array([k1, k2, k3])
    
    T3 = np.zeros((3, 3, M, N), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            T3[i, j, :, :] = k[i] * np.conj(k[j])
    
    return T3


def boxcar_filter_covariance(C, window_size=5):
    """
    Boxcar滤波 - 极化协方差矩阵的简单滑动平均
    
    参数:
        C: 极化协方差矩阵 (p, p, M, N)
        window_size: 窗口大小
    
    返回:
        C_filtered: 滤波后的协方差矩阵 (p, p, M, N)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    p = C.shape[0]
    M, N = C.shape[2], C.shape[3]
    
    C_filtered = np.zeros_like(C)
    
    for i in range(p):
        for j in range(p):
            C_filtered[i, j, :, :] = ndimage.uniform_filter(
                np.real(C[i, j, :, :]), size=window_size
            ) + 1j * ndimage.uniform_filter(
                np.imag(C[i, j, :, :]), size=window_size
            )
    
    return C_filtered


def refined_lee_filter_polarimetric(C, window_size=7, num_looks=1):
    """
    极化Refined Lee滤波
    
    基于极化特征值的自适应滤波，保留极化信息的同时抑制斑点
    
    参数:
        C: 极化协方差矩阵 (p, p, M, N)
        window_size: 窗口大小
        num_looks: 视数
    
    返回:
        C_filtered: 滤波后的协方差矩阵 (p, p, M, N)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    p = C.shape[0]
    M, N = C.shape[2], C.shape[3]
    
    pad = window_size // 2
    C_padded = np.pad(C, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    
    C_filtered = np.zeros_like(C)
    
    cu = np.sqrt(2.0 / num_looks)
    cmax = np.sqrt(2.0) * cu
    
    for i in range(M):
        for j in range(N):
            i_pad, j_pad = i + pad, j + pad
            
            C_center = C[:, :, i, j]
            
            C_window = C_padded[:, :, i_pad-pad:i_pad+pad+1, j_pad-pad:j_pad+pad+1]
            
            span = np.real(np.trace(C_center))
            
            if span < 1e-10:
                C_filtered[:, :, i, j] = C_center
                continue
            
            C_mean = np.mean(C_window.reshape(p, p, -1), axis=2)
            
            try:
                eigenvalues = np.linalg.eigvalsh(C_mean)
                eigenvalues = np.sort(np.real(eigenvalues))[::-1]
                
                if eigenvalues[0] > 1e-10:
                    entropy = 0
                    total = np.sum(eigenvalues[eigenvalues > 0])
                    for eig in eigenvalues:
                        if eig > 1e-10:
                            p_eig = eig / total
                            if p_eig > 0:
                                entropy -= p_eig * np.log(p_eig)
                    entropy /= np.log(p)
                else:
                    entropy = 0
                
                ci = np.std(eigenvalues) / (np.mean(eigenvalues) + 1e-10)
                
                if ci <= cu or entropy < 0.1:
                    C_filtered[:, :, i, j] = C_mean
                elif ci < cmax:
                    weight = (ci - cu) / (cmax - cu)
                    C_filtered[:, :, i, j] = C_mean + weight * (C_center - C_mean)
                else:
                    C_filtered[:, :, i, j] = C_center
                    
            except np.linalg.LinAlgError:
                C_filtered[:, :, i, j] = C_mean
    
    return C_filtered


def polarimetric_whitening_filter(C, window_size=7, num_looks=1):
    """
    极化白化滤波(PWF) - PolSAR斑点抑制的经典算法
    
    核心思想：利用极化协方差矩阵的白化变换，
    在极化域和空间域同时进行斑点抑制。
    
    算法步骤：
    1. 估计局部极化协方差矩阵
    2. 计算白化变换矩阵W = U * Lambda^(-1/2) * U^H
    3. 对白化后的向量进行空间滤波
    4. 通过逆白化变换恢复极化信息
    
    参考: Lee et al., "Speckle filtering of polarimetric SAR images", 1999
    
    参数:
        C: 极化协方差矩阵 (p, p, M, N)
        window_size: 滤波窗口大小
        num_looks: SAR图像视数
    
    返回:
        C_filtered: 滤波后的协方差矩阵 (p, p, M, N)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    p = C.shape[0]
    M, N = C.shape[2], C.shape[3]
    
    pad = window_size // 2
    C_padded = np.pad(C, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    
    C_filtered = np.zeros_like(C)
    
    for i in range(M):
        for j in range(N):
            i_pad, j_pad = i + pad, j + pad
            
            C_center = C[:, :, i, j]
            span_center = np.real(np.trace(C_center))
            
            if span_center < 1e-10:
                C_filtered[:, :, i, j] = C_center
                continue
            
            C_window = C_padded[:, :, i_pad-pad:i_pad+pad+1, j_pad-pad:j_pad+pad+1]
            C_mean = np.mean(C_window.reshape(p, p, -1), axis=2)
            
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(C_mean)
                eigenvalues = np.real(eigenvalues)
                eigenvalues = np.clip(eigenvalues, 1e-10, None)
                
                sqrt_eigenvalues = np.sqrt(eigenvalues)
                
                Lambda_sqrt_inv = np.diag(1.0 / sqrt_eigenvalues)
                Lambda_sqrt = np.diag(sqrt_eigenvalues)
                
                W = eigenvectors @ Lambda_sqrt_inv @ eigenvectors.conj().T
                W_inv = eigenvectors @ Lambda_sqrt @ eigenvectors.conj().T
                
                C_whitened = W @ C_center @ W.conj().T
                
                k_whitened = np.zeros(p, dtype=np.complex128)
                for k in range(p):
                    k_whitened[k] = np.sqrt(C_whitened[k, k]) if C_whitened[k, k] > 0 else 0
                
                k_filtered = k_whitened.copy()
                
                for di in range(-pad, pad + 1):
                    for dj in range(-pad, pad + 1):
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i_pad + di, j_pad + dj
                        if (ni < pad or ni >= C_padded.shape[2] - pad or
                            nj < pad or nj >= C_padded.shape[3] - pad):
                            continue
                        
                        C_neighbor = C_padded[:, :, ni, nj]
                        span_neighbor = np.real(np.trace(C_neighbor))
                        
                        if span_neighbor < 1e-10:
                            continue
                        
                        C_neighbor_whitened = W @ C_neighbor @ W.conj().T
                        k_neighbor = np.zeros(p, dtype=np.complex128)
                        for k in range(p):
                            k_neighbor[k] = np.sqrt(C_neighbor_whitened[k, k]) if C_neighbor_whitened[k, k] > 0 else 0
                        
                        dist = np.sqrt(di**2 + dj**2)
                        weight = np.exp(-dist / (window_size / 2))
                        
                        k_filtered += weight * k_neighbor
                
                weight_sum = 1.0 + np.sum(np.exp(-np.arange(1, pad + 1) / (window_size / 2)))
                k_filtered /= weight_sum
                
                C_filtered_center = np.zeros((p, p), dtype=np.complex128)
                for k in range(p):
                    for l in range(p):
                        C_filtered_center[k, l] = k_filtered[k] * np.conj(k_filtered[l])
                
                C_filtered_center = W_inv @ C_filtered_center @ W_inv.conj().T
                
                span_filtered = np.real(np.trace(C_filtered_center))
                if span_filtered > 1e-10:
                    scale = span_center / span_filtered
                    C_filtered_center *= scale
                
                C_filtered[:, :, i, j] = C_filtered_center
                
            except np.linalg.LinAlgError:
                C_filtered[:, :, i, j] = C_mean
    
    return C_filtered


def enhanced_polarimetric_whitening_filter(C, window_size=7, num_looks=1, edge_threshold=0.3):
    """
    增强型极化白化滤波(EPWF)
    
    在PWF基础上增加边缘检测和自适应窗口机制
    
    参数:
        C: 极化协方差矩阵 (p, p, M, N)
        window_size: 滤波窗口大小
        num_looks: SAR图像视数
        edge_threshold: 边缘检测阈值
    
    返回:
        C_filtered: 滤波后的协方差矩阵 (p, p, M, N)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    p = C.shape[0]
    M, N = C.shape[2], C.shape[3]
    
    pad = window_size // 2
    C_padded = np.pad(C, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    
    C_filtered = np.zeros_like(C)
    
    cu = np.sqrt(2.0 / num_looks)
    cmax = np.sqrt(2.0) * cu
    
    for i in range(M):
        for j in range(N):
            i_pad, j_pad = i + pad, j + pad
            
            C_center = C[:, :, i, j]
            span_center = np.real(np.trace(C_center))
            
            if span_center < 1e-10:
                C_filtered[:, :, i, j] = C_center
                continue
            
            C_window = C_padded[:, :, i_pad-pad:i_pad+pad+1, j_pad-pad:j_pad+pad+1]
            C_mean = np.mean(C_window.reshape(p, p, -1), axis=2)
            
            span_mean = np.real(np.trace(C_mean))
            if span_mean > 1e-10:
                ci = abs(span_center - span_mean) / (span_mean + 1e-10)
            else:
                ci = 0
            
            try:
                if ci <= cu:
                    C_filtered[:, :, i, j] = C_mean
                elif ci < cmax:
                    eigenvalues, eigenvectors = np.linalg.eigh(C_mean)
                    eigenvalues = np.real(eigenvalues)
                    eigenvalues = np.clip(eigenvalues, 1e-10, None)
                    
                    sqrt_eigenvalues = np.sqrt(eigenvalues)
                    Lambda_sqrt_inv = np.diag(1.0 / sqrt_eigenvalues)
                    Lambda_sqrt = np.diag(sqrt_eigenvalues)
                    
                    W = eigenvectors @ Lambda_sqrt_inv @ eigenvectors.conj().T
                    W_inv = eigenvectors @ Lambda_sqrt @ eigenvectors.conj().T
                    
                    C_whitened = W @ C_center @ W.conj().T
                    C_filtered_center = W_inv @ C_whitened @ W_inv.conj().T
                    
                    weight = (ci - cu) / (cmax - cu)
                    C_filtered[:, :, i, j] = C_mean + weight * (C_filtered_center - C_mean)
                else:
                    C_filtered[:, :, i, j] = C_center
                    
            except np.linalg.LinAlgError:
                C_filtered[:, :, i, j] = C_mean
    
    return C_filtered


def idan_filter(C, window_size=7, num_looks=1):
    """
    IDAN滤波 - 基于固有极化特性的滤波
    
    利用极化协方差矩阵的特征向量进行滤波，
    保留极化散射特性的同时抑制斑点噪声
    
    参数:
        C: 极化协方差矩阵 (p, p, M, N)
        window_size: 窗口大小
        num_looks: 视数
    
    返回:
        C_filtered: 滤波后的协方差矩阵 (p, p, M, N)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    p = C.shape[0]
    M, N = C.shape[2], C.shape[3]
    
    pad = window_size // 2
    C_padded = np.pad(C, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    
    C_filtered = np.zeros_like(C)
    
    for i in range(M):
        for j in range(N):
            i_pad, j_pad = i + pad, j + pad
            
            C_center = C[:, :, i, j]
            span_center = np.real(np.trace(C_center))
            
            if span_center < 1e-10:
                C_filtered[:, :, i, j] = C_center
                continue
            
            C_window = C_padded[:, :, i_pad-pad:i_pad+pad+1, j_pad-pad:j_pad+pad+1]
            C_mean = np.mean(C_window.reshape(p, p, -1), axis=2)
            
            try:
                eigvals_center, eigvecs_center = np.linalg.eigh(C_center)
                eigvals_center = np.real(eigvals_center)
                
                eigvals_mean, eigvecs_mean = np.linalg.eigh(C_mean)
                eigvals_mean = np.real(eigvals_mean)
                
                filtered_eigenvalues = np.zeros(p)
                for k in range(p):
                    if eigvals_mean[k] > 1e-10:
                        ratio = eigvals_center[k] / eigvals_mean[k]
                        filtered_eigenvalues[k] = eigvals_mean[k] * min(1.0, max(0.5, ratio))
                    else:
                        filtered_eigenvalues[k] = eigvals_center[k]
                
                Lambda_filtered = np.diag(filtered_eigenvalues)
                C_filtered_center = eigvecs_center @ Lambda_filtered @ eigvecs_center.conj().T
                
                span_filtered = np.real(np.trace(C_filtered_center))
                if span_filtered > 1e-10:
                    scale = span_center / span_filtered
                    C_filtered_center *= scale
                
                C_filtered[:, :, i, j] = C_filtered_center
                
            except np.linalg.LinAlgError:
                C_filtered[:, :, i, j] = C_mean
    
    return C_filtered


def cloude_pottier_decomposition(T):
    """
    Cloude-Pottier H/A/Alpha极化分解
    
    对极化相干矩阵T3进行特征值分解，
    计算散射熵H、反熵A、平均散射角Alpha
    
    参数:
        T: 3x3极化相干矩阵 (3, 3, M, N)
    
    返回:
        H: 散射熵 (M, N) - 0~1，衡量散射随机程度
        A: 反熵 (M, N) - 0~1，衡量各向异性
        Alpha: 平均散射角 (M, N) - 0~90度
        eigenvalues: 特征值 (3, M, N) - 按降序排列
    """
    M, N = T.shape[2], T.shape[3]
    
    H = np.zeros((M, N))
    A = np.zeros((M, N))
    Alpha = np.zeros((M, N))
    eigenvalues = np.zeros((3, M, N))
    
    for i in range(M):
        for j in range(N):
            T3 = T[:, :, i, j]
            
            try:
                eigvals, eigvecs = np.linalg.eigh(T3)
                
                idx = np.argsort(np.real(eigvals))[::-1]
                eigvals = np.real(eigvals[idx])
                eigvecs = eigvecs[:, idx]
                
                eigenvalues[:, i, j] = eigvals
                
                total = np.sum(eigvals)
                if total > 1e-10:
                    p = eigvals / total
                    
                    h = 0
                    for pk in p:
                        if pk > 1e-10:
                            h -= pk * np.log(pk)
                    H[i, j] = h / np.log(3)
                    
                    if p[0] + p[1] > 1e-10:
                        A[i, j] = (p[0] - p[1]) / (p[0] + p[1])
                    
                    alpha = 0
                    for k in range(3):
                        vec = eigvecs[:, k]
                        if abs(vec[0]) > 1e-10:
                            alpha_k = np.degrees(np.arctan2(np.abs(vec[1]), np.abs(vec[0])))
                        else:
                            alpha_k = 90.0
                        alpha += p[k] * alpha_k
                    Alpha[i, j] = alpha
                    
            except np.linalg.LinAlgError:
                H[i, j] = 0
                A[i, j] = 0
                Alpha[i, j] = 0
    
    return H, A, Alpha, eigenvalues


def yamaguchi_decomposition(C):
    """
    Yamaguchi四分量分解
    
    将极化协方差矩阵分解为表面散射、二次散射、
    体散射和螺旋散射四个分量
    
    参数:
        C: 3x3极化协方差矩阵 (3, 3, M, N)
    
    返回:
        Ps: 表面散射分量 (M, N)
        Pd: 二次散射分量 (M, N)
        Pv: 体散射分量 (M, N)
        Pc: 螺旋散射分量 (M, N)
    """
    M, N = C.shape[2], C.shape[3]
    
    Ps = np.zeros((M, N))
    Pd = np.zeros((M, N))
    Pv = np.zeros((M, N))
    Pc = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            C3 = C[:, :, i, j]
            
            try:
                span = np.real(np.trace(C3))
                if span < 1e-10:
                    continue
                
                Pc[i, j] = np.abs(np.imag(C3[0, 2])) / span
                
                C_modified = C3.copy()
                C_modified[0, 2] = np.real(C3[0, 2])
                C_modified[2, 0] = np.real(C3[2, 0])
                
                Pv[i, j] = 15.0 * np.real(C3[2, 2]) / (4.0 * span)
                
                residual = span - Pv[i, j] * span - Pc[i, j] * span
                
                C11 = np.real(C3[0, 0]) - Pv[i, j] * span / 3.0
                C22 = np.real(C3[1, 1]) - Pv[i, j] * span / 3.0
                
                if C11 > C22 and C11 > 0:
                    Pd[i, j] = C22 / span
                    Ps[i, j] = (C11 - C22) / span
                else:
                    Ps[i, j] = C11 / span
                    Pd[i, j] = (C22 - C11) / span
                
                total = Ps[i, j] + Pd[i, j] + Pv[i, j] + Pc[i, j]
                if total > 0:
                    Ps[i, j] /= total
                    Pd[i, j] /= total
                    Pv[i, j] /= total
                    Pc[i, j] /= total
                    
            except:
                continue
    
    return Ps, Pd, Pv, Pc


def calculate_polarization_fidelity(C_original, C_filtered):
    """
    计算极化保真度
    
    评估滤波后极化信息的保留程度
    
    参数:
        C_original: 原始极化协方差矩阵 (p, p, M, N)
        C_filtered: 滤波后极化协方差矩阵 (p, p, M, N)
    
    返回:
        fidelity: 极化保真度 (M, N) - 0~1，越接近1越好
    """
    p = C_original.shape[0]
    M, N = C_original.shape[2], C_original.shape[3]
    
    fidelity = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            C1 = C_original[:, :, i, j]
            C2 = C_filtered[:, :, i, j]
            
            try:
                trace1 = np.real(np.trace(C1))
                trace2 = np.real(np.trace(C2))
                
                if trace1 > 1e-10 and trace2 > 1e-10:
                    C1_norm = C1 / trace1
                    C2_norm = C2 / trace2
                    
                    diff = np.linalg.norm(C1_norm - C2_norm, 'fro')
                    max_diff = np.linalg.norm(C1_norm, 'fro') + np.linalg.norm(C2_norm, 'fro')
                    
                    if max_diff > 1e-10:
                        fidelity[i, j] = 1.0 - diff / max_diff
                    else:
                        fidelity[i, j] = 1.0
                else:
                    fidelity[i, j] = 1.0
                    
            except:
                fidelity[i, j] = 0.5
    
    return fidelity


def compute_polarimetric_features(C):
    """
    计算极化特征参数
    
    参数:
        C: 极化协方差矩阵 (p, p, M, N)
    
    返回:
        features: 极化特征字典
    """
    p = C.shape[0]
    M, N = C.shape[2], C.shape[3]
    
    span = np.zeros((M, N))
    pol_ratio = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            C3 = C[:, :, i, j]
            span[i, j] = np.real(np.trace(C3))
            
            if p >= 2:
                shh = np.real(C3[0, 0])
                svv = np.real(C3[1, 1])
                if svv > 1e-10:
                    pol_ratio[i, j] = shh / svv
    
    features = {
        'span': span,
        'pol_ratio': pol_ratio
    }
    
    if p >= 3:
        cross_pol = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                cross_pol[i, j] = np.real(C[2, 2, i, j])
        features['cross_pol'] = cross_pol
    
    return features


def generate_synthetic_polsar_image(size=128, num_regions=4):
    """
    生成合成的PolSAR图像
    
    创建具有不同极化特性的区域，用于测试极化滤波算法
    
    参数:
        size: 图像大小
        num_regions: 区域数量
    
    返回:
        shh, svv, shv_real, shv_imag: 四个极化通道
    """
    shh = np.zeros((size, size), dtype=np.complex128)
    svv = np.zeros((size, size), dtype=np.complex128)
    shv_real = np.zeros((size, size))
    shv_imag = np.zeros((size, size))
    
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    regions = [
        {'center': (0.25, 0.25), 'radius': 0.15, 'shh': 150, 'svv': 100, 'shv_ratio': 0.3},
        {'center': (0.75, 0.25), 'radius': 0.15, 'shh': 80, 'svv': 180, 'shv_ratio': 0.2},
        {'center': (0.25, 0.75), 'radius': 0.15, 'shh': 200, 'svv': 200, 'shv_ratio': 0.4},
        {'center': (0.75, 0.75), 'radius': 0.15, 'shh': 50, 'svv': 50, 'shv_ratio': 0.5},
    ]
    
    shh_base = np.ones((size, size)) * 50
    svv_base = np.ones((size, size)) * 50
    shv_base = np.ones((size, size)) * 20
    
    for region in regions:
        cx, cy = region['center']
        r = region['radius']
        mask = ((x - cx)**2 + (y - cy)**2) < r**2
        
        shh_base[mask] = region['shh']
        svv_base[mask] = region['svv']
        shv_base[mask] = region['shh'] * region['shv_ratio']
    
    speckle_h = np.random.rayleigh(1, (size, size))
    speckle_v = np.random.rayleigh(1, (size, size))
    speckle_hv = np.random.rayleigh(1, (size, size))
    
    phase_h = np.random.uniform(0, 2*np.pi, (size, size))
    phase_v = np.random.uniform(0, 2*np.pi, (size, size))
    phase_hv = np.random.uniform(0, 2*np.pi, (size, size))
    
    shh = shh_base * speckle_h * np.exp(1j * phase_h)
    svv = svv_base * speckle_v * np.exp(1j * phase_v)
    
    shv_mag = shv_base * speckle_hv
    shv_real = shv_mag * np.cos(phase_hv)
    shv_imag = shv_mag * np.sin(phase_hv)
    
    return shh, svv, shv_real, shv_imag


def covariance_to_intensity(C):
    """
    从协方差矩阵提取极化强度图
    
    参数:
        C: 极化协方差矩阵 (p, p, M, N)
    
    返回:
        intensities: 各极化通道强度图
    """
    p = C.shape[0]
    
    if p >= 2:
        hh_intensity = np.real(C[0, 0, :, :])
        vv_intensity = np.real(C[1, 1, :, :])
        
        if p >= 3:
            hv_intensity = np.real(C[2, 2, :, :])
            return hh_intensity, vv_intensity, hv_intensity
        else:
            return hh_intensity, vv_intensity
    
    return None


def generate_paulirgb_image(T, stretch_type='sqrt'):
    """
    生成Pauli基伪彩色图像
    
    参数:
        T: 3x3极化相干矩阵 (3, 3, M, N)
        stretch_type: 拉伸类型
    
    返回:
        rgb: RGB伪彩色图像 (M, N, 3)
    """
    M, N = T.shape[2], T.shape[3]
    
    R = np.real(T[0, 0, :, :])
    G = np.real(T[1, 1, :, :])
    B = np.real(T[2, 2, :, :])
    
    if stretch_type == 'sqrt':
        R = np.sqrt(np.clip(R, 0, None))
        G = np.sqrt(np.clip(G, 0, None))
        B = np.sqrt(np.clip(B, 0, None))
    
    for channel in [R, G, B]:
        vmin, vmax = np.percentile(channel, 2), np.percentile(channel, 98)
        if vmax > vmin:
            channel[:] = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
    
    rgb = np.zeros((M, N, 3))
    rgb[:, :, 0] = R
    rgb[:, :, 1] = G
    rgb[:, :, 2] = B
    
    return rgb


if __name__ == '__main__':
    print("="*60)
    print("极化SAR (PolSAR) 相干斑滤波模块")
    print("="*60)
    print("\n包含的滤波算法:")
    print("  1. Boxcar滤波 - 简单滑动平均")
    print("  2. 极化Refined Lee滤波")
    print("  3. 极化白化滤波 (PWF)")
    print("  4. 增强型PWF (EPWF)")
    print("  5. IDAN滤波")
    print("\n极化分解方法:")
    print("  1. Cloude-Pottier H/A/Alpha分解")
    print("  2. Yamaguchi四分量分解")
    print("\n评估指标:")
    print("  - 极化保真度 (Polarization Fidelity)")
    print("="*60)
