import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from typing import Tuple
from utils import normalize_data


def baseline_polyfit(y: np.ndarray, degree: int = 3,
                     num_points: int = None) -> np.ndarray:
    """
    多项式拟合基线校正
    
    Args:
        y: 强度数组
        degree: 多项式阶数，建议2-5
        num_points: 用于拟合的点数，默认为全部点数
    
    Returns:
        估计的基线
    """
    x = np.arange(len(y))
    
    if num_points is None or num_points >= len(y):
        x_fit = x
        y_fit = y
    else:
        step = len(y) // num_points
        x_fit = x[::step]
        y_fit = y[::step]
    
    coeffs = np.polyfit(x_fit, y_fit, degree)
    baseline = np.polyval(coeffs, x)
    
    return baseline


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 niter: int = 10) -> np.ndarray:
    """
    非对称最小二乘法（Asymmetric Least Squares）基线校正
    适用于处理荧光背景等具有倾斜基线的光谱
    
    Args:
        y: 强度数组
        lam: 平滑参数，值越大越平滑。对于拉曼光谱，建议1e4-1e6
        p: 非对称参数，0 < p < 1。对于正峰，p应小于0.5
        niter: 迭代次数
    
    Returns:
        估计的基线
    """
    L = len(y)
    D = np.zeros((L, L))
    for i in range(L - 2):
        D[i, i:i+3] = [1, -2, 1]
    D = D[:-2, :]
    D = D.T @ D
    w = np.ones(L)
    
    for _ in range(niter):
        W = np.diag(w)
        Z = W + lam * D
        z = np.linalg.solve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    
    return z


def baseline_airpls(y: np.ndarray, lam: float = 1e5, niter: int = 10) -> np.ndarray:
    """
    自适应迭代加权惩罚最小二乘法（AirPLS）基线校正
    自适应调整权重，更适合处理复杂的荧光背景
    
    Args:
        y: 强度数组
        lam: 平滑参数，建议1e4-1e6
        niter: 迭代次数
    
    Returns:
        估计的基线
    """
    L = len(y)
    D = np.zeros((L, L))
    for i in range(L - 2):
        D[i, i:i+3] = [1, -2, 1]
    D = D[:-2, :]
    D = D.T @ D
    w = np.ones(L)
    
    for i in range(niter):
        W = np.diag(w)
        Z = W + lam * D
        z = np.linalg.solve(Z, w * y)
        d = y - z
        dssn = np.abs(d[d < 0].sum())
        if dssn < 0.001 * np.abs(y).sum() or i == niter - 1:
            break
        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
    
    return z


def baseline_iarpls(y: np.ndarray, lam: float = 1e5, niter: int = 20) -> np.ndarray:
    """
    改进的自适应迭代加权惩罚最小二乘法（IarPLS）基线校正
    优化了权重更新策略，对强荧光背景有更好的效果
    
    Args:
        y: 强度数组
        lam: 平滑参数
        niter: 迭代次数
    
    Returns:
        估计的基线
    """
    L = len(y)
    D = np.zeros((L, L))
    for i in range(L - 2):
        D[i, i:i+3] = [1, -2, 1]
    D = D[:-2, :]
    D = D.T @ D
    w = np.ones(L)
    z_prev = np.zeros(L)
    
    for i in range(niter):
        W = np.diag(w)
        Z = W + lam * D
        z = np.linalg.solve(Z, w * y)
        d = y - z
        
        d_neg = d[d < 0]
        if len(d_neg) == 0:
            break
        
        mean_neg = np.mean(d_neg)
        std_neg = np.std(d_neg)
        
        if std_neg < 1e-10:
            break
        
        t = -2 * (d - mean_neg) / std_neg
        w = 1.0 / (1.0 + np.exp(t))
        
        if i > 0 and np.max(np.abs(z - z_prev)) < 1e-6:
            break
        z_prev = z.copy()
    
    return z


def baseline_median(y: np.ndarray, window_size: int = 51) -> np.ndarray:
    """
    中值滤波基线估计
    适用于快速估计平滑的基线
    
    Args:
        y: 强度数组
        window_size: 窗口大小，应为奇数
    
    Returns:
        估计的基线
    """
    from scipy.ndimage import median_filter
    return median_filter(y, size=window_size)


def remove_baseline(intensities: np.ndarray, method: str = 'als',
                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    基线校正
    
    Args:
        intensities: 强度数组
        method: 校正方法:
            - 'als': 非对称最小二乘法（推荐用于荧光背景）
            - 'airpls': 自适应迭代加权惩罚最小二乘法
            - 'iarpls': 改进的自适应迭代加权惩罚最小二乘法
            - 'polyfit': 多项式拟合（适用于简单的倾斜基线）
            - 'median': 中值滤波
        **kwargs: 传递给具体方法的参数
    
    Returns:
        (校正后的强度, 基线)
    """
    if method == 'als':
        baseline = baseline_als(intensities, **kwargs)
    elif method == 'airpls':
        baseline = baseline_airpls(intensities, **kwargs)
    elif method == 'iarpls':
        baseline = baseline_iarpls(intensities, **kwargs)
    elif method == 'polyfit':
        baseline = baseline_polyfit(intensities, **kwargs)
    elif method == 'median':
        baseline = baseline_median(intensities, **kwargs)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    
    corrected = intensities - baseline
    corrected = np.maximum(corrected, 0)
    
    return corrected, baseline


def remove_fluorescence_background(wavelengths: np.ndarray, intensities: np.ndarray,
                                  baseline_method: str = 'airpls',
                                  baseline_kwargs: dict = None,
                                  smooth_first: bool = True,
                                  smooth_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    专门针对荧光背景的去除处理
    
    Args:
        wavelengths: 波长数组
        intensities: 强度数组
        baseline_method: 基线校正方法
        baseline_kwargs: 基线校正参数
        smooth_first: 是否先平滑再去基线
        smooth_window: 平滑窗口大小
    
    Returns:
        (校正后的强度, 基线)
    """
    if baseline_kwargs is None:
        baseline_kwargs = {'lam': 1e5}
    
    processed = intensities.copy()
    
    if smooth_first:
        processed = savgol_filter(processed, smooth_window, 2)
    
    corrected, baseline = remove_baseline(processed, method=baseline_method, **baseline_kwargs)
    
    return corrected, baseline


def smooth_spectrum(intensities: np.ndarray, method: str = 'savgol',
                    **kwargs) -> np.ndarray:
    """
    光谱平滑
    
    Args:
        intensities: 强度数组
        method: 平滑方法，'savgol', 'gaussian', 'moving_average'
        **kwargs: 传递给具体方法的参数
    
    Returns:
        平滑后的强度数组
    """
    if method == 'savgol':
        window_length = kwargs.get('window_length', 5)
        polyorder = kwargs.get('polyorder', 2)
        return savgol_filter(intensities, window_length, polyorder)
    
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return gaussian_filter1d(intensities, sigma)
    
    elif method == 'moving_average':
        window_size = kwargs.get('window_size', 5)
        kernel = np.ones(window_size) / window_size
        return np.convolve(intensities, kernel, mode='same')
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def normalize_spectrum(intensities: np.ndarray,
                       method: str = 'minmax') -> np.ndarray:
    """
    光谱归一化
    
    Args:
        intensities: 强度数组
        method: 归一化方法，'minmax', 'zscore', 'l2'
    
    Returns:
        归一化后的强度数组
    """
    return normalize_data(intensities, method)


def preprocess_pipeline(wavelengths: np.ndarray, intensities: np.ndarray,
                        smooth: bool = True,
                        smooth_method: str = 'savgol',
                        smooth_kwargs: dict = None,
                        baseline: bool = True,
                        baseline_method: str = 'als',
                        baseline_kwargs: dict = None,
                        normalize: bool = True,
                        normalize_method: str = 'minmax') -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    完整的预处理流水线
    
    Args:
        wavelengths: 波长数组
        intensities: 强度数组
        smooth: 是否平滑
        smooth_method: 平滑方法
        smooth_kwargs: 平滑参数
        baseline: 是否基线校正
        baseline_method: 基线校正方法
        baseline_kwargs: 基线校正参数
        normalize: 是否归一化
        normalize_method: 归一化方法
    
    Returns:
        (处理后的波长, 处理后的强度, 处理信息字典)
    """
    if smooth_kwargs is None:
        smooth_kwargs = {}
    if baseline_kwargs is None:
        baseline_kwargs = {}
    
    info = {}
    processed_intensities = intensities.copy()
    
    if smooth:
        processed_intensities = smooth_spectrum(
            processed_intensities, method=smooth_method, **smooth_kwargs
        )
        info['smoothed'] = True
        info['smooth_method'] = smooth_method
    
    if baseline:
        processed_intensities, baseline_est = remove_baseline(
            processed_intensities, method=baseline_method, **baseline_kwargs
        )
        info['baseline_corrected'] = True
        info['baseline_method'] = baseline_method
        info['baseline'] = baseline_est
    
    if normalize:
        processed_intensities = normalize_spectrum(
            processed_intensities, method=normalize_method
        )
        info['normalized'] = True
        info['normalize_method'] = normalize_method
    
    return wavelengths, processed_intensities, info
