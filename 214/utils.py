import numpy as np
from typing import Tuple, List


def is_monotonic(x: np.ndarray) -> bool:
    """检查数组是否单调递增或递减"""
    return np.all(x[:-1] <= x[1:]) or np.all(x[:-1] >= x[1:])


def interpolate_spectrum(wavelengths: np.ndarray, intensities: np.ndarray,
                         target_wavelengths: np.ndarray) -> np.ndarray:
    """将光谱插值到目标波长轴上"""
    return np.interp(target_wavelengths, wavelengths, intensities)


def resample_spectrum(wavelengths: np.ndarray, intensities: np.ndarray,
                      num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """将光谱重采样到指定点数"""
    new_wavelengths = np.linspace(wavelengths.min(), wavelengths.max(), num_points)
    new_intensities = interpolate_spectrum(wavelengths, intensities, new_wavelengths)
    return new_wavelengths, new_intensities


def get_wavelength_range(wavelengths_list: List[np.ndarray]) -> Tuple[float, float]:
    """获取多个光谱的公共波长范围"""
    min_wl = max(wl.min() for wl in wavelengths_list)
    max_wl = min(wl.max() for wl in wavelengths_list)
    return min_wl, max_wl


def align_spectra(wavelengths_list: List[np.ndarray], intensities_list: List[np.ndarray],
                  num_points: int = 1000) -> Tuple[np.ndarray, List[np.ndarray]]:
    """将多个光谱对齐到同一波长轴上"""
    min_wl, max_wl = get_wavelength_range(wavelengths_list)
    target_wavelengths = np.linspace(min_wl, max_wl, num_points)
    
    aligned_intensities = []
    for wl, inten in zip(wavelengths_list, intensities_list):
        mask = (wl >= min_wl) & (wl <= max_wl)
        aligned = interpolate_spectrum(wl[mask], inten[mask], target_wavelengths)
        aligned_intensities.append(aligned)
    
    return target_wavelengths, aligned_intensities


def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """归一化数据"""
    data = np.asarray(data, dtype=float)
    
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min < 1e-10:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return np.zeros_like(data)
        return (data - mean) / std
    
    elif method == 'l2':
        norm = np.linalg.norm(data)
        if norm < 1e-10:
            return np.zeros_like(data)
        return data / norm
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
