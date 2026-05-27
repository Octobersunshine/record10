import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import trapezoid
from typing import Tuple, List, Dict, Optional


def detect_peaks(intensities: np.ndarray,
                 height: Optional[float] = None,
                 threshold: Optional[float] = None,
                 distance: Optional[int] = None,
                 prominence: Optional[float] = None,
                 width: Optional[float] = None,
                 wlen: Optional[int] = None,
                 rel_height: float = 0.5,
                 plateau_size: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """
    检测谱峰
    
    Args:
        intensities: 强度数组
        height: 峰的最小高度
        threshold: 峰的最小阈值（相对于相邻点）
        distance: 相邻峰之间的最小距离
        prominence: 峰的突出度
        width: 峰的最小宽度
        wlen: 计算突出度时使用的窗口长度
        rel_height: 计算峰宽时的相对高度
        plateau_size: 平台的最小大小
    
    Returns:
        (峰索引数组, 峰属性字典)
    """
    peaks, properties = find_peaks(
        intensities,
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
        plateau_size=plateau_size
    )
    
    return peaks, properties


def get_peak_positions(wavelengths: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """获取峰的波长位置"""
    return wavelengths[peaks]


def get_peak_heights(intensities: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """获取峰的高度"""
    return intensities[peaks]


def calculate_peak_areas(wavelengths: np.ndarray, intensities: np.ndarray,
                         peaks: np.ndarray, widths: np.ndarray,
                         rel_height: float = 0.5) -> np.ndarray:
    """
    计算峰的面积
    
    Args:
        wavelengths: 波长数组
        intensities: 强度数组
        peaks: 峰索引数组
        widths: 峰宽数组
        rel_height: 计算峰宽时的相对高度
    
    Returns:
        峰面积数组
    """
    areas = []
    for i, peak in enumerate(peaks):
        width = widths[i]
        half_height = intensities[peak] * (1 - rel_height)
        
        left_idx = int(np.floor(peak - width / 2))
        right_idx = int(np.ceil(peak + width / 2))
        
        left_idx = max(0, left_idx)
        right_idx = min(len(intensities) - 1, right_idx)
        
        baseline = np.linspace(
            intensities[left_idx], intensities[right_idx],
            right_idx - left_idx + 1
        )
        
        peak_region = intensities[left_idx:right_idx + 1] - baseline
        peak_region = np.maximum(peak_region, 0)
        
        dx = np.diff(wavelengths[left_idx:right_idx + 2])
        area = trapezoid(peak_region, dx=dx.mean() if len(dx) > 0 else 1.0)
        areas.append(area)
    
    return np.array(areas)


def analyze_peaks(wavelengths: np.ndarray, intensities: np.ndarray,
                  **peak_kwargs) -> Dict:
    """
    完整的谱峰分析
    
    Args:
        wavelengths: 波长数组
        intensities: 强度数组
        **peak_kwargs: 传递给detect_peaks的参数
    
    Returns:
        包含所有峰信息的字典
    """
    peaks, properties = detect_peaks(intensities, **peak_kwargs)
    
    if len(peaks) == 0:
        return {
            'peaks': np.array([]),
            'positions': np.array([]),
            'heights': np.array([]),
            'widths': np.array([]),
            'prominences': np.array([]),
            'areas': np.array([]),
            'properties': {}
        }
    
    positions = get_peak_positions(wavelengths, peaks)
    heights = get_peak_heights(intensities, peaks)
    widths = properties.get('widths', np.zeros_like(peaks, dtype=float))
    prominences = properties.get('prominences', np.zeros_like(peaks, dtype=float))
    areas = calculate_peak_areas(wavelengths, intensities, peaks, widths)
    
    return {
        'peaks': peaks,
        'positions': positions,
        'heights': heights,
        'widths': widths,
        'prominences': prominences,
        'areas': areas,
        'properties': properties
    }


def create_peak_vector(wavelengths: np.ndarray, intensities: np.ndarray,
                       peak_analysis: Dict,
                       num_bins: int = 100,
                       wl_range: Optional[Tuple[float, float]] = None,
                       use_height: bool = True) -> np.ndarray:
    """
    创建用于匹配的峰向量
    
    Args:
        wavelengths: 波长数组
        intensities: 强度数组
        peak_analysis: 谱峰分析结果
        num_bins: 分箱数量
        wl_range: 波长范围，默认为数据的范围
        use_height: 是否使用峰高作为权重
    
    Returns:
        峰向量
    """
    if wl_range is None:
        wl_min, wl_max = wavelengths.min(), wavelengths.max()
    else:
        wl_min, wl_max = wl_range
    
    peak_vector = np.zeros(num_bins)
    bin_edges = np.linspace(wl_min, wl_max, num_bins + 1)
    
    positions = peak_analysis['positions']
    heights = peak_analysis['heights']
    
    for i, pos in enumerate(positions):
        if wl_min <= pos <= wl_max:
            bin_idx = np.digitize(pos, bin_edges) - 1
            bin_idx = min(max(bin_idx, 0), num_bins - 1)
            
            if use_height:
                peak_vector[bin_idx] = max(peak_vector[bin_idx], heights[i])
            else:
                peak_vector[bin_idx] += 1
    
    return peak_vector


def filter_peaks(peak_analysis: Dict,
                 min_height: Optional[float] = None,
                 min_prominence: Optional[float] = None,
                 min_width: Optional[float] = None,
                 max_width: Optional[float] = None,
                 wl_range: Optional[Tuple[float, float]] = None) -> Dict:
    """
    过滤谱峰
    
    Args:
        peak_analysis: 谱峰分析结果
        min_height: 最小峰高
        min_prominence: 最小突出度
        min_width: 最小宽度
        max_width: 最大宽度
        wl_range: 波长范围
    
    Returns:
        过滤后的谱峰分析结果
    """
    mask = np.ones_like(peak_analysis['peaks'], dtype=bool)
    
    if min_height is not None:
        mask &= peak_analysis['heights'] >= min_height
    
    if min_prominence is not None:
        mask &= peak_analysis['prominences'] >= min_prominence
    
    if min_width is not None:
        mask &= peak_analysis['widths'] >= min_width
    
    if max_width is not None:
        mask &= peak_analysis['widths'] <= max_width
    
    if wl_range is not None:
        wl_min, wl_max = wl_range
        mask &= (peak_analysis['positions'] >= wl_min) & \
                (peak_analysis['positions'] <= wl_max)
    
    filtered = {
        'peaks': peak_analysis['peaks'][mask],
        'positions': peak_analysis['positions'][mask],
        'heights': peak_analysis['heights'][mask],
        'widths': peak_analysis['widths'][mask],
        'prominences': peak_analysis['prominences'][mask],
        'areas': peak_analysis['areas'][mask],
        'properties': peak_analysis['properties']
    }
    
    return filtered
