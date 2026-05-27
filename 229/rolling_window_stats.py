import numpy as np
from typing import Union, List, Tuple, Optional


def ewma(
    data: Union[List[float], np.ndarray],
    alpha: float = 0.3
) -> np.ndarray:
    """
    计算指数加权移动平均（EWMA）

    参数:
        data: 输入时间序列数据
        alpha: 衰减因子（0 < alpha <= 1），值越大近期数据权重越高

    返回:
        EWMA序列
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    result = np.zeros(n)
    result[0] = data[0]
    
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    
    return result


def rolling_window_stats(
    data: Union[List[float], np.ndarray],
    window_size: Union[int, List[int]],
    step: int = 1,
    mode: str = 'valid',
    percentiles: Optional[List[float]] = None,
    ewma_alpha: Optional[float] = None
) -> dict:
    """
    计算时间序列的移动窗口统计量，支持多窗口并行计算

    参数:
        data: 输入时间序列数据
        window_size: 窗口大小（整数或整数列表，支持多窗口并行计算）
        step: 窗口滑动步长，默认为1
        mode: 边界处理模式
            - 'valid': 仅计算完整窗口（结果长度更短）
            - 'same': 使用部分窗口，边界处填充NaN（结果长度与输入对齐）
        percentiles: 要计算的百分位数列表，例如 [20, 80]
        ewma_alpha: EWMA衰减因子（0 < alpha <= 1），如提供则计算EWMA

    返回:
        包含各统计量的字典。多窗口时返回嵌套字典。
    """
    data = np.asarray(data, dtype=np.float64)
    
    if isinstance(window_size, list):
        result = {}
        for ws in window_size:
            key = f'window_{ws}'
            result[key] = rolling_window_stats(
                data, ws, step, mode, percentiles, None
            )
        
        if ewma_alpha is not None:
            result['ewma'] = ewma(data, ewma_alpha)
        
        return result
    
    if window_size <= 0:
        raise ValueError("窗口大小必须大于0")
    if step <= 0:
        raise ValueError("步长必须大于0")
    if window_size > len(data):
        raise ValueError("窗口大小不能大于数据长度")
    if mode not in ['valid', 'same']:
        raise ValueError("mode 必须是 'valid' 或 'same'")
    
    n = len(data)
    pcts = percentiles if percentiles else []
    
    if mode == 'valid':
        num_windows = (n - window_size) // step + 1
        indices = np.arange(num_windows)[:, np.newaxis] * step + np.arange(window_size)
        windows = data[indices]
        
        stats = {
            'mean': np.mean(windows, axis=1),
            'median': np.median(windows, axis=1),
            'std': np.std(windows, axis=1, ddof=1),
            'max': np.max(windows, axis=1),
            'min': np.min(windows, axis=1)
        }
        
        for p in pcts:
            stats[f'p{p}'] = np.percentile(windows, p, axis=1)
    else:
        num_windows = (n + step - 1) // step
        half_window = window_size // 2
        
        stats = {
            'mean': np.full(num_windows, np.nan),
            'median': np.full(num_windows, np.nan),
            'std': np.full(num_windows, np.nan),
            'max': np.full(num_windows, np.nan),
            'min': np.full(num_windows, np.nan)
        }
        
        for p in pcts:
            stats[f'p{p}'] = np.full(num_windows, np.nan)
        
        for i in range(num_windows):
            center = i * step
            start = max(0, center - half_window)
            end = min(n, center + (window_size - half_window))
            
            if start < end:
                window_data = data[start:end]
                stats['mean'][i] = np.mean(window_data)
                stats['median'][i] = np.median(window_data)
                stats['std'][i] = np.std(window_data, ddof=1) if len(window_data) > 1 else np.nan
                stats['max'][i] = np.max(window_data)
                stats['min'][i] = np.min(window_data)
                
                for p in pcts:
                    stats[f'p{p}'][i] = np.percentile(window_data, p)
    
    if ewma_alpha is not None:
        stats['ewma'] = ewma(data, ewma_alpha)
    
    return stats


if __name__ == "__main__":
    np.random.seed(42)
    data = np.sin(np.linspace(0, 4 * np.pi, 20)) + np.random.normal(0, 0.1, 20)
    data = np.round(data, 3)
    
    print("原始数据:", data.tolist())
    print(f"数据长度: {len(data)}")
    
    print("\n=== 单窗口 + 百分位数 + EWMA ===")
    stats1 = rolling_window_stats(
        data, window_size=5, step=2, mode='valid',
        percentiles=[20, 80], ewma_alpha=0.3
    )
    for key, value in stats1.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {np.round(value, 3)}")
    
    print("\n=== 多窗口并行计算 ===")
    stats_multi = rolling_window_stats(
        data, window_size=[3, 5, 7], step=2, mode='valid',
        percentiles=[25, 75], ewma_alpha=0.3
    )
    for window_key, window_stats in stats_multi.items():
        print(f"\n{window_key}:")
        if isinstance(window_stats, dict):
            for stat_key, value in window_stats.items():
                print(f"  {stat_key}: {np.round(value, 3)}")
        else:
            print(f"  {np.round(window_stats, 3)}")
