import numpy as np
from typing import Union, List, Tuple, Dict


def _handle_nan_values(
    x: np.ndarray,
    y: np.ndarray,
    nan_handling: str
) -> Tuple[np.ndarray, np.ndarray]:
    if nan_handling not in ['ignore', 'mean', 'error']:
        raise ValueError("nan_handling 必须是 'ignore', 'mean' 或 'error'")
    
    nan_mask = np.isnan(x) | np.isnan(y)
    
    if nan_handling == 'error':
        if np.any(nan_mask):
            raise ValueError("向量中存在缺失值(NaN)，请使用 nan_handling 参数指定处理方式")
    
    elif nan_handling == 'mean':
        for dim in range(x.shape[-1]):
            if x.ndim == 1:
                dim_mean = np.nanmean(np.concatenate([[x[dim]], [y[dim]]]))
                if np.isnan(x[dim]):
                    x[dim] = dim_mean
                if np.isnan(y[dim]):
                    y[dim] = dim_mean
            else:
                col_x = x[:, dim]
                col_y = y[:, dim]
                dim_mean = np.nanmean(np.concatenate([col_x, col_y]))
                x[np.isnan(x[:, dim]), dim] = dim_mean
                y[np.isnan(y[:, dim]), dim] = dim_mean
    
    return x, y


def _compute_abs_diff(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    nan_handling: str = 'error'
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if x.shape != y.shape:
        raise ValueError("两个向量的形状必须相同")
    if x.ndim not in (1, 2):
        raise ValueError("输入必须是1维（单个向量）或2维（批量向量）")
    
    x, y = _handle_nan_values(x, y, nan_handling)
    
    diff = np.abs(x - y)
    
    if nan_handling == 'ignore':
        diff[np.isnan(diff)] = 0
    
    return diff


def manhattan_distance(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    nan_handling: str = 'error'
) -> Union[float, np.ndarray]:
    diff = _compute_abs_diff(x, y, nan_handling)
    
    if diff.ndim == 1:
        return np.sum(diff)
    else:
        return np.sum(diff, axis=1)


def euclidean_distance(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    nan_handling: str = 'error'
) -> Union[float, np.ndarray]:
    diff = _compute_abs_diff(x, y, nan_handling)
    
    if diff.ndim == 1:
        return np.sqrt(np.sum(diff ** 2))
    else:
        return np.sqrt(np.sum(diff ** 2, axis=1))


def chebyshev_distance(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    nan_handling: str = 'error'
) -> Union[float, np.ndarray]:
    diff = _compute_abs_diff(x, y, nan_handling)
    
    if diff.ndim == 1:
        return np.max(diff)
    else:
        return np.max(diff, axis=1)


def minkowski_distance(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    p: float = 2,
    nan_handling: str = 'error'
) -> Union[float, np.ndarray]:
    if p <= 0:
        raise ValueError("p 必须大于0")
    
    diff = _compute_abs_diff(x, y, nan_handling)
    
    if diff.ndim == 1:
        return np.sum(diff ** p) ** (1.0 / p)
    else:
        return np.sum(diff ** p, axis=1) ** (1.0 / p)


def dimension_diff_distribution(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    nan_handling: str = 'error'
) -> Dict:
    diff = _compute_abs_diff(x, y, nan_handling)
    
    if diff.ndim == 1:
        total = np.sum(diff)
        if total == 0:
            contribution = np.zeros_like(diff)
        else:
            contribution = diff / total
        
        return {
            'abs_diff': diff,
            'contribution': contribution,
            'sorted_indices': np.argsort(-diff),
            'max_dim': int(np.argmax(diff)),
            'max_diff': float(np.max(diff)),
        }
    else:
        totals = np.sum(diff, axis=1, keepdims=True)
        safe_totals = np.where(totals == 0, 1.0, totals)
        contribution = diff / safe_totals
        
        return {
            'abs_diff': diff,
            'contribution': contribution,
            'sorted_indices': np.argsort(-diff, axis=1),
            'max_dim': np.argmax(diff, axis=1),
            'max_diff': np.max(diff, axis=1),
        }


def compute_distances(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    nan_handling: str = 'error',
    p: float = 2
) -> Dict[str, Union[float, np.ndarray]]:
    return {
        'manhattan': manhattan_distance(x, y, nan_handling),
        'euclidean': euclidean_distance(x, y, nan_handling),
        'chebyshev': chebyshev_distance(x, y, nan_handling),
        'minkowski': minkowski_distance(x, y, p, nan_handling),
    }


if __name__ == "__main__":
    print("=== 单点距离计算示例 ===")
    x1 = [1, 2, 3]
    y1 = [4, 6, 8]
    results = compute_distances(x1, y1, p=3)
    print(f"向量x: {x1}")
    print(f"向量y: {y1}")
    for name, val in results.items():
        if name == 'minkowski':
            print(f"闵可夫斯基距离(L3): {val:.4f}")
        else:
            print(f"{name}距离: {val:.4f}")
    
    print("\n=== 维度差值分布分析 ===")
    dist_info = dimension_diff_distribution(x1, y1)
    print(f"各维度绝对差值: {dist_info['abs_diff']}")
    print(f"各维度贡献比例: {dist_info['contribution']}")
    print(f"贡献排序(降维索引): {dist_info['sorted_indices']}")
    print(f"最大贡献维度: {dist_info['max_dim']}, 差值: {dist_info['max_diff']}")
    
    print("\n=== 批量计算示例 ===")
    x_batch = [[1, 2, 3], [0, 0, 0], [2, 4, 6]]
    y_batch = [[4, 6, 8], [1, 1, 1], [3, 5, 7]]
    results_batch = compute_distances(x_batch, y_batch, p=3)
    for name, val in results_batch.items():
        print(f"{name}距离: {val}")
    
    print("\n=== 批量维度差值分布 ===")
    batch_info = dimension_diff_distribution(x_batch, y_batch)
    for i in range(len(x_batch)):
        print(f"\n点对{i+1}: x={x_batch[i]}, y={y_batch[i]}")
        print(f"  绝对差值: {batch_info['abs_diff'][i]}")
        print(f"  贡献比例: {batch_info['contribution'][i]}")
        print(f"  最大贡献维度: {batch_info['max_dim'][i]}, 差值: {batch_info['max_diff'][i]}")
    
    print("\n=== 闵可夫斯基距离不同p值对比 ===")
    x_m, y_m = [1, 2, 3], [4, 6, 8]
    for p_val in [0.5, 1, 2, 3, 5, 10, 50]:
        d = minkowski_distance(x_m, y_m, p=p_val)
        label = {1: 'L1(曼哈顿)', 2: 'L2(欧氏)'}.get(p_val, f'L{p_val}')
        print(f"  p={p_val:<4} ({label}): {d:.4f}")
    d_inf = chebyshev_distance(x_m, y_m)
    print(f"  L∞   (切比雪夫):  {d_inf:.4f}")
    
    print("\n=== 含NaN的维度差值分布 ===")
    x_nan = [1, np.nan, 3]
    y_nan = [4, 6, np.nan]
    nan_info = dimension_diff_distribution(x_nan, y_nan, nan_handling='ignore')
    print(f"向量x（含NaN）: {x_nan}")
    print(f"向量y（含NaN）: {y_nan}")
    print(f"绝对差值(ignore): {nan_info['abs_diff']}")
    print(f"贡献比例: {nan_info['contribution']}")
    print(f"最大贡献维度: {nan_info['max_dim']}")
