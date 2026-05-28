import numpy as np


def quantile_normalize(matrix, nan_policy='rank', interpolation_method='linear',
                       reference_dist=None, return_reference=False):
    """
    分位数归一化：将多个样本的分布对齐到相同的分位数分布。

    参数:
        matrix: 二维数组或DataFrame，每列一个样本，每行一个特征
        nan_policy: 缺失值处理策略
            - 'drop': 删除包含任何缺失值的行
            - 'interpolate': 使用插值填充缺失值后进行归一化
            - 'rank': 按非缺失值排序，仅对非缺失值进行归一化，
                      缺失值保持NaN不变（默认）
        interpolation_method: 插值方法，当nan_policy='interpolate'时使用
            - 'linear': 线性插值
            - 'nearest': 最近邻插值
            - 'mean': 列均值填充
        reference_dist: 参考分布分位数，一维数组。如果提供，则使用该分布
            作为目标分布而不是计算所有样本的均值分位数。
            用于多批次数据校正，长度应与数据行数相同。
        return_reference: 是否返回使用的参考分布。如果为True，返回
            (normalized_data, reference_dist) 元组。

    返回:
        归一化后的数据矩阵，或 (normalized_data, reference_dist) 元组
    """
    data = np.array(matrix, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("输入必须是二维矩阵")

    n_rows, n_cols = data.shape

    ref = _validate_reference(reference_dist, n_rows)

    has_nan = np.isnan(data).any()

    if not has_nan:
        normalized, ref_used = _quantile_normalize_no_nan(data, ref)
    elif nan_policy == 'drop':
        normalized, ref_used = _quantile_normalize_drop_nan(data, ref)
    elif nan_policy == 'interpolate':
        normalized, ref_used = _quantile_normalize_interpolate(
            data, interpolation_method, ref
        )
    elif nan_policy == 'rank':
        normalized, ref_used = _quantile_normalize_rank_nan(data, ref)
    else:
        raise ValueError(
            f"未知的nan_policy: {nan_policy}，"
            f"可选值: 'drop', 'interpolate', 'rank'"
        )

    if return_reference:
        return normalized, ref_used
    return normalized


def _validate_reference(reference_dist, expected_len):
    """验证参考分布的格式"""
    if reference_dist is None:
        return None
    ref = np.asarray(reference_dist, dtype=np.float64)
    if ref.ndim != 1:
        raise ValueError("参考分布必须是一维数组")
    if len(ref) != expected_len:
        raise ValueError(
            f"参考分布长度 ({len(ref)}) 必须与数据行数 ({expected_len}) 相同"
        )
    return ref


def _quantile_normalize_no_nan(data, reference_dist=None):
    """无缺失值时的标准分位数归一化"""
    n_rows, n_cols = data.shape

    sorted_data = np.sort(data, axis=0)

    if reference_dist is not None:
        ref = np.asarray(reference_dist, dtype=np.float64)
        if len(ref) == n_rows:
            mean_quantiles = ref
        else:
            mean_quantiles = np.interp(
                np.linspace(0, 1, n_rows),
                np.linspace(0, 1, len(ref)),
                np.sort(ref)
            )
    else:
        mean_quantiles = np.mean(sorted_data, axis=1)

    result = np.zeros_like(data)
    for i in range(n_cols):
        order = np.argsort(data[:, i])
        result[order, i] = mean_quantiles

    return result, mean_quantiles


def _quantile_normalize_drop_nan(data, reference_dist=None):
    """删除包含任何缺失值的行后进行归一化"""
    valid_rows = ~np.isnan(data).any(axis=1)
    valid_data = data[valid_rows]

    if valid_data.shape[0] == 0:
        raise ValueError("删除含缺失值行后无有效数据")

    ref_for_valid = None
    if reference_dist is not None:
        ref_for_valid = reference_dist[valid_rows]

    normalized, ref_used = _quantile_normalize_no_nan(valid_data, ref_for_valid)

    result = np.full_like(data, np.nan)
    result[valid_rows] = normalized

    ref_full = np.full(data.shape[0], np.nan)
    ref_full[valid_rows] = ref_used

    return result, ref_full


def _quantile_normalize_interpolate(data, method='linear', reference_dist=None):
    """使用插值填充缺失值后进行归一化"""
    data_filled = data.copy()
    n_cols = data.shape[1]

    for i in range(n_cols):
        col = data[:, i]
        nan_mask = np.isnan(col)

        if not nan_mask.any():
            continue

        valid_mask = ~nan_mask
        x_valid = np.where(valid_mask)[0]
        y_valid = col[valid_mask]

        if len(y_valid) < 2:
            col_mean = np.nanmean(col)
            data_filled[nan_mask, i] = col_mean if np.isfinite(col_mean) else 0
            continue

        x_nan = np.where(nan_mask)[0]

        if method == 'linear':
            data_filled[nan_mask, i] = np.interp(x_nan, x_valid, y_valid)
        elif method == 'nearest':
            for x in x_nan:
                nearest_idx = x_valid[np.argmin(np.abs(x_valid - x))]
                data_filled[x, i] = col[nearest_idx]
        elif method == 'mean':
            col_mean = np.nanmean(col)
            data_filled[nan_mask, i] = col_mean if np.isfinite(col_mean) else 0
        else:
            raise ValueError(
                f"未知的插值方法: {method}，"
                f"可选值: 'linear', 'nearest', 'mean'"
            )

    return _quantile_normalize_no_nan(data_filled, reference_dist)


def _quantile_normalize_rank_nan(data, reference_dist=None):
    """按非缺失值排序，仅对非缺失值进行归一化，缺失值保持NaN不变"""
    n_rows, n_cols = data.shape
    result = np.full_like(data, np.nan)

    ref_sorted = None
    if reference_dist is not None:
        ref_sorted = np.sort(reference_dist)

    for i in range(n_cols):
        col = data[:, i]
        valid_mask = ~np.isnan(col)
        valid_values = col[valid_mask]
        n_valid = len(valid_values)

        if n_valid == 0:
            continue

        sorted_cols = []
        for j in range(n_cols):
            col_j = data[:, j]
            valid_j = col_j[~np.isnan(col_j)]
            if len(valid_j) == n_valid:
                sorted_cols.append(np.sort(valid_j))
            else:
                valid_sorted = np.sort(valid_j)
                interp = np.interp(
                    np.linspace(0, 1, n_valid),
                    np.linspace(0, 1, len(valid_j)),
                    valid_sorted
                )
                sorted_cols.append(interp)

        sorted_matrix = np.column_stack(sorted_cols)

        if ref_sorted is not None:
            if len(ref_sorted) == n_valid:
                mean_quantiles = ref_sorted
            else:
                mean_quantiles = np.interp(
                    np.linspace(0, 1, n_valid),
                    np.linspace(0, 1, len(ref_sorted)),
                    ref_sorted
                )
        else:
            mean_quantiles = np.mean(sorted_matrix, axis=1)

        order = np.argsort(valid_values)
        normalized_valid = np.empty_like(valid_values)
        normalized_valid[order] = mean_quantiles

        result[valid_mask, i] = normalized_valid

    return result, mean_quantiles


def quantile_normalize_mean_rank(matrix, nan_policy='rank',
                                  interpolation_method='linear',
                                  reference_dist=None,
                                  return_reference=False):
    """
    分位数归一化（处理并列值版本）：使用排序的平均值处理并列值。

    参数:
        matrix: 二维数组或DataFrame，每列一个样本，每行一个特征
        nan_policy: 缺失值处理策略
            - 'drop': 删除包含任何缺失值的行
            - 'interpolate': 使用插值填充缺失值后进行归一化
            - 'rank': 按非缺失值排序，仅对非缺失值进行归一化，
                      缺失值保持NaN不变（默认）
        interpolation_method: 插值方法，当nan_policy='interpolate'时使用
            - 'linear': 线性插值
            - 'nearest': 最近邻插值
            - 'mean': 列均值填充
        reference_dist: 参考分布分位数，一维数组。如果提供，则使用该分布
            作为目标分布而不是计算所有样本的均值分位数。
            用于多批次数据校正，长度应与数据行数相同。
        return_reference: 是否返回使用的参考分布。如果为True，返回
            (normalized_data, reference_dist) 元组。

    返回:
        归一化后的数据矩阵，或 (normalized_data, reference_dist) 元组
    """
    from scipy import stats

    data = np.array(matrix, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("输入必须是二维矩阵")

    n_rows = data.shape[0]
    ref = _validate_reference(reference_dist, n_rows)

    has_nan = np.isnan(data).any()

    if not has_nan:
        normalized, ref_used = _quantile_normalize_mean_rank_no_nan(data, ref)
    elif nan_policy == 'drop':
        normalized, ref_used = _quantile_normalize_mean_rank_drop_nan(data, ref)
    elif nan_policy == 'interpolate':
        normalized, ref_used = _quantile_normalize_mean_rank_interpolate(
            data, interpolation_method, ref
        )
    elif nan_policy == 'rank':
        normalized, ref_used = _quantile_normalize_mean_rank_rank_nan(data, ref)
    else:
        raise ValueError(
            f"未知的nan_policy: {nan_policy}，"
            f"可选值: 'drop', 'interpolate', 'rank'"
        )

    if return_reference:
        return normalized, ref_used
    return normalized


def _quantile_normalize_mean_rank_no_nan(data, reference_dist=None):
    """无缺失值时的分位数归一化（处理并列值）"""
    from scipy import stats

    n_rows, n_cols = data.shape

    sorted_data = np.sort(data, axis=0)

    if reference_dist is not None:
        ref = np.asarray(reference_dist, dtype=np.float64)
        if len(ref) == n_rows:
            mean_quantiles = ref
        else:
            mean_quantiles = np.interp(
                np.linspace(0, 1, n_rows),
                np.linspace(0, 1, len(ref)),
                np.sort(ref)
            )
    else:
        mean_quantiles = np.mean(sorted_data, axis=1)

    result = np.zeros_like(data)
    for i in range(n_cols):
        ranks = stats.rankdata(data[:, i], method='average')
        for j in range(n_rows):
            rank = ranks[j]
            lower_idx = int(np.floor(rank)) - 1
            upper_idx = int(np.ceil(rank)) - 1
            if lower_idx == upper_idx:
                result[j, i] = mean_quantiles[lower_idx]
            else:
                frac = rank - (lower_idx + 1)
                result[j, i] = (1 - frac) * mean_quantiles[lower_idx] + \
                    frac * mean_quantiles[upper_idx]

    return result, mean_quantiles


def _quantile_normalize_mean_rank_drop_nan(data, reference_dist=None):
    """删除含缺失值行后进行归一化（处理并列值）"""
    valid_rows = ~np.isnan(data).any(axis=1)
    valid_data = data[valid_rows]

    if valid_data.shape[0] == 0:
        raise ValueError("删除含缺失值行后无有效数据")

    ref_for_valid = None
    if reference_dist is not None:
        ref_for_valid = reference_dist[valid_rows]

    normalized, ref_used = _quantile_normalize_mean_rank_no_nan(
        valid_data, ref_for_valid
    )

    result = np.full_like(data, np.nan)
    result[valid_rows] = normalized

    ref_full = np.full(data.shape[0], np.nan)
    ref_full[valid_rows] = ref_used

    return result, ref_full


def _quantile_normalize_mean_rank_interpolate(data, method='linear',
                                               reference_dist=None):
    """插值填充后进行归一化（处理并列值）"""
    data_filled = data.copy()
    n_cols = data.shape[1]

    for i in range(n_cols):
        col = data[:, i]
        nan_mask = np.isnan(col)

        if not nan_mask.any():
            continue

        valid_mask = ~nan_mask
        x_valid = np.where(valid_mask)[0]
        y_valid = col[valid_mask]

        if len(y_valid) < 2:
            col_mean = np.nanmean(col)
            data_filled[nan_mask, i] = col_mean if np.isfinite(col_mean) else 0
            continue

        x_nan = np.where(nan_mask)[0]

        if method == 'linear':
            data_filled[nan_mask, i] = np.interp(x_nan, x_valid, y_valid)
        elif method == 'nearest':
            for x in x_nan:
                nearest_idx = x_valid[np.argmin(np.abs(x_valid - x))]
                data_filled[x, i] = col[nearest_idx]
        elif method == 'mean':
            col_mean = np.nanmean(col)
            data_filled[nan_mask, i] = col_mean if np.isfinite(col_mean) else 0
        else:
            raise ValueError(
                f"未知的插值方法: {method}，"
                f"可选值: 'linear', 'nearest', 'mean'"
            )

    return _quantile_normalize_mean_rank_no_nan(data_filled, reference_dist)


def _quantile_normalize_mean_rank_rank_nan(data, reference_dist=None):
    """按非缺失值排序的归一化（处理并列值和NaN）"""
    from scipy import stats

    n_rows, n_cols = data.shape
    result = np.full_like(data, np.nan)

    ref_sorted = None
    if reference_dist is not None:
        ref_sorted = np.sort(reference_dist)

    for i in range(n_cols):
        col = data[:, i]
        valid_mask = ~np.isnan(col)
        valid_values = col[valid_mask]
        n_valid = len(valid_values)

        if n_valid == 0:
            continue

        sorted_cols = []
        for j in range(n_cols):
            col_j = data[:, j]
            valid_j = col_j[~np.isnan(col_j)]
            if len(valid_j) == n_valid:
                sorted_cols.append(np.sort(valid_j))
            else:
                valid_sorted = np.sort(valid_j)
                interp = np.interp(
                    np.linspace(0, 1, n_valid),
                    np.linspace(0, 1, len(valid_j)),
                    valid_sorted
                )
                sorted_cols.append(interp)

        sorted_matrix = np.column_stack(sorted_cols)

        if ref_sorted is not None:
            if len(ref_sorted) == n_valid:
                mean_quantiles = ref_sorted
            else:
                mean_quantiles = np.interp(
                    np.linspace(0, 1, n_valid),
                    np.linspace(0, 1, len(ref_sorted)),
                    ref_sorted
                )
        else:
            mean_quantiles = np.mean(sorted_matrix, axis=1)

        ranks = stats.rankdata(valid_values, method='average')
        normalized_valid = np.empty_like(valid_values)
        for k in range(n_valid):
            rank = ranks[k]
            lower_idx = int(np.floor(rank)) - 1
            upper_idx = int(np.ceil(rank)) - 1
            if lower_idx == upper_idx:
                normalized_valid[k] = mean_quantiles[lower_idx]
            else:
                frac = rank - (lower_idx + 1)
                normalized_valid[k] = (
                    (1 - frac) * mean_quantiles[lower_idx] +
                    frac * mean_quantiles[upper_idx]
                )

        result[valid_mask, i] = normalized_valid

    return result, mean_quantiles


def quantile_denormalize(normalized_matrix, original_matrix,
                         nan_policy='rank',
                         interpolation_method='linear'):
    """
    分位数归一化的逆变换：将归一化空间的数据还原到原始分布。

    参数:
        normalized_matrix: 归一化后的数据矩阵（已对齐到参考分布）
        original_matrix: 原始数据矩阵，用于确定原始分位数分布
        nan_policy: 缺失值处理策略，需与归一化时使用的策略一致
            - 'drop': 删除包含任何缺失值的行
            - 'interpolate': 使用插值填充缺失值后进行归一化
            - 'rank': 按非缺失值排序，仅对非缺失值进行归一化，
                      缺失值保持NaN不变（默认）
        interpolation_method: 插值方法，当nan_policy='interpolate'时使用

    返回:
        逆变换后的数据矩阵，形状与normalized_matrix相同
    """
    from scipy import stats

    normalized = np.array(normalized_matrix, dtype=np.float64)
    original = np.array(original_matrix, dtype=np.float64)

    if normalized.shape != original.shape:
        raise ValueError("归一化矩阵和原始矩阵的形状必须相同")
    if normalized.ndim != 2:
        raise ValueError("输入必须是二维矩阵")

    n_rows, n_cols = normalized.shape
    result = np.full_like(normalized, np.nan)

    has_nan = np.isnan(original).any()

    for i in range(n_cols):
        orig_col = original[:, i]
        norm_col = normalized[:, i]

        orig_valid_mask = ~np.isnan(orig_col)
        norm_valid_mask = ~np.isnan(norm_col)
        valid_mask = orig_valid_mask & norm_valid_mask

        if not valid_mask.any():
            continue

        orig_valid = orig_col[valid_mask]
        norm_valid = norm_col[valid_mask]

        if has_nan and nan_policy == 'rank':
            orig_sorted = np.sort(orig_valid)
            ranks = stats.rankdata(norm_valid, method='average')

            denormed = np.empty_like(norm_valid)
            for k in range(len(norm_valid)):
                rank = ranks[k]
                lower_idx = int(np.floor(rank)) - 1
                upper_idx = int(np.ceil(rank)) - 1
                lower_idx = max(0, min(lower_idx, len(orig_sorted) - 1))
                upper_idx = max(0, min(upper_idx, len(orig_sorted) - 1))

                if lower_idx == upper_idx:
                    denormed[k] = orig_sorted[lower_idx]
                else:
                    frac = rank - (lower_idx + 1)
                    denormed[k] = (
                        (1 - frac) * orig_sorted[lower_idx] +
                        frac * orig_sorted[upper_idx]
                    )

            result[valid_mask, i] = denormed

        else:
            if has_nan and nan_policy == 'drop':
                all_valid = ~np.isnan(original).any(axis=1)
                orig_col_valid = orig_col[all_valid]
                norm_col_valid = norm_col[all_valid]
            else:
                orig_col_valid = orig_valid
                norm_col_valid = norm_valid

            if len(orig_col_valid) == 0:
                continue

            orig_sorted = np.sort(orig_col_valid)
            norm_sorted = np.sort(norm_col_valid)

            norm_to_orig = np.empty_like(norm_col_valid)
            order = np.argsort(norm_col_valid)
            norm_to_orig[order] = orig_sorted

            if has_nan and nan_policy == 'drop':
                result[all_valid, i] = norm_to_orig
            else:
                result[valid_mask, i] = norm_to_orig

    return result


def quantile_denormalize_with_reference(normalized_matrix, target_dist,
                                        nan_policy='rank'):
    """
    分位数归一化的逆变换（使用原始目标分布）：将归一化空间的数据
    映射到指定的原始分位数分布。

    对于多批次校正，流程为：
    1. 从参考批次计算 reference_dist（归一化目标分布）
       和 original_quantiles（原始分位数分布）
    2. 待校正批次使用 reference_dist 进行归一化
    3. 使用 original_quantiles 进行逆变换，得到校正到参考批次的数据

    参数:
        normalized_matrix: 归一化后的数据矩阵
        target_dist: 目标分布分位数，一维数组，长度应与数据行数相同。
            应该是参考批次的原始分位数分布（通过 compute_original_quantiles
            计算），而不是归一化用的参考分布。
        nan_policy: 缺失值处理策略，需与归一化时使用的策略一致

    返回:
        逆变换后的数据矩阵
    """
    from scipy import stats

    normalized = np.array(normalized_matrix, dtype=np.float64)
    target = np.asarray(target_dist, dtype=np.float64)

    if normalized.ndim != 2:
        raise ValueError("输入必须是二维矩阵")
    if target.ndim != 1:
        raise ValueError("目标分布必须是一维数组")

    n_rows, n_cols = normalized.shape
    target_sorted = np.sort(target)

    if len(target_sorted) != n_rows:
        target_sorted = np.interp(
            np.linspace(0, 1, n_rows),
            np.linspace(0, 1, len(target_sorted)),
            target_sorted
        )

    result = np.full_like(normalized, np.nan)

    for i in range(n_cols):
        norm_col = normalized[:, i]
        valid_mask = ~np.isnan(norm_col)
        norm_valid = norm_col[valid_mask]

        if len(norm_valid) == 0:
            continue

        if len(norm_valid) == n_rows:
            ranks = stats.rankdata(norm_valid, method='average')
            denormed = np.empty_like(norm_valid)
            for k in range(len(norm_valid)):
                rank = ranks[k]
                lower_idx = int(np.floor(rank)) - 1
                upper_idx = int(np.ceil(rank)) - 1
                if lower_idx == upper_idx:
                    denormed[k] = target_sorted[lower_idx]
                else:
                    frac = rank - (lower_idx + 1)
                    denormed[k] = (
                        (1 - frac) * target_sorted[lower_idx] +
                        frac * target_sorted[upper_idx]
                    )
        else:
            target_interp = np.interp(
                np.linspace(0, 1, len(norm_valid)),
                np.linspace(0, 1, len(target_sorted)),
                target_sorted
            )
            ranks = stats.rankdata(norm_valid, method='average')
            denormed = np.empty_like(norm_valid)
            for k in range(len(norm_valid)):
                rank = ranks[k]
                lower_idx = int(np.floor(rank)) - 1
                upper_idx = int(np.ceil(rank)) - 1
                lower_idx = max(0, min(lower_idx, len(target_interp) - 1))
                upper_idx = max(0, min(upper_idx, len(target_interp) - 1))
                if lower_idx == upper_idx:
                    denormed[k] = target_interp[lower_idx]
                else:
                    frac = rank - (lower_idx + 1)
                    denormed[k] = (
                        (1 - frac) * target_interp[lower_idx] +
                        frac * target_interp[upper_idx]
                    )

        result[valid_mask, i] = denormed

    return result


def compute_reference_distribution(matrix, nan_policy='rank',
                                    interpolation_method='linear'):
    """
    计算数据集的平均分位数分布，可作为多批次校正的参考分布。

    参数:
        matrix: 二维数组或DataFrame，每列一个样本，每行一个特征
        nan_policy: 缺失值处理策略
        interpolation_method: 插值方法

    返回:
        reference_dist: 一维数组，各分位数位置的平均值（归一化目标分布）
    """
    _, ref = quantile_normalize(
        matrix,
        nan_policy=nan_policy,
        interpolation_method=interpolation_method,
        return_reference=True
    )
    return ref


def compute_original_quantiles(matrix, nan_policy='rank'):
    """
    计算数据集的原始分位数分布，用于逆变换时还原到原始空间。

    对于多批次校正，这应该是参考批次的原始数据分位数，
    用于将归一化后的数据逆变换回参考批次的原始值空间。

    参数:
        matrix: 二维数组或DataFrame，每列一个样本，每行一个特征
        nan_policy: 缺失值处理策略

    返回:
        original_quantiles: 一维数组，各样本排序后的平均值（原始空间分位数）
    """
    data = np.array(matrix, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("输入必须是二维矩阵")

    has_nan = np.isnan(data).any()

    if not has_nan:
        sorted_data = np.sort(data, axis=0)
        return np.mean(sorted_data, axis=1)

    n_rows, n_cols = data.shape

    if nan_policy == 'drop':
        valid_rows = ~np.isnan(data).any(axis=1)
        valid_data = data[valid_rows]
        sorted_data = np.sort(valid_data, axis=0)
        mean_vals = np.mean(sorted_data, axis=1)
        result = np.full(n_rows, np.nan)
        result[valid_rows] = mean_vals
        return result

    elif nan_policy == 'rank':
        non_nan_counts = np.sum(~np.isnan(data), axis=1)
        max_valid = np.max(non_nan_counts)

        sorted_cols = []
        for j in range(n_cols):
            col_j = data[:, j]
            valid_j = col_j[~np.isnan(col_j)]
            if len(valid_j) == max_valid:
                sorted_cols.append(np.sort(valid_j))
            else:
                valid_sorted = np.sort(valid_j)
                interp = np.interp(
                    np.linspace(0, 1, max_valid),
                    np.linspace(0, 1, len(valid_j)),
                    valid_sorted
                )
                sorted_cols.append(interp)

        sorted_matrix = np.column_stack(sorted_cols)
        return np.mean(sorted_matrix, axis=1)

    elif nan_policy == 'interpolate':
        data_filled = data.copy()
        for i in range(n_cols):
            col = data[:, i]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                valid_mask = ~nan_mask
                x_valid = np.where(valid_mask)[0]
                y_valid = col[valid_mask]
                x_nan = np.where(nan_mask)[0]
                if len(y_valid) >= 2:
                    data_filled[nan_mask, i] = np.interp(
                        x_nan, x_valid, y_valid
                    )
                else:
                    col_mean = np.nanmean(col)
                    data_filled[nan_mask, i] = (
                        col_mean if np.isfinite(col_mean) else 0
                    )
        sorted_data = np.sort(data_filled, axis=0)
        return np.mean(sorted_data, axis=1)

    else:
        raise ValueError(
            f"未知的nan_policy: {nan_policy}，"
            f"可选值: 'drop', 'interpolate', 'rank'"
        )


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 70)
    print("测试1: 基础分位数归一化（无缺失值）")
    print("=" * 70)
    data = np.array([
        [5, 4, 3],
        [2, 1, 4],
        [3, 4, 6],
        [4, 2, 8],
    ])
    print("原始数据:")
    print(data)
    print()

    normalized, ref = quantile_normalize(data, return_reference=True)
    print("分位数归一化结果:")
    print(normalized)
    print()
    print("参考分布（均值分位数）:")
    print(ref)
    print()
    print("验证每列排序后相同:")
    print(np.sort(normalized, axis=0))
    print()

    print("=" * 70)
    print("测试2: 逆变换 - 从归一化空间还原")
    print("=" * 70)
    denormalized = quantile_denormalize(normalized, data)
    print("原始数据:")
    print(data)
    print()
    print("逆变换还原的数据:")
    print(denormalized)
    print()
    print("还原误差（应接近0）:")
    print(np.abs(data - denormalized))
    print()

    print("=" * 70)
    print("测试3: 自定义参考分布（多批次数据校正）")
    print("=" * 70)
    batch1 = np.array([
        [5, 4, 3],
        [2, 1, 4],
        [3, 4, 6],
        [4, 2, 8],
    ])
    batch2 = np.array([
        [10, 8, 6],
        [4, 2, 8],
        [6, 8, 12],
        [8, 4, 16],
    ])

    print("批次1（参考批次）:")
    print(batch1)
    print("批次2（待校正批次，值为批次1的2倍）:")
    print(batch2)
    print()

    ref_dist = compute_reference_distribution(batch1)
    print("从批次1计算的参考分布:")
    print(ref_dist)
    print()

    batch2_normalized, _ = quantile_normalize(
        batch2, reference_dist=ref_dist, return_reference=True
    )
    print("批次2使用批次1参考分布归一化后:")
    print(batch2_normalized)
    print()
    print("验证批次2归一化后与批次1归一化结果相同:")
    batch1_normalized = quantile_normalize(batch1, reference_dist=ref_dist)
    print("批次1归一化结果:")
    print(batch1_normalized)
    print("批次2归一化结果:")
    print(batch2_normalized)
    print("差异:")
    print(np.abs(batch1_normalized - batch2_normalized))
    print()

    print("=" * 70)
    print("测试4: 逆变换到参考批次的原始分布")
    print("=" * 70)
    batch1_original_quantiles = compute_original_quantiles(batch1)
    print("批次1的原始分位数分布（用于逆变换）:")
    print(batch1_original_quantiles)
    print()
    batch2_denormed = quantile_denormalize_with_reference(
        batch2_normalized, batch1_original_quantiles
    )
    print("批次2归一化数据:")
    print(batch2_normalized)
    print()
    print("使用批次1原始分位数逆变换后（应与批次1原始值一致）:")
    print(batch2_denormed)
    print()
    print("与批次1原始数据的差异:")
    print(np.abs(batch1 - batch2_denormed))
    print()

    print("=" * 70)
    print("测试4b: 使用原始矩阵进行逆变换")
    print("=" * 70)
    batch2_denormed_v2 = quantile_denormalize(
        batch2_normalized, batch1
    )
    print("使用批次1原始矩阵逆变换后:")
    print(batch2_denormed_v2)
    print()
    print("与批次1原始数据的差异:")
    print(np.abs(batch1 - batch2_denormed_v2))
    print()

    print("=" * 70)
    print("测试5: 含缺失值的多批次校正")
    print("=" * 70)
    batch1_nan = np.array([
        [5, 4, 3],
        [2, 1, 4],
        [np.nan, 4, 6],
        [4, 2, 8],
    ])
    batch2_nan = np.array([
        [10, np.nan, 6],
        [4, 2, 8],
        [np.nan, 8, 12],
        [8, 4, 16],
    ])
    print("批次1（含NaN）:")
    print(batch1_nan)
    print("批次2（含NaN）:")
    print(batch2_nan)
    print()

    ref_dist_nan = compute_reference_distribution(
        batch1_nan, nan_policy='rank'
    )
    print("从批次1计算的参考分布:")
    print(ref_dist_nan)
    print()

    batch1_norm = quantile_normalize(
        batch1_nan, reference_dist=ref_dist_nan, nan_policy='rank'
    )
    batch2_norm = quantile_normalize(
        batch2_nan, reference_dist=ref_dist_nan, nan_policy='rank'
    )
    print("批次1归一化:")
    print(batch1_norm)
    print("批次2归一化:")
    print(batch2_norm)
    print()

    print("=" * 70)
    print("测试6: 并列值版本的参考分布和逆变换")
    print("=" * 70)
    data_tie = np.array([
        [5, 4, 3],
        [2, 1, 3],
        [3, 4, 6],
        [4, 2, 8],
    ])
    print("含并列值的数据:")
    print(data_tie)
    print()

    normalized_tie, ref_tie = quantile_normalize_mean_rank(
        data_tie, return_reference=True
    )
    print("归一化结果（并列值处理）:")
    print(normalized_tie)
    print("参考分布（归一化目标）:")
    print(ref_tie)
    print()

    original_quantiles_tie = compute_original_quantiles(data_tie)
    print("原始分位数分布（用于逆变换）:")
    print(original_quantiles_tie)
    print()

    denormed_tie = quantile_denormalize_with_reference(
        normalized_tie, original_quantiles_tie
    )
    print("逆变换还原:")
    print(denormed_tie)
    print("还原误差:")
    print(np.abs(data_tie - denormed_tie))
    print()

    print("=" * 70)
    print("测试7: 实际多批次校正流程")
    print("=" * 70)
    print("场景:")
    print("  - 批次1: 对照组（3个样本）")
    print("  - 批次2: 处理组（3个样本），测量平台不同，数值整体缩放")
    print("  - 目标: 校正批次效应，使两组数据可比")
    print()

    control_batch = np.array([
        [5.1, 4.9, 5.2],
        [2.0, 1.8, 2.1],
        [3.2, 3.0, 3.1],
        [4.1, 4.3, 3.9],
    ])
    treatment_batch = np.array([
        [10.2, 9.8, 10.4],
        [4.0, 3.6, 4.2],
        [6.4, 6.0, 6.2],
        [8.2, 8.6, 7.8],
    ])
    print("对照组（批次1）:")
    print(control_batch)
    print("处理组（批次2，数值约为2倍）:")
    print(treatment_batch)
    print()

    ref_dist = compute_reference_distribution(control_batch)
    print("从对照组计算的参考分布:")
    print(ref_dist)
    print()

    control_norm = quantile_normalize(
        control_batch, reference_dist=ref_dist
    )
    treatment_norm = quantile_normalize(
        treatment_batch, reference_dist=ref_dist
    )
    print("对照组归一化:")
    print(control_norm)
    print("处理组归一化（与对照组同分布）:")
    print(treatment_norm)
    print()
    print("验证归一化后两组分布一致（每列排序后）:")
    print("对照组排序:")
    print(np.sort(control_norm, axis=0))
    print("处理组排序:")
    print(np.sort(treatment_norm, axis=0))
    print()

    print("=" * 70)
    print("测试8: 逆变换的实际应用")
    print("=" * 70)
    print("场景: 在归一化空间进行统计分析后，将结果映射回原始尺度")
    print()

    treatment_effect_norm = treatment_norm - control_norm
    print("处理效应（归一化空间）:")
    print(treatment_effect_norm)
    print()

    control_orig_quantiles = compute_original_quantiles(control_batch)
    print("对照组原始分位数分布:")
    print(control_orig_quantiles)
    print()

    print("验证 quantile_denormalize 还原能力:")
    control_denormed = quantile_denormalize(control_norm, control_batch)
    print("对照组归一化后逆变换还原:")
    print(control_denormed)
    print("还原误差（应为0）:")
    print(np.abs(control_batch - control_denormed))
    print()

    treatment_denormed = quantile_denormalize(treatment_norm, control_batch)
    print("处理组归一化后逆变换到对照组原始尺度:")
    print(treatment_denormed)
    print()
    print("校正前处理组与对照组差异:")
    print(np.abs(treatment_batch - control_batch))
    print("校正后处理组与对照组差异:")
    print(np.abs(treatment_denormed - control_batch))
    print()

    print("=" * 70)
    print("测试9: 用户自定义目标分布")
    print("=" * 70)
    custom_ref = np.array([1.0, 3.0, 5.0, 7.0])
    print("用户自定义参考分布:", custom_ref)
    print()

    custom_normalized = quantile_normalize(
        control_batch, reference_dist=custom_ref
    )
    print("使用自定义参考分布归一化:")
    print(custom_normalized)
    print()
    print("验证排序后等于自定义分布:")
    print(np.sort(custom_normalized, axis=0))
    print()
