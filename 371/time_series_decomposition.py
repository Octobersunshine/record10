import numpy as np
from collections import namedtuple


DecompositionResult = namedtuple('DecompositionResult', [
    'trend', 'seasonal', 'residual', 'period', 'method',
    'model_type', 'residual_stats', 'plot_data'
])


ResidualStats = namedtuple('ResidualStats', [
    'mean', 'std', 'min', 'max', 'median',
    'q25', 'q75', 'skewness', 'kurtosis',
    'acf_lag1', 'ljung_box_pvalue'
])


def _fill_missing_values(ts, method='linear'):
    """
    缺失值填充

    参数:
        ts: 时间序列，可能包含NaN
        method: 'linear' 线性插值, 'forward' 前向填充, 'backward' 后向填充

    返回:
        填充后的时间序列
    """
    ts = np.asarray(ts, dtype=np.float64).copy()
    n = len(ts)

    nan_mask = np.isnan(ts)
    if not nan_mask.any():
        return ts

    if method == 'linear':
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) < 2:
            raise ValueError("至少需要2个有效值进行线性插值")
        ts[nan_mask] = np.interp(
            np.where(nan_mask)[0],
            valid_idx,
            ts[valid_idx]
        )
    elif method == 'forward':
        last_valid = None
        for i in range(n):
            if not nan_mask[i]:
                last_valid = ts[i]
            elif last_valid is not None:
                ts[i] = last_valid
    elif method == 'backward':
        next_valid = None
        for i in range(n - 1, -1, -1):
            if not nan_mask[i]:
                next_valid = ts[i]
            elif next_valid is not None:
                ts[i] = next_valid
    else:
        raise ValueError(f"未知的填充方法: {method}")

    return ts


def _compute_acf(ts, max_lag):
    """计算自相关函数"""
    ts = np.asarray(ts)
    n = len(ts)
    ts_centered = ts - np.mean(ts)
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        acf[lag] = np.sum(ts_centered[:n - lag] * ts_centered[lag:]) / np.sum(ts_centered ** 2)

    return acf


def _detect_seasonal_period(ts, min_period=2, max_period=None, threshold=0.1):
    """通过ACF峰值检测季节周期"""
    ts = np.asarray(ts)
    n = len(ts)

    if max_period is None:
        max_period = n // 2

    if max_period < min_period or max_period < 2:
        return None

    acf = _compute_acf(ts, max_period)
    acf_search = acf[min_period:max_period + 1]

    if len(acf_search) == 0:
        return None

    max_acf = np.max(acf_search)
    if max_acf < threshold:
        return None

    peaks = []
    for i in range(1, len(acf_search) - 1):
        if acf_search[i] > acf_search[i - 1] and acf_search[i] > acf_search[i + 1]:
            if acf_search[i] > threshold * max_acf:
                peaks.append((min_period + i, acf_search[i]))

    if not peaks:
        best_lag = min_period + np.argmax(acf_search)
        return best_lag

    peaks.sort(key=lambda x: -x[1])
    return peaks[0][0]


def _loess_smooth(x, y, span=0.75, degree=2):
    """
    局部加权散点平滑 (LOESS)

    参数:
        x: 自变量
        y: 因变量
        span: 平滑窗口比例 (0, 1]
        degree: 多项式阶数，1或2
    """
    n = len(x)
    k = int(np.ceil(span * n))
    smoothed = np.zeros(n)

    for i in range(n):
        distances = np.abs(x - x[i])
        nearest_idx = np.argsort(distances)[:k]
        max_dist = distances[nearest_idx[-1]]

        weights = np.clip(1 - (distances[nearest_idx] / max_dist) ** 3, 0, 1) ** 3
        weights = weights / weights.sum()

        X = np.vander(x[nearest_idx], degree + 1)
        W = np.diag(weights)

        XtW = X.T @ W
        beta = np.linalg.solve(XtW @ X, XtW @ y[nearest_idx])
        smoothed[i] = np.polyval(beta, x[i])

    return smoothed


def _compute_residual_stats(residual):
    """
    计算残差统计量
    """
    valid_resid = residual[~np.isnan(residual)]
    n = len(valid_resid)

    if n < 4:
        return ResidualStats(*[np.nan] * 10)

    mean = np.mean(valid_resid)
    std = np.std(valid_resid, ddof=1)
    min_val = np.min(valid_resid)
    max_val = np.max(valid_resid)
    median = np.median(valid_resid)
    q25, q75 = np.percentile(valid_resid, [25, 75])

    skewness = np.sum((valid_resid - mean) ** 3) / (n * std ** 3) if std > 0 else np.nan
    kurtosis = np.sum((valid_resid - mean) ** 4) / (n * std ** 4) - 3 if std > 0 else np.nan

    acf_vals = _compute_acf(valid_resid, min(10, n - 1))
    acf_lag1 = acf_vals[1] if len(acf_vals) > 1 else np.nan

    # Ljung-Box检验近似
    lb_stat = n * (n + 2) * np.sum(acf_vals[1:min(6, len(acf_vals))] ** 2 / (n - np.arange(1, min(6, len(acf_vals)))))
    from scipy.special import gammaincc
    try:
        ljung_box_pvalue = gammaincc(5 / 2, lb_stat / 2)
    except:
        ljung_box_pvalue = np.nan

    return ResidualStats(
        mean=mean, std=std, min=min_val, max=max_val,
        median=median, q25=q25, q75=q75,
        skewness=skewness, kurtosis=kurtosis,
        acf_lag1=acf_lag1, ljung_box_pvalue=ljung_box_pvalue
    )


def _prepare_plot_data(ts, trend, seasonal, residual, period):
    """
    准备可视化数据
    """
    n = len(ts)
    x = np.arange(n)

    # 季节模式数据 (一个周期的平均)
    season_pattern = np.zeros(period)
    for i in range(period):
        season_pattern[i] = np.nanmean(seasonal[i::period])

    # 残差的ACF
    valid_resid = residual[~np.isnan(residual)]
    acf_max_lag = min(20, len(valid_resid) // 2)
    residual_acf = _compute_acf(valid_resid, acf_max_lag) if acf_max_lag > 0 else np.array([])

    return {
        'original': {'x': x, 'y': ts},
        'trend': {'x': x, 'y': trend},
        'seasonal': {'x': x, 'y': seasonal},
        'residual': {'x': x, 'y': residual},
        'seasonal_pattern': {'x': np.arange(period), 'y': season_pattern},
        'residual_acf': {'x': np.arange(len(residual_acf)), 'y': residual_acf},
    }


def _moving_average_trend(ts, period):
    """移动平均法估计趋势"""
    n = len(ts)

    if period % 2 == 0:
        ma1 = np.convolve(ts, np.ones(period) / period, mode='valid')
        trend = np.convolve(ma1, np.ones(2) / 2, mode='valid')
        pad_start = period // 2
        pad_end = n - len(trend) - pad_start
    else:
        trend = np.convolve(ts, np.ones(period) / period, mode='valid')
        pad_start = (period - 1) // 2
        pad_end = n - len(trend) - pad_start

    trend = np.pad(trend, (pad_start, pad_end),
                   mode='constant', constant_values=np.nan)
    return trend


def classical_decomposition(ts, period=None, model='additive',
                            fill_missing='linear', min_period=2,
                            max_period=None, acf_threshold=0.1):
    """
    经典时间序列分解（移动平均法）

    参数:
        ts: 时间序列
        period: 季节周期，None则自动检测
        model: 'additive' 加法模型 或 'multiplicative' 乘法模型
        fill_missing: 缺失值填充方法
        ...

    返回:
        DecompositionResult
    """
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts)

    if np.isnan(ts).any():
        ts = _fill_missing_values(ts, method=fill_missing)

    if period is None:
        period = _detect_seasonal_period(ts, min_period, max_period, acf_threshold)
        if period is None:
            raise ValueError("未能自动检测到显著的季节周期")

    period = int(period)
    if period < 2 or period > n // 2:
        raise ValueError("period 必须满足 2 <= period <= n//2")

    if model == 'additive':
        trend = _moving_average_trend(ts, period)
        detrended = ts - trend
    elif model == 'multiplicative':
        if (ts <= 0).any():
            raise ValueError("乘法模型要求所有观测值为正数")
        trend = _moving_average_trend(ts, period)
        detrended = ts / trend
    else:
        raise ValueError(f"未知模型类型: {model}")

    seasonal = np.full(n, np.nan)
    for i in range(period):
        idx = np.arange(i, n, period)
        valid_mask = ~np.isnan(detrended[idx])
        if valid_mask.any():
            seasonal[idx] = np.mean(detrended[idx][valid_mask])

    seasonal_mean = np.nanmean(seasonal)
    if model == 'additive':
        seasonal = seasonal - seasonal_mean
        residual = ts - trend - seasonal
    else:
        seasonal = seasonal / seasonal_mean
        residual = ts / (trend * seasonal)

    residual_stats = _compute_residual_stats(residual)
    plot_data = _prepare_plot_data(ts, trend, seasonal, residual, period)

    return DecompositionResult(
        trend=trend, seasonal=seasonal, residual=residual,
        period=period, method='classical', model_type=model,
        residual_stats=residual_stats, plot_data=plot_data
    )


def stl_decomposition(ts, period=None, seasonal=7, trend=None,
                      low_pass=None, robust=True, model='additive',
                      fill_missing='linear', min_period=2,
                      max_period=None, acf_threshold=0.1):
    """
    STL时间序列分解（基于LOESS）

    参数:
        ts: 时间序列
        period: 季节周期
        seasonal: 季节LOESS窗口
        trend: 趋势LOESS窗口
        low_pass: 低通LOESS窗口
        robust: 是否使用鲁棒迭代
        model: 'additive' 或 'multiplicative'
    """
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts)
    x = np.arange(n)

    if np.isnan(ts).any():
        ts = _fill_missing_values(ts, method=fill_missing)

    if period is None:
        period = _detect_seasonal_period(ts, min_period, max_period, acf_threshold)
        if period is None:
            raise ValueError("未能自动检测到显著的季节周期")

    period = int(period)

    if trend is None:
        trend = max(int(np.ceil(1.5 * period / (1 - 1.5 / seasonal))), 2 * period + 1)
    if low_pass is None:
        low_pass = max(period + (period % 2 == 0), 3)

    work_data = ts.copy()
    seasonal_comp = np.zeros(n)

    n_iter = 5 if not robust else 15
    robustness_weights = np.ones(n)

    for outer_iter in range(n_iter if robust else 1):
        for inner_iter in range(2):
            # 1. 去季节
            deseasonalized = work_data - seasonal_comp

            # 2. 趋势平滑
            trend_comp = _loess_smooth(x, deseasonalized, span=trend / n, degree=1)

            # 3. 去趋势
            detrended = work_data - trend_comp

            # 4. 季节子序列平滑
            period_ts = np.zeros((period, (n + period - 1) // period + 2))
            for i in range(period):
                idx = np.arange(i, n, period)
                vals = detrended[idx]
                padded = np.pad(vals, (1, 1), mode='edge')
                period_ts[i, :len(padded)] = padded

            smoothed_seasonal = np.zeros(n)
            for i in range(period):
                idx = np.arange(i, n, period)
                if len(idx) >= 2:
                    smoothed = _loess_smooth(
                        np.arange(len(period_ts[i])),
                        period_ts[i],
                        span=seasonal / len(period_ts[i]),
                        degree=1
                    )
                    smoothed_seasonal[idx] = smoothed[1:1 + len(idx)]

            # 5. 低通滤波
            seasonal_smoothed = _loess_smooth(
                x, smoothed_seasonal,
                span=low_pass / n, degree=1
            )

            seasonal_comp = smoothed_seasonal - seasonal_smoothed

        if robust:
            residual = work_data - trend_comp - seasonal_comp
            mad = np.median(np.abs(residual - np.median(residual)))
            if mad > 0:
                robustness_weights = np.clip(np.abs(residual) / (6 * mad), 0, 1)
                robustness_weights = (1 - robustness_weights ** 2) ** 2

    trend_comp = _loess_smooth(x, ts - seasonal_comp, span=trend / n, degree=1)
    residual = ts - trend_comp - seasonal_comp

    residual_stats = _compute_residual_stats(residual)
    plot_data = _prepare_plot_data(ts, trend_comp, seasonal_comp, residual, period)

    return DecompositionResult(
        trend=trend_comp, seasonal=seasonal_comp, residual=residual,
        period=period, method='stl', model_type=model,
        residual_stats=residual_stats, plot_data=plot_data
    )


if __name__ == "__main__":
    np.random.seed(42)
    n = 120
    period = 12
    t = np.arange(n)

    print("=" * 60)
    print("测试1: 经典加法分解")
    print("=" * 60)
    trend_true = 0.1 * t + 10
    seasonal_true = 3 * np.sin(2 * np.pi * t / period)
    residual_true = np.random.normal(0, 0.5, n)
    ts = trend_true + seasonal_true + residual_true

    result = classical_decomposition(ts, model='additive')
    print(f"检测周期: {result.period}")
    print(f"残差均值: {result.residual_stats.mean:.4f}")
    print(f"残差标准差: {result.residual_stats.std:.4f}")
    print(f"残差ACF(lag1): {result.residual_stats.acf_lag1:.4f}")

    print("\n" + "=" * 60)
    print("测试2: 经典乘法分解")
    print("=" * 60)
    trend_true_m = 50 + 0.5 * t
    seasonal_true_m = 1 + 0.2 * np.sin(2 * np.pi * t / period)
    ts_m = trend_true_m * seasonal_true_m * (1 + np.random.normal(0, 0.05, n))

    result_m = classical_decomposition(ts_m, model='multiplicative')
    print(f"检测周期: {result_m.period}")
    print(f"季节成分均值: {np.mean(result_m.seasonal):.4f}")

    print("\n" + "=" * 60)
    print("测试3: STL分解 (鲁棒版本)")
    print("=" * 60)
    result_stl = stl_decomposition(ts, robust=True)
    print(f"检测周期: {result_stl.period}")
    print(f"残差均值: {result_stl.residual_stats.mean:.4f}")
    print(f"残差标准差: {result_stl.residual_stats.std:.4f}")

    print("\n" + "=" * 60)
    print("测试4: 含缺失值的序列")
    print("=" * 60)
    ts_missing = ts.copy()
    ts_missing[[10, 25, 40, 55, 70]] = np.nan
    result_missing = classical_decomposition(ts_missing, fill_missing='linear')
    print(f"缺失值数量: {np.sum(np.isnan(ts_missing))}")
    print(f"检测周期: {result_missing.period}")

    print("\n" + "=" * 60)
    print("可视化数据字段:", list(result.plot_data.keys()))
    print("=" * 60)
