import warnings

import numpy as np
import pandas as pd


def classical_decomposition(series, period, model='additive'):
    """
    时间序列经典分解（加法/乘法模型）
    
    参数:
        series: 时间序列数据，可以是list、numpy数组或pandas Series
        period: 季节周期长度（如12表示月度数据，7表示周数据）
        model: 模型类型，'additive'（加法模型）或'multiplicative'（乘法模型）
    
    返回:
        dict: 包含'trend'（趋势）、'seasonal'（季节）、'residual'（残差）三个成分
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    if period < 2:
        raise ValueError("周期长度必须大于等于2")
    if n < 2 * period:
        warnings.warn(
            f"序列长度({n})不足2倍周期长度({2 * period})，"
            f"无法可靠估计季节成分，将执行退化分解（仅趋势+残差）",
            RuntimeWarning,
            stacklevel=2,
        )
        seasonal = np.zeros(n) if model == 'additive' else np.ones(n)
        trend = _calculate_trend(series, period)
        residual = _calculate_residual(series, trend, seasonal, model)

        if isinstance(series, pd.Series):
            trend = pd.Series(trend, index=series.index)
            seasonal = pd.Series(seasonal, index=series.index)
            residual = pd.Series(residual, index=series.index)

        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
        }

    trend = _calculate_trend(series, period)
    
    detrended = _detrend_series(series, trend, model)
    
    seasonal = _calculate_seasonal(detrended, period, n)
    
    residual = _calculate_residual(series, trend, seasonal, model)
    
    if isinstance(series, pd.Series):
        trend = pd.Series(trend, index=series.index)
        seasonal = pd.Series(seasonal, index=series.index)
        residual = pd.Series(residual, index=series.index)
    
    return {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    }


def _calculate_trend(series, period):
    """使用移动平均计算趋势成分"""
    n = len(series)
    trend = np.full(n, np.nan)
    
    if period % 2 == 0:
        half = period // 2
        window = period + 1
        weights = np.ones(window)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights = weights / period
        
        for i in range(half, n - half):
            trend[i] = np.sum(series[i - half:i + half + 1] * weights)
    else:
        half = period // 2
        
        for i in range(half, n - half):
            trend[i] = np.mean(series[i - half:i + half + 1])
    
    return trend


def _detrend_series(series, trend, model):
    """去除趋势成分"""
    if model == 'additive':
        return series - trend
    elif model == 'multiplicative':
        return series / trend
    else:
        raise ValueError("model必须是'additive'或'multiplicative'")


def _calculate_seasonal(detrended, period, n):
    """计算季节成分"""
    seasonal_indices = np.tile(np.arange(period), (n + period - 1) // period)[:n]
    seasonal = np.zeros(n)
    
    for i in range(period):
        mask = seasonal_indices == i
        valid_mask = mask & ~np.isnan(detrended)
        if np.sum(valid_mask) > 0:
            seasonal[mask] = np.nanmean(detrended[valid_mask])
        else:
            seasonal[mask] = np.nan
    
    return seasonal


def _calculate_residual(series, trend, seasonal, model):
    """计算残差成分"""
    if model == 'additive':
        residual = series - trend - seasonal
    else:
        residual = series / (trend * seasonal)
    
    residual[np.isnan(trend)] = np.nan
    return residual


def _next_odd(n):
    n_int = int(np.ceil(n))
    return n_int if n_int % 2 == 1 else n_int + 1


def _tricube(u):
    result = np.zeros_like(u, dtype=np.float64)
    mask = np.abs(u) < 1.0
    result[mask] = (1.0 - np.abs(u[mask]) ** 3) ** 3
    return result


def _loess_smooth(x, y, x_eval, window, degree=1, weights=None):
    n = len(x)
    n_eval = len(x_eval)
    result = np.full(n_eval, np.nan)

    if window < 1:
        window = 1
    q = min(int(window), n)

    for i in range(n_eval):
        dist = np.abs(x - x_eval[i])
        sorted_idx = np.argsort(dist)
        q_actual = min(q, n)
        neighbor_idx = sorted_idx[:q_actual]

        if q_actual < degree + 1:
            continue

        max_dist = dist[neighbor_idx[-1]]
        if max_dist == 0:
            max_dist = 1.0

        u = dist[neighbor_idx] / max_dist
        w = _tricube(u)

        if weights is not None:
            w = w * weights[neighbor_idx]

        x_local = x[neighbor_idx]
        y_local = y[neighbor_idx]

        X = np.column_stack(
            [x_local ** p for p in range(degree + 1)]
        )
        W = np.diag(w)

        try:
            XWX = X.T @ W @ X
            XWy = X.T @ W @ y_local
            if np.linalg.cond(XWX) > 1e12:
                beta = np.linalg.lstsq(XWX, XWy, rcond=None)[0]
            else:
                beta = np.linalg.solve(XWX, XWy)
            result[i] = np.sum(
                [beta[p] * x_eval[i] ** p for p in range(degree + 1)]
            )
        except np.linalg.LinAlgError:
            result[i] = np.nanmean(y_local)

    return result


def _moving_average(x, q):
    n = len(x)
    result = np.full(n, np.nan)
    half = q // 2
    for i in range(half, n - half):
        result[i] = np.mean(x[i - half : i + half + 1])
    return result


def _biweight_weights(residuals):
    median_r = np.nanmedian(np.abs(residuals))
    if median_r == 0:
        return np.ones(len(residuals))
    h = 6.0 * median_r
    u = residuals / h
    w = (1.0 - u ** 2) ** 2
    w[np.abs(u) >= 1.0] = 0.0
    return w


def _auto_stl_parameters(n, period):
    seasonal_window = _next_odd(period)
    trend_window = _next_odd(
        np.ceil((1.5 * period) / (1.0 - 1.5 / seasonal_window))
    )
    low_pass_window = _next_odd(period)
    return seasonal_window, trend_window, low_pass_window


def stl_decomposition(
    series,
    period,
    seasonal_window=None,
    trend_window=None,
    low_pass_window=None,
    degree=1,
    inner_iter=2,
    outer_iter=15,
    robust=True,
):
    """
    STL分解（基于LOESS的季节-趋势分解）
    
    STL使用局部加权回归(LOESS)进行平滑，相比经典分解具有更强的鲁棒性，
    能更好地处理异常值和非固定季节模式。
    
    参数:
        series: 时间序列数据，可以是list、numpy数组或pandas Series
        period: 季节周期长度（如12表示月度数据，7表示周数据）
        seasonal_window: 季节子序列LOESS平滑窗口（奇数，>=3），
                         None则自动选择（默认为next_odd(period)）
        trend_window: 趋势LOESS平滑窗口（奇数），
                      None则自动选择（Cleveland推荐公式）
        low_pass_window: 低通滤波LOESS平滑窗口（奇数），
                         None则自动选择（默认为next_odd(period)）
        degree: LOESS局部回归阶数（0或1，默认1）
        inner_iter: 内循环迭代次数（默认2）
        outer_iter: 外循环（鲁棒）迭代次数（默认15）
        robust: 是否启用鲁棒性迭代（默认True）
    
    返回:
        dict: 包含以下键:
            - 'trend': 趋势成分
            - 'seasonal': 季节成分
            - 'residual': 残差成分
            - 'weights': 最终鲁棒权重（仅robust=True时）
            - 'parameters': 使用的平滑参数
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)

    if period < 2:
        raise ValueError("period must be >= 2")
    if n < 2 * period:
        warnings.warn(
            f"STL: n={n} < 2*period={2*period}, "
            f"decomposition may be unreliable",
            RuntimeWarning,
            stacklevel=2,
        )

    if seasonal_window is None or trend_window is None or low_pass_window is None:
        auto_sw, auto_tw, auto_lpw = _auto_stl_parameters(n, period)
        if seasonal_window is None:
            seasonal_window = auto_sw
        if trend_window is None:
            trend_window = auto_tw
        if low_pass_window is None:
            low_pass_window = auto_lpw

    if seasonal_window < 3 or seasonal_window % 2 == 0:
        raise ValueError(
            f"seasonal_window must be odd and >= 3, got {seasonal_window}"
        )
    if trend_window % 2 == 0:
        raise ValueError(
            f"trend_window must be odd, got {trend_window}"
        )
    if low_pass_window % 2 == 0:
        raise ValueError(
            f"low_pass_window must be odd, got {low_pass_window}"
        )

    x = np.arange(n, dtype=np.float64)
    trend = np.zeros(n)
    seasonal = np.zeros(n)
    robust_weights = np.ones(n)

    actual_outer = outer_iter if robust else 1

    for outer in range(actual_outer):
        for inner in range(inner_iter):
            detrended = series - trend

            cycle_smooth = np.full(n, np.nan)
            for s in range(period):
                idx = np.arange(s, n, period)
                if len(idx) < degree + 2:
                    cycle_smooth[idx] = detrended[idx]
                    continue

                x_sub = x[idx].copy()
                y_sub = detrended[idx].copy()

                sub_weights = robust_weights[idx].copy()

                smoothed = _loess_smooth(
                    x_sub, y_sub, x_sub,
                    window=seasonal_window,
                    degree=degree,
                    weights=sub_weights,
                )
                cycle_smooth[idx] = smoothed

            valid = ~np.isnan(cycle_smooth)
            if np.sum(valid) < 3:
                seasonal = np.zeros(n)
            else:
                ma1 = _moving_average(cycle_smooth, period)
                ma2 = _moving_average(
                    np.where(np.isnan(ma1), 0, ma1), period
                )
                ma3 = _moving_average(
                    np.where(np.isnan(ma2), 0, ma2), 3
                )

                loess_input = np.where(np.isnan(ma3), 0, ma3)
                lp_x = np.arange(n, dtype=np.float64)
                low_pass = _loess_smooth(
                    lp_x, loess_input, lp_x,
                    window=low_pass_window,
                    degree=degree,
                    weights=robust_weights,
                )
                low_pass = np.where(np.isnan(ma3), np.nan, low_pass)

                seasonal = cycle_smooth - np.nan_to_num(low_pass, nan=0.0)

            deseasonalized = series - seasonal

            trend = _loess_smooth(
                x, deseasonalized, x,
                window=trend_window,
                degree=degree,
                weights=robust_weights,
            )
            nan_trend = np.isnan(trend)
            if np.any(nan_trend):
                trend[nan_trend] = np.interp(
                    x[nan_trend], x[~nan_trend], trend[~nan_trend]
                )

        if robust:
            residual = series - seasonal - trend
            robust_weights = _biweight_weights(residual)

    residual = series - seasonal - trend

    parameters = {
        'period': period,
        'seasonal_window': seasonal_window,
        'trend_window': trend_window,
        'low_pass_window': low_pass_window,
        'degree': degree,
        'inner_iter': inner_iter,
        'outer_iter': actual_outer,
        'robust': robust,
    }

    if isinstance(series, pd.Series):
        trend = pd.Series(trend, index=series.index)
        seasonal = pd.Series(seasonal, index=series.index)
        residual = pd.Series(residual, index=series.index)

    result = {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual,
        'parameters': parameters,
    }
    if robust:
        result['weights'] = robust_weights

    return result


def additive_decomposition(series, period):
    """加法模型分解：Y = T + S + R"""
    return classical_decomposition(series, period, model='additive')


def multiplicative_decomposition(series, period):
    """乘法模型分解：Y = T * S * R"""
    return classical_decomposition(series, period, model='multiplicative')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n = 72
    period = 12

    t = np.arange(n, dtype=float)
    trend_true = 100 + 2 * t
    seasonal_true = 10 * np.sin(2 * np.pi * t / period)
    noise = np.random.normal(0, 1.5, n)

    additive_series = trend_true + seasonal_true + noise

    outlier_idx = [20, 45, 60]
    additive_series_outlier = additive_series.copy()
    additive_series_outlier[outlier_idx] += 30

    print("=" * 60)
    print("STL Decomposition Demo")
    print("=" * 60)

    stl_result = stl_decomposition(additive_series, period)
    params = stl_result['parameters']
    print(f"Auto-selected parameters:")
    print(f"  seasonal_window = {params['seasonal_window']}")
    print(f"  trend_window    = {params['trend_window']}")
    print(f"  low_pass_window = {params['low_pass_window']}")
    print(f"  degree          = {params['degree']}")
    print(f"  robust          = {params['robust']}")
    print(f"  outer_iter      = {params['outer_iter']}")

    residual = stl_result['residual']
    valid_r = residual[~np.isnan(residual)]
    print(f"\nResidual std: {np.std(valid_r):.4f}")
    print(f"Residual mean: {np.mean(valid_r):.4f}")

    print("\n" + "=" * 60)
    print("STL with Outliers (Robustness Test)")
    print("=" * 60)

    stl_robust = stl_decomposition(additive_series_outlier, period, robust=True)
    stl_nonrobust = stl_decomposition(additive_series_outlier, period, robust=False)

    print(f"Robust weights at outlier positions:")
    for idx in outlier_idx:
        print(f"  position {idx}: weight = {stl_robust['weights'][idx]:.4f}")

    robust_r = stl_robust['residual']
    nonrobust_r = stl_nonrobust['residual']
    print(f"\nRobust residual std:    {np.nanstd(robust_r):.4f}")
    print(f"Non-robust residual std: {np.nanstd(nonrobust_r):.4f}")

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(additive_series_outlier, 'b-o', markersize=2, label='Original')
    axes[0].plot(outlier_idx, additive_series_outlier[outlier_idx],
                 'rv', markersize=8, label='Outliers')
    axes[0].set_title('Original Series with Outliers')
    axes[0].legend()

    axes[1].plot(stl_robust['trend'], 'r-', linewidth=2, label='Robust STL')
    axes[1].plot(stl_nonrobust['trend'], 'r--', linewidth=1, alpha=0.6,
                 label='Non-robust STL')
    axes[1].set_title('Trend Component')
    axes[1].legend()

    axes[2].plot(stl_robust['seasonal'], 'g-', linewidth=2, label='Robust STL')
    axes[2].plot(stl_nonrobust['seasonal'], 'g--', linewidth=1, alpha=0.6,
                 label='Non-robust STL')
    axes[2].set_title('Seasonal Component')
    axes[2].legend()

    axes[3].plot(stl_robust['residual'], 'k-', linewidth=1, label='Robust')
    axes[3].plot(stl_nonrobust['residual'], 'm--', linewidth=1, alpha=0.6,
                 label='Non-robust')
    axes[3].plot(outlier_idx, stl_robust['residual'][outlier_idx],
                 'rv', markersize=8)
    axes[3].set_title('Residual Component')
    axes[3].legend()

    axes[4].bar(range(n), stl_robust['weights'], color='orange', alpha=0.7)
    axes[4].set_title('Robust Weights')
    axes[4].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig('stl_decomposition_example.png', dpi=100)
    print(f"\nVisualization saved to stl_decomposition_example.png")

    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes2[0].plot(additive_series, 'b-o', markersize=2, label='Original')
    axes2[0].set_title('Original Series (No Outliers)')
    axes2[0].legend()

    axes2[1].plot(stl_result['trend'], 'r-', linewidth=2, label='STL Trend')
    axes2[1].plot(trend_true, 'k--', alpha=0.4, label='True Trend')
    axes2[1].set_title('Trend vs True')
    axes2[1].legend()

    axes2[2].plot(stl_result['seasonal'], 'g-', linewidth=2, label='STL Seasonal')
    axes2[2].plot(seasonal_true, 'k--', alpha=0.4, label='True Seasonal')
    axes2[2].set_title('Seasonal vs True')
    axes2[2].legend()

    axes2[3].plot(stl_result['residual'], 'k-', linewidth=1, label='Residual')
    axes2[3].set_title('Residual')
    axes2[3].set_xlabel('Time')
    axes2[3].legend()

    plt.tight_layout()
    plt.savefig('stl_accuracy_example.png', dpi=100)
    print(f"Accuracy visualization saved to stl_accuracy_example.png")
