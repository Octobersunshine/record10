import warnings


def sma(series, window, fill_edge=True):
    """
    计算简单移动平均 (Simple Moving Average)
    
    参数:
        series (list): 输入时间序列
        window (int): 窗口大小，必须大于0
        fill_edge (bool): 是否前向填充边界值，默认True
    
    返回:
        list: 平滑后的序列
    """
    if window <= 0:
        raise ValueError("窗口大小必须大于0")
    if not series:
        return []
    
    n = len(series)
    
    if window > n:
        warnings.warn(f"窗口大小({window})超过序列长度({n})，已自动调整为序列长度{n}")
        window = n
    
    result = []
    
    for i in range(n):
        if i < window - 1:
            if fill_edge:
                result.append(series[0])
            else:
                result.append(None)
        else:
            window_sum = sum(series[i - window + 1:i + 1])
            result.append(window_sum / window)
    
    return result


def ewma(series, alpha=None, span=None, fill_edge=True):
    """
    计算指数加权移动平均 (Exponentially Weighted Moving Average)
    
    参数:
        series (list): 输入时间序列
        alpha (float): 平滑因子，0 < alpha <= 1，与span二选一
        span (int): 窗口跨度，alpha = 2/(span + 1)，与alpha二选一
        fill_edge (bool): 是否前向填充边界值，默认True
    
    返回:
        list: 平滑后的序列
    """
    if alpha is None and span is None:
        raise ValueError("必须指定alpha或span其中一个参数")
    if alpha is not None and span is not None:
        raise ValueError("不能同时指定alpha和span")
    if not series:
        return []
    
    n = len(series)
    
    if span is not None:
        if span <= 0:
            raise ValueError("span必须大于0")
        if span > n:
            warnings.warn(f"span({span})超过序列长度({n})，已自动调整为序列长度{n}")
            span = n
        alpha = 2 / (span + 1)
    
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha必须在(0, 1]范围内")
    
    result = [0.0] * n
    
    if fill_edge:
        result[0] = series[0]
    else:
        result[0] = series[0]
    
    for i in range(1, n):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    
    return result


def ewma_adjusted(series, alpha=None, span=None, fill_edge=True):
    """
    计算带偏差修正的指数加权移动平均
    
    参数:
        series (list): 输入时间序列
        alpha (float): 平滑因子，0 < alpha <= 1，与span二选一
        span (int): 窗口跨度，alpha = 2/(span + 1)，与alpha二选一
        fill_edge (bool): 是否前向填充边界值，默认True
    
    返回:
        list: 平滑后的序列
    """
    if alpha is None and span is None:
        raise ValueError("必须指定alpha或span其中一个参数")
    if alpha is not None and span is not None:
        raise ValueError("不能同时指定alpha和span")
    if not series:
        return []
    
    n = len(series)
    
    if span is not None:
        if span <= 0:
            raise ValueError("span必须大于0")
        if span > n:
            warnings.warn(f"span({span})超过序列长度({n})，已自动调整为序列长度{n}")
            span = n
        alpha = 2 / (span + 1)
    
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha必须在(0, 1]范围内")
    
    result = [0.0] * n
    bias_correction = [0.0] * n
    
    result[0] = alpha * series[0]
    bias_correction[0] = alpha
    
    for i in range(1, n):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
        bias_correction[i] = alpha + (1 - alpha) * bias_correction[i - 1]
    
    for i in range(n):
        result[i] = result[i] / bias_correction[i]
    
    return result


def _calculate_sse(actual, predicted):
    """
    计算误差平方和 (Sum of Squared Errors)
    
    参数:
        actual (list): 实际值序列
        predicted (list): 预测值序列
    
    返回:
        float: SSE值
    """
    return sum((a - p) ** 2 for a, p in zip(actual, predicted))


def _grid_search_holt(series):
    """
    网格搜索Holt线性趋势模型的最优参数(alpha, beta)
    
    参数:
        series (list): 输入时间序列
    
    返回:
        tuple: (最优alpha, 最优beta, 最小SSE)
    """
    best_alpha, best_beta = 0.3, 0.1
    best_sse = float('inf')
    
    alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    betas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for alpha in alphas:
        for beta in betas:
            try:
                smoothed, _, _ = holt_linear_trend(series, alpha=alpha, beta=beta, auto_params=False)
                sse = _calculate_sse(series, smoothed)
                if sse < best_sse:
                    best_sse = sse
                    best_alpha = alpha
                    best_beta = beta
            except:
                continue
    
    return best_alpha, best_beta, best_sse


def _grid_search_holt_winters(series, seasonal_period, seasonal='add'):
    """
    网格搜索Holt-Winters季节模型的最优参数(alpha, beta, gamma)
    
    参数:
        series (list): 输入时间序列
        seasonal_period (int): 季节周期长度
        seasonal (str): 'add' 或 'mul'
    
    返回:
        tuple: (最优alpha, 最优beta, 最优gamma, 最小SSE)
    """
    best_alpha, best_beta, best_gamma = 0.3, 0.1, 0.1
    best_sse = float('inf')
    
    alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    betas = [0.001, 0.01, 0.05, 0.1, 0.2]
    gammas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                try:
                    smoothed, _, _, _ = holt_winters(
                        series, seasonal_period=seasonal_period,
                        alpha=alpha, beta=beta, gamma=gamma,
                        seasonal=seasonal, auto_params=False
                    )
                    sse = _calculate_sse(series, smoothed)
                    if sse < best_sse:
                        best_sse = sse
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma
                except:
                    continue
    
    return best_alpha, best_beta, best_gamma, best_sse


def holt_linear_trend(series, alpha=None, beta=None, forecast_steps=0, auto_params=True):
    """
    Holt线性趋势模型（双指数平滑）
    
    参数:
        series (list): 输入时间序列
        alpha (float): 水平平滑因子，0 < alpha <= 1
        beta (float): 趋势平滑因子，0 < beta <= 1
        forecast_steps (int): 预测步数，默认0（只返回平滑值）
        auto_params (bool): 是否自动选择最优参数，默认True
    
    返回:
        tuple: (平滑序列, 预测序列, 参数字典)
    """
    if not series:
        return [], [], {}
    
    n = len(series)
    
    if auto_params:
        alpha, beta, sse = _grid_search_holt(series)
    else:
        if alpha is None or beta is None:
            raise ValueError("auto_params=False时必须指定alpha和beta")
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha必须在(0, 1]范围内")
        if beta <= 0 or beta > 1:
            raise ValueError("beta必须在(0, 1]范围内")
    
    if n < 2:
        raise ValueError("序列长度至少为2")
    
    l = [0.0] * n
    b = [0.0] * n
    smoothed = [0.0] * n
    
    l[0] = series[0]
    b[0] = series[1] - series[0]
    smoothed[0] = l[0]
    
    for t in range(1, n):
        l[t] = alpha * series[t] + (1 - alpha) * (l[t - 1] + b[t - 1])
        b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
        smoothed[t] = l[t - 1] + b[t - 1]
    
    forecast = []
    for h in range(1, forecast_steps + 1):
        forecast.append(l[-1] + h * b[-1])
    
    params = {'alpha': alpha, 'beta': beta}
    if auto_params:
        params['sse'] = sse
    
    return smoothed, forecast, params


def holt_winters(series, seasonal_period, alpha=None, beta=None, gamma=None,
                 forecast_steps=0, seasonal='add', auto_params=True):
    """
    Holt-Winters季节模型（三指数平滑）
    
    参数:
        series (list): 输入时间序列
        seasonal_period (int): 季节周期长度（如12表示月度数据的年度周期）
        alpha (float): 水平平滑因子，0 < alpha <= 1
        beta (float): 趋势平滑因子，0 < beta <= 1
        gamma (float): 季节平滑因子，0 < gamma <= 1
        forecast_steps (int): 预测步数，默认0（只返回平滑值）
        seasonal (str): 季节模型类型，'add'（加法）或 'mul'（乘法），默认'add'
        auto_params (bool): 是否自动选择最优参数，默认True
    
    返回:
        tuple: (平滑序列, 预测序列, 组件字典, 参数字典)
    """
    if not series:
        return [], [], {}, {}
    
    if seasonal not in ['add', 'mul']:
        raise ValueError("seasonal必须是'add'或'mul'")
    
    n = len(series)
    
    if n < 2 * seasonal_period:
        raise ValueError(f"序列长度({n})必须至少为2倍季节周期({2 * seasonal_period})")
    
    if auto_params:
        alpha, beta, gamma, sse = _grid_search_holt_winters(series, seasonal_period, seasonal)
    else:
        if alpha is None or beta is None or gamma is None:
            raise ValueError("auto_params=False时必须指定alpha、beta和gamma")
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha必须在(0, 1]范围内")
        if beta <= 0 or beta > 1:
            raise ValueError("beta必须在(0, 1]范围内")
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma必须在(0, 1]范围内")
    
    l = [0.0] * n
    b = [0.0] * n
    s = [0.0] * n
    smoothed = [0.0] * n
    
    season_averages = []
    for i in range(seasonal_period):
        season_data = [series[j] for j in range(i, n, seasonal_period)]
        season_averages.append(sum(season_data) / len(season_data))
    
    overall_avg = sum(season_averages) / seasonal_period
    
    for i in range(seasonal_period):
        if seasonal == 'add':
            s[i] = season_averages[i] - overall_avg
        else:
            s[i] = season_averages[i] / overall_avg if overall_avg != 0 else 1.0
    
    l[seasonal_period - 1] = overall_avg
    b[seasonal_period - 1] = (sum(series[seasonal_period:2 * seasonal_period]) - 
                              sum(series[:seasonal_period])) / (seasonal_period ** 2)
    
    for t in range(seasonal_period, n):
        prev_s = s[t - seasonal_period]
        if seasonal == 'add':
            l[t] = alpha * (series[t] - prev_s) + (1 - alpha) * (l[t - 1] + b[t - 1])
            b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
            s[t] = gamma * (series[t] - l[t]) + (1 - gamma) * prev_s
            smoothed[t] = l[t - 1] + b[t - 1] + prev_s
        else:
            l[t] = alpha * (series[t] / prev_s) + (1 - alpha) * (l[t - 1] + b[t - 1])
            b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
            s[t] = gamma * (series[t] / l[t]) + (1 - gamma) * prev_s
            smoothed[t] = (l[t - 1] + b[t - 1]) * prev_s
    
    for t in range(seasonal_period):
        if seasonal == 'add':
            smoothed[t] = l[seasonal_period - 1] + b[seasonal_period - 1] * (t - seasonal_period + 1) + s[t]
        else:
            base = l[seasonal_period - 1] + b[seasonal_period - 1] * (t - seasonal_period + 1)
            smoothed[t] = base * s[t]
    
    forecast = []
    for h in range(1, forecast_steps + 1):
        season_idx = (n - seasonal_period + (h - 1)) % seasonal_period
        if seasonal == 'add':
            forecast.append(l[-1] + h * b[-1] + s[season_idx])
        else:
            forecast.append((l[-1] + h * b[-1]) * s[season_idx])
    
    components = {
        'level': l,
        'trend': b,
        'seasonal': s
    }
    
    params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'seasonal': seasonal}
    if auto_params:
        params['sse'] = sse
    
    return smoothed, forecast, components, params


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("always")
    
    test_series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print("原始序列:", test_series)
    print("\nSMA (窗口=3):", sma(test_series, window=3))
    print("SMA (窗口=3, 不填充边界):", sma(test_series, window=3, fill_edge=False))
    print("\nEWMA (alpha=0.3):", [round(x, 4) for x in ewma(test_series, alpha=0.3)])
    print("EWMA (span=5):", [round(x, 4) for x in ewma(test_series, span=5)])
    print("\nEWMA (偏差修正, alpha=0.3):", [round(x, 4) for x in ewma_adjusted(test_series, alpha=0.3)])
    
    print("\n" + "="*60)
    print("测试窗口大小超过序列长度的情况")
    print("="*60)
    
    short_series = [1, 2, 3, 4, 5]
    print(f"\n短序列: {short_series}")
    print(f"SMA (窗口=10, 超过序列长度):", sma(short_series, window=10))
    print(f"EWMA (span=20, 超过序列长度):", [round(x, 4) for x in ewma(short_series, span=20)])
    print(f"EWMA (偏差修正, span=20):", [round(x, 4) for x in ewma_adjusted(short_series, span=20)])
    
    print("\n" + "="*60)
    print("Holt线性趋势（双指数平滑）测试")
    print("="*60)
    
    trend_series = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    print(f"\n线性趋势序列: {trend_series}")
    
    smoothed, forecast, params = holt_linear_trend(trend_series, forecast_steps=3)
    print(f"\nHolt自动参数: alpha={params['alpha']:.4f}, beta={params['beta']:.4f}, SSE={params['sse']:.4f}")
    print(f"平滑序列: {[round(x, 4) for x in smoothed]}")
    print(f"预测3步: {[round(x, 4) for x in forecast]}")
    
    smoothed2, forecast2, params2 = holt_linear_trend(trend_series, alpha=0.5, beta=0.1, forecast_steps=3, auto_params=False)
    print(f"\nHolt指定参数(alpha=0.5, beta=0.1):")
    print(f"平滑序列: {[round(x, 4) for x in smoothed2]}")
    print(f"预测3步: {[round(x, 4) for x in forecast2]}")
    
    print("\n" + "="*60)
    print("Holt-Winters季节模型（三指数平滑）测试")
    print("="*60)
    
    seasonal_series = [10, 12, 15, 11, 13, 16, 12, 14, 17, 13, 15, 18,
                       14, 16, 19, 15, 17, 20, 16, 18, 21, 17, 19, 22]
    print(f"\n季节性序列（周期=12）: {seasonal_series}")
    
    smoothed_add, forecast_add, components_add, params_add = holt_winters(
        seasonal_series, seasonal_period=12, forecast_steps=6, seasonal='add'
    )
    print(f"\nHolt-Winters加法模型自动参数:")
    print(f"  alpha={params_add['alpha']:.4f}, beta={params_add['beta']:.4f}, gamma={params_add['gamma']:.4f}")
    print(f"  SSE={params_add['sse']:.4f}")
    print(f"  平滑序列: {[round(x, 4) for x in smoothed_add]}")
    print(f"  预测6步: {[round(x, 4) for x in forecast_add]}")
    
    smoothed_mul, forecast_mul, components_mul, params_mul = holt_winters(
        seasonal_series, seasonal_period=12, forecast_steps=6, seasonal='mul'
    )
    print(f"\nHolt-Winters乘法模型自动参数:")
    print(f"  alpha={params_mul['alpha']:.4f}, beta={params_mul['beta']:.4f}, gamma={params_mul['gamma']:.4f}")
    print(f"  SSE={params_mul['sse']:.4f}")
    print(f"  平滑序列: {[round(x, 4) for x in smoothed_mul]}")
    print(f"  预测6步: {[round(x, 4) for x in forecast_mul]}")
    
    print(f"\n季节组件（加法模型）: {[round(x, 4) for x in components_add['seasonal'][:12]]}")
    print(f"趋势组件（加法模型）: {[round(x, 4) for x in components_add['trend'][:5]]}...")
