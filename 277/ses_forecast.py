import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Optional, Union, Dict


def _acf(y: np.ndarray, maxlags: int) -> np.ndarray:
    n = len(y)
    y_demean = y - np.mean(y)
    acf_vals = np.empty(maxlags + 1)
    acf_vals[0] = 1.0
    for k in range(1, maxlags + 1):
        acf_vals[k] = np.sum(y_demean[k:] * y_demean[:-k]) / np.sum(y_demean ** 2)
    return acf_vals


def detect_seasonal_period(
    data: np.ndarray,
    min_period: int = 2,
    max_period: Optional[int] = None,
    threshold: float = 0.2,
) -> Optional[int]:
    y = np.asarray(data, dtype=float)
    n = len(y)
    if n < 4:
        return None

    if max_period is None:
        max_period = min(n // 2, 24)
    max_period = min(max_period, n // 2)

    if max_period < min_period:
        return None

    acf_vals = _acf(y, max_period)

    candidates = []
    for period in range(min_period, max_period + 1):
        if acf_vals[period] > threshold:
            is_peak = True
            for offset in range(1, min(period, max_period - period + 1)):
                if period - offset >= min_period and acf_vals[period - offset] >= acf_vals[period]:
                    is_peak = False
                    break
                if period + offset <= max_period and acf_vals[period + offset] >= acf_vals[period]:
                    is_peak = False
                    break
            if is_peak:
                candidates.append((period, acf_vals[period]))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def _fit_ses(y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, float]:
    n = len(y)
    l = np.empty(n)
    l[0] = y[0]
    for t in range(1, n):
        l[t] = alpha * y[t] + (1 - alpha) * l[t - 1]
    fitted = np.empty(n)
    fitted[0] = y[0]
    fitted[1:] = l[:-1]
    residuals = y - fitted
    sse = np.sum(residuals ** 2)
    return l, residuals, sse


def _ses_objective(params: np.ndarray, y: np.ndarray) -> float:
    _, _, sse = _fit_ses(y, params[0])
    return sse


def _fit_holt(y: np.ndarray, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(y)
    l = np.empty(n)
    b = np.empty(n)
    l[0] = y[0]
    b[0] = y[1] - y[0] if n >= 2 else 0.0
    for t in range(1, n):
        l[t] = alpha * y[t] + (1 - alpha) * (l[t - 1] + b[t - 1])
        b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
    fitted = np.empty(n)
    fitted[0] = y[0]
    for t in range(1, n):
        fitted[t] = l[t - 1] + b[t - 1]
    residuals = y - fitted
    sse = np.sum(residuals ** 2)
    return l, b, residuals, sse


def _holt_objective(params: np.ndarray, y: np.ndarray) -> float:
    _, _, _, sse = _fit_holt(y, params[0], params[1])
    return sse


def _fit_holt_winters(
    y: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    season_length: int,
    seasonal: str = "add",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(y)
    m = season_length

    if seasonal == "add":
        l = np.empty(n)
        b = np.empty(n)
        s = np.empty(n)
        initial_season = np.array([y[i] - np.mean(y[:m]) for i in range(m)])
        s[:m] = initial_season
        l[0] = y[0] - s[0]
        b[0] = (y[m] - y[0]) / m if n > m else (y[1] - y[0])
        for t in range(1, n):
            if t < m:
                l[t] = alpha * (y[t] - s[t]) + (1 - alpha) * (l[t - 1] + b[t - 1])
                b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
            else:
                l[t] = alpha * (y[t] - s[t - m]) + (1 - alpha) * (l[t - 1] + b[t - 1])
                b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
                s[t] = gamma * (y[t] - l[t]) + (1 - gamma) * s[t - m]
        fitted = np.empty(n)
        fitted[0] = y[0]
        for t in range(1, n):
            if t < m:
                fitted[t] = l[t - 1] + b[t - 1] + s[t]
            else:
                fitted[t] = l[t - 1] + b[t - 1] + s[t - m]
        residuals = y - fitted
        sse = np.sum(residuals ** 2)
        return l, b, s, residuals, sse
    else:
        l = np.empty(n)
        b = np.empty(n)
        s = np.empty(n)
        initial_level = np.mean(y[:m])
        initial_season = np.array([y[i] / initial_level for i in range(m)])
        s[:m] = initial_season
        l[0] = y[0] / s[0]
        b[0] = (y[m] / s[0] - y[0] / s[0]) / m if n > m else (y[1] / s[1] - y[0] / s[0])
        for t in range(1, n):
            if t < m:
                l[t] = alpha * (y[t] / s[t]) + (1 - alpha) * (l[t - 1] + b[t - 1])
                b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
            else:
                l[t] = alpha * (y[t] / s[t - m]) + (1 - alpha) * (l[t - 1] + b[t - 1])
                b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]
                s[t] = gamma * (y[t] / l[t]) + (1 - gamma) * s[t - m]
        fitted = np.empty(n)
        fitted[0] = y[0]
        for t in range(1, n):
            if t < m:
                fitted[t] = (l[t - 1] + b[t - 1]) * s[t]
            else:
                fitted[t] = (l[t - 1] + b[t - 1]) * s[t - m]
        residuals = y - fitted
        sse = np.sum(residuals ** 2)
        return l, b, s, residuals, sse


def _holt_winters_objective(params: np.ndarray, y: np.ndarray, season_length: int, seasonal: str) -> float:
    _, _, _, _, sse = _fit_holt_winters(y, params[0], params[1], params[2], season_length, seasonal)
    return sse


ModelResult = Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], float]


def exponential_smoothing(
    data: np.ndarray,
    model: str = "auto",
    seasonal: Optional[str] = None,
    season_length: Optional[int] = None,
    h: int = 1,
    confidence: float = 0.95,
    params: Optional[Dict[str, float]] = None,
) -> ModelResult:
    """
    指数平滑统一接口：SES、Holt线性趋势、Holt-Winters季节性模型。

    参数
    ----------
    data : np.ndarray
        一维历史时间序列数据。
    model : str
        模型类型：'ses', 'holt', 'holt-winters', 'auto'（自动检测选择）。
    seasonal : str, optional
        季节类型：'add'（加法）、'mul'（乘法），仅用于 holt-winters。
    season_length : int, optional
        季节周期长度。若为 None 且需要则自动检测。
    h : int
        预测步数。
    confidence : float
        预测区间置信水平。
    params : dict, optional
        手动指定参数（如 {'alpha': 0.3, 'beta': 0.1}），None 则自动优化。

    返回
    -------
    forecast : np.ndarray
        预测值。
    lower : np.ndarray
        预测区间下界。
    upper : np.ndarray
        预测区间上界。
    opt_params : dict
        使用的参数。
    sse : float
        拟合误差平方和。
    """
    y = np.asarray(data, dtype=float)
    if y.ndim != 1 or len(y) < 3:
        raise ValueError("data 必须是长度 >= 3 的一维数组")
    if h < 1:
        raise ValueError("h 必须 >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence 必须在 (0, 1) 之间")

    n = len(y)
    z = norm.ppf((1 + confidence) / 2)

    detected_period = detect_seasonal_period(y) if season_length is None else None
    if season_length is None:
        season_length = detected_period

    if model == "auto":
        if season_length is not None and n >= 2 * season_length:
            model = "holt-winters"
            if seasonal is None:
                seasonal = "add"
        elif detected_period is None:
            model = "holt"
        else:
            model = "holt"

    model = model.lower()

    if model == "ses":
        if params is not None and "alpha" in params:
            alpha = params["alpha"]
            if not 0 < alpha < 1:
                raise ValueError("alpha 必须在 (0, 1) 之间")
            l, residuals, sse = _fit_ses(y, alpha)
        else:
            res = minimize(
                _ses_objective,
                x0=[0.3],
                args=(y,),
                method="L-BFGS-B",
                bounds=[(1e-4, 1 - 1e-4)],
            )
            alpha = float(res.x[0])
            l, residuals, sse = _fit_ses(y, alpha)

        sigma2 = np.mean(residuals ** 2)
        sigma = np.sqrt(sigma2) if sigma2 > 0 else 0.0
        forecast = np.full(h, l[-1])
        se = np.array([sigma * np.sqrt(1 + (k - 1) * alpha ** 2) for k in range(1, h + 1)])
        opt_params = {"alpha": alpha}

    elif model == "holt":
        if params is not None and "alpha" in params and "beta" in params:
            alpha, beta = params["alpha"], params["beta"]
            if not (0 < alpha < 1 and 0 < beta < 1):
                raise ValueError("alpha, beta 必须在 (0, 1) 之间")
            l, b, residuals, sse = _fit_holt(y, alpha, beta)
        else:
            res = minimize(
                _holt_objective,
                x0=[0.3, 0.1],
                args=(y,),
                method="L-BFGS-B",
                bounds=[(1e-4, 1 - 1e-4), (1e-4, 1 - 1e-4)],
            )
            alpha, beta = float(res.x[0]), float(res.x[1])
            l, b, residuals, sse = _fit_holt(y, alpha, beta)

        sigma2 = np.mean(residuals ** 2)
        sigma = np.sqrt(sigma2) if sigma2 > 0 else 0.0
        forecast = np.array([l[-1] + k * b[-1] for k in range(1, h + 1)])
        se = np.array([
            sigma * np.sqrt(1 + (k - 1) * (alpha ** 2 + alpha * beta * k + beta ** 2 * k * (2 * k - 1) / 6))
            for k in range(1, h + 1)
        ])
        opt_params = {"alpha": alpha, "beta": beta}

    elif model == "holt-winters":
        if season_length is None:
            raise ValueError("Holt-Winters 需要指定或检测到 season_length")
        m = season_length
        if n < 2 * m:
            raise ValueError(f"Holt-Winters 至少需要 2 个周期的数据 (2*{m} = {2*m})，当前 {n}")

        if seasonal is None:
            seasonal = "add"
        seasonal = seasonal.lower()
        if seasonal not in ("add", "mul"):
            raise ValueError("seasonal 必须是 'add' 或 'mul'")

        if params is not None and all(k in params for k in ("alpha", "beta", "gamma")):
            alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
            if not (0 < alpha < 1 and 0 < beta < 1 and 0 < gamma < 1):
                raise ValueError("alpha, beta, gamma 必须在 (0, 1) 之间")
            l, b, s, residuals, sse = _fit_holt_winters(y, alpha, beta, gamma, m, seasonal)
        else:
            res = minimize(
                _holt_winters_objective,
                x0=[0.3, 0.1, 0.1],
                args=(y, m, seasonal),
                method="L-BFGS-B",
                bounds=[(1e-4, 1 - 1e-4)] * 3,
            )
            alpha, beta, gamma = float(res.x[0]), float(res.x[1]), float(res.x[2])
            l, b, s, residuals, sse = _fit_holt_winters(y, alpha, beta, gamma, m, seasonal)

        sigma2 = np.mean(residuals ** 2)
        sigma = np.sqrt(sigma2) if sigma2 > 0 else 0.0

        if seasonal == "add":
            forecast = np.array([
                l[-1] + k * b[-1] + s[-m + (k - 1) % m]
                for k in range(1, h + 1)
            ])
        else:
            forecast = np.array([
                (l[-1] + k * b[-1]) * s[-m + (k - 1) % m]
                for k in range(1, h + 1)
            ])

        se = np.full(h, sigma)
        for k in range(1, h + 1):
            if seasonal == "add":
                se[k - 1] = sigma * np.sqrt(
                    1 + (k - 1) * (alpha ** 2 + alpha * beta * k + beta ** 2 * k * (2 * k - 1) / 6)
                    + (gamma if k >= m else 0)
                )
            else:
                se[k - 1] = sigma * np.sqrt(1 + (k - 1) * (alpha ** 2 + alpha * beta * k))
        opt_params = {"alpha": alpha, "beta": beta, "gamma": gamma}

    else:
        raise ValueError(f"未知模型: {model}，可选: 'ses', 'holt', 'holt-winters', 'auto'")

    lower = forecast - z * se
    upper = forecast + z * se

    return forecast, lower, upper, opt_params, sse


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("测试 1: 随机游走序列 (无趋势无季节)")
    print("=" * 70)
    series1 = np.cumsum(np.random.randn(50)) + 100
    fc1, lo1, hi1, params1, sse1 = exponential_smoothing(series1, model="ses", h=5)
    print(f"模型: SES")
    print(f"参数: {params1}")
    print(f"SSE: {sse1:.4f}")
    print(f"预测: {fc1.round(3)}")
    print()

    print("=" * 70)
    print("测试 2: 带线性趋势序列")
    print("=" * 70)
    t = np.arange(60)
    series2 = 100 + 0.5 * t + np.random.randn(60) * 2
    fc2, lo2, hi2, params2, sse2 = exponential_smoothing(series2, model="holt", h=10)
    print(f"模型: Holt 线性趋势")
    print(f"参数: {params2}")
    print(f"SSE: {sse2:.4f}")
    print(f"前5个预测: {fc2[:5].round(3)} ...")
    print()

    print("=" * 70)
    print("测试 3: 带季节性序列 (周期=12，加法季节)")
    print("=" * 70)
    period = 12
    t = np.arange(5 * period)
    season_effect = 5 * np.sin(2 * np.pi * t / period)
    series3 = 100 + 0.2 * t + season_effect + np.random.randn(len(t)) * 1.5
    detected = detect_seasonal_period(series3)
    print(f"自动检测季节周期: {detected}")
    fc3, lo3, hi3, params3, sse3 = exponential_smoothing(
        series3, model="holt-winters", seasonal="add", season_length=period, h=period
    )
    print(f"模型: Holt-Winters (加法)")
    print(f"参数: {params3}")
    print(f"SSE: {sse3:.4f}")
    print(f"一个周期的预测: {fc3.round(2)}")
    print()

    print("=" * 70)
    print("测试 4: 乘法季节序列")
    print("=" * 70)
    season_mult = 1 + 0.1 * np.sin(2 * np.pi * t / period)
    series4 = (100 + 0.2 * t) * season_mult + np.random.randn(len(t)) * 2
    fc4, lo4, hi4, params4, sse4 = exponential_smoothing(
        series4, model="holt-winters", seasonal="mul", season_length=period, h=period
    )
    print(f"模型: Holt-Winters (乘法)")
    print(f"参数: {params4}")
    print(f"SSE: {sse4:.4f}")
    print(f"一个周期的预测: {fc4.round(2)}")
    print()

    print("=" * 70)
    print("测试 5: 自动选择模型")
    print("=" * 70)
    fc_auto, lo_auto, hi_auto, params_auto, sse_auto = exponential_smoothing(
        series3, model="auto", h=5
    )
    print(f"自动选择模型参数: {params_auto}")
    print(f"SSE: {sse_auto:.4f}")
    print(f"预测: {fc_auto.round(3)}")
