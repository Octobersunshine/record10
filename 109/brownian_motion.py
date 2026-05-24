import numpy as np
from typing import Tuple, Optional, List, Union


def wiener_process(
    T: float,
    N: int,
    num_paths: int = 1,
    seed: Optional[int] = None,
    return_time: bool = True
) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    生成标准维纳过程（布朗运动）的路径
    
    参数:
        T: 时间区间长度（总时间）
        N: 时间步数（离散化点数）
        num_paths: 模拟路径数量（默认1条）
        seed: 随机种子（可选）
        return_time: 是否返回时间网格（默认True）
    
    返回:
        如果return_time=True，返回 (时间网格, 位置序列)
        否则只返回位置序列
        
    形状:
        时间网格: (N+1,)
        位置序列: (num_paths, N+1) 当 num_paths > 1 时
                  (N+1,) 当 num_paths == 1 时
    
    数学性质:
        - W(0) = 0
        - 增量独立同分布: W(t+Δt) - W(t) ~ N(0, Δt)
        - 连续路径，处处不可微
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_paths, N))
    
    W = np.zeros((num_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    
    t = np.linspace(0, T, N + 1)
    
    if num_paths == 1:
        W = W[0]
    
    if return_time:
        return t, W
    return W


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    num_paths: int = 1,
    seed: Optional[int] = None,
    return_time: bool = True
) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    生成几何布朗运动（GBM）路径，用于资产价格模拟
    
    参数:
        S0: 初始价格
        mu: 漂移率（年化收益率）
        sigma: 波动率（年化）
        T: 时间区间长度
        N: 时间步数
        num_paths: 模拟路径数量
        seed: 随机种子
        return_time: 是否返回时间网格
    
    公式:
        S(t) = S0 * exp((μ - σ²/2)t + σW(t))
    """
    t, W = wiener_process(T, N, num_paths, seed, return_time=True)
    
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    
    S = S0 * np.exp(drift + diffusion)
    
    if return_time:
        return t, S
    return S


def _make_covariance_matrix(t: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    构造布朗运动的协方差矩阵 Cov[W(s), W(t)] = min(s, t)
    
    参数:
        t: 时间点数组，形状 (M,)
        eps: 数值稳定化小量
    
    返回:
        协方差矩阵，形状 (M, M)
    """
    M = len(t)
    cov = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            cov[i, j] = min(t[i], t[j])
    return cov + eps * np.eye(M)


def _cholesky_with_correction(
    cov: np.ndarray,
    reg_param: float = 1e-8,
    max_attempts: int = 10
) -> np.ndarray:
    """
    带数值修正的Cholesky分解，处理协方差矩阵的数值不稳定问题
    
    参数:
        cov: 协方差矩阵
        reg_param: 初始正则化参数
        max_attempts: 最大尝试次数
    
    返回:
        下三角矩阵 L，满足 L @ L.T ≈ cov
    
    修正策略:
        1. 先尝试直接Cholesky分解
        2. 若失败，检查特征值并修正负特征值
        3. 逐步增加正则化参数
    """
    cov = (cov + cov.T) / 2.0
    
    try:
        L = np.linalg.cholesky(cov)
        return L
    except np.linalg.LinAlgError:
        pass
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    min_eig = eigvals.min()
    
    if min_eig < 0:
        correction = -min_eig + reg_param
        cov_corrected = cov + correction * np.eye(cov.shape[0])
        try:
            L = np.linalg.cholesky(cov_corrected)
            return L
        except np.linalg.LinAlgError:
            pass
    
    for attempt in range(max_attempts):
        reg = reg_param * (10 ** attempt)
        try:
            L = np.linalg.cholesky(cov + reg * np.eye(cov.shape[0]))
            return L
        except np.linalg.LinAlgError:
            continue
    
    raise np.linalg.LinAlgError(
        f"Cholesky分解失败，已尝试{max_attempts}次正则化"
    )


def conditional_brownian_motion_cholesky(
    t_target: np.ndarray,
    fixed_times: Union[List[float], np.ndarray],
    fixed_values: Union[List[float], np.ndarray],
    num_paths: int = 1,
    seed: Optional[int] = None,
    return_full: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于Cholesky分解的条件布朗运动生成（精确方法）
    
    解决传统布朗桥插值的问题:
        1. 避免递归插值导致的自相关性偏差
        2. 正确处理多固定点的协方差结构
        3. 通过数值修正保证协方差矩阵正定性
    
    参数:
        t_target: 需要生成的目标时间点数组
        fixed_times: 固定时间点 [t1, t2, ..., tk]
        fixed_values: 固定值 [W(t1), W(t2), ..., W(tk)]
        num_paths: 生成路径数量
        seed: 随机种子
        return_full: 是否包含固定点在输出中
    
    返回:
        (t_sorted, W_paths)
        t_sorted: 排序后的时间点
        W_paths: 形状 (num_paths, len(t_sorted))
    
    数学推导:
        设 W = [W_fixed; W_target]
        协方差矩阵分块: [[C_ff, C_ft], [C_tf, C_tt]]
        条件分布: W_target | W_fixed=w ~ N(μ, Σ)
        μ = C_tf @ C_ff^{-1} @ w
        Σ = C_tt - C_tf @ C_ff^{-1} @ C_ft
    """
    if seed is not None:
        np.random.seed(seed)
    
    fixed_times = np.asarray(fixed_times)
    fixed_values = np.asarray(fixed_values)
    t_target = np.asarray(t_target)
    
    sort_idx = np.argsort(fixed_times)
    fixed_times = fixed_times[sort_idx]
    fixed_values = fixed_values[sort_idx]
    
    if return_full:
        t_all = np.unique(np.concatenate([fixed_times, t_target]))
    else:
        t_all = np.unique(t_target)
    
    t_all = np.sort(t_all)
    
    n_fixed = len(fixed_times)
    n_target = len(t_all)
    
    fixed_mask = np.isin(t_all, fixed_times)
    target_mask = ~fixed_mask
    
    t_fixed = t_all[fixed_mask]
    t_target_only = t_all[target_mask]
    
    if len(t_target_only) == 0:
        W = np.tile(fixed_values, (num_paths, 1))
        return t_all, W
    
    C_ff = _make_covariance_matrix(t_fixed)
    C_ft = np.zeros((n_fixed, len(t_target_only)))
    for i in range(n_fixed):
        for j in range(len(t_target_only)):
            C_ft[i, j] = min(t_fixed[i], t_target_only[j])
    
    C_tf = C_ft.T
    C_tt = _make_covariance_matrix(t_target_only)
    
    L_ff = _cholesky_with_correction(C_ff)
    C_ff_inv = np.linalg.inv(C_ff)
    
    mu_target = C_tf @ C_ff_inv @ fixed_values
    
    Sigma_target = C_tt - C_tf @ C_ff_inv @ C_ft
    Sigma_target = (Sigma_target + Sigma_target.T) / 2.0
    
    L_target = _cholesky_with_correction(Sigma_target)
    
    n_target_only = len(t_target_only)
    Z = np.random.normal(0, 1, size=(num_paths, n_target_only))
    W_target = mu_target + Z @ L_target.T
    
    W_full = np.zeros((num_paths, len(t_all)))
    fixed_positions = np.where(fixed_mask)[0]
    target_positions = np.where(target_mask)[0]
    
    for i, pos in enumerate(fixed_positions):
        W_full[:, pos] = fixed_values[i]
    
    for j, pos in enumerate(target_positions):
        W_full[:, pos] = W_target[:, j]
    
    return t_all, W_full


def brownian_bridge_naive(
    t_target: np.ndarray,
    t0: float,
    t1: float,
    W0: float,
    W1: float,
    num_paths: int = 1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    传统布朗桥插值（用于对比展示自相关性问题）
    
    警告: 此方法在多固定点级联时会产生自相关性偏差
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_target = np.sort(np.asarray(t_target))
    dt = t1 - t0
    
    t_all = np.sort(np.concatenate([[t0, t1], t_target]))
    W = np.zeros((num_paths, len(t_all)))
    W[:, 0] = W0
    W[:, -1] = W1
    
    for i in range(1, len(t_all) - 1):
        ti = t_all[i]
        left_idx = np.max(np.where(t_all <= ti)[0])
        right_idx = np.min(np.where(t_all >= ti)[0])
        
        t_left = t_all[left_idx]
        t_right = t_all[right_idx]
        W_left = W[:, left_idx]
        W_right = W[:, right_idx] if right_idx != i else W1
        
        dt_segment = t_right - t_left
        if dt_segment == 0:
            W[:, i] = W_left
            continue
        
        a = (t_right - ti) / dt_segment
        b = (ti - t_left) / dt_segment
        var = a * b * dt_segment
        
        mean = a * W_left + b * W_right
        W[:, i] = mean + np.random.normal(0, np.sqrt(var), size=num_paths)
    
    return t_all, W


def check_autocorrelation(
    t: np.ndarray,
    W: np.ndarray,
    lag: int = 1
) -> Tuple[float, float]:
    """
    检验增量的自相关性，验证布朗运动性质
    
    返回:
        (自相关系数, 理论值=0)
    """
    if W.ndim == 1:
        W = W.reshape(1, -1)
    
    dW = np.diff(W, axis=1)
    n_paths, n_steps = dW.shape
    
    if lag >= n_steps:
        return np.nan, 0.0
    
    autocorrs = []
    for i in range(n_paths):
        x = dW[i, :-lag]
        y = dW[i, lag:]
        corr = np.corrcoef(x, y)[0, 1]
        autocorrs.append(corr)
    
    return np.mean(autocorrs), 0.0


def fgn_autocovariance(k: int, H: float) -> float:
    """
    分数高斯噪声（FGN）的自协方差函数
    
    参数:
        k: 滞后阶数
        H: Hurst指数 (0 < H < 1)
    
    公式:
        γ(k) = 0.5 * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))
    """
    return 0.5 * (abs(k + 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k - 1) ** (2 * H))


def fbm_circulant_embedding(
    H: float,
    N: int
) -> np.ndarray:
    """
    循环嵌入法构造协方差矩阵的第一行（用于FFT方法）
    
    返回长度为2N的循环协方差序列
    """
    if H == 0.5:
        circ_cov = np.zeros(2 * N)
        circ_cov[0] = 1.0
        return circ_cov
    
    k = np.arange(N)
    gamma = fgn_autocovariance(k, H)
    
    circ_cov = np.zeros(2 * N)
    circ_cov[:N] = gamma
    circ_cov[N:] = gamma[-2::-1] if N > 1 else np.array([])
    
    return circ_cov


def fractional_brownian_motion_fft(
    H: float,
    T: float,
    N: int,
    num_paths: int = 1,
    seed: Optional[int] = None,
    return_time: bool = True,
    method: str = 'circulant'
) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    基于快速傅里叶变换（FFT）的分数布朗运动（fBm）模拟
    
    分数布朗运动性质:
        - 当 H = 0.5 时，退化为标准布朗运动
        - 当 H > 0.5 时，增量正相关（长记忆性、持久性）
        - 当 H < 0.5 时，增量负相关（反持久性）
    
    参数:
        H: Hurst指数，0 < H < 1
        T: 时间区间长度
        N: 时间步数（离散化点数）
        num_paths: 模拟路径数量
        seed: 随机种子
        return_time: 是否返回时间网格
        method: 生成方法 ('circulant' 循环嵌入法)
    
    返回:
        (时间网格, fBm路径) 或 fBm路径
        
    应用领域:
        - 水文学: 河流流量模拟 (H ≈ 0.7-0.8)
        - 通信: 网络流量建模 (H ≈ 0.7-0.9)
        - 金融: 波动率长记忆性 (H ≈ 0.6-0.7)
        - 图像处理: 分形纹理生成
    """
    if not (0 < H < 1):
        raise ValueError("Hurst指数 H 必须在 (0, 1) 范围内")
    
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    scale = dt ** H
    
    if H == 0.5:
        return wiener_process(T, N, num_paths, seed, return_time)
    
    M = 1
    while M < 2 * N:
        M *= 2
    
    k = np.arange(N)
    gamma = np.array([fgn_autocovariance(ki, H) for ki in k])
    
    circ_gamma = np.zeros(M)
    circ_gamma[:N] = gamma
    if N > 1:
        circ_gamma[M - N + 1:] = gamma[-1:0:-1]
    
    eigenvalues = np.fft.fft(circ_gamma).real
    eigenvalues = np.maximum(eigenvalues, 0)
    
    sqrt_eig = np.sqrt(eigenvalues)
    
    Z = np.random.normal(0, 1, size=(num_paths, M)) + \
        1j * np.random.normal(0, 1, size=(num_paths, M))
    Z[:, 0] = Z[:, 0].real * np.sqrt(2)
    Z[:, M//2] = Z[:, M//2].real * np.sqrt(2) if M % 2 == 0 else Z[:, M//2]
    
    Y = sqrt_eig * Z / np.sqrt(2)
    fgn_fft = np.fft.ifft(Y, axis=1).real
    
    fgn = fgn_fft[:, :N] * scale
    
    fBm = np.zeros((num_paths, N + 1))
    fBm[:, 1:] = np.cumsum(fgn, axis=1)
    
    t = np.linspace(0, T, N + 1)
    
    if num_paths == 1:
        fBm = fBm[0]
    
    if return_time:
        return t, fBm
    return fBm


def compute_hurst_exponent(
    t: np.ndarray,
    X: np.ndarray,
    max_lag: int = 20
) -> float:
    """
    基于方差标度律估计Hurst指数
    
    Var[X(t+τ) - X(t)] ∝ τ^(2H)
    
    参数:
        t: 时间网格
        X: 时间序列
        max_lag: 最大滞后阶数
    
    返回:
        估计的Hurst指数
    """
    if X.ndim == 2:
        X = X[0]
    
    dt = t[1] - t[0]
    lags = np.arange(1, max_lag + 1)
    variances = []
    
    for lag in lags:
        diff = X[lag:] - X[:-lag]
        variances.append(np.var(diff))
    
    log_lags = np.log(lags * dt)
    log_vars = np.log(variances)
    
    slope, _ = np.polyfit(log_lags, log_vars, 1)
    H_est = slope / 2.0
    
    return H_est


def autocorrelation_function(
    X: np.ndarray,
    max_lag: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算时间序列增量的自相关函数
    
    参数:
        X: 时间序列 (路径数 x 时间点数) 或 (时间点数,)
        max_lag: 最大滞后阶数
    
    返回:
        (lags, autocorrelations)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    dX = np.diff(X, axis=1)
    n_paths, n = dX.shape
    
    lags = np.arange(1, max_lag + 1)
    acf = np.zeros(max_lag)
    
    mean_dX = np.mean(dX)
    var_dX = np.var(dX)
    
    for k, lag in enumerate(lags):
        if lag >= n:
            acf[k] = np.nan
            continue
        cov = np.mean((dX[:, lag:] - mean_dX) * (dX[:, :-lag] - mean_dX))
        acf[k] = cov / var_dX
    
    return lags, acf


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("=" * 60)
    print("分数布朗运动（FFT方法）验证")
    print("=" * 60)
    
    T = 1.0
    N = 2048
    num_paths = 10
    
    H_values = [0.3, 0.5, 0.7, 0.9]
    fbm_results = {}
    
    for H in H_values:
        t, fBm = fractional_brownian_motion_fft(
            H, T, N, num_paths=num_paths, seed=42
        )
        fbm_results[H] = (t, fBm)
        
        H_est = compute_hurst_exponent(t, fBm[0], max_lag=30)
        lags, acf = autocorrelation_function(fBm[0], max_lag=50)
        
        print(f"\nH = {H}:")
        print(f"  估计 Hurst 指数: {H_est:.4f}")
        print(f"  滞后1阶自相关: {acf[0]:.6f}")
        print(f"  滞后10阶自相关: {acf[9]:.6f}")
    
    print("\n" + "=" * 60)
    print("水文学应用：尼罗河流量长记忆性模拟")
    print("=" * 60)
    
    H_hydro = 0.75
    T_years = 100
    N_days = 365 * 100
    
    t_hydro, flow_base = fractional_brownian_motion_fft(
        H_hydro, T_years, N_days, num_paths=3, seed=123
    )
    
    base_flow = 1000
    seasonal_amp = 300
    seasonal_flow = base_flow + seasonal_amp * np.sin(2 * np.pi * t_hydro)
    flow_sim = seasonal_flow.reshape(1, -1) + 150 * flow_base
    
    print(f"Hurst指数: {H_hydro} (典型水文序列值)")
    print(f"模拟时长: {T_years} 年 ({N_days} 天)")
    print(f"平均流量: {np.mean(flow_sim):.1f} m³/s")
    print(f"流量标准差: {np.std(flow_sim):.1f} m³/s")
    
    print("\n" + "=" * 60)
    print("通信应用：网络流量自相似性模拟")
    print("=" * 60)
    
    H_network = 0.85
    T_sec = 10
    N_packets = 10000
    
    t_net, traffic_fbm = fractional_brownian_motion_fft(
        H_network, T_sec, N_packets, num_paths=1, seed=456
    )
    
    arrival_rate = 1000
    traffic = arrival_rate + 200 * traffic_fbm
    traffic = np.maximum(traffic, 0)
    
    H_net_est = compute_hurst_exponent(t_net, traffic_fbm, max_lag=50)
    print(f"Hurst指数: {H_network} (典型网络流量值)")
    print(f"估计 Hurst 指数: {H_net_est:.4f}")
    print(f"模拟时长: {T_sec} 秒 ({N_packets} 时间点)")
    print(f"平均到达率: {np.mean(traffic):.1f} 包/秒")
    
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, 1)
    colors = ['blue', 'green', 'orange', 'red']
    for i, H in enumerate(H_values):
        t, fBm = fbm_results[H]
        plt.plot(t, fBm[0], label=f'H={H}', color=colors[i], linewidth=0.8)
    plt.title('不同Hurst指数的分数布朗运动')
    plt.xlabel('时间 t')
    plt.ylabel('B_H(t)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 2)
    for i, H in enumerate(H_values):
        t, fBm = fbm_results[H]
        dX = np.diff(fBm[0])
        plt.plot(t[:-1], dX, label=f'H={H}', color=colors[i], linewidth=0.5, alpha=0.7)
    plt.title('分数高斯噪声（增量）')
    plt.xlabel('时间 t')
    plt.ylabel('dB_H(t)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 3)
    max_lag_acf = 100
    for i, H in enumerate(H_values):
        t, fBm = fbm_results[H]
        lags, acf = autocorrelation_function(fBm, max_lag=max_lag_acf)
        plt.plot(lags, acf, label=f'H={H}', color=colors[i], linewidth=1)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.title('增量自相关函数')
    plt.xlabel('滞后阶数 k')
    plt.ylabel('ρ(k)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.yscale('symlog', linthresh=0.01)
    
    plt.subplot(3, 3, 4)
    for i, H in enumerate(H_values):
        t, fBm = fbm_results[H]
        max_lag = 50
        lags_var = np.arange(1, max_lag + 1)
        dt = t[1] - t[0]
        vars_scaled = []
        for lag in lags_var:
            diff = fBm[0][lag:] - fBm[0][:-lag]
            vars_scaled.append(np.var(diff))
        plt.loglog(lags_var * dt, vars_scaled, label=f'H={H}', color=colors[i], linewidth=1)
    plt.title('方差标度律 (log-log)')
    plt.xlabel('时间尺度 τ')
    plt.ylabel('Var[B_H(t+τ)-B_H(t)]')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, which='both')
    
    plt.subplot(3, 3, 5)
    t_plot_hydro = t_hydro[:365*10]
    flow_plot = flow_sim[0][:365*10]
    plt.plot(t_plot_hydro, flow_plot, color='blue', linewidth=0.5)
    plt.axhline(base_flow, color='r', linestyle='--', label='基准流量', alpha=0.7)
    plt.title('水文模拟：河流流量（前10年）')
    plt.xlabel('时间（年）')
    plt.ylabel('流量 (m³/s)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 6)
    n_years = 10
    yearly_max = []
    for y in range(n_years):
        start = y * 365
        end = (y + 1) * 365
        yearly_max.append(np.max(flow_sim[0][start:end]))
    plt.bar(range(1, n_years + 1), yearly_max, color='blue', alpha=0.7)
    plt.title('年最大流量（洪水分析）')
    plt.xlabel('年份')
    plt.ylabel('最大流量 (m³/s)')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(3, 3, 7)
    t_plot_net = t_net[:1000]
    traffic_plot = traffic[0][:1000]
    plt.plot(t_plot_net, traffic_plot, color='purple', linewidth=0.5)
    plt.axhline(arrival_rate, color='r', linestyle='--', label='平均到达率', alpha=0.7)
    plt.title('网络流量模拟（前1000个时间点）')
    plt.xlabel('时间（秒）')
    plt.ylabel('到达率（包/秒）')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 8)
    traffic_sorted = np.sort(traffic[0])[::-1]
    cdf = np.arange(1, len(traffic_sorted) + 1) / len(traffic_sorted)
    plt.loglog(traffic_sorted, 1 - cdf, 'o', markersize=2, color='purple', alpha=0.6)
    plt.title('流量互补累积分布 (CCDF)')
    plt.xlabel('到达率（包/秒）')
    plt.ylabel('P(X > x)')
    plt.grid(True, alpha=0.3, which='both')
    
    plt.subplot(3, 3, 9)
    q = np.linspace(0.5, 0.999, 50)
    quantiles = np.quantile(traffic[0], q)
    plt.plot(q, quantiles, color='purple', linewidth=1.5)
    plt.axhline(arrival_rate, color='r', linestyle='--', label='均值', alpha=0.5)
    plt.title('分位数-分位数图')
    plt.xlabel('分位数')
    plt.ylabel('到达率（包/秒）')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fractional_brownian_motion.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("图表已保存: fractional_brownian_motion.png")
    
    print("\n" + "=" * 60)
    print("标准维纳过程验证")
    print("=" * 60)
    T = 1.0
    N = 1000
    num_paths = 1000
    
    t, W = wiener_process(T, N, num_paths=num_paths, seed=42)
    
    print(f"生成了 {num_paths} 条维纳过程路径")
    print(f"时间步数: {N}")
    print(f"时间步长: {T/N:.6f}")
    print(f"W(T) 的均值: {np.mean(W[:, -1]):.6f} (理论值: 0)")
    print(f"W(T) 的方差: {np.var(W[:, -1]):.6f} (理论值: {T})")
    
    autocorr, _ = check_autocorrelation(t, W, lag=1)
    print(f"增量自相关系数: {autocorr:.6f} (理论值: 0)")
    
    print("\n" + "=" * 60)
    print("条件布朗运动验证（Cholesky方法 vs 传统布朗桥）")
    print("=" * 60)
    
    fixed_times = [0.0, 0.5, 1.0]
    fixed_values = [0.0, 0.5, 0.0]
    t_target = np.linspace(0, 1, 51)
    num_paths_test = 1000
    
    t_chol, W_chol = conditional_brownian_motion_cholesky(
        t_target, fixed_times, fixed_values,
        num_paths=num_paths_test, seed=42
    )
    
    t_naive, W_naive = brownian_bridge_naive(
        t_target, 0.0, 1.0, 0.0, 0.0,
        num_paths=num_paths_test, seed=42
    )
    
    chol_autocorr, _ = check_autocorrelation(t_chol, W_chol, lag=1)
    naive_autocorr, _ = check_autocorrelation(t_naive, W_naive, lag=1)
    
    print(f"\nCholesky方法 - 增量自相关: {chol_autocorr:.6f} (应接近0)")
    print(f"传统布朗桥 - 增量自相关: {naive_autocorr:.6f} (可能有偏差)")
    
    t_mid = 0.5
    idx_mid = np.argmin(np.abs(t_chol - t_mid))
    print(f"Cholesky方法 - W(0.5) 均值: {np.mean(W_chol[:, idx_mid]):.6f} (固定值: 0.5)")
    print(f"Cholesky方法 - W(0.5) 标准差: {np.std(W_chol[:, idx_mid]):.6f} (应为0，固定点)")
    
    print("\n" + "=" * 60)
    print("多固定点条件模拟示例（金融应用）")
    print("=" * 60)
    
    S0 = 100
    mu = 0.05
    sigma = 0.2
    
    obs_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    obs_prices = [100, 95, 105, 100, 102]
    
    obs_log_prices = np.log(obs_prices)
    drift_term = (mu - 0.5 * sigma**2) * np.array(obs_times)
    fixed_W_values = (obs_log_prices - np.log(S0) - drift_term) / sigma
    
    t_fine = np.linspace(0, 1, 101)
    t_cond, W_cond = conditional_brownian_motion_cholesky(
        t_fine, obs_times, fixed_W_values,
        num_paths=5, seed=123
    )
    
    drift_cond = (mu - 0.5 * sigma**2) * t_cond
    S_cond = S0 * np.exp(drift_cond + sigma * W_cond)
    
    print(f"观测时间点: {obs_times}")
    print(f"观测价格: {obs_prices}")
    for i, t_obs in enumerate(obs_times):
        idx = np.argmin(np.abs(t_cond - t_obs))
        price_at_obs = np.mean(S_cond[:, idx])
        print(f"  t={t_obs:.2f}: 模拟价格均值 = {price_at_obs:.2f} (观测值 = {obs_prices[i]})")
    
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for i in range(min(5, num_paths)):
        plt.plot(t, W[i], linewidth=1, alpha=0.8)
    plt.title('标准维纳过程示例路径')
    plt.xlabel('时间 t')
    plt.ylabel('W(t)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    for i in range(5):
        plt.plot(t_chol, W_chol[i], linewidth=1, alpha=0.8)
    for t_fix, w_fix in zip(fixed_times, fixed_values):
        plt.plot(t_fix, w_fix, 'ro', markersize=8, zorder=5)
    plt.title('Cholesky条件布朗运动（多固定点）')
    plt.xlabel('时间 t')
    plt.ylabel('W(t)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    for i in range(5):
        plt.plot(t_cond, S_cond[i], linewidth=1, alpha=0.8)
    plt.plot(obs_times, obs_prices, 'ro-', markersize=8, 
             label='观测点', linewidth=2, zorder=5)
    plt.title('条件几何布朗运动（资产价格）')
    plt.xlabel('时间 t')
    plt.ylabel('价格 S(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    autocorr_lags = range(1, 11)
    chol_corrs = [check_autocorrelation(t_chol, W_chol, lag=l)[0] 
                  for l in autocorr_lags]
    naive_corrs = [check_autocorrelation(t_naive, W_naive, lag=l)[0] 
                   for l in autocorr_lags]
    plt.plot(autocorr_lags, chol_corrs, 'bo-', label='Cholesky方法')
    plt.plot(autocorr_lags, naive_corrs, 'rs-', label='传统布朗桥')
    plt.axhline(0, color='k', linestyle='--', alpha=0.5, label='理论值')
    plt.title('增量自相关性对比')
    plt.xlabel('滞后阶数')
    plt.ylabel('自相关系数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('brownian_motion_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("图表已保存: brownian_motion_comparison.png")
    print("\n所有验证完成！")
