import numpy as np


def simple_moving_average(signal, window_size):
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if window_size <= 0 or window_size > n:
        raise ValueError("窗口大小必须在1到信号长度之间")
    
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - window_size + 1)
        result[i] = np.mean(signal[start:i+1])
    return result


def weighted_moving_average(signal, window_size, weights=None):
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if window_size <= 0 or window_size > n:
        raise ValueError("窗口大小必须在1到信号长度之间")
    
    if weights is None:
        weights = np.arange(1, window_size + 1, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != window_size:
            raise ValueError("权重数组长度必须等于窗口大小")
    
    weights = weights / weights.sum()
    result = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window_size + 1)
        actual_window = i - start + 1
        w = weights[-actual_window:] if actual_window < window_size else weights
        w = w / w.sum()
        result[i] = np.sum(signal[start:i+1] * w)
    return result


def simple_exponential_smoothing(signal, alpha):
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return np.array([])
    if alpha <= 0 or alpha > 1:
        raise ValueError("平滑因子alpha必须在(0, 1]之间")
    
    result = np.zeros(n)
    result[0] = signal[0]
    for i in range(1, n):
        result[i] = alpha * signal[i] + (1 - alpha) * result[i-1]
    return result


def _sg_coefficients(window_size, order):
    half_window = (window_size - 1) // 2
    x = np.arange(-half_window, half_window + 1, dtype=np.float64)
    A = np.vander(x, order + 1, increasing=True)
    ATA = A.T @ A
    inv_ATA = np.linalg.inv(ATA)
    coeffs = inv_ATA @ A.T
    return coeffs[0, :]


def _pad_signal(signal, window_size, mode='mirror'):
    half_window = (window_size - 1) // 2
    n = len(signal)
    if mode == 'mirror':
        left_pad = signal[1:half_window+1][::-1]
        right_pad = signal[-half_window-1:-1][::-1]
    elif mode == 'nearest':
        left_pad = np.full(half_window, signal[0])
        right_pad = np.full(half_window, signal[-1])
    elif mode == 'constant':
        left_pad = np.zeros(half_window)
        right_pad = np.zeros(half_window)
    else:
        raise ValueError(f"不支持的填充模式: {mode}")
    return np.concatenate([left_pad, signal, right_pad])


def savitzky_golay(signal, window_size, order=None, auto_order=True, pad_mode='mirror'):
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return np.array([])
    if window_size < 3:
        raise ValueError("窗口大小必须大于等于3")
    if window_size % 2 == 0:
        window_size += 1
    if window_size > n:
        raise ValueError("窗口大小必须小于等于信号长度")
    
    if order is None:
        if auto_order:
            order = auto_select_order(signal, window_size)
        else:
            order = min(2, window_size - 1)
    
    if order < 0 or order >= window_size:
        raise ValueError("阶数必须在0到窗口大小-1之间")
    
    half_window = (window_size - 1) // 2
    padded = _pad_signal(signal, window_size, pad_mode)
    coeffs = _sg_coefficients(window_size, order)
    
    result = np.zeros(n)
    for i in range(n):
        window_data = padded[i:i + window_size]
        result[i] = np.sum(coeffs * window_data)
    return result


def auto_select_order(signal, window_size, max_order=None):
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    half_window = (window_size - 1) // 2
    
    if max_order is None:
        max_order = min(4, window_size - 2)
    
    if n < window_size * 2:
        return min(2, max_order)
    
    signal_var = np.var(signal)
    noise_var = np.median(np.abs(np.diff(signal))) ** 2 / 2
    noise_var = max(noise_var, 1e-10)
    
    orders = range(0, max_order + 1)
    best_order = min(2, max_order)
    best_aic = float('inf')
    
    for order in orders:
        if order >= window_size:
            continue
        
        try:
            smoothed = savitzky_golay(signal, window_size, order=order, auto_order=False)
            residual = signal - smoothed
            sse = np.sum(residual ** 2)
            
            k = order + 1
            aic = 2 * k + n * np.log(sse / n + 1e-15)
            
            aic += 0.5 * k * np.log(n)
            
            if aic < best_aic:
                best_aic = aic
                best_order = order
        except:
            continue
    
    return best_order


def median_filter(signal, window_size=None, adaptive=True, max_window=9):
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return np.array([])
    
    if window_size is None:
        window_size = 3
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    
    if not adaptive:
        half_window = (window_size - 1) // 2
        padded = _pad_signal(signal, window_size, mode='nearest')
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.median(padded[i:i + window_size])
        return result
    
    half_window = (window_size - 1) // 2
    max_half = (max_window - 1) // 2
    padded = _pad_signal(signal, max_window, mode='nearest')
    result = np.zeros(n)
    
    for i in range(n):
        center = i + max_half
        current_half = half_window
        
        while current_half <= max_half:
            window_data = padded[center - current_half:center + current_half + 1]
            z_med = np.median(window_data)
            z_min = np.min(window_data)
            z_max = np.max(window_data)
            
            if z_min < z_med < z_max:
                z_xy = padded[center]
                if z_min < z_xy < z_max:
                    result[i] = z_xy
                else:
                    result[i] = z_med
                break
            else:
                current_half += 1
        
        if current_half > max_half:
            result[i] = np.median(padded[center - max_half:center + max_half + 1])
    
    return result


def kalman_filter(signal, Q=None, R=None, adaptive=True, dt=1.0):
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return np.array([])
    
    signal_var = np.var(signal)
    noise_est = np.median(np.abs(np.diff(signal))) ** 2 / 2
    noise_est = max(noise_est, 1e-10)
    
    if Q is None:
        Q = noise_est * 0.5
    if R is None:
        R = noise_est
    
    Q_min = Q * 0.01
    Q_max = Q * 100
    R_min = R * 0.1
    R_max = R * 10
    
    A = np.array([[1, dt],
                  [0, 1]])
    H = np.array([[1, 0]])
    
    x = np.zeros((2, n))
    x[0, 0] = signal[0]
    x[1, 0] = 0.0
    
    P = np.eye(2) * signal_var
    
    result = np.zeros(n)
    result[0] = signal[0]
    
    innovation_history = []
    window_size = 10
    
    for k in range(1, n):
        x_pred = A @ x[:, k-1]
        P_pred = A @ P @ A.T
        P_pred[0, 0] += Q
        P_pred[1, 1] += Q
        
        z = signal[k]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S
        
        x[:, k] = x_pred + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P_pred
        
        result[k] = x[0, k]
        
        if adaptive:
            innovation_history.append(y ** 2)
            if len(innovation_history) > window_size:
                innovation_history.pop(0)
            
            if len(innovation_history) >= 5:
                innov_var = np.mean(innovation_history)
                
                if innov_var > S * 2:
                    Q = min(Q_max, Q * 1.2)
                    R = min(R_max, R * 1.1)
                elif innov_var < S * 0.5:
                    Q = max(Q_min, Q * 0.95)
                    R = max(R_min, R * 0.95)
    
    return result


if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("测试1: 含突变峰值的信号平滑对比")
    print("=" * 60)
    t = np.linspace(0, 10, 100)
    signal = np.sin(t) + 0.2 * np.random.randn(100)
    signal[30:40] += 2.0
    signal[60:70] -= 1.5
    window = 7
    
    print(f"\n窗口大小: {window}")
    print(f"信号特征: 正弦波+高斯噪声, 含正峰值(30-40)和负峰值(60-70)\n")
    
    print("--- 简单移动平均 ---")
    sma = simple_moving_average(signal, window)
    print(f"  正峰位置误差: {np.argmax(signal[25:45]) - np.argmax(sma[25:45])}")
    print(f"  正峰幅度误差: {np.max(signal[25:45]) - np.max(sma[25:45]):.4f}")
    
    print("\n--- Savitzky-Golay (自动阶数) ---")
    auto_order = auto_select_order(signal, window)
    print(f"  自动选择的多项式阶数: {auto_order}")
    sg_auto = savitzky_golay(signal, window, auto_order=True)
    print(f"  正峰位置误差: {np.argmax(signal[25:45]) - np.argmax(sg_auto[25:45])}")
    print(f"  正峰幅度误差: {np.max(signal[25:45]) - np.max(sg_auto[25:45]):.4f}")
    
    print("\n--- 中值滤波 (自适应窗口) ---")
    med = median_filter(signal, window_size=5, adaptive=True)
    print(f"  正峰位置误差: {np.argmax(signal[25:45]) - np.argmax(med[25:45])}")
    print(f"  正峰幅度误差: {np.max(signal[25:45]) - np.max(med[25:45]):.4f}")
    
    print("\n--- 卡尔曼滤波 (自适应参数) ---")
    kf = kalman_filter(signal, adaptive=True)
    print(f"  正峰位置误差: {np.argmax(signal[25:45]) - np.argmax(kf[25:45])}")
    print(f"  正峰幅度误差: {np.max(signal[25:45]) - np.max(kf[25:45]):.4f}")
    
    print("\n" + "=" * 50)
    print("峰值幅度误差对比 (越小越好):")
    print("=" * 50)
    methods = ["SMA", "SG(auto)", "Median", "Kalman"]
    errors = [
        abs(np.max(signal[25:45]) - np.max(sma[25:45])),
        abs(np.max(signal[25:45]) - np.max(sg_auto[25:45])),
        abs(np.max(signal[25:45]) - np.max(med[25:45])),
        abs(np.max(signal[25:45]) - np.max(kf[25:45])),
    ]
    for m, e in zip(methods, errors):
        bar = "█" * int(e * 50)
        print(f"  {m:8s}: {e:.4f}  {bar}")
    
    print("\n" + "=" * 60)
    print("测试2: 椒盐噪声鲁棒性对比")
    print("=" * 60)
    t2 = np.linspace(0, 8, 80)
    clean_signal = np.sin(t2) * 3
    noisy_signal = clean_signal.copy()
    salt_pepper_idx = np.random.choice(80, 16, replace=False)
    noisy_signal[salt_pepper_idx[:8]] = 10
    noisy_signal[salt_pepper_idx[8:]] = -10
    
    print(f"\n噪声类型: 椒盐噪声 (20% 污染)")
    print(f"噪声强度: ±10, 信号幅度: ±3\n")
    
    sma_noise = simple_moving_average(noisy_signal, 5)
    sg_noise = savitzky_golay(noisy_signal, 5, order=2, auto_order=False)
    med_noise = median_filter(noisy_signal, window_size=5, adaptive=False)
    med_adapt = median_filter(noisy_signal, window_size=3, adaptive=True, max_window=7)
    
    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))
    
    print("RMSE (相对于干净信号, 越小越好):")
    print(f"  原始噪声:    {rmse(noisy_signal, clean_signal):.4f}")
    print(f"  移动平均:    {rmse(sma_noise, clean_signal):.4f}")
    print(f"  SG滤波:      {rmse(sg_noise, clean_signal):.4f}")
    print(f"  中值滤波:    {rmse(med_noise, clean_signal):.4f}")
    print(f"  自适应中值:  {rmse(med_adapt, clean_signal):.4f}")
    
    print("\n" + "=" * 50)
    print("椒盐噪声抑制效果对比:")
    print("=" * 50)
    methods2 = ["Noisy", "SMA", "SG", "Median", "AdaptMed"]
    rmses = [
        rmse(noisy_signal, clean_signal),
        rmse(sma_noise, clean_signal),
        rmse(sg_noise, clean_signal),
        rmse(med_noise, clean_signal),
        rmse(med_adapt, clean_signal),
    ]
    max_rmse = max(rmses)
    for m, e in zip(methods2, rmses):
        bar_len = int((1 - e / max_rmse) * 40)
        bar = "█" * bar_len
        print(f"  {m:9s}: {e:.4f}  {bar}")
    
    print("\n" + "=" * 60)
    print("测试3: 动态系统跟踪对比")
    print("=" * 60)
    t3 = np.linspace(0, 20, 200)
    true_state = np.zeros_like(t3)
    true_state[:50] = 2.0
    true_state[50:100] = 2.0 + 0.1 * (t3[50:100] - 5)
    true_state[100:150] = 7.0 - 0.05 * (t3[100:150] - 10) ** 2
    true_state[150:] = 3.0 + np.sin(t3[150:] * 2)
    
    dyn_signal = true_state + 0.5 * np.random.randn(200)
    
    print(f"\n信号特征: 阶跃→斜坡→抛物线→正弦的动态变化\n")
    
    sma_dyn = simple_moving_average(dyn_signal, 7)
    sg_dyn = savitzky_golay(dyn_signal, 7, auto_order=True)
    kf_dyn = kalman_filter(dyn_signal, adaptive=True)
    kf_fixed = kalman_filter(dyn_signal, Q=0.01, R=0.25, adaptive=False)
    
    print("跟踪RMSE (越小越好):")
    print(f"  移动平均:    {rmse(sma_dyn, true_state):.4f}")
    print(f"  SG滤波:      {rmse(sg_dyn, true_state):.4f}")
    print(f"  卡尔曼(固定): {rmse(kf_fixed, true_state):.4f}")
    print(f"  卡尔曼(自适应): {rmse(kf_dyn, true_state):.4f}")
    
    kf_improve = (rmse(kf_fixed, true_state) - rmse(kf_dyn, true_state)) / rmse(kf_fixed, true_state) * 100
    print(f"\n自适应卡尔曼相比固定参数提升: {kf_improve:.1f}%")
    
    print("\n" + "=" * 60)
    print("测试4: 快速数值示例")
    print("=" * 60)
    simple_signal = [3, 5, 7, 100, 8, 10, 9, -50, 13, 12]
    print("\n原始信号 (含离群点):", simple_signal)
    print("\n简单移动平均 (窗口=3):")
    print(simple_moving_average(simple_signal, 3).round(2))
    print("\n中值滤波 (窗口=3):")
    print(median_filter(simple_signal, window_size=3, adaptive=False).round(2))
    print("\n卡尔曼滤波:")
    print(kalman_filter(simple_signal, adaptive=True).round(2))
