import numpy as np
from scipy.signal import find_peaks


def compute_amdf(signal, max_lag=None):
    """
    计算平均幅度差函数 (Average Magnitude Difference Function)
    对周期性信号表现为谷值，对噪声相对不敏感
    
    参数:
        signal: 输入信号
        max_lag: 最大滞后值
    
    返回:
        amdf: AMDF函数值
    """
    n = len(signal)
    if max_lag is None:
        max_lag = n // 2
    
    signal = signal - np.mean(signal)
    amdf = np.zeros(max_lag)
    
    for lag in range(1, max_lag):
        amdf[lag] = np.mean(np.abs(signal[lag:] - signal[:-lag]))
    
    return amdf


def compute_autocorr(signal, normalize=True):
    """
    计算自相关函数
    
    参数:
        signal: 输入信号
        normalize: 是否归一化
    
    返回:
        autocorr: 自相关函数值
    """
    n = len(signal)
    signal = signal - np.mean(signal)
    
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:]
    
    if normalize:
        autocorr = autocorr / autocorr[0]
    
    return autocorr


def parabolic_interpolation(x, y, peak_idx):
    """
    抛物线插值细化峰值位置
    
    参数:
        x: x坐标数组
        y: y坐标数组
        peak_idx: 初始峰值索引
    
    返回:
        refined_x: 细化后的峰值位置
        refined_y: 细化后的峰值
    """
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        return x[peak_idx], y[peak_idx]
    
    left = peak_idx - 1
    mid = peak_idx
    right = peak_idx + 1
    
    denom = (y[left] - 2 * y[mid] + y[right])
    if abs(denom) < 1e-10:
        return x[peak_idx], y[peak_idx]
    
    offset = 0.5 * (y[left] - y[right]) / denom
    refined_x = x[mid] + offset * (x[1] - x[0])
    refined_y = y[mid] - 0.25 * (y[left] - y[right]) * offset
    
    return refined_x, refined_y


def detect_periods_autocorr(signal, min_period=2, max_period=None, threshold=0.3):
    """
    自相关法检测周期
    
    返回:
        periods: 候选周期列表（按置信度排序）
        confidences: 对应的置信度
    """
    n = len(signal)
    if max_period is None:
        max_period = n // 2
    
    autocorr = compute_autocorr(signal)
    peaks, props = find_peaks(autocorr, height=threshold, distance=min_period)
    
    if len(peaks) == 0:
        return [], []
    
    peak_heights = props['peak_heights']
    valid_mask = (peaks >= min_period) & (peaks <= max_period)
    peaks = peaks[valid_mask]
    peak_heights = peak_heights[valid_mask]
    
    if len(peaks) == 0:
        return [], []
    
    periods = []
    confidences = []
    for i, peak in enumerate(peaks):
        refined_period, _ = parabolic_interpolation(
            np.arange(len(autocorr)), autocorr, peak
        )
        periods.append(refined_period)
        confidences.append(peak_heights[i])
    
    sorted_idx = np.argsort(confidences)[::-1]
    return [periods[i] for i in sorted_idx], [confidences[i] for i in sorted_idx]


def detect_periods_amdf(signal, min_period=2, max_period=None, threshold_ratio=0.5):
    """
    AMDF法检测周期（寻找谷值）
    
    返回:
        periods: 候选周期列表（按置信度排序）
        confidences: 对应的置信度
    """
    n = len(signal)
    if max_period is None:
        max_period = n // 2
    
    amdf = compute_amdf(signal, max_period)
    
    amdf_inv = -amdf
    
    threshold = np.min(amdf) + threshold_ratio * (np.median(amdf) - np.min(amdf))
    
    valleys, props = find_peaks(-amdf, height=-threshold, distance=min_period)
    valley_depths = -props['peak_heights']
    
    valid_mask = (valleys >= min_period) & (valleys <= max_period)
    valleys = valleys[valid_mask]
    valley_depths = valley_depths[valid_mask]
    
    if len(valleys) == 0:
        return [], []
    
    periods = []
    confidences = []
    for i, valley in enumerate(valleys):
        refined_period, _ = parabolic_interpolation(
            np.arange(len(amdf)), amdf, valley
        )
        periods.append(max(min_period, refined_period))
        
        norm_depth = 1.0 - (valley_depths[i] - np.min(amdf)) / (np.max(amdf) - np.min(amdf) + 1e-10)
        confidences.append(norm_depth)
    
    sorted_idx = np.argsort(confidences)[::-1]
    return [periods[i] for i in sorted_idx], [confidences[i] for i in sorted_idx]


def detect_periods_fft(signal, fs=1.0, min_period=None, max_period=None):
    """
    FFT法检测周期（通过谐波验证）
    
    返回:
        periods: 候选周期列表（按置信度排序）
        confidences: 对应的置信度
    """
    n = len(signal)
    signal_norm = signal - np.mean(signal)
    
    fft_vals = np.fft.fft(signal_norm * np.hanning(n))
    fft_mag = np.abs(fft_vals[:n//2])
    freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
    
    peaks, props = find_peaks(fft_mag, height=np.max(fft_mag) * 0.1)
    
    if len(peaks) == 0:
        return [], []
    
    peak_freqs = freqs[peaks]
    peak_heights = props['peak_heights']
    
    sorted_idx = np.argsort(peak_heights)[::-1]
    top_peaks = peak_freqs[sorted_idx[:min(10, len(sorted_idx))]]
    top_heights = peak_heights[sorted_idx[:min(10, len(sorted_idx))]]
    
    candidates = []
    for i, f_candidate in enumerate(top_peaks):
        if f_candidate <= 0:
            continue
        
        period_candidate = 1.0 / f_candidate
        if min_period is not None and period_candidate < min_period:
            continue
        if max_period is not None and period_candidate > max_period:
            continue
        
        harmonic_score = 0
        for harmonic in range(2, 6):
            expected_freq = f_candidate * harmonic
            nearest_idx = np.argmin(np.abs(peak_freqs - expected_freq))
            nearest_freq = peak_freqs[nearest_idx]
            if abs(nearest_freq - expected_freq) / expected_freq < 0.15:
                harmonic_score += 1
        
        confidence = (harmonic_score / 4.0) * 0.7 + (top_heights[i] / np.max(peak_heights)) * 0.3
        candidates.append((period_candidate, confidence))
    
    candidates.sort(key=lambda x: -x[1])
    periods = [c[0] for c in candidates]
    confidences = [c[1] for c in candidates]
    
    return periods, confidences


def detect_fundamental_period(signal, fs=1.0, method='ensemble', min_period=None, max_period=None):
    """
    鲁棒的基波周期检测（统一接口）
    
    参数:
        signal: 输入信号
        fs: 采样频率
        method: 检测方法
            'autocorr': 自相关法
            'amdf': AMDF法（抗噪能力强）
            'fft': FFT谐波验证法
            'ensemble': 集成法（默认，推荐）
        min_period: 最小周期（采样点数），None则自动确定
        max_period: 最大周期（采样点数），None则自动确定
    
    返回:
        period: 检测到的主周期（采样点数）
        frequency: 检测到的基频 (Hz)
        confidence: 置信度 (0-1)
        all_candidates: 所有候选周期信息
    """
    n = len(signal)
    
    if min_period is None:
        min_period = max(2, n // 100)
    if max_period is None:
        max_period = n // 2
    
    all_candidates = []
    
    if method in ['autocorr', 'ensemble']:
        periods_ac, confs_ac = detect_periods_autocorr(
            signal, min_period, max_period
        )
        for p, c in zip(periods_ac, confs_ac):
            all_candidates.append({'period': p, 'confidence': c, 'method': 'autocorr'})
    
    if method in ['amdf', 'ensemble']:
        periods_amdf, confs_amdf = detect_periods_amdf(
            signal, min_period, max_period
        )
        for p, c in zip(periods_amdf, confs_amdf):
            all_candidates.append({'period': p, 'confidence': c, 'method': 'amdf'})
    
    if method in ['fft', 'ensemble']:
        periods_fft, confs_fft = detect_periods_fft(
            signal, fs, min_period / fs, max_period / fs
        )
        for p, c in zip(periods_fft, confs_fft):
            period_samples = p * fs
            if min_period <= period_samples <= max_period:
                all_candidates.append({'period': period_samples, 'confidence': c, 'method': 'fft'})
    
    if not all_candidates:
        period = max_period / 2
        frequency = fs / period
        return period, frequency, 0.1, []
    
    all_candidates.sort(key=lambda x: -x['confidence'])
    
    if method == 'ensemble' and len(all_candidates) >= 2:
        top_candidates = all_candidates[:min(5, len(all_candidates))]
        
        clusters = []
        for cand in top_candidates:
            p = cand['period']
            matched = False
            for cluster in clusters:
                if abs(p - cluster['mean']) / cluster['mean'] < 0.15:
                    cluster['periods'].append(p)
                    cluster['confidences'].append(cand['confidence'])
                    cluster['count'] += 1
                    cluster['mean'] = np.mean(cluster['periods'])
                    matched = True
                    break
            if not matched:
                clusters.append({
                    'periods': [p],
                    'confidences': [cand['confidence']],
                    'count': 1,
                    'mean': p
                })
        
        cluster_scores = []
        for cluster in clusters:
            score = (np.max(cluster['confidences']) * 0.6 + 
                     min(cluster['count'], 3) / 3.0 * 0.4)
            cluster_scores.append((cluster['mean'], score))
        
        if cluster_scores:
            cluster_scores.sort(key=lambda x: -x[1])
            best_period = cluster_scores[0][0]
            best_confidence = cluster_scores[0][1]
        else:
            best_period = all_candidates[0]['period']
            best_confidence = all_candidates[0]['confidence']
    else:
        best_period = all_candidates[0]['period']
        best_confidence = all_candidates[0]['confidence']
    
    frequency = fs / best_period
    
    return best_period, frequency, min(best_confidence, 1.0), all_candidates


def detect_fundamental_frequency(signal, fs=1.0, method='ensemble', min_period=None, max_period=None):
    """
    鲁棒的基频检测（兼容旧接口）
    
    参数:
        signal: 输入信号
        fs: 采样频率
        method: 检测方法
            'autocorr': 自相关法
            'amdf': AMDF法（抗噪能力强）
            'fft': FFT峰值法
            'ensemble': 集成法（默认，推荐）
        min_period: 最小周期（采样点数）
        max_period: 最大周期（采样点数）
    
    返回:
        f0: 检测到的基频
        confidence: 置信度 (0-1)
    """
    _, f0, confidence, _ = detect_fundamental_period(
        signal, fs, method, min_period, max_period
    )
    return f0, confidence


def fourier_series_fit(signal, n_harmonics, t=None, auto_period=False, fs=1.0, 
                        period_method='ensemble', min_period=None, max_period=None):
    """
    对周期信号进行傅里叶级数拟合，支持自动周期检测
    
    参数:
        signal: 输入的周期信号（离散点）
        n_harmonics: 要拟合的谐波次数（包括直流分量）
        t: 时间点，若为None则假设为均匀采样
        auto_period: 是否自动检测周期（无需用户提供）
        fs: 采样频率（当auto_period=True时需要）
        period_method: 周期检测方法
            'autocorr': 自相关法
            'amdf': AMDF法（抗噪能力强）
            'fft': FFT谐波验证法
            'ensemble': 集成法（默认，推荐）
        min_period: 最小周期（采样点数），None则自动确定
        max_period: 最大周期（采样点数），None则自动确定
    
    返回:
        coeffs: 傅里叶系数，形状为(n_harmonics, 2)，第一列为余弦系数，第二列为正弦系数
        reconstructed: 重构后的信号
        t: 时间点
        period_info: 周期检测信息（当auto_period=True时）
    """
    n_samples = len(signal)
    
    if t is None:
        t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    
    period_info = None
    if auto_period:
        period_samples, f0, confidence, candidates = detect_fundamental_period(
            signal, fs, period_method, min_period, max_period
        )
        period = period_samples / fs
        period_info = {
            'period_samples': period_samples,
            'period_seconds': period,
            'frequency': f0,
            'confidence': confidence,
            'candidates': candidates
        }
    else:
        period = t[-1] - t[0] + (t[1] - t[0])
    
    omega = 2 * np.pi / period
    
    A = np.zeros((n_samples, 2 * n_harmonics))
    
    for k in range(n_harmonics):
        if k == 0:
            A[:, 0] = 1.0
        else:
            A[:, 2*k - 1] = np.cos(k * omega * t)
            A[:, 2*k] = np.sin(k * omega * t)
    
    coeffs_flat, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
    
    coeffs = np.zeros((n_harmonics, 2))
    coeffs[0, 0] = coeffs_flat[0]
    for k in range(1, n_harmonics):
        coeffs[k, 0] = coeffs_flat[2*k - 1]
        coeffs[k, 1] = coeffs_flat[2*k]
    
    reconstructed = np.zeros(n_samples)
    for k in range(n_harmonics):
        if k == 0:
            reconstructed += coeffs[k, 0]
        else:
            reconstructed += coeffs[k, 0] * np.cos(k * omega * t) + \
                             coeffs[k, 1] * np.sin(k * omega * t)
    
    if auto_period:
        return coeffs, reconstructed, t, period_info
    else:
        return coeffs, reconstructed, t


def fft_based_fit(signal, n_harmonics, t=None, auto_period=False, fs=1.0,
                  period_method='ensemble', min_period=None, max_period=None):
    """
    傅里叶级数拟合（支持鲁棒自动周期检测）
    
    参数:
        signal: 输入的周期信号
        n_harmonics: 要保留的谐波次数
        t: 时间点，若为None则假设为均匀采样
        auto_period: 是否自动检测周期
        fs: 采样频率
        period_method: 周期检测方法
        min_period: 最小周期（采样点数）
        max_period: 最大周期（采样点数）
    
    返回:
        coeffs: 傅里叶系数
        reconstructed: 重构信号
        t: 时间点
        period_info: 周期检测信息
    """
    n = len(signal)
    
    if t is None:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    period_info = None
    if auto_period:
        period_samples, f0, confidence, candidates = detect_fundamental_period(
            signal, fs, period_method, min_period, max_period
        )
        period = period_samples / fs
        period_info = {
            'period_samples': period_samples,
            'period_seconds': period,
            'frequency': f0,
            'confidence': confidence,
            'candidates': candidates
        }
    else:
        period = t[-1] - t[0] + (t[1] - t[0])
    
    omega0 = 2 * np.pi / period
    
    A = np.zeros((n, 2 * n_harmonics))
    for k in range(n_harmonics):
        if k == 0:
            A[:, 0] = 1.0
        else:
            A[:, 2*k - 1] = np.cos(k * omega0 * t)
            A[:, 2*k] = np.sin(k * omega0 * t)
    
    coeffs_flat, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
    
    coeffs = np.zeros((n_harmonics, 2))
    coeffs[0, 0] = coeffs_flat[0]
    for k in range(1, n_harmonics):
        coeffs[k, 0] = coeffs_flat[2*k - 1]
        coeffs[k, 1] = coeffs_flat[2*k]
    
    reconstructed = np.zeros(n)
    for k in range(n_harmonics):
        if k == 0:
            reconstructed += coeffs[k, 0]
        else:
            reconstructed += coeffs[k, 0] * np.cos(k * omega0 * t) + \
                             coeffs[k, 1] * np.sin(k * omega0 * t)
    
    if auto_period:
        return coeffs, reconstructed, t, period_info
    else:
        return coeffs, reconstructed, t


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    fs = 100
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    true_f0 = 2.5
    true_period_samples = fs / true_f0
    
    clean_signal = 3 + 2 * np.sin(2 * np.pi * true_f0 * t) + \
                   1.5 * np.cos(2 * np.pi * 2 * true_f0 * t) + \
                   0.5 * np.sin(2 * np.pi * 3 * true_f0 * t)
    
    noise_level = 0.5
    noisy_signal = clean_signal + np.random.normal(0, noise_level, len(t))
    
    print("=" * 70)
    print("自动周期检测测试 (含噪声信号)")
    print("=" * 70)
    print(f"真实周期: {1/true_f0:.4f} s ({true_period_samples:.1f} 采样点)")
    print(f"真实基频: {true_f0:.4f} Hz")
    print(f"噪声水平: {noise_level:.4f}")
    
    methods = ['autocorr', 'amdf', 'fft', 'ensemble']
    results = {}
    
    print("\n" + "-" * 70)
    print(f"{'方法':<12} {'检测周期(s)':<14} {'检测频率(Hz)':<14} {'相对误差(%)':<12} {'置信度':<8}")
    print("-" * 70)
    
    for method in methods:
        period_samples, f0, confidence, _ = detect_fundamental_period(
            noisy_signal, fs=fs, method=method
        )
        period_seconds = period_samples / fs
        rel_error = abs(f0 - true_f0) / true_f0 * 100
        results[method] = (period_samples, f0, confidence)
        print(f"{method:<12} {period_seconds:<14.4f} {f0:<14.4f} {rel_error:<12.2f} {confidence:<8.2f}")
    
    print("-" * 70)
    
    n_harmonics = 5
    coeffs, reconstructed, t_out, period_info = fourier_series_fit(
        noisy_signal, n_harmonics, t, auto_period=True, fs=fs
    )
    
    print("\n" + "=" * 70)
    print("傅里叶系数 (使用集成法自动检测周期):")
    print("=" * 70)
    print(f"检测到的周期: {period_info['period_seconds']:.4f} s "
          f"({period_info['period_samples']:.1f} 采样点)")
    print(f"检测到的基频: {period_info['frequency']:.4f} Hz")
    print(f"检测置信度: {period_info['confidence']:.2f}")
    print()
    print(f"直流分量 (k=0): {coeffs[0, 0]:.4f}")
    for k in range(1, n_harmonics):
        print(f"k={k}: 余弦系数={coeffs[k, 0]:.4f}, 正弦系数={coeffs[k, 1]:.4f}")
    
    mse = np.mean((noisy_signal - reconstructed) ** 2)
    mse_clean = np.mean((clean_signal - reconstructed) ** 2)
    print(f"\n重构均方误差 (相对含噪信号): {mse:.6f}")
    print(f"重构均方误差 (相对纯净信号): {mse_clean:.6f}")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, noisy_signal, 'b-', label='含噪信号', alpha=0.5)
    ax1.plot(t, clean_signal, 'g-', label='纯净信号', linewidth=2, alpha=0.7)
    ax1.plot(t_out, reconstructed, 'r--', label=f'重构信号', linewidth=2)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.legend()
    ax1.set_title(f'信号与傅里叶级数拟合 ({n_harmonics}次谐波)')
    ax1.grid(True)
    
    n = len(noisy_signal)
    autocorr = compute_autocorr(noisy_signal)
    lag_axis = np.arange(n) / fs
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(lag_axis, autocorr, 'b-')
    ax2.axvline(x=1/true_f0, color='g', linestyle='--', label='真实周期')
    ax2.axvline(x=results['autocorr'][0]/fs, color='r', linestyle='--', label='检测周期')
    ax2.set_xlabel('滞后 (s)')
    ax2.set_ylabel('自相关系数')
    ax2.legend()
    ax2.set_title('自相关函数 (ACF)')
    ax2.set_xlim(0, 1.0)
    ax2.grid(True)
    
    amdf = compute_amdf(noisy_signal, max_lag=n//2)
    lag_axis_amdf = np.arange(len(amdf)) / fs
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(lag_axis_amdf, amdf, 'b-')
    ax3.axvline(x=1/true_f0, color='g', linestyle='--', label='真实周期')
    ax3.axvline(x=results['amdf'][0]/fs, color='r', linestyle='--', label='检测周期')
    ax3.set_xlabel('滞后 (s)')
    ax3.set_ylabel('AMDF值')
    ax3.legend()
    ax3.set_title('平均幅度差函数 (AMDF)')
    ax3.set_xlim(0, 1.0)
    ax3.grid(True)
    
    fft_vals = np.fft.fft(noisy_signal * np.hanning(n))
    freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
    fft_mag = np.abs(fft_vals[:n//2])
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(freqs, fft_mag, 'b-')
    ax4.axvline(x=true_f0, color='g', linestyle='--', label='真实基频')
    ax4.axvline(x=results['fft'][1], color='r', linestyle='--', label='检测基频')
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('幅值')
    ax4.legend()
    ax4.set_title('FFT频谱')
    ax4.set_xlim(0, 20)
    ax4.grid(True)
    
    ax5 = fig.add_subplot(gs[2, :])
    x = np.arange(len(methods))
    detected_f0s = [results[m][1] for m in methods]
    confidences = [results[m][2] for m in methods]
    
    bars_f0 = ax5.bar(x - 0.2, detected_f0s, 0.4, label='检测频率', color='skyblue')
    bars_conf = ax5.bar(x + 0.2, confidences, 0.4, label='置信度', color='lightcoral')
    ax5.axhline(y=true_f0, color='g', linestyle='--', linewidth=2, label='真实频率')
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods)
    ax5.set_ylabel('频率 (Hz) / 置信度')
    ax5.legend()
    ax5.set_title('不同周期检测方法对比')
    ax5.grid(True, axis='y')
    
    for bar, f0 in zip(bars_f0, detected_f0s):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{f0:.2f}Hz', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('e:\\temp\\record10\\17\\auto_period_detection.png', dpi=150)
    print("\n检测结果图已保存为 auto_period_detection.png")
