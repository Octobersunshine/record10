import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt


def estimate_noise_mcra(mag, sr, hop_length, init_noise_frames=10, alpha_s=0.9, 
                        alpha_p=0.95, delta=0.05, L=20):
    """
    MCRA (Minima Controlled Recursive Averaging) 自适应噪声估计算法
    
    参数:
        mag: 幅度谱 [n_freq, n_frames]
        sr: 采样率
        hop_length: 帧移
        init_noise_frames: 初始噪声估计帧数
        alpha_s: 语音存在时的平滑系数
        alpha_p: 语音不存在时的平滑系数
        delta: 阈值参数
        L: 最小值搜索窗口长度
    
    返回:
        noise_mag: 估计的噪声幅度谱 [n_freq, n_frames]
    """
    n_freq, n_frames = mag.shape
    noise_mag = np.zeros_like(mag)
    
    noise_mag[:, 0] = np.mean(mag[:, :init_noise_frames], axis=1)
    
    P_min = np.copy(mag[:, 0])
    S = np.copy(mag[:, 0])
    
    for n in range(1, n_frames):
        S = alpha_s * S + (1 - alpha_s) * mag[:, n]
        
        if n >= L:
            P_min = np.min(mag[:, n-L:n], axis=1)
        else:
            P_min = np.min(mag[:, :n+1], axis=1)
        
        I = (S < (1 + delta) * P_min).astype(float)
        
        alpha_n = alpha_p + (1 - alpha_p) * I
        
        noise_mag[:, n] = alpha_n * noise_mag[:, n-1] + (1 - alpha_n) * mag[:, n]
    
    return noise_mag


def estimate_noise_vad(mag, sr, hop_length, init_noise_frames=10, 
                       energy_threshold=0.1, alpha_update=0.95):
    """
    基于VAD（语音活动检测）的自适应噪声估计
    
    参数:
        mag: 幅度谱 [n_freq, n_frames]
        energy_threshold: 能量阈值
        alpha_update: 噪声更新平滑系数
    
    返回:
        noise_mag: 估计的噪声幅度谱 [n_freq, n_frames]
    """
    n_freq, n_frames = mag.shape
    noise_mag = np.zeros_like(mag)
    
    noise_mag[:, 0] = np.mean(mag[:, :init_noise_frames], axis=1)
    
    energy = np.sum(mag ** 2, axis=0)
    energy = energy / np.max(energy)
    
    for n in range(1, n_frames):
        if energy[n] < energy_threshold:
            noise_mag[:, n] = alpha_update * noise_mag[:, n-1] + \
                              (1 - alpha_update) * mag[:, n]
        else:
            noise_mag[:, n] = noise_mag[:, n-1]
    
    return noise_mag


def estimate_snr(mag, noise_mag):
    """
    估计信噪比(SNR)
    
    参数:
        mag: 带噪语音幅度谱
        noise_mag: 噪声幅度谱
    
    返回:
        avg_snr: 平均信噪比 (dB)
        snr_per_frame: 每帧信噪比 (dB)
    """
    if noise_mag.ndim == 2 and noise_mag.shape[1] == mag.shape[1]:
        noise_power = noise_mag ** 2
    else:
        if noise_mag.ndim == 2 and noise_mag.shape[1] == 1:
            noise_power = (noise_mag ** 2)
        else:
            noise_power = (noise_mag.reshape(-1, 1) ** 2)
    
    signal_power = mag ** 2
    
    snr_per_bin = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
    avg_snr = np.mean(snr_per_bin)
    
    return avg_snr, snr_per_bin


def wiener_filter(audio_path, output_path=None, noise_estimation_duration=0.5,
                  n_fft=2048, hop_length=512, noise_estimation_method='mcra',
                  alpha_wiener=0.95, beta_wiener=0.05):
    """
    维纳滤波语音增强
    
    参数:
        audio_path: 输入带噪语音路径或numpy数组
        output_path: 输出增强后语音路径（可选）
        noise_estimation_duration: 用于估计噪声的前导静音时长（秒）
        n_fft: FFT窗口大小
        hop_length: 帧移
        noise_estimation_method: 噪声估计方法 ('fixed', 'mcra', 'vad')
        alpha_wiener: 维纳增益平滑系数
        beta_wiener: 最小增益（谱下限）
    
    返回:
        enhanced_y: 增强后的语音信号
        sr: 采样率
    """
    if isinstance(audio_path, str):
        y, sr = librosa.load(audio_path, sr=None)
    else:
        y = audio_path
        sr = 16000
    
    win_length = n_fft
    
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)
    phase = np.angle(D)
    
    noise_frames = max(1, int(noise_estimation_duration * sr / hop_length))
    
    if noise_estimation_method == 'fixed':
        noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    elif noise_estimation_method == 'mcra':
        noise_mag = estimate_noise_mcra(mag, sr, hop_length, 
                                        init_noise_frames=noise_frames)
    elif noise_estimation_method == 'vad':
        noise_mag = estimate_noise_vad(mag, sr, hop_length, 
                                       init_noise_frames=noise_frames)
    else:
        raise ValueError(f"未知的噪声估计方法: {noise_estimation_method}")
    
    if noise_mag.ndim == 2 and noise_mag.shape[1] == 1:
        noise_power = noise_mag ** 2
    else:
        noise_power = noise_mag ** 2
    
    noisy_power = mag ** 2
    
    post_snr = noisy_power / (noise_power + 1e-10)
    
    wiener_gain = post_snr / (post_snr + 1)
    wiener_gain = np.maximum(wiener_gain, beta_wiener)
    
    if wiener_gain.shape[1] > 1:
        for n in range(1, wiener_gain.shape[1]):
            wiener_gain[:, n] = alpha_wiener * wiener_gain[:, n-1] + \
                                (1 - alpha_wiener) * wiener_gain[:, n]
    
    enhanced_mag = mag * wiener_gain
    
    enhanced_D = enhanced_mag * np.exp(1j * phase)
    enhanced_y = librosa.istft(enhanced_D, hop_length=hop_length, win_length=win_length)
    
    if output_path is not None:
        sf.write(output_path, enhanced_y, sr)
    
    return enhanced_y, sr


def auto_enhance(audio_path, output_path=None, noise_estimation_duration=0.5,
                 n_fft=2048, hop_length=512, noise_estimation_method='mcra',
                 snr_threshold=5.0):
    """
    自动选择增强方法：根据估计的信噪比选择谱减法或维纳滤波
    
    参数:
        audio_path: 输入带噪语音路径或numpy数组
        output_path: 输出增强后语音路径（可选）
        noise_estimation_duration: 用于估计噪声的前导静音时长（秒）
        n_fft: FFT窗口大小
        hop_length: 帧移
        noise_estimation_method: 噪声估计方法 ('fixed', 'mcra', 'vad')
        snr_threshold: SNR阈值(dB)，低于此阈值使用维纳滤波，否则使用谱减法
    
    返回:
        enhanced_y: 增强后的语音信号
        sr: 采样率
        method_used: 使用的增强方法 ('spectral_subtraction' or 'wiener_filter')
        estimated_snr: 估计的信噪比 (dB)
    """
    if isinstance(audio_path, str):
        y, sr = librosa.load(audio_path, sr=None)
    else:
        y = audio_path
        sr = 16000
    
    win_length = n_fft
    
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)
    
    noise_frames = max(1, int(noise_estimation_duration * sr / hop_length))
    
    if noise_estimation_method == 'fixed':
        noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    elif noise_estimation_method == 'mcra':
        noise_mag = estimate_noise_mcra(mag, sr, hop_length, 
                                        init_noise_frames=noise_frames)
    elif noise_estimation_method == 'vad':
        noise_mag = estimate_noise_vad(mag, sr, hop_length, 
                                       init_noise_frames=noise_frames)
    else:
        raise ValueError(f"未知的噪声估计方法: {noise_estimation_method}")
    
    estimated_snr, _ = estimate_snr(mag, noise_mag)
    
    print(f"估计的信噪比: {estimated_snr:.2f} dB")
    
    if estimated_snr < snr_threshold:
        print(f"SNR < {snr_threshold} dB，使用维纳滤波")
        method_used = 'wiener_filter'
        enhanced_y, sr = wiener_filter(y, output_path, 
                                       noise_estimation_duration=noise_estimation_duration,
                                       n_fft=n_fft, hop_length=hop_length,
                                       noise_estimation_method=noise_estimation_method)
    else:
        print(f"SNR >= {snr_threshold} dB，使用谱减法")
        method_used = 'spectral_subtraction'
        enhanced_y, sr = spectral_subtraction(y, output_path,
                                              noise_estimation_duration=noise_estimation_duration,
                                              reduction_factor=1.2,
                                              n_fft=n_fft, hop_length=hop_length,
                                              noise_estimation_method=noise_estimation_method)
    
    return enhanced_y, sr, method_used, estimated_snr


def spectral_subtraction(audio_path, output_path=None, noise_estimation_duration=0.5, 
                         reduction_factor=1.0, n_fft=2048, hop_length=512, 
                         noise_estimation_method='fixed'):
    """
    谱减法语音增强（支持多种噪声估计方法）
    
    参数:
        audio_path: 输入带噪语音路径或numpy数组
        output_path: 输出增强后语音路径（可选）
        noise_estimation_duration: 用于估计噪声的前导静音时长（秒）
        reduction_factor: 降噪强度因子
        n_fft: FFT窗口大小
        hop_length: 帧移
        noise_estimation_method: 噪声估计方法
            - 'fixed': 固定噪声估计（前导静音段）
            - 'mcra': MCRA自适应噪声估计
            - 'vad': VAD-based自适应噪声估计
    
    返回:
        enhanced_y: 增强后的语音信号
        sr: 采样率
    """
    if isinstance(audio_path, str):
        y, sr = librosa.load(audio_path, sr=None)
    else:
        y = audio_path
        sr = 16000
    
    win_length = n_fft
    
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)
    phase = np.angle(D)
    
    noise_frames = max(1, int(noise_estimation_duration * sr / hop_length))
    
    if noise_estimation_method == 'fixed':
        noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    elif noise_estimation_method == 'mcra':
        noise_mag = estimate_noise_mcra(mag, sr, hop_length, 
                                        init_noise_frames=noise_frames)
    elif noise_estimation_method == 'vad':
        noise_mag = estimate_noise_vad(mag, sr, hop_length, 
                                       init_noise_frames=noise_frames)
    else:
        raise ValueError(f"未知的噪声估计方法: {noise_estimation_method}")
    
    enhanced_mag = np.maximum(mag - reduction_factor * noise_mag, 0)
    
    enhanced_D = enhanced_mag * np.exp(1j * phase)
    enhanced_y = librosa.istft(enhanced_D, hop_length=hop_length, win_length=win_length)
    
    if output_path is not None:
        sf.write(output_path, enhanced_y, sr)
    
    return enhanced_y, sr


def adaptive_gain_spectral_subtraction(audio_path, output_path=None, 
                                        noise_estimation_duration=0.5,
                                        n_fft=2048, hop_length=512,
                                        beta=0.05, gamma=1.0):
    """
    改进的自适应增益谱减法 - 使用后验信噪比计算增益
    
    参数:
        beta: 最小增益（谱下限）
        gamma: 增益调整因子
    
    算法原理:
        G(k) = 1 - sqrt(alpha / post_SNR(k))
        其中 alpha 是过减因子，根据后验SNR自适应调整
    """
    if isinstance(audio_path, str):
        y, sr = librosa.load(audio_path, sr=None)
    else:
        y = audio_path
        sr = 16000
    
    win_length = n_fft
    
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)
    phase = np.angle(D)
    
    noise_frames = max(1, int(noise_estimation_duration * sr / hop_length))
    noise_mag = estimate_noise_mcra(mag, sr, hop_length, 
                                    init_noise_frames=noise_frames)
    
    post_snr = (mag ** 2) / (noise_mag ** 2 + 1e-10)
    
    alpha = np.where(post_snr < 1, 4.0, 
                     np.where(post_snr < 3, 3.0, 
                              np.where(post_snr < 10, 2.0, 1.0)))
    
    gain = np.maximum(1 - np.sqrt(alpha / (post_snr + 1e-10)), beta)
    
    gain = gain ** gamma
    
    enhanced_mag = gain * mag
    
    enhanced_D = enhanced_mag * np.exp(1j * phase)
    enhanced_y = librosa.istft(enhanced_D, hop_length=hop_length, win_length=win_length)
    
    if output_path is not None:
        sf.write(output_path, enhanced_y, sr)
    
    return enhanced_y, sr


def multi_band_spectral_subtraction(audio_path, output_path=None, noise_estimation_duration=0.5,
                                     reduction_factor=1.0, n_fft=2048, hop_length=512, n_bands=4,
                                     noise_estimation_method='mcra'):
    """
    多带谱减法，在不同频带使用不同的降噪强度（支持自适应噪声估计）
    
    参数:
        n_bands: 频带数量
        noise_estimation_method: 噪声估计方法 ('fixed', 'mcra', 'vad')
    """
    if isinstance(audio_path, str):
        y, sr = librosa.load(audio_path, sr=None)
    else:
        y = audio_path
        sr = 16000
    
    win_length = n_fft
    
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)
    phase = np.angle(D)
    
    noise_frames = max(1, int(noise_estimation_duration * sr / hop_length))
    
    if noise_estimation_method == 'fixed':
        noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    elif noise_estimation_method == 'mcra':
        noise_mag = estimate_noise_mcra(mag, sr, hop_length, 
                                        init_noise_frames=noise_frames)
    elif noise_estimation_method == 'vad':
        noise_mag = estimate_noise_vad(mag, sr, hop_length, 
                                       init_noise_frames=noise_frames)
    else:
        raise ValueError(f"未知的噪声估计方法: {noise_estimation_method}")
    
    n_freq_bins = mag.shape[0]
    band_size = n_freq_bins // n_bands
    
    enhanced_mag = np.zeros_like(mag)
    for i in range(n_bands):
        start = i * band_size
        end = (i + 1) * band_size if i < n_bands - 1 else n_freq_bins
        band_reduction = reduction_factor * (1.0 - i * 0.1)
        enhanced_mag[start:end, :] = np.maximum(mag[start:end, :] - band_reduction * noise_mag[start:end, :], 0)
    
    enhanced_D = enhanced_mag * np.exp(1j * phase)
    enhanced_y = librosa.istft(enhanced_D, hop_length=hop_length, win_length=win_length)
    
    if output_path is not None:
        sf.write(output_path, enhanced_y, sr)
    
    return enhanced_y, sr


def add_noise(signal, snr_db, noise_type='white'):
    """
    向信号中添加噪声
    
    参数:
        signal: 纯净语音信号
        snr_db: 信噪比（dB）
        noise_type: 噪声类型 ('white', 'pink', 'street')
    """
    signal_power = np.mean(signal ** 2)
    
    if noise_type == 'white':
        noise = np.random.randn(len(signal))
    elif noise_type == 'pink':
        noise = np.random.randn(len(signal))
        noise = np.convolve(noise, np.exp(-np.arange(100) / 20), mode='same')
    elif noise_type == 'street':
        noise = generate_nonstationary_noise(len(signal))
    else:
        noise = np.random.randn(len(signal))
    
    noise_power = np.mean(noise ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_scaling = np.sqrt(signal_power / (noise_power * snr_linear))
    noise = noise * noise_scaling
    
    noisy_signal = signal + noise
    return noisy_signal


def generate_nonstationary_noise(length, sr=16000):
    """
    生成非平稳噪声（模拟街道噪声）
    
    参数:
        length: 噪声长度
        sr: 采样率
    
    返回:
        非平稳噪声信号
    """
    t = np.arange(length) / sr
    
    noise = np.random.randn(length)
    
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    envelope += 0.3 * np.sin(2 * np.pi * 2 * t)
    envelope = np.maximum(envelope, 0.2)
    
    noise = noise * envelope
    
    b = np.exp(-np.arange(50) / 10)
    b = b / np.sum(b)
    noise = np.convolve(noise, b, mode='same')
    
    return noise


def plot_comparison(clean, noisy, enhanced, sr, title='语音增强对比'):
    """绘制波形和频谱对比图"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    time = np.arange(len(clean)) / sr
    
    axes[0, 0].plot(time, clean)
    axes[0, 0].set_title('纯净语音波形')
    axes[0, 0].set_ylabel('幅度')
    
    D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean)), ref=np.max)
    img = librosa.display.specshow(D_clean, sr=sr, ax=axes[0, 1], x_axis='time', y_axis='hz')
    axes[0, 1].set_title('纯净语音频谱')
    plt.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')
    
    axes[1, 0].plot(time, noisy)
    axes[1, 0].set_title('带噪语音波形')
    axes[1, 0].set_ylabel('幅度')
    
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy)), ref=np.max)
    img = librosa.display.specshow(D_noisy, sr=sr, ax=axes[1, 1], x_axis='time', y_axis='hz')
    axes[1, 1].set_title('带噪语音频谱')
    plt.colorbar(img, ax=axes[1, 1], format='%+2.0f dB')
    
    axes[2, 0].plot(time, enhanced)
    axes[2, 0].set_title('增强后语音波形')
    axes[2, 0].set_xlabel('时间 (秒)')
    axes[2, 0].set_ylabel('幅度')
    
    D_enhanced = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced)), ref=np.max)
    img = librosa.display.specshow(D_enhanced, sr=sr, ax=axes[2, 1], x_axis='time', y_axis='hz')
    axes[2, 1].set_title('增强后语音频谱')
    axes[2, 1].set_xlabel('时间 (秒)')
    plt.colorbar(img, ax=axes[2, 1], format='%+2.0f dB')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('spectral_subtraction_comparison.png', dpi=150)
    plt.close()


def test_noise_estimation_methods():
    """对比不同噪声估计方法在非平稳噪声下的效果"""
    print("=== 非平稳噪声下的噪声估计对比测试 ===")
    
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    clean = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-1 * t)
    clean = clean + 0.3 * np.sin(2 * np.pi * 880 * t) * np.exp(-0.5 * t)
    clean = clean + 0.2 * np.sin(2 * np.pi * 1320 * t) * np.exp(-0.3 * t)
    
    noisy = add_noise(clean, snr_db=5, noise_type='street')
    
    print(f"纯净语音长度: {len(clean)} 采样点, 采样率: {sr} Hz")
    print(f"添加非平稳街道噪声, SNR = 5 dB")
    
    print("\n1. 使用固定噪声估计...")
    enhanced_fixed, _ = spectral_subtraction(noisy, noise_estimation_duration=0.2, 
                                             reduction_factor=1.0, noise_estimation_method='fixed')
    
    print("2. 使用MCRA自适应噪声估计...")
    enhanced_mcra, _ = spectral_subtraction(noisy, noise_estimation_duration=0.2, 
                                            reduction_factor=1.0, noise_estimation_method='mcra')
    
    print("3. 使用VAD-based自适应噪声估计...")
    enhanced_vad, _ = spectral_subtraction(noisy, noise_estimation_duration=0.2, 
                                           reduction_factor=1.0, noise_estimation_method='vad')
    
    print("4. 使用自适应增益谱减法...")
    enhanced_adaptive, _ = adaptive_gain_spectral_subtraction(noisy, noise_estimation_duration=0.2)
    
    print("5. 使用维纳滤波...")
    enhanced_wiener, _ = wiener_filter(noisy, noise_estimation_duration=0.2)
    
    sf.write('test_clean.wav', clean, sr)
    sf.write('test_noisy_street.wav', noisy, sr)
    sf.write('test_enhanced_fixed.wav', enhanced_fixed, sr)
    sf.write('test_enhanced_mcra.wav', enhanced_mcra, sr)
    sf.write('test_enhanced_vad.wav', enhanced_vad, sr)
    sf.write('test_enhanced_adaptive.wav', enhanced_adaptive, sr)
    sf.write('test_enhanced_wiener.wav', enhanced_wiener, sr)
    
    print("\n已生成测试文件:")
    print("  - test_clean.wav (纯净语音)")
    print("  - test_noisy_street.wav (带街道噪声的语音)")
    print("  - test_enhanced_fixed.wav (固定噪声估计)")
    print("  - test_enhanced_mcra.wav (MCRA自适应噪声估计)")
    print("  - test_enhanced_vad.wav (VAD-based自适应噪声估计)")
    print("  - test_enhanced_adaptive.wav (自适应增益谱减法)")
    print("  - test_enhanced_wiener.wav (维纳滤波)")
    
    plot_noise_estimation_comparison(clean, noisy, enhanced_fixed, enhanced_mcra, 
                                     enhanced_vad, enhanced_adaptive, enhanced_wiener, sr)
    print("\n对比图已保存到: noise_estimation_comparison.png")
    
    print("\n测试完成！")


def test_auto_enhance():
    """测试自动选择增强方法功能"""
    print("\n=== 自动选择增强方法测试 ===")
    
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    clean = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-1 * t)
    clean = clean + 0.3 * np.sin(2 * np.pi * 880 * t) * np.exp(-0.5 * t)
    
    print("\n--- 测试1: 高信噪比场景 (SNR = 10 dB) ---")
    noisy_high_snr = add_noise(clean, snr_db=10, noise_type='white')
    enhanced_high, sr_used, method_high, snr_est_high = auto_enhance(
        noisy_high_snr, 'test_auto_high_snr.wav', snr_threshold=5.0)
    print(f"实际SNR: 10 dB, 估计SNR: {snr_est_high:.2f} dB, 使用方法: {method_high}")
    
    print("\n--- 测试2: 低信噪比场景 (SNR = 0 dB) ---")
    noisy_low_snr = add_noise(clean, snr_db=0, noise_type='white')
    enhanced_low, sr_used, method_low, snr_est_low = auto_enhance(
        noisy_low_snr, 'test_auto_low_snr.wav', snr_threshold=5.0)
    print(f"实际SNR: 0 dB, 估计SNR: {snr_est_low:.2f} dB, 使用方法: {method_low}")
    
    print("\n--- 测试3: 非平稳噪声场景 (SNR = 3 dB, 街道噪声) ---")
    noisy_street = add_noise(clean, snr_db=3, noise_type='street')
    enhanced_street, sr_used, method_street, snr_est_street = auto_enhance(
        noisy_street, 'test_auto_street.wav', snr_threshold=5.0)
    print(f"实际SNR: 3 dB, 估计SNR: {snr_est_street:.2f} dB, 使用方法: {method_street}")
    
    print("\n自动选择测试完成！")
    print("生成的文件:")
    print("  - test_auto_high_snr.wav (高信噪比)")
    print("  - test_auto_low_snr.wav (低信噪比)")
    print("  - test_auto_street.wav (非平稳噪声)")


def plot_noise_estimation_comparison(clean, noisy, enhanced_fixed, enhanced_mcra, 
                                     enhanced_vad, enhanced_adaptive, enhanced_wiener, sr):
    """绘制不同噪声估计方法的对比图"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    time = np.arange(len(clean)) / sr
    
    axes[0, 0].plot(time, noisy, alpha=0.5, label='带噪')
    axes[0, 0].plot(time, enhanced_fixed, label='固定噪声估计')
    axes[0, 0].set_title('固定噪声估计 vs 带噪语音')
    axes[0, 0].legend()
    axes[0, 0].set_ylabel('幅度')
    
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy)), ref=np.max)
    D_fixed = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_fixed)), ref=np.max)
    img = librosa.display.specshow(D_fixed - D_noisy, sr=sr, ax=axes[0, 1], 
                                   x_axis='time', y_axis='hz')
    axes[0, 1].set_title('固定噪声估计 - 频谱差 (dB)')
    plt.colorbar(img, ax=axes[0, 1])
    
    axes[1, 0].plot(time, noisy, alpha=0.5, label='带噪')
    axes[1, 0].plot(time, enhanced_mcra, label='MCRA自适应谱减')
    axes[1, 0].set_title('MCRA自适应谱减 vs 带噪语音')
    axes[1, 0].legend()
    axes[1, 0].set_ylabel('幅度')
    
    D_mcra = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_mcra)), ref=np.max)
    img = librosa.display.specshow(D_mcra - D_noisy, sr=sr, ax=axes[1, 1], 
                                   x_axis='time', y_axis='hz')
    axes[1, 1].set_title('MCRA自适应谱减 - 频谱差 (dB)')
    plt.colorbar(img, ax=axes[1, 1])
    
    axes[2, 0].plot(time, noisy, alpha=0.5, label='带噪')
    axes[2, 0].plot(time, enhanced_wiener, label='维纳滤波')
    axes[2, 0].set_title('维纳滤波 vs 带噪语音')
    axes[2, 0].legend()
    axes[2, 0].set_xlabel('时间 (秒)')
    axes[2, 0].set_ylabel('幅度')
    
    D_wiener = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_wiener)), ref=np.max)
    img = librosa.display.specshow(D_wiener - D_noisy, sr=sr, ax=axes[2, 1], 
                                   x_axis='time', y_axis='hz')
    axes[2, 1].set_title('维纳滤波 - 频谱差 (dB)')
    axes[2, 1].set_xlabel('时间 (秒)')
    plt.colorbar(img, ax=axes[2, 1])
    
    plt.suptitle('非平稳噪声下不同增强方法对比 (谱减 vs 维纳滤波)', fontsize=14)
    plt.tight_layout()
    plt.savefig('noise_estimation_comparison.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1:
        print("无参数输入，运行自适应噪声估计测试模式...")
        test_noise_estimation_methods()
        test_auto_enhance()
    elif len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        
        method = 'auto'
        reduction_factor = 1.0
        noise_method = 'mcra'
        snr_threshold = 5.0
        
        if len(sys.argv) >= 4:
            method = sys.argv[3].lower()
        if len(sys.argv) >= 5:
            if method in ['spectral', 'wiener']:
                reduction_factor = float(sys.argv[4])
            else:
                snr_threshold = float(sys.argv[4])
        if len(sys.argv) >= 6:
            noise_method = sys.argv[5]
        
        print(f"正在处理: {input_path}")
        print(f"增强模式: {method}")
        
        if method == 'auto':
            print(f"SNR阈值: {snr_threshold} dB")
            enhanced, sr, method_used, estimated_snr = auto_enhance(
                input_path, output_path,
                noise_estimation_method=noise_method,
                snr_threshold=snr_threshold)
            print(f"估计SNR: {estimated_snr:.2f} dB, 使用方法: {method_used}")
        elif method == 'spectral':
            print(f"降噪强度: {reduction_factor}")
            print(f"噪声估计方法: {noise_method}")
            enhanced, sr = spectral_subtraction(
                input_path, output_path,
                reduction_factor=reduction_factor,
                noise_estimation_method=noise_method)
        elif method == 'wiener':
            print(f"噪声估计方法: {noise_method}")
            enhanced, sr = wiener_filter(
                input_path, output_path,
                noise_estimation_method=noise_method)
        else:
            print(f"未知的增强方法: {method}")
            sys.exit(1)
        
        print(f"处理完成，输出已保存到: {output_path}")
    else:
        print("使用方法:")
        print("  运行测试: python spectral_subtraction.py")
        print("")
        print("  处理文件:")
        print("    自动选择方法: python spectral_subtraction.py <输入> <输出> auto [SNR阈值] [噪声估计方法]")
        print("    强制谱减法: python spectral_subtraction.py <输入> <输出> spectral [降噪强度] [噪声估计方法]")
        print("    强制维纳滤波: python spectral_subtraction.py <输入> <输出> wiener [降噪强度] [噪声估计方法]")
        print("")
        print("  噪声估计方法: 'fixed', 'mcra', 'vad'")
        print("  示例:")
        print("    python spectral_subtraction.py noisy.wav enhanced.wav auto 5.0 mcra")
        print("    python spectral_subtraction.py noisy.wav enhanced.wav spectral 1.2 mcra")
        print("    python spectral_subtraction.py noisy.wav enhanced.wav wiener")
