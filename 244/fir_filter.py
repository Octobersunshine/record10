import warnings
import numpy as np
from scipy import signal


def design_fir(
    filter_type,
    freq,
    num_taps,
    sample_rate,
    window='hamming',
    window_param=None,
):
    """
    通用FIR滤波器设计（窗函数法）

    支持低通、高通、带通、带阻四种类型。
    自动将滤波器长度强制为奇数以保证线性相位。

    参数:
        filter_type: 滤波器类型，可选 'lowpass', 'highpass', 'bandpass', 'bandstop'
        freq: 截止频率
              - 低通/高通: 标量 (Hz)
              - 带通/带阻: [f_low, f_high] (Hz)
        num_taps: 滤波器长度（系数个数），将被强制为奇数
        sample_rate: 采样率 (Hz)
        window: 窗函数类型，支持:
            - 固定窗: 'hamming', 'hann', 'bartlett', 'blackman', 'boxcar'
            - 可调节窗: 'kaiser' (需指定 beta), 'chebyshev' (需指定 ripple)
        window_param: 窗函数参数
            - 'kaiser': beta (典型值 5~10，越大旁瓣抑制越强)
            - 'chebyshev': ripple (dB, 正浮点数，最小阻带衰减)

    返回:
        h: 滤波器系数（冲激响应）
        group_delay: 群延迟（整数采样点数）
    """
    valid_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']
    if filter_type not in valid_types:
        raise ValueError(f"filter_type 必须是 {valid_types} 之一，当前为 {filter_type}")

    if num_taps % 2 == 0:
        warnings.warn(
            f"num_taps={num_taps} 为偶数，群延迟 = {(num_taps - 1) / 2} 非整数，"
            f"已自动调整为 num_taps={num_taps + 1} 以确保群延迟为整数。",
            RuntimeWarning,
        )
        num_taps = num_taps + 1

    nyquist = sample_rate / 2.0
    n = np.arange(num_taps)
    alpha = (num_taps - 1) / 2.0
    group_delay = (num_taps - 1) // 2

    if window == 'kaiser':
        if window_param is None:
            window_param = 8.0
        window_func = signal.get_window(('kaiser', window_param), num_taps)
    elif window == 'chebyshev':
        if window_param is None:
            window_param = 60.0
        window_func = signal.get_window(('chebwin', window_param), num_taps)
    else:
        window_func = signal.get_window(window, num_taps)

    if filter_type == 'lowpass':
        fc = freq / nyquist
        h = np.sinc(2 * fc * (n - alpha))

    elif filter_type == 'highpass':
        fc = freq / nyquist
        h = -np.sinc(2 * fc * (n - alpha))
        h[int(alpha)] += 1.0

    elif filter_type == 'bandpass':
        f1, f2 = freq
        f1_norm = f1 / nyquist
        f2_norm = f2 / nyquist
        h = (
            2 * f2_norm * np.sinc(2 * f2_norm * (n - alpha))
            - 2 * f1_norm * np.sinc(2 * f1_norm * (n - alpha))
        )

    elif filter_type == 'bandstop':
        f1, f2 = freq
        f1_norm = f1 / nyquist
        f2_norm = f2 / nyquist
        h = np.sinc(n - alpha) - (
            2 * f2_norm * np.sinc(2 * f2_norm * (n - alpha))
            - 2 * f1_norm * np.sinc(2 * f1_norm * (n - alpha))
        )

    h = h * window_func

    if filter_type in ['lowpass', 'bandpass', 'bandstop']:
        h = h / np.sum(h)

    return h, group_delay


def compute_frequency_response(h, sample_rate, num_points=8000):
    """
    计算滤波器的频率响应

    参数:
        h: 滤波器系数
        sample_rate: 采样率 (Hz)
        num_points: 频率采样点数

    返回:
        freq_hz: 频率轴 (Hz)
        magnitude_db: 幅度响应 (dB)
        phase_rad: 相位响应 (rad, 已解缠)
        group_delay_samples: 群延迟 (采样点数)
    """
    w, H = signal.freqz(h, 1.0, worN=num_points)
    freq_hz = (sample_rate * 0.5 / np.pi) * w
    magnitude_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase_rad = np.unwrap(np.angle(H))

    gd = -np.diff(phase_rad) / np.diff(w)
    group_delay_samples = np.concatenate(([gd[0]], gd))

    return freq_hz, magnitude_db, phase_rad, group_delay_samples


def apply_filter(signal_data, filter_coeffs, compensate_delay=True):
    """
    应用FIR滤波器，可选补偿群延迟

    参数:
        signal_data: 输入信号
        filter_coeffs: 滤波器系数
        compensate_delay: 是否补偿群延迟

    返回:
        filtered_signal: 滤波后的信号
    """
    filtered = signal.lfilter(filter_coeffs, 1.0, signal_data)

    if compensate_delay:
        num_taps = len(filter_coeffs)
        if num_taps % 2 == 1:
            group_delay = (num_taps - 1) // 2
            filtered = filtered[group_delay:]
        else:
            warnings.warn(
                f"滤波器长度为偶数，无法精确补偿相位延迟。",
                RuntimeWarning,
            )

    return filtered


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample_rate = 1000
    num_taps = 61

    filters = [
        ('lowpass', 50, 'Lowpass (fc=50Hz)'),
        ('highpass', 150, 'Highpass (fc=150Hz)'),
        ('bandpass', [60, 140], 'Bandpass (60-140Hz)'),
        ('bandstop', [60, 140], 'Bandstop (60-140Hz)'),
    ]

    windows = [
        ('hamming', None, 'Hamming'),
        ('hann', None, 'Hann'),
        ('blackman', None, 'Blackman'),
        ('kaiser', 8.0, 'Kaiser (beta=8)'),
        ('kaiser', 14.0, 'Kaiser (beta=14)'),
        ('chebyshev', 60.0, 'Chebyshev (ripple=60dB)'),
    ]

    plt.figure(figsize=(16, 12))

    for idx, (f_type, f_val, title) in enumerate(filters):
        plt.subplot(3, 4, idx + 1)
        h, gd = design_fir(f_type, f_val, num_taps, sample_rate, window='hamming')
        freq_hz, mag_db, phase, gd_curve = compute_frequency_response(h, sample_rate)
        plt.plot(freq_hz, mag_db)
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.ylim(-100, 5)
        if f_type in ['lowpass', 'highpass']:
            plt.axvline(f_val, color='r', ls='--')
        else:
            plt.axvline(f_val[0], color='r', ls='--')
            plt.axvline(f_val[1], color='r', ls='--')

        plt.subplot(3, 4, idx + 5)
        plt.stem(h, basefmt='b-')
        plt.title(f'Impulse Response (gd={gd})')
        plt.xlabel('Tap Index')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 4, idx + 9)
        plt.plot(freq_hz, phase)
        ideal_phase = -gd * (2 * np.pi * freq_hz / sample_rate)
        plt.plot(freq_hz, ideal_phase, 'r--', alpha=0.7)
        plt.title('Phase Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (rad)')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('fir_filter_types.png', dpi=100)
    print("Saved fir_filter_types.png")

    plt.figure(figsize=(16, 10))

    for idx, (win_name, win_param, label) in enumerate(windows):
        h, gd = design_fir('lowpass', 50, num_taps, sample_rate, window=win_name, window_param=win_param)
        freq_hz, mag_db, phase, _ = compute_frequency_response(h, sample_rate)
        plt.subplot(2, 3, idx + 1)
        plt.plot(freq_hz, mag_db, label=label)
        plt.axvline(50, color='r', ls='--')
        plt.title(label)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.grid(True)
        plt.ylim(-100, 5)

    plt.tight_layout()
    plt.savefig('fir_window_comparison.png', dpi=100)
    print("Saved fir_window_comparison.png")

    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sig_30hz = np.sin(2 * np.pi * 30 * t)
    sig_100hz = 0.5 * np.sin(2 * np.pi * 100 * t)
    sig_200hz = 0.3 * np.sin(2 * np.pi * 200 * t)
    mixed = sig_30hz + sig_100hz + sig_200hz

    h_bp, gd_bp = design_fir('bandpass', [60, 140], num_taps, sample_rate)
    filtered_bp = apply_filter(mixed, h_bp, compensate_delay=True)
    t_comp = t[gd_bp:]

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, mixed, label='Original (30+100+200 Hz)', alpha=0.5)
    plt.plot(t_comp, filtered_bp, label='Bandpass Filtered (~100Hz)', linewidth=2)
    plt.title('Bandpass Filter: Extract 100Hz Component')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    n = len(mixed)
    fft_orig = np.abs(np.fft.rfft(mixed))
    fft_filt = np.abs(np.fft.rfft(filtered_bp, n=n))
    freqs_fft = np.fft.rfftfreq(n, 1/sample_rate)
    plt.plot(freqs_fft, 20*np.log10(fft_orig/len(mixed)), label='Original Spectrum', alpha=0.5)
    plt.plot(freqs_fft, 20*np.log10(fft_filt/len(filtered_bp)), label='Filtered Spectrum')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 300)

    plt.tight_layout()
    plt.savefig('fir_bandpass_demo.png', dpi=100)
    print("Saved fir_bandpass_demo.png")
