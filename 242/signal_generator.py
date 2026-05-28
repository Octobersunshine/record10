import numpy as np


def generate_signal(wave_type, frequency, sample_rate, amplitude, phase, duration,
                    window=None, auto_period=False, snr=None,
                    carrier_freq=None, mod_freq=None, mod_index=1.0,
                    freq_start=None, freq_end=None, chirp_method='linear'):
    """
    生成常用数字信号。

    参数:
        wave_type (str): 波形类型，可选值：
            'sine'（正弦波）, 'square'（方波）, 'triangle'（三角波）, 'sawtooth'（锯齿波）,
            'am'（调幅波）, 'fm'（调频波）, 'chirp'（扫频信号）
        frequency (float): 信号频率（Hz），对于AM/FM是调制信号频率，可被mod_freq覆盖
        sample_rate (float): 采样率（Hz）
        amplitude (float): 信号幅度
        phase (float): 初始相位（弧度）
        duration (float): 信号持续时间（秒）
        window (str, optional): 加窗类型，可选值：None（不加窗）, 'hann'（汉宁窗）, 'flattop'（平顶窗）
        auto_period (bool): 是否自动调整为整数周期数，避免频谱泄露。默认为 False
        snr (float, optional): 信噪比（dB），添加高斯白噪声。None表示不加噪声
        carrier_freq (float, optional): 载波频率（Hz），用于AM/FM信号
        mod_freq (float, optional): 调制信号频率（Hz），用于AM/FM信号
        mod_index (float): 调制度/调制指数，用于AM/FM信号，默认1.0
            AM: 调制度 0~1，FM: 调制指数（频偏/调制频率）
        freq_start (float, optional): 扫频起始频率（Hz），用于chirp信号
        freq_end (float, optional): 扫频终止频率（Hz），用于chirp信号
        chirp_method (str): 扫频方式，'linear'（线性）或 'exponential'（指数），默认'linear'

    返回:
        tuple: (time (ndarray), signal (ndarray)) - 时间轴和信号序列
    """
    if auto_period:
        base_freq = frequency if frequency > 0 else 1.0
        if wave_type in ['am', 'fm'] and carrier_freq is not None:
            base_freq = carrier_freq
        n_periods = max(1, round(base_freq * duration))
        n_samples = int(np.round(n_periods * sample_rate / base_freq))
        duration = n_samples / sample_rate
        dt = 1.0 / sample_rate
        time = np.arange(n_samples) * dt
        phase_rad = 2 * np.pi * n_periods * (np.arange(n_samples) + 0.5) / n_samples + phase
    else:
        n_samples = int(sample_rate * duration)
        duration = n_samples / sample_rate
        dt = 1.0 / sample_rate
        time = np.arange(n_samples) * dt
        phase_rad = 2 * np.pi * frequency * time + phase

    cycle_phase = (phase_rad / (2 * np.pi)) % 1.0

    if wave_type == 'sine':
        signal = amplitude * np.sin(phase_rad)
    elif wave_type == 'square':
        signal = amplitude * np.where(cycle_phase < 0.5, 1, -1)
    elif wave_type == 'triangle':
        signal = amplitude * (2 * np.abs(2 * (cycle_phase - np.floor(cycle_phase + 0.5))) - 1)
    elif wave_type == 'sawtooth':
        signal = amplitude * (2 * cycle_phase - 1)
    elif wave_type == 'am':
        if carrier_freq is None:
            carrier_freq = frequency * 10
        if mod_freq is None:
            mod_freq = frequency
        if mod_index < 0 or mod_index > 1:
            raise ValueError("AM调制度mod_index应在0~1之间")
        carrier = amplitude * np.sin(2 * np.pi * carrier_freq * time + phase)
        modulating = 1 + mod_index * np.sin(2 * np.pi * mod_freq * time)
        signal = carrier * modulating
    elif wave_type == 'fm':
        if carrier_freq is None:
            carrier_freq = frequency * 10
        if mod_freq is None:
            mod_freq = frequency
        freq_deviation = mod_index * mod_freq
        modulating_integral = np.cumsum(np.sin(2 * np.pi * mod_freq * time)) * dt
        signal = amplitude * np.sin(2 * np.pi * carrier_freq * time
                                    + 2 * np.pi * freq_deviation * modulating_integral
                                    + phase)
    elif wave_type == 'chirp':
        if freq_start is None:
            freq_start = frequency
        if freq_end is None:
            freq_end = frequency * 10
        if chirp_method == 'linear':
            phase_instant = 2 * np.pi * (freq_start * time
                                         + (freq_end - freq_start) * time**2 / (2 * duration))
        elif chirp_method == 'exponential':
            beta = np.log(freq_end / freq_start) / duration
            phase_instant = 2 * np.pi * freq_start * (np.exp(beta * time) - 1) / beta
        else:
            raise ValueError(f"未知扫频方式: {chirp_method}，可选值：'linear', 'exponential'")
        signal = amplitude * np.sin(phase_instant + phase)
    else:
        raise ValueError(f"未知波形类型: {wave_type}，可选值："
                         f"'sine', 'square', 'triangle', 'sawtooth', 'am', 'fm', 'chirp'")

    if window is not None:
        if window == 'hann':
            win = np.hanning(n_samples)
        elif window == 'flattop':
            win = _flattop_window(n_samples)
        else:
            raise ValueError(f"未知窗函数: {window}，可选值：'hann', 'flattop'")
        signal = signal * win

    if snr is not None:
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(snr / 10))
        noise = np.sqrt(noise_power) * np.random.randn(n_samples)
        signal = signal + noise

    return time, signal


def _flattop_window(n):
    """生成平顶窗，用于精确的幅度测量"""
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368

    k = np.arange(n)
    win = (a0 - a1 * np.cos(2 * np.pi * k / (n - 1))
               + a2 * np.cos(4 * np.pi * k / (n - 1))
               - a3 * np.cos(6 * np.pi * k / (n - 1))
               + a4 * np.cos(8 * np.pi * k / (n - 1)))
    return win


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample_rate = 1000
    amp = 1.0
    phase = 0.0
    duration = 2.0

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    t, s_am = generate_signal('am', frequency=5, sample_rate=sample_rate,
                              amplitude=amp, phase=phase, duration=duration,
                              carrier_freq=50, mod_freq=5, mod_index=0.8)
    axes[0, 0].plot(t, s_am)
    axes[0, 0].set_title('调幅波 (AM)')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('幅度')
    axes[0, 0].grid(True)
    axes[0, 0].set_xlim(0, 1)

    t, s_fm = generate_signal('fm', frequency=5, sample_rate=sample_rate,
                              amplitude=amp, phase=phase, duration=duration,
                              carrier_freq=50, mod_freq=5, mod_index=5)
    axes[0, 1].plot(t, s_fm)
    axes[0, 1].set_title('调频波 (FM)')
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('幅度')
    axes[0, 1].grid(True)
    axes[0, 1].set_xlim(0, 1)

    t, s_chirp_lin = generate_signal('chirp', frequency=1, sample_rate=sample_rate,
                                     amplitude=amp, phase=phase, duration=duration,
                                     freq_start=1, freq_end=20, chirp_method='linear')
    axes[1, 0].plot(t, s_chirp_lin)
    axes[1, 0].set_title('线性扫频信号 (Linear Chirp)')
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('幅度')
    axes[1, 0].grid(True)

    t, s_chirp_exp = generate_signal('chirp', frequency=1, sample_rate=sample_rate,
                                     amplitude=amp, phase=phase, duration=duration,
                                     freq_start=1, freq_end=100, chirp_method='exponential')
    axes[1, 1].plot(t, s_chirp_exp)
    axes[1, 1].set_title('指数扫频信号 (Exponential Chirp)')
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('幅度')
    axes[1, 1].grid(True)

    t, s_clean = generate_signal('sine', frequency=5, sample_rate=sample_rate,
                                 amplitude=amp, phase=phase, duration=1.0)
    t, s_noisy = generate_signal('sine', frequency=5, sample_rate=sample_rate,
                                 amplitude=amp, phase=phase, duration=1.0, snr=10)
    axes[2, 0].plot(t, s_clean, label='干净信号', alpha=0.7)
    axes[2, 0].plot(t, s_noisy, label='SNR=10dB', alpha=0.7)
    axes[2, 0].set_title('高斯白噪声示例')
    axes[2, 0].set_xlabel('时间 (s)')
    axes[2, 0].set_ylabel('幅度')
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    t, s_am_noisy = generate_signal('am', frequency=5, sample_rate=sample_rate,
                                    amplitude=amp, phase=phase, duration=1.0,
                                    carrier_freq=50, mod_freq=5, mod_index=0.8, snr=20)
    axes[2, 1].plot(t, s_am_noisy)
    axes[2, 1].set_title('带噪声的AM信号 (SNR=20dB)')
    axes[2, 1].set_xlabel('时间 (s)')
    axes[2, 1].set_ylabel('幅度')
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.show()
