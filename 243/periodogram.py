import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

WINDOW_FUNCTIONS = {
    'rect': np.ones,
    'hann': np.hanning,
    'hamming': np.hamming,
    'blackman': np.blackman,
}


def _compute_psd(signal_windowed, n, fs, win):
    """内部：计算加窗信号的PSD并归一化"""
    win_energy = np.sum(win ** 2)

    fft_result = fft.fft(signal_windowed)
    psd = np.abs(fft_result) ** 2 / (fs * win_energy)

    freq = fft.fftfreq(n, d=1.0 / fs)

    psd = psd[:n // 2 + 1]
    freq = freq[:n // 2 + 1]

    psd[1:-1] *= 2

    return freq, psd


def periodogram(signal, fs, window='hann'):
    """
    周期图法功率谱估计（支持窗函数）

    参数:
        signal: 输入信号序列 (numpy数组)
        fs: 采样率 (Hz)
        window: 窗函数类型，可选 'rect'(矩形窗), 'hann'(汉宁窗),
                'hamming'(海明窗), 'blackman'(布莱克曼窗)，默认 'hann'

    返回:
        freq: 频率轴 (Hz)
        psd: 功率谱密度 (V^2/Hz)
    """
    n = len(signal)

    if window not in WINDOW_FUNCTIONS:
        raise ValueError(
            f"不支持的窗函数: '{window}'，可选: {list(WINDOW_FUNCTIONS.keys())}"
        )

    win = WINDOW_FUNCTIONS[window](n)
    signal_windowed = signal * win

    return _compute_psd(signal_windowed, n, fs, win)


def _dpss(n, k, nw=4):
    """生成前k个离散椭球序列（DPSS）窗函数"""
    W = nw / n

    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    diff = i - j

    sigma = np.where(
        diff == 0,
        2 * W,
        np.sin(2 * np.pi * W * diff) / (np.pi * diff + 1e-15)
    )

    eigvals, eigvecs = np.linalg.eigh(sigma)
    idx = np.argsort(eigvals)[::-1][:k]
    dpss_tapers = eigvecs[:, idx]

    signs = np.sign(dpss_tapers[0, :])
    dpss_tapers *= signs[None, :]

    return dpss_tapers, eigvals[idx]


def welch_psd(signal, fs, nperseg=None, noverlap=None, window='hann'):
    """
    Welch方法功率谱估计（分段重叠平均，降低估计方差）

    参数:
        signal: 输入信号序列 (numpy数组)
        fs: 采样率 (Hz)
        nperseg: 每段长度，默认取 min(256, len(signal))
        noverlap: 重叠样本数，默认取 nperseg // 2（50%重叠）
        window: 窗函数类型，默认 'hann'

    返回:
        freq: 频率轴 (Hz)
        psd: 功率谱密度 (V^2/Hz)
    """
    n = len(signal)

    if nperseg is None:
        nperseg = min(256, n)

    if noverlap is None:
        noverlap = nperseg // 2

    if nperseg > n:
        raise ValueError(f"每段长度 nperseg={nperseg} 不能大于信号长度 n={n}")

    if noverlap >= nperseg:
        raise ValueError(f"重叠样本数 noverlap={noverlap} 必须小于每段长度 nperseg={nperseg}")

    if window not in WINDOW_FUNCTIONS:
        raise ValueError(
            f"不支持的窗函数: '{window}'，可选: {list(WINDOW_FUNCTIONS.keys())}"
        )

    step = nperseg - noverlap
    n_segments = (n - noverlap) // step

    win = WINDOW_FUNCTIONS[window](nperseg)

    psd_total = np.zeros(nperseg // 2 + 1)

    for i in range(n_segments):
        start = i * step
        end = start + nperseg
        segment = signal[start:end]
        segment_windowed = segment * win

        _, psd_seg = _compute_psd(segment_windowed, nperseg, fs, win)
        psd_total += psd_seg

    psd_avg = psd_total / n_segments

    freq = fft.fftfreq(nperseg, d=1.0 / fs)[:nperseg // 2 + 1]

    return freq, psd_avg


def thomson_psd(signal, fs, nw=4, k=None, adaptive=True):
    """
    Thomson多窗谱估计（使用DPSS正交窗函数，提高分辨率同时降低方差）

    参数:
        signal: 输入信号序列 (numpy数组)
        fs: 采样率 (Hz)
        nw: 时间-带宽积，默认4（控制主瓣宽度与窗数量）
        k: 使用的窗函数数量，默认取 2*nw - 1
        adaptive: 是否使用自适应加权，默认True；False表示等权平均

    返回:
        freq: 频率轴 (Hz)
        psd: 功率谱密度 (V^2/Hz)
    """
    n = len(signal)

    if k is None:
        k = max(1, int(2 * nw) - 1)

    tapers, eigvals = _dpss(n, k, nw)

    eigvals = np.clip(eigvals, 1e-10, 1.0 - 1e-10)

    psd_tapers = np.zeros((k, n // 2 + 1))

    freq = None
    for i in range(k):
        taper = tapers[:, i]
        signal_tapered = signal * taper
        freq, psd_taper = _compute_psd(signal_tapered, n, fs, taper)
        psd_tapers[i, :] = psd_taper

    if not adaptive:
        weights = eigvals / np.sum(eigvals)
        psd = np.sum(weights[:, None] * psd_tapers, axis=0)
    else:
        psd = np.mean(psd_tapers, axis=0)

        for _ in range(5):
            psd_mat = psd[None, :]
            mean_taper = np.mean(psd_tapers, axis=0)[None, :]
            weights = (eigvals[:, None] * psd_mat) / \
                      (eigvals[:, None] * psd_mat + (1 - eigvals[:, None]) * mean_taper + 1e-10)
            weights_sq = weights ** 2
            numerator = np.sum(weights_sq * eigvals[:, None] * psd_tapers, axis=0)
            denominator = np.sum(weights_sq * eigvals[:, None], axis=0)
            psd = numerator / (denominator + 1e-10)

    return freq, psd


if __name__ == "__main__":
    np.random.seed(42)
    fs = 500
    t = np.arange(0, 1, 1 / fs)

    signal = 2.0 * np.sin(2 * np.pi * 50.0 * t) + \
             1.0 * np.sin(2 * np.pi * 55.0 * t) + \
             0.5 * np.sin(2 * np.pi * 119.7 * t)
    signal += 1.0 * np.random.randn(len(t))

    methods = [
        ('Periodogram (rect)', lambda s, f: periodogram(s, f, window='rect')),
        ('Periodogram (hann)', lambda s, f: periodogram(s, f, window='hann')),
        ('Welch (50% overlap)', lambda s, f: welch_psd(s, f, nperseg=256, noverlap=128, window='hann')),
        ('Thomson (nw=3, k=5)', lambda s, f: thomson_psd(s, f, nw=3, k=5, adaptive=True)),
    ]

    print("=" * 70)
    print("功率谱估计方法对比")
    print("=" * 70)
    print(f"信号: 50Hz + 55Hz + 119.7Hz, 采样率 {fs}Hz, 长度 {len(signal)} 点")
    print("-" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, func) in zip(axes, methods):
        freq, psd = func(signal, fs)
        ax.plot(freq, 10 * np.log10(psd), linewidth=0.8)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (dB/Hz)')
        ax.set_title(name)
        ax.set_xlim(0, 150)
        ax.grid(True, alpha=0.3)
        ax.axvline(50, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axvline(55, color='r', linestyle='--', alpha=0.5, linewidth=0.8)

        peak_idx = np.argsort(psd)[-4:][::-1]
        print(f"\n【{name}】")
        for idx in peak_idx:
            print(f"  {freq[idx]:.1f} Hz  ->  {10 * np.log10(psd[idx]):.2f} dB/Hz")

    plt.suptitle('PSD Estimation Comparison (50 Hz + 55 Hz + 119.7 Hz + noise)')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("方法特性对比:")
    print("=" * 70)
    print("1. Periodogram (rect): 分辨率最高，但方差大，泄漏严重")
    print("2. Periodogram (hann): 泄漏抑制好，分辨率略有下降")
    print("3. Welch: 分段平均，方差显著降低，分辨率等于分段长度")
    print("4. Thomson: 多窗正交平均，兼顾分辨率与方差，适合密频信号")
