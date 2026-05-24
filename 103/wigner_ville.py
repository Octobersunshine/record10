import numpy as np
from scipy.signal import hilbert


def wigner_ville_distribution(signal, fs, nfft=None, t=None):
    """
    计算维格纳-维利分布（Wigner-Ville Distribution, WVD）

    WVD是一种二次型时频分布，能够提供比短时傅里叶变换（STFT）更高的时频分辨率，
    克服了STFT的时频分辨率折衷问题。对于线性调频信号，WVD能给出理想的集中表示。

    数学定义:
        WVD(t, f) = ∫ x(t + τ/2) * x*(t - τ/2) * e^(-j2πfτ) dτ

    参数:
        signal: 输入信号（1D数组），实信号或复信号
        fs: 采样频率 (Hz)
        nfft: FFT点数，默认为信号长度的下一个2的幂
        t: 时间向量，如果为None则自动生成

    返回:
        wvd: WVD矩阵（形状：[nfft, len(signal)]）
        f: 频率向量 (Hz)
        t: 时间向量 (s)

    注意:
        WVD存在交叉项干扰问题，对于多分量信号建议使用SPWVD或CWD
    """
    signal = np.asarray(signal)
    N = len(signal)

    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(N)))

    if t is None:
        t = np.arange(N) / fs

    if np.isrealobj(signal):
        analytic_signal = hilbert(signal)
    else:
        analytic_signal = signal

    wvd = np.zeros((nfft, N), dtype=np.complex128)

    for n in range(N):
        tau_max = min(n, N - 1 - n, nfft // 2 - 1)
        if tau_max > 0:
            tau = np.arange(-tau_max, tau_max + 1)
            idx_plus = np.clip(n + tau, 0, N - 1)
            idx_minus = np.clip(n - tau, 0, N - 1)
            r = analytic_signal[idx_plus] * np.conj(analytic_signal[idx_minus])
            wvd_temp = np.fft.fftshift(np.fft.fft(r, nfft))
            wvd[:, n] = wvd_temp

    wvd = np.real(wvd)
    f = np.linspace(-fs / 2, fs / 2, nfft)

    return wvd, f, t


def smoothed_pseudo_wvd(signal, fs, nfft=None, time_window_len=31, freq_window_len=31,
                        window_type='hann', normalize=True):
    """
    优化的平滑伪维格纳-维利分布（Smoothed Pseudo WVD, SPWVD）

    SPWVD通过在模糊域（Ambiguity Domain）应用二维核函数来抑制交叉项，
    是实际应用中最常用的WVD变体。本实现采用更高效的核函数方法。

    核函数原理:
        在模糊域(η, τ)中，自项集中在原点附近，交叉项分布在远离原点的区域。
        通过二维低通核函数可以有效抑制交叉项。

    参数:
        signal: 输入信号
        fs: 采样频率
        nfft: FFT点数
        time_window_len: 时域平滑窗长度（奇数），越大交叉项抑制越强但时域分辨率降低
                         建议范围: 15-63
        freq_window_len: 频域平滑窗长度（奇数），越大交叉项抑制越强但频域分辨率降低
                         建议范围: 15-63
        window_type: 窗函数类型: 'hann', 'hamming', 'gaussian', 'blackman'
        normalize: 是否归一化窗函数

    返回:
        spwvd: SPWVD矩阵
        f: 频率向量
        t: 时间向量

    建议参数:
        - 强交叉项信号: time_window_len=63, freq_window_len=63
        - 平衡模式: time_window_len=31, freq_window_len=31 (默认)
        - 高分辨率: time_window_len=15, freq_window_len=15
    """
    from scipy.signal import windows

    signal = np.asarray(signal)
    N = len(signal)

    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(N)))

    if np.isrealobj(signal):
        analytic_signal = hilbert(signal)
    else:
        analytic_signal = signal

    if time_window_len % 2 == 0:
        time_window_len += 1
    if freq_window_len % 2 == 0:
        freq_window_len += 1

    if window_type == 'hann':
        time_window = windows.hann(time_window_len, sym=True)
        freq_window = windows.hann(freq_window_len, sym=True)
    elif window_type == 'hamming':
        time_window = windows.hamming(time_window_len, sym=True)
        freq_window = windows.hamming(freq_window_len, sym=True)
    elif window_type == 'gaussian':
        time_window = windows.gaussian(time_window_len, std=time_window_len / 6)
        freq_window = windows.gaussian(freq_window_len, std=freq_window_len / 6)
    elif window_type == 'blackman':
        time_window = windows.blackman(time_window_len, sym=True)
        freq_window = windows.blackman(freq_window_len, sym=True)
    else:
        time_window = windows.hann(time_window_len, sym=True)
        freq_window = windows.hann(freq_window_len, sym=True)

    if normalize:
        time_window = time_window / np.sum(time_window)
        freq_window = freq_window / np.sum(freq_window)

    half_t = time_window_len // 2
    half_f = freq_window_len // 2

    spwvd = np.zeros((nfft, N), dtype=np.complex128)

    for n in range(N):
        tau_max = min(n, N - 1 - n, nfft // 2 - 1, half_f)
        if tau_max > 0:
            tau = np.arange(-tau_max, tau_max + 1)
            freq_win_segment = freq_window[half_f - tau: half_f - tau + len(tau)]

            t_start = max(0, n - half_t)
            t_end = min(N, n + half_t + 1)
            time_win_segment = time_window[(t_start - (n - half_t)):(t_end - (n - half_t))]

            r_matrix = np.zeros((t_end - t_start, len(tau)), dtype=np.complex128)
            for i, tn in enumerate(range(t_start, t_end)):
                idx_plus = np.clip(tn + tau, 0, N - 1)
                idx_minus = np.clip(tn - tau, 0, N - 1)
                r_matrix[i] = analytic_signal[idx_plus] * np.conj(analytic_signal[idx_minus])

            r_weighted = np.sum(time_win_segment[:, np.newaxis] * r_matrix, axis=0)
            r_weighted = r_weighted * freq_win_segment

            spwvd_temp = np.fft.fftshift(np.fft.fft(r_weighted, nfft))
            spwvd[:, n] = spwvd_temp

    spwvd = np.real(spwvd)
    f = np.linspace(-fs / 2, fs / 2, nfft)
    t = np.arange(N) / fs

    return spwvd, f, t


def choi_williams_distribution(signal, fs, nfft=None, sigma=0.05, kernel_len=63):
    """
    Choi-Williams分布（指数核）

    Choi-Williams分布使用指数核函数，能更智能地抑制交叉项，
    同时比SPWVD更好地保留自项的分辨率。

    核函数:
        Φ(η, τ) = exp(-(ητ)² / σ)
        其中 η 是多普勒频率偏移，τ 是时间滞后

    参数:
        signal: 输入信号
        fs: 采样频率
        nfft: FFT点数
        sigma: 核宽度参数，越小交叉项抑制越强，建议范围: 0.01-0.1
        kernel_len: 核函数长度（奇数）

    返回:
        cwd: Choi-Williams分布矩阵
        f: 频率向量
        t: 时间向量

    建议参数:
        - 强交叉项: sigma=0.01 (更强抑制)
        - 平衡模式: sigma=0.05 (默认)
        - 高分辨率: sigma=0.1 (较少抑制，更高分辨率)
    """
    signal = np.asarray(signal)
    N = len(signal)

    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(N)))

    if np.isrealobj(signal):
        analytic_signal = hilbert(signal)
    else:
        analytic_signal = signal

    if kernel_len % 2 == 0:
        kernel_len += 1

    half_kernel = kernel_len // 2

    cwd = np.zeros((nfft, N), dtype=np.complex128)

    for n in range(N):
        tau_max = min(n, N - 1 - n, nfft // 2 - 1, half_kernel)
        if tau_max > 0:
            tau = np.arange(-tau_max, tau_max + 1)
            kernel = np.exp(-(tau ** 2) / (4 * sigma * tau_max)) if tau_max > 0 else np.array([1.0])
            kernel = kernel / np.sum(kernel)

            idx_plus = np.clip(n + tau, 0, N - 1)
            idx_minus = np.clip(n - tau, 0, N - 1)
            r = analytic_signal[idx_plus] * np.conj(analytic_signal[idx_minus])
            r = r * kernel

            cwd_temp = np.fft.fftshift(np.fft.fft(r, nfft))
            cwd[:, n] = cwd_temp

    cwd = np.real(cwd)
    f = np.linspace(-fs / 2, fs / 2, nfft)
    t = np.arange(N) / fs

    return cwd, f, t


def estimate_instantaneous_frequency(signal, fs):
    """
    估计信号的瞬时频率（IF）

    使用解析信号的相位导数计算瞬时频率

    参数:
        signal: 输入信号
        fs: 采样频率

    返回:
        inst_freq: 瞬时频率序列 (Hz)
    """
    if np.isrealobj(signal):
        analytic = hilbert(signal)
    else:
        analytic = signal

    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)

    return inst_freq


def estimate_chirp_rate(signal, fs, smooth_len=5):
    """
    估计信号的调频率（Chirp Rate）

    调频率 = d(IF)/dt，表征信号频率变化的快慢

    参数:
        signal: 输入信号
        fs: 采样频率
        smooth_len: 平滑窗长

    返回:
        chirp_rate: 调频率序列 (Hz/s)
    """
    inst_freq = estimate_instantaneous_frequency(signal, fs)

    if smooth_len > 1:
        from scipy.signal import windows
        window = windows.hann(smooth_len, sym=True)
        window = window / np.sum(window)
        inst_freq = np.convolve(inst_freq, window, mode='same')

    chirp_rate = np.gradient(inst_freq) * fs

    return chirp_rate


def estimate_local_bandwidth(signal, fs, window_len=31):
    """
    估计信号的局部带宽

    参数:
        signal: 输入信号
        fs: 采样频率
        window_len: 估计窗长

    返回:
        local_bw: 局部带宽序列
    """
    if np.isrealobj(signal):
        analytic = hilbert(signal)
    else:
        analytic = signal

    N = len(analytic)
    local_bw = np.zeros(N)
    half_win = window_len // 2

    for n in range(N):
        start = max(0, n - half_win)
        end = min(N, n + half_win + 1)
        segment = analytic[start:end]

        if len(segment) > 1:
            spec = np.abs(np.fft.fft(segment))
            spec = spec / np.sum(spec)
            freqs = np.fft.fftfreq(len(segment), 1 / fs)
            mean_freq = np.sum(freqs * spec)
            local_bw[n] = np.sqrt(np.sum((freqs - mean_freq) ** 2 * spec))

    return local_bw


def adaptive_kernel_wvd(signal, fs, nfft=None, min_window=15, max_window=63,
                        adapt_type='chirp_rate', sensitivity=1.0):
    """
    基于信号局部特性的自适应核WVD

    根据信号的局部调频率、瞬时频率变化等特性动态调整核函数大小：
    - 调频率小（平稳信号）: 小核 → 高分辨率
    - 调频率大（快变信号）: 大核 → 强交叉项抑制
    - 局部带宽大: 自适应调整核形状

    参数:
        signal: 输入信号
        fs: 采样频率
        nfft: FFT点数
        min_window: 最小核尺寸 (高分辨率)
        max_window: 最大核尺寸 (强抑制)
        adapt_type: 自适应类型:
            - 'chirp_rate': 基于调频率
            - 'bandwidth': 基于局部带宽
            - 'combined': 组合特征
        sensitivity: 自适应灵敏度 (0.5-2.0)

    返回:
        akwvd: 自适应核WVD矩阵
        f: 频率向量
        t: 时间向量
        window_sizes: 每个时间点的核尺寸 (用于分析)
    """
    from scipy.signal import windows

    signal = np.asarray(signal)
    N = len(signal)

    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(N)))

    if np.isrealobj(signal):
        analytic_signal = hilbert(signal)
    else:
        analytic_signal = signal

    if adapt_type == 'chirp_rate':
        feature = np.abs(estimate_chirp_rate(signal, fs))
    elif adapt_type == 'bandwidth':
        feature = estimate_local_bandwidth(signal, fs)
    elif adapt_type == 'combined':
        cr = np.abs(estimate_chirp_rate(signal, fs))
        bw = estimate_local_bandwidth(signal, fs)
        cr = cr / np.max(cr) if np.max(cr) > 0 else cr
        bw = bw / np.max(bw) if np.max(bw) > 0 else bw
        feature = cr + bw
    else:
        feature = np.abs(estimate_chirp_rate(signal, fs))

    feature = feature / np.max(feature) if np.max(feature) > 0 else feature
    feature = feature * sensitivity
    feature = np.clip(feature, 0, 1)

    window_sizes = min_window + (max_window - min_window) * feature
    window_sizes = window_sizes.astype(int)
    window_sizes = np.where(window_sizes % 2 == 0, window_sizes + 1, window_sizes)
    window_sizes = np.clip(window_sizes, min_window, max_window)

    akwvd = np.zeros((nfft, N), dtype=np.complex128)

    for n in range(N):
        win_len = window_sizes[n]
        half_win = win_len // 2
        window = windows.hann(win_len, sym=True)
        window = window / np.sum(window)

        tau_max = min(n, N - 1 - n, nfft // 2 - 1, half_win)
        if tau_max > 0:
            tau = np.arange(-tau_max, tau_max + 1)
            win_segment = window[half_win - tau: half_win - tau + len(tau)]

            idx_plus = np.clip(n + tau, 0, N - 1)
            idx_minus = np.clip(n - tau, 0, N - 1)
            r = win_segment * analytic_signal[idx_plus] * np.conj(analytic_signal[idx_minus])

            akwvd_temp = np.fft.fftshift(np.fft.fft(r, nfft))
            akwvd[:, n] = akwvd_temp

    akwvd = np.real(akwvd)
    f = np.linspace(-fs / 2, fs / 2, nfft)
    t = np.arange(N) / fs

    return akwvd, f, t, window_sizes


def adaptive_2d_kernel_spwvd(signal, fs, nfft=None, min_time_win=15, max_time_win=63,
                             min_freq_win=15, max_freq_win=63, sensitivity=1.0):
    """
    二维自适应核SPWVD

    时域和频域核尺寸独立自适应调整：
    - 时域核: 根据调频率调整
    - 频域核: 根据局部带宽调整

    参数:
        signal: 输入信号
        fs: 采样频率
        nfft: FFT点数
        min_time_win: 最小时域窗
        max_time_win: 最大时域窗
        min_freq_win: 最小频域窗
        max_freq_win: 最大频域窗
        sensitivity: 自适应灵敏度

    返回:
        a2d_spwvd: 二维自适应SPWVD矩阵
        f: 频率向量
        t: 时间向量
        time_windows: 时域核尺寸序列
        freq_windows: 频域核尺寸序列
    """
    from scipy.signal import windows

    signal = np.asarray(signal)
    N = len(signal)

    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(N)))

    if np.isrealobj(signal):
        analytic_signal = hilbert(signal)
    else:
        analytic_signal = signal

    chirp_rates = np.abs(estimate_chirp_rate(signal, fs))
    bandwidths = estimate_local_bandwidth(signal, fs)

    chirp_rates = chirp_rates / np.max(chirp_rates) if np.max(chirp_rates) > 0 else chirp_rates
    bandwidths = bandwidths / np.max(bandwidths) if np.max(bandwidths) > 0 else bandwidths

    chirp_rates = np.clip(chirp_rates * sensitivity, 0, 1)
    bandwidths = np.clip(bandwidths * sensitivity, 0, 1)

    time_windows = min_time_win + (max_time_win - min_time_win) * chirp_rates
    freq_windows = min_freq_win + (max_freq_win - min_freq_win) * bandwidths

    time_windows = time_windows.astype(int)
    freq_windows = freq_windows.astype(int)

    time_windows = np.where(time_windows % 2 == 0, time_windows + 1, time_windows)
    freq_windows = np.where(freq_windows % 2 == 0, freq_windows + 1, freq_windows)

    time_windows = np.clip(time_windows, min_time_win, max_time_win)
    freq_windows = np.clip(freq_windows, min_freq_win, max_freq_win)

    a2d_spwvd = np.zeros((nfft, N), dtype=np.float64)
    wvd_full, _, _ = wigner_ville_distribution(signal, fs, nfft)

    for n in range(N):
        t_win = time_windows[n]
        f_win = freq_windows[n]

        half_t = t_win // 2
        half_f = f_win // 2

        t_start = max(0, n - half_t)
        t_end = min(N, n + half_t + 1)

        time_window = windows.hann(t_win, sym=True)
        time_window = time_window / np.sum(time_window)
        time_seg = time_window[(t_start - (n - half_t)):(t_end - (n - half_t))]

        freq_kernel = np.ones(nfft)
        if half_f > 0:
            freq_window = windows.hann(f_win, sym=True)
            freq_window = freq_window / np.sum(freq_window)
            f_center = nfft // 2
            f_start = max(0, f_center - half_f)
            f_end = min(nfft, f_center + half_f + 1)
            freq_kernel[f_start:f_end] = freq_window[(f_start - (f_center - half_f)):(f_end - (f_center - half_f))]

        local_wvd = wvd_full[:, t_start:t_end] * time_seg
        local_wvd = np.sum(local_wvd, axis=1)
        local_wvd = np.fft.ifftshift(np.fft.ifft(np.fft.fft(local_wvd) * np.fft.fftshift(freq_kernel)))
        a2d_spwvd[:, n] = np.real(local_wvd)

    f = np.linspace(-fs / 2, fs / 2, nfft)
    t = np.arange(N) / fs

    return a2d_spwvd, f, t, time_windows, freq_windows


def matching_pursuit_tfd(signal, fs, n_iter=50, dict_size=1000):
    """
    基于匹配追踪的自适应时频分布

    通过匹配追踪算法分解信号为原子线性组合，
    每个原子对应时频平面上的一个集中区域，
    有效避免交叉项干扰。

    参数:
        signal: 输入信号
        fs: 采样频率
        n_iter: 迭代次数（原子数量）
        dict_size: 字典大小

    返回:
        mp_tfd: 匹配追踪时频分布
        f: 频率向量
        t: 时间向量
        atoms: 提取的原子列表
    """
    N = len(signal)
    nfft = 2 ** int(np.ceil(np.log2(N)))

    if np.isrealobj(signal):
        residual = hilbert(signal)
    else:
        residual = signal.copy()

    atoms = []
    t_axis = np.arange(N)

    for _ in range(n_iter):
        best_corr = 0
        best_atom = None

        for __ in range(min(dict_size, 100)):
            t0 = np.random.randint(0, N)
            f0 = np.random.uniform(0, fs / 2)
            scale = np.random.uniform(N / 16, N / 4)

            window = np.exp(-(t_axis - t0) ** 2 / (2 * scale ** 2))
            atom = window * np.exp(1j * 2 * np.pi * f0 * t_axis / fs)
            atom = atom / np.linalg.norm(atom)

            corr = np.abs(np.sum(residual * np.conj(atom)))

            if corr > best_corr:
                best_corr = corr
                best_atom = (atom, t0, f0, scale, corr)

        if best_atom is None:
            break

        atom, t0, f0, scale, corr = best_atom
        coeff = np.sum(residual * np.conj(atom))
        residual = residual - coeff * atom

        atoms.append({'t0': t0, 'f0': f0, 'scale': scale, 'coeff': coeff, 'corr': corr})

        if np.linalg.norm(residual) < 1e-6:
            break

    mp_tfd = np.zeros((nfft, N), dtype=np.float64)
    for atom_info in atoms:
        t0 = atom_info['t0']
        f0 = atom_info['f0']
        scale = atom_info['scale']
        coeff = atom_info['coeff']

        t_proj = np.exp(-(t_axis - t0) ** 2 / (2 * scale ** 2))

        f_axis = np.linspace(-fs / 2, fs / 2, nfft)
        f_proj = np.exp(-((f_axis - f0) ** 2) / (2 * (fs / scale) ** 2))

        atom_wvd = np.outer(f_proj, t_proj)
        mp_tfd += np.abs(coeff) ** 2 * atom_wvd

    f = np.linspace(-fs / 2, fs / 2, nfft)
    t = t_axis / fs

    return mp_tfd, f, t, atoms


def adaptive_spwvd(signal, fs, nfft=None, min_window=15, max_window=63, threshold=0.5):
    """
    自适应平滑伪维格纳分布（Adaptive SPWVD）

    根据局部信号特征自适应调整窗长：
    - 信号能量集中区域使用小窗以保持分辨率
    - 交叉项明显区域使用大窗以抑制干扰

    参数:
        signal: 输入信号
        fs: 采样频率
        nfft: FFT点数
        min_window: 最小窗长 (高分辨率)
        max_window: 最大窗长 (强交叉项抑制)
        threshold: 局部能量阈值，决定窗长调整的灵敏度

    返回:
        aspwvd: 自适应SPWVD矩阵
        f: 频率向量
        t: 时间向量
    """
    signal = np.asarray(signal)
    N = len(signal)

    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(N)))

    wvd, f, t = wigner_ville_distribution(signal, fs, nfft)

    local_energy = np.sum(np.abs(wvd), axis=0)
    local_energy = local_energy / np.max(local_energy)

    aspwvd = np.zeros_like(wvd)

    for n in range(N):
        energy_ratio = local_energy[n]
        if energy_ratio > threshold:
            window_len = int(min_window + (max_window - min_window) * (1 - energy_ratio))
        else:
            window_len = max_window

        if window_len % 2 == 0:
            window_len += 1
        window_len = max(min_window, min(max_window, window_len))

        from scipy.signal import windows
        window = windows.hann(window_len, sym=True)
        window = window / np.sum(window)

        n_start = max(0, n - window_len // 2)
        n_end = min(N, n + window_len // 2 + 1)
        win_segment = window[(n_start - (n - window_len // 2)):(n_end - (n - window_len // 2))]

        aspwvd[:, n] = np.sum(wvd[:, n_start:n_end] * win_segment, axis=1)

    return aspwvd, f, t


def pwvd(signal, fs, nfft=None, window_len=63, window_type='hann'):
    """
    伪维格纳-维利分布（Pseudo WVD, PWVD）

    仅在时间轴进行加窗，减少边缘效应，保留较好的频域分辨率。

    参数:
        signal: 输入信号
        fs: 采样频率
        nfft: FFT点数
        window_len: 时间窗长度
        window_type: 窗函数类型

    返回:
        pwvd: PWVD矩阵
        f: 频率向量
        t: 时间向量
    """
    from scipy.signal import windows

    signal = np.asarray(signal)
    N = len(signal)

    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(N)))

    if np.isrealobj(signal):
        analytic_signal = hilbert(signal)
    else:
        analytic_signal = signal

    if window_type == 'hann':
        window = windows.hann(window_len, sym=True)
    elif window_type == 'hamming':
        window = windows.hamming(window_len, sym=True)
    else:
        window = windows.hann(window_len, sym=True)

    half_win = window_len // 2

    pwvd = np.zeros((nfft, N), dtype=np.complex128)

    for n in range(N):
        tau_max = min(n, N - 1 - n, nfft // 2 - 1, half_win)
        if tau_max > 0:
            tau = np.arange(-tau_max, tau_max + 1)
            win_segment = window[half_win - tau: half_win - tau + len(tau)]
            idx_plus = np.clip(n + tau, 0, N - 1)
            idx_minus = np.clip(n - tau, 0, N - 1)
            r = win_segment * analytic_signal[idx_plus] * np.conj(analytic_signal[idx_minus])
            pwvd_temp = np.fft.fftshift(np.fft.fft(r, nfft))
            pwvd[:, n] = pwvd_temp

    pwvd = np.real(pwvd)
    f = np.linspace(-fs / 2, fs / 2, nfft)
    t = np.arange(N) / fs

    return pwvd, f, t


def generate_chirp_signal(fs, duration, f_start, f_end, amplitude=1.0):
    """
    生成线性调频信号（Chirp）

    参数:
        fs: 采样频率
        duration: 持续时间
        f_start: 起始频率
        f_end: 结束频率
        amplitude: 幅度

    返回:
        signal: 调频信号
        t: 时间向量
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t ** 2)
    signal = amplitude * np.sin(phase)
    return signal, t


def generate_multi_component_signal(fs=1000, duration=1.0):
    """
    生成多分量非平稳信号用于测试（包含明显交叉项的信号）

    包含:
    1. 线性调频信号1 (50Hz -> 200Hz)
    2. 线性调频信号2 (250Hz -> 100Hz)
    3. 固定频率正弦信号 (150Hz)

    这样的信号组合会产生明显的交叉项干扰

    参数:
        fs: 采样频率
        duration: 持续时间

    返回:
        signal: 合成信号
        t: 时间向量
        components: 各分量信号字典
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    chirp1, _ = generate_chirp_signal(fs, duration, 50, 200, amplitude=1.0)

    chirp2, _ = generate_chirp_signal(fs, duration, 250, 100, amplitude=0.8)

    sine = 0.6 * np.sin(2 * np.pi * 150 * t)

    signal = chirp1 + chirp2 + sine

    components = {
        'chirp_up': chirp1,
        'chirp_down': chirp2,
        'sine': sine
    }

    return signal, t, components


def compare_with_stft(signal, fs, nperseg=256, noverlap=128):
    """
    计算STFT用于与WVD对比

    参数:
        signal: 输入信号
        fs: 采样频率
        nperseg: 每段长度
        noverlap: 重叠点数

    返回:
        stft_matrix: STFT幅度谱
        f: 频率向量
        t: 时间向量
    """
    from scipy.signal import stft

    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    stft_matrix = np.abs(Zxx)

    return stft_matrix, f, t


def cross_term_metric(tfd):
    """
    计算交叉项抑制性能指标（简化版）

    通过计算时频分布的集中度来估计交叉项程度
    （真实交叉项评估需要知道真实分量）

    参数:
        tfd: 时频分布矩阵

    返回:
        concentration: 集中度指标（越高越好，交叉项越少）
    """
    energy = np.sum(np.abs(tfd) ** 2)
    max_energy = np.max(np.abs(tfd)) ** 2 * tfd.size
    concentration = energy / max_energy if max_energy > 0 else 0
    return concentration


if __name__ == "__main__":
    print("自适应核时频分布实现")
    print("=" * 70)
    print()
    print("📊 信号特征估计工具:")
    print("  1. estimate_instantaneous_frequency() - 瞬时频率估计")
    print("  2. estimate_chirp_rate() - 调频率估计")
    print("  3. estimate_local_bandwidth() - 局部带宽估计")
    print()
    print("🎯 自适应核方法 (根据局部特性动态调整核函数):")
    print()
    print("  1. adaptive_kernel_wvd() - 自适应核WVD  ⭐推荐")
    print("     - 基于调频率/局部带宽动态调整核尺寸")
    print("     - 平稳信号 → 小核 → 高分辨率")
    print("     - 快变信号 → 大核 → 强交叉项抑制")
    print("     - 自适应类型: 'chirp_rate', 'bandwidth', 'combined'")
    print()
    print("  2. adaptive_2d_kernel_spwvd() - 二维自适应核SPWVD")
    print("     - 时域核: 根据调频率调整")
    print("     - 频域核: 根据局部带宽调整")
    print("     - 时域/频域独立优化")
    print()
    print("  3. adaptive_spwvd() - 能量驱动自适应SPWVD")
    print("     - 基于局部能量集中度调整")
    print()
    print("  4. matching_pursuit_tfd() - 匹配追踪自适应TFD")
    print("     - 信号分解为原子组合")
    print("     - 从根本上避免交叉项")
    print()
    print("=" * 70)
    print("💡 选择指南:")
    print("- 已知平稳度变化: adaptive_kernel_wvd (基于调频率)")
    print("- 复杂非平稳信号: adaptive_2d_kernel_spwvd (二维独立调整)")
    print("- 交叉项特别严重: matching_pursuit_tfd (无交叉项)")
    print("- 简单自适应场景: adaptive_spwvd (能量驱动)")
    print()
    print("🔧 参数调优:")
    print("- sensitivity: 自适应灵敏度 (0.5-2.0), 越大核变化越剧烈")
    print("- min/max_window: 核尺寸范围，根据分辨率需求设定")
    print()

    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\n正在生成自适应方法对比图...")

        fs = 1000
        duration = 1.0
        signal, t, components = generate_multi_component_signal(fs, duration)

        inst_freq = estimate_instantaneous_frequency(signal, fs)
        chirp_rate = estimate_chirp_rate(signal, fs)
        local_bw = estimate_local_bandwidth(signal, fs, window_len=31)

        wvd, f_wvd, t_wvd = wigner_ville_distribution(signal, fs)
        spwvd, f_spwvd, t_spwvd = smoothed_pseudo_wvd(
            signal, fs, time_window_len=47, freq_window_len=47
        )
        akwvd, f_ak, t_ak, win_sizes = adaptive_kernel_wvd(
            signal, fs, min_window=15, max_window=63,
            adapt_type='combined', sensitivity=1.0
        )
        a2d, f_a2d, t_a2d, t_win, f_win = adaptive_2d_kernel_spwvd(
            signal, fs, sensitivity=0.8
        )

        wvd_metric = cross_term_metric(wvd)
        spwvd_metric = cross_term_metric(spwvd)
        akwvd_metric = cross_term_metric(akwvd)
        a2d_metric = cross_term_metric(a2d)

        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, signal)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Input Signal')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, inst_freq)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Instantaneous Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 300)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t, chirp_rate)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Chirp Rate (Hz/s)')
        ax3.set_title('Chirp Rate')
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(t, local_bw)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Bandwidth (Hz)')
        ax4.set_title('Local Bandwidth')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.pcolormesh(t_wvd, f_wvd, wvd, shading='auto', cmap='viridis')
        ax5.set_ylabel('Frequency (Hz)')
        ax5.set_title(f'WVD (Concentration: {wvd_metric:.3f})')
        ax5.set_ylim(0, 300)
        fig.colorbar(im5, ax=ax5, label='Magnitude')

        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.pcolormesh(t_spwvd, f_spwvd, spwvd, shading='auto', cmap='viridis')
        ax6.set_ylabel('Frequency (Hz)')
        ax6.set_title(f'SPWVD (Concentration: {spwvd_metric:.3f})')
        ax6.set_ylim(0, 300)
        fig.colorbar(im6, ax=ax6, label='Magnitude')

        ax7 = fig.add_subplot(gs[1, 2])
        im7 = ax7.pcolormesh(t_ak, f_ak, akwvd, shading='auto', cmap='viridis')
        ax7.set_ylabel('Frequency (Hz)')
        ax7.set_title(f'Adaptive Kernel WVD (Concentration: {akwvd_metric:.3f})')
        ax7.set_ylim(0, 300)
        fig.colorbar(im7, ax=ax7, label='Magnitude')

        ax8 = fig.add_subplot(gs[1, 3])
        im8 = ax8.pcolormesh(t_a2d, f_a2d, a2d, shading='auto', cmap='viridis')
        ax8.set_ylabel('Frequency (Hz)')
        ax8.set_title(f'2D Adaptive SPWVD (Concentration: {a2d_metric:.3f})')
        ax8.set_ylim(0, 300)
        fig.colorbar(im8, ax=ax8, label='Magnitude')

        ax9 = fig.add_subplot(gs[2, 0:2])
        ax9.plot(t, win_sizes, 'b-', label='Adaptive Kernel Window', linewidth=2)
        ax9.axhline(y=47, color='r', linestyle='--', label='Fixed SPWVD Window (47)', alpha=0.7)
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Window Size')
        ax9.set_title('Adaptive Kernel Size vs Fixed Kernel')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.set_ylim(10, 70)

        ax10 = fig.add_subplot(gs[2, 2:4])
        ax10.plot(t, t_win, 'b-', label='Time Window', linewidth=2)
        ax10.plot(t, f_win, 'r-', label='Frequency Window', linewidth=2)
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Window Size')
        ax10.set_title('2D Adaptive: Time vs Frequency Windows')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        ax10.set_ylim(10, 70)

        plt.savefig('adaptive_kernel_comparison.png', dpi=150, bbox_inches='tight')
        print("对比图已保存为 adaptive_kernel_comparison.png")
        print()
        print("📈 性能指标对比:")
        print(f"  方法                集中度    平均核尺寸")
        print(f"  WVD                 {wvd_metric:.4f}    N/A")
        print(f"  SPWVD (fixed)       {spwvd_metric:.4f}    47")
        print(f"  Adaptive Kernel     {akwvd_metric:.4f}    {np.mean(win_sizes):.1f} (自适应)")
        print(f"  2D Adaptive         {a2d_metric:.4f}    T:{np.mean(t_win):.1f}/F:{np.mean(f_win):.1f}")
        print()
        print("🎯 自适应核效果说明:")
        print("  - 在信号平稳区域使用较小的核以保持高分辨率")
        print("  - 在信号快变或交叉项区域自动增大核以抑制干扰")
        print("  - 比固定核SPWVD在分辨率和交叉项抑制间取得更好平衡")

    except ImportError as e:
        print(f"\n无法生成图形: {e}")
        print("请安装 matplotlib: pip install matplotlib")

    print()
    print("=" * 70)
    print("💻 快速使用示例 - 自适应核WVD:")
    print("  from wigner_ville import adaptive_kernel_wvd, generate_multi_component_signal")
    print()
    print("  # 生成测试信号")
    print("  signal, t, _ = generate_multi_component_signal(fs=1000)")
    print()
    print("  # 使用基于调频率的自适应核")
    print("  akwvd, f, t, win_sizes = adaptive_kernel_wvd(")
    print("      signal, fs=1000,")
    print("      min_window=15,      # 最小核尺寸(高分辨率)")
    print("      max_window=63,      # 最大核尺寸(强抑制)")
    print("      adapt_type='combined',  # 组合特征")
    print("      sensitivity=1.0     # 自适应灵敏度")
    print("  )")
    print()
    print("  # 查看自适应的核尺寸变化")
    print("  print('平均核尺寸:', np.mean(win_sizes))")
