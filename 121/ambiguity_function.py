import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def taylor_window(N, sll=-40, nbar=5):
    """
    生成Taylor窗函数
    
    参数:
        N: 窗长度
        sll: 最大旁瓣电平 (dB)，负值
        nbar: 近旁瓣数量
    
    返回:
        w: Taylor窗函数
    """
    sll_abs = abs(sll)
    
    if sll_abs < 13.3:
        return np.hanning(N)
    
    A = np.arccosh(10**(sll_abs / 20)) / np.pi
    
    sigma = nbar / np.sqrt(A**2 + (nbar - 0.5)**2)
    
    m = np.arange(1, nbar)
    Fm = np.zeros(nbar - 1)
    
    for i, mi in enumerate(m):
        numerator = 1.0
        for j in range(1, nbar):
            if j != mi:
                numerator *= (1 - mi**2 / (sigma**2 * (A**2 + (j - 0.5)**2)))
        
        denominator = 1.0
        for j in range(1, nbar):
            if j != mi:
                denominator *= (1 - mi**2 / j**2)
        
        Fm[i] = (-1)**(mi + 1) * numerator / denominator
    
    n = np.arange(N)
    w = np.ones(N)
    for mi, fm in zip(m, Fm):
        w += 2 * fm * np.cos(2 * np.pi * mi * (n - (N - 1) / 2) / N)
    
    return w / np.max(w)


def generate_lfm_waveform(bandwidth, pulse_width, fs, oversample=4):
    """
    生成线性调频（LFM）波形（支持过采样）
    
    参数:
        bandwidth: 调频带宽 (Hz)
        pulse_width: 脉冲宽度 (s)
        fs: 采样率 (Hz)
        oversample: 过采样倍数（用于抑制栅瓣）
    
    返回:
        t: 时间向量
        s: 复基带信号
    """
    num_samples = int(fs * oversample * pulse_width)
    t = np.linspace(-pulse_width/2, pulse_width/2, num_samples)
    k = bandwidth / pulse_width
    s = np.exp(1j * np.pi * k * t**2)
    return t, s


def generate_cw_waveform(pulse_width, fs, oversample=4):
    """
    生成单频连续波（CW）波形（支持过采样）
    
    参数:
        pulse_width: 脉冲宽度 (s)
        fs: 采样率 (Hz)
        oversample: 过采样倍数
    
    返回:
        t: 时间向量
        s: 复基带信号
    """
    num_samples = int(fs * oversample * pulse_width)
    t = np.linspace(-pulse_width/2, pulse_width/2, num_samples)
    s = np.ones_like(t, dtype=np.complex128)
    return t, s


def generate_p4_code(num_chips, chip_width, fs, oversample=4):
    """
    生成P4码（多相编码）波形
    
    参数:
        num_chips: 码片数量
        chip_width: 码片宽度 (s)
        fs: 采样率 (Hz)
        oversample: 过采样倍数
    
    返回:
        t: 时间向量
        s: 复基带信号
        code: P4码相位序列
    """
    samples_per_chip = int(fs * oversample * chip_width)
    total_samples = num_chips * samples_per_chip
    
    n = np.arange(num_chips)
    p4_phase = np.pi * n**2 / num_chips - np.pi * n
    
    t = np.linspace(0, num_chips * chip_width, total_samples)
    s = np.zeros(total_samples, dtype=np.complex128)
    
    for i in range(num_chips):
        start = i * samples_per_chip
        end = (i + 1) * samples_per_chip
        s[start:end] = np.exp(1j * p4_phase[i])
    
    return t, s, p4_phase


def is_prime(n):
    """检查是否为素数"""
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def find_primitive_root(p):
    """寻找素数p的本原根"""
    if p == 2:
        return 1
    
    phi = p - 1
    factors = set()
    n = phi
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors.add(i)
            n //= i
        i += 1
    if n > 1:
        factors.add(n)
    
    for g in range(2, p):
        is_primitive = True
        for factor in factors:
            if pow(g, phi // factor, p) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    return None


def generate_costas_sequence(p, chip_width, fs, oversample=4, primitive_root=None):
    """
    生成Costas序列波形（基于Welch构造法）
    
    参数:
        p: 素数（Costas序列阶数）
        chip_width: 码片宽度 (s)
        fs: 采样率 (Hz)
        oversample: 过采样倍数
        primitive_root: 本原根（可选，自动寻找）
    
    返回:
        t: 时间向量
        s: 复基带信号
        freq_seq: 频率序列
    """
    if not is_prime(p):
        raise ValueError("Costas序列阶数p必须是素数")
    
    if primitive_root is None:
        primitive_root = find_primitive_root(p)
    
    samples_per_chip = int(fs * oversample * chip_width)
    total_samples = p * samples_per_chip
    
    freq_seq = np.zeros(p, dtype=int)
    for k in range(p):
        freq_seq[k] = pow(primitive_root, k, p) - 1
    
    t = np.linspace(0, p * chip_width, total_samples)
    s = np.zeros(total_samples, dtype=np.complex128)
    
    max_freq = (p - 1) / (2 * chip_width)
    
    for i in range(p):
        start = i * samples_per_chip
        end = (i + 1) * samples_per_chip
        t_chip = t[start:end] - i * chip_width
        freq = freq_seq[i] / (p * chip_width)
        s[start:end] = np.exp(1j * 2 * np.pi * freq * t_chip)
    
    return t, s, freq_seq


def apply_window(s, window_type='taylor', sll=-40):
    """
    对信号施加窗函数
    
    参数:
        s: 输入信号
        window_type: 窗类型 ('taylor', 'hamming', 'hann', 'none')
        sll: Taylor窗旁瓣电平 (dB)
    
    返回:
        s_windowed: 加窗后的信号
        window: 窗函数
    """
    N = len(s)
    
    if window_type == 'taylor':
        window = taylor_window(N, sll=sll)
    elif window_type == 'hamming':
        window = np.hamming(N)
    elif window_type == 'hann':
        window = np.hanning(N)
    else:
        window = np.ones(N)
    
    return s * window, window


def ambiguity_function_fft(s, fs, num_delay=None, num_doppler=None, oversample=2):
    """
    使用FFT快速计算雷达信号的模糊函数（优化版）
    
    参数:
        s: 输入信号（复基带）
        fs: 采样率 (Hz)
        num_delay: 延迟采样点数
        num_doppler: 多普勒采样点数
        oversample: 频谱过采样倍数（抑制栅瓣）
    
    返回:
        delay: 延迟向量 (s)
        doppler: 多普勒向量 (Hz)
        af: 模糊函数矩阵
    """
    N = len(s)
    
    if num_delay is None:
        num_delay = 2 * N
    if num_doppler is None:
        num_doppler = int(2 * N * oversample)
    
    s_pad = np.zeros(num_delay, dtype=np.complex128)
    s_pad[:N] = s
    
    af = np.zeros((num_doppler, num_delay), dtype=np.complex128)
    
    for tau_idx in range(num_delay):
        shift = tau_idx - num_delay // 2
        s_shifted = np.roll(s_pad, shift)[:N]
        prod = s * np.conj(s_shifted)
        af[:, tau_idx] = np.fft.fftshift(np.fft.fft(prod, n=num_doppler)) / N
    
    delay = (np.arange(num_delay) - num_delay // 2) / fs
    doppler = (np.arange(num_doppler) - num_doppler // 2) * fs / num_doppler
    
    return delay, doppler, af


def calculate_sll(response, axis_range=None):
    """
    计算旁瓣电平 (SLL)
    
    参数:
        response: 响应序列（归一化幅度）
        axis_range: 主瓣范围（索引），用于排除主瓣
    
    返回:
        sll_db: 最大旁瓣电平 (dB)
    """
    response_db = 20 * np.log10(np.abs(response) + 1e-10)
    peak_idx = np.argmax(response_db)
    
    if axis_range is None:
        main_lobe_width = len(response) // 20
    else:
        main_lobe_width = int(abs(axis_range[1] - axis_range[0]) / 2)
    
    mask = np.ones_like(response_db, dtype=bool)
    start = max(0, peak_idx - main_lobe_width)
    end = min(len(response_db), peak_idx + main_lobe_width + 1)
    mask[start:end] = False
    
    if np.any(mask):
        sll_db = np.max(response_db[mask])
    else:
        sll_db = -np.inf
    
    return sll_db


def calculate_main_lobe_width(x, response, db_level=-3):
    """
    计算主瓣宽度（指定dB电平处的宽度）
    
    参数:
        x: x轴坐标向量
        response: 响应序列（幅度，非dB）
        db_level: 电平阈值 (dB)，默认-3dB
    
    返回:
        width: 主瓣宽度
        left_idx: 左交点索引
        right_idx: 右交点索引
    """
    response_norm = response / np.max(response)
    response_db = 20 * np.log10(response_norm + 1e-10)
    peak_idx = np.argmax(response_db)
    
    threshold = db_level
    
    left_idx = peak_idx
    while left_idx > 0 and response_db[left_idx] > threshold:
        left_idx -= 1
    
    right_idx = peak_idx
    while right_idx < len(response_db) - 1 and response_db[right_idx] > threshold:
        right_idx += 1
    
    if left_idx > 0 and right_idx < len(response_db) - 1:
        left_interp = x[left_idx] + (x[left_idx + 1] - x[left_idx]) * (threshold - response_db[left_idx]) / (response_db[left_idx + 1] - response_db[left_idx])
        right_interp = x[right_idx - 1] + (x[right_idx] - x[right_idx - 1]) * (threshold - response_db[right_idx - 1]) / (response_db[right_idx] - response_db[right_idx - 1])
        width = right_interp - left_interp
    else:
        width = x[right_idx] - x[left_idx]
        left_interp = x[left_idx]
        right_interp = x[right_idx]
    
    return width, left_interp, right_interp


def calculate_pslr(response, axis_range=None):
    """
    计算峰值旁瓣比 (Peak Sidelobe Ratio, PSLR)
    
    参数:
        response: 响应序列（幅度）
        axis_range: 主瓣范围
    
    返回:
        pslr_db: 峰值旁瓣比 (dB)
    """
    response_norm = response / np.max(response)
    peak_val = np.max(response_norm)
    
    response_db = 20 * np.log10(response_norm + 1e-10)
    peak_idx = np.argmax(response_db)
    
    if axis_range is None:
        main_lobe_width = len(response) // 20
    else:
        main_lobe_width = int(abs(axis_range[1] - axis_range[0]) / 2)
    
    mask = np.ones_like(response_db, dtype=bool)
    start = max(0, peak_idx - main_lobe_width)
    end = min(len(response_db), peak_idx + main_lobe_width + 1)
    mask[start:end] = False
    
    if np.any(mask):
        max_sidelobe = np.max(response_norm[mask])
        pslr_db = 20 * np.log10(max_sidelobe / peak_val + 1e-10)
    else:
        pslr_db = -np.inf
    
    return pslr_db


def calculate_islr(response, axis_range=None):
    """
    计算积分旁瓣比 (Integrated Sidelobe Ratio, ISLR)
    
    参数:
        response: 响应序列（幅度）
        axis_range: 主瓣范围
    
    返回:
        islr_db: 积分旁瓣比 (dB)
    """
    response_power = np.abs(response)**2
    peak_idx = np.argmax(response_power)
    
    if axis_range is None:
        main_lobe_width = len(response) // 20
    else:
        main_lobe_width = int(abs(axis_range[1] - axis_range[0]) / 2)
    
    mask_sidelobe = np.ones_like(response_power, dtype=bool)
    start = max(0, peak_idx - main_lobe_width)
    end = min(len(response_power), peak_idx + main_lobe_width + 1)
    mask_sidelobe[start:end] = False
    
    main_lobe_power = np.sum(response_power[~mask_sidelobe])
    sidelobe_power = np.sum(response_power[mask_sidelobe])
    
    if main_lobe_power > 0:
        islr_db = 10 * np.log10(sidelobe_power / main_lobe_power + 1e-10)
    else:
        islr_db = np.inf
    
    return islr_db


def evaluate_resolution_performance(x, response, x_name='delay'):
    """
    评估分辨性能
    
    参数:
        x: x轴坐标
        response: 响应序列（幅度）
        x_name: 轴名称 ('delay' 或 'doppler')
    
    返回:
        metrics: 性能指标字典
    """
    width_3dB, left_3dB, right_3dB = calculate_main_lobe_width(x, response, db_level=-3)
    width_6dB, _, _ = calculate_main_lobe_width(x, response, db_level=-6)
    
    pslr = calculate_pslr(response)
    islr = calculate_islr(response)
    sll = calculate_sll(response)
    
    metrics = {
        'width_3dB': width_3dB,
        'width_6dB': width_6dB,
        'pslr_db': pslr,
        'islr_db': islr,
        'sll_db': sll,
        'left_3dB': left_3dB,
        'right_3dB': right_3dB
    }
    
    return metrics


def evaluate_waveform_performance(delay, doppler, af, waveform_name=''):
    """
    评估波形的多目标分辨性能
    
    参数:
        delay: 延迟向量
        doppler: 多普勒向量
        af: 模糊函数矩阵
        waveform_name: 波形名称
    
    返回:
        all_metrics: 完整性能指标
    """
    af_mag = np.abs(af)
    af_mag = af_mag / np.max(af_mag)
    
    zero_doppler_idx = np.argmin(np.abs(doppler))
    zero_delay_idx = np.argmin(np.abs(delay))
    
    range_cut = af_mag[zero_doppler_idx, :]
    doppler_cut = af_mag[:, zero_delay_idx]
    
    range_metrics = evaluate_resolution_performance(delay, range_cut, 'delay')
    doppler_metrics = evaluate_resolution_performance(doppler, doppler_cut, 'doppler')
    
    all_metrics = {
        'waveform_name': waveform_name,
        'range': range_metrics,
        'doppler': doppler_metrics
    }
    
    return all_metrics


def plot_ambiguity_function(delay, doppler, af, plot_type='2d', normalize=True, title_suffix='', sll_info=None):
    """
    绘制模糊函数图
    
    参数:
        delay: 延迟向量 (s)
        doppler: 多普勒向量 (Hz)
        af: 模糊函数矩阵
        plot_type: '2d' 等高线图, '3d' 三维图, 'cut' 切割图, 'all' 全部
        normalize: 是否归一化
        title_suffix: 标题后缀
        sll_info: 旁瓣信息
    
    返回:
        sll: 旁瓣电平信息
    """
    af_mag = np.abs(af)
    if normalize:
        af_mag = af_mag / np.max(af_mag)
    
    zero_doppler_idx = np.argmin(np.abs(doppler))
    zero_delay_idx = np.argmin(np.abs(delay))
    range_sll = calculate_sll(af_mag[zero_doppler_idx, :])
    doppler_sll = calculate_sll(af_mag[:, zero_delay_idx])
    
    delay_ms = delay * 1000
    doppler_khz = doppler / 1000
    
    if plot_type in ['2d', 'all']:
        plt.figure(figsize=(10, 8))
        plt.contourf(delay_ms, doppler_khz, 20 * np.log10(af_mag + 1e-6), 
                     levels=np.linspace(-60, 0, 30), cmap=cm.jet)
        plt.colorbar(label='归一化幅度 (dB)')
        plt.xlabel('延迟 (ms)')
        plt.ylabel('多普勒频移 (kHz)')
        title = f'模糊函数 (二维等高线) {title_suffix}'
        title += f'\n距离SLL: {range_sll:.1f} dB, 多普勒SLL: {doppler_sll:.1f} dB'
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
    if plot_type in ['3d', 'all']:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(delay_ms, doppler_khz)
        af_db = 20 * np.log10(af_mag + 1e-6)
        af_db[af_db < -60] = -60
        ax.plot_surface(X, Y, af_db, cmap=cm.jet, linewidth=0, antialiased=True, vmin=-60, vmax=0)
        ax.set_xlabel('延迟 (ms)')
        ax.set_ylabel('多普勒频移 (kHz)')
        ax.set_zlabel('归一化幅度 (dB)')
        ax.set_zlim(-60, 0)
        ax.set_title(f'模糊函数 (三维图) {title_suffix}')
        
    if plot_type in ['cut', 'all']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        range_cut = af_mag[zero_doppler_idx, :]
        range_cut_db = 20 * np.log10(range_cut + 1e-6)
        
        axes[0].plot(delay_ms, range_cut_db, 'b-', linewidth=2)
        axes[0].set_xlabel('延迟 (ms)')
        axes[0].set_ylabel('归一化幅度 (dB)')
        title0 = f'零多普勒切割 (距离模糊) {title_suffix}'
        title0 += f'\nSLL: {range_sll:.1f} dB'
        axes[0].set_title(title0)
        axes[0].grid(True)
        axes[0].set_ylim([-60, 5])
        axes[0].axhline(y=-40, color='r', linestyle='--', alpha=0.5, label='-40 dB 目标')
        axes[0].legend()
        
        doppler_cut = af_mag[:, zero_delay_idx]
        doppler_cut_db = 20 * np.log10(doppler_cut + 1e-6)
        
        axes[1].plot(doppler_khz, doppler_cut_db, 'r-', linewidth=2)
        axes[1].set_xlabel('多普勒频移 (kHz)')
        axes[1].set_ylabel('归一化幅度 (dB)')
        title1 = f'零延迟切割 (多普勒模糊) {title_suffix}'
        title1 += f'\nSLL: {doppler_sll:.1f} dB'
        axes[1].set_title(title1)
        axes[1].grid(True)
        axes[1].set_ylim([-60, 5])
        axes[1].axhline(y=-40, color='r', linestyle='--', alpha=0.5, label='-40 dB 目标')
        axes[1].legend()
        
        plt.tight_layout()
    
    if plot_type != 'all':
        plt.show()
    
    return {'range_sll': range_sll, 'doppler_sll': doppler_sll}


def plot_window_comparison(windows, names):
    """
    对比不同窗函数的时域和频域特性
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    for w, name in zip(windows, names):
        N = len(w)
        t = np.arange(N) / N - 0.5
        
        axes[0].plot(t, w, linewidth=2, label=name)
        
        w_fft = np.fft.fftshift(np.fft.fft(w, 4096))
        w_fft_db = 20 * np.log10(np.abs(w_fft) / np.max(np.abs(w_fft)) + 1e-10)
        freq = np.linspace(-0.5, 0.5, len(w_fft))
        axes[1].plot(freq, w_fft_db, linewidth=2, label=name)
    
    axes[0].set_xlabel('归一化时间')
    axes[0].set_ylabel('幅度')
    axes[0].set_title('窗函数 - 时域')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel('归一化频率')
    axes[1].set_ylabel('幅度 (dB)')
    axes[1].set_title('窗函数 - 频域')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim([-80, 5])
    
    plt.tight_layout()
    plt.show()


def plot_multi_waveform_comparison(waveform_data, metrics_list):
    """
    绘制多波形模糊函数对比
    
    参数:
        waveform_data: 波形数据列表，每个元素为 (name, delay, doppler, af)
        metrics_list: 性能指标列表
    """
    n_waveforms = len(waveform_data)
    
    fig, axes = plt.subplots(2, n_waveforms, figsize=(6 * n_waveforms, 10))
    
    if n_waveforms == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (name, delay, doppler, af) in enumerate(waveform_data):
        af_mag = np.abs(af)
        af_mag = af_mag / np.max(af_mag)
        af_db = 20 * np.log10(af_mag + 1e-6)
        
        im = axes[0, idx].contourf(delay * 1000, doppler / 1000, af_db, 
                                   levels=np.linspace(-60, 0, 30), cmap=cm.jet)
        axes[0, idx].set_xlabel('延迟 (ms)')
        axes[0, idx].set_ylabel('多普勒频移 (kHz)')
        axes[0, idx].set_title(f'{name}\n模糊函数')
        axes[0, idx].grid(True, alpha=0.3)
        
        zero_doppler_idx = np.argmin(np.abs(doppler))
        zero_delay_idx = np.argmin(np.abs(delay))
        range_cut = af_mag[zero_doppler_idx, :]
        doppler_cut = af_mag[:, zero_delay_idx]
        
        range_cut_db = 20 * np.log10(range_cut + 1e-6)
        doppler_cut_db = 20 * np.log10(doppler_cut + 1e-6)
        
        metrics = metrics_list[idx]
        label = f'3dB宽度: {metrics["range"]["width_3dB"]*1e6:.1f}μs\nPSLR: {metrics["range"]["pslr_db"]:.1f}dB'
        axes[1, idx].plot(delay * 1000, range_cut_db, 'b-', linewidth=2, label=label)
        axes[1, idx].plot(delay * 1000, doppler_cut_db, 'r-', linewidth=2, 
                          label=f'多普勒PSLR: {metrics["doppler"]["pslr_db"]:.1f}dB')
        axes[1, idx].set_xlabel('延迟/多普勒 (归一化)')
        axes[1, idx].set_ylabel('幅度 (dB)')
        axes[1, idx].set_title(f'{name}\n距离/多普勒切割')
        axes[1, idx].grid(True)
        axes[1, idx].set_ylim([-60, 5])
        axes[1, idx].legend()
    
    plt.tight_layout()
    plt.show()


def print_performance_table(metrics_list):
    """
    打印性能对比表格
    """
    print("\n" + "=" * 100)
    print("多波形性能对比表")
    print("=" * 100)
    
    header = f"{'波形名称':<15} {'维度':<8} {'3dB宽度':>12} {'PSLR(dB)':>10} {'ISLR(dB)':>10} {'SLL(dB)':>10}"
    print(header)
    print("-" * 100)
    
    for metrics in metrics_list:
        name = metrics['waveform_name']
        for dim in ['range', 'doppler']:
            dim_name = '距离' if dim == 'range' else '多普勒'
            m = metrics[dim]
            width_unit = 'μs' if dim == 'range' else 'kHz'
            width = m['width_3dB'] * (1e6 if dim == 'range' else 1e-3)
            print(f"{name:<15} {dim_name:<8} {width:>10.1f}{width_unit} {m['pslr_db']:>10.1f} {m['islr_db']:>10.1f} {m['sll_db']:>10.1f}")
        print("-" * 100)


def main():
    """
    主函数：多波形模糊函数对比与多目标分辨性能评估
    """
    print("=" * 80)
    print("雷达信号模糊函数 - 多波形对比与多目标分辨性能评估")
    print("=" * 80)
    
    bandwidth = 10e6
    pulse_width = 10e-6
    fs = 2 * bandwidth
    oversample = 4
    num_chips_p4 = 32
    p_costas = 11
    chip_width = pulse_width / num_chips_p4
    
    print("\n参数设置:")
    print(f"  调频带宽: {bandwidth/1e6:.1f} MHz")
    print(f"  脉冲宽度: {pulse_width*1e6:.1f} μs")
    print(f"  基础采样率: {fs/1e6:.1f} MHz")
    print(f"  过采样倍数: {oversample}x")
    print(f"  实际采样率: {fs*oversample/1e6:.1f} MHz")
    print(f"  P4码芯片数: {num_chips_p4}")
    print(f"  Costas序列阶数: {p_costas}")
    print(f"  芯片宽度: {chip_width*1e6:.1f} μs")
    
    print("\n" + "=" * 80)
    print("1. 生成各波形并计算模糊函数")
    print("=" * 80)
    
    waveforms = []
    metrics_list = []
    
    print("\n  1.1 LFM波形...")
    t_lfm, s_lfm = generate_lfm_waveform(bandwidth, pulse_width, fs, oversample=oversample)
    delay_lfm, doppler_lfm, af_lfm = ambiguity_function_fft(s_lfm, fs * oversample, 
                                                             num_delay=300, num_doppler=300)
    metrics_lfm = evaluate_waveform_performance(delay_lfm, doppler_lfm, af_lfm, 'LFM')
    waveforms.append(('LFM', delay_lfm, doppler_lfm, af_lfm))
    metrics_list.append(metrics_lfm)
    print("    ✓ 完成")
    
    print("\n  1.2 LFM + Taylor窗 (-40dB)...")
    s_lfm_taylor, _ = apply_window(s_lfm, window_type='taylor', sll=-40)
    delay_lfm_t, doppler_lfm_t, af_lfm_t = ambiguity_function_fft(s_lfm_taylor, fs * oversample, 
                                                                    num_delay=300, num_doppler=300)
    metrics_lfm_t = evaluate_waveform_performance(delay_lfm_t, doppler_lfm_t, af_lfm_t, 'LFM+Taylor')
    waveforms.append(('LFM+Taylor', delay_lfm_t, doppler_lfm_t, af_lfm_t))
    metrics_list.append(metrics_lfm_t)
    print("    ✓ 完成")
    
    print("\n  1.3 P4码 (32芯片)...")
    t_p4, s_p4, _ = generate_p4_code(num_chips_p4, chip_width, fs, oversample=oversample)
    delay_p4, doppler_p4, af_p4 = ambiguity_function_fft(s_p4, fs * oversample, 
                                                          num_delay=300, num_doppler=300)
    metrics_p4 = evaluate_waveform_performance(delay_p4, doppler_p4, af_p4, 'P4码')
    waveforms.append(('P4码', delay_p4, doppler_p4, af_p4))
    metrics_list.append(metrics_p4)
    print("    ✓ 完成")
    
    print("\n  1.4 Costas序列 (11阶)...")
    t_costas, s_costas, _ = generate_costas_sequence(p_costas, chip_width, fs, oversample=oversample)
    delay_c, doppler_c, af_c = ambiguity_function_fft(s_costas, fs * oversample, 
                                                       num_delay=300, num_doppler=300)
    metrics_c = evaluate_waveform_performance(delay_c, doppler_c, af_c, 'Costas')
    waveforms.append(('Costas', delay_c, doppler_c, af_c))
    metrics_list.append(metrics_c)
    print("    ✓ 完成")
    
    print("\n" + "=" * 80)
    print("2. 多波形模糊函数对比图")
    print("=" * 80)
    plot_multi_waveform_comparison(waveforms, metrics_list)
    
    print("\n" + "=" * 80)
    print("3. 各波形详细模糊函数图")
    print("=" * 80)
    
    for name, delay, doppler, af in waveforms:
        print(f"\n  {name}...")
        plot_ambiguity_function(delay, doppler, af, plot_type='2d', title_suffix=f'- {name}')
    
    print("\n" + "=" * 80)
    print("4. 多目标分辨性能对比")
    print("=" * 80)
    print_performance_table(metrics_list)
    
    print("\n" + "=" * 80)
    print("5. 多目标分辨能力分析")
    print("=" * 80)
    
    print("\n距离分辨能力排序 (3dB主瓣宽度越窄越好):")
    sorted_by_range = sorted(metrics_list, key=lambda x: x['range']['width_3dB'])
    for i, m in enumerate(sorted_by_range, 1):
        width_us = m['range']['width_3dB'] * 1e6
        range_res_m = 3e8 * m['range']['width_3dB'] / 2
        print(f"  {i}. {m['waveform_name']:<15}: {width_us:>8.2f} μs ≈ {range_res_m:>7.2f} m")
    
    print("\n多普勒分辨能力排序 (3dB主瓣宽度越窄越好):")
    sorted_by_doppler = sorted(metrics_list, key=lambda x: x['doppler']['width_3dB'])
    for i, m in enumerate(sorted_by_doppler, 1):
        width_khz = m['doppler']['width_3dB'] / 1e3
        print(f"  {i}. {m['waveform_name']:<15}: {width_khz:>8.2f} kHz")
    
    print("\n旁瓣抑制能力排序 (PSLR越低越好):")
    sorted_by_pslr = sorted(metrics_list, key=lambda x: x['range']['pslr_db'])
    for i, m in enumerate(sorted_by_pslr, 1):
        print(f"  {i}. {m['waveform_name']:<15}: 距离PSLR {m['range']['pslr_db']:>7.2f} dB, 多普勒PSLR {m['doppler']['pslr_db']:>7.2f} dB")
    
    print("\n" + "=" * 80)
    print("总结与建议")
    print("=" * 80)
    
    print("\n波形特性分析:")
    print("  ✓ LFM: 经典波形，多普勒容忍性好，距离旁瓣约-13dB")
    print("  ✓ LFM+Taylor: 旁瓣抑制至-40dB以下，主瓣略有展宽")
    print("  ✓ P4码: 多相编码，多普勒敏感性适中，距离旁瓣较低")
    print("  ✓ Costas: 频率编码，理想图钉型模糊函数，多普勒容忍性好")
    
    print("\n多目标场景建议:")
    print("  - 强杂波环境: 使用LFM+Taylor窗 (低旁瓣)")
    print("  - 需要高距离分辨率: 使用P4码/Costas序列")
    print("  - 高速目标场景: 使用Costas序列 (多普勒容忍性好)")
    print("  - 简单工程实现: 使用LFM")
    
    print("\n" + "=" * 80)
    plt.show()


if __name__ == "__main__":
    main()
