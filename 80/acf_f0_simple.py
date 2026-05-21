import numpy as np
from scipy.signal import medfilt


VOICE_TYPE_PRESETS = {
    'male': {'min_f0': 50, 'max_f0': 250, 'description': '男声：低音域，50-250Hz'},
    'female': {'min_f0': 100, 'max_f0': 500, 'description': '女声：中音域，100-500Hz'},
    'child': {'min_f0': 200, 'max_f0': 800, 'description': '儿童：高音域，200-800Hz'},
    'tenor': {'min_f0': 80, 'max_f0': 350, 'description': '男高音：80-350Hz'},
    'bass': {'min_f0': 40, 'max_f0': 200, 'description': '男低音：40-200Hz'},
    'soprano': {'min_f0': 180, 'max_f0': 700, 'description': '女高音：180-700Hz'},
    'alto': {'min_f0': 120, 'max_f0': 450, 'description': '女低音：120-450Hz'},
    'universal': {'min_f0': 40, 'max_f0': 800, 'description': '通用：40-800Hz，适用所有类型'},
    'auto': {'min_f0': 40, 'max_f0': 1000, 'description': '自动：40-1000Hz，配合自适应调整'}
}


def get_voice_preset(voice_type):
    """
    获取语音类型预设的基频范围
    
    参数:
        voice_type: 语音类型，可选值：
                   'male', 'female', 'child', 'tenor', 'bass', 
                   'soprano', 'alto', 'universal', 'auto'
    
    返回:
        (min_f0, max_f0): 基频范围元组
    """
    voice_type = voice_type.lower()
    if voice_type in VOICE_TYPE_PRESETS:
        preset = VOICE_TYPE_PRESETS[voice_type]
        return preset['min_f0'], preset['max_f0']
    else:
        print(f"警告：未知语音类型 '{voice_type}'，使用通用预设")
        return 40, 800


def estimate_voice_range(y, sr, frame_duration=0.03, n_samples=20):
    """
    自适应估计语音基频范围
    
    参数:
        y: 语音信号
        sr: 采样率
        frame_duration: 帧长
        n_samples: 采样帧数
    
    返回:
        (min_f0, max_f0): 估计的基频范围
    """
    frame_length = int(frame_duration * sr)
    hop_length = int(len(y) / (n_samples + 1))
    
    candidate_f0s = []
    
    for i in range(n_samples):
        start = (i + 1) * hop_length
        if start + frame_length > len(y):
            break
        frame = y[start:start + frame_length] * np.hamming(frame_length)
        
        acf = _fast_acf(frame)
        
        if acf[0] > 0:
            wide_min_period = int(sr / 1000)
            wide_max_period = int(sr / 40)
            acf_roi = acf[wide_min_period:wide_max_period + 1]
            
            peak_ratio = np.max(acf_roi) / acf[0]
            if peak_ratio > 0.05:
                peak_idx = np.argmax(acf_roi)
                period = wide_min_period + peak_idx
                f0_candidate = sr / period
                if 40 <= f0_candidate <= 1000:
                    candidate_f0s.append(f0_candidate)
    
    if len(candidate_f0s) >= 5:
        candidate_f0s = np.array(candidate_f0s)
        median_f0 = np.median(candidate_f0s)
        std_f0 = np.std(candidate_f0s)
        
        min_f0 = max(40, median_f0 - 2 * std_f0)
        max_f0 = min(1000, median_f0 + 2 * std_f0)
        
        min_f0 = max(40, min_f0 * 0.7)
        max_f0 = min(1000, max_f0 * 1.3)
        
        return int(min_f0), int(max_f0)
    else:
        return 60, 600


def acf_f0(y, sr, frame_duration=0.03, hop_duration=0.015, 
           min_f0=None, max_f0=None, voice_type=None, auto_range=False,
           pre_emphasis_coeff=0.97, window_type='hamming', 
           median_filter_order=3, harmonic_removal=True):
    """
    自相关法（ACF）提取语音基频F0（简化版本，直接处理numpy数组）
    
    参数:
        y: 语音信号数组 (1D numpy array)
        sr: 采样率 (Hz)
        frame_duration: 帧长（秒），默认30ms
        hop_duration: 帧移（秒），默认15ms
        min_f0: 最小基频（Hz），如果为None则使用voice_type或默认值
        max_f0: 最大基频（Hz），如果为None则使用voice_type或默认值
        voice_type: 语音类型预设：'male', 'female', 'child', 'tenor', 'bass', 
                   'soprano', 'alto', 'universal', 'auto'
        auto_range: 是否自适应估计基频范围
        pre_emphasis_coeff: 预加重系数，默认0.97
        window_type: 窗函数类型，'hamming', 'hann', 'rectangle'
        median_filter_order: 中值滤波阶数，用于平滑F0轨迹
        harmonic_removal: 是否启用倍频错误校正
    
    返回:
        f0: 基频轨迹数组（Hz）
        times: 对应时间点数组（秒）
    """
    y = np.asarray(y, dtype=np.float64)
    
    if min_f0 is None or max_f0 is None:
        if voice_type is not None:
            min_f0, max_f0 = get_voice_preset(voice_type)
            print(f"使用语音类型预设 '{voice_type}': {min_f0}-{max_f0} Hz")
        elif auto_range:
            min_f0, max_f0 = estimate_voice_range(y, sr)
            print(f"自适应估计基频范围: {min_f0}-{max_f0} Hz")
        else:
            min_f0, max_f0 = get_voice_preset('universal')
            print(f"使用通用基频范围: {min_f0}-{max_f0} Hz")
    
    if pre_emphasis_coeff > 0:
        y = np.append(y[0], y[1:] - pre_emphasis_coeff * y[:-1])
    
    frame_length = int(frame_duration * sr)
    hop_length = int(hop_duration * sr)
    
    if len(y) < frame_length:
        raise ValueError(f"信号长度 ({len(y)}) 小于帧长 ({frame_length})")
    
    n_frames = 1 + int((len(y) - frame_length) / hop_length)
    times = np.arange(n_frames) * hop_duration
    
    min_period = int(sr / max_f0)
    max_period = int(sr / min_f0)
    
    if window_type == 'hamming':
        window = np.hamming(frame_length)
    elif window_type == 'hann':
        window = np.hanning(frame_length)
    else:
        window = np.ones(frame_length)
    
    f0 = np.zeros(n_frames)
    f0_candidates = []
    
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + frame_length] * window
        
        acf = _fast_acf(frame)
        
        acf_roi = acf[min_period:max_period + 1]
        
        if len(acf_roi) > 0 and acf[0] > 0:
            peak_ratio = np.max(acf_roi) / acf[0]
            if peak_ratio > 0.1:
                peak_idx = np.argmax(acf_roi)
                period = min_period + peak_idx
                f0_value = sr / period
                
                if harmonic_removal:
                    f0_candidates.append(f0_value)
                
                f0[i] = f0_value
            else:
                f0[i] = 0
        else:
            f0[i] = 0
    
    if harmonic_removal and len(f0_candidates) >= 10:
        f0_candidates = np.array(f0_candidates)
        median_f0 = np.median(f0_candidates)
        
        for i in range(n_frames):
            if f0[i] > 0:
                ratio = f0[i] / median_f0
                if 1.8 < ratio < 2.2:
                    f0[i] = f0[i] / 2
                elif 2.8 < ratio < 3.2:
                    f0[i] = f0[i] / 3
                elif 0.4 < ratio < 0.6:
                    f0[i] = f0[i] * 2
    
    if median_filter_order > 1:
        f0 = medfilt(f0, kernel_size=median_filter_order)
    
    return f0, times


def _fast_acf(x):
    """
    使用FFT快速计算自相关函数
    """
    N = len(x)
    fft_x = np.fft.fft(x, n=2*N)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real
    return acf[:N]


def compute_acf(frame):
    """
    计算单帧的自相关函数（用于可视化）
    """
    return _fast_acf(frame)


def f0_smoothing(f0, method='median', kernel_size=3):
    """
    F0轨迹平滑处理
    """
    if method == 'median':
        return medfilt(f0, kernel_size=kernel_size)
    elif method == 'mean':
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(f0, size=kernel_size)
    else:
        return f0


def get_voiced_f0(f0):
    """
    获取浊音段的F0值
    """
    return f0[f0 > 0]


if __name__ == '__main__':
    print("自相关法基频提取 - 支持多种语音类型")
    print("=" * 60)
    
    print("\n可用的语音类型预设:")
    for key, value in VOICE_TYPE_PRESETS.items():
        print(f"  {key:12s}: {value['description']}")
    
    print("\n" + "=" * 60)
    print("演示1: 男声（低音 60Hz)")
    print("-" * 60)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    f0_male = 60
    signal_male = (0.5 * np.sin(2 * np.pi * f0_male * t) + 
                   0.25 * np.sin(4 * np.pi * f0_male * t) +
                   0.125 * np.sin(6 * np.pi * f0_male * t))
    signal_male += 0.03 * np.random.randn(len(signal_male))
    
    f0_male_ext, times_male = acf_f0(signal_male, sr, voice_type='male', auto_range=False)
    voiced_male = get_voiced_f0(f0_male_ext)
    
    print(f"目标基频: {f0_male} Hz")
    print(f"提取平均基频: {np.mean(voiced_male):.2f} Hz")
    print(f"提取基频标准差: {np.std(voiced_male):.2f} Hz")
    print(f"浊音帧数: {len(voiced_male)} / {len(f0_male_ext)}")
    
    print("\n" + "=" * 60)
    print("演示2: 女声（高音 300Hz)")
    print("-" * 60)
    
    f0_female = 300
    signal_female = (0.5 * np.sin(2 * np.pi * f0_female * t) + 
                      0.25 * np.sin(4 * np.pi * f0_female * t) +
                      0.125 * np.sin(6 * np.pi * f0_female * t))
    signal_female += 0.03 * np.random.randn(len(signal_female))
    
    f0_female_ext, times_female = acf_f0(signal_female, sr, voice_type='female', auto_range=False)
    voiced_female = get_voiced_f0(f0_female_ext)
    
    print(f"目标基频: {f0_female} Hz")
    print(f"提取平均基频: {np.mean(voiced_female):.2f} Hz")
    print(f"提取基频标准差: {np.std(voiced_female):.2f} Hz")
    print(f"浊音帧数: {len(voiced_female)} / {len(f0_female_ext)}")
    
    print("\n" + "=" * 60)
    print("演示3: 自适应基频范围检测")
    print("-" * 60)
    
    f0_child = 400
    signal_child = (0.5 * np.sin(2 * np.pi * f0_child * t) + 
                   0.25 * np.sin(4 * np.pi * f0_child * t) +
                   0.125 * np.sin(6 * np.pi * f0_child * t))
    signal_child += 0.03 * np.random.randn(len(signal_child))
    
    f0_child_ext, times_child = acf_f0(signal_child, sr, auto_range=True)
    voiced_child = get_voiced_f0(f0_child_ext)
    
    print(f"目标基频: {f0_child} Hz")
    print(f"提取平均基频: {np.mean(voiced_child):.2f} Hz")
    print(f"提取基频标准差: {np.std(voiced_child):.2f} Hz")
    
    print("\n" + "=" * 60)
    print("使用示例:")
    print("-" * 60)
    print("# 男声: f0, times = acf_f0(signal, sr, voice_type='male')")
    print("# 女声: f0, times = acf_f0(signal, sr, voice_type='female')")
    print("# 儿童: f0, times = acf_f0(signal, sr, voice_type='child')")
    print("# 自适应: f0, times = acf_f0(signal, sr, auto_range=True)")
    print("# 自定义: f0, times = acf_f0(signal, sr, min_f0=50, max_f0=500")
