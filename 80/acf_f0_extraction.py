import numpy as np
import librosa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def crepe_f0_extraction(audio_path, fs=None, model_capacity='full', 
                       voice_type=None, auto_range=False,
                       min_f0=None, max_f0=None, conf_threshold=0.5,
                       step_size=10, silent=False):
    """
    CREPE深度学习基频提取（高精度，适合噪声环境）
    
    CREPE是一个基于CNN的深度学习基频提取器，在噪声环境下表现优异
    
    参数:
        audio_path: 音频文件路径
        fs: 采样率，如果为None则使用文件的原始采样率
        model_capacity: 模型容量: 'tiny', 'small', 'medium', 'large', 'full'
                       容量越大精度越高，但速度越慢
        voice_type: 语音类型预设，用于设置基频范围
        auto_range: 是否自适应估计基频范围
        min_f0: 最小基频（Hz）
        max_f0: 最大基频（Hz）
        conf_threshold: 置信度阈值，低于此值的帧视为清音
        step_size: 分析步长（毫秒）
        silent: 是否静默模式
    
    返回:
        f0: 基频轨迹数组（Hz）
        times: 对应时间点数组（秒）
    """
    try:
        import crepe
    except ImportError:
        raise ImportError("CREPE未安装，请运行: pip install crepe")
    
    if min_f0 is None or max_f0 is None:
        if voice_type is not None:
            min_f0, max_f0 = get_voice_preset(voice_type)
            if not silent:
                print(f"[CREPE] 使用语音类型预设 '{voice_type}': {min_f0}-{max_f0} Hz")
        else:
            min_f0, max_f0 = get_voice_preset('universal')
            if not silent:
                print(f"[CREPE] 使用通用基频范围: {min_f0}-{max_f0} Hz")
    
    if fs is None:
        y, sr = librosa.load(audio_path, sr=None)
    else:
        y, sr = librosa.load(audio_path, sr=fs)
    
    if not silent:
        print(f"[CREPE] 加载模型: {model_capacity}")
    
    time, frequency, confidence, _ = crepe.predict(
        y, sr, 
        model_capacity=model_capacity,
        viterbi=True,
        step_size=step_size,
        verbose=0
    )
    
    f0 = frequency.copy()
    f0[(confidence < conf_threshold)] = 0
    
    f0[(frequency < min_f0) | (frequency > max_f0)] = 0
    
    if not silent:
        print(f"[CREPE] 提取完成，浊音率: {np.mean(f0 > 0):.1%}")
    
    return f0, time


def pyin_f0_extraction(audio_path, fs=None, voice_type=None, 
                       min_f0=None, max_f0=None, frame_duration=0.03,
                       hop_duration=0.015, resolution=0.01,
                       boltzmann_parameter=2.0, max_transition_rate=35.92,
                       switch_prob=0.01, silent=False):
    """
    pYIN概率基频提取（高精度，鲁棒性强）
    
    pYIN是YIN算法的概率版本，使用Viterbi解码，精度和鲁棒性都优于传统ACF
    
    参数:
        audio_path: 音频文件路径
        fs: 采样率，如果为None则使用文件的原始采样率
        voice_type: 语音类型预设，用于设置基频范围
        min_f0: 最小基频（Hz）
        max_f0: 最大基频（Hz）
        frame_duration: 帧长（秒）
        hop_duration: 帧移（秒）
        resolution: F0搜索分辨率（半音的分数）
        boltzmann_parameter: Boltzmann分布参数
        max_transition_rate: 最大转换率（半音/秒）
        switch_prob: 清浊切换概率
        silent: 是否静默模式
    
    返回:
        f0: 基频轨迹数组（Hz）
        times: 对应时间点数组（秒）
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa未安装，请运行: pip install librosa")
    
    if min_f0 is None or max_f0 is None:
        if voice_type is not None:
            min_f0, max_f0 = get_voice_preset(voice_type)
            if not silent:
                print(f"[pYIN] 使用语音类型预设 '{voice_type}': {min_f0}-{max_f0} Hz")
        else:
            min_f0, max_f0 = get_voice_preset('universal')
            if not silent:
                print(f"[pYIN] 使用通用基频范围: {min_f0}-{max_f0} Hz")
    
    if fs is None:
        y, sr = librosa.load(audio_path, sr=None)
    else:
        y, sr = librosa.load(audio_path, sr=fs)
    
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    
    if min_f0 is not None:
        fmin = min_f0
    if max_f0 is not None:
        fmax = max_f0
    
    if not silent:
        print(f"[pYIN] 基频范围: {fmin:.1f}-{fmax:.1f} Hz")
    
    hop_length = int(hop_duration * sr)
    frame_length = int(frame_duration * sr)
    
    f0, _, _ = librosa.pyin(
        y, 
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        resolution=resolution,
        boltzmann_parameter=boltzmann_parameter,
        max_transition_rate=max_transition_rate,
        switch_prob=switch_prob
    )
    
    f0 = np.nan_to_num(f0, nan=0.0)
    
    times = np.arange(len(f0)) * hop_duration
    
    if not silent:
        print(f"[pYIN] 提取完成，浊音率: {np.mean(f0 > 0):.1%}")
    
    return f0, times


METHODS_DOC = {
    'acf': '自相关法 - 速度快，适合高信噪比语音',
    'pyin': 'pYIN - 概率YIN算法，精度高，鲁棒性强',
    'crepe': 'CREPE - 深度学习CNN，噪声环境表现优异'
}


def extract_f0(audio_path, method='acf', **kwargs):
    """
    统一的基频提取接口，支持多种算法
    
    参数:
        audio_path: 音频文件路径
        method: 提取方法: 'acf', 'pyin', 'crepe'
        **kwargs: 传递给具体算法的参数
    
    返回:
        f0: 基频轨迹数组（Hz）
        times: 对应时间点数组（秒）
    
    示例:
        # 自相关法（快速）
        f0, times = extract_f0('audio.wav', method='acf', voice_type='male')
        
        # pYIN（高精度）
        f0, times = extract_f0('audio.wav', method='pyin', voice_type='female')
        
        # CREPE（深度学习，抗噪声）
        f0, times = extract_f0('audio.wav', method='crepe', model_capacity='large')
    """
    method = method.lower()
    
    if method == 'acf':
        return acf_f0_extraction(audio_path, **kwargs)
    elif method == 'pyin':
        return pyin_f0_extraction(audio_path, **kwargs)
    elif method == 'crepe':
        return crepe_f0_extraction(audio_path, **kwargs)
    else:
        raise ValueError(f"未知的方法: {method}，可选: acf, pyin, crepe")


def compare_methods(audio_path, methods=['acf', 'pyin', 'crepe'], 
                    voice_type=None, save_path='f0_comparison.png'):
    """
    对比多种基频提取方法的结果
    
    参数:
        audio_path: 音频文件路径
        methods: 要对比的方法列表
        voice_type: 语音类型预设
        save_path: 保存图片路径
    """
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    print(f"音频长度: {duration:.2f}秒, 采样率: {sr} Hz")
    print(f"对比方法: {', '.join(methods)}")
    print("=" * 60)
    
    results = {}
    for method in methods:
        print(f"\n正在使用 {method.upper()} 提取...")
        try:
            f0, times = extract_f0(audio_path, method=method, 
                                   voice_type=voice_type, silent=True)
            results[method] = (f0, times)
            
            voiced = f0 > 0
            if np.any(voiced):
                print(f"  {method.upper()}: 平均F0={np.mean(f0[voiced]):.1f}Hz, "
                      f"范围={np.min(f0[voiced]):.1f}-{np.max(f0[voiced]):.1f}Hz, "
                      f"浊音率={np.mean(voiced):.1%}")
            else:
                print(f"  {method.upper()}: 未检测到浊音")
        except Exception as e:
            print(f"  {method.upper()}: 失败 - {e}")
    
    n_methods = len(results)
    if n_methods == 0:
        print("没有成功的方法")
        return
    
    fig, axes = plt.subplots(n_methods + 1, 1, figsize=(14, 4 * (n_methods + 1)))
    
    axes[0].plot(np.linspace(0, duration, len(y)), y, 'k-', linewidth=0.5)
    axes[0].set_title('语音波形')
    axes[0].set_ylabel('幅值')
    axes[0].grid(True, alpha=0.3)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (method, (f0, times)) in enumerate(results.items()):
        ax = axes[i + 1]
        voiced = f0 > 0
        ax.plot(times[voiced], f0[voiced], '.', color=colors[i % len(colors)], 
                markersize=3, label=method.upper())
        ax.set_title(f'{method.upper()} - {METHODS_DOC[method]}')
        ax.set_ylabel('基频 (Hz)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('时间 (秒)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n对比图已保存: {save_path}")
    plt.show()


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
    """
    frame_length = int(frame_duration * sr)
    hop_length = int(len(y) / (n_samples + 1))
    
    candidate_f0s = []
    
    for i in range(n_samples):
        start = (i + 1) * hop_length
        if start + frame_length > len(y):
            break
        frame = y[start:start + frame_length] * np.hamming(frame_length)
        
        acf = np.correlate(frame, frame, mode='full')
        acf = acf[frame_length - 1:]
        
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


def acf_f0_extraction(audio_path, fs=None, frame_duration=0.03, hop_duration=0.015, 
                     min_f0=None, max_f0=None, voice_type=None, auto_range=False,
                     pre_emphasis_coeff=0.97, window_type='hamming', 
                     median_filter_order=3, harmonic_removal=True, silent=False):
    """
    自相关法（ACF）提取语音基频F0
    
    参数:
        audio_path: 音频文件路径
        fs: 采样率，如果为None则使用文件的原始采样率
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
        silent: 是否静默模式（不打印信息）
    
    返回:
        f0: 基频轨迹数组（Hz）
        times: 对应时间点数组（秒）
    """
    if fs is None:
        y, sr = librosa.load(audio_path, sr=None)
    else:
        y, sr = librosa.load(audio_path, sr=fs)
    
    if min_f0 is None or max_f0 is None:
        if voice_type is not None:
            min_f0, max_f0 = get_voice_preset(voice_type)
            if not silent:
                print(f"使用语音类型预设 '{voice_type}': {min_f0}-{max_f0} Hz")
        elif auto_range:
            min_f0, max_f0 = estimate_voice_range(y, sr)
            if not silent:
                print(f"自适应估计基频范围: {min_f0}-{max_f0} Hz")
        else:
            min_f0, max_f0 = get_voice_preset('universal')
            if not silent:
                print(f"使用通用基频范围: {min_f0}-{max_f0} Hz")
    
    if pre_emphasis_coeff > 0:
        y = np.append(y[0], y[1:] - pre_emphasis_coeff * y[:-1])
    
    frame_length = int(frame_duration * sr)
    hop_length = int(hop_duration * sr)
    
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
        
        acf = np.correlate(frame, frame, mode='full')
        acf = acf[frame_length - 1:]
        
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
        from scipy.signal import medfilt
        f0 = medfilt(f0, kernel_size=median_filter_order)
    
    return f0, times


def plot_f0(f0, times, title='F0轨迹 (自相关法)'):
    """
    绘制F0轨迹
    """
    plt.figure(figsize=(12, 6))
    
    voiced = f0 > 0
    plt.plot(times[voiced], f0[voiced], 'b.', markersize=3, label='浊音段')
    plt.xlabel('时间 (秒)')
    plt.ylabel('基频 (Hz)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def extract_f0_statistics(f0):
    """
    提取F0统计信息
    """
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) == 0:
        return None
    
    stats = {
        'mean_f0': np.mean(voiced_f0),
        'std_f0': np.std(voiced_f0),
        'min_f0': np.min(voiced_f0),
        'max_f0': np.max(voiced_f0),
        'median_f0': np.median(voiced_f0),
        'voiced_ratio': len(voiced_f0) / len(f0)
    }
    return stats


if __name__ == '__main__':
    print("基频提取工具 - 支持多种算法和语音类型")
    print("=" * 70)
    
    print("\n可用的提取方法:")
    for key, value in METHODS_DOC.items():
        print(f"  {key:8s}: {value}")
    
    print("\n可用的语音类型预设:")
    for key, value in VOICE_TYPE_PRESETS.items():
        print(f"  {key:12s}: {value['description']}")
    
    print("\n" + "=" * 70)
    print("使用方法:")
    print("-" * 70)
    print("# 自相关法（快速）")
    print("  f0, times = extract_f0('audio.wav', method='acf', voice_type='male')")
    print("\n# pYIN（高精度，鲁棒）")
    print("  f0, times = extract_f0('audio.wav', method='pyin', voice_type='female')")
    print("\n# CREPE（深度学习，抗噪声）")
    print("  f0, times = extract_f0('audio.wav', method='crepe', model_capacity='large')")
    print("\n# 对比多种方法")
    print("  compare_methods('audio.wav', methods=['acf', 'pyin', 'crepe'])")
    
    print("\n" + "=" * 70)
    print("生成带噪声的测试信号并演示...")
    print("-" * 70)
    
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    f0_true = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] < 0.6:
            f0_true[i] = 80
        elif t[i] < 1.2:
            f0_true[i] = 150
        elif t[i] < 1.8:
            f0_true[i] = 250
        elif t[i] < 2.4:
            f0_true[i] = 350
        else:
            f0_true[i] = 0
    
    test_signal = np.zeros_like(t)
    for i in range(len(t)):
        if f0_true[i] > 0:
            phase = 2 * np.pi * f0_true[i] * t[i]
            test_signal[i] = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.125 * np.sin(3 * phase)
    
    noise_level = 0.15
    test_signal += noise_level * np.random.randn(len(test_signal))
    
    import soundfile as sf
    sf.write('test_signal.wav', test_signal, sr)
    print(f"已生成测试信号: test_signal.wav (噪声水平: {noise_level})")
    print("测试信号F0变化: 80Hz -> 150Hz -> 250Hz -> 350Hz")
    
    print("\n" + "=" * 70)
    print("运行方法对比（ACF vs pYIN）")
    print("-" * 70)
    print("注: CREPE需要下载模型，首次运行会自动下载")
    
    test_methods = ['acf', 'pyin']
    
    try:
        import crepe
        test_methods.append('crepe')
        print("CREPE已安装，将包含在对比中")
    except ImportError:
        print("CREPE未安装，跳过 (安装: pip install crepe)")
    
    results = {}
    for method in test_methods:
        print(f"\n正在使用 {method.upper()} 提取...")
        try:
            f0, times = extract_f0('test_signal.wav', method=method, 
                                   voice_type='universal', silent=True)
            results[method] = (f0, times)
            
            voiced = f0 > 0
            if np.any(voiced):
                f0_voiced = f0[voiced]
                mae = np.mean(np.abs(f0_voiced - 
                    np.interp(times[voiced], t, f0_true)))
                print(f"  {method.upper()}: 平均F0={np.mean(f0_voiced):.1f}Hz, "
                      f"MAE={mae:.1f}Hz, 浊音率={np.mean(voiced):.1%}")
            else:
                print(f"  {method.upper()}: 未检测到浊音")
        except Exception as e:
            print(f"  {method.upper()}: 失败 - {e}")
    
    n_methods = len(results)
    if n_methods > 0:
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=(14, 4 * (n_methods + 1)))
        
        axes[0].plot(t, test_signal, 'k-', linewidth=0.5)
        axes[0].set_title(f'带噪声的语音信号 (噪声水平: {noise_level})')
        axes[0].set_ylabel('幅值')
        axes[0].grid(True, alpha=0.3)
        
        colors = ['blue', 'red', 'green']
        for i, (method, (f0, times)) in enumerate(results.items()):
            ax = axes[i + 1]
            voiced = f0 > 0
            ax.plot(times[voiced], f0[voiced], '.', color=colors[i], 
                    markersize=4, label=f'{method.upper()} 提取')
            ax.plot(t, f0_true, 'r--', alpha=0.7, linewidth=2, label='真实F0')
            ax.set_title(f'{method.upper()} - {METHODS_DOC[method]}')
            ax.set_ylabel('基频 (Hz)')
            ax.set_ylim(0, 450)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        axes[-1].set_xlabel('时间 (秒)')
        plt.tight_layout()
        plt.savefig('f0_methods_comparison.png', dpi=150)
        print(f"\n对比图已保存: f0_methods_comparison.png")
        plt.show()
    
    print("\n" + "=" * 70)
    print("安装CREPE的方法: pip install crepe tensorflow")
    print("TensorFlow可能需要单独安装，请根据系统配置选择")
    print("=" * 70)
