import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


def generate_gearbox_signal(fs=10000, duration=2.0, fault_type=None, seed=42):
    """
    生成齿轮箱振动仿真信号
    
    参数:
        fs: 采样频率 (Hz)
        duration: 信号时长 (s)
        fault_type: 故障类型: None(正常), 'wear'(齿面剥落), 'broken'(断齿)
        seed: 随机种子
    
    返回:
        t: 时间序列
        x: 振动信号
        params: 信号参数字典
    """
    np.random.seed(seed)
    N = int(fs * duration)
    t = np.linspace(0, duration, N, endpoint=False)
    
    fr = 30.0
    n_teeth = 25
    f_mesh = fr * n_teeth
    f_shaft = fr
    
    x = np.zeros_like(t)
    
    if fault_type is None:
        mesh_amp = 1.0
        phase = np.random.uniform(0, 2*np.pi)
        x += mesh_amp * np.sin(2 * np.pi * f_mesh * t + phase)
        x += 0.3 * np.sin(2 * np.pi * 2 * f_mesh * t + np.random.uniform(0, 2*np.pi))
        x += 0.1 * np.sin(2 * np.pi * 3 * f_mesh * t + np.random.uniform(0, 2*np.pi))
    
    elif fault_type == 'wear':
        wear_amp = 1.2
        x += wear_amp * (1 + 0.5 * np.sin(2 * np.pi * f_shaft * t)) * \
              np.sin(2 * np.pi * f_mesh * t + np.random.uniform(0, 2*np.pi))
        x += wear_amp * 0.6 * (1 + 0.4 * np.sin(2 * np.pi * f_shaft * t)) * \
              np.sin(2 * np.pi * 2 * f_mesh * t + np.random.uniform(0, 2*np.pi))
        x += wear_amp * 0.3 * (1 + 0.3 * np.sin(2 * np.pi * f_shaft * t)) * \
              np.sin(2 * np.pi * 3 * f_mesh * t + np.random.uniform(0, 2*np.pi))
    
    elif fault_type == 'broken':
        impact_interval = 1.0 / fr
        impact_amp = 3.0
        decay = 25.0
        resonant_freq = f_mesh
        
        for k in range(int(duration / impact_interval) + 1):
            t0 = k * impact_interval
            idx = np.where((t >= t0) & (t < t0 + 0.15))[0]
            if len(idx) > 0:
                t_rel = t[idx] - t0
                impact = impact_amp * np.exp(-decay * t_rel) * \
                         np.sin(2 * np.pi * resonant_freq * t_rel)
                x[idx] += impact
        
        mesh_amp = 0.5
        x += mesh_amp * np.sin(2 * np.pi * f_mesh * t + np.random.uniform(0, 2*np.pi))
    
    noise = 0.25 * np.random.randn(N)
    x += noise
    
    params = {
        'fs': fs,
        'fr': fr,
        'n_teeth': n_teeth,
        'f_mesh': f_mesh,
        'f_shaft': f_shaft,
        'fault_type': fault_type
    }
    
    return t, x, params


def hilbert_envelope(x):
    """
    希尔伯特变换求包络
    
    参数:
        x: 输入信号
    
    返回:
        envelope: 包络信号
        analytic: 解析信号
    """
    analytic = signal.hilbert(x)
    envelope = np.abs(analytic)
    return envelope, analytic


def med_filter(x, L=31, max_iter=30, tol=1e-4):
    """
    最小熵反卷积 (Minimum Entropy Deconvolution, MED)
    
    用于恢复信号中的脉冲成分，增强故障冲击特征
    
    参数:
        x: 输入信号
        L: 滤波器长度 (推荐奇数)
        max_iter: 最大迭代次数
        tol: 收敛阈值
    
    返回:
        y: MED滤波后信号
        f: MED滤波器系数
    """
    N = len(x)
    x = x - np.mean(x)
    
    f = np.zeros(L)
    f[L//2] = 1.0
    
    X = np.zeros((N - L + 1, L))
    for i in range(L):
        X[:, i] = x[i:N - L + 1 + i]
    
    for _ in range(max_iter):
        y = X @ f
        
        y_abs = np.abs(y)
        y_abs3 = y_abs ** 3
        y_abs4 = y_abs ** 4
        
        sum_y4 = np.sum(y_abs4)
        if sum_y4 < 1e-12:
            break
        
        norm = np.sum(y_abs3 * y)
        if abs(norm) < 1e-12:
            break
        
        f_new = (X.T @ (y_abs3 * y)) / norm
        f_new = f_new / np.linalg.norm(f_new)
        
        f_new[f_new.argmax()] = abs(f_new[f_new.argmax()])
        
        if np.linalg.norm(f_new - f) < tol:
            f = f_new
            break
        f = f_new
    
    y = X @ f
    
    y_full = np.zeros(N)
    y_full[L//2 : N - L//2] = y
    
    return y_full, f


def spectral_kurtosis(x, fs, n_seg=100, overlap=0.5, window='hann'):
    """
    计算谱峭度 (Spectral Kurtosis, SK)
    
    参数:
        x: 输入信号
        fs: 采样频率
        n_seg: STFT的段数
        overlap: 重叠比例
        window: 窗函数
    
    返回:
        freqs: 频率轴
        sk: 谱峭度值
    """
    N = len(x)
    nperseg = N // n_seg
    noverlap = int(nperseg * overlap)
    
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                            window=window, return_onesided=True)
    
    X = np.abs(Zxx)
    
    E1 = np.mean(X, axis=1)
    E2 = np.mean(X**2, axis=1)
    E4 = np.mean(X**4, axis=1)
    
    sk = np.zeros_like(E1)
    valid = E2 > 1e-12
    sk[valid] = E4[valid] / (E2[valid] ** 2) - 2.0
    
    return f, sk


def find_optimal_band_sk(x, fs, f_min=100, f_max=None, band_width=200, 
                         f_prior=None, prior_weight=0.3):
    """
    基于谱峭度寻找最优解调频带（支持先验频率加权）
    
    参数:
        x: 输入信号
        fs: 采样频率
        f_min: 最小搜索频率
        f_max: 最大搜索频率 (默认fs/2)
        band_width: 频带宽度
        f_prior: 先验频率 (如啮合频率)，用于加权
        prior_weight: 先验权重 0~1
    
    返回:
        best_freq: 最优频带中心频率
        best_sk: 最大谱峭度值
        freqs: 频率轴
        sk_values: 各频带的谱峭度
    """
    if f_max is None:
        f_max = fs / 2
    
    f, sk = spectral_kurtosis(x, fs)
    
    center_freqs = np.arange(f_min + band_width//2, f_max - band_width//2, 50)
    sk_band = []
    
    for fc in center_freqs:
        idx = np.where((f >= fc - band_width//2) & (f <= fc + band_width//2))[0]
        if len(idx) > 0:
            sk_mean = np.mean(sk[idx])
            sk_band.append(sk_mean)
        else:
            sk_band.append(0.0)
    
    sk_band = np.array(sk_band)
    
    if f_prior is not None and prior_weight > 0:
        sk_norm = (sk_band - np.min(sk_band)) / (np.max(sk_band) - np.min(sk_band) + 1e-10)
        prior_scores = np.exp(-((center_freqs - f_prior) ** 2) / (2 * (band_width * 2) ** 2))
        combined_scores = (1 - prior_weight) * sk_norm + prior_weight * prior_scores
        best_idx = np.argmax(combined_scores)
    else:
        best_idx = np.argmax(sk_band)
    
    best_freq = center_freqs[best_idx]
    best_sk = sk_band[best_idx]
    
    return best_freq, best_sk, center_freqs, sk_band


def bandpass_filter(x, fs, low_freq, high_freq, order=4):
    """
    带通滤波器
    
    参数:
        x: 输入信号
        fs: 采样频率
        low_freq: 低截止频率
        high_freq: 高截止频率
        order: 滤波器阶数
    
    返回:
        filtered: 滤波后信号
    """
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, x)
    
    return filtered


def optimal_demodulation(x, fs, f_mesh=None, f_shaft=None, method='combine', band_width=500):
    """
    最优解调分析 - 结合谱峭度或MED选择最佳频带
    
    参数:
        x: 原始信号
        fs: 采样频率
        f_mesh: 啮合频率 (可选，用于指导频带选择)
        f_shaft: 转频 (可选)
        method: 'sk'=谱峭度, 'med'=MED, 'combine'=两者结合
        band_width: 频带宽度
    
    返回:
        filtered: 最优频带滤波后信号
        envelope: 包络信号
        freqs_env: 包络谱频率
        spec_env: 包络谱
        opt_info: 最优频带信息字典
    """
    opt_info = {
        'method': method,
        'center_freq': None,
        'band_width': band_width,
        'quality_metric': None
    }
    
    x_processed = x.copy()
    
    if method in ['med', 'combine']:
        x_med, _ = med_filter(x, L=31, max_iter=10)
        x_processed = 0.9 * x + 0.1 * x_med
    
    if f_mesh is not None:
        center_freq = f_mesh
        f_min_sk = max(100, f_mesh - 600)
        f_max_sk = min(fs/3, f_mesh + 600)
        _, best_sk, _, _ = find_optimal_band_sk(
            x, fs, f_min=f_min_sk, f_max=f_max_sk,
            band_width=200
        )
        opt_info['quality_metric'] = best_sk
        opt_info['center_freq'] = center_freq
    elif method in ['sk', 'combine']:
        f_min = 100
        f_max = fs / 3
        best_freq, best_sk, _, _ = find_optimal_band_sk(
            x_processed, fs, f_min=f_min, f_max=f_max, 
            band_width=band_width
        )
        center_freq = best_freq
        opt_info['quality_metric'] = best_sk
        opt_info['center_freq'] = center_freq
    else:
        center_freq = fs / 4
        opt_info['center_freq'] = center_freq
    
    low_freq = max(10, center_freq - band_width // 2)
    high_freq = min(fs / 2 - 10, center_freq + band_width // 2)
    
    filtered = bandpass_filter(x, fs, low_freq, high_freq)
    
    envelope, _ = hilbert_envelope(filtered)
    freqs_env, spec_env = envelope_spectrum(envelope, fs)
    
    return filtered, envelope, freqs_env, spec_env, opt_info


def estimate_rotational_frequency(x, fs, f_range=(10, 100), method='envelope'):
    """
    无转速计转频估计
    
    参数:
        x: 振动信号
        fs: 采样频率
        f_range: 转频搜索范围 (Hz)
        method: 'envelope'=包络谱法, 'kurtogram'=峭度法, 'auto'=自动选择
    
    返回:
        f_est: 估计的转频
        quality: 估计质量指标
    """
    if method == 'auto':
        f1, q1 = estimate_rotational_frequency(x, fs, f_range, method='envelope')
        f2, q2 = estimate_rotational_frequency(x, fs, f_range, method='kurtogram')
        if q1 > q2 * 0.5:
            return f1, q1
        else:
            return f2, q2
    
    if method == 'envelope':
        envelope, _ = hilbert_envelope(x)
        envelope_detrended = envelope - np.mean(envelope)
        
        N = len(envelope_detrended)
        window = np.hanning(N)
        envelope_windowed = envelope_detrended * window
        
        Y = fft(envelope_windowed)
        freqs = fftfreq(N, 1/fs)[:N//2]
        spectrum = 2.0 * np.abs(Y[:N//2]) / N
        
        idx_range = np.where((freqs >= f_range[0]) & (freqs <= f_range[1]))[0]
        if len(idx_range) > 0:
            sorted_idx = idx_range[np.argsort(spectrum[idx_range])[::-1]]
            
            candidates = []
            for idx in sorted_idx[:5]:
                f_cand = freqs[idx]
                quality_cand = spectrum[idx] / (np.mean(spectrum[idx_range]) + 1e-10)
                candidates.append((f_cand, quality_cand))
            
            f_candidates = [c[0] for c in candidates]
            if len(f_candidates) >= 2:
                harmonics_found = False
                for i, f1 in enumerate(f_candidates):
                    for j, f2 in enumerate(f_candidates):
                        if i != j and abs(f2 / f1 - 2.0) < 0.15:
                            f_est = f1 if f1 < f2 else f2
                            quality = max(candidates[i][1], candidates[j][1])
                            harmonics_found = True
                            break
                    if harmonics_found:
                        break
                if not harmonics_found:
                    f_est = f_candidates[0]
                    quality = candidates[0][1]
            else:
                f_est = f_candidates[0]
                quality = candidates[0][1]
        else:
            f_est = np.mean(f_range)
            quality = 0.0
            
        return f_est, quality
    
    elif method == 'kurtogram':
        n_levels = 6
        best_kurt = -1
        best_freq = np.mean(f_range)
        
        for level in range(n_levels):
            bw = fs / (2 ** (level + 2))
            n_bands = int(fs / (2 * bw))
            
            for i in range(n_bands):
                low = i * bw
                high = (i + 1) * bw
                if low < f_range[0] or high > fs/2:
                    continue
                
                if low <= 0 or high >= fs/2:
                    continue
                
                try:
                    b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
                    filtered = signal.filtfilt(b, a, x)
                    
                    env, _ = hilbert_envelope(filtered)
                    env_detrended = env - np.mean(env)
                    env_std = np.std(env_detrended)
                    kurt = np.mean(((env_detrended) / (env_std + 1e-10)) ** 4)
                    
                    N_env = len(env_detrended)
                    Y_env = fft(env_detrended * np.hanning(N_env))
                    freqs_env = fftfreq(N_env, 1/fs)[:N_env//2]
                    spec_env = 2.0 * np.abs(Y_env[:N_env//2]) / N_env
                    
                    idx_range = np.where((freqs_env >= f_range[0]) & (freqs_env <= f_range[1]))[0]
                    if len(idx_range) > 0:
                        peak_idx = idx_range[np.argmax(spec_env[idx_range])]
                        f_candidate = freqs_env[peak_idx]
                        
                        if kurt > best_kurt:
                            best_kurt = kurt
                            best_freq = f_candidate
                except (ValueError, np.linalg.LinAlgError):
                    continue
        
        return best_freq, best_kurt
    
    return np.mean(f_range), 0.0


def compute_phase_function(x, fs, f_rot, n_orders=1):
    """
    计算相位函数用于阶次重采样
    
    参数:
        x: 振动信号
        fs: 采样频率
        f_rot: 转频估计
        n_orders: 阶次倍数
    
    返回:
        phase: 相位函数 (0~2π)
        t_phase: 相位时间轴
    """
    N = len(x)
    t = np.linspace(0, N/fs, N, endpoint=False)
    
    freqs, t_stft, Zxx = signal.stft(x, fs=fs, nperseg=1024, noverlap=512)
    order_freq = f_rot * n_orders
    
    idx_order = np.argmin(np.abs(freqs - order_freq))
    phase_estimate = np.unwrap(np.angle(Zxx[idx_order, :]))
    
    f_phase = interp1d(t_stft, phase_estimate, kind='linear', 
                       fill_value='extrapolate')
    phase = f_phase(t)
    
    phase = 2 * np.pi * f_rot * t + (phase - phase[0])
    
    return phase, t


def order_tracking(x, fs, f_rot, orders_per_rev=360):
    """
    无转速计阶次跟踪 - 等角度重采样
    
    参数:
        x: 原始振动信号
        fs: 采样频率
        f_rot: 估计的转频
        orders_per_rev: 每转采样点数
    
    返回:
        x_order: 阶次域信号
        angle_axis: 角度轴
        fs_order: 阶次域采样率 (samples/rev)
    """
    N = len(x)
    t = np.linspace(0, N/fs, N, endpoint=False)
    
    phase = 2 * np.pi * f_rot * t
    
    n_revs = (phase[-1] - phase[0]) / (2 * np.pi)
    n_samples_order = int(n_revs * orders_per_rev)
    
    target_phase = np.linspace(phase[0], phase[-1], n_samples_order, endpoint=True)
    
    f_interp = interp1d(phase, x, kind='cubic', fill_value='extrapolate')
    x_order = f_interp(target_phase)
    
    angle_axis = np.linspace(0, n_revs * 360, n_samples_order, endpoint=True)
    
    return x_order, angle_axis, orders_per_rev


def order_spectrum(x_order, orders_per_rev):
    """
    计算阶次谱
    
    参数:
        x_order: 阶次域信号
        orders_per_rev: 每转采样点数
    
    返回:
        orders: 阶次轴
        spectrum: 阶次谱幅值
    """
    N = len(x_order)
    x_detrended = x_order - np.mean(x_order)
    
    window = np.hanning(N)
    x_windowed = x_detrended * window
    
    Y = fft(x_windowed)
    orders = (np.arange(N) * orders_per_rev / N)[:N//2]
    spectrum = 2.0 * np.abs(Y[:N//2]) / N
    
    return orders, spectrum


def extract_order_features(x_order, orders_per_rev, target_orders=[1, 2, 3, 25, 50, 75]):
    """
    提取阶次特征
    
    参数:
        x_order: 阶次域信号
        orders_per_rev: 每转采样点数
        target_orders: 目标阶次列表
    
    返回:
        features: 特征字典
    """
    orders, spec = order_spectrum(x_order, orders_per_rev)
    
    features = {}
    for ord_target in target_orders:
        idx = np.argmin(np.abs(orders - ord_target))
        features[f'order_{ord_target}'] = spec[idx]
    
    idx_1x = np.argmin(np.abs(orders - 1))
    amp_1x = spec[idx_1x]
    
    for ord_target in target_orders[1:]:
        idx = np.argmin(np.abs(orders - ord_target))
        features[f'order_ratio_{ord_target}_to_1'] = spec[idx] / (amp_1x + 1e-10)
    
    x_std = np.std(x_order - np.mean(x_order))
    if x_std > 1e-8:
        features['kurtosis_order'] = np.mean(((x_order - np.mean(x_order)) / x_std) ** 4)
    else:
        features['kurtosis_order'] = 0.0
    
    features['rms_order'] = np.sqrt(np.mean(x_order ** 2))
    features['peak_order'] = np.max(np.abs(x_order))
    
    return features


def build_gearbox_cnn(input_length=1024, n_classes=3, n_channels=1):
    """
    构建齿轮箱故障分类CNN模型
    
    参数:
        input_length: 输入信号长度
        n_classes: 类别数
        n_channels: 输入通道数
    
    返回:
        model: CNN模型
    """
    if not TORCH_AVAILABLE:
        return None
    
    try:
        class GearboxCNN(nn.Module):
            def __init__(self, input_length, n_classes, n_channels):
                super(GearboxCNN, self).__init__()
                
                self.features = nn.Sequential(
                    nn.Conv1d(n_channels, 16, kernel_size=64, stride=2, padding=31),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    
                    nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    
                    nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    
                    nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                )
                
                with torch.no_grad():
                    dummy = torch.randn(1, n_channels, input_length)
                    out = self.features(dummy)
                    n_features = out.size(1) * out.size(2)
                
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(n_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_classes)
                )
                
                self.attention_weights = nn.Sequential(
                    nn.Conv1d(128, 1, kernel_size=1),
                    nn.Softmax(dim=2)
                )
            
            def forward(self, x):
                features = self.features(x)
                attn = self.attention_weights(features)
                weighted = features * attn
                pooled = weighted.mean(dim=2) + features.max(dim=2)[0]
                
                out = self.classifier[0](features.flatten(1))
                out = self.classifier[1](out)
                out = self.classifier[2](out)
                out = self.classifier[3](out)
                out = self.classifier[4](out)
                out = self.classifier[5](out)
                out = self.classifier[6](out)
                
                return out
        
        return GearboxCNN(input_length, n_classes, n_channels)
    except Exception as e:
        print(f"CNN模型构建失败: {e}")
        return None


if TORCH_AVAILABLE:
    class GearboxDataset(Dataset):
        """齿轮箱振动数据集"""
        def __init__(self, signals, labels, transform=None):
            self.signals = signals
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.signals)
        
        def __getitem__(self, idx):
            signal = self.signals[idx].reshape(1, -1)
            label = self.labels[idx]
            
            if self.transform:
                signal = self.transform(signal)
            
            return torch.FloatTensor(signal), torch.LongTensor([label])[0]
else:
    GearboxDataset = None


def augment_signal(x, fs, augment_type='noise', **kwargs):
    """
    信号数据增强
    
    参数:
        x: 原始信号
        fs: 采样频率
        augment_type: 增强类型
    
    返回:
        x_aug: 增强后的信号
    """
    x_aug = x.copy()
    
    if augment_type == 'noise':
        snr = kwargs.get('snr', np.random.uniform(0, 10))
        noise = np.random.randn(len(x))
        signal_power = np.sum(x ** 2) / len(x)
        noise_power = signal_power / (10 ** (snr / 10))
        x_aug = x + np.sqrt(noise_power) * noise
    
    elif augment_type == 'shift':
        shift = kwargs.get('shift', np.random.randint(-int(fs*0.1), int(fs*0.1)))
        x_aug = np.roll(x, shift)
    
    elif augment_type == 'scale':
        scale = kwargs.get('scale', np.random.uniform(0.8, 1.2))
        x_aug = x * scale
    
    elif augment_type == 'time_warp':
        factor = kwargs.get('factor', np.random.uniform(0.9, 1.1))
        t = np.arange(len(x))
        t_warped = np.linspace(0, len(x)-1, int(len(x) * factor))
        f_interp = interp1d(t, x, kind='linear', fill_value='extrapolate')
        x_aug = f_interp(t_warped)
        if len(x_aug) > len(x):
            x_aug = x_aug[:len(x)]
        else:
            x_aug = np.pad(x_aug, (0, len(x) - len(x_aug)), mode='edge')
    
    return x_aug


def generate_gearbox_dataset(n_samples_per_class=50, fs=10000, duration=1.0, 
                             augment=True, speed_variation=True):
    """
    生成齿轮箱故障数据集
    
    参数:
        n_samples_per_class: 每类样本数
        fs: 采样频率
        duration: 信号时长
        augment: 是否数据增强
        speed_variation: 是否转速波动
    
    返回:
        X_train: 训练集信号
        y_train: 训练集标签
        X_test: 测试集信号
        y_test: 测试集标签
        class_names: 类别名称
    """
    class_names = ['正常', '齿面剥落', '断齿']
    fault_types = [None, 'wear', 'broken']
    
    X = []
    y = []
    
    for label, fault_type in enumerate(fault_types):
        for i in range(n_samples_per_class):
            if speed_variation:
                speed_factor = 0.8 + np.random.rand() * 0.4
            else:
                speed_factor = 1.0
            
            t, x, params = generate_gearbox_signal(
                fs=fs, duration=duration, 
                fault_type=fault_type, seed=i+label*1000
            )
            
            if speed_variation:
                speed_profile = speed_factor + 0.1 * np.sin(2 * np.pi * 0.5 * t)
                x = x * (1 + 0.05 * np.sin(2 * np.pi * params['fr'] * speed_profile * t))
            
            if augment and i % 2 == 0:
                aug_types = ['noise', 'scale', 'shift']
                aug_type = np.random.choice(aug_types)
                x = augment_signal(x, fs, augment_type=aug_type)
            
            X.append(x.astype(np.float32))
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test, class_names


def train_cnn_model(model, train_loader, val_loader, n_epochs=30, lr=0.001, device='cpu'):
    """
    训练CNN模型
    
    参数:
        model: CNN模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        n_epochs: 训练轮数
        lr: 学习率
        device: 设备
    
    返回:
        model: 训练后的模型
        train_history: 训练历史
    """
    if not TORCH_AVAILABLE or model is None:
        return None, None
    
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        model = model.to(device)
        train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for signals, labels in train_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * signals.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals = signals.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * signals.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            train_history['train_loss'].append(train_loss)
            train_history['train_acc'].append(train_acc)
            train_history['val_loss'].append(val_loss)
            train_history['val_acc'].append(val_acc)
            
            print(f'Epoch [{epoch+1}/{n_epochs}] TrainLoss: {train_loss:.4f} TrainAcc: {train_acc:.2%} '
                  f'ValLoss: {val_loss:.4f} ValAcc: {val_acc:.2%}')
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, train_history
    except Exception as e:
        print(f"CNN训练失败: {e}")
        return None, None


def predict_cnn(model, x, device='cpu'):
    """
    使用CNN模型进行预测
    
    参数:
        model: CNN模型
        x: 输入信号 (n_samples, length) 或 (length,)
        device: 设备
    
    返回:
        predictions: 预测类别
        probabilities: 概率分布
    """
    if not TORCH_AVAILABLE or model is None:
        return None, None
    
    try:
        model.eval()
        
        if x.ndim == 1:
            x = x.reshape(1, 1, -1)
        elif x.ndim == 2:
            x = x.reshape(x.shape[0], 1, x.shape[1])
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(device)
            outputs = model(x_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    except Exception as e:
        print(f"CNN预测失败: {e}")
        return None, None


def extract_traditional_features(x, fs, f_mesh, f_shaft):
    """
    提取传统故障诊断特征用于机器学习
    
    参数:
        x: 振动信号
        fs: 采样频率
        f_mesh: 啮合频率
        f_shaft: 转频
    
    返回:
        features: 特征数组
    """
    filtered, envelope, freqs_env, spec_env, _ = optimal_demodulation(
        x, fs, f_mesh=f_mesh, f_shaft=f_shaft, method='combine'
    )
    
    freqs_raw, spec_raw = envelope_spectrum(x, fs)
    
    def get_peak(f, s, target, tol=10):
        idx = np.where((f >= target - tol) & (f <= target + tol))[0]
        return np.max(s[idx]) if len(idx) > 0 else 0.0
    
    features = []
    
    features.append(get_peak(freqs_env, spec_env, f_shaft))
    features.append(get_peak(freqs_env, spec_env, 2*f_shaft))
    features.append(get_peak(freqs_env, spec_env, 3*f_shaft))
    
    amp_shaft = get_peak(freqs_env, spec_env, f_shaft)
    features.append(get_peak(freqs_env, spec_env, 2*f_shaft) / (amp_shaft + 1e-10))
    features.append(get_peak(freqs_env, spec_env, 3*f_shaft) / (amp_shaft + 1e-10))
    
    amp_mesh = get_peak(freqs_raw, spec_raw, f_mesh)
    features.append(amp_mesh)
    features.append(get_peak(freqs_raw, spec_raw, 2*f_mesh) / (amp_mesh + 1e-10))
    
    env_detrended = envelope - np.mean(envelope)
    env_std = np.std(env_detrended)
    if env_std > 1e-8:
        features.append(np.mean(((env_detrended) / env_std) ** 4))
    else:
        features.append(0.0)
    
    features.append(np.sqrt(np.mean(x ** 2)))
    features.append(np.max(np.abs(x)))
    features.append(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-10))
    
    sideband_sum = 0
    for n in range(-3, 4):
        if n != 0:
            f_sb = f_mesh + n * f_shaft
            sideband_sum += get_peak(freqs_raw, spec_raw, f_sb)
    features.append(sideband_sum / (amp_mesh * 6 + 1e-10))
    
    return np.array(features)


def train_ml_classifier(X_train_features, y_train):
    """
    训练机器学习分类器 (随机森林)
    
    参数:
        X_train_features: 训练特征
        y_train: 训练标签
    
    返回:
        classifier: 训练好的分类器
    """
    if not SKLEARN_AVAILABLE:
        return None
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_features, y_train)
    
    return clf


def combined_diagnosis(x, fs, f_mesh, f_shaft, ml_model=None, class_names=None, use_ml='auto'):
    """
    组合诊断：传统特征 + 机器学习
    
    参数:
        x: 振动信号
        fs: 采样频率
        f_mesh: 啮合频率
        f_shaft: 转频
        ml_model: 机器学习模型
        class_names: 类别名称
        use_ml: 'auto', 'cnn', 'rf', 'none'
    
    返回:
        diagnosis: 诊断结果字典
    """
    filtered, envelope, freqs_env, spec_env, opt_info = optimal_demodulation(
        x, fs, f_mesh=f_mesh, f_shaft=f_shaft, method='combine'
    )
    
    freqs_raw, spec_raw = envelope_spectrum(x, fs)
    
    diag_traditional = fault_diagnosis(
        freqs_env, spec_env, f_mesh, f_shaft, envelope=envelope,
        f_mesh_2nd=2*f_mesh, raw_spectrum=spec_raw
    )
    
    diag_ml = None
    
    if use_ml in ['auto', 'cnn'] and TORCH_AVAILABLE and ml_model is not None and hasattr(ml_model, 'forward'):
        x_normalized = (x - np.mean(x)) / (np.std(x) + 1e-10)
        x_input = x_normalized[:1024] if len(x) > 1024 else np.pad(x, (0, 1024 - len(x)))
        preds, probs = predict_cnn(ml_model, x_input)
        
        if class_names is not None and preds is not None:
            diag_ml = {
                'type': 'CNN',
                'prediction': class_names[preds[0]],
                'probabilities': probs[0],
                'confidence': np.max(probs[0])
            }
    
    elif use_ml in ['auto', 'rf'] and SKLEARN_AVAILABLE and ml_model is not None and hasattr(ml_model, 'predict'):
        features = extract_traditional_features(x, fs, f_mesh, f_shaft).reshape(1, -1)
        pred = ml_model.predict(features)[0]
        proba = ml_model.predict_proba(features)[0]
        
        if class_names is not None:
            diag_ml = {
                'type': 'RandomForest',
                'prediction': class_names[pred],
                'probabilities': proba,
                'confidence': np.max(proba)
            }
    
    diagnosis = {
        'traditional': diag_traditional,
        'ml': diag_ml,
        'optimal_band': opt_info
    }
    
    if diag_ml is not None:
        class_map = {'正常': 0, '齿面剥落': 1, '断齿': 2}
        trad_class = class_map.get(diag_traditional['status'].split(' ')[0], -1)
        ml_class = class_map.get(diag_ml['prediction'], -1)
        
        if trad_class == ml_class and trad_class >= 0:
            diagnosis['final'] = diag_ml['prediction']
            diagnosis['final_confidence'] = max(diag_traditional['confidence'], diag_ml['confidence'])
        elif diag_ml['confidence'] > diag_traditional['confidence']:
            diagnosis['final'] = diag_ml['prediction']
            diagnosis['final_confidence'] = diag_ml['confidence']
        else:
            diagnosis['final'] = diag_traditional['status']
            diagnosis['final_confidence'] = diag_traditional['confidence']
    else:
        diagnosis['final'] = diag_traditional['status']
        diagnosis['final_confidence'] = diag_traditional['confidence']
    
    return diagnosis


def envelope_spectrum(envelope, fs):
    """
    计算包络谱
    
    参数:
        envelope: 包络信号
        fs: 采样频率
    
    返回:
        freqs: 频率轴
        spectrum: 包络谱幅值
    """
    N = len(envelope)
    envelope_detrended = envelope - np.mean(envelope)
    
    window = np.hanning(N)
    envelope_windowed = envelope_detrended * window
    
    Y = fft(envelope_windowed)
    freqs = fftfreq(N, 1/fs)[:N//2]
    spectrum = 2.0 * np.abs(Y[:N//2]) / N
    
    return freqs, spectrum


def find_sidebands(freqs, spectrum, f_mesh, f_shaft, threshold=0.3):
    """
    提取调制边带特征
    
    参数:
        freqs: 频率轴
        spectrum: 频谱
        f_mesh: 啮合频率
        f_shaft: 转频
        threshold: 幅值阈值 (相对于最大幅值的比例)
    
    返回:
        sidebands: 边带信息列表
    """
    sidebands = []
    max_amp = np.max(spectrum)
    
    for n in range(-5, 6):
        if n == 0:
            continue
        f_side = f_mesh + n * f_shaft
        idx = np.argmin(np.abs(freqs - f_side))
        amp = spectrum[idx]
        
        if amp > threshold * max_amp:
            sidebands.append({
                'order': n,
                'frequency': freqs[idx],
                'amplitude': amp,
                'theoretical_freq': f_side
            })
    
    return sidebands


def demodulation_analysis(x, fs, f_mesh, f_shaft, band_order=3):
    """
    带通滤波后包络解调分析
    
    参数:
        x: 原始信号
        fs: 采样频率
        f_mesh: 啮合频率
        f_shaft: 转频
        band_order: 边带阶数
    
    返回:
        filtered: 滤波后信号
        envelope: 包络信号
        freqs_env: 包络谱频率
        spec_env: 包络谱
    """
    low_freq = f_mesh - band_order * f_shaft - 20
    high_freq = f_mesh + band_order * f_shaft + 20
    
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, x)
    
    envelope, _ = hilbert_envelope(filtered)
    freqs_env, spec_env = envelope_spectrum(envelope, fs)
    
    return filtered, envelope, freqs_env, spec_env


def fault_diagnosis(freqs, spectrum, f_mesh, f_shaft, envelope=None, 
                    f_mesh_2nd=None, raw_spectrum=None):
    """
    故障诊断（基于谱特征比值，对MED/谱峭度后信号更鲁棒）
    
    参数:
        freqs: 包络谱频率轴
        spectrum: 包络谱
        f_mesh: 啮合频率
        f_shaft: 转频
        envelope: 包络信号 (用于计算统计特征)
        f_mesh_2nd: 2倍啮合频率
        raw_spectrum: 原始频谱 (可选，用于辅助判断)
    
    返回:
        diagnosis: 诊断结果字典
    """
    diagnosis = {
        'status': '未知',
        'features': {},
        'confidence': 0.0
    }
    
    def get_peak_amp(f_target, tol=10):
        idx = np.where((freqs >= f_target - tol) & (freqs <= f_target + tol))[0]
        if len(idx) > 0:
            return np.max(spectrum[idx])
        return 0.0
    
    def get_raw_peak_amp(f_target, tol=10):
        if raw_spectrum is None:
            return 0.0
        idx = np.where((freqs >= f_target - tol) & (freqs <= f_target + tol))[0]
        if len(idx) > 0:
            return np.max(raw_spectrum[idx])
        return 0.0
    
    amp_shaft = get_peak_amp(f_shaft)
    amp_shaft_2x = get_peak_amp(2 * f_shaft)
    amp_shaft_3x = get_peak_amp(3 * f_shaft)
    amp_mesh_env = get_peak_amp(f_mesh)
    
    amp_mesh_raw = get_raw_peak_amp(f_mesh)
    amp_mesh_2nd_raw = get_raw_peak_amp(2 * f_mesh) if f_mesh_2nd else 0.0
    
    low_freq = max(10, f_shaft * 20)
    high_freq = f_mesh * 0.8
    bg_idx = np.where((freqs >= low_freq) & (freqs <= high_freq) & 
                      (np.abs(np.mod(freqs, f_shaft)) > 5) &
                      (np.abs(freqs - f_mesh) > f_shaft * 2))[0]
    if len(bg_idx) > 5 and np.mean(spectrum[bg_idx]) > 1e-10:
        bg_level = np.median(spectrum[bg_idx])
    else:
        bg_idx2 = np.where(freqs >= f_shaft * 10)[0]
        if len(bg_idx2) > 0:
            bg_level = np.median(spectrum[bg_idx2])
        else:
            bg_level = np.median(spectrum)
    if bg_level < 1e-6:
        bg_level = 1e-6
    
    max_amp = np.max(spectrum)
    
    shaft_snr_1 = amp_shaft / (bg_level + 1e-10)
    shaft_snr_2 = amp_shaft_2x / (bg_level + 1e-10)
    shaft_snr_3 = amp_shaft_3x / (bg_level + 1e-10)
    
    shaft_rel_2 = amp_shaft_2x / (amp_shaft + 1e-10)
    shaft_rel_3 = amp_shaft_3x / (amp_shaft + 1e-10)
    
    sideband_strength_raw = 0
    if raw_spectrum is not None and amp_mesh_raw > 1e-10:
        sideband_sum = 0
        for n in range(-3, 4):
            if n != 0:
                f_sb = f_mesh + n * f_shaft
                idx = np.argmin(np.abs(freqs - f_sb))
                if idx < len(raw_spectrum):
                    sideband_sum += raw_spectrum[idx]
        sideband_strength_raw = sideband_sum / (amp_mesh_raw * 6 + 1e-10)
    
    mesh_harmonic_ratio = 0
    if amp_mesh_raw > 1e-10:
        mesh_harmonic_ratio = amp_mesh_2nd_raw / (amp_mesh_raw + 1e-10)
    
    kurtosis = 0.0
    peak_factor = 0.0
    impulse_factor = 0.0
    if envelope is not None:
        env_detrended = envelope - np.mean(envelope)
        env_std = np.std(env_detrended)
        env_mean = np.mean(np.abs(env_detrended))
        if env_std > 1e-8:
            kurtosis = np.mean(((env_detrended) / env_std) ** 4)
            peak_factor = np.max(np.abs(env_detrended)) / (env_std + 1e-10)
        if env_mean > 1e-8:
            impulse_factor = np.max(np.abs(env_detrended)) / (env_mean + 1e-10)
    
    diagnosis['features'] = {
        '转频幅值': amp_shaft,
        '2倍转频幅值': amp_shaft_2x,
        '3倍转频幅值': amp_shaft_3x,
        '啮合频率幅值(包络谱)': amp_mesh_env,
        '啮合频率幅值(原始谱)': amp_mesh_raw,
        '2倍啮合频率幅值(原始谱)': amp_mesh_2nd_raw,
        '背景水平': bg_level,
        '最大幅值': max_amp,
        '转频信噪比': shaft_snr_1,
        '2倍转频信噪比': shaft_snr_2,
        '3倍转频信噪比': shaft_snr_3,
        '2倍转频/转频': shaft_rel_2,
        '3倍转频/转频': shaft_rel_3,
        '边带相对强度': sideband_strength_raw,
        '2倍啮合/1倍啮合': mesh_harmonic_ratio,
        '峭度': kurtosis,
        '峰值因子': peak_factor,
        '脉冲因子': impulse_factor
    }
    
    shaft_snr_threshold = 20.0
    shaft_amp_threshold = 0.02
    
    if (shaft_snr_1 < shaft_snr_threshold or amp_shaft < shaft_amp_threshold) and mesh_harmonic_ratio < 0.45:
        diagnosis['status'] = '正常'
        diagnosis['confidence'] = 0.9
    elif shaft_snr_1 >= shaft_snr_threshold and amp_shaft >= shaft_amp_threshold:
        if shaft_rel_2 >= 0.3 and shaft_rel_3 >= 0.12 and mesh_harmonic_ratio < 0.2:
            diagnosis['status'] = '断齿故障'
            diagnosis['confidence'] = 0.9
        elif mesh_harmonic_ratio >= 0.45 or sideband_strength_raw >= 0.15:
            diagnosis['status'] = '齿面剥落故障'
            diagnosis['confidence'] = 0.85
        elif shaft_rel_2 >= 0.2 and mesh_harmonic_ratio < 0.25:
            diagnosis['status'] = '断齿故障'
            diagnosis['confidence'] = 0.8
        elif amp_shaft >= shaft_amp_threshold * 10:
            diagnosis['status'] = '齿面剥落故障'
            diagnosis['confidence'] = 0.75
        else:
            diagnosis['status'] = '齿面剥落故障 (早期)'
            diagnosis['confidence'] = 0.7
    else:
        diagnosis['status'] = '正常 (注意观察)'
        diagnosis['confidence'] = 0.7
    
    return diagnosis


def plot_analysis_results(t, x, params, filtered, envelope,
                         freqs_raw, spec_raw, freqs_env, spec_env,
                         sidebands, diagnosis):
    """
    可视化分析结果
    """
    f_mesh = params['f_mesh']
    f_shaft = params['f_shaft']
    fs = params['fs']
    fault_type = params['fault_type']
    
    title_suffix = ''
    if fault_type == 'wear':
        title_suffix = ' (齿面剥落)'
    elif fault_type == 'broken':
        title_suffix = ' (断齿)'
    else:
        title_suffix = ' (正常)'
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t[:2000], x[:2000], 'b-', linewidth=0.8)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.set_title(f'原始振动信号 (0~0.2s){title_suffix}')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t[:2000], filtered[:2000], 'g-', linewidth=0.8, label='滤波后信号')
    ax2.plot(t[:2000], envelope[:2000], 'r-', linewidth=1.5, label='包络信号')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('幅值')
    ax2.set_title('带通滤波与包络信号')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, envelope, 'r-', linewidth=0.6)
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('包络幅值')
    ax3.set_title('完整包络信号')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(freqs_raw, spec_raw, 'b-', linewidth=0.8)
    ax4.axvline(f_mesh, color='r', linestyle='--', label=f'啮合频率 {f_mesh:.1f}Hz', alpha=0.7)
    for n in range(-3, 4):
        if n != 0:
            f_sb = f_mesh + n * f_shaft
            ax4.axvline(f_sb, color='orange', linestyle=':', alpha=0.5)
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('幅值')
    ax4.set_title('原始信号频谱')
    ax4.set_xlim(0, fs/8)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(freqs_env, spec_env, 'g-', linewidth=0.8)
    ax5.axvline(f_shaft, color='r', linestyle='--', label=f'转频 {f_shaft:.1f}Hz', alpha=0.7)
    ax5.axvline(2*f_shaft, color='purple', linestyle='--', label=f'2倍转频 {2*f_shaft:.1f}Hz', alpha=0.6)
    ax5.axvline(3*f_shaft, color='orange', linestyle='--', label=f'3倍转频 {3*f_shaft:.1f}Hz', alpha=0.5)
    ax5.set_xlabel('频率 (Hz)')
    ax5.set_ylabel('幅值')
    ax5.set_title('包络谱 (解调后)')
    ax5.set_xlim(0, f_mesh)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[3, :])
    if sidebands:
        sideband_freqs = [sb['frequency'] for sb in sidebands]
        sideband_amps = [sb['amplitude'] for sb in sidebands]
        sideband_orders = [sb['order'] for sb in sidebands]
        
        markerline, stemlines, baseline = ax6.stem(sideband_freqs, sideband_amps, linefmt='b-', markerfmt='bo', basefmt='r-')
        plt.setp(stemlines, 'linewidth', 0.8)
        for freq, amp, order in zip(sideband_freqs, sideband_amps, sideband_orders):
            ax6.text(freq, amp, f'{order}', ha='center', va='bottom', fontsize=9)
        ax6.set_ylim(0, max(sideband_amps) * 1.3)
    else:
        ax6.text(0.5, 0.5, '无明显调制边带', ha='center', va='center', 
                 fontsize=14, transform=ax6.transAxes, color='gray')
        ax6.set_ylim(0, 1)
    
    ax6.axvline(f_mesh, color='r', linestyle='--', label=f'啮合频率 {f_mesh:.1f}Hz', alpha=0.7)
    ax6.set_xlabel('频率 (Hz)')
    ax6.set_ylabel('幅值')
    ax6.set_title(f'调制边带分析 (边带阶数标记) - 诊断结果: {diagnosis["status"]}')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'齿轮箱振动信号包络解调分析{title_suffix}\n'
                 f'轴转频: {f_shaft:.1f}Hz, 齿数: {params["n_teeth"]}, 啮合频率: {f_mesh:.1f}Hz',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    return fig


def plot_advanced_analysis(t, x, params, filtered_sk, envelope_sk, 
                            freqs_raw, spec_raw, freqs_env_sk, spec_env_sk,
                            freqs_sk_curve, sk_curve, best_freq, opt_info,
                            sidebands, diagnosis):
    """
    高级分析结果可视化（包含谱峭度和MED）
    """
    f_mesh = params['f_mesh']
    f_shaft = params['f_shaft']
    fs = params['fs']
    fault_type = params['fault_type']
    
    title_suffix = ''
    if fault_type == 'wear':
        title_suffix = ' (齿面剥落)'
    elif fault_type == 'broken':
        title_suffix = ' (断齿)'
    else:
        title_suffix = ' (正常)'
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t[:2000], x[:2000], 'b-', linewidth=0.8)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.set_title(f'原始振动信号 (0~0.2s){title_suffix}')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(freqs_sk_curve, sk_curve, 'm-', linewidth=1.0)
    ax2.axvline(best_freq, color='r', linestyle='--', 
                label=f'最优频带 {best_freq:.0f}Hz', alpha=0.8, linewidth=2)
    ax2.axvline(f_mesh, color='g', linestyle=':', 
                label=f'啮合频率 {f_mesh:.0f}Hz', alpha=0.7)
    ax2.set_xlabel('频率 (Hz)')
    ax2.set_ylabel('谱峭度')
    ax2.set_title(f'谱峭度曲线 (最大SK={opt_info["quality_metric"]:.2f})')
    ax2.set_xlim(0, fs/4)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t[:2000], filtered_sk[:2000], 'g-', linewidth=0.8, label='滤波后信号')
    ax3.plot(t[:2000], envelope_sk[:2000], 'r-', linewidth=1.5, label='包络信号')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('幅值')
    ax3.set_title(f'最优频带({best_freq:.0f}Hz)滤波与包络')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(freqs_raw, spec_raw, 'b-', linewidth=0.8)
    ax4.axvline(f_mesh, color='r', linestyle='--', label=f'啮合频率 {f_mesh:.1f}Hz', alpha=0.7)
    for n in range(-3, 4):
        if n != 0:
            f_sb = f_mesh + n * f_shaft
            ax4.axvline(f_sb, color='orange', linestyle=':', alpha=0.5)
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('幅值')
    ax4.set_title('原始信号频谱')
    ax4.set_xlim(0, fs/8)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(freqs_env_sk, spec_env_sk, 'g-', linewidth=0.8)
    ax5.axvline(f_shaft, color='r', linestyle='--', label=f'转频 {f_shaft:.1f}Hz', alpha=0.7)
    ax5.axvline(2*f_shaft, color='purple', linestyle='--', label=f'2倍转频 {2*f_shaft:.1f}Hz', alpha=0.6)
    ax5.axvline(3*f_shaft, color='orange', linestyle='--', label=f'3倍转频 {3*f_shaft:.1f}Hz', alpha=0.5)
    ax5.set_xlabel('频率 (Hz)')
    ax5.set_ylabel('幅值')
    ax5.set_title('最优频带包络谱')
    ax5.set_xlim(0, f_mesh)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[3, :])
    if sidebands:
        sideband_freqs = [sb['frequency'] for sb in sidebands]
        sideband_amps = [sb['amplitude'] for sb in sidebands]
        sideband_orders = [sb['order'] for sb in sidebands]
        
        markerline, stemlines, baseline = ax6.stem(sideband_freqs, sideband_amps, 
                                                   linefmt='b-', markerfmt='bo', basefmt='r-')
        plt.setp(stemlines, 'linewidth', 0.8)
        for freq, amp, order in zip(sideband_freqs, sideband_amps, sideband_orders):
            ax6.text(freq, amp, f'{order}', ha='center', va='bottom', fontsize=9)
        ax6.set_ylim(0, max(sideband_amps) * 1.3)
    else:
        ax6.text(0.5, 0.5, '无明显调制边带', ha='center', va='center', 
                 fontsize=14, transform=ax6.transAxes, color='gray')
        ax6.set_ylim(0, 1)
    
    ax6.axvline(f_mesh, color='r', linestyle='--', label=f'啮合频率 {f_mesh:.1f}Hz', alpha=0.7)
    ax6.set_xlabel('频率 (Hz)')
    ax6.set_ylabel('幅值')
    ax6.set_title(f'调制边带分析 - 诊断结果: {diagnosis["status"]} (置信度: {diagnosis["confidence"]:.1%})')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[4, :])
    feature_labels = ['转频幅值', '2倍转频/转频', '3倍转频/转频', '2倍啮合/1倍啮合']
    feature_values = [
        diagnosis['features']['转频幅值'],
        diagnosis['features']['2倍转频/转频'],
        diagnosis['features']['3倍转频/转频'],
        diagnosis['features']['2倍啮合/1倍啮合']
    ]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax7.bar(feature_labels, feature_values, color=colors, alpha=0.7)
    
    for bar, value in zip(bars, feature_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    threshold_lines = [0.05, 0.25, 0.12, 0.3]
    for i, thresh in enumerate(threshold_lines):
        ax7.axhline(thresh, color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
    
    ax7.set_ylabel('特征值')
    ax7.set_title('诊断特征参数 (虚线为阈值)')
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'齿轮箱振动信号高级分析{title_suffix}\n'
                 f'方法: MED+谱峭度最优解调 | 最优中心频率: {best_freq:.0f}Hz | '
                 f'轴转频: {f_shaft:.1f}Hz, 齿数: {params["n_teeth"]}, 啮合频率: {f_mesh:.1f}Hz',
                 fontsize=14, fontweight='bold', y=0.995)
    
    return fig


def plot_order_analysis(t, x, x_order, angle_axis, orders, spec_order, 
                       order_features, f_est, f_true, params):
    """
    阶次跟踪分析结果可视化
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t[:5000], x[:5000], 'b-', linewidth=0.6)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.set_title('原始时域信号')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(angle_axis[:2000], x_order[:2000], 'g-', linewidth=0.6)
    ax2.set_xlabel('角度 (°)')
    ax2.set_ylabel('幅值')
    ax2.set_title(f'阶次域信号 (估计转频: {f_est:.1f}Hz, 真实: {f_true:.1f}Hz)')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(orders, spec_order, 'r-', linewidth=0.8)
    ax3.axvline(1, color='b', linestyle='--', label='1阶 (转频)', alpha=0.7)
    ax3.axvline(params['n_teeth'], color='g', linestyle='--', 
                label=f'{params["n_teeth"]}阶 (啮合频率)', alpha=0.7)
    ax3.axvline(2*params['n_teeth'], color='purple', linestyle='--', 
                label=f'{2*params["n_teeth"]}阶 (2倍啮合)', alpha=0.6)
    ax3.set_xlabel('阶次')
    ax3.set_ylabel('幅值')
    ax3.set_title('阶次谱')
    ax3.set_xlim(0, 100)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, :])
    feature_names = list(order_features.keys())[:6]
    feature_values = [order_features[k] for k in feature_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
    bars = ax4.bar(feature_names, feature_values, color=colors, alpha=0.7)
    
    for bar, value in zip(bars, feature_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax4.set_ylabel('特征值')
    ax4.set_title('阶次域特征参数')
    ax4.tick_params(axis='x', rotation=15)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('无转速计阶次跟踪分析', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def plot_training_history(train_history):
    """
    CNN训练过程可视化
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_history['train_loss'], 'b-', label='训练损失', linewidth=2)
    ax1.plot(train_history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    ax2.plot(train_history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.set_title('训练准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    return fig


def main():
    print('=' * 80)
    print('齿轮箱振动信号智能诊断系统 (阶次跟踪 + CNN深度学习)')
    print('=' * 80)
    
    fs = 10000
    duration = 1.0
    device = 'cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
    
    print(f'\n运行环境: PyTorch可用={TORCH_AVAILABLE}, 设备={device}')
    
    print('\n' + '=' * 80)
    print('阶段一: 无转速计阶次跟踪分析')
    print('=' * 80)
    
    test_cases = [None, 'wear', 'broken']
    case_names = ['正常', '齿面剥落', '断齿']
    
    order_results = []
    
    for fault_type, case_name in zip(test_cases, case_names):
        print(f'\n测试案例: {case_name}')
        print('-' * 60)
        
        t, x, params = generate_gearbox_signal(fs=fs, duration=duration, 
                                               fault_type=fault_type, seed=42)
        
        print(f'[1/3] 无转速计转频估计...')
        f_est, quality = estimate_rotational_frequency(x, fs, f_range=(20, 50), method='auto')
        f_true = params['fr']
        print(f'  估计转频: {f_est:.2f}Hz, 真实转频: {f_true:.2f}Hz')
        print(f'  估计误差: {abs(f_est - f_true)/f_true*100:.2f}%, 质量指标: {quality:.2f}')
        
        print(f'[2/3] 等角度重采样 (阶次跟踪)...')
        x_order, angle_axis, orders_per_rev = order_tracking(x, fs, f_est, orders_per_rev=360)
        print(f'  阶次域采样率: {orders_per_rev} samples/rev')
        print(f'  阶次域信号长度: {len(x_order)} 采样点')
        
        print(f'[3/3] 阶次谱与特征提取...')
        orders, spec_order = order_spectrum(x_order, orders_per_rev)
        order_features = extract_order_features(x_order, orders_per_rev)
        print(f'  提取阶次特征数: {len(order_features)}')
        
        order_results.append({
            'case': case_name,
            'f_est': f_est,
            'f_true': f_true,
            'features': order_features,
            'correct': (
                (fault_type is None and order_features.get('order_ratio_25_to_1', 0) < 10) or
                (fault_type == 'wear' and order_features.get('order_ratio_50_to_25', 0) > 0.3) or
                (fault_type == 'broken' and order_features.get('kurtosis_order', 0) > 5)
            )
        })
        
        fig = plot_order_analysis(t, x, x_order, angle_axis, orders, spec_order,
                                  order_features, f_est, f_true, params)
        save_name = f'order_analysis_{fault_type if fault_type else "normal"}.png'
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f'  阶次分析图表已保存: {save_name}')
        plt.close(fig)
    
    print('\n阶次跟踪分析完成!')
    
    print('\n' + '=' * 80)
    print('阶段二: 机器学习故障分类')
    print('=' * 80)
    
    class_names = ['正常', '齿面剥落', '断齿']
    ml_model = None
    ml_type = None
    
    if TORCH_AVAILABLE:
        print(f'\n[1/5] 生成齿轮箱数据集...')
        n_samples = 40
        X_train, y_train, X_test, y_test, class_names = generate_gearbox_dataset(
            n_samples_per_class=n_samples, fs=fs, duration=1.0,
            augment=True, speed_variation=True
        )
        print(f'  训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本')
        print(f'  类别: {class_names}')
        print(f'  信号长度: {X_train.shape[1]} 采样点')
        
        print(f'\n[2/5] 构建CNN模型...')
        input_length = min(X_train.shape[1], 1024)
        model = build_gearbox_cnn(input_length=input_length, n_classes=3, n_channels=1)
        print(f'  输入长度: {input_length}')
        print(f'  模型参数数量: {sum(p.numel() for p in model.parameters()):,}')
        
        print(f'\n[3/5] 准备数据加载器...')
        X_train_cnn = X_train[:, :input_length]
        X_test_cnn = X_test[:, :input_length]
        
        X_train_mean = np.mean(X_train_cnn)
        X_train_std = np.std(X_train_cnn) + 1e-10
        X_train_cnn = (X_train_cnn - X_train_mean) / X_train_std
        X_test_cnn = (X_test_cnn - X_train_mean) / X_train_std
        
        train_dataset = GearboxDataset(X_train_cnn, y_train)
        test_dataset = GearboxDataset(X_test_cnn, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print(f'\n[4/5] 训练CNN模型...')
        model, train_history = train_cnn_model(
            model, train_loader, test_loader, n_epochs=20, lr=0.001, device=device
        )
        
        fig_train = plot_training_history(train_history)
        plt.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight')
        print(f'  训练曲线已保存: cnn_training_history.png')
        plt.close(fig_train)
        
        print(f'\n[5/5] 模型评估...')
        model.eval()
        test_preds = []
        test_probs = []
        
        with torch.no_grad():
            for signals, _ in test_loader:
                signals = signals.to(device)
                outputs = model(signals)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                test_preds.extend(preds)
                test_probs.extend(probs)
        
        test_preds = np.array(test_preds)
        accuracy = np.mean(test_preds == y_test)
        print(f'  测试集准确率: {accuracy:.2%}')
        
        print(f'\n分类混淆矩阵:')
        cm = confusion_matrix(y_test, test_preds)
        print('  ' + ''.join([f'{name:>12}' for name in class_names]))
        for i, row in enumerate(cm):
            print(f'  {class_names[i][:4]:>4}' + ''.join([f'{v:>12}' for v in row]))
        
        ml_model = model
        ml_type = 'cnn'
    
    elif SKLEARN_AVAILABLE:
        print('\nPyTorch不可用，使用随机森林分类器 (scikit-learn)')
        print(f'\n[1/4] 生成齿轮箱数据集...')
        n_samples = 40
        X_train, y_train, X_test, y_test, class_names = generate_gearbox_dataset(
            n_samples_per_class=n_samples, fs=fs, duration=1.0,
            augment=True, speed_variation=True
        )
        print(f'  训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本')
        print(f'  类别: {class_names}')
        
        print(f'\n[2/4] 提取传统特征...')
        fr_true = 30.0
        f_mesh_true = fr_true * 25
        
        X_train_features = np.array([
            extract_traditional_features(x, fs, f_mesh_true, fr_true) 
            for x in X_train
        ])
        X_test_features = np.array([
            extract_traditional_features(x, fs, f_mesh_true, fr_true)
            for x in X_test
        ])
        print(f'  特征维度: {X_train_features.shape[1]}')
        
        print(f'\n[3/4] 训练随机森林分类器...')
        ml_model = train_ml_classifier(X_train_features, y_train)
        print(f'  模型训练完成')
        
        print(f'\n[4/4] 模型评估...')
        y_pred_train = ml_model.predict(X_train_features)
        y_pred_test = ml_model.predict(X_test_features)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f'  训练集准确率: {train_acc:.2%}')
        print(f'  测试集准确率: {test_acc:.2%}')
        
        print(f'\n分类混淆矩阵:')
        cm = confusion_matrix(y_test, y_pred_test)
        print('  ' + ''.join([f'{name:>12}' for name in class_names]))
        for i, row in enumerate(cm):
            print(f'  {class_names[i][:4]:>4}' + ''.join([f'{v:>12}' for v in row]))
        
        ml_type = 'rf'
    
    else:
        print('\nPyTorch和scikit-learn均未安装，跳过机器学习部分')
        print('可运行: pip install scikit-learn 安装机器学习库')
    
    print('\n' + '=' * 80)
    print('阶段三: 组合诊断 (传统方法 + 机器学习)')
    print('=' * 80)
    
    combined_results = []
    
    for fault_type, case_name in zip(test_cases, case_names):
        print(f'\n测试案例: {case_name}')
        print('-' * 60)
        
        t, x, params = generate_gearbox_signal(fs=fs, duration=1.0,
                                               fault_type=fault_type, seed=100)
        
        diagnosis = combined_diagnosis(
            x, fs, params['f_mesh'], params['f_shaft'],
            ml_model=ml_model, class_names=class_names, use_ml=ml_type if ml_type else 'none'
        )
        
        print(f'  传统方法诊断: {diagnosis["traditional"]["status"]} '
              f'({diagnosis["traditional"]["confidence"]:.1%})')
        
        if diagnosis.get('ml'):
            print(f'  ML诊断 ({diagnosis["ml"]["type"]}): {diagnosis["ml"]["prediction"]} '
                  f'({diagnosis["ml"]["confidence"]:.1%})')
        
        print(f'  最终诊断: {diagnosis["final"]} '
              f'(置信度: {diagnosis["final_confidence"]:.1%})')
        
        is_correct = (
            (fault_type is None and '正常' in diagnosis['final']) or
            (fault_type == 'wear' and '齿面剥落' in diagnosis['final']) or
            (fault_type == 'broken' and '断齿' in diagnosis['final'])
        )
        
        combined_results.append({
            'case': case_name,
            'final': diagnosis['final'],
            'confidence': diagnosis['final_confidence'],
            'correct': is_correct
        })
    
    print('\n' + '=' * 80)
    print('组合诊断结果汇总:')
    print('=' * 80)
    correct_count = sum(1 for r in combined_results if r['correct'])
    for res in combined_results:
        status = '✓' if res['correct'] else '✗'
        print(f'  {status} {res["case"]}: {res["final"]} (置信度: {res["confidence"]:.1%})')
    print(f'\n组合诊断准确率: {correct_count}/{len(combined_results)} ({100*correct_count/len(combined_results):.1f}%)')
    
    print('\n' + '=' * 80)
    print('所有分析完成!')
    print('=' * 80)


if __name__ == '__main__':
    main()
