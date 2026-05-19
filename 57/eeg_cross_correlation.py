import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from scipy.signal import correlate
from scipy.stats import zscore


def load_sample_eeg_data():
    """
    加载MNE示例EEG数据
    """
    data_path = sample.data_path()
    raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
    raw.pick(picks)
    
    raw.filter(1., 40., fir_design='firwin')
    
    events = mne.find_events(raw)
    event_id = {'auditory/left': 1, 'auditory/right': 2}
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, preload=True)
    
    return raw, epochs


def load_custom_eeg_data(file_path):
    """
    加载自定义EEG数据文件
    支持 .fif, .edf, .set 等格式
    """
    if file_path.endswith('.fif'):
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif file_path.endswith('.edf'):
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif file_path.endswith('.set'):
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    return raw


def compute_cross_correlation(data, channel_names, max_lag=100, fs=1000):
    """
    计算多通道EEG数据的互相关函数
    
    参数:
        data: 形状为 (n_channels, n_times) 的EEG数据
        channel_names: 通道名称列表
        max_lag: 最大滞后（样本点）
        fs: 采样频率（Hz）
    
    返回:
        corr_matrix: 互相关矩阵 (n_channels, n_channels, 2*max_lag+1)
        lags: 滞后时间数组（毫秒）
    """
    n_channels = len(channel_names)
    lags = np.arange(-max_lag, max_lag + 1)
    lags_ms = lags / fs * 1000
    
    corr_matrix = np.zeros((n_channels, n_channels, len(lags)))
    
    data_norm = zscore(data, axis=1)
    n_samples = data.shape[1]
    
    for i in range(n_channels):
        for j in range(n_channels):
            if i <= j:
                x = data_norm[i]
                y = data_norm[j]
                
                corr = correlate(x, y, mode='full')
                center = len(corr) // 2
                corr_slice = corr[center - max_lag:center + max_lag + 1]
                
                corr_slice /= n_samples
                
                corr_matrix[i, j] = corr_slice
                corr_matrix[j, i] = corr_slice
    
    return corr_matrix, lags_ms


def compute_functional_connectivity(corr_matrix, lags_ms, method='max_abs'):
    """
    基于互相关计算功能连接矩阵
    
    参数:
        corr_matrix: 互相关矩阵
        lags_ms: 滞后时间数组
        method: 连接强度计算方法
            'max_abs': 最大绝对值
            'zero_lag': 零滞后相关
            'max_pos': 最大正相关
    
    返回:
        conn_matrix: 功能连接矩阵 (n_channels, n_channels)
    """
    n_channels = corr_matrix.shape[0]
    conn_matrix = np.zeros((n_channels, n_channels))
    
    if method == 'max_abs':
        conn_matrix = np.max(np.abs(corr_matrix), axis=2)
    elif method == 'zero_lag':
        zero_idx = np.argmin(np.abs(lags_ms))
        conn_matrix = corr_matrix[:, :, zero_idx]
    elif method == 'max_pos':
        conn_matrix = np.max(corr_matrix, axis=2)
    
    return conn_matrix


def plot_cross_correlation(corr_matrix, lags_ms, channel_names, channel_pairs=None):
    """
    绘制选定通道对的互相关函数
    
    参数:
        corr_matrix: 互相关矩阵
        lags_ms: 滞后时间数组
        channel_names: 通道名称列表
        channel_pairs: 要绘制的通道对列表，如 [(0,1), (2,3)]
    """
    n_channels = len(channel_names)
    
    if channel_pairs is None:
        channel_pairs = [(i, i+1) for i in range(min(5, n_channels-1))]
    
    n_pairs = len(channel_pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    for idx, (i, j) in enumerate(channel_pairs):
        if idx < len(axes):
            ax = axes[idx]
            corr = corr_matrix[i, j]
            ax.plot(lags_ms, corr, linewidth=2)
            ax.axvline(0, color='r', linestyle='--', alpha=0.7, label='Zero Lag')
            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Lag (ms)')
            ax.set_ylabel('Cross-correlation')
            ax.set_title(f'{channel_names[i]} vs {channel_names[j]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_functional_connectivity(conn_matrix, channel_names, title='Functional Connectivity Matrix'):
    """
    绘制功能连接矩阵热力图
    
    参数:
        conn_matrix: 功能连接矩阵
        channel_names: 通道名称列表
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(conn_matrix, cmap='RdBu_r', aspect='auto', 
                   vmin=-np.max(np.abs(conn_matrix)), vmax=np.max(np.abs(conn_matrix)))
    
    ax.set_xticks(np.arange(len(channel_names)))
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=90, fontsize=8)
    ax.set_yticklabels(channel_names, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Connection Strength')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_topomap(conn_matrix, channel_names, info, channel_idx=None):
    """
    绘制特定通道与其他通道连接强度的拓扑图
    
    参数:
        conn_matrix: 功能连接矩阵
        channel_names: 通道名称列表
        info: MNE Info对象
        channel_idx: 参考通道索引，默认使用第一个通道
    """
    if channel_idx is None:
        channel_idx = 0
    
    conn_values = conn_matrix[channel_idx]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    mne.viz.plot_topomap(conn_values, info, axes=ax, show=False,
                         cmap='RdBu_r', vmin=-np.max(np.abs(conn_values)))
    ax.set_title(f'Connectivity from {channel_names[channel_idx]}')
    plt.colorbar(ax.images[0], ax=ax, label='Connection Strength')
    plt.show()


def compute_dynamic_functional_connectivity(
    data, channel_names, fs=1000, window_size=1000, step_size=500,
    max_lag=100, method='max_abs'
):
    """
    滑动窗口计算动态功能连接
    
    参数:
        data: EEG数据 (n_channels, n_times)
        channel_names: 通道名称列表
        fs: 采样频率（Hz）
        window_size: 窗口大小（样本点）
        step_size: 步长（样本点）
        max_lag: 互相关最大滞后（样本点）
        method: 功能连接计算方法
    
    返回:
        conn_series: 功能连接矩阵序列 (n_windows, n_channels, n_channels)
        window_times: 每个窗口的时间点（秒）
    """
    n_channels = len(channel_names)
    n_samples = data.shape[1]
    
    n_windows = (n_samples - window_size) // step_size + 1
    
    conn_series = np.zeros((n_windows, n_channels, n_channels))
    window_times = np.zeros(n_windows)
    
    data_norm = zscore(data, axis=1)
    
    for w in range(n_windows):
        start = w * step_size
        end = start + window_size
        
        window_data = data_norm[:, start:end]
        
        corr_matrix, lags_ms = compute_cross_correlation(
            window_data, channel_names, max_lag=min(max_lag, window_size//2), fs=fs
        )
        
        conn_matrix = compute_functional_connectivity(corr_matrix, lags_ms, method=method)
        
        conn_series[w] = conn_matrix
        window_times[w] = (start + window_size / 2) / fs
    
    return conn_series, window_times


def plot_dynamic_connectivity_heatmap(
    conn_series, window_times, channel_names, channel_pair=None, figsize=(12, 6)
):
    """
    绘制动态功能连接热力图（时间x通道对）
    
    参数:
        conn_series: 功能连接矩阵序列 (n_windows, n_channels, n_channels)
        window_times: 每个窗口的时间点
        channel_names: 通道名称列表
        channel_pair: 特定通道对，None则显示上三角部分
        figsize: 图表大小
    """
    n_windows, n_channels, _ = conn_series.shape
    
    if channel_pair is not None:
        i, j = channel_pair
        conn_timeseries = conn_series[:, i, j]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(window_times, conn_timeseries, linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Connection Strength')
        ax.set_title(f'Dynamic Connectivity: {channel_names[i]} - {channel_names[j]}')
        ax.grid(True, alpha=0.3)
    else:
        n_pairs = n_channels * (n_channels - 1) // 2
        conn_matrix_flat = np.zeros((n_windows, n_pairs))
        
        pair_labels = []
        idx = 0
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                conn_matrix_flat[:, idx] = conn_series[:, i, j]
                pair_labels.append(f'{channel_names[i]}-{channel_names[j]}')
                idx += 1
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(conn_matrix_flat.T, aspect='auto', cmap='RdBu_r',
                       extent=[window_times[0], window_times[-1], n_pairs-0.5, -0.5],
                       vmin=-np.max(np.abs(conn_matrix_flat)),
                       vmax=np.max(np.abs(conn_matrix_flat)))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel Pairs')
        ax.set_title('Dynamic Functional Connectivity Matrix')
        plt.colorbar(im, ax=ax, label='Connection Strength')
    
    plt.tight_layout()
    plt.show()


def plot_connectivity_timeseries(
    conn_series, window_times, channel_names, pairs_to_plot=None, figsize=(14, 8)
):
    """
    绘制多个通道对的连接强度时间序列
    
    参数:
        conn_series: 功能连接矩阵序列
        window_times: 每个窗口的时间点
        channel_names: 通道名称列表
        pairs_to_plot: 要绘制的通道对列表
        figsize: 图表大小
    """
    n_channels = len(channel_names)
    
    if pairs_to_plot is None:
        pairs_to_plot = [(i, i+1) for i in range(min(5, n_channels-1))]
    
    n_pairs = len(pairs_to_plot)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    for idx, (i, j) in enumerate(pairs_to_plot):
        if idx < len(axes):
            ax = axes[idx]
            conn_ts = conn_series[:, i, j]
            ax.plot(window_times, conn_ts, linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Connection Strength')
            ax.set_title(f'{channel_names[i]} - {channel_names[j]}')
            ax.grid(True, alpha=0.3)
    
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_dynamic_topomap_animation(
    conn_series, window_times, channel_names, info, channel_idx=0, n_frames=5
):
    """
    绘制动态连接拓扑图的关键帧
    
    参数:
        conn_series: 功能连接矩阵序列
        window_times: 每个窗口的时间点
        channel_names: 通道名称列表
        info: MNE Info对象
        channel_idx: 参考通道索引
        n_frames: 显示的帧数
    """
    n_windows = len(window_times)
    frame_indices = np.linspace(0, n_windows-1, n_frames, dtype=int)
    
    n_cols = n_frames
    n_rows = 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4))
    if n_frames == 1:
        axes = [axes]
    
    vmax = np.max(np.abs(conn_series[:, channel_idx, :]))
    
    for idx, (ax, frame_idx) in enumerate(zip(axes, frame_indices)):
        conn_matrix = conn_series[frame_idx]
        conn_values = conn_matrix[channel_idx]
        time_point = window_times[frame_idx]
        
        im, _ = mne.viz.plot_topomap(conn_values, info, axes=ax, show=False,
                                     cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'T = {time_point:.1f}s')
    
    plt.colorbar(im, ax=axes, label='Connection Strength', orientation='horizontal',
                 fraction=0.05, pad=0.05)
    plt.suptitle(f'Dynamic Connectivity from {channel_names[channel_idx]}', y=1.02)
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：演示完整的EEG互相关分析流程
    """
    print("=" * 60)
    print("EEG 互相关分析 - 功能连接分析工具")
    print("=" * 60)
    
    print("\n[1/6] 加载EEG数据...")
    raw, epochs = load_sample_eeg_data()
    
    print(f"采样频率: {raw.info['sfreq']} Hz")
    print(f"通道数量: {len(raw.ch_names)}")
    print(f"时间长度: {raw.times[-1]:.2f} 秒")
    
    print("\n[2/6] 提取数据进行分析...")
    data = raw.get_data()
    channel_names = raw.ch_names
    fs = raw.info['sfreq']
    
    max_lag = int(0.1 * fs)
    print(f"\n[3/6] 计算互相关函数 (最大滞后: {max_lag} 样本点)...")
    corr_matrix, lags_ms = compute_cross_correlation(data, channel_names, max_lag=max_lag, fs=fs)
    
    print("\n[4/6] 计算静态功能连接矩阵...")
    conn_matrix_max = compute_functional_connectivity(corr_matrix, lags_ms, method='max_abs')
    conn_matrix_zero = compute_functional_connectivity(corr_matrix, lags_ms, method='zero_lag')
    
    print("\n[5/6] 计算动态功能连接...")
    window_size = int(2 * fs)
    step_size = int(0.5 * fs)
    print(f"窗口大小: {window_size} 样本点 ({window_size/fs:.1f}秒)")
    print(f"步长: {step_size} 样本点 ({step_size/fs:.1f}秒)")
    
    conn_series, window_times = compute_dynamic_functional_connectivity(
        data, channel_names, fs=fs, window_size=window_size,
        step_size=step_size, max_lag=max_lag, method='max_abs'
    )
    print(f"动态连接矩阵序列形状: {conn_series.shape}")
    
    print("\n[6/6] 可视化结果...")
    
    print("\n--- 绘制通道对的互相关函数 ---")
    channel_pairs = [(0, 1), (0, 5), (2, 7), (3, 10)]
    valid_pairs = [(i, j) for i, j in channel_pairs if i < len(channel_names) and j < len(channel_names)]
    plot_cross_correlation(corr_matrix, lags_ms, channel_names, valid_pairs)
    
    print("\n--- 绘制功能连接矩阵 (最大绝对值) ---")
    plot_functional_connectivity(conn_matrix_max, channel_names, 
                                  title='Functional Connectivity (Max Absolute Cross-correlation)')
    
    print("\n--- 绘制功能连接矩阵 (零滞后) ---")
    plot_functional_connectivity(conn_matrix_zero, channel_names, 
                                  title='Functional Connectivity (Zero-lag Cross-correlation)')
    
    print("\n--- 绘制连接强度拓扑图 ---")
    info_2d = mne.create_info(channel_names, sfreq=fs, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info_2d.set_montage(montage, match_case=False, on_missing='ignore')
    
    plot_topomap(conn_matrix_max, channel_names, info_2d, channel_idx=0)
    
    print("\n--- 绘制动态功能连接时间序列 ---")
    if len(valid_pairs) > 0:
        plot_connectivity_timeseries(conn_series, window_times, channel_names, pairs_to_plot=valid_pairs[:4])
    
    print("\n--- 绘制动态功能连接热力图 ---")
    if len(valid_pairs) > 0:
        plot_dynamic_connectivity_heatmap(conn_series, window_times, channel_names, channel_pair=valid_pairs[0])
    
    print("\n--- 绘制动态连接拓扑图关键帧 ---")
    plot_dynamic_topomap_animation(conn_series, window_times, channel_names, info_2d, channel_idx=0, n_frames=5)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    return corr_matrix, conn_matrix_max, conn_matrix_zero, lags_ms, conn_series, window_times


if __name__ == "__main__":
    results = main()
