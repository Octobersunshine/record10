import numpy as np
import mne
from eeg_cross_correlation import (
    compute_cross_correlation,
    compute_functional_connectivity,
    compute_dynamic_functional_connectivity,
    plot_cross_correlation,
    plot_functional_connectivity,
    plot_topomap,
    plot_connectivity_timeseries,
    plot_dynamic_connectivity_heatmap,
    plot_dynamic_topomap_animation
)


def generate_simulated_eeg(n_channels=10, n_times=10000, fs=250):
    """
    生成模拟EEG数据用于演示
    """
    np.random.seed(42)
    
    data = np.random.randn(n_channels, n_times)
    
    for i in range(1, n_channels):
        alpha_wave = 0.5 * np.sin(2 * np.pi * 10 * np.arange(n_times) / fs)
        data[i] = 0.6 * data[0] + 0.4 * data[i] + alpha_wave
    
    channel_names = [f'EEG{i+1:02d}' for i in range(n_channels)]
    
    return data, channel_names, fs


def example_basic_usage():
    """
    示例1: 基本使用流程
    """
    print("示例1: 基本使用流程")
    print("-" * 50)
    
    data, channel_names, fs = generate_simulated_eeg()
    print(f"数据形状: {data.shape}")
    print(f"通道数: {len(channel_names)}")
    
    corr_matrix, lags_ms = compute_cross_correlation(
        data, channel_names, max_lag=100, fs=fs
    )
    
    print(f"互相关矩阵形状: {corr_matrix.shape}")
    print(f"滞后范围: {lags_ms[0]:.1f} ms 到 {lags_ms[-1]:.1f} ms")
    
    conn_matrix = compute_functional_connectivity(corr_matrix, lags_ms, method='max_abs')
    
    print("\n功能连接矩阵 (前5x5):")
    print(np.round(conn_matrix[:5, :5], 3))
    
    print("\n" + "=" * 50 + "\n")


def example_visualization():
    """
    示例2: 可视化功能
    """
    print("示例2: 可视化演示")
    print("-" * 50)
    
    data, channel_names, fs = generate_simulated_eeg(n_channels=20)
    
    corr_matrix, lags_ms = compute_cross_correlation(
        data, channel_names, max_lag=50, fs=fs
    )
    
    conn_matrix = compute_functional_connectivity(corr_matrix, lags_ms, method='max_abs')
    
    print("绘制通道对互相关函数...")
    plot_cross_correlation(corr_matrix, lags_ms, channel_names, 
                          channel_pairs=[(0, 1), (0, 5), (2, 3)])
    
    print("绘制功能连接矩阵热力图...")
    plot_functional_connectivity(conn_matrix, channel_names)
    
    print("绘制连接拓扑图...")
    info = mne.create_info(channel_names, sfreq=fs, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, match_case=False, on_missing='ignore')
    plot_topomap(conn_matrix, channel_names, info, channel_idx=0)
    
    print("\n" + "=" * 50 + "\n")


def example_different_methods():
    """
    示例3: 不同功能连接计算方法对比
    """
    print("示例3: 不同功能连接计算方法对比")
    print("-" * 50)
    
    data, channel_names, fs = generate_simulated_eeg(n_channels=15)
    
    corr_matrix, lags_ms = compute_cross_correlation(
        data, channel_names, max_lag=100, fs=fs
    )
    
    methods = ['max_abs', 'zero_lag', 'max_pos']
    
    for method in methods:
        conn_matrix = compute_functional_connectivity(corr_matrix, lags_ms, method=method)
        print(f"\n{method}方法 - 平均连接强度: {np.mean(conn_matrix):.4f}")
    
    print("\n" + "=" * 50 + "\n")


def example_dynamic_connectivity():
    """
    示例4: 动态功能连接分析
    """
    print("示例4: 动态功能连接分析")
    print("-" * 50)
    
    n_channels = 10
    n_times = 20000
    fs = 250
    
    data = np.random.randn(n_channels, n_times)
    t = np.arange(n_times) / fs
    
    alpha_wave = 0.5 * np.sin(2 * np.pi * 10 * t)
    data[0] += alpha_wave
    data[1] += 0.8 * alpha_wave + 0.2 * np.random.randn(n_times)
    
    modulation = 0.5 * (1 + np.sin(2 * np.pi * 0.1 * t))
    data[2] += (modulation * alpha_wave) + 0.3 * np.random.randn(n_times)
    
    channel_names = [f'Ch{i+1:02d}' for i in range(n_channels)]
    
    window_size = int(2 * fs)
    step_size = int(0.5 * fs)
    
    print(f"数据形状: {data.shape}")
    print(f"窗口大小: {window_size} 样本点 ({window_size/fs:.1f}秒)")
    print(f"步长: {step_size} 样本点 ({step_size/fs:.1f}秒)")
    
    conn_series, window_times = compute_dynamic_functional_connectivity(
        data, channel_names, fs=fs, window_size=window_size,
        step_size=step_size, max_lag=50, method='max_abs'
    )
    
    print(f"动态连接矩阵序列形状: {conn_series.shape}")
    print(f"时间点数: {len(window_times)}")
    
    mean_conn = np.mean(conn_series, axis=(1, 2))
    print(f"\n平均连接强度 - 最小值: {np.min(mean_conn):.4f}, 最大值: {np.max(mean_conn):.4f}")
    
    print("\n" + "=" * 50 + "\n")
    
    return conn_series, window_times, channel_names, data, fs


if __name__ == "__main__":
    example_basic_usage()
    example_different_methods()
    conn_series, window_times, channel_names, data, fs = example_dynamic_connectivity()
    
    print("要运行可视化示例，请取消相关函数的注释")
    # plot_connectivity_timeseries(conn_series, window_times, channel_names,
    #                             pairs_to_plot=[(0,1), (0,2), (1,2)])
    # plot_dynamic_connectivity_heatmap(conn_series, window_times, channel_names, channel_pair=(0,1))
    # 
    # info = mne.create_info(channel_names, sfreq=fs, ch_types='eeg')
    # montage = mne.channels.make_standard_montage('standard_1020')
    # info.set_montage(montage, match_case=False, on_missing='ignore')
    # plot_dynamic_topomap_animation(conn_series, window_times, channel_names, info, channel_idx=0, n_frames=5)
