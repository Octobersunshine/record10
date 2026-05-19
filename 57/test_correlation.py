import numpy as np
from eeg_cross_correlation import compute_cross_correlation


def test_normalization():
    """
    测试互相关归一化是否正确
    """
    print("=" * 60)
    print("互相关归一化测试")
    print("=" * 60)
    
    np.random.seed(42)
    n_channels = 5
    n_samples = 10000
    fs = 250
    
    data = np.random.randn(n_channels, n_samples)
    channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    corr_matrix, lags_ms = compute_cross_correlation(
        data, channel_names, max_lag=100, fs=fs
    )
    
    zero_lag_idx = np.argmin(np.abs(lags_ms))
    
    print("\n自相关零延迟值测试:")
    print("-" * 40)
    all_correct = True
    for i in range(n_channels):
        autocorr_zero = corr_matrix[i, i, zero_lag_idx]
        print(f"通道 {i+1}: 自相关(零延迟) = {autocorr_zero:.6f}", end="")
        
        if np.abs(autocorr_zero - 1.0) < 1e-3:
            print(" ✓ 正确")
        else:
            print(" ✗ 错误 (应为 ~1.0)")
            all_correct = False
    
    print("\n互相关范围测试:")
    print("-" * 40)
    max_corr = np.max(corr_matrix)
    min_corr = np.min(corr_matrix)
    print(f"最大相关值: {max_corr:.6f}")
    print(f"最小相关值: {min_corr:.6f}")
    
    if max_corr <= 1.01 and min_corr >= -1.01:
        print("✓ 相关值在合理范围内 [-1, 1]")
    else:
        print("✗ 相关值超出合理范围!")
        all_correct = False
    
    print("\n" + "=" * 60)
    if all_correct:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败!")
    print("=" * 60)
    
    return all_correct


def test_correlation_properties():
    """
    测试互相关函数的性质
    """
    print("\n\n互相关函数性质测试")
    print("=" * 60)
    
    n_samples = 5000
    fs = 250
    
    x = np.sin(2 * np.pi * 10 * np.arange(n_samples) / fs)
    y = 0.8 * x + 0.2 * np.random.randn(n_samples)
    
    data = np.vstack([x, y])
    channel_names = ['X', 'Y']
    
    corr_matrix, lags_ms = compute_cross_correlation(
        data, channel_names, max_lag=50, fs=fs
    )
    
    zero_lag_idx = np.argmin(np.abs(lags_ms))
    
    print(f"\nX的自相关(零延迟): {corr_matrix[0, 0, zero_lag_idx]:.6f}")
    print(f"Y的自相关(零延迟): {corr_matrix[1, 1, zero_lag_idx]:.6f}")
    print(f"X-Y互相关(零延迟): {corr_matrix[0, 1, zero_lag_idx]:.6f}")
    print(f"预期X-Y互相关: ~0.8")
    
    max_corr_xy = np.max(corr_matrix[0, 1])
    max_lag_idx = np.argmax(corr_matrix[0, 1])
    max_lag = lags_ms[max_lag_idx]
    print(f"\nX-Y最大互相关: {max_corr_xy:.6f} 在滞后 {max_lag:.1f} ms")
    
    print("\n" + "=" * 60)


def test_dynamic_connectivity():
    """
    测试动态功能连接计算
    """
    print("\n\n动态功能连接测试")
    print("=" * 60)
    
    np.random.seed(42)
    n_channels = 8
    n_samples = 10000
    fs = 250
    
    t = np.arange(n_samples) / fs
    data = np.random.randn(n_channels, n_samples)
    
    alpha_wave = np.sin(2 * np.pi * 10 * t)
    modulation = 0.5 * (1 + np.sin(2 * np.pi * 0.05 * t))
    
    data[0] += 0.5 * alpha_wave
    data[1] += 0.4 * modulation * alpha_wave + 0.2 * np.random.randn(n_samples)
    data[2] += 0.3 * alpha_wave
    
    channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    window_size = int(2 * fs)
    step_size = int(0.5 * fs)
    
    print(f"\n窗口大小: {window_size} 样本点 ({window_size/fs:.1f}秒)")
    print(f"步长: {step_size} 样本点 ({step_size/fs:.1f}秒)")
    
    from eeg_cross_correlation import compute_dynamic_functional_connectivity
    
    conn_series, window_times = compute_dynamic_functional_connectivity(
        data, channel_names, fs=fs, window_size=window_size,
        step_size=step_size, max_lag=50, method='max_abs'
    )
    
    print(f"输出形状: {conn_series.shape}")
    print(f"时间范围: {window_times[0]:.1f}s 到 {window_times[-1]:.1f}s")
    
    conn_ch1_ch2 = conn_series[:, 0, 1]
    conn_ch1_ch3 = conn_series[:, 0, 2]
    
    print(f"\nCh1-Ch2 连接强度:")
    print(f"  平均值: {np.mean(conn_ch1_ch2):.4f}")
    print(f"  标准差: {np.std(conn_ch1_ch2):.4f}")
    print(f"  变异系数: {np.std(conn_ch1_ch2)/np.mean(conn_ch1_ch2):.4f}")
    
    print(f"\nCh1-Ch3 连接强度:")
    print(f"  平均值: {np.mean(conn_ch1_ch3):.4f}")
    print(f"  标准差: {np.std(conn_ch1_ch3):.4f}")
    
    print("\n✓ 动态功能连接计算完成")
    print("\n" + "=" * 60)
    
    return conn_series, window_times, channel_names


if __name__ == "__main__":
    test_normalization()
    test_correlation_properties()
    test_dynamic_connectivity()
