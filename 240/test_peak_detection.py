import numpy as np
from peak_detection import detect_peaks, estimate_noise, compute_prominences, compute_fwhm


def test_basic_peaks():
    print("测试1 - 基础峰值检测 (向后兼容解包):")
    signal = [1, 3, 7, 1, 4, 6, 2, 5, 8, 3]
    radius = 2
    pos, val = detect_peaks(signal, radius)
    print(f"信号: {signal}")
    print(f"半径: {radius}")
    print(f"峰值位置: {pos}")
    print(f"峰值幅值: {val}")

    result = detect_peaks(signal, radius)
    print(f"PeaksResult对象: positions={result.positions}, amplitudes={result.amplitudes}")
    print(f"  fwhm={[round(f, 2) for f in result.fwhm]}")
    print(f"  prominences={[round(p, 2) for p in result.prominences]}")
    print(f"  noise_level={result.noise_level:.4f}")
    print()


def test_edge_cases():
    print("测试2 - 边界情况:")

    empty_signal = []
    result = detect_peaks(empty_signal, 2)
    print(f"空信号: positions={result.positions}, amplitudes={result.amplitudes}")

    single = [5]
    result = detect_peaks(single, 2)
    print(f"单元素: positions={result.positions}, amplitudes={result.amplitudes}")

    two = [1, 3]
    result = detect_peaks(two, 2)
    print(f"两元素: positions={result.positions}, amplitudes={result.amplitudes}")
    print()


def test_plateau_detection():
    print("测试3 - 平坦区域(平台)检测:")

    flat = [2, 2, 2, 2, 2]
    r1 = detect_peaks(flat, 2, include_plateau_midpoint=True)
    print(f"全平台信号 {flat}:")
    print(f"  包含平台中点: positions={r1.positions}, amplitudes={r1.amplitudes}")

    r2 = detect_peaks(flat, 2, include_plateau_midpoint=False)
    print(f"  排除平台: positions={r2.positions}, amplitudes={r2.amplitudes}")

    signal = [1, 3, 5, 5, 5, 3, 1]
    r3 = detect_peaks(signal, 2, include_plateau_midpoint=True)
    print(f"平台峰值信号 {signal}:")
    print(f"  包含平台中点: positions={r3.positions}, fwhm={[round(f, 2) for f in r3.fwhm]}")
    print()


def test_delta_threshold():
    print("测试4 - 相对高度阈值(delta):")

    signal = [1, 3, 5, 4, 6, 5, 2]
    radius = 2
    print(f"信号: {signal}, 半径: {radius}")

    for delta in [0, 1, 2, 3]:
        pos, val = detect_peaks(signal, radius, delta=delta)
        print(f"  delta={delta}: 位置={pos}, 幅值={val}")
    print()


def test_strict_monotonic():
    print("测试5 - 严格单调模式:")

    signal = [1, 3, 5, 4, 6, 7, 5, 2]
    radius = 2
    print(f"信号: {signal}, 半径: {radius}")

    r1 = detect_peaks(signal, radius, strict_monotonic=False)
    print(f"  非严格模式: positions={r1.positions}, prominences={[round(p, 2) for p in r1.prominences]}")

    r2 = detect_peaks(signal, radius, strict_monotonic=True)
    print(f"  严格模式: positions={r2.positions}, prominences={[round(p, 2) for p in r2.prominences]}")
    print()


def test_sort_by_amplitude():
    print("测试6 - 按幅值排序:")
    signal = [1, 3, 7, 1, 4, 6, 2, 5, 8, 3]
    radius = 2

    r_none = detect_peaks(signal, radius, sort_by='none')
    print(f"原始顺序: positions={r_none.positions}, amplitudes={r_none.amplitudes}")

    r_amp = detect_peaks(signal, radius, sort_by='amplitude')
    print(f"按幅值降序: positions={r_amp.positions}, amplitudes={r_amp.amplitudes}")
    
    amps = r_amp.amplitudes
    is_desc = all(amps[i] >= amps[i + 1] for i in range(len(amps) - 1))
    print(f"  幅值是否降序: {is_desc}")
    print()


def test_sort_by_significance():
    print("测试7 - 按显著性排序:")
    signal = [0, 5, 2, 3, 2, 8, 2, 4, 0]
    radius = 2

    r_sig = detect_peaks(signal, radius, sort_by='significance')
    print(f"信号: {signal}")
    print(f"按显著性降序:")
    for i in range(len(r_sig)):
        print(f"  位置={r_sig.positions[i]}, 幅值={r_sig.amplitudes[i]:.1f}, "
              f"显著性={r_sig.prominences[i]:.2f}, FWHM={r_sig.fwhm[i]:.2f}")

    proms = r_sig.prominences
    is_desc = all(proms[i] >= proms[i + 1] for i in range(len(proms) - 1))
    print(f"  显著性是否降序: {is_desc}")
    print()


def test_auto_delta():
    print("测试8 - 基于噪声自动设置阈值:")
    np.random.seed(42)
    x = np.linspace(0, 6 * np.pi, 200)
    clean = np.sin(x)
    noise = 0.2 * np.random.randn(len(x))
    signal = clean + noise
    radius = 10

    r_no_auto = detect_peaks(signal, radius, include_boundaries=False)
    print(f"无自动阈值: 检测到 {len(r_no_auto)} 个峰值, noise_level={r_no_auto.noise_level:.4f}")

    for factor in [2.0, 3.0, 5.0]:
        r_auto = detect_peaks(signal, radius, auto_delta=True, noise_factor=factor,
                               include_boundaries=False)
        effective_delta = factor * r_auto.noise_level
        print(f"  auto_delta=True, noise_factor={factor}: "
              f"检测到 {len(r_auto)} 个峰值, "
              f"effective_delta={effective_delta:.4f}, noise_level={r_auto.noise_level:.4f}")
    print()


def test_fwhm():
    print("测试9 - 半高宽(FWHM)计算:")

    signal = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0]
    radius = 3
    r = detect_peaks(signal, radius)
    print(f"对称三角峰信号: {signal}")
    print(f"  峰值位置: {r.positions}, 幅值: {r.amplitudes}")
    print(f"  FWHM: {[round(f, 2) for f in r.fwhm]}")
    print(f"  显著性: {[round(p, 2) for p in r.prominences]}")

    signal2 = [0, 1, 3, 7, 3, 1, 0, 0, 0, 2, 5, 2, 0]
    radius2 = 2
    r2 = detect_peaks(signal2, radius2)
    print(f"双峰信号: {signal2}")
    for i in range(len(r2)):
        print(f"  峰{i + 1}: 位置={r2.positions[i]}, 幅值={r2.amplitudes[i]:.1f}, "
              f"FWHM={r2.fwhm[i]:.2f}, 显著性={r2.prominences[i]:.2f}")
    print()


def test_noise_estimation():
    print("测试10 - 噪声估计:")
    np.random.seed(42)

    pure_signal = np.sin(np.linspace(0, 2 * np.pi, 100))
    noise_std = estimate_noise(pure_signal)
    print(f"纯正弦波噪声估计: {noise_std:.6f}")

    noisy_signal = np.sin(np.linspace(0, 2 * np.pi, 100)) + 0.5 * np.random.randn(100)
    noisy_std = estimate_noise(noisy_signal)
    print(f"噪声std=0.5的含噪正弦波噪声估计: {noisy_std:.4f}")

    high_noise = np.sin(np.linspace(0, 2 * np.pi, 100)) + 2.0 * np.random.randn(100)
    high_std = estimate_noise(high_noise)
    print(f"噪声std=2.0的含噪正弦波噪声估计: {high_std:.4f}")
    print()


def test_prominence():
    print("测试11 - 显著性(prominence)计算:")
    signal = [0, 5, 2, 8, 2, 3, 2, 1, 0]
    radius = 2
    r = detect_peaks(signal, radius)
    print(f"信号: {signal}")
    for i in range(len(r)):
        print(f"  峰{i + 1}: 位置={r.positions[i]}, 幅值={r.amplitudes[i]:.1f}, "
              f"显著性={r.prominences[i]:.2f}")
    print()


def test_sine_wave():
    print("测试12 - 正弦波完整分析:")
    x = np.linspace(0, 4 * np.pi, 20)
    signal = np.sin(x)
    radius = 3
    r = detect_peaks(signal, radius, include_boundaries=False, sort_by='significance')
    print(f"正弦波峰值分析:")
    for i in range(len(r)):
        print(f"  峰{i + 1}: 位置={r.positions[i]}, 幅值={r.amplitudes[i]:.4f}, "
              f"FWHM={r.fwhm[i]:.2f}, 显著性={r.prominences[i]:.4f}")
    print()


def test_backward_compat():
    print("测试13 - 向后兼容性(元组解包):")
    signal = [1, 3, 7, 1, 4, 6, 2]
    radius = 2

    pos, val = detect_peaks(signal, radius)
    print(f"元组解包: pos={pos}, val={val}")

    result = detect_peaks(signal, radius)
    print(f"对象访问: positions={result.positions}, amplitudes={result.amplitudes}")
    print(f"  len(result)={len(result)}")
    print(f"  result[0]={result[0]}")
    print(f"  确认一致: {pos == result.positions and val == result.amplitudes}")
    print()


def test_complex_signal():
    print("测试14 - 综合测试(复杂信号 + 所有新功能):")
    np.random.seed(42)
    x = np.linspace(0, 8 * np.pi, 150)
    signal = np.sin(x) + 0.3 * np.random.randn(len(x))
    radius = 8

    r = detect_peaks(signal, radius, auto_delta=True, noise_factor=3.0,
                      sort_by='significance', include_boundaries=False)
    print(f"含噪正弦波 (长度={len(signal)}, radius={radius})")
    print(f"噪声估计: {r.noise_level:.4f}")
    print(f"检测到 {len(r)} 个峰值 (按显著性降序):")
    for i in range(len(r)):
        print(f"  峰{i + 1}: 位置={r.positions[i]:3d}, 幅值={r.amplitudes[i]:.4f}, "
              f"FWHM={r.fwhm[i]:.2f}, 显著性={r.prominences[i]:.4f}")


if __name__ == "__main__":
    test_basic_peaks()
    test_edge_cases()
    test_plateau_detection()
    test_delta_threshold()
    test_strict_monotonic()
    test_sort_by_amplitude()
    test_sort_by_significance()
    test_auto_delta()
    test_fwhm()
    test_noise_estimation()
    test_prominence()
    test_sine_wave()
    test_backward_compat()
    test_complex_signal()
    print("\n所有测试完成!")
