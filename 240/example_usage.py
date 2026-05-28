import numpy as np
from peak_detection import detect_peaks, estimate_noise


def main():
    print("=== 一维信号峰值检测示例 (含排序/自动阈值/FWHM) ===\n")

    # 示例1: 基本用法 - 元组解包向后兼容
    print("示例1: 基本用法 (向后兼容元组解包)")
    signal = [1, 3, 7, 1, 4, 6, 2, 5, 8, 3]
    radius = 2
    pos, val = detect_peaks(signal, radius)
    print(f"信号: {signal}")
    print(f"峰值位置: {pos}, 幅值: {val}")
    print()

    # 示例2: PeaksResult完整对象访问
    print("示例2: PeaksResult完整对象访问")
    result = detect_peaks(signal, radius)
    print(f"  positions  = {result.positions}")
    print(f"  amplitudes = {result.amplitudes}")
    print(f"  fwhm       = {[round(f, 2) for f in result.fwhm]}")
    print(f"  prominences= {[round(p, 2) for p in result.prominences]}")
    print(f"  noise_level= {result.noise_level:.4f}")
    print()

    # 示例3: 按幅值排序
    print("示例3: 按幅值排序")
    r = detect_peaks(signal, radius, sort_by='amplitude')
    print(f"  位置={r.positions}, 幅值={r.amplitudes}")
    print()

    # 示例4: 按显著性排序
    print("示例4: 按显著性排序")
    signal2 = [0, 5, 2, 3, 2, 8, 2, 4, 0]
    r = detect_peaks(signal2, 2, sort_by='significance')
    print(f"信号: {signal2}")
    print(f"  按显著性降序:")
    for i in range(len(r)):
        print(f"    位置={r.positions[i]}, 幅值={r.amplitudes[i]:.1f}, "
              f"显著性={r.prominences[i]:.2f}, FWHM={r.fwhm[i]:.2f}")
    print()

    # 示例5: 自动阈值 (基于噪声水平)
    print("示例5: 基于噪声水平自动设置阈值")
    np.random.seed(42)
    x = np.linspace(0, 6 * np.pi, 200)
    signal = np.sin(x) + 0.2 * np.random.randn(len(x))
    radius = 10
    print(f"含噪正弦波 (noise_std≈0.2), radius={radius}")
    print(f"  估计噪声水平: {estimate_noise(signal):.4f}")
    r_no = detect_peaks(signal, radius, include_boundaries=False)
    print(f"  无自动阈值: {len(r_no)} 个峰值")
    r_auto = detect_peaks(signal, radius, auto_delta=True, noise_factor=3.0,
                           include_boundaries=False)
    print(f"  auto_delta=True (3σ): {len(r_auto)} 个峰值")
    r_strict = detect_peaks(signal, radius, auto_delta=True, noise_factor=5.0,
                             include_boundaries=False)
    print(f"  auto_delta=True (5σ): {len(r_strict)} 个峰值")
    print()

    # 示例6: FWHM计算
    print("示例6: 半高宽(FWHM)计算")
    tri = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0]
    r = detect_peaks(tri, 3)
    print(f"对称三角峰: {tri}")
    print(f"  峰值: 位置={r.positions[0]}, 幅值={r.amplitudes[0]:.1f}")
    print(f"  FWHM={r.fwhm[0]:.2f} (理论值=5.0)")
    print(f"  显著性={r.prominences[0]:.2f}")

    gauss_x = np.linspace(-5, 5, 101)
    gauss = 5.0 * np.exp(-gauss_x ** 2)
    r_g = detect_peaks(gauss, 10)
    print(f"高斯峰: 5*exp(-x^2), 采样101点")
    print(f"  峰值: 位置={r_g.positions[0]}, 幅值={r_g.amplitudes[0]:.4f}")
    print(f"  FWHM={r_g.fwhm[0]:.2f} (理论值≈{2 * np.sqrt(np.log(2)) * 10:.2f} 采样点)")
    print(f"  显著性={r_g.prominences[0]:.4f}")
    print()

    # 示例7: 含噪正弦波完整分析
    print("示例7: 含噪正弦波完整分析")
    np.random.seed(42)
    x = np.linspace(0, 8 * np.pi, 150)
    signal = np.sin(x) + 0.3 * np.random.randn(len(x))
    radius = 8

    r = detect_peaks(signal, radius, auto_delta=True, noise_factor=3.0,
                      sort_by='significance', include_boundaries=False)
    print(f"信号长度={len(signal)}, radius={radius}")
    print(f"噪声估计: {r.noise_level:.4f}")
    print(f"检测到 {len(r)} 个峰值 (按显著性降序):")
    for i in range(len(r)):
        print(f"  峰{i + 1}: 位置={r.positions[i]:3d}, 幅值={r.amplitudes[i]:+.4f}, "
              f"FWHM={r.fwhm[i]:.2f}, 显著性={r.prominences[i]:.4f}")
    print()

    # 示例8: 综合应用 - 多峰信号
    print("示例8: 综合应用 - 多峰信号详细分析")
    signal = [0, 1, 5, 5, 3, 1, 0, 2, 8, 2, 0, 3, 6, 3, 0, 1, 2, 1, 0]
    radius = 3
    r = detect_peaks(signal, radius, sort_by='significance')
    print(f"信号: {signal}")
    print(f"噪声水平: {r.noise_level:.4f}")
    print(f"{'峰':>3} {'位置':>4} {'幅值':>6} {'显著性':>8} {'FWHM':>6}")
    print(f"{'---':>3} {'----':>4} {'------':>6} {'--------':>8} {'------':>6}")
    for i in range(len(r)):
        print(f"{i + 1:>3} {r.positions[i]:>4} {r.amplitudes[i]:>6.1f} "
              f"{r.prominences[i]:>8.2f} {r.fwhm[i]:>6.2f}")


if __name__ == "__main__":
    main()
