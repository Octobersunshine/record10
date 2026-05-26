import numpy as np
from prony_analysis import PronyAnalyzer, print_results, plot_results, generate_test_signal


def example_1_tls_esprit():
    """示例1：使用TLS-ESPRIT方法分析"""
    print("="*70)
    print("示例1：TLS-ESPRIT方法 - 总体最小二乘旋转不变技术")
    print("="*70)

    fs = 100.0
    duration = 15.0
    true_modes = [
        {'freq': 0.5, 'damping': 0.04, 'amp': 1.0, 'phase': 0.0},
        {'freq': 1.0, 'damping': 0.06, 'amp': 0.5, 'phase': 0.2},
    ]

    t, signal = generate_test_signal(fs=fs, duration=duration,
                                     modes=true_modes, noise_level=0.02)

    analyzer = PronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))
    result = analyzer.analyze(signal, method='tls_esprit', order=25)

    print("\n真实模式:")
    for i, mode in enumerate(true_modes):
        print(f"  Mode {i+1}: {mode['freq']:.2f} Hz, 阻尼比 = {mode['damping']:.3f}")

    print_results(result)
    print(f"有效秩: {result.get('effective_rank', 'N/A')}")
    print(f"重构MSE: {np.mean((result['processed'] - result['reconstructed'])**2):.6f}")


def example_2_matrix_pencil():
    """示例2：使用改进矩阵束方法分析"""
    print("\n" + "="*70)
    print("示例2：改进矩阵束方法")
    print("="*70)

    fs = 100.0
    duration = 20.0
    true_modes = [
        {'freq': 0.3, 'damping': 0.03, 'amp': 1.0, 'phase': 0.0},
        {'freq': 0.8, 'damping': 0.05, 'amp': 0.6, 'phase': 0.3},
        {'freq': 1.5, 'damping': 0.07, 'amp': 0.3, 'phase': 0.1},
    ]

    t, signal = generate_test_signal(fs=fs, duration=duration,
                                     modes=true_modes, noise_level=0.02)

    analyzer = PronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))
    result = analyzer.analyze(signal, method='matrix_pencil', order=35)

    print("\n真实模式:")
    for i, mode in enumerate(true_modes):
        print(f"  Mode {i+1}: {mode['freq']:.2f} Hz, 阻尼比 = {mode['damping']:.3f}")

    print_results(result)
    print(f"有效秩: {result.get('effective_rank', 'N/A')}")
    print(f"重构MSE: {np.mean((result['processed'] - result['reconstructed'])**2):.6f}")


def example_3_stability_diagram():
    """示例3：使用稳定性图验证模式"""
    print("\n" + "="*70)
    print("示例3：TLS-ESPRIT + 稳定性图验证")
    print("="*70)

    fs = 100.0
    duration = 25.0
    true_modes = [
        {'freq': 0.4, 'damping': 0.04, 'amp': 1.0, 'phase': 0.0},
        {'freq': 1.2, 'damping': 0.06, 'amp': 0.5, 'phase': 0.3},
    ]

    t, signal = generate_test_signal(fs=fs, duration=duration,
                                     modes=true_modes, noise_level=0.03)

    analyzer = PronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))
    result = analyzer.analyze(signal, method='tls_esprit', order=40, use_stability=True)

    print("\n真实模式:")
    for i, mode in enumerate(true_modes):
        print(f"  Mode {i+1}: {mode['freq']:.2f} Hz, 阻尼比 = {mode['damping']:.3f}")

    print_results(result)
    if 'stable_modes' in result:
        print("\n稳定性分析结果:")
        for i, mode in enumerate(result['stable_modes']):
            print(f"  模式 {i+1}: {mode['freq']:.4f} Hz, 阻尼比 = {mode['damping']:.4f}, "
                  f"稳定度 = {mode['stability_score']:.2f}")

    print(f"重构MSE: {np.mean((result['processed'] - result['reconstructed'])**2):.6f}")


def example_4_high_noise():
    """示例4：高噪声情况下的鲁棒性测试"""
    print("\n" + "="*70)
    print("示例4：高噪声鲁棒性测试 (噪声水平 = 5%)")
    print("="*70)

    fs = 100.0
    duration = 20.0
    true_modes = [
        {'freq': 0.6, 'damping': 0.05, 'amp': 1.0, 'phase': 0.0},
        {'freq': 1.1, 'damping': 0.07, 'amp': 0.6, 'phase': 0.2},
    ]

    t, signal = generate_test_signal(fs=fs, duration=duration,
                                     modes=true_modes, noise_level=0.05)

    analyzer = PronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))

    print("\n--- 改进矩阵束方法 ---")
    result_mp = analyzer.analyze(signal, method='matrix_pencil', order=35)
    print_results(result_mp, true_modes=true_modes)
    print(f"有效秩: {result_mp.get('effective_rank', 'N/A')}")
    print(f"重构MSE: {np.mean((result_mp['processed'] - result_mp['reconstructed'])**2):.6f}")

    print("\n--- TLS-ESPRIT + 稳定性图 ---")
    result_stab = analyzer.analyze(signal, method='tls_esprit', order=35, use_stability=True)
    print_results(result_stab, true_modes=true_modes)
    print(f"重构MSE: {np.mean((result_stab['processed'] - result_stab['reconstructed'])**2):.6f}")


def example_5_weak_damping():
    """示例5：弱阻尼模式识别"""
    print("\n" + "="*70)
    print("示例5：弱阻尼模式识别 (阻尼比 < 2%)")
    print("="*70)

    fs = 100.0
    duration = 30.0
    true_modes = [
        {'freq': 0.3, 'damping': 0.015, 'amp': 1.0, 'phase': 0.0},
        {'freq': 0.9, 'damping': 0.025, 'amp': 0.5, 'phase': 0.3},
    ]

    t, signal = generate_test_signal(fs=fs, duration=duration,
                                     modes=true_modes, noise_level=0.01)

    analyzer = PronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))
    result = analyzer.analyze(signal, method='tls_esprit', order=50, use_stability=True)

    print("\n真实模式:")
    for i, mode in enumerate(true_modes):
        print(f"  Mode {i+1}: {mode['freq']:.2f} Hz, 阻尼比 = {mode['damping']:.3f}")

    print_results(result)
    if 'stable_modes' in result:
        print("\n稳定性分析结果:")
        for i, mode in enumerate(result['stable_modes']):
            print(f"  模式 {i+1}: {mode['freq']:.4f} Hz, 阻尼比 = {mode['damping']:.4f}, "
                  f"稳定度 = {mode['stability_score']:.2f}")

    print(f"重构MSE: {np.mean((result['processed'] - result['reconstructed'])**2):.6f}")
    plot_results(result, save_path='weak_damping_result.png')


if __name__ == '__main__':
    np.random.seed(123)

    print("="*70)
    print("电力系统低频振荡模式辨识 - 高级示例")
    print("使用 TLS-ESPRIT / 改进矩阵束方法")
    print("="*70)

    example_1_tls_esprit()
    example_2_matrix_pencil()
    example_3_stability_diagram()
    example_4_high_noise()
    example_5_weak_damping()

    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)
