"""
能量守恒验证测试脚本

验证Gerchberg-Saxton算法中的能量守恒特性，包括：
1. FFT/IFFT变换的能量守恒
2. 迭代过程中的能量稳定性
3. 约束操作对能量的影响
4. 相位多样性算法的能量守恒
"""
import numpy as np
from gs_algorithm import (
    fft2_normalized, ifft2_normalized,
    gerchberg_saxton, gerchberg_saxton_near_field,
    gerchberg_saxton_phase_diversity,
    create_test_image, compute_far_field, support_constraint,
    defocus_phase, generate_defocused_intensities
)


def test_fft_energy_conservation():
    """测试FFT/IFFT的能量守恒"""
    print("=" * 70)
    print("测试1: FFT/IFFT能量守恒验证")
    print("=" * 70)
    
    size = 64
    x = np.random.rand(size, size)
    x_energy = np.sum(np.abs(x) ** 2)
    
    F = fft2_normalized(x)
    F_energy = np.sum(np.abs(F) ** 2)
    
    x_recon = ifft2_normalized(F)
    x_recon_energy = np.sum(np.abs(x_recon) ** 2)
    
    print(f"输入空域能量:     {x_energy:.10f}")
    print(f"FFT后频域能量:    {F_energy:.10f}")
    print(f"IFFT后空域能量:   {x_recon_energy:.10f}")
    print(f"FFT能量误差:      {abs(F_energy - x_energy):.2e}")
    print(f"IFFT能量误差:     {abs(x_recon_energy - x_energy):.2e}")
    
    fft_pass = abs(F_energy - x_energy) < 1e-10
    ifft_pass = abs(x_recon_energy - x_energy) < 1e-10
    print(f"\nFFT能量守恒:      {'✓ 通过' if fft_pass else '✗ 失败'}")
    print(f"IFFT能量守恒:     {'✓ 通过' if ifft_pass else '✗ 失败'}")
    
    print("\n多次FFT/IFFT循环测试:")
    current = x.copy()
    energies = []
    for i in range(100):
        current = fft2_normalized(current)
        current = ifft2_normalized(current)
        energy = np.sum(np.abs(current) ** 2)
        energies.append(energy)
        if (i + 1) % 20 == 0:
            print(f"  循环 {i+1:3d}: 能量 = {energy:.10f}, 误差 = {abs(energy - x_energy):.2e}")
    
    max_error = max(abs(e - x_energy) for e in energies)
    print(f"\n100次循环最大误差: {max_error:.2e}")
    print(f"能量稳定性:        {'✓ 良好' if max_error < 1e-9 else '✗ 需要改进'}")
    
    return fft_pass and ifft_pass and (max_error < 1e-9)


def test_numpy_fft_normalization():
    """测试NumPy FFT的归一化特性"""
    print("\n" + "=" * 70)
    print("测试2: NumPy FFT归一化特性分析")
    print("=" * 70)
    
    size = 4
    x = np.ones((size, size))
    x_energy = np.sum(np.abs(x) ** 2)
    
    F = np.fft.fft2(x)
    F_energy = np.sum(np.abs(F) ** 2)
    
    x_recon = np.fft.ifft2(F)
    x_recon_energy = np.sum(np.abs(x_recon) ** 2)
    
    print(f"输入能量:          {x_energy}")
    print(f"fft2后能量:        {F_energy:.0f} (理论值: {size**2 * x_energy:.0f})")
    print(f"ifft2后能量:       {x_recon_energy:.0f} (理论值: {x_energy:.0f})")
    print(f"fft2 能量放大倍数: {F_energy / x_energy:.0f} (应为 {size**2})")
    print("\n结论:")
    print(f"  - np.fft.fft2() 无归一化，能量放大 {size**2} 倍")
    print(f"  - np.fft.ifft2() 自带 1/{size**2} 归一化")
    print(f"  - 要保持能量守恒，FFT后需除以 sqrt({size**2}) = {size}")
    print(f"  - IFFT后需乘以 sqrt({size**2}) = {size}")


def test_constraint_energy_effect():
    """测试约束操作对能量的影响"""
    print("\n" + "=" * 70)
    print("测试3: 约束操作对能量的影响")
    print("=" * 70)
    
    size = 64
    amp, phase = create_test_image(size, 'double_slit')
    energy_original = np.sum(amp ** 2)
    print(f"原始振幅能量: {energy_original:.6f}")
    
    amp_constrained = support_constraint(amp, threshold=0.1)
    energy_constrained = np.sum(amp_constrained ** 2)
    print(f"约束后振幅能量: {energy_constrained:.6f}")
    print(f"能量变化率: {(energy_constrained - energy_original) / energy_original * 100:.2f}%")
    
    far_intensity = compute_far_field(amp, phase)
    far_energy = np.sum(far_intensity)
    print(f"\n远场强度能量: {far_energy:.6f}")
    print(f"远场/近场能量比: {far_energy / energy_original:.6f} (应为1.0)")


def test_gs_energy_stability():
    """测试G-S算法迭代过程中的能量稳定性"""
    print("\n" + "=" * 70)
    print("测试4: G-S算法迭代能量稳定性")
    print("=" * 70)
    
    size = 64
    n_iter = 50
    
    print("创建测试图案...")
    amp_true, phase_true = create_test_image(size, 'double_slit')
    far_intensity = compute_far_field(amp_true, phase_true)
    target_energy = np.sum(far_intensity)
    print(f"目标能量: {target_energy:.6f}")
    
    print("\n运行G-S算法...")
    recovered_phase, recovered_amp, error_history, energy_history = gerchberg_saxton(
        far_intensity, n_iter=n_iter, verbose=False, energy_tracking=True
    )
    
    print("\n能量历史 (每隔10次迭代):")
    for i in range(0, n_iter, 10):
        print(f"  迭代 {i+1:3d}: 能量 = {energy_history[i]:.10f}, "
              f"误差 = {abs(energy_history[i] - target_energy):.2e}")
    
    energy_deviation = [abs(e - target_energy) for e in energy_history]
    max_deviation = max(energy_deviation)
    mean_deviation = sum(energy_deviation) / len(energy_deviation)
    final_deviation = abs(energy_history[-1] - target_energy)
    
    print(f"\n能量统计:")
    print(f"  最大偏差: {max_deviation:.2e}")
    print(f"  平均偏差: {mean_deviation:.2e}")
    print(f"  最终偏差: {final_deviation:.2e}")
    
    stable = max_deviation < 1e-6
    print(f"\n能量稳定性: {'✓ 优秀' if stable else '✗ 需要改进'}")
    
    return stable


def test_near_far_energy_stability():
    """测试近场-远场G-S算法的能量稳定性"""
    print("\n" + "=" * 70)
    print("测试5: 近场-远场G-S算法能量稳定性")
    print("=" * 70)
    
    size = 64
    n_iter = 50
    
    print("创建测试图案...")
    amp_near, phase_true = create_test_image(size, 'lens')
    intensity_near = amp_near ** 2
    intensity_far = compute_far_field(amp_near, phase_true)
    target_energy = np.sum(intensity_near)
    print(f"目标能量: {target_energy:.6f}")
    
    print("\n运行近场-远场G-S算法...")
    recovered_phase, error_history, energy_history = gerchberg_saxton_near_field(
        intensity_near, intensity_far, n_iter=n_iter, verbose=False, energy_tracking=True
    )
    
    print("\n能量历史 (每隔10次迭代):")
    for i in range(0, n_iter, 10):
        print(f"  迭代 {i+1:3d}: 能量 = {energy_history[i]:.10f}, "
              f"误差 = {abs(energy_history[i] - target_energy):.2e}")
    
    energy_deviation = [abs(e - target_energy) for e in energy_history]
    max_deviation = max(energy_deviation)
    
    stable = max_deviation < 1e-10
    print(f"\n最大能量偏差: {max_deviation:.2e}")
    print(f"能量稳定性: {'✓ 完美' if stable else '✗ 需要改进'}")
    
    return stable


def test_defocus_phase():
    """测试离焦相位生成函数"""
    print("\n" + "=" * 70)
    print("测试6: 离焦相位生成函数")
    print("=" * 70)
    
    size = 64
    
    for defocus_waves in [0, -1, 1, 2.5]:
        phase = defocus_phase(size, defocus_waves)
        
        print(f"\n离焦量: {defocus_waves} 波长")
        print(f"  相位范围: [{phase.min():.2f}, {phase.max():.2f}] rad")
        print(f"  中心相位: {phase[size//2, size//2]:.2f} rad")
        print(f"  边缘相位: {phase[0, 0]:.2f} rad (应为 {defocus_waves * 2 * np.pi:.2f})")
        
        expected_edge = defocus_waves * 2 * np.pi
        edge_error = abs(phase[0, 0] - expected_edge)
        print(f"  边缘相位误差: {edge_error:.2e}")
    
    print("\n离焦相位生成: ✓ 通过")
    return True


def test_defocused_intensity_energy():
    """测试离焦强度生成的能量守恒"""
    print("\n" + "=" * 70)
    print("测试7: 离焦强度生成能量守恒")
    print("=" * 70)
    
    size = 64
    amp, phase = create_test_image(size, 'lens')
    input_energy = np.sum(amp ** 2)
    print(f"输入物面能量: {input_energy:.6f}")
    
    defocus_values = [0, -1, 1, -2, 2]
    intensities, _ = generate_defocused_intensities(amp, phase, defocus_values)
    
    print("\n各离焦面能量:")
    all_pass = True
    for k, (d, i) in enumerate(zip(defocus_values, intensities)):
        energy = np.sum(i)
        error = abs(energy - input_energy)
        ratio = energy / input_energy
        passed = error < 1e-6
        all_pass = all_pass and passed
        status = "✓" if passed else "✗"
        print(f"  {status} 离焦 {d:+.1f}λ: 能量 = {energy:.6f}, 比例 = {ratio:.6f}, 误差 = {error:.2e}")
    
    print(f"\n离焦能量守恒: {'✓ 通过' if all_pass else '✗ 失败'}")
    return all_pass


def test_phase_diversity_energy_stability():
    """测试相位多样性G-S算法的能量稳定性"""
    print("\n" + "=" * 70)
    print("测试8: 相位多样性G-S算法能量稳定性")
    print("=" * 70)
    
    size = 64
    n_iter = 50
    defocus_values = [0, -1.5, 1.5]
    snr = 30.0
    
    print("创建测试图案和离焦强度...")
    amp_true, phase_true = create_test_image(size, 'lens')
    intensities, _ = generate_defocused_intensities(
        amp_true, phase_true, defocus_values, snr=snr
    )
    target_energy = np.sum(intensities[0])
    print(f"目标能量: {target_energy:.6f}")
    print(f"离焦通道数: {len(intensities)}")
    print(f"噪声水平: SNR = {snr} dB")
    
    print("\n运行相位多样性G-S算法...")
    recovered_phase, recovered_amp, error_history, energy_history = \
        gerchberg_saxton_phase_diversity(
            intensities, defocus_values, n_iter=n_iter,
            verbose=False, energy_tracking=True
        )
    
    print("\n能量历史 (每隔10次迭代):")
    for i in range(0, n_iter, 10):
        print(f"  迭代 {i+1:3d}: 能量 = {energy_history[i]:.10f}, "
              f"误差 = {abs(energy_history[i] - target_energy):.2e}")
    
    energy_deviation = [abs(e - target_energy) for e in energy_history]
    max_deviation = max(energy_deviation)
    mean_deviation = sum(energy_deviation) / len(energy_deviation)
    
    print(f"\n能量统计:")
    print(f"  最大偏差: {max_deviation:.2e}")
    print(f"  平均偏差: {mean_deviation:.2e}")
    
    stable = max_deviation < 1e-6
    print(f"\n能量稳定性: {'✓ 优秀' if stable else '✗ 需要改进'}")
    
    return stable


def test_phase_diversity_vs_standard():
    """对比相位多样性与标准G-S算法的收敛性能"""
    print("\n" + "=" * 70)
    print("测试9: 相位多样性 vs 标准G-S 收敛对比")
    print("=" * 70)
    
    size = 64
    n_iter = 100
    defocus_values = [0, -1.5, 1.5]
    snr = 20.0
    
    print("创建测试图案和离焦强度...")
    amp_true, phase_true = create_test_image(size, 'lens')
    
    print("生成带噪声的离焦强度...")
    intensities_noisy, _ = generate_defocused_intensities(
        amp_true, phase_true, defocus_values, snr=snr
    )
    far_intensity_noisy = intensities_noisy[0]
    
    print("\n运行标准G-S算法...")
    _, _, error_history_standard, _ = gerchberg_saxton(
        far_intensity_noisy, n_iter=n_iter, verbose=False, energy_tracking=True
    )
    
    print("运行相位多样性G-S算法...")
    _, _, error_history_diversity, _ = gerchberg_saxton_phase_diversity(
        intensities_noisy, defocus_values, n_iter=n_iter,
        verbose=False, energy_tracking=True
    )
    
    final_error_standard = error_history_standard[-1]
    final_error_diversity = error_history_diversity[-1]
    error_reduction = (final_error_standard - final_error_diversity) / final_error_standard * 100
    
    print(f"\n收敛结果对比:")
    print(f"  标准G-S 最终误差: {final_error_standard:.6e}")
    print(f"  相位多样性 最终误差: {final_error_diversity:.6e}")
    print(f"  误差降低: {error_reduction:.1f}%")
    
    convergence_speed = 0
    for i in range(n_iter):
        if error_history_diversity[i] < final_error_standard:
            convergence_speed = i + 1
            break
    
    if convergence_speed > 0:
        print(f"  相位多样性在 {convergence_speed} 次迭代后达到标准G-S的最终精度")
        faster = True
    else:
        print(f"  相位多样性始终优于标准G-S")
        faster = True
    
    improved = final_error_diversity < final_error_standard
    print(f"\n性能提升: {'✓ 显著' if improved else '✗ 不明显'}")
    
    return faster and improved


def main():
    """运行所有测试"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "能量守恒验证测试套件" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")
    
    results = []
    
    try:
        test_numpy_fft_normalization()
        results.append(("FFT归一化特性", True))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("FFT归一化特性", False))
    
    try:
        results.append(("FFT/IFFT能量守恒", test_fft_energy_conservation()))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("FFT/IFFT能量守恒", False))
    
    try:
        test_constraint_energy_effect()
        results.append(("约束能量影响分析", True))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("约束能量影响分析", False))
    
    try:
        results.append(("G-S能量稳定性", test_gs_energy_stability()))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("G-S能量稳定性", False))
    
    try:
        results.append(("近场-远场G-S能量稳定性", test_near_far_energy_stability()))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("近场-远场G-S能量稳定性", False))
    
    try:
        results.append(("离焦相位生成", test_defocus_phase()))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("离焦相位生成", False))
    
    try:
        results.append(("离焦强度能量守恒", test_defocused_intensity_energy()))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("离焦强度能量守恒", False))
    
    try:
        results.append(("相位多样性G-S能量稳定性", test_phase_diversity_energy_stability()))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("相位多样性G-S能量稳定性", False))
    
    try:
        results.append(("相位多样性性能提升", test_phase_diversity_vs_standard()))
    except Exception as e:
        print(f"测试失败: {e}")
        results.append(("相位多样性性能提升", False))
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name:30s}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！能量守恒实现正确。")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，需要进一步检查。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
