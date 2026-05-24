import numpy as np
import matplotlib.pyplot as plt
from propagation import propagate
from gs_algorithm import GerchbergSaxton, HybridInputOutput
from phase_correction import PhaseReference, PhaseCorrector


def generate_test_object(size=128):
    """
    生成测试物体（带有相位信息）
    """
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    X, Y = np.meshgrid(x, y)
    
    r1 = np.sqrt((X - 15)**2 + (Y - 10)**2)
    r2 = np.sqrt((X + 15)**2 + (Y + 10)**2)
    amplitude = np.exp(-r1**2 / 50) + np.exp(-r2**2 / 50)
    
    phase = 0.5 * np.sin(X / 10) * np.sin(Y / 10)
    
    return amplitude * np.exp(1j * phase)


def example_phase_drift_comparison():
    """
    示例: 比较有无相位校正的重建结果
    """
    print("=" * 60)
    print("相位校正对比示例")
    print("=" * 60)
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distance = 0.1
    
    true_object = generate_test_object(size)
    
    diffracted, _ = propagate(
        true_object, wavelength, pixel_size, distance, 'angular_spectrum'
    )
    measured_intensity = np.abs(diffracted)**2
    
    support = np.ones((size, size), dtype=bool)
    center = size // 2
    support[center-40:center+40, center-40:center+40] = True
    
    np.random.seed(42)
    
    print("\n运行无相位校正的 HIO 算法...")
    hio_no_correction = HybridInputOutput(
        wavelength, pixel_size, distance,
        support=support, beta=0.9,
        phase_reference_type=None
    )
    recon_no_correction = hio_no_correction.reconstruct(
        measured_intensity, num_iterations=200, verbose=False
    )
    print(f"  最终误差: {hio_no_correction.get_errors()[-1]:.6e}")
    
    print("\n运行带相位校正的 HIO 算法（支撑域相位基准）...")
    hio_with_correction = HybridInputOutput(
        wavelength, pixel_size, distance,
        support=support, beta=0.9,
        phase_reference_type='support'
    )
    recon_with_correction = hio_with_correction.reconstruct(
        measured_intensity, num_iterations=200, verbose=False
    )
    print(f"  最终误差: {hio_with_correction.get_errors()[-1]:.6e}")
    
    print("\n运行带相位校正的 HIO 算法（中心点相位基准）...")
    hio_point_correction = HybridInputOutput(
        wavelength, pixel_size, distance,
        support=support, beta=0.9,
        phase_reference_type='point'
    )
    recon_point_correction = hio_point_correction.reconstruct(
        measured_intensity, num_iterations=200, verbose=False
    )
    print(f"  最终误差: {hio_point_correction.get_errors()[-1]:.6e}")
    
    avg_phase_no_corr = np.mean(np.angle(recon_no_correction[support]))
    avg_phase_with_corr = np.mean(np.angle(recon_with_correction[support]))
    avg_phase_point_corr = np.mean(np.angle(recon_point_correction[support]))
    avg_phase_true = np.mean(np.angle(true_object[support]))
    
    print(f"\n真实物体支撑域平均相位: {avg_phase_true:.4f} rad")
    print(f"无校正重建平均相位: {avg_phase_no_corr:.4f} rad (偏移: {avg_phase_no_corr - avg_phase_true:.4f} rad)")
    print(f"支撑域校正平均相位: {avg_phase_with_corr:.4f} rad (偏移: {avg_phase_with_corr - avg_phase_true:.4f} rad)")
    print(f"点校正重建平均相位: {avg_phase_point_corr:.4f} rad (偏移: {avg_phase_point_corr - avg_phase_true:.4f} rad)")
    
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 5, 1)
    plt.imshow(np.abs(true_object), cmap='gray')
    plt.title('真实振幅')
    plt.colorbar()
    
    plt.subplot(2, 5, 2)
    plt.imshow(np.angle(true_object), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('真实相位')
    plt.colorbar()
    
    plt.subplot(2, 5, 3)
    plt.imshow(np.angle(recon_no_correction), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('无校正相位')
    plt.colorbar()
    
    plt.subplot(2, 5, 4)
    plt.imshow(np.angle(recon_with_correction), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('支撑域校正相位')
    plt.colorbar()
    
    plt.subplot(2, 5, 5)
    plt.imshow(np.angle(recon_point_correction), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('点校正相位')
    plt.colorbar()
    
    plt.subplot(2, 5, 6)
    plt.semilogy(hio_no_correction.get_errors(), label='无校正')
    plt.semilogy(hio_with_correction.get_errors(), label='支撑域校正')
    plt.semilogy(hio_point_correction.get_errors(), label='点校正')
    plt.title('迭代误差')
    plt.xlabel('迭代次数')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    phase_diff_no_corr = np.angle(recon_no_correction * np.conj(true_object))
    phase_diff_with_corr = np.angle(recon_with_correction * np.conj(true_object))
    phase_diff_point_corr = np.angle(recon_point_correction * np.conj(true_object))
    
    plt.subplot(2, 5, 7)
    plt.imshow(phase_diff_no_corr, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('无校正相位差')
    plt.colorbar()
    
    plt.subplot(2, 5, 8)
    plt.imshow(phase_diff_with_corr, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('支撑域校正相位差')
    plt.colorbar()
    
    plt.subplot(2, 5, 9)
    plt.imshow(phase_diff_point_corr, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('点校正相位差')
    plt.colorbar()
    
    plt.subplot(2, 5, 10)
    methods = ['无校正', '支撑域校正', '点校正']
    phase_diffs = [np.mean(np.abs(phase_diff_no_corr[support])),
                   np.mean(np.abs(phase_diff_with_corr[support])),
                   np.mean(np.abs(phase_diff_point_corr[support]))]
    plt.bar(methods, phase_diffs, color=['red', 'green', 'blue'])
    plt.title('支撑域内平均绝对相位差')
    plt.ylabel('rad')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('phase_correction_comparison.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 phase_correction_comparison.png")


def example_phase_reference_types():
    """
    示例: 不同相位基准类型的效果
    """
    print("\n" + "=" * 60)
    print("不同相位基准类型对比")
    print("=" * 60)
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distance = 0.1
    
    true_object = generate_test_object(size)
    
    diffracted, _ = propagate(
        true_object, wavelength, pixel_size, distance, 'angular_spectrum'
    )
    measured_intensity = np.abs(diffracted)**2
    
    support = np.ones((size, size), dtype=bool)
    center = size // 2
    support[center-40:center+40, center-40:center+40] = True
    
    reference_types = [None, 'point', 'region', 'support', 'amplitude']
    labels = ['无校正', '单点基准', '区域基准', '支撑域基准', '振幅加权基准']
    
    results = {}
    errors = {}
    
    for ref_type, label in zip(reference_types, labels):
        print(f"\n运行 {label}...")
        np.random.seed(42)
        
        gs = GerchbergSaxton(
            wavelength, pixel_size, distance,
            support=support,
            phase_reference_type=ref_type
        )
        recon = gs.reconstruct(
            measured_intensity, num_iterations=150, verbose=False
        )
        results[label] = recon
        errors[label] = gs.get_errors()
        print(f"  最终误差: {errors[label][-1]:.6e}")
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.angle(true_object), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('真实相位')
    plt.colorbar()
    
    for i, (label, recon) in enumerate(results.items()):
        plt.subplot(2, 3, i + 2)
        plt.imshow(np.angle(recon), cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.title(f'{label}相位')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('phase_reference_types.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 phase_reference_types.png")


def example_custom_phase_reference():
    """
    示例: 自定义相位基准
    """
    print("\n" + "=" * 60)
    print("自定义相位基准示例")
    print("=" * 60)
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distance = 0.1
    
    true_object = generate_test_object(size)
    
    diffracted, _ = propagate(
        true_object, wavelength, pixel_size, distance, 'angular_spectrum'
    )
    measured_intensity = np.abs(diffracted)**2
    
    support = np.ones((size, size), dtype=bool)
    center = size // 2
    support[center-40:center+40, center-40:center+40] = True
    
    zero_phase_mask = np.zeros((size, size), dtype=bool)
    zero_phase_mask[center-10:center+10, center-10:center+10] = True
    
    print("\n创建自定义相位基准（中心区域零相位约束）...")
    custom_reference = PhaseReference(
        'zero_phase',
        mask=zero_phase_mask
    )
    
    custom_corrector = PhaseCorrector(
        global_phase=True,
        phase_tilt=False,
        reference=custom_reference
    )
    
    np.random.seed(42)
    gs_custom = GerchbergSaxton(
        wavelength, pixel_size, distance,
        support=support,
        phase_corrector=custom_corrector
    )
    recon_custom = gs_custom.reconstruct(
        measured_intensity, num_iterations=150, verbose=False
    )
    print(f"  最终误差: {gs_custom.get_errors()[-1]:.6e}")
    
    print("\n创建带相位倾斜移除的相位校正器...")
    corrector_with_tilt = PhaseCorrector(
        global_phase=True,
        phase_tilt=True,
        reference=PhaseReference('support', support=support, phase=0.0)
    )
    
    np.random.seed(42)
    gs_tilt = GerchbergSaxton(
        wavelength, pixel_size, distance,
        support=support,
        phase_corrector=corrector_with_tilt
    )
    recon_tilt = gs_tilt.reconstruct(
        measured_intensity, num_iterations=150, verbose=False
    )
    print(f"  最终误差: {gs_tilt.get_errors()[-1]:.6e}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.angle(true_object), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('真实相位')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.angle(recon_custom), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('零相位约束')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.angle(recon_tilt), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('含倾斜移除')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('custom_phase_reference.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 custom_phase_reference.png")


if __name__ == '__main__':
    np.random.seed(42)
    
    example_phase_drift_comparison()
    example_phase_reference_types()
    example_custom_phase_reference()
    
    print("\n" + "=" * 60)
    print("所有相位校正示例运行完成！")
    print("=" * 60)
