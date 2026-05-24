import numpy as np
import matplotlib.pyplot as plt
from propagation import propagate
from gs_algorithm import GerchbergSaxton, HybridInputOutput
from dm_algorithm import DifferenceMap, RelaxedAveragedAlternatingProjections
from multi_distance import MultiDistanceGS, MultiDistanceHIO, MultiAngleGS, GeneralizedProjection


def generate_test_object(size=128, object_type='double_hole'):
    """
    生成测试物体（复振幅分布）
    """
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    X, Y = np.meshgrid(x, y)
    
    if object_type == 'double_hole':
        r1 = np.sqrt((X - 15)**2 + (Y - 10)**2)
        r2 = np.sqrt((X + 15)**2 + (Y + 10)**2)
        amplitude = np.exp(-r1**2 / 50) + np.exp(-r2**2 / 50)
        phase = 0.5 * np.sin(X / 10) * np.sin(Y / 10)
    elif object_type == 'letter':
        amplitude = np.zeros((size, size))
        amplitude[size//2-20:size//2+20, size//2-5:size//2+5] = 1
        amplitude[size//2-5:size//2+5, size//2-20:size//2+20] = 1
        phase = np.zeros((size, size))
    elif object_type == 'gaussian':
        r = np.sqrt(X**2 + Y**2)
        amplitude = np.exp(-r**2 / 500)
        phase = 0.3 * r / 20
    else:
        amplitude = np.abs(np.random.rand(size, size))
        phase = np.random.rand(size, size) * 2 * np.pi
    
    return amplitude * np.exp(1j * phase)


def generate_diffraction_patterns(object_field, wavelength, pixel_size, distances, 
                                  method='angular_spectrum', noise_level=0.0):
    """
    生成多个衍射强度图
    """
    intensities = []
    
    for distance in distances:
        diffracted, _ = propagate(
            object_field, wavelength, pixel_size, distance, method
        )
        intensity = np.abs(diffracted)**2
        
        if noise_level > 0:
            noise = np.random.randn(*intensity.shape) * noise_level * np.max(intensity)
            intensity = np.maximum(intensity + noise, 0)
        
        intensities.append(intensity)
    
    return intensities


def example_single_distance():
    """
    示例1: 单距离相位恢复（比较不同算法）
    """
    print("=" * 60)
    print("示例1: 单距离相位恢复")
    print("=" * 60)
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distance = 0.1
    
    true_object = generate_test_object(size, 'double_hole')
    
    diffracted, _ = propagate(
        true_object, wavelength, pixel_size, distance, 'angular_spectrum'
    )
    measured_intensity = np.abs(diffracted)**2
    
    support = np.ones((size, size), dtype=bool)
    center = size // 2
    support[center-40:center+40, center-40:center+40] = True
    
    algorithms = {
        'GS': GerchbergSaxton(wavelength, pixel_size, distance, support=support),
        'HIO': HybridInputOutput(wavelength, pixel_size, distance, support=support, beta=0.9),
        'DM': DifferenceMap(wavelength, pixel_size, distance, support=support, beta=0.9),
        'RAAR': RelaxedAveragedAlternatingProjections(wavelength, pixel_size, distance, support=support, beta=0.8)
    }
    
    results = {}
    errors = {}
    
    for name, algorithm in algorithms.items():
        print(f"\n运行 {name} 算法...")
        reconstructed = algorithm.reconstruct(
            measured_intensity, num_iterations=100, verbose=False
        )
        results[name] = reconstructed
        errors[name] = algorithm.get_errors()
        print(f"  最终误差: {errors[name][-1]:.6e}")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(np.abs(true_object), cmap='gray')
    plt.title('真实振幅')
    plt.colorbar()
    
    plt.subplot(2, 4, 2)
    plt.imshow(np.angle(true_object), cmap='hsv')
    plt.title('真实相位')
    plt.colorbar()
    
    plt.subplot(2, 4, 3)
    plt.imshow(np.log10(measured_intensity + 1e-10), cmap='gray')
    plt.title('衍射强度（对数）')
    plt.colorbar()
    
    plt.subplot(2, 4, 4)
    for name, err in errors.items():
        plt.semilogy(err, label=name)
    plt.title('迭代误差')
    plt.xlabel('迭代次数')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    for i, (name, recon) in enumerate(results.items()):
        plt.subplot(2, 4, 5 + i)
        plt.imshow(np.abs(recon), cmap='gray')
        plt.title(f'{name} 重建振幅')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('example_single_distance.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 example_single_distance.png")


def example_multi_distance():
    """
    示例2: 多距离相位恢复
    """
    print("\n" + "=" * 60)
    print("示例2: 多距离相位恢复")
    print("=" * 60)
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distances = [0.05, 0.1, 0.15]
    
    true_object = generate_test_object(size, 'double_hole')
    
    intensities = generate_diffraction_patterns(
        true_object, wavelength, pixel_size, distances,
        method='angular_spectrum', noise_level=0.01
    )
    
    support = np.ones((size, size), dtype=bool)
    center = size // 2
    support[center-40:center+40, center-40:center+40] = True
    
    print("\n运行多距离 GS 算法...")
    md_gs = MultiDistanceGS(
        wavelength, pixel_size, distances,
        support=support
    )
    recon_gs = md_gs.reconstruct(intensities, num_iterations=100, verbose=False)
    print(f"  最终误差: {md_gs.get_errors()[-1]:.6e}")
    
    print("\n运行多距离 HIO 算法...")
    md_hio = MultiDistanceHIO(
        wavelength, pixel_size, distances,
        support=support, beta=0.9
    )
    recon_hio = md_hio.reconstruct(intensities, num_iterations=100, verbose=False)
    print(f"  最终误差: {md_hio.get_errors()[-1]:.6e}")
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 4, 1)
    plt.imshow(np.abs(true_object), cmap='gray')
    plt.title('真实振幅')
    plt.colorbar()
    
    plt.subplot(2, 4, 2)
    plt.imshow(np.angle(true_object), cmap='hsv')
    plt.title('真实相位')
    plt.colorbar()
    
    for i, (dist, intensity) in enumerate(zip(distances, intensities)):
        plt.subplot(2, 4, 3 + i)
        plt.imshow(np.log10(intensity + 1e-10), cmap='gray')
        plt.title(f'距离={dist*100:.0f}cm 强度')
        plt.colorbar()
    
    plt.subplot(2, 4, 6)
    plt.semilogy(md_gs.get_errors(), label='多距离 GS')
    plt.semilogy(md_hio.get_errors(), label='多距离 HIO')
    plt.title('迭代误差')
    plt.xlabel('迭代次数')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 7)
    plt.imshow(np.abs(recon_gs), cmap='gray')
    plt.title('多距离 GS 重建')
    plt.colorbar()
    
    plt.subplot(2, 4, 8)
    plt.imshow(np.abs(recon_hio), cmap='gray')
    plt.title('多距离 HIO 重建')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('example_multi_distance.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 example_multi_distance.png")


def example_multi_angle():
    """
    示例3: 多角度相位恢复
    """
    print("\n" + "=" * 60)
    print("示例3: 多角度相位恢复")
    print("=" * 60)
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distance = 0.1
    
    k = 2 * np.pi / wavelength
    angles = [(0, 0), (k*0.001, 0), (0, k*0.001)]
    
    true_object = generate_test_object(size, 'gaussian')
    
    intensities = []
    for angle in angles:
        kx, ky = angle
        x = np.linspace(-size/2, size/2, size) * pixel_size
        y = np.linspace(-size/2, size/2, size) * pixel_size
        X, Y = np.meshgrid(x, y)
        phase_factor = np.exp(1j * (kx * X + ky * Y))
        illuminated = true_object * phase_factor
        
        diffracted, _ = propagate(
            illuminated, wavelength, pixel_size, distance, 'angular_spectrum'
        )
        intensity = np.abs(diffracted)**2
        
        noise = np.random.randn(*intensity.shape) * 0.005 * np.max(intensity)
        intensity = np.maximum(intensity + noise, 0)
        
        intensities.append(intensity)
    
    support = np.ones((size, size), dtype=bool)
    center = size // 2
    support[center-50:center+50, center-50:center+50] = True
    
    print("\n运行多角度 GS 算法...")
    ma_gs = MultiAngleGS(
        wavelength, pixel_size, distance,
        angles=angles, support=support
    )
    recon = ma_gs.reconstruct(intensities, num_iterations=100, verbose=False)
    print(f"  最终误差: {ma_gs.get_errors()[-1]:.6e}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(np.abs(true_object), cmap='gray')
    plt.title('真实振幅')
    plt.colorbar()
    
    for i, (angle, intensity) in enumerate(zip(angles, intensities)):
        plt.subplot(1, 4, 2 + i)
        plt.imshow(np.log10(intensity + 1e-10), cmap='gray')
        kx, ky = angle
        plt.title(f'角度=({kx/k:.3f}k, {ky/k:.3f}k)')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('example_multi_angle.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 example_multi_angle.png")


def example_generalized():
    """
    示例4: 广义投影算法（混合多距离和多角度）
    """
    print("\n" + "=" * 60)
    print("示例4: 广义投影算法（混合约束）")
    print("=" * 60)
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    
    true_object = generate_test_object(size, 'double_hole')
    
    gp = GeneralizedProjection(
        wavelength, pixel_size,
        method='angular_spectrum',
        support=np.ones((size, size), dtype=bool)
    )
    
    distances = [0.08, 0.12]
    for dist in distances:
        diffracted, _ = propagate(
            true_object, wavelength, pixel_size, dist, 'angular_spectrum'
        )
        intensity = np.abs(diffracted)**2
        gp.add_distance_constraint(dist, intensity, weight=1.0)
    
    k = 2 * np.pi / wavelength
    angles = [(k*0.0005, k*0.0005)]
    for angle in angles:
        kx, ky = angle
        x = np.linspace(-size/2, size/2, size) * pixel_size
        y = np.linspace(-size/2, size/2, size) * pixel_size
        X, Y = np.meshgrid(x, y)
        phase_factor = np.exp(1j * (kx * X + ky * Y))
        illuminated = true_object * phase_factor
        
        diffracted, _ = propagate(
            illuminated, wavelength, pixel_size, 0.1, 'angular_spectrum'
        )
        intensity = np.abs(diffracted)**2
        gp.add_angle_constraint(0.1, angle, intensity, weight=1.0)
    
    print("\n运行广义投影算法 (HIO模式)...")
    recon = gp.reconstruct(num_iterations=100, verbose=False, mode='HIO')
    print(f"  最终误差: {gp.get_errors()[-1]:.6e}")
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(true_object), cmap='gray')
    plt.title('真实振幅')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(recon), cmap='gray')
    plt.title('重建振幅')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.semilogy(gp.get_errors())
    plt.title('迭代误差')
    plt.xlabel('迭代次数')
    plt.ylabel('MSE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('example_generalized.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 example_generalized.png")


if __name__ == '__main__':
    np.random.seed(42)
    
    example_single_distance()
    example_multi_distance()
    example_multi_angle()
    example_generalized()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
