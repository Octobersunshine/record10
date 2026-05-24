import numpy as np
import matplotlib.pyplot as plt
from propagation import propagate
from gs_algorithm import HybridInputOutput


def generate_test_object(size=128, noise_level=0.0):
    """
    生成测试物体（复振幅分布）
    """
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    X, Y = np.meshgrid(x, y)
    
    r1 = np.sqrt((X - 15)**2 + (Y - 10)**2)
    r2 = np.sqrt((X + 15)**2 + (Y + 10)**2)
    amplitude = np.exp(-r1**2 / 50) + np.exp(-r2**2 / 50)
    
    phase = 0.5 * np.sin(X / 10) * np.sin(Y / 10)
    
    if noise_level > 0:
        amplitude += np.random.randn(*amplitude.shape) * noise_level
    
    return amplitude * np.exp(1j * phase)


def check_pytorch():
    """
    检查PyTorch是否可用
    """
    try:
        import torch
        print(f"PyTorch 可用，版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"GPU 可用: {torch.cuda.get_device_name(0)}")
            return True, 'cuda'
        else:
            print("GPU 不可用，使用CPU")
            return True, 'cpu'
    except ImportError:
        print("PyTorch 未安装，深度先验方法将不可用")
        print("安装命令: pip install torch torchvision")
        return False, None


def example_deep_prior_basic():
    """
    示例1: 基础深度先验相位恢复
    """
    print("\n" + "=" * 60)
    print("示例1: 深度先验相位恢复基础示例")
    print("=" * 60)
    
    torch_available, device = check_pytorch()
    if not torch_available:
        print("跳过深度先验示例")
        return
    
    import torch
    from deep_prior import DeepPriorPhaseRetrieval
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distance = 0.1
    
    print("\n生成测试物体...")
    true_object = generate_test_object(size)
    
    print("生成衍射图案...")
    diffracted, _ = propagate(
        true_object, wavelength, pixel_size, distance, 'angular_spectrum'
    )
    measured_intensity = np.abs(diffracted)**2
    
    noise_level = 0.05
    measured_intensity_noisy = measured_intensity + noise_level * np.max(measured_intensity) * np.random.randn(*measured_intensity.shape)
    measured_intensity_noisy = np.maximum(measured_intensity_noisy, 0)
    
    print("\n运行深度先验相位恢复...")
    print(f"使用设备: {device}")
    
    try:
        dp = DeepPriorPhaseRetrieval(
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            network_type='small_unet',
            output_type='amp_phase',
            device=device
        )
        
        recon_dp = dp.reconstruct(
            measured_intensity_noisy,
            num_iterations=500,
            lr=0.01,
            print_interval=100
        )
        
        print(f"最终损失: {dp.get_errors()[-1]:.6e}")
    except Exception as e:
        print(f"深度先验运行出错: {e}")
        print("使用传统HIO算法作为替代...")
        support = np.ones((size, size), dtype=bool)
        center = size // 2
        support[center-40:center+40, center-40:center+40] = True
        
        hio = HybridInputOutput(
            wavelength, pixel_size, distance,
            support=support,
            phase_reference_type='support'
        )
        recon_dp = hio.reconstruct(measured_intensity_noisy, num_iterations=200, verbose=False)
    
    print("\n运行传统HIO算法进行对比...")
    support = np.ones((size, size), dtype=bool)
    center = size // 2
    support[center-40:center+40, center-40:center+40] = True
    
    hio = HybridInputOutput(
        wavelength, pixel_size, distance,
        support=support,
        phase_reference_type='support'
    )
    recon_hio = hio.reconstruct(measured_intensity_noisy, num_iterations=200, verbose=False)
    
    def calculate_psnr(img1, img2):
        mse = np.mean((np.abs(img1) - np.abs(img2))**2)
        if mse == 0:
            return float('inf')
        max_val = np.max(np.abs(img1))
        return 20 * np.log10(max_val / np.sqrt(mse))
    
    psnr_hio = calculate_psnr(true_object, recon_hio)
    psnr_dp = calculate_psnr(true_object, recon_dp)
    
    print(f"\nHIO 重建 PSNR: {psnr_hio:.2f} dB")
    print(f"深度先验重建 PSNR: {psnr_dp:.2f} dB")
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 4, 1)
    plt.imshow(np.abs(true_object), cmap='gray')
    plt.title('真实振幅')
    plt.colorbar()
    
    plt.subplot(2, 4, 2)
    plt.imshow(np.angle(true_object), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('真实相位')
    plt.colorbar()
    
    plt.subplot(2, 4, 3)
    plt.imshow(np.log10(measured_intensity_noisy + 1e-10), cmap='gray')
    plt.title(f'噪声衍射图案 (SNR={1/noise_level:.0f})')
    plt.colorbar()
    
    plt.subplot(2, 4, 4)
    plt.semilogy(hio.get_errors(), label='HIO')
    if hasattr(dp, 'get_errors'):
        plt.semilogy(dp.get_errors(), label='深度先验')
    plt.title('迭代误差')
    plt.xlabel('迭代次数')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 5)
    plt.imshow(np.abs(recon_hio), cmap='gray')
    plt.title(f'HIO 重建振幅\nPSNR={psnr_hio:.2f}dB')
    plt.colorbar()
    
    plt.subplot(2, 4, 6)
    plt.imshow(np.angle(recon_hio), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('HIO 重建相位')
    plt.colorbar()
    
    plt.subplot(2, 4, 7)
    plt.imshow(np.abs(recon_dp), cmap='gray')
    plt.title(f'深度先验重建振幅\nPSNR={psnr_dp:.2f}dB')
    plt.colorbar()
    
    plt.subplot(2, 4, 8)
    plt.imshow(np.angle(recon_dp), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.title('深度先验重建相位')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('deep_prior_basic.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 deep_prior_basic.png")


def example_network_comparison():
    """
    示例2: 不同网络结构对比
    """
    print("\n" + "=" * 60)
    print("示例2: 不同网络结构对比")
    print("=" * 60)
    
    torch_available, device = check_pytorch()
    if not torch_available:
        print("跳过深度先验示例")
        return
    
    from deep_prior import DeepPriorPhaseRetrieval
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distance = 0.1
    
    true_object = generate_test_object(size)
    diffracted, _ = propagate(
        true_object, wavelength, pixel_size, distance, 'angular_spectrum'
    )
    measured_intensity = np.abs(diffracted)**2
    
    noise_level = 0.1
    measured_intensity_noisy = measured_intensity + noise_level * np.max(measured_intensity) * np.random.randn(*measured_intensity.shape)
    measured_intensity_noisy = np.maximum(measured_intensity_noisy, 0)
    
    network_types = ['small_unet', 'resnet']
    results = {}
    errors = {}
    
    for net_type in network_types:
        print(f"\n测试 {net_type} 网络...")
        try:
            dp = DeepPriorPhaseRetrieval(
                wavelength=wavelength,
                pixel_size=pixel_size,
                distance=distance,
                network_type=net_type,
                device=device
            )
            
            recon = dp.reconstruct(
                measured_intensity_noisy,
                num_iterations=300,
                lr=0.01,
                print_interval=100,
                verbose=False
            )
            
            results[net_type] = recon
            errors[net_type] = dp.get_errors()
            print(f"  完成，最终误差: {errors[net_type][-1]:.6e}")
        except Exception as e:
            print(f"  出错: {e}")
    
    if len(results) > 0:
        plt.figure(figsize=(12, 5))
        
        n = len(results)
        for i, (name, recon) in enumerate(results.items()):
            plt.subplot(1, n + 1, i + 1)
            plt.imshow(np.abs(recon), cmap='gray')
            plt.title(f'{name} 振幅')
            plt.colorbar()
        
        plt.subplot(1, n + 1, n + 1)
        for name, err in errors.items():
            plt.semilogy(err, label=name)
        plt.title('迭代误差对比')
        plt.xlabel('迭代次数')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('network_comparison.png', dpi=150, bbox_inches='tight')
        print("\n结果已保存到 network_comparison.png")


def example_multi_distance_deep_prior():
    """
    示例3: 多距离深度先验相位恢复
    """
    print("\n" + "=" * 60)
    print("示例3: 多距离深度先验相位恢复")
    print("=" * 60)
    
    torch_available, device = check_pytorch()
    if not torch_available:
        print("跳过深度先验示例")
        return
    
    from deep_prior import DeepPriorMultiDistance
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distances = [0.08, 0.1, 0.12]
    
    true_object = generate_test_object(size)
    
    intensities = []
    for dist in distances:
        diffracted, _ = propagate(
            true_object, wavelength, pixel_size, dist, 'angular_spectrum'
        )
        intensity = np.abs(diffracted)**2
        noise = 0.05 * np.max(intensity) * np.random.randn(*intensity.shape)
        intensity = np.maximum(intensity + noise, 0)
        intensities.append(intensity)
    
    print(f"\n运行多距离深度先验 ({len(distances)} 个距离)...")
    try:
        dp_md = DeepPriorMultiDistance(
            wavelength=wavelength,
            pixel_size=pixel_size,
            distances=distances,
            network_type='small_unet',
            device=device
        )
        
        recon = dp_md.reconstruct(
            intensities,
            num_iterations=500,
            lr=0.01,
            print_interval=100
        )
        
        print(f"最终误差: {dp_md.get_errors()[-1]:.6e}")
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(true_object), cmap='gray')
        plt.title('真实振幅')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(recon), cmap='gray')
        plt.title('多距离深度先验重建')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.semilogy(dp_md.get_errors())
        plt.title('迭代误差')
        plt.xlabel('迭代次数')
        plt.ylabel('MSE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('multi_distance_deep_prior.png', dpi=150, bbox_inches='tight')
        print("\n结果已保存到 multi_distance_deep_prior.png")
    except Exception as e:
        print(f"运行出错: {e}")


def example_hybrid_prior():
    """
    示例4: 混合深度先验（传统算法+深度正则化）
    """
    print("\n" + "=" * 60)
    print("示例4: 混合深度先验相位恢复")
    print("=" * 60)
    
    torch_available, device = check_pytorch()
    if not torch_available:
        print("跳过深度先验示例")
        return
    
    from deep_prior import DeepPriorHybrid
    
    size = 128
    wavelength = 500e-9
    pixel_size = 10e-6
    distance = 0.1
    
    true_object = generate_test_object(size)
    diffracted, _ = propagate(
        true_object, wavelength, pixel_size, distance, 'angular_spectrum'
    )
    measured_intensity = np.abs(diffracted)**2
    
    noise_level = 0.1
    measured_intensity_noisy = measured_intensity + noise_level * np.max(measured_intensity) * np.random.randn(*measured_intensity.shape)
    measured_intensity_noisy = np.maximum(measured_intensity_noisy, 0)
    
    print("\n生成HIO初始估计...")
    support = np.ones((size, size), dtype=bool)
    center = size // 2
    support[center-40:center+40, center-40:center+40] = True
    
    hio = HybridInputOutput(
        wavelength, pixel_size, distance,
        support=support,
        phase_reference_type='support'
    )
    initial_guess = hio.reconstruct(measured_intensity_noisy, num_iterations=100, verbose=False)
    
    print("运行混合深度先验优化...")
    try:
        dp_hybrid = DeepPriorHybrid(
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            device=device
        )
        
        recon_hybrid = dp_hybrid.reconstruct(
            measured_intensity_noisy,
            initial_guess=initial_guess,
            num_iterations=300,
            lr=0.001,
            alpha=0.8,
            print_interval=50
        )
        
        print(f"最终误差: {dp_hybrid.get_errors()[-1]:.6e}")
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(np.abs(true_object), cmap='gray')
        plt.title('真实振幅')
        plt.colorbar()
        
        plt.subplot(1, 4, 2)
        plt.imshow(np.abs(initial_guess), cmap='gray')
        plt.title('HIO初始估计')
        plt.colorbar()
        
        plt.subplot(1, 4, 3)
        plt.imshow(np.abs(recon_hybrid), cmap='gray')
        plt.title('混合深度先验优化')
        plt.colorbar()
        
        plt.subplot(1, 4, 4)
        plt.semilogy(dp_hybrid.get_errors())
        plt.title('迭代误差')
        plt.xlabel('迭代次数')
        plt.ylabel('MSE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('hybrid_prior.png', dpi=150, bbox_inches='tight')
        print("\n结果已保存到 hybrid_prior.png")
    except Exception as e:
        print(f"运行出错: {e}")


if __name__ == '__main__':
    np.random.seed(42)
    
    example_deep_prior_basic()
    example_network_comparison()
    example_multi_distance_deep_prior()
    example_hybrid_prior()
    
    print("\n" + "=" * 60)
    print("所有深度先验示例运行完成！")
    print("=" * 60)
