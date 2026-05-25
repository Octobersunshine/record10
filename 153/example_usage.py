"""
Richardson-Lucy反卷积使用示例
包含单帧和多帧反卷积演示
"""

import numpy as np
from richardson_lucy import (
    richardson_lucy,
    richardson_lucy_with_regularization,
    generate_gaussian_psf,
    generate_atmospheric_psf,
    blur_image,
    compute_psnr
)


def example_4_lucky_imaging():
    """
    示例4: 幸运成像技术
    """
    from multi_frame_deconvolution import (
        generate_multiframe_psfs,
        generate_multiframe_images,
        lucky_imaging_select,
        lucky_imaging_reconstruct
    )
    
    print("=" * 60)
    print("示例4: 幸运成像 (Lucky Imaging)")
    print("=" * 60)
    
    size = 80
    num_frames = 30
    
    clean = np.zeros((size, size))
    for _ in range(10):
        x, y = np.random.randint(10, size - 10, 2)
        clean[y, x] = np.random.uniform(0.5, 1.0)
    
    psfs = generate_multiframe_psfs(num_frames=num_frames, psf_size=25, seeing=3.0)
    images = generate_multiframe_images(clean, psfs, snr=40)
    
    avg_image = images.mean(axis=0)
    psnr_avg = compute_psnr(clean, avg_image)
    
    selected, scores = lucky_imaging_select(images, selection_fraction=0.2, metric='strehl')
    lucky_recon = lucky_imaging_reconstruct(images, selected, method='mean')
    psnr_lucky = compute_psnr(clean, lucky_recon)
    
    print(f"帧数: {num_frames}")
    print(f"选中帧数: {len(selected)} ({len(selected)/num_frames*100:.0f}%)")
    print(f"简单平均PSNR: {psnr_avg:.2f} dB")
    print(f"幸运成像PSNR: {psnr_lucky:.2f} dB")
    print(f"改善: +{psnr_lucky - psnr_avg:.2f} dB")
    print()


def example_5_multiframe_rl():
    """
    示例5: 多帧RL反卷积 (已知PSF)
    """
    from multi_frame_deconvolution import (
        generate_multiframe_psfs,
        generate_multiframe_images,
        multiframe_richardson_lucy
    )
    
    print("=" * 60)
    print("示例5: 多帧RL反卷积 (已知PSF)")
    print("=" * 60)
    
    size = 80
    num_frames = 20
    
    clean = np.zeros((size, size))
    clean[40, 40] = 1.0
    clean[25, 55] = 0.8
    clean[55, 25] = 0.7
    
    psfs = generate_multiframe_psfs(num_frames=num_frames, psf_size=25, seeing=2.5)
    images = generate_multiframe_images(clean, psfs, snr=35)
    
    avg_psf = psfs.mean(axis=0)
    avg_image = images.mean(axis=0)
    
    single_frame_rl = richardson_lucy(images[0], avg_psf, iterations=50)
    multiframe_rl = multiframe_richardson_lucy(images, avg_psf, iterations=50)
    
    print(f"帧数: {num_frames}")
    print(f"单帧RL PSNR:   {compute_psnr(clean, single_frame_rl):.2f} dB")
    print(f"简单平均PSNR:  {compute_psnr(clean, avg_image):.2f} dB")
    print(f"多帧RL PSNR:   {compute_psnr(clean, multiframe_rl):.2f} dB")
    print(f"改善 (vs 单帧): +{compute_psnr(clean, multiframe_rl) - compute_psnr(clean, single_frame_rl):.2f} dB")
    print()


def example_6_blind_deconvolution():
    """
    示例6: 多帧盲反卷积 (未知PSF)
    """
    from multi_frame_deconvolution import (
        generate_multiframe_psfs,
        generate_multiframe_images,
        multiframe_blind_deconvolution
    )
    
    print("=" * 60)
    print("示例6: 多帧盲反卷积 (同时估计图像和PSF)")
    print("=" * 60)
    
    size = 64
    num_frames = 15
    
    clean = np.zeros((size, size))
    clean[32, 32] = 1.0
    
    psfs = generate_multiframe_psfs(num_frames=num_frames, psf_size=21, seeing=2.5)
    images = generate_multiframe_images(clean, psfs, snr=40)
    
    avg_image = images.mean(axis=0)
    
    print("执行盲反卷积... (约30-40次迭代)")
    blind_result, blind_psf = multiframe_blind_deconvolution(
        images, iterations=20, image_iter=3, psf_iter=2, psf_size=21
    )
    
    print(f"简单平均PSNR: {compute_psnr(clean, avg_image):.2f} dB")
    print(f"盲反卷积PSNR: {compute_psnr(clean, blind_result):.2f} dB")
    print(f"改善: +{compute_psnr(clean, blind_result) - compute_psnr(clean, avg_image):.2f} dB")
    print()


def example_7_noise_suppression():
    """
    示例7: 噪声抑制对比 (早停 vs TV正则化)
    """
    from rl_regularized import (
        richardson_lucy_early_stop,
        richardson_lucy_tv,
        richardson_lucy_adaptive_tv
    )
    from richardson_lucy import richardson_lucy
    
    print("=" * 60)
    print("示例7: 噪声抑制对比演示")
    print("=" * 60)
    
    size = 80
    
    clean = np.zeros((size, size))
    for _ in range(8):
        x, y = np.random.randint(15, size - 15, 2)
        clean[y, x] = np.random.uniform(0.6, 1.0)
    
    psf = generate_atmospheric_psf(size=25, seeing=2.5)
    blurred = blur_image(clean, psf, add_noise_flag=True, snr=25)
    
    print(f"图像尺寸: {size}x{size}, SNR=25 (高噪声)")
    print(f"模糊图像PSNR: {compute_psnr(clean, blurred):.2f} dB")
    print()
    
    print("1. 标准RL (150迭代) - 噪声会放大")
    rl_std = richardson_lucy(blurred, psf, iterations=150)
    print(f"   PSNR: {compute_psnr(clean, rl_std):.2f} dB")
    
    print("\n2. 早停RL (监控残差)")
    rl_early, hist_early = richardson_lucy_early_stop(
        blurred, psf, max_iterations=150, tolerance=1e-4, patience=8
    )
    print(f"   停止于 {len(hist_early['iteration'])} 次迭代")
    print(f"   PSNR: {compute_psnr(clean, rl_early):.2f} dB")
    
    print("\n3. TV正则化RL (λ=0.005)")
    rl_tv, hist_tv = richardson_lucy_tv(
        blurred, psf, iterations=150, lambda_tv=0.005
    )
    print(f"   PSNR: {compute_psnr(clean, rl_tv):.2f} dB")
    
    print("\n4. 自适应TV正则化RL")
    rl_ada, hist_ada = richardson_lucy_adaptive_tv(
        blurred, psf, max_iterations=150
    )
    print(f"   PSNR: {compute_psnr(clean, rl_ada):.2f} dB")
    print(f"   最终λ: {hist_ada['lambda'][-1]:.4f}")
    
    print("\n总结:")
    print(f"  标准RL过拟合: 噪声放大，PSNR较低")
    print(f"  早停法: 自动在最佳点停止")
    print(f"  TV正则化: 有效抑制椒盐噪声")
    print()


def example_1_basic_usage():
    """
    示例1: 基本用法 - 使用高斯PSF
    """
    print("=" * 50)
    print("示例1: 基本RL反卷积 (高斯PSF)")
    print("=" * 50)
    
    size = 64
    image = np.zeros((size, size))
    image[20, 20] = 1.0
    image[40, 30] = 0.8
    image[30, 45] = 0.6
    
    psf = generate_gaussian_psf(size=15, sigma=2.5)
    blurred = blur_image(image, psf, add_noise_flag=True, snr=50)
    
    restored = richardson_lucy(blurred, psf, iterations=50)
    
    psnr_before = compute_psnr(image, blurred)
    psnr_after = compute_psnr(image, restored)
    
    print(f"模糊前PSNR: {psnr_before:.2f} dB")
    print(f"恢复后PSNR: {psnr_after:.2f} dB")
    print(f"改善: {psnr_after - psnr_before:.2f} dB")
    print()


def example_2_atmospheric_turbulence():
    """
    示例2: 大气湍流模糊恢复
    """
    print("=" * 50)
    print("示例2: 大气湍流模糊恢复")
    print("=" * 50)
    
    size = 100
    image = np.zeros((size, size))
    
    for _ in range(15):
        x, y = np.random.randint(10, size - 10, 2)
        image[y, x] = np.random.uniform(0.5, 1.0)
    
    psf = generate_atmospheric_psf(size=25, seeing=3.0)
    blurred = blur_image(image, psf, add_noise_flag=True, snr=40)
    
    restored_basic = richardson_lucy(blurred, psf, iterations=80)
    restored_tv = richardson_lucy_with_regularization(
        blurred, psf, iterations=80, lambda_reg=0.01
    )
    
    print(f"模糊图像PSNR: {compute_psnr(image, blurred):.2f} dB")
    print(f"标准RL恢复:   {compute_psnr(image, restored_basic):.2f} dB")
    print(f"RL+TV正则化:  {compute_psnr(image, restored_tv):.2f} dB")
    print()


def example_3_iteration_analysis():
    """
    示例3: 分析迭代次数对恢复效果的影响
    """
    print("=" * 50)
    print("示例3: 迭代次数分析")
    print("=" * 50)
    
    size = 64
    image = np.zeros((size, size))
    image[32, 32] = 1.0
    
    psf = generate_gaussian_psf(size=21, sigma=3.0)
    blurred = blur_image(image, psf, add_noise_flag=True, snr=30)
    
    iterations_list = [10, 30, 50, 100, 200]
    
    print("迭代次数 | PSNR (dB)")
    print("-" * 25)
    for iters in iterations_list:
        restored = richardson_lucy(blurred, psf, iterations=iters)
        psnr = compute_psnr(image, restored)
        print(f"{iters:7d} | {psnr:8.2f}")
    print()


def load_real_image(filepath):
    """
    加载真实图像的辅助函数
    
    参数:
        filepath: 图像文件路径
    
    返回:
        归一化的灰度图像数组
    """
    try:
        from PIL import Image
        img = Image.open(filepath).convert('L')
        img_array = np.array(img, dtype=np.float64) / 255.0
        return img_array
    except ImportError:
        print("请安装Pillow: pip install pillow")
        return None


if __name__ == "__main__":
    np.random.seed(42)
    
    example_1_basic_usage()
    example_2_atmospheric_turbulence()
    example_3_iteration_analysis()
    example_4_lucky_imaging()
    example_5_multiframe_rl()
    example_6_blind_deconvolution()
    example_7_noise_suppression()
    
    print("=" * 60)
    print("所有示例完成!")
    print("=" * 60)
    print("\n使用说明:")
    print("1. 确保已安装依赖: pip install numpy scipy matplotlib")
    print("2. 单帧RL演示: python richardson_lucy.py")
    print("3. 多帧反卷积演示: python multi_frame_deconvolution.py")
    print("4. 噪声抑制对比: python rl_regularized.py")
    print("5. 简化示例: python example_usage.py")
    print("6. 对于真实图像，使用 load_real_image() 加载")
