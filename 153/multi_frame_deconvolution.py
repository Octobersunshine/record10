"""
多帧盲反卷积与幸运成像模块
用于地基望远镜大气湍流模糊恢复
"""

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from richardson_lucy import (
    richardson_lucy,
    generate_gaussian_psf,
    blur_image,
    compute_psnr
)


def generate_phase_screen(size, r0, L0=1e6):
    """
    生成Kolmogorov湍流相位屏
    
    参数:
        size: 相位屏尺寸
        r0: Fried参数 (像素)
        L0: 外尺度
    
    返回:
        相位屏
    """
    x = np.arange(size) - size // 2
    y = x[:, np.newaxis]
    fx = x / size
    fy = y / size
    
    f = np.sqrt(fx**2 + fy**2)
    f[size // 2, size // 2] = 1e-10
    
    f0 = 1.0 / L0
    fm = 5.92 / (r0 * 1.0)
    
    PSD = 0.023 * r0**(-5.0/3.0) * np.exp(-(f / fm)**2) / (f**2 + f0**2)**(11.0/6.0)
    PSD[size // 2, size // 2] = 0
    
    random_phase = np.random.normal(0, 1, (size, size)) + 1j * np.random.normal(0, 1, (size, size))
    
    phase_ft = np.sqrt(PSD) * random_phase * size
    
    phase = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(phase_ft))).real
    phase = phase - phase.mean()
    
    return phase


def phase_screen_to_psf(phase_screen, wavelength=1.0, pupil_radius=None):
    """
    从相位屏生成PSF
    
    参数:
        phase_screen: 相位屏
        wavelength: 波长 (相对单位)
        pupil_radius: 光瞳半径
    
    返回:
        归一化的PSF
    """
    size = phase_screen.shape[0]
    
    if pupil_radius is None:
        pupil_radius = size // 4
    
    x = np.arange(size) - size // 2
    y = x[:, np.newaxis]
    r = np.sqrt(x**2 + y**2)
    pupil = (r <= pupil_radius).astype(float)
    
    complex_field = pupil * np.exp(1j * phase_screen)
    
    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_field)))
    psf = np.abs(ft)**2
    
    psf = psf / psf.sum()
    
    return psf


def generate_multiframe_psfs(num_frames, psf_size=31, seeing=3.0, r0_variation=0.2):
    """
    生成多帧不同的大气湍流PSF
    
    参数:
        num_frames: 帧数
        psf_size: PSF尺寸
        seeing: 平均视宁度
        r0_variation: r0的相对变化
    
    返回:
        PSF数组 [num_frames, psf_size, psf_size]
    """
    psfs = np.zeros((num_frames, psf_size, psf_size))
    
    phase_size = max(psf_size * 2, 64)
    
    for i in range(num_frames):
        r0_factor = 1.0 + np.random.uniform(-r0_variation, r0_variation)
        current_r0 = psf_size / (seeing * r0_factor)
        
        phase_screen = generate_phase_screen(phase_size, r0=current_r0)
        psf = phase_screen_to_psf(phase_screen, pupil_radius=phase_size // 4)
        
        center = phase_size // 2
        half = psf_size // 2
        psf_cropped = psf[center - half:center + half + 1, center - half:center + half + 1]
        
        psfs[i] = psf_cropped / psf_cropped.sum()
    
    return psfs


def generate_multiframe_images(clean_image, psfs, snr=30):
    """
    生成多帧模糊图像
    
    参数:
        clean_image: 原始清晰图像
        psfs: PSF数组 [num_frames, H, W]
        snr: 信噪比
    
    返回:
        多帧模糊图像 [num_frames, H, W]
    """
    num_frames = psfs.shape[0]
    images = np.zeros((num_frames,) + clean_image.shape)
    
    for i in range(num_frames):
        images[i] = blur_image(clean_image, psfs[i], add_noise_flag=True, snr=snr)
    
    return images


def lucky_imaging_select(images, selection_fraction=0.1, metric='strehl'):
    """
    幸运成像选帧算法
    
    参数:
        images: 多帧图像 [num_frames, H, W]
        selection_fraction: 选择比例 (0-1)
        metric: 选帧质量度量
                'strehl': 峰值强度近似Strehl比
                'gradient': 梯度强度
                'variance': 方差
    
    返回:
        selected_indices: 选中帧的索引
        quality_scores: 各帧的质量分数
    """
    num_frames = images.shape[0]
    num_select = max(1, int(num_frames * selection_fraction))
    
    quality_scores = np.zeros(num_frames)
    
    for i in range(num_frames):
        img = images[i]
        
        if metric == 'strehl':
            quality_scores[i] = img.max()
        
        elif metric == 'gradient':
            gx, gy = np.gradient(img)
            quality_scores[i] = np.mean(np.sqrt(gx**2 + gy**2))
        
        elif metric == 'variance':
            quality_scores[i] = img.var()
        
        elif metric == 'entropy':
            hist, _ = np.histogram(img, bins=64, density=True)
            hist = hist[hist > 0]
            quality_scores[i] = -np.sum(hist * np.log2(hist))
    
    selected_indices = np.argsort(quality_scores)[-num_select:]
    
    return selected_indices, quality_scores


def lucky_imaging_reconstruct(images, selected_indices, method='mean'):
    """
    幸运成像重建
    
    参数:
        images: 多帧图像 [num_frames, H, W]
        selected_indices: 选中帧的索引
        method: 重建方法
                'mean': 简单平均
                'median': 中值
                'weighted': 按质量加权
    
    返回:
        重建图像
    """
    selected = images[selected_indices]
    
    if method == 'mean':
        return selected.mean(axis=0)
    
    elif method == 'median':
        return np.median(selected, axis=0)
    
    elif method == 'weighted':
        peaks = selected.max(axis=(1, 2))
        weights = peaks / peaks.sum()
        return np.average(selected, axis=0, weights=weights)
    
    else:
        return selected.mean(axis=0)


def multiframe_richardson_lucy(images, psf, iterations=50, use_fft=True):
    """
    多帧RL反卷积（已知PSF）
    
    参数:
        images: 多帧模糊图像 [num_frames, H, W]
        psf: 点扩散函数
        iterations: 迭代次数
        use_fft: 是否使用FFT
    
    返回:
        恢复的清晰图像
    """
    num_frames = images.shape[0]
    
    estimate = np.ones_like(images[0]) * images.mean()
    psf_mirror = psf[::-1, ::-1]
    
    for it in range(iterations):
        conv_stack = np.zeros_like(images)
        
        for i in range(num_frames):
            if use_fft:
                conv_stack[i] = fftconvolve(estimate, psf, mode='same')
            else:
                conv_stack[i] = convolve(estimate, psf, mode='constant')
        
        relative_blur = images / (conv_stack + 1e-12)
        avg_relative_blur = relative_blur.mean(axis=0)
        
        if use_fft:
            error_estimate = fftconvolve(avg_relative_blur, psf_mirror, mode='same')
        else:
            error_estimate = convolve(avg_relative_blur, psf_mirror, mode='constant')
        
        estimate *= error_estimate
        estimate = np.clip(estimate, 0, 1)
    
    return estimate


def multiframe_blind_deconvolution(images, psf_init=None, iterations=50, 
                                   image_iter=5, psf_iter=3, 
                                   psf_size=31, use_fft=True,
                                   psf_reg=0.01):
    """
    多帧盲反卷积（交替估计图像和PSF）
    
    使用期望最大化(EM)风格的交替优化
    
    参数:
        images: 多帧模糊图像 [num_frames, H, W]
        psf_init: 初始PSF估计 (None则用高斯初始化)
        iterations: 总迭代次数
        image_iter: 每次PSF更新后的图像迭代次数
        psf_iter: 每次图像更新后的PSF迭代次数
        psf_size: PSF尺寸
        use_fft: 是否使用FFT
        psf_reg: PSF正则化强度
    
    返回:
        estimate: 恢复的清晰图像
        psf_estimate: 估计的PSF
    """
    num_frames = images.shape[0]
    img_shape = images.shape[1:]
    
    if psf_init is None:
        psf_estimate = generate_gaussian_psf(size=psf_size, sigma=2.0)
    else:
        psf_estimate = psf_init.copy()
        psf_size = psf_estimate.shape[0]
    
    psf_estimate = psf_estimate / psf_estimate.sum()
    
    estimate = np.ones(img_shape) * images.mean()
    
    for it in range(iterations):
        for _ in range(image_iter):
            conv_stack = np.zeros_like(images)
            for i in range(num_frames):
                if use_fft:
                    conv_stack[i] = fftconvolve(estimate, psf_estimate, mode='same')
                else:
                    conv_stack[i] = convolve(estimate, psf_estimate, mode='constant')
            
            relative_blur = images / (conv_stack + 1e-12)
            avg_relative_blur = relative_blur.mean(axis=0)
            
            psf_mirror = psf_estimate[::-1, ::-1]
            if use_fft:
                error_estimate = fftconvolve(avg_relative_blur, psf_mirror, mode='same')
            else:
                error_estimate = convolve(avg_relative_blur, psf_mirror, mode='constant')
            
            estimate *= error_estimate
            estimate = np.clip(estimate, 0, 1)
        
        for _ in range(psf_iter):
            psf_estimate_new = np.zeros_like(psf_estimate)
            estimate_mirror = estimate[::-1, ::-1]
            
            for i in range(num_frames):
                if use_fft:
                    conv = fftconvolve(estimate, psf_estimate, mode='same')
                else:
                    conv = convolve(estimate, psf_estimate, mode='constant')
                
                relative_blur = images[i] / (conv + 1e-12)
                
                if use_fft:
                    psf_update = fftconvolve(relative_blur, estimate_mirror, mode='valid')
                else:
                    psf_update = convolve(relative_blur, estimate_mirror, mode='valid')
                
                if psf_update.shape != psf_estimate.shape:
                    ph, pw = psf_estimate.shape
                    uh, uw = psf_update.shape
                    start_h = (uh - ph) // 2
                    start_w = (uw - pw) // 2
                    psf_update = psf_update[start_h:start_h + ph, start_w:start_w + pw]
                
                psf_estimate_new += psf_update
            
            psf_estimate_new = psf_estimate_new / num_frames
            psf_estimate = psf_estimate * psf_estimate_new
            
            psf_estimate = np.maximum(psf_estimate, psf_reg * psf_estimate.max())
            psf_estimate = psf_estimate / psf_estimate.sum()
        
        if (it + 1) % 10 == 0:
            print(f"  迭代 {it + 1}/{iterations} 完成")
    
    return estimate, psf_estimate


def joint_multi_frame_deconvolution(images, num_psf_modes=5, iterations=30, 
                                    image_iter=3, psf_size=31):
    """
    联合多帧反卷积 - 每帧独立PSF，共享清晰图像
    
    参数:
        images: 多帧图像 [num_frames, H, W]
        num_psf_modes: 要估计的独立PSF数量 (可以少于帧数)
        iterations: 总迭代次数
        image_iter: 图像迭代次数
        psf_size: PSF尺寸
    
    返回:
        estimate: 恢复图像
        psf_estimates: 估计的PSF数组
    """
    num_frames = images.shape[0]
    
    if num_psf_modes < num_frames:
        _, quality = lucky_imaging_select(images, selection_fraction=num_psf_modes/num_frames)
        mode_assignments = np.argmin(np.abs(quality[:, None] - quality[::num_frames//num_psf_modes]), axis=1)
    else:
        mode_assignments = np.arange(num_frames)
        num_psf_modes = num_frames
    
    psf_estimates = np.zeros((num_psf_modes, psf_size, psf_size))
    for m in range(num_psf_modes):
        psf_estimates[m] = generate_gaussian_psf(psf_size, sigma=2.5)
    
    estimate = np.ones_like(images[0]) * images.mean()
    
    for it in range(iterations):
        for _ in range(image_iter):
            total_update = np.zeros_like(estimate)
            
            for i in range(num_frames):
                m = mode_assignments[i]
                psf = psf_estimates[m]
                psf_mirror = psf[::-1, ::-1]
                
                conv = fftconvolve(estimate, psf, mode='same')
                relative_blur = images[i] / (conv + 1e-12)
                error_estimate = fftconvolve(relative_blur, psf_mirror, mode='same')
                
                total_update += error_estimate
            
            estimate *= total_update / num_frames
            estimate = np.clip(estimate, 0, 1)
        
        for m in range(num_psf_modes):
            frame_indices = np.where(mode_assignments == m)[0]
            
            if len(frame_indices) == 0:
                continue
            
            psf_update = np.zeros_like(psf_estimates[m])
            estimate_mirror = estimate[::-1, ::-1]
            
            for i in frame_indices:
                conv = fftconvolve(estimate, psf_estimates[m], mode='same')
                relative_blur = images[i] / (conv + 1e-12)
                update = fftconvolve(relative_blur, estimate_mirror, mode='valid')
                
                if update.shape != psf_estimates[m].shape:
                    ph, pw = psf_estimates[m].shape
                    uh, uw = update.shape
                    start_h = (uh - ph) // 2
                    start_w = (uw - pw) // 2
                    update = update[start_h:start_h + ph, start_w:start_w + pw]
                
                psf_update += update
            
            psf_estimates[m] *= psf_update / len(frame_indices)
            psf_estimates[m] = np.maximum(psf_estimates[m], 1e-6 * psf_estimates[m].max())
            psf_estimates[m] = psf_estimates[m] / psf_estimates[m].sum()
        
        if (it + 1) % 10 == 0:
            print(f"  联合迭代 {it + 1}/{iterations} 完成")
    
    return estimate, psf_estimates


def main():
    """
    多帧盲反卷积演示
    """
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("多帧盲反卷积与幸运成像 - 地基望远镜图像恢复演示")
    print("=" * 70)
    
    img_size = 128
    num_frames = 50
    
    clean = np.zeros((img_size, img_size))
    num_stars = 25
    for _ in range(num_stars):
        x, y = np.random.randint(10, img_size - 10, 2)
        brightness = np.random.uniform(0.4, 1.0)
        clean[y, x] = brightness
    
    print(f"\n1. 生成模拟数据:")
    print(f"   图像尺寸: {img_size}x{img_size}")
    print(f"   恒星数量: {num_stars}")
    print(f"   帧数: {num_frames}")
    
    print("\n2. 生成多帧大气湍流PSF...")
    psfs = generate_multiframe_psfs(num_frames=num_frames, psf_size=31, seeing=3.5)
    
    print("3. 生成多帧模糊图像...")
    images = generate_multiframe_images(clean, psfs, snr=35)
    
    avg_image = images.mean(axis=0)
    psnr_avg = compute_psnr(clean, avg_image)
    print(f"   简单平均PSNR: {psnr_avg:.2f} dB")
    
    print("\n4. 幸运成像选帧...")
    selected, scores = lucky_imaging_select(images, selection_fraction=0.2, metric='strehl')
    print(f"   选中帧数: {len(selected)}/{num_frames} ({len(selected)/num_frames*100:.0f}%)")
    
    lucky_recon = lucky_imaging_reconstruct(images, selected, method='mean')
    psnr_lucky = compute_psnr(clean, lucky_recon)
    print(f"   幸运成像PSNR: {psnr_lucky:.2f} dB")
    
    print("\n5. 多帧RL反卷积 (使用平均PSF)...")
    avg_psf = psfs.mean(axis=0)
    mf_rl = multiframe_richardson_lucy(images, avg_psf, iterations=80)
    psnr_mf_rl = compute_psnr(clean, mf_rl)
    print(f"   多帧RL PSNR: {psnr_mf_rl:.2f} dB")
    
    print("\n6. 多帧盲反卷积 (未知PSF)...")
    blind_result, blind_psf = multiframe_blind_deconvolution(
        images, iterations=40, image_iter=5, psf_iter=3, psf_size=31
    )
    psnr_blind = compute_psnr(clean, blind_result)
    print(f"   盲反卷积PSNR: {psnr_blind:.2f} dB")
    
    print("\n7. 联合多帧反卷积...")
    joint_result, joint_psfs = joint_multi_frame_deconvolution(
        images, num_psf_modes=3, iterations=30, psf_size=31
    )
    psnr_joint = compute_psnr(clean, joint_result)
    print(f"   联合反卷积PSNR: {psnr_joint:.2f} dB")
    
    print("\n" + "=" * 70)
    print("方法对比 (PSNR):")
    print("-" * 70)
    print(f"  单帧最佳:      {max([compute_psnr(clean, img) for img in images]):.2f} dB")
    print(f"  简单平均:      {psnr_avg:.2f} dB")
    print(f"  幸运成像:      {psnr_lucky:.2f} dB (+{psnr_lucky - psnr_avg:.2f} dB)")
    print(f"  多帧RL:        {psnr_mf_rl:.2f} dB (+{psnr_mf_rl - psnr_avg:.2f} dB)")
    print(f"  盲反卷积:      {psnr_blind:.2f} dB (+{psnr_blind - psnr_avg:.2f} dB)")
    print(f"  联合反卷积:    {psnr_joint:.2f} dB (+{psnr_joint - psnr_avg:.2f} dB)")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(clean, cmap='gray', origin='lower')
    axes[0, 0].set_title('原始清晰图像')
    
    axes[0, 1].imshow(psfs[0], cmap='hot', origin='lower')
    axes[0, 1].set_title('示例PSF (帧1)')
    
    axes[0, 2].imshow(images[0], cmap='gray', origin='lower')
    axes[0, 2].set_title('单帧模糊图像')
    
    axes[0, 3].imshow(avg_image, cmap='gray', origin='lower')
    axes[0, 3].set_title(f'简单平均\nPSNR: {psnr_avg:.2f} dB')
    
    axes[1, 0].imshow(lucky_recon, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'幸运成像\nPSNR: {psnr_lucky:.2f} dB')
    
    axes[1, 1].imshow(mf_rl, cmap='gray', origin='lower')
    axes[1, 1].set_title(f'多帧RL\nPSNR: {psnr_mf_rl:.2f} dB')
    
    axes[1, 2].imshow(blind_result, cmap='gray', origin='lower')
    axes[1, 2].set_title(f'盲反卷积\nPSNR: {psnr_blind:.2f} dB')
    
    axes[1, 3].imshow(joint_result, cmap='gray', origin='lower')
    axes[1, 3].set_title(f'联合反卷积\nPSNR: {psnr_joint:.2f} dB')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('multiframe_deconvolution_demo.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 multiframe_deconvolution_demo.png")
    
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    main()
