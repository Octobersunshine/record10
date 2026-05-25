"""
正则化Richardson-Lucy反卷积模块
包含早停法和多种正则化策略，抑制噪声放大
"""

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from richardson_lucy import (
    generate_gaussian_psf,
    generate_atmospheric_psf,
    blur_image,
    compute_psnr
)


def compute_residual(image, estimate, psf):
    """
    计算残差 || image - estimate * psf ||
    
    参数:
        image: 观测图像
        estimate: 当前估计图像
        psf: 点扩散函数
    
    返回:
        残差的L2范数
    """
    convolved = fftconvolve(estimate, psf, mode='same')
    residual = np.sum((image - convolved)**2)
    return residual


def compute_tv_norm(image):
    """
    计算总变分(TV)范数
    
    参数:
        image: 输入图像
    
    返回:
        TV范数值 (各向同性)
    """
    gx, gy = np.gradient(image)
    tv = np.sum(np.sqrt(gx**2 + gy**2 + 1e-12))
    return tv


def compute_gradient_l2(estimate):
    """
    计算梯度L2范数（Tikhonov正则化）
    """
    gx, gy = np.gradient(estimate)
    return np.sum(gx**2 + gy**2)


def richardson_lucy_early_stop(image, psf, max_iterations=200, 
                               tolerance=1e-5, patience=5,
                               use_fft=True, verbose=False):
    """
    带早停的Richardson-Lucy反卷积
    
    监控残差变化，当残差不再明显下降或开始上升时停止
    
    参数:
        image: 模糊图像
        psf: 点扩散函数
        max_iterations: 最大迭代次数
        tolerance: 残差相对变化阈值
        patience: 容忍次数（连续多少次变化小于阈值才停止）
        use_fft: 是否使用FFT
        verbose: 是否打印进度
    
    返回:
        estimate: 恢复图像
        history: 迭代历史记录
    """
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    
    psf_mirror = psf[::-1, ::-1]
    estimate = np.ones_like(image) * image.mean()
    
    history = {
        'residual': [],
        'iteration': []
    }
    
    best_estimate = estimate.copy()
    best_residual = float('inf')
    no_improve_count = 0
    
    for i in range(max_iterations):
        if use_fft:
            convolved = fftconvolve(estimate, psf, mode='same')
        else:
            convolved = convolve(estimate, psf, mode='constant')
        
        relative_blur = image / (convolved + 1e-12)
        
        if use_fft:
            error_estimate = fftconvolve(relative_blur, psf_mirror, mode='same')
        else:
            error_estimate = convolve(relative_blur, psf_mirror, mode='constant')
        
        estimate *= error_estimate
        estimate = np.clip(estimate, 0, 1)
        
        residual = compute_residual(image, estimate, psf)
        history['residual'].append(residual)
        history['iteration'].append(i + 1)
        
        if residual < best_residual * (1 - tolerance):
            best_residual = residual
            best_estimate = estimate.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            if verbose:
                print(f"早停于第 {i+1} 次迭代 (残差收敛)")
            break
        
        if verbose and (i + 1) % 20 == 0:
            print(f"迭代 {i+1}/{max_iterations}, 残差: {residual:.6e}")
    
    return best_estimate, history


def richardson_lucy_tv(image, psf, iterations=100, lambda_tv=0.01, 
                       use_fft=True, tv_type='isotropic', verbose=False):
    """
    总变分(TV)正则化Richardson-Lucy反卷积
    
    使用半二次分裂或梯度投影实现TV正则化
    
    参数:
        image: 模糊图像
        psf: 点扩散函数
        iterations: 迭代次数
        lambda_tv: TV正则化强度
        use_fft: 是否使用FFT
        tv_type: 'isotropic' | 'anisotropic'
        verbose: 是否打印进度
    
    返回:
        estimate: 恢复图像
        history: 迭代历史
    """
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    
    psf_mirror = psf[::-1, ::-1]
    estimate = np.ones_like(image) * image.mean()
    
    history = {
        'residual': [],
        'tv_norm': [],
        'psnr': []
    }
    
    for i in range(iterations):
        if use_fft:
            convolved = fftconvolve(estimate, psf, mode='same')
        else:
            convolved = convolve(estimate, psf, mode='constant')
        
        relative_blur = image / (convolved + 1e-12)
        
        if use_fft:
            error_estimate = fftconvolve(relative_blur, psf_mirror, mode='same')
        else:
            error_estimate = convolve(relative_blur, psf_mirror, mode='constant')
        
        data_term = error_estimate
        
        gx, gy = np.gradient(estimate)
        
        if tv_type == 'isotropic':
            tv_grad_mag = np.sqrt(gx**2 + gy**2 + 1e-12)
            gx_normalized = gx / tv_grad_mag
            gy_normalized = gy / tv_grad_mag
            
            div_gx, _ = np.gradient(gx_normalized)
            _, div_gy = np.gradient(gy_normalized)
            div = div_gx + div_gy
            
        else:
            eps = 1e-12
            gx_normalized = gx / (np.abs(gx) + eps)
            gy_normalized = gy / (np.abs(gy) + eps)
            
            div_gx, _ = np.gradient(gx_normalized)
            _, div_gy = np.gradient(gy_normalized)
            div = div_gx + div_gy
        
        tv_term = 1 - lambda_tv * div
        tv_term = np.clip(tv_term, 0.5, 2.0)
        
        estimate *= data_term * tv_term
        estimate = np.clip(estimate, 0, 1)
        
        history['residual'].append(compute_residual(image, estimate, psf))
        history['tv_norm'].append(compute_tv_norm(estimate))
    
    return estimate, history


def richardson_lucy_tikhonov(image, psf, iterations=100, lambda_tik=0.01, 
                             use_fft=True):
    """
    Tikhonov正则化RL反卷积（L2正则化）
    
    参数:
        image: 模糊图像
        psf: 点扩散函数
        iterations: 迭代次数
        lambda_tik: Tikhonov正则化强度
        use_fft: 是否使用FFT
    
    返回:
        estimate: 恢复图像
        history: 迭代历史
    """
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    
    psf_mirror = psf[::-1, ::-1]
    estimate = np.ones_like(image) * image.mean()
    
    history = {'residual': [], 'l2_grad': []}
    
    for i in range(iterations):
        if use_fft:
            convolved = fftconvolve(estimate, psf, mode='same')
        else:
            convolved = convolve(estimate, psf, mode='constant')
        
        relative_blur = image / (convolved + 1e-12)
        
        if use_fft:
            error_estimate = fftconvolve(relative_blur, psf_mirror, mode='same')
        else:
            error_estimate = convolve(relative_blur, psf_mirror, mode='constant')
        
        gx, gy = np.gradient(estimate)
        laplacian = np.gradient(gx)[0] + np.gradient(gy)[1]
        
        denom = 1 + lambda_tik * np.abs(laplacian)
        regularization = 1 / (denom + 1e-8)
        
        estimate *= error_estimate * regularization
        estimate = np.clip(estimate, 0, 1)
        
        history['residual'].append(compute_residual(image, estimate, psf))
        history['l2_grad'].append(compute_gradient_l2(estimate))
    
    return estimate, history


def richardson_lucy_adaptive_tv(image, psf, max_iterations=100, 
                                lambda_init=0.001, lambda_max=0.05,
                                noise_level=None, use_fft=True):
    """
    自适应TV正则化RL反卷积
    
    根据残差变化自适应调整正则化强度
    
    参数:
        image: 模糊图像
        psf: 点扩散函数
        max_iterations: 最大迭代次数
        lambda_init: 初始正则化强度
        lambda_max: 最大正则化强度
        noise_level: 噪声水平估计 (None则自动估计)
        use_fft: 是否使用FFT
    
    返回:
        estimate: 恢复图像
        history: 迭代历史
    """
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    
    if noise_level is None:
        noise_level = np.std(image) * 0.1
    
    psf_mirror = psf[::-1, ::-1]
    estimate = np.ones_like(image) * image.mean()
    
    history = {
        'residual': [],
        'lambda': [],
        'tv_norm': []
    }
    
    residual_prev = float('inf')
    lambda_tv = lambda_init
    
    for i in range(max_iterations):
        if use_fft:
            convolved = fftconvolve(estimate, psf, mode='same')
        else:
            convolved = convolve(estimate, psf, mode='constant')
        
        relative_blur = image / (convolved + 1e-12)
        
        if use_fft:
            error_estimate = fftconvolve(relative_blur, psf_mirror, mode='same')
        else:
            error_estimate = convolve(relative_blur, psf_mirror, mode='constant')
        
        residual = np.mean((image - convolved)**2)
        
        if residual < residual_prev * 0.99:
            lambda_tv = max(lambda_init, lambda_tv * 0.95)
        else:
            lambda_tv = min(lambda_max, lambda_tv * 1.1)
        
        gx, gy = np.gradient(estimate)
        tv_grad_mag = np.sqrt(gx**2 + gy**2 + 1e-12)
        gx_norm = gx / tv_grad_mag
        gy_norm = gy / tv_grad_mag
        
        div_gx, _ = np.gradient(gx_norm)
        _, div_gy = np.gradient(gy_norm)
        div = div_gx + div_gy
        
        tv_term = 1 - lambda_tv * div
        tv_term = np.clip(tv_term, 0.5, 2.0)
        
        estimate *= error_estimate * tv_term
        estimate = np.clip(estimate, 0, 1)
        
        residual_prev = residual
        
        history['residual'].append(residual)
        history['lambda'].append(lambda_tv)
        history['tv_norm'].append(compute_tv_norm(estimate))
    
    return estimate, history


def richardson_lucy_with_reference(image, psf, reference, 
                                   iterations=100, lambda_ref=0.1,
                                   use_fft=True):
    """
    带参考图像的RL反卷积
    
    当有低分辨率参考（如幸运成像均值）时使用
    
    参数:
        image: 模糊图像
        psf: 点扩散函数
        reference: 参考图像（如幸运成像结果）
        iterations: 迭代次数
        lambda_ref: 参考正则化强度
        use_fft: 是否使用FFT
    
    返回:
        estimate: 恢复图像
    """
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    reference = reference.astype(np.float64)
    
    psf_mirror = psf[::-1, ::-1]
    estimate = reference.copy()
    
    for i in range(iterations):
        if use_fft:
            convolved = fftconvolve(estimate, psf, mode='same')
        else:
            convolved = convolve(estimate, psf, mode='constant')
        
        relative_blur = image / (convolved + 1e-12)
        
        if use_fft:
            error_estimate = fftconvolve(relative_blur, psf_mirror, mode='same')
        else:
            error_estimate = convolve(relative_blur, psf_mirror, mode='constant')
        
        ref_term = reference / (estimate + 1e-8)
        ref_term = ref_term ** lambda_ref
        
        estimate *= error_estimate * ref_term
        estimate = np.clip(estimate, 0, 1)
    
    return estimate


def wavelet_denoise(image, threshold=0.1, level=2):
    """
    简单的小波去噪（作为后处理）
    
    参数:
        image: 输入图像
        threshold: 阈值
        level: 分解层数
    
    返回:
        去噪图像
    """
    try:
        import pywt
        
        coeffs = pywt.wavedec2(image, 'db4', level=level)
        
        coeffs_thresh = [coeffs[0]]
        for detail in coeffs[1:]:
            coeffs_thresh.append(tuple(
                pywt.threshold(d, threshold, mode='soft') for d in detail
            ))
        
        return pywt.waverec2(coeffs_thresh, 'db4')
    except ImportError:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=0.5)


def main():
    """
    噪声抑制效果对比演示
    """
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("RL反卷积噪声抑制对比演示")
    print("=" * 70)
    
    img_size = 128
    
    clean = np.zeros((img_size, img_size))
    for _ in range(30):
        x, y = np.random.randint(10, img_size - 10, 2)
        brightness = np.random.uniform(0.4, 1.0)
        radius = np.random.randint(1, 3)
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if 0 <= x + i < img_size and 0 <= y + j < img_size:
                    dist = np.sqrt(i**2 + j**2)
                    if dist <= radius:
                        clean[y + j, x + i] = max(
                            clean[y + j, x + i], 
                            brightness * (1 - dist / (radius + 1))
                        )
    clean = np.clip(clean, 0, 1)
    
    psf = generate_atmospheric_psf(size=31, seeing=3.0)
    blurred = blur_image(clean, psf, add_noise_flag=True, snr=25)
    
    print(f"\n图像尺寸: {img_size}x{img_size}")
    print(f"噪声水平: SNR=25 (较高噪声)")
    
    print("\n1. 标准RL (200迭代，无正则化)...")
    from richardson_lucy import richardson_lucy
    rl_standard = richardson_lucy(blurred, psf, iterations=200, use_fft=True)
    
    print("2. 早停RL (残差监控)...")
    rl_early, history_early = richardson_lucy_early_stop(
        blurred, psf, max_iterations=200, tolerance=1e-4, patience=10
    )
    
    print("3. TV正则化RL (λ=0.005)...")
    rl_tv, history_tv = richardson_lucy_tv(
        blurred, psf, iterations=200, lambda_tv=0.005
    )
    
    print("4. 自适应TV正则化RL...")
    rl_adaptive, history_adaptive = richardson_lucy_adaptive_tv(
        blurred, psf, max_iterations=200
    )
    
    print("5. Tikhonov正则化RL...")
    rl_tik, history_tik = richardson_lucy_tikhonov(
        blurred, psf, iterations=200, lambda_tik=0.001
    )
    
    psnr_blurred = compute_psnr(clean, blurred)
    psnr_standard = compute_psnr(clean, rl_standard)
    psnr_early = compute_psnr(clean, rl_early)
    psnr_tv = compute_psnr(clean, rl_tv)
    psnr_adaptive = compute_psnr(clean, rl_adaptive)
    psnr_tik = compute_psnr(clean, rl_tik)
    
    noise_standard = np.std(rl_standard - blurred)
    noise_early = np.std(rl_early - blurred)
    noise_tv = np.std(rl_tv - blurred)
    noise_adaptive = np.std(rl_adaptive - blurred)
    
    print("\n" + "=" * 70)
    print("方法对比:")
    print("-" * 70)
    print(f"  模糊图像:      PSNR = {psnr_blurred:.2f} dB")
    print(f"  标准RL:        PSNR = {psnr_standard:.2f} dB (噪声放大明显)")
    print(f"  早停RL:        PSNR = {psnr_early:.2f} dB (停止于{len(history_early['iteration'])}次)")
    print(f"  TV正则化RL:    PSNR = {psnr_tv:.2f} dB")
    print(f"  自适应TV:      PSNR = {psnr_adaptive:.2f} dB")
    print(f"  Tikhonov:      PSNR = {psnr_tik:.2f} dB")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(clean, cmap='gray', origin='lower')
    axes[0, 0].set_title('原始清晰图像')
    
    axes[0, 1].imshow(blurred, cmap='gray', origin='lower')
    axes[0, 1].set_title(f'模糊图像\nPSNR: {psnr_blurred:.2f} dB')
    
    axes[0, 2].imshow(rl_standard, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'标准RL (200迭代)\nPSNR: {psnr_standard:.2f} dB')
    
    axes[0, 3].imshow(rl_early, cmap='gray', origin='lower')
    axes[0, 3].set_title(f'早停RL ({len(history_early["iteration"])}迭代)\nPSNR: {psnr_early:.2f} dB')
    
    axes[1, 0].imshow(rl_tv, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'TV正则化RL\nPSNR: {psnr_tv:.2f} dB')
    
    axes[1, 1].imshow(rl_adaptive, cmap='gray', origin='lower')
    axes[1, 1].set_title(f'自适应TV RL\nPSNR: {psnr_adaptive:.2f} dB')
    
    diff_standard = np.abs(rl_standard - clean)
    axes[1, 2].imshow(diff_standard, cmap='hot', origin='lower')
    axes[1, 2].set_title('标准RL误差图\n(可见椒盐噪声)')
    
    diff_tv = np.abs(rl_tv - clean)
    axes[1, 3].imshow(diff_tv, cmap='hot', origin='lower')
    axes[1, 3].set_title('TV正则化误差图\n(噪声被抑制)')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('noise_suppression_demo.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 noise_suppression_demo.png")
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    
    axes2[0].plot(history_early['residual'])
    axes2[0].set_xlabel('迭代')
    axes2[0].set_ylabel('残差')
    axes2[0].set_title('早停RL: 残差变化曲线')
    axes2[0].grid(True, alpha=0.3)
    
    axes2[1].plot(history_tv['tv_norm'], label='TV')
    axes2[1].set_xlabel('迭代')
    axes2[1].set_ylabel('TV范数')
    axes2[1].set_title('TV正则化: TV范数变化')
    axes2[1].grid(True, alpha=0.3)
    
    axes2[2].plot(history_adaptive['lambda'], label='λ(t)')
    axes2[2].set_xlabel('迭代')
    axes2[2].set_ylabel('正则化强度 λ')
    axes2[2].set_title('自适应TV: λ变化曲线')
    axes2[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    print("收敛分析图已保存到 convergence_analysis.png")
    
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    main()
