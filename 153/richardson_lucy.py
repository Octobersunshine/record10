import numpy as np
from scipy.ndimage import convolve, fftconvolve
from scipy.signal import fftconvolve as signal_fftconvolve


def richardson_lucy(image, psf, iterations=50, use_fft=True, clip=True):
    """
    Richardson-Lucy反卷积算法
    
    参数:
        image: 输入的模糊图像 (2D numpy数组)
        psf: 点扩散函数 (2D numpy数组)
        iterations: 迭代次数
        use_fft: 是否使用FFT加速卷积
        clip: 是否裁剪结果到[0,1]范围
    
    返回:
        恢复的图像
    """
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    
    psf_mirror = psf[::-1, ::-1]
    
    estimate = np.ones_like(image) * image.mean()
    
    for i in range(iterations):
        if use_fft:
            convolved = signal_fftconvolve(estimate, psf, mode='same')
        else:
            convolved = convolve(estimate, psf, mode='constant')
        
        relative_blur = image / (convolved + 1e-12)
        
        if use_fft:
            error_estimate = signal_fftconvolve(relative_blur, psf_mirror, mode='same')
        else:
            error_estimate = convolve(relative_blur, psf_mirror, mode='constant')
        
        estimate *= error_estimate
        
        if clip:
            estimate = np.clip(estimate, 0, 1)
    
    return estimate


def richardson_lucy_with_regularization(image, psf, iterations=50, lambda_reg=0.01, use_fft=True):
    """
    带正则化的Richardson-Lucy反卷积（Total Variation正则化）
    
    参数:
        image: 输入的模糊图像
        psf: 点扩散函数
        iterations: 迭代次数
        lambda_reg: 正则化参数
        use_fft: 是否使用FFT
    
    返回:
        恢复的图像
    """
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    
    psf_mirror = psf[::-1, ::-1]
    estimate = np.ones_like(image) * image.mean()
    
    for i in range(iterations):
        if use_fft:
            convolved = signal_fftconvolve(estimate, psf, mode='same')
        else:
            convolved = convolve(estimate, psf, mode='constant')
        
        relative_blur = image / (convolved + 1e-12)
        
        if use_fft:
            error_estimate = signal_fftconvolve(relative_blur, psf_mirror, mode='same')
        else:
            error_estimate = convolve(relative_blur, psf_mirror, mode='constant')
        
        grad_x, grad_y = np.gradient(estimate)
        tv_grad = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        reg_term = 1 - lambda_reg * (np.gradient(grad_x / tv_grad)[0] + np.gradient(grad_y / tv_grad)[1])
        
        estimate *= error_estimate * reg_term
        estimate = np.clip(estimate, 0, 1)
    
    return estimate


def generate_gaussian_psf(size=21, sigma=3.0):
    """
    生成高斯点扩散函数
    
    参数:
        size: PSF尺寸 (奇数)
        sigma: 高斯标准差
    
    返回:
        归一化的高斯PSF
    """
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    
    gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return gaussian / gaussian.sum()


def generate_atmospheric_psf(size=31, seeing=2.5):
    """
    生成模拟大气湍流的PSF (Kolmogorov模型近似)
    
    参数:
        size: PSF尺寸
        seeing: 视宁度 (像素单位)
    
    返回:
        大气湍流PSF
    """
    x = np.arange(0, size, 1, float) - size // 2
    y = x[:, np.newaxis]
    r = np.sqrt(x**2 + y**2)
    
    seeing_pix = seeing
    alpha = 1.0 / (2 * seeing_pix**2)
    psf = np.exp(-(r**2)**(5/6) * alpha)
    psf = psf / psf.sum()
    
    return psf


def add_noise(image, snr=30, noise_type='poisson'):
    """
    给图像添加噪声
    
    参数:
        image: 输入图像
        snr: 信噪比
        noise_type: 'poisson' 或 'gaussian'
    
    返回:
        含噪图像
    """
    if noise_type == 'poisson':
        peak = image.max()
        noisy = np.random.poisson(image * snr / peak) * peak / snr
    else:
        sigma = image.std() / snr
        noisy = image + np.random.normal(0, sigma, image.shape)
    
    return np.clip(noisy, 0, 1)


def blur_image(image, psf, add_noise_flag=True, snr=30):
    """
    用PSF模糊图像并可选添加噪声
    
    参数:
        image: 原始清晰图像
        psf: 点扩散函数
        add_noise_flag: 是否添加噪声
        snr: 信噪比
    
    返回:
        模糊图像
    """
    blurred = signal_fftconvolve(image, psf, mode='same')
    
    if add_noise_flag:
        blurred = add_noise(blurred, snr=snr)
    
    return np.clip(blurred, 0, 1)


def compute_psnr(original, restored):
    """
    计算峰值信噪比(PSNR)
    
    参数:
        original: 原始图像
        restored: 恢复图像
    
    返回:
        PSNR值 (dB)
    """
    mse = np.mean((original - restored)**2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def main():
    """
    示例：演示Richardson-Lucy反卷积
    """
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Richardson-Lucy反卷积 - 天文图像恢复演示")
    print("=" * 60)
    
    img_size = 128
    original = np.zeros((img_size, img_size))
    
    num_stars = 30
    for _ in range(num_stars):
        x = np.random.randint(10, img_size - 10)
        y = np.random.randint(10, img_size - 10)
        brightness = np.random.uniform(0.3, 1.0)
        radius = np.random.randint(1, 4)
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if 0 <= x + i < img_size and 0 <= y + j < img_size:
                    dist = np.sqrt(i**2 + j**2)
                    if dist <= radius:
                        original[y + j, x + i] = max(original[y + j, x + i], 
                                                     brightness * (1 - dist / (radius + 1)))
    
    original = np.clip(original, 0, 1)
    
    print(f"\n生成模拟天文图像: {img_size}x{img_size} 像素, {num_stars} 颗恒星")
    
    psf_size = 31
    seeing = 3.5
    psf = generate_atmospheric_psf(size=psf_size, seeing=seeing)
    print(f"生成大气湍流PSF: 尺寸={psf_size}x{psf_size}, 视宁度={seeing}像素")
    
    blurred = blur_image(original, psf, add_noise_flag=True, snr=40)
    print("生成模糊图像（含噪声）")
    
    iterations = 100
    print(f"\n开始Richardson-Lucy反卷积, 迭代次数: {iterations}")
    
    restored = richardson_lucy(blurred, psf, iterations=iterations, use_fft=True)
    print("标准RL反卷积完成")
    
    lambda_reg = 0.005
    restored_reg = richardson_lucy_with_regularization(blurred, psf, iterations=iterations, lambda_reg=lambda_reg)
    print(f"带TV正则化的RL反卷积完成 (λ={lambda_reg})")
    
    psnr_blurred = compute_psnr(original, blurred)
    psnr_restored = compute_psnr(original, restored)
    psnr_restored_reg = compute_psnr(original, restored_reg)
    
    print(f"\n图像质量评估 (PSNR):")
    print(f"  模糊图像: {psnr_blurred:.2f} dB")
    print(f"  RL恢复:   {psnr_restored:.2f} dB")
    print(f"  RL+TV:    {psnr_restored_reg:.2f} dB")
    print(f"  改善:     {psnr_restored - psnr_blurred:.2f} dB / {psnr_restored_reg - psnr_blurred:.2f} dB")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original, cmap='gray', origin='lower')
    axes[0, 0].set_title('原始清晰图像')
    
    axes[0, 1].imshow(psf, cmap='hot', origin='lower')
    axes[0, 1].set_title('大气湍流PSF')
    
    axes[0, 2].imshow(blurred, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'模糊图像\nPSNR: {psnr_blurred:.2f} dB')
    
    axes[1, 0].imshow(restored, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'RL恢复 ({iterations}次迭代)\nPSNR: {psnr_restored:.2f} dB')
    
    axes[1, 1].imshow(restored_reg, cmap='gray', origin='lower')
    axes[1, 1].set_title(f'RL+TV正则化恢复\nPSNR: {psnr_restored_reg:.2f} dB')
    
    diff = np.abs(restored - original)
    axes[1, 2].imshow(diff, cmap='hot', origin='lower')
    axes[1, 2].set_title('恢复误差图')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('rl_deconvolution_demo.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 rl_deconvolution_demo.png")
    
    plt.show()


if __name__ == "__main__":
    main()
