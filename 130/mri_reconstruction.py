import numpy as np


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    软阈值函数（用于L1正则化）

    Args:
        x: 输入数组
        threshold: 阈值

    Returns:
        软阈值后的数组
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def haar_wavelet_decompose(image: np.ndarray, levels: int = 3) -> list:
    """
    二维Haar小波分解（纯numpy实现，无需外部依赖）

    Args:
        image: 输入图像
        levels: 分解层数

    Returns:
        小波系数列表 [近似, (水平1, 垂直1, 对角1), (水平2, 垂直2, 对角2), ...]
    """
    coeffs = []
    current = image.copy()

    for _ in range(levels):
        h, w = current.shape
        h2, w2 = h // 2, w // 2

        current_padded = np.pad(current, ((0, h % 2), (0, w % 2)), mode='symmetric')

        ll = (current_padded[::2, ::2] + current_padded[::2, 1::2] +
              current_padded[1::2, ::2] + current_padded[1::2, 1::2]) / 2.0
        lh = (current_padded[::2, ::2] - current_padded[::2, 1::2] +
              current_padded[1::2, ::2] - current_padded[1::2, 1::2]) / 2.0
        hl = (current_padded[::2, ::2] + current_padded[::2, 1::2] -
              current_padded[1::2, ::2] - current_padded[1::2, 1::2]) / 2.0
        hh = (current_padded[::2, ::2] - current_padded[::2, 1::2] -
              current_padded[1::2, ::2] + current_padded[1::2, 1::2]) / 2.0

        coeffs.append((lh[:h2, :w2], hl[:h2, :w2], hh[:h2, :w2]))
        current = ll[:h2, :w2]

    coeffs.insert(0, current)
    return coeffs


def haar_wavelet_reconstruct(coeffs: list) -> np.ndarray:
    """
    二维Haar小波重构

    Args:
        coeffs: 小波系数列表

    Returns:
        重构的图像
    """
    current = coeffs[0]

    for i in range(1, len(coeffs)):
        lh, hl, hh = coeffs[i]
        h, w = current.shape
        h2, w2 = h * 2, w * 2

        reconstructed = np.zeros((h2, w2), dtype=current.dtype)

        reconstructed[::2, ::2] = (current + lh + hl + hh) / 2.0
        reconstructed[::2, 1::2] = (current - lh + hl - hh) / 2.0
        reconstructed[1::2, ::2] = (current + lh - hl - hh) / 2.0
        reconstructed[1::2, 1::2] = (current - lh - hl + hh) / 2.0

        current = reconstructed

    return current


def wavelet_soft_threshold(image: np.ndarray, threshold: float, levels: int = 3) -> np.ndarray:
    """
    小波域软阈值（用于稀疏正则化）

    Args:
        image: 输入图像
        threshold: 阈值
        levels: 小波分解层数

    Returns:
        阈值处理后的图像
    """
    coeffs = haar_wavelet_decompose(image, levels)

    coeffs_thresh = [coeffs[0]]
    for i in range(1, len(coeffs)):
        lh, hl, hh = coeffs[i]
        coeffs_thresh.append((
            soft_threshold(lh, threshold),
            soft_threshold(hl, threshold),
            soft_threshold(hh, threshold)
        ))

    return haar_wavelet_reconstruct(coeffs_thresh)


def tv_gradient(image: np.ndarray) -> tuple:
    """
    计算图像的梯度（用于TV正则化）

    Args:
        image: 输入图像

    Returns:
        (水平梯度, 垂直梯度)
    """
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)

    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]

    return grad_x, grad_y


def tv_divergence(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """
    计算梯度的散度（TV的伴随算子）

    Args:
        grad_x: 水平梯度
        grad_y: 垂直梯度

    Returns:
        散度图像
    """
    div = np.zeros_like(grad_x)

    div[:, 1:-1] = grad_x[:, 1:-1] - grad_x[:, :-2]
    div[:, 0] = -grad_x[:, 0]
    div[:, -1] = grad_x[:, -2]

    div[1:-1, :] += grad_y[1:-1, :] - grad_y[:-2, :]
    div[0, :] -= grad_y[0, :]
    div[-1, :] += grad_y[-2, :]

    return div


def tv_prox_gradient(image: np.ndarray, lambda_tv: float, tau: float, iterations: int = 5) -> np.ndarray:
    """
    TV近端算子（使用Chambolle-Pock算法的近似）

    Args:
        image: 输入图像
        lambda_tv: TV正则化参数
        tau: 步长
        iterations: 内部迭代次数

    Returns:
        TV正则化后的图像
    """
    h, w = image.shape
    p_x = np.zeros((h, w), dtype=image.dtype)
    p_y = np.zeros((h, w), dtype=image.dtype)
    sigma = 0.125

    for _ in range(iterations):
        grad_x, grad_y = tv_gradient(image + tau * tv_divergence(p_x, p_y))

        p_x_new = p_x + sigma * grad_x
        p_y_new = p_y + sigma * grad_y

        norm = np.maximum(1.0, np.sqrt(p_x_new**2 + p_y_new**2) / lambda_tv)
        p_x = p_x_new / norm
        p_y = p_y_new / norm

    return image + tau * tv_divergence(p_x, p_y)


def fista_cs_reconstruction(
    kspace_undersampled: np.ndarray,
    mask: np.ndarray,
    lambda_wavelet: float = 0.01,
    lambda_tv: float = 0.005,
    max_iter: int = 100,
    wavelet_levels: int = 3,
    verbose: bool = True
) -> tuple:
    """
    压缩感知MRI重建（FISTA算法 + 小波稀疏 + TV正则化）

    求解问题: min ||M*F(x) - y||^2 + λ1||Wx||_1 + λ2*TV(x)

    Args:
        kspace_undersampled: 欠采样K空间数据
        mask: 采样掩码 (1=采样, 0=未采样)
        lambda_wavelet: 小波L1正则化权重
        lambda_tv: TV正则化权重
        max_iter: 最大迭代次数
        wavelet_levels: 小波分解层数
        verbose: 是否打印进度

    Returns:
        (重建图像, 代价历史)
    """
    nx, ny = kspace_undersampled.shape
    y = kspace_undersampled

    x = np.zeros((nx, ny), dtype=np.complex64)

    x_prev = x.copy()
    t = 1.0

    L = 1.0
    stepsize = 0.9 / L

    cost_history = []

    for k in range(max_iter):
        grad = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
        grad = grad * mask - y
        grad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grad)))
        grad = grad * (nx * ny)

        x_grad = x - stepsize * grad

        x_real = np.real(x_grad)
        x_real = wavelet_soft_threshold(x_real, lambda_wavelet * stepsize, wavelet_levels)
        x_real = tv_prox_gradient(x_real, lambda_tv * stepsize, stepsize)

        x_new = x_real.astype(np.complex64)

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        momentum = (t - 1) / t_new
        x = x_new + momentum * (x_new - x_prev)

        x_prev = x_new
        t = t_new

        if k % 10 == 0 or k == max_iter - 1:
            kspace_current = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
            data_fidelity = 0.5 * np.sum(np.abs(mask * kspace_current - y)**2) / (nx * ny)

            coeffs = haar_wavelet_decompose(np.real(x), wavelet_levels)
            l1_norm = 0
            for i in range(1, len(coeffs)):
                lh, hl, hh = coeffs[i]
                l1_norm += np.sum(np.abs(lh)) + np.sum(np.abs(hl)) + np.sum(np.abs(hh))

            grad_x, grad_y = tv_gradient(np.real(x))
            tv_norm = np.sum(np.sqrt(grad_x**2 + grad_y**2))

            total_cost = data_fidelity + lambda_wavelet * l1_norm + lambda_tv * tv_norm
            cost_history.append(total_cost)

            if verbose and k % 20 == 0:
                print(f"  迭代 {k:4d}: 代价={total_cost:.4e}, 数据保真={data_fidelity:.4e}")

    return np.abs(x), cost_history


def admm_cs_reconstruction(
    kspace_undersampled: np.ndarray,
    mask: np.ndarray,
    lambda_wavelet: float = 0.01,
    lambda_tv: float = 0.005,
    max_iter: int = 50,
    wavelet_levels: int = 3,
    rho: float = 1.0,
    verbose: bool = True
) -> tuple:
    """
    压缩感知MRI重建（ADMM算法 + 小波稀疏 + TV正则化）

    Args:
        kspace_undersampled: 欠采样K空间数据
        mask: 采样掩码
        lambda_wavelet: 小波L1正则化权重
        lambda_tv: TV正则化权重
        max_iter: 最大迭代次数
        wavelet_levels: 小波分解层数
        rho: ADMM惩罚参数
        verbose: 是否打印进度

    Returns:
        (重建图像, 代价历史)
    """
    nx, ny = kspace_undersampled.shape
    y = kspace_undersampled

    x = np.zeros((nx, ny), dtype=np.complex64)
    z_wavelet = np.zeros((nx, ny), dtype=np.float64)
    z_tv = np.zeros((nx, ny), dtype=np.float64)
    u_wavelet = np.zeros((nx, ny), dtype=np.float64)
    u_tv = np.zeros((nx, ny), dtype=np.float64)

    cost_history = []

    for k in range(max_iter):
        rhs_wavelet = z_wavelet - u_wavelet
        rhs_tv = z_tv - u_tv
        rhs = 0.5 * (rhs_wavelet + rhs_tv).astype(np.complex64)

        kspace_rhs = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(rhs)))
        kspace_x = (mask * y + rho * kspace_rhs / (nx * ny)) / (mask + rho / (nx * ny))
        x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_x)))

        x_real = np.real(x)
        z_wavelet = wavelet_soft_threshold(x_real + u_wavelet, lambda_wavelet / rho, wavelet_levels)
        z_tv = tv_prox_gradient(x_real + u_tv, lambda_tv / rho, 1.0 / rho)

        u_wavelet += x_real - z_wavelet
        u_tv += x_real - z_tv

        if k % 5 == 0 or k == max_iter - 1:
            kspace_current = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
            data_fidelity = 0.5 * np.sum(np.abs(mask * kspace_current - y)**2) / (nx * ny)

            coeffs = haar_wavelet_decompose(np.real(x), wavelet_levels)
            l1_norm = 0
            for i in range(1, len(coeffs)):
                lh, hl, hh = coeffs[i]
                l1_norm += np.sum(np.abs(lh)) + np.sum(np.abs(hl)) + np.sum(np.abs(hh))

            grad_x, grad_y = tv_gradient(np.real(x))
            tv_norm = np.sum(np.sqrt(grad_x**2 + grad_y**2))

            total_cost = data_fidelity + lambda_wavelet * l1_norm + lambda_tv * tv_norm
            cost_history.append(total_cost)

            if verbose and k % 10 == 0:
                print(f"  迭代 {k:4d}: 代价={total_cost:.4e}, 数据保真={data_fidelity:.4e}")

    return np.abs(x), cost_history


def compressed_sensing_mri(
    kspace_undersampled: np.ndarray,
    mask: np.ndarray = None,
    method: str = 'fista',
    lambda_wavelet: float = 0.005,
    lambda_tv: float = 0.002,
    max_iter: int = 100,
    wavelet_levels: int = 3,
    verbose: bool = True
) -> np.ndarray:
    """
    压缩感知MRI重建主函数

    Args:
        kspace_undersampled: 欠采样K空间数据（复数）
        mask: 采样掩码（如果为None则自动从kspace推断）
        method: 优化方法 'fista' 或 'admm'
        lambda_wavelet: 小波L1正则化强度
        lambda_tv: TV正则化强度
        max_iter: 最大迭代次数
        wavelet_levels: 小波分解层数
        verbose: 是否显示进度

    Returns:
        重建的MRI图像

    示例:
        >>> # 高度欠采样K空间数据
        >>> reconstructed = compressed_sensing_mri(kspace_undersampled, mask, method='fista',
        ...                                        lambda_wavelet=0.01, lambda_tv=0.005)
    """
    if mask is None:
        mask = (np.abs(kspace_undersampled) > 1e-10).astype(np.float32)

    if method.lower() == 'fista':
        image, _ = fista_cs_reconstruction(
            kspace_undersampled, mask,
            lambda_wavelet=lambda_wavelet,
            lambda_tv=lambda_tv,
            max_iter=max_iter,
            wavelet_levels=wavelet_levels,
            verbose=verbose
        )
    elif method.lower() == 'admm':
        image, _ = admm_cs_reconstruction(
            kspace_undersampled, mask,
            lambda_wavelet=lambda_wavelet,
            lambda_tv=lambda_tv,
            max_iter=max_iter,
            wavelet_levels=wavelet_levels,
            verbose=verbose
        )
    else:
        raise ValueError(f"不支持的方法: {method}，请使用 'fista' 或 'admm'")

    return image


def generate_hanning_window(shape: tuple) -> np.ndarray:
    """
    生成二维汉宁窗

    Args:
        shape: 窗口形状 (Nx, Ny)

    Returns:
        二维汉宁窗数组
    """
    nx, ny = shape
    window_x = np.hanning(nx)
    window_y = np.hanning(ny)
    window_2d = np.outer(window_x, window_y)
    return window_2d


def generate_tukey_window(shape: tuple, alpha: float = 0.5) -> np.ndarray:
    """
    生成二维Tukey窗（余弦锥度窗）

    Args:
        shape: 窗口形状 (Nx, Ny)
        alpha: 锥度参数，0为矩形窗，1为汉宁窗

    Returns:
        二维Tukey窗数组
    """
    nx, ny = shape

    def tukey_1d(n, a):
        if a <= 0:
            return np.ones(n)
        if a >= 1:
            return np.hanning(n)

        n_taper = int(np.round(a * n / 2))
        x = np.linspace(0, 1, n)
        window = np.ones(n)

        for i in range(n_taper):
            window[i] = 0.5 * (1 - np.cos(np.pi * x[i] * 2 / a))

        for i in range(n - n_taper, n):
            window[i] = 0.5 * (1 - np.cos(np.pi * (x[i] - 1 + a) * 2 / a))

        return window

    window_x = tukey_1d(nx, alpha)
    window_y = tukey_1d(ny, alpha)
    window_2d = np.outer(window_x, window_y)
    return window_2d


def generate_gaussian_window(shape: tuple, sigma: float = 0.3) -> np.ndarray:
    """
    生成二维高斯窗

    Args:
        shape: 窗口形状 (Nx, Ny)
        sigma: 高斯标准差（相对于窗口半宽的比例）

    Returns:
        二维高斯窗数组
    """
    nx, ny = shape
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y)
    window_2d = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return window_2d


def apply_kspace_window(
    kspace_data: np.ndarray,
    window_type: str = 'tukey',
    alpha: float = 0.5,
    sigma: float = 0.3
) -> np.ndarray:
    """
    在K空间应用窗函数以抑制吉布斯伪影

    Args:
        kspace_data: K空间复数数据
        window_type: 窗类型，可选 'hanning', 'tukey', 'gaussian'
        alpha: Tukey窗的锥度参数
        sigma: 高斯窗的标准差参数

    Returns:
        加窗后的K空间数据
    """
    shape = kspace_data.shape

    if window_type.lower() == 'hanning':
        window = generate_hanning_window(shape)
    elif window_type.lower() == 'tukey':
        window = generate_tukey_window(shape, alpha)
    elif window_type.lower() == 'gaussian':
        window = generate_gaussian_window(shape, sigma)
    elif window_type.lower() in ['none', 'rectangular']:
        return kspace_data
    else:
        raise ValueError(f"不支持的窗类型: {window_type}")

    return kspace_data * window


def mri_reconstruct_kspace(
    kspace_data: np.ndarray,
    window_type: str = 'none',
    alpha: float = 0.5,
    sigma: float = 0.3
) -> np.ndarray:
    """
    从K空间数据重建MRI图像（二维傅里叶逆变换）

    Args:
        kspace_data: 二维K空间数据，复数数组，形状为 (Nx, Ny)
        window_type: 窗类型，可选 'none', 'hanning', 'tukey', 'gaussian'
        alpha: Tukey窗的锥度参数 (0-1)
        sigma: 高斯窗的标准差参数

    Returns:
        重建的MRI图像，幅度值数组
    """
    if not isinstance(kspace_data, np.ndarray):
        kspace_data = np.array(kspace_data)

    if kspace_data.ndim != 2:
        raise ValueError("K空间数据必须是二维数组")

    if not np.iscomplexobj(kspace_data):
        kspace_data = kspace_data.astype(np.complex64)

    if window_type.lower() != 'none':
        kspace_data = apply_kspace_window(kspace_data, window_type, alpha, sigma)

    image_complex = np.fft.ifftshift(kspace_data)
    image_complex = np.fft.ifft2(image_complex)
    image_complex = np.fft.fftshift(image_complex)

    image_magnitude = np.abs(image_complex)

    return image_magnitude


def generate_phantom_kspace(
    image_size: tuple = (256, 256),
    undersampling_factor: float = 1.0,
    undersampling_pattern: str = 'cartesian'
) -> tuple:
    """
    生成带有明确结构的体模K空间数据（用于演示吉布斯伪影）

    Args:
        image_size: 图像尺寸
        undersampling_factor: 欠采样因子 (1.0为全采样，2.0为50%采样等)
        undersampling_pattern: 欠采样模式 'cartesian' 或 'random'

    Returns:
        (完全采样K空间, 欠采样K空间, 采样掩码)
    """
    nx, ny = image_size
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y)

    image = np.zeros((nx, ny), dtype=np.float32)

    outer_ellipse = ((xx / 0.8)**2 + (yy / 0.6)**2) < 1.0
    image[outer_ellipse] = 0.8

    inner_ellipse = ((xx / 0.4)**2 + (yy / 0.3)**2) < 1.0
    image[inner_ellipse] = 0.3

    for i in range(4):
        angle = np.pi * i / 2
        cx = 0.35 * np.cos(angle)
        cy = 0.35 * np.sin(angle)
        circle = ((xx - cx)**2 + (yy - cy)**2) < (0.08)**2
        image[circle] = 1.0

    image_full = image

    kspace_full = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_full)))
    kspace_full = kspace_full.astype(np.complex64)

    mask = np.ones((nx, ny), dtype=np.float32)

    if undersampling_pattern == 'cartesian':
        step = int(np.round(undersampling_factor))
        for i in range(ny):
            if i % step != 0:
                mask[:, i] = 0
    elif undersampling_pattern == 'random':
        num_keep = int(nx * ny / undersampling_factor)
        indices = np.random.choice(nx * ny, num_keep, replace=False)
        mask_flat = mask.flatten()
        mask_flat[indices] = 1
        mask = mask_flat.reshape((nx, ny))

    center_region = (slice(nx//2 - 10, nx//2 + 10), slice(ny//2 - 10, ny//2 + 10))
    mask[center_region] = 1

    kspace_undersampled = kspace_full * mask

    return kspace_full, kspace_undersampled, mask


def calculate_psnr(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    计算峰值信噪比(PSNR)
    """
    mse = np.mean((image_true - image_test)**2)
    if mse == 0:
        return 100.0
    max_val = np.max(image_true)
    return 20 * np.log10(max_val / np.sqrt(mse))


if __name__ == "__main__":
    print("MRI K空间图像重建 - 完整版")
    print("=" * 70)

    undersampling_factor = 8
    print(f"\n=== 第一部分：生成高度欠采样体模数据 (加速因子: {undersampling_factor}x)")
    kspace_full, kspace_undersampled, mask = generate_phantom_kspace(
        image_size=(128, 128),
        undersampling_factor=undersampling_factor,
        undersampling_pattern='random'
    )

    image_ground_truth = mri_reconstruct_kspace(kspace_full, window_type='none')
    print(f"K空间形状: {kspace_full.shape}")
    print(f"采样率: {mask.mean()*100:.1f}%")

    print(f"\n=== 第二部分：传统重建方法对比")
    image_fft = mri_reconstruct_kspace(kspace_undersampled, window_type='none')
    image_tukey = mri_reconstruct_kspace(kspace_undersampled, window_type='tukey', alpha=0.5)

    psnr_fft = calculate_psnr(image_ground_truth, image_fft)
    psnr_tukey = calculate_psnr(image_ground_truth, image_tukey)

    print(f"直接逆FFT - PSNR: {psnr_fft:.2f} dB")
    print(f"Tukey窗滤波 - PSNR: {psnr_tukey:.2f} dB")

    print(f"\n=== 第三部分：压缩感知重建 (FISTA + 小波 + TV正则化)")
    print("正在执行FISTA算法迭代...")
    image_cs_fista, cost_fista = fista_cs_reconstruction(
        kspace_undersampled, mask,
        lambda_wavelet=0.005,
        lambda_tv=0.002,
        max_iter=150,
        wavelet_levels=3,
        verbose=True
    )
    psnr_fista = calculate_psnr(image_ground_truth, image_cs_fista)
    print(f"压缩感知(FISTA) - PSNR: {psnr_fista:.2f} dB")

    print(f"\n正在执行ADMM算法迭代...")
    image_cs_admm, cost_admm = admm_cs_reconstruction(
        kspace_undersampled, mask,
        lambda_wavelet=0.005,
        lambda_tv=0.002,
        max_iter=80,
        wavelet_levels=3,
        rho=2.0,
        verbose=True
    )
    psnr_admm = calculate_psnr(image_ground_truth, image_cs_admm)
    print(f"压缩感知(ADMM) - PSNR: {psnr_admm:.2f} dB")

    print(f"\n=== 重建质量总结 ===")
    print(f"直接逆FFT:     PSNR = {psnr_fft:.2f} dB")
    print(f"Tukey窗滤波:    PSNR = {psnr_tukey:.2f} dB")
    print(f"压缩感知FISTA:  PSNR = {psnr_fista:.2f} dB")
    print(f"压缩感知ADMM:   PSNR = {psnr_admm:.2f} dB")
    print(f"\n压缩感知提升:   +{psnr_fista - psnr_fft:.2f} dB 相对于直接FFT")

    try:
        import matplotlib.pyplot as plt

        print("\n正在生成可视化结果...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.15)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_ground_truth, cmap='gray', vmin=0, vmax=1)
        ax1.set_title('完全采样\n(真值)', fontsize=10, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(mask, cmap='gray')
        ax2.set_title(f'采样掩码\n{mask.mean()*100:.0f}%采样', fontsize=10)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image_fft, cmap='gray', vmin=0, vmax=1)
        ax3.set_title(f'直接逆FFT\nPSNR: {psnr_fft:.1f} dB', fontsize=10)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(image_tukey, cmap='gray', vmin=0, vmax=1)
        ax4.set_title(f'Tukey窗滤波\nPSNR: {psnr_tukey:.1f} dB', fontsize=10)
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(image_cs_fista, cmap='gray', vmin=0, vmax=1)
        ax5.set_title(f'压缩感知(FISTA)\nPSNR: {psnr_fista:.1f} dB', fontsize=10, fontweight='bold')
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 0])
        kspace_mag = np.log(np.abs(kspace_undersampled) + 1e-10)
        ax6.imshow(kspace_mag, cmap='gray')
        ax6.set_title('欠采样K空间', fontsize=10)
        ax6.axis('off')

        ax7 = fig.add_subplot(gs[1, 1])
        error_fft = np.abs(image_ground_truth - image_fft)
        im7 = ax7.imshow(error_fft, cmap='hot', vmin=0, vmax=0.5)
        ax7.set_title('直接FFT误差', fontsize=10)
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

        ax8 = fig.add_subplot(gs[1, 2])
        error_tukey = np.abs(image_ground_truth - image_tukey)
        im8 = ax8.imshow(error_tukey, cmap='hot', vmin=0, vmax=0.5)
        ax8.set_title('Tukey窗误差', fontsize=10)
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

        ax9 = fig.add_subplot(gs[1, 3])
        error_cs = np.abs(image_ground_truth - image_cs_fista)
        im9 = ax9.imshow(error_cs, cmap='hot', vmin=0, vmax=0.5)
        ax9.set_title('压缩感知误差', fontsize=10)
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

        ax10 = fig.add_subplot(gs[1, 4])
        profile_y = 64
        ax10.plot(image_ground_truth[profile_y, :], 'k-', label='真值', linewidth=2, alpha=0.8)
        ax10.plot(image_fft[profile_y, :], 'b--', label='直接FFT', linewidth=1)
        ax10.plot(image_cs_fista[profile_y, :], 'r-', label='压缩感知', linewidth=1.5)
        ax10.set_title(f'水平线剖面 (y={profile_y})', fontsize=9)
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)

        plt.savefig('compressed_sensing_mri_comparison.png', dpi=150, bbox_inches='tight')
        print("结果已保存为: compressed_sensing_mri_comparison.png")

        fig2, ax = plt.subplots(figsize=(10, 5))
        iterations_fista = np.arange(0, len(cost_fista) * 10, 10)
        iterations_admm = np.arange(0, len(cost_admm) * 5, 5)
        ax.plot(iterations_fista, cost_fista, 'b-o', label='FISTA', markersize=4, linewidth=2)
        ax.plot(iterations_admm, cost_admm, 'r-s', label='ADMM', markersize=4, linewidth=2)
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('目标函数值', fontsize=12)
        ax.set_title('压缩感知算法收敛曲线', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('cs_convergence_curve.png', dpi=150, bbox_inches='tight')
        print("收敛曲线已保存为: cs_convergence_curve.png")

        fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
        x = np.linspace(-1, 1, 100)

        hanning_1d = np.hanning(100)
        tukey_05_1d = generate_tukey_window((100, 100), 0.5)[50, :]
        gaussian_1d = generate_gaussian_window((100, 100), 0.4)[50, :]

        axes[0].plot(x, hanning_1d, 'b-', linewidth=2, label='汉宁窗')
        axes[0].plot(x, tukey_05_1d, 'r-', linewidth=2, label='Tukey (α=0.5)')
        axes[0].plot(x, gaussian_1d, 'm-', linewidth=2, label='高斯 (σ=0.4)')
        axes[0].set_title('K空间窗函数', fontsize=11)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('K空间位置')
        axes[0].set_ylabel('幅度')

        coeffs = haar_wavelet_decompose(image_ground_truth, 2)
        wavelet_viz = np.zeros_like(image_ground_truth)
        h, w = coeffs[0].shape
        wavelet_viz[:h, :w] = coeffs[0] / coeffs[0].max()
        for i in range(1, len(coeffs)):
            lh, hl, hh = coeffs[i]
            scale = 2**(i-1)
            h2, w2 = h // scale, w // scale
            if h2 > 0 and w2 > 0:
                wavelet_viz[:h2, w2:w2*2] = np.abs(lh) / (np.abs(lh).max() + 1e-10)
                wavelet_viz[h2:h2*2, :w2] = np.abs(hl) / (np.abs(hl).max() + 1e-10)
                wavelet_viz[h2:h2*2, w2:w2*2] = np.abs(hh) / (np.abs(hh).max() + 1e-10)
        axes[1].imshow(wavelet_viz, cmap='viridis')
        axes[1].set_title('Haar小波分解(2层)', fontsize=11)
        axes[1].axis('off')

        grad_x, grad_y = tv_gradient(image_ground_truth)
        tv_mag = np.sqrt(grad_x**2 + grad_y**2)
        im = axes[2].imshow(tv_mag, cmap='hot')
        axes[2].set_title('图像梯度 (TV正则化项)', fontsize=11)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig('cs_regularization_visualization.png', dpi=150, bbox_inches='tight')
        print("正则化可视化已保存为: cs_regularization_visualization.png")

    except ImportError as e:
        print(f"\n未安装matplotlib或出错，跳过可视化。")
        print(f"错误: {e}")
        print("可使用: pip install matplotlib 安装可视化库。")
