import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Callable, Tuple, List, Union


def fft2_normalized(x: np.ndarray) -> np.ndarray:
    """
    归一化的2D傅里叶变换，严格保持能量守恒。
    
    通过除以 sqrt(N) 来保证 Parseval 恒成立，即：
    sum(|x|^2) = sum(|FFT(x)|^2)
    
    数学推导：
    numpy.fft.fft2 默认无归一化因子，numpy.fft.ifft2 默认有 1/N 因子
    能量守恒要求: sum(|x|²) = (1/N) * sum(|FFT(x)|²)
    因此归一化: FFT_normalized(x) = FFT(x) / sqrt(N)
    则: sum(|FFT_normalized(x)|²) = (1/N) * sum(|FFT(x)|²) = sum(|x|²)
    
    参数:
        x: 输入的2D数组
        
    返回:
        归一化并fftshift后的傅里叶变换结果
    """
    n = np.prod(x.shape)
    return np.fft.fftshift(np.fft.fft2(x)) / np.sqrt(n)


def ifft2_normalized(x: np.ndarray) -> np.ndarray:
    """
    归一化的2D逆傅里叶变换，严格保持能量守恒。
    
    通过乘以 sqrt(N) 来保证 Parseval 恒成立，即：
    sum(|X|^2) = sum(|IFFT(X)|^2)
    
    数学推导：
    IFFT_normalized(X) = IFFT(X) * sqrt(N)
    则: sum(|IFFT_normalized(X)|²) = N * sum(|IFFT(X)|²) 
        = N * (1/N²) * sum(|X|²) * N  （因为IFFT有1/N因子）
        = sum(|X|²)
    
    参数:
        x: 输入的2D频域数组（已fftshift）
        
    返回:
        归一化后的空域结果
    """
    n = np.prod(x.shape)
    return np.fft.ifft2(np.fft.ifftshift(x)) * np.sqrt(n)


def gerchberg_saxton(
    intensity_measure: Union[np.ndarray, List],
    n_iter: int = 100,
    initial_phase: Optional[np.ndarray] = None,
    domain_constraint: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    verbose: bool = True,
    energy_tracking: bool = True
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[float], List[float]],
    Tuple[np.ndarray, np.ndarray, List[float]]
]:
    """
    Gerchberg-Saxton算法进行相位恢复（严格能量守恒版本）。
    
    算法原理：
        G-S算法是一种迭代相位恢复算法，通过在空域和频域之间交替投影，
        利用已知的强度约束来恢复丢失的相位信息。
        
        迭代流程：
        1. 从频域开始（已知强度，随机初始相位）
        2. 逆傅里叶变换到空域（能量守恒）
        3. 应用空域约束（如支持域）
        4. 傅里叶变换回频域（能量守恒）
        5. 应用频域振幅约束 + 单次能量校正
        6. 重复步骤2-5直到收敛
    
    能量守恒优化说明：
        - 使用归一化FFT/IFFT保证变换过程能量严格守恒
        - 仅在频域振幅替换后进行一次能量校正，避免多次校正导致的波动
        - 校正策略：complex_field *= sqrt(target_energy / current_energy)
        - 这种设计确保迭代过程中能量稳定，加快收敛速度
    
    参数:
        intensity_measure: 测量的强度分布（如远场衍射强度）
        n_iter: 迭代次数，默认100
        initial_phase: 初始相位，默认为[0, 2π)的随机相位
        domain_constraint: 空域约束函数，输入振幅，输出约束后的振幅
        verbose: 是否打印迭代进度信息
        energy_tracking: 是否追踪能量变化历史
    
    返回:
        recovered_phase: 恢复的相位分布 [rad]
        recovered_amplitude: 恢复的振幅分布
        error_history: 每次迭代的均方误差历史
        energy_history: 每次迭代的能量历史（仅当energy_tracking=True时）
    """
    intensity = np.asarray(intensity_measure, dtype=np.float64)
    measured_amplitude = np.sqrt(intensity)
    
    target_energy = np.sum(intensity)
    n_pixels = np.prod(intensity.shape)
    
    if initial_phase is None:
        phase = np.random.rand(*intensity.shape) * 2 * np.pi
    else:
        phase = np.asarray(initial_phase, dtype=np.float64)
    
    complex_field = measured_amplitude * np.exp(1j * phase)
    
    error_history = []
    energy_history = [] if energy_tracking else None
    
    for i in range(n_iter):
        spatial_field = ifft2_normalized(complex_field)
        spatial_amplitude = np.abs(spatial_field)
        spatial_phase = np.angle(spatial_field)
        
        if domain_constraint is not None:
            spatial_amplitude = domain_constraint(spatial_amplitude)
        
        complex_field = fft2_normalized(
            spatial_amplitude * np.exp(1j * spatial_phase)
        )
        
        current_amplitude = np.abs(complex_field)
        current_phase = np.angle(complex_field)
        
        error = np.mean((current_amplitude - measured_amplitude) ** 2)
        error_history.append(error)
        
        complex_field = measured_amplitude * np.exp(1j * current_phase)
        
        current_energy = np.sum(np.abs(complex_field) ** 2)
        if current_energy > 0:
            complex_field *= np.sqrt(target_energy / current_energy)
        
        if energy_tracking:
            current_total_energy = np.sum(np.abs(complex_field) ** 2)
            energy_history.append(current_total_energy)
        
        if verbose and (i + 1) % 10 == 0:
            energy_str = f", 能量: {energy_history[-1]:.4f}" if energy_tracking else ""
            print(f"迭代 {i+1}/{n_iter}, 误差: {error:.6f}{energy_str}")
    
    final_spatial = ifft2_normalized(complex_field)
    recovered_amplitude = np.abs(final_spatial)
    recovered_phase = np.angle(final_spatial)
    
    if energy_tracking:
        return recovered_phase, recovered_amplitude, error_history, energy_history
    return recovered_phase, recovered_amplitude, error_history


def gerchberg_saxton_near_field(
    intensity_near: Union[np.ndarray, List],
    intensity_far: Union[np.ndarray, List],
    n_iter: int = 100,
    initial_phase: Optional[np.ndarray] = None,
    verbose: bool = True,
    energy_tracking: bool = True
) -> Union[
    Tuple[np.ndarray, List[float], List[float]],
    Tuple[np.ndarray, List[float]]
]:
    """
    近场-远场G-S算法，同时利用近场和远场强度约束（严格能量守恒版本）。
    
    算法原理：
        当同时可以测量近场和远场的强度分布时，可以利用双重约束
        来获得更稳定和准确的相位恢复结果。
        
        迭代流程：
        1. 从近场开始（已知近场强度，随机初始相位）
        2. 傅里叶变换到远场（能量守恒）
        3. 应用远场振幅约束
        4. 逆傅里叶变换回近场（能量守恒）
        5. 应用近场振幅约束 + 单次能量校正
        6. 重复步骤2-5直到收敛
    
    能量守恒优化说明：
        - 预先将远场振幅归一化到目标能量，避免迭代中不匹配
        - FFT/IFFT变换过程严格能量守恒
        - 仅在每次迭代结束时进行一次能量校正，确保稳定
        - 双重约束 + 能量守恒 = 更快的收敛速度和更稳定的结果
    
    参数:
        intensity_near: 近场强度分布
        intensity_far: 远场强度分布
        n_iter: 迭代次数，默认100
        initial_phase: 初始相位，默认为随机相位
        verbose: 是否打印进度信息
        energy_tracking: 是否追踪能量变化
    
    返回:
        recovered_phase: 恢复的相位分布 [rad]
        error_history: 误差历史
        energy_history: 能量历史（仅当energy_tracking=True时）
    """
    amp_near = np.sqrt(np.asarray(intensity_near, dtype=np.float64))
    amp_far = np.sqrt(np.asarray(intensity_far, dtype=np.float64))
    
    target_energy = np.sum(intensity_near)
    
    far_energy_init = np.sum(amp_far ** 2)
    if far_energy_init > 0:
        amp_far = amp_far * np.sqrt(target_energy / far_energy_init)
    
    if initial_phase is None:
        phase = np.random.rand(*amp_near.shape) * 2 * np.pi
    else:
        phase = np.asarray(initial_phase, dtype=np.float64)
    
    complex_field = amp_near * np.exp(1j * phase)
    
    error_history = []
    energy_history = [] if energy_tracking else None
    
    for i in range(n_iter):
        far_field = fft2_normalized(complex_field)
        far_phase = np.angle(far_field)
        
        far_field = amp_far * np.exp(1j * far_phase)
        
        complex_field = ifft2_normalized(far_field)
        near_phase = np.angle(complex_field)
        near_amp = np.abs(complex_field)
        
        error = np.mean((near_amp - amp_near) ** 2)
        error_history.append(error)
        
        complex_field = amp_near * np.exp(1j * near_phase)
        
        current_energy = np.sum(np.abs(complex_field) ** 2)
        if current_energy > 0:
            complex_field *= np.sqrt(target_energy / current_energy)
        
        if energy_tracking:
            current_total_energy = np.sum(np.abs(complex_field) ** 2)
            energy_history.append(current_total_energy)
        
        if verbose and (i + 1) % 10 == 0:
            energy_str = f", 能量: {energy_history[-1]:.4f}" if energy_tracking else ""
            print(f"迭代 {i+1}/{n_iter}, 误差: {error:.6f}{energy_str}")
    
    recovered_phase = np.angle(complex_field)
    if energy_tracking:
        return recovered_phase, error_history, energy_history
    return recovered_phase, error_history


def create_test_image(
    size: int = 256,
    pattern_type: str = 'double_slit'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建测试用的复振幅分布。
    
    参数:
        size: 图像大小（像素），默认256
        pattern_type: 图案类型
            - 'double_slit': 双缝干涉图案
            - 'lens': 高斯振幅 + 透镜相位
            - 'gaussian': 高斯光束
            - 'letter': 字母'A'图案
    
    返回:
        amplitude: 振幅分布
        phase: 相位分布 [rad]
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    if pattern_type == 'double_slit':
        amplitude = np.zeros((size, size))
        slit_width = 0.05
        slit_separation = 0.3
        mask1 = (np.abs(X - slit_separation/2) < slit_width) & (np.abs(Y) < 0.4)
        mask2 = (np.abs(X + slit_separation/2) < slit_width) & (np.abs(Y) < 0.4)
        amplitude[mask1 | mask2] = 1.0
        phase = np.zeros((size, size))
        
    elif pattern_type == 'lens':
        amplitude = np.exp(-(X**2 + Y**2) / 0.5)
        phase = -(X**2 + Y**2) * 20
        
    elif pattern_type == 'gaussian':
        amplitude = np.exp(-(X**2 + Y**2) / 0.3)
        phase = np.zeros((size, size))
        
    elif pattern_type == 'letter':
        amplitude = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                xi, yi = X[i, j], Y[i, j]
                if (abs(xi) < 0.6) and (abs(yi - 0.3) < 0.05):
                    amplitude[i, j] = 1
                elif (abs(xi) < 0.6) and (abs(yi) < 0.05):
                    amplitude[i, j] = 1
                elif (abs(xi - 0.5) < 0.05) and (yi > -0.6) and (yi < 0.4):
                    amplitude[i, j] = 1
                elif (abs(xi + 0.5) < 0.05) and (yi > -0.6) and (yi < 0.4):
                    amplitude[i, j] = 1
        phase = np.sin(X * 10) * np.cos(Y * 10) * 2
        
    else:
        raise ValueError(f"未知的图案类型: {pattern_type}")
    
    return amplitude, phase


def compute_far_field(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    计算远场衍射强度（使用归一化FFT保持能量守恒）。
    
    远场衍射对应于夫琅禾费衍射，可通过傅里叶变换计算。
    
    参数:
        amplitude: 输入振幅分布
        phase: 输入相位分布 [rad]
    
    返回:
        far_intensity: 远场强度分布
    """
    complex_field = amplitude * np.exp(1j * phase)
    far_field = fft2_normalized(complex_field)
    far_intensity = np.abs(far_field) ** 2
    return far_intensity


def support_constraint(amplitude: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    简单的支持域约束，将低于阈值的振幅置零。
    
    支持域约束是相位恢复中常用的先验信息，假设物体在
    一定区域外的振幅为零。
    
    参数:
        amplitude: 输入振幅分布
        threshold: 阈值，低于此值的振幅被置零
    
    返回:
        约束后的振幅分布
    """
    return np.where(amplitude > threshold, amplitude, 0)


def demo() -> None:
    """
    G-S算法演示：从远场衍射强度恢复相位。
    """
    print("=" * 60)
    print("Gerchberg-Saxton相位恢复算法演示")
    print("=" * 60)
    
    size = 256
    n_iter = 200
    pattern_type = 'double_slit'
    
    print(f"\n1. 创建测试图案: {pattern_type}")
    amp_true, phase_true = create_test_image(size, pattern_type)
    
    print(f"2. 计算远场衍射强度")
    far_intensity = compute_far_field(amp_true, phase_true)
    
    print(f"3. 使用G-S算法恢复相位...")
    recovered_phase, recovered_amp, error_history, energy_history = gerchberg_saxton(
        far_intensity, n_iter=n_iter, verbose=True
    )
    
    recovered_far = compute_far_field(recovered_amp, recovered_phase)
    
    print("\n4. 可视化结果")
    visualize_results(amp_true, phase_true, far_intensity, 
                      recovered_amp, recovered_phase, recovered_far, error_history)


def visualize_results(
    amp_true: np.ndarray,
    phase_true: np.ndarray,
    far_intensity: np.ndarray,
    recovered_amp: np.ndarray,
    recovered_phase: np.ndarray,
    recovered_far: np.ndarray,
    error_history: List[float]
) -> None:
    """
    可视化相位恢复结果。
    
    参数:
        amp_true: 真实振幅
        phase_true: 真实相位
        far_intensity: 测量的远场强度
        recovered_amp: 恢复的振幅
        recovered_phase: 恢复的相位
        recovered_far: 恢复的远场强度
        error_history: 误差历史
    """
    fig = plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(amp_true, cmap='gray')
    plt.title('原始振幅')
    plt.colorbar()
    
    plt.subplot(2, 4, 2)
    plt.imshow(phase_true, cmap='jet')
    plt.title('原始相位')
    plt.colorbar()
    
    plt.subplot(2, 4, 3)
    plt.imshow(np.log(far_intensity + 1), cmap='gray')
    plt.title('远场强度 (对数)')
    plt.colorbar()
    
    plt.subplot(2, 4, 5)
    plt.imshow(recovered_amp, cmap='gray')
    plt.title('恢复的振幅')
    plt.colorbar()
    
    plt.subplot(2, 4, 6)
    plt.imshow(recovered_phase, cmap='jet')
    plt.title('恢复的相位')
    plt.colorbar()
    
    plt.subplot(2, 4, 7)
    plt.imshow(np.log(recovered_far + 1), cmap='gray')
    plt.title('恢复的远场强度 (对数)')
    plt.colorbar()
    
    plt.subplot(2, 4, 8)
    plt.plot(error_history)
    plt.yscale('log')
    plt.xlabel('迭代次数')
    plt.ylabel('均方误差')
    plt.title('收敛曲线')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gs_phase_recovery.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存至 gs_phase_recovery.png")
    plt.show()


def demo_near_far() -> None:
    """
    近场-远场G-S算法演示。
    """
    print("=" * 60)
    print("近场-远场Gerchberg-Saxton相位恢复算法演示")
    print("=" * 60)
    
    size = 256
    n_iter = 200
    
    print("\n1. 创建测试图案")
    amp_near, phase_true = create_test_image(size, 'lens')
    intensity_near = amp_near ** 2
    
    print("2. 计算远场强度")
    intensity_far = compute_far_field(amp_near, phase_true)
    
    print("3. 使用近场-远场G-S算法恢复相位...")
    recovered_phase, error_history, energy_history = gerchberg_saxton_near_field(
        intensity_near, intensity_far, n_iter=n_iter, verbose=True
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(amp_near, cmap='gray')
    axes[0, 0].set_title('近场振幅')
    
    axes[0, 1].imshow(phase_true, cmap='jet')
    axes[0, 1].set_title('真实相位')
    
    axes[0, 2].imshow(np.log(intensity_far + 1), cmap='gray')
    axes[0, 2].set_title('远场强度')
    
    axes[1, 1].imshow(recovered_phase, cmap='jet')
    axes[1, 1].set_title('恢复的相位')
    
    axes[1, 2].plot(error_history)
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_xlabel('迭代次数')
    axes[1, 2].set_ylabel('误差')
    axes[1, 2].set_title('收敛曲线')
    axes[1, 2].grid(True)
    
    axes[1, 0].imshow(np.abs(recovered_phase - phase_true), cmap='jet')
    axes[1, 0].set_title('相位误差')
    
    plt.tight_layout()
    plt.savefig('gs_near_far.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存至 gs_near_far.png")
    plt.show()


def defocus_phase(
    size: int,
    defocus_waves: float,
    wavelength: float = 1.0,
    pixel_size: float = 1.0
) -> np.ndarray:
    """
    生成离焦相位传递函数。
    
    离焦相位在频域（光瞳面）的表达式为：
        W(ρ) = (2π/λ) * (Δz / 2) * (NA * ρ)²
    
    简化为：
        W(u, v) = defocus_waves * 2π * (u² + v²)
    
    参数:
        size: 图像大小（像素）
        defocus_waves: 离焦量（波长数），正值表示过聚焦，负值表示欠聚焦
        wavelength: 波长（任意单位，仅影响相位尺度，默认1.0）
        pixel_size: 像素大小（与波长同单位，默认1.0）
    
    返回:
        离焦相位分布 [rad]
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    r_squared = X**2 + Y**2
    
    phase = defocus_waves * 2 * np.pi * r_squared
    return phase


def generate_defocused_intensities(
    amplitude: np.ndarray,
    phase: np.ndarray,
    defocus_values: List[float],
    snr: Optional[float] = None
) -> Tuple[List[np.ndarray], List[float]]:
    """
    生成不同离焦量下的强度分布（相位多样性数据）。
    
    对于每个离焦量，计算过程为：
        1. 物面复振幅: A(x,y) * exp[jφ(x,y)]
        2. 傅里叶变换到光瞳面
        3. 应用离焦相位: exp[jW_defocus(u,v)]
        4. 逆傅里叶变换回像面
        5. 取模平方得到强度
    
    参数:
        amplitude: 物面振幅分布
        phase: 物面相位分布 [rad]
        defocus_values: 离焦量列表（波长数），如 [0, -1, 1] 表示焦面、欠焦、过焦
        snr: 信噪比（dB），如果为None则不添加噪声
    
    返回:
        intensities: 各离焦量下的强度分布列表
        defocus_values: 对应的离焦量列表
    """
    size = amplitude.shape[0]
    intensities = []
    
    for defocus_waves in defocus_values:
        defocus = defocus_phase(size, defocus_waves)
        
        obj_field = amplitude * np.exp(1j * phase)
        pupil_field = fft2_normalized(obj_field)
        
        defocused_pupil = pupil_field * np.exp(1j * defocus)
        image_field = ifft2_normalized(defocused_pupil)
        intensity = np.abs(image_field) ** 2
        
        if snr is not None:
            signal_power = np.mean(intensity)
            noise_power = signal_power / (10 ** (snr / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), intensity.shape)
            intensity = np.maximum(intensity + noise, 0)
        
        intensities.append(intensity)
    
    return intensities, defocus_values


def gerchberg_saxton_phase_diversity(
    intensities: Union[List[np.ndarray], List[List]],
    defocus_values: List[float],
    n_iter: int = 100,
    initial_phase: Optional[np.ndarray] = None,
    domain_constraint: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    verbose: bool = True,
    energy_tracking: bool = True
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[float], List[float]],
    Tuple[np.ndarray, np.ndarray, List[float]]
]:
    """
    相位多样性G-S算法，利用多个离焦强度面恢复相位。
    
    算法原理：
        相位多样性通过同时利用多个不同离焦量下的强度测量，
        提供更丰富的约束信息，显著提高：
        1. 收敛稳定性 - 避免陷入局部最优
        2. 抗噪声能力 - 多幅图像平均降低噪声影响
        3. 恢复精度 - 多重约束减少解的歧义性
    
    迭代流程：
        对于每个离焦通道 k:
            1. 当前估计的物面复振幅传播到离焦面 k
            2. 计算与测量强度的误差
        更新物面相位（所有通道的平均贡献）
        应用空域约束
        能量校正
    
    参数:
        intensities: 各离焦量下的强度分布列表
        defocus_values: 对应的离焦量列表（波长数）
        n_iter: 迭代次数，默认100
        initial_phase: 初始相位，默认为随机相位
        domain_constraint: 空域约束函数
        verbose: 是否打印进度信息
        energy_tracking: 是否追踪能量变化
    
    返回:
        recovered_phase: 恢复的物面相位分布 [rad]
        recovered_amplitude: 恢复的物面振幅分布
        error_history: 每次迭代的总误差历史
        energy_history: 每次迭代的能量历史（仅当energy_tracking=True时）
    """
    n_defocus = len(intensities)
    assert n_defocus == len(defocus_values), \
        "intensities和defocus_values长度必须一致"
    assert n_defocus >= 2, "相位多样性至少需要2个离焦面"
    
    intensities = [np.asarray(i, dtype=np.float64) for i in intensities]
    measured_amplitudes = [np.sqrt(i) for i in intensities]
    size = intensities[0].shape[0]
    
    defocus_phases = [
        defocus_phase(size, d) for d in defocus_values
    ]
    
    target_energy = np.sum(intensities[0])
    
    if initial_phase is None:
        phase = np.random.rand(size, size) * 2 * np.pi
    else:
        phase = np.asarray(initial_phase, dtype=np.float64)
    
    amp_init = measured_amplitudes[0].copy()
    init_energy = np.sum(amp_init ** 2)
    if init_energy > 0:
        amp_init *= np.sqrt(target_energy / init_energy)
    
    complex_obj = amp_init * np.exp(1j * phase)
    
    error_history = []
    energy_history = [] if energy_tracking else None
    
    for i in range(n_iter):
        total_phase_update = np.zeros((size, size), dtype=np.float64)
        total_error = 0.0
        
        for k in range(n_defocus):
            obj_field = complex_obj
            pupil_field = fft2_normalized(obj_field)
            
            defocused_pupil = pupil_field * np.exp(1j * defocus_phases[k])
            image_field = ifft2_normalized(defocused_pupil)
            image_phase = np.angle(image_field)
            current_amp = np.abs(image_field)
            
            error = np.mean((current_amp - measured_amplitudes[k]) ** 2)
            total_error += error
            
            constrained_image = measured_amplitudes[k] * np.exp(1j * image_phase)
            
            constrained_pupil = fft2_normalized(constrained_image)
            constrained_pupil_corrected = constrained_pupil * np.exp(-1j * defocus_phases[k])
            
            updated_obj = ifft2_normalized(constrained_pupil_corrected)
            total_phase_update += np.angle(updated_obj)
        
        avg_phase_update = total_phase_update / n_defocus
        current_amp = np.abs(complex_obj)
        
        if domain_constraint is not None:
            current_amp = domain_constraint(current_amp)
        
        complex_obj = current_amp * np.exp(1j * avg_phase_update)
        
        total_error /= n_defocus
        error_history.append(total_error)
        
        current_energy = np.sum(np.abs(complex_obj) ** 2)
        if current_energy > 0:
            complex_obj *= np.sqrt(target_energy / current_energy)
        
        if energy_tracking:
            current_total_energy = np.sum(np.abs(complex_obj) ** 2)
            energy_history.append(current_total_energy)
        
        if verbose and (i + 1) % 10 == 0:
            energy_str = f", 能量: {energy_history[-1]:.4f}" if energy_tracking else ""
            print(f"迭代 {i+1}/{n_iter}, 平均误差: {total_error:.6f}{energy_str}")
    
    recovered_amplitude = np.abs(complex_obj)
    recovered_phase = np.angle(complex_obj)
    
    if energy_tracking:
        return recovered_phase, recovered_amplitude, error_history, energy_history
    return recovered_phase, recovered_amplitude, error_history


def demo_phase_diversity() -> None:
    """
    相位多样性G-S算法演示。
    """
    print("=" * 70)
    print("相位多样性Gerchberg-Saxton相位恢复算法演示")
    print("=" * 70)
    
    size = 128
    n_iter = 200
    pattern_type = 'lens'
    defocus_values = [0, -1.5, 1.5]
    snr = 20.0
    
    print(f"\n1. 创建测试图案: {pattern_type}")
    amp_true, phase_true = create_test_image(size, pattern_type)
    
    print(f"2. 生成 {len(defocus_values)} 个离焦强度面 (离焦量: {defocus_values} 波长)")
    print(f"   添加噪声: SNR = {snr} dB")
    intensities, defocus_used = generate_defocused_intensities(
        amp_true, phase_true, defocus_values, snr=snr
    )
    
    print(f"\n3. 使用相位多样性G-S算法恢复相位...")
    print(f"   离焦通道数: {len(intensities)}")
    recovered_phase, recovered_amp, error_history, energy_history = \
        gerchberg_saxton_phase_diversity(
            intensities, defocus_used, n_iter=n_iter, verbose=True
        )
    
    print("\n4. 可视化结果")
    visualize_phase_diversity_results(
        amp_true, phase_true, intensities, defocus_used,
        recovered_amp, recovered_phase, error_history
    )


def visualize_phase_diversity_results(
    amp_true: np.ndarray,
    phase_true: np.ndarray,
    intensities: List[np.ndarray],
    defocus_values: List[float],
    recovered_amp: np.ndarray,
    recovered_phase: np.ndarray,
    error_history: List[float]
) -> None:
    """
    可视化相位多样性算法结果。
    
    参数:
        amp_true: 真实振幅
        phase_true: 真实相位
        intensities: 各离焦面强度
        defocus_values: 离焦量列表
        recovered_amp: 恢复的振幅
        recovered_phase: 恢复的相位
        error_history: 误差历史
    """
    n_defocus = len(intensities)
    n_cols = max(4, n_defocus + 2)
    
    fig = plt.figure(figsize=(5 * n_cols, 10))
    
    plt.subplot(2, n_cols, 1)
    plt.imshow(amp_true, cmap='gray')
    plt.title('真实振幅')
    plt.colorbar()
    
    plt.subplot(2, n_cols, 2)
    plt.imshow(phase_true, cmap='jet')
    plt.title('真实相位')
    plt.colorbar()
    
    for k in range(n_defocus):
        plt.subplot(2, n_cols, k + 3)
        plt.imshow(np.log(intensities[k] + 1), cmap='gray')
        plt.title(f'离焦面 {defocus_values[k]:.1f}λ 强度')
        plt.colorbar()
    
    plt.subplot(2, n_cols, n_defocus + 3)
    plt.imshow(recovered_amp, cmap='gray')
    plt.title('恢复的振幅')
    plt.colorbar()
    
    plt.subplot(2, n_cols, n_defocus + 4)
    plt.imshow(recovered_phase, cmap='jet')
    plt.title('恢复的相位')
    plt.colorbar()
    
    plt.subplot(2, n_cols, n_defocus + 5)
    phase_error = np.abs(recovered_phase - phase_true)
    phase_error = np.minimum(phase_error, 2 * np.pi - phase_error)
    plt.imshow(phase_error, cmap='jet')
    plt.title(f'相位误差 (均值: {np.mean(phase_error):.3f} rad)')
    plt.colorbar()
    
    plt.subplot(2, n_cols, n_defocus + 6)
    plt.plot(error_history)
    plt.yscale('log')
    plt.xlabel('迭代次数')
    plt.ylabel('均方误差')
    plt.title('收敛曲线')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gs_phase_diversity.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存至 gs_phase_diversity.png")
    plt.show()


if __name__ == "__main__":
    import sys
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'near_far':
            demo_near_far()
        elif mode == 'phase_diversity':
            demo_phase_diversity()
        else:
            print(f"未知模式: {mode}")
            print("可用模式: standard, near_far, phase_diversity")
            demo()
    else:
        demo()
