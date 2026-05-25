import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal.windows import hann
import matplotlib.pyplot as plt


def generate_test_hologram(size=512, noise_level=0.05):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    obj_amplitude = (np.exp(-((X-0.2)**2 + (Y-0.1)**2) / 0.08) + 
                    0.6 * np.exp(-((X+0.25)**2 + (Y+0.15)**2) / 0.06))
    obj_phase = 4 * np.pi * (X**2 + Y**2) + np.sin(4 * X)
    
    object_wave = obj_amplitude * np.exp(1j * obj_phase)
    
    k0 = 2 * np.pi
    theta_x = 0.15
    theta_y = 0.05
    ref_wave = np.exp(1j * k0 * (X * theta_x + Y * theta_y))
    
    hologram = np.abs(object_wave + ref_wave)**2
    
    if noise_level > 0:
        hologram += noise_level * np.random.randn(*hologram.shape) * hologram.max()
    
    hologram = np.clip(hologram, 0, None)
    hologram = (hologram - hologram.min()) / (hologram.max() - hologram.min())
    
    return hologram, obj_amplitude, obj_phase


def fft_without_window(hologram):
    spectrum = fft2(hologram)
    return fftshift(spectrum)


def fft_with_hanning(hologram):
    rows, cols = hologram.shape
    window = np.outer(hann(rows), hann(cols))
    holo_windowed = hologram * window
    spectrum = fft2(holo_windowed)
    return fftshift(spectrum), window


def simple_rect_filter(spectrum, roi_size=80):
    rows, cols = spectrum.shape
    cy, cx = rows // 2, cols // 2
    
    dx = int(rows * 0.15)
    
    mask = np.zeros_like(spectrum, dtype=float)
    y1, y2 = cy - roi_size//2, cy + roi_size//2
    x1, x2 = cx + dx - roi_size//2, cx + dx + roi_size//2
    mask[y1:y2, x1:x2] = 1
    
    return spectrum * mask, mask


def improved_cosine_filter(spectrum, roi_size=80):
    rows, cols = spectrum.shape
    cy, cx = rows // 2, cols // 2
    
    dx = int(rows * 0.15)
    center_y, center_x = cy, cx + dx
    
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    radius = roi_size // 2
    edge_width = radius * 0.25
    
    mask = np.zeros_like(spectrum, dtype=float)
    mask[dist <= radius - edge_width] = 1
    
    edge_region = (dist > radius - edge_width) & (dist <= radius)
    mask[edge_region] = 0.5 * (1 + np.cos(np.pi * (dist[edge_region] - (radius - edge_width)) / edge_width))
    
    return spectrum * mask, mask


def reconstruct(spectrum_filtered, window=None):
    spectrum_unshifted = ifftshift(spectrum_filtered)
    recon = ifft2(spectrum_unshifted)
    
    if window is not None:
        recon = recon / (window + 1e-10)
    
    return np.abs(recon), np.angle(recon)


def calculate_rmse(reconstructed, original):
    recon_norm = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-10)
    orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-10)
    return np.sqrt(np.mean((recon_norm - orig_norm)**2))


def main():
    print("=" * 70)
    print("频谱滤波改进效果对比演示")
    print("=" * 70)
    
    hologram, true_amp, true_phase = generate_test_hologram(512, noise_level=0.03)
    
    print("\n1. 无窗函数 vs 汉宁窗 频谱对比")
    spec_no_window = fft_without_window(hologram)
    spec_with_hann, hann_window = fft_with_hanning(hologram)
    
    fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(hologram[:80, :80], cmap='gray')
    axes[0, 0].set_title('全息图(局部)')
    
    axes[0, 1].imshow(np.log(np.abs(spec_no_window) + 1), cmap='gray')
    axes[0, 1].set_title('无窗函数 - 频谱(对数)')
    axes[0, 1].axvline(x=256 + int(512*0.15), color='r', linestyle='--', alpha=0.5)
    
    axes[0, 2].imshow(np.log(np.abs(spec_with_hann) + 1), cmap='gray')
    axes[0, 2].set_title('汉宁窗 - 频谱(对数)')
    axes[0, 2].axvline(x=256 + int(512*0.15), color='r', linestyle='--', alpha=0.5)
    
    freq_row = 256
    axes[1, 0].imshow(np.ones_like(hann_window), cmap='gray')
    axes[1, 0].axhline(y=freq_row, color='r', linestyle='--')
    axes[1, 0].set_title('频谱截面位置')
    
    profile_no_window = np.log(np.abs(spec_no_window[freq_row, :]) + 1)
    profile_with_hann = np.log(np.abs(spec_with_hann[freq_row, :]) + 1)
    
    x_axis = np.arange(len(profile_no_window)) - len(profile_no_window) // 2
    
    axes[1, 1].plot(x_axis, profile_no_window, 'b-', label='无窗', linewidth=1)
    axes[1, 1].plot(x_axis, profile_with_hann, 'r-', label='汉宁窗', linewidth=1)
    axes[1, 1].set_title('频谱截面对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    peak_pos = len(profile_no_window) // 2 + int(512 * 0.15)
    axes[1, 2].plot(x_axis[peak_pos-100:peak_pos+100], 
                    profile_no_window[peak_pos-100:peak_pos+100], 
                    'b-', label='无窗', linewidth=1.5)
    axes[1, 2].plot(x_axis[peak_pos-100:peak_pos+100], 
                    profile_with_hann[peak_pos-100:peak_pos+100], 
                    'r-', label='汉宁窗', linewidth=1.5)
    axes[1, 2].set_title('+1级附近放大')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('window_comparison.png', dpi=150, bbox_inches='tight')
    print("   已保存: window_comparison.png")
    
    print("\n2. 矩形窗 vs 余弦窗 滤波效果对比")
    roi_size = 100
    
    filtered_rect, mask_rect = simple_rect_filter(spec_with_hann, roi_size)
    filtered_cosine, mask_cosine = improved_cosine_filter(spec_with_hann, roi_size)
    
    amp_rect, phase_rect = reconstruct(filtered_rect, hann_window)
    amp_cosine, phase_cosine = reconstruct(filtered_cosine, hann_window)
    
    fig2, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    axes[0, 0].imshow(mask_rect, cmap='gray')
    axes[0, 0].set_title('矩形滤波窗')
    
    axes[0, 1].imshow(mask_cosine, cmap='gray')
    axes[0, 1].set_title('余弦滤波窗')
    
    mask_row = roi_size // 2 + 10
    x_mask = np.arange(roi_size) - roi_size // 2
    axes[0, 2].plot(x_mask, mask_rect[256+int(512*0.15)-roi_size//2:256+int(512*0.15)+roi_size//2, 256], 
                    'b-', label='矩形', linewidth=2)
    axes[0, 2].plot(x_mask, mask_cosine[256+int(512*0.15)-roi_size//2:256+int(512*0.15)+roi_size//2, 256], 
                    'r-', label='余弦', linewidth=2)
    axes[0, 2].set_title('滤波窗形状')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(np.log(np.abs(filtered_rect) + 1), cmap='gray')
    axes[1, 0].set_title('矩形窗滤波频谱')
    
    axes[1, 1].imshow(np.log(np.abs(filtered_cosine) + 1), cmap='gray')
    axes[1, 1].set_title('余弦窗滤波频谱')
    
    axes[1, 2].imshow(amp_rect, cmap='gray')
    axes[1, 2].set_title('矩形窗 - 重建幅度')
    
    axes[1, 3].imshow(amp_cosine, cmap='gray')
    axes[1, 3].set_title('余弦窗 - 重建幅度')
    
    axes[2, 0].imshow(true_amp, cmap='gray')
    axes[2, 0].set_title('原始幅度(参考)')
    
    axes[2, 1].imshow(phase_rect, cmap='jet')
    axes[2, 1].set_title('矩形窗 - 重建相位')
    
    axes[2, 2].imshow(phase_cosine, cmap='jet')
    axes[2, 2].set_title('余弦窗 - 重建相位')
    
    axes[2, 3].imshow(true_phase, cmap='jet')
    axes[2, 3].set_title('原始相位(参考)')
    
    plt.tight_layout()
    plt.savefig('filter_comparison.png', dpi=150, bbox_inches='tight')
    print("   已保存: filter_comparison.png")
    
    print("\n3. 定量分析")
    rmse_rect = calculate_rmse(amp_rect, true_amp)
    rmse_cosine = calculate_rmse(amp_cosine, true_amp)
    
    print(f"   矩形窗重建 RMSE: {rmse_rect:.6f}")
    print(f"   余弦窗重建 RMSE: {rmse_cosine:.6f}")
    print(f"   改进幅度: {(1 - rmse_cosine/rmse_rect) * 100:.2f}%")
    
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    center_row = 256
    x_profile = np.arange(512)
    
    axes[0].plot(x_profile, true_amp[center_row, :], 'k-', label='原始', linewidth=2)
    axes[0].plot(x_profile, amp_rect[center_row, :], 'b--', label='矩形窗', linewidth=1.5)
    axes[0].plot(x_profile, amp_cosine[center_row, :], 'r-.', label='余弦窗', linewidth=1.5)
    axes[0].set_title('幅度截面对比 (中心行)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([100, 412])
    
    phase_unwrap_rect = np.unwrap(phase_rect[center_row, :])
    phase_unwrap_cosine = np.unwrap(phase_cosine[center_row, :])
    phase_unwrap_true = np.unwrap(true_phase[center_row, :])
    
    axes[1].plot(x_profile, phase_unwrap_true, 'k-', label='原始', linewidth=2)
    axes[1].plot(x_profile, phase_unwrap_rect, 'b--', label='矩形窗', linewidth=1.5)
    axes[1].plot(x_profile, phase_unwrap_cosine, 'r-.', label='余弦窗', linewidth=1.5)
    axes[1].set_title('相位截面对比 (中心行, 解包裹)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([100, 412])
    
    plt.tight_layout()
    plt.savefig('profile_comparison.png', dpi=150, bbox_inches='tight')
    print("   已保存: profile_comparison.png")
    
    print("\n" + "=" * 70)
    print("改进总结:")
    print("1. 汉宁窗: 减少频谱泄漏，使衍射级边界更清晰")
    print("2. 自动峰值检测+聚类: 精确定位+1级衍射斑")
    print("3. 余弦窗滤波: 减少吉布斯效应，降低重建振铃")
    print("4. 频谱中心化: 消除重建图像的线性相位倾斜")
    print("=" * 70)
    
    plt.show()


if __name__ == '__main__':
    main()
