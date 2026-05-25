import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter, maximum_filter, label
from scipy.cluster.vq import kmeans2
from scipy.signal.windows import hann, hamming
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import warnings
warnings.filterwarnings('ignore')


def load_hologram(file_path):
    hologram = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if hologram is None:
        raise ValueError(f"无法加载图像: {file_path}")
    return hologram.astype(np.float64)


def generate_synthetic_hologram(size=512, object_type='complex', noise_level=0.01):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    if object_type == 'circle':
        obj_amplitude = np.exp(-(X**2 + Y**2) / 0.1)
        obj_phase = 2 * np.pi * (X**2 + Y**2)
    elif object_type == 'letter':
        obj_amplitude = np.zeros((size, size))
        cv2.putText(obj_amplitude, 'DH', (size//4, size//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, 1, 8)
        obj_phase = np.zeros((size, size))
    elif object_type == 'complex':
        obj_amplitude = (np.exp(-((X-0.3)**2 + Y**2) / 0.05) + 
                        0.7 * np.exp(-((X+0.3)**2 + Y**2) / 0.05) +
                        0.5 * np.exp(-(X**2 + (Y-0.3)**2) / 0.08))
        obj_phase = 3 * np.pi * (X**2 + Y**2) + np.sin(5 * X)
    else:
        obj_amplitude = np.exp(-(X**2 + Y**2) / 0.05) * np.sin(10 * X)
        obj_phase = 2 * np.pi * X
    
    object_wave = obj_amplitude * np.exp(1j * obj_phase)
    
    k0 = 2 * np.pi / 0.5e-6
    theta_x = 0.08
    theta_y = 0.03
    ref_wave = np.exp(1j * k0 * (X * theta_x + Y * theta_y))
    
    hologram = np.abs(object_wave + ref_wave)**2
    
    if noise_level > 0:
        hologram += noise_level * np.random.randn(*hologram.shape) * hologram.max()
    
    hologram = np.clip(hologram, 0, None)
    hologram = (hologram - hologram.min()) / (hologram.max() - hologram.min()) * 255
    
    return hologram.astype(np.uint8), obj_amplitude, obj_phase


def apply_hanning_window(image):
    rows, cols = image.shape
    window_row = hann(rows)
    window_col = hann(cols)
    window_2d = np.outer(window_row, window_col)
    return image * window_2d, window_2d


def apply_hamming_window(image):
    rows, cols = image.shape
    window_row = hamming(rows)
    window_col = hamming(cols)
    window_2d = np.outer(window_row, window_col)
    return image * window_2d, window_2d


def fft_spectrum(hologram, use_window='hann'):
    if use_window == 'hann':
        holo_windowed, window = apply_hanning_window(hologram)
    elif use_window == 'hamming':
        holo_windowed, window = apply_hamming_window(hologram)
    else:
        holo_windowed = hologram
        window = np.ones_like(hologram)
    
    spectrum = fft2(holo_windowed)
    spectrum_shifted = fftshift(spectrum)
    
    return spectrum, spectrum_shifted, window


def detect_spectrum_peaks(spectrum_shifted, min_distance=20, threshold_rel=0.1):
    magnitude = np.abs(spectrum_shifted)
    magnitude_log = np.log(magnitude + 1)
    
    magnitude_smooth = gaussian_filter(magnitude_log, sigma=2)
    
    neighborhood_size = max(3, min_distance // 2)
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1
    
    local_max = maximum_filter(magnitude_smooth, size=neighborhood_size) == magnitude_smooth
    
    threshold = magnitude_smooth.max() * threshold_rel
    above_threshold = magnitude_smooth > threshold
    
    peaks = local_max & above_threshold
    
    labeled, num_features = label(peaks)
    
    peak_coords = []
    for i in range(1, num_features + 1):
        coords = np.where(labeled == i)
        center_y = int(np.mean(coords[0]))
        center_x = int(np.mean(coords[1]))
        peak_coords.append((center_y, center_x, magnitude_smooth[center_y, center_x]))
    
    peak_coords.sort(key=lambda x: x[2], reverse=True)
    
    return peak_coords, magnitude_log


def cluster_spectrum_peaks(peak_coords, num_clusters=3):
    if len(peak_coords) < num_clusters:
        return peak_coords, np.zeros(len(peak_coords))
    
    coords_array = np.array([[p[0], p[1]] for p in peak_coords])
    weights = np.array([p[2] for p in peak_coords])
    
    centroids, labels = kmeans2(coords_array.astype(float), num_clusters, 
                                minit='points', seed=42)
    
    cluster_peaks = []
    for i in range(num_clusters):
        cluster_mask = labels == i
        if np.any(cluster_mask):
            cluster_coords = coords_array[cluster_mask]
            cluster_weights = weights[cluster_mask]
            weighted_center = np.average(cluster_coords, weights=cluster_weights, axis=0)
            total_weight = np.sum(cluster_weights)
            cluster_peaks.append((int(weighted_center[0]), 
                                  int(weighted_center[1]), 
                                  total_weight))
    
    cluster_peaks.sort(key=lambda x: x[2], reverse=True)
    
    return cluster_peaks, labels


def classify_diffraction_orders(cluster_peaks, image_shape):
    center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
    
    distances = []
    for (y, x, weight) in cluster_peaks:
        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        distances.append(dist)
    
    if len(cluster_peaks) >= 3:
        zero_order_idx = np.argmin(distances)
        zero_order = cluster_peaks[zero_order_idx]
        
        remaining = [i for i in range(len(cluster_peaks)) if i != zero_order_idx]
        if len(remaining) >= 2:
            idx1, idx2 = remaining[:2]
            order1 = cluster_peaks[idx1]
            order2 = cluster_peaks[idx2]
            
            if order1[1] > order2[1]:
                plus_one = order1
                minus_one = order2
            else:
                plus_one = order2
                minus_one = order1
            
            return zero_order, plus_one, minus_one
    
    elif len(cluster_peaks) == 2:
        zero_order_idx = np.argmin(distances)
        zero_order = cluster_peaks[zero_order_idx]
        other_idx = 1 - zero_order_idx
        other_order = cluster_peaks[other_idx]
        
        if other_order[1] > center_x:
            plus_one = other_order
            minus_one = None
        else:
            plus_one = None
            minus_one = other_order
        
        return zero_order, plus_one, minus_one
    
    else:
        return cluster_peaks[0] if cluster_peaks else None, None, None


def calculate_roi_size(spectrum_shape, peak_location, scale_factor=0.25):
    rows, cols = spectrum_shape
    center_y, center_x = rows // 2, cols // 2
    
    peak_y, peak_x = peak_location[:2]
    
    distance_to_center = np.sqrt((peak_y - center_y)**2 + (peak_x - center_x)**2)
    
    roi_size = int(distance_to_center * scale_factor)
    roi_size = max(roi_size, min(rows, cols) // 8)
    roi_size = min(roi_size, min(rows, cols) // 3)
    
    return roi_size


def create_circular_window(shape, center, radius):
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    window = np.zeros(shape, dtype=np.float64)
    window[dist_from_center <= radius] = 1
    
    edge_width = radius * 0.2
    edge_region = (dist_from_center > radius - edge_width) & (dist_from_center <= radius)
    window[edge_region] = np.cos((dist_from_center[edge_region] - (radius - edge_width)) 
                                 * np.pi / (2 * edge_width))
    
    return window


def filter_spectrum_improved(spectrum_shifted, plus_one_order, roi_size=None, 
                            window_type='cosine'):
    if plus_one_order is None:
        raise ValueError("无法定位+1级衍射斑")
    
    peak_y, peak_x = plus_one_order[:2]
    
    if roi_size is None:
        roi_size = calculate_roi_size(spectrum_shifted.shape, (peak_y, peak_x))
    
    if window_type == 'cosine':
        mask = create_circular_window(spectrum_shifted.shape, (peak_y, peak_x), roi_size // 2)
    elif window_type == 'rectangle':
        mask = np.zeros_like(spectrum_shifted, dtype=np.float64)
        y_min = max(0, peak_y - roi_size // 2)
        y_max = min(spectrum_shifted.shape[0], peak_y + roi_size // 2)
        x_min = max(0, peak_x - roi_size // 2)
        x_max = min(spectrum_shifted.shape[1], peak_x + roi_size // 2)
        mask[y_min:y_max, x_min:x_max] = 1
    else:
        raise ValueError(f"未知窗函数类型: {window_type}")
    
    filtered_spectrum = spectrum_shifted * mask
    
    return filtered_spectrum, mask, roi_size


def shift_to_center(spectrum_shifted, peak_location):
    rows, cols = spectrum_shifted.shape
    center_y, center_x = rows // 2, cols // 2
    peak_y, peak_x = peak_location[:2]
    
    dy = center_y - peak_y
    dx = center_x - peak_x
    
    shifted_spectrum = np.roll(spectrum_shifted, (dy, dx), axis=(0, 1))
    
    return shifted_spectrum, (dy, dx)


def reconstruct_wavefront(filtered_spectrum, window=None):
    filtered_spectrum_unshifted = ifftshift(filtered_spectrum)
    reconstructed_wave = ifft2(filtered_spectrum_unshifted)
    
    if window is not None and window.mean() > 0:
        reconstructed_wave = reconstructed_wave / (window + 1e-10)
    
    return reconstructed_wave


def extract_amplitude_phase(reconstructed_wave):
    amplitude = np.abs(reconstructed_wave)
    phase = np.angle(reconstructed_wave)
    
    phase_unwrapped = np.unwrap(np.unwrap(phase, axis=0), axis=1)
    
    return amplitude, phase, phase_unwrapped


def remove_linear_phase(phase_unwrapped):
    rows, cols = phase_unwrapped.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    mask = np.ones_like(phase_unwrapped, dtype=bool)
    center = (rows // 2, cols // 2)
    radius = min(rows, cols) // 3
    y_grid, x_grid = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x_grid - center[1])**2 + (y_grid - center[0])**2)
    mask[dist_from_center > radius] = False
    
    A = np.column_stack([X[mask].flatten(), Y[mask].flatten(), np.ones(mask.sum())])
    b = phase_unwrapped[mask].flatten()
    
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    linear_phase = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    phase_corrected = phase_unwrapped - linear_phase
    
    return phase_corrected


def improved_reconstruction_pipeline(hologram, use_window='hann', 
                                    filter_window='cosine',
                                    auto_detect=True):
    print("\n=== 改进型离轴全息重建 ===")
    
    print(f"1. 应用{use_window}窗函数减少频谱泄漏...")
    spectrum, spectrum_shifted, window = fft_spectrum(hologram, use_window=use_window)
    
    print("2. 检测频谱峰值...")
    peak_coords, magnitude_log = detect_spectrum_peaks(spectrum_shifted)
    print(f"   检测到 {len(peak_coords)} 个峰值")
    
    print("3. 聚类分析分离衍射级...")
    cluster_peaks, _ = cluster_spectrum_peaks(peak_coords, num_clusters=3)
    print(f"   识别到 {len(cluster_peaks)} 个聚类中心")
    
    print("4. 分类衍射级（零级、+1级、-1级）...")
    zero_order, plus_one_order, minus_one_order = classify_diffraction_orders(
        cluster_peaks, hologram.shape)
    
    if zero_order:
        print(f"   零级位置: ({zero_order[0]}, {zero_order[1]})")
    if plus_one_order:
        print(f"   +1级位置: ({plus_one_order[0]}, {plus_one_order[1]})")
    if minus_one_order:
        print(f"   -1级位置: ({minus_one_order[0]}, {minus_one_order[1]})")
    
    if plus_one_order is None:
        print("   警告: 无法定位+1级，尝试使用距离中心最远的峰值...")
        if len(cluster_peaks) > 1:
            center_y, center_x = hologram.shape[0] // 2, hologram.shape[1] // 2
            distances = [np.sqrt((p[0]-center_y)**2 + (p[1]-center_x)**2) 
                        for p in cluster_peaks]
            plus_one_order = cluster_peaks[np.argmax(distances)]
            print(f"   使用位置: ({plus_one_order[0]}, {plus_one_order[1]})")
    
    print(f"5. 使用{filter_window}窗滤波...")
    filtered_spectrum, mask, roi_size = filter_spectrum_improved(
        spectrum_shifted, plus_one_order, window_type=filter_window)
    print(f"   ROI尺寸: {roi_size}")
    
    print("6. 将+1级移到频谱中心...")
    centered_spectrum, shift = shift_to_center(filtered_spectrum, plus_one_order)
    print(f"   平移量: dy={shift[0]}, dx={shift[1]}")
    
    print("7. 逆傅里叶变换重建波前...")
    reconstructed_wave = reconstruct_wavefront(centered_spectrum, window)
    
    print("8. 提取幅度和相位...")
    amplitude, phase, phase_unwrapped = extract_amplitude_phase(reconstructed_wave)
    
    print("9. 校正线性相位倾斜...")
    phase_corrected = remove_linear_phase(phase_unwrapped)
    
    print("=== 重建完成 ===\n")
    
    return {
        'hologram': hologram,
        'window': window,
        'spectrum_shifted': spectrum_shifted,
        'magnitude_log': magnitude_log,
        'peak_coords': peak_coords,
        'cluster_peaks': cluster_peaks,
        'zero_order': zero_order,
        'plus_one_order': plus_one_order,
        'minus_one_order': minus_one_order,
        'mask': mask,
        'filtered_spectrum': filtered_spectrum,
        'centered_spectrum': centered_spectrum,
        'reconstructed_wave': reconstructed_wave,
        'amplitude': amplitude,
        'phase': phase,
        'phase_unwrapped': phase_unwrapped,
        'phase_corrected': phase_corrected,
        'roi_size': roi_size
    }


def plot_improved_results(results, true_amp=None, true_phase=None):
    fig = plt.figure(figsize=(20, 12))
    
    plt.subplot(3, 4, 1)
    plt.imshow(results['hologram'], cmap='gray')
    plt.title('原始全息图')
    plt.colorbar()
    
    plt.subplot(3, 4, 2)
    plt.imshow(results['window'], cmap='gray')
    plt.title('汉宁窗函数')
    plt.colorbar()
    
    plt.subplot(3, 4, 3)
    plt.imshow(results['magnitude_log'], cmap='gray')
    plt.title('傅里叶频谱')
    
    for p in results['cluster_peaks']:
        plt.plot(p[1], p[0], 'ro', markersize=8, markerfacecolor='none')
    if results['plus_one_order']:
        plt.plot(results['plus_one_order'][1], results['plus_one_order'][0], 
                'go', markersize=12, markerfacecolor='none', linewidth=2, label='+1级')
    if results['zero_order']:
        plt.plot(results['zero_order'][1], results['zero_order'][0], 
                'bo', markersize=12, markerfacecolor='none', linewidth=2, label='零级')
    plt.legend()
    plt.colorbar()
    
    plt.subplot(3, 4, 4)
    plt.imshow(results['magnitude_log'], cmap='gray')
    plt.title('+1级滤波区域')
    y, x = np.ogrid[:results['mask'].shape[0], :results['mask'].shape[1]]
    plt.contour(x, y, results['mask'], levels=[0.5], colors='r', linewidths=2)
    plt.colorbar()
    
    plt.subplot(3, 4, 5)
    filtered_mag = np.log(np.abs(results['filtered_spectrum']) + 1)
    plt.imshow(filtered_mag, cmap='gray')
    plt.title('滤波后频谱')
    plt.colorbar()
    
    plt.subplot(3, 4, 6)
    centered_mag = np.log(np.abs(results['centered_spectrum']) + 1)
    plt.imshow(centered_mag, cmap='gray')
    plt.title('中心化频谱')
    plt.colorbar()
    
    plt.subplot(3, 4, 7)
    plt.imshow(results['amplitude'], cmap='gray')
    plt.title('重建幅度')
    plt.colorbar()
    
    plt.subplot(3, 4, 8)
    if true_amp is not None:
        plt.imshow(true_amp, cmap='gray')
        plt.title('原始幅度(对比)')
    else:
        plt.imshow(results['amplitude'], cmap='gray')
        plt.title('重建幅度(无对比)')
    plt.colorbar()
    
    plt.subplot(3, 4, 9)
    plt.imshow(results['phase'], cmap='jet')
    plt.title('包裹相位')
    plt.colorbar()
    
    plt.subplot(3, 4, 10)
    plt.imshow(results['phase_unwrapped'], cmap='jet')
    plt.title('解包裹相位')
    plt.colorbar()
    
    plt.subplot(3, 4, 11)
    plt.imshow(results['phase_corrected'], cmap='jet')
    plt.title('校正后相位')
    plt.colorbar()
    
    plt.subplot(3, 4, 12)
    if true_phase is not None:
        plt.imshow(true_phase, cmap='jet')
        plt.title('原始相位(对比)')
    else:
        plt.imshow(results['phase_corrected'], cmap='jet')
        plt.title('校正后相位(无对比)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('improved_reconstruction.png', dpi=150, bbox_inches='tight')
    print("结果已保存至 improved_reconstruction.png")
    plt.show()
    
    fig2 = plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(results['amplitude'], cmap='gray')
    ax1.set_title('重建幅度')
    fig2.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    rows, cols = results['phase_corrected'].shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    step = max(1, rows // 100)
    surf = ax2.plot_surface(X[::step, ::step], Y[::step, ::step], 
                            results['phase_corrected'][::step, ::step],
                            cmap='jet', linewidth=0, antialiased=True)
    ax2.set_title('相位3D视图')
    fig2.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('phase_3d_view.png', dpi=150, bbox_inches='tight')
    print("3D视图已保存至 phase_3d_view.png")
    plt.show()


def main():
    print("=" * 70)
    print("改进型离轴数字全息图重建程序")
    print("  - 汉宁窗减少频谱泄漏")
    print("  - 自动峰值检测与聚类")
    print("  - 余弦窗滤波减少信息损失")
    print("  - 频谱中心化优化重建质量")
    print("=" * 70)
    
    import os
    
    hologram_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    
    if hologram_files:
        print(f"\n找到以下图像文件:")
        for i, f in enumerate(hologram_files):
            print(f"  {i+1}. {f}")
        
        choice = input("\n选择文件编号 (输入0使用合成全息图): ").strip()
        
        if choice.isdigit() and int(choice) > 0 and int(choice) <= len(hologram_files):
            file_path = hologram_files[int(choice)-1]
            print(f"\n加载全息图: {file_path}")
            hologram = load_hologram(file_path)
            true_amp, true_phase = None, None
        else:
            print("\n生成合成全息图...")
            hologram, true_amp, true_phase = generate_synthetic_hologram(
                size=512, object_type='complex', noise_level=0.02)
    else:
        print("\n未找到图像文件，生成合成全息图...")
        hologram, true_amp, true_phase = generate_synthetic_hologram(
            size=512, object_type='complex', noise_level=0.02)
    
    print(f"\n全息图尺寸: {hologram.shape}")
    
    results = improved_reconstruction_pipeline(
        hologram,
        use_window='hann',
        filter_window='cosine',
        auto_detect=True
    )
    
    print("\n显示结果...")
    plot_improved_results(results, true_amp, true_phase)
    
    print("\n" + "=" * 70)
    print("程序结束")
    print("=" * 70)


if __name__ == '__main__':
    main()
