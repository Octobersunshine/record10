import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
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


def generate_synthetic_hologram(size=512, object_type='circle'):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    if object_type == 'circle':
        obj_amplitude = np.exp(-(X**2 + Y**2) / 0.1)
        obj_phase = 2 * np.pi * (X**2 + Y**2)
    elif object_type == 'letter':
        obj_amplitude = np.zeros((size, size))
        cv2.putText(obj_amplitude, 'A', (size//4, size//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, 1, 10)
        obj_phase = np.zeros((size, size))
    else:
        obj_amplitude = np.exp(-(X**2 + Y**2) / 0.05) * np.sin(10 * X)
        obj_phase = 2 * np.pi * X
    
    object_wave = obj_amplitude * np.exp(1j * obj_phase)
    
    k0 = 2 * np.pi / 0.5e-6
    theta_x = 0.05
    theta_y = 0.02
    ref_wave = np.exp(1j * k0 * (X * theta_x + Y * theta_y))
    
    hologram = np.abs(object_wave + ref_wave)**2
    hologram = (hologram - hologram.min()) / (hologram.max() - hologram.min()) * 255
    
    return hologram.astype(np.uint8), obj_amplitude, obj_phase


def fft_spectrum(hologram):
    spectrum = fft2(hologram)
    spectrum_shifted = fftshift(spectrum)
    return spectrum, spectrum_shifted


def select_roi_callback(eclick, erelease):
    global roi_selected, roi_coords
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    roi_coords = (int(min(x1, x2)), int(max(x1, x2)), 
                  int(min(y1, y2)), int(max(y1, y2)))
    roi_selected = True
    print(f"选择的ROI: x=[{roi_coords[0]}, {roi_coords[1]}], y=[{roi_coords[2]}, {roi_coords[3]}]")


def interactive_roi_selection(spectrum_shifted):
    global roi_selected, roi_coords
    roi_selected = False
    roi_coords = None
    
    magnitude = np.log(np.abs(spectrum_shifted) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(magnitude, cmap='gray')
    ax.set_title('傅里叶频谱 - 用鼠标框选+1级衍射斑')
    
    rs = RectangleSelector(ax, select_roi_callback,
                           useblit=True,
                           button=[1],
                           minspanx=5, minspany=5,
                           spancoords='pixels',
                           interactive=True)
    
    print("请用鼠标在频谱图上框选+1级衍射斑（非中心的亮点）")
    print("选择完成后关闭窗口继续...")
    plt.show()
    
    return roi_coords


def filter_spectrum(spectrum_shifted, roi_coords):
    x_min, x_max, y_min, y_max = roi_coords
    
    mask = np.zeros_like(spectrum_shifted, dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    
    filtered_spectrum = np.zeros_like(spectrum_shifted)
    filtered_spectrum[mask] = spectrum_shifted[mask]
    
    return filtered_spectrum, mask


def reconstruct_wavefront(filtered_spectrum):
    filtered_spectrum_unshifted = ifftshift(filtered_spectrum)
    reconstructed_wave = ifft2(filtered_spectrum_unshifted)
    return reconstructed_wave


def extract_amplitude_phase(reconstructed_wave):
    amplitude = np.abs(reconstructed_wave)
    phase = np.angle(reconstructed_wave)
    
    phase_unwrapped = np.unwrap(phase)
    
    return amplitude, phase, phase_unwrapped


def remove_linear_phase(phase_unwrapped):
    rows, cols = phase_unwrapped.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    A = np.column_stack([X.flatten(), Y.flatten(), np.ones(X.size)])
    b = phase_unwrapped.flatten()
    
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    linear_phase = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    phase_corrected = phase_unwrapped - linear_phase
    
    return phase_corrected


def full_reconstruction_pipeline(hologram, roi_coords=None):
    spectrum, spectrum_shifted = fft_spectrum(hologram)
    
    if roi_coords is None:
        roi_coords = interactive_roi_selection(spectrum_shifted)
    
    if roi_coords is None:
        print("未选择ROI，使用自动检测...")
        magnitude = np.abs(spectrum_shifted)
        center = np.array(magnitude.shape) // 2
        
        mask_center = np.ones_like(magnitude, dtype=bool)
        radius = min(magnitude.shape) // 8
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        mask_center[dist_from_center < radius] = False
        
        masked_mag = magnitude * mask_center
        idx = np.unravel_index(np.argmax(masked_mag), masked_mag.shape)
        
        roi_size = min(magnitude.shape) // 6
        roi_coords = (max(0, idx[1] - roi_size//2), 
                      min(magnitude.shape[1], idx[1] + roi_size//2),
                      max(0, idx[0] - roi_size//2),
                      min(magnitude.shape[0], idx[0] + roi_size//2))
        print(f"自动检测ROI: {roi_coords}")
    
    filtered_spectrum, mask = filter_spectrum(spectrum_shifted, roi_coords)
    
    reconstructed_wave = reconstruct_wavefront(filtered_spectrum)
    
    amplitude, phase, phase_unwrapped = extract_amplitude_phase(reconstructed_wave)
    
    phase_corrected = remove_linear_phase(phase_unwrapped)
    
    return {
        'hologram': hologram,
        'spectrum_shifted': spectrum_shifted,
        'filtered_spectrum': filtered_spectrum,
        'mask': mask,
        'reconstructed_wave': reconstructed_wave,
        'amplitude': amplitude,
        'phase': phase,
        'phase_unwrapped': phase_unwrapped,
        'phase_corrected': phase_corrected,
        'roi_coords': roi_coords
    }


def plot_results(results):
    fig = plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(results['hologram'], cmap='gray')
    plt.title('全息图')
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    magnitude = np.log(np.abs(results['spectrum_shifted']) + 1)
    plt.imshow(magnitude, cmap='gray')
    plt.title('傅里叶频谱')
    plt.colorbar()
    
    plt.subplot(2, 3, 3)
    mag_filtered = np.log(np.abs(results['filtered_spectrum']) + 1)
    plt.imshow(mag_filtered, cmap='gray')
    plt.title('滤波后频谱')
    plt.colorbar()
    
    plt.subplot(2, 3, 4)
    plt.imshow(results['amplitude'], cmap='gray')
    plt.title('重建幅度')
    plt.colorbar()
    
    plt.subplot(2, 3, 5)
    plt.imshow(results['phase'], cmap='jet')
    plt.title('包裹相位')
    plt.colorbar()
    
    plt.subplot(2, 3, 6)
    plt.imshow(results['phase_corrected'], cmap='jet')
    plt.title('解包裹并去除倾斜的相位')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    fig2 = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(results['amplitude'], cmap='gray')
    plt.title('重建幅度')
    plt.colorbar()
    
    ax = plt.subplot(1, 2, 2, projection='3d')
    rows, cols = results['phase_corrected'].shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    surf = ax.plot_surface(X, Y, results['phase_corrected'], 
                           cmap='jet', linewidth=0, antialiased=True)
    ax.set_title('相位3D视图')
    fig2.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("离轴数字全息图重建程序")
    print("=" * 60)
    
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
            hologram, true_amp, true_phase = generate_synthetic_hologram(size=512)
    else:
        print("\n未找到图像文件，生成合成全息图...")
        hologram, true_amp, true_phase = generate_synthetic_hologram(size=512)
    
    print(f"\n全息图尺寸: {hologram.shape}")
    print("\n开始重建...")
    
    results = full_reconstruction_pipeline(hologram)
    
    print("\n重建完成！显示结果...")
    plot_results(results)
    
    if true_amp is not None and true_phase is not None:
        print("\n显示原始物体用于对比...")
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(true_amp, cmap='gray')
        plt.title('原始物体幅度')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(true_phase, cmap='jet')
        plt.title('原始物体相位')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
    print("\n" + "=" * 60)
    print("程序结束")
    print("=" * 60)


if __name__ == '__main__':
    main()
