import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter, maximum_filter, label
from scipy.cluster.vq import kmeans2
from scipy.signal.windows import hann, hamming
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import time
from phase_unwrapping import (
    PhaseUnwrapper, AberrationCorrector, 
    create_quality_map, temporal_unwrapping
)
import warnings
warnings.filterwarnings('ignore')


class LiveCellDigitalHolography:
    def __init__(self, config=None):
        self.config = {
            'use_window': 'hann',
            'filter_window': 'cosine',
            'unwrapping_method': 'quality_guided',
            'zernike_order': 6,
            'remove_aberration_terms': [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -1), (3, 1)],
            'auto_detect': True,
            'noise_level': 0.02,
            'wavelength': 0.532e-6,
            'pixel_size': 6.5e-6,
        }
        if config:
            self.config.update(config)
        
        self.unwrapper = PhaseUnwrapper(method=self.config['unwrapping_method'])
        self.corrector = AberrationCorrector(max_order=self.config['zernike_order'])
        
        self.calibration_done = False
        self.roi_coords = None
        self.plus_one_location = None
        
    def load_hologram(self, file_path):
        hologram = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if hologram is None:
            raise ValueError(f"无法加载图像: {file_path}")
        return hologram.astype(np.float64)
    
    def generate_cell_hologram(self, size=512, cell_type='multiple', time_step=0):
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        obj_amplitude = np.ones((size, size)) * 0.3
        
        if cell_type == 'single':
            cell_params = [
                {'cx': 0, 'cy': 0, 'rx': 0.2, 'ry': 0.15, 'amp': 0.8, 'phase': 1.5}
            ]
        elif cell_type == 'multiple':
            cell_params = [
                {'cx': -0.3 + 0.05 * np.sin(time_step * 0.5), 
                 'cy': -0.1 + 0.03 * np.cos(time_step * 0.3), 
                 'rx': 0.18, 'ry': 0.12, 'amp': 0.9, 'phase': 2.0 + 0.2 * np.sin(time_step * 0.1)},
                {'cx': 0.2 + 0.02 * np.sin(time_step * 0.4), 
                 'cy': 0.15 + 0.04 * np.sin(time_step * 0.2), 
                 'rx': 0.15, 'ry': 0.13, 'amp': 0.85, 'phase': 1.8 + 0.15 * np.cos(time_step * 0.15)},
                {'cx': 0.1, 'cy': -0.25, 'rx': 0.12, 'ry': 0.1, 'amp': 0.75, 'phase': 1.2},
            ]
        elif cell_type == 'dividing':
            t = min(time_step * 0.05, 1.0)
            sep = t * 0.3
            cell_params = [
                {'cx': -sep/2, 'cy': 0, 'rx': 0.15 - t*0.05, 'ry': 0.12, 'amp': 0.9, 'phase': 2.0},
                {'cx': sep/2, 'cy': 0, 'rx': 0.15 - t*0.05, 'ry': 0.12, 'amp': 0.9, 'phase': 2.0},
            ]
        else:
            cell_params = []
        
        for cp in cell_params:
            cell_shape = np.exp(-((X - cp['cx'])**2 / cp['rx']**2 + 
                                 (Y - cp['cy'])**2 / cp['ry']**2))
            obj_amplitude += cp['amp'] * cell_shape
        
        obj_phase = np.zeros((size, size))
        for cp in cell_params:
            cell_shape = np.exp(-((X - cp['cx'])**2 / (cp['rx']*1.2)**2 + 
                                 (Y - cp['cy'])**2 / (cp['ry']*1.2)**2))
            obj_phase += cp['phase'] * cell_shape
        
        nucleus_params = [p for p in cell_params]
        for np_ in nucleus_params:
            nuc_shape = np.exp(-((X - np_['cx'])**2 / (np_['rx']*0.4)**2 + 
                                (Y - np_['cy']*0.3)**2 / (np_['ry']*0.4)**2))
            obj_phase += 0.5 * nuc_shape
        
        object_wave = obj_amplitude * np.exp(1j * obj_phase)
        
        k0 = 2 * np.pi / 0.5e-6
        theta_x = 0.08
        theta_y = 0.03
        ref_wave = np.exp(1j * k0 * (X * theta_x + Y * theta_y))
        
        hologram = np.abs(object_wave + ref_wave)**2
        
        if self.config['noise_level'] > 0:
            hologram += self.config['noise_level'] * np.random.randn(*hologram.shape) * hologram.max()
        
        hologram = np.clip(hologram, 0, None)
        hologram = (hologram - hologram.min()) / (hologram.max() - hologram.min()) * 255
        
        return hologram.astype(np.uint8), obj_amplitude, obj_phase
    
    def _apply_window(self, hologram):
        rows, cols = hologram.shape
        if self.config['use_window'] == 'hann':
            window = np.outer(hann(rows), hann(cols))
        elif self.config['use_window'] == 'hamming':
            window = np.outer(hamming(rows), hamming(cols))
        else:
            window = np.ones_like(hologram)
        return hologram * window, window
    
    def _compute_spectrum(self, hologram):
        holo_windowed, window = self._apply_window(hologram)
        spectrum = fft2(holo_windowed)
        spectrum_shifted = fftshift(spectrum)
        return spectrum_shifted, window
    
    def _detect_spectrum_peaks(self, spectrum_shifted):
        magnitude = np.abs(spectrum_shifted)
        magnitude_log = np.log(magnitude + 1)
        magnitude_smooth = gaussian_filter(magnitude_log, sigma=2)
        
        neighborhood_size = 21
        local_max = maximum_filter(magnitude_smooth, size=neighborhood_size) == magnitude_smooth
        threshold = magnitude_smooth.max() * 0.1
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
    
    def _cluster_peaks(self, peak_coords, num_clusters=3):
        if len(peak_coords) < num_clusters:
            return peak_coords
        
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
        return cluster_peaks
    
    def _classify_diffraction_orders(self, cluster_peaks, image_shape):
        center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
        
        distances = [np.sqrt((p[0]-center_y)**2 + (p[1]-center_x)**2) for p in cluster_peaks]
        
        if len(cluster_peaks) >= 3:
            zero_order_idx = np.argmin(distances)
            zero_order = cluster_peaks[zero_order_idx]
            
            remaining = [i for i in range(len(cluster_peaks)) if i != zero_order_idx]
            idx1, idx2 = remaining[:2]
            order1, order2 = cluster_peaks[idx1], cluster_peaks[idx2]
            
            if order1[1] > order2[1]:
                plus_one, minus_one = order1, order2
            else:
                plus_one, minus_one = order2, order1
            
            return zero_order, plus_one, minus_one
        elif len(cluster_peaks) == 2:
            zero_order_idx = np.argmin(distances)
            zero_order = cluster_peaks[zero_order_idx]
            other = cluster_peaks[1 - zero_order_idx]
            if other[1] > center_x:
                return zero_order, other, None
            else:
                return zero_order, None, other
        else:
            return cluster_peaks[0] if cluster_peaks else None, None, None
    
    def _create_filter_mask(self, shape, center, roi_size):
        cy, cx = center[:2]
        y, x = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        radius = roi_size // 2
        edge_width = radius * 0.25
        
        mask = np.zeros(shape, dtype=np.float64)
        mask[dist <= radius - edge_width] = 1
        
        edge_region = (dist > radius - edge_width) & (dist <= radius)
        mask[edge_region] = 0.5 * (1 + np.cos(
            np.pi * (dist[edge_region] - (radius - edge_width)) / edge_width
        ))
        
        return mask
    
    def calibrate(self, hologram):
        print("正在校准系统...")
        
        spectrum_shifted, window = self._compute_spectrum(hologram)
        
        peak_coords, _ = self._detect_spectrum_peaks(spectrum_shifted)
        print(f"  检测到 {len(peak_coords)} 个频谱峰值")
        
        cluster_peaks = self._cluster_peaks(peak_coords)
        print(f"  聚类得到 {len(cluster_peaks)} 个衍射级")
        
        zero_order, plus_one, minus_one = self._classify_diffraction_orders(
            cluster_peaks, hologram.shape
        )
        
        if plus_one is None:
            print("  警告: 自动检测失败，使用备用位置")
            center_y, center_x = hologram.shape[0] // 2, hologram.shape[1] // 2
            plus_one = (center_y, center_x + int(hologram.shape[1] * 0.15), 0)
        
        self.plus_one_location = plus_one
        self.calibration_done = True
        
        print(f"  +1级位置: ({plus_one[0]}, {plus_one[1]})")
        print("  校准完成!")
        
        return zero_order, plus_one, minus_one
    
    def reconstruct(self, hologram, calibrate_first=True):
        if calibrate_first and not self.calibration_done:
            self.calibrate(hologram)
        
        spectrum_shifted, window = self._compute_spectrum(hologram)
        
        rows, cols = hologram.shape
        center_y, center_x = rows // 2, cols // 2
        
        peak_y, peak_x = self.plus_one_location[:2]
        distance_to_center = np.sqrt((peak_y - center_y)**2 + (peak_x - center_x)**2)
        roi_size = max(int(distance_to_center * 0.4), min(rows, cols) // 6)
        
        mask = self._create_filter_mask(spectrum_shifted.shape, 
                                        (peak_y, peak_x), roi_size)
        
        filtered_spectrum = spectrum_shifted * mask
        
        dy = center_y - peak_y
        dx = center_x - peak_x
        centered_spectrum = np.roll(filtered_spectrum, (dy, dx), axis=(0, 1))
        
        spectrum_unshifted = ifftshift(centered_spectrum)
        reconstructed_wave = ifft2(spectrum_unshifted)
        
        reconstructed_wave = reconstructed_wave / (window + 1e-10)
        
        amplitude = np.abs(reconstructed_wave)
        phase_wrapped = np.angle(reconstructed_wave)
        
        return amplitude, phase_wrapped, spectrum_shifted, mask
    
    def advanced_processing(self, amplitude, phase_wrapped, use_unwrapping=True, 
                           use_correction=True, background_phase=None):
        results = {}
        
        if use_unwrapping:
            t0 = time.time()
            phase_unwrapped = self.unwrapper.unwrap(phase_wrapped, amplitude)
            results['unwrapping_time'] = time.time() - t0
            results['phase_unwrapped'] = phase_unwrapped
        else:
            phase_unwrapped = np.unwrap(np.unwrap(phase_wrapped, axis=0), axis=1)
            results['phase_unwrapped'] = phase_unwrapped
        
        if background_phase is not None:
            phase_unwrapped = phase_unwrapped - background_phase
        
        if use_correction:
            t0 = time.time()
            phase_corrected, aberration = self.corrector.correct(
                phase_unwrapped, 
                remove_orders=self.config['remove_aberration_terms']
            )
            results['correction_time'] = time.time() - t0
            results['phase_corrected'] = phase_corrected
            results['aberration'] = aberration
            results['zernike_coeffs'] = self.corrector.get_aberration_terms()
        else:
            results['phase_corrected'] = phase_unwrapped
            results['aberration'] = np.zeros_like(phase_unwrapped)
        
        return results
    
    def phase_to_height(self, phase_map, refractive_index=1.37):
        wavelength = self.config['wavelength']
        delta_n = refractive_index - 1.0
        height = (phase_map * wavelength) / (2 * np.pi * delta_n)
        return height
    
    def analyze_time_series(self, hologram_series):
        print(f"分析时间序列，共 {len(hologram_series)} 帧...")
        
        amplitudes = []
        phases_wrapped = []
        spectra = []
        
        t0 = time.time()
        for i, holo in enumerate(hologram_series):
            amp, phase_wrapped, spec, _ = self.reconstruct(holo, calibrate_first=(i==0))
            amplitudes.append(amp)
            phases_wrapped.append(phase_wrapped)
            spectra.append(spec)
            
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(hologram_series)} 帧")
        
        print(f"  重建完成，耗时 {time.time() - t0:.2f}s")
        
        t0 = time.time()
        phases_unwrapped = temporal_unwrapping(np.array(phases_wrapped), reference_idx=0)
        print(f"  时间域解包裹完成，耗时 {time.time() - t0:.2f}s")
        
        t0 = time.time()
        reference_phase = phases_unwrapped[0]
        phases_corrected = []
        for i, phase in enumerate(phases_unwrapped):
            corrected, _ = self.corrector.correct(
                phase - reference_phase,
                remove_orders=self.config['remove_aberration_terms']
            )
            phases_corrected.append(corrected)
        print(f"  像差校正完成，耗时 {time.time() - t0:.2f}s")
        
        return {
            'amplitudes': np.array(amplitudes),
            'phases_wrapped': np.array(phases_wrapped),
            'phases_unwrapped': np.array(phases_unwrapped),
            'phases_corrected': np.array(phases_corrected),
            'spectra': np.array(spectra)
        }


def create_demo_animation():
    print("创建活细胞动态演示...")
    
    dh = LiveCellDigitalHolography()
    
    num_frames = 30
    hologram_series = []
    true_amplitudes = []
    true_phases = []
    
    print("  生成全息图序列...")
    for t in range(num_frames):
        holo, amp, phase = dh.generate_cell_hologram(
            size=256, cell_type='multiple', time_step=t
        )
        hologram_series.append(holo)
        true_amplitudes.append(amp)
        true_phases.append(phase)
    
    dh.calibrate(hologram_series[0])
    
    results = dh.analyze_time_series(hologram_series)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    im_holo = axes[0, 0].imshow(hologram_series[0], cmap='gray')
    axes[0, 0].set_title('全息图')
    plt.colorbar(im_holo, ax=axes[0, 0])
    
    im_amp = axes[0, 1].imshow(results['amplitudes'][0], cmap='gray')
    axes[0, 1].set_title('重建幅度')
    plt.colorbar(im_amp, ax=axes[0, 1])
    
    im_phase = axes[0, 2].imshow(results['phases_wrapped'][0], cmap='jet')
    axes[0, 2].set_title('包裹相位')
    plt.colorbar(im_phase, ax=axes[0, 2])
    
    im_unwrap = axes[1, 0].imshow(results['phases_unwrapped'][0], cmap='jet')
    axes[1, 0].set_title('解包裹相位')
    plt.colorbar(im_unwrap, ax=axes[1, 0])
    
    im_corrected = axes[1, 1].imshow(results['phases_corrected'][0], cmap='jet')
    axes[1, 1].set_title('校正后相位')
    plt.colorbar(im_corrected, ax=axes[1, 1])
    
    im_height = axes[1, 2].imshow(
        dh.phase_to_height(results['phases_corrected'][0]) * 1e9, 
        cmap='jet'
    )
    axes[1, 2].set_title('细胞高度 (nm)')
    plt.colorbar(im_height, ax=axes[1, 2])
    
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
    
    def update(frame):
        im_holo.set_data(hologram_series[frame])
        im_amp.set_data(results['amplitudes'][frame])
        im_phase.set_data(results['phases_wrapped'][frame])
        im_unwrap.set_data(results['phases_unwrapped'][frame])
        im_corrected.set_data(results['phases_corrected'][frame])
        im_height.set_data(
            dh.phase_to_height(results['phases_corrected'][frame]) * 1e9
        )
        time_text.set_text(f'时间帧: {frame}/{num_frames-1}')
        return [im_holo, im_amp, im_phase, im_unwrap, im_corrected, im_height, time_text]
    
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=200, blit=True
    )
    
    plt.tight_layout()
    print("  动画创建完成!")
    plt.show()
    
    return ani


def comparison_demo():
    print("相位解包裹与像差校正对比演示")
    print("=" * 60)
    
    dh = LiveCellDigitalHolography()
    
    hologram, true_amp, true_phase = dh.generate_cell_hologram(
        size=256, cell_type='multiple', time_step=5
    )
    
    amplitude, phase_wrapped, spectrum, mask = dh.reconstruct(hologram)
    
    quality_map = create_quality_map(phase_wrapped, amplitude)
    
    unwrapper_simple = PhaseUnwrapper(method='simple')
    unwrapper_qg = PhaseUnwrapper(method='quality_guided')
    unwrapper_ls = PhaseUnwrapper(method='least_squares')
    unwrapper_fft = PhaseUnwrapper(method='fft')
    
    print("\n不同相位解包裹方法对比:")
    
    t0 = time.time()
    phase_simple = unwrapper_simple.unwrap(phase_wrapped)
    print(f"  简单解包裹: {time.time()-t0:.3f}s")
    
    t0 = time.time()
    phase_qg = unwrapper_qg.unwrap(phase_wrapped, amplitude)
    print(f"  质量图引导: {time.time()-t0:.3f}s")
    
    t0 = time.time()
    phase_fft = unwrapper_fft.unwrap(phase_wrapped)
    print(f"  FFT法: {time.time()-t0:.3f}s")
    
    corrector = AberrationCorrector(max_order=6)
    phase_corrected, aberration = corrector.correct(phase_qg)
    
    aberration_terms = corrector.get_aberration_terms()
    print("\n主要Zernike像差系数:")
    zernike_names = {
        (0, 0): '平移(Piston)',
        (1, -1): 'Y倾斜(Tilt Y)',
        (1, 1): 'X倾斜(Tilt X)',
        (2, -2): '像散(Astigmatism)',
        (2, 0): '离焦(Defocus)',
        (2, 2): '像散(Astigmatism)',
        (3, -1): '彗差(Coma Y)',
        (3, 1): '彗差(Coma X)',
    }
    for (n, m), coeff in sorted(aberration_terms.items()):
        if (n, m) in zernike_names and abs(coeff) > 0.01:
            print(f"  {zernike_names[(n,m)]} (n={n},m={m}): {coeff:.4f}")
    
    fig = plt.figure(figsize=(20, 12))
    
    plt.subplot(3, 4, 1)
    plt.imshow(hologram[:80, :80], cmap='gray')
    plt.title('全息图(局部)')
    
    plt.subplot(3, 4, 2)
    plt.imshow(amplitude, cmap='gray')
    plt.title('重建幅度')
    plt.colorbar()
    
    plt.subplot(3, 4, 3)
    plt.imshow(quality_map, cmap='viridis')
    plt.title('相位质量图')
    plt.colorbar()
    
    plt.subplot(3, 4, 4)
    plt.imshow(mask, cmap='gray')
    plt.title('滤波掩模')
    
    plt.subplot(3, 4, 5)
    plt.imshow(phase_wrapped, cmap='jet')
    plt.title('包裹相位')
    plt.colorbar()
    
    plt.subplot(3, 4, 6)
    plt.imshow(phase_simple, cmap='jet')
    plt.title('简单解包裹')
    plt.colorbar()
    
    plt.subplot(3, 4, 7)
    plt.imshow(phase_qg, cmap='jet')
    plt.title('质量图引导解包裹')
    plt.colorbar()
    
    plt.subplot(3, 4, 8)
    plt.imshow(phase_fft, cmap='jet')
    plt.title('FFT法解包裹')
    plt.colorbar()
    
    plt.subplot(3, 4, 9)
    plt.imshow(phase_corrected, cmap='jet')
    plt.title('像差校正后相位')
    plt.colorbar()
    
    plt.subplot(3, 4, 10)
    plt.imshow(aberration, cmap='jet')
    plt.title('估计的系统像差')
    plt.colorbar()
    
    plt.subplot(3, 4, 11)
    plt.imshow(dh.phase_to_height(phase_corrected) * 1e9, cmap='jet')
    plt.title('细胞高度图 (nm)')
    plt.colorbar()
    
    plt.subplot(3, 4, 12)
    center_row = 128
    x = np.arange(256)
    plt.plot(x, phase_wrapped[center_row, :], 'k-', label='包裹', alpha=0.5)
    plt.plot(x, phase_simple[center_row, :], 'b--', label='简单')
    plt.plot(x, phase_qg[center_row, :], 'r-', label='质量图引导')
    plt.plot(x, phase_corrected[center_row, :], 'g-', label='校正后', linewidth=2)
    plt.title('中心行相位截面')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unwrapping_comparison.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存至: unwrapping_comparison.png")
    plt.show()


if __name__ == '__main__':
    print("=" * 70)
    print("离轴数字全息术 - 活细胞动态观测系统")
    print("=" * 70)
    print("\n功能:")
    print("  1. 质量图引导相位解包裹")
    print("  2. 最小二乘法/FFT相位解包裹")
    print("  3. Zernike多项式像差校正")
    print("  4. 时间序列分析与动态可视化")
    print()
    
    comparison_demo()
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)
