import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


def generate_off_axis_hologram(size=512):
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    obj_amp = np.exp(-(X**2 + Y**2) / 2) * (1 + 0.5 * np.sin(2 * X))
    obj_phase = 3 * np.exp(-(X**2 + Y**2) / 3)
    
    object_wave = obj_amp * np.exp(1j * obj_phase)
    
    k = 2 * np.pi
    theta = 0.3
    ref_wave = np.exp(1j * k * X * np.sin(theta))
    
    hologram = np.abs(object_wave + ref_wave)**2
    
    return hologram, obj_amp, obj_phase


def reconstruct_hologram(hologram):
    spectrum = fftshift(fft2(hologram))
    
    rows, cols = hologram.shape
    cy, cx = rows // 2, cols // 2
    
    roi_size = rows // 4
    dx = int(rows * 0.15)
    
    x1, x2 = cx + dx - roi_size//2, cx + dx + roi_size//2
    y1, y2 = cy - roi_size//2, cy + roi_size//2
    
    mask = np.zeros_like(spectrum, dtype=bool)
    mask[y1:y2, x1:x2] = True
    
    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[mask] = spectrum[mask]
    
    reconstructed = ifft2(ifftshift(filtered_spectrum))
    
    amplitude = np.abs(reconstructed)
    phase = np.angle(reconstructed)
    phase_unwrapped = np.unwrap(np.unwrap(phase, axis=0), axis=1)
    
    return spectrum, filtered_spectrum, amplitude, phase, phase_unwrapped


def main():
    print("离轴数字全息图重建演示")
    print("=" * 50)
    
    hologram, true_amp, true_phase = generate_off_axis_hologram(512)
    spectrum, filtered_spec, amp, phase, phase_unwrap = reconstruct_hologram(hologram)
    
    fig = plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(hologram[:100, :100], cmap='gray')
    plt.title('全息图(局部放大)')
    
    plt.subplot(2, 4, 2)
    plt.imshow(np.log(np.abs(spectrum) + 1), cmap='gray')
    plt.title('傅里叶频谱')
    plt.scatter([spectrum.shape[1]//2 + int(512*0.15)], [spectrum.shape[0]//2], 
                c='r', s=100, marker='o', facecolors='none', linewidths=2, label='+1级')
    plt.legend()
    
    plt.subplot(2, 4, 3)
    plt.imshow(np.log(np.abs(filtered_spec) + 1), cmap='gray')
    plt.title('滤波后(提取+1级)')
    
    plt.subplot(2, 4, 4)
    plt.imshow(amp, cmap='gray')
    plt.title('重建幅度')
    
    plt.subplot(2, 4, 5)
    plt.imshow(true_amp, cmap='gray')
    plt.title('原始幅度(对比)')
    
    plt.subplot(2, 4, 6)
    plt.imshow(phase, cmap='jet')
    plt.title('包裹相位')
    
    plt.subplot(2, 4, 7)
    plt.imshow(phase_unwrap, cmap='jet')
    plt.title('解包裹相位')
    
    plt.subplot(2, 4, 8)
    plt.imshow(true_phase, cmap='jet')
    plt.title('原始相位(对比)')
    
    plt.tight_layout()
    plt.savefig('reconstruction_result.png', dpi=150, bbox_inches='tight')
    print("结果已保存至 reconstruction_result.png")
    plt.show()
    
    print("\n重建原理说明:")
    print("1. 全息图 = |物波 + 参考波|²")
    print("2. FFT后频谱包含: 零级(中心)、+1级(物波信息)、-1级(共轭像)")
    print("3. 提取+1级衍射斑进行逆FFT，恢复复振幅")
    print("4. 从复振幅中分离幅度和相位")


if __name__ == '__main__':
    main()
