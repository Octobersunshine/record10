import numpy as np
from gs_algorithm import fft2_normalized, ifft2_normalized

np.random.seed(42)

print("=" * 60)
print("测试FFT能量守恒特性")
print("=" * 60)

x = np.random.rand(256, 256)
print(f"\n输入能量 (空域): {np.sum(np.abs(x)**2):.6f}")

X = fft2_normalized(x)
print(f"FFT后能量 (频域): {np.sum(np.abs(X)**2):.6f}")
print(f"能量差: {np.abs(np.sum(np.abs(x)**2) - np.sum(np.abs(X)**2)):.6e}")

x_recon = ifft2_normalized(X)
print(f"\nIFFT后能量 (空域): {np.sum(np.abs(x_recon)**2):.6f}")
print(f"重建误差: {np.max(np.abs(x - x_recon.real)):.6e}")

print("\n" + "=" * 60)
print("测试迭代过程中的能量变化")
print("=" * 60)

intensity = np.random.rand(256, 256)
measured_amp = np.sqrt(intensity)
target_energy = np.sum(intensity)
print(f"\n目标能量: {target_energy:.6f}")

phase = np.random.rand(*intensity.shape) * 2 * np.pi
complex_field = measured_amp * np.exp(1j * phase)
print(f"初始能量: {np.sum(np.abs(complex_field)**2):.6f}")

print("\n迭代过程中的能量变化:")
for i in range(5):
    spatial_field = ifft2_normalized(complex_field)
    energy1 = np.sum(np.abs(spatial_field)**2)
    
    spatial_amp = np.abs(spatial_field)
    spatial_phase = np.angle(spatial_field)
    
    complex_field = fft2_normalized(spatial_amp * np.exp(1j * spatial_phase))
    energy2 = np.sum(np.abs(complex_field)**2)
    
    current_amp = np.abs(complex_field)
    current_phase = np.angle(complex_field)
    
    complex_field = measured_amp * np.exp(1j * current_phase)
    energy3 = np.sum(np.abs(complex_field)**2)
    
    print(f"迭代 {i+1}: IFFT后={energy1:.6f}, FFT后={energy2:.6f}, 振幅约束后={energy3:.6f}")
