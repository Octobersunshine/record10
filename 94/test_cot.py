import numpy as np
from cot import OrderTracking, generate_test_signal

print("测试阶次跟踪模块...")

fs = 1000
duration = 2

t, vibration, tach_signal, true_rpm = generate_test_signal(
    fs=fs,
    duration=duration,
    rpm_start=600,
    rpm_end=1200,
    orders=[1, 2, 3],
    amplitudes=[1, 0.5, 0.3],
    noise_level=0.1
)

print(f"信号长度: {len(vibration)} 采样点")
print(f"转速范围: {true_rpm[0]:.1f} -> {true_rpm[-1]:.1f} RPM")

cot = OrderTracking(fs=fs, vibration_signal=vibration, tach_signal=tach_signal)
rpm = cot.calculate_instantaneous_rpm()

print(f"\n瞬时转速计算完成")
print(f"  计算转速范围: {rpm.min():.1f} -> {rpm.max():.1f} RPM")
print(f"  转速误差: {np.mean(np.abs(rpm - true_rpm)):.2f} RPM")

angular_axis, resampled_signal = cot.compute_order_tracking(orders_per_rev=360)
print(f"\n角域重采样完成")
print(f"  角域采样点数: {len(resampled_signal)}")
print(f"  总转角: {angular_axis[-1] / (2 * np.pi):.2f} 转")

orders, magnitude = cot.get_order_spectrum()
print(f"\n阶次谱计算完成")
print(f"  阶次范围: 0 -> {orders[-1]:.1f}")

peak_indices = np.argsort(magnitude)[-5:][::-1]
print(f"\n主要阶次峰值:")
for idx in peak_indices[:3]:
    print(f"  阶次 {orders[idx]:.2f}X - 幅值 {magnitude[idx]:.4f}")

print("\n✓ 阶次跟踪模块测试成功!")
