import numpy as np
import matplotlib.pyplot as plt
from cot import OrderTracking, generate_test_signal


def example_basic_cot():
    print("=== 阶次跟踪基础示例 ===")

    fs = 10000
    duration = 5

    print(f"生成测试信号...")
    print(f"采样率: {fs} Hz, 时长: {duration} s")
    print(f"转速范围: 600 -> 3000 RPM (升速)")
    print(f"包含阶次: 1, 2, 3, 4.5")

    t, vibration, tach_signal, true_rpm = generate_test_signal(
        fs=fs,
        duration=duration,
        rpm_start=600,
        rpm_end=3000,
        orders=[1, 2, 3, 4.5],
        amplitudes=[1, 0.5, 0.3, 0.2],
        noise_level=0.1
    )

    cot = OrderTracking(fs=fs, vibration_signal=vibration, tach_signal=tach_signal)

    print("\n计算瞬时转速...")
    rpm = cot.calculate_instantaneous_rpm(pulses_per_rev=1)

    print("\n执行阶次跟踪重采样...")
    angular_axis, resampled_signal = cot.compute_order_tracking(orders_per_rev=720)
    print(f"角域采样点数: {len(resampled_signal)}")

    print("\n计算阶次谱...")
    orders, magnitude = cot.get_order_spectrum()

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('阶次跟踪(COT)分析结果', fontsize=16, fontweight='bold')

    axes[0, 0].plot(t, vibration)
    axes[0, 0].set_title('时域振动信号')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('幅值')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, true_rpm, 'r-', label='真实转速', linewidth=2)
    axes[0, 1].plot(t, rpm, 'b--', label='计算转速', alpha=0.7)
    axes[0, 1].set_title('转速曲线')
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('转速 (RPM)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(angular_axis / (2 * np.pi), resampled_signal)
    axes[1, 0].set_title('角域重采样信号')
    axes[1, 0].set_xlabel('转角 (转)')
    axes[1, 0].set_ylabel('幅值')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(orders, magnitude)
    axes[1, 1].set_title('阶次谱')
    axes[1, 1].set_xlabel('阶次')
    axes[1, 1].set_ylabel('幅值')
    axes[1, 1].set_xlim(0, 10)
    axes[1, 1].grid(True, alpha=0.3)

    for order in [1, 2, 3, 4.5]:
        axes[1, 1].axvline(x=order, color='r', linestyle='--', alpha=0.5, linewidth=1)
        axes[1, 1].text(order + 0.1, np.max(magnitude) * 0.9, f'{order}X', color='r')

    freqs = np.fft.fftfreq(len(vibration), 1/fs)[:len(vibration)//2]
    fft_magnitude = 2 * np.abs(np.fft.fft(vibration * np.hanning(len(vibration))))[:len(vibration)//2] / len(vibration)

    axes[2, 0].plot(freqs, fft_magnitude)
    axes[2, 0].set_title('传统频谱 (频域)')
    axes[2, 0].set_xlabel('频率 (Hz)')
    axes[2, 0].set_ylabel('幅值')
    axes[2, 0].set_xlim(0, 500)
    axes[2, 0].grid(True, alpha=0.3)

    spectrogram_data = []
    window_size = fs
    hop_size = fs // 4
    n_windows = (len(vibration) - window_size) // hop_size + 1

    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size
        segment = vibration[start:end] * np.hanning(window_size)
        spec = 2 * np.abs(np.fft.fft(segment))[:window_size//2] / window_size
        spectrogram_data.append(spec)

    spectrogram_data = np.array(spectrogram_data).T
    time_spec = np.arange(n_windows) * hop_size / fs
    freq_spec = np.fft.fftfreq(window_size, 1/fs)[:window_size//2]

    im = axes[2, 1].pcolormesh(time_spec, freq_spec, spectrogram_data,
                               cmap='jet', shading='auto')
    axes[2, 1].set_title('瀑布图 (时频分析)')
    axes[2, 1].set_xlabel('时间 (s)')
    axes[2, 1].set_ylabel('频率 (Hz)')
    axes[2, 1].set_ylim(0, 500)
    plt.colorbar(im, ax=axes[2, 1], label='幅值')

    rpm_line = (true_rpm[::hop_size] / 60)
    for order in [1, 2, 3]:
        axes[2, 1].plot(time_spec, rpm_line * order, 'w--', alpha=0.7, linewidth=1)

    plt.tight_layout()
    plt.savefig('cot_analysis.png', dpi=150, bbox_inches='tight')
    print("\n分析结果已保存到 cot_analysis.png")
    plt.show()

    print("\n=== 分析完成 ===")
    print(f"最大转速误差: {np.max(np.abs(rpm - true_rpm)):.2f} RPM")
    print(f"平均转速误差: {np.mean(np.abs(rpm - true_rpm)):.2f} RPM")


def example_gearbox():
    print("\n=== 齿轮箱故障模拟示例 ===")

    fs = 20000
    duration = 8

    z1 = 20
    z2 = 40
    gear_mesh_order = z1
    sideband_orders = [gear_mesh_order - 1, gear_mesh_order, gear_mesh_order + 1]

    print(f"齿轮参数: 小齿轮 {z1}齿, 大齿轮 {z2}齿")
    print(f"啮合阶次: {gear_mesh_order}X (带边带)")

    t, vibration, tach_signal, true_rpm = generate_test_signal(
        fs=fs,
        duration=duration,
        rpm_start=300,
        rpm_end=1500,
        orders=sideband_orders,
        amplitudes=[0.2, 1.0, 0.2],
        noise_level=0.05
    )

    cot = OrderTracking(fs=fs, vibration_signal=vibration, rpm_signal=true_rpm)

    angular_axis, resampled_signal = cot.compute_order_tracking(orders_per_rev=1024)
    orders, magnitude = cot.get_order_spectrum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('齿轮箱阶次跟踪分析', fontsize=16, fontweight='bold')

    axes[0, 0].plot(t, true_rpm, 'r-')
    axes[0, 0].set_title('转速曲线')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('转速 (RPM)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(angular_axis / (2 * np.pi), resampled_signal)
    axes[0, 1].set_title('角域重采样信号')
    axes[0, 1].set_xlabel('转角 (转)')
    axes[0, 1].set_ylabel('幅值')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(orders, magnitude)
    axes[1, 0].set_title('阶次谱 (全范围)')
    axes[1, 0].set_xlabel('阶次')
    axes[1, 0].set_ylabel('幅值')
    axes[1, 0].set_xlim(0, 50)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(orders, magnitude)
    axes[1, 1].set_title(f'阶次谱 (啮合阶次附近 {gear_mesh_order}X)')
    axes[1, 1].set_xlabel('阶次')
    axes[1, 1].set_ylabel('幅值')
    axes[1, 1].set_xlim(gear_mesh_order - 5, gear_mesh_order + 5)
    axes[1, 1].grid(True, alpha=0.3)

    for sb_order in sideband_orders:
        axes[1, 1].axvline(x=sb_order, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('gearbox_cot.png', dpi=150, bbox_inches='tight')
    print("齿轮箱分析结果已保存到 gearbox_cot.png")
    plt.show()


def example_bearing():
    print("\n=== 轴承故障阶次分析示例 ===")

    fs = 25000
    duration = 6

    bpfo_order = 3.5
    bpfi_order = 5.4
    ftf_order = 0.4

    print(f"轴承特征阶次:")
    print(f"  BPFO (外圈故障): {bpfo_order}X")
    print(f"  BPFI (内圈故障): {bpfi_order}X")
    print(f"  FTF (保持架故障): {ftf_order}X")

    t, vibration, tach_signal, true_rpm = generate_test_signal(
        fs=fs,
        duration=duration,
        rpm_start=500,
        rpm_end=2500,
        orders=[bpfo_order, bpfi_order, ftf_order],
        amplitudes=[0.8, 0.6, 0.3],
        noise_level=0.15
    )

    cot = OrderTracking(fs=fs, vibration_signal=vibration, rpm_signal=true_rpm)
    angular_axis, resampled_signal = cot.compute_order_tracking(orders_per_rev=2048)
    orders, magnitude = cot.get_order_spectrum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(orders, magnitude, linewidth=1)
    ax.set_title('轴承故障阶次谱分析', fontsize=14, fontweight='bold')
    ax.set_xlabel('阶次')
    ax.set_ylabel('幅值')
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)

    for order, name, color in [(bpfo_order, 'BPFO', 'r'),
                               (bpfi_order, 'BPFI', 'g'),
                               (ftf_order, 'FTF', 'b')]:
        ax.axvline(x=order, color=color, linestyle='--', alpha=0.7, linewidth=2)
        ax.text(order + 0.1, np.max(magnitude) * 0.9, f'{name}\n{order}X',
                color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig('bearing_cot.png', dpi=150, bbox_inches='tight')
    print("轴承分析结果已保存到 bearing_cot.png")
    plt.show()


if __name__ == '__main__':
    example_basic_cot()
    example_gearbox()
    example_bearing()
