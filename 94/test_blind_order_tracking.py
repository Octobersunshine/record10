import numpy as np
import matplotlib.pyplot as plt
from cot import OrderTracking, generate_test_signal


def test_blind_order_tracking():
    print("=" * 60)
    print("无转速计盲阶次跟踪测试")
    print("=" * 60)

    fs = 5000
    duration = 8

    print(f"\n测试参数:")
    print(f"  采样率: {fs} Hz")
    print(f"  时长: {duration} s")
    print(f"  转速范围: 300 -> 1800 RPM")
    print(f"  包含阶次: 1, 2, 3, 4.5")

    t, vibration, _, true_rpm = generate_test_signal(
        fs=fs,
        duration=duration,
        rpm_start=300,
        rpm_end=1800,
        orders=[1, 2, 3, 4.5],
        amplitudes=[1.0, 0.7, 0.5, 0.3],
        noise_level=0.08
    )

    print(f"\n信号长度: {len(vibration)} 采样点")
    print(f"真实转速范围: {true_rpm[0]:.1f} -> {true_rpm[-1]:.1f} RPM")

    cot = OrderTracking(fs=fs, vibration_signal=vibration)

    print("\n" + "-" * 40)
    print("方法1: 单阶次脊线跟踪 (order_hint=1)")
    print("-" * 40)

    try:
        angular_axis, resampled = cot.compute_blind_order_tracking(
            orders_per_rev=360,
            order_hint=1,
            nperseg=1024,
            n_peaks=10,
            max_accel=500,
            method='ridge'
        )

        orders, magnitude = cot.get_order_spectrum()

        rpm_error = np.mean(np.abs(cot.instantaneous_rpm - true_rpm))
        print(f"  ✓ 转速估计成功!")
        print(f"  平均转速误差: {rpm_error:.1f} RPM")
        print(f"  相对误差: {rpm_error / np.mean(true_rpm) * 100:.1f}%")

        rpm_est_1 = cot.instantaneous_rpm.copy()
        orders_1 = orders.copy()
        magnitude_1 = magnitude.copy()

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        rpm_est_1 = None
        orders_1 = None
        magnitude_1 = None

    print("\n" + "-" * 40)
    print("方法2: 多阶次融合跟踪")
    print("-" * 40)

    try:
        angular_axis_multi, resampled_multi = cot.compute_blind_order_tracking(
            orders_per_rev=360,
            order_hint=1,
            nperseg=1024,
            n_peaks=15,
            max_accel=800,
            method='multi_order'
        )

        orders_multi, magnitude_multi = cot.get_order_spectrum()

        rpm_error_multi = np.mean(np.abs(cot.instantaneous_rpm - true_rpm))
        print(f"  ✓ 多阶次融合成功!")
        print(f"  平均转速误差: {rpm_error_multi:.1f} RPM")
        print(f"  相对误差: {rpm_error_multi / np.mean(true_rpm) * 100:.1f}%")

        rpm_est_multi = cot.instantaneous_rpm.copy()

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        rpm_est_multi = None

    print("\n" + "=" * 60)
    print("生成可视化对比图...")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    freqs, times, stft_mag = cot.compute_stft(nperseg=1024)
    im = ax1.pcolormesh(times, freqs, 20 * np.log10(stft_mag + 1e-10),
                        cmap='jet', shading='auto', vmin=-60, vmax=0)
    ax1.set_title('时频图 (STFT)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('频率 (Hz)')
    ax1.set_ylim(0, 200)
    plt.colorbar(im, ax=ax1, label='幅值 (dB)')

    ax2 = fig.add_subplot(gs[0, 1])
    if hasattr(cot, 'peak_candidates'):
        peak_times = [p['time'] for p in cot.peak_candidates]
        peak_freqs = [p['freq'] for p in cot.peak_candidates]
        ax2.scatter(peak_times, peak_freqs, s=3, c='red', alpha=0.5, label='检测峰值')
        ax2.set_title('时频峰值检测', fontsize=10, fontweight='bold')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('频率 (Hz)')
        ax2.set_ylim(0, 200)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, true_rpm, 'g-', linewidth=2, label='真实转速')
    if rpm_est_1 is not None:
        ax3.plot(t, rpm_est_1, 'r--', linewidth=1.5, alpha=0.8, label='单阶次估计')
    ax3.set_title('转速曲线对比 - 单阶次法', fontsize=10, fontweight='bold')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('转速 (RPM)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, true_rpm, 'g-', linewidth=2, label='真实转速')
    if rpm_est_multi is not None:
        ax4.plot(t, rpm_est_multi, 'b--', linewidth=1.5, alpha=0.8, label='多阶次融合')
    ax4.set_title('转速曲线对比 - 多阶次融合', fontsize=10, fontweight='bold')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('转速 (RPM)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    if orders_1 is not None:
        ax5.plot(orders_1, magnitude_1, 'r-', linewidth=1)
        for order in [1, 2, 3, 4.5]:
            ax5.axvline(x=order, color='g', linestyle='--', alpha=0.6, linewidth=1.5)
        ax5.set_title('阶次谱 (盲阶次跟踪)', fontsize=10, fontweight='bold')
        ax5.set_xlabel('阶次')
        ax5.set_ylabel('幅值')
        ax5.set_xlim(0, 10)
        ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    if rpm_est_1 is not None:
        error = np.abs(rpm_est_1 - true_rpm)
        ax6.plot(t, error, 'r-', linewidth=0.8, label='单阶次')
    if rpm_est_multi is not None:
        error_multi = np.abs(rpm_est_multi - true_rpm)
        ax6.plot(t, error_multi, 'b-', linewidth=0.8, label='多阶次融合')
    ax6.set_title('转速估计误差', fontsize=10, fontweight='bold')
    ax6.set_xlabel('时间 (s)')
    ax6.set_ylabel('误差 (RPM)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('无转速计盲阶次跟踪效果', fontsize=14, fontweight='bold', y=0.99)
    plt.savefig('blind_order_tracking.png', dpi=150, bbox_inches='tight')
    print("对比图已保存为: blind_order_tracking.png")
    plt.show()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


def test_different_speed_profiles():
    print("\n\n" + "=" * 60)
    print("不同转速曲线测试")
    print("=" * 60)

    fs = 5000
    duration = 6

    test_cases = [
        {
            'name': '线性升速',
            'rpm_start': 200,
            'rpm_end': 2000,
            'orders': [1, 2, 3]
        },
        {
            'name': '线性降速',
            'rpm_start': 2000,
            'rpm_end': 200,
            'orders': [1, 2, 3, 5]
        },
        {
            'name': '正弦变速',
            'rpm_start': 600,
            'rpm_end': 600,
            'orders': [1, 2]
        }
    ]

    results = []

    for case in test_cases:
        print(f"\n测试: {case['name']}")

        if case['name'] == '正弦变速':
            t = np.arange(fs * duration) / fs
            instantaneous_rpm = 600 + 400 * np.sin(2 * np.pi * 0.1 * t)
            angular_velocity = 2 * np.pi * instantaneous_rpm / 60
            angle = np.cumsum(angular_velocity) / fs

            vibration = np.zeros_like(t)
            for order in case['orders']:
                vibration += 1.0 / order * np.sin(order * angle)
            vibration += 0.05 * np.random.randn(len(vibration))
        else:
            t, vibration, _, instantaneous_rpm = generate_test_signal(
                fs=fs,
                duration=duration,
                rpm_start=case['rpm_start'],
                rpm_end=case['rpm_end'],
                orders=case['orders'],
                amplitudes=[1.0 / o for o in case['orders']],
                noise_level=0.05
            )

        cot = OrderTracking(fs=fs, vibration_signal=vibration)

        try:
            cot.compute_blind_order_tracking(
                orders_per_rev=360,
                order_hint=1,
                nperseg=1024,
                n_peaks=10,
                max_accel=1000,
                method='ridge'
            )

            rpm_error = np.mean(np.abs(cot.instantaneous_rpm - instantaneous_rpm))
            rel_error = rpm_error / np.mean(instantaneous_rpm) * 100

            print(f"  ✓ 成功!")
            print(f"    平均误差: {rpm_error:.1f} RPM")
            print(f"    相对误差: {rel_error:.1f}%")

            results.append({
                'name': case['name'],
                'true_rpm': instantaneous_rpm,
                'est_rpm': cot.instantaneous_rpm,
                'error': rpm_error,
                'rel_error': rel_error
            })

        except Exception as e:
            print(f"  ✗ 失败: {e}")

    if len(results) > 0:
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
        if len(results) == 1:
            axes = [axes]

        for i, result in enumerate(results):
            axes[i].plot(t, result['true_rpm'], 'g-', linewidth=2, label='真实转速')
            axes[i].plot(t, result['est_rpm'], 'r--', linewidth=1.5, label='估计转速')
            axes[i].set_title(f"{result['name']} (误差: {result['error']:.1f} RPM)", fontweight='bold')
            axes[i].set_xlabel('时间 (s)')
            axes[i].set_ylabel('转速 (RPM)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('speed_profiles_test.png', dpi=150)
        print(f"\n转速曲线测试图已保存为: speed_profiles_test.png")
        plt.show()


def main():
    test_blind_order_tracking()
    test_different_speed_profiles()


if __name__ == '__main__':
    main()
