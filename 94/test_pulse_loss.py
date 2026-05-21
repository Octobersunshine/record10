import numpy as np
import matplotlib.pyplot as plt
from cot import OrderTracking, generate_test_signal_with_missing_pulses


def test_pulse_loss_correction():
    print("=" * 60)
    print("脉冲丢失场景下的阶次跟踪修复效果测试")
    print("=" * 60)

    fs = 10000
    duration = 5

    missing_pulses = [50, 100, 150, 200, 250]
    print(f"\n测试配置:")
    print(f"  采样率: {fs} Hz")
    print(f"  时长: {duration} s")
    print(f"  转速范围: 600 -> 3000 RPM")
    print(f"  真实阶次: 1, 2, 3, 4.5")
    print(f"  人工丢失脉冲位置: {missing_pulses}")

    t, vibration, tach_signal, true_rpm = generate_test_signal_with_missing_pulses(
        fs=fs,
        duration=duration,
        rpm_start=600,
        rpm_end=3000,
        orders=[1, 2, 3, 4.5],
        amplitudes=[1, 0.5, 0.3, 0.2],
        noise_level=0.05,
        missing_pulse_indices=missing_pulses
    )

    pulse_count_original = np.sum(tach_signal == 1)
    print(f"\n原始信号脉冲数: {pulse_count_original}")

    print("\n" + "-" * 40)
    print("方法1: 原始阶次跟踪 (未修复)")
    print("-" * 40)

    cot_original = OrderTracking(fs=fs, vibration_signal=vibration, tach_signal=tach_signal)
    try:
        rpm_original = cot_original.calculate_instantaneous_rpm()
        angular_axis_original, resampled_original = cot_original.compute_order_tracking(orders_per_rev=720)
        orders_original, magnitude_original = cot_original.get_order_spectrum()

        rpm_error_original = np.mean(np.abs(rpm_original - true_rpm))
        print(f"  平均转速误差: {rpm_error_original:.2f} RPM")
        print(f"  最大转速误差: {np.max(np.abs(rpm_original - true_rpm)):.2f} RPM")
    except Exception as e:
        print(f"  原始方法出错: {e}")
        rpm_original = None
        orders_original = None
        magnitude_original = None

    print("\n" + "-" * 40)
    print("方法2: 鲁棒阶次跟踪 (已修复)")
    print("-" * 40)

    cot_robust = OrderTracking(fs=fs, vibration_signal=vibration, tach_signal=tach_signal)
    try:
        rpm_robust = cot_robust.calculate_instantaneous_rpm_robust(
            anomaly_threshold=3.0,
            anomaly_method='mad',
            smooth_method='spline'
        )
        angular_axis_robust, resampled_robust = cot_robust.compute_order_tracking_robust(orders_per_rev=720)
        orders_robust, magnitude_robust = cot_robust.get_order_spectrum()

        print(f"  {cot_robust.get_correction_summary()}")
        rpm_error_robust = np.mean(np.abs(rpm_robust - true_rpm))
        print(f"  平均转速误差: {rpm_error_robust:.2f} RPM")
        print(f"  最大转速误差: {np.max(np.abs(rpm_robust - true_rpm)):.2f} RPM")

        if rpm_original is not None:
            improvement = (rpm_error_original - rpm_error_robust) / rpm_error_original * 100
            print(f"  误差降低: {improvement:.1f}%")
    except Exception as e:
        print(f"  鲁棒方法出错: {e}")
        rpm_robust = None
        orders_robust = None
        magnitude_robust = None

    print("\n" + "=" * 60)
    print("生成可视化对比图...")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, tach_signal, 'b-', linewidth=0.5)
    ax1.set_title('转速计信号 (含丢失脉冲)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    pulse_positions = np.where(tach_signal == 1)[0] / fs
    ax2.plot(pulse_positions, np.ones_like(pulse_positions), '|', markersize=15, color='b', label='检测到的脉冲')
    for idx in missing_pulses:
        if idx < len(pulse_positions):
            ax2.axvline(x=pulse_positions[idx], color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax2.set_title('脉冲位置 (红虚线标记丢失脉冲)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('时间 (s)')
    ax2.set_yticks([])
    ax2.set_xlim(0, 2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    if rpm_original is not None:
        ax3.plot(t, true_rpm, 'g-', linewidth=2, label='真实转速')
        ax3.plot(t, rpm_original, 'r--', linewidth=1.5, alpha=0.7, label='原始方法')
        ax3.set_title('转速曲线 - 原始方法', fontsize=10, fontweight='bold')
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('转速 (RPM)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if rpm_robust is not None:
        ax4.plot(t, true_rpm, 'g-', linewidth=2, label='真实转速')
        ax4.plot(t, rpm_robust, 'b--', linewidth=1.5, alpha=0.7, label='鲁棒方法')
        ax4.set_title('转速曲线 - 鲁棒方法', fontsize=10, fontweight='bold')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('转速 (RPM)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    if orders_original is not None:
        ax5.plot(orders_original, magnitude_original, 'r-', linewidth=0.8)
        for order in [1, 2, 3, 4.5]:
            ax5.axvline(x=order, color='g', linestyle='--', alpha=0.6, linewidth=1.5)
        ax5.set_title('阶次谱 - 原始方法 (注意假峰)', fontsize=10, fontweight='bold')
        ax5.set_xlabel('阶次')
        ax5.set_ylabel('幅值')
        ax5.set_xlim(0, 10)
        ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    if orders_robust is not None:
        ax6.plot(orders_robust, magnitude_robust, 'b-', linewidth=0.8)
        for order in [1, 2, 3, 4.5]:
            ax6.axvline(x=order, color='g', linestyle='--', alpha=0.6, linewidth=1.5)
        ax6.set_title('阶次谱 - 鲁棒方法 (假峰消除)', fontsize=10, fontweight='bold')
        ax6.set_xlabel('阶次')
        ax6.set_ylabel('幅值')
        ax6.set_xlim(0, 10)
        ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[3, :])
    if rpm_original is not None and rpm_robust is not None:
        error_original = np.abs(rpm_original - true_rpm)
        error_robust = np.abs(rpm_robust - true_rpm)
        ax7.plot(t, error_original, 'r-', linewidth=0.8, label=f'原始方法 (平均: {np.mean(error_original):.1f} RPM)')
        ax7.plot(t, error_robust, 'b-', linewidth=0.8, label=f'鲁棒方法 (平均: {np.mean(error_robust):.1f} RPM)')
        ax7.set_title('转速误差对比', fontsize=10, fontweight='bold')
        ax7.set_xlabel('时间 (s)')
        ax7.set_ylabel('误差 (RPM)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    plt.suptitle('脉冲丢失场景下阶次跟踪修复效果对比', fontsize=14, fontweight='bold', y=0.99)
    plt.savefig('pulse_loss_comparison.png', dpi=150, bbox_inches='tight')
    print("对比图已保存为: pulse_loss_comparison.png")
    plt.show()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


def test_severe_pulse_loss():
    print("\n\n" + "=" * 60)
    print("严重脉冲丢失场景测试 (10%丢失)")
    print("=" * 60)

    fs = 10000
    duration = 3

    np.random.seed(42)
    num_pulses = int((600 + 1800) / 2 / 60 * duration)
    missing_pulses = np.random.choice(range(20, num_pulses - 20), size=int(num_pulses * 0.1), replace=False)
    missing_pulses = sorted(missing_pulses)

    print(f"丢失脉冲数量: {len(missing_pulses)} ({len(missing_pulses)/num_pulses*100:.1f}%)")

    t, vibration, tach_signal, true_rpm = generate_test_signal_with_missing_pulses(
        fs=fs,
        duration=duration,
        rpm_start=600,
        rpm_end=1800,
        orders=[1, 2, 3],
        amplitudes=[1, 0.5, 0.3],
        noise_level=0.05,
        missing_pulse_indices=missing_pulses
    )

    cot_original = OrderTracking(fs=fs, vibration_signal=vibration, tach_signal=tach_signal)
    cot_robust = OrderTracking(fs=fs, vibration_signal=vibration, tach_signal=tach_signal)

    try:
        rpm_original = cot_original.calculate_instantaneous_rpm()
        rpm_robust = cot_robust.calculate_instantaneous_rpm_robust()

        print(f"\n原始方法 - 平均误差: {np.mean(np.abs(rpm_original - true_rpm)):.2f} RPM")
        print(f"鲁棒方法 - 平均误差: {np.mean(np.abs(rpm_robust - true_rpm)):.2f} RPM")
        print(f"修正统计: {cot_robust.get_correction_summary()}")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(t, true_rpm, 'g-', linewidth=2, label='真实转速')
        axes[0].plot(t, rpm_original, 'r--', linewidth=1, alpha=0.7, label='原始方法')
        axes[0].set_title('严重脉冲丢失 - 原始方法', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, true_rpm, 'g-', linewidth=2, label='真实转速')
        axes[1].plot(t, rpm_robust, 'b--', linewidth=1, alpha=0.7, label='鲁棒方法')
        axes[1].set_title('严重脉冲丢失 - 鲁棒方法', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('severe_pulse_loss.png', dpi=150)
        print("严重丢失测试图已保存为: severe_pulse_loss.png")
        plt.show()

    except Exception as e:
        print(f"测试出错: {e}")


def main():
    test_pulse_loss_correction()
    test_severe_pulse_loss()


if __name__ == '__main__':
    main()
