import numpy as np
import matplotlib.pyplot as plt
from radial_velocity import (
    RadarParameters,
    IQDataSimulator,
    PulsePairProcessor,
    VelocityDealiaser,
)


def test_basic_velocity_retrieval():
    print("=" * 60)
    print("测试1: 基本速度反演")
    print("=" * 60)

    params = RadarParameters(wavelength=0.1, prf=1000)
    print(f"雷达参数: 波长={params.wavelength}m, PRF={params.prf}Hz")
    print(f"Nyquist速度: {params.nyquist_velocity:.2f} m/s")

    simulator = IQDataSimulator(params, num_gates=5, num_pulses=128)
    true_velocities = np.array([-20, -5, 0, 5, 20])

    iq_data = simulator.simulate_iq(true_velocities, snr_db=30)
    processor = PulsePairProcessor(params)
    results = processor.process(iq_data)

    retrieved_velocities = results['velocity']

    print("\n距离库 | 真实速度(m/s) | 反演速度(m/s) | 误差(m/s)")
    print("-" * 55)
    for i in range(len(true_velocities)):
        error = retrieved_velocities[i] - true_velocities[i]
        print(f"{i:4d}   | {true_velocities[i]:12.2f}   | {retrieved_velocities[i]:12.2f}   | {error:8.4f}")

    rmse = np.sqrt(np.mean((retrieved_velocities - true_velocities) ** 2))
    print(f"\nRMSE: {rmse:.4f} m/s")

    return rmse < 0.5


def test_velocity_profile():
    print("\n" + "=" * 60)
    print("测试2: 连续速度剖面反演")
    print("=" * 60)

    params = RadarParameters(wavelength=0.1, prf=1000)
    num_gates = 200

    simulator = IQDataSimulator(params, num_gates=num_gates, num_pulses=64)

    true_velocities = np.linspace(-20, 20, num_gates)
    iq_data = simulator.simulate_iq(true_velocities, snr_db=25)

    processor = PulsePairProcessor(params)
    results = processor.process(iq_data)
    retrieved_velocities = results['velocity']

    rmse = np.sqrt(np.mean((retrieved_velocities - true_velocities) ** 2))
    print(f"速度剖面反演 RMSE: {rmse:.4f} m/s")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(true_velocities, label='真实速度', linewidth=2)
    axes[0].plot(retrieved_velocities, label='反演速度', linestyle='--', alpha=0.8)
    axes[0].set_xlabel('距离库')
    axes[0].set_ylabel('径向速度 (m/s)')
    axes[0].set_title('速度剖面对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(retrieved_velocities - true_velocities, 'r-', alpha=0.7)
    axes[1].set_xlabel('距离库')
    axes[1].set_ylabel('误差 (m/s)')
    axes[1].set_title('反演误差')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('velocity_profile_test.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 velocity_profile_test.png")

    return rmse < 1.0


def test_velocity_dealiasing():
    print("\n" + "=" * 60)
    print("测试3: 速度折叠与解算")
    print("=" * 60)

    params = RadarParameters(wavelength=0.1, prf=500)
    nyquist = params.nyquist_velocity
    print(f"Nyquist速度: {nyquist:.2f} m/s")

    num_gates = 100
    true_velocities = np.linspace(-30, 30, num_gates)

    simulator = IQDataSimulator(params, num_gates=num_gates, num_pulses=128)
    iq_data = simulator.simulate_iq(true_velocities, snr_db=30)

    processor = PulsePairProcessor(params)
    aliased_velocities = processor.compute_velocity(iq_data)

    dealiaser = VelocityDealiaser(params)
    dealiased_velocities = dealiaser.dealias_1d(aliased_velocities)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(true_velocities, 'b-', label='真实速度', linewidth=2)
    axes[0].axhline(y=nyquist, color='r', linestyle='--', label=f'+Nyquist ({nyquist:.1f} m/s)')
    axes[0].axhline(y=-nyquist, color='r', linestyle='--', label=f'-Nyquist ({nyquist:.1f} m/s)')
    axes[0].set_ylabel('速度 (m/s)')
    axes[0].set_title('真实速度场')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(aliased_velocities, 'orange', label='折叠后的速度', linewidth=2)
    axes[1].axhline(y=nyquist, color='r', linestyle='--')
    axes[1].axhline(y=-nyquist, color='r', linestyle='--')
    axes[1].set_ylabel('速度 (m/s)')
    axes[1].set_title('脉冲对法反演（折叠后）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dealiased_velocities, 'g-', label='解算后速度', linewidth=2)
    axes[2].plot(true_velocities, 'b--', label='真实速度', alpha=0.5)
    axes[2].set_xlabel('距离库')
    axes[2].set_ylabel('速度 (m/s)')
    axes[2].set_title('速度解算结果')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('velocity_dealiasing_test.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 velocity_dealiasing_test.png")

    rmse_aliased = np.sqrt(np.mean((aliased_velocities - true_velocities) ** 2))
    rmse_dealiased = np.sqrt(np.mean((dealiased_velocities - true_velocities) ** 2))
    print(f"折叠后 RMSE: {rmse_aliased:.2f} m/s")
    print(f"解算后 RMSE: {rmse_dealiased:.2f} m/s")

    return rmse_dealiased < rmse_aliased


def test_2d_radial_field():
    print("\n" + "=" * 60)
    print("测试4: 2D径向速度场（PPI扫描）")
    print("=" * 60)

    params = RadarParameters(wavelength=0.1, prf=800)
    num_rays = 36
    num_gates = 100

    theta = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    r = np.linspace(1, 100, num_gates)
    R, Theta = np.meshgrid(r, theta)

    wind_speed = 15
    wind_dir = np.pi / 4
    true_velocity_field = wind_speed * np.cos(Theta - wind_dir)

    processor = PulsePairProcessor(params)
    simulator = IQDataSimulator(params, num_gates=num_gates, num_pulses=64)

    retrieved_field = np.zeros_like(true_velocity_field)

    for ray in range(num_rays):
        iq_data = simulator.simulate_iq(true_velocity_field[ray, :], snr_db=25)
        retrieved_field[ray, :] = processor.compute_velocity(iq_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].pcolormesh(Theta, R, true_velocity_field, shading='auto', cmap='RdBu_r')
    axes[0].set_xlabel('方位角 (rad)')
    axes[0].set_ylabel('距离 (km)')
    axes[0].set_title('真实径向速度场')
    plt.colorbar(im0, ax=axes[0], label='速度 (m/s)')

    im1 = axes[1].pcolormesh(Theta, R, retrieved_field, shading='auto', cmap='RdBu_r')
    axes[1].set_xlabel('方位角 (rad)')
    axes[1].set_ylabel('距离 (km)')
    axes[1].set_title('反演径向速度场')
    plt.colorbar(im1, ax=axes[1], label='速度 (m/s)')

    plt.tight_layout()
    plt.savefig('radial_velocity_2d.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 radial_velocity_2d.png")

    rmse = np.sqrt(np.mean((retrieved_field - true_velocity_field) ** 2))
    print(f"2D速度场反演 RMSE: {rmse:.4f} m/s")

    return rmse < 1.0


def test_snr_impact():
    print("\n" + "=" * 60)
    print("测试5: SNR对反演精度的影响")
    print("=" * 60)

    params = RadarParameters(wavelength=0.1, prf=1000)
    num_gates = 100

    true_velocity = np.full(num_gates, 10.0)
    simulator = IQDataSimulator(params, num_gates=num_gates, num_pulses=64)
    processor = PulsePairProcessor(params)

    snr_values = [0, 5, 10, 15, 20, 25, 30]
    rmse_values = []

    for snr in snr_values:
        iq_data = simulator.simulate_iq(true_velocity, snr_db=snr)
        retrieved = processor.compute_velocity(iq_data)
        rmse = np.sqrt(np.mean((retrieved - true_velocity) ** 2))
        rmse_values.append(rmse)
        print(f"SNR = {snr:2d} dB, RMSE = {rmse:.4f} m/s")

    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, rmse_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (m/s)')
    plt.title('SNR对速度反演精度的影响')
    plt.grid(True, alpha=0.3)
    plt.savefig('snr_impact.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 snr_impact.png")

    return rmse_values[-1] < 0.5


def main():
    np.random.seed(42)

    tests = [
        ("基本速度反演", test_basic_velocity_retrieval),
        ("连续速度剖面", test_velocity_profile),
        ("速度折叠解算", test_velocity_dealiasing),
        ("2D径向速度场", test_2d_radial_field),
        ("SNR影响分析", test_snr_impact),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"测试 {name} 失败: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
