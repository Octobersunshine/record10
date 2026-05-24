import numpy as np
import matplotlib.pyplot as plt
from radial_velocity import (
    RadarParameters,
    IQDataSimulator,
    PulsePairProcessor,
    AdvancedVelocityDealiaser,
)


def test_gradient_unwrapping_1d():
    print("=" * 70)
    print("测试1: 1D梯度展开算法")
    print("=" * 70)

    params = RadarParameters(wavelength=0.1, prf=500)
    nyquist = params.nyquist_velocity
    print(f"Nyquist速度: {nyquist:.2f} m/s")

    num_gates = 200
    true_velocities = np.linspace(-40, 40, num_gates)

    simulator = IQDataSimulator(params, num_gates=num_gates, num_pulses=128)
    iq_data = simulator.simulate_iq(true_velocities, snr_db=30)

    processor = PulsePairProcessor(params)
    aliased_velocities = processor.compute_velocity(iq_data)

    dealiaser = AdvancedVelocityDealiaser(params)
    dealiased_simple = dealiaser.dealias_1d_simple(aliased_velocities)
    dealiased_gradient = dealiaser.dealias_1d_gradient(aliased_velocities)

    rmse_simple = np.sqrt(np.mean((dealiased_simple - true_velocities) ** 2))
    rmse_gradient = np.sqrt(np.mean((dealiased_gradient - true_velocities) ** 2))

    print(f"\n简单去折叠 RMSE: {rmse_simple:.4f} m/s")
    print(f"梯度展开 RMSE: {rmse_gradient:.4f} m/s")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(true_velocities, 'b-', label='真实速度', linewidth=2)
    axes[0].axhline(y=nyquist, color='r', linestyle='--', label=f'±Nyquist ({nyquist:.1f})')
    axes[0].axhline(y=-nyquist, color='r', linestyle='--')
    axes[0].set_ylabel('速度 (m/s)')
    axes[0].set_title('真实速度场')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(aliased_velocities, 'orange', label='折叠后速度', linewidth=2)
    axes[1].axhline(y=nyquist, color='r', linestyle='--')
    axes[1].axhline(y=-nyquist, color='r', linestyle='--')
    axes[1].set_ylabel('速度 (m/s)')
    axes[1].set_title('脉冲对法反演（折叠）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(true_velocities, 'b--', label='真实速度', alpha=0.5)
    axes[2].plot(dealiased_gradient, 'g-', label='梯度展开后', linewidth=2)
    axes[2].set_xlabel('距离库')
    axes[2].set_ylabel('速度 (m/s)')
    axes[2].set_title('梯度展开结果')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_unwrapping_1d.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到 gradient_unwrapping_1d.png")

    return rmse_gradient < 1.0


def test_gradient_unwrapping_with_mask():
    print("\n" + "=" * 70)
    print("测试2: 带质量掩码的梯度展开")
    print("=" * 70)

    params = RadarParameters(wavelength=0.1, prf=500)
    nyquist = params.nyquist_velocity
    print(f"Nyquist速度: {nyquist:.2f} m/s")

    num_gates = 200
    true_velocities = np.linspace(-35, 35, num_gates)

    simulator = IQDataSimulator(params, num_gates=num_gates, num_pulses=128)
    iq_data = simulator.simulate_iq(true_velocities, snr_db=20)

    processor = PulsePairProcessor(params)
    aliased_velocities = processor.compute_velocity(iq_data)
    power = processor.compute_power(iq_data)

    mask = np.ones(num_gates, dtype=bool)
    mask[50:70] = False
    mask[130:160] = False
    aliased_velocities[~mask] = np.nan

    dealiaser = AdvancedVelocityDealiaser(params)
    dealiased_gradient = dealiaser.dealias_1d_gradient(aliased_velocities, mask=mask)

    valid_mask = ~np.isnan(aliased_velocities)
    rmse_gradient = np.sqrt(np.mean(
        (dealiased_gradient[valid_mask] - true_velocities[valid_mask]) ** 2
    ))

    print(f"梯度展开（带掩码）RMSE: {rmse_gradient:.4f} m/s")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(true_velocities, 'b-', label='真实速度', linewidth=2)
    axes[0].plot(aliased_velocities, 'orange', label='折叠后（含无效值）', linewidth=2)
    axes[0].axhline(y=nyquist, color='r', linestyle='--')
    axes[0].axhline(y=-nyquist, color='r', linestyle='--')
    axes[0].set_ylabel('速度 (m/s)')
    axes[0].set_title('含数据缺失的速度场')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(true_velocities, 'b--', label='真实速度', alpha=0.5)
    axes[1].plot(dealiased_gradient, 'g-', label='梯度展开后', linewidth=2)
    axes[1].set_xlabel('距离库')
    axes[1].set_ylabel('速度 (m/s)')
    axes[1].set_title('梯度展开结果（处理数据缺失）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_unwrapping_with_mask.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 gradient_unwrapping_with_mask.png")

    return rmse_gradient < 2.0


def test_region_growing_2d():
    print("\n" + "=" * 70)
    print("测试3: 2D区域生长法去折叠")
    print("=" * 70)

    params = RadarParameters(wavelength=0.1, prf=400)
    nyquist = params.nyquist_velocity
    print(f"Nyquist速度: {nyquist:.2f} m/s")

    num_rays = 36
    num_gates = 100

    theta = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    r = np.linspace(1, 100, num_gates)
    R, Theta = np.meshgrid(r, theta)

    wind_speed = 30
    wind_dir = np.pi / 4
    true_velocity_field = wind_speed * np.cos(Theta - wind_dir)

    processor = PulsePairProcessor(params)
    simulator = IQDataSimulator(params, num_gates=num_gates, num_pulses=64)

    aliased_field = np.zeros_like(true_velocity_field)
    for ray in range(num_rays):
        iq_data = simulator.simulate_iq(true_velocity_field[ray, :], snr_db=30)
        aliased_field[ray, :] = processor.compute_velocity(iq_data)

    dealiaser = AdvancedVelocityDealiaser(params)
    dealiased_rg = dealiaser.dealias_2d_region_growing(aliased_field)
    dealiased_grad = dealiaser.dealias_2d_gradient(aliased_field)
    dealiased_hybrid = dealiaser.dealias_2d_hybrid(aliased_field)

    rmse_aliased = np.sqrt(np.mean((aliased_field - true_velocity_field) ** 2))
    rmse_rg = np.sqrt(np.mean((dealiased_rg - true_velocity_field) ** 2))
    rmse_grad = np.sqrt(np.mean((dealiased_grad - true_velocity_field) ** 2))
    rmse_hybrid = np.sqrt(np.mean((dealiased_hybrid - true_velocity_field) ** 2))

    print(f"\n折叠后 RMSE: {rmse_aliased:.4f} m/s")
    print(f"区域生长 RMSE: {rmse_rg:.4f} m/s")
    print(f"梯度展开 RMSE: {rmse_grad:.4f} m/s")
    print(f"混合方法 RMSE: {rmse_hybrid:.4f} m/s")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    vmin, vmax = -np.max(np.abs(true_velocity_field)), np.max(np.abs(true_velocity_field))

    im0 = axes[0, 0].pcolormesh(Theta, R, true_velocity_field, shading='auto',
                                cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('真实速度场')
    plt.colorbar(im0, ax=axes[0, 0], label='速度 (m/s)')

    im1 = axes[0, 1].pcolormesh(Theta, R, aliased_field, shading='auto',
                                cmap='RdBu_r', vmin=-nyquist, vmax=nyquist)
    axes[0, 1].set_title(f'折叠后速度场 (Nyquist={nyquist:.1f})')
    plt.colorbar(im1, ax=axes[0, 1], label='速度 (m/s)')

    im2 = axes[1, 0].pcolormesh(Theta, R, dealiased_rg, shading='auto',
                                cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('区域生长法去折叠')
    plt.colorbar(im2, ax=axes[1, 0], label='速度 (m/s)')

    im3 = axes[1, 1].pcolormesh(Theta, R, dealiased_hybrid, shading='auto',
                                cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('混合方法去折叠')
    plt.colorbar(im3, ax=axes[1, 1], label='速度 (m/s)')

    for ax in axes.flat:
        ax.set_xlabel('方位角 (rad)')
        ax.set_ylabel('距离 (km)')

    plt.tight_layout()
    plt.savefig('region_growing_2d.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 region_growing_2d.png")

    return rmse_rg < rmse_aliased and rmse_hybrid < 1.0


def test_multi_seed_region_growing():
    print("\n" + "=" * 70)
    print("测试4: 多种子点区域生长法")
    print("=" * 70)

    params = RadarParameters(wavelength=0.1, prf=350)
    nyquist = params.nyquist_velocity
    print(f"Nyquist速度: {nyquist:.2f} m/s")

    num_rays = 40
    num_gates = 120

    theta = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    r = np.linspace(1, 120, num_gates)
    R, Theta = np.meshgrid(r, theta)

    wind_speed = 40
    wind_dir = np.pi / 3
    true_velocity_field = wind_speed * np.cos(Theta - wind_dir)

    processor = PulsePairProcessor(params)
    simulator = IQDataSimulator(params, num_gates=num_gates, num_pulses=64)

    aliased_field = np.zeros_like(true_velocity_field)
    for ray in range(num_rays):
        iq_data = simulator.simulate_iq(true_velocity_field[ray, :], snr_db=25)
        aliased_field[ray, :] = processor.compute_velocity(iq_data)

    mask = np.ones_like(aliased_field, dtype=bool)
    mask[10:15, 30:50] = False
    mask[25:30, 70:90] = False
    aliased_field[~mask] = np.nan

    dealiaser = AdvancedVelocityDealiaser(params)
    dealiased_single = dealiaser.dealias_2d_region_growing(aliased_field, mask=mask)
    dealiased_multi = dealiaser.dealias_2d_multi_seed(aliased_field, mask=mask, num_seeds=4)

    valid_mask = ~np.isnan(aliased_field)
    rmse_single = np.sqrt(np.mean(
        (dealiased_single[valid_mask] - true_velocity_field[valid_mask]) ** 2
    ))
    rmse_multi = np.sqrt(np.mean(
        (dealiased_multi[valid_mask] - true_velocity_field[valid_mask]) ** 2
    ))

    print(f"\n单种子点 RMSE: {rmse_single:.4f} m/s")
    print(f"多种子点 RMSE: {rmse_multi:.4f} m/s")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    vmin, vmax = -np.max(np.abs(true_velocity_field)), np.max(np.abs(true_velocity_field))

    im0 = axes[0, 0].pcolormesh(Theta, R, true_velocity_field, shading='auto',
                                cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('真实速度场')
    plt.colorbar(im0, ax=axes[0, 0], label='速度 (m/s)')

    im1 = axes[0, 1].pcolormesh(Theta, R, aliased_field, shading='auto',
                                cmap='RdBu_r', vmin=-nyquist, vmax=nyquist)
    axes[0, 1].set_title(f'折叠后（含数据缺失）')
    plt.colorbar(im1, ax=axes[0, 1], label='速度 (m/s)')

    im2 = axes[1, 0].pcolormesh(Theta, R, dealiased_single, shading='auto',
                                cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f'单种子点 (RMSE={rmse_single:.2f})')
    plt.colorbar(im2, ax=axes[1, 0], label='速度 (m/s)')

    im3 = axes[1, 1].pcolormesh(Theta, R, dealiased_multi, shading='auto',
                                cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'多种子点 (RMSE={rmse_multi:.2f})')
    plt.colorbar(im3, ax=axes[1, 1], label='速度 (m/s)')

    for ax in axes.flat:
        ax.set_xlabel('方位角 (rad)')
        ax.set_ylabel('距离 (km)')

    plt.tight_layout()
    plt.savefig('multi_seed_dealiasing.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 multi_seed_dealiasing.png")

    return rmse_multi < rmse_single


def test_quality_control_integration():
    print("\n" + "=" * 70)
    print("测试5: 质量控制与去折叠集成")
    print("=" * 70)

    params = RadarParameters(wavelength=0.1, prf=500)
    nyquist = params.nyquist_velocity
    print(f"Nyquist速度: {nyquist:.2f} m/s")

    num_gates = 150
    true_velocities = np.linspace(-35, 35, num_gates)

    simulator = IQDataSimulator(params, num_gates=num_gates, num_pulses=64)

    snr_profile = np.linspace(-5, 30, num_gates)
    iq_data = np.zeros((num_gates, 64), dtype=np.complex128)

    for i in range(num_gates):
        iq_single = simulator.simulate_iq(
            np.array([true_velocities[i]]), snr_db=snr_profile[i]
        )
        iq_data[i, :] = iq_single[0, :]

    processor = PulsePairProcessor(params)
    aliased_velocities = processor.compute_velocity(iq_data)
    power = processor.compute_power(iq_data)

    noise_floor = np.min(power) * 0.5

    dealiaser = AdvancedVelocityDealiaser(params)

    snr_mask = dealiaser.quality_control.create_snr_mask(
        power, noise_floor, snr_threshold_db=5.0
    )
    texture_mask = dealiaser.quality_control.create_texture_mask(
        aliased_velocities, texture_threshold=8.0
    )
    combined_mask = snr_mask & texture_mask

    masked_velocities = aliased_velocities.copy()
    masked_velocities[~combined_mask] = np.nan

    dealiased_no_mask = dealiaser.dealias_1d_gradient(aliased_velocities)
    dealiased_with_mask = dealiaser.dealias_1d_gradient(masked_velocities, mask=combined_mask)

    valid_mask = snr_mask
    rmse_no_mask = np.sqrt(np.mean(
        (dealiased_no_mask[valid_mask] - true_velocities[valid_mask]) ** 2
    ))
    rmse_with_mask = np.sqrt(np.mean(
        (dealiased_with_mask[valid_mask] - true_velocities[valid_mask]) ** 2
    ))

    print(f"\n无质量控制 RMSE: {rmse_no_mask:.4f} m/s")
    print(f"有质量控制 RMSE: {rmse_with_mask:.4f} m/s")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(snr_profile, 'b-', label='SNR (dB)', linewidth=2)
    axes[0].axhline(y=5, color='r', linestyle='--', label='阈值 (5 dB)')
    axes[0].set_ylabel('SNR (dB)')
    axes[0].set_title('SNR剖面')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(true_velocities, 'b--', label='真实速度', alpha=0.5)
    axes[1].plot(aliased_velocities, 'orange', label='折叠后', linewidth=1.5)
    axes[1].plot(masked_velocities, 'go', label='有效数据点', markersize=3, alpha=0.7)
    axes[1].axhline(y=nyquist, color='r', linestyle='--')
    axes[1].axhline(y=-nyquist, color='r', linestyle='--')
    axes[1].set_ylabel('速度 (m/s)')
    axes[1].set_title('质量掩码筛选')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(true_velocities, 'b--', label='真实速度', alpha=0.5)
    axes[2].plot(dealiased_with_mask, 'g-', label='去折叠后（带QC）', linewidth=2)
    axes[2].set_xlabel('距离库')
    axes[2].set_ylabel('速度 (m/s)')
    axes[2].set_title('质量控制后的去折叠结果')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quality_control_dealiasing.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 quality_control_dealiasing.png")

    return rmse_with_mask < rmse_no_mask * 1.5


def main():
    np.random.seed(42)

    tests = [
        ("1D梯度展开", test_gradient_unwrapping_1d),
        ("带掩码梯度展开", test_gradient_unwrapping_with_mask),
        ("2D区域生长法", test_region_growing_2d),
        ("多种子点区域生长", test_multi_seed_region_growing),
        ("质量控制集成", test_quality_control_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"测试 {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
