import numpy as np
import matplotlib.pyplot as plt
from radial_velocity import RadarParameters
from dual_doppler_wind import (
    RadarPosition,
    WindField3D,
    CoordinateTransformer,
    DualDopplerSynthesizer,
    MultiRadarWindFusion,
    VerticalWindShear,
)


def generate_synthetic_wind_field(grid_x: np.ndarray, grid_y: np.ndarray,
                                   grid_z: np.ndarray, wind_type: str = 'supercell') -> WindField3D:
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

    if wind_type == 'uniform':
        u = np.full_like(X, 10.0)
        v = np.full_like(Y, 5.0)
        w = np.full_like(Z, 0.0)

    elif wind_type == 'shear':
        u = 5.0 + 8.0 * Z
        v = 3.0 + 4.0 * Z
        w = np.zeros_like(Z)

    elif wind_type == 'supercell':
        center_x, center_y = 0.0, 0.0
        R = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        theta = np.arctan2(Y - center_y, X - center_x)

        u_env = 5.0 + 6.0 * Z
        v_env = 3.0 + 5.0 * Z

        vortex_radius = 15.0
        vortex_strength = 20.0
        u_vortex = -vortex_strength * np.sin(theta) * np.exp(-R / vortex_radius)
        v_vortex = vortex_strength * np.cos(theta) * np.exp(-R / vortex_radius)

        w_strength = 15.0
        w = w_strength * np.exp(-R / (2 * vortex_radius)) * (1 - Z / 12.0)
        w[Z > 12.0] = 0

        u = u_env + u_vortex
        v = v_env + v_vortex

    else:
        raise ValueError(f"Unknown wind type: {wind_type}")

    return WindField3D(u=u, v=v, w=w, x=grid_x, y=grid_y, z=grid_z)


def test_coordinate_transform():
    print("=" * 70)
    print("测试1: 坐标变换")
    print("=" * 70)

    radar_pos = RadarPosition(x=0.0, y=0.0, z=0.0)
    transformer = CoordinateTransformer()

    azimuth = np.array([0.0, 90.0, 180.0, 270.0])
    elevation = np.array([0.0, 5.0, 10.0, 15.0])
    range_km = np.array([10.0, 20.0, 30.0, 40.0])

    x, y, z = transformer.radar_to_cartesian(azimuth, elevation, range_km, radar_pos)
    az_back, el_back, r_back = transformer.cartesian_to_radar(x, y, z, radar_pos)

    print(f"原始方位角: {azimuth}")
    print(f"反演方位角: {np.round(az_back, 2)}")
    print(f"原始仰角: {elevation}")
    print(f"反演仰角: {np.round(el_back, 2)}")
    print(f"原始距离: {range_km}")
    print(f"反演距离: {np.round(r_back, 2)}")

    error_az = np.max(np.abs(az_back - azimuth))
    error_el = np.max(np.abs(el_back - elevation))
    error_r = np.max(np.abs(r_back - range_km))

    print(f"\n最大误差: 方位角={error_az:.4f}°, 仰角={error_el:.4f}°, 距离={error_r:.4f}km")

    return error_az < 0.01 and error_el < 0.01 and error_r < 0.01


def test_dual_doppler_retrieval():
    print("\n" + "=" * 70)
    print("测试2: 双多普勒风场反演")
    print("=" * 70)

    params = RadarParameters(wavelength=0.1, prf=1000)
    radar1_pos = RadarPosition(x=-20.0, y=0.0, z=0.0)
    radar2_pos = RadarPosition(x=20.0, y=0.0, z=0.0)

    grid_x = np.linspace(-30, 30, 31)
    grid_y = np.linspace(-30, 30, 31)
    grid_z = np.linspace(0.5, 10, 11)

    true_wind = generate_synthetic_wind_field(grid_x, grid_y, grid_z, wind_type='shear')

    transformer = CoordinateTransformer()
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

    az1, el1, r1 = transformer.cartesian_to_radar(X, Y, Z, radar1_pos)
    az2, el2, r2 = transformer.cartesian_to_radar(X, Y, Z, radar2_pos)

    vr1 = transformer.compute_radial_velocity(true_wind.u, true_wind.v, true_wind.w, az1, el1)
    vr2 = transformer.compute_radial_velocity(true_wind.u, true_wind.v, true_wind.w, az2, el2)

    synthesizer = DualDopplerSynthesizer(radar1_pos, radar2_pos, params, params)
    retrieved_wind = synthesizer.synthesize_3d_wind(grid_x, grid_y, grid_z, vr1, vr2)

    valid_mask = ~np.isnan(retrieved_wind.u)
    rmse_u = np.sqrt(np.mean((retrieved_wind.u[valid_mask] - true_wind.u[valid_mask]) ** 2))
    rmse_v = np.sqrt(np.mean((retrieved_wind.v[valid_mask] - true_wind.v[valid_mask]) ** 2))
    rmse_w = np.sqrt(np.mean((retrieved_wind.w[valid_mask] - true_wind.w[valid_mask]) ** 2))

    print(f"基线距离: {synthesizer.compute_baseline():.1f} km")
    print(f"风场反演 RMSE - u: {rmse_u:.4f} m/s, v: {rmse_v:.4f} m/s, w: {rmse_w:.4f} m/s")

    z_level = 5
    z_idx = np.argmin(np.abs(grid_z - z_level))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    wind_level = 5

    im0 = axes[0, 0].pcolormesh(grid_x, grid_y, true_wind.u[:, :, z_idx].T, cmap='RdBu_r')
    axes[0, 0].set_title(f'真实 u 分量 (z={grid_z[z_idx]:.1f}km)')
    plt.colorbar(im0, ax=axes[0, 0], label='u (m/s)')

    im1 = axes[0, 1].pcolormesh(grid_x, grid_y, retrieved_wind.u[:, :, z_idx].T, cmap='RdBu_r')
    axes[0, 1].set_title(f'反演 u 分量')
    plt.colorbar(im1, ax=axes[0, 1], label='u (m/s)')

    error_u = retrieved_wind.u[:, :, z_idx] - true_wind.u[:, :, z_idx]
    im2 = axes[0, 2].pcolormesh(grid_x, grid_y, error_u.T, cmap='RdBu_r')
    axes[0, 2].set_title('u 误差')
    plt.colorbar(im2, ax=axes[0, 2], label='error (m/s)')

    im3 = axes[1, 0].pcolormesh(grid_x, grid_y, true_wind.v[:, :, z_idx].T, cmap='RdBu_r')
    axes[1, 0].set_title(f'真实 v 分量')
    plt.colorbar(im3, ax=axes[1, 0], label='v (m/s)')

    im4 = axes[1, 1].pcolormesh(grid_x, grid_y, retrieved_wind.v[:, :, z_idx].T, cmap='RdBu_r')
    axes[1, 1].set_title(f'反演 v 分量')
    plt.colorbar(im4, ax=axes[1, 1], label='v (m/s)')

    error_v = retrieved_wind.v[:, :, z_idx] - true_wind.v[:, :, z_idx]
    im5 = axes[1, 2].pcolormesh(grid_x, grid_y, error_v.T, cmap='RdBu_r')
    axes[1, 2].set_title('v 误差')
    plt.colorbar(im5, ax=axes[1, 2], label='error (m/s)')

    for ax in axes.flat:
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')

    plt.tight_layout()
    plt.savefig('dual_doppler_retrieval.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 dual_doppler_retrieval.png")

    return rmse_u < 1.0 and rmse_v < 1.0


def test_kalman_fusion():
    print("\n" + "=" * 70)
    print("测试3: 卡尔曼滤波多雷达融合")
    print("=" * 70)

    params = RadarParameters(wavelength=0.1, prf=1000)
    radar1_pos = RadarPosition(x=-20.0, y=0.0, z=0.0)
    radar2_pos = RadarPosition(x=20.0, y=0.0, z=0.0)
    radar3_pos = RadarPosition(x=0.0, y=25.0, z=0.0)

    grid_x = np.linspace(-25, 25, 26)
    grid_y = np.linspace(-25, 25, 26)
    grid_z = np.linspace(0.5, 8, 9)

    true_wind = generate_synthetic_wind_field(grid_x, grid_y, grid_z, wind_type='shear')

    transformer = CoordinateTransformer()
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

    radar_positions = [radar1_pos, radar2_pos, radar3_pos]
    vr_fields = []
    for pos in radar_positions:
        az, el, r = transformer.cartesian_to_radar(X, Y, Z, pos)
        vr = transformer.compute_radial_velocity(true_wind.u, true_wind.v, true_wind.w, az, el)
        vr_noise = vr + np.random.normal(0, 1.0, vr.shape)
        vr_fields.append(vr_noise)

    fusion = MultiRadarWindFusion(radar_positions, [params, params, params])

    wind_avg = fusion.fuse_radial_velocities(grid_x, grid_y, grid_z, vr_fields, method='simple_average')
    wind_kalman = fusion.fuse_radial_velocities(grid_x, grid_y, grid_z, vr_fields, method='kalman',
                                                 process_noise=0.05, measurement_noise=2.0)

    valid_mask_avg = ~np.isnan(wind_avg.u)
    valid_mask_kf = ~np.isnan(wind_kalman.u)

    rmse_avg_u = np.sqrt(np.mean((wind_avg.u[valid_mask_avg] - true_wind.u[valid_mask_avg]) ** 2))
    rmse_avg_v = np.sqrt(np.mean((wind_avg.v[valid_mask_avg] - true_wind.v[valid_mask_avg]) ** 2))
    rmse_kf_u = np.sqrt(np.mean((wind_kalman.u[valid_mask_kf] - true_wind.u[valid_mask_kf]) ** 2))
    rmse_kf_v = np.sqrt(np.mean((wind_kalman.v[valid_mask_kf] - true_wind.v[valid_mask_kf]) ** 2))

    print(f"简单平均融合 - RMSE: u={rmse_avg_u:.4f} m/s, v={rmse_avg_v:.4f} m/s")
    print(f"卡尔曼滤波融合 - RMSE: u={rmse_kf_u:.4f} m/s, v={rmse_kf_v:.4f} m/s")

    z_idx = 4
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].pcolormesh(grid_x, grid_y, true_wind.u[:, :, z_idx].T, cmap='RdBu_r')
    axes[0, 0].set_title('真实 u 分量')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(grid_x, grid_y, wind_avg.u[:, :, z_idx].T, cmap='RdBu_r')
    axes[0, 1].set_title('简单平均融合')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].pcolormesh(grid_x, grid_y, wind_kalman.u[:, :, z_idx].T, cmap='RdBu_r')
    axes[0, 2].set_title('卡尔曼滤波融合')
    plt.colorbar(im2, ax=axes[0, 2])

    error_avg = wind_avg.u[:, :, z_idx] - true_wind.u[:, :, z_idx]
    error_kf = wind_kalman.u[:, :, z_idx] - true_wind.u[:, :, z_idx]

    vmax = np.max(np.abs([error_avg, error_kf]))
    im3 = axes[1, 0].pcolormesh(grid_x, grid_y, error_avg.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title(f'简单平均误差 (RMSE={rmse_avg_u:.2f})')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(grid_x, grid_y, error_kf.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title(f'卡尔曼误差 (RMSE={rmse_kf_u:.2f})')
    plt.colorbar(im4, ax=axes[1, 1])

    axes[1, 2].bar(['Simple Avg', 'Kalman'], [rmse_avg_u, rmse_kf_u], color=['orange', 'green'])
    axes[1, 2].set_ylabel('RMSE (m/s)')
    axes[1, 2].set_title('u分量反演误差对比')

    plt.tight_layout()
    plt.savefig('kalman_fusion_comparison.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 kalman_fusion_comparison.png")

    return rmse_kf_u < rmse_avg_u * 1.2


def test_vertical_wind_shear():
    print("\n" + "=" * 70)
    print("测试4: 垂直风切变计算")
    print("=" * 70)

    params = RadarParameters(wavelength=0.1, prf=1000)
    radar1_pos = RadarPosition(x=-20.0, y=0.0, z=0.0)
    radar2_pos = RadarPosition(x=20.0, y=0.0, z=0.0)

    grid_x = np.linspace(-30, 30, 31)
    grid_y = np.linspace(-30, 30, 31)
    grid_z = np.linspace(0.5, 12, 13)

    true_wind = generate_synthetic_wind_field(grid_x, grid_y, grid_z, wind_type='supercell')

    transformer = CoordinateTransformer()
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

    az1, el1, r1 = transformer.cartesian_to_radar(X, Y, Z, radar1_pos)
    az2, el2, r2 = transformer.cartesian_to_radar(X, Y, Z, radar2_pos)

    vr1 = transformer.compute_radial_velocity(true_wind.u, true_wind.v, true_wind.w, az1, el1)
    vr2 = transformer.compute_radial_velocity(true_wind.u, true_wind.v, true_wind.w, az2, el2)

    synthesizer = DualDopplerSynthesizer(radar1_pos, radar2_pos, params, params)
    retrieved_wind = synthesizer.synthesize_3d_wind(grid_x, grid_y, grid_z, vr1, vr2)

    shear_calculator = VerticalWindShear(retrieved_wind)
    shear_mag = shear_calculator.compute_shear_magnitude()
    _, _, bulk_shear = shear_calculator.compute_bulk_shear(0.5, 6.0)
    helicity = shear_calculator.compute_helicity(0.0, 3.0)
    hazards = shear_calculator.detect_convective_hazard(shear_threshold=8.0, helicity_threshold=100.0)

    print(f"垂直切变范围: {np.nanmin(shear_mag):.2f} - {np.nanmax(shear_mag):.2f} s^-1")
    print(f"0-6km体切变范围: {np.nanmin(bulk_shear):.2f} - {np.nanmax(bulk_shear):.2f} s^-1")
    print(f"风暴相对螺旋度范围: {np.nanmin(helicity):.2f} - {np.nanmax(helicity):.2f} m^2/s^2")
    print(f"危险区域格点数: {np.sum(hazards['hazard_mask'])} / {hazards['hazard_mask'].size}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    z_idx_mid = 6
    im0 = axes[0, 0].pcolormesh(grid_x, grid_y, shear_mag[:, :, z_idx_mid].T, cmap='hot')
    axes[0, 0].set_title(f'垂直风切变 (z={grid_z[z_idx_mid]:.1f}km)')
    plt.colorbar(im0, ax=axes[0, 0], label='shear (s^-1)')

    im1 = axes[0, 1].pcolormesh(grid_x, grid_y, bulk_shear.T, cmap='hot')
    axes[0, 1].set_title('0-6km 体切变')
    plt.colorbar(im1, ax=axes[0, 1], label='shear (s^-1)')

    im2 = axes[0, 2].pcolormesh(grid_x, grid_y, helicity.T, cmap='hot')
    axes[0, 2].set_title('0-3km 风暴相对螺旋度')
    plt.colorbar(im2, ax=axes[0, 2], label='SRH (m^2/s^2)')

    w_mid = retrieved_wind.w[:, :, z_idx_mid]
    im3 = axes[1, 0].pcolormesh(grid_x, grid_y, w_mid.T, cmap='RdBu_r')
    axes[1, 0].set_title(f'垂直速度 w (z={grid_z[z_idx_mid]:.1f}km)')
    plt.colorbar(im3, ax=axes[1, 0], label='w (m/s)')

    im4 = axes[1, 1].pcolormesh(grid_x, grid_y, hazards['max_shear'].T, cmap='hot')
    axes[1, 1].set_title('最大垂直切变')
    plt.colorbar(im4, ax=axes[1, 1], label='max shear (s^-1)')

    im5 = axes[1, 2].pcolormesh(grid_x, grid_y, hazards['hazard_mask'].T, cmap='Reds')
    axes[1, 2].set_title('强对流危险区域')
    plt.colorbar(im5, ax=axes[1, 2], label='hazard')

    for ax in axes.flat:
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')

    plt.tight_layout()
    plt.savefig('vertical_wind_shear.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 vertical_wind_shear.png")

    return True


def test_wind_profile_analysis():
    print("\n" + "=" * 70)
    print("测试5: 风廓线分析")
    print("=" * 70)

    grid_x = np.array([0.0])
    grid_y = np.array([0.0])
    grid_z = np.linspace(0.1, 15, 31)

    true_wind = generate_synthetic_wind_field(grid_x, grid_y, grid_z, wind_type='supercell')

    shear_calculator = VerticalWindShear(true_wind)
    shear_mag = shear_calculator.compute_shear_magnitude()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].plot(true_wind.u[0, 0, :], grid_z, 'b-', label='u', linewidth=2)
    axes[0].plot(true_wind.v[0, 0, :], grid_z, 'r-', label='v', linewidth=2)
    axes[0].set_xlabel('风速 (m/s)')
    axes[0].set_ylabel('高度 (km)')
    axes[0].set_title('风分量廓线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(true_wind.speed[0, 0, :], grid_z, 'g-', linewidth=2)
    axes[1].set_xlabel('风速 (m/s)')
    axes[1].set_ylabel('高度 (km)')
    axes[1].set_title('风速大小廓线')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(shear_mag[0, 0, :], grid_z, 'm-', linewidth=2)
    axes[2].set_xlabel('风切变 (s^-1)')
    axes[2].set_ylabel('高度 (km)')
    axes[2].set_title('垂直风切变廓线')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wind_profile_analysis.png', dpi=150, bbox_inches='tight')
    print("结果已保存到 wind_profile_analysis.png")

    return True


def main():
    np.random.seed(42)

    tests = [
        ("坐标变换", test_coordinate_transform),
        ("双多普勒风场反演", test_dual_doppler_retrieval),
        ("卡尔曼滤波融合", test_kalman_fusion),
        ("垂直风切变计算", test_vertical_wind_shear),
        ("风廓线分析", test_wind_profile_analysis),
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
