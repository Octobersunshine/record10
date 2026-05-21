import numpy as np
from phd_filter import (
    GM_PHD_Filter, ConstantVelocityModel, RadarMeasurementModel,
    GaussianComponent, simulate_extended_target_measurements, polar_to_cartesian
)


def run_extended_target_test():
    np.random.seed(42)
    dt = 1.0
    sigma_q = 0.8
    sigma_r = 1.5
    sigma_theta = 0.015
    motion_model = ConstantVelocityModel(dt, sigma_q)
    measurement_model = RadarMeasurementModel(sigma_r, sigma_theta)
    birth_components = [
        GaussianComponent(0.15, [50.0, 50.0, 0.0, 0.0], np.diag([100, 100, 10, 10])),
        GaussianComponent(0.15, [-50.0, 30.0, 0.0, 0.0], np.diag([100, 100, 10, 10])),
    ]
    phd_filter = GM_PHD_Filter(
        motion_model=motion_model,
        measurement_model=measurement_model,
        birth_intensity=birth_components,
        survival_prob=0.99,
        detection_prob=0.95,
        clutter_intensity=5e-7,
        clutter_rate=8,
        missed_detection_decay=0.75,
        confirmed_weight_threshold=0.4,
        max_consecutive_missed=12,
        is_extended_target=True,
        gamma_mean=8.0,
        gamma_variance=3.0
    )
    targets = [
        {'position': np.array([80.0, 20.0]), 'velocity': np.array([-4.0, 1.5])},
        {'position': np.array([-20.0, 60.0]), 'velocity': np.array([2.5, -2.0])},
    ]
    n_steps = 25
    ship_length = 45.0
    ship_width = 12.0
    avg_measurements_per_ship = 8
    print("=" * 80)
    print("扩展目标PHD滤波器演示 - 舰船跟踪")
    print("Extended Target PHD Filter - Ship Tracking")
    print("=" * 80)
    print(f"\n配置参数:")
    print(f"  - 舰船尺寸: {ship_length}m (长) x {ship_width}m (宽)")
    print(f"  - 每艘船期望量测数: {avg_measurements_per_ship}")
    print(f"  - 杂波率: {phd_filter.clutter_rate}")
    print(f"  - 扩展目标模式: 开启")
    print(f"  - Gamma分布参数: mean={phd_filter.gamma_mean}, var={phd_filter.gamma_variance}")
    print(f"\n初始目标: {len(targets)} 艘")
    for i, t in enumerate(targets):
        print(f"  船 {i+1}: 位置=({t['position'][0]:.1f}, {t['position'][1]:.1f}), "
              f"速度=({t['velocity'][0]:.1f}, {t['velocity'][1]:.1f})")
    print("\n开始跟踪...\n")
    for step in range(n_steps):
        true_positions, measurements = simulate_extended_target_measurements(
            targets, dt, sigma_r, sigma_theta,
            clutter_rate=8, clutter_range=(50, 200),
            ship_length=ship_length, ship_width=ship_width,
            avg_measurements_per_ship=avg_measurements_per_ship
        )
        estimated_targets = phd_filter.step(measurements)
        ship_measurements = len(measurements) - np.random.poisson(8)
        print(f"时间步 {step+1:2d}/{n_steps}:")
        print(f"  真实目标数: {len(true_positions)}, 量测总数: {len(measurements)}")
        print(f"  (舰船量测约: ~{ship_measurements}个, 杂波约: ~8个)")
        print(f"  估计目标数: {len(estimated_targets)}")
        for i, est in enumerate(estimated_targets):
            miss_info = f", 连续漏检={est['consecutive_missed']}" if est['consecutive_missed'] > 0 else ""
            pos_error = np.linalg.norm(est['position'] - true_positions[min(i, len(true_positions)-1)])
            print(f"    估计船 {i+1}: 位置=({est['position'][0]:6.1f}, {est['position'][1]:6.1f}), "
                  f"权重={est['weight']:.3f}, 位置误差={pos_error:.1f}m{miss_info}")
        if step == 8:
            print("\n  *** 新增舰船出现 ***")
            targets.append({'position': np.array([-60.0, -40.0]), 'velocity': np.array([3.5, 2.5])})
        if step == 18 and len(targets) > 1:
            print("\n  *** 一艘舰船离开监视区域 ***")
            targets.pop(1)
        print()
    print("=" * 80)
    print("扩展目标PHD滤波特性:")
    print("  1. 单个目标产生多个量测（泊松分布，Gamma分布建模）")
    print("  2. 利用所有关联量测的均值更新目标状态，提高精度")
    print("  3. 量测越多权重增益越大（1 + 0.3*(m-1)）")
    print("  4. 支持目标空间扩展建模（舰船轮廓）")
    print("  5. 可通过 is_extended_target 参数切换标准/扩展模式")
    print("=" * 80)
    return phd_filter


if __name__ == "__main__":
    run_extended_target_test()
