import numpy as np
from phd_filter import GM_PHD_Filter, ConstantVelocityModel, RadarMeasurementModel, GaussianComponent


def simulate_with_occlusion(target, dt, sigma_r, sigma_theta, occlusion_steps):
    true_pos = target['position'] + target['velocity'] * dt
    target['position'] = true_pos.copy()
    if dt in occlusion_steps:
        measurements = []
        is_occluded = True
    else:
        r = np.sqrt(true_pos[0]**2 + true_pos[1]**2)
        theta = np.arctan2(true_pos[1], true_pos[0])
        z_r = r + np.random.normal(0, sigma_r)
        z_theta = theta + np.random.normal(0, sigma_theta)
        measurements = [np.array([z_r, z_theta])]
        is_occluded = False
    return true_pos, measurements, is_occluded


def run_occlusion_test():
    np.random.seed(42)
    dt = 1.0
    sigma_q = 0.5
    sigma_r = 1.0
    sigma_theta = 0.02
    motion_model = ConstantVelocityModel(dt, sigma_q)
    measurement_model = RadarMeasurementModel(sigma_r, sigma_theta)
    birth_components = [
        GaussianComponent(0.1, [100, 0, -5, 2], np.diag([50, 50, 5, 5])),
    ]
    phd_filter = GM_PHD_Filter(
        motion_model=motion_model,
        measurement_model=measurement_model,
        birth_intensity=birth_components,
        survival_prob=0.99,
        detection_prob=0.98,
        clutter_intensity=1e-6,
        clutter_rate=2,
        missed_detection_decay=0.7,
        confirmed_weight_threshold=0.3,
        max_consecutive_missed=15
    )
    target = {'position': np.array([100.0, 0.0]), 'velocity': np.array([-5.0, 2.0])}
    n_steps = 40
    occlusion_start = 15
    occlusion_end = 25
    occlusion_steps = list(range(occlusion_start, occlusion_end))
    print("=" * 70)
    print("PHD滤波器遮挡测试 - 验证漏检时的目标存活能力")
    print("=" * 70)
    print(f"\n测试配置:")
    print(f"  - 目标初始位置: {target['position']}")
    print(f"  - 目标速度: {target['velocity']}")
    print(f"  - 遮挡开始: 时间步 {occlusion_start}")
    print(f"  - 遮挡结束: 时间步 {occlusion_end}")
    print(f"  - 遮挡持续: {occlusion_end - occlusion_start} 步")
    print(f"  - 漏检衰减因子: {phd_filter.missed_detection_decay}")
    print(f"  - 确认目标阈值: {phd_filter.confirmed_weight_threshold}")
    print(f"  - 最大连续漏检: {phd_filter.max_consecutive_missed} 步\n")
    print("-" * 70)
    print(f"{'时间步':<8}{'真实位置':<20}{'状态':<12}{'权重':<10}{'连续漏检':<10}")
    print("-" * 70)
    target_detected_after_occlusion = False
    max_weight_during_occlusion = 0
    for step in range(1, n_steps + 1):
        true_pos, measurements, is_occluded = simulate_with_occlusion(
            target, step, sigma_r, sigma_theta, occlusion_steps
        )
        estimated_targets = phd_filter.step(measurements)
        status = "遮挡中" if is_occluded else "正常"
        weight_str = "-"
        missed_str = "-"
        if estimated_targets:
            max_weight = max(t['weight'] for t in estimated_targets)
            weight_str = f"{max_weight:.4f}"
            if is_occluded:
                max_weight_during_occlusion = max(max_weight_during_occlusion, max_weight)
            if step > occlusion_end and max_weight > 0.5:
                target_detected_after_occlusion = True
        if phd_filter.components:
            max_missed = max(c.consecutive_missed for c in phd_filter.components)
            missed_str = str(max_missed)
        pos_str = f"[{true_pos[0]:.1f}, {true_pos[1]:.1f}]"
        print(f"{step:<8}{pos_str:<20}{status:<12}{weight_str:<10}{missed_str:<10}")
    print("-" * 70)
    print("\n测试结果:")
    print(f"  - 遮挡期间最大权重: {max_weight_during_occlusion:.4f}")
    print(f"  - 遮挡后目标重新检测: {'成功 ✓' if target_detected_after_occlusion else '失败 ✗'}")
    print("\n关键改进说明:")
    print("  1. 已确认目标漏检时使用 0.7 衰减因子（原 0.02）")
    print("  2. 已确认目标剪枝阈值降低 10 倍")
    print("  3. 引入连续漏检计数，超过15步才删除")
    print("  4. 检测到量测时重置连续漏检计数")
    print("=" * 70)
    return target_detected_after_occlusion


if __name__ == "__main__":
    run_occlusion_test()
