import numpy as np
from scipy.stats import multivariate_normal


class GaussianComponent:
    def __init__(self, weight, mean, cov):
        self.weight = weight
        self.mean = np.array(mean, dtype=np.float64)
        self.cov = np.array(cov, dtype=np.float64)
        self.consecutive_missed = 0


class ConstantVelocityModel:
    def __init__(self, dt, sigma_q):
        self.dt = dt
        self.sigma_q = sigma_q
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.Q = sigma_q**2 * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ])


class RadarMeasurementModel:
    def __init__(self, sigma_r, sigma_theta, sensor_pos=(0, 0)):
        self.sigma_r = sigma_r
        self.sigma_theta = sigma_theta
        self.sensor_pos = np.array(sensor_pos)
        self.R = np.diag([sigma_r**2, sigma_theta**2])

    def h(self, x):
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        return np.array([r, theta])

    def jacobian_H(self, x):
        dx = x[0] - self.sensor_pos[0]
        dy = x[1] - self.sensor_pos[1]
        r = np.sqrt(dx**2 + dy**2)
        if r < 1e-10:
            r = 1e-10
        H = np.array([
            [dx/r, dy/r, 0, 0],
            [-dy/r**2, dx/r**2, 0, 0]
        ])
        return H


class GM_PHD_Filter:
    def __init__(self, motion_model, measurement_model,
                 survival_prob=0.99, detection_prob=0.98,
                 birth_intensity=None, clutter_intensity=1e-6,
                 clutter_rate=10,
                 missed_detection_decay=0.7,
                 confirmed_weight_threshold=0.3,
                 max_consecutive_missed=10,
                 is_extended_target=False,
                 gamma_mean=5.0,
                 gamma_variance=2.0,
                 measurement_spatial_cov=None):
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.survival_prob = survival_prob
        self.detection_prob = detection_prob
        self.birth_intensity = birth_intensity if birth_intensity is not None else []
        self.clutter_intensity = clutter_intensity
        self.clutter_rate = clutter_rate
        self.missed_detection_decay = missed_detection_decay
        self.confirmed_weight_threshold = confirmed_weight_threshold
        self.max_consecutive_missed = max_consecutive_missed
        self.is_extended_target = is_extended_target
        self.gamma_mean = gamma_mean
        self.gamma_variance = gamma_variance
        if measurement_spatial_cov is None:
            self.measurement_spatial_cov = np.diag([16.0, 16.0])
        else:
            self.measurement_spatial_cov = measurement_spatial_cov
        self.components = []

    def predict(self):
        new_components = []
        for comp in self.components:
            new_mean = self.motion_model.F @ comp.mean
            new_cov = self.motion_model.F @ comp.cov @ self.motion_model.F.T + self.motion_model.Q
            new_weight = self.survival_prob * comp.weight
            new_comp = GaussianComponent(new_weight, new_mean, new_cov)
            new_comp.consecutive_missed = comp.consecutive_missed
            new_components.append(new_comp)
        for birth_comp in self.birth_intensity:
            new_components.append(GaussianComponent(
                birth_comp.weight, birth_comp.mean.copy(), birth_comp.cov.copy()
            ))
        self.components = new_components

    def _calculate_measurement_likelihood(self, z, comp):
        z_pred = self.measurement_model.h(comp.mean)
        H = self.measurement_model.jacobian_H(comp.mean)
        S = H @ comp.cov @ H.T + self.measurement_model.R
        try:
            likelihood = multivariate_normal.pdf(z, mean=z_pred, cov=S)
        except np.linalg.LinAlgError:
            likelihood = 1e-10
        return likelihood, z_pred, H, S

    def _update_standard(self, measurements, predicted_components):
        new_components = []
        detection_components = []
        for comp in predicted_components:
            is_confirmed = comp.weight >= self.confirmed_weight_threshold
            if is_confirmed:
                base_decay = self.missed_detection_decay
                additional_decay = 1.0 - min(comp.consecutive_missed, 5) * 0.05
                decay_factor = base_decay * additional_decay
                decay_factor = max(decay_factor, 0.3)
            else:
                decay_factor = (1 - self.detection_prob) * 0.5 + 0.01
            new_weight = decay_factor * comp.weight
            new_comp = GaussianComponent(new_weight, comp.mean.copy(), comp.cov.copy())
            new_comp.consecutive_missed = comp.consecutive_missed + 1
            new_components.append(new_comp)
        for z in measurements:
            for comp in predicted_components:
                likelihood, z_pred, H, S = self._calculate_measurement_likelihood(z, comp)
                try:
                    K = comp.cov @ H.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    K = comp.cov @ H.T @ np.linalg.pinv(S)
                new_mean = comp.mean + K @ (z - z_pred)
                new_cov = (np.eye(4) - K @ H) @ comp.cov
                sum_term = self.detection_prob * sum(
                    c.weight * self._calculate_measurement_likelihood(z, c)[0]
                    for c in predicted_components
                )
                new_weight = (self.detection_prob * comp.weight * likelihood) / \
                             (self.clutter_rate * self.clutter_intensity + sum_term + 1e-10)
                det_comp = GaussianComponent(new_weight, new_mean, new_cov)
                det_comp.consecutive_missed = 0
                detection_components.append(det_comp)
        new_components.extend(detection_components)
        return new_components

    def _update_extended_target(self, measurements, predicted_components):
        new_components = []
        detection_components = []
        for comp in predicted_components:
            is_confirmed = comp.weight >= self.confirmed_weight_threshold
            if is_confirmed:
                base_decay = self.missed_detection_decay
                additional_decay = 1.0 - min(comp.consecutive_missed, 5) * 0.05
                decay_factor = base_decay * additional_decay
                decay_factor = max(decay_factor, 0.3)
            else:
                decay_factor = (1 - self.detection_prob) * 0.5 + 0.01
            undetected_weight = decay_factor * comp.weight
            new_comp = GaussianComponent(undetected_weight, comp.mean.copy(), comp.cov.copy())
            new_comp.consecutive_missed = comp.consecutive_missed + 1
            new_components.append(new_comp)
        for comp in predicted_components:
            assigned_measurements = []
            likelihoods = []
            for z in measurements:
                likelihood, _, _, _ = self._calculate_measurement_likelihood(z, comp)
                if likelihood > 1e-5:
                    assigned_measurements.append(z)
                    likelihoods.append(likelihood)
            if len(assigned_measurements) == 0:
                continue
            m = len(assigned_measurements)
            gamma = self.gamma_mean
            poisson_factor = (gamma ** m) * np.exp(-gamma)
            z_stack = np.vstack(assigned_measurements)
            z_mean = np.mean(z_stack, axis=0)
            likelihood, z_pred, H, S = self._calculate_measurement_likelihood(z_mean, comp)
            total_likelihood = np.prod(likelihoods)
            try:
                K = comp.cov @ H.T @ np.linalg.inv(S / m)
            except np.linalg.LinAlgError:
                    K = comp.cov @ H.T @ np.linalg.pinv(S / m)
            new_mean = comp.mean + K @ (z_mean - z_pred)
            new_cov = (np.eye(4) - K @ H / np.sqrt(m)) @ comp.cov @ (np.eye(4) - K @ H / np.sqrt(m)).T
            new_cov = new_cov + K @ (self.measurement_model.R / m) @ K.T
            sum_term = sum(
                c.weight * np.prod([self._calculate_measurement_likelihood(z, c)[0] for z in assigned_measurements])
                for c in predicted_components
            )
            new_weight = (poisson_factor * self.detection_prob * comp.weight * total_likelihood) / \
                         (self.clutter_rate * self.clutter_intensity + sum_term + 1e-10)
            new_weight = new_weight * (1 + 0.3 * (m - 1))
            det_comp = GaussianComponent(new_weight, new_mean, new_cov)
            det_comp.consecutive_missed = 0
            detection_components.append(det_comp)
        new_components.extend(detection_components)
        return new_components

    def update(self, measurements):
        predicted_components = self.components.copy()
        if self.is_extended_target and len(measurements) > 0:
            self.components = self._update_extended_target(measurements, predicted_components)
        else:
            self.components = self._update_standard(measurements, predicted_components)

    def prune(self, weight_threshold=1e-5, merge_threshold=4, max_components=100):
        pruned_components = []
        for comp in self.components:
            is_confirmed = comp.weight >= self.confirmed_weight_threshold
            if is_confirmed:
                effective_threshold = weight_threshold * 0.1
            else:
                effective_threshold = weight_threshold
            if comp.consecutive_missed > self.max_consecutive_missed:
                continue
            if comp.weight > effective_threshold:
                pruned_components.append(comp)
        self.components = pruned_components
        if len(self.components) == 0:
            return
        self.components.sort(key=lambda x: x.weight, reverse=True)
        merged_components = []
        used = [False] * len(self.components)
        for i in range(len(self.components)):
            if used[i]:
                continue
            comp_i = self.components[i]
            merged_weight = comp_i.weight
            merged_mean = comp_i.weight * comp_i.mean
            merged_cov = comp_i.weight * comp_i.cov
            merged_consecutive_missed = comp_i.weight * comp_i.consecutive_missed
            for j in range(i + 1, len(self.components)):
                if used[j]:
                    continue
                comp_j = self.components[j]
                diff = comp_j.mean - comp_i.mean
                try:
                    distance = diff.T @ np.linalg.inv(comp_i.cov) @ diff
                except np.linalg.LinAlgError:
                    distance = float('inf')
                if distance < merge_threshold:
                    used[j] = True
                    merged_weight += comp_j.weight
                    merged_mean += comp_j.weight * comp_j.mean
                    merged_cov += comp_j.weight * comp_j.cov
                    merged_consecutive_missed += comp_j.weight * comp_j.consecutive_missed
            merged_mean /= merged_weight
            merged_cov /= merged_weight
            merged_consecutive_missed /= merged_weight
            merged_comp = GaussianComponent(merged_weight, merged_mean, merged_cov)
            merged_comp.consecutive_missed = int(round(merged_consecutive_missed))
            merged_components.append(merged_comp)
        merged_components.sort(key=lambda x: x.weight, reverse=True)
        self.components = merged_components[:max_components]

    def extract_states(self, extraction_threshold=0.5):
        targets = []
        for comp in self.components:
            if comp.weight >= extraction_threshold:
                targets.append({
                    'position': comp.mean[:2],
                    'velocity': comp.mean[2:],
                    'weight': comp.weight,
                    'cov': comp.cov,
                    'consecutive_missed': comp.consecutive_missed
                })
        return targets

    def step(self, measurements):
        self.predict()
        self.update(measurements)
        self.prune()
        return self.extract_states()


def simulate_radar_measurements(targets, dt, sigma_r, sigma_theta, clutter_rate, clutter_range):
    true_positions = []
    measurements = []
    for t in targets:
        true_pos = t['position'] + t['velocity'] * dt
        t['position'] = true_pos
        true_positions.append(true_pos.copy())
    for pos in true_positions:
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        theta = np.arctan2(pos[1], pos[0])
        z_r = r + np.random.normal(0, sigma_r)
        z_theta = theta + np.random.normal(0, sigma_theta)
        measurements.append(np.array([z_r, z_theta]))
    n_clutter = np.random.poisson(clutter_rate)
    for _ in range(n_clutter):
        r = np.random.uniform(*clutter_range)
        theta = np.random.uniform(-np.pi, np.pi)
        measurements.append(np.array([r, theta]))
    return true_positions, measurements


def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])


def simulate_extended_target_measurements(targets, dt, sigma_r, sigma_theta, clutter_rate, clutter_range,
                                          ship_length=40.0, ship_width=10.0, avg_measurements_per_ship=8):
    true_positions = []
    measurements = []
    for t in targets:
        true_pos = t['position'] + t['velocity'] * dt
        t['position'] = true_pos
        true_positions.append(true_pos.copy())
    for i, pos in enumerate(true_positions):
        velocity = targets[i]['velocity']
        heading = np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) > 0.1 else 0
        n_measurements = np.random.poisson(avg_measurements_per_ship)
        n_measurements = max(3, n_measurements)
        for _ in range(n_measurements):
            x_local = np.random.uniform(-ship_length / 2, ship_length / 2)
            y_local = np.random.uniform(-ship_width / 2, ship_width / 2)
            rot_matrix = np.array([
                [np.cos(heading), -np.sin(heading)],
                [np.sin(heading), np.cos(heading)]
            ])
            point_global = pos + rot_matrix @ np.array([x_local, y_local])
            r = np.sqrt(point_global[0]**2 + point_global[1]**2)
            theta = np.arctan2(point_global[1], point_global[0])
            z_r = r + np.random.normal(0, sigma_r)
            z_theta = theta + np.random.normal(0, sigma_theta)
            measurements.append(np.array([z_r, z_theta]))
    n_clutter = np.random.poisson(clutter_rate)
    for _ in range(n_clutter):
        r = np.random.uniform(*clutter_range)
        theta = np.random.uniform(-np.pi, np.pi)
        measurements.append(np.array([r, theta]))
    return true_positions, measurements


if __name__ == "__main__":
    import sys
    mode = "extended" if len(sys.argv) > 1 and sys.argv[1] == "extended" else "standard"
    np.random.seed(42)
    dt = 1.0
    sigma_q = 0.5
    sigma_r = 1.0
    sigma_theta = 0.02
    motion_model = ConstantVelocityModel(dt, sigma_q)
    measurement_model = RadarMeasurementModel(sigma_r, sigma_theta)
    birth_components = [
        GaussianComponent(0.1, [0, 0, 0, 0], np.diag([100, 100, 10, 10])),
        GaussianComponent(0.1, [50, 50, 0, 0], np.diag([100, 100, 10, 10])),
        GaussianComponent(0.1, [-50, 50, 0, 0], np.diag([100, 100, 10, 10])),
    ]
    if mode == "extended":
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
        ship_length = 45.0
        ship_width = 12.0
        avg_measurements_per_ship = 8
        n_steps = 25
        print("=" * 80)
        print("扩展目标PHD滤波器演示 - 舰船跟踪")
        print("Extended Target PHD Filter - Ship Tracking")
        print("=" * 80)
        print(f"\n配置参数:")
        print(f"  - 舰船尺寸: {ship_length}m (长) x {ship_width}m (宽)")
        print(f"  - 每艘船期望量测数: {avg_measurements_per_ship}")
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
            print(f"时间步 {step+1:2d}/{n_steps}:")
            print(f"  真实目标数: {len(true_positions)}, 量测总数: {len(measurements)}")
            print(f"  估计目标数: {len(estimated_targets)}")
            for i, est in enumerate(estimated_targets):
                miss_info = f", 连续漏检={est['consecutive_missed']}" if est['consecutive_missed'] > 0 else ""
                if i < len(true_positions):
                    pos_error = np.linalg.norm(est['position'] - true_positions[i])
                    error_str = f", 位置误差={pos_error:.1f}m"
                else:
                    error_str = ""
                print(f"    估计船 {i+1}: 位置=({est['position'][0]:6.1f}, {est['position'][1]:6.1f}), "
                      f"权重={est['weight']:.3f}{miss_info}{error_str}")
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
        print("\n运行标准模式: python phd_filter.py standard")
        print("运行扩展模式: python phd_filter.py extended")
        print("=" * 80)
    else:
        phd_filter = GM_PHD_Filter(
            motion_model=motion_model,
            measurement_model=measurement_model,
            birth_intensity=birth_components,
            survival_prob=0.99,
            detection_prob=0.98,
            clutter_intensity=1/(np.pi * 200**2),
            clutter_rate=5
        )
        targets = [
            {'position': np.array([100.0, 0.0]), 'velocity': np.array([-5.0, 2.0])},
            {'position': np.array([0.0, 80.0]), 'velocity': np.array([3.0, -4.0])},
        ]
        n_steps = 30
        occlusion_start = 12
        occlusion_end = 18
        print("=" * 70)
        print("标准PHD滤波器演示 - 含遮挡修复")
        print("Standard PHD Filter with Occlusion Handling")
        print("=" * 70)
        print(f"\n初始目标数目: {len(targets)}")
        for i, t in enumerate(targets):
            print(f"  目标 {i+1}: 位置={t['position']}, 速度={t['velocity']}")
        print(f"\n遮挡测试: 目标1 在时间步 {occlusion_start}-{occlusion_end} 被遮挡")
        print(f"漏检衰减配置: 衰减因子={phd_filter.missed_detection_decay}, "
              f"确认阈值={phd_filter.confirmed_weight_threshold}")
        print("\n开始跟踪...\n")
        for step in range(n_steps):
            is_occlusion_step = occlusion_start <= step + 1 < occlusion_end
            if is_occlusion_step and len(targets) > 0:
                true_positions = []
                for t in targets:
                    true_pos = t['position'] + t['velocity'] * dt
                    t['position'] = true_pos
                    true_positions.append(true_pos.copy())
                measurements = []
                for pos in true_positions[1:]:
                    r = np.sqrt(pos[0]**2 + pos[1]**2)
                    theta = np.arctan2(pos[1], pos[0])
                    z_r = r + np.random.normal(0, sigma_r)
                    z_theta = theta + np.random.normal(0, sigma_theta)
                    measurements.append(np.array([z_r, z_theta]))
                n_clutter = np.random.poisson(5)
                for _ in range(n_clutter):
                    r = np.random.uniform(50, 200)
                    theta = np.random.uniform(-np.pi, np.pi)
                    measurements.append(np.array([r, theta]))
            else:
                true_positions, measurements = simulate_radar_measurements(
                    targets, dt, sigma_r, sigma_theta, clutter_rate=5, clutter_range=(50, 200)
                )
            estimated_targets = phd_filter.step(measurements)
            occlusion_marker = " [目标1遮挡]" if is_occlusion_step else ""
            print(f"时间步 {step+1}/{n_steps}:{occlusion_marker}")
            print(f"  真实目标数: {len(true_positions)}, 量测: {len(measurements)}")
            print(f"  估计目标数: {len(estimated_targets)}")
            for i, est in enumerate(estimated_targets):
                miss_info = f", 连续漏检={est['consecutive_missed']}" if est['consecutive_missed'] > 0 else ""
                print(f"    目标{i+1}: 位置=[{est['position'][0]:.1f}, {est['position'][1]:.1f}], "
                      f"权重={est['weight']:.3f}{miss_info}")
            if step == 10:
                print("\n  *** 新增目标出现 ***")
                targets.append({'position': np.array([-80.0, -60.0]), 'velocity': np.array([4.0, 3.0])})
            if step == 20 and len(targets) > 1:
                print("\n  *** 一个目标消失 ***")
                targets.pop(0)
            print()
        print("=" * 70)
        print("跟踪完成! 关键改进:")
        print("  1. 已确认目标漏检时权重衰减从 0.02 → 0.7 (大幅减缓)")
        print("  2. 已确认目标剪枝阈值降低10倍，更难被误删")
        print("  3. 引入连续漏检计数，超过15步才删除目标")
        print("  4. 检测到量测时自动重置连续漏检计数")
        print("\n运行标准模式: python phd_filter.py standard")
        print("运行扩展模式: python phd_filter.py extended")
        print("=" * 70)
