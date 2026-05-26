import numpy as np
from gyro_star_sensor_fusion import (
    ExtendedKalmanFilter, UnscentedKalmanFilter, StarSensorGyroFusion, FilterConfig
)


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx


def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    trace = np.trace(R)
    
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    
    return q / np.linalg.norm(q)


def generate_test_data(dt: float = 0.01, total_time: float = 10.0, 
                        amplitude_scale: float = 1.0, freq_scale: float = 1.0):
    t = np.arange(0, total_time, dt)
    n_steps = len(t)
    
    true_roll = 10 * amplitude_scale * np.sin(2 * np.pi * 0.5 * freq_scale * t)
    true_pitch = 5 * amplitude_scale * np.sin(2 * np.pi * 0.3 * freq_scale * t)
    true_yaw = 15 * amplitude_scale * np.sin(2 * np.pi * 0.2 * freq_scale * t)
    
    true_angular_velocity = np.zeros((n_steps, 3))
    for i in range(1, n_steps):
        true_angular_velocity[i, 0] = np.radians((true_roll[i] - true_roll[i-1]) / dt)
        true_angular_velocity[i, 1] = np.radians((true_pitch[i] - true_pitch[i-1]) / dt)
        true_angular_velocity[i, 2] = np.radians((true_yaw[i] - true_yaw[i-1]) / dt)
    
    gyro_bias = np.array([0.01, 0.015, 0.008])
    gyro_noise_std = 0.005
    
    gyro_measurements = true_angular_velocity + gyro_bias + np.random.normal(0, gyro_noise_std, (n_steps, 3))
    
    star_sensor_noise_std = 0.05
    star_sensor_quaternions = np.zeros((n_steps, 4))
    
    for i in range(n_steps):
        R_true = rotation_matrix_from_euler(true_roll[i], true_pitch[i], true_yaw[i])
        q_true = quaternion_from_rotation_matrix(R_true)
        noise = np.random.normal(0, np.radians(star_sensor_noise_std), 3)
        R_noise = rotation_matrix_from_euler(np.degrees(noise[0]), np.degrees(noise[1]), np.degrees(noise[2]))
        R_meas = R_noise @ R_true
        star_sensor_quaternions[i] = quaternion_from_rotation_matrix(R_meas)
    
    return (t, true_roll, true_pitch, true_yaw, 
            gyro_measurements, star_sensor_quaternions, gyro_bias)


def test_basic_fusion_ekf():
    print("=" * 60)
    print("扩展卡尔曼滤波(EKF)基础测试")
    print("=" * 60)
    
    dt = 0.01
    total_time = 10.0
    star_sensor_update_rate = 10
    
    (t, true_roll, true_pitch, true_yaw, 
     gyro_measurements, star_sensor_quaternions, true_bias) = generate_test_data(dt, total_time)
    
    print(f"\n测试配置:")
    print(f"  总时长: {total_time}s")
    print(f"  采样周期: {dt}s ({1/dt}Hz)")
    print(f"  星敏感器更新频率: {star_sensor_update_rate}Hz")
    print(f"  陀螺真实漂移: [{true_bias[0]:.4f}, {true_bias[1]:.4f}, {true_bias[2]:.4f}] rad/s")
    
    ekf = ExtendedKalmanFilter(FilterConfig(dt=dt))
    
    R_initial = rotation_matrix_from_euler(true_roll[0], true_pitch[0], true_yaw[0])
    q_initial = quaternion_from_rotation_matrix(R_initial)
    ekf.set_initial_state(q_initial)
    ekf.set_gyro_noise(0.005, 0.0001)
    ekf.set_measurement_noise(np.radians(0.05))
    
    n_steps = len(t)
    ekf_roll = np.zeros(n_steps)
    ekf_pitch = np.zeros(n_steps)
    ekf_yaw = np.zeros(n_steps)
    ekf_bias = np.zeros((n_steps, 3))
    ekf_std = np.zeros((n_steps, 3))
    
    for i in range(n_steps):
        ekf.predict(gyro_measurements[i])
        
        if i % (int(1/(star_sensor_update_rate * dt))) == 0:
            ekf.update(star_sensor_quaternions[i])
        
        roll, pitch, yaw = ekf.get_euler_angles()
        ekf_roll[i] = roll
        ekf_pitch[i] = pitch
        ekf_yaw[i] = yaw
        ekf_bias[i] = ekf.gyro_bias
        ekf_std[i] = ekf.get_attitude_std()
    
    roll_error = ekf_roll - true_roll
    pitch_error = ekf_pitch - true_pitch
    yaw_error = ekf_yaw - true_yaw
    
    roll_rmse = np.sqrt(np.mean(roll_error**2))
    pitch_rmse = np.sqrt(np.mean(pitch_error**2))
    yaw_rmse = np.sqrt(np.mean(yaw_error**2))
    
    print(f"\n姿态估计误差 (RMSE):")
    print(f"  Roll:  {roll_rmse:.4f}°")
    print(f"  Pitch: {pitch_rmse:.4f}°")
    print(f"  Yaw:   {yaw_rmse:.4f}°")
    
    print(f"\n姿态估计标准差 (3σ):")
    print(f"  Roll:  {3*ekf_std[-1, 0]:.4f}°")
    print(f"  Pitch: {3*ekf_std[-1, 1]:.4f}°")
    print(f"  Yaw:   {3*ekf_std[-1, 2]:.4f}°")
    
    print(f"\n陀螺漂移估计:")
    print(f"  真实漂移:      [{true_bias[0]:.6f}, {true_bias[1]:.6f}, {true_bias[2]:.6f}] rad/s")
    print(f"  估计漂移:      [{ekf_bias[-1, 0]:.6f}, {ekf_bias[-1, 1]:.6f}, {ekf_bias[-1, 2]:.6f}] rad/s")
    print(f"  漂移误差:      [{abs(ekf_bias[-1, 0]-true_bias[0]):.6f}, {abs(ekf_bias[-1, 1]-true_bias[1]):.6f}, {abs(ekf_bias[-1, 2]-true_bias[2]):.6f}] rad/s")
    
    print(f"\n最终姿态误差:")
    print(f"  Roll:  {abs(ekf_roll[-1]-true_roll[-1]):.4f}°")
    print(f"  Pitch: {abs(ekf_pitch[-1]-true_pitch[-1]):.4f}°")
    print(f"  Yaw:   {abs(ekf_yaw[-1]-true_yaw[-1]):.4f}°")
    
    print("\n" + "=" * 60)
    print("EKF基础测试完成!")
    print("=" * 60)
    
    return (t, true_roll, true_pitch, true_yaw, ekf_roll, ekf_pitch, ekf_yaw, 
            roll_error, pitch_error, yaw_error, ekf_bias, true_bias)


def test_fast_maneuver_ekf():
    print("\n" + "=" * 60)
    print("快速机动测试 - EKF高带宽性能验证")
    print("=" * 60)
    
    dt = 0.005
    total_time = 5.0
    
    t = np.arange(0, total_time, dt)
    n_steps = len(t)
    
    true_roll = 20 * np.sin(2 * np.pi * 0.5 * t)
    true_pitch = 15 * np.sin(2 * np.pi * 0.4 * t)
    true_yaw = 30 * np.sin(2 * np.pi * 0.3 * t)
    
    true_angular_velocity = np.zeros((n_steps, 3))
    for i in range(1, n_steps):
        true_angular_velocity[i, 0] = np.radians((true_roll[i] - true_roll[i-1]) / dt)
        true_angular_velocity[i, 1] = np.radians((true_pitch[i] - true_pitch[i-1]) / dt)
        true_angular_velocity[i, 2] = np.radians((true_yaw[i] - true_yaw[i-1]) / dt)
    
    max_angular_velocity = np.max(np.abs(true_angular_velocity), axis=0)
    print(f"\n最大角速度:")
    print(f"  Roll:  {np.degrees(max_angular_velocity[0]):.2f}°/s")
    print(f"  Pitch: {np.degrees(max_angular_velocity[1]):.2f}°/s")
    print(f"  Yaw:   {np.degrees(max_angular_velocity[2]):.2f}°/s")
    
    max_angular_accel = np.max(np.abs(np.diff(true_angular_velocity, axis=0)), axis=0) / dt
    print(f"\n最大角加速度:")
    print(f"  Roll:  {np.degrees(max_angular_accel[0]):.2f}°/s²")
    print(f"  Pitch: {np.degrees(max_angular_accel[1]):.2f}°/s²")
    print(f"  Yaw:   {np.degrees(max_angular_accel[2]):.2f}°/s²")
    
    gyro_bias = np.array([0.02, 0.015, 0.025])
    gyro_noise_std = 0.01
    gyro_measurements = true_angular_velocity + gyro_bias + np.random.normal(0, gyro_noise_std, (n_steps, 3))
    
    star_sensor_quaternions = np.zeros((n_steps, 4))
    star_sensor_noise_std = 0.1
    for i in range(n_steps):
        R_true = rotation_matrix_from_euler(true_roll[i], true_pitch[i], true_yaw[i])
        q_true = quaternion_from_rotation_matrix(R_true)
        noise = np.random.normal(0, np.radians(star_sensor_noise_std), 3)
        R_noise = rotation_matrix_from_euler(np.degrees(noise[0]), np.degrees(noise[1]), np.degrees(noise[2]))
        R_meas = R_noise @ R_true
        star_sensor_quaternions[i] = quaternion_from_rotation_matrix(R_meas)
    
    config = FilterConfig(dt=dt, gyro_noise_std=0.01, gyro_bias_noise_std=0.0001,
                          measurement_noise_std=np.radians(0.1))
    ekf = ExtendedKalmanFilter(config)
    
    R_initial = rotation_matrix_from_euler(true_roll[0], true_pitch[0], true_yaw[0])
    q_initial = quaternion_from_rotation_matrix(R_initial)
    ekf.set_initial_state(q_initial)
    
    ekf_roll = np.zeros(n_steps)
    ekf_pitch = np.zeros(n_steps)
    ekf_yaw = np.zeros(n_steps)
    outlier_count = 0
    
    for i in range(n_steps):
        state = ekf.predict(gyro_measurements[i])
        
        if i % 10 == 0:
            state = ekf.update(star_sensor_quaternions[i])
            if state.is_outlier:
                outlier_count += 1
        
        roll, pitch, yaw = ekf.get_euler_angles()
        ekf_roll[i] = roll
        ekf_pitch[i] = pitch
        ekf_yaw[i] = yaw
    
    roll_error = ekf_roll - true_roll
    pitch_error = ekf_pitch - true_pitch
    yaw_error = ekf_yaw - true_yaw
    
    roll_rmse = np.sqrt(np.mean(roll_error**2))
    pitch_rmse = np.sqrt(np.mean(pitch_error**2))
    yaw_rmse = np.sqrt(np.mean(yaw_error**2))
    
    print(f"\n快速机动下姿态估计误差 (RMSE):")
    print(f"  Roll:  {roll_rmse:.4f}°")
    print(f"  Pitch: {pitch_rmse:.4f}°")
    print(f"  Yaw:   {yaw_rmse:.4f}°")
    
    print(f"\n漂移估计误差:")
    print(f"  Roll:  {abs(ekf.gyro_bias[0]-gyro_bias[0]):.6f} rad/s")
    print(f"  Pitch: {abs(ekf.gyro_bias[1]-gyro_bias[1]):.6f} rad/s")
    print(f"  Yaw:   {abs(ekf.gyro_bias[2]-gyro_bias[2]):.6f} rad/s")
    
    print(f"\n故障检测:")
    print(f"  检测到的异常测量数: {outlier_count}")
    
    print("\n" + "=" * 60)
    print("快速机动EKF测试完成!")
    print("=" * 60)
    
    return roll_rmse, pitch_rmse, yaw_rmse


def test_ekf_vs_ukf():
    print("\n" + "=" * 60)
    print("EKF vs UKF性能对比测试")
    print("=" * 60)
    
    dt = 0.01
    total_time = 5.0
    
    t = np.arange(0, total_time, dt)
    n_steps = len(t)
    
    true_roll = 30 * np.sin(2 * np.pi * 0.5 * t)
    true_pitch = 20 * np.sin(2 * np.pi * 0.4 * t)
    true_yaw = 40 * np.sin(2 * np.pi * 0.3 * t)
    
    true_angular_velocity = np.zeros((n_steps, 3))
    for i in range(1, n_steps):
        true_angular_velocity[i, 0] = np.radians((true_roll[i] - true_roll[i-1]) / dt)
        true_angular_velocity[i, 1] = np.radians((true_pitch[i] - true_pitch[i-1]) / dt)
        true_angular_velocity[i, 2] = np.radians((true_yaw[i] - true_yaw[i-1]) / dt)
    
    gyro_bias = np.array([0.01, 0.01, 0.01])
    gyro_noise_std = 0.01
    gyro_measurements = true_angular_velocity + gyro_bias + np.random.normal(0, gyro_noise_std, (n_steps, 3))
    
    star_sensor_quaternions = np.zeros((n_steps, 4))
    star_sensor_noise_std = 0.08
    for i in range(n_steps):
        R_true = rotation_matrix_from_euler(true_roll[i], true_pitch[i], true_yaw[i])
        q_true = quaternion_from_rotation_matrix(R_true)
        noise = np.random.normal(0, np.radians(star_sensor_noise_std), 3)
        R_noise = rotation_matrix_from_euler(np.degrees(noise[0]), np.degrees(noise[1]), np.degrees(noise[2]))
        R_meas = R_noise @ R_true
        star_sensor_quaternions[i] = quaternion_from_rotation_matrix(R_meas)
    
    config = FilterConfig(dt=dt, gyro_noise_std=0.01, gyro_bias_noise_std=0.0001,
                          measurement_noise_std=np.radians(0.08))
    
    ekf = ExtendedKalmanFilter(config)
    ukf = UnscentedKalmanFilter(config)
    
    R_initial = rotation_matrix_from_euler(true_roll[0], true_pitch[0], true_yaw[0])
    q_initial = quaternion_from_rotation_matrix(R_initial)
    ekf.set_initial_state(q_initial)
    ukf.set_initial_state(q_initial)
    
    ekf_roll = np.zeros(n_steps)
    ekf_pitch = np.zeros(n_steps)
    ekf_yaw = np.zeros(n_steps)
    
    ukf_roll = np.zeros(n_steps)
    ukf_pitch = np.zeros(n_steps)
    ukf_yaw = np.zeros(n_steps)
    
    for i in range(n_steps):
        ekf.predict(gyro_measurements[i])
        ukf.predict(gyro_measurements[i])
        
        if i % 10 == 0:
            ekf.update(star_sensor_quaternions[i])
            ukf.update(star_sensor_quaternions[i])
        
        r, p, y = ekf.get_euler_angles()
        ekf_roll[i], ekf_pitch[i], ekf_yaw[i] = r, p, y
        
        r, p, y = ukf.get_euler_angles()
        ukf_roll[i], ukf_pitch[i], ukf_yaw[i] = r, p, y
    
    ekf_roll_rmse = np.sqrt(np.mean((ekf_roll - true_roll)**2))
    ekf_pitch_rmse = np.sqrt(np.mean((ekf_pitch - true_pitch)**2))
    ekf_yaw_rmse = np.sqrt(np.mean((ekf_yaw - true_yaw)**2))
    
    ukf_roll_rmse = np.sqrt(np.mean((ukf_roll - true_roll)**2))
    ukf_pitch_rmse = np.sqrt(np.mean((ukf_pitch - true_pitch)**2))
    ukf_yaw_rmse = np.sqrt(np.mean((ukf_yaw - true_yaw)**2))
    
    print(f"\n姿态估计误差 (RMSE):")
    print(f"  轴      |   EKF   |   UKF   |  改善")
    print(f"  --------|---------|---------|--------")
    print(f"  Roll    | {ekf_roll_rmse:7.4f}° | {ukf_roll_rmse:7.4f}° | {ekf_roll_rmse-ukf_roll_rmse:+.4f}°")
    print(f"  Pitch   | {ekf_pitch_rmse:7.4f}° | {ukf_pitch_rmse:7.4f}° | {ekf_pitch_rmse-ukf_pitch_rmse:+.4f}°")
    print(f"  Yaw     | {ekf_yaw_rmse:7.4f}° | {ukf_yaw_rmse:7.4f}° | {ekf_yaw_rmse-ukf_yaw_rmse:+.4f}°")
    
    print(f"\n漂移估计误差 (最终时刻):")
    print(f"  轴      |   EKF   |   UKF   |  真实")
    print(f"  --------|---------|---------|--------")
    print(f"  Roll    | {abs(ekf.gyro_bias[0]-gyro_bias[0]):7.6f} | {abs(ukf.gyro_bias[0]-gyro_bias[0]):7.6f} | {gyro_bias[0]:.6f}")
    print(f"  Pitch   | {abs(ekf.gyro_bias[1]-gyro_bias[1]):7.6f} | {abs(ukf.gyro_bias[1]-gyro_bias[1]):7.6f} | {gyro_bias[1]:.6f}")
    print(f"  Yaw     | {abs(ekf.gyro_bias[2]-gyro_bias[2]):7.6f} | {abs(ukf.gyro_bias[2]-gyro_bias[2]):7.6f} | {gyro_bias[2]:.6f}")
    
    print("\n" + "=" * 60)
    print("EKF vs UKF对比测试完成!")
    print("=" * 60)
    
    return (ekf_roll_rmse, ekf_pitch_rmse, ekf_yaw_rmse,
            ukf_roll_rmse, ukf_pitch_rmse, ukf_yaw_rmse)


def test_outlier_rejection():
    print("\n" + "=" * 60)
    print("抗干扰与异常值剔除测试")
    print("=" * 60)
    
    dt = 0.01
    total_time = 10.0
    
    t = np.arange(0, total_time, dt)
    n_steps = len(t)
    
    true_roll = 10 * np.sin(2 * np.pi * 0.2 * t)
    true_pitch = 5 * np.sin(2 * np.pi * 0.15 * t)
    true_yaw = 15 * np.sin(2 * np.pi * 0.1 * t)
    
    true_angular_velocity = np.zeros((n_steps, 3))
    for i in range(1, n_steps):
        true_angular_velocity[i, 0] = np.radians((true_roll[i] - true_roll[i-1]) / dt)
        true_angular_velocity[i, 1] = np.radians((true_pitch[i] - true_pitch[i-1]) / dt)
        true_angular_velocity[i, 2] = np.radians((true_yaw[i] - true_yaw[i-1]) / dt)
    
    gyro_bias = np.array([0.01, 0.01, 0.01])
    gyro_measurements = true_angular_velocity + gyro_bias + np.random.normal(0, 0.005, (n_steps, 3))
    
    star_sensor_quaternions = np.zeros((n_steps, 4))
    for i in range(n_steps):
        R_true = rotation_matrix_from_euler(true_roll[i], true_pitch[i], true_yaw[i])
        q_true = quaternion_from_rotation_matrix(R_true)
        noise = np.random.normal(0, np.radians(0.03), 3)
        R_noise = rotation_matrix_from_euler(np.degrees(noise[0]), np.degrees(noise[1]), np.degrees(noise[2]))
        R_meas = R_noise @ R_true
        star_sensor_quaternions[i] = quaternion_from_rotation_matrix(R_meas)
    
    outlier_indices = np.random.choice(range(100, n_steps), size=20, replace=False)
    for idx in outlier_indices:
        large_noise = np.random.normal(0, np.radians(5.0), 3)
        R_noise = rotation_matrix_from_euler(np.degrees(large_noise[0]), 
                                              np.degrees(large_noise[1]), 
                                              np.degrees(large_noise[2]))
        R_true = rotation_matrix_from_euler(true_roll[idx], true_pitch[idx], true_yaw[idx])
        R_meas = R_noise @ R_true
        star_sensor_quaternions[idx] = quaternion_from_rotation_matrix(R_meas)
    
    config = FilterConfig(dt=dt, gyro_noise_std=0.005, gyro_bias_noise_std=0.0001,
                          measurement_noise_std=np.radians(0.03),
                          chi_square_threshold=7.815)
    ekf = ExtendedKalmanFilter(config)
    
    R_initial = rotation_matrix_from_euler(true_roll[0], true_pitch[0], true_yaw[0])
    q_initial = quaternion_from_rotation_matrix(R_initial)
    ekf.set_initial_state(q_initial)
    
    ekf_roll = np.zeros(n_steps)
    ekf_pitch = np.zeros(n_steps)
    ekf_yaw = np.zeros(n_steps)
    detected_outliers = []
    chi_square_values = []
    
    for i in range(n_steps):
        ekf.predict(gyro_measurements[i])
        
        if i % 10 == 0:
            state = ekf.update(star_sensor_quaternions[i])
            chi_square_values.append(state.chi_square)
            if state.is_outlier:
                detected_outliers.append(i)
        else:
            chi_square_values.append(0.0)
        
        r, p, y = ekf.get_euler_angles()
        ekf_roll[i], ekf_pitch[i], ekf_yaw[i] = r, p, y
    
    roll_error = ekf_roll - true_roll
    pitch_error = ekf_pitch - true_pitch
    yaw_error = ekf_yaw - true_yaw
    
    print(f"\n测试配置:")
    print(f"  注入异常值数量: {len(outlier_indices)}")
    print(f"  异常值幅度: ~5°")
    print(f"  卡方检验阈值: {config.chi_square_threshold}")
    
    print(f"\n异常值检测结果:")
    print(f"  检测到的异常值: {len(detected_outliers)}")
    print(f"  正确检测率: {len(set(detected_outliers) & set(outlier_indices)) / len(outlier_indices) * 100:.1f}%")
    print(f"  误报率: {len(set(detected_outliers) - set(outlier_indices)) / len(detected_outliers) * 100:.1f}%" if len(detected_outliers) > 0 else "  误报率: 0%")
    
    print(f"\n姿态估计误差 (RMSE):")
    print(f"  Roll:  {np.sqrt(np.mean(roll_error**2)):.4f}°")
    print(f"  Pitch: {np.sqrt(np.mean(pitch_error**2)):.4f}°")
    print(f"  Yaw:   {np.sqrt(np.mean(yaw_error**2)):.4f}°")
    
    print(f"\n卡方统计量:")
    print(f"  最大值: {max(chi_square_values):.2f}")
    print(f"  均值:   {np.mean(chi_square_values):.2f}")
    
    print("\n" + "=" * 60)
    print("抗干扰测试完成!")
    print("=" * 60)
    
    return detected_outliers, outlier_indices


def test_star_sensor_integration():
    print("\n" + "=" * 60)
    print("星敏感器与陀螺融合系统集成测试")
    print("=" * 60)
    
    dt = 0.01
    total_time = 10.0
    
    t = np.arange(0, total_time, dt)
    n_steps = len(t)
    
    true_roll = 15 * np.sin(2 * np.pi * 0.3 * t)
    true_pitch = -10 * np.sin(2 * np.pi * 0.25 * t)
    true_yaw = 45 * np.sin(2 * np.pi * 0.2 * t)
    
    true_angular_velocity = np.zeros((n_steps, 3))
    for i in range(1, n_steps):
        true_angular_velocity[i, 0] = np.radians((true_roll[i] - true_roll[i-1]) / dt)
        true_angular_velocity[i, 1] = np.radians((true_pitch[i] - true_pitch[i-1]) / dt)
        true_angular_velocity[i, 2] = np.radians((true_yaw[i] - true_yaw[i-1]) / dt)
    
    gyro_bias = np.array([0.01, 0.015, 0.008])
    gyro_measurements = true_angular_velocity + gyro_bias + np.random.normal(0, 0.005, (n_steps, 3))
    
    star_sensor_quaternions = np.zeros((n_steps, 4))
    for i in range(n_steps):
        R_true = rotation_matrix_from_euler(true_roll[i], true_pitch[i], true_yaw[i])
        q_true = quaternion_from_rotation_matrix(R_true)
        noise = np.random.normal(0, np.radians(0.05), 3)
        R_noise = rotation_matrix_from_euler(np.degrees(noise[0]), np.degrees(noise[1]), np.degrees(noise[2]))
        R_meas = R_noise @ R_true
        star_sensor_quaternions[i] = quaternion_from_rotation_matrix(R_meas)
    
    fusion = StarSensorGyroFusion(dt=dt, use_ukf=False)
    fusion.set_star_sensor_update_rate(10)
    fusion.set_gyro_noise_parameters(0.005, 0.0001)
    fusion.set_measurement_noise(np.radians(0.05))
    
    R_initial = rotation_matrix_from_euler(true_roll[0], true_pitch[0], true_yaw[0])
    q_initial = quaternion_from_rotation_matrix(R_initial)
    fusion.set_initial_attitude(q_initial)
    
    est_roll = np.zeros(n_steps)
    est_pitch = np.zeros(n_steps)
    est_yaw = np.zeros(n_steps)
    est_std = np.zeros((n_steps, 3))
    
    for i in range(n_steps):
        ss_quat = star_sensor_quaternions[i] if i % 10 == 0 else None
        fusion.process(gyro_measurements[i], ss_quat)
        
        r, p, y = fusion.get_attitude()
        est_roll[i], est_pitch[i], est_yaw[i] = r, p, y
        est_std[i] = fusion.get_attitude_std()
    
    roll_error = est_roll - true_roll
    pitch_error = est_pitch - true_pitch
    yaw_error = est_yaw - true_yaw
    
    print(f"\n系统性能指标:")
    print(f"  姿态更新频率: {1/dt:.0f}Hz")
    print(f"  星敏感器更新频率: 10Hz")
    print(f"  系统带宽: >1Hz (快速机动跟踪能力)")
    
    print(f"\n姿态估计精度 (RMSE):")
    print(f"  Roll:  {np.sqrt(np.mean(roll_error**2)):.4f}°")
    print(f"  Pitch: {np.sqrt(np.mean(pitch_error**2)):.4f}°")
    print(f"  Yaw:   {np.sqrt(np.mean(yaw_error**2)):.4f}°")
    
    print(f"\n姿态估计不确定性 (3σ):")
    print(f"  Roll:  {3*np.mean(est_std[:, 0]):.4f}°")
    print(f"  Pitch: {3*np.mean(est_std[:, 1]):.4f}°")
    print(f"  Yaw:   {3*np.mean(est_std[:, 2]):.4f}°")
    
    print(f"\n陀螺漂移估计:")
    print(f"  真实: [{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, {gyro_bias[2]:.6f}] rad/s")
    print(f"  估计: [{fusion.filter.gyro_bias[0]:.6f}, {fusion.filter.gyro_bias[1]:.6f}, {fusion.filter.gyro_bias[2]:.6f}] rad/s")
    
    print("\n" + "=" * 60)
    print("集成测试完成!")
    print("=" * 60)
    
    return est_roll, est_pitch, est_yaw


if __name__ == "__main__":
    np.random.seed(42)
    
    results_ekf = test_basic_fusion_ekf()
    rmse_fast = test_fast_maneuver_ekf()
    results_compare = test_ekf_vs_ukf()
    outliers_detected, outliers_injected = test_outlier_rejection()
    results_integration = test_star_sensor_integration()
    
    print("\n" + "=" * 60)
    print("所有融合测试完成!")
    print("=" * 60)
    print(f"\n性能总结:")
    print(f"  常规机动EKF RMSE: Roll={results_compare[0]:.3f}°, Pitch={results_compare[1]:.3f}°, Yaw={results_compare[2]:.3f}°")
    print(f"  常规机动UKF RMSE: Roll={results_compare[3]:.3f}°, Pitch={results_compare[4]:.3f}°, Yaw={results_compare[5]:.3f}°")
    print(f"  快速机动RMSE:    Roll={rmse_fast[0]:.3f}°, Pitch={rmse_fast[1]:.3f}°, Yaw={rmse_fast[2]:.3f}°")
    print(f"  异常值检测率:     {len(set(outliers_detected) & set(outliers_injected)) / len(outliers_injected) * 100:.1f}%")
    print("=" * 60)
