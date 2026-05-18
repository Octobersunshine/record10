import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter1D:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = np.array(initial_state, dtype=float)
        self.covariance = np.array(initial_covariance, dtype=float)
        self.process_noise = np.array(process_noise, dtype=float)
        self.measurement_noise = measurement_noise

        self.dt = 1.0
        self.F = np.array([[1, self.dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.B = np.array([[0.5 * self.dt**2], [self.dt]])

    def predict(self, acceleration=0):
        self.state = self.F @ self.state + self.B.flatten() * acceleration
        self.covariance = self.F @ self.covariance @ self.F.T + self.process_noise

    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.measurement_noise
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        
        I_KH = np.eye(2) - K @ self.H
        R_matrix = np.array([[self.measurement_noise]])
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R_matrix @ K.T
        self.covariance = (self.covariance + self.covariance.T) / 2
        eps = 1e-8
        self.covariance[0, 0] += eps
        self.covariance[1, 1] += eps

    def get_state(self):
        return self.state[0], self.state[1]


def generate_sensor_data(true_positions, noise_std):
    return true_positions + np.random.normal(0, noise_std, size=true_positions.shape)


def simulate_constant_velocity(num_steps, initial_position, velocity, measurement_noise_std):
    time = np.arange(num_steps)
    true_positions = initial_position + velocity * time
    measurements = generate_sensor_data(true_positions, measurement_noise_std)
    return time, true_positions, measurements


def simulate_constant_acceleration(num_steps, initial_position, initial_velocity, acceleration, measurement_noise_std):
    time = np.arange(num_steps)
    true_positions = initial_position + initial_velocity * time + 0.5 * acceleration * time**2
    true_velocities = initial_velocity + acceleration * time
    measurements = generate_sensor_data(true_positions, measurement_noise_std)
    return time, true_positions, true_velocities, measurements


def run_constant_velocity_demo():
    num_steps = 50
    initial_position = 0
    velocity = 2.0
    measurement_noise_std = 3.0

    time, true_positions, measurements = simulate_constant_velocity(
        num_steps, initial_position, velocity, measurement_noise_std
    )

    initial_state = [measurements[0], 0]
    initial_covariance = [[100, 0], [0, 100]]
    process_noise = [[0.1, 0], [0, 0.1]]
    measurement_noise = measurement_noise_std**2

    kf = KalmanFilter1D(initial_state, initial_covariance, process_noise, measurement_noise)

    estimated_positions = []
    estimated_velocities = []

    for z in measurements:
        kf.predict()
        kf.update(z)
        pos, vel = kf.get_state()
        estimated_positions.append(pos)
        estimated_velocities.append(vel)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(time, true_positions, 'g-', label='真实位置')
    plt.plot(time, measurements, 'r.', markersize=4, label='观测值')
    plt.plot(time, estimated_positions, 'b-', label='估计位置')
    plt.xlabel('时间步')
    plt.ylabel('位置')
    plt.legend()
    plt.title('位置估计 (匀速运动)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(time, [velocity]*num_steps, 'g-', label='真实速度')
    plt.plot(time, estimated_velocities, 'b-', label='估计速度')
    plt.xlabel('时间步')
    plt.ylabel('速度')
    plt.legend()
    plt.title('速度估计 (匀速运动)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('kalman_constant_velocity.png', dpi=150)
    plt.show()

    print(f"最终位置误差: {abs(true_positions[-1] - estimated_positions[-1]):.4f}")
    print(f"最终速度误差: {abs(velocity - estimated_velocities[-1]):.4f}")


def run_constant_acceleration_demo():
    num_steps = 50
    initial_position = 0
    initial_velocity = 0
    acceleration = 0.5
    measurement_noise_std = 5.0

    time, true_positions, true_velocities, measurements = simulate_constant_acceleration(
        num_steps, initial_position, initial_velocity, acceleration, measurement_noise_std
    )

    initial_state = [measurements[0], 0]
    initial_covariance = [[100, 0], [0, 100]]
    process_noise = [[1, 0], [0, 1]]
    measurement_noise = measurement_noise_std**2

    kf = KalmanFilter1D(initial_state, initial_covariance, process_noise, measurement_noise)

    estimated_positions = []
    estimated_velocities = []

    for z in measurements:
        kf.predict(acceleration=acceleration)
        kf.update(z)
        pos, vel = kf.get_state()
        estimated_positions.append(pos)
        estimated_velocities.append(vel)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(time, true_positions, 'g-', label='真实位置')
    plt.plot(time, measurements, 'r.', markersize=4, label='观测值')
    plt.plot(time, estimated_positions, 'b-', label='估计位置')
    plt.xlabel('时间步')
    plt.ylabel('位置')
    plt.legend()
    plt.title('位置估计 (匀加速运动)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(time, true_velocities, 'g-', label='真实速度')
    plt.plot(time, estimated_velocities, 'b-', label='估计速度')
    plt.xlabel('时间步')
    plt.ylabel('速度')
    plt.legend()
    plt.title('速度估计 (匀加速运动)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('kalman_constant_acceleration.png', dpi=150)
    plt.show()

    print(f"最终位置误差: {abs(true_positions[-1] - estimated_positions[-1]):.4f}")
    print(f"最终速度误差: {abs(true_velocities[-1] - estimated_velocities[-1]):.4f}")


class AdaptiveKalmanFilter1D:
    def __init__(self, initial_state, initial_covariance, initial_process_noise, 
                 initial_measurement_noise, window_size=10, 
                 adapt_process_noise=True, adapt_measurement_noise=True):
        self.state = np.array(initial_state, dtype=float)
        self.covariance = np.array(initial_covariance, dtype=float)
        self.process_noise = np.array(initial_process_noise, dtype=float)
        self.measurement_noise = initial_measurement_noise
        
        self.dt = 1.0
        self.F = np.array([[1, self.dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.B = np.array([[0.5 * self.dt**2], [self.dt]])
        
        self.window_size = window_size
        self.residual_window = []
        self.innovation_cov_window = []
        
        self.adapt_process_noise = adapt_process_noise
        self.adapt_measurement_noise = adapt_measurement_noise
        
        self.q_history = []
        self.r_history = []
        
        self.alpha = 2 / (window_size + 1)

    def predict(self, acceleration=0):
        self.state = self.F @ self.state + self.B.flatten() * acceleration
        self.covariance = self.F @ self.covariance @ self.F.T + self.process_noise

    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.measurement_noise
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        
        I_KH = np.eye(2) - K @ self.H
        R_matrix = np.array([[self.measurement_noise]])
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R_matrix @ K.T
        self.covariance = (self.covariance + self.covariance.T) / 2
        eps = 1e-8
        self.covariance[0, 0] += eps
        self.covariance[1, 1] += eps
        
        self._adapt_noise_covariance(y, S)

    def _adapt_noise_covariance(self, residual, S_predicted):
        self.residual_window.append(residual)
        self.innovation_cov_window.append(S_predicted)
        
        if len(self.residual_window) > self.window_size:
            self.residual_window.pop(0)
            self.innovation_cov_window.pop(0)
        
        if len(self.residual_window) >= self.window_size // 2:
            residuals = np.array(self.residual_window)
            actual_innovation_cov = np.var(residuals)
            
            if self.adapt_measurement_noise:
                predicted_innovation = np.mean(self.innovation_cov_window)
                r_adjustment = actual_innovation_cov - predicted_innovation
                new_R = self.measurement_noise + self.alpha * r_adjustment
                self.measurement_noise = max(1e-6, new_R)
            
            if self.adapt_process_noise:
                HPH = self.H @ self.covariance @ self.H.T
                q_estimate = actual_innovation_cov - self.measurement_noise - HPH
                q_scale = max(0.1, 1.0 + self.alpha * q_estimate / np.trace(self.process_noise))
                self.process_noise = self.process_noise * q_scale
        
        self.q_history.append(np.trace(self.process_noise))
        self.r_history.append(self.measurement_noise)

    def get_state(self):
        return self.state[0], self.state[1]

    def get_noise_params(self):
        return self.process_noise.copy(), self.measurement_noise


def generate_sensor_data_with_changing_noise(true_positions, noise_stds):
    measurements = []
    for pos, noise_std in zip(true_positions, noise_stds):
        measurements.append(pos + np.random.normal(0, noise_std))
    return np.array(measurements)


def run_adaptive_comparison_demo():
    print("=" * 60)
    print("自适应卡尔曼滤波 vs 标准卡尔曼滤波")
    print("=" * 60)
    
    num_steps = 100
    initial_position = 0
    velocity = 2.0
    
    time = np.arange(num_steps)
    true_positions = initial_position + velocity * time
    
    noise_std_profile = np.ones(num_steps) * 3.0
    noise_std_profile[30:60] = 10.0
    noise_std_profile[60:] = 1.0
    
    measurements = generate_sensor_data_with_changing_noise(true_positions, noise_std_profile)
    
    initial_state = [measurements[0], 0]
    initial_covariance = [[100, 0], [0, 100]]
    initial_process_noise = [[0.1, 0], [0, 0.1]]
    initial_measurement_noise = 9.0
    
    kf_std = AdaptiveKalmanFilter1D(
        initial_state, initial_covariance, initial_process_noise, 
        initial_measurement_noise, adapt_process_noise=False, 
        adapt_measurement_noise=False
    )
    
    kf_adaptive = AdaptiveKalmanFilter1D(
        initial_state, initial_covariance, initial_process_noise,
        initial_measurement_noise, window_size=15,
        adapt_process_noise=False, adapt_measurement_noise=True
    )
    
    std_positions = []
    adaptive_positions = []
    r_estimates = []
    
    for z in measurements:
        kf_std.predict()
        kf_std.update(z)
        pos_std, _ = kf_std.get_state()
        std_positions.append(pos_std)
        
        kf_adaptive.predict()
        kf_adaptive.update(z)
        pos_adapt, _ = kf_adaptive.get_state()
        adaptive_positions.append(pos_adapt)
        _, r = kf_adaptive.get_noise_params()
        r_estimates.append(r)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(time, true_positions, 'g-', label='真实位置', linewidth=2)
    ax1.plot(time, measurements, 'r.', markersize=3, alpha=0.5, label='观测值')
    ax1.plot(time, std_positions, 'b--', label='标准卡尔曼')
    ax1.plot(time, adaptive_positions, 'm-', label='自适应卡尔曼')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('位置')
    ax1.legend()
    ax1.set_title('位置估计对比')
    ax1.grid(True)
    
    ax2 = axes[0, 1]
    std_error = np.abs(np.array(std_positions) - true_positions)
    adaptive_error = np.abs(np.array(adaptive_positions) - true_positions)
    ax2.plot(time, std_error, 'b-', label=f'标准卡尔曼 (平均={np.mean(std_error):.3f})')
    ax2.plot(time, adaptive_error, 'm-', label=f'自适应卡尔曼 (平均={np.mean(adaptive_error):.3f})')
    ax2.axvspan(30, 60, alpha=0.2, color='orange', label='高噪声期')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('绝对误差')
    ax2.legend()
    ax2.set_title('估计误差对比')
    ax2.grid(True)
    
    ax3 = axes[1, 0]
    ax3.plot(time, noise_std_profile**2, 'r-', label='真实噪声方差', linewidth=2)
    ax3.plot(time, r_estimates, 'm-', label='自适应估计值')
    ax3.axhline(y=initial_measurement_noise, color='b', linestyle='--', 
                label='标准卡尔曼(固定值)')
    ax3.axvspan(30, 60, alpha=0.2, color='orange')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('测量噪声方差 R')
    ax3.legend()
    ax3.set_title('R参数自适应估计')
    ax3.grid(True)
    
    ax4 = axes[1, 1]
    improvement = (np.mean(std_error) - np.mean(adaptive_error)) / np.mean(std_error) * 100
    methods = ['标准卡尔曼', '自适应卡尔曼']
    errors = [np.mean(std_error), np.mean(adaptive_error)]
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax4.bar(methods, errors, color=colors, alpha=0.7)
    ax4.set_ylabel('平均位置误差')
    ax4.set_title(f'性能对比 (改进 {improvement:.1f}%)')
    ax4.grid(True, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('adaptive_kalman_comparison.png', dpi=150)
    plt.show()
    
    print("\n结果统计:")
    print("-" * 60)
    print(f"标准卡尔曼平均误差: {np.mean(std_error):.4f}")
    print(f"自适应卡尔曼平均误差: {np.mean(adaptive_error):.4f}")
    print(f"相对改进: {improvement:.2f}%")
    print("\n噪声估计精度:")
    print(f"{'阶段':^12} | {'真实R':^10} | {'估计R':^10} | {'误差%':^10}")
    print("-" * 50)
    for start, end, name in [(0, 30, '低噪声期'), (30, 60, '高噪声期'), (60, 100, '超低噪声')]:
        true_r = np.mean(noise_std_profile[start:end]**2)
        est_r = np.mean(r_estimates[start:end])
        err_pct = abs(est_r - true_r) / true_r * 100
        print(f"{name:^12} | {true_r:^10.2f} | {est_r:^10.2f} | {err_pct:^10.1f}%")


def real_time_demo():
    print("=" * 60)
    print("一维卡尔曼滤波 - 实时演示")
    print("=" * 60)
    
    measurement_noise_std = 2.0
    initial_state = [0, 0]
    initial_covariance = [[10, 0], [0, 10]]
    process_noise = [[0.01, 0], [0, 0.01]]
    measurement_noise = measurement_noise_std**2

    kf = KalmanFilter1D(initial_state, initial_covariance, process_noise, measurement_noise)

    true_position = 0
    true_velocity = 1.0

    print("\n时间步 | 真实位置 | 观测值 | 估计位置 | 估计速度")
    print("-" * 55)

    for step in range(20):
        true_position += true_velocity
        measurement = true_position + np.random.normal(0, measurement_noise_std)

        kf.predict()
        kf.update(measurement)
        est_pos, est_vel = kf.get_state()

        print(f"{step:6d} | {true_position:8.2f} | {measurement:7.2f} | {est_pos:8.2f} | {est_vel:8.2f}")


if __name__ == "__main__":
    np.random.seed(42)

    print("1. 匀速运动演示")
    run_constant_velocity_demo()

    print("\n" + "="*60)
    print("2. 匀加速运动演示")
    run_constant_acceleration_demo()

    print("\n" + "="*60)
    print("3. 实时演示")
    real_time_demo()
    
    print("\n" + "="*60)
    print("4. 自适应卡尔曼滤波对比演示")
    run_adaptive_comparison_demo()
