import numpy as np


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


class SimpleAdaptiveKalmanFilter1D:
    def __init__(self, initial_state, initial_covariance, initial_process_noise, 
                 initial_measurement_noise, forget_factor=0.95):
        self.state = np.array(initial_state, dtype=float)
        self.covariance = np.array(initial_covariance, dtype=float)
        self.process_noise = np.array(initial_process_noise, dtype=float)
        self.measurement_noise = initial_measurement_noise
        
        self.dt = 1.0
        self.F = np.array([[1, self.dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.B = np.array([[0.5 * self.dt**2], [self.dt]])
        
        self.forget_factor = forget_factor
        self.innovation_var_estimate = None
        
        self.q_history = []
        self.r_history = []

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
        
        self._adapt_simple(y, S)

    def _adapt_simple(self, residual, S):
        innovation_sq = residual ** 2
        
        if self.innovation_var_estimate is None:
            self.innovation_var_estimate = innovation_sq
        else:
            self.innovation_var_estimate = (self.forget_factor * self.innovation_var_estimate + 
                                          (1 - self.forget_factor) * innovation_sq)
        
        r_adjustment = (self.innovation_var_estimate - S) * 0.1
        self.measurement_noise = max(1e-6, self.measurement_noise + r_adjustment)
        
        if self.innovation_var_estimate > 2 * S:
            self.process_noise = self.process_noise * 1.02
        elif self.innovation_var_estimate < 0.5 * S:
            self.process_noise = self.process_noise * 0.98
        
        self.process_noise = np.maximum(self.process_noise, 1e-9)
        
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


def run_adaptive_demo():
    np.random.seed(42)
    
    print("=" * 70)
    print("自适应卡尔曼滤波演示 - 噪声时变场景")
    print("=" * 70)
    
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
    
    print("\n" + "=" * 70)
    print("方法1: 标准卡尔曼滤波 (Q,R固定)")
    print("=" * 70)
    
    kf_std = AdaptiveKalmanFilter1D(
        initial_state, initial_covariance, initial_process_noise, 
        initial_measurement_noise, adapt_process_noise=False, 
        adapt_measurement_noise=False
    )
    
    std_pos_errors = []
    for z in measurements:
        kf_std.predict()
        kf_std.update(z)
        pos, _ = kf_std.get_state()
        std_pos_errors.append(abs(true_positions[len(std_pos_errors)] - pos))
    
    print(f"平均位置误差: {np.mean(std_pos_errors):.4f}")
    print(f"最终R (固定): {initial_measurement_noise:.4f}")
    
    print("\n" + "=" * 70)
    print("方法2: 自适应卡尔曼滤波 (R在线估计)")
    print("=" * 70)
    
    kf_adaptive = AdaptiveKalmanFilter1D(
        initial_state, initial_covariance, initial_process_noise,
        initial_measurement_noise, window_size=15,
        adapt_process_noise=False, adapt_measurement_noise=True
    )
    
    adaptive_pos_errors = []
    r_estimates = []
    
    for z in measurements:
        kf_adaptive.predict()
        kf_adaptive.update(z)
        pos, _ = kf_adaptive.get_state()
        _, r = kf_adaptive.get_noise_params()
        adaptive_pos_errors.append(abs(true_positions[len(adaptive_pos_errors)] - pos))
        r_estimates.append(r)
    
    print(f"平均位置误差: {np.mean(adaptive_pos_errors):.4f}")
    print(f"最终R (估计): {r_estimates[-1]:.4f}")
    print(f"真实R在最后阶段: {noise_std_profile[-1]**2:.4f}")
    
    print("\n" + "-" * 70)
    print(f"{'阶段':^15} | {'真实噪声方差':^15} | {'估计噪声方差':^15}")
    print("-" * 70)
    print(f"{'前期(0-29)':^15} | {noise_std_profile[0]**2:^15.2f} | {np.mean(r_estimates[0:30]):^15.2f}")
    print(f"{'中期(30-59)':^15} | {noise_std_profile[30]**2:^15.2f} | {np.mean(r_estimates[30:60]):^15.2f}")
    print(f"{'后期(60-99)':^15} | {noise_std_profile[60]**2:^15.2f} | {np.mean(r_estimates[60:]):^15.2f}")
    print("-" * 70)
    
    improvement = (np.mean(std_pos_errors) - np.mean(adaptive_pos_errors)) / np.mean(std_pos_errors) * 100
    print(f"\n自适应滤波相对改进: {improvement:.2f}%")
    
    print("\n" + "=" * 70)
    print("自适应卡尔曼滤波原理说明")
    print("=" * 70)
    print("协方差匹配方法:")
    print("  1. 维护一个残差窗口，计算实际残差方差")
    print("  2. 比较实际残差方差与预测残差方差")
    print("  3. 若实际 > 预测，说明噪声被低估，增大R/Q")
    print("  4. 若实际 < 预测，说明噪声被高估，减小R/Q")
    print("\n关键参数:")
    print("  - window_size: 滑动窗口大小，控制响应速度")
    print("  - forget_factor: 遗忘因子，指数加权平均")
    print("\n适用场景:")
    print("  - 传感器特性随时间变化")
    print("  - 运动模式切换（匀速→加速→转弯）")
    print("  - 环境干扰强度变化")


if __name__ == "__main__":
    run_adaptive_demo()
