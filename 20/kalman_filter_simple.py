import numpy as np


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


def run_demo():
    np.random.seed(42)
    
    print("=" * 70)
    print("一维卡尔曼滤波演示 - 实时估计位置和速度")
    print("=" * 70)
    
    num_steps = 30
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

    print(f"\n真实速度: {velocity}")
    print(f"观测噪声标准差: {measurement_noise_std}")
    print("\n" + "-" * 70)
    print(f"{'时间步':^8} | {'真实位置':^10} | {'观测值':^10} | {'估计位置':^10} | {'估计速度':^10}")
    print("-" * 70)

    for i, z in enumerate(measurements):
        kf.predict()
        kf.update(z)
        pos, vel = kf.get_state()
        
        if i < 10 or i >= num_steps - 5:
            print(f"{i:^8} | {true_positions[i]:^10.2f} | {z:^10.2f} | {pos:^10.2f} | {vel:^10.2f}")
        elif i == 10:
            print(f"{'...':^8} | {'...':^10} | {'...':^10} | {'...':^10} | {'...':^10}")

    print("-" * 70)
    final_pos_error = abs(true_positions[-1] - pos)
    final_vel_error = abs(velocity - vel)
    print(f"\n最终位置误差: {final_pos_error:.4f}")
    print(f"最终速度误差: {final_vel_error:.4f}")
    
    print("\n" + "=" * 70)
    print("卡尔曼滤波说明:")
    print("=" * 70)
    print("状态向量: [位置, 速度]")
    print("预测步骤: 根据运动模型预测下一时刻的状态")
    print("更新步骤: 使用观测值修正预测状态")
    print("关键参数:")
    print("  - 过程噪声: 模型不确定性")
    print("  - 测量噪声: 传感器不确定性")
    print("  - 卡尔曼增益: 平衡预测和观测的权重")


def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def test_numerical_stability():
    np.random.seed(42)
    
    print("=" * 70)
    print("数值稳定性测试 - 极小观测噪声场景")
    print("=" * 70)
    
    num_steps = 100
    initial_position = 0
    velocity = 2.0
    
    measurement_noise_std = 1e-6
    print(f"\n观测噪声标准差: {measurement_noise_std}")
    print(f"观测噪声方差 R: {measurement_noise_std**2}")
    
    time = np.arange(num_steps)
    true_positions = initial_position + velocity * time
    measurements = true_positions + np.random.normal(0, measurement_noise_std, size=true_positions.shape)
    
    initial_state = [measurements[0], 0]
    initial_covariance = [[100, 0], [0, 100]]
    process_noise = [[0.001, 0], [0, 0.001]]
    measurement_noise = measurement_noise_std**2
    
    kf = KalmanFilter1D(initial_state, initial_covariance, process_noise, measurement_noise)
    
    print("\n" + "-" * 70)
    print(f"{'时间步':^8} | {'P[0,0]':^12} | {'P[1,1]':^12} | {'正定':^8} | {'对称':^8}")
    print("-" * 70)
    
    for i, z in enumerate(measurements):
        kf.predict()
        kf.update(z)
        
        P = kf.covariance
        is_pd = is_positive_definite(P)
        is_symmetric = np.allclose(P, P.T)
        
        if i < 10 or i >= num_steps - 5 or i % 20 == 0:
            print(f"{i:^8} | {P[0,0]:^12.2e} | {P[1,1]:^12.2e} | {'✓' if is_pd else '✗':^8} | {'✓' if is_symmetric else '✗':^8}")
        elif i == 10:
            print(f"{'...':^8} | {'...':^12} | {'...':^12} | {'...':^8} | {'...':^8}")
    
    print("-" * 70)
    print("\n测试完成!")
    print(f"最终P矩阵正定: {'✓ 通过' if is_pd else '✗ 失败'}")
    print(f"最终P矩阵对称: {'✓ 通过' if is_symmetric else '✗ 失败'}")
    
    final_pos, final_vel = kf.get_state()
    print(f"\n最终位置误差: {abs(true_positions[-1] - final_pos):.2e}")
    print(f"最终速度误差: {abs(velocity - final_vel):.2e}")


if __name__ == "__main__":
    run_demo()
    print("\n\n")
    test_numerical_stability()
