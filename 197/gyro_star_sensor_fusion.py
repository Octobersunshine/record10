import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class FilterConfig:
    dt: float = 0.01
    gyro_noise_std: float = 0.005
    gyro_bias_noise_std: float = 0.0001
    measurement_noise_std: float = 0.001
    initial_attitude_std: float = 0.01
    initial_bias_std: float = 0.1
    chi_square_threshold: float = 7.815


@dataclass
class FilterState:
    quaternion: np.ndarray
    gyro_bias: np.ndarray
    P: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    chi_square: float
    is_outlier: bool


class AttitudeKalmanFilter:
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.gyro_bias = np.zeros(3)
        self.P = np.eye(6)
        
        self._initialize_covariance()
        
        self.angular_velocity = np.zeros(3)
        self.last_measurement_valid = True
        self.consecutive_outliers = 0
        self.filter_history: List[FilterState] = []

    def _initialize_covariance(self):
        self.P = np.eye(6)
        self.P[0:3, 0:3] *= self.config.initial_attitude_std ** 2
        self.P[3:6, 3:6] *= self.config.initial_bias_std ** 2

    def predict(self, gyro_measurement: np.ndarray) -> FilterState:
        self.angular_velocity = gyro_measurement - self.gyro_bias
        
        q = self.quaternion
        omega = self.angular_velocity
        
        q_new = self._quaternion_exponential(q, omega, self.config.dt)
        q_new = q_new / np.linalg.norm(q_new)
        self.quaternion = q_new
        
        F = self._compute_state_transition_matrix(q, omega)
        Q = self._compute_process_noise_covariance(omega)
        
        P_new = F @ self.P @ F.T + Q
        P_new = (P_new + P_new.T) / 2
        
        eigvals = np.linalg.eigvalsh(P_new)
        if np.any(eigvals < 0):
            eigvals_fix, eigvecs = np.linalg.eigh(P_new)
            eigvals_fix = np.maximum(eigvals_fix, 1e-12)
            P_new = eigvecs @ np.diag(eigvals_fix) @ eigvecs.T
        
        self.P = P_new
        
        state = FilterState(
            quaternion=self.quaternion.copy(),
            gyro_bias=self.gyro_bias.copy(),
            P=self.P.copy(),
            innovation=np.zeros(3),
            innovation_covariance=np.zeros((3, 3)),
            chi_square=0.0,
            is_outlier=False
        )
        self.filter_history.append(state)
        return state

    def update(self, star_sensor_quaternion: np.ndarray) -> FilterState:
        z = star_sensor_quaternion.copy()
        q_pred = self.quaternion
        
        if np.dot(q_pred, z) < 0:
            z = -z
        
        error_quat = self._quaternion_multiply(
            self._quaternion_conjugate(q_pred), z
        )
        
        if error_quat[0] < 0:
            error_quat = -error_quat
        
        delta_theta = 2.0 * error_quat[1:4]
        
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)
        
        R = np.eye(3) * self.config.measurement_noise_std ** 2
        
        y = delta_theta
        S = H @ self.P @ H.T + R
        
        chi_square = y @ np.linalg.inv(S) @ y
        is_outlier = chi_square > self.config.chi_square_threshold
        
        if is_outlier:
            self.consecutive_outliers += 1
            if self.consecutive_outliers >= 5:
                self._reinitialize_filter(z)
                is_outlier = False
                self.consecutive_outliers = 0
        else:
            self.consecutive_outliers = 0
            K = self.P @ H.T @ np.linalg.inv(S)
            delta_x = K @ y
            
            delta_theta_kf = delta_x[0:3]
            delta_bias = delta_x[3:6]
            
            delta_quat = np.concatenate([[1.0], delta_theta_kf / 2.0])
            delta_quat = delta_quat / np.linalg.norm(delta_quat)
            
            self.quaternion = self._quaternion_multiply(self.quaternion, delta_quat)
            self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
            
            self.gyro_bias = self.gyro_bias + delta_bias
            
            I = np.eye(6)
            P_new = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
            P_new = (P_new + P_new.T) / 2
            
            eigvals = np.linalg.eigvalsh(P_new)
            if np.any(eigvals < 0):
                eigvals_fix, eigvecs = np.linalg.eigh(P_new)
                eigvals_fix = np.maximum(eigvals_fix, 1e-12)
                P_new = eigvecs @ np.diag(eigvals_fix) @ eigvecs.T
            
            self.P = P_new
        
        self.last_measurement_valid = not is_outlier
        
        state = FilterState(
            quaternion=self.quaternion.copy(),
            gyro_bias=self.gyro_bias.copy(),
            P=self.P.copy(),
            innovation=y,
            innovation_covariance=S,
            chi_square=chi_square,
            is_outlier=is_outlier
        )
        if self.filter_history:
            self.filter_history[-1] = state
        return state

    def _compute_state_transition_matrix(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        F = np.eye(6)
        dt = self.config.dt
        
        omega_skew = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-8:
            cos_omega_dt = np.cos(omega_norm * dt)
            sin_omega_dt = np.sin(omega_norm * dt)
            alpha = (1 - cos_omega_dt) / (omega_norm ** 2)
            beta = sin_omega_dt / omega_norm
            gamma = dt / 2.0 - sin_omega_dt / (2 * omega_norm)
            
            phi_rot = cos_omega_dt * np.eye(3) + alpha * np.outer(omega, omega) + beta * omega_skew
            
            bias_input = -dt * np.eye(3) - gamma * omega_skew + \
                        (1 / omega_norm ** 2) * (dt / 2.0 - sin_omega_dt / omega_norm) * np.outer(omega, omega)
        else:
            phi_rot = np.eye(3) + omega_skew * dt + 0.5 * omega_skew @ omega_skew * dt ** 2
            bias_input = -dt * np.eye(3) - dt ** 2 / 2.0 * omega_skew - dt ** 3 / 6.0 * omega_skew @ omega_skew
        
        F[0:3, 0:3] = phi_rot
        F[0:3, 3:6] = bias_input
        
        return F

    def _compute_process_noise_covariance(self, omega: np.ndarray) -> np.ndarray:
        dt = self.config.dt
        sigma_g = self.config.gyro_noise_std
        sigma_bg = self.config.gyro_bias_noise_std
        
        omega_norm = np.linalg.norm(omega)
        maneuver_factor = 1.0 + 10.0 * np.tanh(omega_norm / 2.0)
        
        Q = np.zeros((6, 6))
        
        Q[0:3, 0:3] = sigma_g ** 2 * dt ** 2 * maneuver_factor * np.eye(3)
        Q[3:6, 3:6] = sigma_bg ** 2 * dt * np.eye(3)
        Q[0:3, 3:6] = -0.5 * sigma_g ** 2 * dt ** 3 * maneuver_factor * np.eye(3)
        Q[3:6, 0:3] = Q[0:3, 3:6].T
        
        return Q

    def _reinitialize_filter(self, measurement_quat: np.ndarray):
        self.quaternion = measurement_quat.copy()
        self._initialize_covariance()

    def set_initial_state(self, quaternion: np.ndarray, gyro_bias: np.ndarray = None):
        self.quaternion = quaternion / np.linalg.norm(quaternion)
        if gyro_bias is not None:
            self.gyro_bias = gyro_bias.copy()
        self._initialize_covariance()

    def set_gyro_noise(self, noise_std: float, bias_noise_std: float):
        self.config.gyro_noise_std = noise_std
        self.config.gyro_bias_noise_std = bias_noise_std

    def set_measurement_noise(self, noise_std: float):
        self.config.measurement_noise_std = noise_std

    def get_euler_angles(self) -> Tuple[float, float, float]:
        q0, q1, q2, q3 = self.quaternion
        
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2.0 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    def get_rotation_matrix(self) -> np.ndarray:
        q0, q1, q2, q3 = self.quaternion
        
        R = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        
        return R

    def get_attitude_std(self) -> Tuple[float, float, float]:
        att_std = np.sqrt(np.diag(self.P[0:3, 0:3]))
        return np.degrees(att_std[0]), np.degrees(att_std[1]), np.degrees(att_std[2])

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def _quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quaternion_exponential(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        omega_norm = np.linalg.norm(omega)
        if omega_norm < 1e-8:
            return q
        
        half_angle = 0.5 * omega_norm * dt
        cos_half = np.cos(half_angle)
        sin_half = np.sin(half_angle) / omega_norm
        
        delta_q = np.array([
            cos_half,
            sin_half * omega[0],
            sin_half * omega[1],
            sin_half * omega[2]
        ])
        
        return AttitudeKalmanFilter._quaternion_multiply(q, delta_q)


class UnscentedKalmanFilter(AttitudeKalmanFilter):
    def __init__(self, config: FilterConfig = None, alpha: float = 1e-3, 
                 beta: float = 2.0, kappa: float = 0.0):
        super().__init__(config)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.n = 6
        self.lambda_ = alpha ** 2 * (self.n + kappa) - self.n
        
        self._compute_weights()

    def _compute_weights(self):
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha ** 2 + self.beta)
        
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1.0 / (2 * (self.n + self.lambda_))
            self.Wc[i] = 1.0 / (2 * (self.n + self.lambda_))

    def _generate_sigma_points(self) -> np.ndarray:
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = np.concatenate([np.zeros(3), self.gyro_bias])
        
        P_sym = (self.P + self.P.T) / 2
        P_scaled = (self.n + self.lambda_) * P_sym
        
        try:
            sqrt_matrix = np.linalg.cholesky(P_scaled)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(P_scaled)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_matrix = eigvecs @ np.diag(np.sqrt(eigvals))
        
        for i in range(self.n):
            sigma_points[i + 1] = sigma_points[0] + sqrt_matrix[:, i]
            sigma_points[i + 1 + self.n] = sigma_points[0] - sqrt_matrix[:, i]
        
        return sigma_points

    def predict(self, gyro_measurement: np.ndarray) -> FilterState:
        self.angular_velocity = gyro_measurement - self.gyro_bias
        sigma_points = self._generate_sigma_points()
        
        propagated_quats = []
        propagated_biases = []
        
        for i in range(2 * self.n + 1):
            delta_theta = sigma_points[i, 0:3]
            bias = sigma_points[i, 3:6]
            
            delta_quat = np.concatenate([[1.0], delta_theta / 2.0])
            delta_quat = delta_quat / np.linalg.norm(delta_quat)
            q_sigma = self._quaternion_multiply(self.quaternion, delta_quat)
            
            omega_sigma = gyro_measurement - bias
            q_sigma = self._quaternion_exponential(q_sigma, omega_sigma, self.config.dt)
            
            propagated_quats.append(q_sigma)
            propagated_biases.append(bias)
        
        q_mean = self._quaternion_mean(propagated_quats)
        
        mean_bias = np.zeros(3)
        for i in range(2 * self.n + 1):
            mean_bias += self.Wm[i] * propagated_biases[i]
        
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            error_quat = self._quaternion_multiply(
                self._quaternion_conjugate(q_mean), propagated_quats[i]
            )
            if error_quat[0] < 0:
                error_quat = -error_quat
            delta_theta = 2.0 * error_quat[1:4]
            delta_bias = propagated_biases[i] - mean_bias
            delta_x = np.concatenate([delta_theta, delta_bias])
            P_pred += self.Wc[i] * np.outer(delta_x, delta_x)
        
        Q = self._compute_process_noise_covariance(self.angular_velocity)
        P_pred += Q
        P_pred = (P_pred + P_pred.T) / 2
        
        eigvals = np.linalg.eigvalsh(P_pred)
        if np.any(eigvals < 0):
            eigvals_fix, eigvecs = np.linalg.eigh(P_pred)
            eigvals_fix = np.maximum(eigvals_fix, 1e-12)
            P_pred = eigvecs @ np.diag(eigvals_fix) @ eigvecs.T
        
        self.quaternion = q_mean
        self.gyro_bias = mean_bias
        self.P = P_pred
        
        state = FilterState(
            quaternion=self.quaternion.copy(),
            gyro_bias=self.gyro_bias.copy(),
            P=self.P.copy(),
            innovation=np.zeros(3),
            innovation_covariance=np.zeros((3, 3)),
            chi_square=0.0,
            is_outlier=False
        )
        self.filter_history.append(state)
        return state

    def _quaternion_mean(self, quaternions: List[np.ndarray]) -> np.ndarray:
        q_mean = quaternions[0].copy()
        
        max_iterations = 100
        for _ in range(max_iterations):
            error_vectors = []
            for q in quaternions:
                error_q = self._quaternion_multiply(self._quaternion_conjugate(q_mean), q)
                if error_q[0] < 0:
                    error_q = -error_q
                error_vec = 2.0 * error_q[1:4]
                error_vectors.append(error_vec)
            
            mean_error = np.zeros(3)
            for i in range(len(quaternions)):
                mean_error += self.Wm[i] * error_vectors[i]
            
            error_norm = np.linalg.norm(mean_error)
            if error_norm < 1e-12:
                break
            
            delta_q = np.concatenate([[np.cos(error_norm / 2)], 
                                       np.sin(error_norm / 2) * mean_error / error_norm])
            q_mean = self._quaternion_multiply(q_mean, delta_q)
        
        return q_mean

    def update(self, star_sensor_quaternion: np.ndarray) -> FilterState:
        z = star_sensor_quaternion.copy()
        q_pred = self.quaternion
        
        if np.dot(q_pred, z) < 0:
            z = -z
        
        sigma_points = self._generate_sigma_points()
        
        predicted_measurements = []
        for i in range(2 * self.n + 1):
            delta_theta = sigma_points[i, 0:3]
            bias = sigma_points[i, 3:6]
            
            delta_quat = np.concatenate([[1.0], delta_theta / 2.0])
            delta_quat = delta_quat / np.linalg.norm(delta_quat)
            q_sigma = self._quaternion_multiply(self.quaternion, delta_quat)
            
            error_q = self._quaternion_multiply(self._quaternion_conjugate(q_sigma), z)
            if error_q[0] < 0:
                error_q = -error_q
            predicted_measurements.append(2.0 * error_q[1:4])
        
        y_mean = np.zeros(3)
        for i in range(2 * self.n + 1):
            y_mean += self.Wm[i] * predicted_measurements[i]
        
        Pyy = np.zeros((3, 3))
        Pxy = np.zeros((self.n, 3))
        
        for i in range(2 * self.n + 1):
            dy = predicted_measurements[i] - y_mean
            
            error_quat = self._quaternion_multiply(
                self._quaternion_conjugate(self.quaternion), self.quaternion
            )
            delta_theta = 2.0 * error_quat[1:4]
            delta_bias = sigma_points[i, 3:6] - self.gyro_bias
            dx = np.concatenate([delta_theta, delta_bias])
            
            Pyy += self.Wc[i] * np.outer(dy, dy)
            Pxy += self.Wc[i] * np.outer(dx, dy)
        
        R = np.eye(3) * self.config.measurement_noise_std ** 2
        Pyy += R
        
        K = Pxy @ np.linalg.inv(Pyy)
        
        innovation = np.zeros(3)
        error_q = self._quaternion_multiply(self._quaternion_conjugate(self.quaternion), z)
        if error_q[0] < 0:
            error_q = -error_q
        innovation = 2.0 * error_q[1:4] - y_mean
        
        chi_square = innovation @ np.linalg.inv(Pyy) @ innovation
        is_outlier = chi_square > self.config.chi_square_threshold
        
        if not is_outlier:
            delta_x = K @ innovation
            
            delta_theta = delta_x[0:3]
            delta_bias = delta_x[3:6]
            
            delta_quat = np.concatenate([[1.0], delta_theta / 2.0])
            delta_quat = delta_quat / np.linalg.norm(delta_quat)
            
            self.quaternion = self._quaternion_multiply(self.quaternion, delta_quat)
            self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
            
            self.gyro_bias = self.gyro_bias + delta_bias
            
            P_new = self.P - K @ Pyy @ K.T
            P_new = (P_new + P_new.T) / 2
            
            eigvals = np.linalg.eigvalsh(P_new)
            if np.any(eigvals < 0):
                eigvals_fix, eigvecs = np.linalg.eigh(P_new)
                eigvals_fix = np.maximum(eigvals_fix, 1e-12)
                P_new = eigvecs @ np.diag(eigvals_fix) @ eigvecs.T
            
            self.P = P_new
        
        state = FilterState(
            quaternion=self.quaternion.copy(),
            gyro_bias=self.gyro_bias.copy(),
            P=self.P.copy(),
            innovation=innovation,
            innovation_covariance=Pyy,
            chi_square=chi_square,
            is_outlier=is_outlier
        )
        if self.filter_history:
            self.filter_history[-1] = state
        return state


class ExtendedKalmanFilter(AttitudeKalmanFilter):
    pass


class StarSensorGyroFusion:
    def __init__(self, dt: float = 0.01, use_ukf: bool = False):
        self.config = FilterConfig(dt=dt)
        
        if use_ukf:
            self.filter = UnscentedKalmanFilter(self.config)
        else:
            self.filter = ExtendedKalmanFilter(self.config)
        
        self.dt = dt
        self.star_sensor_update_rate = 10
        self.steps_since_last_update = 0

    def set_star_sensor_update_rate(self, rate: float):
        self.star_sensor_update_rate = rate

    def process_gyro(self, gyro_measurement: np.ndarray) -> FilterState:
        return self.filter.predict(gyro_measurement)

    def process_star_sensor(self, star_sensor_quaternion: np.ndarray) -> FilterState:
        return self.filter.update(star_sensor_quaternion)

    def process(self, gyro_measurement: np.ndarray, 
                star_sensor_quaternion: Optional[np.ndarray] = None) -> FilterState:
        state = self.process_gyro(gyro_measurement)
        
        self.steps_since_last_update += 1
        update_interval = int(1.0 / (self.star_sensor_update_rate * self.dt))
        
        if star_sensor_quaternion is not None and self.steps_since_last_update >= update_interval:
            state = self.process_star_sensor(star_sensor_quaternion)
            self.steps_since_last_update = 0
        
        return state

    def get_attitude(self) -> Tuple[float, float, float]:
        return self.filter.get_euler_angles()

    def get_rotation_matrix(self) -> np.ndarray:
        return self.filter.get_rotation_matrix()

    def get_attitude_std(self) -> Tuple[float, float, float]:
        return self.filter.get_attitude_std()

    def set_initial_attitude(self, quaternion: np.ndarray):
        self.filter.set_initial_state(quaternion)

    def set_gyro_noise_parameters(self, noise_std: float, bias_std: float):
        self.filter.set_gyro_noise(noise_std, bias_std)

    def set_measurement_noise(self, noise_std: float):
        self.filter.set_measurement_noise(noise_std)
