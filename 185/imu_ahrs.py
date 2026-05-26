import numpy as np
from typing import Tuple, Optional, List, Dict


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    half_angle = angle / 2.0
    return np.array([
        np.cos(half_angle),
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle)
    ], dtype=np.float64)


def euler_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
    pitch = np.arcsin(np.clip(2.0 * (w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


class ZUPTDetector:
    def __init__(self, 
                 sample_rate: float = 100.0,
                 gyro_threshold: float = 0.05,
                 accel_magnitude_threshold: float = 0.1,
                 accel_variance_threshold: float = 0.02,
                 window_size: int = 5):
        self.dt = 1.0 / sample_rate
        self.gyro_threshold = gyro_threshold
        self.accel_magnitude_threshold = accel_magnitude_threshold
        self.accel_variance_threshold = accel_variance_threshold
        self.window_size = window_size
        
        self.accel_window: List[np.ndarray] = []
        self.gyro_window: List[np.ndarray] = []
        self.gravity = 9.81
        
    def detect(self, gyro: np.ndarray, accel: np.ndarray) -> bool:
        self.gyro_window.append(gyro.copy())
        self.accel_window.append(accel.copy())
        
        if len(self.gyro_window) > self.window_size:
            self.gyro_window.pop(0)
            self.accel_window.pop(0)
        
        if len(self.gyro_window) < self.window_size:
            return False
        
        gyro_arr = np.array(self.gyro_window)
        accel_arr = np.array(self.accel_window)
        
        gyro_norm_mean = np.mean(np.linalg.norm(gyro_arr, axis=1))
        if gyro_norm_mean > self.gyro_threshold:
            return False
        
        accel_magnitude_mean = np.mean(np.abs(
            np.linalg.norm(accel_arr, axis=1) - self.gravity
        ))
        if accel_magnitude_mean > self.accel_magnitude_threshold:
            return False
        
        accel_variance = np.mean(np.var(accel_arr, axis=0))
        if accel_variance > self.accel_variance_threshold:
            return False
        
        return True
    
    def reset(self):
        self.accel_window.clear()
        self.gyro_window.clear()


class ESKF:
    def __init__(self, 
                 sample_rate: float = 100.0,
                 gravity: float = 9.81,
                 initial_position: Optional[np.ndarray] = None):
        self.dt = 1.0 / sample_rate
        self.gravity = gravity
        
        self.position = initial_position if initial_position is not None else np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.bias_gyro = np.zeros(3, dtype=np.float64)
        self.bias_accel = np.zeros(3, dtype=np.float64)
        
        self.P = np.eye(15) * 0.01
        
        self.Q = np.eye(15)
        self.Q[0:3, 0:3] *= 1e-5
        self.Q[3:6, 3:6] *= 1e-4
        self.Q[6:9, 6:9] *= 1e-5
        self.Q[9:12, 9:12] *= 1e-10
        self.Q[12:15, 12:15] *= 1e-6
        
        self.mag_declination = 0.0
        self.mag_inclination = np.deg2rad(60.0)
        self.mag_field_strength = 50.0e-6
        
        self.pressure_ref = 101325.0
        self.temperature_ref = 288.15
        
        self.R_accel = np.eye(3) * 0.1
        self.R_mag = np.eye(3) * 0.01
        self.R_baro = np.array([[100.0]])
        self.R_zupt = np.eye(3) * 0.01
        
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'quaternion': self.quaternion.copy(),
            'bias_gyro': self.bias_gyro.copy(),
            'bias_accel': self.bias_accel.copy()
        }
    
    def get_euler(self) -> np.ndarray:
        return euler_from_quaternion(self.quaternion)
    
    def get_rotation_matrix(self) -> np.ndarray:
        return quaternion_to_rotation_matrix(self.quaternion)
    
    def predict(self, gyro: np.ndarray, accel: np.ndarray) -> None:
        omega = gyro - self.bias_gyro
        
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-10:
            q_delta = quaternion_from_axis_angle(omega, omega_norm * self.dt)
            self.quaternion = quaternion_multiply(self.quaternion, q_delta)
            self.quaternion = quaternion_normalize(self.quaternion)
        
        R = self.get_rotation_matrix()
        accel_world = R @ (accel - self.bias_accel)
        accel_world[2] -= self.gravity
        
        self.position += self.velocity * self.dt + 0.5 * accel_world * self.dt**2
        self.velocity += accel_world * self.dt
        
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * self.dt
        
        R = self.get_rotation_matrix()
        F[3:6, 6:9] = -R @ skew_symmetric(accel - self.bias_accel) * self.dt
        F[3:6, 12:15] = -R * self.dt
        
        omega_skew = skew_symmetric(omega)
        F[6:9, 6:9] = np.eye(3) - omega_skew * self.dt
        F[6:9, 9:12] = -np.eye(3) * self.dt
        
        G = np.zeros((15, 12))
        G[3:6, 0:3] = R * self.dt
        G[6:9, 3:6] = np.eye(3) * self.dt
        G[9:12, 6:9] = np.eye(3) * self.dt
        G[12:15, 9:12] = np.eye(3) * self.dt
        
        Q_i = np.eye(12)
        Q_i[0:3, 0:3] *= 1e-4
        Q_i[3:6, 3:6] *= 1e-4
        Q_i[6:9, 6:9] *= 1e-6
        Q_i[9:12, 9:12] *= 1e-10
        
        self.P = F @ self.P @ F.T + G @ Q_i @ G.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P = np.clip(self.P, -1e6, 1e6)
    
    def _inject_error(self, delta_x: np.ndarray) -> None:
        max_pos = 1.0
        max_vel = 1.0
        max_theta = 0.5
        max_bias_g = 0.01
        max_bias_a = 0.1
        
        delta_x = delta_x.copy()
        delta_x[0:3] = np.clip(delta_x[0:3], -max_pos, max_pos)
        delta_x[3:6] = np.clip(delta_x[3:6], -max_vel, max_vel)
        delta_x[6:9] = np.clip(delta_x[6:9], -max_theta, max_theta)
        delta_x[9:12] = np.clip(delta_x[9:12], -max_bias_g, max_bias_g)
        delta_x[12:15] = np.clip(delta_x[12:15], -max_bias_a, max_bias_a)
        
        self.position += delta_x[0:3]
        self.velocity += delta_x[3:6]
        
        dtheta = delta_x[6:9]
        dtheta_norm = np.linalg.norm(dtheta)
        if dtheta_norm > 1e-10:
            dq = quaternion_from_axis_angle(dtheta, dtheta_norm)
            self.quaternion = quaternion_multiply(self.quaternion, dq)
            self.quaternion = quaternion_normalize(self.quaternion)
        
        self.bias_gyro += delta_x[9:12]
        self.bias_accel += delta_x[12:15]
    
    def _reset_covariance(self) -> None:
        G = np.eye(15)
        G[6:9, 6:9] = np.eye(3) - 0.5 * skew_symmetric(np.zeros(3))
        self.P = G @ self.P @ G.T
    
    def update_accel(self, accel: np.ndarray) -> None:
        R = self.get_rotation_matrix()
        
        g_body = R.T @ np.array([0, 0, self.gravity])
        
        a_pred = g_body + self.bias_accel
        
        H = np.zeros((3, 15))
        H[:, 12:15] = np.eye(3)
        
        R_body = R.T
        H[:, 6:9] = R_body @ skew_symmetric(np.array([0, 0, self.gravity]))
        
        y = accel - a_pred
        
        S = H @ self.P @ H.T + self.R_accel
        
        K = self.P @ H.T @ np.linalg.inv(S)
        
        delta_x = K @ y
        
        self._inject_error(delta_x)
        
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.P = np.clip(self.P, -1e6, 1e6)
        
        self._reset_covariance()
    
    def update_magnetometer(self, mag: np.ndarray) -> None:
        R = self.get_rotation_matrix()
        
        mag_field_enu = np.array([
            self.mag_field_strength * np.cos(self.mag_inclination) * np.cos(self.mag_declination),
            self.mag_field_strength * np.cos(self.mag_inclination) * np.sin(self.mag_declination),
            self.mag_field_strength * np.sin(self.mag_inclination)
        ])
        
        mag_body_pred = R.T @ mag_field_enu
        
        mag_norm = np.linalg.norm(mag)
        if mag_norm < 1e-10:
            return
        
        mag_normalized = mag / mag_norm
        mag_pred_normalized = mag_body_pred / (np.linalg.norm(mag_body_pred) + 1e-10)
        
        H = np.zeros((3, 15))
        mag_skew = skew_symmetric(mag_pred_normalized)
        H[:, 6:9] = -mag_skew
        
        y = mag_normalized - mag_pred_normalized
        
        S = H @ self.P @ H.T + self.R_mag
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        
        delta_x = K @ y
        
        self._inject_error(delta_x)
        
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.P = np.clip(self.P, -1e6, 1e6)
        
        self._reset_covariance()
    
    def update_barometer(self, pressure: float, temperature: Optional[float] = None) -> None:
        if temperature is None:
            temperature = self.temperature_ref
        
        R_gas = 8.31432
        g0 = 9.80665
        M = 0.0289644
        
        height_pred = self.position[2]
        
        ratio = 1 - 0.0065 * height_pred / self.temperature_ref
        if ratio > 0.001:
            pressure_pred = self.pressure_ref * (ratio) ** 5.25588
        else:
            pressure_pred = self.pressure_ref
        
        if pressure_pred < 1000 or pressure_pred > 200000:
            pressure_pred = pressure
        
        H = np.zeros((1, 15))
        H[0, 2] = 1.0
        
        innov = pressure - pressure_pred
        if abs(innov) > 5000:
            innov = np.sign(innov) * 5000
        
        y = np.array([innov])
        
        S = H @ self.P @ H.T + self.R_baro
        
        K = self.P @ H.T @ np.linalg.inv(S)
        
        delta_x = K @ y
        
        self._inject_error(delta_x)
        
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.P = np.clip(self.P, -1e6, 1e6)
        
        self._reset_covariance()
    
    def update_zupt(self) -> None:
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        
        y = -self.velocity
        
        S = H @ self.P @ H.T + self.R_zupt
        
        K = self.P @ H.T @ np.linalg.inv(S)
        
        delta_x = K @ y
        
        self._inject_error(delta_x)
        
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P
        
        self.velocity = np.zeros(3)
        
        self._reset_covariance()
    
    def calibrate(self, 
                gyro_data: np.ndarray, 
                accel_data: np.ndarray,
                mag_data: Optional[np.ndarray] = None,
                num_samples: int = 100):
        self.bias_gyro = np.mean(gyro_data[:num_samples], axis=0)
        
        accel_mean = np.mean(accel_data[:num_samples], axis=0)
        g_dir = accel_mean / np.linalg.norm(accel_mean)
        z_dir = np.array([0.0, 0.0, 1.0])
        
        v = np.cross(g_dir, z_dir)
        s = np.linalg.norm(v)
        c = np.dot(g_dir, z_dir)
        
        if s > 1e-10:
            v_skew = skew_symmetric(v)
            R = np.eye(3) + v_skew + v_skew @ v_skew * ((1 - c) / (s ** 2))
        else:
            R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
        
        self.bias_accel = accel_mean - R.T @ np.array([0, 0, self.gravity])
        
        self.quaternion = self._rotation_to_quaternion(R)
        
        if mag_data is not None:
            mag_mean = np.mean(mag_data[:num_samples], axis=0)
            mag_world = R @ mag_mean
            yaw = np.arctan2(mag_world[1], mag_world[0])
            q_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            self.quaternion = quaternion_multiply(self.quaternion, q_yaw)
            self.quaternion = quaternion_normalize(self.quaternion)
        
        self.P = np.eye(15) * 0.01
        self.P[0:3, 0:3] = np.eye(3) * 0.001
        self.P[3:6, 3:6] = np.eye(3) * 0.001
        self.P[6:9, 6:9] = np.eye(3) * 0.0001
        self.P[9:12, 9:12] = np.eye(3) * 1e-8
        self.P[12:15, 12:15] = np.eye(3) * 1e-6
    
    def _rotation_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        tr = np.trace(R)
        
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        return np.array([w, x, y, z], dtype=np.float64)
    
    def reset(self):
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.bias_gyro = np.zeros(3, dtype=np.float64)
        self.bias_accel = np.zeros(3, dtype=np.float64)
        self.P = np.eye(15) * 0.01


class MultiSensorAHRS:
    def __init__(self, 
                 sample_rate: float = 100.0,
                 gravity: float = 9.81,
                 enable_magnetometer: bool = True,
                 enable_barometer: bool = True,
                 enable_zupt: bool = True):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.gravity = gravity
        
        self.eskf = ESKF(sample_rate=sample_rate, gravity=gravity)
        self.eskf.mag_declination = np.deg2rad(5.0)
        self.eskf.mag_inclination = np.deg2rad(60.0)
        self.eskf.mag_field_strength = 50.0e-6
        self.magnetic_field = np.array([25.0e-6, 5.0e-6, 45.0e-6])
        
        self.enable_magnetometer = enable_magnetometer
        self.enable_barometer = enable_barometer
        self.enable_zupt = enable_zupt
        
        if enable_zupt:
            self.zupt_detector = ZUPTDetector(sample_rate=sample_rate)
        
        self.is_static = False
        self.zupt_count = 0
        
        self.mag_update_interval = int(sample_rate // 10)
        self.baro_update_interval = int(sample_rate // 20)
        self.sample_count = 0
        
        self.mag_bias = np.zeros(3, dtype=np.float64)
        
    def update(self, 
               gyro: np.ndarray, 
               accel: np.ndarray,
               mag: Optional[np.ndarray] = None,
               pressure: Optional[float] = None,
               temperature: Optional[float] = None) -> Dict[str, np.ndarray]:
        
        self.sample_count += 1
        
        self.is_static = False
        if self.enable_zupt:
            self.is_static = self.zupt_detector.detect(gyro, accel)
        
        self.eskf.predict(gyro, accel)
        
        if self.is_static:
            self.eskf.update_accel(accel)
        
        if self.enable_magnetometer and mag is not None:
            if self.sample_count % self.mag_update_interval == 0:
                mag_corrected = mag - self.mag_bias
                original_R_mag = self.eskf.R_mag.copy()
                if not self.is_static:
                    self.eskf.R_mag = np.eye(3) * 0.1
                self.eskf.update_magnetometer(mag_corrected)
                self.eskf.R_mag = original_R_mag
        
        if self.enable_barometer and pressure is not None and self.is_static:
            if self.sample_count % self.baro_update_interval == 0:
                self.eskf.update_barometer(pressure, temperature)
        
        if self.is_static and self.enable_zupt:
            self.eskf.update_zupt()
            self.zupt_count += 1
        
        state = self.eskf.get_state()
        state['is_static'] = np.array([self.is_static])
        state['euler'] = self.eskf.get_euler()
        
        return state
    
    def calibrate(self, 
                  gyro_data: np.ndarray, 
                  accel_data: np.ndarray,
                  mag_data: Optional[np.ndarray] = None,
                  pressure_data: Optional[np.ndarray] = None,
                  num_samples: int = 100):
        self.eskf.calibrate(gyro_data, accel_data, mag_data, num_samples)
        
        if pressure_data is not None:
            self.eskf.pressure_ref = np.mean(pressure_data[:num_samples])
        
        if mag_data is not None:
            self.mag_bias = np.mean(mag_data[:num_samples], axis=0)
            R = self.eskf.get_rotation_matrix()
            mag_world = R @ self.mag_bias
            self.magnetic_field = mag_world / np.linalg.norm(mag_world) * 50e-6
    
    def reset(self):
        self.eskf.reset()
        self.zupt_count = 0
        self.sample_count = 0
        if hasattr(self, 'zupt_detector'):
            self.zupt_detector.reset()


def generate_multisensor_test_data(duration: float = 20.0, 
                                  sample_rate: float = 100.0,
                                  static_samples_initial: int = 200) -> Tuple:
    dt = 1.0 / sample_rate
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) * dt
    
    gyro = np.zeros((num_samples, 3), dtype=np.float64)
    accel = np.zeros((num_samples, 3), dtype=np.float64)
    mag = np.zeros((num_samples, 3), dtype=np.float64)
    pressure = np.zeros(num_samples, dtype=np.float64)
    static_truth = np.zeros(num_samples, dtype=bool)
    true_position = np.zeros((num_samples, 3), dtype=np.float64)
    
    gyro_bias = np.array([0.01, -0.005, 0.008])
    accel_bias = np.array([0.02, 0.01, -0.03])
    mag_bias = np.array([5.0e-6, -3.0e-6, 2.0e-6])
    
    mag_inclination = np.deg2rad(60.0)
    mag_declination = np.deg2rad(5.0)
    mag_field_strength = 50.0e-6
    
    mag_world = np.array([
        mag_field_strength * np.cos(mag_inclination) * np.cos(mag_declination),
        mag_field_strength * np.cos(mag_inclination) * np.sin(mag_declination),
        mag_field_strength * np.sin(mag_inclination)
    ])
    
    segments = [
        {'type': 'static', 'start': 0, 'end': static_samples_initial, 'pos': np.array([0, 0, 0])},
        {'type': 'walk_straight', 'start': static_samples_initial, 'end': int(4 * sample_rate), 
         'speed': 1.0, 'direction': np.array([1, 0, 0]), 'height': 0.0},
        {'type': 'static', 'start': int(4 * sample_rate), 'end': int(5 * sample_rate), 
         'pos': np.array([2, 0, 0])},
        {'type': 'turn', 'start': int(5 * sample_rate), 'end': int(6 * sample_rate), 
         'omega': np.array([0, 0, np.pi/2]), 'pos': np.array([2, 0, 0])},
        {'type': 'walk_straight', 'start': int(6 * sample_rate), 'end': int(9 * sample_rate), 
         'speed': 1.0, 'direction': np.array([0, 1, 0]), 'height': 0.5},
        {'type': 'static', 'start': int(9 * sample_rate), 'end': int(10 * sample_rate), 
         'pos': np.array([2, 3, 0.5])},
        {'type': 'turn', 'start': int(10 * sample_rate), 'end': int(11 * sample_rate), 
         'omega': np.array([0, 0, -np.pi/2]), 'pos': np.array([2, 3, 0.5])},
        {'type': 'walk_straight', 'start': int(11 * sample_rate), 'end': int(14 * sample_rate), 
         'speed': 1.0, 'direction': np.array([-1, 0, 0]), 'height': 1.0},
        {'type': 'static', 'start': int(14 * sample_rate), 'end': int(15 * sample_rate), 
         'pos': np.array([-1, 3, 1.0])},
        {'type': 'walk_straight', 'start': int(15 * sample_rate), 'end': int(20 * sample_rate), 
         'speed': 0.8, 'direction': np.array([0, -1, 0]), 'height': 0.0},
    ]
    
    yaw = 0.0
    current_pos = np.zeros(3)
    
    for seg_idx, seg in enumerate(segments):
        for i in range(seg['start'], min(seg['end'], num_samples)):
            if seg['type'] == 'static':
                static_truth[i] = True
                angular_vel = np.zeros(3)
                linear_accel = np.zeros(3)
                current_pos = seg['pos'].copy()
            
            elif seg['type'] == 'walk_straight':
                static_truth[i] = False
                t_seg = (i - seg['start']) * dt
                
                vel = seg['speed'] * seg['direction']
                pos_start = seg.get('start_pos', segments[seg_idx - 1]['pos'].copy())
                
                current_pos = pos_start + vel * t_seg
                current_pos[2] = seg.get('height', 0.0)
                
                angular_vel = np.zeros(3)
                step_phase = 2 * np.pi * 2.0 * t_seg
                linear_accel = np.array([
                    0.1 * np.sin(step_phase),
                    0.05 * np.sin(step_phase * 2),
                    0.2 * np.abs(np.sin(step_phase))
                ])
            
            elif seg['type'] == 'turn':
                static_truth[i] = False
                t_seg = (i - seg['start']) * dt
                current_pos = seg['pos'].copy()
                angular_vel = seg['omega'].copy()
                yaw += angular_vel[2] * dt
                linear_accel = np.zeros(3)
            
            true_position[i] = current_pos.copy()
            
            R = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            gravity_body = R.T @ np.array([0, 0, 9.81])
            linear_accel_body = R.T @ linear_accel
            mag_body = R.T @ mag_world
            
            gyro[i] = angular_vel + gyro_bias + np.random.randn(3) * 0.005
            accel[i] = gravity_body + linear_accel_body + accel_bias + np.random.randn(3) * 0.03
            mag[i] = mag_body + mag_bias + np.random.randn(3) * 0.5e-6
            
            pressure[i] = 101325.0 * (1 - 0.0065 * current_pos[2] / 288.15) ** 5.25588 + np.random.randn() * 5.0
    
    return t, gyro, accel, mag, pressure, static_truth, true_position, segments


def run_multisensor_comparison():
    import matplotlib
    matplotlib.use('Agg')
    
    sample_rate = 100.0
    duration = 20.0
    
    print("=" * 80)
    print("Multi-Sensor Indoor Localization with ESKF")
    print("=" * 80)
    
    t, gyro, accel, mag, pressure, static_truth, true_position, segments = \
        generate_multisensor_test_data(duration, sample_rate)
    
    configs = [
        {
            "name": "IMU Only (No Correction)",
            "enable_mag": False,
            "enable_baro": False,
            "enable_zupt": False,
            "color": "r",
            "linestyle": "--"
        },
        {
            "name": "IMU + ZUPT",
            "enable_mag": False,
            "enable_baro": False,
            "enable_zupt": True,
            "color": "g",
            "linestyle": "-."
        },
        {
            "name": "IMU + ZUPT + Mag",
            "enable_mag": True,
            "enable_baro": False,
            "enable_zupt": True,
            "color": "m",
            "linestyle": ":"
        },
        {
            "name": "Full ESKF (IMU + ZUPT + Mag + Baro)",
            "enable_mag": True,
            "enable_baro": True,
            "enable_zupt": True,
            "color": "b",
            "linestyle": "-"
        },
    ]
    
    results = {}
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Running: {cfg['name']}")
        print(f"{'='*60}")
        
        ahrs = MultiSensorAHRS(
            sample_rate=sample_rate,
            enable_magnetometer=cfg["enable_mag"],
            enable_barometer=cfg["enable_baro"],
            enable_zupt=cfg["enable_zupt"]
        )
        
        num_samples = len(t)
        positions = np.zeros((num_samples, 3))
        velocities = np.zeros((num_samples, 3))
        eulers = np.zeros((num_samples, 3))
        quaternions = np.zeros((num_samples, 4))
        zupt_detected = np.zeros(num_samples, dtype=bool)
        
        static_samples_init = segments[0]['end']
        ahrs.calibrate(
            gyro[:static_samples_init],
            accel[:static_samples_init],
            mag[:static_samples_init] if cfg["enable_mag"] else None,
            pressure[:static_samples_init] if cfg["enable_baro"] else None,
            num_samples=static_samples_init
        )
        
        print(f"Initial Gyro Bias: {ahrs.eskf.bias_gyro}")
        print(f"Initial Accel Bias: {ahrs.eskf.bias_accel}")
        print(f"True Gyro Bias:   [0.01, -0.005, 0.008]")
        print(f"True Accel Bias:  [0.02, 0.01, -0.03]")
        
        for i in range(num_samples):
            state = ahrs.update(
                gyro[i],
                accel[i],
                mag[i] if cfg["enable_mag"] else None,
                pressure[i] if cfg["enable_baro"] else None
            )
            
            positions[i] = state['position']
            velocities[i] = state['velocity']
            eulers[i] = state['euler']
            quaternions[i] = state['quaternion']
            zupt_detected[i] = state['is_static'][0]
        
        pos_error = positions - true_position
        pos_error_norm = np.linalg.norm(pos_error[:, 0:2], axis=1)
        height_error = np.abs(pos_error[:, 2])
        
        print(f"\nFinal Position: {positions[-1]}")
        print(f"True Position:  {true_position[-1]}")
        print(f"Position Error (XY): {pos_error_norm[-1]:.4f} m")
        print(f"Mean Position Error (XY): {np.mean(pos_error_norm):.4f} m")
        print(f"Max Position Error (XY): {np.max(pos_error_norm):.4f} m")
        print(f"Mean Height Error: {np.mean(height_error):.4f} m")
        
        static_mask = static_truth & (np.arange(num_samples) >= static_samples_init)
        if np.any(static_mask):
            vel_in_static = np.mean(np.linalg.norm(velocities[static_mask], axis=1))
            print(f"Avg Velocity in Static: {vel_in_static:.6f} m/s")
        
        print(f"ZUPT Detections: {ahrs.zupt_count}")
        
        if cfg["enable_mag"]:
            yaw_final = eulers[-1, 2]
            yaw_final = np.arctan2(np.sin(yaw_final), np.cos(yaw_final))
            print(f"Final Yaw: {np.rad2deg(yaw_final):.2f} deg")
        
        results[cfg['name']] = {
            'positions': positions,
            'velocities': velocities,
            'eulers': eulers,
            'quaternions': quaternions,
            'zupt_detected': zupt_detected,
            'pos_error_norm': pos_error_norm,
            'height_error': height_error,
            'color': cfg['color'],
            'linestyle': cfg['linestyle']
        }
    
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for name in [configs[i]['name'] for i in range(len(configs))]:
        res = results[name]
        print(f"\n{name}:")
        print(f"  Mean XY Error: {np.mean(res['pos_error_norm']):.4f} m")
        print(f"  Max XY Error:  {np.max(res['pos_error_norm']):.4f} m")
        print(f"  Mean Height Error: {np.mean(res['height_error']):.4f} m")
    
    improvement = (np.mean(results["IMU Only (No Correction)"]["pos_error_norm"]) - \
                np.mean(results["Full ESKF (IMU + ZUPT + Mag + Baro)"]["pos_error_norm"]))
    improvement_pct = improvement / np.mean(results["IMU Only (No Correction)"]["pos_error_norm"]) * 100
    print(f"\nFull ESKF vs IMU Only Improvement: {improvement:.4f} m ({improvement_pct:.2f}%)")
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        
        axes[0, 0].plot(true_position[:, 0], true_position[:, 1], 'k-', label='Ground Truth', linewidth=2)
        for name, res in results.items():
            axes[0, 0].plot(res['positions'][:, 0], res['positions'][:, 1], 
                            color=res['color'], linestyle=res['linestyle'],
                            label=name, alpha=0.8)
        axes[0, 0].set_title('XY Trajectory')
        axes[0, 0].set_xlabel('X Position (m)')
        axes[0, 0].set_ylabel('Y Position (m)')
        axes[0, 0].grid(True)
        axes[0, 0].axis('equal')
        axes[0, 0].legend()
        
        for name, res in results.items():
            axes[0, 1].plot(t, res['pos_error_norm'], 
                            color=res['color'], linestyle=res['linestyle'],
                            label=name)
        axes[0, 1].set_title('XY Position Error')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Error (m)')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        axes[1, 0].plot(t, true_position[:, 2], 'k-', label='Ground Truth', linewidth=2)
        for name, res in results.items():
            axes[1, 0].plot(t, res['positions'][:, 2], 
                            color=res['color'], linestyle=res['linestyle'],
                            label=name)
        axes[1, 0].set_title('Height (Z Position)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Height (m)')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        for name, res in results.items():
            axes[1, 1].plot(t, np.rad2deg(res['eulers'][:, 2]),
                            color=res['color'], linestyle=res['linestyle'],
                            label=name)
        axes[1, 1].set_title('Yaw Angle')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Yaw (deg)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        for name, res in results.items():
            axes[2, 0].plot(t, np.linalg.norm(res['velocities'], axis=1),
                            color=res['color'], linestyle=res['linestyle'],
                            label=name)
        axes[2, 0].set_title('Velocity Norm')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Velocity (m/s)')
        axes[2, 0].grid(True)
        axes[2, 0].legend()
        
        for seg in segments:
            if seg['type'] == 'static':
                axes[2, 0].axvspan(seg['start']/sample_rate, seg['end']/sample_rate,
                               alpha=0.2, color='green')
        
        bar_width = 0.35
        names = list(results.keys())
        x_pos = np.arange(len(names))
        mean_errors = [np.mean(results[name]['pos_error_norm']) for name in names]
        max_errors = [np.max(results[name]['pos_error_norm']) for name in names]
        
        axes[2, 1].bar(x_pos - bar_width/2, mean_errors, bar_width, label='Mean XY Error', alpha=0.7)
        axes[2, 1].bar(x_pos + bar_width/2, max_errors, bar_width, label='Max XY Error', alpha=0.7)
        axes[2, 1].set_title('Position Error Comparison')
        axes[2, 1].set_ylabel('Error (m)')
        axes[2, 1].set_xticks(x_pos)
        axes[2, 1].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
        axes[2, 1].grid(True)
        axes[2, 1].legend()
        
        plt.tight_layout()
        plt.savefig('e:/temp/record10/185/multisensor_results.png', dpi=150)
        print("\nPlot saved to multisensor_results.png")
        
        fig2, ax = plt.subplots(figsize=(12, 10))
        ax.plot(true_position[:, 0], true_position[:, 1], 'k-', label='Ground Truth', linewidth=3)
        full_eskf = results["Full ESKF (IMU + ZUPT + Mag + Baro)"]
        ax.plot(full_eskf['positions'][:, 0], full_eskf['positions'][:, 1], 
                'b-', label='Full ESKF', linewidth=2, alpha=0.9)
        imu_only = results["IMU Only (No Correction)"]
        ax.plot(imu_only['positions'][:, 0], imu_only['positions'][:, 1], 
                'r--', label='IMU Only', linewidth=1.5, alpha=0.7)
        
        for seg in segments:
            if seg['type'] == 'static':
                ax.plot(seg['pos'][0], seg['pos'][1], 'go', markersize=10,
                       label='Static Point' if seg['start'] == segments[2]['start'] else "")
        
        ax.set_title('Indoor Localization Trajectory Comparison')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        plt.savefig('e:/temp/record10/185/trajectory_final.png', dpi=150)
        print("Trajectory plot saved to trajectory_final.png")
        
    except ImportError:
        print("\nmatplotlib not installed, skipping plots")
    
    return results


if __name__ == "__main__":
    run_multisensor_comparison()
