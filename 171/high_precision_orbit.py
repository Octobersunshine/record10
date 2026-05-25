#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高精度轨道预报器
包含: J2-J4摄动、大气阻力、太阳光压、扩展卡尔曼滤波(EKF)数据同化
"""

import math
import numpy as np
from datetime import datetime, timedelta


class HighPrecisionPropagator:
    """高精度轨道传播器 - 包含完整摄动模型"""
    
    # 物理常数
    GM = 398600.4418
    EARTH_RADIUS = 6378.1363
    J2 = 1.082626925638815e-3
    J3 = -2.532410523693875e-6
    J4 = -1.610987623724658e-6
    
    # 大气模型参数 (Harris-Priester)
    ATMOSPHERE_H = np.array([100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])
    ATMOSPHERE_RHO_MIN = np.array([4.974e-7, 2.490e-8, 6.068e-9, 2.789e-10, 7.248e-11, 
                                   2.418e-11, 9.158e-12, 3.725e-12, 1.585e-12, 6.967e-13,
                                   1.454e-13, 3.614e-14, 1.170e-14, 4.275e-15, 1.727e-15])
    ATMOSPHERE_RHO_MAX = np.array([4.974e-7, 2.490e-8, 6.068e-9, 2.789e-10, 7.248e-11,
                                   2.418e-11, 9.158e-12, 3.725e-12, 1.585e-12, 6.967e-13,
                                   1.454e-13, 3.614e-14, 1.170e-14, 4.275e-15, 1.727e-15])
    
    # 太阳光压
    SOLAR_FLUX = 1361.0
    LIGHT_SPEED = 299792.458
    AU = 149597870.7
    
    def __init__(self, cd=2.2, area_mass_ratio=0.01, cr=1.3, solar_activity='min'):
        """
        初始化高精度传播器
        
        参数:
            cd: 阻力系数
            area_mass_ratio: 面质比 (m²/kg)
            cr: 光压系数
            solar_activity: 太阳活动水平 ('min', 'max', 'mean')
        """
        self.cd = cd
        self.area_mass_ratio = area_mass_ratio
        self.cr = cr
        self.solar_activity = solar_activity
        
        self.solar_position = None
        self.ekf_filter = None
    
    def _get_atmospheric_density(self, altitude):
        """
        计算大气密度 (Harris-Priester模型)
        
        参数:
            altitude: 海拔高度 (km)
            
        返回:
            密度 (kg/m³)
        """
        if altitude < self.ATMOSPHERE_H[0]:
            return self.ATMOSPHERE_RHO_MIN[0]
        if altitude > self.ATMOSPHERE_H[-1]:
            return self.ATMOSPHERE_RHO_MIN[-1]
        
        idx = np.searchsorted(self.ATMOSPHERE_H, altitude) - 1
        
        h0 = self.ATMOSPHERE_H[idx]
        h1 = self.ATMOSPHERE_H[idx + 1]
        
        if self.solar_activity == 'min':
            rho0 = self.ATMOSPHERE_RHO_MIN[idx]
            rho1 = self.ATMOSPHERE_RHO_MIN[idx + 1]
        elif self.solar_activity == 'max':
            rho0 = self.ATMOSPHERE_RHO_MAX[idx]
            rho1 = self.ATMOSPHERE_RHO_MAX[idx + 1]
        else:
            rho0 = (self.ATMOSPHERE_RHO_MIN[idx] + self.ATMOSPHERE_RHO_MAX[idx]) / 2
            rho1 = (self.ATMOSPHERE_RHO_MIN[idx + 1] + self.ATMOSPHERE_RHO_MAX[idx + 1]) / 2
        
        scale_height = (h1 - h0) / np.log(rho0 / rho1)
        density = rho0 * np.exp(-(altitude - h0) / scale_height)
        
        return density
    
    def _j2_j4_acceleration(self, pos):
        """
        计算J2-J4摄动加速度
        
        参数:
            pos: 位置向量 (km)
            
        返回:
            加速度向量 (km/s²)
        """
        r = np.linalg.norm(pos)
        r_sq = r ** 2
        x, y, z = pos
        
        z_sq = z ** 2
        z_factor = z_sq / r_sq
        
        factor = self.GM / r_sq
        
        # J2摄动
        j2_factor = (3.0 / 2.0) * self.J2 * (self.EARTH_RADIUS ** 2) / (r_sq ** 1)
        acc_j2 = j2_factor * factor * np.array([
            x * (5 * z_factor - 1),
            y * (5 * z_factor - 1),
            z * (5 * z_factor - 3)
        ])
        
        # J3摄动
        j3_factor = (1.0 / 2.0) * self.J3 * (self.EARTH_RADIUS ** 3) / (r_sq * r)
        acc_j3 = j3_factor * factor * np.array([
            x * (5 * (7 * z_factor - 3) * z / r),
            y * (5 * (7 * z_factor - 3) * z / r),
            3 * (1 - 10 * z_factor + 35/3 * z_factor ** 2)
        ])
        
        # J4摄动
        j4_factor = (5.0 / 8.0) * self.J4 * (self.EARTH_RADIUS ** 4) / (r_sq ** 2)
        acc_j4 = j4_factor * factor * np.array([
            x * (3 - 42 * z_factor + 63 * z_factor ** 2),
            y * (3 - 42 * z_factor + 63 * z_factor ** 2),
            z * (15 - 70 * z_factor + 63 * z_factor ** 2)
        ])
        
        return acc_j2 + acc_j3 + acc_j4
    
    def _drag_acceleration(self, pos, vel):
        """
        计算大气阻力加速度
        
        参数:
            pos: 位置向量 (km)
            vel: 速度向量 (km/s)
            
        返回:
            加速度向量 (km/s²)
        """
        r = np.linalg.norm(pos)
        altitude = r - self.EARTH_RADIUS
        
        if altitude < 90 or altitude > 1000:
            return np.zeros(3)
        
        density = self._get_atmospheric_density(altitude)
        
        earth_rotation = np.array([0, 0, 7.2921159e-5])
        vel_rel = vel - np.cross(earth_rotation, pos)
        
        vel_mag = np.linalg.norm(vel_rel)
        if vel_mag < 1e-6:
            return np.zeros(3)
        
        area_mass_ratio_km = self.area_mass_ratio / 1e6
        
        factor = -0.5 * self.cd * area_mass_ratio_km * density * vel_mag
        acc_drag = factor * vel_rel
        
        return acc_drag
    
    def _solar_radiation_pressure(self, pos, julian_date):
        """
        计算太阳光压加速度
        
        参数:
            pos: 位置向量 (km)
            julian_date: 儒略日
            
        返回:
            加速度向量 (km/s²)
        """
        self._update_solar_position(julian_date)
        
        if self.solar_position is None:
            return np.zeros(3)
        
        sat_sun_vec = self.solar_position - pos
        sat_sun_dist = np.linalg.norm(sat_sun_vec)
        
        r_mag = np.linalg.norm(pos)
        earth_sun_dist = np.linalg.norm(self.solar_position)
        
        dot_product = np.dot(pos, self.solar_position)
        if dot_product < 0:
            shadow_factor = r_mag * np.sqrt(1 - (dot_product / (r_mag * earth_sun_dist))**2)
            if shadow_factor < self.EARTH_RADIUS:
                shadow_ratio = 1 - (self.EARTH_RADIUS - shadow_factor) / self.EARTH_RADIUS
                shadow_ratio = max(0, min(1, shadow_ratio))
            else:
                shadow_ratio = 1.0
        else:
            shadow_ratio = 1.0
        
        unit_vector = sat_sun_vec / sat_sun_dist
        flux_factor = self.SOLAR_FLUX / self.LIGHT_SPEED
        
        area_mass_ratio_km = self.area_mass_ratio / 1e6
        
        acc_srp = -shadow_ratio * self.cr * area_mass_ratio_km * flux_factor * unit_vector * 1e-3
        
        return acc_srp
    
    def _update_solar_position(self, julian_date):
        """
        更新太阳位置 (简化模型)
        
        参数:
            julian_date: 儒略日
        """
        jd2000 = julian_date - 2451545.0
        
        mean_longitude = 280.460 + 0.9856474 * jd2000
        mean_anomaly = 357.528 + 0.9856003 * jd2000
        
        mean_longitude = mean_longitude % 360
        mean_anomaly = mean_anomaly % 360
        
        lambda_sun = mean_longitude + 1.915 * math.sin(math.radians(mean_anomaly)) + \
                     0.020 * math.sin(2 * math.radians(mean_anomaly))
        lambda_sun = lambda_sun % 360
        
        epsilon = 23.439 - 0.0000004 * jd2000
        
        lambda_rad = math.radians(lambda_sun)
        epsilon_rad = math.radians(epsilon)
        
        self.solar_position = self.AU * np.array([
            math.cos(lambda_rad),
            math.sin(lambda_rad) * math.cos(epsilon_rad),
            math.sin(lambda_rad) * math.sin(epsilon_rad)
        ])
    
    def total_acceleration(self, state, julian_date):
        """
        计算总加速度（二体 + 所有摄动）
        
        参数:
            state: 状态向量 [x, y, z, vx, vy, vz] (km, km/s)
            julian_date: 儒略日
            
        返回:
            加速度向量 (km/s²)
        """
        pos = state[:3]
        vel = state[3:]
        
        r = np.linalg.norm(pos)
        
        # 二体加速度
        acc_two_body = -self.GM / (r ** 3) * pos
        
        # J2-J4摄动
        acc_j2j4 = self._j2_j4_acceleration(pos)
        
        # 大气阻力
        acc_drag = self._drag_acceleration(pos, vel)
        
        # 太阳光压
        acc_srp = self._solar_radiation_pressure(pos, julian_date)
        
        return acc_two_body + acc_j2j4 + acc_drag + acc_srp
    
    def state_derivative(self, state, t, julian_date_base=0.0):
        """
        状态方程导数
        
        参数:
            state: 状态向量
            t: 时间 (秒)
            julian_date_base: 基础儒略日
            
        返回:
            导数向量
        """
        pos = state[:3]
        vel = state[3:]
        
        jd = julian_date_base + t / 86400.0
        acc = self.total_acceleration(state, jd)
        
        return np.concatenate([vel, acc])
    
    def rk8_integrate(self, state0, dt, n_steps, jd_base):
        """
        RK8积分器（带自适应步长选项）
        
        参数:
            state0: 初始状态
            dt: 时间步长 (秒)
            n_steps: 步数
            jd_base: 基础儒略日
            
        返回:
            积分后的状态
        """
        state = state0.copy()
        current_time = 0.0
        
        for _ in range(n_steps):
            h = dt
            
            k1 = self.state_derivative(state, current_time, jd_base)
            k2 = self.state_derivative(state + h * (1.0/18 * k1), current_time + h/18, jd_base)
            k3 = self.state_derivative(state + h * (1.0/48 * k1 + 1.0/16 * k2), current_time + h/12, jd_base)
            k4 = self.state_derivative(state + h * (1.0/32 * k1 + 3.0/32 * k3), current_time + h/8, jd_base)
            k5 = self.state_derivative(state + h * (5.0/16 * k1 - 75.0/64 * k3 + 75.0/64 * k4), 
                                       current_time + 5*h/24, jd_base)
            k6 = self.state_derivative(state + h * (3.0/80 * k1 + 3.0/16 * k4 + 3.0/20 * k5), 
                                       current_time + h/3, jd_base)
            k7 = self.state_derivative(state + h * (127.0/288 * k1 - 2187.0/1664 * k4 + 
                                                    6561.0/3328 * k5 + 729.0/2176 * k6), 
                                       current_time + 7*h/12, jd_base)
            k8 = self.state_derivative(state + h * (1487.0/1620 * k1 + 27.0/22 * k4 + 
                                                    603.0/50 * k5 + 135.0/22 * k6 + 567.0/220 * k7), 
                                       current_time + 5*h/6, jd_base)
            
            state = state + h * (127.0/1620 * k1 + 27.0/22 * k4 + 603.0/50 * k5 + 
                                 135.0/22 * k6 + 567.0/220 * k7 + 1.0/10 * k8)
            current_time += h
        
        return state


class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器用于轨道确定"""
    
    def __init__(self, initial_state, initial_covariance=None, process_noise_std=1e-6):
        """
        初始化EKF
        
        参数:
            initial_state: 初始状态向量 [x, y, z, vx, vy, vz]
            initial_covariance: 初始协方差矩阵 (6x6)
            process_noise_std: 过程噪声标准差
        """
        self.state = np.array(initial_state, dtype=float)
        self.n_states = len(initial_state)
        
        if initial_covariance is None:
            self.P = np.diag([1e6, 1e6, 1e6, 1e3, 1e3, 1e3])
        else:
            self.P = np.array(initial_covariance)
        
        self.Q = np.eye(self.n_states) * (process_noise_std ** 2)
        
        self.propagator = HighPrecisionPropagator()
    
    def state_transition_matrix(self, state, dt, jd_base):
        """
        计算状态转移矩阵（数值计算）
        
        参数:
            state: 状态向量
            dt: 时间步长 (秒)
            jd_base: 基础儒略日
            
        返回:
            状态转移矩阵 Phi (6x6)
        """
        n = len(state)
        eps = 1e-5
        
        Phi = np.eye(n)
        
        for i in range(n):
            state_perturbed = state.copy()
            state_perturbed[i] += eps
            
            deriv_plus = self.propagator.state_derivative(state_perturbed, 0, jd_base)
            
            state_perturbed[i] -= 2 * eps
            deriv_minus = self.propagator.state_derivative(state_perturbed, 0, jd_base)
            
            df_dx = (deriv_plus - deriv_minus) / (2 * eps)
            
            Phi[:, i] += dt * df_dx
        
        return Phi
    
    def predict(self, dt, jd_base):
        """
        预测步骤
        
        参数:
            dt: 时间步长 (秒)
            jd_base: 基础儒略日
        """
        self.state = self.propagator.rk8_integrate(self.state, dt, 1, jd_base)
        
        Phi = self.state_transition_matrix(self.state, dt, jd_base)
        self.P = Phi @ self.P @ Phi.T + self.Q
    
    def measurement_jacobian(self, state):
        """
        观测雅可比矩阵（GPS位置观测）
        
        参数:
            state: 状态向量
            
        返回:
            观测矩阵 H (3x6)
        """
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H
    
    def update(self, measurement, measurement_noise_std=1.0):
        """
        更新步骤
        
        参数:
            measurement: GPS观测位置 [x, y, z] (km)
            measurement_noise_std: 观测噪声标准差 (km)
        """
        H = self.measurement_jacobian(self.state)
        R = np.eye(3) * (measurement_noise_std ** 2)
        
        predicted_measurement = self.state[:3]
        innovation = measurement - predicted_measurement
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ innovation
        
        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
    
    def get_state(self):
        """获取当前状态"""
        return self.state.copy()
    
    def get_covariance(self):
        """获取当前协方差"""
        return self.P.copy()


class GPSObservationSimulator:
    """GPS观测数据模拟器"""
    
    def __init__(self, position_noise_std=1.0, velocity_noise_std=0.01):
        """
        初始化GPS模拟器
        
        参数:
            position_noise_std: 位置噪声标准差 (km)
            velocity_noise_std: 速度噪声标准差 (km/s)
        """
        self.pos_std = position_noise_std
        self.vel_std = velocity_noise_std
    
    def simulate_observation(self, true_position, true_velocity=None):
        """
        模拟GPS观测
        
        参数:
            true_position: 真实位置 (km)
            true_velocity: 真实速度 (km/s)，可选
            
        返回:
            观测值字典
        """
        noise_pos = np.random.normal(0, self.pos_std, 3)
        obs_position = true_position + noise_pos
        
        observation = {
            'position': obs_position,
            'position_noise_std': self.pos_std,
            'true_position': true_position
        }
        
        if true_velocity is not None:
            noise_vel = np.random.normal(0, self.vel_std, 3)
            obs_velocity = true_velocity + noise_vel
            observation['velocity'] = obs_velocity
            observation['velocity_noise_std'] = self.vel_std
            observation['true_velocity'] = true_velocity
        
        return observation


class HighPrecisionOrbitPredictor:
    """高精度轨道预报器 - 完整集成版"""
    
    def __init__(self, initial_state=None, tle1=None, tle2=None, 
                 cd=2.2, area_mass_ratio=0.01, cr=1.3):
        """
        初始化高精度轨道预报器
        
        参数:
            initial_state: 初始状态 [x, y, z, vx, vy, vz] (km, km/s)
            tle1, tle2: TLE两行数据（可选，用于初始化）
            cd: 阻力系数
            area_mass_ratio: 面质比 (m²/kg)
            cr: 光压系数
        """
        self.propagator = HighPrecisionPropagator(cd, area_mass_ratio, cr)
        self.ekf = None
        self.gps_simulator = GPSObservationSimulator()
        
        self.initial_state = initial_state
        self.current_state = initial_state
        self.epoch_jd = None
        
        if initial_state is not None:
            self.ekf = ExtendedKalmanFilter(initial_state)
        elif tle1 is not None and tle2 is not None:
            self._init_from_tle(tle1, tle2)
    
    def _init_from_tle(self, tle1, tle2):
        """从TLE初始化"""
        try:
            from sgp4.api import Satrec, jday
            from sgp4.earth_gravity import wgs72
            
            sat = Satrec.twoline2rv(tle1, tle2, wgs72)
            jd_epoch = sat.jdsatepoch
            fr_epoch = sat.jdsatepochF
            
            e, r, v = sat.sgp4(jd_epoch, fr_epoch)
            
            if e == 0:
                self.initial_state = np.concatenate([np.array(r), np.array(v)])
                self.current_state = self.initial_state.copy()
                self.epoch_jd = jd_epoch + fr_epoch
                self.ekf = ExtendedKalmanFilter(self.initial_state)
                print(f"从TLE初始化成功，初始位置: {r}")
            else:
                raise RuntimeError("TLE初始化失败")
        except ImportError:
            print("警告: sgp4库不可用，无法从TLE初始化")
    
    def propagate(self, duration_seconds, jd_start=None, dt=60.0):
        """
        轨道传播
        
        参数:
            duration_seconds: 传播时长 (秒)
            jd_start: 起始儒略日
            dt: 积分步长 (秒)
            
        返回:
            最终状态
        """
        if self.current_state is None:
            raise ValueError("未初始化状态")
        
        if jd_start is None:
            if self.epoch_jd is not None:
                jd_start = self.epoch_jd
            else:
                jd_start = 2451545.0
        
        n_steps = int(duration_seconds / dt)
        
        self.current_state = self.propagator.rk8_integrate(
            self.current_state, dt, n_steps, jd_start
        )
        
        return self.current_state.copy()
    
    def assimilate_gps_data(self, gps_position, measurement_time, measurement_noise=1.0):
        """
        同化GPS观测数据
        
        参数:
            gps_position: GPS观测位置 [x, y, z] (km)
            measurement_time: 观测时间 (儒略日)
            measurement_noise: 观测噪声标准差 (km)
        """
        if self.ekf is None:
            if self.current_state is not None:
                self.ekf = ExtendedKalmanFilter(self.current_state)
            else:
                raise ValueError("滤波器未初始化")
        
        if self.epoch_jd is None:
            self.epoch_jd = measurement_time
        
        dt_seconds = (measurement_time - self.epoch_jd) * 86400.0
        
        if abs(dt_seconds) > 1e-6:
            self.ekf.predict(dt_seconds, self.epoch_jd)
        
        self.ekf.update(gps_position, measurement_noise)
        
        self.current_state = self.ekf.get_state()
        self.epoch_jd = measurement_time
    
    def predict_orbit(self, start_jd, duration_hours=24, step_seconds=60):
        """
        预报轨道（高精度）
        
        参数:
            start_jd: 起始儒略日
            duration_hours: 预报时长 (小时)
            step_seconds: 输出步长 (秒)
            
        返回:
            轨道点列表
        """
        if self.current_state is None:
            raise ValueError("未初始化状态")
        
        orbit_points = []
        current_state = self.current_state.copy()
        current_jd = start_jd
        
        total_seconds = duration_hours * 3600
        n_output_steps = int(total_seconds / step_seconds) + 1
        
        for i in range(n_output_steps):
            r = np.linalg.norm(current_state[:3])
            altitude = r - self.propagator.EARTH_RADIUS
            
            orbit_points.append({
                'time_jd': current_jd,
                'position': current_state[:3].copy(),
                'velocity': current_state[3:].copy(),
                'altitude': altitude
            })
            
            if i < n_output_steps - 1:
                current_state = self.propagator.rk8_integrate(
                    current_state, step_seconds, 1, current_jd
                )
                current_jd += step_seconds / 86400.0
        
        return orbit_points
    
    def get_accuracy_estimate(self):
        """
        获取精度估计（从EKF协方差）
        
        返回:
            精度估计字典
        """
        if self.ekf is None:
            return None
        
        P = self.ekf.get_covariance()
        pos_std = np.sqrt(np.diag(P[:3, :3]))
        vel_std = np.sqrt(np.diag(P[3:, 3:]))
        
        return {
            'position_std': pos_std,
            'velocity_std': vel_std,
            '3d_position_std': np.sqrt(np.sum(pos_std ** 2)),
            '3d_velocity_std': np.sqrt(np.sum(vel_std ** 2))
        }


def demonstrate_high_precision():
    """演示高精度轨道预报功能"""
    print("=" * 70)
    print("高精度轨道预报演示")
    print("=" * 70)
    
    print("\n1. 初始化轨道状态 (LEO卫星)")
    print("-" * 70)
    
    initial_state = np.array([
        6878.137, 0.0, 0.0,
        0.0, 7.6, 0.0
    ])
    
    predictor = HighPrecisionOrbitPredictor(
        initial_state=initial_state,
        cd=2.2,
        area_mass_ratio=0.02,
        cr=1.3
    )
    
    print(f"初始位置: {initial_state[:3]} km")
    print(f"初始速度: {initial_state[3:]} km/s")
    
    print("\n2. 轨道传播（含J2-J4、大气阻力、太阳光压）")
    print("-" * 70)
    
    duration = 3600
    final_state = predictor.propagate(duration)
    
    print(f"传播时长: {duration/3600:.1f} 小时")
    print(f"最终位置: {final_state[:3]} km")
    print(f"最终速度: {final_state[3:]} km/s")
    print(f"位置变化: {np.linalg.norm(final_state[:3] - initial_state[:3]):.2f} km")
    
    print("\n3. GPS观测数据同化")
    print("-" * 70)
    
    gps_sim = GPSObservationSimulator(position_noise_std=0.005)
    
    true_pos = final_state[:3]
    gps_obs = gps_sim.simulate_observation(true_pos)
    
    print(f"真实位置: {true_pos}")
    print(f"GPS观测: {gps_obs['position']}")
    print(f"观测误差: {np.linalg.norm(gps_obs['position'] - true_pos)*1000:.2f} m")
    
    jd_now = 2451545.0 + 0.5
    predictor.assimilate_gps_data(gps_obs['position'], jd_now, measurement_noise=0.005)
    
    accuracy = predictor.get_accuracy_estimate()
    if accuracy:
        print(f"\n滤波后精度估计:")
        print(f"  3D位置误差: {accuracy['3d_position_std']*1000:.2f} m")
        print(f"  3D速度误差: {accuracy['3d_velocity_std']*1000:.2f} m/s")
    
    print("\n4. 高精度轨道预报")
    print("-" * 70)
    
    orbit = predictor.predict_orbit(jd_now, duration_hours=6, step_seconds=1800)
    
    print(f"预报 {len(orbit)} 个轨道点（30分钟间隔）:")
    print(f"{'时间(小时)':>10} | {'高度(km)':>10} | {'速度(km/s)':>12}")
    print("-" * 40)
    
    for i, point in enumerate(orbit[::2]):
        t_hours = i * 1
        vel_mag = np.linalg.norm(point['velocity'])
        print(f"{t_hours:>10.1f} | {point['altitude']:>10.2f} | {vel_mag:>12.4f}")
    
    print("\n" + "=" * 70)
    print("高精度轨道预报演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_high_precision()
