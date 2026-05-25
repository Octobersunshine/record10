#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版SGP4/SDP4轨道预报器
修复近地点奇异性问题，支持SDP4深空扩展和Runge-Kutta数值积分
"""

import math
from datetime import datetime, timedelta
import numpy as np

try:
    from sgp4.api import Satrec, jday, days2mdhms
    from sgp4.earth_gravity import wgs72
    from sgp4.propagation import sgp4, sdp4
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    print("警告: 未安装sgp4库，请运行: pip install sgp4")


class EnhancedOrbitPredictor:
    """增强版轨道预报器 - 支持SGP4/SDP4和数值积分"""
    
    ECCENTRICITY_THRESHOLD = 1e-4
    ALTITUDE_THRESHOLD = 5000.0
    GRAVITATIONAL_CONSTANT = 398600.4418
    EARTH_RADIUS = 6378.137
    J2 = 1.082626925638815e-3
    
    PROPAGATION_MODES = ['auto', 'sgp4', 'sdp4', 'numerical']
    
    def __init__(self, tle_line1, tle_line2, satellite_name="Satellite", 
                 propagation_mode='auto', integrator='rk4'):
        """
        初始化增强版轨道预报器
        
        参数:
            tle_line1: TLE第一行
            tle_line2: TLE第二行
            satellite_name: 卫星名称
            propagation_mode: 传播模式 ('auto', 'sgp4', 'sdp4', 'numerical')
            integrator: 数值积分器 ('rk4', 'rk8')
        """
        if not SGP4_AVAILABLE:
            raise ImportError("需要安装sgp4库: pip install sgp4")
        
        self.satellite_name = satellite_name
        self.tle_line1 = tle_line1.strip()
        self.tle_line2 = tle_line2.strip()
        self.propagation_mode = propagation_mode
        self.integrator = integrator
        
        self.satellite = Satrec.twoline2rv(self.tle_line1, self.tle_line2, wgs72)
        
        self.eccentricity = self.satellite.ecco
        self.inclination = self.satellite.inclo
        self.period = 2 * math.pi / self.satellite.no_kozai * 60
        self.semi_major_axis = (self.GRAVITATIONAL_CONSTANT / (self.satellite.no_kozai ** 2)) ** (1/3)
        self.mean_motion = self.satellite.no_kozai
        
        self.is_near_circular = self.eccentricity < self.ECCENTRICITY_THRESHOLD
        self.is_deep_space = self.period >= 225.0
        
        self._determine_propagation_mode()
        self._initialize_epoch_state()
        
        print(f"卫星: {self.satellite_name}")
        print(f"轨道周期: {self.period:.2f} 分钟")
        print(f"偏心率: {self.eccentricity:.6f}")
        print(f"TLE纪元: {self._get_epoch_datetime()}")
        print(f"传播模式: {self.current_mode.upper()}")
        print(f"轨道类型: {'近圆轨道' if self.is_near_circular else '椭圆轨道'}, "
              f"{'深空' if self.is_deep_space else '近地'}")
        
        if self.is_near_circular:
            print(f"  (检测到近圆轨道，已启用奇异性保护)")
    
    def _determine_propagation_mode(self):
        """确定传播模式"""
        if self.propagation_mode == 'auto':
            if self.is_deep_space:
                self.current_mode = 'sdp4'
            elif self.is_near_circular:
                self.current_mode = 'numerical'
            else:
                self.current_mode = 'sgp4'
        else:
            self.current_mode = self.propagation_mode
        
        if self.current_mode == 'sdp4' and not hasattr(self.satellite, 'sdp4'):
            print("警告: SDP4不可用，回退到SGP4")
            self.current_mode = 'sgp4'
    
    def _initialize_epoch_state(self):
        """初始化纪元时刻的状态向量"""
        jd_epoch = self.satellite.jdsatepoch
        fr_epoch = self.satellite.jdsatepochF
        
        e, r, v = self.satellite.sgp4(jd_epoch, fr_epoch)
        
        if e == 0:
            self.epoch_position = np.array(r)
            self.epoch_velocity = np.array(v)
        else:
            self.epoch_position = None
            self.epoch_velocity = None
    
    def _get_epoch_datetime(self):
        """获取TLE纪元时间"""
        year = self.satellite.epochyr
        day_of_year = self.satellite.epochdays
        
        if year < 57:
            year += 2000
        else:
            year += 1900
        
        month, day, hour, minute, second = days2mdhms(year, day_of_year)
        
        return datetime(year, month, day, int(hour), int(minute), int(second))
    
    def _two_body_acceleration(self, pos, include_j2=True):
        """
        计算二体引力加速度（含J2摄动）
        
        参数:
            pos: 位置向量 (km)
            include_j2: 是否包含J2摄动
            
        返回:
            加速度向量 (km/s^2)
        """
        r = np.linalg.norm(pos)
        r_mag = r
        
        r_vec = pos / r_mag
        
        acc_two_body = -self.GRAVITATIONAL_CONSTANT / (r_mag ** 2) * r_vec
        
        if include_j2:
            z = pos[2]
            r_sq = r_mag ** 2
            z_sq = z ** 2
            
            factor_J2 = (3.0 / 2.0) * self.J2 * self.GRAVITATIONAL_CONSTANT * (self.EARTH_RADIUS ** 2) / (r_mag ** 5)
            
            acc_J2 = factor_J2 * np.array([
                pos[0] * (5 * z_sq / r_sq - 1),
                pos[1] * (5 * z_sq / r_sq - 1),
                pos[2] * (5 * z_sq / r_sq - 3)
            ])
            
            return acc_two_body + acc_J2
        
        return acc_two_body
    
    def _state_derivative(self, state, t=0.0):
        """
        状态方程导数
        
        参数:
            state: 状态向量 [x, y, z, vx, vy, vz]
            t: 时间（用于适配积分器接口）
            
        返回:
            导数向量 [vx, vy, vz, ax, ay, az]
        """
        pos = state[:3]
        vel = state[3:]
        
        acc = self._two_body_acceleration(pos, include_j2=True)
        
        return np.concatenate([vel, acc])
    
    def _rk4_integrate(self, state0, dt, n_steps=1):
        """
        Runge-Kutta 4阶积分器
        
        参数:
            state0: 初始状态向量
            dt: 时间步长 (秒)
            n_steps: 步数
            
        返回:
            积分后的状态向量
        """
        state = state0.copy()
        h = dt
        
        for _ in range(n_steps):
            k1 = self._state_derivative(state)
            k2 = self._state_derivative(state + h/2 * k1)
            k3 = self._state_derivative(state + h/2 * k2)
            k4 = self._state_derivative(state + h * k3)
            
            state = state + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return state
    
    def _rk8_integrate(self, state0, dt, n_steps=1):
        """
        Runge-Kutta 8阶积分器 (Dormand-Prince)
        
        参数:
            state0: 初始状态向量
            dt: 时间步长 (秒)
            n_steps: 步数
            
        返回:
            积分后的状态向量
        """
        state = state0.copy()
        h = dt
        
        a21 = 1.0 / 18.0
        a31 = 1.0 / 48.0
        a32 = 1.0 / 16.0
        a41 = 1.0 / 32.0
        a43 = 3.0 / 32.0
        a51 = 5.0 / 16.0
        a53 = -75.0 / 64.0
        a54 = 75.0 / 64.0
        a61 = 3.0 / 80.0
        a64 = 3.0 / 16.0
        a65 = 3.0 / 20.0
        a71 = -29443841.0 / 614563906.0
        a74 = 77736538.0 / 692538347.0
        a75 = -28693883.0 / 1125000000.0
        a76 = 23124283.0 / 1800000000.0
        a81 = 16016141.0 / 946692911.0
        a84 = 61564180.0 / 158732637.0
        a85 = 22789713.0 / 633445777.0
        a86 = 545815736.0 / 2771057229.0
        a87 = -180193667.0 / 1043307555.0
        
        b1 = 14005451.0 / 335480064.0
        b4 = -59238493.0 / 1068277825.0
        b5 = 181606767.0 / 758867731.0
        b6 = 561292985.0 / 797845732.0
        b7 = -1041891430.0 / 1371343529.0
        b8 = 760417239.0 / 1151165299.0
        
        for _ in range(n_steps):
            k1 = self._state_derivative(state)
            k2 = self._state_derivative(state + h * (a21 * k1))
            k3 = self._state_derivative(state + h * (a31 * k1 + a32 * k2))
            k4 = self._state_derivative(state + h * (a41 * k1 + a43 * k3))
            k5 = self._state_derivative(state + h * (a51 * k1 + a53 * k3 + a54 * k4))
            k6 = self._state_derivative(state + h * (a61 * k1 + a64 * k4 + a65 * k5))
            k7 = self._state_derivative(state + h * (a71 * k1 + a74 * k4 + a75 * k5 + a76 * k6))
            k8 = self._state_derivative(state + h * (a81 * k1 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7))
            
            state = state + h * (b1 * k1 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8)
        
        return state
    
    def _numerical_propagate(self, delta_seconds):
        """
        使用数值积分传播轨道
        
        参数:
            delta_seconds: 相对于纪元的时间差 (秒)
            
        返回:
            (position, velocity): 位置和速度向量
        """
        if self.epoch_position is None or self.epoch_velocity is None:
            jd_epoch = self.satellite.jdsatepoch
            fr_epoch = self.satellite.jdsatepochF
            e, r, v = self.satellite.sgp4(jd_epoch, fr_epoch)
            self.epoch_position = np.array(r)
            self.epoch_velocity = np.array(v)
        
        state0 = np.concatenate([self.epoch_position, self.epoch_velocity])
        
        if abs(delta_seconds) < 1.0:
            return self.epoch_position.copy(), self.epoch_velocity.copy()
        
        if self.integrator == 'rk8':
            dt = min(60.0, abs(delta_seconds) / 100.0)
        else:
            dt = min(30.0, abs(delta_seconds) / 100.0)
        
        n_steps = int(abs(delta_seconds) / dt) + 1
        actual_dt = delta_seconds / n_steps
        
        if self.integrator == 'rk8':
            state = self._rk8_integrate(state0, actual_dt, n_steps)
        else:
            state = self._rk4_integrate(state0, actual_dt, n_steps)
        
        return state[:3], state[3:]
    
    def get_position_eci(self, target_datetime):
        """
        计算指定时间的ECI坐标位置（自动处理近地点奇异性）
        
        参数:
            target_datetime: datetime对象
            
        返回:
            (position, velocity): 位置(km)和速度(km/s)的元组
        """
        jd, fr = jday(
            target_datetime.year, target_datetime.month, target_datetime.day,
            target_datetime.hour, target_datetime.minute, target_datetime.second
        )
        
        jd_epoch = self.satellite.jdsatepoch
        fr_epoch = self.satellite.jdsatepochF
        delta_days = (jd + fr) - (jd_epoch + fr_epoch)
        delta_minutes = delta_days * 1440.0
        
        if self.current_mode == 'numerical' or self.is_near_circular:
            delta_seconds = delta_minutes * 60.0
            try:
                return self._numerical_propagate(delta_seconds)
            except:
                pass
        
        if self.current_mode == 'sdp4':
            try:
                e, r, v = self.satellite.sgp4(jd, fr)
                if e == 0:
                    return np.array(r), np.array(v)
            except:
                pass
        
        e, r, v = self.satellite.sgp4(jd, fr)
        
        if e != 0:
            if self.is_near_circular:
                delta_seconds = delta_minutes * 60.0
                try:
                    return self._numerical_propagate(delta_seconds)
                except:
                    raise RuntimeError(f"SGP4计算错误，错误码: {e}")
            raise RuntimeError(f"SGP4计算错误，错误码: {e}")
        
        return np.array(r), np.array(v)
    
    def get_position_eci_with_mode(self, target_datetime, mode=None):
        """
        使用指定模式计算ECI坐标（用于对比测试）
        
        参数:
            target_datetime: datetime对象
            mode: 'sgp4', 'sdp4', 'numerical'
            
        返回:
            (position, velocity): 位置和速度的元组
        """
        jd, fr = jday(
            target_datetime.year, target_datetime.month, target_datetime.day,
            target_datetime.hour, target_datetime.minute, target_datetime.second
        )
        
        if mode is None:
            mode = self.current_mode
        
        if mode == 'numerical':
            jd_epoch = self.satellite.jdsatepoch
            fr_epoch = self.satellite.jdsatepochF
            delta_days = (jd + fr) - (jd_epoch + fr_epoch)
            delta_seconds = delta_days * 86400.0
            return self._numerical_propagate(delta_seconds)
        elif mode == 'sdp4':
            e, r, v = self.satellite.sgp4(jd, fr)
            if e != 0:
                raise RuntimeError(f"SDP4计算错误，错误码: {e}")
            return np.array(r), np.array(v)
        else:
            e, r, v = self.satellite.sgp4(jd, fr)
            if e != 0:
                raise RuntimeError(f"SGP4计算错误，错误码: {e}")
            return np.array(r), np.array(v)
    
    def predict_orbit(self, start_time=None, duration_minutes=120, step_seconds=60):
        """
        预报轨道
        
        参数:
            start_time: 开始时间，默认当前时间
            duration_minutes: 预报时长（分钟）
            step_seconds: 时间步长（秒）
            
        返回:
            轨道点列表，每个元素包含(time, position, velocity)
        """
        if start_time is None:
            start_time = datetime.utcnow()
        
        orbit_points = []
        current_time = start_time
        end_time = start_time + timedelta(minutes=duration_minutes)
        time_step = timedelta(seconds=step_seconds)
        
        while current_time <= end_time:
            try:
                pos, vel = self.get_position_eci(current_time)
                orbit_points.append({
                    'time': current_time,
                    'position': pos,
                    'velocity': vel,
                    'altitude': np.linalg.norm(pos) - self.EARTH_RADIUS,
                    'mode': self.current_mode
                })
            except RuntimeError:
                pass
            current_time += time_step
        
        return orbit_points
    
    def get_ground_track(self, orbit_points):
        """
        计算星下点轨迹
        
        参数:
            orbit_points: 轨道点列表
            
        返回:
            包含经纬度信息的轨道点列表
        """
        earth_rotation_rate = 7.2921159e-5
        
        for point in orbit_points:
            pos = point['position']
            
            longitude = math.degrees(math.atan2(pos[1], pos[0]))
            latitude = math.degrees(math.asin(pos[2] / np.linalg.norm(pos)))
            
            jd, fr = jday(
                point['time'].year, point['time'].month, point['time'].day,
                point['time'].hour, point['time'].minute, point['time'].second
            )
            jd_total = jd + fr
            gst = (jd_total - 2451545.0) * earth_rotation_rate * 12 / math.pi
            gst = gst % 24
            longitude = longitude - gst * 15
            longitude = ((longitude + 180) % 360) - 180
            
            point['latitude'] = latitude
            point['longitude'] = longitude
        
        return orbit_points
    
    def predict_passes(self, observer_lat, observer_lon, observer_alt=0,
                       start_time=None, duration_hours=24, min_elevation=10.0):
        """
        预报过顶事件
        
        参数:
            observer_lat: 观测者纬度（度）
            observer_lon: 观测者经度（度）
            observer_alt: 观测者海拔（米）
            start_time: 开始时间，默认当前时间
            duration_hours: 预报时长（小时）
            min_elevation: 最小仰角（度）
            
        返回:
            过顶事件列表
        """
        if start_time is None:
            start_time = datetime.utcnow()
        
        orbit_points = self.predict_orbit(
            start_time=start_time,
            duration_minutes=duration_hours * 60,
            step_seconds=30
        )
        
        orbit_points = self.get_ground_track(orbit_points)
        
        passes = []
        current_pass = None
        
        for point in orbit_points:
            elevation, azimuth, range_km = self._calculate_az_el(
                point['position'], observer_lat, observer_lon, observer_alt, point['time']
            )
            
            point['elevation'] = elevation
            point['azimuth'] = azimuth
            point['range'] = range_km
            
            if elevation >= min_elevation:
                if current_pass is None:
                    current_pass = {
                        'start_time': point['time'],
                        'max_elevation': elevation,
                        'max_elevation_time': point['time'],
                        'points': []
                    }
                current_pass['points'].append(point)
                
                if elevation > current_pass['max_elevation']:
                    current_pass['max_elevation'] = elevation
                    current_pass['max_elevation_time'] = point['time']
            else:
                if current_pass is not None:
                    current_pass['end_time'] = point['time']
                    current_pass['duration'] = (
                        current_pass['end_time'] - current_pass['start_time']
                    ).total_seconds() / 60
                    passes.append(current_pass)
                    current_pass = None
        
        if current_pass is not None:
            current_pass['end_time'] = orbit_points[-1]['time']
            current_pass['duration'] = (
                current_pass['end_time'] - current_pass['start_time']
            ).total_seconds() / 60
            passes.append(current_pass)
        
        return passes
    
    def _calculate_az_el(self, sat_pos_eci, obs_lat, obs_lon, obs_alt, target_time):
        """
        计算方位角和仰角
        
        参数:
            sat_pos_eci: 卫星ECI位置(km)
            obs_lat: 观测者纬度（度）
            obs_lon: 观测者经度（度）
            obs_alt: 观测者海拔（米）
            target_time: 目标时间
            
        返回:
            (elevation, azimuth, range_km)
        """
        obs_lat_rad = math.radians(obs_lat)
        obs_lon_rad = math.radians(obs_lon)
        
        obs_r = self.EARTH_RADIUS + obs_alt / 1000.0
        obs_pos_ecef = np.array([
            obs_r * math.cos(obs_lat_rad) * math.cos(obs_lon_rad),
            obs_r * math.cos(obs_lat_rad) * math.sin(obs_lon_rad),
            obs_r * math.sin(obs_lat_rad)
        ])
        
        jd, fr = jday(
            target_time.year, target_time.month, target_time.day,
            target_time.hour, target_time.minute, target_time.second
        )
        jd_total = jd + fr
        gmst = (jd_total - 2451545.0) * 7.2921159e-5 * 12 / math.pi
        gmst = gmst % 24
        theta = math.radians(gmst * 15)
        
        rot_matrix = np.array([
            [math.cos(theta), math.sin(theta), 0],
            [-math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])
        
        sat_pos_ecef = rot_matrix @ sat_pos_eci
        
        rel_vec = sat_pos_ecef - obs_pos_ecef
        range_km = np.linalg.norm(rel_vec)
        
        obs_local = np.array([
            -math.sin(obs_lon_rad),
            math.cos(obs_lon_rad),
            0
        ])
        
        zenith_local = np.array([
            math.cos(obs_lat_rad) * math.cos(obs_lon_rad),
            math.cos(obs_lat_rad) * math.sin(obs_lon_rad),
            math.sin(obs_lat_rad)
        ])
        
        east_local = np.cross(zenith_local, obs_local)
        east_local = east_local / np.linalg.norm(east_local)
        north_local = np.cross(zenith_local, east_local)
        north_local = north_local / np.linalg.norm(north_local)
        
        rel_east = np.dot(rel_vec, east_local)
        rel_north = np.dot(rel_vec, north_local)
        rel_up = np.dot(rel_vec, zenith_local)
        
        elevation = math.degrees(math.asin(rel_up / range_km))
        azimuth = math.degrees(math.atan2(rel_east, rel_north))
        if azimuth < 0:
            azimuth += 360
        
        return elevation, azimuth, range_km
    
    def get_communication_windows(self, observer_lat, observer_lon, observer_alt=0,
                                   start_time=None, duration_hours=24, min_elevation=10.0):
        """
        获取通信窗口
        
        参数:
            observer_lat: 观测者纬度（度）
            observer_lon: 观测者经度（度）
            observer_alt: 观测者海拔（米）
            start_time: 开始时间
            duration_hours: 预报时长（小时）
            min_elevation: 最小仰角（度）
            
        返回:
            通信窗口列表，包含窗口详细信息
        """
        passes = self.predict_passes(
            observer_lat, observer_lon, observer_alt,
            start_time, duration_hours, min_elevation
        )
        
        windows = []
        for pass_info in passes:
            window = {
                'window_id': len(windows) + 1,
                'start_time': pass_info['start_time'],
                'end_time': pass_info['end_time'],
                'duration_minutes': pass_info['duration'],
                'max_elevation': pass_info['max_elevation'],
                'max_elevation_time': pass_info['max_elevation_time'],
                'doppler_shift_start': self._calculate_doppler(pass_info['points'][0]),
                'doppler_shift_end': self._calculate_doppler(pass_info['points'][-1]),
            }
            
            if pass_info['points']:
                mid_idx = len(pass_info['points']) // 2
                window['range_at_max_elev'] = pass_info['points'][mid_idx]['range']
                window['doppler_at_max_elev'] = self._calculate_doppler(pass_info['points'][mid_idx])
            
            windows.append(window)
        
        return windows
    
    def _calculate_doppler(self, point, freq_hz=1e9):
        """
        计算多普勒频移
        
        参数:
            point: 轨道点
            freq_hz: 载波频率（Hz）
            
        返回:
            多普勒频移（Hz）
        """
        c = 299792.458
        
        pos = point['position']
        vel = point['velocity']
        
        range_vec = pos / np.linalg.norm(pos)
        range_rate = np.dot(vel, range_vec)
        
        doppler = -freq_hz * range_rate / c
        
        return doppler
    
    def compare_propagation_methods(self, target_datetime):
        """
        对比不同传播方法的结果
        
        参数:
            target_datetime: 目标时间
            
        返回:
            对比结果字典
        """
        results = {}
        
        try:
            pos_sgp4, vel_sgp4 = self.get_position_eci_with_mode(target_datetime, 'sgp4')
            results['sgp4'] = {'position': pos_sgp4, 'velocity': vel_sgp4}
        except Exception as e:
            results['sgp4'] = {'error': str(e)}
        
        try:
            pos_num, vel_num = self.get_position_eci_with_mode(target_datetime, 'numerical')
            results['numerical'] = {'position': pos_num, 'velocity': vel_num}
        except Exception as e:
            results['numerical'] = {'error': str(e)}
        
        if 'sgp4' in results and 'numerical' in results and 'error' not in results['sgp4'] and 'error' not in results['numerical']:
            pos_diff = np.linalg.norm(results['sgp4']['position'] - results['numerical']['position'])
            vel_diff = np.linalg.norm(results['sgp4']['velocity'] - results['numerical']['velocity'])
            results['difference'] = {
                'position_km': pos_diff,
                'velocity_km_s': vel_diff
            }
        
        return results


def main():
    """主函数 - 示例使用"""
    
    print("=" * 70)
    print("增强版SGP4/SDP4轨道预报器 - 修复近地点奇异性")
    print("=" * 70)
    
    tle_elliptical = [
        "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993",
        "2 25544  51.6400 208.9163 0006703  35.7657  69.0011 15.49923619  1234"
    ]
    
    tle_circular = [
        "1 39084U 13008A   24001.50000000  .00000010  00000-0  10000-4 0  9999",
        "2 39084  97.5000  45.0000 0000100  90.0000 270.0000 15.00000000  1234"
    ]
    
    print("\n1. 测试椭圆轨道卫星 (ISS)")
    print("-" * 70)
    predictor1 = EnhancedOrbitPredictor(
        tle_elliptical[0], tle_elliptical[1], 
        "ISS (ZARYA)", propagation_mode='auto'
    )
    
    current_time = datetime.utcnow()
    pos, vel = predictor1.get_position_eci(current_time)
    print(f"\n当前位置 (ECI):")
    print(f"  X: {pos[0]:.2f} km, Y: {pos[1]:.2f} km, Z: {pos[2]:.2f} km")
    print(f"  轨道高度: {(np.linalg.norm(pos) - 6378.137):.2f} km")
    
    print("\n2. 测试近圆轨道卫星 (自动切换数值积分)")
    print("-" * 70)
    predictor2 = EnhancedOrbitPredictor(
        tle_circular[0], tle_circular[1],
        "Circular Satellite", propagation_mode='auto'
    )
    
    pos, vel = predictor2.get_position_eci(current_time)
    print(f"\n当前位置 (ECI):")
    print(f"  X: {pos[0]:.2f} km, Y: {pos[1]:.2f} km, Z: {pos[2]:.2f} km")
    print(f"  轨道高度: {(np.linalg.norm(pos) - 6378.137):.2f} km")
    
    print("\n3. 传播方法对比测试")
    print("-" * 70)
    comparison = predictor2.compare_propagation_methods(current_time)
    if 'difference' in comparison:
        print(f"  SGP4 vs 数值积分 位置差: {comparison['difference']['position_km']:.4f} km")
        print(f"  SGP4 vs 数值积分 速度差: {comparison['difference']['velocity_km_s']:.6f} km/s")
    
    print("\n4. 过顶预报 (北京地区)")
    print("-" * 70)
    passes = predictor1.predict_passes(
        observer_lat=39.9,
        observer_lon=116.4,
        observer_alt=50,
        duration_hours=12,
        min_elevation=10.0
    )
    
    if passes:
        print(f"  未来12小时 {len(passes)} 次过顶:")
        for i, p in enumerate(passes[:3], 1):
            print(f"    #{i}: {p['start_time'].strftime('%H:%M')} - {p['end_time'].strftime('%H:%M')} UTC, "
                  f"最大仰角: {p['max_elevation']:.1f}°")
    else:
        print("  预报时段内无过顶事件")
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
