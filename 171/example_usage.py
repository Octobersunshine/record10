#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版SGP4/SDP4轨道预报器 - 使用示例
"""

from datetime import datetime, timedelta
from sgp4_orbit_predictor import EnhancedOrbitPredictor
import numpy as np


def example1_basic_position():
    """示例1: 基本ECI位置计算"""
    print("=" * 70)
    print("示例1: 基本ECI位置计算")
    print("=" * 70)
    
    tle1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993"
    tle2 = "2 25544  51.6400 208.9163 0006703  35.7657  69.0011 15.49923619  1234"
    
    predictor = EnhancedOrbitPredictor(tle1, tle2, "ISS (ZARYA)", propagation_mode='auto')
    
    target_time = datetime.utcnow()
    pos, vel = predictor.get_position_eci(target_time)
    
    print(f"\n目标时间: {target_time} UTC")
    print(f"传播模式: {predictor.current_mode.upper()}")
    print(f"ECI位置: X={pos[0]:.2f} km, Y={pos[1]:.2f} km, Z={pos[2]:.2f} km")
    print(f"ECI速度: Vx={vel[0]:.4f} km/s, Vy={vel[1]:.4f} km/s, Vz={vel[2]:.4f} km/s")
    print(f"轨道高度: {(np.linalg.norm(pos) - 6378.137):.2f} km")
    print(f"飞行速度: {np.linalg.norm(vel):.4f} km/s")


def example2_circular_orbit():
    """示例2: 近圆轨道处理（自动切换数值积分）"""
    print("\n" + "=" * 70)
    print("示例2: 近圆轨道处理 - 自动切换数值积分避免奇异性")
    print("=" * 70)
    
    tle_circular = [
        "1 43013U 17073A   24001.50000000  .00000010  00000-0  10000-4 0  9999",
        "2 43013  97.5000  45.0000 0000050  90.0000 270.0000 15.20000000  1234"
    ]
    
    predictor = EnhancedOrbitPredictor(
        tle_circular[0], tle_circular[1], 
        "Near-Circular Satellite", 
        propagation_mode='auto'
    )
    
    print(f"\n近圆轨道检测: {'是' if predictor.is_near_circular else '否'}")
    print(f"偏心率: {predictor.eccentricity:.8f}")
    print(f"使用传播模式: {predictor.current_mode.upper()}")
    
    target_time = datetime.utcnow()
    pos, vel = predictor.get_position_eci(target_time)
    print(f"\n当前位置 (ECI):")
    print(f"  X={pos[0]:.2f} km, Y={pos[1]:.2f} km, Z={pos[2]:.2f} km")
    print(f"  轨道高度: {(np.linalg.norm(pos) - 6378.137):.2f} km")


def example3_propagation_comparison():
    """示例3: 不同传播方法对比"""
    print("\n" + "=" * 70)
    print("示例3: 传播方法对比 (SGP4 vs 数值积分)")
    print("=" * 70)
    
    tle1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993"
    tle2 = "2 25544  51.6400 208.9163 0006703  35.7657  69.0011 15.49923619  1234"
    
    predictor = EnhancedOrbitPredictor(tle1, tle2, "ISS", propagation_mode='sgp4')
    
    base_time = datetime.utcnow()
    hours_ahead = [0, 1, 3, 6, 12]
    
    print(f"\n{'时间(小时后)':^12} | {'SGP4 vs 数值积分位置差(km)':^30}")
    print("-" * 50)
    
    for h in hours_ahead:
        target_time = base_time + timedelta(hours=h)
        comparison = predictor.compare_propagation_methods(target_time)
        
        if 'difference' in comparison:
            pos_diff = comparison['difference']['position_km']
            print(f"{h:^12} | {pos_diff:^30.4f}")
        else:
            print(f"{h:^12} | {'计算失败':^30}")


def example4_rk4_vs_rk8():
    """示例4: RK4 vs RK8积分器对比"""
    print("\n" + "=" * 70)
    print("示例4: RK4 vs RK8 数值积分器对比")
    print("=" * 70)
    
    tle_circular = [
        "1 39084U 13008A   24001.50000000  .00000010  00000-0  10000-4 0  9999",
        "2 39084  97.5000  45.0000 0000100  90.0000 270.0000 15.00000000  1234"
    ]
    
    predictor_rk4 = EnhancedOrbitPredictor(
        tle_circular[0], tle_circular[1], "Test-RK4",
        propagation_mode='numerical', integrator='rk4'
    )
    
    predictor_rk8 = EnhancedOrbitPredictor(
        tle_circular[0], tle_circular[1], "Test-RK8",
        propagation_mode='numerical', integrator='rk8'
    )
    
    target_time = datetime.utcnow() + timedelta(hours=6)
    
    pos_rk4, vel_rk4 = predictor_rk4.get_position_eci(target_time)
    pos_rk8, vel_rk8 = predictor_rk8.get_position_eci(target_time)
    
    pos_diff = np.linalg.norm(pos_rk4 - pos_rk8)
    vel_diff = np.linalg.norm(vel_rk4 - vel_rk8)
    
    print(f"\n6小时预报后:")
    print(f"  RK4位置: X={pos_rk4[0]:.2f}, Y={pos_rk4[1]:.2f}, Z={pos_rk4[2]:.2f} km")
    print(f"  RK8位置: X={pos_rk8[0]:.2f}, Y={pos_rk8[1]:.2f}, Z={pos_rk8[2]:.2f} km")
    print(f"  位置差: {pos_diff:.6f} km")
    print(f"  速度差: {vel_diff:.6f} km/s")


def example5_pass_prediction():
    """示例5: 过顶预报"""
    print("\n" + "=" * 70)
    print("示例5: 过顶预报 (上海地区)")
    print("=" * 70)
    
    tle1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993"
    tle2 = "2 25544  51.6400 208.9163 0006703  35.7657  69.0011 15.49923619  1234"
    
    predictor = EnhancedOrbitPredictor(tle1, tle2, "ISS", propagation_mode='auto')
    
    passes = predictor.predict_passes(
        observer_lat=31.23,
        observer_lon=121.47,
        observer_alt=4,
        duration_hours=12,
        min_elevation=15.0
    )
    
    if passes:
        print(f"\n未来12小时内 {len(passes)} 次过顶 (仰角>=15°):")
        print("-" * 80)
        print(f"{'#':>2} | {'开始时间':^19} | {'结束时间':^19} | {'时长':>6} | {'最大仰角':>8}")
        print("-" * 80)
        
        for i, p in enumerate(passes, 1):
            print(f"{i:>2} | "
                  f"{p['start_time'].strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"{p['end_time'].strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"{p['duration']:>5.1f}m | "
                  f"{p['max_elevation']:>7.1f}°")
    else:
        print("预报时段内无过顶事件")


def example6_communication_windows():
    """示例6: 通信窗口分析"""
    print("\n" + "=" * 70)
    print("示例6: 通信窗口详细分析")
    print("=" * 70)
    
    tle1 = "1 37849U 11061A   24001.50000000  .00000100  00000-0  50000-4 0  9999"
    tle2 = "2 37849  98.5000 100.0000 0015000  80.0000 280.0000 14.50000000  1234"
    
    predictor = EnhancedOrbitPredictor(tle1, tle2, "ZY-3 01", propagation_mode='auto')
    
    windows = predictor.get_communication_windows(
        observer_lat=40.0,
        observer_lon=116.0,
        observer_alt=50,
        duration_hours=6,
        min_elevation=20.0
    )
    
    if windows:
        print(f"\n可用通信窗口 (仰角>=20°):")
        print("-" * 90)
        print(f"{'窗口':>4} | {'时间段':^20} | {'时长':>6} | {'最大仰角':>8} | {'最近距离':>10}")
        print("-" * 90)
        
        for w in windows:
            time_range = f"{w['start_time'].strftime('%H:%M')}-{w['end_time'].strftime('%H:%M')}"
            range_str = f"{w.get('range_at_max_elev', 0):.1f} km"
            print(f"{w['window_id']:>4} | "
                  f"{time_range:^20} | "
                  f"{w['duration_minutes']:>5.1f}m | "
                  f"{w['max_elevation']:>7.1f}° | "
                  f"{range_str:>10}")
    else:
        print("预报时段内无可用通信窗口")


def example7_orbit_propagation():
    """示例7: 轨道预报输出"""
    print("\n" + "=" * 70)
    print("示例7: 多时间点轨道预报")
    print("=" * 70)
    
    tle1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993"
    tle2 = "2 25544  51.6400 208.9163 0006703  35.7657  69.0011 15.49923619  1234"
    
    predictor = EnhancedOrbitPredictor(tle1, tle2, "ISS", propagation_mode='auto')
    
    orbit = predictor.predict_orbit(duration_minutes=90, step_seconds=300)
    orbit = predictor.get_ground_track(orbit)
    
    print(f"\n预报 {len(orbit)} 个轨道点 (每5分钟一个):")
    print("-" * 75)
    print(f"{'时间':^19} | {'纬度':>8} | {'经度':>8} | {'高度(km)':>10} | {'模式':>8}")
    print("-" * 75)
    
    for point in orbit[::2]:
        print(f"{point['time'].strftime('%Y-%m-%d %H:%M:%S')} | "
              f"{point['latitude']:>8.2f} | "
              f"{point['longitude']:>8.2f} | "
              f"{point['altitude']:>10.2f} | "
              f"{point['mode']:>8}")


def example8_force_mode_switch():
    """示例8: 强制使用不同传播模式"""
    print("\n" + "=" * 70)
    print("示例8: 强制指定传播模式")
    print("=" * 70)
    
    tle1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993"
    tle2 = "2 25544  51.6400 208.9163 0006703  35.7657  69.0011 15.49923619  1234"
    
    modes = ['sgp4', 'numerical']
    target_time = datetime.utcnow() + timedelta(hours=3)
    
    for mode in modes:
        predictor = EnhancedOrbitPredictor(
            tle1, tle2, f"ISS-{mode.upper()}", 
            propagation_mode=mode
        )
        pos, vel = predictor.get_position_eci(target_time)
        print(f"\n{mode.upper()} 模式:")
        print(f"  位置: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f} km")
        print(f"  高度: {(np.linalg.norm(pos) - 6378.137):.2f} km")


if __name__ == "__main__":
    try:
        example1_basic_position()
        example2_circular_orbit()
        example3_propagation_comparison()
        example4_rk4_vs_rk8()
        example5_pass_prediction()
        example6_communication_windows()
        example7_orbit_propagation()
        example8_force_mode_switch()
        
        print("\n" + "=" * 70)
        print("所有示例执行完成!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"错误: {e}")
        print("请先安装依赖: pip install sgp4 numpy")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
