import numpy as np
from droplet_evaporation import DropletEvaporation


def example_basic_usage():
    print("示例1: 基本使用")
    print("-" * 50)
    
    model = DropletEvaporation(R0=500e-6, theta0=np.radians(45), T=298.15)
    
    results_ccr = model.simulate_constant_contact_radius(max_time=0.5)
    print(f"恒定接触半径模式:")
    print(f"  蒸发时间: {results_ccr['t'][-1]:.4f} s")
    print(f"  初始体积: {results_ccr['V'][0] * 1e12:.2f} pL")
    print(f"  最终体积: {results_ccr['V'][-1] * 1e12:.2f} pL")
    print()
    

def example_different_temperatures():
    print("示例2: 不同温度下的蒸发对比")
    print("-" * 50)
    
    R0 = 500e-6
    theta0 = np.radians(60)
    
    for T in [283.15, 298.15, 313.15]:
        model = DropletEvaporation(R0=R0, theta0=theta0, T=T)
        results = model.simulate_constant_contact_angle(max_time=1.0)
        
        V0 = results['V'][0]
        V_end = results['V'][-1]
        evap_ratio = (V0 - V_end) / V0 * 100
        
        print(f"温度 {T-273.15:.0f}°C:")
        print(f"  蒸发时间: {results['t'][-1]:.4f} s")
        print(f"  蒸发比例: {evap_ratio:.1f}%")
        print(f"  饱和蒸气压: {model.Psat:.1f} Pa")
    print()


def example_different_contact_angles():
    print("示例3: 不同初始接触角的对比")
    print("-" * 50)
    
    R0 = 500e-6
    T = 298.15
    
    for theta_deg in [30, 60, 90]:
        theta0 = np.radians(theta_deg)
        model = DropletEvaporation(R0=R0, theta0=theta0, T=T)
        results = model.simulate_constant_contact_radius(max_time=0.5)
        
        print(f"初始接触角 {theta_deg}°:")
        print(f"  初始体积: {results['V'][0] * 1e12:.2f} pL")
        print(f"  初始高度: {results['h'][0] * 1e6:.2f} μm")
        print(f"  g(θ) = {model._g_theta(theta0):.4f}")
    print()


def example_picoliter_droplet():
    print("示例4: 皮升级液滴模拟")
    print("-" * 50)
    
    target_volume = 100e-12
    theta0 = np.radians(45)
    T = 298.15
    
    R = (3 * target_volume / (np.pi * (1 - np.cos(theta0))**2 * (2 + np.cos(theta0)))) ** (1/3)
    
    model = DropletEvaporation(R0=R, theta0=theta0, T=T)
    results = model.simulate_constant_contact_angle(max_time=1.0)
    
    print(f"目标体积: {target_volume * 1e12:.1f} pL")
    print(f"所需接触半径: {R * 1e6:.2f} μm")
    print(f"实际初始体积: {results['V'][0] * 1e12:.2f} pL")
    print(f"蒸发完成时间: {results['t'][-1]:.4f} s")
    print()


def example_time_series_data():
    print("示例5: 获取时间序列数据")
    print("-" * 50)
    
    model = DropletEvaporation(R0=500e-6, theta0=np.radians(60), T=298.15)
    results = model.simulate_constant_contact_radius(max_time=0.3)
    
    t = results['t']
    theta = np.degrees(results['theta'])
    V = results['V'] * 1e12
    
    print("时间点数据:")
    for i in range(0, len(t), len(t) // 5):
        print(f"  t={t[i]:.3f}s, θ={theta[i]:.1f}°, V={V[i]:.1f}pL")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("微液滴蒸发模型 - 使用示例")
    print("=" * 60)
    print()
    
    example_basic_usage()
    example_different_temperatures()
    example_different_contact_angles()
    example_picoliter_droplet()
    example_time_series_data()
    
    print("=" * 60)
    print("示例运行完成!")
    print("=" * 60)
