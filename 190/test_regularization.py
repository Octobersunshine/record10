import numpy as np
from droplet_evaporation import DropletEvaporation


def test_singularity_behavior():
    print("=" * 70)
    print("测试: 奇异点行为分析")
    print("=" * 70)
    
    R0 = 500e-6
    theta0 = np.radians(60)
    
    print("\n1. 原始模型在小接触角下的g(θ)值:")
    print("-" * 50)
    
    model_none = DropletEvaporation(R0=R0, theta0=theta0, T=298.15,
                                    regularization_method='none')
    
    theta_values = [60, 30, 10, 5, 3, 1, 0.5, 0.1]
    
    print(f"{'θ (°)':>8} {'g(θ) 原始':>12} {'dV/dt (pL/s)':>15}")
    print("-" * 40)
    
    for theta_deg in theta_values:
        theta = np.radians(theta_deg)
        g = model_none._g_theta_original(theta)
        dVdt = model_none.dVdt_constant_R(R0, theta)
        print(f"{theta_deg:>8.1f} {g:>12.2f} {dVdt*1e12:>15.2f}")
    
    print("\n2. 不同正则化方法在小接触角下的g(θ)值:")
    print("-" * 50)
    
    methods = {
        'none': '无正则化',
        'cutoff': '截止法 (θ_min=3°)',
        'molecular_cutoff': '分子尺度截止',
        'average_flux': '平均通量',
        'saturation': '饱和函数'
    }
    
    theta_test = [10, 5, 3, 1, 0.5, 0.1]
    
    header = f"{'θ (°)':>8}"
    for method in methods:
        header += f" {method:>18}"
    print(header)
    print("-" * (8 + 18 * len(methods)))
    
    for theta_deg in theta_test:
        theta = np.radians(theta_deg)
        row = f"{theta_deg:>8.1f}"
        for method in methods:
            model = DropletEvaporation(R0=R0, theta0=theta0, T=298.15,
                                       regularization_method=method,
                                       theta_min=np.radians(3.0))
            g = model._g_theta(theta, R0)
            row += f" {g:>18.2f}"
        print(row)
    
    print("\n3. 蒸发速率比较 (恒定接触半径模式):")
    print("-" * 50)
    
    for method, desc in methods.items():
        model = DropletEvaporation(R0=R0, theta0=theta0, T=298.15,
                                   regularization_method=method,
                                   theta_min=np.radians(3.0))
        
        dVdt = model.dVdt_constant_R(R0, theta0)
        V0 = model._volume_from_R_theta(R0, theta0)
        est_time = V0 / abs(dVdt)
        
        results = model.simulate_constant_contact_radius(max_time=est_time * 1.5)
        
        print(f"\n{desc}:")
        print(f"  初始蒸发速率: {abs(dVdt)*1e12:.2f} pL/s")
        print(f"  预计蒸发时间: {est_time:.4f} s")
        if len(results['t']) > 0:
            print(f"  实际模拟时间: {results['t'][-1]:.4f} s")
            print(f"  最终接触角: {np.degrees(results['theta'][-1]):.2f}°")
            print(f"  最终体积: {results['V'][-1]*1e12:.4f} pL")
            
            fluxes = np.abs(np.gradient(results['V']*1e12, results['t']))
            print(f"  最大蒸发速率: {np.max(fluxes):.2f} pL/s")
            print(f"  最小蒸发速率: {np.min(fluxes):.2f} pL/s")
        else:
            print(f"  模拟结果为空 (奇异点导致积分失败)")
    
    print("\n" + "=" * 70)


def test_small_theta_simulation():
    print("\n" + "=" * 70)
    print("测试: 小初始接触角模拟")
    print("=" * 70)
    
    R0 = 500e-6
    theta0_small = np.radians(15)
    
    print(f"\n初始接触角: 15°")
    print(f"初始接触半径: {R0*1e6:.0f} μm")
    
    for method in ['none', 'cutoff', 'molecular_cutoff']:
        model = DropletEvaporation(R0=R0, theta0=theta0_small, T=298.15,
                                   regularization_method=method,
                                   theta_min=np.radians(3.0))
        
        dVdt = model.dVdt_constant_R(R0, theta0_small)
        V0 = model._volume_from_R_theta(R0, theta0_small)
        est_time = V0 / abs(dVdt)
        
        results = model.simulate_constant_contact_radius(max_time=est_time * 2)
        
        print(f"\n{method}:")
        print(f"  预计蒸发时间: {est_time:.4f} s")
        if len(results['t']) > 0:
            print(f"  模拟点数: {len(results['t'])}")
            print(f"  最终体积: {results['V'][-1]*1e12:.4f} pL")
            
            dV = np.diff(results['V'])
            dt = np.diff(results['t'])
            fluxes = np.abs(dV / dt) * 1e12
            
            if len(fluxes) > 0:
                print(f"  最大瞬时通量: {np.max(fluxes):.2f} pL/s")
                print(f"  通量发散因子: {np.max(fluxes)/np.min(fluxes):.2f}")
        else:
            print(f"  模拟失败!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_singularity_behavior()
    test_small_theta_simulation()
    print("\n所有测试完成!")
