import numpy as np
import matplotlib.pyplot as plt
from phonon_monte_carlo import PhononMonteCarlo, Layer, Superlattice
import time


def example_basic_mc():
    print("=" * 70)
    print("示例1: 基础蒙特卡洛热导率计算")
    print("=" * 70)
    
    mc = PhononMonteCarlo(material='Si', structure='thin_film', L=100e-9, seed=42)
    
    result = mc.run_simulation(
        n_phonons=2000,
        n_steps=300,
        dt=1e-13,
        T_hot=310,
        T_cold=290
    )
    
    print(f"\n模拟结果:")
    print(f"  热导率: {result['thermal_conductivity']:.2f} W/mK")
    print(f"  热流密度: {result['heat_flux']:.2e} W/m²")
    print(f"  模拟时间: {result['simulation_time_s']:.2f} s")
    
    stats = mc.get_phonon_statistics()
    if stats:
        print(f"\n声子统计:")
        print(f"  活跃声子数: {stats['n_active']}")
        print(f"  平均散射次数: {stats['avg_scatters']:.2f}")
        print(f"  分支分布: {stats['branch_distribution']}")
    
    return result


def example_size_effect():
    print("\n" + "=" * 70)
    print("示例2: 尺寸效应分析 (热导率随特征尺寸变化)")
    print("=" * 70)
    
    mc = PhononMonteCarlo(material='Si', structure='thin_film', seed=42)
    
    L_array = np.logspace(-8, -6, 6)
    L_array = np.concatenate([L_array, [1e-5]])
    
    T = 300
    kappas, kappa_bulk = mc.size_effect_analysis(
        L_array, T=T, n_phonons=1500, n_steps=250
    )
    
    print(f"\n体材料热导率: {kappa_bulk:.2f} W/mK")
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(L_array * 1e9, kappas, 'b-o', linewidth=2, markersize=7, label='蒙特卡洛')
    plt.axhline(y=kappa_bulk, color='k', linestyle='--', alpha=0.7, label='体材料极限')
    
    plt.xlabel('特征尺寸 L (nm)', fontsize=12)
    plt.ylabel('热导率 κ (W/mK)', fontsize=12)
    plt.title('硅薄膜热导率尺寸效应 (T = 300K)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('size_effect_monte_carlo.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存到: size_effect_monte_carlo.png")
    
    return L_array, kappas, kappa_bulk


def example_superlattice():
    print("\n" + "=" * 70)
    print("示例3: Si/Ge超晶格热导率分析")
    print("=" * 70)
    
    mc = PhononMonteCarlo(material='Si/Ge', structure='superlattice', seed=42)
    
    period_array = np.logspace(-9, -8, 5)
    
    T = 300
    kappas_sl = mc.superlattice_thermal_conductivity(
        period_array, T=T, n_phonons=2000, n_steps=300
    )
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(period_array * 1e9, kappas_sl, 'r-s', linewidth=2, markersize=7, label='Si/Ge超晶格')
    
    plt.xlabel('超晶格周期 (nm)', fontsize=12)
    plt.ylabel('热导率 κ (W/mK)', fontsize=12)
    plt.title('Si/Ge超晶格热导率 (T = 300K)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('superlattice_thermal_conductivity.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: superlattice_thermal_conductivity.png")
    
    return period_array, kappas_sl


def example_phonon_trajectories():
    print("\n" + "=" * 70)
    print("示例4: 声子轨迹可视化")
    print("=" * 70)
    
    mc = PhononMonteCarlo(material='Si', structure='thin_film', L=200e-9, seed=123)
    
    mc.run_simulation(n_phonons=50, n_steps=500, dt=1e-13)
    
    trajectories = []
    for packet in mc.phonon_packets[:10]:
        positions = []
        mc2 = PhononMonteCarlo(material='Si', structure='thin_film', L=200e-9, seed=packet.phonon_id)
        mc2.initialize_phonons(1, 310, 290)
        p = mc2.phonon_packets[0]
        pos = [p.position[2].copy()]
        
        for _ in range(200):
            p.position += p.velocity * 1e-13
            z_old = pos[-1]
            z_new = p.position[2]
            
            L_half = mc2.L / 2
            if z_new >= L_half:
                p.position[2] = 2 * L_half - z_new
                p.velocity[2] *= -1
            elif z_new <= -L_half:
                p.position[2] = -2 * L_half - z_new
                p.velocity[2] *= -1
            
            pos.append(p.position[2])
        
        trajectories.append(pos)
    
    plt.figure(figsize=(12, 5))
    
    time_axis = np.arange(201) * 1e-13 * 1e9
    
    for i, traj in enumerate(trajectories[:6]):
        plt.plot(time_axis, np.array(traj) * 1e9, linewidth=1.5, alpha=0.8, label=f'声子 {i+1}')
    
    plt.axhline(y=100, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=-100, color='k', linestyle='--', alpha=0.5)
    plt.fill_between(time_axis, -100, 100, alpha=0.1, color='gray')
    
    plt.xlabel('时间 (ns)', fontsize=12)
    plt.ylabel('z位置 (nm)', fontsize=12)
    plt.title('声子在薄膜中的运动轨迹', fontsize=14)
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phonon_trajectories.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: phonon_trajectories.png")


def example_interface_transmission_effect():
    print("\n" + "=" * 70)
    print("示例5: 界面透射率对超晶格热导率的影响")
    print("=" * 70)
    
    T = 300
    period = 10e-9
    transmissions = [0.4, 0.6, 0.8, 0.9, 0.95]
    kappas = []
    
    for trans in transmissions:
        print(f"\n界面透射率: {trans}")
        
        layer_thickness = period / 2
        layers = [
            Layer(layer_thickness, 'Si', trans),
            Layer(layer_thickness, 'Ge', trans),
        ]
        
        mc = PhononMonteCarlo(material='Si/Ge', structure='superlattice', seed=42)
        mc.superlattice = Superlattice(layers)
        mc.L = period * 20
        
        kappa = mc.thermal_conductivity_mc(T, n_phonons=1500, n_steps=250, L=period*20)
        kappas.append(kappa)
        print(f"  κ = {kappa:.2f} W/mK")
    
    plt.figure(figsize=(10, 6))
    plt.plot(transmissions, kappas, 'g-o', linewidth=2, markersize=7)
    plt.xlabel('界面声子透射率', fontsize=12)
    plt.ylabel('热导率 κ (W/mK)', fontsize=12)
    plt.title('界面透射率对超晶格热导率的影响', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('interface_transmission_effect.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: interface_transmission_effect.png")


def example_temperature_dependence():
    print("\n" + "=" * 70)
    print("示例6: 温度依赖性分析")
    print("=" * 70)
    
    L = 100e-9
    mc = PhononMonteCarlo(material='Si', structure='thin_film', L=L, seed=42)
    
    T_array = [200, 300, 400, 500]
    kappas = []
    
    for T in T_array:
        print(f"\nT = {T} K")
        kappa = mc.thermal_conductivity_mc(T, n_phonons=1500, n_steps=250, L=L)
        kappas.append(kappa)
        print(f"  κ = {kappa:.2f} W/mK")
    
    plt.figure(figsize=(10, 6))
    plt.plot(T_array, kappas, 'b-s', linewidth=2, markersize=7, label=f'L = {L*1e9:.0f} nm')
    plt.xlabel('温度 T (K)', fontsize=12)
    plt.ylabel('热导率 κ (W/mK)', fontsize=12)
    plt.title('薄膜热导率的温度依赖性', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temperature_dependence_mc.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: temperature_dependence_mc.png")


def example_compare_methods():
    print("\n" + "=" * 70)
    print("示例7: 蒙特卡洛 vs 解析BTE对比")
    print("=" * 70)
    
    from phonon_bte_enhanced import EnhancedPhononBTE
    
    L = 200e-9
    T = 300
    
    bte = EnhancedPhononBTE(material='Si', L=L, use_branch_resolved=True)
    kappa_bte = bte.thermal_conductivity(T)
    print(f"解析BTE (模式分辨): κ = {kappa_bte:.2f} W/mK")
    
    mc = PhononMonteCarlo(material='Si', structure='thin_film', L=L, seed=42)
    kappa_mc = mc.thermal_conductivity_mc(T, n_phonons=3000, n_steps=400, L=L)
    print(f"蒙特卡洛模拟: κ = {kappa_mc:.2f} W/mK")
    
    diff = abs(kappa_mc - kappa_bte) / kappa_bte * 100
    print(f"相对差异: {diff:.1f}%")
    
    methods = ['解析BTE\n(模式分辨)', '蒙特卡洛\n模拟']
    values = [kappa_bte, kappa_mc]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, values, color=['#1f77b4', '#ff7f0e'], alpha=0.8, width=0.5)
    
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12)
    
    plt.ylabel('热导率 κ (W/mK)', fontsize=12)
    plt.title(f'方法对比 (T = {T}K, L = {L*1e9:.0f}nm)', fontsize=14)
    plt.ylim([0, max(values) * 1.15])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: method_comparison.png")


def main():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    start_time = time.time()
    
    example_basic_mc()
    example_size_effect()
    example_superlattice()
    example_phonon_trajectories()
    example_interface_transmission_effect()
    example_temperature_dependence()
    example_compare_methods()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"所有蒙特卡洛示例完成！总耗时: {total_time:.1f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    main()
