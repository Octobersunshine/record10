import numpy as np
import matplotlib.pyplot as plt
from phonon_bte import PhononBTE
import time

def example_bulk_silicon():
    print("=" * 60)
    print("示例1: 体硅材料的晶格热导率随温度变化")
    print("=" * 60)
    
    bte_si = PhononBTE(material='Si', L=None)
    
    T_array = np.linspace(100, 800, 20)
    kappa_array = []
    
    start_time = time.time()
    for T in T_array:
        kappa = bte_si.thermal_conductivity(T)
        kappa_array.append(kappa)
        print(f"T = {T:.0f} K, κ = {kappa:.2f} W/mK")
    
    elapsed_time = time.time() - start_time
    print(f"\n计算完成，耗时: {elapsed_time:.2f} 秒")
    
    plt.figure(figsize=(10, 6))
    plt.plot(T_array, kappa_array, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('温度 T (K)', fontsize=14)
    plt.ylabel('晶格热导率 κ (W/mK)', fontsize=14)
    plt.title('体硅材料的晶格热导率随温度变化', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig('thermal_conductivity_vs_T_Si.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: thermal_conductivity_vs_T_Si.png")
    
    return T_array, kappa_array

def example_material_comparison():
    print("\n" + "=" * 60)
    print("示例2: 不同材料的热导率对比")
    print("=" * 60)
    
    materials = ['Si', 'Ge', 'GaAs']
    colors = ['b', 'r', 'g']
    T_array = np.linspace(100, 600, 15)
    
    plt.figure(figsize=(10, 6))
    
    for material, color in zip(materials, colors):
        bte = PhononBTE(material=material, L=None)
        kappa_array = [bte.thermal_conductivity(T) for T in T_array]
        plt.plot(T_array, kappa_array, f'{color}-o', linewidth=2, markersize=5, label=material)
        print(f"{material}: T=300K, κ = {bte.thermal_conductivity(300):.2f} W/mK")
    
    plt.xlabel('温度 T (K)', fontsize=14)
    plt.ylabel('晶格热导率 κ (W/mK)', fontsize=14)
    plt.title('不同半导体材料的晶格热导率对比', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('thermal_conductivity_comparison.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: thermal_conductivity_comparison.png")

def example_size_effect():
    print("\n" + "=" * 60)
    print("示例3: 纳米尺度下的尺寸效应 (Casimir极限)")
    print("=" * 60)
    
    T = 300
    L_array = np.logspace(-9, -5, 20)
    
    bte = PhononBTE(material='Si', L=None)
    kappa_bulk = bte.thermal_conductivity(T)
    print(f"体硅在 {T}K 时的热导率: {kappa_bulk:.2f} W/mK")
    
    kappa_nano = bte.size_effect_thermal_conductivity(T, L_array)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(L_array * 1e9, kappa_nano, 'r-o', linewidth=2, markersize=6)
    plt.axhline(y=kappa_bulk, color='k', linestyle='--', label='体材料极限')
    plt.xlabel('特征尺寸 L (nm)', fontsize=14)
    plt.ylabel('晶格热导率 κ (W/mK)', fontsize=14)
    plt.title(f'硅纳米结构热导率的尺寸效应 (T = {T}K)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('size_effect_thermal_conductivity.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: size_effect_thermal_conductivity.png")

def example_spectral_conductivity():
    print("\n" + "=" * 60)
    print("示例4: 频谱热导率分布")
    print("=" * 60)
    
    T = 300
    bte = PhononBTE(material='Si', L=None)
    
    omega_D = bte.debye_frequency()
    omega_array = np.linspace(0.01 * omega_D, omega_D, 100)
    
    spectral_kappa = [bte.spectral_thermal_conductivity(omega, T) for omega in omega_array]
    
    plt.figure(figsize=(10, 6))
    plt.plot(omega_array / 1e12, spectral_kappa, 'b-', linewidth=2)
    plt.xlabel('角频率 ω (THz)', fontsize=14)
    plt.ylabel('频谱热导率 dκ/dω', fontsize=14)
    plt.title(f'硅的频谱热导率分布 (T = {T}K)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig('spectral_thermal_conductivity.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: spectral_thermal_conductivity.png")

def example_cumulative_mfp():
    print("\n" + "=" * 60)
    print("示例5: 累积热导率与平均自由程的关系")
    print("=" * 60)
    
    T = 300
    bte = PhononBTE(material='Si', L=None)
    kappa_total = bte.thermal_conductivity(T)
    print(f"总体热导率: {kappa_total:.2f} W/mK")
    
    lambda_array = np.logspace(-9, -5, 30)
    kappa_cum = []
    
    for lam in lambda_array:
        kc = bte.cumulative_thermal_conductivity(T, max_lambda=lam)
        kappa_cum.append(kc / kappa_total)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(lambda_array * 1e9, kappa_cum, 'g-o', linewidth=2, markersize=5)
    plt.xlabel('平均自由程 λ (nm)', fontsize=14)
    plt.ylabel('累积热导率 κ_cum / κ_total', fontsize=14)
    plt.title(f'硅的累积热导率分布 (T = {T}K)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.savefig('cumulative_mfp.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: cumulative_mfp.png")
    
    lambda_50 = lambda_array[np.where(np.array(kappa_cum) >= 0.5)[0][0]]
    print(f"贡献50%热导率的声子平均自由程约为: {lambda_50 * 1e9:.1f} nm")

def example_relaxation_time():
    print("\n" + "=" * 60)
    print("示例6: 弛豫时间分析")
    print("=" * 60)
    
    T = 300
    bte = PhononBTE(material='Si', L=100e-9)
    
    omega_D = bte.debye_frequency()
    omega_array = np.logspace(np.log10(0.01 * omega_D), np.log10(omega_D), 50)
    
    tau_u = [bte.tau_umklapp(omega, T) for omega in omega_array]
    tau_b = [bte.tau_boundary(omega) for omega in omega_array]
    tau_i = [bte.tau_impurity(omega) for omega in omega_array]
    tau_total = [bte.relaxation_time(omega, T) for omega in omega_array]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(omega_array / 1e12, tau_u, 'r-', linewidth=2, label='Umklapp散射')
    plt.loglog(omega_array / 1e12, tau_b, 'b--', linewidth=2, label='边界散射 (L=100nm)')
    plt.loglog(omega_array / 1e12, tau_i, 'g-.', linewidth=2, label='杂质散射')
    plt.loglog(omega_array / 1e12, tau_total, 'k-', linewidth=3, label='总弛豫时间')
    plt.xlabel('角频率 ω (THz)', fontsize=14)
    plt.ylabel('弛豫时间 τ (s)', fontsize=14)
    plt.title(f'不同散射机制的弛豫时间 (T = {T}K)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('relaxation_time.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: relaxation_time.png")

def main():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    example_bulk_silicon()
    example_material_comparison()
    example_size_effect()
    example_spectral_conductivity()
    example_cumulative_mfp()
    example_relaxation_time()
    
    print("\n" + "=" * 60)
    print("所有示例计算完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
