import numpy as np
import matplotlib.pyplot as plt
from phonon_bte_enhanced import EnhancedPhononBTE
import time


def example_model_comparison():
    print("=" * 70)
    print("对比: 灰体近似 (单弛豫时间) vs 模式分辨弛豫时间")
    print("=" * 70)
    
    bte = EnhancedPhononBTE(material='Si', L=None)
    
    T_array = np.linspace(100, 600, 15)
    kappa_gray = []
    kappa_mode_resolved = []
    
    start_time = time.time()
    for T in T_array:
        bte.use_branch_resolved = False
        k_g = bte.thermal_conductivity(T)
        kappa_gray.append(k_g)
        
        bte.use_branch_resolved = True
        k_m = bte.thermal_conductivity(T)
        kappa_mode_resolved.append(k_m)
        
        diff = (k_m - k_g) / k_g * 100
        print(f"T = {T:.0f} K | 灰体: {k_g:.2f} | 模式分辨: {k_m:.2f} | 差异: {diff:+.1f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\n计算完成，耗时: {elapsed_time:.2f} 秒")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(T_array, kappa_gray, 'r--o', linewidth=2, markersize=5, label='灰体近似')
    plt.plot(T_array, kappa_mode_resolved, 'b-s', linewidth=2, markersize=5, label='模式分辨')
    plt.xlabel('温度 T (K)', fontsize=12)
    plt.ylabel('热导率 κ (W/mK)', fontsize=12)
    plt.title('两种模型热导率对比', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    diff_percent = [(k_m - k_g) for k_m, k_g in zip(kappa_mode_resolved, kappa_gray)]
    diff_percent = np.array(diff_percent) / np.array(kappa_gray) * 100
    plt.plot(T_array, diff_percent, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('温度 T (K)', fontsize=12)
    plt.ylabel('相对差异 (%)', fontsize=12)
    plt.title('模式分辨 vs 灰体近似', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存到: model_comparison.png")


def example_branch_contributions():
    print("\n" + "=" * 70)
    print("各声子支对热导率的贡献")
    print("=" * 70)
    
    bte = EnhancedPhononBTE(material='Si', L=None, use_branch_resolved=True)
    
    T = 300
    contributions = bte.get_branch_contributions(T)
    
    total_kappa = sum(contributions.values())
    print(f"\n温度 T = {T} K")
    print(f"总热导率: {total_kappa:.2f} W/mK")
    print("\n各声子支贡献:")
    for branch, kappa in contributions.items():
        percent = kappa / total_kappa * 100
        print(f"  {branch}: {kappa:.2f} W/mK ({percent:.1f}%)")
    
    T_array = np.linspace(100, 600, 10)
    contribs_T = {'LA': [], 'TA1': [], 'TA2': []}
    
    for T in T_array:
        c = bte.get_branch_contributions(T)
        for branch in contribs_T:
            contribs_T[branch].append(c.get(branch, 0))
    
    plt.figure(figsize=(10, 6))
    colors = ['b', 'r', 'g']
    bottom = np.zeros(len(T_array))
    
    for i, (branch, kappas) in enumerate(contribs_T.items()):
        plt.bar(T_array, kappas, width=40, bottom=bottom, label=branch, alpha=0.7)
        bottom += np.array(kappas)
    
    plt.xlabel('温度 T (K)', fontsize=12)
    plt.ylabel('热导率 κ (W/mK)', fontsize=12)
    plt.title('各声子支热导率贡献', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('branch_contributions.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: branch_contributions.png")


def example_relaxation_time_comparison():
    print("\n" + "=" * 70)
    print("不同声子支的弛豫时间对比")
    print("=" * 70)
    
    bte = EnhancedPhononBTE(material='Si', L=None)
    T = 300
    
    k_D = bte.debye_wavevector()
    k_array = np.logspace(np.log10(k_D/1000), np.log10(k_D), 50)
    
    plt.figure(figsize=(10, 6))
    
    branches = ['LA', 'TA1']
    linestyles = ['-', '--']
    colors = ['b', 'r']
    
    for branch, ls, color in zip(branches, linestyles, colors):
        tau_array = []
        for k in k_array:
            omega = bte.dispersion_relation(k, branch)
            tau = bte.relaxation_time(k, omega, T, branch)
            tau_array.append(tau)
        
        omega_array = [bte.dispersion_relation(k, branch) for k in k_array]
        plt.loglog(np.array(omega_array) / 1e12, tau_array, f'{color}{ls}', linewidth=2, label=branch)
    
    plt.xlabel('角频率 ω (THz)', fontsize=12)
    plt.ylabel('弛豫时间 τ (s)', fontsize=12)
    plt.title(f'不同声子支弛豫时间对比 (T = {T}K)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('tau_branch_comparison.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: tau_branch_comparison.png")


def example_scattering_mechanisms_branch():
    print("\n" + "=" * 70)
    print("各散射机制对不同声子支的贡献")
    print("=" * 70)
    
    bte = EnhancedPhononBTE(material='Si', L=None)
    T = 300
    branch = 'TA1'
    
    k_D = bte.debye_wavevector()
    k_array = np.logspace(np.log10(k_D/1000), np.log10(k_D), 40)
    
    tau_n = []
    tau_u = []
    tau_iso = []
    tau_total = []
    
    for k in k_array:
        omega = bte.dispersion_relation(k, branch)
        tau_n.append(bte.tau_normal(omega, T, branch))
        tau_u.append(bte.tau_umklapp(omega, T, branch))
        tau_iso.append(bte.tau_isotope(omega, branch))
        tau_total.append(bte.relaxation_time(k, omega, T, branch))
    
    omega_array = [bte.dispersion_relation(k, branch) for k in k_array]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(np.array(omega_array) / 1e12, tau_n, 'b-', linewidth=2, label='N过程')
    plt.loglog(np.array(omega_array) / 1e12, tau_u, 'r--', linewidth=2, label='U过程')
    plt.loglog(np.array(omega_array) / 1e12, tau_iso, 'g-.', linewidth=2, label='同位素散射')
    plt.loglog(np.array(omega_array) / 1e12, tau_total, 'k-', linewidth=3, label='总弛豫时间')
    plt.xlabel('角频率 ω (THz)', fontsize=12)
    plt.ylabel('弛豫时间 τ (s)', fontsize=12)
    plt.title(f'{branch}支各散射机制 (T = {T}K)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'scattering_mechanisms_{branch}.png', dpi=150, bbox_inches='tight')
    print(f"图表已保存到: scattering_mechanisms_{branch}.png")


def example_dispersion_correction():
    print("\n" + "=" * 70)
    print("色散关系修正对比")
    print("=" * 70)
    
    bte_linear = EnhancedPhononBTE(material='Si', L=None, use_dispersion_correction=False)
    bte_corrected = EnhancedPhononBTE(material='Si', L=None, use_dispersion_correction=True)
    
    k_D = bte_linear.debye_wavevector()
    k_array = np.linspace(0, k_D, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    branch = 'LA'
    omega_linear = [bte_linear.dispersion_relation(k, branch) for k in k_array]
    omega_corrected = [bte_corrected.dispersion_relation(k, branch) for k in k_array]
    
    axes[0].plot(k_array / 1e9, np.array(omega_linear) / 1e12, 'r--', linewidth=2, label='线性色散')
    axes[0].plot(k_array / 1e9, np.array(omega_corrected) / 1e12, 'b-', linewidth=2, label='修正色散')
    axes[0].set_xlabel('波矢 k (nm⁻¹)', fontsize=12)
    axes[0].set_ylabel('角频率 ω (THz)', fontsize=12)
    axes[0].set_title(f'{branch}支色散关系', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    vg_linear = [bte_linear.group_velocity(k, branch) for k in k_array]
    vg_corrected = [bte_corrected.group_velocity(k, branch) for k in k_array]
    
    axes[1].plot(k_array / 1e9, vg_linear, 'r--', linewidth=2, label='线性色散')
    axes[1].plot(k_array / 1e9, vg_corrected, 'b-', linewidth=2, label='修正色散')
    axes[1].set_xlabel('波矢 k (nm⁻¹)', fontsize=12)
    axes[1].set_ylabel('群速度 v_g (m/s)', fontsize=12)
    axes[1].set_title(f'{branch}支群速度', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dispersion_correction.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: dispersion_correction.png")
    
    T = 300
    kappa_linear = bte_linear.thermal_conductivity(T)
    kappa_corrected = bte_corrected.thermal_conductivity(T)
    print(f"\n线性色散模型: κ = {kappa_linear:.2f} W/mK")
    print(f"修正色散模型: κ = {kappa_corrected:.2f} W/mK")
    print(f"差异: {(kappa_corrected - kappa_linear) / kappa_linear * 100:+.1f}%")


def example_size_effect_mode_resolved():
    print("\n" + "=" * 70)
    print("模式分辨下的尺寸效应")
    print("=" * 70)
    
    bte = EnhancedPhononBTE(material='Si', L=None)
    T = 300
    
    L_array = np.logspace(-9, -5, 15)
    
    kappa_bulk = bte.thermal_conductivity(T)
    print(f"体材料热导率: {kappa_bulk:.2f} W/mK")
    
    print("\n各尺寸下的热导率:")
    print(f"{'尺寸 (nm)':>12} {'κ (W/mK)':>15} {'κ/κ_bulk':>12}")
    print("-" * 45)
    
    kappas = []
    for L in L_array:
        bte.L = L
        kappa = bte.thermal_conductivity(T)
        kappas.append(kappa)
        print(f"{L*1e9:>12.1f} {kappa:>15.2f} {kappa/kappa_bulk:>12.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(L_array * 1e9, kappas, 'b-o', linewidth=2, markersize=5)
    plt.axhline(y=kappa_bulk, color='k', linestyle='--', label='体材料极限')
    plt.xlabel('特征尺寸 L (nm)', fontsize=12)
    plt.ylabel('热导率 κ (W/mK)', fontsize=12)
    plt.title(f'模式分辨模型的尺寸效应 (T = {T}K)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('size_effect_mode_resolved.png', dpi=150, bbox_inches='tight')
    print("图表已保存到: size_effect_mode_resolved.png")


def example_first_principles_interface():
    print("\n" + "=" * 70)
    print("第一性原理数据接口演示")
    print("=" * 70)
    
    bte = EnhancedPhononBTE(material='Si', L=None)
    
    omega = np.logspace(12, 14, 100)
    
    tau_LA = 1e-12 * (omega / 1e13)**(-1.5)
    tau_TA = 1e-12 * (omega / 1e13)**(-2.0)
    
    vg_LA = 8000 * np.ones_like(omega)
    vg_TA = 5000 * np.ones_like(omega)
    
    dos_LA = omega**2 / (2 * np.pi**2 * 8000**3)
    dos_TA = omega**2 / (2 * np.pi**2 * 5000**3)
    
    tau_dict = {'LA': tau_LA, 'TA1': tau_TA, 'TA2': tau_TA}
    vg_dict = {'LA': vg_LA, 'TA1': vg_TA, 'TA2': vg_TA}
    dos_dict = {'LA': dos_LA, 'TA1': dos_TA, 'TA2': dos_TA}
    
    bte.load_first_principles_data(omega, tau_dict, vg_dict, dos_dict)
    
    T = 300
    kappa_fp = bte.thermal_conductivity(T)
    print(f"\n基于模拟第一性原理数据的热导率: {kappa_fp:.2f} W/mK")


def main():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    example_model_comparison()
    example_branch_contributions()
    example_relaxation_time_comparison()
    example_scattering_mechanisms_branch()
    example_dispersion_correction()
    example_size_effect_mode_resolved()
    example_first_principles_interface()
    
    print("\n" + "=" * 70)
    print("所有模式分辨计算示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
