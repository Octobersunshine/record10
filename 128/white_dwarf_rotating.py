import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf
import matplotlib.pyplot as plt

G = 6.67430e-11      
m_H = 1.67372e-27    
h = 6.62607e-34      
m_e = 9.10938e-31    
c = 2.99792e8        
k_B = 1.38065e-23    
eV_to_J = 1.60218e-19


def fermi_integral_half(eta):
    """
    1/2阶费米积分 F_{1/2}(η)
    使用解析近似
    """
    if eta < -5:
        return np.exp(eta)
    elif eta > 5:
        return (2/(3*np.sqrt(np.pi))) * eta**(3/2) * (1 + 5*np.pi**2/(8*eta**2))
    else:
        x = np.linspace(0, 20, 2000)
        integrand = np.sqrt(x) / (1 + np.exp(x - eta))
        return np.trapz(integrand, x)


def fermi_integral_3half(eta):
    """
    3/2阶费米积分 F_{3/2}(η)
    """
    if eta < -5:
        return np.exp(eta)
    elif eta > 5:
        return (2/(5*np.sqrt(np.pi))) * eta**(5/2) * (1 + 5*np.pi**2/(8*eta**2))
    else:
        x = np.linspace(0, 20, 2000)
        integrand = x**(3/2) / (1 + np.exp(x - eta))
        return np.trapz(integrand, x)


def pressure_thermal_electron(n_e, T):
    """
    有限温度电子压力（包含简并压和热压）
    使用Sommerfeld展开近似
    n_e: 电子数密度
    T: 温度
    """
    if T <= 0:
        n_e_clipped = max(n_e, 1e10)
        p_F = h * (3 * n_e_clipped / (8 * np.pi))**(1/3)
        x_F = p_F / (m_e * c)
        
        if x_F < 0.1:
            return (h**2 / (20 * m_e)) * (3 / np.pi)**(2/3) * n_e_clipped**(5/3)
        else:
            return (h * c / 8) * (3 / np.pi)**(1/3) * n_e_clipped**(4/3)
    
    n_e_clipped = max(n_e, 1e10)
    
    e_F = (h**2 / (8 * m_e)) * (3 * n_e_clipped / np.pi)**(2/3)
    kT = k_B * T
    
    if kT <= 0:
        eta = 100.0
    else:
        eta = e_F / kT
    
    if eta < -3:
        P = n_e_clipped * k_B * T
    elif eta < 30:
        F_3_2 = fermi_integral_3half(eta)
        P = (8 * np.pi * (2 * m_e)**(3/2) / (3 * h**3)) * (kT)**(5/2) * F_3_2
    else:
        p_F = h * (3 * n_e_clipped / (8 * np.pi))**(1/3)
        x_F = p_F / (m_e * c)
        
        if x_F < 0.1:
            P_deg = (h**2 / (20 * m_e)) * (3 / np.pi)**(2/3) * n_e_clipped**(5/3)
        else:
            P_deg = (h * c / 8) * (3 / np.pi)**(1/3) * n_e_clipped**(4/3)
        
        P_thermal = (np.pi**2 / 12) * (kT**2 / e_F) * n_e_clipped * k_B * T
        P = P_deg + P_thermal
    
    return P


def pressure_ionic(n_i, T, Z=6):
    """离子热压（理想气体）"""
    return n_i * k_B * T


def pressure_radiation(rho, T):
    """辐射压"""
    a = 4 * 5.67037e-8 / c
    return a * T**4 / 3


def eos_full(rho, T, mu_e=2.0, Z=6):
    """
    完整物态方程：电子简并压 + 电子热压 + 离子热压 + 辐射压
    """
    n_e = rho / (mu_e * m_H)
    n_i = rho / (Z * m_H)
    
    P_e = pressure_thermal_electron(n_e, T)
    P_i = pressure_ionic(n_i, T, Z)
    P_rad = pressure_radiation(rho, T)
    
    P_total = P_e + P_i + P_rad
    
    return P_total


def internal_energy_electron(n_e, T):
    """电子内能"""
    n_e_clipped = max(n_e, 1e10)
    e_F = (h**2 / (8 * m_e)) * (3 * n_e_clipped / np.pi)**(2/3)
    kT = k_B * T
    
    if kT <= 0:
        return (3/5) * n_e_clipped * e_F
    
    eta = e_F / kT
    
    if eta > 30:
        u_deg = (3/5) * n_e_clipped * e_F
        u_thermal = (np.pi**2 / 4) * n_e_clipped * kT**2 / e_F * k_B * T
        return u_deg + u_thermal
    else:
        F_3_2 = fermi_integral_3half(eta)
        F_5_2 = fermi_integral_5half(eta)
        return (8 * np.pi * (2 * m_e)**(3/2) / (h**3)) * (kT)**(5/2) * (3/2) * F_5_2


def fermi_integral_5half(eta):
    """5/2阶费米积分"""
    if eta < -5:
        return np.exp(eta)
    elif eta > 5:
        return (2/(7*np.sqrt(np.pi))) * eta**(7/2) * (1 + 35*np.pi**2/(8*eta**2))
    else:
        x = np.linspace(0, 20, 2000)
        integrand = x**(5/2) / (1 + np.exp(x - eta))
        return np.trapz(integrand, x)


def rho_from_pressure_T(P, T, mu_e=2.0, Z=6, max_iter=50, tol=1e-10):
    """从P和T反求rho（牛顿迭代）"""
    if P <= 0:
        return 0.0
    
    rho = 1e3
    for _ in range(max_iter):
        P_current = eos_full(rho, T, mu_e, Z)
        dP_drho = (eos_full(rho * 1.001, T, mu_e, Z) - eos_full(rho * 0.999, T, mu_e, Z)) / (0.002 * rho)
        
        if abs(P_current - P) < tol * P:
            break
        
        if dP_drho > 0:
            rho = rho + (P - P_current) / dP_drho
        
        if rho < 1e-10:
            rho = 1e-10
    
    return rho


def rotating_structure_eq(r, y, Omega=0.0, T=1e7, mu_e=2.0, Z=6):
    """
    旋转白矮星结构方程（球对称近似 + 离心力修正）
    y[0] = P(r) - 压力
    y[1] = M(r) - 包围质量
    """
    P, M = y
    
    if r == 0:
        return [0.0, 0.0]
    
    if P <= 0:
        return [0.0, 0.0]
    
    rho = rho_from_pressure_T(P, T, mu_e, Z)
    
    if rho <= 0:
        return [0.0, 0.0]
    
    g_eff = G * M / r**2 - Omega**2 * r
    
    if g_eff <= 0:
        return [0.0, 0.0]
    
    dP_dr = -rho * g_eff
    dM_dr = 4 * np.pi * r**2 * rho
    
    return [dP_dr, dM_dr]


def integrate_rotating(rho_c, Omega=0.0, T=1e7, mu_e=2.0, Z=6, r_max=2e7):
    """积分旋转白矮星结构"""
    P_c = eos_full(rho_c, T, mu_e, Z)
    
    def event(r, y, Omega, T, mu_e, Z):
        return y[0]
    event.terminal = True
    event.direction = -1
    
    def event_centrifugal(r, y, Omega, T, mu_e, Z):
        P, M = y
        if r < 1e-6 or M < 1e20:
            return 1.0
        return G * M / r**2 - Omega**2 * r
    event_centrifugal.terminal = True
    event_centrifugal.direction = -1
    
    y0 = [P_c, 0.0]
    r_start = 1e-6
    
    try:
        sol = solve_ivp(
            rotating_structure_eq, [r_start, r_max], y0, args=(Omega, T, mu_e, Z),
            method='RK45', events=[event, event_centrifugal], dense_output=True,
            rtol=1e-8, atol=1e-12, max_step=1e4
        )
        
        if not sol.t_events or (len(sol.t_events[0]) == 0 and len(sol.t_events[1]) == 0):
            return None, None, None, None
        
        if len(sol.t_events[0]) > 0:
            R = sol.t_events[0][0]
        else:
            R = sol.t_events[1][0]
        
        M_total = sol.sol(R)[1]
        
        r_points = np.linspace(0, R, 2000)
        r_points[0] = r_start
        
        P_profile = sol.sol(r_points)[0]
        rho_profile = np.array([rho_from_pressure_T(P, T, mu_e, Z) for P in P_profile])
        
        r_points[0] = 0
        
        return R, M_total, r_points, rho_profile
    except:
        return None, None, None, None


def critical_angular_velocity(M, R_polar):
    """
    临界角速度（赤道质量流失极限）
    Kepler角速度
    """
    if R_polar <= 0 or M <= 0:
        return 0
    return np.sqrt(G * M / R_polar**3)


def mass_shedding_limit(rho_c, T=1e7, mu_e=2.0, Z=6, tol=1e-4, max_iter=30):
    """
    用迭代法求给定中心密度下的质量流失极限
    返回: M_max, Omega_c, R_polar, R_eq
    """
    R0, M0, _, _ = integrate_rotating(rho_c, 0.0, T, mu_e, Z)
    
    if R0 is None or M0 is None:
        return None, None, None, None
    
    Omega_kepler_est = np.sqrt(G * M0 / R0**3)
    
    Omega_low = 0.0
    Omega_high = Omega_kepler_est * 1.5
    
    for _ in range(max_iter):
        Omega_mid = (Omega_low + Omega_high) / 2
        R, M, _, _ = integrate_rotating(rho_c, Omega_mid, T, mu_e, Z)
        
        if R is None:
            Omega_high = Omega_mid
            continue
        
        Omega_kepler = np.sqrt(G * M / R**3)
        
        if abs(Omega_mid - Omega_kepler) < tol * Omega_kepler:
            R_eq = R * (Omega_mid / Omega_kepler_est)**(2/3)
            return M, Omega_mid, R, R_eq
        
        if Omega_mid < Omega_kepler:
            Omega_low = Omega_mid
        else:
            Omega_high = Omega_mid
    
    R, M, _, _ = integrate_rotating(rho_c, Omega_low, T, mu_e, Z)
    return M, Omega_low, R, R


def rotating_mass_radius_relation(rho_c_min=1e8, rho_c_max=1e13, num_points=15, 
                                   T=1e7, mu_e=2.0, Z=6):
    """计算旋转白矮星的质量-半径关系"""
    rho_c_list = np.logspace(np.log10(rho_c_min), np.log10(rho_c_max), num_points)
    
    results = {
        'rho_c': [],
        'M_static': [],
        'R_static': [],
        'M_shedding': [],
        'Omega_c': [],
        'R_polar': [],
        'R_eq': []
    }
    
    for i, rho_c in enumerate(rho_c_list):
        print(f"正在计算 rho_c = {rho_c:.2e} ({i+1}/{num_points})...")
        
        R0, M0, _, _ = integrate_rotating(rho_c, 0.0, T, mu_e, Z)
        
        if R0 is None or M0 is None:
            continue
        
        M_shed, Omega_c, R_p, R_eq = mass_shedding_limit(rho_c, T, mu_e, Z)
        
        results['rho_c'].append(rho_c)
        results['M_static'].append(M0)
        results['R_static'].append(R0)
        results['M_shedding'].append(M_shed if M_shed else np.nan)
        results['Omega_c'].append(Omega_c if Omega_c else np.nan)
        results['R_polar'].append(R_p if R_p else np.nan)
        results['R_eq'].append(R_eq if R_eq else np.nan)
    
    for key in results:
        results[key] = np.array(results[key])
    
    return results


def temperature_effects(rho_c=1e9, T_list=[1e6, 1e7, 1e8, 1e9], mu_e=2.0, Z=6):
    """计算不同温度下的白矮星性质"""
    results = {
        'T': [],
        'M': [],
        'R': []
    }
    
    for T in T_list:
        R, M, _, _ = integrate_rotating(rho_c, 0.0, T, mu_e, Z)
        if R and M:
            results['T'].append(T)
            results['M'].append(M)
            results['R'].append(R)
    
    for key in results:
        results[key] = np.array(results[key])
    
    return results


def plot_rotating_results(results, temp_results=None):
    """绘制旋转和温度效应的结果"""
    fig = plt.figure(figsize=(16, 12))
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    valid = ~np.isnan(results['M_static'])
    M_sun_static = results['M_static'][valid] / 1.989e30
    R_earth_static = results['R_static'][valid] / 6.371e6
    ax1.plot(M_sun_static, R_earth_static, 'b-o', linewidth=2, markersize=5, label='静态')
    
    valid_shed = ~np.isnan(results['M_shedding'])
    M_sun_shed = results['M_shedding'][valid_shed] / 1.989e30
    R_earth_polar = results['R_polar'][valid_shed] / 6.371e6
    ax1.plot(M_sun_shed, R_earth_polar, 'r-o', linewidth=2, markersize=5, label='质量流失极限')
    
    ax1.set_xlabel('质量 (M☉)', fontsize=12)
    ax1.set_ylabel('极半径 (R⊕)', fontsize=12)
    ax1.set_title('质量-半径关系', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    M_enhancement = (results['M_shedding'][valid_shed] - results['M_static'][valid_shed]) / results['M_static'][valid_shed] * 100
    ax2.plot(M_sun_shed, M_enhancement, 'g-o', linewidth=2, markersize=5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('静态质量 (M☉)', fontsize=12)
    ax2.set_ylabel('质量提升 (%)', fontsize=12)
    ax2.set_title('旋转对质量上限的提升', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    period_min = 2 * np.pi / results['Omega_c'][valid_shed] / 60
    ax3.plot(M_sun_shed, period_min, 'm-o', linewidth=2, markersize=5)
    ax3.set_xlabel('质量 (M☉)', fontsize=12)
    ax3.set_ylabel('旋转周期 (分钟)', fontsize=12)
    ax3.set_title('临界旋转周期', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    rho_c_valid = results['rho_c'][valid_shed]
    flattening = (results['R_eq'][valid_shed] - results['R_polar'][valid_shed]) / results['R_eq'][valid_shed]
    ax4.semilogx(rho_c_valid, flattening * 100, 'c-o', linewidth=2, markersize=5)
    ax4.set_xlabel('中心密度 (kg/m³)', fontsize=12)
    ax4.set_ylabel('扁率 (%)', fontsize=12)
    ax4.set_title('星体扁率', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    if temp_results is not None and len(temp_results['T']) > 0:
        ax5 = fig.add_subplot(gs[1, 1])
        T_K = temp_results['T']
        M_T = temp_results['M'] / 1.989e30
        ax5.semilogx(T_K, M_T, 'r-o', linewidth=2, markersize=5)
        ax5.set_xlabel('温度 (K)', fontsize=12)
        ax5.set_ylabel('质量 (M☉)', fontsize=12)
        ax5.set_title('温度对质量的影响', fontsize=14)
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        R_T = temp_results['R'] / 6.371e6
        ax6.semilogx(T_K, R_T, 'b-o', linewidth=2, markersize=5)
        ax6.set_xlabel('温度 (K)', fontsize=12)
        ax6.set_ylabel('半径 (R⊕)', fontsize=12)
        ax6.set_title('温度对半径的影响', fontsize=14)
        ax6.grid(True, alpha=0.3)
    else:
        ax5 = fig.add_subplot(gs[1, 1:])
        rho_c_example = 1e9
        Omega_list = [0, 0.3, 0.6, 0.9]
        colors = ['blue', 'green', 'orange', 'red']
        
        for Omega_factor, color in zip(Omega_list, colors):
            _, _, r, rho = integrate_rotating(rho_c_example, 
                                               Omega_factor * 0.01, 1e7, 2.0, 6)
            if r is not None and len(r) > 1:
                r_norm = r / r[-1]
                rho_norm = rho / rho_c_example
                ax5.plot(r_norm, rho_norm, color=color, linewidth=2,
                        label=f'Ω={Omega_factor*100:.0f}%')
        
        ax5.set_xlabel('r/R', fontsize=12)
        ax5.set_ylabel('ρ/ρ_c', fontsize=12)
        ax5.set_title('旋转对密度轮廓的影响', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.savefig('white_dwarf_rotating.png', dpi=150, bbox_inches='tight')
    print("\n图像已保存为: white_dwarf_rotating.png")
    
    return fig


def print_summary(results, temp_results=None):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("旋转白矮星计算结果摘要")
    print("="*80)
    
    valid = ~np.isnan(results['M_shedding'])
    if np.any(valid):
        max_idx = np.argmax(results['M_shedding'][valid])
        
        print(f"\n最大质量（质量流失极限）:")
        print(f"  M_max = {results['M_shedding'][valid][max_idx]/1.989e30:.3f} M☉")
        print(f"  对应中心密度: {results['rho_c'][valid][max_idx]:.2e} kg/m³")
        print(f"  临界角速度: {results['Omega_c'][valid][max_idx]:.4f} rad/s")
        print(f"  临界周期: {2*np.pi/results['Omega_c'][valid][max_idx]/60:.2f} 分钟")
        print(f"  极半径: {results['R_polar'][valid][max_idx]/1000:.1f} km")
        print(f"  赤道半径: {results['R_eq'][valid][max_idx]/1000:.1f} km")
        print(f"  扁率: {(results['R_eq'][valid][max_idx]-results['R_polar'][valid][max_idx])/results['R_eq'][valid][max_idx]*100:.1f}%")
        
        M_gain = (results['M_shedding'][valid][max_idx] - results['M_static'][valid][max_idx]) / results['M_static'][valid][max_idx] * 100
        print(f"  相比静态质量提升: {M_gain:.1f}%")
    
    print(f"\n静态钱德拉塞卡极限: {np.nanmax(results['M_static'])/1.989e30:.3f} M☉")
    
    if temp_results is not None and len(temp_results['T']) > 0:
        print(f"\n温度效应 (ρ_c = 10⁹ kg/m³):")
        for i in range(len(temp_results['T'])):
            print(f"  T = {temp_results['T'][i]:.1e} K, "
                  f"M = {temp_results['M'][i]/1.989e30:.3f} M☉, "
                  f"R = {temp_results['R'][i]/1000:.1f} km")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("旋转白矮星结构求解器")
    print("="*50)
    print("特性:")
    print("- 有限温度物态方程（简并压+热压+辐射压）")
    print("- 旋转效应（离心力修正）")
    print("- 质量流失极限计算")
    print("- 温度效应分析")
    print("="*50 + "\n")
    
    T = 1e7
    mu_e = 2.0
    Z = 6
    
    print(f"计算参数:")
    print(f"  温度 T = {T:.1e} K")
    print(f"  电子平均分子量 μ_e = {mu_e}")
    print(f"  原子序数 Z = {Z} (C/O核)")
    print()
    
    print("正在计算质量-半径关系（包含旋转效应）...")
    results = rotating_mass_radius_relation(1e8, 5e12, 12, T, mu_e, Z)
    
    print("\n正在计算温度效应...")
    temp_results = temperature_effects(1e9, [1e6, 5e6, 1e7, 5e7, 1e8], mu_e, Z)
    
    print("\n正在生成图像...")
    fig = plot_rotating_results(results, temp_results)
    
    print_summary(results, temp_results)
    
    print("完成!")
    plt.show()
