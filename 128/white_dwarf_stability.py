import numpy as np
from scipy.integrate import solve_ivp, RK45
import matplotlib.pyplot as plt
from scipy.optimize import newton, brentq

G = 6.67430e-11      
m_H = 1.67372e-27    
h = 6.62607e-34      
m_e = 9.10938e-31    
c = 2.99792e8        


def degenerate_eos(rho, mu_e=2.0):
    """
    完整的电子简并物态方程（包含相对论修正）
    P = f(rho)
    """
    n_e = rho / (mu_e * m_H)
    
    x = (h / (2 * m_e * c)) * (3 * n_e / np.pi)**(1/3)
    
    P_nonrel = (h**2 / (20 * m_e)) * (3 / np.pi)**(2/3) * n_e**(5/3)
    P_extrel = (h * c / 8) * (3 / np.pi)**(1/3) * n_e**(4/3)
    
    P = P_nonrel / np.sqrt(1 + x**2) + P_extrel * (x / np.sqrt(1 + x**2))
    
    return P


def dP_drho(rho, mu_e=2.0, eps=1e-6):
    """物态方程的导数"""
    return (degenerate_eos(rho + eps, mu_e) - degenerate_eos(rho - eps, mu_e)) / (2 * eps)


def stellar_structure_eq(r, y, mu_e=2.0):
    """
    恒星结构方程（流体静力学平衡 + 质量守恒）
    y[0] = P(r) - 压力
    y[1] = M(r) - 包围质量
    """
    P, M = y
    
    if r == 0:
        return [0.0, 0.0]
    
    if P <= 0:
        return [0.0, 0.0]
    
    rho = density_from_pressure(P, mu_e)
    
    dP_dr = -G * M * rho / (r**2)
    dM_dr = 4 * np.pi * r**2 * rho
    
    return [dP_dr, dM_dr]


def density_from_pressure(P, mu_e=2.0, max_iter=100, tol=1e-12):
    """
    通过牛顿迭代从压力反求密度
    解决 P = f(rho) 的逆问题
    """
    if P <= 0:
        return 0.0
    
    rho = 1e3
    for _ in range(max_iter):
        P_current = degenerate_eos(rho, mu_e)
        dP = dP_drho(rho, mu_e)
        
        if abs(P_current - P) < tol * P:
            break
        
        rho = rho + (P - P_current) / dP
        
        if rho < 1e-10:
            rho = 1e-10
    
    return rho


def integrate_stellar_structure(rho_c, mu_e=2.0, r_max=1e10, method='RK45', 
                                 rtol=1e-8, atol=1e-12, max_step=1e6):
    """
    自适应步长积分恒星结构方程
    从中心积分到表面（P=0）
    """
    P_c = degenerate_eos(rho_c, mu_e)
    
    def event(r, y, mu_e):
        return y[0]
    event.terminal = True
    event.direction = -1
    
    y0 = [P_c, 0.0]
    
    r_start = 1e-6
    
    sol = solve_ivp(
        stellar_structure_eq, [r_start, r_max], y0, args=(mu_e,),
        method=method, events=event, dense_output=True,
        rtol=rtol, atol=atol, max_step=max_step
    )
    
    if not sol.t_events or len(sol.t_events[0]) == 0:
        return None, None, None, None
    
    R = sol.t_events[0][0]
    M_total = sol.sol(R)[1]
    
    r_points = np.linspace(0, R, 5000)
    r_points[0] = r_start
    
    P_profile = sol.sol(r_points)[0]
    rho_profile = np.array([density_from_pressure(P, mu_e) for P in P_profile])
    
    r_points[0] = 0
    
    return R, M_total, r_points, rho_profile


def mass_at_rho_c(rho_c, mu_e=2.0):
    """给定中心密度，计算总质量（用于牛顿迭代）"""
    R, M, _, _ = integrate_stellar_structure(rho_c, mu_e)
    if M is None:
        return np.nan
    return M


def find_max_mass_newton(mu_e=2.0, rho_c_guess=1e13, tol=1e-6, max_iter=50):
    """
    使用牛顿迭代法寻找最大质量（钱德拉塞卡极限）
    寻找 dM/drho_c = 0 的点
    """
    rho_c = rho_c_guess
    
    for i in range(max_iter):
        drho = rho_c * 0.01
        
        M1 = mass_at_rho_c(rho_c - drho, mu_e)
        M2 = mass_at_rho_c(rho_c + drho, mu_e)
        
        if np.isnan(M1) or np.isnan(M2):
            rho_c *= 0.9
            continue
        
        dM_drho = (M2 - M1) / (2 * drho)
        
        M3 = mass_at_rho_c(rho_c + 2*drho, mu_e)
        dM_drho2 = (M3 - mass_at_rho_c(rho_c, mu_e)) / (2 * drho)
        d2M_drho2 = (dM_drho2 - dM_drho) / (2 * drho)
        
        if abs(d2M_drho2) < 1e-30:
            break
            
        delta_rho = -dM_drho / d2M_drho2
        
        if abs(delta_rho) < tol * rho_c:
            break
            
        rho_c_new = rho_c + delta_rho
        
        if rho_c_new < 1e10:
            rho_c_new = 1e10
        if rho_c_new > 1e15:
            rho_c_new = 1e15
            
        rho_c = rho_c_new
    
    M_max = mass_at_rho_c(rho_c, mu_e)
    
    return M_max, rho_c


def find_max_mass_brent(mu_e=2.0, rho_min=1e10, rho_max=1e14, num_points=50):
    """
    使用扫描法寻找最大质量点
    更稳定但稍慢
    """
    rho_c_list = np.logspace(np.log10(rho_min), np.log10(rho_max), num_points)
    M_list = []
    
    for rho_c in rho_c_list:
        M = mass_at_rho_c(rho_c, mu_e)
        M_list.append(M)
    
    M_list = np.array(M_list)
    valid_idx = ~np.isnan(M_list)
    rho_c_list = rho_c_list[valid_idx]
    M_list = M_list[valid_idx]
    
    if len(M_list) == 0:
        return None, None
    
    max_idx = np.argmax(M_list)
    
    return M_list[max_idx], rho_c_list[max_idx]


def mass_radius_relation_stable(rho_c_min=1e8, rho_c_max=5e13, num_points=30, mu_e=2.0):
    """
    计算稳定的质量-半径关系
    自动检测并终止在不稳定区域之前
    """
    rho_c_list = np.logspace(np.log10(rho_c_min), np.log10(rho_c_max), num_points)
    R_list = []
    M_list = []
    valid_rho_c = []
    
    for i, rho_c in enumerate(rho_c_list):
        try:
            R, M, _, _ = integrate_stellar_structure(rho_c, mu_e)
            if R is not None and M is not None and R > 0 and M > 0:
                R_list.append(R)
                M_list.append(M)
                valid_rho_c.append(rho_c)
                
                if i > 3 and M_list[-1] < M_list[-2] and M_list[-2] < M_list[-3]:
                    print(f"检测到不稳定区域，在 rho_c = {rho_c:.2e} 处终止")
                    break
        except Exception as e:
            print(f"rho_c = {rho_c:.2e} 积分失败: {e}")
            break
    
    return np.array(M_list), np.array(R_list), np.array(valid_rho_c)


def lane_emden_stable(n, xi_max=20.0, dxi_min=1e-8, dxi_max=0.1, tol=1e-10):
    """
    自适应步长的Lane-Emden方程求解器
    使用嵌入式RK方法控制误差
    """
    xi = 0.0
    theta = 1.0
    dtheta = 0.0
    dxi = 1e-4
    
    xi_list = [xi]
    theta_list = [theta]
    dtheta_list = [dtheta]
    
    def derivatives(xi, theta, dtheta):
        if xi == 0:
            return dtheta, -theta**n / 3.0
        return dtheta, -2.0 * dtheta / xi - theta**n
    
    while xi < xi_max and theta > tol:
        k1_v, k1_d = derivatives(xi, theta, dtheta)
        k2_v, k2_d = derivatives(xi + dxi/2, theta + dxi/2*k1_v, dtheta + dxi/2*k1_d)
        k3_v, k3_d = derivatives(xi + dxi/2, theta + dxi/2*k2_v, dtheta + dxi/2*k2_d)
        k4_v, k4_d = derivatives(xi + dxi, theta + dxi*k3_v, dtheta + dxi*k3_d)
        
        theta_new = theta + dxi/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        dtheta_new = dtheta + dxi/6 * (k1_d + 2*k2_d + 2*k3_d + k4_d)
        
        k5_v, k5_d = derivatives(xi + dxi, theta_new, dtheta_new)
        
        theta_5th = theta + dxi/24 * (k1_v + 3*k2_v + 3*k3_v + k4_v)
        error = abs(theta_5th - theta_new)
        
        if error > 0:
            dxi_new = dxi * (tol / error)**0.2
        else:
            dxi_new = dxi * 2
            
        dxi_new = max(min(dxi_new, dxi_max), dxi_min)
        
        if error < tol or dxi <= dxi_min:
            xi = xi + dxi
            theta = theta_new
            dtheta = dtheta_new
            xi_list.append(xi)
            theta_list.append(theta)
            dtheta_list.append(dtheta)
        
        dxi = dxi_new
    
    xi_array = np.array(xi_list)
    theta_array = np.array(theta_list)
    dtheta_array = np.array(dtheta_list)
    
    if theta_array[-1] < 0:
        idx = np.where(theta_array > 0)[0]
        if len(idx) > 1:
            i1, i2 = idx[-1], idx[-1] + 1
            if i2 < len(theta_array):
                t1, t2 = theta_array[i1], theta_array[i2]
                x1, x2 = xi_array[i1], xi_array[i2]
                xi1 = x1 - t1 * (x2 - x1) / (t2 - t1)
                dtheta1 = dtheta_array[i1] + (dtheta_array[i2] - dtheta_array[i1]) * (xi1 - x1) / (x2 - x1)
            else:
                xi1 = xi_array[idx[-1]]
                dtheta1 = dtheta_array[idx[-1]]
        else:
            xi1 = xi_array[-1]
            dtheta1 = dtheta_array[-1]
    else:
        xi1 = xi_array[-1]
        dtheta1 = dtheta_array[-1]
    
    return xi_array, theta_array, dtheta_array, xi1, dtheta1


def plot_improved_results():
    """绘制改进后的结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    mu_e = 2.0
    
    print("正在计算稳定的质量-半径关系（完整物态方程）...")
    M_list, R_list, rho_c_list = mass_radius_relation_stable(1e8, 1e14, 25, mu_e)
    
    print("正在计算钱德拉塞卡极限质量...")
    M_ch, rho_c_ch = find_max_mass_brent(mu_e, 1e12, 5e14, 30)
    print(f"钱德拉塞卡极限: {M_ch/1.989e30:.4f} M☉, 对应中心密度: {rho_c_ch:.2e} kg/m³")
    
    M_sun = M_list / 1.989e30
    R_earth = R_list / 6.371e6
    
    axes[0, 0].plot(M_sun, R_earth, 'b-o', linewidth=2, markersize=5, label='完整物态方程')
    if M_ch is not None:
        axes[0, 0].axvline(x=M_ch/1.989e30, color='red', linestyle='--', linewidth=2,
                          label=f'钱德拉塞卡极限 = {M_ch/1.989e30:.3f} M☉')
    axes[0, 0].set_xlabel('质量 (M☉)', fontsize=12)
    axes[0, 0].set_ylabel('半径 (R⊕)', fontsize=12)
    axes[0, 0].set_title('质量-半径关系（完整物态方程）', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(rho_c_list, M_sun, 'g-o', linewidth=2, markersize=5)
    if M_ch is not None and rho_c_ch is not None:
        axes[0, 1].plot(rho_c_ch, M_ch/1.989e30, 'r*', markersize=15, label='最大质量点')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('中心密度 ρ_c (kg/m³)', fontsize=12)
    axes[0, 1].set_ylabel('质量 (M☉)', fontsize=12)
    axes[0, 1].set_title('质量-中心密度关系', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    rho_c_examples = [1e9, 1e11, 1e13]
    colors = ['blue', 'green', 'red']
    labels = [f'ρ_c = 10⁹ kg/m³', f'ρ_c = 10¹¹ kg/m³', f'ρ_c = 10¹³ kg/m³']
    
    for i, rho_c in enumerate(rho_c_examples):
        print(f"正在积分 ρ_c = {rho_c:.1e}...")
        R, M, r, rho = integrate_stellar_structure(rho_c, mu_e)
        if R is not None:
            r_norm = r / R
            rho_norm = rho / rho_c
            axes[1, 0].plot(r_norm, rho_norm, color=colors[i], linewidth=2, 
                          label=labels[i] + f', M={M/1.989e30:.3f} M☉')
    
    axes[1, 0].set_xlabel('r/R', fontsize=12)
    axes[1, 0].set_ylabel('ρ/ρ_c', fontsize=12)
    axes[1, 0].set_title('密度轮廓（完整物态方程）', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    n_values = [1.5, 3.0]
    labels_le = ['n=1.5', 'n=3.0']
    for i, n in enumerate(n_values):
        xi, theta, dtheta, xi1, dtheta1 = lane_emden_stable(n)
        axes[1, 1].plot(xi, theta, linewidth=2, label=f'{labels_le[i]}, ξ₁={xi1:.3f}')
    
    axes[1, 1].set_xlabel('ξ', fontsize=12)
    axes[1, 1].set_ylabel('θ(ξ)', fontsize=12)
    axes[1, 1].set_title('Lane-Emden方程（自适应步长）', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.savefig('white_dwarf_stability.png', dpi=150, bbox_inches='tight')
    print("\n图像已保存为: white_dwarf_stability.png")
    
    return fig, M_ch, rho_c_ch


def print_comparison():
    """打印不同方法的对比"""
    mu_e = 2.0
    
    print("\n" + "="*70)
    print("方法对比: 钱德拉塞卡极限质量计算")
    print("="*70)
    
    print("\n1. 经典Lane-Emden n=3 解析结果:")
    _, _, _, xi1, dtheta1 = lane_emden_stable(3.0)
    M_ch_le = (np.sqrt(6) / 32) * (np.pi * c**3 / G**(3/2)) * \
              (h / (2 * np.pi * m_H))**(3/2) * (1 / mu_e**2) * (-xi1**2 * dtheta1)
    print(f"   M_Ch = {M_ch_le/1.989e30:.4f} M☉")
    
    print("\n2. 完整物态方程 + 扫描法:")
    M_ch_brent, rho_c_brent = find_max_mass_brent(mu_e, 1e12, 1e14, 20)
    if M_ch_brent is not None:
        print(f"   M_Ch = {M_ch_brent/1.989e30:.4f} M☉")
        print(f"   对应 ρ_c = {rho_c_brent:.2e} kg/m³")
    
    print("\n3. 部分中心密度的白矮星性质:")
    print(f"{'ρ_c (kg/m³)':<15} {'R (km)':<12} {'M (M☉)':<12} {'ρ_c/ρ_avg':<12}")
    print("-"*60)
    
    for rho_c in [1e8, 1e9, 1e10, 1e11, 1e12]:
        R, M, r, rho = integrate_stellar_structure(rho_c, mu_e)
        if R is not None:
            rho_avg = 3*M/(4*np.pi*R**3)
            print(f"{rho_c:<15.1e} {R/1000:<12.1f} {M/1.989e30:<12.4f} {rho_c/rho_avg:<12.1f}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print("改进版白矮星结构求解器")
    print("="*50)
    print("特性:")
    print("- 完整的电子简并物态方程（含相对论修正）")
    print("- 自适应步长积分器")
    print("- 稳定性检测与自动终止")
    print("- 牛顿迭代法求最大质量")
    print("="*50)
    
    print_comparison()
    
    print("正在生成结果图像...")
    fig, M_ch, rho_c_ch = plot_improved_results()
    
    print("\n完成!")
    plt.show()
