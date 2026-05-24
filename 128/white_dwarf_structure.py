import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lane_emden_equation(xi, y, n):
    """
    Lane-Emden方程: 
    d²θ/dξ² + (2/ξ) dθ/dξ + θ^n = 0
    转化为一阶方程组:
    y[0] = θ
    y[1] = dθ/dξ
    """
    theta, dtheta_dxi = y
    if xi == 0:
        d2theta_dxi2 = -theta**n / 3.0
    else:
        d2theta_dxi2 = -2.0 * dtheta_dxi / xi - theta**n
    return [dtheta_dxi, d2theta_dxi2]

def solve_lane_emden(n, xi_max=20.0, num_points=10000):
    """
    求解Lane-Emden方程
    n: 多方指数
    返回: xi数组, theta数组, dtheta/dxi数组, xi1 (表面位置)
    """
    y0 = [1.0, 0.0]  
    
    def event(xi, y, n):
        return y[0]
    event.terminal = True
    event.direction = -1
    
    sol = solve_ivp(
        lane_emden_equation, [0, xi_max], y0, args=(n,),
        method='RK45', events=event, dense_output=True,
        rtol=1e-10, atol=1e-12
    )
    
    xi = np.linspace(0, sol.t[-1], num_points)
    theta = sol.sol(xi)[0]
    dtheta_dxi = sol.sol(xi)[1]
    
    xi1 = sol.t_events[0][0]
    dtheta_dxi1 = sol.sol(xi1)[1]
    
    return xi, theta, dtheta_dxi, xi1, dtheta_dxi1

def white_dwarf_properties(n, rho_c, mu_e=2.0):
    """
    计算白矮星的物理性质
    n: 多方指数
    rho_c: 中心密度 (kg/m³)
    mu_e: 电子平均分子量 (对于He: 2.0, 对于C/O: 2.0)
    返回: 半径R, 质量M, 密度轮廓rho(r)
    """
    xi, theta, dtheta_dxi, xi1, dtheta_dxi1 = solve_lane_emden(n)
    
    G = 6.67430e-11  
    m_H = 1.67372e-27  
    h = 6.62607e-34  
    m_e = 9.10938e-31  
    c = 2.99792e8  
    
    if n == 1.5:  
        K = (h**2 / (20 * m_e)) * (3 / np.pi)**(2/3) * (1 / (mu_e * m_H))**(5/3)
        alpha_n = np.sqrt((n + 1) * K / (4 * np.pi * G)) * rho_c**((1 - n) / (2 * n))
    elif n == 3.0:  
        K = (h * c / 8) * (3 / np.pi)**(1/3) * (1 / (mu_e * m_H))**(4/3)
        alpha_n = np.sqrt((n + 1) * K / (4 * np.pi * G)) * rho_c**((1 - n) / (2 * n))
    else:
        K = 1e10  
        alpha_n = np.sqrt((n + 1) * K / (4 * np.pi * G)) * rho_c**((1 - n) / (2 * n))
    
    R = alpha_n * xi1  
    M = 4 * np.pi * alpha_n**3 * rho_c * (-xi1**2 * dtheta_dxi1)
    
    r = alpha_n * xi
    rho = rho_c * theta**n
    
    return R, M, r, rho

def chandrasekhar_mass(mu_e=2.0):
    """
    计算钱德拉塞卡极限质量
    """
    h = 6.62607e-34  
    c = 2.99792e8    
    G = 6.67430e-11  
    m_H = 1.67372e-27  
    
    _, _, _, xi1, dtheta_dxi1 = solve_lane_emden(3.0)
    
    M_ch = (np.sqrt(6) / 32) * (np.pi * c**3 / G**(3/2)) * \
           (h / (2 * np.pi * m_H))**(3/2) * (1 / mu_e**2) * \
           (-xi1**2 * dtheta_dxi1)
    
    return M_ch

def mass_radius_relation(n, rho_c_min, rho_c_max, num_points=50, mu_e=2.0):
    """
    计算质量-半径关系
    """
    rho_c_list = np.logspace(np.log10(rho_c_min), np.log10(rho_c_max), num_points)
    R_list = []
    M_list = []
    
    for rho_c in rho_c_list:
        try:
            R, M, _, _ = white_dwarf_properties(n, rho_c, mu_e)
            R_list.append(R)
            M_list.append(M)
        except:
            pass
    
    return np.array(M_list), np.array(R_list), rho_c_list

def plot_results():
    """
    绘制所有结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    n_values = [1.5, 3.0]
    labels = ['n=1.5 (非相对论)', 'n=3.0 (极端相对论)']
    colors = ['blue', 'red']
    
    rho_c_example = 1e9  
    mu_e = 2.0
    
    for i, n in enumerate(n_values):
        xi, theta, dtheta_dxi, xi1, dtheta_dxi1 = solve_lane_emden(n)
        axes[0, 0].plot(xi, theta, label=labels[i], color=colors[i], linewidth=2)
        axes[0, 1].plot(xi, theta**n, label=labels[i], color=colors[i], linewidth=2)
    
    axes[0, 0].set_xlabel(r'$\xi$', fontsize=12)
    axes[0, 0].set_ylabel(r'$\theta(\xi)$', fontsize=12)
    axes[0, 0].set_title('Lane-Emden方程解', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    axes[0, 1].set_xlabel(r'$\xi$', fontsize=12)
    axes[0, 1].set_ylabel(r'$\rho/\rho_c = \theta^n$', fontsize=12)
    axes[0, 1].set_title('归一化密度轮廓', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for i, n in enumerate(n_values):
        M_list, R_list, rho_c_list = mass_radius_relation(
            n, 1e7, 1e13, 50, mu_e
        )
        M_sun = M_list / 1.989e30
        R_earth = R_list / 6.371e6
        axes[1, 0].plot(M_sun, R_earth, label=labels[i], color=colors[i], linewidth=2, marker='o', markersize=4)
    
    M_ch = chandrasekhar_mass(mu_e)
    M_ch_sun = M_ch / 1.989e30
    axes[1, 0].axvline(x=M_ch_sun, color='green', linestyle='--', linewidth=2, 
                       label=f'钱德拉塞卡极限 = {M_ch_sun:.3f} M☉')
    
    axes[1, 0].set_xlabel('质量 (M☉)', fontsize=12)
    axes[1, 0].set_ylabel('半径 (R⊕)', fontsize=12)
    axes[1, 0].set_title('白矮星质量-半径关系', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    n = 1.5
    R, M, r, rho = white_dwarf_properties(n, rho_c_example, mu_e)
    r_normalized = r / R
    rho_normalized = rho / rho_c_example
    axes[1, 1].plot(r_normalized, rho_normalized, 'b-', linewidth=2, label=f'n=1.5, ρ_c={rho_c_example:.1e} kg/m³')
    
    n = 3.0
    R, M, r, rho = white_dwarf_properties(n, rho_c_example, mu_e)
    r_normalized = r / R
    rho_normalized = rho / rho_c_example
    axes[1, 1].plot(r_normalized, rho_normalized, 'r-', linewidth=2, label=f'n=3.0, ρ_c={rho_c_example:.1e} kg/m³')
    
    axes[1, 1].set_xlabel('r/R', fontsize=12)
    axes[1, 1].set_ylabel('ρ/ρ_c', fontsize=12)
    axes[1, 1].set_title('密度轮廓 (物理坐标)', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.savefig('white_dwarf_structure.png', dpi=150, bbox_inches='tight')
    print("图像已保存为: white_dwarf_structure.png")
    
    return fig

def print_example_properties():
    """
    打印示例白矮星的性质
    """
    rho_c_list = [1e8, 1e9, 1e10, 1e12]  
    mu_e = 2.0
    
    print("\n" + "="*70)
    print("白矮星性质示例 (非相对论, n=1.5)")
    print("="*70)
    print(f"{'中心密度 (kg/m³)':<20} {'质量 (M☉)':<15} {'半径 (R⊕)':<15} {'中心压力 (Pa)':<20}")
    print("-"*70)
    
    for rho_c in rho_c_list:
        R, M, r, rho = white_dwarf_properties(1.5, rho_c, mu_e)
        M_sun = M / 1.989e30
        R_earth = R / 6.371e6
        P_c = (1.0 / 20.0) * (6.62607e-34**2 / 9.10938e-31) * (3 / np.pi)**(2/3) * \
              (rho_c / (2.0 * 1.67372e-27))**(5/3)
        print(f"{rho_c:<20.1e} {M_sun:<15.3f} {R_earth:<15.3f} {P_c:<20.2e}")
    
    M_ch = chandrasekhar_mass(mu_e)
    print("\n" + "="*70)
    print(f"钱德拉塞卡极限质量 (μ_e={mu_e}): {M_ch/1.989e30:.4f} M☉")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("求解白矮星结构方程 (Lane-Emden方程)")
    print("="*50)
    
    print_example_properties()
    
    print("正在生成图像...")
    plot_results()
    print("\n完成!")
    plt.show()
