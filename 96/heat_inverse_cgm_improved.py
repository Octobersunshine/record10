import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d


def solve_heat_direct_cn(L, T_total, alpha, Nx, Nt, q, T_left, T_right, order=2):
    """
    求解一维热传导正问题 - Crank-Nicolson格式（二阶时间精度）
    支持一阶和二阶边界条件离散
    
    参数:
        L: 空间域长度
        T_total: 总时间
        alpha: 热扩散系数
        Nx: 空间网格数
        Nt: 时间步数
        q: 边界热流 (x=0处)，时间序列
        T_left: x=0处初始温度
        T_right: x=L处初始温度
        order: 边界条件精度阶数 (1或2)
    
    返回:
        x: 空间坐标
        t: 时间坐标
        T: 温度场 T(x,t)
    """
    dx = L / (Nx - 1)
    dt = T_total / (Nt - 1)
    
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T_total, Nt)
    
    T = np.zeros((Nx, Nt))
    T[:, 0] = T_left + (T_right - T_left) * x / L
    
    Fo = alpha * dt / dx**2
    
    if order == 1:
        main_diag_A = np.ones(Nx) * (1 + Fo)
        upper_diag_A = np.ones(Nx - 1) * (-Fo/2)
        lower_diag_A = np.ones(Nx - 1) * (-Fo/2)
        main_diag_A[0] = 1 + Fo
        main_diag_A[-1] = 1
        upper_diag_A[0] = -Fo/2
        lower_diag_A[-1] = 0
        
        main_diag_B = np.ones(Nx) * (1 - Fo)
        upper_diag_B = np.ones(Nx - 1) * (Fo/2)
        lower_diag_B = np.ones(Nx - 1) * (Fo/2)
        main_diag_B[0] = 1 - Fo
        main_diag_B[-1] = 1
        upper_diag_B[0] = Fo/2
        lower_diag_B[-1] = 0
        
    else:
        main_diag_A = np.ones(Nx) * (1 + Fo)
        upper_diag_A = np.ones(Nx - 1) * (-Fo/2)
        lower_diag_A = np.ones(Nx - 1) * (-Fo/2)
        main_diag_A[0] = 1 + 2*Fo
        main_diag_A[-1] = 1
        upper_diag_A[0] = -2*Fo/2
        lower_diag_A[-1] = 0
        
        main_diag_B = np.ones(Nx) * (1 - Fo)
        upper_diag_B = np.ones(Nx - 1) * (Fo/2)
        lower_diag_B = np.ones(Nx - 1) * (Fo/2)
        main_diag_B[0] = 1 - 2*Fo
        main_diag_B[-1] = 1
        upper_diag_B[0] = 2*Fo/2
        lower_diag_B[-1] = 0
    
    A = diags([lower_diag_A, main_diag_A, upper_diag_A], [-1, 0, 1], format='csr')
    B = diags([lower_diag_B, main_diag_B, upper_diag_B], [-1, 0, 1], format='csr')
    
    q_interp = interp1d(t, q, kind='linear', fill_value='extrapolate')
    
    for n in range(Nt - 1):
        b = B.dot(T[:, n])
        
        if order == 1:
            b[0] += Fo * dx * (q_interp(t[n]) + q_interp(t[n+1])) / alpha
        else:
            b[0] += 2 * Fo * dx * (q_interp(t[n]) + q_interp(t[n+1])) / alpha
        
        b[-1] = 2 * T_right
        
        T[:, n + 1] = spsolve(A, b)
    
    return x, t, T


def solve_heat_direct_high_order(L, T_total, alpha, Nx, Nt, q, T_left, T_right):
    """
    高精度正问题求解器：使用四阶空间差分和二阶时间差分（Crank-Nicolson）
    """
    dx = L / (Nx - 1)
    dt = T_total / (Nt - 1)
    
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T_total, Nt)
    
    T = np.zeros((Nx, Nt))
    T[:, 0] = T_left + (T_right - T_left) * x / L
    
    Fo = alpha * dt / dx**2
    
    main_diag_A = np.ones(Nx) * (1 + 5/2 * Fo)
    upper_diag_A1 = np.ones(Nx - 1) * (-4/3 * Fo)
    lower_diag_A1 = np.ones(Nx - 1) * (-4/3 * Fo)
    upper_diag_A2 = np.ones(Nx - 2) * (Fo / 12)
    lower_diag_A2 = np.ones(Nx - 2) * (Fo / 12)
    
    main_diag_A[0] = 1 + 2*Fo
    main_diag_A[1] = 1 + 5/2 * Fo
    main_diag_A[-2] = 1 + 5/2 * Fo
    main_diag_A[-1] = 1
    
    upper_diag_A1[0] = -2*Fo
    upper_diag_A1[-1] = -Fo/2
    lower_diag_A1[0] = -Fo/2
    lower_diag_A1[-1] = 0
    
    upper_diag_A2[0] = 0
    upper_diag_A2[-1] = 0
    lower_diag_A2[0] = 0
    lower_diag_A2[-1] = 0
    
    A = diags([lower_diag_A2, lower_diag_A1, main_diag_A, upper_diag_A1, upper_diag_A2], 
              [-2, -1, 0, 1, 2], format='csr')
    
    main_diag_B = np.ones(Nx) * (1 - 5/2 * Fo)
    upper_diag_B1 = np.ones(Nx - 1) * (4/3 * Fo)
    lower_diag_B1 = np.ones(Nx - 1) * (4/3 * Fo)
    upper_diag_B2 = np.ones(Nx - 2) * (-Fo / 12)
    lower_diag_B2 = np.ones(Nx - 2) * (-Fo / 12)
    
    main_diag_B[0] = 1 - 2*Fo
    main_diag_B[1] = 1 - 5/2 * Fo
    main_diag_B[-2] = 1 - 5/2 * Fo
    main_diag_B[-1] = 1
    
    upper_diag_B1[0] = 2*Fo
    upper_diag_B1[-1] = Fo/2
    lower_diag_B1[0] = Fo/2
    lower_diag_B1[-1] = 0
    
    upper_diag_B2[0] = 0
    upper_diag_B2[-1] = 0
    lower_diag_B2[0] = 0
    lower_diag_B2[-1] = 0
    
    B = diags([lower_diag_B2, lower_diag_B1, main_diag_B, upper_diag_B1, upper_diag_B2], 
              [-2, -1, 0, 1, 2], format='csr')
    
    q_interp = interp1d(t, q, kind='cubic', fill_value='extrapolate')
    
    for n in range(Nt - 1):
        b = B.dot(T[:, n])
        b[0] += 2 * Fo * dx * (q_interp(t[n]) + q_interp(t[n+1])) / alpha
        b[-1] = 2 * T_right
        
        T[:, n + 1] = spsolve(A, b)
    
    return x, t, T


def compute_gradient_improved(L, T_total, alpha, Nx, Nt, T_measured, x_measured, t_measured, q_guess, solver='cn'):
    """
    改进的梯度计算：使用高精度正问题求解器
    """
    dx = L / (Nx - 1)
    dt = T_total / (Nt - 1)
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T_total, Nt)
    
    if solver == 'cn':
        _, _, T_current = solve_heat_direct_cn(L, T_total, alpha, Nx, Nt, q_guess, 20, 100, order=2)
    else:
        _, _, T_current = solve_heat_direct_high_order(L, T_total, alpha, Nx, Nt, q_guess, 20, 100)
    
    residual = np.zeros_like(T_current)
    for i, xi in enumerate(x_measured):
        idx_x = np.argmin(np.abs(x - xi))
        for j, tj in enumerate(t_measured):
            idx_t = np.argmin(np.abs(t - tj))
            residual[idx_x, idx_t] = T_current[idx_x, idx_t] - T_measured[i, j]
    
    lambda_adj = np.zeros((Nx, Nt))
    Fo = alpha * dt / dx**2
    
    main_diag_A = np.ones(Nx) * (1 + Fo)
    upper_diag_A = np.ones(Nx - 1) * (-Fo/2)
    lower_diag_A = np.ones(Nx - 1) * (-Fo/2)
    main_diag_A[0] = 1 + 2*Fo
    main_diag_A[-1] = 1
    upper_diag_A[0] = -Fo
    lower_diag_A[-1] = 0
    
    A_adj = diags([lower_diag_A, main_diag_A, upper_diag_A], [-1, 0, 1], format='csr')
    
    for n in range(Nt - 1, 0, -1):
        b = lambda_adj[:, n].copy()
        b += dt * residual[:, n]
        b[-1] = 0
        
        lambda_adj[:, n - 1] = spsolve(A_adj, b)
    
    gradient = -2 * dx * lambda_adj[0, :]
    
    return gradient, np.sum(residual**2) * dt


def cgm_inverse_improved(L, T_total, alpha, Nx, Nt, T_measured, x_measured, t_measured, 
                         q_initial, max_iter=100, tol=1e-6, beta_type='FR', solver='cn'):
    """
    改进的共轭梯度法，支持高精度求解器
    """
    q = q_initial.copy()
    
    gradient, cost = compute_gradient_improved(L, T_total, alpha, Nx, Nt, 
                                                T_measured, x_measured, t_measured, q, solver)
    
    d = -gradient.copy()
    
    cost_history = [cost]
    q_history = [q.copy()]
    
    for k in range(max_iter):
        alpha_cg = 0.1
        best_cost = cost
        best_q = q.copy()
        
        for line_iter in range(10):
            q_new = q + alpha_cg * d
            _, cost_new = compute_gradient_improved(L, T_total, alpha, Nx, Nt,
                                                     T_measured, x_measured, t_measured, q_new, solver)
            if cost_new < best_cost:
                best_cost = cost_new
                best_q = q_new.copy()
            alpha_cg *= 0.5
        
        q = best_q
        cost = best_cost
        
        gradient_new, _ = compute_gradient_improved(L, T_total, alpha, Nx, Nt,
                                                     T_measured, x_measured, t_measured, q, solver)
        
        if beta_type == 'FR':
            beta = np.dot(gradient_new, gradient_new) / (np.dot(gradient, gradient) + 1e-15)
        else:
            beta = np.dot(gradient_new, gradient_new - gradient) / (np.dot(gradient, gradient) + 1e-15)
        
        beta = max(0, beta)
        
        d = -gradient_new + beta * d
        gradient = gradient_new.copy()
        
        cost_history.append(cost)
        q_history.append(q.copy())
        
        if cost < tol:
            print(f"收敛于迭代 {k + 1}")
            break
    
    return q, cost_history, q_history


def grid_convergence_study(L, T_total, alpha, q_true, T_left, T_right, Nx_list, Nt_list):
    """
    网格收敛性研究：验证数值解的收敛阶
    """
    errors = []
    
    for Nx, Nt in zip(Nx_list, Nt_list):
        x, t, T = solve_heat_direct_cn(L, T_total, alpha, Nx, Nt, q_true, T_left, T_right, order=2)
        errors.append(T.copy())
    
    print("\n=== 网格收敛性研究 ===")
    for i in range(len(Nx_list)-1):
        dx1 = L / (Nx_list[i] - 1)
        dx2 = L / (Nx_list[i+1] - 1)
        
        T1 = errors[i][::int(Nx_list[i]/Nx_list[i+1]), ::int(Nt_list[i]/Nt_list[i+1])]
        T2 = errors[i+1]
        
        min_nx = min(T1.shape[0], T2.shape[0])
        min_nt = min(T1.shape[1], T2.shape[1])
        
        error_L2 = np.sqrt(np.mean((T1[:min_nx, :min_nt] - T2[:min_nx, :min_nt])**2))
        
        order = np.log2(error_L2 / (np.sqrt(np.mean((errors[i+1] - errors[i])**2)) if i>0 else error_L2))
        
        print(f"网格 {Nx_list[i]}x{Nt_list[i]} -> {Nx_list[i+1]}x{Nt_list[i+1]}")
        print(f"  L2误差: {error_L2:.6e}")
        if i > 0:
            print(f"  收敛阶: {order:.2f}")
    
    return errors


def generate_test_data_improved(Nx_ref=200, Nt_ref=400):
    """
    生成测试数据：使用精细网格生成参考解（减小系统误差）
    """
    L = 1.0
    T_total = 10.0
    alpha = 0.01
    
    t = np.linspace(0, T_total, Nt_ref)
    q_true = 50 + 30 * np.sin(2 * np.pi * t / T_total)
    
    x_ref, t_ref, T_true_ref = solve_heat_direct_cn(L, T_total, alpha, Nx_ref, Nt_ref, q_true, 20, 100, order=2)
    
    x_measured = np.array([0.2, 0.5, 0.8])
    t_measured_idx = np.arange(0, Nt_ref, 20)
    t_measured = t_ref[t_measured_idx]
    
    T_measured = np.zeros((len(x_measured), len(t_measured)))
    for i, xi in enumerate(x_measured):
        idx_x = np.argmin(np.abs(x_ref - xi))
        for j, tj_idx in enumerate(t_measured_idx):
            T_measured[i, j] = T_true_ref[idx_x, tj_idx]
    
    noise_level = 0.005
    T_measured += noise_level * np.std(T_measured) * np.random.randn(*T_measured.shape)
    
    return L, T_total, alpha, x_measured, t_measured, T_measured, q_true, t


def main():
    print("=" * 70)
    print("热传导反问题求解 - 改进版共轭梯度法(CGM)")
    print("=" * 70)
    
    print("\n1. 生成高精度测试数据...")
    L, T_total, alpha, x_measured, t_measured, T_measured, q_true, t = generate_test_data_improved()
    
    print(f"   空间域: [0, {L}] m")
    print(f"   时间域: [0, {T_total}] s")
    print(f"   热扩散系数: {alpha} m²/s")
    print(f"   测量位置: {x_measured} m")
    print(f"   测量点数: {T_measured.shape}")
    
    Nx_solve = 100
    Nt_solve = 200
    q_initial = 50 * np.ones_like(t)
    
    print(f"\n2. 求解配置: {Nx_solve}x{Nt_solve} 网格")
    print("   求解器: Crank-Nicolson (二阶时间+空间精度)")
    
    print("\n3. 运行改进的共轭梯度法...")
    q_opt, cost_history, q_history = cgm_inverse_improved(
        L, T_total, alpha, Nx_solve, Nt_solve,
        T_measured, x_measured, t_measured,
        q_initial, max_iter=30, tol=1e-5, beta_type='FR', solver='cn'
    )
    
    print("\n4. 结果分析...")
    print(f"   初始目标函数值: {cost_history[0]:.6f}")
    print(f"   最终目标函数值: {cost_history[-1]:.6f}")
    print(f"   迭代次数: {len(cost_history) - 1}")
    
    rmse = np.sqrt(np.mean((q_opt - q_true)**2))
    max_error = np.max(np.abs(q_opt - q_true))
    print(f"   热流反演RMSE: {rmse:.4f} W/m²")
    print(f"   热流反演最大误差: {max_error:.4f} W/m²")
    
    print("\n5. 执行网格收敛性验证...")
    Nx_list = [25, 50, 100, 200]
    Nt_list = [50, 100, 200, 400]
    grid_convergence_study(L, T_total, alpha, q_true, 20, 100, Nx_list, Nt_list)
    
    fig = plt.figure(figsize=(18, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t, q_true, 'b-', linewidth=2.5, label='精确热流')
    ax1.plot(t, q_opt, 'r--', linewidth=2, label='反演热流')
    ax1.set_xlabel('时间 (s)', fontsize=11)
    ax1.set_ylabel('热流 (W/m²)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title('边界热流对比', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(cost_history, 'b-o', markersize=5, linewidth=2)
    ax2.set_xlabel('迭代次数', fontsize=11)
    ax2.set_ylabel('目标函数值 (log)', fontsize=11)
    ax2.set_title('目标函数收敛历史', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    error = np.abs(q_opt - q_true)
    ax3.plot(t, error, 'g-', linewidth=2)
    ax3.fill_between(t, 0, error, alpha=0.3, color='g')
    ax3.set_xlabel('时间 (s)', fontsize=11)
    ax3.set_ylabel('绝对误差', fontsize=11)
    ax3.set_title('热流反演误差', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 3, 4)
    x_plot, t_plot, T_opt = solve_heat_direct_cn(L, T_total, alpha, Nx_solve, Nt_solve, q_opt, 20, 100, order=2)
    im = ax4.contourf(t_plot, x_plot, T_opt, 50, cmap='hot')
    ax4.set_xlabel('时间 (s)', fontsize=11)
    ax4.set_ylabel('位置 (m)', fontsize=11)
    ax4.set_title('反演热流下的温度场', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    ax5 = plt.subplot(2, 3, 5)
    for i in range(0, len(q_history), 5):
        ax5.plot(t, q_history[i], '-', alpha=0.5, linewidth=1, label=f'迭代{i}' if i%10==0 else "")
    ax5.plot(t, q_true, 'k-', linewidth=2, label='精确解')
    ax5.set_xlabel('时间 (s)', fontsize=11)
    ax5.set_ylabel('热流 (W/m²)', fontsize=11)
    ax5.set_title('热流迭代演化过程', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    relative_error = (q_opt - q_true) / (np.abs(q_true) + 1e-10) * 100
    ax6.plot(t, relative_error, 'r-', linewidth=2)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('时间 (s)', fontsize=11)
    ax6.set_ylabel('相对误差 (%)', fontsize=11)
    ax6.set_title('热流反演相对误差', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heat_inverse_cgm_improved_results.png', dpi=150, bbox_inches='tight')
    print("\n6. 结果已保存至 'heat_inverse_cgm_improved_results.png'")
    
    print("\n" + "=" * 70)
    print("求解完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
