import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def solve_heat_direct(L, T_total, alpha, Nx, Nt, q0, T_left, T_right):
    """
    求解一维热传导正问题
    
    参数:
        L: 空间域长度
        T_total: 总时间
        alpha: 热扩散系数
        Nx: 空间网格数
        Nt: 时间步数
        q0: 边界热流 (x=0处)
        T_left: x=0处初始温度
        T_right: x=L处初始温度
    
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
    
    main_diag = np.ones(Nx) * (1 + 2 * Fo)
    upper_diag = np.ones(Nx - 1) * (-Fo)
    lower_diag = np.ones(Nx - 1) * (-Fo)
    
    main_diag[0] = 1 + Fo
    main_diag[-1] = 1
    
    upper_diag[0] = -Fo
    lower_diag[-1] = 0
    
    A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
    
    for n in range(Nt - 1):
        b = T[:, n].copy()
        b[0] += 2 * Fo * dx * q0 / alpha
        b[-1] = T_right
        
        T[:, n + 1] = spsolve(A, b)
    
    return x, t, T


def compute_gradient(L, T_total, alpha, Nx, Nt, T_measured, x_measured, t_measured, q_guess):
    """
    计算目标函数关于热流q的梯度
    """
    dx = L / (Nx - 1)
    dt = T_total / (Nt - 1)
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T_total, Nt)
    
    _, _, T_current = solve_heat_direct(L, T_total, alpha, Nx, Nt, q_guess, 20, 100)
    
    residual = np.zeros_like(T_current)
    for i, xi in enumerate(x_measured):
        idx_x = np.argmin(np.abs(x - xi))
        for j, tj in enumerate(t_measured):
            idx_t = np.argmin(np.abs(t - tj))
            residual[idx_x, idx_t] = T_current[idx_x, idx_t] - T_measured[i, j]
    
    lambda_adj = np.zeros((Nx, Nt))
    Fo = alpha * dt / dx**2
    
    main_diag = np.ones(Nx) * (1 + 2 * Fo)
    upper_diag = np.ones(Nx - 1) * (-Fo)
    lower_diag = np.ones(Nx - 1) * (-Fo)
    main_diag[0] = 1 + Fo
    main_diag[-1] = 1
    
    A_adj = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
    
    for n in range(Nt - 1, 0, -1):
        b = lambda_adj[:, n].copy()
        b += dt * residual[:, n]
        b[-1] = 0
        
        lambda_adj[:, n - 1] = spsolve(A_adj, b)
    
    gradient = -2 * dx * lambda_adj[0, :]
    
    return gradient, np.sum(residual**2) * dt


def cgm_inverse(L, T_total, alpha, Nx, Nt, T_measured, x_measured, t_measured, 
                q_initial, max_iter=100, tol=1e-6, beta_type='FR'):
    """
    共轭梯度法求解热传导反问题
    
    参数:
        L, T_total, alpha, Nx, Nt: 正问题参数
        T_measured: 测量的温度数据 [位置数 x 时间数]
        x_measured: 测量位置
        t_measured: 测量时间
        q_initial: 热流初始猜测
        max_iter: 最大迭代次数
        tol: 收敛容差
        beta_type: beta计算方式 ('FR' Fletcher-Reeves, 'PR' Polak-Ribiere)
    
    返回:
        q_opt: 优化后的热流
        cost_history: 目标函数历史
        q_history: 热流迭代历史
    """
    q = q_initial.copy()
    Nt_q = len(q)
    
    gradient, cost = compute_gradient(L, T_total, alpha, Nx, Nt, 
                                       T_measured, x_measured, t_measured, q)
    
    d = -gradient.copy()
    
    cost_history = [cost]
    q_history = [q.copy()]
    
    for k in range(max_iter):
        alpha_cg = 0.1
        for line_iter in range(20):
            q_new = q + alpha_cg * d
            _, cost_new = compute_gradient(L, T_total, alpha, Nx, Nt,
                                            T_measured, x_measured, t_measured, q_new)
            if cost_new < cost:
                break
            alpha_cg *= 0.5
        
        q = q_new
        cost = cost_new
        
        gradient_new, _ = compute_gradient(L, T_total, alpha, Nx, Nt,
                                            T_measured, x_measured, t_measured, q)
        
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


def generate_test_data():
    """
    生成测试数据：已知精确热流，生成温度测量数据
    """
    L = 1.0
    T_total = 10.0
    alpha = 0.01
    Nx = 50
    Nt = 100
    
    t = np.linspace(0, T_total, Nt)
    q_true = 50 + 30 * np.sin(2 * np.pi * t / T_total)
    
    x, t_grid, T_true = solve_heat_direct(L, T_total, alpha, Nx, Nt, q_true, 20, 100)
    
    x_measured = np.array([0.2, 0.5, 0.8])
    t_measured = t_grid[::5]
    
    T_measured = np.zeros((len(x_measured), len(t_measured)))
    for i, xi in enumerate(x_measured):
        idx_x = np.argmin(np.abs(x - xi))
        for j, tj in enumerate(t_measured):
            idx_t = np.argmin(np.abs(t_grid - tj))
            T_measured[i, j] = T_true[idx_x, idx_t]
    
    noise_level = 0.01
    T_measured += noise_level * np.std(T_measured) * np.random.randn(*T_measured.shape)
    
    return L, T_total, alpha, Nx, Nt, x_measured, t_measured, T_measured, q_true, t


def main():
    print("=" * 60)
    print("热传导反问题求解 - 共轭梯度法(CGM)")
    print("=" * 60)
    
    print("\n1. 生成测试数据...")
    L, T_total, alpha, Nx, Nt, x_measured, t_measured, T_measured, q_true, t = generate_test_data()
    
    print(f"   空间域: [0, {L}] m")
    print(f"   时间域: [0, {T_total}] s")
    print(f"   热扩散系数: {alpha} m²/s")
    print(f"   测量位置: {x_measured} m")
    print(f"   测量点数: {T_measured.shape}")
    
    print("\n2. 初始化反问题求解...")
    q_initial = 50 * np.ones_like(t)
    
    print("\n3. 运行共轭梯度法...")
    q_opt, cost_history, q_history = cgm_inverse(
        L, T_total, alpha, Nx, Nt,
        T_measured, x_measured, t_measured,
        q_initial, max_iter=50, tol=1e-4, beta_type='FR'
    )
    
    print("\n4. 结果分析...")
    print(f"   初始目标函数值: {cost_history[0]:.6f}")
    print(f"   最终目标函数值: {cost_history[-1]:.6f}")
    print(f"   迭代次数: {len(cost_history) - 1}")
    
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, q_true, 'b-', linewidth=2, label='精确热流')
    ax1.plot(t, q_opt, 'r--', linewidth=2, label='反演热流')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('热流 (W/m²)')
    ax1.legend()
    ax1.set_title('边界热流对比')
    ax1.grid(True)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.semilogy(cost_history, 'b-o', markersize=4)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('目标函数值 (log)')
    ax2.set_title('目标函数收敛历史')
    ax2.grid(True)
    
    ax3 = plt.subplot(2, 2, 3)
    error = np.abs(q_opt - q_true)
    ax3.plot(t, error, 'g-', linewidth=2)
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('绝对误差')
    ax3.set_title('热流反演误差')
    ax3.grid(True)
    
    ax4 = plt.subplot(2, 2, 4)
    x_plot, t_plot, T_opt = solve_heat_direct(L, T_total, alpha, Nx, Nt, q_opt, 20, 100)
    im = ax4.contourf(t_plot, x_plot, T_opt, 50, cmap='hot')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('位置 (m)')
    ax4.set_title('反演热流下的温度场')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('heat_inverse_cgm_results.png', dpi=150, bbox_inches='tight')
    print("\n5. 结果已保存至 'heat_inverse_cgm_results.png'")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("求解完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
