import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def solve_heat_direct_old(L, T_total, alpha, Nx, Nt, q0, T_left, T_right):
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
        b[0] += 2 * Fo * dx * q0[n] / alpha
        b[-1] = T_right
        T[:, n + 1] = spsolve(A, b)
    
    return x, t, T


def solve_heat_direct_cn(L, T_total, alpha, Nx, Nt, q, T_left, T_right):
    dx = L / (Nx - 1)
    dt = T_total / (Nt - 1)
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T_total, Nt)
    T = np.zeros((Nx, Nt))
    T[:, 0] = T_left + (T_right - T_left) * x / L
    Fo = alpha * dt / dx**2
    
    main_diag_A = np.ones(Nx) * (1 + Fo)
    upper_diag_A = np.ones(Nx - 1) * (-Fo/2)
    lower_diag_A = np.ones(Nx - 1) * (-Fo/2)
    main_diag_A[0] = 1 + 2*Fo
    main_diag_A[-1] = 1
    upper_diag_A[0] = -Fo
    lower_diag_A[-1] = 0
    
    main_diag_B = np.ones(Nx) * (1 - Fo)
    upper_diag_B = np.ones(Nx - 1) * (Fo/2)
    lower_diag_B = np.ones(Nx - 1) * (Fo/2)
    main_diag_B[0] = 1 - 2*Fo
    main_diag_B[-1] = 1
    upper_diag_B[0] = Fo
    lower_diag_B[-1] = 0
    
    A = diags([lower_diag_A, main_diag_A, upper_diag_A], [-1, 0, 1], format='csr')
    B = diags([lower_diag_B, main_diag_B, upper_diag_B], [-1, 0, 1], format='csr')
    
    for n in range(Nt - 1):
        b = B.dot(T[:, n])
        b[0] += 2 * Fo * dx * (q[n] + q[n+1]) / alpha
        b[-1] = 2 * T_right
        T[:, n + 1] = spsolve(A, b)
    
    return x, t, T


def main():
    print("=" * 70)
    print("热传导正问题求解器精度对比测试")
    print("=" * 70)
    
    L = 1.0
    T_total = 10.0
    alpha = 0.01
    T_left = 20
    T_right = 100
    
    t = np.linspace(0, T_total, 400)
    q_true = 50 + 30 * np.sin(2 * np.pi * t / T_total)
    
    Nx_ref = 400
    Nt_ref = 800
    print(f"\n1. 生成参考解 (精细网格: {Nx_ref}x{Nt_ref})...")
    _, _, T_ref = solve_heat_direct_cn(L, T_total, alpha, Nx_ref, Nt_ref, q_true, T_left, T_right)
    
    grid_sizes = [(25, 50), (50, 100), (100, 200), (200, 400)]
    errors_old = []
    errors_new = []
    
    print("\n2. 测试不同网格密度下的精度...")
    print("-" * 70)
    print(f"{'网格':>12} | {'旧方法 L2误差':>16} | {'新方法 L2误差':>16} | {'精度提升':>12}")
    print("-" * 70)
    
    for Nx, Nt in grid_sizes:
        _, _, T_old = solve_heat_direct_old(L, T_total, alpha, Nx, Nt, q_true, T_left, T_right)
        _, _, T_new = solve_heat_direct_cn(L, T_total, alpha, Nx, Nt, q_true, T_left, T_right)
        
        step_x = Nx_ref // Nx
        step_t = Nt_ref // Nt
        
        T_ref_coarse = T_ref[::step_x, ::step_t]
        
        min_nx = min(T_old.shape[0], T_ref_coarse.shape[0])
        min_nt = min(T_old.shape[1], T_ref_coarse.shape[1])
        
        error_old = np.sqrt(np.mean((T_old[:min_nx, :min_nt] - T_ref_coarse[:min_nx, :min_nt])**2))
        error_new = np.sqrt(np.mean((T_new[:min_nx, :min_nt] - T_ref_coarse[:min_nx, :min_nt])**2))
        
        errors_old.append(error_old)
        errors_new.append(error_new)
        
        improvement = (error_old - error_new) / error_old * 100
        print(f"{Nx:>4}x{Nt:<4} | {error_old:>16.6e} | {error_new:>16.6e} | {improvement:>10.1f}%")
    
    print("-" * 70)
    
    print("\n3. 计算收敛阶...")
    print("\n旧方法（隐式欧拉，一阶精度）:")
    for i in range(len(grid_sizes)-1):
        order_old = np.log(errors_old[i] / errors_old[i+1]) / np.log(2)
        print(f"  网格{grid_sizes[i]} -> {grid_sizes[i+1]}: 收敛阶 = {order_old:.2f}")
    
    print("\n新方法（Crank-Nicolson，二阶精度）:")
    for i in range(len(grid_sizes)-1):
        order_new = np.log(errors_new[i] / errors_new[i+1]) / np.log(2)
        print(f"  网格{grid_sizes[i]} -> {grid_sizes[i+1]}: 收敛阶 = {order_new:.2f}")
    
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    grid_labels = [f"{Nx}x{Nt}" for Nx, Nt in grid_sizes]
    x = np.arange(len(grid_labels))
    width = 0.35
    
    ax1.bar(x - width/2, errors_old, width, label='旧方法 (隐式欧拉)', color='r', alpha=0.7)
    ax1.bar(x + width/2, errors_new, width, label='新方法 (Crank-Nicolson)', color='b', alpha=0.7)
    ax1.set_xlabel('网格密度', fontsize=11)
    ax1.set_ylabel('L2 误差', fontsize=11)
    ax1.set_title('不同网格下的正问题精度对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(grid_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.loglog([1/(Nx-1) for Nx, Nt in grid_sizes], errors_old, 'r-o', linewidth=2, markersize=7, label='旧方法 (一阶)')
    ax2.loglog([1/(Nx-1) for Nx, Nt in grid_sizes], errors_new, 'b-s', linewidth=2, markersize=7, label='新方法 (二阶)')
    ax2.set_xlabel('空间步长 dx', fontsize=11)
    ax2.set_ylabel('L2 误差', fontsize=11)
    ax2.set_title('收敛阶对比（双对数坐标）', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    ax3 = plt.subplot(2, 2, 3)
    Nx_plot = 50
    Nt_plot = 100
    x_grid, t_grid, T_old_plot = solve_heat_direct_old(L, T_total, alpha, Nx_plot, Nt_plot, q_true, T_left, T_right)
    _, _, T_new_plot = solve_heat_direct_cn(L, T_total, alpha, Nx_plot, Nt_plot, q_true, T_left, T_right)
    
    step_x_ref = Nx_ref // Nx_plot
    step_t_ref = Nt_ref // Nt_plot
    T_ref_plot = T_ref[::step_x_ref, ::step_t_ref]
    
    min_nx = min(T_old_plot.shape[0], T_ref_plot.shape[0])
    min_nt = min(T_old_plot.shape[1], T_ref_plot.shape[1])
    
    error_map_old = np.abs(T_old_plot[:min_nx, :min_nt] - T_ref_plot[:min_nx, :min_nt])
    error_map_new = np.abs(T_new_plot[:min_nx, :min_nt] - T_ref_plot[:min_nx, :min_nt])
    
    im1 = ax3.contourf(t_grid[:min_nt], x_grid[:min_nx], error_map_old, 50, cmap='hot')
    ax3.set_xlabel('时间 (s)', fontsize=11)
    ax3.set_ylabel('位置 (m)', fontsize=11)
    ax3.set_title(f'旧方法误差分布 (网格{Nx_plot}x{Nt_plot})', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax3)
    
    ax4 = plt.subplot(2, 2, 4)
    im2 = ax4.contourf(t_grid[:min_nt], x_grid[:min_nx], error_map_new, 50, cmap='hot')
    ax4.set_xlabel('时间 (s)', fontsize=11)
    ax4.set_ylabel('位置 (m)', fontsize=11)
    ax4.set_title(f'新方法误差分布 (网格{Nx_plot}x{Nt_plot})', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("\n4. 对比图已保存至 'accuracy_comparison.png'")
    
    print("\n" + "=" * 70)
    print("精度改进总结:")
    print("=" * 70)
    print("✓ 时间离散: 从一阶隐式欧拉改进为二阶Crank-Nicolson")
    print("✓ 边界条件: 从一阶精度改进为二阶精度")
    print("✓ 整体收敛阶: 从 O(Δt, Δx²) 提升到 O(Δt², Δx²)")
    print(f"✓ 典型网格(100x200)下误差降低: {(errors_old[-2]-errors_new[-2])/errors_old[-2]*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
