import numpy as np
from scipy.linalg import eigh
from modal_analysis_lanczos import LanczosSolver, subspace_iteration


def create_close_modes_system(n=50, close_pair_idx=10, perturbation=1e-4):
    """
    创建具有密集模态的测试系统
    
    通过微调刚度矩阵的特定位置，使两个模态频率非常接近
    """
    M = np.eye(n)
    
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
        if i > 0:
            K[i, i-1] = -1.0
        if i < n - 1:
            K[i, i+1] = -1.0
    
    K[close_pair_idx, close_pair_idx] *= (1 + perturbation)
    K[close_pair_idx+1, close_pair_idx+1] *= (1 + perturbation)
    K[close_pair_idx, close_pair_idx+1] *= (1 - perturbation * 0.5)
    K[close_pair_idx+1, close_pair_idx] *= (1 - perturbation * 0.5)
    
    return K, M


def test_eigh_with_close_modes():
    """
    测试scipy.eigh在密集模态下的表现（作为参考）
    """
    print("=" * 80)
    print("测试1: scipy.eigh 对密集模态的求解（参考解）")
    print("=" * 80)
    
    K, M = create_close_modes_system(n=30, close_pair_idx=8, perturbation=1e-5)
    
    eigvals, eigvecs = eigh(K, M, subset_by_index=[0, 15])
    frequencies = np.sqrt(np.maximum(eigvals, 0)) / (2 * np.pi)
    
    print("\n前15阶频率:")
    print("-" * 50)
    for i in range(15):
        print(f"  {i+1:2d}: {frequencies[i]:.10f} Hz")
    
    print("\n检测密集模态 (相对差 < 1e-3):")
    print("-" * 50)
    for i in range(1, 15):
        rel_diff = abs(frequencies[i] - frequencies[i-1]) / max(frequencies[i], frequencies[i-1])
        if rel_diff < 1e-3:
            print(f"  模态 {i} 和 {i+1}: 相对差 = {rel_diff:.2e}")
    
    return frequencies, eigvecs


def test_lanczos_with_close_modes():
    """
    测试Lanczos算法在密集模态下的表现
    """
    print("\n" + "=" * 80)
    print("测试2: Lanczos算法 对密集模态的求解")
    print("=" * 80)
    
    K, M = create_close_modes_system(n=30, close_pair_idx=8, perturbation=1e-5)
    
    print("\n--- 单向量Lanczos ---")
    solver1 = LanczosSolver(K, M)
    solver1.solve(num_eigenvalues=15, max_iter=80, tol=1e-9, 
                  reortho='full', verbose=True, block_size=None)
    solver1.normalize_mode_shapes('mass')
    solver1.print_results(precision=10)
    
    print("\n--- 块Lanczos (block_size=3) ---")
    solver2 = LanczosSolver(K, M)
    solver2.solve(num_eigenvalues=15, max_iter=40, tol=1e-9, 
                  reortho='full', verbose=True, block_size=3)
    solver2.normalize_mode_shapes('mass')
    solver2.print_results(precision=10)
    
    return solver1, solver2


def test_subspace_iteration_with_close_modes():
    """
    测试子空间迭代法在密集模态下的表现
    """
    print("\n" + "=" * 80)
    print("测试3: 子空间迭代法 对密集模态的求解")
    print("=" * 80)
    
    K, M = create_close_modes_system(n=30, close_pair_idx=8, perturbation=1e-5)
    
    eigvals, eigvecs, hist = subspace_iteration(
        K, M, num_eigenvalues=15, max_iter=100, tol=1e-9, verbose=True
    )
    frequencies = np.sqrt(np.maximum(eigvals, 0)) / (2 * np.pi)
    
    print("\n子空间迭代得到的前15阶频率:")
    print("-" * 50)
    for i in range(15):
        print(f"  {i+1:2d}: {frequencies[i]:.10f} Hz")
    
    residuals = hist[-1]['residuals']
    print("\n最终残差:")
    for i in range(15):
        print(f"  模态 {i+1:2d}: {residuals[i]:.2e}")
    
    return eigvals, eigvecs


def compare_all_methods():
    """
    比较所有方法的结果
    """
    print("\n" + "=" * 80)
    print("测试4: 各种方法结果对比")
    print("=" * 80)
    
    K, M = create_close_modes_system(n=30, close_pair_idx=8, perturbation=1e-5)
    
    # 参考解
    eigvals_eigh, eigvecs_eigh = eigh(K, M, subset_by_index=[0, 14])
    freqs_eigh = np.sqrt(np.maximum(eigvals_eigh, 0)) / (2 * np.pi)
    
    # Lanczos
    solver = LanczosSolver(K, M)
    solver.solve(num_eigenvalues=15, max_iter=80, tol=1e-9, 
                  reortho='full', verbose=False, block_size=3)
    freqs_lanczos = solver.get_frequencies()
    
    # 子空间迭代
    eigvals_sub, eigvecs_sub, _ = subspace_iteration(
        K, M, num_eigenvalues=15, max_iter=100, tol=1e-9, verbose=False
    )
    freqs_sub = np.sqrt(np.maximum(eigvals_sub, 0)) / (2 * np.pi)
    
    print("\n频率对比:")
    print("=" * 100)
    print(f"{'模态':<6} {'scipy.eigh':<20} {'Lanczos':<20} {'子空间迭代':<20} {'误差(Lanczos)':<15} {'误差(子空间)':<15}")
    print("-" * 100)
    
    for i in range(15):
        err_lanczos = abs(freqs_lanczos[i] - freqs_eigh[i]) / freqs_eigh[i]
        err_sub = abs(freqs_sub[i] - freqs_eigh[i]) / freqs_eigh[i]
        print(f"{i+1:<6} {freqs_eigh[i]:<20.10f} {freqs_lanczos[i]:<20.10f} {freqs_sub[i]:<20.10f} {err_lanczos:<15.2e} {err_sub:<15.2e}")
    
    print("\n最大相对误差:")
    print(f"  Lanczos:    {np.max(np.abs(freqs_lanczos - freqs_eigh) / freqs_eigh):.2e}")
    print(f"  子空间迭代: {np.max(np.abs(freqs_sub - freqs_eigh) / freqs_eigh):.2e}")


def convergence_analysis():
    """
    收敛性分析 - 展示残差随迭代的变化
    """
    print("\n" + "=" * 80)
    print("测试5: 收敛性分析（残差随迭代的变化）")
    print("=" * 80)
    
    K, M = create_close_modes_system(n=40, close_pair_idx=10, perturbation=1e-5)
    
    solver = LanczosSolver(K, M)
    solver.solve(num_eigenvalues=10, max_iter=60, tol=1e-12, 
                  reortho='full', verbose=False)
    
    print("\n收敛历史:")
    print("-" * 70)
    print(f"{'迭代':<8} {'最大残差':<15} {'模态1残差':<15} {'模态5残差':<15} {'模态10残差':<15}")
    print("-" * 70)
    
    for item in solver.convergence_history:
        it = item['iteration']
        res = item['residuals']
        print(f"{it:<8} {np.max(res):<15.2e} {res[0]:<15.2e} {res[4]:<15.2e} {res[9]:<15.2e}")
    
    print("\n最终残差范数:")
    for i in range(10):
        print(f"  模态 {i+1:2d}: {solver.residual_norms[i]:.2e}")
    
    all_converged = np.all(solver.residual_norms < 1e-9)
    print(f"\n所有模态残差 < 1e-9: {'是 ✓' if all_converged else '否 ✗'}")


def extreme_close_modes_test():
    """
    极端密集模态测试
    """
    print("\n" + "=" * 80)
    print("测试6: 极端密集模态测试 (频率相对差 ~1e-6)")
    print("=" * 80)
    
    K, M = create_close_modes_system(n=40, close_pair_idx=10, perturbation=1e-7)
    
    eigvals_eigh, _ = eigh(K, M, subset_by_index=[0, 15])
    freqs_eigh = np.sqrt(np.maximum(eigvals_eigh, 0)) / (2 * np.pi)
    
    print("\n密集模态区域频率:")
    for i in range(8, 13):
        rel_diff = abs(freqs_eigh[i] - freqs_eigh[i-1]) / max(freqs_eigh[i], freqs_eigh[i-1])
        print(f"  模态 {i} -> {i+1}: {freqs_eigh[i-1]:.10f} -> {freqs_eigh[i]:.10f} 相对差={rel_diff:.2e}")
    
    print("\n使用块Lanczos求解 (block_size=4):")
    solver = LanczosSolver(K, M)
    solver.solve(num_eigenvalues=15, max_iter=60, tol=1e-9, 
                  reortho='full', verbose=True, block_size=4)
    solver.normalize_mode_shapes('mass')
    solver.print_results(precision=10)
    
    freqs_lanczos = solver.get_frequencies()
    errors = np.abs(freqs_lanczos[:15] - freqs_eigh) / freqs_eigh
    print(f"\n与scipy.eigh的最大相对误差: {np.max(errors):.2e}")


if __name__ == "__main__":
    np.random.seed(42)
    
    test_eigh_with_close_modes()
    test_lanczos_with_close_modes()
    test_subspace_iteration_with_close_modes()
    compare_all_methods()
    convergence_analysis()
    extreme_close_modes_test()
    
    print("\n" + "=" * 80)
    print("密集模态处理总结:")
    print("=" * 80)
    print("1. Lanczos算法 + 完全重正交化: 适用于大多数情况")
    print("2. 块Lanczos (block_size > 1): 对密集模态更鲁棒")
    print("3. 子空间迭代法: 最稳定但较慢，适合前几阶模态")
    print("4. 收敛判据: 残差范数 ||Kφ - λMφ|| < 1e-9")
    print("=" * 80)
