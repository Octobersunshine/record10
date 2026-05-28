import numpy as np
import time
from typing import Callable, Optional, Tuple


def moore_penrose_pseudoinverse(A):
    """
    通过SVD计算矩阵的Moore-Penrose伪逆。
    
    对于矩阵A，SVD分解为 A = U Σ V^T
    伪逆 A^+ = V Σ^+ U^T
    其中Σ^+是将Σ的非零元素取倒数后转置。
    
    使用相对阈值进行小奇异值截断:
        tolerance = max(Σ) * eps * max(m, n)
    这样可以自适应不同规模和量级的矩阵。
    
    参数:
        A: 输入矩阵 (m x n)，可以是奇异矩阵或非方阵
    
    返回:
        A_plus: 伪逆矩阵 (n x m)
    """
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    
    max_sigma = S[0] if len(S) > 0 else 0.0
    tolerance = max_sigma * np.finfo(S.dtype).eps * max(A.shape)
    S_plus = np.zeros_like(S, dtype=float)
    mask = S > tolerance
    S_plus[mask] = 1.0 / S[mask]
    
    S_plus_diag = np.diag(S_plus)
    
    A_plus = VT.T @ S_plus_diag @ U.T
    
    return A_plus


def solve_least_squares(A, b, method='svd'):
    """
    求解最小二乘问题 min ||Ax - b||_2，返回最小范数解。
    
    当A列满秩时，解是唯一的：x = (A^T A)^{-1} A^T b
    当A秩亏缺时，返回最小范数解：x = A^+ b
    
    参数:
        A: 系数矩阵 (m x n)
        b: 右端向量 (m,) 或 (m, k)
        method: 'svd' - 使用SVD伪逆求解
    
    返回:
        x: 最小范数解 (n,) 或 (n, k)
        residual: 残差 ||Ax - b||_2
    """
    if method == 'svd':
        A_plus = moore_penrose_pseudoinverse(A)
        x = A_plus @ b
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    residual = np.linalg.norm(A @ x - b)
    return x, residual


def landweber_pseudoinverse(A, max_iter=50000, tol=1e-8, alpha=None, verbose=False):
    """
    使用Landweber迭代法近似计算伪逆。
    
    使用更有效的迭代格式: X_{k+1} = X_k + alpha * A^T (I - A X_k)
    
    适用于大规模稀疏矩阵，不需要显式存储整个矩阵。
    
    参数:
        A: 输入矩阵 (m x n)
        max_iter: 最大迭代次数
        tol: 收敛阈值
        alpha: 步长参数，默认 1.9 / ||A||_2^2 (最优收敛)
        verbose: 是否打印迭代信息
    
    返回:
        X: 近似伪逆矩阵 (n x m)
        info: 包含迭代信息的字典
    """
    m, n = A.shape
    
    if alpha is None:
        sigma_max = np.linalg.svd(A, compute_uv=False)[0]
        alpha = 1.0 / (sigma_max ** 2) if sigma_max > 0 else 1.0
    
    X = np.zeros((n, m))
    A_T = A.T
    I = np.eye(m)
    
    residuals = []
    for k in range(max_iter):
        AX = A @ X
        X_new = X + alpha * (A_T @ (I - AX))
        
        diff = np.linalg.norm(X_new - X, 'fro') / max(1e-15, np.linalg.norm(X, 'fro'))
        residuals.append(diff)
        
        if verbose and k % 500 == 0:
            print(f"迭代 {k}: 相对变化 = {diff:.2e}")
        
        X = X_new
        
        if diff < tol:
            if verbose:
                print(f"在第 {k} 次迭代收敛")
            break
    
    info = {
        'iterations': k + 1,
        'converged': diff < tol,
        'final_residual': diff,
        'residuals': residuals
    }
    
    return X, info


def conjugate_gradient_pinv(A, max_iter=1000, tol=1e-10, verbose=False):
    """
    使用共轭梯度法求解 (A^T A) X = A^T，得到伪逆 X = A^+。
    
    对每一列e_i，求解 (A^T A) x_i = A^T e_i
    
    参数:
        A: 输入矩阵 (m x n)
        max_iter: 最大迭代次数
        tol: 收敛阈值
        verbose: 是否打印迭代信息
    
    返回:
        X: 近似伪逆矩阵 (n x m)
        info: 包含迭代信息的字典
    """
    m, n = A.shape
    A_T = A.T
    ATA = A_T @ A
    
    X = np.zeros((n, m))
    total_iters = 0
    
    for i in range(m):
        b = A_T[:, i]
        
        x = np.zeros(n)
        r = b - ATA @ x
        p = r.copy()
        rs_old = r @ r
        
        for k in range(max_iter):
            Ap = ATA @ p
            alpha = rs_old / (p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = r @ r
            
            if np.sqrt(rs_new) < tol:
                break
            
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        
        X[:, i] = x
        total_iters += k + 1
        
        if verbose and (i + 1) % 10 == 0:
            print(f"已求解 {i+1}/{m} 列")
    
    info = {
        'avg_iterations': total_iters / m,
        'total_iterations': total_iters
    }
    
    return X, info


def cgls_least_squares(A, b, max_iter=1000, tol=1e-12, verbose=False):
    """
    使用CGLS (Conjugate Gradient Least Squares) 直接求解最小二乘问题。
    
    CGLS求解 min ||Ax - b||_2，等价于求解 A^T A x = A^T b
    但在数值上更稳定。
    
    参数:
        A: 系数矩阵 (m x n)
        b: 右端向量 (m,)
        max_iter: 最大迭代次数
        tol: 收敛阈值（相对残差）
        verbose: 是否打印迭代信息
    
    返回:
        x: 最小二乘解
        info: 迭代信息
    """
    m, n = A.shape
    
    x = np.zeros(n, dtype=float)
    r = b - A @ x
    s = A.T @ r
    p = s.copy()
    gamma_old = s @ s
    
    if gamma_old == 0:
        info = {'iterations': 0, 'converged': True, 'final_residual': 0, 'residuals': [0]}
        return x, info
    
    initial_residual = np.linalg.norm(r)
    residuals = []
    
    for k in range(max_iter):
        Ap = A @ p
        Ap_norm_sq = Ap @ Ap
        
        if Ap_norm_sq < 1e-30:
            break
            
        alpha = gamma_old / Ap_norm_sq
        x = x + alpha * p
        r = r - alpha * Ap
        s = A.T @ r
        gamma_new = s @ s
        
        residual_norm = np.linalg.norm(r)
        relative_residual = residual_norm / initial_residual
        residuals.append(relative_residual)
        
        if verbose and k % 10 == 0:
            print(f"迭代 {k}: 相对残差 = {relative_residual:.2e}")
        
        if relative_residual < tol:
            break
        
        if gamma_new < 1e-30:
            break
            
        beta = gamma_new / gamma_old
        p = s + beta * p
        gamma_old = gamma_new
    
    info = {
        'iterations': k + 1,
        'converged': relative_residual < tol,
        'final_residual': residual_norm,
        'relative_residual': relative_residual,
        'residuals': residuals
    }
    
    return x, info


def verify_pseudoinverse(A, A_plus):
    """
    验证伪逆的四个Penrose条件:
    1. A * A_plus * A = A
    2. A_plus * A * A_plus = A_plus
    3. (A * A_plus)^T = A * A_plus
    4. (A_plus * A)^T = A_plus * A
    """
    condition1 = np.allclose(A @ A_plus @ A, A)
    condition2 = np.allclose(A_plus @ A @ A_plus, A_plus)
    condition3 = np.allclose((A @ A_plus).T, A @ A_plus)
    condition4 = np.allclose((A_plus @ A).T, A_plus @ A)
    
    print("Penrose条件验证:")
    print(f"  条件1 (AA^+A = A): {'✓ 通过' if condition1 else '✗ 失败'}")
    print(f"  条件2 (A^+AA^+ = A^+): {'✓ 通过' if condition2 else '✗ 失败'}")
    print(f"  条件3 ((AA^+)^T = AA^+): {'✓ 通过' if condition3 else '✗ 失败'}")
    print(f"  条件4 ((A^+A)^T = A^+A): {'✓ 通过' if condition4 else '✗ 失败'}")
    
    return condition1 and condition2 and condition3 and condition4


def compare_methods(A, b=None):
    """
    比较不同方法计算伪逆和求解最小二乘的性能。
    """
    m, n = A.shape
    print(f"\n{'='*60}")
    print(f"矩阵规模: {m}x{n}, 秩: {np.linalg.matrix_rank(A)}")
    print(f"{'='*60}")
    
    methods = {
        'SVD (直接)': lambda: moore_penrose_pseudoinverse(A),
        'CG伪逆': lambda: conjugate_gradient_pinv(A, max_iter=500, tol=1e-10)[0],
    }
    
    if m * n <= 1000:
        methods['Landweber迭代'] = lambda: landweber_pseudoinverse(A, max_iter=5000, tol=1e-5)[0]
    
    results = {}
    for name, func in methods.items():
        start = time.time()
        try:
            X = func()
            elapsed = time.time() - start
            error = np.linalg.norm(X - np.linalg.pinv(A), 'fro')
            results[name] = {'time': elapsed, 'error': error}
            print(f"{name:15s}: 耗时 {elapsed:.4f}s, 误差 {error:.2e}")
        except Exception as e:
            print(f"{name:15s}: 失败 - {e}")
    
    if b is not None:
        print(f"\n最小二乘求解对比:")
        ls_methods = {
            'SVD伪逆': lambda: solve_least_squares(A, b, method='svd')[0],
            'CGLS': lambda: cgls_least_squares(A, b, max_iter=min(n, 500), tol=1e-12)[0],
        }
        
        x_exact = np.linalg.lstsq(A, b, rcond=None)[0]
        
        for name, func in ls_methods.items():
            start = time.time()
            x = func()
            elapsed = time.time() - start
            error = np.linalg.norm(x - x_exact)
            residual = np.linalg.norm(A @ x - b)
            print(f"{name:10s}: 耗时 {elapsed:.4f}s, 解误差 {error:.2e}, 残差 {residual:.2e}")
    
    return results


def test_least_squares():
    """测试最小二乘求解"""
    print("\n" + "="*60)
    print("测试最小二乘求解")
    print("="*60)
    
    np.random.seed(42)
    
    print("\n1. 超定方程组 (m > n):")
    A = np.random.rand(100, 10)
    x_true = np.random.rand(10)
    b = A @ x_true + 0.01 * np.random.randn(100)
    
    x_ls, residual = solve_least_squares(A, b)
    x_np, *_ = np.linalg.lstsq(A, b, rcond=None)
    
    print(f"  解误差 (与numpy.lstsq对比): {np.linalg.norm(x_ls - x_np):.2e}")
    print(f"  残差 ||Ax - b||: {residual:.2e}")
    print(f"  与真实解误差: {np.linalg.norm(x_ls - x_true):.2e}")
    
    print("\n2. 欠定方程组 - 最小范数解 (m < n):")
    A = np.random.rand(5, 20)
    x_true = np.random.rand(20)
    b = A @ x_true
    
    x_ls, residual = solve_least_squares(A, b)
    x_np = np.linalg.pinv(A) @ b
    
    print(f"  解误差 (与numpy.pinv对比): {np.linalg.norm(x_ls - x_np):.2e}")
    print(f"  残差 ||Ax - b||: {residual:.2e}")
    print(f"  解的范数: {np.linalg.norm(x_ls):.4f}")
    
    print("\n3. 秩亏缺方程组:")
    A = np.random.rand(20, 10)
    A[:, -1] = A[:, 0]
    b = np.random.rand(20)
    
    x_ls, residual = solve_least_squares(A, b)
    x_np = np.linalg.pinv(A) @ b
    
    print(f"  矩阵秩: {np.linalg.matrix_rank(A)}")
    print(f"  解误差 (与numpy.pinv对比): {np.linalg.norm(x_ls - x_np):.2e}")
    print(f"  残差 ||Ax - b||: {residual:.2e}")


def test_iterative_methods():
    """测试迭代法伪逆"""
    print("\n" + "="*60)
    print("测试迭代法伪逆")
    print("="*60)
    
    np.random.seed(42)
    
    print("\n1. 小规模矩阵测试:")
    A = np.random.rand(10, 8)
    
    print(f"  SVD伪逆:")
    X_svd = moore_penrose_pseudoinverse(A)
    
    print(f"\n  Landweber迭代:")
    X_landweber, info_land = landweber_pseudoinverse(A, max_iter=5000, tol=1e-8, verbose=False)
    print(f"    迭代次数: {info_land['iterations']}")
    print(f"    收敛: {info_land['converged']}")
    print(f"    与SVD误差: {np.linalg.norm(X_landweber - X_svd, 'fro'):.2e}")
    
    print(f"\n  CG伪逆:")
    X_cg, info_cg = conjugate_gradient_pinv(A, max_iter=500, tol=1e-8)
    print(f"    平均迭代次数: {info_cg['avg_iterations']:.1f}")
    print(f"    与SVD误差: {np.linalg.norm(X_cg - X_svd, 'fro'):.2e}")
    
    print("\n2. CGLS最小二乘求解:")
    b = np.random.rand(10)
    x_cgls, info_cgls = cgls_least_squares(A, b, max_iter=100, tol=1e-10)
    x_svd = X_svd @ b
    print(f"  迭代次数: {info_cgls['iterations']}")
    print(f"  与SVD解误差: {np.linalg.norm(x_cgls - x_svd):.2e}")
    print(f"  残差: {info_cgls['final_residual']:.2e}")


def test_performance_comparison():
    """性能对比测试"""
    print("\n" + "="*60)
    print("性能对比测试")
    print("="*60)
    
    np.random.seed(42)
    
    sizes = [(20, 15), (100, 50), (200, 100)]
    
    for m, n in sizes:
        A = np.random.rand(m, n)
        b = np.random.rand(m)
        compare_methods(A, b)


def test_examples():
    """测试各种类型的矩阵"""
    np.set_printoptions(precision=4, suppress=True)
    
    np.random.seed(42)
    large_scale = np.random.rand(5, 5) * 1e6
    small_scale = np.random.rand(5, 5) * 1e-6
    
    large_ill_conditioned = np.random.rand(10, 10) * 1e10
    large_ill_conditioned[:, -1] = large_ill_conditioned[:, 0] * 1e-9
    
    examples = [
        ("非奇异方阵 (2x2)", np.array([[1, 2], [3, 4]])),
        ("奇异方阵 (2x2)", np.array([[1, 2], [2, 4]])),
        ("行满秩矩阵 (2x3)", np.array([[1, 2, 3], [4, 5, 6]])),
        ("列满秩矩阵 (3x2)", np.array([[1, 4], [2, 5], [3, 6]])),
        ("秩亏缺矩阵 (3x3)", np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
        ("零矩阵 (2x3)", np.zeros((2, 3))),
        ("单行矩阵 (1x3)", np.array([[1, 2, 3]])),
        ("单列矩阵 (3x1)", np.array([[1], [2], [3]])),
        (f"大数值矩阵 (5x5, ~1e6)", large_scale),
        (f"小数值矩阵 (5x5, ~1e-6)", small_scale),
        (f"大数值病态矩阵 (10x10, ~1e10)", large_ill_conditioned),
    ]
    
    for name, A in examples:
        print("=" * 60)
        print(f"测试: {name}")
        print(f"矩阵形状: {A.shape}")
        print(f"矩阵秩: {np.linalg.matrix_rank(A)}")
        
        S = np.linalg.svd(A, compute_uv=False)
        max_sigma = S[0] if len(S) > 0 else 0
        tolerance = max_sigma * np.finfo(S.dtype).eps * max(A.shape)
        print(f"最大奇异值: {max_sigma:.2e}")
        print(f"截断阈值: {tolerance:.2e}")
        print(f"有效奇异值数量: {np.sum(S > tolerance)}/{len(S)}")
        
        print("\n原矩阵 A (前3行前3列):")
        rows, cols = min(3, A.shape[0]), min(3, A.shape[1])
        print(A[:rows, :cols])
        if A.shape[0] > 3 or A.shape[1] > 3:
            print("...")
        
        A_plus = moore_penrose_pseudoinverse(A)
        print(f"\n伪逆 A^+ 形状: {A_plus.shape}")
        
        A_plus_numpy = np.linalg.pinv(A)
        print(f"与 numpy.linalg.pinv 对比: {np.allclose(A_plus, A_plus_numpy)}")
        
        print()
        verify_pseudoinverse(A, A_plus)
        print()


if __name__ == "__main__":
    test_examples()
    test_least_squares()
    test_iterative_methods()
    test_performance_comparison()
