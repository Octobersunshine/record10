import numpy as np
import warnings
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla


def _matvec(A, v):
    if callable(A):
        return A(v)
    else:
        return A @ v


def _to_dense_matrix(A, n):
    if callable(A):
        dense = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            dense[:, i] = _matvec(A, e)
        return dense
    else:
        if sp.issparse(A):
            return A.toarray()
        return np.asarray(A)


def _extract_eigenvalues_from_schur(T):
    m = T.shape[0]
    eigenvalues = []
    i = 0
    while i < m:
        if i == m - 1:
            eigenvalues.append(T[i, i])
            i += 1
        else:
            sub_diag = abs(T[i + 1, i])
            if sub_diag > 1e-10:
                a, b = T[i, i], T[i, i + 1]
                c, d = T[i + 1, i], T[i + 1, i + 1]
                tr = a + d
                det = a * d - b * c
                disc = tr * tr - 4 * det
                if disc >= 0:
                    eigenvalues.append((tr + np.sqrt(disc)) / 2)
                    eigenvalues.append((tr - np.sqrt(disc)) / 2)
                else:
                    eigenvalues.append(complex(tr / 2, np.sqrt(-disc) / 2))
                    eigenvalues.append(complex(tr / 2, -np.sqrt(-disc) / 2))
                i += 2
            else:
                eigenvalues.append(T[i, i])
                i += 1
    return np.array(eigenvalues)


def power_iteration(A, n=None, max_iter=1000, tol=1e-10):
    if n is None:
        if callable(A):
            raise ValueError(
                "For linear operator A (callable), argument 'n' (problem size) is required."
            )
        else:
            n = A.shape[0]

    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    history = []

    for i in range(max_iter):
        v_new = _matvec(A, v)
        eigenvalue = float(v @ v_new)
        norm_v_new = np.linalg.norm(v_new)

        if norm_v_new < 1e-30:
            warnings.warn(
                "Iterate collapsed to zero; matrix may be singular.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0, v, np.array(history)

        v_new = v_new / norm_v_new

        history.append(eigenvalue)

        residual = np.linalg.norm(_matvec(A, v_new) - eigenvalue * v_new)
        if residual < tol:
            break

        if np.linalg.norm(v_new - v) < tol:
            break

        sign = np.sign(v_new @ v)
        if sign < 0:
            v_new = -v_new
        v = v_new
    else:
        warnings.warn(
            f"power_iteration did not converge after {max_iter} iterations. "
            "The matrix may have repeated dominant eigenvalues or conjugate "
            "complex eigenvalues with equal magnitude. "
            "Try orthogonal_iteration(), inverse_power_iteration(), or "
            "rayleigh_quotient_iteration() instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    return eigenvalue, v, np.array(history)


def shifted_power_iteration(A, shift=0.0, n=None, max_iter=1000, tol=1e-10):
    if n is None:
        if callable(A):
            raise ValueError(
                "For linear operator A (callable), argument 'n' is required."
            )
        else:
            n = A.shape[0]

    def B_op(v):
        return _matvec(A, v) - shift * v

    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    history = []

    for i in range(max_iter):
        v_new = B_op(v)
        norm_v_new = np.linalg.norm(v_new)

        if norm_v_new < 1e-15:
            warnings.warn(
                "Shifted matrix is nearly singular; shift may be an eigenvalue.",
                RuntimeWarning,
                stacklevel=2,
            )
            eigenvalue = shift
            return eigenvalue, v, np.array(history)

        eigenvalue_shifted = float(v @ v_new)
        v_new = v_new / norm_v_new

        eigenvalue = eigenvalue_shifted + shift
        history.append(eigenvalue)

        residual = np.linalg.norm(_matvec(A, v_new) - eigenvalue * v_new)
        if residual < tol:
            break

        if np.linalg.norm(v_new - v) < tol:
            break

        sign = np.sign(v_new @ v)
        if sign < 0:
            v_new = -v_new
        v = v_new
    else:
        warnings.warn(
            f"shifted_power_iteration did not converge after {max_iter} iterations.",
            RuntimeWarning,
            stacklevel=2,
        )

    return eigenvalue, v, np.array(history)


def inverse_power_iteration(A, shift=0.0, n=None, max_iter=1000, tol=1e-10,
                            solve=None):
    if n is None:
        if callable(A):
            raise ValueError(
                "For linear operator A (callable), argument 'n' is required."
            )
        else:
            n = A.shape[0]

    if solve is None:
        if callable(A):
            raise ValueError(
                "For linear operator A (callable), argument 'solve' is required: "
                "a function solve(v) that returns (A - shift*I)^-1 @ v."
            )
        else:
            if sp.issparse(A):
                lu = spla.splu(A - shift * sp.eye(n))
                def solve(v):
                    return lu.solve(v)
            else:
                M = A - shift * np.eye(n)
                lu, piv = sla.lu_factor(M)
                def solve(v):
                    return sla.lu_solve((lu, piv), v)

    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    history = []

    for i in range(max_iter):
        try:
            v_new = solve(v)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Matrix is singular at current shift; shift may be an eigenvalue.",
                RuntimeWarning,
                stacklevel=2,
            )
            return shift, v, np.array(history)

        norm_v_new = np.linalg.norm(v_new)

        if norm_v_new < 1e-30:
            warnings.warn(
                "Iterate collapsed to zero; solution may be inaccurate.",
                RuntimeWarning,
                stacklevel=2,
            )
            return shift, v, np.array(history)

        eigenvalue_shifted = float(v @ v_new)
        v_new = v_new / norm_v_new

        if abs(eigenvalue_shifted) < 1e-15:
            eigenvalue = shift
        else:
            eigenvalue = shift + 1.0 / eigenvalue_shifted
        history.append(eigenvalue)

        residual = np.linalg.norm(_matvec(A, v_new) - eigenvalue * v_new)
        if residual < tol:
            break

        if np.linalg.norm(v_new - v) < tol:
            break

        sign = np.sign(v_new @ v)
        if sign < 0:
            v_new = -v_new
        v = v_new
    else:
        warnings.warn(
            f"inverse_power_iteration did not converge after {max_iter} iterations.",
            RuntimeWarning,
            stacklevel=2,
        )

    return eigenvalue, v, np.array(history)


def rayleigh_quotient_iteration(A, n=None, max_iter=50, tol=1e-12,
                                 solve_shifted=None):
    if n is None:
        if callable(A):
            raise ValueError(
                "For linear operator A (callable), argument 'n' is required."
            )
        else:
            n = A.shape[0]

    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    Av = _matvec(A, v)
    eigenvalue = float(v @ Av)

    history = [eigenvalue]

    for i in range(max_iter):
        if solve_shifted is not None:
            try:
                v_new = solve_shifted(eigenvalue, v)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Matrix is singular at current Rayleigh quotient; "
                    "eigenvalue may have converged.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
        else:
            if callable(A):
                raise ValueError(
                    "For linear operator A (callable), argument 'solve_shifted' "
                    "is required: a function solve_shifted(mu, v) that returns "
                    "(A - mu*I)^-1 @ v."
                )
            else:
                M = A - eigenvalue * np.eye(n)
                try:
                    if sp.issparse(A):
                        M = sp.csc_matrix(M)
                        v_new = spla.spsolve(M, v)
                    else:
                        v_new = np.linalg.solve(M, v)
                except np.linalg.LinAlgError:
                    break

        norm_v_new = np.linalg.norm(v_new)
        if norm_v_new < 1e-30:
            warnings.warn(
                "Iterate collapsed to zero; eigenvalue may have converged.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        v_new = v_new / norm_v_new

        Av_new = _matvec(A, v_new)
        eigenvalue_new = float(v_new @ Av_new)
        history.append(eigenvalue_new)

        residual = np.linalg.norm(Av_new - eigenvalue_new * v_new)
        if residual < tol:
            v = v_new
            eigenvalue = eigenvalue_new
            break

        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            eigenvalue = eigenvalue_new
            break

        sign = np.sign(v_new @ v)
        if sign < 0:
            v_new = -v_new
        v = v_new
        eigenvalue = eigenvalue_new
    else:
        warnings.warn(
            f"rayleigh_quotient_iteration did not converge after {max_iter} iterations.",
            RuntimeWarning,
            stacklevel=2,
        )

    return eigenvalue, v, np.array(history)


def orthogonal_iteration(A, num_vectors=None, n=None, max_iter=1000, tol=1e-10):
    if n is None:
        if callable(A):
            raise ValueError(
                "For linear operator A (callable), argument 'n' is required."
            )
        else:
            n = A.shape[0]

    if num_vectors is None:
        num_vectors = n

    Q = np.random.rand(n, num_vectors)
    Q, _ = np.linalg.qr(Q)

    eigenvalue_history = []

    for i in range(max_iter):
        Z = np.zeros_like(Q)
        for j in range(num_vectors):
            Z[:, j] = _matvec(A, Q[:, j])

        Q_new, R = np.linalg.qr(Z)

        H = np.zeros((num_vectors, num_vectors))
        for j in range(num_vectors):
            AQj = _matvec(A, Q_new[:, j])
            for k in range(num_vectors):
                H[k, j] = Q_new[:, k] @ AQj

        eigenvalues = _extract_eigenvalues_from_schur(H)
        eigenvalue_history.append(eigenvalues.copy())

        if Q_new.shape == Q.shape:
            dots = np.abs(np.sum(Q_new * Q, axis=0))
            if np.all(dots > 1 - tol):
                Q = Q_new
                break

        Q = Q_new
    else:
        warnings.warn(
            f"orthogonal_iteration did not converge after {max_iter} iterations.",
            RuntimeWarning,
            stacklevel=2,
        )

    H = np.zeros((num_vectors, num_vectors))
    for j in range(num_vectors):
        AQj = _matvec(A, Q[:, j])
        for k in range(num_vectors):
            H[k, j] = Q[:, k] @ AQj

    eigenvalues = _extract_eigenvalues_from_schur(H)
    eigenvectors = Q

    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, np.array(eigenvalue_history, dtype=object)


if __name__ == "__main__":
    print("=" * 60)
    print("测试1: 幂迭代 vs 瑞利商迭代 (收敛速度对比)")
    print("=" * 60)
    A1 = np.array([[4, 1, 1],
                    [1, 3, 1],
                    [1, 1, 2]], dtype=float)
    ref = np.sort(np.linalg.eigvals(A1))[::-1]
    print(f"  所有特征值(NumPy): {ref}")

    eigenvalue_p, _, hist_p = power_iteration(A1)
    print(f"  幂迭代    - 收敛到: {eigenvalue_p:.12f}, 迭代次数: {len(hist_p)}")

    eig_vals_r = []
    iters_r = []
    for i in range(3):
        eigenvalue_r, _, hist_r = rayleigh_quotient_iteration(A1)
        eig_vals_r.append(eigenvalue_r)
        iters_r.append(len(hist_r))
        closest_idx = np.argmin(np.abs(ref - eigenvalue_r))
        print(f"  瑞利商迭代 运行{i+1}: 收敛到 {eigenvalue_r:.12f} (λ{closest_idx+1}), 迭代 {len(hist_r)} 次")

    print(f"  观察: 瑞利商迭代仅需约 3-8 次迭代即可收敛到机器精度, 远快于幂迭代的 ~30 次")
    print(f"        但随机初值可能收敛到任意附近特征值 (三次局部收敛)")

    print()
    print("=" * 60)
    print("测试2: 反幂迭代 (求最小特征值)")
    print("=" * 60)
    A2 = np.array([[4, 1, 1],
                    [1, 3, 1],
                    [1, 1, 2]], dtype=float)

    eigenvalue_inv, _, hist_inv = inverse_power_iteration(A2)
    print(f"  反幂迭代(无位移) - 特征值: {eigenvalue_inv:.12f}, 迭代次数: {len(hist_inv)}")

    ref = np.sort(np.abs(np.linalg.eigvals(A2)))
    print(f"  NumPy验证 - 模最小特征值: {ref[0]:.12f}")

    eigenvalue_inv_s, _, hist_inv_s = inverse_power_iteration(A2, shift=3.0)
    print(f"  反幂迭代(shift=3.0) - 特征值: {eigenvalue_inv_s:.12f}, 迭代次数: {len(hist_inv_s)}")
    print(f"  NumPy验证 - 最接近3.0的特征值: {ref[np.argmin(np.abs(ref - 3.0))]:.12f}")

    print()
    print("=" * 60)
    print("测试3: 瑞利商迭代 - 从任意初值收敛到附近特征值")
    print("=" * 60)
    A3 = np.array([[2, 1, 0],
                    [1, 3, 1],
                    [0, 1, 4]], dtype=float)
    ref = np.sort(np.linalg.eigvals(A3))
    print(f"  所有特征值(NumPy): {ref}")

    results = []
    for _ in range(5):
        eigenvalue_r, v_r, hist_r = rayleigh_quotient_iteration(A3)
        results.append(eigenvalue_r)
        print(f"  运行{_+1}: 收敛到 {eigenvalue_r:.8f}, 迭代 {len(hist_r)} 次")

    print(f"  收敛到的特征值集合: {sorted(set(round(x, 6) for x in results))}")

    print()
    print("=" * 60)
    print("测试4: 线性算子接口 (矩阵向量乘法函数)")
    print("=" * 60)
    A4 = np.array([[5, 2, 0],
                    [2, 4, 1],
                    [0, 1, 3]], dtype=float)

    def A4_op(v):
        return A4 @ v

    n4 = 3
    eigenvalue_p, _, hist_p = power_iteration(A4_op, n=n4)
    print(f"  幂迭代(算子)    - 特征值: {eigenvalue_p:.10f}, 迭代次数: {len(hist_p)}")

    def solve_shifted_4(mu, v):
        M = A4 - mu * np.eye(3)
        return np.linalg.solve(M, v)

    eigenvalue_r, _, hist_r = rayleigh_quotient_iteration(A4_op, n=n4, solve_shifted=solve_shifted_4)
    closest_idx = np.argmin(np.abs(np.linalg.eigvals(A4) - eigenvalue_r))
    print(f"  瑞利商迭代(算子) - 特征值: {eigenvalue_r:.10f} (λ{closest_idx+1}), 迭代次数: {len(hist_r)}")

    ref = np.sort(np.linalg.eigvals(A4))[::-1]
    print(f"  NumPy验证 - 最大特征值: {ref[0]:.10f}")

    print()
    print("=" * 60)
    print("测试5: 线性算子 + 反幂迭代 (提供自定义 solve)")
    print("=" * 60)
    A5 = np.array([[5, 2, 0],
                    [2, 4, 1],
                    [0, 1, 3]], dtype=float)

    def A5_op(v):
        return A5 @ v

    shift5 = 0.0
    M5 = A5 - shift5 * np.eye(3)
    lu5, piv5 = sla.lu_factor(M5)

    def solve5(v):
        return sla.lu_solve((lu5, piv5), v)

    eigenvalue_inv, _, hist_inv = inverse_power_iteration(A5_op, n=3, shift=shift5, solve=solve5)
    print(f"  反幂迭代(算子) - 特征值: {eigenvalue_inv:.10f}, 迭代次数: {len(hist_inv)}")

    ref = np.sort(np.abs(np.linalg.eigvals(A5)))
    print(f"  NumPy验证 - 模最小特征值: {ref[0]:.10f}")

    print()
    print("=" * 60)
    print("测试6: 稀疏矩阵 (SciPy sparse)")
    print("=" * 60)
    n6 = 50
    d_main = 2.0 * np.ones(n6)
    d_off = -1.0 * np.ones(n6 - 1)
    A6_sparse = sp.diags([d_off, d_main, d_off], [-1, 0, 1], format="csc")

    eigenvalue_p, _, hist_p = power_iteration(A6_sparse, tol=1e-6, max_iter=2000)
    print(f"  幂迭代(稀疏)    - 最大特征值: {eigenvalue_p:.10f}, 迭代次数: {len(hist_p)}")

    eigenvalue_inv, _, hist_inv = inverse_power_iteration(A6_sparse, shift=0.0)
    print(f"  反幂迭代(稀疏)  - 最小特征值: {eigenvalue_inv:.10f}, 迭代次数: {len(hist_inv)}")

    eigenvalue_r, _, hist_r = rayleigh_quotient_iteration(A6_sparse)
    closest_idx = np.argmin(np.abs(np.linalg.eigvals(A6_sparse.toarray()) - eigenvalue_r))
    print(f"  瑞利商迭代(稀疏) - 特征值: {eigenvalue_r:.10f} (λ{closest_idx+1}), 迭代次数: {len(hist_r)}")

    A6_dense = A6_sparse.toarray()
    ref = np.sort(np.linalg.eigvals(A6_dense))
    print(f"  NumPy验证 - 最小: {ref[0].real:.10f}, 最大: {ref[-1].real:.10f}")
    print(f"  说明: 三对角矩阵特征值密集 (2±2cos(kπ/(n+1))), 导致幂迭代收敛较慢")

    print()
    print("=" * 60)
    print("测试7: 瑞利商迭代 三次收敛验证")
    print("=" * 60)
    A7 = np.array([[3, 1, 0],
                    [1, 2, 1],
                    [0, 1, 1]], dtype=float)
    ref = np.linalg.eigvals(A7)
    print(f"  特征值: {ref}")

    eigenvalue_r, _, hist_r = rayleigh_quotient_iteration(A7, tol=1e-14)
    print(f"  收敛历史 (共 {len(hist_r)} 次):")
    for i, lam in enumerate(hist_r):
        err = abs(lam - ref[np.argmin(np.abs(ref - lam))])
        print(f"    迭代 {i}: λ = {lam:.14f}, 误差 = {err:.2e}")
    print(f"  观察: 误差从 {abs(hist_r[0] - hist_r[-1]):.1e} 快速降至 0, 显示三次收敛特征")

    print()
    print("=" * 60)
    print("测试8: 反幂迭代 + 位移 = 求指定特征值")
    print("=" * 60)
    A8 = np.diag([1.0, 2.5, 4.0, 6.0, 10.0])
    ref = np.linalg.eigvals(A8)
    print(f"  特征值: {ref}")

    targets = [0.5, 3.0, 7.0]
    for t in targets:
        eig, _, hist = inverse_power_iteration(A8, shift=t, tol=1e-12)
        print(f"  位移={t:5.1f}: 收敛到 {eig:.10f} (最接近的特征值 {ref[np.argmin(np.abs(ref - t))]:.1f}), 迭代 {len(hist)} 次")
