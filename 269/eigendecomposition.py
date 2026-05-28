import warnings
import numpy as np
from scipy.linalg import schur


def is_symmetric(A, tol=1e-8):
    A = np.array(A, dtype=float)
    return np.allclose(A, A.T, atol=tol)


def jacobi_eigenvalues(A, tol=1e-10, max_iter=1000):
    A = A.copy()
    n = A.shape[0]
    V = np.eye(n)

    for _ in range(max_iter):
        p, q = np.triu_indices(n, 1)
        off_diag = np.abs(A[p, q])
        max_idx = np.argmax(off_diag)
        p, q = p[max_idx], q[max_idx]
        a_pq = A[p, q]

        if abs(a_pq) < tol:
            break

        theta = (A[q, q] - A[p, p]) / (2 * a_pq)
        t = np.sign(theta) / (abs(theta) + np.sqrt(theta**2 + 1)) if theta != 0 else 1.0
        c = 1 / np.sqrt(1 + t**2)
        s = t * c

        J = np.eye(n)
        J[p, p] = c
        J[q, q] = c
        J[p, q] = s
        J[q, p] = -s

        A = J.T @ A @ J
        V = V @ J

    eigenvalues = np.diag(A)
    return eigenvalues, V


def sort_eigenvalues_descending(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors


def eigendecomposition(A):
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("输入必须是方阵")

    if is_symmetric(A):
        eigenvalues, P = jacobi_eigenvalues(A)
        eigenvalues, P = sort_eigenvalues_descending(eigenvalues, P)
        D = np.diag(eigenvalues)
        return D, P

    eigenvalues, P = np.linalg.eig(A)

    has_complex_eig = np.iscomplexobj(eigenvalues) and not np.allclose(eigenvalues.imag, 0)
    det = np.linalg.det(P)
    is_defective = np.isclose(det, 0)

    if is_defective or has_complex_eig:
        warn_msg = "矩阵为亏损矩阵（不可对角化），" if is_defective else "矩阵包含复数特征值，"
        warn_msg += "已使用 Schur 分解作为替代：A = Q T Qᵀ。"
        warn_msg += " Jordan 标准形不适合数值计算，因其对扰动极度敏感。"
        warnings.warn(warn_msg, UserWarning, stacklevel=2)

        T, Q = schur(A)
        return T, Q

    eigenvalues = eigenvalues.real
    P = P.real
    eigenvalues, P = sort_eigenvalues_descending(eigenvalues, P)
    D = np.diag(eigenvalues)
    return D, P


def verify_decomposition(A, D, P, tol=1e-8):
    A = np.array(A, dtype=float)
    D = np.array(D, dtype=float)
    P = np.array(P, dtype=float)

    is_diagonal = np.allclose(D - np.diag(np.diagonal(D)), 0, atol=tol)

    if is_diagonal:
        P_inv = np.linalg.inv(P)
        reconstructed = P @ D @ P_inv
    else:
        reconstructed = P @ D @ P.T

    return np.allclose(reconstructed, A, atol=tol)


if __name__ == "__main__":
    A1 = [[4, 1], [2, 3]]
    print("=== 测试1：可对角化矩阵 ===")
    D1, P1 = eigendecomposition(A1)
    print("原始矩阵 A:")
    print(A1)
    print("\n特征值矩阵 D (降序):")
    print(D1)
    print("\n特征向量矩阵 P:")
    print(P1)
    print("\n验证结果:", verify_decomposition(A1, D1, P1))

    A2 = [[1, 1], [0, 1]]
    print("\n=== 测试2：亏损矩阵（Jordan 块） ===")
    D2, P2 = eigendecomposition(A2)
    print("原始矩阵 A:")
    print(A2)
    print("\nSchur 上三角阵 T:")
    print(D2)
    print("\n正交矩阵 Q:")
    print(P2)
    print("\n验证 A = Q T Qᵀ:")
    print(P2 @ D2 @ P2.T)
    print("\n验证结果:", verify_decomposition(A2, D2, P2))

    A3 = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
    print("\n=== 测试3：实对称矩阵（Jacobi 算法） ===")
    D3, P3 = eigendecomposition(A3)
    print("原始矩阵 A:")
    print(A3)
    print("是否对称:", is_symmetric(A3))
    print("\n特征值矩阵 D (降序):")
    print(D3)
    print("\n特征向量矩阵 P (正交):")
    print(P3)
    print("\n验证 Pᵀ P = I (正交性):")
    print(np.allclose(P3.T @ P3, np.eye(3), atol=1e-8))
    print("\n验证 A = P D Pᵀ:")
    print(P3 @ D3 @ P3.T)
    print("验证结果:", verify_decomposition(A3, D3, P3))
