import numpy as np
from collections import defaultdict


def qr_classic_gram_schmidt(A):
    A = np.array(A, dtype=float)
    m, n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]

        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise ValueError(f"Column {j} is linearly dependent on previous columns.")

        R[j, j] = norm
        Q[:, j] = v / norm

    return Q, R


def qr_modified_gram_schmidt(A):
    A = np.array(A, dtype=float)
    m, n = A.shape

    V = A.copy()
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] < 1e-12:
            raise ValueError(f"Column {i} is linearly dependent on previous columns.")
        Q[:, i] = V[:, i] / R[i, i]

        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] -= R[i, j] * Q[:, i]

    return Q, R


def qr_householder(A):
    A = np.array(A, dtype=float)
    m, n = A.shape

    R = A.copy()
    Q = np.eye(m)

    for k in range(min(m - 1, n)):
        x = R[k:, k].copy()
        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        if alpha == 0:
            continue
        v = x.copy()
        v[0] -= alpha
        v = v / np.linalg.norm(v)

        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)

    Q = Q[:, :n]
    R = R[:n, :]

    return Q, R


def qr_givens(A):
    A = np.array(A, dtype=float)
    m, n = A.shape

    R = A.copy()
    Q = np.eye(m)

    for j in range(n):
        for i in range(m - 1, j, -1):
            a = R[i - 1, j]
            b = R[i, j]
            if abs(b) < 1e-14:
                continue

            r = np.hypot(a, b)
            c = a / r
            s = b / r

            G = np.array([[c, s], [-s, c]])
            R[i - 1:i + 1, j:] = G @ R[i - 1:i + 1, j:]
            Q[:, i - 1:i + 1] = Q[:, i - 1:i + 1] @ G.T

    Q = Q[:, :n]
    R = R[:n, :]

    return Q, R


class SparseMatrix:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.rows = defaultdict(dict)

    @classmethod
    def from_dense(cls, A):
        A = np.array(A, dtype=float)
        m, n = A.shape
        sp = cls(m, n)
        for i in range(m):
            for j in range(n):
                if abs(A[i, j]) > 1e-14:
                    sp.rows[i][j] = A[i, j]
        return sp

    def to_dense(self):
        A = np.zeros((self.m, self.n))
        for i, cols in self.rows.items():
            for j, val in cols.items():
                A[i, j] = val
        return A

    def get(self, i, j):
        return self.rows[i].get(j, 0.0)

    def set(self, i, j, val):
        if abs(val) < 1e-14:
            if j in self.rows[i]:
                del self.rows[i][j]
        else:
            self.rows[i][j] = val

    def row_nz(self, i):
        return sorted(self.rows[i].items())

    def col_nz(self, j):
        result = []
        for i in range(self.m):
            if j in self.rows[i]:
                result.append((i, self.rows[i][j]))
        return result

    def multiply_vector(self, v):
        result = np.zeros(self.m)
        for i, cols in self.rows.items():
            s = 0.0
            for j, val in cols.items():
                s += val * v[j]
            result[i] = s
        return result


def qr_sparse_givens(A_sparse):
    m, n = A_sparse.m, A_sparse.n
    R = SparseMatrix(m, n)
    for i in range(m):
        for j, val in A_sparse.rows[i].items():
            R.set(i, j, val)

    rotations = []

    for j in range(n):
        for i in range(m - 1, j, -1):
            a = R.get(i - 1, j)
            b = R.get(i, j)
            if abs(b) < 1e-14:
                continue

            r = np.hypot(a, b)
            c = a / r
            s = b / r

            rotations.append((i, c, s))

            for k in range(j, n):
                r_ik = R.get(i, k)
                r_i1k = R.get(i - 1, k)
                new_i1 = c * r_i1k + s * r_ik
                new_i = -s * r_i1k + c * r_ik
                R.set(i - 1, k, new_i1)
                R.set(i, k, new_i)

    Q = np.eye(m)
    for i, c, s in reversed(rotations):
        G_T = np.array([[c, -s], [s, c]])
        Q[i - 1:i + 1, :] = G_T @ Q[i - 1:i + 1, :]

    Q = Q[:, :n]
    R_dense = np.zeros((n, n))
    for i in range(n):
        for j, val in R.rows[i].items():
            if j < n:
                R_dense[i, j] = val

    return Q, R_dense


def solve_triangular(R, b):
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= R[i, j] * x[j]
        x[i] = s / R[i, i]
    return x


def least_squares_qr(A, b, method="householder"):
    methods = {
        "classic_gs": qr_classic_gram_schmidt,
        "modified_gs": qr_modified_gram_schmidt,
        "householder": qr_householder,
        "givens": qr_givens,
    }
    if method not in methods:
        raise ValueError(f"Unknown method: {method}")

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    Q, R = methods[method](A)
    Q_full = np.zeros((m, m))
    Q_full[:, :n] = Q
    for i in range(n, m):
        Q_full[i, i] = 1.0

    Qb = Q.T @ b
    x = solve_triangular(R, Qb[:n])

    residual = np.linalg.norm(A @ x - b)
    return x, residual


def least_squares_sparse_qr(A_sparse, b):
    Q, R = qr_sparse_givens(A_sparse)
    Qb = Q.T @ b
    x = solve_triangular(R, Qb[:R.shape[0]])
    residual = np.linalg.norm(A_sparse.multiply_vector(x) - b)
    return x, residual


def hilbert_matrix(n):
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])


def banded_matrix(m, n, bandwidth=3):
    A = np.zeros((m, n))
    for i in range(m):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            A[i, j] = 1.0 / (abs(i - j) + 1)
    return A


if __name__ == "__main__":
    print("=" * 70)
    print("Test 1: 10x10 Hilbert matrix - Orthogonality")
    print("=" * 70)

    H = hilbert_matrix(10)
    methods = [
        ("Classic GS", qr_classic_gram_schmidt),
        ("Modified GS", qr_modified_gram_schmidt),
        ("Householder", qr_householder),
        ("Givens", qr_givens),
    ]

    for name, func in methods:
        Q, R = func(H)
        orth_err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), ord="fro")
        reconst_err = np.linalg.norm(H - Q @ R, ord="fro")
        print(f"\n{name:12s}: orth_err={orth_err:.2e}, reconst_err={reconst_err:.2e}")

    print("\n" + "=" * 70)
    print("Test 2: Least Squares on random overdetermined system")
    print("=" * 70)

    np.random.seed(42)
    m, n = 50, 10
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true + 0.1 * np.random.randn(m)

    for method in ["classic_gs", "modified_gs", "householder", "givens"]:
        x, resid = least_squares_qr(A, b, method=method)
        err = np.linalg.norm(x - x_true)
        print(f"  {method:12s}: ||x - x_true||={err:.2e}, residual={resid:.2e}")

    print("\n" + "=" * 70)
    print("Test 3: Sparse QR (Givens) on banded matrix")
    print("=" * 70)

    A_banded = banded_matrix(20, 10, bandwidth=2)
    A_sparse = SparseMatrix.from_dense(A_banded)

    print(f"  Dense size: {A_banded.size} elements")
    print(f"  Sparse nnz: {sum(len(cols) for cols in A_sparse.rows.values())} elements")

    Q_sp, R_sp = qr_sparse_givens(A_sparse)
    Q_de, R_de = qr_givens(A_banded)

    orth_sparse = np.linalg.norm(Q_sp.T @ Q_sp - np.eye(10))
    orth_dense = np.linalg.norm(Q_de.T @ Q_de - np.eye(10))
    reconst_sparse = np.linalg.norm(A_banded - Q_sp @ R_sp)
    reconst_dense = np.linalg.norm(A_banded - Q_de @ R_de)

    print(f"\n  Sparse Givens:  orth_err={orth_sparse:.2e}, reconst_err={reconst_sparse:.2e}")
    print(f"  Dense Givens:   orth_err={orth_dense:.2e}, reconst_err={reconst_dense:.2e}")

    print("\n" + "=" * 70)
    print("Test 4: Sparse Least Squares")
    print("=" * 70)

    b_sparse = A_banded @ np.ones(10) + 0.01 * np.random.randn(20)
    x_sp, resid_sp = least_squares_sparse_qr(A_sparse, b_sparse)
    x_de, resid_de = least_squares_qr(A_banded, b_sparse, method="givens")

    print(f"  Sparse solution:  {x_sp[:5]}...")
    print(f"  Dense solution:   {x_de[:5]}...")
    print(f"  ||x_sp - x_de|| = {np.linalg.norm(x_sp - x_de):.2e}")
    print(f"  Sparse residual = {resid_sp:.2e}, Dense residual = {resid_de:.2e}")
