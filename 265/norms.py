import numpy as np


def vector_p_norm(v, p):
    v = np.array(v, dtype=float)
    if p == 0:
        return np.count_nonzero(v)
    if p == np.inf:
        return np.max(np.abs(v))
    if p == -np.inf:
        return np.min(np.abs(v))
    return np.sum(np.abs(v) ** p) ** (1.0 / p)


def vector_norms(v):
    v = np.array(v, dtype=float)
    norm_0 = np.count_nonzero(v)
    norm_1 = np.sum(np.abs(v))
    norm_2 = np.sqrt(np.sum(v ** 2))
    norm_inf = np.max(np.abs(v))
    return norm_0, norm_1, norm_2, norm_inf


def matrix_norms(A):
    A = np.array(A, dtype=float)
    norm_1 = np.max(np.sum(np.abs(A), axis=0))
    norm_f = np.sqrt(np.sum(A ** 2))
    norm_inf = np.max(np.sum(np.abs(A), axis=1))
    return norm_1, norm_f, norm_inf


def nuclear_norm(A):
    A = np.array(A, dtype=float)
    return np.sum(np.linalg.svd(A, compute_uv=False))


def schatten_p_norm(A, p):
    A = np.array(A, dtype=float)
    sigma = np.linalg.svd(A, compute_uv=False)
    if p == 0:
        return np.count_nonzero(sigma)
    if p == np.inf:
        return np.max(sigma)
    if p == -np.inf:
        return np.min(sigma[sigma > 0]) if np.any(sigma > 0) else 0.0
    return np.sum(sigma ** p) ** (1.0 / p)


if __name__ == "__main__":
    v = [3, -4, 5]
    print(f"向量 v = {v}")
    n0, n1, n2, ninf = vector_norms(v)
    print(f"  0-范数 (非零元素个数): {n0}")
    print(f"  1-范数:               {n1}")
    print(f"  2-范数:               {n2}")
    print(f"  无穷范数:             {ninf}")

    print()

    print("通过 vector_p_norm 计算:")
    for p in [0, 1, 2, np.inf]:
        val = vector_p_norm(v, p)
        label = "∞" if p == np.inf else str(int(p))
        print(f"  p={label}: {val}")

    print()

    v_sparse = [0, -4, 0, 0, 5, 0]
    print(f"稀疏向量 v_sparse = {v_sparse}")
    n0s, n1s, n2s, ninfs = vector_norms(v_sparse)
    print(f"  0-范数 (非零元素个数): {n0s}")
    print(f"  1-范数:               {n1s}")
    print(f"  2-范数:               {n2s}")
    print(f"  无穷范数:             {ninfs}")

    print()

    A = [[1, 2, -3],
         [-4, 5, 6],
         [7, -8, 9]]
    print(f"矩阵 A = {A}")
    m1, mf, minf = matrix_norms(A)
    print(f"  1-范数 (列和最大):   {m1}")
    print(f"  Frobenius范数:       {mf}")
    print(f"  无穷范数 (行和最大): {minf}")

    print()

    nn = nuclear_norm(A)
    print(f"  核范数 (奇异值之和): {nn}")

    print()

    print("Schatten-p 范数:")
    for p in [0, 0.5, 1, 2, np.inf]:
        val = schatten_p_norm(A, p)
        label = "∞" if p == np.inf else str(p)
        note = ""
        if p == 0:
            note = " (矩阵秩)"
        elif p == 1:
            note = " = 核范数"
        elif p == 2:
            note = " = Frobenius范数"
        elif p == np.inf:
            note = " = 谱范数"
        print(f"  p={label}: {val}{note}")

    print()

    sigma = np.linalg.svd(np.array(A, dtype=float), compute_uv=False)
    print(f"  奇异值: {sigma}")
