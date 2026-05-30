import time
from typing import List, Tuple, Union, Optional

Number = Union[int, float]
Matrix = List[List[Number]]

BLOCK_SIZE = 64
STRASSEN_THRESHOLD = 64


def _validate_matrices(A: Matrix, B: Matrix) -> Tuple[int, int, int, int]:
    cols_a = len(A[0])
    rows_b = len(B)

    if cols_a != rows_b:
        raise ValueError(
            f"矩阵A的列数({cols_a})不等于矩阵B的行数({rows_b})，无法相乘"
        )

    for i, row in enumerate(A):
        if len(row) != cols_a:
            raise ValueError(f"矩阵A的第{i}行列数不一致，期望{cols_a}，实际{len(row)}")

    cols_b = len(B[0])
    for i, row in enumerate(B):
        if len(row) != cols_b:
            raise ValueError(f"矩阵B的第{i}行列数不一致，期望{cols_b}，实际{len(row)}")

    return len(A), cols_a, rows_b, cols_b


def _add(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    m = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]


def _sub(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    m = len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]


def _pad_matrix(A: Matrix, target: int) -> Matrix:
    n = len(A)
    m = len(A[0])
    if n == target and m == target:
        return A
    padded = [[0.0] * target for _ in range(target)]
    for i in range(n):
        for j in range(m):
            padded[i][j] = A[i][j]
    return padded


def _unpad_matrix(A: Matrix, rows: int, cols: int) -> Matrix:
    return [[A[i][j] for j in range(cols)] for i in range(rows)]


def _strassen_multiply(A: Matrix, B: Matrix, n: int) -> Matrix:
    if n <= STRASSEN_THRESHOLD:
        result = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                a_ik = A[i][k]
                for j in range(n):
                    result[i][j] += a_ik * B[k][j]
        return result

    half = n // 2

    A11 = [[A[i][j] for j in range(half)] for i in range(half)]
    A12 = [[A[i][j] for j in range(half, n)] for i in range(half)]
    A21 = [[A[i][j] for j in range(half)] for i in range(half, n)]
    A22 = [[A[i][j] for j in range(half, n)] for i in range(half, n)]

    B11 = [[B[i][j] for j in range(half)] for i in range(half)]
    B12 = [[B[i][j] for j in range(half, n)] for i in range(half)]
    B21 = [[B[i][j] for j in range(half)] for i in range(half, n)]
    B22 = [[B[i][j] for j in range(half, n)] for i in range(half, n)]

    M1 = _strassen_multiply(_add(A11, A22), _add(B11, B22), half)
    M2 = _strassen_multiply(_add(A21, A22), B11, half)
    M3 = _strassen_multiply(A11, _sub(B12, B22), half)
    M4 = _strassen_multiply(A22, _sub(B21, B11), half)
    M5 = _strassen_multiply(_add(A11, A12), B22, half)
    M6 = _strassen_multiply(_sub(A21, A11), _add(B11, B12), half)
    M7 = _strassen_multiply(_sub(A12, A22), _add(B21, B22), half)

    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)

    C = [[0.0] * n for _ in range(n)]
    for i in range(half):
        for j in range(half):
            C[i][j] = C11[i][j]
            C[i][j + half] = C12[i][j]
            C[i + half][j] = C21[i][j]
            C[i + half][j + half] = C22[i][j]

    return C


def strassen_multiply(A: Matrix, B: Matrix) -> Tuple[Matrix, float]:
    rows_a, cols_a, rows_b, cols_b = _validate_matrices(A, B)

    start = time.perf_counter()

    max_dim = max(rows_a, cols_a, cols_b)
    n = 1
    while n < max_dim:
        n <<= 1

    A_pad = _pad_matrix(A, n)
    B_pad = _pad_matrix(B, n)

    C_pad = _strassen_multiply(A_pad, B_pad, n)

    result = _unpad_matrix(C_pad, rows_a, cols_b)

    elapsed = time.perf_counter() - start

    return result, elapsed


def matrix_multiply(
    A: Matrix,
    B: Matrix,
    block_size: int = BLOCK_SIZE,
    method: str = "block"
) -> Tuple[Matrix, float]:
    if method == "strassen":
        return strassen_multiply(A, B)
    elif method == "block":
        pass
    else:
        raise ValueError(f"未知的乘法方法: {method}，可选 'block' 或 'strassen'")

    rows_a, cols_a, rows_b, cols_b = _validate_matrices(A, B)

    start = time.perf_counter()

    result = [[0] * cols_b for _ in range(rows_a)]

    for ii in range(0, rows_a, block_size):
        i_end = min(ii + block_size, rows_a)
        for kk in range(0, cols_a, block_size):
            k_end = min(kk + block_size, cols_a)
            for jj in range(0, cols_b, block_size):
                j_end = min(jj + block_size, cols_b)
                for i in range(ii, i_end):
                    for k in range(kk, k_end):
                        a_ik = A[i][k]
                        for j in range(jj, j_end):
                            result[i][j] += a_ik * B[k][j]

    elapsed = time.perf_counter() - start

    return result, elapsed


def optimal_matrix_chain(dims: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
    n = len(dims) - 1
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    return m, s


def _build_chain(s: List[List[int]], i: int, j: int) -> str:
    if i == j:
        return f"A{i}"
    else:
        return f"({_build_chain(s, i, s[i][j])} × {_build_chain(s, s[i][j] + 1, j)})"


def get_optimal_chain_order(dims: List[int]) -> Tuple[str, int]:
    if len(dims) < 2:
        raise ValueError("维度列表至少需要2个元素")

    n = len(dims) - 1
    for i, d in enumerate(dims):
        if d <= 0:
            raise ValueError(f"维度必须为正整数，第{i}个维度为{d}")

    m, s = optimal_matrix_chain(dims)
    order = _build_chain(s, 0, n - 1)
    min_ops = m[0][n - 1]

    return order, min_ops


def matrix_chain_multiply(
    matrices: List[Matrix],
    method: str = "block",
    block_size: int = BLOCK_SIZE
) -> Tuple[Matrix, float, str]:
    if len(matrices) < 2:
        raise ValueError("至少需要2个矩阵才能进行链乘法")

    dims = []
    dims.append(len(matrices[0]))
    for i, mat in enumerate(matrices):
        if len(mat[0]) != len(matrices[i + 1]) if i + 1 < len(matrices) else True:
            pass
        dims.append(len(mat[0]))

    for i in range(len(matrices) - 1):
        cols_i = len(matrices[i][0])
        rows_next = len(matrices[i + 1])
        if cols_i != rows_next:
            raise ValueError(
                f"矩阵{i}的列数({cols_i})不等于矩阵{i+1}的行数({rows_next})"
            )

    dims = [len(matrices[0])] + [len(mat[0]) for mat in matrices]

    _, s = optimal_matrix_chain(dims)

    def _multiply_chain(i: int, j: int) -> Matrix:
        if i == j:
            return matrices[i]
        k = s[i][j]
        left = _multiply_chain(i, k)
        right = _multiply_chain(k + 1, j)
        result, _ = matrix_multiply(left, right, block_size=block_size, method=method)
        return result

    start = time.perf_counter()
    n = len(matrices)
    result = _multiply_chain(0, n - 1)
    elapsed = time.perf_counter() - start

    order = _build_chain(s, 0, n - 1)

    return result, elapsed, order


if __name__ == "__main__":
    A = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    B = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]

    result_block, elapsed_block = matrix_multiply(A, B, method="block")
    result_strassen, elapsed_strassen = matrix_multiply(A, B, method="strassen")

    print("=== 普通分块乘法 ===")
    print("乘积矩阵 A × B:")
    for row in result_block:
        print(f"  {row}")
    print(f"计算耗时: {elapsed_block:.6f} 秒")

    print()
    print("=== Strassen 分治乘法 ===")
    print("乘积矩阵 A × B:")
    for row in result_strassen:
        print(f"  {row}")
    print(f"计算耗时: {elapsed_strassen:.6f} 秒")

    print()
    print("=== 矩阵链乘法最优顺序 ===")
    dims = [10, 20, 50, 1, 100]
    order, min_ops = get_optimal_chain_order(dims)
    print(f"维度序列: {dims}")
    print(f"最优结合顺序: {order}")
    print(f"最少标量乘法次数: {min_ops}")

    print()
    print("=== 矩阵链乘法示例 ===")
    A1 = [[1, 2], [3, 4], [5, 6]]
    A2 = [[1, 2, 3], [4, 5, 6]]
    A3 = [[1], [2], [3]]
    chain_result, chain_time, chain_order = matrix_chain_multiply([A1, A2, A3])
    print(f"结合顺序: {chain_order}")
    print(f"链乘结果:")
    for row in chain_result:
        print(f"  {row}")
    print(f"计算耗时: {chain_time:.6f} 秒")
