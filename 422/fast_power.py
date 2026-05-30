def fast_power(a, b, m=None):
    if b < 0:
        raise ValueError("指数b必须是非负整数")
    
    if m is not None and m == 1:
        return 0
    
    if b == 0:
        return 1 if m is None else 1 % m
    
    result = 1
    base = a
    
    if m is not None:
        base = base % m
    
    while b > 0:
        if b & 1:
            result = result * base
            if m is not None:
                result = result % m
        
        base = base * base
        if m is not None:
            base = base % m
        
        b >>= 1
    
    return result


def mat_mul(A, B, m=None):
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    
    if cols_a != rows_b:
        raise ValueError(f"矩阵维度不匹配: A为{rows_a}x{cols_a}, B为{rows_b}x{cols_b}")
    
    result = [[0] * cols_b for _ in range(rows_a)]
    
    for i in range(rows_a):
        for k in range(cols_a):
            if A[i][k] == 0:
                continue
            for j in range(cols_b):
                result[i][j] += A[i][k] * B[k][j]
                if m is not None:
                    result[i][j] %= m
    
    if m is not None:
        for i in range(rows_a):
            for j in range(cols_b):
                result[i][j] %= m
    
    return result


def mat_power(M, n, m=None):
    if n < 0:
        raise ValueError("指数n必须是非负整数")
    
    if m is not None and m == 1:
        size = len(M)
        return [[0] * size for _ in range(size)]
    
    size = len(M)
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    
    if n == 0:
        return result
    
    base = [row[:] for row in M]
    
    if m is not None:
        for i in range(size):
            for j in range(size):
                base[i][j] %= m
    
    while n > 0:
        if n & 1:
            result = mat_mul(result, base, m)
        
        base = mat_mul(base, base, m)
        n >>= 1
    
    return result


def fib_matrix(n, m=None):
    if n < 0:
        raise ValueError("n必须是非负整数")
    
    if n == 0:
        return 0
    if n == 1:
        return 1 if m is None else 1 % m
    
    F = [[1, 1], [1, 0]]
    Fn = mat_power(F, n - 1, m)
    return Fn[0][0]


def fib_naive(n, m=None):
    if n < 0:
        raise ValueError("n必须是非负整数")
    
    if n == 0:
        return 0
    if n == 1:
        return 1 if m is None else 1 % m
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
        if m is not None:
            b %= m
    
    return b


if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print("测试标量快速幂")
    print("=" * 50)
    print(f"2^10 = {fast_power(2, 10)}")
    print(f"3^13 = {fast_power(3, 13)}")
    print(f"2^1000 (前50位): {str(fast_power(2, 1000))[:50]}...")
    print(f"(2^10) % 1000 = {fast_power(2, 10, 1000)}")
    print(f"(3^13) % 7 = {fast_power(3, 13, 7)}")
    print(f"(123456789^987654321) % 1000000007 = {fast_power(123456789, 987654321, 1000000007)}")
    
    print("\n标量快速幂边界情况")
    print(f"5^0 = {fast_power(5, 0)} (预期: 1)")
    print(f"0^0 = {fast_power(0, 0)} (预期: 1)")
    print(f"(5^0) % 7 = {fast_power(5, 0, 7)} (预期: 1)")
    print(f"(5^0) % 1 = {fast_power(5, 0, 1)} (预期: 0)")
    print(f"(123^456) % 1 = {fast_power(123, 456, 1)} (预期: 0)")
    print(f"(0^0) % 1 = {fast_power(0, 0, 1)} (预期: 0)")
    
    print("\n" + "=" * 50)
    print("测试矩阵快速幂")
    print("=" * 50)
    
    I = [[1, 0], [0, 1]]
    M = [[1, 1], [1, 0]]
    print(f"单位矩阵 [[1,0],[0,1]]^5 = {mat_power(I, 5)}")
    print(f"[[1,1],[1,0]]^0 = {mat_power(M, 0)} (预期: 单位矩阵)")
    print(f"[[1,1],[1,0]]^1 = {mat_power(M, 1)}")
    print(f"[[1,1],[1,0]]^5 = {mat_power(M, 5)}")
    print(f"[[1,1],[1,0]]^5 % 10 = {mat_power(M, 5, 10)}")
    
    print("\n" + "=" * 50)
    print("测试斐波那契数列（矩阵法 vs 递推法）")
    print("=" * 50)
    
    fib_tests = [0, 1, 2, 5, 10, 20, 50]
    print(f"{'n':>6} | {'矩阵法':>20} | {'递推法':>20} | {'一致':>4}")
    print("-" * 60)
    for n in fib_tests:
        fm = fib_matrix(n)
        fn = fib_naive(n)
        print(f"{n:>6} | {fm:>20} | {fn:>20} | {'✓' if fm == fn else '✗':>4}")
    
    print(f"\nfib(50) % 1000000007 = {fib_matrix(50, 1000000007)}")
    print(f"fib(100) % 1000000007 = {fib_matrix(100, 1000000007)}")
    
    print("\n" + "=" * 50)
    print("性能对比：矩阵快速幂 vs 普通递推")
    print("=" * 50)
    
    test_ns = [10000, 100000, 500000, 1000000]
    MOD = 1000000007
    
    print(f"{'n':>10} | {'矩阵法(ms)':>12} | {'递推法(ms)':>12} | {'加速比':>8}")
    print("-" * 55)
    
    for n in test_ns:
        start = time.perf_counter()
        result_m = fib_matrix(n, MOD)
        t_matrix = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        result_n = fib_naive(n, MOD)
        t_naive = (time.perf_counter() - start) * 1000
        
        assert result_m == result_n, f"结果不一致: n={n}, 矩阵法={result_m}, 递推法={result_n}"
        ratio = t_naive / t_matrix if t_matrix > 0 else float('inf')
        print(f"{n:>10} | {t_matrix:>12.3f} | {t_naive:>12.3f} | {ratio:>7.2f}x")
