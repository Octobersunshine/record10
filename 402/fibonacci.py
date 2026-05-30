import time


def fibonacci(n):
    if n <= 0:
        return []
    result = [0, 1]
    for i in range(2, n + 1):
        result.append(result[i - 1] + result[i - 2])
    return result


def fib_nth_iterative(n, mod=None):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0 % mod if mod else 0
    a, b = 0, 1
    for _ in range(2, n + 1):
        c = a + b
        if mod:
            c %= mod
        a, b = b, c
    return b


def _fast_doubling(n, mod):
    if n == 0:
        return (0, 1)
    a, b = _fast_doubling(n >> 1, mod)
    c = a * ((2 * b - a) % mod) % mod
    d = (a * a + b * b) % mod
    if n & 1:
        return (d, (c + d) % mod)
    else:
        return (c, d)


def fib_nth_fast(n, mod=None):
    if n < 0:
        raise ValueError("n must be non-negative")
    if mod is None:
        mod = 1 << 1024
        result = _fast_doubling(n, mod)[0]
        return result
    return _fast_doubling(n, mod)[0]


if __name__ == "__main__":
    for n in [0, 1, 2, 10]:
        print(f"n={n}: {fibonacci(n)}")

    n_large = 10000
    result_large = fibonacci(n_large)
    print(f"\nn={n_large} 最后一个数的位数: {len(str(result_large[-1]))}")

    M = 10**9 + 7
    test_cases = [10, 100, 1000, 10000, 10**6, 10**9]

    print(f"\n=== 正确性验证 (mod {M}) ===")
    for n in test_cases[:4]:
        iter_val = fib_nth_iterative(n, M)
        fast_val = fib_nth_fast(n, M)
        status = "✓" if iter_val == fast_val else "✗"
        print(f"n={n}: 迭代={iter_val}, 快速倍乘={fast_val} {status}")

    print(f"\n=== 性能对比 ===")
    for n in test_cases:
        if n <= 10**6:
            t0 = time.perf_counter()
            v1 = fib_nth_iterative(n, M)
            t1 = time.perf_counter()
            t_iter = (t1 - t0) * 1000
        else:
            t_iter = None
            v1 = None

        t0 = time.perf_counter()
        v2 = fib_nth_fast(n, M)
        t1 = time.perf_counter()
        t_fast = (t1 - t0) * 1000

        if t_iter is not None:
            speedup = t_iter / t_fast if t_fast > 0 else float('inf')
            print(f"n={n:>9}: 迭代={t_iter:8.3f}ms, 快速倍乘={t_fast:8.3f}ms, 加速比={speedup:6.1f}x, 结果一致={v1==v2}")
        else:
            print(f"n={n:>9}: 迭代= (跳过), 快速倍乘={t_fast:8.3f}ms, F(n) mod {M} = {v2}")
