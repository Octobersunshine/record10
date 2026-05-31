def matrix_mult(a, b, mod):
    return [
        [(a[0][0] * b[0][0] + a[0][1] * b[1][0]) % mod,
         (a[0][0] * b[0][1] + a[0][1] * b[1][1]) % mod],
        [(a[1][0] * b[0][0] + a[1][1] * b[1][0]) % mod,
         (a[1][0] * b[0][1] + a[1][1] * b[1][1]) % mod]
    ]

def matrix_pow(mat, power, mod):
    result = [[1, 0], [0, 1]]
    while power > 0:
        if power % 2 == 1:
            result = matrix_mult(result, mat, mod)
        mat = matrix_mult(mat, mat, mod)
        power //= 2
    return result

def fibonacci_mod_iter(n, mod):
    if mod == 1:
        return 0
    if n == 0:
        return 0 % mod
    if n == 1:
        return 1 % mod
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, (a + b) % mod
    return b % mod

def pisano_period(mod):
    if mod == 1:
        return 1
    a, b = 0, 1
    for i in range(1, 6 * mod + 1):
        a, b = b, (a + b) % mod
        if a == 0 and b == 1:
            return i
    return 6 * mod

def fibonacci_mod(n, mod, use_pisano='auto', pisano_threshold=10000):
    if mod == 1:
        return 0
    if use_pisano == 'auto':
        use_pisano = mod <= pisano_threshold
    if use_pisano:
        period = pisano_period(mod)
        n = n % period
    if n == 0:
        return 0 % mod
    if n == 1:
        return 1 % mod
    fib_matrix = [[1, 1], [1, 0]]
    mat_n = matrix_pow(fib_matrix, n - 1, mod)
    return mat_n[0][0] % mod

def batch_fibonacci_mod(queries, use_pisano='auto'):
    results = []
    for n, mod in queries:
        results.append(fibonacci_mod(n, mod, use_pisano=use_pisano))
    return results

def test_pisano_period():
    test_cases = [
        (1, 1),
        (2, 3),
        (3, 8),
        (4, 6),
        (5, 20),
        (6, 24),
        (7, 16),
        (8, 12),
        (9, 24),
        (10, 60),
    ]
    print("Testing Pisano period:")
    all_pass = True
    for mod, expected in test_cases:
        result = pisano_period(mod)
        status = "✓" if result == expected else "✗"
        print(f"{status} π({mod}) = {result} (expected: {expected})")
        if result != expected:
            all_pass = False
    if all_pass:
        print("All Pisano period tests passed!")
    return all_pass

def test_fibonacci_mod():
    test_cases = [
        (0, 1, 0, {}),
        (1, 1, 0, {}),
        (100, 1, 0, {}),
        (0, 100, 0, {}),
        (1, 100, 1, {}),
        (2, 100, 1, {}),
        (3, 100, 2, {}),
        (4, 100, 3, {}),
        (5, 100, 5, {}),
        (6, 100, 8, {}),
        (10, 1000, 55, {}),
        (10, 7, 6, {}),
        (20, 100, 65, {}),
        (10**6, 10**9 + 7, 918091266, {'use_pisano': False}),
        (10**6, 1000, 875, {}),
    ]
    print("Testing fibonacci_mod:")
    all_pass = True
    for n, mod, expected, kwargs in test_cases:
        result = fibonacci_mod(n, mod, **kwargs)
        status = "✓" if result == expected else "✗"
        print(f"{status} F({n}) mod {mod} = {result} (expected: {expected})")
        if result != expected:
            all_pass = False
    if all_pass:
        print("All fibonacci_mod tests passed!")
    return all_pass

def test_batch():
    queries = [(5, 100), (10, 7), (15, 1000), (20, 100)]
    expected = [5, 6, 610, 65]
    results = batch_fibonacci_mod(queries)
    print("\nTesting batch_fibonacci_mod:")
    all_pass = True
    for (n, mod), result, exp in zip(queries, results, expected):
        status = "✓" if result == exp else "✗"
        print(f"{status} F({n}) mod {mod} = {result} (expected: {exp})")
        if result != exp:
            all_pass = False
    if all_pass:
        print("All batch tests passed!")
    return all_pass

def performance_comparison():
    import time
    print("\n" + "=" * 60)
    print("Performance Comparison: Iterative vs Fast Matrix Power")
    print("=" * 60)
    
    mod = 10**9 + 7
    
    for n in [10**3, 10**4, 10**5, 10**6, 10**7]:
        print(f"\nn = {n}, mod = {mod}")
        
        start = time.time()
        r1 = fibonacci_mod_iter(n, mod)
        iter_time = time.time() - start
        
        start = time.time()
        r2 = fibonacci_mod(n, mod, use_pisano=False)
        fast_time = time.time() - start
        
        assert r1 == r2, f"Results differ: {r1} vs {r2}"
        
        speedup = iter_time / fast_time if fast_time > 0 else float('inf')
        print(f"  Iterative:   {iter_time:.6f}s, result = {r1}")
        print(f"  Fast Power:  {fast_time:.6f}s, result = {r2}")
        print(f"  Speedup:     {speedup:.2f}x")
    
    print(f"\n{'=' * 60}")
    print("Very large n (only fast power):")
    print(f"{'=' * 60}")
    for n in [10**12, 10**15, 10**18]:
        start = time.time()
        result = fibonacci_mod(n, mod, use_pisano=False)
        fast_time = time.time() - start
        print(f"n = {n}: {fast_time:.6f}s, F(n) mod {mod} = {result}")

if __name__ == "__main__":
    all_pass = True
    all_pass &= test_pisano_period()
    print()
    all_pass &= test_fibonacci_mod()
    all_pass &= test_batch()
    
    if not all_pass:
        print("\nSome tests failed!")
        exit(1)
    
    performance_comparison()
