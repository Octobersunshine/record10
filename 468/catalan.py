def catalan(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    c = 1
    for i in range(n):
        c = c * 2 * (2 * i + 1) // (i + 2)
    return c


def catalan_mod(n, p):
    if n < 0:
        raise ValueError("n must be non-negative")
    c = 1 % p
    for i in range(n):
        inv = pow(i + 2, p - 2, p)
        c = c * 2 * (2 * i + 1) % p
        c = c * inv % p
    return c


def catalan_prefix(n, p=None):
    if n < 0:
        raise ValueError("n must be non-negative")
    res = [0] * (n + 1)
    res[0] = 1
    if p is None:
        for i in range(n):
            res[i + 1] = res[i] * 2 * (2 * i + 1) // (i + 2)
    else:
        for i in range(n):
            inv = pow(i + 2, p - 2, p)
            res[i + 1] = res[i] * 2 * (2 * i + 1) % p
            res[i + 1] = res[i + 1] * inv % p
    return res


def count_binary_trees(n, p=None):
    if p is None:
        return catalan(n)
    return catalan_mod(n, p)


if __name__ == "__main__":
    print("=== Basic Catalan Numbers ===")
    test_cases = [0, 1, 2, 3, 4, 5, 10, 20, 100, 1000]
    for n in test_cases:
        c = catalan(n)
        if n <= 20:
            print(f"C_{n} = {c}")
        else:
            s = str(c)
            print(f"C_{n} = {s[:20]}...({len(s)} digits)")

    print("\n=== Modular Arithmetic (p = 10^9+7) ===")
    MOD = 10**9 + 7
    for n in [0, 1, 2, 3, 4, 5, 10, 100, 1000]:
        c_mod = catalan_mod(n, MOD)
        print(f"C_{n} mod {MOD} = {c_mod}")

    print("\n=== Prefix (first 10 Catalan numbers) ===")
    prefix = catalan_prefix(10)
    for i, v in enumerate(prefix):
        print(f"C_{i} = {v}")

    print("\n=== Binary Tree Count (n nodes) ===")
    for n in [0, 1, 2, 3, 4, 5, 10]:
        exact = count_binary_trees(n)
        mod = count_binary_trees(n, MOD)
        formula = "(1/(n+1))*C(2n,n)"
        print(f"n={n}: trees={exact} (mod={mod}), formula={formula}")

    print("\n=== Consistency Check ===")
    n_check = 15
    prefix_all = catalan_prefix(n_check)
    single_all = [catalan(i) for i in range(n_check + 1)]
    print(f"Prefix vs Single match: {prefix_all == single_all}")
    for i in range(n_check + 1):
        assert prefix_all[i] == single_all[i], f"Mismatch at i={i}"
    print("All checks passed!")
