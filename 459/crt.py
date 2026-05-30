def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def mod_inverse(a, m):
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        return None
    return x % m


def crt(remainders, moduli):
    if len(remainders) != len(moduli):
        raise ValueError("余数列表和模数列表长度必须相同")

    n = len(remainders)
    if n == 0:
        raise ValueError("至少需要一个同余方程")

    for m in moduli:
        if m <= 0:
            raise ValueError("模数必须为正整数")

    current_a = remainders[0] % moduli[0]
    current_m = moduli[0]

    for i in range(1, n):
        a1, m1 = current_a, current_m
        a2, m2 = remainders[i] % moduli[i], moduli[i]

        g, p, _ = extended_gcd(m1, m2)
        diff = a2 - a1

        if diff % g != 0:
            raise ValueError(
                f"同余方程组无解：x ≡ {a1} mod {m1} 与 x ≡ {a2} mod {m2} 矛盾 "
                f"(gcd={g}, diff={diff}, diff % g = {diff % g})"
            )

        lcm = m1 // g * m2
        tmp = (diff // g * p) % (m2 // g)
        current_a = (a1 + tmp * m1) % lcm
        current_m = lcm

    return current_a, current_m


def batch_crt(systems):
    results = []
    for remainders, moduli in systems:
        try:
            x, M = crt(remainders, moduli)
            results.append((x, M, None))
        except ValueError as e:
            results.append((None, None, str(e)))
    return results


def factorize(n):
    factors = []
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                n //= d
                count += 1
            factors.append((d, count))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors


def get_prime_powers(modulus):
    factors = factorize(modulus)
    prime_powers = []
    for p, e in factors:
        prime_powers.append(p ** e)
    return prime_powers


def comb_mod_prime_power(n, k, p, e):
    if k < 0 or k > n:
        return 0
    pe = p ** e

    def legendre(n, p):
        count = 0
        while n > 0:
            n //= p
            count += n
        return count

    def factorial_mod(n, p, pe):
        if n == 0:
            return 1
        res = 1
        for i in range(1, pe + 1):
            if i % p != 0:
                res = (res * i) % pe
        res = pow(res, n // pe, pe)
        for i in range(1, n % pe + 1):
            if i % p != 0:
                res = (res * i) % pe
        return (res * factorial_mod(n // p, p, pe)) % pe

    exp_p = legendre(n, p) - legendre(k, p) - legendre(n - k, p)
    if exp_p >= e:
        return 0

    numerator = factorial_mod(n, p, pe)
    denominator = (factorial_mod(k, p, pe) * factorial_mod(n - k, p, pe)) % pe
    inv_denominator = mod_inverse(denominator, pe)

    if inv_denominator is None:
        raise ValueError(f"无法求 {denominator} 在模 {pe} 下的逆元")

    result = (numerator * inv_denominator) % pe
    result = (result * pow(p, exp_p, pe)) % pe
    return result


def comb_mod_crt(n, k, modulus):
    if modulus <= 0:
        raise ValueError("模数必须为正整数")
    if k < 0 or k > n:
        return 0

    factors = factorize(modulus)
    if not factors:
        return 0

    remainders = []
    moduli = []
    for p, e in factors:
        pe = p ** e
        r = comb_mod_prime_power(n, k, p, e)
        remainders.append(r)
        moduli.append(pe)

    x, _ = crt(remainders, moduli)
    return x


if __name__ == "__main__":
    print("=" * 60)
    print("测试 1: crt() 返回 (解, 模) 元组")
    print("=" * 60)
    print("测试用例 1.1 (模数互质):")
    print("x ≡ 2 mod 3, x ≡ 3 mod 5, x ≡ 2 mod 7")
    try:
        x, M = crt([2, 3, 2], [3, 5, 7])
        print(f"解: x ≡ {x} mod {M}")
        print(f"验证: {x}%3={x%3}, {x}%5={x%5}, {x}%7={x%7}")
    except ValueError as e:
        print(f"错误: {e}")

    print("\n测试用例 1.2 (模数不互质，有解):")
    print("x ≡ 1 mod 4, x ≡ 3 mod 6")
    try:
        x, M = crt([1, 3], [4, 6])
        print(f"解: x ≡ {x} mod {M}")
        print(f"验证: {x}%4={x%4}, {x}%6={x%6}")
    except ValueError as e:
        print(f"错误: {e}")

    print("\n测试用例 1.3 (模数不互质，无解):")
    print("x ≡ 1 mod 4, x ≡ 0 mod 6")
    try:
        x, M = crt([1, 0], [4, 6])
        print(f"解: x ≡ {x} mod {M}")
    except ValueError as e:
        print(f"错误: {e}")

    print("\n" + "=" * 60)
    print("测试 2: batch_crt() 批量求解多组同余式")
    print("=" * 60)
    systems = [
        ([2, 3, 2], [3, 5, 7]),
        ([1, 3], [4, 6]),
        ([1, 0], [4, 6]),
        ([3, 7, 6], [6, 8, 9]),
        ([5, 11], [12, 18]),
    ]
    results = batch_crt(systems)
    for i, ((rems, mods), (x, M, err)) in enumerate(zip(systems, results), 1):
        if err:
            print(f"方程组 {i}: {rems} mod {mods} → 无解: {err[:50]}...")
        else:
            print(f"方程组 {i}: {rems} mod {mods} → x ≡ {x} mod {M}")

    print("\n" + "=" * 60)
    print("测试 3: comb_mod_crt() 组合数模运算")
    print("=" * 60)

    print("\n测试用例 3.1: C(10, 3) mod 100 = 120 mod 100 = 20")
    result = comb_mod_crt(10, 3, 100)
    print(f"结果: {result} {'✓' if result == 20 else '✗'}")

    print("\n测试用例 3.2: C(20, 5) mod 231 = 15504 mod 231")
    expected = 15504 % 231
    result = comb_mod_crt(20, 5, 231)
    print(f"结果: {result}, 期望: {expected} {'✓' if result == expected else '✗'}")

    print("\n测试用例 3.3: C(100, 10) mod 1000 (模数含质数幂 8×125)")
    import math
    expected = math.comb(100, 10) % 1000
    result = comb_mod_crt(100, 10, 1000)
    print(f"结果: {result}, 期望: {expected} {'✓' if result == expected else '✗'}")

    print("\n测试用例 3.4: C(5, 2) mod 7 = 10 mod 7 = 3")
    result = comb_mod_crt(5, 2, 7)
    print(f"结果: {result} {'✓' if result == 3 else '✗'}")

    print("\n测试用例 3.5: C(10, 5) mod 12 (模数 4×3)")
    expected = 252 % 12
    result = comb_mod_crt(10, 5, 12)
    print(f"结果: {result}, 期望: {expected} {'✓' if result == expected else '✗'}")

    print("\n测试用例 3.6: C(0, 0) mod 100 = 1")
    result = comb_mod_crt(0, 0, 100)
    print(f"结果: {result} {'✓' if result == 1 else '✗'}")

    print("\n测试用例 3.7: C(5, 10) mod 100 = 0 (k > n)")
    result = comb_mod_crt(5, 10, 100)
    print(f"结果: {result} {'✓' if result == 0 else '✗'}")

    print("\n" + "=" * 60)
    print("测试 4: 辅助函数测试")
    print("=" * 60)
    print(f"factorize(100) = {factorize(100)}")
    print(f"factorize(231) = {factorize(231)}")
    print(f"get_prime_powers(100) = {get_prime_powers(100)}")
    print(f"get_prime_powers(231) = {get_prime_powers(231)}")
