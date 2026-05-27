import math
import random


_MR_WITNESSES_64 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_random = random.SystemRandom()


def _miller_rabin(n, a):
    if a % n == 0:
        return True
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return True
    return False


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    if n < 100:
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if n == p:
                return True
            if n % p == 0:
                return False
    for a in _MR_WITNESSES_64:
        if not _miller_rabin(n, a):
            return False
    return True


def _pollards_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3

    while True:
        c = _random.randrange(1, n - 1)
        f = lambda x: (pow(x, 2, n) + c) % n
        x, y, d = 2, 2, 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def factorize(n):
    factors = {}
    def _factorize(n):
        if n == 1:
            return
        if is_prime(n):
            factors[n] = factors.get(n, 0) + 1
            return
        d = _pollards_rho(n)
        _factorize(d)
        _factorize(n // d)
    if n < 2:
        return factors
    _factorize(n)
    return dict(sorted(factors.items()))


def twin_primes(limit):
    primes = sieve_of_eratosthenes(limit)
    twins = []
    for i in range(len(primes) - 1):
        if primes[i + 1] - primes[i] == 2:
            twins.append((primes[i], primes[i + 1]))
    return twins


def twin_primes_in_range(n, m):
    primes = primes_in_range(n, m + 2)
    twins = []
    for i in range(len(primes) - 1):
        if primes[i + 1] - primes[i] == 2 and primes[i] >= n:
            twins.append((primes[i], primes[i + 1]))
    return twins


def goldbach(even_n):
    if even_n < 4 or even_n % 2 != 0:
        raise ValueError("Goldbach conjecture applies to even numbers >= 4")
    if even_n == 4:
        return (2, 2)
    if is_prime(even_n - 3):
        return (3, even_n - 3)
    primes = sieve_of_eratosthenes(even_n // 2 + 1)
    for p in primes:
        if p > 2 and is_prime(even_n - p):
            return (p, even_n - p)
    return None


def sieve_of_eratosthenes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i : limit+1 : i] = [False] * len(sieve[i*i : limit+1 : i])
    return [i for i, is_p in enumerate(sieve) if is_p]


def primes_in_range(n, m):
    if n > m:
        return []
    if m < 2:
        return []
    if n < 2:
        n = 2
    if m - n < 1000:
        return [x for x in range(n, m + 1) if is_prime(x)]
    all_primes = sieve_of_eratosthenes(m)
    return [p for p in all_primes if p >= n]


def primes_up_to(n):
    return sieve_of_eratosthenes(n)


if __name__ == "__main__":
    import time

    print("Testing is_prime (small numbers):")
    test_numbers = [1, 2, 3, 4, 17, 25, 97, 100]
    for num in test_numbers:
        print(f"  {num} is prime: {is_prime(num)}")

    print("\nTesting is_prime (large numbers > 10^12):")
    large_tests = [
        (999999999989, True),
        (1000000000039, True),
        (1000000000041, False),
        (999999999999999989, True),
        (999999999999999990, False),
    ]
    for num, expected in large_tests:
        start = time.perf_counter()
        result = is_prime(num)
        elapsed = time.perf_counter() - start
        status = "OK" if result == expected else "FAIL"
        print(f"  {num} -> {result} [{status}] {elapsed*1e6:.1f} us")

    print("\nPrimes up to 30:", primes_up_to(30))
    print("Primes in range [10, 50]:", primes_in_range(10, 50))

    print("\n=== Twin Primes ===")
    print("Twin primes up to 100:", twin_primes(100))
    print("Twin primes in [100, 200]:", twin_primes_in_range(100, 200))

    print("\n=== Goldbach Conjecture ===")
    for even in [4, 10, 28, 100, 1000000]:
        p, q = goldbach(even)
        print(f"  {even} = {p} + {q}  (verify: {p}+{q}={p+q}, both prime: {is_prime(p) and is_prime(q)})")

    print("\n=== Factorization (Pollard's Rho) ===")
    factor_tests = [12, 1001, 123456789, 999999937, 999999999989 * 1000000000039]
    for n in factor_tests:
        start = time.perf_counter()
        factors = factorize(n)
        elapsed = time.perf_counter() - start
        product = 1
        for p, e in factors.items():
            product *= p ** e
        status = "OK" if product == n else "FAIL"
        print(f"  {n} = {factors} [{status}] {elapsed*1e3:.2f} ms")
