import math
import random
import time


def is_prime_trial(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def miller_rabin(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    
    if n < 2**64:
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    else:
        bases = [random.randint(2, min(n - 2, 2**31 - 1)) for _ in range(10)]
    
    for a in bases:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def pollards_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    while True:
        c = random.randint(1, n - 1)
        f = lambda x: (pow(x, 2, n) + c) % n
        x, y, d = 2, 2, 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def prime_factorization(n, timed=False):
    start_time = time.time() if timed else None
    
    if n == 1:
        result = {1: 1}
        if timed:
            elapsed = (time.time() - start_time) * 1000
            return result, elapsed
        return result
    
    if n < 1:
        raise ValueError("n必须是正整数")
    
    factors = {}
    
    def _factor(n):
        if n == 1:
            return
        if miller_rabin(n):
            factors[n] = factors.get(n, 0) + 1
            return
        d = pollards_rho(n)
        _factor(d)
        _factor(n // d)
    
    _factor(n)
    
    if timed:
        elapsed = (time.time() - start_time) * 1000
        return factors, elapsed
    return factors


def format_factors(factors):
    if 1 in factors and len(factors) == 1:
        return "1不是质数也不是合数"
    terms = []
    for prime, exponent in sorted(factors.items()):
        if prime == 1:
            continue
        terms.append(f"{prime}^{exponent}")
    return "×".join(terms)


def get_prime_info(n):
    if n == 1:
        return "1不是质数也不是合数"
    if miller_rabin(n):
        return f"{n}是素数"
    return f"{n}是合数"


def factorize_with_time(n):
    factors, elapsed = prime_factorization(n, timed=True)
    result_str = format_factors(factors)
    if n == 1:
        return f"{result_str} (耗时: {elapsed:.4f} ms)"
    return f"{n}={result_str} (耗时: {elapsed:.4f} ms)"


if __name__ == "__main__":
    test_numbers = [
        1, 2, 3, 12, 17, 100, 97, 1024,
        999999937,
        1000000007,
        10**12,
        999999937 * 1000000007,
        2**60 - 1,
    ]
    
    for num in test_numbers:
        print(f"{num}: {get_prime_info(num)}")
        print(f"  {factorize_with_time(num)}")
        print()

