def trailing_zeros(n):
    count = 0
    divisor = 5
    while divisor <= n:
        count += n // divisor
        divisor *= 5
    return count


def prime_factors(b):
    factors = {}
    divisor = 2
    while divisor * divisor <= b:
        while b % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            b //= divisor
        divisor += 1
    if b > 1:
        factors[b] = factors.get(b, 0) + 1
    return factors


def count_factor_in_factorial(n, p):
    count = 0
    divisor = p
    while divisor <= n:
        count += n // divisor
        divisor *= p
    return count


def trailing_zeros_base(n, b):
    if b < 2 or b > 1000:
        raise ValueError("b must be between 2 and 1000")
    if n < 0:
        raise ValueError("n must be non-negative")

    factors = prime_factors(b)
    min_zeros = float("inf")

    for p, e in factors.items():
        count_p = count_factor_in_factorial(n, p)
        zeros_for_p = count_p // e
        if zeros_for_p < min_zeros:
            min_zeros = zeros_for_p

    return min_zeros


if __name__ == "__main__":
    print("=== 十进制 (b=10) ===")
    test_cases = [5, 10, 25, 50, 100, 125, 1000, 1000000]
    for n in test_cases:
        print(f"{n}! 末尾零的个数: {trailing_zeros(n)} (专用函数) / {trailing_zeros_base(n, 10)} (通用函数)")

    print("\n=== 任意进制测试 ===")
    test_pairs = [
        (10, 2), (10, 8), (10, 16), (10, 12),
        (20, 6), (20, 7), (20, 10), (20, 25),
        (100, 2), (100, 10), (100, 12), (100, 60),
        (25, 1000),
    ]
    for n, b in test_pairs:
        factors = prime_factors(b)
        result = trailing_zeros_base(n, b)
        factor_str = " * ".join([f"{p}^{e}" for p, e in factors.items()])
        print(f"{n}! 以 {b} ({factor_str}) 为基数的末尾零个数: {result}")
