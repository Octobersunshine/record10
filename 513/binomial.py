from fractions import Fraction
import math


def _check_boundary(n, k):
    if not isinstance(n, int) or not isinstance(k, int):
        raise TypeError("n and k must be integers")
    if n < 0:
        raise ValueError("n must be non-negative")
    return k < 0 or k > n


def _is_prime(p):
    if p < 2:
        return False
    if p == 2:
        return True
    if p % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(p)) + 1, 2):
        if p % i == 0:
            return False
    return True


def binomial_coefficient_factorial(n, k):
    if _check_boundary(n, k):
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def binomial_coefficient_fraction(n, k):
    if _check_boundary(n, k):
        return Fraction(0, 1)
    if k == 0 or k == n:
        return Fraction(1, 1)
    k = min(k, n - k)
    result = Fraction(1, 1)
    for i in range(k):
        result = result * Fraction(n - i, i + 1)
    return result


def binomial_coefficient_log(n, k):
    if _check_boundary(n, k):
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    log_result = 0.0
    for i in range(k):
        log_result += math.log(n - i) - math.log(i + 1)
    return math.exp(log_result)


def binomial_coefficient_mod_prime(n, k, p):
    if not _is_prime(p):
        raise ValueError(f"{p} is not a prime number")
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    numerator = 1
    denominator = 1
    for i in range(k):
        numerator = (numerator * (n - i)) % p
        denominator = (denominator * (i + 1)) % p
    return (numerator * pow(denominator, p - 2, p)) % p


def lucas_theorem(n, k, p):
    if not _is_prime(p):
        raise ValueError(f"{p} is not a prime number")
    if k < 0 or k > n:
        return 0
    if k == 0:
        return 1
    result = 1
    while n > 0 or k > 0:
        ni = n % p
        ki = k % p
        if ki > ni:
            return 0
        result = (result * binomial_coefficient_mod_prime(ni, ki, p)) % p
        n = n // p
        k = k // p
    return result


def pascals_triangle(n, use_fraction=False):
    if n <= 0:
        return []
    one = Fraction(1, 1) if use_fraction else 1
    triangle = [[one]]
    for i in range(1, n):
        row = [one]
        for j in range(1, i):
            row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
        row.append(one)
        triangle.append(row)
    return triangle


def pascals_triangle_fraction(n):
    return pascals_triangle(n, use_fraction=True)


def get_pascal_value(row, col):
    return binomial_coefficient_factorial(row, col)


def binomial_coefficient_recursive(n, k):
    if _check_boundary(n, k):
        return 0
    if k == 0 or k == n:
        return 1
    triangle = pascals_triangle(n + 1)
    return triangle[n][k]


def binomial_expansion(n, a="a", b="b", use_latex=False):
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0:
        return "1"
    terms = []
    for k in range(n + 1):
        coeff = binomial_coefficient_factorial(n, k)
        a_exp = n - k
        b_exp = k
        term_parts = []
        if coeff != 1 or (a_exp == 0 and b_exp == 0):
            term_parts.append(str(coeff))
        if a_exp > 0:
            if use_latex:
                term_parts.append(a + ("^{" + str(a_exp) + "}" if a_exp != 1 else ""))
            else:
                term_parts.append(a + ("^" + str(a_exp) if a_exp != 1 else ""))
        if b_exp > 0:
            if use_latex:
                term_parts.append(b + ("^{" + str(b_exp) + "}" if b_exp != 1 else ""))
            else:
                term_parts.append(b + ("^" + str(b_exp) if b_exp != 1 else ""))
        terms.append("".join(term_parts))
    if use_latex:
        return "(" + a + " + " + b + ")^{" + str(n) + "} = " + " + ".join(terms)
    return "(" + a + " + " + b + ")^" + str(n) + " = " + " + ".join(terms)


if __name__ == "__main__":
    test_cases = [
        (8, 3),
        (100, 50),
        (10, -1),
        (10, 11),
        (0, 0),
        (5, 0),
        (5, 5),
    ]

    print("=" * 70)
    print("Testing Binomial Coefficients")
    print("=" * 70)

    for n, k in test_cases:
        print(f"\n--- C({n}, {k}) ---")
        try:
            print(f"  Integer:   {binomial_coefficient_factorial(n, k)}")
            print(f"  Fraction:  {binomial_coefficient_fraction(n, k)}")
            print(f"  Log approx: {binomial_coefficient_log(n, k):.6e}")
        except (ValueError, TypeError) as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("Lucas Theorem - Mod Prime")
    print("=" * 70)
    n, k, p = 1000, 500, 13
    print(f"\nC({n}, {k}) mod {p} = {lucas_theorem(n, k, p)}")
    n, k, p = 10, 3, 7
    print(f"C({n}, {k}) mod {p} = {lucas_theorem(n, k, p)} (direct: {binomial_coefficient_factorial(n, k) % p})")

    print("\n" + "=" * 70)
    print("Binomial Expansion")
    print("=" * 70)
    for n in range(0, 7):
        print(f"\n  {binomial_expansion(n)}")

    print("\n" + "=" * 70)
    print("Pascal's Triangle Direct Lookup")
    print("=" * 70)
    lookups = [(0, 0), (3, 1), (5, 2), (10, 5)]
    for row, col in lookups:
        print(f"\n  Pascal's Triangle[{row}][{col}] = {get_pascal_value(row, col)}")

    print("\n" + "=" * 70)
    print("Pascal's Triangle (first 8 rows, integer)")
    print("=" * 70)
    triangle = pascals_triangle(8)
    width = len("  ".join(map(str, triangle[-1])))
    for row in triangle:
        line = "  ".join(map(str, row))
        print(line.center(width))

    print("\n" + "=" * 70)
    print("Pascal's Triangle (first 5 rows, fraction)")
    print("=" * 70)
    triangle_frac = pascals_triangle_fraction(5)
    for row in triangle_frac:
        print(f"  {row}")
