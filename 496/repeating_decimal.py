from math import gcd


DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _prime_factors(n):
    factors = {}
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 2
    if n > 2:
        factors[n] = 1
    return factors


def _euler_phi(n):
    if n == 0:
        return 0
    result = n
    factors = _prime_factors(n)
    for p in factors:
        result -= result // p
    return result


def _get_divisors(n):
    divisors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)


def _modular_order(base, m):
    if gcd(base, m) != 1:
        return None
    phi = _euler_phi(m)
    divisors = _get_divisors(phi)
    for d in divisors:
        if pow(base, d, m) == 1:
            return d
    return phi


def _count_factor(q, factor):
    count = 0
    while q % factor == 0:
        q //= factor
        count += 1
    return count


def _remove_base_factors(q, base):
    for prime in _prime_factors(base):
        while q % prime == 0:
            q //= prime
    return q


def _is_finite_expansion(q, base):
    return _remove_base_factors(q, base) == 1


def _digit_to_char(d):
    if d < len(DIGITS):
        return DIGITS[d]
    return f"[{d}]"


def _format_digits(digits):
    return "".join(_digit_to_char(d) for d in digits)


def detect_repeating_decimal(p, q, base=10):
    if q == 0:
        raise ValueError("Denominator cannot be zero")
    if base < 2:
        raise ValueError("Base must be >= 2")

    g = gcd(p, q)
    p //= g
    q //= g

    sign = ""
    if (p < 0) ^ (q < 0):
        sign = "-"
    if p < 0:
        p = -p
    if q < 0:
        q = -q

    integer_part = p // q
    remainder = p % q

    if remainder == 0:
        return {
            "decimal_str": sign + _digit_to_char(integer_part),
            "cycle": None,
            "cycle_start": None,
            "cycle_length": None,
            "is_finite": True,
            "non_repeating_length": 0,
            "base": base,
            "integer_part": integer_part,
            "stats": {
                "total_digits": 0,
                "digit_counts": {},
                "theoretical_cycle_length": None,
            }
        }

    base_factors = _prime_factors(base)
    max_count = 0
    q_remaining = q
    for prime in base_factors:
        count = _count_factor(q_remaining, prime)
        max_count = max(max_count, count)
    non_repeating_len = max_count
    q_coprime = _remove_base_factors(q, base)
    is_finite = q_coprime == 1

    theoretical_cycle_length = None
    if not is_finite:
        theoretical_cycle_length = _modular_order(base, q_coprime)

    remainder_positions = {}
    decimal_digits = []
    position = 0
    cycle_start_pos = None
    cycle_length = None
    cycle = None

    while remainder != 0:
        if remainder in remainder_positions:
            cycle_start_pos = remainder_positions[remainder]
            cycle_length = position - cycle_start_pos
            non_repeating_part = decimal_digits[:cycle_start_pos]
            repeating_part = decimal_digits[cycle_start_pos:]
            cycle = _format_digits(repeating_part)

            non_repeating_str = _format_digits(non_repeating_part)
            decimal_str = f"{sign}{_digit_to_char(integer_part)}.{non_repeating_str}({cycle})"
            break

        remainder_positions[remainder] = position
        remainder *= base
        digit = remainder // q
        decimal_digits.append(digit)
        remainder = remainder % q
        position += 1

        if is_finite and position > non_repeating_len + 1:
            decimal_str = f"{sign}{_digit_to_char(integer_part)}.{_format_digits(decimal_digits)}"
            break
    else:
        decimal_str = f"{sign}{_digit_to_char(integer_part)}.{_format_digits(decimal_digits)}"

    digit_counts = {}
    for d in decimal_digits:
        digit_counts[d] = digit_counts.get(d, 0) + 1

    return {
        "decimal_str": decimal_str,
        "cycle": cycle,
        "cycle_start": cycle_start_pos,
        "cycle_length": cycle_length,
        "is_finite": is_finite,
        "non_repeating_length": non_repeating_len,
        "base": base,
        "integer_part": integer_part,
        "stats": {
            "total_digits": len(decimal_digits),
            "digit_counts": digit_counts,
            "theoretical_cycle_length": theoretical_cycle_length,
            "coprime_denominator": q_coprime,
            "euler_phi": _euler_phi(q_coprime) if q_coprime > 1 else None,
        }
    }


def format_result(p, q, base=10):
    return detect_repeating_decimal(p, q, base)


def _print_result(info, p, q):
    base = info["base"]
    print(f"\n{p}/{q} (base {base}):")
    print(f"  小数表示: {info['decimal_str']}")
    if info["cycle"] is not None:
        print(f"  循环节: {info['cycle']}")
        print(f"  循环节起始位置: 小数点后第 {info['cycle_start']} 位")
        print(f"  循环节长度: {info['cycle_length']}")
        print(f"  理论循环节长度: {info['stats']['theoretical_cycle_length']}")
        print(f"  非循环部分长度: {info['non_repeating_length']}")
        print(f"  与基数互质的分母部分: {info['stats']['coprime_denominator']}")
        if info['stats']['euler_phi'] is not None:
            print(f"  欧拉函数 φ({info['stats']['coprime_denominator']}) = {info['stats']['euler_phi']}")
    else:
        print(f"  类型: 有限小数 ({info['non_repeating_length']} 位)")
    print(f"  总小数位数: {info['stats']['total_digits']}")
    if info['stats']['digit_counts']:
        print(f"  数字统计: {info['stats']['digit_counts']}")


if __name__ == "__main__":
    print("=" * 60)
    print("十进制测试 (base 10)")
    print("=" * 60)
    test_cases = [
        (1, 7),
        (1, 3),
        (1, 6),
        (1, 8),
        (1, 17),
        (1, 9),
    ]
    for p, q in test_cases:
        info = format_result(p, q, base=10)
        _print_result(info, p, q)

    print("\n" + "=" * 60)
    print("二进制测试 (base 2)")
    print("=" * 60)
    test_cases_bin = [
        (1, 3),
        (1, 5),
        (1, 7),
        (1, 9),
        (1, 11),
    ]
    for p, q in test_cases_bin:
        info = format_result(p, q, base=2)
        _print_result(info, p, q)

    print("\n" + "=" * 60)
    print("八进制测试 (base 8)")
    print("=" * 60)
    test_cases_oct = [
        (1, 3),
        (1, 5),
        (1, 7),
        (1, 9),
    ]
    for p, q in test_cases_oct:
        info = format_result(p, q, base=8)
        _print_result(info, p, q)

    print("\n" + "=" * 60)
    print("十六进制测试 (base 16)")
    print("=" * 60)
    test_cases_hex = [
        (1, 3),
        (1, 5),
        (1, 7),
        (1, 9),
        (1, 11),
    ]
    for p, q in test_cases_hex:
        info = format_result(p, q, base=16)
        _print_result(info, p, q)
