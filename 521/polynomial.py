def poly_add(a, b):
    n = max(len(a), len(b))
    result = [0] * n
    for i in range(len(a)):
        result[i] += a[i]
    for i in range(len(b)):
        result[i] += b[i]
    return _strip(result)


def poly_sub(a, b):
    n = max(len(a), len(b))
    result = [0] * n
    for i in range(len(a)):
        result[i] += a[i]
    for i in range(len(b)):
        result[i] -= b[i]
    return _strip(result)


def poly_mul(a, b):
    if not a or not b:
        return [0]
    n = len(a) + len(b) - 1
    result = [0] * n
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] += a[i] * b[j]
    return _strip(result)


def poly_div(a, b):
    b_stripped = _strip(b)
    if len(b_stripped) == 1 and b_stripped[0] == 0:
        raise ZeroDivisionError("Division by zero polynomial")

    a = list(a)
    b_degree = len(b_stripped) - 1
    b_lead = b_stripped[b_degree]

    quotient = [0] * len(a)
    remainder = list(a)

    for i in range(len(a) - 1, b_degree - 1, -1):
        if remainder[i] == 0:
            continue
        coeff = remainder[i] / b_lead
        quotient[i - b_degree] = coeff
        for j in range(b_degree + 1):
            remainder[i - b_degree + j] -= coeff * b_stripped[j]

    q = _strip(quotient)
    r = _strip(remainder)

    lhs = poly_add(poly_mul(q, b_stripped), r)
    rhs = _strip(a)
    if not _poly_approx_equal(lhs, rhs):
        raise ArithmeticError(
            f"Division verification failed: q*b + r = {lhs}, expected {rhs}"
        )

    return q, r


def poly_eval(p, x):
    p = _strip(p)
    result = 0.0
    for i in range(len(p) - 1, -1, -1):
        result = result * x + p[i]
    return result


def poly_deriv(p):
    p = _strip(p)
    if len(p) == 1:
        return [0]
    result = [0] * (len(p) - 1)
    for i in range(1, len(p)):
        result[i - 1] = p[i] * i
    return _strip(result)


def poly_gcd(a, b):
    a = _strip(a)
    b = _strip(b)
    if len(b) == 1 and b[0] == 0:
        return _make_monic(a)
    while not (len(b) == 1 and abs(b[0]) < 1e-10):
        _, r = poly_div(a, b)
        r = _strip_near_zero(r)
        a = b
        b = r
    return _make_monic(a)


def poly_antideriv(p, c=0):
    p = _strip(p)
    result = [c]
    for i in range(len(p)):
        result.append(p[i] / (i + 1))
    return _strip(result)


def poly_integrate(p, a, b):
    P = poly_antideriv(p)
    return poly_eval(P, b) - poly_eval(P, a)


def poly_integrate_numerical(p, a, b, n=1000):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = poly_eval(p, a) + poly_eval(p, b)
    for i in range(1, n, 2):
        s += 4 * poly_eval(p, a + i * h)
    for i in range(2, n, 2):
        s += 2 * poly_eval(p, a + i * h)
    return s * h / 3


def _strip(p):
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p if p else [0]


def _strip_near_zero(p, tol=1e-10):
    p = list(p)
    while len(p) > 1 and abs(p[-1]) < tol:
        p.pop()
    return p if p else [0]


def _make_monic(p):
    p = _strip(p)
    if len(p) == 1 and p[0] == 0:
        return [0]
    lead = p[-1]
    return [c / lead for c in p]


def _poly_approx_equal(a, b, tol=1e-9):
    a = _strip(a)
    b = _strip(b)
    if len(a) != len(b):
        return False
    return all(abs(ai - bi) <= tol for ai, bi in zip(a, b))


if __name__ == "__main__":
    a = [1, 2, 3]
    b = [4, 5]

    print(f"a = {a}  => 1 + 2x + 3x²")
    print(f"b = {b}  => 4 + 5x")
    print(f"a + b = {poly_add(a, b)}  => 5 + 7x + 3x²")
    print(f"a - b = {poly_sub(a, b)}  => -3 - 3x + 3x²")
    print(f"a * b = {poly_mul(a, b)}  => 4 + 13x + 22x² + 15x³")

    q, r = poly_div(a, b)
    print(f"a / b => quotient = {q}, remainder = {r}")
    print(f"  verify: q*b + r = {poly_add(poly_mul(q, b), r)}")

    print()
    c = [1, 0, 0, 1]
    d = [1, -1, 1]
    print(f"c = {c}  => 1 + x³")
    print(f"d = {d}  => 1 - x + x²")
    print(f"c + d = {poly_add(c, d)}")
    print(f"c - d = {poly_sub(c, d)}")
    print(f"c * d = {poly_mul(c, d)}")

    q2, r2 = poly_div(c, d)
    print(f"c / d => quotient = {q2}, remainder = {r2}")
    print(f"  verify: q*d + r = {poly_add(poly_mul(q2, d), r2)}")

    print()
    print("=== 多项式求值 (霍纳方法) ===")
    p = [1, 0, 3, -2]
    print(f"p = {p}  => 1 + 3x² - 2x³")
    for x in [0, 1, 2, -1]:
        val = poly_eval(p, x)
        print(f"  p({x}) = {val}")

    print()
    print("=== 导数计算 ===")
    p = [1, 2, 3, 4]
    print(f"p = {p}  => 1 + 2x + 3x² + 4x³")
    dp = poly_deriv(p)
    print(f"p' = {dp}  => 2 + 6x + 12x²")
    print(f"p'' = {poly_deriv(dp)}")
    print(f"p''' = {poly_deriv(poly_deriv(dp))}")
    print(f"p'''' = {poly_deriv(poly_deriv(poly_deriv(dp)))}")

    print()
    print("=== 零多项式测试 ===")
    try:
        poly_div([1, 2], [0])
    except ZeroDivisionError as e:
        print(f"除以 [0]: {e}")

    try:
        poly_div([1, 2], [0, 0, 0])
    except ZeroDivisionError as e:
        print(f"除以 [0, 0, 0]: {e}")

    print()
    e = [2, -4, 0, 3]
    f = [1, -2]
    print(f"e = {e}  => 2 - 4x + 3x³")
    print(f"f = {f}  => 1 - 2x")
    q3, r3 = poly_div(e, f)
    print(f"e / f => quotient = {q3}, remainder = {r3}")
    print(f"  verify: q*f + r = {poly_add(poly_mul(q3, f), r3)}")

    print()
    print("=== 多项式GCD (最大公因式) ===")
    g1 = [1, -1]
    g2 = [1, -2, 1]
    print(f"g1 = {g1}  => 1 - x")
    print(f"g2 = {g2}  => 1 - 2x + x² = (1-x)²")
    gcd_result = poly_gcd(g1, g2)
    print(f"gcd(g1, g2) = {gcd_result}  => (1-x)")

    h1 = [2, 2, -4, -4, 2, 2]
    h2 = [6, 10, -4, -10, 4]
    print(f"h1 = {h1}")
    print(f"h2 = {h2}")
    gcd_h = poly_gcd(h1, h2)
    print(f"gcd(h1, h2) = {gcd_h}")

    k1 = [1, 0, -1]
    k2 = [1, -2, 1]
    print(f"k1 = {k1}  => 1 - x² = (1-x)(1+x)")
    print(f"k2 = {k2}  => 1 - 2x + x² = (1-x)²")
    gcd_k = poly_gcd(k1, k2)
    print(f"gcd(k1, k2) = {gcd_k}  => (1-x)")

    print()
    print("=== 定积分 (解析法 vs 数值法) ===")
    p_int = [3, 0, 2, 0, 1]
    print(f"p = {p_int}  => 3 + 2x² + x⁴")
    exact = poly_integrate(p_int, 0, 1)
    numerical = poly_integrate_numerical(p_int, 0, 1)
    print(f"∫₀¹ p(x)dx  解析 = {exact:.10f}")
    print(f"∫₀¹ p(x)dx  数值 = {numerical:.10f}")
    print(f"  差值 = {abs(exact - numerical):.2e}")

    p_int2 = [1, 0, 0, 0, 0, 1]
    print(f"p = {p_int2}  => 1 + x⁵")
    exact2 = poly_integrate(p_int2, -1, 2)
    numerical2 = poly_integrate_numerical(p_int2, -1, 2)
    print(f"∫₋₁² p(x)dx  解析 = {exact2:.10f}")
    print(f"∫₋₁² p(x)dx  数值 = {numerical2:.10f}")
    print(f"  差值 = {abs(exact2 - numerical2):.2e}")

    print()
    print("=== sympy 对比验证 ===")
    try:
        from sympy import Symbol, gcd as sym_gcd, Poly, integrate as sym_integrate, Rational
        x = Symbol('x')

        g1_sym = 1 - x
        g2_sym = 1 - 2*x + x**2
        print(f"sympy gcd(g1, g2) = {sym_gcd(g1_sym, g2_sym)}")
        print(f"我们的 gcd(g1, g2) = {gcd_result}")

        k1_sym = 1 - x**2
        k2_sym = 1 - 2*x + x**2
        print(f"sympy gcd(k1, k2) = {sym_gcd(k1_sym, k2_sym)}")
        print(f"我们的 gcd(k1, k2) = {gcd_k}")

        p_sym = 3 + 2*x**2 + x**4
        sym_val = float(sym_integrate(p_sym, (x, 0, 1)))
        print(f"sympy ∫₀¹ p(x)dx = {sym_val:.10f}")
        print(f"我们的 ∫₀¹ p(x)dx = {exact:.10f}")
    except ImportError:
        print("sympy 未安装，跳过对比验证")
