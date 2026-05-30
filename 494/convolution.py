from fwht import fwht


def xor_convolution(a, b):
    n = max(len(a), len(b))
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    ta, _ = fwht(a, transform="xor")
    tb, _ = fwht(b, transform="xor")
    tc = [ta[i] * tb[i] for i in range(len(ta))]
    result, _ = fwht(tc, inverse=True, original_len=n, transform="xor")
    return result


def or_convolution(a, b):
    n = max(len(a), len(b))
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    ta, _ = fwht(a, transform="or")
    tb, _ = fwht(b, transform="or")
    tc = [ta[i] * tb[i] for i in range(len(ta))]
    result, _ = fwht(tc, inverse=True, original_len=n, transform="or")
    return result


def and_convolution(a, b):
    n = max(len(a), len(b))
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    ta, _ = fwht(a, transform="and")
    tb, _ = fwht(b, transform="and")
    tc = [ta[i] * tb[i] for i in range(len(ta))]
    result, _ = fwht(tc, inverse=True, original_len=n, transform="and")
    return result


def _popcount(x):
    c = 0
    while x:
        c += 1
        x &= x - 1
    return c


def subset_convolution(f, g):
    n = max(len(f), len(g))
    m = 0
    while (1 << m) < n:
        m += 1
    n = 1 << m

    f = f + [0] * (n - len(f))
    g = g + [0] * (n - len(g))

    F = [[0] * n for _ in range(m + 1)]
    G = [[0] * n for _ in range(m + 1)]

    for s in range(n):
        pc = _popcount(s)
        F[pc][s] = f[s]
        G[pc][s] = g[s]

    for k in range(m + 1):
        F[k], _ = fwht(F[k], transform="or")
        G[k], _ = fwht(G[k], transform="or")

    H = [[0] * n for _ in range(m + 1)]
    for s in range(n):
        for i in range(m + 1):
            val = 0
            for j in range(i + 1):
                val += F[j][s] * G[i - j][s]
            H[i][s] = val

    for k in range(m + 1):
        H[k], _ = fwht(H[k], inverse=True, original_len=n, transform="or")

    result = [0] * n
    for s in range(n):
        result[s] = H[_popcount(s)][s]

    return result
