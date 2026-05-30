def _next_power_of_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p


_BUTTERFLY = {
    "xor": {
        "forward": lambda x, y: (x + y, x - y),
        "inverse": lambda x, y: (x + y, x - y),
    },
    "or": {
        "forward": lambda x, y: (x, x + y),
        "inverse": lambda x, y: (x, y - x),
    },
    "and": {
        "forward": lambda x, y: (x + y, y),
        "inverse": lambda x, y: (x - y, y),
    },
}


def fwht(a, inverse=False, original_len=None, transform="xor"):
    if transform not in _BUTTERFLY:
        raise ValueError(f"Unknown transform '{transform}', choose from {list(_BUTTERFLY)}")

    a = list(a)
    orig = len(a)
    ops = _BUTTERFLY[transform]["inverse"] if inverse else _BUTTERFLY[transform]["forward"]

    if not inverse:
        n = _next_power_of_two(orig)
        if n != orig:
            a.extend([0] * (n - orig))
        original_len = orig
    else:
        n = len(a)
        original_len = original_len if original_len is not None else n

    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                a[j], a[j + step] = ops(a[j], a[j + step])
        step <<= 1

    if inverse and transform == "xor":
        for i in range(n):
            a[i] = a[i] / n

    if inverse and original_len < n:
        a = a[:original_len]

    return a, original_len
