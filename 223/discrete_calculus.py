import numpy as np


_VALID_BOUNDARY = {'truncate', 'fill', 'extrapolate'}


def _validate_boundary(boundary):
    if boundary not in _VALID_BOUNDARY:
        raise ValueError(
            f"boundary 必须是 {_VALID_BOUNDARY} 之一，当前为 {boundary}"
        )


def _get_dx(x, h, n):
    if x is not None:
        x = np.asarray(x, dtype=np.float64)
        if len(x) != n:
            raise ValueError(f"x 与 y 长度不匹配: {len(x)} vs {n}")
        dx = x[1:] - x[:-1]
        if np.any(dx <= 0):
            raise ValueError("x 必须严格递增")
        return dx, x
    else:
        return np.full(n - 1, h, dtype=np.float64), None


def forward_difference(y, x=None, h=1.0, boundary='truncate', fill_value=np.nan):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2:
        raise ValueError("序列长度至少为2")
    _validate_boundary(boundary)

    dx, x_arr = _get_dx(x, h, n)
    diff_core = (y[1:] - y[:-1]) / dx

    if boundary == 'truncate':
        return diff_core
    elif boundary == 'fill':
        diff = np.full(n, fill_value, dtype=np.float64)
        diff[:-1] = diff_core
        return diff
    else:
        diff = np.zeros(n, dtype=np.float64)
        diff[:-1] = diff_core
        if n >= 3:
            diff[-1] = 2 * diff_core[-1] - diff_core[-2]
        else:
            diff[-1] = diff_core[-1]
        return diff


def backward_difference(y, x=None, h=1.0, boundary='truncate', fill_value=np.nan):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2:
        raise ValueError("序列长度至少为2")
    _validate_boundary(boundary)

    dx, x_arr = _get_dx(x, h, n)
    diff_core = (y[1:] - y[:-1]) / dx

    if boundary == 'truncate':
        return diff_core
    elif boundary == 'fill':
        diff = np.full(n, fill_value, dtype=np.float64)
        diff[1:] = diff_core
        return diff
    else:
        diff = np.zeros(n, dtype=np.float64)
        diff[1:] = diff_core
        if n >= 3:
            diff[0] = 2 * diff_core[0] - diff_core[1]
        else:
            diff[0] = diff_core[0]
        return diff


def central_difference(y, x=None, h=1.0, boundary='truncate', fill_value=np.nan):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3:
        raise ValueError("序列长度至少为3")
    _validate_boundary(boundary)

    dx, x_arr = _get_dx(x, h, n)

    if x_arr is not None:
        h_left = dx[:-1]
        h_right = dx[1:]
        diff_core = (
            h_left ** 2 * y[2:] + (h_right ** 2 - h_left ** 2) * y[1:-1] - h_right ** 2 * y[:-2]
        ) / (h_left * h_right * (h_left + h_right))
    else:
        diff_core = (y[2:] - y[:-2]) / (2 * h)

    if boundary == 'truncate':
        return diff_core
    elif boundary == 'fill':
        diff = np.full(n, fill_value, dtype=np.float64)
        diff[1:-1] = diff_core
        return diff
    else:
        diff = np.zeros(n, dtype=np.float64)
        diff[1:-1] = diff_core
        diff[0] = 2 * diff_core[0] - diff_core[1]
        diff[-1] = 2 * diff_core[-1] - diff_core[-2]
        return diff


def cumulative_trapezoidal_integral(y, x=None, h=1.0):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2:
        raise ValueError("序列长度至少为2")

    dx, _ = _get_dx(x, h, n)
    integral = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        integral[i] = integral[i - 1] + (y[i] + y[i - 1]) * dx[i - 1] / 2.0
    return integral


def cumulative_simpson_integral(y, x=None, h=1.0):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3:
        raise ValueError("辛普森积分序列长度至少为3")

    dx, x_arr = _get_dx(x, h, n)
    integral = np.zeros(n, dtype=np.float64)

    if x_arr is None:
        integral[1] = (y[0] + y[1]) * h / 2.0
        for i in range(2, n):
            if i % 2 == 0:
                simpson_part = h / 3.0 * (y[i - 2] + 4 * y[i - 1] + y[i])
                integral[i] = integral[i - 2] + simpson_part
            else:
                integral[i] = integral[i - 1] + (y[i] + y[i - 1]) * h / 2.0
    else:
        for i in range(1, n):
            integral[i] = integral[i - 1] + (y[i] + y[i - 1]) * dx[i - 1] / 2.0

    return integral
