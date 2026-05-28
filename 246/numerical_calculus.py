import numpy as np
from typing import List, Tuple, Optional, Union, Callable


def _sort_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sort_indices = np.argsort(x)
    return x[sort_indices], y[sort_indices]


def _validate_inputs(x, y, min_len=2):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < min_len:
        raise ValueError(f"At least {min_len} points are required")
    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    return x_arr, y_arr


def trapezoidal_integration(x: List[float], y: List[float]) -> float:
    x_arr, y_arr = _validate_inputs(x, y, min_len=2)
    x_arr, y_arr = _sort_data(x_arr, y_arr)

    dx = np.diff(x_arr)
    if np.any(dx <= 0):
        raise ValueError("x values must be strictly increasing after sorting (duplicate x found)")

    h_left = np.zeros(len(x_arr))
    h_right = np.zeros(len(x_arr))
    h_left[1:] = dx
    h_right[:-1] = dx
    total_h = h_left + h_right
    total_h[total_h == 0] = 1.0

    weights = np.zeros(len(x_arr))
    weights[0] = dx[0] / 2.0
    weights[-1] = dx[-1] / 2.0
    weights[1:-1] = (dx[:-1] + dx[1:]) / 2.0

    integral = np.sum(weights * y_arr)

    return float(integral)


def log_trapezoidal_integration(x: List[float], y: List[float]) -> float:
    x_arr, y_arr = _validate_inputs(x, y, min_len=2)
    x_arr, y_arr = _sort_data(x_arr, y_arr)

    if np.any(x_arr <= 0):
        raise ValueError("x values must be positive for logarithmic integration")

    u_arr = np.log(x_arr)
    du = np.diff(u_arr)

    if np.any(du <= 0):
        raise ValueError("log(x) values must be strictly increasing (duplicate x found)")

    weights = np.zeros(len(u_arr))
    weights[0] = du[0] / 2.0
    weights[-1] = du[-1] / 2.0
    weights[1:-1] = (du[:-1] + du[1:]) / 2.0

    integral = np.sum(weights * y_arr)

    return float(integral)


def _trapezoidal_single(a: float, b: float, fa: float, fb: float) -> float:
    return (b - a) * (fa + fb) / 2.0


def simpsons_integration(x: List[float], y: List[float]) -> float:
    x_arr, y_arr = _validate_inputs(x, y, min_len=3)
    x_arr, y_arr = _sort_data(x_arr, y_arr)

    dx = np.diff(x_arr)
    if np.any(dx <= 0):
        raise ValueError("x values must be strictly increasing after sorting (duplicate x found)")

    n = len(x_arr)
    integral = 0.0

    if n == 3:
        h = dx[0]
        if abs(dx[0] - dx[1]) < 1e-10:
            integral = (h / 3.0) * (y_arr[0] + 4 * y_arr[1] + y_arr[2])
        else:
            integral = _simpsons_nonuniform_single(x_arr[0], x_arr[1], x_arr[2],
                                                   y_arr[0], y_arr[1], y_arr[2])
        return float(integral)

    if n % 2 == 1:
        for i in range(0, n - 2, 2):
            if abs(dx[i] - dx[i+1]) < 1e-10:
                h = dx[i]
                integral += (h / 3.0) * (y_arr[i] + 4 * y_arr[i+1] + y_arr[i+2])
            else:
                integral += _simpsons_nonuniform_single(x_arr[i], x_arr[i+1], x_arr[i+2],
                                                        y_arr[i], y_arr[i+1], y_arr[i+2])
    else:
        for i in range(0, n - 3, 2):
            if abs(dx[i] - dx[i+1]) < 1e-10:
                h = dx[i]
                integral += (h / 3.0) * (y_arr[i] + 4 * y_arr[i+1] + y_arr[i+2])
            else:
                integral += _simpsons_nonuniform_single(x_arr[i], x_arr[i+1], x_arr[i+2],
                                                        y_arr[i], y_arr[i+1], y_arr[i+2])
        h_last = dx[-1]
        integral += (h_last / 2.0) * (y_arr[-2] + y_arr[-1])

    return float(integral)


def _simpsons_nonuniform_single(x0: float, x1: float, x2: float,
                                y0: float, y1: float, y2: float) -> float:
    h0 = x1 - x0
    h1 = x2 - x1

    if h0 <= 0 or h1 <= 0:
        raise ValueError("x values must be strictly increasing")

    coeff = (h0 + h1) / 6.0
    term0 = (2.0 - h1 / h0) * y0
    term1 = ((h0 + h1) ** 2 / (h0 * h1)) * y1
    term2 = (2.0 - h0 / h1) * y2

    integral = coeff * (term0 + term1 + term2)

    return float(integral)


def adaptive_simpsons_integration(x: List[float], y: List[float],
                                  tol: float = 1e-8,
                                  max_depth: int = 50,
                                  interp_kind: str = 'cubic') -> float:
    x_arr, y_arr = _validate_inputs(x, y, min_len=2)
    x_arr, y_arr = _sort_data(x_arr, y_arr)

    dx = np.diff(x_arr)
    if np.any(dx <= 0):
        raise ValueError("x values must be strictly increasing after sorting (duplicate x found)")

    if interp_kind == 'cubic' and len(x_arr) >= 4:
        from scipy.interpolate import CubicSpline
        interp_func = CubicSpline(x_arr, y_arr)
        func = lambda t: float(interp_func(t))
    else:
        func = lambda t: float(np.interp(t, x_arr, y_arr))

    total = 0.0
    for i in range(len(x_arr) - 1):
        a, b = x_arr[i], x_arr[i + 1]
        fa, fb = y_arr[i], y_arr[i + 1]
        refined = _adaptive_simpsons_refine(a, b, fa, fb, func, tol, max_depth, 0)
        total += refined

    return float(total)


def _adaptive_simpsons_refine(a: float, b: float, fa: float, fb: float,
                              func: Callable[[float], float],
                              tol: float, max_depth: int, depth: int) -> float:
    if depth >= max_depth:
        return (b - a) * (fa + fb) / 2.0

    c = (a + b) / 2.0
    d = (a + c) / 2.0
    e = (c + b) / 2.0
    fc = func(c)
    fd = func(d)
    fe = func(e)

    h = b - a
    S = (h / 6.0) * (fa + 4 * fc + fb)
    S2 = (h / 12.0) * (fa + 4 * fd + 2 * fc + 4 * fe + fb)

    error = abs(S2 - S) / 15.0

    if error < tol:
        return S2 + (S2 - S) / 15.0

    return (_adaptive_simpsons_refine(a, c, fa, fc, func, tol / 2.0, max_depth, depth + 1) +
            _adaptive_simpsons_refine(c, b, fc, fb, func, tol / 2.0, max_depth, depth + 1))


def _adaptive_refine(a: float, b: float, fa: float, fb: float,
                     whole: float, func: Callable[[float], float],
                     tol: float, max_depth: int, depth: int) -> float:
    if depth >= max_depth:
        return whole

    mid = (a + b) / 2.0
    fm = func(mid)

    left = _trapezoidal_single(a, mid, fa, fm)
    right = _trapezoidal_single(mid, b, fm, fb)

    refined = left + right
    error = abs(refined - whole) / 3.0

    if error < tol:
        return refined + (refined - whole) / 15.0

    return (_adaptive_refine(a, mid, fa, fm, left, func, tol / 2.0, max_depth, depth + 1) +
            _adaptive_refine(mid, b, fm, fb, right, func, tol / 2.0, max_depth, depth + 1))


def adaptive_trapezoidal_integration(x: List[float], y: List[float],
                                     tol: float = 1e-8,
                                     max_depth: int = 50,
                                     interp_kind: str = 'cubic') -> float:
    x_arr, y_arr = _validate_inputs(x, y, min_len=2)
    x_arr, y_arr = _sort_data(x_arr, y_arr)

    dx = np.diff(x_arr)
    if np.any(dx <= 0):
        raise ValueError("x values must be strictly increasing after sorting (duplicate x found)")

    if interp_kind == 'cubic' and len(x_arr) >= 4:
        from scipy.interpolate import CubicSpline
        interp_func = CubicSpline(x_arr, y_arr)
        func = lambda t: float(interp_func(t))
    else:
        func = lambda t: float(np.interp(t, x_arr, y_arr))

    total = 0.0
    for i in range(len(x_arr) - 1):
        a, b = x_arr[i], x_arr[i + 1]
        fa, fb = y_arr[i], y_arr[i + 1]
        whole = _trapezoidal_single(a, b, fa, fb)
        refined = _adaptive_refine(a, b, fa, fb, whole, func, tol, max_depth, 0)
        total += refined

    return float(total)


def log_adaptive_trapezoidal_integration(x: List[float], y: List[float],
                                         tol: float = 1e-8,
                                         max_depth: int = 50,
                                         interp_kind: str = 'cubic') -> float:
    x_arr, y_arr = _validate_inputs(x, y, min_len=2)
    x_arr, y_arr = _sort_data(x_arr, y_arr)

    if np.any(x_arr <= 0):
        raise ValueError("x values must be positive for logarithmic integration")

    u_arr = np.log(x_arr)

    du = np.diff(u_arr)
    if np.any(du <= 0):
        raise ValueError("log(x) values must be strictly increasing (duplicate x found)")

    if interp_kind == 'cubic' and len(u_arr) >= 4:
        from scipy.interpolate import CubicSpline
        interp_func = CubicSpline(u_arr, y_arr)
        func = lambda t: float(interp_func(t))
    else:
        func = lambda t: float(np.interp(t, u_arr, y_arr))

    total = 0.0
    for i in range(len(u_arr) - 1):
        a, b = u_arr[i], u_arr[i + 1]
        fa, fb = y_arr[i], y_arr[i + 1]
        whole = _trapezoidal_single(a, b, fa, fb)
        refined = _adaptive_refine(a, b, fa, fb, whole, func, tol, max_depth, 0)
        total += refined

    return float(total)


def central_difference(x: List[float], y: List[float],
                      target_points: Optional[Union[float, List[float]]] = None) -> Union[float, List[float]]:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        raise ValueError("At least 3 points are required for central difference")

    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)

    x_arr, y_arr = _sort_data(x_arr, y_arr)

    n = len(x_arr)
    derivatives = np.zeros(n)

    derivatives[0] = (y_arr[1] - y_arr[0]) / (x_arr[1] - x_arr[0])

    for i in range(1, n - 1):
        h_left = x_arr[i] - x_arr[i-1]
        h_right = x_arr[i+1] - x_arr[i]
        derivatives[i] = (y_arr[i+1] - y_arr[i-1]) / (h_left + h_right)

    derivatives[-1] = (y_arr[-1] - y_arr[-2]) / (x_arr[-1] - x_arr[-2])

    if target_points is None:
        return [float(d) for d in derivatives]

    if isinstance(target_points, (int, float)):
        target = float(target_points)
        idx = np.searchsorted(x_arr, target)
        if idx == 0:
            return float(derivatives[0])
        elif idx == n:
            return float(derivatives[-1])
        elif x_arr[idx] == target:
            return float(derivatives[idx])
        else:
            return float(np.interp(target, x_arr, derivatives))

    results = []
    for tp in target_points:
        idx = np.searchsorted(x_arr, tp)
        if idx == 0:
            results.append(float(derivatives[0]))
        elif idx == n:
            results.append(float(derivatives[-1]))
        elif x_arr[idx] == tp:
            results.append(float(derivatives[idx]))
        else:
            results.append(float(np.interp(tp, x_arr, derivatives)))

    return results


def second_derivative(x: List[float], y: List[float],
                      target_points: Optional[Union[float, List[float]]] = None) -> Union[float, List[float]]:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        raise ValueError("At least 3 points are required for second derivative")

    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)

    x_arr, y_arr = _sort_data(x_arr, y_arr)

    n = len(x_arr)
    d2y = np.zeros(n)

    for i in range(1, n - 1):
        h_left = x_arr[i] - x_arr[i-1]
        h_right = x_arr[i+1] - x_arr[i]

        d2y[i] = 2.0 * (h_left * y_arr[i+1] - (h_left + h_right) * y_arr[i] + h_right * y_arr[i-1]) / (h_left * h_right * (h_left + h_right))

    if n >= 4:
        h0 = x_arr[1] - x_arr[0]
        h1 = x_arr[2] - x_arr[1]

        d2y[0] = 2.0 * (y_arr[0] / (h0 * (h0 + h1))
                        - y_arr[1] / (h0 * h1)
                        + y_arr[2] / (h1 * (h0 + h1)))

        h_last = x_arr[-1] - x_arr[-2]
        h_prev = x_arr[-2] - x_arr[-3]

        d2y[-1] = 2.0 * (y_arr[-3] / (h_prev * (h_prev + h_last))
                         - y_arr[-2] / (h_prev * h_last)
                         + y_arr[-1] / (h_last * (h_prev + h_last)))
    else:
        d2y[0] = d2y[1]
        d2y[-1] = d2y[1]

    if target_points is None:
        return [float(d) for d in d2y]

    if isinstance(target_points, (int, float)):
        target = float(target_points)
        idx = np.searchsorted(x_arr, target)
        if idx == 0:
            return float(d2y[0])
        elif idx == n:
            return float(d2y[-1])
        elif x_arr[idx] == target:
            return float(d2y[idx])
        else:
            return float(np.interp(target, x_arr, d2y))

    results = []
    for tp in target_points:
        idx = np.searchsorted(x_arr, tp)
        if idx == 0:
            results.append(float(d2y[0]))
        elif idx == n:
            results.append(float(d2y[-1]))
        elif x_arr[idx] == tp:
            results.append(float(d2y[idx]))
        else:
            results.append(float(np.interp(tp, x_arr, d2y)))

    return results


def spline_derivative(x: List[float], y: List[float],
                      target_points: Optional[Union[float, List[float]]] = None,
                      order: int = 1,
                      extrapolate: bool = True) -> Union[float, List[float]]:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 4:
        raise ValueError("At least 4 points are required for cubic spline interpolation")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)

    x_arr, y_arr = _sort_data(x_arr, y_arr)

    n = len(x_arr)

    try:
        from scipy.interpolate import CubicSpline
        if extrapolate:
            cs = CubicSpline(x_arr, y_arr, bc_type='not-a-knot', extrapolate=True)
        else:
            cs = CubicSpline(x_arr, y_arr, bc_type='not-a-knot', extrapolate=False)
    except ImportError:
        cs = _cubic_spline_interpolant(x_arr, y_arr, bc_type='not-a-knot', extrapolate=extrapolate)

    if target_points is None:
        eval_points = x_arr
    elif isinstance(target_points, (int, float)):
        eval_points = np.array([float(target_points)])
    else:
        eval_points = np.array([float(tp) for tp in target_points])

    derivs = cs(eval_points, nu=order)

    if isinstance(target_points, (int, float)):
        return float(derivs[0])

    return [float(d) for d in derivs]


def _cubic_spline_interpolant(x: np.ndarray, y: np.ndarray, 
                              bc_type: str = 'not-a-knot',
                              extrapolate: bool = True):
    n = len(x)
    h = np.diff(x)

    A = np.zeros((n, n))
    b = np.zeros(n)

    if bc_type == 'natural':
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        b[0] = 0.0
        b[-1] = 0.0
    elif bc_type == 'not-a-knot':
        if n < 4:
            A[0, 0] = 1.0
            A[-1, -1] = 1.0
            b[0] = 0.0
            b[-1] = 0.0
        else:
            A[0, 0] = h[1]
            A[0, 1] = -(h[0] + h[1])
            A[0, 2] = h[0]
            b[0] = 0.0

            A[-1, -3] = h[-2]
            A[-1, -2] = -(h[-3] + h[-2])
            A[-1, -1] = h[-3]
            b[-1] = 0.0
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    for i in range(1, n - 1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2.0 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    c = np.linalg.solve(A, b)

    d = np.zeros(n - 1)
    b_coeff = np.zeros(n - 1)
    a_coeff = y[:-1].copy()

    for i in range(n - 1):
        d[i] = (c[i+1] - c[i]) / (3.0 * h[i])
        b_coeff[i] = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2.0 * c[i]) / 3.0

    def interpolant(t, nu=0):
        t_arr = np.atleast_1d(np.array(t, dtype=np.float64))
        result = np.zeros_like(t_arr)

        for j, tj in enumerate(t_arr):
            if tj < x[0]:
                if not extrapolate:
                    result[j] = np.nan
                    continue
                t_rel = tj - x[0]
                if nu == 0:
                    result[j] = a_coeff[0] + b_coeff[0] * t_rel + c[0] * t_rel**2 + d[0] * t_rel**3
                elif nu == 1:
                    result[j] = b_coeff[0] + 2.0 * c[0] * t_rel + 3.0 * d[0] * t_rel**2
                else:
                    result[j] = 2.0 * c[0] + 6.0 * d[0] * t_rel
            elif tj > x[-1]:
                if not extrapolate:
                    result[j] = np.nan
                    continue
                t_rel = tj - x[-2]
                if nu == 0:
                    result[j] = a_coeff[-1] + b_coeff[-1] * t_rel + c[-2] * t_rel**2 + d[-1] * t_rel**3
                elif nu == 1:
                    result[j] = b_coeff[-1] + 2.0 * c[-2] * t_rel + 3.0 * d[-1] * t_rel**2
                else:
                    result[j] = 2.0 * c[-2] + 6.0 * d[-1] * t_rel
            else:
                idx = np.searchsorted(x, tj) - 1
                if idx < 0:
                    idx = 0
                if idx >= n - 1:
                    idx = n - 2
                t_rel = tj - x[idx]
                if nu == 0:
                    result[j] = a_coeff[idx] + b_coeff[idx] * t_rel + c[idx] * t_rel**2 + d[idx] * t_rel**3
                elif nu == 1:
                    result[j] = b_coeff[idx] + 2.0 * c[idx] * t_rel + 3.0 * d[idx] * t_rel**2
                else:
                    result[j] = 2.0 * c[idx] + 6.0 * d[idx] * t_rel

        return result

    return interpolant


def compute_derivatives(x: List[float], y: List[float],
                        target_points: Optional[Union[float, List[float]]] = None,
                        order: int = 1,
                        method: str = 'difference') -> Union[float, List[float]]:
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    method = method.lower()
    if method == 'difference':
        if order == 1:
            return central_difference(x, y, target_points)
        else:
            return second_derivative(x, y, target_points)
    elif method == 'spline':
        return spline_derivative(x, y, target_points, order=order, extrapolate=True)
    else:
        raise ValueError("method must be 'difference' or 'spline'")


def compute_integration_and_differentiation(x: List[float], y: List[float],
                                         target_points: Optional[Union[float, List[float]]] = None,
                                         log_scale: bool = False,
                                         adaptive: bool = False,
                                         method: str = 'trapezoidal',
                                         tol: float = 1e-8,
                                         deriv_order: int = 1,
                                         deriv_method: str = 'difference') -> Tuple[float, Union[float, List[float]]]:
    method = method.lower()
    if method not in ('trapezoidal', 'simpsons'):
        raise ValueError("integration method must be 'trapezoidal' or 'simpsons'")

    if adaptive:
        if method == 'simpsons':
            if log_scale:
                integral = log_adaptive_trapezoidal_integration(x, y, tol=tol)
            else:
                integral = adaptive_simpsons_integration(x, y, tol=tol)
        else:
            if log_scale:
                integral = log_adaptive_trapezoidal_integration(x, y, tol=tol)
            else:
                integral = adaptive_trapezoidal_integration(x, y, tol=tol)
    else:
        if method == 'simpsons':
            integral = simpsons_integration(x, y)
        else:
            if log_scale:
                integral = log_trapezoidal_integration(x, y)
            else:
                integral = trapezoidal_integration(x, y)

    derivatives = compute_derivatives(x, y, target_points, order=deriv_order, method=deriv_method)
    return integral, derivatives


if __name__ == "__main__":
    print("=" * 60)
    print("1. 等间距数据测试: y = x^2, x = [0,1,2,3,4,5]")
    print("=" * 60)
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 4, 9, 16, 25]

    integral = trapezoidal_integration(x, y)
    print(f"  梯形积分: {integral:.6f}  (理论值: {125/3:.6f})")

    derivatives = central_difference(x, y)
    print(f"  导数: {[f'{d:.2f}' for d in derivatives]}  (理论: 2x)")
    print()

    print("=" * 60)
    print("2. 非等间距数据测试: y = x^2")
    print("=" * 60)
    x_nonuni = [0, 0.5, 1, 3, 5]
    y_nonuni = [0, 0.25, 1, 9, 25]

    integral_nonuni = trapezoidal_integration(x_nonuni, y_nonuni)
    print(f"  x = {x_nonuni}")
    print(f"  y = {y_nonuni}")
    print(f"  梯形积分: {integral_nonuni:.6f}  (理论值: {125/3:.6f})")

    derivatives_nonuni = central_difference(x_nonuni, y_nonuni)
    print(f"  导数: {[f'{d:.4f}' for d in derivatives_nonuni]}")
    print()

    print("=" * 60)
    print("3. 无序x数据测试 (验证排序)")
    print("=" * 60)
    x_unsorted = [3, 0, 5, 1, 2, 4]
    y_unsorted = [9, 0, 25, 1, 4, 16]

    integral_unsorted = trapezoidal_integration(x_unsorted, y_unsorted)
    print(f"  x = {x_unsorted}")
    print(f"  y = {y_unsorted}")
    print(f"  梯形积分: {integral_unsorted:.6f}  (与排序后结果一致)")
    print()

    print("=" * 60)
    print("4. 对数坐标积分测试: y = 1/x, x = [1,10,100,1000]")
    print("=" * 60)
    x_log = [1, 10, 100, 1000]
    y_log = [1.0, 0.1, 0.01, 0.001]

    integral_log = log_trapezoidal_integration(x_log, y_log)
    exact_log_int = 1.0 - 1.0/1000
    print(f"  x = {x_log}")
    print(f"  y = {y_log}")
    print(f"  对数积分 int(y d(ln x)) = int(1/x^2 dx): {integral_log:.6f}  (理论值: {exact_log_int:.6f})")

    integral_lin = trapezoidal_integration(x_log, y_log)
    exact_lin_int = np.log(1000)
    print(f"  线性积分 int(y dx) = int(1/x dx): {integral_lin:.6f}  (理论值: {exact_lin_int:.6f})")
    print()

    print("=" * 60)
    print("5. 自适应积分测试: y = x^3, x = [0,1,2,3,4,5]")
    print("=" * 60)
    x3 = [0, 1, 2, 3, 4, 5]
    y3 = [0, 1, 8, 27, 64, 125]

    integral_adaptive = adaptive_trapezoidal_integration(x3, y3, tol=1e-10)
    integral_trap = trapezoidal_integration(x3, y3)
    exact_x3 = 5**4 / 4
    print(f"  x = {x3}, y = x^3")
    print(f"  梯形积分:       {integral_trap:.6f}")
    print(f"  自适应积分:     {integral_adaptive:.6f}")
    print(f"  理论值(int x^3 dx): {exact_x3:.6f}")
    print(f"  梯形误差:       {abs(integral_trap - exact_x3):.6e}")
    print(f"  自适应误差:     {abs(integral_adaptive - exact_x3):.6e}")
    print()

    print("=" * 60)
    print("6. 对数坐标自适应积分测试: y = x^2, x = [1,2,5,10]")
    print("=" * 60)
    x_log2 = [1, 2, 5, 10]
    y_log2 = [1, 4, 25, 100]

    integral_log_trap2 = log_trapezoidal_integration(x_log2, y_log2)
    integral_log_adapt2 = log_adaptive_trapezoidal_integration(x_log2, y_log2, tol=1e-8)
    exact_log2 = (10**2 - 1**2) / 2
    print(f"  x = {x_log2}, y = x^2")
    print(f"  int(x^2 d(ln x)) = int(x dx), 理论值(1到10): {exact_log2:.6f}")
    print(f"  对数梯形积分:   {integral_log_trap2:.6f}")
    print(f"  对数自适应积分: {integral_log_adapt2:.6f}")
    print(f"  梯形误差:       {abs(integral_log_trap2 - exact_log2):.6e}")
    print(f"  自适应误差:     {abs(integral_log_adapt2 - exact_log2):.6e}")
    print()

    print("=" * 60)
    print("7. 综合接口 compute_integration_and_differentiation 测试")
    print("=" * 60)
    x_test = [1, 2, 3, 4, 5]
    y_test = [1, 4, 9, 16, 25]

    integral_default, deriv_default = compute_integration_and_differentiation(x_test, y_test)
    print(f"  默认(线性+梯形): 积分={integral_default:.6f}, 导数={[f'{d:.2f}' for d in deriv_default]}")

    integral_log3, deriv_log3 = compute_integration_and_differentiation(x_test, y_test, log_scale=True)
    print(f"  对数坐标梯形:    积分={integral_log3:.6f}")

    integral_adapt3, deriv_adapt3 = compute_integration_and_differentiation(x_test, y_test, adaptive=True)
    print(f"  线性自适应:      积分={integral_adapt3:.6f}")

    integral_log_adapt3, _ = compute_integration_and_differentiation(x_test, y_test, log_scale=True, adaptive=True)
    print(f"  对数自适应:      积分={integral_log_adapt3:.6f}")
    print()

    print("=" * 60)
    print("8. 辛普森积分测试 (等间距): y = x^3")
    print("=" * 60)
    x_simp = [0, 1, 2, 3, 4, 5, 6]
    y_simp = [0, 1, 8, 27, 64, 125, 216]
    exact_simp = 6**4 / 4

    integral_trap = trapezoidal_integration(x_simp, y_simp)
    integral_simp = simpsons_integration(x_simp, y_simp)
    print(f"  x = {x_simp}, y = x^3")
    print(f"  梯形积分:    {integral_trap:.6f}")
    print(f"  辛普森积分:  {integral_simp:.6f}")
    print(f"  理论值:      {exact_simp:.6f}")
    print(f"  梯形误差:    {abs(integral_trap - exact_simp):.6e}")
    print(f"  辛普森误差:  {abs(integral_simp - exact_simp):.6e}")
    print()

    print("=" * 60)
    print("9. 辛普森积分测试 (非等间距): y = x^4")
    print("=" * 60)
    x_simp2 = [0, 1, 2, 4, 6]
    y_simp2 = [0, 1, 16, 256, 1296]
    exact_simp2 = 6**5 / 5

    integral_trap2 = trapezoidal_integration(x_simp2, y_simp2)
    integral_simp2 = simpsons_integration(x_simp2, y_simp2)
    print(f"  x = {x_simp2}, y = x^4")
    print(f"  梯形积分:    {integral_trap2:.6f}")
    print(f"  辛普森积分:  {integral_simp2:.6f}")
    print(f"  理论值:      {exact_simp2:.6f}")
    print(f"  梯形误差:    {abs(integral_trap2 - exact_simp2):.6e}")
    print(f"  辛普森误差:  {abs(integral_simp2 - exact_simp2):.6e}")
    print()

    print("=" * 60)
    print("10. 自适应辛普森积分测试: y = x^5")
    print("=" * 60)
    x_ada = [0, 2, 4, 6]
    y_ada = [0, 32, 1024, 7776]
    exact_ada = 6**6 / 6

    integral_simp3 = simpsons_integration(x_ada, y_ada)
    integral_ada_simp = adaptive_simpsons_integration(x_ada, y_ada, tol=1e-10)
    print(f"  x = {x_ada}, y = x^5")
    print(f"  辛普森积分:     {integral_simp3:.6f}")
    print(f"  自适应辛普森:   {integral_ada_simp:.6f}")
    print(f"  理论值:         {exact_ada:.6f}")
    print(f"  辛普森误差:     {abs(integral_simp3 - exact_ada):.6e}")
    print(f"  自适应误差:     {abs(integral_ada_simp - exact_ada):.6e}")
    print()

    print("=" * 60)
    print("11. 二阶导数测试: y = x^3 (理论 d2y/dx2 = 6x)")
    print("=" * 60)
    x_d2 = [0, 1, 2, 3, 4, 5]
    y_d2 = [0, 1, 8, 27, 64, 125]

    d2y_diff = second_derivative(x_d2, y_d2)
    d2y_spline = spline_derivative(x_d2, y_d2, order=2)
    d2y_exact = [6 * xi for xi in x_d2]

    print(f"  x = {x_d2}")
    print(f"  y = x^3")
    print(f"  理论二阶导:     {[f'{d:.2f}' for d in d2y_exact]}")
    print(f"  差分二阶导:     {[f'{d:.2f}' for d in d2y_diff]}")
    print(f"  样条二阶导:     {[f'{d:.2f}' for d in d2y_spline]}")
    print()

    print("=" * 60)
    print("12. 端点误差对比: 差分 vs 样条 (一阶导数)")
    print("=" * 60)
    x_end = [0, 1, 2, 3, 4, 5, 6, 7]
    y_end = [xi**2 for xi in x_end]
    dy_exact = [2 * xi for xi in x_end]

    dy_diff = central_difference(x_end, y_end)
    dy_spline = spline_derivative(x_end, y_end)

    print(f"  x = {x_end}, y = x^2")
    print(f"  理论一阶导:     {[f'{d:.2f}' for d in dy_exact]}")
    print(f"  差分一阶导:     {[f'{d:.2f}' for d in dy_diff]}")
    print(f"  样条一阶导:     {[f'{d:.2f}' for d in dy_spline]}")
    print(f"  差分端点误差(左):  {abs(dy_diff[0] - dy_exact[0]):.4f}")
    print(f"  样条端点误差(左):  {abs(dy_spline[0] - dy_exact[0]):.4f}")
    print(f"  差分端点误差(右):  {abs(dy_diff[-1] - dy_exact[-1]):.4f}")
    print(f"  样条端点误差(右):  {abs(dy_spline[-1] - dy_exact[-1]):.4f}")
    print()

    print("=" * 60)
    print("13. 样条外推测试 (超出数据范围求导)")
    print("=" * 60)
    x_ext = [1, 2, 3, 4, 5]
    y_ext = [xi**2 for xi in x_ext]
    target_ext = [-0.5, 0, 0.5, 5.5, 6, 6.5]
    dy_exact_ext = [2 * t for t in target_ext]
    d2y_exact_ext = [2.0 for _ in target_ext]

    dy_ext = spline_derivative(x_ext, y_ext, target_points=target_ext, order=1, extrapolate=True)
    d2y_ext = spline_derivative(x_ext, y_ext, target_points=target_ext, order=2, extrapolate=True)

    print(f"  数据范围: x = [{x_ext[0]}, {x_ext[-1]}], y = x^2")
    print(f"  外推点:    {target_ext}")
    print(f"  理论一阶导:  {[f'{d:.2f}' for d in dy_exact_ext]}")
    print(f"  样条一阶导:  {[f'{d:.2f}' for d in dy_ext]}")
    print(f"  理论二阶导:  {[f'{d:.2f}' for d in d2y_exact_ext]}")
    print(f"  样条二阶导:  {[f'{d:.2f}' for d in d2y_ext]}")
    print()

    print("=" * 60)
    print("14. compute_derivatives 综合接口测试")
    print("=" * 60)
    x_cd = [0, 1, 2, 3, 4, 5]
    y_cd = [xi**3 for xi in x_cd]

    dy1_diff = compute_derivatives(x_cd, y_cd, order=1, method='difference')
    dy1_spline = compute_derivatives(x_cd, y_cd, order=1, method='spline')
    dy2_diff = compute_derivatives(x_cd, y_cd, order=2, method='difference')
    dy2_spline = compute_derivatives(x_cd, y_cd, order=2, method='spline')

    print(f"  y = x^3")
    print(f"  一阶导(差分): {[f'{d:.2f}' for d in dy1_diff]}")
    print(f"  一阶导(样条): {[f'{d:.2f}' for d in dy1_spline]}")
    print(f"  二阶导(差分): {[f'{d:.2f}' for d in dy2_diff]}")
    print(f"  二阶导(样条): {[f'{d:.2f}' for d in dy2_spline]}")
    print()

    print("=" * 60)
    print("15. 综合接口全参数测试")
    print("=" * 60)
    x_all = [1, 2, 3, 4, 5, 6, 7]
    y_all = [xi**3 for xi in x_all]

    int_simp, d1_spline = compute_integration_and_differentiation(
        x_all, y_all, method='simpsons', deriv_order=1, deriv_method='spline')
    int_ada_simp, d2_spline = compute_integration_and_differentiation(
        x_all, y_all, adaptive=True, method='simpsons', deriv_order=2, deriv_method='spline', tol=1e-10)

    exact_all = 7**4 / 4 - 1**4 / 4
    print(f"  y = x^3, x in [1,7]")
    print(f"  辛普森积分 + 样条一阶导:")
    print(f"    积分 = {int_simp:.6f} (理论 {exact_all:.6f})")
    print(f"    一阶导 = {[f'{d:.2f}' for d in d1_spline]}")
    print(f"  自适应辛普森 + 样条二阶导:")
    print(f"    积分 = {int_ada_simp:.6f} (误差 {abs(int_ada_simp - exact_all):.6e})")
    print(f"    二阶导 = {[f'{d:.2f}' for d in d2_spline]} (理论 6x)")
