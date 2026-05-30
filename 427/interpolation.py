from typing import List, Tuple
import bisect

MAX_DEGREE = 10


def _find_exact_match(x_points: List[float], y_points: List[float], x: float, tol: float = 1e-12) -> Tuple[bool, float]:
    for xi, yi in zip(x_points, y_points):
        if abs(x - xi) < tol:
            return True, yi
    return False, 0.0


def piecewise_linear_interpolation(x_points: List[float], y_points: List[float], x: float) -> Tuple[float, List[float]]:
    n = len(x_points)
    if n != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    if n < 2:
        raise ValueError("piecewise linear interpolation requires at least 2 points")

    sorted_pairs = sorted(zip(x_points, y_points), key=lambda p: p[0])
    sx = [p[0] for p in sorted_pairs]
    sy = [p[1] for p in sorted_pairs]

    matched, val = _find_exact_match(sx, sy, x)
    if matched:
        return val, []

    if x <= sx[0]:
        return sy[0], []
    if x >= sx[-1]:
        return sy[-1], []

    idx = bisect.bisect_right(sx, x) - 1
    idx = max(0, min(idx, n - 2))

    x0, x1 = sx[idx], sx[idx + 1]
    y0, y1 = sy[idx], sy[idx + 1]

    t = (x - x0) / (x1 - x0)
    result = y0 + t * (y1 - y0)

    return result, []


def lagrange_interpolation(x_points: List[float], y_points: List[float], x: float) -> Tuple[float, List[float]]:
    n = len(x_points)
    if n != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    if n < 1:
        raise ValueError("at least 1 point is required")

    matched, val = _find_exact_match(x_points, y_points, x)
    if matched:
        coefficients = _build_coefficients(x_points, y_points)
        return val, coefficients

    if n - 1 > MAX_DEGREE:
        result, _ = piecewise_linear_interpolation(x_points, y_points, x)
        return result, []

    result = 0.0
    coefficients = [0.0] * n

    for i in range(n):
        term = y_points[i]
        numerator_poly = [1.0]

        for j in range(n):
            if j != i:
                denominator = x_points[i] - x_points[j]
                term *= (x - x_points[j]) / denominator

                new_poly = [0.0] * (len(numerator_poly) + 1)
                for k in range(len(numerator_poly)):
                    new_poly[k] += numerator_poly[k] * (-x_points[j])
                    new_poly[k + 1] += numerator_poly[k]
                numerator_poly = [coef / denominator for coef in new_poly]

        result += term
        for k in range(len(numerator_poly)):
            coefficients[k] += y_points[i] * numerator_poly[k]

    return result, coefficients


def newton_interpolation(x_points: List[float], y_points: List[float], x: float) -> Tuple[float, List[float]]:
    n = len(x_points)
    if n != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    if n < 1:
        raise ValueError("at least 1 point is required")

    matched, val = _find_exact_match(x_points, y_points, x)
    if matched:
        coefficients = _build_coefficients(x_points, y_points)
        return val, coefficients

    if n - 1 > MAX_DEGREE:
        result, _ = piecewise_linear_interpolation(x_points, y_points, x)
        return result, []

    diff_table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        diff_table[i][0] = y_points[i]

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / (x_points[i + j] - x_points[i])

    coefficients = [diff_table[0][j] for j in range(n)]

    result = coefficients[0]
    product = 1.0
    for i in range(1, n):
        product *= (x - x_points[i - 1])
        result += coefficients[i] * product

    poly_coefficients = [0.0] * n
    poly_coefficients[0] = coefficients[0]

    base_poly = [1.0]
    for i in range(1, n):
        new_base = [0.0] * (len(base_poly) + 1)
        for j in range(len(base_poly)):
            new_base[j] += base_poly[j] * (-x_points[i - 1])
            new_base[j + 1] += base_poly[j]
        base_poly = new_base

        for j in range(len(base_poly)):
            poly_coefficients[j] += coefficients[i] * base_poly[j]

    return result, poly_coefficients


def _build_coefficients(x_points: List[float], y_points: List[float]) -> List[float]:
    n = len(x_points)
    if n - 1 > MAX_DEGREE:
        return []

    diff_table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        diff_table[i][0] = y_points[i]

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / (x_points[i + j] - x_points[i])

    coefficients = [diff_table[0][j] for j in range(n)]

    poly_coefficients = [0.0] * n
    poly_coefficients[0] = coefficients[0]

    base_poly = [1.0]
    for i in range(1, n):
        new_base = [0.0] * (len(base_poly) + 1)
        for j in range(len(base_poly)):
            new_base[j] += base_poly[j] * (-x_points[i - 1])
            new_base[j + 1] += base_poly[j]
        base_poly = new_base

        for j in range(len(base_poly)):
            poly_coefficients[j] += coefficients[i] * base_poly[j]

    return poly_coefficients


def evaluate_polynomial(coefficients: List[float], x: float) -> float:
    if not coefficients:
        return float('nan')
    result = 0.0
    for i, coef in enumerate(coefficients):
        result += coef * (x ** i)
    return result


def _solve_tridiagonal(a: List[float], b: List[float], c: List[float], d: List[float]) -> List[float]:
    n = len(d)
    cp = [0.0] * n
    dp = [0.0] * n
    x = [0.0] * n

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m

    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def cubic_spline_interpolation(x_points: List[float], y_points: List[float], x: float) -> Tuple[float, List[float]]:
    n = len(x_points)
    if n != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    if n < 3:
        raise ValueError("cubic spline interpolation requires at least 3 points")

    sorted_pairs = sorted(zip(x_points, y_points), key=lambda p: p[0])
    sx = [p[0] for p in sorted_pairs]
    sy = [p[1] for p in sorted_pairs]

    matched, val = _find_exact_match(sx, sy, x)
    if matched:
        return val, []

    if x <= sx[0]:
        return sy[0], []
    if x >= sx[-1]:
        return sy[-1], []

    h = [sx[i + 1] - sx[i] for i in range(n - 1)]
    mu = [h[i] / (h[i] + h[i + 1]) for i in range(n - 2)]
    lam = [1.0 - mu[i] for i in range(n - 2)]

    d = [0.0] * n
    for i in range(1, n - 1):
        d[i] = 6.0 * ((sy[i + 1] - sy[i]) / h[i] - (sy[i] - sy[i - 1]) / h[i - 1]) / (h[i] + h[i - 1])

    a = [0.0] * n
    b = [2.0] * n
    c = [0.0] * n

    for i in range(1, n - 1):
        a[i] = mu[i - 1]
        c[i] = lam[i - 1]

    b[0] = 1.0
    c[0] = 0.0
    d[0] = 0.0
    a[n - 1] = 0.0
    b[n - 1] = 1.0
    d[n - 1] = 0.0

    M = _solve_tridiagonal(a, b, c, d)

    idx = bisect.bisect_right(sx, x) - 1
    idx = max(0, min(idx, n - 2))

    x0, x1 = sx[idx], sx[idx + 1]
    y0, y1 = sy[idx], sy[idx + 1]
    M0, M1 = M[idx], M[idx + 1]
    hi = h[idx]

    result = (
        M0 * (x1 - x) ** 3 / (6.0 * hi)
        + M1 * (x - x0) ** 3 / (6.0 * hi)
        + (y0 - M0 * hi ** 2 / 6.0) * (x1 - x) / hi
        + (y1 - M1 * hi ** 2 / 6.0) * (x - x0) / hi
    )

    return result, []


def generate_curve_points(
    x_points: List[float],
    y_points: List[float],
    method: str = "lagrange",
    num_points: int = 200
) -> Tuple[List[float], List[float]]:
    method = method.lower()
    if method not in ["lagrange", "newton", "spline", "linear"]:
        raise ValueError("method must be one of: 'lagrange', 'newton', 'spline', 'linear'")

    sorted_pairs = sorted(zip(x_points, y_points), key=lambda p: p[0])
    sx = [p[0] for p in sorted_pairs]
    sy = [p[1] for p in sorted_pairs]

    x_min, x_max = sx[0], sx[-1]
    step = (x_max - x_min) / (num_points - 1)
    x_curve = [x_min + i * step for i in range(num_points)]
    y_curve = []

    for x in x_curve:
        if method == "lagrange":
            y, _ = lagrange_interpolation(sx, sy, x)
        elif method == "newton":
            y, _ = newton_interpolation(sx, sy, x)
        elif method == "spline":
            y, _ = cubic_spline_interpolation(sx, sy, x)
        else:
            y, _ = piecewise_linear_interpolation(sx, sy, x)
        y_curve.append(y)

    return x_curve, y_curve


def compute_errors(
    x_points: List[float],
    y_points: List[float],
    true_func,
    num_test_points: int = 100
) -> dict:
    sorted_pairs = sorted(zip(x_points, y_points), key=lambda p: p[0])
    sx = [p[0] for p in sorted_pairs]
    sy = [p[1] for p in sorted_pairs]

    x_min, x_max = sx[0], sx[-1]
    step = (x_max - x_min) / (num_test_points - 1)
    test_x = [x_min + i * step for i in range(num_test_points)]

    errors = {
        "lagrange": [],
        "newton": [],
        "spline": [],
        "linear": []
    }

    for x in test_x:
        true_y = true_func(x)
        for method in errors:
            if method == "lagrange":
                y, _ = lagrange_interpolation(sx, sy, x)
            elif method == "newton":
                y, _ = newton_interpolation(sx, sy, x)
            elif method == "spline":
                y, _ = cubic_spline_interpolation(sx, sy, x)
            else:
                y, _ = piecewise_linear_interpolation(sx, sy, x)
            errors[method].append(abs(true_y - y))

    stats = {}
    for method, errs in errors.items():
        stats[method] = {
            "max_error": max(errs),
            "mean_error": sum(errs) / len(errs)
        }

    return stats


if __name__ == "__main__":
    import math

    print("=" * 60)
    print("测试1: 基本插值 (4点, 次数3 <= MAX_DEGREE=10)")
    print("=" * 60)

    x_points = [0.0, 1.0, 2.0, 3.0]
    y_points = [1.0, 2.71828, 7.38906, 20.08554]
    x_eval = 1.5

    lagrange_result, lagrange_coeffs = lagrange_interpolation(x_points, y_points, x_eval)
    newton_result, newton_coeffs = newton_interpolation(x_points, y_points, x_eval)
    spline_result, _ = cubic_spline_interpolation(x_points, y_points, x_eval)

    print(f"\n插值点 x = {x_eval}")
    print(f"拉格朗日插值结果: {lagrange_result:.6f}")
    print(f"牛顿插值结果: {newton_result:.6f}")
    print(f"三次样条插值结果: {spline_result:.6f}")

    print("\n在原始数据点上验证精确通过 (精度修复):")
    for xi, yi in zip(x_points, y_points):
        lag_val, _ = lagrange_interpolation(x_points, y_points, xi)
        new_val, _ = newton_interpolation(x_points, y_points, xi)
        spl_val, _ = cubic_spline_interpolation(x_points, y_points, xi)
        print(f"  x = {xi}: 真实值 = {yi:.10f}, 拉格朗日 = {lag_val:.10f}, 牛顿 = {new_val:.10f}, 样条 = {spl_val:.10f}")

    print("\n" + "=" * 60)
    print("测试2: Runge函数插值误差对比 (8点, 次数7 <= MAX_DEGREE=10)")
    print("=" * 60)

    n_runge = 8
    x_runge = [i * 2.0 / (n_runge - 1) - 1.0 for i in range(n_runge)]
    y_runge = [1.0 / (1.0 + 25.0 * xi ** 2) for xi in x_runge]

    def runge_func(x):
        return 1.0 / (1.0 + 25.0 * x ** 2)

    error_stats = compute_errors(x_runge, y_runge, runge_func, num_test_points=100)

    print(f"\nRunge函数 1/(1+25x^2) 在 [-1, 1] 区间的误差统计:")
    print(f"{'方法':<12} {'最大误差':<15} {'平均误差':<15}")
    print("-" * 45)
    for method in ["lagrange", "newton", "spline", "linear"]:
        stats = error_stats[method]
        method_name = {"lagrange": "拉格朗日", "newton": "牛顿", "spline": "三次样条", "linear": "分段线性"}[method]
        print(f"{method_name:<12} {stats['max_error']:<15.6e} {stats['mean_error']:<15.6e}")

    print("\n" + "=" * 60)
    print("测试3: 高次插值降级 + 样条对比 (12点, 次数11 > MAX_DEGREE=10)")
    print("=" * 60)

    n_runge2 = 12
    x_runge2 = [i * 2.0 / (n_runge2 - 1) - 1.0 for i in range(n_runge2)]
    y_runge2 = [1.0 / (1.0 + 25.0 * xi ** 2) for xi in x_runge2]
    x_eval_runge = 0.5

    lag_val_r, lag_coeffs_r = lagrange_interpolation(x_runge2, y_runge2, x_eval_runge)
    new_val_r, new_coeffs_r = newton_interpolation(x_runge2, y_runge2, x_eval_runge)
    spl_val_r, _ = cubic_spline_interpolation(x_runge2, y_runge2, x_eval_runge)
    pw_val_r, _ = piecewise_linear_interpolation(x_runge2, y_runge2, x_eval_runge)
    true_val_r = 1.0 / (1.0 + 25.0 * x_eval_runge ** 2)

    print(f"\nRunge函数评估点 x = {x_eval_runge}")
    print(f"真实值: {true_val_r:.10f}")
    print(f"拉格朗日 (已降级为分段线性): {lag_val_r:.10f}, 系数为空: {lag_coeffs_r == []}")
    print(f"牛顿 (已降级为分段线性): {new_val_r:.10f}, 系数为空: {new_coeffs_r == []}")
    print(f"三次样条 (未降级): {spl_val_r:.10f}")
    print(f"分段线性插值: {pw_val_r:.10f}")

    print("\n" + "=" * 60)
    print("测试4: 生成插值曲线点集 (用于前端绘图)")
    print("=" * 60)

    x_curve_lag, y_curve_lag = generate_curve_points(x_runge, y_runge, method="lagrange", num_points=10)
    x_curve_spl, y_curve_spl = generate_curve_points(x_runge, y_runge, method="spline", num_points=10)

    print(f"\n拉格朗日插值曲线点集 (前10个点):")
    for i in range(len(x_curve_lag)):
        print(f"  ({x_curve_lag[i]:.4f}, {y_curve_lag[i]:.6f})")

    print(f"\n三次样条插值曲线点集 (前10个点):")
    for i in range(len(x_curve_spl)):
        print(f"  ({x_curve_spl[i]:.4f}, {y_curve_spl[i]:.6f})")

    print("\n" + "=" * 60)
    print("测试5: 在已知点上验证精确匹配 (所有方法)")
    print("=" * 60)

    test_xi = x_runge2[5]
    test_yi = y_runge2[5]
    lag_v, _ = lagrange_interpolation(x_runge2, y_runge2, test_xi)
    new_v, _ = newton_interpolation(x_runge2, y_runge2, test_xi)
    spl_v, _ = cubic_spline_interpolation(x_runge2, y_runge2, test_xi)
    pw_v, _ = piecewise_linear_interpolation(x_runge2, y_runge2, test_xi)
    print(f"\nx = {test_xi}: 真实值 = {test_yi:.10f}")
    print(f"  拉格朗日 = {lag_v:.10f}, 牛顿 = {new_v:.10f}")
    print(f"  三次样条 = {spl_v:.10f}, 分段线性 = {pw_v:.10f}")
