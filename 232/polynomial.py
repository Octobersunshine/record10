import warnings

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre

MAX_DEGREE = 10


def horner_eval(coefficients, x):
    result = 0.0
    for coeff in coefficients:
        result = result * x + coeff
    return result


def _legendre_vandermonde(x, degree):
    n = len(x)
    m = degree + 1
    L = np.zeros((n, m))
    L[:, 0] = 1.0
    if degree >= 1:
        L[:, 1] = x
    for k in range(1, degree):
        L[:, k + 1] = ((2 * k + 1) * x * L[:, k] - k * L[:, k - 1]) / (k + 1)
    return L


def least_squares_fit(x_data, y_data, degree):
    if degree > MAX_DEGREE:
        raise ValueError(
            f"拟合次数 {degree} 超过最大限制 {MAX_DEGREE}，"
            f"高次多项式拟合数值不稳定，建议使用次数 ≤ {MAX_DEGREE}"
        )

    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)

    x_min, x_max = x_arr.min(), x_arr.max()
    span = x_max - x_min
    if span < 1e-15:
        x_mapped = np.zeros_like(x_arr)
    else:
        x_mapped = 2.0 * (x_arr - x_min) / span - 1.0

    L = _legendre_vandermonde(x_mapped, degree)

    cond = np.linalg.cond(L)
    info = {"cond_number": float(cond), "warning": None}
    if cond > 1e10:
        msg = f"条件数 {cond:.2e} 过大，拟合结果可能不可靠"
        info["warning"] = msg
        warnings.warn(msg, RuntimeWarning)

    leg_coeffs, _, _, _ = np.linalg.lstsq(L, y_arr, rcond=None)

    leg_poly = Legendre(leg_coeffs, domain=[x_min, x_max])
    std_poly = leg_poly.convert(kind=Polynomial)
    coefficients = std_poly.coef[::-1].tolist()

    return coefficients, info


def weighted_least_squares_fit(x_data, y_data, weights, degree):
    if degree > MAX_DEGREE:
        raise ValueError(
            f"拟合次数 {degree} 超过最大限制 {MAX_DEGREE}，"
            f"高次多项式拟合数值不稳定，建议使用次数 ≤ {MAX_DEGREE}"
        )

    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

    if len(w_arr) != len(x_arr):
        raise ValueError("权重数组长度必须与数据点数一致")
    if np.any(w_arr < 0):
        raise ValueError("权重必须非负")

    x_min, x_max = x_arr.min(), x_arr.max()
    span = x_max - x_min
    if span < 1e-15:
        x_mapped = np.zeros_like(x_arr)
    else:
        x_mapped = 2.0 * (x_arr - x_min) / span - 1.0

    L = _legendre_vandermonde(x_mapped, degree)

    W_sqrt = np.sqrt(w_arr)
    L_w = L * W_sqrt[:, np.newaxis]
    y_w = y_arr * W_sqrt

    cond = np.linalg.cond(L_w)
    info = {"cond_number": float(cond), "warning": None}
    if cond > 1e10:
        msg = f"条件数 {cond:.2e} 过大，拟合结果可能不可靠"
        info["warning"] = msg
        warnings.warn(msg, RuntimeWarning)

    leg_coeffs, _, _, _ = np.linalg.lstsq(L_w, y_w, rcond=None)

    leg_poly = Legendre(leg_coeffs, domain=[x_min, x_max])
    std_poly = leg_poly.convert(kind=Polynomial)
    coefficients = std_poly.coef[::-1].tolist()

    return coefficients, info


def poly_derivative(coefficients):
    n = len(coefficients)
    if n <= 1:
        return [0.0]
    deriv = []
    for i, coeff in enumerate(coefficients):
        power = n - 1 - i
        if power > 0:
            deriv.append(coeff * power)
    if not deriv:
        return [0.0]
    return deriv


def poly_integral(coefficients, C=0.0):
    n = len(coefficients)
    integ = []
    for i, coeff in enumerate(coefficients):
        power = n - 1 - i
        integ.append(coeff / (power + 1))
    integ.append(C)
    return integ


if __name__ == "__main__":
    poly_coeffs = [1.0, 2.0, 3.0]
    x = 2.0
    value = horner_eval(poly_coeffs, x)
    print(f"霍纳方法求值结果: {value}")

    x_data = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_data = [1.0, 3.0, 7.0, 13.0, 21.0]

    coeffs, info = least_squares_fit(x_data, y_data, 2)
    print(f"最小二乘拟合系数 (2次): {coeffs}")
    print(f"条件数: {info['cond_number']:.2e}")
    print(f"警告: {info['warning']}")

    print("\n===== 加权最小二乘拟合 =====")
    weights = [1.0, 1.0, 1.0, 10.0, 1.0]
    w_coeffs, w_info = weighted_least_squares_fit(x_data, y_data, weights, 2)
    print(f"加权拟合系数 (权重第4点×10): {w_coeffs}")
    print(f"条件数: {w_info['cond_number']:.2e}")

    np.random.seed(42)
    n_pts = 30
    x_noisy = np.linspace(-1, 1, n_pts)
    y_noisy = np.sin(2 * x_noisy) + 0.1 * np.random.randn(n_pts)
    w_random = np.random.uniform(0.5, 2.0, n_pts)

    w_coeffs2, w_info2 = weighted_least_squares_fit(
        x_noisy.tolist(), y_noisy.tolist(), w_random.tolist(), 5
    )
    print(f"带随机权重5次拟合, 条件数: {w_info2['cond_number']:.2e}")
    print(f"在 x=0.5 处的加权拟合值: {horner_eval(w_coeffs2, 0.5):.6f}")

    print("\n===== 多项式求导 =====")
    p = [3.0, 2.0, 1.0]
    print(f"P(x) = 3x² + 2x + 1, 系数: {p}")
    dp = poly_derivative(p)
    print(f"P'(x) = 6x + 2, 系数: {dp}")
    ddp = poly_derivative(dp)
    print(f"P''(x) = 6, 系数: {ddp}")
    dddp = poly_derivative(ddp)
    print(f"P'''(x) = 0, 系数: {dddp}")
    print(f"P'(2) = {horner_eval(dp, 2.0)} (期望 14.0)")
    print(f"P''(2) = {horner_eval(ddp, 2.0)} (期望 6.0)")

    print("\n===== 多项式积分 =====")
    ip = poly_integral(p, C=0.0)
    print(f"∫P(x)dx = x³ + x² + x + C, 系数: {ip}")
    ip_c5 = poly_integral(p, C=5.0)
    print(f"∫P(x)dx (C=5), 系数: {ip_c5}")
    print(f"∫P(x)dx 在 x=2 处的值 (C=0): {horner_eval(ip, 2.0):.6f} (期望 8+4+2=14.0)")
    print(f"∫P(x)dx 在 x=2 处的值 (C=5): {horner_eval(ip_c5, 2.0):.6f} (期望 19.0)")

    print("\n===== 导数+积分互逆验证 =====")
    p_orig = [1.0, -3.0, 0.0, 2.0]
    dp_orig = poly_derivative(p_orig)
    ip_orig = poly_integral(dp_orig, C=2.0)
    print(f"原多项式: {p_orig}")
    print(f"导函数:   {dp_orig}")
    print(f"导函数积分 (C=2): {ip_orig}")
    print(f"恢复原系数: a₀={ip_orig[0]:.1f}, a₁={ip_orig[1]:.1f}, a₂={ip_orig[2]:.1f}, a₃={ip_orig[3]:.1f}")

    for deg in [5, 10, 11]:
        print(f"\n--- 拟合次数: {deg} ---")
        try:
            c, info = least_squares_fit(x_noisy.tolist(), y_noisy.tolist(), deg)
            print(f"条件数: {info['cond_number']:.2e}")
            if info["warning"]:
                print(f"警告: {info['warning']}")
            val = horner_eval(c, 0.5)
            print(f"在 x=0.5 处的拟合值: {val:.6f}")
        except ValueError as e:
            print(f"错误: {e}")
