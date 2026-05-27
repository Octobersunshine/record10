def _validate_points(x_points, y_points, min_points=2):
    if len(x_points) != len(y_points):
        raise ValueError("x_points 和 y_points 长度必须相同")
    if len(x_points) < min_points:
        raise ValueError(f"至少需要{min_points}个点")
    for i in range(1, len(x_points)):
        if x_points[i] <= x_points[i-1]:
            raise ValueError("x_points 必须严格按升序排列")


def _validate_extrapolation(extrapolation):
    valid = {"raise", "const", "linear"}
    if extrapolation not in valid:
        raise ValueError(f"extrapolation 必须是 {valid} 之一")


def _find_segment(x_points, xq):
    n = len(x_points)
    left, right = 0, n - 1
    while left < right - 1:
        mid = (left + right) // 2
        if x_points[mid] <= xq:
            left = mid
        else:
            right = mid
    return left


def _handle_extrapolation(x_points, y_points, xq, left_slope, right_slope, extrapolation):
    x_min, x_max = x_points[0], x_points[-1]
    if xq < x_min:
        if extrapolation == "raise":
            raise ValueError(f"查询点 {xq} 超出范围 [{x_min}, {x_max}]")
        elif extrapolation == "const":
            return y_points[0]
        else:
            return y_points[0] + left_slope * (xq - x_min)
    if xq > x_max:
        if extrapolation == "raise":
            raise ValueError(f"查询点 {xq} 超出范围 [{x_min}, {x_max}]")
        elif extrapolation == "const":
            return y_points[-1]
        else:
            return y_points[-1] + right_slope * (xq - x_max)
    return None


def _solve_tridiagonal(lower, main, upper, rhs):
    n = len(main)
    c = [0.0] * n
    d = [0.0] * n
    c[0] = upper[0] / main[0]
    d[0] = rhs[0] / main[0]
    for i in range(1, n):
        denom = main[i] - lower[i] * c[i - 1]
        if i < n - 1:
            c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    x = [0.0] * n
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _eval_cubic(a, b, c, d, x0, xq):
    dx = xq - x0
    return a + b * dx + c * dx * dx + d * dx * dx * dx


def linear_interpolation(x_points, y_points, x_query, extrapolation="const"):
    """
    一维线性插值

    参数:
        x_points: 已知点的x坐标列表，必须按升序排列
        y_points: 已知点的y坐标列表
        x_query: 查询点的x坐标，可以是单个值或列表
        extrapolation: 外推处理方式 ("raise" / "const" / "linear")

    返回:
        插值后的y值
    """
    _validate_extrapolation(extrapolation)
    _validate_points(x_points, y_points, min_points=2)

    x = [float(v) for v in x_points]
    y = [float(v) for v in y_points]

    left_slope = (y[1] - y[0]) / (x[1] - x[0])
    right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])

    def eval_single(xq):
        ext = _handle_extrapolation(x, y, xq, left_slope, right_slope, extrapolation)
        if ext is not None:
            return ext
        i = _find_segment(x, xq)
        t = (xq - x[i]) / (x[i + 1] - x[i])
        return y[i] + t * (y[i + 1] - y[i])

    if isinstance(x_query, (list, tuple)):
        return [eval_single(float(q)) for q in x_query]
    return eval_single(float(x_query))


def cubic_spline_interp(x_points, y_points, x_query, extrapolation="const"):
    """
    三次样条插值（自然边界条件: S''(x_0)=0, S''(x_n)=0）

    参数:
        x_points: 已知点的x坐标列表，必须按升序排列
        y_points: 已知点的y坐标列表
        x_query: 查询点的x坐标，可以是单个值或列表
        extrapolation: 外推处理方式 ("raise" / "const" / "linear")

    返回:
        字典，包含:
            "y": 插值结果（单个值或列表）
            "coeffs": 各段系数列表 [(a_i, b_i, c_i, d_i), ...]
    """
    _validate_extrapolation(extrapolation)
    _validate_points(x_points, y_points, min_points=3)

    x = [float(v) for v in x_points]
    y = [float(v) for v in y_points]
    n = len(x) - 1

    h = [x[i + 1] - x[i] for i in range(n)]

    if n == 1:
        m = [0.0, 0.0]
    elif n == 2:
        rhs_1 = 6.0 * ((y[2] - y[1]) / h[1] - (y[1] - y[0]) / h[0])
        m_1 = rhs_1 / (2.0 * (h[0] + h[1]))
        m = [0.0, m_1, 0.0]
    else:
        td_lower = [0.0] + [h[i] for i in range(n - 1)] + [0.0]
        td_main = [1.0] + [2.0 * (h[i] + h[i + 1]) for i in range(n - 1)] + [1.0]
        td_upper = [0.0] + [h[i + 1] for i in range(n - 1)] + [0.0]
        rhs = [0.0] + [6.0 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i]) for i in range(n - 1)] + [0.0]
        m = _solve_tridiagonal(td_lower, td_main, td_upper, rhs)

    coeffs = []
    for i in range(n):
        a_i = y[i]
        b_i = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * m[i] + m[i + 1]) / 6.0
        c_i = m[i] / 2.0
        d_i = (m[i + 1] - m[i]) / (6.0 * h[i])
        coeffs.append((a_i, b_i, c_i, d_i))

    left_slope = coeffs[0][1]
    last = coeffs[-1]
    right_slope = last[1] + 2.0 * last[2] * h[-1] + 3.0 * last[3] * h[-1] ** 2

    def eval_single(xq):
        ext = _handle_extrapolation(x, y, xq, left_slope, right_slope, extrapolation)
        if ext is not None:
            return ext
        i = _find_segment(x, xq)
        a_i, b_i, c_i, d_i = coeffs[i]
        return _eval_cubic(a_i, b_i, c_i, d_i, x[i], xq)

    if isinstance(x_query, (list, tuple)):
        result = [eval_single(float(q)) for q in x_query]
    else:
        result = eval_single(float(x_query))

    return {"y": result, "coeffs": coeffs}


def akima_interp(x_points, y_points, x_query, extrapolation="const"):
    """
    Akima插值（基于局部斜率加权的分段三次插值，对异常值更鲁棒）

    参数:
        x_points: 已知点的x坐标列表，必须按升序排列
        y_points: 已知点的y坐标列表
        x_query: 查询点的x坐标，可以是单个值或列表
        extrapolation: 外推处理方式 ("raise" / "const" / "linear")

    返回:
        字典，包含:
            "y": 插值结果（单个值或列表）
            "coeffs": 各段系数列表 [(a_i, b_i, c_i, d_i), ...]
            "slopes": 各数据点处的斜率 [t_0, t_1, ..., t_n]
    """
    _validate_extrapolation(extrapolation)
    _validate_points(x_points, y_points, min_points=3)

    x = [float(v) for v in x_points]
    y = [float(v) for v in y_points]
    n = len(x) - 1

    s = [(y[i + 1] - y[i]) / (x[i + 1] - x[i]) for i in range(n)]

    if n >= 3:
        s_ext = [
            3.0 * s[0] - 3.0 * s[1] + s[2],
            2.0 * s[0] - s[1],
        ] + s + [
            2.0 * s[-1] - s[-2],
            3.0 * s[-1] - 3.0 * s[-2] + s[-3],
        ]
    elif n == 2:
        s_ext = [2.0 * s[0] - s[1], s[0]] + s + [s[-1], 2.0 * s[-1] - s[-2]]
    else:
        s_ext = [s[0], s[0]] + s + [s[0], s[0]]

    t = []
    for i in range(n + 1):
        si_minus2 = s_ext[i]
        si_minus1 = s_ext[i + 1]
        si = s_ext[i + 2]
        si_plus1 = s_ext[i + 3]
        w1 = abs(si_plus1 - si)
        w2 = abs(si_minus1 - si_minus2)
        if w1 + w2 == 0.0:
            t.append((si_minus1 + si) / 2.0)
        else:
            t.append((w1 * si_minus1 + w2 * si) / (w1 + w2))

    coeffs = []
    for i in range(n):
        h_i = x[i + 1] - x[i]
        a_i = y[i]
        b_i = t[i]
        c_i = (3.0 * s[i] - 2.0 * t[i] - t[i + 1]) / h_i
        d_i = (t[i] + t[i + 1] - 2.0 * s[i]) / (h_i * h_i)
        coeffs.append((a_i, b_i, c_i, d_i))

    left_slope = t[0]
    right_slope = t[-1]

    def eval_single(xq):
        ext = _handle_extrapolation(x, y, xq, left_slope, right_slope, extrapolation)
        if ext is not None:
            return ext
        i = _find_segment(x, xq)
        a_i, b_i, c_i, d_i = coeffs[i]
        return _eval_cubic(a_i, b_i, c_i, d_i, x[i], xq)

    if isinstance(x_query, (list, tuple)):
        result = [eval_single(float(q)) for q in x_query]
    else:
        result = eval_single(float(x_query))

    return {"y": result, "coeffs": coeffs, "slopes": t}


if __name__ == "__main__":
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    x2 = [0.0, 1.0, 2.0, 3.0, 4.0]
    y2 = [0.0, 1.0, 4.0, 9.0, 16.0]
    queries = [0.5, 1.5, 2.5, 3.5]

    print("=" * 60)
    print("线性插值")
    print("=" * 60)
    print(f"数据点 y = x^2: x = {x2}, y = {y2}")
    for q in queries:
        print(f"  linear_interp({q}) = {linear_interpolation(x2, y2, q)}")
    print(f"  批量查询: {linear_interpolation(x2, y2, queries)}")
    print()

    print("=" * 60)
    print("三次样条插值（自然边界条件）")
    print("=" * 60)
    print(f"数据点 y = x^2: x = {x2}, y = {y2}")
    sp = cubic_spline_interp(x2, y2, 0.5)
    print(f"  单点查询: spline(0.5) = {sp['y']}")
    sp_batch = cubic_spline_interp(x2, y2, queries)
    print(f"  批量查询: spline({queries}) = {sp_batch['y']}")
    print(f"  各段系数 (a, b, c, d):")
    for i, c in enumerate(sp_batch["coeffs"]):
        print(f"    段{i} [{x2[i]}, {x2[i+1]}]: a={c[0]:.4f}, b={c[1]:.4f}, c={c[2]:.4f}, d={c[3]:.4f}")
    print()

    print("=" * 60)
    print("Akima插值")
    print("=" * 60)
    print(f"数据点 y = x^2: x = {x2}, y = {y2}")
    ak = akima_interp(x2, y2, 0.5)
    print(f"  单点查询: akima(0.5) = {ak['y']}")
    ak_batch = akima_interp(x2, y2, queries)
    print(f"  批量查询: akima({queries}) = {ak_batch['y']}")
    print(f"  各点斜率: {[f'{s:.4f}' for s in ak_batch['slopes']]}")
    print(f"  各段系数 (a, b, c, d):")
    for i, c in enumerate(ak_batch["coeffs"]):
        print(f"    段{i} [{x2[i]}, {x2[i+1]}]: a={c[0]:.4f}, b={c[1]:.4f}, c={c[2]:.4f}, d={c[3]:.4f}")
    print()

    print("=" * 60)
    print("三种方法对比: y = x^2")
    print("=" * 60)
    dense_q = [i * 0.5 for i in range(9)]
    lin = linear_interpolation(x2, y2, dense_q)
    sp_d = cubic_spline_interp(x2, y2, dense_q)
    ak_d = akima_interp(x2, y2, dense_q)
    print(f"{'x':>6s}  {'真实值':>8s}  {'线性':>8s}  {'样条':>8s}  {'Akima':>8s}")
    print("-" * 46)
    for i, q in enumerate(dense_q):
        true_val = q * q
        print(f"{q:6.1f}  {true_val:8.4f}  {lin[i]:8.4f}  {sp_d['y'][i]:8.4f}  {ak_d['y'][i]:8.4f}")
    print()

    print("=" * 60)
    print("外推测试: y = x^2, 查询 x = -1 和 x = 10")
    print("=" * 60)
    for ext_mode in ["const", "linear", "raise"]:
        print(f"\n  外推模式: {ext_mode}")
        for xq in [-1.0, 10.0]:
            try:
                lin_v = linear_interpolation(x2, y2, xq, extrapolation=ext_mode)
                sp_v = cubic_spline_interp(x2, y2, xq, extrapolation=ext_mode)["y"]
                ak_v = akima_interp(x2, y2, xq, extrapolation=ext_mode)["y"]
                print(f"    x={xq:5.1f} -> 线性={lin_v:8.4f}, 样条={sp_v:8.4f}, Akima={ak_v:8.4f}")
            except ValueError as e:
                print(f"    x={xq:5.1f} -> {e}")
    print()

    print("=" * 60)
    print("含异常值数据: Akima vs 样条")
    print("=" * 60)
    x_spiky = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y_spiky = [0.0, 1.0, 1.0, 10.0, 1.0, 1.0, 0.0]
    dense6 = [i * 0.5 for i in range(13)]
    sp_spike = cubic_spline_interp(x_spiky, y_spiky, dense6)
    ak_spike = akima_interp(x_spiky, y_spiky, dense6)
    print(f"{'x':>5s}  {'样条':>8s}  {'Akima':>8s}")
    print("-" * 26)
    for i, q in enumerate(dense6):
        print(f"{q:5.1f}  {sp_spike['y'][i]:8.4f}  {ak_spike['y'][i]:8.4f}")
