import numpy as np
import warnings


MAX_DEGREE = 10
CONDITION_NUMBER_WARNING_THRESHOLD = 1e10


def check_degree(degree):
    if degree < 0:
        raise ValueError(f"多项式次数不能为负数，当前值: {degree}")
    if degree > MAX_DEGREE:
        raise ValueError(
            f"多项式次数不能超过 {MAX_DEGREE}，当前值: {degree}。"
            f"高次多项式容易导致病态矩阵问题，建议使用低次拟合或正交多项式方法。"
        )


def compute_condition_number(X):
    singular_values = np.linalg.svd(X, compute_uv=False)
    max_sv = np.max(singular_values)
    min_sv = np.min(singular_values)
    if min_sv == 0:
        return np.inf
    return max_sv / min_sv


def check_condition_number(cond, method_name):
    if cond > CONDITION_NUMBER_WARNING_THRESHOLD:
        warnings.warn(
            f"{method_name} 条件数过大 ({cond:.2e})，"
            f"可能存在病态矩阵问题，建议使用正交多项式方法。",
            UserWarning,
            stacklevel=3
        )


def compute_r_squared(y, y_pred):
    """
    计算决定系数 R²
    
    R² = 1 - SS_res / SS_tot
    SS_res = Σ(y_i - y_pred_i)²
    SS_tot = Σ(y_i - y_mean)²
    
    参数:
        y: 真实值数组
        y_pred: 预测值数组
    
    返回:
        r_squared: 决定系数
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_residual_analysis(y, y_pred, max_lag=None):
    """
    残差分析：计算残差序列和自相关数据
    
    参数:
        y: 真实值数组
        y_pred: 预测值数组
        max_lag: 自相关最大滞后阶数，默认为 min(len(y)//2, 10)
    
    返回:
        dict: {
            'residuals': 残差序列,
            'residual_mean': 残差均值,
            'residual_std': 残差标准差,
            'autocorrelation': 自相关系数数组（lag 0 到 max_lag）,
            'durbin_watson': Durbin-Watson统计量
        }
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y - y_pred
    n = len(residuals)
    
    if max_lag is None:
        max_lag = min(n // 2, 10)
    max_lag = min(max_lag, n - 1)
    
    autocorr = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            r_mean = np.mean(residuals)
            centered = residuals - r_mean
            var = np.sum(centered ** 2)
            if var == 0:
                autocorr[lag] = 0.0
            else:
                autocorr[lag] = np.sum(centered[:n - lag] * centered[lag:]) / var
    
    dw_numerator = np.sum(np.diff(residuals) ** 2)
    dw_denominator = np.sum(residuals ** 2)
    durbin_watson = dw_numerator / dw_denominator if dw_denominator > 0 else 0.0
    
    return {
        'residuals': residuals,
        'residual_mean': np.mean(residuals),
        'residual_std': np.std(residuals, ddof=1),
        'autocorrelation': autocorr,
        'durbin_watson': durbin_watson,
    }


def build_design_matrix(x, basis_functions):
    """
    根据基函数列表构建设计矩阵
    
    参数:
        x: 自变量数组
        basis_functions: 基函数列表，每个元素是一个函数 f(x) -> array
                         或字符串标识: '1', 'x', 'x^2', 'x^3', 'exp', 'log', 'sin', 'cos', 'sqrt', '1/x'
    
    返回:
        X: 设计矩阵，每列对应一个基函数
        names: 基函数名称列表
    """
    x = np.asarray(x, dtype=float)
    
    basis_map = {
        '1': lambda t: np.ones_like(t),
        'x': lambda t: t.copy(),
        'x^2': lambda t: t ** 2,
        'x^3': lambda t: t ** 3,
        'exp': lambda t: np.exp(t),
        'log': lambda t: np.log(np.maximum(t, 1e-300)),
        'sin': lambda t: np.sin(t),
        'cos': lambda t: np.cos(t),
        'sqrt': lambda t: np.sqrt(np.maximum(t, 0)),
        '1/x': lambda t: 1.0 / np.where(np.abs(t) < 1e-300, 1e-300, t),
    }
    
    columns = []
    names = []
    
    for bf in basis_functions:
        if isinstance(bf, str):
            if bf not in basis_map:
                raise ValueError(
                    f"未知基函数标识: '{bf}'。"
                    f"支持的标识: {list(basis_map.keys())}"
                )
            columns.append(basis_map[bf](x))
            names.append(bf)
        elif callable(bf):
            col = np.asarray(bf(x), dtype=float)
            if col.shape != x.shape:
                raise ValueError(
                    f"基函数返回值形状不匹配: 期望 {x.shape}，得到 {col.shape}"
                )
            columns.append(col)
            names.append(getattr(bf, '__name__', f'f{len(columns)}'))
        else:
            raise TypeError(f"基函数必须是字符串或可调用对象，得到 {type(bf)}")
    
    X = np.column_stack(columns)
    return X, names


def polyfit_numpy(x, y, degree):
    check_degree(degree)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    X = np.vander(x, degree + 1)
    cond = compute_condition_number(X)
    check_condition_number(cond, "numpy.polyfit")
    
    coeffs = np.polyfit(x, y, degree)
    
    y_pred = np.polyval(coeffs, x)
    mse = np.mean((y - y_pred) ** 2)
    
    return coeffs, mse, cond, y_pred


def polyfit_manual(x, y, degree):
    check_degree(degree)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    
    X = np.vander(x, degree + 1)
    cond = compute_condition_number(X)
    check_condition_number(cond, "手动正规方程")
    
    XtX = X.T @ X
    Xty = X.T @ y
    
    coeffs = np.linalg.solve(XtX, Xty)
    
    y_pred = X @ coeffs
    mse = np.mean((y - y_pred) ** 2)
    
    return coeffs, mse, cond, y_pred


def legendre_polynomial(x, n):
    if n == 0:
        return np.ones_like(x, dtype=float)
    elif n == 1:
        return np.asarray(x, dtype=float)
    else:
        P_prev = np.ones_like(x, dtype=float)
        P_curr = np.asarray(x, dtype=float)
        for k in range(2, n + 1):
            P_next = ((2 * k - 1) * x * P_curr - (k - 1) * P_prev) / k
            P_prev, P_curr = P_curr, P_next
        return P_curr


def polyfit_legendre(x, y, degree):
    check_degree(degree)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    
    x_min, x_max = np.min(x), np.max(x)
    if x_max > x_min:
        x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
    else:
        x_normalized = np.zeros_like(x)
    
    X_legendre = np.column_stack([
        legendre_polynomial(x_normalized, k) for k in range(degree + 1)
    ])
    
    cond = compute_condition_number(X_legendre)
    
    coeffs_legendre, _, _, _ = np.linalg.lstsq(X_legendre, y, rcond=None)
    
    n_fine = max(2000, degree * 100 + 1)
    t_fine = np.linspace(-1, 1, n_fine)
    
    X_fine_legendre = np.column_stack([
        legendre_polynomial(t_fine, k) for k in range(degree + 1)
    ])
    y_fine = X_fine_legendre @ coeffs_legendre
    
    if x_max > x_min:
        scale = (x_max - x_min) / 2.0
        shift = (x_max + x_min) / 2.0
        x_fine_orig = t_fine * scale + shift
        coeffs = np.polyfit(x_fine_orig, y_fine, degree)
    else:
        coeffs = np.polyfit(t_fine, y_fine, degree)
    
    y_pred = np.polyval(coeffs, x)
    mse = np.mean((y - y_pred) ** 2)
    
    return coeffs, mse, cond, y_pred


def polyfit_custom(x, y, basis_functions):
    """
    使用自定义基函数进行最小二乘拟合
    
    参数:
        x: 自变量数组
        y: 因变量数组
        basis_functions: 基函数列表，支持字符串标识或可调用对象
                         字符串标识: '1', 'x', 'x^2', 'x^3', 'exp', 'log', 'sin', 'cos', 'sqrt', '1/x'
                         可调用对象: 接受numpy数组，返回等形状数组
    
    返回:
        dict: {
            'coeffs': 各基函数对应的系数,
            'basis_names': 基函数名称列表,
            'mse': 均方误差,
            'cond': 设计矩阵条件数,
            'y_pred': 预测值数组
        }
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    X, names = build_design_matrix(x, basis_functions)
    cond = compute_condition_number(X)
    check_condition_number(cond, "自定义基函数")
    
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    y_pred = X @ coeffs
    mse = np.mean((y - y_pred) ** 2)
    
    return {
        'coeffs': coeffs,
        'basis_names': names,
        'mse': mse,
        'cond': cond,
        'y_pred': y_pred,
    }


def polyfit_predict(coeffs, x_new, method='legendre', x_train=None, degree=None):
    """
    使用拟合结果对新数据点进行预测
    
    参数:
        coeffs: 拟合得到的多项式系数（从高次到低次）
        x_new: 新的自变量值或数组
        method: 拟合方法（仅 'numpy', 'manual', 'legendre' 需要此参数）
        x_train: 训练数据的x值（Legendre方法需要用于归一化）
        degree: 多项式次数（用于验证）
    
    返回:
        y_pred: 预测值数组
    """
    x_new = np.asarray(x_new, dtype=float)
    return np.polyval(coeffs, x_new)


def polyfit_custom_predict(result, x_new):
    """
    使用自定义基函数拟合结果对新数据点进行预测
    
    参数:
        result: polyfit_custom 返回的字典
        x_new: 新的自变量值或数组
    
    返回:
        y_pred: 预测值数组
    """
    x_new = np.asarray(x_new, dtype=float)
    X_new, _ = build_design_matrix(x_new, result['basis_names'])
    return X_new @ result['coeffs']


def polyfit(x, y, degree, method='legendre'):
    """
    多项式最小二乘拟合的统一接口
    
    参数:
        x: 自变量数组
        y: 因变量数组
        degree: 拟合多项式的次数
        method: 'legendre' (推荐), 'numpy', 或 'manual'
    
    返回:
        dict: {
            'coeffs': 多项式系数（从高次到低次）,
            'mse': 均方误差,
            'r_squared': 决定系数 R²,
            'cond': 设计矩阵条件数,
            'y_pred': 拟合预测值,
            'residuals': 残差分析结果字典
        }
    """
    if method == 'numpy':
        coeffs, mse, cond, y_pred = polyfit_numpy(x, y, degree)
    elif method == 'manual':
        coeffs, mse, cond, y_pred = polyfit_manual(x, y, degree)
    elif method == 'legendre':
        coeffs, mse, cond, y_pred = polyfit_legendre(x, y, degree)
    else:
        raise ValueError("method must be 'legendre', 'numpy', or 'manual'")
    
    y = np.asarray(y, dtype=float)
    r_squared = compute_r_squared(y, y_pred)
    residual_info = compute_residual_analysis(y, y_pred)
    
    return {
        'coeffs': coeffs,
        'mse': mse,
        'r_squared': r_squared,
        'cond': cond,
        'y_pred': y_pred,
        'residuals': residual_info,
    }


def print_polynomial(coeffs):
    terms = []
    degree = len(coeffs) - 1
    
    for i, coeff in enumerate(coeffs):
        power = degree - i
        if abs(coeff) < 1e-10:
            continue
        
        coeff_str = f"{coeff:.4f}"
        
        if power == 0:
            terms.append(coeff_str)
        elif power == 1:
            terms.append(f"{coeff_str}x")
        else:
            terms.append(f"{coeff_str}x^{power}")
    
    return " + ".join(terms).replace(" + -", " - ")


def print_custom_equation(coeffs, names):
    terms = []
    for c, n in zip(coeffs, names):
        if abs(c) < 1e-10:
            continue
        terms.append(f"{c:.4f}*{n}")
    return " + ".join(terms).replace(" + -", " - ")


if __name__ == "__main__":
    np.random.seed(42)
    x = np.linspace(0.1, 5, 50)
    y_true = 0.5 * x**3 - 2 * x**2 + 3 * x + 1
    y = y_true + np.random.normal(0, 0.5, size=len(x))
    degree = 3
    
    print("=" * 70)
    print("一、多项式拟合测试")
    print(f"数据: 50个带噪声的三次函数采样点，拟合次数: {degree}")
    print("=" * 70)
    
    for method in ['numpy', 'manual', 'legendre']:
        print(f"\n[{method}]")
        result = polyfit(x, y, degree, method=method)
        print(f"  多项式: {print_polynomial(result['coeffs'])}")
        print(f"  MSE:    {result['mse']:.6f}")
        print(f"  R²:     {result['r_squared']:.6f}")
        print(f"  条件数: {result['cond']:.2e}")
        res = result['residuals']
        print(f"  残差均值: {res['residual_mean']:.6e}")
        print(f"  残差标准差: {res['residual_std']:.6f}")
        print(f"  Durbin-Watson: {res['durbin_watson']:.4f}")
        print(f"  自相关(lag1-5): {res['autocorrelation'][1:6].round(4)}")
    
    print("\n" + "=" * 70)
    print("二、预测功能测试")
    print("=" * 70)
    result = polyfit(x, y, degree, method='legendre')
    x_test = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y_test_pred = polyfit_predict(result['coeffs'], x_test)
    print(f"  新数据点: {x_test}")
    print(f"  预测值:   {y_test_pred.round(4)}")
    
    print("\n" + "=" * 70)
    print("三、自定义基函数拟合测试")
    print("=" * 70)
    
    print("\n1) 线性+对数+指数基函数: y = a + b*x + c*log(x) + d*exp(x)")
    result_custom = polyfit_custom(x, y, ['1', 'x', 'log', 'exp'])
    print(f"  方程: {print_custom_equation(result_custom['coeffs'], result_custom['basis_names'])}")
    print(f"  MSE:    {result_custom['mse']:.6f}")
    print(f"  条件数: {result_custom['cond']:.2e}")
    r2_custom = compute_r_squared(y, result_custom['y_pred'])
    print(f"  R²:     {r2_custom:.6f}")
    
    x_test2 = np.array([1.0, 2.0, 3.0])
    y_test_pred2 = polyfit_custom_predict(result_custom, x_test2)
    print(f"  预测 x={x_test2} -> y={y_test_pred2.round(4)}")
    
    print("\n2) 幂+正弦+余弦基函数: y = a + b*x + c*x² + d*sin(x) + e*cos(x)")
    result_custom2 = polyfit_custom(x, y, ['1', 'x', 'x^2', 'sin', 'cos'])
    print(f"  方程: {print_custom_equation(result_custom2['coeffs'], result_custom2['basis_names'])}")
    print(f"  MSE:    {result_custom2['mse']:.6f}")
    print(f"  条件数: {result_custom2['cond']:.2e}")
    r2_custom2 = compute_r_squared(y, result_custom2['y_pred'])
    print(f"  R²:     {r2_custom2:.6f}")
    
    print("\n3) 自定义Lambda基函数: y = a + b*sqrt(x) + c/x")
    result_custom3 = polyfit_custom(x, y, ['1', 'sqrt', '1/x'])
    print(f"  方程: {print_custom_equation(result_custom3['coeffs'], result_custom3['basis_names'])}")
    print(f"  MSE:    {result_custom3['mse']:.6f}")
    r2_custom3 = compute_r_squared(y, result_custom3['y_pred'])
    print(f"  R²:     {r2_custom3:.6f}")
    
    print("\n4) 使用callable自定义基函数: y = a + b*x + c*sin(2*pi*x)")
    result_custom4 = polyfit_custom(x, y, [
        lambda t: np.ones_like(t),
        lambda t: t,
        lambda t: np.sin(2 * np.pi * t),
    ])
    print(f"  方程: {print_custom_equation(result_custom4['coeffs'], result_custom4['basis_names'])}")
    print(f"  MSE:    {result_custom4['mse']:.6f}")
    r2_custom4 = compute_r_squared(y, result_custom4['y_pred'])
    print(f"  R²:     {r2_custom4:.6f}")
    
    print("\n" + "=" * 70)
    print("四、残差分析详细示例")
    print("=" * 70)
    result = polyfit(x, y, 3, method='legendre')
    res = result['residuals']
    print(f"  残差序列 (前10个): {res['residuals'][:10].round(4)}")
    print(f"  残差均值:  {res['residual_mean']:.6e} (应接近0)")
    print(f"  残差标准差: {res['residual_std']:.6f}")
    print(f"  Durbin-Watson: {res['durbin_watson']:.4f} (接近2表示无自相关)")
    print(f"  自相关系数:")
    for lag in range(1, len(res['autocorrelation'])):
        print(f"    lag {lag}: {res['autocorrelation'][lag]:.4f}")
    
    print("\n测试完成！")
