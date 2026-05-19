import numpy as np


def _bspline_basis(x, knots, degree=3):
    """
    生成B样条基函数矩阵
    """
    x = np.asarray(x)
    knots = np.asarray(knots)
    n = len(x)
    n_knots = len(knots)
    n_basis = n_knots - degree - 1
    
    basis = np.zeros((n, n_basis))
    
    for i in range(n_basis):
        for j in range(n):
            basis[j, i] = _bspline_eval(x[j], knots, i, degree)
    
    return basis


def _bspline_eval(x, knots, i, k):
    """
    递归计算第i个k次B样条在x处的值
    """
    if k == 0:
        return 1.0 if knots[i] <= x < knots[i + 1] else 0.0
    
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]
    
    term1 = 0.0
    if denom1 != 0:
        term1 = (x - knots[i]) / denom1 * _bspline_eval(x, knots, i, k - 1)
    
    term2 = 0.0
    if denom2 != 0:
        term2 = (knots[i + k + 1] - x) / denom2 * _bspline_eval(x, knots, i + 1, k - 1)
    
    return term1 + term2


def _compute_penalty_matrix(n_basis, degree=3):
    """
    计算二阶导数惩罚矩阵
    """
    D = np.zeros((n_basis - 2, n_basis))
    for i in range(n_basis - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
    return D.T @ D


def smoothing_spline(x, y, lam=None, n_knots=None, degree=3):
    """
    样条平滑
    
    参数:
        x: 数组，x坐标数据点
        y: 数组，y坐标数据点
        lam: float，平滑参数，None则用GCV自动选择
        n_knots: int，节点数量，None则设为数据点数量
        degree: int，样条次数，默认3次（三次样条）
    
    返回:
        coeffs: 样条系数
        knots: 节点
        lam: 使用的平滑参数
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    n = len(x)
    if n != len(y):
        raise ValueError("x和y的长度必须相同")
    
    if n_knots is None:
        n_knots = min(n, 50)
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    knots = np.linspace(x_sorted[0], x_sorted[-1], n_knots)
    knots = np.concatenate([
        np.full(degree, x_sorted[0]),
        knots,
        np.full(degree, x_sorted[-1])
    ])
    
    B = _bspline_basis(x_sorted, knots, degree)
    n_basis = B.shape[1]
    P = _compute_penalty_matrix(n_basis, degree)
    
    if lam is None:
        lam = _gcv_search(B, y_sorted, P)
    
    coeffs = np.linalg.solve(B.T @ B + lam * P, B.T @ y_sorted)
    
    return coeffs, knots, lam


def _gcv_score(B, y, P, lam):
    """
    计算GCV得分
    """
    n = len(y)
    BtB = B.T @ B
    M = BtB + lam * P
    invM = np.linalg.inv(M)
    H = B @ invM @ B.T
    
    y_hat = H @ y
    residuals = y - y_hat
    rss = np.sum(residuals ** 2)
    
    trace_H = np.trace(H)
    gcv = rss / (n * (1 - trace_H / n) ** 2)
    
    return gcv


def _gcv_search(B, y, P, n_lambdas=100):
    """
    搜索最佳平滑参数lambda
    """
    lambdas = np.logspace(-10, 10, n_lambdas)
    scores = np.zeros(n_lambdas)
    
    for i, lam in enumerate(lambdas):
        scores[i] = _gcv_score(B, y, P, lam)
    
    best_idx = np.argmin(scores)
    return lambdas[best_idx]


def evaluate_spline(x_eval, coeffs, knots, degree=3):
    """
    在给定点计算样条值
    """
    B = _bspline_basis(x_eval, knots, degree)
    return B @ coeffs


def least_squares_polyfit(x, y, deg):
    """
    最小二乘多项式拟合
    
    参数:
        x: 数组，x坐标数据点
        y: 数组，y坐标数据点
        deg: int，多项式次数
    
    返回:
        数组，多项式系数，从最高次到常数项
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    n = len(x)
    if n != len(y):
        raise ValueError("x和y的长度必须相同")
    
    if deg < 0:
        raise ValueError("多项式次数不能为负数")
    
    A = np.vander(x, deg + 1)
    
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    
    return coeffs


if __name__ == "__main__":
    x = [0, 1, 2, 3, 4]
    y = [1, 3, 5, 7, 9]
    deg = 1
    
    coeffs = least_squares_polyfit(x, y, deg)
    print(f"多项式系数: {coeffs}")
    print(f"拟合多项式: y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}")
    
    x2 = [0, 1, 2, 3, 4]
    y2 = [0, 1, 4, 9, 16]
    deg2 = 2
    
    coeffs2 = least_squares_polyfit(x2, y2, deg2)
    print(f"\n二次多项式系数: {coeffs2}")
    print(f"拟合多项式: y = {coeffs2[0]:.4f}x^2 + {coeffs2[1]:.4f}x + {coeffs2[2]:.4f}")
    
    x3 = [0, 1]
    y3 = [1, 3]
    deg3 = 3
    
    coeffs3 = least_squares_polyfit(x3, y3, deg3)
    print(f"\n欠定系统(2个点, 3次多项式)系数: {coeffs3}")
    print(f"系数范数: {np.linalg.norm(coeffs3):.4f}")
    
    np.random.seed(42)
    x4 = np.linspace(0, 10, 50)
    y4_true = np.sin(x4) + 0.5 * x4
    y4 = y4_true + np.random.normal(0, 0.2, 50)
    
    coeffs_spline, knots_spline, lam_best = smoothing_spline(x4, y4)
    y_smooth = evaluate_spline(x4, coeffs_spline, knots_spline)
    
    print(f"\n样条平滑测试:")
    print(f"GCV选择的最佳平滑参数: {lam_best:.2e}")
    print(f"原始数据RMSE: {np.sqrt(np.mean((y4 - y4_true)**2)):.4f}")
    print(f"平滑后RMSE: {np.sqrt(np.mean((y_smooth - y4_true)**2)):.4f}")
