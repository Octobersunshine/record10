import numpy as np


def autocorrelation(series, k):
    n = len(series)
    mean = np.mean(series)
    numerator = np.sum((series[:n - k] - mean) * (series[k:] - mean))
    denominator = np.sum((series - mean) ** 2)
    return numerator / denominator


def autocovariance(series, k):
    n = len(series)
    mean = np.mean(series)
    return np.sum((series[:n - k] - mean) * (series[k:] - mean)) / n


def levinson_durbin(series, p, eps=1e-10):
    r = np.array([autocovariance(series, k) for k in range(p + 1)])
    r[0] += eps

    a = np.zeros(p + 1)
    a[0] = 1.0
    e = r[0]

    for k in range(1, p + 1):
        ak = -np.sum(a[:k] * r[k:0:-1]) / e
        a[1:k + 1] += ak * a[k - 1::-1]
        a[k] = ak
        e *= (1 - ak * ak)

    return -a[1:], e


def predict_next(series, ar_coeffs):
    p = len(ar_coeffs)
    mean = np.mean(series)
    series_centered = series - mean

    last_p = series_centered[-p:][::-1]
    pred_centered = np.sum(ar_coeffs * last_p)
    return pred_centered + mean


def ar_model(series, p, eps=1e-10):
    ar_coeffs, var = levinson_durbin(series, p, eps=eps)
    next_pred = predict_next(series, ar_coeffs)
    return ar_coeffs, var, next_pred


def ma_model(series, q, eps=1e-10, n_iter=100):
    n = len(series)
    mean = np.mean(series)
    series_centered = series - mean

    ar_order = max(q, int(np.log(n)))
    ar_coeffs, _ = levinson_durbin(series, ar_order, eps=eps)

    residuals = np.zeros(n)
    for t in range(ar_order, n):
        pred = np.sum(ar_coeffs * series_centered[t - ar_order:t][::-1])
        residuals[t] = series_centered[t] - pred

    for _ in range(n_iter):
        X = np.zeros((n - q, q))
        y = series_centered[q:]

        for t in range(q, n):
            X[t - q, :] = residuals[t - q:t][::-1]

        theta = np.linalg.lstsq(X, y, rcond=None)[0]

        new_residuals = np.zeros(n)
        for t in range(q, n):
            pred = np.sum(theta * residuals[t - q:t][::-1])
            new_residuals[t] = series_centered[t] - pred

        if np.max(np.abs(new_residuals - residuals)) < 1e-8:
            break
        residuals = new_residuals

    var = np.var(residuals[q:])

    last_q = residuals[-q:][::-1]
    pred_centered = np.sum(theta * last_q)
    next_pred = pred_centered + mean

    return theta, var, next_pred


def arma_model(series, p, q, eps=1e-10, n_iter=50):
    n = len(series)
    mean = np.mean(series)
    series_centered = series - mean

    ar_coeffs = np.zeros(p)
    ma_coeffs = np.zeros(q)
    residuals = np.zeros(n)

    for _ in range(n_iter):
        X = np.zeros((n - max(p, q), p + q))
        y = series_centered[max(p, q):]

        for t in range(max(p, q), n):
            X[t - max(p, q), :p] = series_centered[t - p:t][::-1]
            X[t - max(p, q), p:] = residuals[t - q:t][::-1]

        params = np.linalg.lstsq(X, y, rcond=None)[0]
        ar_coeffs = params[:p]
        ma_coeffs = params[p:]

        new_residuals = np.zeros(n)
        for t in range(max(p, q), n):
            ar_part = np.sum(ar_coeffs * series_centered[t - p:t][::-1])
            ma_part = np.sum(ma_coeffs * residuals[t - q:t][::-1])
            new_residuals[t] = series_centered[t] - ar_part - ma_part

        if np.max(np.abs(new_residuals - residuals)) < 1e-8:
            break
        residuals = new_residuals

    var = np.var(residuals[max(p, q):])

    ar_part = np.sum(ar_coeffs * series_centered[-p:][::-1])
    ma_part = np.sum(ma_coeffs * residuals[-q:][::-1])
    pred_centered = ar_part + ma_part
    next_pred = pred_centered + mean

    return ar_coeffs, ma_coeffs, var, next_pred


def aic(n_params, var, n):
    return n * np.log(var) + 2 * n_params


def bic(n_params, var, n):
    return n * np.log(var) + n_params * np.log(n)


def auto_order_selection(series, max_p, max_q=None, criterion='aic', model_type='ar', eps=1e-10):
    if max_q is None:
        max_q = max_p

    n = len(series)
    orders = []
    criterion_values = []

    if model_type == 'ar':
        for p in range(1, max_p + 1):
            _, var, _ = ar_model(series, p, eps=eps)
            ic = aic(p, var, n) if criterion == 'aic' else bic(p, var, n)
            orders.append((p, 0))
            criterion_values.append(ic)
    elif model_type == 'ma':
        for q in range(1, max_q + 1):
            _, var, _ = ma_model(series, q, eps=eps)
            ic = aic(q, var, n) if criterion == 'aic' else bic(q, var, n)
            orders.append((0, q))
            criterion_values.append(ic)
    elif model_type == 'arma':
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                if p == 0 and q == 0:
                    continue
                if p == 0:
                    _, var, _ = ma_model(series, q, eps=eps)
                    n_params = q
                elif q == 0:
                    _, var, _ = ar_model(series, p, eps=eps)
                    n_params = p
                else:
                    _, _, var, _ = arma_model(series, p, q, eps=eps)
                    n_params = p + q
                ic = aic(n_params, var, n) if criterion == 'aic' else bic(n_params, var, n)
                orders.append((p, q))
                criterion_values.append(ic)

    best_idx = np.argmin(criterion_values)
    best_order = orders[best_idx]

    return {
        'best_order': best_order,
        'orders': orders,
        'criterion_values': criterion_values,
        'criterion': criterion
    }


if __name__ == "__main__":
    np.random.seed(42)
    n = 200

    print("=" * 60)
    print("1. AR模型测试 (真实系数: [0.6, -0.2])")
    print("=" * 60)
    ar_params = np.array([0.6, -0.2])
    p_true = len(ar_params)
    ar_series = np.zeros(n)
    for t in range(p_true, n):
        ar_series[t] = np.sum(ar_params * ar_series[t - p_true:t][::-1]) + np.random.normal(0, 1)

    ar_coeffs, ar_var, ar_pred = ar_model(ar_series, p=2)
    print("估计AR系数:", ar_coeffs)
    print("残差方差:", ar_var)
    print("未来一步预测值:", ar_pred)

    print("\n=== AR模型自动定阶 ===")
    result_ar = auto_order_selection(ar_series, max_p=6, criterion='aic', model_type='ar')
    print(f"推荐阶数 (AIC): p={result_ar['best_order'][0]}")
    for (p, q), ic in zip(result_ar['orders'], result_ar['criterion_values']):
        print(f"  AR({p}): {result_ar['criterion']} = {ic:.2f}")

    print("\n" + "=" * 60)
    print("2. MA模型测试 (真实系数: [0.4, 0.2])")
    print("=" * 60)
    ma_params = np.array([0.4, 0.2])
    q_true = len(ma_params)
    errors = np.random.normal(0, 1, n)
    ma_series = np.zeros(n)
    for t in range(q_true, n):
        ma_series[t] = errors[t] + np.sum(ma_params * errors[t - q_true:t][::-1])

    ma_coeffs, ma_var, ma_pred = ma_model(ma_series, q=2)
    print("估计MA系数:", ma_coeffs)
    print("残差方差:", ma_var)
    print("未来一步预测值:", ma_pred)

    print("\n=== MA模型自动定阶 ===")
    result_ma = auto_order_selection(ma_series, max_p=4, criterion='bic', model_type='ma')
    print(f"推荐阶数 (BIC): q={result_ma['best_order'][1]}")
    for (p, q), ic in zip(result_ma['orders'], result_ma['criterion_values']):
        print(f"  MA({q}): {result_ma['criterion']} = {ic:.2f}")

    print("\n" + "=" * 60)
    print("3. ARMA模型测试 (真实AR: [0.5], 真实MA: [0.3])")
    print("=" * 60)
    arma_ar = np.array([0.5])
    arma_ma = np.array([0.3])
    p_arma = len(arma_ar)
    q_arma = len(arma_ma)
    errors_arma = np.random.normal(0, 1, n)
    arma_series = np.zeros(n)
    for t in range(max(p_arma, q_arma), n):
        ar_part = np.sum(arma_ar * arma_series[t - p_arma:t][::-1])
        ma_part = np.sum(arma_ma * errors_arma[t - q_arma:t][::-1])
        arma_series[t] = ar_part + errors_arma[t] + ma_part

    est_ar, est_ma, arma_var, arma_pred = arma_model(arma_series, p=1, q=1)
    print("估计AR系数:", est_ar)
    print("估计MA系数:", est_ma)
    print("残差方差:", arma_var)
    print("未来一步预测值:", arma_pred)

    print("\n=== ARMA模型自动定阶 (max_p=3, max_q=3) ===")
    result_arma = auto_order_selection(arma_series, max_p=3, max_q=3, criterion='aic', model_type='arma')
    print(f"推荐阶数 (AIC): (p={result_arma['best_order'][0]}, q={result_arma['best_order'][1]})")

    print("\n" + "=" * 60)
    print("4. 数值稳定性测试 (高阶p)")
    print("=" * 60)
    try:
        p_high = 20
        coeffs_high, var_high, pred_high = ar_model(ar_series, p=p_high)
        print(f"AR({p_high}) 拟合成功!")
        print(f"残差方差: {var_high:.6f}")
        print(f"系数范数: {np.linalg.norm(coeffs_high):.6f}")
    except Exception as e:
        print(f"数值不稳定，拟合失败: {e}")

    print("\n=== 正则化参数测试 ===")
    for eps in [0, 1e-15, 1e-10, 1e-5]:
        coeffs, var, _ = ar_model(ar_series, p=5, eps=eps)
        print(f"eps={eps}: 系数范数={np.linalg.norm(coeffs):.6f}, 残差方差={var:.6f}")
