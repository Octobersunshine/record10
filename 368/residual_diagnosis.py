import warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge



def compute_vif(X):
    n, k = X.shape
    vif_dict = {}
    r_squared_list = []

    if k == 1:
        return {0: 1.0}, [0.0], "无共线性（仅一个特征）"

    for j in range(k):
        X_j = X[:, j]
        X_others = np.delete(X, j, axis=1)

        if n <= X_others.shape[1] + 1:
            vif_dict[j] = np.inf
            r_squared_list.append(1.0)
            continue

        model_aux = LinearRegression(fit_intercept=True)
        model_aux.fit(X_others, X_j)
        X_j_pred = model_aux.predict(X_others)
        ss_res = np.sum((X_j - X_j_pred) ** 2)
        ss_tot = np.sum((X_j - np.mean(X_j)) ** 2)
        r_squared = 1 - ss_res / max(ss_tot, 1e-20)
        r_squared = np.clip(r_squared, 0, 1 - 1e-15)
        r_squared_list.append(r_squared)
        vif_dict[j] = 1.0 / (1.0 - r_squared)

    max_vif = max(vif_dict.values())
    high_vif_features = [j for j, v in vif_dict.items() if v >= 10]
    moderate_vif_features = [j for j, v in vif_dict.items() if 5 <= v < 10]

    if max_vif < 5:
        collinearity_status = "无明显多重共线性（所有VIF < 5）"
    elif max_vif < 10:
        collinearity_status = f"存在中等程度多重共线性（最大VIF = {max_vif:.2f}）"
    else:
        collinearity_status = f"存在严重多重共线性（最大VIF = {max_vif:.2f}）"

    return vif_dict, r_squared_list, collinearity_status


def _generate_collinearity_advice(vif_dict):
    k = len(vif_dict)
    high_vif = sorted(vif_dict.items(), key=lambda x: -x[1])

    advice = []
    if k == 1:
        return advice

    severe = [(j, v) for j, v in high_vif if v >= 10]
    moderate = [(j, v) for j, v in high_vif if 5 <= v < 10]

    if severe:
        feats = ", ".join([f"X{j}(VIF={v:.2f})" for j, v in severe])
        advice.append(f"[严重] 特征 {feats} 存在严重多重共线性，建议优先考虑：")
        advice.append(f"  1. 剔除VIF最高的特征 {severe[0][0]}；")
        advice.append(f"  2. 或使用正则化回归（岭回归、Lasso）；")
        advice.append(f"  3. 或对高相关特征进行降维（PCA、因子分析）。")

    if moderate:
        feats = ", ".join([f"X{j}(VIF={v:.2f})" for j, v in moderate])
        advice.append(f"[警告] 特征 {feats} 存在中等程度多重共线性，建议关注：")
        advice.append(f"  1. 若回归系数符号与理论预期相反，可能是共线性所致；")
        advice.append(f"  2. 可考虑增加样本量或使用正则化回归。")

    if not severe and not moderate:
        advice.append("[信息] 所有特征VIF < 5，多重共线性不构成主要问题。")

    return advice


def _diagnose_dof(n, p):
    messages = []
    if n <= p:
        messages.append(
            f"[严重] 样本量(n={n}) <= 参数量(p={p}), OLS自由度不足, "
            f"残差方差无法估计, 已自动切换为岭回归。"
        )
        messages.append(
            f"  建议: 增加样本量至少到 n > {p}, 或减少特征数。"
        )
    elif n < 2 * p:
        messages.append(
            f"[警告] 样本量(n={n})相对参数量(p={p})偏少, "
            f"残差估计可能不稳定, 建议样本量至少为参数量的2倍以上。"
        )
    if n <= 3:
        messages.append(
            f"[警告] 样本量极少(n={n}), 诊断指标可靠性很低, "
            f"仅供参考, 不宜作为决策依据。"
        )
    return messages


def _compute_ridge_hat_matrix(X_with_intercept, alpha):
    n, p = X_with_intercept.shape
    penalty_matrix = np.eye(p)
    penalty_matrix[0, 0] = 0.0
    A = X_with_intercept.T @ X_with_intercept + alpha * penalty_matrix
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
        warnings.warn("矩阵求逆失败, 已改用伪逆, 诊断结果可能不可靠。")
    H = X_with_intercept @ A_inv @ X_with_intercept.T
    return H, A_inv


def residual_diagnosis(X, y, alpha=1.0, force_ridge=False):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, k = X.shape
    p = k + 1

    diagnostic_messages = _diagnose_dof(n, p)

    vif_dict, r_squared_list, collinearity_status = compute_vif(X)
    collinearity_advice = _generate_collinearity_advice(vif_dict)
    diagnostic_messages.extend(collinearity_advice)

    use_ridge = force_ridge or (n <= p)

    X_with_intercept = np.column_stack([np.ones(n), X])

    if use_ridge:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_with_intercept, y)
        y_pred = model.predict(X_with_intercept)

        H, _ = _compute_ridge_hat_matrix(X_with_intercept, alpha)
        h_ii = np.diag(H)

        effective_df = np.trace(H)
        residuals = y - y_pred
        MSE = np.sum(residuals ** 2) / max(n - effective_df, 1e-10)

        diagnostic_messages.append(
            f"[信息] 岭回归 alpha={alpha}, 有效自由度={effective_df:.4f}, "
            f"残差自由度={n - effective_df:.4f}。"
        )
        if n <= p:
            diagnostic_messages.append(
                f"[信息] 岭回归通过正则化约束使矩阵可逆, "
                f"但残差诊断的含义与OLS不同, 结果仅供参考。"
            )
    else:
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        y_pred = model.predict(X)

        residuals = y - y_pred

        try:
            H = X_with_intercept @ np.linalg.inv(
                X_with_intercept.T @ X_with_intercept
            ) @ X_with_intercept.T
        except np.linalg.LinAlgError:
            warnings.warn("X'X矩阵奇异, 自动切换为岭回归。")
            H, _ = _compute_ridge_hat_matrix(X_with_intercept, alpha)
            use_ridge = True

        h_ii = np.diag(H)
        MSE = np.sum(residuals ** 2) / (n - p)

    standardized_residuals = residuals / np.sqrt(max(MSE, 1e-20))

    h_clamped = np.clip(h_ii, 0, 1 - 1e-10)
    internally_studentized = residuals / np.sqrt(max(MSE, 1e-20) * (1 - h_clamped))

    externally_studentized = np.zeros(n)
    for i in range(n):
        X_loo = np.delete(X_with_intercept, i, axis=0)
        y_loo = np.delete(y, i)
        n_loo = n - 1
        loo_dof_sufficient = (not use_ridge) and (n_loo - p > 0)

        if loo_dof_sufficient:
            y_pred_loo = np.delete(y_pred, i)
            residuals_loo = y_loo - y_pred_loo
            MSE_loo = np.sum(residuals_loo ** 2) / (n_loo - p)
        else:
            ridge_loo = Ridge(alpha=alpha, fit_intercept=False)
            ridge_loo.fit(X_loo, y_loo)
            y_pred_loo = ridge_loo.predict(X_loo)
            residuals_loo = y_loo - y_pred_loo
            H_loo, _ = _compute_ridge_hat_matrix(X_loo, alpha)
            effective_df_loo = np.trace(H_loo)
            MSE_loo = np.sum(residuals_loo ** 2) / max(n_loo - effective_df_loo, 1e-10)

        externally_studentized[i] = residuals[i] / np.sqrt(
            max(MSE_loo, 1e-20) * (1 - h_clamped[i])
        )

    cooks_distance = (internally_studentized ** 2 / p) * (h_clamped / (1 - h_clamped))

    leverage_threshold = 2 * p / n
    outliers = np.where(cooks_distance > 1)[0].tolist()
    high_leverage_points = np.where(h_ii > leverage_threshold)[0].tolist()

    for msg in diagnostic_messages:
        print(msg)

    return {
        'residuals': residuals,
        'standardized_residuals': standardized_residuals,
        'internally_studentized_residuals': internally_studentized,
        'externally_studentized_residuals': externally_studentized,
        'cooks_distance': cooks_distance,
        'leverage_values': h_ii,
        'outliers': outliers,
        'high_leverage_points': high_leverage_points,
        'used_ridge': use_ridge,
        'diagnostic_messages': diagnostic_messages,
        'vif': vif_dict,
        'r_squared_vif': r_squared_list,
        'collinearity_status': collinearity_status,
        'collinearity_advice': collinearity_advice
    }


def print_diagnosis_report(results, k):
    n = len(results['residuals'])
    p = k + 1

    print("\n残差诊断结果:")
    print("=" * 90)
    header = (
        f"{'样本':>6} {'残差':>10} {'标准残差':>10} "
        f"{'内学生化':>10} {'外学生化':>10} {'Cook距离':>10} {'杠杆值':>10}"
    )
    print(header)
    print("-" * 90)
    for i in range(n):
        print(
            f"{i:>6} {results['residuals'][i]:>10.4f} "
            f"{results['standardized_residuals'][i]:>10.4f} "
            f"{results['internally_studentized_residuals'][i]:>10.4f} "
            f"{results['externally_studentized_residuals'][i]:>10.4f} "
            f"{results['cooks_distance'][i]:>10.4f} "
            f"{results['leverage_values'][i]:>10.4f}"
        )

    print("\n" + "=" * 90)
    print("多重共线性诊断 (VIF):")
    print("-" * 90)
    print(f"{'特征':>8} {'R²':>10} {'VIF':>12} {'状态':>20}")
    print("-" * 90)
    for j in range(k):
        vif = results['vif'][j]
        r2 = results['r_squared_vif'][j]
        if vif == np.inf:
            vif_str = "inf"
        else:
            vif_str = f"{vif:.4f}"
        if vif >= 10:
            status = "严重共线性"
        elif vif >= 5:
            status = "中等共线性"
        else:
            status = "正常"
        print(f"{f'X{j}':>8} {r2:>10.4f} {vif_str:>12} {status:>20}")

    print("\n" + "=" * 90)
    print(f"共线性状态: {results['collinearity_status']}")
    print(f"回归方式: {'岭回归' if results['used_ridge'] else 'OLS'}")
    print(f"异常点索引 (Cook距离 > 1): {results['outliers']}")
    print(f"高杠杆点索引 (h_ii > 2p/n): {results['high_leverage_points']}")
    print(f"杠杆阈值: {2 * p / n:.4f}")

    if results['outliers']:
        print(f"\n诊断建议: 发现 {len(results['outliers'])} 个异常点, "
              f"建议检查这些样本是否存在数据录入错误或特殊背景;")
    if results['high_leverage_points']:
        print(f"诊断建议: 发现 {len(results['high_leverage_points'])} 个高杠杆点, "
              f"这些样本在自变量空间中位置极端, 可能对回归结果产生过度影响;")
    if results['used_ridge']:
        print("诊断建议: 当前使用岭回归, 诊断指标的含义与OLS不完全一致, "
              "建议优先增加样本量以使用标准OLS诊断。")
    if results['collinearity_advice']:
        for advice in results['collinearity_advice']:
            if advice.startswith("[信息]"):
                continue
            print(f"诊断建议: {advice}")


if __name__ == "__main__":
    print("=" * 60)
    print("场景1: 正常情况 (样本量充足)")
    print("=" * 60)
    np.random.seed(42)
    n, k = 30, 2
    X = np.random.randn(n, k)
    X_int = np.column_stack([np.ones(n), X])
    true_beta = np.array([3.0, 1.5, -2.0])
    y = X_int @ true_beta + np.random.randn(n) * 0.5
    X[5] = [10.0, 10.0]
    y[5] = 100.0
    X[20] = [-8.0, 8.0]
    y[20] = -50.0
    results1 = residual_diagnosis(X, y)
    print_diagnosis_report(results1, k)

    print("\n\n" + "=" * 60)
    print("场景2: 样本量不足 (n <= p)")
    print("=" * 60)
    np.random.seed(7)
    n2, k2 = 3, 5
    X2 = np.random.randn(n2, k2)
    y2 = np.random.randn(n2)
    results2 = residual_diagnosis(X2, y2)
    print_diagnosis_report(results2, k2)

    print("\n\n" + "=" * 60)
    print("场景3: 样本量偏少但刚好够用 (n 略大于 p)")
    print("=" * 60)
    np.random.seed(10)
    n3, k3 = 6, 4
    X3 = np.random.randn(n3, k3)
    X3_int = np.column_stack([np.ones(n3), X3])
    y3 = X3_int @ np.random.randn(k3 + 1) + np.random.randn(n3) * 0.3
    results3 = residual_diagnosis(X3, y3)
    print_diagnosis_report(results3, k3)

    print("\n\n" + "=" * 60)
    print("场景4: 存在多重共线性")
    print("=" * 60)
    np.random.seed(15)
    n4, k4 = 40, 4
    X4 = np.random.randn(n4, k4)
    X4[:, 2] = 0.9 * X4[:, 0] + 0.1 * np.random.randn(n4)
    X4[:, 3] = -0.8 * X4[:, 1] + 0.15 * np.random.randn(n4)
    X4_int = np.column_stack([np.ones(n4), X4])
    y4 = X4_int @ np.array([2.0, 1.0, -0.5, 2.5, -1.5]) + np.random.randn(n4) * 0.3
    results4 = residual_diagnosis(X4, y4)
    print_diagnosis_report(results4, k4)
