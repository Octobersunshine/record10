import warnings

import numpy as np


def cmds(distance_matrix, n_components=2):
    """
    Classical Multidimensional Scaling (Classical MDS)

    当距离矩阵不满足三角不等式时，双中心化矩阵B可能出现负特征值，
    此时会发出警告，并将负特征值截断为0。

    Parameters:
    -----------
    distance_matrix : numpy.ndarray, shape (n_samples, n_samples)
        距离矩阵，必须是对称的
    n_components : int, default=2
        降维后的维度数（1-10）

    Returns:
    --------
    coords : numpy.ndarray, shape (n_samples, n_components)
        低维空间中的坐标
    eigenvalues : numpy.ndarray, shape (n_components,)
        前n_components个最大的特征值（负值已截断为0）
    """
    _validate_n_components(n_components)

    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]

    if D.shape[1] != n:
        raise ValueError("距离矩阵必须是方阵")

    if not np.allclose(D, D.T):
        raise ValueError("距离矩阵必须是对称的")

    J = np.eye(n) - np.ones((n, n)) / n

    D_squared = D ** 2

    B = -0.5 * J @ D_squared @ J

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    n_negative = np.sum(eigenvalues < -1e-10)
    if n_negative > 0:
        neg_vals = eigenvalues[eigenvalues < -1e-10]
        warnings.warn(
            f"双中心化矩阵B有{n_negative}个显著负特征值"
            f"（最小值: {neg_vals.min():.6f}），"
            f"距离矩阵可能不满足三角不等式。"
            f"负特征值已截断为0，建议使用 metric_mds() 获得更好的结果。",
            RuntimeWarning,
            stacklevel=2,
        )

    eigenvalues_clipped = np.maximum(eigenvalues, 0)

    top_eigenvalues = eigenvalues_clipped[:n_components]
    top_eigenvectors = eigenvectors[:, :n_components]

    coords = top_eigenvectors * np.sqrt(top_eigenvalues)

    return coords, top_eigenvalues


def _validate_n_components(n_components):
    if not isinstance(n_components, (int, np.integer)):
        raise TypeError(f"n_components 必须是整数，得到 {type(n_components).__name__}")
    if n_components < 1 or n_components > 10:
        raise ValueError(f"n_components 必须在1-10之间，得到 {n_components}")


def _compute_pairwise_distances(X):
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _compute_stress(D_target, D_current):
    mask = D_target > 0
    raw_stress = np.sum((D_target[mask] - D_current[mask]) ** 2)
    scale = np.sum(D_target[mask] ** 2)
    if scale == 0:
        return 0.0
    return np.sqrt(raw_stress / scale)


def _isotonic_regression(y, weight=None):
    """
    Pool Adjacent Violators Algorithm (PAVA) 保序回归

    给定 y，找到在单调非降约束下使加权平方误差最小的 x。

    Parameters:
    -----------
    y : numpy.ndarray
        待回归的值（按自变量排序）
    weight : numpy.ndarray or None
        每个点的权重

    Returns:
    --------
    x : numpy.ndarray
        保序回归结果（单调非降）
    """
    n = len(y)
    if weight is None:
        weight = np.ones(n)

    x = y.astype(np.float64).copy()
    w = weight.astype(np.float64).copy()

    blocks = [[i] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        val_left = np.average(x[blocks[i]], weights=w[blocks[i]])
        val_right = np.average(x[blocks[i + 1]], weights=w[blocks[i + 1]])

        if val_left <= val_right:
            i += 1
        else:
            merged = blocks[i] + blocks[i + 1]
            blocks[i] = merged
            del blocks[i + 1]
            avg = np.average(x[merged], weights=w[merged])
            x[merged] = avg
            w[merged] = np.sum(w[merged])
            if i > 0:
                i -= 1

    return x


def _compute_disparities(D_target, D_current):
    """
    计算非度量MDS中的 disparities（差异量）

    提取上三角的原始不相似度和当前距离，
    按原始不相似度排序后进行保序回归，再映射回矩阵形式。
    """
    n = D_target.shape[0]

    rows, cols = np.triu_indices(n, k=1)
    delta = D_target[rows, cols]
    d = D_current[rows, cols]

    order = np.argsort(delta)
    delta_sorted = delta[order]
    d_sorted = d[order]

    dhat_sorted = _isotonic_regression(d_sorted)

    rank_order_inv = np.empty_like(order)
    rank_order_inv[order] = np.arange(len(order))
    dhat = dhat_sorted[rank_order_inv]

    disparities = np.zeros((n, n))
    disparities[rows, cols] = dhat
    disparities[cols, rows] = dhat

    return disparities


def _compute_stress_nonmetric(D_target, D_current, disparities):
    """
    计算非度量MDS的Kruskal Stress-1

    Stress-1 = sqrt( sum((d_ij - dhat_ij)^2) / sum(d_ij^2) )
    """
    mask = D_target > 0
    numerator = np.sum((D_current[mask] - disparities[mask]) ** 2)
    denominator = np.sum(D_current[mask] ** 2)
    if denominator == 0:
        return 0.0
    return np.sqrt(numerator / denominator)


def stress_quality(stress):
    """
    根据Kruskal (1964)的标准评估Stress拟合质量

    Parameters:
    -----------
    stress : float
        Kruskal Stress-1 值

    Returns:
    --------
    quality : str
        拟合质量等级描述

    References:
    -----------
    Kruskal, J. B. (1964). Multidimensional scaling by optimizing
    goodness of fit to a nonmetric hypothesis. Psychometrika, 29(1), 1-27.
    """
    if stress < 0.0:
        return "无效（Stress不应为负值）"
    elif stress < 0.025:
        return "极好（Excellent）"
    elif stress < 0.05:
        return "很好（Good）"
    elif stress < 0.10:
        return "一般（Fair）"
    elif stress < 0.20:
        return "较差（Poor）"
    else:
        return "差（Bad）"


def metric_mds(
    distance_matrix,
    n_components=2,
    n_init=4,
    max_iter=300,
    eps=1e-6,
    learning_rate=0.01,
    random_state=None,
    verbose=False,
):
    """
    Metric MDS（度量MDS），使用梯度下降最小化Kruskal应力函数

    即使距离矩阵不满足三角不等式，也能找到最优的低维表示。

    Parameters:
    -----------
    distance_matrix : numpy.ndarray, shape (n_samples, n_samples)
        距离矩阵，必须是对称的
    n_components : int, default=2
        降维后的维度数（1-10）
    n_init : int, default=4
        随机初始化的次数，取最优结果
    max_iter : int, default=300
        最大迭代次数
    eps : float, default=1e-6
        应力变化小于此值时提前停止
    learning_rate : float, default=0.01
        梯度下降学习率
    random_state : int or None, default=None
        随机种子
    verbose : bool, default=False
        是否打印迭代信息

    Returns:
    --------
    coords : numpy.ndarray, shape (n_samples, n_components)
        低维空间中的坐标
    stress : float
        最终的Kruskal应力值（stress-1）
    """
    _validate_n_components(n_components)

    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]

    if D.shape[1] != n:
        raise ValueError("距离矩阵必须是方阵")

    if not np.allclose(D, D.T):
        raise ValueError("距离矩阵必须是对称的")

    rng = np.random.RandomState(random_state)

    best_coords = None
    best_stress = np.inf

    tri_mask = D > 0

    for init_idx in range(n_init):
        X = rng.randn(n, n_components) * 0.1

        prev_stress = np.inf

        for iteration in range(max_iter):
            D_current = _compute_pairwise_distances(X)

            stress = _compute_stress(D, D_current)

            if verbose and iteration % 50 == 0:
                print(f"  初始化{init_idx + 1}, 迭代{iteration}: stress = {stress:.6f}")

            if abs(prev_stress - stress) < eps:
                if verbose:
                    print(f"  初始化{init_idx + 1}, 迭代{iteration}: 收敛 (stress = {stress:.6f})")
                break

            prev_stress = stress

            grad = np.zeros_like(X)
            for i in range(n):
                for j in range(n):
                    if i == j or not tri_mask[i, j]:
                        continue
                    d_ij = D_current[i, j]
                    if d_ij < 1e-10:
                        continue
                    diff_ij = X[i] - X[j]
                    ratio = (D[i, j] - d_ij) / d_ij
                    grad[i] -= ratio * diff_ij

            grad *= 2.0
            X -= learning_rate * grad

        D_final = _compute_pairwise_distances(X)
        final_stress = _compute_stress(D, D_final)

        if final_stress < best_stress:
            best_stress = final_stress
            best_coords = X.copy()

    return best_coords, best_stress


def nonmetric_mds(
    distance_matrix,
    n_components=2,
    n_init=4,
    max_iter=300,
    eps=1e-6,
    learning_rate=0.01,
    random_state=None,
    verbose=False,
):
    """
    Non-metric MDS（非度量MDS），仅保持距离的排序关系

    使用保序回归（Isotonic Regression / PAVA）将原始不相似度映射为
    单调的 disparities，再通过梯度下降最小化当前距离与 disparities
    之间的Kruskal Stress-1。

    适用于心理测量学等场景，其中原始数据仅为排序量表（ordinal scale），
    距离的绝对数值无意义，只有相对大小关系有意义。

    Parameters:
    -----------
    distance_matrix : numpy.ndarray, shape (n_samples, n_samples)
        不相似度/距离矩阵，必须是对称的，只需满足偏序关系
    n_components : int, default=2
        目标维度（1-10）
    n_init : int, default=4
        随机初始化的次数，取最优结果
    max_iter : int, default=300
        最大迭代次数
    eps : float, default=1e-6
        Stress变化小于此值时提前停止
    learning_rate : float, default=0.01
        梯度下降学习率
    random_state : int or None, default=None
        随机种子
    verbose : bool, default=False
        是否打印迭代信息

    Returns:
    --------
    coords : numpy.ndarray, shape (n_samples, n_components)
        低维空间中的坐标
    stress : float
        最终的Kruskal Stress-1值
    disparities : numpy.ndarray, shape (n_samples, n_samples)
        保序回归后的disparities矩阵（单调变换结果）

    Notes:
    ------
    算法流程（Shepard-Kruskal 方法）：
    1. 初始化低维坐标（随机或由classical MDS提供）
    2. 计算当前低维点之间的距离 d_ij
    3. 对 d_ij 按原始不相似度 δ_ij 的排序进行保序回归，得到 d̂_ij
    4. 计算梯度 ∂Stress/∂X，用梯度下降更新坐标
    5. 重复2-4直到收敛

    Kruskal Stress-1 拟合质量参考标准：
      < 0.025  极好
      < 0.05   很好
      < 0.10   一般
      < 0.20   较差
      >= 0.20  差
    可使用 stress_quality() 函数获取评估。
    """
    _validate_n_components(n_components)

    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]

    if D.shape[1] != n:
        raise ValueError("距离矩阵必须是方阵")

    if not np.allclose(D, D.T):
        raise ValueError("距离矩阵必须是对称的")

    rng = np.random.RandomState(random_state)

    best_coords = None
    best_stress = np.inf
    best_disparities = None

    for init_idx in range(n_init):
        X = rng.randn(n, n_components) * 0.1

        prev_stress = np.inf

        for iteration in range(max_iter):
            D_current = _compute_pairwise_distances(X)

            disparities = _compute_disparities(D, D_current)

            stress = _compute_stress_nonmetric(D, D_current, disparities)

            if verbose and iteration % 50 == 0:
                print(
                    f"  初始化{init_idx + 1}, 迭代{iteration}: "
                    f"stress = {stress:.6f} ({stress_quality(stress)})"
                )

            if abs(prev_stress - stress) < eps:
                if verbose:
                    print(
                        f"  初始化{init_idx + 1}, 迭代{iteration}: "
                        f"收敛 (stress = {stress:.6f})"
                    )
                break

            prev_stress = stress

            grad = np.zeros_like(X)
            for i in range(n):
                for j in range(n):
                    if i == j or D[i, j] <= 0:
                        continue
                    d_ij = D_current[i, j]
                    if d_ij < 1e-10:
                        continue
                    diff_ij = X[i] - X[j]
                    ratio = (d_ij - disparities[i, j]) / d_ij
                    grad[i] += ratio * diff_ij

            grad *= 2.0
            X -= learning_rate * grad

        D_final = _compute_pairwise_distances(X)
        final_disparities = _compute_disparities(D, D_final)
        final_stress = _compute_stress_nonmetric(D, D_final, final_disparities)

        if final_stress < best_stress:
            best_stress = final_stress
            best_coords = X.copy()
            best_disparities = final_disparities

    return best_coords, best_stress, best_disparities


if __name__ == "__main__":
    print("=" * 60)
    print("测试1: 正常距离矩阵（满足三角不等式）")
    print("=" * 60)

    np.random.seed(42)
    n_points = 6
    X_true = np.random.randn(n_points, 3)

    from scipy.spatial.distance import pdist, squareform

    D_normal = squareform(pdist(X_true))

    print("\n--- Classical MDS ---")
    coords_c, eigvals_c = cmds(D_normal, n_components=2)
    print("坐标:\n", coords_c)
    print("特征值:", eigvals_c)

    print("\n--- Metric MDS ---")
    coords_m, stress_m = metric_mds(D_normal, n_components=2, random_state=42)
    print("坐标:\n", coords_m)
    print(f"Stress: {stress_m:.6f} -> {stress_quality(stress_m)}")

    print("\n--- Non-metric MDS ---")
    coords_n, stress_n, disp_n = nonmetric_mds(D_normal, n_components=2, random_state=42)
    print("坐标:\n", coords_n)
    print(f"Stress: {stress_n:.6f} -> {stress_quality(stress_n)}")

    print("\n" + "=" * 60)
    print("测试2: 不满足三角不等式的距离矩阵")
    print("=" * 60)

    D_bad = np.array(
        [
            [0.0, 1.0, 10.0],
            [1.0, 0.0, 1.0],
            [10.0, 1.0, 0.0],
        ]
    )
    print("\n距离矩阵:")
    print(D_bad)
    print("注意: d(0,2)=10 > d(0,1)+d(1,2)=2，不满足三角不等式")

    print("\n--- Classical MDS ---")
    coords_c2, eigvals_c2 = cmds(D_bad, n_components=2)
    print("坐标:\n", coords_c2)
    print("特征值:", eigvals_c2)

    print("\n--- Metric MDS ---")
    coords_m2, stress_m2 = metric_mds(D_bad, n_components=2, random_state=42)
    print("坐标:\n", coords_m2)
    print(f"Stress: {stress_m2:.6f} -> {stress_quality(stress_m2)}")

    print("\n--- Non-metric MDS ---")
    coords_n2, stress_n2, disp_n2 = nonmetric_mds(D_bad, n_components=2, random_state=42)
    print("坐标:\n", coords_n2)
    print(f"Stress: {stress_n2:.6f} -> {stress_quality(stress_n2)}")
    print("Disparities（保序回归后的单调变换）:")
    print(disp_n2)

    print("\n" + "=" * 60)
    print("测试3: 心理测量学场景 — 仅有排序数据")
    print("=" * 60)

    D_rank = np.array(
        [
            [0, 2, 5, 8],
            [2, 0, 3, 7],
            [5, 3, 0, 4],
            [8, 7, 4, 0],
        ]
    )
    print("\n排序距离矩阵（仅大小关系有意义）:")
    print(D_rank)

    print("\n--- Non-metric MDS ---")
    coords_n3, stress_n3, disp_n3 = nonmetric_mds(D_rank, n_components=2, random_state=42)
    print("坐标:\n", coords_n3)
    print(f"Stress: {stress_n3:.6f} -> {stress_quality(stress_n3)}")
    print("Disparities:")
    print(disp_n3)

    print("\n" + "=" * 60)
    print("测试4: 不同维度 (1-5) 的Stress对比")
    print("=" * 60)

    np.random.seed(0)
    X_big = np.random.randn(15, 5)
    D_big = squareform(pdist(X_big))

    print(f"\n{'维度':>4} | {'Classical MDS':>20} | {'Metric MDS':>20} | {'Non-metric MDS':>20}")
    print("-" * 76)

    for dim in range(1, 6):
        _, eig_c = cmds(D_big, n_components=dim)
        _, s_m = metric_mds(D_big, n_components=dim, random_state=0)
        _, s_n, _ = nonmetric_mds(D_big, n_components=dim, random_state=0)

        eig_explained = np.sum(eig_c) / np.sum(np.maximum(np.linalg.eigh(
            -0.5 * (np.eye(15) - np.ones((15, 15)) / 15)
            @ (D_big ** 2)
            @ (np.eye(15) - np.ones((15, 15)) / 15)
        )[0], 0))

        print(f"{dim:>4} | {eig_explained:>17.4f} | {s_m:>17.6f} | {s_n:>17.6f} ({stress_quality(s_n)})")

    print(f"\nStress质量标准 (Kruskal 1964):")
    print(f"  < 0.025  极好")
    print(f"  < 0.05   很好")
    print(f"  < 0.10   一般")
    print(f"  < 0.20   较差")
    print(f"  >= 0.20  差")
