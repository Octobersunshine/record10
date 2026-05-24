import numpy as np


def partition_entropy(u):
    """
    计算划分熵（Partition Entropy）
    值越小表示聚类效果越好（越清晰）
    """
    n = u.shape[0]
    u = np.clip(u, 1e-10, 1 - 1e-10)
    entropy = -np.sum(u * np.log(u)) / n
    return entropy


def xie_beni_index(data, u, centers, m=2):
    """
    计算Xie-Beni指数
    值越小表示聚类效果越好（紧凑且分离）
    """
    n_samples, c = u.shape
    um = u ** m

    compactness = np.sum(um * np.linalg.norm(data[:, np.newaxis] - centers, axis=2) ** 2)

    min_sep = np.inf
    for i in range(c):
        for j in range(i + 1, c):
            sep = np.linalg.norm(centers[i] - centers[j]) ** 2
            if sep < min_sep:
                min_sep = sep

    if min_sep == 0:
        return np.inf

    return compactness / (n_samples * min_sep)


def select_best_m(data, c, m_range=np.arange(1.1, 5.1, 0.1), criterion='xie_beni', 
                  max_iter=1000, tol=1e-5, random_state=None):
    """
    自动选择最佳模糊系数m

    参数:
        data: 数据点矩阵
        c: 聚类数目
        m_range: 待搜索的m值范围
        criterion: 选择准则，'xie_beni' 或 'entropy'
        max_iter: FCM最大迭代次数
        tol: FCM收敛阈值
        random_state: 随机种子

    返回:
        best_m: 最佳m值
        best_score: 最佳评分
        scores: 各m值对应的评分
    """
    scores = []
    
    for m in m_range:
        u, centers = fuzzy_c_means(data, c, m=m, max_iter=max_iter, 
                                   tol=tol, random_state=random_state)
        
        if criterion == 'xie_beni':
            score = xie_beni_index(data, u, centers, m)
        elif criterion == 'entropy':
            score = partition_entropy(u)
        else:
            raise ValueError("criterion 必须是 'xie_beni' 或 'entropy'")
        
        scores.append(score)
    
    best_idx = np.argmin(scores)
    best_m = m_range[best_idx]
    best_score = scores[best_idx]
    
    return best_m, best_score, scores


def fuzzy_c_means_auto_m(data, c, m_range=np.arange(1.1, 5.1, 0.1), criterion='xie_beni',
                         max_iter=1000, tol=1e-5, random_state=None):
    """
    自动选择m的模糊C均值聚类

    返回:
        u: 隶属度矩阵
        centers: 聚类中心
        best_m: 自动选择的最佳m值
    """
    best_m, _, _ = select_best_m(data, c, m_range=m_range, criterion=criterion,
                                 max_iter=max_iter, tol=tol, random_state=random_state)
    u, centers = fuzzy_c_means(data, c, m=best_m, max_iter=max_iter,
                               tol=tol, random_state=random_state)
    return u, centers, best_m


def fuzzy_c_means(data, c, m=2, max_iter=1000, tol=1e-5, random_state=None):
    """
    模糊C均值（FCM）聚类算法

    参数:
        data: 数据点矩阵，形状为 (n_samples, n_features)
        c: 聚类数目
        m: 模糊系数（m > 1），控制聚类的模糊程度
        max_iter: 最大迭代次数
        tol: 收敛阈值，隶属度矩阵的变化小于该值时停止迭代
        random_state: 随机种子，用于重现结果

    返回:
        u: 隶属度矩阵，形状为 (n_samples, c)，u[i,j] 表示第i个样本属于第j个簇的隶属度
        centers: 聚类中心，形状为 (c, n_features)
    """
    if m <= 1:
        raise ValueError("模糊系数 m 必须大于 1")

    n_samples, n_features = data.shape

    if random_state is not None:
        np.random.seed(random_state)

    u = np.random.rand(n_samples, c)
    u = u / np.sum(u, axis=1, keepdims=True)

    for _ in range(max_iter):
        u_old = u.copy()

        um = u ** m
        centers = np.dot(um.T, data) / np.sum(um.T, axis=1, keepdims=True)

        dist = np.zeros((n_samples, c))
        for j in range(c):
            dist[:, j] = np.linalg.norm(data - centers[j], axis=1)

        dist = np.maximum(dist, 1e-10)

        exponent = 2 / (m - 1)
        u_new = np.zeros((n_samples, c))
        for j in range(c):
            u_new[:, j] = 1 / np.sum((dist[:, j] / dist) ** exponent, axis=1)

        u = u_new

        if np.linalg.norm(u - u_old) < tol:
            break

    return u, centers


def compute_neighbors_2d(height, width, window_size=3):
    """
    计算2D图像每个像素的邻域索引

    参数:
        height: 图像高度
        width: 图像宽度
        window_size: 邻域窗口大小（奇数）

    返回:
        neighbors: 形状为 (height*width, n_neighbors) 的邻域索引数组
    """
    if window_size % 2 == 0:
        raise ValueError("window_size 必须是奇数")

    half = window_size // 2
    n_pixels = height * width
    neighbors_list = []

    for i in range(height):
        for j in range(width):
            idx = i * width + j
            current_neighbors = []
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        nidx = ni * width + nj
                        current_neighbors.append(nidx)
            neighbors_list.append(current_neighbors)

    max_neighbors = max(len(n) for n in neighbors_list)
    neighbors = np.zeros((n_pixels, max_neighbors), dtype=np.int64)
    for i, n in enumerate(neighbors_list):
        neighbors[i, :len(n)] = n

    return neighbors


def compute_neighborhood_mean(data, neighbors):
    """
    计算每个样本的邻域均值

    参数:
        data: 数据矩阵 (n_samples, n_features)
        neighbors: 邻域索引数组

    返回:
        neighborhood_mean: 邻域均值矩阵 (n_samples, n_features)
    """
    n_samples, n_features = data.shape
    neighborhood_mean = np.zeros_like(data)

    for i in range(n_samples):
        n_indices = neighbors[i]
        valid_indices = n_indices[n_indices >= 0]
        neighborhood_mean[i] = np.mean(data[valid_indices], axis=0)

    return neighborhood_mean


def fcm_s(data, c, spatial_weight=0.5, m=2, max_iter=1000, tol=1e-5,
          neighbors=None, image_shape=None, window_size=3, random_state=None):
    """
    空间约束模糊C均值（FCM_S）聚类算法

    通过引入空间邻域信息，提高抗噪性，特别适用于图像分割

    参数:
        data: 数据点矩阵，形状为 (n_samples, n_features)
        c: 聚类数目
        spatial_weight: 空间约束权重 α (0 ≤ α < 1)，越大空间约束越强
        m: 模糊系数（m > 1）
        max_iter: 最大迭代次数
        tol: 收敛阈值
        neighbors: 预计算的邻域索引数组（可选）
        image_shape: 图像形状 (height, width)，用于自动计算邻域
        window_size: 邻域窗口大小（奇数），默认3×3
        random_state: 随机种子

    返回:
        u: 隶属度矩阵
        centers: 聚类中心
    """
    if m <= 1:
        raise ValueError("模糊系数 m 必须大于 1")
    if spatial_weight < 0 or spatial_weight >= 1:
        raise ValueError("空间权重 spatial_weight 必须在 [0, 1) 范围内")

    n_samples, n_features = data.shape

    if neighbors is None and image_shape is not None:
        neighbors = compute_neighbors_2d(image_shape[0], image_shape[1], window_size)

    if neighbors is not None:
        neighborhood_data = compute_neighborhood_mean(data, neighbors)
    else:
        neighborhood_data = data

    if random_state is not None:
        np.random.seed(random_state)

    u = np.random.rand(n_samples, c)
    u = u / np.sum(u, axis=1, keepdims=True)

    alpha = spatial_weight
    beta = 1 - alpha

    for _ in range(max_iter):
        u_old = u.copy()

        um = u ** m
        um_sum = np.sum(um, axis=0, keepdims=True)
        centers = (beta * np.dot(um.T, data) + alpha * np.dot(um.T, neighborhood_data)) / \
                  ((beta + alpha) * um_sum.T)

        dist = np.zeros((n_samples, c))
        spatial_dist = np.zeros((n_samples, c))
        for j in range(c):
            dist[:, j] = np.linalg.norm(data - centers[j], axis=1)
            spatial_dist[:, j] = np.linalg.norm(neighborhood_data - centers[j], axis=1)

        dist = np.maximum(dist, 1e-10)
        spatial_dist = np.maximum(spatial_dist, 1e-10)

        combined_dist = beta * (dist ** 2) + alpha * (spatial_dist ** 2)
        combined_dist = np.maximum(combined_dist, 1e-10)

        exponent = 1 / (m - 1)
        u_new = np.zeros((n_samples, c))
        for j in range(c):
            ratio = (combined_dist[:, j, np.newaxis] / combined_dist) ** exponent
            u_new[:, j] = 1 / np.sum(ratio, axis=1)

        u = u_new

        if np.linalg.norm(u - u_old) < tol:
            break

    return u, centers


def fcm_s1(data, c, spatial_weight=0.5, m=2, max_iter=1000, tol=1e-5,
           neighbors=None, image_shape=None, window_size=3, random_state=None):
    """
    FCM_S1 算法 - 使用原始像素计算距离，仅在更新中心时考虑邻域均值

    参数:
        data: 数据点矩阵
        c: 聚类数目
        spatial_weight: 空间约束权重 α
        m: 模糊系数
        max_iter: 最大迭代次数
        tol: 收敛阈值
        neighbors: 邻域索引
        image_shape: 图像形状
        window_size: 邻域窗口大小
        random_state: 随机种子

    返回:
        u: 隶属度矩阵
        centers: 聚类中心
    """
    if m <= 1:
        raise ValueError("模糊系数 m 必须大于 1")

    n_samples, n_features = data.shape

    if neighbors is None and image_shape is not None:
        neighbors = compute_neighbors_2d(image_shape[0], image_shape[1], window_size)

    if neighbors is not None:
        neighborhood_data = compute_neighborhood_mean(data, neighbors)
    else:
        neighborhood_data = data

    if random_state is not None:
        np.random.seed(random_state)

    u = np.random.rand(n_samples, c)
    u = u / np.sum(u, axis=1, keepdims=True)

    alpha = spatial_weight
    beta = 1 - alpha

    for _ in range(max_iter):
        u_old = u.copy()

        um = u ** m
        um_sum = np.sum(um, axis=0, keepdims=True)
        centers = (beta * np.dot(um.T, data) + alpha * np.dot(um.T, neighborhood_data)) / \
                  ((beta + alpha) * um_sum.T)

        dist = np.zeros((n_samples, c))
        for j in range(c):
            dist[:, j] = np.linalg.norm(data - centers[j], axis=1)

        dist = np.maximum(dist, 1e-10)

        exponent = 2 / (m - 1)
        u_new = np.zeros((n_samples, c))
        for j in range(c):
            u_new[:, j] = 1 / np.sum((dist[:, j] / dist) ** exponent, axis=1)

        u = u_new

        if np.linalg.norm(u - u_old) < tol:
            break

    return u, centers


def fcm_s2(data, c, spatial_weight=0.5, m=2, max_iter=1000, tol=1e-5,
           neighbors=None, image_shape=None, window_size=3, random_state=None):
    """
    FCM_S2 算法 - 使用邻域均值计算距离和更新中心

    参数:
        data: 数据点矩阵
        c: 聚类数目
        spatial_weight: 空间约束权重 α
        m: 模糊系数
        max_iter: 最大迭代次数
        tol: 收敛阈值
        neighbors: 邻域索引
        image_shape: 图像形状
        window_size: 邻域窗口大小
        random_state: 随机种子

    返回:
        u: 隶属度矩阵
        centers: 聚类中心
    """
    if m <= 1:
        raise ValueError("模糊系数 m 必须大于 1")

    n_samples, n_features = data.shape

    if neighbors is None and image_shape is not None:
        neighbors = compute_neighbors_2d(image_shape[0], image_shape[1], window_size)

    if neighbors is not None:
        neighborhood_data = compute_neighborhood_mean(data, neighbors)
    else:
        neighborhood_data = data

    if random_state is not None:
        np.random.seed(random_state)

    u = np.random.rand(n_samples, c)
    u = u / np.sum(u, axis=1, keepdims=True)

    for _ in range(max_iter):
        u_old = u.copy()

        um = u ** m
        um_sum = np.sum(um, axis=0, keepdims=True)
        centers = np.dot(um.T, neighborhood_data) / um_sum.T

        dist = np.zeros((n_samples, c))
        for j in range(c):
            dist[:, j] = np.linalg.norm(neighborhood_data - centers[j], axis=1)

        dist = np.maximum(dist, 1e-10)

        exponent = 2 / (m - 1)
        u_new = np.zeros((n_samples, c))
        for j in range(c):
            u_new[:, j] = 1 / np.sum((dist[:, j] / dist) ** exponent, axis=1)

        u = u_new

        if np.linalg.norm(u - u_old) < tol:
            break

    return u, centers


if __name__ == "__main__":
    np.random.seed(42)
    data1 = np.random.randn(50, 2) + np.array([2, 2])
    data2 = np.random.randn(50, 2) + np.array([-2, -2])
    data3 = np.random.randn(50, 2) + np.array([2, -2])
    data = np.vstack([data1, data2, data3])

    c = 3

    print("=" * 60)
    print("1. 固定m=2的FCM聚类结果")
    print("=" * 60)
    u, centers = fuzzy_c_means(data, c, m=2, random_state=42)
    print("隶属度矩阵形状:", u.shape)
    print("\n前5个样本的隶属度:")
    print(u[:5])
    print("\n划分熵:", partition_entropy(u))
    print("Xie-Beni指数:", xie_beni_index(data, u, centers, m=2))

    print("\n" + "=" * 60)
    print("2. 基于Xie-Beni指数自动选择最佳m")
    print("=" * 60)
    best_m_xb, best_score_xb, scores_xb = select_best_m(
        data, c, m_range=np.arange(1.2, 4.1, 0.2), criterion='xie_beni', random_state=42
    )
    print(f"最佳m值 (Xie-Beni): {best_m_xb:.1f}")
    print(f"最佳Xie-Beni指数: {best_score_xb:.4f}")

    print("\n" + "=" * 60)
    print("3. 基于划分熵自动选择最佳m")
    print("=" * 60)
    best_m_ent, best_score_ent, scores_ent = select_best_m(
        data, c, m_range=np.arange(1.2, 4.1, 0.2), criterion='entropy', random_state=42
    )
    print(f"最佳m值 (划分熵): {best_m_ent:.1f}")
    print(f"最佳划分熵: {best_score_ent:.4f}")

    print("\n" + "=" * 60)
    print("4. 使用自动选择的m（Xie-Beni）进行聚类")
    print("=" * 60)
    u_auto, centers_auto, best_m = fuzzy_c_means_auto_m(
        data, c, m_range=np.arange(1.2, 4.1, 0.2), criterion='xie_beni', random_state=42
    )
    print(f"自动选择的m值: {best_m:.1f}")
    print("\n前5个样本的隶属度:")
    print(u_auto[:5])
    print("\n聚类中心:")
    print(centers_auto)

    print("\n" + "=" * 60)
    print("5. 对比不同m值的效果（m过小vs m过大）")
    print("=" * 60)
    for m_test in [1.1, 2.0, 5.0]:
        u_test, centers_test = fuzzy_c_means(data, c, m=m_test, random_state=42)
        print(f"\nm = {m_test}:")
        print(f"  隶属度最大值（前3样本）: {np.max(u_test[:3], axis=1)}")
        print(f"  隶属度最小值（前3样本）: {np.min(u_test[:3], axis=1)}")
        print(f"  划分熵: {partition_entropy(u_test):.4f}")
        print(f"  Xie-Beni指数: {xie_beni_index(data, u_test, centers_test, m=m_test):.4f}")

    print("\n" + "=" * 60)
    print("6. FCM_S 空间约束聚类（模拟图像数据）")
    print("=" * 60)
    height, width = 15, 10
    n_pixels = height * width

    image_clean = np.zeros((height, width, 1))
    image_clean[:5, :, 0] = 0.2
    image_clean[5:10, :, 0] = 0.6
    image_clean[10:, :, 0] = 0.9

    np.random.seed(42)
    noise = np.random.randn(height, width, 1) * 0.15
    image_noisy = image_clean + noise
    image_noisy = np.clip(image_noisy, 0, 1)

    image_flat = image_noisy.reshape(-1, 1)
    image_shape = (height, width)

    print(f"图像形状: {height}×{width}, 噪声水平: 0.15")

    print("\n  6.1 标准FCM聚类结果:")
    u_fcm, centers_fcm = fuzzy_c_means(image_flat, c=3, m=2, random_state=42)
    labels_fcm = np.argmax(u_fcm, axis=1)
    print(f"    聚类中心: {centers_fcm.flatten()}")
    print(f"    划分熵: {partition_entropy(u_fcm):.4f}")

    print("\n  6.2 FCM_S空间约束聚类 (α=0.3):")
    u_fcms, centers_fcms = fcm_s(
        image_flat, c=3, spatial_weight=0.3, m=2,
        image_shape=image_shape, window_size=3, random_state=42
    )
    labels_fcms = np.argmax(u_fcms, axis=1)
    print(f"    聚类中心: {centers_fcms.flatten()}")
    print(f"    划分熵: {partition_entropy(u_fcms):.4f}")

    print("\n  6.3 FCM_S1聚类 (α=0.3):")
    u_fcms1, centers_fcms1 = fcm_s1(
        image_flat, c=3, spatial_weight=0.3, m=2,
        image_shape=image_shape, window_size=3, random_state=42
    )
    labels_fcms1 = np.argmax(u_fcms1, axis=1)
    print(f"    聚类中心: {centers_fcms1.flatten()}")
    print(f"    划分熵: {partition_entropy(u_fcms1):.4f}")

    print("\n  6.4 FCM_S2聚类:")
    u_fcms2, centers_fcms2 = fcm_s2(
        image_flat, c=3, m=2,
        image_shape=image_shape, window_size=3, random_state=42
    )
    labels_fcms2 = np.argmax(u_fcms2, axis=1)
    print(f"    聚类中心: {centers_fcms2.flatten()}")
    print(f"    划分熵: {partition_entropy(u_fcms2):.4f}")

    print("\n  6.5 空间约束权重α的影响:")
    for alpha in [0.1, 0.3, 0.5]:
        u_a, _ = fcm_s(
            image_flat, c=3, spatial_weight=alpha, m=2,
            image_shape=image_shape, window_size=3, random_state=42
        )
        print(f"    α={alpha}: 划分熵={partition_entropy(u_a):.4f}")

    print("\n" + "=" * 60)
    print("7. FCM_S抗噪性对比")
    print("=" * 60)
    print("  空间约束使得邻域像素更可能属于同一簇")
    print("  减少噪声点对聚类结果的影响")
    print("  适用于图像分割等具有空间连续性的数据")
