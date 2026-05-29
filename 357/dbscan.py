import numpy as np
from collections import deque


def compute_distance_matrix(X):
    X = np.asarray(X, dtype=float)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def compute_k_distances(dist_matrix, k):
    n = dist_matrix.shape[0]
    k_distances = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(dist_matrix[i])
        k_distances[i] = sorted_dists[k]
    return k_distances


def recommend_eps_knee(X, k=None, min_samples=None):
    if k is None:
        if min_samples is None:
            k = 4
        else:
            k = min_samples - 1

    dist_matrix = compute_distance_matrix(X)
    k_distances = compute_k_distances(dist_matrix, k)
    sorted_kd = np.sort(k_distances)

    n = len(sorted_kd)
    if n < 3:
        return np.median(sorted_kd), sorted_kd

    x = np.arange(n)
    y = sorted_kd

    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    if norm < 1e-10:
        return np.median(y), sorted_kd

    dists_to_line = np.abs(dy * x - dx * y + dx * y[0] - dy * x[0]) / norm

    knee_idx = np.argmax(dists_to_line)
    recommended_eps = float(sorted_kd[knee_idx])

    return recommended_eps, sorted_kd


def dbscan(X, eps, min_samples):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    dist_matrix = compute_distance_matrix(X)

    neighbors = [np.where(dist_matrix[i] <= eps)[0] for i in range(n)]

    is_core = np.array([len(neighbors[i]) >= min_samples for i in range(n)])

    point_type = ["noise"] * n
    for i in range(n):
        if is_core[i]:
            point_type[i] = "core"

    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        if visited[i] or not is_core[i]:
            continue

        queue = deque([i])
        visited[i] = True
        labels[i] = cluster_id

        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    labels[neighbor] = cluster_id
                    if not is_core[neighbor]:
                        point_type[neighbor] = "border"
                    else:
                        queue.append(neighbor)

        cluster_id += 1

    return labels, point_type


def adaptive_dbscan(X, min_samples, k=None, mode='global', eps_override=None, local_scale=1.0):
    """
    自适应 DBSCAN 聚类算法。

    参数:
        X: 样本数据 (n_samples × n_features)
        min_samples: 核心点所需最小邻居数（含自身）
        k: 用于计算 k-距离图的 k 值，默认取 min_samples-1
        mode: 'global' - 全局自适应（拐点法推荐单一 eps）
              'local'  - 局部自适应（每个点使用自身 k-邻域距离作为 eps）
        eps_override: 手动覆盖 eps，设置后忽略自动推荐
        local_scale: 局部模式下 eps 的缩放系数，>1 扩大邻域，<1 缩小邻域

    返回:
        labels: 聚类标签，噪声点为 -1
        point_type: 每个点的类型（'core' / 'border' / 'noise'）
        info: 包含推荐 eps 等信息的字典
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if k is None:
        k = min_samples - 1

    info = {
        'k': k,
        'min_samples': min_samples,
        'mode': mode,
        'eps_used': None,
        'k_distances': None,
        'local_scale': local_scale
    }

    dist_matrix = compute_distance_matrix(X)
    k_distances = compute_k_distances(dist_matrix, k)
    info['k_distances'] = k_distances

    if eps_override is not None:
        info['eps_used'] = eps_override
        return dbscan(X, eps_override, min_samples) + (info,)

    if mode == 'global':
        recommended_eps, _ = recommend_eps_knee(X, k=k)
        info['eps_used'] = recommended_eps
        return dbscan(X, recommended_eps, min_samples) + (info,)

    elif mode == 'local':
        scaled_eps = k_distances * local_scale
        info['eps_used'] = scaled_eps.copy()

        neighbors = []
        for i in range(n):
            eps_i = scaled_eps[i]
            reachable = np.where(dist_matrix[i] <= eps_i)[0]
            neighbors.append(reachable)

        is_core = np.array([len(neighbors[i]) >= min_samples for i in range(n)])

        point_type = ["noise"] * n
        for i in range(n):
            if is_core[i]:
                point_type[i] = "core"

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i] or not is_core[i]:
                continue

            queue = deque([i])
            visited[i] = True
            labels[i] = cluster_id

            while queue:
                current = queue.popleft()
                for neighbor in neighbors[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        labels[neighbor] = cluster_id
                        if not is_core[neighbor]:
                            point_type[neighbor] = "border"
                        else:
                            queue.append(neighbor)

            cluster_id += 1

        return labels, point_type, info

    else:
        raise ValueError(f"未知 mode: {mode}，请使用 'global' 或 'local'")


if __name__ == "__main__":
    print("=" * 60)
    print("测试 1: 密度均匀数据 - 对比原始 DBSCAN 与全局自适应")
    print("=" * 60)
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) + [2, 2]
    cluster2 = np.random.randn(50, 2) + [-2, -2]
    noise = np.random.uniform(-6, 6, (10, 2))
    X1 = np.vstack([cluster1, cluster2, noise])

    print("\n--- 原始 DBSCAN (eps=1.2, min_samples=5) ---")
    labels_fixed, types_fixed = dbscan(X1, eps=1.2, min_samples=5)
    print(f"簇数量: {labels_fixed.max() + 1 if labels_fixed.max() >= 0 else 0}")
    print(f"噪声点: {np.sum(labels_fixed == -1)}")

    print("\n--- 自适应 DBSCAN (global 模式, min_samples=5) ---")
    labels_global, types_global, info_global = adaptive_dbscan(X1, min_samples=5, mode='global')
    print(f"推荐 eps: {info_global['eps_used']:.4f}")
    print(f"簇数量: {labels_global.max() + 1 if labels_global.max() >= 0 else 0}")
    print(f"噪声点: {np.sum(labels_global == -1)}")

    print("\n--- 自适应 DBSCAN (global 模式 + eps_override=1.2) ---")
    labels_ov, types_ov, info_ov = adaptive_dbscan(X1, min_samples=5, mode='global', eps_override=1.2)
    print(f"使用 eps: {info_ov['eps_used']}")
    print(f"簇数量: {labels_ov.max() + 1 if labels_ov.max() >= 0 else 0}")
    print(f"与固定 eps 结果一致: {np.array_equal(labels_ov, labels_fixed)}")

    print("\n" + "=" * 60)
    print("测试 2: 密度差异大的数据 - 对比 global 与 local 模式")
    print("=" * 60)
    np.random.seed(123)
    dense_cluster = np.random.randn(80, 2) * 0.2 + [0, 0]
    sparse_cluster = np.random.randn(40, 2) * 1.5 + [5, 5]
    X2 = np.vstack([dense_cluster, sparse_cluster])

    print(f"\n数据集: 高密度簇(80点, σ=0.2) + 低密度簇(40点, σ=1.5)")

    rec_eps, _ = recommend_eps_knee(X2, min_samples=5)
    print(f"\n拐点法推荐 eps: {rec_eps:.4f}")

    print("\n--- 全局自适应 DBSCAN ---")
    labels_g, types_g, info_g = adaptive_dbscan(X2, min_samples=5, mode='global')
    core_g = sum(1 for t in types_g if t == 'core')
    border_g = sum(1 for t in types_g if t == 'border')
    noise_g = sum(1 for t in types_g if t == 'noise')
    print(f"使用 eps: {info_g['eps_used']:.4f}")
    print(f"簇数量: {labels_g.max() + 1 if labels_g.max() >= 0 else 0}")
    print(f"核心点: {core_g}, 边界点: {border_g}, 噪声点: {noise_g}")
    print(f"高密度簇被识别点数: {np.sum(labels_g[:80] != -1)}")
    print(f"低密度簇被识别点数: {np.sum(labels_g[80:] != -1)}")

    print("\n--- 局部自适应 DBSCAN (local_scale=1.0) ---")
    labels_l, types_l, info_l = adaptive_dbscan(X2, min_samples=5, mode='local')
    core_l = sum(1 for t in types_l if t == 'core')
    border_l = sum(1 for t in types_l if t == 'border')
    noise_l = sum(1 for t in types_l if t == 'noise')
    print(f"使用 eps 范围: [{info_l['eps_used'].min():.4f}, {info_l['eps_used'].max():.4f}]")
    print(f"簇数量: {labels_l.max() + 1 if labels_l.max() >= 0 else 0}")
    print(f"核心点: {core_l}, 边界点: {border_l}, 噪声点: {noise_l}")
    print(f"高密度簇被识别点数: {np.sum(labels_l[:80] != -1)}")
    print(f"低密度簇被识别点数: {np.sum(labels_l[80:] != -1)}")

    print("\n--- 局部自适应 DBSCAN (local_scale=3.0，减少过分割) ---")
    labels_l3, types_l3, info_l3 = adaptive_dbscan(X2, min_samples=5, mode='local', local_scale=3.0)
    core_l3 = sum(1 for t in types_l3 if t == 'core')
    border_l3 = sum(1 for t in types_l3 if t == 'border')
    noise_l3 = sum(1 for t in types_l3 if t == 'noise')
    print(f"使用 eps 范围: [{info_l3['eps_used'].min():.4f}, {info_l3['eps_used'].max():.4f}]")
    print(f"簇数量: {labels_l3.max() + 1 if labels_l3.max() >= 0 else 0}")
    print(f"核心点: {core_l3}, 边界点: {border_l3}, 噪声点: {noise_l3}")
    print(f"高密度簇被识别点数: {np.sum(labels_l3[:80] != -1)}")
    print(f"低密度簇被识别点数: {np.sum(labels_l3[80:] != -1)}")

    print("\n" + "=" * 60)
    print("测试 3: 与 sklearn 对比验证")
    print("=" * 60)
    try:
        from sklearn.cluster import DBSCAN as SKDBSCAN
        sk_labels = SKDBSCAN(eps=1.2, min_samples=5).fit_predict(X1)
        my_labels, _ = dbscan(X1, eps=1.2, min_samples=5)
        print(f"与 sklearn DBSCAN 结果一致: {np.array_equal(sk_labels, my_labels)}")
    except ImportError:
        print("未安装 sklearn，跳过对比")


def compute_core_distances(dist_matrix, min_samples):
    n = dist_matrix.shape[0]
    k = min_samples - 1
    core_dists = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(dist_matrix[i])
        core_dists[i] = sorted_dists[k]
    return core_dists


def compute_mutual_reachability(dist_matrix, core_dists):
    n = dist_matrix.shape[0]
    mr_matrix = np.zeros_like(dist_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            mr = max(core_dists[i], core_dists[j], dist_matrix[i, j])
            mr_matrix[i, j] = mr
            mr_matrix[j, i] = mr
    return mr_matrix


def build_mst_prim(mr_matrix):
    n = mr_matrix.shape[0]
    INF = float('inf')

    key = np.full(n, INF)
    parent = np.full(n, -1, dtype=int)
    in_mst = np.zeros(n, dtype=bool)

    key[0] = 0

    for _ in range(n):
        u = -1
        min_key = INF
        for i in range(n):
            if not in_mst[i] and key[i] < min_key:
                min_key = key[i]
                u = i

        if u == -1:
            break

        in_mst[u] = True

        for v in range(n):
            if not in_mst[v] and mr_matrix[u, v] < key[v]:
                key[v] = mr_matrix[u, v]
                parent[v] = u

    edges = []
    for v in range(1, n):
        u = parent[v]
        edges.append((mr_matrix[u, v], u, v))

    edges.sort(key=lambda x: x[0])

    return edges


def build_hierarchy_from_mst(mst_edges, n):
    class DSU:
        def __init__(self, size):
            self.parent = {}
            for i in range(size):
                self.parent[i] = i

        def find(self, x):
            while self.parent[x] != x:
                x = self.parent[x]
            return x

        def union(self, x, y, new_id):
            rx = self.find(x)
            ry = self.find(y)
            self.parent[rx] = new_id
            self.parent[ry] = new_id
            self.parent[new_id] = new_id

    dsu = DSU(n)
    next_id = n

    hierarchy = []
    cluster_children = {}
    cluster_size = {i: 1 for i in range(n)}

    for dist, u, v in mst_edges:
        root_u = dsu.find(u)
        root_v = dsu.find(v)

        if root_u == root_v:
            continue

        hierarchy.append({
            'dist': dist,
            'parent_id': next_id,
            'child_a': root_u,
            'child_b': root_v,
            'size_a': cluster_size[root_u],
            'size_b': cluster_size[root_v]
        })

        cluster_children[next_id] = (root_u, root_v)
        cluster_size[next_id] = cluster_size[root_u] + cluster_size[root_v]

        dsu.union(root_u, root_v, next_id)
        next_id += 1

    root_id = dsu.find(0) if n > 0 else -1
    return hierarchy, cluster_children, cluster_size, root_id


def compute_cluster_stability(hierarchy, cluster_children, cluster_size, n, root_id, min_cluster_size, mst_edges):
    if len(mst_edges) < 2:
        return [(root_id, 1.0)] if cluster_size.get(root_id, 0) >= min_cluster_size else []

    edge_dists = np.array([e[0] for e in mst_edges])
    gaps = np.diff(edge_dists)

    if len(gaps) == 0:
        return [(root_id, 1.0)] if cluster_size.get(root_id, 0) >= min_cluster_size else []

    best_idx = np.argmax(gaps)
    cut_distance = edge_dists[best_idx]

    node_birth = {}
    for entry in hierarchy:
        node_birth[entry['parent_id']] = entry['dist']

    selected = []

    def find_clusters_at_cut(node):
        if node < n:
            return []

        birth = node_birth.get(node, float('inf'))

        if birth > cut_distance:
            children = cluster_children[node]
            result = []
            for child in children:
                if child >= n:
                    result.extend(find_clusters_at_cut(child))
            return result

        if cluster_size.get(node, 0) >= min_cluster_size:
            return [node]
        else:
            children = cluster_children[node]
            result = []
            for child in children:
                if child >= n:
                    result.extend(find_clusters_at_cut(child))
            return result

    selected_ids = find_clusters_at_cut(root_id)
    return [(cid, 1.0) for cid in selected_ids]


def assign_labels_from_clusters(selected_clusters, cluster_children, n, min_cluster_size):
    labels = np.full(n, -1, dtype=int)
    label_map = {}
    next_label = 0

    def get_points(node):
        if node < n:
            return [node]
        children = cluster_children[node]
        return get_points(children[0]) + get_points(children[1])

    for cluster_id, _ in selected_clusters:
        points = get_points(cluster_id)
        if len(points) >= min_cluster_size:
            label_map[cluster_id] = next_label
            for p in points:
                if labels[p] == -1:
                    labels[p] = next_label
            next_label += 1

    return labels


def silhouette_score(X, labels):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if len(unique_labels) < 2 or len(unique_labels) == n:
        return 0.0

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    scores = np.zeros(n)

    for i in range(n):
        if labels[i] == -1:
            scores[i] = 0.0
            continue

        same_cluster = labels == labels[i]
        same_cluster[i] = False

        if np.sum(same_cluster) == 0:
            a_i = 0.0
        else:
            a_i = np.mean(dist_matrix[i, same_cluster])

        min_b = float('inf')
        for l in unique_labels:
            if l == labels[i]:
                continue
            other_cluster = labels == l
            if np.sum(other_cluster) > 0:
                b = np.mean(dist_matrix[i, other_cluster])
                if b < min_b:
                    min_b = b

        b_i = min_b

        if max(a_i, b_i) > 0:
            scores[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            scores[i] = 0.0

    valid_scores = scores[labels != -1]
    if len(valid_scores) == 0:
        return 0.0

    return np.mean(valid_scores)


def hdbscan(X, min_samples=5, min_cluster_size=None):
    """
    HDBSCAN（层次密度聚类）算法。

    参数:
        X: 样本数据 (n_samples × n_features)
        min_samples: 核心距离计算使用的邻居数
        min_cluster_size: 最小簇大小，默认等于 min_samples

    返回:
        labels: 聚类标签，噪声点为 -1
        hierarchy: 层次聚类树
        info: 包含算法信息的字典
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if min_cluster_size is None:
        min_cluster_size = min_samples

    dist_matrix = compute_distance_matrix(X)

    core_dists = compute_core_distances(dist_matrix, min_samples)

    mr_matrix = compute_mutual_reachability(dist_matrix, core_dists)

    mst_edges = build_mst_prim(mr_matrix)

    hierarchy, cluster_children, cluster_size, root_id = build_hierarchy_from_mst(mst_edges, n)

    selected_clusters = compute_cluster_stability(
        hierarchy, cluster_children, cluster_size, n, root_id, min_cluster_size, mst_edges
    )

    labels = assign_labels_from_clusters(
        selected_clusters, cluster_children, n, min_cluster_size
    )

    sil_score = silhouette_score(X, labels)

    info = {
        'min_samples': min_samples,
        'min_cluster_size': min_cluster_size,
        'core_dists': core_dists,
        'mst_edges': mst_edges,
        'hierarchy': hierarchy,
        'selected_clusters': selected_clusters,
        'silhouette_score': sil_score
    }

    point_type = ["noise"] * n
    for i in range(n):
        if labels[i] != -1:
            eps_i = core_dists[i]
            neighbors = np.where(dist_matrix[i] <= eps_i)[0]
            if len(neighbors) >= min_samples:
                point_type[i] = "core"
            else:
                point_type[i] = "border"

    return labels, point_type, info


if __name__ == "__main__":
    print("=" * 60)
    print("测试 1: 密度均匀数据 - 对比原始 DBSCAN 与全局自适应")
    print("=" * 60)
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) + [2, 2]
    cluster2 = np.random.randn(50, 2) + [-2, -2]
    noise = np.random.uniform(-6, 6, (10, 2))
    X1 = np.vstack([cluster1, cluster2, noise])

    print("\n--- 原始 DBSCAN (eps=1.2, min_samples=5) ---")
    labels_fixed, types_fixed = dbscan(X1, eps=1.2, min_samples=5)
    print(f"簇数量: {labels_fixed.max() + 1 if labels_fixed.max() >= 0 else 0}")
    print(f"噪声点: {np.sum(labels_fixed == -1)}")

    print("\n--- 自适应 DBSCAN (global 模式, min_samples=5) ---")
    labels_global, types_global, info_global = adaptive_dbscan(X1, min_samples=5, mode='global')
    print(f"推荐 eps: {info_global['eps_used']:.4f}")
    print(f"簇数量: {labels_global.max() + 1 if labels_global.max() >= 0 else 0}")
    print(f"噪声点: {np.sum(labels_global == -1)}")

    print("\n--- 自适应 DBSCAN (global 模式 + eps_override=1.2) ---")
    labels_ov, types_ov, info_ov = adaptive_dbscan(X1, min_samples=5, mode='global', eps_override=1.2)
    print(f"使用 eps: {info_ov['eps_used']}")
    print(f"簇数量: {labels_ov.max() + 1 if labels_ov.max() >= 0 else 0}")
    print(f"与固定 eps 结果一致: {np.array_equal(labels_ov, labels_fixed)}")

    print("\n" + "=" * 60)
    print("测试 2: 密度差异大的数据 - 对比 global 与 local 模式")
    print("=" * 60)
    np.random.seed(123)
    dense_cluster = np.random.randn(80, 2) * 0.2 + [0, 0]
    sparse_cluster = np.random.randn(40, 2) * 1.5 + [5, 5]
    X2 = np.vstack([dense_cluster, sparse_cluster])

    print(f"\n数据集: 高密度簇(80点, σ=0.2) + 低密度簇(40点, σ=1.5)")

    rec_eps, _ = recommend_eps_knee(X2, min_samples=5)
    print(f"\n拐点法推荐 eps: {rec_eps:.4f}")

    print("\n--- 全局自适应 DBSCAN ---")
    labels_g, types_g, info_g = adaptive_dbscan(X2, min_samples=5, mode='global')
    core_g = sum(1 for t in types_g if t == 'core')
    border_g = sum(1 for t in types_g if t == 'border')
    noise_g = sum(1 for t in types_g if t == 'noise')
    print(f"使用 eps: {info_g['eps_used']:.4f}")
    print(f"簇数量: {labels_g.max() + 1 if labels_g.max() >= 0 else 0}")
    print(f"核心点: {core_g}, 边界点: {border_g}, 噪声点: {noise_g}")
    print(f"高密度簇被识别点数: {np.sum(labels_g[:80] != -1)}")
    print(f"低密度簇被识别点数: {np.sum(labels_g[80:] != -1)}")

    print("\n--- 局部自适应 DBSCAN (local_scale=1.0) ---")
    labels_l, types_l, info_l = adaptive_dbscan(X2, min_samples=5, mode='local')
    core_l = sum(1 for t in types_l if t == 'core')
    border_l = sum(1 for t in types_l if t == 'border')
    noise_l = sum(1 for t in types_l if t == 'noise')
    print(f"使用 eps 范围: [{info_l['eps_used'].min():.4f}, {info_l['eps_used'].max():.4f}]")
    print(f"簇数量: {labels_l.max() + 1 if labels_l.max() >= 0 else 0}")
    print(f"核心点: {core_l}, 边界点: {border_l}, 噪声点: {noise_l}")
    print(f"高密度簇被识别点数: {np.sum(labels_l[:80] != -1)}")
    print(f"低密度簇被识别点数: {np.sum(labels_l[80:] != -1)}")

    print("\n--- 局部自适应 DBSCAN (local_scale=3.0，减少过分割) ---")
    labels_l3, types_l3, info_l3 = adaptive_dbscan(X2, min_samples=5, mode='local', local_scale=3.0)
    core_l3 = sum(1 for t in types_l3 if t == 'core')
    border_l3 = sum(1 for t in types_l3 if t == 'border')
    noise_l3 = sum(1 for t in types_l3 if t == 'noise')
    print(f"使用 eps 范围: [{info_l3['eps_used'].min():.4f}, {info_l3['eps_used'].max():.4f}]")
    print(f"簇数量: {labels_l3.max() + 1 if labels_l3.max() >= 0 else 0}")
    print(f"核心点: {core_l3}, 边界点: {border_l3}, 噪声点: {noise_l3}")
    print(f"高密度簇被识别点数: {np.sum(labels_l3[:80] != -1)}")
    print(f"低密度簇被识别点数: {np.sum(labels_l3[80:] != -1)}")

    print("\n" + "=" * 60)
    print("测试 3: HDBSCAN 层次密度聚类")
    print("=" * 60)

    print("\n--- HDBSCAN 在密度差异数据上 ---")
    labels_h, types_h, info_h = hdbscan(X2, min_samples=5, min_cluster_size=10)
    core_h = sum(1 for t in types_h if t == 'core')
    border_h = sum(1 for t in types_h if t == 'border')
    noise_h = sum(1 for t in types_h if t == 'noise')
    print(f"簇数量: {labels_h.max() + 1 if labels_h.max() >= 0 else 0}")
    print(f"核心点: {core_h}, 边界点: {border_h}, 噪声点: {noise_h}")
    print(f"高密度簇被识别点数: {np.sum(labels_h[:80] != -1)}")
    print(f"低密度簇被识别点数: {np.sum(labels_h[80:] != -1)}")
    print(f"轮廓系数: {info_h['silhouette_score']:.4f}")

    print("\n--- HDBSCAN 在均匀密度数据上 ---")
    labels_h2, types_h2, info_h2 = hdbscan(X1, min_samples=5)
    core_h2 = sum(1 for t in types_h2 if t == 'core')
    border_h2 = sum(1 for t in types_h2 if t == 'border')
    noise_h2 = sum(1 for t in types_h2 if t == 'noise')
    print(f"簇数量: {labels_h2.max() + 1 if labels_h2.max() >= 0 else 0}")
    print(f"核心点: {core_h2}, 边界点: {border_h2}, 噪声点: {noise_h2}")
    print(f"轮廓系数: {info_h2['silhouette_score']:.4f}")

    print("\n" + "=" * 60)
    print("测试 4: 与 sklearn 对比验证")
    print("=" * 60)
    try:
        from sklearn.cluster import DBSCAN as SKDBSCAN
        sk_labels = SKDBSCAN(eps=1.2, min_samples=5).fit_predict(X1)
        my_labels, _ = dbscan(X1, eps=1.2, min_samples=5)
        print(f"与 sklearn DBSCAN 结果一致: {np.array_equal(sk_labels, my_labels)}")
    except ImportError:
        print("未安装 sklearn，跳过 DBSCAN 对比")

    try:
        from sklearn.cluster import HDBSCAN as SKHDBSCAN
        sk_hdb = SKHDBSCAN(min_cluster_size=10, min_samples=5)
        sk_hdb_labels = sk_hdb.fit_predict(X2)
        print(f"\nsklearn HDBSCAN 结果:")
        print(f"  簇数量: {sk_hdb_labels.max() + 1 if sk_hdb_labels.max() >= 0 else 0}")
        print(f"  噪声点: {np.sum(sk_hdb_labels == -1)}")
        print(f"  高密度簇识别: {np.sum(sk_hdb_labels[:80] != -1)}/80")
        print(f"  低密度簇识别: {np.sum(sk_hdb_labels[80:] != -1)}/40")
        try:
            from sklearn.metrics import silhouette_score as sk_sil
            sk_score = sk_sil(X2, sk_hdb_labels)
            print(f"  sklearn 轮廓系数: {sk_score:.4f}")
            print(f"  我们的轮廓系数: {info_h['silhouette_score']:.4f}")
        except ImportError:
            pass
    except ImportError:
        print("未安装 sklearn HDBSCAN，跳过 HDBSCAN 对比")
