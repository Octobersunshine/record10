import numpy as np
import heapq
import json
from collections import deque
from scipy.spatial.distance import pdist, squareform


def agglomerative_clustering(X, n_clusters=2, linkage='single', metric='euclidean',
                             memory_efficient='auto'):
    """
    凝聚层次聚类（自底向上）

    支持两种实现:
    - 标准版本: O(n²) 内存，速度快，适合小样本
    - 内存优化版本: O(n) 内存，基于最近邻链/Prim算法，适合大样本

    参数:
        X: 样本数据矩阵, shape=(n_samples, n_features)
        n_clusters: 目标聚类数
        linkage: 连接准则, 'single'（单链）、'complete'（全链）、'average'（平均链）
        metric: 距离度量, 默认为'euclidean'
        memory_efficient: 是否使用内存优化版本
            'auto': n_samples > 500 时自动启用
            True/False: 强制启用/禁用

    返回:
        Z: 聚类树（树状图数据）, shape=(n_samples-1, 4)
           每行: [簇i, 簇j, 距离, 新簇大小]
        labels: 聚类标签, shape=(n_samples,)
    """
    n_samples = X.shape[0]

    if memory_efficient == 'auto':
        memory_efficient = n_samples > 500

    if memory_efficient:
        if metric != 'euclidean':
            raise ValueError("内存优化版本仅支持 'euclidean' 距离")
        return _agglomerative_fast(X, n_clusters, linkage)
    else:
        return _agglomerative_standard(X, n_clusters, linkage, metric)


def _agglomerative_standard(X, n_clusters, linkage, metric):
    """
    标准版本 - 使用完整距离矩阵
    内存复杂度: O(n²)，速度快，适合小样本
    """
    n_samples = X.shape[0]

    dist_matrix = squareform(pdist(X, metric=metric))

    clusters = {i: [i] for i in range(n_samples)}
    cluster_ids = list(range(n_samples))
    next_cluster_id = n_samples

    Z = np.zeros((n_samples - 1, 4))
    label_cluster_ids = None

    for step in range(n_samples - 1):
        min_dist = np.inf
        c_i, c_j = -1, -1

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                ci_id = cluster_ids[i]
                cj_id = cluster_ids[j]
                dist = _calculate_linkage_distance(
                    dist_matrix, clusters[ci_id], clusters[cj_id], linkage
                )
                if dist < min_dist:
                    min_dist = dist
                    c_i, c_j = ci_id, cj_id

        if len(cluster_ids) == n_clusters and label_cluster_ids is None:
            label_cluster_ids = cluster_ids.copy()

        merged = clusters[c_i] + clusters[c_j]
        clusters[next_cluster_id] = merged
        cluster_ids.remove(c_i)
        cluster_ids.remove(c_j)
        cluster_ids.append(next_cluster_id)

        Z[step] = [c_i, c_j, min_dist, len(merged)]
        next_cluster_id += 1

    if label_cluster_ids is None:
        label_cluster_ids = cluster_ids

    labels = np.zeros(n_samples, dtype=int)
    for label_idx, cluster_id in enumerate(label_cluster_ids):
        for sample_idx in clusters[cluster_id]:
            labels[sample_idx] = label_idx

    return Z, labels


def _calculate_linkage_distance(dist_matrix, cluster1, cluster2, linkage):
    """计算两个簇之间的连接距离 - 标准版本"""
    distances = []
    for i in cluster1:
        for j in cluster2:
            distances.append(dist_matrix[i, j])

    if linkage == 'single':
        return min(distances)
    elif linkage == 'complete':
        return max(distances)
    elif linkage == 'average':
        return np.mean(distances)
    else:
        raise ValueError(f"不支持的连接准则: {linkage}")


def _agglomerative_fast(X, n_clusters, linkage):
    """
    内存优化版本
    - single: 类Prim算法 + Kruskal MST
    - complete/average: 最近邻链算法
    """
    if linkage == 'single':
        return _agglomerative_single_link(X, n_clusters)
    elif linkage in ['complete', 'average']:
        return _agglomerative_nn_chain(X, n_clusters, linkage)
    else:
        raise ValueError(f"不支持的连接准则: {linkage}")


def _agglomerative_single_link(X, n_clusters):
    """
    单链聚类 - 类Prim算法构建MST + Kruskal合并
    内存复杂度: O(n)，时间复杂度: O(n²)
    """
    n_samples = X.shape[0]

    next_cluster_id = n_samples

    Z = np.zeros((n_samples - 1, 4))

    nn_dist = np.full(n_samples, np.inf)
    nn_idx = np.full(n_samples, -1)
    in_mst = np.zeros(n_samples, dtype=bool)

    start = 0
    in_mst[start] = True

    for i in range(n_samples):
        if i != start:
            diff = X[start] - X[i]
            nn_dist[i] = np.sqrt(np.sum(diff ** 2))
            nn_idx[i] = start

    mst_edges = []

    for _ in range(n_samples - 1):
        min_dist = np.inf
        u = -1
        for i in range(n_samples):
            if not in_mst[i] and nn_dist[i] < min_dist:
                min_dist = nn_dist[i]
                u = i

        if u == -1:
            break

        in_mst[u] = True
        v = nn_idx[u]
        mst_edges.append((min_dist, u, v))

        for w in range(n_samples):
            if not in_mst[w]:
                diff = X[u] - X[w]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < nn_dist[w]:
                    nn_dist[w] = dist
                    nn_idx[w] = u

    mst_edges.sort(key=lambda x: x[0])

    parent = list(range(n_samples))
    size = [1] * n_samples

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cluster_map = {i: i for i in range(n_samples)}
    next_new_id = n_samples
    num_clusters = n_samples
    labels = None

    for step, (dist, u, v) in enumerate(mst_edges):
        if num_clusters == n_clusters and labels is None:
            labels = np.zeros(n_samples, dtype=int)
            root_to_label = {}
            label_counter = 0
            for i in range(n_samples):
                root = find(i)
                if root not in root_to_label:
                    root_to_label[root] = label_counter
                    label_counter += 1
                labels[i] = root_to_label[root]

        root_u = find(u)
        root_v = find(v)

        if root_u != root_v:
            orig_u = cluster_map[root_u]
            orig_v = cluster_map[root_v]

            parent[root_v] = root_u
            size[root_u] += size[root_v]

            Z[step] = [orig_u, orig_v, dist, size[root_u]]
            cluster_map[root_u] = next_new_id
            next_new_id += 1
            num_clusters -= 1

    if labels is None:
        labels = np.zeros(n_samples, dtype=int)

    return Z, labels


def _agglomerative_nn_chain(X, n_clusters, linkage):
    """
    全链/平均链 - 最近邻链算法
    内存复杂度: O(n)
    """
    n_samples = X.shape[0]

    clusters = {i: {'indices': [i], 'size': 1} for i in range(n_samples)}
    active_clusters = set(range(n_samples))
    next_cluster_id = n_samples

    Z = np.zeros((n_samples - 1, 4))
    labels = None

    nn_cache = {}

    step = 0
    while len(active_clusters) > 1:
        if len(active_clusters) == n_clusters and labels is None:
            labels = np.zeros(n_samples, dtype=int)
            for label_idx, cluster_id in enumerate(active_clusters):
                for sample_idx in clusters[cluster_id]['indices']:
                    labels[sample_idx] = label_idx

        chain = []
        start_c = next(iter(active_clusters))
        chain.append(start_c)

        while True:
            current = chain[-1]

            if current in nn_cache and nn_cache[current][1] in active_clusters:
                nn_dist, nn = nn_cache[current]
            else:
                nn_dist, nn = _find_nearest_neighbor(
                    X, clusters, current, active_clusters, linkage
                )
                nn_cache[current] = (nn_dist, nn)

            if len(chain) >= 2 and nn == chain[-2]:
                c_i = chain[-2]
                c_j = chain[-1]
                merge_dist = nn_dist
                break
            else:
                chain.append(nn)

        merged_indices = clusters[c_i]['indices'] + clusters[c_j]['indices']
        merged_size = clusters[c_i]['size'] + clusters[c_j]['size']
        clusters[next_cluster_id] = {
            'indices': merged_indices,
            'size': merged_size
        }

        Z[step] = [c_i, c_j, merge_dist, merged_size]
        step += 1

        active_clusters.remove(c_i)
        active_clusters.remove(c_j)
        active_clusters.add(next_cluster_id)

        if c_i in nn_cache:
            del nn_cache[c_i]
        if c_j in nn_cache:
            del nn_cache[c_j]

        del clusters[c_i]
        del clusters[c_j]

        next_cluster_id += 1

    if labels is None:
        labels = np.zeros(n_samples, dtype=int)

    return Z, labels


def _find_nearest_neighbor(X, clusters, current_id, active_clusters, linkage):
    """找到当前簇的最近邻 - 延迟计算距离"""
    current_cluster = clusters[current_id]
    min_dist = np.inf
    nearest = None

    for other_id in active_clusters:
        if other_id == current_id:
            continue

        dist = _compute_linkage_distance_fast(
            X, current_cluster, clusters[other_id], linkage
        )
        if dist < min_dist:
            min_dist = dist
            nearest = other_id

    return min_dist, nearest


def _compute_linkage_distance_fast(X, cluster1, cluster2, linkage):
    """计算两个簇之间的连接距离 - 内存优化版本"""
    indices1 = cluster1['indices']
    indices2 = cluster2['indices']

    if linkage == 'single':
        min_dist = np.inf
        for i in indices1:
            for j in indices2:
                diff = X[i] - X[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    elif linkage == 'complete':
        max_dist = 0
        for i in indices1:
            for j in indices2:
                diff = X[i] - X[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    elif linkage == 'average':
        total_dist = 0.0
        count = 0
        for i in indices1:
            for j in indices2:
                diff = X[i] - X[j]
                total_dist += np.sqrt(np.sum(diff ** 2))
                count += 1
        return total_dist / count if count > 0 else 0

    else:
        raise ValueError(f"不支持的连接准则: {linkage}")


def divisive_clustering(X, n_clusters=2, linkage='average', metric='euclidean',
                        max_dist=None, min_cluster_size=1):
    """
    分裂层次聚类（自顶向下，DIANA算法）

    DIANA (Divisive Analysis): 从一个包含所有样本的簇开始，
    每次选择一个簇分裂为两个子簇，直到达到目标聚类数或停止阈值。

    参数:
        X: 样本数据矩阵, shape=(n_samples, n_features)
        n_clusters: 目标聚类数
        linkage: 连接准则, 'single'、'complete'、'average'
        metric: 距离度量, 默认为'euclidean'
        max_dist: 分裂停止阈值（最大允许簇内直径）
            当簇的直径小于此值时不再分裂
        min_cluster_size: 最小簇大小
            簇大小小于此值时不再分裂

    返回:
        tree_data: 分裂树结构字典
            - root: 根节点ID
            - nodes: 所有节点信息
            - splits: 分裂历史记录
            - n_leaves: 叶节点（最终簇）数量
        labels: 聚类标签, shape=(n_samples,)
    """
    n_samples = X.shape[0]

    dist_matrix = squareform(pdist(X, metric=metric))

    clusters = [list(range(n_samples))]
    cluster_tree = []
    labels = None

    next_node_id = 1
    node_info = {0: {'indices': list(range(n_samples)), 'parent': None,
                     'split_distance': 0, 'children': []}}

    while len(clusters) < n_clusters:
        cluster_to_split = None
        max_diameter = -1
        cluster_idx_in_list = -1

        for idx, cluster in enumerate(clusters):
            if len(cluster) <= min_cluster_size:
                continue

            diameter = _compute_cluster_diameter(dist_matrix, cluster)

            if max_dist is not None and diameter < max_dist:
                continue

            if diameter > max_diameter:
                max_diameter = diameter
                cluster_to_split = cluster
                cluster_idx_in_list = idx

        if cluster_to_split is None:
            break

        subcluster1, subcluster2, split_dist = _diana_split(
            dist_matrix, cluster_to_split, linkage
        )

        if len(subcluster1) == 0 or len(subcluster2) == 0:
            break

        parent_id = _find_node_id(node_info, cluster_to_split)

        node1_id = next_node_id
        next_node_id += 1
        node2_id = next_node_id
        next_node_id += 1

        node_info[node1_id] = {
            'indices': subcluster1,
            'parent': parent_id,
            'split_distance': split_dist,
            'children': []
        }
        node_info[node2_id] = {
            'indices': subcluster2,
            'parent': parent_id,
            'split_distance': split_dist,
            'children': []
        }
        node_info[parent_id]['children'] = [node1_id, node2_id]

        cluster_tree.append({
            'parent': parent_id,
            'children': [node1_id, node2_id],
            'distance': split_dist,
            'sizes': [len(subcluster1), len(subcluster2)]
        })

        clusters.pop(cluster_idx_in_list)
        clusters.append(subcluster1)
        clusters.append(subcluster2)

    labels = np.zeros(n_samples, dtype=int)
    for label_idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            labels[sample_idx] = label_idx

    tree_data = {
        'root': 0,
        'nodes': node_info,
        'splits': cluster_tree,
        'n_leaves': len(clusters)
    }

    return tree_data, labels


def _find_node_id(node_info, indices):
    """根据样本索引找到对应的节点ID"""
    indices_set = set(indices)
    for node_id, info in node_info.items():
        if set(info['indices']) == indices_set:
            return node_id
    return None


def _compute_cluster_diameter(dist_matrix, cluster):
    """计算簇的直径（簇内最大距离）"""
    if len(cluster) <= 1:
        return 0
    max_dist = 0
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            dist = dist_matrix[cluster[i], cluster[j]]
            if dist > max_dist:
                max_dist = dist
    return max_dist


def _diana_split(dist_matrix, cluster, linkage='average'):
    """
    DIANA分裂算法: 将一个簇分裂为两个子簇

    步骤:
    1. 找到簇内与其他样本平均距离最大的样本作为"分裂种子"
    2. 依次将样本分配到距离更近的子簇
    """
    n = len(cluster)
    if n <= 1:
        return cluster, [], 0

    avg_dists = np.zeros(n)
    for i in range(n):
        total = 0
        for j in range(n):
            if i != j:
                total += dist_matrix[cluster[i], cluster[j]]
        avg_dists[i] = total / (n - 1) if n > 1 else 0

    seed_idx = np.argmax(avg_dists)
    seed = cluster[seed_idx]

    splinter_group = [seed]
    old_group = [cluster[i] for i in range(n) if i != seed_idx]

    while True:
        moved = False
        max_diff = -np.inf
        move_idx = -1

        for idx, point in enumerate(old_group):
            dist_to_splinter = 0
            for s in splinter_group:
                dist_to_splinter += dist_matrix[point, s]
            dist_to_splinter /= len(splinter_group)

            dist_to_old = 0
            for o in old_group:
                if o != point:
                    dist_to_old += dist_matrix[point, o]
            if len(old_group) > 1:
                dist_to_old /= (len(old_group) - 1)
            else:
                dist_to_old = np.inf

            diff = dist_to_old - dist_to_splinter
            if diff > max_diff and diff > 0:
                max_diff = diff
                move_idx = idx

        if move_idx >= 0:
            point_to_move = old_group.pop(move_idx)
            splinter_group.append(point_to_move)
            moved = True

        if not moved or len(old_group) == 0 or len(splinter_group) == 0:
            break

    if len(splinter_group) == 0 or len(old_group) == 0:
        mid = n // 2
        splinter_group = cluster[:mid]
        old_group = cluster[mid:]

    split_dist = _compute_linkage_dist_between(
        dist_matrix, splinter_group, old_group, linkage
    )

    return splinter_group, old_group, split_dist


def _compute_linkage_dist_between(dist_matrix, cluster1, cluster2, linkage):
    """计算两个簇之间的连接距离"""
    if len(cluster1) == 0 or len(cluster2) == 0:
        return 0

    distances = []
    for i in cluster1:
        for j in cluster2:
            distances.append(dist_matrix[i, j])

    if linkage == 'single':
        return min(distances)
    elif linkage == 'complete':
        return max(distances)
    elif linkage == 'average':
        return np.mean(distances)
    else:
        raise ValueError(f"不支持的连接准则: {linkage}")


def compute_linkage_distances(Z, n_samples):
    """
    计算聚类树各层的连接距离，用于确定聚类停止阈值

    参数:
        Z: 凝聚聚类的树状图矩阵, shape=(n_samples-1, 4)
        n_samples: 样本总数

    返回:
        distances: 各次合并的距离数组, shape=(n_samples-1,)
        cluster_sizes: 各次合并后的簇大小数组
        suggested_thresholds: 建议的停止阈值字典
            - elbow: 肘点（距离梯度最大处）
            - gap: 最大间隔处
            - mean: 平均距离
            - median: 中位数距离
    """
    distances = Z[:, 2]
    cluster_sizes = Z[:, 3]

    if len(distances) > 2:
        gradients = np.diff(distances)

        elbow_idx = np.argmax(gradients)
        elbow_threshold = (distances[elbow_idx] + distances[elbow_idx + 1]) / 2

        gap_idx = np.argmax(gradients)
        gap_threshold = distances[gap_idx + 1]
    else:
        elbow_threshold = np.mean(distances) if len(distances) > 0 else 0
        gap_threshold = elbow_threshold

    suggested_thresholds = {
        'elbow': float(elbow_threshold),
        'gap': float(gap_threshold),
        'mean': float(np.mean(distances)) if len(distances) > 0 else 0,
        'median': float(np.median(distances)) if len(distances) > 0 else 0
    }

    return distances, cluster_sizes, suggested_thresholds


def tree_to_json(Z, n_samples, labels=None, X=None, max_depth=None):
    """
    将凝聚层次聚类树转换为JSON格式的可视化数据

    参数:
        Z: 聚类树矩阵, shape=(n_samples-1, 4)
        n_samples: 样本总数
        labels: 聚类标签数组（可选）
        X: 样本数据矩阵（可选，用于导出样本坐标）
        max_depth: 最大导出深度（可选）

    返回:
        json_data: 可直接用于可视化的JSON格式字典
    """
    root_id = 2 * n_samples - 2

    node_map = {}
    for i in range(n_samples):
        node_map[i] = {
            'id': i,
            'name': f'sample_{i}',
            'type': 'leaf',
            'distance': 0,
            'size': 1,
            'children': [],
            'sample_index': i
        }
        if labels is not None:
            node_map[i]['cluster'] = int(labels[i])
        if X is not None:
            node_map[i]['coordinates'] = X[i].tolist()

    for step in range(n_samples - 1):
        c1, c2, dist, size = Z[step]
        c1, c2 = int(c1), int(c2)
        node_id = n_samples + step

        node_map[node_id] = {
            'id': node_id,
            'name': f'cluster_{node_id}',
            'type': 'internal',
            'distance': float(dist),
            'size': int(size),
            'children': [c1, c2]
        }

    def build_tree(node_id, depth=0):
        node = dict(node_map[node_id])
        if max_depth is not None and depth >= max_depth:
            node['children'] = []
            return node

        children = []
        for child_id in node_map[node_id]['children']:
            child_tree = build_tree(child_id, depth + 1)
            children.append(child_tree)
        node['children'] = children
        return node

    tree_data = build_tree(root_id)

    distances = Z[:, 2].tolist() if len(Z) > 0 else []
    sizes = Z[:, 3].tolist() if len(Z) > 0 else []

    json_data = {
        'version': '1.0',
        'n_samples': n_samples,
        'n_clusters': int(len(np.unique(labels))) if labels is not None else 1,
        'root': tree_data,
        'linkage_distances': [float(d) for d in distances],
        'cluster_sizes': [int(s) for s in sizes],
        'metadata': {
            'algorithm': 'agglomerative_hierarchical_clustering',
            'node_count': int(2 * n_samples - 1),
            'max_distance': float(max(distances)) if distances else 0
        }
    }

    if labels is not None:
        unique_labels = np.unique(labels)
        cluster_info = {}
        for lbl in unique_labels:
            indices = np.where(labels == lbl)[0].tolist()
            cluster_info[int(lbl)] = {
                'size': len(indices),
                'samples': indices
            }
        json_data['clusters'] = cluster_info

    return json_data


def divisive_tree_to_json(tree_data, labels=None, X=None):
    """
    将分裂层次聚类树转换为JSON格式的可视化数据

    参数:
        tree_data: divisive_clustering返回的树结构
        labels: 聚类标签数组（可选）
        X: 样本数据矩阵（可选）

    返回:
        json_data: JSON格式的可视化数据字典
    """
    nodes = tree_data['nodes']

    def build_tree(node_id):
        node = nodes[node_id]
        is_leaf = len(node['children']) == 0

        result = {
            'id': int(node_id),
            'name': f'cluster_{node_id}' if not is_leaf else f'leaf_{node_id}',
            'type': 'internal' if not is_leaf else 'leaf',
            'distance': float(node['split_distance']),
            'size': int(len(node['indices'])),
            'children': [],
            'sample_indices': node['indices']
        }

        if is_leaf and labels is not None and len(node['indices']) > 0:
            sample_idx = node['indices'][0]
            result['cluster'] = int(labels[sample_idx])

        if is_leaf and X is not None and len(node['indices']) > 0:
            if len(node['indices']) == 1:
                result['coordinates'] = X[node['indices'][0]].tolist()
            else:
                result['coordinates'] = np.mean(
                    [X[i] for i in node['indices']], axis=0
                ).tolist()

        for child_id in node['children']:
            result['children'].append(build_tree(child_id))

        return result

    root_tree = build_tree(tree_data['root'])

    splits = []
    for split in tree_data['splits']:
        splits.append({
            'parent': int(split['parent']),
            'children': [int(c) for c in split['children']],
            'distance': float(split['distance']),
            'sizes': [int(s) for s in split['sizes']]
        })

    json_data = {
        'version': '1.0',
        'algorithm': 'diana_divisive_clustering',
        'n_samples': int(len(labels)) if labels is not None else 0,
        'n_clusters': int(tree_data['n_leaves']),
        'root': root_tree,
        'splits': splits,
        'metadata': {
            'node_count': int(len(nodes)),
            'max_split_distance': float(max(
                [s['distance'] for s in splits]
            )) if splits else 0
        }
    }

    if labels is not None:
        unique_labels = np.unique(labels)
        cluster_info = {}
        for lbl in unique_labels:
            indices = np.where(labels == lbl)[0].tolist()
            cluster_info[int(lbl)] = {
                'size': len(indices),
                'samples': indices
            }
        json_data['clusters'] = cluster_info

    return json_data
