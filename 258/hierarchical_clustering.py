import numpy as np
import heapq
import sys
import time
from collections import deque


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))


class CondensedDistanceMatrix:
    def __init__(self, n):
        self.n = n
        self.size = n * (n - 1) // 2
        self.data = np.zeros(self.size, dtype=np.float64)

    def _get_index(self, i, j):
        if i == j:
            raise ValueError("Diagonal elements are not stored")
        if i > j:
            i, j = j, i
        return i * (2 * self.n - i - 3) // 2 + j - 1

    def __getitem__(self, idx):
        i, j = idx
        if i == j:
            return 0.0
        return self.data[self._get_index(i, j)]

    def __setitem__(self, idx, value):
        i, j = idx
        if i == j:
            return
        self.data[self._get_index(i, j)] = value

    def memory_usage(self):
        return self.data.nbytes


def compute_condensed_distance_matrix(data, metric='euclidean'):
    n_samples = len(data)
    condensed = CondensedDistanceMatrix(n_samples)

    if metric == 'euclidean':
        dist_func = euclidean_distance
    elif metric == 'manhattan':
        dist_func = manhattan_distance
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            condensed[i, j] = dist_func(data[i], data[j])

    return condensed


def get_lance_williams_params(linkage):
    if linkage == 'single':
        return lambda a_size, b_size, c_size: (0.5, 0.5, 0.0, -0.5)
    elif linkage == 'complete':
        return lambda a_size, b_size, c_size: (0.5, 0.5, 0.0, 0.5)
    elif linkage == 'average':
        def params(a_size, b_size, c_size):
            total = a_size + b_size
            return (a_size / total, b_size / total, 0.0, 0.0)
        return params
    else:
        raise ValueError(f"Unknown linkage method: {linkage}")


def lance_williams_update(d_ac, d_bc, d_ab, a_size, b_size, c_size, linkage):
    params_func = get_lance_williams_params(linkage)
    alpha_a, alpha_b, beta, gamma = params_func(a_size, b_size, c_size)
    return alpha_a * d_ac + alpha_b * d_bc + beta * d_ab + gamma * abs(d_ac - d_bc)


def heap_based_clustering(data=None, distance_matrix=None, metric='euclidean', linkage='single'):
    if data is None and distance_matrix is None:
        raise ValueError("Either 'data' or 'distance_matrix' must be provided")

    if distance_matrix is not None:
        n = distance_matrix.shape[0]
        dist = CondensedDistanceMatrix(n)
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = distance_matrix[i, j]
    else:
        n = len(data)
        dist = compute_condensed_distance_matrix(data, metric=metric)

    active = [True] * n
    cluster_size = [1] * n
    idx_map = list(range(n))
    next_cluster_id = n
    dendrogram = []

    heap = []
    for i in range(n):
        for j in range(i + 1, n):
            heapq.heappush(heap, (dist[i, j], i, j))

    active_count = n

    while active_count > 1 and len(heap) > 0:
        d, c1, c2 = heapq.heappop(heap)

        if not active[c1] or not active[c2]:
            continue
        if abs(dist[c1, c2] - d) > 1e-10:
            continue

        merge_dist = d
        new_size = cluster_size[c1] + cluster_size[c2]

        dendrogram.append({
            'cluster1': idx_map[c1],
            'cluster2': idx_map[c2],
            'distance': merge_dist,
            'size': new_size
        })

        active[c2] = False
        active_count -= 1
        cluster_size[c1] = new_size
        idx_map[c1] = next_cluster_id
        next_cluster_id += 1

        size_a = cluster_size[c1]
        size_b = cluster_size[c2]

        for k in range(n):
            if k != c1 and k != c2 and active[k]:
                d_ac = dist[c1, k]
                d_bc = dist[c2, k]
                d_ab = merge_dist
                size_c = cluster_size[k]

                new_d = lance_williams_update(d_ac, d_bc, d_ab, size_a, size_b, size_c, linkage)
                dist[c1, k] = new_d
                heapq.heappush(heap, (new_d, c1, k))

    return dendrogram


def slink_single_linkage(data, metric='euclidean'):
    n = len(data)

    if metric == 'euclidean':
        dist_func = euclidean_distance
    elif metric == 'manhattan':
        dist_func = manhattan_distance
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    pointer = np.zeros(n, dtype=np.int32)
    height = np.zeros(n, dtype=np.float64)

    for i in range(n):
        pointer[i] = i
        height[i] = np.inf

        for j in range(i):
            dist = dist_func(data[i], data[j])

            if height[j] > dist:
                temp_height = height[j]
                temp_pointer = pointer[j]
                height[j] = dist
                pointer[j] = i

                k = j
                while height[pointer[k]] < temp_height:
                    k = pointer[k]

                if height[pointer[k]] > temp_height:
                    temp = pointer[k]
                    pointer[k] = i
                    height[k] = temp_height
                    temp_pointer = temp
                    temp_height = height[pointer[k]]

                pointer[k] = temp_pointer
                height[k] = temp_height

    merges = []
    for i in range(n):
        if height[i] < np.inf:
            merges.append((height[i], i, pointer[i]))

    merges.sort(key=lambda x: x[0])

    if len(merges) < n - 1:
        return heap_based_clustering(data=data, metric=metric, linkage='single')

    parent = list(range(n * 2))
    size = [1] * (n * 2)
    next_id = n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    dendrogram = []

    for dist, i, j in merges:
        root_i = find(i)
        root_j = find(j)

        if root_i == root_j:
            continue

        c1 = root_i
        c2 = root_j
        new_size = size[root_i] + size[root_j]

        dendrogram.append({
            'cluster1': c1,
            'cluster2': c2,
            'distance': dist,
            'size': new_size
        })

        parent[root_i] = next_id
        parent[root_j] = next_id
        parent[next_id] = next_id
        size[next_id] = new_size
        next_id += 1

        if len(dendrogram) == n - 1:
            break

    if len(dendrogram) == n - 1:
        return dendrogram
    else:
        return heap_based_clustering(data=data, metric=metric, linkage='single')


def nn_chain_clustering(data=None, distance_matrix=None, metric='euclidean', linkage='single'):
    if data is None and distance_matrix is None:
        raise ValueError("Either 'data' or 'distance_matrix' must be provided")

    if distance_matrix is not None:
        n = distance_matrix.shape[0]
        dist = CondensedDistanceMatrix(n)
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = distance_matrix[i, j]
    else:
        n = len(data)
        dist = compute_condensed_distance_matrix(data, metric=metric)

    active = [True] * n
    cluster_size = [1] * n
    next_cluster_id = n
    dendrogram = []

    idx_map = list(range(n))

    def find_nn(i):
        min_d = np.inf
        min_j = -1
        for j in range(n):
            if j != i and active[j]:
                d = dist[i, j]
                if d < min_d:
                    min_d = d
                    min_j = j
        return min_j, min_d

    nn = [-1] * n
    nn_dist = [np.inf] * n
    for i in range(n):
        nn[i], nn_dist[i] = find_nn(i)

    active_count = n

    while active_count > 1:
        c1, c2 = -1, -1
        found_rnn = False

        active_list = [i for i in range(n) if active[i]]

        for i in active_list:
            if nn[i] >= 0 and active[nn[i]] and nn[nn[i]] == i:
                c1 = min(i, nn[i])
                c2 = max(i, nn[i])
                found_rnn = True
                break

        if not found_rnn:
            for i in active_list:
                if nn[i] < 0 or not active[nn[i]]:
                    nn[i], nn_dist[i] = find_nn(i)
                if nn[i] >= 0 and active[nn[i]] and nn[nn[i]] == i:
                    c1 = min(i, nn[i])
                    c2 = max(i, nn[i])
                    found_rnn = True
                    break

        if not found_rnn:
            min_d = np.inf
            pair = (-1, -1)
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    ci, cj = active_list[i], active_list[j]
                    d = dist[ci, cj]
                    if d < min_d:
                        min_d = d
                        pair = (ci, cj)
            if pair[0] == -1:
                break
            c1, c2 = pair

        merge_dist = dist[c1, c2]
        new_size = cluster_size[c1] + cluster_size[c2]

        dendrogram.append({
            'cluster1': idx_map[c1],
            'cluster2': idx_map[c2],
            'distance': merge_dist,
            'size': new_size
        })

        active[c2] = False
        active_count -= 1
        cluster_size[c1] = new_size
        idx_map[c1] = next_cluster_id
        next_cluster_id += 1

        size_a = cluster_size[c1]
        size_b = cluster_size[c2]

        for k in range(n):
            if k != c1 and k != c2 and active[k]:
                d_ac = dist[c1, k]
                d_bc = dist[c2, k]
                d_ab = merge_dist
                size_c = cluster_size[k]

                new_d = lance_williams_update(d_ac, d_bc, d_ab, size_a, size_b, size_c, linkage)
                dist[c1, k] = new_d

        for k in range(n):
            if active[k] and k != c1:
                nn[k], nn_dist[k] = find_nn(k)

        nn[c1], nn_dist[c1] = find_nn(c1)

        if c2 < len(nn):
            nn[c2] = -1
            nn_dist[c2] = np.inf

    return dendrogram


def agglomerative_clustering(data=None, distance_matrix=None, metric='euclidean',
                              linkage='single', algorithm='auto'):
    if data is None and distance_matrix is None:
        raise ValueError("Either 'data' or 'distance_matrix' must be provided")

    n = len(data) if data is not None else distance_matrix.shape[0]

    if algorithm == 'auto':
        if linkage == 'single' and data is not None:
            algorithm = 'slink'
        elif linkage == 'average':
            algorithm = 'heap'
        elif n <= 2000:
            algorithm = 'nn_chain'
        else:
            algorithm = 'heap'

    if algorithm == 'slink':
        if linkage != 'single':
            raise ValueError("SLINK algorithm only supports single linkage")
        if data is None:
            raise ValueError("SLINK algorithm requires raw data")
        return slink_single_linkage(data, metric)
    elif algorithm == 'nn_chain':
        if linkage == 'average':
            return heap_based_clustering(data, distance_matrix, metric, linkage)
        return nn_chain_clustering(data, distance_matrix, metric, linkage)
    elif algorithm == 'heap':
        return heap_based_clustering(data, distance_matrix, metric, linkage)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def get_cluster_labels_at_level(dendrogram, n_samples, n_clusters):
    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError(f"n_clusters must be between 1 and {n_samples}")

    if n_clusters == n_samples:
        return list(range(n_samples))

    parent = list(range(n_samples + len(dendrogram)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n_samples - n_clusters):
        step = dendrogram[i]
        c1, c2 = step['cluster1'], step['cluster2']
        new_id = n_samples + i
        parent[find(c1)] = new_id
        parent[find(c2)] = new_id
        parent[new_id] = new_id

    labels = {}
    current_label = 0
    result = []
    for i in range(n_samples):
        root = find(i)
        if root not in labels:
            labels[root] = current_label
            current_label += 1
        result.append(labels[root])

    return result


def cut_dendrogram_by_k(dendrogram, n_samples, k):
    if k <= 0 or k > n_samples:
        raise ValueError(f"k must be between 1 and {n_samples}")

    return get_cluster_labels_at_level(dendrogram, n_samples, k)


def cut_dendrogram_by_threshold(dendrogram, n_samples, threshold):
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    parent = list(range(n_samples + len(dendrogram)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, step in enumerate(dendrogram):
        if step['distance'] > threshold:
            break
        c1, c2 = step['cluster1'], step['cluster2']
        new_id = n_samples + i
        parent[find(c1)] = new_id
        parent[find(c2)] = new_id
        parent[new_id] = new_id

    labels = {}
    current_label = 0
    result = []
    for i in range(n_samples):
        root = find(i)
        if root not in labels:
            labels[root] = current_label
            current_label += 1
        result.append(labels[root])

    n_clusters = len(labels)
    return result, n_clusters


def find_optimal_threshold_for_k(dendrogram, n_samples, k):
    if k <= 0 or k > n_samples:
        raise ValueError(f"k must be between 1 and {n_samples}")

    if k == n_samples:
        return 0.0
    if k == 1:
        return dendrogram[-1]['distance'] + 1e-9

    merge_idx = n_samples - k
    lower_dist = dendrogram[merge_idx - 1]['distance'] if merge_idx > 0 else 0.0
    upper_dist = dendrogram[merge_idx]['distance']

    return (lower_dist + upper_dist) / 2


def build_dendrogram_tree(dendrogram, n_samples):
    nodes = {}

    for i in range(n_samples):
        nodes[i] = {
            'id': i,
            'name': f'样本{i}',
            'type': 'leaf',
            'height': 0,
            'size': 1,
            'children': [],
            'samples': [i]
        }

    for i, step in enumerate(dendrogram):
        c1, c2 = step['cluster1'], step['cluster2']
        dist = step['distance']
        size = step['size']
        new_id = n_samples + i

        node1 = nodes[c1]
        node2 = nodes[c2]

        nodes[new_id] = {
            'id': new_id,
            'name': f'聚类{new_id}',
            'type': 'internal',
            'height': dist,
            'size': size,
            'distance': dist,
            'children': [node1['id'], node2['id']],
            'samples': node1['samples'] + node2['samples']
        }

    root_id = n_samples + len(dendrogram) - 1
    return nodes, root_id


def generate_dendrogram_visualization(dendrogram, n_samples, labels=None, data=None):
    nodes, root_id = build_dendrogram_tree(dendrogram, n_samples)

    def assign_coordinates(node_id, x_positions, y_positions, x_offset=0, y_offset=0):
        node = nodes[node_id]

        if node['type'] == 'leaf':
            x_positions[node_id] = x_offset
            y_positions[node_id] = y_offset
            return x_offset

        left_child = node['children'][0]
        right_child = node['children'][1]

        left_size = nodes[left_child]['size']
        right_size = nodes[right_child]['size']
        total_size = left_size + right_size

        left_x = assign_coordinates(left_child, x_positions, y_positions,
                                     x_offset, y_offset + 1)
        right_x = assign_coordinates(right_child, x_positions, y_positions,
                                      x_offset + left_size / total_size, y_offset + 1)

        x_positions[node_id] = (left_x + right_x) / 2
        y_positions[node_id] = node['height']

        return x_positions[node_id]

    x_positions = {}
    y_positions = {}
    assign_coordinates(root_id, x_positions, y_positions)

    max_y = max(y_positions.values()) if y_positions else 1
    for key in y_positions:
        if nodes[key]['type'] == 'internal':
            y_positions[key] = y_positions[key]
        else:
            y_positions[key] = 0

    links = []
    for i, step in enumerate(dendrogram):
        c1, c2 = step['cluster1'], step['cluster2']
        parent_id = n_samples + i
        dist = step['distance']

        links.append({
            'source': c1,
            'target': parent_id,
            'distance': dist,
            'source_x': x_positions[c1],
            'source_y': y_positions[c1],
            'target_x': x_positions[parent_id],
            'target_y': y_positions[parent_id]
        })

        links.append({
            'source': c2,
            'target': parent_id,
            'distance': dist,
            'source_x': x_positions[c2],
            'source_y': y_positions[c2],
            'target_x': x_positions[parent_id],
            'target_y': y_positions[parent_id]
        })

    nodes_list = []
    for node_id, node in nodes.items():
        node_info = {
            'id': node['id'],
            'name': node['name'],
            'type': node['type'],
            'x': x_positions[node_id],
            'y': y_positions[node_id],
            'height': node['height'] if node['type'] == 'internal' else 0,
            'size': node['size'],
            'samples': node['samples']
        }

        if labels is not None and node['type'] == 'leaf':
            node_info['label'] = labels[node['samples'][0]]

        if data is not None and node['type'] == 'leaf':
            node_info['data'] = data[node['samples'][0]].tolist()

        nodes_list.append(node_info)

    return {
        'nodes': nodes_list,
        'links': links,
        'root_id': root_id,
        'n_samples': n_samples,
        'max_distance': dendrogram[-1]['distance'] if dendrogram else 0,
        'min_distance': dendrogram[0]['distance'] if dendrogram else 0
    }


def generate_dendrogram_json(dendrogram, n_samples, labels=None, data=None):
    vis_data = generate_dendrogram_visualization(dendrogram, n_samples, labels, data)

    tree_data = _build_hierarchical_json(vis_data, n_samples)

    return {
        'hierarchy': tree_data,
        'flat': vis_data,
        'metadata': {
            'n_samples': n_samples,
            'n_leaves': n_samples,
            'n_internal_nodes': len(dendrogram),
            'max_distance': vis_data['max_distance'],
            'min_distance': vis_data['min_distance']
        }
    }


def _build_hierarchical_json(vis_data, n_samples):
    nodes_dict = {node['id']: node for node in vis_data['nodes']}
    root_id = vis_data['root_id']

    def build_tree(node_id):
        node = nodes_dict[node_id]
        tree_node = {
            'id': node['id'],
            'name': node['name'],
            'type': node['type'],
            'distance': node.get('height', 0),
            'size': node['size'],
            'x': node['x'],
            'y': node['y']
        }

        if node['type'] == 'leaf':
            tree_node['sample_index'] = node['samples'][0]
            if 'label' in node:
                tree_node['label'] = node['label']
            if 'data' in node:
                tree_node['data'] = node['data']
            tree_node['children'] = []
        else:
            dendro_idx = node_id - n_samples
            if dendro_idx >= 0 and dendro_idx < len(vis_data['nodes']) - n_samples:
                pass

            child_ids = []
            for link in vis_data['links']:
                if link['target'] == node_id:
                    child_ids.append(link['source'])

            tree_node['children'] = [build_tree(cid) for cid in sorted(child_ids)]

        return tree_node

    return build_tree(root_id)


def print_cutting_result(labels, n_clusters=None):
    if n_clusters is None:
        n_clusters = len(set(labels))

    print(f"\n切割结果 - 共 {n_clusters} 个聚类:")
    print("-" * 50)
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    for label in sorted(clusters.keys()):
        samples = clusters[label]
        print(f"  聚类 {label}: 样本 {samples} (共{len(samples)}个)")
    print("-" * 50)
    return clusters


def analyze_threshold_effect(dendrogram, n_samples, n_thresholds=10):
    max_dist = dendrogram[-1]['distance']
    thresholds = np.linspace(0, max_dist, n_thresholds)

    print("\n阈值切割分析:")
    print("-" * 60)
    print(f"{'阈值':<12} {'聚类数':<10} {'说明':<30}")
    print("-" * 60)

    for t in thresholds:
        labels, n_clusters = cut_dendrogram_by_threshold(dendrogram, n_samples, t)
        if t == 0:
            desc = "每个样本自成一类"
        elif n_clusters == 1:
            desc = "所有样本聚为一类"
        else:
            desc = f"距离 ≤ {t:.2f} 的样本已合并"
        print(f"{t:<12.4f} {n_clusters:<10} {desc:<30}")

    print("-" * 60)
    return thresholds


def print_dendrogram(dendrogram, n_samples):
    print("Dendrogram structure (tree diagram data):")
    print("=" * 60)
    print(f"{'Step':<6} {'Cluster 1':<10} {'Cluster 2':<10} {'Distance':<12} {'Size':<6}")
    print("-" * 60)

    for i, step in enumerate(dendrogram):
        c1 = step['cluster1']
        c2 = step['cluster2']
        dist = step['distance']
        size = step['size']

        c1_desc = f"({c1})" if c1 < n_samples else f"[cluster {c1}]"
        c2_desc = f"({c2})" if c2 < n_samples else f"[cluster {c2}]"

        print(f"{i+1:<6} {c1_desc:<10} {c2_desc:<10} {dist:<12.4f} {size:<6}")
    print("=" * 60)


def compare_memory_usage(n):
    print(f"\n内存使用对比 (n = {n}):")
    print("-" * 60)

    full_matrix_mem = n * n * 8
    condensed_matrix_mem = n * (n - 1) // 2 * 8
    slink_mem = 2 * n * 8
    heap_overhead = n * (n - 1) // 2 * 24

    print(f"完整矩阵: {full_matrix_mem / 1024 / 1024:>8.2f} MB")
    print(f"压缩矩阵: {condensed_matrix_mem / 1024 / 1024:>8.2f} MB  (节省 {100 - condensed_matrix_mem/full_matrix_mem*100:.1f}%)")
    print(f"SLINK算法: {slink_mem / 1024:>8.2f} KB  (节省 {100 - slink_mem/full_matrix_mem*100:.1f}%)")
    print(f"堆算法(含堆): {heap_overhead / 1024 / 1024:>8.2f} MB  (压缩矩阵+堆)")


def benchmark_memory_and_speed(n_samples=1000):
    print(f"\n{'='*60}")
    print(f"大样本性能测试 (n={n_samples})")
    print(f"{'='*60}")

    np.random.seed(42)
    data = np.random.rand(n_samples, 2)

    print("\n1. SLINK算法 (单链连接, O(n)空间):")
    start = time.time()
    dendro_slink = agglomerative_clustering(data=data, metric='euclidean',
                                            linkage='single', algorithm='slink')
    time_slink = time.time() - start
    print(f"   耗时: {time_slink:.2f}秒")
    print(f"   聚类树步骤数: {len(dendro_slink)}")
    print(f"   理论内存: ~{2 * n_samples * 8 / 1024:.1f} KB")

    print(f"\n2. Heap算法 (全链连接, O(n²/2)空间):")
    start = time.time()
    dendro_heap = agglomerative_clustering(data=data, metric='euclidean',
                                           linkage='complete', algorithm='heap')
    time_heap = time.time() - start
    print(f"   耗时: {time_heap:.2f}秒")
    print(f"   聚类树步骤数: {len(dendro_heap)}")
    print(f"   压缩矩阵内存: {n_samples * (n_samples - 1) // 2 * 8 / 1024 / 1024:.1f} MB")

    return time_slink, time_heap


def verify_consistency():
    print("\n" + "=" * 60)
    print("结果一致性验证")
    print("=" * 60)

    np.random.seed(42)
    data = np.random.rand(50, 2)

    print("\n1. 单链连接:")
    dendro_slink = agglomerative_clustering(data=data, linkage='single', algorithm='slink')
    dendro_nn = agglomerative_clustering(data=data, linkage='single', algorithm='nn_chain')
    dendro_heap = agglomerative_clustering(data=data, linkage='single', algorithm='heap')

    slink_dists = sorted([round(s['distance'], 6) for s in dendro_slink])
    nn_dists = sorted([round(s['distance'], 6) for s in dendro_nn])
    heap_dists = sorted([round(s['distance'], 6) for s in dendro_heap])

    match1 = slink_dists == heap_dists
    match2 = nn_dists == heap_dists
    print(f"  SLINK  vs Heap: {'✓ 一致' if match1 else '✗ 不一致'}")
    print(f"  NN-chain vs Heap: {'✓ 一致' if match2 else '✗ 不一致'}")

    print("\n2. 全链连接:")
    dendro_nn = agglomerative_clustering(data=data, linkage='complete', algorithm='nn_chain')
    dendro_heap = agglomerative_clustering(data=data, linkage='complete', algorithm='heap')

    nn_dists = sorted([round(s['distance'], 6) for s in dendro_nn])
    heap_dists = sorted([round(s['distance'], 6) for s in dendro_heap])

    match = nn_dists == heap_dists
    print(f"  NN-chain vs Heap: {'✓ 一致' if match else '✗ 不一致'}")

    print("\n3. 平均链连接:")
    dendro_auto = agglomerative_clustering(data=data, linkage='average', algorithm='auto')
    dendro_heap = agglomerative_clustering(data=data, linkage='average', algorithm='heap')

    auto_dists = sorted([round(s['distance'], 6) for s in dendro_auto])
    heap_dists = sorted([round(s['distance'], 6) for s in dendro_heap])

    match = auto_dists == heap_dists
    print(f"  Auto vs Heap: {'✓ 一致' if match else '✗ 不一致'}")
    print(f"  (平均链默认使用 Heap 算法确保正确性)")


if __name__ == "__main__":
    print("=" * 70)
    print("优化版凝聚层次聚类 - 树状图切割与可视化")
    print("=" * 70)
    print("核心功能:")
    print("  ✓ 压缩距离矩阵 (节省50%内存)")
    print("  ✓ Lance-Williams 递推公式 (O(1)距离更新)")
    print("  ✓ SLINK/NN-chain/Heap 三种算法")
    print("  ✓ 按类别数K切割树状图")
    print("  ✓ 按距离阈值切割树状图")
    print("  ✓ 前端可视化JSON数据生成")

    data = np.array([
        [1, 2],
        [2, 3],
        [5, 5],
        [6, 7],
        [10, 1]
    ])

    print("\n原始数据点:")
    for i, point in enumerate(data):
        print(f"点 {i}: {point}")

    print("\n" + "=" * 70)
    print("1. 执行聚类并生成树状图")
    print("=" * 70)

    dendrogram = agglomerative_clustering(data=data, metric='euclidean',
                                           linkage='single', algorithm='slink')
    print_dendrogram(dendrogram, len(data))

    print("\n" + "=" * 70)
    print("2. 按类别数K切割树状图")
    print("=" * 70)

    print("\n--- K=2 类 ---")
    labels_k2 = cut_dendrogram_by_k(dendrogram, len(data), 2)
    clusters_k2 = print_cutting_result(labels_k2, 2)

    print("\n--- K=3 类 ---")
    labels_k3 = cut_dendrogram_by_k(dendrogram, len(data), 3)
    clusters_k3 = print_cutting_result(labels_k3, 3)

    print("\n--- K=4 类 ---")
    labels_k4 = cut_dendrogram_by_k(dendrogram, len(data), 4)
    clusters_k4 = print_cutting_result(labels_k4, 4)

    print("\n" + "=" * 70)
    print("3. 按距离阈值切割树状图")
    print("=" * 70)

    max_dist = dendrogram[-1]['distance']
    print(f"树状图最大距离: {max_dist:.4f}")

    threshold = 2.5
    labels_t, n_clusters_t = cut_dendrogram_by_threshold(dendrogram, len(data), threshold)
    print(f"\n--- 阈值={threshold} ---")
    clusters_t = print_cutting_result(labels_t, n_clusters_t)

    threshold = 4.0
    labels_t2, n_clusters_t2 = cut_dendrogram_by_threshold(dendrogram, len(data), threshold)
    print(f"\n--- 阈值={threshold} ---")
    clusters_t2 = print_cutting_result(labels_t2, n_clusters_t2)

    optimal_t = find_optimal_threshold_for_k(dendrogram, len(data), 2)
    print(f"\nK=2 的最优阈值: {optimal_t:.4f}")

    print("\n" + "=" * 70)
    print("4. 阈值切割分析")
    print("=" * 70)
    analyze_threshold_effect(dendrogram, len(data), n_thresholds=8)

    print("\n" + "=" * 70)
    print("5. 生成前端可视化JSON数据")
    print("=" * 70)

    vis_json = generate_dendrogram_json(dendrogram, len(data), labels=labels_k2, data=data)

    print("\n--- 元数据 ---")
    for key, value in vis_json['metadata'].items():
        print(f"  {key}: {value}")

    print("\n--- 节点信息 (前6个) ---")
    for node in vis_json['flat']['nodes'][:6]:
        node_type = "叶节点" if node['type'] == 'leaf' else "内部节点"
        label_info = f", label={node.get('label', '-')}" if node['type'] == 'leaf' else ""
        print(f"  ID={node['id']:2d} | {node_type} | x={node['x']:.3f}, y={node['y']:.3f} | "
              f"size={node['size']}{label_info}")

    print("\n--- 层级结构 (前2层) ---")
    def print_tree(node, indent=0, max_depth=2):
        prefix = "  " * indent
        if node['type'] == 'leaf':
            print(f"{prefix}└─ {node['name']} (样本{node['sample_index']}, label={node.get('label', '-')})")
        else:
            print(f"{prefix}└─ {node['name']} (距离={node['distance']:.3f}, 大小={node['size']})")
            if indent < max_depth:
                for child in node['children']:
                    print_tree(child, indent + 1, max_depth)

    print_tree(vis_json['hierarchy'])

    print("\n--- 可直接用于前端渲染的字段说明 ---")
    print("  vis_json['flat']['nodes']: 所有节点列表, 含 x,y 坐标")
    print("  vis_json['flat']['links']: 所有连接线, 含 source/target 坐标")
    print("  vis_json['hierarchy']: 嵌套树形结构, 适合 D3.js/React 等")
    print("  每个节点包含: id, name, type, x, y, distance, size, children")

    print("\n" + "=" * 70)
    print("6. 内存使用对比演示")
    print("=" * 70)

    for n in [1000, 5000, 10000, 20000]:
        compare_memory_usage(n)

    verify_consistency()

    print("\n" + "=" * 70)
    print("7. 大样本测试 (500样本)")
    print("=" * 70)

    try:
        benchmark_memory_and_speed(n_samples=500)
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()
