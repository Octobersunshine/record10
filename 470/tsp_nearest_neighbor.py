import warnings
import numpy as np
from collections import defaultdict


def _check_symmetry(distance_matrix):
    if not np.allclose(distance_matrix, distance_matrix.T):
        warnings.warn(
            "距离矩阵不对称，MST/Christofides适用于对称TSP，非对称结果可能不理想",
            UserWarning,
            stacklevel=3,
        )


def _path_cost(path, distance_matrix):
    return sum(distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))


def _nn_from(distance_matrix, start):
    n = len(distance_matrix)
    visited = [False] * n
    path = [start]
    visited[start] = True
    current = start

    for _ in range(n - 1):
        next_city = None
        min_dist = float('inf')
        for city in range(n):
            if not visited[city] and distance_matrix[current][city] < min_dist:
                min_dist = distance_matrix[current][city]
                next_city = city
        path.append(next_city)
        visited[next_city] = True
        current = next_city

    path.append(start)
    return path, _path_cost(path, distance_matrix)


def nearest_neighbor_tsp(distance_matrix, start=None):
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    _check_symmetry(distance_matrix)
    n = len(distance_matrix)

    if start is not None:
        return _nn_from(distance_matrix, start)

    best_path = None
    best_dist = float('inf')
    for s in range(n):
        path, dist = _nn_from(distance_matrix, s)
        if dist < best_dist:
            best_dist = dist
            best_path = path
    return best_path, best_dist


def two_opt_tsp(distance_matrix, initial_path=None, max_iter=1000):
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    _check_symmetry(distance_matrix)
    n = len(distance_matrix)

    if initial_path is None:
        initial_path, _ = nearest_neighbor_tsp(distance_matrix)
    else:
        initial_path = list(initial_path)

    best_path = initial_path[:-1]
    best_cost = _path_cost(best_path + [best_path[0]], distance_matrix)
    improved = True
    iterations = 0

    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_path = best_path[:i] + best_path[i:j][::-1] + best_path[j:]
                new_cost = _path_cost(new_path + [new_path[0]], distance_matrix)
                if new_cost < best_cost - 1e-9:
                    best_path = new_path
                    best_cost = new_cost
                    improved = True

    best_path.append(best_path[0])
    return best_path, best_cost


def prim_mst(distance_matrix):
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(distance_matrix)
    key = [float('inf')] * n
    parent = [-1] * n
    in_mst = [False] * n
    key[0] = 0

    for _ in range(n):
        u = -1
        min_key = float('inf')
        for v in range(n):
            if not in_mst[v] and key[v] < min_key:
                min_key = key[v]
                u = v
        if u == -1:
            break
        in_mst[u] = True
        for v in range(n):
            if not in_mst[v] and distance_matrix[u][v] < key[v]:
                key[v] = distance_matrix[u][v]
                parent[v] = u

    mst_edges = []
    mst_cost = 0.0
    for v in range(1, n):
        u = parent[v]
        if u >= 0:
            mst_edges.append((u, v))
            mst_cost += distance_matrix[u][v]
    return mst_edges, mst_cost


def _dfs_euler(adj, start):
    stack = [start]
    path = []
    visited_edges = defaultdict(int)

    while stack:
        u = stack[-1]
        found = False
        for v in adj[u]:
            if visited_edges[(u, v)] == 0 and visited_edges[(v, u)] == 0:
                visited_edges[(u, v)] = 1
                visited_edges[(v, u)] = 1
                stack.append(v)
                found = True
                break
        if not found:
            path.append(stack.pop())
    return path[::-1]


def christofides_tsp(distance_matrix):
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    _check_symmetry(distance_matrix)
    n = len(distance_matrix)

    mst_edges, _ = prim_mst(distance_matrix)

    degree = [0] * n
    adj = defaultdict(list)
    for u, v in mst_edges:
        degree[u] += 1
        degree[v] += 1
        adj[u].append(v)
        adj[v].append(u)

    odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]
    k = len(odd_vertices)

    if k > 0:
        used = [False] * k
        for i in range(0, k, 2):
            best_j = -1
            min_dist = float('inf')
            for j in range(i + 1, k):
                if not used[j] and distance_matrix[odd_vertices[i]][odd_vertices[j]] < min_dist:
                    min_dist = distance_matrix[odd_vertices[i]][odd_vertices[j]]
                    best_j = j
            if best_j >= 0:
                u, v = odd_vertices[i], odd_vertices[best_j]
                adj[u].append(v)
                adj[v].append(u)
                used[best_j] = True

    euler = _dfs_euler(adj, 0)

    visited = [False] * n
    tour = []
    for v in euler:
        if not visited[v]:
            tour.append(v)
            visited[v] = True
    tour.append(tour[0])

    return tour, _path_cost(tour, distance_matrix)


def mst_approx_tsp(distance_matrix):
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    _check_symmetry(distance_matrix)
    n = len(distance_matrix)

    mst_edges, mst_cost = prim_mst(distance_matrix)

    adj = defaultdict(list)
    for u, v in mst_edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * n
    tour = []

    def dfs(u):
        visited[u] = True
        tour.append(u)
        for v in sorted(adj[u], key=lambda x: distance_matrix[u][x]):
            if not visited[v]:
                dfs(v)

    dfs(0)
    tour.append(0)

    return tour, _path_cost(tour, distance_matrix)


def compare_heuristics(distance_matrix, optimal=None):
    print("=" * 60)
    print("TSP 启发式算法对比")
    print("=" * 60)

    results = {}

    nn_path, nn_dist = nearest_neighbor_tsp(distance_matrix)
    results['最近邻（多起点）'] = (nn_path, nn_dist)

    two_path, two_dist = two_opt_tsp(distance_matrix, nn_path)
    results['最近邻 + 2-opt'] = (two_path, two_dist)

    mst_path, mst_dist = mst_approx_tsp(distance_matrix)
    results['MST 近似（两倍）'] = (mst_path, mst_dist)

    chr_path, chr_dist = christofides_tsp(distance_matrix)
    results['Christofides（1.5倍）'] = (chr_path, chr_dist)

    if optimal is not None:
        opt_dist = _path_cost(optimal, distance_matrix)
        print(f"{'算法':<20} {'总距离':>10} {'路径'}")
        print("-" * 60)
        print(f"{'最优解':<20} {opt_dist:>10.2f} {optimal}")
        print("-" * 60)
        for name, (path, dist) in results.items():
            ratio = dist / opt_dist if opt_dist > 0 else float('inf')
            print(f"{name:<20} {dist:>10.2f} {ratio:>5.2f}x  {path[:-1]}")
    else:
        print(f"{'算法':<20} {'总距离':>10} {'路径'}")
        print("-" * 60)
        for name, (path, dist) in results.items():
            print(f"{name:<20} {dist:>10.2f}  {path[:-1]}")

    print("=" * 60)
    return results


if __name__ == "__main__":
    sym_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ])

    large_matrix = np.array([
        [0, 29, 20, 21, 16, 31],
        [29, 0, 15, 29, 28, 11],
        [20, 15, 0, 15, 14, 13],
        [21, 29, 15, 0, 4, 23],
        [16, 28, 14, 4, 0, 22],
        [31, 11, 13, 23, 22, 0],
    ])

    print("\n=== 4 城市 TSP 对比 ===")
    compare_heuristics(sym_matrix, optimal=[0, 1, 3, 2, 0])

    print("\n=== 6 城市 TSP 对比 ===")
    compare_heuristics(large_matrix)

    print("\n=== 各算法单独调用示例 ===")
    path, dist = christofides_tsp(large_matrix)
    print(f"Christofides 路径: {path}")
    print(f"Christofides 总距离: {dist}")

    mst_edges, mst_cost = prim_mst(large_matrix)
    print(f"\nMST 边: {mst_edges}")
    print(f"MST 总成本: {mst_cost}")
