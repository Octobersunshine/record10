import heapq
from typing import Dict, List, Tuple, Optional, Set

WHITE, GRAY, BLACK = 0, 1, 2


def find_cycle_path(graph: Dict[int, List[int]]) -> Optional[List[int]]:
    color: Dict[int, int] = {node: WHITE for node in graph}
    path: List[int] = []

    def dfs(node: int) -> Optional[List[int]]:
        color[node] = GRAY
        path.append(node)

        for neighbor in sorted(graph[node]):
            if color[neighbor] == GRAY:
                idx = path.index(neighbor)
                return list(path[idx:]) + [neighbor]
            elif color[neighbor] == WHITE:
                cycle = dfs(neighbor)
                if cycle is not None:
                    return cycle

        path.pop()
        color[node] = BLACK
        return None

    for start_node in sorted(graph.keys()):
        if color[start_node] == WHITE:
            cycle = dfs(start_node)
            if cycle is not None:
                return cycle

    return None


def topological_sort_kahn(graph: Dict[int, List[int]]) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    in_degree: Dict[int, int] = {node: 0 for node in graph}

    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    min_heap: List[int] = []
    for node in in_degree:
        if in_degree[node] == 0:
            heapq.heappush(min_heap, node)

    top_order: List[int] = []
    visited_count: int = 0

    while min_heap:
        current: int = heapq.heappop(min_heap)
        top_order.append(current)
        visited_count += 1

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(min_heap, neighbor)

    if visited_count == len(graph):
        return top_order, None
    else:
        cycle_path: Optional[List[int]] = find_cycle_path(graph)
        return None, cycle_path


def all_topological_sorts(graph: Dict[int, List[int]]) -> Tuple[Optional[List[List[int]]], Optional[List[int]]]:
    in_degree: Dict[int, int] = {node: 0 for node in graph}

    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    result: List[List[int]] = []
    visited: Set[int] = set()

    def backtrack(current: List[int]) -> None:
        if len(current) == len(graph):
            result.append(list(current))
            return

        for node in sorted(graph.keys()):
            if node not in visited and in_degree[node] == 0:
                visited.add(node)
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1

                current.append(node)
                backtrack(current)
                current.pop()

                for neighbor in graph[node]:
                    in_degree[neighbor] += 1
                visited.remove(node)

    backtrack([])

    if result:
        return result, None
    else:
        cycle_path = find_cycle_path(graph)
        return None, cycle_path


def topological_sort_weighted(
    graph: Dict[int, List[int]],
    weights: Dict[int, float]
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    in_degree: Dict[int, int] = {node: 0 for node in graph}

    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    max_heap: List[Tuple[float, int]] = []
    for node in in_degree:
        if in_degree[node] == 0:
            weight = weights.get(node, 0)
            heapq.heappush(max_heap, (-weight, node))

    top_order: List[int] = []
    visited_count: int = 0

    while max_heap:
        neg_weight, current = heapq.heappop(max_heap)
        top_order.append(current)
        visited_count += 1

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                weight = weights.get(neighbor, 0)
                heapq.heappush(max_heap, (-weight, neighbor))

    if visited_count == len(graph):
        return top_order, None
    else:
        cycle_path = find_cycle_path(graph)
        return None, cycle_path


def critical_path_length(
    graph: Dict[int, List[int]],
    node_weights: Optional[Dict[int, float]] = None,
    edge_weights: Optional[Dict[Tuple[int, int], float]] = None
) -> Tuple[Optional[float], Optional[List[int]], Optional[List[int]]]:
    if node_weights is None:
        node_weights = {node: 0 for node in graph}
    if edge_weights is None:
        edge_weights = {}

    top_order, cycle_path = topological_sort_kahn(graph)
    if top_order is None:
        return None, None, cycle_path

    dist: Dict[int, float] = {node: node_weights.get(node, 0) for node in graph}
    prev: Dict[int, Optional[int]] = {node: None for node in graph}

    for node in top_order:
        for neighbor in graph[node]:
            edge_w = edge_weights.get((node, neighbor), 0)
            new_dist = dist[node] + node_weights.get(neighbor, 0) + edge_w
            if new_dist > dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = node

    end_node = max(dist, key=dist.get)
    max_length = dist[end_node]

    path: List[int] = []
    current: Optional[int] = end_node
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()

    return max_length, path, None


def format_graph_input(pairs: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    graph: Dict[int, List[int]] = {}

    for u, v in pairs:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)

    return graph


def format_cycle_path(cycle_path: List[int]) -> str:
    return " -> ".join(str(n) for n in cycle_path)


def main() -> None:
    print("=" * 60)
    print("拓扑排序 - 完整功能集")
    print("=" * 60)

    test_cases = [
        {
            "name": "DAG - 4个节点（多种拓扑排序）",
            "edges": [(1, 2), (1, 3), (2, 4), (3, 4)],
            "description": "演示所有可能的拓扑排序",
            "weights": {1: 10, 2: 5, 3: 8, 4: 3}
        },
        {
            "name": "DAG - 3个节点（所有排序）",
            "edges": [(1, 3), (2, 3)],
            "description": "只有3种可能的拓扑排序",
            "weights": {1: 1, 2: 5, 3: 2}
        },
        {
            "name": "DAG - 关键路径测试",
            "edges": [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)],
            "description": "计算最长路径",
            "weights": {1: 3, 2: 2, 3: 4, 4: 1, 5: 5},
            "edge_weights": {(1, 2): 1, (1, 3): 2, (2, 4): 3, (3, 4): 1, (4, 5): 2}
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"测试用例 {i}: {test_case['name']}")
        print(f"描述: {test_case['description']}")
        print(f"边: {test_case['edges']}")
        print('-' * 60)

        graph = format_graph_input(test_case['edges'])
        print(f"邻接表: {graph}")

        top_order, cycle_path = topological_sort_kahn(graph)
        if top_order is not None:
            print(f"✓ Kahn排序结果: {top_order}")
        else:
            print(f"✗ 存在环: {format_cycle_path(cycle_path) if cycle_path else '未知'}")
            continue

        all_orders, _ = all_topological_sorts(graph)
        if all_orders:
            print(f"✓ 所有可能的拓扑排序 ({len(all_orders)} 种):")
            for idx, order in enumerate(all_orders, 1):
                print(f"   {idx}. {order}")

        weights = test_case.get('weights', {})
        if weights:
            print(f"  节点权重: {weights}")
            weighted_order, _ = topological_sort_weighted(graph, weights)
            if weighted_order:
                print(f"✓ 按权重优先排序: {weighted_order}")

        node_w = test_case.get('weights', {})
        edge_w = test_case.get('edge_weights', {})
        if node_w or edge_w:
            cp_length, cp_path, _ = critical_path_length(graph, node_w, edge_w)
            if cp_length is not None:
                print(f"✓ 关键路径长度: {cp_length}")
                print(f"  关键路径: {' -> '.join(map(str, cp_path)) if cp_path else '无'}")

    print(f"\n{'=' * 60}")
    print("自定义综合测试")
    print('=' * 60)

    custom_edges = [(5, 11), (7, 11), (7, 8), (3, 8), (3, 10), (11, 2), (11, 9), (11, 10), (8, 9)]
    custom_weights = {3: 5, 5: 3, 7: 7, 8: 4, 11: 10, 2: 1, 9: 6, 10: 2}
    custom_node_w = {3: 2, 5: 1, 7: 3, 8: 2, 11: 4, 2: 1, 9: 2, 10: 1}

    print(f"边: {custom_edges}")
    custom_graph = format_graph_input(custom_edges)

    top_order, _ = topological_sort_kahn(custom_graph)
    print(f"✓ Kahn排序: {top_order}")

    weighted_order, _ = topological_sort_weighted(custom_graph, custom_weights)
    print(f"✓ 权重优先排序: {weighted_order}")
    print(f"  权重: {custom_weights}")

    cp_length, cp_path, _ = critical_path_length(custom_graph, custom_node_w)
    if cp_length is not None:
        print(f"✓ 关键路径长度: {cp_length}")
        print(f"  关键路径: {' -> '.join(map(str, cp_path))}")


if __name__ == "__main__":
    main()
