import numpy as np
from scipy.sparse import csr_matrix


def pagerank(links, alpha=0.85, personalization=None, max_iter=100, tol=1e-6):
    """
    使用PageRank算法计算网页排名，支持个性化偏好

    参数:
        links: 有向边列表，每个元素是 (source, target) 元组
        alpha: 阻尼系数，默认0.85
        personalization: 个性化偏好，可以是节点列表或权重字典 {节点: 权重}
                        为None时使用均匀分布（标准PageRank）
        max_iter: 最大迭代次数
        tol: 收敛阈值

    返回:
        节点排名字典，{节点: 排名值}
    """
    if not links:
        return {}

    nodes = set()
    for src, dst in links:
        nodes.add(src)
        nodes.add(dst)
    
    nodes = sorted(nodes)
    n = len(nodes)
    
    if n == 0:
        return {}

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    row = []
    col = []
    data = []

    out_degree = {node: 0 for node in nodes}
    for src, dst in links:
        out_degree[src] += 1

    for src, dst in links:
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        row.append(dst_idx)
        col.append(src_idx)
        data.append(1.0 / out_degree[src])

    M = csr_matrix((data, (row, col)), shape=(n, n))

    rank = np.ones(n) / n
    
    if personalization is None:
        personal_vec = np.ones(n) / n
    else:
        personal_vec = np.zeros(n)
        if isinstance(personalization, (list, set)):
            weight = 1.0 / len(personalization)
            for node in personalization:
                if node in node_to_idx:
                    personal_vec[node_to_idx[node]] = weight
        elif isinstance(personalization, dict):
            total_weight = sum(personalization.values())
            for node, weight in personalization.items():
                if node in node_to_idx:
                    personal_vec[node_to_idx[node]] = weight / total_weight
        if personal_vec.sum() == 0:
            personal_vec = np.ones(n) / n
    
    dangling_nodes = np.array([out_degree[node] == 0 for node in nodes], dtype=bool)

    for _ in range(max_iter):
        dangling_contrib = rank[dangling_nodes].sum()
        new_rank = alpha * M.dot(rank) + alpha * dangling_contrib * personal_vec + (1 - alpha) * personal_vec
        new_rank = new_rank / new_rank.sum()
        diff = np.abs(new_rank - rank).sum()
        rank = new_rank
        if diff < tol:
            break

    result = {nodes[i]: rank[i] for i in range(n)}
    return dict(sorted(result.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    links = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'C'),
        ('C', 'A'),
        ('D', 'A'),
        ('D', 'C'),
    ]
    
    print("=== 标准PageRank ===")
    ranks = pagerank(links)
    for node, score in ranks.items():
        print(f"{node}: {score:.6f}")
    print(f"PR值总和: {sum(ranks.values()):.10f}")
    
    print("\n=== 个性化PageRank (偏好节点: ['D']) ===")
    ranks_personal = pagerank(links, personalization=['D'])
    for node, score in ranks_personal.items():
        print(f"{node}: {score:.6f}")
    print(f"PR值总和: {sum(ranks_personal.values()):.10f}")
    
    print("\n=== 个性化PageRank (权重字典: {{'B': 3, 'D': 1}}) ===")
    ranks_weighted = pagerank(links, personalization={'B': 3, 'D': 1})
    for node, score in ranks_weighted.items():
        print(f"{node}: {score:.6f}")
    print(f"PR值总和: {sum(ranks_weighted.values()):.10f}")
    
    print("\n=== 包含悬挂节点的个性化测试 (E无出链, 偏好: ['E']) ===")
    links2 = links + [('A', 'E')]
    ranks2 = pagerank(links2, personalization=['E'])
    for node, score in ranks2.items():
        print(f"{node}: {score:.6f}")
    print(f"PR值总和: {sum(ranks2.values()):.10f}")
