import numpy as np
import time
import matplotlib.pyplot as plt
from rrt_star import RRTStar, Node
from scipy.spatial import cKDTree


def test_bruteforce_vs_kdtree(node_counts=[100, 500, 1000, 2000, 5000, 10000]):
    print("=" * 70)
    print("近邻搜索性能对比：暴力搜索 vs KD-Tree")
    print("=" * 70)
    
    bf_times = []
    kdt_times = []
    speedups = []
    
    for n_nodes in node_counts:
        print(f"\n测试节点数: {n_nodes}")
        
        nodes = []
        coords = np.random.rand(n_nodes, 2) * 10000
        for i in range(n_nodes):
            node = Node(coords[i, 0], coords[i, 1])
            nodes.append(node)
        
        query_point = Node(np.random.rand() * 10000, np.random.rand() * 10000)
        
        n_queries = 100 if n_nodes <= 1000 else 20
        bf_total = 0
        for _ in range(n_queries):
            t_start = time.perf_counter()
            idx = RRTStar.get_nearest_node_index(nodes, query_point)
            bf_total += time.perf_counter() - t_start
        bf_avg = bf_total / n_queries
        bf_times.append(bf_avg)
        
        kdtree = cKDTree(coords)
        kdt_total = 0
        for _ in range(n_queries):
            t_start = time.perf_counter()
            _, idx = kdtree.query([query_point.x, query_point.y], k=1)
            kdt_total += time.perf_counter() - t_start
        kdt_avg = kdt_total / n_queries
        kdt_times.append(kdt_avg)
        
        speedup = bf_avg / kdt_avg
        speedups.append(speedup)
        
        print(f"  暴力搜索: {bf_avg*1000:.4f} ms")
        print(f"  KD-Tree:   {kdt_avg*1000:.6f} ms")
        print(f"  加速比:    {speedup:.1f}x")
    
    print("\n" + "=" * 70)
    print("性能总结")
    print("=" * 70)
    for i, n in enumerate(node_counts):
        print(f"{n:6d} 节点: 加速 {speedups[i]:>6.1f}x")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(node_counts, np.array(bf_times)*1000, 'o-', label='暴力搜索', linewidth=2, markersize=8)
    axes[0].plot(node_counts, np.array(kdt_times)*1000, 's-', label='KD-Tree', linewidth=2, markersize=8)
    axes[0].set_xlabel('节点数量', fontsize=12)
    axes[0].set_ylabel('搜索时间 (ms)', fontsize=12)
    axes[0].set_title('近邻搜索时间对比', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    axes[1].plot(node_counts, speedups, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('节点数量', fontsize=12)
    axes[1].set_ylabel('加速比 (暴力/KD-Tree)', fontsize=12)
    axes[1].set_title('KD-Tree 相对加速比', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    for i, (n, s) in enumerate(zip(node_counts, speedups)):
        axes[1].annotate(f'{s:.1f}x', (n, s), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("\n性能对比图已保存为 performance_comparison.png")
    plt.show()


def test_rrt_star_performance():
    print("\n" + "=" * 70)
    print("RRT* 完整算法性能测试")
    print("=" * 70)
    
    start = [0.0, 0.0]
    goal = [9000.0, 9000.0]
    
    obstacle_list = [
        [3000.0, 3000.0, 800.0],
        [5000.0, 6000.0, 600.0],
        [7000.0, 2000.0, 500.0],
        [[2000.0, 7000.0], [2500.0, 8000.0], [3000.0, 7500.0]],
    ]
    
    search_area = [0, 10000]
    
    print("\n大规模地图 (10000x10000网格) 测试:")
    
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        search_area=search_area,
        expand_dis=200.0,
        goal_sample_rate=10,
        max_iter=1000,
        connect_circle_dist=300.0,
        kdtree_rebuild_freq=50
    )
    
    t_start = time.perf_counter()
    path = rrt_star.planning(animation=False)
    total_time = time.perf_counter() - t_start
    
    stats = rrt_star.get_performance_stats()
    
    print(f"  总运行时间: {total_time:.2f} 秒")
    print(f"  节点总数: {stats['node_count']}")
    print(f"  近邻搜索次数: {stats['search_count']}")
    print(f"  搜索总耗时: {stats['total_search_time']:.2f} 秒")
    print(f"  平均单次搜索: {stats['avg_search_time']*1000:.4f} ms")
    print(f"  搜索占总时间: {stats['total_search_time']/total_time*100:.1f}%")
    
    if path is None:
        print("\n  警告: 未找到路径，尝试增加迭代次数或调整参数")


def test_radius_search():
    print("\n" + "=" * 70)
    print("半径近邻搜索性能对比 (find_near_nodes)")
    print("=" * 70)
    
    node_counts = [500, 1000, 2000, 5000]
    radius = 500
    
    bf_times = []
    kdt_times = []
    
    for n_nodes in node_counts:
        print(f"\n测试节点数: {n_nodes}, 搜索半径: {radius}")
        
        coords = np.random.rand(n_nodes, 2) * 10000
        nodes = []
        for i in range(n_nodes):
            node = Node(coords[i, 0], coords[i, 1])
            nodes.append(node)
        
        query_node = Node(np.random.rand() * 10000, np.random.rand() * 10000)
        
        n_queries = 50 if n_nodes <= 2000 else 10
        
        bf_total = 0
        for _ in range(n_queries):
            t_start = time.perf_counter()
            dist_list = [(n.x - query_node.x) ** 2 + (n.y - query_node.y) ** 2 for n in nodes]
            near_inds = [dist_list.index(i) for i in dist_list if i <= radius ** 2]
            bf_total += time.perf_counter() - t_start
        bf_avg = bf_total / n_queries
        bf_times.append(bf_avg)
        
        kdtree = cKDTree(coords)
        kdt_total = 0
        for _ in range(n_queries):
            t_start = time.perf_counter()
            near_inds = kdtree.query_ball_point([query_node.x, query_node.y], radius)
            kdt_total += time.perf_counter() - t_start
        kdt_avg = kdt_total / n_queries
        kdt_times.append(kdt_avg)
        
        speedup = bf_avg / kdt_avg
        print(f"  暴力搜索: {bf_avg*1000:.4f} ms, 找到 {len(near_inds)} 个近邻")
        print(f"  KD-Tree:   {kdt_avg*1000:.6f} ms")
        print(f"  加速比:    {speedup:.1f}x")


if __name__ == '__main__':
    test_bruteforce_vs_kdtree()
    test_radius_search()
    test_rrt_star_performance()
