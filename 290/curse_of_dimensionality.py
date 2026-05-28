import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import time


def generate_random_points(num_points, dimension, low=0, high=1):
    return np.random.uniform(low, high, size=(num_points, dimension))


def calculate_distance_statistics(points):
    distances = pdist(points, metric='euclidean')
    mean_dist = np.mean(distances)
    var_dist = np.var(distances)
    std_dist = np.std(distances)
    cv = std_dist / mean_dist if mean_dist > 0 else 0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    ratio = min_dist / max_dist if max_dist > 0 else 0
    return mean_dist, var_dist, std_dist, cv, min_dist, max_dist, ratio, distances


def calculate_distance_statistics_incremental(num_points, dimension, low=0, high=1, block_size=10):
    """
    增量计算欧氏距离统计量，内存复杂度 O(n*block_size) 而非 O(n*d)
    
    原理：逐块生成坐标，累加距离平方，避免存储完整的 n×d 矩阵
    dist² = Σₖ (xᵢₖ - xⱼₖ)²，其中 k=1..d
    
    使用分块策略平衡内存和速度：
    - block_size=1: 内存最小 O(n)，速度较慢
    - block_size=d: 等同于原方法，内存 O(n*d)，速度最快
    """
    n = num_points
    num_pairs = n * (n - 1) // 2
    
    dist_sq = np.zeros(num_pairs, dtype=np.float64)
    
    triu_indices = np.triu_indices(n, k=1)
    row_idx, col_idx = triu_indices
    
    remaining = dimension
    while remaining > 0:
        current_block = min(block_size, remaining)
        coords_block = np.random.uniform(low, high, size=(n, current_block))
        
        diff = coords_block[row_idx] - coords_block[col_idx]
        diff_sq_sum = np.sum(diff ** 2, axis=1)
        
        dist_sq += diff_sq_sum
    remaining -= current_block

    distances = np.sqrt(dist_sq)
    
    mean_dist = np.mean(distances)
    var_dist = np.var(distances)
    std_dist = np.std(distances)
    cv = std_dist / mean_dist if mean_dist > 0 else 0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    ratio = min_dist / max_dist if max_dist > 0 else 0
    
    return mean_dist, var_dist, std_dist, cv, min_dist, max_dist, ratio, distances


def calculate_memory_usage(num_points, dimension, block_size=10):
    """计算两种方法的理论内存占用（MB）"""
    n = num_points
    d = dimension
    num_pairs = n * (n - 1) // 2
    
    mem_original = (n * d * 8 + num_pairs * 8) / (1024 ** 2)
    mem_incremental = (n * block_size * 8 + num_pairs * 8) / (1024 ** 2)
    
    return mem_original, mem_incremental


def simulate_curse_of_dimensionality(max_dim=100, num_points=500, use_incremental=True, block_size=10):
    dimensions = range(1, max_dim + 1)
    means = []
    variances = []
    stds = []
    cvs = []
    min_dists = []
    max_dists = []
    ratios = []
    
    for dim in dimensions:
        if use_incremental:
            mean_dist, var_dist, std_dist, cv, min_dist, max_dist, ratio, _ = calculate_distance_statistics_incremental(
                num_points, dim, block_size=block_size
            )
        else:
            points = generate_random_points(num_points, dim)
            mean_dist, var_dist, std_dist, cv, min_dist, max_dist, ratio, _ = calculate_distance_statistics(points)
        
        means.append(mean_dist)
        variances.append(var_dist)
        stds.append(std_dist)
        cvs.append(cv)
        min_dists.append(min_dist)
        max_dists.append(max_dist)
        ratios.append(ratio)
        
        if dim % 10 == 0:
            print(f"维度 {dim:3d}: 均值={mean_dist:.4f}, 标准差={std_dist:.4f}, 变异系数={cv:.4f}, 最近/最远={ratio:.4f}")
    
    return dimensions, means, variances, stds, cvs, min_dists, max_dists, ratios


def compare_performance(num_points=2000, max_dim=100):
    """对比两种方法的性能和内存占用"""
    print("=" * 70)
    print(f"性能对比测试: {num_points}个点, 维度1-{max_dim}")
    print("=" * 70)
    
    block_sizes = [1, 10, 50, 100]
    print(f"\n理论内存占用 (d={max_dim}时):")
    mem_orig, _ = calculate_memory_usage(num_points, max_dim, block_size=1)
    print(f"  原方法 (O(n*d)): {mem_orig:.2f} MB")
    
    for bs in block_sizes:
        _, mem_inc = calculate_memory_usage(num_points, max_dim, block_size=bs)
        savings = 100 * (1 - mem_inc / mem_orig) if mem_orig > 0 else 0
        print(f"  增量法 (block_size={bs}): {mem_inc:.2f} MB, 节省 {savings:.1f}%")
    
    print("\n运行时间测试...")
    np.random.seed(42)
    start = time.time()
    simulate_curse_of_dimensionality(max_dim, num_points, use_incremental=False)
    time_orig = time.time() - start
    print(f"  原方法:   {time_orig:.2f} 秒")
    
    for bs in block_sizes:
        np.random.seed(42)
        start = time.time()
        simulate_curse_of_dimensionality(max_dim, num_points, use_incremental=True, block_size=bs)
        time_inc = time.time() - start
        print(f"  增量法 (block_size={bs}): {time_inc:.2f} 秒 ({time_inc/time_orig:.2f}x)")
    
    print("=" * 70)


def test_correctness(num_points=500, dimension=50):
    """验证增量计算结果的正确性"""
    print("=" * 70)
    print(f"正确性验证: {num_points}个点, {dimension}维")
    print("=" * 70)
    
    np.random.seed(42)
    points = generate_random_points(num_points, dimension)
    mean1, var1, std1, cv1, min1, max1, ratio1, dist1 = calculate_distance_statistics(points)
    
    mean2, var2, std2, cv2, min2, max2, ratio2, dist2 = calculate_distance_statistics_incremental_from_points(
        points
    )
    
    print(f"\n原方法:   均值={mean1:.6f}, 标准差={std1:.6f}, 最近/最远={ratio1:.4f}")
    print(f"增量法:   均值={mean2:.6f}, 标准差={std2:.6f}, 最近/最远={ratio2:.4f}")
    print(f"均值误差: {abs(mean1-mean2):.2e}")
    print(f"标准差误差: {abs(std1-std2):.2e}")
    print(f"距离最大误差: {np.max(np.abs(dist1 - dist2)):.2e}")
    
    if np.allclose(dist1, dist2, atol=1e-10):
        print("\n✓ 验证通过：两种方法计算结果一致！")
    else:
        print("\n✗ 验证失败：结果不一致")
    print("=" * 70)


def calculate_distance_statistics_incremental_from_points(points):
    """从已有的点集逐列计算，用于验证算法正确性"""
    n, d = points.shape
    num_pairs = n * (n - 1) // 2
    
    dist_sq = np.zeros(num_pairs, dtype=np.float64)
    triu_indices = np.triu_indices(n, k=1)
    
    for k in range(d):
        coord = points[:, k]
        diff_sq = (coord[triu_indices[0]] - coord[triu_indices[1]]) ** 2
        dist_sq += diff_sq
    
    distances = np.sqrt(dist_sq)
    
    mean_dist = np.mean(distances)
    var_dist = np.var(distances)
    std_dist = np.std(distances)
    cv = std_dist / mean_dist if mean_dist > 0 else 0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    ratio = min_dist / max_dist if max_dist > 0 else 0
    
    return mean_dist, var_dist, std_dist, cv, min_dist, max_dist, ratio, distances


def plot_results(dimensions, means, variances, stds, cvs, min_dists, max_dists, ratios):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(dimensions, means, 'b-', linewidth=2)
    ax00.set_xlabel('维度', fontsize=12)
    ax00.set_ylabel('平均距离', fontsize=12)
    ax00.set_title('(a) 平均距离随维度的变化', fontsize=14)
    ax00.grid(True, alpha=0.3)
    
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(dimensions, cvs, 'm-', linewidth=2)
    ax01.set_xlabel('维度', fontsize=12)
    ax01.set_ylabel('变异系数 (标准差/均值)', fontsize=12)
    ax01.set_title('(b) 变异系数随维度的变化（距离趋同）', fontsize=14)
    ax01.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(dimensions, min_dists, 'g-', linewidth=2, label='最近点距离')
    ax10.plot(dimensions, max_dists, 'r-', linewidth=2, label='最远点距离')
    ax10.plot(dimensions, means, 'b--', linewidth=2, label='平均距离')
    ax10.set_xlabel('维度', fontsize=12)
    ax10.set_ylabel('距离', fontsize=12)
    ax10.set_title('(c) 最近/最远点距离随维度的变化', fontsize=14)
    ax10.legend(fontsize=10)
    ax10.grid(True, alpha=0.3)
    
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(dimensions, ratios, 'purple', linewidth=2)
    ax11.set_xlabel('维度', fontsize=12)
    ax11.set_ylabel('最近点距离 / 最远点距离', fontsize=12)
    ax11.set_title('(d) 距离比率随维度的变化', fontsize=14)
    ax11.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='比率=1（完全趋同）')
    ax11.fill_between(dimensions, ratios, 1.0, alpha=0.2, color='purple')
    ax11.legend(fontsize=10)
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim([0, 1.05])
    
    ax20 = fig.add_subplot(gs[2, 0])
    relative_min = np.array(min_dists) / np.array(means)
    relative_max = np.array(max_dists) / np.array(means)
    ax20.plot(dimensions, relative_min, 'g-', linewidth=2, label='最近/平均')
    ax20.plot(dimensions, relative_max, 'r-', linewidth=2, label='最远/平均')
    ax20.set_xlabel('维度', fontsize=12)
    ax20.set_ylabel('相对距离（与均值的比值）', fontsize=12)
    ax20.set_title('(e) 相对距离随维度的变化', fontsize=14)
    ax20.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax20.legend(fontsize=10)
    ax20.grid(True, alpha=0.3)
    
    ax21 = fig.add_subplot(gs[2, 1])
    spread = np.array(max_dists) - np.array(min_dists)
    ax21.plot(dimensions, spread, 'orange', linewidth=2)
    ax21.set_xlabel('维度', fontsize=12)
    ax21.set_ylabel('距离范围（最远-最近）', fontsize=12)
    ax21.set_title('(f) 距离范围随维度的变化', fontsize=14)
    ax21.grid(True, alpha=0.3)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.savefig('curse_of_dimensionality.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存为 curse_of_dimensionality.png")
    plt.show()


def plot_distance_distribution(dimensions_to_plot, num_points=500, use_incremental=True, block_size=10):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    all_distances = []
    all_stats = []
    
    for idx, dim in enumerate(dimensions_to_plot):
        if use_incremental:
            mean_d, var_d, std_d, cv, min_d, max_d, ratio, distances = calculate_distance_statistics_incremental(
                num_points, dim, block_size=block_size
            )
        else:
            points = generate_random_points(num_points, dim)
            mean_d, var_d, std_d, cv, min_d, max_d, ratio, distances = calculate_distance_statistics(points)
        
        all_distances.append(distances)
        all_stats.append((mean_d, std_d, min_d, max_d, ratio))
        
        axes[idx].hist(distances, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        axes[idx].set_xlabel('欧氏距离', fontsize=12)
        axes[idx].set_ylabel('概率密度', fontsize=12)
        axes[idx].set_title(f'{dim}维空间的距离分布', fontsize=14)
        axes[idx].grid(True, alpha=0.3)
        
        axes[idx].axvline(mean_d, color='red', linestyle='--', linewidth=2, label=f'均值={mean_d:.2f}')
        axes[idx].axvline(mean_d - std_d, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ范围')
        axes[idx].axvline(mean_d + std_d, color='orange', linestyle=':', linewidth=1.5)
        axes[idx].axvline(min_d, color='green', linestyle='-.', linewidth=1.5, label=f'最近={min_d:.2f}')
        axes[idx].axvline(max_d, color='purple', linestyle='-.', linewidth=1.5, label=f'最远={max_d:.2f}')
        
        if idx == 0:
            axes[idx].legend(fontsize=9, loc='upper right')
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.savefig('distance_distribution.png', dpi=150, bbox_inches='tight')
    print("距离分布图已保存为 distance_distribution.png")
    plt.show()
    
    return all_distances, all_stats


def plot_teaching_visualization(dimensions_to_plot, num_points=500, use_incremental=True, block_size=10):
    """教学可视化：对比不同维度下距离分布的变化，解释为什么需要降维"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    all_distances = []
    all_stats = []
    
    for idx, dim in enumerate(dimensions_to_plot):
        if use_incremental:
            mean_d, var_d, std_d, cv, min_d, max_d, ratio, distances = calculate_distance_statistics_incremental(
                num_points, dim, block_size=block_size
            )
        else:
            points = generate_random_points(num_points, dim)
            mean_d, var_d, std_d, cv, min_d, max_d, ratio, distances = calculate_distance_statistics(points)
        
        all_distances.append(distances)
        all_stats.append((mean_d, std_d, min_d, max_d, ratio, cv))
    
    ax0 = fig.add_subplot(gs[0, 0])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (dim, dists) in enumerate(zip(dimensions_to_plot, all_distances)):
        ax0.hist(dists, bins=50, density=True, alpha=0.4, color=colors[idx], 
                label=f'{dim}维', edgecolor='black', linewidth=0.5)
    ax0.set_xlabel('欧氏距离', fontsize=12)
    ax0.set_ylabel('概率密度', fontsize=12)
    ax0.set_title('(a) 不同维度下距离分布对比（归一化）', fontsize=14)
    ax0.legend(fontsize=10)
    ax0.grid(True, alpha=0.3)
    
    ax1 = fig.add_subplot(gs[0, 1])
    x_vals = np.arange(len(dimensions_to_plot))
    means = [s[0] for s in all_stats]
    stds = [s[1] for s in all_stats]
    mins = [s[2] for s in all_stats]
    maxs = [s[3] for s in all_stats]
    
    width = 0.2
    ax1.bar(x_vals - 1.5*width, means, width, label='均值', color='#1f77b4', alpha=0.7)
    ax1.bar(x_vals - 0.5*width, stds, width, label='标准差', color='#ff7f0e', alpha=0.7)
    ax1.bar(x_vals + 0.5*width, mins, width, label='最近点', color='#2ca02c', alpha=0.7)
    ax1.bar(x_vals + 1.5*width, maxs, width, label='最远点', color='#d62728', alpha=0.7)
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels([f'{d}维' for d in dimensions_to_plot])
    ax1.set_ylabel('距离', fontsize=12)
    ax1.set_title('(b) 距离统计量对比', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = fig.add_subplot(gs[1, 0])
    ratios = [s[4] for s in all_stats]
    cvs = [s[5] for s in all_stats]
    
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(dimensions_to_plot, ratios, 'o-', color='#9467bd', linewidth=2, markersize=8, 
                    label='最近/最远 比率')
    line2 = ax2_twin.plot(dimensions_to_plot, cvs, 's-', color='#e377c2', linewidth=2, markersize=8, 
                         label='变异系数')
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='理想趋同(比率=1)')
    ax2.set_xlabel('维度', fontsize=12)
    ax2.set_ylabel('最近/最远 比率', fontsize=12, color='#9467bd')
    ax2_twin.set_ylabel('变异系数', fontsize=12, color='#e377c2')
    ax2.set_title('(c) 距离趋同指标随维度变化', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='#9467bd')
    ax2_twin.tick_params(axis='y', labelcolor='#e377c2')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10, loc='center right')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    for idx, (dim, dists) in enumerate(zip(dimensions_to_plot, all_distances)):
        normalized_dists = (dists - np.mean(dists)) / np.std(dists)
        ax3.hist(normalized_dists, bins=30, density=True, alpha=0.4, color=colors[idx],
                label=f'{dim}维', edgecolor='black', linewidth=0.5)
    
    x = np.linspace(-4, 4, 100)
    normal_pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    ax3.plot(x, normal_pdf, 'k--', linewidth=2, label='标准正态分布')
    ax3.set_xlabel('标准化距离 (Z-score)', fontsize=12)
    ax3.set_ylabel('概率密度', fontsize=12)
    ax3.set_title('(d) 标准化后的距离分布（中心极限定理）', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-4, 4])
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.savefig('teaching_visualization.png', dpi=150, bbox_inches='tight')
    print("教学可视化图已保存为 teaching_visualization.png")
    plt.show()


def print_teaching_explanation(dimensions, all_stats):
    """打印教学解释内容"""
    print("\n" + "=" * 70)
    print("📚 教学解释：为什么高维空间需要降维方法？")
    print("=" * 70)
    
    for idx, (dim, stats) in enumerate(zip(dimensions, all_stats)):
        mean_d, std_d, min_d, max_d, ratio, cv = stats
        print(f"\n【{dim}维空间】")
        print(f"  · 平均距离: {mean_d:.4f}")
        print(f"  · 最近点距离: {min_d:.4f}, 最远点距离: {max_d:.4f}")
        print(f"  · 最近/最远 比率: {ratio:.4f} {'← 接近1，距离趋同！' if ratio > 0.7 else ''}")
        print(f"  · 变异系数: {cv:.4f} {'← 很小，区分度低！' if cv < 0.15 else ''}")
    
    print("\n" + "-" * 70)
    print("🔍 核心问题：")
    print("  1. 在低维空间（如1维），最近点和最远点差异很大（比率很小）")
    print("     → KNN、聚类等基于距离的算法效果很好")
    print("  2. 在高维空间（如100维），所有点对的距离几乎相等（比率→1）")
    print("     → 无法区分'近邻'和'远邻'，距离度量失效！")
    print("\n📌 解决方案：")
    print("  · PCA/LDA: 线性降维，保留主要方差")
    print("  · t-SNE/UMAP: 非线性降维，保留局部结构")
    print("  · 自动编码器: 深度学习降维方法")
    print("=" * 70)


def plot_memory_comparison(num_points=2000, max_dim=500):
    """绘制内存占用对比图"""
    dimensions = [1, 10, 50, 100, 200, 500, 1000]
    mem_orig_list = []
    mem_inc_list = []
    
    for d in dimensions:
        mo, mi = calculate_memory_usage(num_points, d)
        mem_orig_list.append(mo)
        mem_inc_list.append(mi)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(dimensions, mem_orig_list, 'b-o', linewidth=2, label='原方法 (O(n*d))')
    ax.semilogy(dimensions, mem_inc_list, 'r-s', linewidth=2, label='增量法 (O(n))')
    ax.set_xlabel('维度', fontsize=12)
    ax.set_ylabel('内存占用 (MB)', fontsize=12)
    ax.set_title(f'内存占用对比 (n={num_points}个点)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.savefig('memory_comparison.png', dpi=150, bbox_inches='tight')
    print("内存对比图已保存为 memory_comparison.png")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='维数灾难模拟')
    parser.add_argument('--test', action='store_true', help='运行正确性验证')
    parser.add_argument('--benchmark', action='store_true', help='运行性能对比测试')
    parser.add_argument('--large', action='store_true', help='大规模测试(10000点, 1000维)')
    parser.add_argument('--teach', action='store_true', help='教学演示模式')
    parser.add_argument('--num_points', type=int, default=500, help='随机点数量')
    parser.add_argument('--max_dim', type=int, default=100, help='最大维度')
    args = parser.parse_args()
    
    np.random.seed(42)
    
    if args.test:
        test_correctness(num_points=500, dimension=100)
        return
    
    if args.benchmark:
        compare_performance(num_points=2000, max_dim=100)
        return
    
    if args.large:
        print("=" * 70)
        print("大规模测试: 10000个点, 维度1-1000 (使用增量计算)")
        print("=" * 70)
        block_size = 10
        mem_orig, mem_inc = calculate_memory_usage(10000, 1000, block_size=block_size)
        print(f"\n理论内存占用 (d=1000时):")
        print(f"  原方法 (O(n*d)): {mem_orig:.2f} MB ({mem_orig/1024:.2f} GB)")
        print(f"  增量法 (block_size={block_size}): {mem_inc:.2f} MB")
        print(f"  节省内存:        {mem_orig - mem_inc:.2f} MB ({100*(1-mem_inc/mem_orig):.1f}%)")
        print("\n开始模拟...")
        
        dimensions, means, variances, stds, cvs, min_dists, max_dists, ratios = simulate_curse_of_dimensionality(
            max_dim=1000, num_points=10000, use_incremental=True, block_size=block_size
        )
        
        print("\n" + "=" * 70)
        print("关键维度数据摘要：")
        print("=" * 70)
        key_dims = [1, 10, 50, 100, 200, 500, 1000]
        for d in key_dims:
            idx = d - 1
            print(f"维度 {d:4d}: 均值={means[idx]:.4f}, 标准差={stds[idx]:.4f}, 变异系数={cvs[idx]:.4f}, 最近/最远={ratios[idx]:.4f}")
        
        plot_memory_comparison(num_points=10000, max_dim=1000)
        return
    
    if args.teach:
        print("=" * 70)
        print("🎓 维数灾难教学演示模式")
        print("=" * 70)
        print(f"参数：每个维度生成 {args.num_points} 个随机点")
        print("=" * 70)
        
        teach_dims = [2, 5, 20, 100]
        
        print("\n生成教学可视化图表...")
        _, all_stats = plot_distance_distribution(teach_dims, num_points=args.num_points, 
                                                  use_incremental=True, block_size=10)
        
        print("\n生成综合教学对比图...")
        plot_teaching_visualization(teach_dims, num_points=args.num_points, 
                                   use_incremental=True, block_size=10)
        
        print_teaching_explanation(teach_dims, all_stats)
        return
    
    print("=" * 70)
    print("维数灾难模拟：高维空间中随机点的欧氏距离分布分析")
    print("=" * 70)
    print(f"参数：每个维度生成 {args.num_points} 个随机点，维度范围 1-{args.max_dim}")
    print("使用增量计算模式 (内存复杂度 O(n))")
    print("=" * 70)
    
    dimensions, means, variances, stds, cvs, min_dists, max_dists, ratios = simulate_curse_of_dimensionality(
        max_dim=args.max_dim, num_points=args.num_points, use_incremental=True, block_size=10
    )
    
    print("\n" + "=" * 70)
    print("关键维度数据摘要：")
    print("=" * 70)
    key_dims = [1, 5, 10, 20, 50, 100]
    key_dims = [d for d in key_dims if d <= args.max_dim]
    for d in key_dims:
        idx = d - 1
        print(f"维度 {d:3d}: 均值={means[idx]:.4f}, 标准差={stds[idx]:.4f}, 变异系数={cvs[idx]:.4f}, 最近/最远={ratios[idx]:.4f}")
    
    print("\n" + "=" * 70)
    print("现象分析：")
    print("=" * 70)
    print("1. 平均距离随维度增加而增大（近似与√d成正比）")
    print("2. 距离标准差基本保持不变，但均值不断增大")
    print("3. 变异系数（标准差/均值）随维度增加而减小并趋近于0")
    print("4. 最近点与最远点的距离比率趋近于1，说明距离趋同")
    print("5. 在高维空间中，所有点对之间的距离几乎相等")
    print("6. 这使得基于距离的算法（如KNN、聚类）在高维空间中失效")
    print("=" * 70)
    
    plot_results(dimensions, means, variances, stds, cvs, min_dists, max_dists, ratios)
    
    print("\n" + "=" * 70)
    print("生成不同维度的距离分布直方图...")
    print("=" * 70)
    plot_dims = [1, 10, 50, 100]
    plot_dims = [d for d in plot_dims if d <= args.max_dim]
    plot_distance_distribution(plot_dims, num_points=args.num_points, use_incremental=True, block_size=10)


if __name__ == "__main__":
    main()
