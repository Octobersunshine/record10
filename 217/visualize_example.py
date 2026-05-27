import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from poisson_disk_sampling import PoissonDiskSampler3D, BridsonSampler3D, point_cloud_simplification


def visualize_point_cloud_comparison():
    """对比原始点云和简化后的点云"""
    np.random.seed(42)
    
    print("生成原始点云...")
    n_original = 5000
    theta = np.random.uniform(0, 2*np.pi, n_original)
    phi = np.arccos(np.random.uniform(-1, 1, n_original))
    r = np.random.uniform(0, 5, n_original)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    original_points = np.column_stack([x, y, z])
    
    print("进行泊松圆盘采样简化...")
    simplified = point_cloud_simplification(original_points, target_count=500)
    
    print(f"原始: {len(original_points)} 点, 简化后: {len(simplified)} 点")
    
    fig = plt.figure(figsize=(15, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
                c='blue', s=1, alpha=0.5)
    ax1.set_title(f'原始点云 ({len(original_points)} 点)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(simplified[:, 0], simplified[:, 1], simplified[:, 2], 
                c='red', s=10, alpha=0.8)
    ax2.set_title(f'泊松圆盘采样简化 ({len(simplified)} 点)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('point_cloud_comparison.png', dpi=150, bbox_inches='tight')
    print("已保存: point_cloud_comparison.png")
    plt.close()


def visualize_bridson_sampling():
    """可视化 Bridson 算法的空间采样效果"""
    print("\n执行 Bridson 泊松圆盘采样...")
    bounds = np.array([[0, 10], [0, 10], [0, 10]])
    
    sampler = BridsonSampler3D(min_radius=1.2, bounds=bounds, num_candidates=30)
    samples = sampler.sample()
    
    print(f"采样点数: {len(samples)}")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], 
               c='green', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_title(f'Bridson 泊松圆盘采样 (min_radius=1.2, {len(samples)} 点)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    
    plt.tight_layout()
    plt.savefig('bridson_sampling.png', dpi=150, bbox_inches='tight')
    print("已保存: bridson_sampling.png")
    plt.close()


def analyze_sampling_quality():
    """分析采样质量 - 距离分布"""
    print("\n分析采样质量...")
    
    np.random.seed(42)
    n_points = 2000
    original_points = np.random.rand(n_points, 3) * 10
    
    min_radius = 0.8
    sampler = PoissonDiskSampler3D(min_radius=min_radius)
    sampled = sampler.sample(original_points)
    
    print(f"采样点数: {len(sampled)}")
    
    distances = []
    for i in range(len(sampled)):
        for j in range(i+1, len(sampled)):
            dist = np.linalg.norm(sampled[i] - sampled[j])
            distances.append(dist)
    
    distances = np.array(distances)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(distances, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=min_radius, color='red', linestyle='--', label=f'min_radius = {min_radius}')
    axes[0].set_xlabel('点间距离')
    axes[0].set_ylabel('频数')
    axes[0].set_title('采样点间距分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    min_distances = []
    for i in range(len(sampled)):
        other_points = np.delete(sampled, i, axis=0)
        dists = np.linalg.norm(other_points - sampled[i], axis=1)
        min_distances.append(np.min(dists))
    
    min_distances = np.array(min_distances)
    
    axes[1].hist(min_distances, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=min_radius, color='red', linestyle='--', label=f'min_radius = {min_radius}')
    axes[1].set_xlabel('最近邻距离')
    axes[1].set_ylabel('频数')
    axes[1].set_title('最近邻距离分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sampling_quality.png', dpi=150, bbox_inches='tight')
    print("已保存: sampling_quality.png")
    plt.close()
    
    print(f"最小间距统计:")
    print(f"  理论最小间距: {min_radius}")
    print(f"  实际最小最近邻距离: {np.min(min_distances):.4f}")
    print(f"  平均最近邻距离: {np.mean(min_distances):.4f}")
    print(f"  所有点满足最小距离约束: {np.all(min_distances >= min_radius * 0.99)}")


if __name__ == "__main__":
    print("=" * 60)
    print("泊松圆盘采样可视化演示")
    print("=" * 60)
    
    visualize_point_cloud_comparison()
    visualize_bridson_sampling()
    analyze_sampling_quality()
    
    print("\n" + "=" * 60)
    print("所有可视化图片已生成完成！")
    print("=" * 60)
