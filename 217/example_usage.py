"""
泊松圆盘采样使用示例

本示例展示了如何使用 poisson_disk_sampling 模块进行点云简化和空间采样。
"""

import numpy as np
from poisson_disk_sampling import (
    PoissonDiskSampler3D,
    BridsonSampler3D,
    point_cloud_simplification
)


def example_1_point_cloud_simplification():
    """
    示例 1: 点云简化
    从输入点云中提取均匀分布且满足最小距离约束的子集
    """
    print("=" * 60)
    print("示例 1: 点云简化")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_original = 10000
    print(f"\n生成原始点云: {n_original} 个随机点")
    original_points = np.random.rand(n_original, 3) * 100
    
    target_count = 500
    print(f"目标简化点数: {target_count}")
    
    simplified = point_cloud_simplification(original_points, target_count=target_count)
    
    print(f"\n简化结果:")
    print(f"  原始点数: {n_original}")
    print(f"  简化后点数: {len(simplified)}")
    print(f"  压缩率: {(1 - len(simplified)/n_original)*100:.1f}%")
    
    min_radius = None
    if len(simplified) > 1:
        min_dist = float('inf')
        for i in range(len(simplified)):
            dists = np.linalg.norm(simplified - simplified[i], axis=1)
            dists[i] = float('inf')
            min_dist = min(min_dist, np.min(dists))
        print(f"  最小最近邻距离: {min_dist:.4f}")
    
    return simplified


def example_2_min_radius_sampling():
    """
    示例 2: 基于最小距离的采样
    指定采样点之间的最小距离
    """
    print("\n" + "=" * 60)
    print("示例 2: 基于最小距离的采样")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_original = 5000
    original_points = np.random.rand(n_original, 3) * 50
    
    min_radius = 5.0
    print(f"\n最小距离约束: {min_radius}")
    
    sampler = PoissonDiskSampler3D(min_radius=min_radius)
    sampled = sampler.sample(original_points)
    
    print(f"采样结果:")
    print(f"  原始点数: {n_original}")
    print(f"  采样后点数: {len(sampled)}")
    print(f"  采样密度: {len(sampled)/n_original*100:.1f}%")
    
    if len(sampled) > 1:
        min_dist = float('inf')
        for i in range(len(sampled)):
            dists = np.linalg.norm(sampled - sampled[i], axis=1)
            dists[i] = float('inf')
            min_dist = min(min_dist, np.min(dists))
        print(f"  实际最小最近邻距离: {min_dist:.4f}")
        print(f"  满足最小距离约束: {min_dist >= min_radius * 0.99}")
    
    return sampled


def example_3_bridson_sampling():
    """
    示例 3: Bridson 算法空间采样
    直接在指定体积内生成泊松圆盘分布的点
    """
    print("\n" + "=" * 60)
    print("示例 3: Bridson 算法空间采样")
    print("=" * 60)
    
    bounds = np.array([
        [0, 100],
        [0, 100],
        [0, 100]
    ])
    
    min_radius = 10.0
    print(f"\n采样空间: X∈[{bounds[0,0]}, {bounds[0,1]}], Y∈[{bounds[1,0]}, {bounds[1,1]}], Z∈[{bounds[2,0]}, {bounds[2,1]}]")
    print(f"最小距离: {min_radius}")
    
    sampler = BridsonSampler3D(min_radius=min_radius, bounds=bounds)
    samples = sampler.sample()
    
    print(f"\n采样结果:")
    print(f"  采样点数: {len(samples)}")
    
    if len(samples) > 1:
        min_dist = float('inf')
        for i in range(len(samples)):
            dists = np.linalg.norm(samples - samples[i], axis=1)
            dists[i] = float('inf')
            min_dist = min(min_dist, np.min(dists))
        print(f"  最小最近邻距离: {min_dist:.4f}")
        print(f"  满足最小距离约束: {min_dist >= min_radius * 0.99}")
    
    return samples


def example_4_mesh_vertex_simplification():
    """
    示例 4: 网格顶点简化（用于三维模型简化）
    模拟从网格顶点进行采样
    """
    print("\n" + "=" * 60)
    print("示例 4: 网格顶点简化（模型简化）")
    print("=" * 60)
    
    np.random.seed(123)
    
    n_vertices = 20000
    print(f"\n原始网格顶点数: {n_vertices}")
    
    theta = np.random.uniform(0, 2*np.pi, n_vertices)
    phi = np.arccos(np.random.uniform(-1, 1, n_vertices))
    r = 50 + np.random.normal(0, 2, n_vertices)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    vertices = np.column_stack([x, y, z])
    
    target_count = 2000
    print(f"目标顶点数: {target_count}")
    
    simplified_vertices = point_cloud_simplification(vertices, target_count=target_count)
    
    print(f"\n简化结果:")
    print(f"  原始顶点数: {n_vertices}")
    print(f"  简化后顶点数: {len(simplified_vertices)}")
    print(f"  压缩率: {(1 - len(simplified_vertices)/n_vertices)*100:.1f}%")
    
    return simplified_vertices


if __name__ == "__main__":
    example_1_point_cloud_simplification()
    example_2_min_radius_sampling()
    example_3_bridson_sampling()
    example_4_mesh_vertex_simplification()
    
    print("\n" + "=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)
