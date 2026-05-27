"""
高级泊松圆盘采样使用示例

展示如何使用改进的算法解决 Dart Throwing 收敛慢的问题
"""

import numpy as np
from advanced_poisson_disk import (
    PrioritySampler,
    GapFillingSampler, 
    MaximalPoissonDiskSampler
)


def example_1_basic_usage():
    """基础使用示例"""
    print("=" * 70)
    print("示例 1: 基础使用 - 最大泊松圆盘采样")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_points = 10000
    print(f"\n生成 {n_points} 个随机点...")
    points = np.random.rand(n_points, 3) * 10
    
    min_radius = 0.6
    print(f"最小距离约束: {min_radius}")
    
    sampler = MaximalPoissonDiskSampler(min_radius=min_radius)
    
    target_count = 3000
    print(f"目标采样点数: {target_count} (目标采样率: {target_count/n_points*100:.1f}%)")
    
    print("\n执行采样...")
    samples = sampler.sample(points, target_count=target_count)
    
    print(f"\n采样结果:")
    print(f"  原始点数: {n_points}")
    print(f"  采样点数: {len(samples)}")
    print(f"  实际采样率: {len(samples)/n_points*100:.1f}%")
    print(f"  压缩率: {(1 - len(samples)/n_points)*100:.1f}%")
    
    from scipy.spatial import KDTree
    min_dist = float('inf')
    if len(samples) > 1:
        tree = KDTree(samples)
        for i in range(len(samples)):
            dist, _ = tree.query(samples[i], k=2)
            if len(dist) > 1:
                min_dist = min(min_dist, dist[1])
    
    print(f"  最小最近邻距离: {min_dist:.4f}")
    print(f"  满足最小距离约束: {min_dist >= min_radius * 0.99}")


def example_2_high_density_sampling():
    """高采样率场景 - 展示改进算法的优势"""
    print("\n" + "=" * 70)
    print("示例 2: 高采样率场景 - Dart Throwing 的痛点")
    print("=" * 70)
    
    from poisson_disk_sampling import PoissonDiskSampler3D
    
    np.random.seed(42)
    
    n_points = 20000
    points = np.random.rand(n_points, 3) * 20
    
    min_radius = 0.8
    target_count = 8000
    
    print(f"\n测试场景:")
    print(f"  输入点数: {n_points}")
    print(f"  最小距离: {min_radius}")
    print(f"  目标采样率: {target_count/n_points*100:.1f}%")
    
    print("\n--- 传统 Dart Throwing 算法 ---")
    import time
    start = time.time()
    dart_sampler = PoissonDiskSampler3D(min_radius=min_radius, max_samples=target_count)
    dart_samples = dart_sampler.sample(points)
    dart_time = time.time() - start
    
    print(f"  采样点数: {len(dart_samples)}")
    print(f"  实际采样率: {len(dart_samples)/n_points*100:.1f}%")
    print(f"  耗时: {dart_time:.3f}秒")
    
    print("\n--- 改进的最大泊松圆盘采样 ---")
    start = time.time()
    max_sampler = MaximalPoissonDiskSampler(min_radius=min_radius)
    max_samples = max_sampler.sample(points, target_count=target_count)
    max_time = time.time() - start
    
    print(f"  采样点数: {len(max_samples)}")
    print(f"  实际采样率: {len(max_samples)/n_points*100:.1f}%")
    print(f"  耗时: {max_time:.3f}秒")
    
    print("\n--- 对比结果 ---")
    count_improvement = (len(max_samples) - len(dart_samples)) / len(dart_samples) * 100
    speedup = dart_time / max_time if max_time > 0 else float('inf')
    
    print(f"  填充率提升: {'+' if count_improvement >= 0 else ''}{count_improvement:.1f}%")
    print(f"  速度比: {speedup:.1f}x")
    
    if count_improvement > 0:
        print(f"\n✓ 改进算法成功解决了 Dart Throwing 在高采样率下收敛慢的问题！")


def example_3_different_algorithms():
    """对比不同算法的适用场景"""
    print("\n" + "=" * 70)
    print("示例 3: 不同算法的选择指南")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_points = 8000
    points = np.random.rand(n_points, 3) * 10
    
    test_cases = [
        {'name': '低采样率 (10%)', 'min_radius': 1.5, 'target_rate': 0.1},
        {'name': '中采样率 (25%)', 'min_radius': 0.9, 'target_rate': 0.25},
        {'name': '高采样率 (40%)', 'min_radius': 0.6, 'target_rate': 0.4},
    ]
    
    for tc in test_cases:
        print(f"\n场景: {tc['name']}")
        print("-" * 50)
        
        target_count = int(n_points * tc['target_rate'])
        
        algorithms = [
            ("优先级采样", PrioritySampler(min_radius=tc['min_radius'])),
            ("间隙填充", GapFillingSampler(min_radius=tc['min_radius'])),
            ("最大泊松圆盘", MaximalPoissonDiskSampler(min_radius=tc['min_radius'])),
        ]
        
        results = []
        for name, sampler in algorithms:
            import time
            start = time.time()
            samples = sampler.sample(points, target_count=target_count)
            elapsed = time.time() - start
            
            from scipy.spatial import KDTree
            min_dist = float('inf')
            if len(samples) > 1:
                tree = KDTree(samples)
                for i in range(len(samples)):
                    dist, _ = tree.query(samples[i], k=2)
                    if len(dist) > 1:
                        min_dist = min(min_dist, dist[1])
            
            valid = min_dist >= tc['min_radius'] * 0.99
            rate = len(samples) / n_points * 100
            
            results.append({
                'name': name, 'count': len(samples), 
                'rate': rate, 'time': elapsed, 'valid': valid
            })
            
            print(f"  {name}: {len(samples)}点 ({rate:.1f}%), {elapsed:.3f}s, {'✓' if valid else '✗'}")
    
    print("\n算法选择指南:")
    print("  • 低采样率场景: 优先级采样 (最快)")
    print("  • 中采样率场景: 间隙填充 (均衡)")
    print("  • 高采样率场景: 最大泊松圆盘 (填充率最高)")
    print("  • 所有场景通用: 最大泊松圆盘 (最稳健)")


def example_4_point_cloud_simplification():
    """点云简化实际应用"""
    print("\n" + "=" * 70)
    print("示例 4: 实际应用 - 点云简化")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n生成模拟点云...")
    n_original = 50000
    
    theta = np.random.uniform(0, 2*np.pi, n_original)
    phi = np.arccos(np.random.uniform(-1, 1, n_original))
    r = 30 + np.random.normal(0, 5, n_original)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    point_cloud = np.column_stack([x, y, z])
    
    print(f"原始点云: {n_original} 个点")
    
    target_counts = [10000, 5000, 2000, 1000]
    
    for target in target_counts:
        print(f"\n--- 简化到 {target} 点 ---")
        
        min_radius = 1.5
        sampler = MaximalPoissonDiskSampler(min_radius=min_radius)
        
        import time
        start = time.time()
        simplified = sampler.sample(point_cloud, target_count=target)
        elapsed = time.time() - start
        
        from scipy.spatial import KDTree
        min_dist = float('inf')
        if len(simplified) > 1:
            tree = KDTree(simplified)
            for i in range(len(simplified)):
                dist, _ = tree.query(simplified[i], k=2)
                if len(dist) > 1:
                    min_dist = min(min_dist, dist[1])
        
        valid = min_dist >= min_radius * 0.99
        
        print(f"  简化后: {len(simplified)} 点")
        print(f"  压缩率: {(1 - len(simplified)/n_original)*100:.1f}%")
        print(f"  耗时: {elapsed:.3f}秒")
        print(f"  最小距离约束: {'✓' if valid else '✗'}")


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_high_density_sampling()
    example_3_different_algorithms()
    example_4_point_cloud_simplification()
    
    print("\n" + "=" * 70)
    print("所有示例执行完成！")
    print("=" * 70)
