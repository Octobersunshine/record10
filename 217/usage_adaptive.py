"""
自适应泊松圆盘采样 - 完整使用示例

展示如何根据曲率和特征密度进行自适应采样
"""

import numpy as np
from adaptive_poisson_disk import (
    LocalFeatureAnalyzer,
    AdaptiveRadiusEstimator,
    AdaptivePoissonDiskSampler,
    CurvatureAwareSimplifier
)


def example_1_basic_feature_analysis():
    """示例 1: 基本局部特征分析"""
    print("=" * 70)
    print("示例 1: 局部特征分析")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n生成带特征的测试点云...")
    n_points = 2000
    
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.arccos(np.random.uniform(-1, 1, n_points))
    r = 10 + 2 * np.sin(4 * theta) * np.cos(2 * phi)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.column_stack([x, y, z])
    
    print(f"点云大小: {n_points} 个点")
    
    print("\n分析局部特征...")
    analyzer = LocalFeatureAnalyzer(k_neighbors=15)
    features = analyzer.analyze(points)
    
    print("\n特征统计:")
    print(f"  曲率:")
    print(f"    最小值: {np.min(features['curvatures']):.4f}")
    print(f"    最大值: {np.max(features['curvatures']):.4f}")
    print(f"    平均值: {np.mean(features['curvatures']):.4f}")
    
    print(f"  特征密度:")
    print(f"    最小值: {np.min(features['feature_density']):.4f}")
    print(f"    最大值: {np.max(features['feature_density']):.4f}")
    print(f"    平均值: {np.mean(features['feature_density']):.4f}")
    
    print(f"  局部尺度:")
    print(f"    最小值: {np.min(features['local_scales']):.4f}")
    print(f"    最大值: {np.max(features['local_scales']):.4f}")
    print(f"    平均值: {np.mean(features['local_scales']):.4f}")
    
    return points, features


def example_2_adaptive_radius_estimation():
    """示例 2: 自适应半径估计"""
    print("\n" + "=" * 70)
    print("示例 2: 自适应采样半径估计")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_points = 2000
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.arccos(np.random.uniform(-1, 1, n_points))
    r = 10 + 2 * np.sin(4 * theta) * np.cos(2 * phi)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.column_stack([x, y, z])
    
    analyzer = LocalFeatureAnalyzer(k_neighbors=10)
    features = analyzer.analyze(points)
    
    base_radius = 1.0
    print(f"\n基础采样半径: {base_radius}")
    
    print("\n配置不同的权重:")
    configs = [
        ("曲率主导", 0.8, 0.1, 0.1),
        ("均衡配置", 0.5, 0.3, 0.2),
        ("特征密度主导", 0.2, 0.7, 0.1),
    ]
    
    for name, curv_w, dens_w, scale_w in configs:
        estimator = AdaptiveRadiusEstimator(
            base_radius=base_radius,
            min_radius_factor=0.3,
            max_radius_factor=3.0,
            curvature_weight=curv_w,
            density_weight=dens_w,
            scale_weight=scale_w
        )
        
        radii = estimator.compute_adaptive_radii(points, features)
        
        print(f"\n  {name} (曲率:{curv_w}, 密度:{dens_w}, 尺度:{scale_w}):")
        print(f"    最小半径: {np.min(radii):.4f}")
        print(f"    最大半径: {np.max(radii):.4f}")
        print(f"    平均半径: {np.mean(radii):.4f}")
        
        high_curv_mask = features['curvatures'] > np.percentile(features['curvatures'], 75)
        low_curv_mask = features['curvatures'] < np.percentile(features['curvatures'], 25)
        
        print(f"    高曲率区域平均半径: {np.mean(radii[high_curv_mask]):.4f}")
        print(f"    低曲率区域平均半径: {np.mean(radii[low_curv_mask]):.4f}")


def example_3_adaptive_sampling():
    """示例 3: 自适应泊松圆盘采样"""
    print("\n" + "=" * 70)
    print("示例 3: 自适应泊松圆盘采样")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n生成测试点云...")
    n_points = 3000
    
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.arccos(np.random.uniform(-1, 1, n_points))
    r = 10 + 3 * np.sin(5 * theta) * np.cos(3 * phi)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.column_stack([x, y, z])
    
    print(f"原始点数: {n_points}")
    
    target_count = 500
    print(f"目标采样数: {target_count} (采样率: {target_count/n_points*100:.1f}%)")
    
    print("\n执行自适应采样...")
    sampler = AdaptivePoissonDiskSampler(
        base_radius=1.0,
        min_radius_factor=0.3,
        max_radius_factor=3.0,
        curvature_weight=0.6,
        density_weight=0.3,
        scale_weight=0.1,
        k_neighbors=10
    )
    
    samples, indices, features = sampler.sample(points, target_count=target_count)
    
    print(f"\n采样结果:")
    print(f"  采样点数: {len(samples)}")
    print(f"  实际采样率: {len(samples)/n_points*100:.1f}%")
    
    high_curv_threshold = np.percentile(features['curvatures'], 75)
    low_curv_threshold = np.percentile(features['curvatures'], 25)
    
    orig_high = np.sum(features['curvatures'] >= high_curv_threshold)
    orig_low = np.sum(features['curvatures'] <= low_curv_threshold)
    sampled_high = np.sum(features['curvatures'][indices] >= high_curv_threshold)
    sampled_low = np.sum(features['curvatures'][indices] <= low_curv_threshold)
    
    print(f"\n曲率区域分析:")
    print(f"  高曲率区域 (Top 25%):")
    print(f"    原始: {orig_high} 点 ({orig_high/n_points*100:.1f}%)")
    print(f"    采样: {sampled_high} 点 ({sampled_high/len(samples)*100:.1f}%)")
    print(f"    富集倍数: {sampled_high/len(samples) / (orig_high/n_points):.2f}x")
    
    print(f"  低曲率区域 (Bottom 25%):")
    print(f"    原始: {orig_low} 点 ({orig_low/n_points*100:.1f}%)")
    print(f"    采样: {sampled_low} 点 ({sampled_low/len(samples)*100:.1f}%)")
    print(f"    稀疏倍数: {sampled_low/len(samples) / (orig_low/n_points):.2f}x")


def example_4_curvature_aware_simplification():
    """示例 4: 曲率感知的点云简化"""
    print("\n" + "=" * 70)
    print("示例 4: 曲率感知的点云简化")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n生成复杂点云...")
    n_original = 10000
    
    theta = np.random.uniform(0, 2*np.pi, n_original)
    phi = np.arccos(np.random.uniform(-1, 1, n_original))
    r = 20 + 5 * np.sin(6 * theta) * np.cos(4 * phi) + 2 * np.sin(10 * phi)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    point_cloud = np.column_stack([x, y, z])
    
    print(f"原始点云: {n_original} 个点")
    
    simplification_levels = [5000, 2000, 1000, 500]
    
    for target in simplification_levels:
        print(f"\n--- 简化到 {target} 点 ---")
        
        simplifier = CurvatureAwareSimplifier(
            target_count=target,
            sensitivity=1.5
        )
        
        simplified, indices = simplifier.simplify(point_cloud)
        
        compression_rate = (1 - len(simplified)/n_original) * 100
        print(f"  简化后: {len(simplified)} 点")
        print(f"  压缩率: {compression_rate:.1f}%")


def example_5_compare_uniform_vs_adaptive():
    """示例 5: 均匀采样 vs 自适应采样对比"""
    print("\n" + "=" * 70)
    print("示例 5: 均匀采样 vs 自适应采样 对比")
    print("=" * 70)
    
    from advanced_poisson_disk import MaximalPoissonDiskSampler
    
    np.random.seed(42)
    
    print("\n生成带丰富特征的测试点云...")
    n_points = 5000
    
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.arccos(np.random.uniform(-1, 1, n_points))
    r = 10 + 4 * np.sin(5 * theta) * np.cos(3 * phi)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.column_stack([x, y, z])
    
    print(f"原始点数: {n_points}")
    
    target_count = 500
    print(f"目标采样数: {target_count}")
    
    print("\n--- 均匀采样 ---")
    bbox_diag = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
    density = target_count / (bbox_diag ** 3)
    uniform_radius = 1 / (2 * np.cbrt(density * np.pi * 4 / 3))
    
    uniform_sampler = MaximalPoissonDiskSampler(min_radius=uniform_radius)
    uniform_samples = uniform_sampler.sample(points, target_count=target_count)
    print(f"  采样点数: {len(uniform_samples)}")
    
    print("\n--- 自适应采样 ---")
    adaptive_sampler = AdaptivePoissonDiskSampler(
        base_radius=uniform_radius,
        min_radius_factor=0.3,
        max_radius_factor=3.0,
        curvature_weight=0.6,
        density_weight=0.3,
        scale_weight=0.1,
        k_neighbors=10
    )
    adaptive_samples, adaptive_indices, features = adaptive_sampler.sample(
        points, target_count=target_count
    )
    print(f"  采样点数: {len(adaptive_samples)}")
    
    print("\n--- 特征保留对比 ---")
    analyzer = LocalFeatureAnalyzer(k_neighbors=10)
    
    uniform_features = analyzer.analyze(uniform_samples)
    uniform_curvatures = uniform_features['curvatures']
    
    adaptive_curvatures = features['curvatures'][adaptive_indices]
    
    high_curv_threshold = np.percentile(features['curvatures'], 75)
    
    orig_high_ratio = np.sum(features['curvatures'] >= high_curv_threshold) / n_points * 100
    uniform_high_ratio = np.sum(uniform_curvatures >= high_curv_threshold) / len(uniform_samples) * 100
    adaptive_high_ratio = np.sum(adaptive_curvatures >= high_curv_threshold) / len(adaptive_samples) * 100
    
    print(f"  原始高曲率点比例: {orig_high_ratio:.1f}%")
    print(f"  均匀采样保留高曲率点: {uniform_high_ratio:.1f}%")
    print(f"  自适应采样保留高曲率点: {adaptive_high_ratio:.1f}%")
    
    if adaptive_high_ratio > uniform_high_ratio:
        improvement = adaptive_high_ratio - uniform_high_ratio
        print(f"\n✓ 自适应采样在高曲率特征保留上提升了 {improvement:.1f} 个百分点")


def example_6_sensitivity_tuning():
    """示例 6: 曲率敏感度调优"""
    print("\n" + "=" * 70)
    print("示例 6: 曲率敏感度调优")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_points = 3000
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.arccos(np.random.uniform(-1, 1, n_points))
    r = 10 + 3 * np.sin(5 * theta) * np.cos(3 * phi)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.column_stack([x, y, z])
    
    target_count = 300
    
    print(f"\n原始点数: {n_points}, 目标采样数: {target_count}")
    print("\n不同敏感度的效果:")
    
    sensitivities = [0.5, 1.0, 1.5, 2.0]
    
    analyzer = LocalFeatureAnalyzer(k_neighbors=10)
    features = analyzer.analyze(points)
    high_curv_threshold = np.percentile(features['curvatures'], 75)
    
    for sens in sensitivities:
        simplifier = CurvatureAwareSimplifier(
            target_count=target_count,
            sensitivity=sens
        )
        
        simplified, indices = simplifier.simplify(points)
        
        high_curv_count = np.sum(features['curvatures'][indices] >= high_curv_threshold)
        high_curv_ratio = high_curv_count / len(simplified) * 100
        
        print(f"  敏感度={sens}: 高曲率点比例={high_curv_ratio:.1f}%")


if __name__ == "__main__":
    example_1_basic_feature_analysis()
    example_2_adaptive_radius_estimation()
    example_3_adaptive_sampling()
    example_4_curvature_aware_simplification()
    example_5_compare_uniform_vs_adaptive()
    example_6_sensitivity_tuning()
    
    print("\n" + "=" * 70)
    print("所有示例执行完成！")
    print("=" * 70)
