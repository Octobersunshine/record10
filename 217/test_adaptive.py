"""测试自适应泊松圆盘采样"""
import numpy as np
from scipy.spatial import KDTree
from adaptive_poisson_disk import (
    LocalFeatureAnalyzer, 
    AdaptivePoissonDiskSampler,
    CurvatureAwareSimplifier
)

np.random.seed(42)

print("=" * 70)
print("自适应泊松圆盘采样测试")
print("=" * 70)

print("\n生成测试点云...")
n_points = 5000
theta = np.random.uniform(0, 2*np.pi, n_points)
phi = np.arccos(np.random.uniform(-1, 1, n_points))
r = 10 + 3 * np.sin(5 * theta) * np.cos(3 * phi)

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)
points = np.column_stack([x, y, z])

print(f"点云大小: {len(points)} 个点")

print("\n测试局部特征分析...")
analyzer = LocalFeatureAnalyzer(k_neighbors=10)
features = analyzer.analyze(points)

print(f"曲率范围: [{np.min(features['curvatures']):.4f}, {np.max(features['curvatures']):.4f}]")
print(f"特征密度范围: [{np.min(features['feature_density']):.4f}, {np.max(features['feature_density']):.4f}]")
print(f"局部尺度范围: [{np.min(features['local_scales']):.4f}, {np.max(features['local_scales']):.4f}]")

print("\n测试自适应采样...")
sampler = AdaptivePoissonDiskSampler(
    base_radius=1.0,
    min_radius_factor=0.3,
    max_radius_factor=2.0,
    k_neighbors=10
)

samples, indices, features = sampler.sample(points, target_count=500)

print(f"采样完成: {len(samples)} 个点")

high_curv_threshold = np.percentile(features['curvatures'], 75)
orig_high = np.sum(features['curvatures'] >= high_curv_threshold)
sampled_high = np.sum(features['curvatures'][indices] >= high_curv_threshold)

print(f"原始高曲率点数: {orig_high} ({orig_high/n_points*100:.1f}%)")
print(f"采样后高曲率点数: {sampled_high} ({sampled_high/len(samples)*100:.1f}%)")
enrichment = sampled_high/len(samples) / (orig_high/n_points)
print(f"高曲率点富集: {enrichment:.2f}x")

print("\n测试曲率感知简化器...")
simplifier = CurvatureAwareSimplifier(target_count=300)
simplified, simplified_indices = simplifier.simplify(points)

print(f"简化完成: {len(simplified)} 个点")

simplified_high_curv = np.sum(features['curvatures'][simplified_indices] >= high_curv_threshold)
print(f"简化后高曲率点数: {simplified_high_curv} ({simplified_high_curv/len(simplified)*100:.1f}%)")

print("\n✓ 自适应采样测试成功！")
