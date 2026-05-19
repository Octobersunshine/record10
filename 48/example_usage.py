import numpy as np
from persistent_homology import (
    compute_persistent_homology,
    normalize_point_cloud,
    generate_sample_point_cloud,
    barcodes_to_persistence_image,
    compute_persistence_images,
    persistence_images_to_vector,
    print_barcodes_summary
)


print("=" * 70)
print("持久图像 (Persistence Image) 完整使用指南")
print("=" * 70)


print("\n=== 示例 1: 基础持久图像转换 ===")
np.random.seed(42)
circle_cloud = generate_sample_point_cloud('circle', n_points=60)

result = compute_persistent_homology(
    point_cloud=circle_cloud,
    max_dimension=1,
    normalize=True,
    normalize_method='standard'
)

h1_barcode = result['barcodes'][1]
print(f"H1条形码数量: {len(h1_barcode)}")

image = barcodes_to_persistence_image(
    h1_barcode,
    resolution=(32, 32),
    weight_func='linear',
    sigma=0.1,
    normalize=True
)

print(f"持久图像形状: {image.shape}")
print(f"持久图像范围: [{image.min():.4f}, {image.max():.4f}]")
print(f"非零像素数: {np.sum(image > 0.01)} / {image.size}")


print("\n=== 示例 2: 不同权重函数对比 ===")
for weight_name in ['uniform', 'linear', 'logarithmic']:
    img = barcodes_to_persistence_image(
        h1_barcode,
        resolution=(32, 32),
        weight_func=weight_name,
        sigma=0.1
    )
    print(f"{weight_name:12s}: 总和={img.sum():8.4f}, 最大值={img.max():.4f}")


print("\n=== 示例 3: 不同分辨率对比 ===")
for res in [(16, 16), (32, 32), (64, 64)]:
    img = barcodes_to_persistence_image(h1_barcode, resolution=res)
    print(f"分辨率 {res}: 总维度={img.size:4d}, 均值={img.mean():.6f}")


print("\n=== 示例 4: 批量生成多维度持久图像 ===")
sphere_cloud = generate_sample_point_cloud('sphere', n_points=80)
result_sphere = compute_persistent_homology(
    point_cloud=sphere_cloud,
    max_dimension=2,
    normalize=True
)

images = compute_persistence_images(
    result_sphere,
    resolution=(40, 40),
    weight_func='linear',
    sigma=0.15,
    shared_range=True,
    normalize=True
)

for dim in sorted(images.keys()):
    print(f"H{dim}图像形状: {images[dim].shape}, 范围: [{images[dim].min():.3f}, {images[dim].max():.3f}]")


print("\n=== 示例 5: 转换为机器学习特征向量 ===")
feature_vector = persistence_images_to_vector(images)
print(f"完整特征向量形状: {feature_vector.shape}")
print(f"特征向量维度 = 3维 × 40×40 = {3 * 40 * 40}")

feature_h1 = persistence_images_to_vector(images, dims=[1])
print(f"仅H1特征向量形状: {feature_h1.shape}")

feature_h0h1 = persistence_images_to_vector(images, dims=[0, 1])
print(f"H0+H1特征向量形状: {feature_h0h1.shape}")

print(f"\n特征向量统计:")
print(f"  均值 = {feature_vector.mean():.6f}")
print(f"  标准差 = {feature_vector.std():.6f}")
print(f"  L2范数 = {np.linalg.norm(feature_vector):.4f}")


print("\n=== 示例 6: 模拟机器学习数据集 ===")
print("(不同拓扑形状 → 不同持久图像 → 分类任务)")

shapes = ['circle', 'sphere', 'torus', 'cloud']
X = []
y = []

for label, shape_type in enumerate(shapes):
    for _ in range(3):
        cloud = generate_sample_point_cloud(shape_type, n_points=60)
        result = compute_persistent_homology(cloud, max_dimension=2, normalize=True)
        images = compute_persistence_images(result, resolution=(20, 20))
        features = persistence_images_to_vector(images)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"\n数据集统计:")
print(f"  X形状 (样本数 × 特征维度): {X.shape}")
print(f"  y形状 (标签数量): {y.shape}")
print(f"  形状类别: {shapes}")
print(f"  每类样本数: 3")
print(f"\n特征矩阵可直接用于:")
print(f"  - SVM, Random Forest, XGBoost")
print(f"  - 神经网络分类/回归")
print(f"  - 聚类分析 (K-means, DBSCAN)")


print("\n=== 示例 7: 参数调优建议 ===")
print("分辨率选择:")
print("  - 快速实验: (16, 16) → 256维/维")
print("  - 标准任务: (32, 32) → 1024维/维")
print("  - 高精度: (64, 64) → 4096维/维")
print("\nSigma (高斯带宽)选择:")
print("  - 0.05: 尖锐峰, 细节丰富")
print("  - 0.1:  平衡 (推荐)")
print("  - 0.2:  平滑, 抗噪")
print("\n权重函数选择:")
print("  - uniform: 所有点同等重要")
print("  - linear:  长生命周期权重更高 (推荐)")
print("  - logarithmic: 超长生命周期权重更高")


print("\n" + "=" * 70)
print("持久图像转换完成!")
print(f"  - 条形码(变长) → 持久图像(固定大小) → 特征向量(固定长度)")
print(f"  - 适用于: 分类、回归、聚类、降维可视化等机器学习任务")
print("=" * 70)
