import numpy as np
from dbscan import hdbscan, silhouette_score

print("=" * 60)
print("HDBSCAN 测试")
print("=" * 60)

np.random.seed(42)
dense = np.random.randn(15, 2) * 0.2 + [0, 0]
sparse = np.random.randn(12, 2) * 0.8 + [5, 5]
X = np.vstack([dense, sparse])

print(f"\n数据集: 高密度簇(15点) + 低密度簇(12点), 共 {len(X)} 点")

labels, types, info = hdbscan(X, min_samples=5, min_cluster_size=5)
core = sum(1 for t in types if t == 'core')
border = sum(1 for t in types if t == 'border')
noise = sum(1 for t in types if t == 'noise')

print(f"\nHDBSCAN 结果:")
print(f"  簇数量: {labels.max() + 1 if labels.max() >= 0 else 0}")
print(f"  核心点: {core}, 边界点: {border}, 噪声点: {noise}")
print(f"  高密度簇识别: {np.sum(labels[:15] != -1)}/15")
print(f"  低密度簇识别: {np.sum(labels[15:] != -1)}/12")
print(f"  轮廓系数: {info['silhouette_score']:.4f}")
print(f"  层次树节点数: {len(info['hierarchy'])}")

print("\n聚类标签:")
print(f"  高密度簇: {labels[:15]}")
print(f"  低密度簇: {labels[15:]}")

print("\n✓ HDBSCAN 测试完成!")
