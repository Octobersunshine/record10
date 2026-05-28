import numpy as np
import time
from hierarchical_clustering import agglomerative_clustering, compare_memory_usage, _original_agglomerative

print('测试平均链连接:')
data = np.array([[1,2],[2,3],[5,5],[6,7],[10,1]])
dendro = agglomerative_clustering(data=data, metric='euclidean', linkage='average', optimized=True)
print(f'平均链步骤数: {len(dendro)}')
for step in dendro:
    print(f"  合并: {step['cluster1']} & {step['cluster2']}, 距离: {step['distance']:.4f}")

print()
print('测试500样本性能 (单链):')
np.random.seed(42)
data_large = np.random.rand(500, 2)
start = time.time()
dendro_large = agglomerative_clustering(data=data_large, metric='euclidean', linkage='single', optimized=True)
print(f'500样本耗时: {time.time() - start:.2f}秒')
print(f'聚类树步骤数: {len(dendro_large)}')

print()
print('测试300样本性能 (全链):')
np.random.seed(42)
data_large = np.random.rand(300, 2)
start = time.time()
dendro_large = agglomerative_clustering(data=data_large, metric='euclidean', linkage='complete', optimized=True)
print(f'300样本耗时: {time.time() - start:.2f}秒')
print(f'聚类树步骤数: {len(dendro_large)}')

print()
print('10000样本内存对比:')
compare_memory_usage(10000)

print()
print('验证平均链结果一致性:')
dendro_original = _original_agglomerative(data=data, metric='euclidean', linkage='average')
dendro_optimized = agglomerative_clustering(data=data, metric='euclidean', linkage='average', optimized=True)

print('\n原始算法结果:')
for step in dendro_original:
    print(f"  合并: {step['cluster1']} & {step['cluster2']}, 距离: {step['distance']:.4f}")

print('\n优化算法结果:')
for step in dendro_optimized:
    print(f"  合并: {step['cluster1']} & {step['cluster2']}, 距离: {step['distance']:.4f}")
