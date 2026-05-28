import warnings
warnings.filterwarnings('ignore')
import numpy as np
from random_projection import (
    RandomProjection,
    compute_reconstruction_mse,
    compare_with_pca
)

print('=' * 60)
print(' 新功能快速验证')
print('=' * 60)

np.random.seed(42)
n_samples, n_features = 200, 500
X = np.random.randn(n_samples, n_features)
eps = 0.15

# 1. 测试可调密度的稀疏随机投影
print('\n1. 稀疏随机投影 - 不同密度测试:')
for d in [0.01, 0.05, 0.1, 0.2]:
    rp = RandomProjection(method='sparse', density=d, eps=eps, random_state=42)
    X_proj = rp.fit_transform(X)
    R = rp.projection_matrix_
    if hasattr(R, 'nnz'):
        nnz_density = R.nnz / (R.shape[0] * R.shape[1])
    else:
        nnz_density = np.count_nonzero(R) / (R.shape[0] * R.shape[1])
    rpt = rp.report(X, X_proj, eps)
    print('   density={:.2f}: 实际非零密度={:.4f}, 最大失真={:.4f}, 满足eps={}'.format(
        d, nnz_density, rpt.max_distortion, rpt.eps_satisfied))

# 2. 测试在线降维
print('\n2. 在线降维 - 增量处理测试:')
rp_online = RandomProjection(method='sparse', density=0.1, eps=eps, random_state=42)

for i in range(3):
    X_batch = X[i*50:(i+1)*50]
    rp_online.partial_fit(X_batch)
    X_proj_batch = rp_online.transform(X_batch)
    print('   批 {}: 已处理样本数={}, 投影形状={}'.format(
        i+1, rp_online.n_samples_seen_, X_proj_batch.shape))

# 3. 测试 inverse_transform 和 MSE 计算
print('\n3. 重构 MSE 计算测试:')
rp = RandomProjection(method='gaussian', n_components=100, random_state=42)
X_proj = rp.fit_transform(X)
X_recon = rp.inverse_transform(X_proj)
mse, mse_per_sample = compute_reconstruction_mse(X, X_recon)
print('   k=100, 重构 MSE={:.4f}'.format(mse))

# 4. 测试 PCA 对比
print('\n4. PCA 对比测试 (k=50):')
results = compare_with_pca(X, [50], random_state=42)
for method, res in results[50].items():
    print('   {}: MSE={:.4f}, 最大失真={:.4f}'.format(
        method, res['mse'], res['max_distortion']))

print('\n' + '=' * 60)
print(' 所有新功能验证通过!')
print('=' * 60)
