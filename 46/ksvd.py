import numpy as np


def orthogonal_mp(D, Y, n_nonzero_coefs):
    """
    正交匹配追踪算法实现
    
    参数:
        D: 字典矩阵，形状为 (n_features, n_atoms)
        Y: 信号矩阵，形状为 (n_features, n_samples)
        n_nonzero_coefs: 稀疏度（非零元素数量）
    
    返回:
        X: 稀疏系数矩阵，形状为 (n_atoms, n_samples)
    """
    n_features, n_atoms = D.shape
    n_samples = Y.shape[1]
    
    X = np.zeros((n_atoms, n_samples))
    
    for i in range(n_samples):
        y = Y[:, i]
        residual = y.copy()
        indices = []
        
        for _ in range(n_nonzero_coefs):
            correlations = D.T @ residual
            best_idx = np.argmax(np.abs(correlations))
            indices.append(best_idx)
            
            D_selected = D[:, indices]
            x_selected = np.linalg.lstsq(D_selected, y, rcond=None)[0]
            residual = y - D_selected @ x_selected
        
        X[indices, i] = x_selected
    
    return X


def ksvd(Y, n_atoms, max_iter=10, tol=1e-6, sparsity=5):
    """
    K-SVD算法实现
    
    参数:
        Y: 信号矩阵，形状为 (n_features, n_samples)
        n_atoms: 字典原子数量
        max_iter: 最大迭代次数
        tol: 收敛阈值
        sparsity: 稀疏表示的非零元素数量
    
    返回:
        D: 学习到的过完备字典，形状为 (n_features, n_atoms)
        X: 稀疏表示系数矩阵，形状为 (n_atoms, n_samples)
    """
    n_features, n_samples = Y.shape
    
    D = np.random.randn(n_features, n_atoms)
    D = D / np.linalg.norm(D, axis=0)
    
    for iter_num in range(max_iter):
        X = orthogonal_mp(D, Y, n_nonzero_coefs=sparsity)
        
        residual = Y - D @ X
        error = np.linalg.norm(residual)
        
        if error < tol:
            break
        
        for k in range(n_atoms):
            indices = np.where(X[k, :] != 0)[0]
            
            if len(indices) == 0:
                continue
            
            x_k_old = X[k, indices].copy()
            D[:, k] = 0
            E_k = Y[:, indices] - D @ X[:, indices]
            
            U, S, Vt = np.linalg.svd(E_k, full_matrices=False)
            
            d_k_new = U[:, 0]
            x_k_new = S[0] * Vt[0, :]
            
            if np.dot(x_k_old, x_k_new) < 0:
                d_k_new = -d_k_new
                x_k_new = -x_k_new
            
            D[:, k] = d_k_new
            X[k, indices] = x_k_new
        
        D = D / np.linalg.norm(D, axis=0)
    
    return D, X


class OnlineDictionaryLearning:
    """
    Mairal等人的在线字典学习算法实现
    
    支持流式数据更新，无需一次性加载全部信号
    """
    
    def __init__(self, n_atoms, n_features, sparsity=5, alpha=0.8, batch_size=1):
        """
        初始化在线字典学习器
        
        参数:
            n_atoms: 字典原子数量
            n_features: 信号特征维度
            sparsity: 稀疏表示的非零元素数量
            alpha: 遗忘因子，用于旧样本的权重衰减
            batch_size: 每批次处理的样本数
        """
        self.n_atoms = n_atoms
        self.n_features = n_features
        self.sparsity = sparsity
        self.alpha = alpha
        self.batch_size = batch_size
        
        self.D = np.random.randn(n_features, n_atoms)
        self.D = self.D / np.linalg.norm(self.D, axis=0)
        
        self.A = np.zeros((n_atoms, n_atoms))
        self.B = np.zeros((n_features, n_atoms))
        
        self.t = 0
        self.total_samples = 0
    
    def partial_fit(self, Y_batch):
        """
        使用一批新数据更新字典
        
        参数:
            Y_batch: 批次信号矩阵，形状为 (n_features, batch_size)
        
        返回:
            X_batch: 当前批次的稀疏系数
        """
        if Y_batch.ndim == 1:
            Y_batch = Y_batch.reshape(-1, 1)
        
        batch_size = Y_batch.shape[1]
        self.total_samples += batch_size
        
        X_batch = orthogonal_mp(self.D, Y_batch, n_nonzero_coefs=self.sparsity)
        
        eta = (self.alpha * self.t + batch_size) / (self.t + batch_size)
        
        A_batch = X_batch @ X_batch.T
        B_batch = Y_batch @ X_batch.T
        
        self.A = eta * self.A + (1 - eta) * A_batch
        self.B = eta * self.B + (1 - eta) * B_batch
        
        for k in range(self.n_atoms):
            if self.A[k, k] < 1e-10:
                continue
            
            d_k = (self.B[:, k] - self.D @ self.A[:, k]) / self.A[k, k] + self.D[:, k]
            d_k = d_k / (np.linalg.norm(d_k) + 1e-10)
            
            self.D[:, k] = d_k
        
        self.t += batch_size
        
        return X_batch
    
    def transform(self, Y):
        """
        对信号进行稀疏编码
        
        参数:
            Y: 信号矩阵，形状为 (n_features, n_samples)
        
        返回:
            X: 稀疏系数矩阵
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return orthogonal_mp(self.D, Y, n_nonzero_coefs=self.sparsity)
    
    def get_dictionary(self):
        """返回当前字典"""
        return self.D.copy()
    
    def reset(self):
        """重置学习器状态"""
        self.D = np.random.randn(self.n_features, self.n_atoms)
        self.D = self.D / np.linalg.norm(self.D, axis=0)
        self.A = np.zeros((self.n_atoms, self.n_atoms))
        self.B = np.zeros((self.n_features, self.n_atoms))
        self.t = 0
        self.total_samples = 0


if __name__ == "__main__":
    np.random.seed(42)
    
    n_features = 20
    n_samples = 100
    n_atoms = 50
    sparsity = 5
    
    Y = np.random.randn(n_features, n_samples)
    
    print(f"信号矩阵形状: {Y.shape}")
    print(f"字典原子数: {n_atoms}")
    print(f"稀疏度: {sparsity}")
    
    D, X = ksvd(Y, n_atoms, max_iter=20, sparsity=sparsity)
    
    print(f"\n学习到的字典形状: {D.shape}")
    print(f"稀疏系数矩阵形状: {X.shape}")
    
    reconstruction_error = np.linalg.norm(Y - D @ X)
    print(f"重构误差: {reconstruction_error:.4f}")
    
    avg_sparsity = np.mean(np.count_nonzero(X, axis=0))
    print(f"平均稀疏度: {avg_sparsity:.2f}")
    
    atom_energies = np.sum(X ** 2, axis=1)
    print(f"\n原子能量分布统计:")
    print(f"  最小能量: {np.min(atom_energies):.4f}")
    print(f"  最大能量: {np.max(atom_energies):.4f}")
    print(f"  平均能量: {np.mean(atom_energies):.4f}")
    print(f"  能量标准差: {np.std(atom_energies):.4f}")
    
    active_atoms = np.sum(atom_energies > 1e-6)
    print(f"  活跃原子数: {active_atoms}/{n_atoms}")
    
    atom_norms = np.linalg.norm(D, axis=0)
    print(f"\n字典原子范数统计:")
    print(f"  最小范数: {np.min(atom_norms):.6f}")
    print(f"  最大范数: {np.max(atom_norms):.6f}")
    print(f"  平均范数: {np.mean(atom_norms):.6f}")
    
    print("\n" + "="*60)
    print("在线字典学习演示 (Mairal等人方法)")
    print("="*60)
    
    n_features = 20
    n_samples_stream = 200
    n_atoms = 50
    sparsity = 5
    batch_size = 10
    
    print(f"\n信号特征维度: {n_features}")
    print(f"流式样本总数: {n_samples_stream}")
    print(f"批次大小: {batch_size}")
    print(f"字典原子数: {n_atoms}")
    print(f"稀疏度: {sparsity}")
    
    odl = OnlineDictionaryLearning(
        n_atoms=n_atoms,
        n_features=n_features,
        sparsity=sparsity,
        alpha=0.9,
        batch_size=batch_size
    )
    
    print(f"\n开始流式学习...")
    
    errors = []
    for batch_idx in range(0, n_samples_stream, batch_size):
        Y_batch = np.random.randn(n_features, batch_size)
        
        X_batch = odl.partial_fit(Y_batch)
        
        reconstruction = odl.get_dictionary() @ X_batch
        error = np.linalg.norm(Y_batch - reconstruction) / batch_size
        errors.append(error)
        
        if (batch_idx // batch_size + 1) % 5 == 0:
            print(f"  批次 {batch_idx // batch_size + 1}: 平均重构误差 = {error:.4f}")
    
    print(f"\n流式学习完成!")
    print(f"已处理样本数: {odl.total_samples}")
    
    D_online = odl.get_dictionary()
    print(f"\n在线学习字典形状: {D_online.shape}")
    
    Y_test = np.random.randn(n_features, 50)
    X_test = odl.transform(Y_test)
    test_error = np.linalg.norm(Y_test - D_online @ X_test) / 50
    print(f"\n测试集重构误差: {test_error:.4f}")
    
    atom_norms_online = np.linalg.norm(D_online, axis=0)
    print(f"\n在线学习字典原子范数统计:")
    print(f"  最小范数: {np.min(atom_norms_online):.6f}")
    print(f"  最大范数: {np.max(atom_norms_online):.6f}")
    print(f"  平均范数: {np.mean(atom_norms_online):.6f}")
    
    print(f"\n学习曲线 (误差变化):")
    print(f"  初始误差: {errors[0]:.4f}")
    print(f"  最终误差: {errors[-1]:.4f}")
    if len(errors) > 1:
        print(f"  误差下降率: {((errors[0] - errors[-1]) / errors[0] * 100):.1f}%")
