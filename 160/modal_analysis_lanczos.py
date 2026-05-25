import numpy as np
from scipy.linalg import eigh, solve, cholesky, cho_solve, cho_factor
import warnings


class LanczosSolver:
    """
    Lanczos算法求解广义特征值问题 KΦ = ω²MΦ
    
    特点:
    - 适用于大型稀疏矩阵
    - 自动处理密集模态
    - 收敛性跟踪（残差范数<1e-9）
    - 支持重正交化避免数值不稳定
    """
    
    def __init__(self, K, M, check_positive_definite=True):
        """
        初始化Lanczos求解器
        
        参数:
            K: 刚度矩阵 (n x n)
            M: 质量矩阵 (n x n)
            check_positive_definite: 是否检查M的正定性
        """
        self.K = np.asarray(K, dtype=np.float64)
        self.M = np.asarray(M, dtype=np.float64)
        self.n = self.K.shape[0]
        
        self._validate_matrices()
        
        if check_positive_definite:
            self._check_positive_definite()
        
        self.M_chol = None
        self._factorize_M()
        
        self.eigenvalues = None
        self.eigenvectors = None
        self.convergence_history = []
        self.residual_norms = None
    
    def _validate_matrices(self):
        """验证矩阵有效性"""
        if self.K.shape[0] != self.K.shape[1]:
            raise ValueError("刚度矩阵必须是方阵")
        if self.M.shape[0] != self.M.shape[1]:
            raise ValueError("质量矩阵必须是方阵")
        if self.K.shape != self.M.shape:
            raise ValueError("刚度矩阵和质量矩阵维度必须相同")
    
    def _check_positive_definite(self):
        """检查质量矩阵正定性"""
        try:
            cholesky(self.M)
        except np.linalg.LinAlgError:
            warnings.warn("质量矩阵可能不是严格正定的，算法可能不稳定")
    
    def _factorize_M(self):
        """Cholesky分解M用于高效求解"""
        try:
            self.M_chol = cho_factor(self.M, lower=True)
        except np.linalg.LinAlgError:
            self.M_chol = None
    
    def _M_solve(self, b):
        """求解 Mx = b"""
        if self.M_chol is not None:
            return cho_solve(self.M_chol, b)
        else:
            return solve(self.M, b)
    
    def _M_norm(self, x):
        """计算M-范数 ||x||_M = sqrt(x^T M x)"""
        return np.sqrt(np.dot(x, self.M @ x))
    
    def _M_inner(self, x, y):
        """计算M-内积 x^T M y"""
        return np.dot(x, self.M @ y)
    
    def solve(self, num_eigenvalues, max_iter=None, tol=1e-9, 
              block_size=None, reortho='full', verbose=False):
        """
        执行Lanczos迭代求解特征值
        
        参数:
            num_eigenvalues: 要求解的特征值数量（最小的）
            max_iter: 最大迭代次数（默认: max(2*num_eigenvalues, 30)）
            tol: 收敛容差（残差范数<tol）
            block_size: 块Lanczos的块大小（None为单向量）
            reortho: 重正交化方式 ('full', 'selective', None)
            verbose: 是否打印迭代信息
        
        返回:
            eigenvalues: 特征值（ω²）
            eigenvectors: 特征向量矩阵
        """
        if block_size is None:
            return self._solve_single_vector(num_eigenvalues, max_iter, tol, reortho, verbose)
        else:
            return self._solve_block(num_eigenvalues, max_iter, tol, block_size, reortho, verbose)
    
    def _solve_single_vector(self, num_eigenvalues, max_iter, tol, reortho, verbose):
        """单向量Lanczos算法"""
        if max_iter is None:
            max_iter = max(2 * num_eigenvalues, 30)
        max_iter = min(max_iter, self.n)
        
        if num_eigenvalues > max_iter:
            raise ValueError(f"num_eigenvalues({num_eigenvalues})不能大于max_iter({max_iter})")
        
        V = np.zeros((self.n, max_iter + 1))
        alpha = np.zeros(max_iter)
        beta = np.zeros(max_iter)
        
        v = np.random.randn(self.n)
        v = v / self._M_norm(v)
        V[:, 0] = v
        
        w = self.K @ v
        alpha[0] = self._M_inner(v, w)
        w = w - alpha[0] * self.M @ v
        
        if reortho == 'full':
            w = w - self.M @ V[:, :1] @ (V[:, :1].T @ w)
        
        converged = False
        self.convergence_history = []
        
        for j in range(1, max_iter):
            beta[j] = self._M_norm(w)
            
            if beta[j] < 1e-14:
                if verbose:
                    print(f"迭代 {j}: beta≈0，重新启动")
                w = np.random.randn(self.n)
                w = w - V[:, :j] @ (V[:, :j].T @ self.M @ w)
                w = w / self._M_norm(w)
                V[:, j] = w
                w = self.K @ w - V[:, :j] @ (V[:, :j].T @ self.K @ w)
                continue
            
            v_new = w / beta[j]
            
            if reortho == 'full':
                v_new = v_new - V[:, :j] @ (V[:, :j].T @ self.M @ v_new)
                v_new = v_new / self._M_norm(v_new)
            
            V[:, j] = v_new
            
            w = self.K @ v_new - beta[j] * self.M @ V[:, j-1]
            alpha[j] = self._M_inner(v_new, w)
            w = w - alpha[j] * self.M @ v_new
            
            if reortho == 'full':
                w = w - self.M @ V[:, :j+1] @ (V[:, :j+1].T @ w)
            
            if j >= num_eigenvalues and j % 3 == 0:
                T = np.diag(alpha[:j+1]) + np.diag(beta[1:j+1], 1) + np.diag(beta[1:j+1], -1)
                theta, s = eigh(T)
                
                idx = np.argsort(theta)
                theta = theta[idx]
                s = s[:, idx]
                
                R = np.zeros(num_eigenvalues)
                for i in range(min(num_eigenvalues, j+1)):
                    R[i] = abs(beta[j] * s[j, i])
                
                self.convergence_history.append({
                    'iteration': j,
                    'residuals': R.copy(),
                    'eigenvalues': theta[:num_eigenvalues].copy()
                })
                
                max_residual = np.max(R[:num_eigenvalues])
                
                if verbose:
                    print(f"迭代 {j}: 最大残差 = {max_residual:.2e}")
                
                if max_residual < tol:
                    if verbose:
                        print(f"在迭代 {j} 次后收敛！")
                    converged = True
                    break
        
        if not converged:
            T = np.diag(alpha[:max_iter]) + np.diag(beta[1:max_iter], 1) + np.diag(beta[1:max_iter], -1)
            theta, s = eigh(T)
            idx = np.argsort(theta)
            theta = theta[idx]
            s = s[:, idx]
            
            R = np.zeros(num_eigenvalues)
            for i in range(num_eigenvalues):
                R[i] = abs(beta[max_iter-1] * s[max_iter-1, i])
            
            warnings.warn(f"Lanczos在{max_iter}次迭代后未完全收敛，最大残差={np.max(R):.2e}")
        
        T = np.diag(alpha[:j+1]) + np.diag(beta[1:j+1], 1) + np.diag(beta[1:j+1], -1)
        theta, s = eigh(T)
        idx = np.argsort(theta)
        theta = theta[idx]
        s = s[:, idx]
        
        self.eigenvalues = theta[:num_eigenvalues]
        self.eigenvectors = V[:, :j+1] @ s[:, :num_eigenvalues]
        
        self._compute_residual_norms()
        
        return self.eigenvalues, self.eigenvectors
    
    def _solve_block(self, num_eigenvalues, max_iter, tol, block_size, reortho, verbose):
        """块Lanczos算法（更好处理密集模态）"""
        if max_iter is None:
            max_iter = max(int(1.5 * num_eigenvalues / block_size) + 5, 20)
        
        num_blocks = max_iter
        block_size = min(block_size, num_eigenvalues)
        
        V = np.zeros((self.n, num_blocks * block_size))
        alpha = [np.zeros((block_size, block_size)) for _ in range(num_blocks)]
        beta = [np.zeros((block_size, block_size)) for _ in range(num_blocks - 1)]
        
        V_block = np.random.randn(self.n, block_size)
        for i in range(block_size):
            V_block[:, i] = V_block[:, i] - V_block[:, :i] @ (V_block[:, :i].T @ self.M @ V_block[:, i])
            V_block[:, i] = V_block[:, i] / self._M_norm(V_block[:, i])
        
        V[:, :block_size] = V_block
        
        W = self.K @ V_block
        alpha[0] = V_block.T @ self.M @ W
        
        W = W - V_block @ alpha[0]
        W = W - self.M @ V_block @ (V_block.T @ W)
        
        converged = False
        self.convergence_history = []
        
        for j in range(1, num_blocks):
            B = V[:, (j-1)*block_size:j*block_size].T @ self.M @ W
            beta[j-1] = B
            
            try:
                Q, R = np.linalg.qr(W - V[:, :j*block_size] @ (V[:, :j*block_size].T @ self.M @ W))
                V_block = Q
            except:
                V_block, _ = np.linalg.qr(W)
            
            for i in range(block_size):
                V_block[:, i] = V_block[:, i] / self._M_norm(V_block[:, i])
            
            start_idx = j * block_size
            end_idx = start_idx + block_size
            if end_idx > self.n:
                break
            
            V[:, start_idx:end_idx] = V_block
            
            W = self.K @ V_block - V[:, (j-1)*block_size:j*block_size] @ beta[j-1].T
            alpha[j] = V_block.T @ self.M @ W
            
            W = W - V_block @ alpha[j]
            W = W - V[:, :end_idx] @ (V[:, :end_idx].T @ self.M @ W)
            
            if (j + 1) * block_size >= num_eigenvalues:
                m = (j + 1) * block_size
                T = np.zeros((m, m))
                for k in range(j + 1):
                    T[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size] = alpha[k]
                for k in range(j):
                    T[k*block_size:(k+1)*block_size, (k+1)*block_size:(k+2)*block_size] = beta[k]
                    T[(k+1)*block_size:(k+2)*block_size, k*block_size:(k+1)*block_size] = beta[k].T
                
                theta, s = eigh(T)
                idx = np.argsort(theta)
                theta = theta[idx]
                s = s[:, idx]
                
                residuals = np.linalg.norm(
                    self.K @ V[:, :m] @ s[:, :num_eigenvalues] - 
                    self.M @ V[:, :m] @ s[:, :num_eigenvalues] @ np.diag(theta[:num_eigenvalues]),
                    axis=0
                )
                
                self.convergence_history.append({
                    'iteration': j,
                    'residuals': residuals.copy(),
                    'eigenvalues': theta[:num_eigenvalues].copy()
                })
                
                max_residual = np.max(residuals[:num_eigenvalues])
                
                if verbose:
                    print(f"块迭代 {j}: 最大残差 = {max_residual:.2e}")
                
                if max_residual < tol:
                    if verbose:
                        print(f"在 {j} 次块迭代后收敛！")
                    converged = True
                    m_used = m
                    s_used = s
                    theta_used = theta
                    break
        
        if not converged:
            warnings.warn(f"块Lanczos未完全收敛")
        
        if converged:
            self.eigenvalues = theta_used[:num_eigenvalues]
            self.eigenvectors = V[:, :m_used] @ s_used[:, :num_eigenvalues]
        else:
            m = j * block_size
            T = np.zeros((m, m))
            for k in range(j):
                T[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size] = alpha[k]
            for k in range(j-1):
                T[k*block_size:(k+1)*block_size, (k+1)*block_size:(k+2)*block_size] = beta[k]
                T[(k+1)*block_size:(k+2)*block_size, k*block_size:(k+1)*block_size] = beta[k].T
            
            theta, s = eigh(T)
            idx = np.argsort(theta)
            theta = theta[idx]
            s = s[:, idx]
            
            self.eigenvalues = theta[:num_eigenvalues]
            self.eigenvectors = V[:, :m] @ s[:, :num_eigenvalues]
        
        self._compute_residual_norms()
        
        return self.eigenvalues, self.eigenvectors
    
    def _compute_residual_norms(self):
        """计算每个特征对的残差范数"""
        self.residual_norms = np.zeros(len(self.eigenvalues))
        for i in range(len(self.eigenvalues)):
            residual = self.K @ self.eigenvectors[:, i] - self.eigenvalues[i] * self.M @ self.eigenvectors[:, i]
            self.residual_norms[i] = np.linalg.norm(residual)
    
    def get_frequencies(self):
        """获取固有频率（Hz）"""
        eigenvalues_pos = np.maximum(self.eigenvalues, 0.0)
        return np.sqrt(eigenvalues_pos) / (2 * np.pi)
    
    def get_natural_frequencies(self):
        """获取固有角频率（rad/s）"""
        eigenvalues_pos = np.maximum(self.eigenvalues, 0.0)
        return np.sqrt(eigenvalues_pos)
    
    def normalize_mode_shapes(self, method='mass'):
        """归一化振型"""
        if method == 'mass':
            for i in range(self.eigenvectors.shape[1]):
                phi = self.eigenvectors[:, i]
                mass_norm = phi.T @ self.M @ phi
                if mass_norm > 1e-12:
                    self.eigenvectors[:, i] = phi / np.sqrt(mass_norm)
        elif method == 'max':
            for i in range(self.eigenvectors.shape[1]):
                phi = self.eigenvectors[:, i]
                max_val = np.max(np.abs(phi))
                if max_val > 1e-12:
                    self.eigenvectors[:, i] = phi / max_val
    
    def detect_close_modes(self, freq_tol=1e-3):
        """
        检测密集模态（频率接近的模态）
        
        参数:
            freq_tol: 频率相对差阈值
        
        返回:
            密集模态分组列表
        """
        frequencies = self.get_frequencies()
        close_groups = []
        current_group = [0]
        
        for i in range(1, len(frequencies)):
            rel_diff = abs(frequencies[i] - frequencies[i-1]) / max(frequencies[i], frequencies[i-1])
            if rel_diff < freq_tol:
                current_group.append(i)
            else:
                if len(current_group) > 1:
                    close_groups.append(current_group.copy())
                current_group = [i]
        
        if len(current_group) > 1:
            close_groups.append(current_group)
        
        return close_groups
    
    def print_results(self, precision=6):
        """打印求解结果"""
        frequencies = self.get_frequencies()
        natural_freqs = self.get_natural_frequencies()
        
        print("=" * 90)
        print("Lanczos模态分析结果")
        print("=" * 90)
        print(f"{'阶数':<6} {'角频率(rad/s)':<18} {'频率(Hz)':<15} {'周期(s)':<15} {'残差范数':<15}")
        print("-" * 90)
        
        for i in range(len(frequencies)):
            freq = frequencies[i]
            omega = natural_freqs[i]
            period = 1.0 / freq if freq > 1e-10 else np.inf
            res = self.residual_norms[i] if self.residual_norms is not None else 0.0
            print(f"{i+1:<6} {omega:<18.{precision}f} {freq:<15.{precision}f} {period:<15.{precision}f} {res:<15.2e}")
        
        print("-" * 90)
        
        close_modes = self.detect_close_modes()
        if close_modes:
            print("\n检测到密集模态分组:")
            for group in close_modes:
                print(f"  模态 {[g+1 for g in group]}: 频率 = {frequencies[group]} Hz")
        else:
            print("\n未检测到密集模态")
        
        print()
    
    def verify_orthogonality(self, tol=1e-6):
        """验证振型正交性"""
        Phi = self.eigenvectors
        mass_ortho = Phi.T @ self.M @ Phi
        stiffness_ortho = Phi.T @ self.K @ Phi
        
        print("=" * 70)
        print("振型正交性验证")
        print("=" * 70)
        
        print("\n质量正交性 (Φ^T M Φ 应为对角阵，对角线为1):")
        print(np.round(mass_ortho, 8))
        
        print("\n刚度正交性 (Φ^T K Φ 应为对角阵):")
        print(np.round(stiffness_ortho, 8))
        
        off_diag = mass_ortho - np.diag(np.diag(mass_ortho))
        error = np.max(np.abs(off_diag))
        print(f"\n质量正交非对角元最大误差: {error:.2e}")
        print(f"正交性验证: {'通过' if error < tol else '未通过'}")
        
        return error < tol


def subspace_iteration(K, M, num_eigenvalues, max_iter=50, tol=1e-9, verbose=False):
    """
    子空间迭代法（同时反迭代）
    
    适合求解前几阶特征值，对密集模态鲁棒
    """
    K = np.asarray(K, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    n = K.shape[0]
    
    subspace_dim = min(2 * num_eigenvalues + 4, n)
    X = np.random.randn(n, subspace_dim)
    
    M_chol = cho_factor(M, lower=True)
    K_chol = cho_factor(K, lower=True)
    
    eigenvalues_old = np.zeros(num_eigenvalues)
    convergence_history = []
    
    for iteration in range(max_iter):
        Y = np.zeros_like(X)
        for j in range(subspace_dim):
            rhs = M @ X[:, j]
            Y[:, j] = cho_solve(K_chol, rhs)
        
        X_new = np.zeros_like(X)
        for j in range(subspace_dim):
            X_new[:, j] = Y[:, j]
            for k in range(j):
                X_new[:, j] -= X_new[:, k] * (X_new[:, k].T @ M @ Y[:, j])
            norm = np.sqrt(X_new[:, j].T @ M @ X_new[:, j])
            if norm > 1e-14:
                X_new[:, j] /= norm
        
        X = X_new
        
        K_proj = X.T @ K @ X
        M_proj = X.T @ M @ X
        
        eigvals_proj, eigvecs_proj = eigh(K_proj, M_proj)
        idx = np.argsort(eigvals_proj)
        eigvals_proj = eigvals_proj[idx]
        eigvecs_proj = eigvecs_proj[:, idx]
        
        eigenvalues = eigvals_proj[:num_eigenvalues]
        eigenvectors = X @ eigvecs_proj[:, :num_eigenvalues]
        
        residuals = np.zeros(num_eigenvalues)
        for i in range(num_eigenvalues):
            res = K @ eigenvectors[:, i] - eigenvalues[i] * M @ eigenvectors[:, i]
            residuals[i] = np.linalg.norm(res)
        
        convergence_history.append({
            'iteration': iteration,
            'eigenvalues': eigenvalues.copy(),
            'residuals': residuals.copy()
        })
        
        max_residual = np.max(residuals)
        
        if verbose and iteration % 5 == 0:
            print(f"子空间迭代 {iteration}: 最大残差 = {max_residual:.2e}")
        
        if max_residual < tol:
            if verbose:
                print(f"子空间迭代在 {iteration} 次后收敛！")
            break
        
        eigenvalues_old = eigenvalues.copy()
    
    return eigenvalues, eigenvectors, convergence_history


def example_close_modes():
    """示例1: 密集模态测试（频率接近的系统）"""
    print("\n" + "=" * 70)
    print("示例1: 密集模态测试（弹簧-质量系统）")
    print("=" * 70)
    
    n = 20
    m_base = 1.0
    k_base = 1000.0
    
    M = np.eye(n) * m_base
    
    K = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            K[i, i] = k_base * 1.0
            K[i, i+1] = -k_base * 1.0
        elif i == n - 1:
            K[i, i-1] = -k_base * 1.0
            K[i, i] = k_base * 1.0
        else:
            K[i, i-1] = -k_base * 1.0
            K[i, i] = k_base * 2.0
            K[i, i+1] = -k_base * 1.0
    
    K[5, 5] *= 1.001
    K[6, 6] *= 1.001
    K[5, 6] *= 0.999
    K[6, 5] *= 0.999
    
    print(f"\n系统自由度: {n}")
    print("设置第6-7个质量间刚度略有变化以制造密集模态")
    print()
    
    solver = LanczosSolver(K, M)
    print("开始Lanczos迭代...")
    solver.solve(num_eigenvalues=10, max_iter=60, tol=1e-9, reortho='full', verbose=True)
    solver.normalize_mode_shapes('mass')
    solver.print_results()
    solver.verify_orthogonality()
    
    print("\n" + "-" * 50)
    print("使用子空间迭代法验证:")
    print("-" * 50)
    eigvals_sub, eigvecs_sub, hist = subspace_iteration(K, M, num_eigenvalues=10, tol=1e-9, verbose=True)
    freqs_sub = np.sqrt(np.maximum(eigvals_sub, 0)) / (2 * np.pi)
    print("\n子空间迭代得到的频率:", freqs_sub[:5])
    
    return solver


def example_large_system():
    """示例2: 大型系统测试"""
    print("\n" + "=" * 70)
    print("示例2: 大型系统测试（100自由度）")
    print("=" * 70)
    
    n = 100
    M = np.eye(n)
    K = np.zeros((n, n))
    
    for i in range(n):
        K[i, i] = 2.0
        if i > 0:
            K[i, i-1] = -1.0
        if i < n - 1:
            K[i, i+1] = -1.0
    
    K *= 1000.0
    
    print(f"\n系统自由度: {n}")
    print(f"求解前8阶模态")
    print()
    
    solver = LanczosSolver(K, M)
    solver.solve(num_eigenvalues=8, max_iter=40, tol=1e-9, reortho='full', verbose=True)
    solver.normalize_mode_shapes('mass')
    solver.print_results()
    
    return solver


def compare_with_eigh():
    """示例3: 与scipy.eigh对比验证"""
    print("\n" + "=" * 70)
    print("示例3: 与scipy.eigh对比验证")
    print("=" * 70)
    
    n = 30
    M = np.eye(n)
    K = np.zeros((n, n))
    
    for i in range(n):
        K[i, i] = 2.0 + 0.1 * i
        if i > 0:
            K[i, i-1] = -1.0
        if i < n - 1:
            K[i, i+1] = -1.0
    
    K *= 1000.0
    
    print("\n使用scipy.eigh求解（参考解）:")
    print("-" * 50)
    eigvals_eigh, eigvecs_eigh = eigh(K, M, subset_by_index=[0, 9])
    freqs_eigh = np.sqrt(np.maximum(eigvals_eigh, 0)) / (2 * np.pi)
    
    print("前10阶频率:")
    for i in range(10):
        print(f"  {i+1}: {freqs_eigh[i]:.8f} Hz")
    
    print("\n使用Lanczos求解:")
    print("-" * 50)
    solver = LanczosSolver(K, M)
    solver.solve(num_eigenvalues=10, max_iter=50, tol=1e-9, reortho='full', verbose=False)
    solver.print_results(precision=8)
    
    print("\n频率误差对比:")
    print("-" * 50)
    freqs_lanczos = solver.get_frequencies()
    for i in range(10):
        rel_error = abs(freqs_lanczos[i] - freqs_eigh[i]) / freqs_eigh[i]
        print(f"  模态 {i+1}: 相对误差 = {rel_error:.2e}")
    
    max_error = np.max(np.abs(freqs_lanczos - freqs_eigh) / freqs_eigh)
    print(f"\n最大相对误差: {max_error:.2e}")


if __name__ == "__main__":
    print("=" * 70)
    print("Lanczos算法模态分析工具")
    print("求解广义特征值问题: KΦ = ω²MΦ")
    print("特点: 密集模态鲁棒性 | 收敛跟踪 | 残差<1e-9")
    print("=" * 70)
    
    example_close_modes()
    example_large_system()
    compare_with_eigh()
    
    print("\n" + "=" * 70)
    print("使用指南:")
    print("-" * 50)
    print("1. 创建求解器: solver = LanczosSolver(K, M)")
    print("2. 求解特征值: solver.solve(num_eigenvalues=10, tol=1e-9)")
    print("3. 获取结果:")
    print("   - frequencies = solver.get_frequencies()")
    print("   - mode_shapes = solver.eigenvectors")
    print("   - residuals = solver.residual_norms")
    print("4. 检测密集模态: solver.detect_close_modes()")
    print("=" * 70)
