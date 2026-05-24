import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from tqdm import tqdm


def anderson_hamiltonian(N, W, t=1.0):
    """
    构造一维安德森模型的哈密顿量
    
    参数:
        N: 格点数
        W: 无序强度
        t: 跃迁能
    """
    diag = np.random.uniform(-W/2, W/2, N)
    off_diag = -t * np.ones(N-1)
    H = diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(N, N))
    return H


def transfer_matrix_direct(N, W, E, t=1.0):
    """
    直接传输矩阵法（易数值溢出，用于对比）
    
    参数:
        N: 格点数
        W: 无序强度
        E: 能量
        t: 跃迁能
    """
    epsilon = np.random.uniform(-W/2, W/2, N)
    psi = np.zeros(N, dtype=complex)
    psi[0] = 1.0
    psi[1] = (E - epsilon[0]) / t
    
    log_psi = np.zeros(N)
    log_psi[0] = np.log(np.abs(psi[0]) + 1e-10)
    log_psi[1] = np.log(np.abs(psi[1]) + 1e-10)
    
    for n in range(2, N):
        psi[n] = ((E - epsilon[n-1]) * psi[n-1] - t * psi[n-2]) / t
        log_psi[n] = np.log(np.abs(psi[n]) + 1e-10)
    
    x = np.arange(N)
    slope, _ = np.polyfit(x, log_psi, 1)
    lyapunov = 2 * np.abs(slope)
    localization_length = 1 / lyapunov if lyapunov > 0 else np.inf
    
    return psi, localization_length, lyapunov


def transfer_matrix_gram_schmidt(N, W, E, t=1.0, ortho_interval=5):
    """
    使用Gram-Schmidt正交化的稳定传输矩阵法
    
    算法原理:
    1. 用两个线性无关的初始向量构造2维基
    2. 每步传输后，定期进行Gram-Schmidt正交化和归一化
    3. 累积归一化因子计算李雅普诺夫指数
    
    参数:
        N: 格点数
        W: 无序强度
        E: 能量
        t: 跃迁能
        ortho_interval: 正交化间隔（格点数）
    """
    epsilon = np.random.uniform(-W/2, W/2, N)
    
    v1 = np.array([1.0, 0.0], dtype=np.float64)
    v2 = np.array([0.0, 1.0], dtype=np.float64)
    
    log_norm_sum = 0.0
    ortho_count = 0
    
    for n in range(1, N):
        M = np.array([
            [(E - epsilon[n]) / t, -1.0],
            [1.0, 0.0]
        ], dtype=np.float64)
        
        v1 = M @ v1
        v2 = M @ v2
        
        if n % ortho_interval == 0 or n == N - 1:
            u1 = v1
            norm1 = np.linalg.norm(u1)
            u1_normalized = u1 / norm1 if norm1 > 1e-15 else u1
            
            u2 = v2 - np.dot(v2, u1_normalized) * u1_normalized
            norm2 = np.linalg.norm(u2)
            u2_normalized = u2 / norm2 if norm2 > 1e-15 else u2
            
            log_norm_sum += np.log(norm1 + 1e-15) + np.log(norm2 + 1e-15)
            
            v1 = u1_normalized
            v2 = u2_normalized
            ortho_count += 1
    
    lyapunov = log_norm_sum / (2 * N)
    localization_length = 1 / lyapunov if lyapunov > 0 else np.inf
    
    return None, localization_length, lyapunov


def transfer_matrix_method(N, W, E, t=1.0, method='gram_schmidt'):
    """
    传输矩阵法计算波函数和李雅普诺夫指数（统一接口）
    
    参数:
        N: 格点数
        W: 无序强度
        E: 能量
        t: 跃迁能
        method: 'gram_schmidt' (稳定) 或 'direct' (直接)
    """
    if method == 'gram_schmidt':
        return transfer_matrix_gram_schmidt(N, W, E, t)
    else:
        return transfer_matrix_direct(N, W, E, t)


def compute_eigenstates(N, W, num_states=10, t=1.0):
    """
    计算本征态和本征值
    
    参数:
        N: 格点数
        W: 无序强度
        num_states: 计算的本征态数目
        t: 跃迁能
    """
    H = anderson_hamiltonian(N, W, t)
    eigenvalues, eigenvectors = eigs(H, k=num_states, which='SM')
    return eigenvalues.real, eigenvectors.real


def inverse_participation_ratio(psi):
    """
    计算逆参与比 (IPR)
    IPR大表示局域化，IPR小表示扩展态
    """
    return np.sum(np.abs(psi)**4) / (np.sum(np.abs(psi)**2)**2)


def localization_length_vs_disorder(W_values, N=200, E=0.0, samples=20, t=1.0):
    """
    计算不同无序强度下的局域化长度
    
    参数:
        W_values: 无序强度数组
        N: 格点数
        E: 能量
        samples: 每个无序强度的样本数
        t: 跃迁能
    """
    xi_means = []
    xi_stds = []
    
    for W in tqdm(W_values, desc='计算局域化长度'):
        xi_list = []
        for _ in range(samples):
            _, xi, _ = transfer_matrix_method(N, W, E, t)
            if np.isfinite(xi):
                xi_list.append(xi)
        xi_means.append(np.mean(xi_list))
        xi_stds.append(np.std(xi_list))
    
    return np.array(xi_means), np.array(xi_stds)


def plot_wavefunctions():
    """
    绘制不同无序强度下的波函数
    """
    N = 100
    W_values = [0.0, 2.0, 5.0, 10.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for ax, W in zip(axes, W_values):
        H = anderson_hamiltonian(N, W)
        eigenvalues, eigenvectors = eigs(H, k=5, which='SM')
        
        for i in range(3):
            psi = eigenvectors[:, i].real
            psi_norm = psi / np.sqrt(np.sum(psi**2))
            ax.plot(psi_norm + i*0.5, label=f'E={eigenvalues[i].real:.2f}')
        
        ax.set_title(f'W = {W}')
        ax.set_xlabel('格点位置')
        ax.set_ylabel('波函数振幅 (偏移)')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('wavefunctions.png', dpi=150)
    plt.close()


def plot_localization_length():
    """
    绘制局域化长度随无序强度的变化
    """
    W_values = np.linspace(0.5, 15.0, 20)
    xi_means, xi_stds = localization_length_vs_disorder(W_values, N=300, samples=30)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(W_values, xi_means, yerr=xi_stds, fmt='o-', capsize=5)
    plt.xlabel('无序强度 W', fontsize=12)
    plt.ylabel('局域化长度 ξ', fontsize=12)
    plt.title('局域化长度随无序强度的变化', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('localization_length.png', dpi=150)
    plt.close()


def plot_ipr():
    """
    绘制逆参与比随无序强度的变化
    """
    N = 200
    W_values = np.linspace(0.1, 10.0, 15)
    ipr_means = []
    ipr_stds = []
    
    for W in tqdm(W_values, desc='计算IPR'):
        ipr_list = []
        for _ in range(10):
            H = anderson_hamiltonian(N, W)
            eigenvalues, eigenvectors = eigs(H, k=5, which='SM')
            for i in range(5):
                ipr = inverse_participation_ratio(eigenvectors[:, i].real)
                ipr_list.append(ipr)
        ipr_means.append(np.mean(ipr_list))
        ipr_stds.append(np.std(ipr_list))
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(W_values, ipr_means, yerr=ipr_stds, fmt='o-', capsize=5)
    plt.xlabel('无序强度 W', fontsize=12)
    plt.ylabel('逆参与比 (IPR)', fontsize=12)
    plt.title('逆参与比随无序强度的变化', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('ipr.png', dpi=150)
    plt.close()


def test_numerical_stability():
    """
    测试长链系统下两种方法的数值稳定性
    """
    print("\n数值稳定性测试 (长链系统)")
    print("-" * 50)
    
    N_values = [100, 500, 1000, 5000, 10000]
    W = 5.0
    E = 0.0
    
    print(f"{'格点数 N':<12} {'方法':<15} {'李雅普诺夫指数':<15} {'局域化长度':<15} {'状态':<10}")
    print("-" * 70)
    
    for N in N_values:
        try:
            _, xi_gs, lyap_gs = transfer_matrix_method(N, W, E, method='gram_schmidt')
            print(f"{N:<12} {'Gram-Schmidt':<15} {lyap_gs:<15.6f} {xi_gs:<15.2f} {'稳定':<10}")
        except Exception as e:
            print(f"{N:<12} {'Gram-Schmidt':<15} {'N/A':<15} {'N/A':<15} {'错误':<10}")
        
        if N <= 1000:
            try:
                _, xi_direct, lyap_direct = transfer_matrix_method(N, W, E, method='direct')
                status = "稳定" if np.isfinite(xi_direct) else "溢出"
                print(f"{'':<12} {'直接法':<15} {lyap_direct:<15.6f} {xi_direct:<15.2f} {status:<10}")
            except:
                print(f"{'':<12} {'直接法':<15} {'N/A':<15} {'N/A':<15} {'溢出':<10}")
        print()


def test_ortho_interval():
    """
    测试不同正交化间隔的影响
    """
    print("\n正交化间隔影响测试")
    print("-" * 50)
    
    N = 5000
    W = 5.0
    E = 0.0
    
    intervals = [1, 5, 10, 50, 100]
    
    print(f"{'间隔':<10} {'李雅普诺夫指数':<15} {'局域化长度':<15}")
    print("-" * 40)
    
    for interval in intervals:
        _, xi, lyap = transfer_matrix_gram_schmidt(N, W, E, ortho_interval=interval)
        print(f"{interval:<10} {lyap:<15.6f} {xi:<15.2f}")


def main():
    print("=" * 60)
    print("一维安德森局域化模拟")
    print("(使用Gram-Schmidt正交化的稳定传输矩阵法)")
    print("=" * 60)
    
    print("\n0. 数值稳定性验证...")
    test_numerical_stability()
    test_ortho_interval()
    
    print("\n1. 绘制不同无序强度下的波函数...")
    plot_wavefunctions()
    print("   已保存: wavefunctions.png")
    
    print("\n2. 计算局域化长度随无序强度的变化...")
    plot_localization_length()
    print("   已保存: localization_length.png")
    
    print("\n3. 计算逆参与比随无序强度的变化...")
    plot_ipr()
    print("   已保存: ipr.png")
    
    print("\n" + "=" * 60)
    print("模拟完成！")
    print("=" * 60)
    
    N = 100
    W = 5.0
    print(f"\n示例: N={N}, W={W}")
    print("-" * 40)
    
    H = anderson_hamiltonian(N, W)
    eigenvalues, eigenvectors = eigs(H, k=3, which='SM')
    
    for i in range(3):
        psi = eigenvectors[:, i].real
        ipr = inverse_participation_ratio(psi)
        print(f"本征态 {i+1}: E={eigenvalues[i].real:.4f}, IPR={ipr:.4f}")
    
    _, xi, lyap = transfer_matrix_method(N, W, E=0.0, method='gram_schmidt')
    print(f"\n传输矩阵法 (Gram-Schmidt):")
    print(f"  李雅普诺夫指数 = {lyap:.4f}")
    print(f"  局域化长度 ξ = {xi:.2f} 格点")


if __name__ == "__main__":
    main()
