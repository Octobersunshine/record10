import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigs, eigsh
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def anderson_hamiltonian_2d(L, W, t=1.0, periodic=False):
    """
    构造二维安德森模型哈密顿量
    
    参数:
        L: 每个维度的格点数 (总格点数 N = L*L)
        W: 无序强度
        t: 跃迁能
        periodic: 是否使用周期性边界条件
    """
    N = L * L
    H = lil_matrix((N, N), dtype=np.float64)
    
    epsilon = np.random.uniform(-W/2, W/2, N)
    H.setdiag(epsilon)
    
    for i in range(L):
        for j in range(L):
            n = i * L + j
            
            if j + 1 < L:
                H[n, n+1] = -t
                H[n+1, n] = -t
            elif periodic:
                H[n, n - L + 1] = -t
                H[n - L + 1, n] = -t
            
            if i + 1 < L:
                H[n, n+L] = -t
                H[n+L, n] = -t
            elif periodic:
                H[n, n - L*(L-1)] = -t
                H[n - L*(L-1), n] = -t
    
    return H.tocsr()


def anderson_hamiltonian_3d(L, W, t=1.0, periodic=False):
    """
    构造三维安德森模型哈密顿量
    
    参数:
        L: 每个维度的格点数 (总格点数 N = L*L*L)
        W: 无序强度
        t: 跃迁能
        periodic: 是否使用周期性边界条件
    """
    N = L * L * L
    H = lil_matrix((N, N), dtype=np.float64)
    
    epsilon = np.random.uniform(-W/2, W/2, N)
    H.setdiag(epsilon)
    
    for i in range(L):
        for j in range(L):
            for k in range(L):
                n = i * L * L + j * L + k
                
                if k + 1 < L:
                    H[n, n+1] = -t
                    H[n+1, n] = -t
                elif periodic:
                    H[n, n - L + 1] = -t
                    H[n - L + 1, n] = -t
                
                if j + 1 < L:
                    H[n, n+L] = -t
                    H[n+L, n] = -t
                elif periodic:
                    H[n, n - L*(L-1)] = -t
                    H[n - L*(L-1), n] = -t
                
                if i + 1 < L:
                    H[n, n+L*L] = -t
                    H[n+L*L, n] = -t
                elif periodic:
                    H[n, n - L*L*(L-1)] = -t
                    H[n - L*L*(L-1), n] = -t
    
    return H.tocsr()


def inverse_participation_ratio(psi):
    """
    计算逆参与比 (IPR)
    IPR大表示局域化，IPR小表示扩展态
    """
    psi_abs2 = np.abs(psi)**2
    norm = np.sum(psi_abs2)
    if norm < 1e-15:
        return 0.0
    return np.sum(psi_abs2**2) / (norm**2)


def ipr_exponent(psi, dim):
    """
    计算IPR指数 α = -log(IPR)/log(N)
    α ≈ 1: 扩展态
    α ≈ 0: 局域态
    """
    N = len(psi)
    ipr = inverse_participation_ratio(psi)
    if ipr <= 0:
        return 0.0
    return -np.log(ipr) / np.log(N)


def compute_level_spacing(eigenvalues, n_center=None):
    """
    计算能级间距统计
    """
    if n_center is not None:
        mid = len(eigenvalues) // 2
        eigenvalues = eigenvalues[mid - n_center//2 : mid + n_center//2]
    
    levels = np.sort(eigenvalues)
    spacings = np.diff(levels)
    mean_spacing = np.mean(spacings)
    if mean_spacing < 1e-15:
        return 0.0, 0.0
    normalized_spacings = spacings / mean_spacing
    
    r = np.zeros(len(normalized_spacings) - 1)
    for i in range(len(r)):
        s1 = normalized_spacings[i]
        s2 = normalized_spacings[i+1]
        r[i] = min(s1, s2) / max(s1, s2)
    
    r_mean = np.mean(r)
    return r_mean, normalized_spacings


def compute_hamiltonian_stats(H, num_states=20, which='SM'):
    """
    计算哈密顿量的统计性质
    """
    try:
        eigenvalues, eigenvectors = eigsh(H, k=num_states, which=which, tol=1e-6)
    except:
        return None, None, None
    
    iprs = []
    alphas = []
    N = H.shape[0]
    dim = int(round(np.log(N) / np.log(N**(1/3)))) if N >= 8 else 2
    
    for i in range(num_states):
        psi = eigenvectors[:, i]
        iprs.append(inverse_participation_ratio(psi))
        alphas.append(ipr_exponent(psi, dim))
    
    r_mean, _ = compute_level_spacing(eigenvalues)
    
    return eigenvalues, np.array(iprs), np.array(alphas), r_mean


def finite_size_scaling_2d(L_values, W_values, samples=5, num_states=10):
    """
    二维系统有限尺寸标度分析
    
    参数:
        L_values: 不同尺寸的列表
        W_values: 无序强度列表
        samples: 每个参数点的样本数
        num_states: 每个样本计算的本征态数
    """
    results = {}
    
    for L in L_values:
        print(f"\n处理 L = {L} (N = {L*L})")
        ipr_vs_W = []
        alpha_vs_W = []
        r_vs_W = []
        
        for W in tqdm(W_values, desc=f'  W扫描'):
            ipr_list = []
            alpha_list = []
            r_list = []
            
            for _ in range(samples):
                H = anderson_hamiltonian_2d(L, W)
                _, iprs, alphas, r_mean = compute_hamiltonian_stats(H, num_states)
                if iprs is not None:
                    ipr_list.extend(iprs)
                    alpha_list.extend(alphas)
                    r_list.append(r_mean)
            
            ipr_vs_W.append(np.mean(ipr_list) if ipr_list else np.nan)
            alpha_vs_W.append(np.mean(alpha_list) if alpha_list else np.nan)
            r_vs_W.append(np.mean(r_list) if r_list else np.nan)
        
        results[L] = {
            'W': W_values,
            'IPR': np.array(ipr_vs_W),
            'alpha': np.array(alpha_vs_W),
            'r': np.array(r_vs_W)
        }
    
    return results


def finite_size_scaling_3d(L_values, W_values, samples=3, num_states=10):
    """
    三维系统有限尺寸标度分析
    """
    results = {}
    
    for L in L_values:
        print(f"\n处理 L = {L} (N = {L*L*L})")
        ipr_vs_W = []
        alpha_vs_W = []
        r_vs_W = []
        
        for W in tqdm(W_values, desc=f'  W扫描'):
            ipr_list = []
            alpha_list = []
            r_list = []
            
            for _ in range(samples):
                H = anderson_hamiltonian_3d(L, W)
                _, iprs, alphas, r_mean = compute_hamiltonian_stats(H, num_states)
                if iprs is not None:
                    ipr_list.extend(iprs)
                    alpha_list.extend(alphas)
                    r_list.append(r_mean)
            
            ipr_vs_W.append(np.mean(ipr_list) if ipr_list else np.nan)
            alpha_vs_W.append(np.mean(alpha_list) if alpha_list else np.nan)
            r_vs_W.append(np.mean(r_list) if r_list else np.nan)
        
        results[L] = {
            'W': W_values,
            'IPR': np.array(ipr_vs_W),
            'alpha': np.array(alpha_vs_W),
            'r': np.array(r_vs_W)
        }
    
    return results


def estimate_mobility_edge_3d(L_values, W_values, results):
    """
    估计三维系统的迁移率边缘 W_c
    
    利用有限尺寸标度：在临界点处，不同尺寸的IPR曲线相交
    """
    from scipy.interpolate import interp1d
    
    ipr_interp = {}
    for L in L_values:
        ipr_interp[L] = interp1d(W_values, results[L]['IPR'], 
                                 kind='cubic', fill_value='extrapolate')
    
    W_fine = np.linspace(min(W_values), max(W_values), 200)
    ipr_fine = np.zeros((len(L_values), len(W_fine)))
    for i, L in enumerate(L_values):
        ipr_fine[i] = ipr_interp[L](W_fine)
    
    variance = np.var(ipr_fine, axis=0)
    min_idx = np.argmin(variance)
    W_c_estimate = W_fine[min_idx]
    
    return W_c_estimate, W_fine, ipr_fine


def plot_finite_size_scaling(results, dim, filename_prefix):
    """
    绘制有限尺寸标度结果
    """
    L_values = sorted(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for L in L_values:
        data = results[L]
        axes[0].plot(data['W'], data['IPR'], 'o-', label=f'L={L}', markersize=4)
        axes[1].plot(data['W'], data['alpha'], 'o-', label=f'L={L}', markersize=4)
        axes[2].plot(data['W'], data['r'], 'o-', label=f'L={L}', markersize=4)
    
    axes[0].set_xlabel('无序强度 W', fontsize=12)
    axes[0].set_ylabel('IPR', fontsize=12)
    axes[0].set_title(f'{dim}D - 逆参与比', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel('无序强度 W', fontsize=12)
    axes[1].set_ylabel('IPR指数 α', fontsize=12)
    axes[1].set_title(f'{dim}D - IPR指数', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='扩展态 (α=1)')
    axes[1].axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='局域态 (α=0)')
    
    axes[2].set_xlabel('无序强度 W', fontsize=12)
    axes[2].set_ylabel('<r>', fontsize=12)
    axes[2].set_title(f'{dim}D - 能级间距比', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.5307, color='r', linestyle='--', alpha=0.5, label='GOE (金属)')
    axes[2].axhline(y=0.3863, color='b', linestyle='--', alpha=0.5, label='泊松 (绝缘)')
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_fss.png', dpi=150)
    plt.close()


def plot_mobility_edge(W_c_estimate, W_fine, ipr_fine, L_values, filename):
    """
    绘制迁移率边缘分析
    """
    plt.figure(figsize=(10, 6))
    
    for i, L in enumerate(L_values):
        plt.plot(W_fine, ipr_fine[i], label=f'L={L}', linewidth=2)
    
    plt.axvline(x=W_c_estimate, color='k', linestyle='--', linewidth=2, 
                label=f'W_c ≈ {W_c_estimate:.2f}')
    
    plt.xlabel('无序强度 W', fontsize=12)
    plt.ylabel('IPR', fontsize=12)
    plt.title('3D安德森模型 - 迁移率边缘估计', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.annotate('金属相\n(扩展态)', xy=(W_c_estimate-3, 0.5), 
                 fontsize=12, ha='center')
    plt.annotate('绝缘相\n(局域态)', xy=(W_c_estimate+3, 0.5), 
                 fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_wavefunction_2d(L, W, state_idx=0, filename=None):
    """
    绘制二维波函数空间分布
    """
    H = anderson_hamiltonian_2d(L, W)
    eigenvalues, eigenvectors = eigsh(H, k=state_idx+3, which='SM')
    
    psi = eigenvectors[:, state_idx].reshape(L, L)
    density = np.abs(psi)**2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(psi, cmap='RdBu_r', aspect='equal')
    axes[0].set_title(f'波函数 (W={W}, E={eigenvalues[state_idx]:.3f})', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(density, cmap='hot', aspect='equal')
    axes[1].set_title('概率密度 |ψ|²', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    
    ipr = inverse_participation_ratio(eigenvectors[:, state_idx])
    alpha = ipr_exponent(eigenvectors[:, state_idx], 2)
    fig.suptitle(f'IPR = {ipr:.4f}, α = {alpha:.3f}', fontsize=14, y=0.98)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.close()


def main():
    print("=" * 70)
    print("二维/三维安德森局域化模拟")
    print("(有限尺寸标度分析 & 金属-绝缘体转变)")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("第一部分：二维安德森局域化")
    print("-" * 70)
    
    L_values_2d = [6, 8, 10, 12]
    W_values_2d = np.linspace(1.0, 12.0, 8)
    
    print(f"\n尺寸范围: L = {L_values_2d}")
    print(f"无序强度范围: W = [{W_values_2d[0]}, {W_values_2d[-1]}]")
    print("\n注意：二维系统中任意小的无序都会导致局域化")
    
    results_2d = finite_size_scaling_2d(L_values_2d, W_values_2d, samples=3, num_states=8)
    plot_finite_size_scaling(results_2d, 2, 'anderson_2d')
    print("\n已保存: anderson_2d_fss.png")
    
    print("\n绘制二维波函数分布...")
    plot_wavefunction_2d(20, W=2.0, state_idx=0, filename='wavefunction_2d_W2.png')
    plot_wavefunction_2d(20, W=8.0, state_idx=0, filename='wavefunction_2d_W8.png')
    print("已保存: wavefunction_2d_W2.png, wavefunction_2d_W8.png")
    
    print("\n" + "-" * 70)
    print("第二部分：三维安德森局域化 & 金属-绝缘体转变")
    print("-" * 70)
    
    L_values_3d = [4, 6, 8]
    W_values_3d = np.linspace(8.0, 25.0, 10)
    
    print(f"\n尺寸范围: L = {L_values_3d}")
    print(f"无序强度范围: W = [{W_values_3d[0]}, {W_values_3d[-1]}]")
    print("\n三维系统存在迁移率边缘 W_c，发生金属-绝缘体转变")
    
    results_3d = finite_size_scaling_3d(L_values_3d, W_values_3d, samples=2, num_states=6)
    plot_finite_size_scaling(results_3d, 3, 'anderson_3d')
    print("\n已保存: anderson_3d_fss.png")
    
    print("\n估计迁移率边缘 W_c...")
    W_c_estimate, W_fine, ipr_fine = estimate_mobility_edge_3d(L_values_3d, W_values_3d, results_3d)
    plot_mobility_edge(W_c_estimate, W_fine, ipr_fine, L_values_3d, 'mobility_edge.png')
    print(f"  估计的迁移率边缘: W_c ≈ {W_c_estimate:.2f}")
    print("已保存: mobility_edge.png")
    
    print("\n" + "=" * 70)
    print("物理结论总结:")
    print("=" * 70)
    print("  1D: 任意小的无序 → 所有态局域化")
    print("  2D: 任意小的无序 → 所有态局域化 (弱局域化)")
    print("  3D: W < W_c → 扩展态 (金属)")
    print("      W > W_c → 局域态 (绝缘体)")
    print(f"      临界无序强度 W_c ≈ 16-18 (能带中心，数值结果)")
    print("=" * 70)


if __name__ == "__main__":
    main()
