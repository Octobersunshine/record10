import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class TightBindingFerromagnet:
    def __init__(self, model='square', t=1.0, t_nn=0.0, J=0.5, soc=0.0):
        self.model = model
        self.t = t
        self.t_nn = t_nn
        self.J = J
        self.soc = soc
        
    def hamiltonian(self, kx, ky, kz=0):
        if self.model == 'simple_cubic':
            return self._simple_cubic_hamiltonian(kx, ky, kz)
        elif self.model == 'square':
            return self._square_hamiltonian(kx, ky)
        elif self.model == 'kane_mele':
            return self._kane_mele_hamiltonian(kx, ky)
        else:
            raise ValueError(f"Unknown model: {self.model}")
    
    def _square_hamiltonian(self, kx, ky):
        eps_k = -2 * self.t * (np.cos(kx) + np.cos(ky))
        eps_k -= 4 * self.t_nn * np.cos(kx) * np.cos(ky)
        
        h = np.zeros((2, 2), dtype=complex)
        h[0, 0] = eps_k + self.J
        h[1, 1] = eps_k - self.J
        
        if self.soc > 0:
            h[0, 1] = 1j * self.soc * (np.sin(kx) - 1j * np.sin(ky))
            h[1, 0] = -1j * self.soc * (np.sin(kx) + 1j * np.sin(ky))
        
        return h
    
    def _simple_cubic_hamiltonian(self, kx, ky, kz):
        eps_k = -2 * self.t * (np.cos(kx) + np.cos(ky) + np.cos(kz))
        
        h = np.zeros((2, 2), dtype=complex)
        h[0, 0] = eps_k + self.J
        h[1, 1] = eps_k - self.J
        
        if self.soc > 0:
            h[0, 1] = self.soc * (np.sin(kx) + 1j * np.sin(ky))
            h[1, 0] = self.soc * (np.sin(kx) - 1j * np.sin(ky))
        
        return h
    
    def _kane_mele_hamiltonian(self, kx, ky):
        t1 = self.t
        t2 = self.t_nn
        lambda_soc = self.soc
        
        f_k = 1 + np.exp(-1j * kx) + np.exp(-1j * (kx + ky))
        g_k = 1j * t2 * (np.sin(kx) - np.sin(ky) + np.sin(kx + ky))
        
        h = np.zeros((4, 4), dtype=complex)
        
        h[:2, :2] = np.array([[self.J, g_k], [-g_k, self.J]])
        h[2:, 2:] = np.array([[-self.J, -np.conj(g_k)], [np.conj(g_k), -self.J]])
        h[:2, 2:] = t1 * np.array([[np.conj(f_k), 0], [0, np.conj(f_k)]])
        h[2:, :2] = t1 * np.array([[f_k, 0], [0, f_k]])
        
        return h
    
    def hamiltonian_gradient(self, kx, ky, kz=0, delta=1e-4):
        h = self.hamiltonian(kx, ky, kz)
        h_dx = (self.hamiltonian(kx + delta, ky, kz) - 
                self.hamiltonian(kx - delta, ky, kz)) / (2 * delta)
        h_dy = (self.hamiltonian(kx, ky + delta, kz) - 
                self.hamiltonian(kx, ky - delta, kz)) / (2 * delta)
        return h, h_dx, h_dy


def gaussian_smoothing(x, eta):
    return np.exp(-x**2 / (2 * eta**2)) / (eta * np.sqrt(2 * np.pi))


def lorentzian_smoothing(x, eta):
    return eta / (np.pi * (x**2 + eta**2))


def berry_curvature_regularized(model, kx, ky, kz=0, eta=0.05, smoothing_type='gaussian'):
    h, h_dx, h_dy = model.hamiltonian_gradient(kx, ky, kz)
    
    eigvals, eigvecs = np.linalg.eigh(h)
    n_bands = len(eigvals)
    
    Omega = np.zeros(n_bands)
    
    for n in range(n_bands):
        psi_n = eigvecs[:, n]
        omega_n = 0j
        
        for m in range(n_bands):
            if m != n:
                psi_m = eigvecs[:, m]
                
                v_nm_x = np.vdot(psi_m, h_dx @ psi_n)
                v_nm_y = np.vdot(psi_m, h_dy @ psi_n)
                
                energy_diff = eigvals[n] - eigvals[m]
                
                numerator = 2j * (v_nm_x * np.conj(v_nm_y) - v_nm_y * np.conj(v_nm_x))
                
                if smoothing_type == 'gaussian':
                    weight = energy_diff**2 * gaussian_smoothing(energy_diff, eta)
                    if abs(energy_diff) > 1e-10:
                        omega_n += numerator * gaussian_smoothing(energy_diff, eta)
                elif smoothing_type == 'lorentzian':
                    omega_n += numerator * energy_diff / (energy_diff**2 + eta**2)
                else:
                    if abs(energy_diff) > eta:
                        omega_n += numerator / energy_diff**2
        
        Omega[n] = np.real(omega_n)
    
    return Omega, eigvals


def berry_curvature_kubo_formula(model, kx, ky, kz=0, eta=0.05, method='semiclassical'):
    h, h_dx, h_dy = model.hamiltonian_gradient(kx, ky, kz)
    
    eigvals, eigvecs = np.linalg.eigh(h)
    n_bands = len(eigvals)
    
    Omega = np.zeros(n_bands)
    
    if method == 'semiclassical':
        for n in range(n_bands):
            psi_n = eigvecs[:, n]
            omega_n = 0j
            
            for m in range(n_bands):
                if m != n:
                    psi_m = eigvecs[:, m]
                    
                    v_nm_x = np.vdot(psi_m, h_dx @ psi_n)
                    v_nm_y = np.vdot(psi_m, h_dy @ psi_n)
                    
                    energy_diff = eigvals[n] - eigvals[m]
                    
                    numerator = 2j * (v_nm_x * np.conj(v_nm_y) - v_nm_y * np.conj(v_nm_x))
                    
                    omega_n += numerator * energy_diff / (energy_diff**2 + eta**2)
            
            Omega[n] = np.real(omega_n)
    
    elif method == 'kubo':
        for n in range(n_bands):
            for m in range(n_bands):
                if m != n:
                    psi_n = eigvecs[:, n]
                    psi_m = eigvecs[:, m]
                    
                    v_nm_x = np.vdot(psi_m, h_dx @ psi_n)
                    v_mn_y = np.vdot(psi_n, h_dy @ psi_m)
                    
                    energy_diff = eigvals[m] - eigvals[n]
                    
                    lorentzian = eta / (np.pi * (energy_diff**2 + eta**2))
                    
                    Omega[n] += 2 * np.imag(v_nm_x * v_mn_y) * lorentzian
    
    return Omega, eigvals


def fermi_distribution(E, mu, kT):
    if kT < 1e-10:
        return np.where(E <= mu, 1.0, 0.0)
    return 1.0 / (1.0 + np.exp((E - mu) / kT))


def compute_dos(eigenvalues, E_grid, sigma=0.05):
    dos = np.zeros_like(E_grid)
    for E_flat in eigenvalues.flatten():
        dos += np.exp(-(E_grid - E_flat)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return dos / eigenvalues.size


def detect_degeneracy_points(model, kx, ky, threshold=1e-3):
    h = model.hamiltonian(kx, ky)
    eigvals = np.linalg.eigvalsh(h)
    
    min_gap = np.min(np.abs(np.diff(eigvals)))
    return min_gap < threshold, min_gap


def adaptive_k_grid(model, base_res=20, refine_level=2, gap_threshold=0.1, eta=0.05):
    print(f"  初始化自适应k网格...")
    print(f"  基础分辨率: {base_res}, 细化级别: {refine_level}")
    
    base_k = np.linspace(-np.pi, np.pi, base_res)
    base_dk = (2 * np.pi) / (base_res - 1)
    
    k_points = []
    weights = []
    
    for i in range(base_res - 1):
        for j in range(base_res - 1):
            kx_center = (base_k[i] + base_k[i+1]) / 2
            ky_center = (base_k[j] + base_k[j+1]) / 2
            
            is_degenerate, min_gap = detect_degeneracy_points(model, kx_center, ky_center, gap_threshold)
            
            if is_degenerate:
                current_level = 0
                stack = [(base_k[i], base_k[i+1], base_k[j], base_k[j+1], current_level)]
                
                while stack:
                    kx0, kx1, ky0, ky1, level = stack.pop()
                    
                    if level >= refine_level:
                        kx_mid = (kx0 + kx1) / 2
                        ky_mid = (ky0 + ky1) / 2
                        area = (kx1 - kx0) * (ky1 - ky0)
                        k_points.append((kx_mid, ky_mid))
                        weights.append(area)
                    else:
                        kx_mid = (kx0 + kx1) / 2
                        ky_mid = (ky0 + ky1) / 2
                        
                        _, gap = detect_degeneracy_points(model, kx_mid, ky_mid, gap_threshold)
                        
                        if gap < gap_threshold:
                            stack.extend([
                                (kx0, kx_mid, ky0, ky_mid, level + 1),
                                (kx_mid, kx1, ky0, ky_mid, level + 1),
                                (kx0, kx_mid, ky_mid, ky1, level + 1),
                                (kx_mid, kx1, ky_mid, ky1, level + 1),
                            ])
                        else:
                            area = (kx1 - kx0) * (ky1 - ky0)
                            k_points.append((kx_mid, ky_mid))
                            weights.append(area)
            else:
                area = base_dk * base_dk
                k_points.append((kx_center, ky_center))
                weights.append(area)
    
    print(f"  自适应网格完成: {len(k_points)} 个k点")
    return np.array(k_points), np.array(weights)


def calculate_ahc_adaptive(model, k_points, weights, mu_min=-6, mu_max=6, num_mu=100, 
                           kT=0.02, eta=0.05, method='semiclassical'):
    n_bands = model.hamiltonian(0, 0).shape[0]
    n_k = len(k_points)
    
    all_bc = np.zeros((n_k, n_bands))
    all_eigvals = np.zeros((n_k, n_bands))
    
    for idx, (kx, ky) in enumerate(k_points):
        bc, eigvals = berry_curvature_kubo_formula(model, kx, ky, eta=eta, method=method)
        all_bc[idx] = bc
        all_eigvals[idx] = eigvals
    
    mu_list = np.linspace(mu_min, mu_max, num_mu)
    ahc_list = np.zeros(num_mu)
    
    for idx, mu in enumerate(mu_list):
        ahc = 0.0
        for n in range(n_bands):
            occ = fermi_distribution(all_eigvals[:, n], mu, kT)
            ahc += np.sum(all_bc[:, n] * occ * weights)
        
        ahc_list[idx] = ahc / (2 * np.pi)
    
    return mu_list, ahc_list, all_bc, all_eigvals


def calculate_ahc_uniform(model, k_res=50, mu_min=-6, mu_max=6, num_mu=100, 
                          kT=0.02, eta=0.05, method='semiclassical', dim=2):
    print(f"  均匀k点网格: {k_res} x {k_res}")
    
    k = np.linspace(-np.pi, np.pi, k_res)
    dk = (2 * np.pi / k_res) ** dim
    
    n_bands = model.hamiltonian(0, 0).shape[0]
    all_bc = np.zeros((k_res, k_res, n_bands))
    all_eigvals = np.zeros((k_res, k_res, n_bands))
    
    for i, kx in enumerate(k):
        for j, ky in enumerate(k):
            bc, eigvals = berry_curvature_kubo_formula(model, kx, ky, eta=eta, method=method)
            all_bc[i, j] = bc
            all_eigvals[i, j] = eigvals
    
    mu_list = np.linspace(mu_min, mu_max, num_mu)
    ahc_list = np.zeros(num_mu)
    
    for idx, mu in enumerate(mu_list):
        ahc = 0.0
        for n in range(n_bands):
            occ = fermi_distribution(all_eigvals[:, :, n], mu, kT)
            ahc += np.sum(all_bc[:, :, n] * occ) * dk
        
        ahc_list[idx] = ahc / (2 * np.pi) ** (dim - 1)
    
    return mu_list, ahc_list, all_bc, all_eigvals


def compare_regularization_methods(model, k_res=30):
    print("\n" + "=" * 60)
    print("对比不同正则化方法")
    print("=" * 60)
    
    eta_values = [0.01, 0.05, 0.1, 0.2]
    methods = ['semiclassical', 'kubo']
    
    results = {}
    
    for method in methods:
        for eta in eta_values:
            print(f"\n计算: {method}, η = {eta}")
            mu_list, ahc_list, _, _ = calculate_ahc_uniform(
                model, k_res=k_res, mu_min=-5, mu_max=5, num_mu=80,
                eta=eta, method=method
            )
            results[(method, eta)] = (mu_list, ahc_list)
    
    return results


def plot_comparison(results, model):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    
    for idx, method in enumerate(['semiclassical', 'kubo']):
        ax = axes[idx]
        eta_values = [0.01, 0.05, 0.1, 0.2]
        
        for i, eta in enumerate(eta_values):
            mu_list, ahc_list = results[(method, eta)]
            ax.plot(mu_list, ahc_list, color=colors[i], 
                   label=f'η = {eta}', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('费米能 E_F (t)', fontsize=12)
        ax.set_ylabel('σ_xy (e²/h)', fontsize=12)
        ax.set_title(f'{method} 方法', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=200, bbox_inches='tight')
    print("\n  对比图已保存: regularization_comparison.png")


def plot_ahc_results(mu_list, ahc_list, model, all_eigvals, figsize=(16, 10)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(mu_list, ahc_list, 'b-', linewidth=2.5)
    ax1.set_xlabel('费米能 E_F (t)', fontsize=12)
    ax1.set_ylabel('异常霍尔电导率 σ_xy (e²/h)', fontsize=12)
    ax1.set_title('异常霍尔电导率 vs 费米能 (正则化计算)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='k', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax2 = fig.add_subplot(gs[0, 1])
    k_path = np.linspace(-np.pi, np.pi, 200)
    n_bands = model.hamiltonian(0, 0).shape[0]
    bands = np.zeros((len(k_path), n_bands))
    
    for i, kx in enumerate(k_path):
        h = model.hamiltonian(kx, 0)
        eigvals = np.linalg.eigvalsh(h)
        bands[i] = eigvals
    
    colors = ['r', 'b', 'g', 'orange']
    labels = ['自旋向上', '自旋向下', '能带3', '能带4']
    for n in range(min(n_bands, 2)):
        ax2.plot(k_path / np.pi, bands[:, n], color=colors[n], 
                label=labels[n], linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('k_x / π', fontsize=12)
    ax2.set_ylabel('能量 (t)', fontsize=12)
    ax2.set_title('能带结构 (k_y = 0)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    ax3 = fig.add_subplot(gs[1, 0])
    E_grid = np.linspace(-6, 6, 200)
    dos = compute_dos(all_eigvals, E_grid, sigma=0.1)
    ax3.plot(E_grid, dos, 'g-', linewidth=2)
    ax3.fill_between(E_grid, dos, alpha=0.3, color='g')
    ax3.set_xlabel('能量 (t)', fontsize=12)
    ax3.set_ylabel('态密度 (DOS)', fontsize=12)
    ax3.set_title('电子态密度', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    info_text = f"""
    正则化方法说明
    ─────────────────────
    问题: 能带交叉点处贝里曲率发散
    原因: Ω ~ 1/(E_n - E_m)²
    
    解决方案1: Lorentzian展宽
    Ω_n(k) = Σ (E_n-E_m) / [(E_n-E_m)² + η²] × 矩阵元
    
    解决方案2: Kubo公式
    σ_xy = - (e²/h) ∫ dE f'(E) ∫ dk/(2π)² 
           × Σ_n Im[⟨n|v_x|m⟩⟨m|v_y|n⟩] δ(E-E_n)
    
    自适应网格
    ─────────────────────
    - 检测能隙较小的区域 (gap < threshold)
    - 在简并点附近自动细化k点
    - 提高精度同时控制计算量
    
    模型参数
    ─────────────────────
    晶格模型: {model.model}
    最近邻跃迁 t = {model.t}
    次近邻跃迁 t' = {model.t_nn}
    交换劈裂 J = {model.J}
    自旋轨道耦合 λ = {model.soc}
    """
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig('ahc_regularized_results.png', dpi=200, bbox_inches='tight')
    print("  结果图已保存: ahc_regularized_results.png")
    return fig


def plot_adaptive_grid(k_points, weights, model, figsize=(16, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    area_weights = weights / np.max(weights)
    
    scatter = axes[0].scatter(k_points[:, 0], k_points[:, 1], 
                             c=area_weights, s=10, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel('k_x', fontsize=12)
    axes[0].set_ylabel('k_y', fontsize=12)
    axes[0].set_title('自适应k点网格 (颜色表示局部分辨率)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0], label='相对面积')
    axes[0].set_xlim(-np.pi, np.pi)
    axes[0].set_ylim(-np.pi, np.pi)
    axes[0].set_aspect('equal')
    
    gaps = np.zeros(len(k_points))
    for i, (kx, ky) in enumerate(k_points):
        _, gaps[i] = detect_degeneracy_points(model, kx, ky)
    
    scatter2 = axes[1].scatter(k_points[:, 0], k_points[:, 1], 
                              c=gaps, s=10, cmap='jet', alpha=0.7)
    axes[1].set_xlabel('k_x', fontsize=12)
    axes[1].set_ylabel('k_y', fontsize=12)
    axes[1].set_title('k空间最小能隙分布', fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='最小能隙 (t)')
    axes[1].set_xlim(-np.pi, np.pi)
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('adaptive_k_grid.png', dpi=200, bbox_inches='tight')
    print("  自适应网格图已保存: adaptive_k_grid.png")
    return fig


def run_calculation(model_type='square', t=1.0, t_nn=0.2, J=0.5, soc=0.1, 
                    k_res=30, use_adaptive=True, eta=0.05):
    print("=" * 70)
    print("紧束缚模型 - 正则化贝里曲率与异常霍尔电导率计算")
    print("=" * 70)
    
    model = TightBindingFerromagnet(
        model=model_type,
        t=t,
        t_nn=t_nn,
        J=J,
        soc=soc
    )
    
    print(f"\n模型参数:")
    print(f"  模型类型: {model_type}")
    print(f"  最近邻跃迁 t = {t}")
    print(f"  次近邻跃迁 t' = {t_nn}")
    print(f"  交换劈裂 J = {J}")
    print(f"  自旋轨道耦合 λ = {soc}")
    print(f"  正则化参数 η = {eta}")
    
    print(f"\n计算方法: {'自适应网格' if use_adaptive else '均匀网格'}")
    
    if use_adaptive:
        k_points, weights = adaptive_k_grid(
            model, base_res=k_res//2, refine_level=2, 
            gap_threshold=0.15, eta=eta
        )
        
        print(f"\n开始计算 (自适应积分)...")
        mu_list, ahc_list, all_bc, all_eigvals = calculate_ahc_adaptive(
            model, k_points, weights,
            mu_min=-5, mu_max=5, num_mu=80,
            kT=0.02, eta=eta, method='semiclassical'
        )
    else:
        print(f"\n开始计算 (均匀网格)...")
        mu_list, ahc_list, all_bc, all_eigvals = calculate_ahc_uniform(
            model, k_res=k_res, mu_min=-5, mu_max=5, num_mu=80,
            kT=0.02, eta=eta, method='semiclassical', dim=2
        )
    
    print("\n计算完成!")
    
    print(f"\n特征能量点的异常霍尔电导率:")
    key_mus = [-3, -2, -1, 0, 1, 2, 3]
    for mu in key_mus:
        idx = np.argmin(np.abs(mu_list - mu))
        print(f"  E_F = {mu:+.1f} t: σ_xy = {ahc_list[idx]:.6f} e²/h")
    
    max_idx = np.argmax(np.abs(ahc_list))
    print(f"\n最大电导率: |σ_xy|_max = {np.abs(ahc_list[max_idx]):.6f} e²/h")
    print(f"  对应费米能: E_F = {mu_list[max_idx]:.3f} t")
    
    print(f"\n生成可视化结果...")
    fig1 = plot_ahc_results(mu_list, ahc_list, model, all_eigvals)
    
    if use_adaptive:
        fig2 = plot_adaptive_grid(k_points, weights, model)
    
    print("\n" + "=" * 70)
    print("计算完成! 结果已保存为 PNG 图像文件。")
    print("=" * 70)
    
    return mu_list, ahc_list, model, all_bc, all_eigvals


if __name__ == "__main__":
    print("\n可选模型:")
    print("  1. square - 二维正方格子 (推荐)")
    print("  2. simple_cubic - 简单立方格子")
    print("  3. kane_mele - Kane-Mele 模型")
    
    model_choice = input("\n请选择模型 (默认: square): ").strip() or "square"
    
    try:
        t = float(input("最近邻跃迁 t (默认: 1.0): ").strip() or "1.0")
        t_nn = float(input("次近邻跃迁 t' (默认: 0.2): ").strip() or "0.2")
        J = float(input("交换劈裂 J (默认: 0.5): ").strip() or "0.5")
        soc = float(input("自旋轨道耦合 λ (默认: 0.1): ").strip() or "0.1")
        eta = float(input("正则化参数 η (默认: 0.05): ").strip() or "0.05")
        k_res = int(input("基础k点分辨率 (默认: 30): ").strip() or "30")
        use_adaptive = input("使用自适应网格? (y/n, 默认: y): ").strip().lower() != 'n'
    except ValueError:
        print("输入无效，使用默认参数...")
        t, t_nn, J, soc, eta, k_res, use_adaptive = 1.0, 0.2, 0.5, 0.1, 0.05, 30, True
    
    run_calculation(
        model_type=model_choice,
        t=t,
        t_nn=t_nn,
        J=J,
        soc=soc,
        k_res=k_res,
        use_adaptive=use_adaptive,
        eta=eta
    )
    
    plt.show()
