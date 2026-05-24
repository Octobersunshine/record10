import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class TightBindingFerromagnet:
    def __init__(self, model='simple_cubic', t=1.0, t_nn=0.0, J=0.5, soc=0.0):
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


def berry_curvature_kubo(model, kx, ky, kz=0, eta=1e-3):
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
                
                denom = (eigvals[n] - eigvals[m]) ** 2
                
                if abs(denom) > 1e-10:
                    omega_n += 2j * (v_nm_x * np.conj(v_nm_y) - 
                                    v_nm_y * np.conj(v_nm_x)) / denom
        
        Omega[n] = np.real(omega_n)
    
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


def calculate_ahc(model, k_res=50, mu_min=-6, mu_max=6, num_mu=100, kT=0.02, dim=2):
    print(f"  k点网格: {k_res} x {k_res}" + (f" x {k_res}" if dim == 3 else ""))
    print(f"  费米能范围: [{mu_min}, {mu_max}] t")
    
    k = np.linspace(-np.pi, np.pi, k_res)
    dk = (2 * np.pi / k_res) ** dim
    
    if dim == 2:
        n_bands = model.hamiltonian(0, 0).shape[0]
        all_bc = np.zeros((k_res, k_res, n_bands))
        all_eigvals = np.zeros((k_res, k_res, n_bands))
        
        for i, kx in enumerate(k):
            for j, ky in enumerate(k):
                bc, eigvals = berry_curvature_kubo(model, kx, ky)
                all_bc[i, j] = bc
                all_eigvals[i, j] = eigvals
    else:
        n_bands = model.hamiltonian(0, 0, 0).shape[0]
        all_bc = np.zeros((k_res, k_res, k_res, n_bands))
        all_eigvals = np.zeros((k_res, k_res, k_res, n_bands))
        
        for i, kx in enumerate(k):
            for j, ky in enumerate(k):
                for l, kz in enumerate(k):
                    bc, eigvals = berry_curvature_kubo(model, kx, ky, kz)
                    all_bc[i, j, l] = bc
                    all_eigvals[i, j, l] = eigvals
    
    mu_list = np.linspace(mu_min, mu_max, num_mu)
    ahc_list = np.zeros(num_mu)
    
    for idx, mu in enumerate(mu_list):
        ahc = 0.0
        for n in range(n_bands):
            occ = fermi_distribution(all_eigvals[..., n], mu, kT)
            ahc += np.sum(all_bc[..., n] * occ) * dk
        
        ahc_list[idx] = ahc / (2 * np.pi) ** (dim - 1)
    
    return mu_list, ahc_list, all_bc, all_eigvals


def plot_ahc_results(mu_list, ahc_list, model, all_eigvals, figsize=(16, 10)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(mu_list, ahc_list, 'b-', linewidth=2.5)
    ax1.set_xlabel('费米能 E_F (t)', fontsize=12)
    ax1.set_ylabel('异常霍尔电导率 σ_xy (e²/h)', fontsize=12)
    ax1.set_title('异常霍尔电导率 vs 费米能', fontsize=14, fontweight='bold')
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
    模型参数
    ─────────────────────
    晶格模型: {model.model}
    最近邻跃迁 t = {model.t}
    次近邻跃迁 t' = {model.t_nn}
    交换劈裂 J = {model.J}
    自旋轨道耦合 λ = {model.soc}
    
    计算参数
    ─────────────────────
    k点分辨率: 见计算输出
    温度 smearing: kT = 0.02 t
    
    物理原理
    ─────────────────────
    异常霍尔电导率:
    σ_xy = (e²/h) ∫ dk/(2π)² Ω_n(k) f(E_n(k))
    
    贝里曲率:
    Ω_n(k) = i Σ_{m≠n} ⟨n|∂_x H|m⟩⟨m|∂_y H|n⟩ - c.c.
                        (E_n - E_m)²
    """
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig('ahc_comprehensive_results.png', dpi=200, bbox_inches='tight')
    print("  结果图已保存: ahc_comprehensive_results.png")
    return fig


def plot_berry_curvature_maps(all_bc, k_res, figsize=(14, 5)):
    n_bands = all_bc.shape[-1]
    
    if all_bc.ndim == 4:
        all_bc_2d = all_bc[:, :, k_res // 2, :]
    else:
        all_bc_2d = all_bc
    
    fig, axes = plt.subplots(1, min(n_bands, 2), figsize=figsize)
    if min(n_bands, 2) == 1:
        axes = [axes]
    
    for n in range(min(n_bands, 2)):
        im = axes[n].imshow(all_bc_2d[:, :, n].T, origin='lower',
                           extent=[-np.pi, np.pi, -np.pi, np.pi],
                           cmap='RdBu_r', aspect='equal')
        axes[n].set_xlabel('k_x', fontsize=12)
        axes[n].set_ylabel('k_y', fontsize=12)
        axes[n].set_title(f'贝里曲率 - 能带 {n+1}', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=axes[n])
    
    plt.tight_layout()
    plt.savefig('berry_curvature_maps.png', dpi=200, bbox_inches='tight')
    print("  贝里曲率图已保存: berry_curvature_maps.png")
    return fig


def run_calculation(model_type='square', t=1.0, t_nn=0.2, J=0.5, soc=0.1, 
                    k_res=30, num_mu=80):
    print("=" * 70)
    print("紧束缚模型 - 贝里曲率与异常霍尔电导率计算")
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
    
    print(f"\n开始计算...")
    mu_list, ahc_list, all_bc, all_eigvals = calculate_ahc(
        model,
        k_res=k_res,
        mu_min=-5,
        mu_max=5,
        num_mu=num_mu,
        kT=0.02,
        dim=2
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
    fig2 = plot_berry_curvature_maps(all_bc, k_res)
    
    print("\n" + "=" * 70)
    print("计算完成! 结果已保存为 PNG 图像文件。")
    print("=" * 70)
    
    return mu_list, ahc_list, model, all_bc, all_eigvals


if __name__ == "__main__":
    print("\n可选模型:")
    print("  1. square - 二维正方格子 (推荐用于快速测试)")
    print("  2. simple_cubic - 简单立方格子")
    print("  3. kane_mele - Kane-Mele 模型 (拓扑绝缘体)")
    
    model_choice = input("\n请选择模型 (默认: square): ").strip() or "square"
    
    try:
        t = float(input("最近邻跃迁 t (默认: 1.0): ").strip() or "1.0")
        t_nn = float(input("次近邻跃迁 t' (默认: 0.2): ").strip() or "0.2")
        J = float(input("交换劈裂 J (默认: 0.5): ").strip() or "0.5")
        soc = float(input("自旋轨道耦合 λ (默认: 0.1): ").strip() or "0.1")
        k_res = int(input("k点分辨率 (默认: 30): ").strip() or "30")
    except ValueError:
        print("输入无效，使用默认参数...")
        t, t_nn, J, soc, k_res = 1.0, 0.2, 0.5, 0.1, 30
    
    run_calculation(
        model_type=model_choice,
        t=t,
        t_nn=t_nn,
        J=J,
        soc=soc,
        k_res=k_res
    )
    
    plt.show()
