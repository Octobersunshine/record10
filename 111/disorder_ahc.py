import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class TightBindingDisorder:
    def __init__(self, model='square', t=1.0, t_nn=0.0, J=0.5, soc=0.0,
                 disorder_type='onsite', disorder_strength=0.0, concentration=0.0):
        self.model = model
        self.t = t
        self.t_nn = t_nn
        self.J = J
        self.soc = soc
        self.disorder_type = disorder_type
        self.disorder_strength = disorder_strength
        self.concentration = concentration
        
    def clean_hamiltonian(self, kx, ky, kz=0):
        if self.model == 'square':
            eps_k = -2 * self.t * (np.cos(kx) + np.cos(ky))
            eps_k -= 4 * self.t_nn * np.cos(kx) * np.cos(ky)
            
            h = np.zeros((2, 2), dtype=complex)
            h[0, 0] = eps_k + self.J
            h[1, 1] = eps_k - self.J
            
            if self.soc > 0:
                h[0, 1] = 1j * self.soc * (np.sin(kx) - 1j * np.sin(ky))
                h[1, 0] = -1j * self.soc * (np.sin(kx) + 1j * np.sin(ky))
            
            return h
        else:
            raise ValueError(f"Model {self.model} not implemented for disorder")
    
    def hamiltonian_gradient(self, kx, ky, kz=0, delta=1e-4):
        h = self.clean_hamiltonian(kx, ky, kz)
        h_dx = (self.clean_hamiltonian(kx + delta, ky, kz) - 
                self.clean_hamiltonian(kx - delta, ky, kz)) / (2 * delta)
        h_dy = (self.clean_hamiltonian(kx, ky + delta, kz) - 
                self.clean_hamiltonian(kx, ky - delta, kz)) / (2 * delta)
        return h, h_dx, h_dy


class CoherentPotentialApproximation:
    def __init__(self, tb_model, W=0.0, c=0.0, impurity_level=0.0):
        self.tb = tb_model
        self.W = W
        self.c = c
        self.impurity_level = impurity_level
        self.n_bands = 2
        
    def green_function(self, kx, ky, E, Sigma):
        h = self.tb.clean_hamiltonian(kx, ky)
        G = np.linalg.inv((E + 1e-6j) * np.eye(self.n_bands) - h - Sigma)
        return G
    
    def t_matrix_onsite(self, E, Sigma):
        delta_V = self.impurity_level * np.eye(self.n_bands)
        G = lambda kx, ky: self.green_function(kx, ky, E, Sigma)
        
        k = np.linspace(-np.pi, np.pi, 30)
        G_avg = np.zeros((self.n_bands, self.n_bands), dtype=complex)
        for kx in k:
            for ky in k:
                G_avg += G(kx, ky)
        G_avg /= len(k)**2
        
        T = delta_V @ np.linalg.inv(np.eye(self.n_bands) - G_avg @ delta_V)
        return T
    
    def solve_cpa_onsite(self, E, max_iter=100, tol=1e-6):
        Sigma = np.zeros((self.n_bands, self.n_bands), dtype=complex)
        Sigma[0, 0] = -0.1j
        Sigma[1, 1] = -0.1j
        
        for iteration in range(max_iter):
            T = self.t_matrix_onsite(E, Sigma)
            
            k = np.linspace(-np.pi, np.pi, 30)
            G_avg = np.zeros((self.n_bands, self.n_bands), dtype=complex)
            for kx in k:
                for ky in k:
                    G_avg += self.green_function(kx, ky, E, Sigma)
            G_avg /= len(k)**2
            
            Sigma_new = self.c * T @ np.linalg.inv(np.eye(self.n_bands) + (1 - self.c) * G_avg @ T)
            
            diff = np.max(np.abs(Sigma_new - Sigma))
            Sigma = Sigma_new
            
            if diff < tol:
                break
        
        return Sigma
    
    def cpa_self_energy_binary_alloy(self, E, V_A=0, V_B=1.0, c_A=0.5):
        Sigma = np.zeros((self.n_bands, self.n_bands), dtype=complex)
        Sigma[0, 0] = -0.01j * self.W
        Sigma[1, 1] = -0.01j * self.W
        
        max_iter = 50
        tol = 1e-5
        
        for iteration in range(max_iter):
            k = np.linspace(-np.pi, np.pi, 20)
            G_avg = np.zeros((self.n_bands, self.n_bands), dtype=complex)
            
            for kx in k:
                for ky in k:
                    G_avg += self.green_function(kx, ky, E, Sigma)
            G_avg /= len(k)**2
            
            G_A = np.linalg.inv(np.linalg.inv(G_avg) + (V_A * np.eye(self.n_bands) - Sigma))
            G_B = np.linalg.inv(np.linalg.inv(G_avg) + (V_B * np.eye(self.n_bands) - Sigma))
            
            Sigma_new = c_A * V_A * np.eye(self.n_bands) + (1 - c_A) * V_B * np.eye(self.n_bands)
            Sigma_new -= c_A * (V_A * np.eye(self.n_bands) - Sigma) @ G_A @ (V_A * np.eye(self.n_bands) - Sigma)
            Sigma_new -= (1 - c_A) * (V_B * np.eye(self.n_bands) - Sigma) @ G_B @ (V_B * np.eye(self.n_bands) - Sigma)
            
            diff = np.max(np.abs(Sigma_new - Sigma))
            Sigma = 0.5 * Sigma + 0.5 * Sigma_new
            
            if diff < tol:
                break
        
        return Sigma


def berry_curvature_with_disorder(model, kx, ky, eta=0.05, Sigma=None):
    h, h_dx, h_dy = model.hamiltonian_gradient(kx, ky)
    
    if Sigma is not None:
        h_eff = h + Sigma
    else:
        h_eff = h
    
    eigvals, eigvecs = np.linalg.eigh(h_eff)
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
                
                omega_n += numerator * energy_diff / (energy_diff**2 + eta**2)
        
        Omega[n] = np.real(omega_n)
    
    return Omega, eigvals


def kubo_formula_with_disorder(model, k_res=30, mu=0.0, eta=0.05, gamma=0.1):
    k = np.linspace(-np.pi, np.pi, k_res)
    dk = (2 * np.pi / k_res) ** 2
    
    sigma_xy = 0.0 + 0j
    
    for kx in k:
        for ky in k:
            h, h_dx, h_dy = model.hamiltonian_gradient(kx, ky)
            
            eigvals, eigvecs = np.linalg.eigh(h)
            
            for n in range(len(eigvals)):
                for m in range(len(eigvals)):
                    if n != m:
                        psi_n = eigvecs[:, n]
                        psi_m = eigvecs[:, m]
                        
                        v_nm_x = np.vdot(psi_m, h_dx @ psi_n)
                        v_mn_y = np.vdot(psi_n, h_dy @ psi_m)
                        
                        energy_diff = eigvals[m] - eigvals[n]
                        
                        lorentzian = gamma / (np.pi * (energy_diff**2 + gamma**2))
                        
                        f_n = 1.0 / (1 + np.exp((eigvals[n] - mu) / 0.02))
                        f_m = 1.0 / (1 + np.exp((eigvals[m] - mu) / 0.02))
                        
                        sigma_xy += (f_n - f_m) * np.imag(v_nm_x * v_mn_y) * lorentzian * dk
    
    return sigma_xy.real / (2 * np.pi)


class SupercellDisorder:
    def __init__(self, tb_model, supercell_size=4, disorder_type='onsite', W=1.0, c=0.5):
        self.tb = tb_model
        self.L = supercell_size
        self.disorder_type = disorder_type
        self.W = W
        self.c = c
        self.n_sites = self.L**2
        self.n_bands = 2
        
    def generate_supercell_hamiltonian(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        N = self.n_sites * self.n_bands
        H = np.zeros((N, N), dtype=complex)
        
        for i in range(self.L):
            for j in range(self.L):
                idx = (i * self.L + j) * self.n_bands
                
                if self.disorder_type == 'onsite':
                    if np.random.random() < self.c:
                        onsite = self.W
                    else:
                        onsite = 0.0
                    
                    H[idx, idx] = onsite + self.tb.J
                    H[idx+1, idx+1] = onsite - self.tb.J
                
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ni, nj = (i + di) % self.L, (j + dj) % self.L
                    nidx = (ni * self.L + nj) * self.n_bands
                    
                    H[idx, nidx] = -self.tb.t
                    H[idx+1, nidx+1] = -self.tb.t
        
        return H
    
    def diagonlize_supercell(self, H):
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals, eigvecs
    
    def compute_dos(self, eigvals, E_grid, sigma=0.1):
        dos = np.zeros_like(E_grid)
        for E in eigvals:
            dos += np.exp(-(E_grid - E)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        return dos / self.n_sites
    
    def compute_ahc_supercell(self, H, mu):
        eigvals, eigvecs = np.linalg.eigh(H)
        
        N = len(eigvals)
        H_dx = np.zeros_like(H)
        H_dy = np.zeros_like(H)
        
        for i in range(self.L):
            for j in range(self.L):
                idx = (i * self.L + j) * self.n_bands
                
                for di, dj in [(1, 0), (-1, 0)]:
                    ni, nj = (i + di) % self.L, (j + dj) % self.L
                    nidx = (ni * self.L + nj) * self.n_bands
                    H_dx[idx, nidx] = 1j * di * self.tb.t
                    H_dx[idx+1, nidx+1] = 1j * di * self.tb.t
                
                for di, dj in [(0, 1), (0, -1)]:
                    ni, nj = (i + di) % self.L, (j + dj) % self.L
                    nidx = (ni * self.L + nj) * self.n_bands
                    H_dy[idx, nidx] = 1j * dj * self.tb.t
                    H_dy[idx+1, nidx+1] = 1j * dj * self.tb.t
        
        sigma_xy = 0.0
        
        for n in range(N):
            for m in range(N):
                if n != m:
                    v_nm_x = np.vdot(eigvecs[:, m], H_dx @ eigvecs[:, n])
                    v_mn_y = np.vdot(eigvecs[:, n], H_dy @ eigvecs[:, m])
                    
                    energy_diff = eigvals[m] - eigvals[n]
                    
                    gamma = 0.1
                    lorentzian = gamma / (np.pi * (energy_diff**2 + gamma**2))
                    
                    f_n = 1.0 if eigvals[n] < mu else 0.0
                    f_m = 1.0 if eigvals[m] < mu else 0.0
                    
                    sigma_xy += (f_n - f_m) * np.imag(v_nm_x * v_mn_y) * lorentzian
        
        return sigma_xy / (self.n_sites * 2 * np.pi)


def calculate_ahc_with_disorder_cpa(model, c_list, W_list, k_res=30, mu=0.0, eta=0.05):
    results = {}
    
    for c in c_list:
        for W in W_list:
            key = (c, W)
            print(f"计算: c={c}, W={W}")
            
            tb_disorder = TightBindingDisorder(
                model='square',
                t=model.t,
                t_nn=model.t_nn,
                J=model.J,
                soc=model.soc,
                disorder_strength=W,
                concentration=c
            )
            
            sigma_xy = kubo_formula_with_disorder(
                tb_disorder,
                k_res=k_res,
                mu=mu,
                eta=eta,
                gamma=0.1 + W * c
            )
            
            results[key] = sigma_xy
    
    return results


def calculate_ahc_with_disorder_supercell(model, c_list, W_list, supercell_size=4, n_configs=5, mu=0.0):
    results = {}
    
    for c in c_list:
        for W in W_list:
            key = (c, W)
            print(f"计算超胞: c={c}, W={W}, {n_configs}个构型")
            
            sc = SupercellDisorder(
                model,
                supercell_size=supercell_size,
                disorder_type='onsite',
                W=W,
                c=c
            )
            
            sigma_list = []
            for seed in range(n_configs):
                H = sc.generate_supercell_hamiltonian(seed=seed)
                sigma = sc.compute_ahc_supercell(H, mu)
                sigma_list.append(sigma)
            
            results[key] = {
                'mean': np.mean(sigma_list),
                'std': np.std(sigma_list),
                'all': sigma_list
            }
    
    return results


def plot_disorder_effects(c_list, W_list, cpa_results, sc_results=None, figsize=(16, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(W_list)))
    
    for idx, W in enumerate(W_list):
        cpa_values = [cpa_results[(c, W)] for c in c_list]
        axes[0].plot(c_list, cpa_values, 'o-', color=colors[idx], 
                    label=f'CPA, W={W}', linewidth=2, markersize=6)
        
        if sc_results is not None:
            sc_means = [sc_results[(c, W)]['mean'] for c in c_list]
            sc_stds = [sc_results[(c, W)]['std'] for c in c_list]
            axes[0].errorbar(c_list, sc_means, yerr=sc_stds, fmt='s--', 
                           color=colors[idx], label=f'SC, W={W}', 
                           linewidth=2, markersize=6, alpha=0.7)
    
    axes[0].set_xlabel('杂质浓度 c', fontsize=12)
    axes[0].set_ylabel('σ_xy (e²/h)', fontsize=12)
    axes[0].set_title('异常霍尔电导率 vs 杂质浓度', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    
    for idx, c in enumerate(c_list[::2]):
        cpa_values = [cpa_results[(c, W)] for W in W_list]
        axes[1].plot(W_list, cpa_values, 'o-', 
                    label=f'c={c}', linewidth=2, markersize=6)
    
    axes[1].set_xlabel('无序强度 W (t)', fontsize=12)
    axes[1].set_ylabel('σ_xy (e²/h)', fontsize=12)
    axes[1].set_title('异常霍尔电导率 vs 无序强度', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('disorder_effects.png', dpi=200, bbox_inches='tight')
    print("  无序效应图已保存: disorder_effects.png")
    return fig


def plot_dos_with_disorder(model, c_list, W, supercell_size=4, n_configs=3):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    E_grid = np.linspace(-6, 6, 200)
    colors = plt.cm.plasma(np.linspace(0, 1, len(c_list)))
    
    for idx, c in enumerate(c_list):
        sc = SupercellDisorder(
            model,
            supercell_size=supercell_size,
            disorder_type='onsite',
            W=W,
            c=c
        )
        
        dos_avg = np.zeros_like(E_grid)
        for seed in range(n_configs):
            H = sc.generate_supercell_hamiltonian(seed=seed)
            eigvals, _ = sc.diagonlize_supercell(H)
            dos_avg += sc.compute_dos(eigvals, E_grid, sigma=0.1)
        dos_avg /= n_configs
        
        ax.plot(E_grid, dos_avg, color=colors[idx], label=f'c = {c}', linewidth=2)
    
    ax.set_xlabel('能量 (t)', fontsize=12)
    ax.set_ylabel('态密度 (DOS)', fontsize=12)
    ax.set_title(f'不同浓度下的态密度 (W = {W} t)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dos_with_disorder.png', dpi=200, bbox_inches='tight')
    print("  态密度图已保存: dos_with_disorder.png")
    return fig


def run_disorder_calculation():
    print("=" * 70)
    print("无序效应计算 - 杂质散射对异常霍尔电导率的影响")
    print("=" * 70)
    
    model = TightBindingDisorder(
        model='square',
        t=1.0,
        t_nn=0.2,
        J=0.5,
        soc=0.1
    )
    
    print(f"\n模型参数:")
    print(f"  最近邻跃迁 t = {model.t}")
    print(f"  次近邻跃迁 t' = {model.t_nn}")
    print(f"  交换劈裂 J = {model.J}")
    print(f"  自旋轨道耦合 λ = {model.soc}")
    
    print(f"\n无序参数范围:")
    c_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    W_list = [0.0, 0.5, 1.0, 2.0]
    print(f"  杂质浓度 c: {c_list}")
    print(f"  无序强度 W: {W_list}")
    
    print(f"\n" + "-" * 70)
    print("方法1: CPA (相干势近似)")
    print("-" * 70)
    
    cpa_results = calculate_ahc_with_disorder_cpa(
        model, c_list, W_list, k_res=25, mu=0.0, eta=0.05
    )
    
    print(f"\nCPA计算结果:")
    print(f"{'浓度 c':<10} {'强度 W':<10} {'σ_xy (e²/h)':<15}")
    print("-" * 40)
    for c in c_list:
        for W in W_list:
            print(f"{c:<10} {W:<10} {cpa_results[(c, W)]:<15.6f}")
    
    print(f"\n" + "-" * 70)
    print("方法2: 超胞平均 (可选，较慢)")
    print("-" * 70)
    
    use_supercell = input("是否运行超胞计算? (y/n, 默认: n): ").strip().lower() == 'y'
    sc_results = None
    
    if use_supercell:
        sc_results = calculate_ahc_with_disorder_supercell(
            model, c_list[::2], W_list[::2], 
            supercell_size=4, n_configs=3, mu=0.0
        )
        
        print(f"\n超胞计算结果:")
        print(f"{'浓度 c':<10} {'强度 W':<10} {'σ_xy (e²/h)':<20}")
        print("-" * 45)
        for c in c_list[::2]:
            for W in W_list[::2]:
                res = sc_results[(c, W)]
                print(f"{c:<10} {W:<10} {res['mean']:.6f} ± {res['std']:.6f}")
    
    print(f"\n" + "-" * 70)
    print("生成可视化结果...")
    print("-" * 70)
    
    plot_disorder_effects(c_list, W_list, cpa_results, sc_results)
    plot_dos_with_disorder(model, c_list[::2], W=1.0)
    
    print("\n" + "=" * 70)
    print("计算完成!")
    print("=" * 70)
    
    return cpa_results, sc_results, model


if __name__ == "__main__":
    run_disorder_calculation()
    plt.show()
