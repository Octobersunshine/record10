import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm


class TightBindingModel:
    def __init__(self, lattice_type='square', t=1.0, t2=0.0, exchange=0.5):
        self.lattice_type = lattice_type
        self.t = t
        self.t2 = t2
        self.exchange = exchange
        
    def hamiltonian(self, kx, ky):
        if self.lattice_type == 'square':
            h0 = -2 * self.t * (np.cos(kx) + np.cos(ky))
            h0 -= 4 * self.t2 * np.cos(kx) * np.cos(ky)
            
            h = np.array([
                [h0 + self.exchange, 0],
                [0, h0 - self.exchange]
            ], dtype=complex)
            
        elif self.lattice_type == 'haldane':
            t1 = self.t
            t2 = self.t2
            phi = np.pi / 2
            
            f_k = 1 + np.exp(-1j * kx) + np.exp(-1j * (kx + ky))
            g_k = t2 * np.exp(1j * phi) * (np.exp(-1j * kx) - np.exp(-1j * ky) + 
                                           np.exp(-1j * (kx + ky)))
            
            h = np.array([
                [self.exchange + 2 * t2 * np.cos(phi) * (np.cos(kx) + np.cos(ky) + np.cos(kx + ky)),
                 t1 * np.conj(f_k) + np.conj(g_k)],
                [t1 * f_k + g_k,
                 -self.exchange + 2 * t2 * np.cos(phi) * (np.cos(kx) + np.cos(ky) + np.cos(kx + ky))]
            ], dtype=complex)
            
        else:
            raise ValueError(f"Unknown lattice type: {self.lattice_type}")
            
        return h
    
    def hamiltonian_gradient(self, kx, ky, delta=1e-5):
        h = self.hamiltonian(kx, ky)
        h_dx = (self.hamiltonian(kx + delta, ky) - self.hamiltonian(kx - delta, ky)) / (2 * delta)
        h_dy = (self.hamiltonian(kx, ky + delta) - self.hamiltonian(kx, ky - delta)) / (2 * delta)
        return h, h_dx, h_dy


def compute_berry_curvature(model, kx, ky, n=0):
    h, h_dx, h_dy = model.hamiltonian_gradient(kx, ky)
    
    eigvals, eigvecs = np.linalg.eigh(h)
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    psi_n = eigvecs[:, n]
    
    berry_curv = 0j
    for m in range(len(eigvals)):
        if m != n:
            psi_m = eigvecs[:, m]
            
            matrix_element_x = np.vdot(psi_m, h_dx @ psi_n)
            matrix_element_y = np.vdot(psi_m, h_dy @ psi_n)
            
            denom = (eigvals[n] - eigvals[m]) ** 2
            
            if abs(denom) > 1e-10:
                berry_curv += 2j * (matrix_element_x * np.conj(matrix_element_y) - 
                                   matrix_element_y * np.conj(matrix_element_x)) / denom
    
    return np.real(berry_curv), eigvals[n], eigvals


def compute_berry_curvature_fast(model, kx, ky):
    h, h_dx, h_dy = model.hamiltonian_gradient(kx, ky)
    
    eigvals, eigvecs = np.linalg.eigh(h)
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    berry_curvs = np.zeros(2)
    
    for n in range(2):
        psi_n = eigvecs[:, n]
        berry_curv = 0j
        
        for m in range(2):
            if m != n:
                psi_m = eigvecs[:, m]
                
                matrix_element_x = np.vdot(psi_m, h_dx @ psi_n)
                matrix_element_y = np.vdot(psi_m, h_dy @ psi_n)
                
                denom = (eigvals[n] - eigvals[m]) ** 2
                
                if abs(denom) > 1e-10:
                    berry_curv += 2j * (matrix_element_x * np.conj(matrix_element_y) - 
                                       matrix_element_y * np.conj(matrix_element_x)) / denom
        
        berry_curvs[n] = np.real(berry_curv)
    
    return berry_curvs, eigvals


def fermi_dirac(energy, mu, kT=0.01):
    if kT < 1e-10:
        return np.where(energy < mu, 1.0, 0.0)
    return 1.0 / (1.0 + np.exp((energy - mu) / kT))


def compute_ahc_vs_fermi_energy(model, k_res=50, mu_min=-6, mu_max=6, num_mu=100):
    kx_list = np.linspace(-np.pi, np.pi, k_res)
    ky_list = np.linspace(-np.pi, np.pi, k_res)
    dk = (2 * np.pi / k_res) ** 2
    
    all_bc = np.zeros((k_res, k_res, 2))
    all_eigvals = np.zeros((k_res, k_res, 2))
    
    for i, kx in enumerate(kx_list):
        for j, ky in enumerate(ky_list):
            bc, eigvals = compute_berry_curvature_fast(model, kx, ky)
            all_bc[i, j] = bc
            all_eigvals[i, j] = eigvals
    
    mu_list = np.linspace(mu_min, mu_max, num_mu)
    ahc_list = np.zeros(num_mu)
    
    for idx, mu in enumerate(mu_list):
        ahc = 0.0
        for n in range(2):
            occ = fermi_dirac(all_eigvals[:, :, n], mu)
            ahc += np.sum(all_bc[:, :, n] * occ) * dk
        
        ahc_list[idx] = ahc / (2 * np.pi)
    
    return mu_list, ahc_list, all_bc, all_eigvals


def plot_results(mu_list, ahc_list, model):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(mu_list, ahc_list, 'b-', linewidth=2)
    axes[0].set_xlabel('费米能 (E/t)', fontsize=12)
    axes[0].set_ylabel('异常霍尔电导率 σ_xy', fontsize=12)
    axes[0].set_title('异常霍尔电导率 vs 费米能', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    kx = np.linspace(-np.pi, np.pi, 100)
    ky = 0
    bands_up = []
    bands_dn = []
    for k in kx:
        h = model.hamiltonian(k, ky)
        eigvals = np.linalg.eigvalsh(h)
        bands_up.append(eigvals[1])
        bands_dn.append(eigvals[0])
    
    axes[1].plot(kx / np.pi, bands_up, 'r-', label='自旋向上', linewidth=2)
    axes[1].plot(kx / np.pi, bands_dn, 'b-', label='自旋向下', linewidth=2)
    axes[1].set_xlabel('k_x / π', fontsize=12)
    axes[1].set_ylabel('能量 (E/t)', fontsize=12)
    axes[1].set_title('能带结构 (k_y=0)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ahc_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_berry_curvature_map(all_bc, all_eigvals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im0 = axes[0].imshow(all_bc[:, :, 0].T, origin='lower', 
                         extent=[-np.pi, np.pi, -np.pi, np.pi],
                         cmap='RdBu_r', aspect='auto')
    axes[0].set_xlabel('k_x', fontsize=12)
    axes[0].set_ylabel('k_y', fontsize=12)
    axes[0].set_title('贝里曲率 - 低能带', fontsize=14)
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(all_bc[:, :, 1].T, origin='lower',
                         extent=[-np.pi, np.pi, -np.pi, np.pi],
                         cmap='RdBu_r', aspect='auto')
    axes[1].set_xlabel('k_x', fontsize=12)
    axes[1].set_ylabel('k_y', fontsize=12)
    axes[1].set_title('贝里曲率 - 高能带', fontsize=14)
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('berry_curvature.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("紧束缚模型 - 贝里曲率与异常霍尔电导率计算")
    print("=" * 60)
    
    model = TightBindingModel(
        lattice_type='square',
        t=1.0,
        t2=0.2,
        exchange=0.5
    )
    
    print(f"晶格类型: {model.lattice_type}")
    print(f"最近邻跃迁 t = {model.t}")
    print(f"次近邻跃迁 t2 = {model.t2}")
    print(f"交换劈裂 J = {model.exchange}")
    
    print("\n计算中...")
    mu_list, ahc_list, all_bc, all_eigvals = compute_ahc_vs_fermi_energy(
        model,
        k_res=40,
        mu_min=-5,
        mu_max=5,
        num_mu=80
    )
    
    print("计算完成！")
    
    print("\n关键费米能处的异常霍尔电导率:")
    key_mus = [-3, -1, 0, 1, 3]
    for mu in key_mus:
        idx = np.argmin(np.abs(mu_list - mu))
        print(f"  E_F = {mu:+.1f} t: σ_xy = {ahc_list[idx]:.6f}")
    
    print("\n生成图像...")
    plot_results(mu_list, ahc_list, model)
    plot_berry_curvature_map(all_bc, all_eigvals)
    
    print("\n图像已保存: ahc_results.png, berry_curvature.png")


if __name__ == "__main__":
    main()
