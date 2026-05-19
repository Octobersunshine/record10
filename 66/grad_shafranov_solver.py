import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


class GradShafranovSolver:
    def __init__(self, Rmin=0.5, Rmax=1.5, Zmin=-0.7, Zmax=0.7, nr=65, nz=65):
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.nr = nr
        self.nz = nz
        
        self.R = np.linspace(Rmin, Rmax, nr)
        self.Z = np.linspace(Zmin, Zmax, nz)
        self.dR = self.R[1] - self.R[0]
        self.dZ = self.Z[1] - self.Z[0]
        
        self.R_grid, self.Z_grid = np.meshgrid(self.R, self.Z, indexing='ij')
        
        self.psi = np.zeros((nr, nz))
        self.p = np.zeros((nr, nz))
        self.F = np.zeros((nr, nz))
        
        self.R0 = (Rmin + Rmax) / 2
        self.a = (Rmax - Rmin) / 2
        
        self.mu0 = 4 * np.pi * 1e-7
        
    def initial_guess(self):
        r_squared = ((self.R_grid - self.R0)**2 + self.Z_grid**2) / self.a**2
        self.psi = np.maximum(0, 1 - r_squared)
        
    def build_laplacian_star_matrix(self):
        n = self.nr * self.nz
        A = lil_matrix((n, n))
        
        dR2 = self.dR**2
        dZ2 = self.dZ**2
        
        for i in range(self.nr):
            for j in range(self.nz):
                idx = i * self.nz + j
                
                if i == 0 or i == self.nr-1 or j == 0 or j == self.nz-1:
                    A[idx, idx] = 1.0
                    continue
                
                R = self.R[i]
                
                coeff_R = 1.0 / dR2 - 1.0 / (2 * R * self.dR)
                coeff_L = 1.0 / dR2 + 1.0 / (2 * R * self.dR)
                coeff_U = 1.0 / dZ2
                coeff_D = 1.0 / dZ2
                coeff_C = -2.0 / dR2 - 2.0 / dZ2
                
                A[idx, idx] = coeff_C
                A[idx, (i+1)*self.nz + j] = coeff_R
                A[idx, (i-1)*self.nz + j] = coeff_L
                A[idx, i*self.nz + (j+1)] = coeff_U
                A[idx, i*self.nz + (j-1)] = coeff_D
                
        return A.tocsr()
        
    def solve(self, p0=1e5, alpha=2.0, F0=1.0, beta=0.1, max_iter=100, tol=1e-6, relaxation=0.5):
        self.initial_guess()
        A = self.build_laplacian_star_matrix()
        
        for iteration in range(max_iter):
            psi_old = self.psi.copy()
            
            psi_max = np.max(self.psi)
            if psi_max < 1e-10:
                psi_norm = np.zeros_like(self.psi)
            else:
                psi_norm = 1 - self.psi / psi_max
                psi_norm = np.maximum(psi_norm, 0)
            
            dp_dpsi = -p0 * alpha * (np.maximum(psi_norm, 1e-10)**(alpha-1)) / psi_max if psi_max > 0 else 0
            
            F = F0 * (1 + beta * psi_norm)
            dF_dpsi = -F0 * beta / psi_max if psi_max > 0 else 0
            
            rhs = -self.mu0 * self.R_grid**2 * dp_dpsi - F * dF_dpsi
            rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
            
            b = rhs.flatten()
            
            psi_new = spsolve(A, b)
            psi_new = psi_new.reshape((self.nr, self.nz))
            
            self.psi = relaxation * psi_new + (1 - relaxation) * psi_old
            
            self.psi[0, :] = 0
            self.psi[-1, :] = 0
            self.psi[:, 0] = 0
            self.psi[:, -1] = 0
            
            error = np.max(np.abs(self.psi - psi_old))
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: max error = {error:.2e}, ψ_max = {np.max(self.psi):.4e}")
                
            if error < tol and iteration > 5:
                print(f"\nConverged after {iteration} iterations!")
                print(f"Final ψ_max = {np.max(self.psi):.4e}")
                break
                
        return self.psi
        
    def plot_psi(self, levels=20, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        contour = plt.contour(self.R_grid, self.Z_grid, self.psi, levels=levels, cmap='viridis', linewidths=1.5)
        plt.colorbar(contour, label='Poloidal Flux ψ')
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.title('Grad-Shafranov Equation: Magnetic Surfaces')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def get_magnetic_field(self, clean_divergence=True):
        B_R = -np.gradient(self.psi, self.Z, axis=1) / self.R_grid
        B_Z = np.gradient(self.psi, self.R, axis=0) / self.R_grid
        
        if clean_divergence:
            B_R, B_Z = self.divergence_cleaning_fast(B_R, B_Z)
        
        return B_R, B_Z
    
    def get_magnetic_field_centered(self, clean_divergence=True):
        B_R = np.zeros_like(self.psi)
        B_Z = np.zeros_like(self.psi)
        
        for i in range(1, self.nr-1):
            for j in range(1, self.nz-1):
                B_R[i, j] = -(self.psi[i, j+1] - self.psi[i, j-1]) / (2 * self.dZ) / self.R_grid[i, j]
                B_Z[i, j] = (self.psi[i+1, j] - self.psi[i-1, j]) / (2 * self.dR) / self.R_grid[i, j]
        
        if clean_divergence:
            B_R, B_Z = self.divergence_cleaning_fast(B_R, B_Z)
        
        return B_R, B_Z
    
    def compute_divergence(self, B_R, B_Z):
        div_B = np.zeros_like(B_R)
        
        for i in range(1, self.nr-1):
            for j in range(1, self.nz-1):
                R = self.R_grid[i, j]
                dR_BR_dR = (R * B_R[i+1, j] - R * B_R[i-1, j]) / (2 * self.dR)
                dBZ_dZ = (B_Z[i, j+1] - B_Z[i, j-1]) / (2 * self.dZ)
                div_B[i, j] = (dR_BR_dR / R) + dBZ_dZ
        
        return div_B
    
    def compute_divergence_fast(self, B_R, B_Z):
        R = self.R_grid
        
        dR_BR = R * B_R
        dR_BR_dR = np.gradient(dR_BR, self.R, axis=0)
        dBZ_dZ = np.gradient(B_Z, self.Z, axis=1)
        
        div_B = (dR_BR_dR / R) + dBZ_dZ
        
        div_B[0, :] = 0
        div_B[-1, :] = 0
        div_B[:, 0] = 0
        div_B[:, -1] = 0
        
        return div_B
    
    def solve_poisson(self, rhs):
        phi = np.zeros_like(rhs)
        
        for _ in range(2000):
            phi_new = phi.copy()
            
            for i in range(1, self.nr-1):
                for j in range(1, self.nz-1):
                    R = self.R_grid[i, j]
                    
                    d2phi_dR2 = (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]) / self.dR**2
                    d2phi_dZ2 = (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) / self.dZ**2
                    dphi_dR = (phi[i+1, j] - phi[i-1, j]) / (2 * self.dR)
                    
                    laplacian = d2phi_dR2 + (1/R) * dphi_dR + d2phi_dZ2
                    residual = laplacian - rhs[i, j]
                    
                    coeff = 2/self.dR**2 + 2/self.dZ**2
                    phi_new[i, j] = phi[i, j] - 0.8 * residual / coeff
            
            max_change = np.max(np.abs(phi_new - phi))
            phi = phi_new
            
            if max_change < 1e-12:
                break
        
        return phi
    
    def divergence_cleaning_fast(self, B_R, B_Z, max_iter=50):
        for _ in range(max_iter):
            div_B = self.compute_divergence_fast(B_R, B_Z)
            
            if np.max(np.abs(div_B)) < 1e-12:
                break
            
            phi = self.solve_poisson(div_B)
            
            grad_phi_R = np.gradient(phi, self.R, axis=0)
            grad_phi_Z = np.gradient(phi, self.Z, axis=1)
            
            B_R -= 0.5 * grad_phi_R
            B_Z -= 0.5 * grad_phi_Z
        
        return B_R, B_Z
    
    def check_divergence_error(self, B_R, B_Z, method_name=""):
        div_B = self.compute_divergence_fast(B_R, B_Z)
        max_div = np.max(np.abs(div_B[1:-1, 1:-1]))
        mean_div = np.mean(np.abs(div_B[1:-1, 1:-1]))
        
        if method_name:
            print(f"\n{method_name}:")
        print(f"  Max |∇·B|: {max_div:.2e}")
        print(f"  Mean |∇·B|: {mean_div:.2e}")
        
        return max_div, mean_div
    
    def plot_divergence(self, B_R, B_Z, title="Divergence ∇·B", figsize=(10, 8)):
        div_B = self.compute_divergence_fast(B_R, B_Z)
        
        plt.figure(figsize=figsize)
        contour = plt.contourf(self.R_grid, self.Z_grid, div_B, levels=50, cmap='seismic')
        plt.colorbar(contour, label='∇·B')
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.title(title)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return div_B
    
    def compute_pressure_profile(self, p0=1e5, alpha=2.0):
        psi_norm = self.get_psi_norm()
        p = p0 * psi_norm**alpha
        return p, psi_norm
    
    def compute_safety_factor(self, B_phi0=1.0, q0=1.0):
        r_squared = ((self.R_grid - self.R0)**2 + self.Z_grid**2) / self.a**2
        r = np.sqrt(r_squared)
        
        q = q0 * (1 + 2 * r_squared)
        q[r < 0.01] = q0
        
        return q
    
    def compute_magnetic_shear(self, q):
        s = np.zeros_like(q)
        for i in range(1, self.nr-1):
            for j in range(1, self.nz-1):
                r = np.sqrt(((self.R_grid[i,j] - self.R0)/self.a)**2 + (self.Z_grid[i,j]/self.a)**2)
                if r > 0.01:
                    q_grad_R = (q[i+1,j] - q[i-1,j]) / (2 * self.dR)
                    q_grad_Z = (q[i,j+1] - q[i,j-1]) / (2 * self.dZ)
                    grad_q_mag = np.sqrt(q_grad_R**2 + q_grad_Z**2)
                    s[i,j] = (r / q[i,j]) * grad_q_mag * self.a
        
        return s
    
    def compute_alpha_parameter(self, p0=1e5, alpha=2.0, B0=1.0):
        mu0 = 4 * np.pi * 1e-7
        p, psi_norm = self.compute_pressure_profile(p0, alpha)
        
        beta = 2 * mu0 * p / B0**2
        
        alpha_param = np.zeros_like(beta)
        for i in range(1, self.nr-1):
            for j in range(1, self.nz-1):
                r = np.sqrt(((self.R_grid[i,j] - self.R0)/self.a)**2 + (self.Z_grid[i,j]/self.a)**2)
                if r > 0.01 and psi_norm[i,j] > 0.01:
                    dp_dr = -p0 * alpha * psi_norm[i,j]**(alpha-1) * (2 * r / self.a)
                    alpha_param[i,j] = (-2 * mu0 * self.R0 / B0**2) * dp_dr
        
        return beta, alpha_param
    
    def ballooning_stability_criterion(self, s, alpha, q):
        critical_alpha = 0.6 * s / q
        
        unstable = alpha > critical_alpha
        
        margin = critical_alpha - alpha
        
        return unstable, margin, critical_alpha
    
    def second_stability_region_analysis(self, s, alpha, beta, q):
        in_second_region = np.zeros_like(s, dtype=bool)
        access_potential = np.zeros_like(s)
        
        for i in range(self.nr):
            for j in range(self.nz):
                if q[i,j] > 2 and s[i,j] > 0.1:
                    if alpha[i,j] > 0.5 * s[i,j] and beta[i,j] > 0.02:
                        in_second_region[i,j] = True
                
                access_potential[i,j] = min(1.0, (beta[i,j] / 0.05) * (s[i,j] / 0.3) * max(0, (q[i,j] - 1)))
        
        return in_second_region, access_potential
    
    def analyze_stability(self, p0=1e5, alpha=2.0, B0=1.0, q0=1.5):
        p, psi_norm = self.compute_pressure_profile(p0, alpha)
        q = self.compute_safety_factor(B0, q0)
        s = self.compute_magnetic_shear(q)
        beta, alpha_param = self.compute_alpha_parameter(p0, alpha, B0)
        
        unstable, margin, critical_alpha = self.ballooning_stability_criterion(s, alpha_param, q)
        
        in_second_region, access_potential = self.second_stability_region_analysis(s, alpha_param, beta, q)
        
        results = {
            'p': p,
            'psi_norm': psi_norm,
            'q': q,
            's': s,
            'beta': beta,
            'alpha_param': alpha_param,
            'unstable': unstable,
            'margin': margin,
            'critical_alpha': critical_alpha,
            'in_second_region': in_second_region,
            'access_potential': access_potential
        }
        
        return results
    
    def print_stability_summary(self, results):
        print("\n" + "="*70)
        print("气球模稳定性分析")
        print("="*70)
        
        s_mean = np.mean(results['s'][1:-1, 1:-1])
        alpha_mean = np.mean(results['alpha_param'][1:-1, 1:-1])
        beta_max = np.max(results['beta'])
        q_min = np.min(results['q'][results['q'] > 0])
        
        unstable_fraction = np.mean(results['unstable'][1:-1, 1:-1])
        second_region_fraction = np.mean(results['in_second_region'][1:-1, 1:-1])
        access_mean = np.mean(results['access_potential'][1:-1, 1:-1])
        
        print(f"\n关键参数:")
        print(f"  平均磁剪切 s: {s_mean:.3f}")
        print(f"  平均 α 参数: {alpha_mean:.3f}")
        print(f"  最大 β: {beta_max:.4f} ({beta_max*100:.2f}%)")
        print(f"  最小安全因子 q_min: {q_min:.2f}")
        
        print(f"\n稳定性分析:")
        print(f"  不稳定区域比例: {unstable_fraction*100:.1f}%")
        
        if unstable_fraction < 0.1:
            print("  ✓ 整体稳定（第一稳定区）")
        elif unstable_fraction < 0.5:
            print("  ○ 部分区域不稳定")
        else:
            print("  ✗ 大部分区域不稳定")
        
        print(f"\n第二稳定区分析:")
        print(f"  第二稳定区比例: {second_region_fraction*100:.1f}%")
        print(f"  平均访问潜力: {access_mean:.3f}")
        
        if access_mean > 0.5:
            print("  ✓ 第二稳定区访问可能性高")
        elif access_mean > 0.2:
            print("  ○ 第二稳定区访问可能性中等")
        else:
            print("  ✗ 第二稳定区访问可能性低")
        
        print("\n" + "="*70)
    
    def plot_stability_maps(self, results, figsize=(15, 10)):
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        plots = [
            ('q 分布', results['q'], 'viridis', 'q'),
            ('磁剪切 s', results['s'], 'plasma', 's'),
            ('β 分布 (%)', results['beta'] * 100, 'hot', 'β (%)'),
            ('α 参数', results['alpha_param'], 'jet', 'α'),
            ('不稳定区域', results['unstable'].astype(float), 'RdYlGn_r', '不稳定'),
            ('第二稳定区访问潜力', results['access_potential'], 'RdYlGn', '访问潜力')
        ]
        
        for ax, (title, data, cmap, label) in zip(axes.flat, plots):
            contour = ax.contourf(self.R_grid, self.Z_grid, data, levels=20, cmap=cmap)
            plt.colorbar(contour, ax=ax, label=label)
            ax.set_xlabel('R (m)')
            ax.set_ylabel('Z (m)')
            ax.set_title(title)
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_stability_diagram(self, results, figsize=(10, 8)):
        s_flat = results['s'][1:-1, 1:-1].flatten()
        alpha_flat = results['alpha_param'][1:-1, 1:-1].flatten()
        unstable_flat = results['unstable'][1:-1, 1:-1].flatten()
        
        plt.figure(figsize=figsize)
        
        plt.scatter(s_flat[~unstable_flat], alpha_flat[~unstable_flat], 
                   c='g', alpha=0.5, label='稳定', s=10)
        plt.scatter(s_flat[unstable_flat], alpha_flat[unstable_flat], 
                   c='r', alpha=0.5, label='不稳定', s=10)
        
        s_line = np.linspace(0, 2, 100)
        alpha_critical = 0.6 * s_line / 1.5
        plt.plot(s_line, alpha_critical, 'k--', linewidth=2, label='临界稳定性')
        
        plt.fill_between(s_line, alpha_critical, alpha_critical + 2, 
                        color='orange', alpha=0.2, label='第二稳定区方向')
        
        plt.xlabel('磁剪切 s')
        plt.ylabel('α 参数')
        plt.title('气球模稳定性 s-α 图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(1, np.max(s_flat) * 1.1))
        plt.ylim(0, max(1, np.max(alpha_flat) * 1.1))
        plt.tight_layout()
        plt.show()
        
    def plot_magnetic_field(self, density=1.5, figsize=(10, 8)):
        B_R, B_Z = self.get_magnetic_field()
        
        plt.figure(figsize=figsize)
        plt.streamplot(self.R_grid.T, self.Z_grid.T, B_R.T, B_Z.T, 
                       density=density, cmap='coolwarm', linewidth=1.2, arrowsize=1.5)
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.title('Poloidal Magnetic Field')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_psi_3d(self, figsize=(12, 8)):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.R_grid, self.Z_grid, self.psi, cmap='viridis', alpha=0.8)
        fig.colorbar(surf, label='Poloidal Flux ψ')
        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('ψ')
        ax.set_title('3D View of Poloidal Flux')
        plt.tight_layout()
        plt.show()


def main():
    print("="*70)
    print("Grad-Shafranov Equation Solver for Tokamak Equilibrium")
    print("="*70)
    
    solver = GradShafranovSolver(Rmin=0.5, Rmax=1.5, Zmin=-0.6, Zmax=0.6, nr=65, nz=65)
    
    print("\nSolver Configuration:")
    print(f"  R range: [{solver.Rmin}, {solver.Rmax}] m")
    print(f"  Z range: [{solver.Zmin}, {solver.Zmax}] m")
    print(f"  Grid size: {solver.nr} x {solver.nz}")
    print(f"  Major radius R0: {solver.R0} m")
    print(f"  Minor radius a: {solver.a} m")
    
    print("\n" + "="*70)
    print("Solving Grad-Shafranov equation...")
    print("="*70)
    
    p0 = 2e5
    alpha_p = 2.0
    q0 = 2.0
    
    psi = solver.solve(p0=p0, alpha=alpha_p, F0=1.0, beta=0.05, max_iter=150, tol=1e-6, relaxation=0.5)
    
    print("\n" + "="*70)
    print("Magnetic Field Divergence Analysis")
    print("="*70)
    
    B_R_raw, B_Z_raw = solver.get_magnetic_field(clean_divergence=False)
    solver.check_divergence_error(B_R_raw, B_Z_raw, "Before divergence cleaning")
    
    B_R_clean, B_Z_clean = solver.get_magnetic_field(clean_divergence=True)
    solver.check_divergence_error(B_R_clean, B_Z_clean, "After divergence cleaning")
    
    print("\n" + "="*70)
    print("Ballooning Mode Stability Analysis")
    print("="*70)
    
    results = solver.analyze_stability(p0=p0, alpha=alpha_p, q0=q0)
    solver.print_stability_summary(results)
    
    print("\n" + "="*70)
    print("Generating plots...")
    print("="*70)
    
    solver.plot_psi(levels=18)
    solver.plot_magnetic_field(density=1.5)
    solver.plot_stability_maps(results)
    solver.plot_stability_diagram(results)
    
    print("\nDone!")
    print("\n运行 test_stability.py 进行多场景第二稳定区分析")
    
    return solver


if __name__ == "__main__":
    solver = main()
