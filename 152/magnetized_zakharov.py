import numpy as np
import matplotlib.pyplot as plt
import time

class MagnetizedZakharov2D:
    def __init__(self, N=256, L=50.0, dt=0.01, t_max=50.0,
                 dealiasing='2/3', B0=0.0, omega_c=0.5, theta=0.0):
        self.N = N
        self.L = L
        self.dt = dt
        self.t_max = t_max
        self.n_steps = int(t_max / dt)
        self.dealiasing = dealiasing
        
        self.B0 = B0
        self.omega_c = omega_c
        self.theta = theta
        
        self.x = np.linspace(-L/2, L/2, N, endpoint=False)
        self.y = np.linspace(-L/2, L/2, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.kx = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
        self.ky = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.k_sq = self.KX**2 + self.KY**2
        self.k_max = np.max(np.abs(self.kx))
        
        self._setup_magnetic_dispersion()
        self._setup_dealiasing()
        
        self.E_plus = None
        self.E_minus = None
        self.n = None
        self.n_t = None
    
    def _setup_magnetic_dispersion(self):
        k_parallel = self.KX * np.cos(self.theta) + self.KY * np.sin(self.theta)
        k_perp = np.sqrt(self.k_sq - k_parallel**2 + 1e-10)
        
        omega_pe = 1.0
        omega_L = omega_pe**2 / (omega_pe + self.omega_c)
        omega_R = omega_pe**2 / (omega_pe - self.omega_c)
        
        k_sq_safe = np.where(self.k_sq == 0, 1e-10, self.k_sq)
        omega_plus = np.sqrt(omega_pe**2 + 3 * k_sq_safe + self.omega_c**2 + 
                            np.sqrt((3 * k_sq_safe + self.omega_c**2)**2 + 
                                   12 * self.omega_c**2 * k_perp**2)) / np.sqrt(2)
        omega_minus = np.sqrt(omega_pe**2 + 3 * k_sq_safe + self.omega_c**2 - 
                             np.sqrt((3 * k_sq_safe + self.omega_c**2)**2 + 
                                    12 * self.omega_c**2 * k_perp**2)) / np.sqrt(2)
        
        self.prop_half_plus = np.exp(-1j * omega_plus * self.dt / 2)
        self.prop_half_minus = np.exp(-1j * omega_minus * self.dt / 2)
        
        k_sq_safe = np.where(self.k_sq == 0, 1e-10, self.k_sq)
        self.k_sq_safe = k_sq_safe
        self.sqrt_k = np.sqrt(k_sq_safe)
        self.n_prop_cos = np.cos(self.sqrt_k * self.dt)
        self.n_prop_sin = np.sin(self.sqrt_k * self.dt) / self.sqrt_k
    
    def _setup_dealiasing(self):
        if self.dealiasing == '2/3':
            k_cutoff = 2.0/3.0 * self.k_max
            mask_x = (np.abs(self.KX) <= k_cutoff).astype(float)
            mask_y = (np.abs(self.KY) <= k_cutoff).astype(float)
            self.dealias_mask = mask_x * mask_y
        elif self.dealiasing == 'exponential':
            k_cutoff = 2.0/3.0 * self.k_max
            k_mag = np.sqrt(self.k_sq)
            k_ratio = k_mag / k_cutoff
            self.dealias_mask = np.exp(-36.0 * k_ratio**36)
            self.dealias_mask = np.where(k_ratio <= 1.0, 1.0, self.dealias_mask)
        elif self.dealiasing == 'none':
            self.dealias_mask = np.ones_like(self.k_sq)
    
    def _apply_dealiasing(self, field_hat):
        return field_hat * self.dealias_mask
    
    def set_initial_condition(self, amp=3.0, width=5.0, noise_amp=0.05, mode='both'):
        r_sq = self.X**2 + self.Y**2
        profile = amp * np.exp(-r_sq / (2 * width**2))
        
        if mode == 'both':
            self.E_plus = profile * (1 + noise_amp * np.random.randn(self.N, self.N)) * 0.5
            self.E_minus = profile * (1 + noise_amp * np.random.randn(self.N, self.N)) * 0.5
        elif mode == 'plus':
            self.E_plus = profile * (1 + noise_amp * np.random.randn(self.N, self.N))
            self.E_minus = np.zeros_like(self.E_plus)
        elif mode == 'minus':
            self.E_minus = profile * (1 + noise_amp * np.random.randn(self.N, self.N))
            self.E_plus = np.zeros_like(self.E_minus)
        
        E_plus_hat = self._apply_dealiasing(np.fft.fft2(self.E_plus))
        E_minus_hat = self._apply_dealiasing(np.fft.fft2(self.E_minus))
        self.E_plus = np.fft.ifft2(E_plus_hat)
        self.E_minus = np.fft.ifft2(E_minus_hat)
        
        E_total_sq = np.abs(self.E_plus)**2 + np.abs(self.E_minus)**2
        E_sq_hat = self._apply_dealiasing(np.fft.fft2(E_total_sq))
        self.n = -np.fft.ifft2(E_sq_hat * self.k_sq / self.k_sq_safe).real
        self.n_t = np.zeros_like(self.n)
    
    def step(self):
        E_plus_hat = np.fft.fft2(self.E_plus)
        E_minus_hat = np.fft.fft2(self.E_minus)
        
        E_plus_hat *= self.prop_half_plus
        E_minus_hat *= self.prop_half_minus
        
        self.E_plus = np.fft.ifft2(E_plus_hat)
        self.E_minus = np.fft.ifft2(E_minus_hat)
        
        n_hat = np.fft.fft2(self.n)
        n_t_hat = np.fft.fft2(self.n_t)
        
        E_total_sq = np.abs(self.E_plus)**2 + np.abs(self.E_minus)**2
        E_sq_hat = self._apply_dealiasing(np.fft.fft2(E_total_sq))
        rhs_hat = -E_sq_hat * self.k_sq
        
        n_new_hat = n_hat * self.n_prop_cos + n_t_hat * self.n_prop_sin + \
                    rhs_hat * (1 - self.n_prop_cos) / self.k_sq_safe
        n_t_new_hat = -n_hat * self.k_sq * self.n_prop_sin + n_t_hat * self.n_prop_cos + \
                      rhs_hat * self.n_prop_sin
        
        n_new_hat = self._apply_dealiasing(n_new_hat)
        n_t_new_hat = self._apply_dealiasing(n_t_new_hat)
        
        self.n = np.fft.ifft2(n_new_hat).real
        self.n_t = np.fft.ifft2(n_t_new_hat).real
        
        n_hat = self._apply_dealiasing(np.fft.fft2(self.n))
        self.n = np.fft.ifft2(n_hat).real
        
        phase_factor = np.exp(-1j * self.n * self.dt)
        self.E_plus *= phase_factor
        self.E_minus *= phase_factor
        
        E_plus_hat = np.fft.fft2(self.E_plus)
        E_minus_hat = np.fft.fft2(self.E_minus)
        
        E_plus_hat = self._apply_dealiasing(E_plus_hat)
        E_minus_hat = self._apply_dealiasing(E_minus_hat)
        
        E_plus_hat *= self.prop_half_plus
        E_minus_hat *= self.prop_half_minus
        
        self.E_plus = np.fft.ifft2(E_plus_hat)
        self.E_minus = np.fft.ifft2(E_minus_hat)
    
    def get_total_energy(self):
        return (np.sum(np.abs(self.E_plus)**2 + np.abs(self.E_minus)**2) * 
                (self.L / self.N)**2)
    
    def get_max_intensity(self):
        return np.max(np.abs(self.E_plus)**2 + np.abs(self.E_minus)**2)

def run_magnetized_simulation():
    print("=" * 70)
    print("磁化等离子体 Zakharov 方程求解器")
    print("激光等离子体相互作用 - 惯性约束聚变相关模拟")
    print("=" * 70)
    
    B0 = 1.0
    omega_c = 0.3
    theta = np.pi / 4
    
    solver = MagnetizedZakharov2D(N=256, L=60.0, dt=0.02, t_max=25.0,
                                  dealiasing='2/3', B0=B0, omega_c=omega_c, theta=theta)
    
    print(f"\n磁场参数:")
    print(f"  B0 = {B0}, ω_c/ω_pe = {omega_c}")
    print(f"  磁场与x轴夹角: {theta*180/np.pi:.1f}°")
    
    solver.set_initial_condition(amp=4.0, width=6.0, noise_amp=0.08, mode='both')
    
    print(f"\n网格大小: {solver.N} x {solver.N}")
    print(f"初始总能量: {solver.get_total_energy():.4f}")
    print(f"初始最大强度: {solver.get_max_intensity():.4f}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'磁化等离子体模拟 (ω_c/ω_pe = {omega_c})', fontsize=16, fontweight='bold')
    
    im1 = axes[0, 0].imshow(np.abs(solver.E_plus)**2, cmap='hot',
                           extent=[-solver.L/2, solver.L/2, -solver.L/2, solver.L/2],
                           origin='lower', vmin=0, vmax=15)
    axes[0, 0].set_title('右旋波强度 |E+|²', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(np.abs(solver.E_minus)**2, cmap='hot',
                           extent=[-solver.L/2, solver.L/2, -solver.L/2, solver.L/2],
                           origin='lower', vmin=0, vmax=15)
    axes[0, 1].set_title('左旋波强度 |E-|²', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(solver.n, cmap='RdBu_r',
                           extent=[-solver.L/2, solver.L/2, -solver.L/2, solver.L/2],
                           origin='lower')
    axes[0, 2].set_title('密度扰动 n', fontsize=12)
    plt.colorbar(im3, ax=axes[0, 2])
    
    time_list = [0]
    energy_list = [solver.get_total_energy()]
    max_int_list = [solver.get_max_intensity()]
    energy_plus_list = [np.sum(np.abs(solver.E_plus)**2) * (solver.L/solver.N)**2]
    energy_minus_list = [np.sum(np.abs(solver.E_minus)**2) * (solver.L/solver.N)**2]
    
    line1, = axes[1, 0].plot(time_list, energy_list, 'g-', linewidth=2, label='总能量')
    line1p, = axes[1, 0].plot(time_list, energy_plus_list, 'r--', linewidth=2, label='E+能量')
    line1m, = axes[1, 0].plot(time_list, energy_minus_list, 'b--', linewidth=2, label='E-能量')
    axes[1, 0].set_title('能量守恒', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    line2, = axes[1, 1].plot(time_list, max_int_list, 'r-', linewidth=2)
    axes[1, 1].set_title('最大强度演化', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    k_sorted = np.fft.fftshift(solver.kx)
    spectrum_plus = np.fft.fftshift(np.sum(np.abs(np.fft.fft2(solver.E_plus))**2, axis=0))
    spectrum_minus = np.fft.fftshift(np.sum(np.abs(np.fft.fft2(solver.E_minus))**2, axis=0))
    line3p, = axes[1, 2].plot(k_sorted, spectrum_plus, 'r-', label='E+谱')
    line3m, = axes[1, 2].plot(k_sorted, spectrum_minus, 'b-', label='E-谱')
    axes[1, 2].set_title('波谱 (kx方向)', fontsize=12)
    axes[1, 2].set_xlim(-solver.k_max, solver.k_max)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\n开始模拟...")
    start_time = time.time()
    plot_interval = 10
    
    for step in range(solver.n_steps):
        solver.step()
        
        if step % plot_interval == 0:
            current_time = (step + 1) * solver.dt
            current_energy = solver.get_total_energy()
            current_max = solver.get_max_intensity()
            
            time_list.append(current_time)
            energy_list.append(current_energy)
            max_int_list.append(current_max)
            energy_plus_list.append(np.sum(np.abs(solver.E_plus)**2) * (solver.L/solver.N)**2)
            energy_minus_list.append(np.sum(np.abs(solver.E_minus)**2) * (solver.L/solver.N)**2)
            
            im1.set_data(np.abs(solver.E_plus)**2)
            im2.set_data(np.abs(solver.E_minus)**2)
            im3.set_data(solver.n)
            im3.set_clim(vmin=np.min(solver.n), vmax=np.max(solver.n))
            
            line1.set_data(time_list, energy_list)
            line1p.set_data(time_list, energy_plus_list)
            line1m.set_data(time_list, energy_minus_list)
            axes[1, 0].set_xlim(0, current_time + solver.dt * plot_interval)
            axes[1, 0].set_ylim(0.9 * min(energy_list), 1.1 * max(energy_list))
            
            line2.set_data(time_list, max_int_list)
            axes[1, 1].set_xlim(0, current_time + solver.dt * plot_interval)
            axes[1, 1].set_ylim(0, 1.2 * max(max_int_list))
            
            spectrum_plus = np.fft.fftshift(np.sum(np.abs(np.fft.fft2(solver.E_plus))**2, axis=0))
            spectrum_minus = np.fft.fftshift(np.sum(np.abs(np.fft.fft2(solver.E_minus))**2, axis=0))
            line3p.set_ydata(spectrum_plus)
            line3m.set_ydata(spectrum_minus)
            axes[1, 2].set_ylim(0, 1.2 * max(np.max(spectrum_plus), np.max(spectrum_minus)))
            
            fig.suptitle(f'磁化等离子体模拟 - t = {current_time:.2f}', fontsize=16, fontweight='bold')
            
            plt.pause(0.01)
            
            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"步骤 {step}/{solver.n_steps} | t={current_time:.2f} | "
                      f"能量={current_energy:.4f} | 最大强度={current_max:.2f} | "
                      f"耗时={elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n模拟完成! 总耗时: {total_time:.2f}秒")
    print(f"最终能量: {solver.get_total_energy():.4f}")
    
    plt.savefig('magnetized_zakharov.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存至: magnetized_zakharov.png")
    
    plt.show()

if __name__ == "__main__":
    run_magnetized_simulation()
