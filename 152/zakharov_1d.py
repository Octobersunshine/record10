import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

class ZakharovSolver1D:
    def __init__(self, N=1024, L=100.0, dt=0.01, t_max=50.0, 
                 dealiasing='2/3', filter_parameter=36.0):
        self.N = N
        self.L = L
        self.dt = dt
        self.t_max = t_max
        self.n_steps = int(t_max / dt)
        self.dealiasing = dealiasing
        
        self.x = np.linspace(-L/2, L/2, N, endpoint=False)
        self.k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
        self.k_sq = self.k**2
        self.k_max = np.max(np.abs(self.k))
        
        self.prop_half = np.exp(-1j * self.k_sq * dt / 2)
        
        k_sq_safe = np.where(self.k_sq == 0, 1e-10, self.k_sq)
        self.k_sq_safe = k_sq_safe
        self.sqrt_k = np.sqrt(k_sq_safe)
        self.n_prop_cos = np.cos(self.sqrt_k * dt)
        self.n_prop_sin = np.sin(self.sqrt_k * dt) / self.sqrt_k
        
        self._setup_dealiasing(filter_parameter)
        
        self.E = None
        self.n = None
        self.n_t = None
    
    def _setup_dealiasing(self, filter_parameter):
        if self.dealiasing == '2/3':
            k_cutoff = 2.0/3.0 * self.k_max
            self.dealias_mask = (np.abs(self.k) <= k_cutoff).astype(float)
        elif self.dealiasing == 'exponential':
            k_cutoff = 2.0/3.0 * self.k_max
            k_ratio = self.k / k_cutoff
            self.dealias_mask = np.exp(-filter_parameter * k_ratio**36)
            self.dealias_mask = np.where(np.abs(k_ratio) <= 1.0, 1.0, self.dealias_mask)
        elif self.dealiasing == 'none':
            self.dealias_mask = np.ones_like(self.k)
        else:
            raise ValueError(f"Unknown dealiasing method: {self.dealiasing}")
    
    def _apply_dealiasing(self, field_hat):
        return field_hat * self.dealias_mask
        
    def set_initial_condition(self, amp=2.5, width=8.0, noise_amp=0.02, two_pulse=False):
        if two_pulse:
            r1 = (self.x - 10.0)**2
            r2 = (self.x + 10.0)**2
            self.E = amp * (np.exp(-r1 / (2 * width**2)) + np.exp(-r2 / (2 * width**2)))
        else:
            r_sq = self.x**2
            self.E = amp * np.exp(-r_sq / (2 * width**2))
        
        self.E *= (1 + noise_amp * np.random.randn(self.N))
        
        E_hat = self._apply_dealiasing(np.fft.fft(self.E))
        self.E = np.fft.ifft(E_hat)
        
        E_sq_hat = self._apply_dealiasing(np.fft.fft(np.abs(self.E)**2))
        self.n = -np.fft.ifft(E_sq_hat * self.k_sq / self.k_sq_safe).real
        self.n_t = np.zeros_like(self.n)
        
    def step(self):
        E_hat = np.fft.fft(self.E)
        E_hat *= self.prop_half
        self.E = np.fft.ifft(E_hat)
        
        n_hat = np.fft.fft(self.n)
        n_t_hat = np.fft.fft(self.n_t)
        
        E_sq_hat = self._apply_dealiasing(np.fft.fft(np.abs(self.E)**2))
        rhs_hat = -E_sq_hat * self.k_sq
        
        n_new_hat = n_hat * self.n_prop_cos + n_t_hat * self.n_prop_sin + \
                    rhs_hat * (1 - self.n_prop_cos) / self.k_sq_safe
        n_t_new_hat = -n_hat * self.k_sq * self.n_prop_sin + n_t_hat * self.n_prop_cos + \
                      rhs_hat * self.n_prop_sin
        
        n_new_hat = self._apply_dealiasing(n_new_hat)
        n_t_new_hat = self._apply_dealiasing(n_t_new_hat)
        
        self.n = np.fft.ifft(n_new_hat).real
        self.n_t = np.fft.ifft(n_t_new_hat).real
        
        n_hat = self._apply_dealiasing(np.fft.fft(self.n))
        self.n = np.fft.ifft(n_hat).real
        
        self.E *= np.exp(-1j * self.n * self.dt)
        
        E_hat = np.fft.fft(self.E)
        E_hat = self._apply_dealiasing(E_hat)
        E_hat *= self.prop_half
        self.E = np.fft.ifft(E_hat)
        
    def get_energy(self):
        return np.sum(np.abs(self.E)**2) * (self.L / self.N)
    
    def get_max_intensity(self):
        return np.max(np.abs(self.E)**2)

def run_1d_simulation(dealiasing_method='2/3'):
    print("=" * 60)
    print("Zakharov方程 1D 求解器")
    print("描述朗缪尔波与离子声波的耦合及塌缩过程")
    print("=" * 60)
    
    solver = ZakharovSolver1D(N=1024, L=150.0, dt=0.01, t_max=40.0,
                              dealiasing=dealiasing_method)
    
    print("\n设置初始条件（双脉冲碰撞）...")
    solver.set_initial_condition(amp=3.0, width=6.0, noise_amp=0.03, two_pulse=True)
    
    print(f"网格点数: {solver.N}")
    print(f"计算域: [-{solver.L/2}, {solver.L/2}]")
    print(f"时间步长: {solver.dt}, 总步数: {solver.n_steps}")
    print(f"去混叠方法: {solver.dealiasing}")
    print(f"初始能量: {solver.get_energy():.4f}")
    print(f"初始最大强度: {solver.get_max_intensity():.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Zakharov方程 1D 模拟 - 波-波相互作用与塌缩', fontsize=15, fontweight='bold')
    
    line1, = axes[0, 0].plot(solver.x, np.abs(solver.E)**2, 'r-', linewidth=2)
    axes[0, 0].set_title('朗缪尔波强度 |E|²', fontsize=12)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('|E|²')
    axes[0, 0].set_ylim(0, 20)
    axes[0, 0].grid(True, alpha=0.3)
    
    line2, = axes[0, 1].plot(solver.x, solver.n, 'b-', linewidth=2)
    axes[0, 1].set_title('密度扰动 n', fontsize=12)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('n')
    axes[0, 1].grid(True, alpha=0.3)
    
    time_list = [0]
    energy_list = [solver.get_energy()]
    max_int_list = [solver.get_max_intensity()]
    
    line3, = axes[1, 0].plot(time_list, energy_list, 'g-', linewidth=2)
    axes[1, 0].set_title('能量守恒', fontsize=12)
    axes[1, 0].set_xlabel('时间 t')
    axes[1, 0].set_ylabel('总能量')
    axes[1, 0].grid(True, alpha=0.3)
    
    line4, = axes[1, 1].plot(time_list, max_int_list, 'r-', linewidth=2)
    axes[1, 1].set_title('最大强度演化', fontsize=12)
    axes[1, 1].set_xlabel('时间 t')
    axes[1, 1].set_ylabel('max(|E|²)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\n开始模拟...")
    start_time = time.time()
    plot_interval = 5
    
    for step in range(solver.n_steps):
        solver.step()
        
        if step % plot_interval == 0:
            current_time = (step + 1) * solver.dt
            current_energy = solver.get_energy()
            current_max = solver.get_max_intensity()
            
            time_list.append(current_time)
            energy_list.append(current_energy)
            max_int_list.append(current_max)
            
            line1.set_ydata(np.abs(solver.E)**2)
            axes[0, 0].set_ylim(0, max(15, 1.2 * current_max))
            
            line2.set_ydata(solver.n)
            axes[0, 1].set_ylim(1.1 * np.min(solver.n), 1.1 * np.max(solver.n))
            
            line3.set_data(time_list, energy_list)
            axes[1, 0].set_xlim(0, current_time + solver.dt * plot_interval)
            axes[1, 0].set_ylim(0.95 * min(energy_list), 1.05 * max(energy_list))
            
            line4.set_data(time_list, max_int_list)
            axes[1, 1].set_xlim(0, current_time + solver.dt * plot_interval)
            axes[1, 1].set_ylim(0, 1.2 * max(max_int_list))
            
            fig.suptitle(f'Zakharov方程 1D 模拟 - t = {current_time:.2f}', fontsize=15, fontweight='bold')
            
            plt.pause(0.001)
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"步骤 {step}/{solver.n_steps} | t={current_time:.2f} | "
                      f"能量={current_energy:.4f} | 最大强度={current_max:.2f} | "
                      f"耗时={elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n模拟完成! 总耗时: {total_time:.2f}秒")
    print(f"最终能量: {solver.get_energy():.4f}")
    print(f"最终最大强度: {solver.get_max_intensity():.4f}")
    
    plt.savefig('zakharov_1d_result.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存至: zakharov_1d_result.png")
    
    plt.show()

if __name__ == "__main__":
    run_1d_simulation(dealiasing_method='2/3')
