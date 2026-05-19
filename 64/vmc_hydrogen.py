import numpy as np
import matplotlib.pyplot as plt


class VMC_Hydrogen:
    def __init__(self, alpha, n_walkers=1000, n_steps=1000, step_size=1.0,
                 target_accept_rate=0.5, adjust_interval=10, adjust_factor=1.05):
        self.alpha = alpha
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.step_size = step_size
        self.target_accept_rate = target_accept_rate
        self.adjust_interval = adjust_interval
        self.adjust_factor = adjust_factor
        self.step_size_history = []
        
    def wave_function(self, r):
        return np.exp(-self.alpha * r)
    
    def log_wave_function_derivative(self, r):
        return -r
    
    def local_energy(self, r):
        return -0.5 * self.alpha**2 + (self.alpha - 1.0) / r
    
    def metropolis_step(self, positions):
        new_positions = positions + np.random.uniform(-self.step_size, self.step_size, size=positions.shape)
        r_old = np.linalg.norm(positions, axis=1)
        r_new = np.linalg.norm(new_positions, axis=1)
        
        ratio = (self.wave_function(r_new) / self.wave_function(r_old))**2
        accept = np.random.rand(self.n_walkers) < ratio
        
        positions[accept] = new_positions[accept]
        return positions, np.mean(accept)
    
    def adjust_step_size(self, accept_rate):
        if accept_rate > self.target_accept_rate:
            self.step_size *= self.adjust_factor
        else:
            self.step_size /= self.adjust_factor
    
    def compute_stochastic_reconfiguration(self, positions, E_L, S_diag_reg=1e-6):
        r = np.linalg.norm(positions, axis=1)
        O_k = self.log_wave_function_derivative(r)
        
        E_mean = np.mean(E_L)
        O_mean = np.mean(O_k)
        
        S_kk = np.mean(O_k**2) - O_mean**2 + S_diag_reg
        F_k = np.mean(E_L * O_k) - E_mean * O_mean
        
        delta_alpha = -F_k / S_kk
        
        return delta_alpha, E_mean, S_kk, F_k
    
    def run(self, thermalization=200, adapt_step_size=True, verbose=False,
            compute_gradient=False):
        positions = np.random.randn(self.n_walkers, 3)
        energies = []
        accept_rates = []
        recent_accept_rates = []
        gradients = []
        
        for step in range(self.n_steps):
            positions, accept_rate = self.metropolis_step(positions)
            accept_rates.append(accept_rate)
            recent_accept_rates.append(accept_rate)
            
            if adapt_step_size and step < thermalization and (step + 1) % self.adjust_interval == 0:
                avg_accept = np.mean(recent_accept_rates[-self.adjust_interval:])
                self.adjust_step_size(avg_accept)
                
                if verbose and (step + 1) % (self.adjust_interval * 5) == 0:
                    print(f"  步 {step+1}: 接受率 = {avg_accept:.2%}, 步长 = {self.step_size:.4f}")
            
            self.step_size_history.append(self.step_size)
            
            if step >= thermalization:
                r = np.linalg.norm(positions, axis=1)
                e_local = self.local_energy(r)
                energies.append(np.mean(e_local))
                
                if compute_gradient:
                    delta_alpha, _, _, _ = self.compute_stochastic_reconfiguration(positions, e_local)
                    gradients.append(delta_alpha)
        
        if compute_gradient:
            return np.array(energies), np.mean(accept_rates), np.array(self.step_size_history), np.mean(gradients)
        return np.array(energies), np.mean(accept_rates), np.array(self.step_size_history)
    
    def virtual_time_propagation(self, n_iterations=50, dt=0.1, n_steps_per_iter=500,
                                 thermalization=200, verbose=True):
        alpha_history = [self.alpha]
        energy_history = []
        gradient_history = []
        
        original_n_steps = self.n_steps
        self.n_steps = n_steps_per_iter
        
        if verbose:
            print(f"\n虚拟时间演化开始:")
            print(f"  初始 alpha = {self.alpha:.6f}")
            print(f"  时间步长 dt = {dt}")
            print(f"  迭代次数 = {n_iterations}")
            print(f"  每迭代步数 = {n_steps_per_iter}")
            print(f"\n{'迭代':^6} {'alpha':^12} {'能量':^12} {'梯度':^12} {'步长':^10}")
            print("-" * 56)
        
        for iteration in range(n_iterations):
            energies, _, step_size_hist, avg_gradient = self.run(
                thermalization=thermalization,
                adapt_step_size=True,
                verbose=False,
                compute_gradient=True
            )
            
            current_energy = np.mean(energies)
            energy_history.append(current_energy)
            gradient_history.append(avg_gradient)
            
            self.alpha += dt * avg_gradient
            alpha_history.append(self.alpha)
            
            if verbose and (iteration % 5 == 0 or iteration == n_iterations - 1):
                print(f"{iteration:^6} {self.alpha:^12.6f} {current_energy:^12.6f} "
                      f"{avg_gradient:^12.6f} {step_size_hist[-1]:^10.4f}")
        
        if verbose:
            print("-" * 56)
            print(f"\n虚拟时间演化完成!")
            print(f"  最终 alpha = {self.alpha:.6f}")
            print(f"  最终能量 = {energy_history[-1]:.6f} Hartree")
        
        self.n_steps = original_n_steps
        
        return np.array(alpha_history), np.array(energy_history), np.array(gradient_history)


def optimize_alpha(alpha_range, n_walkers=2000, n_steps=2000, initial_step_size=0.1):
    energies = []
    variances = []
    final_accept_rates = []
    final_step_sizes = []
    
    for alpha in alpha_range:
        vmc = VMC_Hydrogen(alpha, n_walkers=n_walkers, n_steps=n_steps, 
                          step_size=initial_step_size, target_accept_rate=0.5)
        energy_samples, avg_accept, step_size_hist = vmc.run(thermalization=500, adapt_step_size=True)
        energies.append(np.mean(energy_samples))
        variances.append(np.var(energy_samples))
        final_accept_rates.append(avg_accept)
        final_step_sizes.append(step_size_hist[-1])
    
    return np.array(energies), np.array(variances), np.array(final_accept_rates), np.array(final_step_sizes)


if __name__ == "__main__":
    print("=" * 70)
    print("变分蒙特卡洛计算氢原子基态能量")
    print("(自适应步长 + 随机重配置 + 虚拟时间演化)")
    print("=" * 70)
    
    exact_energy = -0.5
    print(f"\n氢原子精确基态能量: {exact_energy:.6f} Hartree")
    print(f"目标接受率: 50% (Metropolis采样最优范围)")
    
    print("\n" + "-" * 70)
    print("第一部分: 虚拟时间演化自动优化参数")
    print("-" * 70)
    
    initial_alpha = 0.7
    vmc_optimizer = VMC_Hydrogen(initial_alpha, n_walkers=2000, n_steps=500, step_size=0.1)
    
    alpha_hist, energy_hist, grad_hist = vmc_optimizer.virtual_time_propagation(
        n_iterations=40, dt=0.05, n_steps_per_iter=500, thermalization=200, verbose=True
    )
    
    print("\n" + "-" * 70)
    print("第二部分: 网格扫描验证优化结果")
    print("-" * 70)
    print("\n(热化阶段自动调整步长...)")
    
    alpha_range = np.linspace(0.8, 1.2, 20)
    energies, variances, accept_rates_list, step_sizes_list = optimize_alpha(
        alpha_range, n_walkers=2000, n_steps=2000, initial_step_size=0.1
    )
    
    min_idx = np.argmin(energies)
    optimal_alpha_grid = alpha_range[min_idx]
    min_energy_grid = energies[min_idx]
    
    print(f"\n网格扫描最优 alpha: {optimal_alpha_grid:.4f}")
    print(f"网格扫描最小能量: {min_energy_grid:.6f} Hartree")
    print(f"虚拟时间演化最终 alpha: {alpha_hist[-1]:.4f}")
    print(f"两种方法差异: {abs(optimal_alpha_grid - alpha_hist[-1]):.6f}")
    
    print("\n" + "-" * 70)
    print("第三部分: 使用优化后的参数进行详细计算")
    print("-" * 70)
    
    optimal_alpha = alpha_hist[-1]
    vmc_final = VMC_Hydrogen(optimal_alpha, n_walkers=5000, n_steps=5000, step_size=0.1)
    energy_samples, accept_rate, step_size_hist = vmc_final.run(
        thermalization=1000, adapt_step_size=True, verbose=True
    )
    
    final_energy = np.mean(energy_samples)
    energy_error = np.std(energy_samples) / np.sqrt(len(energy_samples))
    
    print(f"\n最终步长: {step_size_hist[-1]:.4f}")
    print(f"平均接受率: {accept_rate:.2%}")
    print(f"计算得到的基态能量: {final_energy:.6f} ± {energy_error:.6f} Hartree")
    print(f"与精确值的相对误差: {abs((final_energy - exact_energy) / exact_energy) * 100:.4f}%")
    
    print("\n" + "=" * 70)
    print("计算完成!")
    print("=" * 70)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(alpha_range, energies, 'b-o', markersize=4, label='网格扫描')
    ax1.plot(alpha_hist, energy_hist, 'r-', linewidth=2, label='虚拟时间演化')
    ax1.axhline(y=exact_energy, color='g', linestyle='--', linewidth=2, label='精确值')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('能量 (Hartree)')
    ax1.set_title('能量 vs alpha (网格+演化)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(alpha_range, variances, 'g-o', markersize=4)
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('方差')
    ax2.set_title('能量方差 vs alpha')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(alpha_range, accept_rates_list * 100, 'm-o', markersize=4)
    ax3.axhline(y=50, color='r', linestyle='--', label='目标 50%')
    ax3.set_xlabel('alpha')
    ax3.set_ylabel('接受率 (%)')
    ax3.set_title('接受率 vs alpha')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(alpha_range, step_sizes_list, 'c-o', markersize=4)
    ax4.set_xlabel('alpha')
    ax4.set_ylabel('最终步长')
    ax4.set_title('自适应步长 vs alpha')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.plot(alpha_hist, 'b-', linewidth=2)
    ax5.axhline(y=1.0, color='r', linestyle='--', label='理论最优 alpha=1')
    ax5.set_xlabel('迭代次数')
    ax5.set_ylabel('alpha')
    ax5.set_title('虚拟时间演化: alpha 收敛过程')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.plot(energy_hist, 'b-', linewidth=2)
    ax6.axhline(y=exact_energy, color='r', linestyle='--', label='精确能量')
    ax6.set_xlabel('迭代次数')
    ax6.set_ylabel('能量 (Hartree)')
    ax6.set_title('虚拟时间演化: 能量收敛过程')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.plot(grad_hist, 'g-', linewidth=1.5, alpha=0.7)
    ax7.axhline(y=0, color='r', linestyle='--', label='梯度=0 (收敛)')
    ax7.set_xlabel('迭代次数')
    ax7.set_ylabel('梯度 dE/dα')
    ax7.set_title('随机重配置: 梯度演化')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.plot(step_size_hist, 'b-', linewidth=1.5)
    ax8.axvline(x=1000, color='r', linestyle='--', label='热化结束')
    ax8.set_xlabel('蒙特卡洛步')
    ax8.set_ylabel('步长')
    ax8.set_title('步长自适应调整过程')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[3, 0:2])
    ax9.plot(energy_samples, 'b-', linewidth=0.5, alpha=0.7)
    ax9.axhline(y=final_energy, color='r', linestyle='--', linewidth=2, label=f'平均值: {final_energy:.4f}')
    ax9.set_xlabel('蒙特卡洛步')
    ax9.set_ylabel('局域能量 (Hartree)')
    ax9.set_title('最终计算: 能量演化')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(gs[3, 2:])
    ax10.hist(energy_samples, bins=50, density=True, alpha=0.7, color='blue')
    ax10.axvline(x=final_energy, color='r', linestyle='--', linewidth=2)
    ax10.axvline(x=exact_energy, color='g', linestyle='--', linewidth=2, label='精确值')
    ax10.set_xlabel('能量 (Hartree)')
    ax10.set_ylabel('概率密度')
    ax10.set_title('最终计算: 能量分布')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    plt.savefig('vmc_hydrogen_full_results.png', dpi=150, bbox_inches='tight')
    print("\n完整结果图像已保存为 'vmc_hydrogen_full_results.png'")
    plt.show()
