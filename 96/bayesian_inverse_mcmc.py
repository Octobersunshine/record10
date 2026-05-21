import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


def solve_heat_direct_cn(L, T_total, alpha, Nx, Nt, q, T_left, T_right):
    dx = L / (Nx - 1)
    dt = T_total / (Nt - 1)
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T_total, Nt)
    T = np.zeros((Nx, Nt))
    T[:, 0] = T_left + (T_right - T_left) * x / L
    Fo = alpha * dt / dx**2
    
    main_diag_A = np.ones(Nx) * (1 + Fo)
    upper_diag_A = np.ones(Nx - 1) * (-Fo/2)
    lower_diag_A = np.ones(Nx - 1) * (-Fo/2)
    main_diag_A[0] = 1 + 2*Fo
    main_diag_A[-1] = 1
    upper_diag_A[0] = -Fo
    lower_diag_A[-1] = 0
    
    main_diag_B = np.ones(Nx) * (1 - Fo)
    upper_diag_B = np.ones(Nx - 1) * (Fo/2)
    lower_diag_B = np.ones(Nx - 1) * (Fo/2)
    main_diag_B[0] = 1 - 2*Fo
    main_diag_B[-1] = 1
    upper_diag_B[0] = Fo
    lower_diag_B[-1] = 0
    
    A = diags([lower_diag_A, main_diag_A, upper_diag_A], [-1, 0, 1], format='csr')
    B = diags([lower_diag_B, main_diag_B, upper_diag_B], [-1, 0, 1], format='csr')
    
    for n in range(Nt - 1):
        b = B.dot(T[:, n])
        b[0] += 2 * Fo * dx * (q[n] + q[n+1]) / alpha
        b[-1] = 2 * T_right
        T[:, n + 1] = spsolve(A, b)
    
    return x, t, T


def compute_likelihood(q, L, T_total, alpha, Nx, Nt, T_measured, x_measured, t_measured, sigma_meas):
    x, t, T_pred = solve_heat_direct_cn(L, T_total, alpha, Nx, Nt, q, 20, 100)
    
    residual = 0.0
    for i, xi in enumerate(x_measured):
        idx_x = np.argmin(np.abs(x - xi))
        for j, tj in enumerate(t_measured):
            idx_t = np.argmin(np.abs(t - tj))
            residual += (T_pred[idx_x, idx_t] - T_measured[i, j])**2
    
    log_likelihood = -0.5 * residual / (sigma_meas**2)
    return log_likelihood


def compute_prior(q, prior_mean, prior_std):
    log_prior = np.sum(norm.logpdf(q, prior_mean, prior_std))
    return log_prior


def compute_posterior(q, L, T_total, alpha, Nx, Nt, T_measured, x_measured, t_measured, 
                      sigma_meas, prior_mean, prior_std):
    log_likelihood = compute_likelihood(q, L, T_total, alpha, Nx, Nt, 
                                        T_measured, x_measured, t_measured, sigma_meas)
    log_prior = compute_prior(q, prior_mean, prior_std)
    return log_likelihood + log_prior


class MCMCSampler:
    def __init__(self, log_posterior_func, dim, proposal_scale=1.0):
        self.log_posterior_func = log_posterior_func
        self.dim = dim
        self.proposal_scale = proposal_scale
        self.samples = []
        self.log_posteriors = []
        self.acceptance_rate = 0.0
        
    def propose(self, current):
        proposal = current + np.random.normal(0, self.proposal_scale, self.dim)
        return proposal
    
    def sample(self, initial_state, n_samples, burn_in=0, thin=1, adapt=True):
        current = initial_state.copy()
        current_log_posterior = self.log_posterior_func(current)
        
        samples = []
        log_posteriors = []
        n_accepted = 0
        
        print(f"\n开始MCMC采样...")
        print(f"参数维度: {self.dim}")
        print(f"总采样数: {n_samples}")
        print(f"燃烧期: {burn_in}")
        print(f"抽样间隔: {thin}")
        
        for i in range(n_samples):
            proposal = self.propose(current)
            proposal_log_posterior = self.log_posterior_func(proposal)
            
            log_alpha = proposal_log_posterior - current_log_posterior
            
            if np.log(np.random.uniform()) < log_alpha:
                current = proposal
                current_log_posterior = proposal_log_posterior
                n_accepted += 1
            
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(current.copy())
                log_posteriors.append(current_log_posterior)
            
            if adapt and i < burn_in and i > 0 and i % 50 == 0:
                current_rate = n_accepted / (i + 1)
                if current_rate < 0.2:
                    self.proposal_scale *= 0.8
                elif current_rate > 0.5:
                    self.proposal_scale *= 1.2
            
            if i % 100 == 0:
                current_rate = n_accepted / (i + 1) if i > 0 else 0
                print(f"  迭代 {i:5d}/{n_samples}, 接受率: {current_rate:.3f}, "
                      f"建议尺度: {self.proposal_scale:.4f}", end='\r')
        
        self.samples = np.array(samples)
        self.log_posteriors = np.array(log_posteriors)
        self.acceptance_rate = n_accepted / n_samples
        
        print(f"\n\n采样完成!")
        print(f"最终接受率: {self.acceptance_rate:.3f}")
        print(f"保留样本数: {len(self.samples)}")
        
        return self.samples, self.log_posteriors
    
    def get_stats(self):
        if len(self.samples) == 0:
            return None
        
        mean = np.mean(self.samples, axis=0)
        std = np.std(self.samples, axis=0)
        ci_low = np.percentile(self.samples, 2.5, axis=0)
        ci_high = np.percentile(self.samples, 97.5, axis=0)
        
        return {
            'mean': mean,
            'std': std,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'median': np.median(self.samples, axis=0),
            'q25': np.percentile(self.samples, 25, axis=0),
            'q75': np.percentile(self.samples, 75, axis=0)
        }
    
    def geweke_test(self, first_frac=0.1, last_frac=0.5):
        if len(self.samples) < 100:
            return None
        
        n = len(self.samples)
        n_first = int(n * first_frac)
        n_last = int(n * last_frac)
        
        z_scores = []
        for i in range(self.dim):
            mean1 = np.mean(self.samples[:n_first, i])
            mean2 = np.mean(self.samples[-n_last:, i])
            var1 = np.var(self.samples[:n_first, i])
            var2 = np.var(self.samples[-n_last:, i])
            
            z = (mean1 - mean2) / np.sqrt(var1/n_first + var2/n_last)
            z_scores.append(z)
        
        return np.array(z_scores)


class DRAMSampler(MCMCSampler):
    def __init__(self, log_posterior_func, dim, proposal_scale=1.0, n_stages=2):
        super().__init__(log_posterior_func, dim, proposal_scale)
        self.n_stages = n_stages
        self.scales = [proposal_scale * (0.5**i) for i in range(n_stages)]
    
    def propose(self, current, stage=0):
        scale = self.scales[stage]
        proposal = current + np.random.normal(0, scale, self.dim)
        return proposal
    
    def sample(self, initial_state, n_samples, burn_in=0, thin=1, adapt=True):
        current = initial_state.copy()
        current_log_posterior = self.log_posterior_func(current)
        
        samples = []
        log_posteriors = []
        n_accepted = 0
        
        print(f"\n开始DRAM采样（延迟拒绝自适应Metropolis）...")
        print(f"参数维度: {self.dim}")
        print(f"总采样数: {n_samples}")
        print(f"燃烧期: {burn_in}")
        print(f"延迟拒绝阶段数: {self.n_stages}")
        
        for i in range(n_samples):
            accepted = False
            proposal = self.propose(current, stage=0)
            proposal_log_posterior = self.log_posterior_func(proposal)
            
            log_alpha1 = proposal_log_posterior - current_log_posterior
            
            if np.log(np.random.uniform()) < log_alpha1:
                current = proposal
                current_log_posterior = proposal_log_posterior
                accepted = True
            else:
                for stage in range(1, self.n_stages):
                    proposal2 = self.propose(current, stage=stage)
                    proposal2_log_posterior = self.log_posterior_func(proposal2)
                    
                    log_alpha2_stage = proposal2_log_posterior - current_log_posterior
                    
                    if np.log(np.random.uniform()) < log_alpha2_stage:
                        current = proposal2
                        current_log_posterior = proposal2_log_posterior
                        accepted = True
                        break
            
            if accepted:
                n_accepted += 1
            
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(current.copy())
                log_posteriors.append(current_log_posterior)
            
            if adapt and i < burn_in and i > 0 and i % 50 == 0:
                current_rate = n_accepted / (i + 1)
                if current_rate < 0.15:
                    for s in range(self.n_stages):
                        self.scales[s] *= 0.7
                elif current_rate > 0.6:
                    for s in range(self.n_stages):
                        self.scales[s] *= 1.3
            
            if i % 100 == 0:
                current_rate = n_accepted / (i + 1) if i > 0 else 0
                print(f"  迭代 {i:5d}/{n_samples}, 接受率: {current_rate:.3f}, "
                      f"主建议尺度: {self.scales[0]:.4f}", end='\r')
        
        self.samples = np.array(samples)
        self.log_posteriors = np.array(log_posteriors)
        self.acceptance_rate = n_accepted / n_samples
        
        print(f"\n\n采样完成!")
        print(f"最终接受率: {self.acceptance_rate:.3f}")
        print(f"保留样本数: {len(self.samples)}")
        
        return self.samples, self.log_posteriors


def generate_test_data(Nx_ref=100, Nt_ref=200):
    L = 1.0
    T_total = 5.0
    alpha = 0.01
    
    t = np.linspace(0, T_total, Nt_ref)
    q_true = 50 + 20 * np.sin(2 * np.pi * t / T_total) + 10 * np.sin(4 * np.pi * t / T_total)
    
    x_ref, t_ref, T_true_ref = solve_heat_direct_cn(L, T_total, alpha, Nx_ref, Nt_ref, q_true, 20, 80)
    
    x_measured = np.array([0.3, 0.5, 0.7])
    t_measured_idx = np.arange(0, Nt_ref, 10)
    t_measured = t_ref[t_measured_idx]
    
    T_measured = np.zeros((len(x_measured), len(t_measured)))
    for i, xi in enumerate(x_measured):
        idx_x = np.argmin(np.abs(x_ref - xi))
        for j, tj_idx in enumerate(t_measured_idx):
            T_measured[i, j] = T_true_ref[idx_x, tj_idx]
    
    noise_level = 0.02
    sigma_meas = noise_level * np.std(T_measured)
    T_measured += sigma_meas * np.random.randn(*T_measured.shape)
    
    return L, T_total, alpha, x_measured, t_measured, T_measured, q_true, t, sigma_meas


def run_bayesian_inversion():
    print("=" * 70)
    print("贝叶斯热传导反问题 - MCMC不确定性分析")
    print("=" * 70)
    
    print("\n1. 生成测试数据...")
    L, T_total, alpha, x_measured, t_measured, T_measured, q_true, t, sigma_meas = generate_test_data()
    
    print(f"   空间域: [0, {L}] m")
    print(f"   时间域: [0, {T_total}] s")
    print(f"   测量噪声标准差: {sigma_meas:.4f}")
    print(f"   测量位置: {x_measured} m")
    
    Nx = 50
    Nt = 100
    t_solve = np.linspace(0, T_total, Nt)
    
    print(f"\n2. 设置贝叶斯反问题...")
    print(f"   求解网格: {Nx}x{Nt}")
    
    q_interp = interp1d(t, q_true, kind='linear')
    q_true_solve = q_interp(t_solve)
    
    prior_mean = 50.0
    prior_std = 20.0
    print(f"   先验分布: N({prior_mean}, {prior_std}²)")
    
    def log_posterior(q):
        return compute_posterior(q, L, T_total, alpha, Nx, Nt, 
                                 T_measured, x_measured, t_measured,
                                 sigma_meas, prior_mean, prior_std)
    
    print("\n3. 初始化MCMC采样器...")
    initial_q = prior_mean * np.ones(Nt)
    sampler = DRAMSampler(log_posterior, dim=Nt, proposal_scale=5.0, n_stages=2)
    
    n_samples = 2000
    burn_in = 1000
    thin = 2
    
    samples, log_posteriors = sampler.sample(initial_q, n_samples, burn_in=burn_in, thin=thin, adapt=True)
    
    print("\n4. 后验统计分析...")
    stats = sampler.get_stats()
    
    q_mean = stats['mean']
    q_std = stats['std']
    q_ci_low = stats['ci_low']
    q_ci_high = stats['ci_high']
    
    rmse_mean = np.sqrt(np.mean((q_mean - q_true_solve)**2))
    coverage = np.mean((q_true_solve >= q_ci_low) & (q_true_solve <= q_ci_high))
    
    print(f"   后验均值RMSE: {rmse_mean:.4f} W/m²")
    print(f"   95%置信区间覆盖率: {coverage:.2%}")
    print(f"   平均后验标准差: {np.mean(q_std):.4f} W/m²")
    
    z_scores = sampler.geweke_test()
    if z_scores is not None:
        max_z = np.max(np.abs(z_scores))
        print(f"   Geweke收敛检验 max|z|: {max_z:.4f}")
    
    print("\n5. 生成可视化结果...")
    fig = plt.figure(figsize=(18, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t_solve, q_true_solve, 'k-', linewidth=2.5, label='精确热流')
    ax1.plot(t_solve, q_mean, 'r-', linewidth=2, label='后验均值')
    ax1.fill_between(t_solve, q_ci_low, q_ci_high, alpha=0.3, color='r', label='95%置信区间')
    ax1.fill_between(t_solve, stats['q25'], stats['q75'], alpha=0.5, color='r', label='50%置信区间')
    ax1.set_xlabel('时间 (s)', fontsize=11)
    ax1.set_ylabel('热流 (W/m²)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title('热流后验估计与不确定性区间', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(log_posteriors, 'b-', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('样本索引', fontsize=11)
    ax2.set_ylabel('对数后验概率', fontsize=11)
    ax2.set_title('后验概率轨迹（燃烧期后）', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(samples[:, 25], 'b-', linewidth=0.5, alpha=0.7)
    ax3.axhline(y=q_true_solve[25], color='r', linestyle='--', linewidth=2, label='真值')
    ax3.set_xlabel('样本索引', fontsize=11)
    ax3.set_ylabel('热流值 (W/m²)', fontsize=11)
    ax3.set_title(f'参数轨迹 (t={t_solve[25]:.2f}s)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 3, 4)
    time_idx = [10, 25, 50, 75]
    for idx in time_idx:
        ax4.hist(samples[:, idx], bins=30, alpha=0.6, density=True, 
                 label=f't={t_solve[idx]:.1f}s')
        ax4.axvline(x=q_true_solve[idx], color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('热流值 (W/m²)', fontsize=11)
    ax4.set_ylabel('概率密度', fontsize=11)
    ax4.set_title('边际后验分布', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t_solve, q_std, 'g-', linewidth=2)
    ax5.fill_between(t_solve, 0, q_std, alpha=0.3, color='g')
    ax5.set_xlabel('时间 (s)', fontsize=11)
    ax5.set_ylabel('后验标准差 (W/m²)', fontsize=11)
    ax5.set_title('后验不确定性随时间变化', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    n_random = min(50, len(samples))
    random_idx = np.random.choice(len(samples), n_random, replace=False)
    for idx in random_idx:
        ax6.plot(t_solve, samples[idx], 'b-', linewidth=0.5, alpha=0.3)
    ax6.plot(t_solve, q_true_solve, 'r-', linewidth=2, label='真值')
    ax6.set_xlabel('时间 (s)', fontsize=11)
    ax6.set_ylabel('热流 (W/m²)', fontsize=11)
    ax6.set_title(f'{n_random}个后验样本', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_mcmc_results.png', dpi=150, bbox_inches='tight')
    print("\n6. 结果已保存至 'bayesian_mcmc_results.png'")
    
    fig2 = plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(samples.T)
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, 
                    extent=[0, T_total, T_total, 0])
    plt.xlabel('时间 (s)', fontsize=12)
    plt.ylabel('时间 (s)', fontsize=12)
    plt.title('后验样本相关系数矩阵', fontsize=14, fontweight='bold')
    plt.colorbar(im, label='相关系数')
    plt.tight_layout()
    plt.savefig('posterior_correlation.png', dpi=150, bbox_inches='tight')
    print("   相关系数矩阵已保存至 'posterior_correlation.png'")
    
    print("\n" + "=" * 70)
    print("贝叶斯反演完成！")
    print("=" * 70)
    
    return sampler, stats


if __name__ == "__main__":
    sampler, stats = run_bayesian_inversion()
