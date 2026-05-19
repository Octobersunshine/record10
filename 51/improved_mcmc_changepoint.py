import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, nbinom
from scipy.special import loggamma


class ImprovedMCMCChangepointDetection:
    def __init__(self, data, max_changepoints=10, prior_type='regularized_poisson', 
                 penalty_strength=1.0):
        self.data = np.array(data)
        self.n = len(data)
        self.max_changepoints = max_changepoints
        self.prior_type = prior_type
        self.penalty_strength = penalty_strength
        self.samples = []
        self.posterior_probs = None
        self.ncp_posterior = None

    def _log_marginal_likelihood(self, changepoints):
        sorted_cp = sorted([0] + list(changepoints) + [self.n])
        log_marg = 0.0
        
        for i in range(len(sorted_cp) - 1):
            start, end = sorted_cp[i], sorted_cp[i + 1]
            if end <= start:
                continue
            segment = self.data[start:end]
            n_seg = len(segment)
            
            if n_seg < 2:
                return -np.inf
            
            mu0 = 0.0
            kappa0 = 0.01
            alpha0 = 0.01
            beta0 = 0.01
            
            kappa_n = kappa0 + n_seg
            mu_n = (kappa0 * mu0 + n_seg * np.mean(segment)) / kappa_n
            alpha_n = alpha0 + n_seg / 2
            beta_n = beta0 + 0.5 * np.sum((segment - np.mean(segment))**2) + \
                     (kappa0 * n_seg * (np.mean(segment) - mu0)**2) / (2 * kappa_n)
            
            log_marg += -0.5 * n_seg * np.log(2 * np.pi) + \
                        0.5 * np.log(kappa0 / kappa_n) + \
                        alpha0 * np.log(beta0) - alpha_n * np.log(beta_n) + \
                        loggamma(alpha_n) - loggamma(alpha0)
        
        return log_marg

    def _log_prior(self, changepoints):
        k = len(changepoints)
        if k > self.max_changepoints:
            return -np.inf
        
        if self.prior_type == 'regularized_poisson':
            lam = 2.0 / self.penalty_strength
            log_prior = poisson.logpmf(k, mu=lam)
            
            if k > 0:
                log_prior -= k * np.log(self.n - 1)
                
                for cp in changepoints:
                    for other_cp in changepoints:
                        if cp != other_cp:
                            dist = abs(cp - other_cp)
                            if dist < 10:
                                log_prior -= (10 - dist) * 0.1 * self.penalty_strength
        
        elif self.prior_type == 'negative_binomial':
            n = 2
            p = 0.5 / (1 + self.penalty_strength * 0.5)
            log_prior = nbinom.logpmf(k, n=n, p=p)
            
            if k > 0:
                log_prior -= k * np.log(self.n - 1)
        
        elif self.prior_type == 'strongly_regularized':
            lam = 1.0 / self.penalty_strength
            log_prior = poisson.logpmf(k, mu=lam)
            
            if k > 0:
                log_prior -= k * np.log(self.n - 1)
                log_prior -= k * k * 0.5 * self.penalty_strength
        
        else:
            log_prior = poisson.logpmf(k, mu=2)
            if k > 0:
                log_prior -= k * np.log(self.n - 1)
        
        return log_prior

    def _log_posterior(self, changepoints):
        return self._log_marginal_likelihood(changepoints) + self._log_prior(changepoints)

    def _propose(self, changepoints):
        changepoints = set(changepoints)
        rand = np.random.rand()
        
        if rand < 0.4 and len(changepoints) > 0:
            cp = np.random.choice(list(changepoints))
            changepoints.remove(cp)
        elif rand < 0.7:
            new_cp = np.random.randint(1, self.n)
            changepoints.add(new_cp)
        else:
            if len(changepoints) > 0:
                cp = np.random.choice(list(changepoints))
                changepoints.remove(cp)
                delta = np.random.randint(-10, 11)
                new_cp = max(1, min(self.n - 1, cp + delta))
                changepoints.add(new_cp)
        
        return sorted(changepoints)

    def run_mcmc(self, n_iterations=20000, burn_in=5000, thin=10, verbose=True):
        current_cp = []
        current_log_post = self._log_posterior(current_cp)
        
        accept_count = 0
        
        for i in range(n_iterations):
            proposed_cp = self._propose(current_cp)
            proposed_log_post = self._log_posterior(proposed_cp)
            
            log_accept_ratio = proposed_log_post - current_log_post
            
            if np.log(np.random.rand()) < log_accept_ratio:
                current_cp = proposed_cp
                current_log_post = proposed_log_post
                accept_count += 1
            
            if i >= burn_in and i % thin == 0:
                self.samples.append(current_cp.copy())
            
            if verbose and i % 5000 == 0 and i > 0:
                accept_rate = accept_count / i
                print(f"Iteration {i}/{n_iterations}, Acceptance Rate: {accept_rate:.3f}")
        
        if verbose:
            print(f"MCMC completed. Final acceptance rate: {accept_count / n_iterations:.3f}")
            print(f"Collected {len(self.samples)} samples after burn-in and thinning.")
        
        return self.samples

    def compute_posterior_probs(self):
        if not self.samples:
            raise ValueError("Run MCMC first!")
        
        self.posterior_probs = np.zeros(self.n)
        
        for sample in self.samples:
            for cp in sample:
                if 0 <= cp < self.n:
                    self.posterior_probs[cp] += 1
        
        self.posterior_probs /= len(self.samples)
        
        self.ncp_posterior = np.zeros(self.max_changepoints + 1)
        ncp_counts = [len(s) for s in self.samples]
        for ncp in ncp_counts:
            if ncp <= self.max_changepoints:
                self.ncp_posterior[ncp] += 1
        self.ncp_posterior /= len(self.samples)
        
        return self.posterior_probs

    def get_map_estimate(self):
        if not self.samples:
            raise ValueError("Run MCMC first!")
        
        best_sample = None
        best_log_post = -np.inf
        
        for sample in self.samples:
            log_post = self._log_posterior(sample)
            if log_post > best_log_post:
                best_log_post = log_post
                best_sample = sample
        
        return sorted(best_sample)

    def get_bma_estimate(self, threshold=0.5):
        if self.posterior_probs is None:
            self.compute_posterior_probs()
        
        bma_cp = [i for i, p in enumerate(self.posterior_probs) if p >= threshold]
        return sorted(bma_cp)

    def get_model_averaged_changepoints(self, n_models=3):
        if self.ncp_posterior is None:
            self.compute_posterior_probs()
        
        top_ncp = np.argsort(self.ncp_posterior)[-n_models:][::-1]
        
        results = []
        for ncp in top_ncp:
            if self.ncp_posterior[ncp] > 0.01:
                samples_with_ncp = [s for s in self.samples if len(s) == ncp]
                if samples_with_ncp:
                    probs_ncp = np.zeros(self.n)
                    for s in samples_with_ncp:
                        for cp in s:
                            if 0 <= cp < self.n:
                                probs_ncp[cp] += 1
                    probs_ncp /= len(samples_with_ncp)
                    results.append({
                        'n_changepoints': ncp,
                        'posterior_prob': self.ncp_posterior[ncp],
                        'changepoint_probs': probs_ncp
                    })
        
        return results

    def plot_results(self, figsize=(14, 12)):
        if self.posterior_probs is None:
            self.compute_posterior_probs()
        
        map_cp = self.get_map_estimate()
        bma_cp = self.get_bma_estimate(threshold=0.5)
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 2, 1.5, 1.5])
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.data, 'b-', label='Data', linewidth=1)
        for i, cp in enumerate(map_cp):
            ax1.axvline(x=cp, color='g', linestyle='--', linewidth=2, alpha=0.7, 
                       label='MAP Changepoint' if i == 0 else "")
        for i, cp in enumerate(bma_cp):
            if cp not in map_cp:
                ax1.axvline(x=cp, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                           label='BMA Changepoint (p>=0.5)' if i == 0 else "")
        ax1.set_ylabel('Value')
        ax1.set_title('Time Series with Changepoint Estimates')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(self.posterior_probs, 'r-', linewidth=2, label='Posterior Probability')
        ax2.fill_between(range(self.n), self.posterior_probs, alpha=0.3, color='red')
        ax2.axhline(y=0.5, color='k', linestyle=':', alpha=0.5, label='Threshold (0.5)')
        ax2.set_ylabel('Probability')
        ax2.set_title('Posterior Probability of Changepoint (Bayesian Model Averaged)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        ax3 = fig.add_subplot(gs[2])
        ncp_range = range(len(self.ncp_posterior))
        ax3.bar(ncp_range, self.ncp_posterior, color='purple', alpha=0.7, width=0.6)
        for i, prob in enumerate(self.ncp_posterior):
            if prob > 0.01:
                ax3.text(i, prob + 0.01, f'{prob:.2f}', ha='center', fontsize=9)
        ax3.set_xlabel('Number of Changepoints')
        ax3.set_ylabel('Posterior Probability')
        ax3.set_title('Posterior Distribution of Number of Changepoints')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(ncp_range)
        
        ax4 = fig.add_subplot(gs[3])
        models = self.get_model_averaged_changepoints(n_models=3)
        for i, model in enumerate(models):
            ax4.plot(model['changepoint_probs'], 
                    label=f'{model["n_changepoints"]} changepoints (p={model["posterior_prob"]:.2f})',
                    linewidth=2, alpha=0.8)
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Probability')
        ax4.set_title('Changepoint Probabilities by Model Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def print_summary(self):
        if self.posterior_probs is None:
            self.compute_posterior_probs()
        
        print("\n" + "="*70)
        print("贝叶斯变点检测结果摘要")
        print("="*70)
        
        print(f"\n数据长度: {self.n}")
        print(f"MCMC样本数: {len(self.samples)}")
        print(f"先验类型: {self.prior_type}")
        print(f"惩罚强度: {self.penalty_strength}")
        
        print("\n--- 变点数量后验分布 ---")
        for ncp in range(len(self.ncp_posterior)):
            if self.ncp_posterior[ncp] > 0.01:
                print(f"  {ncp} 个变点: 概率 = {self.ncp_posterior[ncp]:.4f}")
        
        print("\n--- MAP估计 ---")
        map_cp = self.get_map_estimate()
        print(f"  变点位置: {map_cp}")
        print(f"  变点数量: {len(map_cp)}")
        
        print("\n--- 贝叶斯模型平均 (BMA) ---")
        print("  高概率变点位置 (p >= 0.5):")
        bma_cp = self.get_bma_estimate(threshold=0.5)
        for cp in bma_cp:
            print(f"    位置 {cp}: 概率 = {self.posterior_probs[cp]:.4f}")
        
        print("\n--- 所有高概率变点 (p >= 0.3) ---")
        high_prob = [(i, p) for i, p in enumerate(self.posterior_probs) if p >= 0.3]
        for pos, prob in sorted(high_prob, key=lambda x: -x[1]):
            print(f"  位置 {pos}: 概率 = {prob:.4f}")
        
        print("\n" + "="*70)


def generate_test_data(n=200, changepoints=[50, 120], means=[0, 5, 2], stds=[1, 1.5, 0.8]):
    data = np.zeros(n)
    segments = [0] + changepoints + [n]
    
    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        data[start:end] = np.random.normal(means[i], stds[i], end - start)
    
    return data


if __name__ == '__main__':
    np.random.seed(42)
    
    data = generate_test_data(n=200, changepoints=[60, 140], means=[0, 6, 2], stds=[1, 1.5, 0.8])
    
    print("运行改进的MCMC贝叶斯变点检测...")
    print("使用正则化Poisson先验 + 边际似然计算")
    
    bcd = ImprovedMCMCChangepointDetection(
        data, 
        max_changepoints=8,
        prior_type='regularized_poisson',
        penalty_strength=2.0
    )
    
    bcd.run_mcmc(n_iterations=25000, burn_in=5000, thin=10)
    
    bcd.compute_posterior_probs()
    bcd.print_summary()
    
    bcd.plot_results()
