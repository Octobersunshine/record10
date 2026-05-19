import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, poisson


class MCMCChangepointDetection:
    def __init__(self, data, max_changepoints=10):
        self.data = np.array(data)
        self.n = len(data)
        self.max_changepoints = max_changepoints
        self.samples = []
        self.posterior_probs = None

    def _log_likelihood(self, changepoints):
        sorted_cp = sorted([0] + list(changepoints) + [self.n])
        log_like = 0.0
        
        for i in range(len(sorted_cp) - 1):
            start, end = sorted_cp[i], sorted_cp[i + 1]
            if end <= start:
                continue
            segment = self.data[start:end]
            n_seg = len(segment)
            
            mu = np.mean(segment)
            sigma2 = np.var(segment) if n_seg > 1 else 1.0
            sigma2 = max(sigma2, 1e-6)
            
            log_like += np.sum(norm.logpdf(segment, loc=mu, scale=np.sqrt(sigma2)))
        
        return log_like

    def _log_prior(self, changepoints):
        k = len(changepoints)
        if k > self.max_changepoints:
            return -np.inf
        
        log_prior = poisson.logpmf(k, mu=2)
        
        if k > 0:
            log_prior -= k * np.log(self.n - 1)
        
        return log_prior

    def _log_posterior(self, changepoints):
        return self._log_likelihood(changepoints) + self._log_prior(changepoints)

    def _propose(self, changepoints):
        changepoints = set(changepoints)
        rand = np.random.rand()
        
        if rand < 0.33 and len(changepoints) > 0:
            cp = np.random.choice(list(changepoints))
            changepoints.remove(cp)
        elif rand < 0.66:
            new_cp = np.random.randint(1, self.n)
            changepoints.add(new_cp)
        else:
            if len(changepoints) > 0:
                cp = np.random.choice(list(changepoints))
                changepoints.remove(cp)
                new_cp = np.random.randint(1, self.n)
                changepoints.add(new_cp)
        
        return sorted(changepoints)

    def run_mcmc(self, n_iterations=10000, burn_in=2000, thin=10):
        current_cp = []
        current_log_post = self._log_posterior(current_cp)
        
        for i in range(n_iterations):
            proposed_cp = self._propose(current_cp)
            proposed_log_post = self._log_posterior(proposed_cp)
            
            log_accept_ratio = proposed_log_post - current_log_post
            
            if np.log(np.random.rand()) < log_accept_ratio:
                current_cp = proposed_cp
                current_log_post = proposed_log_post
            
            if i >= burn_in and i % thin == 0:
                self.samples.append(current_cp.copy())
        
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

    def plot_results(self, figsize=(12, 10)):
        if self.posterior_probs is None:
            self.compute_posterior_probs()
        
        map_cp = self.get_map_estimate()
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        axes[0].plot(self.data, 'b-', label='Data', linewidth=1)
        for cp in map_cp:
            axes[0].axvline(x=cp, color='g', linestyle='--', linewidth=2, alpha=0.7, label='MAP Changepoint' if cp == map_cp[0] else "")
        axes[0].set_ylabel('Value')
        axes[0].set_title('Time Series with MAP Changepoints')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.posterior_probs, 'r-', linewidth=2, label='Posterior Probability')
        axes[1].fill_between(range(self.n), self.posterior_probs, alpha=0.3, color='red')
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Posterior Probability of Changepoint at Each Position')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        
        n_cp_dist = [len(s) for s in self.samples]
        max_n = max(n_cp_dist) if n_cp_dist else 0
        counts = np.bincount(n_cp_dist, minlength=max_n + 1)
        probs = counts / len(self.samples)
        axes[2].bar(range(len(probs)), probs, color='purple', alpha=0.7)
        axes[2].set_xlabel('Number of Changepoints')
        axes[2].set_ylabel('Probability')
        axes[2].set_title('Posterior Distribution of Number of Changepoints')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


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
    
    print("Running MCMC changepoint detection...")
    mcmc_bcd = MCMCChangepointDetection(data, max_changepoints=10)
    samples = mcmc_bcd.run_mcmc(n_iterations=20000, burn_in=5000, thin=10)
    
    posterior_probs = mcmc_bcd.compute_posterior_probs()
    map_cp = mcmc_bcd.get_map_estimate()
    
    print(f"\nMAP estimate of changepoints: {map_cp}")
    print(f"\nTop 10 most probable changepoint positions:")
    top_indices = np.argsort(posterior_probs)[-10:][::-1]
    for idx in top_indices:
        print(f"Position {idx}: Probability = {posterior_probs[idx]:.4f}")
    
    mcmc_bcd.plot_results()
