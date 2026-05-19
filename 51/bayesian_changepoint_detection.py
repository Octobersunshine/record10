import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma


class BayesianChangepointDetection:
    def __init__(self, data, hazard=0.01, mu0=0, kappa0=1, alpha0=1, beta0=1):
        self.data = np.array(data)
        self.n = len(data)
        self.hazard = hazard
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.posterior_probs = None

    def _pred_prob(self, t, s):
        if s >= t:
            return 0.0
        
        segment_data = self.data[s:t]
        n = len(segment_data)
        
        if n == 0:
            return 0.0
        
        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * np.mean(segment_data)) / kappa_n
        alpha_n = self.alpha0 + n / 2
        beta_n = self.beta0 + 0.5 * np.sum((segment_data - np.mean(segment_data))**2) + \
                 (self.kappa0 * n * (np.mean(segment_data) - self.mu0)**2) / (2 * kappa_n)
        
        df = 2 * alpha_n
        scale = np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))
        
        return norm.logpdf(self.data[t], loc=mu_n, scale=scale)

    def offline_changepoint_detection(self):
        log_R = np.zeros((self.n + 1, self.n + 1))
        log_R[0, 0] = 0.0
        
        for t in range(1, self.n + 1):
            log_p = np.zeros(t)
            
            for s in range(t):
                log_p[s] = self._pred_prob(t - 1, s)
            
            log_growth = log_R[0:t, t - 1] + log_p + np.log(1 - self.hazard)
            log_sum = log_R[0:t, t - 1] + log_p
            
            max_log_sum = np.max(log_sum)
            log_sum_norm = log_sum - max_log_sum
            log_cp = np.log(self.hazard) + max_log_sum + np.log(np.sum(np.exp(log_sum_norm)))
            
            log_R[0, t] = log_cp
            log_R[1:t + 1, t] = log_growth
            
            max_log_R = np.max(log_R[:, t])
            log_R[:, t] = log_R[:, t] - max_log_R - np.log(np.sum(np.exp(log_R[:, t] - max_log_R)))
        
        R = np.exp(log_R)
        self.posterior_probs = np.zeros(self.n)
        
        for t in range(self.n):
            self.posterior_probs[t] = np.sum(R[t, t:self.n])
        
        return self.posterior_probs

    def plot_results(self, figsize=(12, 8)):
        if self.posterior_probs is None:
            self.offline_changepoint_detection()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        ax1.plot(self.data, 'b-', label='Data')
        ax1.set_ylabel('Value')
        ax1.set_title('Time Series Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.posterior_probs, 'r-', linewidth=2, label='Changepoint Probability')
        ax2.fill_between(range(self.n), self.posterior_probs, alpha=0.3, color='red')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.set_title('Posterior Probability of Changepoint')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
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
    
    bcd = BayesianChangepointDetection(data, hazard=0.01)
    posterior_probs = bcd.offline_changepoint_detection()
    
    print("Top 10 most probable changepoint positions:")
    top_indices = np.argsort(posterior_probs)[-10:][::-1]
    for idx in top_indices:
        print(f"Position {idx}: Probability = {posterior_probs[idx]:.4f}")
    
    bcd.plot_results()
