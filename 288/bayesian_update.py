from dataclasses import dataclass, field
import math
from typing import Tuple, Optional, Callable, List, Dict, Any
import numpy as np
import emcee


_SMOOTH_FLOOR = 1e-6
_JEFFREYS_ALPHA = 0.5
_JEFFREYS_BETA = 0.5


@dataclass
class BetaBinomial:
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    prior_correction: str = "smooth"
    corrected: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.prior_correction not in ("smooth", "jeffreys"):
            raise ValueError(f"prior_correction must be 'smooth' or 'jeffreys', got '{self.prior_correction}'")

        is_degenerate = self.alpha_prior <= 0 or self.beta_prior <= 0
        if not is_degenerate:
            return

        self.corrected = True

        if self.prior_correction == "smooth":
            self.alpha_prior = max(self.alpha_prior, _SMOOTH_FLOOR)
            self.beta_prior = max(self.beta_prior, _SMOOTH_FLOOR)
        else:
            self.alpha_prior = _JEFFREYS_ALPHA
            self.beta_prior = _JEFFREYS_BETA

    def update(self, successes: int, trials: int) -> Tuple[float, float]:
        if trials < 0 or successes < 0 or successes > trials:
            raise ValueError("Invalid input: trials >= 0 and 0 <= successes <= trials")
        
        alpha_posterior = self.alpha_prior + successes
        beta_posterior = self.beta_prior + (trials - successes)
        
        return alpha_posterior, beta_posterior

    def posterior_predictive(self, successes: int, trials: int, 
                           new_successes: int, new_trials: int) -> float:
        alpha_post, beta_post = self.update(successes, trials)
        
        def beta_func(a, b):
            return math.gamma(a) * math.gamma(b) / math.gamma(a + b)
        
        numerator = math.comb(new_trials, new_successes)
        numerator *= beta_func(alpha_post + new_successes, beta_post + new_trials - new_successes)
        denominator = beta_func(alpha_post, beta_post)
        
        return numerator / denominator

    def posterior_mean(self, successes: int, trials: int) -> float:
        alpha_post, beta_post = self.update(successes, trials)
        return alpha_post / (alpha_post + beta_post)

    def posterior_variance(self, successes: int, trials: int) -> float:
        alpha_post, beta_post = self.update(successes, trials)
        return (alpha_post * beta_post) / ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))


@dataclass
class NormalNormal:
    mu_prior: float = 0.0
    tau_prior: float = 1.0
    tau_likelihood: float = 1.0

    def update(self, data: list) -> Tuple[float, float]:
        if not data:
            return self.mu_prior, self.tau_prior
        
        n = len(data)
        sample_mean = sum(data) / n
        
        tau_posterior = self.tau_prior + n * self.tau_likelihood
        mu_posterior = (self.tau_prior * self.mu_prior + n * self.tau_likelihood * sample_mean) / tau_posterior
        
        return mu_posterior, tau_posterior

    def posterior_predictive(self, data: list, new_x: float) -> float:
        mu_post, tau_post = self.update(data)
        
        tau_pred = 1 / (1 / tau_post + 1 / self.tau_likelihood)
        mu_pred = mu_post
        
        return (1 / math.sqrt(2 * math.pi / tau_pred)) * math.exp(-0.5 * tau_pred * (new_x - mu_pred) ** 2)

    def posterior_mean(self, data: list) -> float:
        mu_post, _ = self.update(data)
        return mu_post

    def posterior_variance(self, data: list) -> float:
        _, tau_post = self.update(data)
        return 1 / tau_post


def compute_hpd(samples, cred_mass: float = 0.95) -> np.ndarray:
    if not 0 < cred_mass < 1:
        raise ValueError("cred_mass must be between 0 and 1")
    
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    n_dim = samples.shape[1]
    hpd_intervals = np.zeros((n_dim, 2))
    
    for d in range(n_dim):
        dim_samples = np.sort(samples[:, d])
        n = len(dim_samples)
        n_cred = int(np.floor(cred_mass * n))
        
        if n_cred == 0:
            hpd_intervals[d] = [dim_samples[0], dim_samples[-1]]
            continue
        
        interval_widths = dim_samples[n_cred:] - dim_samples[:n - n_cred]
        min_idx = np.argmin(interval_widths)
        
        hpd_intervals[d] = [dim_samples[min_idx], dim_samples[min_idx + n_cred]]
    
    return hpd_intervals


@dataclass
class MCMCSampler:
    log_prior: Callable[[np.ndarray], float]
    log_likelihood: Callable[[np.ndarray, Any], float]
    ndim: int
    data: Any = None

    def log_posterior(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta)
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(theta, self.data)
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll

    def sample(self, 
               n_walkers: int = 32,
               n_steps: int = 5000,
               n_burn: int = 1000,
               initial_guess: Optional[np.ndarray] = None,
               initial_scale: float = 0.1,
               progress: bool = False) -> Dict[str, Any]:
        
        if n_walkers <= 2 * self.ndim:
            raise ValueError(f"n_walkers must be > 2 * ndim ({2 * self.ndim})")
        
        if initial_guess is None:
            initial_guess = np.zeros(self.ndim)
        
        initial_guess = np.asarray(initial_guess)
        if initial_guess.shape != (self.ndim,):
            raise ValueError(f"initial_guess must have shape ({self.ndim},)")
        
        p0 = initial_guess + initial_scale * np.random.randn(n_walkers, self.ndim)
        
        sampler = emcee.EnsembleSampler(n_walkers, self.ndim, self.log_posterior)
        sampler.run_mcmc(p0, n_steps, progress=progress)
        
        chain = sampler.get_chain(discard=n_burn, flat=True)
        log_prob = sampler.get_log_prob(discard=n_burn, flat=True)
        
        return {
            "samples": chain,
            "log_prob": log_prob,
            "acceptance_rate": np.mean(sampler.acceptance_fraction),
            "autocorr_time": sampler.get_autocorr_time(tol=0) if sampler.get_autocorr_time(tol=0).size > 0 else None,
            "n_walkers": n_walkers,
            "n_steps": n_steps,
            "n_burn": n_burn,
            "ndim": self.ndim
        }

    def compute_statistics(self, samples: np.ndarray, 
                          cred_mass: float = 0.95,
                          param_names: Optional[List[str]] = None) -> Dict[str, Any]:
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        
        if param_names is None:
            param_names = [f"param_{i}" for i in range(samples.shape[1])]
        
        if len(param_names) != samples.shape[1]:
            raise ValueError("param_names length must match number of dimensions")
        
        hpd_intervals = compute_hpd(samples, cred_mass)
        
        stats = {}
        for i, name in enumerate(param_names):
            dim_samples = samples[:, i]
            stats[name] = {
                "mean": float(np.mean(dim_samples)),
                "median": float(np.median(dim_samples)),
                "std": float(np.std(dim_samples)),
                "hpd_low": float(hpd_intervals[i, 0]),
                "hpd_high": float(hpd_intervals[i, 1]),
                "cred_mass": cred_mass
            }
        
        return stats


def demo_mcmc_beta_binomial():
    print("\n=== MCMC: Beta-Binomial (非共轭先验演示) ===")
    
    successes = 7
    trials = 10
    
    def log_prior(theta):
        p = theta[0]
        if 0 < p < 1:
            return 0.0
        return -np.inf
    
    def log_likelihood(theta, data):
        p = theta[0]
        successes, trials = data
        if 0 < p < 1:
            return successes * np.log(p) + (trials - successes) * np.log(1 - p)
        return -np.inf
    
    sampler = MCMCSampler(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        ndim=1,
        data=(successes, trials)
    )
    
    print(f"观测数据: {successes}次成功 / {trials}次试验")
    print("先验: Uniform(0, 1) (非共轭)")
    print("\n正在运行 MCMC 采样...")
    
    result = sampler.sample(
        n_walkers=50,
        n_steps=10000,
        n_burn=2000,
        initial_guess=np.array([0.5]),
        initial_scale=0.1
    )
    
    print(f"\n采样完成:")
    print(f"  样本数量: {len(result['samples'])}")
    print(f"  平均接受率: {result['acceptance_rate']:.4f}")
    
    stats = sampler.compute_statistics(
        result['samples'],
        cred_mass=0.95,
        param_names=["p"]
    )
    
    print("\n后验统计量:")
    for name, s in stats.items():
        print(f"  {name}:")
        print(f"    均值: {s['mean']:.4f}")
        print(f"    中位数: {s['median']:.4f}")
        print(f"    标准差: {s['std']:.4f}")
        print(f"    95% HPD 区间: [{s['hpd_low']:.4f}, {s['hpd_high']:.4f}]")
    
    analytical_mean = (successes + 1) / (trials + 2)
    print(f"\n解析解均值 (共轭 Beta(1,1)先验): {analytical_mean:.4f}")


def demo_mcmc_normal():
    print("\n=== MCMC: Normal-Normal (非共轭先验演示) ===")
    
    np.random.seed(42)
    true_mu = 5.0
    true_sigma = 1.0
    data = np.random.normal(true_mu, true_sigma, size=30)
    
    def log_prior(theta):
        mu, log_sigma = theta
        if -10 < mu < 10 and -5 < log_sigma < 5:
            return -1.5 * np.log(1 + mu**2) - 0.5 * log_sigma**2
        return -np.inf
    
    def log_likelihood(theta, data):
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)
        if sigma <= 0:
            return -np.inf
        return -0.5 * np.sum(((data - mu) / sigma)**2 + 2 * np.log(sigma) + np.log(2 * np.pi))
    
    sampler = MCMCSampler(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        ndim=2,
        data=data
    )
    
    print(f"观测数据: {len(data)} 个样本, 样本均值 = {np.mean(data):.4f}")
    print("先验: μ ~ Cauchy(0, 1), log(σ) ~ Normal(0, 1) (非共轭)")
    print("\n正在运行 MCMC 采样...")
    
    result = sampler.sample(
        n_walkers=50,
        n_steps=15000,
        n_burn=5000,
        initial_guess=np.array([0.0, 0.0]),
        initial_scale=0.5
    )
    
    print(f"\n采样完成:")
    print(f"  样本数量: {len(result['samples'])}")
    print(f"  平均接受率: {result['acceptance_rate']:.4f}")
    
    stats = sampler.compute_statistics(
        result['samples'],
        cred_mass=0.95,
        param_names=["mu", "sigma"]
    )
    
    samples = result['samples']
    sigma_samples = np.exp(samples[:, 1])
    stats["sigma"]["mean"] = float(np.mean(sigma_samples))
    stats["sigma"]["median"] = float(np.median(sigma_samples))
    stats["sigma"]["std"] = float(np.std(sigma_samples))
    sigma_hpd = compute_hpd(sigma_samples, 0.95)
    stats["sigma"]["hpd_low"] = float(sigma_hpd[0, 0])
    stats["sigma"]["hpd_high"] = float(sigma_hpd[0, 1])
    
    print("\n后验统计量:")
    for name, s in stats.items():
        print(f"  {name}:")
        print(f"    均值: {s['mean']:.4f}")
        print(f"    中位数: {s['median']:.4f}")
        print(f"    标准差: {s['std']:.4f}")
        print(f"    95% HPD 区间: [{s['hpd_low']:.4f}, {s['hpd_high']:.4f}]")
    
    print(f"\n真实值: μ = {true_mu}, σ = {true_sigma}")


def demo_beta_binomial():
    print("=== Beta-Binomial 模型演示 ===")
    
    alpha_prior = 2.0
    beta_prior = 2.0
    model = BetaBinomial(alpha_prior, beta_prior)
    
    print(f"先验分布: Beta(α={alpha_prior}, β={beta_prior})")
    
    successes = 7
    trials = 10
    print(f"\n观测数据: {successes}次成功 / {trials}次试验")
    
    alpha_post, beta_post = model.update(successes, trials)
    print(f"后验分布: Beta(α={alpha_post}, β={beta_post})")
    
    post_mean = model.posterior_mean(successes, trials)
    post_var = model.posterior_variance(successes, trials)
    print(f"后验均值: {post_mean:.4f}")
    print(f"后验方差: {post_var:.6f}")
    
    new_trials = 5
    new_successes = 3
    pred_prob = model.posterior_predictive(successes, trials, new_successes, new_trials)
    print(f"\n后验预测概率 (未来{new_trials}次试验中成功{new_successes}次): {pred_prob:.6f}")


def demo_normal_normal():
    print("\n=== Normal-Normal 模型演示 ===")
    
    mu_prior = 5.0
    tau_prior = 0.5
    tau_likelihood = 1.0
    model = NormalNormal(mu_prior, tau_prior, tau_likelihood)
    
    print(f"先验分布: Normal(μ={mu_prior}, τ={tau_prior})")
    print(f"似然精度: τ={tau_likelihood}")
    
    data = [4.2, 5.1, 4.8, 5.5, 4.9]
    print(f"\n观测数据: {data}")
    print(f"样本均值: {sum(data)/len(data):.4f}")
    
    mu_post, tau_post = model.update(data)
    print(f"\n后验分布: Normal(μ={mu_post:.4f}, τ={tau_post:.4f})")
    
    post_mean = model.posterior_mean(data)
    post_var = model.posterior_variance(data)
    print(f"后验均值: {post_mean:.4f}")
    print(f"后验方差: {post_var:.6f}")
    
    new_x = 5.0
    pred_prob = model.posterior_predictive(data, new_x)
    print(f"\n后验预测密度 (x={new_x}): {pred_prob:.6f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--mcmc":
        demo_beta_binomial()
        demo_normal_normal()
        demo_mcmc_beta_binomial()
        demo_mcmc_normal()
    else:
        demo_beta_binomial()
        demo_normal_normal()
        print("\n提示: 运行 'python bayesian_update.py --mcmc' 可同时运行 MCMC 演示")
