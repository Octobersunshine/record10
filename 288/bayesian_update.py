from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, Callable, List

from scipy import stats
import numpy as np
import emcee


class PriorCorrectionStrategy(str, Enum):
    SMOOTH = "smooth"
    JEFFREYS = "jeffreys"
    ERROR = "error"


MIN_ALPHA_BETA = 1e-6
JEFFREYS_ALPHA_BETA = 0.5
DEFAULT_HPD_PROB = 0.95


def _correct_prior_params(
    alpha: float,
    beta: float,
    strategy: PriorCorrectionStrategy,
) -> Tuple[float, float, Optional[str]]:
    if alpha >= MIN_ALPHA_BETA and beta >= MIN_ALPHA_BETA:
        return alpha, beta, None

    if strategy == PriorCorrectionStrategy.ERROR:
        raise ValueError(
            f"先验参数 α 和 β 必须至少为 {MIN_ALPHA_BETA} (当前 α={alpha}, β={beta})"
        )

    if strategy == PriorCorrectionStrategy.JEFFREYS:
        new_alpha = JEFFREYS_ALPHA_BETA if alpha < MIN_ALPHA_BETA else alpha
        new_beta = JEFFREYS_ALPHA_BETA if beta < MIN_ALPHA_BETA else beta
        correction_note = (
            f"检测到退化先验 (α={alpha}, β={beta})，"
            f"自动替换为 Jeffreys 先验 (α={new_alpha}, β={new_beta})"
        )
        return new_alpha, new_beta, correction_note

    new_alpha = max(alpha, MIN_ALPHA_BETA)
    new_beta = max(beta, MIN_ALPHA_BETA)
    correction_note = (
        f"检测到退化先验 (α={alpha}, β={beta})，"
        f"自动平滑至最小值 (α={new_alpha}, β={new_beta})"
    )
    return new_alpha, new_beta, correction_note


@dataclass
class BetaBinomialResult:
    prior_alpha: float
    prior_beta: float
    posterior_alpha: float
    posterior_beta: float
    prior_mean: float
    posterior_mean: float
    posterior_predictive_prob: float
    corrected_prior_alpha: Optional[float] = None
    corrected_prior_beta: Optional[float] = None
    correction_note: Optional[str] = None

    def summary(self) -> str:
        lines = []
        if self.correction_note:
            lines.append(f"⚠️  {self.correction_note}")
        if self.corrected_prior_alpha is not None and self.corrected_prior_beta is not None:
            lines.append(
                f"输入先验: Beta(α={self.prior_alpha}, β={self.prior_beta})"
            )
            lines.append(
                f"修正后先验: Beta(α={self.corrected_prior_alpha}, β={self.corrected_prior_beta})"
            )
        else:
            lines.append(f"先验分布: Beta(α={self.prior_alpha}, β={self.prior_beta})")
        lines.append(f"先验均值: {self.prior_mean:.6f}")
        lines.append(f"后验分布: Beta(α={self.posterior_alpha}, β={self.posterior_beta})")
        lines.append(f"后验均值: {self.posterior_mean:.6f}")
        lines.append(
            f"后验预测概率 (下一次试验成功的概率): {self.posterior_predictive_prob:.6f}"
        )
        return "\n".join(lines)


@dataclass
class NormalNormalResult:
    prior_mu: float
    prior_sigma2: float
    known_likelihood_sigma2: float
    posterior_mu: float
    posterior_sigma2: float
    prior_mean: float
    posterior_mean: float
    posterior_predictive_mean: float
    posterior_predictive_variance: float

    def summary(self) -> str:
        lines = [
            f"先验分布: N(μ={self.prior_mu}, σ²={self.prior_sigma2})",
            f"先验均值: {self.prior_mean:.6f}",
            f"后验分布: N(μ={self.posterior_mu}, σ²={self.posterior_sigma2})",
            f"后验均值: {self.posterior_mean:.6f}",
            f"后验预测分布: N(μ={self.posterior_predictive_mean:.6f}, σ²={self.posterior_predictive_variance:.6f})",
        ]
        return "\n".join(lines)


@dataclass
class MCMCResult:
    param_names: List[str]
    samples: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    hpd_intervals: np.ndarray
    n_samples_total: int
    n_burn_in: int
    n_walkers: int
    acceptance_fraction: np.ndarray
    autocorr_time: Optional[np.ndarray] = None
    ess: Optional[np.ndarray] = None

    @property
    def n_dim(self) -> int:
        return self.samples.shape[1]

    def summary(self) -> str:
        lines = [
            "MCMC 采样结果",
            f"总样本数: {self.n_samples_total}",
            f"预热样本数: {self.n_burn_in}",
            f"参数数量: {self.n_dim}",
            "",
        ]
        for i, name in enumerate(self.param_names):
            hpd_low, hpd_high = self.hpd_intervals[i]
            lines.append(f"参数 {name}:")
            lines.append(f"  均值: {self.mean[i]:.6f}")
            lines.append(f"  中位数: {self.median[i]:.6f}")
            lines.append(
                f"  {DEFAULT_HPD_PROB * 100:.0f}% HPD 区间: [{hpd_low:.6f}, {hpd_high:.6f}]"
            )
            lines.append(f"  接受率: {self.acceptance_fraction[i]:.4f}")
            if self.ess is not None:
                lines.append(f"  ESS: {self.ess[i]:.1f}")
        return "\n".join(lines)


def _hpd_interval(
    posterior_samples: np.ndarray,
    prob: float = DEFAULT_HPD_PROB,
) -> np.ndarray:
    if not 0 < prob < 1:
        raise ValueError("prob 必须在 (0, 1) 之间")

    n_samples = len(posterior_samples)
    sorted_samples = np.sort(posterior_samples)
    n_included = int(np.floor(prob * n_samples))
    if n_included < 2:
        n_included = 2

    n_intervals = n_samples - n_included + 1
    interval_widths = np.zeros(n_intervals)
    for i in range(n_intervals):
        interval_widths[i] = sorted_samples[i + n_included - 1] - sorted_samples[i]

    best_idx = np.argmin(interval_widths)
    return np.array(
        [sorted_samples[best_idx], sorted_samples[best_idx + n_included - 1]]
    )


def calculate_hpd(
    samples: np.ndarray,
    prob: float = DEFAULT_HPD_PROB,
) -> np.ndarray:
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n_params = samples.shape[1]
    intervals = np.zeros((n_params, 2))
    for i in range(n_params):
        intervals[i] = _hpd_interval(samples[:, i], prob)
    return intervals


def _effective_sample_size(
    samples: np.ndarray,
) -> np.ndarray:
    n_params = samples.shape[1]
    ess = np.zeros(n_params)
    for i in range(n_params):
        chain = samples[:, i]
        n = len(chain)

        mean_chain = np.mean(chain)
        centered = chain - mean_chain

        max_lag = min(n // 4, 200)
        autocorr = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                autocorr[lag] = np.sum(centered[:-lag] * centered[lag:]) / np.sum(centered ** 2)

        tau = 1.0
        for lag in range(1, max_lag):
            if autocorr[lag] + autocorr[lag - 1] < 0:
                break
            tau += 2.0 * autocorr[lag]

        ess[i] = n / tau
    return ess


def mcmc_update(
    log_posterior: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    param_names: Optional[List[str]] = None,
    n_walkers: int = 32,
    n_samples: int = 5000,
    n_burn_in: int = 1000,
    hpd_prob: float = DEFAULT_HPD_PROB,
    random_seed: Optional[int] = 42,
    progress: bool = True,
) -> MCMCResult:
    initial_params = np.asarray(initial_params)
    n_dim = len(initial_params)

    if param_names is None:
        param_names = [f"θ_{i}" for i in range(n_dim)]

    if len(param_names) != n_dim:
        raise ValueError("param_names 长度必须与 initial_params 长度一致")

    if n_walkers < 2 * n_dim + 2:
        n_walkers = 2 * n_dim + 2

    if random_seed is not None:
        np.random.seed(random_seed)

    def log_prob_fn(params):
        result = log_posterior(params)
        if not np.isfinite(result):
            return -np.inf
        return float(result)

    initial_positions = initial_params + 1e-4 * np.random.randn(n_walkers, n_dim)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_fn)

    if progress:
        print(f"预热阶段 ({n_burn_in} 步)...")
    sampler.run_mcmc(initial_positions, n_burn_in, progress=progress)

    sampler.reset()

    if progress:
        print(f"采样阶段 ({n_samples} 步)...")
    sampler.run_mcmc(None, n_samples, progress=progress)

    acceptance_fraction = np.mean(sampler.acceptance_fraction)

    samples = sampler.get_chain(discard=0, flat=True)

    try:
        autocorr_time = sampler.get_autocorr_time(tol=0)
    except Exception:
        autocorr_time = None

    mean = np.mean(samples, axis=0)
    median = np.median(samples, axis=0)
    hpd_intervals = calculate_hpd(samples, hpd_prob)

    ess = _effective_sample_size(samples)

    acceptance_fraction_per_param = np.full(n_dim, acceptance_fraction)

    return MCMCResult(
        param_names=param_names,
        samples=samples,
        mean=mean,
        median=median,
        hpd_intervals=hpd_intervals,
        n_samples_total=n_samples,
        n_burn_in=n_burn_in,
        n_walkers=n_walkers,
        acceptance_fraction=acceptance_fraction_per_param,
        autocorr_time=autocorr_time,
        ess=ess,
    )


def log_posterior_beta_binomial(
    theta: np.ndarray,
    successes: int,
    total_trials: int,
    log_prior: Optional[Callable[[float], float]] = None,
) -> float:
    p = theta[0]
    if p <= 0 or p >= 1:
        return -np.inf

    log_likelihood = successes * np.log(p) + (total_trials - successes) * np.log(1 - p)

    if log_prior is None:
        log_prior_val = 0.0
    else:
        log_prior_val = log_prior(p)

    return log_likelihood + log_prior_val


def log_posterior_normal(
    theta: np.ndarray,
    observations: np.ndarray,
    known_sigma2: Optional[float] = None,
    log_prior: Optional[Callable[[np.ndarray], float]] = None,
) -> float:
    if known_sigma2 is None:
        if len(theta) < 2:
            return -np.inf
        mu, sigma2 = theta[0], theta[1]
        if sigma2 <= 0:
            return -np.inf
    else:
        mu = theta[0]
        sigma2 = known_sigma2

    n = len(observations)
    log_likelihood = -0.5 * np.sum((observations - mu) ** 2) / sigma2
    log_likelihood -= 0.5 * n * np.log(2 * np.pi * sigma2)

    if log_prior is None:
        log_prior_val = 0.0
    else:
        log_prior_val = log_prior(theta)

    return log_likelihood + log_prior_val


def beta_binomial_mcmc(
    successes: int,
    total_trials: int,
    log_prior: Optional[Callable[[float], float]] = None,
    initial_p: float = 0.5,
    n_walkers: int = 32,
    n_samples: int = 5000,
    n_burn_in: int = 1000,
    hpd_prob: float = DEFAULT_HPD_PROB,
    random_seed: Optional[int] = 42,
    progress: bool = True,
) -> MCMCResult:
    def log_post(theta):
        return log_posterior_beta_binomial(theta, successes, total_trials, log_prior)

    return mcmc_update(
        log_post,
        initial_params=np.array([initial_p]),
        param_names=["p"],
        n_walkers=n_walkers,
        n_samples=n_samples,
        n_burn_in=n_burn_in,
        hpd_prob=hpd_prob,
        random_seed=random_seed,
        progress=progress,
    )


def normal_mcmc(
    observations: np.ndarray,
    known_sigma2: Optional[float] = None,
    log_prior: Optional[Callable[[np.ndarray], float]] = None,
    initial_mu: float = 0.0,
    initial_sigma2: float = 1.0,
    n_walkers: int = 32,
    n_samples: int = 5000,
    n_burn_in: int = 1000,
    hpd_prob: float = DEFAULT_HPD_PROB,
    random_seed: Optional[int] = 42,
    progress: bool = True,
) -> MCMCResult:
    def log_post(theta):
        return log_posterior_normal(theta, observations, known_sigma2, log_prior)

    if known_sigma2 is None:
        initial_params = np.array([initial_mu, initial_sigma2])
        param_names = ["mu", "sigma2"]
    else:
        initial_params = np.array([initial_mu])
        param_names = ["mu"]

    return mcmc_update(
        log_post,
        initial_params=initial_params,
        param_names=param_names,
        n_walkers=n_walkers,
        n_samples=n_samples,
        n_burn_in=n_burn_in,
        hpd_prob=hpd_prob,
        random_seed=random_seed,
        progress=progress,
    )


def beta_binomial_update(
    prior_alpha: float,
    prior_beta: float,
    successes: int,
    total_trials: int,
    correction_strategy: PriorCorrectionStrategy = PriorCorrectionStrategy.JEFFREYS,
) -> BetaBinomialResult:
    if successes < 0 or total_trials < 0:
        raise ValueError("成功次数和总试验次数不能为负数")
    if successes > total_trials:
        raise ValueError("成功次数不能超过总试验次数")

    corrected_alpha, corrected_beta, correction_note = _correct_prior_params(
        prior_alpha, prior_beta, correction_strategy
    )

    failures = total_trials - successes

    posterior_alpha = corrected_alpha + successes
    posterior_beta = corrected_beta + failures

    prior_mean = corrected_alpha / (corrected_alpha + corrected_beta)
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

    posterior_predictive_prob = posterior_alpha / (posterior_alpha + posterior_beta)

    result_kwargs = dict(
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        posterior_alpha=posterior_alpha,
        posterior_beta=posterior_beta,
        prior_mean=prior_mean,
        posterior_mean=posterior_mean,
        posterior_predictive_prob=posterior_predictive_prob,
    )

    if correction_note is not None:
        result_kwargs.update(
            corrected_prior_alpha=corrected_alpha,
            corrected_prior_beta=corrected_beta,
            correction_note=correction_note,
        )

    return BetaBinomialResult(**result_kwargs)


def normal_normal_update(
    prior_mu: float,
    prior_sigma2: float,
    known_likelihood_sigma2: float,
    observations: np.ndarray,
) -> NormalNormalResult:
    if prior_sigma2 <= 0:
        raise ValueError("先验方差 σ²₀ 必须为正数")
    if known_likelihood_sigma2 <= 0:
        raise ValueError("似然方差 σ² 必须为正数")

    n = len(observations)
    sample_mean = np.mean(observations)

    posterior_sigma2 = 1.0 / (1.0 / prior_sigma2 + n / known_likelihood_sigma2)

    posterior_mu = posterior_sigma2 * (
        prior_mu / prior_sigma2 + n * sample_mean / known_likelihood_sigma2
    )

    posterior_predictive_mean = posterior_mu
    posterior_predictive_variance = posterior_sigma2 + known_likelihood_sigma2

    return NormalNormalResult(
        prior_mu=prior_mu,
        prior_sigma2=prior_sigma2,
        known_likelihood_sigma2=known_likelihood_sigma2,
        posterior_mu=posterior_mu,
        posterior_sigma2=posterior_sigma2,
        prior_mean=prior_mu,
        posterior_mean=posterior_mu,
        posterior_predictive_mean=posterior_predictive_mean,
        posterior_predictive_variance=posterior_predictive_variance,
    )


def beta_binomial_posterior_pdf(
    posterior_alpha: float, posterior_beta: float, x: np.ndarray
) -> np.ndarray:
    return stats.beta.pdf(x, posterior_alpha, posterior_beta)


def normal_normal_posterior_pdf(
    posterior_mu: float, posterior_sigma2: float, x: np.ndarray
) -> np.ndarray:
    return stats.norm.pdf(x, posterior_mu, np.sqrt(posterior_sigma2))


if __name__ == "__main__":
    print("=" * 60)
    print("Beta-Binomial 共轭先验贝叶斯更新")
    print("=" * 60)

    bb_result = beta_binomial_update(
        prior_alpha=2.0,
        prior_beta=5.0,
        successes=7,
        total_trials=10,
    )
    print(bb_result.summary())

    print()
    print("=" * 60)
    print("退化先验修复测试 (Jeffreys 策略)")
    print("=" * 60)
    bb_jeffreys = beta_binomial_update(
        prior_alpha=0.0,
        prior_beta=0.0,
        successes=7,
        total_trials=10,
        correction_strategy=PriorCorrectionStrategy.JEFFREYS,
    )
    print(bb_jeffreys.summary())

    print()
    print("=" * 60)
    print("退化先验修复测试 (平滑策略)")
    print("=" * 60)
    bb_smooth = beta_binomial_update(
        prior_alpha=0.0,
        prior_beta=0.0,
        successes=7,
        total_trials=10,
        correction_strategy=PriorCorrectionStrategy.SMOOTH,
    )
    print(bb_smooth.summary())

    print()
    print("=" * 60)
    print("退化先验修复验证 (PDF 积分)")
    print("=" * 60)
    x = np.linspace(0, 1, 1000)
    jeffreys_pdf = beta_binomial_posterior_pdf(
        bb_jeffreys.posterior_alpha, bb_jeffreys.posterior_beta, x
    )
    smooth_pdf = beta_binomial_posterior_pdf(
        bb_smooth.posterior_alpha, bb_smooth.posterior_beta, x
    )
    print(f"Jeffreys 策略后验 PDF 积分: {np.trapezoid(jeffreys_pdf, x):.6f} (应接近1.0)")
    print(f"平滑策略后验 PDF 积分: {np.trapezoid(smooth_pdf, x):.6f} (应接近1.0)")

    print()
    print("=" * 60)
    print("退化先验修复测试 (ERROR 策略 - 期望抛出异常)")
    print("=" * 60)
    try:
        beta_binomial_update(
            prior_alpha=0.0,
            prior_beta=0.0,
            successes=7,
            total_trials=10,
            correction_strategy=PriorCorrectionStrategy.ERROR,
        )
        print("❌ 错误：期望抛出异常但没有抛出")
    except ValueError as e:
        print(f"✅ 正确抛出异常: {e}")

    print()
    print("=" * 60)
    print("Normal-Normal 共轭先验贝叶斯更新")
    print("=" * 60)

    obs = np.array([4.8, 5.2, 4.6, 5.1, 5.3, 4.9, 5.0])
    nn_result = normal_normal_update(
        prior_mu=5.0,
        prior_sigma2=1.0,
        known_likelihood_sigma2=0.5,
        observations=obs,
    )
    print(nn_result.summary())

    print()
    print("=" * 60)
    print("验证：Normal-Normal 后验预测分布")
    print("=" * 60)
    x_norm = np.linspace(posterior_mu := nn_result.posterior_mu - 3, nn_result.posterior_mu + 3, 1000)
    predictive_pdf = stats.norm.pdf(
        x_norm,
        nn_result.posterior_predictive_mean,
        np.sqrt(nn_result.posterior_predictive_variance),
    )
    print(f"预测分布 PDF 积分: {np.trapezoid(predictive_pdf, x_norm):.6f} (应接近1.0)")

    print()
    print("=" * 60)
    print("MCMC 采样示例 1: Beta-Binomial (非共轭先验)")
    print("=" * 60)

    def bimodal_log_prior(p: float) -> float:
        if p <= 0 or p >= 1:
            return -np.inf
        peak1 = stats.norm.pdf(p, 0.3, 0.1)
        peak2 = stats.norm.pdf(p, 0.7, 0.1)
        return np.log(0.5 * peak1 + 0.5 * peak2)

    print("使用双峰先验 p ~ 0.5*N(0.3, 0.1²) + 0.5*N(0.7, 0.1²)")
    mcmc_bb = beta_binomial_mcmc(
        successes=7,
        total_trials=10,
        log_prior=bimodal_log_prior,
        n_samples=3000,
        n_burn_in=500,
        progress=True,
    )
    print(mcmc_bb.summary())

    print()
    print("MCMC 后验均值与共轭解析解对比:")
    print(f"  MCMC 后验均值: {mcmc_bb.mean[0]:.6f}")
    print(f"  解析后验均值 (无先验): {7/10:.6f}")

    print()
    print("=" * 60)
    print("MCMC 采样示例 2: Normal (已知方差，非共轭先验)")
    print("=" * 60)

    def heavy_tailed_log_prior(theta: np.ndarray) -> float:
        mu = theta[0]
        return stats.cauchy.logpdf(mu, loc=5.0, scale=1.0)

    print("使用柯西先验 μ ~ Cauchy(5.0, 1.0) (厚尾先验)")
    mcmc_norm = normal_mcmc(
        observations=obs,
        known_sigma2=0.5,
        log_prior=heavy_tailed_log_prior,
        initial_mu=5.0,
        n_samples=3000,
        n_burn_in=500,
        progress=True,
    )
    print(mcmc_norm.summary())

    print()
    print("MCMC 后验均值与共轭解析解对比:")
    print(f"  MCMC 后验均值: {mcmc_norm.mean[0]:.6f}")
    print(f"  解析后验均值: {nn_result.posterior_mu:.6f}")

    print()
    print("=" * 60)
    print("MCMC 采样示例 3: Normal (未知方差，两参数估计)")
    print("=" * 60)

    def gamma_log_prior_sigma2(theta: np.ndarray) -> float:
        if len(theta) < 2:
            return -np.inf
        mu, sigma2 = theta[0], theta[1]
        if sigma2 <= 0:
            return -np.inf
        log_prior_mu = stats.norm.logpdf(mu, 5.0, 2.0)
        log_prior_sigma2 = stats.gamma.logpdf(sigma2, a=2.0, scale=1.0)
        return log_prior_mu + log_prior_sigma2

    print("使用 μ ~ N(5.0, 2.0²), σ² ~ Gamma(2.0, 1.0) 先验")
    mcmc_norm2 = normal_mcmc(
        observations=obs,
        known_sigma2=None,
        log_prior=gamma_log_prior_sigma2,
        initial_mu=5.0,
        initial_sigma2=1.0,
        n_samples=5000,
        n_burn_in=1000,
        progress=True,
    )
    print(mcmc_norm2.summary())

    print()
    print("=" * 60)
    print("MCMC 采样示例 4: 自定义任意后验分布")
    print("=" * 60)

    def custom_log_posterior(theta: np.ndarray) -> float:
        x, y = theta[0], theta[1]
        r = np.sqrt(x**2 + y**2)
        log_likelihood = -0.5 * ((x - 1.0) ** 2 + (y - 2.0) ** 2) / 0.5
        log_prior = -np.log1p(r**2)
        return log_likelihood + log_prior

    print("估计二维参数 (x, y)，先验为柯西分布")
    mcmc_custom = mcmc_update(
        log_posterior=custom_log_posterior,
        initial_params=np.array([0.0, 0.0]),
        param_names=["x", "y"],
        n_samples=3000,
        n_burn_in=500,
        progress=True,
    )
    print(mcmc_custom.summary())
    print(f"  真实值: x=1.0, y=2.0")

    print()
    print("=" * 60)
    print("HPD 区间验证")
    print("=" * 60)
    test_samples = np.random.normal(0, 1, 10000)
    hpd_95 = calculate_hpd(test_samples, prob=0.95)
    hpd_68 = calculate_hpd(test_samples, prob=0.68)
    print(f"N(0,1) 样本 95% HPD 区间: [{hpd_95[0, 0]:.3f}, {hpd_95[0, 1]:.3f}] (应接近 [-1.96, 1.96])")
    print(f"N(0,1) 样本 68% HPD 区间: [{hpd_68[0, 0]:.3f}, {hpd_68[0, 1]:.3f}] (应接近 [-1.0, 1.0])")
