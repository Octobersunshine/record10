import numpy as np
from scipy import stats
from scipy.special import logsumexp, gammaln
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class StudentTLog:
    def __init__(self, alpha: float = 0.1, beta: float = 0.001, mu: float = 0.0, kappa: float = 1.0):
        self.alpha0 = np.float64(alpha)
        self.beta0 = np.float64(beta)
        self.mu0 = np.float64(mu)
        self.kappa0 = np.float64(kappa)
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.mu = np.float64(mu)
        self.kappa = np.float64(kappa)
        self.n = 0

    def logpdf(self, x: float) -> float:
        x = np.float64(x)
        
        df = 2 * self.alpha
        if df <= 0 or self.beta <= 0 or self.kappa <= 0:
            return -np.inf
        
        scale_sq = self.beta * (self.kappa + 1) / (self.alpha * self.kappa)
        if scale_sq <= 0:
            return -np.inf
        
        scale = np.sqrt(scale_sq)
        z = (x - self.mu) / scale
        
        log_t = (gammaln((df + 1) / 2) 
                 - gammaln(df / 2) 
                 - 0.5 * np.log(np.pi * df) 
                 - np.log(scale)
                 - (df + 1) / 2 * np.log1p(z ** 2 / df))
        
        return np.clip(log_t, -1000, 10)

    def update(self, x: float) -> None:
        x = np.float64(x)
        self.n += 1
        
        mu_temp = (self.kappa * self.mu + x) / (self.kappa + 1)
        delta_sq = (x - self.mu) ** 2
        self.beta = self.beta + self.kappa * delta_sq / (2 * (self.kappa + 1))
        self.mu = mu_temp
        self.kappa += 1
        self.alpha += 0.5
        
        self.beta = max(self.beta, 1e-10)
        self.alpha = max(self.alpha, 1e-10)
        self.kappa = max(self.kappa, 1e-10)

    def copy(self):
        new_model = StudentTLog(self.alpha0, self.beta0, self.mu0, self.kappa0)
        new_model.alpha = self.alpha
        new_model.beta = self.beta
        new_model.mu = self.mu
        new_model.kappa = self.kappa
        new_model.n = self.n
        return new_model


class Particle:
    def __init__(self, run_length: int, model: StudentTLog, log_weight: float = 0.0):
        self.run_length = run_length
        self.model = model
        self.log_weight = log_weight


class BOCPDParticleFilter:
    def __init__(
        self,
        hazard: float = 1 / 252,
        model_params: Optional[dict] = None,
        n_particles: int = 1000,
        ess_threshold_ratio: float = 0.5,
        max_run_length: int = 5000
    ):
        self.hazard = np.float64(hazard)
        self.log_hazard = np.log(self.hazard)
        self.log_1m_hazard = np.log(1 - self.hazard)
        self.model_params = model_params or {}
        self.n_particles = n_particles
        self.ess_threshold = int(n_particles * ess_threshold_ratio)
        self.max_run_length = max_run_length
        
        self.particles: List[Particle] = []
        self._initialize_particles()
        
        self.cp_probabilities = []
        self.timestep = 0
        self.log_ml = []

    def _initialize_particles(self) -> None:
        self.particles = []
        for _ in range(self.n_particles):
            model = StudentTLog(**self.model_params)
            self.particles.append(Particle(run_length=0, model=model, log_weight=-np.log(self.n_particles)))

    def _compute_ess(self, log_weights: np.ndarray) -> float:
        log_weights = np.array(log_weights)
        log_weights_norm = log_weights - logsumexp(log_weights)
        weights = np.exp(log_weights_norm)
        return 1.0 / np.sum(weights ** 2)

    def _resample(self) -> None:
        log_weights = np.array([p.log_weight for p in self.particles])
        log_weights = log_weights - logsumexp(log_weights)
        weights = np.exp(log_weights)
        
        indices = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            p=weights
        )
        
        new_particles = []
        for idx in indices:
            old_p = self.particles[idx]
            new_p = Particle(
                run_length=old_p.run_length,
                model=old_p.model.copy(),
                log_weight=-np.log(self.n_particles)
            )
            new_particles.append(new_p)
        
        self.particles = new_particles

    def update(self, x: float) -> float:
        x = np.float64(x)
        self.timestep += 1
        
        new_particles = []
        all_log_weights = []
        
        for particle in self.particles:
            log_pred = particle.model.logpdf(x)
            
            if np.isinf(log_pred) and log_pred < 0:
                log_growth = -np.inf
                log_cp = -np.inf
            else:
                log_growth = particle.log_weight + log_pred + self.log_1m_hazard
                log_cp = particle.log_weight + log_pred + self.log_hazard
            
            if not np.isinf(log_growth):
                new_model = particle.model.copy()
                new_model.update(x)
                new_run_length = min(particle.run_length + 1, self.max_run_length)
                new_particles.append(Particle(new_run_length, new_model, log_growth))
                all_log_weights.append(log_growth)
            
            if not np.isinf(log_cp):
                new_model = StudentTLog(**self.model_params)
                new_model.update(x)
                new_particles.append(Particle(0, new_model, log_cp))
                all_log_weights.append(log_cp)
        
        if len(new_particles) == 0:
            self._initialize_particles()
            for p in self.particles:
                p.model.update(x)
            return 0.5
        
        if len(new_particles) > self.n_particles * 2:
            all_log_weights = np.array(all_log_weights)
            top_indices = np.argsort(all_log_weights)[-self.n_particles * 2:]
            new_particles = [new_particles[i] for i in top_indices]
            all_log_weights = all_log_weights[top_indices]
        
        if len(new_particles) > self.n_particles:
            all_log_weights = np.array([p.log_weight for p in new_particles])
            probs = np.exp(all_log_weights - logsumexp(all_log_weights))
            selected = np.random.choice(
                len(new_particles),
                size=self.n_particles,
                p=probs,
                replace=True
            )
            new_particles = [new_particles[i] for i in selected]
            for p in new_particles:
                p.log_weight = -np.log(self.n_particles)
        
        self.particles = new_particles
        
        log_weights = np.array([p.log_weight for p in self.particles])
        log_weights_norm = log_weights - logsumexp(log_weights)
        
        for i, p in enumerate(self.particles):
            p.log_weight = log_weights_norm[i]
        
        ess = self._compute_ess([p.log_weight for p in self.particles])
        if ess < self.ess_threshold:
            self._resample()
        
        cp_prob = self._compute_cp_probability()
        self.cp_probabilities.append(cp_prob)
        
        return cp_prob

    def _compute_cp_probability(self) -> float:
        if not self.particles:
            return 0.0
        
        log_weights = np.array([p.log_weight for p in self.particles])
        run_lengths = np.array([p.run_length for p in self.particles])
        
        cp_mask = (run_lengths == 0)
        if not np.any(cp_mask):
            return 1e-10
        
        cp_log_weights = log_weights[cp_mask]
        total_log_weight = logsumexp(log_weights)
        cp_log_total = logsumexp(cp_log_weights)
        
        prob = np.exp(cp_log_total - total_log_weight)
        return np.clip(prob, 1e-10, 1.0 - 1e-10)

    def get_run_length_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.particles:
            return np.array([0]), np.array([1.0])
        
        log_weights = np.array([p.log_weight for p in self.particles])
        run_lengths = np.array([p.run_length for p in self.particles])
        
        unique_rl = np.unique(run_lengths)
        probs = []
        
        for rl in unique_rl:
            mask = (run_lengths == rl)
            rl_log_prob = logsumexp(log_weights[mask])
            probs.append(rl_log_prob)
        
        probs = np.array(probs)
        probs = np.exp(probs - logsumexp(probs))
        
        return unique_rl, probs

    def detect_changepoints(
        self, 
        data: np.ndarray, 
        threshold: float = 0.5,
        min_delay: int = 5,
        verbose: bool = True
    ) -> Tuple[List[int], np.ndarray]:
        n = len(data)
        cp_probs = np.zeros(n)
        
        for t, x in enumerate(data):
            cp_prob = self.update(x)
            cp_probs[t] = cp_prob
            
            if verbose and (t + 1) % 100 == 0:
                print(f"已处理 {t+1}/{n} 个数据点...")
        
        changepoints = []
        last_cp = -min_delay
        
        for t in range(n):
            if cp_probs[t] > threshold and (t - last_cp) >= min_delay:
                changepoints.append(t)
                last_cp = t
        
        return changepoints, cp_probs


class BOCPDLogExact:
    def __init__(
        self,
        hazard: float = 1 / 252,
        model_params: Optional[dict] = None,
        max_duration: int = 500,
        pruning_threshold: float = 1e-10
    ):
        self.hazard = np.float64(hazard)
        self.log_hazard = np.log(self.hazard)
        self.log_1m_hazard = np.log(1 - self.hazard)
        self.max_duration = max_duration
        self.pruning_threshold = pruning_threshold
        self.model_params = model_params or {}
        
        self.log_R = np.array([0.0], dtype=np.float64)
        self.models = [StudentTLog(**self.model_params)]
        self.cp_probs = []

    def update(self, x: float) -> float:
        x = np.float64(x)
        T = len(self.log_R)
        
        log_pred = np.zeros(T, dtype=np.float64)
        for i, model in enumerate(self.models):
            log_pred[i] = model.logpdf(x)
        
        log_pred = np.clip(log_pred, -500, 10)
        
        log_growth = self.log_R + log_pred + self.log_1m_hazard
        log_cp = logsumexp(self.log_R + log_pred + self.log_hazard)
        
        new_log_R = np.zeros(T + 1, dtype=np.float64)
        new_log_R[1:] = log_growth
        new_log_R[0] = log_cp
        
        log_max = np.max(new_log_R)
        new_log_R -= log_max
        
        new_log_R = np.clip(new_log_R, -500, 0)
        
        log_norm = logsumexp(new_log_R)
        new_log_R -= log_norm
        
        for model in self.models:
            model.update(x)
        self.models.insert(0, StudentTLog(**self.model_params))
        
        self.log_R = new_log_R
        
        if len(self.log_R) > self.max_duration:
            tail_log_prob = logsumexp(self.log_R[self.max_duration:])
            self.log_R = self.log_R[:self.max_duration]
            self.models = self.models[:self.max_duration]
            self.log_R[-1] = logsumexp([self.log_R[-1], tail_log_prob])
        
        weights = np.exp(self.log_R)
        keep_mask = weights > self.pruning_threshold
        if not keep_mask[0]:
            keep_mask[0] = True
        
        self.log_R = self.log_R[keep_mask]
        self.models = [self.models[i] for i in np.where(keep_mask)[0]]
        
        log_norm = logsumexp(self.log_R)
        self.log_R -= log_norm
        
        cp_prob = np.clip(np.exp(self.log_R[0]), 1e-10, 1.0 - 1e-10)
        self.cp_probs.append(cp_prob)
        
        return cp_prob

    def detect_changepoints(
        self, 
        data: np.ndarray, 
        threshold: float = 0.5,
        verbose: bool = True
    ) -> Tuple[List[int], np.ndarray]:
        n = len(data)
        cp_probs = np.zeros(n)
        
        for t, x in enumerate(data):
            cp_probs[t] = self.update(x)
            if verbose and (t + 1) % 100 == 0:
                print(f"已处理 {t+1}/{n} 个数据点...")
        
        changepoints = np.where(cp_probs > threshold)[0]
        return list(changepoints), cp_probs


def generate_simulated_data(
    n_samples: int = 1000,
    changepoint_locations: list = None,
    volatility_levels: list = None,
    seed: int = 42
) -> Tuple[np.ndarray, List[int]]:
    np.random.seed(seed)
    
    if changepoint_locations is None:
        changepoint_locations = [250, 500, 750]
    if volatility_levels is None:
        volatility_levels = [0.01, 0.05, 0.01, 0.08]
    
    changepoint_locations = [0] + changepoint_locations + [n_samples]
    returns = []
    
    for i in range(len(changepoint_locations) - 1):
        start = changepoint_locations[i]
        end = changepoint_locations[i + 1]
        vol = volatility_levels[i]
        df = 5
        segment = stats.t.rvs(df=df, loc=0, scale=vol, size=end - start)
        returns.extend(segment)
    
    true_cps = changepoint_locations[1:-1]
    return np.array(returns), true_cps


def plot_results(
    returns: np.ndarray,
    true_cps: List[int],
    detected_cps: List[int],
    cp_probs: np.ndarray,
    title: str = 'BOCPD波动率变点检测',
    save_path: str = 'bocpd_stable_detection.png'
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(returns, color='blue', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('对数收益率')
    ax1.set_title(title)
    
    for cp in true_cps:
        ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.7, 
                   label='真实变点' if cp == true_cps[0] else "")
    
    for cp in detected_cps:
        ax1.axvline(x=cp, color='green', linestyle='-', alpha=0.7, 
                   label='检测变点' if cp == detected_cps[0] else "")
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(cp_probs, color='purple', linewidth=0.8)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='阈值 (0.5)')
    ax2.set_xlabel('时间')
    ax2.set_ylabel('变点概率')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存至 {save_path}")


def test_particle_filter():
    print("=" * 60)
    print("测试粒子滤波BOCPD (长序列稳定性)")
    print("=" * 60)
    
    n_samples = 2000
    true_cps = [500, 1000, 1500]
    volatility_levels = [0.01, 0.05, 0.01, 0.08]
    
    print(f"\n生成 {n_samples} 个模拟数据点...")
    returns, true_cps = generate_simulated_data(n_samples, true_cps, volatility_levels, seed=42)
    print(f"真实变点位置: {true_cps}")
    
    print("\n初始化粒子滤波BOCPD (2000粒子)...")
    bocpd_pf = BOCPDParticleFilter(
        hazard=1/200,
        model_params={'alpha': 0.1, 'beta': 0.001, 'mu': 0.0, 'kappa': 1.0},
        n_particles=2000,
        ess_threshold_ratio=0.5,
        max_run_length=1000
    )
    
    print("运行变点检测...")
    detected_cps_pf, cp_probs_pf = bocpd_pf.detect_changepoints(
        returns, 
        threshold=0.3,
        min_delay=20,
        verbose=True
    )
    
    print(f"\n粒子滤波检测到的变点: {detected_cps_pf}")
    
    plot_results(
        returns, true_cps, detected_cps_pf, cp_probs_pf,
        title='粒子滤波BOCPD波动率变点检测',
        save_path='bocpd_particle_filter.png'
    )
    
    return detected_cps_pf, cp_probs_pf


def test_log_exact():
    print("\n" + "=" * 60)
    print("测试对数域精确BOCPD")
    print("=" * 60)
    
    n_samples = 1000
    true_cps = [250, 500, 750]
    volatility_levels = [0.01, 0.05, 0.01, 0.08]
    
    print(f"\n生成 {n_samples} 个模拟数据点...")
    returns, true_cps = generate_simulated_data(n_samples, true_cps, volatility_levels, seed=42)
    print(f"真实变点位置: {true_cps}")
    
    print("\n初始化对数域精确BOCPD...")
    bocpd_log = BOCPDLogExact(
        hazard=1/100,
        model_params={'alpha': 0.1, 'beta': 0.001, 'mu': 0.0, 'kappa': 1.0},
        max_duration=500,
        pruning_threshold=1e-10
    )
    
    print("运行变点检测...")
    detected_cps_log, cp_probs_log = bocpd_log.detect_changepoints(
        returns,
        verbose=True
    )
    
    print(f"\n对数域精确BOCPD检测到的变点: {detected_cps_log}")
    
    plot_results(
        returns, true_cps, detected_cps_log, cp_probs_log,
        title='对数域精确BOCPD波动率变点检测',
        save_path='bocpd_log_exact.png'
    )
    
    return detected_cps_log, cp_probs_log


def compare_stability():
    print("\n" + "=" * 60)
    print("长序列稳定性对比测试 (5000数据点)")
    print("=" * 60)
    
    n_samples = 5000
    true_cps = [1000, 2000, 3000, 4000]
    volatility_levels = [0.01, 0.05, 0.01, 0.08, 0.02]
    
    returns, true_cps = generate_simulated_data(n_samples, true_cps, volatility_levels, seed=123)
    print(f"真实变点位置: {true_cps}")
    
    print("\n测试粒子滤波BOCPD...")
    bocpd_pf = BOCPDParticleFilter(
        hazard=1/300,
        n_particles=1500,
        ess_threshold_ratio=0.3
    )
    
    try:
        detected_cps, cp_probs = bocpd_pf.detect_changepoints(
            returns, 
            threshold=0.3,
            min_delay=50,
            verbose=True
        )
        print(f"粒子滤波检测成功! 检测到的变点: {detected_cps}")
        
        plot_results(
            returns, true_cps, detected_cps, cp_probs,
            title='粒子滤波BOCPD长序列检测 (n=5000)',
            save_path='bocpd_long_sequence.png'
        )
    except Exception as e:
        print(f"粒子滤波出现错误: {e}")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_particle_filter()
    test_log_exact()
    compare_stability()
