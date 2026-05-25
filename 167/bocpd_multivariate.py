import numpy as np
from scipy import stats
from scipy.special import logsumexp, gammaln, logdet
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MultivariateStudentT:
    def __init__(
        self,
        dim: int,
        nu: float = None,
        mu: np.ndarray = None,
        Lambda: np.ndarray = None,
        kappa: float = 1.0
    ):
        self.dim = dim
        self.nu0 = nu if nu is not None else dim + 2
        self.mu0 = mu if mu is not None else np.zeros(dim)
        self.Lambda0 = Lambda if Lambda is not None else np.eye(dim) * 0.01
        self.kappa0 = kappa
        
        self.nu = self.nu0
        self.mu = self.mu0.copy()
        self.Lambda = self.Lambda0.copy()
        self.kappa = self.kappa0
        self.n = 0
        
        self.nu = max(self.nu, dim + 1)
        self._ensure_psd()

    def _ensure_psd(self) -> None:
        try:
            np.linalg.cholesky(self.Lambda)
        except np.linalg.LinAlgError:
            self.Lambda = self.Lambda + np.eye(self.dim) * 1e-6
            self._ensure_psd()

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        
        df = self.nu
        if df <= self.dim:
            return -np.inf
        
        try:
            scale_factor = (self.kappa + 1) / (self.kappa * (df - self.dim))
            Sigma = self.Lambda * scale_factor
            
            try:
                L = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                Sigma = Sigma + np.eye(self.dim) * 1e-6
                L = np.linalg.cholesky(Sigma)
            
            diff = x - self.mu
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, diff))
            mahalanobis = np.dot(diff, alpha)
            
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            log_t = (gammaln((df + self.dim) / 2)
                     - gammaln(df / 2)
                     - 0.5 * self.dim * np.log(np.pi * df)
                     - 0.5 * log_det
                     - (df + self.dim) / 2 * np.log1p(mahalanobis / df))
            
            return np.clip(log_t, -1000, 10)
            
        except Exception as e:
            return -1000

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        
        kappa_new = self.kappa + 1
        mu_new = (self.kappa * self.mu + x) / kappa_new
        
        diff = x - self.mu
        outer = np.outer(diff, diff)
        Lambda_new = self.Lambda + (self.kappa / kappa_new) * outer
        
        self.mu = mu_new
        self.Lambda = Lambda_new
        self.kappa = kappa_new
        self.nu += 1
        
        self._ensure_psd()

    def get_correlation(self) -> np.ndarray:
        scale_factor = (self.kappa + 1) / (self.kappa * max(self.nu - self.dim, 1))
        Sigma = self.Lambda * scale_factor
        std = np.sqrt(np.diag(Sigma))
        corr = Sigma / np.outer(std, std)
        return np.clip(corr, -1, 1)

    def get_covariance(self) -> np.ndarray:
        scale_factor = (self.kappa + 1) / (self.kappa * max(self.nu - self.dim, 1))
        return self.Lambda * scale_factor

    def copy(self):
        new_model = MultivariateStudentT(
            dim=self.dim,
            nu=self.nu0,
            mu=self.mu0.copy(),
            Lambda=self.Lambda0.copy(),
            kappa=self.kappa0
        )
        new_model.nu = self.nu
        new_model.mu = self.mu.copy()
        new_model.Lambda = self.Lambda.copy()
        new_model.kappa = self.kappa
        new_model.n = self.n
        return new_model


class MultiVarParticle:
    def __init__(self, run_length: int, model: MultivariateStudentT, log_weight: float = 0.0):
        self.run_length = run_length
        self.model = model
        self.log_weight = log_weight


class MultivariateBOCPD:
    def __init__(
        self,
        dim: int,
        hazard: float = 1 / 252,
        model_params: Optional[dict] = None,
        n_particles: int = 500,
        ess_threshold_ratio: float = 0.3,
        max_run_length: int = 1000
    ):
        self.dim = dim
        self.hazard = np.float64(hazard)
        self.log_hazard = np.log(self.hazard)
        self.log_1m_hazard = np.log(1 - self.hazard)
        self.model_params = model_params or {}
        self.n_particles = n_particles
        self.ess_threshold = int(n_particles * ess_threshold_ratio)
        self.max_run_length = max_run_length
        
        self.particles: List[MultiVarParticle] = []
        self._initialize_particles()
        
        self.cp_probabilities = []
        self.timestep = 0
        self.estimated_corr = []
        self.estimated_cov = []

    def _initialize_particles(self) -> None:
        self.particles = []
        for _ in range(self.n_particles):
            model = MultivariateStudentT(dim=self.dim, **self.model_params)
            self.particles.append(MultiVarParticle(
                run_length=0, 
                model=model, 
                log_weight=-np.log(self.n_particles)
            ))

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
            new_p = MultiVarParticle(
                run_length=old_p.run_length,
                model=old_p.model.copy(),
                log_weight=-np.log(self.n_particles)
            )
            new_particles.append(new_p)
        
        self.particles = new_particles

    def update(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
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
                new_particles.append(MultiVarParticle(new_run_length, new_model, log_growth))
                all_log_weights.append(log_growth)
            
            if not np.isinf(log_cp):
                new_model = MultivariateStudentT(dim=self.dim, **self.model_params)
                new_model.update(x)
                new_particles.append(MultiVarParticle(0, new_model, log_cp))
                all_log_weights.append(log_cp)
        
        if len(new_particles) == 0:
            self._initialize_particles()
            for p in self.particles:
                p.model.update(x)
            self._save_estimates()
            return 0.5
        
        if len(new_particles) > self.n_particles * 2:
            all_log_weights = np.array(all_log_weights)
            top_indices = np.argsort(all_log_weights)[-self.n_particles * 2:]
            new_particles = [new_particles[i] for i in top_indices]
        
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
        self._save_estimates()
        
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

    def _save_estimates(self) -> None:
        weights = np.exp(np.array([p.log_weight for p in self.particles]))
        weights = weights / weights.sum()
        
        avg_corr = np.zeros((self.dim, self.dim))
        avg_cov = np.zeros((self.dim, self.dim))
        
        for i, p in enumerate(self.particles):
            avg_corr += weights[i] * p.model.get_correlation()
            avg_cov += weights[i] * p.model.get_covariance()
        
        self.estimated_corr.append(avg_corr)
        self.estimated_cov.append(avg_cov)

    def get_portfolio_risk(self, weights: np.ndarray = None) -> np.ndarray:
        if weights is None:
            weights = np.ones(self.dim) / self.dim
        
        portfolio_vol = []
        for cov in self.estimated_cov:
            var = weights @ cov @ weights
            portfolio_vol.append(np.sqrt(max(var, 0)))
        
        return np.array(portfolio_vol)

    def get_avg_correlation(self) -> np.ndarray:
        avg_corr = []
        for corr in self.estimated_corr:
            upper = corr[np.triu_indices(self.dim, k=1)]
            avg_corr.append(np.mean(upper))
        return np.array(avg_corr)

    def detect_changepoints(
        self,
        data: np.ndarray,
        threshold: float = 0.5,
        min_delay: int = 10,
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


def generate_multivariate_data(
    n_samples: int = 1000,
    n_assets: int = 4,
    changepoint_locations: List[int] = None,
    corr_matrices: List[np.ndarray] = None,
    volatilities: List[np.ndarray] = None,
    seed: int = 42
) -> Tuple[np.ndarray, List[int], List[np.ndarray]]:
    np.random.seed(seed)
    
    if changepoint_locations is None:
        changepoint_locations = [300, 600]
    
    if corr_matrices is None:
        corr1 = np.array([
            [1.0, 0.3, 0.2, 0.1],
            [0.3, 1.0, 0.4, 0.2],
            [0.2, 0.4, 1.0, 0.5],
            [0.1, 0.2, 0.5, 1.0]
        ])
        corr2 = np.array([
            [1.0, 0.8, 0.7, 0.6],
            [0.8, 1.0, 0.8, 0.7],
            [0.7, 0.8, 1.0, 0.8],
            [0.6, 0.7, 0.8, 1.0]
        ])
        corr3 = np.array([
            [1.0, 0.2, 0.1, 0.0],
            [0.2, 1.0, 0.3, 0.1],
            [0.1, 0.3, 1.0, 0.4],
            [0.0, 0.1, 0.4, 1.0]
        ])
        corr_matrices = [corr1, corr2, corr3]
    
    if volatilities is None:
        vol1 = np.array([0.015, 0.02, 0.025, 0.018])
        vol2 = np.array([0.04, 0.05, 0.06, 0.045])
        vol3 = np.array([0.02, 0.025, 0.03, 0.022])
        volatilities = [vol1, vol2, vol3]
    
    changepoint_locations = [0] + changepoint_locations + [n_samples]
    returns = []
    
    for i in range(len(changepoint_locations) - 1):
        start = changepoint_locations[i]
        end = changepoint_locations[i + 1]
        n_seg = end - start
        
        corr = corr_matrices[i]
        vol = volatilities[i]
        cov = np.outer(vol, vol) * corr
        
        segment = np.random.multivariate_normal(np.zeros(n_assets), cov, n_seg)
        returns.append(segment)
    
    returns = np.vstack(returns)
    true_cps = changepoint_locations[1:-1]
    
    return returns, true_cps, corr_matrices


def generate_financial_crisis_data(
    n_samples: int = 1200,
    n_assets: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, List[int], List[str]]:
    np.random.seed(seed)
    
    true_cps = [300, 600, 900]
    regimes = ["正常期", "金融危机", "复苏期", "后危机"]
    
    corr_normal = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_normal[i, j] = corr_normal[j, i] = 0.2 + 0.1 * np.random.rand()
    
    corr_crisis = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_crisis[i, j] = corr_crisis[j, i] = 0.7 + 0.15 * np.random.rand()
    
    corr_recovery = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_recovery[i, j] = corr_recovery[j, i] = 0.4 + 0.15 * np.random.rand()
    
    corr_post = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_post[i, j] = corr_post[j, i] = 0.3 + 0.1 * np.random.rand()
    
    vol_normal = np.array([0.015, 0.018, 0.022, 0.016, 0.020])
    vol_crisis = vol_normal * 3.0
    vol_recovery = vol_normal * 1.8
    vol_post = vol_normal * 1.2
    
    corr_matrices = [corr_normal, corr_crisis, corr_recovery, corr_post]
    volatilities = [vol_normal, vol_crisis, vol_recovery, vol_post]
    
    returns, _, _ = generate_multivariate_data(
        n_samples=n_samples,
        n_assets=n_assets,
        changepoint_locations=true_cps,
        corr_matrices=corr_matrices,
        volatilities=volatilities,
        seed=seed
    )
    
    return returns, true_cps, regimes


def plot_multivariate_results(
    returns: np.ndarray,
    true_cps: List[int],
    detected_cps: List[int],
    cp_probs: np.ndarray,
    bocpd: MultivariateBOCPD,
    asset_names: List[str] = None,
    save_path: str = 'multivariate_bocpd.png'
) -> None:
    n_assets = returns.shape[1]
    if asset_names is None:
        asset_names = [f'资产{i+1}' for i in range(n_assets)]
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[2, 1.5, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(n_assets):
        ax1.plot(returns[:, i], label=asset_names[i], alpha=0.6, linewidth=0.5)
    ax1.set_ylabel('收益率')
    ax1.set_title('多资产收益率与系统性风险变点检测')
    ax1.legend(loc='upper right', ncol=n_assets)
    ax1.grid(True, alpha=0.3)
    
    for cp in true_cps:
        ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.7, linewidth=2)
    for cp in detected_cps:
        ax1.axvline(x=cp, color='green', linestyle='-', alpha=0.7, linewidth=1.5)
    
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(cp_probs, color='purple', linewidth=1)
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='阈值 (0.5)')
    ax2.set_ylabel('变点概率')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for cp in true_cps:
        ax2.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
    for cp in detected_cps:
        ax2.axvline(x=cp, color='green', linestyle='-', alpha=0.5)
    
    ax3 = fig.add_subplot(gs[2, :])
    avg_corr = bocpd.get_avg_correlation()
    ax3.plot(avg_corr, color='blue', linewidth=1.2)
    ax3.set_ylabel('平均相关性')
    ax3.grid(True, alpha=0.3)
    for cp in true_cps:
        ax3.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
    
    ax4 = fig.add_subplot(gs[3, :])
    port_vol = bocpd.get_portfolio_risk()
    ax4.plot(port_vol, color='darkred', linewidth=1.2)
    ax4.set_ylabel('投资组合波动率')
    ax4.set_xlabel('时间')
    ax4.grid(True, alpha=0.3)
    for cp in true_cps:
        ax4.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存至 {save_path}")


def plot_correlation_heatmaps(
    bocpd: MultivariateBOCPD,
    true_cps: List[int],
    asset_names: List[str] = None,
    save_path: str = 'correlation_heatmaps.png'
) -> None:
    n_assets = bocpd.dim
    if asset_names is None:
        asset_names = [f'资产{i+1}' for i in range(n_assets)]
    
    check_points = [50] + [cp + 50 for cp in true_cps]
    check_points = [cp for cp in check_points if cp < len(bocpd.estimated_corr)]
    n_plots = len(check_points)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    cmap = LinearSegmentedColormap.from_list('corr_cmap', ['#FF4444', '#FFFFFF', '#4444FF'], N=100)
    
    for idx, (ax, t) in enumerate(zip(axes, check_points)):
        corr = bocpd.estimated_corr[t]
        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            vmin=-1,
            vmax=1,
            xticklabels=asset_names,
            yticklabels=asset_names,
            cbar=idx == n_plots - 1
        )
        ax.set_title(f'时间 t={t}')
    
    plt.suptitle('相关性结构演化', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"相关性热图已保存至 {save_path}")


def test_multivariate_bocpd():
    print("=" * 70)
    print("多变量BOCPD系统性风险检测测试")
    print("=" * 70)
    
    n_samples = 1200
    n_assets = 5
    
    print(f"\n生成金融危机模拟数据 ({n_samples}天, {n_assets}个资产)...")
    returns, true_cps, regimes = generate_financial_crisis_data(
        n_samples=n_samples,
        n_assets=n_assets,
        seed=42
    )
    print(f"市场阶段: {regimes}")
    print(f"真实变点位置: {true_cps}")
    
    print("\n初始化多变量BOCPD粒子滤波...")
    bocpd_multi = MultivariateBOCPD(
        dim=n_assets,
        hazard=1/200,
        model_params={'nu': n_assets + 5, 'kappa': 1.0},
        n_particles=800,
        ess_threshold_ratio=0.3,
        max_run_length=500
    )
    
    print("运行变点检测...")
    detected_cps, cp_probs = bocpd_multi.detect_changepoints(
        returns,
        threshold=0.4,
        min_delay=30,
        verbose=True
    )
    
    print(f"\n检测到的变点位置: {detected_cps}")
    
    asset_names = ['股票', '债券', '商品', '外汇', '房地产']
    
    print("\n生成可视化结果...")
    plot_multivariate_results(
        returns, true_cps, detected_cps, cp_probs,
        bocpd_multi, asset_names=asset_names,
        save_path='multivariate_risk_detection.png'
    )
    
    plot_correlation_heatmaps(
        bocpd_multi, true_cps, asset_names=asset_names,
        save_path='correlation_evolution.png'
    )
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    
    return bocpd_multi, detected_cps, cp_probs


def portfolio_risk_monitoring_example():
    print("\n" + "=" * 70)
    print("投资组合风险监控示例")
    print("=" * 70)
    
    returns, true_cps, regimes = generate_financial_crisis_data(
        n_samples=1000, n_assets=4, seed=123
    )
    
    bocpd = MultivariateBOCPD(
        dim=4,
        hazard=1/150,
        n_particles=600
    )
    
    print("实时监控模拟...")
    risk_warnings = []
    
    for t in range(len(returns)):
        cp_prob = bocpd.update(returns[t])
        
        if cp_prob > 0.7:
            risk_warnings.append((t, cp_prob))
            if len(risk_warnings) <= 3:
                print(f"  [警告] t={t}: 高变点概率 {cp_prob:.3f} - 可能的系统性风险事件")
    
    port_vol = bocpd.get_portfolio_risk()
    print(f"\n投资组合波动率范围: [{port_vol.min():.4f}, {port_vol.max():.4f}]")
    print(f"检测到 {len(risk_warnings)} 个高风险预警")
    
    return bocpd


if __name__ == "__main__":
    test_multivariate_bocpd()
    portfolio_risk_monitoring_example()
