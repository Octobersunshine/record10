import numpy as np
from scipy import stats
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class StudentT:
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, mu: float = 0.0, kappa: float = 1.0):
        self.alpha0 = alpha
        self.beta0 = beta
        self.mu0 = mu
        self.kappa0 = kappa
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.kappa = kappa

    def pdf(self, data: float) -> float:
        df = 2 * self.alpha
        loc = self.mu
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        return stats.t.pdf(data, df=df, loc=loc, scale=scale)

    def logpdf(self, data: float) -> float:
        df = 2 * self.alpha
        loc = self.mu
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        return stats.t.logpdf(data, df=df, loc=loc, scale=scale)

    def update(self, data: float) -> None:
        mu_temp = (self.kappa * self.mu + data) / (self.kappa + 1)
        self.beta = self.beta + self.kappa * (data - self.mu) ** 2 / (2 * (self.kappa + 1))
        self.mu = mu_temp
        self.kappa += 1
        self.alpha += 0.5


class BOCPD:
    def __init__(
        self,
        hazard: float = 1 / 252,
        model_params: Optional[dict] = None,
        max_duration: int = 1000
    ):
        self.hazard = hazard
        self.max_duration = max_duration
        self.model_params = model_params or {}
        
        self.log_R = np.zeros((1, 1))
        self.log_R[0, 0] = 0.0
        
        self.models = []
        self._add_new_model()
        
        self.changepoints = []
        self.max_run_length = 0

    def _add_new_model(self) -> None:
        self.models.append(StudentT(**self.model_params))

    def update(self, x: float) -> np.ndarray:
        T = len(self.log_R)
        
        log_predictive = np.zeros(T)
        for i, model in enumerate(self.models):
            log_predictive[i] = model.logpdf(x)
        
        log_hazard = np.log(self.hazard)
        log_1_hazard = np.log(1 - self.hazard)
        
        log_growth = self.log_R[:, -1] + log_predictive + log_1_hazard
        
        log_cp = logsumexp(self.log_R[:, -1] + log_predictive + log_hazard)
        
        new_log_R = np.zeros((T + 1, T + 1))
        new_log_R[1:T+1, T] = log_growth
        new_log_R[0, T] = log_cp
        
        log_max = np.max(new_log_R[:, T])
        new_log_R[:, T] -= log_max
        new_log_R[:, T] = np.clip(new_log_R[:, T], -500, 0)
        
        log_norm = logsumexp(new_log_R[:, T])
        new_log_R[:, T] -= log_norm
        
        for model in self.models:
            model.update(x)
        self._add_new_model()
        
        self.log_R = new_log_R
        
        if len(self.models) > self.max_duration:
            self.models = self.models[-self.max_duration:]
            self.log_R = self.log_R[-self.max_duration:, -self.max_duration:]
        
        return np.exp(self.log_R[:, -1])

    def detect_changepoints(self, data: np.ndarray, threshold: float = 0.5) -> Tuple[list, np.ndarray]:
        n = len(data)
        run_length_probs = np.zeros((n, min(n, self.max_duration)))
        
        for t, x in enumerate(data):
            r_probs = self.update(x)
            run_length_probs[t, :len(r_probs)] = r_probs
        
        cp_probs = run_length_probs[:, 0]
        changepoints = np.where(cp_probs > threshold)[0]
        
        return list(changepoints), cp_probs


def generate_simulated_data(
    n_samples: int = 1000,
    changepoint_locations: list = None,
    volatility_levels: list = None
) -> Tuple[np.ndarray, list]:
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
        segment = np.random.normal(0, vol, end - start)
        returns.extend(segment)
    
    true_cps = changepoint_locations[1:-1]
    return np.array(returns), true_cps


def plot_results(
    returns: np.ndarray,
    true_cps: list,
    detected_cps: list,
    cp_probs: np.ndarray,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    ax1.plot(returns, color='blue', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('对数收益率')
    ax1.set_title('BOCPD波动率变点检测')
    
    for cp in true_cps:
        ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.7, label='真实变点' if cp == true_cps[0] else "")
    
    for cp in detected_cps:
        ax1.axvline(x=cp, color='green', linestyle='-', alpha=0.7, label='检测变点' if cp == detected_cps[0] else "")
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(cp_probs, color='purple', linewidth=0.8)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='阈值 (0.5)')
    ax2.set_xlabel('时间')
    ax2.set_ylabel('变点概率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bocpd_volatility_detection.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    np.random.seed(42)
    
    print("生成模拟金融数据...")
    n_samples = 1000
    true_cps = [250, 500, 750]
    volatility_levels = [0.01, 0.05, 0.01, 0.08]
    returns, true_cps = generate_simulated_data(n_samples, true_cps, volatility_levels)
    
    print(f"真实变点位置: {true_cps}")
    print(f"各阶段波动率: {volatility_levels}")
    
    print("\n初始化BOCPD检测器...")
    bocpd = BOCPD(
        hazard=1/100,
        model_params={'alpha': 0.1, 'beta': 0.001, 'mu': 0.0, 'kappa': 1.0},
        max_duration=500
    )
    
    print("运行变点检测...")
    detected_cps, cp_probs = bocpd.detect_changepoints(returns, threshold=0.5)
    
    print(f"\n检测到的变点位置: {detected_cps}")
    
    print("\n可视化结果...")
    plot_results(returns, true_cps, detected_cps, cp_probs)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
