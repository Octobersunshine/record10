import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import norm


class OnlineBayesianChangepointDetection:
    """
    在线贝叶斯变点检测
    
    使用贝叶斯因子进行序列更新，适用于实时流数据处理
    基于 Bayesian Online Changepoint Detection 算法 (Adams & MacKay, 2007)
    """
    
    def __init__(self, hazard=0.01, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0,
                 max_run_length=1000, threshold=0.5):
        """
        初始化在线变点检测器
        
        参数:
            hazard: 危险率（在任意时刻发生变点的先验概率）
            mu0, kappa0, alpha0, beta0: 正态-逆Gamma先验的超参数
            max_run_length: 保留的最大运行长度
            threshold: 变点告警阈值
        """
        self.hazard = hazard
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.max_run_length = max_run_length
        self.threshold = threshold
        
        self.reset()
    
    def reset(self):
        """重置检测器状态"""
        self.t = 0
        self.data = deque(maxlen=self.max_run_length)
        
        self.run_length_probs = np.array([1.0])
        
        self.sufficient_stats = [(self.mu0, self.kappa0, self.alpha0, self.beta0)]
        
        self.changepoints = []
        self.changepoint_prob_history = []
        self.data_history = []
        
        self.current_max_rl = 0
    
    def _pred_prob(self, x, stats):
        """计算预测概率（使用学生t分布）"""
        mu, kappa, alpha, beta = stats
        
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        
        return norm.logpdf(x, loc=mu, scale=scale)
    
    def _update_sufficient_stats(self, x, stats):
        """更新充分统计量"""
        mu, kappa, alpha, beta = stats
        
        kappa_new = kappa + 1
        mu_new = (kappa * mu + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new = beta + (kappa * (x - mu)**2) / (2 * kappa_new)
        
        return (mu_new, kappa_new, alpha_new, beta_new)
    
    def update(self, x):
        """
        处理新数据点，更新后验概率
        
        参数:
            x: 新的数据点
            
        返回:
            当前时刻是变点的概率
        """
        self.t += 1
        self.data.append(x)
        self.data_history.append(x)
        
        T = len(self.run_length_probs)
        new_rl_probs = np.zeros(T + 1)
        
        log_pred_probs = np.zeros(T)
        for r in range(T):
            log_pred_probs[r] = self._pred_prob(x, self.sufficient_stats[r])
        
        max_logp = np.max(log_pred_probs)
        pred_probs = np.exp(log_pred_probs - max_logp)
        
        for r in range(T):
            growth_prob = self.run_length_probs[r] * pred_probs[r] * (1 - self.hazard)
            new_rl_probs[r + 1] = growth_prob
            
            cp_prob = self.run_length_probs[r] * pred_probs[r] * self.hazard
            new_rl_probs[0] += cp_prob
        
        new_rl_probs /= np.sum(new_rl_probs)
        
        if len(new_rl_probs) > self.max_run_length:
            total_tail = np.sum(new_rl_probs[self.max_run_length:])
            new_rl_probs = new_rl_probs[:self.max_run_length]
            new_rl_probs[-1] += total_tail
        
        self.run_length_probs = new_rl_probs
        
        new_stats = [(self.mu0, self.kappa0, self.alpha0, self.beta0)]
        for r in range(min(T, len(self.run_length_probs) - 1)):
            new_stat = self._update_sufficient_stats(x, self.sufficient_stats[r])
            new_stats.append(new_stat)
        
        self.sufficient_stats = new_stats
        
        cp_prob = self.run_length_probs[0]
        self.changepoint_prob_history.append(cp_prob)
        
        if cp_prob > self.threshold:
            self.changepoints.append(self.t - 1)
        
        return cp_prob
    
    def update_batch(self, data_stream):
        """
        批量处理数据流
        
        参数:
            data_stream: 可迭代的数据序列
        """
        for x in data_stream:
            self.update(x)
    
    def get_current_segment_stats(self):
        """获取当前段的统计量"""
        best_rl = np.argmax(self.run_length_probs)
        if best_rl < len(self.sufficient_stats):
            mu, kappa, alpha, beta = self.sufficient_stats[best_rl]
            return {
                'run_length': best_rl,
                'mean': mu,
                'variance': beta / (alpha - 1) if alpha > 1 else np.nan,
                'probability': self.run_length_probs[best_rl]
            }
        return None
    
    def get_bayes_factor(self):
        """
        计算变点的贝叶斯因子
        
        BF = P(变点 | 数据) / P(无变点 | 数据)
        """
        if len(self.run_length_probs) >= 2:
            p_cp = self.run_length_probs[0]
            p_no_cp = np.sum(self.run_length_probs[1:])
            if p_no_cp > 0:
                return p_cp / p_no_cp
        return 0.0
    
    def plot_results(self, figsize=(14, 10)):
        """可视化检测结果"""
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        axes[0].plot(self.data_history, 'b-', label='Data', linewidth=1)
        for cp in self.changepoints:
            axes[0].axvline(x=cp, color='r', linestyle='--', linewidth=2, alpha=0.7,
                           label='Detected Changepoint' if cp == self.changepoints[0] else "")
        axes[0].set_ylabel('Value')
        axes[0].set_title('Time Series Data with Detected Changepoints')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.changepoint_prob_history, 'r-', linewidth=2, label='Changepoint Probability')
        axes[1].fill_between(range(len(self.changepoint_prob_history)), 
                           self.changepoint_prob_history, alpha=0.3, color='red')
        axes[1].axhline(y=self.threshold, color='k', linestyle=':', alpha=0.5, 
                       label=f'Threshold = {self.threshold}')
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Changepoint Probability Over Time (Online Detection)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        
        bayes_factors = []
        for i in range(len(self.data_history)):
            temp_detector = OnlineBayesianChangepointDetection(
                hazard=self.hazard, mu0=self.mu0, kappa0=self.kappa0,
                alpha0=self.alpha0, beta0=self.beta0, threshold=self.threshold
            )
            temp_detector.update_batch(self.data_history[:i+1])
            bayes_factors.append(temp_detector.get_bayes_factor())
        
        axes[2].plot(bayes_factors, 'g-', linewidth=2, label='Bayes Factor')
        axes[2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='BF = 1 (Equal Evidence)')
        axes[2].axhline(y=3.0, color='orange', linestyle=':', alpha=0.5, label='BF = 3 (Substantial)')
        axes[2].axhline(y=10.0, color='red', linestyle=':', alpha=0.5, label='BF = 10 (Strong)')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Bayes Factor')
        axes[2].set_title('Bayes Factor for Changepoint Evidence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_summary(self):
        """打印检测摘要"""
        print("\n" + "="*70)
        print("在线贝叶斯变点检测结果摘要")
        print("="*70)
        print(f"\n处理数据点数量: {self.t}")
        print(f"检测到变点数量: {len(self.changepoints)}")
        print(f"检测阈值: {self.threshold}")
        print(f"危险率参数: {self.hazard}")
        
        if self.changepoints:
            print(f"\n检测到的变点位置: {self.changepoints}")
            print("\n各变点的后验概率:")
            for cp in self.changepoints:
                if cp < len(self.changepoint_prob_history):
                    prob = self.changepoint_prob_history[cp]
                    bf = self._calculate_bf_at_point(cp)
                    print(f"  位置 {cp}: 概率 = {prob:.4f}, 贝叶斯因子 = {bf:.2f}")
        
        current_stats = self.get_current_segment_stats()
        if current_stats:
            print(f"\n当前段统计:")
            print(f"  运行长度: {current_stats['run_length']}")
            print(f"  均值估计: {current_stats['mean']:.4f}")
            print(f"  方差估计: {current_stats['variance']:.4f}")
            print(f"  概率: {current_stats['probability']:.4f}")
        
        print("\n" + "="*70)
    
    def _calculate_bf_at_point(self, t):
        """计算某时刻的贝叶斯因子（用于摘要）"""
        if t < len(self.data_history):
            temp_detector = OnlineBayesianChangepointDetection(
                hazard=self.hazard, mu0=self.mu0, kappa0=self.kappa0,
                alpha0=self.alpha0, beta0=self.beta0
            )
            temp_detector.update_batch(self.data_history[:t+1])
            return temp_detector.get_bayes_factor()
        return 0.0


class AdaptiveOnlineChangepointDetection(OnlineBayesianChangepointDetection):
    """
    自适应在线贝叶斯变点检测
    
    根据数据统计特性自适应调整危险率参数
    """
    
    def __init__(self, hazard_min=0.001, hazard_max=0.1, adaptation_rate=0.01, **kwargs):
        super().__init__(**kwargs)
        self.hazard_min = hazard_min
        self.hazard_max = hazard_max
        self.adaptation_rate = adaptation_rate
        self.hazard_history = []
    
    def update(self, x):
        """带自适应危险率的更新"""
        if self.t > 20:
            recent_data = list(self.data)[-20:]
            recent_var = np.var(recent_data)
            recent_change = abs(x - np.mean(recent_data))
            
            if recent_var > 0:
                z_score = recent_change / np.sqrt(recent_var)
                if z_score > 2.0:
                    self.hazard = min(self.hazard_max, self.hazard * (1 + self.adaptation_rate))
                else:
                    self.hazard = max(self.hazard_min, self.hazard * (1 - self.adaptation_rate))
        
        self.hazard_history.append(self.hazard)
        
        return super().update(x)
    
    def plot_results(self, figsize=(14, 12)):
        """扩展可视化，包含危险率变化"""
        fig = super().plot_results(figsize=figsize)
        
        fig2, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.plot(self.hazard_history, 'm-', linewidth=2, label='Adaptive Hazard Rate')
        ax.set_xlabel('Time')
        ax.set_ylabel('Hazard Rate')
        ax.set_title('Adaptive Hazard Rate Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig, fig2


def generate_stream_data(n=500, changepoints=[100, 250, 380], 
                        means=[0, 5, 2, -3], stds=[1, 1.5, 1, 0.8]):
    """生成模拟流数据"""
    data = np.zeros(n)
    segments = [0] + changepoints + [n]
    
    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        data[start:end] = np.random.normal(means[i], stds[i], end - start)
    
    return data, changepoints


def demo_online_detection():
    """演示在线变点检测"""
    print("="*70)
    print("在线贝叶斯变点检测演示")
    print("="*70)
    
    np.random.seed(42)
    data, true_cp = generate_stream_data(n=300, changepoints=[80, 180, 250],
                                        means=[0, 4, 1, -2], stds=[1, 1.2, 0.9, 1.1])
    
    print(f"\n真实变点位置: {true_cp}")
    print(f"真实变点数量: {len(true_cp)}")
    
    print("\n" + "-"*70)
    print("使用标准在线变点检测器")
    print("-"*70)
    
    detector = OnlineBayesianChangepointDetection(
        hazard=0.01,
        threshold=0.4
    )
    
    print("模拟流数据处理（逐点处理）...")
    for i, x in enumerate(data):
        cp_prob = detector.update(x)
        
        if i % 50 == 0:
            bf = detector.get_bayes_factor()
            print(f"  t = {i:3d}: 变点概率 = {cp_prob:.4f}, 贝叶斯因子 = {bf:.2f}")
    
    detector.print_summary()
    
    print("\n" + "-"*70)
    print("使用自适应在线变点检测器")
    print("-"*70)
    
    adaptive_detector = AdaptiveOnlineChangepointDetection(
        hazard=0.01,
        threshold=0.4,
        hazard_min=0.001,
        hazard_max=0.05,
        adaptation_rate=0.02
    )
    
    print("模拟流数据处理（带自适应危险率）...")
    for x in data:
        adaptive_detector.update(x)
    
    adaptive_detector.print_summary()
    
    print("\n" + "="*70)
    print("贝叶斯因子解读指南")
    print("="*70)
    print("  BF < 1:     支持无变点")
    print("  1 < BF < 3: 微弱支持变点")
    print("  3 < BF < 10:实质性支持变点")
    print("  BF > 10:    强烈支持变点")
    print("  BF > 100:   决定性支持变点")
    print("\n" + "="*70)
    
    detector.plot_results()
    
    return detector, adaptive_detector


if __name__ == '__main__':
    demo_online_detection()
