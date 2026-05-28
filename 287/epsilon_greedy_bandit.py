import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from abc import ABC, abstractmethod


class MultiArmedBandit:
    def __init__(self, k: int, reward_means: List[float], reward_stds: List[float] = None):
        self.k = k
        self.reward_means = np.array(reward_means)
        self.reward_stds = np.array(reward_stds) if reward_stds else np.ones(k)
        self.best_mean = np.max(self.reward_means)

    def pull(self, arm: int) -> float:
        return np.random.normal(self.reward_means[arm], self.reward_stds[arm])


class BanditAlgorithm(ABC):
    @abstractmethod
    def select_arm(self) -> int:
        pass

    @abstractmethod
    def update(self, arm: int, reward: float):
        pass

    @abstractmethod
    def get_best_arm(self) -> int:
        pass


class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, k: int, epsilon: float = 0.1, strategy: str = 'fixed',
                 decay_rate: float = 0.0, inverse_t_scale: float = 1.0):
        self.k = k
        self.epsilon = epsilon
        self.strategy = strategy
        self.decay_rate = decay_rate
        self.inverse_t_scale = inverse_t_scale
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.total_counts = 0

    def get_current_epsilon(self) -> float:
        if self.strategy == 'fixed':
            return self.epsilon
        elif self.strategy == 'inverse_t':
            return min(1.0, self.inverse_t_scale / (self.total_counts + 1))
        elif self.strategy == 'exponential':
            return self.epsilon * np.exp(-self.decay_rate * self.total_counts)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def select_arm(self) -> int:
        current_epsilon = self.get_current_epsilon()
        if np.random.random() < current_epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.values)

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.total_counts += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

    def get_best_arm(self) -> int:
        return np.argmax(self.values)


class UCB(BanditAlgorithm):
    def __init__(self, k: int, c: float = 2.0):
        self.k = k
        self.c = c
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.total_counts = 0

    def select_arm(self) -> int:
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.total_counts) / self.counts
        )
        return np.argmax(ucb_values)

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.total_counts += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

    def get_best_arm(self) -> int:
        return np.argmax(self.values)

    def get_ucb_values(self) -> np.ndarray:
        if self.total_counts == 0:
            return np.zeros(self.k)
        ucb_values = np.zeros(self.k)
        for arm in range(self.k):
            if self.counts[arm] > 0:
                ucb_values[arm] = self.values[arm] + self.c * np.sqrt(
                    np.log(self.total_counts) / self.counts[arm]
                )
        return ucb_values


class ThompsonSampling(BanditAlgorithm):
    def __init__(self, k: int, prior_mu: float = 0.0, prior_nu: float = 1.0,
                 prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.k = k
        self.prior_mu = prior_mu
        self.prior_nu = prior_nu
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        self.counts = np.zeros(k)
        self.sum_rewards = np.zeros(k)
        self.sum_squared_rewards = np.zeros(k)

        self.mu = np.ones(k) * prior_mu
        self.nu = np.ones(k) * prior_nu
        self.alpha = np.ones(k) * prior_alpha
        self.beta = np.ones(k) * prior_beta

    def select_arm(self) -> int:
        samples = np.zeros(self.k)
        for arm in range(self.k):
            tau = np.random.gamma(self.alpha[arm], 1.0 / self.beta[arm])
            sigma = 1.0 / np.sqrt(self.nu[arm] * tau)
            samples[arm] = np.random.normal(self.mu[arm], sigma)
        return np.argmax(samples)

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.sum_squared_rewards[arm] += reward ** 2

        n = self.counts[arm]
        mu0 = self.prior_mu
        nu0 = self.prior_nu
        alpha0 = self.prior_alpha
        beta0 = self.prior_beta

        x_bar = self.sum_rewards[arm] / n

        self.nu[arm] = nu0 + n
        self.mu[arm] = (nu0 * mu0 + n * x_bar) / self.nu[arm]
        self.alpha[arm] = alpha0 + n / 2.0

        sum_sq_dev = self.sum_squared_rewards[arm] - n * x_bar ** 2
        self.beta[arm] = beta0 + 0.5 * sum_sq_dev + \
            (nu0 * n * (x_bar - mu0) ** 2) / (2 * (nu0 + n))

    def get_best_arm(self) -> int:
        return np.argmax(self.mu)

    def get_expected_values(self) -> np.ndarray:
        return self.mu


def run_experiment(bandit: MultiArmedBandit, algorithm: BanditAlgorithm,
                   num_steps: int) -> Tuple[List[float], List[float], dict]:
    cumulative_regret = 0.0
    cumulative_reward = 0.0
    regret_curve = []
    reward_curve = []
    extra_data = {}

    for step in range(num_steps):
        arm = algorithm.select_arm()
        reward = bandit.pull(arm)
        algorithm.update(arm, reward)

        regret = bandit.best_mean - bandit.reward_means[arm]
        cumulative_regret += regret
        cumulative_reward += reward

        reward_curve.append(cumulative_reward)
        regret_curve.append(cumulative_regret)

    return reward_curve, regret_curve, extra_data


def run_multi_experiment(bandit: MultiArmedBandit, algorithm_factory,
                         num_steps: int, num_runs: int = 20):
    all_rewards = []
    all_regrets = []
    final_recommendations = []

    for run in range(num_runs):
        algorithm = algorithm_factory()
        reward_curve, regret_curve, _ = run_experiment(bandit, algorithm, num_steps)
        all_rewards.append(reward_curve)
        all_regrets.append(regret_curve)
        final_recommendations.append(algorithm.get_best_arm())

    avg_rewards = np.mean(all_rewards, axis=0)
    avg_regrets = np.mean(all_regrets, axis=0)
    std_regrets = np.std(all_regrets, axis=0)

    return avg_rewards, avg_regrets, std_regrets, final_recommendations


def main():
    np.random.seed(42)

    k = 10
    reward_means = np.random.normal(0, 1, k)
    print(f"各臂的真实奖励均值: {reward_means}")
    print(f"最优臂: {np.argmax(reward_means)}, 奖励均值: {np.max(reward_means):.4f}")

    bandit = MultiArmedBandit(k, reward_means)
    num_steps = 1000
    num_runs = 20

    algorithms = [
        ('ε-Greedy (ε=0.1)', lambda: EpsilonGreedy(k, epsilon=0.1, strategy='fixed')),
        ('ε-Greedy (ε=5/t)', lambda: EpsilonGreedy(k, strategy='inverse_t', inverse_t_scale=5.0)),
        ('UCB (c=1.0)', lambda: UCB(k, c=1.0)),
        ('UCB (c=2.0)', lambda: UCB(k, c=2.0)),
        ('Thompson Sampling', lambda: ThompsonSampling(k)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    results = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (name, factory) in enumerate(algorithms):
        avg_rewards, avg_regrets, std_regrets, recommendations = run_multi_experiment(
            bandit, factory, num_steps, num_runs
        )
        results[name] = {
            'rewards': avg_rewards,
            'regrets': avg_regrets,
            'std_regrets': std_regrets,
            'recommendations': recommendations,
            'color': colors[idx]
        }

        best_arm_count = recommendations.count(np.argmax(reward_means))
        print(f"\n{name}:")
        print(f"  最终累积奖励: {avg_rewards[-1]:.2f}")
        print(f"  最终累积遗憾: {avg_regrets[-1]:.2f}")
        print(f"  正确识别最优臂比例: {best_arm_count}/{num_runs} ({100*best_arm_count/num_runs:.1f}%)")

    for name, data in results.items():
        axes[0, 0].plot(data['regrets'], label=name, color=data['color'], linewidth=2)
        axes[0, 1].plot(data['rewards'], label=name, color=data['color'], linewidth=2)

    axes[0, 0].set_xlabel('Steps', fontsize=10)
    axes[0, 0].set_ylabel('Cumulative Regret', fontsize=10)
    axes[0, 0].set_title('Cumulative Regret Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Steps', fontsize=10)
    axes[0, 1].set_ylabel('Cumulative Reward', fontsize=10)
    axes[0, 1].set_title('Cumulative Reward Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    algorithm_names = list(results.keys())
    final_regrets = [results[name]['regrets'][-1] for name in algorithm_names]
    final_rewards = [results[name]['rewards'][-1] for name in algorithm_names]
    bar_colors = [results[name]['color'] for name in algorithm_names]

    x_pos = np.arange(len(algorithm_names))
    bar_width = 0.35

    axes[1, 0].bar(x_pos - bar_width/2, final_regrets, bar_width, color=bar_colors, alpha=0.7)
    axes[1, 0].set_xlabel('Algorithms', fontsize=10)
    axes[1, 0].set_ylabel('Final Cumulative Regret', fontsize=10)
    axes[1, 0].set_title('Final Regret Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(algorithm_names, rotation=45, ha='right', fontsize=7)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].axis('off')
    best_strategy = max(results.keys(), key=lambda k: results[k]['rewards'][-1])
    axes[1, 1].text(0.05, 0.95, 'Performance Summary (Final Step):',
                    fontsize=11, fontweight='bold', transform=axes[1, 1].transAxes)

    y_pos = 0.88
    for name, data in results.items():
        marker = '★ ' if name == best_strategy else '  '
        best_arm_count = data['recommendations'].count(np.argmax(reward_means))
        accuracy = 100 * best_arm_count / num_runs

        axes[1, 1].text(0.05, y_pos, f'{marker}{name}:', fontsize=9,
                        fontweight='bold' if name == best_strategy else 'normal',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.08, y_pos - 0.05,
                        f'Reward={data["rewards"][-1]:.1f}, Regret={data["regrets"][-1]:.1f}, Accuracy={accuracy:.0f}%',
                        fontsize=8, transform=axes[1, 1].transAxes)
        y_pos -= 0.12

    plt.tight_layout()
    plt.savefig('e:/temp/record10/287/algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    regret_data = {
        'steps': np.arange(num_steps),
    }
    for name, data in results.items():
        regret_data[name] = data['regrets']

    np.savez('e:/temp/record10/287/regret_curve_data.npz', **regret_data)

    print(f"\n对比图像已保存至: e:\\temp\\record10\\287\\algorithm_comparison.png")
    print(f"遗憾曲线数据已保存至: e:\\temp\\record10\\287\\regret_curve_data.npz")
    print(f"\n最佳策略: {best_strategy}")

    print("\n" + "="*60)
    print("各算法最终累积遗憾排名（越低越好）:")
    print("="*60)
    sorted_by_regret = sorted(results.items(), key=lambda x: x[1]['regrets'][-1])
    for rank, (name, data) in enumerate(sorted_by_regret, 1):
        print(f"{rank}. {name}: {data['regrets'][-1]:.2f}")


if __name__ == "__main__":
    main()
