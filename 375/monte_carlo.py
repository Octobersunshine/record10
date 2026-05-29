import numpy as np
import warnings
from scipy import stats
from typing import Callable, Dict, List, Tuple, Optional


DEFAULT_MIN_SAMPLES = 10000
DEFAULT_CONVERGENCE_THRESHOLD = 0.001
DEFAULT_WINDOW_SIZE = 1000
DEFAULT_HISTOGRAM_BINS = 50
SUPPORTED_SAMPLING_METHODS = ['random', 'lhs']


class MonteCarloSimulator:
    def __init__(
        self,
        distributions: Dict[str, Dict],
        predict_fn: Callable,
        seed: int = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        window_size: int = DEFAULT_WINDOW_SIZE,
        sampling_method: str = 'lhs',
        histogram_bins: int = DEFAULT_HISTOGRAM_BINS
    ):
        """
        初始化蒙特卡洛模拟器

        Args:
            distributions: 输入变量的概率分布定义
                格式: {
                    'var_name': {
                        'type': 'normal' | 'uniform',
                        'params': {参数...}
                    },
                    ...
                }
                正态分布参数: {'mean': μ, 'std': σ}
                均匀分布参数: {'low': 最小值, 'high': 最大值}
            predict_fn: 预测函数，输入为变量字典，输出为标量或数组
            seed: 随机种子，用于可重复性
            min_samples: 最小采样次数建议（默认10000）
            convergence_threshold: 收敛判断的相对误差阈值（默认0.001，即0.1%）
            window_size: 移动平均窗口大小（默认1000）
            sampling_method: 采样方法，'random'（简单随机采样）或 'lhs'（拉丁超立方采样，默认）
            histogram_bins: 直方图的区间数（默认50）
        """
        if sampling_method not in SUPPORTED_SAMPLING_METHODS:
            raise ValueError(
                f"不支持的采样方法: {sampling_method}。"
                f"支持的方法: {SUPPORTED_SAMPLING_METHODS}"
            )

        self.distributions = distributions
        self.predict_fn = predict_fn
        self.min_samples = min_samples
        self.convergence_threshold = convergence_threshold
        self.window_size = window_size
        self.sampling_method = sampling_method
        self.histogram_bins = histogram_bins
        if seed is not None:
            np.random.seed(seed)
        self._seed = seed

    def _sample_from_distribution(
        self,
        dist_spec: Dict,
        uniform_samples: np.ndarray
    ) -> np.ndarray:
        """
        根据分布定义和均匀分布样本生成变量样本

        Args:
            dist_spec: 分布定义
            uniform_samples: [0, 1]区间的均匀分布样本

        Returns:
            转换后的变量样本数组
        """
        dist_type = dist_spec['type']
        params = dist_spec['params']

        if dist_type == 'normal':
            return stats.norm.ppf(
                uniform_samples,
                loc=params['mean'],
                scale=params['std']
            )
        elif dist_type == 'uniform':
            return stats.uniform.ppf(
                uniform_samples,
                loc=params['low'],
                scale=params['high'] - params['low']
            )
        else:
            raise ValueError(f"不支持的分布类型: {dist_type}")

    def _generate_lhs_samples(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        生成拉丁超立方采样（LHS）样本

        拉丁超立方采样将每个变量的分布空间划分为n个等概率区间，
        从每个区间中随机采样一个点，并随机排列各变量的采样顺序，
        确保采样空间被均匀覆盖。

        Args:
            n_samples: 采样次数

        Returns:
            变量名到样本数组的字典
        """
        var_names = list(self.distributions.keys())
        n_vars = len(var_names)

        permutations = np.zeros((n_vars, n_samples), dtype=int)
        for i in range(n_vars):
            permutations[i] = np.random.permutation(n_samples)

        u = np.random.uniform(size=(n_vars, n_samples))
        stratified_u = (permutations + u) / n_samples

        samples = {}
        for i, var_name in enumerate(var_names):
            samples[var_name] = self._sample_from_distribution(
                self.distributions[var_name],
                stratified_u[i]
            )

        return samples

    def _generate_random_samples(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        生成简单随机采样样本

        Args:
            n_samples: 采样次数

        Returns:
            变量名到样本数组的字典
        """
        samples = {}
        for var_name, dist_spec in self.distributions.items():
            dist_type = dist_spec['type']
            params = dist_spec['params']

            if dist_type == 'normal':
                samples[var_name] = np.random.normal(
                    params['mean'], params['std'], n_samples
                )
            elif dist_type == 'uniform':
                samples[var_name] = np.random.uniform(
                    params['low'], params['high'], n_samples
                )
            else:
                raise ValueError(f"不支持的分布类型: {dist_type}")

        return samples

    def _generate_all_inputs(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        根据选择的采样方法生成所有输入变量样本

        Args:
            n_samples: 采样次数

        Returns:
            变量名到样本数组的字典
        """
        if self.sampling_method == 'lhs':
            return self._generate_lhs_samples(n_samples)
        else:
            return self._generate_random_samples(n_samples)

    def _check_min_samples(self, n_samples: int) -> None:
        """检查采样次数是否达到最小建议值"""
        if n_samples < self.min_samples:
            warnings.warn(
                f"采样次数({n_samples})小于建议的最小采样次数({self.min_samples})，"
                f"结果可能不稳定。建议增加采样次数以提高结果可靠性。",
                UserWarning,
                stacklevel=2
            )

    def _compute_convergence_diagnostics(
        self,
        samples: np.ndarray
    ) -> Dict:
        """
        计算收敛性诊断指标

        Args:
            samples: 输出样本数组

        Returns:
            收敛性诊断字典:
            - 'checkpoints': 检查点采样数列表
            - 'mean_trajectory': 各检查点对应的均值
            - 'moving_average': 移动平均序列
            - 'moving_std': 移动标准差序列
            - 'is_converged': 是否收敛
            - 'convergence_point': 收敛时的采样数（若未收敛则为None）
            - 'relative_error': 最终相对误差
            - 'std_error': 标准误 (std / sqrt(n))
            - 'recommended_samples': 达到收敛建议的采样次数
        """
        n = len(samples)

        checkpoints = self._generate_checkpoints(n)
        mean_trajectory = [
            float(np.mean(samples[:cp])) for cp in checkpoints
        ]

        moving_average, moving_std = self._compute_moving_stats(
            samples, self.window_size
        )

        is_converged, convergence_point, relative_error = \
            self._assess_convergence(
                mean_trajectory, checkpoints, self.convergence_threshold
            )

        std_error = float(np.std(samples) / np.sqrt(n))

        recommended_samples = self._estimate_required_samples(
            samples, self.convergence_threshold
        )

        return {
            'checkpoints': checkpoints,
            'mean_trajectory': mean_trajectory,
            'moving_average': moving_average,
            'moving_std': moving_std,
            'is_converged': is_converged,
            'convergence_point': convergence_point,
            'relative_error': relative_error,
            'std_error': std_error,
            'recommended_samples': recommended_samples
        }

    @staticmethod
    def _generate_checkpoints(n: int) -> List[int]:
        """生成检查点，用于观察均值轨迹"""
        if n <= 100:
            return list(range(1, n + 1))

        checkpoints = []
        if n >= 1000:
            checkpoints.extend([10, 20, 50, 100, 200, 500])

        step = max(n // 100, 1)
        start = max(checkpoints[-1] + 1, 1) if checkpoints else 1
        checkpoints.extend(range(start, n + 1, step))

        if checkpoints[-1] != n:
            checkpoints.append(n)

        return [cp for cp in checkpoints if cp <= n]

    @staticmethod
    def _compute_moving_stats(
        samples: np.ndarray,
        window_size: int
    ) -> tuple:
        """
        计算移动平均和移动标准差

        Args:
            samples: 样本数组
            window_size: 窗口大小

        Returns:
            (moving_average, moving_std) 元组
        """
        n = len(samples)
        if n < window_size:
            window_size = max(n // 10, 1)

        cumsum = np.cumsum(samples, dtype=np.float64)
        cumsum_sq = np.cumsum(samples ** 2, dtype=np.float64)

        moving_average = []
        moving_std = []

        for i in range(n):
            if i < window_size:
                ma = cumsum[i] / (i + 1)
                var = (cumsum_sq[i] / (i + 1)) - ma ** 2
            else:
                ma = (cumsum[i] - cumsum[i - window_size]) / window_size
                var = (
                    (cumsum_sq[i] - cumsum_sq[i - window_size]) / window_size
                    - ma ** 2
                )
            moving_average.append(float(ma))
            moving_std.append(float(np.sqrt(max(var, 0))))

        return moving_average, moving_std

    @staticmethod
    def _assess_convergence(
        mean_trajectory: List[float],
        checkpoints: List[int],
        threshold: float
    ) -> tuple:
        """
        评估收敛性

        Args:
            mean_trajectory: 均值轨迹
            checkpoints: 检查点
            threshold: 相对误差阈值

        Returns:
            (is_converged, convergence_point, relative_error)
        """
        if len(mean_trajectory) < 10:
            return False, None, float('inf')

        final_mean = mean_trajectory[-1]

        if abs(final_mean) < 1e-10:
            absolute_errors = [abs(m - final_mean) for m in mean_trajectory]
            relative_error = float(max(absolute_errors[-5:]))
        else:
            relative_errors = [
                abs(m - final_mean) / abs(final_mean)
                for m in mean_trajectory
            ]
            relative_error = float(max(relative_errors[-5:]))

        is_converged = relative_error < threshold

        convergence_point = None
        if is_converged:
            for i in range(len(mean_trajectory) - 5, -1, -1):
                window_errors = [
                    abs(mean_trajectory[j] - final_mean) / abs(final_mean)
                    if abs(final_mean) >= 1e-10
                    else abs(mean_trajectory[j] - final_mean)
                    for j in range(i, min(i + 5, len(mean_trajectory)))
                ]
                if all(e < threshold for e in window_errors):
                    convergence_point = checkpoints[i]
                else:
                    break

        return is_converged, convergence_point, relative_error

    @staticmethod
    def _estimate_required_samples(
        samples: np.ndarray,
        threshold: float
    ) -> int:
        """
        估计达到指定收敛阈值所需的采样次数

        使用中心极限定理估计：n ≈ (z * σ / ε)²
        其中 z 取2（95%置信度），σ 为样本标准差，ε 为绝对误差
        """
        std = np.std(samples)
        mean = np.mean(samples)

        if abs(mean) < 1e-10:
            absolute_epsilon = threshold
        else:
            absolute_epsilon = abs(mean) * threshold

        z_score = 2.0
        estimated_n = int(np.ceil((z_score * std / absolute_epsilon) ** 2))

        return max(estimated_n, len(samples))

    def _compute_histogram(
        self,
        samples: np.ndarray
    ) -> Dict:
        """
        计算直方图数据

        Args:
            samples: 输出样本数组

        Returns:
            直方图数据字典:
            - 'counts': 每个区间的频数
            - 'bin_edges': 区间边界
            - 'bin_centers': 区间中心
            - 'bin_width': 区间宽度
            - 'density': 归一化的密度（面积为1）
        """
        counts, bin_edges = np.histogram(
            samples,
            bins=self.histogram_bins,
            density=False
        )
        density, _ = np.histogram(
            samples,
            bins=bin_edges,
            density=True
        )

        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_centers': bin_centers.tolist(),
            'bin_width': float(bin_width),
            'density': density.tolist()
        }

    def run(self, n_samples: int) -> Dict:
        """
        运行N次蒙特卡洛采样

        Args:
            n_samples: 采样次数

        Returns:
            包含输出统计信息的字典:
            - 'samples': 所有输出样本
            - 'mean': 均值
            - 'std': 标准差
            - 'min': 最小值
            - 'max': 最大值
            - 'percentiles': {5: 5分位数, 25: 25分位数, 50: 中位数, 75: 75分位数, 95: 95分位数}
            - 'convergence': 收敛性诊断结果
            - 'histogram': 直方图数据
            - 'sampling_method': 使用的采样方法
        """
        self._check_min_samples(n_samples)

        input_samples = self._generate_all_inputs(n_samples)

        outputs = []
        for i in range(n_samples):
            inputs = {
                var_name: input_samples[var_name][i]
                for var_name in input_samples.keys()
            }
            output = self.predict_fn(**inputs)
            outputs.append(output)

        samples = np.array(outputs)

        convergence = self._compute_convergence_diagnostics(samples)
        histogram = self._compute_histogram(samples)

        return {
            'samples': samples,
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'min': float(np.min(samples)),
            'max': float(np.max(samples)),
            'percentiles': {
                5: float(np.percentile(samples, 5)),
                25: float(np.percentile(samples, 25)),
                50: float(np.percentile(samples, 50)),
                75: float(np.percentile(samples, 75)),
                95: float(np.percentile(samples, 95))
            },
            'convergence': convergence,
            'histogram': histogram,
            'sampling_method': self.sampling_method
        }


def run_monte_carlo(
    distributions: Dict[str, Dict],
    predict_fn: Callable,
    n_samples: int,
    seed: int = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    window_size: int = DEFAULT_WINDOW_SIZE,
    sampling_method: str = 'lhs',
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS
) -> Dict:
    """
    便捷函数：运行蒙特卡洛模拟

    Args:
        distributions: 输入变量的概率分布定义
        predict_fn: 预测函数
        n_samples: 采样次数
        seed: 随机种子
        min_samples: 最小采样次数建议
        convergence_threshold: 收敛判断的相对误差阈值
        window_size: 移动平均窗口大小
        sampling_method: 采样方法，'random'或'lhs'（默认）
        histogram_bins: 直方图的区间数

    Returns:
        统计结果字典
    """
    simulator = MonteCarloSimulator(
        distributions,
        predict_fn,
        seed,
        min_samples,
        convergence_threshold,
        window_size,
        sampling_method,
        histogram_bins
    )
    return simulator.run(n_samples)


def print_stats(
    result: Dict,
    show_convergence: bool = True,
    show_percentiles: bool = True,
    show_histogram: bool = False
) -> None:
    """
    打印统计结果

    Args:
        result: run() 方法返回的结果字典
        show_convergence: 是否显示收敛性诊断信息
        show_percentiles: 是否显示分位数表
        show_histogram: 是否显示直方图（ASCII形式）
    """
    print("=" * 60)
    print("蒙特卡洛模拟结果")
    print("=" * 60)
    print(f"样本数: {len(result['samples'])}")
    print(f"采样方法: {result['sampling_method'].upper()}")
    print("-" * 60)
    print(f"均值:   {result['mean']:.6f}")
    print(f"标准差: {result['std']:.6f}")
    print(f"最小值: {result['min']:.6f}")
    print(f"最大值: {result['max']:.6f}")

    if show_percentiles:
        print("-" * 60)
        print("分位数表:")
        print(f"  {'分位数':<10} {'值':<15}")
        print(f"  {'-'*9:<10} {'-'*14:<15}")
        pcts = sorted(result['percentiles'].items())
        for p, v in pcts:
            pct_label = f"P{p:02d}"
            if p == 5:
                pct_label += " (0.05)"
            elif p == 50:
                pct_label += " (0.50)"
            elif p == 95:
                pct_label += " (0.95)"
            print(f"  {pct_label:<10} {v:<15.6f}")

    if show_convergence and 'convergence' in result:
        conv = result['convergence']
        print("-" * 60)
        print("收敛性诊断:")
        print(f"  标准误:         {conv['std_error']:.8f}")
        print(f"  相对误差:       {conv['relative_error']:.8f}")
        print(f"  是否收敛:       {'是' if conv['is_converged'] else '否'}")
        if conv['convergence_point']:
            print(f"  收敛采样点:     {conv['convergence_point']}")
        print(f"  建议采样次数:   {conv['recommended_samples']}")

        if not conv['is_converged']:
            print(f"  ⚠ 警告: 当前采样尚未收敛，建议增加采样次数。")

    if show_histogram and 'histogram' in result:
        hist = result['histogram']
        print("-" * 60)
        print("直方图 (频率分布):")
        max_count = max(hist['counts'])
        scale = 30 / max_count if max_count > 0 else 1
        for i in range(len(hist['counts'])):
            if hist['counts'][i] > 0:
                bar = '█' * int(hist['counts'][i] * scale)
                print(
                    f"  [{hist['bin_edges'][i]:>8.2f}, "
                    f"{hist['bin_edges'][i+1]:>8.2f}) "
                    f"{bar} ({hist['counts'][i]})"
                )

    print("=" * 60)


def print_percentile_table(result: Dict) -> None:
    """
    打印格式化的分位数表（重点显示0.05, 0.5, 0.95分位数）

    Args:
        result: run() 方法返回的结果字典
    """
    print("=" * 50)
    print("输出变量分位数表")
    print("=" * 50)
    print(f"{'分位数':<12} {'概率':<10} {'值':<15}")
    print("-" * 50)

    key_percentiles = {
        5: 0.05,
        25: 0.25,
        50: 0.50,
        75: 0.75,
        95: 0.95
    }

    for p in [5, 25, 50, 75, 95]:
        v = result['percentiles'][p]
        marker = "  ← 关键分位数" if p in [5, 50, 95] else ""
        print(
            f"{'P' + str(p):<12} {key_percentiles[p]:<10.2f} "
            f"{v:<15.6f}{marker}"
        )
    print("=" * 50)


if __name__ == "__main__":
    def example_predict(x: float, y: float) -> float:
        """示例预测函数：计算两个变量的和的平方"""
        return (x + y) ** 2

    distributions = {
        'x': {
            'type': 'normal',
            'params': {'mean': 0, 'std': 1}
        },
        'y': {
            'type': 'uniform',
            'params': {'low': -1, 'high': 1}
        }
    }

    print("=" * 70)
    print("测试1: 不同采样次数下的收敛对比 (LHS vs Random)")
    print("=" * 70)
    print()

    true_mean = 4 / 3
    print(f"理论均值 (E[(x+y)²]): {true_mean:.6f}")
    print()

    for n_test in [1000, 5000, 20000]:
        result_random = run_monte_carlo(
            distributions=distributions,
            predict_fn=example_predict,
            n_samples=n_test,
            seed=42,
            sampling_method='random',
            min_samples=1
        )
        result_lhs = run_monte_carlo(
            distributions=distributions,
            predict_fn=example_predict,
            n_samples=n_test,
            seed=42,
            sampling_method='lhs',
            min_samples=1
        )

        error_random = abs(result_random['mean'] - true_mean)
        error_lhs = abs(result_lhs['mean'] - true_mean)

        print(f"N={n_test:>5}: "
              f"Random均值={result_random['mean']:.6f} (误差={error_random:.6f}), "
              f"LHS均值={result_lhs['mean']:.6f} (误差={error_lhs:.6f}), "
              f"LHS改进={(error_random - error_lhs) / error_random * 100:+.1f}%")

    print()
    print("=" * 70)
    print("测试2: 拉丁超立方采样（推荐10000次）")
    print("=" * 70)
    result = run_monte_carlo(
        distributions=distributions,
        predict_fn=example_predict,
        n_samples=10000,
        seed=42,
        sampling_method='lhs',
        histogram_bins=20
    )
    print_stats(result, show_convergence=True, show_percentiles=True)
    print()
    print_percentile_table(result)
    print()
    print("直方图数据:")
    hist = result['histogram']
    print(f"  区间数: {len(hist['counts'])}")
    print(f"  区间宽度: {hist['bin_width']:.4f}")
    print(f"  区间边界: {[round(b, 2) for b in hist['bin_edges'][:5]]}...")
    print(f"  频数统计: {hist['counts'][:5]}...")
    print(f"  密度统计: {[round(d, 4) for d in hist['density'][:5]]}...")

    print()
    print("=" * 70)
    print("测试3: 验证直方图数据可用于可视化")
    print("=" * 70)
    print("result['histogram'] 包含:")
    print("  - counts: 每个区间的样本数量")
    print("  - bin_edges: 区间边界数组")
    print("  - bin_centers: 区间中心数组")
    print("  - bin_width: 区间宽度")
    print("  - density: 归一化密度（面积为1）")
    print()
    print("使用示例:")
    print("  import matplotlib.pyplot as plt")
    print("  plt.bar(hist['bin_centers'], hist['density'], width=hist['bin_width'])")
    print("  plt.show()")
