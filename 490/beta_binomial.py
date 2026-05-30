import math
from typing import Literal, Optional

PriorMode = Literal["smooth", "jeffreys", "strict"]

MIN_ALPHA_BETA = 1e-6
JEFFREYS_ALPHA_BETA = 0.5


def _handle_degenerate_prior(
    alpha: float, beta: float, mode: PriorMode
) -> tuple[float, float, Optional[str]]:
    """
    处理退化的先验参数（α或β≤0）。

    Args:
        alpha: 先验参数α
        beta: 先验参数β
        mode: 处理模式
            - "smooth": 将小于MIN_ALPHA_BETA的参数平滑到MIN_ALPHA_BETA
            - "jeffreys": 将退化参数替换为Jeffreys先验(0.5, 0.5)
            - "strict": 严格模式，不允许退化参数

    Returns:
        (处理后的α, 处理后的β, 警告信息或None)
    """
    warning = None
    is_degenerate = alpha <= 0 or beta <= 0
    is_below_min = (0 < alpha < MIN_ALPHA_BETA) or (0 < beta < MIN_ALPHA_BETA)

    if mode == "strict":
        if is_degenerate or is_below_min:
            raise ValueError(
                f"先验参数α和β必须大于等于{MIN_ALPHA_BETA}，"
                f"当前α={alpha}, β={beta}"
            )
        return alpha, beta, None

    if is_degenerate:
        if mode == "jeffreys":
            if alpha <= 0:
                alpha = JEFFREYS_ALPHA_BETA
            if beta <= 0:
                beta = JEFFREYS_ALPHA_BETA
            warning = (
                f"检测到退化先验参数，已自动替换为Jeffreys先验: "
                f"α={alpha}, β={beta}"
            )
        elif mode == "smooth":
            if alpha <= 0:
                alpha = MIN_ALPHA_BETA
            if beta <= 0:
                beta = MIN_ALPHA_BETA
            warning = (
                f"检测到退化先验参数，已平滑到最小值: "
                f"α={alpha}, β={beta}"
            )
    elif is_below_min and mode == "smooth":
        if alpha < MIN_ALPHA_BETA:
            alpha = MIN_ALPHA_BETA
        if beta < MIN_ALPHA_BETA:
            beta = MIN_ALPHA_BETA
        warning = (
            f"先验参数小于最小值{MIN_ALPHA_BETA}，已自动平滑: "
            f"α={alpha}, β={beta}"
        )

    return alpha, beta, warning


def beta_binomial_update(
    alpha: float,
    beta: float,
    k: int,
    n: int,
    prior_mode: PriorMode = "smooth",
    return_warning: bool = False,
) -> tuple[tuple[float, float], float] | tuple[tuple[float, float], float, Optional[str]]:
    """
    Beta-Binomial模型的贝叶斯更新。

    处理退化先验：当α或β≤0时，根据prior_mode参数自动处理，避免无法归一化。

    Args:
        alpha: Beta先验分布的参数α
        beta: Beta先验分布的参数β
        k: 观测到的成功次数
        n: 观测到的试验次数
        prior_mode: 退化先验处理模式
            - "smooth": 平滑模式（默认），将≤0的参数设为1e-6
            - "jeffreys": 将≤0的参数替换为Jeffreys先验(0.5, 0.5)
            - "strict": 严格模式，不允许退化参数，直接抛出异常
        return_warning: 是否返回警告信息

    Returns:
        默认返回：(后验参数(α', β'), 后验预测概率)
        如果return_warning=True，返回：(后验参数, 后验预测概率, 警告信息或None)

    Raises:
        ValueError: 如果参数不合法（strict模式下退化参数也会抛出）
        TypeError: 如果k或n不是整数
    """
    if prior_mode not in ("smooth", "jeffreys", "strict"):
        raise ValueError(
            'prior_mode必须是"smooth", "jeffreys", "strict"之一'
        )

    alpha, beta, warning = _handle_degenerate_prior(alpha, beta, prior_mode)

    if n < 0:
        raise ValueError("试验次数n不能为负数")
    if k < 0 or k > n:
        raise ValueError("成功次数k必须满足0 ≤ k ≤ n")
    if not isinstance(k, int) or not isinstance(n, int):
        raise TypeError("k和n必须为整数")

    posterior_alpha = alpha + k
    posterior_beta = beta + (n - k)
    posterior_predictive = posterior_alpha / (posterior_alpha + posterior_beta)

    if return_warning:
        return (posterior_alpha, posterior_beta), posterior_predictive, warning
    return (posterior_alpha, posterior_beta), posterior_predictive


def beta_pdf(
    theta: float, alpha: float, beta: float, prior_mode: PriorMode = "smooth"
) -> float:
    """
    计算Beta分布的概率密度函数值。

    Args:
        theta: 自变量，取值范围(0, 1)
        alpha: Beta分布的参数α
        beta: Beta分布的参数β
        prior_mode: 退化先验处理模式，同beta_binomial_update

    Returns:
        Beta分布在theta处的概率密度值
    """
    if theta <= 0 or theta >= 1:
        return 0.0

    alpha, beta, _ = _handle_degenerate_prior(alpha, beta, prior_mode)

    B = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
    return (theta ** (alpha - 1)) * ((1 - theta) ** (beta - 1)) / B


if __name__ == "__main__":
    print("=" * 60)
    print("示例1: 正常先验参数")
    print("=" * 60)
    alpha_prior = 2.0
    beta_prior = 2.0
    k_obs = 7
    n_obs = 10

    (alpha_post, beta_post), pred_prob = beta_binomial_update(
        alpha_prior, beta_prior, k_obs, n_obs
    )

    print(f"先验参数: α={alpha_prior}, β={beta_prior}")
    print(f"观测数据: 成功k={k_obs}, 试验n={n_obs}")
    print(f"后验参数: α'={alpha_post}, β'={beta_post}")
    print(f"后验预测概率(下一次成功): {pred_prob:.4f}\n")

    print("=" * 60)
    print("示例2: 退化先验α=0, β=0，使用smooth模式（默认）")
    print("=" * 60)
    (alpha_post, beta_post), pred_prob, warning = beta_binomial_update(
        0, 0, 7, 10, return_warning=True
    )
    print(f"警告: {warning}")
    print(f"后验参数: α'={alpha_post}, β'={beta_post}")
    print(f"后验预测概率: {pred_prob:.4f}\n")

    print("=" * 60)
    print("示例3: 退化先验α=0, β=0，使用jeffreys模式")
    print("=" * 60)
    (alpha_post, beta_post), pred_prob, warning = beta_binomial_update(
        0, 0, 7, 10, prior_mode="jeffreys", return_warning=True
    )
    print(f"警告: {warning}")
    print(f"后验参数: α'={alpha_post}, β'={beta_post}")
    print(f"后验预测概率: {pred_prob:.4f}\n")

    print("=" * 60)
    print("示例4: 部分退化先验α=0, β=3，使用jeffreys模式")
    print("=" * 60)
    (alpha_post, beta_post), pred_prob, warning = beta_binomial_update(
        0, 3, 7, 10, prior_mode="jeffreys", return_warning=True
    )
    print(f"警告: {warning}")
    print(f"后验参数: α'={alpha_post}, β'={beta_post}")
    print(f"后验预测概率: {pred_prob:.4f}\n")

    print("=" * 60)
    print("示例5: 严格模式strict（退化参数会抛出异常）")
    print("=" * 60)
    try:
        beta_binomial_update(0, 0, 7, 10, prior_mode="strict")
    except ValueError as e:
        print(f"异常捕获: {e}")


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """
    计算正态分布的概率密度函数值。

    Args:
        x: 自变量
        mu: 均值
        sigma: 标准差（必须>0）

    Returns:
        正态分布在x处的概率密度值

    Raises:
        ValueError: 如果sigma <= 0
    """
    if sigma <= 0:
        raise ValueError("标准差sigma必须大于0")
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def normal_normal_update(
    mu0: float, sigma0: float, x_obs: float | list[float], sigma: float
) -> tuple[float, float, list[dict[str, float]] | None]:
    """
    Normal-Normal模型的贝叶斯更新（方差已知）。

    数学原理：
    - 先验: θ ~ N(mu0, sigma0²)
    - 似然: x | θ ~ N(θ, sigma²)
    - 后验: θ | x ~ N(mu_n, sigma_n²)
    其中：
        1/sigma_n² = 1/sigma0² + n/sigma²
        mu_n = sigma_n² * (mu0/sigma0² + sum(x)/sigma²)

    Args:
        mu0: 先验均值
        sigma0: 先验标准差（必须>0）
        x_obs: 观测值，可以是单个数值或列表（支持多个观测）
        sigma: 观测噪声的标准差（必须>0）

    Returns:
        (后验均值mu_n, 后验标准差sigma_n, 顺序更新历史或None)
        历史记录包含每次更新后的参数：[{"step": int, "mu": float, "sigma": float, "x_obs": float}, ...]
    """
    if sigma0 <= 0 or sigma <= 0:
        raise ValueError("标准差sigma0和sigma必须大于0")

    if isinstance(x_obs, (int, float)):
        x_list = [float(x_obs)]
    elif isinstance(x_obs, list):
        x_list = [float(x) for x in x_obs]
    else:
        raise TypeError("x_obs必须是数值或数值列表")

    history = []
    current_mu = mu0
    current_sigma = sigma0

    for i, x in enumerate(x_list):
        precision_prior = 1.0 / (current_sigma ** 2)
        precision_likelihood = 1.0 / (sigma ** 2)
        precision_post = precision_prior + precision_likelihood

        sigma_post = 1.0 / math.sqrt(precision_post)
        mu_post = sigma_post ** 2 * (
            current_mu * precision_prior + x * precision_likelihood
        )

        history.append({
            "step": i + 1,
            "x_obs": x,
            "mu": mu_post,
            "sigma": sigma_post
        })

        current_mu = mu_post
        current_sigma = sigma_post

    return current_mu, current_sigma, history


class NormalNormalSequential:
    """
    Normal-Normal模型的顺序贝叶斯更新类。

    支持逐步添加观测数据，维护后验分布状态，并可生成可视化数据。
    """

    def __init__(self, mu0: float, sigma0: float, sigma: float):
        """
        初始化顺序更新器。

        Args:
            mu0: 初始先验均值
            sigma0: 初始先验标准差
            sigma: 观测噪声标准差
        """
        if sigma0 <= 0 or sigma <= 0:
            raise ValueError("标准差必须大于0")

        self.mu0_initial = mu0
        self.sigma0_initial = sigma0
        self.sigma = sigma

        self.current_mu = mu0
        self.current_sigma = sigma0
        self.observations: list[float] = []
        self.history: list[dict[str, float]] = []

    def update(self, x_obs: float | list[float]) -> tuple[float, float]:
        """
        添加新观测并更新后验。

        Args:
            x_obs: 单个或多个观测值

        Returns:
            (更新后的均值, 更新后的标准差)
        """
        if isinstance(x_obs, (int, float)):
            x_list = [float(x_obs)]
        else:
            x_list = [float(x) for x in x_obs]

        for x in x_list:
            precision_prior = 1.0 / (self.current_sigma ** 2)
            precision_likelihood = 1.0 / (self.sigma ** 2)
            precision_post = precision_prior + precision_likelihood

            sigma_post = 1.0 / math.sqrt(precision_post)
            mu_post = sigma_post ** 2 * (
                self.current_mu * precision_prior + x * precision_likelihood
            )

            self.observations.append(x)
            self.history.append({
                "step": len(self.observations),
                "x_obs": x,
                "mu": mu_post,
                "sigma": sigma_post
            })

            self.current_mu = mu_post
            self.current_sigma = sigma_post

        return self.current_mu, self.current_sigma

    def get_posterior(self) -> tuple[float, float]:
        """获取当前后验参数。"""
        return self.current_mu, self.current_sigma

    def get_visualization_data(
        self, num_points: int = 200, num_std: float = 4.0
    ) -> dict[str, list[float] | list[dict]]:
        """
        生成可视化数据。

        Args:
            num_points: x轴采样点数
            num_std: 显示范围为均值±num_std*标准差

        Returns:
            包含以下键的字典：
            - x_values: x轴坐标列表
            - prior: 先验分布PDF值列表
            - posterior: 当前后验分布PDF值列表
            - likelihoods: 每次观测的似然曲线列表
            - history: 顺序更新历史（含每次后验参数）
            - observations: 所有观测值列表
        """
        all_mus = [self.mu0_initial] + [h["mu"] for h in self.history]
        all_sigmas = [self.sigma0_initial] + [h["sigma"] for h in self.history]

        max_mu = max(all_mus)
        min_mu = min(all_mus)
        max_sigma = max(all_sigmas)

        x_min = min_mu - num_std * max_sigma
        x_max = max_mu + num_std * max_sigma
        x_step = (x_max - x_min) / (num_points - 1)

        x_values = [x_min + i * x_step for i in range(num_points)]
        prior = [normal_pdf(x, self.mu0_initial, self.sigma0_initial) for x in x_values]
        posterior = [normal_pdf(x, self.current_mu, self.current_sigma) for x in x_values]

        likelihoods = []
        for obs in self.observations:
            likelihood_curve = [normal_pdf(x, obs, self.sigma) for x in x_values]
            likelihoods.append({
                "x_obs": obs,
                "y_values": likelihood_curve
            })

        return {
            "x_values": x_values,
            "prior": prior,
            "posterior": posterior,
            "likelihoods": likelihoods,
            "history": self.history,
            "observations": self.observations,
            "prior_mu": self.mu0_initial,
            "prior_sigma": self.sigma0_initial,
            "posterior_mu": self.current_mu,
            "posterior_sigma": self.current_sigma
        }

    def reset(self) -> None:
        """重置到初始先验状态。"""
        self.current_mu = self.mu0_initial
        self.current_sigma = self.sigma0_initial
        self.observations = []
        self.history = []


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Normal-Normal模型示例")
    print("=" * 60)

    print("\n--- 单次更新 ---")
    mu0, sigma0 = 50.0, 10.0
    x_obs = 65.0
    sigma = 5.0

    mu_post, sigma_post, history = normal_normal_update(mu0, sigma0, x_obs, sigma)
    print(f"先验: N({mu0}, {sigma0}²)")
    print(f"观测: x={x_obs}, σ={sigma}")
    print(f"后验: N({mu_post:.4f}, {sigma_post:.4f}²)")

    print("\n--- 批量更新（多个观测） ---")
    observations = [62.0, 68.0, 65.0]
    mu_post, sigma_post, history = normal_normal_update(mu0, sigma0, observations, sigma)
    print(f"先验: N({mu0}, {sigma0}²)")
    print(f"观测序列: {observations}")
    print(f"后验: N({mu_post:.4f}, {sigma_post:.4f}²)")
    print("更新历史:")
    for h in history:
        print(f"  Step {h['step']}: x={h['x_obs']:.1f} → μ={h['mu']:.4f}, σ={h['sigma']:.4f}")

    print("\n--- 顺序更新类示例 ---")
    sequential = NormalNormalSequential(mu0=50.0, sigma0=10.0, sigma=5.0)
    print(f"初始: μ={sequential.current_mu}, σ={sequential.current_sigma}")

    sequential.update(60.0)
    print(f"观测60后: μ={sequential.current_mu:.4f}, σ={sequential.current_sigma:.4f}")

    sequential.update([65.0, 62.0, 68.0])
    print(f"继续观测后: μ={sequential.current_mu:.4f}, σ={sequential.current_sigma:.4f}")

    print("\n--- 生成可视化数据 ---")
    viz_data = sequential.get_visualization_data(num_points=10)
    print(f"x轴范围: [{viz_data['x_values'][0]:.2f}, {viz_data['x_values'][-1]:.2f}]")
    print(f"采样点数: {len(viz_data['x_values'])}")
    print(f"似然曲线数量: {len(viz_data['likelihoods'])}")
    print(f"历史记录数: {len(viz_data['history'])}")
