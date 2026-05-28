import numpy as np
import math
from itertools import combinations
from typing import Callable, Optional, Tuple


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float] = None,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, np.ndarray, str]:
    """
    排列检验（非参数假设检验）

    用于检验两组样本是否来自同一分布。当总组合数较小时使用精确枚举，
    当组合数巨大时自动使用蒙特卡洛随机抽样。

    参数:
        group1: 第一组样本数据
        group2: 第二组样本数据
        statistic: 检验统计量函数，输入为两组数据，输出为统计量值
                   默认为均值差 (mean(group1) - mean(group2))
        n_permutations: 蒙特卡洛抽样次数，默认10000次
        alternative: 备择假设类型
                     - "two-sided": 双侧检验 (统计量 != 0)
                     - "greater": 单侧检验 (统计量 > 0)
                     - "less": 单侧检验 (统计量 < 0)
        random_state: 随机种子，用于可重复性

    返回:
        observed_stat: 观测到的检验统计量值
        p_value: 检验的p值
        p_value_std: p值的抽样标准差（蒙特卡洛方法的误差估计）
        null_distribution: 零分布的统计量数组
        method: 使用的方法 ("exact" 或 "monte_carlo")

    示例:
        >>> group1 = np.array([28, 30, 32, 35, 38])
        >>> group2 = np.array([22, 24, 26, 29, 31])
        >>> observed, p_val, p_std, null_dist, method = permutation_test(group1, group2, n_permutations=1000)
        >>> print(f"观测统计量: {observed:.4f}, p值: {p_val:.4f}, 方法: {method}")
    """
    if statistic is None:
        def statistic(x, y): return np.mean(x) - np.mean(y)

    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative必须是'two-sided', 'greater'或'less'")

    if random_state is not None:
        np.random.seed(random_state)

    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    observed_stat = statistic(group1, group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    n_total = len(combined)

    n_combinations = math.comb(n_total, n1)
    max_exact = n_permutations

    if n_combinations <= max_exact:
        method = "exact"
        null_distribution = np.zeros(n_combinations)

        for idx, combo in enumerate(combinations(range(n_total), n1)):
            indices = np.array(combo)
            mask = np.ones(n_total, dtype=bool)
            mask[indices] = False
            perm_group1 = combined[indices]
            perm_group2 = combined[mask]
            null_distribution[idx] = statistic(perm_group1, perm_group2)

    else:
        method = "monte_carlo"
        null_distribution = np.zeros(n_permutations)

        for i in range(n_permutations):
            permuted = np.random.permutation(combined)
            perm_group1 = permuted[:n1]
            perm_group2 = permuted[n1:]
            null_distribution[i] = statistic(perm_group1, perm_group2)

    if alternative == "greater":
        extreme_count = np.sum(null_distribution >= observed_stat)
    elif alternative == "less":
        extreme_count = np.sum(null_distribution <= observed_stat)
    else:
        extreme_count = np.sum(np.abs(null_distribution) >= np.abs(observed_stat))

    n_null = len(null_distribution)
    p_value = extreme_count / n_null

    if method == "monte_carlo":
        p_value_std = np.sqrt(p_value * (1 - p_value) / n_null)
    else:
        p_value_std = 0.0

    return observed_stat, p_value, p_value_std, null_distribution, method


def mean_diff(x: np.ndarray, y: np.ndarray) -> float:
    """均值差统计量: mean(x) - mean(y)"""
    return np.mean(x) - np.mean(y)


def median_diff(x: np.ndarray, y: np.ndarray) -> float:
    """中位数差统计量: median(x) - median(y)"""
    return np.median(x) - np.median(y)


def t_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """
    t检验统计量（假设方差不等的Welch's t）
    t = (mean(x) - mean(y)) / sqrt(var(x)/n1 + var(y)/n2)
    """
    n1, n2 = len(x), len(y)
    mean_diff = np.mean(x) - np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    se = np.sqrt(var_x / n1 + var_y / n2)
    return mean_diff / se if se > 0 else 0.0


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d效应量
    d = (mean(x) - mean(y)) / pooled_std
    """
    n1, n2 = len(x), len(y)
    mean_diff = np.mean(x) - np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_var = ((n1 - 1) * var_x + (n2 - 1) * var_y) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    return mean_diff / pooled_std if pooled_std > 0 else 0.0


def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """
    解释p值的显著性

    参数:
        p_value: 检验的p值
        alpha: 显著性水平，默认0.05

    返回:
        解释文本
    """
    if p_value < alpha:
        return f"p = {p_value:.4f} < {alpha}，在α={alpha}水平下显著，拒绝零假设"
    else:
        return f"p = {p_value:.4f} ≥ {alpha}，在α={alpha}水平下不显著，不拒绝零假设"


def interpret_effect_size(effect_size: float) -> str:
    """
    解释Cohen's d效应量大小

    参数:
        effect_size: Cohen's d值

    返回:
        效应量大小描述
    """
    abs_d = abs(effect_size)
    if abs_d < 0.2:
        size = "极小"
    elif abs_d < 0.5:
        size = "小"
    elif abs_d < 0.8:
        size = "中等"
    elif abs_d < 1.2:
        size = "大"
    else:
        size = "极大"
    return f"Cohen's d = {effect_size:.4f}，效应量{size}"


def f_statistic(groups: list) -> float:
    """
    单因素ANOVA的F统计量

    F = MS_between / MS_within
    MS_between = SS_between / df_between
    MS_within = SS_within / df_within

    参数:
        groups: 各组数据列表，如 [group1, group2, group3]

    返回:
        F统计量值
    """
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand_mean = np.mean(np.concatenate(groups))

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    df_between = k - 1
    df_within = n_total - k

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 1e-10

    return ms_between / ms_within if ms_within > 0 else 0.0


def eta_squared(groups: list) -> float:
    """
    Eta squared (η²) 效应量 - 方差分析中组间变异占总变异的比例

    η² = SS_between / SS_total
    取值范围: [0, 1]

    参数:
        groups: 各组数据列表

    返回:
        η²效应量值
    """
    grand_mean = np.mean(np.concatenate(groups))

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((np.concatenate(groups) - grand_mean) ** 2)

    return ss_between / ss_total if ss_total > 0 else 0.0


def omega_squared(groups: list) -> float:
    """
    Omega squared (ω²) 效应量 - 修正的η²，对小样本有更少偏差

    ω² = (SS_between - df_between * MS_within) / (SS_total + MS_within)

    参数:
        groups: 各组数据列表

    返回:
        ω²效应量值
    """
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand_mean = np.mean(np.concatenate(groups))

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = n_total - k
    ms_within = ss_within / df_within if df_within > 0 else 0

    numerator = ss_between - df_between * ms_within
    denominator = ss_total + ms_within

    return max(0.0, numerator / denominator) if denominator > 0 else 0.0


def median_anova_statistic(groups: list) -> float:
    """
    基于中位数的ANOVA统计量 - 对异常值更稳健

    统计量 = 组间中位数的方差 / 组内中位数绝对偏差的均值

    参数:
        groups: 各组数据列表

    返回:
        中位数ANOVA统计量值
    """
    k = len(groups)
    medians = [np.median(g) for g in groups]
    grand_median = np.median(np.concatenate(groups))

    between_var = np.var(medians, ddof=1) if k > 1 else 0
    within_mad = np.mean([np.median(np.abs(g - np.median(g))) for g in groups])

    return between_var / within_mad if within_mad > 0 else 0.0


def permutation_anova(
    groups: list,
    statistic: Callable[[list], float] = None,
    effect_size: Callable[[list], float] = None,
    n_permutations: int = 10000,
    ci_level: float = 0.95,
    n_bootstrap: int = 2000,
    random_state: Optional[int] = None,
) -> dict:
    """
    置换方差分析（Permutation ANOVA）- 多组非参数比较

    通过随机打乱组标签生成零分布，检验多组间是否存在显著差异。

    参数:
        groups: 各组数据列表，如 [group1, group2, group3]
        statistic: 检验统计量函数，输入为组列表，输出为统计量值
                   默认为 F 统计量
        effect_size: 效应量函数，输入为组列表，输出为效应量值
                     默认为 eta_squared (η²)
        n_permutations: 置换次数，默认10000次
        ci_level: 效应量置信区间水平，默认0.95
        n_bootstrap: Bootstrap抽样次数用于计算效应量置信区间，默认2000次
        random_state: 随机种子，用于可重复性

    返回:
        包含以下键的字典:
            - observed_stat: 观测统计量值
            - p_value: 检验p值
            - p_value_std: p值的抽样标准差
            - effect_size: 观测效应量值
            - effect_size_ci: 效应量置信区间 (low, high)
            - null_distribution: 零分布数组
            - method: 使用的方法 ("monte_carlo")
            - statistic_name: 统计量名称
            - effect_size_name: 效应量名称

    示例:
        >>> group1 = np.random.normal(0, 1, 30)
        >>> group2 = np.random.normal(0.5, 1, 30)
        >>> group3 = np.random.normal(1, 1, 30)
        >>> result = permutation_anova([group1, group2, group3], n_permutations=1000)
        >>> print(f"F={result['observed_stat']:.2f}, p={result['p_value']:.4f}")
    """
    if statistic is None:
        statistic = f_statistic
        stat_name = "F"
    else:
        stat_name = statistic.__name__

    if effect_size is None:
        effect_size = eta_squared
        effect_name = "eta_squared"
    else:
        effect_name = effect_size.__name__

    if random_state is not None:
        np.random.seed(random_state)

    groups = [np.asarray(g) for g in groups]
    k = len(groups)
    if k < 2:
        raise ValueError("至少需要两组数据")

    n_per_group = [len(g) for g in groups]
    n_total = sum(n_per_group)

    combined = np.concatenate(groups)
    group_labels = np.concatenate([np.full(n, i) for i, n in enumerate(n_per_group)])

    observed_stat = statistic(groups)
    observed_es = effect_size(groups)

    null_distribution = np.zeros(n_permutations)

    for i in range(n_permutations):
        permuted_labels = np.random.permutation(group_labels)
        permuted_groups = [combined[permuted_labels == j] for j in range(k)]
        null_distribution[i] = statistic(permuted_groups)

    p_value = np.mean(null_distribution >= observed_stat)
    p_value_std = np.sqrt(p_value * (1 - p_value) / n_permutations)

    es_bootstrap = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_groups = []
        for g in groups:
            boot_idx = np.random.randint(0, len(g), len(g))
            boot_groups.append(g[boot_idx])
        es_bootstrap[i] = effect_size(boot_groups)

    alpha = 1 - ci_level
    ci_low = np.percentile(es_bootstrap, 100 * alpha / 2)
    ci_high = np.percentile(es_bootstrap, 100 * (1 - alpha / 2))

    return {
        "observed_stat": observed_stat,
        "p_value": p_value,
        "p_value_std": p_value_std,
        "effect_size": observed_es,
        "effect_size_ci": (ci_low, ci_high),
        "null_distribution": null_distribution,
        "method": "monte_carlo",
        "statistic_name": stat_name,
        "effect_size_name": effect_name,
        "n_groups": k,
        "n_per_group": n_per_group,
        "bootstrap_es": es_bootstrap,
    }


def interpret_anova_effect_size(es: float, es_name: str = "η²") -> str:
    """
    解释ANOVA效应量大小

    参数:
        es: 效应量值 (η²或ω²)
        es_name: 效应量名称

    返回:
        效应量大小描述
    """
    if es < 0.01:
        size = "极小"
    elif es < 0.06:
        size = "小"
    elif es < 0.14:
        size = "中等"
    else:
        size = "大"
    return f"{es_name} = {es:.4f}，效应量{size}"


def tukey_hsd_permutation(groups: list, n_permutations: int = 10000, alpha: float = 0.05) -> list:
    """
    置换检验版事后多重比较（Bonferroni校正）

    对所有组对进行两两组间比较，并进行Bonferroni多重比较校正。

    参数:
        groups: 各组数据列表
        n_permutations: 每组比较的置换次数
        alpha: 显著性水平

    返回:
        比较结果列表，每项包含组索引、均值差、p值、校正后显著性
    """
    k = len(groups)
    n_comparisons = k * (k - 1) / 2
    alpha_corrected = alpha / n_comparisons if n_comparisons > 0 else alpha
    results = []

    for i in range(k):
        for j in range(i + 1, k):
            _, p_val, _, _, _ = permutation_test(
                groups[i], groups[j],
                statistic=mean_diff,
                n_permutations=n_permutations,
                alternative="two-sided",
            )
            mean_d = np.mean(groups[i]) - np.mean(groups[j])
            significant = p_val < alpha_corrected
            results.append({
                "comparison": f"Group {i+1} vs Group {j+1}",
                "group_i": i,
                "group_j": j,
                "mean_diff": mean_d,
                "p_value": p_val,
                "p_value_bonferroni": min(1.0, p_val * n_comparisons),
                "significant": significant,
                "alpha_corrected": alpha_corrected,
            })
    return results


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("排列检验示例 (精确检验 vs 蒙特卡洛抽样)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("场景1: 小样本 (n1=5, n2=5), 总组合数 C(10,5) = 252, 使用精确检验")
    print("=" * 70)

    group1_small = np.array([28, 30, 32, 35, 38])
    group2_small = np.array([22, 24, 26, 29, 31])

    n1, n2 = len(group1_small), len(group2_small)
    n_total = n1 + n2
    n_combs = math.comb(n_total, n1)
    print(f"\n组1 (n={n1}): {group1_small}")
    print(f"组2 (n={n2}): {group2_small}")
    print(f"总组合数 C({n_total},{n1}) = {n_combs}")

    observed, p_val, p_std, null_dist, method = permutation_test(
        group1_small, group2_small,
        statistic=mean_diff,
        n_permutations=10000,
        alternative="two-sided",
        random_state=42
    )
    print(f"\n方法: {'精确枚举' if method == 'exact' else '蒙特卡洛抽样'}")
    print(f"观测均值差: {observed:.4f}")
    print(f"p值: {p_val:.6f}")
    print(f"p值标准差: {p_std:.6f}")
    print(interpret_p_value(p_val))

    print("\n" + "=" * 70)
    print("场景2: 大样本 (n1=30, n2=25), 总组合数巨大, 使用蒙特卡洛抽样")
    print("=" * 70)

    group1 = np.random.normal(loc=10, scale=2, size=30)
    group2 = np.random.normal(loc=12, scale=2, size=25)

    n1, n2 = len(group1), len(group2)
    n_total = n1 + n2
    n_combs = math.comb(n_total, n1)
    print(f"\n组1 (n={n1}): 均值={np.mean(group1):.2f}, 标准差={np.std(group1, ddof=1):.2f}")
    print(f"组2 (n={n2}): 均值={np.mean(group2):.2f}, 标准差={np.std(group2, ddof=1):.2f}")
    print(f"总组合数 C({n_total},{n1}) = {n_combs:.2e} (约 {n_combs})")

    observed, p_val, p_std, null_dist, method = permutation_test(
        group1, group2,
        statistic=mean_diff,
        n_permutations=10000,
        alternative="two-sided",
        random_state=42
    )
    ci_low = max(0.0, p_val - 1.96 * p_std)
    ci_high = min(1.0, p_val + 1.96 * p_std)
    print(f"\n方法: {'精确枚举' if method == 'exact' else '蒙特卡洛抽样'}")
    print(f"观测均值差: {observed:.4f}")
    print(f"p值: {p_val:.6f}")
    print(f"p值标准差 (抽样误差): {p_std:.6f}")
    print(f"p值 95%置信区间: [{ci_low:.6f}, {ci_high:.6f}]")
    print(interpret_p_value(p_val))

    print("\n--- 效应量分析 ---")
    d = cohens_d(group1, group2)
    print(interpret_effect_size(d))

    print("\n" + "-" * 70)
    print("示例3: 使用t统计量，单侧检验（组1 < 组2）")
    print("-" * 70)
    observed_t, p_val_t, p_std_t, _, method_t = permutation_test(
        group1, group2,
        statistic=t_statistic,
        n_permutations=10000,
        alternative="less",
        random_state=42
    )
    print(f"方法: {'精确枚举' if method_t == 'exact' else '蒙特卡洛抽样'}")
    print(f"观测t统计量: {observed_t:.4f}")
    print(f"p值: {p_val_t:.6f}")
    print(f"p值标准差: {p_std_t:.6f}")
    print(interpret_p_value(p_val_t))

    print("\n" + "-" * 70)
    print("示例4: 使用中位数差统计量")
    print("-" * 70)
    observed_med, p_val_med, p_std_med, _, method_med = permutation_test(
        group1, group2,
        statistic=median_diff,
        n_permutations=10000,
        alternative="two-sided",
        random_state=42
    )
    print(f"方法: {'精确枚举' if method_med == 'exact' else '蒙特卡洛抽样'}")
    print(f"观测中位数差: {observed_med:.4f}")
    print(f"p值: {p_val_med:.6f}")
    print(f"p值标准差: {p_std_med:.6f}")
    print(interpret_p_value(p_val_med))

    print("\n" + "=" * 70)
    print("零分布摘要 (蒙特卡洛抽样)")
    print("=" * 70)
    print(f"抽样次数: {len(null_dist)}")
    print(f"零分布均值: {np.mean(null_dist):.4f}")
    print(f"零分布标准差: {np.std(null_dist):.4f}")
    print(f"零分布2.5%分位数: {np.percentile(null_dist, 2.5):.4f}")
    print(f"零分布97.5%分位数: {np.percentile(null_dist, 97.5):.4f}")

    print("\n" + "=" * 70)
    print("抽样误差说明:")
    print("=" * 70)
    print("蒙特卡洛p值服从二项分布 B(n, p)，其标准差为 sqrt(p*(1-p)/n)")
    print("95%置信区间为 p ± 1.96 * sqrt(p*(1-p)/n)")
    print(f"当前n={len(null_dist)}, p={p_val:.6f}")
    print(f"标准误 SE = {p_std:.6f}")
    print(f"95% CI = [{ci_low:.6f}, {ci_high:.6f}]")
    print("\n若需要更高精度，可增加 n_permutations 参数:")
    print("  - n=1000:  SE ≈ 0.005 (当p=0.05时)")
    print("  - n=10000: SE ≈ 0.0015 (当p=0.05时)")
    print("  - n=100000: SE ≈ 0.0005 (当p=0.05时)")

    print("\n" + "=" * 70)
    print("置换方差分析 (Permutation ANOVA) - 四组比较")
    print("=" * 70)

    np.random.seed(42)
    group_a = np.random.normal(loc=5.0, scale=1.0, size=25)
    group_b = np.random.normal(loc=5.5, scale=1.0, size=25)
    group_c = np.random.normal(loc=6.5, scale=1.0, size=25)
    group_d = np.random.normal(loc=7.0, scale=1.0, size=25)
    groups = [group_a, group_b, group_c, group_d]

    print(f"\n各组描述统计:")
    for i, g in enumerate(groups):
        print(f"  组{i+1} (n={len(g)}): 均值={np.mean(g):.2f}, SD={np.std(g, ddof=1):.2f}, 中位数={np.median(g):.2f}")

    print("\n--- 示例1: 标准F检验置换 ANOVA (eta^2效应量) ---")
    result = permutation_anova(
        groups,
        statistic=f_statistic,
        effect_size=eta_squared,
        n_permutations=5000,
        ci_level=0.95,
        n_bootstrap=2000,
        random_state=42
    )

    es_low, es_high = result["effect_size_ci"]
    print(f"统计量: {result['statistic_name']} = {result['observed_stat']:.4f}")
    print(f"p值: {result['p_value']:.6f} (SE = {result['p_value_std']:.6f})")
    print(interpret_p_value(result['p_value']))
    print(f"\n效应量: {interpret_anova_effect_size(result['effect_size'], result['effect_size_name'])}")
    print(f"效应量 95% CI: [{es_low:.4f}, {es_high:.4f}]")

    print("\n--- 示例2: 使用omega^2效应量 (更保守的估计) ---")
    result2 = permutation_anova(
        groups,
        statistic=f_statistic,
        effect_size=omega_squared,
        n_permutations=5000,
        ci_level=0.95,
        n_bootstrap=2000,
        random_state=42
    )
    es_low2, es_high2 = result2["effect_size_ci"]
    print(f"效应量: {interpret_anova_effect_size(result2['effect_size'], 'omega_squared')}")
    print(f"效应量 95% CI: [{es_low2:.4f}, {es_high2:.4f}]")

    print("\n--- 示例3: 稳健ANOVA - 中位数统计量 ---")
    result3 = permutation_anova(
        groups,
        statistic=median_anova_statistic,
        effect_size=eta_squared,
        n_permutations=5000,
        ci_level=0.95,
        n_bootstrap=2000,
        random_state=42
    )
    es_low3, es_high3 = result3["effect_size_ci"]
    print(f"稳健统计量 = {result3['observed_stat']:.4f}")
    print(f"p值: {result3['p_value']:.6f}")
    print(interpret_p_value(result3['p_value']))

    print("\n--- 事后多重比较 (Bonferroni校正) ---")
    posthoc = tukey_hsd_permutation(groups, n_permutations=5000, alpha=0.05)
    print(f"比较次数: {len(posthoc)}, Bonferroni校正α = {posthoc[0]['alpha_corrected']:.4f}")
    for comp in posthoc:
        sig_mark = "*" if comp["significant"] else "ns"
        print(f"  {comp['comparison']}: 均值差={comp['mean_diff']:+.2f}, "
              f"原始p={comp['p_value']:.4f}, Bonferroni p={comp['p_value_bonferroni']:.4f} {sig_mark}")

    print("\n" + "=" * 70)
    print("零分布摘要 (Permutation ANOVA)")
    print("=" * 70)
    print(f"置换次数: {len(result['null_distribution'])}")
    print(f"零分布均值: {np.mean(result['null_distribution']):.4f}")
    print(f"零分布中位数: {np.median(result['null_distribution']):.4f}")
    print(f"零分布95%分位数: {np.percentile(result['null_distribution'], 95):.4f}")
    print(f"观测F值: {result['observed_stat']:.4f}")
    print(f"p值: {result['p_value']:.6f}")

    print("\n" + "=" * 70)
    print("效应量置信区间说明:")
    print("=" * 70)
    print("通过Bootstrap方法对每组独立重抽样，计算效应量的经验分布")
    print(f"Bootstrap次数: {result['n_groups']}组 × {len(result['bootstrap_es'])}次重抽样")
    print(f"eta^2 观测值: {result['effect_size']:.4f}")
    print(f"eta^2 Bootstrap均值: {np.mean(result['bootstrap_es']):.4f}")
    print(f"eta^2 Bootstrap中位数: {np.median(result['bootstrap_es']):.4f}")
    print(f"eta^2 95% CI: [{es_low:.4f}, {es_high:.4f}]")
    print("\neta^2解释标准:")
    print("  < 0.01: 极小效应")
    print("  0.01 - 0.06: 小效应")
    print("  0.06 - 0.14: 中等效应")
    print("  > 0.14: 大效应")
