import numpy as np
from scipy import stats, special
from typing import List, Dict, Tuple, Optional


ANDERSON_SUPPORTED_DISTS = {'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'weibull_min'}


def perform_goodness_of_fit(
    data: np.ndarray,
    dist_func: stats.rv_continuous,
    params: Tuple[float, ...],
    dist_key: str
) -> Dict:
    """
    执行Kolmogorov-Smirnov检验和Anderson-Darling检验。

    Args:
        data: 数据数组
        dist_func: scipy分布函数
        params: 分布参数元组
        dist_key: 分布标识

    Returns:
        包含检验结果的字典
    """
    gof_results = {}

    try:
        ks_stat, ks_pvalue = stats.kstest(data, dist_key, args=params)
        gof_results['ks_statistic'] = ks_stat
        gof_results['ks_pvalue'] = ks_pvalue
    except Exception:
        gof_results['ks_statistic'] = np.nan
        gof_results['ks_pvalue'] = np.nan

    ad_dist_key = dist_key
    if ad_dist_key in ANDERSON_SUPPORTED_DISTS:
        try:
            try:
                anderson_result = stats.anderson(data, dist=ad_dist_key, method='interpolate')
                gof_results['ad_statistic'] = anderson_result.statistic
                gof_results['ad_pvalue'] = anderson_result.pvalue
                gof_results['ad_critical_values'] = getattr(anderson_result, 'critical_values', None)
                gof_results['ad_significance_levels'] = getattr(anderson_result, 'significance_level', None)
            except TypeError:
                anderson_result = stats.anderson(data, dist=ad_dist_key)
                gof_results['ad_statistic'] = anderson_result.statistic
                gof_results['ad_critical_values'] = anderson_result.critical_values
                gof_results['ad_significance_levels'] = anderson_result.significance_level
                ad_pvalue = _anderson_pvalue(anderson_result.statistic, dist_key)
                gof_results['ad_pvalue'] = ad_pvalue
        except Exception:
            gof_results['ad_statistic'] = np.nan
            gof_results['ad_pvalue'] = np.nan
    else:
        gof_results['ad_statistic'] = np.nan
        gof_results['ad_pvalue'] = np.nan

    return gof_results


def _anderson_pvalue(ad_stat: float, dist_key: str) -> float:
    """
    近似计算Anderson-Darling检验的p值。

    基于文献中的近似公式计算p值上界。
    """
    if np.isnan(ad_stat):
        return np.nan

    ad_stat_adjusted = ad_stat * (1.0 + 0.75 / 100 + 2.25 / (100 ** 2))

    if ad_stat_adjusted < 0.2:
        p_value = 1.0 - np.exp(-13.436 + 101.14 * ad_stat_adjusted - 223.73 * ad_stat_adjusted ** 2)
    elif ad_stat_adjusted < 0.34:
        p_value = 1.0 - np.exp(-8.318 + 42.796 * ad_stat_adjusted - 59.938 * ad_stat_adjusted ** 2)
    elif ad_stat_adjusted < 0.6:
        p_value = np.exp(0.9177 - 4.279 * ad_stat_adjusted - 1.38 * ad_stat_adjusted ** 2)
    elif ad_stat_adjusted < 10:
        p_value = np.exp(1.2937 - 5.709 * ad_stat_adjusted + 0.0186 * ad_stat_adjusted ** 2)
    else:
        p_value = 1e-5

    return max(min(p_value, 1.0), 1e-10)


def compute_composite_rank(
    all_results: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    计算综合排名，结合AIC、KS p值、AD p值。

    排名策略：
    - AIC：越小越好，基础权重40%
    - KS p值：越大越好（不能拒绝原假设），基础权重30%
    - AD p值：越大越好（不能拒绝原假设），基础权重30%
    - 当AD检验不可用时，将KS权重提升至60%
    - 综合得分越低表示越优

    综合得分计算：
    score = (aic_rank / n) * w_aic +
            (1 - ks_pvalue_norm) * w_ks +
            (1 - ad_pvalue_norm) * w_ad

    其中pvalue_norm = min(pvalue, 0.05) / 0.05（超过0.05的p值视为等价）

    Args:
        all_results: 所有分布的拟合结果

    Returns:
        添加了排名信息的结果字典
    """
    dists = list(all_results.keys())
    n = len(dists)

    aic_values = {k: all_results[k]['aic'] for k in dists if np.isfinite(all_results[k]['aic'])}
    ks_pvalues = {k: all_results[k]['ks_pvalue'] for k in dists if np.isfinite(all_results[k].get('ks_pvalue', np.nan))}
    ad_pvalues = {k: all_results[k]['ad_pvalue'] for k in dists if np.isfinite(all_results[k].get('ad_pvalue', np.nan))}

    aic_ranks = {k: i + 1 for i, (k, _) in enumerate(sorted(aic_values.items(), key=lambda x: x[1]))}
    ks_ranks = {k: i + 1 for i, (k, _) in enumerate(sorted(ks_pvalues.items(), key=lambda x: -x[1]))}
    ad_ranks = {k: i + 1 for i, (k, _) in enumerate(sorted(ad_pvalues.items(), key=lambda x: -x[1]))}

    def normalize_pvalue(p):
        if np.isnan(p):
            return 0.0
        return min(p, 0.05) / 0.05

    for k in dists:
        aic_rank = aic_ranks.get(k, n)
        ks_rank = ks_ranks.get(k, n)
        ad_rank = ad_ranks.get(k, n)

        ks_p = all_results[k].get('ks_pvalue', np.nan)
        ad_p = all_results[k].get('ad_pvalue', np.nan)

        ks_p_norm = normalize_pvalue(ks_p)
        ad_p_norm = normalize_pvalue(ad_p)

        has_ks = np.isfinite(ks_p)
        has_ad = np.isfinite(ad_p)

        if has_ks and has_ad:
            w_aic, w_ks, w_ad = 0.4, 0.3, 0.3
        elif has_ks and not has_ad:
            w_aic, w_ks, w_ad = 0.4, 0.6, 0.0
        elif not has_ks and has_ad:
            w_aic, w_ks, w_ad = 0.4, 0.0, 0.6
        else:
            w_aic, w_ks, w_ad = 1.0, 0.0, 0.0

        composite_score = (
            (aic_rank / n) * w_aic +
            (1 - ks_p_norm) * w_ks +
            (1 - ad_p_norm) * w_ad
        )

        all_results[k]['ranking'] = {
            'aic_rank': aic_rank,
            'ks_rank': ks_rank,
            'ad_rank': ad_rank,
            'composite_score': composite_score,
            'weights_used': {'aic': w_aic, 'ks': w_ks, 'ad': w_ad},
            'has_ks': has_ks,
            'has_ad': has_ad,
        }

    sorted_by_composite = sorted(dists, key=lambda k: all_results[k]['ranking']['composite_score'])
    for rank, k in enumerate(sorted_by_composite, 1):
        all_results[k]['ranking']['overall_rank'] = rank

    return all_results


def weibull_logpdf_stable(
    x: np.ndarray,
    c: float,
    loc: float = 0.0,
    scale: float = 1.0,
    c_tol: float = 1e-3
) -> np.ndarray:
    """
    数值稳定的威布尔分布对数概率密度函数。

    当形状参数c接近1时，威布尔分布退化为指数分布，此时直接使用
    指数分布的logpdf避免(c-1)*log(...)项导致的数值不稳定。

    Args:
        x: 数据点
        c: 形状参数
        loc: 位置参数
        scale: 尺度参数
        c_tol: 判定c接近1的容差

    Returns:
        logpdf值数组
    """
    x = np.asarray(x, dtype=np.float64)
    z = (x - loc) / scale

    logpdf = np.empty_like(z, dtype=np.float64)

    valid = z > 0
    logpdf[~valid] = -np.inf

    z_valid = z[valid]

    if abs(c - 1.0) < c_tol:
        logpdf[valid] = -np.log(scale) - z_valid
        return logpdf

    log_c = np.log(c)
    log_scale = np.log(scale)

    log_z = np.log(z_valid)

    c_minus_1 = c - 1.0
    if abs(c_minus_1) < 1e-10:
        log_term = special.xlogy(c_minus_1, z_valid)
    else:
        log_term = c_minus_1 * log_z

    z_pow_c = np.power(z_valid, c)

    logpdf[valid] = log_c - log_scale + log_term - z_pow_c

    return logpdf


def fit_distributions(
    data: np.ndarray,
    distributions: Optional[List[str]] = None
) -> Tuple[str, Dict, Dict[str, Dict]]:
    """
    将一维数据拟合到常见概率分布，通过MLE估计参数，返回最适配的分布。

    Args:
        data: 一维数组，待拟合的数据
        distributions: 要拟合的分布名称列表，可选值：'norm', 'expon', 'gamma', 'weibull_min'
                       如果为None，则拟合所有四种分布

    Returns:
        best_dist_name: 最适配的分布名称
        best_dist_info: 最适配分布的信息，包含参数、对数似然值、AIC
        all_results: 所有拟合分布的详细结果字典
    """
    if distributions is None:
        distributions = ['norm', 'expon', 'gamma', 'weibull_min']

    dist_map = {
        'norm': {'name': '正态分布', 'func': stats.norm},
        'expon': {'name': '指数分布', 'func': stats.expon},
        'gamma': {'name': '伽马分布', 'func': stats.gamma},
        'weibull_min': {'name': '威布尔分布', 'func': stats.weibull_min},
    }

    data = np.asarray(data).ravel()
    n = len(data)

    if n < 2:
        raise ValueError("数据点数量不足，至少需要2个数据点")

    all_results = {}
    warnings = []

    for dist_key in distributions:
        if dist_key not in dist_map:
            raise ValueError(f"不支持的分布: {dist_key}，支持的分布: {list(dist_map.keys())}")

        dist_info = dist_map[dist_key]
        dist_func = dist_info['func']

        try:
            params = dist_func.fit(data)

            if dist_key == 'weibull_min':
                c, loc, scale = params
                logpdf_values = weibull_logpdf_stable(data, c, loc, scale)

                if abs(c - 1.0) < 1e-2:
                    msg = (f"威布尔分布形状参数c={c:.4f}接近1，"
                           f"建议使用指数分布作为替代（参数更少，更稳定）")
                    warnings.append(msg)
            else:
                logpdf_values = dist_func.logpdf(data, *params)

            valid_logpdf = logpdf_values[np.isfinite(logpdf_values)]

            if len(valid_logpdf) < n * 0.9:
                print(f"警告: {dist_info['name']} 有超过10%的数据点概率密度为无穷小，跳过该分布")
                continue

            log_likelihood = np.sum(valid_logpdf)

            k = len(params)
            aic = 2 * k - 2 * log_likelihood

            param_names = dist_func.shapes.split(', ') if dist_func.shapes else []
            param_names.extend(['loc', 'scale'])
            param_dict = dict(zip(param_names, params))

            gof = perform_goodness_of_fit(data, dist_func, params, dist_key)

            all_results[dist_key] = {
                'name': dist_info['name'],
                'params': param_dict,
                'params_tuple': params,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'num_params': k,
                'ks_statistic': gof['ks_statistic'],
                'ks_pvalue': gof['ks_pvalue'],
                'ad_statistic': gof['ad_statistic'],
                'ad_pvalue': gof['ad_pvalue'],
            }

        except Exception as e:
            print(f"拟合 {dist_info['name']} 时出错: {e}")
            continue

    if warnings:
        print("\n" + "!" * 60)
        for msg in warnings:
            print(f"! 建议: {msg}")
        print("!" * 60 + "\n")

    if not all_results:
        raise RuntimeError("所有分布拟合均失败")

    all_results = compute_composite_rank(all_results)

    best_dist_key = min(
        all_results.keys(),
        key=lambda k: all_results[k]['ranking']['composite_score']
    )
    best_dist_info = all_results[best_dist_key]
    best_dist_name = best_dist_info['name']

    return best_dist_name, best_dist_info, all_results


def print_results(best_name: str, best_info: Dict, all_results: Dict[str, Dict]) -> None:
    """
    打印拟合结果的格式化输出，包含优度检验和综合排名。
    """
    print("=" * 60)
    print("概率分布拟合结果")
    print("=" * 60)
    print(f"\n最适配分布: {best_name}")
    print(f"  综合排名: 第{best_info['ranking']['overall_rank']}名")
    print(f"  对数似然值: {best_info['log_likelihood']:.4f}")
    print(f"  AIC 值: {best_info['aic']:.4f}")

    ks_p = best_info['ks_pvalue']
    ad_p = best_info['ad_pvalue']
    ks_str = f"KS 检验: 统计量={best_info['ks_statistic']:.4f}, p值={ks_p:.4f}" if np.isfinite(ks_p) else "KS 检验: 不可用"
    ad_str = f"AD 检验: 统计量={best_info['ad_statistic']:.4f}, p值={ad_p:.4f}" if np.isfinite(ad_p) else "AD 检验: 该分布不支持"
    print(f"  {ks_str}")
    print(f"  {ad_str}")

    print(f"  参数:")
    for param_name, param_value in best_info['params'].items():
        print(f"    {param_name}: {param_value:.6f}")

    print(f"\n  显著性检验说明:")
    print(f"    KS/AD p值 > 0.05: 不能拒绝数据服从该分布的假设")
    print(f"    KS/AD p值 <= 0.05: 拒绝数据服从该分布的假设")
    print(f"    注: Anderson-Darling检验仅支持部分分布（正态、指数、威布尔等）")

    print("\n" + "-" * 60)
    print("所有分布拟合结果比较（按综合排名排序）:")
    print("-" * 60)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['ranking']['composite_score']
    )

    for i, (dist_key, info) in enumerate(sorted_results, 1):
        rank = info['ranking']
        ks_p = info['ks_pvalue']
        ad_p = info['ad_pvalue']
        has_ks = np.isfinite(ks_p)
        has_ad = np.isfinite(ad_p)

        ks_pass = "[OK]" if (has_ks and ks_p > 0.05) else ("[X]" if has_ks else "[N/A]")
        ad_pass = "[OK]" if (has_ad and ad_p > 0.05) else ("[X]" if has_ad else "[N/A]")

        ks_str = f"stat={info['ks_statistic']:.4f}, p={ks_p:.4f}" if has_ks else "不支持"
        ad_str = f"stat={info['ad_statistic']:.4f}, p={ad_p:.4f}" if has_ad else "不支持"

        weights = rank['weights_used']
        weight_str = f"权重(AIC/KS/AD)={weights['aic']:.1f}/{weights['ks']:.1f}/{weights['ad']:.1f}"

        print(f"\n{i}. {info['name']} (综合得分: {rank['composite_score']:.3f}, {weight_str})")
        print(f"   单项排名: AIC={rank['aic_rank']}, KS={rank['ks_rank']}, AD={rank['ad_rank']}")
        print(f"   对数似然: {info['log_likelihood']:.4f}, AIC: {info['aic']:.4f}")
        print(f"   KS 检验: {ks_pass} {ks_str}")
        print(f"   AD 检验: {ad_pass} {ad_str}")
        print(f"   参数: {info['params']}")

    print("\n" + "-" * 60)
    print("数据生成与风险评估建议:")
    print("-" * 60)
    top3 = sorted_results[:3]
    for i, (dist_key, info) in enumerate(top3, 1):
        ks_p = info['ks_pvalue']
        ad_p = info['ad_pvalue']
        has_ks = np.isfinite(ks_p)
        has_ad = np.isfinite(ad_p)
        ks_ok = has_ks and ks_p > 0.05
        ad_ok = has_ad and ad_p > 0.05

        available_tests = sum([has_ks, has_ad])
        passed_tests = sum([ks_ok, ad_ok])

        if available_tests >= 2 and passed_tests == 2:
            recommendation = "强烈推荐用于数据生成和风险评估（两项检验均通过）"
        elif passed_tests >= 1:
            if available_tests == 1:
                recommendation = "可考虑使用（仅一项检验可用且通过），建议结合业务场景"
            else:
                recommendation = "可考虑使用（部分检验通过），建议结合业务场景判断"
        else:
            recommendation = "不推荐，分布拟合不佳（检验未通过或不可用）"

        if not has_ad:
            recommendation += " [AD检验不支持]"

        print(f"  {i}. {info['name']}: {recommendation}")


if __name__ == "__main__":
    np.random.seed(42)

    print("测试1: 生成正态分布数据")
    data_norm = np.random.normal(loc=5.0, scale=2.0, size=1000)
    best_name, best_info, all_results = fit_distributions(data_norm)
    print_results(best_name, best_info, all_results)

    print("\n" + "=" * 60)
    print("测试2: 生成指数分布数据")
    print("=" * 60)
    data_expon = np.random.exponential(scale=3.0, size=1000)
    best_name, best_info, all_results = fit_distributions(data_expon)
    print_results(best_name, best_info, all_results)

    print("\n" + "=" * 60)
    print("测试3: 生成伽马分布数据")
    print("=" * 60)
    data_gamma = np.random.gamma(shape=2.0, scale=2.0, size=1000)
    best_name, best_info, all_results = fit_distributions(data_gamma)
    print_results(best_name, best_info, all_results)

    print("\n" + "=" * 60)
    print("测试4: 生成威布尔分布数据")
    print("=" * 60)
    data_weibull = np.random.weibull(a=1.5, size=1000) * 2.0
    best_name, best_info, all_results = fit_distributions(data_weibull)
    print_results(best_name, best_info, all_results)

    print("\n" + "=" * 60)
    print("测试5: 生成形状参数接近1的威布尔分布数据（c=1.005）")
    print("=" * 60)
    data_weibull_c1 = np.random.weibull(a=1.005, size=1000) * 3.0
    best_name, best_info, all_results = fit_distributions(data_weibull_c1)
    print_results(best_name, best_info, all_results)

    print("\n" + "=" * 60)
    print("测试6: 验证稳定logpdf与scipy原生logpdf的一致性")
    print("=" * 60)
    test_x = np.linspace(0.1, 10, 100)
    for c_test in [0.5, 0.99, 0.999, 1.0, 1.001, 1.01, 2.0]:
        stable = weibull_logpdf_stable(test_x, c_test, loc=0, scale=2)
        native = stats.weibull_min.logpdf(test_x, c_test, loc=0, scale=2)
        max_diff = np.max(np.abs(stable - native))
        print(f"  c={c_test:.3f}, 最大误差: {max_diff:.2e}")

    print("\n" + "=" * 60)
    print("测试7: 验证c→1时的数值稳定性")
    print("=" * 60)
    data_c1 = np.random.exponential(scale=3.0, size=500)
    params_weibull = stats.weibull_min.fit(data_c1)
    print(f"  拟合威布尔分布得到c={params_weibull[0]:.6f}")
    stable_ll = np.sum(weibull_logpdf_stable(data_c1, *params_weibull))
    native_ll = np.sum(stats.weibull_min.logpdf(data_c1, *params_weibull))
    print(f"  稳定实现对数似然: {stable_ll:.4f}")
    print(f"  原生实现对数似然: {native_ll:.4f}")
    if np.isfinite(stable_ll) and not np.isfinite(native_ll):
        print("  [OK] 稳定实现成功处理了数值不稳定情况！")

    print("\n" + "=" * 60)
    print("测试8: 验证c极其接近1时的数值稳定性（c=1±1e-12）")
    print("=" * 60)
    test_x_edge = np.linspace(0.001, 10, 1000)
    for c_edge in [1.0 - 1e-12, 1.0 + 1e-12, 1.0 + 1e-15]:
        stable_edge = weibull_logpdf_stable(test_x_edge, c_edge, loc=0, scale=2)
        native_edge = stats.weibull_min.logpdf(test_x_edge, c_edge, loc=0, scale=2)
        nans_stable = np.sum(~np.isfinite(stable_edge))
        nans_native = np.sum(~np.isfinite(native_edge))
        max_diff_edge = np.max(np.abs(stable_edge[np.isfinite(stable_edge) & np.isfinite(native_edge)] -
                                       native_edge[np.isfinite(stable_edge) & np.isfinite(native_edge)]))
        print(f"  c={c_edge:.15f}")
        print(f"    稳定实现NaN/inf数量: {nans_stable}")
        print(f"    原生实现NaN/inf数量: {nans_native}")
        print(f"    有效点最大误差: {max_diff_edge:.2e}")
        if nans_stable == 0 and nans_native > 0:
            print("    [OK] 稳定实现优于原生实现！")

    print("\n" + "=" * 60)
    print("测试9: scipy.special.xlogy 处理0*log(0)的稳定性")
    print("=" * 60)
    x_test = np.array([0.0, 1e-300, 1e-200, 1e-100, 0.5, 1.0, 10.0])
    y_test = np.array([0.0, 1e-300, 1e-200, 1e-100, 0.5, 1.0, 10.0])
    result_special = special.xlogy(x_test, y_test)
    result_naive = x_test * np.log(y_test)
    print(f"  x: {x_test}")
    print(f"  y: {y_test}")
    print(f"  special.xlogy: {result_special}")
    print(f"  朴素x*log(y): {result_naive}")
    print(f"  special.xlogy 有效点: {np.sum(np.isfinite(result_special))}/{len(x_test)}")
    print(f"  朴素实现有效点: {np.sum(np.isfinite(result_naive))}/{len(x_test)}")
    if np.sum(np.isfinite(result_special)) > np.sum(np.isfinite(result_naive)):
        print("  [OK] scipy.special.xlogy 成功处理了0*log(0)的不定式！")

    print("\n" + "=" * 60)
    print("测试10: 优度检验和综合排名功能验证")
    print("=" * 60)
    data_test = np.random.normal(loc=0, scale=1, size=500)
    best_name, best_info, all_results = fit_distributions(data_test)

    print("\n  各分布优度检验结果:")
    for dist_key, info in sorted(all_results.items(), key=lambda x: x[1]['ranking']['overall_rank']):
        ks_ok = info['ks_pvalue'] > 0.05
        ad_ok = info['ad_pvalue'] > 0.05
        ks_mark = "[PASS]" if ks_ok else "[FAIL]"
        ad_mark = "[PASS]" if ad_ok else "[FAIL]"
        print(f"    {info['name']}:")
        print(f"      KS 检验: {ks_mark} stat={info['ks_statistic']:.4f}, p={info['ks_pvalue']:.4f}")
        print(f"      AD 检验: {ad_mark} stat={info['ad_statistic']:.4f}, p={info['ad_pvalue']:.4f}")
        print(f"      综合排名: 第{info['ranking']['overall_rank']}名, 得分={info['ranking']['composite_score']:.3f}")

    print("\n  综合排名算法验证:")
    expected_best = 'norm'
    actual_best = min(all_results.keys(), key=lambda k: all_results[k]['ranking']['overall_rank'])
    if actual_best == expected_best:
        print(f"    [OK] 正确识别出正态分布为最优拟合")
    else:
        print(f"    [WARN] 最优拟合为{all_results[actual_best]['name']}，预期为正态分布")

    print("\n" + "=" * 60)
    print("测试11: 风险评估场景 - 模拟保险索赔数据")
    print("=" * 60)
    np.random.seed(123)
    claims_data = np.random.gamma(shape=1.5, scale=1000, size=2000)
    claims_data = claims_data[claims_data > 0]
    print(f"  模拟保险索赔数据: {len(claims_data)}条, 均值={claims_data.mean():.0f}, 中位数={np.median(claims_data):.0f}")
    best_name, best_info, all_results = fit_distributions(claims_data)
    print_results(best_name, best_info, all_results)
