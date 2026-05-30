import numpy as np
from scipy import stats
from scipy import special


_WEIBULL_C_THRESHOLD = 0.90


def _get_distribution(name, params):
    if name == "normal":
        return stats.norm(loc=params["loc"], scale=params["scale"])
    elif name == "exponential":
        return stats.expon(loc=params["loc"], scale=params["scale"])
    elif name == "gamma":
        return stats.gamma(a=params["a"], loc=params["loc"], scale=params["scale"])
    elif name == "weibull":
        return stats.weibull_min(c=params["c"], loc=params["loc"], scale=params["scale"])
    else:
        raise ValueError(f"未知分布: {name}")


def _anderson_darling_p_value(x, dist_name, params):
    x = np.asarray(x, dtype=float)
    n = len(x)
    dist = _get_distribution(dist_name, params)
    cdf_vals = np.sort(dist.cdf(x))
    cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)

    i = np.arange(1, n + 1)
    ad_stat = -n - np.sum(
        (2 * i - 1) / n * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1]))
    )
    ad_stat *= (1 + 0.75 / n + 2.25 / (n ** 2))

    if ad_stat < 0.2:
        p_val = 1 - np.exp(-13.436 + 101.14 * ad_stat - 223.73 * ad_stat ** 2)
    elif ad_stat < 0.34:
        p_val = 1 - np.exp(-8.318 + 42.796 * ad_stat - 59.938 * ad_stat ** 2)
    elif ad_stat < 0.6:
        p_val = np.exp(0.9177 - 4.279 * ad_stat - 1.38 * ad_stat ** 2)
    elif ad_stat < 10:
        p_val = np.exp(1.2937 - 5.709 * ad_stat + 0.0186 * ad_stat ** 2)
    else:
        p_val = 1e-5

    return float(ad_stat), float(np.clip(p_val, 1e-5, 0.999))


def _goodness_of_fit(data, candidate):
    name = candidate["name"]
    params = candidate["params"]

    if name == "normal":
        test_data = data
    else:
        test_data = data[data > 0]

    if len(test_data) < 5:
        return {
            "ks_statistic": None,
            "ks_p_value": None,
            "ad_statistic": None,
            "ad_p_value": None,
        }

    dist = _get_distribution(name, params)

    try:
        ks_stat, ks_p = stats.kstest(test_data, dist.cdf, N=len(test_data))
        ks_stat = float(ks_stat)
        ks_p = float(ks_p)
    except Exception:
        ks_stat, ks_p = None, None

    try:
        ad_stat, ad_p = _anderson_darling_p_value(test_data, name, params)
    except Exception:
        ad_stat, ad_p = None, None

    return {
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "ad_statistic": ad_stat,
        "ad_p_value": ad_p,
    }


def _weibull_logpdf_sum(x, c, scale):
    x = np.asarray(x, dtype=float)
    n = len(x)
    log_x = np.log(x)
    log_scale = np.log(scale)
    x_over_scale = x / scale
    x_over_scale_c = x_over_scale ** c
    ll = (
        n * np.log(c)
        - n * c * log_scale
        + (c - 1) * np.sum(log_x)
        - np.sum(x_over_scale_c)
    )
    return ll


def _weibull_score_equations(c, x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    log_x = np.log(x)
    x_c = x ** c
    sum_log_x = np.sum(log_x)
    sum_x_c_log_x = np.sum(x_c * log_x)
    sum_x_c = np.sum(x_c)
    return (
        n / c
        + sum_log_x
        - n * sum_x_c_log_x / sum_x_c
    )


def _fit_weibull_stable(x):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    n = len(x)
    if n < 2:
        raise ValueError("正样本不足")

    c_init = 1.0
    c_guess = np.exp(
        special.polygamma(0, n) / n
        - np.sum(np.log(x)) / n
    )
    c_guess = np.clip(c_guess, 0.1, 10.0)

    try:
        from scipy import optimize
        def objective(c):
            return abs(_weibull_score_equations(c, x))
        result = optimize.minimize_scalar(
            objective,
            bounds=(0.01, 20.0),
            method="bounded",
        )
        c_opt = result.x
    except Exception:
        c_opt = c_guess

    c_opt = float(np.clip(c_opt, 0.01, 20.0))
    scale_opt = (np.sum(x ** c_opt) / n) ** (1.0 / c_opt)
    scale_opt = float(np.clip(scale_opt, 1e-10, 1e10))

    return c_opt, 0.0, scale_opt


def _compute_rankings(candidates):
    valid_ks = [c for c in candidates if c.get("ks_p_value") is not None]
    valid_ad = [c for c in candidates if c.get("ad_p_value") is not None]

    for c in candidates:
        c["ks_rank"] = None
        c["ad_rank"] = None
        c["aic_rank"] = None
        c["composite_score"] = None

    if valid_ks:
        sorted_ks = sorted(valid_ks, key=lambda c: (-c["ks_p_value"], -c["ad_p_value"] if c.get("ad_p_value") is not None else 0))
        for rank, c in enumerate(sorted_ks, 1):
            c["ks_rank"] = rank

    if valid_ad:
        sorted_ad = sorted(valid_ad, key=lambda c: -c["ad_p_value"])
        for rank, c in enumerate(sorted_ad, 1):
            c["ad_rank"] = rank

    sorted_aic = sorted(candidates, key=lambda c: c["aic"])
    for rank, c in enumerate(sorted_aic, 1):
        c["aic_rank"] = rank

    for c in candidates:
        components = []
        if c["aic_rank"] is not None:
            components.append(c["aic_rank"] * 0.4)
        if c["ks_rank"] is not None:
            components.append(c["ks_rank"] * 0.3)
        if c["ad_rank"] is not None:
            components.append(c["ad_rank"] * 0.3)
        if components:
            c["composite_score"] = float(np.mean(components))

    ranked = sorted(candidates, key=lambda c: c["composite_score"] if c["composite_score"] is not None else float("inf"))
    for rank, c in enumerate(ranked, 1):
        c["overall_rank"] = rank

    return candidates


def fit_distributions(data):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)

    if n < 2:
        raise ValueError("数据点数不足，至少需要2个有效数据")

    candidates = []

    try:
        loc, scale = stats.norm.fit(data)
        ll = np.sum(stats.norm.logpdf(data, loc=loc, scale=scale))
        k = 2
        candidates.append({
            "name": "normal",
            "params": {"loc": loc, "scale": scale},
            "log_likelihood": ll,
            "aic": 2 * k - 2 * ll,
        })
    except Exception:
        pass

    positive = data[data > 0]
    if len(positive) > 0:
        try:
            loc, scale = stats.expon.fit(positive, floc=0)
            ll = np.sum(stats.expon.logpdf(positive, loc=loc, scale=scale))
            k = 1
            candidates.append({
                "name": "exponential",
                "params": {"loc": loc, "scale": scale},
                "log_likelihood": ll,
                "aic": 2 * k - 2 * ll,
            })
        except Exception:
            pass

    if len(positive) > 0:
        try:
            a, loc, scale = stats.gamma.fit(positive, floc=0)
            ll = np.sum(stats.gamma.logpdf(positive, a=a, loc=loc, scale=scale))
            k = 2
            candidates.append({
                "name": "gamma",
                "params": {"a": a, "loc": loc, "scale": scale},
                "log_likelihood": ll,
                "aic": 2 * k - 2 * ll,
            })
        except Exception:
            pass

    if len(positive) > 0:
        try:
            c, loc, scale = _fit_weibull_stable(positive)
            ll = _weibull_logpdf_sum(positive, c, scale)
            k = 2
            weibull_result = {
                "name": "weibull",
                "params": {"c": c, "loc": 0.0, "scale": scale},
                "log_likelihood": ll,
                "aic": 2 * k - 2 * ll,
            }
            if abs(c - 1.0) < _WEIBULL_C_THRESHOLD - 0.1:
                weibull_result["note"] = "形状参数接近1，建议使用指数分布"
            candidates.append(weibull_result)
        except Exception:
            pass

    if not candidates:
        raise RuntimeError("所有分布拟合均失败，请检查数据有效性")

    for c in candidates:
        gof = _goodness_of_fit(data, c)
        c.update(gof)

    candidates = _compute_rankings(candidates)

    best = min(candidates, key=lambda c: c["composite_score"] if c["composite_score"] is not None else float("inf"))
    best_by_aic = min(candidates, key=lambda c: c["aic"])

    return {
        "best": best,
        "best_by_aic": best_by_aic,
        "all": sorted(candidates, key=lambda c: c["composite_score"] if c["composite_score"] is not None else float("inf")),
    }


def _format_pvalue(p):
    if p is None:
        return "N/A"
    if p >= 0.999:
        return ">0.999"
    if p <= 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _format_stat(s):
    return f"{s:.4f}" if s is not None else "N/A"


def print_report(result):
    print("=" * 90)
    print("分布拟合与优度检验报告")
    print("=" * 90)
    print(f"\n最佳拟合分布（综合排名）: {result['best']['name']}")
    print(f"综合排名: 第 {result['best']['overall_rank']} 名")
    print(f"参数: {result['best']['params']}")
    print(f"对数似然值: {result['best']['log_likelihood']:.4f}")
    print(f"AIC: {result['best']['aic']:.4f}")
    print(f"KS 统计量: {_format_stat(result['best'].get('ks_statistic'))}, KS p值: {_format_pvalue(result['best'].get('ks_p_value'))}")
    print(f"AD 统计量: {_format_stat(result['best'].get('ad_statistic'))}, AD p值: {_format_pvalue(result['best'].get('ad_p_value'))}")

    if "note" in result["best"]:
        print(f"备注: {result['best']['note']}")

    print(f"\n按 AIC 最佳: {result['best_by_aic']['name']} (AIC={result['best_by_aic']['aic']:.4f})")

    print("\n" + "-" * 90)
    print("所有候选分布（按综合排名）:")
    print("-" * 90)
    header = (
        f"{'排名':<6} {'分布':<14} {'AIC':>10} {'对数似然':>12} "
        f"{'KS p值':>10} {'AD p值':>10} {'参数'}"
    )
    print(header)
    print("-" * 90)

    for d in result["all"]:
        marker = " ★" if d["name"] == result["best"]["name"] else ""
        note = f"  [{d['note']}]" if "note" in d else ""
        rank = str(d["overall_rank"]) if d.get("overall_rank") else "-"

        ks_p = _format_pvalue(d.get("ks_p_value"))
        ad_p = _format_pvalue(d.get("ad_p_value"))
        param_str = ", ".join(f"{k}={v:.4f}" for k, v in d["params"].items())

        print(
            f"{rank:<6} {d['name']:<14} {d['aic']:>10.4f} {d['log_likelihood']:>12.4f} "
            f"{ks_p:>10} {ad_p:>10} {param_str}{marker}{note}"
        )

    print("-" * 90)
    print("风险评估提示:")
    print("  • KS / AD p值 > 0.05：无法拒绝原假设，分布拟合可接受")
    print("  • KS / AD p值 ≤ 0.05：拒绝原假设，分布拟合不佳")
    print("  • 综合排名权重：AIC 40% + KS 30% + AD 30%")
    print("=" * 90)


def generate_samples(result, size=1000, distribution=None):
    if distribution is None:
        distribution = result["best"]["name"]

    target = None
    for d in result["all"]:
        if d["name"] == distribution:
            target = d
            break
    if target is None:
        raise ValueError(f"分布 '{distribution}' 不在候选列表中")

    dist = _get_distribution(target["name"], target["params"])
    samples = dist.rvs(size=size)

    gof_ks = target.get("ks_p_value", None)
    gof_ad = target.get("ad_p_value", None)
    risk_warning = None
    if (gof_ks is not None and gof_ks <= 0.05) or (gof_ad is not None and gof_ad <= 0.05):
        risk_warning = f"分布拟合优度检验不通过 (KS p={_format_pvalue(gof_ks)}, AD p={_format_pvalue(gof_ad)})，生成数据存在较大风险"

    return {
        "samples": samples,
        "distribution": target["name"],
        "params": target["params"],
        "risk_warning": risk_warning,
        "ks_p_value": gof_ks,
        "ad_p_value": gof_ad,
    }


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 90)
    print("测试1: Gamma 分布数据 (shape=2.0, scale=3.0)")
    print("=" * 90)
    sample1 = np.random.gamma(shape=2.0, scale=3.0, size=500)
    result1 = fit_distributions(sample1)
    print_report(result1)

    print("\n" + "=" * 90)
    print("测试2: 指数分布数据 (scale=2.5) - 威布尔 c→1")
    print("=" * 90)
    sample2 = np.random.exponential(scale=2.5, size=500)
    result2 = fit_distributions(sample2)
    print_report(result2)

    print("\n" + "=" * 90)
    print("测试3: 威布尔分布数据 (c=1.2, scale=3.0) - 接近指数")
    print("=" * 90)
    sample3 = np.random.weibull(a=1.2, size=500) * 3.0
    result3 = fit_distributions(sample3)
    print_report(result3)

    print("\n" + "=" * 90)
    print("测试4: 数据生成与风险评估")
    print("=" * 90)
    gen_result = generate_samples(result3, size=10)
    print(f"生成分布: {gen_result['distribution']}")
    print(f"参数: {gen_result['params']}")
    print(f"样本: {gen_result['samples'][:5]}...")
    print(f"KS p值: {_format_pvalue(gen_result['ks_p_value'])}")
    print(f"AD p值: {_format_pvalue(gen_result['ad_p_value'])}")
    if gen_result['risk_warning']:
        print(f"⚠️  风险警告: {gen_result['risk_warning']}")
    else:
        print("✅ 拟合优度检验通过，生成数据风险较低")
    print("=" * 90)
