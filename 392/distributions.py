import math
from scipy.special import beta as beta_func, betainc, gamma as gamma_func, digamma, polygamma


def uniform_pdf(x, a=0.0, b=1.0):
    if a >= b:
        raise ValueError("参数a必须小于b")
    return 1.0 / (b - a) if a <= x <= b else 0.0


def uniform_cdf(x, a=0.0, b=1.0):
    if a >= b:
        raise ValueError("参数a必须小于b")
    if x < a:
        return 0.0
    elif x > b:
        return 1.0
    else:
        return (x - a) / (b - a)


def _validate_beta_params(alpha, beta):
    errors = []
    if alpha <= 0:
        errors.append(f"α={alpha} 不满足 α>0")
    if beta <= 0:
        errors.append(f"β={beta} 不满足 β>0")
    if errors:
        msg = "；".join(errors)
        raise ValueError(
            f"Beta分布参数无效: {msg}。"
            f"当α或β≤0时，Beta(α,β)无法归一化（Beta函数发散），"
            f"请调整参数使 α>0 且 β>0。"
            f"建议: 均匀先验用(1,1)，弱先验用(0.5,0.5)以上的值。"
        )


def beta_pdf(x, alpha, beta):
    _validate_beta_params(alpha, beta)
    if x < 0 or x > 1:
        return 0.0
    if x == 0.0:
        if alpha < 1:
            return float('inf')
        elif alpha == 1:
            return 1.0 / beta_func(alpha, beta)
        else:
            return 0.0
    if x == 1.0:
        if beta < 1:
            return float('inf')
        elif beta == 1:
            return 1.0 / beta_func(alpha, beta)
        else:
            return 0.0
    return (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / beta_func(alpha, beta)


def beta_cdf(x, alpha, beta):
    _validate_beta_params(alpha, beta)
    if x <= 0:
        return 0.0
    elif x >= 1:
        return 1.0
    else:
        return betainc(alpha, beta, x)


def manual_beta_pdf(x, alpha, beta):
    _validate_beta_params(alpha, beta)
    if x < 0 or x > 1:
        return 0.0

    def gamma(z):
        if z <= 0:
            raise ValueError("Gamma函数参数必须大于0")
        if z == int(z):
            return math.factorial(z - 1)
        return math.gamma(z)

    B = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    if x == 0.0:
        if alpha < 1:
            return float('inf')
        elif alpha == 1:
            return 1.0 / B
        else:
            return 0.0
    if x == 1.0:
        if beta < 1:
            return float('inf')
        elif beta == 1:
            return 1.0 / B
        else:
            return 0.0
    return (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / B


def _validate_dirichlet_params(alphas):
    alphas_list = list(alphas)
    if len(alphas_list) < 2:
        raise ValueError(
            f"Dirichlet分布参数无效: α向量长度={len(alphas_list)}，"
            f"Dirichlet是Beta的高维推广，需要至少2个类别（α长度≥2）。"
        )
    errors = []
    for i, a in enumerate(alphas_list):
        if a <= 0:
            errors.append(f"α[{i}]={a} 不满足 α>0")
    if errors:
        msg = "；".join(errors)
        raise ValueError(
            f"Dirichlet分布参数无效: {msg}。"
            f"当任意α≤0时，Dirichlet(α)无法归一化（多元Beta函数发散），"
            f"请调整参数使所有 α_i>0。"
        )
    return alphas_list


def _validate_dirichlet_x(x, k):
    x_list = list(x)
    if len(x_list) != k:
        raise ValueError(
            f"Dirichlet分布维度不匹配: x长度={len(x_list)}, α长度={k}。"
            f"输入的概率向量x必须与参数α向量长度相同。"
        )
    s = sum(x_list)
    if abs(s - 1.0) > 1e-10:
        raise ValueError(
            f"Dirichlet分布输入无效: sum(x)={s}。"
            f"x必须是概率单纯形上的点，即所有x_i≥0且sum(x)=1。"
        )
    for xi in x_list:
        if xi < 0:
            raise ValueError(
                f"Dirichlet分布输入无效: x包含负值 {xi}。"
                f"所有x_i必须≥0。"
            )
    return x_list


def dirichlet_pdf(x, alphas):
    alphas_list = _validate_dirichlet_params(alphas)
    k = len(alphas_list)
    x_list = _validate_dirichlet_x(x, k)

    B_alpha = 1.0
    for a in alphas_list:
        B_alpha *= gamma_func(a)
    B_alpha /= gamma_func(sum(alphas_list))

    has_inf = False
    has_zero = False
    log_pdf = 0.0
    for xi, ai in zip(x_list, alphas_list):
        if xi == 0.0:
            if ai < 1:
                has_inf = True
            elif ai > 1:
                has_zero = True
        elif xi == 1.0:
            if ai < 1:
                has_inf = True
            elif ai > 1:
                has_zero = True
        else:
            log_pdf += (ai - 1) * math.log(xi)

    if has_inf:
        return float('inf')
    if has_zero:
        return 0.0

    return math.exp(log_pdf) / B_alpha


def beta_mean(alpha, beta):
    _validate_beta_params(alpha, beta)
    return alpha / (alpha + beta)


def beta_variance(alpha, beta):
    _validate_beta_params(alpha, beta)
    ab_sum = alpha + beta
    return (alpha * beta) / (ab_sum ** 2 * (ab_sum + 1))


def beta_stats(alpha, beta):
    _validate_beta_params(alpha, beta)
    mean = alpha / (alpha + beta)
    ab_sum = alpha + beta
    var = (alpha * beta) / (ab_sum ** 2 * (ab_sum + 1))
    return {"mean": mean, "variance": var, "std": math.sqrt(var)}


def dirichlet_mean(alphas):
    alphas_list = _validate_dirichlet_params(alphas)
    a_sum = sum(alphas_list)
    return [a / a_sum for a in alphas_list]


def dirichlet_variance(alphas):
    alphas_list = _validate_dirichlet_params(alphas)
    a_sum = sum(alphas_list)
    vars_ = []
    for a in alphas_list:
        mean_i = a / a_sum
        var_i = (mean_i * (1 - mean_i)) / (a_sum + 1)
        vars_.append(var_i)
    return vars_


def dirichlet_covariance(alphas):
    alphas_list = _validate_dirichlet_params(alphas)
    k = len(alphas_list)
    a_sum = sum(alphas_list)
    means = [a / a_sum for a in alphas_list]
    cov = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                cov[i][j] = (means[i] * (1 - means[i])) / (a_sum + 1)
            else:
                cov[i][j] = (-means[i] * means[j]) / (a_sum + 1)
    return cov


def dirichlet_stats(alphas):
    alphas_list = _validate_dirichlet_params(alphas)
    mean = dirichlet_mean(alphas_list)
    variance = dirichlet_variance(alphas_list)
    covariance = dirichlet_covariance(alphas_list)
    a_sum = sum(alphas_list)
    return {
        "mean": mean,
        "variance": variance,
        "std": [math.sqrt(v) for v in variance],
        "covariance": covariance,
        "concentration": a_sum,
    }


def beta_fit_moments(samples):
    if len(samples) < 2:
        raise ValueError("矩估计需要至少2个样本")
    for s in samples:
        if s < 0 or s > 1:
            raise ValueError(f"Beta分布样本必须在[0,1]区间内，发现{s}")

    n = len(samples)
    mu = sum(samples) / n
    var = sum((s - mu) ** 2 for s in samples) / (n - 1) if n > 1 else 0.0

    if var <= 0:
        raise ValueError(
            f"样本方差({var:.6f}) ≤ 0，无法进行矩估计。"
            f"所有样本值几乎相同（={mu:.6f}），"
            f"Beta分布退化为确定性分布。"
        )

    if var >= mu * (1 - mu):
        raise ValueError(
            f"样本方差({var:.6f})过大，"
            f"Beta分布最大方差为μ(1-μ)={mu*(1-mu):.6f}。"
            f"数据可能不服从Beta分布，或存在极端值。"
        )

    common = (mu * (1 - mu) / var) - 1
    alpha = mu * common
    beta_ = (1 - mu) * common

    return {
        "alpha": alpha,
        "beta": beta_,
        "sample_mean": mu,
        "sample_variance": var,
        "method": "moments",
    }


def beta_fit_mle(samples, max_iter=1000, tol=1e-10):
    if len(samples) < 2:
        raise ValueError("MLE需要至少2个样本")
    for s in samples:
        if s <= 0 or s >= 1:
            raise ValueError(
                f"Beta分布MLE样本必须在(0,1)开区间内，发现{s}。"
                f"边界值会导致对数似然发散。"
                f"可尝试微小扰动：用s←(s*(n-1)+0.5)/n。"
            )

    n = len(samples)
    sum_log = sum(math.log(s) for s in samples)
    sum_log1m = sum(math.log(1 - s) for s in samples)
    mean_log = sum_log / n
    mean_log1m = sum_log1m / n

    try:
        init = beta_fit_moments(samples)
        alpha = init["alpha"]
        beta_ = init["beta"]
    except ValueError:
        alpha = 1.0
        beta_ = 1.0

    if alpha <= 0 or beta_ <= 0:
        alpha = 1.0
        beta_ = 1.0

    for _ in range(max_iter):
        a_old, b_old = alpha, beta_
        ab_sum = alpha + beta_

        digamma_a = digamma(alpha)
        digamma_b = digamma(beta_)
        digamma_ab = digamma(ab_sum)

        trigamma_a = polygamma(1, alpha)
        trigamma_b = polygamma(1, beta_)
        trigamma_ab = polygamma(1, ab_sum)

        score_a = mean_log + digamma_ab - digamma_a
        score_b = mean_log1m + digamma_ab - digamma_b

        h_aa = trigamma_ab - trigamma_a
        h_bb = trigamma_ab - trigamma_b
        h_ab = trigamma_ab

        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-15:
            break

        delta_a = (h_bb * score_a - h_ab * score_b) / det
        delta_b = (h_aa * score_b - h_ab * score_a) / det

        step = 1.0
        for _ in range(50):
            new_a = alpha - step * delta_a
            new_b = beta_ - step * delta_b
            if new_a > 1e-10 and new_b > 1e-10:
                new_ab = new_a + new_b
                new_score_a = mean_log + digamma(new_ab) - digamma(new_a)
                new_score_b = mean_log1m + digamma(new_ab) - digamma(new_b)
                new_norm = new_score_a ** 2 + new_score_b ** 2
                old_norm = score_a ** 2 + score_b ** 2
                if new_norm <= old_norm + 1e-10:
                    alpha, beta_ = new_a, new_b
                    break
            step *= 0.5
        else:
            break

        if abs(alpha - a_old) < tol and abs(beta_ - b_old) < tol:
            break

    mu = sum(samples) / n
    var = sum((s - mu) ** 2 for s in samples) / (n - 1)

    return {
        "alpha": alpha,
        "beta": beta_,
        "sample_mean": mu,
        "sample_variance": var,
        "method": "mle",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("连续均匀分布示例 (a=0, b=1)")
    print("=" * 60)
    test_points = [-0.5, 0.0, 0.3, 0.5, 0.7, 1.0, 1.5]
    for x in test_points:
        print(f"x = {x:>5.1f}: PDF = {uniform_pdf(x):.4f}, CDF = {uniform_cdf(x):.4f}")

    print("\n" + "=" * 60)
    print("连续均匀分布示例 (a=2, b=5)")
    print("=" * 60)
    test_points = [1.0, 2.0, 3.0, 3.5, 5.0, 6.0]
    for x in test_points:
        print(f"x = {x:>5.1f}: PDF = {uniform_pdf(x, 2, 5):.4f}, CDF = {uniform_cdf(x, 2, 5):.4f}")

    print("\n" + "=" * 60)
    print("Beta分布示例 (α=2, β=5)")
    print("=" * 60)
    test_points = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for x in test_points:
        pdf_val = beta_pdf(x, 2, 5)
        cdf_val = beta_cdf(x, 2, 5)
        manual_pdf = manual_beta_pdf(x, 2, 5)
        print(f"x = {x:>4.1f}: PDF = {pdf_val:.6f}, CDF = {cdf_val:.6f}, 手动PDF = {manual_pdf:.6f}")

    print("\n" + "=" * 60)
    print("Beta分布示例 - 贝叶斯先验常用参数")
    print("=" * 60)
    test_cases = [
        (0.5, 1, 1, "均匀先验 (α=1, β=1)"),
        (0.3, 2, 2, "对称先验 (α=2, β=2)"),
        (0.7, 5, 1, "偏向1的先验 (α=5, β=1)"),
        (0.2, 1, 5, "偏向0的先验 (α=1, β=5)"),
        (0.4, 10, 10, "集中在0.5 (α=10, β=10)"),
    ]
    for x, alpha, beta, desc in test_cases:
        pdf_val = beta_pdf(x, alpha, beta)
        cdf_val = beta_cdf(x, alpha, beta)
        print(f"{desc}")
        print(f"  x = {x}: PDF = {pdf_val:.6f}, CDF = {cdf_val:.6f}")

    print("\n" + "=" * 60)
    print("Beta分布退化参数测试")
    print("=" * 60)
    degenerate_cases = [
        (0.5, 0.5, 0, "Beta(0.5, 0) — β=0，无法归一化"),
        (0.5, 0, 2, "Beta(0, 2) — α=0，无法归一化"),
        (0.5, -1, 2, "Beta(-1, 2) — α<0，无法归一化"),
        (0.5, 0, 0, "Beta(0, 0) — α=β=0，无法归一化"),
    ]
    for x, alpha, beta, desc in degenerate_cases:
        try:
            pdf_val = beta_pdf(x, alpha, beta)
            print(f"{desc}: PDF = {pdf_val}")
        except ValueError as e:
            print(f"{desc}: [ERROR] {e}")

    print("\n" + "=" * 60)
    print("Beta分布边界行为 (α<1 或 β<1 时的 x=0/1)")
    print("=" * 60)
    boundary_cases = [
        (0.0, 0.5, 2, "x=0, α=0.5<1 → PDF=∞"),
        (1.0, 2, 0.5, "x=1, β=0.5<1 → PDF=∞"),
        (0.0, 1, 2, "x=0, α=1 → PDF有限"),
        (1.0, 2, 1, "x=1, β=1 → PDF有限"),
        (0.0, 2, 3, "x=0, α=2>1 → PDF=0"),
        (1.0, 3, 2, "x=1, β=2>1 → PDF=0"),
    ]
    for x, alpha, beta, desc in boundary_cases:
        try:
            pdf_val = beta_pdf(x, alpha, beta)
            print(f"Beta({alpha},{beta}), {desc}: PDF = {pdf_val}")
        except ValueError as e:
            print(f"Beta({alpha},{beta}), {desc}: ✗ {e}")

    print("\n" + "=" * 60)
    print("Beta分布统计量 (均值、方差)")
    print("=" * 60)
    stat_cases = [
        (1, 1, "均匀分布"),
        (2, 5, "左偏分布"),
        (5, 2, "右偏分布"),
        (10, 10, "对称集中分布"),
    ]
    for alpha, beta, desc in stat_cases:
        stats = beta_stats(alpha, beta)
        print(f"Beta({alpha}, {beta}) - {desc}:")
        print(f"  均值 = {stats['mean']:.6f}, 方差 = {stats['variance']:.6f}, 标准差 = {stats['std']:.6f}")

    print("\n" + "=" * 60)
    print("Dirichlet分布示例")
    print("=" * 60)
    dirichlet_cases = [
        ([1, 1, 1], [1/3, 1/3, 1/3], "均匀Dirichlet (α=[1,1,1])"),
        ([2, 2, 2], [1/3, 1/3, 1/3], "对称Dirichlet (α=[2,2,2])"),
        ([5, 1, 1], [0.7, 0.2, 0.1], "偏向类别1 (α=[5,1,1])"),
        ([1, 5, 1], [0.2, 0.7, 0.1], "偏向类别2 (α=[1,5,1])"),
        ([10, 10, 10], [1/3, 1/3, 1/3], "集中对称 (α=[10,10,10])"),
    ]
    for alphas, x, desc in dirichlet_cases:
        try:
            pdf_val = dirichlet_pdf(x, alphas)
            stats = dirichlet_stats(alphas)
            print(f"\n{desc}")
            print(f"  x={[round(v,3) for v in x]}: PDF = {pdf_val:.6f}")
            print(f"  均值 = {[round(m,4) for m in stats['mean']]}")
            print(f"  方差 = {[round(v,6) for v in stats['variance']]}")
            print(f"  集中参数 = {stats['concentration']}")
        except ValueError as e:
            print(f"\n{desc}: ✗ {e}")

    print("\n" + "=" * 60)
    print("Dirichlet分布退化/边界测试")
    print("=" * 60)
    degenerate_dirichlet = [
        ([0.5, 0], [0.5, 0.5], "α包含0"),
        ([-1, 2], [0.5, 0.5], "α包含负数"),
        ([1, 1], [0.6, 0.5], "sum(x)≠1"),
        ([1, 1], [-0.1, 1.1], "x包含负值"),
        ([2, 0.5], [1.0, 0.0], "x=1对应α=2>1, x=0对应α=0.5<1→PDF=∞"),
        ([0.5, 2], [0.0, 1.0], "x=0对应α=0.5<1→PDF=∞"),
        ([3, 2], [0.0, 1.0], "x=0对应α=3>1→PDF=0"),
    ]
    for alphas, x, desc in degenerate_dirichlet:
        try:
            pdf_val = dirichlet_pdf(x, alphas)
            print(f"Dirichlet({alphas}), x={x}, {desc}: PDF = {pdf_val}")
        except ValueError as e:
            print(f"Dirichlet({alphas}), x={x}, {desc}: ✗ {e}")

    print("\n" + "=" * 60)
    print("Beta分布参数估计 - 矩估计 & MLE")
    print("=" * 60)
    import random
    random.seed(42)

    true_params = [(2, 5), (5, 2), (10, 10), (0.5, 0.5)]
    for alpha_true, beta_true in true_params:
        print(f"\n真实参数: α={alpha_true}, β={beta_true}")
        samples = []
        for _ in range(1000):
            x = random.betavariate(alpha_true, beta_true)
            if x <= 0:
                x = 1e-10
            elif x >= 1:
                x = 1 - 1e-10
            samples.append(x)

        try:
            mom_result = beta_fit_moments(samples)
            print(f"  矩估计: α={mom_result['alpha']:.4f}, β={mom_result['beta']:.4f}")
            print(f"    样本均值={mom_result['sample_mean']:.4f}, 样本方差={mom_result['sample_variance']:.6f}")
        except ValueError as e:
            print(f"  矩估计失败: {e}")

        try:
            mle_result = beta_fit_mle(samples)
            print(f"  MLE:    α={mle_result['alpha']:.4f}, β={mle_result['beta']:.4f}")
            print(f"    样本均值={mle_result['sample_mean']:.4f}, 样本方差={mle_result['sample_variance']:.6f}")
        except ValueError as e:
            print(f"  MLE失败: {e}")

    print("\n" + "=" * 60)
    print("参数估计 - 边界情况测试")
    print("=" * 60)
    edge_test_cases = [
        ([0.1, 0.2, 0.3, 0.4, 0.5], "方差正常的样本"),
        ([0.5] * 10, "方差为0的样本（所有值相同）"),
    ]
    for samples, desc in edge_test_cases:
        try:
            result = beta_fit_moments(samples)
            print(f"{desc}: 矩估计 α={result['alpha']:.4f}, β={result['beta']:.4f}")
        except ValueError as e:
            print(f"{desc}: 矩估计 ✗ {e}")

    print("\n" + "=" * 60)
    print("Dirichlet分布协方差矩阵示例")
    print("=" * 60)
    alphas_ex = [2, 3, 5]
    stats = dirichlet_stats(alphas_ex)
    print(f"α = {alphas_ex}")
    print(f"均值 = {[round(m,4) for m in stats['mean']]}")
    print("协方差矩阵:")
    for row in stats['covariance']:
        print(f"  [{', '.join(f'{v:.6f}' for v in row)}]")
