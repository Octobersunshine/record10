import math


_LOG_2PI = math.log(2.0 * math.pi)


def _log_gamma(x):
    return math.lgamma(x)


def _beta(a, b):
    return math.exp(_log_gamma(a) + _log_gamma(b) - _log_gamma(a + b))


def _log_beta(a, b):
    return _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)


def _betainc(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta_ab = _log_beta(a, b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta_ab) * _betacf(x, a, b) / a
    else:
        return 1.0 - math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta_ab) * _betacf(1.0 - x, b, a) / b


def _betacf(x, a, b, max_iter=200, eps=3e-7):
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1.0) < eps:
            break
    return h


def _gammainc(x, a):
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        return _gser(x, a)
    else:
        return 1.0 - _gcf(x, a)


def _gser(x, a, max_iter=200, eps=3e-7):
    gln = _log_gamma(a)
    ap = a
    total = 1.0 / a
    del_ = total
    for _ in range(max_iter):
        ap += 1.0
        del_ *= x / ap
        total += del_
        if abs(del_) < abs(total) * eps:
            break
    return total * math.exp(a * math.log(x) - x - gln)


def _gcf(x, a, max_iter=200, eps=3e-7):
    gln = _log_gamma(a)
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1.0) < eps:
            break
    return math.exp(a * math.log(x) - x - gln) * h


def normal_pdf(x, mu=0.0, sigma=1.0, log=False):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma
    log_val = -0.5 * z * z - math.log(sigma) - 0.5 * _LOG_2PI
    if log:
        return log_val
    return math.exp(log_val)


def normal_cdf(x, mu=0.0, sigma=1.0, log=False):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma
    cdf_val = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if log:
        if cdf_val <= 0:
            return -float('inf')
        return math.log(cdf_val)
    return cdf_val


def t_pdf(x, df, log=False):
    if df <= 0:
        raise ValueError("df must be positive")
    log_val = _log_gamma((df + 1.0) / 2.0) - _log_gamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    log_val += -(df + 1.0) / 2.0 * math.log(1.0 + x * x / df)
    if log:
        return log_val
    return math.exp(log_val)


def t_cdf(x, df, log=False):
    if df <= 0:
        raise ValueError("df must be positive")
    if x == 0:
        cdf_val = 0.5
    elif x > 0:
        cdf_val = 1.0 - 0.5 * _betainc(df / (df + x * x), df / 2.0, 0.5)
    else:
        cdf_val = 0.5 * _betainc(df / (df + x * x), df / 2.0, 0.5)
    if log:
        if cdf_val <= 0:
            return -float('inf')
        return math.log(cdf_val)
    return cdf_val


def t_ppf(p, df):
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_t_ppf_positive(1.0 - p, df)
    return _t_ppf_positive(p, df)


def _t_ppf_positive(p, df, max_iter=100, eps=1e-8):
    x = normal_ppf(p)
    for _ in range(max_iter):
        c = t_cdf(x, df)
        pdf = t_pdf(x, df)
        if pdf == 0:
            break
        dx = (c - p) / pdf
        x -= dx
        if abs(dx) < eps:
            break
    return x


def t_pvalue(t_stat, df, alternative='two-sided'):
    if alternative == 'two-sided':
        return 2.0 * min(t_cdf(t_stat, df), 1.0 - t_cdf(t_stat, df))
    elif alternative == 'greater':
        return 1.0 - t_cdf(t_stat, df)
    elif alternative == 'less':
        return t_cdf(t_stat, df)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


def chi2_pdf(x, df, log=False):
    if df <= 0:
        raise ValueError("df must be positive")
    if x < 0:
        return -float('inf') if log else 0.0
    if x == 0:
        if df < 2:
            return float('inf')
        elif df == 2:
            return -float('inf') if log else 0.5
        else:
            return -float('inf') if log else 0.0
    log_val = (df / 2.0 - 1.0) * math.log(x) - x / 2.0 - (df / 2.0) * math.log(2.0) - _log_gamma(df / 2.0)
    if log:
        return log_val
    return math.exp(log_val)


def chi2_cdf(x, df, log=False):
    if df <= 0:
        raise ValueError("df must be positive")
    if x <= 0:
        return -float('inf') if log else 0.0
    cdf_val = _gammainc(x / 2.0, df / 2.0)
    if log:
        if cdf_val <= 0:
            return -float('inf')
        return math.log(cdf_val)
    return cdf_val


def chi2_ppf(p, df):
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")
    x = max(0.1, df + math.sqrt(2 * df) * normal_ppf(p))
    for _ in range(100):
        c = chi2_cdf(x, df)
        pdf = chi2_pdf(x, df)
        if pdf == 0:
            break
        dx = (c - p) / pdf
        x = max(1e-10, x - dx)
        if abs(dx) < 1e-8 * max(1, x):
            break
    return x


def chi2_pvalue(chi2_stat, df):
    return 1.0 - chi2_cdf(chi2_stat, df)


def f_pdf(x, dfn, dfd, log=False):
    if dfn <= 0 or dfd <= 0:
        raise ValueError("dfn and dfd must be positive")
    if x <= 0:
        return -float('inf') if log else 0.0
    log_val = (dfn / 2.0) * math.log(dfn) + (dfd / 2.0) * math.log(dfd)
    log_val += _log_gamma((dfn + dfd) / 2.0) - _log_gamma(dfn / 2.0) - _log_gamma(dfd / 2.0)
    log_val += (dfn / 2.0 - 1.0) * math.log(x)
    log_val += -((dfn + dfd) / 2.0) * math.log(dfd + dfn * x)
    if log:
        return log_val
    return math.exp(log_val)


def f_cdf(x, dfn, dfd, log=False):
    if dfn <= 0 or dfd <= 0:
        raise ValueError("dfn and dfd must be positive")
    if x <= 0:
        return -float('inf') if log else 0.0
    cdf_val = _betainc((dfn * x) / (dfd + dfn * x), dfn / 2.0, dfd / 2.0)
    if log:
        if cdf_val <= 0:
            return -float('inf')
        return math.log(cdf_val)
    return cdf_val


def f_ppf(p, dfn, dfd):
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")
    if dfn <= 0 or dfd <= 0:
        raise ValueError("dfn and dfd must be positive")
    x = 1.0
    for _ in range(100):
        c = f_cdf(x, dfn, dfd)
        pdf = f_pdf(x, dfn, dfd)
        if pdf == 0:
            break
        dx = (c - p) / pdf
        x = max(1e-10, x - dx)
        if abs(dx) < 1e-8 * max(1, x):
            break
    return x


def f_pvalue(f_stat, dfn, dfd):
    return 1.0 - f_cdf(f_stat, dfn, dfd)


def normal_fit(sample):
    n = len(sample)
    if n < 2:
        raise ValueError("sample must have at least 2 elements")
    mu = sum(sample) / n
    var = sum((x - mu) ** 2 for x in sample) / (n - 1)
    sigma = math.sqrt(var)
    return mu, sigma


def exponential_fit(sample):
    n = len(sample)
    if n < 1:
        raise ValueError("sample must have at least 1 element")
    if any(x < 0 for x in sample):
        raise ValueError("exponential sample must be non-negative")
    mean = sum(sample) / n
    if mean <= 0:
        raise ValueError("sample mean must be positive")
    lam = 1.0 / mean
    return lam


def t_fit(sample):
    n = len(sample)
    if n < 3:
        raise ValueError("sample must have at least 3 elements for t-distribution")
    mu = sum(sample) / n
    var = sum((x - mu) ** 2 for x in sample) / (n - 1)
    if var <= 1:
        df = 1e6
    else:
        df = max(3.0, 2.0 * var / (var - 1.0))
    return df, mu, math.sqrt(var)


def chi2_fit(sample):
    n = len(sample)
    if n < 1:
        raise ValueError("sample must have at least 1 element")
    if any(x < 0 for x in sample):
        raise ValueError("chi-square sample must be non-negative")
    mean = sum(sample) / n
    df = max(1.0, mean)
    return df


def f_fit(sample):
    n = len(sample)
    if n < 3:
        raise ValueError("sample must have at least 3 elements")
    if any(x <= 0 for x in sample):
        raise ValueError("F sample must be positive")
    mean = sum(sample) / n
    var = sum((x - mean) ** 2 for x in sample) / (n - 1)
    if mean <= 1:
        dfn, dfd = 1.0, 1e6
    else:
        d = 2.0 * mean ** 2 * (mean + 1) / (var * (mean - 1))
        dfn = max(1.0, 2.0 * mean / (mean - 1))
        dfd = max(1.0, d - dfn)
    return dfn, dfd


def normal_ppf(p, mu=0.0, sigma=1.0):
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    return mu + sigma * z


def exponential_pdf(x, lam=1.0, log=False):
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if x < 0:
        return -float('inf') if log else 0.0
    log_val = math.log(lam) - lam * x
    if log:
        return log_val
    return math.exp(log_val)


def exponential_cdf(x, lam=1.0, log=False):
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if x < 0:
        return -float('inf') if log else 0.0
    cdf_val = 1.0 - math.exp(-lam * x)
    if log:
        if cdf_val <= 0:
            return -float('inf')
        return math.log(cdf_val)
    return cdf_val


if __name__ == "__main__":
    print("=== 正态分布 Normal Distribution ===")
    mu, sigma = 0.0, 1.0
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        print(f"x={x:+.1f}  PDF={normal_pdf(x, mu, sigma):.6f}  CDF={normal_cdf(x, mu, sigma):.6f}")
    for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        print(f"p={p:.2f}   PPF={normal_ppf(p, mu, sigma):.6f}")

    print()
    print("--- 大z-score下PDF下溢对比 ---")
    for x in [20.0, 30.0, 40.0, 50.0]:
        pdf_val = normal_pdf(x, mu, sigma)
        logpdf_val = normal_pdf(x, mu, sigma, log=True)
        print(f"x={x:.0f}  PDF={pdf_val}  log(PDF)={logpdf_val:.4f}")

    print()
    print("=== 指数分布 Exponential Distribution ===")
    lam = 1.5
    for x in [0.0, 0.5, 1.0, 2.0, 3.0]:
        print(f"x={x:.1f}  PDF={exponential_pdf(x, lam):.6f}  log(PDF)={exponential_pdf(x, lam, log=True):.6f}  CDF={exponential_cdf(x, lam):.6f}")

    print()
    print("--- 大x下指数分布PDF下溢对比 ---")
    for x in [10.0, 20.0, 30.0]:
        pdf_val = exponential_pdf(x, lam)
        logpdf_val = exponential_pdf(x, lam, log=True)
        print(f"x={x:.0f}  PDF={pdf_val}  log(PDF)={logpdf_val:.4f}")

    print()
    print("=== t分布 t-Distribution ===")
    df = 10
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        print(f"x={x:+.1f}  PDF={t_pdf(x, df):.6f}  CDF={t_cdf(x, df):.6f}")
    for p in [0.05, 0.95, 0.975]:
        print(f"p={p:.3f}  PPF={t_ppf(p, df):.6f}")
    t_stat = 2.228
    print(f"\nt={t_stat}, df=10  two-sided p={t_pvalue(t_stat, df):.6f}")
    print(f"t={t_stat}, df=10  upper-tail p={t_pvalue(t_stat, df, 'greater'):.6f}")

    print()
    print("=== 卡方分布 Chi-square Distribution ===")
    df_chi2 = 5
    for x in [1.0, 3.0, 5.0, 10.0, 15.0]:
        print(f"x={x:.1f}  PDF={chi2_pdf(x, df_chi2):.6f}  CDF={chi2_cdf(x, df_chi2):.6f}")
    for p in [0.90, 0.95, 0.99]:
        print(f"p={p:.3f}  PPF={chi2_ppf(p, df_chi2):.6f}")
    chi2_stat = 11.07
    print(f"\nchi2={chi2_stat}, df=5  p-value={chi2_pvalue(chi2_stat, df_chi2):.6f}")

    print()
    print("=== F分布 F-Distribution ===")
    dfn, dfd = 3, 16
    for x in [0.5, 1.0, 2.0, 3.0, 5.0]:
        print(f"x={x:.1f}  PDF={f_pdf(x, dfn, dfd):.6f}  CDF={f_cdf(x, dfn, dfd):.6f}")
    for p in [0.90, 0.95, 0.99]:
        print(f"p={p:.3f}  PPF={f_ppf(p, dfn, dfd):.6f}")
    f_stat = 3.24
    print(f"\nF={f_stat}, dfn=3, dfd=16  p-value={f_pvalue(f_stat, dfn, dfd):.6f}")

    print()
    print("=== 矩估计 Moment Estimation ===")
    normal_sample = [1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.0, 0.9, 1.2]
    mu_hat, sigma_hat = normal_fit(normal_sample)
    print(f"正态样本拟合: μ̂={mu_hat:.4f}, σ̂={sigma_hat:.4f}")

    exp_sample = [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.8, 0.6, 1.0]
    lam_hat = exponential_fit(exp_sample)
    print(f"指数样本拟合: λ̂={lam_hat:.4f}")
