import math


def one_sample_t_test(sample, mu0, alternative="two-sided"):
    n = len(sample)
    if n < 2:
        raise ValueError("样本容量至少为2")

    mean = sum(sample) / n
    variance = sum((x - mean) ** 2 for x in sample) / (n - 1)

    if variance == 0:
        if mean == mu0:
            t_stat = 0.0
        else:
            t_stat = float('inf') if mean > mu0 else float('-inf')
        p_value = _compute_p(t_stat, n - 1, alternative)
        return t_stat, p_value

    se = math.sqrt(variance / n)
    t_stat = (mean - mu0) / se
    df = n - 1
    p_value = _compute_p(t_stat, df, alternative)

    return t_stat, p_value


def independent_two_sample_t_test(sample1, sample2, equal_var=True, alternative="two-sided"):
    n1, n2 = len(sample1), len(sample2)
    if n1 < 2 or n2 < 2:
        raise ValueError("每个样本容量至少为2")

    mean1 = sum(sample1) / n1
    mean2 = sum(sample2) / n2
    var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1)

    t_stat = mean1 - mean2

    if equal_var:
        sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        if sp2 == 0:
            if mean1 == mean2:
                return 0.0, 1.0
            else:
                t_val = float('inf') if mean1 > mean2 else float('-inf')
                return t_val, 0.0
        se = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        df = n1 + n2 - 2
    else:
        if var1 == 0 and var2 == 0:
            if mean1 == mean2:
                return 0.0, 1.0
            else:
                t_val = float('inf') if mean1 > mean2 else float('-inf')
                return t_val, 0.0
        se = math.sqrt(var1 / n1 + var2 / n2)
        df_num = (var1 / n1 + var2 / n2) ** 2
        df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        if df_den == 0:
            df = 1
        else:
            df = df_num / df_den

    t_stat = t_stat / se
    p_value = _compute_p(t_stat, df, alternative)

    return t_stat, p_value


def paired_t_test(sample1, sample2, alternative="two-sided"):
    if len(sample1) != len(sample2):
        raise ValueError("配对样本容量必须相同")
    if len(sample1) < 2:
        raise ValueError("样本容量至少为2")

    diffs = [a - b for a, b in zip(sample1, sample2)]
    return one_sample_t_test(diffs, 0.0, alternative=alternative)


def levene_test(*samples, center="median"):
    if len(samples) < 2:
        raise ValueError("至少需要两组样本")

    k = len(samples)
    ni = [len(s) for s in samples]
    n_total = sum(ni)

    for s in samples:
        if len(s) < 2:
            raise ValueError("每组样本容量至少为2")

    if center == "median":
        centroids = [_median(s) for s in samples]
    elif center == "mean":
        centroids = [sum(s) / len(s) for s in samples]
    else:
        raise ValueError("center 必须为 'median' 或 'mean'")

    zij = []
    for s, c in zip(samples, centroids):
        zij.append([abs(x - c) for x in s])

    all_z = []
    for group in zij:
        all_z.extend(group)
    z_bar = sum(all_z) / n_total

    z_group_means = [sum(g) / len(g) for g in zij]

    ss_between = sum(ni[j] * (z_group_means[j] - z_bar) ** 2 for j in range(k))
    ss_within = sum(
        sum((zij[j][i] - z_group_means[j]) ** 2 for i in range(ni[j]))
        for j in range(k)
    )

    df1 = k - 1
    df2 = n_total - k

    if ss_within == 0:
        if ss_between == 0:
            return 0.0, 1.0, df1, df2
        else:
            return float('inf'), 0.0, df1, df2

    f_stat = (ss_between / df1) / (ss_within / df2)
    p_value = 1.0 - _f_cdf(f_stat, df1, df2)

    return f_stat, p_value, df1, df2


def _median(data):
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _compute_p(t_stat, df, alternative):
    if alternative == "two-sided":
        return 2 * _t_cdf(-abs(t_stat), df)
    elif alternative == "less":
        return _t_cdf(t_stat, df)
    elif alternative == "greater":
        return 1.0 - _t_cdf(t_stat, df)
    else:
        raise ValueError("alternative 必须为 'two-sided', 'less' 或 'greater'")


def _t_cdf(t, df):
    x = df / (df + t * t)
    if t >= 0:
        return 1.0 - _regularized_incomplete_beta(x, df / 2, 0.5) / 2
    else:
        return _regularized_incomplete_beta(x, df / 2, 0.5) / 2


def _f_cdf(f, d1, d2):
    if f <= 0:
        return 0.0
    x = d1 * f / (d1 * f + d2)
    return _regularized_incomplete_beta(x, d1 / 2, d2 / 2)


def _regularized_incomplete_beta(x, a, b):
    if x < 0 or x > 1:
        raise ValueError("x must be in [0, 1]")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(1 - x, b, a)

    lbeta = _log_beta(a, b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta)

    TINY = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < TINY:
        d = TINY
    d = 1.0 / d
    h = d

    for m in range(1, 300):
        m2 = 2 * m

        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        d = 1.0 / d
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        h *= c * d

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        d = 1.0 / d
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        delta = c * d
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break

    return front * h / a


def _log_beta(a, b):
    return _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)


def _log_gamma(x):
    cof = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


if __name__ == "__main__":
    print("=" * 50)
    print("1. 单样本 t 检验")
    print("=" * 50)
    sample = [5.1, 4.9, 5.0, 5.2, 4.8, 5.3, 4.7, 5.1, 5.0, 4.9]
    mu0 = 5.0
    for alt in ["two-sided", "less", "greater"]:
        t, p = one_sample_t_test(sample, mu0, alternative=alt)
        print(f"  alternative={alt:10s}  t={t:.6f}  p={p:.6f}")

    print()
    print("=" * 50)
    print("2. 独立双样本 t 检验")
    print("=" * 50)
    s1 = [2.3, 3.1, 2.8, 3.5, 2.9, 3.0, 2.7, 3.2]
    s2 = [3.5, 4.1, 3.8, 4.5, 3.9, 4.0, 3.7, 4.2]
    for eq in [True, False]:
        label = "等方差" if eq else "异方差"
        t, p = independent_two_sample_t_test(s1, s2, equal_var=eq)
        print(f"  {label}: t={t:.6f}  p={p:.6f}")
    t, p = independent_two_sample_t_test(s1, s2, alternative="greater")
    print(f"  等方差(右尾): t={t:.6f}  p={p:.6f}")

    print()
    print("=" * 50)
    print("3. 配对样本 t 检验")
    print("=" * 50)
    before = [85, 90, 78, 92, 88, 76, 95, 89]
    after = [88, 93, 81, 95, 90, 79, 98, 92]
    for alt in ["two-sided", "less", "greater"]:
        t, p = paired_t_test(before, after, alternative=alt)
        print(f"  alternative={alt:10s}  t={t:.6f}  p={p:.6f}")

    print()
    print("=" * 50)
    print("4. Levene 方差齐性检验")
    print("=" * 50)
    s3 = [1.2, 1.5, 1.3, 1.7, 1.4, 1.6, 1.5, 1.3]
    f, p, df1, df2 = levene_test(s1, s2, s3, center="median")
    print(f"  center=median: F={f:.6f}  p={p:.6f}  df1={df1}  df2={df2}")
    f, p, df1, df2 = levene_test(s1, s2, s3, center="mean")
    print(f"  center=mean:   F={f:.6f}  p={p:.6f}  df1={df1}  df2={df2}")
