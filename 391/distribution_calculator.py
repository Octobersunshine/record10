import math
import numpy as np
from scipy.stats import norm, expon, t, chi2, f


def normal_distribution_pdf(x, mean=0, std_dev=1, log=False):
    if log:
        return norm.logpdf(x, loc=mean, scale=std_dev)
    return norm.pdf(x, loc=mean, scale=std_dev)


def normal_distribution_cdf(x, mean=0, std_dev=1):
    return norm.cdf(x, loc=mean, scale=std_dev)


def normal_distribution_ppf(probability, mean=0, std_dev=1):
    if not (0 <= probability <= 1):
        raise ValueError("概率值必须在0到1之间")
    return norm.ppf(probability, loc=mean, scale=std_dev)


def exponential_distribution_pdf(x, rate=1, log=False):
    if x < 0:
        return float('-inf') if log else 0.0
    if log:
        return expon.logpdf(x, scale=1/rate)
    return expon.pdf(x, scale=1/rate)


def exponential_distribution_cdf(x, rate=1):
    if x < 0:
        return 0.0
    return expon.cdf(x, scale=1/rate)


def normal_distribution_pdf_manual(x, mean=0, std_dev=1, log=False):
    z = (x - mean) / std_dev
    log_pdf = -math.log(std_dev) - 0.5 * math.log(2 * math.pi) - 0.5 * z * z
    if log:
        return log_pdf
    return math.exp(log_pdf)


def exponential_distribution_pdf_manual(x, rate=1, log=False):
    if x < 0:
        return float('-inf') if log else 0.0
    log_pdf = math.log(rate) - rate * x
    if log:
        return log_pdf
    return math.exp(log_pdf)


def exponential_distribution_cdf_manual(x, rate=1):
    if x < 0:
        return 0.0
    return 1 - math.exp(-rate * x)


def t_distribution_pdf(x, df, log=False):
    if df <= 0:
        raise ValueError("自由度必须大于0")
    if log:
        return t.logpdf(x, df)
    return t.pdf(x, df)


def t_distribution_cdf(x, df):
    if df <= 0:
        raise ValueError("自由度必须大于0")
    return t.cdf(x, df)


def t_distribution_ppf(probability, df):
    if df <= 0:
        raise ValueError("自由度必须大于0")
    if not (0 <= probability <= 1):
        raise ValueError("概率值必须在0到1之间")
    return t.ppf(probability, df)


def t_distribution_p_value(statistic, df, alternative='two-sided'):
    if alternative not in ['two-sided', 'greater', 'less']:
        raise ValueError("alternative必须是'two-sided', 'greater', 或 'less'")
    
    if alternative == 'two-sided':
        return 2 * min(t.cdf(statistic, df), 1 - t.cdf(statistic, df))
    elif alternative == 'greater':
        return 1 - t.cdf(statistic, df)
    else:
        return t.cdf(statistic, df)


def chi2_distribution_pdf(x, df, log=False):
    if df <= 0:
        raise ValueError("自由度必须大于0")
    if x < 0:
        return float('-inf') if log else 0.0
    if log:
        return chi2.logpdf(x, df)
    return chi2.pdf(x, df)


def chi2_distribution_cdf(x, df):
    if df <= 0:
        raise ValueError("自由度必须大于0")
    if x < 0:
        return 0.0
    return chi2.cdf(x, df)


def chi2_distribution_ppf(probability, df):
    if df <= 0:
        raise ValueError("自由度必须大于0")
    if not (0 <= probability <= 1):
        raise ValueError("概率值必须在0到1之间")
    return chi2.ppf(probability, df)


def chi2_distribution_p_value(statistic, df):
    if df <= 0:
        raise ValueError("自由度必须大于0")
    if statistic < 0:
        return 1.0
    return 1 - chi2.cdf(statistic, df)


def f_distribution_pdf(x, dfn, dfd, log=False):
    if dfn <= 0 or dfd <= 0:
        raise ValueError("自由度必须大于0")
    if x < 0:
        return float('-inf') if log else 0.0
    if log:
        return f.logpdf(x, dfn, dfd)
    return f.pdf(x, dfn, dfd)


def f_distribution_cdf(x, dfn, dfd):
    if dfn <= 0 or dfd <= 0:
        raise ValueError("自由度必须大于0")
    if x < 0:
        return 0.0
    return f.cdf(x, dfn, dfd)


def f_distribution_ppf(probability, dfn, dfd):
    if dfn <= 0 or dfd <= 0:
        raise ValueError("自由度必须大于0")
    if not (0 <= probability <= 1):
        raise ValueError("概率值必须在0到1之间")
    return f.ppf(probability, dfn, dfd)


def f_distribution_p_value(statistic, dfn, dfd):
    if dfn <= 0 or dfd <= 0:
        raise ValueError("自由度必须大于0")
    if statistic < 0:
        return 1.0
    return 1 - f.cdf(statistic, dfn, dfd)


def estimate_normal_parameters(data):
    n = len(data)
    if n == 0:
        raise ValueError("数据不能为空")
    
    mu_hat = sum(data) / n
    variance_hat = sum((x - mu_hat) ** 2 for x in data) / n
    sigma_hat = math.sqrt(variance_hat)
    
    return {'mean': mu_hat, 'std_dev': sigma_hat, 'variance': variance_hat, 'n': n}


def estimate_normal_parameters_unbiased(data):
    n = len(data)
    if n < 2:
        raise ValueError("无偏估计需要至少2个样本")
    
    mu_hat = sum(data) / n
    variance_hat = sum((x - mu_hat) ** 2 for x in data) / (n - 1)
    sigma_hat = math.sqrt(variance_hat)
    
    return {'mean': mu_hat, 'std_dev': sigma_hat, 'variance': variance_hat, 'n': n}


def estimate_exponential_parameter(data):
    n = len(data)
    if n == 0:
        raise ValueError("数据不能为空")
    
    if any(x < 0 for x in data):
        raise ValueError("指数分布数据必须非负")
    
    sample_mean = sum(data) / n
    lambda_hat = 1 / sample_mean if sample_mean > 0 else float('inf')
    
    return {'rate': lambda_hat, 'mean': sample_mean, 'n': n}


def one_sample_t_test(data, pop_mean=0):
    n = len(data)
    if n < 2:
        raise ValueError("t检验需要至少2个样本")
    
    sample_mean = sum(data) / n
    sample_var = sum((x - sample_mean) ** 2 for x in data) / (n - 1)
    se = math.sqrt(sample_var / n)
    
    t_stat = (sample_mean - pop_mean) / se
    df = n - 1
    p_value = t_distribution_p_value(t_stat, df, alternative='two-sided')
    
    return {
        't_statistic': t_stat,
        'df': df,
        'p_value': p_value,
        'sample_mean': sample_mean,
        'pop_mean': pop_mean,
        'standard_error': se
    }


if __name__ == "__main__":
    print("=" * 60)
    print("正态分布计算示例")
    print("=" * 60)
    
    mean = 0
    std_dev = 1
    x_value = 1.96
    
    print(f"\n参数: 均值={mean}, 标准差={std_dev}")
    print(f"x = {x_value}")
    print(f"PDF(x) = {normal_distribution_pdf(x_value, mean, std_dev):.6f}")
    print(f"logPDF(x) = {normal_distribution_pdf(x_value, mean, std_dev, log=True):.6f}")
    print(f"CDF(x) = {normal_distribution_cdf(x_value, mean, std_dev):.6f}")
    
    prob = 0.975
    print(f"\n概率 = {prob}")
    print(f"PPF(prob) = {normal_distribution_ppf(prob, mean, std_dev):.6f}")
    
    print("\n" + "=" * 60)
    print("指数分布计算示例")
    print("=" * 60)
    
    rate = 0.5
    x_exp = 2.0
    
    print(f"\n参数: 速率λ = {rate}")
    print(f"x = {x_exp}")
    print(f"PDF(x) = {exponential_distribution_pdf(x_exp, rate):.6f}")
    print(f"logPDF(x) = {exponential_distribution_pdf(x_exp, rate, log=True):.6f}")
    print(f"CDF(x) = {exponential_distribution_cdf(x_exp, rate):.6f}")
    
    print("\n" + "=" * 60)
    print("对数概率解决数值下溢问题演示")
    print("=" * 60)
    
    z_scores = [10, 20, 30, 40, 50]
    print(f"\n{'z-score':>10} {'PDF(x)':>20} {'logPDF(x)':>20}")
    print("-" * 55)
    for z in z_scores:
        pdf_val = normal_distribution_pdf(z, mean=0, std_dev=1)
        logpdf_val = normal_distribution_pdf(z, mean=0, std_dev=1, log=True)
        print(f"{z:>10} {pdf_val:>20.6e} {logpdf_val:>20.6f}")
    
    print("\n" + "=" * 60)
    print("t分布计算示例")
    print("=" * 60)
    
    df_t = 10
    t_stat = 2.5
    print(f"\n参数: 自由度={df_t}")
    print(f"t统计量 = {t_stat}")
    print(f"PDF(t) = {t_distribution_pdf(t_stat, df_t):.6f}")
    print(f"CDF(t) = {t_distribution_cdf(t_stat, df_t):.6f}")
    print(f"双侧p值 = {t_distribution_p_value(t_stat, df_t, 'two-sided'):.6f}")
    print(f"单侧p值(右侧) = {t_distribution_p_value(t_stat, df_t, 'greater'):.6f}")
    
    print("\n" + "=" * 60)
    print("卡方分布计算示例")
    print("=" * 60)
    
    df_chi2 = 5
    chi2_stat = 12.0
    print(f"\n参数: 自由度={df_chi2}")
    print(f"卡方统计量 = {chi2_stat}")
    print(f"PDF(χ²) = {chi2_distribution_pdf(chi2_stat, df_chi2):.6f}")
    print(f"CDF(χ²) = {chi2_distribution_cdf(chi2_stat, df_chi2):.6f}")
    print(f"p值 = {chi2_distribution_p_value(chi2_stat, df_chi2):.6f}")
    
    print("\n" + "=" * 60)
    print("F分布计算示例")
    print("=" * 60)
    
    dfn, dfd = 3, 16
    f_stat = 3.24
    print(f"\n参数: 分子自由度={dfn}, 分母自由度={dfd}")
    print(f"F统计量 = {f_stat}")
    print(f"PDF(F) = {f_distribution_pdf(f_stat, dfn, dfd):.6f}")
    print(f"CDF(F) = {f_distribution_cdf(f_stat, dfn, dfd):.6f}")
    print(f"p值 = {f_distribution_p_value(f_stat, dfn, dfd):.6f}")
    
    print("\n" + "=" * 60)
    print("矩估计示例")
    print("=" * 60)
    
    np.random.seed(42)
    normal_sample = np.random.normal(loc=5.0, scale=2.0, size=100)
    normal_est = estimate_normal_parameters(normal_sample)
    normal_est_unbiased = estimate_normal_parameters_unbiased(normal_sample)
    print(f"\n正态分布样本 (真实: μ=5.0, σ=2.0):")
    print(f"  样本量 = {normal_est['n']}")
    print(f"  矩估计 μ̂ = {normal_est['mean']:.4f}")
    print(f"  矩估计 σ̂ = {normal_est['std_dev']:.4f}")
    print(f"  无偏估计 σ̂ = {normal_est_unbiased['std_dev']:.4f}")
    
    exp_sample = np.random.exponential(scale=2.0, size=100)
    exp_est = estimate_exponential_parameter(exp_sample)
    print(f"\n指数分布样本 (真实: λ=0.5):")
    print(f"  样本量 = {exp_est['n']}")
    print(f"  矩估计 λ̂ = {exp_est['rate']:.4f}")
    
    print("\n" + "=" * 60)
    print("单样本t检验示例")
    print("=" * 60)
    
    t_test_sample = [1.2, 2.3, 0.8, 1.5, 2.1, 1.8, 0.9, 1.4]
    t_test_result = one_sample_t_test(t_test_sample, pop_mean=1.0)
    print(f"\n样本: {t_test_sample}")
    print(f"原假设: μ = {t_test_result['pop_mean']}")
    print(f"样本均值 = {t_test_result['sample_mean']:.4f}")
    print(f"t统计量 = {t_test_result['t_statistic']:.4f}")
    print(f"自由度 = {t_test_result['df']}")
    print(f"p值 = {t_test_result['p_value']:.6f}")
    
    print("\n" + "=" * 60)
    print("手动计算验证 (不使用scipy)")
    print("=" * 60)
    
    print(f"\n正态分布 PDF(1.96, 0, 1) = {normal_distribution_pdf_manual(1.96):.6f}")
    print(f"正态分布 logPDF(1.96, 0, 1) = {normal_distribution_pdf_manual(1.96, log=True):.6f}")
    print(f"指数分布 PDF(2.0, 0.5) = {exponential_distribution_pdf_manual(2.0, 0.5):.6f}")
    print(f"指数分布 logPDF(2.0, 0.5) = {exponential_distribution_pdf_manual(2.0, 0.5, log=True):.6f}")
    print(f"指数分布 CDF(2.0, 0.5) = {exponential_distribution_cdf_manual(2.0, 0.5):.6f}")
