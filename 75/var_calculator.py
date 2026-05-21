import numpy as np
from scipy import stats
from scipy.stats import t, norm, genpareto


def historical_var(returns, confidence_level=0.95):
    returns = np.asarray(returns)
    if len(returns) == 0:
        raise ValueError("收益率数据不能为空")
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    return sorted_returns[index]


def normal_var(returns, confidence_level=0.95):
    mu = np.mean(returns)
    sigma = np.std(returns)
    z_score = norm.ppf(1 - confidence_level)
    return mu + z_score * sigma


def t_var(returns, confidence_level=0.95):
    df, loc, scale = t.fit(returns)
    t_quantile = t.ppf(1 - confidence_level, df=df, loc=loc, scale=scale)
    return t_quantile


def cornish_fisher_var(returns, confidence_level=0.95):
    mu = np.mean(returns)
    sigma = np.std(returns)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    z = norm.ppf(1 - confidence_level)
    
    z_cf = z + (z**2 - 1) * skewness / 6 + \
          (z**3 - 3*z) * kurtosis / 24 - \
          (2*z**3 - 5*z) * skewness**2 / 36
    
    return mu + z_cf * sigma


def evt_var(returns, confidence_level=0.95, threshold_percentile=0.1):
    returns = np.asarray(returns)
    sorted_returns = np.sort(returns)
    
    threshold_idx = int(threshold_percentile * len(sorted_returns))
    threshold = sorted_returns[threshold_idx]
    
    excesses = returns[returns < threshold] - threshold
    
    if len(excesses) < 10:
        return historical_var(returns, confidence_level)
    
    try:
        shape, loc, scale = genpareto.fit(-excesses)
        
        n = len(returns)
        n_excess = len(excesses)
        
        var_evt = threshold - (scale / shape) * (((n / n_excess) * (1 - confidence_level))**(-shape) - 1)
        return var_evt
    except:
        return historical_var(returns, confidence_level)


def historical_es(returns, confidence_level=0.95):
    var = historical_var(returns, confidence_level)
    returns = np.asarray(returns)
    tail_returns = returns[returns <= var]
    if len(tail_returns) == 0:
        return var
    return np.mean(tail_returns)


def normal_es(returns, confidence_level=0.95):
    mu = np.mean(returns)
    sigma = np.std(returns)
    alpha = 1 - confidence_level
    z_alpha = norm.ppf(alpha)
    es = mu - sigma * (norm.pdf(z_alpha) / alpha)
    return es


def t_es(returns, confidence_level=0.95):
    df, loc, scale = t.fit(returns)
    alpha = 1 - confidence_level
    t_alpha = t.ppf(alpha, df=df)
    
    numerator = t.pdf(t_alpha, df=df) * (df + t_alpha**2)
    denominator = alpha * (df - 1)
    es = loc - scale * (numerator / denominator)
    return es


def cornish_fisher_es(returns, confidence_level=0.95):
    var = cornish_fisher_var(returns, confidence_level)
    returns = np.asarray(returns)
    tail_returns = returns[returns <= var]
    if len(tail_returns) == 0:
        return var
    return np.mean(tail_returns)


def evt_es(returns, confidence_level=0.95, threshold_percentile=0.1):
    returns = np.asarray(returns)
    sorted_returns = np.sort(returns)
    
    threshold_idx = int(threshold_percentile * len(sorted_returns))
    threshold = sorted_returns[threshold_idx]
    
    excesses = returns[returns < threshold] - threshold
    
    if len(excesses) < 10:
        return historical_es(returns, confidence_level)
    
    try:
        shape, loc, scale = genpareto.fit(-excesses)
        
        n = len(returns)
        n_excess = len(excesses)
        
        var_evt = threshold - (scale / shape) * (((n / n_excess) * (1 - confidence_level))**(-shape) - 1)
        
        if shape != 1:
            es_evt = (var_evt + scale - shape * threshold) / (1 - shape)
        else:
            es_evt = var_evt + scale
        
        return es_evt
    except:
        return historical_es(returns, confidence_level)


VAR_METHODS = {
    '历史模拟法': historical_var,
    '正态分布': normal_var,
    't分布': t_var,
    'Cornish-Fisher': cornish_fisher_var,
    '极值理论(EVT)': evt_var
}

ES_METHODS = {
    '历史模拟法': historical_es,
    '正态分布': normal_es,
    't分布': t_es,
    'Cornish-Fisher': cornish_fisher_es,
    '极值理论(EVT)': evt_es
}


def calculate_all_vars(returns, confidence_levels=[0.95, 0.99]):
    results = {}
    for cl in confidence_levels:
        level_str = f"{int(cl * 100)}%"
        results[level_str] = {}
        for name, func in VAR_METHODS.items():
            results[level_str][name] = func(returns, cl)
    return results


def calculate_all_es(returns, confidence_levels=[0.95, 0.99]):
    results = {}
    for cl in confidence_levels:
        level_str = f"{int(cl * 100)}%"
        results[level_str] = {}
        for name, func in ES_METHODS.items():
            results[level_str][name] = func(returns, cl)
    return results


def backtest_var_es(returns, confidence_level=0.95, window_size=250):
    returns = np.asarray(returns)
    n = len(returns)
    
    results = {
        method: {
            'var_violations': [],
            'es_violations': [],
            'var_values': [],
            'es_values': [],
            'actual_returns': []
        }
        for method in VAR_METHODS.keys()
    }
    
    for i in range(window_size, n):
        train_data = returns[i - window_size:i]
        actual_return = returns[i]
        
        for method in VAR_METHODS.keys():
            var = VAR_METHODS[method](train_data, confidence_level)
            es = ES_METHODS[method](train_data, confidence_level)
            
            results[method]['var_values'].append(var)
            results[method]['es_values'].append(es)
            results[method]['actual_returns'].append(actual_return)
            
            if actual_return < var:
                results[method]['var_violations'].append(i)
            if actual_return < es:
                results[method]['es_violations'].append(i)
    
    return results


def analyze_backtest(backtest_results, confidence_level, test_periods):
    expected_violations = (1 - confidence_level) * test_periods
    
    analysis = {}
    for method, data in backtest_results.items():
        var_violation_count = len(data['var_violations'])
        es_violation_count = len(data['es_violations'])
        
        var_violation_ratio = var_violation_count / test_periods
        es_violation_ratio = es_violation_count / test_periods
        
        violation_returns = np.array(data['actual_returns'])[
            np.array(data['actual_returns']) < np.array(data['var_values'])
        ]
        
        avg_excess_loss = np.mean(
            np.array(data['var_values'])[np.array(data['actual_returns']) < np.array(data['var_values'])] -
            violation_returns
        ) if len(violation_returns) > 0 else 0
        
        es_covered_ratio = np.mean(
            np.array(data['actual_returns'])[np.array(data['actual_returns']) < np.array(data['var_values'])] >=
            np.array(data['es_values'])[np.array(data['actual_returns']) < np.array(data['var_values'])]
        ) if len(violation_returns) > 0 else 1.0
        
        analysis[method] = {
            'VaR违反次数': var_violation_count,
            'VaR违反率': var_violation_ratio,
            '期望违反次数': expected_violations,
            '平均超额损失': avg_excess_loss,
            'ES违反次数': es_violation_count,
            'ES违反率': es_violation_ratio,
            'ES覆盖尾部比例': es_covered_ratio
        }
    
    return analysis


if __name__ == "__main__":
    np.random.seed(42)
    
    normal_returns = np.random.normal(0.001, 0.02, 1000)
    t_returns = stats.t.rvs(df=5, loc=0.001, scale=0.02, size=1000)
    
    print("=" * 70)
    print("第一部分：VaR与ES对比")
    print("=" * 70)
    
    for data_name, returns in [("正态分布数据", normal_returns), ("t分布数据(肥尾)", t_returns)]:
        print(f"\n【{data_name}】")
        print(f"  均值: {np.mean(returns):.4f}, 标准差: {np.std(returns):.4f}")
        print(f"  偏度: {stats.skew(returns):.4f}, 峰度: {stats.kurtosis(returns):.4f}")
        print()
        
        var_results = calculate_all_vars(returns)
        es_results = calculate_all_es(returns)
        
        for level in ['95%', '99%']:
            print(f"  {level} 置信水平:")
            print(f"  {'方法':20s} {'VaR':>12s} {'ES':>12s} {'ES-VaR':>12s}")
            print("  " + "-" * 60)
            for method in VAR_METHODS.keys():
                var = var_results[level][method]
                es = es_results[level][method]
                print(f"  {method:20s} {var:12.4f} {es:12.4f} {es-var:12.4f}")
            print()
    
    print("=" * 70)
    print("第二部分：滚动窗口回测")
    print("=" * 70)
    
    test_returns = stats.t.rvs(df=5, loc=0.001, scale=0.02, size=1500)
    confidence_level = 0.95
    window_size = 250
    
    print(f"\n回测设置:")
    print(f"  数据: t分布(df=5, 肥尾)")
    print(f"  总样本: {len(test_returns)}天")
    print(f"  滚动窗口: {window_size}天")
    print(f"  回测期: {len(test_returns) - window_size}天")
    print(f"  置信水平: {int(confidence_level*100)}%")
    print()
    
    backtest_results = backtest_var_es(test_returns, confidence_level, window_size)
    analysis = analyze_backtest(backtest_results, confidence_level, len(test_returns) - window_size)
    
    print(f"{'方法':20s} {'VaR违反次数':>12s} {'VaR违反率':>12s} {'ES覆盖比例':>12s} {'平均超额损失':>12s}")
    print("-" * 70)
    for method, metrics in analysis.items():
        print(f"{method:20s} {metrics['VaR违反次数']:12d} {metrics['VaR违反率']:12.4f} "
              f"{metrics['ES覆盖尾部比例']:12.4f} {metrics['平均超额损失']:12.4f}")
    
    print()
    print("=" * 70)
    print("关键结论:")
    print("=" * 70)
    print("1. VaR vs ES:")
    print("   - VaR只告诉我们'最坏的x%情况至少损失多少'")
    print("   - ES告诉我们'当损失超过VaR时，平均损失是多少'")
    print("   - ES是一致风险度量，满足次可加性")
    print()
    print("2. 肥尾数据下正态分布的问题:")
    print("   - VaR违反率 >> 理论值（风险低估）")
    print("   - 实际超额损失远大于正态分布假设")
    print()
    print("3. ES的优势:")
    print("   - 更全面描述尾部风险")
    print("   - 鼓励分散投资（次可加性）")
    print("   - 监管趋同（巴塞尔协议III推荐ES）")
