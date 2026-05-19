from pearson3 import PearsonIIIFitter, calculate_design_flood, NonstationaryPearsonIII, NonstationaryP3Advanced


def example_basic_usage():
    print("示例1: 基本使用 - 拟合分布并计算设计洪水")
    print("-" * 60)
    
    flood_series = [820, 960, 1050, 780, 1200, 920, 1100, 890, 1020, 950,
                    1300, 880, 1080, 990, 1150, 850, 1000, 1120, 930, 1060]
    
    fitter = PearsonIIIFitter(flood_series)
    
    params = fitter.fit(method='l_moments', robust=True)
    
    print("统计参数（稳健L-矩法）:")
    print(f"  均值: {params['mean']:.2f}")
    print(f"  标准差: {params['std']:.2f}")
    print(f"  Cv: {params['cv']:.4f}")
    print(f"  Cs: {params['cs']:.4f}")
    print()
    
    T = 100
    design_value = fitter.design_flood(T)
    print(f"{T}年一遇设计洪水: {design_value:.2f} m³/s")
    
    x = 1200
    rp = fitter.get_return_period(x)
    print(f"洪水流量 {x} m³/s 的重现期: {rp:.1f} 年")
    print()


def example_multiple_methods():
    print("示例2: 多种拟合方法比较")
    print("-" * 60)
    
    flood_series = [820, 960, 1050, 780, 1200, 920, 1100, 890, 1020, 950,
                    1300, 880, 1080, 990, 1150, 850, 1000, 1120, 930, 1060]
    
    methods = [
        ('传统矩法', 'moments', {}),
        ('传统L-矩法', 'l_moments', {'robust': False}),
        ('稳健L-矩法', 'l_moments', {'robust': True}),
        ('截断L-矩法', 'trimmed_l_moments', {}),
        ('加权L-矩法(Huber)', 'weighted_l_moments', {'weight_type': 'huber'}),
    ]
    
    T = 100
    print(f"{'方法':<25} {'Cs':<10} {'{T}年一遇'.format(T=T):<15}")
    print("-" * 50)
    
    for name, method, kwargs in methods:
        fitter = PearsonIIIFitter(flood_series)
        params = fitter.fit(method=method, **kwargs)
        design = fitter.design_flood(T)
        print(f"{name:<25} {params['cs']:<10.4f} {design:<15.2f}")
    print()


def example_outlier_impact():
    print("示例3: 异常值（特大洪水）对不同方法的影响")
    print("-" * 60)
    
    base_data = [820, 960, 1050, 780, 1200, 920, 1100, 890, 1020, 950,
                 1300, 880, 1080, 990, 1150, 850, 1000, 1120, 930, 1060]
    
    data_with_outliers = base_data + [2500, 2800]
    
    print("原始数据 vs 含异常值数据的参数对比:")
    print(f"{'方法':<25} {'原始Cs':<12} {'含异常值Cs':<15} {'变化率':<10}")
    print("-" * 65)
    
    methods = [
        ('传统矩法', 'moments', {}),
        ('传统L-矩法', 'l_moments', {'robust': False}),
        ('稳健L-矩法(5%)', 'l_moments', {'robust': True, 'trim_percent': 0.05}),
        ('截断L-矩法', 'trimmed_l_moments', {'trim_left': 0.05, 'trim_right': 0.1}),
        ('加权L-矩法', 'weighted_l_moments', {'weight_type': 'huber', 'c': 1.5}),
    ]
    
    for name, method, kwargs in methods:
        fitter1 = PearsonIIIFitter(base_data)
        params1 = fitter1.fit(method=method, **kwargs)
        
        fitter2 = PearsonIIIFitter(data_with_outliers)
        params2 = fitter2.fit(method=method, **kwargs)
        
        change = abs(params2['cs'] - params1['cs']) / params1['cs'] * 100
        print(f"{name:<25} {params1['cs']:<12.4f} {params2['cs']:<15.4f} {change:<10.1f}%")
    
    print()
    print("注: 变化率越小，方法对异常值越稳健")
    print()


def example_probability_functions():
    print("示例4: 概率密度和累积分布函数")
    print("-" * 60)
    
    flood_series = [820, 960, 1050, 780, 1200, 920, 1100, 890, 1020, 950]
    
    fitter = PearsonIIIFitter(flood_series)
    fitter.fit(method='trimmed_l_moments', trim_left=0.05, trim_right=0.1)
    
    x_values = [800, 900, 1000, 1100, 1200]
    
    print(f"{'流量(m³/s)':<12} {'PDF':<15} {'CDF':<15}")
    print("-" * 45)
    for x in x_values:
        pdf = fitter.pdf(x)
        cdf = fitter.cdf(x)
        print(f"{x:<12} {pdf:<15.6f} {cdf:<15.4f}")
    print()


def example_quantile():
    print("示例5: 计算指定频率的分位数")
    print("-" * 60)
    
    flood_series = [820, 960, 1050, 780, 1200, 920, 1100, 890, 1020, 950]
    
    fitter = PearsonIIIFitter(flood_series)
    fitter.fit(method='weighted_l_moments', weight_type='tukey', c=4.685)
    
    probabilities = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001]
    
    print(f"{'频率 p':<12} {'分位数 (m³/s)':<15}")
    print("-" * 30)
    for p in probabilities:
        q = fitter.quantile(p)
        print(f"{p:<12.4f} {q:<15.2f}")
    print()


def example_method_comparison_table():
    print("示例6: 各方法适用场景对比")
    print("=" * 70)
    print(f"{'方法':<20} {'优点':<25} {'适用场景':<25}")
    print("-" * 70)
    methods_info = [
        ('传统矩法', '计算简单', '数据质量好，无异常值'),
        ('传统L-矩法', '比矩法稳健', '轻度异常值的数据'),
        ('稳健L-矩法', '截断异常值', '存在少量特大洪水'),
        ('截断L-矩法', '非对称截断', '右尾有特大洪水'),
        ('加权L-矩法', '自动权重调整', '异常值程度不明'),
    ]
    for name, advantage, scenario in methods_info:
        print(f"{name:<20} {advantage:<25} {scenario:<25}")
    print("=" * 70)
    print()


def example_nonstationary_basic():
    print("示例7: 非平稳P-III分布基本使用")
    print("-" * 70)
    
    import numpy as np
    np.random.seed(123)
    
    n_years = 40
    time = np.arange(n_years)
    climate_index = 0.05 * time + np.random.randn(n_years) * 0.1
    
    mu0 = 900
    mu1 = 200
    sigma0 = np.log(120)
    sigma1 = 0.2
    alpha = 2.5
    
    mu = mu0 + mu1 * climate_index
    sigma = np.exp(sigma0 + sigma1 * climate_index)
    
    from scipy.stats import gamma as gamma_dist
    flood_data = mu - sigma + gamma_dist.rvs(a=alpha, scale=sigma/alpha, size=n_years)
    
    model = NonstationaryPearsonIII(flood_data, climate_index)
    params = model.fit()
    
    if params['converged']:
        print(f"数据: {n_years}年洪水序列 + 气候指标协变量")
        print(f"\n模型参数估计:")
        print(f"  位置参数: μ = {params['theta'][0]:.2f} + {params['theta'][1]:.2f} × 气候指标")
        print(f"  尺度参数: log(σ) = {params['gamma'][0]:.3f} + {params['gamma'][1]:.3f} × 气候指标")
        print(f"  形状参数: α = {params['alpha']:.4f}")
        
        print(f"\n时变设计洪水计算:")
        for T in [20, 50, 100]:
            for cov_val in [-0.5, 0.0, 0.5]:
                df = model.design_flood(T, cov_val)
                print(f"  {T}年一遇, 气候指标={cov_val:.1f}: {df:.1f} m³/s")
    else:
        print("模型未收敛:", params['message'])
    
    print()


def example_nonstationary_model_selection():
    print("示例8: 非平稳P-III模型比较与选择")
    print("-" * 70)
    
    import numpy as np
    np.random.seed(456)
    
    n_years = 50
    time = np.arange(n_years)
    trend = time / n_years
    
    mu0 = 1000
    mu1 = 300
    sigma0 = np.log(150)
    sigma1 = 0.0
    alpha = 3.0
    
    mu = mu0 + mu1 * trend
    sigma = np.exp(sigma0 + sigma1 * trend)
    
    from scipy.stats import gamma as gamma_dist
    flood_data = mu - sigma + gamma_dist.rvs(a=alpha, scale=sigma/alpha, size=n_years)
    
    adv_model = NonstationaryP3Advanced(flood_data, trend.reshape(-1, 1))
    model_results, best_model = adv_model.fit_multi_model()
    
    print(f"多模型比较结果 (AIC准则):")
    print(f"{'模型':<15} {'AIC':<12} {'BIC':<12} {'描述':<25}")
    print("-" * 65)
    
    model_descriptions = {
        'stationary': '平稳模型',
        'mu_trend': '仅位置参数有趋势',
        'sigma_trend': '仅尺度参数有趋势',
        'both_trend': '位置和尺度都有趋势'
    }
    
    for name, result in model_results.items():
        if result['success']:
            print(f"{name:<15} {result['aic']:<12.1f} {result['bic']:<12.1f} {model_descriptions[name]:<25}")
    
    print(f"\n最优模型: {best_model} ({model_descriptions[best_model]})")
    
    print(f"\n各重现期设计洪水趋势:")
    for T in [10, 20, 50, 100]:
        df_start = adv_model.design_flood(T, 0.0)
        df_end = adv_model.design_flood(T, 1.0)
        change = (df_end - df_start) / df_start * 100
        print(f"  {T}年一遇: {df_start:.0f} → {df_end:.0f} m³/s (变化{change:+.1f}%)")
    
    print()


def example_nonstationary_summary():
    print("示例9: 非平稳频率分析方法总结")
    print("=" * 70)
    
    print(f"\n{'方法':<25} {'协变量':<15} {'适用场景':<30}")
    print("-" * 70)
    
    methods = [
        ('平稳P-III', '无', '序列无显著趋势'),
        ('位置参数非平稳', '位置参数', '均值随时间/气候变化'),
        ('尺度参数非平稳', '尺度参数', '变异性随时间变化'),
        ('双参数非平稳', '位置+尺度', '均值和方差都变化'),
    ]
    
    for name, cov, scenario in methods:
        print(f"{name:<25} {cov:<15} {scenario:<30}")
    
    print("=" * 70)
    
    print(f"\n常用协变量:")
    print(f"  - 时间趋势 (年序)")
    print(f"  - 气候指标 (SOI, PDO, AMO等)")
    print(f"  - 水库运行指标")
    print(f"  - 城市化程度指标")
    print(f"  - 土地利用变化指标")
    
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_multiple_methods()
    example_outlier_impact()
    example_probability_functions()
    example_quantile()
    example_method_comparison_table()
    example_nonstationary_basic()
    example_nonstationary_model_selection()
    example_nonstationary_summary()
    
    print("所有示例完成!")
