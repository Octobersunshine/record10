import numpy as np
from scipy import stats
from scipy.optimize import minimize

class CauchyDistributionTester:
    def __init__(self):
        self.cauchy_location = None
        self.cauchy_scale = None
        self.norm_mean = None
        self.norm_std = None
        self.data = None
    
    def _robust_cauchy_fit(self, data):
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        scale = max(iqr / 2.0, 0.01)
        
        def neg_loglik(params):
            loc, s = params
            if s <= 0:
                return np.inf
            return -np.sum(stats.cauchy.logpdf(data, loc=loc, scale=s))
        
        best_ll = np.inf
        best_params = None
        
        for i in range(5):
            if i == 0:
                x0 = [median, scale]
            else:
                x0 = [
                    median + np.random.uniform(-scale * 2, scale * 2),
                    scale * np.random.uniform(0.5, 2.0)
                ]
            
            bounds = [(None, None), (1e-6, None)]
            result = minimize(neg_loglik, x0, method='L-BFGS-B', bounds=bounds)
            
            if result.fun < best_ll:
                best_ll = result.fun
                best_params = result.x
        
        return best_params[0], best_params[1]
    
    def fit(self, data):
        self.data = np.asarray(data)
        
        self.cauchy_location, self.cauchy_scale = self._robust_cauchy_fit(self.data)
        
        self.norm_mean = np.mean(self.data)
        self.norm_std = np.std(self.data, ddof=1)
        
        return self
    
    def ks_test_cauchy(self):
        if self.data is None:
            raise ValueError("请先调用fit()方法拟合数据")
        
        statistic, p_value = stats.kstest(self.data, 'cauchy', 
                                         args=(self.cauchy_location, self.cauchy_scale))
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_cauchy': p_value > 0.05
        }
    
    def ks_test_normal(self):
        if self.data is None:
            raise ValueError("请先调用fit()方法拟合数据")
        
        statistic, p_value = stats.kstest(self.data, 'norm', 
                                         args=(self.norm_mean, self.norm_std))
        return {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    def anderson_darling_test(self):
        if self.data is None:
            raise ValueError("请先调用fit()方法拟合数据")
        
        result = stats.anderson(self.data, dist='norm')
        return {
            'statistic': result.statistic,
            'critical_values': result.critical_values,
            'significance_level': result.significance_level,
            'is_normal_at_5pct': result.statistic < result.critical_values[2]
        }
    
    def outlier_sensitivity_test(self, outlier_magnitudes=[5, 10, 20], n_outliers=5):
        if self.data is None:
            raise ValueError("请先调用fit()方法拟合数据")
        
        results = {}
        
        original_cauchy = (self.cauchy_location, self.cauchy_scale)
        original_normal = (self.norm_mean, self.norm_std)
        
        for mag in outlier_magnitudes:
            data_with_outliers = np.concatenate([
                self.data,
                np.full(n_outliers, np.max(self.data) + mag * np.std(self.data))
            ])
            
            cauchy_loc, cauchy_sc = self._robust_cauchy_fit(data_with_outliers)
            
            norm_mean = np.mean(data_with_outliers)
            norm_std = np.std(data_with_outliers, ddof=1)
            
            cauchy_loc_change = abs(cauchy_loc - original_cauchy[0]) / abs(original_cauchy[0]) if original_cauchy[0] != 0 else abs(cauchy_loc - original_cauchy[0])
            cauchy_sc_change = abs(cauchy_sc - original_cauchy[1]) / original_cauchy[1]
            
            norm_mean_change = abs(norm_mean - original_normal[0]) / abs(original_normal[0]) if original_normal[0] != 0 else abs(norm_mean - original_normal[0])
            norm_std_change = abs(norm_std - original_normal[1]) / original_normal[1]
            
            results[f'outlier_{mag}sigma'] = {
                'cauchy': {
                    'location': cauchy_loc,
                    'scale': cauchy_sc,
                    'location_change_pct': cauchy_loc_change * 100,
                    'scale_change_pct': cauchy_sc_change * 100
                },
                'normal': {
                    'mean': norm_mean,
                    'std': norm_std,
                    'mean_change_pct': norm_mean_change * 100,
                    'std_change_pct': norm_std_change * 100
                }
            }
        
        return results
    
    def qq_statistics(self):
        if self.data is None:
            raise ValueError("请先调用fit()方法拟合数据")
        
        sorted_data = np.sort(self.data)
        n = len(sorted_data)
        theoretical_quantiles = np.linspace(1/(n+1), n/(n+1), n)
        
        cauchy_theoretical = stats.cauchy.ppf(theoretical_quantiles, 
                                               loc=self.cauchy_location, 
                                               scale=self.cauchy_scale)
        
        norm_theoretical = stats.norm.ppf(theoretical_quantiles, 
                                          loc=self.norm_mean, 
                                          scale=self.norm_std)
        
        cauchy_corr = np.corrcoef(sorted_data, cauchy_theoretical)[0, 1]
        norm_corr = np.corrcoef(sorted_data, norm_theoretical)[0, 1]
        
        cauchy_mse = np.mean((sorted_data - cauchy_theoretical) ** 2)
        norm_mse = np.mean((sorted_data - norm_theoretical) ** 2)
        
        return {
            'cauchy': {
                'correlation': cauchy_corr,
                'mse': cauchy_mse
            },
            'normal': {
                'correlation': norm_corr,
                'mse': norm_mse
            }
        }
    
    def log_likelihood_comparison(self):
        if self.data is None:
            raise ValueError("请先调用fit()方法拟合数据")
        
        cauchy_ll = np.sum(stats.cauchy.logpdf(self.data, 
                                                loc=self.cauchy_location, 
                                                scale=self.cauchy_scale))
        
        norm_ll = np.sum(stats.norm.logpdf(self.data, 
                                           loc=self.norm_mean, 
                                           scale=self.norm_std))
        
        n = len(self.data)
        k = 2
        
        cauchy_aic = 2 * k - 2 * cauchy_ll
        norm_aic = 2 * k - 2 * norm_ll
        
        cauchy_bic = k * np.log(n) - 2 * cauchy_ll
        norm_bic = k * np.log(n) - 2 * norm_ll
        
        return {
            'cauchy': {
                'log_likelihood': cauchy_ll,
                'aic': cauchy_aic,
                'bic': cauchy_bic
            },
            'normal': {
                'log_likelihood': norm_ll,
                'aic': norm_aic,
                'bic': norm_bic
            }
        }
    
    def comprehensive_test(self, significance_level=0.05):
        if self.data is None:
            raise ValueError("请先调用fit()方法拟合数据")
        
        ks_cauchy = self.ks_test_cauchy()
        ks_normal = self.ks_test_normal()
        qq_stats = self.qq_statistics()
        ll_stats = self.log_likelihood_comparison()
        
        is_likely_cauchy = (
            ks_cauchy['p_value'] > significance_level and
            qq_stats['cauchy']['correlation'] > qq_stats['normal']['correlation'] and
            ll_stats['cauchy']['aic'] < ll_stats['normal']['aic']
        )
        
        is_likely_normal = (
            ks_normal['p_value'] > significance_level and
            qq_stats['normal']['correlation'] > qq_stats['cauchy']['correlation'] and
            ll_stats['normal']['aic'] < ll_stats['cauchy']['aic']
        )
        
        conclusion = '不确定'
        if is_likely_cauchy and not is_likely_normal:
            conclusion = '更可能是柯西分布'
        elif is_likely_normal and not is_likely_cauchy:
            conclusion = '更可能是正态分布'
        elif is_likely_cauchy and is_likely_normal:
            if (ll_stats['cauchy']['aic'] < ll_stats['normal']['aic'] and 
                qq_stats['cauchy']['correlation'] > qq_stats['normal']['correlation']):
                conclusion = '两种分布都拟合良好，略倾向柯西分布'
            else:
                conclusion = '两种分布都拟合良好，略倾向正态分布'
        
        return {
            'cauchy_params': {
                'location': self.cauchy_location,
                'scale': self.cauchy_scale
            },
            'normal_params': {
                'mean': self.norm_mean,
                'std': self.norm_std
            },
            'ks_test': {
                'cauchy': ks_cauchy,
                'normal': ks_normal
            },
            'qq_statistics': qq_stats,
            'information_criteria': ll_stats,
            'conclusion': conclusion,
            'is_likely_cauchy': is_likely_cauchy,
            'is_likely_normal': is_likely_normal
        }
    
    def print_comprehensive_report(self):
        result = self.comprehensive_test()
        
        print("=" * 70)
        print("柯西分布 vs 正态分布 综合检验报告")
        print("=" * 70)
        
        print("\n【拟合参数】")
        print(f"  柯西分布: 位置参数 = {result['cauchy_params']['location']:.4f}, "
              f"尺度参数 = {result['cauchy_params']['scale']:.4f}")
        print(f"  正态分布: 均值 = {result['normal_params']['mean']:.4f}, "
              f"标准差 = {result['normal_params']['std']:.4f}")
        
        print("\n【KS检验 (p值 > 0.05 表示不能拒绝原假设)】")
        print(f"  柯西分布: 统计量 = {result['ks_test']['cauchy']['statistic']:.4f}, "
              f"p值 = {result['ks_test']['cauchy']['p_value']:.4f}")
        print(f"           {'通过检验 ✓' if result['ks_test']['cauchy']['p_value'] > 0.05 else '拒绝原假设 ✗'}")
        print(f"  正态分布: 统计量 = {result['ks_test']['normal']['statistic']:.4f}, "
              f"p值 = {result['ks_test']['normal']['p_value']:.4f}")
        print(f"           {'通过检验 ✓' if result['ks_test']['normal']['p_value'] > 0.05 else '拒绝原假设 ✗'}")
        
        print("\n【QQ图相关系数 (越接近1越好)】")
        print(f"  柯西分布: 相关系数 = {result['qq_statistics']['cauchy']['correlation']:.6f}, "
              f"MSE = {result['qq_statistics']['cauchy']['mse']:.4f}")
        print(f"  正态分布: 相关系数 = {result['qq_statistics']['normal']['correlation']:.6f}, "
              f"MSE = {result['qq_statistics']['normal']['mse']:.4f}")
        better = '柯西' if result['qq_statistics']['cauchy']['correlation'] > result['qq_statistics']['normal']['correlation'] else '正态'
        print(f"  QQ图拟合更好: {better}分布")
        
        print("\n【信息准则 (越小越好)】")
        print(f"  柯西分布: AIC = {result['information_criteria']['cauchy']['aic']:.2f}, "
              f"BIC = {result['information_criteria']['cauchy']['bic']:.2f}")
        print(f"  正态分布: AIC = {result['information_criteria']['normal']['aic']:.2f}, "
              f"BIC = {result['information_criteria']['normal']['bic']:.2f}")
        aic_better = '柯西' if result['information_criteria']['cauchy']['aic'] < result['information_criteria']['normal']['aic'] else '正态'
        print(f"  AIC更优: {aic_better}分布")
        
        print("\n" + "=" * 70)
        print(f"【结论】{result['conclusion']}")
        print("=" * 70)
        
        return result


def generate_sample_data(distribution='cauchy', size=1000, **kwargs):
    if distribution == 'cauchy':
        loc = kwargs.get('location', 0.0)
        scale = kwargs.get('scale', 1.0)
        return stats.cauchy.rvs(loc=loc, scale=scale, size=size)
    elif distribution == 'normal':
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 1.0)
        return stats.norm.rvs(loc=mean, scale=std, size=size)
    elif distribution == 'mixed':
        size1 = size // 2
        size2 = size - size1
        loc = kwargs.get('location', 0.0)
        scale = kwargs.get('scale', 1.0)
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 1.0)
        return np.concatenate([
            stats.cauchy.rvs(loc=loc, scale=scale, size=size1),
            stats.norm.rvs(loc=mean, scale=std, size=size2)
        ])
    else:
        raise ValueError(f"未知分布类型: {distribution}")


if __name__ == "__main__":
    print("测试1: 纯柯西分布数据")
    print("-" * 70)
    np.random.seed(42)
    data_cauchy = generate_sample_data('cauchy', size=500, location=2.0, scale=1.5)
    tester = CauchyDistributionTester()
    tester.fit(data_cauchy)
    tester.print_comprehensive_report()
    
    print("\n\n测试2: 纯正态分布数据")
    print("-" * 70)
    data_normal = generate_sample_data('normal', size=500, mean=2.0, std=1.5)
    tester2 = CauchyDistributionTester()
    tester2.fit(data_normal)
    tester2.print_comprehensive_report()
    
    print("\n\n测试3: 异常值影响对比")
    print("-" * 70)
    data_clean = generate_sample_data('cauchy', size=200, location=0.0, scale=1.0)
    tester3 = CauchyDistributionTester()
    tester3.fit(data_clean)
    
    print("原始拟合结果:")
    print(f"  柯西: 位置={tester3.cauchy_location:.4f}, 尺度={tester3.cauchy_scale:.4f}")
    print(f"  正态: 均值={tester3.norm_mean:.4f}, 标准差={tester3.norm_std:.4f}")
    
    outlier_results = tester3.outlier_sensitivity_test(outlier_magnitudes=[5, 10], n_outliers=5)
    
    print("\n异常值影响:")
    for mag, res in outlier_results.items():
        print(f"\n  {mag} 倍标准差的异常值:")
        print(f"    柯西位置变化: {res['cauchy']['location_change_pct']:.2f}%, "
              f"尺度变化: {res['cauchy']['scale_change_pct']:.2f}%")
        print(f"    正态均值变化: {res['normal']['mean_change_pct']:.2f}%, "
              f"标准差变化: {res['normal']['std_change_pct']:.2f}%")
    
    print("\n" + "=" * 70)
    print("使用示例:")
    print("  from cauchy_distribution_test import CauchyDistributionTester")
    print("  tester = CauchyDistributionTester()")
    print("  tester.fit(your_data)")
    print("  result = tester.comprehensive_test()")
    print("  print(result['conclusion'])")
    print("=" * 70)
