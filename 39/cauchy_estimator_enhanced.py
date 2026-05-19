import numpy as np
from scipy import stats
from scipy.optimize import minimize, basinhopping, differential_evolution

class DistributionTester:
    @staticmethod
    def ks_test_cauchy(data, location, scale):
        statistic, p_value = stats.kstest(data, 'cauchy', args=(location, scale))
        return {'statistic': statistic, 'p_value': p_value}
    
    @staticmethod
    def ks_test_normal(data, loc, scale):
        statistic, p_value = stats.kstest(data, 'norm', args=(loc, scale))
        return {'statistic': statistic, 'p_value': p_value}
    
    @staticmethod
    def aic(log_likelihood, n_params=2):
        return 2 * n_params - 2 * log_likelihood
    
    @staticmethod
    def bic(log_likelihood, n_samples, n_params=2):
        return n_params * np.log(n_samples) - 2 * log_likelihood
    
    @staticmethod
    def qq_statistics(data, distribution, *params):
        sorted_data = np.sort(data)
        n = len(sorted_data)
        theoretical_quantiles = distribution.ppf(np.arange(1, n + 1) / (n + 1), *params)
        
        correlation = np.corrcoef(sorted_data, theoretical_quantiles)[0, 1]
        
        slope, intercept = np.polyfit(theoretical_quantiles, sorted_data, 1)
        
        residuals = sorted_data - (intercept + slope * theoretical_quantiles)
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        return {
            'correlation': correlation,
            'slope': slope,
            'intercept': intercept,
            'rmse': rmse,
            'sorted_data': sorted_data,
            'theoretical_quantiles': theoretical_quantiles
        }

class CauchyEstimator:
    def __init__(self):
        self.location = None
        self.scale = None
        self._fit_method = None
        self._tester = DistributionTester()
    
    @staticmethod
    def _robust_initial_guess(data):
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        scale = iqr / 2.0
        return median, max(scale, 0.01)
    
    @staticmethod
    def _negative_log_likelihood(params, data):
        location, scale = params
        if scale <= 0:
            return np.inf
        return -np.sum(stats.cauchy.logpdf(data, loc=location, scale=scale))
    
    def _fit_default(self, data):
        self.location, self.scale = stats.cauchy.fit(data)
        return self.location, self.scale
    
    def _fit_robust(self, data):
        x0 = self._robust_initial_guess(data)
        bounds = [(None, None), (1e-6, None)]
        result = minimize(self._negative_log_likelihood, x0, args=(data,), 
                         method='L-BFGS-B', bounds=bounds)
        self.location, self.scale = result.x
        return self.location, self.scale
    
    def _fit_multistart(self, data, n_starts=10):
        best_ll = np.inf
        best_params = None
        
        median, scale = self._robust_initial_guess(data)
        
        for i in range(n_starts):
            if i == 0:
                x0 = [median, scale]
            else:
                x0 = [
                    median + np.random.uniform(-scale * 2, scale * 2),
                    scale * np.random.uniform(0.5, 2.0)
                ]
            
            bounds = [(None, None), (1e-6, None)]
            result = minimize(self._negative_log_likelihood, x0, args=(data,),
                             method='L-BFGS-B', bounds=bounds)
            
            if result.fun < best_ll:
                best_ll = result.fun
                best_params = result.x
        
        self.location, self.scale = best_params
        return self.location, self.scale
    
    def _fit_basinhopping(self, data):
        x0 = self._robust_initial_guess(data)
        
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': [(None, None), (1e-6, None)],
            'args': (data,)
        }
        
        result = basinhopping(
            self._negative_log_likelihood, x0, 
            minimizer_kwargs=minimizer_kwargs,
            niter=50,
            seed=42
        )
        
        self.location, self.scale = result.x
        return self.location, self.scale
    
    def _fit_de(self, data):
        data_range = np.max(data) - np.min(data)
        data_min = np.min(data)
        
        bounds = [
            (data_min - data_range * 0.5, data_min + data_range * 1.5),
            (1e-6, data_range * 0.5)
        ]
        
        result = differential_evolution(
            self._negative_log_likelihood,
            bounds,
            args=(data,),
            seed=42,
            maxiter=100
        )
        
        self.location, self.scale = result.x
        return self.location, self.scale
    
    def fit(self, data, method='multistart'):
        data = np.asarray(data)
        self._fit_method = method
        
        methods = {
            'default': self._fit_default,
            'robust': self._fit_robust,
            'multistart': self._fit_multistart,
            'basinhopping': self._fit_basinhopping,
            'de': self._fit_de
        }
        
        if method not in methods:
            raise ValueError(f"未知的方法: {method}. 可用方法: {list(methods.keys())}")
        
        return methods[method](data)
    
    def fit_all_methods(self, data):
        data = np.asarray(data)
        results = {}
        
        methods = ['default', 'robust', 'multistart', 'basinhopping', 'de']
        
        for method in methods:
            try:
                loc, scale = self.fit(data, method=method)
                ll = self.log_likelihood(data)
                results[method] = {
                    'x0': loc,
                    'gamma': scale,
                    'log_likelihood': ll
                }
            except Exception as e:
                results[method] = {'error': str(e)}
        
        best_method = max(results.keys(), 
                         key=lambda k: results[k].get('log_likelihood', -np.inf))
        
        self.location = results[best_method]['x0']
        self.scale = results[best_method]['gamma']
        self._fit_method = f'best_{best_method}'
        
        return results, best_method
    
    def pdf(self, x):
        if self.location is None or self.scale is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        return stats.cauchy.pdf(x, loc=self.location, scale=self.scale)
    
    def get_params(self):
        if self.location is None or self.scale is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        return {'x0': self.location, 'gamma': self.scale, 'method': self._fit_method}
    
    def log_likelihood(self, data):
        if self.location is None or self.scale is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        return np.sum(stats.cauchy.logpdf(data, loc=self.location, scale=self.scale))
    
    def test_distribution(self, data, alpha=0.05):
        data = np.asarray(data)
        
        if self.location is None or self.scale is None:
            self.fit(data)
        
        cauchy_ks = self._tester.ks_test_cauchy(data, self.location, self.scale)
        cauchy_ll = self.log_likelihood(data)
        cauchy_aic = self._tester.aic(cauchy_ll, n_params=2)
        cauchy_bic = self._tester.bic(cauchy_ll, len(data), n_params=2)
        cauchy_qq = self._tester.qq_statistics(data, stats.cauchy, self.location, self.scale)
        
        normal_mu, normal_sigma = np.mean(data), np.std(data, ddof=1)
        normal_ks = self._tester.ks_test_normal(data, normal_mu, normal_sigma)
        normal_ll = np.sum(stats.norm.logpdf(data, loc=normal_mu, scale=normal_sigma))
        normal_aic = self._tester.aic(normal_ll, n_params=2)
        normal_bic = self._tester.bic(normal_ll, len(data), n_params=2)
        normal_qq = self._tester.qq_statistics(data, stats.norm, normal_mu, normal_sigma)
        
        is_cauchy_by_ks = cauchy_ks['p_value'] > alpha
        is_normal_by_ks = normal_ks['p_value'] > alpha
        
        preferred_by_aic = 'cauchy' if cauchy_aic < normal_aic else 'normal'
        preferred_by_bic = 'cauchy' if cauchy_bic < normal_bic else 'normal'
        
        qq_correlation_diff = cauchy_qq['correlation'] - normal_qq['correlation']
        preferred_by_qq = 'cauchy' if qq_correlation_diff > 0 else 'normal'
        
        votes = 0
        if preferred_by_aic == 'cauchy': votes += 1
        if preferred_by_bic == 'cauchy': votes += 1
        if preferred_by_qq == 'cauchy': votes += 1
        
        if votes >= 2:
            recommended_distribution = 'cauchy'
        elif votes <= 0:
            recommended_distribution = 'normal'
        else:
            recommended_distribution = 'uncertain'
        
        return {
            'cauchy': {
                'params': {'x0': self.location, 'gamma': self.scale},
                'ks_test': cauchy_ks,
                'log_likelihood': cauchy_ll,
                'aic': cauchy_aic,
                'bic': cauchy_bic,
                'qq_stats': {k: v for k, v in cauchy_qq.items() if k not in ['sorted_data', 'theoretical_quantiles']}
            },
            'normal': {
                'params': {'mu': normal_mu, 'sigma': normal_sigma},
                'ks_test': normal_ks,
                'log_likelihood': normal_ll,
                'aic': normal_aic,
                'bic': normal_bic,
                'qq_stats': {k: v for k, v in normal_qq.items() if k not in ['sorted_data', 'theoretical_quantiles']}
            },
            'comparison': {
                'is_cauchy_by_ks': is_cauchy_by_ks,
                'is_normal_by_ks': is_normal_by_ks,
                'preferred_by_aic': preferred_by_aic,
                'preferred_by_bic': preferred_by_bic,
                'preferred_by_qq_correlation': preferred_by_qq,
                'recommended_distribution': recommended_distribution,
                'aic_diff': normal_aic - cauchy_aic,
                'bic_diff': normal_bic - cauchy_bic,
                'qq_correlation_diff': qq_correlation_diff
            },
            'qq_data': {
                'cauchy': {
                    'sorted_data': cauchy_qq['sorted_data'],
                    'theoretical_quantiles': cauchy_qq['theoretical_quantiles']
                },
                'normal': {
                    'sorted_data': normal_qq['sorted_data'],
                    'theoretical_quantiles': normal_qq['theoretical_quantiles']
                }
            }
        }
    
    def outlier_robustness_test(self, data, n_outliers=5, outlier_magnitude=10):
        data = np.asarray(data).copy()
        n = len(data)
        
        if self.location is None or self.scale is None:
            self.fit(data)
        
        original_x0, original_gamma = self.location, self.scale
        
        normal_mu_original = np.mean(data)
        normal_sigma_original = np.std(data, ddof=1)
        
        outlier_positions = np.random.choice(n, n_outliers, replace=False)
        outlier_values = original_x0 + outlier_magnitude * original_gamma * np.random.choice([-1, 1], n_outliers)
        data_with_outliers = data.copy()
        data_with_outliers[outlier_positions] = outlier_values
        
        self.fit(data_with_outliers)
        cauchy_x0_outlier, cauchy_gamma_outlier = self.location, self.scale
        
        normal_mu_outlier = np.mean(data_with_outliers)
        normal_sigma_outlier = np.std(data_with_outliers, ddof=1)
        
        cauchy_x0_change = abs(cauchy_x0_outlier - original_x0) / (original_gamma + 1e-10)
        cauchy_gamma_change = abs(cauchy_gamma_outlier - original_gamma) / (original_gamma + 1e-10)
        
        normal_mu_change = abs(normal_mu_outlier - normal_mu_original) / (normal_sigma_original + 1e-10)
        normal_sigma_change = abs(normal_sigma_outlier - normal_sigma_original) / (normal_sigma_original + 1e-10)
        
        robustness_ratio_x0 = normal_mu_change / (cauchy_x0_change + 1e-10)
        robustness_ratio_gamma = normal_sigma_change / (cauchy_gamma_change + 1e-10)
        
        return {
            'original': {
                'cauchy': {'x0': original_x0, 'gamma': original_gamma},
                'normal': {'mu': normal_mu_original, 'sigma': normal_sigma_original}
            },
            'with_outliers': {
                'cauchy': {'x0': cauchy_x0_outlier, 'gamma': cauchy_gamma_outlier},
                'normal': {'mu': normal_mu_outlier, 'sigma': normal_sigma_outlier}
            },
            'changes': {
                'cauchy_x0_relative_change': cauchy_x0_change,
                'cauchy_gamma_relative_change': cauchy_gamma_change,
                'normal_mu_relative_change': normal_mu_change,
                'normal_sigma_relative_change': normal_sigma_change,
                'robustness_ratio_x0': robustness_ratio_x0,
                'robustness_ratio_gamma': robustness_ratio_gamma
            },
            'outlier_info': {
                'n_outliers': n_outliers,
                'outlier_magnitude': outlier_magnitude,
                'outlier_positions': outlier_positions,
                'outlier_values': outlier_values
            },
            'conclusion': {
                'cauchy_more_robust_x0': robustness_ratio_x0 > 1,
                'cauchy_more_robust_gamma': robustness_ratio_gamma > 1,
                'robustness_ratio_x0': robustness_ratio_x0,
                'robustness_ratio_gamma': robustness_ratio_gamma
            }
        }

def estimate_cauchy_from_array(data_array, method='multistart'):
    estimator = CauchyEstimator()
    location, scale = estimator.fit(data_array, method=method)
    return location, scale, estimator

def generate_sample_data(true_location=0.0, true_scale=1.0, size=1000, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    return stats.cauchy.rvs(loc=true_location, scale=true_scale, size=size)

def generate_normal_data(mu=0.0, sigma=1.0, size=1000, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    return stats.norm.rvs(loc=mu, scale=sigma, size=size)

if __name__ == "__main__":
    print("柯西分布参数估计 - 增强版（含分布检验与异常值稳健性分析）")
    print("=" * 80)
    
    np.random.seed(42)
    
    print("\n=== 测试1: 真实柯西数据的分布检验 ===")
    true_x0 = 2.0
    true_gamma = 1.5
    cauchy_data = generate_sample_data(true_x0, true_gamma, size=200, random_seed=42)
    
    estimator = CauchyEstimator()
    estimator.fit(cauchy_data, method='multistart')
    
    test_results = estimator.test_distribution(cauchy_data)
    
    print(f"真实参数: x0={true_x0}, γ={true_gamma}")
    print(f"估计参数: x0={estimator.location:.4f}, γ={estimator.scale:.4f}")
    print()
    print("柯西分布检验:")
    print(f"  KS检验: 统计量={test_results['cauchy']['ks_test']['statistic']:.4f}, "
          f"p值={test_results['cauchy']['ks_test']['p_value']:.4f}")
    print(f"  AIC={test_results['cauchy']['aic']:.2f}, BIC={test_results['cauchy']['bic']:.2f}")
    print(f"  QQ相关系数={test_results['cauchy']['qq_stats']['correlation']:.4f}")
    print()
    print("正态分布检验:")
    print(f"  KS检验: 统计量={test_results['normal']['ks_test']['statistic']:.4f}, "
          f"p值={test_results['normal']['ks_test']['p_value']:.4f}")
    print(f"  AIC={test_results['normal']['aic']:.2f}, BIC={test_results['normal']['bic']:.2f}")
    print(f"  QQ相关系数={test_results['normal']['qq_stats']['correlation']:.4f}")
    print()
    print(f"推荐分布: {test_results['comparison']['recommended_distribution'].upper()}")
    print(f"  AIC偏好: {test_results['comparison']['preferred_by_aic']}")
    print(f"  BIC偏好: {test_results['comparison']['preferred_by_bic']}")
    print(f"  QQ相关系数偏好: {test_results['comparison']['preferred_by_qq_correlation']}")
    
    print("\n=== 测试2: 真实正态数据的分布检验 ===")
    normal_data = generate_normal_data(2.0, 1.5, size=200, random_seed=123)
    
    estimator2 = CauchyEstimator()
    estimator2.fit(normal_data, method='multistart')
    test_results2 = estimator2.test_distribution(normal_data)
    
    print(f"推荐分布: {test_results2['comparison']['recommended_distribution'].upper()}")
    print(f"  AIC差值(正态-柯西): {test_results2['comparison']['aic_diff']:.2f}")
    print(f"  BIC差值(正态-柯西): {test_results2['comparison']['bic_diff']:.2f}")
    
    print("\n=== 测试3: 异常值稳健性检验 ===")
    base_data = generate_sample_data(0.0, 1.0, size=100, random_seed=456)
    
    estimator3 = CauchyEstimator()
    estimator3.fit(base_data, method='de')
    
    robustness_results = estimator3.outlier_robustness_test(base_data, n_outliers=5, outlier_magnitude=10)
    
    print("参数变化对比(相对尺度):")
    print(f"  柯西位置参数变化: {robustness_results['changes']['cauchy_x0_relative_change']:.4f}")
    print(f"  正态均值变化:     {robustness_results['changes']['normal_mu_relative_change']:.4f}")
    print(f"  柯西尺度参数变化: {robustness_results['changes']['cauchy_gamma_relative_change']:.4f}")
    print(f"  正态标准差变化:   {robustness_results['changes']['normal_sigma_relative_change']:.4f}")
    print()
    print("稳健性比率(正态变化/柯西变化，越大说明柯西越稳健):")
    print(f"  位置参数稳健性比率: {robustness_results['changes']['robustness_ratio_x0']:.2f}x")
    print(f"  尺度参数稳健性比率: {robustness_results['changes']['robustness_ratio_gamma']:.2f}x")
    print()
    print(f"结论: 柯西分布对异常值更稳健?")
    print(f"  位置参数: {'是' if robustness_results['conclusion']['cauchy_more_robust_x0'] else '否'}")
    print(f"  尺度参数: {'是' if robustness_results['conclusion']['cauchy_more_robust_gamma'] else '否'}")
    
    print("\n" + "=" * 80)
    print("使用示例:")
    print("  from cauchy_estimator_enhanced import CauchyEstimator")
    print("  estimator = CauchyEstimator()")
    print("  estimator.fit(data)")
    print("  test_result = estimator.test_distribution(data)")
    print("  print(test_result['comparison']['recommended_distribution'])")
    print("  robustness = estimator.outlier_robustness_test(data)")
    print("=" * 80)
