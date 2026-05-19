import numpy as np
from scipy import stats
from scipy.optimize import minimize, basinhopping, differential_evolution

class CauchyEstimator:
    def __init__(self):
        self.location = None
        self.scale = None
        self._fit_method = None
    
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

def estimate_cauchy_from_array(data_array, method='multistart'):
    estimator = CauchyEstimator()
    location, scale = estimator.fit(data_array, method=method)
    return location, scale, estimator

def generate_sample_data(true_location=0.0, true_scale=1.0, size=1000, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    return stats.cauchy.rvs(loc=true_location, scale=true_scale, size=size)

if __name__ == "__main__":
    print("柯西分布参数估计 - 改进版（解决似然函数多峰问题）")
    print("=" * 70)
    
    true_x0 = 2.0
    true_gamma = 1.5
    
    np.random.seed(42)
    data = generate_sample_data(true_location=true_x0, true_scale=true_gamma, size=500, random_seed=42)
    
    print(f"真实参数: x0 = {true_x0}, γ = {true_gamma}")
    print(f"样本量: {len(data)}")
    print()
    
    estimator = CauchyEstimator()
    
    print("各方法估计结果对比:")
    print("-" * 70)
    print(f"{'方法':<15} {'x0估计':<12} {'γ估计':<12} {'对数似然':<15} {'|Δx0|':<10} {'|Δγ|':<10}")
    print("-" * 70)
    
    methods = ['default', 'robust', 'multistart', 'basinhopping', 'de']
    
    for method in methods:
        try:
            loc, scale = estimator.fit(data, method=method)
            ll = estimator.log_likelihood(data)
            err_x0 = abs(loc - true_x0)
            err_gamma = abs(scale - true_gamma)
            print(f"{method:<15} {loc:<12.6f} {scale:<12.6f} {ll:<15.2f} {err_x0:<10.6f} {err_gamma:<10.6f}")
        except Exception as e:
            print(f"{method:<15} {'失败':<12} {'失败':<12} {'N/A':<15} {'N/A':<10} {'N/A':<10}")
    
    print("-" * 70)
    print()
    
    print("使用 fit_all_methods() 自动选择最优方法:")
    results, best_method = estimator.fit_all_methods(data)
    
    print(f"\n最优方法: {best_method}")
    print(f"最优参数: x0 = {estimator.location:.6f}, γ = {estimator.scale:.6f}")
    print(f"最优对数似然: {estimator.log_likelihood(data):.2f}")
    print()
    
    print("方法说明:")
    print("  'default'    : SciPy默认方法（容易陷入局部最优）")
    print("  'robust'     : 使用中位数和四分位距作为初始值")
    print("  'multistart' : 多起点局部优化（推荐，平衡精度与速度）")
    print("  'basinhopping': Basin-Hopping全局优化")
    print("  'de'         : 差分进化全局优化（最可靠但较慢）")
    print()
    print("使用示例:")
    print("  from cauchy_estimator_improved import CauchyEstimator")
    print("  estimator = CauchyEstimator()")
    print("  estimator.fit(data, method='multistart')")
    print("  params = estimator.get_params()")
    print("  print(params)")
