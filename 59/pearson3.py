import numpy as np
from scipy import stats
from scipy.special import gamma
from scipy.optimize import minimize
import warnings


class PearsonIIIFitter:
    def __init__(self, data, outliers=None):
        self.data = np.array(data, dtype=np.float64)
        self.n = len(self.data)
        if self.n < 3:
            raise ValueError("数据样本量至少需要3个")
        self.params = None
        self.moments = None
        self.l_moments = None
        self.outliers = outliers if outliers is not None else []

    def calculate_moments(self):
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)
        cv = std / mean
        
        skew = np.sum((self.data - mean)**3) / (self.n * std**3)
        skew = skew * self.n / (self.n - 2)
        
        self.moments = {
            'mean': mean,
            'std': std,
            'cv': cv,
            'cs': skew
        }
        return self.moments

    def calculate_l_moments(self, robust=False, trim_percent=0.05):
        sorted_data = np.sort(self.data)
        n = self.n
        
        if robust:
            trim_num = int(np.floor(n * trim_percent))
            sorted_data = sorted_data[trim_num:n-trim_num]
            n_trimmed = len(sorted_data)
        else:
            n_trimmed = n
        
        b0 = np.mean(sorted_data)
        
        b1 = 0.0
        for i in range(n_trimmed):
            b1 += (i / (n_trimmed - 1)) * sorted_data[i]
        b1 /= n_trimmed
        
        b2 = 0.0
        for i in range(n_trimmed):
            b2 += (i * (i - 1) / ((n_trimmed - 1) * (n_trimmed - 2))) * sorted_data[i]
        b2 /= n_trimmed
        
        l1 = b0
        l2 = 2 * b1 - b0
        l3 = 6 * b2 - 6 * b1 + b0
        
        l_cv = l2 / l1 if l1 != 0 else 0
        l_skew = l3 / l2 if l2 != 0 else 0
        
        self.l_moments = {
            'l1': l1,
            'l2': l2,
            'l3': l3,
            'l_cv': l_cv,
            'l_skew': l_skew,
            'trim_percent': trim_percent if robust else 0
        }
        return self.l_moments

    def calculate_trimmed_l_moments(self, trim_left=0.05, trim_right=0.1):
        sorted_data = np.sort(self.data)
        n = self.n
        
        trim_left_num = int(np.floor(n * trim_left))
        trim_right_num = int(np.floor(n * trim_right))
        
        trimmed_data = sorted_data[trim_left_num:n-trim_right_num]
        n_trimmed = len(trimmed_data)
        
        if n_trimmed < 3:
            raise ValueError("截断后样本量不足")
        
        b0 = np.mean(trimmed_data)
        
        b1 = 0.0
        for i in range(n_trimmed):
            b1 += (i / (n_trimmed - 1)) * trimmed_data[i]
        b1 /= n_trimmed
        
        b2 = 0.0
        for i in range(n_trimmed):
            b2 += (i * (i - 1) / ((n_trimmed - 1) * (n_trimmed - 2))) * trimmed_data[i]
        b2 /= n_trimmed
        
        l1 = b0
        l2 = 2 * b1 - b0
        l3 = 6 * b2 - 6 * b1 + b0
        
        l_cv = l2 / l1 if l1 != 0 else 0
        l_skew = l3 / l2 if l2 != 0 else 0
        
        self.l_moments = {
            'l1': l1,
            'l2': l2,
            'l3': l3,
            'l_cv': l_cv,
            'l_skew': l_skew,
            'trim_left': trim_left,
            'trim_right': trim_right
        }
        return self.l_moments

    def calculate_weighted_l_moments(self, weight_type='huber', c=1.5):
        sorted_data = np.sort(self.data)
        n = self.n
        
        median = np.median(sorted_data)
        mad = np.median(np.abs(sorted_data - median))
        if mad == 0:
            mad = np.std(sorted_data) / 0.6745
        
        weights = np.ones(n)
        for i in range(n):
            residual = (sorted_data[i] - median) / mad
            if weight_type == 'huber':
                if abs(residual) > c:
                    weights[i] = c / abs(residual)
            elif weight_type == 'tukey':
                if abs(residual) <= c:
                    weights[i] = (1 - (residual / c)**2)**2
                else:
                    weights[i] = 0
        
        weights = weights / np.sum(weights) * n
        
        b0 = np.sum(weights * sorted_data) / n
        
        b1 = 0.0
        for i in range(n):
            weight_sum = np.sum(weights[:i+1])
            b1 += ((weight_sum - weights[i]) / (n - 1)) * sorted_data[i] * weights[i]
        b1 /= n
        
        b2 = 0.0
        for i in range(n):
            weight_sum = np.sum(weights[:i+1])
            term = ((weight_sum - weights[i]) * (weight_sum - 2 * weights[i])) / ((n - 1) * (n - 2))
            b2 += term * sorted_data[i] * weights[i]
        b2 /= n
        
        l1 = b0
        l2 = 2 * b1 - b0
        l3 = 6 * b2 - 6 * b1 + b0
        
        l_cv = l2 / l1 if l1 != 0 else 0
        l_skew = l3 / l2 if l2 != 0 else 0
        
        self.l_moments = {
            'l1': l1,
            'l2': l2,
            'l3': l3,
            'l_cv': l_cv,
            'l_skew': l_skew,
            'weight_type': weight_type,
            'c': c
        }
        return self.l_moments

    def fit(self, method='moments', robust=False, **kwargs):
        if method == 'moments':
            self.calculate_moments()
            mean = self.moments['mean']
            std = self.moments['std']
            cs = self.moments['cs']
            
            if abs(cs) < 0.001:
                cs = 0.001
            
            alpha = 4.0 / (cs ** 2)
            beta = 2.0 / (std * cs)
            a0 = mean - 2.0 * std / cs
            
            self.params = {
                'a0': a0,
                'alpha': alpha,
                'beta': beta,
                'mean': mean,
                'std': std,
                'cs': cs,
                'cv': self.moments['cv'],
                'method': 'moments'
            }
        
        elif method == 'l_moments':
            if robust:
                trim_percent = kwargs.get('trim_percent', 0.05)
                self.calculate_l_moments(robust=True, trim_percent=trim_percent)
            else:
                self.calculate_l_moments(robust=False)
            
            l1 = self.l_moments['l1']
            l2 = self.l_moments['l2']
            l_skew = self.l_moments['l_skew']
            
            if abs(l_skew) < 0.001:
                l_skew = 0.001
            
            cs = self._l_skew_to_cs(l_skew)
            std = l2 * np.sqrt(np.pi)
            
            alpha = 4.0 / (cs ** 2)
            beta = 2.0 / (std * cs)
            a0 = l1 - 2.0 * std / cs
            
            self.params = {
                'a0': a0,
                'alpha': alpha,
                'beta': beta,
                'mean': l1,
                'std': std,
                'cs': cs,
                'cv': std / l1 if l1 != 0 else 0,
                'method': 'l_moments',
                'robust': robust
            }
        
        elif method == 'trimmed_l_moments':
            trim_left = kwargs.get('trim_left', 0.05)
            trim_right = kwargs.get('trim_right', 0.1)
            self.calculate_trimmed_l_moments(trim_left, trim_right)
            
            l1 = self.l_moments['l1']
            l2 = self.l_moments['l2']
            l_skew = self.l_moments['l_skew']
            
            if abs(l_skew) < 0.001:
                l_skew = 0.001
            
            cs = self._l_skew_to_cs(l_skew)
            std = l2 * np.sqrt(np.pi)
            
            alpha = 4.0 / (cs ** 2)
            beta = 2.0 / (std * cs)
            a0 = l1 - 2.0 * std / cs
            
            self.params = {
                'a0': a0,
                'alpha': alpha,
                'beta': beta,
                'mean': l1,
                'std': std,
                'cs': cs,
                'cv': std / l1 if l1 != 0 else 0,
                'method': 'trimmed_l_moments',
                'trim_left': trim_left,
                'trim_right': trim_right
            }
        
        elif method == 'weighted_l_moments':
            weight_type = kwargs.get('weight_type', 'huber')
            c = kwargs.get('c', 1.5)
            self.calculate_weighted_l_moments(weight_type, c)
            
            l1 = self.l_moments['l1']
            l2 = self.l_moments['l2']
            l_skew = self.l_moments['l_skew']
            
            if abs(l_skew) < 0.001:
                l_skew = 0.001
            
            cs = self._l_skew_to_cs(l_skew)
            std = l2 * np.sqrt(np.pi)
            
            alpha = 4.0 / (cs ** 2)
            beta = 2.0 / (std * cs)
            a0 = l1 - 2.0 * std / cs
            
            self.params = {
                'a0': a0,
                'alpha': alpha,
                'beta': beta,
                'mean': l1,
                'std': std,
                'cs': cs,
                'cv': std / l1 if l1 != 0 else 0,
                'method': 'weighted_l_moments',
                'weight_type': weight_type,
                'c': c
            }
        
        else:
            raise ValueError(f"不支持的拟合方法: {method}")
        
        return self.params

    def _l_skew_to_cs(self, l_skew):
        if l_skew <= 0 or l_skew >= 1:
            return 2.0
        
        def equation(cs):
            if cs <= 0:
                return 1e10
            
            alpha = 4.0 / (cs ** 2)
            from scipy.special import gamma as gamma_func
            
            t3 = (gamma_func(alpha + 1/3) / gamma_func(alpha)) ** 3
            l_skew_calc = (3 * t3 - 1) / (1 - t3)
            
            return (l_skew_calc - l_skew) ** 2
        
        result = minimize(equation, x0=2.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
        cs_opt = result.x[0]
        
        return cs_opt

    def _phi(self, p):
        cs = self.params['cs']
        alpha = self.params['alpha']
        
        if abs(cs) < 1e-6:
            return stats.norm.ppf(1 - p)
        
        def objective(x):
            from scipy.stats import gamma as gamma_dist
            return gamma_dist.cdf(x, a=alpha, scale=2/cs) - (1 - p)
        
        from scipy.optimize import brentq
        
        try:
            if p < 0.5:
                x_min, x_max = 0, 100
            else:
                x_min, x_max = -50, 50
            
            phi = brentq(objective, x_min, x_max)
        except:
            phi = self._approx_phi(1 - p)
        
        return phi

    def _approx_phi(self, p):
        cs = self.params['cs']
        z = stats.norm.ppf(p)
        
        phi = z + (z**2 - 1) * cs / 6 + \
              (z**3 - 6 * z) * cs**2 / 36 - \
              (z**4 - 5 * z**2 + 2) * cs**3 / 81 + \
              (3 * z**5 - 37 * z**3 + 71 * z) * cs**4 / 3888
        
        return phi

    def pdf(self, x):
        if self.params is None:
            raise RuntimeError("请先调用fit()方法拟合分布")
        
        a0 = self.params['a0']
        alpha = self.params['alpha']
        beta = self.params['beta']
        
        if x <= a0:
            return 0.0
        
        x_shifted = x - a0
        return (beta**alpha / gamma(alpha)) * (x_shifted**(alpha - 1)) * np.exp(-beta * x_shifted)

    def cdf(self, x):
        if self.params is None:
            raise RuntimeError("请先调用fit()方法拟合分布")
        
        a0 = self.params['a0']
        alpha = self.params['alpha']
        beta = self.params['beta']
        
        if x <= a0:
            return 0.0
        
        x_shifted = x - a0
        from scipy.stats import gamma as gamma_dist
        return gamma_dist.cdf(x_shifted * beta, a=alpha)

    def quantile(self, p):
        if self.params is None:
            raise RuntimeError("请先调用fit()方法拟合分布")
        
        mean = self.params['mean']
        std = self.params['std']
        phi = self._phi(p)
        
        return mean + std * phi

    def design_flood(self, T):
        p = 1.0 / T
        return self.quantile(p)

    def get_return_period(self, x):
        cdf_val = self.cdf(x)
        if cdf_val >= 1.0:
            return float('inf')
        return 1.0 / (1.0 - cdf_val)


def calculate_design_flood(flood_series, return_periods, method='moments', **kwargs):
    fitter = PearsonIIIFitter(flood_series)
    params = fitter.fit(method=method, **kwargs)
    
    results = {
        'parameters': params,
        'design_floods': {}
    }
    
    for T in return_periods:
        results['design_floods'][T] = fitter.design_flood(T)
    
    return results


def compare_methods_with_outliers():
    print("=" * 80)
    print("不同方法对异常值（特大洪水）的稳健性比较")
    print("=" * 80)
    
    base_data = [820, 960, 1050, 780, 1200, 920, 1100, 890, 1020, 950,
                 1300, 880, 1080, 990, 1150, 850, 1000, 1120, 930, 1060]
    
    data_with_outlier = base_data + [2500, 2800]
    
    methods = [
        ('传统矩法', 'moments', {}),
        ('传统L-矩法', 'l_moments', {'robust': False}),
        ('稳健L-矩法(截断5%)', 'l_moments', {'robust': True, 'trim_percent': 0.05}),
        ('截断L-矩法(5%,10%)', 'trimmed_l_moments', {'trim_left': 0.05, 'trim_right': 0.1}),
        ('加权L-矩法(Huber)', 'weighted_l_moments', {'weight_type': 'huber', 'c': 1.5}),
        ('加权L-矩法(Tukey)', 'weighted_l_moments', {'weight_type': 'tukey', 'c': 4.685}),
    ]
    
    print(f"\n{'方法':<25} {'均值':<10} {'Cs':<10} {'100年一遇':<15}")
    print("-" * 65)
    
    for name, method, kwargs in methods:
        fitter = PearsonIIIFitter(data_with_outlier)
        params = fitter.fit(method=method, **kwargs)
        design_100 = fitter.design_flood(100)
        print(f"{name:<25} {params['mean']:<10.1f} {params['cs']:<10.4f} {design_100:<15.1f}")
    
    print("=" * 80)


def main():
    compare_methods_with_outliers()
    print()
    
    flood_data = [820, 960, 1050, 780, 1200, 920, 1100, 890, 1020, 950,
                  1300, 880, 1080, 990, 1150, 850, 1000, 1120, 930, 1060]
    
    return_periods = [10, 20, 50, 100, 200, 500, 1000]
    
    results = calculate_design_flood(flood_data, return_periods, method='l_moments', robust=True)
    
    print("=" * 60)
    print("皮尔逊III型分布参数估计结果（稳健L-矩法）")
    print("=" * 60)
    params = results['parameters']
    print(f"均值 (μ): {params['mean']:.2f} m³/s")
    print(f"标准差 (σ): {params['std']:.2f}")
    print(f"变差系数 (Cv): {params['cv']:.4f}")
    print(f"偏态系数 (Cs): {params['cs']:.4f}")
    print(f"位置参数 (a0): {params['a0']:.2f}")
    print(f"形状参数 (α): {params['alpha']:.4f}")
    print(f"尺度参数 (β): {params['beta']:.6f}")
    print()
    
    print("=" * 60)
    print("设计洪水计算结果")
    print("=" * 60)
    print(f"{'重现期(年)':<12} {'频率(%)':<12} {'设计洪水(m³/s)':<15}")
    print("-" * 60)
    for T in return_periods:
        p = 100.0 / T
        flood = results['design_floods'][T]
        print(f"{T:<12} {p:<12.4f} {flood:<15.2f}")
    print("=" * 60)


class NonstationaryPearsonIII:
    def __init__(self, data, covariates):
        self.data = np.array(data, dtype=np.float64)
        self.covariates = np.array(covariates, dtype=np.float64)
        
        if self.covariates.ndim == 1:
            self.covariates = self.covariates.reshape(-1, 1)
        
        self.n = len(self.data)
        self.n_covariates = self.covariates.shape[1]
        
        if self.n != self.covariates.shape[0]:
            raise ValueError("数据和协变量长度不匹配")
        
        self.params = None
        self.model = None

    def _design_matrix(self, covariate_idx=0):
        cov = self.covariates[:, covariate_idx]
        X = np.column_stack([np.ones(self.n), cov])
        return X

    def _location_function(self, theta, X):
        return X @ theta

    def _scale_function(self, gamma, X):
        return np.exp(X @ gamma)

    def _negative_log_likelihood(self, params):
        n_params = len(params)
        n_loc_params = 2 + self.n_covariates - 1
        
        theta = params[:2]
        gamma = params[2:4]
        alpha = params[4]
        
        X = self._design_matrix(0)
        
        mu = self._location_function(theta, X)
        sigma = self._scale_function(gamma, X)
        
        beta = alpha / sigma
        
        valid = (beta > 0) & (self.data > mu)
        
        if not np.all(valid):
            return 1e10
        
        from scipy.special import gamma as gamma_func
        
        log_likelihood = -alpha * np.log(sigma)
        log_likelihood += (alpha - 1) * np.log(self.data - mu)
        log_likelihood -= alpha * (self.data - mu) / sigma
        log_likelihood -= self.n * np.log(gamma_func(alpha))
        
        return -np.sum(log_likelihood)

    def fit(self, method='MLE'):
        if method == 'MLE':
            X = self._design_matrix(0)
            
            initial_mu = np.mean(self.data)
            initial_sigma = np.std(self.data)
            initial_alpha = 4.0
            
            initial_params = [
                initial_mu - 0.1 * initial_mu, 0.0,
                np.log(initial_sigma), 0.0,
                initial_alpha
            ]
            
            bounds = [
                (None, None), (None, None),
                (None, None), (None, None),
                (0.1, 10.0)
            ]
            
            result = minimize(
                self._negative_log_likelihood,
                x0=initial_params,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            
            if result.success:
                self.params = {
                    'theta': result.x[:2],
                    'gamma': result.x[2:4],
                    'alpha': result.x[4],
                    'method': 'MLE',
                    'converged': True,
                    'optimization_result': result
                }
            else:
                self.params = {
                    'method': 'MLE',
                    'converged': False,
                    'message': result.message
                }
                warnings.warn(f"优化未收敛: {result.message}")
        
        return self.params

    def get_time_varying_params(self, covariate_values=None):
        if self.params is None:
            raise RuntimeError("请先调用fit()方法拟合模型")
        
        if covariate_values is None:
            X = self._design_matrix(0)
        else:
            covariate_values = np.array(covariate_values)
            if covariate_values.ndim == 1:
                covariate_values = covariate_values.reshape(-1, 1)
            n = len(covariate_values)
            X = np.column_stack([np.ones(n), covariate_values[:, 0]])
        
        mu = self._location_function(self.params['theta'], X)
        sigma = self._scale_function(self.params['gamma'], X)
        alpha = self.params['alpha']
        beta = alpha / sigma
        a0 = mu - alpha / beta
        
        return {
            'location': mu,
            'scale': sigma,
            'shape': alpha,
            'beta': beta,
            'a0': a0
        }

    def pdf(self, x, covariate_value):
        params = self.get_time_varying_params([covariate_value])
        mu = params['location'][0]
        sigma = params['scale'][0]
        alpha = params['shape'][0]
        
        if x <= mu - alpha * (sigma / alpha):
            return 0.0
        
        from scipy.stats import gamma as gamma_dist
        return gamma_dist.pdf(x - mu + sigma, a=alpha, scale=sigma/alpha)

    def cdf(self, x, covariate_value):
        params = self.get_time_varying_params([covariate_value])
        mu = params['location'][0]
        sigma = params['scale'][0]
        alpha = params['shape'][0]
        
        if x <= mu - alpha * (sigma / alpha):
            return 0.0
        
        from scipy.stats import gamma as gamma_dist
        return gamma_dist.cdf(x - mu + sigma, a=alpha, scale=sigma/alpha)

    def quantile(self, p, covariate_value):
        params = self.get_time_varying_params([covariate_value])
        mu = params['location'][0]
        sigma = params['scale'][0]
        alpha = params['shape'][0]
        
        from scipy.stats import gamma as gamma_dist
        gamma_quantile = gamma_dist.ppf(1 - p, a=alpha, scale=sigma/alpha)
        
        return mu - sigma + gamma_quantile

    def design_flood(self, T, covariate_value):
        p = 1.0 / T
        return self.quantile(p, covariate_value)

    def trend_test(self):
        if self.params is None:
            raise RuntimeError("请先调用fit()方法拟合模型")
        
        theta = self.params['theta']
        gamma = self.params['gamma']
        
        results = {
            'location_trend': theta[1],
            'location_intercept': theta[0],
            'scale_trend': gamma[1],
            'scale_intercept': gamma[0],
            'shape_param': self.params['alpha']
        }
        
        return results

    def plot_trend(self, return_periods=None, save_path=None):
        import matplotlib.pyplot as plt
        
        if return_periods is None:
            return_periods = [10, 20, 50, 100]
        
        cov_range = np.linspace(np.min(self.covariates[:, 0]), 
                                np.max(self.covariates[:, 0]), 100)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        tv_params = self.get_time_varying_params(cov_range)
        
        axes[0].plot(cov_range, tv_params['location'], 'b-', linewidth=2, label='位置参数 μ')
        axes[0].set_xlabel('协变量')
        axes[0].set_ylabel('位置参数')
        axes[0].set_title('位置参数随协变量变化')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(cov_range, tv_params['scale'], 'r-', linewidth=2, label='尺度参数 σ')
        axes[1].set_xlabel('协变量')
        axes[1].set_ylabel('尺度参数')
        axes[1].set_title('尺度参数随协变量变化')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class NonstationaryP3Advanced(NonstationaryPearsonIII):
    def __init__(self, data, covariates):
        super().__init__(data, covariates)
        self.models = {}
        
    def fit_multi_model(self):
        models = {
            'stationary': {'location_cov': [], 'scale_cov': []},
            'mu_trend': {'location_cov': [0], 'scale_cov': []},
            'sigma_trend': {'location_cov': [], 'scale_cov': [0]},
            'both_trend': {'location_cov': [0], 'scale_cov': [0]}
        }
        
        results = {}
        
        for name, model_config in models.items():
            result = self._fit_specific_model(model_config)
            results[name] = result
        
        best_model = min(results.keys(), 
                        key=lambda k: results[k].get('aic', float('inf')))
        
        self.models = results
        self.best_model = best_model
        
        return results, best_model
    
    def _fit_specific_model(self, config):
        n_loc_params = 1 + len(config['location_cov'])
        n_scale_params = 1 + len(config['scale_cov'])
        
        def nll(params):
            theta = params[:n_loc_params]
            gamma = params[n_loc_params:n_loc_params+n_scale_params]
            alpha = params[-1]
            
            X_mu = np.ones(self.n)
            for idx in config['location_cov']:
                X_mu = np.column_stack([X_mu, self.covariates[:, idx]])
            
            mu = X_mu @ theta
            
            X_sigma = np.ones(self.n)
            for idx in config['scale_cov']:
                X_sigma = np.column_stack([X_sigma, self.covariates[:, idx]])
            
            sigma = np.exp(X_sigma @ gamma)
            
            if np.any(sigma <= 0) or np.any(self.data <= mu - alpha * (sigma/alpha)):
                return 1e10
            
            from scipy.special import gamma as gamma_func
            
            log_pdf = -alpha * np.log(sigma)
            log_pdf += (alpha - 1) * np.log(self.data - mu + sigma)
            log_pdf -= alpha * (self.data - mu + sigma) / sigma
            log_pdf -= np.log(gamma_func(alpha))
            
            return -np.sum(log_pdf)
        
        initial_mu = np.mean(self.data)
        initial_sigma = np.std(self.data)
        
        initial = []
        initial.append(initial_mu)
        initial.extend([0.0] * len(config['location_cov']))
        initial.append(np.log(initial_sigma))
        initial.extend([0.0] * len(config['scale_cov']))
        initial.append(4.0)
        
        bounds = []
        bounds.extend([(None, None)] * (1 + len(config['location_cov'])))
        bounds.extend([(None, None)] * (1 + len(config['scale_cov'])))
        bounds.append((0.1, 10.0))
        
        result = minimize(nll, x0=initial, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            k = len(result.x)
            aic = 2 * k + 2 * result.fun
            bic = k * np.log(self.n) + 2 * result.fun
            
            return {
                'success': True,
                'params': result.x,
                'nll': result.fun,
                'aic': aic,
                'bic': bic,
                'config': config
            }
        else:
            return {'success': False, 'message': result.message}

    def get_design_flood_trend(self, T, covariate_values=None):
        if covariate_values is None:
            covariate_values = self.covariates[:, 0]
        
        design_floods = []
        for cov in covariate_values:
            df = self.design_flood(T, cov)
            design_floods.append(df)
        
        return np.array(design_floods)


def demonstrate_nonstationary():
    print("=" * 80)
    print("非平稳P-III分布频率分析演示")
    print("=" * 80)
    
    np.random.seed(42)
    n_years = 50
    
    time = np.arange(n_years)
    trend = 2.0 * time / n_years
    
    true_mu0 = 1000
    true_mu1 = 50
    true_sigma0 = np.log(150)
    true_sigma1 = 0.3
    true_alpha = 3.0
    
    mu = true_mu0 + true_mu1 * trend
    sigma = np.exp(true_sigma0 + true_sigma1 * trend)
    
    from scipy.stats import gamma as gamma_dist
    flood_data = mu - sigma + gamma_dist.rvs(a=true_alpha, scale=sigma/true_alpha, size=n_years)
    
    print(f"\n模拟数据: {n_years}年洪水序列")
    print(f"真实趋势: 位置参数 μ = {true_mu0} + {true_mu1} * 时间")
    print(f"真实趋势: 尺度参数 log(σ) = {true_sigma0:.3f} + {true_sigma1} * 时间")
    print(f"真实形状参数 α = {true_alpha}")
    print()
    
    model = NonstationaryPearsonIII(flood_data, trend)
    params = model.fit()
    
    if params['converged']:
        print("模型拟合结果 (极大似然估计):")
        print(f"  位置参数: μ = {params['theta'][0]:.2f} + {params['theta'][1]:.2f} * 协变量")
        print(f"  尺度参数: log(σ) = {params['gamma'][0]:.3f} + {params['gamma'][1]:.3f} * 协变量")
        print(f"  形状参数: α = {params['alpha']:.4f}")
        print()
        
        trend_results = model.trend_test()
        print("趋势检验结果:")
        print(f"  位置参数趋势: {trend_results['location_trend']:.2f}")
        print(f"  尺度参数趋势: {trend_results['scale_trend']:.3f}")
        print()
        
        print("时变设计洪水:")
        test_covariates = [0.0, 0.5, 1.0]
        for T in [20, 50, 100]:
            print(f"\n  {T}年一遇设计洪水:")
            for cov in test_covariates:
                df = model.design_flood(T, cov)
                print(f"    协变量={cov:.1f}: {df:.1f} m³/s")
    
    print("\n" + "=" * 80)
    print("多模型比较与选择")
    print("=" * 80)
    
    adv_model = NonstationaryP3Advanced(flood_data, trend.reshape(-1, 1))
    model_results, best_model = adv_model.fit_multi_model()
    
    print(f"\n{'模型':<15} {'AIC':<12} {'BIC':<12} {'成功':<8}")
    print("-" * 50)
    
    for name, result in model_results.items():
        if result['success']:
            print(f"{name:<15} {result['aic']:<12.1f} {result['bic']:<12.1f} {'是':<8}")
        else:
            print(f"{name:<15} {'N/A':<12} {'N/A':<12} {'否':<8}")
    
    print(f"\n最优模型: {best_model}")
    print("=" * 80)


if __name__ == "__main__":
    main()
    print("\n\n")
    demonstrate_nonstationary()
