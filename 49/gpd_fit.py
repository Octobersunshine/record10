import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class GPDFit:
    def __init__(self, data):
        self.data = np.array(data)
        self.threshold = None
        self.exceedances = None
        self.scale = None
        self.shape = None
        self.loc = None
        self.n_exceed = None

    def mrl_plot(self, thresholds=None, plot=False):
        if thresholds is None:
            thresholds = np.quantile(self.data, np.linspace(0.5, 0.95, 20))
        
        mrl_values = []
        n_exceedances = []
        
        for u in thresholds:
            exceed = self.data[self.data > u] - u
            if len(exceed) > 0:
                mrl_values.append(np.mean(exceed))
                n_exceedances.append(len(exceed))
            else:
                mrl_values.append(np.nan)
                n_exceedances.append(0)
        
        if plot:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(thresholds, mrl_values, 'bo-')
                plt.xlabel('Threshold')
                plt.ylabel('Mean Residual Life')
                plt.title('Mean Residual Life Plot')
                plt.grid(True, alpha=0.3)
                plt.show()
            except ImportError:
                print("matplotlib not available for plotting")
        
        return thresholds, mrl_values, n_exceedances

    def fit(self, threshold=None, quantile=0.95, method='MLE'):
        if threshold is None:
            threshold = np.quantile(self.data, quantile)
        
        self.threshold = threshold
        exceedances = self.data[self.data > threshold] - threshold
        self.exceedances = exceedances
        self.n_exceed = len(exceedances)
        
        if len(exceedances) < 5:
            raise ValueError(f"Too few exceedances: {len(exceedances)}. Try a lower threshold.")
        
        if method == 'MLE':
            self.shape, self.loc, self.scale = self._stable_mle_fit(exceedances)
        elif method == 'prob_weighted':
            self.shape, self.loc, self.scale = self._prob_weighted_moments(exceedances)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'MLE' or 'prob_weighted'.")
        
        return self

    def _stable_mle_fit(self, x):
        n = len(x)
        x_sorted = np.sort(x)
        
        shape_pwm, _, scale_pwm = self._prob_weighted_moments(x)
        
        def neg_loglik(params):
            shape, scale = params
            
            if scale <= 1e-10:
                return 1e10
            
            if shape <= -0.499:
                return 1e10
            
            if shape > 5.0:
                return 1e10 + (shape - 5.0) ** 2
            
            z = x / scale
            
            if abs(shape) < 1e-8:
                loglik = -n * np.log(scale) - np.sum(z)
            else:
                term = 1 + shape * z
                min_term = np.min(term)
                if min_term <= 1e-10:
                    return 1e10 + (1e-10 - min_term) * 1e6
                
                log_term = np.log(term)
                loglik = -n * np.log(scale) - (1/shape + 1) * np.sum(log_term)
            
            if not np.isfinite(loglik):
                return 1e10
            
            return -loglik

        def neg_loglik_reg(params, lambda_reg=0.001):
            shape, scale = params
            nll = neg_loglik(params)
            reg = lambda_reg * (shape ** 2)
            return nll + reg

        initial_guesses = [
            [shape_pwm, scale_pwm],
            [0.0, np.mean(x)],
            [0.01, np.mean(x)],
            [-0.01, np.mean(x)],
            [0.05, np.mean(x) * 0.9],
            [-0.05, np.mean(x) * 1.1],
            [0.1, np.mean(x)],
        ]

        best_nll = np.inf
        best_params = None

        bounds = [(-0.49, 5.0), (1e-5, np.max(x) * 10)]

        for x0 in initial_guesses:
            try:
                result = minimize(
                    neg_loglik_reg,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if result.success and result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x
            except Exception:
                continue

        if best_params is None:
            try:
                result = minimize(
                    neg_loglik_reg,
                    [shape_pwm, scale_pwm],
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'fatol': 1e-12}
                )
                if result.success:
                    best_params = result.x
                    best_nll = result.fun
            except Exception:
                pass

        if best_params is None:
            try:
                result = minimize(
                    neg_loglik_reg,
                    [0.0, np.mean(x)],
                    method='Powell',
                    options={'maxiter': 2000}
                )
                if result.success:
                    best_params = result.x
                    best_nll = result.fun
            except Exception:
                pass

        if best_params is None:
            warnings.warn("MLE optimization failed, falling back to probability weighted moments")
            return shape_pwm, 0.0, scale_pwm

        shape, scale = best_params

        if abs(shape) < 1e-4:
            shape_mle_at_zero = -n * np.log(np.mean(x)) - n
            shape_current_nll = best_nll
            
            if abs(shape_current_nll - (-shape_mle_at_zero)) < 0.1:
                shape = 0.0
                scale = np.mean(x)

        if shape > 0.5 and abs(shape - shape_pwm) > 0.3:
            warnings.warn(f"Shape estimate {shape:.4f} seems high, falling back to PWM")
            shape, scale = shape_pwm, scale_pwm

        return shape, 0.0, scale

    def profile_likelihood_shape(self, shape_values=None):
        if self.exceedances is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        x = self.exceedances
        n = len(x)
        
        if shape_values is None:
            shape_values = np.linspace(-0.4, 0.8, 100)
        
        profile_nll = []
        
        for shape in shape_values:
            def scale_nll(scale):
                if scale <= 1e-10:
                    return 1e10
                
                z = x / scale
                
                if abs(shape) < 1e-6:
                    loglik = -n * np.log(scale) - np.sum(z)
                else:
                    term = 1 + shape * z
                    if np.any(term <= 1e-10):
                        return 1e10
                    loglik = -n * np.log(scale) - (1/shape + 1) * np.sum(np.log(term))
                
                return -loglik
            
            result = minimize(
                scale_nll,
                [np.mean(x)],
                method='L-BFGS-B',
                bounds=[(1e-6, None)],
                options={'maxiter': 500}
            )
            
            if result.success:
                profile_nll.append(result.fun)
            else:
                profile_nll.append(np.nan)
        
        return shape_values, np.array(profile_nll)

    def standard_errors(self):
        if self.exceedances is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        x = self.exceedances
        n = len(x)
        shape, scale = self.shape, self.scale
        
        if abs(shape) < 1e-6:
            se_shape = np.sqrt(2 / n)
            se_scale = scale / np.sqrt(n)
        else:
            term1 = (1 + shape) ** 2 / n
            term2 = (1 + 2 * shape) / n
            fisher_shape = term2 / (shape ** 2)
            fisher_scale = term1 / (scale ** 2)
            fisher_cross = (1 + shape) / (shape * scale * n)
            
            det = fisher_shape * fisher_scale - fisher_cross ** 2
            se_shape = np.sqrt(fisher_scale / det)
            se_scale = np.sqrt(fisher_shape / det)
        
        return {'shape_se': se_shape, 'scale_se': se_scale}

    def _prob_weighted_moments(self, x):
        n = len(x)
        x_sorted = np.sort(x)
        
        b0 = np.mean(x_sorted)
        b1 = np.sum(x_sorted * (np.arange(n) / (n - 1))) / n
        
        if b1 == 0:
            shape = 0
        else:
            shape = 2 - b0 / (b0 - 2 * b1)
        
        scale = (2 * b0 * b1) / (b0 - 2 * b1)
        
        return shape, 0, scale

    def return_level(self, return_period):
        if self.scale is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        n_total = len(self.data)
        zeta = self.n_exceed / n_total
        
        if abs(self.shape) < 1e-10:
            rl = self.threshold + self.scale * np.log(return_period * zeta)
        else:
            rl = self.threshold + (self.scale / self.shape) * (
                (return_period * zeta) ** self.shape - 1
            )
        
        return rl

    def return_levels(self, return_periods):
        return np.array([self.return_level(rp) for rp in return_periods])

    def summary(self):
        if self.scale is None:
            return "Model not fitted yet."
        
        summary_dict = {
            'threshold': self.threshold,
            'n_exceedances': self.n_exceed,
            'shape_parameter': self.shape,
            'scale_parameter': self.scale,
            'location_parameter': self.loc
        }
        
        return summary_dict

    def plot_return_levels(self, max_return_period=1000):
        try:
            import matplotlib.pyplot as plt
            
            return_periods = np.logspace(0, np.log10(max_return_period), 100)
            rl_values = self.return_levels(return_periods)
            
            empirical_rp = []
            empirical_rl = np.sort(self.data[self.data > self.threshold])
            for i in range(1, len(empirical_rl) + 1):
                emp_prob = i / (len(self.data) + 1)
                rp = 1 / ((1 - np.quantile(self.data, 0.95)) * (1 - emp_prob))
                empirical_rp.append(rp)
            
            plt.figure(figsize=(12, 6))
            plt.semilogx(return_periods, rl_values, 'b-', linewidth=2, label='GPD Fit')
            plt.semilogx(empirical_rp, empirical_rl[::-1], 'ro', markersize=4, label='Empirical Data')
            plt.xlabel('Return Period')
            plt.ylabel('Return Level')
            plt.title('Return Level Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except ImportError:
            print("matplotlib not available for plotting")


class NonstationaryGPD:
    def __init__(self, data, covariates):
        self.data = np.array(data)
        self.covariates = np.array(covariates)
        self.n = len(data)
        
        if self.covariates.ndim == 1:
            self.covariates = self.covariates.reshape(-1, 1)
        
        self.n_cov = self.covariates.shape[1]
        self.threshold = None
        self.exceedances = None
        self.exceed_covariates = None
        self.n_exceed = None
        
        self.loc_coefs = None
        self.scale_coefs = None
        self.shape = None
        
    def _design_matrix_loc(self, X):
        return np.column_stack([np.ones(len(X)), X])
    
    def _design_matrix_scale(self, X):
        return np.column_stack([np.ones(len(X)), X])
    
    def _get_loc(self, X, coefs):
        X_design = self._design_matrix_loc(X)
        return X_design @ coefs
    
    def _get_scale(self, X, coefs):
        X_design = self._design_matrix_scale(X)
        return np.exp(X_design @ coefs)
    
    def fit(self, threshold=None, quantile=0.95, location_cov=True, scale_cov=True):
        if threshold is None:
            if np.isscalar(quantile):
                threshold = np.quantile(self.data, quantile)
            else:
                threshold = quantile
        
        self.threshold = threshold
        
        if np.isscalar(threshold):
            exceed_mask = self.data > threshold
            self.exceedances = self.data[exceed_mask] - threshold
            self.exceed_covariates = self.covariates[exceed_mask]
        else:
            exceed_mask = self.data > threshold
            self.exceedances = self.data[exceed_mask] - threshold[exceed_mask]
            self.exceed_covariates = self.covariates[exceed_mask]
        
        self.n_exceed = len(self.exceedances)
        
        if self.n_exceed < 10:
            raise ValueError(f"Too few exceedances: {self.n_exceed}. Try a lower threshold.")
        
        n_loc_params = 1 + (self.n_cov if location_cov else 0)
        n_scale_params = 1 + (self.n_cov if scale_cov else 0)
        n_shape_params = 1
        
        total_params = n_loc_params + n_scale_params + n_shape_params
        
        def neg_loglik(params):
            idx = 0
            loc_coefs = params[idx:idx + n_loc_params]
            idx += n_loc_params
            scale_coefs = params[idx:idx + n_scale_params]
            idx += n_scale_params
            shape = params[idx]
            
            if shape <= -0.499:
                return 1e10
            if shape > 5.0:
                return 1e10 + (shape - 5.0) ** 2
            
            loc = self._get_loc(self.exceed_covariates, loc_coefs)
            scale = self._get_scale(self.exceed_covariates, scale_coefs)
            
            if np.min(scale) <= 1e-10:
                return 1e10
            
            z = (self.exceedances - loc) / scale
            
            if abs(shape) < 1e-8:
                loglik = -np.sum(np.log(scale)) - np.sum(z)
            else:
                term = 1 + shape * z
                min_term = np.min(term)
                if min_term <= 1e-10:
                    return 1e10 + (1e-10 - min_term) * 1e6
                
                log_term = np.log(term)
                loglik = -np.sum(np.log(scale)) - (1/shape + 1) * np.sum(log_term)
            
            if not np.isfinite(loglik):
                return 1e10
            
            return -loglik
        
        def neg_loglik_reg(params, lambda_reg=0.001):
            nll = neg_loglik(params)
            reg = lambda_reg * np.sum(params ** 2)
            return nll + reg
        
        x_mean = np.mean(self.exceedances)
        x_std = np.std(self.exceedances)
        
        initial_guesses = []
        
        base_loc = [x_mean * 0.1] + [0.0] * (n_loc_params - 1)
        base_scale = [np.log(x_std)] + [0.0] * (n_scale_params - 1)
        base_shape = [0.0]
        
        initial_guesses.append(base_loc + base_scale + base_shape)
        initial_guesses.append(base_loc + base_scale + [0.1])
        initial_guesses.append(base_loc + base_scale + [-0.05])
        
        best_nll = np.inf
        best_params = None
        
        bounds = []
        bounds += [(None, None)] * n_loc_params
        bounds += [(None, None)] * n_scale_params
        bounds += [(-0.49, 5.0)]
        
        for x0 in initial_guesses:
            try:
                result = minimize(
                    neg_loglik_reg,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9}
                )
                
                if result.success and result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x
            except Exception:
                continue
        
        if best_params is None:
            try:
                result = minimize(
                    neg_loglik_reg,
                    initial_guesses[0],
                    method='Nelder-Mead',
                    options={'maxiter': 4000}
                )
                if result.success:
                    best_params = result.x
            except Exception:
                pass
        
        if best_params is None:
            raise RuntimeError("Nonstationary GPD fitting failed")
        
        idx = 0
        self.loc_coefs = best_params[idx:idx + n_loc_params]
        idx += n_loc_params
        self.scale_coefs = best_params[idx:idx + n_scale_params]
        idx += n_scale_params
        self.shape = best_params[idx]
        
        self.location_cov = location_cov
        self.scale_cov = scale_cov
        
        return self
    
    def get_params_at(self, covariates):
        covariates = np.array(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        
        loc = self._get_loc(covariates, self.loc_coefs)
        scale = self._get_scale(covariates, self.scale_coefs)
        
        return loc, scale, self.shape
    
    def return_level(self, return_period, covariates, threshold=None):
        if threshold is None:
            threshold = self.threshold
        
        loc, scale, shape = self.get_params_at(covariates)
        n_total = self.n
        zeta = self.n_exceed / n_total
        
        if abs(shape) < 1e-10:
            rl = threshold + loc + scale * np.log(return_period * zeta)
        else:
            rl = threshold + loc + (scale / shape) * (
                (return_period * zeta) ** shape - 1
            )
        
        return rl
    
    def summary(self):
        if self.loc_coefs is None:
            return "Model not fitted yet."
        
        summary_dict = {
            'threshold': self.threshold,
            'n_exceedances': self.n_exceed,
            'shape_parameter': self.shape,
            'location_coefficients': self.loc_coefs,
            'scale_coefficients': self.scale_coefs,
            'location_covariates_used': self.location_cov,
            'scale_covariates_used': self.scale_cov
        }
        
        return summary_dict
    
    def plot_trend(self, covariate_index=0, covariate_name='Covariate'):
        try:
            import matplotlib.pyplot as plt
            
            cov_values = np.linspace(
                np.min(self.covariates[:, covariate_index]),
                np.max(self.covariates[:, covariate_index]),
                100
            )
            
            pred_cov = np.zeros((100, self.n_cov))
            pred_cov[:, covariate_index] = cov_values
            
            loc, scale, shape = self.get_params_at(pred_cov)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(cov_values, loc, 'b-', linewidth=2)
            axes[0].set_xlabel(covariate_name)
            axes[0].set_ylabel('Location Parameter')
            axes[0].set_title('Location Parameter Trend')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(cov_values, scale, 'r-', linewidth=2)
            axes[1].set_xlabel(covariate_name)
            axes[1].set_ylabel('Scale Parameter')
            axes[1].set_title('Scale Parameter Trend')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")


def pot_gpd_fit(data, quantile=0.95, return_periods=[10, 20, 50, 100]):
    model = GPDFit(data)
    model.fit(quantile=quantile)
    
    results = {
        'summary': model.summary(),
        'return_levels': {}
    }
    
    for rp in return_periods:
        results['return_levels'][f'{rp}-year'] = model.return_level(rp)
    
    return model, results


def nonstationary_pot_fit(data, covariates, quantile=0.95, location_cov=True, scale_cov=True):
    model = NonstationaryGPD(data, covariates)
    model.fit(quantile=quantile, location_cov=location_cov, scale_cov=scale_cov)
    return model


if __name__ == '__main__':
    np.random.seed(42)
    
    print("=" * 70)
    print("TEST 1: Gumbel distribution (ξ close to 0)")
    print("=" * 70)
    data1 = np.random.gumbel(loc=50, scale=10, size=1000)
    model1, results1 = pot_gpd_fit(data1, quantile=0.95, return_periods=[10, 20, 50, 100])
    print(f"Threshold: {results1['summary']['threshold']:.4f}")
    print(f"Number of exceedances: {results1['summary']['n_exceedances']}")
    print(f"Shape parameter (ξ): {results1['summary']['shape_parameter']:.6f}")
    print(f"Scale parameter (σ): {results1['summary']['scale_parameter']:.4f}")
    se1 = model1.standard_errors()
    print(f"Standard errors - ξ: {se1['shape_se']:.4f}, σ: {se1['scale_se']:.4f}")
    print("Return Levels:")
    for rp, rl in results1['return_levels'].items():
        print(f"  {rp}: {rl:.4f}")
    print()
    
    print("=" * 70)
    print("TEST 2: Exponential distribution (ξ = 0 exactly)")
    print("=" * 70)
    data2 = np.random.exponential(scale=10, size=1000)
    model2, results2 = pot_gpd_fit(data2, quantile=0.90, return_periods=[10, 20, 50, 100])
    print(f"Threshold: {results2['summary']['threshold']:.4f}")
    print(f"Number of exceedances: {results2['summary']['n_exceedances']}")
    print(f"Shape parameter (ξ): {results2['summary']['shape_parameter']:.6f}")
    print(f"Scale parameter (σ): {results2['summary']['scale_parameter']:.4f}")
    se2 = model2.standard_errors()
    print(f"Standard errors - ξ: {se2['shape_se']:.4f}, σ: {se2['scale_se']:.4f}")
    print("Return Levels:")
    for rp, rl in results2['return_levels'].items():
        print(f"  {rp}: {rl:.4f}")
    print()
    
    print("=" * 70)
    print("TEST 3: Pareto distribution (ξ > 0)")
    print("=" * 70)
    data3 = stats.genpareto.rvs(c=0.2, loc=0, scale=20, size=1000)
    model3, results3 = pot_gpd_fit(data3, quantile=0.90, return_periods=[10, 20, 50, 100])
    print(f"Threshold: {results3['summary']['threshold']:.4f}")
    print(f"Number of exceedances: {results3['summary']['n_exceedances']}")
    print(f"Shape parameter (ξ): {results3['summary']['shape_parameter']:.6f}")
    print(f"Scale parameter (σ): {results3['summary']['scale_parameter']:.4f}")
    se3 = model3.standard_errors()
    print(f"Standard errors - ξ: {se3['shape_se']:.4f}, σ: {se3['scale_se']:.4f}")
    print("Return Levels:")
    for rp, rl in results3['return_levels'].items():
        print(f"  {rp}: {rl:.4f}")
    print()
    
    print("=" * 70)
    print("TEST 4: Short-tailed distribution (ξ < 0)")
    print("=" * 70)
    data4 = stats.genpareto.rvs(c=-0.1, loc=0, scale=20, size=1000)
    model4, results4 = pot_gpd_fit(data4, quantile=0.90, return_periods=[10, 20, 50, 100])
    print(f"Threshold: {results4['summary']['threshold']:.4f}")
    print(f"Number of exceedances: {results4['summary']['n_exceedances']}")
    print(f"Shape parameter (ξ): {results4['summary']['shape_parameter']:.6f}")
    print(f"Scale parameter (σ): {results4['summary']['scale_parameter']:.4f}")
    se4 = model4.standard_errors()
    print(f"Standard errors - ξ: {se4['shape_se']:.4f}, σ: {se4['scale_se']:.4f}")
    print("Return Levels:")
    for rp, rl in results4['return_levels'].items():
        print(f"  {rp}: {rl:.4f}")
    print()
    
    print("=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    
    print("=" * 70)
    print("NONSTATIONARY GPD TEST: Time as Covariate")
    print("=" * 70)
    
    n_years = 100
    time = np.arange(n_years)
    
    trend_strength = 0.15
    base_scale = 8.0
    true_shape = 0.1
    
    np.random.seed(42)
    data_trend = []
    for t in time:
        loc_t = t * trend_strength
        scale_t = base_scale * (1 + 0.002 * t)
        exceedance = stats.genpareto.rvs(c=true_shape, loc=loc_t, scale=scale_t, size=1)[0]
        data_trend.append(50 + exceedance)
    
    data_trend = np.array(data_trend)
    
    print(f"Generated data with linear trend in location: {trend_strength} per year")
    print(f"True shape parameter: {true_shape}")
    print()
    
    model_ns = nonstationary_pot_fit(
        data_trend, 
        time, 
        quantile=0.85,
        location_cov=True,
        scale_cov=True
    )
    
    summary_ns = model_ns.summary()
    print(f"Threshold: {summary_ns['threshold']:.4f}")
    print(f"Number of exceedances: {summary_ns['n_exceedances']}")
    print(f"Shape parameter (ξ): {summary_ns['shape_parameter']:.6f}")
    print()
    print("Location coefficients:")
    for i, coef in enumerate(summary_ns['location_coefficients']):
        if i == 0:
            print(f"  Intercept: {coef:.4f}")
        else:
            print(f"  Covariate {i}: {coef:.4f}")
    print()
    print("Scale coefficients (log link):")
    for i, coef in enumerate(summary_ns['scale_coefficients']):
        if i == 0:
            print(f"  Intercept: {coef:.4f} (exp = {np.exp(coef):.4f})")
        else:
            print(f"  Covariate {i}: {coef:.4f}")
    print()
    
    test_times = [0, 50, 99]
    for t in test_times:
        rl_100 = model_ns.return_level(100, [t])
        print(f"100-year return level at time {t}: {rl_100:.4f}")
    
    print()
    print("=" * 70)
    print("NONSTATIONARY ANALYSIS COMPLETE")
    print("=" * 70)
