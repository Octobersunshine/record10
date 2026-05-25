import numpy as np
from scipy.optimize import least_squares, curve_fit


class StiffnessIdentifier:
    def __init__(self, method='ransac', **kwargs):
        self.displacements = []
        self.forces = []
        self.K = None
        self.K_std = None
        self.r_squared = None
        self.inlier_mask = None
        self.method = method
        self.kwargs = kwargs
    
    def add_data_point(self, displacement, force):
        self.displacements.append(displacement)
        self.forces.append(force)
    
    def clear_data(self):
        self.displacements = []
        self.forces = []
        self.K = None
        self.K_std = None
        self.r_squared = None
        self.inlier_mask = None
    
    def _compute_stats(self, x, y, K, inlier_mask=None):
        if inlier_mask is None:
            inlier_mask = np.ones_like(x, dtype=bool)
        
        x_in = x[inlier_mask]
        y_in = y[inlier_mask]
        
        y_pred = K * x_in
        ss_res = np.sum((y_in - y_pred) ** 2)
        ss_tot = np.sum((y_in - np.mean(y_in)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        
        n = np.sum(inlier_mask)
        if n > 1:
            mse = ss_res / (n - 1)
            self.K_std = np.sqrt(mse / np.sum(x_in ** 2))
        else:
            self.K_std = np.nan
        
        return self.r_squared, self.K_std
    
    def _lstsq(self, x, y):
        X = x.reshape(-1, 1)
        K, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        return K[0]
    
    def _huber_loss(self, residual, delta=1.0):
        abs_res = np.abs(residual)
        return np.where(abs_res <= delta, 
                        0.5 * residual ** 2, 
                        delta * (abs_res - 0.5 * delta))
    
    def _huber(self, x, y):
        delta = self.kwargs.get('delta', 1.5)
        
        def huber_cost(K):
            res = y - K[0] * x
            return self._huber_loss(res, delta)
        
        K0 = np.array([np.mean(y / (x + 1e-10))])
        result = least_squares(huber_cost, K0, method='lm')
        K = result.x[0]
        
        y_pred = K * x
        residuals = np.abs(y - y_pred)
        inlier_mask = residuals <= delta * np.std(residuals)
        
        return K, inlier_mask
    
    def _ransac(self, x, y):
        n = len(x)
        min_samples = self.kwargs.get('min_samples', 2)
        residual_threshold = self.kwargs.get('residual_threshold', 5.0)
        max_trials = self.kwargs.get('max_trials', 1000)
        
        best_inlier_count = 0
        best_inlier_mask = np.ones(n, dtype=bool)
        
        for _ in range(max_trials):
            sample_indices = np.random.choice(n, min_samples, replace=False)
            x_sample = x[sample_indices] + 1e-10
            y_sample = y[sample_indices]
            
            K_sample = np.mean(y_sample / x_sample)
            
            residuals = np.abs(y - K_sample * x)
            inlier_mask = residuals <= residual_threshold
            inlier_count = np.sum(inlier_mask)
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask = inlier_mask.copy()
        
        if best_inlier_count >= min_samples:
            K = self._lstsq(x[best_inlier_mask], y[best_inlier_mask])
        else:
            K = np.mean(y / (x + 1e-10))
        
        return K, best_inlier_mask
    
    def _softl1(self, x, y):
        f_scale = self.kwargs.get('f_scale', 1.0)
        
        def residual_scalar(K):
            return y - K[0] * x
        
        K0 = np.array([np.mean(y / (x + 1e-10))])
        result = least_squares(residual_scalar, K0, loss='soft_l1', f_scale=f_scale)
        K = result.x[0]
        
        y_pred = K * x
        residuals = np.abs(y - y_pred)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        inlier_mask = residuals <= 2.5 * mad
        
        return K, inlier_mask
    
    def identify(self):
        if len(self.displacements) < 2:
            raise ValueError("至少需要2个数据点")
        
        x = np.array(self.displacements)
        y = np.array(self.forces)
        
        if self.method == 'lstsq':
            self.K = self._lstsq(x, y)
            self.inlier_mask = np.ones_like(x, dtype=bool)
        elif self.method == 'huber':
            self.K, self.inlier_mask = self._huber(x, y)
        elif self.method == 'ransac':
            self.K, self.inlier_mask = self._ransac(x, y)
        elif self.method == 'softl1':
            self.K, self.inlier_mask = self._softl1(x, y)
        else:
            raise ValueError(f"未知方法: {self.method}，可选方法: lstsq, huber, ransac, softl1")
        
        self._compute_stats(x, y, self.K, self.inlier_mask)
        
        return self.K, self.K_std, self.r_squared, self.inlier_mask
    
    def get_results(self):
        return {
            'K': self.K,
            'K_std': self.K_std,
            'r_squared': self.r_squared,
            'num_points': len(self.displacements),
            'num_inliers': np.sum(self.inlier_mask) if self.inlier_mask is not None else len(self.displacements),
            'method': self.method,
            'inlier_mask': self.inlier_mask
        }
    
    def predict_force(self, displacement):
        if self.K is None:
            raise ValueError("请先调用 identify() 辨识刚度")
        return self.K * displacement
    
    def predict_displacement(self, force):
        if self.K is None or self.K == 0:
            raise ValueError("请先调用 identify() 辨识刚度，且刚度不能为0")
        return force / self.K
    
    def get_inliers(self):
        if self.inlier_mask is None:
            return np.array(self.displacements), np.array(self.forces)
        x = np.array(self.displacements)
        y = np.array(self.forces)
        return x[self.inlier_mask], y[self.inlier_mask]
    
    def get_outliers(self):
        if self.inlier_mask is None:
            return np.array([]), np.array([])
        x = np.array(self.displacements)
        y = np.array(self.forces)
        return x[~self.inlier_mask], y[~self.inlier_mask]


def identify_stiffness_from_arrays(displacements, forces, method='ransac', **kwargs):
    identifier = StiffnessIdentifier(method=method, **kwargs)
    for d, f in zip(displacements, forces):
        identifier.add_data_point(d, f)
    return identifier.identify()


def compare_methods(displacements, forces, K_true=None):
    methods = ['lstsq', 'huber', 'ransac', 'softl1']
    method_names = ['标准最小二乘', 'Huber损失', 'RANSAC', 'Soft L1']
    results = {}
    
    for method, name in zip(methods, method_names):
        try:
            K, K_std, r2, mask = identify_stiffness_from_arrays(
                displacements, forces, method=method)
            results[name] = {
                'K': K, 'K_std': K_std, 'r2': r2, 
                'inliers': np.sum(mask), 'mask': mask
            }
        except Exception as e:
            print(f"{name} 失败: {e}")
    
    print('=' * 70)
    print(f"{'方法':<15} {'刚度(N/m)':<15} {'R²':<10} {'内点数':<10}", end='')
    if K_true is not None:
        print(f"{'误差(%)':<10}")
    else:
        print()
    print('=' * 70)
    
    for name, res in results.items():
        print(f"{name:<15} {res['K']:<15.2f} {res['r2']:<10.4f} {res['inliers']:<10d}", end='')
        if K_true is not None:
            error = abs(res['K'] - K_true) / K_true * 100
            print(f"{error:<10.2f}")
        else:
            print()
    print('=' * 70)
    
    return results


def demo():
    print("=" * 60)
    print("StiffnessIdentifier 鲁棒回归演示")
    print("=" * 60)
    
    np.random.seed(42)
    K_true = 520.0
    
    print(f"\n模拟真实刚度: {K_true} N/m")
    print("生成含离群值的力-位移数据...")
    
    n_points = 50
    x = np.linspace(0, 0.02, n_points)
    y_true = K_true * x
    noise = np.random.normal(0, 3, n_points)
    y = y_true + noise
    
    n_outliers = int(n_points * 0.15)
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    y[outlier_idx] += np.random.uniform(-25, 25, n_outliers)
    
    print(f"数据点数: {n_points}, 离群点数: {n_outliers}")
    
    print("\n" + "=" * 60)
    print("方法对比:")
    print("=" * 60)
    
    compare_methods(x, y, K_true)
    
    print("\n" + "=" * 60)
    print("RANSAC方法详细结果:")
    print("=" * 60)
    
    identifier = StiffnessIdentifier(method='ransac', residual_threshold=5.0)
    for xi, yi in zip(x, y):
        identifier.add_data_point(xi, yi)
    
    K, K_std, r2, mask = identifier.identify()
    results = identifier.get_results()
    
    print(f"刚度 K = {K:.2f} ± {K_std:.2f} N/m")
    print(f"决定系数 R² = {r2:.6f}")
    print(f"有效点数: {results['num_inliers']}/{results['num_points']}")
    print(f"相对误差: {abs(K - K_true)/K_true*100:.4f}%")
    
    in_x, in_y = identifier.get_inliers()
    out_x, out_y = identifier.get_outliers()
    print(f"剔除的异常点数: {len(out_x)}")
    
    x_test = 0.015
    f_pred = identifier.predict_force(x_test)
    print(f"\n预测: 位移 {x_test*1000:.1f} mm -> 力 {f_pred:.2f} N")


class SoftTissueIdentifier:
    def __init__(self, model_type='hunt_crossley', **kwargs):
        self.displacements = []
        self.velocities = []
        self.forces = []
        self.model_type = model_type
        self.kwargs = kwargs
        self.params = None
        self.r_squared = None
        self.inlier_mask = None
    
    def add_data_point(self, displacement, velocity, force):
        self.displacements.append(displacement)
        self.velocities.append(velocity)
        self.forces.append(force)
    
    def clear_data(self):
        self.displacements = []
        self.velocities = []
        self.forces = []
        self.params = None
        self.r_squared = None
        self.inlier_mask = None
    
    @staticmethod
    def hunt_crossley_force(x, v, K, B, n, m):
        x_pos = np.maximum(x, 0)
        return K * np.power(x_pos, n) + B * np.power(x_pos, m) * v
    
    @staticmethod
    def linear_kelvin_voigt(x, v, K, B):
        return K * x + B * v
    
    def _fit_linear(self, x, v, F):
        X = np.column_stack([x, v])
        params, residuals, rank, s = np.linalg.lstsq(X, F, rcond=None)
        K, B = params
        return np.array([K, B, 1.0, 0.0])
    
    def _fit_hunt_crossley(self, x, v, F, fix_m=True, m_fixed=None):
        if fix_m:
            def model_wrapper(xv, K, B, n):
                x, v = xv
                m = n - 1 if m_fixed is None else m_fixed
                x_pos = np.maximum(x, 0)
                return K * np.power(x_pos, n) + B * np.power(x_pos, m) * v
            
            xv_data = np.vstack([x, v])
            try:
                popt, _ = curve_fit(model_wrapper, xv_data, F,
                                   p0=self.kwargs.get('p0', [1000.0, 10.0, 1.5]),
                                   bounds=self.kwargs.get('bounds', ([100, 0, 0.5], [1e6, 1000, 3.0])))
                K, B, n = popt
                m = n - 1 if m_fixed is None else m_fixed
                return np.array([K, B, n, m])
            except:
                return np.array([1000.0, 10.0, 1.5, 0.5])
        else:
            def model_wrapper(xv, K, B, n, m):
                x, v = xv
                x_pos = np.maximum(x, 0)
                return K * np.power(x_pos, n) + B * np.power(x_pos, m) * v
            
            xv_data = np.vstack([x, v])
            try:
                popt, _ = curve_fit(model_wrapper, xv_data, F,
                                   p0=self.kwargs.get('p0', [1000.0, 10.0, 1.5, 0.5]),
                                   bounds=self.kwargs.get('bounds', ([100, 0, 0.5, 0], [1e6, 1000, 3.0, 2.0])))
                return popt
            except:
                return np.array([1000.0, 10.0, 1.5, 0.5])
    
    def _fit_ransac_hc(self, x, v, F):
        n_points = len(x)
        min_samples = self.kwargs.get('min_samples', 10)
        residual_threshold = self.kwargs.get('residual_threshold', 1.0)
        max_trials = self.kwargs.get('max_trials', 200)
        fix_m = self.kwargs.get('fix_m', True)
        m_fixed = self.kwargs.get('m_fixed', None)
        
        best_inlier_count = 0
        best_inlier_mask = np.ones(n_points, dtype=bool)
        best_params = None
        
        for _ in range(max_trials):
            sample_idx = np.random.choice(n_points, min_samples, replace=False)
            x_sample = x[sample_idx]
            v_sample = v[sample_idx]
            F_sample = F[sample_idx]
            
            try:
                params = self._fit_hunt_crossley(x_sample, v_sample, F_sample, fix_m, m_fixed)
                K, B, n, m = params
                
                F_pred = self.hunt_crossley_force(x, v, K, B, n, m)
                residuals = np.abs(F - F_pred)
                inlier_mask = residuals <= residual_threshold
                inlier_count = np.sum(inlier_mask)
                
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_inlier_mask = inlier_mask.copy()
                    best_params = params
            except:
                continue
        
        if best_params is None:
            return self._fit_hunt_crossley(x, v, F, fix_m, m_fixed), np.ones(n_points, dtype=bool)
        
        x_in = x[best_inlier_mask]
        v_in = v[best_inlier_mask]
        F_in = F[best_inlier_mask]
        
        try:
            final_params = self._fit_hunt_crossley(x_in, v_in, F_in, fix_m, m_fixed)
        except:
            final_params = best_params
        
        return final_params, best_inlier_mask
    
    def identify(self):
        if len(self.displacements) < 5:
            raise ValueError("至少需要5个数据点")
        
        x = np.array(self.displacements)
        v = np.array(self.velocities)
        F = np.array(self.forces)
        
        if self.model_type == 'linear':
            self.params = self._fit_linear(x, v, F)
            F_pred = self.linear_kelvin_voigt(x, v, self.params[0], self.params[1])
            self.inlier_mask = np.ones_like(x, dtype=bool)
        elif self.model_type == 'hunt_crossley':
            fix_m = self.kwargs.get('fix_m', True)
            m_fixed = self.kwargs.get('m_fixed', None)
            self.params = self._fit_hunt_crossley(x, v, F, fix_m, m_fixed)
            F_pred = self.hunt_crossley_force(x, v, *self.params)
            self.inlier_mask = np.ones_like(x, dtype=bool)
        elif self.model_type == 'ransac':
            self.params, self.inlier_mask = self._fit_ransac_hc(x, v, F)
            F_pred = self.hunt_crossley_force(x, v, *self.params)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")
        
        x_in = x[self.inlier_mask]
        F_in = F[self.inlier_mask]
        F_pred_in = F_pred[self.inlier_mask]
        
        ss_res = np.sum((F_in - F_pred_in) ** 2)
        ss_tot = np.sum((F_in - np.mean(F_in)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        
        return self.params, self.r_squared, self.inlier_mask
    
    def predict_force(self, displacement, velocity):
        if self.params is None:
            raise ValueError("请先调用 identify() 辨识参数")
        
        if self.model_type == 'linear':
            return self.linear_kelvin_voigt(displacement, velocity, self.params[0], self.params[1])
        else:
            return self.hunt_crossley_force(displacement, velocity, *self.params)
    
    def get_results(self):
        if self.params is None:
            return {}
        
        results = {
            'model_type': self.model_type,
            'K': self.params[0],
            'B': self.params[1],
            'r_squared': self.r_squared,
            'num_points': len(self.displacements),
            'num_inliers': np.sum(self.inlier_mask) if self.inlier_mask is not None else len(self.displacements)
        }
        
        if self.model_type != 'linear':
            results['n'] = self.params[2]
            results['m'] = self.params[3]
        
        return results
    
    def get_inliers(self):
        if self.inlier_mask is None:
            return (np.array(self.displacements), 
                    np.array(self.velocities), 
                    np.array(self.forces))
        x = np.array(self.displacements)
        v = np.array(self.velocities)
        f = np.array(self.forces)
        return x[self.inlier_mask], v[self.inlier_mask], f[self.inlier_mask]
    
    def get_outliers(self):
        if self.inlier_mask is None:
            return np.array([]), np.array([]), np.array([])
        x = np.array(self.displacements)
        v = np.array(self.velocities)
        f = np.array(self.forces)
        return x[~self.inlier_mask], v[~self.inlier_mask], f[~self.inlier_mask]


def compare_soft_tissue_models(x, v, F, true_params=None):
    models = ['linear', 'hunt_crossley', 'ransac']
    model_names = ['线性Kelvin-Voigt', 'Hunt-Crossley', 'RANSAC鲁棒HC']
    results = {}
    
    for model, name in zip(models, model_names):
        try:
            identifier = SoftTissueIdentifier(model_type=model)
            for xi, vi, fi in zip(x, v, F):
                identifier.add_data_point(xi, vi, fi)
            params, r2, mask = identifier.identify()
            results[name] = {
                'params': params,
                'r2': r2,
                'inliers': np.sum(mask),
                'mask': mask
            }
        except Exception as e:
            print(f"{name} 失败: {e}")
    
    print('=' * 80)
    header = f"{'模型':<18} {'K':<12} {'B':<10}"
    if true_params:
        header += f" {'n':<8} {'m':<8} {'R²':<10} {'误差%':<10}"
    else:
        header += f" {'n':<8} {'m':<8} {'R²':<10} {'内点数':<10}"
    print(header)
    print('=' * 80)
    
    for name, res in results.items():
        p = res['params']
        line = f"{name:<18} {p[0]:<12.1f} {p[1]:<10.2f}"
        if len(p) >= 4:
            line += f" {p[2]:<8.3f} {p[3]:<8.3f}"
        else:
            line += f" {'-':<8} {'-':<8}"
        line += f" {res['r2']:<10.4f}"
        
        if true_params:
            if 'K' in true_params:
                error = abs(p[0] - true_params['K']) / true_params['K'] * 100
                line += f" {error:<10.2f}"
        else:
            line += f" {res['inliers']:<10d}"
        print(line)
    
    print('=' * 80)
    return results


def demo_soft_tissue():
    print("=" * 70)
    print("SoftTissueIdentifier 软组织辨识演示")
    print("=" * 70)
    
    np.random.seed(42)
    
    from soft_tissue_identification import generate_soft_tissue_data
    
    true_params = {'K': 8000.0, 'B': 80.0, 'n': 1.6, 'm': 0.6}
    x, v, F, _ = generate_soft_tissue_data(
        K=true_params['K'], B=true_params['B'],
        n=true_params['n'], m=true_params['m'],
        noise_level=0.8, num_points=150, outlier_ratio=0.08
    )
    
    print(f"\n真实参数: K={true_params['K']}, B={true_params['B']}, n={true_params['n']}, m={true_params['m']}")
    print(f"数据点数: {len(F)}, 位移范围: {np.min(x)*1000:.1f}-{np.max(x)*1000:.1f} mm")
    
    print("\n" + "=" * 70)
    print("模型对比:")
    print("=" * 70)
    
    compare_soft_tissue_models(x, v, F, true_params)
    
    print("\n" + "=" * 70)
    print("RANSAC鲁棒Hunt-Crossley详细结果:")
    print("=" * 70)
    
    identifier = SoftTissueIdentifier(model_type='ransac', residual_threshold=1.5)
    for xi, vi, fi in zip(x, v, F):
        identifier.add_data_point(xi, vi, fi)
    
    params, r2, mask = identifier.identify()
    results = identifier.get_results()
    
    print(f"模型类型: {results['model_type']}")
    print(f"刚度系数 K = {results['K']:.2f} N/m^{results['n']:.3f}")
    print(f"阻尼系数 B = {results['B']:.2f} N·s/m^{results['m']+1:.3f}")
    print(f"非线性指数 n = {results['n']:.4f}")
    print(f"阻尼指数 m = {results['m']:.4f}")
    print(f"决定系数 R² = {r2:.6f}")
    print(f"有效数据点: {results['num_inliers']}/{results['num_points']}")
    
    error_K = abs(results['K'] - true_params['K']) / true_params['K'] * 100
    error_n = abs(results['n'] - true_params['n']) / true_params['n'] * 100
    print(f"K相对误差: {error_K:.2f}%")
    print(f"n相对误差: {error_n:.2f}%")
    
    x_test, v_test = 0.008, 0.01
    f_pred = identifier.predict_force(x_test, v_test)
    print(f"\n预测: 位移 {x_test*1000:.1f} mm, 速度 {v_test*1000:.1f} mm/s -> 力 {f_pred:.2f} N")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'soft':
        demo_soft_tissue()
    else:
        demo()
