import numpy as np
from scipy import stats

class CauchyEstimator:
    def __init__(self):
        self.location = None
        self.scale = None
    
    def fit(self, data):
        self.location, self.scale = stats.cauchy.fit(data)
        return self.location, self.scale
    
    def pdf(self, x):
        if self.location is None or self.scale is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        return stats.cauchy.pdf(x, loc=self.location, scale=self.scale)
    
    def get_params(self):
        if self.location is None or self.scale is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        return {'x0': self.location, 'gamma': self.scale}
    
    def log_likelihood(self, data):
        if self.location is None or self.scale is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")
        return np.sum(np.log(self.pdf(data)))

def estimate_cauchy_from_array(data_array):
    estimator = CauchyEstimator()
    location, scale = estimator.fit(data_array)
    return location, scale, estimator

def generate_sample_data(true_location=0.0, true_scale=1.0, size=1000, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    return stats.cauchy.rvs(loc=true_location, scale=true_scale, size=size)

if __name__ == "__main__":
    print("柯西分布参数估计示例")
    print("=" * 50)
    
    true_x0 = 2.0
    true_gamma = 1.5
    
    data = generate_sample_data(true_location=true_x0, true_scale=true_gamma, size=1000, random_seed=42)
    
    estimated_x0, estimated_gamma, estimator = estimate_cauchy_from_array(data)
    
    print(f"真实参数:")
    print(f"  位置参数 x0 = {true_x0}")
    print(f"  尺度参数 γ = {true_gamma}")
    print()
    print(f"估计参数:")
    print(f"  位置参数 x0 = {estimated_x0:.6f}")
    print(f"  尺度参数 γ = {estimated_gamma:.6f}")
    print()
    print(f"估计误差:")
    print(f"  |Δx0| = {abs(estimated_x0 - true_x0):.6f}")
    print(f"  |Δγ| = {abs(estimated_gamma - true_gamma):.6f}")
    print()
    
    sample_points = [1.0, 2.0, 3.0]
    print(f"在几个点上的概率密度:")
    for x in sample_points:
        print(f"  f({x}) = {estimator.pdf(x):.6f}")
    
    print()
    print(f"对数似然值: {estimator.log_likelihood(data):.2f}")
    print()
    print("使用示例:")
    print("  from cauchy_estimator import CauchyEstimator, estimate_cauchy_from_array")
    print("  data = [your_data_here]")
    print("  x0, gamma, _ = estimate_cauchy_from_array(data)")
    print("  print(f'x0 = {x0}, gamma = {gamma}')")
