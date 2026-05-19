import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def estimate_cauchy_parameters(data):
    location, scale = stats.cauchy.fit(data)
    return location, scale

def cauchy_pdf(x, location, scale):
    return stats.cauchy.pdf(x, loc=location, scale=scale)

def generate_cauchy_data(location, scale, size=1000):
    return stats.cauchy.rvs(loc=location, scale=scale, size=size)

if __name__ == "__main__":
    true_location = 2.0
    true_scale = 1.5
    
    np.random.seed(42)
    data = generate_cauchy_data(true_location, true_scale, size=1000)
    
    estimated_location, estimated_scale = estimate_cauchy_parameters(data)
    
    print(f"真实参数: 位置参数 x0 = {true_location}, 尺度参数 γ = {true_scale}")
    print(f"估计参数: 位置参数 x0 = {estimated_location:.4f}, 尺度参数 γ = {estimated_scale:.4f}")
    print(f"估计误差: |Δx0| = {abs(estimated_location - true_location):.4f}, |Δγ| = {abs(estimated_scale - true_scale):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='样本数据直方图')
    
    x = np.linspace(min(data), max(data), 1000)
    plt.plot(x, cauchy_pdf(x, estimated_location, estimated_scale), 'r-', linewidth=2, 
             label=f'拟合的柯西分布 (x0={estimated_location:.2f}, γ={estimated_scale:.2f})')
    plt.plot(x, cauchy_pdf(x, true_location, true_scale), 'b--', linewidth=2, 
             label=f'真实的柯西分布 (x0={true_location}, γ={true_scale})')
    
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('柯西分布参数估计')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-10, 15)
    plt.show()
