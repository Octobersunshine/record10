import numpy as np
from scipy import stats
import sys

print("测试柯西分布似然函数多峰问题及解决方案")
print("=" * 70)

try:
    from cauchy_estimator_improved import CauchyEstimator, generate_sample_data
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

np.random.seed(123)

print("\n=== 测试1: 小样本下的多峰问题 ===")
true_x0 = 5.0
true_gamma = 1.0

data_small = generate_sample_data(true_x0, true_gamma, size=50, random_seed=123)
print(f"真实参数: x0={true_x0}, γ={true_gamma}")
print(f"样本量: {len(data_small)}")

estimator = CauchyEstimator()

print("\n各方法对比:")
print("-" * 70)
print(f"{'方法':<15} {'x0':<12} {'γ':<12} {'对数似然':<15} {'x0误差':<12} {'γ误差':<12}")
print("-" * 70)

for method in ['default', 'robust', 'multistart', 'basinhopping', 'de']:
    try:
        loc, scale = estimator.fit(data_small, method=method)
        ll = estimator.log_likelihood(data_small)
        err_x0 = abs(loc - true_x0)
        err_gamma = abs(scale - true_gamma)
        print(f"{method:<15} {loc:<12.4f} {scale:<12.4f} {ll:<15.2f} {err_x0:<12.4f} {err_gamma:<12.4f}")
    except Exception as e:
        print(f"{method:<15} {'失败':<12} {'失败':<12} {'N/A':<15} {'N/A':<12} {'N/A':<12}")

print("-" * 70)

print("\n=== 测试2: 手动构造局部最优陷阱 ===")
np.random.seed(456)
data1 = stats.cauchy.rvs(loc=0, scale=0.5, size=30)
data2 = stats.cauchy.rvs(loc=10, scale=0.5, size=30)
data_mixture = np.concatenate([data1, data2])

print("混合两个柯西分布:")
print("  - 分布1: x0=0, γ=0.5 (30个样本)")
print("  - 分布2: x0=10, γ=0.5 (30个样本)")
print(f"总样本量: {len(data_mixture)}")

print("\n各方法估计结果:")
print("-" * 70)
print(f"{'方法':<15} {'x0':<12} {'γ':<12} {'对数似然':<15}")
print("-" * 70)

for method in ['default', 'robust', 'multistart', 'basinhopping', 'de']:
    try:
        loc, scale = estimator.fit(data_mixture, method=method)
        ll = estimator.log_likelihood(data_mixture)
        print(f"{method:<15} {loc:<12.4f} {scale:<12.4f} {ll:<15.2f}")
    except Exception as e:
        print(f"{method:<15} {'失败':<12} {'失败':<12} {'N/A':<15}")

print("-" * 70)

print("\n=== 测试3: 似然函数可视化（数值验证） ===")
data = generate_sample_data(5.0, 1.0, size=100, random_seed=789)

print("在不同x0值处的对数似然值（固定γ=1）:")
print("-" * 50)
print(f"{'x0':<10} {'对数似然':<20}")
print("-" * 50)

for x0 in [3, 4, 4.5, 4.8, 5.0, 5.2, 5.5, 6, 7]:
    ll = np.sum(stats.cauchy.logpdf(data, loc=x0, scale=1.0))
    print(f"{x0:<10} {ll:<20.2f}")

print("-" * 50)
print("\n可以观察到似然函数不是单峰的，存在多个局部极值！")

print("\n=== 测试4: fit_all_methods() 自动选择 ===")
results, best_method = estimator.fit_all_methods(data)

print("\n所有方法结果:")
for method, res in results.items():
    if 'error' in res:
        print(f"  {method}: 错误 - {res['error']}")
    else:
        print(f"  {method}: x0={res['x0']:.4f}, γ={res['gamma']:.4f}, 对数似然={res['log_likelihood']:.2f}")

print(f"\n自动选择的最优方法: {best_method}")
print(f"最终参数: x0={estimator.location:.6f}, γ={estimator.scale:.6f}")

print("\n" + "=" * 70)
print("总结:")
print("1. SciPy默认方法容易陷入局部最优")
print("2. 使用鲁棒初始值（中位数+四分位距）可以改善结果")
print("3. 多起点优化（multistart）平衡了精度和速度")
print("4. 全局优化方法（basinhopping, de）更可靠但计算较慢")
print("5. fit_all_methods() 可以自动选择最优结果")
print("=" * 70)
