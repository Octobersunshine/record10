import random
from distributions import (
    beta_fit_moments, beta_fit_mle, beta_stats, beta_mean, beta_variance,
    dirichlet_pdf, dirichlet_mean, dirichlet_variance, dirichlet_covariance, dirichlet_stats
)

print("=" * 70)
print("Beta分布参数估计对比 (矩估计 vs MLE)")
print("=" * 70)
random.seed(42)
true_params = [(2, 5), (5, 2), (10, 10), (0.5, 0.5)]

for at, bt in true_params:
    samples = []
    for _ in range(1000):
        x = random.betavariate(at, bt)
        if x <= 0:
            x = 1e-10
        elif x >= 1:
            x = 1 - 1e-10
        samples.append(x)
    
    mom = beta_fit_moments(samples)
    mle = beta_fit_mle(samples)
    
    err_mom_a = abs(mom['alpha'] - at)
    err_mom_b = abs(mom['beta'] - bt)
    err_mle_a = abs(mle['alpha'] - at)
    err_mle_b = abs(mle['beta'] - bt)
    
    print(f"真实: α={at:>5.2f}, β={bt:>5.2f}")
    print(f"  矩估计: α={mom['alpha']:>7.4f}, β={mom['beta']:>7.4f} | 误差: |Δα|={err_mom_a:.4f}, |Δβ|={err_mom_b:.4f}")
    print(f"  MLE:    α={mle['alpha']:>7.4f}, β={mle['beta']:>7.4f} | 误差: |Δα|={err_mle_a:.4f}, |Δβ|={err_mle_b:.4f}")

print()
print("=" * 70)
print("Beta统计量计算")
print("=" * 70)
for at, bt in [(1, 1), (2, 5), (10, 10)]:
    stats = beta_stats(at, bt)
    mean_direct = beta_mean(at, bt)
    var_direct = beta_variance(at, bt)
    print(f"Beta({at:>2d},{bt:>2d}): 均值={stats['mean']:.6f}, 方差={stats['variance']:.6f}, 标准差={stats['std']:.6f}")
    assert abs(stats['mean'] - mean_direct) < 1e-10
    assert abs(stats['variance'] - var_direct) < 1e-10
print("  beta_mean/beta_variance 与 beta_stats 结果一致 ✓")

print()
print("=" * 70)
print("Dirichlet分布测试")
print("=" * 70)
dirichlet_cases = [
    ([1, 1, 1], [1/3, 1/3, 1/3], "均匀Dirichlet"),
    ([2, 2, 2], [1/3, 1/3, 1/3], "对称Dirichlet"),
    ([5, 1, 1], [0.7, 0.2, 0.1], "偏向类别1"),
]

for alphas, x, desc in dirichlet_cases:
    pdf_val = dirichlet_pdf(x, alphas)
    mean = dirichlet_mean(alphas)
    var = dirichlet_variance(alphas)
    stats = dirichlet_stats(alphas)
    print(f"\n{desc} (α={alphas}):")
    print(f"  PDF(x={[round(v,3) for v in x]}) = {pdf_val:.6f}")
    print(f"  均值 = {[round(m,4) for m in mean]}")
    print(f"  方差 = {[round(v,6) for v in var]}")
    print(f"  集中参数 = {stats['concentration']}")
    
    for i in range(len(mean)):
        assert abs(mean[i] - stats['mean'][i]) < 1e-10
        assert abs(var[i] - stats['variance'][i]) < 1e-10
print("\n  dirichlet_mean/dirichlet_variance 与 dirichlet_stats 结果一致 ✓")

print()
print("=" * 70)
print("Dirichlet协方差矩阵验证")
print("=" * 70)
alphas = [2, 3, 5]
cov = dirichlet_covariance(alphas)
stats = dirichlet_stats(alphas)
print(f"α = {alphas}")
print("协方差矩阵:")
for row in cov:
    print(f"  [{', '.join(f'{v:.6f}' for v in row)}]")

for i in range(len(alphas)):
    for j in range(len(alphas)):
        assert abs(cov[i][j] - stats['covariance'][i][j]) < 1e-10
print("  dirichlet_covariance 与 dirichlet_stats 结果一致 ✓")

sum_row = [sum(row) for row in cov]
print(f"  每行求和: {[round(s, 10) for s in sum_row]} (理论上应为0，因为sum(x_i)=1)")

print()
print("=" * 70)
print("Dirichlet边界情况测试")
print("=" * 70)
boundary_cases = [
    ([2, 0.5], [1.0, 0.0], float('inf'), "x=[1,0], α=[2,0.5] → inf (x2=0,α2<1)"),
    ([0.5, 2], [0.0, 1.0], float('inf'), "x=[0,1], α=[0.5,2] → inf (x1=0,α1<1)"),
    ([3, 2], [0.0, 1.0], 0.0, "x=[0,1], α=[3,2] → 0 (α都>1)"),
    ([0.5, 0.5], [0.0, 1.0], float('inf'), "x=[0,1], α=[0.5,0.5] → inf"),
    ([1, 1, 1], [1/3, 1/3, 1/3], 2.0, "均匀Dirichlet PDF=2 ✓"),
]

all_passed = True
for alphas, x, expected, desc in boundary_cases:
    result = dirichlet_pdf(x, alphas)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"  {status} {desc}: 实际={result}, 期望={expected}")

print()
if all_passed:
    print("所有边界测试通过 ✓")
else:
    print("部分边界测试失败 ✗")

print()
print("=" * 70)
print("参数估计边界情况测试")
print("=" * 70)

try:
    beta_fit_moments([0.5] * 10)
    print("  [ERROR] 方差为0的样本应当抛出异常")
except ValueError as e:
    print(f"  ✓ 方差为0的样本正确抛出异常: {e}")

try:
    beta_fit_moments([0.1, 0.9])
    result = beta_fit_moments([0.1, 0.9])
    print(f"  ✓ 小样本估计: α={result['alpha']:.4f}, β={result['beta']:.4f}")
except Exception as e:
    print(f"  [ERROR] 小样本估计失败: {e}")

print()
print("=" * 70)
print("所有测试完成")
print("=" * 70)
