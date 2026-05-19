import numpy as np
from scipy import stats
import sys

print("柯西分布增强功能测试")
print("=" * 80)

try:
    from cauchy_estimator_enhanced import CauchyEstimator, generate_sample_data, generate_normal_data
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

np.random.seed(42)

def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

def print_distribution_test_results(test_results):
    print("\n柯西分布:")
    print(f"  参数: x0={test_results['cauchy']['params']['x0']:.4f}, "
          f"γ={test_results['cauchy']['params']['gamma']:.4f}")
    print(f"  KS检验: 统计量={test_results['cauchy']['ks_test']['statistic']:.4f}, "
          f"p值={test_results['cauchy']['ks_test']['p_value']:.4f}")
    print(f"  信息准则: AIC={test_results['cauchy']['aic']:.2f}, "
          f"BIC={test_results['cauchy']['bic']:.2f}")
    print(f"  QQ相关系数: {test_results['cauchy']['qq_stats']['correlation']:.4f}")
    
    print("\n正态分布:")
    print(f"  参数: μ={test_results['normal']['params']['mu']:.4f}, "
          f"σ={test_results['normal']['params']['sigma']:.4f}")
    print(f"  KS检验: 统计量={test_results['normal']['ks_test']['statistic']:.4f}, "
          f"p值={test_results['normal']['ks_test']['p_value']:.4f}")
    print(f"  信息准则: AIC={test_results['normal']['aic']:.2f}, "
          f"BIC={test_results['normal']['bic']:.2f}")
    print(f"  QQ相关系数: {test_results['normal']['qq_stats']['correlation']:.4f}")
    
    print("\n分布判断:")
    comp = test_results['comparison']
    print(f"  KS检验接受柯西: {'是' if comp['is_cauchy_by_ks'] else '否'}")
    print(f"  KS检验接受正态: {'是' if comp['is_normal_by_ks'] else '否'}")
    print(f"  AIC偏好: {comp['preferred_by_aic']}")
    print(f"  BIC偏好: {comp['preferred_by_bic']}")
    print(f"  QQ相关系数偏好: {comp['preferred_by_qq_correlation']}")
    print(f"  AIC差值(正态-柯西): {comp['aic_diff']:.2f}")
    print(f"  推荐分布: {comp['recommended_distribution'].upper()}")

def print_robustness_results(robustness_results):
    print("\n原始参数:")
    orig = robustness_results['original']
    print(f"  柯西: x0={orig['cauchy']['x0']:.4f}, γ={orig['cauchy']['gamma']:.4f}")
    print(f"  正态: μ={orig['normal']['mu']:.4f}, σ={orig['normal']['sigma']:.4f}")
    
    print("\n加入异常值后的参数:")
    outl = robustness_results['with_outliers']
    print(f"  柯西: x0={outl['cauchy']['x0']:.4f}, γ={outl['cauchy']['gamma']:.4f}")
    print(f"  正态: μ={outl['normal']['mu']:.4f}, σ={outl['normal']['sigma']:.4f}")
    
    print("\n相对参数变化(相对于原始尺度):")
    changes = robustness_results['changes']
    print(f"  柯西位置变化: {changes['cauchy_x0_relative_change']:.4f} 个γ")
    print(f"  正态均值变化: {changes['normal_mu_relative_change']:.4f} 个σ")
    print(f"  柯西尺度变化: {changes['cauchy_gamma_relative_change']:.4f} 个γ")
    print(f"  正态标准差变化: {changes['normal_sigma_relative_change']:.4f} 个σ")
    
    print("\n稳健性对比(正态变化/柯西变化):")
    print(f"  位置参数稳健性比率: {changes['robustness_ratio_x0']:.2f}x")
    print(f"  尺度参数稳健性比率: {changes['robustness_ratio_gamma']:.2f}x")
    
    print("\n结论:")
    conc = robustness_results['conclusion']
    print(f"  柯西位置参数更稳健: {'是' if conc['cauchy_more_robust_x0'] else '否'}")
    print(f"  柯西尺度参数更稳健: {'是' if conc['cauchy_more_robust_gamma'] else '否'}")
    
    outlier_info = robustness_results['outlier_info']
    print(f"\n异常值信息:")
    print(f"  异常值数量: {outlier_info['n_outliers']}")
    print(f"  异常值大小: {outlier_info['outlier_magnitude']} 倍尺度参数")
    print(f"  异常值示例: {outlier_info['outlier_values'][:3]}")

print_section("测试1: 柯西分布数据检验")
cauchy_data = generate_sample_data(true_location=5.0, true_scale=2.0, size=300, random_seed=42)

estimator = CauchyEstimator()
estimator.fit(cauchy_data, method='multistart')
test_results_cauchy = estimator.test_distribution(cauchy_data)

print(f"真实分布: 柯西分布(x0=5.0, γ=2.0)")
print(f"样本量: {len(cauchy_data)}")
print_distribution_test_results(test_results_cauchy)

print_section("测试2: 正态分布数据检验")
normal_data = generate_normal_data(mu=5.0, sigma=2.0, size=300, random_seed=123)

estimator2 = CauchyEstimator()
estimator2.fit(normal_data, method='multistart')
test_results_normal = estimator2.test_distribution(normal_data)

print(f"真实分布: 正态分布(μ=5.0, σ=2.0)")
print(f"样本量: {len(normal_data)}")
print_distribution_test_results(test_results_normal)

print_section("测试3: 混合分布数据检验")
mixed_data = np.concatenate([
    generate_sample_data(0, 1, 150, random_seed=1),
    generate_normal_data(5, 1, 150, random_seed=2)
])

estimator3 = CauchyEstimator()
estimator3.fit(mixed_data, method='multistart')
test_results_mixed = estimator3.test_distribution(mixed_data)

print(f"真实分布: 混合分布(柯西+正态)")
print(f"样本量: {len(mixed_data)}")
print_distribution_test_results(test_results_mixed)

print_section("测试4: 异常值稳健性检验(柯西数据)")
base_cauchy = generate_sample_data(0, 1, 100, random_seed=456)

estimator4 = CauchyEstimator()
estimator4.fit(base_cauchy, method='de')
robustness_cauchy = estimator4.outlier_robustness_test(
    base_cauchy, n_outliers=5, outlier_magnitude=15
)

print(f"基础数据: 柯西分布(x0=0, γ=1), 样本量={len(base_cauchy)}")
print_robustness_results(robustness_cauchy)

print_section("测试5: 异常值稳健性检验(正态数据)")
base_normal = generate_normal_data(0, 1, 100, random_seed=789)

estimator5 = CauchyEstimator()
estimator5.fit(base_normal, method='de')
robustness_normal = estimator5.outlier_robustness_test(
    base_normal, n_outliers=5, outlier_magnitude=15
)

print(f"基础数据: 正态分布(μ=0, σ=1), 样本量={len(base_normal)}")
print_robustness_results(robustness_normal)

print_section("测试6: 极端异常值检验")
extreme_data = generate_sample_data(0, 1, 50, random_seed=111)

estimator6 = CauchyEstimator()
estimator6.fit(extreme_data, method='de')
robustness_extreme = estimator6.outlier_robustness_test(
    extreme_data, n_outliers=3, outlier_magnitude=50
)

print(f"基础数据: 柯西分布(x0=0, γ=1), 样本量={len(extreme_data)}")
print(f"异常值: 3个, 大小=50倍尺度参数")

changes = robustness_extreme['changes']
print(f"\n位置参数稳健性比率: {changes['robustness_ratio_x0']:.2f}x")
print(f"尺度参数稳健性比率: {changes['robustness_ratio_gamma']:.2f}x")

print_section("总结")
print("""
关键发现:
1. 分布检验:
   - KS检验通过p值判断数据是否符合理论分布
   - AIC/BIC信息准则用于模型比较，值越小越好
   - QQ图相关系数量化理论与样本分位数的线性关系

2. 推荐机制:
   - 综合AIC、BIC、QQ相关系数进行投票
   - 2票及以上推荐该分布，否则标记为不确定

3. 异常值稳健性:
   - 柯西分布由于重尾特性，对异常值显著更稳健
   - 稳健性比率 > 1 表示柯西比正态受异常值影响更小
   - 通常位置参数(x0)的稳健性提升尤为明显

使用建议:
   - 对金融数据、物理测量等可能有极端值的数据，优先考虑柯西分布
   - 使用test_distribution()进行数据分布诊断
   - 使用outlier_robustness_test()量化评估异常值影响
""")

print("=" * 80)
print("\n快速使用示例:")
print("  from cauchy_estimator_enhanced import CauchyEstimator")
print("  estimator = CauchyEstimator()")
print("  estimator.fit(your_data)")
print("  ")
print("  # 分布检验")
print("  result = estimator.test_distribution(your_data)")
print("  print('推荐分布:', result['comparison']['recommended_distribution'])")
print("  ")
print("  # 异常值稳健性检验")
print("  robustness = estimator.outlier_robustness_test(your_data)")
print("  print('稳健性比率:', robustness['changes']['robustness_ratio_x0'])")
print("=" * 80)
