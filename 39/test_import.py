import sys
print("Python 版本:", sys.version)
print()

try:
    import numpy as np
    print("✓ numpy 导入成功")
except ImportError as e:
    print("✗ numpy 导入失败:", e)

try:
    from scipy import stats
    print("✓ scipy 导入成功")
except ImportError as e:
    print("✗ scipy 导入失败:", e)

try:
    from cauchy_estimator import CauchyEstimator, estimate_cauchy_from_array, generate_sample_data
    print("✓ cauchy_estimator 导入成功")
except ImportError as e:
    print("✗ cauchy_estimator 导入失败:", e)

print()
print("=" * 50)
print("开始测试功能...")
print()

try:
    data = generate_sample_data(true_location=2.0, true_scale=1.5, size=100, random_seed=42)
    print("✓ 生成测试数据成功")
    print(f"  数据长度: {len(data)}")
    print(f"  前5个样本: {data[:5]}")
except Exception as e:
    print("✗ 生成测试数据失败:", e)

print()

try:
    x0, gamma, estimator = estimate_cauchy_from_array(data)
    print("✓ 参数估计成功")
    print(f"  估计的位置参数 x0 = {x0:.6f}")
    print(f"  估计的尺度参数 γ = {gamma:.6f}")
except Exception as e:
    print("✗ 参数估计失败:", e)

print()

try:
    params = estimator.get_params()
    print("✓ 获取参数成功")
    print(f"  通过get_params()获取: x0={params['x0']:.6f}, γ={params['gamma']:.6f}")
except Exception as e:
    print("✗ 获取参数失败:", e)

print()

try:
    pdf_val = estimator.pdf(2.0)
    print("✓ 概率密度计算成功")
    print(f"  f(2.0) = {pdf_val:.6f}")
except Exception as e:
    print("✗ 概率密度计算失败:", e)

print()

try:
    ll = estimator.log_likelihood(data)
    print("✓ 对数似然计算成功")
    print(f"  对数似然值 = {ll:.2f}")
except Exception as e:
    print("✗ 对数似然计算失败:", e)

print()
print("=" * 50)
print("所有测试完成!")
