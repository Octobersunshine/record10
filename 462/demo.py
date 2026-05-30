"""
数值微分功能完整演示 - 高阶导数、批量计算、精度对比
运行: python demo.py
"""
import sys
sys.path.insert(0, r'e:\temp\record10\462')

from numerical_differentiation import NumericalDifferentiation
import numpy as np

print("=" * 80)
print("数值微分 - 完整功能演示")
print("=" * 80)

# 测试函数 f(x) = x^3 + sin(x)
def f(x):
    return x**3 + np.sin(x)

def df_exact(x):
    return 3 * x**2 + np.cos(x)

def d2f_exact(x):
    return 6 * x - np.sin(x)

def d3f_exact(x):
    return 6 - np.cos(x)

nd = NumericalDifferentiation()
x = 1.5

print(f"\n测试函数: f(x) = x^3 + sin(x)")
print(f"计算点: x = {x}")
print(f"精确一阶导数: {df_exact(x):.12f}")
print(f"精确二阶导数: {d2f_exact(x):.12f}")
print(f"精确三阶导数: {d3f_exact(x):.12f}")

print("\n" + "=" * 80)
print("1. 高阶导数计算 (二阶、三阶)")
print("=" * 80)

# 二阶导数
print("\n--- 二阶导数 ---")
d2_central, err_est = nd.second_derivative_central(f, x, estimate_error=True)
print(f"中心差分 (自动步长): {d2_central:.12f}")
print(f"  估计误差: {err_est:.2e}")
print(f"  真实误差: {abs(d2_central - d2f_exact(x)):.2e}")

d2_rich, err_rich = nd.second_derivative_central(f, x, richardson=True)
print(f"\n中心差分 + Richardson: {d2_rich:.12f}")
print(f"  估计误差: {err_rich:.2e}")
print(f"  真实误差: {abs(d2_rich - d2f_exact(x)):.2e}")

# 三阶导数
print("\n--- 三阶导数 ---")
d3_central, err_est = nd.third_derivative_central(f, x, estimate_error=True)
print(f"中心差分 (自动步长): {d3_central:.12f}")
print(f"  估计误差: {err_est:.2e}")
print(f"  真实误差: {abs(d3_central - d3f_exact(x)):.2e}")

d3_rich, err_rich = nd.third_derivative_central(f, x, richardson=True)
print(f"\n中心差分 + Richardson: {d3_rich:.12f}")
print(f"  估计误差: {err_rich:.2e}")
print(f"  真实误差: {abs(d3_rich - d3f_exact(x)):.2e}")

print("\n" + "=" * 80)
print("2. 批量求导 (数组输入)")
print("=" * 80)

x_array = np.linspace(0, np.pi, 5)
print(f"\n计算点数组: {x_array}")

# 批量一阶导数
d1_batch, err1_batch = nd.central_difference(f, x_array, estimate_error=True)
print(f"\n--- 批量一阶导数 ---")
for xi, di, ei in zip(x_array, d1_batch, err1_batch):
    exact = df_exact(xi)
    true_err = abs(di - exact)
    print(f"x = {xi:.4f}: {di:.10f} (估计: {ei:.2e}, 真实: {true_err:.2e})")

# 批量二阶导数
d2_batch, err2_batch = nd.second_derivative_central(f, x_array, estimate_error=True)
print(f"\n--- 批量二阶导数 ---")
for xi, di, ei in zip(x_array, d2_batch, err2_batch):
    exact = d2f_exact(xi)
    true_err = abs(di - exact)
    print(f"x = {xi:.4f}: {di:.10f} (估计: {ei:.2e}, 真实: {true_err:.2e})")

# 使用统一接口
print("\n--- 使用 compute_derivative 统一接口 ---")
d1 = nd.compute_derivative('central', f, x, order=1)
d2 = nd.compute_derivative('central', f, x, order=2)
d3 = nd.compute_derivative('central', f, x, order=3)
print(f"order=1: {d1:.10f}")
print(f"order=2: {d2:.10f}")
print(f"order=3: {d3:.10f}")

print("\n" + "=" * 80)
print("3. 精度对比分析")
print("=" * 80)

# 不同方法在不同步长下的精度
print(f"\n--- 不同步长下的误差对比 (一阶导数, 中心差分) ---")
print(f"{'步长 h':>12} {'导数近似':>15} {'真实误差':>12}")
print("-" * 45)
for h in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
    d = nd.central_difference(f, x, h=h)
    err = abs(d - df_exact(x))
    marker = " <-- 接近最优" if abs(h - nd._optimal_h(x, 'central')) < 1e-7 else ""
    print(f"{h:>12.2e} {d:>15.10f} {err:>12.2e}{marker}")

print(f"\n理论最优步长: {nd._optimal_h(x, 'central'):.2e}")

print("\n" + "=" * 80)
print("4. 稳定性分析 (观察误差随步长变化趋势)")
print("=" * 80)

stability = nd.stability_analysis(f, x, order=1, method='central', n_steps=20)

# 找到误差最小的区域
min_err_idx = np.argmin(stability['step_errors'][1:]) + 1
optimal_h = stability['h'][min_err_idx]
optimal_d = stability['derivatives'][min_err_idx]

print(f"\n中心差分一阶导数稳定性分析:")
print(f"搜索到的最优步长: {optimal_h:.2e}")
print(f"对应导数值: {optimal_d:.12f}")
print(f"精确值: {df_exact(x):.12f}")
print(f"真实误差: {abs(optimal_d - df_exact(x)):.2e}")

print("\n" + "=" * 80)
print("5. 不同方法对比 (前向 vs 后向 vs 中心)")
print("=" * 80)

print(f"\n在 x = {x} 处的一阶导数:")
print(f"{'方法':>10} {'结果':>18} {'估计误差':>14} {'真实误差':>12}")
print("-" * 60)

for method in ['forward', 'backward', 'central']:
    d, err_est = nd.compute_derivative(method, f, x, order=1, estimate_error=True)
    true_err = abs(d - df_exact(x))
    print(f"{method:>10} {d:>18.10f} {err_est:>14.2e} {true_err:>12.2e}")

print("\n" + "=" * 80)
print("6. 完整使用总结")
print("=" * 80)
print("""
核心功能清单:
  ✓ 一阶导数: forward_difference, backward_difference, central_difference
  ✓ 二阶导数: second_derivative_forward/backward/central
  ✓ 三阶导数: third_derivative_central (仅中心差分)
  ✓ 自动步长选择: 基于 √ε 和 ε^(1/3) 的最优公式
  ✓ Richardson 外推: 消除低阶误差, 提高精度
  ✓ 误差估计: 基于 Richardson 外推的误差估计
  ✓ 批量计算: 支持 numpy 数组输入
  ✓ 离散点支持: 处理 (x_points, y_points) 格式数据
  ✓ 精度对比: compare_methods 方法横向对比
  ✓ 稳定性分析: stability_analysis 观察误差变化趋势

推荐使用模式:
  # 最高精度, 带误差估计
  d, err = nd.central_difference(f, x, richardson=True)
  
  # 批量计算
  d_array = nd.central_difference(f, x_array)
  
  # 高阶导数
  d2 = nd.compute_derivative('central', f, x, order=2)
""")

print("=" * 80)
print("演示完成!")
print("=" * 80)
