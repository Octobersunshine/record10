import numpy as np
from box_counting import (
    box_counting_dimension,
    generate_fractal_points,
    generate_multifractal_points,
    test_robustness,
    generalized_dimension,
    information_dimension,
    correlation_dimension,
    correlation_dimension_gp,
    plot_multifractal_spectrum,
    load_binary_image
)

print("=" * 70)
print("盒计数法分形维数估计 - 多重网格平均改进版")
print("=" * 70)

print("\n1. 谢尔宾斯基三角形测试 (单分形)")
print("   理论维数: D_q ≈ 1.585 (单分形，D_q不随q变化)")
print("-" * 60)
sierpinski = generate_fractal_points('sierpinski', num_points=20000)
D, sizes, counts, slope, intercept, r_value, counts_std, all_counts = box_counting_dimension(
    sierpinski, is_image=False, min_box_size=2, num_sizes=12,
    use_multigrid=True, num_offsets=10, seed=42
)
print(f"   盒维数 D_0 (多重网格平均): {D:.4f}")
print(f"   R²值: {r_value**2:.4f}")

print("\n2. 广义维数谱 D_q 测试 (多分形点集)")
print("-" * 60)
print("生成多分形点集（二项测度）...")
multifractal = generate_multifractal_points(num_points=50000, seed=42)
print(f"点集大小: {len(multifractal)}")

print("\n计算广义维数谱 D_q (q从-5到5)...")
q_values, D_q, r_values, box_sizes, all_data = generalized_dimension(
    multifractal, min_box_size=4, num_sizes=10,
    use_multigrid=True, num_offsets=5, seed=42
)

print("\nD_q 结果:")
for q, d, r in zip(q_values, D_q, r_values):
    print(f"  q={q:4.1f}: D_q={d:.4f}, R²={r:.4f}")

print("\n关键维数:")
idx_q0 = np.argmin(np.abs(q_values - 0))
idx_q1 = np.argmin(np.abs(q_values - 1))
idx_q2 = np.argmin(np.abs(q_values - 2))
print(f"  盒维数 D_0 = {D_q[idx_q0]:.4f}")
print(f"  信息维数 D_1 = {D_q[idx_q1]:.4f}")
print(f"  关联维数 D_2 = {D_q[idx_q2]:.4f}")

if D_q[idx_q0] > D_q[idx_q1] > D_q[idx_q2]:
    print("  ✓ 检测到多分形特性: D_0 > D_1 > D_2")
else:
    print("  单分形特性: D_q 近似常数")

print("\n3. 专门计算函数测试:")
print("-" * 60)
print("信息维数 D_1 (专门函数):")
D1, r1, _, _ = information_dimension(
    multifractal, min_box_size=4, num_sizes=10,
    use_multigrid=True, num_offsets=5, seed=42
)
print(f"  D_1 = {D1:.4f}, R² = {r1:.4f}")

print("\n关联维数 D_2 (盒计数法):")
D2, r2, _, _ = correlation_dimension(
    multifractal, min_box_size=4, num_sizes=10,
    use_multigrid=True, num_offsets=5, seed=42
)
print(f"  D_2 = {D2:.4f}, R² = {r2:.4f}")

print("\n关联维数 D_2 (Grassberger-Procaccia算法, 小规模测试):")
small_points = multifractal[::5]  # 减少点数加速计算
D2_gp, r2_gp, _, _ = correlation_dimension_gp(
    small_points, num_eps=10, min_eps=0.02
)
print(f"  D_2 = {D2_gp:.4f}, R² = {r2_gp:.4f}")

print("\n4. 鲁棒性对比测试:")
print("-" * 60)
results_single, results_multi = test_robustness(
    sierpinski, num_trials=15, min_box_size=2, num_sizes=10, num_offsets=10
)

print("\n" + "=" * 70)
print("使用示例:")
print("=" * 70)
print("""
# 1. 盒维数 (D_0)
from box_counting import box_counting_dimension, generate_fractal_points
points = generate_fractal_points('sierpinski', num_points=20000)
D0, sizes, counts, _, _, r_value, _, _ = box_counting_dimension(
    points, is_image=False, use_multigrid=True, num_offsets=10
)
print(f"盒维数 D_0 = {D0:.4f}")

# 2. 广义维数谱 D_q (多分形分析)
from box_counting import generalized_dimension, generate_multifractal_points
multi_points = generate_multifractal_points(num_points=50000)
q_values, D_q, r_values, _, _ = generalized_dimension(
    multi_points, q_values=[-2, -1, 0, 1, 2, 3],
    use_multigrid=True, num_offsets=5
)
for q, d in zip(q_values, D_q):
    print(f"D_{q} = {d:.4f}")

# 3. 信息维数 D_1
from box_counting import information_dimension
D1, r1, _, _ = information_dimension(points, use_multigrid=True)
print(f"信息维数 D_1 = {D1:.4f}")

# 4. 关联维数 D_2
from box_counting import correlation_dimension, correlation_dimension_gp
D2, r2, _, _ = correlation_dimension(points, use_multigrid=True)
print(f"关联维数 D_2 (盒计数) = {D2:.4f}")

# Grassberger-Procaccia算法 (更精确但更慢)
D2_gp, r2_gp, _, _ = correlation_dimension_gp(points[::5])
print(f"关联维数 D_2 (GP算法) = {D2_gp:.4f}")
""")

print("\n=== 测试完成 ===")
