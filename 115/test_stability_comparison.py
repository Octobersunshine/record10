import numpy as np
import matplotlib.pyplot as plt
import time
from geodesic_solver import (
    Sphere, Torus, Cylinder,
    compute_geodesic_length,
    check_arc_length_parameterization
)

print("=" * 80)
print("测地线求解器稳定性综合对比测试")
print("=" * 80)

methods = [
    ('RK45', lambda s, *args: s.compute_geodesic_rk45(*args)),
    ('约束积分', lambda s, *args: s.compute_geodesic_constrained(*args)),
    ('隐式中点', lambda s, *args: s.compute_geodesic_implicit_midpoint(*args)),
    ('自适应步长', lambda s, *args: s.compute_geodesic_adaptive(*args))
]

print("\n" + "=" * 80)
print("测试1: 球面上大圆 - 标准情况")
print("=" * 80)

sphere = Sphere(radius=1.0)
u0, v0 = np.pi / 4, 0.0
du0, dv0 = 0.5, 1.0
t_span = (0, 20)
num_points = 500

results_sphere = []

for name, method_func in methods:
    t0 = time.time()
    try:
        geo = method_func(sphere, u0, v0, du0, dv0, t_span, num_points)
        t1 = time.time()
        
        length = compute_geodesic_length(geo)
        arc_check = check_arc_length_parameterization(sphere, geo)
        
        theoretical_length = 20.0
        
        results_sphere.append({
            'name': name,
            'success': geo['success'],
            'time': t1 - t0,
            'length': length,
            'error': abs(length - theoretical_length),
            'speed_mean': arc_check['mean_speed'],
            'speed_std': arc_check['std_speed'],
            'speed_var': arc_check['speed_variation']
        })
        
        print(f"\n{name}:")
        print(f"  成功: {geo['success']}")
        print(f"  耗时: {t1 - t0:.4f}s")
        print(f"  长度: {length:.6f} (理论: {theoretical_length:.6f})")
        print(f"  误差: {abs(length - theoretical_length):.6f}")
        print(f"  速度变化: {arc_check['speed_variation'] * 100:.4f}%")
    except Exception as e:
        print(f"\n{name}: 失败 - {str(e)}")
        results_sphere.append({
            'name': name,
            'success': False,
            'error': str(e)
        })

print("\n" + "=" * 80)
print("测试2: 高曲率环面 - 刚性测试")
print("=" * 80)

torus_thin = Torus(major_radius=3.0, minor_radius=0.2)
u0_t, v0_t = 0.0, 0.0
du0_t, dv0_t = 1.0, 0.5
t_span_t = (0, 50)
num_points_t = 1000

results_torus = []

for name, method_func in methods:
    t0 = time.time()
    try:
        geo = method_func(torus_thin, u0_t, v0_t, du0_t, dv0_t, t_span_t, num_points_t)
        t1 = time.time()
        
        length = compute_geodesic_length(geo)
        arc_check = check_arc_length_parameterization(torus_thin, geo)
        
        points = geo['points']
        max_deviation = np.max(np.abs(np.linalg.norm(points, axis=1) - torus_thin.R))
        
        results_torus.append({
            'name': name,
            'success': geo['success'],
            'time': t1 - t0,
            'length': length,
            'speed_mean': arc_check['mean_speed'],
            'speed_std': arc_check['std_speed'],
            'speed_var': arc_check['speed_variation'],
            'surface_deviation': max_deviation
        })
        
        print(f"\n{name}:")
        print(f"  成功: {geo['success']}")
        print(f"  耗时: {t1 - t0:.4f}s")
        print(f"  长度: {length:.6f}")
        print(f"  速度变化: {arc_check['speed_variation'] * 100:.4f}%")
        print(f"  曲面偏离: {max_deviation:.2e}")
    except Exception as e:
        print(f"\n{name}: 失败 - {str(e)}")
        results_torus.append({
            'name': name,
            'success': False,
            'error': str(e)
        })

print("\n" + "=" * 80)
print("测试3: 变分法对比 - 两点边值问题")
print("=" * 80)

u_start, v_start = np.pi / 6, 0.0
u_end, v_end = 5 * np.pi / 6, np.pi / 2
n_points_var = 40

t0 = time.time()
geo_var = sphere.compute_geodesic_variational(u_start, v_start, u_end, v_end, n_points_var, max_iter=1000)
t1 = time.time()

t0_fast = time.time()
geo_var_fast = sphere.compute_geodesic_variational_fast(u_start, v_start, u_end, v_end, n_points_var, max_iter=1000)
t1_fast = time.time()

p1 = sphere.parametric(u_start, v_start)
p2 = sphere.parametric(u_end, v_end)
central_angle = np.arccos(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
theoretical_length = central_angle * 1.0

len_var = compute_geodesic_length(geo_var)
len_fast = compute_geodesic_length(geo_var_fast)

print(f"\n标准变分法 (数值梯度):")
print(f"  成功: {geo_var['success']}")
print(f"  耗时: {t1 - t0:.4f}s")
print(f"  迭代次数: {geo_var.get('nit', 'N/A')}")
print(f"  能量: {geo_var['energy']:.6f}")
print(f"  长度: {len_var:.6f}")
print(f"  误差: {abs(len_var - theoretical_length):.6f}")

print(f"\n快速变分法 (解析梯度):")
print(f"  成功: {geo_var_fast['success']}")
print(f"  耗时: {t1_fast - t0_fast:.4f}s")
print(f"  迭代次数: {geo_var_fast.get('nit', 'N/A')}")
print(f"  能量: {geo_var_fast['energy']:.6f}")
print(f"  长度: {len_fast:.6f}")
print(f"  误差: {abs(len_fast - theoretical_length):.6f}")
print(f"  加速比: {(t1 - t0) / (t1_fast - t0_fast):.2f}x")

print("\n" + "=" * 80)
print("测试4: 可视化对比")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
fig.suptitle('测地线求解器稳定性对比', fontsize=16)

u_grid = np.linspace(0, np.pi, 30)
v_grid = np.linspace(0, 2 * np.pi, 30)
U, V = np.meshgrid(u_grid, v_grid)
X, Y, Z = np.zeros_like(U), np.zeros_like(U), np.zeros_like(U)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        p = sphere.parametric(U[i, j], V[i, j])
        X[i, j], Y[i, j], Z[i, j] = p

geo_results = []
for name, method_func in methods:
    try:
        geo = method_func(sphere, u0, v0, du0, dv0, (0, 10), 200)
        geo_results.append((name, geo))
    except:
        pass

for idx, (name, geo) in enumerate(geo_results):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    ax.plot_surface(X, Y, Z, alpha=0.2, color='lightblue')
    points = geo['points']
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2)
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], color='green', s=80, label='起点')
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='blue', s=80, label='终点')
    
    arc_check = check_arc_length_parameterization(sphere, geo)
    ax.set_title(f'{name}\n速度变化: {arc_check["speed_variation"]*100:.3f}%')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

for idx in range(len(geo_results), 6):
    row, col = idx // 3, idx % 3
    axes[row, col].axis('off')

plt.tight_layout()

print("\n" + "=" * 80)
print("测试5: 自适应步长统计")
print("=" * 80)

geo_adaptive = sphere.compute_geodesic_adaptive(
    u0, v0, du0, dv0,
    t_span=(0, 20),
    num_points=500
)

if 'dt_stats' in geo_adaptive:
    stats = geo_adaptive['dt_stats']
    print(f"\n自适应步长统计:")
    print(f"  平均步长: {stats['mean_dt']:.6f}")
    print(f"  最小步长: {stats['min_dt']:.6f}")
    print(f"  最大步长: {stats['max_dt']:.6f}")
    print(f"  总步数: {stats['n_steps']}")
    print(f"  效率提升: {stats['n_steps'] / (20/1e-3):.2%} (相比固定小步长)")

print("\n" + "=" * 80)
print("测试完成! 正在显示图形...")
print("=" * 80)

plt.show()
