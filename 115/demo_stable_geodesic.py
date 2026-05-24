import numpy as np
import matplotlib.pyplot as plt
from geodesic_solver import (
    Sphere, Torus, Cylinder, 
    visualize_geodesic, 
    compute_geodesic_length,
    check_arc_length_parameterization
)

print("=" * 70)
print("稳定测地线求解器演示 - 能量变分法与约束积分")
print("=" * 70)

print("\n" + "=" * 70)
print("1. 约束积分法演示 (带弧长参数化约束)")
print("=" * 70)

sphere = Sphere(radius=1.0)

u0 = np.pi / 4
v0 = 0.0
du0 = 0.5
dv0 = 1.0

print("\n--- 球面上的大圆 ---")
print(f"初始位置: u={u0:.4f}, v={v0:.4f}")
print(f"初始方向: du={du0:.4f}, dv={dv0:.4f}")

geo_constrained = sphere.compute_geodesic_constrained(
    u0, v0, du0, dv0,
    t_span=(0, 6),
    num_points=500
)

length_constrained = compute_geodesic_length(geo_constrained)
arc_check = check_arc_length_parameterization(sphere, geo_constrained)

print(f"求解成功: {geo_constrained['success']}")
print(f"测地线长度: {length_constrained:.6f}")
print(f"理论大圆长度(π): {np.pi:.6f}")
print(f"弧长参数化检查:")
print(f"  平均速度: {arc_check['mean_speed']:.6f}")
print(f"  速度标准差: {arc_check['std_speed']:.6f}")
print(f"  相对变化率: {arc_check['speed_variation']*100:.6f}%")

fig1, ax1 = visualize_geodesic(
    sphere, geo_constrained,
    title="球面上的测地线 - 约束积分法",
    u_range=(0, np.pi),
    v_range=(0, 2*np.pi)
)

print("\n" + "=" * 70)
print("2. 环面上的约束积分演示 (高曲率区域稳定性测试)")
print("=" * 70)

torus = Torus(major_radius=2.0, minor_radius=0.5)

u0_torus = 0.0
v0_torus = 0.0
du0_torus = 1.0
dv0_torus = 0.3

print("\n--- 环面上的测地线 (小管径=高曲率) ---")
print(f"环面参数: 大半径={torus.R}, 小半径={torus.r}")
print(f"初始位置: u={u0_torus:.4f}, v={v0_torus:.4f}")

geo_torus_constrained = torus.compute_geodesic_constrained(
    u0_torus, v0_torus, du0_torus, dv0_torus,
    t_span=(0, 30),
    num_points=1000
)

length_torus = compute_geodesic_length(geo_torus_constrained)
arc_check_torus = check_arc_length_parameterization(torus, geo_torus_constrained)

print(f"求解成功: {geo_torus_constrained['success']}")
print(f"测地线长度: {length_torus:.6f}")
print(f"弧长参数化检查:")
print(f"  平均速度: {arc_check_torus['mean_speed']:.6f}")
print(f"  速度标准差: {arc_check_torus['std_speed']:.6f}")
print(f"  相对变化率: {arc_check_torus['speed_variation']*100:.6f}%")

fig2, ax2 = visualize_geodesic(
    torus, geo_torus_constrained,
    title="环面上的测地线 - 约束积分法 (高曲率稳定)",
    u_range=(0, 2*np.pi),
    v_range=(0, 2*np.pi)
)

print("\n" + "=" * 70)
print("3. 变分法演示 - 两点间最短路径")
print("=" * 70)

u_start = np.pi / 6
v_start = 0.0
u_end = 5 * np.pi / 6
v_end = np.pi / 2

print("\n--- 球面上两点间最短路径 ---")
print(f"起点: u={u_start:.4f}, v={v_start:.4f}")
print(f"终点: u={u_end:.4f}, v={v_end:.4f}")

geo_variational = sphere.compute_geodesic_variational(
    u_start, v_start, u_end, v_end,
    n_points=60,
    max_iter=2000
)

length_var = compute_geodesic_length(geo_variational)

p1 = sphere.parametric(u_start, v_start)
p2 = sphere.parametric(u_end, v_end)
central_angle = np.arccos(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
theoretical_length = central_angle * 1.0

print(f"求解成功: {geo_variational['success']}")
print(f"优化消息: {geo_variational['message']}")
print(f"最终能量: {geo_variational['energy']:.6f}")
print(f"测地线长度: {length_var:.6f}")
print(f"理论长度(中心角×半径): {theoretical_length:.6f}")
print(f"误差: {abs(length_var - theoretical_length):.6f} ({abs(length_var - theoretical_length)/theoretical_length*100:.4f}%)")

fig3, ax3 = visualize_geodesic(
    sphere, geo_variational,
    title="球面两点间测地线 - 能量变分法",
    u_range=(0, np.pi),
    v_range=(0, 2*np.pi)
)

print("\n" + "=" * 70)
print("4. 打靶法演示 - 给定起点终点求初始方向")
print("=" * 70)

print("\n--- 圆柱面上的打靶法 ---")
cylinder = Cylinder(radius=1.0)

u0_cyl = 0.0
v0_cyl = 0.0
u_end_cyl = np.pi
v_end_cyl = 2.0

print(f"起点: u={u0_cyl:.4f}, v={v0_cyl:.4f}")
print(f"终点: u={u_end_cyl:.4f}, v={v_end_cyl:.4f}")

geo_shooting = cylinder.compute_geodesic_shooting(
    u0_cyl, v0_cyl, u_end_cyl, v_end_cyl,
    n_points=100,
    max_shots=100
)

length_shoot = compute_geodesic_length(geo_shooting)

print(f"求解成功: {geo_shooting['success']}")
print(f"最优初始方向: du={geo_shooting['initial_velocity'][0]:.6f}, dv={geo_shooting['initial_velocity'][1]:.6f}")
print(f"终点误差: {geo_shooting['target_error']:.6e}")
print(f"测地线长度: {length_shoot:.6f}")

theoretical_cyl = np.sqrt((np.pi * 1.0)**2 + (2.0)**2)
print(f"理论螺旋线长度: {theoretical_cyl:.6f}")
print(f"误差: {abs(length_shoot - theoretical_cyl):.6f}")

fig4, ax4 = visualize_geodesic(
    cylinder, geo_shooting,
    title="圆柱面测地线 - 打靶法",
    u_range=(0, 2*np.pi),
    v_range=(-1, 4)
)

print("\n" + "=" * 70)
print("5. 方法对比: RK45 vs 约束积分 (稳定性测试)")
print("=" * 70)

print("\n--- 长时间积分对比 ---")
t_long = 20

geo_rk45 = sphere.compute_geodesic_rk45(
    u0, v0, du0, dv0,
    t_span=(0, t_long),
    num_points=500
)

geo_constrained_long = sphere.compute_geodesic_constrained(
    u0, v0, du0, dv0,
    t_span=(0, t_long),
    num_points=500
)

arc_check_rk45 = check_arc_length_parameterization(sphere, geo_rk45)
arc_check_constrained = check_arc_length_parameterization(sphere, geo_constrained_long)

print(f"积分时长: {t_long}")
print("\nRK45方法:")
print(f"  求解成功: {geo_rk45['success']}")
print(f"  平均速度: {arc_check_rk45['mean_speed']:.6f}")
print(f"  速度标准差: {arc_check_rk45['std_speed']:.6f}")
print(f"  相对变化率: {arc_check_rk45['speed_variation']*100:.4f}%")

print("\n约束积分法:")
print(f"  求解成功: {geo_constrained_long['success']}")
print(f"  平均速度: {arc_check_constrained['mean_speed']:.6f}")
print(f"  速度标准差: {arc_check_constrained['std_speed']:.6f}")
print(f"  相对变化率: {arc_check_constrained['speed_variation']*100:.4f}%")

fig5, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': '3d'})

u_grid = np.linspace(0, np.pi, 30)
v_grid = np.linspace(0, 2*np.pi, 30)
U, V = np.meshgrid(u_grid, v_grid)
X, Y, Z = np.zeros_like(U), np.zeros_like(U), np.zeros_like(U)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        p = sphere.parametric(U[i, j], V[i, j])
        X[i, j], Y[i, j], Z[i, j] = p

for idx, (geo, title) in enumerate([(geo_rk45, 'RK45 (可能漂移)'), (geo_constrained_long, '约束积分 (稳定)')]):
    ax = axes[idx]
    ax.plot_surface(X, Y, Z, alpha=0.2, color='lightblue')
    points = geo['points']
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2)
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], color='green', s=100, label='起点')
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='blue', s=100, label='终点')
    ax.set_title(title)
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

plt.tight_layout()

print("\n" + "=" * 70)
print("计算完成! 正在显示图形...")
print("=" * 70)

plt.show()
