import numpy as np
import matplotlib.pyplot as plt
import time
from heat_method_geodesic import (
    generate_sphere_mesh,
    generate_torus_mesh,
    generate_cylinder_mesh,
    HeatMethodGeodesic,
    visualize_distance_field,
    visualize_shortest_path
)

print("=" * 80)
print("热方法测地线距离场演示")
print("=" * 80)

print("\n" + "=" * 80)
print("1. 球面测地线距离场")
print("=" * 80)

print("\n生成球面网格...")
sphere_mesh = generate_sphere_mesh(radius=1.0, n_lat=30, n_lon=60)
print(f"  顶点数: {sphere_mesh.n_vertices}")
print(f"  三角面数: {sphere_mesh.n_faces}")

print("\n初始化热方法求解器...")
solver_sphere = HeatMethodGeodesic(sphere_mesh)

source_idx = 0
print(f"\n计算距离场 (源点索引: {source_idx})...")
t0 = time.time()
dist_sphere = solver_sphere.compute_distance_field([source_idx])
t1 = time.time()
print(f"  耗时: {t1 - t0:.4f}s")
print(f"  最大距离: {np.max(dist_sphere):.4f}")
print(f"  理论最大距离 (πR): {np.pi * 1.0:.4f}")

source_pos = sphere_mesh.vertices[source_idx]
far_idx = np.argmax(dist_sphere)
far_pos = sphere_mesh.vertices[far_idx]
central_angle = np.arccos(np.clip(np.dot(source_pos, far_pos), -1, 1))
theoretical_max = central_angle * 1.0
print(f"  最远点距离: {np.max(dist_sphere):.4f} (理论: {theoretical_max:.4f})")
print(f"  误差: {abs(np.max(dist_sphere) - theoretical_max):.4f}")

target_idx = len(sphere_mesh.vertices) // 2 + 150
distance_pair = solver_sphere.geodesic_distance(source_idx, target_idx)
print(f"\n两点 {source_idx} -> {target_idx} 测地距离: {distance_pair:.4f}")
print(f"  欧氏距离: {np.linalg.norm(sphere_mesh.vertices[source_idx] - sphere_mesh.vertices[target_idx]):.4f}")

fig1, ax1 = visualize_distance_field(
    sphere_mesh, dist_sphere,
    title="球面测地线距离场 (热方法)"
)

print("\n" + "=" * 80)
print("2. 球面最短路径回溯")
print("=" * 80)

print(f"\n从目标点 {target_idx} 回溯到源点 {source_idx}...")
path_sphere = solver_sphere.shortest_path_indices(
    source_idx, target_idx,
    dist_field=dist_sphere,
    n_steps=200
)

path_length_euclidean = np.sum(np.linalg.norm(np.diff(path_sphere, axis=0), axis=1))
print(f"  路径点数: {len(path_sphere)}")
print(f"  路径长度(近似): {path_length_euclidean:.4f}")
print(f"  距离场查询: {dist_sphere[target_idx]:.4f}")

fig2, ax2 = visualize_shortest_path(
    sphere_mesh, dist_sphere, path_sphere,
    source_idx, target_idx,
    title="球面最短路径 (梯度下降回溯)"
)

print("\n" + "=" * 80)
print("3. 环面测地线距离场")
print("=" * 80)

print("\n生成环面网格...")
torus_mesh = generate_torus_mesh(R=2.0, r=0.8, n_u=50, n_v=30)
print(f"  顶点数: {torus_mesh.n_vertices}")
print(f"  三角面数: {torus_mesh.n_faces}")

print("\n初始化热方法求解器...")
solver_torus = HeatMethodGeodesic(torus_mesh)

source_idx_t = 0
print(f"\n计算距离场 (源点索引: {source_idx_t})...")
t0 = time.time()
dist_torus = solver_torus.compute_distance_field([source_idx_t])
t1 = time.time()
print(f"  耗时: {t1 - t0:.4f}s")
print(f"  最大距离: {np.max(dist_torus):.4f}")

fig3, ax3 = visualize_distance_field(
    torus_mesh, dist_torus,
    title="环面测地线距离场 (热方法)"
)

target_idx_t = len(torus_mesh.vertices) // 2 + 500
print(f"\n从目标点 {target_idx_t} 回溯到源点 {source_idx_t}...")
path_torus = solver_torus.shortest_path_indices(
    source_idx_t, target_idx_t,
    dist_field=dist_torus,
    n_steps=200
)

fig4, ax4 = visualize_shortest_path(
    torus_mesh, dist_torus, path_torus,
    source_idx_t, target_idx_t,
    title="环面最短路径"
)

print("\n" + "=" * 80)
print("4. 圆柱面测地线距离场")
print("=" * 80)

print("\n生成圆柱面网格...")
cyl_mesh = generate_cylinder_mesh(radius=1.0, height=4.0, n_theta=40, n_z=25)
print(f"  顶点数: {cyl_mesh.n_vertices}")
print(f"  三角面数: {cyl_mesh.n_faces}")

print("\n初始化热方法求解器...")
solver_cyl = HeatMethodGeodesic(cyl_mesh)

source_idx_c = 0
print(f"\n计算距离场 (源点索引: {source_idx_c})...")
t0 = time.time()
dist_cyl = solver_cyl.compute_distance_field([source_idx_c])
t1 = time.time()
print(f"  耗时: {t1 - t0:.4f}s")
print(f"  最大距离: {np.max(dist_cyl):.4f}")

fig5, ax5 = visualize_distance_field(
    cyl_mesh, dist_cyl,
    title="圆柱面测地线距离场 (热方法)"
)

target_idx_c = len(cyl_mesh.vertices) // 2 + 200
distance_cyl = solver_cyl.geodesic_distance(source_idx_c, target_idx_c)
print(f"\n两点 {source_idx_c} -> {target_idx_c} 测地距离: {distance_cyl:.4f}")

src = cyl_mesh.vertices[source_idx_c]
tgt = cyl_mesh.vertices[target_idx_c]
theta1 = np.arctan2(src[1], src[0])
theta2 = np.arctan2(tgt[1], tgt[0])
d_theta = min(abs(theta2 - theta1), 2 * np.pi - abs(theta2 - theta1))
d_z = abs(tgt[2] - src[2])
theoretical_cyl = np.sqrt((1.0 * d_theta)**2 + d_z**2)
print(f"  理论螺旋线长度: {theoretical_cyl:.4f}")
print(f"  误差: {abs(distance_cyl - theoretical_cyl):.4f}")

print("\n" + "=" * 80)
print("5. 多点源距离场")
print("=" * 80)

source_indices = [0, 500, 1000]
print(f"\n计算多点源距离场 (源点: {source_indices})...")
dist_multi = solver_sphere.compute_distance_field(source_indices)

fig6, ax6 = visualize_distance_field(
    sphere_mesh, dist_multi,
    title="球面多点源距离场"
)

for idx in source_indices:
    pos = sphere_mesh.vertices[idx]
    ax6.scatter(pos[0], pos[1], pos[2], color='green', s=200, zorder=10)

print("\n" + "=" * 80)
print("计算完成! 正在显示图形...")
print("=" * 80)

plt.show()
