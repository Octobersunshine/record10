import numpy as np
import matplotlib.pyplot as plt
from geodesic_solver import Sphere, Torus, Cylinder, visualize_geodesic, compute_geodesic_length

print("=" * 60)
print("测地线求解器演示")
print("=" * 60)

print("\n1. 球面上的测地线 (大圆)")
print("-" * 60)
sphere = Sphere(radius=1.0)

u0_sphere = np.pi / 4
v0_sphere = 0.0
du0_sphere = 0.5
dv0_sphere = 1.0

geo_sphere = sphere.compute_geodesic(
    u0_sphere, v0_sphere, 
    du0_sphere, dv0_sphere,
    t_span=(0, 6),
    num_points=500
)

length_sphere = compute_geodesic_length(geo_sphere)
print(f"求解成功: {geo_sphere['success']}")
print(f"测地线长度: {length_sphere:.6f}")
print(f"理论大圆长度(周长一半): {np.pi:.6f}")

fig1, ax1 = visualize_geodesic(
    sphere, geo_sphere, 
    title="球面上的测地线 (大圆)",
    u_range=(0, np.pi),
    v_range=(0, 2*np.pi)
)

print("\n2. 环面上的测地线")
print("-" * 60)
torus = Torus(major_radius=2.0, minor_radius=1.0)

u0_torus = 0.0
v0_torus = 0.0
du0_torus = 1.0
dv0_torus = 0.5

geo_torus = torus.compute_geodesic(
    u0_torus, v0_torus,
    du0_torus, dv0_torus,
    t_span=(0, 20),
    num_points=1000
)

length_torus = compute_geodesic_length(geo_torus)
print(f"求解成功: {geo_torus['success']}")
print(f"测地线长度: {length_torus:.6f}")

fig2, ax2 = visualize_geodesic(
    torus, geo_torus,
    title="环面上的测地线",
    u_range=(0, 2*np.pi),
    v_range=(0, 2*np.pi)
)

print("\n3. 圆柱面上的测地线 (螺旋线)")
print("-" * 60)
cylinder = Cylinder(radius=1.0)

u0_cyl = 0.0
v0_cyl = 0.0
du0_cyl = 1.0
dv0_cyl = 0.5

geo_cyl = cylinder.compute_geodesic(
    u0_cyl, v0_cyl,
    du0_cyl, dv0_cyl,
    t_span=(0, 10),
    num_points=500
)

length_cyl = compute_geodesic_length(geo_cyl)
print(f"求解成功: {geo_cyl['success']}")
print(f"测地线长度: {length_cyl:.6f}")
print(f"理论螺旋线长度: {np.sqrt((2*np.pi*1.0*1.59)**2 + (5)**2):.6f} (近似)")

fig3, ax3 = visualize_geodesic(
    cylinder, geo_cyl,
    title="圆柱面上的测地线 (螺旋线)",
    u_range=(0, 2*np.pi),
    v_range=(-2, 8)
)

print("\n4. 环面上的闭测地线 (环绕环管)")
print("-" * 60)
u0_torus2 = 0.0
v0_torus2 = 0.0
du0_torus2 = 1.0
dv0_torus2 = 0.0

geo_torus2 = torus.compute_geodesic(
    u0_torus2, v0_torus2,
    du0_torus2, dv0_torus2,
    t_span=(0, 6.3),
    num_points=500
)

length_torus2 = compute_geodesic_length(geo_torus2)
print(f"求解成功: {geo_torus2['success']}")
print(f"测地线长度: {length_torus2:.6f}")
print(f"理论小圆周长: {2*np.pi*1.0:.6f}")

fig4, ax4 = visualize_geodesic(
    torus, geo_torus2,
    title="环面上的闭测地线 (环绕环管)",
    u_range=(0, 2*np.pi),
    v_range=(0, 2*np.pi)
)

print("\n" + "=" * 60)
print("计算完成! 正在显示图形...")
print("=" * 60)

plt.show()
