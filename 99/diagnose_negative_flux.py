import numpy as np
from sn_solver import Mesh, SNQuadrature, SNSolver, Material, solve_sn_1d

print("=" * 60)
print("诊断负通量问题")
print("=" * 60)

# 测试1: 强散射、粗网格条件（容易出现负通量）
print("\n测试1: 强散射、粗网格条件")
print("-" * 40)
x_min, x_max = 0.0, 10.0
n_cells = 20
sn_order = 8
sigma_t = 1.0
sigma_s = 0.99  # 强散射
source = np.ones(n_cells)

mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order,
                              sigma_t, sigma_s, source, tol=1e-8)

min_phi = np.min(phi)
min_psi = np.min(psi)
print(f"  标量通量最小值: {min_phi:.6e}")
print(f"  角通量最小值: {min_psi:.6e}")
print(f"  负标量通量点数: {np.sum(phi < 0)} / {len(phi)}")
print(f"  负角通量点数: {np.sum(psi < 0)} / {psi.size}")

if min_phi < 0:
    idx = np.argmin(phi)
    print(f"  最低通量位置 x={mesh.x_centers[idx]:.2f}, phi={phi[idx]:.6e}")

# 测试2: 不同SN阶数
print("\n测试2: 不同SN阶数（粗网格）")
print("-" * 40)
for sn_order in [2, 4, 8, 16]:
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order,
                                  sigma_t, sigma_s, source, tol=1e-8,
                                  max_iter=200)
    print(f"  SN{sn_order:2d}: min(phi)={np.min(phi):.6e}, "
          f"min(psi)={np.min(psi):.6e}, "
          f"负phi数={np.sum(phi < -1e-10):2d}")

# 测试3: 不同光学厚度（sigma_t * dx）
print("\n测试3: 不同光学厚度 (S8)")
print("-" * 40)
print(f"  {'n_cells':>8s}  {'dx':>6s}  {'sigma_t*dx':>10s}  "
      f"{'min(phi)':>12s}  {'负phi数':>8s}")
for n_cells in [5, 10, 20, 40, 80, 160]:
    mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, 8,
                                  sigma_t, sigma_s, np.ones(n_cells),
                                  tol=1e-8, max_iter=200)
    opt_thick = sigma_t * mesh.dx
    neg_count = np.sum(phi < -1e-10)
    print(f"  {n_cells:8d}  {mesh.dx:6.2f}  {opt_thick:10.3f}  "
          f"{np.min(phi):12.6e}  {neg_count:8d}")

# 测试4: 边界层附近
print("\n测试4: 真空边界附近的角通量分布")
print("-" * 40)
mesh, phi, psi = solve_sn_1d(x_min, x_max, 20, 8,
                              sigma_t, sigma_s, np.ones(20), tol=1e-8)
quad = SNQuadrature(8)
print("  左边界（x=0）的角通量:")
print(f"    {'方向(mu)':>10s}  {'psi':>12s}")
for n in range(len(quad.mus)):
    if quad.mus[n] < 0:  # 入射方向
        print(f"    {quad.mus[n]:10.6f}  {psi[n, 0]:12.6e}")

print("\n  右边界（x=10）的角通量:")
print(f"    {'方向(mu)':>10s}  {'psi':>12s}")
for n in range(len(quad.mus)):
    if quad.mus[n] > 0:  # 入射方向
        print(f"    {quad.mus[n]:10.6f}  {psi[n, -1]:12.6e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
