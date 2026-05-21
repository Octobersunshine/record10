import numpy as np
from sn_solver import solve_sn_1d, Mesh, SNQuadrature, SNSolver, Material

print("=" * 60)
print("极端条件下的负通量检测")
print("=" * 60 + "\n")

# 分析菱形差分的稳定性条件
print("菱形差分稳定性分析:")
print("ψ_{i+1} = [ψ_i * (2μ/Δx - σ_t) + 2Q_i] / (2μ/Δx + σ_t)")
print("当 2μ/Δx - σ_t < 0 即 σ_tΔx > 2μ 时，系数为负")
print("对小角度μ，更容易出现负系数\n")

quad = SNQuadrature(8)
print(f"S8求积组角度: {quad.mus}")
print(f"最小|μ|: {np.min(np.abs(quad.mus)):.6f}")
print(f"临界光学厚度(对最小|μ|): {2*np.min(np.abs(quad.mus)):.6f}\n")

# 测试1: 边界附近 + 极大光学厚度 + 零源
print("测试1: 左半段有源，右半段零源，σ_tΔx = 10")
print("-" * 60)

x_min, x_max = 0.0, 10.0
n_cells = 20
sn_order = 8

sigma_t = 20.0  # 强吸收
sigma_s = 0.0
dx = (x_max - x_min) / n_cells
print(f"dx = {dx:.4f}, σ_tΔx = {sigma_t * dx:.4f}")

# 左半段有源，右半段零源
source = np.zeros(n_cells)
source[:n_cells//2] = 1.0

mesh, phi, psi = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                              sigma_t, sigma_s, source, tol=1e-8)

print(f"\n角通量最小值: {np.min(psi):.6e}")
print(f"角通量负值数量: {np.sum(psi < 0)} / {psi.size}")
print(f"标量通量最小值: {np.min(phi):.6e}")
print(f"标量通量负值数量: {np.sum(phi < 0)} / {phi.size}")

if np.min(phi) < -1e-10 or np.min(psi) < -1e-10:
    print("\n⚠ 检测到负通量！")
    neg_psi = np.where(psi < -1e-10)
    for n, i in zip(neg_psi[0][:15], neg_psi[1][:15]):
        mu = quad.mus[n]
        print(f"  角度{n:2d}(μ={mu:+.4f}), 边{i:2d}: ψ={psi[n,i]:.6e}")
else:
    print("\n✓ 未检测到显著负通量 (|ψ| < 1e-10)")

# 测试2: 纯散射介质 + 边界源（可能在内部产生负通量）
print("\n\n测试2: 纯散射介质 + 左边界入射（无内源）")
print("-" * 60)

sigma_t = 2.0
sigma_s = 2.0  # 纯散射，无吸收
sigma_a = sigma_t - sigma_s
print(f"σ_a = {sigma_a}, c = σ_s/σ_t = {sigma_s/sigma_t:.4f}")
print(f"σ_tΔx = {sigma_t * dx:.4f}")

source = np.zeros(n_cells)

# 创建一个带有入射边界条件的求解器
mesh2 = Mesh(x_min, x_max, n_cells)
quad2 = SNQuadrature(sn_order)
materials2 = [Material(sigma_t, sigma_s) for _ in range(n_cells)]
solver = SNSolver(mesh2, quad2, materials2, 'vacuum', 'vacuum')

# 设置左边界入射（正方向）
for n in range(sn_order):
    if quad2.mus[n] > 0:
        solver.psi[n, 0] = 1.0  # 入射通量

phi, psi = solver.solve(source, tol=1e-8)

print(f"\n角通量最小值: {np.min(psi):.6e}")
print(f"角通量负值数量: {np.sum(psi < -1e-10)} / {psi.size}")
print(f"标量通量最小值: {np.min(phi):.6e}")
print(f"标量通量负值数量: {np.sum(phi < -1e-10)} / {phi.size}")

if np.min(phi) < -1e-10 or np.min(psi) < -1e-10:
    print("\n⚠ 检测到负通量！")
else:
    print("\n✓ 未检测到显著负通量")

# 测试3: 逐步增加光学厚度，找到负通量临界值
print("\n\n测试3: 寻找负通量临界光学厚度")
print("-" * 60)

n_cells = 10
x_min, x_max = 0.0, 1.0
source = np.ones(n_cells)
sn_order = 8
quad3 = SNQuadrature(sn_order)

for sigma_t in [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    dx = (x_max - x_min) / n_cells
    tau = sigma_t * dx
    mesh3, phi3, psi3 = solve_sn_1d(x_min, x_max, n_cells, sn_order, 
                                     sigma_t, 0.0, source, tol=1e-8, max_iter=2000)
    
    min_psi = np.min(psi3)
    min_phi = np.min(phi3)
    has_neg = min_psi < -1e-10 or min_phi < -1e-10
    
    status = "⚠ 负通量" if has_neg else "✓ 正"
    print(f"  σ_tΔx = {tau:6.2f}: min(ψ)={min_psi:+.3e}, min(φ)={min_phi:+.3e} {status}")
    
    if has_neg:
        break

print("\n" + "=" * 60)
print("结论: 菱形差分方法在以下情况可能产生负通量:")
print("  1. 光学厚度 σ_tΔx > 2|μ| (系数为负)")
print("  2. 边界附近（入射通量为0）")
print("  3. 零源区域（只有散射源）")
print("  4. 小角度方向（μ接近0）")
print("=" * 60)
