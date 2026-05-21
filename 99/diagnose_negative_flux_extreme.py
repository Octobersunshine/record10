import numpy as np
from sn_solver import SNSolver, Mesh, SNQuadrature, Material


def check_negative_flux(phi, psi, name=""):
    """检查是否存在负通量"""
    print(f"\n  --- {name} ---")
    
    neg_phi_mask = phi < -1e-10
    if np.any(neg_phi_mask):
        n_neg = np.sum(neg_phi_mask)
        min_val = np.min(phi)
        print(f"  ⚠ 标量通量: 发现 {n_neg} 个负值，最小值 = {min_val:.6e}")
        print(f"    位置: {np.where(neg_phi_mask)[0]}")
        has_neg_phi = True
    else:
        print(f"  ✓ 标量通量: 无负值 (min = {np.min(phi):.6e})")
        has_neg_phi = False
    
    neg_psi_mask = psi < -1e-10
    if np.any(neg_psi_mask):
        n_neg = np.sum(neg_psi_mask)
        min_val = np.min(psi)
        print(f"  ⚠ 角通量: 发现 {n_neg} 个负值，最小值 = {min_val:.6e}")
        neg_positions = np.where(neg_psi_mask)
        print(f"    角度索引: {np.unique(neg_positions[0])}")
        has_neg_psi = True
    else:
        print(f"  ✓ 角通量: 无负值 (min = {np.min(psi):.6e})")
        has_neg_psi = False
    
    return has_neg_phi or has_neg_psi


def test_high_opacity_coarse_mesh():
    """测试高光学厚度 + 粗网格 - 菱形差分最容易出问题的场景"""
    print("=" * 60)
    print("极端测试: 高光学厚度 + 粗网格 (σ_t * Δx >> 1)")
    print("=" * 60)
    
    x_min, x_max = 0.0, 1.0
    n_cells = 5
    sn_order = 8
    
    # 高截面，粗网格 → σ_t * Δx = 10 * 0.2 = 2
    sigma_t_list = [5.0, 10.0, 20.0, 50.0]
    sigma_s_ratio = 0.9
    source = np.ones(n_cells)
    
    for sigma_t in sigma_t_list:
        sigma_s = sigma_s_ratio * sigma_t
        mesh = Mesh(x_min, x_max, n_cells)
        quad = SNQuadrature(sn_order)
        materials = [Material(sigma_t, sigma_s) for _ in range(n_cells)]
        
        solver = SNSolver(mesh, quad, materials, 'vacuum', 'vacuum')
        phi, psi = solver.solve(source, max_iter=2000, tol=1e-8)
        
        tau = sigma_t * mesh.dx
        check_negative_flux(phi, psi, f"σ_t={sigma_t:.1f}, τ={tau:.2f}")


def test_small_angle():
    """测试小角度方向（μ接近0）最容易出现负通量"""
    print("\n" + "=" * 60)
    print("极端测试: 检查小角度方向的角通量")
    print("=" * 60)
    
    x_min, x_max = 0.0, 1.0
    n_cells = 5
    sn_order = 8
    
    sigma_t = 20.0
    sigma_s = 0.9 * sigma_t
    source = np.ones(n_cells)
    
    mesh = Mesh(x_min, x_max, n_cells)
    quad = SNQuadrature(sn_order)
    materials = [Material(sigma_t, sigma_s) for _ in range(n_cells)]
    
    solver = SNSolver(mesh, quad, materials, 'vacuum', 'vacuum')
    phi, psi = solver.solve(source, max_iter=2000, tol=1e-8)
    
    print(f"\n  各角度方向的最小角通量:")
    for n in range(len(quad.mus)):
        mu = quad.mus[n]
        min_psi = np.min(psi[n, :])
        if min_psi < -1e-10:
            print(f"    μ={mu:+.6f}: min={min_psi:.6e} ⚠")
        else:
            print(f"    μ={mu:+.6f}: min={min_psi:.6e} ✓")


def test_large_sn_small_mesh():
    """测试高SN阶数 + 非常小的网格数"""
    print("\n" + "=" * 60)
    print("极端测试: 高SN阶数 + 极少网格")
    print("=" * 60)
    
    x_min, x_max = 0.0, 2.0
    sn_order_list = [4, 8, 16]
    n_cells_list = [2, 3, 4]
    
    sigma_t = 10.0
    sigma_s = 0.95 * sigma_t
    
    for sn_order in sn_order_list:
        for n_cells in n_cells_list:
            source = np.ones(n_cells)
            mesh = Mesh(x_min, x_max, n_cells)
            quad = SNQuadrature(sn_order)
            materials = [Material(sigma_t, sigma_s) for _ in range(n_cells)]
            
            solver = SNSolver(mesh, quad, materials, 'vacuum', 'vacuum')
            phi, psi = solver.solve(source, max_iter=2000, tol=1e-8)
            
            tau = sigma_t * mesh.dx
            check_negative_flux(phi, psi, f"SN{sn_order}, 网格={n_cells}, τ={tau:.2f}")


def test_interface_problem():
    """测试材料界面附近 - 截面突变导致的振荡"""
    print("\n" + "=" * 60)
    print("极端测试: 材料界面 (截面突变)")
    print("=" * 60)
    
    x_min, x_max = 0.0, 2.0
    n_cells = 10
    sn_order = 8
    
    # 左半部分高截面，右半部分低截面
    materials = []
    for i in range(n_cells):
        if i < n_cells // 2:
            materials.append(Material(sigma_t=50.0, sigma_s=49.0))
        else:
            materials.append(Material(sigma_t=1.0, sigma_s=0.5))
    
    source = np.ones(n_cells)
    source[n_cells // 4] = 10.0  # 强源在高截面区
    
    mesh = Mesh(x_min, x_max, n_cells)
    quad = SNQuadrature(sn_order)
    
    solver = SNSolver(mesh, quad, materials, 'vacuum', 'vacuum')
    phi, psi = solver.solve(source, max_iter=2000, tol=1e-8)
    
    check_negative_flux(phi, psi, "材料界面问题")
    
    print(f"\n  界面附近通量:")
    for i in range(n_cells//2 - 2, n_cells//2 + 3):
        print(f"    细胞 {i} (x={mesh.x_centers[i]:.2f}): φ={phi[i]:.6f}")


def main():
    print("=" * 60)
    print("极端条件下负通量诊断")
    print("=" * 60)
    
    test_high_opacity_coarse_mesh()
    test_small_angle()
    test_large_sn_small_mesh()
    test_interface_problem()
    
    print("\n" + "=" * 60)
    print("极端诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
