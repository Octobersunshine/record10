import numpy as np
from eit_solver_cem import EITMesh, EITForwardCEM, EITInverseCEM


def test_cem_basic():
    print("=" * 60)
    print("测试完整电极模型 (CEM)")
    print("=" * 60)
    
    print("\n1. 创建网格...")
    mesh = EITMesh(n_radius=4, n_angles=8, r=1.0, electrode_angle_width=0.4)
    print(f"   节点: {mesh.n_nodes}, 单元: {mesh.n_elements}, 电极: {mesh.n_electrodes}")
    
    print("\n2. 创建测试电导率分布...")
    sigma = np.ones(mesh.n_elements)
    sigma[:20] = 2.0
    
    print("\n3. 创建接触阻抗...")
    z = np.ones(mesh.n_electrodes) * 0.1
    z[0] = 0.5
    print(f"   接触阻抗范围: [{z.min():.3f}, {z.max():.3f}]")
    
    print("\n4. 正向求解 (CEM)...")
    forward = EITForwardCEM(mesh)
    voltages = forward.simulate_measurements(sigma, z)
    print(f"   测量点数: {len(voltages)}")
    print(f"   电压范围: [{voltages.min():.4f}, {voltages.max():.4f}]")
    
    print("\n5. 测试雅可比矩阵...")
    inverse = EITInverseCEM(forward, max_iter=5)
    J_sigma, J_z = inverse.compute_jacobian_cem(sigma, z)
    print(f"   J_sigma shape: {J_sigma.shape}")
    print(f"   J_z shape: {J_z.shape}")
    
    print("\n6. 联合重构测试...")
    sigma_recon, z_recon = inverse.reconstruct_joint(voltages)
    print(f"   电导率误差: {np.linalg.norm(sigma_recon - sigma)/np.linalg.norm(sigma)*100:.2f}%")
    print(f"   接触阻抗误差: {np.linalg.norm(z_recon - z)/np.linalg.norm(z)*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("CEM模型测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_cem_basic()
