import numpy as np
import matplotlib.pyplot as plt
from eit_solver import EITMesh, EITForward, EITInverse, visualize_conductivity


def create_phantom_conductivity(mesh, phantom_type='circle'):
    sigma = np.ones(mesh.n_elements)
    
    elem_centers = np.zeros((mesh.n_elements, 2))
    for e_idx in range(mesh.n_elements):
        elem_nodes = mesh.nodes[mesh.elements[e_idx]]
        elem_centers[e_idx] = np.mean(elem_nodes, axis=0)
    
    if phantom_type == 'circle':
        center_x, center_y = 0.3, 0.2
        radius = 0.25
        dist = np.sqrt((elem_centers[:, 0] - center_x)**2 + (elem_centers[:, 1] - center_y)**2)
        sigma[dist < radius] = 3.0
        
    elif phantom_type == 'two_circles':
        center1_x, center1_y = 0.4, 0.0
        center2_x, center2_y = -0.3, 0.3
        radius = 0.2
        dist1 = np.sqrt((elem_centers[:, 0] - center1_x)**2 + (elem_centers[:, 1] - center1_y)**2)
        dist2 = np.sqrt((elem_centers[:, 0] - center2_x)**2 + (elem_centers[:, 1] - center2_y)**2)
        sigma[dist1 < radius] = 4.0
        sigma[dist2 < radius] = 0.3
        
    elif phantom_type == 'annulus':
        inner_r = 0.2
        outer_r = 0.5
        dist = np.sqrt(elem_centers[:, 0]**2 + elem_centers[:, 1]**2)
        sigma[(dist > inner_r) & (dist < outer_r)] = 2.0
    
    return sigma


def main():
    print("=" * 60)
    print("电阻抗断层成像 (EIT) 演示")
    print("=" * 60)
    
    print("\n1. 生成有限元网格...")
    mesh = EITMesh(n_radius=6, n_angles=12, r=1.0)
    print(f"   节点数: {mesh.n_nodes}")
    print(f"   单元数: {mesh.n_elements}")
    print(f"   电极数: {mesh.n_electrodes}")
    
    print("\n2. 创建电导率分布模型...")
    sigma_true = create_phantom_conductivity(mesh, phantom_type='two_circles')
    
    fig1, _ = visualize_conductivity(mesh, sigma_true, "真实电导率分布")
    plt.savefig('true_conductivity.png', dpi=150, bbox_inches='tight')
    print("   真实电导率分布图已保存: true_conductivity.png")
    
    print("\n3. 正向求解 - 模拟边界电压测量...")
    forward_solver = EITForward(mesh)
    measured_voltages = forward_solver.simulate_measurements(sigma_true)
    print(f"   测量数据点数: {len(measured_voltages)}")
    
    noise_level = 0.01
    measured_voltages_noisy = measured_voltages * (1 + noise_level * np.random.randn(len(measured_voltages)))
    print(f"   添加了 {noise_level*100}% 的高斯噪声")
    
    print("\n4. 逆问题求解 - 重构电导率分布...")
    inverse_solver = EITInverse(forward_solver, max_iter=15, reg_param=1e-2, tol=1e-4)
    sigma_recon = inverse_solver.reconstruct(measured_voltages_noisy)
    
    fig2, _ = visualize_conductivity(mesh, sigma_recon, "重构电导率分布")
    plt.savefig('reconstructed_conductivity.png', dpi=150, bbox_inches='tight')
    print("   重构电导率分布图已保存: reconstructed_conductivity.png")
    
    print("\n5. 误差分析...")
    rel_error = np.linalg.norm(sigma_recon - sigma_true) / np.linalg.norm(sigma_true)
    print(f"   相对重构误差: {rel_error*100:.2f}%")
    
    fig3, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    plt.sca(axes[0])
    elem_centers = np.zeros((mesh.n_elements, 2))
    for e_idx in range(mesh.n_elements):
        elem_nodes = mesh.nodes[mesh.elements[e_idx]]
        elem_centers[e_idx] = np.mean(elem_nodes, axis=0)
    
    xi = np.linspace(-1.1, 1.1, 100)
    yi = np.linspace(-1.1, 1.1, 100)
    XI, YI = np.meshgrid(xi, yi)
    
    from scipy.interpolate import griddata
    zi_true = griddata(elem_centers, sigma_true, (XI, YI), method='cubic')
    zi_recon = griddata(elem_centers, sigma_recon, (XI, YI), method='cubic')
    
    mask = XI**2 + YI**2 > 1.0
    zi_true[mask] = np.nan
    zi_recon[mask] = np.nan
    
    im1 = axes[0].pcolormesh(XI, YI, zi_true, cmap='viridis', shading='auto')
    axes[0].set_aspect('equal')
    axes[0].set_title('真实分布', fontsize=14)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].pcolormesh(XI, YI, zi_recon, cmap='viridis', shading='auto')
    axes[1].set_aspect('equal')
    axes[1].set_title('重构分布', fontsize=14)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    from matplotlib.patches import Circle
    for ax in axes:
        circle = Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        elec_x = mesh.nodes[mesh.electrodes, 0]
        elec_y = mesh.nodes[mesh.electrodes, 1]
        ax.scatter(elec_x, elec_y, c='red', s=80, marker='o', edgecolors='white', zorder=5)
    
    plt.suptitle(f'EIT重构结果 (相对误差: {rel_error*100:.2f}%)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('eit_comparison.png', dpi=150, bbox_inches='tight')
    print("   对比图已保存: eit_comparison.png")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    main()
