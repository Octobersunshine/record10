import numpy as np
import matplotlib.pyplot as plt
from eit_solver_cem import EITMesh, EITForwardCEM, EITInverseCEM, visualize_conductivity


def create_phantom_conductivity(mesh, center=(0.3, 0.2), radius=0.25, conductivity=3.0):
    sigma = np.ones(mesh.n_elements)
    
    elem_centers = np.zeros((mesh.n_elements, 2))
    for e_idx in range(mesh.n_elements):
        elem_nodes = mesh.nodes[mesh.elements[e_idx]]
        elem_centers[e_idx] = np.mean(elem_nodes, axis=0)
    
    cx, cy = center
    dist = np.sqrt((elem_centers[:, 0] - cx)**2 + (elem_centers[:, 1] - cy)**2)
    sigma[dist < radius] = conductivity
    
    return sigma


def create_mismatched_contact_impedance(n_elec, mismatch_pattern='one_side', base_z=0.1, factor=5.0):
    z = np.ones(n_elec) * base_z
    
    if mismatch_pattern == 'one_side':
        for i in range(n_elec // 2):
            z[i] *= factor
    elif mismatch_pattern == 'opposite':
        z[0] *= factor
        z[n_elec // 2] *= factor
    elif mismatch_pattern == 'alternating':
        for i in range(0, n_elec, 2):
            z[i] *= factor
    elif mismatch_pattern == 'single':
        z[0] *= factor
    
    return z


def run_demo():
    print("=" * 70)
    print("  电极接触阻抗建模对比演示")
    print("  展示：接触阻抗不匹配 -> 图像偏心 -> CEM模型修复")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("步骤 1: 生成有限元网格")
    print("=" * 70)
    mesh = EITMesh(n_radius=6, n_angles=16, r=1.0, electrode_angle_width=0.25)
    print(f"  节点数: {mesh.n_nodes}")
    print(f"  单元数: {mesh.n_elements}")
    print(f"  电极数: {mesh.n_electrodes}")
    
    print("\n" + "=" * 70)
    print("步骤 2: 创建真实电导率分布 (中心偏移的圆形异常)")
    print("=" * 70)
    sigma_true = create_phantom_conductivity(mesh, center=(0.3, 0.0), radius=0.25, conductivity=3.0)
    
    print("\n" + "=" * 70)
    print("步骤 3: 创建不匹配的接触阻抗 (左侧电极阻抗升高)")
    print("=" * 70)
    z_true = create_mismatched_contact_impedance(mesh.n_electrodes, mismatch_pattern='one_side', 
                                                 base_z=0.1, factor=4.0)
    print(f"  基准接触阻抗: 0.1 Ω·m²")
    print(f"  不匹配因子: 4.0x")
    print(f"  接触阻抗范围: [{z_true.min():.3f}, {z_true.max():.3f}]")
    print(f"  接触阻抗均值: {z_true.mean():.3f}")
    
    print("\n" + "=" * 70)
    print("步骤 4: 使用CEM模型模拟真实测量数据 (包含接触阻抗效应)")
    print("=" * 70)
    forward = EITForwardCEM(mesh)
    measured_voltages = forward.simulate_measurements(sigma_true, z_true)
    print(f"  测量数据点数: {len(measured_voltages)}")
    
    noise_level = 0.005
    measured_voltages_noisy = measured_voltages * (1 + noise_level * np.random.randn(len(measured_voltages)))
    print(f"  添加 {noise_level*100}% 高斯噪声")
    
    print("\n" + "=" * 70)
    print("步骤 5: 方法A - 忽略接触阻抗 (简单点电极模型)")
    print("  预期结果: 重构图像出现偏心/伪影")
    print("=" * 70)
    
    z_ignore = np.ones(mesh.n_electrodes) * 0.001
    inverse_simple = EITInverseCEM(forward, max_iter=15, reg_sigma=5e-3, reg_z=1e-2, tol=1e-4)
    
    print("  正在重构 (假设接触阻抗可忽略)...")
    sigma_ignore_z = inverse_simple.reconstruct_with_known_z(measured_voltages_noisy, z_ignore)
    
    error_ignore = np.linalg.norm(sigma_ignore_z - sigma_true) / np.linalg.norm(sigma_true)
    print(f"  相对重构误差: {error_ignore*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("步骤 6: 方法B - 使用CEM模型同时重构电导率和接触阻抗")
    print("  预期结果: 图像偏心被修正，接触阻抗被估计")
    print("=" * 70)
    
    inverse_cem = EITInverseCEM(forward, max_iter=20, reg_sigma=5e-3, reg_z=5e-2, tol=1e-4)
    
    print("  正在联合重构 (电导率 + 接触阻抗)...")
    sigma_joint, z_est = inverse_cem.reconstruct_joint(measured_voltages_noisy)
    
    error_joint = np.linalg.norm(sigma_joint - sigma_true) / np.linalg.norm(sigma_true)
    z_error = np.linalg.norm(z_est - z_true) / np.linalg.norm(z_true)
    print(f"  电导率相对误差: {error_joint*100:.2f}%")
    print(f"  接触阻抗相对误差: {z_error*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("步骤 7: 结果对比可视化")
    print("=" * 70)
    
    fig = plt.figure(figsize=(20, 12))
    
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1, _ = visualize_conductivity(mesh, sigma_true, "真实电导率分布", ax1)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2, _ = visualize_conductivity(mesh, sigma_ignore_z, "忽略接触阻抗 (偏心明显)", ax2)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3, _ = visualize_conductivity(mesh, sigma_joint, "CEM联合重构 (修正后)", ax3)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    ax4 = fig.add_subplot(gs[0, 3])
    error_map = np.abs(sigma_joint - sigma_true)
    im4, _ = visualize_conductivity(mesh, error_map, "绝对误差分布", ax4)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = fig.add_subplot(gs[1, :])
    electrode_indices = np.arange(mesh.n_electrodes)
    angles = 2 * np.pi * electrode_indices / mesh.n_electrodes
    
    ax5.bar(angles, z_true, width=0.3, alpha=0.7, label='真实接触阻抗', color='blue')
    ax5.bar(angles + 0.15, z_est, width=0.3, alpha=0.7, label='估计接触阻抗', color='red')
    ax5.set_xlabel('电极角度 (弧度)')
    ax5.set_ylabel('接触阻抗 (Ω·m²)')
    ax5.set_title('接触阻抗对比', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, :2])
    error_bar_data = [error_ignore * 100, error_joint * 100]
    labels = ['忽略接触阻抗', 'CEM联合重构']
    bars = ax6.bar(labels, error_bar_data, color=['orange', 'green'], alpha=0.7)
    ax6.set_ylabel('相对重构误差 (%)')
    ax6.set_title('重构误差对比', fontsize=14)
    ax6.set_ylim(0, max(error_bar_data) * 1.3)
    for bar, err in zip(bars, error_bar_data):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{err:.1f}%', ha='center', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')
    
    ax7 = fig.add_subplot(gs[2, 2:])
    center_true = np.array([0.3, 0.0])
    elem_centers = np.zeros((mesh.n_elements, 2))
    for e_idx in range(mesh.n_elements):
        elem_nodes = mesh.nodes[mesh.elements[e_idx]]
        elem_centers[e_idx] = np.mean(elem_nodes, axis=0)
    
    weighted_center_ignore = np.average(elem_centers, weights=sigma_ignore_z, axis=0)
    weighted_center_joint = np.average(elem_centers, weights=sigma_joint, axis=0)
    
    ax7.scatter(center_true[0], center_true[1], s=300, c='blue', marker='*', 
                label=f'真实中心 ({center_true[0]:.2f}, {center_true[1]:.2f})', zorder=5)
    ax7.scatter(weighted_center_ignore[0], weighted_center_ignore[1], s=200, c='orange', marker='x', 
                label=f'忽略阻抗 ({weighted_center_ignore[0]:.2f}, {weighted_center_ignore[1]:.2f})', 
                linewidth=3, zorder=5)
    ax7.scatter(weighted_center_joint[0], weighted_center_joint[1], s=200, c='green', marker='+', 
                label=f'CEM重构 ({weighted_center_joint[0]:.2f}, {weighted_center_joint[1]:.2f})', 
                linewidth=3, zorder=5)
    
    circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    ax7.add_artist(circle)
    
    ax7.set_xlim(-1.1, 1.1)
    ax7.set_ylim(-1.1, 1.1)
    ax7.set_aspect('equal')
    ax7.set_title('异常区域质心对比 (图像偏心可视化)', fontsize=14)
    ax7.legend(loc='lower right', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('电极接触阻抗建模对EIT重构的影响', fontsize=18, y=0.995)
    plt.savefig('cem_contact_impedance_comparison.png', dpi=150, bbox_inches='tight')
    print("  对比图已保存: cem_contact_impedance_comparison.png")
    
    print("\n" + "=" * 70)
    print("结果分析:")
    print("=" * 70)
    print(f"  忽略接触阻抗的误差: {error_ignore*100:.2f}%")
    print(f"  CEM联合重构的误差: {error_joint*100:.2f}%")
    print(f"  误差改善: {(error_ignore - error_joint)/error_ignore*100:.1f}%")
    
    bias_ignore = np.linalg.norm(weighted_center_ignore - center_true)
    bias_joint = np.linalg.norm(weighted_center_joint - center_true)
    print(f"\n  忽略阻抗的质心偏移: {bias_ignore:.3f}")
    print(f"  CEM重构的质心偏移: {bias_joint:.3f}")
    print(f"  偏心改善: {(bias_ignore - bias_joint)/bias_ignore*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("关键结论:")
    print("=" * 70)
    print("  1. 电极接触阻抗不匹配会导致重构图像出现明显偏心")
    print("  2. 简单点电极模型无法处理接触阻抗效应")
    print("  3. 完整电极模型(CEM)通过同时重构电导率和接触阻抗")
    print("     可以有效修正图像偏心，显著提高重构精度")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    run_demo()
