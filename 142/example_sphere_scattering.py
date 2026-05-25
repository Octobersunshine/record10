import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fmbem.mesh import generate_sphere_mesh
from fmbem.bem import AcousticScattering


def main():
    print("=" * 60)
    print("FMBEM 声学散射示例 - 球体散射")
    print("=" * 60)

    frequency = 1000
    c = 343
    k = 2 * np.pi * frequency / c
    wavelength = 2 * np.pi / k

    print(f"\n频率: {frequency} Hz")
    print(f"波数: {k:.2f} m^-1")
    print(f"波长: {wavelength:.3f} m")

    radius = 1.0
    n_theta = 12
    n_phi = 24
    mesh = generate_sphere_mesh(radius=radius, n_theta=n_theta, n_phi=n_phi)

    print(f"\n网格信息:")
    print(f"  单元数: {mesh.num_elements}")
    print(f"  顶点数: {mesh.num_vertices}")
    print(f"  平均单元尺寸: {np.sqrt(np.mean(mesh.areas)):.3f} m")
    print(f"  每波长单元数: {wavelength / np.sqrt(np.mean(mesh.areas)):.1f}")

    scattering = AcousticScattering(mesh, k, c=c)

    direction = np.array([1.0, 0.0, 0.0])
    p_incident = scattering.set_incident_plane_wave(direction, amplitude=1.0)

    print(f"\n入射波: 平面波, 方向 = {direction}")

    print("\n求解边界积分方程...")
    p_surface, v_surface = scattering.solve(
        p_incident,
        method='dirichlet',
        use_fmm=False
    )

    print("求解完成!")
    print(f"  表面声压幅值范围: [{np.min(np.abs(p_surface)):.3f}, {np.max(np.abs(p_surface)):.3f}]")

    r_obs = 10 * radius
    n_theta_obs = 36
    n_phi_obs = 72

    theta = np.linspace(0, np.pi, n_theta_obs)
    phi = np.linspace(0, 2 * np.pi, n_phi_obs, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi)

    X = r_obs * np.sin(THETA) * np.cos(PHI)
    Y = r_obs * np.sin(THETA) * np.sin(PHI)
    Z = r_obs * np.cos(THETA)

    field_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    print("\n计算远场散射...")
    p_scattered = scattering.compute_scattered_field(field_points, p_surface, v_surface)
    p_scattered = p_scattered.reshape(THETA.shape)

    print("远场计算完成!")

    ts = 20 * np.log10(np.abs(p_scattered) * r_obs / 1.0)

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(231, projection='3d')
    centroids = mesh.centroids
    ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                c=np.abs(p_surface), cmap='viridis', s=20)
    ax1.set_title('表面声压幅值')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')

    ax2 = fig.add_subplot(232)
    phi_idx = n_phi_obs // 2
    ax2.plot(theta * 180 / np.pi, ts[:, phi_idx])
    ax2.set_xlabel('极角 (度)')
    ax2.set_ylabel('目标强度 (dB)')
    ax2.set_title('XY平面内目标强度方向图')
    ax2.grid(True)

    ax3 = fig.add_subplot(233, projection='polar')
    theta_idx = n_theta_obs // 2
    ax3.plot(phi, ts[theta_idx, :])
    ax3.set_theta_zero_location('E')
    ax3.set_title('XZ平面内目标强度方向图')

    ax4 = fig.add_subplot(234, projection='3d')
    R = np.abs(p_scattered)
    ax4.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(R / R.max()),
                     alpha=0.8)
    ax4.set_title('远场声压分布')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')

    ax5 = fig.add_subplot(235)
    im = ax5.pcolormesh(PHI * 180 / np.pi, THETA * 180 / np.pi, ts.T,
                        cmap='viridis', shading='auto')
    ax5.set_xlabel('方位角 (度)')
    ax5.set_ylabel('极角 (度)')
    ax5.set_title('目标强度伪彩图 (dB)')
    plt.colorbar(im, ax=ax5)

    ax6 = fig.add_subplot(236)
    ax6.hist(np.abs(p_surface), bins=20, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('表面声压幅值')
    ax6.set_ylabel('单元数')
    ax6.set_title('表面声压分布直方图')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'sphere_scattering_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存至: {output_file}")

    print("\n" + "=" * 60)
    print("统计信息:")
    print(f"  最大目标强度: {np.max(ts):.2f} dB")
    print(f"  最小目标强度: {np.min(ts):.2f} dB")
    print(f"  平均目标强度: {np.mean(ts):.2f} dB")
    print("=" * 60)

    plt.show()


if __name__ == '__main__':
    main()
