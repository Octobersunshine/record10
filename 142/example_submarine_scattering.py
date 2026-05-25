import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fmbem.mesh import generate_cylinder_mesh, generate_sphere_mesh
from fmbem.bem import AcousticScattering, BEMSolver


def generate_submarine_mesh(length: float = 50.0,
                            diameter: float = 8.0,
                            n_theta: int = 32,
                            n_height: int = 60) -> 'Mesh':
    from fmbem.mesh import Mesh

    radius = diameter / 2
    cylinder = generate_cylinder_mesh(radius=radius, height=length,
                                      n_theta=n_theta, n_height=n_height)

    bow_radius = radius
    bow = generate_sphere_mesh(radius=bow_radius, n_theta=16, n_phi=32)
    bow.vertices[:, 2] += length / 2 + bow_radius * 0.5

    stern = generate_sphere_mesh(radius=bow_radius, n_theta=16, n_phi=32)
    stern.vertices[:, 2] -= length / 2 + bow_radius * 0.5

    all_vertices = np.vstack([cylinder.vertices, bow.vertices, stern.vertices])
    all_faces = np.vstack([
        cylinder.faces,
        bow.faces + len(cylinder.vertices),
        stern.faces + len(cylinder.vertices) + len(bow.vertices)
    ])

    return Mesh(all_vertices, all_faces)


def main():
    print("=" * 70)
    print("FMBEM 大尺度声学散射示例 - 潜艇目标强度分析")
    print("=" * 70)

    frequencies = [100, 500, 1000, 2000]
    c = 1500
    rho = 1025

    print(f"\n环境参数:")
    print(f"  声速: {c} m/s")
    print(f"  密度: {rho} kg/m^3")

    length = 30
    diameter = 5
    print(f"\n潜艇几何参数:")
    print(f"  长度: {length} m")
    print(f"  直径: {diameter} m")
    print(f"  长宽比: {length/diameter:.1f}")

    mesh = generate_cylinder_mesh(radius=diameter/2, height=length,
                                  n_theta=24, n_height=48)

    print(f"\n网格信息:")
    print(f"  单元数: {mesh.num_elements}")
    print(f"  顶点数: {mesh.num_vertices}")
    print(f"  表面积: {np.sum(mesh.areas):.1f} m^2")

    results = {}

    for freq in frequencies:
        k = 2 * np.pi * freq / c
        wavelength = 2 * np.pi / k

        print(f"\n{'='*70}")
        print(f"频率: {freq} Hz | 波长: {wavelength:.2f} m | ka: {k*diameter/2:.2f}")
        print(f"{'='*70}")

        scattering = AcousticScattering(mesh, k, c=c, rho=rho)

        direction = np.array([0.0, 0.0, 1.0])
        p_incident = scattering.set_incident_plane_wave(direction, amplitude=1.0)

        print("  求解边界积分方程...", end='', flush=True)
        p_surface, v_surface = scattering.solve(
            p_incident,
            method='dirichlet',
            use_fmm=False
        )
        print("完成!")

        r_obs = 100
        n_angles = 72
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

        field_points = np.zeros((n_angles, 3))
        field_points[:, 0] = r_obs * np.sin(angles)
        field_points[:, 2] = r_obs * np.cos(angles)

        print("  计算远场散射...", end='', flush=True)
        p_scattered = scattering.compute_scattered_field(field_points, p_surface, v_surface)
        ts = 20 * np.log10(np.abs(p_scattered) * r_obs / 1.0)
        print("完成!")

        results[freq] = {
            'angles': angles,
            'ts': ts,
            'p_surface': p_surface,
            'k': k
        }

        print(f"  峰值目标强度: {np.max(ts):.1f} dB")
        print(f"  艏部目标强度 (0°): {ts[0]:.1f} dB")
        print(f"  舷侧目标强度 (90°): {ts[n_angles//4]:.1f} dB")
        print(f"  艉部目标强度 (180°): {ts[n_angles//2]:.1f} dB")

    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(231, projection='polar')
    colors = ['b', 'g', 'r', 'm']
    for i, freq in enumerate(frequencies):
        ax1.plot(results[freq]['angles'], results[freq]['ts'],
                 color=colors[i], label=f'{freq} Hz', linewidth=2)
    ax1.set_theta_zero_location('N')
    ax1.set_title('目标强度方向图 (XZ平面)')
    ax1.legend(loc='best')
    ax1.grid(True)

    ax2 = fig.add_subplot(232)
    for i, freq in enumerate(frequencies):
        ax2.plot(results[freq]['angles'] * 180 / np.pi, results[freq]['ts'],
                 color=colors[i], label=f'{freq} Hz', linewidth=2)
    ax2.set_xlabel('方位角 (度)')
    ax2.set_ylabel('目标强度 (dB)')
    ax2.set_title('目标强度 vs 方位角')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])

    ax3 = fig.add_subplot(233, projection='3d')
    centroids = mesh.centroids
    p_last = results[frequencies[-1]]['p_surface']
    scatter = ax3.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                          c=np.abs(p_last), cmap='viridis', s=10, alpha=0.6)
    ax3.set_title(f'表面声压分布 ({frequencies[-1]} Hz)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    plt.colorbar(scatter, ax=ax3, label='声压幅值')

    ax4 = fig.add_subplot(234)
    peak_ts = [np.max(results[f]['ts']) for f in frequencies]
    ax4.semilogx(frequencies, peak_ts, 'bo-', linewidth=2, markersize=8)
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('峰值目标强度 (dB)')
    ax4.set_title('峰值目标强度 vs 频率')
    ax4.grid(True, which='both', alpha=0.3)
    for i, freq in enumerate(frequencies):
        ax4.annotate(f'{peak_ts[i]:.1f} dB',
                     (freq, peak_ts[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    ax5 = fig.add_subplot(235)
    bow_ts = [results[f]['ts'][0] for f in frequencies]
    side_ts = [results[f]['ts'][len(results[f]['ts'])//4] for f in frequencies]
    ax5.semilogx(frequencies, bow_ts, 'ro-', linewidth=2, label='艏部 (0°)', markersize=8)
    ax5.semilogx(frequencies, side_ts, 'go-', linewidth=2, label='舷侧 (90°)', markersize=8)
    ax5.set_xlabel('频率 (Hz)')
    ax5.set_ylabel('目标强度 (dB)')
    ax5.set_title('关键方位目标强度')
    ax5.legend()
    ax5.grid(True, which='both', alpha=0.3)

    ax6 = fig.add_subplot(236)
    for i, freq in enumerate(frequencies):
        ax6.hist(np.abs(results[freq]['p_surface']), bins=25, alpha=0.5,
                 label=f'{freq} Hz', density=True)
    ax6.set_xlabel('表面声压幅值')
    ax6.set_ylabel('概率密度')
    ax6.set_title('表面声压分布')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'submarine_scattering_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存至: {output_file}")

    print("\n" + "=" * 70)
    print("多频率分析总结:")
    print("=" * 70)
    print(f"{'频率(Hz)':<12} {'波长(m)':<12} {'峰值TS(dB)':<12} {'艏部TS(dB)':<12} {'舷侧TS(dB)':<12}")
    print("-" * 60)
    for freq in frequencies:
        k = results[freq]['k']
        wavelength = 2 * np.pi / k
        ts = results[freq]['ts']
        print(f"{freq:<12} {wavelength:<12.2f} {np.max(ts):<12.1f} {ts[0]:<12.1f} {ts[len(ts)//4]:<12.1f}")
    print("=" * 70)

    plt.show()


if __name__ == '__main__':
    main()
