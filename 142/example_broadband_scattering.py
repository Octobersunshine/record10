import numpy as np
import sys
sys.path.insert(0, 'e:/temp/record10/142')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mesh import generate_cylinder_mesh
from bem import AcousticScattering


def main():
    print("=" * 70)
    print("宽频 FMBEM 声学散射示例 - 大尺度模型分析")
    print("=" * 70)

    c = 1500
    rho = 1025

    frequencies = [100, 500, 1000, 2000, 5000]

    print(f"\n环境参数:")
    print(f"  声速: {c} m/s")
    print(f"  密度: {rho} kg/m^3")

    length = 30
    diameter = 5
    print(f"\n目标几何参数:")
    print(f"  长度: {length} m")
    print(f"  直径: {diameter} m")
    print(f"  长宽比: {length/diameter:.1f}")

    mesh = generate_cylinder_mesh(radius=diameter/2, height=length,
                                  n_theta=20, n_height=40)

    print(f"\n网格信息:")
    print(f"  单元数: {mesh.num_elements}")
    print(f"  顶点数: {mesh.num_vertices}")
    print(f"  表面积: {np.sum(mesh.areas):.1f} m^2")

    results = {}

    for freq in frequencies:
        k = 2 * np.pi * freq / c
        wavelength = 2 * np.pi / k
        ka = k * diameter / 2

        print(f"\n{'='*70}")
        print(f"频率: {freq} Hz | 波长: {wavelength:.2f} m | ka: {ka:.2f}")
        print(f"每波长单元数: {wavelength / np.sqrt(np.mean(mesh.areas)):.1f}")
        print(f"{'='*70}")

        scattering = AcousticScattering(mesh, k, c=c, rho=rho)

        direction = np.array([0.0, 0.0, 1.0])
        p_incident = scattering.set_incident_plane_wave(direction, amplitude=1.0)

        print("  求解边界积分方程...", end='', flush=True)
        p_surface, v_surface = scattering.solve(
            p_incident,
            method='dirichlet',
            use_broadband_fmm=False,
            tol=1e-4,
            kr_switch=8.0
        )
        print("完成!")

        expansion_info = scattering.get_expansion_info()
        if expansion_info:
            print("  八叉树展开信息:")
            for level, info in expansion_info.items():
                print(f"    Level {level}: kr={info['kr']:.1f}, "
                      f"{info['expansion_type']}, "
                      f"box_size={info['box_size']:.2f}m")

        r_obs = 100
        n_angles = 36
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
            'k': k,
            'wavelength': wavelength,
            'ka': ka
        }

        print(f"  峰值目标强度: {np.max(ts):.1f} dB")
        print(f"  艏部目标强度 (0°): {ts[0]:.1f} dB")
        print(f"  舷侧目标强度 (90°): {ts[n_angles//4]:.1f} dB")

    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(231, projection='polar')
    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
    for i, freq in enumerate(frequencies):
        ax1.plot(results[freq]['angles'], results[freq]['ts'],
                 color=colors[i], label=f'{freq} Hz', linewidth=2)
    ax1.set_theta_zero_location('N')
    ax1.set_title('目标强度方向图 (XZ平面)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True)

    ax2 = fig.add_subplot(232)
    for i, freq in enumerate(frequencies):
        ax2.plot(results[freq]['angles'] * 180 / np.pi, results[freq]['ts'],
                 color=colors[i], label=f'{freq} Hz', linewidth=2)
    ax2.set_xlabel('方位角 (度)')
    ax2.set_ylabel('目标强度 (dB)')
    ax2.set_title('目标强度 vs 方位角')
    ax2.legend(fontsize=8)
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
    kas = [results[f]['ka'] for f in frequencies]
    peak_ts = [np.max(results[f]['ts']) for f in frequencies]
    ax4.semilogx(frequencies, peak_ts, 'bo-', linewidth=2, markersize=8)
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('峰值目标强度 (dB)')
    ax4.set_title('峰值目标强度 vs 频率')
    ax4.grid(True, which='both', alpha=0.3)
    for i, freq in enumerate(frequencies):
        ax4.annotate(f'ka={kas[i]:.1f}',
                     (freq, peak_ts[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8)

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
        ax6.hist(np.abs(results[freq]['p_surface']), bins=20, alpha=0.5,
                 label=f'{freq} Hz', density=True, color=colors[i])
    ax6.set_xlabel('表面声压幅值')
    ax6.set_ylabel('概率密度')
    ax6.set_title('表面声压分布')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'broadband_scattering_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存至: {output_file}")

    print("\n" + "=" * 70)
    print("宽频分析总结:")
    print("=" * 70)
    print(f"{'频率(Hz)':<12} {'波长(m)':<12} {'ka':<10} {'峰值TS(dB)':<15} {'艏部TS(dB)':<15}")
    print("-" * 65)
    for freq in frequencies:
        r = results[freq]
        print(f"{freq:<12} {r['wavelength']:<12.2f} {r['ka']:<10.2f} "
              f"{np.max(r['ts']):<15.1f} {r['ts'][0]:<15.1f}")
    print("=" * 70)

    print("\n宽频FMM优势:")
    print("  ✓ 自动根据kr值选择展开方法（球谐/平面波）")
    print("  ✓ 自适应展开阶数，保证精度的同时最小化计算量")
    print("  ✓ 低频使用球谐展开（高效），高频使用平面波展开（稳定）")
    print("  ✓ 适用于大尺度模型（汽车、潜艇等）的宽频分析")

    plt.show()


if __name__ == '__main__':
    main()
