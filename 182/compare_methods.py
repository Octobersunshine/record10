import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import j1
from scipy import linalg

from photonic_crystal_fem import PhotonicCrystalFEM, plot_band_structure, print_gap_analysis
from photonic_crystal_pwe import PhotonicCrystal2D

rcParams['font.sans-serif'] = ['SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False


def run_pwe_calculation(params, k_path, num_bands, num_planes_list):
    results = {}
    for n_planes in num_planes_list:
        print(f"\n  PWE: 平面波数量 = {n_planes}")
        pc = PhotonicCrystal2D(**params)
        bands_TM, bands_TE = pc.compute_band_structure(
            k_path, mode='both', num_bands=num_bands, num_planes=n_planes
        )
        results[n_planes] = {
            'TM': bands_TM,
            'TE': bands_TE,
            'n_planes': n_planes
        }
    return results


def run_fem_calculation(params, k_path, num_bands, resolutions):
    results = {}
    for res in resolutions:
        print(f"\n  FEM: 网格分辨率 = {res}")
        pc = PhotonicCrystalFEM(**params)
        pc.generate_mesh(resolution=res)
        bands_TM, bands_TE = pc.compute_band_structure(
            k_path, mode='both', num_bands=num_bands
        )
        results[res] = {
            'TM': bands_TM,
            'TE': bands_TE,
            'n_nodes': len(pc.nodes),
            'n_elements': len(pc.elements),
            'resolution': res
        }
    return results


def plot_convergence_comparison(pwe_results, fem_results, k_point_idx=0, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pwe_keys = sorted(pwe_results.keys())
    fem_keys = sorted(fem_results.keys())

    if len(pwe_keys) >= 2:
        ref_key = pwe_keys[-1]
        ref_TM = pwe_results[ref_key]['TM'][k_point_idx, :]
        ref_TE = pwe_results[ref_key]['TE'][k_point_idx, :]

        errors_TM = []
        errors_TE = []
        n_list = []

        for key in pwe_keys:
            err_TM = np.mean(np.abs(pwe_results[key]['TM'][k_point_idx, :] - ref_TM) / np.abs(ref_TM) * 100)
            err_TE = np.mean(np.abs(pwe_results[key]['TE'][k_point_idx, :] - ref_TE) / np.abs(ref_TE) * 100)
            errors_TM.append(err_TM)
            errors_TE.append(err_TE)
            n_list.append(pwe_results[key]['n_planes'])

        axes[0].loglog(n_list, errors_TM, 'bo-', linewidth=2, markersize=8, label='TM 模式')
        axes[0].loglog(n_list, errors_TE, 'rs--', linewidth=2, markersize=8, label='TE 模式')
        axes[0].set_xlabel('平面波数量', fontsize=14)
        axes[0].set_ylabel('相对误差 (%)', fontsize=14)
        axes[0].set_title('PWE 收敛性分析', fontsize=16)
        axes[0].grid(True, alpha=0.3, which='both')
        axes[0].legend(fontsize=12)

    if len(fem_keys) >= 2:
        ref_key = fem_keys[-1]
        ref_TM = fem_results[ref_key]['TM'][k_point_idx, :]
        ref_TE = fem_results[ref_key]['TE'][k_point_idx, :]

        errors_TM = []
        errors_TE = []
        n_list = []

        for key in fem_keys:
            err_TM = np.mean(np.abs(fem_results[key]['TM'][k_point_idx, :] - ref_TM) / np.abs(ref_TM) * 100)
            err_TE = np.mean(np.abs(fem_results[key]['TE'][k_point_idx, :] - ref_TE) / np.abs(ref_TE) * 100)
            errors_TM.append(err_TM)
            errors_TE.append(err_TE)
            n_list.append(fem_results[key]['n_nodes'])

        axes[1].loglog(n_list, errors_TM, 'bo-', linewidth=2, markersize=8, label='TM 模式')
        axes[1].loglog(n_list, errors_TE, 'rs--', linewidth=2, markersize=8, label='TE 模式')
        axes[1].set_xlabel('节点数量', fontsize=14)
        axes[1].set_ylabel('相对误差 (%)', fontsize=14)
        axes[1].set_title('FEM 收敛性分析', fontsize=16)
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].legend(fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"收敛性对比图已保存到: {save_path}")
    plt.close()


def plot_band_comparison(pwe_bands_TM, pwe_bands_TE, fem_bands_TM, fem_bands_TE,
                         k_path, k_labels, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    n_segments = len(k_labels) - 1
    n_per_seg = (len(k_path) - 1) // n_segments
    x_positions = [i * n_per_seg for i in range(n_segments + 1)]
    x = np.arange(len(k_path))

    axes[0].set_title('TM 模式对比', fontsize=16)
    for band in range(pwe_bands_TM.shape[1]):
        axes[0].plot(x, pwe_bands_TM[:, band], 'b-', linewidth=2.5, label='PWE' if band == 0 else "")
        axes[0].plot(x, fem_bands_TM[:, band], 'r--', linewidth=2.5, label='FEM' if band == 0 else "")
    for xi in x_positions:
        axes[0].axvline(x=xi, color='k', linestyle='-', alpha=0.6, linewidth=1)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(k_labels, fontsize=14)
    axes[0].set_ylabel('频率 (ωa/2πc)', fontsize=14)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=12)
    axes[0].set_ylim(0, None)

    axes[1].set_title('TE 模式对比', fontsize=16)
    for band in range(pwe_bands_TE.shape[1]):
        axes[1].plot(x, pwe_bands_TE[:, band], 'b-', linewidth=2.5, label='PWE' if band == 0 else "")
        axes[1].plot(x, fem_bands_TE[:, band], 'r--', linewidth=2.5, label='FEM' if band == 0 else "")
    for xi in x_positions:
        axes[1].axvline(x=xi, color='k', linestyle='-', alpha=0.6, linewidth=1)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(k_labels, fontsize=14)
    axes[1].set_ylabel('频率 (ωa/2πc)', fontsize=14)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=12)
    axes[1].set_ylim(0, None)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"能带对比图已保存到: {save_path}")
    plt.close()


def plot_high_contrast_case():
    print("\n" + "=" * 60)
    print("高对比度介电常数测试 (ε2/ε1 = 20)")
    print("=" * 60)

    params_high = {
        'lattice_type': 'triangular',
        'a': 1.0,
        'eps1': 1.0,
        'eps2': 20.0,
        'radius': 0.3
    }

    num_bands = 6
    k_path = PhotonicCrystalFEM(**params_high).generate_k_path(n_points_per_segment=10)
    k_labels = PhotonicCrystalFEM(**params_high).get_k_labels()

    print(f"\nPWE 计算 (ε2/ε1 = {params_high['eps2']}):")
    pwe_bands_TM, pwe_bands_TE = PhotonicCrystal2D(**params_high).compute_band_structure(
        k_path, mode='both', num_bands=num_bands, num_planes=60
    )

    print(f"\nFEM 计算 (ε2/ε1 = {params_high['eps2']}):")
    pc_fem_high = PhotonicCrystalFEM(**params_high)
    pc_fem_high.generate_mesh(resolution=0.08)
    fem_bands_TM, fem_bands_TE = pc_fem_high.compute_band_structure(
        k_path, mode='both', num_bands=num_bands
    )

    print_gap_analysis(pwe_bands_TM, "TM (PWE, 高对比度)")
    print_gap_analysis(fem_bands_TM, "TM (FEM, 高对比度)")

    plot_band_comparison(
        pwe_bands_TM, pwe_bands_TE, fem_bands_TM, fem_bands_TE,
        k_path, k_labels, save_path='band_comparison_high_contrast.png'
    )


def main():
    print("=" * 60)
    print("PWE vs FEM 收敛性和精度对比分析")
    print("=" * 60)

    params = {
        'lattice_type': 'triangular',
        'a': 1.0,
        'eps1': 1.0,
        'eps2': 12.0,
        'radius': 0.2
    }

    num_bands = 6
    k_path = PhotonicCrystalFEM(**params).generate_k_path(n_points_per_segment=8)
    k_labels = PhotonicCrystalFEM(**params).get_k_labels()

    print(f"\n参数设置:")
    print(f"  晶格类型: {params['lattice_type']}")
    print(f"  介电常数比: {params['eps2']}/{params['eps1']}")
    print(f"  圆柱半径: {params['radius']}a")
    print(f"  k点数: {len(k_path)}")
    print(f"  能带数: {num_bands}")

    print("\n" + "=" * 60)
    print("PWE 收敛性测试")
    print("=" * 60)

    num_planes_list = [20, 40, 60, 100]
    pwe_results = run_pwe_calculation(params, k_path, num_bands, num_planes_list)

    print("\n" + "=" * 60)
    print("FEM 收敛性测试")
    print("=" * 60)

    resolutions = [0.15, 0.12, 0.1, 0.08]
    fem_results = run_fem_calculation(params, k_path, num_bands, resolutions)

    print("\n" + "=" * 60)
    print("生成对比图表")
    print("=" * 60)

    plot_convergence_comparison(
        pwe_results, fem_results, k_point_idx=0,
        save_path='convergence_comparison.png'
    )

    ref_pwe_TM = pwe_results[num_planes_list[-1]]['TM']
    ref_pwe_TE = pwe_results[num_planes_list[-1]]['TE']
    ref_fem_TM = fem_results[resolutions[-1]]['TM']
    ref_fem_TE = fem_results[resolutions[-1]]['TE']

    plot_band_comparison(
        ref_pwe_TM, ref_pwe_TE, ref_fem_TM, ref_fem_TE,
        k_path, k_labels, save_path='band_comparison.png'
    )

    print("\n" + "=" * 60)
    print("方法对比总结")
    print("=" * 60)

    pwe_best = pwe_results[num_planes_list[-1]]
    fem_best = fem_results[resolutions[-1]]

    print(f"\nPWE (最优):")
    print(f"  平面波数量: {pwe_best['n_planes']}")
    print(f"  矩阵大小: {pwe_best['n_planes']} x {pwe_best['n_planes']}")

    print(f"\nFEM (最优):")
    print(f"  节点数量: {fem_best['n_nodes']}")
    print(f"  单元数量: {fem_best['n_elements']}")
    print(f"  矩阵大小: {fem_best['n_nodes'] - len(PhotonicCrystalFEM(**params).boundary_pairs) if PhotonicCrystalFEM(**params).boundary_pairs else fem_best['n_nodes']}")

    plot_high_contrast_case()

    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    print("""
    1. PWE方法:
       - 对于低对比度结构（ε<10）收敛较快
       - 高对比度时需要大量平面波才能收敛
       - 实现简单但对不连续介电常数适应性差

    2. FEM方法:
       - 无论对比度高低都能快速收敛
       - 三角网格能更好地拟合圆柱界面
       - 矩阵稀疏，存储效率高
       - 更适合复杂几何形状

    3. 推荐:
       - 高对比度结构 → 优先使用FEM
       - 简单规则结构 → PWE可接受
       - 需要高精度 → FEM更可靠
    """)

    print("\n所有分析完成！")


if __name__ == '__main__':
    main()
