import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

from photonic_crystal_fem import PhotonicCrystalFEM, print_gap_analysis
from photonic_crystal_pwe import PhotonicCrystal2D

rcParams['font.sans-serif'] = ['SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False


def main():
    print("=" * 70)
    print("PWE vs FEM 收敛性对比 - 二维光子晶体能带计算")
    print("=" * 70)

    params = {
        'lattice_type': 'triangular',
        'a': 1.0,
        'eps1': 1.0,
        'eps2': 12.0,
        'radius': 0.2
    }

    num_bands = 5
    k_point = np.array([0.0, 0.0])

    print(f"\n参数设置:")
    print(f"  晶格: {params['lattice_type']}, ε2/ε1 = {params['eps2']}/{params['eps1']}")
    print(f"  半径: r = {params['radius']}a")
    print(f"  测试k点: Γ (0, 0)")
    print(f"  计算能带数: {num_bands}")

    print("\n" + "=" * 70)
    print("PWE方法测试（不同平面波数量）")
    print("=" * 70)

    pwe_planes = [20, 40, 60, 100]
    pwe_results = []
    pwe_times = []

    for n_planes in pwe_planes:
        print(f"\n  平面波数: {n_planes:4d}", end="")
        pc_pwe = PhotonicCrystal2D(**params)
        pc_pwe.generate_reciprocal_vectors(n_planes)

        start_time = time.time()
        freqs = pc_pwe.solve_eigenvalues(k_point, mode='TM', num_bands=num_bands)
        elapsed = time.time() - start_time

        # 跳过第一个接近0的声学模式
        valid_freqs = freqs[freqs > 0.01][:num_bands]

        pwe_results.append(valid_freqs)
        pwe_times.append(elapsed)

        print(f"  耗时: {elapsed:6.2f}s  频率: {valid_freqs}")

    print("\n" + "=" * 70)
    print("FEM方法测试（不同网格分辨率）")
    print("=" * 70)

    fem_resolutions = [0.2, 0.15, 0.12, 0.1]
    fem_results = []
    fem_times = []
    fem_n_nodes = []

    for res in fem_resolutions:
        print(f"\n  分辨率: {res:4.2f}", end="")
        pc_fem = PhotonicCrystalFEM(**params)
        pc_fem.generate_mesh(resolution=res)

        start_time = time.time()
        freqs = pc_fem.solve_eigenvalues(k_point, mode='TM', num_bands=num_bands)
        elapsed = time.time() - start_time

        valid_freqs = freqs[freqs > 0.01][:num_bands]

        fem_results.append(valid_freqs)
        fem_times.append(elapsed)
        fem_n_nodes.append(len(pc_fem.nodes))

        print(f"  节点: {len(pc_fem.nodes):4d}  耗时: {elapsed:6.2f}s  频率: {valid_freqs}")

    print("\n" + "=" * 70)
    print("收敛性分析")
    print("=" * 70)

    if len(pwe_results) >= 2:
        ref_pwe = pwe_results[-1]
        print(f"\nPWE收敛性 (参考: {pwe_planes[-1]} 平面波):")
        for i in range(len(pwe_planes) - 1):
            err = np.mean(np.abs(pwe_results[i] - ref_pwe) / np.abs(ref_pwe) * 100)
            print(f"  {pwe_planes[i]:4d} 平面波 → 平均相对误差: {err:7.2f}%")

    if len(fem_results) >= 2:
        ref_fem = fem_results[-1]
        print(f"\nFEM收敛性 (参考: {fem_resolutions[-1]} 分辨率, {fem_n_nodes[-1]} 节点):")
        for i in range(len(fem_resolutions) - 1):
            err = np.mean(np.abs(fem_results[i] - ref_fem) / np.abs(ref_fem) * 100)
            print(f"  {fem_resolutions[i]:4.2f} 分辨率 ({fem_n_nodes[i]:3d} 节点) → 平均相对误差: {err:7.2f}%")

    print("\n" + "=" * 70)
    print("方法对比")
    print("=" * 70)

    print(f"\n{'方法':<15} {'自由度':<10} {'耗时(s)':<10} {'精度(%)':<12}")
    print("-" * 55)

    for i in range(len(pwe_planes)):
        err = np.mean(np.abs(pwe_results[i] - ref_pwe) / np.abs(ref_pwe) * 100) if i < len(pwe_planes) - 1 else 0.0
        print(f"{'PWE':<15} {pwe_planes[i]:<10} {pwe_times[i]:<10.2f} {err:<12.2f}")

    for i in range(len(fem_resolutions)):
        err = np.mean(np.abs(fem_results[i] - ref_fem) / np.abs(ref_fem) * 100) if i < len(fem_resolutions) - 1 else 0.0
        print(f"{'FEM':<15} {fem_n_nodes[i]:<10} {fem_times[i]:<10.2f} {err:<12.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if len(pwe_results) >= 2:
        errors_pwe = []
        for i in range(len(pwe_planes)):
            err = np.mean(np.abs(pwe_results[i] - ref_pwe) / np.abs(ref_pwe) * 100) if i < len(pwe_planes) - 1 else 0.01
            errors_pwe.append(err)
        axes[0].loglog(pwe_planes, errors_pwe, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('平面波数量', fontsize=14)
        axes[0].set_ylabel('相对误差 (%)', fontsize=14)
        axes[0].set_title('PWE 收敛曲线', fontsize=16)
        axes[0].grid(True, alpha=0.3, which='both')

    if len(fem_results) >= 2:
        errors_fem = []
        for i in range(len(fem_resolutions)):
            err = np.mean(np.abs(fem_results[i] - ref_fem) / np.abs(ref_fem) * 100) if i < len(fem_resolutions) - 1 else 0.01
            errors_fem.append(err)
        axes[1].loglog(fem_n_nodes, errors_fem, 'rs-', linewidth=2, markersize=8)
        axes[1].set_xlabel('节点数量', fontsize=14)
        axes[1].set_ylabel('相对误差 (%)', fontsize=14)
        axes[1].set_title('FEM 收敛曲线', fontsize=16)
        axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('convergence_comparison_quick.png', dpi=300, bbox_inches='tight')
    print(f"\n收敛曲线图已保存到: convergence_comparison_quick.png")
    plt.close()

    print("\n" + "=" * 70)
    print("结论:")
    print("=" * 70)
    print("""
    1. PWE方法:
       ✓ 实现简单，小问题计算快
       ✗ 高对比度介电常数收敛慢，需要大量平面波
       ✗ 介电常数突变处Gibbs现象严重

    2. FEM方法:
       ✓ 自适应三角网格更好地拟合圆柱界面
       ✓ 收敛速度快，精度随节点数稳步提高
       ✓ 适合任意几何形状和高对比度结构
       ✗ 矩阵组装相对复杂

    3. 适用场景:
       • 低对比度、规则结构 → PWE足够
       • 高对比度、复杂结构 → FEM更优
       • 需要高精度结果 → FEM更可靠
    """)

    print("\n对比测试完成！")


if __name__ == '__main__':
    main()
