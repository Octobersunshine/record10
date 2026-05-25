import numpy as np
from nmr_2d_fitting import (
    auto_peak_fit_2d,
    print_2d_results,
    generate_2d_test_data,
    generate_hsqc_style_data,
    generate_cosy_style_data
)


def example_basic_2d_fit():
    print("=" * 80)
    print("示例1: 基础二维峰拟合（高斯线形）")
    print("=" * 80)
    
    np.random.seed(42)
    x, y, Z, true_params = generate_2d_test_data(
        n_peaks=5,
        noise_level=0.05,
        peak_type='gaussian',
        size=128
    )
    
    print(f"\n真实峰数: {len(true_params)}")
    print("真实峰位置 (F2, F1, 振幅):")
    for p in true_params:
        print(f"  峰 {p['peak_index']}: ({p['x_position']:.2f}, {p['y_position']:.2f}), 振幅={p['amplitude']:.2f}")
    
    print("\n开始二维峰拟合...")
    results, popt, peaks = auto_peak_fit_2d(
        x, y, Z,
        peak_type='gaussian',
        threshold=3,
        min_distance=5,
        max_peaks=10,
        use_sa=True,
        sa_iter=300,
        plot=True,
        verbose=True
    )
    
    print_2d_results(results)
    
    return results


def example_hsqc_fitting():
    print("\n" + "=" * 80)
    print("示例2: HSQC 谱图拟合（异核单量子相干谱）")
    print("=" * 80)
    
    np.random.seed(123)
    x, y, Z, true_params = generate_hsqc_style_data(
        n_peaks=6,
        noise_level=0.04
    )
    
    print(f"\nHSQC 谱特征:")
    print(f"  F2维度 (13C): {x.min():.0f} - {x.max():.0f} ppm")
    print(f"  F1维度 (1H): {y.min():.0f} - {y.max():.0f} ppm")
    print(f"  真实峰数: {len(true_params)}")
    
    print("\n开始HSQC谱拟合...")
    results, popt, peaks = auto_peak_fit_2d(
        x, y, Z,
        peak_type='gaussian',
        threshold=3,
        min_distance=8,
        max_peaks=15,
        use_sa=True,
        sa_iter=400,
        plot=True,
        verbose=True
    )
    
    print_2d_results(results)
    
    print("\n峰归属建议（蛋白质主链）:")
    for r in results:
        if r['x_position'] > 160:
            assign = "可能是羰基碳 (C')"
        elif 120 < r['x_position'] < 145:
            assign = "可能是芳香碳"
        elif 45 < r['x_position'] < 65:
            assign = "可能是α碳 (Cα)"
        elif 15 < r['x_position'] < 45:
            assign = "可能是侧链碳"
        else:
            assign = "需要进一步分析"
        print(f"  峰 {r['peak_index']}: ({r['x_position']:.1f}, {r['y_position']:.2f}) - {assign}")
    
    return results


def example_cosy_fitting():
    print("\n" + "=" * 80)
    print("示例3: COSY 谱图拟合（同核相关谱）")
    print("=" * 80)
    
    np.random.seed(456)
    x, y, Z, true_params = generate_cosy_style_data(
        n_peaks=4,
        noise_level=0.05
    )
    
    print(f"\nCOSY 谱特征:")
    print(f"  F2维度 (1H): {x.min():.0f} - {x.max():.0f} ppm")
    print(f"  F1维度 (1H): {y.min():.0f} - {y.max():.0f} ppm")
    print(f"  真实峰数: {len(true_params)} (包括对角峰和交叉峰)")
    
    print("\n开始COSY谱拟合...")
    results, popt, peaks = auto_peak_fit_2d(
        x, y, Z,
        peak_type='gaussian',
        threshold=3,
        min_distance=10,
        max_peaks=20,
        use_sa=True,
        sa_iter=300,
        plot=True,
        verbose=True
    )
    
    print_2d_results(results)
    
    print("\nCOSY 峰分析:")
    diagonal_peaks = []
    cross_peaks = []
    
    for r in results:
        if abs(r['x_position'] - r['y_position']) < 0.5:
            diagonal_peaks.append(r)
        else:
            cross_peaks.append(r)
    
    print(f"  对角峰数量: {len(diagonal_peaks)}")
    for p in diagonal_peaks:
        print(f"    - ({p['x_position']:.2f}, {p['y_position']:.2f})")
    
    print(f"  交叉峰数量: {len(cross_peaks)}")
    for p in cross_peaks:
        print(f"    - ({p['x_position']:.2f}, {p['y_position']:.2f}) -> "
              f"{p['x_position']:.2f} 与 {p['y_position']:.2f} 存在 J-耦合")
    
    return results


def example_compare_peak_types():
    print("\n" + "=" * 80)
    print("示例4: 比较不同峰形的拟合效果")
    print("=" * 80)
    
    np.random.seed(789)
    x, y, Z, _ = generate_2d_test_data(
        n_peaks=4,
        noise_level=0.06,
        peak_type='gaussian',
        size=100
    )
    
    print("\n使用高斯线形拟合:")
    results_gauss, popt_gauss, _ = auto_peak_fit_2d(
        x, y, Z,
        peak_type='gaussian',
        threshold=3,
        min_distance=5,
        max_peaks=8,
        use_sa=True,
        sa_iter=200,
        plot=False,
        verbose=False
    )
    print(f"  检测到 {len(results_gauss)} 个峰")
    
    print("\n使用洛伦兹线形拟合:")
    results_lorentz, popt_lorentz, _ = auto_peak_fit_2d(
        x, y, Z,
        peak_type='lorentzian',
        threshold=3,
        min_distance=5,
        max_peaks=8,
        use_sa=True,
        sa_iter=200,
        plot=False,
        verbose=False
    )
    print(f"  检测到 {len(results_lorentz)} 个峰")
    
    xy_mesh = np.meshgrid(x, y)
    from nmr_2d_fitting import multi_peak_2d
    
    z_fit_gauss = multi_peak_2d(xy_mesh, *popt_gauss, peak_type='gaussian').reshape(Z.shape)
    z_fit_lorentz = multi_peak_2d(xy_mesh, *popt_lorentz, peak_type='lorentzian').reshape(Z.shape)
    
    mse_gauss = np.mean((z_fit_gauss - Z) ** 2)
    mse_lorentz = np.mean((z_fit_lorentz - Z) ** 2)
    
    print(f"\n拟合误差 (MSE):")
    print(f"  高斯线形: {mse_gauss:.6f}")
    print(f"  洛伦兹线形: {mse_lorentz:.6f}")
    
    if mse_gauss < mse_lorentz:
        print("  -> 高斯线形拟合效果更好")
    else:
        print("  -> 洛伦兹线形拟合效果更好")
    
    return results_gauss, results_lorentz


def example_without_sa():
    print("\n" + "=" * 80)
    print("示例5: 比较有无模拟退火的拟合效果")
    print("=" * 80)
    
    np.random.seed(111)
    x, y, Z, true_params = generate_2d_test_data(
        n_peaks=6,
        noise_level=0.05,
        size=128
    )
    
    print("\n无模拟退火（仅局部优化）:")
    results_no_sa, popt_no_sa, _ = auto_peak_fit_2d(
        x, y, Z,
        peak_type='gaussian',
        threshold=3,
        min_distance=5,
        max_peaks=10,
        use_sa=False,
        plot=False,
        verbose=False
    )
    
    print("\n有模拟退火（全局优化）:")
    results_sa, popt_sa, _ = auto_peak_fit_2d(
        x, y, Z,
        peak_type='gaussian',
        threshold=3,
        min_distance=5,
        max_peaks=10,
        use_sa=True,
        sa_iter=500,
        plot=True,
        verbose=True
    )
    
    xy_mesh = np.meshgrid(x, y)
    from nmr_2d_fitting import multi_peak_2d
    
    z_fit_no_sa = multi_peak_2d(xy_mesh, *popt_no_sa, peak_type='gaussian').reshape(Z.shape)
    z_fit_sa = multi_peak_2d(xy_mesh, *popt_sa, peak_type='gaussian').reshape(Z.shape)
    
    mse_no_sa = np.mean((z_fit_no_sa - Z) ** 2)
    mse_sa = np.mean((z_fit_sa - Z) ** 2)
    
    print(f"\n拟合误差对比:")
    print(f"  无模拟退火: {mse_no_sa:.6f}")
    print(f"  有模拟退火: {mse_sa:.6f}")
    print(f"  改进: {(mse_no_sa - mse_sa) / mse_no_sa * 100:.2f}%")
    
    return results_sa


def example_custom_data_workflow():
    print("\n" + "=" * 80)
    print("示例6: 处理真实二维NMR数据的工作流")
    print("=" * 80)
    
    print("\n=== 步骤1: 准备数据格式 ===")
    print("您的二维NMR数据应转换为以下格式:")
    print("  - x: F2维度的化学位移数组 (1D array)")
    print("  - y: F1维度的化学位移数组 (1D array)")
    print("  - Z: 2D强度矩阵 (shape: [len(y), len(x)])")
    
    print("\n=== 步骤2: 从常见格式加载数据 ===")
    print("\n从NMRPipe格式加载:")
    print("  import nmrglue as ng")
    print("  dic, data = ng.pipe.read('your_spectrum.ft2')")
    print("  uc_x = ng.pipe.make_uc(dic, data, 0)")
    print("  uc_y = ng.pipe.make_uc(dic, data, 1)")
    print("  x = uc_x.ppm_scale()")
    print("  y = uc_y.ppm_scale()")
    print("  Z = data")
    
    print("\n从Sparky格式加载:")
    print("  import nmrglue as ng")
    print("  dic, data = ng.sparky.read('your_spectrum.ucsf')")
    print("  # ... 类似上面的处理 ...")
    
    print("\n从numpy数组加载:")
    print("  data = np.load('your_spectrum.npy')")
    print("  x = np.linspace(x_min, x_max, data.shape[1])")
    print("  y = np.linspace(y_min, y_max, data.shape[0])")
    print("  Z = data")
    
    print("\n=== 步骤3: 执行峰拟合 ===")
    print("  from nmr_2d_fitting import auto_peak_fit_2d, print_2d_results")
    print("")
    print("  results, popt, peaks = auto_peak_fit_2d(")
    print("      x, y, Z,")
    print("      peak_type='gaussian',   # 峰形: 'gaussian', 'lorentzian', 'voigt'")
    print("      threshold=3,            # 峰检测阈值 (信噪比)")
    print("      min_distance=8,         # 峰之间最小距离 (像素)")
    print("      max_peaks=50,           # 最大峰数")
    print("      use_sa=True,            # 是否使用模拟退火")
    print("      sa_iter=500,            # 模拟退火迭代次数")
    print("      plot=True,              # 是否绘图")
    print("      verbose=True            # 是否显示详细信息")
    print("  )")
    
    print("\n=== 步骤4: 分析和保存结果 ===")
    print("  print_2d_results(results)")
    print("")
    print("  # 保存到CSV")
    print("  import pandas as pd")
    print("  df = pd.DataFrame(results)")
    print("  df.to_csv('2d_peak_list.csv', index=False)")
    
    print("\n=== 参数调整建议 ===")
    print("  - threshold: 根据噪声水平调整，3-5倍噪声是标准值")
    print("  - min_distance: 根据谱图分辨率和峰密度调整")
    print("  - sa_iter: 200-500通常足够，复杂谱图可增加到1000")
    print("  - peak_type: NMR通常用高斯，高分辨可用洛伦兹")


if __name__ == "__main__":
    np.random.seed(42)
    
    example_basic_2d_fit()
    
    results_hsqc = example_hsqc_fitting()
    
    results_cosy = example_cosy_fitting()
    
    example_compare_peak_types()
    
    example_without_sa()
    
    example_custom_data_workflow()
    
    print("\n" + "=" * 80)
    print("模拟退火全局优化说明:")
    print("=" * 80)
    print("模拟退火 (Simulated Annealing) 是一种基于蒙特卡洛的全局优化算法:")
    print("  1. 从局部优化的解开始")
    print("  2. 在高温下允许接受较差的解（避免局部最优）")
    print("  3. 逐渐降低温度，减少接受较差解的概率")
    print("  4. 最终收敛到全局最优解")
    print("")
    print("优点:")
    print("  - 避免陷入局部最优")
    print("  - 对初始参数不敏感")
    print("  - 适合高维、多峰的优化问题")
    print("")
    print("缺点:")
    print("  - 计算成本较高")
    print("  - 需要调整迭代次数和温度参数")
    print("")
    print("对于二维NMR峰拟合，模拟退火通常能提高5-20%的拟合精度")
    print("=" * 80)
