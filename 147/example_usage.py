import numpy as np
from nmr_peak_fitting import (
    auto_peak_fit, 
    auto_peak_fit_with_ic,
    print_results, 
    generate_test_data,
    generate_overlapping_peaks_data
)


def example_basic_fit():
    print("=" * 80)
    print("示例1: 基础峰拟合（固定峰数）")
    print("=" * 80)
    
    x, y, _ = generate_test_data(n_peaks=4, noise_level=0.02)
    
    print("\n使用高斯线形拟合:")
    results, popt, props = auto_peak_fit(
        x, y, 
        peak_type='gaussian',
        height_threshold=None,
        distance=10,
        prominence=0.3,
        width_range=(1, 50),
        plot=False
    )
    print_results(results)
    
    return results


def example_overlapping_peaks_bic():
    print("\n" + "=" * 80)
    print("示例2: 重叠峰拟合 - 使用BIC自动选择最优峰数")
    print("=" * 80)
    
    np.random.seed(42)
    x, y, true_params = generate_overlapping_peaks_data(
        n_true_peaks=3, 
        noise_level=0.03, 
        overlap=True
    )
    
    print("\n真实峰参数 (3个重叠峰):")
    print_results(true_params)
    
    print("\n使用BIC自动选择最优峰数:")
    results, popt, ic_analysis = auto_peak_fit_with_ic(
        x, y,
        peak_type='gaussian',
        criterion='bic',
        max_peaks=8,
        min_peaks=1,
        plot=True,
        verbose=True
    )
    print_results(results)
    
    print(f"\n真实峰数: 3, BIC选择的峰数: {ic_analysis['best_n']}")
    
    return results, ic_analysis


def example_compare_aic_bic():
    print("\n" + "=" * 80)
    print("示例3: 比较AIC和BIC准则的选择结果")
    print("=" * 80)
    
    np.random.seed(123)
    x, y, true_params = generate_overlapping_peaks_data(
        n_true_peaks=4, 
        noise_level=0.02, 
        overlap=True
    )
    
    print("\n真实峰参数 (4个重叠峰):")
    print_results(true_params)
    
    print("\n使用BIC准则:")
    results_bic, _, ic_bic = auto_peak_fit_with_ic(
        x, y,
        peak_type='gaussian',
        criterion='bic',
        max_peaks=10,
        min_peaks=1,
        plot=False,
        verbose=True
    )
    print_results(results_bic)
    
    print("\n使用AIC准则:")
    results_aic, _, ic_aic = auto_peak_fit_with_ic(
        x, y,
        peak_type='gaussian',
        criterion='aic',
        max_peaks=10,
        min_peaks=1,
        plot=False,
        verbose=True
    )
    print_results(results_aic)
    
    print(f"\n真实峰数: 4")
    print(f"BIC选择峰数: {ic_bic['best_n']}")
    print(f"AIC选择峰数: {ic_aic['best_n']}")
    
    return results_bic, results_aic


def example_lorentzian_with_ic():
    print("\n" + "=" * 80)
    print("示例4: 洛伦兹线形 + BIC自动选峰")
    print("=" * 80)
    
    x = np.linspace(0, 12, 2000)
    y = np.zeros_like(x)
    
    from nmr_peak_fitting import lorentzian
    true_peaks = [
        (1.5, 2.5, 0.15),
        (1.8, 2.8, 0.12),
        (2.0, 5.0, 0.1),
        (1.2, 7.5, 0.2)
    ]
    
    for amp, cen, gamma in true_peaks:
        y += lorentzian(x, amp, cen, gamma)
    y += np.random.normal(0, 0.02, len(y))
    
    print(f"\n真实峰数: {len(true_peaks)}")
    print("真实峰位置:", [p[1] for p in true_peaks])
    
    results, popt, ic_analysis = auto_peak_fit_with_ic(
        x, y,
        peak_type='lorentzian',
        criterion='bic',
        max_peaks=8,
        min_peaks=1,
        plot=True,
        verbose=True
    )
    print_results(results)
    
    return results, ic_analysis


def example_voigt_with_ic():
    print("\n" + "=" * 80)
    print("示例5: Voigt线形 + BIC自动选峰")
    print("=" * 80)
    
    x = np.linspace(0, 10, 1500)
    y = np.zeros_like(x)
    
    from nmr_peak_fitting import voigt
    true_peaks = [
        (1.0, 3.0, 0.08, 0.05),
        (1.2, 3.4, 0.07, 0.06),
        (0.8, 6.0, 0.1, 0.08)
    ]
    
    for amp, cen, sigma, gamma in true_peaks:
        y += voigt(x, amp, cen, sigma, gamma)
    y += np.random.normal(0, 0.015, len(y))
    
    print(f"\n真实峰数: {len(true_peaks)}")
    print("真实峰位置:", [p[1] for p in true_peaks])
    
    results, popt, ic_analysis = auto_peak_fit_with_ic(
        x, y,
        peak_type='voigt',
        criterion='bic',
        max_peaks=6,
        min_peaks=1,
        plot=True,
        verbose=True
    )
    print_results(results)
    
    return results, ic_analysis


def example_custom_data_workflow():
    print("\n" + "=" * 80)
    print("示例6: 处理真实NMR数据的完整工作流")
    print("=" * 80)
    
    print("\n假设您有一个CSV文件 'nmr_data.csv'，格式如下:")
    print("chemical_shift,intensity")
    print("10.0,0.01")
    print("9.99,0.02")
    print("...")
    
    print("\n=== 步骤1: 加载数据 ===")
    print("import numpy as np")
    print("data = np.loadtxt('nmr_data.csv', delimiter=',', skiprows=1)")
    print("x = data[:, 0]  # 化学位移列 (ppm)")
    print("y = data[:, 1]  # 强度列")
    
    print("\n=== 步骤2: 使用BIC自动选择最优峰数 (推荐) ===")
    print("from nmr_peak_fitting import auto_peak_fit_with_ic, print_results")
    print("")
    print("results, popt, ic_analysis = auto_peak_fit_with_ic(")
    print("    x, y,")
    print("    peak_type='gaussian',   # 峰形: 'gaussian', 'lorentzian', 'voigt'")
    print("    criterion='bic',        # 信息准则: 'bic' (保守) 或 'aic' (自由)")
    print("    max_peaks=15,           # 最大测试峰数")
    print("    min_peaks=1,            # 最小测试峰数")
    print("    plot=True,              # 是否绘图")
    print("    verbose=True            # 是否显示详细信息")
    print(")")
    
    print("\n=== 步骤3: 查看结果 ===")
    print("print_results(results)")
    
    print("\n=== 步骤4: 保存结果 ===")
    print("import pandas as pd")
    print("df = pd.DataFrame(results)")
    print("df.to_csv('peak_fitting_results.csv', index=False)")
    print("")
    print("# 保存AIC/BIC分析数据")
    print("ic_df = pd.DataFrame({")
    print("    'n_peaks': ic_analysis['n_peaks_range'],")
    print("    'aic': ic_analysis['aic_values'],")
    print("    'bic': ic_analysis['bic_values']")
    print("})")
    print("ic_df.to_csv('ic_analysis.csv', index=False)")
    
    print("\n=== 参数调整建议 ===")
    print("- 如果BIC选择的峰数太少，尝试使用AIC")
    print("- 如果AIC选择的峰数太多（过拟合），尝试使用BIC")
    print("- 增加 max_peaks 以测试更多峰数模型")
    print("- 对于高噪声数据，BIC通常更稳健")


if __name__ == "__main__":
    np.random.seed(42)
    
    example_basic_fit()
    
    results_bic, ic_bic = example_overlapping_peaks_bic()
    
    results_bic4, results_aic4 = example_compare_aic_bic()
    
    results_lorentz, ic_lorentz = example_lorentzian_with_ic()
    
    print("\n" + "=" * 80)
    print("AIC vs BIC 准则说明:")
    print("=" * 80)
    print("AIC (赤池信息准则):")
    print("  - 公式: AIC = 2k + n ln(RSS/n)")
    print("  - 对模型复杂度的惩罚较轻")
    print("  - 倾向于选择更复杂的模型（更多峰）")
    print("  - 适用于预测最优的情况")
    print("")
    print("BIC (贝叶斯信息准则):")
    print("  - 公式: BIC = k ln(n) + n ln(RSS/n)")
    print("  - 对模型复杂度的惩罚较重 (ln(n) > 2 当 n > 7)")
    print("  - 倾向于选择更简单的模型（更少峰）")
    print("  - 对过拟合的抑制更强，更保守")
    print("  - 适用于样本量大的情况")
    print("")
    print("NMR谱拟合建议:")
    print("  - 优先使用 BIC，更能防止过拟合")
    print("  - 如果确信有重叠峰未被检测到，尝试 AIC")
    print("  - 可视化 AIC/BIC 随峰数的变化曲线辅助判断")
    print("=" * 80)
