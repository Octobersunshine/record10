import mpmath as mp
import math
from zeta_statistics import (
    compute_zeta_zeros,
    count_zeros_N,
    compute_normalized_gaps,
    wigner_surmise_p,
    wigner_surmise_F,
    poisson_p,
    poisson_F,
    histogram,
    cumulative_distribution
)


def generate_plot_data(T_max: float = 500, precision: int = 50):
    print("=" * 70)
    print("生成可视化数据文件")
    print("=" * 70)
    
    print(f"\n计算零点到 T = {T_max}...")
    zeros = compute_zeta_zeros(0, T_max, precision)
    print(f"共找到 {len(zeros)} 个零点")
    
    with open("data_zeros_list.txt", "w") as f:
        f.write("# 黎曼Zeta函数零点 (临界线)\n")
        f.write("# 格式: 序号, 虚部 γ_n\n")
        for i, g in enumerate(zeros, 1):
            f.write(f"{i} {g:.12f}\n")
    print("✓ 已生成: data_zeros_list.txt")
    
    with open("data_NT_function.txt", "w") as f:
        f.write("# N(T) 零点计数函数\n")
        f.write("# T, 实际N(T), 估计N(T), S(T)\n")
        for T in range(50, int(T_max) + 1, 50):
            N_actual = sum(1 for g in zeros if g <= T)
            N_est, main_term, S_T = count_zeros_N(T, precision)
            f.write(f"{T} {N_actual} {N_est} {S_T:.6f}\n")
    print("✓ 已生成: data_NT_function.txt")
    
    normalized_gaps = compute_normalized_gaps(zeros)
    
    bin_centers, bin_left, counts, densities = histogram(normalized_gaps, bins=40, range_max=4.0)
    
    with open("data_gap_distribution.txt", "w") as f:
        f.write("# 归一化间隔分布\n")
        f.write("# s, 计数, 经验密度, Wigner推测, Poisson分布\n")
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
        for i, s in enumerate(bin_centers):
            wigner = wigner_surmise_p(s)
            poisson = poisson_p(s)
            f.write(f"{s:.4f} {counts[i]} {densities[i]:.6f} {wigner:.6f} {poisson:.6f}\n")
    print("✓ 已生成: data_gap_distribution.txt")
    
    x_vals, F_emp = cumulative_distribution(normalized_gaps, points=200, range_max=4.0)
    
    with open("data_cumulative_distribution.txt", "w") as f:
        f.write("# 累积分布函数 (CDF)\n")
        f.write("# s, 经验F(s), Wigner_F(s), Poisson_F(s)\n")
        for x, F in zip(x_vals, F_emp):
            F_w = wigner_surmise_F(x)
            F_p = poisson_F(x)
            f.write(f"{x:.4f} {F:.6f} {F_w:.6f} {F_p:.6f}\n")
    print("✓ 已生成: data_cumulative_distribution.txt")
    
    with open("data_gap_statistics.txt", "w") as f:
        f.write("# 间隔统计摘要\n")
        f.write(f"# T_max = {T_max}, 零点数量 = {len(zeros)}\n")
        f.write("#\n")
        
        mean_gap = sum(normalized_gaps) / len(normalized_gaps)
        var_gap = sum((g - mean_gap)**2 for g in normalized_gaps) / len(normalized_gaps)
        
        f.write(f"mean_normalized_gap = {mean_gap:.8f}\n")
        f.write(f"variance_normalized_gap = {var_gap:.8f}\n")
        f.write(f"std_normalized_gap = {math.sqrt(var_gap):.8f}\n")
        f.write(f"min_normalized_gap = {min(normalized_gaps):.8f}\n")
        f.write(f"max_normalized_gap = {max(normalized_gaps):.8f}\n")
        f.write(f"median_normalized_gap = {sorted(normalized_gaps)[len(normalized_gaps)//2]:.8f}\n")
        
        near_zero = sum(1 for g in normalized_gaps if g < 0.1) / len(normalized_gaps)
        f.write(f"fraction_gaps_less_0.1 = {near_zero:.6f}\n")
        
        for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            frac = sum(1 for g in normalized_gaps if g > threshold) / len(normalized_gaps)
            f.write(f"fraction_gaps_greater_{threshold:.1f} = {frac:.6f}\n")
        
        f.write("\n# 与理论对比\n")
        f.write(f"GUE_Wigner_variance = 0.17735\n")
        f.write(f"Poisson_variance = 1.0\n")
    
    print("✓ 已生成: data_gap_statistics.txt")
    
    print("\n" + "=" * 70)
    print("所有数据文件生成完成!")
    print("=" * 70)
    
    return zeros, normalized_gaps


if __name__ == "__main__":
    T_MAX = 400
    PRECISION = 50
    zeros, gaps = generate_plot_data(T_MAX, PRECISION)
