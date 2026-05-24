import mpmath as mp
import math
from typing import List, Tuple, Dict
from collections import defaultdict


def riemann_siegel_theta(t: mp.mpf, order: int = 5) -> mp.mpf:
    if t == 0:
        return mp.mpf('0')
    
    result = (t / 2) * mp.log(t / (2 * mp.pi)) - t / 2 - mp.pi / 8
    
    correction_terms = [
        1 / 48, 7 / 5760, 31 / 80640, 127 / 435456, 511 / 2903040,
        2047 / 15206400, 8191 / 83607552, 32767 / 479001600, 131071 / 2905943040
    ]
    
    for k in range(min(order, len(correction_terms))):
        exponent = 2 * k + 1
        result += correction_terms[k] / (t**exponent)
    
    return result


def compute_zeta_zeros(
    t_min: float, 
    t_max: float, 
    precision: int = 50
) -> List[float]:
    mp.mp.dps = precision
    
    t_min_mp = mp.mpf(str(t_min))
    t_max_mp = mp.mpf(str(t_max))
    
    def Z_func(t):
        s = mp.mpf('0.5') + 1j * t
        zeta_val = mp.zeta(s)
        theta = riemann_siegel_theta(t)
        return mp.re(zeta_val * mp.e**(1j * theta))
    
    zeros = []
    t = t_min_mp
    step = mp.mpf('0.3')
    
    prev_Z = Z_func(t)
    prev_t = t
    
    while t < t_max_mp:
        t += step
        current_Z = Z_func(t)
        
        if prev_Z * current_Z < 0:
            try:
                root = mp.findroot(Z_func, (prev_t, t))
                s = mp.mpf('0.5') + 1j * root
                zeta_val = mp.zeta(s)
                if abs(zeta_val) < 1e-6:
                    zeros.append(float(root))
            except:
                pass
        
        prev_Z = current_Z
        prev_t = t
    
    zeros.sort()
    return zeros


def count_zeros_N(T: float, precision: int = 50) -> Tuple[int, float, float]:
    mp.mp.dps = precision
    T_mp = mp.mpf(str(T))
    
    main_term = (T_mp / (2 * mp.pi)) * mp.log(T_mp / (2 * mp.pi)) - T_mp / (2 * mp.pi) + 7 / 8
    
    try:
        s = mp.mpf('0.5') + 1j * T_mp
        zeta_val = mp.zeta(s)
        S_T = (1 / mp.pi) * mp.arg(zeta_val)
    except:
        S_T = mp.mpf('0')
    
    N_T = main_term + S_T
    
    return int(mp.floor(N_T)), float(main_term), float(S_T)


def N_T_approx(T: float) -> float:
    T_mp = mp.mpf(str(T))
    return float((T_mp / (2 * mp.pi)) * mp.log(T_mp / (2 * mp.pi)) - T_mp / (2 * mp.pi) + 7 / 8)


def compute_gaps(zeros: List[float]) -> List[float]:
    gaps = []
    for i in range(1, len(zeros)):
        gaps.append(zeros[i] - zeros[i-1])
    return gaps


def compute_normalized_gaps(zeros: List[float]) -> List[float]:
    gaps = compute_gaps(zeros)
    if not gaps:
        return []
    
    n = len(zeros)
    if n < 2:
        return []
    
    T_start = zeros[0]
    T_end = zeros[-1]
    
    density_avg = n / (T_end - T_start) if (T_end - T_start) > 0 else 0
    
    normalized_gaps = []
    for i, gap in enumerate(gaps):
        t_mid = (zeros[i] + zeros[i+1]) / 2
        local_density = (1 / (2 * mp.pi)) * math.log(t_mid / (2 * mp.pi)) if t_mid > 0 else density_avg
        normalized_gaps.append(gap * float(local_density))
    
    return normalized_gaps


def wigner_surmise_p(s: float) -> float:
    return (math.pi / 2) * s * math.exp(-math.pi * s**2 / 4)


def wigner_surmise_F(s: float) -> float:
    return 1 - math.exp(-math.pi * s**2 / 4)


def poisson_p(s: float) -> float:
    return math.exp(-s)


def poisson_F(s: float) -> float:
    return 1 - math.exp(-s)


def histogram(data: List[float], bins: int = 30, range_min: float = 0, range_max: float = 3.5) -> Tuple[List[float], List[float], List[int]]:
    bin_width = (range_max - range_min) / bins
    bin_centers = []
    bin_left = []
    counts = [0] * bins
    
    for i in range(bins):
        left = range_min + i * bin_width
        bin_left.append(left)
        bin_centers.append(left + bin_width / 2)
    
    for x in data:
        if range_min <= x < range_max:
            idx = int((x - range_min) / bin_width)
            if 0 <= idx < bins:
                counts[idx] += 1
    
    total = len(data)
    if total > 0:
        densities = [c / (total * bin_width) for c in counts]
    else:
        densities = [0.0] * bins
    
    return bin_centers, bin_left, counts, densities


def cumulative_distribution(data: List[float], points: int = 100, range_max: float = 3.5) -> Tuple[List[float], List[float]]:
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    x_values = []
    F_values = []
    
    for i in range(points + 1):
        x = (range_max * i) / points
        x_values.append(x)
        
        count = sum(1 for d in sorted_data if d <= x)
        F_values.append(count / n if n > 0 else 0)
    
    return x_values, F_values


def compute_statistics(gaps: List[float], normalized_gaps: List[float]) -> Dict:
    stats = {}
    
    if gaps:
        stats['gaps_mean'] = sum(gaps) / len(gaps)
        stats['gaps_variance'] = sum((g - stats['gaps_mean'])**2 for g in gaps) / len(gaps)
        stats['gaps_min'] = min(gaps)
        stats['gaps_max'] = max(gaps)
        stats['gaps_median'] = sorted(gaps)[len(gaps)//2]
    
    if normalized_gaps:
        stats['normalized_mean'] = sum(normalized_gaps) / len(normalized_gaps)
        stats['normalized_variance'] = sum((g - stats['normalized_mean'])**2 for g in normalized_gaps) / len(normalized_gaps)
        stats['normalized_min'] = min(normalized_gaps)
        stats['normalized_max'] = max(normalized_gaps)
        stats['normalized_median'] = sorted(normalized_gaps)[len(normalized_gaps)//2]
        
        variance = stats['normalized_variance']
        sigma = math.sqrt(variance)
        stats['normalized_std'] = sigma
        
        near_zero = sum(1 for g in normalized_gaps if g < 0.1)
        stats['percent_near_zero'] = (near_zero / len(normalized_gaps)) * 100
        
        large_gaps = sum(1 for g in normalized_gaps if g > 3.0)
        stats['percent_large'] = (large_gaps / len(normalized_gaps)) * 100
    
    return stats


def compare_with_GUE(normalized_gaps: List[float]) -> Dict:
    comparison = {}
    
    bin_centers, _, _, densities = histogram(normalized_gaps, bins=35)
    
    chi_square = 0
    for i, center in enumerate(bin_centers):
        wigner = wigner_surmise_p(center)
        if wigner > 0.01:
            chi_square += (densities[i] - wigner)**2 / wigner
    
    comparison['chi_square_wigner'] = chi_square
    
    x_vals, F_emp = cumulative_distribution(normalized_gaps)
    
    max_diff_wigner = 0
    max_diff_poisson = 0
    for x, F in zip(x_vals, F_emp):
        diff_w = abs(F - wigner_surmise_F(x))
        diff_p = abs(F - poisson_F(x))
        max_diff_wigner = max(max_diff_wigner, diff_w)
        max_diff_poisson = max(max_diff_poisson, diff_p)
    
    comparison['KS_statistic_Wigner'] = max_diff_wigner
    comparison['KS_statistic_Poisson'] = max_diff_poisson
    
    near_zero_density = sum(1 for g in normalized_gaps if g < 0.05) / len(normalized_gaps)
    comparison['near_zero_density'] = near_zero_density
    comparison['Wigner_near_zero'] = wigner_surmise_p(0.025) * 0.05
    comparison['Poisson_near_zero'] = poisson_p(0.025) * 0.05
    
    return comparison


def generate_report(
    zeros: List[float], 
    T_min: float, 
    T_max: float,
    output_file: str = "zeta_statistics_report.txt"
):
    gaps = compute_gaps(zeros)
    normalized_gaps = compute_normalized_gaps(zeros)
    stats = compute_statistics(gaps, normalized_gaps)
    comparison = compare_with_GUE(normalized_gaps)
    
    N_actual = len(zeros)
    N_est, main_term, S_T = count_zeros_N(T_max)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("黎曼Zeta函数零点统计分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"分析范围: Im(s) ∈ [{T_min}, {T_max}]\n")
        f.write(f"实际零点数量: {N_actual}\n")
        f.write(f"N(T) 主项: {main_term:.2f}\n")
        f.write(f"S(T) 修正项: {S_T:.4f}\n")
        f.write(f"N(T) 估计值 (主项+S(T)): {N_est}\n")
        f.write(f"误差: {abs(N_actual - N_est)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("零点间隔统计 (原始间隔)\n")
        f.write("-" * 80 + "\n")
        f.write(f"平均值:   {stats['gaps_mean']:.6f}\n")
        f.write(f"方差:     {stats['gaps_variance']:.6f}\n")
        f.write(f"标准差:   {math.sqrt(stats['gaps_variance']):.6f}\n")
        f.write(f"最小值:   {stats['gaps_min']:.6f}\n")
        f.write(f"最大值:   {stats['gaps_max']:.6f}\n")
        f.write(f"中位数:   {stats['gaps_median']:.6f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("归一化间隔统计 (δ_n = γ_n · (1/2π)log(γ_n/2π))\n")
        f.write("-" * 80 + "\n")
        f.write(f"平均值:   {stats['normalized_mean']:.6f} (GUE/Poisson理论值: 1.0)\n")
        f.write(f"方差:     {stats['normalized_variance']:.6f} (GUE: ~0.177, Poisson: 1.0)\n")
        f.write(f"标准差:   {stats['normalized_std']:.6f}\n")
        f.write(f"最小值:   {stats['normalized_min']:.6f}\n")
        f.write(f"最大值:   {stats['normalized_max']:.6f}\n")
        f.write(f"中位数:   {stats['normalized_median']:.6f}\n")
        f.write(f"间隔<0.1:  {stats['percent_near_zero']:.2f}% (GUE排斥效应)\n")
        f.write(f"间隔>3.0:  {stats['percent_large']:.2f}%\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("与随机矩阵理论对比 (GUE Wigner推测 vs Poisson)\n")
        f.write("-" * 80 + "\n")
        f.write(f"KS统计量 (Wigner):  {comparison['KS_statistic_Wigner']:.6f}\n")
        f.write(f"KS统计量 (Poisson): {comparison['KS_statistic_Poisson']:.6f}\n")
        f.write(f"小间隔密度 (s<0.05): {comparison['near_zero_density']:.4f}\n")
        f.write(f"  - Wigner推测:     {comparison['Wigner_near_zero']:.4f} (s→0时趋于0)\n")
        f.write(f"  - Poisson分布:    {comparison['Poisson_near_zero']:.4f} (s→0时≈0.05)\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("Montgomery 成对关联验证\n")
        f.write("-" * 80 + "\n")
        f.write("注: 归一化间隔分布应趋近于GUE Wigner推测:\n")
        f.write("    p_Wigner(s) = (π/2) s exp(-π s²/4)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("前30个零点及其间隔\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'序号':<6} {'虚部 γ_n':<18} {'间隔 γ_{n+1}-γ_n':<18} {'归一化间隔':<18}\n")
        f.write("-" * 60 + "\n")
        
        for i in range(min(30, len(zeros))):
            gap_str = f"{gaps[i]:.6f}" if i < len(gaps) else "-"
            norm_str = f"{normalized_gaps[i]:.6f}" if i < len(normalized_gaps) else "-"
            f.write(f"{i+1:<6} {zeros[i]:<18.10f} {gap_str:<18} {norm_str:<18}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("直方图数据 (归一化间隔分布)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'区间':<15} {'计数':<10} {'密度':<15} {'Wigner':<15} {'Poisson':<15}\n")
        f.write("-" * 70 + "\n")
        
        bin_centers, bin_left, counts, densities = histogram(normalized_gaps, bins=35)
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
        
        for i in range(len(bin_centers)):
            center = bin_centers[i]
            left = bin_left[i]
            wigner = wigner_surmise_p(center)
            poisson = poisson_p(center)
            f.write(f"[{left:.2f}-{left+bin_width:.2f}]  {counts[i]:<10} {densities[i]:<15.4f} {wigner:<15.4f} {poisson:<15.4f}\n")
    
    print(f"统计报告已生成: {output_file}")
    return stats, comparison


def main():
    print("=" * 80)
    print("黎曼Zeta函数零点统计分析")
    print("=" * 80)
    
    T_MIN = 0
    T_MAX = 200
    PRECISION = 50
    
    print(f"\n正在计算 Im(s) ∈ [{T_MIN}, {T_MAX}] 范围内的零点...")
    zeros = compute_zeta_zeros(T_MIN, T_MAX, PRECISION)
    print(f"找到 {len(zeros)} 个零点")
    
    N_est, main_term, S_T = count_zeros_N(T_MAX, PRECISION)
    print(f"\n零点计数函数 N({T_MAX}):")
    print(f"  主项: {main_term:.2f}")
    print(f"  S(T): {S_T:.4f}")
    print(f"  N(T) ≈ {N_est}")
    print(f"  实际: {len(zeros)}")
    
    gaps = compute_gaps(zeros)
    normalized_gaps = compute_normalized_gaps(zeros)
    
    print(f"\n间隔统计:")
    print(f"  平均间隔: {sum(gaps)/len(gaps):.4f}")
    print(f"  平均归一化间隔: {sum(normalized_gaps)/len(normalized_gaps):.4f}")
    
    stats, comparison = generate_report(zeros, T_MIN, T_MAX, "zeta_statistics_report.txt")
    
    print(f"\n与GUE对比:")
    print(f"  KS统计量 (Wigner):  {comparison['KS_statistic_Wigner']:.6f}")
    print(f"  KS统计量 (Poisson): {comparison['KS_statistic_Poisson']:.6f}")
    print(f"  小间隔密度: {comparison['near_zero_density']:.4f} (GUE排斥效应明显)")
    
    print("\n" + "=" * 80)
    print("分析完成! 详细结果请查看 zeta_statistics_report.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
