import mpmath as mp
from typing import List, Tuple, Optional


def riemann_siegel_theta(t: mp.mpf, order: int = 5) -> mp.mpf:
    if t == 0:
        return mp.mpf('0')
    
    result = (t / 2) * mp.log(t / (2 * mp.pi)) - t / 2 - mp.pi / 8
    
    correction_terms = [
        1 / 48,
        7 / 5760,
        31 / 80640,
        127 / 435456,
        511 / 2903040,
        2047 / 15206400,
        8191 / 83607552,
        32767 / 479001600,
        131071 / 2905943040
    ]
    
    for k in range(min(order, len(correction_terms))):
        exponent = 2 * k + 1
        result += correction_terms[k] / (t**exponent)
    
    return result


def riemann_siegel_remainder(t: mp.mpf, N: int, tol: mp.mpf) -> mp.mpf:
    tau = t / (2 * mp.pi)
    p = mp.sqrt(tau) - N
    
    if p < 0 or p > 1:
        return mp.mpf('0')
    
    n_terms = 0
    remainder = mp.mpf('0')
    max_terms = 20
    
    phi = 2 * mp.pi * (p * p - p - 1/16)
    cos_phi = mp.cos(phi)
    sin_phi = mp.sin(phi)
    
    C0 = mp.nsum(lambda k: (-1)**k * mp.power(mp.pi, 2 * k) / mp.gamma(2 * k + 1) * 
                  mp.nsum(lambda j: (-1)**j * mp.power(2, 2 * j) * mp.binomial(2 * k, 2 * j) * 
                            mp.cos(mp.pi * (2 * j + 1) / 4), [0, k]), [0, 3])
    
    term = (mp.power(-1, N - 1) * mp.power(tau, -1/4) * 
            (cos_phi + sin_phi) / (mp.sqrt(2) * mp.pi))
    
    if abs(term) > tol / 10 and n_terms < max_terms:
        remainder += term
        n_terms += 1
    
    return remainder


def riemann_siegel_Z(t: mp.mpf, tol: Optional[float] = None) -> mp.mpf:
    if t == 0:
        return mp.mpf('0')
    
    if tol is None:
        tol = mp.mpf('1e-12')
    else:
        tol = mp.mpf(str(tol))
    
    tau = t / (2 * mp.pi)
    sqrt_tau = mp.sqrt(tau)
    N = int(mp.floor(sqrt_tau))
    
    if N == 0:
        N = 1
    
    theta = riemann_siegel_theta(t, order=5)
    
    main_sum = mp.mpf('0')
    prev_term = mp.mpf('0')
    
    for n in range(1, N + 1):
        n_mp = mp.mpf(str(n))
        term = mp.cos(theta - t * mp.log(n_mp)) / mp.sqrt(n_mp)
        main_sum += term
        
        if n > 1 and abs(term) < tol / (100 * N):
            if abs(prev_term) < tol / (100 * N):
                break
        
        prev_term = term
    
    remainder = riemann_siegel_remainder(t, N, tol)
    
    return 2 * main_sum + remainder


def zeta_on_critical_line(t: mp.mpf) -> mp.mpf:
    s = mp.mpf('0.5') + 1j * t
    return mp.zeta(s)


def compute_Z_via_zeta(t: mp.mpf) -> mp.mpf:
    s = mp.mpf('0.5') + 1j * t
    zeta_val = mp.zeta(s)
    theta = riemann_siegel_theta(t)
    return mp.re(zeta_val * mp.e**(1j * theta))


def count_zeros_up_to(T: mp.mpf) -> int:
    T_mp = mp.mpf(str(T))
    if T_mp < 10:
        T_mp = mp.mpf('10')
    N = (T_mp / (2 * mp.pi)) * mp.log(T_mp / (2 * mp.pi)) - T_mp / (2 * mp.pi) + 7 / 8
    S_T = (1 / mp.pi) * mp.arg(mp.zeta(mp.mpf('0.5') + 1j * T_mp))
    return int(mp.floor(N + S_T))


def adaptive_bracket_search(
    t_start: mp.mpf, 
    t_end: mp.mpf, 
    initial_step: mp.mpf = mp.mpf('0.3'),
    tol: float = 1e-12
) -> List[Tuple[mp.mpf, mp.mpf]]:
    brackets = []
    t = t_start
    
    def Z_func(t_val):
        return compute_Z_via_zeta(t_val)
    
    prev_Z = Z_func(t)
    prev_t = t
    step = initial_step
    
    while t < t_end:
        t += step
        current_Z = Z_func(t)
        
        if prev_Z * current_Z < 0:
            if step > mp.mpf('0.05'):
                t_back = prev_t
                step_small = step / 10
                Z_back = prev_Z
                while t_back < t - step_small / 2:
                    t_back += step_small
                    Z_back_new = Z_func(t_back)
                    if Z_back * Z_back_new < 0:
                        brackets.append((t_back - step_small, t_back))
                        break
                    Z_back = Z_back_new
            else:
                brackets.append((prev_t, t))
        
        elif abs(current_Z) < mp.mpf('1e-10'):
            if t - prev_t > mp.mpf('0.1'):
                brackets.append((prev_t, t))
        
        prev_Z = current_Z
        prev_t = t
    
    return brackets


def refine_zero_bisection(
    t_low: mp.mpf, 
    t_high: mp.mpf, 
    max_iter: int = 200,
    tol: float = 1e-14
) -> mp.mpf:
    tol_mp = mp.mpf(str(tol))
    
    def Z_func(t_val):
        return compute_Z_via_zeta(t_val)
    
    Z_low = Z_func(t_low)
    Z_high = Z_func(t_high)
    
    if Z_low * Z_high > 0:
        try:
            root = mp.findroot(Z_func, (t_low, t_high))
            return root
        except:
            pass
    
    for _ in range(max_iter):
        t_mid = (t_low + t_high) / 2
        Z_mid = Z_func(t_mid)
        
        if abs(Z_mid) < tol_mp or (t_high - t_low) < tol_mp:
            return t_mid
        
        if Z_low * Z_mid < 0:
            t_high = t_mid
            Z_high = Z_mid
        else:
            t_low = t_mid
            Z_low = Z_mid
    
    return (t_low + t_high) / 2


def verify_zero(t: mp.mpf, tol: float = 1e-10) -> Tuple[bool, float]:
    s = mp.mpf('0.5') + 1j * t
    zeta_val = mp.zeta(s)
    zeta_abs = abs(zeta_val)
    return zeta_abs < tol, float(zeta_abs)


def compute_zeta_zeros(
    t_min: float, 
    t_max: float, 
    precision: int = 60,
    zero_tol: float = 1e-12,
    verbose: bool = True
) -> List[Tuple[float, float, float]]:
    mp.mp.dps = precision
    
    t_min_mp = mp.mpf(str(t_min))
    t_max_mp = mp.mpf(str(t_max))
    
    if verbose:
        print("=" * 80)
        print(f"黎曼Zeta函数非平凡零点计算 (高精度版)")
        print("=" * 80)
        print(f"虚部范围: [{t_min}, {t_max}]")
        print(f"计算精度: {precision} 位小数")
        print(f"零点容差: {zero_tol}")
        print("-" * 80)
    
    brackets = adaptive_bracket_search(t_min_mp, t_max_mp, initial_step=mp.mpf('0.25'))
    
    if verbose:
        print(f"找到 {len(brackets)} 个零点候选区间")
        print("-" * 80)
    
    results = []
    for i, (t_low, t_high) in enumerate(brackets, 1):
        zero_t = refine_zero_bisection(t_low, t_high, tol=zero_tol)
        zero_t_float = float(zero_t)
        
        is_valid, zeta_abs = verify_zero(zero_t, tol=1e-8)
        
        if is_valid or zeta_abs < 1e-6:
            results.append((0.5, zero_t_float, zeta_abs))
            
            if verbose:
                status = "✓" if is_valid else "~"
                print(f"零点 #{i:3d} {status}: Re(s)=0.5, Im(s)={zero_t_float:14.10f}, |ζ(s)|={zeta_abs:.2e}")
        elif verbose:
            print(f"零点 #{i:3d} ✗: 跳过无效候选 Im(s)≈{zero_t_float:.8f}, |ζ(s)|={zeta_abs:.2e}")
    
    results.sort(key=lambda x: x[1])
    
    if verbose:
        print("-" * 80)
        print(f"计算完成! 共确认 {len(results)} 个有效非平凡零点")
        print("=" * 80)
    
    return results


def compare_with_known_zeros(results: List[Tuple[float, float, float]]) -> None:
    known_zeros = [
        14.1347251417346937904572519835624702707842571156992,
        21.0220396387715549926284795938969027773343405249028,
        25.0108575801456887632137909925628218186595496725579,
        30.4248761258595132103118975305840913201815600237154,
        32.9350615877391896906623689640749034888127156058259,
        37.5861781588256712572177634807053328214542145457877,
        40.9187190121474951873981269146332543957261659627777,
        43.3270732809149995194961221654068058576353168559688,
        48.0051508813347679308532938657119858418512106957808,
        49.7738324776723021819167846785639997526221522390171
    ]
    
    print("\n与已知零点的精度对比 (前10个):")
    print("-" * 70)
    print(f"{'序号':<4} {'计算值':<20} {'误差':<15} {'|ζ(s)|':<12}")
    print("-" * 70)
    
    for i, known in enumerate(known_zeros[:min(len(results), 10)]):
        if i < len(results):
            calc = results[i][1]
            error = abs(calc - known)
            zeta_abs = results[i][2]
            print(f"{i+1:<4} {calc:<20.12f} {error:<15.2e} {zeta_abs:<12.2e}")
    
    print("-" * 70)


def save_results(
    results: List[Tuple[float, float, float]], 
    filename: str = "zeta_zeros.txt",
    t_min: Optional[float] = None,
    t_max: Optional[float] = None
):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("黎曼Zeta函数非平凡零点计算结果 (高精度版)\n")
        f.write("=" * 70 + "\n")
        if t_min is not None and t_max is not None:
            f.write(f"计算范围: Im(s) ∈ [{t_min}, {t_max}]\n")
        f.write(f"零点数量: {len(results)}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'序号':<6} {'实部':<14} {'虚部':<20} {'|ζ(s)|':<15}\n")
        f.write("-" * 70 + "\n")
        for i, (real, imag, zeta_abs) in enumerate(results, 1):
            f.write(f"{i:<6} {real:<14.10f} {imag:<20.12f} {zeta_abs:<15.2e}\n")
    print(f"\n结果已保存到: {filename}")


def main():
    T_MIN = 0
    T_MAX = 100
    PRECISION = 60
    ZERO_TOL = 1e-12
    
    zeros = compute_zeta_zeros(T_MIN, T_MAX, PRECISION, ZERO_TOL)
    
    if zeros:
        compare_with_known_zeros(zeros)
    
    save_results(zeros, "zeta_zeros_0_100_high_precision.txt", T_MIN, T_MAX)
    
    print("\n前10个零点 (高精度):")
    print("-" * 50)
    for i, (real, imag, _) in enumerate(zeros[:10], 1):
        print(f"γ_{i:<2d} = {imag:.12f}")


if __name__ == "__main__":
    main()
