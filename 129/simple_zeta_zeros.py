import mpmath as mp
from typing import List, Tuple


def find_zeta_zeros(t_min: float, t_max: float, precision: int = 30) -> List[Tuple[float, float]]:
    mp.mp.dps = precision
    
    print(f"计算黎曼Zeta函数在 Im(s) ∈ [{t_min}, {t_max}] 内的非平凡零点")
    print(f"精度: {precision} 位小数")
    print("-" * 60)
    
    def Z(t):
        s = mp.mpf('0.5') + 1j * t
        return mp.re(mp.zeta(s) * mp.e**(1j * mp.arg(mp.gamma(s/2)) - (t/2) * mp.log(mp.pi)))
    
    zeros = []
    t = mp.mpf(str(t_min))
    step = mp.mpf('0.3')
    
    prev_z = Z(t)
    prev_t = t
    
    while t < t_max:
        t += step
        current_z = Z(t)
        
        if prev_z * current_z < 0:
            try:
                root = mp.findroot(Z, (prev_t, t))
                s = mp.mpf('0.5') + 1j * root
                zeta_val = mp.zeta(s)
                
                if abs(zeta_val) < 1e-6:
                    zeros.append((0.5, float(root)))
                    print(f"零点 #{len(zeros)}: Re(s) = 0.5, Im(s) = {float(root):.8f}, |ζ| = {abs(zeta_val):.2e}")
            except:
                pass
        
        prev_z = current_z
        prev_t = t
    
    print("-" * 60)
    print(f"共找到 {len(zeros)} 个非平凡零点")
    
    return zeros


def find_zeros_xi_method(t_min: float, t_max: float, precision: int = 40) -> List[Tuple[float, float]]:
    mp.mp.dps = precision
    
    print(f"使用Xi函数方法计算零点...")
    print("-" * 60)
    
    def xi(t):
        s = mp.mpf('0.5') + 1j * t
        return mp.re(mp.zeta(s) * mp.gamma(s/2) * mp.power(mp.pi, -s/2) * (s - 1) / 2)
    
    zeros = []
    t = mp.mpf(str(t_min))
    step = mp.mpf('0.4')
    
    prev_xi = xi(t)
    prev_t = t
    
    while t < t_max:
        t += step
        current_xi = xi(t)
        
        if prev_xi * current_xi < 0:
            try:
                root = mp.findroot(xi, (prev_t, t))
                s = mp.mpf('0.5') + 1j * root
                zeta_val = mp.zeta(s)
                
                if abs(zeta_val) < 1e-5:
                    zeros.append((0.5, float(root)))
                    print(f"零点 #{len(zeros)}: Im(s) = {float(root):.8f}")
            except:
                pass
        
        prev_xi = current_xi
        prev_t = t
    
    return zeros


if __name__ == "__main__":
    T_MIN = 0
    T_MAX = 100
    
    zeros = find_zeros_xi_method(T_MIN, T_MAX)
    
    with open("zeta_zeros_result.txt", "w") as f:
        f.write("黎曼Zeta函数非平凡零点\n")
        f.write("=" * 40 + "\n")
        f.write(f"范围: Im(s) 从 {T_MIN} 到 {T_MAX}\n")
        f.write("=" * 40 + "\n")
        for i, (re, im) in enumerate(zeros, 1):
            f.write(f"{i}: s = {re} + {im:.8f}i\n")
    
    print(f"\n结果已保存到 zeta_zeros_result.txt")
    print(f"\n前5个零点:")
    for i, (re, im) in enumerate(zeros[:5], 1):
        print(f"  γ_{i} = {im:.6f}")
