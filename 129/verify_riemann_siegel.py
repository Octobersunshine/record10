import mpmath as mp
from riemann_zeta_zeros import (
    riemann_siegel_theta,
    riemann_siegel_Z,
    compute_Z_via_zeta,
    count_zeros_up_to
)


def test_theta_accuracy():
    print("=" * 70)
    print("测试 Theta 函数精度")
    print("=" * 70)
    
    test_values = [14.1347, 21.0220, 50.0, 100.0, 200.0]
    
    for t_val in test_values:
        t = mp.mpf(str(t_val))
        theta1 = riemann_siegel_theta(t, order=1)
        theta5 = riemann_siegel_theta(t, order=5)
        theta9 = riemann_siegel_theta(t, order=9)
        
        print(f"\nt = {t_val}:")
        print(f"  1阶修正:  {theta1}")
        print(f"  5阶修正:  {theta5}")
        print(f"  9阶修正:  {theta9}")
        print(f"  差值 (5-1): {abs(theta5 - theta1):.2e}")
        print(f"  差值 (9-5): {abs(theta9 - theta5):.2e}")
    
    print()


def test_Z_function_accuracy():
    print("=" * 70)
    print("测试 Z 函数精度 (黎曼-西格尔 vs 直接计算)")
    print("=" * 70)
    
    mp.mp.dps = 50
    
    test_points = [
        10.0, 14.1347, 21.0220, 25.0109, 
        30.4249, 50.0, 75.0, 100.0
    ]
    
    print(f"\n{'t':<12} {'Z_rs':<20} {'Z_direct':<20} {'误差':<15}")
    print("-" * 70)
    
    for t_val in test_points:
        t = mp.mpf(str(t_val))
        
        Z_rs = riemann_siegel_Z(t, tol=1e-12)
        Z_direct = compute_Z_via_zeta(t)
        
        error = abs(Z_rs - Z_direct)
        
        print(f"{t_val:<12.4f} {float(Z_rs):<20.12f} {float(Z_direct):<20.12f} {float(error):<15.2e}")
    
    print()


def test_zero_count():
    print("=" * 70)
    print("测试零点计数公式")
    print("=" * 70)
    
    mp.mp.dps = 40
    
    test_T = [10, 50, 100, 200]
    
    for T in test_T:
        T_mp = mp.mpf(str(T))
        count = count_zeros_up_to(T_mp)
        
        expected = {
            10: 0,
            50: 10,
            100: 29,
            200: 79
        }.get(T, "?")
        
        print(f"T = {T:4d}: 计算得到 N(T) = {count:3d}, 预期约 {expected}")
    
    print()


def test_zeros_at_known_points():
    print("=" * 70)
    print("验证已知零点处的 Z 函数值")
    print("=" * 70)
    
    mp.mp.dps = 60
    
    known_zeros = [
        14.134725141734693790457251983562,
        21.022039638771554992628479593897,
        25.010857580145688763213790992563,
        30.424876125859513210311897530584,
    ]
    
    print(f"\n{'零点':<25} {'Z(γ)':<20} {'|ζ(0.5+iγ)|':<20}")
    print("-" * 70)
    
    for gamma in known_zeros:
        t = mp.mpf(str(gamma))
        
        Z_val = compute_Z_via_zeta(t)
        s = mp.mpf('0.5') + 1j * t
        zeta_abs = abs(mp.zeta(s))
        
        print(f"{gamma:<25.12f} {float(Z_val):<20.2e} {float(zeta_abs):<20.2e}")
    
    print()


def test_adaptive_truncation():
    print("=" * 70)
    print("测试自适应截断效果")
    print("=" * 70)
    
    mp.mp.dps = 50
    
    t_values = [50.0, 100.0, 200.0]
    tolerances = [1e-6, 1e-9, 1e-12]
    
    for t_val in t_values:
        t = mp.mpf(str(t_val))
        print(f"\nt = {t_val}:")
        
        Z_exact = compute_Z_via_zeta(t)
        print(f"  精确值 (直接计算): {Z_exact}")
        
        for tol in tolerances:
            Z_rs = riemann_siegel_Z(t, tol=tol)
            error = abs(Z_rs - Z_exact)
            print(f"  容差 {tol:<8} -> 误差: {float(error):.2e}")
    
    print()


def main():
    test_theta_accuracy()
    test_Z_function_accuracy()
    test_zero_count()
    test_zeros_at_known_points()
    test_adaptive_truncation()
    
    print("=" * 70)
    print("精度验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
