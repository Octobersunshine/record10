import numpy as np
from main import (
    talbot_inversion, talbot_inversion_enhanced, 
    talbot_inversion_mp, talbot_inversion_extreme,
    get_adaptive_params
)

def compare_precision_levels():
    """比较不同精度级别的效果"""
    print("=" * 90)
    print("多精度级别对比测试: F(s) = 1/(s+1), f(t) = e^(-t)")
    print("=" * 90)
    
    F_s = "1/(s+1)"
    t_values = [0.1, 1.0, 10.0, 50.0]
    
    print(f"{'t':>8} {'精度级别':>15} {'f(t)真实值':>15} {'计算值':>15} {'误差':>15} {'N':>6}")
    print("-" * 90)
    
    for t in t_values:
        f_true = np.exp(-t)
        
        f_double, N_double = talbot_inversion_enhanced(F_s, t)
        err_double = abs(f_double - f_true)
        print(f"{t:>8.2f} {'double(64位)':>15} {f_true:>15.10f} {f_double:>15.10f} {err_double:>15.2e} {N_double:>6}")
        
        f_high, N_high = talbot_inversion_mp(F_s, t, prec=50)
        err_high = abs(f_high - f_true)
        print(f"{' ':>8} {'high(50位)':>15} {' ':>15} {f_high:>15.10f} {err_high:>15.2e} {N_high:>6}")
        
        f_extreme, N_extreme = talbot_inversion_extreme(F_s, t)
        err_extreme = abs(f_extreme - f_true)
        print(f"{' ':>8} {'extreme(100位)':>15} {' ':>15} {f_extreme:>15.10f} {err_extreme:>15.2e} {N_extreme:>6}")
        print("-" * 90)
    print()

def test_exponential_high_precision():
    """高精度测试指数函数: F(s) = 1/(s+1), f(t) = e^(-t)"""
    print("=" * 90)
    print("高精度测试指数函数: F(s) = 1/(s+1), f(t) = e^(-t) [50位精度]")
    print("=" * 90)
    
    F_s = "1/(s+1)"
    t_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    print(f"{'t':>10} {'f(t)真实值':>20} {'高精度计算':>20} {'误差':>15} {'N':>6}")
    print("-" * 90)
    
    for t in t_values:
        f_true = np.exp(-t)
        f_pred, N_used = talbot_inversion_mp(F_s, t, prec=50)
        error = abs(f_pred - f_true)
        print(f"{t:>10.2f} {f_true:>20.12f} {f_pred:>20.12f} {error:>15.2e} {N_used:>6}")
    print()

def test_step_function_extreme():
    """极端精度测试阶跃函数: F(s) = 1/s, f(t) = 1"""
    print("=" * 90)
    print("极端精度测试阶跃函数: F(s) = 1/s, f(t) = 1 [100位精度+三重外推]")
    print("=" * 90)
    
    F_s = "1/s"
    t_values = [0.1, 1.0, 10.0, 50.0]
    
    print(f"{'t':>10} {'f(t)真实值':>20} {'极端精度计算':>20} {'误差':>15} {'N':>6}")
    print("-" * 90)
    
    for t in t_values:
        f_true = 1.0
        f_pred, N_used = talbot_inversion_extreme(F_s, t)
        error = abs(f_pred - f_true)
        print(f"{t:>10.2f} {f_true:>20.12f} {f_pred:>20.12f} {error:>15.2e} {N_used:>6}")
    print()

def show_adaptive_params():
    """显示不同t值对应的自适应参数"""
    print("=" * 70)
    print("自适应参数表")
    print("=" * 70)
    
    t_values = [0.05, 0.5, 5.0, 50.0, 200.0]
    print(f"{'t':>10} {'N':>6} {'c1':>8} {'c2':>8} {'c3':>8} {'c4':>8} {'shift':>8}")
    print("-" * 70)
    
    for t in t_values:
        N, c1, c2, c3, c4, shift = get_adaptive_params(t)
        print(f"{t:>10.2f} {N:>6} {c1:>8.4f} {c2:>8.4f} {c3:>8.4f} {c4:>8.4f} {shift:>8.4f}")
    print()

def show_precision_summary():
    """显示精度级别摘要"""
    print("=" * 70)
    print("支持的精度级别")
    print("=" * 70)
    print("  double (默认) : 64位双精度浮点数, 快速计算")
    print("  high          : 50位多精度计算 (mpmath)")
    print("  extreme       : 100位多精度 + 三重外推, 最高精度")
    print()

if __name__ == "__main__":
    show_adaptive_params()
    show_precision_summary()
    compare_precision_levels()
    test_exponential_high_precision()
    test_step_function_extreme()
