import numpy as np
from discrete_calculus import (
    forward_difference,
    backward_difference,
    central_difference,
    cumulative_trapezoidal_integral,
    cumulative_simpson_integral
)


def test_equally_spaced():
    print("=" * 60)
    print("测试1: 等间距序列 (向后兼容)")
    print("=" * 60)

    h = 0.1
    x = np.arange(0, 1.0 + h, h)
    y = x ** 2
    n = len(y)

    print(f"\n原序列 y = x^2, n = {n}, h = {h}")

    print("\n--- truncate 模式 ---")
    fd = forward_difference(y, h=h, boundary='truncate')
    bd = backward_difference(y, h=h, boundary='truncate')
    cd = central_difference(y, h=h, boundary='truncate')
    print(f"前向差分长度: {len(fd)} (预期 n-1 = {n-1}) ✓" if len(fd) == n-1 else "✗")
    print(f"后向差分长度: {len(bd)} (预期 n-1 = {n-1}) ✓" if len(bd) == n-1 else "✗")
    print(f"中心差分长度: {len(cd)} (预期 n-2 = {n-2}) ✓" if len(cd) == n-2 else "✗")
    print(f"中心差分内部点误差: {np.max(np.abs(cd - 2 * x[1:-1])):.6f}")

    print("\n--- fill 模式 (使用 x 参数) ---")
    cd_fill = central_difference(y, x=x, boundary='fill')
    print(f"中心差分(fill)长度: {len(cd_fill)} (预期 n = {n}) ✓" if len(cd_fill) == n else "✗")
    print(f"首尾是否nan: {np.isnan(cd_fill[0]) and np.isnan(cd_fill[-1])}")


def test_non_equally_spaced():
    print("\n" + "=" * 60)
    print("测试2: 非等间距序列")
    print("=" * 60)

    x = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5])
    y = x ** 2
    dy_theory = 2 * x

    print(f"\n自定义时间轴 x = {x}")
    print(f"时间间隔 dx = {x[1:] - x[:-1]}")
    print(f"y = x^2, 理论导数 = 2x = {dy_theory}")

    print("\n--- 前向差分 ---")
    fd = forward_difference(y, x=x, boundary='truncate')
    print(f"结果: {fd}")
    print(f"理论值: {(y[1:] - y[:-1]) / (x[1:] - x[:-1])}")
    print(f"误差: {np.max(np.abs(fd - (y[1:] - y[:-1]) / (x[1:] - x[:-1]))):.6f}")

    print("\n--- 后向差分 ---")
    bd = backward_difference(y, x=x, boundary='truncate')
    print(f"结果: {bd}")
    print(f"误差: {np.max(np.abs(bd - (y[1:] - y[:-1]) / (x[1:] - x[:-1]))):.6f}")

    print("\n--- 中心差分 (非等间距公式) ---")
    cd = central_difference(y, x=x, boundary='truncate')
    print(f"结果: {cd}")
    print(f"理论导数(内部点): {dy_theory[1:-1]}")
    print(f"最大误差: {np.max(np.abs(cd - dy_theory[1:-1])):.6f}")

    print("\n--- boundary='extrapolate' 模式 ---")
    cd_ext = central_difference(y, x=x, boundary='extrapolate')
    print(f"中心差分结果: {cd_ext}")
    print(f"理论导数:     {dy_theory}")


def test_integrals():
    print("\n" + "=" * 60)
    print("测试3: 积分功能 (梯形 vs 辛普森)")
    print("=" * 60)

    h = 0.1
    x = np.arange(0, 1.0 + h, h)
    y = np.sin(x)
    theory = 1 - np.cos(x)

    print(f"\n测试函数: y = sin(x), h = {h}, n = {len(x)}")
    print(f"理论积分终值: {theory[-1]:.8f}")

    trap_integral = cumulative_trapezoidal_integral(y, h=h)
    simp_integral = cumulative_simpson_integral(y, h=h)

    print(f"\n梯形积分终值: {trap_integral[-1]:.8f}, 误差: {abs(trap_integral[-1] - theory[-1]):.8f}")
    print(f"辛普森积分终值: {simp_integral[-1]:.8f}, 误差: {abs(simp_integral[-1] - theory[-1]):.8f}")
    print(f"辛普森积分精度提升: {(abs(trap_integral[-1] - theory[-1]) / abs(simp_integral[-1] - theory[-1]) if abs(simp_integral[-1] - theory[-1]) > 1e-15 else float('inf')):.1f}x")

    print("\n--- 非等间距积分 ---")
    x_unequal = np.array([0.0, 0.1, 0.25, 0.45, 0.7, 1.0])
    y_unequal = np.sin(x_unequal)
    theory_unequal = 1 - np.cos(x_unequal)
    trap_unequal = cumulative_trapezoidal_integral(y_unequal, x=x_unequal)
    simp_unequal = cumulative_simpson_integral(y_unequal, x=x_unequal)

    print(f"非等间距 x = {x_unequal}")
    print(f"理论积分终值: {theory_unequal[-1]:.8f}")
    print(f"梯形积分终值: {trap_unequal[-1]:.8f}, 误差: {abs(trap_unequal[-1] - theory_unequal[-1]):.8f}")
    print(f"辛普森积分终值: {simp_unequal[-1]:.8f} (非等间距退化为梯形)")


def test_boundary_options():
    print("\n" + "=" * 60)
    print("测试4: 边界处理选项 (非等间距下)")
    print("=" * 60)

    x = np.array([0.0, 0.2, 0.5, 0.9, 1.4])
    y = x ** 3

    print(f"\n原序列: x={x}, y={y}")

    print("\n--- truncate ---")
    fd_t = forward_difference(y, x=x, boundary='truncate')
    cd_t = central_difference(y, x=x, boundary='truncate')
    print(f"前向差分: {fd_t}, 长度={len(fd_t)}")
    print(f"中心差分: {cd_t}, 长度={len(cd_t)}")

    print("\n--- fill (fill_value=-999) ---")
    fd_f = forward_difference(y, x=x, boundary='fill', fill_value=-999)
    cd_f = central_difference(y, x=x, boundary='fill', fill_value=-999)
    print(f"前向差分: {fd_f}, 长度={len(fd_f)}")
    print(f"中心差分: {cd_f}, 长度={len(cd_f)}")

    print("\n--- extrapolate ---")
    fd_e = forward_difference(y, x=x, boundary='extrapolate')
    cd_e = central_difference(y, x=x, boundary='extrapolate')
    print(f"前向差分: {fd_e}, 长度={len(fd_e)}")
    print(f"中心差分: {cd_e}, 长度={len(cd_e)}")


def test_error_handling():
    print("\n" + "=" * 60)
    print("测试5: 错误处理")
    print("=" * 60)

    print("\n--- x与y长度不匹配 ---")
    try:
        forward_difference([1, 2, 3], x=[0, 1])
        print("✗ 未报错")
    except ValueError as e:
        print(f"✓ 正确抛出: {e}")

    print("\n--- x非严格递增 ---")
    try:
        forward_difference([1, 2, 3], x=[0, 1, 0.5])
        print("✗ 未报错")
    except ValueError as e:
        print(f"✓ 正确抛出: {e}")

    print("\n--- 无效 boundary ---")
    try:
        forward_difference([1, 2, 3], boundary='invalid')
        print("✗ 未报错")
    except ValueError as e:
        print(f"✓ 正确抛出: {e}")

    print("\n--- 辛普森积分长度不足 ---")
    try:
        cumulative_simpson_integral([1, 2])
        print("✗ 未报错")
    except ValueError as e:
        print(f"✓ 正确抛出: {e}")


def test_cumulative_integral_values():
    print("\n" + "=" * 60)
    print("测试6: 累计积分值验证")
    print("=" * 60)

    h = 0.1
    x = np.arange(0, 1.0 + h, h)
    y = x ** 2
    theory = x ** 3 / 3.0

    trap = cumulative_trapezoidal_integral(y, h=h)
    simp = cumulative_simpson_integral(y, h=h)

    print(f"\ny = x^2, 积分理论值 = x^3/3")
    print(f"x       理论值      梯形        辛普森")
    for i in [0, 2, 5, 8, 10]:
        print(f"{x[i]:.1f}     {theory[i]:.6f}    {trap[i]:.6f}    {simp[i]:.6f}")

    print(f"\n梯形积分最大误差: {np.max(np.abs(trap - theory)):.6f}")
    print(f"辛普森积分最大误差: {np.max(np.abs(simp - theory)):.6f}")


if __name__ == "__main__":
    test_equally_spaced()
    test_non_equally_spaced()
    test_integrals()
    test_boundary_options()
    test_error_handling()
    test_cumulative_integral_values()
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
