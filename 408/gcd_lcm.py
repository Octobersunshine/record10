from functools import reduce
from typing import List, Tuple


def gcd(a: int, b: int, return_steps: bool = False) -> int | Tuple[int, List[Tuple[int, int, int, int]]]:
    a_orig, b_orig = a, b
    a, b = abs(a), abs(b)
    steps = []
    while b:
        q, r = divmod(a, b)
        steps.append((a, b, q, r))
        a, b = b, r
    if return_steps:
        return a, steps
    return a


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return abs(a), 1 if a >= 0 else -1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def gcd_multi(*args: int) -> int:
    if len(args) == 0:
        raise ValueError("至少需要一个参数")
    return reduce(gcd, args)


def lcm_multi(*args: int) -> int:
    if len(args) == 0:
        raise ValueError("至少需要一个参数")
    return reduce(lcm, args)


def compute_gcd_lcm(a: int, b: int) -> Tuple[int, int]:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("请输入整数")
    g = gcd(a, b)
    l = lcm(a, b)
    return g, l


def print_steps(steps: List[Tuple[int, int, int, int]], a_orig: int, b_orig: int) -> None:
    print(f"GCD({a_orig}, {b_orig}) 的辗转相除过程:")
    for i, (a, b, q, r) in enumerate(steps, 1):
        print(f"  步骤{i}: {a} = {b} × {q} + {r}")
    if steps:
        print(f"  结果: GCD = {steps[-1][1]}")
    else:
        print(f"  结果: GCD = {abs(a_orig)}")


if __name__ == "__main__":
    try:
        a = int(input("请输入第一个整数: "))
        b = int(input("请输入第二个整数: "))

        g, steps = gcd(a, b, return_steps=True)
        l = lcm(a, b)
        print(f"\n最大公约数 (GCD): {g}")
        print(f"最小公倍数 (LCM): {l}")
        print()
        print_steps(steps, a, b)

        g_ext, x, y = extended_gcd(a, b)
        print(f"\n扩展欧几里得算法: {a}×({x}) + {b}×({y}) = {g_ext}")
        print(f"验证: {a}×{x} + {b}×{y} = {a * x + b * y}")

        print(f"\n多参数测试:")
        print(f"gcd_multi(12, 18, 24, 30) = {gcd_multi(12, 18, 24, 30)}")
        print(f"lcm_multi(12, 18, 24, 30) = {lcm_multi(12, 18, 24, 30)}")

    except ValueError as e:
        print(f"错误: {e}")
