import random
import math


def check_fermat(x, y, z, n):
    if n <= 2:
        raise ValueError("费马大定理仅适用于 n > 2 的情况")
    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError("x, y, z 必须为正整数")
    if z <= max(x, y):
        return False
    return pow(x, n) + pow(y, n) == pow(z, n)


def _find_z_upper_bound(x, y, n, max_value):
    x_pow = pow(x, n)
    y_pow = pow(y, n)
    sum_pow = x_pow + y_pow
    
    z_min = max(x, y) + 1
    z_max = min(max_value, int(sum_pow ** (1.0 / n)) + 1)
    
    while z_max >= z_min and pow(z_max, n) > sum_pow:
        z_max -= 1
    
    return z_min, z_max, x_pow, y_pow


def brute_force_fermat(max_value=100, min_n=3, max_n=5, show_progress=True):
    solutions = []
    total_iterations = 0
    skipped_iterations = 0
    
    for n in range(min_n, max_n + 1):
        if show_progress:
            print(f"  正在搜索 n={n}...")
        
        for x in range(1, max_value + 1):
            for y in range(x, max_value + 1):
                total_iterations += 1
                
                z_min, z_max, x_pow, y_pow = _find_z_upper_bound(x, y, n, max_value)
                sum_pow = x_pow + y_pow
                
                if z_max < z_min:
                    skipped_iterations += (max_value - max(x, y))
                    continue
                
                for z in range(z_min, z_max + 1):
                    total_iterations += 1
                    z_pow = pow(z, n)
                    
                    if z_pow == sum_pow:
                        solutions.append((x, y, z, n))
                        if x != y:
                            solutions.append((y, x, z, n))
                    elif z_pow > sum_pow:
                        break
        
        if show_progress:
            print(f"    n={n} 搜索完成")
    
    if show_progress:
        original_total = (max_n - min_n + 1) * max_value ** 3
        print(f"\n  性能统计:")
        print(f"    原始理论迭代次数: {original_total:,}")
        print(f"    实际迭代次数: {total_iterations:,}")
        print(f"    优化率: {(1 - total_iterations / original_total) * 100:.2f}%")
    
    return solutions


def generate_pythagorean_triples(max_value):
    triples = []
    for m in range(2, int(math.isqrt(max_value)) + 1):
        for n in range(1, m):
            if (m - n) % 2 == 0 or math.gcd(m, n) != 1:
                continue
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            if c > max_value:
                break
            triples.append((a, b, c))
            k = 2
            while k * c <= max_value:
                triples.append((k * a, k * b, k * c))
                k += 1
    triples.sort(key=lambda t: t[2])
    return triples


def is_prime_fermat(n, k=5):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    for _ in range(k):
        a = random.randrange(2, n)
        if pow(a, n - 1, n) != 1:
            return False
    return True


def is_prime_deterministic(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def fermat_history():
    history = [
        ("1637年", "费马在读巴歇校订的《算术》时，在页边写下著名的批注："
         "'我发现了一个美妙的证明，可惜这里的空白太小，写不下。'"
         " 这标志着费马大定理的诞生。"),
        ("1753年", "欧拉证明了 n=3 的情况，是费马之后第一个给出严格证明的数学家。"
         " 他使用了无限递降法。"),
        ("1825年", "勒让德和狄利克雷独立证明了 n=5 的情况。"
         " 狄利克雷当时年仅20岁。"),
        ("1832年", "拉梅证明了 n=7 的情况。"),
        ("1847年", "库默尔证明了对于所有'正则素数'，费马大定理成立。"
         " 这一结果覆盖了大量但非全部的指数。"),
        ("1955年", "谷山丰提出'谷山-志村猜想'，建立了椭圆曲线与模形式之间的联系。"
         " 这一猜想后来成为证明费马大定理的关键桥梁。"),
        ("1986年", "里贝特证明：如果谷山-志村猜想成立，则费马大定理成立。"
         " 这将费马大定理的证明归结为证明半稳定的椭圆曲线是模曲线。"),
        ("1993年", "安德鲁·怀尔斯在剑桥大学做了为期三天的系列演讲，"
         " 在最后一天宣布证明了费马大定理，震惊数学界。"),
        ("1994年", "怀尔斯与泰勒合作修补了证明中的一个漏洞。"
         " 完整的证明发表于1995年，长达129页。"
         " 证明使用了模形式、椭圆曲线、伽罗瓦表示等现代数学工具。"),
        ("2016年", "怀尔斯因证明费马大定理被授予阿贝尔奖，"
         " 这是数学界的最高荣誉之一。"),
    ]
    return history


def _demonstrate_float_precision_issue():
    print("\n  浮点数精度问题演示:")
    print(f"    使用 ** 运算符: 9999999999999999**2 = {9999999999999999 ** 2}")
    print(f"    使用 pow() 函数: pow(9999999999999999, 2) = {pow(9999999999999999, 2)}")
    print(f"    使用浮点数: (9999999999999999.0)**2 = {int(9999999999999999.0 ** 2)}")
    print(f"    注意: 整数幂运算保证精确，浮点数运算可能丢失精度")


def _demonstrate_large_number_correctness():
    print("\n  大整数精确性验证:")
    x, y, z, n = 100, 100, 126, 3
    xn = pow(x, n)
    yn = pow(y, n)
    zn = pow(z, n)
    print(f"    {x}^{n} + {y}^{n} = {xn} + {yn} = {xn + yn}")
    print(f"    {z}^{n} = {zn}")
    print(f"    是否相等: {xn + yn == zn}")
    
    print(f"\n  范围限制验证:")
    x, y, n = 50, 60, 3
    z_min, z_max, x_pow, y_pow = _find_z_upper_bound(x, y, n, 200)
    print(f"    x={x}, y={y}, n={n}:")
    print(f"    x^{n} = {x_pow}, y^{n} = {y_pow}, 和 = {x_pow + y_pow}")
    print(f"    z 搜索范围: [{z_min}, {z_max}]")
    print(f"    原始范围: [{max(x, y) + 1}, 200]")
    print(f"    减少搜索量: {200 - max(x, y) - (z_max - z_min + 1)} 个数")


if __name__ == "__main__":
    print("=== 费马大定理验证工具 (优化版) ===\n")
    
    print("--- 1. 费马大定理历史证明简述 ---")
    for year, desc in fermat_history():
        print(f"  [{year}] {desc}")
    
    print("\n--- 2. 勾股数生成 (n=2, x,y,z ≤ 100) ---")
    triples = generate_pythagorean_triples(100)
    print(f"  共找到 {len(triples)} 组勾股数:")
    for i, (a, b, c) in enumerate(triples[:10]):
        print(f"    {a}² + {b}² = {c}²  (验证: {a*a} + {b*b} = {c*c})")
    if len(triples) > 10:
        print(f"    ... 还有 {len(triples) - 10} 组")
    
    print("\n--- 3. 费马小定理质数判定 ---")
    test_numbers = [2, 3, 7, 11, 13, 15, 17, 21, 25, 37, 41, 561, 997]
    print(f"  {'数字':>6}  {'费马判定':>10}  {'确定性判定':>10}  {'一致':>4}")
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 10}  {'─' * 4}")
    for num in test_numbers:
        fermat_result = is_prime_fermat(num, k=10)
        det_result = is_prime_deterministic(num)
        match = "✓" if fermat_result == det_result else "✗"
        print(f"  {num:>6}  {'素数' if fermat_result else '合数':>10}  "
              f"{'素数' if det_result else '合数':>10}  {match:>4}")
    
    print("\n  注: 561 是卡迈克尔数 (伪素数)，费马小定理可能误判")
    
    print("\n--- 4. 浮点数精度保障验证 ---")
    _demonstrate_float_precision_issue()
    _demonstrate_large_number_correctness()
    
    test_cases = [
        (3, 4, 5, 3),
        (5, 12, 13, 3),
        (1, 1, 2, 3),
        (6, 8, 10, 3),
    ]
    
    print("\n--- 5. 特定情况验证 (n>2) ---")
    for x, y, z, n in test_cases:
        try:
            result = check_fermat(x, y, z, n)
            print(f"  x={x}, y={y}, z={z}, n={n}: {result}")
        except ValueError as e:
            print(f"  x={x}, y={y}, z={z}, n={n}: 错误 - {e}")
    
    print("\n--- 6. 暴力验证 (x, y, z ≤ 100, n=3,4,5) ---")
    print("正在搜索...")
    found_solutions = brute_force_fermat(max_value=100, min_n=3, max_n=5)
    
    if found_solutions:
        print(f"\n找到 {len(found_solutions)} 个解:")
        for sol in found_solutions:
            print(f"  x={sol[0]}, y={sol[1]}, z={sol[2]}, n={sol[3]}")
    else:
        print("\n未找到任何解，验证了费马大定理在该范围内成立。")
    
    print("\n--- 7. 边界情况与提前终止 ---")
    try:
        check_fermat(1, 1, 2, 2)
    except ValueError as e:
        print(f"  n=2 时: 正确抛出错误 - {e}")
    
    try:
        check_fermat(0, 1, 1, 3)
    except ValueError as e:
        print(f"  x=0 时: 正确抛出错误 - {e}")
    
    print(f"  check_fermat(5, 10, 10, 3) = {check_fermat(5, 10, 10, 3)}")
    print(f"  (z <= max(x,y) 时直接返回 False，无需计算幂)")
