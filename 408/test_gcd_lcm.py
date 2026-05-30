from gcd_lcm import compute_gcd_lcm, gcd, lcm, extended_gcd, gcd_multi, lcm_multi

print("=" * 60)
print("测试1: 基本GCD和LCM (含零和负数)")
print("=" * 60)

test_cases = [
    (12, 18, 6, 36),
    (25, 15, 5, 75),
    (7, 13, 1, 91),
    (100, 25, 25, 100),
    (48, 18, 6, 144),
    (0, 5, 5, 0),
    (5, 0, 5, 0),
    (0, 0, 0, 0),
    (-12, 18, 6, 36),
    (12, -18, 6, 36),
    (-12, -18, 6, 36),
    (-7, 13, 1, 91),
]

all_passed = True
for a, b, expected_gcd, expected_lcm in test_cases:
    result_gcd, result_lcm = compute_gcd_lcm(a, b)
    status = "✓" if result_gcd == expected_gcd and result_lcm == expected_lcm else "✗"
    if result_gcd != expected_gcd or result_lcm != expected_lcm:
        all_passed = False
    print(f"{status} a={a}, b={b}: GCD={result_gcd} (期望:{expected_gcd}), LCM={result_lcm} (期望:{expected_lcm})")

print("\n" + "=" * 60)
print("测试2: 扩展欧几里得算法 (ax + by = GCD)")
print("=" * 60)

ext_test_cases = [
    (35, 15),
    (12, 18),
    (25, 15),
    (7, 13),
    (48, 18),
    (-35, 15),
    (35, -15),
    (0, 5),
    (5, 0),
]

for a, b in ext_test_cases:
    g, x, y = extended_gcd(a, b)
    check = a * x + b * y
    status = "✓" if check == g else "✗"
    if check != g:
        all_passed = False
    print(f"{status} {a}×({x}) + {b}×({y}) = {g}  [验证: {check}]")

print("\n" + "=" * 60)
print("测试3: 多参数GCD和LCM")
print("=" * 60)

multi_test_cases = [
    ((12, 18, 24), 6, 72),
    ((12, 18, 24, 30), 6, 360),
    ((5, 10, 15, 20, 25), 5, 300),
    ((7, 11, 13), 1, 1001),
    ((0, 5, 10), 5, 0),
    ((-12, 18, -24), 6, 72),
]

for nums, expected_gcd, expected_lcm in multi_test_cases:
    result_gcd = gcd_multi(*nums)
    result_lcm = lcm_multi(*nums)
    status = "✓" if result_gcd == expected_gcd and result_lcm == expected_lcm else "✗"
    if result_gcd != expected_gcd or result_lcm != expected_lcm:
        all_passed = False
    print(f"{status} gcd_multi{nums} = {result_gcd} (期望:{expected_gcd})")
    print(f"   lcm_multi{nums} = {result_lcm} (期望:{expected_lcm})")

print("\n" + "=" * 60)
print("测试4: 辗转相除过程步骤")
print("=" * 60)

step_test_cases = [
    (48, 18, 3),
    (35, 15, 2),
    (1071, 462, 3),
    (7, 13, 4),
]

for a, b, expected_steps in step_test_cases:
    g, steps = gcd(a, b, return_steps=True)
    status = "✓" if len(steps) == expected_steps else "✗"
    if len(steps) != expected_steps:
        all_passed = False
    print(f"{status} GCD({a}, {b}) = {g}, 步骤数: {len(steps)} (期望:{expected_steps})")
    for i, (na, nb, q, r) in enumerate(steps, 1):
        print(f"   步骤{i}: {na} = {nb} × {q} + {r}")

print("\n" + "=" * 60)
if all_passed:
    print("✓ 所有测试通过!")
else:
    print("✗ 部分测试失败!")
print("=" * 60)
