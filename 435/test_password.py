from password_strength import check_password_strength

test_cases = [
    ("123456", "纯数字短密码"),
    ("abcdef", "纯小写字母"),
    ("ABCDEF", "纯大写字母"),
    ("abc123", "小写字母+数字"),
    ("Password", "大小写字母"),
    ("Password123", "大小写字母+数字"),
    ("Password123!", "完整组合"),
    ("StrongP@ssw0rd", "强密码"),
]

blacklist_test_cases = [
    ("password", "经典弱口令-password"),
    ("PASSWORD", "经典弱口令-PASSWORD(大写)"),
    ("Password", "经典弱口令-Password"),
    ("123456", "经典弱口令-123456"),
    ("123456789", "经典弱口令-123456789"),
    ("qwerty", "经典弱口令-qwerty"),
    ("abc123", "经典弱口令-abc123"),
    ("111111", "经典弱口令-111111"),
    ("password123", "经典弱口令-password123"),
    ("password1", "经典弱口令-password1"),
    ("iloveyou", "经典弱口令-iloveyou"),
    ("monkey", "经典弱口令-monkey"),
    ("dragon", "经典弱口令-dragon"),
    ("football", "经典弱口令-football"),
    ("baseball", "经典弱口令-baseball"),
    ("admin123", "经典弱口令-admin123"),
    ("root123", "经典弱口令-root123"),
    ("test123", "经典弱口令-test123"),
    ("welcome123", "经典弱口令-welcome123"),
    ("hello123", "经典弱口令-hello123"),
    ("000000", "经典弱口令-000000"),
    ("654321", "经典弱口令-654321"),
    ("987654321", "经典弱口令-987654321"),
    ("qazwsx", "经典弱口令-qazwsx"),
    ("zxcvbnm", "经典弱口令-zxcvbnm"),
]

print("=" * 60)
print("密码强度检测测试 - 普通密码")
print("=" * 60)

for password, description in test_cases:
    result = check_password_strength(password)
    print(f"\n测试密码: '{password}' ({description})")
    print(f"  强度: {result['strength']}")
    print(f"  得分: {result['score']}/{result['max_score']}")
    print(f"  黑名单命中: {'是' if result.get('in_blacklist') else '否'}")
    if result['suggestions']:
        print(f"  建议: {', '.join(result['suggestions'])}")

print("\n" + "=" * 60)
print("密码强度检测测试 - 黑名单弱口令")
print("=" * 60)

blacklist_hits = 0
for password, description in blacklist_test_cases:
    result = check_password_strength(password)
    is_blacklist = result.get('in_blacklist', False)
    if is_blacklist and result['strength'] == '极弱':
        blacklist_hits += 1
    print(f"\n测试密码: '{password}' ({description})")
    print(f"  强度: {result['strength']}")
    print(f"  得分: {result['score']}/{result['max_score']}")
    print(f"  黑名单命中: {'是 ✓' if is_blacklist else '否 ✗'}")
    if result['suggestions']:
        print(f"  建议: {', '.join(result['suggestions'])}")

print("\n" + "=" * 60)
print(f"测试结果: {blacklist_hits}/{len(blacklist_test_cases)} 个黑名单弱口令被正确识别")
print("=" * 60)

non_blacklist_test = ["MyStr0ngP@ss!", "C0mpl3x!P@ss", "9Kx$mP2#vL", "H@ckM31fY0uC@n", "N0t1nB1@ckL1st"]
print("\n" + "=" * 60)
print("密码强度检测测试 - 非黑名单强密码")
print("=" * 60)

for password in non_blacklist_test:
    result = check_password_strength(password)
    is_blacklist = result.get('in_blacklist', False)
    print(f"\n测试密码: '{password}'")
    print(f"  强度: {result['strength']}")
    print(f"  得分: {result['score']}/{result['max_score']}")
    print(f"  黑名单命中: {'是' if is_blacklist else '否'}")
    if result['suggestions']:
        print(f"  建议: {', '.join(result['suggestions'])}")

print("\n" + "=" * 60)
