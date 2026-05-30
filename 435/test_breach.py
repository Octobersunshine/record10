from password_breach import (
    PasswordBreachChecker,
    check_password_breach,
    check_passwords_breach_batch
)
from password_strength import check_password_strength, check_passwords_strength_batch


print("=" * 60)
print("密码泄露检测功能测试")
print("=" * 60)

print("\n" + "=" * 60)
print("测试1: 单个密码泄露检测 (常见弱口令)")
print("=" * 60)

test_passwords = [
    "password",
    "123456",
    "qwerty",
    "letmein",
    "MyStr0ngP@ss123!",
]

for pwd in test_passwords:
    result = check_password_breach(pwd, use_cache=False)
    print(f"\n密码: '{pwd}'")
    print(f"  SHA1: {result['full_hash']}")
    print(f"  已泄露: {'是' if result['breached'] else '否'}")
    print(f"  泄露次数: {result['breach_count']:,}")
    print(f"  来源: {result['source']}")

print("\n" + "=" * 60)
print("测试2: 批量密码泄露检测")
print("=" * 60)

batch_passwords = [
    "password",
    "123456",
    "qwerty",
    "abc123",
    "letmein",
    "monkey",
    "dragon",
    "MyStr0ngP@ss!@#",
    "C0mpl3x!P@ssw0rd",
    "N3v3rB3Br0k3n!",
]

results = check_passwords_breach_batch(batch_passwords, use_cache=False)
breached = [r for r in results if r['breached']]

print(f"\n检测了 {len(results)} 个密码")
print(f"发现泄露: {len(breached)} 个")

for result in results:
    status = "⚠️  已泄露" if result['breached'] else "✓ 安全"
    count = f" ({result['breach_count']:,} 次)" if result['breached'] else ""
    print(f"  {status}: '{result['password']}'{count}")

print("\n" + "=" * 60)
print("测试3: 集成到强度检测 - 泄露对评级的影响")
print("=" * 60)

test_cases = [
    ("Password1!", "大小写+数字+特殊字符"),
    ("Letmein123!", "常见词变体"),
    ("Qwerty123!", "键盘序列变体"),
    ("MyStr0ngP@ss!", "高强度密码"),
    ("C0mpl3x!P@ss", "高强度密码2"),
]

for pwd, desc in test_cases:
    print(f"\n测试: '{pwd}' ({desc})")
    
    result_no_breach = check_password_strength(pwd, check_breach=False)
    print(f"  无泄露检测: 强度={result_no_breach['strength']}, 得分={result_no_breach['score']}/6")
    
    result_with_breach = check_password_strength(pwd, check_breach=True, use_cache=False)
    breach_info = result_with_breach.get('breach_check', {})
    
    if breach_info.get('breached'):
        print(f"  有泄露检测: 强度={result_with_breach['strength']}, 得分={result_with_breach['score']}/6")
        print(f"  泄露次数: {breach_info['breach_count']:,}")
        print(f"  强度降级: {result_no_breach['strength']} → {result_with_breach['strength']}")
    else:
        print(f"  有泄露检测: 强度={result_with_breach['strength']}, 得分={result_with_breach['score']}/6")
        print(f"  未发现泄露")

print("\n" + "=" * 60)
print("测试4: 批量强度检测 + 泄露检测")
print("=" * 60)

batch_passwords = [
    "password",
    "Password123!",
    "123456",
    "qwerty",
    "MyStr0ngP@ss!",
    "abc123",
    "C0mpl3x!P@ss",
    "letmein",
    "9Kx$mP2#vL",
    "N0t1nB1@ckL1st",
]

results = check_passwords_strength_batch(batch_passwords, check_breach=True, use_cache=False)

print(f"\n{'密码':<20} {'强度':<6} {'得分':<8} {'黑名单':<8} {'泄露':<8} {'泄露次数':<12}")
print("-" * 70)

for pwd, result in zip(batch_passwords, results):
    breach = result.get('breach_check', {})
    is_breached = breach.get('breached', False)
    breach_count = breach.get('breach_count', 0)
    
    pwd_display = pwd if len(pwd) <= 18 else pwd[:15] + "..."
    blacklist = "是" if result.get('in_blacklist') else "否"
    breached = "是" if is_breached else "否"
    count_str = f"{breach_count:,}" if breach_count > 0 else "-"
    
    print(f"{pwd_display:<20} {result['strength']:<6} {result['score']}/6{'':<3} {blacklist:<8} {breached:<8} {count_str:<12}")

print("\n" + "=" * 60)
print("测试5: 本地缓存功能")
print("=" * 60)

checker = PasswordBreachChecker()

print("\n第一次检测 (API):")
result1 = checker.check_password("password", use_cache=True)
print(f"  来源: {result1['source']}")

print("\n第二次检测 (缓存):")
result2 = checker.check_password("password", use_cache=True)
print(f"  来源: {result2['source']}")

print("\n第三次检测 (跳过缓存):")
result3 = checker.check_password("password", use_cache=False)
print(f"  来源: {result3['source']}")

stats = checker.get_statistics()
print(f"\n缓存统计:")
print(f"  总记录数: {stats['total_cached_records']:,}")
print(f"  泄露实例总数: {stats['total_breach_instances']:,}")
print(f"  数据库: {stats['database_path']}")

print("\n" + "=" * 60)
print("测试6: k-anonymity 隐私保护验证")
print("=" * 60)

pwd = "password"
full_hash = PasswordBreachChecker._get_sha1_hash(pwd)
prefix = full_hash[:5]
suffix = full_hash[5:]

print(f"\n密码: '{pwd}'")
print(f"SHA1哈希: {full_hash}")
print(f"发送给API的前缀: {prefix} (仅前5位)")
print(f"本地匹配的后缀: {suffix} (永不发送)")
print(f"\n✓ 采用k-anonymity模型，保护密码隐私")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
