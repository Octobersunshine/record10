import re
from datetime import datetime

from weak_passwords import WEAK_PASSWORDS
from password_breach import PasswordBreachChecker, check_passwords_breach_batch


def check_password_strength(password, check_breach=False, use_cache=True, breach_checker=None):
    lower_password = password.lower()
    in_blacklist = lower_password in WEAK_PASSWORDS

    if in_blacklist:
        result = {
            "strength": "极弱",
            "score": 0,
            "max_score": 6,
            "suggestions": ["该密码是常见弱口令，请立即更换为更复杂的密码"],
            "in_blacklist": True
        }
    else:
        score = 0
        suggestions = []

        if len(password) >= 8:
            score += 1
        else:
            suggestions.append("增加长度到至少8个字符")

        if len(password) >= 12:
            score += 1

        if re.search(r'[a-z]', password):
            score += 1
        else:
            suggestions.append("添加小写字母")

        if re.search(r'[A-Z]', password):
            score += 1
        else:
            suggestions.append("添加大写字母")

        if re.search(r'[0-9]', password):
            score += 1
        else:
            suggestions.append("添加数字")

        if re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'\"\\|,.<>/?]', password):
            score += 1
        else:
            suggestions.append("添加特殊字符（如!@#$%^&*等）")

        if score <= 2:
            strength = "弱"
        elif score <= 4:
            strength = "中"
        else:
            strength = "强"

        result = {
            "strength": strength,
            "score": score,
            "max_score": 6,
            "suggestions": suggestions,
            "in_blacklist": False
        }

    if check_breach:
        if breach_checker is None:
            breach_checker = PasswordBreachChecker()
        
        breach_result = breach_checker.check_password(password, use_cache)
        result["breach_check"] = breach_result
        
        if breach_result["breached"]:
            breach_count = breach_result["breach_count"]
            result["suggestions"].insert(
                0, 
                f"⚠️  该密码已在数据泄露中出现 {breach_count:,} 次，强烈建议立即更换！"
            )
            
            if not in_blacklist and result["strength"] != "极弱":
                if breach_count >= 1000000:
                    result["strength"] = "极弱"
                elif breach_count >= 100000:
                    if result["strength"] == "强":
                        result["strength"] = "弱"
                    elif result["strength"] == "中":
                        result["strength"] = "弱"
                elif breach_count >= 1000:
                    if result["strength"] == "强":
                        result["strength"] = "中"

    return result


def check_passwords_strength_batch(passwords, check_breach=False, use_cache=True):
    results = []
    breach_checker = PasswordBreachChecker() if check_breach else None
    
    for password in passwords:
        result = check_password_strength(
            password, 
            check_breach=check_breach, 
            use_cache=use_cache,
            breach_checker=breach_checker
        )
        results.append(result)
    
    return results


def format_timestamp(timestamp):
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return "未知"


def print_result(password, result, show_breach_details=False):
    print(f"\n{'='*60}")
    print(f"密码: {password}")
    print(f"{'='*60}")
    print(f"强度评级: {result['strength']}")
    print(f"得分: {result['score']}/{result['max_score']}")
    print(f"黑名单命中: {'是 ⚠️' if result.get('in_blacklist') else '否'}")
    
    breach_check = result.get('breach_check')
    if breach_check:
        print(f"泄露检测: {'已泄露 ⚠️' if breach_check['breached'] else '未泄露 ✓'}")
        if breach_check['breached']:
            print(f"泄露次数: {breach_check['breach_count']:,} 次")
            print(f"数据来源: {breach_check['source']}")
            print(f"检测时间: {format_timestamp(breach_check['last_checked'])}")
        if show_breach_details:
            print(f"SHA1哈希: {breach_check['full_hash']}")
    
    if result['suggestions']:
        print("\n改进建议:")
        for i, suggestion in enumerate(result['suggestions'], 1):
            print(f"  {i}. {suggestion}")
    else:
        print("\n恭喜！您的密码非常安全！")


def main():
    print("=" * 60)
    print("=== 密码强度检测工具 ===")
    print("=" * 60)
    print("\n1. 单个密码检测")
    print("2. 批量密码检测")
    print("3. 查看本地缓存统计")
    print("4. 清除本地缓存")
    
    choice = input("\n请选择功能 (1-4): ").strip()
    
    if choice == "3":
        checker = PasswordBreachChecker()
        stats = checker.get_statistics()
        print(f"\n{'='*60}")
        print("本地缓存统计")
        print(f"{'='*60}")
        print(f"缓存记录数: {stats['total_cached_records']:,}")
        print(f"泄露实例总数: {stats['total_breach_instances']:,}")
        print(f"最后同步时间: {format_timestamp(stats['last_sync_timestamp'])}")
        print(f"数据库路径: {stats['database_path']}")
        return
    
    if choice == "4":
        confirm = input("\n确定要清除所有本地缓存吗？(y/N): ").strip().lower()
        if confirm == 'y':
            checker = PasswordBreachChecker()
            checker.clear_cache()
            print("\n缓存已清除！")
        else:
            print("\n操作已取消。")
        return
    
    check_breach = input("\n是否启用泄露检测？(y/N): ").strip().lower() == 'y'
    use_cache = True
    
    if check_breach:
        use_cache_input = input("是否使用本地缓存？(Y/n): ").strip().lower()
        use_cache = use_cache_input != 'n'
    
    if choice == "1":
        password = input("\n请输入要检测的密码: ")
        if not password:
            print("密码不能为空！")
            return
        
        result = check_password_strength(password, check_breach=check_breach, use_cache=use_cache)
        print_result(password, result, show_breach_details=True)
        
    elif choice == "2":
        print("\n请输入要检测的密码（每行一个，输入空行结束）:")
        passwords = []
        while True:
            pwd = input()
            if not pwd:
                break
            passwords.append(pwd)
        
        if not passwords:
            print("未输入任何密码！")
            return
        
        print(f"\n正在检测 {len(passwords)} 个密码...")
        results = check_passwords_strength_batch(passwords, check_breach=check_breach, use_cache=use_cache)
        
        breached_count = sum(1 for r in results if r.get('breach_check', {}).get('breached'))
        blacklist_count = sum(1 for r in results if r.get('in_blacklist'))
        
        for password, result in zip(passwords, results):
            print_result(password, result)
        
        print(f"\n{'='*60}")
        print("批量检测汇总")
        print(f"{'='*60}")
        print(f"检测总数: {len(passwords)}")
        print(f"黑名单命中: {blacklist_count}")
        if check_breach:
            print(f"已泄露密码: {breached_count}")
        
        if check_breach and breached_count > 0:
            print(f"\n⚠️  警告：{breached_count} 个密码已在数据泄露中出现，请立即更换！")
        
    else:
        print("无效的选择！")


if __name__ == "__main__":
    main()
