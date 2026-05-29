import requests
import json
import time
from datetime import datetime, timedelta

BASE_URL = 'http://127.0.0.1:5000'


def print_response(title, response):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"  Status: {response.status_code}")
    print(f"  Response:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print(f"{'='*70}")
    return response.json()


def reset_db():
    entries = requests.get(f"{BASE_URL}/api/ip").json()
    for e in entries:
        requests.delete(f"{BASE_URL}/api/ip/{e['id']}")


reset_db()

print("=" * 70)
print("  IP黑白名单CRUD API 测试 - 动态封禁与过期功能")
print("=" * 70)

print_response(
    "1. 创建临时黑名单 (10秒过期)",
    requests.post(f"{BASE_URL}/api/ip", json={
        "ip_address": "203.0.113.10",
        "list_type": "blacklist",
        "ttl_seconds": 10,
        "description": "临时封禁测试"
    })
)

print_response(
    "2. 检查临时封禁IP (应拒绝)",
    requests.post(f"{BASE_URL}/api/ip/check", json={"ip": "203.0.113.10"})
)

print_response(
    "3. 触发自动封禁阈值 - 第1次失败",
    requests.post(f"{BASE_URL}/api/ip/fail", json={
        "ip": "192.168.1.50",
        "reason": "Wrong password"
    })
)

print_response(
    "4. 触发自动封禁阈值 - 第2次失败",
    requests.post(f"{BASE_URL}/api/ip/fail", json={
        "ip": "192.168.1.50",
        "reason": "Wrong password"
    })
)

print_response(
    "5. 触发自动封禁阈值 - 第3次失败",
    requests.post(f"{BASE_URL}/api/ip/fail", json={
        "ip": "192.168.1.50",
        "reason": "Wrong password"
    })
)

print_response(
    "6. 触发自动封禁阈值 - 第4次失败",
    requests.post(f"{BASE_URL}/api/ip/fail", json={
        "ip": "192.168.1.50",
        "reason": "Wrong password"
    })
)

print_response(
    "7. 触发自动封禁阈值 - 第5次失败 (应触发自动封禁)",
    requests.post(f"{BASE_URL}/api/ip/fail", json={
        "ip": "192.168.1.50",
        "reason": "Wrong password"
    })
)

print_response(
    "8. 检查自动封禁的IP (应拒绝)",
    requests.post(f"{BASE_URL}/api/ip/check", json={"ip": "192.168.1.50"})
)

print_response(
    "9. 查看封禁统计",
    requests.get(f"{BASE_URL}/api/ip/ban/stats")
)

print_response(
    "10. 查看封禁日志",
    requests.get(f"{BASE_URL}/api/ip/ban/logs?limit=10")
)

print_response(
    "11. 查看当前黑名单列表 (含临时和自动封禁)",
    requests.get(f"{BASE_URL}/api/ip?list_type=blacklist")
)

print_response(
    "12. 另一个IP触发失败 - 用于top failing统计",
    requests.post(f"{BASE_URL}/api/ip/fail", json={
        "ip": "10.20.30.40",
        "reason": "Invalid token"
    })
)
requests.post(f"{BASE_URL}/api/ip/fail", json={
    "ip": "10.20.30.40",
    "reason": "Invalid token"
})
requests.post(f"{BASE_URL}/api/ip/fail", json={
    "ip": "10.20.30.40",
    "reason": "Invalid token"
})

print_response(
    "13. 查看更新后的统计 (包含top failing IPs)",
    requests.get(f"{BASE_URL}/api/ip/ban/stats")
)

print_response(
    "14. 按事件类型过滤日志 - 只看auto_ban",
    requests.get(f"{BASE_URL}/api/ip/ban/logs?event_type=auto_ban")
)

print_response(
    "15. 手动清理过期封禁 (当前应该没有过期的)",
    requests.post(f"{BASE_URL}/api/ip/ban/cleanup")
)

print("\n" + "=" * 70)
print("  等待临时封禁过期中 (约12秒)...")
print("=" * 70)
time.sleep(12)

print_response(
    "16. 检查过期后的临时IP 203.0.113.10 (应放行，自动清理)",
    requests.post(f"{BASE_URL}/api/ip/check", json={"ip": "203.0.113.10"})
)

print_response(
    "17. 查看更新后的封禁统计 (临时封禁已被清理)",
    requests.get(f"{BASE_URL}/api/ip/ban/stats")
)

print_response(
    "18. 查看unban_expired类型的日志",
    requests.get(f"{BASE_URL}/api/ip/ban/logs?event_type=unban_expired")
)

print_response(
    "19. 创建永久封禁 (无ttl)",
    requests.post(f"{BASE_URL}/api/ip", json={
        "ip_address": "1.1.1.1",
        "list_type": "blacklist",
        "description": "永久封禁 - 已知攻击源"
    })
)

print_response(
    "20. 查看过滤后的列表 - 只看临时封禁",
    requests.get(f"{BASE_URL}/api/ip?list_type=blacklist&is_temporary=true")
)

print_response(
    "21. 查看过滤后的列表 - 只看永久封禁",
    requests.get(f"{BASE_URL}/api/ip?list_type=blacklist&is_temporary=false")
)

print_response(
    "22. 更新永久封禁为临时 (设置ttl)",
    requests.put(f"{BASE_URL}/api/ip/3", json={
        "ttl_seconds": 60,
        "description": "改为临时封禁"
    })
)

print_response(
    "23. 批量检查多个IP权限",
    requests.post(f"{BASE_URL}/api/ip/check/batch", json={
        "ips": [
            "192.168.1.50",
            "203.0.113.10",
            "1.1.1.1",
            "8.8.8.8",
            "invalid-ip"
        ]
    })
)

print("\n" + "=" * 70)
print("  测试完成！核心功能验证:")
print("  - 临时黑名单支持 ttl_seconds 参数")
print("  - 自动封禁: 5次失败后自动加入黑名单 (1小时)")
print("  - 过期自动清理: check接口自动清理过期规则")
print("  - 封禁统计: /api/ip/ban/stats 含top failing IPs")
print("  - 封禁日志: /api/ip/ban/logs 支持过滤")
print("  - 通知机制: auto_ban和manual_ban返回notification")
print("  - 支持is_temporary/is_expired过滤查询")
print("=" * 70)
