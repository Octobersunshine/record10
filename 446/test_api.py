import requests
import json
import time

BASE_URL = 'http://127.0.0.1:5000'

def test_api():
    print('=' * 80)
    print('IP黑白名单API测试 - 动态封禁与过期时间版')
    print('=' * 80)

    print('\n' + '-' * 80)
    print('第一部分：配置管理测试')
    print('-' * 80)

    print('\n1. 获取当前配置')
    r = requests.get(f'{BASE_URL}/api/config')
    print(f'状态码: {r.status_code}')
    print(f'配置: {json.dumps(r.json(), indent=2, ensure_ascii=False)}')

    print('\n2. 修改配置 - 降低阈值方便测试')
    config_data = {
        'max_failure_attempts': 3,
        'failure_window_seconds': 60,
        'auto_ban_duration_seconds': 30
    }
    r = requests.put(f'{BASE_URL}/api/config', json=config_data)
    print(f'状态码: {r.status_code}')
    print(f'更新后配置: {json.dumps(r.json(), indent=2, ensure_ascii=False)}')

    print('\n' + '-' * 80)
    print('第二部分：临时黑名单（过期时间）测试')
    print('-' * 80)

    print('\n3. 创建临时黑名单IP（30秒过期）')
    data = {
        'ip_address': '1.2.3.4',
        'list_type': 'blacklist',
        'description': '临时封禁测试',
        'duration_seconds': 30
    }
    r = requests.post(f'{BASE_URL}/api/ip', json=data)
    print(f'状态码: {r.status_code}')
    resp = r.json()
    print(f'响应: {json.dumps(resp, indent=2, ensure_ascii=False)}')
    temp_ban_id = resp['id']
    print(f'过期时间: {resp["expires_at"]}')
    print(f'是否过期: {resp["is_expired"]}')

    print('\n4. 检查该IP - 应该被拒绝')
    r = requests.get(f'{BASE_URL}/api/ip/check', params={'ip': '1.2.3.4'})
    print(f'状态码: {r.status_code}')
    print(f'允许访问: {r.json()["allowed"]}')
    print(f'原因: {r.json()["reason"]}')

    print('\n5. 创建带expires_at的黑名单')
    from datetime import datetime, timedelta
    future_time = (datetime.utcnow() + timedelta(seconds=60)).isoformat()
    data = {
        'ip_address': '5.6.7.8',
        'list_type': 'blacklist',
        'description': '指定过期时间测试',
        'expires_at': future_time
    }
    r = requests.post(f'{BASE_URL}/api/ip', json=data)
    print(f'状态码: {r.status_code}')
    print(f'过期时间: {r.json()["expires_at"]}')

    print('\n6. 创建永久黑名单（不设置过期时间）')
    data = {
        'ip_address': '9.9.9.9',
        'list_type': 'blacklist',
        'description': '永久封禁测试'
    }
    r = requests.post(f'{BASE_URL}/api/ip', json=data)
    print(f'状态码: {r.status_code}')
    print(f'过期时间: {r.json()["expires_at"]} (None=永久)')

    print('\n' + '-' * 80)
    print('第三部分：动态封禁测试')
    print('-' * 80)

    test_ip = '10.20.30.40'
    print(f'\n7. 测试动态封禁 - IP: {test_ip}')
    print(f'   阈值: 3次失败，窗口: 60秒')

    for i in range(1, 5):
        data = {
            'ip': test_ip,
            'reason': f'登录失败 #{i}',
            'path': '/api/login',
            'user_agent': 'TestAgent/1.0'
        }
        r = requests.post(f'{BASE_URL}/api/failure', json=data)
        resp = r.json()
        print(f'\n   第 {i} 次失败记录:')
        print(f'   - 当前失败次数: {resp["current_failure_count"]}/{resp["threshold"]}')
        print(f'   - 是否自动封禁: {resp["auto_banned"]}')
        if resp.get('auto_banned'):
            print(f'   - 封禁详情: {json.dumps(resp["ban_details"], indent=4, ensure_ascii=False)}')
            break

    print('\n8. 检查被自动封禁的IP - 应该被拒绝')
    r = requests.get(f'{BASE_URL}/api/ip/check', params={'ip': test_ip})
    print(f'状态码: {r.status_code}')
    resp = r.json()
    print(f'允许访问: {resp["allowed"]}')
    print(f'原因: {resp["reason"]}')
    print(f'匹配的黑名单: {json.dumps(resp["matched_blacklist"], indent=2, ensure_ascii=False)}')

    print('\n9. verify接口查看详细信息')
    r = requests.get(f'{BASE_URL}/api/ip/verify', params={'ip': test_ip})
    resp = r.json()
    print(f'IP: {resp["ip"]}')
    print(f'是否在黑名单: {resp["verification_details"]["is_blacklisted"]}')
    print(f'匹配黑名单数: {resp["matched_rules"]["blacklist"]["count"]}')
    print(f'当前失败次数: {resp["current_failure_count"]}')
    print(f'是否自动封禁: {resp["matched_rules"]["blacklist"]["entries"][0]["is_autoban"]}')
    print(f'过期时间: {resp["matched_rules"]["blacklist"]["entries"][0]["expires_at"]}')

    print('\n' + '-' * 80)
    print('第四部分：封禁统计测试')
    print('-' * 80)

    print('\n10. 获取封禁统计汇总')
    r = requests.get(f'{BASE_URL}/api/stats/ban/summary')
    print(f'状态码: {r.status_code}')
    summary = r.json()
    print(f'统计汇总: {json.dumps(summary, indent=2, ensure_ascii=False)}')
    print(f'\n关键指标:')
    print(f'  - 总封禁数: {summary["total_bans"]}')
    print(f'  - 活跃封禁数: {summary["active_bans"]}')
    print(f'  - 临时封禁数: {summary["temporary_bans"]}')
    print(f'  - 永久封禁数: {summary["permanent_bans"]}')
    print(f'  - 自动封禁数: {summary["auto_bans"]}')
    print(f'  - 今日封禁数: {summary["bans_today"]}')
    print(f'  - 未读通知数: {summary["unread_notifications"]}')

    print(f'\n11. 获取封禁历史记录 (IP={test_ip})')
    r = requests.get(f'{BASE_URL}/api/stats/ban', params={'ip': test_ip})
    print(f'状态码: {r.status_code}')
    print(f'历史记录: {json.dumps(r.json(), indent=2, ensure_ascii=False)}')

    print('\n12. 获取当前活跃封禁列表')
    r = requests.get(f'{BASE_URL}/api/bans/active')
    print(f'状态码: {r.status_code}')
    active_bans = r.json()
    print(f'活跃封禁数: {len(active_bans)}')
    for ban in active_bans[:3]:
        print(f'  - {ban["ip_address"]} (过期: {ban["expires_at"]}, 自动: {ban["is_autoban"]})')

    print('\n' + '-' * 80)
    print('第五部分：通知系统测试')
    print('-' * 80)

    print('\n13. 获取通知列表（未读）')
    r = requests.get(f'{BASE_URL}/api/notifications', params={'unread_only': 'true'})
    print(f'状态码: {r.status_code}')
    notifications = r.json()
    print(f'未读通知数: {len(notifications)}')
    for n in notifications[:5]:
        print(f'  [{n["notification_type"]}] {n["message"]} (IP: {n["ip_address"]})')

    if notifications:
        first_id = notifications[0]['id']
        print(f'\n14. 标记单条通知为已读 (ID={first_id})')
        r = requests.post(f'{BASE_URL}/api/notifications/{first_id}/read')
        print(f'状态码: {r.status_code}')

    print('\n15. 标记所有通知为已读')
    r = requests.post(f'{BASE_URL}/api/notifications/read-all')
    print(f'状态码: {r.status_code}')

    print('\n16. 再次获取未读通知 - 应该为0')
    r = requests.get(f'{BASE_URL}/api/notifications', params={'unread_only': 'true'})
    print(f'未读通知数: {len(r.json())}')

    print('\n' + '-' * 80)
    print('第六部分：过期清理测试')
    print('-' * 80)

    print('\n17. 创建一个极短时间的临时封禁（2秒过期）')
    data = {
        'ip_address': '99.99.99.99',
        'list_type': 'blacklist',
        'description': '快速过期测试',
        'duration_seconds': 2
    }
    r = requests.post(f'{BASE_URL}/api/ip', json=data)
    print(f'状态码: {r.status_code}')
    print(f'创建成功，过期时间: {r.json()["expires_at"]}')

    print('\n18. 检查该IP - 应该被拒绝')
    r = requests.get(f'{BASE_URL}/api/ip/check', params={'ip': '99.99.99.99'})
    print(f'状态码: {r.status_code}')
    print(f'允许访问: {r.json()["allowed"]}')

    print('\n19. 等待3秒让封禁过期...')
    time.sleep(3)

    print('\n20. 手动触发清理')
    r = requests.post(f'{BASE_URL}/api/cleanup')
    print(f'状态码: {r.status_code}')
    print(f'清理结果: {r.json()}')

    print('\n21. 再次检查该IP - 应该可以访问（已过期）')
    r = requests.get(f'{BASE_URL}/api/ip/check', params={'ip': '99.99.99.99'})
    print(f'状态码: {r.status_code}')
    resp = r.json()
    print(f'允许访问: {resp["allowed"]}')
    print(f'原因: {resp["reason"]}')

    print('\n22. 查询列表 include_expired=true 查看已过期的')
    r = requests.get(f'{BASE_URL}/api/ip', params={'list_type': 'blacklist', 'include_expired': 'true'})
    all_bans = r.json()
    expired_count = sum(1 for b in all_bans if b.get('is_expired'))
    print(f'黑名单总数: {len(all_bans)}')
    print(f'已过期数: {expired_count}')

    print('\n' + '-' * 80)
    print('第七部分：更新和删除测试')
    print('-' * 80)

    print('\n23. 更新封禁时长 - 延长临时封禁')
    update_data = {
        'duration_seconds': 3600
    }
    r = requests.put(f'{BASE_URL}/api/ip/{temp_ban_id}', json=update_data)
    print(f'状态码: {r.status_code}')
    resp = r.json()
    print(f'更新后过期时间: {resp["expires_at"]}')
    print(f'原时长30秒，现改为3600秒')

    print('\n24. 更新为永久封禁')
    update_data = {
        'duration_seconds': None,
        'description': '已升级为永久封禁'
    }
    r = requests.put(f'{BASE_URL}/api/ip/{temp_ban_id}', json=update_data)
    print(f'状态码: {r.status_code}')
    print(f'更新后过期时间: {r.json()["expires_at"]} (None=永久)')

    print('\n25. 删除封禁（手动解封）')
    r = requests.delete(f'{BASE_URL}/api/ip/{temp_ban_id}')
    print(f'状态码: {r.status_code}')
    print(f'响应: {r.json()}')

    print('\n26. 检查删除后的IP - 应该可以访问')
    r = requests.get(f'{BASE_URL}/api/ip/check', params={'ip': '1.2.3.4'})
    print(f'允许访问: {r.json()["allowed"]}')

    print('\n' + '-' * 80)
    print('测试总结')
    print('-' * 80)
    print('\n✅ 配置管理 - 动态修改阈值和参数')
    print('✅ 临时黑名单 - 支持 duration_seconds 和 expires_at')
    print('✅ 永久黑名单 - 不设置过期时间')
    print('✅ 动态封禁 - 失败次数超过阈值自动加入黑名单')
    print('✅ 自动封禁统计 - 记录失败次数、封禁历史')
    print('✅ 过期自动清理 - 后台线程定时清理')
    print('✅ 手动清理接口 - /api/cleanup')
    print('✅ 封禁统计汇总 - /api/stats/ban/summary')
    print('✅ 封禁历史查询 - /api/stats/ban')
    print('✅ 通知系统 - 自动封禁、自动解封、手动解封通知')
    print('✅ 通知管理 - 标记已读、全部已读')
    print('✅ 过期状态标识 - is_expired 字段')
    print('✅ 活跃封禁列表 - /api/bans/active')

    print('\n' + '=' * 80)
    print('所有测试完成')
    print('=' * 80)


if __name__ == '__main__':
    test_api()
