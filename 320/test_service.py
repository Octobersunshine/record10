import requests
import sys

BASE_URL = 'http://localhost:5000'


def test_all():
    print('=== Test 1: 基础创建短链接 ===')
    long_url = 'https://www.python.org'
    response = requests.post(f'{BASE_URL}/shorten', json={'url': long_url})
    assert response.status_code == 201
    data = response.json()
    assert 'short_code' in data
    assert 'created_at' in data
    auto_code = data['short_code']
    print(f'自动生成短码: {auto_code}')
    print('✓ 基础创建测试通过\n')

    print('=== Test 2: 自定义别名 ===')
    response = requests.post(f'{BASE_URL}/shorten', json={
        'url': 'https://github.com',
        'alias': 'my-github'
    })
    assert response.status_code == 201
    data = response.json()
    assert data['short_code'] == 'my-github'
    print(f'自定义别名: {data["short_code"]}')
    print('✓ 自定义别名测试通过\n')

    print('=== Test 3: 别名冲突 ===')
    response = requests.post(f'{BASE_URL}/shorten', json={
        'url': 'https://example.com',
        'alias': 'my-github'
    })
    assert response.status_code == 409
    print('✓ 别名冲突测试通过\n')

    print('=== Test 4: 非法别名 ===')
    response = requests.post(f'{BASE_URL}/shorten', json={
        'url': 'https://example.com',
        'alias': 'invalid alias!'
    })
    assert response.status_code == 400
    print('✓ 非法别名测试通过\n')

    print('=== Test 5: 访问短链接并统计 ===')
    for i in range(5):
        r = requests.get(f'{BASE_URL}/{auto_code}', allow_redirects=False)
        assert r.status_code == 302
    print(f'访问 {auto_code} 5次')

    for i in range(3):
        r = requests.get(f'{BASE_URL}/my-github', allow_redirects=False)
        assert r.status_code == 302
    print('访问 my-github 3次')
    print('✓ 访问统计测试通过\n')

    print('=== Test 6: 单个短链接统计报表 ===')
    response = requests.get(f'{BASE_URL}/stats/{auto_code}')
    assert response.status_code == 200
    data = response.json()
    assert data['total_visits'] == 5
    assert data['short_code'] == auto_code
    assert 'created_at' in data
    assert 'unique_visitors' in data
    assert 'top_ips' in data
    assert 'recent_visits' in data
    assert len(data['all_visits']) == 5
    assert data['all_visits'][0]['ip'] is not None
    assert data['all_visits'][0]['timestamp'] is not None
    print(f'总访问次数: {data["total_visits"]}')
    print(f'独立访客: {data["unique_visitors"]}')
    print(f'最近访问时间: {data["recent_visits"][-1]["timestamp"]}')
    print('✓ 单个统计报表测试通过\n')

    print('=== Test 7: 全局统计报表 ===')
    response = requests.get(f'{BASE_URL}/stats')
    assert response.status_code == 200
    data = response.json()
    assert data['total_short_urls'] >= 2
    assert data['total_visits_all'] == 8
    assert len(data['rankings']) >= 2
    assert data['rankings'][0]['total_visits'] >= data['rankings'][1]['total_visits']
    print(f'短链接总数: {data["total_short_urls"]}')
    print(f'总访问量: {data["total_visits_all"]}')
    print(f'排名第一: {data["rankings"][0]["short_code"]} ({data["rankings"][0]["total_visits"]}次)')
    print('✓ 全局统计报表测试通过\n')

    print('=== Test 8: 不存在的短码统计 ===')
    response = requests.get(f'{BASE_URL}/stats/nonexist')
    assert response.status_code == 404
    print('✓ 404统计测试通过\n')

    print('=== 所有测试通过！ ===')


if __name__ == '__main__':
    try:
        test_all()
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
