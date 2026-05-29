import requests
import json

BASE_URL = 'http://localhost:5000'


def test_health():
    print("=== 测试健康检查接口 ===")
    response = requests.get(f'{BASE_URL}/health')
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_get_ip():
    print("=== 测试GET单IP查询接口 ===")
    ip = '8.8.8.8'
    response = requests.get(f'{BASE_URL}/api/ip/{ip}')
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_post_ip():
    print("=== 测试POST单IP查询接口 ===")
    data = {'ip': '1.1.1.1'}
    response = requests.post(f'{BASE_URL}/api/ip', json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_batch_lookup():
    print("=== 测试批量查询接口 ===")
    data = {
        'ips': ['8.8.8.8', '1.1.1.1', '114.114.114.114', '208.67.222.222']
    }
    response = requests.post(f'{BASE_URL}/api/ip/batch', json=data)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"总数: {result.get('total')}")
    for item in result.get('results', []):
        print(f"  {item['ip']:15} - {item['country']['name']}")
    print()


def test_invalid_ip():
    print("=== 测试无效IP ===")
    ip = '999.999.999.999'
    response = requests.get(f'{BASE_URL}/api/ip/{ip}')
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


if __name__ == '__main__':
    print("IP查询API测试脚本")
    print("请确保API服务已启动 (python app.py)")
    print("=" * 50)
    print()

    try:
        test_health()
        test_get_ip()
        test_post_ip()
        test_batch_lookup()
        test_invalid_ip()
        print("所有测试完成！")
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到API服务")
        print("请先运行: python app.py")
