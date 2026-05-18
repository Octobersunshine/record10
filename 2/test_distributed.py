import io
import time
import numpy as np
import requests


BASE_URL = 'http://localhost:5000'


def test_distributed_status():
    print('=' * 60)
    print('测试分布式计算状态')
    print('=' * 60)
    
    response = requests.get(f'{BASE_URL}/distributed/status')
    data = response.json()
    
    print(f'\n分布式计算启用: {data.get("enabled")}')
    print(f'Redis连接: {data.get("redis_connected")}')
    print(f'阈值: {data.get("threshold")}')
    print(f'块大小: {data.get("block_size")}')
    print(f'待处理任务: {data.get("pending_tasks")}')
    print(f'总结果: {data.get("total_results")}')
    
    if not data.get('redis_connected'):
        print('\n警告: Redis未连接!')
        print('请先启动Redis服务器: redis-server')
        return False
    
    return True


def test_distributed_vs_local():
    print('\n' + '=' * 60)
    print('性能对比: 分布式计算 vs 本地计算')
    print('=' * 60)
    
    size = 3000
    
    print(f'\n矩阵大小: {size}x{size}')
    print(f'分布式阈值: 2000 (大于阈值使用分布式)')
    
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    
    print('\n--- 测试分布式计算 ---')
    data = np.array([a, b])
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    
    start = time.time()
    response = requests.post(
        f'{BASE_URL}/multiply/distributed',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    dist_time = time.time() - start
    dist_result = response.json()
    
    print(f'分布式耗时: {dist_time:.4f}s')
    print(f'使用分布式: {dist_result.get("used_distributed")}')
    print(f'Job ID: {dist_result.get("job_id")}')
    print(f'块数量: {dist_result.get("num_blocks")}')
    print(f'Frobenius范数: {dist_result.get("frobenius_norm"):.4f}')
    
    print('\n--- 测试本地计算 ---')
    data = np.array([a[:2000, :2000], b[:2000, :2000]])
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    
    start = time.time()
    response = requests.post(
        f'{BASE_URL}/multiply',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    local_time = time.time() - start
    local_result = response.json()
    
    print(f'本地耗时: {local_time:.4f}s')
    print(f'使用分布式: {local_result.get("used_distributed")}')
    print(f'Frobenius范数: {local_result.get("frobenius_norm"):.4f}')


def test_correctness():
    print('\n' + '=' * 60)
    print('验证分布式计算正确性')
    print('=' * 60)
    
    size = 2500
    
    print(f'\n矩阵大小: {size}x{size}')
    
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    
    print('计算本地参考结果...')
    expected_result = np.dot(a, b)
    expected_norm = np.linalg.norm(expected_result, 'fro')
    print(f'本地计算完成, 范数: {expected_norm:.4f}')
    
    print('\n发送分布式计算请求...')
    data = np.array([a, b])
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    
    start = time.time()
    response = requests.post(
        f'{BASE_URL}/multiply/distributed',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'},
        timeout=300
    )
    dist_time = time.time() - start
    dist_result = response.json()
    
    print(f'分布式计算完成, 耗时: {dist_time:.4f}s')
    print(f'范数: {dist_result.get("frobenius_norm"):.4f}')
    print(f'使用分布式: {dist_result.get("used_distributed")}')
    
    actual_result = np.array(dist_result['result'])
    
    diff = np.max(np.abs(expected_result - actual_result))
    norm_diff = abs(expected_norm - dist_result['frobenius_norm'])
    
    print(f'\n结果最大差异: {diff:.2e}')
    print(f'范数差异: {norm_diff:.2e}')
    
    if diff < 1e-3 and norm_diff < 1e-3:
        print('✓ 分布式计算结果正确!')
        return True
    else:
        print('✗ 分布式计算结果有差异!')
        return False


def test_auto_switch():
    print('\n' + '=' * 60)
    print('测试自动切换机制')
    print('=' * 60)
    
    sizes = [1000, 2000, 2500]
    
    for size in sizes:
        print(f'\n--- 测试 {size}x{size} 矩阵 ---')
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        
        data = np.array([a, b])
        buffer = io.BytesIO()
        np.save(buffer, data)
        buffer.seek(0)
        
        start = time.time()
        response = requests.post(
            f'{BASE_URL}/multiply',
            data=buffer.read(),
            headers={'Content-Type': 'application/octet-stream'}
        )
        elapsed = time.time() - start
        result = response.json()
        
        print(f'耗时: {elapsed:.4f}s')
        print(f'使用分布式: {result.get("used_distributed")}')
        print(f'使用稀疏: {result.get("used_sparse")}')


def main():
    print('启动分布式计算测试...\n')
    
    if not test_distributed_status():
        print('\n请启动Redis后再运行测试!')
        return
    
    print('\n请确保已启动至少一个worker节点!')
    print('启动命令: python worker.py [worker_id]\n')
    print('按回车键继续...')
    input()
    
    test_auto_switch()
    test_distributed_vs_local()
    test_correctness()
    
    print('\n' + '=' * 60)
    print('所有测试完成!')
    print('=' * 60)


if __name__ == '__main__':
    try:
        main()
    except requests.exceptions.ConnectionError:
        print('错误: 无法连接到服务器。')
        print('请先启动服务器: python app.py')
