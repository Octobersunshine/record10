import io
import time
import numpy as np
import requests


BASE_URL = 'http://localhost:5000'


def generate_sparse_matrix(rows, cols, density):
    mask = np.random.rand(rows, cols) < density
    matrix = np.random.rand(rows, cols) * mask
    return matrix


def test_sparse_vs_dense():
    print('=' * 60)
    print('性能对比: 稀疏矩阵 vs 稠密矩阵')
    print('=' * 60)
    
    size = 1000
    sparse_density = 0.05
    dense_density = 0.5
    
    print(f'\n矩阵大小: {size}x{size}')
    print(f'稀疏密度: {sparse_density * 100}% (应使用稀疏乘法)')
    print(f'稠密密度: {dense_density * 100}% (应使用稠密乘法)')
    
    print('\n--- 测试稀疏矩阵 (密度 5%) ---')
    a_sparse = generate_sparse_matrix(size, size, sparse_density)
    b_sparse = generate_sparse_matrix(size, size, sparse_density)
    
    print(f'A 实际密度: {np.count_nonzero(a_sparse) / a_sparse.size * 100:.2f}%')
    print(f'B 实际密度: {np.count_nonzero(b_sparse) / b_sparse.size * 100:.2f}%')
    
    data = np.array([a_sparse, b_sparse])
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    
    start = time.time()
    response = requests.post(
        f'{BASE_URL}/multiply',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    sparse_time = time.time() - start
    result = response.json()
    print(f'稀疏乘法耗时: {sparse_time:.4f}s')
    print(f'使用稀疏路径: {result.get("used_sparse", False)}')
    
    print('\n--- 测试稠密矩阵 (密度 50%) ---')
    a_dense = generate_sparse_matrix(size, size, dense_density)
    b_dense = generate_sparse_matrix(size, size, dense_density)
    
    print(f'A 实际密度: {np.count_nonzero(a_dense) / a_dense.size * 100:.2f}%')
    print(f'B 实际密度: {np.count_nonzero(b_dense) / b_dense.size * 100:.2f}%')
    
    data = np.array([a_dense, b_dense])
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    
    start = time.time()
    response = requests.post(
        f'{BASE_URL}/multiply',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    dense_time = time.time() - start
    result = response.json()
    print(f'稠密乘法耗时: {dense_time:.4f}s')
    print(f'使用稀疏路径: {result.get("used_sparse", False)}')
    
    print(f'\n稀疏/稠密性能比: {dense_time / sparse_time:.2f}x')
    
    return sparse_time, dense_time


def test_density_threshold():
    print('\n' + '=' * 60)
    print('密度阈值测试 (阈值 10%)')
    print('=' * 60)
    
    size = 500
    densities = [0.01, 0.05, 0.09, 0.10, 0.11, 0.15, 0.20, 0.50]
    
    print(f'\n矩阵大小: {size}x{size}')
    print(f'阈值: 10%')
    print()
    print(f'{"密度":>10} {"耗时(s)":>12} {"使用稀疏":>10}')
    print('-' * 35)
    
    for density in densities:
        a = generate_sparse_matrix(size, size, density)
        b = generate_sparse_matrix(size, size, density)
        
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
        
        print(f'{density * 100:>8.1f}% {elapsed:>12.4f} {str(result.get("used_sparse", False)):>10}')


def test_contiguous_vs_non_contiguous():
    print('\n' + '=' * 60)
    print('性能对比: 连续内存 vs 非连续内存（转置视图）')
    print('=' * 60)
    
    size = 500
    a_orig = np.random.rand(size, size)
    b_orig = np.random.rand(size, size)
    
    a_contig = a_orig.copy(order='C')
    b_contig = b_orig.copy(order='C')
    
    a_trans = a_orig.T
    b_trans = b_orig.T
    
    print(f'\n矩阵大小: {size}x{size}')
    print(f'A 连续: {a_contig.flags.c_contiguous}, A 转置连续: {a_trans.flags.c_contiguous}')
    print(f'B 连续: {b_contig.flags.c_contiguous}, B 转置连续: {b_trans.flags.c_contiguous}')
    
    print('\n--- 测试连续内存矩阵 ---')
    data_contig = np.array([a_contig, b_contig])
    buffer = io.BytesIO()
    np.save(buffer, data_contig)
    buffer.seek(0)
    
    start = time.time()
    response = requests.post(
        f'{BASE_URL}/multiply',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    contig_time = time.time() - start
    print(f'连续内存: {contig_time:.4f}s')
    
    print('\n--- 测试非连续内存（转置）矩阵 ---')
    data_trans = np.array([a_trans, b_trans])
    buffer = io.BytesIO()
    np.save(buffer, data_trans)
    buffer.seek(0)
    
    start = time.time()
    response = requests.post(
        f'{BASE_URL}/multiply',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    trans_time = time.time() - start
    print(f'非连续内存: {trans_time:.4f}s')
    print(f'性能差异: {(trans_time - contig_time) / contig_time * 100:.2f}%')
    
    return contig_time, trans_time


def test_batch_performance():
    print('\n' + '=' * 60)
    print('批量处理性能测试')
    print('=' * 60)
    
    batch_size = 10
    size = 200
    
    batch_data = []
    for i in range(batch_size):
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        batch_data.append({'matrix_a': a.tolist(), 'matrix_b': b.tolist()})
    
    print(f'\n批量大小: {batch_size}, 矩阵大小: {size}x{size}')
    
    start = time.time()
    response = requests.post(f'{BASE_URL}/multiply', json={'batch': batch_data})
    elapsed = time.time() - start
    
    print(f'总耗时: {elapsed:.4f}s')
    print(f'平均每对: {elapsed / batch_size:.4f}s')
    
    return elapsed


def test_metrics():
    print('\n' + '=' * 60)
    print('查看 Prometheus 指标')
    print('=' * 60)
    
    response = requests.get(f'{BASE_URL}/metrics')
    metrics_text = response.text
    
    print('\n--- 稀疏矩阵相关指标 ---')
    for line in metrics_text.split('\n'):
        if 'sparse_' in line and 'bucket' not in line:
            print(line)
    
    print('\n--- 矩阵拷贝相关指标 ---')
    for line in metrics_text.split('\n'):
        if 'matrix_copy' in line and 'bucket' not in line:
            print(line)
    
    print('\n--- 请求相关指标 ---')
    for line in metrics_text.split('\n'):
        if 'matrix_request' in line and 'bucket' not in line:
            print(line)


def verify_correctness():
    print('\n' + '=' * 60)
    print('验证结果正确性')
    print('=' * 60)
    
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    
    response = requests.post(f'{BASE_URL}/multiply', json={
        'matrix_a': a,
        'matrix_b': b
    })
    result = response.json()
    
    expected = [[19, 22], [43, 50]]
    print(f'\n预期结果: {expected}')
    print(f'实际结果: {result["result"]}')
    print(f'Frobenius 范数: {result["frobenius_norm"]:.4f}')
    print(f'使用稀疏路径: {result.get("used_sparse", False)}')
    
    if result['result'] == expected:
        print('✓ 结果正确!')
        return True
    else:
        print('✗ 结果错误!')
        return False


def verify_sparse_correctness():
    print('\n' + '=' * 60)
    print('验证稀疏矩阵结果正确性')
    print('=' * 60)
    
    size = 100
    density = 0.05
    
    a = generate_sparse_matrix(size, size, density)
    b = generate_sparse_matrix(size, size, density)
    
    expected_result = np.dot(a, b)
    expected_norm = np.linalg.norm(expected_result, 'fro')
    
    data = np.array([a, b])
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    
    response = requests.post(
        f'{BASE_URL}/multiply',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    result = response.json()
    
    actual_result = np.array(result['result'])
    actual_norm = result['frobenius_norm']
    
    result_diff = np.max(np.abs(expected_result - actual_result))
    norm_diff = abs(expected_norm - actual_norm)
    
    print(f'\n矩阵大小: {size}x{size}, 密度: {density * 100}%')
    print(f'使用稀疏路径: {result.get("used_sparse", False)}')
    print(f'结果最大差异: {result_diff:.2e}')
    print(f'范数差异: {norm_diff:.2e}')
    
    if result_diff < 1e-10 and norm_diff < 1e-10:
        print('✓ 稀疏矩阵结果正确!')
        return True
    else:
        print('✗ 稀疏矩阵结果有差异!')
        return False


if __name__ == '__main__':
    try:
        print('启动性能测试...\n')
        
        correct1 = verify_correctness()
        correct2 = verify_sparse_correctness()
        
        if correct1 and correct2:
            sparse_t, dense_t = test_sparse_vs_dense()
            test_density_threshold()
            contig_t, trans_t = test_contiguous_vs_non_contiguous()
            batch_t = test_batch_performance()
            test_metrics()
            
            print('\n' + '=' * 60)
            print('测试完成!')
            print('=' * 60)
        else:
            print('\n结果正确性验证失败，请检查代码!')
            
    except requests.exceptions.ConnectionError:
        print('错误: 无法连接到服务器。')
        print('请先启动服务器: python app.py')
