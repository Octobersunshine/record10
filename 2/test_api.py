import io
import json
import numpy as np
import requests


BASE_URL = 'http://localhost:5000'


def test_single_matrix_json():
    print('Test 1: Single matrix multiplication (JSON)')
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    response = requests.post(f'{BASE_URL}/multiply', json={
        'matrix_a': matrix_a,
        'matrix_b': matrix_b
    })
    print(f'Status: {response.status_code}')
    print(f'Response: {json.dumps(response.json(), indent=2)}')
    print()


def test_batch_matrix_json():
    print('Test 2: Batch matrix multiplication (JSON)')
    batch_data = [
        {'matrix_a': [[1, 2], [3, 4]], 'matrix_b': [[5, 6], [7, 8]]},
        {'matrix_a': [[1, 0], [0, 1]], 'matrix_b': [[2, 0], [0, 2]]},
        {'matrix_a': [[1, 1], [1, 1]], 'matrix_b': [[1, 1], [1, 1]]}
    ]
    response = requests.post(f'{BASE_URL}/multiply', json={'batch': batch_data})
    print(f'Status: {response.status_code}')
    print(f'Response: {json.dumps(response.json(), indent=2)}')
    print()


def test_single_matrix_binary():
    print('Test 3: Single matrix multiplication (binary)')
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])
    data = np.array([matrix_a, matrix_b])
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    response = requests.post(
        f'{BASE_URL}/multiply',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    print(f'Status: {response.status_code}')
    print(f'Response: {json.dumps(response.json(), indent=2)}')
    print()


def test_batch_matrix_binary():
    print('Test 4: Batch matrix multiplication (binary)')
    batch_data = np.array([
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [[[1, 0], [0, 1]], [[2, 0], [0, 2]]]
    ])
    buffer = io.BytesIO()
    np.save(buffer, batch_data)
    buffer.seek(0)
    response = requests.post(
        f'{BASE_URL}/multiply',
        data=buffer.read(),
        headers={'Content-Type': 'application/octet-stream'}
    )
    print(f'Status: {response.status_code}')
    print(f'Response: {json.dumps(response.json(), indent=2)}')
    print()


def test_metrics():
    print('Test 5: Prometheus metrics')
    response = requests.get(f'{BASE_URL}/metrics')
    print(f'Status: {response.status_code}')
    print('Metrics (first 500 chars):')
    print(response.text[:500])
    print()


def test_health():
    print('Test 6: Health check')
    response = requests.get(f'{BASE_URL}/health')
    print(f'Status: {response.status_code}')
    print(f'Response: {json.dumps(response.json(), indent=2)}')
    print()


if __name__ == '__main__':
    try:
        test_health()
        test_single_matrix_json()
        test_batch_matrix_json()
        test_single_matrix_binary()
        test_batch_matrix_binary()
        test_metrics()
        print('All tests completed!')
    except requests.exceptions.ConnectionError:
        print('Error: Could not connect to server.')
        print('Please start the server first with: python app.py')
