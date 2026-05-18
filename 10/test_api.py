import requests
import json
import numpy as np
import sys
sys.path.append('.')
from app import kmeans_with_empty_cluster_handling, recommend_k_with_elbow, calculate_sse


def test_kmeans_api():
    url = 'http://localhost:5000/kmeans'
    
    data = {
        'points': [
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0]
        ],
        'k': 2
    }
    
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    print('=' * 50)
    print('Test 1: Basic 2D K-means')
    print('Status Code:', response.status_code)
    print('Response:', json.dumps(response.json(), indent=2))


def test_3d_kmeans():
    url = 'http://localhost:5000/kmeans'
    
    data = {
        'points': [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0],
            [1.5, 2.5, 3.5],
            [10.5, 11.5, 12.5]
        ],
        'k': 2
    }
    
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    print('\n' + '=' * 50)
    print('Test 2: 3D K-means')
    print('Status Code:', response.status_code)
    print('Response:', json.dumps(response.json(), indent=2))


def test_empty_cluster_handling_direct():
    print('\n' + '=' * 50)
    print('Test 3: Direct test of empty cluster handling')
    print('Strategy: Pick random point from largest cluster')
    
    X = np.array([
        [1.0, 1.0],
        [1.1, 1.1],
        [1.2, 1.2],
        [10.0, 10.0],
        [10.1, 10.1]
    ])
    
    print('\nData points:')
    print(X)
    print(f'\nRunning K-means with k=3 (potential empty cluster scenario)...')
    
    try:
        labels, centroids = kmeans_with_empty_cluster_handling(X, k=3)
        print(f'\nSuccess! No division by zero error.')
        print(f'Labels: {labels}')
        print(f'Centroids:\n{centroids}')
        
        cluster_sizes = [np.sum(labels == i) for i in range(3)]
        print(f'\nCluster sizes: {cluster_sizes}')
        print(f'All clusters have points: {all(size > 0 for size in cluster_sizes)}')
        print(f'K value preserved: {len(centroids) == 3}')
        
    except Exception as e:
        print(f'Error: {e}')


def test_multiple_empty_clusters():
    print('\n' + '=' * 50)
    print('Test 5: Multiple empty clusters scenario')
    
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [10.0, 10.0],
        [10.1, 10.1]
    ])
    
    print(f'\nData points: {len(X)} points')
    print(f'Running K-means with k=4 (multiple potential empty clusters)...')
    
    try:
        labels, centroids = kmeans_with_empty_cluster_handling(X, k=4)
        print(f'\nSuccess! No division by zero error.')
        
        cluster_sizes = [np.sum(labels == i) for i in range(4)]
        print(f'Cluster sizes: {cluster_sizes}')
        print(f'All clusters have points: {all(size > 0 for size in cluster_sizes)}')
        print(f'K value preserved: {len(centroids) == 4}')
        
    except Exception as e:
        print(f'Error: {e}')


def test_edge_case_k_equals_n():
    print('\n' + '=' * 50)
    print('Test 4: Edge case - k equals number of points')
    
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    
    labels, centroids = kmeans_with_empty_cluster_handling(X, k=3)
    print(f'Labels: {labels}')
    print(f'Centroids equal to points: {np.allclose(centroids, X)}')


def test_elbow_method():
    print('\n' + '=' * 50)
    print('Test 6: Elbow Method for K recommendation')
    
    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [10.0, 10.0],
        [10.5, 10.5],
        [11.0, 11.0],
        [20.0, 20.0],
        [20.5, 20.5],
        [21.0, 21.0]
    ])
    
    print(f'\nData points: {len(X)} points (3 distinct clusters)')
    print('Running elbow method...')
    
    recommended_k, k_values, sse_values = recommend_k_with_elbow(X)
    
    print(f'\nRecommended K: {recommended_k}')
    print('\nK values and SSE:')
    for k, sse in zip(k_values, sse_values):
        print(f'  K={k}: SSE={sse:.4f}')
    
    print(f'\nExpected K around 3 (3 distinct clusters)')
    print(f'Recommendation matches expectation: {recommended_k in [2, 3, 4]}')


def test_elbow_api():
    print('\n' + '=' * 50)
    print('Test 7: Elbow Method API Endpoint')
    
    url = 'http://localhost:5000/kmeans/recommend-k'
    
    data = {
        'points': [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [10.0, 10.0],
            [10.5, 10.5],
            [11.0, 11.0]
        ],
        'max_k': 5
    }
    
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    print('Status Code:', response.status_code)
    if response.status_code == 200:
        result = response.json()
        print(f'Recommended K: {result["recommended_k"]}')
        print(f'K values: {result["k_values"]}')
        print(f'SSE values: {[round(sse, 4) for sse in result["sse_values"]]}')


if __name__ == '__main__':
    test_empty_cluster_handling_direct()
    test_multiple_empty_clusters()
    test_edge_case_k_equals_n()
    test_elbow_method()
    
    try:
        test_kmeans_api()
        test_3d_kmeans()
        test_elbow_api()
    except:
        print('\nNote: Start the server with "python app.py" first to run API tests')
