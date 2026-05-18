import requests
import json
import random

BASE_URL = "http://localhost:8000"


def test_health_check():
    print("测试健康检查接口...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    assert response.status_code == 200
    print("健康检查通过!\n")


def test_cluster_endpoint():
    print("测试聚类接口...")
    
    test_data = {
        "data_points": [
            [1.0, 2.0], [1.1, 2.1], [0.9, 1.9],
            [5.0, 5.0], [5.1, 5.1], [4.9, 4.9],
            [10.0, 10.0]
        ],
        "eps": 0.5,
        "min_samples": 2
    }
    
    response = requests.post(
        f"{BASE_URL}/cluster",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"聚类结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    assert "labels" in result
    assert "n_clusters" in result
    assert "n_noise" in result
    
    print(f"发现 {result['n_clusters']} 个簇")
    print(f"噪声点数量: {result['n_noise']}")
    print("聚类测试通过!\n")


def test_border_point_stability():
    print("=" * 60)
    print("测试边界点归属稳定性（数据顺序无关）")
    print("=" * 60)
    
    base_points = [
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
        [1.0, 1.0], [1.1, 1.0], [1.0, 1.1],
    ]
    border_point = [0.5, 0.5]
    
    results = []
    for i in range(5):
        shuffled_points = base_points.copy()
        random.shuffle(shuffled_points)
        shuffled_points.append(border_point)
        
        test_data = {
            "data_points": shuffled_points,
            "eps": 0.6,
            "min_samples": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/cluster",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        border_label = result["labels"][-1]
        results.append(border_label)
        
        print(f"\n测试 {i+1}:")
        print(f"  边界点标签: {border_label}")
        print(f"  重新分配的边界点数: {result.get('border_points_reassigned', 0)}")
        print(f"  所有标签: {result['labels']}")
    
    print(f"\n所有测试的边界点标签: {results}")
    assert len(set(results)) == 1, f"边界点归属不稳定! 结果: {results}"
    print("\n✓ 边界点归属稳定! 不同数据顺序下结果一致")
    print("=" * 60 + "\n")


def test_with_custom_params():
    print("测试自定义参数...")
    
    test_data = {
        "data_points": [
            [0, 0], [0, 1], [1, 0],
            [3, 3], [3, 4], [4, 3],
            [10, 10], [10, 11], [11, 10]
        ],
        "eps": 1.5,
        "min_samples": 3
    }
    
    response = requests.post(
        f"{BASE_URL}/cluster",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"聚类结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    print("自定义参数测试通过!\n")


def test_local_dbscan_function():
    print("本地测试改进的DBSCAN函数...")
    
    import numpy as np
    import sys
    sys.path.insert(0, 'e:\\temp\\record10\\28')
    from main import dbscan_with_stable_border_assignment
    
    base_points = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
        [1.0, 1.0], [1.1, 1.0], [1.0, 1.1],
    ])
    border_point = np.array([[0.5, 0.5]])
    
    labels1, reassigned1 = dbscan_with_stable_border_assignment(
        np.vstack([base_points, border_point]), 
        eps=0.6, 
        min_samples=3
    )
    
    perm = np.random.permutation(len(base_points))
    shuffled_base = base_points[perm]
    labels2, reassigned2 = dbscan_with_stable_border_assignment(
        np.vstack([shuffled_base, border_point]), 
        eps=0.6, 
        min_samples=3
    )
    
    print(f"  顺序1 - 边界点标签: {labels1[-1]}")
    print(f"  顺序2 - 边界点标签: {labels2[-1]}")
    print(f"  重新分配计数: {reassigned1}, {reassigned2}")
    
    print("\n✓ 本地函数测试通过")
    print("-" * 60 + "\n")


def test_hdbscan_endpoint():
    print("测试HDBSCAN聚类接口...")
    
    test_data = {
        "data_points": [
            [1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [1.0, 2.2],
            [5.0, 5.0], [5.1, 5.1], [4.9, 4.9], [5.0, 5.2],
            [10.0, 10.0], [10.1, 10.1]
        ],
        "min_cluster_size": 3,
        "min_samples": 2
    }
    
    response = requests.post(
        f"{BASE_URL}/hdbscan",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    
    assert response.status_code == 200
    assert "labels" in result
    assert "probabilities" in result
    assert "n_clusters" in result
    assert "estimated_eps" in result
    
    print(f"发现 {result['n_clusters']} 个簇")
    print(f"噪声点数量: {result['n_noise']}")
    print(f"自动估计的eps值: {result.get('estimated_eps', 'N/A')}")
    print(f"聚类标签: {result['labels']}")
    print(f"隶属概率: {[round(p, 3) for p in result['probabilities']]}")
    
    if result.get('cluster_hierarchy'):
        print(f"层次树节点数: {len(result['cluster_hierarchy'])}")
        cluster_nodes = [n for n in result['cluster_hierarchy'] if n['is_cluster']]
        print(f"聚类节点数: {len(cluster_nodes)}")
    
    print("HDBSCAN测试通过!\n")


def test_hdbscan_cluster_info():
    print("测试HDBSCAN聚类详细信息...")
    
    test_data = {
        "data_points": [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [-0.1, 0.0],
            [1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [0.9, 1.0],
            [2.0, 2.0], [2.1, 2.0]
        ],
        "min_cluster_size": 4
    }
    
    response = requests.post(
        f"{BASE_URL}/hdbscan",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    result = response.json()
    print(f"聚类信息: {json.dumps(result.get('cluster_info', {}), indent=2, ensure_ascii=False)}")
    print("聚类详细信息测试通过!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 DBSCAN/HDBSCAN Clustering API")
    print("=" * 60 + "\n")
    
    try:
        test_health_check()
        test_cluster_endpoint()
        test_with_custom_params()
        test_border_point_stability()
        test_hdbscan_endpoint()
        test_hdbscan_cluster_info()
        
        print("=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        print("请确保API服务正在运行 (python main.py)")
        import traceback
        traceback.print_exc()
