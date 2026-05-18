import requests
import json
import numpy as np

BASE_URL = "http://localhost:5000"


def test_health():
    print("测试健康检查接口...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    print()


def test_iterative_refinement_enabled():
    print("测试启用迭代refinement的良态矩阵...")
    
    A = [[2, 1], [1, 1]]
    b = [3, 2]
    
    payload = {"A": A, "b": b, "method": "lu", "iterative_refine": True}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"使用方法: {result.get('used_method')}")
    print(f"解 x: {result.get('x')}")
    print(f"相对残差: {result.get('relative_residual'):.2e}")
    
    refine_info = result.get('iterative_refinement', {})
    if refine_info.get('enabled'):
        print(f"迭代次数: {refine_info.get('iterations')}")
        print(f"初始残差: {refine_info.get('initial_residual'):.2e}")
        print(f"最终残差: {refine_info.get('final_residual'):.2e}")
        print(f"残差收敛历史: {[f'{r:.2e}' for r in refine_info.get('residual_history', [])]}")
    print()


def test_iterative_refinement_disabled():
    print("测试禁用迭代refinement...")
    
    A = [[2, 1], [1, 1]]
    b = [3, 2]
    
    payload = {"A": A, "b": b, "method": "lu", "iterative_refine": False}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"使用方法: {result.get('used_method')}")
    print(f"解 x: {result.get('x')}")
    print(f"相对残差: {result.get('relative_residual'):.2e}")
    print(f"迭代refinement启用: {result.get('iterative_refinement', {}).get('enabled')}")
    print()


def test_refinement_vs_no_refinement_comparison():
    print("对比启用 vs 禁用迭代refinement...")
    
    epsilon = 1e-10
    A = [[1, 1], [1, 1 + epsilon]]
    b = [2, 2 + epsilon]
    
    cond = np.linalg.cond(A)
    print(f"条件数: {cond:.2e}")
    print(f"预期解: x ≈ [1, 1]")
    print()
    
    print("禁用迭代refinement:")
    payload = {"A": A, "b": b, "method": "truncated_svd", "iterative_refine": False}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    result = response.json()
    x_no_refine = result.get('x')
    res_no_refine = result.get('relative_residual')
    print(f"  解 x: {x_no_refine}")
    print(f"  相对残差: {res_no_refine:.2e}")
    print()
    
    print("启用迭代refinement:")
    payload = {"A": A, "b": b, "method": "truncated_svd", "iterative_refine": True}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    result = response.json()
    x_refine = result.get('x')
    res_refine = result.get('relative_residual')
    refine_info = result.get('iterative_refinement', {})
    print(f"  解 x: {x_refine}")
    print(f"  相对残差: {res_refine:.2e}")
    print(f"  迭代次数: {refine_info.get('iterations')}")
    print(f"  残差收敛历史: {[f'{r:.2e}' for r in refine_info.get('residual_history', [])]}")
    print()
    
    if res_refine < res_no_refine:
        improvement = (1 - res_refine / res_no_refine) * 100
        print(f"残差改进: {improvement:.2f}%")
    print()


def test_custom_iteration_parameters():
    print("测试自定义迭代参数...")
    
    epsilon = 1e-11
    A = [[1, 1], [1, 1 + epsilon]]
    b = [2, 2 + epsilon]
    
    print("默认参数 (max_iter=10, tol=1e-12):")
    payload = {"A": A, "b": b, "method": "truncated_svd", "iterative_refine": True}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    result = response.json()
    refine_info = result.get('iterative_refinement', {})
    print(f"  迭代次数: {refine_info.get('iterations')}")
    print(f"  最终残差: {refine_info.get('final_residual'):.2e}")
    print()
    
    print("严格参数 (max_iter=20, tol=1e-15):")
    payload = {"A": A, "b": b, "method": "truncated_svd", "iterative_refine": True,
               "max_iterations": 20, "residual_tolerance": 1e-15}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    result = response.json()
    refine_info = result.get('iterative_refinement', {})
    print(f"  迭代次数: {refine_info.get('iterations')}")
    print(f"  最终残差: {refine_info.get('final_residual'):.2e}")
    print()
    
    print("较少迭代 (max_iter=3, tol=1e-12):")
    payload = {"A": A, "b": b, "method": "truncated_svd", "iterative_refine": True,
               "max_iterations": 3, "residual_tolerance": 1e-12}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    result = response.json()
    refine_info = result.get('iterative_refinement', {})
    print(f"  迭代次数: {refine_info.get('iterations')}")
    print(f"  最终残差: {refine_info.get('final_residual'):.2e}")
    print()


def test_hilbert_matrix_with_refinement():
    print("测试Hilbert矩阵使用迭代refinement...")
    
    n = 6
    A = [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]
    x_true = [1.0] * n
    b = [sum(A[i][j] * x_true[j] for j in range(n)) for i in range(n)]
    
    cond = np.linalg.cond(A)
    print(f"Hilbert矩阵 {n}x{n}")
    print(f"条件数: {cond:.2e}")
    print(f"预期解: x = {x_true}")
    print()
    
    print("禁用迭代refinement:")
    payload = {"A": A, "b": b, "method": "truncated_svd", "iterative_refine": False}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    result = response.json()
    x_no = result.get('x')
    err_no = np.linalg.norm(np.array(x_no) - np.array(x_true))
    print(f"  解 x: {[f'{v:.6f}' for v in x_no]}")
    print(f"  相对残差: {result.get('relative_residual'):.2e}")
    print(f"  解误差: {err_no:.2e}")
    print()
    
    print("启用迭代refinement:")
    payload = {"A": A, "b": b, "method": "truncated_svd", "iterative_refine": True,
               "max_iterations": 15, "residual_tolerance": 1e-14}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    result = response.json()
    x_ref = result.get('x')
    err_ref = np.linalg.norm(np.array(x_ref) - np.array(x_true))
    refine_info = result.get('iterative_refinement', {})
    print(f"  解 x: {[f'{v:.6f}' for v in x_ref]}")
    print(f"  相对残差: {result.get('relative_residual'):.2e}")
    print(f"  解误差: {err_ref:.2e}")
    print(f"  迭代次数: {refine_info.get('iterations')}")
    print()
    
    if err_ref < err_no:
        improvement = (1 - err_ref / err_no) * 100
        print(f"解误差改进: {improvement:.2f}%")
    print()


def test_batch_with_refinement():
    print("测试批量求解带迭代refinement...")
    
    problems = [
        {
            "A": [[2, 1], [1, 1]],
            "b": [3, 2],
            "iterative_refine": True
        },
        {
            "A": [[1, 1], [1, 1 + 1e-11]],
            "b": [2, 2 + 1e-11],
            "iterative_refine": True,
            "max_iterations": 15
        },
        {
            "A": [[1, 2], [3, 4]],
            "b": [5, 11],
            "iterative_refine": False
        }
    ]
    
    payload = {"problems": problems, "method": "auto"}
    response = requests.post(f"{BASE_URL}/batch_solve", json=payload)
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"总数: {result.get('total')}, 成功: {result.get('success_count')}")
    print()
    
    for i, r in enumerate(result.get('results', [])):
        if r.get('success'):
            refine_info = r.get('iterative_refinement', {})
            refine_enabled = refine_info.get('enabled', False)
            print(f"问题 {i}: 方法={r.get('used_method')}, 条件数={r.get('condition_number'):.2e}")
            print(f"       refine={'启用' if refine_enabled else '禁用'}, ", end="")
            if refine_enabled:
                print(f"迭代={refine_info.get('iterations')}次, 残差={r.get('relative_residual'):.2e}")
            else:
                print(f"残差={r.get('relative_residual'):.2e}")
        else:
            print(f"问题 {i}: 失败 - {r.get('error')}")
    print()


def test_lu_refinement_fallback():
    print("测试LU分解refinement失败时自动回退到伪逆...")
    
    A = [[1e-15, 1], [1, 1]]
    b = [1, 2]
    
    cond = np.linalg.cond(A)
    print(f"条件数: {cond:.2e}")
    
    payload = {"A": A, "b": b, "method": "lu", "iterative_refine": True}
    response = requests.post(f"{BASE_URL}/solve", json=payload)
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"使用方法: {result.get('used_method')}")
    print(f"解 x: {result.get('x')}")
    print(f"相对残差: {result.get('relative_residual'):.2e}")
    
    refine_info = result.get('iterative_refinement', {})
    if refine_info.get('enabled'):
        print(f"迭代次数: {refine_info.get('iterations')}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("线性方程组求解 API - 迭代Refinement测试")
    print("=" * 70)
    print()
    
    try:
        test_health()
        test_iterative_refinement_enabled()
        test_iterative_refinement_disabled()
        test_refinement_vs_no_refinement_comparison()
        test_custom_iteration_parameters()
        test_hilbert_matrix_with_refinement()
        test_batch_with_refinement()
        test_lu_refinement_fallback()
        
        print("=" * 70)
        print("所有测试完成!")
        print("=" * 70)
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器。请先运行 'python app.py' 启动服务器。")
