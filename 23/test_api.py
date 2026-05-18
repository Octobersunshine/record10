import requests
import json

API_URL = "http://localhost:8000/train"

test_data = {
    "X": [
        [1.2, 3.4, 0.8, 5.1],
        [2.1, 4.5, 1.2, 6.2],
        [0.9, 2.8, 0.5, 4.8],
        [3.2, 5.6, 2.1, 7.3],
        [1.8, 3.9, 0.9, 5.5],
        [2.5, 4.8, 1.5, 6.5],
        [0.7, 2.5, 0.4, 4.2],
        [3.5, 5.9, 2.3, 7.8],
        [1.5, 3.7, 0.7, 5.3],
        [2.8, 5.2, 1.8, 6.8]
    ],
    "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "n_estimators": 100,
    "random_state": 42,
    "task_type": "classification",
    "perform_significance_test": True,
    "n_permutations": 50,
    "alpha": 0.05
}


def test_train_endpoint():
    print("=" * 80)
    print("测试随机森林特征重要性API（含显著性检验）")
    print("=" * 80)
    
    try:
        response = requests.post(API_URL, json=test_data)
        
        print(f"\n状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n成功: {result['success']}")
            print(f"消息: {result['message']}")
            print(f"模型类型: {result['model_type']}")
            print(f"总特征数: {result['total_features']}")
            print(f"显著特征数: {result['significant_count']}")
            print(f"显著性检验已执行: {result['significance_test_performed']}")
            
            print("\n" + "=" * 80)
            print("所有特征重要性排序（含p值）")
            print("=" * 80)
            print(f"{'排名':<6} {'特征索引':<12} {'重要性':<12} {'p值':<12} {'显著':<8}")
            print("-" * 80)
            
            total_importance = sum(f['importance'] for f in result['feature_importances'])
            
            for feat in result['feature_importances']:
                p_val = f"{feat['p_value']:.4f}" if feat['p_value'] is not None else "N/A"
                is_sig = "✓" if feat['is_significant'] else "✗"
                print(f"{feat['rank']:<6} {feat['feature_index']:<12} {feat['importance']:<12.6f} {p_val:<12} {is_sig:<8}")
            
            print("=" * 80)
            print(f"\n总重要性和: {total_importance:.6f} (应为1.0)")
            
            if result['significant_features']:
                print("\n" + "=" * 80)
                print("显著特征列表 (p < α)")
                print("=" * 80)
                print(f"{'排名':<6} {'特征索引':<12} {'重要性':<12} {'p值':<12}")
                print("-" * 80)
                for feat in result['significant_features']:
                    print(f"{feat['rank']:<6} {feat['feature_index']:<12} {feat['importance']:<12.6f} {feat['p_value']:<12.4f}")
                print("=" * 80)
            else:
                print("\n⚠️  未检测到显著特征")
            
        else:
            print(f"错误: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n错误: 无法连接到API服务器")
        print("请先运行以下命令启动服务器:")
        print("  python main.py")
        print("  或")
        print("  uvicorn main:app --reload")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")


def print_sample_request():
    print("\n" + "=" * 80)
    print("示例请求数据（含显著性检验参数）")
    print("=" * 80)
    print(json.dumps(test_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print_sample_request()
    test_train_endpoint()
