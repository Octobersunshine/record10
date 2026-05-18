import numpy as np
import requests
import json

BASE_URL = "http://localhost:5000"

def generate_sample_data():
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(20))
    return X.tolist(), y.tolist()

def example_1_basic_usage():
    print("=" * 60)
    print("示例 1: 直接使用 SVRModel 类")
    print("=" * 60)
    
    from svr_model import SVRModel
    
    X, y = generate_sample_data()
    
    model = SVRModel()
    result = model.train(X, y, kernel='rbf', C=100, epsilon=0.1)
    
    print(f"训练完成!")
    print(f"MSE: {result['mse']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"R²: {result['r2']:.4f}")
    print(f"支持向量数量: {result['support_vectors']}")
    
    X_test = [[x] for x in np.linspace(0, 5, 10)]
    predictions = model.predict(X_test)
    print("\n预测结果:")
    for x, pred in zip(X_test, predictions):
        print(f"  X={x[0]:.2f} -> y={pred:.4f}")

def example_2_api_usage():
    print("\n" + "=" * 60)
    print("示例 2: 使用 REST API")
    print("=" * 60)
    
    X, y = generate_sample_data()
    
    print("1. 检查服务状态...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   状态: {response.json()}")
    except:
        print("   错误: 无法连接到服务，请先运行 python app.py")
        return
    
    print("\n2. 训练模型...")
    train_data = {
        "X": X,
        "y": y,
        "model_id": "my_svr_model",
        "kernel": "rbf",
        "C": 100,
        "epsilon": 0.1,
        "test_size": 0.2
    }
    response = requests.post(f"{BASE_URL}/api/train", json=train_data)
    result = response.json()
    print(f"   {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\n3. 进行预测...")
    predict_data = {
        "X": [[0.5], [1.5], [2.5], [3.5], [4.5]],
        "model_id": "my_svr_model"
    }
    response = requests.post(f"{BASE_URL}/api/predict", json=predict_data)
    result = response.json()
    print(f"   {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\n4. 获取模型信息...")
    response = requests.get(f"{BASE_URL}/api/model/my_svr_model")
    result = response.json()
    print(f"   {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\n5. 保存模型...")
    response = requests.post(f"{BASE_URL}/api/model/my_svr_model/save")
    result = response.json()
    print(f"   {json.dumps(result, indent=2, ensure_ascii=False)}")

def example_3_different_kernels():
    print("\n" + "=" * 60)
    print("示例 3: 对比不同核函数效果")
    print("=" * 60)
    
    from svr_model import SVRModel
    
    X, y = generate_sample_data()
    
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    for kernel in kernels:
        model = SVRModel()
        result = model.train(X, y, kernel=kernel, C=100, epsilon=0.1)
        print(f"{kernel:10s} - R²: {result['r2']:.4f}, 支持向量: {result['support_vectors']}")

def example_4_robust_svr():
    print("\n" + "=" * 60)
    print("示例 4: 鲁棒SVR - 解决异常值敏感问题")
    print("=" * 60)
    
    from robust_svr import RobustSVRModel
    
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(50, 1), axis=0)
    y = np.sin(X).ravel() + 0.2 * np.random.randn(50)
    
    outlier_idx = [10, 25, 40]
    for idx in outlier_idx:
        y[idx] += 8 * (1 if idx % 2 == 0 else -1)
    
    print(f"添加了 {len(outlier_idx)} 个异常值: {outlier_idx}")
    print()
    
    print("A. 不使用异常值处理:")
    model1 = RobustSVRModel()
    res1 = model1.train(X, y, kernel='rbf', C=100, epsilon=0.1, remove_outliers=False)
    print(f"   支持向量数: {res1['support_vectors']}")
    print(f"   R²: {res1['r2']:.4f}")
    
    print("\nB. 使用异常值检测 + 移除:")
    model2 = RobustSVRModel()
    res2 = model2.train(X, y, kernel='rbf', C=100, epsilon=0.1, remove_outliers=True)
    outlier_info = model2.get_outlier_info()
    print(f"   检测到异常值: {outlier_info['outlier_indices']}")
    print(f"   支持向量数: {res2['support_vectors']}")
    print(f"   R²: {res2['r2']:.4f}")
    
    print("\nC. 使用Nu-SVR (控制支持向量比例):")
    model3 = RobustSVRModel()
    res3 = model3.train(X, y, kernel='rbf', C=100, nu=0.3, use_nusvr=True, remove_outliers=True)
    print(f"   支持向量数: {res3['support_vectors']}")
    print(f"   支持向量比例: {res3['sv_ratio']:.2%}")
    print(f"   R²: {res3['r2']:.4f}")
    
    print(f"\n支持向量减少: {res1['support_vectors'] - res3['support_vectors']} 个 ({(res1['support_vectors']-res3['support_vectors'])/res1['support_vectors']:.1%})")


if __name__ == "__main__":
    example_1_basic_usage()
    example_3_different_kernels()
    example_4_robust_svr()
    print("\n" + "=" * 60)
    print("提示: 运行 'python app.py' 启动API服务后,")
    print("      再运行此脚本可测试 API 功能")
    print("=" * 60)
    print("\n运行 'python test_robustness.py' 可查看完整鲁棒性对比测试")
