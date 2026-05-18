import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from online_svr import OnlineSVRModel


def generate_time_series_data(n_samples=200, noise=0.1, drift=True):
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    
    if drift:
        y = np.sin(X).ravel() + 0.1 * X.ravel() + noise * np.random.randn(n_samples)
    else:
        y = np.sin(X).ravel() + noise * np.random.randn(n_samples)
    
    return X, y


def test_online_learning():
    print("=" * 70)
    print("在线SVR增量学习测试")
    print("=" * 70)
    
    X_all, y_all = generate_time_series_data(n_samples=200, noise=0.15, drift=True)
    
    n_initial = 50
    X_initial = X_all[:n_initial]
    y_initial = y_all[:n_initial]
    
    print(f"\n1. 初始化模型 (初始样本数: {n_initial})")
    model = OnlineSVRModel(
        retrain_threshold=20,
        new_sample_weight=2.0,
        preserve_support_vectors=True
    )
    
    init_result = model.initial_train(X_initial, y_initial, C=100, epsilon=0.1)
    print(f"   初始支持向量数: {init_result['n_support_vectors']}")
    print(f"   初始训练时间: {init_result['train_duration']:.4f}s")
    
    print(f"\n2. 增量添加新样本 (每次5个, 共30个)...")
    n_incremental = 30
    batch_size = 5
    
    for i in range(n_initial, n_initial + n_incremental, batch_size):
        X_batch = X_all[i:i+batch_size]
        y_batch = y_all[i:i+batch_size]
        
        result = model.add_sample(X_batch, y_batch, auto_retrain=True)
        
        sv_count = len(model.model.support_) if model.is_trained else 'N/A'
        retrained = result['retrain_triggered']
        status = "→ 触发重训练" if retrained else "  (缓存中)"
        
        print(f"   添加样本 {i}-{i+batch_size-1}: 缓存={result['results'][0]['buffer_size']}, SV={sv_count} {status}")
    
    print(f"\n3. 手动触发重训练...")
    retrain_result = model.retrain(reason='manual_trigger')
    print(f"   新样本数: {retrain_result['n_new_samples']}")
    print(f"   总样本数: {retrain_result['n_total_samples']}")
    print(f"   支持向量数: {retrain_result['n_support_vectors']}")
    print(f"   重训练时间: {retrain_result['train_duration']:.4f}s")
    
    print(f"\n4. 带置信度预测...")
    X_test = np.array([[2.0], [5.0], [8.0]])
    predictions = model.predict_with_confidence(X_test, n_neighbors=5)
    
    for x, (pred, conf) in zip(X_test, predictions):
        print(f"   X={x[0]:.1f} → 预测={pred:.4f}, 置信度={conf:.3f}")
    
    print(f"\n5. 缓存信息...")
    buffer_info = model.get_buffer_info()
    print(f"   缓存大小: {buffer_info.get('buffer_size', 0)}")
    print(f"   已标记样本: {buffer_info.get('labeled_samples', 0)}")
    
    return model, X_all, y_all, n_initial


def compare_warm_vs_cold():
    print("\n" + "=" * 70)
    print("热启动 vs 冷启动 对比测试")
    print("=" * 70)
    
    X_all, y_all = generate_time_series_data(n_samples=150, noise=0.1)
    
    n_initial = 50
    X_initial = X_all[:n_initial]
    y_initial = y_all[:n_initial]
    
    model_warm = OnlineSVRModel(preserve_support_vectors=True)
    model_warm.initial_train(X_initial, y_initial, C=100, epsilon=0.1)
    
    for i in range(n_initial, n_initial + 30):
        model_warm.add_sample(X_all[i], y_all[i], auto_retrain=False)
    
    result_warm = model_warm.retrain(reason='warm_start', warm_start=True)
    print(f"热启动 (保留支持向量):")
    print(f"  支持向量数: {result_warm['n_support_vectors']}")
    print(f"  训练时间: {result_warm['train_duration']:.4f}s")
    print(f"  使用了支持向量: {result_warm['used_support_vectors']}")
    
    model_cold = OnlineSVRModel(preserve_support_vectors=False)
    model_cold.initial_train(X_initial, y_initial, C=100, epsilon=0.1)
    
    for i in range(n_initial, n_initial + 30):
        model_cold.add_sample(X_all[i], y_all[i], auto_retrain=False)
    
    result_cold = model_cold.retrain(reason='cold_start', warm_start=False)
    print(f"\n冷启动 (不保留支持向量):")
    print(f"  支持向量数: {result_cold['n_support_vectors']}")
    print(f"  训练时间: {result_cold['train_duration']:.4f}s")
    
    return model_warm, model_cold


def visualize_learning_progress():
    print("\n" + "=" * 70)
    print("生成学习过程可视化...")
    print("=" * 70)
    
    model, X_all, y_all, n_initial = test_online_learning()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    
    ax = axes[0, 0]
    ax.scatter(X_all[:n_initial], y_all[:n_initial], c='blue', s=30, alpha=0.6, label='初始样本')
    ax.scatter(X_all[n_initial:], y_all[n_initial:], c='orange', s=30, alpha=0.4, label='后续样本')
    
    model_init = OnlineSVRModel()
    model_init.initial_train(X_all[:n_initial], y_all[:n_initial], C=100, epsilon=0.1)
    y_init_pred = model_init.predict(X_plot)
    ax.plot(X_plot, y_init_pred, 'r-', lw=2, label='初始预测')
    ax.set_title('初始训练状态', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.scatter(X_all[:n_initial], y_all[:n_initial], c='blue', s=30, alpha=0.6, label='初始样本')
    ax.scatter(X_all[n_initial:n_initial+15], y_all[n_initial:n_initial+15], 
               c='green', s=50, marker='*', label='新增样本')
    
    y_pred_2 = model.predict(X_plot)
    ax.plot(X_plot, y_pred_2, 'purple', lw=2, label='更新后预测')
    ax.set_title('添加15个样本后', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.scatter(X_all, y_all, c='gray', s=20, alpha=0.3, label='所有样本')
    
    predictions = model.predict_with_confidence(X_plot, n_neighbors=3)
    preds = [p for p, c in predictions]
    confs = [c for p, c in predictions]
    
    scatter = ax.scatter(X_plot, preds, c=confs, cmap='viridis', s=30, alpha=0.8)
    ax.plot(X_plot, preds, 'k-', lw=1, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='置信度')
    ax.set_title('预测置信度分布', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('预测值')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.axis('off')
    
    info_text = """
在线SVR核心特性:

1. 支持向量保留机制
   ✓ 保留历史支持向量作为"知识"
   ✓ 避免完全从零开始重训练
   ✓ 加速训练过程

2. 智能重训练触发
   ✓ 基于新样本数量阈值
   ✓ 基于时间间隔
   ✓ 支持手动触发

3. 新样本加权
   ✓ 新样本权重更高(默认2x)
   ✓ 更快适应新概念
   ✓ 平衡新旧数据影响

4. 预测置信度
   ✓ 基于最近邻历史数据距离
   ✓ 量化预测可靠性
   ✓ 帮助业务决策
    """
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig('online_svr_demo.png', dpi=150, bbox_inches='tight')
    print(f"可视化图表已保存至: online_svr_demo.png")


def api_usage_example():
    print("\n" + "=" * 70)
    print("API 使用示例")
    print("=" * 70)
    
    print("""
1. 初始化在线SVR模型:
POST /api/online/init
{
  "X": [[1.0], [2.0], [3.0], ...],
  "y": [2.5, 4.5, 6.5, ...],
  "model_id": "my_online_model",
  "C": 100,
  "epsilon": 0.1,
  "retrain_threshold": 50,
  "new_sample_weight": 2.0
}

2. 增量添加样本:
POST /api/online/add
{
  "model_id": "my_online_model",
  "X": [[4.0], [5.0]],
  "y": [8.5, 10.5],
  "auto_retrain": true
}

3. 手动触发重训练:
POST /api/online/retrain
{
  "model_id": "my_online_model",
  "warm_start": true
}

4. 带置信度预测:
POST /api/online/predict
{
  "model_id": "my_online_model",
  "X": [[6.0], [7.0]],
  "with_confidence": true
}

5. 获取缓存信息:
GET /api/online/model/my_online_model/buffer
    """)


if __name__ == "__main__":
    test_online_learning()
    compare_warm_vs_cold()
    visualize_learning_progress()
    api_usage_example()
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
