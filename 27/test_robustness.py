import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from svr_model import SVRModel
from robust_svr import RobustSVRModel


def generate_data_with_outliers(n_samples=100, n_outliers=5, outlier_magnitude=10, noise_level=0.3):
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
    y = np.sin(X).ravel() + noise_level * np.random.randn(n_samples)
    
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    for idx in outlier_indices:
        y[idx] += outlier_magnitude * (1 if np.random.rand() > 0.5 else -1)
    
    return X, y, outlier_indices


def compare_models():
    print("=" * 70)
    print("SVR 鲁棒性对比测试")
    print("=" * 70)
    
    X, y, true_outliers = generate_data_with_outliers(n_samples=100, n_outliers=8, outlier_magnitude=8)
    
    print(f"\n数据集: {len(X)} 个样本, 其中 {len(true_outliers)} 个异常值")
    print(f"真实异常值索引: {sorted(true_outliers)}")
    print()
    
    results = []
    
    print("-" * 70)
    print("模型 1: 原始 SVR (无异常值处理)")
    print("-" * 70)
    model_original = SVRModel()
    res_original = model_original.train(X, y, kernel='rbf', C=100, epsilon=0.1)
    print(f"支持向量数: {res_original['support_vectors']}")
    print(f"R² 分数: {res_original['r2']:.4f}")
    print(f"RMSE: {res_original['rmse']:.4f}")
    results.append(("原始SVR", res_original, model_original))
    print()
    
    print("-" * 70)
    print("模型 2: 鲁棒 SVR (仅移除异常值)")
    print("-" * 70)
    model_robust1 = RobustSVRModel()
    res_robust1 = model_robust1.train(X, y, kernel='rbf', C=100, epsilon=0.1, 
                                      remove_outliers=True, use_robust_scaler=False)
    outlier_info = model_robust1.get_outlier_info()
    print(f"检测到异常值: {outlier_info['outlier_count']}")
    print(f"异常值索引: {outlier_info['outlier_indices']}")
    print(f"支持向量数: {res_robust1['support_vectors']}")
    print(f"R² 分数: {res_robust1['r2']:.4f}")
    print(f"RMSE: {res_robust1['rmse']:.4f}")
    results.append(("鲁棒SVR(移除异常值)", res_robust1, model_robust1))
    print()
    
    print("-" * 70)
    print("模型 3: 鲁棒 SVR (移除异常值 + 鲁棒归一化)")
    print("-" * 70)
    model_robust2 = RobustSVRModel()
    res_robust2 = model_robust2.train(X, y, kernel='rbf', C=100, epsilon=0.1,
                                      remove_outliers=True, use_robust_scaler=True)
    outlier_info = model_robust2.get_outlier_info()
    print(f"检测到异常值: {outlier_info['outlier_count']}")
    print(f"支持向量数: {res_robust2['support_vectors']}")
    print(f"R² 分数: {res_robust2['r2']:.4f}")
    print(f"RMSE: {res_robust2['rmse']:.4f}")
    results.append(("鲁棒SVR(移除+鲁棒归一化)", res_robust2, model_robust2))
    print()
    
    print("-" * 70)
    print("模型 4: Nu-SVR (控制支持向量比例)")
    print("-" * 70)
    model_nusvr = RobustSVRModel()
    res_nusvr = model_nusvr.train(X, y, kernel='rbf', C=100, nu=0.3,
                                  use_nusvr=True, remove_outliers=True, use_robust_scaler=True)
    outlier_info = model_nusvr.get_outlier_info()
    print(f"检测到异常值: {outlier_info['outlier_count']}")
    print(f"支持向量数: {res_nusvr['support_vectors']}")
    print(f"支持向量比例: {res_nusvr['sv_ratio']:.2%}")
    print(f"R² 分数: {res_nusvr['r2']:.4f}")
    print(f"RMSE: {res_nusvr['rmse']:.4f}")
    results.append(("Nu-SVR", res_nusvr, model_nusvr))
    print()
    
    print("-" * 70)
    print("模型 5: 鲁棒 SVR (限制支持向量最大比例)")
    print("-" * 70)
    model_limited = RobustSVRModel()
    res_limited = model_limited.train(X, y, kernel='rbf', C=100, epsilon=0.1,
                                      remove_outliers=True, use_robust_scaler=True, max_sv_ratio=0.3)
    print(f"支持向量数: {res_limited['support_vectors']}")
    print(f"支持向量比例: {res_limited['sv_ratio']:.2%}")
    print(f"R² 分数: {res_limited['r2']:.4f}")
    print(f"RMSE: {res_limited['rmse']:.4f}")
    results.append(("鲁棒SVR(限制SV比例)", res_limited, model_limited))
    print()
    
    print("=" * 70)
    print("总结对比")
    print("=" * 70)
    print(f"{'模型名称':<30} {'支持向量数':>12} {'SV比例':>10} {'R²':>10} {'RMSE':>10}")
    print("-" * 70)
    for name, res, _ in results:
        sv_ratio = res.get('sv_ratio', res['support_vectors'] / len(y))
        print(f"{name:<30} {res['support_vectors']:>12} {sv_ratio:>10.2%} {res['r2']:>10.4f} {res['rmse']:>10.4f}")
    print()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    
    for idx, (name, res, model) in enumerate(results):
        ax = axes[idx]
        
        y_pred = model.predict(X_plot)
        
        inliers_mask = np.ones(len(y), dtype=bool)
        if hasattr(model, 'outlier_mask') and model.outlier_mask is not None:
            inliers_mask = model.outlier_mask
        
        ax.scatter(X[inliers_mask], y[inliers_mask], c='blue', s=50, label='正常值', alpha=0.6)
        ax.scatter(X[~inliers_mask], y[~inliers_mask], c='red', s=100, marker='x', label='异常值', linewidths=2)
        ax.scatter(X[true_outliers], y[true_outliers], c='orange', s=150, marker='o', facecolors='none', label='真实异常值', linewidths=2)
        ax.plot(X_plot, y_pred, c='green', lw=2, label='预测曲线')
        
        sv_count = res['support_vectors']
        ax.set_title(f'{name}\nSV: {sv_count}, R²: {res["r2"]:.3f}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[5].axis('off')
    axes[5].text(0.1, 0.9, '关键改进点:', fontsize=12, fontweight='bold')
    axes[5].text(0.1, 0.8, '1. 异常值检测 (Z-score, IQR, Cook距离)', fontsize=10)
    axes[5].text(0.1, 0.7, '2. 鲁棒归一化 (RobustScaler)', fontsize=10)
    axes[5].text(0.1, 0.6, '3. Nu-SVR 控制支持向量比例', fontsize=10)
    axes[5].text(0.1, 0.5, '4. 支持向量数量限制', fontsize=10)
    axes[5].text(0.1, 0.4, '5. 自动参数调优', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('robustness_comparison.png', dpi=150, bbox_inches='tight')
    print(f"对比图已保存至: robustness_comparison.png")
    print()
    
    print("=" * 70)
    print("API 使用示例")
    print("=" * 70)
    print("\n启用鲁棒模式训练:")
    print("""
POST /api/train
{
  "X": [[1.0], [2.0], [3.0], ...],
  "y": [2.5, 4.5, 6.5, ...],
  "model_id": "robust_model",
  "use_robust": true,
  "use_nusvr": true,
  "nu": 0.3,
  "remove_outliers": true,
  "outlier_method": "combined",
  "use_robust_scaler": true,
  "max_sv_ratio": 0.5
}
    """)
    print("查询异常值信息:")
    print("GET /api/model/robust_model/outliers")
    print()


def single_outlier_test():
    print("\n" + "=" * 70)
    print("单个离群点影响测试")
    print("=" * 70)
    
    np.random.seed(123)
    X = np.sort(5 * np.random.rand(50, 1), axis=0)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(50)
    
    print("\n无异常值时:")
    model_clean = SVRModel()
    res_clean = model_clean.train(X, y, kernel='rbf', C=100, epsilon=0.1)
    print(f"  支持向量数: {res_clean['support_vectors']} ({res_clean['support_vectors']/len(y):.1%})")
    
    print("\n添加单个极端异常值后:")
    y_with_outlier = y.copy()
    y_with_outlier[25] = 100
    model_outlier = SVRModel()
    res_outlier = model_outlier.train(X, y_with_outlier, kernel='rbf', C=100, epsilon=0.1)
    print(f"  支持向量数: {res_outlier['support_vectors']} ({res_outlier['support_vectors']/len(y):.1%})")
    print(f"  支持向量增加: {res_outlier['support_vectors'] - res_clean['support_vectors']}")
    
    print("\n使用鲁棒SVR处理后:")
    model_robust = RobustSVRModel()
    res_robust = model_robust.train(X, y_with_outlier, kernel='rbf', C=100, epsilon=0.1, remove_outliers=True)
    outlier_info = model_robust.get_outlier_info()
    print(f"  检测到异常值: {outlier_info['outlier_count']} 个")
    print(f"  支持向量数: {res_robust['support_vectors']} ({res_robust['sv_ratio']:.1%})")
    print(f"  相比原始SVR减少: {res_outlier['support_vectors'] - res_robust['support_vectors']}")
    print()


if __name__ == "__main__":
    compare_models()
    single_outlier_test()
    print("=" * 70)
    print("测试完成!")
    print("=" * 70)
