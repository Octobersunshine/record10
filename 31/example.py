import numpy as np
from sklearn.preprocessing import StandardScaler
from pca_regression import (
    pca_linear_regression,
    pls_regression,
    auto_select_best_model,
    get_pca_components,
    get_explained_variance_ratio,
    verify_standardization_consistency
)


np.random.seed(42)
n_samples = 100
n_features = 10

X = np.random.randn(n_samples, n_features) * 10 + 5
true_coef = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
y = X @ true_coef + np.random.randn(n_samples) * 0.5

split_idx = int(n_samples * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("=" * 60)
print("1. PCR (PCA + 线性回归) 模型")
print("=" * 60)
pcr_model = pca_linear_regression(X_train, y_train, n_components=5)
print("训练集R²得分:", pcr_model.score(X_train, y_train))
print("测试集R²得分:", pcr_model.score(X_test, y_test))
print("解释方差比例:", get_explained_variance_ratio(pcr_model))

print("\n" + "=" * 60)
print("2. PLS (偏最小二乘回归) 模型")
print("=" * 60)
pls_model = pls_regression(X_train, y_train, n_components=5)
print("训练集R²得分:", pls_model.score(X_train, y_train))
print("测试集R²得分:", pls_model.score(X_test, y_test))
print("PLS回归系数前5个特征:", pls_model.coef_[:5].flatten())

print("\n" + "=" * 60)
print("3. 自动选择最佳模型")
print("=" * 60)
result = auto_select_best_model(X, y, n_components=5, val_size=0.2, metric='mse')
print(f"最佳方法: {result['best_method']}")
print(f"PCR验证集{result['metric']}: {result['pcr_score']:.4f}")
print(f"PLS验证集{result['metric']}: {result['pls_score']:.4f}")
print(f"PCR训练集{result['metric']}: {result['pcr_train_score']:.4f}")
print(f"PLS训练集{result['metric']}: {result['pls_train_score']:.4f}")

best_model = result['best_model']
print(f"\n最佳模型测试集R²得分: {best_model.score(X_test, y_test):.4f}")

print("\n" + "=" * 60)
print("4. 预测示例 (前5个测试样本)")
print("=" * 60)
y_pred_pcr = pcr_model.predict(X_test)
y_pred_pls = pls_model.predict(X_test)
y_pred_best = best_model.predict(X_test)
print("真实值:", y_test[:5])
print("PCR预测:", y_pred_pcr[:5])
print("PLS预测:", y_pred_pls[:5])
print("最佳预测:", y_pred_best[:5])

print("\n" + "=" * 60)
print("5. 验证标准化一致性")
print("=" * 60)
result_std = verify_standardization_consistency(pcr_model, X_train, X_test)
print("PCR训练集标准化一致:", result_std['train_consistent'])
print("PCR测试集标准化一致:", result_std['test_consistent'])
print("使用的均值:", result_std['mean_used'][:3], "...")

print("\n" + "=" * 60)
print("6. 手动验证预测流程 - 错误vs正确")
print("=" * 60)
print("错误方式: 直接对测试集标准化 (使用测试集自身统计量)")
wrong_scaler = StandardScaler()
X_test_wrong_scaled = wrong_scaler.fit_transform(X_test)
X_test_wrong_pca = pcr_model.pca.transform(X_test_wrong_scaled)
y_pred_wrong = pcr_model.regressor.predict(X_test_wrong_pca)
print("错误预测结果:", y_pred_wrong[:5])
print("正确预测结果:", y_pred_pcr[:5])
print("差异:", np.abs(y_pred_wrong[:5] - y_pred_pcr[:5]))
