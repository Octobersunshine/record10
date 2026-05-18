import numpy as np
from logistic_regression import LogisticRegression


def generate_data(n_samples=1000):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    true_weights = np.array([2, -1.5])
    z = np.dot(X, true_weights) + 1
    prob = 1 / (1 + np.exp(-z))
    y = (prob >= 0.5).astype(int)
    return X, y


def train_val_split(X, y, val_ratio=0.2):
    n_samples = len(X)
    n_val = int(n_samples * val_ratio)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def main():
    X, y = generate_data(n_samples=1000)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2)

    print("=" * 60)
    print("1. 基础训练（无正则化，无早停）")
    print("=" * 60)
    model1 = LogisticRegression(
        learning_rate=0.1,
        num_iterations=5000,
        fit_intercept=True
    )
    model1.fit(X_train, y_train)
    print(f"模型权重: {model1.get_weights()}")
    train_acc1 = np.mean(model1.predict(X_train) == y_train)
    val_acc1 = np.mean(model1.predict(X_val) == y_val)
    print(f"训练集准确率: {train_acc1:.4f}")
    print(f"验证集准确率: {val_acc1:.4f}")
    print(f"最终损失: {model1.get_loss_history()[-1]:.4f}")

    print("\n" + "=" * 60)
    print("2. L2正则化训练")
    print("=" * 60)
    model2 = LogisticRegression(
        learning_rate=0.1,
        num_iterations=5000,
        fit_intercept=True,
        reg_lambda=0.1
    )
    model2.fit(X_train, y_train)
    print(f"模型权重: {model2.get_weights()}")
    train_acc2 = np.mean(model2.predict(X_train) == y_train)
    val_acc2 = np.mean(model2.predict(X_val) == y_val)
    print(f"训练集准确率: {train_acc2:.4f}")
    print(f"验证集准确率: {val_acc2:.4f}")
    print(f"最终损失（含正则项）: {model2.get_loss_history()[-1]:.4f}")

    print("\n" + "=" * 60)
    print("3. 早停机制训练")
    print("=" * 60)
    model3 = LogisticRegression(
        learning_rate=0.1,
        num_iterations=5000,
        fit_intercept=True,
        early_stopping=True,
        patience=50,
        min_delta=0.0001
    )
    model3.fit(X_train, y_train, X_val, y_val)
    print(f"模型权重: {model3.get_weights()}")
    print(f"最佳迭代轮数: {model3.get_best_iteration()}")
    train_acc3 = np.mean(model3.predict(X_train) == y_train)
    val_acc3 = np.mean(model3.predict(X_val) == y_val)
    print(f"训练集准确率: {train_acc3:.4f}")
    print(f"验证集准确率: {val_acc3:.4f}")

    print("\n" + "=" * 60)
    print("4. L2正则化 + 早停")
    print("=" * 60)
    model4 = LogisticRegression(
        learning_rate=0.1,
        num_iterations=5000,
        fit_intercept=True,
        reg_lambda=0.05,
        early_stopping=True,
        patience=50
    )
    model4.fit(X_train, y_train, X_val, y_val)
    print(f"模型权重: {model4.get_weights()}")
    print(f"最佳迭代轮数: {model4.get_best_iteration()}")
    train_acc4 = np.mean(model4.predict(X_train) == y_train)
    val_acc4 = np.mean(model4.predict(X_val) == y_val)
    print(f"训练集准确率: {train_acc4:.4f}")
    print(f"验证集准确率: {val_acc4:.4f}")

    print("\n" + "=" * 60)
    print("预测API使用示例")
    print("=" * 60)
    X_new = np.array([[1.0, -0.5], [0.0, 0.0], [-1.0, 1.0]])
    probas = model4.predict_proba(X_new)
    predictions = model4.predict(X_new)
    for i, (proba, pred) in enumerate(zip(probas, predictions)):
        print(f"样本 {i+1}: 概率={proba:.4f}, 预测类别={pred}")


if __name__ == "__main__":
    main()
