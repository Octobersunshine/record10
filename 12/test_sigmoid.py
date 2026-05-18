import numpy as np
from logistic_regression import LogisticRegression


def test_sigmoid_stability():
    model = LogisticRegression()

    test_values = np.array([-1000, -100, -50, -10, 0, 10, 50, 100, 1000])
    print("测试sigmoid函数数值稳定性:")
    print("-" * 50)

    for z in test_values:
        result = model._sigmoid(np.array([z]))[0]
        print(f"z = {z:6d} -> sigmoid(z) = {result:.10f}")

    print("\n是否有NaN?", np.isnan(model._sigmoid(test_values)).any())
    print("是否有inf?", np.isinf(model._sigmoid(test_values)).any())


def test_training_with_extreme_values():
    print("\n" + "=" * 50)
    print("测试极端值下的训练过程:")
    print("=" * 50)

    np.random.seed(42)
    X = np.random.randn(100, 2) * 10
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(learning_rate=0.001, num_iterations=100)
    weights = model.fit(X, y)

    print(f"训练完成!")
    print(f"最终权重: {weights}")
    print(f"损失历史中是否有NaN?", np.isnan(model.get_loss_history()).any())
    print(f"损失历史中是否有inf?", np.isinf(model.get_loss_history()).any())

    if len(model.get_loss_history()) > 0:
        print(f"第一个损失值: {model.get_loss_history()[0]:.6f}")
        print(f"最后一个损失值: {model.get_loss_history()[-1]:.6f}")


if __name__ == "__main__":
    test_sigmoid_stability()
    test_training_with_extreme_values()
