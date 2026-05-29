import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    z = np.clip(z, -700, 700)
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = np.clip(z, -700, 700)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def categorical_cross_entropy(y_true_onehot, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))


def one_hot(y, n_classes):
    n_samples = y.shape[0]
    oh = np.zeros((n_samples, n_classes))
    oh[np.arange(n_samples), y.astype(int)] = 1
    return oh


class BinaryLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 lambda_reg=0.0, patience=0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.patience = patience
        self.weights = None
        self.bias = None
        self.losses = []
        self.val_losses = []

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []
        self.val_losses = []

        has_val = X_val is not None and y_val is not None
        best_val_loss = np.inf
        best_weights = None
        best_bias = None
        wait = 0

        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)

            loss = binary_cross_entropy(y, y_pred)
            reg_loss = 0.5 * self.lambda_reg * np.sum(self.weights ** 2)
            self.losses.append(loss + reg_loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            dw += self.lambda_reg * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if has_val:
                val_pred = sigmoid(np.dot(X_val, self.weights) + self.bias)
                val_loss = binary_cross_entropy(y_val, val_pred)
                self.val_losses.append(val_loss)

                if self.patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = self.weights.copy()
                        best_bias = self.bias
                        wait = 0
                    else:
                        wait += 1
                        if wait >= self.patience:
                            print(f"  Early stopping at iteration {i}, "
                                  f"best val loss: {best_val_loss:.4f}")
                            self.weights = best_weights
                            self.bias = best_bias
                            break

            if i % 100 == 0:
                msg = f"  Iteration {i}, Train Loss: {self.losses[-1]:.4f}"
                if has_val:
                    msg += f", Val Loss: {self.val_losses[-1]:.4f}"
                print(msg)

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 lambda_reg=0.0, patience=0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.patience = patience
        self.weights = None
        self.bias = None
        self.n_classes = None
        self.losses = []
        self.val_losses = []

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)
        self.losses = []
        self.val_losses = []

        y_oh = one_hot(y, self.n_classes)
        has_val = X_val is not None and y_val is not None
        y_val_oh = one_hot(y_val, self.n_classes) if has_val else None

        best_val_loss = np.inf
        best_weights = None
        best_bias = None
        wait = 0

        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = softmax(z)

            loss = categorical_cross_entropy(y_oh, y_pred)
            reg_loss = 0.5 * self.lambda_reg * np.sum(self.weights ** 2)
            self.losses.append(loss + reg_loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_oh))
            db = (1 / n_samples) * np.sum(y_pred - y_oh, axis=0)
            dw += self.lambda_reg * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if has_val:
                val_pred = softmax(np.dot(X_val, self.weights) + self.bias)
                val_loss = categorical_cross_entropy(y_val_oh, val_pred)
                self.val_losses.append(val_loss)

                if self.patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = self.weights.copy()
                        best_bias = self.bias.copy()
                        wait = 0
                    else:
                        wait += 1
                        if wait >= self.patience:
                            print(f"  Early stopping at iteration {i}, "
                                  f"best val loss: {best_val_loss:.4f}")
                            self.weights = best_weights
                            self.bias = best_bias
                            break

            if i % 100 == 0:
                msg = f"  Iteration {i}, Train Loss: {self.losses[-1]:.4f}"
                if has_val:
                    msg += f", Val Loss: {self.val_losses[-1]:.4f}"
                print(msg)

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return softmax(z)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def plot_loss_curve(train_losses, val_losses=None, title='Loss Descent Curve',
                    save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # ===================== 二分类演示 =====================
    print("=" * 60)
    print("Binary Logistic Regression (L2 + Early Stopping)")
    print("=" * 60)

    X_bin = np.random.randn(300, 2)
    y_bin = (X_bin[:, 0] + X_bin[:, 1] > 0).astype(int)

    X_train_b = X_bin[:180]
    y_train_b = y_bin[:180]
    X_val_b = X_bin[180:240]
    y_val_b = y_bin[180:240]
    X_test_b = X_bin[240:]
    y_test_b = y_bin[240:]

    bin_model = BinaryLogisticRegression(
        learning_rate=0.1,
        n_iterations=1000,
        lambda_reg=0.01,
        patience=20
    )
    bin_model.fit(X_train_b, y_train_b, X_val_b, y_val_b)

    print(f"\n  Weights: {bin_model.weights}")
    print(f"  Bias:    {bin_model.bias:.4f}")
    test_probs_b = bin_model.predict_proba(X_test_b)
    test_preds_b = bin_model.predict(X_test_b)
    acc_b = np.mean(test_preds_b == y_test_b)
    print(f"  Test Accuracy: {acc_b:.4f}")

    print("\n  Sample predictions (first 5):")
    for i in range(5):
        print(f"    True={y_test_b[i]}, Prob={test_probs_b[i]:.4f}, "
              f"Pred={test_preds_b[i]}")

    plot_loss_curve(bin_model.losses, bin_model.val_losses,
                    title='Binary Logistic Regression - Loss Curve',
                    save_path='binary_loss_curve.png')

    # ===================== 多分类演示 =====================
    print("\n" + "=" * 60)
    print("Softmax Regression (L2 + Early Stopping)")
    print("=" * 60)

    n_per_class = 100
    X_multi = np.vstack([
        np.random.randn(n_per_class, 2) + [2, 0],
        np.random.randn(n_per_class, 2) + [0, 2],
        np.random.randn(n_per_class, 2) + [-2, 0],
    ])
    y_multi = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    perm = np.random.permutation(len(y_multi))
    X_multi = X_multi[perm]
    y_multi = y_multi[perm]

    n_total = len(y_multi)
    X_train_m = X_multi[:int(n_total * 0.6)]
    y_train_m = y_multi[:int(n_total * 0.6)]
    X_val_m = X_multi[int(n_total * 0.6):int(n_total * 0.8)]
    y_val_m = y_multi[int(n_total * 0.6):int(n_total * 0.8)]
    X_test_m = X_multi[int(n_total * 0.8):]
    y_test_m = y_multi[int(n_total * 0.8):]

    softmax_model = SoftmaxRegression(
        learning_rate=0.1,
        n_iterations=1000,
        lambda_reg=0.01,
        patience=20
    )
    softmax_model.fit(X_train_m, y_train_m, X_val_m, y_val_m)

    print(f"\n  Weights shape: {softmax_model.weights.shape}")
    print(f"  Bias: {softmax_model.bias}")
    test_probs_m = softmax_model.predict_proba(X_test_m)
    test_preds_m = softmax_model.predict(X_test_m)
    acc_m = np.mean(test_preds_m == y_test_m)
    print(f"  Test Accuracy: {acc_m:.4f}")

    print("\n  Sample predictions (first 5):")
    for i in range(5):
        print(f"    True={y_test_m[i]}, Probs={test_probs_m[i]}, "
              f"Pred={test_preds_m[i]}")

    plot_loss_curve(softmax_model.losses, softmax_model.val_losses,
                    title='Softmax Regression - Loss Curve',
                    save_path='softmax_loss_curve.png')
