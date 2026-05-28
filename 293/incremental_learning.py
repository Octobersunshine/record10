import numpy as np
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.t = 0

    @abstractmethod
    def update(self, weights, grad, indices=None):
        pass


class SGDOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = None

    def update(self, weights, grad, indices=None):
        self.t += 1
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)

        if indices is None:
            if self.nesterov:
                v_prev = self.velocity.copy()
                self.velocity = self.momentum * self.velocity - self.learning_rate * grad
                weights += -self.momentum * v_prev + (1 + self.momentum) * self.velocity
            else:
                self.velocity = self.momentum * self.velocity - self.learning_rate * grad
                weights += self.velocity
        else:
            if self.nesterov:
                v_prev = self.velocity[indices].copy()
                self.velocity[indices] = self.momentum * self.velocity[indices] - self.learning_rate * grad
                weights[indices] += -self.momentum * v_prev + (1 + self.momentum) * self.velocity[indices]
            else:
                self.velocity[indices] = self.momentum * self.velocity[indices] - self.learning_rate * grad
                weights[indices] += self.velocity[indices]


class AdagradOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.G = None

    def update(self, weights, grad, indices=None):
        self.t += 1
        if self.G is None:
            self.G = np.zeros_like(weights)

        if indices is None:
            self.G += grad ** 2
            weights -= self.learning_rate * grad / (np.sqrt(self.G) + self.epsilon)
        else:
            self.G[indices] += grad ** 2
            weights[indices] -= self.learning_rate * grad / (np.sqrt(self.G[indices]) + self.epsilon)


class AdamOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, weights, grad, indices=None):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        lr_t = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        if indices is None:
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
            weights -= lr_t * self.m / (np.sqrt(self.v) + self.epsilon)
        else:
            self.m[indices] = self.beta1 * self.m[indices] + (1 - self.beta1) * grad
            self.v[indices] = self.beta2 * self.v[indices] + (1 - self.beta2) * (grad ** 2)
            weights[indices] -= lr_t * self.m[indices] / (np.sqrt(self.v[indices]) + self.epsilon)


class BaseLinearModel(ABC):
    def __init__(self, n_features, learning_rate=0.01, standardize=True,
                 optimizer='adam', l1_reg=0.0, l2_reg=0.0, sparse=False, random_state=None):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.standardize = standardize
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.sparse = sparse
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.weights = np.random.randn(n_features + 1) * 0.01
        self.n_samples_seen = 0
        self._running_mean = np.zeros(n_features)
        self._running_m2 = np.zeros(n_features)
        self._stats_count = 0

        self._init_optimizer(optimizer)

    def _init_optimizer(self, optimizer):
        if isinstance(optimizer, str):
            opt_name = optimizer.lower()
            if opt_name == 'sgd':
                self.optimizer = SGDOptimizer(learning_rate=self.learning_rate)
            elif opt_name == 'sgd_momentum':
                self.optimizer = SGDOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            elif opt_name == 'adagrad':
                self.optimizer = AdagradOptimizer(learning_rate=self.learning_rate)
            elif opt_name == 'adam':
                self.optimizer = AdamOptimizer(learning_rate=self.learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            self.optimizer = optimizer

    def _add_bias(self, X):
        if X.ndim == 1:
            return np.insert(X, 0, 1.0)
        return np.insert(X, 0, 1.0, axis=1)

    def _update_stats(self, x):
        self._stats_count += 1
        delta = x - self._running_mean
        self._running_mean += delta / self._stats_count
        delta2 = x - self._running_mean
        self._running_m2 += delta * delta2

    def _normalize(self, X):
        if not self.standardize or self._stats_count < 2:
            return X
        std = np.sqrt(self._running_m2 / (self._stats_count - 1))
        std = np.where(std < 1e-8, 1.0, std)
        if X.ndim == 1:
            return (X - self._running_mean) / std
        return (X - self._running_mean) / std

    @abstractmethod
    def _activation(self, z):
        pass

    @abstractmethod
    def _loss_gradient(self, y_pred, y_true):
        pass

    def _apply_regularization(self, X_bias, indices=None):
        if self.l2_reg > 0 or self.l1_reg > 0:
            if indices is None:
                reg_indices = np.arange(1, len(self.weights))
            else:
                reg_indices = indices[indices > 0]

            if self.l2_reg > 0 and len(reg_indices) > 0:
                self.weights[reg_indices] -= self.learning_rate * self.l2_reg * self.weights[reg_indices]

            if self.l1_reg > 0 and len(reg_indices) > 0:
                sign_w = np.sign(self.weights[reg_indices])
                self.weights[reg_indices] -= self.learning_rate * self.l1_reg * sign_w
                self.weights[reg_indices] = np.where(
                    np.abs(self.weights[reg_indices]) < self.learning_rate * self.l1_reg,
                    0,
                    self.weights[reg_indices]
                )

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)
            y = y.reshape(-1)

        batch_size = X.shape[0]

        for i in range(batch_size):
            self._update_stats(X[i])
            X_norm = self._normalize(X[i])
            X_bias = self._add_bias(X_norm)

            if self.sparse:
                non_zero_mask = np.abs(X_bias) > 1e-10
                indices = np.where(non_zero_mask)[0]
                X_sparse = X_bias[non_zero_mask]

                z = np.dot(X_sparse, self.weights[indices])
                y_pred = self._activation(z)
                grad_loss = self._loss_gradient(y_pred, y[i])
                grad = grad_loss * X_sparse

                self.optimizer.update(self.weights, grad, indices)
                self._apply_regularization(X_bias, indices)
            else:
                z = np.dot(X_bias, self.weights)
                y_pred = self._activation(z)
                grad_loss = self._loss_gradient(y_pred, y[i])
                grad = grad_loss * X_bias

                self.optimizer.update(self.weights, grad)
                self._apply_regularization(X_bias)

            self.n_samples_seen += 1

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_norm = self._normalize(X)
        X_bias = self._add_bias(X_norm)

        if self.sparse and X_bias.ndim == 2:
            predictions = np.zeros(X_bias.shape[0])
            for i in range(X_bias.shape[0]):
                non_zero_mask = np.abs(X_bias[i]) > 1e-10
                indices = np.where(non_zero_mask)[0]
                z = np.dot(X_bias[i, indices], self.weights[indices])
                predictions[i] = self._activation(z)
            return predictions
        else:
            z = np.dot(X_bias, self.weights)
            return self._activation(z)


class SGDLinearRegression(BaseLinearModel):
    def _activation(self, z):
        return z

    def _loss_gradient(self, y_pred, y_true):
        return y_pred - y_true

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class SGDLogisticRegression(BaseLinearModel):
    def _activation(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _loss_gradient(self, y_pred, y_true):
        return y_pred - y_true

    def predict_proba(self, X):
        return self.predict(X)

    def predict(self, X):
        proba = super().predict(X)
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def demo_optimizer_comparison():
    print("=" * 60)
    print("优化器对比演示（不同尺度特征）")
    print("=" * 60)

    np.random.seed(42)
    n_features = 5
    scales = np.array([0.001, 0.1, 10.0, 100.0, 1000.0])
    true_weights = np.array([2.0, -1.5, 0.8, -0.5, 1.2, 0.3])

    print(f"特征尺度: {scales}")
    print(f"真实权重: {true_weights}")
    print()

    n_total_samples = 1500
    X_test = np.random.randn(100, n_features) * scales
    y_test = true_weights[0] + np.dot(X_test, true_weights[1:]) + np.random.randn(100) * 0.5

    optimizers = {
        'SGD (lr=0.0001)': {'optimizer': 'sgd', 'lr': 0.0001, 'standardize': False},
        'SGD +标准化 (lr=0.01)': {'optimizer': 'sgd', 'lr': 0.01, 'standardize': True},
        'Adagrad (lr=0.1)': {'optimizer': 'adagrad', 'lr': 0.1, 'standardize': False},
        'Adam (lr=0.01)': {'optimizer': 'adam', 'lr': 0.01, 'standardize': False},
    }

    models = {}
    histories = {name: [] for name in optimizers}
    checkpoints = [50, 100, 200, 500, 1000, 1500]

    for name, config in optimizers.items():
        models[name] = SGDLinearRegression(
            n_features=n_features,
            learning_rate=config['lr'],
            standardize=config['standardize'],
            optimizer=config['optimizer'],
            random_state=42
        )

    for i in range(n_total_samples):
        X = np.random.randn(n_features) * scales
        y = true_weights[0] + np.dot(X, true_weights[1:]) + np.random.randn() * 0.5

        for name, model in models.items():
            model.partial_fit(X, y)

        if (i + 1) in checkpoints:
            print(f"\n--- 样本 {i+1} ---")
            for name, model in models.items():
                r2 = model.score(X_test, y_test)
                histories[name].append(r2)
                print(f"  {name:25s} | R² = {r2:.4f}")

    print()
    print("最终结果对比:")
    for name in optimizers:
        print(f"  {name:25s} | R² = {histories[name][-1]:.4f}")
    print()


def demo_regularization():
    print("=" * 60)
    print("L1/L2 正则化演示（OGD风格）")
    print("=" * 60)

    np.random.seed(42)
    n_features = 10
    true_weights = np.zeros(n_features + 1)
    true_weights[0] = 1.0
    true_weights[[2, 5, 8]] = [2.0, -1.5, 0.8]

    print(f"真实权重 (稀疏): {true_weights}")
    print()

    n_total_samples = 2000
    X_test = np.random.randn(200, n_features)
    y_test = true_weights[0] + np.dot(X_test, true_weights[1:]) + np.random.randn(200) * 0.3

    configs = {
        '无正则化': {'l1': 0.0, 'l2': 0.0},
        'L2 (0.01)': {'l1': 0.0, 'l2': 0.01},
        'L1 (0.01)': {'l1': 0.01, 'l2': 0.0},
        'L1+L2 (0.01, 0.005)': {'l1': 0.01, 'l2': 0.005},
    }

    models = {}
    for name, cfg in configs.items():
        models[name] = SGDLinearRegression(
            n_features=n_features,
            learning_rate=0.01,
            standardize=True,
            optimizer='adam',
            l1_reg=cfg['l1'],
            l2_reg=cfg['l2'],
            random_state=42
        )

    for i in range(n_total_samples):
        X = np.random.randn(n_features)
        y = true_weights[0] + np.dot(X, true_weights[1:]) + np.random.randn() * 0.3
        for model in models.values():
            model.partial_fit(X, y)

    print("结果对比:")
    for name, model in models.items():
        r2 = model.score(X_test, y_test)
        n_zeros = np.sum(np.abs(model.weights[1:]) < 1e-4)
        print(f"  {name:20s} | R² = {r2:.4f} | 稀疏权重数: {n_zeros}/{n_features}")
        print(f"    学习权重: {np.round(model.weights, 3)}")
    print()


def demo_sparse_features():
    print("=" * 60)
    print("稀疏特征更新演示")
    print("=" * 60)

    np.random.seed(42)
    n_features = 100
    sparsity = 0.95

    true_weights = np.zeros(n_features + 1)
    true_weights[0] = 0.5
    important_indices = np.random.choice(n_features, 10, replace=False) + 1
    true_weights[important_indices] = np.random.randn(10)

    print(f"特征数: {n_features}, 稀疏度: {sparsity*100}%")
    print(f"非零真实权重数: {np.sum(true_weights != 0)}")
    print()

    n_total_samples = 3000
    X_test = np.random.randn(100, n_features)
    X_test = np.where(np.random.rand(100, n_features) < sparsity, 0, X_test)
    y_test = true_weights[0] + np.dot(X_test, true_weights[1:]) + np.random.randn(100) * 0.2

    model_sparse = SGDLinearRegression(
        n_features=n_features,
        learning_rate=0.01,
        standardize=False,
        optimizer='adam',
        l1_reg=0.0001,
        sparse=True,
        random_state=42
    )

    model_dense = SGDLinearRegression(
        n_features=n_features,
        learning_rate=0.01,
        standardize=False,
        optimizer='adam',
        l1_reg=0.0001,
        sparse=False,
        random_state=42
    )

    import time
    sparse_times = []
    dense_times = []

    for i in range(n_total_samples):
        X = np.random.randn(n_features)
        X = np.where(np.random.rand(n_features) < sparsity, 0, X)
        y = true_weights[0] + np.dot(X, true_weights[1:]) + np.random.randn() * 0.2

        t0 = time.time()
        model_sparse.partial_fit(X, y)
        sparse_times.append(time.time() - t0)

        t0 = time.time()
        model_dense.partial_fit(X, y)
        dense_times.append(time.time() - t0)

    r2_sparse = model_sparse.score(X_test, y_test)
    r2_dense = model_dense.score(X_test, y_test)

    print("性能对比:")
    print(f"  稀疏模式 R²: {r2_sparse:.4f}")
    print(f"  稠密模式 R²: {r2_dense:.4f}")
    print(f"  稀疏平均时间: {np.mean(sparse_times)*1000:.3f}ms")
    print(f"  稠密平均时间: {np.mean(dense_times)*1000:.3f}ms")
    print(f"  加速比: {np.mean(dense_times)/np.mean(sparse_times):.2f}x")
    print()

    print("前20个权重对比:")
    print(f"  真实:   {np.round(true_weights[:20], 3)}")
    print(f"  稀疏:   {np.round(model_sparse.weights[:20], 3)}")
    print(f"  稠密:   {np.round(model_dense.weights[:20], 3)}")
    print()


def demo_linear_regression():
    print("=" * 60)
    print("线性回归增量学习演示")
    print("=" * 60)

    np.random.seed(42)
    n_features = 3
    true_weights = np.array([2.5, -1.0, 3.0, 0.5])

    model = SGDLinearRegression(n_features=n_features, learning_rate=0.01, standardize=True,
                                optimizer='adam', random_state=42)

    n_total_samples = 1000
    for i in range(n_total_samples):
        X = np.random.randn(n_features)
        y = true_weights[0] + np.dot(X, true_weights[1:]) + np.random.randn() * 0.1
        model.partial_fit(X, y)

        if (i + 1) % 200 == 0:
            X_test = np.random.randn(50, n_features)
            y_test = true_weights[0] + np.dot(X_test, true_weights[1:]) + np.random.randn(50) * 0.1
            r2 = model.score(X_test, y_test)
            print(f"已处理 {i+1:4d} 个样本 | R² = {r2:.4f}")

    print(f"\n真实权重: {true_weights}")
    print(f"学习权重: {np.round(model.weights, 4)}")
    print()


def demo_logistic_regression():
    print("=" * 60)
    print("逻辑回归增量学习演示")
    print("=" * 60)

    np.random.seed(42)
    n_features = 2
    true_weights = np.array([-1.0, 2.0, -1.5])

    model = SGDLogisticRegression(n_features=n_features, learning_rate=0.05, standardize=True,
                                  optimizer='adam', random_state=42)

    n_total_samples = 1000
    for i in range(n_total_samples):
        X = np.random.randn(n_features)
        z = true_weights[0] + np.dot(X, true_weights[1:])
        prob = 1 / (1 + np.exp(-z))
        y = (np.random.random() < prob).astype(int)
        model.partial_fit(X, y)

        if (i + 1) % 200 == 0:
            X_test = np.random.randn(50, n_features)
            z_test = true_weights[0] + np.dot(X_test, true_weights[1:])
            y_test = (1 / (1 + np.exp(-z_test)) > 0.5).astype(int)
            acc = model.score(X_test, y_test)
            print(f"已处理 {i+1:4d} 个样本 | 准确率 = {acc:.4f}")

    print(f"\n真实权重: {true_weights}")
    print(f"学习权重: {np.round(model.weights, 4)}")
    print()


if __name__ == "__main__":
    demo_optimizer_comparison()
    demo_regularization()
    demo_sparse_features()
    demo_linear_regression()
    demo_logistic_regression()
