import numpy as np


class LDA:
    def __init__(self, n_components=None, reg_param=None, solver='regularized'):
        self.n_components = n_components
        self.reg_param = reg_param
        self.solver = solver
        self.W = None
        self.mean_overall = None
        self.class_means = None
        self.classes = None
        self.is_singular = False
        self.condition_number = None
        self.reg_param_used = None
        self.reg_param_suggested = None
        self.shared_cov = None
        self.shared_cov_inv = None
        self.shared_cov_logdet = None
        self.class_priors = None

    def _compute_inverse(self, S_W, n_features):
        self.is_singular = False
        eigvals_sw = np.linalg.eigvalsh(S_W)
        eigvals_sw = np.sort(np.abs(eigvals_sw))[::-1]
        self.condition_number = eigvals_sw[0] / max(eigvals_sw[-1], 1e-300)

        SINGULARITY_THRESHOLD = 1e10
        self.is_singular = self.condition_number > SINGULARITY_THRESHOLD

        trace_sw = np.trace(S_W)
        self.reg_param_suggested = 1e-4 * trace_sw / n_features if trace_sw > 0 else 1e-4

        if self.solver == 'pseudo':
            S_W_inv = np.linalg.pinv(S_W)
            self.reg_param_used = None
            return S_W_inv

        if self.solver == 'regularized':
            delta = self.reg_param
            if delta is None:
                if self.is_singular:
                    delta = self.reg_param_suggested
                else:
                    delta = 0.0
            self.reg_param_used = delta
            S_W_reg = S_W + delta * np.eye(n_features)
            S_W_inv = np.linalg.inv(S_W_reg)
            return S_W_inv

        raise ValueError(f"Unknown solver: {self.solver}. Use 'regularized' or 'pseudo'.")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        if self.n_components is None:
            self.n_components = n_classes - 1
        else:
            self.n_components = min(self.n_components, n_classes - 1)

        self.mean_overall = np.mean(X, axis=0)

        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        self.class_means = {}

        for c in self.classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self.class_means[c] = mean_c

            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))

            n_c = X_c.shape[0]
            mean_diff = (mean_c - self.mean_overall).reshape(n_features, 1)
            S_B += n_c * np.dot(mean_diff, mean_diff.T)

        S_W_inv = self._compute_inverse(S_W, n_features)
        eigvals, eigvecs = np.linalg.eig(np.dot(S_W_inv, S_B))

        eigvecs = eigvecs.T
        idxs = np.argsort(np.abs(eigvals))[::-1]
        eigvecs = eigvecs[idxs]

        self.W = eigvecs[:self.n_components].T.real

        n_samples = X.shape[0]
        self.shared_cov = S_W / (n_samples - len(self.classes))
        delta = self.reg_param_used if self.reg_param_used is not None else 0.0
        shared_cov_reg = self.shared_cov + delta * np.eye(n_features)
        self.shared_cov_inv = np.linalg.inv(shared_cov_reg)
        _, self.shared_cov_logdet = np.linalg.slogdet(shared_cov_reg)

        self.class_priors = {}
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples

    def _lda_discriminant(self, x, c):
        mean_c = self.class_means[c]
        prior_c = self.class_priors[c]

        diff = x - mean_c
        quadratic = np.dot(np.dot(diff.T, self.shared_cov_inv), diff)
        return -0.5 * self.shared_cov_logdet - 0.5 * quadratic + np.log(prior_c)

    def transform(self, X):
        return np.dot(X, self.W)

    def predict(self, X):
        predictions = []
        for x in X:
            scores = {}
            for c in self.classes:
                scores[c] = self._lda_discriminant(x, c)
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)


class QDA:
    def __init__(self, reg_param=None):
        self.reg_param = reg_param
        self.classes = None
        self.class_means = None
        self.class_covs = None
        self.class_covs_inv = None
        self.class_covs_logdet = None
        self.class_priors = None
        self.reg_param_used = None

    def _regularize_cov(self, cov, n_features):
        trace_cov = np.trace(cov)
        reg_suggested = 1e-4 * trace_cov / n_features if trace_cov > 0 else 1e-4

        if self.reg_param is None:
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = np.min(eigvals)
            if min_eig < 1e-8:
                delta = reg_suggested
            else:
                delta = 0.0
        else:
            delta = self.reg_param

        cov_reg = cov + delta * np.eye(n_features)
        return cov_reg, delta

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_means = {}
        self.class_covs = {}
        self.class_covs_inv = {}
        self.class_covs_logdet = {}
        self.class_priors = {}
        max_delta = 0.0

        for c in self.classes:
            X_c = X[y == c]
            n_c = X_c.shape[0]
            mean_c = np.mean(X_c, axis=0)
            self.class_means[c] = mean_c

            cov_c = np.dot((X_c - mean_c).T, (X_c - mean_c)) / max(n_c - 1, 1)
            self.class_covs[c] = cov_c

            cov_reg, delta = self._regularize_cov(cov_c, n_features)
            max_delta = max(max_delta, delta)

            cov_inv = np.linalg.inv(cov_reg)
            _, logdet = np.linalg.slogdet(cov_reg)

            self.class_covs_inv[c] = cov_inv
            self.class_covs_logdet[c] = logdet
            self.class_priors[c] = n_c / n_samples

        self.reg_param_used = max_delta

    def _discriminant_function(self, x, c):
        mean_c = self.class_means[c]
        cov_inv_c = self.class_covs_inv[c]
        logdet_c = self.class_covs_logdet[c]
        prior_c = self.class_priors[c]

        diff = x - mean_c
        quadratic = np.dot(np.dot(diff.T, cov_inv_c), diff)
        return -0.5 * logdet_c - 0.5 * quadratic + np.log(prior_c)

    def predict(self, X):
        predictions = []
        for x in X:
            scores = {}
            for c in self.classes:
                scores[c] = self._discriminant_function(x, c)
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("=" * 60)
    print("测试1: Iris数据集 - LDA vs QDA对比")
    print("=" * 60)
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)
    acc_lda = accuracy_score(y_test, y_pred_lda)

    qda = QDA()
    qda.fit(X_train, y_train)
    y_pred_qda = qda.predict(X_test)
    acc_qda = accuracy_score(y_test, y_pred_qda)

    print(f"LDA 准确率: {acc_lda:.4f}")
    print(f"QDA 准确率: {acc_qda:.4f}")
    print(f"准确率差异 (QDA-LDA): {acc_qda - acc_lda:+.4f}")
    print(f"QDA使用正则化δ: {qda.reg_param_used}")

    print()
    print("=" * 60)
    print("测试2: 不同协方差模拟数据 - LDA vs QDA对比")
    print("=" * 60)
    np.random.seed(42)
    n_per_class = 100

    mean0 = np.array([0, 0])
    cov0 = np.array([[1, 0], [0, 1]])
    X0 = np.random.multivariate_normal(mean0, cov0, n_per_class)

    mean1 = np.array([2, 2])
    cov1 = np.array([[2, 1], [1, 2]])
    X1 = np.random.multivariate_normal(mean1, cov1, n_per_class)

    mean2 = np.array([-2, 1])
    cov2 = np.array([[0.5, 0.1], [0.1, 0.5]])
    X2 = np.random.multivariate_normal(mean2, cov2, n_per_class)

    X_sim = np.vstack([X0, X1, X2])
    y_sim = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_sim, y_sim, test_size=0.3, random_state=42
    )

    lda_sim = LDA()
    lda_sim.fit(Xs_train, ys_train)
    ys_pred_lda = lda_sim.predict(Xs_test)
    acc_lda_sim = accuracy_score(ys_test, ys_pred_lda)

    qda_sim = QDA()
    qda_sim.fit(Xs_train, ys_train)
    ys_pred_qda = qda_sim.predict(Xs_test)
    acc_qda_sim = accuracy_score(ys_test, ys_pred_qda)

    print(f"各类别协方差不同 (QDA假设成立场景)")
    print(f"LDA 准确率: {acc_lda_sim:.4f}")
    print(f"QDA 准确率: {acc_qda_sim:.4f}")
    print(f"准确率差异 (QDA-LDA): {acc_qda_sim - acc_lda_sim:+.4f}")
    print(f"QDA使用正则化δ: {qda_sim.reg_param_used}")

    print()
    print("=" * 60)
    print("测试3: 奇异矩阵场景 - LDA vs QDA")
    print("=" * 60)
    n = 30
    X_base = np.random.randn(n, 3)
    col_linear = 2 * X_base[:, 0] + 3 * X_base[:, 1]
    col_zero = np.zeros(n)
    X_singular = np.column_stack([X_base, col_linear, col_zero])
    y_sing = np.array([0] * 10 + [1] * 10 + [2] * 10)

    lda_sing = LDA()
    lda_sing.fit(X_singular, y_sing)
    y_pred_lda_sing = lda_sing.predict(X_singular)
    acc_lda_sing = accuracy_score(y_sing, y_pred_lda_sing)

    qda_sing = QDA()
    qda_sing.fit(X_singular, y_sing)
    y_pred_qda_sing = qda_sing.predict(X_singular)
    acc_qda_sing = accuracy_score(y_sing, y_pred_qda_sing)

    print(f"LDA 条件数: {lda_sing.condition_number:.2e}, 是否奇异: {lda_sing.is_singular}")
    print(f"LDA 准确率: {acc_lda_sing:.4f}, 正则化δ: {lda_sing.reg_param_used}")
    print(f"QDA 准确率: {acc_qda_sing:.4f}, 正则化δ: {qda_sing.reg_param_used}")
    print(f"准确率差异 (QDA-LDA): {acc_qda_sing - acc_lda_sing:+.4f}")

    print()
    print("=" * 60)
    print("测试4: 手写数字(sklearn) - LDA vs QDA")
    print("=" * 60)
    from sklearn.datasets import load_digits

    digits = load_digits()
    X_d, y_d = digits.data, digits.target

    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        X_d, y_d, test_size=0.3, random_state=42
    )

    lda_d = LDA()
    lda_d.fit(Xd_train, yd_train)
    yd_pred_lda = lda_d.predict(Xd_test)
    acc_lda_d = accuracy_score(yd_test, yd_pred_lda)

    qda_d = QDA()
    qda_d.fit(Xd_train, yd_train)
    yd_pred_qda = qda_d.predict(Xd_test)
    acc_qda_d = accuracy_score(yd_test, yd_pred_qda)

    print(f"LDA 条件数: {lda_d.condition_number:.2e}, 是否奇异: {lda_d.is_singular}")
    print(f"LDA 准确率: {acc_lda_d:.4f}, 正则化δ: {lda_d.reg_param_used}")
    print(f"QDA 准确率: {acc_qda_d:.4f}, 正则化δ: {qda_d.reg_param_used}")
    print(f"准确率差异 (QDA-LDA): {acc_qda_d - acc_lda_d:+.4f}")
