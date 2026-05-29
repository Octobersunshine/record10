import numpy as np
from cvxopt import matrix, solvers


solvers.options['show_progress'] = False


def linear_kernel(X1, X2):
    return X1 @ X2.T


def poly_kernel(X1, X2, degree=3, coef0=1.0, gamma=1.0):
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def rbf_kernel(X1, X2, gamma=1.0):
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
    dist = X1_sq + X2_sq - 2 * X1 @ X2.T
    return np.exp(-gamma * dist)


def generate_data(n_per_class=100, separable=True):
    np.random.seed(42)
    if separable:
        X_pos = np.random.randn(n_per_class, 2) + np.array([2, 2])
        X_neg = np.random.randn(n_per_class, 2) + np.array([-2, -2])
    else:
        X_pos = np.random.randn(n_per_class, 2) + np.array([1, 1])
        X_neg = np.random.randn(n_per_class, 2) + np.array([-1, -1])

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def make_moons(n_samples=200, noise=0.1):
    np.random.seed(42)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    outer_x = np.cos(np.linspace(0, np.pi, n_out))
    outer_y = np.sin(np.linspace(0, np.pi, n_out))

    inner_x = 1 - np.cos(np.linspace(0, np.pi, n_in))
    inner_y = 1 - np.sin(np.linspace(0, np.pi, n_in)) - 0.5

    X = np.vstack([np.column_stack([outer_x, outer_y]),
                   np.column_stack([inner_x, inner_y])])
    y = np.hstack([np.ones(n_out), -np.ones(n_in)])

    X += np.random.randn(n_samples, 2) * noise

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def make_xor(n_samples=200, noise=0.1):
    np.random.seed(42)
    n_per_quad = n_samples // 4

    X = []
    y = []

    centers = [
        (np.array([1, 1]), 1),
        (np.array([-1, -1]), 1),
        (np.array([1, -1]), -1),
        (np.array([-1, 1]), -1),
    ]

    for center, label in centers:
        X_quad = np.random.randn(n_per_quad, 2) * 0.5 + center
        X.append(X_quad)
        y.extend([label] * n_per_quad)

    X = np.vstack(X)
    y = np.array(y, dtype=float)
    X += np.random.randn(n_samples, 2) * noise

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


class SVM:
    def __init__(self, solver='smo', kernel='rbf', C=1.0, degree=3, coef0=1.0,
                 gamma=1.0, lr=0.001, epochs=1000, tol=1e-6, max_passes=10):
        self.solver = solver
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.max_passes = max_passes

        self.w = None
        self.b = None
        self.support_vectors_ = None
        self.support_alphas_ = None
        self.support_labels_ = None
        self.alphas_ = None
        self.errors_ = None
        self.X_train_ = None
        self.y_train_ = None

    def _compute_kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        if self.kernel == 'linear':
            return linear_kernel(X1, X2)
        elif self.kernel == 'poly':
            return poly_kernel(X1, X2, degree=self.degree, coef0=self.coef0, gamma=self.gamma)
        elif self.kernel == 'rbf':
            return rbf_kernel(X1, X2, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_kernel_single(self, X1, x2):
        if self.kernel == 'linear':
            return X1 @ x2
        elif self.kernel == 'poly':
            return (self.gamma * (X1 @ x2) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            dist = np.sum((X1 - x2) ** 2, axis=1)
            return np.exp(-self.gamma * dist)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _solve_qp(self, X, y):
        n, d = X.shape
        y = y.astype(float)

        K = self._compute_kernel_matrix(X)
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n))

        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = matrix(np.hstack([np.zeros(n), self.C * np.ones(n)]))

        A = matrix(y.reshape(1, -1))
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x']).flatten()
        self.alphas_ = alphas

        sv_mask = alphas > self.tol
        self.support_alphas_ = alphas[sv_mask]
        self.support_vectors_ = X[sv_mask]
        self.support_labels_ = y[sv_mask]

        if self.kernel == 'linear':
            self.w = np.sum(
                (self.support_alphas_ * self.support_labels_)[:, None] * self.support_vectors_,
                axis=0
            )
        else:
            self.w = None

        margin_mask = (alphas > self.tol) & (alphas < self.C - self.tol)
        if np.any(margin_mask):
            b_values = []
            margin_idx = np.where(margin_mask)[0]
            self.b = 0.0
            for idx in margin_idx:
                if self.kernel == 'linear':
                    b_i = y[idx] - self.w @ X[idx]
                else:
                    pred = np.sum(
                        self.support_alphas_ * self.support_labels_ *
                        self._compute_kernel_single(self.support_vectors_, X[idx])
                    )
                    b_i = y[idx] - pred
                b_values.append(b_i)
            self.b = np.mean(b_values)
        else:
            sv_idx = np.where(sv_mask)[0]
            if self.kernel == 'linear':
                b_values = y[sv_idx] - X[sv_idx] @ self.w
                self.b = np.mean(b_values)
            else:
                self.b = 0.0
                preds = np.zeros(len(sv_idx))
                for i, idx in enumerate(sv_idx):
                    preds[i] = np.sum(
                        self.support_alphas_ * self.support_labels_ *
                        self._compute_kernel_single(self.support_vectors_, X[idx])
                    )
                self.b = np.mean(y[sv_idx] - preds)

    def _solve_smo(self, X, y):
        n, d = X.shape
        y = y.astype(float)

        alphas = np.zeros(n)
        b = 0.0
        passes = 0

        K = self._compute_kernel_matrix(X)
        errors = np.zeros(n)

        def _compute_error(i):
            return np.sum(alphas * y * K[:, i]) + b - y[i]

        def _update_all_errors():
            for k in range(n):
                errors[k] = _compute_error(k)

        _update_all_errors()

        while passes < self.max_passes:
            num_changed_alphas = 0

            for i in range(n):
                E_i = errors[i]
                r_i = y[i] * E_i

                if (r_i < -self.tol and alphas[i] < self.C) or (r_i > self.tol and alphas[i] > 0):
                    best_j = -1
                    best_delta = -1.0
                    for jj in range(n):
                        if jj == i:
                            continue
                        if alphas[jj] > self.tol and alphas[jj] < self.C - self.tol:
                            delta = abs(errors[jj] - E_i)
                            if delta > best_delta:
                                best_delta = delta
                                best_j = jj

                    if best_j >= 0:
                        j = best_j
                    else:
                        j = np.random.randint(n)
                        while j == i:
                            j = np.random.randint(n)

                    E_j = errors[j]
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    if y[i] != y[j]:
                        L = max(0.0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0.0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])

                    if L == H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alphas[j] = alphas[j] - y[j] * (E_i - E_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])

                    b1 = b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] \
                         - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] \
                         - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    _update_all_errors()

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        self.alphas_ = alphas
        self.b = b
        self.errors_ = errors

        sv_mask = alphas > self.tol
        self.support_alphas_ = alphas[sv_mask]
        self.support_vectors_ = X[sv_mask]
        self.support_labels_ = y[sv_mask]

        if self.kernel == 'linear':
            self.w = np.sum(
                (self.support_alphas_ * self.support_labels_)[:, None] * self.support_vectors_,
                axis=0
            )
        else:
            self.w = None

    def _solve_gd(self, X, y):
        if self.kernel != 'linear':
            raise ValueError("Gradient descent solver only supports linear kernel. Use 'qp' or 'smo' for non-linear kernels.")

        n, d = X.shape
        y = y.astype(float)

        self.w = np.zeros(d)
        self.b = 0.0

        for epoch in range(self.epochs):
            margins = y * (X @ self.w + self.b)
            violated = margins < 1.0

            grad_w = self.w.copy()
            grad_b = 0.0

            if np.any(violated):
                grad_w -= self.C * np.sum((y[violated, None] * X[violated]), axis=0)
                grad_b -= self.C * np.sum(y[violated])

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        decision = y * (X @ self.w + self.b)
        sv_mask = decision < 1.0 + self.tol
        self.support_vectors_ = X[sv_mask]
        self.support_labels_ = y[sv_mask]
        self.support_alphas_ = None
        self.alphas_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        assert set(y).issubset({-1.0, 1.0}), "Labels must be -1 or 1"

        self.X_train_ = X
        self.y_train_ = y

        if self.solver == 'qp':
            self._solve_qp(X, y)
        elif self.solver == 'smo':
            self._solve_smo(X, y)
        elif self.solver == 'gd':
            self._solve_gd(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)

        if self.kernel == 'linear' and self.w is not None:
            return X @ self.w + self.b
        else:
            if X.ndim == 1:
                X = X.reshape(1, -1)

            n_samples = X.shape[0]
            dec = np.zeros(n_samples)
            for i in range(n_samples):
                k = self._compute_kernel_single(self.support_vectors_, X[i])
                dec[i] = np.sum(self.support_alphas_ * self.support_labels_ * k) + self.b
            return dec

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y.astype(float))


def cross_val_score(svm, X, y, cv=5):
    n = len(X)
    indices = np.random.permutation(n)
    fold_size = n // cv
    scores = []

    for i in range(cv):
        start = i * fold_size
        end = start + fold_size if i < cv - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        svm.fit(X_train, y_train)
        scores.append(svm.score(X_val, y_val))

    return np.array(scores)


def grid_search_cv(X, y, param_grid, cv=5, solver='smo'):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_score = -1.0
    best_params = None
    all_results = []

    from itertools import product
    for combo in product(*values):
        params = dict(zip(keys, combo))
        svm = SVM(solver=solver, **params)
        scores = cross_val_score(svm, X, y, cv=cv)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        all_results.append((params, mean_score, std_score))

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score, all_results


if __name__ == '__main__':
    np.random.seed(42)

    linear_datasets = [
        ("Linearly Separable", *generate_data(100, separable=True)),
        ("Linearly Inseparable", *generate_data(100, separable=False)),
    ]

    non_linear_datasets = [
        ("XOR Dataset", *make_xor(100, noise=0.1)),
        ("Moons Dataset", *make_moons(100, noise=0.1)),
    ]

    for dataset_name, X, y in linear_datasets:
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        print()
        print("=" * 70)
        print(f"Dataset: {dataset_name}")
        print("=" * 70)

        for solver_name, solver_type in [("QP (CVXOPT)", "qp"), ("SMO", "smo")]:
            print()
            print(f"--- {solver_name} - Linear Kernel (C=1.0) ---")
            svm = SVM(solver=solver_type, C=1.0, kernel='linear', max_passes=20)
            svm.fit(X_train, y_train)

            print(f"  Weight vector w:      {svm.w}")
            print(f"  Bias b:               {svm.b:.6f}")
            print(f"  Support vectors:      {len(svm.support_vectors_)}")
            if svm.support_alphas_ is not None and len(svm.support_alphas_) <= 10:
                print(f"  Support alphas:       {np.round(svm.support_alphas_, 4)}")
            print(f"  Training accuracy:    {svm.score(X_train, y_train):.4f}")
            print(f"  Test accuracy:        {svm.score(X_test, y_test):.4f}")
            preds = svm.predict(X_test[:5])
            print(f"  First 5 test preds:   {preds}")

    for dataset_name, X, y in non_linear_datasets:
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        print()
        print("=" * 70)
        print(f"Dataset: {dataset_name}")
        print("=" * 70)

        print()
        print("--- Linear Kernel Baseline ---")
        svm_linear = SVM(solver='qp', C=1.0, kernel='linear')
        svm_linear.fit(X_train, y_train)
        print(f"  Training accuracy:    {svm_linear.score(X_train, y_train):.4f}")
        print(f"  Test accuracy:        {svm_linear.score(X_test, y_test):.4f}")

        param_grid_small = {
            'C': [0.1, 1.0, 10.0],
            'gamma': [0.1, 1.0],
            'kernel': ['rbf', 'poly'],
        }

        print()
        print("--- Grid Search + Cross Validation (CV=3, solver=qp) ---")
        best_params, best_cv_score, all_results = grid_search_cv(
            X_train, y_train, param_grid_small, cv=3, solver='qp'
        )
        print(f"  Best parameters: {best_params}")
        print(f"  Best CV score:   {best_cv_score:.4f}")

        print()
        print("  Top 5 results:")
        for params, mean, std in sorted(all_results, key=lambda x: -x[1])[:5]:
            print(f"    {params} -> mean={mean:.4f}, std={std:.4f}")

        for solver_name, solver_type in [("QP (CVXOPT)", "qp"), ("SMO", "smo")]:
            print()
            print(f"--- {solver_name} - Best Model ({best_params['kernel']}, C={best_params['C']}, gamma={best_params['gamma']}) ---")
            svm = SVM(solver=solver_type, max_passes=20, **best_params)
            svm.fit(X_train, y_train)

            print(f"  Bias b:               {svm.b:.6f}")
            print(f"  Support vectors:      {len(svm.support_vectors_)}")
            print(f"  Training accuracy:    {svm.score(X_train, y_train):.4f}")
            print(f"  Test accuracy:        {svm.score(X_test, y_test):.4f}")
            preds = svm.predict(X_test[:5])
            print(f"  First 5 test preds:   {preds}")
