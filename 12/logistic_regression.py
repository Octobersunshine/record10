import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True,
                 reg_lambda=0.0, early_stopping=False, patience=10, min_delta=0.0001):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.reg_lambda = reg_lambda
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.weights = None
        self.loss_history = []
        self.val_loss_history = []
        self.best_weights = None
        self.best_iteration = 0

    def _sigmoid(self, z):
        pos_mask = z >= 0
        neg_mask = ~pos_mask
        result = np.zeros_like(z, dtype=np.float64)
        result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
        exp_z = np.exp(z[neg_mask])
        result[neg_mask] = exp_z / (1 + exp_z)
        return result

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _compute_loss(self, y_true, y_pred, weights=None):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cross_entropy = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        if self.reg_lambda > 0 and weights is not None:
            if self.fit_intercept:
                reg_weights = weights[1:]
            else:
                reg_weights = weights
            l2_term = 0.5 * self.reg_lambda * np.sum(reg_weights ** 2)
            cross_entropy += l2_term

        return cross_entropy

    def fit(self, X, y, X_val=None, y_val=None):
        if self.fit_intercept:
            X = self._add_intercept(X)
            if X_val is not None:
                X_val = self._add_intercept(X_val)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.loss_history = []
        self.val_loss_history = []
        self.best_weights = None
        self.best_iteration = 0
        best_val_loss = float('inf')
        patience_counter = 0

        for i in range(self.num_iterations):
            z = np.dot(X, self.weights)
            y_pred = self._sigmoid(z)

            gradient = np.dot(X.T, (y_pred - y)) / n_samples
            if self.reg_lambda > 0:
                reg_gradient = self.reg_lambda * self.weights
                if self.fit_intercept:
                    reg_gradient[0] = 0
                gradient += reg_gradient

            self.weights -= self.learning_rate * gradient

            loss = self._compute_loss(y, y_pred, self.weights)
            self.loss_history.append(loss)

            if self.early_stopping and X_val is not None and y_val is not None:
                z_val = np.dot(X_val, self.weights)
                y_pred_val = self._sigmoid(z_val)
                val_loss = self._compute_loss(y_val, y_pred_val, self.weights)
                self.val_loss_history.append(val_loss)

                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    self.best_weights = self.weights.copy()
                    self.best_iteration = i
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self.weights = self.best_weights
                        break

        if self.early_stopping and self.best_weights is not None:
            self.weights = self.best_weights

        return self.weights

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        z = np.dot(X, self.weights)
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def get_weights(self):
        return self.weights

    def get_loss_history(self):
        return self.loss_history

    def get_val_loss_history(self):
        return self.val_loss_history

    def get_best_iteration(self):
        return self.best_iteration
