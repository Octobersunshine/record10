import numpy as np


class GaussianNaiveBayes:
    VAR_FLOOR = 1e-9

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.means = {}
        self.variances = {}
        self.zero_var_mask = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.means[c] = np.mean(X_c, axis=0)
            raw_var = np.var(X_c, axis=0)
            self.zero_var_mask[c] = (raw_var == 0)
            self.variances[c] = np.maximum(raw_var, self.VAR_FLOOR)

    def _log_gaussian_pdf(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - np.power(x - mean, 2) / (2 * var)

    def predict(self, X):
        predictions = []
        posteriors = []

        for x in X:
            class_posteriors = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                log_pdf = self._log_gaussian_pdf(x, self.means[c], self.variances[c])
                log_pdf[self.zero_var_mask[c]] = 0.0
                likelihood = np.sum(log_pdf)
                class_posteriors[c] = prior + likelihood

            total = np.logaddexp.reduce(list(class_posteriors.values()))
            normalized_posteriors = {c: np.exp(posterior - total) for c, posterior in class_posteriors.items()}
            predicted_class = max(class_posteriors, key=class_posteriors.get)

            predictions.append(predicted_class)
            posteriors.append(normalized_posteriors)

        return np.array(predictions), posteriors


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.feature_log_probs = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            feature_count = np.sum(X_c, axis=0) + self.alpha
            total_count = np.sum(feature_count)
            self.feature_log_probs[c] = np.log(feature_count / total_count)

    def predict(self, X):
        predictions = []
        posteriors = []

        for x in X:
            class_posteriors = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(x * self.feature_log_probs[c])
                class_posteriors[c] = prior + likelihood

            total = np.logaddexp.reduce(list(class_posteriors.values()))
            normalized_posteriors = {c: np.exp(posterior - total) for c, posterior in class_posteriors.items()}
            predicted_class = max(class_posteriors, key=class_posteriors.get)

            predictions.append(predicted_class)
            posteriors.append(normalized_posteriors)

        return np.array(predictions), posteriors


class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X, y):
        X = (X > self.binarize).astype(np.float64)
        self.classes = np.unique(y)
        self.priors = {}
        self.feature_log_probs_present = {}
        self.feature_log_probs_absent = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            n_c = len(X_c)
            feature_present = np.sum(X_c, axis=0) + self.alpha
            feature_absent = n_c - np.sum(X_c, axis=0) + self.alpha
            denom = n_c + 2 * self.alpha
            self.feature_log_probs_present[c] = np.log(feature_present / denom)
            self.feature_log_probs_absent[c] = np.log(feature_absent / denom)

    def predict(self, X):
        X = (X > self.binarize).astype(np.float64)
        predictions = []
        posteriors = []

        for x in X:
            class_posteriors = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                log_prob = x * self.feature_log_probs_present[c] + (1 - x) * self.feature_log_probs_absent[c]
                likelihood = np.sum(log_prob)
                class_posteriors[c] = prior + likelihood

            total = np.logaddexp.reduce(list(class_posteriors.values()))
            normalized_posteriors = {c: np.exp(posterior - total) for c, posterior in class_posteriors.items()}
            predicted_class = max(class_posteriors, key=class_posteriors.get)

            predictions.append(predicted_class)
            posteriors.append(normalized_posteriors)

        return np.array(predictions), posteriors


class NaiveBayes:
    VALID_TYPES = {"gaussian", "multinomial", "bernoulli", "auto"}

    def __init__(self, model_type="auto", alpha=1.0, binarize=0.0):
        if model_type not in self.VALID_TYPES:
            raise ValueError(f"model_type must be one of {self.VALID_TYPES}, got '{model_type}'")
        self.model_type = model_type
        self.alpha = alpha
        self.binarize = binarize
        self.model = None

    @staticmethod
    def _detect_model_type(X):
        if np.all((X == 0) | (X == 1)):
            return "bernoulli"
        if np.all(X >= 0) and np.all(X == X.astype(int)):
            return "multinomial"
        return "gaussian"

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        effective_type = self.model_type
        if effective_type == "auto":
            effective_type = self._detect_model_type(X)

        if effective_type == "gaussian":
            self.model = GaussianNaiveBayes()
        elif effective_type == "multinomial":
            self.model = MultinomialNaiveBayes(alpha=self.alpha)
        elif effective_type == "bernoulli":
            self.model = BernoulliNaiveBayes(alpha=self.alpha, binarize=self.binarize)

        self.effective_type = effective_type
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict(X)


if __name__ == "__main__":
    print("=" * 50)
    print("1. Gaussian Naive Bayes (连续特征)")
    print("=" * 50)
    X_gauss = np.array([
        [1.0, 2.0, 3.0],
        [1.5, 1.8, 3.0],
        [5.0, 8.0, 3.0],
        [6.0, 8.0, 3.0],
        [1.2, 1.9, 3.0],
        [5.5, 7.5, 3.0]
    ])
    y_gauss = np.array([0, 0, 1, 1, 0, 1])

    model = GaussianNaiveBayes()
    model.fit(X_gauss, y_gauss)
    print("先验概率:", model.priors)
    preds, posts = model.predict(np.array([[1.3, 2.1, 3.0], [5.2, 7.8, 3.0]]))
    print("预测类别:", preds)
    print("后验概率:", posts)

    print()
    print("=" * 50)
    print("2. Multinomial Naive Bayes (词频/离散特征)")
    print("=" * 50)
    X_multi = np.array([
        [3, 0, 1],
        [2, 1, 0],
        [0, 5, 3],
        [1, 4, 2],
        [4, 0, 1],
        [0, 6, 4]
    ])
    y_multi = np.array([0, 0, 1, 1, 0, 1])

    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X_multi, y_multi)
    print("先验概率:", model.priors)
    preds, posts = model.predict(np.array([[3, 0, 1], [0, 5, 3]]))
    print("预测类别:", preds)
    print("后验概率:", posts)

    print()
    print("=" * 50)
    print("3. Bernoulli Naive Bayes (二元特征)")
    print("=" * 50)
    X_bern = np.array([
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])
    y_bern = np.array([0, 0, 1, 1, 0, 1])

    model = BernoulliNaiveBayes(alpha=1.0)
    model.fit(X_bern, y_bern)
    print("先验概率:", model.priors)
    preds, posts = model.predict(np.array([[1, 0, 1], [0, 1, 1]]))
    print("预测类别:", preds)
    print("后验概率:", posts)

    print()
    print("=" * 50)
    print("4. NaiveBayes 自动选择模式")
    print("=" * 50)

    auto_gauss = NaiveBayes(model_type="auto")
    auto_gauss.fit(X_gauss, y_gauss)
    print(f"连续特征数据 → 自动选择: {auto_gauss.effective_type}")
    preds, _ = auto_gauss.predict(np.array([[1.3, 2.1, 3.0]]))
    print("预测类别:", preds)

    auto_multi = NaiveBayes(model_type="auto")
    auto_multi.fit(X_multi, y_multi)
    print(f"词频特征数据 → 自动选择: {auto_multi.effective_type}")
    preds, _ = auto_multi.predict(np.array([[3, 0, 1]]))
    print("预测类别:", preds)

    auto_bern = NaiveBayes(model_type="auto")
    auto_bern.fit(X_bern, y_bern)
    print(f"二元特征数据 → 自动选择: {auto_bern.effective_type}")
    preds, _ = auto_bern.predict(np.array([[1, 0, 1]]))
    print("预测类别:", preds)
