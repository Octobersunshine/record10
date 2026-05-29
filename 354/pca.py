import warnings

import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.eigenvalues_ = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.n_components > min(n_samples, n_features):
            warnings.warn(
                f"n_components={self.n_components} 超过了 min(n_samples, n_features)"
                f"={min(n_samples, n_features)}，将自动调整为 {min(n_samples, n_features)}",
                RuntimeWarning,
            )
            self.n_components = min(n_samples, n_features)

        if n_samples < n_features:
            warnings.warn(
                f"样本数({n_samples})小于特征数({n_features})，协方差矩阵为奇异矩阵，"
                "使用SVD替代特征分解以保证数值稳定性",
                RuntimeWarning,
            )

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.eigenvalues_ = S**2 / (n_samples - 1)

        total_variance = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = self.eigenvalues_[:self.n_components] / total_variance

        self.components_ = Vt[:self.n_components].T

        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components_.T + self.mean_


class KernelPCA:
    def __init__(self, n_components, kernel='rbf', gamma=None, degree=3, coef0=1):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alphas_ = None
        self.lambdas_ = None
        self.X_fit_ = None
        self.K_center_ = None

    def _compute_kernel(self, X, Y=None):
        if Y is None:
            Y = X

        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)

        n_features = X.shape[1]
        gamma = self.gamma if self.gamma is not None else 1.0 / n_features

        if self.kernel == 'linear':
            return X @ Y.T
        elif self.kernel == 'rbf':
            sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * X @ Y.T
            return np.exp(-gamma * sq_dists)
        elif self.kernel == 'poly':
            return (gamma * X @ Y.T + self.coef0) ** self.degree
        elif self.kernel == 'sigmoid':
            return np.tanh(gamma * X @ Y.T + self.coef0)
        else:
            raise ValueError(f"不支持的核函数: {self.kernel}")

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        n_samples = X.shape[0]

        if self.n_components >= n_samples:
            warnings.warn(
                f"n_components={self.n_components} 超过或等于样本数{n_samples}，"
                f"将自动调整为 {n_samples - 1}",
                RuntimeWarning,
            )
            self.n_components = n_samples - 1

        self.X_fit_ = X

        K = self._compute_kernel(X)

        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        self.K_center_ = K_centered

        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        eigenvalues = np.maximum(eigenvalues, 0)

        self.lambdas_ = eigenvalues[:self.n_components]
        self.alphas_ = eigenvectors[:, :self.n_components] / np.sqrt(self.lambdas_ + 1e-10)

        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        n_samples_fit = self.X_fit_.shape[0]
        n_samples = X.shape[0]

        K = self._compute_kernel(X, self.X_fit_)

        one_n = np.ones((n_samples, n_samples_fit)) / n_samples_fit
        one_fit = np.ones((n_samples_fit, n_samples_fit)) / n_samples_fit

        K_centered = K - one_n @ self.K_center_ - K @ one_fit + one_n @ self.K_center_ @ one_fit

        return K_centered @ self.alphas_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def reconstruct(self, X_transformed):
        K_reconstructed = X_transformed @ np.diag(np.sqrt(self.lambdas_ + 1e-10)) @ self.alphas_.T
        return K_reconstructed

    def reconstruction_error(self, X):
        X_transformed = self.transform(X)
        n_samples = X.shape[0]
        K_original = self.K_center_
        K_reconstructed = X_transformed @ np.diag(self.lambdas_) @ self.alphas_.T

        error = np.mean((K_original - K_reconstructed) ** 2)
        return error


def main():
    print("=" * 60)
    print("场景1: 标准PCA (样本数 > 特征数)")
    print("=" * 60)
    np.random.seed(42)
    X_normal = np.random.randn(100, 5)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_normal)
    X_reconstructed = pca.inverse_transform(X_reduced)
    recon_error = np.mean((X_normal - X_reconstructed) ** 2)

    print("降维后的数据形状:", X_reduced.shape)
    print("解释方差比:", pca.explained_variance_ratio_)
    print("累计解释方差比:", np.sum(pca.explained_variance_ratio_))
    print("重建误差 (MSE):", recon_error)
    print("降维后的前5个样本:\n", X_reduced[:5])

    print("\n" + "=" * 60)
    print("场景2: 样本数 < 特征数 (协方差矩阵奇异)")
    print("=" * 60)
    np.random.seed(42)
    X_singular = np.random.randn(10, 50)

    pca2 = PCA(n_components=3)
    X_reduced2 = pca2.fit_transform(X_singular)

    print("降维后的数据形状:", X_reduced2.shape)
    print("解释方差比:", pca2.explained_variance_ratio_)
    print("累计解释方差比:", np.sum(pca2.explained_variance_ratio_))
    print("特征值:", pca2.eigenvalues_[:3])

    print("\n" + "=" * 60)
    print("场景3: 核PCA - RBF核 (非线性降维)")
    print("=" * 60)
    np.random.seed(42)
    X_circle = np.random.randn(200, 2)
    labels = np.array([0 if x1**2 + x2**2 < 1 else 1 for x1, x2 in X_circle])

    kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=10.0)
    X_kpca_rbf = kpca_rbf.fit_transform(X_circle)
    recon_error_rbf = kpca_rbf.reconstruction_error(X_circle)

    print("原始数据形状:", X_circle.shape)
    print("RBF核降维后形状:", X_kpca_rbf.shape)
    print("RBF核重建误差 (核空间MSE):", recon_error_rbf)
    print("降维后的前5个样本:\n", X_kpca_rbf[:5])

    print("\n" + "=" * 60)
    print("场景4: 核PCA - 多项式核")
    print("=" * 60)
    kpca_poly = KernelPCA(n_components=2, kernel='poly', degree=3, gamma=1.0, coef0=1)
    X_kpca_poly = kpca_poly.fit_transform(X_circle)
    recon_error_poly = kpca_poly.reconstruction_error(X_circle)

    print("多项式核 (degree=3) 降维后形状:", X_kpca_poly.shape)
    print("多项式核重建误差:", recon_error_poly)
    print("降维后的前5个样本:\n", X_kpca_poly[:5])

    print("\n" + "=" * 60)
    print("场景5: 线性核PCA 与 标准PCA 对比")
    print("=" * 60)
    kpca_linear = KernelPCA(n_components=2, kernel='linear')
    X_kpca_linear = kpca_linear.fit_transform(X_normal)

    print("标准PCA前3个样本:\n", X_reduced[:3])
    print("线性核PCA前3个样本:\n", X_kpca_linear[:3])


if __name__ == "__main__":
    main()
