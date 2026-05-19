import numpy as np


class BayesianLinearRegression:
    def __init__(self, alpha=None, beta=None, a=1.0, b=1.0, eps=1e-10, 
                 max_iter=100, tol=1e-6, fit_method='evidence'):
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.fit_method = fit_method
        self.w_mean = None
        self.w_cov = None
        self.a_post = None
        self.b_post = None
        self.Lambda_n_chol = None
        self.n_features = None
        self.alpha_history = []
        self.beta_history = []

    def _fit_fixed(self, X_with_bias, y, alpha, beta):
        n_samples, n_features = X_with_bias.shape
        
        Lambda_0 = alpha * np.eye(n_features)
        mu_0 = np.zeros(n_features)

        Lambda_n = beta * (X_with_bias.T @ X_with_bias) + Lambda_0
        
        Lambda_n_reg = Lambda_n + self.eps * np.eye(n_features)
        
        try:
            self.Lambda_n_chol = np.linalg.cholesky(Lambda_n_reg)
            temp = np.linalg.solve(self.Lambda_n_chol, np.eye(n_features))
            Lambda_n_inv = np.linalg.solve(self.Lambda_n_chol.T, temp)
        except np.linalg.LinAlgError:
            Lambda_n_inv = np.linalg.pinv(Lambda_n_reg)
            self.Lambda_n_chol = None

        mu_n = Lambda_n_inv @ (beta * X_with_bias.T @ y + Lambda_0 @ mu_0)

        self.a_post = self.a + n_samples / 2
        
        residuals = y - X_with_bias @ mu_n
        rss = residuals.T @ residuals
        prior_contrib = mu_n.T @ Lambda_0 @ mu_n
        
        self.b_post = self.b + 0.5 * (beta * rss + prior_contrib)
        
        self.b_post = max(self.b_post, self.eps)

        self.w_mean = mu_n
        sigma2 = self.b_post / self.a_post
        self.w_cov = sigma2 * Lambda_n_inv
        
        w, v = np.linalg.eigh(self.w_cov)
        w = np.maximum(w, self.eps)
        self.w_cov = v @ np.diag(w) @ v.T

        return Lambda_n, Lambda_n_inv, mu_n

    def _evidence_approximation(self, X_with_bias, y):
        n_samples, n_features = X_with_bias.shape
        
        alpha = 1.0 if self.alpha is None else self.alpha
        beta = 10.0 if self.beta is None else self.beta
        
        self.alpha_history = []
        self.beta_history = []
        
        for i in range(self.max_iter):
            alpha_old, beta_old = alpha, beta
            self.alpha_history.append(alpha)
            self.beta_history.append(beta)
            
            Lambda_n, Lambda_n_inv, mu_n = self._fit_fixed(X_with_bias, y, alpha, beta)
            
            eigenvalues = np.linalg.eigvalsh(beta * X_with_bias.T @ X_with_bias)
            
            gamma = np.sum(eigenvalues / (eigenvalues + alpha))
            
            residuals = y - X_with_bias @ mu_n
            rss = residuals.T @ residuals
            
            alpha = gamma / (mu_n.T @ mu_n + self.eps)
            beta = (n_samples - gamma) / (rss + self.eps)
            
            alpha = max(alpha, self.eps)
            beta = max(beta, self.eps)
            
            if abs(alpha - alpha_old) < self.tol and abs(beta - beta_old) < self.tol:
                break
        
        self.alpha = alpha
        self.beta = beta
        self.alpha_history.append(alpha)
        self.beta_history.append(beta)
        
        return alpha, beta

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_with_bias = np.hstack([X, np.ones((n_samples, 1))])
        self.n_features = n_features + 1
        
        if self.fit_method == 'evidence':
            alpha, beta = self._evidence_approximation(X_with_bias, y)
        else:
            alpha = 1.0 if self.alpha is None else self.alpha
            beta = 10.0 if self.beta is None else self.beta
            self.alpha = alpha
            self.beta = beta
        
        self._fit_fixed(X_with_bias, y, alpha, beta)
        
        return self

    def get_hyperparams(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'alpha_history': self.alpha_history,
            'beta_history': self.beta_history
        }

    def get_posterior_params(self):
        return {
            'w_mean': self.w_mean,
            'w_cov': self.w_cov,
            'a_post': self.a_post,
            'b_post': self.b_post
        }

    def predict(self, X_new, return_std=False):
        n_samples = X_new.shape[0]
        X_new_with_bias = np.hstack([X_new, np.ones((n_samples, 1))])

        y_mean = X_new_with_bias @ self.w_mean

        if self.Lambda_n_chol is not None:
            try:
                temp = np.linalg.solve(self.Lambda_n_chol, X_new_with_bias.T)
                x_Lambda_inv_x = np.sum(temp * temp, axis=0)
            except:
                Lambda_n_inv = np.linalg.inv(self.w_cov * self.a_post / self.b_post + self.eps * np.eye(self.n_features))
                x_Lambda_inv_x = np.sum(X_new_with_bias @ Lambda_n_inv * X_new_with_bias, axis=1)
        else:
            Lambda_n_inv = np.linalg.inv(self.w_cov * self.a_post / self.b_post + self.eps * np.eye(self.n_features))
            x_Lambda_inv_x = np.sum(X_new_with_bias @ Lambda_n_inv * X_new_with_bias, axis=1)

        sigma2 = self.b_post / self.a_post
        y_var = sigma2 * (1 + x_Lambda_inv_x)
        y_var = np.maximum(y_var, self.eps)
        y_std = np.sqrt(y_var)

        if return_std:
            return y_mean, y_std
        return y_mean

    def predict_distribution(self, X_new):
        n_samples = X_new.shape[0]
        X_new_with_bias = np.hstack([X_new, np.ones((n_samples, 1))])

        y_mean = X_new_with_bias @ self.w_mean

        if self.Lambda_n_chol is not None:
            try:
                temp = np.linalg.solve(self.Lambda_n_chol, X_new_with_bias.T)
                x_Lambda_inv_x = np.sum(temp * temp, axis=0)
            except:
                Lambda_n_inv = np.linalg.inv(self.w_cov * self.a_post / self.b_post + self.eps * np.eye(self.n_features))
                x_Lambda_inv_x = np.sum(X_new_with_bias @ Lambda_n_inv * X_new_with_bias, axis=1)
        else:
            Lambda_n_inv = np.linalg.inv(self.w_cov * self.a_post / self.b_post + self.eps * np.eye(self.n_features))
            x_Lambda_inv_x = np.sum(X_new_with_bias @ Lambda_n_inv * X_new_with_bias, axis=1)

        sigma2 = self.b_post / self.a_post
        y_var = sigma2 * (1 + x_Lambda_inv_x)
        y_var = np.maximum(y_var, self.eps)

        df = 2 * self.a_post

        return {
            'mean': y_mean,
            'var': y_var,
            'df': df,
            'dist_type': 't-distribution'
        }


if __name__ == '__main__':
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    true_w = np.array([2.0, 1.0])
    noise_std = 0.5
    y = X @ true_w[:-1] + true_w[-1] + np.random.randn(100) * noise_std

    print("=" * 50)
    print("使用证据近似自动估计超参数")
    print("=" * 50)
    model_evidence = BayesianLinearRegression(fit_method='evidence', max_iter=100, tol=1e-6)
    model_evidence.fit(X, y)
    
    hyperparams = model_evidence.get_hyperparams()
    print(f"估计的 alpha: {hyperparams['alpha']:.4f}")
    print(f"估计的 beta: {hyperparams['beta']:.4f}")
    print(f"理论 beta (1/noise^2): {1/(noise_std**2):.4f}")
    print(f"迭代次数: {len(hyperparams['alpha_history'])}")

    posterior = model_evidence.get_posterior_params()
    print(f"\n后验权重均值: {posterior['w_mean']}")
    print(f"真实权重: {true_w}")
    print(f"后验权重方差对角线: {np.diag(posterior['w_cov'])}")

    print("\n" + "=" * 50)
    print("使用固定超参数对比")
    print("=" * 50)
    model_fixed = BayesianLinearRegression(alpha=1.0, beta=4.0, fit_method='fixed')
    model_fixed.fit(X, y)
    
    posterior_fixed = model_fixed.get_posterior_params()
    print(f"固定 alpha=1.0, beta=4.0")
    print(f"后验权重均值: {posterior_fixed['w_mean']}")

    X_new = np.linspace(0, 10, 5).reshape(-1, 1)
    pred_dist = model_evidence.predict_distribution(X_new)
    print("\n预测分布:")
    print("均值:", pred_dist['mean'])
    print("方差:", pred_dist['var'])
    print("自由度:", pred_dist['df'])

    y_mean, y_std = model_evidence.predict(X_new, return_std=True)
    print("\n预测结果:")
    print("预测均值:", y_mean)
    print("预测标准差:", y_std)
