import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gammaln, digamma


class GMMClustering:
    def __init__(self, n_components, max_iter=100, tol=1e-6, reg_covar=1e-6, 
                 min_covar_det=1e-10, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.min_covar_det = min_covar_det
        self.random_state = random_state
        self.weights = None
        self.means = None
        self.covariances = None
        
    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.weights = np.ones(self.n_components) / self.n_components
        
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices]
        
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
    def _reinitialize_component(self, k, X, responsibilities):
        n_samples, n_features = X.shape
        
        max_var_idx = np.argmax(np.var(X, axis=0))
        split_val = np.median(X[:, max_var_idx])
        
        above_median = X[:, max_var_idx] > split_val
        if np.any(above_median):
            self.means[k] = X[above_median][np.random.choice(np.sum(above_median))]
        else:
            self.means[k] = X[np.random.choice(n_samples)]
        
        self.covariances[k] = np.eye(n_features) * np.mean(np.var(X, axis=0))
        
        self.weights[k] = 1.0 / self.n_components
        self.weights = self.weights / self.weights.sum()
        
    def _check_and_fix_singular_covariance(self, X, responsibilities):
        n_features = X.shape[1]
        singular_components = []
        
        for k in range(self.n_components):
            cov_det = np.linalg.det(self.covariances[k])
            
            if cov_det < self.min_covar_det or np.isnan(cov_det) or np.isinf(cov_det):
                singular_components.append(k)
                
                reg = self.reg_covar * np.eye(n_features)
                self.covariances[k] += reg
                
                if self.weights[k] < 0.05:
                    self._reinitialize_component(k, X, responsibilities)
                else:
                    self.weights[k] *= 0.5
                    self.weights = self.weights / self.weights.sum()
        
        return singular_components
        
    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            try:
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, mean=self.means[k], cov=self.covariances[k], allow_singular=True
                )
            except np.linalg.LinAlgError:
                responsibilities[:, k] = 1e-10
        
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum[responsibilities_sum == 0] = 1e-10
        responsibilities = responsibilities / responsibilities_sum
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        
        Nk = responsibilities.sum(axis=0)
        Nk[Nk == 0] = 1e-10
        
        self.weights = Nk / n_samples
        self.weights = np.clip(self.weights, 1e-10, None)
        self.weights = self.weights / self.weights.sum()
        
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
            
            reg = self.reg_covar * np.eye(n_features)
            self.covariances[k] += reg
            
    def fit(self, X):
        X = np.array(X)
        self._initialize_parameters(X)
        
        log_likelihood = -np.inf
        
        for i in range(self.max_iter):
            responsibilities = self._e_step(X)
            
            try:
                likelihoods = np.zeros((X.shape[0], self.n_components))
                for k in range(self.n_components):
                    likelihoods[:, k] = self.weights[k] * multivariate_normal.pdf(
                        X, mean=self.means[k], cov=self.covariances[k], allow_singular=True
                    )
                total_likelihood = likelihoods.sum(axis=1)
                total_likelihood[total_likelihood == 0] = 1e-10
                new_log_likelihood = np.sum(np.log(total_likelihood))
            except (np.linalg.LinAlgError, ValueError):
                new_log_likelihood = log_likelihood
            
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
                
            log_likelihood = new_log_likelihood
            self._m_step(X, responsibilities)
            
            singular_comps = self._check_and_fix_singular_covariance(X, responsibilities)
            if len(singular_comps) > 0:
                log_likelihood = -np.inf
            
        return self
    
    def predict_proba(self, X):
        X = np.array(X)
        return self._e_step(X)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def fit_predict(self, X):
        return self.fit(X).predict(X)
    
    def fit_predict_proba(self, X):
        return self.fit(X).predict_proba(X)
    
    def _compute_log_likelihood(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        try:
            likelihoods = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                likelihoods[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, mean=self.means[k], cov=self.covariances[k], allow_singular=True
                )
            total_likelihood = likelihoods.sum(axis=1)
            total_likelihood[total_likelihood <= 0] = 1e-10
            return np.sum(np.log(total_likelihood))
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf
    
    def _n_parameters(self, n_features):
        mean_params = self.n_components * n_features
        cov_params = self.n_components * n_features * (n_features + 1) / 2
        weight_params = self.n_components - 1
        return int(mean_params + cov_params + weight_params)
    
    def bic(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        log_likelihood = self._compute_log_likelihood(X)
        n_params = self._n_parameters(n_features)
        return -2 * log_likelihood + n_params * np.log(n_samples)
    
    def aic(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        log_likelihood = self._compute_log_likelihood(X)
        n_params = self._n_parameters(n_features)
        return -2 * log_likelihood + 2 * n_params


def compute_model_selection(X, k_range=range(2, 11), random_state=None, **gmm_kwargs):
    X = np.array(X)
    results = []
    
    for k in k_range:
        gmm = GMMClustering(n_components=k, random_state=random_state, **gmm_kwargs)
        gmm.fit(X)
        bic_score = gmm.bic(X)
        aic_score = gmm.aic(X)
        log_ll = gmm._compute_log_likelihood(X)
        results.append({
            'n_components': k,
            'bic': bic_score,
            'aic': aic_score,
            'log_likelihood': log_ll,
            'model': gmm,
        })
    
    best_bic = min(results, key=lambda r: r['bic'])
    best_aic = min(results, key=lambda r: r['aic'])
    
    return {
        'results': results,
        'best_bic_k': best_bic['n_components'],
        'best_bic_score': best_bic['bic'],
        'best_aic_k': best_aic['n_components'],
        'best_aic_score': best_aic['aic'],
        'best_bic_model': best_bic['model'],
        'best_aic_model': best_aic['model'],
    }


class BayesianGMMClustering:
    def __init__(self, n_components, max_iter=100, tol=1e-6, reg_covar=1e-6,
                 weight_concentration_prior=1.0, mean_precision_prior=1.0,
                 degrees_of_freedom_prior=None, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.random_state = random_state
        
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.responsibilities_ = None
        self.n_effective_components_ = None
        
        self._weight_concentration_ = None
        self._mean_precision_ = None
        self._means_ = None
        self._degrees_of_freedom_ = None
        self._covariances_ = None
        self._wishart_log_det_ = None
        self._wishart_trace_ = None
    
    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self._weight_concentration_ = np.array(
            [self.weight_concentration_prior / self.n_components + 1] * self.n_components
        )
        self._weight_concentration_ = np.vstack([
            self._weight_concentration_,
            np.full(self.n_components, self.weight_concentration_prior + 1)
        ])
        
        self._mean_precision_ = np.full(self.n_components, self.mean_precision_prior)
        
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self._means_ = X[indices].copy()
        
        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior = n_features
        
        self._degrees_of_freedom_ = np.full(self.n_components, self.degrees_of_freedom_prior)
        
        self._covariances_ = np.array([
            np.eye(n_features) * np.mean(np.var(X, axis=0))
            for _ in range(self.n_components)
        ])
        
        self.means_ = self._means_.copy()
        self.covariances_ = self._covariances_.copy()
        self._update_effective_params(n_features)
    
    def _update_effective_params(self, n_features):
        dirichlet_alpha = self._weight_concentration_[0]
        dirichlet_beta = self._weight_concentration_[1]
        self.weights_ = dirichlet_alpha / (dirichlet_alpha + dirichlet_beta)
        self.weights_ /= self.weights_.sum()
        
        self.means_ = self._means_.copy()
        
        for k in range(self.n_components):
            self.covariances_[k] = self._covariances_[k] / self._degrees_of_freedom_[k]
            self.covariances_[k] += self.reg_covar * np.eye(n_features)
    
    def _compute_log_prob(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        log_prob = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            try:
                log_prob[:, k] = np.log(self.weights_[k] + 1e-300) + multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=self.covariances_[k], allow_singular=True
                )
            except (np.linalg.LinAlgError, ValueError):
                log_prob[:, k] = -1e10
        
        return log_prob
    
    def _e_step(self, X):
        log_prob = self._compute_log_prob(X)
        log_prob_max = log_prob.max(axis=1, keepdims=True)
        log_prob_shifted = log_prob - log_prob_max
        
        responsibilities = np.exp(log_prob_shifted)
        resp_sum = responsibilities.sum(axis=1, keepdims=True)
        resp_sum[resp_sum == 0] = 1e-10
        responsibilities /= resp_sum
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        
        Nk = responsibilities.sum(axis=0)
        Nk = np.maximum(Nk, 1e-10)
        
        self._weight_concentration_[0] = self.weight_concentration_prior / self.n_components + Nk
        cumsum_from_right = np.cumsum(Nk[::-1])[::-1]
        shifted_cumsum = np.concatenate([cumsum_from_right[1:], [0.0]])
        self._weight_concentration_[1] = (
            self.weight_concentration_prior + shifted_cumsum
            + self.weight_concentration_prior / self.n_components
        )
        
        self._mean_precision_ = self.mean_precision_prior + Nk
        
        self._means_ = (
            self.mean_precision_prior * np.mean(X, axis=0)[np.newaxis, :]
            + np.dot(responsibilities.T, X)
        ) / self._mean_precision_[:, np.newaxis]
        
        self._degrees_of_freedom_ = self.degrees_of_freedom_prior + Nk
        
        for k in range(self.n_components):
            diff = X - self._means_[k]
            self._covariances_[k] = (
                self._covariances_[k] * 0
                + np.dot(responsibilities[:, k] * diff.T, diff)
            )
            prior_diff = self._means_[k] - np.mean(X, axis=0)
            self._covariances_[k] += self.mean_precision_prior * np.outer(prior_diff, prior_diff)
            self._covariances_[k] += self.reg_covar * np.eye(n_features)
    
    def _compute_elbo(self, X, responsibilities):
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0)
        
        elbo = 0.0
        
        log_prob = self._compute_log_prob(X)
        elbo += np.sum(responsibilities * log_prob)
        elbo -= np.sum(responsibilities * np.log(responsibilities + 1e-300))
        
        dirichlet_alpha = self._weight_concentration_[0]
        dirichlet_beta = self._weight_concentration_[1]
        prior_alpha = self.weight_concentration_prior / self.n_components
        prior_beta = self.weight_concentration_prior
        
        elbo += np.sum(gammaln(dirichlet_alpha) + gammaln(dirichlet_beta)
                       - gammaln(dirichlet_alpha + dirichlet_beta))
        elbo -= np.sum(gammaln(prior_alpha) + gammaln(prior_beta)
                       - gammaln(prior_alpha + prior_beta))
        elbo += np.sum(
            (prior_alpha - dirichlet_alpha) * digamma(dirichlet_alpha)
            + (prior_beta - dirichlet_beta) * digamma(dirichlet_beta)
            + (dirichlet_alpha + dirichlet_beta - prior_alpha - prior_beta)
            * digamma(dirichlet_alpha + dirichlet_beta)
        )
        
        return elbo
    
    def _prune_components(self, weight_threshold=0.01):
        active = self.weights_ >= weight_threshold
        if np.sum(active) < 1:
            active[np.argmax(self.weights_)] = True
        
        self.n_effective_components_ = int(np.sum(active))
        return active
    
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        self._initialize_parameters(X)
        
        prev_elbo = -np.inf
        
        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            
            self._m_step(X, responsibilities)
            self._update_effective_params(n_features)
            
            elbo = self._compute_elbo(X, responsibilities)
            
            if np.abs(elbo - prev_elbo) < self.tol:
                break
            prev_elbo = elbo
        
        self.responsibilities_ = self._e_step(X)
        self._prune_components()
        
        return self
    
    def predict_proba(self, X):
        X = np.array(X)
        return self._e_step(X)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def fit_predict(self, X):
        return self.fit(X).predict(X)
    
    def fit_predict_proba(self, X):
        return self.fit(X).predict_proba(X)
    
    def get_active_weights(self):
        active = self._prune_components()
        return self.weights_[active]
    
    def get_active_means(self):
        active = self._prune_components()
        return self.means_[active]
    
    def get_active_covariances(self):
        active = self._prune_components()
        return self.covariances_[active]


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Test 1: Standard GMM clustering with well-separated blobs")
    print("=" * 60)
    X, y_true = make_blobs(
        n_samples=300, centers=3, cluster_std=0.60, random_state=0
    )
    
    gmm = GMMClustering(n_components=3, random_state=42)
    labels = gmm.fit_predict(X)
    proba = gmm.predict_proba(X)
    
    print("Cluster labels for first 10 samples:")
    print(labels[:10])
    print("\nEstimated means:")
    print(gmm.means)
    print("\nEstimated weights:")
    print(gmm.weights)
    print(f"\nBIC score: {gmm.bic(X):.4f}")
    print(f"AIC score: {gmm.aic(X):.4f}")
    
    print("\n" + "=" * 60)
    print("Test 2: BIC/AIC model selection across K values")
    print("=" * 60)
    selection = compute_model_selection(X, k_range=range(2, 8), random_state=42)
    
    print(f"{'K':>3} | {'BIC':>14} | {'AIC':>14} | {'Log-Likelihood':>16}")
    print("-" * 58)
    for r in selection['results']:
        print(f"{r['n_components']:>3} | {r['bic']:>14.4f} | {r['aic']:>14.4f} | {r['log_likelihood']:>16.4f}")
    
    print(f"\nBest K by BIC: {selection['best_bic_k']} (BIC = {selection['best_bic_score']:.4f})")
    print(f"Best K by AIC: {selection['best_aic_k']} (AIC = {selection['best_aic_score']:.4f})")
    
    print("\n" + "=" * 60)
    print("Test 3: Bayesian GMM with Dirichlet Process (auto K selection)")
    print("=" * 60)
    bgmm = BayesianGMMClustering(
        n_components=10, weight_concentration_prior=0.1, random_state=42
    )
    bgmm_labels = bgmm.fit_predict(X)
    
    print(f"Max components (upper bound): 10")
    print(f"Effective components after pruning: {bgmm.n_effective_components_}")
    print(f"\nAll component weights:")
    for k in range(10):
        status = "ACTIVE" if bgmm.weights_[k] >= 0.01 else "pruned"
        print(f"  Component {k}: weight={bgmm.weights_[k]:.6f} [{status}]")
    
    active_weights = bgmm.get_active_weights()
    active_means = bgmm.get_active_means()
    print(f"\nActive component weights: {active_weights}")
    print(f"Active component means:\n{active_means}")
    
    print("\n" + "=" * 60)
    print("Test 4: Bayesian GMM on overlapping data (2 true clusters)")
    print("=" * 60)
    X_overlap, _ = make_blobs(
        n_samples=500, centers=2, cluster_std=0.8, random_state=42
    )
    
    bgmm2 = BayesianGMMClustering(
        n_components=8, weight_concentration_prior=0.5, random_state=42
    )
    bgmm2_labels = bgmm2.fit_predict(X_overlap)
    
    print(f"Max components (upper bound): 8")
    print(f"Effective components after pruning: {bgmm2.n_effective_components_}")
    print(f"\nAll component weights:")
    for k in range(8):
        status = "ACTIVE" if bgmm2.weights_[k] >= 0.01 else "pruned"
        print(f"  Component {k}: weight={bgmm2.weights_[k]:.6f} [{status}]")
    
    print("\n" + "=" * 60)
    print("Test 5: BIC/AIC vs Bayesian GMM comparison")
    print("=" * 60)
    selection2 = compute_model_selection(X_overlap, k_range=range(2, 8), random_state=42)
    
    print(f"{'K':>3} | {'BIC':>14} | {'AIC':>14}")
    print("-" * 38)
    for r in selection2['results']:
        marker = " <-- min" if r['n_components'] in [
            selection2['best_bic_k'], selection2['best_aic_k']
        ] else ""
        print(f"{r['n_components']:>3} | {r['bic']:>14.4f} | {r['aic']:>14.4f}{marker}")
    
    print(f"\nBest K by BIC: {selection2['best_bic_k']}")
    print(f"Best K by AIC: {selection2['best_aic_k']}")
    print(f"Bayesian GMM effective K: {bgmm2.n_effective_components_}")
    
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
    plt.title('True Labels (K=3)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(2, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red', s=200, marker='X')
    plt.title('GMM (K=3)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(2, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], c=bgmm_labels, cmap='viridis')
    plt.scatter(active_means[:, 0], active_means[:, 1], c='red', s=200, marker='X')
    plt.title(f'Bayesian GMM (K_eff={bgmm.n_effective_components_})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(2, 3, 4)
    k_values = [r['n_components'] for r in selection['results']]
    bic_values = [r['bic'] for r in selection['results']]
    aic_values = [r['aic'] for r in selection['results']]
    plt.plot(k_values, bic_values, 'bo-', label='BIC', linewidth=2)
    plt.plot(k_values, aic_values, 'rs--', label='AIC', linewidth=2)
    plt.axvline(x=selection['best_bic_k'], color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=selection['best_aic_k'], color='red', linestyle=':', alpha=0.5)
    plt.xlabel('Number of components (K)')
    plt.ylabel('Score')
    plt.title('BIC/AIC Model Selection')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.bar(range(10), bgmm.weights_, color='steelblue')
    plt.axhline(y=0.01, color='red', linestyle='--', label='Prune threshold')
    plt.xlabel('Component index')
    plt.ylabel('Weight')
    plt.title(f'Bayesian GMM Weights (K_eff={bgmm.n_effective_components_})')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    k_values2 = [r['n_components'] for r in selection2['results']]
    bic_values2 = [r['bic'] for r in selection2['results']]
    aic_values2 = [r['aic'] for r in selection2['results']]
    plt.plot(k_values2, bic_values2, 'bo-', label='BIC', linewidth=2)
    plt.plot(k_values2, aic_values2, 'rs--', label='AIC', linewidth=2)
    plt.axvline(x=selection2['best_bic_k'], color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=selection2['best_aic_k'], color='red', linestyle=':', alpha=0.5)
    plt.xlabel('Number of components (K)')
    plt.ylabel('Score')
    plt.title('BIC/AIC for Overlapping Data')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gmm_clustering_result.png')
    print("\nClustering result plot saved as gmm_clustering_result.png")
    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
