import numpy as np


def _compute_pairwise_distances(X):
    sum_X = np.sum(X ** 2, axis=1)
    D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2.0 * X @ X.T
    np.maximum(D, 0, out=D)
    return D


def _pca_init(X, n_components, random_state=None):
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:n_components]]
    Y = X_centered @ eigvecs
    Y = Y / np.std(Y[:, 0]) * 1e-4
    return Y


def _binary_search_perplexity(D_i, target_perplexity, tol=1e-5, max_iter=50):
    lo, hi = 1e-20, 1e4
    sigma_sq = 1.0
    for _ in range(max_iter):
        P_i = np.exp(-D_i / (2.0 * sigma_sq))
        P_i[D_i == 0] = 0.0
        sum_P = np.maximum(P_i.sum(), 1e-12)
        P_i /= sum_P
        entropy = -np.sum(P_i[P_i > 1e-12] * np.log2(P_i[P_i > 1e-12]))
        perp = 2.0 ** entropy
        if abs(perp - target_perplexity) < tol:
            break
        if perp < target_perplexity:
            lo = sigma_sq
            sigma_sq = (sigma_sq + hi) / 2.0 if hi < 1e4 else sigma_sq * 2.0
        else:
            hi = sigma_sq
            sigma_sq = (lo + sigma_sq) / 2.0
    return sigma_sq


def _compute_high_dim_affinities(X, perplexity=30.0):
    n = X.shape[0]
    D = _compute_pairwise_distances(X)
    P = np.zeros((n, n))
    for i in range(n):
        sigma_sq = _binary_search_perplexity(D[i], perplexity)
        P[i] = np.exp(-D[i] / (2.0 * sigma_sq))
        P[i][D[i] == 0] = 0.0
        row_sum = P[i].sum()
        if row_sum > 0:
            P[i] /= row_sum
    P = (P + P.T) / (2.0 * n)
    np.maximum(P, 1e-12, out=P)
    return P


def _compute_low_dim_affinities(Y):
    D = _compute_pairwise_distances(Y)
    Q = 1.0 / (1.0 + D)
    np.fill_diagonal(Q, 0.0)
    Q_sum = np.maximum(Q.sum(), 1e-12)
    Q /= Q_sum
    np.maximum(Q, 1e-12, out=Q)
    return Q


def _compute_gradient(P, Q, Y):
    n = Y.shape[0]
    D = _compute_pairwise_distances(Y)
    PQ_diff = P - Q
    inv_dist = 1.0 / (1.0 + D)
    grad = np.zeros_like(Y)
    for i in range(n):
        diff = Y[i] - Y
        coeff = (PQ_diff[i] * inv_dist[i])[:, np.newaxis]
        grad[i] = 4.0 * np.sum(coeff * diff, axis=0)
    return grad


def tsne(
    X,
    n_components=2,
    perplexity=None,
    learning_rate=200.0,
    n_iter=1000,
    momentum=0.8,
    early_exaggeration=4.0,
    early_exaggeration_iters=100,
    init="pca",
    early_stop_tol=1e-6,
    early_stop_patience=50,
    random_state=None,
):
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    if n < 2:
        raise ValueError("t-SNE requires at least 2 samples, got {}".format(n))

    if init not in ("random", "pca"):
        raise ValueError("init must be 'random' or 'pca', got {}".format(init))

    auto_perplexity = min(30.0, n - 1)
    if perplexity is None:
        perplexity = auto_perplexity
    else:
        if perplexity >= n:
            raise ValueError(
                "perplexity ({}) must be less than n_samples ({}). "
                "Consider using the auto-recommended value: {}".format(perplexity, n, auto_perplexity)
            )

    rng = np.random.RandomState(random_state)

    P = _compute_high_dim_affinities(X, perplexity)
    P *= early_exaggeration

    if init == "pca":
        Y = _pca_init(X, n_components, random_state)
    else:
        Y = rng.randn(n, n_components) * 1e-4

    Y_prev = Y.copy()
    gains = np.ones_like(Y)

    best_loss = float("inf")
    patience_counter = 0
    n_iter_actual = 0
    loss_history = []

    for it in range(n_iter):
        n_iter_actual = it + 1
        Q = _compute_low_dim_affinities(Y)
        grad = _compute_gradient(P, Q, Y)

        gains = (gains + 0.2) * ((grad > 0) != (Y - Y_prev > 0)) + \
                (gains * 0.8) * ((grad > 0) == (Y - Y_prev > 0))
        gains = np.maximum(gains, 0.01)

        current_momentum = momentum if it >= 250 else 0.5
        Y_new = Y - learning_rate * gains * grad + current_momentum * (Y - Y_prev)
        Y_prev = Y.copy()
        Y = Y_new

        if it == early_exaggeration_iters:
            P /= early_exaggeration

        if it >= early_exaggeration_iters:
            Q = _compute_low_dim_affinities(Y)
            loss = np.sum(P * np.log(np.maximum(P / Q, 1e-12)))
            loss_history.append(loss)

            if abs(best_loss - loss) < early_stop_tol:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
            else:
                patience_counter = 0

            if loss < best_loss:
                best_loss = loss

    return {
        "embedding": Y,
        "x": Y[:, 0],
        "y": Y[:, 1],
        "z": Y[:, 2] if n_components == 3 else None,
        "n_iter": n_iter_actual,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "early_stopped": n_iter_actual < n_iter,
    }


if __name__ == "__main__":
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data
    labels = digits.target

    result = tsne(X, n_components=2, n_iter=500, init="pca", random_state=42)
    embedding = result["embedding"]

    print("t-SNE completed:")
    print("  Iterations:", result["n_iter"])
    print("  Early stopped:", result["early_stopped"])
    print("  Final loss:", result["final_loss"])
    print("  Embedding shape:", embedding.shape)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=5, alpha=0.8)
    plt.colorbar(scatter, label="Digit Label")
    plt.title("t-SNE Visualization of Handwritten Digits")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig("tsne_result.png", dpi=150)
    print("Saved t-SNE plot to tsne_result.png")
