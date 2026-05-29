import numpy as np


def compute_loss(X, y, W):
    m = X.shape[0]
    y_pred = X @ W
    error = y_pred - y
    return (error.T @ error).item() / (2 * m)


def compute_gradient(X, y, W):
    m = X.shape[0]
    y_pred = X @ W
    error = y_pred - y
    return X.T @ error / m


def train_linear_regression(X, y, optimizer="sgd", lr=0.01, n_epochs=100, batch_size=None,
                            beta=0.9, beta1=0.9, beta2=0.999, eps=1e-8,
                            tol=1e-8, lr_min=1e-10, adaptive_lr=True, verbose=False):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    m, n = X.shape
    W = np.zeros((n, 1))

    if batch_size is None:
        batch_size = m
    batch_size = min(batch_size, m)

    v = np.zeros_like(W)
    m_t = np.zeros_like(W)
    v_t = np.zeros_like(W)
    t = 0

    loss_curve = []
    lr_log = []
    epoch = 0
    current_lr = lr

    indices = np.arange(m)

    while epoch < n_epochs:
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, m, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            loss_old = compute_loss(X, y, W)

            grad = compute_gradient(X_batch, y_batch, W)

            W_prev = W.copy()

            if optimizer == "sgd":
                W = W - current_lr * grad
            elif optimizer == "momentum":
                v = beta * v + (1 - beta) * grad
                W = W - current_lr * v
            elif optimizer == "adam":
                t += 1
                m_t = beta1 * m_t + (1 - beta1) * grad
                v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)
                m_hat = m_t / (1 - beta1 ** t)
                v_hat = v_t / (1 - beta2 ** t)
                W = W - current_lr * m_hat / (np.sqrt(v_hat) + eps)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

            loss_new = compute_loss(X, y, W)

            if adaptive_lr and loss_new > loss_old:
                current_lr = max(current_lr / 2, lr_min)
                W = W_prev
                if verbose:
                    print(f"[epoch {epoch:3d}] loss increased ({loss_old:.6f} -> {loss_new:.6f}), "
                          f"halving lr to {current_lr:.2e}")
                v = np.zeros_like(W)
                continue

            epoch_loss += loss_new
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else loss_old
        loss_curve.append(avg_loss)
        lr_log.append(current_lr)

        if epoch > 0 and abs(loss_curve[-2] - loss_curve[-1]) < tol:
            if verbose:
                print(f"Converged at epoch {epoch}")
            break

        epoch += 1

    return loss_curve, W, lr_log, epoch + 1


if __name__ == "__main__":
    np.random.seed(42)
    m, n = 200, 3
    X_true = np.random.randn(m, n)
    true_W = np.array([[3.0], [-1.5], [2.0]])
    y_true = X_true @ true_W + np.random.randn(m, 1) * 0.5

    X_b = np.c_[np.ones((m, 1)), X_true]

    optimizers = [
        ("BGD", "sgd", m, 0.05, True),
        ("SGD", "sgd", 1, 0.01, False),
        ("Mini-batch SGD", "sgd", 32, 0.01, True),
        ("Momentum", "momentum", 32, 0.01, True),
        ("Adam", "adam", 32, 0.01, True),
    ]

    print("=" * 70)
    print(f"Optimizer Comparison (m={m}, n={n}), true W: {true_W.ravel()}")
    print("=" * 70)

    results = {}
    for name, opt, bs, lr, adapt in optimizers:
        print(f"\n--- {name} ---")
        loss_curve, W_final, lr_log, epochs = train_linear_regression(
            X_b, y_true, optimizer=opt, lr=lr, n_epochs=200, batch_size=bs,
            adaptive_lr=adapt, verbose=False
        )
        results[name] = {
            "loss": loss_curve,
            "weights": W_final,
            "epochs": epochs,
            "final_lr": lr_log[-1],
        }
        print(f"  Epochs:       {epochs}")
        print(f"  Final loss:   {loss_curve[-1]:.6f}")
        print(f"  Initial loss: {loss_curve[0]:.6f}")
        print(f"  Final lr:     {lr_log[-1]:.2e}")
        print(f"  Weights:      {W_final.ravel()}")

    print("\n" + "=" * 70)
    print("Convergence Comparison")
    print("=" * 70)
    for name, res in results.items():
        print(f"{name:20s}: loss = {res['loss'][-1]:.6f} (epochs={res['epochs']})")
