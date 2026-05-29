import numpy as np


def batch_gradient_descent(X, y, lr=0.01, n_iters=1000, tol=1e-8, lr_min=1e-10):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    m, n = X.shape
    W = np.zeros((n, 1))
    loss_curve = []
    lr_log = []

    i = 0
    while i < n_iters:
        y_pred = X @ W
        error = y_pred - y
        loss_old = ((error.T @ error) / (2 * m)).item()
        loss_curve.append(loss_old)
        lr_log.append(lr)

        grad = X.T @ error / m
        W_prev = W.copy()
        W = W - lr * grad

        y_pred_new = X @ W
        error_new = y_pred_new - y
        loss_new = ((error_new.T @ error_new) / (2 * m)).item()

        if loss_new > loss_old:
            lr = max(lr / 2, lr_min)
            W = W_prev
            print(f"[iter {i:4d}] loss increased ({loss_old:.6f} -> {loss_new:.6f}), "
                  f"reverting W, halving lr to {lr:.2e}")
            loss_curve.pop()
            lr_log.pop()
            continue

        if i > 0 and abs(loss_curve[-2] - loss_curve[-1]) < tol:
            i += 1
            break

        i += 1

    return loss_curve, W, lr_log


if __name__ == "__main__":
    np.random.seed(42)
    m, n = 200, 3
    X_true = np.random.randn(m, n)
    true_W = np.array([[3.0], [-1.5], [2.0]])
    y_true = X_true @ true_W + np.random.randn(m, 1) * 0.5

    X_b = np.c_[np.ones(m), X_true]

    print("=== lr=0.1 (normal) ===")
    loss_curve, W_final, lr_log = batch_gradient_descent(X_b, y_true, lr=0.1, n_iters=500)
    print(f"Final weights: {W_final.ravel()}")
    print(f"Initial loss:  {loss_curve[0]:.6f}")
    print(f"Final loss:    {loss_curve[-1]:.6f}")
    print(f"Iterations:    {len(loss_curve)}")
    print(f"Final lr:      {lr_log[-1]:.2e}")

    print("\n=== lr=5.0 (too large, triggers adaptive) ===")
    loss_curve2, W_final2, lr_log2 = batch_gradient_descent(X_b, y_true, lr=5.0, n_iters=500)
    print(f"Final weights: {W_final2.ravel()}")
    print(f"Initial loss:  {loss_curve2[0]:.6f}")
    print(f"Final loss:    {loss_curve2[-1]:.6f}")
    print(f"Iterations:    {len(loss_curve2)}")
    print(f"Final lr:      {lr_log2[-1]:.2e}")
