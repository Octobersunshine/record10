import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma, kv


def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)


def matern_kernel(X1, X2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    dist = np.sqrt(np.maximum(sqdist, 1e-12))
    d = np.sqrt(2 * nu) * dist / length_scale

    if nu == 0.5:
        K = np.exp(-d)
    elif nu == 1.5:
        K = (1.0 + d) * np.exp(-d)
    elif nu == 2.5:
        K = (1.0 + d + d**2 / 3.0) * np.exp(-d)
    else:
        K = (2**(1 - nu) / gamma(nu)) * d**nu * kv(nu, d)
        K[dist == 0] = 1.0

    return sigma_f**2 * K


def rational_quadratic_kernel(X1, X2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    alpha_clamped = max(alpha, 1e-3)
    denom = 2 * alpha_clamped * length_scale**2
    factor = 1.0 + sqdist / denom
    factor = np.maximum(factor, 1e-12)
    exponent = np.clip(-alpha_clamped, -100, 100)
    return sigma_f**2 * factor ** exponent


KERNEL_FUNCTIONS = {
    "RBF": rbf_kernel,
    "Matern": matern_kernel,
    "RationalQuadratic": rational_quadratic_kernel,
}


def _stable_cholesky(K, max_tries=8, initial_jitter=1e-6, jitter_scale=10.0):
    jitter = initial_jitter
    for i in range(max_tries):
        try:
            L = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
            return L, jitter
        except np.linalg.LinAlgError:
            jitter *= jitter_scale
    raise np.linalg.LinAlgError(
        f"Cholesky decomposition failed after {max_tries} attempts. "
        f"Final jitter: {jitter:.2e}"
    )


def _log_marginal_likelihood(params, X, y, kernel_name, jitter=1e-6):
    kernel_func = KERNEL_FUNCTIONS[kernel_name]
    kernel_alpha = None
    nu = None

    if kernel_name == "RBF":
        length_scale, sigma_f, sigma_n = np.exp(params)
        K = kernel_func(X, X, length_scale=length_scale, sigma_f=sigma_f)
    elif kernel_name == "Matern":
        length_scale, sigma_f, sigma_n, nu = np.exp(params[0]), np.exp(params[1]), np.exp(params[2]), params[3]
        K = kernel_func(X, X, length_scale=length_scale, sigma_f=sigma_f, nu=nu)
    elif kernel_name == "RationalQuadratic":
        length_scale, sigma_f, sigma_n, kernel_alpha = np.exp(params[0]), np.exp(params[1]), np.exp(params[2]), np.exp(params[3])
        kernel_alpha = max(float(kernel_alpha), 1e-3)
        K = kernel_func(X, X, length_scale=length_scale, sigma_f=sigma_f, alpha=kernel_alpha)

    K[np.diag_indices_from(K)] += sigma_n**2

    try:
        L, _ = _stable_cholesky(K, initial_jitter=jitter)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros_like(params)

    n = X.shape[0]
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    log_lml = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)

    K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
    Q = K_inv - alpha[:, None] @ alpha[None, :]

    grad = np.zeros_like(params)

    if kernel_name == "RBF":
        dK_dl = K * (sqdist_matrix(X) / length_scale**3)
        dK_df = 2 * K / sigma_f
        dK_dn = 2 * sigma_n * np.eye(n)

        grad[0] = 0.5 * np.trace(Q @ dK_dl)
        grad[1] = 0.5 * np.trace(Q @ dK_df)
        grad[2] = 0.5 * np.trace(Q @ dK_dn)

    elif kernel_name == "Matern":
        dK_dl, dK_df = _matern_gradients(X, length_scale, sigma_f, nu)
        dK_dn = 2 * sigma_n * np.eye(n)

        grad[0] = 0.5 * np.trace(Q @ dK_dl) * np.exp(params[0])
        grad[1] = 0.5 * np.trace(Q @ dK_df) * np.exp(params[1])
        grad[2] = 0.5 * np.trace(Q @ dK_dn) * np.exp(params[2])
        grad[3] = 0.0

    elif kernel_name == "RationalQuadratic":
        dK_dl, dK_df, dK_dalpha = _rq_gradients(X, length_scale, sigma_f, kernel_alpha)
        dK_dn = 2 * sigma_n * np.eye(n)

        grad[0] = 0.5 * np.trace(Q @ dK_dl) * np.exp(params[0])
        grad[1] = 0.5 * np.trace(Q @ dK_df) * np.exp(params[1])
        grad[2] = 0.5 * np.trace(Q @ dK_dn) * np.exp(params[2])
        grad[3] = 0.5 * np.trace(Q @ dK_dalpha) * np.exp(params[3])

    return -log_lml, grad


def sqdist_matrix(X):
    return np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * X @ X.T


def _matern_gradients(X, length_scale, sigma_f, nu):
    sqdist = sqdist_matrix(X)
    dist = np.sqrt(np.maximum(sqdist, 1e-12))
    d = np.sqrt(2 * nu) * dist / length_scale

    if nu == 1.5:
        K = (1.0 + d) * np.exp(-d)
        dK_dl = sigma_f**2 * d * (1.0 + d) * np.exp(-d) / length_scale
    elif nu == 2.5:
        K = (1.0 + d + d**2 / 3.0) * np.exp(-d)
        dK_dl = sigma_f**2 * d**2 * (1.0 + d) * np.exp(-d) / (3.0 * length_scale)
    elif nu == 0.5:
        K = np.exp(-d)
        dK_dl = sigma_f**2 * d * np.exp(-d) / length_scale
    else:
        return np.zeros_like(sqdist), 2.0 * sigma_f * np.ones_like(sqdist)

    dK_df = 2.0 * sigma_f * K
    return dK_dl, dK_df


def _rq_gradients(X, length_scale, sigma_f, alpha):
    sqdist = sqdist_matrix(X)
    alpha_clamped = max(float(alpha), 1e-3)
    denom = 2 * alpha_clamped * length_scale**2
    factor = 1.0 + sqdist / denom
    factor = np.maximum(factor, 1e-12)
    log_factor = np.log(factor)

    exponent = np.clip(-alpha_clamped, -100, 100)
    exponent2 = np.clip(-alpha_clamped - 1, -100, 100)
    factor_pow = factor ** exponent2

    dK_dl = sigma_f**2 * alpha_clamped * factor_pow * sqdist / (alpha_clamped * length_scale**3)
    dK_df = 2.0 * sigma_f * factor ** exponent
    dK_dalpha = sigma_f**2 * factor ** exponent * (
        log_factor - sqdist / denom / factor
    )
    return dK_dl, dK_df, dK_dalpha


def optimize_hyperparameters(X_train, y_train, kernel_name, jitter=1e-6, n_restarts=5):
    kernel_func = KERNEL_FUNCTIONS[kernel_name]

    if kernel_name == "RBF":
        n_params = 3
        bounds = [(-5, 5), (-5, 5), (-8, 2)]
    elif kernel_name == "Matern":
        n_params = 4
        bounds = [(-5, 5), (-5, 5), (-8, 2), (0.5, 2.5)]
    elif kernel_name == "RationalQuadratic":
        n_params = 4
        bounds = [(-5, 5), (-5, 5), (-8, 2), (-5, 5)]

    best_lml = -np.inf
    best_params = None

    for restart in range(n_restarts):
        if kernel_name == "RBF":
            x0 = np.array([np.log(1.0), np.log(1.0), np.log(0.1)])
        elif kernel_name == "Matern":
            x0 = np.array([np.log(1.0), np.log(1.0), np.log(0.1), 1.5])
        elif kernel_name == "RationalQuadratic":
            x0 = np.array([np.log(1.0), np.log(1.0), np.log(0.1), np.log(1.0)])

        if restart > 0:
            x0[:3] += np.random.randn(3) * 0.5
            if len(x0) > 3:
                if kernel_name == "Matern":
                    x0[3] = np.random.choice([0.5, 1.5, 2.5])
                else:
                    x0[3] += np.random.randn() * 0.5

        result = minimize(
            fun=_log_marginal_likelihood,
            x0=x0,
            args=(X_train, y_train, kernel_name, jitter),
            method="L-BFGS-B",
            bounds=bounds,
            jac=True,
            options={"maxiter": 500},
        )

        if not result.success:
            continue

        current_lml = -result.fun
        if current_lml > best_lml:
            best_lml = current_lml
            best_params = result.x

    if best_params is None:
        best_params = x0
        best_lml = -_log_marginal_likelihood(x0, X_train, y_train, kernel_name, jitter)[0]

    param_dict = {}
    if kernel_name == "RBF":
        param_dict = {
            "length_scale": np.exp(best_params[0]),
            "sigma_f": np.exp(best_params[1]),
            "sigma_n": np.exp(best_params[2]),
        }
    elif kernel_name == "Matern":
        param_dict = {
            "length_scale": np.exp(best_params[0]),
            "sigma_f": np.exp(best_params[1]),
            "sigma_n": np.exp(best_params[2]),
            "nu": best_params[3],
        }
    elif kernel_name == "RationalQuadratic":
        param_dict = {
            "length_scale": np.exp(best_params[0]),
            "sigma_f": np.exp(best_params[1]),
            "sigma_n": np.exp(best_params[2]),
            "alpha": np.exp(best_params[3]),
        }

    return param_dict, best_lml


def auto_select_kernel(X_train, y_train, jitter=1e-6, n_restarts=5):
    kernels = ["RBF", "Matern", "RationalQuadratic"]
    results = {}

    for kernel_name in kernels:
        params, lml = optimize_hyperparameters(X_train, y_train, kernel_name, jitter, n_restarts)
        results[kernel_name] = {"params": params, "lml": lml}
        print(f"  {kernel_name}: LML = {lml:.4f}, params = {params}")

    best_kernel = max(results.keys(), key=lambda k: results[k]["lml"])
    return best_kernel, results[best_kernel]["params"], results


def gaussian_process_regression(X_train, y_train, X_test, kernel_name="auto",
                                length_scale=1.0, sigma_f=1.0, sigma_n=1e-8,
                                nu=1.5, alpha=1.0,
                                optimize=True, jitter=1e-6, n_restarts=5):
    if optimize or kernel_name == "auto":
        if kernel_name == "auto":
            print("自动选择最优核函数...")
            kernel_name, best_params, all_results = auto_select_kernel(
                X_train, y_train, jitter, n_restarts
            )
            length_scale = best_params["length_scale"]
            sigma_f = best_params["sigma_f"]
            sigma_n = best_params["sigma_n"]
            if "nu" in best_params:
                nu = best_params["nu"]
            if "alpha" in best_params:
                alpha = best_params["alpha"]
        else:
            print(f"优化 {kernel_name} 核的超参数...")
            best_params, best_lml = optimize_hyperparameters(
                X_train, y_train, kernel_name, jitter, n_restarts
            )
            length_scale = best_params["length_scale"]
            sigma_f = best_params["sigma_f"]
            sigma_n = best_params["sigma_n"]
            if "nu" in best_params:
                nu = best_params["nu"]
            if "alpha" in best_params:
                alpha = best_params["alpha"]
            all_results = None
            print(f"最优 LML = {best_lml:.4f}, params = {best_params}")

    kernel_func = KERNEL_FUNCTIONS[kernel_name]

    if kernel_name == "RBF":
        K = kernel_func(X_train, X_train, length_scale=length_scale, sigma_f=sigma_f)
        K_s = kernel_func(X_train, X_test, length_scale=length_scale, sigma_f=sigma_f)
        K_ss = kernel_func(X_test, X_test, length_scale=length_scale, sigma_f=sigma_f)
    elif kernel_name == "Matern":
        K = kernel_func(X_train, X_train, length_scale=length_scale, sigma_f=sigma_f, nu=nu)
        K_s = kernel_func(X_train, X_test, length_scale=length_scale, sigma_f=sigma_f, nu=nu)
        K_ss = kernel_func(X_test, X_test, length_scale=length_scale, sigma_f=sigma_f, nu=nu)
    elif kernel_name == "RationalQuadratic":
        K = kernel_func(X_train, X_train, length_scale=length_scale, sigma_f=sigma_f, alpha=alpha)
        K_s = kernel_func(X_train, X_test, length_scale=length_scale, sigma_f=sigma_f, alpha=alpha)
        K_ss = kernel_func(X_test, X_test, length_scale=length_scale, sigma_f=sigma_f, alpha=alpha)

    K[np.diag_indices_from(K)] += sigma_n**2
    K_ss[np.diag_indices_from(K_ss)] += sigma_n**2

    L, applied_jitter = _stable_cholesky(K, initial_jitter=jitter)
    if applied_jitter > jitter:
        print(f"Warning: jitter increased to {applied_jitter:.2e} for numerical stability")

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = K_s.T @ alpha

    v = np.linalg.solve(L, K_s)
    var = np.diag(K_ss) - np.sum(v**2, axis=0)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    z_score = 1.96
    lower = mu - z_score * std
    upper = mu + z_score * std

    optimized_params = {
        "kernel_name": kernel_name,
        "length_scale": length_scale,
        "sigma_f": sigma_f,
        "sigma_n": sigma_n,
    }
    if kernel_name == "Matern":
        optimized_params["nu"] = nu
    elif kernel_name == "RationalQuadratic":
        optimized_params["alpha"] = alpha

    return mu, lower, upper, std, optimized_params


if __name__ == "__main__":
    np.random.seed(42)

    X_train = np.linspace(-5, 5, 20).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + 0.1 * np.random.randn(X_train.shape[0])

    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)

    print("=" * 60)
    print("=== 场景1：自动选择核函数 + 超参数优化 ===")
    print("=" * 60)
    mu, lower, upper, std, params = gaussian_process_regression(
        X_train, y_train, X_test,
        kernel_name="auto",
        optimize=True,
        n_restarts=3
    )
    print(f"\n最优核函数: {params['kernel_name']}")
    print(f"优化参数: {params}")
    print("测试点预测结果（前5个）:")
    for i in range(5):
        print(f"  x = {X_test[i, 0]:.3f}, 均值 = {mu[i]:.4f}, "
              f"95% CI = [{lower[i]:.4f}, {upper[i]:.4f}]")

    print("\n" + "=" * 60)
    print("=== 场景2：指定 Matern 核 + 超参数优化 ===")
    print("=" * 60)
    mu2, lower2, upper2, std2, params2 = gaussian_process_regression(
        X_train, y_train, X_test,
        kernel_name="Matern",
        optimize=True,
        n_restarts=3
    )
    print(f"\n优化参数: {params2}")
    print(f"预测均值（前5个）: {np.round(mu2[:5], 4)}")

    print("\n" + "=" * 60)
    print("=== 场景3：指定 RBF 核 + 手动参数（不优化） ===")
    print("=" * 60)
    mu3, lower3, upper3, std3, params3 = gaussian_process_regression(
        X_train, y_train, X_test,
        kernel_name="RBF",
        length_scale=1.0, sigma_f=1.0, sigma_n=0.1,
        optimize=False
    )
    print(f"使用参数: {params3}")
    print(f"预测均值（前5个）: {np.round(mu3[:5], 4)}")

    print("\n" + "=" * 60)
    print("=== 场景4：极端场景（近距离重复点）鲁棒性测试 ===")
    print("=" * 60)
    X_dup = np.vstack([X_train, X_train[-3:] + 1e-10])
    y_dup = np.concatenate([y_train, y_train[-3:]])
    mu4, lower4, upper4, std4, params4 = gaussian_process_regression(
        X_dup, y_dup, X_test,
        kernel_name="RBF",
        optimize=True,
        sigma_n=1e-10,
        jitter=1e-6,
        n_restarts=3
    )
    print(f"\n优化参数: {params4}")
    print(f"条件数: {np.linalg.cond(rbf_kernel(X_dup, X_dup, params4['length_scale'], params4['sigma_f'])):.2e}")
    print(f"方差非负检查: {np.all(std4 >= 0)}")
    print(f"预测均值（前5个）: {np.round(mu4[:5], 4)}")
