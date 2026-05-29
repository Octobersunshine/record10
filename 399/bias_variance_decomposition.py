import warnings

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def true_function(X):
    return np.sin(2 * np.pi * X)


def generate_data(n_samples, noise_std=0.3, random_state=None):
    rng = np.random.RandomState(random_state)
    X = rng.uniform(0, 1, size=n_samples)
    y = true_function(X) + rng.normal(0, noise_std, size=n_samples)
    return X, y


def bias_variance_decomposition(model_factory, n_simulations=500, n_train=100,
                                 noise_std=0.3, n_test_points=200):
    if n_simulations < 200:
        warnings.warn(
            f"n_simulations={n_simulations} < 200: decomposition may be unstable. "
            f"Recommend n_simulations >= 200.",
            UserWarning, stacklevel=2,
        )
    if n_train < 50:
        warnings.warn(
            f"n_train={n_train} < 50: bias estimate may be inflated. "
            f"Recommend n_train >= 50.",
            UserWarning, stacklevel=2,
        )

    rng = np.random.RandomState(42)
    X_test = np.linspace(0, 1, n_test_points).reshape(-1, 1)
    f_true = true_function(X_test.ravel())

    predictions = np.zeros((n_simulations, n_test_points))

    for i in range(n_simulations):
        X_train, y_train = generate_data(n_train, noise_std=noise_std,
                                          random_state=rng.randint(0, 2**31))
        model = model_factory()
        model.fit(X_train.reshape(-1, 1), y_train)
        predictions[i] = model.predict(X_test)

    mean_prediction = predictions.mean(axis=0)

    bias_sq_raw = np.mean((mean_prediction - f_true) ** 2)
    variance = np.mean(np.var(predictions, axis=0, ddof=0))

    mc_correction = variance / n_simulations
    bias_sq = max(0.0, bias_sq_raw - mc_correction)

    irreducible_error = noise_std ** 2
    total_error = bias_sq + variance + irreducible_error

    pointwise_bias_sq = (mean_prediction - f_true) ** 2 - np.var(predictions, axis=0, ddof=0) / n_simulations
    pointwise_bias_sq = np.maximum(pointwise_bias_sq, 0.0)
    pointwise_var = np.var(predictions, axis=0, ddof=0)

    se_bias_sq = np.std(pointwise_bias_sq) / np.sqrt(n_test_points)
    se_variance = np.std(pointwise_var) / np.sqrt(n_test_points)

    return {
        "bias_sq": bias_sq,
        "bias_sq_raw": bias_sq_raw,
        "mc_correction": mc_correction,
        "variance": variance,
        "irreducible_error": irreducible_error,
        "total_error": total_error,
        "se_bias_sq": se_bias_sq,
        "se_variance": se_variance,
        "X_test": X_test.ravel(),
        "f_true": f_true,
        "mean_prediction": mean_prediction,
        "all_predictions": predictions,
    }


def make_linear():
    return LinearRegression()


def make_poly(degree):
    def factory():
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lr", LinearRegression()),
        ])
    return factory


def make_tree(max_depth):
    def factory():
        return DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    return factory


def make_ridge_poly(degree, alpha):
    def factory():
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("ridge", Ridge(alpha=alpha)),
        ])
    return factory


def sweep_complexity(model_family, param_values, n_simulations=500,
                     n_train=100, noise_std=0.3):
    results = []
    for val in param_values:
        factory = model_family(val)
        res = bias_variance_decomposition(factory, n_simulations=n_simulations,
                                           n_train=n_train, noise_std=noise_std)
        results.append({
            "param_value": val,
            "bias_sq": res["bias_sq"],
            "variance": res["variance"],
            "irreducible_error": res["irreducible_error"],
            "total_error": res["total_error"],
            "se_bias_sq": res["se_bias_sq"],
            "se_variance": res["se_variance"],
        })
    return results


def plot_tradeoff_curves(sweep_results_list, save_path="bias_variance_tradeoff.png"):
    fig, axes = plt.subplots(1, len(sweep_results_list), figsize=(7 * len(sweep_results_list), 6))
    if len(sweep_results_list) == 1:
        axes = [axes]

    for ax, (title, xlabel, results) in zip(axes, sweep_results_list):
        params = [r["param_value"] for r in results]
        bias_sq = [r["bias_sq"] for r in results]
        variance = [r["variance"] for r in results]
        noise = [r["irreducible_error"] for r in results]
        total = [r["total_error"] for r in results]
        se_b = [r["se_bias_sq"] for r in results]
        se_v = [r["se_variance"] for r in results]

        ax.fill_between(params,
                        [b - s for b, s in zip(bias_sq, se_b)],
                        [b + s for b, s in zip(bias_sq, se_b)],
                        alpha=0.15, color="#e74c3c")
        ax.fill_between(params,
                        [v - s for v, s in zip(variance, se_v)],
                        [v + s for v, s in zip(variance, se_v)],
                        alpha=0.15, color="#3498db")

        ax.plot(params, bias_sq, "o-", color="#e74c3c", linewidth=2,
                markersize=5, label="Bias²")
        ax.plot(params, variance, "s-", color="#3498db", linewidth=2,
                markersize=5, label="Variance")
        ax.axhline(y=noise[0], color="#2ecc71", linestyle="--", linewidth=1.5,
                    label=f"Irreducible Error (σ²={noise[0]:.3f})")
        ax.plot(params, total, "^-", color="#9b59b6", linewidth=2,
                markersize=5, label="Total Error")

        optimal_idx = np.argmin(total)
        ax.axvline(x=params[optimal_idx], color="gray", linestyle=":",
                    linewidth=1, alpha=0.7)
        ax.annotate(f"Best: {params[optimal_idx]}",
                    xy=(params[optimal_idx], total[optimal_idx]),
                    xytext=(params[optimal_idx], total[optimal_idx] + 0.03),
                    fontsize=9, ha="center", color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Error", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Bias-Variance Tradeoff Curves", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def run_experiment():
    noise_std = 0.3
    n_simulations = 500
    n_train = 100

    configs = [
        ("Linear Regression", make_linear),
        ("Poly-2 Regression", lambda: make_poly(2)()),
        ("Poly-3 Regression", lambda: make_poly(3)()),
        ("Poly-5 Regression", lambda: make_poly(5)()),
        ("Poly-9 Regression", lambda: make_poly(9)()),
        ("Tree depth=1", make_tree(1)),
        ("Tree depth=3", make_tree(3)),
        ("Tree depth=5", make_tree(5)),
        ("Tree depth=10", make_tree(10)),
    ]

    print(f"{'Model':25s} | {'Bias²':>8s} | {'SE(B²)':>8s} | {'Var':>8s} | "
          f"{'SE(Var)':>8s} | {'MC-corr':>8s} | {'σ²':>6s} | {'Total':>8s}")
    print("-" * 110)

    results = []
    for name, factory in configs:
        res = bias_variance_decomposition(factory, n_simulations=n_simulations,
                                           n_train=n_train, noise_std=noise_std)
        results.append((name, res))
        print(f"{name:25s} | {res['bias_sq']:8.4f} | {res['se_bias_sq']:8.4f} | "
              f"{res['variance']:8.4f} | {res['se_variance']:8.4f} | "
              f"{res['mc_correction']:8.5f} | {res['irreducible_error']:6.4f} | "
              f"{res['total_error']:8.4f}")

    print(f"\nMC-corr = Variance / n_simulations (correction applied to remove "
          f"upward bias in Bias² estimate)")
    print(f"n_simulations={n_simulations}, n_train={n_train}, noise_std={noise_std}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    for idx, (name, res) in enumerate(results):
        ax = axes[idx]
        ax.set_title(name, fontsize=11)
        for i in range(0, n_simulations, max(1, n_simulations // 40)):
            ax.plot(res["X_test"], res["all_predictions"][i],
                    color="lightblue", linewidth=0.4, alpha=0.6)
        ax.plot(res["X_test"], res["f_true"], "r-", linewidth=2, label="True f(x)")
        ax.plot(res["X_test"], res["mean_prediction"], "b--", linewidth=2,
                label="E[ŷ]")
        ax.set_ylim(-2.0, 2.0)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.suptitle("Bias-Variance Decomposition: Predictions vs True Function",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("bias_variance_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()

    names = [r[0] for r in results]
    bias_vals = [r[1]["bias_sq"] for r in results]
    var_vals = [r[1]["variance"] for r in results]
    noise_vals = [r[1]["irreducible_error"] for r in results]

    x_pos = np.arange(len(names))
    width = 0.5

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(x_pos, bias_vals, width, label="Bias² (corrected)", color="#e74c3c")
    ax2.bar(x_pos, var_vals, width, bottom=bias_vals,
            label="Variance", color="#3498db")
    bottom2 = [b + v for b, v in zip(bias_vals, var_vals)]
    ax2.bar(x_pos, noise_vals, width, bottom=bottom2,
            label="Irreducible Error (σ²)", color="#2ecc71")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Error")
    ax2.set_title("Bias² (MC-corrected) + Variance + Irreducible Error by Model")
    ax2.legend()
    plt.tight_layout()
    ax2.figure.savefig("bias_variance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nFigures saved: bias_variance_predictions.png, bias_variance_bar.png")

    print("\n" + "=" * 80)
    print("Sweeping model complexity for tradeoff curves...")
    print("=" * 80)

    tree_depths = list(range(1, 16))
    tree_results = sweep_complexity(
        make_tree, tree_depths,
        n_simulations=n_simulations, n_train=n_train, noise_std=noise_std,
    )
    print(f"\n  Decision Tree sweep (depth 1-15) complete.")

    poly_degrees = list(range(1, 13))
    poly_results = sweep_complexity(
        make_poly, poly_degrees,
        n_simulations=n_simulations, n_train=n_train, noise_std=noise_std,
    )
    print(f"  Polynomial Regression sweep (degree 1-12) complete.")

    alphas = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    ridge_family = lambda alpha: make_ridge_poly(9, alpha)
    ridge_results = sweep_complexity(
        ridge_family, alphas,
        n_simulations=n_simulations, n_train=n_train, noise_std=noise_std,
    )
    print(f"  Ridge Poly-9 sweep (alpha {alphas[0]}-{alphas[-1]}) complete.")

    sweep_data = [
        ("Decision Tree", "Max Depth", tree_results),
        ("Polynomial Regression", "Polynomial Degree", poly_results),
        ("Ridge (Poly-9)", "Regularization α (log scale)", ridge_results),
    ]

    tradeoff_path = plot_tradeoff_curves(sweep_data,
                                          save_path="bias_variance_tradeoff.png")
    print(f"\nTradeoff curve saved: {tradeoff_path}")

    for title, xlabel, sresults in sweep_data:
        best = min(sresults, key=lambda r: r["total_error"])
        print(f"\n  {title} optimal: {xlabel}={best['param_value']}  "
              f"Bias²={best['bias_sq']:.4f}  Var={best['variance']:.4f}  "
              f"Total={best['total_error']:.4f}")

    decomposition_data = {
        "model_results": [(name, {k: v for k, v in res.items()
                                   if k not in ("all_predictions",)})
                          for name, res in results],
        "tree_sweep": tree_results,
        "poly_sweep": poly_results,
        "ridge_sweep": ridge_results,
    }

    return decomposition_data


if __name__ == "__main__":
    run_experiment()
