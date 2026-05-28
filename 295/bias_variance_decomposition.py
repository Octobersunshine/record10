import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def true_function(X):
    return np.sin(X).ravel() + 0.1 * X.ravel()


def generate_independent_dataset(n_samples=800, noise_std=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(0, 4 * np.pi, n_samples).reshape(-1, 1)
    y_true = true_function(X)
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise
    return X, y


def generate_fixed_test_points(n_test=500):
    X_test = np.linspace(0, 4 * np.pi, n_test).reshape(-1, 1)
    y_test_true = true_function(X_test)
    return X_test, y_test_true


def bias_variance_decomposition(model, X_test, y_test_true,
                                 n_datasets=100, n_train_samples=800, noise_std=0.3):
    n_test = len(y_test_true)
    predictions = np.zeros((n_datasets, n_test))

    for i in range(n_datasets):
        X_train, y_train = generate_independent_dataset(n_train_samples, noise_std, seed=i + 1000)
        model.fit(X_train, y_train)
        predictions[i] = model.predict(X_test)

    avg_prediction = np.mean(predictions, axis=0)

    bias_squared = np.mean((avg_prediction - y_test_true) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    irreducible_error = noise_std ** 2
    total_error = bias_squared + variance + irreducible_error

    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'irreducible_error': irreducible_error,
        'total_error': total_error,
        'predictions': predictions,
        'avg_prediction': avg_prediction
    }


def bias_variance_by_complexity(X_test, y_test_true, n_datasets=100,
                                 n_train_samples=800, noise_std=0.3):
    poly_degrees = list(range(1, 16))
    tree_depths = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25]
    rf_n_estimators = [1, 5, 10, 20, 30, 50, 80, 100, 150, 200]

    poly_results = {'complexity': [], 'bias_squared': [], 'variance': [],
                    'irreducible_error': [], 'total_error': []}
    for deg in poly_degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=deg)),
            ('linear', LinearRegression())
        ])
        r = bias_variance_decomposition(model, X_test, y_test_true,
                                         n_datasets, n_train_samples, noise_std)
        poly_results['complexity'].append(deg)
        poly_results['bias_squared'].append(r['bias_squared'])
        poly_results['variance'].append(r['variance'])
        poly_results['irreducible_error'].append(r['irreducible_error'])
        poly_results['total_error'].append(r['total_error'])
        print(f"  Poly deg={deg}: Bias²={r['bias_squared']:.4f}  Var={r['variance']:.4f}  Total={r['total_error']:.4f}")

    tree_results = {'complexity': [], 'bias_squared': [], 'variance': [],
                    'irreducible_error': [], 'total_error': []}
    for depth in tree_depths:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        r = bias_variance_decomposition(model, X_test, y_test_true,
                                         n_datasets, n_train_samples, noise_std)
        tree_results['complexity'].append(depth)
        tree_results['bias_squared'].append(r['bias_squared'])
        tree_results['variance'].append(r['variance'])
        tree_results['irreducible_error'].append(r['irreducible_error'])
        tree_results['total_error'].append(r['total_error'])
        print(f"  Tree depth={depth}: Bias²={r['bias_squared']:.4f}  Var={r['variance']:.4f}  Total={r['total_error']:.4f}")

    rf_results = {'complexity': [], 'bias_squared': [], 'variance': [],
                  'irreducible_error': [], 'total_error': []}
    for n_est in rf_n_estimators:
        model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
        r = bias_variance_decomposition(model, X_test, y_test_true,
                                         n_datasets, n_train_samples, noise_std)
        rf_results['complexity'].append(n_est)
        rf_results['bias_squared'].append(r['bias_squared'])
        rf_results['variance'].append(r['variance'])
        rf_results['irreducible_error'].append(r['irreducible_error'])
        rf_results['total_error'].append(r['total_error'])
        print(f"  RF n_est={n_est}: Bias²={r['bias_squared']:.4f}  Var={r['variance']:.4f}  Total={r['total_error']:.4f}")

    return poly_results, tree_results, rf_results


def expected_error_summary(results_dict):
    rows = []
    for name, r in results_dict.items():
        total = r['total_error']
        rows.append({
            'model': name,
            'bias_squared': r['bias_squared'],
            'variance': r['variance'],
            'irreducible_error': r['irreducible_error'],
            'total_error': total,
            'bias_pct': r['bias_squared'] / total * 100,
            'var_pct': r['variance'] / total * 100,
            'irred_pct': r['irreducible_error'] / total * 100,
        })
    return rows


def plot_u_shaped_curve(ax, complexity, bias_sq, variance, irred, total, x_label, title):
    ax.fill_between(complexity, 0, bias_sq, alpha=0.5, color='#e74c3c', label='Bias²')
    ax.fill_between(complexity, bias_sq, np.array(bias_sq) + np.array(variance),
                    alpha=0.5, color='#3498db', label='Variance')
    ax.fill_between(complexity, np.array(bias_sq) + np.array(variance), total,
                    alpha=0.5, color='#95a5a6', label='Irreducible Error')
    ax.plot(complexity, total, 'k-', linewidth=2.5, label='Total Error')
    ax.plot(complexity, bias_sq, '--', color='#e74c3c', linewidth=1.5)
    ax.plot(complexity, variance, '--', color='#3498db', linewidth=1.5)
    ax.axhline(y=irred, color='gray', linestyle=':', linewidth=1.2)

    min_idx = np.argmin(total)
    ax.annotate(f'Optimal\n({complexity[min_idx]}, {total[min_idx]:.3f})',
                xy=(complexity[min_idx], total[min_idx]),
                xytext=(complexity[min_idx] + (complexity[-1] - complexity[0]) * 0.1,
                        total[min_idx] + (max(total) - min(total)) * 0.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Expected Error', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)


def main():
    np.random.seed(42)

    X_test, y_test_true = generate_fixed_test_points(n_test=500)
    n_datasets = 100
    noise_std = 0.3

    models = {
        'Linear Regression': LinearRegression(),
        'Polynomial Regression (deg=3)': Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ]),
        'Decision Tree (depth=3)': DecisionTreeRegressor(max_depth=3, random_state=42),
        'Decision Tree (depth=10)': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest (n=50)': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        results[name] = bias_variance_decomposition(
            model, X_test, y_test_true,
            n_datasets=n_datasets, n_train_samples=800, noise_std=noise_std
        )
        print(f"  Bias²={results[name]['bias_squared']:.4f}  Var={results[name]['variance']:.4f}"
              f"  Irred={results[name]['irreducible_error']:.4f}  Total={results[name]['total_error']:.4f}")

    print("\n" + "=" * 80)
    print("Scanning model complexity for trade-off curves...")
    print("=" * 80)

    print("\n[1/3] Polynomial Regression complexity scan:")
    poly_results, tree_results, rf_results = bias_variance_by_complexity(
        X_test, y_test_true, n_datasets=n_datasets,
        n_train_samples=800, noise_std=noise_std
    )

    error_summary = expected_error_summary(results)

    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ---- Row 1: Model comparison ----

    ax1 = fig.add_subplot(gs[0, 0])
    model_names = list(models.keys())
    bias_vals = [results[n]['bias_squared'] for n in model_names]
    var_vals = [results[n]['variance'] for n in model_names]
    irred_vals = [results[n]['irreducible_error'] for n in model_names]
    x_pos = np.arange(len(model_names))
    w = 0.5
    ax1.bar(x_pos, bias_vals, w, label='Bias²', color='#e74c3c', alpha=0.8)
    ax1.bar(x_pos, var_vals, w, bottom=bias_vals, label='Variance', color='#3498db', alpha=0.8)
    ax1.bar(x_pos, irred_vals, w,
            bottom=np.array(bias_vals) + np.array(var_vals),
            label='Irreducible', color='#95a5a6', alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=35, ha='right', fontsize=8)
    ax1.set_ylabel('Error')
    ax1.set_title('① Error Decomposition by Model', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.2)

    ax2 = fig.add_subplot(gs[0, 1])
    for row in error_summary:
        ax2.scatter(row['bias_pct'], row['var_pct'], s=row['total_error'] * 800,
                    alpha=0.7, edgecolors='black', linewidths=0.8, zorder=3)
        ax2.annotate(row['model'], (row['bias_pct'], row['var_pct']),
                     fontsize=7, ha='center', va='bottom',
                     xytext=(0, 8), textcoords='offset points')
    ax2.set_xlabel('Bias² (%)', fontsize=10)
    ax2.set_ylabel('Variance (%)', fontsize=10)
    ax2.set_title('② Error Composition (bubble size = Total Error)', fontweight='bold')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)

    ax3 = fig.add_subplot(gs[0, 2])
    categories = ['Bias²', 'Variance', 'Irreducible']
    bar_width = 0.15
    colors_bar = ['#e74c3c', '#3498db', '#95a5a6']
    for i, row in enumerate(error_summary):
        vals = [row['bias_pct'], row['var_pct'], row['irred_pct']]
        positions = np.arange(3) + i * bar_width
        bars = ax3.bar(positions, vals, bar_width, color=colors_bar, alpha=0.6 + i * 0.08)
        for bar, v in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{v:.0f}%', ha='center', va='bottom', fontsize=6)
    ax3.set_xticks(np.arange(3) + bar_width * (len(error_summary) - 1) / 2)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('③ Error Component Proportions', fontweight='bold')
    ax3.grid(axis='y', alpha=0.2)

    # ---- Row 2: U-shaped trade-off curves ----

    ax4 = fig.add_subplot(gs[1, 0])
    plot_u_shaped_curve(ax4,
                        poly_results['complexity'],
                        poly_results['bias_squared'],
                        poly_results['variance'],
                        poly_results['irreducible_error'][0],
                        poly_results['total_error'],
                        'Polynomial Degree', '④ Polynomial: Bias-Variance Trade-off')

    ax5 = fig.add_subplot(gs[1, 1])
    plot_u_shaped_curve(ax5,
                        tree_results['complexity'],
                        tree_results['bias_squared'],
                        tree_results['variance'],
                        tree_results['irreducible_error'][0],
                        tree_results['total_error'],
                        'Tree Max Depth', '⑤ Decision Tree: Bias-Variance Trade-off')

    ax6 = fig.add_subplot(gs[1, 2])
    plot_u_shaped_curve(ax6,
                        rf_results['complexity'],
                        rf_results['bias_squared'],
                        rf_results['variance'],
                        rf_results['irreducible_error'][0],
                        rf_results['total_error'],
                        'Number of Estimators', '⑥ Random Forest: Bias-Variance Trade-off')

    # ---- Row 3: Classic diagram + predictions + summary table ----

    ax7 = fig.add_subplot(gs[2, 0])
    complexity_range = np.linspace(0, 1, 200)
    bias_curve = 0.6 * (1 - complexity_range) ** 2
    var_curve = 0.05 + 0.7 * complexity_range ** 3
    irred_line = np.full_like(complexity_range, 0.09)
    total_curve = bias_curve + var_curve + irred_line
    ax7.fill_between(complexity_range, 0, bias_curve, alpha=0.3, color='#e74c3c')
    ax7.fill_between(complexity_range, bias_curve, bias_curve + var_curve, alpha=0.3, color='#3498db')
    ax7.fill_between(complexity_range, bias_curve + var_curve, total_curve, alpha=0.3, color='#95a5a6')
    ax7.plot(complexity_range, bias_curve, '-', color='#e74c3c', linewidth=2.5, label='Bias² (decreasing)')
    ax7.plot(complexity_range, var_curve, '-', color='#3498db', linewidth=2.5, label='Variance (increasing)')
    ax7.plot(complexity_range, total_curve, '-', color='#2ecc71', linewidth=3, label='Total Error (U-shaped)')
    ax7.axhline(y=0.09, color='gray', linestyle=':', linewidth=1.5, label='Irreducible Error')
    min_idx = np.argmin(total_curve)
    ax7.axvline(x=complexity_range[min_idx], color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    ax7.annotate('Sweet Spot\n(Best Complexity)',
                 xy=(complexity_range[min_idx], total_curve[min_idx]),
                 xytext=(complexity_range[min_idx] + 0.12, total_curve[min_idx] + 0.15),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
    ax7.text(0.05, 0.55, 'Underfitting\n(High Bias)', fontsize=10, color='#e74c3c',
             fontweight='bold', alpha=0.7)
    ax7.text(0.75, 0.55, 'Overfitting\n(High Variance)', fontsize=10, color='#3498db',
             fontweight='bold', alpha=0.7)
    ax7.set_xlabel('Model Complexity →', fontsize=11)
    ax7.set_ylabel('Error', fontsize=11)
    ax7.set_title('⑦ Classic Bias-Variance Trade-off Diagram', fontweight='bold', fontsize=12)
    ax7.legend(fontsize=8, loc='center right')
    ax7.grid(True, alpha=0.2)
    ax7.set_ylim(bottom=0)

    ax8 = fig.add_subplot(gs[2, 1])
    sort_idx = np.argsort(X_test.ravel())
    X_sorted = X_test[sort_idx]
    y_true_sorted = y_test_true[sort_idx]
    ax8.plot(X_sorted, y_true_sorted, 'k--', label='True function', linewidth=2.5)
    colors_pred = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
    for i, (name, color) in enumerate(zip(model_names, colors_pred)):
        avg_pred_sorted = results[name]['avg_prediction'][sort_idx]
        ax8.plot(X_sorted, avg_pred_sorted, color=color, label=name, linewidth=1.5)
    X_sample, y_sample = generate_independent_dataset(200, noise_std, seed=42)
    ax8.scatter(X_sample[:80], y_sample[:80], alpha=0.2, s=15, color='gray', label='Sample data')
    ax8.set_xlabel('X')
    ax8.set_ylabel('y')
    ax8.set_title('⑧ Model Predictions vs True Function', fontweight='bold')
    ax8.legend(fontsize=7, loc='upper left')
    ax8.grid(True, alpha=0.2)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    table_data = []
    col_labels = ['Model', 'Bias²', 'Variance', 'Irred.', 'Total', 'B%', 'V%', 'I%']
    for row in error_summary:
        table_data.append([
            row['model'],
            f"{row['bias_squared']:.4f}",
            f"{row['variance']:.4f}",
            f"{row['irreducible_error']:.4f}",
            f"{row['total_error']:.4f}",
            f"{row['bias_pct']:.1f}%",
            f"{row['var_pct']:.1f}%",
            f"{row['irred_pct']:.1f}%",
        ])
    table = ax9.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == 0:
            cell.set_text_props(fontsize=6.5)
        else:
            if r > 0:
                try:
                    val = float(cell.get_text().get_text().replace('%', ''))
                    if c in (1, 5):
                        cell.set_facecolor('#fadbd8')
                    elif c in (2, 6):
                        cell.set_facecolor('#d6eaf8')
                    elif c in (3, 7):
                        cell.set_facecolor('#eaecee')
                except ValueError:
                    pass
    ax9.set_title('⑨ Expected Error Decomposition Summary', fontweight='bold', fontsize=12, pad=20)

    fig.suptitle('Bias-Variance Decomposition: Model Complexity & Error Analysis\n'
                 f'(Method: {n_datasets} independent datasets, noise_std={noise_std}, '
                 f'known true function)',
                 fontsize=15, fontweight='bold', y=0.99)

    plt.savefig('bias_variance_decomposition.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'bias_variance_decomposition.png'")

    print("\n" + "=" * 85)
    print("EXPECTED ERROR DECOMPOSITION SUMMARY")
    print("=" * 85)
    print(f"{'Model':<35} {'Bias²':>8} {'Variance':>10} {'Irred.':>8} {'Total':>8}"
          f"  {'B%':>6} {'V%':>6} {'I%':>6}")
    print("-" * 85)
    for row in error_summary:
        print(f"{row['model']:<35} {row['bias_squared']:>8.4f} {row['variance']:>10.4f}"
              f" {row['irreducible_error']:>8.4f} {row['total_error']:>8.4f}"
              f"  {row['bias_pct']:>5.1f}% {row['var_pct']:>5.1f}% {row['irred_pct']:>5.1f}%")
    print("=" * 85)

    print("\nOptimal Complexity Analysis:")
    poly_min = np.argmin(poly_results['total_error'])
    tree_min = np.argmin(tree_results['total_error'])
    rf_min = np.argmin(rf_results['total_error'])
    print(f"  Polynomial Regression: optimal degree = {poly_results['complexity'][poly_min]}"
          f", min total error = {poly_results['total_error'][poly_min]:.4f}")
    print(f"  Decision Tree:         optimal depth  = {tree_results['complexity'][tree_min]}"
          f", min total error = {tree_results['total_error'][tree_min]:.4f}")
    print(f"  Random Forest:         optimal n_est  = {rf_results['complexity'][rf_min]}"
          f", min total error = {rf_results['total_error'][rf_min]:.4f}")


if __name__ == '__main__':
    main()
