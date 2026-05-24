import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt


def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def variational_equation(t, extended_state, sigma, rho, beta):
    x, y, z = extended_state[:3]
    W = extended_state[3:].reshape(3, 3)

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    J = np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])

    dW = J @ W

    return np.concatenate([[dx, dy, dz], dW.flatten()])


def householder_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for k in range(n):
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = 1

        v = x + np.sign(x[0]) * np.linalg.norm(x) * e
        v = v / np.linalg.norm(v)

        H = np.eye(m - k) - 2 * np.outer(v, v)
        R[k:, :] = H @ R[k:, :]
        Q[:, k:] = Q[:, k:] @ H

    return Q, R


def reorthogonalize_householder(W):
    eps = 1e-15
    Q, R = householder_qr(W)

    diag_R = np.diag(R)
    signs = np.sign(diag_R)
    signs[signs == 0] = 1

    Q = Q * signs
    R = R * signs[:, np.newaxis]

    diag_R = np.abs(np.diag(R))
    diag_R = np.maximum(diag_R, eps)

    orth_error = np.max(np.abs(Q.T @ Q - np.eye(3)))
    if orth_error > 1e-8:
        Q, _ = householder_qr(Q)

    return Q, diag_R


def check_orthogonality(Q, step_num, verbose=False):
    orth_error = np.max(np.abs(Q.T @ Q - np.eye(Q.shape[0])))
    if verbose and orth_error > 1e-6:
        print(f"Warning (step {step_num}): Orthogonality error = {orth_error:.2e}")
    return orth_error


def benettin_lyapunov(initial_state, sigma, rho, beta, t_total, dt, renorm_interval=0.1):
    n_steps = int(t_total / dt)
    n_renorm = max(1, int(renorm_interval / dt)) if renorm_interval >= dt else 1

    state = np.array(initial_state, dtype=np.float64)
    W = np.eye(3, dtype=np.float64)

    lambda_sum = np.zeros(3, dtype=np.float64)
    lambda_history = []
    t_history = []
    orth_errors = []

    renorm_count = 0

    for step in range(n_steps):
        t_start = step * dt
        t_end = (step + 1) * dt

        extended_state = np.concatenate([state, W.flatten()])

        sol = solve_ivp(
            variational_equation,
            [t_start, t_end],
            extended_state,
            args=(sigma, rho, beta),
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            t_eval=[t_end]
        )

        state = sol.y[:, -1][:3]
        W = sol.y[:, -1][3:].reshape(3, 3)

        if (step + 1) % n_renorm == 0:
            renorm_count += 1
            current_time = (step + 1) * dt

            orth_error_before = check_orthogonality(W, step)
            W, diag_R = reorthogonalize_householder(W)
            orth_error_after = check_orthogonality(W, step)

            orth_errors.append(orth_error_before)

            log_diag = np.log(diag_R)
            lambda_sum += log_diag
            current_lambda = lambda_sum / current_time

            lambda_history.append(current_lambda.copy())
            t_history.append(current_time)

    stats = {
        'max_orth_error': np.max(orth_errors) if orth_errors else 0,
        'mean_orth_error': np.mean(orth_errors) if orth_errors else 0,
        'renorm_count': renorm_count
    }

    return np.array(t_history), np.array(lambda_history), stats


def test_different_rho():
    sigma = 10.0
    beta = 8.0 / 3.0
    initial_state = [1.0, 1.0, 1.0]
    t_total = 200.0
    dt = 0.01

    rho_values = [14.0, 28.0, 100.0]
    results = {}

    for rho in rho_values:
        print(f"\nCalculating for rho = {rho}...")
        t_hist, lambda_hist, stats = benettin_lyapunov(
            initial_state, sigma, rho, beta, t_total, dt, renorm_interval=0.1
        )
        results[rho] = (t_hist, lambda_hist, stats)
        max_lambda = lambda_hist[-1, 0]
        print(f"  Maximum Lyapunov exponent: {max_lambda:.6f}")
        print(f"  Max orthogonality error: {stats['max_orth_error']:.2e}")
        if max_lambda > 0.01:
            print(f"  System is CHAOTIC")
        else:
            print(f"  System is NOT chaotic")

    return results


def main():
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    initial_state = [1.0, 1.0, 1.0]
    t_total = 200.0
    dt = 0.01
    renorm_interval = 0.1

    print("=" * 70)
    print("LYAPUNOV EXPONENT CALCULATION - IMPROVED BENETTIN METHOD")
    print("         (Householder QR with numerical stabilization)")
    print("=" * 70)
    print(f"\nLorenz system parameters:")
    print(f"  σ = {sigma}")
    print(f"  ρ = {rho}")
    print(f"  β = {beta:.4f}")
    print(f"\nIntegration parameters:")
    print(f"  Total time = {t_total}")
    print(f"  Time step = {dt}")
    print(f"  Reorthogonalization interval = {renorm_interval}")
    print(f"  Initial state = {initial_state}")
    print(f"\nNumerical settings:")
    print(f"  Integration method: RK45")
    print(f"  Relative tolerance: 1e-10")
    print(f"  Absolute tolerance: 1e-12")
    print(f"  QR decomposition: Householder transformation")
    print("\n" + "=" * 70)

    print("\nCalculating Lyapunov exponents with stabilized Householder QR...")
    t_history, lambda_history, stats = benettin_lyapunov(
        initial_state, sigma, rho, beta, t_total, dt, renorm_interval=renorm_interval
    )

    final_exponents = lambda_history[-1]
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nLyapunov exponents (sorted descending):")
    sorted_indices = np.argsort(final_exponents)[::-1]
    for i, idx in enumerate(sorted_indices):
        print(f"  λ{i+1} = {final_exponents[idx]:.6f}")

    max_lambda = final_exponents[0]
    print(f"\nMaximum Lyapunov exponent: {max_lambda:.6f}")

    if max_lambda > 0.01:
        print(f"\n✓ The system is CHAOTIC")
        print(f"  (Positive maximum Lyapunov exponent indicates")
        print(f"   exponential divergence of nearby trajectories)")
    elif abs(max_lambda) < 0.01:
        print(f"\n? The system may be at a bifurcation point")
    else:
        print(f"\n✗ The system is NOT chaotic")
        print(f"  (Negative maximum Lyapunov exponent indicates")
        print(f"   convergence to a fixed point)")

    print("\nSum of Lyapunov exponents (trace check):")
    trace = -sigma - 1 - beta
    sum_exponents = np.sum(final_exponents)
    print(f"  Theoretical (trace of Jacobian): {trace:.6f}")
    print(f"  Calculated:                      {sum_exponents:.6f}")
    print(f"  Relative error:                  {abs((sum_exponents - trace) / trace):.2e}")

    print(f"\nNumerical stability statistics:")
    print(f"  Number of reorthogonalizations:  {stats['renorm_count']}")
    print(f"  Max orthogonality error before:  {stats['max_orth_error']:.2e}")
    print(f"  Mean orthogonality error before: {stats['mean_orth_error']:.2e}")

    print("\nExpected values for Lorenz (σ=10, ρ=28, β=8/3):")
    print(f"  λ₁ ≈ 0.9056,  λ₂ ≈ 0.0,  λ₃ ≈ -14.5723")

    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 3, 1)
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    labels = ['λ₁ (largest)', 'λ₂', 'λ₃ (smallest)']
    for i in range(3):
        ax1.plot(t_history, lambda_history[:, i], color=colors[i], label=labels[i], linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Lyapunov Exponent', fontsize=11)
    ax1.set_title('Convergence of Lyapunov Exponents', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim([-16, 2])

    ax2 = plt.subplot(2, 3, 2)
    t_series = np.linspace(0, 50, 10000)
    trajectory = odeint(lorenz, initial_state, t_series, args=(sigma, rho, beta))
    ax2.plot(trajectory[:, 0], trajectory[:, 2], linewidth=0.7, alpha=0.8, color='#e41a1c')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('z', fontsize=11)
    ax2.set_title('Lorenz Attractor (x-z Plane)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t_series, trajectory[:, 0], label='x', linewidth=0.8, color='#e41a1c')
    ax3.plot(t_series, trajectory[:, 1], label='y', linewidth=0.8, color='#377eb8', alpha=0.7)
    ax3.plot(t_series, trajectory[:, 2], label='z', linewidth=0.8, color='#4daf4a', alpha=0.7)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('State Variables', fontsize=11)
    ax3.set_title('Time Series of Lorenz System', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle=':')

    ax4 = plt.subplot(2, 3, 4, projection='3d')
    ax4.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=0.5, alpha=0.8, color='#984ea3')
    ax4.set_xlabel('x', fontsize=10)
    ax4.set_ylabel('y', fontsize=10)
    ax4.set_zlabel('z', fontsize=10)
    ax4.set_title('3D Lorenz Attractor', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5, projection='3d')
    ax5.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=0.4, alpha=0.6, color='#ff7f00')
    ax5.set_xlabel('x', fontsize=9)
    ax5.set_ylabel('y', fontsize=9)
    ax5.set_zlabel('z', fontsize=9)
    ax5.set_title('Phase Space Trajectory', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    convergence_window = max(1, int(len(t_history) * 0.1))
    rolling_mean = np.convolve(lambda_history[:, 0], np.ones(convergence_window)/convergence_window, mode='valid')
    ax6.plot(t_history[convergence_window-1:], rolling_mean, linewidth=2, color='#e41a1c', label='Smoothed λ₁')
    ax6.axhline(y=0.9056, color='green', linestyle='--', label='Theoretical value (≈0.9056)')
    ax6.set_xlabel('Time', fontsize=11)
    ax6.set_ylabel('Maximum Lyapunov Exponent', fontsize=11)
    ax6.set_title('Convergence of Max Lyapunov Exponent', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, linestyle=':')
    ax6.set_ylim([0.7, 1.1])

    plt.tight_layout()
    plt.savefig('lyapunov_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'lyapunov_analysis.png'")
    print("=" * 70)

    plt.show()


def delay_embedding(ts, embedding_dim, delay):
    n = len(ts)
    n_vectors = n - (embedding_dim - 1) * delay

    if n_vectors <= 0:
        raise ValueError(f"Time series too short for embedding_dim={embedding_dim}, delay={delay}")

    embedded = np.zeros((n_vectors, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = ts[i * delay : i * delay + n_vectors]

    return embedded


def compute_mutual_information(ts, max_delay, bins=50):
    mi_values = []

    for tau in range(1, max_delay + 1):
        x = ts[:-tau]
        y = ts[tau:]

        hist_2d, _, _ = np.histogram2d(x, y, bins=bins, density=True)
        hist_x, _ = np.histogram(x, bins=bins, density=True)
        hist_y, _ = np.histogram(y, bins=bins, density=True)

        eps = 1e-15
        mi = 0.0
        dx = (x.max() - x.min()) / bins
        dy = (y.max() - y.min()) / bins

        for i in range(bins):
            for j in range(bins):
                if hist_2d[i, j] > eps and hist_x[i] > eps and hist_y[j] > eps:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist_x[i] * hist_y[j])) * dx * dy

        mi_values.append(mi)

    return np.array(mi_values)


def find_optimal_delay(ts, max_delay=100, bins=50):
    mi_values = compute_mutual_information(ts, max_delay, bins)
    first_min_idx = np.argmin(mi_values)
    return first_min_idx + 1, mi_values


def false_nearest_neighbors(embedded, threshold=10.0):
    n_points, dim = embedded.shape

    if dim < 2:
        raise ValueError("Need at least dimension 2 for FNN")

    fnn_ratio = 0.0
    for i in range(n_points):
        distances = np.sum((embedded - embedded[i]) ** 2, axis=1)
        distances[i] = np.inf

        sorted_indices = np.argsort(distances)
        nn_idx = sorted_indices[0]

        if distances[nn_idx] > 0:
            distance_increase = abs(embedded[i, -1] - embedded[nn_idx, -1]) / np.sqrt(distances[nn_idx])
            if distance_increase > threshold:
                fnn_ratio += 1

    return fnn_ratio / n_points


def find_embedding_dimension(ts, max_dim=15, delay=1, threshold=10.0, fnn_criterion=0.01):
    fnn_values = []

    for dim in range(2, max_dim + 1):
        embedded = delay_embedding(ts, dim, delay)
        fnn_ratio = false_nearest_neighbors(embedded, threshold)
        fnn_values.append(fnn_ratio)

        if fnn_ratio < fnn_criterion:
            return dim, fnn_values

    return max_dim, fnn_values


def estimate_jacobian_knn(embedded, point_idx, k=10):
    n_points, dim = embedded.shape

    distances = np.sum((embedded - embedded[point_idx]) ** 2, axis=1)
    sorted_indices = np.argsort(distances)

    neighbor_indices = sorted_indices[1 : k + 1]

    X = embedded[neighbor_indices] - embedded[point_idx]

    Y = np.zeros_like(X)
    for j, idx in enumerate(neighbor_indices):
        if idx + 1 < n_points:
            Y[j] = embedded[idx + 1] - embedded[point_idx + 1] if point_idx + 1 < n_points else embedded[idx]

    J, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return J.T


def lyapunov_from_timeseries(ts, embedding_dim, delay, k=20, n_steps=1000, dt=1.0):
    embedded = delay_embedding(ts, embedding_dim, delay)
    n_points = len(embedded)

    W = np.eye(embedding_dim)
    lambda_sum = np.zeros(embedding_dim)
    lambda_history = []
    t_history = []

    start_idx = 100
    end_idx = min(n_points - 1, start_idx + n_steps)

    renorm_interval = 10
    renorm_count = 0

    for i, point_idx in enumerate(range(start_idx, end_idx)):
        J = estimate_jacobian_knn(embedded, point_idx, k=k)

        W = J @ W

        if (i + 1) % renorm_interval == 0:
            renorm_count += 1
            W, R = householder_qr(W)

            diag_R = np.diag(R)
            signs = np.sign(diag_R)
            signs[signs == 0] = 1
            diag_R = np.abs(diag_R)
            diag_R = np.maximum(diag_R, 1e-15)

            W = W * signs

            lambda_sum += np.log(diag_R)
            current_time = (i + 1) * dt
            current_lambda = lambda_sum / current_time

            lambda_history.append(current_lambda.copy())
            t_history.append(current_time)

    return np.array(t_history), np.array(lambda_history), embedded


def generate_lorenz_timeseries(T, dt, sigma=10.0, rho=28.0, beta=8.0/3.0, initial_state=[1.0, 1.0, 1.0]):
    t = np.linspace(0, T, int(T / dt) + 1)
    trajectory = odeint(lorenz, initial_state, t, args=(sigma, rho, beta))
    return t, trajectory


def main_full_spectrum():
    print("=" * 70)
    print("FULL SPECTRUM LYAPUNOV EXPONENTS FROM TIME SERIES")
    print("        (Delay Coordinate Embedding + Benettin Method)")
    print("=" * 70)

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    print(f"\nGenerating Lorenz time series data...")
    T = 200.0
    dt = 0.01
    t, trajectory = generate_lorenz_timeseries(T, dt, sigma, rho, beta)

    ts = trajectory[:, 0]
    print(f"  Time series length: {len(ts)} points")
    print(f"  Sampling interval: {dt}")
    print(f"  Observable: x-component")

    print(f"\n{'='*70}")
    print("Step 1: Finding optimal delay time (mutual information)")
    print("=" * 70)

    optimal_delay, mi_values = find_optimal_delay(ts, max_delay=50)
    print(f"  Optimal delay τ = {optimal_delay}")

    print(f"\n{'='*70}")
    print("Step 2: Finding embedding dimension (false nearest neighbors)")
    print("=" * 70)

    embedding_dim, fnn_values = find_embedding_dimension(ts, max_dim=10, delay=optimal_delay, fnn_criterion=0.02)
    print(f"  Embedding dimension d = {embedding_dim}")

    print(f"\n{'='*70}")
    print("Step 3: Computing Lyapunov spectrum from embedded data")
    print("=" * 70)
    print(f"  k-neighbors: 20")
    print(f"  Renormalization interval: 10 steps")

    t_hist, lambda_hist, embedded = lyapunov_from_timeseries(
        ts, embedding_dim, optimal_delay, k=20, n_steps=2000, dt=dt
    )

    final_exponents = lambda_hist[-1]
    sorted_exponents = np.sort(final_exponents)[::-1]

    print(f"\n{'='*70}")
    print("FULL SPECTRUM RESULTS")
    print("=" * 70)
    print(f"\nLyapunov spectrum (sorted descending):")
    for i, exp_val in enumerate(sorted_exponents):
        print(f"  λ{i+1} = {exp_val:.6f}")

    max_lambda = sorted_exponents[0]
    print(f"\nMaximum Lyapunov exponent: {max_lambda:.6f}")

    if max_lambda > 0.01:
        print(f"\n✓ The system is CHAOTIC")
    elif abs(max_lambda) < 0.01:
        print(f"\n? The system may be periodic or at a bifurcation")
    else:
        print(f"\n✗ The system is NOT chaotic")

    print(f"\nExpected values for Lorenz attractor:")
    print(f"  λ₁ ≈ 0.9056,  λ₂ ≈ 0.0,  λ₃ ≈ -14.5723")

    fig = plt.figure(figsize=(18, 12))

    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(t, ts, linewidth=0.5, color='#377eb8')
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('x(t)', fontsize=10)
    ax1.set_title('Original Time Series (x-component)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    delays = np.arange(1, len(mi_values) + 1)
    ax2.plot(delays, mi_values, 'o-', linewidth=1, markersize=3, color='#e41a1c')
    ax2.axvline(x=optimal_delay, color='green', linestyle='--', label=f'Optimal τ={optimal_delay}')
    ax2.set_xlabel('Delay τ', fontsize=10)
    ax2.set_ylabel('Mutual Information', fontsize=10)
    ax2.set_title('Mutual Information vs Delay', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 3, 3)
    dims = np.arange(2, len(fnn_values) + 2)
    ax3.semilogy(dims, fnn_values, 'o-', linewidth=1, markersize=4, color='#4daf4a')
    ax3.axvline(x=embedding_dim, color='red', linestyle='--', label=f'd={embedding_dim}')
    ax3.axhline(y=0.02, color='gray', linestyle=':', label='2% threshold')
    ax3.set_xlabel('Embedding Dimension', fontsize=10)
    ax3.set_ylabel('FNN Ratio', fontsize=10)
    ax3.set_title('False Nearest Neighbors', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 3, 4, projection='3d')
    if embedding_dim >= 3:
        ax4.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], linewidth=0.3, alpha=0.7, color='#984ea3')
    ax4.set_xlabel('x(t)', fontsize=9)
    ax4.set_ylabel(f'x(t+{optimal_delay})', fontsize=9)
    ax4.set_zlabel(f'x(t+{2*optimal_delay})', fontsize=9)
    ax4.set_title('Reconstructed Attractor (3D Projection)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)
    colors = plt.cm.viridis(np.linspace(0, 1, embedding_dim))
    for i in range(embedding_dim):
        ax5.plot(t_hist, lambda_hist[:, i], color=colors[i], linewidth=1.2, label=f'λ{i+1}')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time Steps', fontsize=10)
    ax5.set_ylabel('Lyapunov Exponent', fontsize=10)
    ax5.set_title('Full Lyapunov Spectrum Convergence', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 3, 6)
    pos_exponents = sorted_exponents[sorted_exponents > 0.01]
    neutral_exponents = sorted_exponents[np.abs(sorted_exponents) <= 0.01]
    neg_exponents = sorted_exponents[sorted_exponents < -0.01]

    bar_labels = [f'λ{i+1}' for i in range(len(sorted_exponents))]
    bar_colors = ['#e41a1c' if e > 0.01 else '#377eb8' if e < -0.01 else '#4daf4a' for e in sorted_exponents]

    bars = ax6.bar(bar_labels, sorted_exponents, color=bar_colors, alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_ylabel('Lyapunov Exponent', fontsize=10)
    ax6.set_title('Lyapunov Spectrum', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, sorted_exponents):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height + (0.1 if height > 0 else -0.5),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(trajectory[:, 0], trajectory[:, 2], linewidth=0.5, alpha=0.7, color='#e41a1c')
    ax7.set_xlabel('x', fontsize=10)
    ax7.set_ylabel('z', fontsize=10)
    ax7.set_title('Original Lorenz Attractor (x-z)', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 8)
    if embedding_dim >= 2:
        ax8.plot(embedded[:, 0], embedded[:, 1], linewidth=0.5, alpha=0.7, color='#377eb8')
    ax8.set_xlabel('x(t)', fontsize=10)
    ax8.set_ylabel(f'x(t+{optimal_delay})', fontsize=10)
    ax8.set_title('Reconstructed (2D Projection)', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    ax9 = plt.subplot(3, 3, 9)
    kaplan_yorke_dim = 0.0
    cumulative_sum = 0.0
    for i, exp_val in enumerate(sorted_exponents):
        if cumulative_sum + exp_val >= 0:
            cumulative_sum += exp_val
            kaplan_yorke_dim = i + 1
        else:
            if i > 0:
                kaplan_yorke_dim = i + cumulative_sum / abs(exp_val)
            break

    info = [
        f'Kaplan-Yorke Dimension: {kaplan_yorke_dim:.2f}',
        f'Positive exponents: {len(pos_exponents)}',
        f'Neutral exponents: {len(neutral_exponents)}',
        f'Negative exponents: {len(neg_exponents)}',
        f'Sum of exponents: {np.sum(sorted_exponents):.4f}',
        f'Embedding dim: {embedding_dim}',
        f'Time delay: {optimal_delay}'
    ]

    ax9.text(0.05, 0.95, '\n'.join(info), transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', family='monospace')
    ax9.set_title('Chaos Analysis Summary', fontsize=11, fontweight='bold')
    ax9.axis('off')

    plt.tight_layout()
    plt.savefig('full_spectrum_lyapunov.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'full_spectrum_lyapunov.png'")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        main_full_spectrum()
    else:
        main()
