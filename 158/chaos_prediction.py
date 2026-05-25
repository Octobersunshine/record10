import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial.distance import cdist
from scipy.stats import entropy


def lorenz_system(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


def generate_lorenz_data(n_steps=10000, dt=0.01, initial_state=[1.0, 1.0, 1.0]):
    t = np.arange(0, n_steps * dt, dt)
    states = odeint(lorenz_system, initial_state, t)
    return t, states


def compute_mutual_information(x, y, bins=50):
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
    pxy = hist_2d / hist_2d.sum()
    
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    px_py = np.outer(px, py)
    
    mask = (pxy > 0) & (px_py > 0)
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / px_py[mask]))
    
    return mi


def find_optimal_delay(time_series, max_delay=100, bins=50):
    mi_values = []
    for tau in range(1, max_delay + 1):
        x = time_series[:-tau]
        y = time_series[tau:]
        mi = compute_mutual_information(x, y, bins=bins)
        mi_values.append(mi)
    
    mi_values = np.array(mi_values)
    first_min_index = np.where(np.diff(np.sign(np.diff(mi_values))) > 0)[0]
    
    if len(first_min_index) > 0:
        optimal_delay = first_min_index[0] + 2
    else:
        optimal_delay = np.argmin(mi_values) + 1
    
    return optimal_delay, mi_values


def delay_embedding(time_series, embedding_dim, time_delay):
    n = len(time_series)
    embedded = np.zeros((n - (embedding_dim - 1) * time_delay, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * time_delay : i * time_delay + len(embedded)]
    return embedded


def false_nearest_neighbors(time_series, time_delay, max_dim=15, threshold=10.0):
    n = len(time_series)
    fnn_percentages = []
    
    for d in range(1, max_dim + 1):
        if d == 1:
            embedded = time_series[:-(time_delay)].reshape(-1, 1)
        else:
            embedded = delay_embedding(time_series, d, time_delay)
            if len(embedded) < 2:
                fnn_percentages.append(100.0)
                continue
        
        fnn_count = 0
        total_count = 0
        
        for i in range(len(embedded)):
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            distances[i] = np.inf
            
            if len(distances) > 1:
                nn_idx = np.argmin(distances)
                r_d = distances[nn_idx]
                
                if r_d > 0 and d < max_dim:
                    if (i + d * time_delay) < len(time_series) and (nn_idx + d * time_delay) < len(time_series):
                        if d == 1:
                            next_dist = abs(time_series[i + time_delay] - time_series[nn_idx + time_delay])
                        else:
                            next_pt1 = np.append(embedded[i], time_series[i + d * time_delay])
                            next_pt2 = np.append(embedded[nn_idx], time_series[nn_idx + d * time_delay])
                            next_dist = np.linalg.norm(next_pt1 - next_pt2)
                        
                        if (next_dist / r_d) > threshold:
                            fnn_count += 1
                    total_count += 1
        
        if total_count > 0:
            fnn_percentage = (fnn_count / total_count) * 100
        else:
            fnn_percentage = 100.0
        fnn_percentages.append(fnn_percentage)
    
    fnn_percentages = np.array(fnn_percentages)
    
    optimal_dim_idx = np.where(fnn_percentages < 5.0)[0]
    if len(optimal_dim_idx) > 0:
        optimal_dim = optimal_dim_idx[0] + 1
    else:
        optimal_dim = np.argmin(fnn_percentages) + 1
    
    return optimal_dim, fnn_percentages


def find_nearest_neighbors(query_point, data, k):
    distances = cdist([query_point], data)[0]
    indices = np.argsort(distances)[:k]
    return indices, distances[indices]


def local_linear_predict(embedded_data, k=10):
    n_points = len(embedded_data)
    predictions = np.zeros(n_points - 1)
    
    for i in range(n_points - 1):
        current_point = embedded_data[i]
        
        if i < k:
            neighbor_indices = np.arange(min(k, n_points - 1))
        else:
            neighbor_indices, _ = find_nearest_neighbors(current_point, embedded_data[:i], k)
        
        X = np.column_stack([embedded_data[idx] for idx in neighbor_indices])
        Y = np.column_stack([embedded_data[idx + 1] for idx in neighbor_indices])
        
        X_mean = np.mean(X, axis=1, keepdims=True)
        Y_mean = np.mean(Y, axis=1, keepdims=True)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        
        A = Y_centered @ np.linalg.pinv(X_centered)
        b = Y_mean - A @ X_mean
        
        predicted_next = A @ current_point.reshape(-1, 1) + b
        predictions[i] = predicted_next[0, 0]
    
    return predictions


def rbf_predict(embedded_data, k=50, epsilon=1.0):
    n_points = len(embedded_data)
    predictions = np.zeros(n_points - 1)
    
    for i in range(n_points - 1):
        current_point = embedded_data[i]
        
        if i < k:
            train_indices = np.arange(min(k, n_points - 1))
        else:
            train_indices, _ = find_nearest_neighbors(current_point, embedded_data[:i], k)
            train_indices = np.sort(train_indices)
        
        centers = embedded_data[train_indices]
        targets = embedded_data[train_indices + 1, 0]
        
        distances = cdist(centers, centers)
        Phi = np.exp(-distances ** 2 / (2 * epsilon ** 2))
        
        weights = np.linalg.lstsq(Phi, targets, rcond=None)[0]
        
        test_distances = cdist([current_point], centers)[0]
        test_Phi = np.exp(-test_distances ** 2 / (2 * epsilon ** 2))
        predictions[i] = np.dot(test_Phi, weights)
    
    return predictions


class EchoStateNetwork:
    def __init__(self, input_size=1, reservoir_size=500, output_size=1, 
                 spectral_radius=1.2, sparsity=0.95, input_scaling=1.0, 
                 ridge_alpha=1e-6, random_state=42):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.ridge_alpha = ridge_alpha
        
        np.random.seed(random_state)
        
        self.W_in = np.random.uniform(-input_scaling, input_scaling, 
                                      (reservoir_size, input_size + 1))
        
        self.W_res = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) < sparsity
        self.W_res[mask] = 0
        
        radius = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        self.W_res = (self.W_res / radius) * spectral_radius
        
        self.W_out = None
        self.reservoir_states = None
    
    def _update_reservoir(self, state, input_data):
        combined_input = np.vstack([np.ones((1, 1)), input_data.reshape(-1, 1)])
        pre_activation = np.dot(self.W_in, combined_input) + np.dot(self.W_res, state)
        return np.tanh(pre_activation)
    
    def train(self, input_data, teacher_data, washout=100):
        n_samples = len(input_data)
        states = np.zeros((self.reservoir_size, n_samples))
        
        current_state = np.zeros((self.reservoir_size, 1))
        for i in range(n_samples):
            current_state = self._update_reservoir(current_state, input_data[i])
            states[:, i] = current_state.flatten()
        
        self.reservoir_states = states
        
        states_with_bias = np.vstack([np.ones((1, n_samples)), states])
        
        X = states_with_bias[:, washout:].T
        Y = teacher_data[washout:].reshape(-1, self.output_size)
        
        self.W_out = np.dot(np.linalg.inv(np.dot(X.T, X) + 
                                          self.ridge_alpha * np.eye(X.shape[1])),
                            np.dot(X.T, Y)).T
        
        return states
    
    def predict(self, input_data, n_steps=1, continuation=False):
        if self.W_out is None:
            raise ValueError("Model not trained yet!")
        
        if not continuation:
            states = np.zeros((self.reservoir_size, len(input_data)))
            current_state = np.zeros((self.reservoir_size, 1))
            
            for i in range(len(input_data)):
                current_state = self._update_reservoir(current_state, input_data[i])
                states[:, i] = current_state.flatten()
            
            states_with_bias = np.vstack([np.ones((1, len(input_data))), states])
            predictions = np.dot(self.W_out, states_with_bias).flatten()
            
            return predictions, states
        else:
            predictions = np.zeros(n_steps)
            current_state = self.reservoir_states[:, -1].reshape(-1, 1) if self.reservoir_states is not None else np.zeros((self.reservoir_size, 1))
            current_input = input_data[-1] if np.isscalar(input_data[-1]) else input_data[-1].flatten()[0]
            
            for i in range(n_steps):
                current_state = self._update_reservoir(current_state, current_input)
                state_with_bias = np.vstack([np.ones((1, 1)), current_state])
                pred = np.dot(self.W_out, state_with_bias).flatten()[0]
                predictions[i] = pred
                current_input = pred
            
            return predictions


def calculate_lyapunov_time_lorenz(dt=0.01, max_time=10.0):
    lambda_max = 0.9
    lyapunov_time = 1.0 / lambda_max
    return lyapunov_time


def calculate_prediction_horizon(errors, threshold=0.5, dt=0.01):
    normalized_error = errors / np.max(errors)
    horizon_idx = np.where(normalized_error > threshold)[0]
    if len(horizon_idx) > 0:
        return horizon_idx[0] * dt
    else:
        return len(errors) * dt


def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def main():
    print("=" * 70)
    print("Chaos Time Series Prediction with ESN (Reservoir Computing)")
    print("=" * 70)
    
    print("\n1. Generating Lorenz system data...")
    n_steps = 15000
    dt = 0.01
    t, states = generate_lorenz_data(n_steps=n_steps, dt=dt)
    x_series = states[:, 0]
    print(f"   Generated {len(x_series)} time steps (dt={dt})")
    
    lyapunov_time = calculate_lyapunov_time_lorenz(dt)
    print(f"   Lorenz system Lyapunov time: ~{lyapunov_time:.2f} time units")
    
    print("\n2. Finding optimal time delay using Mutual Information...")
    max_delay = 50
    optimal_delay, mi_values = find_optimal_delay(x_series, max_delay=max_delay, bins=50)
    print(f"   Optimal time delay τ = {optimal_delay}")
    
    print("\n3. Finding optimal embedding dimension using False Nearest Neighbors...")
    max_dim = 15
    optimal_dim, fnn_percentages = false_nearest_neighbors(
        x_series, time_delay=optimal_delay, max_dim=max_dim, threshold=10.0
    )
    print(f"   Optimal embedding dimension m = {optimal_dim}")
    
    print(f"\n4. Performing delay embedding (m={optimal_dim}, τ={optimal_delay})...")
    embedded_data = delay_embedding(x_series, optimal_dim, optimal_delay)
    print(f"   Embedded data shape: {embedded_data.shape}")
    
    print("\n5. Running Local Linear Prediction...")
    k_ll = min(20, len(embedded_data) // 10)
    ll_predictions = local_linear_predict(embedded_data, k=k_ll)
    
    print("   Running RBF Prediction...")
    k_rbf = min(50, len(embedded_data) // 5)
    epsilon = np.std(embedded_data) * 2
    rbf_predictions = rbf_predict(embedded_data, k=k_rbf, epsilon=epsilon)
    
    print("\n6. Training Echo State Network (ESN)...")
    n_train = int(0.7 * len(x_series))
    n_test = len(x_series) - n_train
    
    train_input = x_series[:n_train].reshape(-1, 1)
    train_teacher = x_series[1:n_train+1]
    
    esn = EchoStateNetwork(
        input_size=1,
        reservoir_size=800,
        spectral_radius=1.2,
        sparsity=0.98,
        input_scaling=0.5,
        ridge_alpha=1e-6
    )
    
    reservoir_states = esn.train(train_input, train_teacher, washout=200)
    print(f"   Reservoir size: {esn.reservoir_size} neurons")
    print(f"   Spectral radius: {esn.spectral_radius}")
    print(f"   Sparsity: {esn.sparsity * 100:.1f}%")
    
    test_input = x_series[n_train:-1].reshape(-1, 1)
    esn_predictions, _ = esn.predict(test_input, n_steps=1, continuation=False)
    
    actual = embedded_data[1:, 0]
    esn_actual = x_series[n_train+1:n_train+1+len(esn_predictions)]
    
    min_len = min(len(actual), len(ll_predictions), len(rbf_predictions))
    actual_short = actual[:min_len]
    ll_pred = ll_predictions[:min_len]
    rbf_pred = rbf_predictions[:min_len]
    
    ll_rmse = calculate_rmse(actual_short, ll_pred)
    rbf_rmse = calculate_rmse(actual_short, rbf_pred)
    esn_rmse = calculate_rmse(esn_actual, esn_predictions)
    
    print(f"\n7. One-step Prediction Results:")
    print(f"   Local Linear RMSE: {ll_rmse:.6f}")
    print(f"   RBF RMSE:          {rbf_rmse:.6f}")
    print(f"   ESN RMSE:          {esn_rmse:.6f}")
    
    print(f"\n8. Long-term prediction (generative mode)...")
    horizon_steps = int(5 * lyapunov_time / dt)
    
    seed_input = x_series[n_train-100:n_train]
    _, seed_states = esn.predict(seed_input, n_steps=1, continuation=False)
    esn.reservoir_states = seed_states
    
    long_term_pred = esn.predict(x_series[n_train-10:n_train], n_steps=horizon_steps, continuation=True)
    long_term_actual = x_series[n_train:n_train+horizon_steps]
    
    min_len_long = min(len(long_term_pred), len(long_term_actual))
    long_term_pred = long_term_pred[:min_len_long]
    long_term_actual = long_term_actual[:min_len_long]
    
    cumulative_error = np.abs(long_term_pred - long_term_actual)
    pred_horizon = calculate_prediction_horizon(cumulative_error, threshold=np.std(long_term_actual)*0.5, dt=dt)
    esn_long_rmse = calculate_rmse(long_term_actual, long_term_pred)
    
    print(f"   Prediction horizon: {horizon_steps} steps ({horizon_steps*dt:.1f} time units)")
    print(f"   ESN long-term RMSE: {esn_long_rmse:.4f}")
    print(f"   Valid prediction (~0.5σ): ~{pred_horizon:.2f} time units")
    
    fig = plt.figure(figsize=(18, 16))
    
    ax1 = fig.add_subplot(5, 3, 1)
    ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Lorenz Attractor (X-Y Plane)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(5, 3, 2, projection='3d')
    ax2.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', linewidth=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Lorenz Attractor (3D)')
    
    ax3 = fig.add_subplot(5, 3, 3)
    delays = np.arange(1, max_delay + 1)
    ax3.plot(delays, mi_values, 'b-', linewidth=2)
    ax3.axvline(x=optimal_delay, color='r', linestyle='--', linewidth=2, label=f'Optimal τ={optimal_delay}')
    ax3.set_xlabel('Time Delay τ')
    ax3.set_ylabel('Mutual Information')
    ax3.set_title('Mutual Information vs Time Delay')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(5, 3, 4)
    dims = np.arange(1, max_dim + 1)
    ax4.plot(dims, fnn_percentages, 'b-o', linewidth=2, markersize=4)
    ax4.axvline(x=optimal_dim, color='r', linestyle='--', linewidth=2, label=f'Optimal m={optimal_dim}')
    ax4.axhline(y=5.0, color='g', linestyle=':', linewidth=1.5, label='5% Threshold')
    ax4.set_xlabel('Embedding Dimension m')
    ax4.set_ylabel('Percentage of FNN (%)')
    ax4.set_title('False Nearest Neighbors Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(5, 3, 5, projection='3d')
    if optimal_dim >= 3:
        ax5.plot(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], 'r-', linewidth=0.3)
        ax5.set_xlabel('x(t)')
        ax5.set_ylabel('x(t+τ)')
        ax5.set_zlabel('x(t+2τ)')
    else:
        ax5.plot(embedded_data[:, 0], embedded_data[:, 1], 'r-', linewidth=0.3)
        ax5.set_xlabel('x(t)')
        ax5.set_ylabel('x(t+τ)')
    ax5.set_title(f'Reconstructed Attractor (m={optimal_dim}, τ={optimal_delay})')
    
    ax6 = fig.add_subplot(5, 3, 6)
    ax6.plot(actual_short, label='Actual', linewidth=1.5)
    ax6.plot(ll_pred, label='Local Linear', linewidth=0.7, alpha=0.9)
    ax6.plot(rbf_pred, label='RBF', linewidth=0.7, alpha=0.9)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('X Value')
    ax6.set_title('One-step Prediction Comparison')
    ax6.legend(fontsize=8)
    ax6.set_xlim(0, min(800, len(actual_short)))
    ax6.grid(True, alpha=0.3)
    
    n_neurons_show = min(20, esn.reservoir_size)
    ax7 = fig.add_subplot(5, 3, 7)
    for i in range(n_neurons_show):
        ax7.plot(reservoir_states[i, -500:] + i * 2, linewidth=0.7)
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Neuron Activation (offset)')
    ax7.set_title(f'ESN Reservoir States ({n_neurons_show} neurons)')
    ax7.set_yticks([])
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(5, 3, 8)
    t_horizon = np.arange(horizon_steps) * dt
    ax8.plot(t_horizon, long_term_actual, label='Actual', linewidth=1.5)
    ax8.plot(t_horizon, long_term_pred, label=f'ESN Predicted', linewidth=1.2, alpha=0.9)
    ax8.axvline(x=lyapunov_time, color='r', linestyle='--', linewidth=1.5, label=f'Lyapunov time ({lyapunov_time:.1f})')
    ax8.axvline(x=pred_horizon, color='g', linestyle=':', linewidth=2, label=f'Valid horizon ({pred_horizon:.1f})')
    ax8.set_xlabel('Time (Lyapunov units)')
    ax8.set_ylabel('X Value')
    ax8.set_title(f'Long-term ESN Prediction\nRMSE={esn_long_rmse:.4f}')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(5, 3, 9)
    ax9.plot(t_horizon, cumulative_error, 'b-', linewidth=1)
    ax9.axhline(y=np.std(long_term_actual)*0.5, color='r', linestyle='--', linewidth=1.5, label='0.5σ threshold')
    ax9.axvline(x=pred_horizon, color='g', linestyle=':', linewidth=2, label=f'Valid horizon ({pred_horizon:.1f})')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Absolute Error')
    ax9.set_title('Prediction Error Growth')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(5, 3, 10)
    ax10.plot(actual_short - ll_pred, 'b-', linewidth=0.5)
    ax10.set_xlabel('Time Step')
    ax10.set_ylabel('Error')
    ax10.set_title(f'Local Linear Error (RMSE={ll_rmse:.4f})')
    ax10.set_xlim(0, min(800, len(actual_short)))
    ax10.grid(True, alpha=0.3)
    
    ax11 = fig.add_subplot(5, 3, 11)
    ax11.plot(actual_short - rbf_pred, 'r-', linewidth=0.5)
    ax11.set_xlabel('Time Step')
    ax11.set_ylabel('Error')
    ax11.set_title(f'RBF Error (RMSE={rbf_rmse:.4f})')
    ax11.set_xlim(0, min(800, len(actual_short)))
    ax11.grid(True, alpha=0.3)
    
    ax12 = fig.add_subplot(5, 3, 12)
    test_indices = np.arange(len(esn_predictions))
    ax12.plot(esn_actual - esn_predictions, 'g-', linewidth=0.5)
    ax12.set_xlabel('Time Step')
    ax12.set_ylabel('Error')
    ax12.set_title(f'ESN One-step Error (RMSE={esn_rmse:.4f})')
    ax12.set_xlim(0, min(800, len(esn_predictions)))
    ax12.grid(True, alpha=0.3)
    
    ax13 = fig.add_subplot(5, 3, 13)
    ax13.scatter(actual_short, ll_pred, s=1, alpha=0.3)
    ax13.plot([actual_short.min(), actual_short.max()], 
              [actual_short.min(), actual_short.max()], 'r--', linewidth=2)
    ax13.set_xlabel('Actual')
    ax13.set_ylabel('Predicted')
    ax13.set_title('Local Linear: Actual vs Predicted')
    ax13.grid(True, alpha=0.3)
    
    ax14 = fig.add_subplot(5, 3, 14)
    ax14.scatter(actual_short, rbf_pred, s=1, alpha=0.3)
    ax14.plot([actual_short.min(), actual_short.max()], 
              [actual_short.min(), actual_short.max()], 'r--', linewidth=2)
    ax14.set_xlabel('Actual')
    ax14.set_ylabel('Predicted')
    ax14.set_title('RBF: Actual vs Predicted')
    ax14.grid(True, alpha=0.3)
    
    ax15 = fig.add_subplot(5, 3, 15)
    ax15.scatter(esn_actual, esn_predictions, s=1, alpha=0.3)
    ax15.plot([esn_actual.min(), esn_actual.max()], 
              [esn_actual.min(), esn_actual.max()], 'r--', linewidth=2)
    ax15.set_xlabel('Actual')
    ax15.set_ylabel('Predicted')
    ax15.set_title('ESN: Actual vs Predicted')
    ax15.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chaos_prediction_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to 'chaos_prediction_results.png'")
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_horizon, long_term_actual, label='Actual Lorenz', linewidth=2, alpha=0.8)
    plt.plot(t_horizon, long_term_pred, label='ESN Prediction', linewidth=1.5)
    plt.axvline(x=lyapunov_time, color='r', linestyle='--', linewidth=2, 
                label=f'1 Lyapunov time = {lyapunov_time:.1f}')
    plt.xlabel('Time (units)')
    plt.ylabel('X(t)')
    plt.title(f'ESN Long-term Prediction of Lorenz System\nValid Horizon: ~{pred_horizon:.1f} time units ({pred_horizon/lyapunov_time:.1f} λ⁻¹)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('esn_long_term_prediction.png', dpi=150, bbox_inches='tight')
    print("ESN long-term prediction plot saved to 'esn_long_term_prediction.png'")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"  Embedding Parameters:")
    print(f"    - Time delay (τ) by Mutual Information: {optimal_delay}")
    print(f"    - Embedding dimension (m) by FNN: {optimal_dim}")
    print(f"\n  One-step Prediction RMSE:")
    print(f"    - Local Linear: {ll_rmse:.6f}")
    print(f"    - RBF:          {rbf_rmse:.6f}")
    print(f"    - ESN:          {esn_rmse:.6f}")
    print(f"\n  ESN Long-term Prediction ({horizon_steps*dt:.1f} time units):")
    print(f"    - Lyapunov time (λ⁻¹): ~{lyapunov_time:.2f}")
    print(f"    - Long-term RMSE: {esn_long_rmse:.4f}")
    print(f"    - Valid prediction horizon: ~{pred_horizon:.2f} time units")
    print(f"    - Valid horizon / Lyapunov time: {pred_horizon/lyapunov_time:.2f}")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()
