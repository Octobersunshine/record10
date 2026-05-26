import numpy as np
from scipy.linalg import svd, lstsq, eig
from scipy.signal import detrend, butter, filtfilt, hilbert
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class RobustPronyAnalyzer:
    def __init__(self, fs: float = 100.0, freq_range: Tuple[float, float] = (0.1, 2.0)):
        self.fs = fs
        self.dt = 1.0 / fs
        self.freq_min, self.freq_max = freq_range

    def preprocess_signal(self, signal: np.ndarray, detrend_order: int = 1,
                          filter_order: int = 4, svd_denoise: bool = True,
                          svd_rank: Optional[int] = None) -> np.ndarray:
        signal = detrend(signal, type='linear' if detrend_order >= 1 else 'constant')
        nyquist = 0.5 * self.fs
        low = max(self.freq_min / nyquist, 0.001)
        high = min(self.freq_max / nyquist, 0.99)
        b, a = butter(filter_order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        if svd_denoise:
            filtered = self._svd_denoise(filtered, rank=svd_rank)
        return filtered

    def _svd_denoise(self, signal: np.ndarray, rank: Optional[int] = None,
                     window_size: Optional[int] = None) -> np.ndarray:
        N = len(signal)
        if window_size is None:
            window_size = min(N // 4, 100)
        L = window_size
        K = N - L + 1
        H = np.zeros((L, K))
        for i in range(K):
            H[:, i] = signal[i:i + L]
        U, s, Vh = svd(H, full_matrices=False)
        if rank is None:
            threshold = 2.858 * np.median(s)
            rank = np.sum(s > threshold)
            rank = max(rank, 4)
            rank = min(rank, L // 2)
        U_r = U[:, :rank]
        s_r = s[:rank]
        Vh_r = Vh[:rank, :]
        H_denoised = U_r @ np.diag(s_r) @ Vh_r
        denoised = np.zeros(N)
        count = np.zeros(N)
        for i in range(K):
            denoised[i:i + L] += H_denoised[:, i]
            count[i:i + L] += 1
        return denoised / count

    def _hankel_matrix(self, signal: np.ndarray, L: int) -> np.ndarray:
        N = len(signal)
        K = N - L + 1
        H = np.zeros((L, K), dtype=complex)
        for i in range(K):
            H[:, i] = signal[i:i + L]
        return H

    def tls_esprit(self, signal: np.ndarray, model_order: Optional[int] = None,
                   max_order: int = 50) -> Dict:
        N = len(signal)
        analytic = hilbert(signal)
        L = min(N // 3, max_order)
        H = self._hankel_matrix(analytic, L)
        U, s, Vh = svd(H, full_matrices=False)
        if model_order is None:
            threshold = 2.858 * np.median(s)
            model_order = np.sum(s > threshold)
            model_order = max(model_order, 2)
            model_order = min(model_order, L // 2)
        M = model_order
        U_s = U[:, :M]
        U_s1 = U_s[:-1, :]
        U_s2 = U_s[1:, :]
        UU = np.hstack((U_s1, U_s2))
        U_uu, s_uu, Vh_uu = svd(UU, full_matrices=False)
        V_uu = Vh_uu.conj().T
        V_12 = V_uu[:M, M:]
        V_22 = V_uu[M:, M:]
        try:
            Phi = -V_12 @ np.linalg.inv(V_22)
        except:
            Phi, _, _, _ = lstsq(V_22, -V_12, cond=1e-12)
        try:
            eigvals, _ = eig(Phi)
        except:
            return self._empty_result(M, analytic)
        return self._extract_modes(eigvals, analytic, M, s)

    def matrix_pencil(self, signal: np.ndarray, pencil_param: Optional[int] = None,
                      model_order: Optional[int] = None, max_order: int = 50) -> Dict:
        N = len(signal)
        analytic = hilbert(signal)
        if pencil_param is None:
            pencil_param = min(N // 3, max_order)
        L = pencil_param
        H = self._hankel_matrix(analytic, L)
        H1 = H[:, :-1]
        H2 = H[:, 1:]
        U, s, Vh = svd(H1, full_matrices=False)
        if model_order is None:
            threshold = 2.858 * np.median(s)
            model_order = np.sum(s > threshold)
            model_order = max(model_order, 2)
            model_order = min(model_order, L - 1)
        M = model_order
        U1 = U[:, :M]
        V1 = Vh[:M, :].conj().T
        S1_inv = np.diag(1.0 / s[:M])
        Y = V1 @ S1_inv @ U1.conj().T @ H2
        try:
            eigvals, _ = eig(Y)
        except:
            return self._empty_result(M, analytic)
        return self._extract_modes(eigvals, analytic, M, s)

    def _extract_modes(self, eigvals: np.ndarray, signal: np.ndarray,
                       model_order: int, singular_values: np.ndarray) -> Dict:
        N = len(signal)
        eigvals = eigvals[np.abs(eigvals) > 1e-6]
        z = eigvals
        p = np.log(z + 1e-15) / self.dt
        sigma = np.real(p)
        omega = np.abs(np.imag(p))
        freq = omega / (2 * np.pi)
        damping_ratio = np.zeros_like(sigma)
        omega_nonzero = omega > 1e-6
        damping_ratio[omega_nonzero] = -sigma[omega_nonzero] / np.sqrt(
            sigma[omega_nonzero]**2 + omega[omega_nonzero]**2)
        valid_mask = (freq >= self.freq_min) & (freq <= self.freq_max) & \
                     (damping_ratio > 0) & (damping_ratio < 0.3) & \
                     (np.abs(z) > 0.85) & (np.abs(z) < 1.05)
        sigma = sigma[valid_mask]
        freq = freq[valid_mask]
        damping_ratio = damping_ratio[valid_mask]
        z = z[valid_mask]
        if len(z) == 0:
            return {
                'freq': np.array([]),
                'damping_ratio': np.array([]),
                'sigma': np.array([]),
                'amplitude': np.array([]),
                'phase': np.array([]),
                'reconstructed': np.zeros_like(signal.real),
                'order': model_order,
                'singular_values': singular_values
            }
        V = np.zeros((N, len(z)), dtype=complex)
        t = np.arange(N)
        for i in range(len(z)):
            V[:, i] = z[i] ** t
        h, _, _, _ = lstsq(V, signal, cond=1e-12)
        amplitude = np.abs(h)
        phase = np.angle(h)
        freq, damping_ratio, sigma, amplitude, phase = self._remove_conjugate_duplicates(
            freq, damping_ratio, sigma, amplitude, phase, tol=5e-3
        )
        sort_idx = np.argsort(amplitude)[::-1]
        freq = freq[sort_idx]
        damping_ratio = damping_ratio[sort_idx]
        sigma = sigma[sort_idx]
        amplitude = amplitude[sort_idx]
        phase = phase[sort_idx]
        reconstructed = np.zeros(N)
        t_sec = np.arange(N) * self.dt
        for i in range(len(freq)):
            reconstructed += amplitude[i] * np.exp(sigma[i] * t_sec) * np.cos(
                2 * np.pi * freq[i] * t_sec + phase[i])
        return {
            'freq': freq,
            'damping_ratio': damping_ratio,
            'sigma': sigma,
            'amplitude': amplitude,
            'phase': phase,
            'reconstructed': reconstructed,
            'order': model_order,
            'singular_values': singular_values
        }

    def _empty_result(self, model_order: int, signal: np.ndarray) -> Dict:
        return {
            'freq': np.array([]),
            'damping_ratio': np.array([]),
            'sigma': np.array([]),
            'amplitude': np.array([]),
            'phase': np.array([]),
            'reconstructed': np.zeros_like(signal.real),
            'order': model_order,
            'singular_values': np.array([])
        }

    def _remove_conjugate_duplicates(self, freqs: np.ndarray, damping: np.ndarray,
                                     sigma: np.ndarray, amplitude: np.ndarray,
                                     phase: np.ndarray, tol: float = 1e-3) -> Tuple:
        if len(freqs) == 0:
            return freqs, damping, sigma, amplitude, phase
        keep_indices = []
        used = set()
        for i in range(len(freqs)):
            if i in used:
                continue
            keep_indices.append(i)
            used.add(i)
            for j in range(i + 1, len(freqs)):
                if j in used:
                    continue
                freq_diff = abs(freqs[i] - freqs[j])
                damp_diff = abs(damping[i] - damping[j])
                if freq_diff < tol and damp_diff < tol:
                    used.add(j)
        return (freqs[keep_indices], damping[keep_indices], sigma[keep_indices],
                amplitude[keep_indices], phase[keep_indices])

    def stabilization_diagram(self, signal: np.ndarray, method: str = 'matrix_pencil',
                              order_range: range = range(4, 51, 2),
                              freq_tol: float = 0.03, damp_tol: float = 0.03) -> Dict:
        all_modes = []
        for order in order_range:
            try:
                if method == 'tls_esprit':
                    result = self.tls_esprit(signal, model_order=order)
                else:
                    result = self.matrix_pencil(signal, model_order=order)
                for f, d, a in zip(result['freq'], result['damping_ratio'], result['amplitude']):
                    all_modes.append({
                        'order': order,
                        'freq': f,
                        'damping': d,
                        'amplitude': a
                    })
            except:
                continue
        stable_modes = self._cluster_modes(all_modes, freq_tol, damp_tol)
        return {
            'all_modes': all_modes,
            'stable_modes': stable_modes,
            'order_range': list(order_range)
        }

    def _cluster_modes(self, modes: List[Dict], freq_tol: float = 0.03,
                       damp_tol: float = 0.03, min_count: int = 3) -> List[Dict]:
        if len(modes) < min_count:
            return []
        data = np.array([[m['freq'], m['damping']] for m in modes])
        weights = np.array([m['amplitude'] for m in modes])
        freq_range = self.freq_max - self.freq_min
        damp_range = 0.5
        normalized_data = np.zeros_like(data)
        normalized_data[:, 0] = data[:, 0] / freq_range
        normalized_data[:, 1] = data[:, 1] / damp_range
        tol_vec = np.array([freq_tol / freq_range, damp_tol / damp_range])
        linkage_matrix = linkage(normalized_data, method='average', metric='euclidean')
        threshold = np.linalg.norm(tol_vec)
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        unique_clusters = np.unique(clusters)
        stable_modes = []
        for c in unique_clusters:
            mask = clusters == c
            if np.sum(mask) >= min_count:
                cluster_data = data[mask]
                cluster_weights = weights[mask]
                mean_freq = np.average(cluster_data[:, 0], weights=cluster_weights)
                mean_damp = np.average(cluster_data[:, 1], weights=cluster_weights)
                mean_amp = np.mean(cluster_weights)
                stable_modes.append({
                    'freq': mean_freq,
                    'damping': mean_damp,
                    'amplitude': mean_amp,
                    'count': int(np.sum(mask)),
                    'freq_std': np.std(cluster_data[:, 0]),
                    'damping_std': np.std(cluster_data[:, 1])
                })
        stable_modes.sort(key=lambda x: x['amplitude'], reverse=True)
        return stable_modes

    def analyze(self, signal: np.ndarray, method: str = 'matrix_pencil',
                use_stabilization: bool = True, **kwargs) -> Dict:
        preprocess_kwargs = {
            'detrend_order': kwargs.pop('detrend_order', 1),
            'filter_order': kwargs.pop('filter_order', 4),
            'svd_denoise': kwargs.pop('svd_denoise', True),
            'svd_rank': kwargs.pop('svd_rank', None)
        }
        processed = self.preprocess_signal(signal, **preprocess_kwargs)
        if use_stabilization:
            stab_kwargs = {
                'order_range': kwargs.pop('order_range', range(4, 51, 2)),
                'freq_tol': kwargs.pop('freq_tol', 0.03),
                'damp_tol': kwargs.pop('damp_tol', 0.03)
            }
            stab_result = self.stabilization_diagram(processed, method=method, **stab_kwargs)
            result = self._get_final_result(processed, stab_result, method)
            result['stabilization'] = stab_result
        else:
            if method == 'tls_esprit':
                result = self.tls_esprit(processed, **kwargs)
            else:
                result = self.matrix_pencil(processed, **kwargs)
        result['original'] = signal
        result['processed'] = processed
        result['time'] = np.arange(len(signal)) * self.dt
        result['method'] = method
        return result

    def _get_final_result(self, signal: np.ndarray, stab_result: Dict, method: str) -> Dict:
        stable_modes = stab_result['stable_modes']
        N = len(signal)
        if len(stable_modes) == 0:
            if method == 'tls_esprit':
                return self.tls_esprit(signal)
            else:
                return self.matrix_pencil(signal)
        freqs = np.array([m['freq'] for m in stable_modes])
        damps = np.array([m['damping'] for m in stable_modes])
        sigma = -2 * np.pi * freqs * damps
        V = np.zeros((N, len(freqs) * 2), dtype=complex)
        t = np.arange(N)
        for i in range(len(freqs)):
            omega = 2 * np.pi * freqs[i]
            z1 = np.exp((sigma[i] + 1j * omega) * self.dt)
            z2 = np.exp((sigma[i] - 1j * omega) * self.dt)
            V[:, 2 * i] = z1 ** t
            V[:, 2 * i + 1] = z2 ** t
        analytic = hilbert(signal)
        h, _, _, _ = lstsq(V, analytic, cond=1e-12)
        amplitude = np.zeros(len(freqs))
        phase = np.zeros(len(freqs))
        for i in range(len(freqs)):
            c1 = h[2 * i]
            c2 = h[2 * i + 1]
            amplitude[i] = np.abs(c1 + c2)
            phase[i] = np.angle(c1 + c2)
        reconstructed = np.zeros(N)
        t_sec = t * self.dt
        for i in range(len(freqs)):
            reconstructed += amplitude[i] * np.exp(sigma[i] * t_sec) * np.cos(
                2 * np.pi * freqs[i] * t_sec + phase[i])
        return {
            'freq': freqs,
            'damping_ratio': damps,
            'sigma': sigma,
            'amplitude': amplitude,
            'phase': phase,
            'reconstructed': reconstructed,
            'order': len(freqs) * 2,
            'singular_values': np.array([]),
            'stable_count': np.array([m['count'] for m in stable_modes]),
            'freq_std': np.array([m['freq_std'] for m in stable_modes]),
            'damping_std': np.array([m['damping_std'] for m in stable_modes])
        }


def generate_test_signal(fs: float = 100.0, duration: float = 10.0,
                         modes: List[Dict] = None, noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fs
    t = np.arange(0, duration, dt)
    if modes is None:
        modes = [
            {'freq': 0.4, 'damping': 0.05, 'amp': 1.0, 'phase': 0.0},
            {'freq': 1.2, 'damping': 0.08, 'amp': 0.5, 'phase': 0.5},
        ]
    signal = np.zeros_like(t)
    for mode in modes:
        sigma = -2 * np.pi * mode['freq'] * mode['damping']
        signal += mode['amp'] * np.exp(sigma * t) * np.cos(
            2 * np.pi * mode['freq'] * t + mode['phase'])
    noise = noise_level * np.random.randn(len(t))
    signal += noise
    return t, signal


def print_results(result: Dict, true_modes: List[Dict] = None):
    print("\n" + "="*80)
    print(f"Prony Analysis Results - {result.get('method', 'Unknown').upper()} Method")
    print("="*80)
    if 'stable_count' in result:
        print(f"{'Mode':<6} {'Freq (Hz)':<14} {'Damping':<12} {'Amplitude':<12} {'Stability':<12} {'Freq Std':<12}")
        print("-"*80)
        for i, (f, d, a, c, fs_std) in enumerate(zip(result['freq'], result['damping_ratio'],
                                                     result['amplitude'], result['stable_count'],
                                                     result['freq_std'])):
            print(f"{i+1:<6} {f:<14.4f} {d:<12.4f} {a:<12.4f} {c:<12d} {fs_std:<12.6f}")
    else:
        print(f"{'Mode':<6} {'Freq (Hz)':<14} {'Damping':<12} {'Amplitude':<12}")
        print("-"*80)
        for i, (f, d, a) in enumerate(zip(result['freq'], result['damping_ratio'],
                                          result['amplitude'])):
            print(f"{i+1:<6} {f:<14.4f} {d:<12.4f} {a:<12.4f}")
    print("="*80)
    if true_modes is not None:
        print("\nTrue Modes:")
        print(f"{'Mode':<6} {'Freq (Hz)':<14} {'Damping':<12}")
        print("-"*40)
        for i, mode in enumerate(true_modes):
            print(f"{i+1:<6} {mode['freq']:<14.4f} {mode['damping']:<12.4f}")
        print("-"*40)


def plot_results(result: Dict, save_path: str = None):
    has_stab = 'stabilization' in result
    if has_stab:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    t = result['time']
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, result['original'], 'b-', label='Original', alpha=0.7, linewidth=1.5)
    ax1.plot(t, result['processed'], 'g-', label='Preprocessed', alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Original vs Preprocessed Signal', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, result['processed'], 'g-', label='Processed', alpha=0.7, linewidth=1.5)
    ax2.plot(t, result['reconstructed'], 'r--', label='Reconstructed', alpha=0.9, linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title('Prony Reconstruction', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    modes_x = np.arange(len(result['freq']))
    ax3 = fig.add_subplot(gs[1, 0])
    bars1 = ax3.bar(modes_x, result['freq'], color='skyblue', edgecolor='navy',
                    alpha=0.7, width=0.6)
    ax3.set_xlabel('Mode Index', fontsize=11)
    ax3.set_ylabel('Frequency (Hz)', fontsize=11)
    ax3.set_title('Oscillation Frequency', fontsize=12, fontweight='bold')
    ax3.set_xticks(modes_x)
    ax3.set_xticklabels([f'Mode {i+1}' for i in modes_x], fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    for bar, freq in zip(bars1, result['freq']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{freq:.3f} Hz', ha='center', va='bottom', fontsize=9)
    ax4 = fig.add_subplot(gs[1, 1])
    bars2 = ax4.bar(modes_x, result['damping_ratio'], color='lightcoral',
                    edgecolor='darkred', alpha=0.7, width=0.6)
    ax4.set_xlabel('Mode Index', fontsize=11)
    ax4.set_ylabel('Damping Ratio', fontsize=11)
    ax4.set_title('Damping Ratio', fontsize=12, fontweight='bold')
    ax4.set_xticks(modes_x)
    ax4.set_xticklabels([f'Mode {i+1}' for i in modes_x], fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    for bar, damp in zip(bars2, result['damping_ratio']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{damp:.4f}', ha='center', va='bottom', fontsize=9)
    if has_stab:
        ax5 = fig.add_subplot(gs[2, :])
        stab = result['stabilization']
        all_modes = stab['all_modes']
        orders = np.array([m['order'] for m in all_modes])
        freqs = np.array([m['freq'] for m in all_modes])
        damps = np.array([m['damping'] for m in all_modes])
        amps = np.array([m['amplitude'] for m in all_modes])
        scatter = ax5.scatter(freqs, orders, c=damps, s=amps * 50,
                              cmap='viridis', alpha=0.6, edgecolors='gray', linewidth=0.5)
        stable = stab['stable_modes']
        for i, mode in enumerate(stable):
            ax5.axvline(x=mode['freq'], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax5.text(mode['freq'] + 0.01, np.max(orders),
                    f'Mode {i+1}: {mode["freq"]:.3f} Hz',
                    color='red', fontsize=9, va='top', rotation=90)
        ax5.set_xlabel('Frequency (Hz)', fontsize=11)
        ax5.set_ylabel('Model Order', fontsize=11)
        ax5.set_title('Stabilization Diagram - Frequency vs Model Order',
                     fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, linestyle='--')
        cbar = plt.colorbar(scatter, ax=ax5, pad=0.01)
        cbar.set_label('Damping Ratio', fontsize=10)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    np.random.seed(42)
    fs = 100.0
    duration = 20.0
    noise_level = 0.02
    print("="*70)
    print("Robust Prony Analysis for Power System Low-Frequency Oscillation")
    print(f"Using TLS-ESPRIT and Matrix Pencil Methods")
    print(f"Noise level: {noise_level}")
    print("="*70)
    true_modes = [
        {'freq': 0.4, 'damping': 0.05, 'amp': 1.0, 'phase': 0.0},
        {'freq': 1.2, 'damping': 0.08, 'amp': 0.6, 'phase': 0.3},
        {'freq': 0.8, 'damping': 0.03, 'amp': 0.4, 'phase': 0.1},
    ]
    print("\nTrue modes:")
    for i, mode in enumerate(true_modes):
        print(f"  Mode {i+1}: {mode['freq']:.2f} Hz, damping = {mode['damping']:.3f}, amp = {mode['amp']:.2f}")
    t, signal = generate_test_signal(fs=fs, duration=duration,
                                      modes=true_modes, noise_level=noise_level)
    analyzer = RobustPronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))
    print("\n" + "="*70)
    print("1. Matrix Pencil Method with Stabilization Diagram")
    print("="*70)
    result_mp = analyzer.analyze(signal, method='matrix_pencil', use_stabilization=True,
                                  svd_denoise=True, order_range=range(6, 36, 3))
    print_results(result_mp, true_modes=true_modes)
    mse_mp = np.mean((result_mp['processed'] - result_mp['reconstructed'])**2)
    print(f"\nReconstruction MSE: {mse_mp:.8f}")
    print("\n" + "="*70)
    print("2. TLS-ESPRIT Method with Stabilization Diagram")
    print("="*70)
    result_tls = analyzer.analyze(signal, method='tls_esprit', use_stabilization=True,
                                   svd_denoise=True, order_range=range(6, 36, 3))
    print_results(result_tls, true_modes=true_modes)
    mse_tls = np.mean((result_tls['processed'] - result_tls['reconstructed'])**2)
    print(f"\nReconstruction MSE: {mse_tls:.8f}")
    print("\n" + "="*70)
    print("Comparison Summary:")
    print("="*70)
    print(f"{'Method':<25} {'Modes Found':<15} {'MSE':<15}")
    print("-"*60)
    print(f"{'Matrix Pencil':<25} {len(result_mp['freq']):<15d} {mse_mp:<15.8f}")
    print(f"{'TLS-ESPRIT':<25} {len(result_tls['freq']):<15d} {mse_tls:<15.8f}")
    print("="*70)
    print("\nAccuracy Comparison (vs True Modes):")
    print("="*70)
    for result, method_name in [(result_mp, 'Matrix Pencil'), (result_tls, 'TLS-ESPRIT')]:
        print(f"\n{method_name}:")
        for i, (f, d) in enumerate(zip(result['freq'], result['damping_ratio'])):
            true_mode = min(true_modes, key=lambda m: abs(m['freq'] - f))
            freq_err = abs(f - true_mode['freq']) / true_mode['freq'] * 100
            damp_err = abs(d - true_mode['damping']) / max(true_mode['damping'], 1e-6) * 100
            print(f"  Mode {i+1}: Freq = {f:.4f} Hz (error: {freq_err:.2f}%), "
                  f"Damping = {d:.4f} (error: {damp_err:.2f}%)")
    plot_results(result_mp, save_path='prony_robust_result.png')


if __name__ == '__main__':
    main()
