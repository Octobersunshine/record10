import numpy as np
from scipy.linalg import svd, lstsq, toeplitz, eigh
from scipy.signal import detrend, butter, filtfilt, hilbert
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class PronyAnalyzer:
    def __init__(self, fs: float = 100.0, freq_range: Tuple[float, float] = (0.1, 2.0)):
        self.fs = fs
        self.dt = 1.0 / fs
        self.freq_min, self.freq_max = freq_range

    def preprocess_signal(self, signal: np.ndarray, detrend_order: int = 1,
                          filter_order: int = 4) -> np.ndarray:
        signal = detrend(signal, type='linear' if detrend_order >= 1 else 'constant')
        nyquist = 0.5 * self.fs
        low = max(self.freq_min / nyquist, 0.001)
        high = min(self.freq_max / nyquist, 0.99)
        b, a = butter(filter_order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered

    def _estimate_noise_level(self, signal: np.ndarray) -> float:
        diff_signal = np.diff(signal)
        noise_std = np.std(diff_signal) / np.sqrt(2)
        return noise_std

    def _hankel_matrix(self, signal: np.ndarray, L: int) -> np.ndarray:
        N = len(signal)
        K = N - L + 1
        H = np.zeros((L, K), dtype=complex)
        for i in range(K):
            H[:, i] = signal[i:i + L]
        return H

    def _tls_esprit(self, signal: np.ndarray, order: int = None,
                    threshold: float = None) -> Dict:
        N = len(signal)
        analytic = hilbert(signal)
        if order is None:
            order = min(N // 3, 20)
        L = min(order, N // 2)
        H = self._hankel_matrix(analytic, L)
        U, S, Vh = svd(H, full_matrices=False)
        if threshold is None:
            noise_level = self._estimate_noise_level(np.real(analytic))
            signal_power = np.std(np.real(analytic))
            snr = signal_power / max(noise_level, 1e-10)
            if snr > 20:
                threshold = 0.001
            elif snr > 5:
                threshold = 0.01
            else:
                threshold = 0.05
        effective_rank = np.sum(S > threshold * S[0])
        effective_rank = max(effective_rank, 4)
        M = min(effective_rank * 2, L - 1)
        M = max(M, 4)
        M = min(M, L // 2)
        U1 = U[:, :M]
        U0 = U1[:-1, :]
        U1_shift = U1[1:, :]
        U_aug = np.hstack([U0, U1_shift])
        _, _, V_aug = svd(U_aug, full_matrices=False)
        V_21 = V_aug[M:, :M]
        V_22 = V_aug[M:, M:]
        Phi = -V_21 @ np.linalg.inv(V_22 + 1e-12 * np.eye(M))
        eigvals, _ = np.linalg.eig(Phi)
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
                     (np.abs(z) > 0.9) & (np.abs(z) < 1.02)
        sigma = sigma[valid_mask]
        freq = freq[valid_mask]
        damping_ratio = damping_ratio[valid_mask]
        z = z[valid_mask]
        singular_values = S
        return {
            'z': z, 'sigma': sigma, 'freq': freq,
            'damping_ratio': damping_ratio, 'singular_values': singular_values,
            'effective_rank': effective_rank, 'M': M
        }

    def _matrix_pencil_improved(self, signal: np.ndarray, order: int = None,
                                threshold: float = None) -> Dict:
        N = len(signal)
        analytic = hilbert(signal)
        if order is None:
            order = min(N // 3, 20)
        L = min(order, N // 2)
        H = self._hankel_matrix(analytic, L)
        H1 = H[:-1, :]
        H2 = H[1:, :]
        U, S, Vh = svd(H1, full_matrices=False)
        if threshold is None:
            noise_level = self._estimate_noise_level(np.real(analytic))
            signal_power = np.std(np.real(analytic))
            snr = signal_power / max(noise_level, 1e-10)
            if snr > 20:
                threshold = 0.001
            elif snr > 5:
                threshold = 0.01
            else:
                threshold = 0.05
        M = np.sum(S > threshold * S[0])
        M = max(M, 4)
        M = min(M, L - 1)
        U1 = U[:, :M]
        V1 = Vh[:M, :].conj().T
        S1_inv = np.diag(1.0 / (S[:M] + 1e-12))
        Y = V1 @ S1_inv @ U1.conj().T @ H2
        eigvals, _ = np.linalg.eig(Y)
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
                     (np.abs(z) > 0.9) & (np.abs(z) < 1.02)
        sigma = sigma[valid_mask]
        freq = freq[valid_mask]
        damping_ratio = damping_ratio[valid_mask]
        z = z[valid_mask]
        singular_values = S
        return {
            'z': z, 'sigma': sigma, 'freq': freq,
            'damping_ratio': damping_ratio, 'singular_values': singular_values,
            'effective_rank': M, 'M': M
        }

    def _stability_diagram_analysis(self, signal: np.ndarray,
                                    max_order: int = 30,
                                    method: str = 'tls_esprit') -> Dict:
        N = len(signal)
        all_modes = []
        min_order = 8
        orders = range(min_order, min(max_order, N // 3) + 1, 2)
        for order in orders:
            try:
                if method == 'tls_esprit':
                    result = self._tls_esprit(signal, order=order)
                else:
                    result = self._matrix_pencil_improved(signal, order=order)
                for i in range(len(result['freq'])):
                    all_modes.append({
                        'order': order,
                        'freq': result['freq'][i],
                        'damping': result['damping_ratio'][i],
                        'sigma': result['sigma'][i],
                        'z': result['z'][i]
                    })
            except Exception:
                continue
        if len(all_modes) == 0:
            return {'stable_modes': [], 'all_modes': []}
        stable_modes = self._identify_stable_modes(all_modes, orders)
        return {'stable_modes': stable_modes, 'all_modes': all_modes, 'orders': list(orders)}

    def _identify_stable_modes(self, all_modes: List[Dict],
                               orders: List[int],
                               freq_tol: float = 0.02,
                               damp_tol: float = 0.02,
                               min_stability: int = 3) -> List[Dict]:
        if len(all_modes) == 0:
            return []
        freq_clusters = []
        for mode in all_modes:
            found = False
            for cluster in freq_clusters:
                if abs(mode['freq'] - cluster['center_freq']) < freq_tol * cluster['center_freq']:
                    cluster['modes'].append(mode)
                    cluster['center_freq'] = np.mean([m['freq'] for m in cluster['modes']])
                    found = True
                    break
            if not found:
                freq_clusters.append({
                    'center_freq': mode['freq'],
                    'modes': [mode]
                })
        stable_modes = []
        for cluster in freq_clusters:
            if len(cluster['modes']) >= min_stability:
                damp_values = [m['damping'] for m in cluster['modes']]
                damp_std = np.std(damp_values)
                freq_values = [m['freq'] for m in cluster['modes']]
                freq_std = np.std(freq_values)
                mean_freq = cluster['center_freq']
                mean_damp = np.mean(damp_values)
                mean_sigma = np.mean([m['sigma'] for m in cluster['modes']])
                stability_score = len(cluster['modes']) / len(orders)
                if freq_std < freq_tol * mean_freq and damp_std < damp_tol * max(mean_damp, 0.01):
                    stable_modes.append({
                        'freq': mean_freq,
                        'damping': mean_damp,
                        'sigma': mean_sigma,
                        'stability_score': stability_score,
                        'num_orders': len(cluster['modes'])
                    })
        stable_modes.sort(key=lambda x: x['stability_score'], reverse=True)
        return stable_modes

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

    def _compute_amplitudes(self, signal: np.ndarray, z: np.ndarray,
                            freq: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = len(signal)
        if len(z) == 0:
            return np.array([]), np.array([])
        V = np.zeros((N, len(z)), dtype=complex)
        t = np.arange(N)
        for i in range(len(z)):
            V[:, i] = z[i] ** t
        h, _, _, _ = lstsq(V, signal, cond=1e-12)
        amplitude = np.abs(h)
        phase = np.angle(h)
        return amplitude, phase

    def _reconstruct_signal(self, N: int, amplitude: np.ndarray, phase: np.ndarray,
                            freq: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        reconstructed = np.zeros(N)
        t_sec = np.arange(N) * self.dt
        for i in range(len(freq)):
            reconstructed += amplitude[i] * np.exp(sigma[i] * t_sec) * np.cos(
                2 * np.pi * freq[i] * t_sec + phase[i])
        return reconstructed

    def prony_analysis(self, signal: np.ndarray, order: Optional[int] = None,
                       method: str = 'tls_esprit',
                       use_stability: bool = False,
                       threshold: float = None,
                       **kwargs) -> Dict:
        N = len(signal)
        if method == 'tls_esprit':
            result = self._tls_esprit(signal, order=order, threshold=threshold)
        elif method == 'matrix_pencil':
            result = self._matrix_pencil_improved(signal, order=order, threshold=threshold)
        elif method == 'classic':
            return self._prony_classic(signal, order=order)
        else:
            result = self._tls_esprit(signal, order=order, threshold=threshold)
        if use_stability and method in ['tls_esprit', 'matrix_pencil']:
            stab_result = self._stability_diagram_analysis(
                signal, max_order=order if order else 30, method=method)
            stable_modes = stab_result['stable_modes']
            if len(stable_modes) > 0:
                result['freq'] = np.array([m['freq'] for m in stable_modes])
                result['damping_ratio'] = np.array([m['damping'] for m in stable_modes])
                result['sigma'] = np.array([m['sigma'] for m in stable_modes])
                result['z'] = np.exp(np.array([m['sigma'] for m in stable_modes]) * self.dt +
                                     1j * 2 * np.pi * np.array([m['freq'] for m in stable_modes]) * self.dt)
                result['stable_modes'] = stable_modes
        z = result['z']
        freq = result['freq']
        sigma = result['sigma']
        damping_ratio = result['damping_ratio']
        if len(z) == 0:
            return {
                'freq': np.array([]),
                'damping_ratio': np.array([]),
                'sigma': np.array([]),
                'amplitude': np.array([]),
                'phase': np.array([]),
                'reconstructed': np.zeros_like(signal),
                'order': order if order else 0,
                'method': method,
                'effective_rank': result.get('effective_rank', 0)
            }
        analytic = hilbert(signal)
        amplitude, phase = self._compute_amplitudes(analytic, z, freq, sigma)
        freq, damping_ratio, sigma, amplitude, phase = self._remove_conjugate_duplicates(
            freq, damping_ratio, sigma, amplitude, phase, tol=5e-3
        )
        sort_idx = np.argsort(amplitude)[::-1]
        freq = freq[sort_idx]
        damping_ratio = damping_ratio[sort_idx]
        sigma = sigma[sort_idx]
        amplitude = amplitude[sort_idx]
        phase = phase[sort_idx]
        reconstructed = self._reconstruct_signal(N, amplitude, phase, freq, sigma)
        return {
            'freq': freq,
            'damping_ratio': damping_ratio,
            'sigma': sigma,
            'amplitude': amplitude,
            'phase': phase,
            'reconstructed': reconstructed,
            'order': order if order else 0,
            'method': method,
            'effective_rank': result.get('effective_rank', 0),
            'singular_values': result.get('singular_values', None)
        }

    def _prony_classic(self, signal: np.ndarray, order: Optional[int] = None) -> Dict:
        N = len(signal)
        if order is None:
            order = min(N // 4, 12)
        order = min(order, N // 2 - 1)
        M = N - order
        y = signal[order:N]
        Y = toeplitz(signal[order-1:-1], signal[order-1::-1])
        a, _, _, _ = lstsq(Y, y, cond=1e-12)
        a_coeff = np.concatenate(([1.0], -a))
        roots = np.roots(a_coeff)
        roots = roots[np.abs(roots) > 1e-6]
        z = roots
        p = np.log(z + 1e-15) / self.dt
        sigma = np.real(p)
        omega = np.abs(np.imag(p))
        freq = omega / (2 * np.pi)
        damping_ratio = np.zeros_like(sigma)
        omega_nonzero = omega > 1e-6
        damping_ratio[omega_nonzero] = -sigma[omega_nonzero] / np.sqrt(
            sigma[omega_nonzero]**2 + omega[omega_nonzero]**2)
        valid_mask = (freq >= self.freq_min) & (freq <= self.freq_max) & \
                     (damping_ratio > 0) & (damping_ratio < 1.0) & \
                     (np.abs(z) < 1.0 + 1e-6)
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
                'reconstructed': np.zeros_like(signal),
                'order': order,
                'method': 'classic',
                'effective_rank': 0
            }
        analytic = hilbert(signal)
        amplitude, phase = self._compute_amplitudes(analytic, z, freq, sigma)
        freq, damping_ratio, sigma, amplitude, phase = self._remove_conjugate_duplicates(
            freq, damping_ratio, sigma, amplitude, phase, tol=1e-2
        )
        sort_idx = np.argsort(amplitude)[::-1]
        freq = freq[sort_idx]
        damping_ratio = damping_ratio[sort_idx]
        sigma = sigma[sort_idx]
        amplitude = amplitude[sort_idx]
        phase = phase[sort_idx]
        reconstructed = self._reconstruct_signal(N, amplitude, phase, freq, sigma)
        return {
            'freq': freq,
            'damping_ratio': damping_ratio,
            'sigma': sigma,
            'amplitude': amplitude,
            'phase': phase,
            'reconstructed': reconstructed,
            'order': order,
            'method': 'classic',
            'effective_rank': 0
        }

    def analyze(self, signal: np.ndarray, **kwargs) -> Dict:
        processed = self.preprocess_signal(signal)
        result = self.prony_analysis(processed, **kwargs)
        result['original'] = signal
        result['processed'] = processed
        result['time'] = np.arange(len(signal)) * self.dt
        return result


def generate_test_signal(fs: float = 100.0, duration: float = 10.0,
                         modes: List[Dict] = None,
                         noise_level: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
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
    print("\n" + "="*70)
    print(f"Prony Analysis Results - Method: {result.get('method', 'unknown').upper()}")
    print("="*70)
    print(f"{'Mode':<6} {'Frequency (Hz)':<18} {'Damping Ratio':<15} {'Amplitude':<12}")
    print("-"*70)
    for i, (f, d, a) in enumerate(zip(result['freq'], result['damping_ratio'],
                                      result['amplitude'])):
        print(f"{i+1:<6} {f:<18.4f} {d:<15.4f} {a:<12.4f}")
    print("="*70)
    if true_modes is not None:
        print("\nTrue Modes:")
        print(f"{'Mode':<6} {'Frequency (Hz)':<18} {'Damping Ratio':<15}")
        print("-"*50)
        for i, mode in enumerate(true_modes):
            print(f"{i+1:<6} {mode['freq']:<18.4f} {mode['damping']:<15.4f}")
        print("-"*50)


def plot_results(result: Dict, save_path: str = None):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
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
    ax2.set_title(f'Signal Reconstruction ({result.get("method", "").upper()})', fontsize=12, fontweight='bold')
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
    if result.get('singular_values') is not None:
        ax5 = fig.add_subplot(gs[2, :])
        s = result['singular_values']
        s_norm = s / s[0]
        ax5.semilogy(range(1, len(s)+1), s_norm, 'b-o', markersize=4, linewidth=2)
        ax5.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='1% threshold')
        ax5.set_xlabel('Singular Value Index', fontsize=11)
        ax5.set_ylabel('Normalized Singular Value', fontsize=11)
        ax5.set_title('Singular Value Spectrum (Noise Estimation)', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, linestyle='--')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    np.random.seed(42)
    fs = 100.0
    duration = 20.0
    print("="*70)
    print("电力系统低频振荡模式辨识 - Prony分析工具")
    print("使用 TLS-ESPRIT / 改进的矩阵束方法提高鲁棒性")
    print("="*70)
    print("\n生成测试信号（含3个振荡模式 + 噪声）...")
    true_modes = [
        {'freq': 0.4, 'damping': 0.05, 'amp': 1.0, 'phase': 0.0},
        {'freq': 1.2, 'damping': 0.08, 'amp': 0.6, 'phase': 0.3},
        {'freq': 0.8, 'damping': 0.03, 'amp': 0.4, 'phase': 0.1},
    ]
    t, signal = generate_test_signal(fs=fs, duration=duration, modes=true_modes, noise_level=0.03)
    analyzer = PronyAnalyzer(fs=fs, freq_range=(0.1, 2.0))
    print("\n" + "="*70)
    print("方法1: TLS-ESPRIT (总体最小二乘旋转不变技术)")
    print("="*70)
    result_tls = analyzer.analyze(signal, method='tls_esprit', order=30)
    print_results(result_tls, true_modes=true_modes)
    mse_tls = np.mean((result_tls['processed'] - result_tls['reconstructed'])**2)
    print(f"重构MSE: {mse_tls:.6f}, 有效秩: {result_tls.get('effective_rank', 'N/A')}")
    print("\n" + "="*70)
    print("方法2: 改进矩阵束方法 (Improved Matrix Pencil)")
    print("="*70)
    result_mp = analyzer.analyze(signal, method='matrix_pencil', order=30)
    print_results(result_mp, true_modes=true_modes)
    mse_mp = np.mean((result_mp['processed'] - result_mp['reconstructed'])**2)
    print(f"重构MSE: {mse_mp:.6f}, 有效秩: {result_mp.get('effective_rank', 'N/A')}")
    print("\n" + "="*70)
    print("方法3: TLS-ESPRIT + 稳定性图验证")
    print("="*70)
    result_stab = analyzer.analyze(signal, method='tls_esprit', order=30, use_stability=True)
    print_results(result_stab, true_modes=true_modes)
    mse_stab = np.mean((result_stab['processed'] - result_stab['reconstructed'])**2)
    print(f"重构MSE: {mse_stab:.6f}")
    print("\n" + "="*70)
    print("对比总结:")
    print("="*70)
    print(f"{'方法':<25} {'MSE':<15} {'检测模式数':<15}")
    print("-"*55)
    print(f"{'TLS-ESPRIT':<25} {mse_tls:<15.6f} {len(result_tls['freq']):<15}")
    print(f"{'Improved Matrix Pencil':<25} {mse_mp:<15.6f} {len(result_mp['freq']):<15}")
    print(f"{'TLS-ESPRIT + Stability':<25} {mse_stab:<15.6f} {len(result_stab['freq']):<15}")
    print("="*70)
    plot_results(result_tls, save_path='prony_analysis_plot.png')


if __name__ == '__main__':
    main()
