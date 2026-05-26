import numpy as np
from scipy.linalg import svd, lstsq, eig
from scipy.signal import detrend, butter, filtfilt, hilbert
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class RecursivePronyAnalyzer:
    def __init__(self, fs: float = 100.0, freq_range: Tuple[float, float] = (0.1, 2.0),
                 window_size: int = 200, overlap: int = 150):
        self.fs = fs
        self.dt = 1.0 / fs
        self.freq_min, self.freq_max = freq_range
        self.window_size = window_size
        self.overlap = overlap
        self.step = window_size - overlap
        self.signal_buffer = deque(maxlen=window_size * 2)
        self.mode_history = []
        self.sample_count = 0

    def update(self, new_sample: float) -> Optional[Dict]:
        self.signal_buffer.append(new_sample)
        self.sample_count += 1
        if len(self.signal_buffer) >= self.window_size:
            if self.sample_count % self.step == 0 or len(self.signal_buffer) == self.window_size:
                signal = np.array(list(self.signal_buffer)[-self.window_size:])
                result = self._analyze_window(signal)
                if result is not None and len(result['freq']) > 0:
                    self.mode_history.append(result)
                    return result
        return None

    def _analyze_window(self, signal: np.ndarray) -> Optional[Dict]:
        N = len(signal)
        try:
            signal = detrend(signal, type='linear')
            analytic = hilbert(signal)
            L = min(N // 3, 60)
            H = self._hankel_matrix(analytic, L)
            H1 = H[:, :-1]
            H2 = H[:, 1:]
            U, s, Vh = svd(H1, full_matrices=False)
            threshold = 2.858 * np.median(s)
            M = np.sum(s > threshold)
            M = max(M, 4)
            M = min(M, L - 1)
            U1 = U[:, :M]
            V1 = Vh[:M, :].conj().T
            S1_inv = np.diag(1.0 / s[:M])
            Y = V1 @ S1_inv @ U1.conj().T @ H2
            eigvals, _ = eig(Y)
            result = self._extract_modes(eigvals, analytic, M)
            if result is not None and len(result['freq']) > 0:
                result['timestamp'] = self.sample_count * self.dt
                return result
        except:
            pass
        return None

    def _hankel_matrix(self, signal: np.ndarray, L: int) -> np.ndarray:
        N = len(signal)
        K = N - L + 1
        H = np.zeros((L, K), dtype=complex)
        for i in range(K):
            H[:, i] = signal[i:i + L]
        return H

    def _extract_modes(self, eigvals: np.ndarray, signal: np.ndarray,
                       model_order: int) -> Optional[Dict]:
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
                     (damping_ratio > 0) & (damping_ratio < 0.5) & \
                     (np.abs(z) > 0.85) & (np.abs(z) < 1.05)
        sigma = sigma[valid_mask]
        freq = freq[valid_mask]
        damping_ratio = damping_ratio[valid_mask]
        z = z[valid_mask]
        if len(z) == 0:
            return None
        V = np.zeros((N, len(z)), dtype=complex)
        t = np.arange(N)
        for i in range(len(z)):
            V[:, i] = z[i] ** t
        h, _, _, _ = lstsq(V, signal, cond=1e-12)
        amplitude = np.abs(h)
        phase = np.angle(h)
        freq, damping_ratio, sigma, amplitude, phase = self._remove_conjugate_duplicates(
            freq, damping_ratio, sigma, amplitude, phase, tol=5e-3)
        sort_idx = np.argsort(amplitude)[::-1]
        return {
            'freq': freq[sort_idx],
            'damping_ratio': damping_ratio[sort_idx],
            'sigma': sigma[sort_idx],
            'amplitude': amplitude[sort_idx],
            'phase': phase[sort_idx]
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


class KalmanModeTracker:
    def __init__(self, n_modes: int = 3, fs: float = 100.0,
                 process_noise: float = 0.0001, measurement_noise: float = 0.01):
        self.n_modes = n_modes
        self.fs = fs
        self.dt = 1.0 / fs
        self.state_dim = n_modes * 4
        self.Q = np.eye(self.state_dim) * process_noise
        self.R = np.eye(n_modes * 2) * measurement_noise
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 0.1
        self.initialized = False
        self.tracked_frequencies = []
        self.tracked_damping = []
        self.tracked_amplitude = []
        self.tracked_phase = []

    def initialize(self, modes: Dict):
        n_detected = min(len(modes['freq']), self.n_modes)
        sorted_idx = np.argsort(modes.get('amplitude', np.ones(n_detected)))[::-1]
        sorted_freqs = modes['freq'][sorted_idx]
        sorted_damping = modes['damping_ratio'][sorted_idx]
        sorted_amp = modes['amplitude'][sorted_idx]
        sorted_phase = modes['phase'][sorted_idx]
        for i in range(self.n_modes):
            if i < n_detected:
                self.x[4*i] = sorted_freqs[i]
                self.x[4*i+1] = sorted_damping[i]
                self.x[4*i+2] = sorted_amp[i]
                self.x[4*i+3] = sorted_phase[i]
            else:
                self.x[4*i] = 0.3 + i * 0.5
                self.x[4*i+1] = 0.05
                self.x[4*i+2] = 0.1
                self.x[4*i+3] = 0.0
        self.tracked_frequencies = [self.x[4*i] for i in range(self.n_modes)]
        self.initialized = True

    def _match_modes(self, measurement: Dict) -> np.ndarray:
        measured_freqs = measurement['freq']
        n_measured = len(measured_freqs)
        mapping = np.zeros(self.n_modes, dtype=int)
        if n_measured == 0:
            return mapping
        cost_matrix = np.zeros((self.n_modes, n_measured))
        for i in range(self.n_modes):
            for j in range(n_measured):
                freq_diff = abs(measured_freqs[j] - self.tracked_frequencies[i])
                amp = measurement['amplitude'][j] if 'amplitude' in measurement else 1.0
                cost_matrix[i, j] = freq_diff + 0.1 * (1.0 / max(amp, 0.01))
        used_measured = set()
        sorted_tracked = np.argsort([self.tracked_frequencies[i] for i in range(self.n_modes)])
        for i in sorted_tracked:
            valid_j = [j for j in range(n_measured) if j not in used_measured]
            if valid_j:
                best_j = min(valid_j, key=lambda j: cost_matrix[i, j])
                mapping[i] = best_j
                used_measured.add(best_j)
            else:
                mapping[i] = min(i, n_measured - 1)
        return mapping

    def predict(self):
        F = np.eye(self.state_dim)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: Dict):
        if not self.initialized:
            self.initialize(measurement)
            return self.get_state()
        if len(measurement['freq']) == 0:
            self.predict()
            return self.get_state()
        self.predict()
        mapping = self._match_modes(measurement)
        z = np.zeros(self.n_modes * 2)
        H = np.zeros((self.n_modes * 2, self.state_dim))
        for i in range(self.n_modes):
            j = mapping[i]
            z[2*i] = measurement['freq'][j]
            z[2*i+1] = measurement['damping_ratio'][j]
            H[2*i, 4*i] = 1.0
            H[2*i+1, 4*i+1] = 1.0
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
            self.x = self.x + K @ y
            self.P = (np.eye(self.state_dim) - K @ H) @ self.P
        except:
            pass
        for i in range(self.n_modes):
            j = mapping[i]
            if j < len(measurement['freq']):
                alpha = 0.2
                self.tracked_frequencies[i] = (1 - alpha) * self.tracked_frequencies[i] + \
                                              alpha * measurement['freq'][j]
        return self.get_state()

    def get_state(self) -> Dict:
        freq = []
        damping = []
        amplitude = []
        phase = []
        for i in range(self.n_modes):
            freq.append(max(0.01, min(2.0, self.x[4*i])))
            damping.append(max(0.001, min(0.5, self.x[4*i+1])))
            amplitude.append(max(0.001, self.x[4*i+2]))
            phase.append(self.x[4*i+3])
        return {
            'freq': np.array(freq),
            'damping_ratio': np.array(damping),
            'amplitude': np.array(amplitude),
            'phase': np.array(phase)
        }


class PSSParameterTuner:
    def __init__(self):
        self.pss_types = {
            'PSS1A': {'K': 10.0, 'T1': 0.154, 'T2': 0.033, 'T3': 0.033, 'T4': 0.154},
            'PSS2A': {'K': 15.0, 'T1': 0.154, 'T2': 0.033, 'T3': 0.033, 'T4': 0.154},
            'PSS4B': {'K': 8.0, 'T1': 0.2, 'T2': 0.05, 'T3': 0.05, 'T4': 0.2},
        }

    def analyze_stability(self, modes: Dict) -> Dict:
        stability_info = {
            'modes': [],
            'overall_stability': True,
            'recommendations': []
        }
        freqs = modes['freq']
        damps = modes['damping_ratio']
        amps = modes.get('amplitude', np.ones_like(freqs))
        for i, (f, d, a) in enumerate(zip(freqs, damps, amps)):
            mode_info = {
                'frequency': f,
                'damping': d,
                'amplitude': a,
                'category': self._categorize_mode(f, d),
                'status': self._assess_mode(d)
            }
            stability_info['modes'].append(mode_info)
            if mode_info['status'] != 'good':
                stability_info['overall_stability'] = False
                stability_info['recommendations'].append(
                    self._get_recommendation(f, d)
                )
        return stability_info

    def _categorize_mode(self, freq: float, damping: float) -> str:
        if freq < 0.8:
            return '区域振荡模式 (Inter-area)'
        else:
            return '本地振荡模式 (Local)'

    def _assess_mode(self, damping: float) -> str:
        if damping >= 0.1:
            return 'good'
        elif damping >= 0.05:
            return 'acceptable'
        else:
            return 'poor'

    def _get_recommendation(self, freq: float, damping: float) -> str:
        if damping < 0.05:
            return f"模式 {freq:.2f}Hz 阻尼比不足 ({damping:.4f})，建议增加PSS增益或调整相位补偿"
        elif damping < 0.1:
            return f"模式 {freq:.2f}Hz 阻尼比偏低 ({damping:.4f})，建议优化PSS参数"
        else:
            return f"模式 {freq:.2f}Hz 阻尼比良好 ({damping:.4f})"

    def tune_pss_parameters(self, modes: Dict, pss_type: str = 'PSS2A',
                            target_damping: float = 0.15) -> Dict:
        if pss_type not in self.pss_types:
            pss_type = 'PSS2A'
        base_params = self.pss_types[pss_type].copy()
        tuning_result = {
            'original_params': base_params.copy(),
            'tuned_params': {},
            'analysis': []
        }
        for i, (f, d) in enumerate(zip(modes['freq'], modes['damping_ratio'])):
            analysis = {
                'mode': f'Mode {i+1}',
                'frequency': f,
                'original_damping': d,
                'target_damping': target_damping,
                'required_action': ''
            }
            if d < target_damping:
                gain_factor = target_damping / max(d, 0.001)
                tuned_K = base_params['K'] * gain_factor
                tuned_K = min(tuned_K, 50.0)
                if f < 0.8:
                    analysis['required_action'] = '区域模式 - 建议增加PSS增益并优化相位补偿'
                else:
                    analysis['required_action'] = '本地模式 - 建议微调PSS参数'
            else:
                analysis['required_action'] = '阻尼充足 - 无需调整'
            tuning_result['analysis'].append(analysis)
        tuning_result['tuned_params'] = base_params.copy()
        if not tuning_result['tuned_params']:
            tuning_result['tuned_params'] = base_params.copy()
        return tuning_result

    def generate_pss_report(self, modes: Dict, stability_info: Dict,
                            tuning_result: Dict) -> str:
        report = []
        report.append("="*70)
        report.append("电力系统稳定器(PSS)参数整定报告")
        report.append("="*70)
        report.append("\n1. 振荡模式分析:")
        report.append("-"*50)
        for i, mode_info in enumerate(stability_info['modes']):
            report.append(f"\n  模式 {i+1}:")
            report.append(f"    频率: {mode_info['frequency']:.4f} Hz")
            report.append(f"    阻尼比: {mode_info['damping']:.4f}")
            report.append(f"    类型: {mode_info['category']}")
            report.append(f"    状态: {'良好' if mode_info['status'] == 'good' else '需要关注'}")
        report.append("\n2. 稳定性评估:")
        report.append("-"*50)
        if stability_info['overall_stability']:
            report.append("  系统整体稳定性: 良好")
        else:
            report.append("  系统整体稳定性: 需要改善")
            report.append("\n  建议措施:")
            for rec in stability_info['recommendations']:
                report.append(f"    - {rec}")
        report.append("\n3. PSS参数整定建议:")
        report.append("-"*50)
        for analysis in tuning_result['analysis']:
            report.append(f"\n  {analysis['mode']}:")
            report.append(f"    频率: {analysis['frequency']:.4f} Hz")
            report.append(f"    原阻尼比: {analysis['original_damping']:.4f}")
            report.append(f"    建议: {analysis['required_action']}")
        report.append("\n" + "="*70)
        return "\n".join(report)


class WAMSOnlineAnalyzer:
    def __init__(self, fs: float = 100.0, freq_range: Tuple[float, float] = (0.1, 2.0),
                 window_size: int = 200, overlap: int = 150, n_modes: int = 3):
        self.fs = fs
        self.freq_range = freq_range
        self.recursive_prony = RecursivePronyAnalyzer(fs, freq_range, window_size, overlap)
        self.kalman_tracker = KalmanModeTracker(n_modes, fs)
        self.pss_tuner = PSSParameterTuner()
        self.time_history = []
        self.freq_history = []
        self.damping_history = []
        self.amp_history = []
        self.is_initialized = False
        self.n_modes = n_modes

    def process_sample(self, timestamp: float, measurement: float) -> Optional[Dict]:
        result = self.recursive_prony.update(measurement)
        if result is not None:
            tracked = self.kalman_tracker.update(result)
            self.time_history.append(timestamp)
            self.freq_history.append(tracked['freq'].copy())
            self.damping_history.append(tracked['damping_ratio'].copy())
            self.amp_history.append(tracked['amplitude'].copy())
            self.is_initialized = True
            stability = self.pss_tuner.analyze_stability(tracked)
            tuning = self.pss_tuner.tune_pss_parameters(tracked)
            return {
                'timestamp': timestamp,
                'tracked_modes': tracked,
                'raw_modes': result,
                'stability': stability,
                'tuning': tuning
            }
        return None

    def process_batch(self, signal: np.ndarray, t: np.ndarray) -> List[Dict]:
        results = []
        for i, (time_val, sample) in enumerate(zip(t, signal)):
            result = self.process_sample(time_val, sample)
            if result is not None:
                results.append(result)
        return results

    def get_current_modes(self) -> Optional[Dict]:
        if self.is_initialized and len(self.freq_history) > 0:
            return {
                'freq': self.freq_history[-1],
                'damping_ratio': self.damping_history[-1],
                'amplitude': self.amp_history[-1] if len(self.amp_history) > 0 else np.ones_like(self.freq_history[-1]),
                'phase': np.zeros_like(self.freq_history[-1])
            }
        return None

    def get_tracking_history(self) -> Dict:
        if len(self.time_history) == 0:
            return {'time': [], 'freq': [], 'damping': []}
        return {
            'time': np.array(self.time_history),
            'freq': np.array(self.freq_history),
            'damping': np.array(self.damping_history)
        }


def simulate_wams_data(fs: float = 100.0, duration: float = 30.0,
                       modes: List[Dict] = None, noise_level: float = 0.01,
                       change_time: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fs
    t = np.arange(0, duration, dt)
    signal = np.zeros_like(t)
    if modes is None:
        modes = [
            {'freq': 0.4, 'damping': 0.05, 'amp': 1.0, 'phase': 0.0},
            {'freq': 1.2, 'damping': 0.08, 'amp': 0.5, 'phase': 0.3},
        ]
    for i, mode in enumerate(modes):
        if i == 0:
            sigma1 = -2 * np.pi * mode['freq'] * mode['damping']
            sigma2 = -2 * np.pi * mode['freq'] * (mode['damping'] * 0.6)
            mask = t < change_time
            signal[mask] += mode['amp'] * np.exp(sigma1 * t[mask]) * np.cos(
                2 * np.pi * mode['freq'] * t[mask] + mode['phase'])
            signal[~mask] += mode['amp'] * 0.8 * np.exp(sigma2 * t[~mask]) * np.cos(
                2 * np.pi * mode['freq'] * t[~mask] + mode['phase'])
        else:
            sigma = -2 * np.pi * mode['freq'] * mode['damping']
            signal += mode['amp'] * np.exp(sigma * t) * np.cos(
                2 * np.pi * mode['freq'] * t + mode['phase'])
    signal += noise_level * np.random.randn(len(t))
    return t, signal


def plot_wams_results(tracking_history: Dict, raw_signal: np.ndarray, t: np.ndarray,
                      true_modes: List[Dict], save_path: str = None):
    time = tracking_history['time']
    freq = tracking_history['freq']
    damping = tracking_history['damping']
    if len(time) == 0:
        print("No tracking history to plot")
        return
    n_modes = freq.shape[1]
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, raw_signal, 'b-', alpha=0.7, linewidth=0.8)
    ax0.set_xlabel('Time (s)', fontsize=11)
    ax0.set_ylabel('Signal', fontsize=11)
    ax0.set_title('WAMS Measurement Signal', fontsize=12, fontweight='bold')
    ax0.grid(True, alpha=0.3)
    ax1 = fig.add_subplot(gs[1, 0])
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for i in range(n_modes):
        ax1.plot(time, freq[:, i], '-', color=colors[i % len(colors)],
                linewidth=1.5, label=f'Tracked Mode {i+1}')
        if i < len(true_modes):
            ax1.axhline(y=true_modes[i]['freq'], color=colors[i % len(colors)],
                       linestyle='--', alpha=0.5, linewidth=1, label=f'True {true_modes[i]["freq"]:.1f}Hz')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Frequency (Hz)', fontsize=11)
    ax1.set_title('Frequency Tracking', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 2.5)
    ax2 = fig.add_subplot(gs[1, 1])
    for i in range(n_modes):
        ax2.plot(time, damping[:, i], '-', color=colors[i % len(colors)],
                linewidth=1.5, label=f'Mode {i+1}')
    ax2.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Minimum (0.05)')
    ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Target (0.1)')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Damping Ratio', fontsize=11)
    ax2.set_title('Damping Ratio Tracking', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.3)
    ax3 = fig.add_subplot(gs[2, 0])
    for i in range(n_modes):
        if i < len(true_modes):
            true_freq = true_modes[i]['freq']
            freq_error = np.abs(freq[:, i] - true_freq) / true_freq * 100
            ax3.plot(time, freq_error, '-', color=colors[i % len(colors)],
                    linewidth=1.5, label=f'Mode {i+1}')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Frequency Error (%)', fontsize=11)
    ax3.set_title('Frequency Tracking Error', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax4 = fig.add_subplot(gs[2, 1])
    mode_freqs = [freq[:, i] for i in range(n_modes)]
    mode_damps = [damping[:, i] for i in range(n_modes)]
    if len(time) > 0:
        scatter = ax4.scatter(np.concatenate(mode_freqs), np.concatenate(mode_damps),
                             c=np.tile(time, n_modes), cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax4, label='Time (s)')
    ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Minimum damping')
    ax4.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Target damping')
    ax4.set_xlabel('Frequency (Hz)', fontsize=11)
    ax4.set_ylabel('Damping Ratio', fontsize=11)
    ax4.set_title('Mode Trajectory in Frequency-Damping Plane', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 2.5)
    ax4.set_ylim(0, 0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    print("="*70)
    print("广域量测系统(WAMS)在线振荡模式辨识")
    print("递归Prony + 卡尔曼滤波实时跟踪")
    print("电力系统稳定器(PSS)参数整定")
    print("="*70)

    fs = 100.0
    duration = 30.0
    window_size = 150
    overlap = 125

    print(f"\n模拟参数:")
    print(f"  采样频率: {fs} Hz")
    print(f"  仿真时长: {duration} s")
    print(f"  分析窗口: {window_size} 采样点 ({window_size/fs:.1f} s)")
    print(f"  窗口重叠: {overlap} 采样点 ({overlap/fs:.1f} s)")

    true_modes = [
        {'freq': 0.4, 'damping': 0.06, 'amp': 1.0, 'phase': 0.0},
        {'freq': 1.2, 'damping': 0.08, 'amp': 0.5, 'phase': 0.3},
    ]
    print(f"\n真实振荡模式:")
    for i, mode in enumerate(true_modes):
        print(f"  模式 {i+1}: {mode['freq']:.2f} Hz, 阻尼比 = {mode['damping']:.3f}")

    print(f"\n生成模拟WAMS数据（含15s处阻尼变化）...")
    np.random.seed(42)
    t, signal = simulate_wams_data(fs=fs, duration=duration,
                                    modes=true_modes, noise_level=0.015,
                                    change_time=15.0)

    analyzer = WAMSOnlineAnalyzer(fs=fs, window_size=window_size, overlap=overlap, n_modes=2)

    print(f"\n开始在线分析...")
    print(f"{'时间(s)':<10} {'模式1频率':<12} {'模式1阻尼':<12} {'模式2频率':<12} {'模式2阻尼':<12} {'状态'}")
    print("-"*75)

    results = []
    report_generated = False
    last_print_time = -1

    for i, (time_val, sample) in enumerate(zip(t, signal)):
        result = analyzer.process_sample(time_val, sample)
        if result is not None:
            results.append(result)
            tracked = result['tracked_modes']
            stability = result['stability']
            status = "稳定" if stability['overall_stability'] else "不稳定"
            if abs(time_val - last_print_time) >= 2.0 or last_print_time < 0:
                f1 = tracked['freq'][0] if len(tracked['freq']) > 0 else 0
                d1 = tracked['damping_ratio'][0] if len(tracked['damping_ratio']) > 0 else 0
                f2 = tracked['freq'][1] if len(tracked['freq']) > 1 else 0
                d2 = tracked['damping_ratio'][1] if len(tracked['damping_ratio']) > 1 else 0
                print(f"{time_val:<10.1f} {f1:<12.4f} {d1:<12.4f} {f2:<12.4f} {d2:<12.4f} {status}")
                last_print_time = time_val
            if not stability['overall_stability'] and not report_generated and time_val > 20:
                tuning = result['tuning']
                report = analyzer.pss_tuner.generate_pss_report(tracked, stability, tuning)
                print(f"\n{report}")
                report_generated = True

    print(f"\n分析完成!")
    print(f"  总分析窗口数: {len(results)}")

    tracking_history = analyzer.get_tracking_history()
    if len(tracking_history['time']) > 0:
        print(f"\n最终跟踪结果:")
        final_freq = tracking_history['freq'][-1]
        final_damping = tracking_history['damping'][-1]
        for i in range(len(final_freq)):
            true_mode = min(true_modes, key=lambda m: abs(m['freq'] - final_freq[i]))
            freq_err = abs(final_freq[i] - true_mode['freq']) / true_mode['freq'] * 100
            print(f"  模式 {i+1}: 频率 = {final_freq[i]:.4f} Hz (真实: {true_mode['freq']:.4f} Hz, 误差: {freq_err:.2f}%), "
                  f"阻尼比 = {final_damping[i]:.4f} (真实: {true_mode['damping']:.4f})")

        plot_wams_results(tracking_history, signal, t, true_modes, save_path='wams_analysis_result.png')

    print(f"\n当前系统状态:")
    current_modes = analyzer.get_current_modes()
    if current_modes:
        stability = analyzer.pss_tuner.analyze_stability(current_modes)
        print(f"  整体稳定性: {'良好' if stability['overall_stability'] else '需要改善'}")
        for mode_info in stability['modes']:
            status_text = '良好' if mode_info['status'] == 'good' else ('可接受' if mode_info['status'] == 'acceptable' else '不足')
            print(f"  {mode_info['category']}: {mode_info['frequency']:.3f} Hz, 阻尼比 = {mode_info['damping']:.4f} ({status_text})")
        for rec in stability['recommendations']:
            print(f"  建议: {rec}")


if __name__ == '__main__':
    main()
