import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import find_peaks, butter, filtfilt, medfilt, stft


class OrderTracking:
    def __init__(self, fs, vibration_signal, tach_signal=None, rpm_signal=None):
        self.fs = fs
        self.vibration_signal = np.asarray(vibration_signal)
        self.tach_signal = np.asarray(tach_signal) if tach_signal is not None else None
        self.rpm_signal = np.asarray(rpm_signal) if rpm_signal is not None else None
        self.time_axis = np.arange(len(vibration_signal)) / fs
        self.angular_axis = None
        self.resampled_signal = None
        self.instantaneous_rpm = None
        self.angle = None
        self.corrected_pulses = None
        self.pulse_corrections = []

    def extract_tach_pulses(self, threshold=None, min_distance=10):
        if self.tach_signal is None:
            raise ValueError("No tachometer signal provided")

        if threshold is None:
            threshold = np.mean(self.tach_signal) + 2 * np.std(self.tach_signal)

        peaks, _ = find_peaks(self.tach_signal, height=threshold, distance=min_distance)
        return peaks

    def detect_anomalous_pulses(self, pulse_indices, threshold_factor=3.0, method='mad'):
        if len(pulse_indices) < 5:
            return pulse_indices / self.fs, []

        pulse_times = pulse_indices / self.fs
        intervals = np.diff(pulse_times)

        if method == 'mad':
            median_interval = np.median(intervals)
            mad = np.median(np.abs(intervals - median_interval))
            threshold = threshold_factor * mad
            anomalies = np.where(np.abs(intervals - median_interval) > threshold)[0]
        elif method == 'std':
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            threshold = threshold_factor * std_interval
            anomalies = np.where(np.abs(intervals - mean_interval) > threshold)[0]
        elif method == 'percentile':
            q1 = np.percentile(intervals, 25)
            q3 = np.percentile(intervals, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold_factor * iqr
            upper_bound = q3 + threshold_factor * iqr
            anomalies = np.where((intervals < lower_bound) | (intervals > upper_bound))[0]
        else:
            raise ValueError(f"Unknown method: {method}")

        return pulse_times, anomalies

    def interpolate_missing_pulses(self, pulse_times, anomalies, pulses_per_rev=1):
        if len(anomalies) == 0:
            return pulse_times, []

        corrected_times = list(pulse_times)
        corrections = []
        offset = 0

        anomalies_sorted = sorted(anomalies)

        for anomaly_idx in anomalies_sorted:
            adjusted_idx = anomaly_idx + offset

            if adjusted_idx >= len(corrected_times) - 1:
                continue

            t1 = corrected_times[adjusted_idx]
            t2 = corrected_times[adjusted_idx + 1]
            interval = t2 - t1

            if adjusted_idx > 0 and adjusted_idx < len(corrected_times) - 2:
                prev_interval = corrected_times[adjusted_idx] - corrected_times[adjusted_idx - 1]
                next_interval = corrected_times[adjusted_idx + 2] - corrected_times[adjusted_idx + 1]
                expected_interval = (prev_interval + next_interval) / 2
            elif adjusted_idx > 0:
                expected_interval = corrected_times[adjusted_idx] - corrected_times[adjusted_idx - 1]
            else:
                expected_interval = corrected_times[adjusted_idx + 2] - corrected_times[adjusted_idx + 1]

            if interval > 1.5 * expected_interval:
                num_missing = int(round(interval / expected_interval)) - 1
                for i in range(1, num_missing + 1):
                    new_time = t1 + i * (interval / (num_missing + 1))
                    corrected_times.insert(adjusted_idx + i, new_time)
                    corrections.append({
                        'type': 'inserted',
                        'time': new_time,
                        'index': adjusted_idx + i
                    })
                offset += num_missing
            elif interval < 0.5 * expected_interval:
                del corrected_times[adjusted_idx + 1]
                corrections.append({
                    'type': 'deleted',
                    'time': t2,
                    'index': adjusted_idx + 1
                })
                offset -= 1

        return np.array(corrected_times), corrections

    def smooth_rpm_with_outlier_rejection(self, pulse_times, rpm_values, smooth_window=5):
        if len(rpm_values) < smooth_window:
            return rpm_values

        rpm_median = medfilt(rpm_values, kernel_size=smooth_window)

        residuals = np.abs(rpm_values - rpm_median)
        mad = np.median(residuals)
        threshold = 3 * mad if mad > 0 else 0.1 * np.mean(rpm_values)

        mask = residuals <= threshold
        clean_rpm = rpm_values.copy()
        clean_rpm[~mask] = rpm_median[~mask]

        return clean_rpm

    def calculate_instantaneous_rpm(self, pulses_per_rev=1, smooth=True, window_size=10):
        if self.tach_signal is not None:
            pulse_indices = self.extract_tach_pulses()
            pulse_times = pulse_indices / self.fs

            if len(pulse_times) < 2:
                raise ValueError("Not enough tach pulses detected")

            time_intervals = np.diff(pulse_times)
            rpm_values = 60 / (pulses_per_rev * time_intervals)

            rpm_interp = interp1d(
                pulse_times[:-1], rpm_values, kind='linear',
                fill_value='extrapolate', bounds_error=False
            )
            self.instantaneous_rpm = rpm_interp(self.time_axis)

            if smooth and len(self.instantaneous_rpm) > window_size:
                kernel = np.ones(window_size) / window_size
                self.instantaneous_rpm = np.convolve(self.instantaneous_rpm, kernel, mode='same')

        elif self.rpm_signal is not None:
            if len(self.rpm_signal) != len(self.time_axis):
                rpm_interp = interp1d(
                    np.linspace(0, self.time_axis[-1], len(self.rpm_signal)),
                    self.rpm_signal, kind='linear',
                    fill_value='extrapolate', bounds_error=False
                )
                self.instantaneous_rpm = rpm_interp(self.time_axis)
            else:
                self.instantaneous_rpm = self.rpm_signal
        else:
            raise ValueError("Either tach_signal or rpm_signal must be provided")

        return self.instantaneous_rpm

    def calculate_instantaneous_rpm_robust(self, pulses_per_rev=1,
                                           anomaly_threshold=3.0,
                                           anomaly_method='mad',
                                           smooth_method='spline',
                                           smooth_window=10):
        if self.tach_signal is not None:
            pulse_indices = self.extract_tach_pulses()
            pulse_times = pulse_indices / self.fs

            if len(pulse_times) < 5:
                raise ValueError("Not enough tach pulses detected")

            pulse_times, anomalies = self.detect_anomalous_pulses(
                pulse_indices, threshold_factor=anomaly_threshold, method=anomaly_method
            )

            pulse_times, corrections = self.interpolate_missing_pulses(
                pulse_times, anomalies, pulses_per_rev
            )

            self.corrected_pulses = pulse_times
            self.pulse_corrections = corrections

            time_intervals = np.diff(pulse_times)
            rpm_values = 60 / (pulses_per_rev * time_intervals)

            rpm_values = self.smooth_rpm_with_outlier_rejection(
                pulse_times[:-1], rpm_values, smooth_window=min(5, len(rpm_values)//2)
            )

            if smooth_method == 'spline':
                rpm_spline = UnivariateSpline(
                    pulse_times[:-1], rpm_values, s=len(rpm_values) * 0.1, k=3
                )
                self.instantaneous_rpm = rpm_spline(self.time_axis)
            elif smooth_method == 'linear':
                rpm_interp = interp1d(
                    pulse_times[:-1], rpm_values, kind='linear',
                    fill_value='extrapolate', bounds_error=False
                )
                self.instantaneous_rpm = rpm_interp(self.time_axis)
            elif smooth_method == 'median':
                rpm_interp = interp1d(
                    pulse_times[:-1], rpm_values, kind='linear',
                    fill_value='extrapolate', bounds_error=False
                )
                self.instantaneous_rpm = medfilt(rpm_interp(self.time_axis), smooth_window)
            else:
                raise ValueError(f"Unknown smooth method: {smooth_method}")

        elif self.rpm_signal is not None:
            if len(self.rpm_signal) != len(self.time_axis):
                rpm_interp = interp1d(
                    np.linspace(0, self.time_axis[-1], len(self.rpm_signal)),
                    self.rpm_signal, kind='linear',
                    fill_value='extrapolate', bounds_error=False
                )
                self.instantaneous_rpm = rpm_interp(self.time_axis)
            else:
                self.instantaneous_rpm = self.rpm_signal
        else:
            raise ValueError("Either tach_signal or rpm_signal must be provided")

        return self.instantaneous_rpm

    def calculate_angular_displacement(self):
        if self.instantaneous_rpm is None:
            self.calculate_instantaneous_rpm()

        angular_velocity = 2 * np.pi * self.instantaneous_rpm / 60
        dt = 1 / self.fs
        self.angle = np.cumsum(angular_velocity) * dt
        return self.angle

    def calculate_angular_displacement_robust(self, method='integral'):
        if self.instantaneous_rpm is None:
            self.calculate_instantaneous_rpm_robust()

        if method == 'integral':
            angular_velocity = 2 * np.pi * self.instantaneous_rpm / 60
            dt = 1 / self.fs
            self.angle = np.cumsum(angular_velocity) * dt
        elif method == 'pulse_based' and self.corrected_pulses is not None:
            pulse_angles = np.arange(len(self.corrected_pulses)) * 2 * np.pi

            angle_interp = interp1d(
                self.corrected_pulses, pulse_angles, kind='cubic',
                fill_value='extrapolate', bounds_error=False
            )
            self.angle = angle_interp(self.time_axis)
        else:
            angular_velocity = 2 * np.pi * self.instantaneous_rpm / 60
            dt = 1 / self.fs
            self.angle = np.cumsum(angular_velocity) * dt

        return self.angle

    def compute_order_tracking(self, orders_per_rev=360, method='linear'):
        if self.angle is None:
            self.calculate_angular_displacement()

        total_angle = self.angle[-1]
        num_samples = int(total_angle * orders_per_rev / (2 * np.pi))

        self.angular_axis = np.linspace(0, total_angle, num_samples)

        interpolator = interp1d(
            self.angle, self.vibration_signal, kind=method,
            fill_value='extrapolate', bounds_error=False
        )
        self.resampled_signal = interpolator(self.angular_axis)

        return self.angular_axis, self.resampled_signal

    def phase_corrected_resampling(self, orders_per_rev=360, method='cubic'):
        if self.angle is None:
            self.calculate_angular_displacement_robust()

        angle_diff = np.diff(self.angle)
        angle_grad = np.abs(np.gradient(angle_diff))
        jump_threshold = np.mean(angle_grad) + 5 * np.std(angle_grad)
        jump_indices = np.where(angle_grad > jump_threshold)[0]

        if len(jump_indices) > 0:
            for jump_idx in jump_indices:
                if jump_idx < 10 or jump_idx > len(self.angle) - 10:
                    continue
                before = self.angle[max(0, jump_idx - 10):jump_idx]
                after = self.angle[jump_idx:min(len(self.angle), jump_idx + 10)]
                jump_size = np.mean(after) - np.mean(before)
                expected_jump = np.mean(np.diff(before[-5:])) * 10
                if abs(jump_size - expected_jump) > 0.5 * expected_jump:
                    correction = np.linspace(0, jump_size - expected_jump, len(self.angle) - jump_idx)
                    self.angle[jump_idx:] -= correction

        total_angle = self.angle[-1]
        num_samples = int(total_angle * orders_per_rev / (2 * np.pi))

        self.angular_axis = np.linspace(0, total_angle, num_samples)

        interpolator = interp1d(
            self.angle, self.vibration_signal, kind=method,
            fill_value='extrapolate', bounds_error=False
        )
        self.resampled_signal = interpolator(self.angular_axis)

        return self.angular_axis, self.resampled_signal

    def compute_order_tracking_robust(self, orders_per_rev=360,
                                       anomaly_threshold=3.0,
                                       anomaly_method='mad',
                                       smooth_method='spline',
                                       resample_method='cubic'):
        self.calculate_instantaneous_rpm_robust(
            anomaly_threshold=anomaly_threshold,
            anomaly_method=anomaly_method,
            smooth_method=smooth_method
        )

        self.calculate_angular_displacement_robust()

        return self.phase_corrected_resampling(
            orders_per_rev=orders_per_rev,
            method=resample_method
        )

    def get_order_spectrum(self, window='hann'):
        if self.resampled_signal is None:
            raise ValueError("Perform order tracking first")

        n = len(self.resampled_signal)

        if window == 'hann':
            win = np.hanning(n)
        elif window == 'hamming':
            win = np.hamming(n)
        else:
            win = np.ones(n)

        signal_windowed = self.resampled_signal * win

        spectrum = np.fft.fft(signal_windowed)
        magnitude = 2 * np.abs(spectrum[:n//2]) / n

        angle_step = self.angular_axis[1] - self.angular_axis[0]
        orders = np.fft.fftfreq(n, d=angle_step)[:n//2] * 2 * np.pi

        return orders, magnitude

    def get_rpm_profile(self):
        if self.instantaneous_rpm is None:
            self.calculate_instantaneous_rpm()
        return self.time_axis, self.instantaneous_rpm

    def get_correction_summary(self):
        if not self.pulse_corrections:
            return "No pulse corrections applied"

        inserted = sum(1 for c in self.pulse_corrections if c['type'] == 'inserted')
        deleted = sum(1 for c in self.pulse_corrections if c['type'] == 'deleted')

        return f"Pulse corrections: {inserted} inserted, {deleted} deleted"

    def compute_stft(self, nperseg=1024, noverlap=None, window='hann'):
        if noverlap is None:
            noverlap = nperseg // 2

        freqs, times, Zxx = stft(
            self.vibration_signal,
            fs=self.fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            boundary='zeros'
        )

        self.stft_freqs = freqs
        self.stft_times = times
        self.stft_magnitude = np.abs(Zxx)

        return freqs, times, self.stft_magnitude

    def detect_tfr_peaks(self, min_freq_ratio=0.02, max_freq_ratio=0.48, n_peaks=5):
        if not hasattr(self, 'stft_magnitude'):
            self.compute_stft()

        min_freq = min_freq_ratio * self.fs / 2
        max_freq = max_freq_ratio * self.fs / 2

        freq_mask = (self.stft_freqs >= min_freq) & (self.stft_freqs <= max_freq)
        freqs_filtered = self.stft_freqs[freq_mask]
        mag_filtered = self.stft_magnitude[freq_mask, :]

        peak_candidates = []

        for t_idx in range(len(self.stft_times)):
            time_slice = mag_filtered[:, t_idx]
            peaks, peak_props = find_peaks(
                time_slice,
                height=np.max(time_slice) * 0.1,
                distance=5
            )

            if len(peaks) > 0:
                peak_heights = peak_props['peak_heights']
                sorted_idx = np.argsort(peak_heights)[::-1]
                top_peaks = peaks[sorted_idx[:n_peaks]]

                for p in top_peaks:
                    peak_candidates.append({
                        'time': self.stft_times[t_idx],
                        'time_idx': t_idx,
                        'freq': freqs_filtered[p],
                        'freq_idx': np.where(freq_mask)[0][p],
                        'magnitude': time_slice[p]
                    })

        self.peak_candidates = peak_candidates
        return peak_candidates

    def build_transition_cost_matrix(self, peak_candidates, max_accel=1000):
        n_candidates = len(peak_candidates)

        if n_candidates < 2:
            return None

        cost_matrix = np.inf * np.ones((n_candidates, n_candidates))

        for i in range(n_candidates):
            for j in range(n_candidates):
                if peak_candidates[j]['time_idx'] <= peak_candidates[i]['time_idx']:
                    continue

                dt = peak_candidates[j]['time'] - peak_candidates[i]['time']
                if dt <= 0:
                    continue

                freq_i = peak_candidates[i]['freq']
                freq_j = peak_candidates[j]['freq']

                accel = abs((freq_j - freq_i) / dt)
                mag_i = peak_candidates[i]['magnitude']
                mag_j = peak_candidates[j]['magnitude']

                mag_cost = 1.0 / (0.01 + (mag_i + mag_j) / 2)

                if accel <= max_accel:
                    cost_matrix[i, j] = accel * 0.1 + mag_cost * 10

        return cost_matrix

    def find_optimal_path_dp(self, peak_candidates, cost_matrix, smooth_weight=0.5):
        n_candidates = len(peak_candidates)

        if n_candidates < 2:
            return []

        dp = np.inf * np.ones(n_candidates)
        prev = -np.ones(n_candidates, dtype=int)

        for i in range(n_candidates):
            dp[i] = 1.0 / (0.01 + peak_candidates[i]['magnitude'])

        for j in range(n_candidates):
            for i in range(j):
                if cost_matrix[i, j] < np.inf:
                    if dp[i] + cost_matrix[i, j] < dp[j]:
                        dp[j] = dp[i] + cost_matrix[i, j]
                        prev[j] = i

        end_idx = np.argmin(dp)

        path = []
        current = end_idx
        while current != -1:
            path.append(current)
            current = prev[current]

        path = path[::-1]

        if len(path) < 2:
            return []

        return path

    def extract_rpm_from_path(self, peak_candidates, path, order_hint=1):
        if len(path) < 2:
            raise ValueError("Path too short to extract RPM")

        path_times = []
        path_freqs = []

        for idx in path:
            path_times.append(peak_candidates[idx]['time'])
            path_freqs.append(peak_candidates[idx]['freq'])

        path_times = np.array(path_times)
        path_freqs = np.array(path_freqs)

        rpm_raw = path_freqs * 60 / order_hint

        mask = rpm_raw > 0
        if np.sum(mask) < 2:
            raise ValueError("Valid RPM points insufficient")

        rpm_interp = interp1d(
            path_times[mask], rpm_raw[mask],
            kind='linear', fill_value='extrapolate', bounds_error=False
        )
        self.instantaneous_rpm = rpm_interp(self.time_axis)

        spline = UnivariateSpline(
            self.time_axis, self.instantaneous_rpm,
            s=len(self.instantaneous_rpm) * 100, k=3
        )
        self.instantaneous_rpm = spline(self.time_axis)

        self.instantaneous_rpm = np.maximum(self.instantaneous_rpm, 0)

        return self.instantaneous_rpm

    def estimate_rpm_from_ridge_following(self, order_hint=1,
                                           nperseg=1024,
                                           n_peaks=10,
                                           max_accel=2000,
                                           min_freq_ratio=0.01,
                                           max_freq_ratio=0.45):
        self.compute_stft(nperseg=nperseg)

        peaks = self.detect_tfr_peaks(
            min_freq_ratio=min_freq_ratio,
            max_freq_ratio=max_freq_ratio,
            n_peaks=n_peaks
        )

        if len(peaks) < 10:
            raise ValueError(f"Not enough peaks detected: {len(peaks)}")

        cost_matrix = self.build_transition_cost_matrix(peaks, max_accel=max_accel)

        if cost_matrix is None:
            raise ValueError("Failed to build cost matrix")

        path = self.find_optimal_path_dp(peaks, cost_matrix)

        if len(path) < 5:
            raise ValueError(f"Path too short: {len(path)}")

        return self.extract_rpm_from_path(peaks, path, order_hint=order_hint)

    def multi_order_ridge_tracking(self, orders_to_track=[1, 2, 3],
                                    nperseg=1024, n_peaks=15, max_accel=2500):
        self.compute_stft(nperseg=nperseg)

        peaks = self.detect_tfr_peaks(n_peaks=n_peaks)

        if len(peaks) < 20:
            raise ValueError(f"Not enough peaks detected: {len(peaks)}")

        cost_matrix = self.build_transition_cost_matrix(peaks, max_accel=max_accel)

        rpm_estimates = []

        for order in orders_to_track:
            try:
                path = self.find_optimal_path_dp(peaks, cost_matrix)
                if len(path) >= 5:
                    self.extract_rpm_from_path(peaks, path, order_hint=order)
                    rpm_estimates.append({
                        'order': order,
                        'rpm': self.instantaneous_rpm.copy(),
                        'path_length': len(path)
                    })
            except:
                continue

        if len(rpm_estimates) == 0:
            raise ValueError("Failed to estimate RPM from any order")

        rpm_matrix = np.array([est['rpm'] for est in rpm_estimates])
        self.instantaneous_rpm = np.median(rpm_matrix, axis=0)

        return self.instantaneous_rpm, rpm_estimates

    def compute_blind_order_tracking(self, orders_per_rev=360,
                                       order_hint=1,
                                       nperseg=1024,
                                       n_peaks=10,
                                       max_accel=2000,
                                       method='ridge'):
        if method == 'ridge':
            self.estimate_rpm_from_ridge_following(
                order_hint=order_hint,
                nperseg=nperseg,
                n_peaks=n_peaks,
                max_accel=max_accel
            )
        elif method == 'multi_order':
            self.multi_order_ridge_tracking(
                orders_to_track=[0.5, 1, 2, 3],
                nperseg=nperseg,
                n_peaks=n_peaks,
                max_accel=max_accel
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.calculate_angular_displacement()

        total_angle = self.angle[-1]
        num_samples = int(total_angle * orders_per_rev / (2 * np.pi))

        self.angular_axis = np.linspace(0, total_angle, num_samples)

        interpolator = interp1d(
            self.angle, self.vibration_signal, kind='cubic',
            fill_value='extrapolate', bounds_error=False
        )
        self.resampled_signal = interpolator(self.angular_axis)

        return self.angular_axis, self.resampled_signal


def generate_test_signal(fs=10000, duration=5, rpm_start=600, rpm_end=3000,
                         orders=[1, 2, 3, 4.5], amplitudes=[1, 0.5, 0.3, 0.2],
                         noise_level=0.1):
    t = np.arange(fs * duration) / fs

    rpm_rate = (rpm_end - rpm_start) / duration
    instantaneous_rpm = rpm_start + rpm_rate * t
    angular_velocity = 2 * np.pi * instantaneous_rpm / 60

    angle = np.cumsum(angular_velocity) / fs

    vibration = np.zeros_like(t)
    for order, amp in zip(orders, amplitudes):
        vibration += amp * np.sin(order * angle)

    vibration += noise_level * np.random.randn(len(vibration))

    tach_angle = angle % (2 * np.pi)
    tach_signal = np.where(np.diff(tach_angle) < -np.pi, 1, 0)
    tach_signal = np.concatenate([[0], tach_signal])

    return t, vibration, tach_signal, instantaneous_rpm


def generate_test_signal_with_missing_pulses(fs=10000, duration=5,
                                               rpm_start=600, rpm_end=3000,
                                               orders=[1, 2, 3, 4.5],
                                               amplitudes=[1, 0.5, 0.3, 0.2],
                                               noise_level=0.1,
                                               missing_pulse_indices=None,
                                               extra_pulse_indices=None):
    t = np.arange(fs * duration) / fs

    rpm_rate = (rpm_end - rpm_start) / duration
    instantaneous_rpm = rpm_start + rpm_rate * t
    angular_velocity = 2 * np.pi * instantaneous_rpm / 60

    angle = np.cumsum(angular_velocity) / fs

    vibration = np.zeros_like(t)
    for order, amp in zip(orders, amplitudes):
        vibration += amp * np.sin(order * angle)

    vibration += noise_level * np.random.randn(len(vibration))

    tach_angle = angle % (2 * np.pi)
    tach_signal = np.where(np.diff(tach_angle) < -np.pi, 1, 0)
    tach_signal = np.concatenate([[0], tach_signal])

    if missing_pulse_indices is not None and len(missing_pulse_indices) > 0:
        pulse_positions = np.where(tach_signal == 1)[0]
        for idx in missing_pulse_indices:
            if idx < len(pulse_positions):
                tach_signal[pulse_positions[idx]] = 0

    if extra_pulse_indices is not None and len(extra_pulse_indices) > 0:
        for idx in extra_pulse_indices:
            if idx < len(tach_signal):
                tach_signal[idx] = 1

    return t, vibration, tach_signal, instantaneous_rpm


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
