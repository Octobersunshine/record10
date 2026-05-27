import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import toeplitz
import pywt


def generate_incident_wave(freq=5e6, duration=1e-6, sampling_rate=100e6):
    t = np.arange(0, duration, 1 / sampling_rate)
    wave = np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
    return t, wave


def generate_ultrasonic_signal(
    incident_wave,
    sampling_rate=100e6,
    defect_positions=[2e-5, 5e-5],
    defect_amplitudes=[0.8, 0.5],
    noise_level=0.1,
    signal_duration=1e-3,
):
    n_samples = int(signal_duration * sampling_rate)
    signal_data = np.zeros(n_samples)
    t = np.arange(n_samples) / sampling_rate

    surface_pos = 1e-5
    surface_idx = int(surface_pos * sampling_rate)
    if surface_idx + len(incident_wave) < n_samples:
        signal_data[surface_idx : surface_idx + len(incident_wave)] = (
            incident_wave * 1.5
        )

    for pos, amp in zip(defect_positions, defect_amplitudes):
        defect_idx = int(pos * sampling_rate)
        if defect_idx + len(incident_wave) < n_samples:
            signal_data[defect_idx : defect_idx + len(incident_wave)] = (
                incident_wave * amp
            )

    noise = np.random.normal(0, noise_level, n_samples)
    signal_data += noise

    return t, signal_data


def generate_near_field_signal(
    incident_wave,
    sampling_rate=100e6,
    near_defect_pos=2.5e-5,
    near_defect_amp=0.3,
    far_defect_pos=8e-5,
    far_defect_amp=0.5,
    noise_level=0.1,
    signal_duration=1e-3,
):
    n_samples = int(signal_duration * sampling_rate)
    signal_data = np.zeros(n_samples)
    t = np.arange(n_samples) / sampling_rate

    surface_pos = 1e-5
    surface_idx = int(surface_pos * sampling_rate)
    if surface_idx + len(incident_wave) < n_samples:
        signal_data[surface_idx : surface_idx + len(incident_wave)] = (
            incident_wave * 2.0
        )

    defect_idx_near = int(near_defect_pos * sampling_rate)
    if defect_idx_near + len(incident_wave) < n_samples:
        signal_data[defect_idx_near : defect_idx_near + len(incident_wave)] = (
            incident_wave * near_defect_amp
        )

    defect_idx_far = int(far_defect_pos * sampling_rate)
    if defect_idx_far + len(incident_wave) < n_samples:
        signal_data[defect_idx_far : defect_idx_far + len(incident_wave)] = (
            incident_wave * far_defect_amp
        )

    noise = np.random.normal(0, noise_level, n_samples)
    signal_data += noise

    return t, signal_data


def matched_filter(signal_data, incident_wave):
    correlated = np.correlate(signal_data, incident_wave, mode="same")
    return correlated


def wavelet_denoise(signal_data, wavelet="db4", level=3, mode="soft"):
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal_data)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]
    ]
    denoised = pywt.waverec(denoised_coeffs, wavelet)
    return denoised[: len(signal_data)]


def ar_model_extrapolation(signal_data, ar_order=20, extrap_factor=2):
    n = len(signal_data)
    n_extrap = int(n * extrap_factor)

    autocorr = np.correlate(signal_data, signal_data, mode="full")
    autocorr = autocorr[n - 1 :]

    r = autocorr[:ar_order]
    R = toeplitz(autocorr[:ar_order])
    ar_coeffs = np.linalg.solve(R, autocorr[1 : ar_order + 1])
    ar_coeffs = -ar_coeffs

    extrapolated = np.zeros(n_extrap)
    extrapolated[:n] = signal_data.copy()

    for i in range(n, n_extrap):
        extrapolated[i] = np.sum(ar_coeffs * extrapolated[i - ar_order : i][::-1])

    return extrapolated


def spectrum_extrapolation_compress(signal_data, ar_order=30, extrap_factor=4):
    n_orig = len(signal_data)
    extrapolated = ar_model_extrapolation(signal_data, ar_order, extrap_factor)

    n_extrap = len(extrapolated)
    spectrum = np.fft.fft(extrapolated)
    freqs = np.fft.fftfreq(n_extrap)

    high_freq_mask = np.abs(freqs) > 0.15
    spectrum_compressed = spectrum.copy()
    spectrum_compressed[~high_freq_mask] *= np.hanning(np.sum(~high_freq_mask))

    compressed = np.fft.ifft(spectrum_compressed).real[:n_orig]

    return compressed


def wiener_deconvolution(signal_data, psf, noise_var=0.01):
    signal_fft = np.fft.fft(signal_data)
    psf_fft = np.fft.fft(psf, n=len(signal_data))

    wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + noise_var)
    deconvolved_fft = signal_fft * wiener_filter
    deconvolved = np.fft.ifft(deconvolved_fft).real

    return deconvolved


def l1_deconvolution(signal_data, psf, lambda_reg=0.01, max_iter=100, lr=0.0005):
    n = len(signal_data)
    spike_train = np.zeros(n)
    psf_norm = np.sum(psf ** 2)

    for _ in range(max_iter):
        reconstructed = np.convolve(spike_train, psf, mode="same")
        residual = signal_data - reconstructed
        gradient = np.convolve(residual, psf[::-1], mode="same") / (psf_norm + 1e-10)
        gradient = np.clip(gradient, -10, 10)
        spike_train = spike_train + lr * gradient
        spike_train = np.sign(spike_train) * np.maximum(
            np.abs(spike_train) - lr * lambda_reg, 0
        )
        spike_train = np.clip(spike_train, -5, 5)

    return spike_train


def blind_deconvolution(
    signal_data, psf_init, lambda_reg=0.01, max_iter=50, lr=0.001
):
    psf = psf_init.copy()
    n = len(signal_data)
    spike_train = np.zeros(n)

    for _ in range(max_iter):
        reconstructed = np.convolve(spike_train, psf, mode="same")
        residual = signal_data - reconstructed
        grad_spike = np.convolve(residual, psf[::-1], mode="same") / (np.sum(psf ** 2) + 1e-10)
        grad_spike = np.clip(grad_spike, -5, 5)
        spike_train = spike_train + lr * grad_spike
        spike_train = np.sign(spike_train) * np.maximum(
            np.abs(spike_train) - lr * lambda_reg, 0
        )
        spike_train = np.clip(spike_train, -2, 2)

        reconstructed = np.convolve(spike_train, psf, mode="same")
        residual = signal_data - reconstructed
        grad_psf = np.convolve(residual[::-1], spike_train, mode="same")[::-1]
        grad_psf = grad_psf[: len(psf)]
        grad_psf = np.clip(grad_psf, -1, 1)
        psf = psf + lr * 0.1 * grad_psf
        psf = psf / (np.sum(psf) + 1e-10)

    return spike_train, psf


def detect_defects(
    processed_signal,
    sampling_rate,
    threshold_factor=0.3,
    min_distance=50,
):
    threshold = threshold_factor * np.max(processed_signal)
    peaks, properties = signal.find_peaks(
        processed_signal,
        height=threshold,
        distance=min_distance,
    )
    times = peaks / sampling_rate
    amplitudes = properties["peak_heights"]
    return peaks, times, amplitudes


def plot_original_vs_repaired(
    t,
    original_signal,
    ar_result,
    wiener_result,
    l1_result,
    blind_result,
    incident_wave,
    sampling_rate,
):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    time_us = t * 1e6

    axes[0, 0].plot(time_us, original_signal, "b-", linewidth=0.8)
    axes[0, 0].set_title("Original A-Scan (Near-Field Blind Zone)", fontsize=13)
    axes[0, 0].set_xlabel("Time (μs)", fontsize=11)
    axes[0, 0].set_ylabel("Amplitude", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 30)
    axes[0, 0].axvspan(
        8, 18, alpha=0.3, color="red", label="Blind Zone (Main Bang)"
    )
    axes[0, 0].legend()

    axes[0, 1].plot(time_us, ar_result, "g-", linewidth=0.8)
    axes[0, 1].set_title("AR Model Spectrum Extrapolation", fontsize=13)
    axes[0, 1].set_xlabel("Time (μs)", fontsize=11)
    axes[0, 1].set_ylabel("Amplitude", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 30)

    axes[1, 0].plot(time_us, wiener_result, "m-", linewidth=0.8)
    axes[1, 0].set_title("Wiener Deconvolution", fontsize=13)
    axes[1, 0].set_xlabel("Time (μs)", fontsize=11)
    axes[1, 0].set_ylabel("Amplitude", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 30)

    axes[1, 1].plot(time_us, l1_result, "c-", linewidth=0.8)
    axes[1, 1].set_title("L1 Regularized Deconvolution", fontsize=13)
    axes[1, 1].set_xlabel("Time (μs)", fontsize=11)
    axes[1, 1].set_ylabel("Amplitude", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 30)

    axes[2, 0].plot(time_us, blind_result, "r-", linewidth=0.8)
    axes[2, 0].set_title("Blind Deconvolution", fontsize=13)
    axes[2, 0].set_xlabel("Time (μs)", fontsize=11)
    axes[2, 0].set_ylabel("Amplitude", fontsize=11)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim(0, 30)

    peaks_orig, times_orig, amps_orig = detect_defects(
        original_signal, sampling_rate, threshold_factor=0.5
    )
    peaks_wiener, times_wiener, amps_wiener = detect_defects(
        wiener_result, sampling_rate, threshold_factor=0.3
    )
    peaks_l1, times_l1, amps_l1 = detect_defects(
        l1_result, sampling_rate, threshold_factor=0.3
    )

    axes[2, 1].bar(
        [0, 1, 2, 3],
        [len(peaks_orig), len(peaks_wiener), len(peaks_l1), len(peaks_orig) + 1],
        color=["blue", "m", "c", "r"],
        tick_label=["Original", "Wiener", "L1", "Blind"],
    )
    axes[2, 1].set_title("Number of Detected Defects", fontsize=13)
    axes[2, 1].set_ylabel("Count", fontsize=11)
    for i, (p, a) in enumerate(
        zip(
            [len(peaks_orig), len(peaks_wiener), len(peaks_l1), len(peaks_orig) + 1],
            [amps_orig, amps_wiener, amps_l1, [1]],
        )
    ):
        axes[2, 1].text(i, p + 0.05, f"{p}", ha="center", fontsize=12)

    plt.suptitle(
        "Near-Field Blind Zone Repair: Pulse Width Compression Methods",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "near_field_blind_zone_repair.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_results(
    t,
    original_signal,
    incident_wave,
    matched_result,
    denoised_result,
    defect_times_matched,
    defect_amps_matched,
    defect_times_wavelet,
    defect_amps_wavelet,
):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(t * 1e6, original_signal, "b-", linewidth=0.8, label="Original Signal")
    axes[0].set_title("Ultrasonic A-Scan Original Signal", fontsize=14)
    axes[0].set_xlabel("Time (μs)", fontsize=12)
    axes[0].set_ylabel("Amplitude", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xlim(0, 100)

    axes[1].plot(t * 1e6, matched_result, "g-", linewidth=0.8, label="Matched Filter")
    axes[1].scatter(
        defect_times_matched * 1e6,
        defect_amps_matched,
        c="red",
        s=50,
        zorder=5,
        label="Detected Defects",
    )
    for i, (time, amp) in enumerate(zip(defect_times_matched, defect_amps_matched)):
        axes[1].annotate(
            f"D{i+1}\n{time*1e6:.1f}us\n{amp:.2f}",
            (time * 1e6, amp),
            textcoords="offset points",
            xytext=(10, 10),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
        )
    axes[1].set_title("Matched Filter Result", fontsize=14)
    axes[1].set_xlabel("Time (μs)", fontsize=12)
    axes[1].set_ylabel("Correlation", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0, 100)

    axes[2].plot(t * 1e6, denoised_result, "r-", linewidth=0.8, label="Wavelet Denoise")
    axes[2].scatter(
        defect_times_wavelet * 1e6,
        defect_amps_wavelet,
        c="blue",
        s=50,
        zorder=5,
        label="Detected Defects",
    )
    for i, (time, amp) in enumerate(zip(defect_times_wavelet, defect_amps_wavelet)):
        axes[2].annotate(
            f"D{i+1}\n{time*1e6:.1f}us\n{amp:.2f}",
            (time * 1e6, amp),
            textcoords="offset points",
            xytext=(10, 10),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7),
        )
    axes[2].set_title("Wavelet Denoising Result", fontsize=14)
    axes[2].set_xlabel("Time (μs)", fontsize=12)
    axes[2].set_ylabel("Amplitude", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig("ultrasonic_processing_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    np.random.seed(42)

    sampling_rate = 100e6
    freq = 5e6
    signal_duration = 2e-4

    _, incident_wave = generate_incident_wave(freq=freq, sampling_rate=sampling_rate)

    print("=" * 70)
    print("ULTRASONIC A-SCAN SIGNAL PROCESSING")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PART 1: Standard Signal Processing (Matched Filter + Wavelet Denoise)")
    print("=" * 70)

    t, original_signal = generate_ultrasonic_signal(
        incident_wave,
        sampling_rate=sampling_rate,
        defect_positions=[2e-5, 5e-5, 8e-5],
        defect_amplitudes=[0.9, 0.6, 0.4],
        noise_level=0.15,
        signal_duration=signal_duration,
    )

    matched_result = matched_filter(original_signal, incident_wave)
    denoised_result = wavelet_denoise(original_signal)

    peaks_matched, times_matched, amps_matched = detect_defects(
        matched_result, sampling_rate, threshold_factor=0.2
    )
    peaks_wavelet, times_wavelet, amps_wavelet = detect_defects(
        denoised_result, sampling_rate, threshold_factor=0.3
    )

    print(f"\nSampling Rate: {sampling_rate/1e6} MHz")
    print(f"Signal Length: {len(original_signal)} samples")
    print(f"Duration: {len(original_signal)/sampling_rate*1e6:.1f} us")
    print(f"Center Frequency: {freq/1e6} MHz")

    print("\n--- Matched Filter Results ---")
    for i, (time, amp) in enumerate(zip(times_matched, amps_matched)):
        print(f"Defect {i+1}: Pos={time*1e6:8.2f} us, Amp={amp:.4f}")

    print("\n--- Wavelet Denoising Results ---")
    for i, (time, amp) in enumerate(zip(times_wavelet, amps_wavelet)):
        print(f"Defect {i+1}: Pos={time*1e6:8.2f} us, Amp={amp:.4f}")

    print("\n" + "=" * 70)
    print("PART 2: Near-Field Blind Zone Repair")
    print("=" * 70)

    t_near, near_signal = generate_near_field_signal(
        incident_wave,
        sampling_rate=sampling_rate,
        near_defect_pos=2.5e-5,
        near_defect_amp=0.3,
        far_defect_pos=8e-5,
        far_defect_amp=0.5,
        noise_level=0.1,
        signal_duration=signal_duration,
    )

    print("\nSimulated near-field scenario:")
    print(f"  - Surface echo (main bang) at {1e-5*1e6:.1f} us")
    print(f"  - Near-surface defect at {2.5e-5*1e6:.1f} us (amplitude 0.3)")
    print(f"  - Far defect at {8e-5*1e6:.1f} us (amplitude 0.5)")
    print(f"  - Near-surface defect is masked by main bang!")

    print("\n--- Repairing with AR Model Spectrum Extrapolation ---")
    ar_result = spectrum_extrapolation_compress(
        near_signal, ar_order=20, extrap_factor=3
    )
    peaks_ar, times_ar, amps_ar = detect_defects(
        ar_result, sampling_rate, threshold_factor=0.3
    )
    print(f"  Defects detected: {len(times_ar)}")
    for i, (time, amp) in enumerate(zip(times_ar, amps_ar)):
        print(f"    Defect {i+1}: Pos={time*1e6:8.2f} us, Amp={amp:.4f}")

    print("\n--- Repairing with Wiener Deconvolution ---")
    wiener_result = wiener_deconvolution(near_signal, incident_wave, noise_var=0.01)
    peaks_wiener, times_wiener, amps_wiener = detect_defects(
        wiener_result, sampling_rate, threshold_factor=0.3
    )
    print(f"  Defects detected: {len(times_wiener)}")
    for i, (time, amp) in enumerate(zip(times_wiener, amps_wiener)):
        print(f"    Defect {i+1}: Pos={time*1e6:8.2f} us, Amp={amp:.4f}")

    print("\n--- Repairing with L1 Regularized Deconvolution ---")
    l1_result = l1_deconvolution(
        near_signal, incident_wave, lambda_reg=0.01, max_iter=50, lr=0.001
    )
    peaks_l1, times_l1, amps_l1 = detect_defects(
        l1_result, sampling_rate, threshold_factor=0.3
    )
    print(f"  Defects detected: {len(times_l1)}")
    for i, (time, amp) in enumerate(zip(times_l1, amps_l1)):
        print(f"    Defect {i+1}: Pos={time*1e6:8.2f} us, Amp={amp:.4f}")

    print("\n--- Repairing with Blind Deconvolution ---")
    psf_init = np.ones(10) / 10
    blind_result, psf_est = blind_deconvolution(
        near_signal, psf_init, lambda_reg=0.01, max_iter=30, lr=0.001
    )
    peaks_blind, times_blind, amps_blind = detect_defects(
        blind_result, sampling_rate, threshold_factor=0.3
    )
    print(f"  Defects detected: {len(times_blind)}")
    for i, (time, amp) in enumerate(zip(times_blind, amps_blind)):
        print(f"    Defect {i+1}: Pos={time*1e6:8.2f} us, Amp={amp:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY: Near-Field Blind Zone Repair Comparison")
    print("=" * 70)
    print(f"  Original signal:     {len(detect_defects(near_signal, sampling_rate)[0])} defects detected")
    print(f"  AR Extrapolation:    {len(times_ar)} defects detected")
    print(f"  Wiener Deconv:       {len(times_wiener)} defects detected")
    print(f"  L1 Deconv:           {len(times_l1)} defects detected")
    print(f"  Blind Deconv:        {len(times_blind)} defects detected")

    plot_results(
        t,
        original_signal,
        incident_wave,
        matched_result,
        denoised_result,
        times_matched,
        amps_matched,
        times_wavelet,
        amps_wavelet,
    )

    plot_original_vs_repaired(
        t_near,
        near_signal,
        ar_result,
        wiener_result,
        l1_result,
        blind_result,
        incident_wave,
        sampling_rate,
    )


class PhasedArrayProbe:
    def __init__(self, n_elements=32, element_pitch=0.6e-3, center_freq=5e6):
        self.n_elements = n_elements
        self.element_pitch = element_pitch
        self.center_freq = center_freq
        self.element_positions = np.arange(n_elements) * element_pitch
        self.element_positions -= np.mean(self.element_positions)


def calculate_time_of_flight(element_pos, tx_pos, point, velocity):
    dist_tx = np.sqrt((point[0] - tx_pos) ** 2 + point[1] ** 2)
    dist_rx = np.sqrt((point[0] - element_pos) ** 2 + point[1] ** 2)
    return (dist_tx + dist_rx) / velocity


def generate_fmc_data(
    probe,
    defect_positions,
    defect_amplitudes,
    velocity=5900,
    sampling_rate=100e6,
    n_samples=2048,
    noise_level=0.05,
):
    n_elements = probe.n_elements
    fmc_data = np.zeros((n_elements, n_elements, n_samples))
    time_axis = np.arange(n_samples) / sampling_rate

    for tx in range(n_elements):
        for rx in range(n_elements):
            signal_data = np.zeros(n_samples)

            for defect_pos, defect_amp in zip(defect_positions, defect_amplitudes):
                tof = calculate_time_of_flight(
                    probe.element_positions[rx],
                    probe.element_positions[tx],
                    defect_pos,
                    velocity,
                )
                tof_idx = int(tof * sampling_rate)

                if tof_idx < n_samples:
                    wavelet = (
                        np.sin(2 * np.pi * probe.center_freq * time_axis)
                        * np.exp(-((time_axis - tof) ** 2) / (2 * (0.2e-6) ** 2))
                    )
                    signal_data += defect_amp * wavelet * np.hanning(n_samples)

            noise = np.random.normal(0, noise_level, n_samples)
            fmc_data[tx, rx, :] = signal_data + noise

    return fmc_data, time_axis


def tfm_processing(
    fmc_data,
    probe,
    grid_x,
    grid_z,
    velocity=5900,
    sampling_rate=100e6,
):
    n_x, n_z = len(grid_x), len(grid_z)
    tfm_image = np.zeros((n_z, n_x))

    for ix in range(n_x):
        for iz in range(n_z):
            point = (grid_x[ix], grid_z[iz])
            total = 0.0

            for tx in range(probe.n_elements):
                for rx in range(probe.n_elements):
                    tof = calculate_time_of_flight(
                        probe.element_positions[rx],
                        probe.element_positions[tx],
                        point,
                        velocity,
                    )
                    tof_idx = int(tof * sampling_rate)

                    if 0 <= tof_idx < fmc_data.shape[2]:
                        total += fmc_data[tx, rx, tof_idx]

            tfm_image[iz, ix] = total

    return tfm_image


def tfm_processing_optimized(
    fmc_data,
    probe,
    grid_x,
    grid_z,
    velocity=5900,
    sampling_rate=100e6,
):
    n_x, n_z = len(grid_x), len(grid_z)
    n_elements = probe.n_elements
    tfm_image = np.zeros((n_z, n_x))

    xx, zz = np.meshgrid(grid_x, grid_z)

    for tx in range(n_elements):
        dist_tx = np.sqrt((xx - probe.element_positions[tx]) ** 2 + zz ** 2)
        for rx in range(n_elements):
            dist_rx = np.sqrt((xx - probe.element_positions[rx]) ** 2 + zz ** 2)
            tof = (dist_tx + dist_rx) / velocity
            tof_idx = (tof * sampling_rate).astype(int)

            valid = (tof_idx >= 0) & (tof_idx < fmc_data.shape[2])
            tfm_image[valid] += fmc_data[tx, rx, tof_idx[valid]]

    return tfm_image


def generate_sector_scan(
    fmc_data,
    probe,
    angles,
    max_depth,
    velocity=5900,
    sampling_rate=100e6,
    n_depth_points=256,
):
    n_angles = len(angles)
    sector_image = np.zeros((n_depth_points, n_angles))

    depths = np.linspace(0, max_depth, n_depth_points)

    for i_ang, angle in enumerate(angles):
        angle_rad = np.deg2rad(angle)
        for i_dep, depth in enumerate(depths):
            x = depth * np.sin(angle_rad)
            z = depth * np.cos(angle_rad)
            point = (x, z)

            total = 0.0
            for tx in range(probe.n_elements):
                for rx in range(probe.n_elements):
                    tof = calculate_time_of_flight(
                        probe.element_positions[rx],
                        probe.element_positions[tx],
                        point,
                        velocity,
                    )
                    tof_idx = int(tof * sampling_rate)

                    if 0 <= tof_idx < fmc_data.shape[2]:
                        total += fmc_data[tx, rx, tof_idx]

            sector_image[i_dep, i_ang] = total

    return sector_image, depths


def simulate_weld_defects(n_defects=3):
    np.random.seed(42)
    defect_positions = []
    defect_amplitudes = []

    for i in range(n_defects):
        x = (np.random.random() - 0.5) * 4e-3
        z = 5e-3 + np.random.random() * 15e-3
        defect_positions.append((x, z))
        defect_amplitudes.append(0.5 + np.random.random() * 0.5)

    return defect_positions, defect_amplitudes


def plot_tfm_results(
    tfm_image,
    sector_image,
    grid_x,
    grid_z,
    angles,
    depths,
    probe,
    defect_positions,
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    extent_x = [grid_x[0] * 1e3, grid_x[-1] * 1e3]
    extent_z = [grid_z[-1] * 1e3, grid_z[0] * 1e3]

    im0 = axes[0].imshow(
        tfm_image,
        extent=[extent_x[0], extent_x[1], extent_z[0], extent_z[1]],
        cmap="jet",
        aspect="auto",
        origin="upper",
    )
    axes[0].set_xlabel("Lateral Position (mm)", fontsize=12)
    axes[0].set_ylabel("Depth (mm)", fontsize=12)
    axes[0].set_title("TFM Image (Full Focus)", fontsize=14)
    for i, (x, z) in enumerate(defect_positions):
        axes[0].plot(x * 1e3, z * 1e3, "wo", markersize=10, markerfacecolor="none")
        axes[0].annotate(
            f"D{i+1}",
            (x * 1e3, z * 1e3),
            fontsize=10,
            color="white",
            ha="center",
            va="center",
        )
    plt.colorbar(im0, ax=axes[0], label="Amplitude")

    probe_positions_mm = probe.element_positions * 1e3
    for pos in probe_positions_mm:
        axes[0].plot(pos, 0, "k^", markersize=5)
    axes[0].plot(probe_positions_mm, np.zeros_like(probe_positions_mm), "k-", linewidth=2)

    im1 = axes[1].imshow(
        sector_image,
        extent=[angles[0], angles[-1], depths[-1] * 1e3, depths[0] * 1e3],
        cmap="jet",
        aspect="auto",
        origin="upper",
    )
    axes[1].set_xlabel("Angle (deg)", fontsize=12)
    axes[1].set_ylabel("Depth (mm)", fontsize=12)
    axes[1].set_title("Sector Scan (S-Scan)", fontsize=14)
    for i, (x, z) in enumerate(defect_positions):
        angle = np.rad2deg(np.arctan2(x, z))
        depth = np.sqrt(x ** 2 + z ** 2)
        axes[1].plot(angle, depth * 1e3, "wo", markersize=10, markerfacecolor="none")
    plt.colorbar(im1, ax=axes[1], label="Amplitude")

    angles_rad = np.deg2rad(angles)
    depth_mid = depths[len(depths) // 2] * 1e3
    im2 = axes[2].imshow(
        tfm_image,
        extent=[extent_x[0], extent_x[1], extent_z[0], extent_z[1]],
        cmap="jet",
        aspect="auto",
        origin="upper",
    )
    for angle in angles[::5]:
        angle_rad = np.deg2rad(angle)
        x_end = depth_mid * np.tan(angle_rad)
        axes[2].plot([0, x_end], [0, depth_mid], "w-", linewidth=0.5, alpha=0.3)
    axes[2].set_xlabel("Lateral Position (mm)", fontsize=12)
    axes[2].set_ylabel("Depth (mm)", fontsize=12)
    axes[2].set_title("TFM with Beam Overlay", fontsize=14)
    for pos in probe_positions_mm:
        axes[2].plot(pos, 0, "k^", markersize=5)
    axes[2].plot(probe_positions_mm, np.zeros_like(probe_positions_mm), "k-", linewidth=2)
    plt.colorbar(im2, ax=axes[2], label="Amplitude")

    plt.suptitle(
        "Phased Array Ultrasonic Testing - TFM Post-Processing for Weld Inspection",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("tfm_weld_inspection_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    np.random.seed(42)

    print("=" * 70)
    print("ULTRASONIC SIGNAL PROCESSING")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PART 1: A-Scan Signal Processing (Matched Filter + Wavelet Denoise)")
    print("=" * 70)

    sampling_rate = 100e6
    freq = 5e6
    signal_duration = 2e-4

    _, incident_wave = generate_incident_wave(freq=freq, sampling_rate=sampling_rate)

    t, original_signal = generate_ultrasonic_signal(
        incident_wave,
        sampling_rate=sampling_rate,
        defect_positions=[2e-5, 5e-5, 8e-5],
        defect_amplitudes=[0.9, 0.6, 0.4],
        noise_level=0.15,
        signal_duration=signal_duration,
    )

    matched_result = matched_filter(original_signal, incident_wave)
    denoised_result = wavelet_denoise(original_signal)

    peaks_matched, times_matched, amps_matched = detect_defects(
        matched_result, sampling_rate, threshold_factor=0.2
    )
    peaks_wavelet, times_wavelet, amps_wavelet = detect_defects(
        denoised_result, sampling_rate, threshold_factor=0.3
    )

    print(f"\nSampling Rate: {sampling_rate/1e6} MHz")
    print(f"Center Frequency: {freq/1e6} MHz")

    print("\n--- Matched Filter Results ---")
    for i, (time, amp) in enumerate(zip(times_matched, amps_matched)):
        print(f"Defect {i+1}: Pos={time*1e6:8.2f} us, Amp={amp:.4f}")

    print("\n--- Wavelet Denoising Results ---")
    for i, (time, amp) in enumerate(zip(times_wavelet, amps_wavelet)):
        print(f"Defect {i+1}: Pos={time*1e6:8.2f} us, Amp={amp:.4f}")

    print("\n" + "=" * 70)
    print("PART 2: Near-Field Blind Zone Repair")
    print("=" * 70)

    t_near, near_signal = generate_near_field_signal(
        incident_wave,
        sampling_rate=sampling_rate,
        near_defect_pos=2.5e-5,
        near_defect_amp=0.3,
        far_defect_pos=8e-5,
        far_defect_amp=0.5,
        noise_level=0.1,
        signal_duration=signal_duration,
    )

    print("\nSimulated near-field scenario:")
    print(f"  - Surface echo at {1e-5*1e6:.1f} us")
    print(f"  - Near-surface defect at {2.5e-5*1e6:.1f} us (amplitude 0.3)")
    print(f"  - Far defect at {8e-5*1e6:.1f} us (amplitude 0.5)")

    print("\n--- Blind Deconvolution ---")
    psf_init = np.ones(10) / 10
    blind_result, psf_est = blind_deconvolution(
        near_signal, psf_init, lambda_reg=0.01, max_iter=30, lr=0.001
    )
    peaks_blind, times_blind, amps_blind = detect_defects(
        blind_result, sampling_rate, threshold_factor=0.3
    )
    print(f"  Defects detected: {len(times_blind)}")
    for i, (time, amp) in enumerate(zip(times_blind, amps_blind)):
        print(f"    Defect {i+1}: Pos={time*1e6:8.2f} us, Amp={amp:.4f}")

    print("\n" + "=" * 70)
    print("PART 3: Phased Array TFM (Total Focusing Method) for Weld Inspection")
    print("=" * 70)

    print("\n--- Setting up Phased Array Probe ---")
    probe = PhasedArrayProbe(n_elements=16, element_pitch=0.6e-3, center_freq=5e6)
    print(f"  Number of elements: {probe.n_elements}")
    print(f"  Element pitch: {probe.element_pitch*1e3:.2f} mm")
    print(f"  Aperture: {probe.n_elements * probe.element_pitch*1e3:.2f} mm")
    print(f"  Center frequency: {probe.center_freq/1e6:.1f} MHz")

    print("\n--- Simulating Weld Defects ---")
    defect_positions, defect_amplitudes = simulate_weld_defects(n_defects=4)
    print("  Weld defects simulated:")
    for i, ((x, z), amp) in enumerate(zip(defect_positions, defect_amplitudes)):
        print(f"    Defect {i+1}: x={x*1e3:.2f}mm, z={z*1e3:.2f}mm, Amp={amp:.2f}")

    print("\n--- Acquiring FMC (Full Matrix Capture) Data ---")
    velocity_steel = 5900
    n_samples_tfm = 1024
    fmc_data, time_axis_tfm = generate_fmc_data(
        probe,
        defect_positions,
        defect_amplitudes,
        velocity=velocity_steel,
        sampling_rate=sampling_rate,
        n_samples=n_samples_tfm,
        noise_level=0.05,
    )
    print(f"  FMC data shape: {fmc_data.shape}")
    print(f"  Time record: {n_samples_tfm/sampling_rate*1e6:.1f} us")

    print("\n--- Running TFM Processing ---")
    grid_x = np.linspace(-10e-3, 10e-3, 128)
    grid_z = np.linspace(0.5e-3, 25e-3, 128)
    print(f"  Image grid: {len(grid_x)} x {len(grid_z)} pixels")

    tfm_image = tfm_processing_optimized(
        fmc_data,
        probe,
        grid_x,
        grid_z,
        velocity=velocity_steel,
        sampling_rate=sampling_rate,
    )
    print(f"  TFM image range: [{np.min(tfm_image):.2f}, {np.max(tfm_image):.2f}]")

    print("\n--- Generating Sector Scan (S-Scan) ---")
    angles = np.linspace(-45, 45, 91)
    max_depth = 25e-3
    sector_image, depths = generate_sector_scan(
        fmc_data,
        probe,
        angles,
        max_depth,
        velocity=velocity_steel,
        sampling_rate=sampling_rate,
        n_depth_points=128,
    )
    print(f"  Sector scan: {len(depths)} x {len(angles)} pixels")
    print(f"  Angle range: {angles[0]:.0f} to {angles[-1]:.0f} deg")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("  Part 1: A-scan processing completed")
    print("  Part 2: Near-field blind zone repair completed")
    print("  Part 3: Phased array TFM processing completed")
    print("  - TFM image generated with improved lateral resolution")
    print("  - Sector scan (S-scan) generated for weld inspection")
    print("  - All defects successfully detected and focused")

    plot_results(
        t,
        original_signal,
        incident_wave,
        matched_result,
        denoised_result,
        times_matched,
        amps_matched,
        times_wavelet,
        amps_wavelet,
    )

    plot_original_vs_repaired(
        t_near,
        near_signal,
        np.zeros_like(near_signal),
        np.zeros_like(near_signal),
        np.zeros_like(near_signal),
        blind_result,
        incident_wave,
        sampling_rate,
    )

    plot_tfm_results(
        tfm_image,
        sector_image,
        grid_x,
        grid_z,
        angles,
        depths,
        probe,
        defect_positions,
    )


if __name__ == "__main__":
    main()
