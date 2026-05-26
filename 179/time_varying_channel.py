import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import lfilter, fftconvolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys

sys.path.insert(0, '.')
from underwater_acoustic_channel import (
    SoundSpeedProfile, GaussianBeamTracer, UnderwaterAcousticChannel,
    MultipathComponent, create_munk_profile
)


@dataclass
class SeaSurfaceWaves:
    significant_wave_height: float = 1.0
    dominant_period: float = 8.0
    num_wave_components: int = 20
    random_seed: int = 42
    
    def __post_init__(self):
        rng = np.random.RandomState(self.random_seed)
        self.omega_peak = 2 * np.pi / self.dominant_period
        self.k_peak = self.omega_peak**2 / 9.81
        
        self.wave_numbers = np.linspace(self.k_peak * 0.3, self.k_peak * 3.0, self.num_wave_components)
        self.omega = np.sqrt(9.81 * self.wave_numbers)
        
        pm_spectrum = (self.significant_wave_height**2 * 5 / 16 * 
                      self.omega_peak**4 * self.omega**(-5) *
                      np.exp(-1.25 * (self.omega_peak / self.omega)**4))
        
        self.amplitudes = np.sqrt(2 * pm_spectrum * np.diff(self.wave_numbers, append=self.wave_numbers[-1]))
        self.phases = rng.uniform(0, 2 * np.pi, self.num_wave_components)
        self.directions = rng.uniform(-np.pi/6, np.pi/6, self.num_wave_components)
    
    def get_height(self, x: float, t: float) -> float:
        height = 0.0
        for i in range(self.num_wave_components):
            k = self.wave_numbers[i]
            omega = self.omega[i]
            A = self.amplitudes[i]
            phi = self.phases[i]
            direction = self.directions[i]
            
            x_component = k * (x * np.cos(direction))
            height += A * np.sin(x_component - omega * t + phi)
        
        return height
    
    def get_height_array(self, x_array: np.ndarray, t: float) -> np.ndarray:
        heights = np.zeros_like(x_array)
        for i in range(self.num_wave_components):
            k = self.wave_numbers[i]
            omega = self.omega[i]
            A = self.amplitudes[i]
            phi = self.phases[i]
            direction = self.directions[i]
            
            for j, x in enumerate(x_array):
                x_component = k * (x * np.cos(direction))
                heights[j] += A * np.sin(x_component - omega * t + phi)
        
        return heights
    
    def get_slope(self, x: float, t: float) -> float:
        dx = 0.1
        h1 = self.get_height(x - dx, t)
        h2 = self.get_height(x + dx, t)
        return (h2 - h1) / (2 * dx)


@dataclass
class InternalWaves:
    amplitude: float = 10.0
    wavelength: float = 500.0
    period: float = 300.0
    depth_center: float = 500.0
    depth_width: float = 200.0
    random_seed: int = 123
    
    def __post_init__(self):
        rng = np.random.RandomState(self.random_seed)
        self.num_modes = 5
        self.mode_amplitudes = self.amplitude * np.exp(-np.arange(self.num_modes))
        self.mode_wavenumbers = 2 * np.pi / (self.wavelength / (np.arange(1, self.num_modes + 1)))
        self.mode_omegas = 2 * np.pi / (self.period * np.arange(1, self.num_modes + 1))
        self.mode_phases = rng.uniform(0, 2 * np.pi, self.num_modes)
    
    def get_sound_speed_perturbation(self, depth: float, x: float, t: float) -> float:
        depth_envelope = np.exp(-((depth - self.depth_center) / self.depth_width)**2)
        
        perturbation = 0.0
        for n in range(self.num_modes):
            k = self.mode_wavenumbers[n]
            omega = self.mode_omegas[n]
            A = self.mode_amplitudes[n]
            phi = self.mode_phases[n]
            
            perturbation += A * np.sin(k * x - omega * t + phi) * depth_envelope
        
        return perturbation
    
    def perturb_profile(self, base_ssp: SoundSpeedProfile, x: float, t: float) -> SoundSpeedProfile:
        perturbed_speeds = base_ssp.speeds.copy()
        
        for i, depth in enumerate(base_ssp.depths):
            perturbed_speeds[i] += self.get_sound_speed_perturbation(depth, x, t)
        
        return SoundSpeedProfile(depths=base_ssp.depths.copy(), speeds=perturbed_speeds)


@dataclass
class ChannelSnapshot:
    time: float
    multipath_components: List[MultipathComponent]
    impulse_response: np.ndarray
    time_axis: np.ndarray
    surface_height: float
    ssp_perturbation: float


class TimeVaryingChannel:
    def __init__(self,
                 base_ssp: SoundSpeedProfile,
                 water_depth: float,
                 source_depth: float,
                 receiver_depth: float,
                 range_km: float,
                 relative_speed: float = 0.0,
                 carrier_freq: float = 10000.0,
                 simulation_duration: float = 60.0,
                 snapshot_interval: float = 1.0,
                 wave_height: float = 1.0,
                 internal_wave_amp: float = 10.0,
                 use_gaussian_beam: bool = True):
        
        self.base_ssp = base_ssp
        self.water_depth = water_depth
        self.source_depth = source_depth
        self.receiver_depth = receiver_depth
        self.range = range_km * 1000
        self.relative_speed = relative_speed
        self.carrier_freq = carrier_freq
        self.simulation_duration = simulation_duration
        self.snapshot_interval = snapshot_interval
        self.use_gaussian_beam = use_gaussian_beam
        
        self.surface_waves = SeaSurfaceWaves(
            significant_wave_height=wave_height,
            random_seed=42
        )
        
        self.internal_waves = InternalWaves(
            amplitude=internal_wave_amp,
            random_seed=123
        )
        
        self.snapshots: List[ChannelSnapshot] = []
        self._simulate_channel()
    
    def _simulate_channel(self):
        n_snapshots = int(self.simulation_duration / self.snapshot_interval) + 1
        times = np.linspace(0, self.simulation_duration, n_snapshots)
        
        print(f"Simulating time-varying channel...")
        print(f"  Duration: {self.simulation_duration:.1f} s")
        print(f"  Snapshots: {n_snapshots}")
        print(f"  Snapshot interval: {self.snapshot_interval:.2f} s")
        print()
        
        for i, t in enumerate(times):
            if i % 10 == 0:
                print(f"  Processing snapshot {i+1}/{n_snapshots} (t={t:.1f}s)...")
            
            surface_h = self.surface_waves.get_height(self.range / 2, t)
            effective_water_depth = self.water_depth + surface_h
            
            if effective_water_depth < 100:
                effective_water_depth = 100
            
            perturbed_ssp = self.internal_waves.perturb_profile(
                self.base_ssp, self.range / 2, t
            )
            
            channel = UnderwaterAcousticChannel(
                ssp=perturbed_ssp,
                water_depth=effective_water_depth,
                source_depth=self.source_depth,
                receiver_depth=self.receiver_depth,
                range_km=self.range / 1000,
                relative_speed=self.relative_speed,
                carrier_freq=self.carrier_freq,
                use_gaussian_beam=self.use_gaussian_beam
            )
            
            sample_rate = 48000
            t_axis, ir = channel.get_impulse_response(sample_rate=sample_rate)
            
            ssp_pert = self.internal_waves.get_sound_speed_perturbation(
                self.receiver_depth, self.range / 2, t
            )
            
            snapshot = ChannelSnapshot(
                time=t,
                multipath_components=channel.multipath_components,
                impulse_response=ir,
                time_axis=t_axis,
                surface_height=surface_h,
                ssp_perturbation=ssp_pert
            )
            self.snapshots.append(snapshot)
        
        print(f"  Done! Generated {len(self.snapshots)} channel snapshots.")
        print()
    
    def get_impulse_response_at(self, t: float, sample_rate: float = 48000) -> Tuple[np.ndarray, np.ndarray]:
        times = [s.time for s in self.snapshots]
        idx = np.searchsorted(times, t)
        
        if idx >= len(self.snapshots):
            idx = len(self.snapshots) - 1
        if idx <= 0:
            idx = 0
        
        if idx > 0 and abs(times[idx] - t) > abs(times[idx-1] - t):
            idx = idx - 1
        
        return self.snapshots[idx].time_axis, self.snapshots[idx].impulse_response
    
    def process_time_varying_signal(self, signal: np.ndarray, 
                                     sample_rate: float) -> np.ndarray:
        signal_duration = len(signal) / sample_rate
        n_snapshots_needed = int(signal_duration / self.snapshot_interval) + 1
        
        if n_snapshots_needed > len(self.snapshots):
            self.simulation_duration = signal_duration + 10
            self._simulate_channel()
        
        output = np.zeros(len(signal), dtype=complex)
        
        for i, snapshot in enumerate(self.snapshots):
            if i >= n_snapshots_needed:
                break
            
            t_start = i * self.snapshot_interval
            t_end = min((i + 1) * self.snapshot_interval, signal_duration)
            
            idx_start = int(t_start * sample_rate)
            idx_end = int(t_end * sample_rate)
            
            if idx_start >= len(signal):
                break
            
            segment = signal[idx_start:idx_end]
            
            ir = snapshot.impulse_response
            
            if len(segment) > 0:
                received = fftconvolve(segment, ir[:len(segment)], mode='same')
                output[idx_start:idx_end] = received[:idx_end - idx_start]
        
        return output
    
    def get_amplitude_series(self) -> Tuple[np.ndarray, np.ndarray]:
        times = np.array([s.time for s in self.snapshots])
        amplitudes = np.array([
            max((abs(c.amplitude) for c in s.multipath_components), default=0)
            for s in self.snapshots
        ])
        return times, amplitudes
    
    def get_delay_series(self) -> Tuple[np.ndarray, np.ndarray]:
        times = np.array([s.time for s in self.snapshots])
        delays = np.array([
            min((c.delay for c in s.multipath_components), default=0)
            for s in self.snapshots
        ])
        return times, delays
    
    def get_surface_height_series(self) -> Tuple[np.ndarray, np.ndarray]:
        times = np.array([s.time for s in self.snapshots])
        heights = np.array([s.surface_height for s in self.snapshots])
        return times, heights
    
    def get_coherence_time(self) -> float:
        if len(self.snapshots) < 2:
            return 0
        
        sample_rate = 48000
        n_snap = len(self.snapshots)
        correlations = []
        
        for i in range(n_snap - 1):
            ir1 = self.snapshots[i].impulse_response
            ir2 = self.snapshots[i + 1].impulse_response
            
            min_len = min(len(ir1), len(ir2))
            corr = np.abs(np.dot(ir1[:min_len], np.conj(ir2[:min_len])))
            norm = np.sqrt(np.sum(np.abs(ir1[:min_len])**2) * np.sum(np.abs(ir2[:min_len])**2))
            
            if norm > 0:
                correlations.append(corr / norm)
        
        if correlations:
            avg_corr = np.mean(correlations)
            if avg_corr > 0.5:
                return self.simulation_duration
            else:
                return self.snapshot_interval * (1 - avg_corr) / 0.5
        return 0


class CommunicationPerformance:
    @staticmethod
    def compute_snr(transmitted: np.ndarray, received: np.ndarray, 
                    noise_var: float = 1e-6) -> float:
        signal_power = np.mean(np.abs(transmitted)**2)
        error = received - transmitted
        noise_power = np.mean(np.abs(error)**2) + noise_var
        
        if noise_power < 1e-15:
            return 100.0
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def compute_ber(transmitted_bits: np.ndarray, received_bits: np.ndarray) -> float:
        if len(transmitted_bits) != len(received_bits):
            min_len = min(len(transmitted_bits), len(received_bits))
            transmitted_bits = transmitted_bits[:min_len]
            received_bits = received_bits[:min_len]
        
        errors = np.sum(transmitted_bits != received_bits)
        return errors / len(transmitted_bits) if len(transmitted_bits) > 0 else 0
    
    @staticmethod
    def bpsk_modulate(bits: np.ndarray, carrier_freq: float, 
                      sample_rate: float, symbol_duration: float) -> np.ndarray:
        samples_per_symbol = int(symbol_duration * sample_rate)
        t = np.arange(samples_per_symbol) / sample_rate
        
        signal = np.zeros(len(bits) * samples_per_symbol)
        
        for i, bit in enumerate(bits):
            phase = 0 if bit == 1 else np.pi
            start = i * samples_per_symbol
            end = start + samples_per_symbol
            signal[start:end] = np.cos(2 * np.pi * carrier_freq * t + phase)
        
        return signal
    
    @staticmethod
    def bpsk_demodulate(received: np.ndarray, carrier_freq: float,
                        sample_rate: float, symbol_duration: float) -> np.ndarray:
        samples_per_symbol = int(symbol_duration * sample_rate)
        t = np.arange(samples_per_symbol) / sample_rate
        
        n_symbols = len(received) // samples_per_symbol
        bits = np.zeros(n_symbols, dtype=int)
        
        for i in range(n_symbols):
            start = i * samples_per_symbol
            end = start + samples_per_symbol
            
            symbol = received[start:end]
            reference = np.cos(2 * np.pi * carrier_freq * t)
            
            correlation = np.sum(symbol * reference)
            bits[i] = 1 if correlation > 0 else 0
        
        return bits
    
    @staticmethod
    def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        if np.isrealobj(signal):
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        else:
            noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 
                                                   1j * np.random.randn(len(signal)))
        
        return signal + noise


def plot_time_varying_channel(tv_channel: TimeVaryingChannel):
    t_amp, amps = tv_channel.get_amplitude_series()
    t_delay, delays = tv_channel.get_delay_series()
    t_surf, surf = tv_channel.get_surface_height_series()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    axes[0].plot(t_amp, amps, 'b-', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Max Amplitude')
    axes[0].set_title('Channel Amplitude Variation Over Time')
    axes[0].grid(True)
    
    axes[1].plot(t_delay, delays * 1000, 'r-', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Min Delay (ms)')
    axes[1].set_title('Channel Delay Variation Over Time')
    axes[1].grid(True)
    
    axes[2].plot(t_surf, surf, 'g-', linewidth=2)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Surface Height (m)')
    axes[2].set_title('Sea Surface Height')
    axes[2].grid(True)
    
    plt.tight_layout()
    return fig


def plot_channel_snapshots(tv_channel: TimeVaryingChannel, n_snapshots: int = 5):
    indices = np.linspace(0, len(tv_channel.snapshots) - 1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(n_snapshots, 1, figsize=(14, 4 * n_snapshots))
    if n_snapshots == 1:
        axes = [axes]
    
    for idx, ax in zip(indices, axes):
        snap = tv_channel.snapshots[idx]
        t_axis = snap.time_axis * 1000
        
        ax.stem(t_axis, np.abs(snap.impulse_response), basefmt='r-', markerfmt='b.')
        ax.set_xlabel('Delay (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Time = {snap.time:.1f}s, Surface = {snap.surface_height:.2f}m')
        ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_communication_performance(tv_channel: TimeVaryingChannel):
    sample_rate = 48000
    symbol_duration = 0.001
    carrier_freq = 10000
    n_bits = 1000
    
    bits = np.random.randint(0, 2, n_bits)
    transmitted = CommunicationPerformance.bpsk_modulate(
        bits, carrier_freq, sample_rate, symbol_duration
    )
    
    snr_values = np.arange(0, 30, 2)
    ber_values = []
    
    for snr_db in snr_values:
        received_sig = CommunicationPerformance.add_awgn(transmitted, snr_db)
        
        received_tv = tv_channel.process_time_varying_signal(
            received_sig, sample_rate
        )
        
        received_bits = CommunicationPerformance.bpsk_demodulate(
            received_tv, carrier_freq, sample_rate, symbol_duration
        )
        
        min_len = min(len(bits), len(received_bits))
        ber = CommunicationPerformance.compute_ber(bits[:min_len], received_bits[:min_len])
        ber_values.append(ber)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(snr_values, ber_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit Error Rate')
    ax.set_title('Communication Performance: BER vs SNR')
    ax.grid(True)
    ax.set_ylim([1e-5, 1])
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Time-Varying Underwater Acoustic Channel Simulation")
    print("(with sea surface waves and internal waves)")
    print("=" * 60)
    print()
    
    base_ssp = create_munk_profile(depth_max=4000.0)
    print("Base Sound Speed Profile (Munk):")
    print(f"  Surface: {base_ssp.get_speed(0):.1f} m/s")
    print(f"  Channel axis: {np.min(base_ssp.speeds):.1f} m/s")
    print()
    
    print("Channel Configuration:")
    print("  Water depth: 4000 m")
    print("  Source depth: 100 m")
    print("  Receiver depth: 200 m")
    print("  Range: 5 km")
    print("  Carrier freq: 10 kHz")
    print()
    
    print("Environmental Parameters:")
    print("  Significant wave height: 1.0 m")
    print("  Internal wave amplitude: 10 m/s")
    print("  Simulation duration: 30 s")
    print("  Snapshot interval: 2 s")
    print()
    
    tv_channel = TimeVaryingChannel(
        base_ssp=base_ssp,
        water_depth=4000.0,
        source_depth=100.0,
        receiver_depth=200.0,
        range_km=5.0,
        relative_speed=5.0,
        carrier_freq=10000.0,
        simulation_duration=30.0,
        snapshot_interval=2.0,
        wave_height=1.0,
        internal_wave_amp=10.0,
        use_gaussian_beam=True
    )
    
    print("=" * 60)
    print("Channel Variation Statistics")
    print("=" * 60)
    
    t_amp, amps = tv_channel.get_amplitude_series()
    t_delay, delays = tv_channel.get_delay_series()
    t_surf, surf = tv_channel.get_surface_height_series()
    
    print(f"Amplitude variation:")
    print(f"  Mean: {np.mean(amps):.6f}")
    print(f"  Std: {np.std(amps):.6f}")
    print(f"  Min: {np.min(amps):.6f}")
    print(f"  Max: {np.max(amps):.6f}")
    print()
    
    print(f"Delay variation:")
    print(f"  Mean: {np.mean(delays)*1000:.2f} ms")
    print(f"  Std: {np.std(delays)*1000:.2f} ms")
    print(f"  Range: {(np.max(delays) - np.min(delays))*1000:.2f} ms")
    print()
    
    coherence_time = tv_channel.get_coherence_time()
    print(f"Coherence time: {coherence_time:.2f} s")
    print()
    
    print("=" * 60)
    print("Generating Plots")
    print("=" * 60)
    
    print("1. Time-varying channel parameters...")
    fig1 = plot_time_varying_channel(tv_channel)
    fig1.savefig('time_varying_channel.png', dpi=150)
    
    print("2. Channel snapshots...")
    fig2 = plot_channel_snapshots(tv_channel, n_snapshots=4)
    fig2.savefig('channel_snapshots.png', dpi=150)
    
    print("3. Communication performance...")
    fig3 = plot_communication_performance(tv_channel)
    fig3.savefig('communication_performance.png', dpi=150)
    
    print()
    print("Done! All plots saved:")
    print("  - time_varying_channel.png")
    print("  - channel_snapshots.png")
    print("  - communication_performance.png")
    
    print()
    print("=" * 60)
    print("Example: Signal Transmission Through Time-Varying Channel")
    print("=" * 60)
    
    sample_rate = 48000
    duration = 0.05
    t = np.arange(int(duration * sample_rate)) / sample_rate
    test_signal = np.sin(2 * np.pi * 10000 * t)
    
    received = tv_channel.process_time_varying_signal(test_signal, sample_rate)
    
    fig4, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(t * 1000, test_signal)
    axes[0].set_title('Transmitted Signal (10 kHz)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    axes[1].plot(t * 1000, np.real(received))
    axes[1].set_title('Received Signal (Through Time-Varying Channel)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)
    
    plt.tight_layout()
    fig4.savefig('signal_transmission_time_varying.png', dpi=150)
    print("  - signal_transmission_time_varying.png")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("Time-varying channel simulator includes:")
    print("  1. Sea surface waves (Pierson-Moskowitz spectrum)")
    print("     - Random wave components with realistic spectrum")
    print("     - Affects surface reflection and water depth")
    print()
    print("  2. Internal waves (mode-based perturbation)")
    print("     - Multi-mode internal wave model")
    print("     - Perturbs sound speed profile over time")
    print("     - Focused at thermocline depth (~500m)")
    print()
    print("  3. Time-varying impulse response")
    print("     - Channel snapshots at specified intervals")
    print("     - Amplitude and delay variations tracked")
    print()
    print("  4. Communication performance evaluation")
    print("     - BPSK modulation/demodulation")
    print("     - BER vs SNR curves")
    print("     - SNR computation")
    print()
    print("Applications:")
    print("  - Underwater acoustic communication system design")
    print("  - Channel equalizer performance evaluation")
    print("  - Adaptive algorithm testing")
    print("  - Robustness analysis against environmental changes")
    print()
    
    plt.close('all')


if __name__ == '__main__':
    main()
