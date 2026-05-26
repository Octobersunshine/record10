import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SoundSpeedProfile:
    depths: np.ndarray
    speeds: np.ndarray
    
    def __post_init__(self):
        self.depths = np.asarray(self.depths)
        self.speeds = np.asarray(self.speeds)
        self._interpolator = interp1d(self.depths, self.speeds, 
                                      kind='linear', fill_value='extrapolate')
    
    def get_speed(self, depth: float) -> float:
        return float(self._interpolator(depth))
    
    def get_gradient(self, depth: float) -> float:
        idx = np.searchsorted(self.depths, depth) - 1
        idx = np.clip(idx, 0, len(self.depths) - 2)
        return (self.speeds[idx + 1] - self.speeds[idx]) / (self.depths[idx + 1] - self.depths[idx])
    
    def get_curvature(self, depth: float) -> float:
        c = self.get_speed(depth)
        dc_dz = self.get_gradient(depth)
        return -dc_dz / c


@dataclass
class RayPath:
    launch_angle: float
    times: np.ndarray
    depths: np.ndarray
    ranges: np.ndarray
    arc_lengths: np.ndarray
    beam_widths: np.ndarray
    amplitudes: np.ndarray
    num_surface_bounces: int = 0
    num_bottom_bounces: int = 0
    
    @property
    def travel_time(self) -> float:
        return self.times[-1]
    
    @property
    def final_range(self) -> float:
        return self.ranges[-1]
    
    @property
    def final_depth(self) -> float:
        return self.depths[-1]
    
    @property
    def total_arc_length(self) -> float:
        return self.arc_lengths[-1]


@dataclass
class MultipathComponent:
    delay: float
    amplitude: float
    phase: float
    doppler_shift: float
    launch_angle: float
    arrival_angle: float
    num_surface_bounces: int
    num_bottom_bounces: int
    beam_contribution: float = 0.0


class GaussianBeamTracer:
    def __init__(self, ssp: SoundSpeedProfile, 
                 water_depth: float,
                 bottom_reflection_coeff: float = 0.8,
                 surface_reflection_coeff: float = 0.99,
                 absorption_coeff: float = 0.0001,
                 initial_beam_width: float = 5.0):
        self.ssp = ssp
        self.water_depth = water_depth
        self.bottom_reflection_coeff = bottom_reflection_coeff
        self.surface_reflection_coeff = surface_reflection_coeff
        self.absorption_coeff = absorption_coeff
        self.initial_beam_width = initial_beam_width
    
    def _compute_beam_width(self, arc_length: float, freq: float, c: float) -> float:
        wavelength = c / freq
        w0 = self.initial_beam_width
        rayleigh_range = np.pi * w0**2 / wavelength
        w = w0 * np.sqrt(1.0 + (arc_length / rayleigh_range)**2)
        return w
    
    def _compute_beam_field(self, depth: float, ray_depth: float, beam_width: float) -> complex:
        r_perp = abs(depth - ray_depth)
        if beam_width < 1e-10:
            return 0.0
        gaussian_envelope = np.exp(-(r_perp / beam_width)**2)
        phase_term = np.exp(-1j * np.pi * r_perp**2 / (wavelength * beam_width**2))
        return gaussian_envelope * phase_term
    
    def trace_ray(self, 
                  source_depth: float,
                  launch_angle: float,
                  max_range: float,
                  freq: float = 10000.0,
                  max_bounces: int = 10,
                  dt: float = 0.01) -> RayPath:
        c0 = self.ssp.get_speed(source_depth)
        angle_rad = np.deg2rad(launch_angle)
        wavelength = c0 / freq
        
        times = [0.0]
        depths = [source_depth]
        ranges = [0.0]
        arc_lengths = [0.0]
        beam_widths = [self.initial_beam_width]
        amplitudes = [1.0]
        
        vz = c0 * np.sin(angle_rad)
        vx = c0 * np.cos(angle_rad)
        
        current_depth = source_depth
        current_range = 0.0
        current_time = 0.0
        current_arc = 0.0
        current_amplitude = 1.0
        
        surface_bounces = 0
        bottom_bounces = 0
        
        direction = 1 if vz > 0 else -1
        
        while current_range < max_range:
            c = self.ssp.get_speed(current_depth)
            dc_dz = self.ssp.get_gradient(current_depth)
            
            current_angle = np.arctan2(vz, vx)
            
            vz += direction * dc_dz * dt
            speed_mag = np.sqrt(vx**2 + vz**2)
            
            vx = c * np.cos(current_angle)
            vz = c * np.sin(current_angle) * direction
            
            dz = vz * dt
            dr = vx * dt
            step_dist = np.sqrt(dz**2 + dr**2)
            
            new_depth = current_depth + dz
            
            if new_depth <= 0:
                surface_bounces += 1
                if surface_bounces > max_bounces:
                    break
                    
                t_to_surface = -current_depth / vz if vz != 0 else dt
                step_dist = np.sqrt((vx * t_to_surface)**2 + current_depth**2)
                current_range += vx * t_to_surface
                current_time += t_to_surface
                current_arc += step_dist
                
                vz = -vz
                direction = -direction
                current_depth = 0.0
                current_amplitude *= self.surface_reflection_coeff
                
            elif new_depth >= self.water_depth:
                bottom_bounces += 1
                if bottom_bounces > max_bounces:
                    break
                    
                t_to_bottom = (self.water_depth - current_depth) / vz if vz != 0 else dt
                step_dist = np.sqrt((vx * t_to_bottom)**2 + (self.water_depth - current_depth)**2)
                current_range += vx * t_to_bottom
                current_time += t_to_bottom
                current_arc += step_dist
                
                vz = -vz
                direction = -direction
                current_depth = self.water_depth
                current_amplitude *= self.bottom_reflection_coeff
                
            else:
                current_depth = new_depth
                current_range += dr
                current_time += dt
                current_arc += step_dist
            
            current_amplitude *= np.exp(-self.absorption_coeff * step_dist / 1000)
            
            beam_width = self._compute_beam_width(current_arc, freq, c)
            
            times.append(current_time)
            depths.append(current_depth)
            ranges.append(current_range)
            arc_lengths.append(current_arc)
            beam_widths.append(beam_width)
            
            if current_range > 0:
                geometric_spreading = 1.0 / np.sqrt(current_range + 1.0)
            else:
                geometric_spreading = 1.0
            amplitudes.append(current_amplitude * geometric_spreading)
        
        return RayPath(
            launch_angle=launch_angle,
            times=np.array(times),
            depths=np.array(depths),
            ranges=np.array(ranges),
            arc_lengths=np.array(arc_lengths),
            beam_widths=np.array(beam_widths),
            amplitudes=np.array(amplitudes),
            num_surface_bounces=surface_bounces,
            num_bottom_bounces=bottom_bounces
        )
    
    def compute_beam_contribution(self, 
                                   ray: RayPath, 
                                   target_range: float, 
                                   target_depth: float,
                                   freq: float) -> Tuple[float, float, float]:
        range_idx = np.searchsorted(ray.ranges, target_range)
        if range_idx <= 0 or range_idx >= len(ray.ranges):
            return 0.0, 0.0, 0.0
        
        r1, r2 = ray.ranges[range_idx-1], ray.ranges[range_idx]
        t1, t2 = ray.times[range_idx-1], ray.times[range_idx]
        d1, d2 = ray.depths[range_idx-1], ray.depths[range_idx]
        s1, s2 = ray.arc_lengths[range_idx-1], ray.arc_lengths[range_idx]
        w1, w2 = ray.beam_widths[range_idx-1], ray.beam_widths[range_idx]
        a1, a2 = ray.amplitudes[range_idx-1], ray.amplitudes[range_idx]
        
        if r2 > r1:
            alpha = (target_range - r1) / (r2 - r1)
        else:
            alpha = 1.0
        
        delay = t1 + alpha * (t2 - t1)
        ray_depth = d1 + alpha * (d2 - d1)
        arc_len = s1 + alpha * (s2 - s1)
        beam_width = w1 + alpha * (w2 - w1)
        ray_amp = a1 + alpha * (a2 - a1)
        
        r_perp = abs(target_depth - ray_depth)
        
        if beam_width < 1e-10:
            return 0.0, delay, ray_amp
        
        wavelength = self.ssp.get_speed(ray_depth) / freq
        rayleigh_range = np.pi * self.initial_beam_width**2 / wavelength
        
        gaussian_envelope = np.exp(-(r_perp / beam_width)**2)
        
        phase_shift = np.pi * r_perp**2 / (wavelength * beam_width)
        gouy_phase = np.arctan2(arc_len, rayleigh_range)
        
        total_phase = phase_shift - gouy_phase
        
        beam_amp = ray_amp * gaussian_envelope / np.sqrt(beam_width / self.initial_beam_width)
        
        return beam_amp, delay, total_phase
    
    def find_eigenrays(self,
                       source_depth: float,
                       receiver_depth: float,
                       receiver_range: float,
                       freq: float = 10000.0,
                       angle_min: float = -80.0,
                       angle_max: float = 80.0,
                       num_angles: int = 160,
                       depth_tolerance: float = 100.0,
                       max_bounces: int = 10) -> List[RayPath]:
        angles = np.linspace(angle_min, angle_max, num_angles)
        eigenrays = []
        
        for angle in angles:
            ray = self.trace_ray(source_depth, angle, receiver_range * 1.1, freq, max_bounces)
            
            range_idx = np.searchsorted(ray.ranges, receiver_range)
            if range_idx > 0 and range_idx < len(ray.ranges):
                r1, r2 = ray.ranges[range_idx-1], ray.ranges[range_idx]
                d1, d2 = ray.depths[range_idx-1], ray.depths[range_idx]
                if r2 > r1:
                    alpha = (receiver_range - r1) / (r2 - r1)
                    depth_interp = d1 + alpha * (d2 - d1)
                else:
                    depth_interp = ray.depths[range_idx]
                
                if abs(depth_interp - receiver_depth) < depth_tolerance:
                    eigenrays.append(ray)
        
        return eigenrays
    
    def compute_field(self,
                      source_depth: float,
                      receiver_depth: float,
                      receiver_range: float,
                      freq: float = 10000.0,
                      angle_min: float = -80.0,
                      angle_max: float = 80.0,
                      num_angles: int = 200,
                      max_bounces: int = 10) -> Tuple[complex, List[RayPath]]:
        angles = np.linspace(angle_min, angle_max, num_angles)
        total_field = 0.0 + 0.0j
        all_rays = []
        
        d_angle = np.deg2rad(angle_max - angle_min) / (num_angles - 1)
        
        for angle in angles:
            ray = self.trace_ray(source_depth, angle, receiver_range * 1.1, freq, max_bounces)
            all_rays.append(ray)
            
            beam_amp, delay, phase = self.compute_beam_contribution(
                ray, receiver_range, receiver_depth, freq
            )
            
            if abs(beam_amp) > 1e-15:
                c_avg = self.ssp.get_speed(receiver_depth)
                k = 2 * np.pi * freq / c_avg
                travel_phase = k * c_avg * delay
                total_phase = travel_phase + phase
                total_field += beam_amp * np.exp(1j * total_phase) * d_angle
        
        total_field *= np.sqrt(freq / c_avg) / np.sqrt(8 * np.pi)
        
        return total_field, all_rays


class UnderwaterAcousticChannel:
    def __init__(self,
                 ssp: SoundSpeedProfile,
                 water_depth: float,
                 source_depth: float,
                 receiver_depth: float,
                 range_km: float,
                 relative_speed: float = 0.0,
                 carrier_freq: float = 10000.0,
                 use_gaussian_beam: bool = True):
        self.ssp = ssp
        self.water_depth = water_depth
        self.source_depth = source_depth
        self.receiver_depth = receiver_depth
        self.range = range_km * 1000
        self.relative_speed = relative_speed
        self.carrier_freq = carrier_freq
        self.use_gaussian_beam = use_gaussian_beam
        
        self.tracer = GaussianBeamTracer(ssp, water_depth)
        self.multipath_components: List[MultipathComponent] = []
        self._all_rays: List[RayPath] = []
        self._compute_channel()
    
    def _compute_doppler_shift(self, delay: float, launch_angle: float) -> float:
        c_avg = np.mean(self.ssp.speeds)
        angle_rad = np.deg2rad(launch_angle)
        doppler = (self.relative_speed * np.cos(angle_rad) / c_avg) * self.carrier_freq
        return doppler
    
    def _compute_channel(self):
        if self.use_gaussian_beam:
            self._compute_channel_gaussian_beam()
        else:
            self._compute_channel_ray()
    
    def _compute_channel_gaussian_beam(self):
        eigenrays = self.tracer.find_eigenrays(
            self.source_depth,
            self.receiver_depth,
            self.range,
            freq=self.carrier_freq
        )
        
        self._all_rays = eigenrays
        
        for ray in eigenrays:
            beam_amp, delay, beam_phase = self.tracer.compute_beam_contribution(
                ray, self.range, self.receiver_depth, self.carrier_freq
            )
            
            if abs(beam_amp) < 1e-15:
                continue
            
            range_idx = np.searchsorted(ray.ranges, self.range)
            if range_idx >= len(ray.ranges):
                range_idx = len(ray.ranges) - 1
            if range_idx <= 0:
                range_idx = 1
            
            arrival_angle = np.rad2deg(np.arctan2(
                ray.depths[range_idx] - ray.depths[range_idx-1],
                ray.ranges[range_idx] - ray.ranges[range_idx-1] + 1e-10
            ))
            
            doppler_shift = self._compute_doppler_shift(delay, ray.launch_angle)
            
            total_phase = beam_phase + np.random.uniform(0, 2*np.pi)
            
            component = MultipathComponent(
                delay=delay,
                amplitude=beam_amp,
                phase=total_phase,
                doppler_shift=doppler_shift,
                launch_angle=ray.launch_angle,
                arrival_angle=arrival_angle,
                num_surface_bounces=ray.num_surface_bounces,
                num_bottom_bounces=ray.num_bottom_bounces,
                beam_contribution=abs(beam_amp)
            )
            self.multipath_components.append(component)
        
        self.multipath_components.sort(key=lambda x: x.delay)
    
    def _compute_channel_ray(self):
        eigenrays = self.tracer.find_eigenrays(
            self.source_depth,
            self.receiver_depth,
            self.range,
            freq=self.carrier_freq
        )
        
        self._all_rays = eigenrays
        
        for ray in eigenrays:
            range_idx = np.searchsorted(ray.ranges, self.range)
            if range_idx >= len(ray.ranges):
                range_idx = len(ray.ranges) - 1
            
            if range_idx > 0:
                r1, r2 = ray.ranges[range_idx-1], ray.ranges[range_idx]
                t1, t2 = ray.times[range_idx-1], ray.times[range_idx]
                a1, a2 = ray.amplitudes[range_idx-1], ray.amplitudes[range_idx]
                
                if r2 > r1:
                    alpha = (self.range - r1) / (r2 - r1)
                    delay = t1 + alpha * (t2 - t1)
                    amplitude = a1 + alpha * (a2 - a1)
                else:
                    delay = t2
                    amplitude = a2
            else:
                delay = ray.times[range_idx]
                amplitude = ray.amplitudes[range_idx]
            
            arrival_angle = np.rad2deg(np.arctan2(
                ray.depths[range_idx] - ray.depths[max(0, range_idx-1)],
                ray.ranges[range_idx] - ray.ranges[max(0, range_idx-1)] + 1e-10
            ))
            
            doppler_shift = self._compute_doppler_shift(delay, ray.launch_angle)
            
            component = MultipathComponent(
                delay=delay,
                amplitude=amplitude,
                phase=np.random.uniform(0, 2*np.pi),
                doppler_shift=doppler_shift,
                launch_angle=ray.launch_angle,
                arrival_angle=arrival_angle,
                num_surface_bounces=ray.num_surface_bounces,
                num_bottom_bounces=ray.num_bottom_bounces
            )
            self.multipath_components.append(component)
        
        self.multipath_components.sort(key=lambda x: x.delay)
    
    def get_impulse_response(self, 
                             sample_rate: float = 48000,
                             duration_ms: float = None) -> Tuple[np.ndarray, np.ndarray]:
        if duration_ms is None:
            if self.multipath_components:
                max_delay = max(c.delay for c in self.multipath_components)
                duration_ms = max(100.0, max_delay * 1000 * 1.1)
            else:
                duration_ms = 10000.0
        
        duration = duration_ms / 1000.0
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        
        h = np.zeros(num_samples, dtype=complex)
        
        for comp in self.multipath_components:
            sample_idx = int(comp.delay * sample_rate)
            if sample_idx < num_samples:
                doppler_phase = 2 * np.pi * comp.doppler_shift * t[sample_idx]
                h[sample_idx] = comp.amplitude * np.exp(1j * (comp.phase + doppler_phase))
        
        return t, h
    
    def get_power_delay_profile(self,
                                sample_rate: float = 48000,
                                duration_ms: float = None) -> Tuple[np.ndarray, np.ndarray]:
        t, h = self.get_impulse_response(sample_rate, duration_ms)
        pdp = np.abs(h)**2
        return t, pdp
    
    def get_transmission_loss(self, freq: float = None) -> float:
        if freq is None:
            freq = self.carrier_freq
        
        field, _ = self.tracer.compute_field(
            self.source_depth,
            self.receiver_depth,
            self.range,
            freq=freq
        )
        
        pressure = abs(field)
        if pressure < 1e-15:
            return 100.0
        tl = -20 * np.log10(pressure)
        return max(0.0, tl)
    
    def get_coherence_bandwidth(self) -> float:
        if len(self.multipath_components) < 2:
            return 0
        
        delays = [c.delay for c in self.multipath_components]
        amplitudes = np.array([abs(c.amplitude) for c in self.multipath_components])
        
        weight_sum = np.sum(amplitudes)
        if weight_sum < 1e-10:
            amplitudes = np.ones_like(amplitudes)
            weight_sum = np.sum(amplitudes)
        
        mean_delay = np.average(delays, weights=amplitudes)
        rms_delay_spread = np.sqrt(np.average(
            (np.array(delays) - mean_delay)**2, 
            weights=amplitudes
        ))
        
        if rms_delay_spread > 0:
            return 1 / (5 * rms_delay_spread)
        return 0
    
    def process_signal(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        if self.multipath_components:
            max_delay = max(c.delay for c in self.multipath_components)
            duration_ms = max(50.0, max_delay * 1000 * 1.1 + len(signal)/sample_rate*1000)
        else:
            duration_ms = len(signal)/sample_rate*1000 + 50
        
        t, h = self.get_impulse_response(sample_rate, duration_ms)
        
        output = np.zeros(len(signal) + len(h) - 1, dtype=complex)
        
        for i, comp in enumerate(self.multipath_components):
            delay_samples = int(comp.delay * sample_rate)
            if delay_samples >= len(output):
                continue
            
            doppler_factor = 1 + comp.doppler_shift / self.carrier_freq
            t_signal = np.arange(len(signal)) / sample_rate
            doppler_t = t_signal * doppler_factor
            
            interpolator = interp1d(t_signal, signal, kind='linear', 
                                   fill_value=0.0, bounds_error=False)
            doppler_signal = interpolator(doppler_t)
            
            end_idx = min(delay_samples + len(signal), len(output))
            output[delay_samples:end_idx] += comp.amplitude * doppler_signal[:end_idx - delay_samples] * np.exp(1j * comp.phase)
        
        return output[:len(signal)]
    
    def plot_ray_paths(self, max_rays: int = 10, show_beams: bool = True):
        angles = np.linspace(-40, 40, max_rays)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for angle in angles:
            ray = self.tracer.trace_ray(self.source_depth, angle, self.range * 1.1, self.carrier_freq)
            
            ax.plot(ray.ranges / 1000, ray.depths, linewidth=1, alpha=0.7)
            
            if show_beams and len(ray.beam_widths) > 0:
                n_points = len(ray.ranges)
                upper = np.minimum(ray.depths + ray.beam_widths, self.water_depth)
                lower = np.maximum(ray.depths - ray.beam_widths, 0)
                ax.fill_between(ray.ranges / 1000, lower, upper, alpha=0.1)
        
        ax.axhline(y=0, color='blue', linestyle='--', label='Surface')
        ax.axhline(y=self.water_depth, color='brown', linestyle='--', label='Bottom')
        ax.plot(0, self.source_depth, 'ro', markersize=10, label='Source')
        ax.plot(self.range / 1000, self.receiver_depth, 'go', markersize=10, label='Receiver')
        
        ax.set_xlabel('Range (km)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Gaussian Beam Ray Paths')
        ax.legend()
        ax.grid(True)
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_beam_width(self, num_rays: int = 5):
        angles = np.linspace(-20, 20, num_rays)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for angle in angles:
            ray = self.tracer.trace_ray(self.source_depth, angle, self.range * 1.1, self.carrier_freq)
            
            axes[0].plot(ray.arc_lengths, ray.beam_widths, 
                        label=f'{angle:.1f}°')
            axes[1].plot(ray.ranges / 1000, ray.beam_widths,
                        label=f'{angle:.1f}°')
        
        axes[0].set_xlabel('Arc Length (m)')
        axes[0].set_ylabel('Beam Width (m)')
        axes[0].set_title('Beam Width vs Arc Length')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_xlabel('Range (km)')
        axes[1].set_ylabel('Beam Width (m)')
        axes[1].set_title('Beam Width vs Range')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_transmission_loss(self, freq: float = None, num_depths: int = 30):
        if freq is None:
            freq = self.carrier_freq
        
        test_depths = np.linspace(10, self.water_depth - 10, num_depths)
        tl_values = []
        
        for depth in test_depths:
            field, _ = self.tracer.compute_field(
                self.source_depth, depth, self.range, freq=freq
            )
            pressure = abs(field)
            if pressure < 1e-15:
                tl_values.append(100.0)
            else:
                tl_values.append(-20 * np.log10(pressure))
        
        fig, ax = plt.subplots(figsize=(8, 10))
        
        ax.plot(tl_values, test_depths, 'b-', linewidth=2)
        ax.set_xlabel('Transmission Loss (dB)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Transmission Loss at {self.range/1000:.1f} km, {freq/1000:.1f} kHz')
        ax.grid(True)
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_impulse_response(self, sample_rate: float = 48000, duration_ms: float = None):
        if duration_ms is None:
            if self.multipath_components:
                max_delay = max(c.delay for c in self.multipath_components)
                duration_ms = max(100.0, max_delay * 1000 * 1.1)
            else:
                duration_ms = 100.0
        t, h = self.get_impulse_response(sample_rate, duration_ms)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].stem(t * 1000, np.abs(h), basefmt='r-')
        axes[0].set_xlabel('Delay (ms)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Channel Impulse Response (Gaussian Beam)')
        axes[0].grid(True)
        
        pdp = np.abs(h)**2
        pdp_db = 10 * np.log10(pdp + 1e-10)
        axes[1].plot(t * 1000, pdp_db)
        axes[1].set_xlabel('Delay (ms)')
        axes[1].set_ylabel('Power (dB)')
        axes[1].set_title('Power Delay Profile')
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_multipath_info(self):
        if not self.multipath_components:
            return None
        
        delays = [c.delay * 1000 for c in self.multipath_components]
        amplitudes = [20 * np.log10(abs(c.amplitude) + 1e-10) for c in self.multipath_components]
        doppler_shifts = [c.doppler_shift for c in self.multipath_components]
        beam_contribs = [c.beam_contribution for c in self.multipath_components]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].scatter(delays, amplitudes, c=beam_contribs, cmap='viridis', s=80, alpha=0.7)
        axes[0].set_xlabel('Delay (ms)')
        axes[0].set_ylabel('Amplitude (dB)')
        axes[0].set_title('Multipath Components - Delay vs Amplitude (colored by beam contribution)')
        axes[0].grid(True)
        
        axes[1].scatter(delays, doppler_shifts, c='r', s=50, alpha=0.7)
        axes[1].set_xlabel('Delay (ms)')
        axes[1].set_ylabel('Doppler Shift (Hz)')
        axes[1].set_title('Multipath Components - Delay vs Doppler Shift')
        axes[1].grid(True)
        
        angles = [c.launch_angle for c in self.multipath_components]
        axes[2].scatter(angles, delays, c=beam_contribs, cmap='viridis', s=80, alpha=0.7)
        axes[2].set_xlabel('Launch Angle (degrees)')
        axes[2].set_ylabel('Delay (ms)')
        axes[2].set_title('Multipath Components - Launch Angle vs Delay')
        axes[2].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison_ray_vs_beam(self):
        beam_channel = UnderwaterAcousticChannel(
            ssp=self.ssp,
            water_depth=self.water_depth,
            source_depth=self.source_depth,
            receiver_depth=self.receiver_depth,
            range_km=self.range / 1000,
            relative_speed=self.relative_speed,
            carrier_freq=self.carrier_freq,
            use_gaussian_beam=True
        )
        
        ray_channel = UnderwaterAcousticChannel(
            ssp=self.ssp,
            water_depth=self.water_depth,
            source_depth=self.source_depth,
            receiver_depth=self.receiver_depth,
            range_km=self.range / 1000,
            relative_speed=self.relative_speed,
            carrier_freq=self.carrier_freq,
            use_gaussian_beam=False
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        delays_beam = [c.delay * 1000 for c in beam_channel.multipath_components]
        amps_beam = [20 * np.log10(abs(c.amplitude) + 1e-10) for c in beam_channel.multipath_components]
        
        delays_ray = [c.delay * 1000 for c in ray_channel.multipath_components]
        amps_ray = [20 * np.log10(abs(c.amplitude) + 1e-10) for c in ray_channel.multipath_components]
        
        axes[0, 0].stem(delays_beam, amps_beam, basefmt='b-')
        axes[0, 0].set_xlabel('Delay (ms)')
        axes[0, 0].set_ylabel('Amplitude (dB)')
        axes[0, 0].set_title('Gaussian Beam Method')
        axes[0, 0].grid(True)
        
        axes[0, 1].stem(delays_ray, amps_ray, basefmt='r-')
        axes[0, 1].set_xlabel('Delay (ms)')
        axes[0, 1].set_ylabel('Amplitude (dB)')
        axes[0, 1].set_title('Standard Ray Method')
        axes[0, 1].grid(True)
        
        n_depths = 20
        test_depths = np.linspace(10, self.water_depth - 10, n_depths)
        tl_beam = []
        tl_ray = []
        
        for depth in test_depths:
            field_b, _ = beam_channel.tracer.compute_field(
                self.source_depth, depth, self.range, freq=self.carrier_freq
            )
            field_r, _ = ray_channel.tracer.compute_field(
                self.source_depth, depth, self.range, freq=self.carrier_freq
            )
            
            p_b = abs(field_b)
            p_r = abs(field_r)
            
            tl_beam.append(-20 * np.log10(p_b) if p_b > 1e-15 else 100)
            tl_ray.append(-20 * np.log10(p_r) if p_r > 1e-15 else 100)
        
        axes[1, 0].plot(tl_beam, test_depths, 'b-', linewidth=2, label='Gaussian Beam')
        axes[1, 0].plot(tl_ray, test_depths, 'r--', linewidth=2, label='Standard Ray')
        axes[1, 0].set_xlabel('Transmission Loss (dB)')
        axes[1, 0].set_ylabel('Depth (m)')
        axes[1, 0].set_title('Transmission Loss Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].invert_yaxis()
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, 'Gaussian Beam vs Standard Ray:', 
                       fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.75, '• Gaussian Beam smooths caustics via finite beam width', 
                       fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.65, '• Standard Ray diverges at caustics', 
                       fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.55, '• Beam includes Gouy phase shift', 
                       fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.45, '• Beam diffraction adds natural energy spreading', 
                       fontsize=10, transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        return fig


def create_munk_profile(depth_max: float = 5000.0, num_points: int = 200) -> SoundSpeedProfile:
    z = np.linspace(0, depth_max, num_points)
    z_km = z / 1000
    
    c0 = 1500.0
    eps = 0.00737
    z0 = 1.3
    
    c = c0 * (1 + eps * (z_km - z0 + np.exp(-(z_km - z0))))
    
    return SoundSpeedProfile(depths=z, speeds=c)


def create_isovelocity_profile(depth_max: float = 5000.0, 
                                speed: float = 1500.0) -> SoundSpeedProfile:
    z = np.array([0, depth_max])
    c = np.array([speed, speed])
    return SoundSpeedProfile(depths=z, speeds=c)


def create_linear_gradient_profile(depth_max: float = 5000.0,
                                    surface_speed: float = 1520.0,
                                    bottom_speed: float = 1480.0) -> SoundSpeedProfile:
    z = np.array([0, depth_max])
    c = np.array([surface_speed, bottom_speed])
    return SoundSpeedProfile(depths=z, speeds=c)


def main():
    print("=== Underwater Acoustic Channel Simulation (Gaussian Beam) ===")
    print()
    
    ssp = create_munk_profile(depth_max=4000.0)
    print("Sound Speed Profile (Munk profile):")
    print(f"  Surface speed: {ssp.get_speed(0):.2f} m/s")
    print(f"  Deep sound channel axis speed: {np.min(ssp.speeds):.2f} m/s")
    print()
    
    print("=" * 60)
    print("Method 1: Gaussian Beam Tracing (with caustic smoothing)")
    print("=" * 60)
    
    channel = UnderwaterAcousticChannel(
        ssp=ssp,
        water_depth=4000.0,
        source_depth=100.0,
        receiver_depth=200.0,
        range_km=5.0,
        relative_speed=5.0,
        carrier_freq=10000.0,
        use_gaussian_beam=True
    )
    
    print(f"Channel Configuration:")
    print(f"  Water depth: {channel.water_depth} m")
    print(f"  Source depth: {channel.source_depth} m")
    print(f"  Receiver depth: {channel.receiver_depth} m")
    print(f"  Range: {channel.range / 1000:.1f} km")
    print(f"  Relative speed: {channel.relative_speed} m/s")
    print(f"  Carrier frequency: {channel.carrier_freq / 1000:.1f} kHz")
    print()
    
    print(f"Found {len(channel.multipath_components)} multipath components (Gaussian Beam):")
    for i, comp in enumerate(channel.multipath_components[:10]):
        print(f"  Path {i+1}: delay={comp.delay*1000:.2f}ms, "
              f"amp={abs(comp.amplitude):.6f}, "
              f"beam_contrib={comp.beam_contribution:.6f}, "
              f"doppler={comp.doppler_shift:.1f}Hz, "
              f"angle={comp.launch_angle:.1f}°, "
              f"bounces=(S:{comp.num_surface_bounces}, B:{comp.num_bottom_bounces})")
    if len(channel.multipath_components) > 10:
        print(f"  ... and {len(channel.multipath_components) - 10} more paths")
    print()
    
    coherence_bw = channel.get_coherence_bandwidth()
    print(f"Coherence Bandwidth (approx): {coherence_bw / 1000:.2f} kHz")
    
    tl = channel.get_transmission_loss()
    print(f"Transmission Loss: {tl:.2f} dB")
    print()
    
    print("Generating plots...")
    fig1 = channel.plot_ray_paths()
    fig2 = channel.plot_impulse_response()
    fig3 = channel.plot_multipath_info()
    fig4 = channel.plot_beam_width()
    fig5 = channel.plot_transmission_loss()
    fig6 = channel.plot_comparison_ray_vs_beam()
    
    print("Testing signal transmission...")
    sample_rate = 48000
    duration = 0.01
    t_signal = np.arange(int(duration * sample_rate)) / sample_rate
    test_signal = np.sin(2 * np.pi * 10000 * t_signal)
    
    received_signal = channel.process_signal(test_signal, sample_rate)
    
    fig7, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(t_signal * 1000, test_signal)
    axes[0].set_title('Transmitted Signal (10 kHz tone)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    t_received = np.arange(len(received_signal)) / sample_rate
    axes[1].plot(t_received * 1000, np.real(received_signal))
    axes[1].set_title('Received Signal (with multipath and Doppler, Gaussian Beam)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    print("Saving plots to files...")
    fig1.savefig('ray_paths_gaussian_beam.png', dpi=150)
    fig2.savefig('impulse_response_gaussian_beam.png', dpi=150)
    if fig3 is not None:
        fig3.savefig('multipath_info_gaussian_beam.png', dpi=150)
    fig4.savefig('beam_width_evolution.png', dpi=150)
    fig5.savefig('transmission_loss.png', dpi=150)
    fig6.savefig('comparison_ray_vs_beam.png', dpi=150)
    fig7.savefig('signal_transmission_gaussian_beam.png', dpi=150)
    
    print()
    print("=" * 60)
    print("Method 2: Standard Ray Tracing (for comparison)")
    print("=" * 60)
    
    channel_ray = UnderwaterAcousticChannel(
        ssp=ssp,
        water_depth=4000.0,
        source_depth=100.0,
        receiver_depth=200.0,
        range_km=5.0,
        relative_speed=5.0,
        carrier_freq=10000.0,
        use_gaussian_beam=False
    )
    
    print(f"Found {len(channel_ray.multipath_components)} multipath components (Standard Ray):")
    for i, comp in enumerate(channel_ray.multipath_components[:10]):
        print(f"  Path {i+1}: delay={comp.delay*1000:.2f}ms, "
              f"amp={abs(comp.amplitude):.6f}, "
              f"doppler={comp.doppler_shift:.1f}Hz, "
              f"angle={comp.launch_angle:.1f}°, "
              f"bounces=(S:{comp.num_surface_bounces}, B:{comp.num_bottom_bounces})")
    print()
    
    tl_ray = channel_ray.get_transmission_loss()
    print(f"Transmission Loss (Standard Ray): {tl_ray:.2f} dB")
    print(f"Transmission Loss (Gaussian Beam): {tl:.2f} dB")
    print(f"Difference: {abs(tl - tl_ray):.2f} dB")
    print()
    
    print("Done! All plots saved.")
    print("  - ray_paths_gaussian_beam.png")
    print("  - impulse_response_gaussian_beam.png")
    print("  - multipath_info_gaussian_beam.png")
    print("  - beam_width_evolution.png")
    print("  - transmission_loss.png")
    print("  - comparison_ray_vs_beam.png")
    print("  - signal_transmission_gaussian_beam.png")
    
    plt.close('all')


if __name__ == '__main__':
    main()
