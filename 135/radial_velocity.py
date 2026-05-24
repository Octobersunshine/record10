import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RadarParameters:
    wavelength: float
    prf: float
    nyquist_velocity: Optional[float] = None

    def __post_init__(self):
        if self.nyquist_velocity is None:
            self.nyquist_velocity = self.wavelength * self.prf / 4.0

    @property
    def prt(self) -> float:
        return 1.0 / self.prf


class IQDataSimulator:
    def __init__(self, params: RadarParameters, num_gates: int = 1000, num_pulses: int = 128):
        self.params = params
        self.num_gates = num_gates
        self.num_pulses = num_pulses

    def simulate_iq(self, velocity_field: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        if velocity_field.shape != (self.num_gates,):
            raise ValueError(f"Velocity field shape {velocity_field.shape} does not match num_gates {self.num_gates}")

        iq_data = np.zeros((self.num_gates, self.num_pulses), dtype=np.complex128)
        phase = np.random.uniform(0, 2 * np.pi, self.num_gates)

        for pulse_idx in range(self.num_pulses):
            doppler_phase = 4 * np.pi * velocity_field * self.params.prt * pulse_idx / self.params.wavelength
            signal = np.exp(1j * (phase + doppler_phase))
            noise_power = 10 ** (-snr_db / 10.0)
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(self.num_gates) + 1j * np.random.randn(self.num_gates)
            )
            iq_data[:, pulse_idx] = signal + noise

        return iq_data


class PulsePairProcessor:
    def __init__(self, params: RadarParameters):
        self.params = params

    def autocorrelation(self, iq_data: np.ndarray, lag: int = 1) -> np.ndarray:
        if iq_data.ndim == 1:
            iq_data = iq_data.reshape(1, -1)

        num_gates, num_pulses = iq_data.shape
        r1 = np.zeros(num_gates, dtype=np.complex128)

        for gate in range(num_gates):
            valid_pulses = num_pulses - lag
            if valid_pulses > 0:
                r1[gate] = np.sum(
                    iq_data[gate, lag:] * np.conj(iq_data[gate, :-lag])
                ) / valid_pulses

        return r1

    def compute_velocity(self, iq_data: np.ndarray) -> np.ndarray:
        r1 = self.autocorrelation(iq_data, lag=1)
        phase_diff = np.angle(r1)
        velocity = (self.params.wavelength / (4 * np.pi * self.params.prt)) * phase_diff
        return velocity

    def compute_spectrum_width(self, iq_data: np.ndarray) -> np.ndarray:
        r0 = np.mean(np.abs(iq_data) ** 2, axis=1)
        r1 = self.autocorrelation(iq_data, lag=1)
        r1_abs = np.abs(r1)
        r1_abs = np.clip(r1_abs, 0, r0)
        ratio = np.sqrt(2 * (1 - r1_abs / r0))
        spectrum_width = (self.params.wavelength / (4 * np.pi * self.params.prt)) * ratio
        return spectrum_width

    def compute_power(self, iq_data: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(iq_data) ** 2, axis=1)

    def process(self, iq_data: np.ndarray) -> dict:
        return {
            'velocity': self.compute_velocity(iq_data),
            'spectrum_width': self.compute_spectrum_width(iq_data),
            'power': self.compute_power(iq_data),
        }


class QualityControl:
    def __init__(self, params: RadarParameters):
        self.params = params

    def create_snr_mask(self, power: np.ndarray, noise_floor: float, snr_threshold_db: float = 0.0) -> np.ndarray:
        snr = 10 * np.log10(power / noise_floor)
        return snr >= snr_threshold_db

    def create_texture_mask(self, velocity: np.ndarray, texture_threshold: float = 5.0) -> np.ndarray:
        if velocity.ndim == 1:
            texture = np.abs(np.gradient(velocity))
        else:
            grad_r, grad_a = np.gradient(velocity)
            texture = np.sqrt(grad_r ** 2 + grad_a ** 2)
        return texture <= texture_threshold

    def create_combined_mask(self, velocity: np.ndarray, power: np.ndarray,
                             noise_floor: float, snr_threshold_db: float = 0.0,
                             texture_threshold: float = 5.0) -> np.ndarray:
        snr_mask = self.create_snr_mask(power, noise_floor, snr_threshold_db)
        texture_mask = self.create_texture_mask(velocity, texture_threshold)
        return snr_mask & texture_mask


class GradientUnwrapper:
    def __init__(self, params: RadarParameters):
        self.params = params
        self.nyquist = params.nyquist_velocity
        self.vmax = 2 * self.nyquist

    def _wrap_to_nyquist(self, value: float) -> float:
        while value > self.nyquist:
            value -= self.vmax
        while value < -self.nyquist:
            value += self.vmax
        return value

    def unwrap_1d_gradient(self, velocity: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if mask is None:
            mask = np.ones_like(velocity, dtype=bool)

        unwrapped = velocity.copy().astype(float)
        valid = mask & ~np.isnan(velocity)

        if not np.any(valid):
            return unwrapped

        first_valid = np.argmax(valid)
        offset = 0

        for i in range(first_valid + 1, len(velocity)):
            if not valid[i]:
                continue

            prev_valid = i - 1
            while prev_valid >= 0 and not valid[prev_valid]:
                prev_valid -= 1

            if prev_valid >= 0:
                diff = velocity[i] - velocity[prev_valid]
                while diff > self.nyquist:
                    offset -= self.vmax
                    diff -= self.vmax
                while diff < -self.nyquist:
                    offset += self.vmax
                    diff += self.vmax

            unwrapped[i] = velocity[i] + offset

        offset = 0
        for i in range(first_valid - 1, -1, -1):
            if not valid[i]:
                continue

            next_valid = i + 1
            while next_valid < len(velocity) and not valid[next_valid]:
                next_valid += 1

            if next_valid < len(velocity):
                diff = velocity[i] - velocity[next_valid]
                while diff > self.nyquist:
                    offset -= self.vmax
                    diff -= self.vmax
                while diff < -self.nyquist:
                    offset += self.vmax
                    diff += self.vmax

            unwrapped[i] = velocity[i] + offset

        return unwrapped

    def unwrap_2d_gradient(self, velocity_field: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
        if velocity_field.ndim != 2:
            raise ValueError("Input must be 2D array")

        if mask is None:
            mask = np.ones_like(velocity_field, dtype=bool)

        unwrapped = velocity_field.copy().astype(float)
        num_rays, num_gates = velocity_field.shape

        for ray in range(num_rays):
            unwrapped[ray, :] = self.unwrap_1d_gradient(unwrapped[ray, :], mask[ray, :])

        for gate in range(num_gates):
            unwrapped[:, gate] = self.unwrap_1d_gradient(unwrapped[:, gate], mask[:, gate])

        return unwrapped


class RegionGrowingDealiaser:
    def __init__(self, params: RadarParameters):
        self.params = params
        self.nyquist = params.nyquist_velocity
        self.vmax = 2 * self.nyquist

    def _get_neighbors(self, idx: Tuple[int, int], shape: Tuple[int, int]) -> list:
        neighbors = []
        ray, gate = idx
        for dr, dg in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, ng = ray + dr, gate + dg
            if 0 <= nr < shape[0] and 0 <= ng < shape[1]:
                neighbors.append((nr, ng))
        return neighbors

    def _compute_offset(self, v1: float, v2: float) -> int:
        diff = v1 - v2
        if diff > self.nyquist:
            n = int((diff + self.nyquist) / self.vmax)
            return -n
        elif diff < -self.nyquist:
            n = int((-diff + self.nyquist) / self.vmax)
            return n
        return 0

    def dealias_2d_region_growing(self, velocity_field: np.ndarray,
                                  seed_point: Optional[Tuple[int, int]] = None,
                                  mask: Optional[np.ndarray] = None) -> np.ndarray:
        if velocity_field.ndim != 2:
            raise ValueError("Input must be 2D array")

        if mask is None:
            mask = np.ones_like(velocity_field, dtype=bool)

        mask = mask & ~np.isnan(velocity_field)

        if not np.any(mask):
            return velocity_field

        num_rays, num_gates = velocity_field.shape

        if seed_point is None:
            center_ray, center_gate = num_rays // 2, num_gates // 2
            while not mask[center_ray, center_gate]:
                candidates = np.argwhere(mask)
                if len(candidates) > 0:
                    center_ray, center_gate = candidates[0]
                    break
            seed_point = (center_ray, center_gate)

        if not mask[seed_point]:
            candidates = np.argwhere(mask)
            if len(candidates) > 0:
                seed_point = tuple(candidates[0])
            else:
                return velocity_field

        unwrapped = velocity_field.copy().astype(float)
        processed = np.zeros_like(velocity_field, dtype=bool)
        offsets = np.zeros_like(velocity_field, dtype=int)

        from collections import deque
        queue = deque([seed_point])
        processed[seed_point] = True

        while queue:
            current = queue.popleft()
            cr, cg = current

            for neighbor in self._get_neighbors(current, (num_rays, num_gates)):
                nr, ng = neighbor

                if not mask[nr, ng] or processed[nr, ng]:
                    continue

                v_current = unwrapped[cr, cg] + offsets[cr, cg] * self.vmax
                v_neighbor = velocity_field[nr, ng]

                offset = self._compute_offset(v_current, v_neighbor)
                offsets[nr, ng] = offsets[cr, cg] + offset

                processed[nr, ng] = True
                queue.append(neighbor)

        unwrapped += offsets * self.vmax

        return unwrapped

    def dealias_2d_multi_seed(self, velocity_field: np.ndarray,
                              mask: Optional[np.ndarray] = None,
                              num_seeds: int = 5) -> np.ndarray:
        if velocity_field.ndim != 2:
            raise ValueError("Input must be 2D array")

        if mask is None:
            mask = np.ones_like(velocity_field, dtype=bool)

        mask = mask & ~np.isnan(velocity_field)

        if not np.any(mask):
            return velocity_field

        unwrapped = velocity_field.copy()

        num_rays, num_gates = velocity_field.shape
        step_ray = max(1, num_rays // num_seeds)
        step_gate = max(1, num_gates // num_seeds)

        seeds = []
        for i in range(num_seeds):
            for j in range(num_seeds):
                r = min(i * step_ray, num_rays - 1)
                g = min(j * step_gate, num_gates - 1)
                if mask[r, g]:
                    seeds.append((r, g))

        if not seeds:
            candidates = np.argwhere(mask)
            if len(candidates) > 0:
                seeds = [tuple(candidates[0])]

        for seed in seeds:
            unwrapped = self.dealias_2d_region_growing(unwrapped, seed, mask)

        return unwrapped


class AdvancedVelocityDealiaser:
    def __init__(self, params: RadarParameters):
        self.params = params
        self.nyquist = params.nyquist_velocity
        self.gradient_unwrapper = GradientUnwrapper(params)
        self.region_growing = RegionGrowingDealiaser(params)
        self.quality_control = QualityControl(params)

    def dealias_1d_simple(self, velocity: np.ndarray,
                          reference_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        dealiased = velocity.copy()

        if reference_velocity is not None:
            for i in range(len(velocity)):
                if not np.isnan(velocity[i]) and not np.isnan(reference_velocity[i]):
                    while dealiased[i] - reference_velocity[i] > self.nyquist:
                        dealiased[i] -= 2 * self.nyquist
                    while dealiased[i] - reference_velocity[i] < -self.nyquist:
                        dealiased[i] += 2 * self.nyquist
        else:
            for i in range(1, len(velocity)):
                if not np.isnan(velocity[i]) and not np.isnan(velocity[i - 1]):
                    while dealiased[i] - dealiased[i - 1] > self.nyquist:
                        dealiased[i] -= 2 * self.nyquist
                    while dealiased[i] - dealiased[i - 1] < -self.nyquist:
                        dealiased[i] += 2 * self.nyquist

        return dealiased

    def dealias_1d_gradient(self, velocity: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.gradient_unwrapper.unwrap_1d_gradient(velocity, mask)

    def dealias_2d_gradient(self, velocity_field: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.gradient_unwrapper.unwrap_2d_gradient(velocity_field, mask)

    def dealias_2d_region_growing(self, velocity_field: np.ndarray,
                                  seed_point: Optional[Tuple[int, int]] = None,
                                  mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.region_growing.dealias_2d_region_growing(velocity_field, seed_point, mask)

    def dealias_2d_multi_seed(self, velocity_field: np.ndarray,
                              mask: Optional[np.ndarray] = None,
                              num_seeds: int = 5) -> np.ndarray:
        return self.region_growing.dealias_2d_multi_seed(velocity_field, mask, num_seeds)

    def dealias_2d_hybrid(self, velocity_field: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        grad_result = self.dealias_2d_gradient(velocity_field, mask)
        final_result = self.dealias_2d_region_growing(grad_result, mask=mask)
        return final_result
