import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from radial_velocity import RadarParameters


@dataclass
class RadarPosition:
    x: float
    y: float
    z: float = 0.0


@dataclass
class WindField3D:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2)

    @property
    def direction(self) -> np.ndarray:
        return np.arctan2(self.v, self.u)


class CoordinateTransformer:
    @staticmethod
    def radar_to_cartesian(azimuth: np.ndarray, elevation: np.ndarray,
                           range_km: np.ndarray, radar_pos: RadarPosition) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)

        x = range_km * np.cos(el_rad) * np.sin(az_rad) + radar_pos.x
        y = range_km * np.cos(el_rad) * np.cos(az_rad) + radar_pos.y
        z = range_km * np.sin(el_rad) + radar_pos.z

        return x, y, z

    @staticmethod
    def cartesian_to_radar(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           radar_pos: RadarPosition) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx = x - radar_pos.x
        dy = y - radar_pos.y
        dz = z - radar_pos.z

        range_km = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        azimuth = np.arctan2(dx, dy)
        azimuth = np.rad2deg(np.mod(azimuth, 2 * np.pi))
        elevation = np.rad2deg(np.arcsin(dz / np.maximum(range_km, 1e-6)))

        return azimuth, elevation, range_km

    @staticmethod
    def compute_radial_velocity(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                                azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)

        unit_r = np.array([
            np.cos(el_rad) * np.sin(az_rad),
            np.cos(el_rad) * np.cos(az_rad),
            np.sin(el_rad)
        ])

        if u.ndim == 1:
            vr = u * unit_r[0] + v * unit_r[1] + w * unit_r[2]
        else:
            vr = u * unit_r[0, ...] + v * unit_r[1, ...] + w * unit_r[2, ...]

        return vr


class DualDopplerSynthesizer:
    def __init__(self, radar1_pos: RadarPosition, radar2_pos: RadarPosition,
                 params1: RadarParameters, params2: RadarParameters):
        self.radar1_pos = radar1_pos
        self.radar2_pos = radar2_pos
        self.params1 = params1
        self.params2 = params2
        self.coord_transform = CoordinateTransformer()

    def compute_baseline(self) -> float:
        dx = self.radar2_pos.x - self.radar1_pos.x
        dy = self.radar2_pos.y - self.radar1_pos.y
        dz = self.radar2_pos.z - self.radar1_pos.z
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def inverse_wind_retrieval(self, vr1: np.ndarray, vr2: np.ndarray,
                               x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        az1, el1, r1 = self.coord_transform.cartesian_to_radar(x, y, z, self.radar1_pos)
        az2, el2, r2 = self.coord_transform.cartesian_to_radar(x, y, z, self.radar2_pos)

        az1_rad = np.deg2rad(az1)
        el1_rad = np.deg2rad(el1)
        az2_rad = np.deg2rad(az2)
        el2_rad = np.deg2rad(el2)

        a11 = np.cos(el1_rad) * np.sin(az1_rad)
        a12 = np.cos(el1_rad) * np.cos(az1_rad)
        a13 = np.sin(el1_rad)

        a21 = np.cos(el2_rad) * np.sin(az2_rad)
        a22 = np.cos(el2_rad) * np.cos(az2_rad)
        a23 = np.sin(el2_rad)

        det = a11 * a22 - a12 * a21

        mask_valid = np.abs(det) > 1e-3

        u = np.zeros_like(vr1)
        v = np.zeros_like(vr1)
        w = np.zeros_like(vr1)

        u[mask_valid] = (a22[mask_valid] * vr1[mask_valid] - a12[mask_valid] * vr2[mask_valid]) / det[mask_valid]
        v[mask_valid] = (a11[mask_valid] * vr2[mask_valid] - a21[mask_valid] * vr1[mask_valid]) / det[mask_valid]

        w_valid = (vr1[mask_valid] - a11[mask_valid] * u[mask_valid] - a12[mask_valid] * v[mask_valid]) / np.maximum(a13[mask_valid], 1e-3)
        w[mask_valid] = np.clip(w_valid, -20, 20)

        u[~mask_valid] = np.nan
        v[~mask_valid] = np.nan
        w[~mask_valid] = np.nan

        return u, v, w

    def synthesize_3d_wind(self, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                           vr1_field: np.ndarray, vr2_field: np.ndarray) -> WindField3D:
        X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

        u, v, w = self.inverse_wind_retrieval(vr1_field, vr2_field, X, Y, Z)

        return WindField3D(u=u, v=v, w=w, x=grid_x, y=grid_y, z=grid_z)


class KalmanFilterWindFusion:
    def __init__(self, nx: int, ny: int, nz: int, process_noise: float = 0.1,
                 measurement_noise: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.state_dim = 3
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.x = np.zeros((nx, ny, nz, 3))
        self.P = np.ones((nx, ny, nz, 3, 3)) * 100.0
        self.Q = np.eye(3) * process_noise
        self.R = np.eye(1) * measurement_noise

    def initialize(self, initial_wind: np.ndarray):
        self.x = initial_wind.copy()
        self.P = np.ones((self.nx, self.ny, self.nz, 3, 3)) * 10.0

    def predict(self):
        self.P = self.P + self.Q[np.newaxis, np.newaxis, np.newaxis, :, :]

    def update_radial(self, vr_meas: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray,
                      radar_idx: int = 0):
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)

        H = np.zeros((self.nx, self.ny, self.nz, 1, 3))
        H[..., 0, 0] = np.cos(el_rad) * np.sin(az_rad)
        H[..., 0, 1] = np.cos(el_rad) * np.cos(az_rad)
        H[..., 0, 2] = np.sin(el_rad)

        vr_pred = (self.x[..., 0] * H[..., 0, 0] +
                   self.x[..., 1] * H[..., 0, 1] +
                   self.x[..., 2] * H[..., 0, 2])

        y = vr_meas - vr_pred

        H_T = np.transpose(H, (0, 1, 2, 4, 3))
        S = np.matmul(np.matmul(H, self.P), H_T) + self.R

        S_inv = np.linalg.inv(S)
        K = np.matmul(np.matmul(self.P, H_T), S_inv)

        y_expanded = y[..., np.newaxis, np.newaxis]
        K_y = np.matmul(K, y_expanded)
        self.x = self.x + K_y[..., 0, 0]

        I = np.eye(3)[np.newaxis, np.newaxis, np.newaxis, :, :]
        I_KH = I - np.matmul(K, H)
        self.P = np.matmul(np.matmul(I_KH, self.P), np.transpose(I_KH, (0, 1, 2, 4, 3)))

    def get_wind_field(self, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray) -> WindField3D:
        return WindField3D(
            u=self.x[..., 0],
            v=self.x[..., 1],
            w=self.x[..., 2],
            x=grid_x,
            y=grid_y,
            z=grid_z
        )


class MultiRadarWindFusion:
    def __init__(self, radar_positions: List[RadarPosition],
                 radar_params: List[RadarParameters]):
        self.radar_positions = radar_positions
        self.radar_params = radar_params
        self.coord_transform = CoordinateTransformer()

    def fuse_radial_velocities(self, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                                vr_fields: List[np.ndarray], method: str = 'kalman',
                                **kwargs) -> WindField3D:
        nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)

        if method == 'simple_average':
            return self._simple_average_fusion(grid_x, grid_y, grid_z, vr_fields)
        elif method == 'kalman':
            return self._kalman_fusion(grid_x, grid_y, grid_z, vr_fields, **kwargs)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def _simple_average_fusion(self, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                               vr_fields: List[np.ndarray]) -> WindField3D:
        nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)
        X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

        u_list = []
        v_list = []
        w_list = []

        for i in range(len(self.radar_positions) - 1):
            synthesizer = DualDopplerSynthesizer(
                self.radar_positions[i],
                self.radar_positions[i + 1],
                self.radar_params[i],
                self.radar_params[i + 1]
            )
            u, v, w = synthesizer.inverse_wind_retrieval(
                vr_fields[i], vr_fields[i + 1], X, Y, Z
            )
            u_list.append(u)
            v_list.append(v)
            w_list.append(w)

        u_avg = np.nanmean(u_list, axis=0)
        v_avg = np.nanmean(v_list, axis=0)
        w_avg = np.nanmean(w_list, axis=0)

        return WindField3D(u=u_avg, v=v_avg, w=w_avg, x=grid_x, y=grid_y, z=grid_z)

    def _kalman_fusion(self, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                       vr_fields: List[np.ndarray], **kwargs) -> WindField3D:
        nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)
        X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

        kf = KalmanFilterWindFusion(nx, ny, nz,
                                    process_noise=kwargs.get('process_noise', 0.1),
                                    measurement_noise=kwargs.get('measurement_noise', 1.0))

        if len(self.radar_positions) >= 2:
            synthesizer = DualDopplerSynthesizer(
                self.radar_positions[0],
                self.radar_positions[1],
                self.radar_params[0],
                self.radar_params[1]
            )
            u0, v0, w0 = synthesizer.inverse_wind_retrieval(
                vr_fields[0], vr_fields[1], X, Y, Z
            )
            initial_wind = np.stack([u0, v0, w0], axis=-1)
            initial_wind[np.isnan(initial_wind)] = 0
            kf.initialize(initial_wind)

        for i, (radar_pos, vr_field) in enumerate(zip(self.radar_positions, vr_fields)):
            az, el, r = self.coord_transform.cartesian_to_radar(X, Y, Z, radar_pos)
            kf.predict()
            kf.update_radial(vr_field, az, el, radar_idx=i)

        return kf.get_wind_field(grid_x, grid_y, grid_z)


class VerticalWindShear:
    def __init__(self, wind_field: WindField3D):
        self.wind_field = wind_field
        self.dz = np.diff(wind_field.z)
        if len(self.dz) > 0:
            self.dz = np.mean(self.dz)
        else:
            self.dz = 1.0

    def compute_shear_magnitude(self) -> np.ndarray:
        u = self.wind_field.u
        v = self.wind_field.v

        du_dz = np.gradient(u, self.dz, axis=2)
        dv_dz = np.gradient(v, self.dz, axis=2)

        shear_mag = np.sqrt(du_dz ** 2 + dv_dz ** 2)
        return shear_mag

    def compute_shear_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        u = self.wind_field.u
        v = self.wind_field.v

        du_dz = np.gradient(u, self.dz, axis=2)
        dv_dz = np.gradient(v, self.dz, axis=2)

        return du_dz, dv_dz

    def compute_bulk_shear(self, z_bottom: float, z_top: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = self.wind_field.z
        idx_bottom = np.argmin(np.abs(z - z_bottom))
        idx_top = np.argmin(np.abs(z - z_top))

        u_bottom = self.wind_field.u[..., idx_bottom]
        v_bottom = self.wind_field.v[..., idx_bottom]
        u_top = self.wind_field.u[..., idx_top]
        v_top = self.wind_field.v[..., idx_top]

        delta_u = u_top - u_bottom
        delta_v = v_top - v_bottom
        delta_z = z[idx_top] - z[idx_bottom]

        shear_u = delta_u / delta_z
        shear_v = delta_v / delta_z
        shear_mag = np.sqrt(shear_u ** 2 + shear_v ** 2)

        return shear_u, shear_v, shear_mag

    def compute_helicity(self, z_bottom: float = 0.0, z_top: float = 3.0) -> np.ndarray:
        z = self.wind_field.z
        mask = (z >= z_bottom) & (z <= z_top)

        if not np.any(mask):
            return np.zeros_like(self.wind_field.u[..., 0])

        u = self.wind_field.u[..., mask]
        v = self.wind_field.v[..., mask]
        z_levels = z[mask]

        du_dz = np.gradient(u, z_levels, axis=2)
        dv_dz = np.gradient(v, z_levels, axis=2)

        helicity = -u * dv_dz + v * du_dz
        helicity_integrated = np.trapz(helicity, z_levels, axis=2)

        return helicity_integrated

    def detect_convective_hazard(self, shear_threshold: float = 10.0,
                                 helicity_threshold: float = 150.0) -> dict:
        shear_mag = self.compute_shear_magnitude()
        helicity = self.compute_helicity()

        max_shear = np.max(shear_mag, axis=2)
        hazard_mask = (max_shear > shear_threshold) | (helicity > helicity_threshold)

        return {
            'shear_magnitude': shear_mag,
            'helicity': helicity,
            'max_shear': max_shear,
            'hazard_mask': hazard_mask,
            'shear_threshold_exceeded': max_shear > shear_threshold,
            'helicity_threshold_exceeded': helicity > helicity_threshold
        }
