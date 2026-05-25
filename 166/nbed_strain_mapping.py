import numpy as np
from scipy import optimize, ndimage
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import find_peaks
from typing import List, Tuple, Optional, Dict, Callable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StrainTensor:
    eps_xx: float
    eps_yy: float
    eps_xy: float
    gamma_xy: float
    eps_1: float
    eps_2: float
    theta: float
    error: float


@dataclass
class NBEDSpot:
    x: float
    y: float
    intensity: float
    subpixel_x: float
    subpixel_y: float
    hkl: Optional[Tuple[int, int, int]] = None
    gx: float = 0.0
    gy: float = 0.0


@dataclass
class NBEDPattern:
    row: int
    col: int
    position_xy: Tuple[float, float]
    image: np.ndarray
    spots: List[NBEDSpot]
    center_x: float
    center_y: float
    strain: Optional[StrainTensor] = None
    valid: bool = True


@dataclass
class StrainMap:
    scan_shape: Tuple[int, int]
    pixel_size: float
    step_size: float
    eps_xx: np.ndarray
    eps_yy: np.ndarray
    eps_xy: np.ndarray
    gamma_xy: np.ndarray
    eps_1: np.ndarray
    eps_2: np.ndarray
    theta: np.ndarray
    error_map: np.ndarray
    valid_mask: np.ndarray


class SubpixelSpotDetector:
    @staticmethod
    def fit_gaussian_2d(image: np.ndarray, center_x: int, center_y: int, 
                        window_size: int = 7) -> Tuple[float, float, float]:
        half = window_size // 2
        x_start = max(0, center_x - half)
        x_end = min(image.shape[1], center_x + half + 1)
        y_start = max(0, center_y - half)
        y_end = min(image.shape[0], center_y + half + 1)
        
        patch = image[y_start:y_end, x_start:x_end]
        
        if patch.size < 9:
            return float(center_x), float(center_y), 0.0
        
        patch = patch.astype(np.float64)
        patch = patch - patch.min()
        if patch.max() > 0:
            patch = patch / patch.max()
        
        y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
        
        def gaussian_function(params):
            A, x0, y0, sigma_x, sigma_y, offset = params
            model = offset + A * np.exp(
                -((x_grid - x0)**2 / (2 * sigma_x**2) + 
                  (y_grid - y0)**2 / (2 * sigma_y**2))
            )
            return np.sum((patch - model)**2)
        
        initial_guess = [1.0, center_x, center_y, 1.5, 1.5, 0.0]
        bounds = [(0.1, 2.0), (x_start, x_end-1), (y_start, y_end-1), 
                  (0.5, 5.0), (0.5, 5.0), (0.0, 0.5)]
        
        try:
            result = optimize.minimize(gaussian_function, initial_guess, 
                                      method='L-BFGS-B', bounds=bounds,
                                      options={'maxiter': 100})
            if result.success:
                return result.x[1], result.x[2], result.fun
        except:
            pass
        
        return float(center_x), float(center_y), float('inf')
    
    @staticmethod
    def centroid_2d(image: np.ndarray, center_x: int, center_y: int,
                    window_size: int = 5) -> Tuple[float, float]:
        half = window_size // 2
        x_start = max(0, center_x - half)
        x_end = min(image.shape[1], center_x + half + 1)
        y_start = max(0, center_y - half)
        y_end = min(image.shape[0], center_y + half + 1)
        
        patch = image[y_start:y_end, x_start:x_end].astype(np.float64)
        
        if patch.sum() <= 0:
            return float(center_x), float(center_y)
        
        y_grid, x_grid = np.mgrid[y_start:y_end, x_start:x_end]
        
        total = patch.sum()
        cx = (x_grid * patch).sum() / total
        cy = (y_grid * patch).sum() / total
        
        return cx, cy
    
    @staticmethod
    def detect_spots(image: np.ndarray, threshold: float = 0.1,
                    min_distance: int = 5, max_spots: int = 50,
                    method: str = 'gaussian') -> List[NBEDSpot]:
        image_filtered = gaussian_filter(image, sigma=1.0)
        
        local_max = maximum_filter(image_filtered, size=min_distance)
        maxima = (image_filtered == local_max)
        
        threshold_abs = threshold * image_filtered.max()
        maxima = maxima & (image_filtered > threshold_abs)
        
        y_coords, x_coords = np.where(maxima)
        intensities = image_filtered[y_coords, x_coords]
        
        sort_idx = np.argsort(intensities)[::-1]
        x_coords = x_coords[sort_idx[:max_spots]]
        y_coords = y_coords[sort_idx[:max_spots]]
        intensities = intensities[sort_idx[:max_spots]]
        
        spots = []
        for x, y, intensity in zip(x_coords, y_coords, intensities):
            if method == 'gaussian':
                sx, sy, err = SubpixelSpotDetector.fit_gaussian_2d(image, int(x), int(y))
            else:
                sx, sy = SubpixelSpotDetector.centroid_2d(image, int(x), int(y))
                err = 0.0
            
            spot = NBEDSpot(
                x=float(x), y=float(y),
                intensity=float(intensity),
                subpixel_x=sx, subpixel_y=sy
            )
            spots.append(spot)
        
        return spots


class NBEDStrainAnalyzer:
    def __init__(self, camera_length: float, wavelength: float = 0.02508,
                 scan_step_size_nm: float = 1.0,
                 camera_pixel_size_um: float = 10.0):
        self.camera_length = camera_length
        self.wavelength = wavelength
        self.scan_step_size_nm = scan_step_size_nm
        self.camera_pixel_size_um = camera_pixel_size_um
        
        self.reference_center: Optional[Tuple[float, float]] = None
        self.reference_spots: Optional[List[NBEDSpot]] = None
        self.reference_g_vectors: Optional[List[Tuple[float, float]]] = None
        
        self.scan_patterns: List[NBEDPattern] = []
        self.strain_map: Optional[StrainMap] = None
        
    def set_reference(self, reference_image: np.ndarray,
                     threshold: float = 0.1,
                     center_method: str = 'auto'):
        spots = SubpixelSpotDetector.detect_spots(
            reference_image, threshold=threshold, method='gaussian'
        )
        
        if center_method == 'auto':
            center_spot = max(spots, key=lambda s: s.intensity)
            center_x, center_y = center_spot.subpixel_x, center_spot.subpixel_y
        else:
            center_x = reference_image.shape[1] / 2
            center_y = reference_image.shape[0] / 2
        
        self.reference_center = (center_x, center_y)
        self.reference_spots = spots
        
        g_vectors = []
        for spot in spots:
            dx = spot.subpixel_x - center_x
            dy = spot.subpixel_y - center_y
            r_pixels = np.sqrt(dx**2 + dy**2)
            r_mm = r_pixels * self.camera_pixel_size_um / 1000.0
            
            if r_mm > 0:
                d = (self.wavelength * self.camera_length) / r_mm
                g = 1.0 / d
                
                theta = np.arctan2(dy, dx)
                gx = g * np.cos(theta)
                gy = g * np.sin(theta)
                
                g_vectors.append((gx, gy))
                spot.gx = gx
                spot.gy = gy
        
        self.reference_g_vectors = g_vectors
        
        return spots
    
    def analyze_single_pattern(self, image: np.ndarray, row: int = 0, col: int = 0,
                              threshold: float = 0.1) -> NBEDPattern:
        spots = SubpixelSpotDetector.detect_spots(
            image, threshold=threshold, method='gaussian'
        )
        
        center_spot = max(spots, key=lambda s: s.intensity)
        center_x, center_y = center_spot.subpixel_x, center_spot.subpixel_y
        
        if self.reference_center is None:
            self.reference_center = (center_x, center_y)
        
        for spot in spots:
            dx = spot.subpixel_x - center_x
            dy = spot.subpixel_y - center_y
            r_pixels = np.sqrt(dx**2 + dy**2)
            r_mm = r_pixels * self.camera_pixel_size_um / 1000.0
            
            if r_mm > 0:
                d = (self.wavelength * self.camera_length) / r_mm
                g = 1.0 / d
                
                theta = np.arctan2(dy, dx)
                spot.gx = g * np.cos(theta)
                spot.gy = g * np.sin(theta)
        
        pattern = NBEDPattern(
            row=row, col=col,
            position_xy=(col * self.scan_step_size_nm, row * self.scan_step_size_nm),
            image=image, spots=spots,
            center_x=center_x, center_y=center_y
        )
        
        if self.reference_g_vectors is not None:
            strain = self._calculate_strain_from_spots(spots, center_x, center_y)
            pattern.strain = strain
        
        return pattern
    
    def _calculate_strain_from_spots(self, spots: List[NBEDSpot],
                                     center_x: float, center_y: float) -> Optional[StrainTensor]:
        if self.reference_g_vectors is None or len(self.reference_g_vectors) < 2:
            return None
        
        current_g = []
        for spot in spots:
            dx = spot.subpixel_x - center_x
            dy = spot.subpixel_y - center_y
            r_pixels = np.sqrt(dx**2 + dy**2)
            r_mm = r_pixels * self.camera_pixel_size_um / 1000.0
            
            if r_mm > 0.1:
                d = (self.wavelength * self.camera_length) / r_mm
                g = 1.0 / d
                
                theta = np.arctan2(dy, dx)
                current_g.append((g * np.cos(theta), g * np.sin(theta), g, theta))
        
        if len(current_g) < 2:
            return None
        
        ref_g_magnitudes = [np.sqrt(gx**2 + gy**2) 
                           for gx, gy in self.reference_g_vectors]
        ref_g_thetas = [np.arctan2(gy, gx) 
                       for gx, gy in self.reference_g_vectors]
        
        strain_measurements = []
        
        for gx_curr, gy_curr, g_curr, theta_curr in current_g:
            for i, (gx_ref, gy_ref) in enumerate(self.reference_g_vectors):
                g_ref = np.sqrt(gx_ref**2 + gy_ref**2)
                theta_ref = np.arctan2(gy_ref, gx_ref)
                
                theta_diff = abs(theta_curr - theta_ref)
                theta_diff = min(theta_diff, 2*np.pi - theta_diff)
                
                if theta_diff < 0.2:
                    delta_g = (g_curr - g_ref) / g_ref if g_ref != 0 else 0
                    strain_measurements.append((theta_ref, delta_g, g_ref))
                    break
        
        if len(strain_measurements) < 2:
            return None
        
        angles = np.array([m[0] for m in strain_measurements])
        deltas = np.array([m[1] for m in strain_measurements])
        
        def strain_model(params, angles):
            eps_xx, eps_yy, eps_xy = params
            return (eps_xx * np.cos(angles)**2 + 
                    eps_yy * np.sin(angles)**2 + 
                    2 * eps_xy * np.cos(angles) * np.sin(angles))
        
        def objective(params):
            return np.sum((deltas - strain_model(params, angles))**2)
        
        initial_guess = [0.0, 0.0, 0.0]
        bounds = [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
        
        try:
            result = optimize.minimize(objective, initial_guess,
                                      method='L-BFGS-B', bounds=bounds)
            if result.success:
                eps_xx, eps_yy, eps_xy = result.x
                
                gamma_xy = 2 * eps_xy
                
                eps_1 = (eps_xx + eps_yy) / 2 + np.sqrt(
                    ((eps_xx - eps_yy) / 2)**2 + eps_xy**2
                )
                eps_2 = (eps_xx + eps_yy) / 2 - np.sqrt(
                    ((eps_xx - eps_yy) / 2)**2 + eps_xy**2
                )
                
                theta_rad = 0.5 * np.arctan2(2 * eps_xy, eps_xx - eps_yy)
                theta_deg = np.degrees(theta_rad)
                
                return StrainTensor(
                    eps_xx=eps_xx, eps_yy=eps_yy,
                    eps_xy=eps_xy, gamma_xy=gamma_xy,
                    eps_1=eps_1, eps_2=eps_2,
                    theta=theta_deg,
                    error=result.fun / len(deltas)
                )
        except:
            pass
        
        return None
    
    def analyze_scan(self, scan_data: np.ndarray,
                    ref_position: Optional[Tuple[int, int]] = None,
                    threshold: float = 0.1,
                    show_progress: bool = True) -> StrainMap:
        if scan_data.ndim != 4:
            raise ValueError("扫描数据应为4D数组: (rows, cols, height, width)")
        
        n_rows, n_cols = scan_data.shape[0], scan_data.shape[1]
        
        if ref_position is None:
            ref_row, ref_col = n_rows // 2, n_cols // 2
        else:
            ref_row, ref_col = ref_position
        
        print(f"设置参考图案: 位置 ({ref_row}, {ref_col})")
        self.set_reference(scan_data[ref_row, ref_col], threshold=threshold)
        
        patterns = []
        iterator = tqdm(range(n_rows)) if show_progress else range(n_rows)
        
        for row in iterator:
            for col in range(n_cols):
                pattern = self.analyze_single_pattern(
                    scan_data[row, col], row, col, threshold=threshold
                )
                patterns.append(pattern)
        
        self.scan_patterns = patterns
        
        eps_xx = np.zeros((n_rows, n_cols))
        eps_yy = np.zeros((n_rows, n_cols))
        eps_xy = np.zeros((n_rows, n_cols))
        gamma_xy = np.zeros((n_rows, n_cols))
        eps_1 = np.zeros((n_rows, n_cols))
        eps_2 = np.zeros((n_rows, n_cols))
        theta = np.zeros((n_rows, n_cols))
        error_map = np.zeros((n_rows, n_cols))
        valid_mask = np.zeros((n_rows, n_cols), dtype=bool)
        
        for pattern in patterns:
            r, c = pattern.row, pattern.col
            if pattern.strain is not None:
                eps_xx[r, c] = pattern.strain.eps_xx
                eps_yy[r, c] = pattern.strain.eps_yy
                eps_xy[r, c] = pattern.strain.eps_xy
                gamma_xy[r, c] = pattern.strain.gamma_xy
                eps_1[r, c] = pattern.strain.eps_1
                eps_2[r, c] = pattern.strain.eps_2
                theta[r, c] = pattern.strain.theta
                error_map[r, c] = pattern.strain.error
                valid_mask[r, c] = True
                pattern.valid = True
        
        self.strain_map = StrainMap(
            scan_shape=(n_rows, n_cols),
            pixel_size=self.scan_step_size_nm,
            step_size=self.scan_step_size_nm,
            eps_xx=eps_xx, eps_yy=eps_yy,
            eps_xy=eps_xy, gamma_xy=gamma_xy,
            eps_1=eps_1, eps_2=eps_2,
            theta=theta,
            error_map=error_map,
            valid_mask=valid_mask
        )
        
        return self.strain_map
    
    def plot_strain_maps(self, save_path: Optional[str] = None,
                        strain_range: Optional[Tuple[float, float]] = None):
        if self.strain_map is None:
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        strain_components = [
            ('ε_xx', self.strain_map.eps_xx),
            ('ε_yy', self.strain_map.eps_yy),
            ('ε_xy', self.strain_map.eps_xy),
            ('γ_xy', self.strain_map.gamma_xy),
            ('ε₁ (主应变1)', self.strain_map.eps_1),
            ('ε₂ (主应变2)', self.strain_map.eps_2),
            ('主应变方向 (°)', self.strain_map.theta),
            ('拟合误差', self.strain_map.error_map)
        ]
        
        for i, (title, data) in enumerate(strain_components):
            ax = axes[i // 4, i % 4]
            
            if 'ε' in title and strain_range is not None:
                vmin, vmax = strain_range
            elif 'ε' in title:
                data_valid = data[self.strain_map.valid_mask]
                if len(data_valid) > 0:
                    vmin = np.percentile(data_valid, 5)
                    vmax = np.percentile(data_valid, 95)
                else:
                    vmin, vmax = data.min(), data.max()
            else:
                vmin, vmax = None, None
            
            im = ax.imshow(data, cmap='seismic' if 'ε' in title else 'viridis',
                          aspect='equal', origin='lower', vmin=vmin, vmax=vmax)
            
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel(f'X (步, {self.scan_step_size_nm} nm/步)')
            ax.set_ylabel(f'Y (步, {self.scan_step_size_nm} nm/步)')
            
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        
        plt.suptitle(f'NBED 应变场映射 (扫描: {self.strain_map.scan_shape[0]}×{self.strain_map.scan_shape[1]})',
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_vector_field(self, save_path: Optional[str] = None,
                         scale: float = 1000.0):
        if self.strain_map is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        n_rows, n_cols = self.strain_map.scan_shape
        X, Y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        
        valid = self.strain_map.valid_mask
        
        dx = self.strain_map.eps_xx * X + self.strain_map.eps_xy * Y
        dy = self.strain_map.eps_xy * X + self.strain_map.eps_yy * Y
        
        ax = axes[0]
        strain_mag = np.sqrt(self.strain_map.eps_xx**2 + self.strain_map.eps_yy**2)
        im = ax.imshow(strain_mag * 100, cmap='hot', aspect='equal', origin='lower')
        ax.quiver(X[valid], Y[valid], 
                 dx[valid] * scale, dy[valid] * scale,
                 color='cyan', scale=1.0, width=0.005, alpha=0.7)
        ax.set_title('位移矢量场 (放大显示)', fontsize=12)
        ax.set_xlabel('X (扫描步)')
        ax.set_ylabel('Y (扫描步)')
        plt.colorbar(im, ax=ax, label='应变大小 (%)')
        
        ax = axes[1]
        theta_rad = np.radians(self.strain_map.theta)
        u = np.cos(theta_rad)
        v = np.sin(theta_rad)
        im = ax.imshow(self.strain_map.eps_1 * 100, cmap='seismic', aspect='equal', origin='lower')
        ax.quiver(X[valid], Y[valid], u[valid], v[valid],
                 color='white', scale=30, width=0.003, alpha=0.8)
        ax.set_title('主应变方向', fontsize=12)
        ax.set_xlabel('X (扫描步)')
        ax.set_ylabel('Y (扫描步)')
        plt.colorbar(im, ax=ax, label='主应变 ε₁ (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_strain_summary(self):
        if self.strain_map is None:
            return
        
        valid = self.strain_map.valid_mask
        
        print("=" * 70)
        print("NBED 应变场分析报告")
        print("=" * 70)
        print(f"\n扫描参数:")
        print(f"  扫描范围: {self.strain_map.scan_shape[0]} × {self.strain_map.scan_shape[1]} 像素")
        print(f"  步长: {self.scan_step_size_nm} nm")
        print(f"  视场: {self.strain_map.scan_shape[1] * self.scan_step_size_nm:.1f} × "
              f"{self.strain_map.scan_shape[0] * self.scan_step_size_nm:.1f} nm²")
        print(f"  有效像素: {valid.sum()}/{valid.size} ({valid.sum()/valid.size*100:.1f}%)")
        
        print(f"\n应变统计 (x100%):")
        print(f"  ε_xx: 平均 = {np.mean(self.strain_map.eps_xx[valid])*100:.3f}%, "
              f"范围 = [{np.min(self.strain_map.eps_xx[valid])*100:.3f}, "
              f"{np.max(self.strain_map.eps_xx[valid])*100:.3f}]%")
        print(f"  ε_yy: 平均 = {np.mean(self.strain_map.eps_yy[valid])*100:.3f}%, "
              f"范围 = [{np.min(self.strain_map.eps_yy[valid])*100:.3f}, "
              f"{np.max(self.strain_map.eps_yy[valid])*100:.3f}]%")
        print(f"  γ_xy: 平均 = {np.mean(self.strain_map.gamma_xy[valid])*100:.3f}%, "
              f"范围 = [{np.min(self.strain_map.gamma_xy[valid])*100:.3f}, "
              f"{np.max(self.strain_map.gamma_xy[valid])*100:.3f}]%")
        print(f"  ε₁: 最大主应变 = {np.max(self.strain_map.eps_1[valid])*100:.3f}%")
        print(f"  ε₂: 最小主应变 = {np.min(self.strain_map.eps_2[valid])*100:.3f}%")
        
        print("\n" + "=" * 70)


def generate_synthetic_nbed_data(scan_shape: Tuple[int, int] = (20, 20),
                                image_shape: Tuple[int, int] = (256, 256),
                                strain_field: str = 'uniform',
                                a_lattice: float = 4.08,
                                camera_length: float = 200.0,
                                wavelength: float = 0.02508,
                                pixel_size_um: float = 10.0) -> np.ndarray:
    n_rows, n_cols = scan_shape
    img_h, img_w = image_shape
    scan_data = np.zeros((n_rows, n_cols, img_h, img_w), dtype=np.float32)
    
    Y, X = np.mgrid[0:img_h, 0:img_w]
    center_y, center_x = img_h // 2, img_w // 2
    
    for row in range(n_rows):
        for col in range(n_cols):
            rx = (col - n_cols/2) / (n_cols/2)
            ry = (row - n_rows/2) / (n_rows/2)
            
            if strain_field == 'uniform':
                eps_xx, eps_yy, eps_xy = 0.01, -0.005, 0.0
            elif strain_field == 'bending':
                eps_xx = 0.02 * ry
                eps_yy = -0.01 * ry
                eps_xy = 0.0
            elif strain_field == 'shear':
                eps_xx = 0.0
                eps_yy = 0.0
                eps_xy = 0.01 * rx
            elif strain_field == 'dislocation':
                r = np.sqrt(rx**2 + ry**2) + 0.1
                theta = np.arctan2(ry, rx)
                eps_xx = 0.005 * np.cos(theta) / r
                eps_yy = -0.005 * np.cos(theta) / r
                eps_xy = 0.005 * np.sin(theta) / r
            else:
                eps_xx, eps_yy, eps_xy = 0, 0, 0
            
            image = np.zeros((img_h, img_w), dtype=np.float32)
            
            for h in range(-4, 5):
                for k in range(-4, 5):
                    if h == 0 and k == 0:
                        continue
                    
                    if (h % 2 != k % 2):
                        continue
                    
                    s = h**2 + k**2
                    if s == 0:
                        continue
                    
                    d = a_lattice / np.sqrt(s)
                    r_mm = (wavelength * camera_length) / d
                    r_pixels = r_mm * 1000.0 / pixel_size_um
                    
                    theta = np.arctan2(k, h)
                    
                    gx = np.cos(theta)
                    gy = np.sin(theta)
                    
                    gx_strained = gx * (1 + eps_xx) + gy * eps_xy
                    gy_strained = gx * eps_xy + gy * (1 + eps_yy)
                    
                    g_mag = np.sqrt(gx_strained**2 + gy_strained**2)
                    if g_mag > 0:
                        gx_strained /= g_mag
                        gy_strained /= g_mag
                    
                    r_scaled = r_pixels / g_mag
                    
                    sx = center_x + r_scaled * gx_strained
                    sy = center_y + r_scaled * gy_strained
                    
                    if 0 <= sx < img_w and 0 <= sy < img_h:
                        sigma = 2.5
                        intensity = 100.0 / (1 + s * 0.1)
                        image += intensity * np.exp(
                            -((X - sx)**2 + (Y - sy)**2) / (2 * sigma**2)
                        )
            
            image += np.random.normal(0, 1.0, image.shape)
            image = np.maximum(0, image)
            
            scan_data[row, col] = image
    
    return scan_data


def main():
    print("NBED 应变场映射演示")
    print("=" * 70)
    
    camera_length = 200.0
    wavelength = 0.02508
    scan_step_nm = 2.0
    pixel_size_um = 10.0
    
    print("\n生成合成NBED数据集 (弯曲应变场)...")
    scan_data = generate_synthetic_nbed_data(
        scan_shape=(16, 16),
        image_shape=(128, 128),
        strain_field='bending',
        a_lattice=4.08,
        camera_length=camera_length,
        wavelength=wavelength,
        pixel_size_um=pixel_size_um
    )
    
    print(f"数据集尺寸: {scan_data.shape}")
    
    analyzer = NBEDStrainAnalyzer(
        camera_length=camera_length,
        wavelength=wavelength,
        scan_step_size_nm=scan_step_nm,
        camera_pixel_size_um=pixel_size_um
    )
    
    print("\n分析应变场...")
    strain_map = analyzer.analyze_scan(
        scan_data,
        ref_position=(8, 8),
        threshold=0.15,
        show_progress=True
    )
    
    analyzer.print_strain_summary()
    
    print("\n生成应变场映射图...")
    analyzer.plot_strain_maps(strain_range=(-0.02, 0.02))
    
    print("\n生成矢量场图...")
    analyzer.plot_vector_field(scale=500)
    
    print("\n分析完成!")


if __name__ == "__main__":
    main()
