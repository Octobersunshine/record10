import numpy as np
from scipy import optimize, signal
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Optional, Dict, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class CrystalStructure(Enum):
    SIMPLE_CUBIC = "sc"
    BCC = "bcc"
    FCC = "fcc"
    DIAMOND = "diamond"
    HCP = "hcp"
    ORTHORHOMBIC = "orthorhombic"


@dataclass
class DiffractionSpot:
    x: float
    y: float
    intensity: float = 1.0
    h: Optional[int] = None
    k: Optional[int] = None
    l: Optional[int] = None
    error: float = 0.0
    confidence: float = 0.0
    is_higher_order_laue: bool = False
    is_secondary_diffraction: bool = False
    match_score: float = 0.0


@dataclass
class LatticeParameters:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    
    def reciprocal_metric_tensor(self) -> np.ndarray:
        alpha = np.radians(self.alpha)
        beta = np.radians(self.beta)
        gamma = np.radians(self.gamma)
        
        V = self.a * self.b * self.c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        
        g11 = (self.b**2 * self.c**2 * np.sin(alpha)**2) / V**2
        g22 = (self.a**2 * self.c**2 * np.sin(beta)**2) / V**2
        g33 = (self.a**2 * self.b**2 * np.sin(gamma)**2) / V**2
        g12 = (self.a * self.b * self.c**2 * 
               (np.cos(alpha) * np.cos(beta) - np.cos(gamma))) / V**2
        g13 = (self.a * self.b**2 * self.c * 
               (np.cos(gamma) * np.cos(alpha) - np.cos(beta))) / V**2
        g23 = (self.a**2 * self.b * self.c * 
               (np.cos(beta) * np.cos(gamma) - np.cos(alpha))) / V**2
        
        return np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])
    
    def dspacing(self, h: int, k: int, l: int) -> float:
        g_tensor = self.reciprocal_metric_tensor()
        hkl = np.array([h, k, l])
        g_squared = hkl @ g_tensor @ hkl.T
        if g_squared <= 0:
            return float('inf')
        return 1.0 / np.sqrt(g_squared)


@dataclass
class CrystalOrientation:
    zone_axis: np.ndarray
    rotation_angle: float
    confidence: float
    refinement_rms: float = 0.0
    template_correlation: float = 0.0


class StructureFactorCalculator:
    @staticmethod
    def calculate_f2(h: int, k: int, l: int, structure: CrystalStructure) -> float:
        if structure == CrystalStructure.SIMPLE_CUBIC:
            return 1.0
        
        elif structure == CrystalStructure.BCC:
            if (h + k + l) % 2 == 0:
                return 4.0
            return 0.0
        
        elif structure == CrystalStructure.FCC:
            if (h % 2 == k % 2) and (k % 2 == l % 2):
                return 16.0
            return 0.0
        
        elif structure == CrystalStructure.DIAMOND:
            if (h % 2 == k % 2) and (k % 2 == l % 2):
                if (h + k + l) % 4 == 0:
                    return 64.0
                elif (h + k + l) % 2 == 0:
                    return 32.0
            return 0.0
        
        elif structure == CrystalStructure.HCP:
            if (h + 2*k) % 3 == 0 and l % 2 == 0:
                return 4.0
            elif (h + 2*k) % 3 != 0 and l % 2 != 0:
                return 3.0
            return 0.0
        
        else:
            return 1.0
    
    @staticmethod
    def is_extinct(h: int, k: int, l: int, structure: CrystalStructure) -> bool:
        return StructureFactorCalculator.calculate_f2(h, k, l, structure) < 1e-6


class TemplateMatcher:
    def __init__(self, image_size: Tuple[int, int] = (512, 512), 
                 pixel_scale: float = 1.0):
        self.image_size = image_size
        self.pixel_scale = pixel_scale
        self.center = (image_size[0] // 2, image_size[1] // 2)
    
    def generate_template(self, spots: List[Tuple[float, float, float]],
                         sigma: float = 3.0) -> np.ndarray:
        template = np.zeros(self.image_size, dtype=np.float64)
        
        for x, y, intensity in spots:
            px = int(self.center[0] + x / self.pixel_scale)
            py = int(self.center[1] + y / self.pixel_scale)
            
            if 0 <= px < self.image_size[0] and 0 <= py < self.image_size[1]:
                template[py, px] = intensity
        
        template = gaussian_filter(template, sigma=sigma)
        
        if template.max() > 0:
            template = template / template.max()
        
        return template
    
    def cross_correlate(self, template: np.ndarray, 
                        target: np.ndarray) -> Tuple[float, Tuple[int, int]]:
        if template.shape != target.shape:
            from scipy.ndimage import zoom
            scale = np.array(target.shape) / np.array(template.shape)
            template = zoom(template, scale, order=1)
        
        correlation = signal.correlate2d(target - target.mean(), 
                                        template - template.mean(),
                                        mode='same', boundary='wrap')
        
        max_corr = correlation.max()
        max_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        norm_factor = (np.linalg.norm(target - target.mean()) * 
                      np.linalg.norm(template - template.mean()))
        
        if norm_factor > 0:
            max_corr = max_corr / norm_factor
        
        return max_corr, max_pos
    
    def normalized_correlation(self, template: np.ndarray, 
                              target: np.ndarray) -> float:
        t_mean = template.mean()
        s_mean = target.mean()
        
        t_std = template.std()
        s_std = target.std()
        
        if t_std == 0 or s_std == 0:
            return 0.0
        
        corr = np.sum((template - t_mean) * (target - s_mean))
        corr /= (t_std * s_std * template.size)
        
        return corr


class EnhancedTEMCalibrator:
    def __init__(self, camera_length: float, wavelength: float = 0.02508,
                 pixel_size: float = 1.0,
                 crystal_structure: CrystalStructure = CrystalStructure.FCC):
        self.camera_length = camera_length
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.crystal_structure = crystal_structure
        self.spots: List[DiffractionSpot] = []
        self.reciprocal_lattice = None
        self.lattice_params: Optional[LatticeParameters] = None
        self.orientation: Optional[CrystalOrientation] = None
        self.center: Tuple[float, float] = (0, 0)
        self.template_matcher = TemplateMatcher()
        self.validation_scores = {}
        
    @classmethod
    def from_accelerating_voltage(cls, camera_length: float, 
                                  voltage: float = 200.0,
                                  crystal_structure: CrystalStructure = CrystalStructure.FCC):
        wavelength = cls._calculate_electron_wavelength(voltage)
        return cls(camera_length, wavelength, crystal_structure=crystal_structure)
    
    @staticmethod
    def _calculate_electron_wavelength(voltage: float) -> float:
        h = 6.62607015e-34
        m0 = 9.1093837015e-31
        e = 1.602176634e-19
        c = 299792458
        
        voltage_v = voltage * 1000
        
        wavelength_m = h / np.sqrt(2 * m0 * e * voltage_v * (1 + e * voltage_v / (2 * m0 * c**2)))
        wavelength_angstrom = wavelength_m * 1e10
        
        return wavelength_angstrom
    
    def add_spots(self, spots: List[Tuple[float, float]], 
                  intensities: Optional[List[float]] = None):
        for i, (x, y) in enumerate(spots):
            intensity = intensities[i] if intensities and i < len(intensities) else 1.0
            self.spots.append(DiffractionSpot(x=x, y=y, intensity=intensity))
    
    def find_center_auto(self, method: str = 'centroid') -> Tuple[float, float]:
        if len(self.spots) < 3:
            return (0, 0)
        
        coords = np.array([[s.x, s.y] for s in self.spots])
        
        if method == 'centroid':
            intensities = np.array([s.intensity for s in self.spots])
            center_x = np.sum(coords[:, 0] * intensities) / np.sum(intensities)
            center_y = np.sum(coords[:, 1] * intensities) / np.sum(intensities)
        elif method == 'geometric':
            center_x = np.mean(coords[:, 0])
            center_y = np.mean(coords[:, 1])
        elif method == 'median':
            center_x = np.median(coords[:, 0])
            center_y = np.median(coords[:, 1])
        elif method == 'symmetry':
            center_x = np.median([(s1.x + s2.x) / 2 
                                for i, s1 in enumerate(self.spots)
                                for s2 in self.spots[i+1:]])
            center_y = np.median([(s1.y + s2.y) / 2
                                for i, s1 in enumerate(self.spots)
                                for s2 in self.spots[i+1:]])
        else:
            raise ValueError(f"未知的中心检测方法: {method}")
        
        self.center = (center_x, center_y)
        return self.center
    
    def calculate_radius(self, spot: DiffractionSpot) -> float:
        dx = spot.x - self.center[0]
        dy = spot.y - self.center[1]
        return np.sqrt(dx**2 + dy**2) * self.pixel_size
    
    def calculate_angle(self, spot: DiffractionSpot) -> float:
        dx = spot.x - self.center[0]
        dy = spot.y - self.center[1]
        return np.arctan2(dy, dx)
    
    def radius_to_dspacing(self, radius_mm: float) -> float:
        return (self.wavelength * self.camera_length) / radius_mm if radius_mm > 0 else float('inf')
    
    def detect_higher_order_laue(self, spot: DiffractionSpot, 
                               zero_order_spots: List[DiffractionSpot]) -> bool:
        r = self.calculate_radius(spot)
        if r == 0:
            return False
        
        theta = self.calculate_angle(spot)
        d = self.radius_to_dspacing(r)
        
        for zo_spot in zero_order_spots:
            zo_r = self.calculate_radius(zo_spot)
            if zo_r == 0:
                continue
            
            zo_d = self.radius_to_dspacing(zo_r)
            ratio = d / zo_d
            
            if abs(ratio - 1.0) < 0.05:
                return False
        
        for n in [2, 3]:
            expected_d = d * n
            for zo_spot in zero_order_spots:
                zo_r = self.calculate_radius(zo_spot)
                if zo_r == 0:
                    continue
                zo_d = self.radius_to_dspacing(zo_r)
                if abs(expected_d - zo_d) / zo_d < 0.1:
                    return True
        
        return False
    
    def detect_secondary_diffraction(self, spot: DiffractionSpot,
                                    indexed_spots: List[DiffractionSpot]) -> bool:
        if spot.h is None:
            return False
        
        r = self.calculate_radius(spot)
        theta = self.calculate_angle(spot)
        
        g_spot = np.array([
            np.cos(theta),
            np.sin(theta)
        ]) * (1.0 / self.radius_to_dspacing(r) if r > 0 else 0)
        
        for i, s1 in enumerate(indexed_spots):
            if s1.h is None or s1 is spot:
                continue
            
            r1 = self.calculate_radius(s1)
            theta1 = self.calculate_angle(s1)
            g1 = np.array([
                np.cos(theta1),
                np.sin(theta1)
            ]) * (1.0 / self.radius_to_dspacing(r1) if r1 > 0 else 0)
            
            for s2 in indexed_spots[i+1:]:
                if s2.h is None or s2 is spot:
                    continue
                
                r2 = self.calculate_radius(s2)
                theta2 = self.calculate_angle(s2)
                g2 = np.array([
                    np.cos(theta2),
                    np.sin(theta2)
                ]) * (1.0 / self.radius_to_dspacing(r2) if r2 > 0 else 0)
                
                g_sum = g1 + g2
                g_diff = g1 - g2
                
                if np.linalg.norm(g_spot - g_sum) < 0.05 * np.linalg.norm(g_spot):
                    return True
                if np.linalg.norm(g_spot - g_diff) < 0.05 * np.linalg.norm(g_spot):
                    return True
        
        return False
    
    def generate_theoretical_pattern(self, zone_axis: np.ndarray,
                                    max_index: int = 8) -> List[Tuple[int, int, int, float, float, float]]:
        if self.lattice_params is None:
            return []
        
        theoretical_spots = []
        
        for h in range(-max_index, max_index + 1):
            for k in range(-max_index, max_index + 1):
                for l in range(-max_index, max_index + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    
                    if zone_axis[0] * h + zone_axis[1] * k + zone_axis[2] * l != 0:
                        continue
                    
                    if StructureFactorCalculator.is_extinct(h, k, l, self.crystal_structure):
                        continue
                    
                    d = self.lattice_params.dspacing(h, k, l)
                    if d == float('inf') or d <= 0:
                        continue
                    
                    radius = (self.wavelength * self.camera_length) / d
                    
                    g1, g2 = self._get_reciprocal_basis_vectors(zone_axis)
                    
                    hkl = np.array([h, k, l])
                    coeff1 = np.dot(hkl, g1) / (np.dot(g1, g1) + 1e-10)
                    coeff2 = np.dot(hkl, g2) / (np.dot(g2, g2) + 1e-10)
                    
                    x = coeff1 * radius
                    y = coeff2 * radius
                    
                    f2 = StructureFactorCalculator.calculate_f2(h, k, l, self.crystal_structure)
                    intensity = f2 / (d**2) if d > 0 else 0
                    
                    theoretical_spots.append((h, k, l, x, y, intensity))
        
        return theoretical_spots
    
    def _get_reciprocal_basis_vectors(self, zone_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        za = np.array(zone_axis, dtype=float)
        za = za / (np.linalg.norm(za) + 1e-10)
        
        if abs(za[0]) < abs(za[1]) and abs(za[0]) < abs(za[2]):
            g1 = np.array([1, 0, 0])
        elif abs(za[1]) < abs(za[2]):
            g1 = np.array([0, 1, 0])
        else:
            g1 = np.array([0, 0, 1])
        
        g1 = g1 - np.dot(g1, za) * za
        if np.linalg.norm(g1) > 1e-10:
            g1 = g1 / np.linalg.norm(g1)
        
        g2 = np.cross(za, g1)
        if np.linalg.norm(g2) > 1e-10:
            g2 = g2 / np.linalg.norm(g2)
        
        return g1, g2
    
    def template_matching_calibration(self, candidate_zone_axes: List[np.ndarray],
                                     max_index: int = 6) -> Tuple[np.ndarray, float]:
        experimental_image = self._spots_to_image()
        
        best_zone = None
        best_correlation = -1
        
        for zone_axis in candidate_zone_axes:
            theoretical_spots = self.generate_theoretical_pattern(zone_axis, max_index)
            
            if len(theoretical_spots) < 3:
                continue
            
            template_spots = [(s[3], s[4], s[5]) for s in theoretical_spots]
            template = self.template_matcher.generate_template(template_spots)
            
            correlation, _ = self.template_matcher.cross_correlate(template, experimental_image)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_zone = zone_axis
        
        return best_zone, best_correlation
    
    def _spots_to_image(self, image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        image = np.zeros(image_size, dtype=np.float64)
        center = (image_size[0] // 2, image_size[1] // 2)
        
        max_r = max([self.calculate_radius(s) for s in self.spots]) if self.spots else 1
        scale = min(image_size) / (4 * max_r) if max_r > 0 else 1
        
        for spot in self.spots:
            x = int(center[0] + (spot.x - self.center[0]) * scale)
            y = int(center[1] + (spot.y - self.center[1]) * scale)
            
            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                image[y, x] = spot.intensity
        
        return gaussian_filter(image, sigma=2.0)
    
    def calibrate(self, lattice_type: str = 'cubic',
                  known_params: Optional[LatticeParameters] = None,
                  refine: bool = True,
                  use_template_matching: bool = True,
                  detect_spurious: bool = True) -> Dict:
        if len(self.spots) < 3:
            raise ValueError("需要至少3个衍射斑点进行标定")
        
        sorted_spots = sorted(self.spots, key=self.calculate_radius)
        
        if known_params is not None:
            self.lattice_params = known_params
        else:
            self.lattice_params = self._estimate_lattice_parameters(
                sorted_spots, lattice_type
            )
        
        self._index_spots_with_extinction(sorted_spots, lattice_type)
        
        indexed_spots = [s for s in self.spots if s.h is not None]
        
        if detect_spurious and len(indexed_spots) >= 3:
            for spot in self.spots:
                if spot.h is not None:
                    spot.is_higher_order_laue = self.detect_higher_order_laue(spot, indexed_spots[:10])
                    spot.is_secondary_diffraction = self.detect_secondary_diffraction(spot, indexed_spots)
                    
                    if spot.is_higher_order_laue or spot.is_secondary_diffraction:
                        spot.confidence *= 0.5
        
        if refine and len(indexed_spots) >= 3:
            self._refine_calibration(lattice_type)
        
        self._calculate_orientation()
        
        if use_template_matching and self.orientation is not None:
            candidate_zones = self._generate_candidate_zone_axes()
            if candidate_zones:
                best_zone, corr = self.template_matching_calibration(candidate_zones)
                if best_zone is not None:
                    self.orientation.zone_axis = best_zone
                    self.orientation.template_correlation = corr
        
        self.reciprocal_lattice = self.reconstruct_reciprocal_lattice()
        
        return {
            'lattice_params': self.lattice_params,
            'orientation': self.orientation,
            'spots': self.spots,
            'reciprocal_lattice': self.reciprocal_lattice
        }
    
    def _generate_candidate_zone_axes(self) -> List[np.ndarray]:
        candidates = []
        for u in range(-3, 4):
            for v in range(-3, 4):
                for w in range(-3, 4):
                    if u == 0 and v == 0 and w == 0:
                        continue
                    gcd = np.gcd.reduce([abs(u), abs(v), abs(w)])
                    if gcd > 0:
                        candidates.append(np.array([u//gcd, v//gcd, w//gcd]))
        
        unique_candidates = []
        seen = set()
        for c in candidates:
            key = tuple(sorted(tuple(abs(x) for x in c)))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        
        return unique_candidates[:50]
    
    def _estimate_lattice_parameters(self, sorted_spots: List[DiffractionSpot],
                                     lattice_type: str) -> LatticeParameters:
        d_values = []
        for spot in sorted_spots[1:15]:
            r = self.calculate_radius(spot)
            if r > 0:
                d_values.append(self.radius_to_dspacing(r))
        
        if not d_values:
            return LatticeParameters(a=0.4, b=0.4, c=0.4, alpha=90, beta=90, gamma=90)
        
        if lattice_type == 'cubic':
            g_values = [1.0/d for d in d_values]
            g_squared = [g**2 for g in g_values]
            
            allowed_sums = []
            for h in range(1, 10):
                for k in range(0, h+1):
                    for l in range(0, k+1):
                        s = h**2 + k**2 + l**2
                        
                        if not StructureFactorCalculator.is_extinct(h, k, l, self.crystal_structure):
                            if s not in allowed_sums:
                                allowed_sums.append(s)
            allowed_sums.sort()
            
            best_a = 0
            best_score = float('inf')
            
            for s_idx, s in enumerate(allowed_sums[:12]):
                if s_idx >= len(d_values):
                    break
                a_estimate = np.sqrt(s) * d_values[s_idx]
                
                score = 0
                count = 0
                for d in d_values[:10]:
                    g = 1.0/d
                    for allowed_s in allowed_sums[:15]:
                        expected_g = np.sqrt(allowed_s) / a_estimate
                        if abs(g - expected_g) / expected_g < 0.1:
                            score += abs(g - expected_g)**2
                            count += 1
                            break
                
                if count > 3 and score / count < best_score:
                    best_score = score / count
                    best_a = a_estimate
            
            if best_a == 0:
                best_a = d_values[0] * np.sqrt(2)
            
            return LatticeParameters(a=best_a, b=best_a, c=best_a,
                                     alpha=90, beta=90, gamma=90)
        
        elif lattice_type == 'hexagonal':
            return LatticeParameters(a=0.321, b=0.321, c=0.521,
                                     alpha=90, beta=90, gamma=120)
        
        else:
            return LatticeParameters(a=0.4, b=0.4, c=0.4,
                                     alpha=90, beta=90, gamma=90)
    
    def _index_spots_with_extinction(self, sorted_spots: List[DiffractionSpot], 
                                      lattice_type: str):
        if self.lattice_params is None:
            return
        
        g_tensor = self.lattice_params.reciprocal_metric_tensor()
        
        for spot in sorted_spots:
            r = self.calculate_radius(spot)
            if r == 0:
                spot.h, spot.k, spot.l = 0, 0, 0
                spot.confidence = 1.0
                continue
            
            d_measured = self.radius_to_dspacing(r)
            g_measured = 1.0 / d_measured
            
            candidates = []
            
            max_index = 8
            for h in range(-max_index, max_index + 1):
                for k in range(-max_index, max_index + 1):
                    for l in range(-max_index, max_index + 1):
                        if h == 0 and k == 0 and l == 0:
                            continue
                        
                        if StructureFactorCalculator.is_extinct(h, k, l, self.crystal_structure):
                            continue
                        
                        hkl = np.array([h, k, l])
                        g_squared = hkl @ g_tensor @ hkl.T
                        
                        if g_squared <= 0:
                            continue
                        
                        g_calculated = np.sqrt(g_squared)
                        error = abs(g_measured - g_calculated) / g_measured
                        
                        if error < 0.15:
                            f2 = StructureFactorCalculator.calculate_f2(h, k, l, self.crystal_structure)
                            score = (1 - error) * np.log(f2 + 1)
                            candidates.append((h, k, l, error, score))
            
            if candidates:
                candidates.sort(key=lambda x: -x[4])
                best = candidates[0]
                spot.h, spot.k, spot.l = best[0], best[1], best[2]
                spot.error = best[3]
                spot.confidence = max(0, 1 - best[3])
                spot.match_score = best[4]
            else:
                spot.confidence = 0.0
    
    def _refine_calibration(self, lattice_type: str, max_iterations: int = 100):
        indexed_spots = [s for s in self.spots 
                        if s.h is not None and not s.is_higher_order_laue 
                        and not s.is_secondary_diffraction]
        
        if len(indexed_spots) < 4:
            return
        
        if self.lattice_params is None:
            return
        
        def objective(params):
            if lattice_type == 'cubic':
                a = params[0]
                lp = LatticeParameters(a=a, b=a, c=a, alpha=90, beta=90, gamma=90)
            elif lattice_type == 'hexagonal':
                a, c = params
                lp = LatticeParameters(a=a, b=a, c=c, alpha=90, beta=90, gamma=120)
            else:
                a, b, c = params[:3]
                lp = LatticeParameters(a=a, b=b, c=c, alpha=90, beta=90, gamma=90)
            
            g_tensor = lp.reciprocal_metric_tensor()
            
            error = 0
            weights = 0
            
            for spot in indexed_spots:
                hkl = np.array([spot.h, spot.k, spot.l])
                
                if StructureFactorCalculator.is_extinct(spot.h, spot.k, spot.l, self.crystal_structure):
                    continue
                
                g_calc_sq = hkl @ g_tensor @ hkl.T
                if g_calc_sq <= 0:
                    continue
                
                r = self.calculate_radius(spot)
                d_meas = self.radius_to_dspacing(r)
                g_meas_sq = (1.0 / d_meas) ** 2 if d_meas != 0 else 0
                
                weight = spot.confidence * spot.intensity
                error += weight * (g_calc_sq - g_meas_sq) ** 2
                weights += weight
            
            return error / (weights + 1e-10)
        
        try:
            if lattice_type == 'cubic':
                x0 = [self.lattice_params.a]
                bounds = [(0.1, 10.0)]
            elif lattice_type == 'hexagonal':
                x0 = [self.lattice_params.a, self.lattice_params.c]
                bounds = [(0.1, 10.0), (0.1, 20.0)]
            else:
                x0 = [self.lattice_params.a, self.lattice_params.b, self.lattice_params.c]
                bounds = [(0.1, 10.0)] * 3
            
            result = optimize.minimize(objective, x0, method='L-BFGS-B', 
                                       bounds=bounds,
                                       options={'maxiter': max_iterations})
            
            if result.success:
                if lattice_type == 'cubic':
                    a = result.x[0]
                    self.lattice_params = LatticeParameters(a=a, b=a, c=a, 
                                                           alpha=90, beta=90, gamma=90)
                elif lattice_type == 'hexagonal':
                    a, c = result.x
                    self.lattice_params = LatticeParameters(a=a, b=a, c=c,
                                                           alpha=90, beta=90, gamma=120)
                
                rms = np.sqrt(result.fun)
                if self.orientation:
                    self.orientation.refinement_rms = rms
        
        except Exception as e:
            print(f"精修警告: {e}")
    
    def _calculate_orientation(self):
        indexed_spots = [s for s in self.spots 
                        if s.h is not None and s.confidence > 0.5
                        and not s.is_higher_order_laue]
        
        if len(indexed_spots) < 3:
            indexed_spots = [s for s in self.spots if s.h is not None]
        
        if len(indexed_spots) < 2:
            self.orientation = CrystalOrientation(
                zone_axis=np.array([0, 0, 1]),
                rotation_angle=0.0,
                confidence=0.0
            )
            return
        
        def gcd3(a, b, c):
            from math import gcd
            return gcd(gcd(abs(int(a)), abs(int(b))), abs(int(c)))
        
        best_zone = None
        best_votes = 0
        
        for i in range(min(5, len(indexed_spots))):
            for j in range(i+1, min(6, len(indexed_spots))):
                g1 = np.array([indexed_spots[i].h, indexed_spots[i].k, indexed_spots[i].l])
                g2 = np.array([indexed_spots[j].h, indexed_spots[j].k, indexed_spots[j].l])
                
                zone_axis = np.cross(g1, g2)
                
                if np.linalg.norm(zone_axis) > 0:
                    div = gcd3(zone_axis[0], zone_axis[1], zone_axis[2])
                    if div != 0:
                        zone_axis = zone_axis // div
                    
                    if zone_axis[0] < 0 or (zone_axis[0] == 0 and zone_axis[1] < 0) or \
                       (zone_axis[0] == 0 and zone_axis[1] == 0 and zone_axis[2] < 0):
                        zone_axis = -zone_axis
                    
                    votes = 0
                    for spot in indexed_spots:
                        if np.dot(zone_axis, [spot.h, spot.k, spot.l]) == 0:
                            votes += spot.confidence
                    
                    if votes > best_votes:
                        best_votes = votes
                        best_zone = zone_axis
        
        if best_zone is None:
            g1 = np.array([indexed_spots[0].h, indexed_spots[0].k, indexed_spots[0].l])
            g2 = np.array([indexed_spots[1].h, indexed_spots[1].k, indexed_spots[1].l])
            best_zone = np.cross(g1, g2)
        
        confidence = best_votes / len(indexed_spots) if indexed_spots else 0
        
        self.orientation = CrystalOrientation(
            zone_axis=best_zone.astype(int),
            rotation_angle=self._calculate_rotation_angle(indexed_spots, best_zone),
            confidence=confidence
        )
    
    def _calculate_rotation_angle(self, indexed_spots: List[DiffractionSpot], 
                                  zone_axis: np.ndarray) -> float:
        if len(indexed_spots) < 2:
            return 0.0
        
        angles = []
        for spot in indexed_spots[:5]:
            if spot.h is None:
                continue
            
            theta_exp = self.calculate_angle(spot)
            
            g1, g2 = self._get_reciprocal_basis_vectors(zone_axis)
            hkl = np.array([spot.h, spot.k, spot.l])
            
            proj1 = np.dot(hkl, g1)
            proj2 = np.dot(hkl, g2)
            theta_theory = np.arctan2(proj2, proj1)
            
            angles.append((theta_exp - theta_theory))
        
        if angles:
            return np.degrees(np.median(angles))
        return 0.0
    
    def reconstruct_reciprocal_lattice(self) -> np.ndarray:
        reciprocal_points = []
        
        for spot in self.spots:
            r = self.calculate_radius(spot)
            if r == 0:
                reciprocal_points.append([0, 0, 0])
                continue
            
            d = self.radius_to_dspacing(r)
            g = 1.0 / d
            
            theta = self.calculate_angle(spot)
            
            gx = g * np.cos(theta)
            gy = g * np.sin(theta)
            
            reciprocal_points.append([gx, gy, 0])
        
        return np.array(reciprocal_points)
    
    def plot_calibration(self, save_path: Optional[str] = None,
                        show_indices: bool = True,
                        highlight_spurious: bool = True):
        if not self.spots:
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_diffraction_pattern(ax1, show_indices, highlight_spurious)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_reciprocal_space(ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_dspacing_curve(ax3)
        
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_error_distribution(ax4)
        
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_confidence_distribution(ax5)
        
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_template_matching(ax6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_diffraction_pattern(self, ax, show_indices: bool, highlight_spurious: bool):
        x_coords = [s.x for s in self.spots]
        y_coords = [s.y for s in self.spots]
        colors = []
        
        for spot in self.spots:
            if highlight_spurious and spot.is_higher_order_laue:
                colors.append('orange')
            elif highlight_spurious and spot.is_secondary_diffraction:
                colors.append('purple')
            elif spot.confidence > 0.8:
                colors.append('green')
            elif spot.confidence > 0.5:
                colors.append('blue')
            else:
                colors.append('red')
        
        scatter = ax.scatter(x_coords, y_coords, c=colors, s=60, alpha=0.8, edgecolors='black')
        ax.scatter([self.center[0]], [self.center[1]], 
                   c='red', s=200, marker='+', linewidth=2, label='中心')
        ax.set_xlabel('X (像素)')
        ax.set_ylabel('Y (像素)')
        ax.set_title('衍射花样 (颜色: 置信度/类型)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='高置信度 (>80%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='中置信度 (50-80%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='低置信度 (<50%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='高阶劳厄带'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='二次衍射'),
        ]
        ax.legend(handles=legend_elements, fontsize=8)
        
        if show_indices:
            for spot in self.spots:
                if spot.h is not None and spot.confidence > 0.3:
                    label = f'({spot.h},{spot.k},{spot.l})'
                    ax.annotate(label, (spot.x, spot.y), 
                                textcoords="offset points", xytext=(5, 5), 
                                ha='center', fontsize=7, color='darkred')
    
    def _plot_reciprocal_space(self, ax):
        if self.reciprocal_lattice is None:
            return
        
        ax.scatter(self.reciprocal_lattice[:, 0], 
                   self.reciprocal_lattice[:, 1], 
                   c='teal', s=60, alpha=0.7, edgecolors='black')
        ax.scatter([0], [0], c='red', s=200, marker='+')
        ax.set_xlabel('g_x (Å⁻¹)')
        ax.set_ylabel('g_y (Å⁻¹)')
        ax.set_title('倒易空间重构')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    def _plot_dspacing_curve(self, ax):
        radii = []
        d_spacings = []
        confidences = []
        
        for spot in self.spots:
            r = self.calculate_radius(spot)
            if r > 0:
                radii.append(r)
                d_spacings.append(self.radius_to_dspacing(r))
                confidences.append(spot.confidence)
        
        if radii:
            scatter = ax.scatter(radii, d_spacings, c=confidences, 
                               cmap='viridis', s=50, alpha=0.8)
            ax.set_xlabel('斑点半径 (mm)')
            ax.set_ylabel('d-间距 (Å)')
            ax.set_title('Rd = λL 关系 (颜色: 置信度)')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='置信度')
            
            sort_idx = np.argsort(radii)
            ax.plot(np.array(radii)[sort_idx], np.array(d_spacings)[sort_idx], 
                   'k--', alpha=0.3)
    
    def _plot_error_distribution(self, ax):
        indexed_spots = [s for s in self.spots if s.h is not None and s.error > 0]
        if indexed_spots:
            errors = [s.error * 100 for s in indexed_spots]
            ax.hist(errors, bins=15, alpha=0.7, edgecolor='black', color='steelblue')
            ax.axvline(np.mean(errors), color='red', linestyle='--', 
                      label=f'平均: {np.mean(errors):.2f}%')
            ax.set_xlabel('标定误差 (%)')
            ax.set_ylabel('斑点数量')
            ax.set_title('标定误差分布')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_confidence_distribution(self, ax):
        confidences = [s.confidence for s in self.spots]
        ax.hist(confidences, bins=15, alpha=0.7, edgecolor='black', color='seagreen')
        ax.axvline(np.mean(confidences), color='red', linestyle='--',
                  label=f'平均: {np.mean(confidences):.2f}')
        ax.set_xlabel('置信度')
        ax.set_ylabel('斑点数量')
        ax.set_title('斑点置信度分布')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        hol_count = sum(1 for s in self.spots if s.is_higher_order_laue)
        sd_count = sum(1 for s in self.spots if s.is_secondary_diffraction)
        ax.text(0.05, 0.95, f'高阶劳厄带: {hol_count}\n二次衍射: {sd_count}',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_template_matching(self, ax):
        experimental = self._spots_to_image()
        ax.imshow(experimental, cmap='hot', origin='lower')
        ax.set_title('实验衍射图 (用于模板匹配)')
        ax.axis('off')
        
        if self.orientation and self.orientation.template_correlation > 0:
            ax.text(0.05, 0.95, f'模板相关系数: {self.orientation.template_correlation:.3f}',
                   transform=ax.transAxes, va='top', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def print_report(self):
        print("=" * 80)
        print("增强型 TEM 电子衍射花样标定报告")
        print("=" * 80)
        print(f"\n仪器参数:")
        print(f"  相机长度: {self.camera_length} mm")
        print(f"  电子波长: {self.wavelength:.5f} Å")
        print(f"  像素大小: {self.pixel_size} mm/像素")
        print(f"  晶体结构: {self.crystal_structure.value.upper()}")
        print(f"  标定中心: ({self.center[0]:.2f}, {self.center[1]:.2f}) 像素")
        
        print(f"\n衍射斑点统计:")
        print(f"  总斑点数: {len(self.spots)}")
        indexed_count = sum(1 for s in self.spots if s.h is not None)
        hol_count = sum(1 for s in self.spots if s.is_higher_order_laue)
        sd_count = sum(1 for s in self.spots if s.is_secondary_diffraction)
        high_conf_count = sum(1 for s in self.spots if s.confidence > 0.8)
        print(f"  已标定: {indexed_count} 个 ({indexed_count/len(self.spots)*100:.1f}%)")
        print(f"  高置信度: {high_conf_count} 个")
        print(f"  高阶劳厄带: {hol_count} 个 (已标记)")
        print(f"  二次衍射: {sd_count} 个 (已标记)")
        
        if self.lattice_params:
            print("\n" + "-" * 80)
            print("晶格参数 (消光规则验证后):")
            print("-" * 80)
            print(f"  a = {self.lattice_params.a:.5f} Å")
            print(f"  b = {self.lattice_params.b:.5f} Å")
            print(f"  c = {self.lattice_params.c:.5f} Å")
            print(f"  α = {self.lattice_params.alpha:.1f}°")
            print(f"  β = {self.lattice_params.beta:.1f}°")
            print(f"  γ = {self.lattice_params.gamma:.1f}°")
        
        if self.orientation:
            print("\n" + "-" * 80)
            print("晶体取向 (投票法 + 模板验证):")
            print("-" * 80)
            zone = self.orientation.zone_axis.astype(int)
            print(f"  晶带轴: [{zone[0]} {zone[1]} {zone[2]}]")
            print(f"  旋转角: {self.orientation.rotation_angle:.2f}°")
            print(f"  投票置信度: {self.orientation.confidence*100:.1f}%")
            if self.orientation.refinement_rms > 0:
                print(f"  精修RMS: {self.orientation.refinement_rms:.6f}")
            if self.orientation.template_correlation > 0:
                print(f"  模板相关系数: {self.orientation.template_correlation:.3f}")
        
        print("\n" + "-" * 80)
        print("标定斑点详情 (按半径排序):")
        print("-" * 80)
        print(f"{'#':>3} {'X':>8} {'Y':>8} {'R(mm)':>7} {'d(Å)':>8} "
              f"{'hkl':>10} {'误差%':>7} {'置信度':>7} {'标记':>12}")
        print("-" * 80)
        
        for i, spot in enumerate(sorted(self.spots, key=self.calculate_radius)):
            r = self.calculate_radius(spot)
            d = self.radius_to_dspacing(r) if r > 0 else 0
            hkl = f"({spot.h},{spot.k},{spot.l})" if spot.h is not None else "---"
            err = f"{spot.error*100:.2f}" if spot.h is not None else "---"
            conf = f"{spot.confidence:.2f}" if spot.h is not None else "---"
            
            flags = []
            if spot.is_higher_order_laue:
                flags.append("HOLZ")
            if spot.is_secondary_diffraction:
                flags.append("2nd")
            flag_str = ",".join(flags) if flags else ""
            
            print(f"{i+1:>3} {spot.x:>8.1f} {spot.y:>8.1f} {r:>7.3f} {d:>8.4f} "
                  f"{hkl:>10} {err:>7} {conf:>7} {flag_str:>12}")
        
        print("\n" + "=" * 80)


def generate_test_pattern(calibrator: EnhancedTEMCalibrator, 
                         a_true: float, zone_axis: List[int],
                         add_noise: bool = True,
                         add_spurious: bool = True) -> List[Tuple[float, float, float]]:
    spots = [(0, 0, 100.0)]
    
    for h in range(-5, 6):
        for k in range(-5, 6):
            for l in range(-5, 6):
                if h == 0 and k == 0 and l == 0:
                    continue
                
                if h * zone_axis[0] + k * zone_axis[1] + l * zone_axis[2] != 0:
                    continue
                
                if StructureFactorCalculator.is_extinct(h, k, l, calibrator.crystal_structure):
                    continue
                
                s = h**2 + k**2 + l**2
                d = a_true / np.sqrt(s) if s > 0 else float('inf')
                radius = (calibrator.wavelength * calibrator.camera_length) / d
                
                g1, g2 = calibrator._get_reciprocal_basis_vectors(zone_axis)
                hkl = np.array([h, k, l])
                coeff1 = np.dot(hkl, g1) / (np.dot(g1, g1) + 1e-10)
                coeff2 = np.dot(hkl, g2) / (np.dot(g2, g2) + 1e-10)
                
                x = coeff1 * radius
                y = coeff2 * radius
                
                if add_noise:
                    x += np.random.normal(0, 0.3)
                    y += np.random.normal(0, 0.3)
                
                f2 = StructureFactorCalculator.calculate_f2(h, k, l, calibrator.crystal_structure)
                intensity = f2 / (1 + s * 0.05)
                
                if not any(np.isclose(x, s[0], atol=3) and np.isclose(y, s[1], atol=3) for s in spots):
                    if abs(x) < 200 and abs(y) < 200:
                        spots.append((x, y, intensity))
    
    if add_spurious:
        for _ in range(3):
            base_spot = spots[np.random.randint(1, len(spots))]
            x = base_spot[0] * 1.414 + np.random.normal(0, 2)
            y = base_spot[1] * 1.414 + np.random.normal(0, 2)
            spots.append((x, y, 20.0))
    
    return spots


def main():
    print("增强型 TEM 衍射标定程序 - 模板匹配 + 伪影检测演示")
    print("=" * 80)
    
    camera_length = 200.0
    voltage = 200.0
    a_true = 4.079
    zone_axis_true = [0, 0, 1]
    
    calibrator = EnhancedTEMCalibrator.from_accelerating_voltage(
        camera_length=camera_length,
        voltage=voltage,
        crystal_structure=CrystalStructure.FCC
    )
    
    np.random.seed(42)
    test_spots = generate_test_pattern(calibrator, a_true, zone_axis_true)
    
    print(f"\n生成测试斑点: {len(test_spots)} 个")
    print(f"真实晶格参数: a = {a_true} Å")
    print(f"真实晶带轴: {zone_axis_true}")
    
    calibrator.add_spots([(s[0], s[1]) for s in test_spots],
                         intensities=[s[2] for s in test_spots])
    
    calibrator.find_center_auto(method='centroid')
    
    print("\n开始标定 (使用模板匹配 + 伪影检测)...")
    results = calibrator.calibrate(
        lattice_type='cubic',
        refine=True,
        use_template_matching=True,
        detect_spurious=True
    )
    
    calibrator.print_report()
    
    print("\n生成可视化图像...")
    calibrator.plot_calibration(show_indices=True, highlight_spurious=True)


if __name__ == "__main__":
    main()
