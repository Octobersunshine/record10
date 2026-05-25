import numpy as np
from scipy import optimize, signal
from scipy.spatial import distance_matrix
from typing import List, Tuple, Optional, Dict, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DiffractionSpot:
    x: float
    y: float
    intensity: float = 1.0
    h: Optional[int] = None
    k: Optional[int] = None
    l: Optional[int] = None
    error: float = 0.0


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


@dataclass
class CrystalOrientation:
    zone_axis: np.ndarray
    rotation_angle: float
    confidence: float
    refinement_rms: float = 0.0


class AdvancedTEMCalibrator:
    def __init__(self, camera_length: float, wavelength: float = 0.02508,
                 pixel_size: float = 1.0):
        self.camera_length = camera_length
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.spots: List[DiffractionSpot] = []
        self.reciprocal_lattice = None
        self.lattice_params: Optional[LatticeParameters] = None
        self.orientation: Optional[CrystalOrientation] = None
        self.center: Tuple[float, float] = (0, 0)
        
    @classmethod
    def from_accelerating_voltage(cls, camera_length: float, voltage: float = 200.0):
        wavelength = cls._calculate_electron_wavelength(voltage)
        return cls(camera_length, wavelength)
    
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
        return (self.wavelength * self.camera_length) / radius_mm
    
    def dspacing_to_reciprocal(self, d: float) -> float:
        return 1.0 / d if d != 0 else 0.0
    
    def calibrate(self, lattice_type: str = 'cubic',
                  known_params: Optional[LatticeParameters] = None,
                  refine: bool = True) -> Dict:
        if len(self.spots) < 3:
            raise ValueError("需要至少3个衍射斑点进行标定")
        
        sorted_spots = sorted(self.spots, key=self.calculate_radius)
        
        if known_params is not None:
            self.lattice_params = known_params
        else:
            self.lattice_params = self._estimate_lattice_parameters(
                sorted_spots, lattice_type
            )
        
        self._index_spots(sorted_spots, lattice_type)
        
        if refine:
            self._refine_calibration(lattice_type)
        
        self._calculate_orientation()
        
        self.reciprocal_lattice = self.reconstruct_reciprocal_lattice()
        
        return {
            'lattice_params': self.lattice_params,
            'orientation': self.orientation,
            'spots': self.spots,
            'reciprocal_lattice': self.reciprocal_lattice
        }
    
    def _estimate_lattice_parameters(self, sorted_spots: List[DiffractionSpot],
                                     lattice_type: str) -> LatticeParameters:
        d_values = []
        for spot in sorted_spots[1:11]:
            r = self.calculate_radius(spot)
            if r > 0:
                d_values.append(self.radius_to_dspacing(r))
        
        if not d_values:
            return LatticeParameters(a=0.4, b=0.4, c=0.4, alpha=90, beta=90, gamma=90)
        
        if lattice_type == 'cubic':
            g_values = [1.0/d for d in d_values]
            g_squared = [g**2 for g in g_values]
            
            ratios = []
            for i in range(1, min(10, len(g_squared))):
                ratios.append(g_squared[i] / g_squared[0])
            
            possible_sums = []
            for h in range(1, 8):
                for k in range(0, h+1):
                    for l in range(0, k+1):
                        s = h**2 + k**2 + l**2
                        if s not in possible_sums:
                            possible_sums.append(s)
            possible_sums.sort()
            
            best_a = 0
            best_score = float('inf')
            
            for s in possible_sums[:15]:
                a_estimate = np.sqrt(s) * d_values[0]
                
                score = 0
                for i, d in enumerate(d_values[:8]):
                    g = 1.0/d
                    expected_g_squared = (i+1) / a_estimate**2
                    score += abs(g**2 - expected_g_squared)
                
                if score < best_score:
                    best_score = score
                    best_a = a_estimate
            
            return LatticeParameters(a=best_a, b=best_a, c=best_a,
                                     alpha=90, beta=90, gamma=90)
        
        elif lattice_type == 'tetragonal':
            return LatticeParameters(a=0.38, b=0.38, c=0.5,
                                     alpha=90, beta=90, gamma=90)
        
        elif lattice_type == 'hexagonal':
            return LatticeParameters(a=0.32, b=0.32, c=0.52,
                                     alpha=90, beta=90, gamma=120)
        
        else:
            return LatticeParameters(a=0.4, b=0.4, c=0.4,
                                     alpha=90, beta=90, gamma=90)
    
    def _index_spots(self, sorted_spots: List[DiffractionSpot], lattice_type: str):
        if self.lattice_params is None:
            return
        
        g_tensor = self.lattice_params.reciprocal_metric_tensor()
        
        for spot in sorted_spots:
            r = self.calculate_radius(spot)
            if r == 0:
                continue
            
            d_measured = self.radius_to_dspacing(r)
            g_measured = 1.0 / d_measured
            
            best_match = None
            min_error = float('inf')
            
            max_index = 6
            for h in range(-max_index, max_index + 1):
                for k in range(-max_index, max_index + 1):
                    for l in range(-max_index, max_index + 1):
                        if h == 0 and k == 0 and l == 0:
                            continue
                        
                        hkl = np.array([h, k, l])
                        g_squared = hkl @ g_tensor @ hkl.T
                        
                        if g_squared <= 0:
                            continue
                        
                        g_calculated = np.sqrt(g_squared)
                        error = abs(g_measured - g_calculated) / g_measured
                        
                        if error < min_error:
                            min_error = error
                            best_match = (h, k, l)
            
            if best_match and min_error < 0.15:
                spot.h, spot.k, spot.l = best_match
                spot.error = min_error
    
    def _refine_calibration(self, lattice_type: str, max_iterations: int = 50):
        indexed_spots = [s for s in self.spots if s.h is not None]
        if len(indexed_spots) < 3:
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
            for spot in indexed_spots:
                hkl = np.array([spot.h, spot.k, spot.l])
                g_calc_sq = hkl @ g_tensor @ hkl.T
                if g_calc_sq <= 0:
                    continue
                
                r = self.calculate_radius(spot)
                d_meas = self.radius_to_dspacing(r)
                g_meas_sq = (1.0 / d_meas) ** 2 if d_meas != 0 else 0
                
                error += (g_calc_sq - g_meas_sq) ** 2
            
            return error
        
        try:
            if lattice_type == 'cubic':
                x0 = [self.lattice_params.a]
            elif lattice_type == 'hexagonal':
                x0 = [self.lattice_params.a, self.lattice_params.c]
            else:
                x0 = [self.lattice_params.a, self.lattice_params.b, self.lattice_params.c]
            
            bounds = [(0.1, 10.0)] * len(x0)
            
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
                
                rms = np.sqrt(result.fun / len(indexed_spots))
                if self.orientation:
                    self.orientation.refinement_rms = rms
        
        except Exception as e:
            print(f"精修警告: {e}")
    
    def _calculate_orientation(self):
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
        
        g1 = np.array([indexed_spots[1].h, indexed_spots[1].k, indexed_spots[1].l])
        g2 = np.array([indexed_spots[2].h, indexed_spots[2].k, indexed_spots[2].l])
        
        zone_axis = np.cross(g1, g2)
        
        if np.linalg.norm(zone_axis) > 0:
            div = gcd3(zone_axis[0], zone_axis[1], zone_axis[2])
            if div != 0:
                zone_axis = zone_axis // div
        
        p1 = np.array([indexed_spots[1].x - self.center[0], 
                       indexed_spots[1].y - self.center[1]])
        p2 = np.array([indexed_spots[2].x - self.center[0], 
                       indexed_spots[2].y - self.center[1]])
        
        if np.linalg.norm(p1) > 0 and np.linalg.norm(p2) > 0:
            angle_p = np.arctan2(p2[1], p2[0]) - np.arctan2(p1[1], p1[0])
            angle_g = np.arctan2(g2[1], g2[0]) - np.arctan2(g1[1], g1[0])
            rotation = np.degrees(angle_p - angle_g)
        else:
            rotation = 0.0
        
        confidence = len(indexed_spots) / len(self.spots)
        
        self.orientation = CrystalOrientation(
            zone_axis=zone_axis,
            rotation_angle=rotation,
            confidence=confidence
        )
    
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
                        show_indices: bool = True):
        if not self.spots:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        ax1 = axes[0, 0]
        x_coords = [s.x for s in self.spots]
        y_coords = [s.y for s in self.spots]
        intensities = [s.intensity for s in self.spots]
        
        scatter = ax1.scatter(x_coords, y_coords, c=intensities, 
                            cmap='viridis', s=50, alpha=0.7)
        ax1.scatter([self.center[0]], [self.center[1]], 
                   c='red', s=200, marker='+', linewidth=2, label='中心')
        ax1.set_xlabel('X (像素)')
        ax1.set_ylabel('Y (像素)')
        ax1.set_title('原始衍射花样')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='强度')
        
        if show_indices:
            for spot in self.spots:
                if spot.h is not None:
                    label = f'({spot.h},{spot.k},{spot.l})'
                    ax1.annotate(label, (spot.x, spot.y), 
                                textcoords="offset points", xytext=(5, 5), 
                                ha='center', fontsize=8, color='red')
        
        ax2 = axes[0, 1]
        if self.reciprocal_lattice is not None:
            ax2.scatter(self.reciprocal_lattice[:, 0], 
                       self.reciprocal_lattice[:, 1], 
                       c='green', s=50, alpha=0.7)
            ax2.scatter([0], [0], c='red', s=200, marker='+')
            ax2.set_xlabel('g_x (Å⁻¹)')
            ax2.set_ylabel('g_y (Å⁻¹)')
            ax2.set_title('倒易空间重构')
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
        
        ax3 = axes[1, 0]
        radii = []
        d_spacings = []
        for spot in self.spots:
            r = self.calculate_radius(spot)
            if r > 0:
                radii.append(r)
                d_spacings.append(self.radius_to_dspacing(r))
        
        if radii:
            ax3.plot(sorted(radii), sorted(d_spacings), 'bo-', alpha=0.7)
            ax3.set_xlabel('斑点半径 (mm)')
            ax3.set_ylabel('d-间距 (Å)')
            ax3.set_title('Rd = λL 关系曲线')
            ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        indexed_spots = [s for s in self.spots if s.h is not None]
        if indexed_spots:
            errors = [s.error * 100 for s in indexed_spots]
            ax4.hist(errors, bins=15, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('标定误差 (%)')
            ax4.set_ylabel('斑点数量')
            ax4.set_title(f'标定误差分布 (平均: {np.mean(errors):.2f}%)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_report(self):
        print("=" * 70)
        print("高级 TEM 电子衍射花样标定报告")
        print("=" * 70)
        print(f"\n仪器参数:")
        print(f"  相机长度: {self.camera_length} mm")
        print(f"  电子波长: {self.wavelength:.5f} Å")
        print(f"  像素大小: {self.pixel_size} mm/像素")
        print(f"  标定中心: ({self.center[0]:.2f}, {self.center[1]:.2f}) 像素")
        print(f"\n衍射斑点: {len(self.spots)} 个")
        indexed_count = sum(1 for s in self.spots if s.h is not None)
        print(f"成功标定: {indexed_count} 个 ({indexed_count/len(self.spots)*100:.1f}%)")
        
        if self.lattice_params:
            print("\n" + "-" * 70)
            print("晶格参数:")
            print("-" * 70)
            print(f"  a = {self.lattice_params.a:.4f} Å")
            print(f"  b = {self.lattice_params.b:.4f} Å")
            print(f"  c = {self.lattice_params.c:.4f} Å")
            print(f"  α = {self.lattice_params.alpha:.1f}°")
            print(f"  β = {self.lattice_params.beta:.1f}°")
            print(f"  γ = {self.lattice_params.gamma:.1f}°")
        
        if self.orientation:
            print("\n" + "-" * 70)
            print("晶体取向:")
            print("-" * 70)
            zone = self.orientation.zone_axis.astype(int)
            print(f"  晶带轴: [{zone[0]} {zone[1]} {zone[2]}]")
            print(f"  旋转角: {self.orientation.rotation_angle:.2f}°")
            print(f"  置信度: {self.orientation.confidence*100:.1f}%")
            if self.orientation.refinement_rms > 0:
                print(f"  精修RMS: {self.orientation.refinement_rms:.6f}")
        
        print("\n" + "-" * 70)
        print("标定斑点详情:")
        print("-" * 70)
        print(f"{'#':>3} {'X(像素)':>10} {'Y(像素)':>10} {'R(mm)':>8} "
              f"{'d(Å)':>8} {'hkl':>10} {'误差%':>8}")
        print("-" * 70)
        
        for i, spot in enumerate(sorted(self.spots, key=self.calculate_radius)):
            r = self.calculate_radius(spot)
            d = self.radius_to_dspacing(r) if r > 0 else 0
            hkl = f"({spot.h},{spot.k},{spot.l})" if spot.h is not None else "---"
            err = f"{spot.error*100:.2f}" if spot.h is not None else "---"
            print(f"{i+1:>3} {spot.x:>10.2f} {spot.y:>10.2f} {r:>8.3f} "
                  f"{d:>8.4f} {hkl:>10} {err:>8}")
        
        print("\n" + "=" * 70)


def main():
    print("高级 TEM 衍射标定程序演示")
    print("=" * 70)
    
    camera_length = 200.0
    voltage = 200.0
    
    calibrator = AdvancedTEMCalibrator.from_accelerating_voltage(
        camera_length=camera_length,
        voltage=voltage
    )
    
    np.random.seed(42)
    spots = []
    a_true = 4.079
    
    for h in range(-4, 5):
        for k in range(-4, 5):
            for l in range(-4, 5):
                if h == 0 and k == 0 and l == 0:
                    continue
                if h * 0 + k * 0 + l * 1 != 0:
                    continue
                
                s = h**2 + k**2 + l**2
                if s == 0:
                    continue
                
                d = a_true / np.sqrt(s)
                radius = (calibrator.wavelength * camera_length) / d
                
                angle = np.arctan2(k, h) if (h != 0 or k != 0) else 0
                
                noise = np.random.normal(0, 0.5)
                x = (radius + noise) * np.cos(angle)
                y = (radius + noise) * np.sin(angle)
                
                if not any(np.isclose(x, s[0], atol=2) and np.isclose(y, s[1], atol=2) for s in spots):
                    if abs(x) < 150 and abs(y) < 150:
                        intensity = 100.0 / (1 + s * 0.1)
                        spots.append((x, y, intensity))
    
    calibrator.add_spots([(s[0], s[1]) for s in spots], 
                         intensities=[s[2] for s in spots])
    
    calibrator.find_center_auto(method='centroid')
    
    print("\n开始标定...")
    results = calibrator.calibrate(lattice_type='cubic', refine=True)
    
    calibrator.print_report()
    
    print("\n生成可视化图像...")
    calibrator.plot_calibration(show_indices=True)


if __name__ == "__main__":
    main()
