import numpy as np
from scipy import optimize
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class DiffractionSpot:
    x: float
    y: float
    intensity: float = 1.0
    h: Optional[int] = None
    k: Optional[int] = None
    l: Optional[int] = None


@dataclass
class LatticeParameters:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


@dataclass
class CrystalOrientation:
    zone_axis: np.ndarray
    rotation_angle: float
    confidence: float


class TEMDiffractionCalibrator:
    def __init__(self, camera_length: float, wavelength: float = 0.02508):
        self.camera_length = camera_length
        self.wavelength = wavelength
        self.spots: List[DiffractionSpot] = []
        self.reciprocal_lattice = None
        self.lattice_params: Optional[LatticeParameters] = None
        self.orientation: Optional[CrystalOrientation] = None

    def add_spots(self, spots: List[Tuple[float, float]]):
        for x, y in spots:
            self.spots.append(DiffractionSpot(x=x, y=y))

    def calculate_radius(self, spot: DiffractionSpot, center: Tuple[float, float] = (0, 0)) -> float:
        dx = spot.x - center[0]
        dy = spot.y - center[1]
        return np.sqrt(dx**2 + dy**2)

    def radius_to_dspacing(self, radius: float) -> float:
        return (self.wavelength * self.camera_length) / radius

    def dspacing_to_reciprocal(self, d: float) -> float:
        return 1.0 / d if d != 0 else 0.0

    def calibrate_pattern(self, center: Tuple[float, float] = (0, 0), 
                          lattice_type: str = 'cubic', 
                          known_params: Optional[LatticeParameters] = None) -> Dict:
        if len(self.spots) < 3:
            raise ValueError("需要至少3个衍射斑点进行标定")

        radii = [self.calculate_radius(spot, center) for spot in self.spots]
        d_spacings = [self.radius_to_dspacing(r) for r in radii if r > 0]
        reciprocal_spacings = [self.dspacing_to_reciprocal(d) for d in d_spacings]

        sorted_indices = np.argsort(reciprocal_spacings)
        sorted_reciprocal = np.array(reciprocal_spacings)[sorted_indices]
        sorted_radii = np.array(radii)[sorted_indices]
        sorted_d = np.array(d_spacings)[sorted_indices]

        if known_params is not None:
            self.lattice_params = known_params
        else:
            self.lattice_params = self._estimate_lattice_parameters(
                sorted_reciprocal, lattice_type
            )

        self._index_spots(lattice_type)
        self._calculate_orientation()

        return {
            'radii': sorted_radii,
            'd_spacings': sorted_d,
            'reciprocal_spacings': sorted_reciprocal,
            'lattice_params': self.lattice_params,
            'orientation': self.orientation,
            'spots': self.spots
        }

    def _estimate_lattice_parameters(self, reciprocal_spacings: np.ndarray, 
                                     lattice_type: str) -> LatticeParameters:
        if lattice_type == 'cubic':
            ratios = reciprocal_spacings / reciprocal_spacings[0]
            ratios_squared = ratios ** 2
            
            allowed_hkl = []
            for h in range(6):
                for k in range(6):
                    for l in range(6):
                        if h == 0 and k == 0 and l == 0:
                            continue
                        s = h**2 + k**2 + l**2
                        allowed_hkl.append((s, h, k, l))
            
            allowed_hkl.sort(key=lambda x: x[0])
            unique_s = sorted(list(set([x[0] for x in allowed_hkl])))
            
            best_fit = float('inf')
            best_a = 0
            
            for i, s in enumerate(unique_s[:10]):
                if i >= len(reciprocal_spacings):
                    break
                a_estimate = np.sqrt(s) / reciprocal_spacings[i]
                total_error = 0
                for j, g in enumerate(reciprocal_spacings[:len(unique_s)//2]):
                    if j < len(unique_s):
                        expected = np.sqrt(unique_s[j]) / a_estimate
                        total_error += (g - expected) ** 2
                if total_error < best_fit:
                    best_fit = total_error
                    best_a = a_estimate
            
            return LatticeParameters(a=best_a, b=best_a, c=best_a,
                                     alpha=90, beta=90, gamma=90)
        
        elif lattice_type == 'hexagonal':
            return LatticeParameters(a=0.321, b=0.321, c=0.521,
                                     alpha=90, beta=90, gamma=120)
        
        else:
            return LatticeParameters(a=0.4, b=0.4, c=0.4,
                                     alpha=90, beta=90, gamma=90)

    def _index_spots(self, lattice_type: str = 'cubic'):
        if self.lattice_params is None:
            raise ValueError("请先确定晶格参数")

        for spot in self.spots:
            r = self.calculate_radius(spot)
            if r == 0:
                continue
            d = self.radius_to_dspacing(r)
            
            best_match = None
            min_error = float('inf')
            
            for h in range(-6, 7):
                for k in range(-6, 7):
                    for l in range(-6, 7):
                        if h == 0 and k == 0 and l == 0:
                            continue
                        
                        d_calc = self._calculate_dspacing(h, k, l, lattice_type)
                        error = abs(d - d_calc)
                        
                        if error < min_error:
                            min_error = error
                            best_match = (h, k, l)
            
            if best_match:
                spot.h, spot.k, spot.l = best_match

    def _calculate_dspacing(self, h: int, k: int, l: int, lattice_type: str) -> float:
        if self.lattice_params is None:
            return float('inf')
        
        a, b, c = self.lattice_params.a, self.lattice_params.b, self.lattice_params.c
        alpha = np.radians(self.lattice_params.alpha)
        beta = np.radians(self.lattice_params.beta)
        gamma = np.radians(self.lattice_params.gamma)

        if lattice_type == 'cubic':
            return a / np.sqrt(h**2 + k**2 + l**2) if (h**2 + k**2 + l**2) != 0 else float('inf')
        
        elif lattice_type == 'hexagonal':
            return 1.0 / np.sqrt(4/3 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)
        
        else:
            return 1.0 / np.sqrt(
                (h**2 / a**2) * np.sin(alpha)**2 +
                (k**2 / b**2) * np.sin(beta)**2 +
                (l**2 / c**2) * np.sin(gamma)**2
            )

    def _calculate_orientation(self):
        indexed_spots = [s for s in self.spots if s.h is not None]
        if len(indexed_spots) < 2:
            return

        g1 = np.array([indexed_spots[0].h, indexed_spots[0].k, indexed_spots[0].l])
        g2 = np.array([indexed_spots[1].h, indexed_spots[1].k, indexed_spots[1].l])
        
        zone_axis = np.cross(g1, g2)
        
        if np.linalg.norm(zone_axis) > 0:
            zone_axis = zone_axis // np.gcd.reduce(np.abs(zone_axis).astype(int))
        
        v1 = np.array([indexed_spots[0].x, indexed_spots[0].y, 0])
        v2 = np.array([indexed_spots[1].x, indexed_spots[1].y, 0])
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            angle_rad = np.arctan2(v2[1], v2[0]) - np.arctan2(g2[1], g2[0])
            rotation_angle = np.degrees(angle_rad)
        else:
            rotation_angle = 0.0

        self.orientation = CrystalOrientation(
            zone_axis=zone_axis,
            rotation_angle=rotation_angle,
            confidence=min(1.0, len(indexed_spots) / len(self.spots))
        )

    def reconstruct_reciprocal_lattice(self, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
        reciprocal_points = []
        
        for spot in self.spots:
            r = self.calculate_radius(spot, center)
            if r == 0:
                reciprocal_points.append([0, 0, 0])
                continue
            
            d = self.radius_to_dspacing(r)
            g = 1.0 / d
            
            dx = spot.x - center[0]
            dy = spot.y - center[1]
            theta = np.arctan2(dy, dx)
            
            gx = g * np.cos(theta)
            gy = g * np.sin(theta)
            
            reciprocal_points.append([gx, gy, 0])
        
        self.reciprocal_lattice = np.array(reciprocal_points)
        return self.reciprocal_lattice

    def plot_diffraction_pattern(self, center: Tuple[float, float] = (0, 0), 
                                 save_path: Optional[str] = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        x_coords = [s.x for s in self.spots]
        y_coords = [s.y for s in self.spots]
        
        ax1.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.7)
        ax1.scatter([center[0]], [center[1]], c='red', s=100, marker='+', label='中心斑点')
        ax1.set_xlabel('X (像素)')
        ax1.set_ylabel('Y (像素)')
        ax1.set_title('电子衍射花样')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend()
        
        for i, spot in enumerate(self.spots):
            if spot.h is not None:
                label = f'({spot.h},{spot.k},{spot.l})'
                ax1.annotate(label, (spot.x, spot.y), 
                            textcoords="offset points", xytext=(5, 5), ha='center')
        
        if self.reciprocal_lattice is not None:
            ax2.scatter(self.reciprocal_lattice[:, 0], 
                       self.reciprocal_lattice[:, 1], 
                       c='green', s=50, alpha=0.7)
            ax2.set_xlabel('g_x (Å⁻¹)')
            ax2.set_ylabel('g_y (Å⁻¹)')
            ax2.set_title('倒易空间重构')
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def print_calibration_report(self):
        print("=" * 60)
        print("TEM 电子衍射花样标定报告")
        print("=" * 60)
        print(f"\n相机长度: {self.camera_length} mm")
        print(f"电子波长: {self.wavelength} Å (200kV)")
        print(f"\n衍射斑点数量: {len(self.spots)}")
        
        if self.lattice_params:
            print("\n" + "-" * 60)
            print("晶格参数:")
            print("-" * 60)
            print(f"a = {self.lattice_params.a:.4f} Å")
            print(f"b = {self.lattice_params.b:.4f} Å")
            print(f"c = {self.lattice_params.c:.4f} Å")
            print(f"α = {self.lattice_params.alpha:.1f}°")
            print(f"β = {self.lattice_params.beta:.1f}°")
            print(f"γ = {self.lattice_params.gamma:.1f}°")
        
        if self.orientation:
            print("\n" + "-" * 60)
            print("晶体取向:")
            print("-" * 60)
            print(f"晶带轴: [{int(self.orientation.zone_axis[0])} "
                  f"{int(self.orientation.zone_axis[1])} "
                  f"{int(self.orientation.zone_axis[2])}]")
            print(f"旋转角度: {self.orientation.rotation_angle:.2f}°")
            print(f"标定置信度: {self.orientation.confidence*100:.1f}%")
        
        print("\n" + "-" * 60)
        print("标定的衍射斑点:")
        print("-" * 60)
        print(f"{'序号':>4} {'X(像素)':>10} {'Y(像素)':>10} {'R(像素)':>10} "
              f"{'d(Å)':>10} {'晶面指数':>12}")
        print("-" * 60)
        
        for i, spot in enumerate(self.spots):
            r = self.calculate_radius(spot)
            d = self.radius_to_dspacing(r) if r > 0 else 0
            hkl = f"({spot.h},{spot.k},{spot.l})" if spot.h is not None else "未标定"
            print(f"{i+1:>4} {spot.x:>10.2f} {spot.y:>10.2f} {r:>10.2f} "
                  f"{d:>10.4f} {hkl:>12}")
        
        print("\n" + "=" * 60)


def main():
    print("TEM 电子衍射花样标定程序")
    print("=" * 60)
    
    camera_length = 200.0
    wavelength = 0.02508
    
    calibrator = TEMDiffractionCalibrator(
        camera_length=camera_length,
        wavelength=wavelength
    )
    
    spots = [
        (0, 0),
        (50, 0),
        (0, 50),
        (50, 50),
        (-50, 0),
        (0, -50),
        (100, 0),
        (0, 100),
        (100, 100),
        (75, 25),
        (25, 75),
    ]
    
    calibrator.add_spots(spots)
    
    print("\n正在进行标定...")
    results = calibrator.calibrate_pattern(
        center=(0, 0),
        lattice_type='cubic'
    )
    
    reciprocal_lattice = calibrator.reconstruct_reciprocal_lattice()
    print(f"\n倒易空间重构完成，共 {len(reciprocal_lattice)} 个点")
    
    calibrator.print_calibration_report()
    
    print("\n正在生成可视化图像...")
    calibrator.plot_diffraction_pattern()


if __name__ == "__main__":
    main()
