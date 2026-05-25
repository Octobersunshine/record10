import numpy as np
from tem_diffraction_calibration import (
    TEMDiffractionCalibrator, 
    LatticeParameters, 
    DiffractionSpot
)


def example_cubic_calibration():
    print("=" * 70)
    print("示例 1: 立方晶系电子衍射花样标定")
    print("=" * 70)
    
    camera_length = 200.0
    wavelength = 0.02508
    
    calibrator = TEMDiffractionCalibrator(
        camera_length=camera_length,
        wavelength=wavelength
    )
    
    a_fcc = 4.08
    spots_pixels = generate_cubic_diffraction_spots(a_fcc, camera_length, wavelength)
    
    print(f"\n生成的衍射斑点数量: {len(spots_pixels)}")
    calibrator.add_spots(spots_pixels)
    
    results = calibrator.calibrate_pattern(
        center=(0, 0),
        lattice_type='cubic'
    )
    
    reciprocal_lattice = calibrator.reconstruct_reciprocal_lattice()
    
    calibrator.print_calibration_report()
    
    return calibrator


def example_with_known_parameters():
    print("\n" + "=" * 70)
    print("示例 2: 使用已知晶格参数进行标定")
    print("=" * 70)
    
    camera_length = 150.0
    
    calibrator = TEMDiffractionCalibrator(
        camera_length=camera_length,
        wavelength=0.02508
    )
    
    known_params = LatticeParameters(
        a=3.615,
        b=3.615,
        c=3.615,
        alpha=90,
        beta=90,
        gamma=90
    )
    
    spots = [
        (0, 0),
        (37.2, 0),
        (0, 37.2),
        (37.2, 37.2),
        (-37.2, 0),
        (0, -37.2),
        (52.6, 52.6),
        (74.4, 0),
    ]
    
    calibrator.add_spots(spots)
    
    results = calibrator.calibrate_pattern(
        center=(0, 0),
        lattice_type='cubic',
        known_params=known_params
    )
    
    reciprocal_lattice = calibrator.reconstruct_reciprocal_lattice()
    
    calibrator.print_calibration_report()
    
    return calibrator


def example_hexagonal_calibration():
    print("\n" + "=" * 70)
    print("示例 3: 六方晶系电子衍射花样标定")
    print("=" * 70)
    
    camera_length = 200.0
    wavelength = 0.02508
    
    calibrator = TEMDiffractionCalibrator(
        camera_length=camera_length,
        wavelength=wavelength
    )
    
    known_params = LatticeParameters(
        a=0.321 * 10,
        b=0.321 * 10,
        c=0.521 * 10,
        alpha=90,
        beta=90,
        gamma=120
    )
    
    spots = generate_hexagonal_diffraction_spots(known_params, camera_length, wavelength)
    calibrator.add_spots(spots)
    
    results = calibrator.calibrate_pattern(
        center=(0, 0),
        lattice_type='hexagonal',
        known_params=known_params
    )
    
    reciprocal_lattice = calibrator.reconstruct_reciprocal_lattice()
    
    calibrator.print_calibration_report()
    
    return calibrator


def generate_cubic_diffraction_spots(a: float, camera_length: float, 
                                     wavelength: float, 
                                     zone_axis: list = None) -> list:
    if zone_axis is None:
        zone_axis = [0, 0, 1]
    
    spots = [(0, 0)]
    
    for h in range(-5, 6):
        for k in range(-5, 6):
            for l in range(-5, 6):
                if h == 0 and k == 0 and l == 0:
                    continue
                
                if (h * zone_axis[0] + k * zone_axis[1] + l * zone_axis[2]) != 0:
                    continue
                
                hkl_sum = h**2 + k**2 + l**2
                if hkl_sum == 0:
                    continue
                
                d = a / np.sqrt(hkl_sum)
                radius = (wavelength * camera_length) / d
                
                angle = np.arctan2(k, h) if (h != 0 or k != 0) else 0
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                if not any(np.isclose(x, s[0]) and np.isclose(y, s[1]) for s in spots):
                    spots.append((x, y))
    
    return spots


def generate_hexagonal_diffraction_spots(params: LatticeParameters, 
                                        camera_length: float, 
                                        wavelength: float) -> list:
    spots = [(0, 0)]
    
    a = params.a
    c = params.c
    
    for h in range(-4, 5):
        for k in range(-4, 5):
            for l in range(-2, 3):
                if h == 0 and k == 0 and l == 0:
                    continue
                
                g_squared = 4/3 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2
                if g_squared <= 0:
                    continue
                
                d = 1.0 / np.sqrt(g_squared)
                radius = (wavelength * camera_length) / d
                
                angle = np.arctan2(np.sqrt(3) * (h + 2*k), (2*h + k)) if (h != 0 or k != 0) else 0
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                if not any(np.isclose(x, s[0], atol=1) and np.isclose(y, s[1], atol=1) for s in spots):
                    if abs(x) < 200 and abs(y) < 200:
                        spots.append((x, y))
    
    return spots


def detailed_calculation_example():
    print("\n" + "=" * 70)
    print("示例 4: 详细计算过程演示")
    print("=" * 70)
    
    camera_length = 200.0
    wavelength = 0.02508
    
    print(f"\n基本参数:")
    print(f"  相机长度 L = {camera_length} mm")
    print(f"  电子波长 λ = {wavelength} Å (200kV)")
    
    radius_pixels = 50.0
    print(f"\n衍射斑点半径 R = {radius_pixels} 像素")
    
    d = (wavelength * camera_length) / radius_pixels
    print(f"\n根据布拉格定律 Rd = λL:")
    print(f"  d = λL / R = {wavelength} × {camera_length} / {radius_pixels}")
    print(f"  d = {d:.4f} Å")
    
    g = 1.0 / d
    print(f"\n倒易空间矢量大小 |g| = 1/d = {g:.4f} Å⁻¹")
    
    calibrator = TEMDiffractionCalibrator(camera_length, wavelength)
    
    spots = [
        (0, 0),
        (50, 0),
        (0, 50),
        (70.7, 70.7),
        (-50, 0),
        (100, 0),
    ]
    
    calibrator.add_spots(spots)
    reciprocal = calibrator.reconstruct_reciprocal_lattice()
    
    print(f"\n倒易空间重构坐标:")
    for i, (x, y, z) in enumerate(reciprocal):
        print(f"  斑点 {i+1}: ({x:.4f}, {y:.4f}, {z:.4f}) Å⁻¹")


if __name__ == "__main__":
    calibrator1 = example_cubic_calibration()
    calibrator2 = example_with_known_parameters()
    calibrator3 = example_hexagonal_calibration()
    detailed_calculation_example()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成！")
    print("=" * 70)
