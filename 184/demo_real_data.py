from rotation_curve_fitter import GalaxyRotationCurveFitter
import numpy as np

np.random.seed(123)

print("=" * 65)
print("DEMO: Fitting Real Galaxy Rotation Curve Data")
print("=" * 65)

fitter = GalaxyRotationCurveFitter()

print("\nLoading data from sample_observation_data.csv...")
fitter.load_from_file('sample_observation_data.csv', delimiter=',', skiprows=2)

print(f"Loaded {len(fitter.r)} data points")
print(f"Radius range: {fitter.r.min():.1f} - {fitter.r.max():.1f} kpc")
print(f"Velocity range: {fitter.v.min():.1f} - {fitter.v.max():.1f} km/s")

print("\n" + "-" * 65)
print("Fitting both NFW and Burkert models...")
fitter.fit_both()

fitter.print_results()

print("\n" + "-" * 65)
print("Generating plot...")
fitter.plot('real_data_fit.png', show=False)
print("Plot saved as: real_data_fit.png")

print("\n" + "=" * 65)
print("Physical Interpretation:")
print("=" * 65)

if fitter.nfw_params is not None:
    rho_s, r_s = fitter.nfw_params
    rho_s_physical = rho_s * 1e-9
    print(f"\nNFW Profile:")
    print(f"  Characteristic density: {rho_s_physical:.3e} M_sun/pc^3")
    print(f"  Scale radius: {r_s:.2f} kpc")
    print(f"  Concentration parameter (c = R_vir/r_s, assuming R_vir ~ 100 kpc): ~{100/r_s:.1f}")

if fitter.burkert_params is not None:
    rho_0, r_0 = fitter.burkert_params
    rho_0_physical = rho_0 * 1e-9
    print(f"\nBurkert Profile:")
    print(f"  Central density: {rho_0_physical:.3e} M_sun/pc^3")
    print(f"  Core radius: {r_0:.2f} kpc")

print("\n" + "=" * 65)
