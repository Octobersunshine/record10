from baryon_models import BaryonicModels, generate_sample_rotation_curve
import numpy as np

print("Testing baryon models...")

baryon = BaryonicModels()

r = np.linspace(2, 30, 10)

M_disk = 0.05
R_disk = 3.0
v_disk = baryon.stellar_disk(r, M_disk, R_disk)
print(f"Stellar disk velocity range: {v_disk.min():.1f} to {v_disk.max():.1f} km/s")

M_bulge = 0.01
r_bulge = 0.5
v_bulge = baryon.stellar_bulge(r, M_bulge, r_bulge)
print(f"Bulge velocity range: {v_bulge.min():.1f} to {v_bulge.max():.1f} km/s")

M_hi = 0.015
R_hi = 8.0
v_hi = baryon.hi_gas_disk(r, M_hi, R_hi)
print(f"HI gas velocity range: {v_hi.min():.1f} to {v_hi.max():.1f} km/s")

M_h2 = 0.005
R_h2 = 4.0
v_h2 = baryon.molecular_gas(r, M_h2, R_h2)
print(f"H2 gas velocity range: {v_h2.min():.1f} to {v_h2.max():.1f} km/s")

baryon_params = [M_disk, R_disk, M_bulge, r_bulge, M_hi, R_hi, M_h2, R_h2]
v_total, components = baryon.total_baryonic_v(r, baryon_params)
print(f"Total baryon velocity range: {v_total.min():.1f} to {v_total.max():.1f} km/s")

print("\nTesting generate_sample_rotation_curve...")
r, v_obs, v_err, true_params = generate_sample_rotation_curve()
print(f"Generated velocity range: {v_obs.min():.1f} to {v_obs.max():.1f} km/s")

if v_obs.max() < 400:
    print("SUCCESS: Velocities are in reasonable range!")
else:
    print(f"WARNING: Velocities too high! Max: {v_obs.max():.1f}")
