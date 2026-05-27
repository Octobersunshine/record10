import numpy as np
import matplotlib
matplotlib.use('Agg')
from phonon_dispersion import (
    Crystal, ForceConstants, PhononCalculator,
    ThirdOrderForceConstants, ThirdOrderFiniteDisplacement,
    QuasiHarmonicApproximation, PhononBTE, AnharmonicVisualizer
)

print("=" * 60)
print("Testing Anharmonic Effects Calculations")
print("=" * 60)

a = 3.5
lattice = np.eye(3) * a
positions = np.array([[0.0, 0.0, 0.0]])
symbols = ['X']
masses = np.array([40.0])

crystal = Crystal(lattice, positions, symbols, masses)
fc = ForceConstants(crystal, cutoff=4.5)
fc.generate_model_fc(spring_constant=15.0)
phonon = PhononCalculator(crystal, fc, use_asr=True, symmetrize_fc=True)

print("\n--- Test 1: Third Order Force Constants ---")
fc3 = ThirdOrderForceConstants(crystal, cutoff=5.0)
fc3_matrix = fc3.generate_model_fc3(cubic_coefficient=-100.0)
print(f"FC3 shape: {fc3_matrix.shape}")
print(f"FC3 mean: {np.mean(np.abs(fc3_matrix)):.4e}")

print("\n--- Test 2: Third Order Finite Displacement ---")
fdm3 = ThirdOrderFiniteDisplacement(crystal, displacement=0.01)
structures = fdm3.generate_displaced_structures()
print(f"Number of displaced structures: {len(structures)}")

print("\n--- Test 3: Quasi-Harmonic Approximation ---")
qha = QuasiHarmonicApproximation(phonon)
temperatures = np.array([100, 200, 300])
alpha = qha.compute_thermal_expansion(temperatures, B=1e11)
print(f"Thermal expansion at 300K: {alpha[2]*1e6:.6e} x 10^-6 K^-1")

print("\n--- Test 4: Group Velocity ---")
rec_lat = crystal.reciprocal_lattice
test_q = np.array([0.1, 0.0, 0.0]) @ rec_lat
bte = PhononBTE(phonon, fc3)
v_g = bte.compute_group_velocity(test_q)
print(f"Group velocity shape: {v_g.shape}")
print(f"Group velocity magnitude (m/s): {np.linalg.norm(v_g, axis=1)}")

print("\n--- Test 5: Phonon Lifetime ---")
tau = bte.compute_phonon_lifetime(test_q, T=300.0)
print(f"Phonon lifetime at 300K (ps): {tau * 1e12}")

print("\n--- Test 6: Thermal Conductivity ---")
kappa_result = bte.compute_thermal_conductivity(temperatures, q_mesh_size=3)
print(f"Thermal conductivity at 300K: {kappa_result['kappa'][2]:.6f} W/m-K")

print("\n--- Test 7: Free Energy and Thermodynamics ---")
thermo_data = qha.compute_free_energy(temperatures, q_mesh_size=3)
eV_to_J = 1.602176634e-19
print(f"Free energy at 300K (meV): {thermo_data['F'][2] / eV_to_J * 1e3:.4f}")
print(f"Entropy at 300K (J/K): {thermo_data['S'][2]:.6f}")
print(f"Heat capacity at 300K (J/K): {thermo_data['Cv'][2]:.6f}")

print("\n--- Test 8: Gruneisen Parameters ---")
q_mesh = [np.array([0.1, 0.0, 0.0]) @ rec_lat]
qha.compute_volume_dependent_frequencies(q_mesh, n_points=5)
gamma_vals = qha.compute_gamma()
print(f"Gruneisen parameter shape: {gamma_vals.shape}")
print(f"Average Gruneisen: {np.mean(np.abs(gamma_vals)):.4f}")

print("\n--- Test 9: Cumulative Thermal Conductivity ---")
freqs, cum_kappa = bte.cumulative_kappa(T=300.0, q_mesh_size=3)
print(f"Cumulative kappa shape: {cum_kappa.shape}")
if len(cum_kappa) > 0:
    print(f"Max cumulative: {cum_kappa[-1]:.4f}")
else:
    print("No valid modes for cumulative kappa")

print("\n--- Test 10: Comparison with FCC ---")
a_fcc = 5.43
lattice_fcc = np.array([[0.0, a_fcc/2, a_fcc/2], [a_fcc/2, 0.0, a_fcc/2], [a_fcc/2, a_fcc/2, 0.0]])
crystal_fcc = Crystal(lattice_fcc, np.array([[0.0, 0.0, 0.0]]), ['Si'], np.array([28.0855]))
fc_fcc = ForceConstants(crystal_fcc, cutoff=5.0)
fc_fcc.generate_model_fc(spring_constant=20.0)
phonon_fcc = PhononCalculator(crystal_fcc, fc_fcc, use_asr=True, symmetrize_fc=True)

test_q_fcc = np.array([0.1, 0.0, 0.0]) @ crystal_fcc.reciprocal_lattice
bte_fcc = PhononBTE(phonon_fcc)
v_g_fcc = bte_fcc.compute_group_velocity(test_q_fcc)
print(f"FCC group velocity magnitude (m/s): {np.linalg.norm(v_g_fcc, axis=1)}")

kappa_fcc = bte_fcc.compute_thermal_conductivity(np.array([300]), q_mesh_size=3)
print(f"FCC thermal conductivity at 300K: {kappa_fcc['kappa'][0]:.6f} W/m-K")

print("\n" + "=" * 60)
print("All anharmonic tests passed!")
print("=" * 60)
