import numpy as np
import matplotlib
matplotlib.use('Agg')

from cosmology import Cosmology
from halo_galaxy_connection import HODModel, AbundanceMatching
from galaxy_statistics import StellarMassFunction, GalaxyCorrelationFunction, compute_galaxy_bias
from galaxy_visualization import (
    plot_stellar_mass_function,
    plot_correlation_function,
    plot_galaxy_bias,
    plot_hod,
    plot_galaxy_distribution,
    plot_stellar_mass_vs_halo_mass
)

def generate_mock_halos(box_size, n_halos=1000, seed=42):
    np.random.seed(seed)
    
    logMh_min = 11.0
    logMh_max = 15.0
    
    alpha = -1.0
    logMh = np.random.uniform(logMh_min, logMh_max, n_halos)
    weights = (10**(logMh - 12))**alpha
    logMh = np.random.choice(logMh, n_halos, p=weights/np.sum(weights))
    
    M_h = 10**logMh
    
    pos = np.random.rand(n_halos, 3) * box_size
    
    cosmo = Cosmology()
    rho_crit = 2.775e11 * cosmo.h**2
    rho_vir = 200.0 * rho_crit
    r_vir = (3.0 * M_h / (4.0 * np.pi * rho_vir))**(1.0/3.0)
    
    halos = []
    for i in range(n_halos):
        halo = {
            'id': i,
            'mass': M_h[i],
            'center': pos[i],
            'radius': r_vir[i],
            'z': 0.0
        }
        halos.append(halo)
    
    return halos

def main():
    print("=" * 70)
    print("Semi-Analytic Galaxy Formation - Test Suite")
    print("=" * 70)
    
    cosmo = Cosmology()
    box_size = 256.0
    
    print("\nGenerating mock halo catalog...")
    halos = generate_mock_halos(box_size, n_halos=2000, seed=42)
    print(f"Generated {len(halos)} halos")
    
    print("\n" + "-" * 70)
    print("Test 1: Halo Occupation Distribution (HOD)")
    print("-" * 70)
    
    hod = HODModel(cosmo=cosmo)
    
    Mh_plot = np.logspace(11, 15, 100)
    N_cen = hod.N_cen(Mh_plot)
    N_sat = hod.N_sat(Mh_plot)
    N_tot = hod.N_total(Mh_plot)
    
    plot_hod(Mh_plot, N_cen, N_sat, N_tot, output_file='hod.png')
    
    print("\nPopulating halos with galaxies using HOD...")
    galaxies_hod = hod.assign_galaxies(halos)
    print(f"Assigned {len(galaxies_hod)} galaxies from HOD model")
    
    print("\n" + "-" * 70)
    print("Test 2: Stellar Mass Function")
    print("-" * 70)
    
    smf_calc = StellarMassFunction()
    
    stellar_masses = np.array([g['stellar_mass'] for g in galaxies_hod])
    logM_mid, phi, _ = smf_calc.compute_smf(stellar_masses, box_size, 
                                             n_bins=12, logM_min=8.5, logM_max=11.5)
    
    plot_stellar_mass_function(logM_mid, phi, output_file='smf.png')
    
    print("\n" + "-" * 70)
    print("Test 3: Galaxy Correlation Function")
    print("-" * 70)
    
    gcf = GalaxyCorrelationFunction()
    
    mass_threshold = 10**10.0
    galaxies_massive = [g for g in galaxies_hod if g['stellar_mass'] > mass_threshold]
    print(f"Using {len(galaxies_massive)} galaxies above {mass_threshold:.2e} M_sun/h")
    
    r_mid, xi, _ = gcf.compute_xi_simple(galaxies_massive, box_size, 
                                          n_bins=15, r_min=0.5, r_max=30.0)
    
    plot_correlation_function(r_mid, xi, output_file='xi.png')
    
    print("\n" + "-" * 70)
    print("Test 4: Galaxy Bias")
    print("-" * 70)
    
    n_dm_particles = 50000
    dm_pos = np.random.rand(n_dm_particles, 3) * box_size
    
    r_bias, bias, xi_gal, xi_dm = compute_galaxy_bias(
        galaxies_massive, box_size, dm_pos,
        n_bins=12, r_min=1.0, r_max=20.0
    )
    
    plot_galaxy_bias(r_bias, bias, output_file='bias.png')
    
    print("\n" + "-" * 70)
    print("Test 5: Galaxy Visualization")
    print("-" * 70)
    
    plot_galaxy_distribution(galaxies_hod, box_size, output_file='galaxy_distribution.png')
    plot_stellar_mass_vs_halo_mass(galaxies_hod, output_file='mstar_mhalo.png')
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("Output files:")
    print("  - hod.png: Halo Occupation Distribution")
    print("  - smf.png: Stellar Mass Function")
    print("  - xi.png: Galaxy Correlation Function")
    print("  - bias.png: Galaxy Bias")
    print("  - galaxy_distribution.png: Galaxy Distribution")
    print("  - mstar_mhalo.png: M*-Mh Relation")
    print("=" * 70)

if __name__ == "__main__":
    main()
