import numpy as np
import matplotlib
matplotlib.use('Agg')

from cosmology import Cosmology
from simulation import NBodySimulation
from halo_finder import FriendsOfFriends, mass_function
from halo_galaxy_connection import HODModel
from galaxy_statistics import StellarMassFunction, GalaxyCorrelationFunction
from galaxy_visualization import (
    plot_stellar_mass_function,
    plot_correlation_function,
    plot_galaxy_distribution,
    plot_stellar_mass_vs_halo_mass,
    plot_sam_galaxy_properties
)
from galaxy_formation import SemiAnalyticModel

def main():
    print("=" * 70)
    print("Full Pipeline: N-body + Halo Finding + Galaxy Formation")
    print("=" * 70)
    
    cosmo = Cosmology()
    
    npart = 32**3
    box_size = 128.0
    ngrid = 64
    
    print("\n" + "=" * 70)
    print("Step 1: N-body Simulation (P³M)")
    print("=" * 70)
    
    sim = NBodySimulation(
        npart=npart,
        box_size=box_size,
        z_init=50.0,
        ngrid=ngrid,
        seed=42,
        cosmo=cosmo,
        use_p3m=True
    )
    
    sim.initialize()
    sim.run(a_final=1.0, nsteps=20)
    
    final = sim.get_final_snapshot()
    pos = final['pos']
    
    print("\n" + "=" * 70)
    print("Step 2: Halo Finding (FoF)")
    print("=" * 70)
    
    rho_mean = cosmo.rho_m(z=0.0)
    volume_per_particle = box_size**3 / sim.npart
    particle_mass = rho_mean * volume_per_particle
    
    fof = FriendsOfFriends(pos, box_size, link_length=0.2, n_min=10)
    halos = fof.find_halos()
    halos = fof.calculate_halo_properties(particle_mass)
    
    halo_list = []
    for i, halo in enumerate(halos):
        halo_dict = {
            'id': i,
            'mass': halo['mass'],
            'center': halo['center'],
            'radius': halo['radius'],
            'size': halo['size'],
            'z': 0.0
        }
        halo_list.append(halo_dict)
    
    print(f"Found {len(halo_list)} halos")
    
    print("\n" + "=" * 70)
    print("Step 3: Populate Halos with Galaxies (HOD)")
    print("=" * 70)
    
    hod = HODModel(cosmo=cosmo)
    galaxies = hod.assign_galaxies(halo_list)
    
    print(f"Assigned {len(galaxies)} galaxies")
    
    print("\n" + "=" * 70)
    print("Step 4: Stellar Mass Function")
    print("=" * 70)
    
    smf_calc = StellarMassFunction()
    stellar_masses = np.array([g['stellar_mass'] for g in galaxies])
    
    if len(stellar_masses) > 0:
        logM_mid, phi, _ = smf_calc.compute_smf(stellar_masses, box_size,
                                                 n_bins=10, logM_min=8.0, logM_max=12.0)
        plot_stellar_mass_function(logM_mid, phi, output_file='smf_full.png')
    else:
        print("Warning: No galaxies found!")
    
    print("\n" + "=" * 70)
    print("Step 5: Galaxy Correlation Function")
    print("=" * 70)
    
    gcf = GalaxyCorrelationFunction()
    
    if len(galaxies) >= 10:
        mass_threshold = np.percentile(stellar_masses, 50)
        galaxies_massive = [g for g in galaxies if g['stellar_mass'] > mass_threshold]
        print(f"Using {len(galaxies_massive)} galaxies above {mass_threshold:.2e} M_sun/h")
        
        if len(galaxies_massive) >= 10:
            r_mid, xi, _ = gcf.compute_xi_simple(galaxies_massive, box_size,
                                                  n_bins=10, r_min=1.0, r_max=30.0)
            plot_correlation_function(r_mid, xi, output_file='xi_full.png')
    
    print("\n" + "=" * 70)
    print("Step 6: Semi-Analytic Model (SAM)")
    print("=" * 70)
    
    sam = SemiAnalyticModel(cosmo=cosmo)
    sam_galaxies = sam.populate_halos(halo_list, z=0.0)
    
    print(f"Evolved {len(sam_galaxies)} galaxies with SAM")
    
    if len(sam_galaxies) > 0:
        plot_sam_galaxy_properties(sam_galaxies, output_file='sam_properties.png')
    
    print("\n" + "-" * 70)
    print("Plotting galaxy distributions")
    print("-" * 70)
    
    if len(galaxies) > 0:
        plot_galaxy_distribution(galaxies, box_size, output_file='galaxy_distribution_full.png')
        plot_stellar_mass_vs_halo_mass(galaxies, output_file='mstar_mhalo_full.png')
    
    print("\n" + "=" * 70)
    print("Full pipeline complete!")
    print("Output files:")
    print("  - smf_full.png: Stellar Mass Function")
    print("  - xi_full.png: Galaxy Correlation Function")
    print("  - galaxy_distribution_full.png: Galaxy Distribution")
    print("  - mstar_mhalo_full.png: M*-Mh Relation")
    print("  - sam_properties.png: SAM Galaxy Properties")
    print("=" * 70)

if __name__ == "__main__":
    main()
