import numpy as np
import matplotlib
matplotlib.use('Agg')

from cosmology import Cosmology
from simulation import NBodySimulation
from visualization import (
    plot_density_slice, 
    plot_projection, 
    plot_particles,
    plot_power_spectrum_vs_scale
)
from halo_finder import find_halos_and_mass_function, plot_mass_function
from halo_galaxy_connection import HODModel
from galaxy_statistics import StellarMassFunction, GalaxyCorrelationFunction
from galaxy_visualization import (
    plot_stellar_mass_function,
    plot_correlation_function,
    plot_galaxy_distribution,
    plot_stellar_mass_vs_halo_mass
)

def main():
    print("=" * 70)
    print("N-body Simulation + Galaxy Formation Pipeline")
    print("=" * 70)
    
    cosmo = Cosmology(
        Omega_m=0.3089,
        Omega_L=0.6911,
        Omega_b=0.0486,
        h=0.6774,
        ns=0.9667,
        sigma8=0.8159
    )
    
    npart = 32**3
    box_size = 128.0
    z_init = 50.0
    use_p3m = True
    use_galaxy_formation = True
    
    sim = NBodySimulation(
        npart=npart,
        box_size=box_size,
        z_init=z_init,
        ngrid=64,
        seed=42,
        cosmo=cosmo,
        use_p3m=use_p3m
    )
    
    sim.initialize()
    
    print("\nInitial conditions:")
    print(f"  Number of particles: {sim.npart}")
    print(f"  Box size: {box_size} Mpc/h")
    print(f"  Grid size: {sim.ngrid}^3")
    print(f"  Initial redshift: z={z_init}")
    print(f"  Using P³M: {use_p3m}")
    print(f"  Using Galaxy Formation: {use_galaxy_formation}")
    
    init_snapshot = sim.get_final_snapshot()
    if init_snapshot is None:
        init_density = sim.get_density_field()
        init_z = sim.z
    else:
        init_density = init_snapshot['delta']
        init_z = init_snapshot['z']
    
    plot_density_slice(init_density, box_size, z=init_z, output_file='density_slice_initial.png')
    plot_particles(sim.pos, box_size, z=init_z, output_file='particles_initial.png')
    
    sim.run(a_final=1.0, nsteps=30)
    
    final_snapshot = sim.get_final_snapshot()
    final_density = final_snapshot['delta']
    final_pos = final_snapshot['pos']
    final_z = final_snapshot['z']
    
    print(f"\nFinal redshift: z={final_z:.4f}")
    
    plot_density_slice(final_density, box_size, z=final_z, output_file='density_slice_final.png')
    plot_projection(final_density, box_size, z=final_z, output_file='density_projection_final.png')
    plot_particles(final_pos, box_size, z=final_z, output_file='particles_final.png')
    plot_power_spectrum_vs_scale(final_density, box_size, z=final_z, output_file='power_spectrum_final.png')
    
    print("\n" + "-" * 70)
    print("Running halo finder...")
    print("-" * 70)
    
    rho_mean = cosmo.rho_m(z=0.0)
    volume_per_particle = box_size**3 / sim.npart
    particle_mass = rho_mean * volume_per_particle
    
    print(f"Particle mass: {particle_mass:.2e} M_sun/h")
    
    halos, mass_mid, dndlogM = find_halos_and_mass_function(
        final_pos,
        box_size,
        link_length=0.2,
        n_min=10,
        particle_mass=particle_mass
    )
    
    if len(halos) > 0:
        plot_mass_function(mass_mid, dndlogM, z=final_z, output_file='mass_function.png')
        
        print(f"\nHalo statistics:")
        print(f"  Number of halos: {len(halos)}")
        print(f"  Most massive halo: {max(h['mass'] for h in halos):.2e} M_sun/h")
        print(f"  Least massive halo: {min(h['mass'] for h in halos):.2e} M_sun/h")
    
    if use_galaxy_formation and len(halos) > 0:
        print("\n" + "-" * 70)
        print("Galaxy Formation: Populating halos with galaxies")
        print("-" * 70)
        
        halo_list = []
        for i, halo in enumerate(halos):
            halo_dict = {
                'id': i,
                'mass': halo['mass'],
                'center': halo['center'],
                'radius': halo['radius'],
                'z': 0.0
            }
            halo_list.append(halo_dict)
        
        hod = HODModel(cosmo=cosmo)
        galaxies = hod.assign_galaxies(halo_list)
        
        print(f"Assigned {len(galaxies)} galaxies")
        
        print("\n" + "-" * 70)
        print("Computing stellar mass function...")
        print("-" * 70)
        
        smf_calc = StellarMassFunction()
        stellar_masses = np.array([g['stellar_mass'] for g in galaxies])
        
        logM_mid, phi, _ = smf_calc.compute_smf(stellar_masses, box_size,
                                                 n_bins=10, logM_min=8.0, logM_max=12.0)
        
        plot_stellar_mass_function(logM_mid, phi, output_file='stellar_mass_function.png')
        
        print("\n" + "-" * 70)
        print("Computing galaxy correlation function...")
        print("-" * 70)
        
        gcf = GalaxyCorrelationFunction()
        
        if len(galaxies) >= 20:
            mass_threshold = np.percentile(stellar_masses, 50)
            galaxies_massive = [g for g in galaxies if g['stellar_mass'] > mass_threshold]
            print(f"Using {len(galaxies_massive)} galaxies above {mass_threshold:.2e} M_sun/h")
            
            if len(galaxies_massive) >= 10:
                r_mid, xi, _ = gcf.compute_xi_simple(galaxies_massive, box_size,
                                                      n_bins=12, r_min=1.0, r_max=30.0)
                plot_correlation_function(r_mid, xi, output_file='galaxy_correlation_function.png')
        
        print("\n" + "-" * 70)
        print("Generating galaxy visualization...")
        print("-" * 70)
        
        plot_galaxy_distribution(galaxies, box_size, output_file='galaxy_distribution.png')
        plot_stellar_mass_vs_halo_mass(galaxies, output_file='mstar_mhalo_relation.png')
    
    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("Output files:")
    print("  - density_slice_initial.png")
    print("  - particles_initial.png")
    print("  - density_slice_final.png")
    print("  - density_projection_final.png")
    print("  - particles_final.png")
    print("  - power_spectrum_final.png")
    print("  - mass_function.png")
    if use_galaxy_formation and len(halos) > 0:
        print("  - stellar_mass_function.png")
        print("  - galaxy_correlation_function.png")
        print("  - galaxy_distribution.png")
        print("  - mstar_mhalo_relation.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
