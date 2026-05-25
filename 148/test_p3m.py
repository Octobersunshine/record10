import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cosmology import Cosmology
from simulation import NBodySimulation
from visualization import (
    plot_density_slice, 
    plot_projection, 
    plot_particles,
    plot_power_spectrum_vs_scale
)
from halo_finder import find_halos_and_mass_function, plot_mass_function

def run_comparison():
    print("=" * 70)
    print("PM vs P³M Comparison Test")
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
    ngrid = 64
    
    print("\n" + "=" * 70)
    print("RUN 1: PM METHOD")
    print("=" * 70)
    
    sim_pm = NBodySimulation(
        npart=npart,
        box_size=box_size,
        z_init=z_init,
        ngrid=ngrid,
        seed=42,
        cosmo=cosmo,
        use_p3m=False
    )
    
    sim_pm.initialize()
    sim_pm.run(a_final=1.0, nsteps=30)
    
    final_pm = sim_pm.get_final_snapshot()
    
    print("\n" + "=" * 70)
    print("RUN 2: P³M METHOD")
    print("=" * 70)
    
    sim_p3m = NBodySimulation(
        npart=npart,
        box_size=box_size,
        z_init=z_init,
        ngrid=ngrid,
        seed=42,
        cosmo=cosmo,
        use_p3m=True
    )
    
    sim_p3m.initialize()
    sim_p3m.run(a_final=1.0, nsteps=30)
    
    final_p3m = sim_p3m.get_final_snapshot()
    
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    
    compare_density_slices(
        final_pm['delta'], final_p3m['delta'],
        box_size, final_pm['z']
    )
    
    compare_power_spectra(
        final_pm['delta'], final_p3m['delta'],
        box_size, final_pm['z']
    )
    
    compare_halo_mass_functions(
        final_pm['pos'], final_p3m['pos'],
        box_size, cosmo, sim_pm.npart, final_pm['z']
    )
    
    compare_particle_distributions(
        final_pm['pos'], final_p3m['pos'],
        box_size, final_pm['z']
    )
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("Output files:")
    print("  - density_comparison.png")
    print("  - power_spectrum_comparison.png")
    print("  - mass_function_comparison.png")
    print("  - particle_comparison.png")
    print("=" * 70)

def compare_density_slices(delta_pm, delta_p3m, box_size, z):
    ng = delta_pm.shape[0]
    slice_idx = ng // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    slice_pm = delta_pm[slice_idx, :, :]
    slice_p3m = delta_p3m[slice_idx, :, :]
    
    vmin = min(slice_pm.min(), slice_p3m.min())
    vmax = max(slice_pm.max(), slice_p3m.max())
    
    im1 = axes[0].imshow(slice_pm, cmap='inferno', 
                         extent=[0, box_size, 0, box_size],
                         origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'PM Method (z={z:.2f})')
    axes[0].set_xlabel('x [Mpc/h]')
    axes[0].set_ylabel('y [Mpc/h]')
    
    im2 = axes[1].imshow(slice_p3m, cmap='inferno', 
                         extent=[0, box_size, 0, box_size],
                         origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'P³M Method (z={z:.2f})')
    axes[1].set_xlabel('x [Mpc/h]')
    
    cbar = fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label(r'$\delta = \rho/\bar{\rho} - 1$')
    
    plt.tight_layout()
    plt.savefig('density_comparison.png', dpi=150)
    plt.close()
    print("Saved density comparison to density_comparison.png")

def compare_power_spectra(delta_pm, delta_p3m, box_size, z):
    def compute_pk(delta):
        ng = delta.shape[0]
        kf = 2.0 * np.pi / box_size
        
        delta_k = np.fft.fftn(delta)
        Pk = np.abs(delta_k)**2 * box_size**3 / ng**6
        
        kx = np.fft.fftfreq(ng, d=1.0/ng) * kf
        ky = np.fft.fftfreq(ng, d=1.0/ng) * kf
        kz = np.fft.fftfreq(ng, d=1.0/ng) * kf
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        
        k_bins = np.logspace(np.log10(kf), np.log10(ng * kf / 2), 20)
        k_mid = np.sqrt(k_bins[1:] * k_bins[:-1])
        
        Pk_mean = np.zeros_like(k_mid)
        for i in range(len(k_bins) - 1):
            mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
            if np.sum(mask) > 0:
                Pk_mean[i] = np.mean(Pk[mask])
        
        return k_mid, Pk_mean
    
    k_pm, Pk_pm = compute_pk(delta_pm)
    k_p3m, Pk_p3m = compute_pk(delta_p3m)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    mask_pm = Pk_pm > 0
    mask_p3m = Pk_p3m > 0
    
    ax.loglog(k_pm[mask_pm], Pk_pm[mask_pm], 'o-', label='PM', alpha=0.7)
    ax.loglog(k_p3m[mask_p3m], Pk_p3m[mask_p3m], 's-', label='P³M', alpha=0.7)
    
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k) [(Mpc/h)$^3$]')
    ax.set_title(f'Power Spectrum Comparison (z={z:.2f})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('power_spectrum_comparison.png', dpi=150)
    plt.close()
    print("Saved power spectrum comparison to power_spectrum_comparison.png")

def compare_halo_mass_functions(pos_pm, pos_p3m, box_size, cosmo, npart, z):
    rho_mean = cosmo.rho_m(z=0.0)
    volume_per_particle = box_size**3 / npart
    particle_mass = rho_mean * volume_per_particle
    
    print(f"\nRunning halo finder for PM...")
    halos_pm, mm_pm, dndlogM_pm = find_halos_and_mass_function(
        pos_pm, box_size, link_length=0.2, n_min=10, particle_mass=particle_mass
    )
    
    print(f"\nRunning halo finder for P³M...")
    halos_p3m, mm_p3m, dndlogM_p3m = find_halos_and_mass_function(
        pos_p3m, box_size, link_length=0.2, n_min=10, particle_mass=particle_mass
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    mask_pm = dndlogM_pm > 0
    mask_p3m = dndlogM_p3m > 0
    
    if np.sum(mask_pm) > 0:
        ax.loglog(mm_pm[mask_pm], dndlogM_pm[mask_pm], 'o-', 
                  label=f'PM ({len(halos_pm)} halos)', alpha=0.7)
    
    if np.sum(mask_p3m) > 0:
        ax.loglog(mm_p3m[mask_p3m], dndlogM_p3m[mask_p3m], 's-', 
                  label=f'P³M ({len(halos_p3m)} halos)', alpha=0.7)
    
    ax.set_xlabel('Mass [M$_\odot$/h]')
    ax.set_ylabel('dn/dlogM [h$^3$ Mpc$^{-3}$]')
    ax.set_title(f'Halo Mass Function Comparison (z={z:.2f})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('mass_function_comparison.png', dpi=150)
    plt.close()
    print("Saved mass function comparison to mass_function_comparison.png")

def compare_particle_distributions(pos_pm, pos_p3m, box_size, z):
    npart = pos_pm.shape[0]
    max_plot = min(npart, 50000)
    
    idx = np.random.choice(npart, max_plot, replace=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    axes[0].scatter(pos_pm[idx, 0], pos_pm[idx, 1], s=0.5, alpha=0.5, c='k')
    axes[0].set_title(f'PM Method (z={z:.2f})')
    axes[0].set_xlabel('x [Mpc/h]')
    axes[0].set_ylabel('y [Mpc/h]')
    axes[0].set_xlim(0, box_size)
    axes[0].set_ylim(0, box_size)
    
    axes[1].scatter(pos_p3m[idx, 0], pos_p3m[idx, 1], s=0.5, alpha=0.5, c='k')
    axes[1].set_title(f'P³M Method (z={z:.2f})')
    axes[1].set_xlabel('x [Mpc/h]')
    axes[1].set_xlim(0, box_size)
    axes[1].set_ylim(0, box_size)
    
    plt.tight_layout()
    plt.savefig('particle_comparison.png', dpi=150)
    plt.close()
    print("Saved particle distribution comparison to particle_comparison.png")

if __name__ == "__main__":
    run_comparison()
