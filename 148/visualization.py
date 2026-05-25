import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_density_slice(density, box_size, z=0.0, slice_idx=None, output_file='density_slice.png'):
    ng = density.shape[0]
    
    if slice_idx is None:
        slice_idx = ng // 2
    
    slice_2d = density[slice_idx, :, :]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(slice_2d, cmap='inferno', 
                   extent=[0, box_size, 0, box_size],
                   origin='lower')
    
    ax.set_xlabel('x [Mpc/h]')
    ax.set_ylabel('y [Mpc/h]')
    ax.set_title(f'Density Field Slice (z={z:.2f})')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\delta = \rho/\bar{\rho} - 1$')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved density slice to {output_file}")

def plot_projection(density, box_size, z=0.0, axis='z', output_file='density_projection.png'):
    if axis == 'x':
        proj = np.mean(density, axis=0)
    elif axis == 'y':
        proj = np.mean(density, axis=1)
    else:
        proj = np.mean(density, axis=2)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(proj, cmap='inferno', 
                   extent=[0, box_size, 0, box_size],
                   origin='lower')
    
    ax.set_xlabel('x [Mpc/h]')
    ax.set_ylabel('y [Mpc/h]')
    ax.set_title(f'Density Field Projection (z={z:.2f})')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\delta = \rho/\bar{\rho} - 1$')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved density projection to {output_file}")

def plot_particles(pos, box_size, z=0.0, axis1=0, axis2=1, max_particles=100000, output_file='particles.png'):
    npart = pos.shape[0]
    
    if npart > max_particles:
        idx = np.random.choice(npart, max_particles, replace=False)
        pos_plot = pos[idx]
    else:
        pos_plot = pos
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.scatter(pos_plot[:, axis1], pos_plot[:, axis2], s=0.5, alpha=0.5, c='k')
    
    labels = ['x [Mpc/h]', 'y [Mpc/h]', 'z [Mpc/h]']
    ax.set_xlabel(labels[axis1])
    ax.set_ylabel(labels[axis2])
    ax.set_title(f'Particle Distribution (z={z:.2f})')
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved particle plot to {output_file}")

def plot_power_spectrum_vs_scale(density, box_size, z=0.0, output_file='power_spectrum.png'):
    ng = density.shape[0]
    kf = 2.0 * np.pi / box_size
    
    delta_k = np.fft.fftn(density)
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
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.loglog(k_mid, Pk_mean, 'o-', label='Measured')
    
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k) [(Mpc/h)$^3$]')
    ax.set_title(f'Power Spectrum (z={z:.2f})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved power spectrum to {output_file}")
    
    return k_mid, Pk_mean
