import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_stellar_mass_function(logM_mid, phi, output_file='smf.png'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    mask = phi > 0
    ax.semilogy(logM_mid[mask], phi[mask], 'o-', label='Simulation', color='blue')
    
    from galaxy_statistics import StellarMassFunction
    smf = StellarMassFunction()
    logMs_plot = np.linspace(8, 12, 100)
    phi_schechter = smf.schechter_function(logMs_plot)
    ax.semilogy(logMs_plot, phi_schechter, '--', label='Schechter fit', color='red', alpha=0.7)
    
    ax.set_xlabel(r'$\log_{10}(M_* / M_\odot)$')
    ax.set_ylabel(r'$\Phi$ [h$^3$ Mpc$^{-3}$ dex$^{-1}$]')
    ax.set_title('Stellar Mass Function')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([1e-5, 1e-1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved stellar mass function to {output_file}")

def plot_correlation_function(r_mid, xi, output_file='xi.png'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    mask = xi > 0
    ax.loglog(r_mid[mask], xi[mask], 'o-', label='Simulation', color='blue')
    
    from galaxy_statistics import GalaxyCorrelationFunction
    gcf = GalaxyCorrelationFunction()
    r_plot = np.logspace(-0.5, 1.7, 100)
    xi_pl = gcf.power_law(r_plot, r0=5.0, gamma=1.8)
    ax.loglog(r_plot, xi_pl, '--', label=r'$r_0=5h^{-1}$Mpc, $\gamma=1.8$', color='red', alpha=0.7)
    
    ax.set_xlabel(r'$r$ [h$^{-1}$ Mpc]')
    ax.set_ylabel(r'$\xi(r)$')
    ax.set_title('Galaxy Correlation Function')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved correlation function to {output_file}")

def plot_galaxy_bias(r_mid, bias, output_file='bias.png'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    mask = bias > 0
    ax.semilogx(r_mid[mask], bias[mask], 'o-', label='Simulation', color='blue')
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No bias')
    
    ax.set_xlabel(r'$r$ [h$^{-1}$ Mpc]')
    ax.set_ylabel(r'$b(r)$')
    ax.set_title('Galaxy Bias')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 3])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved galaxy bias to {output_file}")

def plot_hod(M_h, N_cen, N_sat, N_total, output_file='hod.png'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    ax.loglog(M_h, N_cen, '-', label=r'$\langle N_{cen} \rangle$', color='blue', linewidth=2)
    ax.loglog(M_h, N_sat, '-', label=r'$\langle N_{sat} \rangle$', color='green', linewidth=2)
    ax.loglog(M_h, N_total, '-', label=r'$\langle N_{tot} \rangle$', color='red', linewidth=2)
    
    ax.set_xlabel(r'$M_h$ [h$^{-1}$ M$_\odot$]')
    ax.set_ylabel(r'$\langle N \rangle$')
    ax.set_title('Halo Occupation Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved HOD plot to {output_file}")

def plot_galaxy_distribution(galaxies, box_size, output_file='galaxy_distribution.png'):
    pos = np.array([g['pos'] for g in galaxies])
    
    n_gal = len(pos)
    max_plot = min(n_gal, 50000)
    
    if n_gal > max_plot:
        idx = np.random.choice(n_gal, max_plot, replace=False)
        pos_plot = pos[idx]
    else:
        pos_plot = pos
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.scatter(pos_plot[:, 0], pos_plot[:, 1], s=0.5, alpha=0.6, c='darkblue')
    
    ax.set_xlabel('x [Mpc/h]')
    ax.set_ylabel('y [Mpc/h]')
    ax.set_title(f'Galaxy Distribution ({n_gal} galaxies)')
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved galaxy distribution to {output_file}")

def plot_stellar_mass_vs_halo_mass(galaxies, output_file='mstar_mhalo.png'):
    m_halo = np.array([g['halo_mass'] for g in galaxies])
    m_star = np.array([g['stellar_mass'] for g in galaxies])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    ax.loglog(m_halo, m_star, 'o', alpha=0.5, markersize=2)
    
    ax.set_xlabel(r'$M_h$ [h$^{-1}$ M$_\odot$]')
    ax.set_ylabel(r'$M_*$ [h$^{-1}$ M$_\odot$]')
    ax.set_title('Stellar Mass vs Halo Mass')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved M*-Mh relation to {output_file}")

def plot_sam_galaxy_properties(galaxies, output_file='sam_properties.png'):
    m_stars = np.array([g.m_stars for g in galaxies])
    m_gas_cold = np.array([g.m_gas_cold for g in galaxies])
    m_bh = np.array([g.m_bh for g in galaxies])
    sfr = np.array([g.sfr for g in galaxies])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    ax.hist(np.log10(m_stars[m_stars > 0]), bins=30, alpha=0.7, color='blue')
    ax.set_xlabel(r'$\log_{10}(M_* / M_\odot)$')
    ax.set_ylabel('Count')
    ax.set_title('Stellar Mass Distribution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(np.log10(m_gas_cold[m_gas_cold > 0]), bins=30, alpha=0.7, color='green')
    ax.set_xlabel(r'$\log_{10}(M_{gas,cold} / M_\odot)$')
    ax.set_ylabel('Count')
    ax.set_title('Cold Gas Mass Distribution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.hist(np.log10(m_bh[m_bh > 0]), bins=30, alpha=0.7, color='red')
    ax.set_xlabel(r'$\log_{10}(M_{BH} / M_\odot)$')
    ax.set_ylabel('Count')
    ax.set_title('Black Hole Mass Distribution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    mask = sfr > 0
    ax.hist(np.log10(sfr[mask]), bins=30, alpha=0.7, color='purple')
    ax.set_xlabel(r'$\log_{10}(SFR / M_\odot yr^{-1})$')
    ax.set_ylabel('Count')
    ax.set_title('Star Formation Rate Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved SAM galaxy properties to {output_file}")
