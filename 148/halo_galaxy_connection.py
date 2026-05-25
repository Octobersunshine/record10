import numpy as np
from cosmology import Cosmology

class HODModel:
    def __init__(self, cosmo=None):
        self.cosmo = cosmo if cosmo is not None else Cosmology()
        
        self.logM_min = 12.0
        self.logM1 = 13.0
        self.sigma_logM = 0.2
        self.alpha = 1.0
        self.kappa = 0.5
        
        self.M_min = 10**self.logM_min
        self.M1 = 10**self.logM1
    
    def N_cen(self, M_h):
        M_h = np.atleast_1d(M_h)
        
        x = np.log10(M_h / self.M_min)
        N_c = 0.5 * (1.0 + np.tanh(x / self.sigma_logM))
        
        return N_c
    
    def N_sat(self, M_h):
        M_h = np.atleast_1d(M_h)
        
        M_cut = self.kappa * self.M_min
        mask = M_h > M_cut
        
        N_s = np.zeros_like(M_h)
        N_s[mask] = ((M_h[mask] - M_cut) / self.M1)**self.alpha
        
        return N_s
    
    def N_total(self, M_h):
        return self.N_cen(M_h) + self.N_sat(M_h)
    
    def assign_galaxies(self, halos):
        galaxies = []
        galaxy_id = 0
        
        for halo in halos:
            M_h = halo['mass']
            center = halo.get('center', np.zeros(3))
            r_vir = halo.get('radius', 0.1)
            
            N_c = np.random.poisson(self.N_cen(M_h)[0])
            
            for i in range(N_c):
                galaxy = {
                    'id': galaxy_id,
                    'halo_id': halo.get('id', 0),
                    'halo_mass': M_h,
                    'type': 'central',
                    'pos': center + np.random.normal(0, r_vir * 0.1, 3),
                    'stellar_mass': self.stellar_mass_from_halo_mass(M_h, 'central')
                }
                galaxies.append(galaxy)
                galaxy_id += 1
            
            N_s = np.random.poisson(self.N_sat(M_h)[0])
            
            for i in range(N_s):
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.arccos(np.random.uniform(-1, 1))
                r = np.random.uniform(0, r_vir)
                
                pos_sat = center + np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
                
                galaxy = {
                    'id': galaxy_id,
                    'halo_id': halo.get('id', 0),
                    'halo_mass': M_h,
                    'type': 'satellite',
                    'pos': pos_sat,
                    'stellar_mass': self.stellar_mass_from_halo_mass(M_h, 'satellite')
                }
                galaxies.append(galaxy)
                galaxy_id += 1
        
        return galaxies
    
    def stellar_mass_from_halo_mass(self, M_h, gal_type='central'):
        M_h = np.atleast_1d(M_h)
        
        logMh = np.log10(M_h)
        
        logM1_bins = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        logMs_bins = np.array([7.0, 8.5, 9.8, 10.6, 11.1, 11.3])
        
        logMs = np.interp(logMh, logM1_bins, logMs_bins)
        
        logMs += np.random.normal(0, 0.15, size=logMs.shape)
        
        return 10**logMs


class AbundanceMatching:
    def __init__(self, cosmo=None):
        self.cosmo = cosmo if cosmo is not None else Cosmology()
        
        self.sigma_scatter = 0.15
    
    def halo_mass_function(self, M_h, z=0.0):
        M_h = np.atleast_1d(M_h)
        
        sigma8 = self.cosmo.sigma8
        Omega_m = self.cosmo.Omega_m
        h = self.cosmo.h
        
        A = 0.0186 * (sigma8 / 0.8)**(-0.14)
        a = 1.47 * (sigma8 / 0.8)**(-0.06)
        b = 2.57 * (sigma8 / 0.8)**(0.06)
        c = 1.19
        
        rho_m = 2.775e11 * h**2 * Omega_m
        
        delta_c = 1.686
        sigma = self.cosmo.sigma_R(8.0, np.logspace(-3, 2, 1000), 
                                   self.cosmo.power_spectrum(np.logspace(-3, 2, 1000), z))
        sigma = sigma * (8.0 / (3.0 * M_h / (4.0 * np.pi * rho_m))**(1.0/3.0))
        
        f = A * np.sqrt(2.0 * a / np.pi) * (1.0 + (sigma**2 / (a * delta_c**2))**b) * \
            np.exp(-a * delta_c**2 / (2.0 * sigma**2))
        
        n = f * rho_m / M_h * dlogMhdlogM
        
        return n
    
    def stellar_mass_function(self, M_star, z=0.0):
        M_star = np.atleast_1d(M_star)
        logMs = np.log10(M_star)
        
        phi_star = 1e-2
        logMs_star = 10.7
        alpha = -1.25
        
        x = 10**(logMs - logMs_star)
        phi = np.log(10.0) * phi_star * x**(alpha + 1) * np.exp(-x)
        
        return phi
    
    def abundance_match(self, halos, z=0.0):
        halo_masses = np.array([h['mass'] for h in halos])
        
        sorted_idx = np.argsort(halo_masses)[::-1]
        sorted_masses = halo_masses[sorted_idx]
        
        n_halos = len(sorted_masses)
        cum_halos = np.arange(1, n_halos + 1)
        
        logMs = np.linspace(8, 12, 1000)
        Ms = 10**logMs
        phi = self.stellar_mass_function(Ms, z)
        
        cum_galaxies = np.trapz(phi[::-1], 10**logMs[::-1])[::-1]
        
        stellar_masses = np.interp(cum_halos, cum_galaxies, Ms)
        
        scatter = np.random.normal(0, self.sigma_scatter, n_halos)
        stellar_masses = stellar_masses * 10**scatter
        
        galaxies = []
        for i, idx in enumerate(sorted_idx):
            halo = halos[idx]
            galaxy = {
                'id': i,
                'halo_id': halo.get('id', 0),
                'halo_mass': halo['mass'],
                'type': 'central',
                'pos': halo.get('center', np.zeros(3)),
                'stellar_mass': stellar_masses[i]
            }
            galaxies.append(galaxy)
        
        return galaxies
