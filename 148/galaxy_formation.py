import numpy as np
from cosmology import Cosmology

class Galaxy:
    def __init__(self, halo_id=0, mass=0.0):
        self.halo_id = halo_id
        self.mass = mass
        
        self.m_halo = 0.0
        self.m_vir = 0.0
        self.r_vir = 0.0
        self.v_vir = 0.0
        
        self.m_gas_hot = 0.0
        self.m_gas_cold = 0.0
        self.m_stars = 0.0
        self.m_bh = 0.0
        
        self.m_metals_hot = 0.0
        self.m_metals_cold = 0.0
        self.m_metals_stars = 0.0
        
        self.sfr = 0.0
        self.sfr_100myr = 0.0
        
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        
        self.central = True
        self.z_formation = 0.0
        
        self.m_stars_bulge = 0.0
        self.m_stars_disk = 0.0

class SemiAnalyticModel:
    def __init__(self, cosmo=None):
        self.cosmo = cosmo if cosmo is not None else Cosmology()
        
        self.f_baryon = self.cosmo.Omega_b / self.cosmo.Omega_m
        
        self.t_dyn0 = 0.1
        self.f_star = 0.02
        self.epsilon_sn = 0.005
        self.v_wind = 350.0
        
        self.f_bh = 0.001
        self.epsilon_radio = 0.05
        
        self.f_cool = 1.0
        
        self.galaxies = []
        self.halos = []
    
    def initialize_halo(self, halo):
        m_vir = halo['mass']
        z = halo.get('z', 0.0)
        
        m_gas = m_vir * self.f_baryon
        m_stars = 0.0
        m_bh = 0.0
        
        galaxy = Galaxy()
        galaxy.halo_id = halo.get('id', 0)
        galaxy.m_halo = m_vir
        galaxy.m_vir = m_vir
        galaxy.r_vir = self.virial_radius(m_vir, z)
        galaxy.v_vir = self.virial_velocity(m_vir, z)
        
        galaxy.m_gas_hot = m_gas
        galaxy.m_gas_cold = 0.0
        galaxy.m_stars = m_stars
        galaxy.m_bh = m_bh
        
        galaxy.pos = halo.get('center', np.zeros(3))
        galaxy.central = True
        galaxy.z_formation = z
        
        return galaxy
    
    def virial_radius(self, m_vir, z):
        rho_crit = 2.775e11 * self.cosmo.h**2
        rho_vir = 200.0 * rho_crit * (1.0 + z)**3
        
        r_vir = (3.0 * m_vir / (4.0 * np.pi * rho_vir))**(1.0/3.0)
        return r_vir
    
    def virial_velocity(self, m_vir, z):
        r_vir = self.virial_radius(m_vir, z)
        G = 4.30091e-9
        v_vir = np.sqrt(G * m_vir / r_vir)
        return v_vir
    
    def dynamical_time(self, m_vir, z):
        r_vir = self.virial_radius(m_vir, z)
        v_vir = self.virial_velocity(m_vir, z)
        t_dyn = r_vir / v_vir
        return t_dyn
    
    def cooling_radius(self, m_vir, z):
        return 0.1 * self.virial_radius(m_vir, z)
    
    def cooling_time(self, m_vir, z):
        t_dyn = self.dynamical_time(m_vir, z)
        t_cool = 10.0 * t_dyn
        return t_cool
    
    def cool_gas(self, galaxy, dt, z):
        if galaxy.m_gas_hot <= 0:
            return 0.0
        
        t_cool = self.cooling_time(galaxy.m_vir, z)
        t_dyn = self.dynamical_time(galaxy.m_vir, z)
        
        m_cool_dt = galaxy.m_gas_hot * dt / t_cool
        m_cool_dt = min(m_cool_dt, galaxy.m_gas_hot)
        
        m_cool_dt *= self.f_cool
        
        galaxy.m_gas_hot -= m_cool_dt
        galaxy.m_gas_cold += m_cool_dt
        
        return m_cool_dt
    
    def form_stars(self, galaxy, dt, z):
        if galaxy.m_gas_cold <= 0:
            galaxy.sfr = 0.0
            return 0.0
        
        t_dyn = self.dynamical_time(galaxy.m_vir, z)
        
        sfr = self.f_star * galaxy.m_gas_cold / t_dyn
        m_star_dt = sfr * dt
        m_star_dt = min(m_star_dt, galaxy.m_gas_cold)
        
        galaxy.m_gas_cold -= m_star_dt
        galaxy.m_stars += m_star_dt
        galaxy.sfr = m_star_dt / dt
        
        return m_star_dt
    
    def supernovae_feedback(self, galaxy, dt, z):
        if galaxy.sfr <= 0:
            return 0.0, 0.0
        
        m_stars_formed = galaxy.sfr * dt
        
        E_sn = self.epsilon_sn * m_stars_formed * (1e5)**2
        
        m_heat = 0.0
        m_eject = 0.0
        
        if galaxy.v_vir < self.v_wind:
            f_eject = 0.5
            m_eject = f_eject * m_stars_formed
            m_heat = m_stars_formed - m_eject
        else:
            m_heat = m_stars_formed
        
        m_heat = min(m_heat, galaxy.m_gas_cold)
        galaxy.m_gas_cold -= m_heat
        galaxy.m_gas_hot += m_heat
        
        m_eject = min(m_eject, galaxy.m_gas_cold)
        galaxy.m_gas_cold -= m_eject
        
        return m_heat, m_eject
    
    def agn_feedback(self, galaxy, dt, z):
        if galaxy.m_bh <= 0 or galaxy.m_gas_hot <= 0:
            return 0.0
        
        L_eddington = 1.3e38 * galaxy.m_bh
        L_radio = self.epsilon_radio * L_eddington
        
        t_radio = 1e8
        m_heat = L_radio * t_radio / (galaxy.v_vir**2)
        m_heat = min(m_heat, galaxy.m_gas_cold)
        
        galaxy.m_gas_cold -= m_heat
        galaxy.m_gas_hot += m_heat
        
        return m_heat
    
    def black_hole_growth(self, galaxy, dt, z):
        if galaxy.m_gas_cold <= 0:
            return 0.0
        
        m_bh_dt = self.f_bh * galaxy.sfr * dt
        m_bh_dt = min(m_bh_dt, galaxy.m_gas_cold)
        
        galaxy.m_gas_cold -= m_bh_dt
        galaxy.m_bh += m_bh_dt
        
        return m_bh_dt
    
    def evolve_galaxy(self, galaxy, dt, z):
        self.cool_gas(galaxy, dt, z)
        
        m_star = self.form_stars(galaxy, dt, z)
        
        self.supernovae_feedback(galaxy, dt, z)
        
        self.black_hole_growth(galaxy, dt, z)
        
        self.agn_feedback(galaxy, dt, z)
        
        return m_star
    
    def populate_halos(self, halos, z=0.0):
        self.galaxies = []
        
        for halo in halos:
            if halo['mass'] < 1e10:
                continue
            
            galaxy = self.initialize_halo(halo)
            
            n_steps = 100
            z_list = np.linspace(z, 20.0, n_steps)
            a_list = 1.0 / (1.0 + z_list)
            
            for i in range(n_steps - 1):
                z1 = z_list[i]
                z2 = z_list[i + 1]
                a1 = 1.0 / (1.0 + z1)
                a2 = 1.0 / (1.0 + z2)
                
                H1 = self.cosmo.H(z1)
                H2 = self.cosmo.H(z2)
                H_avg = 0.5 * (H1 + H2)
                dt = (a2 - a1) / (a2 * H_avg)
                
                dt_gyr = dt * 977.8 / 1e9
                
                galaxy.m_vir = halo['mass'] * (1.0 + z2) / (1.0 + z)
                
                self.evolve_galaxy(galaxy, dt_gyr, z2)
            
            self.galaxies.append(galaxy)
        
        return self.galaxies
    
    def get_stellar_masses(self):
        return np.array([g.m_stars for g in self.galaxies])
    
    def get_galaxy_positions(self):
        return np.array([g.pos for g in self.galaxies])
