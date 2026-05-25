import numpy as np

class Cosmology:
    def __init__(self, Omega_m=0.3089, Omega_L=0.6911, Omega_b=0.0486, 
                 h=0.6774, ns=0.9667, sigma8=0.8159):
        self.Omega_m = Omega_m
        self.Omega_L = Omega_L
        self.Omega_b = Omega_b
        self.h = h
        self.ns = ns
        self.sigma8 = sigma8
        self.Omega_cdm = Omega_m - Omega_b
    
    def E(self, z):
        a = 1.0 / (1.0 + z)
        return np.sqrt(self.Omega_m * (1.0 + z)**3 + self.Omega_L)
    
    def H(self, z):
        return 100.0 * self.h * self.E(z)
    
    def D(self, z):
        a = 1.0 / (1.0 + z)
        Omega_m_z = self.Omega_m * (1.0 + z)**3 / self.E(z)**2
        Omega_L_z = self.Omega_L / self.E(z)**2
        
        g1 = 2.5 * Omega_m_z
        g2 = Omega_m_z**(4.0/7.0) - Omega_L_z + (1.0 + Omega_m_z/2.0) * (1.0 + Omega_L_z/70.0)
        return a * g1 / g2
    
    def f(self, z):
        Omega_m_z = self.Omega_m * (1.0 + z)**3 / self.E(z)**2
        return Omega_m_z**0.55
    
    def transfer_EH(self, k):
        h = self.h
        k_h = k / h
        
        Omega_b_h2 = self.Omega_b * h**2
        Omega_cdm_h2 = self.Omega_cdm * h**2
        Omega_m_h2 = self.Omega_m * h**2
        
        theta = 2.7255 / 2.7
        s = 44.5 * np.log(9.83 / Omega_m_h2) / np.sqrt(1.0 + 10.0 * Omega_b_h2**0.75)
        
        alpha_gamma = 1.0 - 0.328 * np.log(431.0 * Omega_m_h2) * self.Omega_b / self.Omega_m + \
                       0.38 * np.log(22.3 * Omega_m_h2) * (self.Omega_b / self.Omega_m)**2
        
        gamma_eff = self.Omega_m * h * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * s)**4))
        
        q = k * theta**2 / gamma_eff
        
        L = np.log(2.0 * np.e + 1.8 * q)
        C = 14.2 + 731.0 / (1.0 + 62.5 * q)
        T = L / (L + C * q**2)
        
        return T
    
    def power_spectrum(self, k, z=0.0):
        k0 = 0.05
        T = self.transfer_EH(k)
        P_prim = (k / k0)**(self.ns - 1.0)
        P_lin = k**self.ns * T**2
        P_lin = P_lin / P_lin[np.argmin(np.abs(k - k0))]
        
        sigma8_calc = self.sigma_R(8.0, k, P_lin)
        P_lin = P_lin * (self.sigma8 / sigma8_calc)**2
        
        D_z = self.D(z)
        D_0 = self.D(0.0)
        P_lin = P_lin * (D_z / D_0)**2
        
        return P_lin
    
    def sigma_R(self, R, k, Pk):
        W = 3.0 * (np.sin(k * R) - k * R * np.cos(k * R)) / (k * R)**3
        integrand = k**2 * Pk * W**2
        sigma2 = np.trapz(integrand, k) / (2.0 * np.pi**2)
        return np.sqrt(sigma2)
    
    def rho_m(self, z=0.0):
        rho_crit = 2.775e11 * self.h**2
        return rho_crit * self.Omega_m * (1.0 + z)**3
