import numpy as np
from pm_solver import PMSolver

class P3MSolver:
    def __init__(self, ngrid, box_size, cosmo, r_switch=None, r_soft=None):
        self.ngrid = ngrid
        self.box_size = box_size
        self.cell_size = box_size / ngrid
        self.cosmo = cosmo
        
        if r_switch is None:
            self.r_switch = 3.0 * self.cell_size
        else:
            self.r_switch = r_switch
            
        if r_soft is None:
            self.r_soft = 0.5 * self.cell_size
        else:
            self.r_soft = r_soft
        
        self.kf = 2.0 * np.pi / box_size
        self.setup_k_vectors()
        self.setup_screening_function()
        
        self.pm = PMSolver(ngrid, box_size, cosmo)
        
        self.build_domain_decomposition()
    
    def setup_k_vectors(self):
        ng = self.ngrid
        kx = np.fft.fftfreq(ng, d=1.0/ng) * self.kf
        ky = np.fft.fftfreq(ng, d=1.0/ng) * self.kf
        kz = np.fft.fftfreq(ng, d=1.0/ng) * self.kf
        
        self.kx, self.ky, self.kz = np.meshgrid(kx, ky, kz, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k = np.sqrt(self.k2)
        self.k2[0, 0, 0] = 1.0
    
    def setup_screening_function(self):
        r_s = self.r_switch
        k = self.k
        
        self.Wk = np.exp(-(k * r_s)**2 / 2.0)
        self.Wk[0, 0, 0] = 0.0
    
    def W_short(self, r):
        r_s = self.r_switch
        r_soft = self.r_soft
        
        r_eff = np.sqrt(r**2 + r_soft**2)
        
        W_long = 1.0 - np.exp(-(r_eff / r_s)**2 / 2.0)
        W_short = 1.0 - W_long
        
        return W_short
    
    def build_domain_decomposition(self):
        pass
    
    def cic_deposit(self, pos, mass=1.0):
        return self.pm.cic_deposit(pos, mass)
    
    def compute_long_range_potential(self, delta, a):
        delta_k = np.fft.fftn(delta)
        
        delta_k_smoothed = delta_k * self.Wk
        
        factor = 3.0 * self.cosmo.Omega_m * (100.0 * self.cosmo.h)**2 / 2.0
        phi_k = -factor * delta_k_smoothed / (self.k2 * a)
        
        phi = np.fft.ifftn(phi_k).real
        
        return phi
    
    def compute_long_range_forces(self, phi):
        phi_k = np.fft.fftn(phi)
        
        ax_k = -1j * self.kx * phi_k
        ay_k = -1j * self.ky * phi_k
        az_k = -1j * self.kz * phi_k
        
        ax = np.fft.ifftn(ax_k).real
        ay = np.fft.ifftn(ay_k).real
        az = np.fft.ifftn(az_k).real
        
        return ax, ay, az
    
    def get_long_range_accelerations(self, pos, a):
        delta = self.cic_deposit(pos)
        phi = self.compute_long_range_potential(delta, a)
        ax, ay, az = self.compute_long_range_forces(phi)
        
        npart = pos.shape[0]
        acc = np.zeros((npart, 3), dtype=np.float64)
        
        acc[:, 0] = self.pm.cic_interpolate(pos, ax)
        acc[:, 1] = self.pm.cic_interpolate(pos, ay)
        acc[:, 2] = self.pm.cic_interpolate(pos, az)
        
        return acc, delta
    
    def compute_short_range_accelerations(self, pos, a, masses=None):
        npart = pos.shape[0]
        acc_short = np.zeros((npart, 3), dtype=np.float64)
        
        if masses is None:
            masses = np.ones(npart)
        
        bs = self.box_size
        r_cut = self.r_switch * 2.5
        
        G = 1.0
        factor = G * 3.0 * self.cosmo.Omega_m * (100.0 * self.cosmo.h)**2 / (2.0 * a)
        
        cell_size = r_cut
        n_cells = int(np.ceil(bs / cell_size))
        cell_size = bs / n_cells
        
        cell_dict = {}
        for i in range(npart):
            cx = int(pos[i, 0] / cell_size) % n_cells
            cy = int(pos[i, 1] / cell_size) % n_cells
            cz = int(pos[i, 2] / cell_size) % n_cells
            key = (cx, cy, cz)
            if key not in cell_dict:
                cell_dict[key] = []
            cell_dict[key].append(i)
        
        for i in range(npart):
            xi, yi, zi = pos[i]
            cx = int(xi / cell_size) % n_cells
            cy = int(yi / cell_size) % n_cells
            cz = int(zi / cell_size) % n_cells
            
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        key = ((cx + dx) % n_cells, 
                               (cy + dy) % n_cells, 
                               (cz + dz) % n_cells)
                        if key in cell_dict:
                            neighbors.extend(cell_dict[key])
            
            for j in neighbors:
                if j <= i:
                    continue
                
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                dz = zi - pos[j, 2]
                
                dx = dx - bs * np.round(dx / bs)
                dy = dy - bs * np.round(dy / bs)
                dz = dz - bs * np.round(dz / bs)
                
                r2 = dx*dx + dy*dy + dz*dz
                
                if r2 > r_cut * r_cut or r2 < 1e-10:
                    continue
                
                r = np.sqrt(r2)
                
                W_s = self.W_short(r)
                
                r_soft = self.r_soft
                r_eff = np.sqrt(r2 + r_soft * r_soft)
                r3 = r_eff * r_eff * r_eff
                
                f_mag = factor * W_s / r3
                
                acc_short[i, 0] += f_mag * dx * masses[j]
                acc_short[i, 1] += f_mag * dy * masses[j]
                acc_short[i, 2] += f_mag * dz * masses[j]
                
                acc_short[j, 0] -= f_mag * dx * masses[i]
                acc_short[j, 1] -= f_mag * dy * masses[i]
                acc_short[j, 2] -= f_mag * dz * masses[i]
        
        return acc_short
    
    def get_accelerations(self, pos, a, masses=None):
        acc_long, delta = self.get_long_range_accelerations(pos, a)
        
        acc_short = self.compute_short_range_accelerations(pos, a, masses)
        
        acc = acc_long + acc_short
        
        return acc, delta, acc_long, acc_short
