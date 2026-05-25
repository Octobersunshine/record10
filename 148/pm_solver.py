import numpy as np

class PMSolver:
    def __init__(self, ngrid, box_size, cosmo):
        self.ngrid = ngrid
        self.box_size = box_size
        self.cell_size = box_size / ngrid
        self.cosmo = cosmo
        
        self.kf = 2.0 * np.pi / box_size
        self.setup_k_vectors()
        
    def setup_k_vectors(self):
        ng = self.ngrid
        kx = np.fft.fftfreq(ng, d=1.0/ng) * self.kf
        ky = np.fft.fftfreq(ng, d=1.0/ng) * self.kf
        kz = np.fft.fftfreq(ng, d=1.0/ng) * self.kf
        
        self.kx, self.ky, self.kz = np.meshgrid(kx, ky, kz, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0, 0, 0] = 1.0
    
    def cic_deposit(self, pos, mass=1.0):
        ng = self.ngrid
        bs = self.box_size
        cs = self.cell_size
        
        rho = np.zeros((ng, ng, ng), dtype=np.float64)
        
        pos = np.mod(pos, bs)
        idx = (pos / cs).astype(np.int32)
        frac = (pos / cs) - idx
        
        idx1 = np.mod(idx, ng)
        idx2 = np.mod(idx + 1, ng)
        
        npart = pos.shape[0]
        
        for i in range(npart):
            i1, j1, k1 = idx1[i]
            i2, j2, k2 = idx2[i]
            fx, fy, fz = frac[i]
            
            w000 = (1 - fx) * (1 - fy) * (1 - fz)
            w100 = fx * (1 - fy) * (1 - fz)
            w010 = (1 - fx) * fy * (1 - fz)
            w001 = (1 - fx) * (1 - fy) * fz
            w110 = fx * fy * (1 - fz)
            w101 = fx * (1 - fy) * fz
            w011 = (1 - fx) * fy * fz
            w111 = fx * fy * fz
            
            rho[i1, j1, k1] += w000
            rho[i2, j1, k1] += w100
            rho[i1, j2, k1] += w010
            rho[i1, j1, k2] += w001
            rho[i2, j2, k1] += w110
            rho[i2, j1, k2] += w101
            rho[i1, j2, k2] += w011
            rho[i2, j2, k2] += w111
        
        rho_mean = np.mean(rho)
        if rho_mean > 0:
            rho = rho / rho_mean - 1.0
        
        return rho
    
    def cic_interpolate(self, pos, field):
        ng = self.ngrid
        bs = self.box_size
        cs = self.cell_size
        
        pos = np.mod(pos, bs)
        idx = (pos / cs).astype(np.int32)
        frac = (pos / cs) - idx
        
        idx1 = np.mod(idx, ng)
        idx2 = np.mod(idx + 1, ng)
        
        npart = pos.shape[0]
        result = np.zeros(npart, dtype=np.float64)
        
        for i in range(npart):
            i1, j1, k1 = idx1[i]
            i2, j2, k2 = idx2[i]
            fx, fy, fz = frac[i]
            
            w000 = (1 - fx) * (1 - fy) * (1 - fz)
            w100 = fx * (1 - fy) * (1 - fz)
            w010 = (1 - fx) * fy * (1 - fz)
            w001 = (1 - fx) * (1 - fy) * fz
            w110 = fx * fy * (1 - fz)
            w101 = fx * (1 - fy) * fz
            w011 = (1 - fx) * fy * fz
            w111 = fx * fy * fz
            
            result[i] = w000 * field[i1, j1, k1] + \
                        w100 * field[i2, j1, k1] + \
                        w010 * field[i1, j2, k1] + \
                        w001 * field[i1, j1, k2] + \
                        w110 * field[i2, j2, k1] + \
                        w101 * field[i2, j1, k2] + \
                        w011 * field[i1, j2, k2] + \
                        w111 * field[i2, j2, k2]
        
        return result
    
    def compute_potential(self, delta, a):
        delta_k = np.fft.fftn(delta)
        
        rho_crit = 2.775e11 * self.cosmo.h**2
        factor = 3.0 * self.cosmo.Omega_m * (100.0 * self.cosmo.h)**2 / 2.0
        phi_k = -factor * delta_k / (self.k2 * a)
        
        phi = np.fft.ifftn(phi_k).real
        
        return phi
    
    def compute_forces(self, phi):
        phi_k = np.fft.fftn(phi)
        
        ax_k = -1j * self.kx * phi_k
        ay_k = -1j * self.ky * phi_k
        az_k = -1j * self.kz * phi_k
        
        ax = np.fft.ifftn(ax_k).real
        ay = np.fft.ifftn(ay_k).real
        az = np.fft.ifftn(az_k).real
        
        return ax, ay, az
    
    def get_accelerations(self, pos, a):
        delta = self.cic_deposit(pos)
        phi = self.compute_potential(delta, a)
        ax, ay, az = self.compute_forces(phi)
        
        npart = pos.shape[0]
        acc = np.zeros((npart, 3), dtype=np.float64)
        
        acc[:, 0] = self.cic_interpolate(pos, ax)
        acc[:, 1] = self.cic_interpolate(pos, ay)
        acc[:, 2] = self.cic_interpolate(pos, az)
        
        return acc, delta
